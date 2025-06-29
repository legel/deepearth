#!/usr/bin/env python3
"""
DeepEarth Embedding Preparation Script

Converts vision embeddings from HuggingFace Parquet format to high-performance
memory-mapped binary format for the DeepEarth dashboard.

This script:
1. Downloads the complete dataset from HuggingFace (if needed) using snapshot_download
2. Converts 159 Parquet files to a single memory-mapped file
3. Creates SQLite index for fast lookups
4. Validates the conversion

Expected runtime: ~50 minutes for 7,949 embeddings (~206GB)

Author: DeepEarth Project
License: MIT
"""

import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import time
import logging
import sqlite3
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import argparse
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingConverter:
    """Convert DeepEarth embeddings to memory-mapped binary format"""
    
    def __init__(self, dataset_dir: str, output_dir: str = "."):
        """
        Initialize converter.
        
        Args:
            dataset_dir: Path to HuggingFace dataset directory
            output_dir: Where to save output files
        """
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        
        # Verify dataset directory exists
        if not self.dataset_dir.exists():
            raise ValueError(f"Dataset directory not found: {self.dataset_dir}")
        
        # Load dataset configuration
        self.config_path = self.dataset_dir / "dataset_config.json"
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Use default configuration
            self.config = {
                'data_paths': {
                    'observations': 'observations.parquet',
                    'vision_embeddings_dir': 'vision_embeddings',
                    'vision_index': 'vision_index.parquet'
                }
            }
        
        # Output files
        self.embeddings_file = self.output_dir / "embeddings.mmap"
        self.index_db = self.output_dir / "embeddings_index.db"
        
        # Embedding specifications (V-JEPA-2)
        self.embedding_size = 6488064  # 8 × 24 × 24 × 1408
        self.embedding_dtype = np.float32
        self.bytes_per_embedding = self.embedding_size * 4  # 4 bytes per float32
        
        # Statistics
        self.stats = {
            'embeddings_processed': 0,
            'total_size_gb': 0,
            'processing_time': 0,
            'avg_retrieval_time_ms': 0
        }
        
    def create_index_database(self):
        """Create SQLite database for fast GBIF ID -> file offset mapping"""
        logger.info("Creating SQLite index database...")
        
        # Remove existing database
        if self.index_db.exists():
            self.index_db.unlink()
        
        conn = sqlite3.connect(self.index_db)
        cursor = conn.cursor()
        
        # Create index table with all metadata for fast filtering
        cursor.execute('''
            CREATE TABLE embedding_index (
                gbif_id INTEGER PRIMARY KEY,
                file_offset INTEGER NOT NULL,
                taxon_id TEXT,
                taxon_name TEXT,
                latitude REAL,
                longitude REAL,
                year INTEGER,
                month INTEGER,
                day INTEGER,
                hour INTEGER,
                has_vision BOOLEAN,
                split TEXT,
                file_idx INTEGER
            )
        ''')
        
        # Create indexes for common queries
        cursor.execute('CREATE INDEX idx_spatial ON embedding_index(latitude, longitude)')
        cursor.execute('CREATE INDEX idx_temporal ON embedding_index(year, month)')
        cursor.execute('CREATE INDEX idx_taxon ON embedding_index(taxon_name)')
        cursor.execute('CREATE INDEX idx_vision ON embedding_index(has_vision)')
        
        conn.commit()
        conn.close()
        
        logger.info(f"Created index database: {self.index_db}")
    
    def load_observations(self) -> pd.DataFrame:
        """Load observation metadata"""
        logger.info("Loading observations metadata...")
        obs_path = self.dataset_dir / self.config['data_paths']['observations']
        
        if not obs_path.exists():
            raise FileNotFoundError(f"Observations file not found: {obs_path}")
        
        return pd.read_parquet(obs_path)
    
    def load_vision_index(self) -> pd.DataFrame:
        """Load vision embeddings index"""
        logger.info("Loading vision embeddings index...")
        index_path = self.dataset_dir / self.config['data_paths']['vision_index']
        
        if not index_path.exists():
            raise FileNotFoundError(f"Vision index not found: {index_path}")
        
        return pd.read_parquet(index_path)
    
    def convert_embeddings(self):
        """Convert all embeddings to memory-mapped binary format"""
        logger.info("Converting embeddings to memory-mapped format...")
        
        # Load indices and metadata
        vision_index = self.load_vision_index()
        observations = self.load_observations()
        
        # Create GBIF ID to observation mapping
        obs_dict = observations.set_index('gbif_id').to_dict('index')
        
        # Create SQLite database
        self.create_index_database()
        
        # Remove existing binary file
        if self.embeddings_file.exists():
            self.embeddings_file.unlink()
            logger.info("Removed existing embeddings file")
        
        total_embeddings = len(vision_index)
        total_bytes = total_embeddings * self.bytes_per_embedding
        
        logger.info(f"Target size: {total_bytes / (1024**3):.1f} GB for {total_embeddings} embeddings")
        logger.info("Using incremental file writing to avoid memory issues")
        
        # Open binary file for incremental writing
        binary_file = open(self.embeddings_file, 'wb')
        
        # Connect to database for index updates
        conn = sqlite3.connect(self.index_db)
        cursor = conn.cursor()
        
        # Group by file for efficient loading
        file_groups = vision_index.groupby('file_idx')
        
        current_byte_offset = 0
        processed_count = 0
        start_time = time.time()
        
        print(f"\n🎯 Converting {total_embeddings} embeddings from {len(file_groups)} files")
        print(f"📁 Output file: {self.embeddings_file}")
        print(f"📊 Index database: {self.index_db}")
        print("=" * 80)
        
        try:
            for file_idx, file_data in tqdm(file_groups, desc="Processing files"):
                vision_dir = self.dataset_dir / self.config['data_paths']['vision_embeddings_dir']
                file_path = vision_dir / f"embeddings_{file_idx:06d}.parquet"
                
                if not file_path.exists():
                    logger.warning(f"File not found: {file_path}")
                    continue
                
                try:
                    # Load embeddings from Parquet file
                    embeddings_df = pd.read_parquet(file_path)
                    
                    # Process embeddings in this file
                    batch_inserts = []
                    
                    for _, row in file_data.iterrows():
                        gbif_id = row['gbif_id']
                        
                        # Get embedding from the file
                        embedding_data = embeddings_df[embeddings_df['gbif_id'] == gbif_id]
                        if len(embedding_data) == 0:
                            logger.warning(f"No embedding found for GBIF {gbif_id}")
                            continue
                        
                        embedding = embedding_data.iloc[0]['embedding']
                        
                        # Convert to numpy array
                        if hasattr(embedding, 'values'):
                            embedding = embedding.values
                        embedding = np.array(embedding, dtype=self.embedding_dtype).flatten()
                        
                        if embedding.shape[0] != self.embedding_size:
                            logger.warning(f"Unexpected embedding size for GBIF {gbif_id}: {embedding.shape}")
                            continue
                        
                        # Write embedding directly to binary file
                        embedding_bytes = embedding.tobytes()
                        binary_file.write(embedding_bytes)
                        binary_file.flush()  # Ensure it's written to disk
                        
                        # Get observation metadata
                        obs_metadata = obs_dict.get(gbif_id, {})
                        
                        # Prepare database insert with current byte offset
                        batch_inserts.append((
                            int(gbif_id),
                            current_byte_offset,  # Byte offset in file
                            obs_metadata.get('taxon_id', ''),
                            obs_metadata.get('taxon_name', ''),
                            float(obs_metadata.get('latitude', 0.0)),
                            float(obs_metadata.get('longitude', 0.0)),
                            int(obs_metadata.get('year', 0)),
                            int(obs_metadata.get('month', 0)) if pd.notna(obs_metadata.get('month')) else 0,
                            int(obs_metadata.get('day', 0)) if pd.notna(obs_metadata.get('day')) else 0,
                            int(obs_metadata.get('hour', 0)) if pd.notna(obs_metadata.get('hour')) else 0,
                            bool(obs_metadata.get('has_vision', False)),
                            obs_metadata.get('split', ''),
                            int(file_idx)
                        ))
                        
                        current_byte_offset += self.bytes_per_embedding
                        processed_count += 1
                    
                    # Batch insert to database with duplicate handling
                    if batch_inserts:
                        try:
                            cursor.executemany('''
                                INSERT OR REPLACE INTO embedding_index 
                                (gbif_id, file_offset, taxon_id, taxon_name, latitude, longitude, 
                                 year, month, day, hour, has_vision, split, file_idx)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', batch_inserts)
                            conn.commit()
                        except sqlite3.Error as e:
                            logger.error(f"Database error: {e}")
                            # Continue processing even if some inserts fail
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    continue
        
        finally:
            # Clean up
            binary_file.close()
            conn.close()
        
        # Update statistics
        self.stats['embeddings_processed'] = processed_count
        self.stats['processing_time'] = time.time() - start_time
        self.stats['total_size_gb'] = (processed_count * self.bytes_per_embedding) / (1024**3)
        
        logger.info(f"Converted {processed_count} embeddings to memory-mapped format")
        logger.info(f"Processing took {self.stats['processing_time']:.2f} seconds")
        logger.info(f"File size: {self.stats['total_size_gb']:.1f} GB")
        logger.info(f"Average rate: {processed_count/self.stats['processing_time']:.1f} embeddings/sec")
    
    def verify_conversion(self, sample_size: int = 20):
        """Verify the conversion by testing random retrievals"""
        logger.info(f"Verifying conversion with {sample_size} samples...")
        
        if not self.embeddings_file.exists() or not self.index_db.exists():
            logger.error("Conversion files not found!")
            return []
        
        # Connect to database
        conn = sqlite3.connect(self.index_db)
        cursor = conn.cursor()
        
        # Get total number of embeddings
        cursor.execute("SELECT COUNT(*) as count FROM embedding_index")
        total_embeddings = cursor.fetchone()[0]
        logger.info(f"Total embeddings in index: {total_embeddings}")
        
        # Get random sample of GBIF IDs
        cursor.execute(f"SELECT gbif_id, file_offset FROM embedding_index ORDER BY RANDOM() LIMIT {sample_size}")
        sample_data = cursor.fetchall()
        
        # Open binary file for reading
        binary_file = open(self.embeddings_file, 'rb')
        
        retrieval_times = []
        
        for gbif_id, file_offset in sample_data:
            start_time = time.time()
            
            try:
                # Seek to position and read embedding
                binary_file.seek(file_offset)
                embedding_bytes = binary_file.read(self.bytes_per_embedding)
                
                if len(embedding_bytes) != self.bytes_per_embedding:
                    logger.error(f"Incomplete read for GBIF {gbif_id}")
                    continue
                
                # Convert bytes back to numpy array
                embedding = np.frombuffer(embedding_bytes, dtype=self.embedding_dtype)
                
                retrieval_time = (time.time() - start_time) * 1000  # Convert to ms
                retrieval_times.append(retrieval_time)
                
                logger.debug(f"Retrieved GBIF {gbif_id}: {embedding.shape} in {retrieval_time:.3f}ms")
                
            except Exception as e:
                logger.error(f"Error retrieving GBIF {gbif_id}: {e}")
        
        binary_file.close()
        conn.close()
        
        # Calculate statistics
        if retrieval_times:
            avg_time = np.mean(retrieval_times)
            min_time = np.min(retrieval_times)
            max_time = np.max(retrieval_times)
            p95_time = np.percentile(retrieval_times, 95)
            
            self.stats['avg_retrieval_time_ms'] = avg_time
            
            logger.info(f"✅ Verification complete!")
            logger.info(f"Average retrieval time: {avg_time:.3f}ms")
            logger.info(f"Min retrieval time: {min_time:.3f}ms")
            logger.info(f"Max retrieval time: {max_time:.3f}ms")
            logger.info(f"P95 retrieval time: {p95_time:.3f}ms")
            
            if avg_time < 10.0:
                logger.info("✅ Target <10ms retrieval time achieved!")
            elif avg_time < 100.0:
                logger.info("⚠️ Retrieval time is acceptable but not optimal")
            else:
                logger.warning(f"❌ Retrieval time {avg_time:.3f}ms exceeds 100ms target")
        
        return retrieval_times
    
    def print_summary(self):
        """Print conversion summary"""
        print("\n" + "="*80)
        print("🌍 DeepEarth Memory-Mapped Conversion Summary")
        print("="*80)
        print(f"Dataset directory: {self.dataset_dir}")
        print(f"Embeddings processed: {self.stats['embeddings_processed']:,}")
        print(f"Processing time: {self.stats['processing_time']/60:.1f} minutes")
        print(f"File size: {self.stats['total_size_gb']:.1f} GB")
        print(f"Average processing rate: {self.stats['embeddings_processed']/self.stats['processing_time']:.1f} embeddings/sec")
        print(f"Average retrieval time: {self.stats['avg_retrieval_time_ms']:.3f}ms")
        print(f"\nOutput files:")
        print(f"  Memory-mapped file: {self.embeddings_file}")
        print(f"  SQLite index: {self.index_db}")
        print("\n✅ Ready for high-performance embedding retrieval!")
        print("="*80)


def download_dataset(dataset_name: str = "deepearth/central-florida-native-plants", 
                    output_dir: str = "huggingface_dataset"):
    """
    Download dataset from HuggingFace using snapshot_download to get ALL files.
    
    This downloads the complete dataset including:
    - observations.parquet
    - vision_embeddings/ directory with all embedding files
    - vision_index.parquet
    - dataset_info.json
    - Any other files in the repository
    
    Args:
        dataset_name: HuggingFace dataset identifier
        output_dir: Where to save the dataset
    """
    try:
        from huggingface_hub import snapshot_download
        
        print(f"📥 Downloading {dataset_name} from HuggingFace...")
        print("This will download ALL files including vision embeddings...")
        print("This may take a while for large datasets...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Download complete snapshot of the dataset repository
        snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",
            local_dir=str(output_path),
            resume_download=True  # Allow resuming interrupted downloads
        )
        
        print(f"✅ Dataset saved to {output_path}")
        
        # Verify critical files exist
        critical_files = [
            "observations.parquet",
            "vision_index.parquet",
            "vision_embeddings"
        ]
        
        missing_files = []
        for file_name in critical_files:
            file_path = output_path / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        if missing_files:
            print(f"⚠️ Warning: Some expected files are missing: {missing_files}")
            print("The dataset may be incomplete or have a different structure.")
        else:
            print("✅ All critical files downloaded successfully!")
            
            # Count vision embedding files
            vision_dir = output_path / "vision_embeddings"
            if vision_dir.exists() and vision_dir.is_dir():
                embedding_files = list(vision_dir.glob("embeddings_*.parquet"))
                print(f"📊 Found {len(embedding_files)} vision embedding files")
        
        return True
    except ImportError:
        print("❌ Please install huggingface-hub: pip install huggingface-hub")
        return False
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main conversion function"""
    parser = argparse.ArgumentParser(
        description="Convert DeepEarth embeddings to memory-mapped format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert from existing dataset directory
  python prepare_embeddings.py /path/to/huggingface_dataset

  # Download from HuggingFace first
  python prepare_embeddings.py --download deepearth/central-florida-native-plants

  # Specify output directory
  python prepare_embeddings.py /path/to/dataset --output-dir /path/to/output
        """
    )
    
    parser.add_argument('dataset_dir', nargs='?', 
                       help='Path to HuggingFace dataset directory')
    parser.add_argument('--download', type=str,
                       help='Download complete dataset from HuggingFace first (uses snapshot_download)')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for mmap files (default: current directory)')
    parser.add_argument('--verify-samples', type=int, default=50,
                       help='Number of samples for verification (default: 50)')
    
    args = parser.parse_args()
    
    # Handle dataset download if requested
    if args.download:
        dataset_dir = "huggingface_dataset"
        if not download_dataset(args.download, dataset_dir):
            return 1
    elif args.dataset_dir:
        dataset_dir = args.dataset_dir
    else:
        print("❌ Please specify dataset directory or use --download flag")
        parser.print_help()
        return 1
    
    # Verify dataset directory exists
    if not Path(dataset_dir).exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run conversion
    converter = EmbeddingConverter(dataset_dir, output_dir)
    
    try:
        # Convert embeddings to memory-mapped format
        converter.convert_embeddings()
        
        # Verify conversion
        converter.verify_conversion(sample_size=args.verify_samples)
        
        # Print summary
        converter.print_summary()
        
        return 0
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())