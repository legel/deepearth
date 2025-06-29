#!/usr/bin/env python3
"""
Hugging Face Dataset Loader for DeepEarth Multimodal Explorer

This module provides efficient loading and caching of observation data from the 
new Hugging Face dataset structure (v0.2.0), including vision embeddings, 
language embeddings, and observation metadata.
"""

import os
import json
import ast
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Set, Any
import logging
from functools import lru_cache
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HuggingFaceDataLoader:
    """Efficient loader for DeepEarth datasets with configurable structure."""
    
    def __init__(self, config_path: str = "dataset_config.json"):
        """
        Initialize the data loader.
        
        Args:
            config_path: Path to the dataset configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.base_dir = Path(self.config['data_paths']['base_dir']).resolve()
        
        # Data storage
        self.observations = None
        self.vision_index = None
        self.vision_metadata = None
        self.dataset_info = None
        
        # Caches
        self.vision_embedding_cache = {}
        self.vision_file_cache = {}
        self.embedding_access_count = {}
        
        # Indices for fast lookup
        self.gbif_to_idx = {}
        self.taxon_to_gbifs = {}
        self.vision_gbif_to_file_info = {}
        
        # Load data
        self._load_data()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load dataset configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config = json.load(f)
            
        logger.info(f"Loaded configuration for: {config['dataset_name']}")
        return config
        
    def _load_data(self):
        """Load all data with progress tracking."""
        logger.info(f"Loading {self.config['dataset_name']}...")
        
        # Load observations
        self._load_observations()
        
        # Load vision index and metadata
        self._load_vision_data()
        
        # Load dataset info
        self._load_dataset_info()
        
        # Create indices for fast lookup
        self._create_indices()
        
        logger.info("Dataset loaded successfully!")
        self._print_summary()
        
    def _load_observations(self):
        """Load observation data from Parquet."""
        logger.info("Loading observations...")
        
        obs_path = self.base_dir / self.config['data_paths']['observations']
        if not obs_path.exists():
            raise FileNotFoundError(f"Observations file not found: {obs_path}")
            
        self.observations = pd.read_parquet(obs_path)
        
        # Standardize data types
        self.observations['gbif_id'] = self.observations['gbif_id'].astype(np.int64)
        self.observations['taxon_id'] = self.observations['taxon_id'].astype(str)
        
        # Parse image URLs if they're strings or numpy arrays
        if 'image_urls' in self.observations.columns:
            def parse_urls(url_data):
                if isinstance(url_data, np.ndarray):
                    # Handle numpy array case
                    if len(url_data) > 0:
                        url_str = url_data[0]  # Get the string from array
                        if isinstance(url_str, str):
                            # Split by semicolon if multiple URLs
                            return url_str.split(';') if ';' in url_str else [url_str]
                    return []
                elif isinstance(url_data, str):
                    try:
                        # Try to parse as literal (for list strings)
                        return ast.literal_eval(url_data)
                    except:
                        # Split by semicolon if multiple URLs
                        return url_data.split(';') if ';' in url_data else [url_data]
                elif isinstance(url_data, list):
                    return url_data
                else:
                    return []
            
            self.observations['image_urls'] = self.observations['image_urls'].apply(parse_urls)
            
        logger.info(f"Loaded {len(self.observations)} observations")
        
    def _load_vision_data(self):
        """Load vision embeddings index and metadata."""
        logger.info("Loading vision data...")
        
        # Load vision index
        vision_index_path = self.base_dir / self.config['data_paths']['vision_index']
        if vision_index_path.exists():
            self.vision_index = pd.read_parquet(vision_index_path)
            self.vision_index['gbif_id'] = self.vision_index['gbif_id'].astype(np.int64)
            logger.info(f"Loaded vision index with {len(self.vision_index)} entries")
        else:
            logger.warning("Vision index not found")
            
        # Load vision metadata
        vision_meta_path = self.base_dir / self.config['data_paths']['vision_metadata']
        if vision_meta_path.exists():
            with open(vision_meta_path, 'r') as f:
                self.vision_metadata = json.load(f)
            logger.info("Loaded vision metadata")
        else:
            logger.warning("Vision metadata not found")
            
    def _load_dataset_info(self):
        """Load dataset info."""
        info_path = self.base_dir / self.config['data_paths']['dataset_info']
        if info_path.exists():
            with open(info_path, 'r') as f:
                self.dataset_info = json.load(f)
            logger.info("Loaded dataset info")
        else:
            logger.warning("Dataset info not found")
            
    def _create_indices(self):
        """Create indices for fast lookup."""
        logger.info("Creating indices...")
        
        # GBIF ID to row index
        self.gbif_to_idx = {
            gbif_id: idx for idx, gbif_id in enumerate(self.observations['gbif_id'])
        }
        
        # Taxon ID to GBIF IDs
        self.taxon_to_gbifs = {}
        for _, row in self.observations.iterrows():
            taxon_id = row['taxon_id']
            if taxon_id not in self.taxon_to_gbifs:
                self.taxon_to_gbifs[taxon_id] = []
            self.taxon_to_gbifs[taxon_id].append(row['gbif_id'])
            
        # Vision GBIF ID to file info
        if self.vision_index is not None:
            self.vision_gbif_to_file_info = {}
            for _, row in self.vision_index.iterrows():
                gbif_id = row['gbif_id']
                self.vision_gbif_to_file_info[gbif_id] = {
                    'file_idx': row['file_idx'],
                    'row_idx': row.get('row_idx', 0)
                }
                
    def _print_summary(self):
        """Print dataset summary."""
        print("\n" + "="*80)
        print(f"Dataset: {self.config['dataset_name']}")
        print("="*80)
        print(f"Version: {self.config['dataset_version']}")
        print(f"Total observations: {len(self.observations):,}")
        print(f"Unique species: {len(self.taxon_to_gbifs):,}")
        
        if self.vision_index is not None:
            print(f"Vision embeddings: {len(self.vision_index):,}")
            
        # Data splits
        if 'split' in self.observations.columns:
            print("\nData splits:")
            for split, count in self.observations['split'].value_counts().items():
                print(f"  {split}: {count:,} observations")
                
        # Temporal range
        if 'year' in self.observations.columns:
            print(f"\nTemporal range: {int(self.observations['year'].min())}-{int(self.observations['year'].max())}")
            
        # Geographic bounds
        if 'latitude' in self.observations.columns and 'longitude' in self.observations.columns:
            print(f"\nGeographic bounds:")
            print(f"  Latitude: {self.observations['latitude'].min():.4f}° to {self.observations['latitude'].max():.4f}°")
            print(f"  Longitude: {self.observations['longitude'].min():.4f}° to {self.observations['longitude'].max():.4f}°")
            
        print("="*80 + "\n")
        
    def get_observation(self, gbif_id: int) -> Optional[pd.Series]:
        """Get observation by GBIF ID."""
        gbif_id = int(gbif_id)
        if gbif_id in self.gbif_to_idx:
            return self.observations.iloc[self.gbif_to_idx[gbif_id]]
        return None
        
    def get_observations_by_taxon(self, taxon_id: Union[str, int]) -> pd.DataFrame:
        """Get all observations for a taxon."""
        taxon_id = str(taxon_id)
        if taxon_id in self.taxon_to_gbifs:
            gbif_ids = self.taxon_to_gbifs[taxon_id]
            indices = [self.gbif_to_idx[gbif_id] for gbif_id in gbif_ids]
            return self.observations.iloc[indices]
        return pd.DataFrame()
        
    def get_vision_embedding(self, gbif_id: int, image_num: int = 1) -> Optional[torch.Tensor]:
        """
        Get vision embedding for a specific observation.
        
        Args:
            gbif_id: GBIF ID of the observation
            image_num: Image number (not used in v0.2.0 structure)
            
        Returns:
            Flattened vision embedding tensor or None if not found
        """
        gbif_id = int(gbif_id)
        
        # Check cache first
        cache_key = f"{gbif_id}_{image_num}"
        if cache_key in self.vision_embedding_cache:
            self.embedding_access_count[cache_key] = self.embedding_access_count.get(cache_key, 0) + 1
            return self.vision_embedding_cache[cache_key]
            
        # Check if this observation has vision embeddings
        if gbif_id not in self.vision_gbif_to_file_info:
            return None
            
        file_info = self.vision_gbif_to_file_info[gbif_id]
        file_idx = file_info['file_idx']
        
        # Load the parquet file containing this embedding
        embedding = self._load_vision_embedding_from_file(file_idx, gbif_id)
        
        # Cache the embedding
        if embedding is not None and self.config['caching']['enable_embedding_cache']:
            self._cache_embedding(cache_key, embedding)
            
        return embedding
        
    def _load_vision_embedding_from_file(self, file_idx: int, gbif_id: int) -> Optional[torch.Tensor]:
        """Load vision embedding from specific parquet file and convert to PyTorch tensor."""
        vision_dir = self.base_dir / self.config['data_paths']['vision_embeddings_dir']
        file_path = vision_dir / f"embeddings_{file_idx:06d}.parquet"
        
        if not file_path.exists():
            logger.warning(f"Vision embedding file not found: {file_path}")
            return None
            
        try:
            # Check if file is already cached
            if file_idx in self.vision_file_cache:
                df = self.vision_file_cache[file_idx]
            else:
                df = pd.read_parquet(file_path)
                # Cache the file if not too large
                if len(self.vision_file_cache) < 50:  # Limit file cache
                    self.vision_file_cache[file_idx] = df
                    
            # Find the specific embedding
            embedding_row = df[df['gbif_id'] == gbif_id]
            if len(embedding_row) == 0:
                return None
                
            embedding = embedding_row.iloc[0]['embedding']
            
            # Convert to numpy array if not already
            if hasattr(embedding, 'values'):
                embedding = embedding.values
            
            # Convert to PyTorch tensor (make a copy to avoid numpy warning)
            if isinstance(embedding, np.ndarray):
                embedding = torch.from_numpy(embedding.copy()).float()
            elif not isinstance(embedding, torch.Tensor):
                embedding = torch.tensor(embedding, dtype=torch.float32)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error loading vision embedding from {file_path}: {e}")
            return None
            
    def _cache_embedding(self, cache_key: str, embedding: np.ndarray):
        """Cache embedding with LRU eviction."""
        max_cache_size = self.config['caching']['max_embedding_cache_size']
        
        # Check cache size
        if len(self.vision_embedding_cache) >= max_cache_size:
            # Evict least recently used
            if self.embedding_access_count:
                lru_key = min(self.embedding_access_count.keys(), 
                             key=lambda k: self.embedding_access_count[k])
                del self.vision_embedding_cache[lru_key]
                del self.embedding_access_count[lru_key]
                
        self.vision_embedding_cache[cache_key] = embedding
        self.embedding_access_count[cache_key] = 1
        
    def reshape_vision_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Reshape V-JEPA 2 embedding to 4D structure based on validated implementation.
        
        Args:
            embedding: Flattened embedding tensor of shape (6,488,064,)
        
        Returns:
            4D embedding tensor of shape (8, 24, 24, 1408) - (temporal, height, width, features)
        """
        # Use PyTorch's fast view operation instead of reshape
        # Step 1: View as 2D [4608, 1408]
        embedding_2d = embedding.view(4608, 1408)
        
        # Step 2: View as 3D [8, 576, 1408] (validated dashboard structure)
        embedding_3d = embedding_2d.view(8, 576, 1408)
        
        # Step 3: View spatial 576 -> 24×24
        embedding_4d = embedding_3d.view(8, 24, 24, 1408)
        
        return embedding_4d
        
    def get_temporal_frame(self, embedding_4d: torch.Tensor, frame_idx: int) -> torch.Tensor:
        """Get a specific temporal frame (0-7)."""
        return embedding_4d[frame_idx]  # Shape: (24, 24, 1408)
        
    def get_spatial_patch(self, embedding_4d: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
        """Get a specific spatial patch."""
        return embedding_4d[t, h, w]  # Shape: (1408,)
        
    def get_image_level_embedding(self, embedding_4d: torch.Tensor) -> torch.Tensor:
        """Get mean embedding across all patches for image-level tasks."""
        return embedding_4d.mean(dim=(0, 1, 2))  # Shape: (1408,)
        
    def clear_cache(self):
        """Clear embedding cache to free memory."""
        self.vision_embedding_cache.clear()
        self.vision_file_cache.clear()
        self.embedding_access_count.clear()
        logger.info("All caches cleared")
        
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'embedding_cache_size': len(self.vision_embedding_cache),
            'file_cache_size': len(self.vision_file_cache),
            'max_embedding_cache_size': self.config['caching']['max_embedding_cache_size'],
            'total_accesses': sum(self.embedding_access_count.values()),
            'unique_accesses': len(self.embedding_access_count)
        }
        
    def get_species_with_embeddings(self) -> List[str]:
        """Get list of taxon IDs that have language embeddings."""
        # In v0.2.0, all species have language embeddings denormalized in observations
        return list(self.taxon_to_gbifs.keys())
        
    def get_language_embedding(self, taxon_id: str) -> Optional[np.ndarray]:
        """
        Get language embedding for a taxon.
        In v0.2.0, language embeddings are denormalized in the observations table.
        """
        observations = self.get_observations_by_taxon(taxon_id)
        if len(observations) > 0:
            # Get language embedding from first observation (all same for species)
            lang_emb = observations.iloc[0]['language_embedding']
            if isinstance(lang_emb, str):
                try:
                    return np.array(ast.literal_eval(lang_emb))
                except:
                    return None
            elif isinstance(lang_emb, (list, np.ndarray)):
                return np.array(lang_emb)
        return None
        
    def search_species(self, query: str) -> List[Dict]:
        """Search for species by name."""
        query_lower = query.lower()
        
        results = []
        unique_taxa = self.observations[['taxon_id', 'taxon_name']].drop_duplicates()
        
        for _, row in unique_taxa.iterrows():
            if query_lower in row['taxon_name'].lower():
                observations = self.get_observations_by_taxon(row['taxon_id'])
                has_vision = int(row['taxon_id']) in [
                    self.vision_index.iloc[i]['gbif_id'] 
                    for i in range(len(self.vision_index))
                ] if self.vision_index is not None else False
                
                results.append({
                    'taxon_id': row['taxon_id'],
                    'taxon_name': row['taxon_name'],
                    'observation_count': len(observations),
                    'has_vision_embeddings': has_vision
                })
                
        # Sort by observation count
        results.sort(key=lambda x: x['observation_count'], reverse=True)
        
        return results


# Example usage and testing
if __name__ == "__main__":
    # Initialize loader
    try:
        loader = HuggingFaceDataLoader()
        
        # Test basic functionality
        print("\n" + "="*60)
        print("Testing HuggingFaceDataLoader")
        print("="*60)
        
        # Test observation lookup
        test_gbif = list(loader.gbif_to_idx.keys())[0]
        obs = loader.get_observation(test_gbif)
        print(f"\nSample observation (GBIF {test_gbif}):")
        print(f"  Species: {obs['taxon_name']}")
        print(f"  Location: ({obs['latitude']:.4f}, {obs['longitude']:.4f})")
        print(f"  Year: {obs['year']}")
        print(f"  Has vision: {obs['has_vision']}")
        
        # Test species search
        print("\nSearching for 'Quercus'...")
        results = loader.search_species('Quercus')
        for result in results[:3]:
            print(f"  {result['taxon_name']}: {result['observation_count']} observations")
            
        # Test language embedding
        if len(results) > 0:
            test_taxon = results[0]['taxon_id']
            lang_emb = loader.get_language_embedding(test_taxon)
            if lang_emb is not None:
                print(f"\nLanguage embedding for {results[0]['taxon_name']}:")
                print(f"  Shape: {lang_emb.shape}")
                print(f"  Type: {lang_emb.dtype}")
            
        # Test vision embedding loading
        if loader.vision_index is not None and len(loader.vision_index) > 0:
            print("\nTesting vision embedding loading...")
            test_vision_gbif = loader.vision_index.iloc[0]['gbif_id']
            embedding = loader.get_vision_embedding(test_vision_gbif)
            if embedding is not None:
                print(f"  Flattened embedding shape: {embedding.shape}")
                
                # Test reshaping
                embedding_4d = loader.reshape_vision_embedding(embedding)
                print(f"  Reshaped to 4D: {embedding_4d.shape}")
                print(f"  Structure: (temporal, height, width, features)")
                
                # Test different representations
                frame_0 = loader.get_temporal_frame(embedding_4d, 0)
                print(f"  Temporal frame 0: {frame_0.shape}")
                
                image_emb = loader.get_image_level_embedding(embedding_4d)
                print(f"  Image-level embedding: {image_emb.shape}")
            else:
                print("  No embedding found")
                
        # Test cache stats
        print("\nCache statistics:")
        cache_stats = loader.get_cache_stats()
        for key, value in cache_stats.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        logger.error(f"Error testing data loader: {e}")
        import traceback
        traceback.print_exc()