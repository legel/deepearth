#!/usr/bin/env python3
"""
Optimized DeepEarth Multimodal Training Script with Embedding Cache

Key optimizations:
1. Cache vision embedding files in memory to avoid repeated loading
2. Pre-load all required embedding files at dataset initialization
3. Use efficient batch loading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import json
import argparse
import logging
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import umap
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VisionMLP(nn.Module):
    """MLP to project V-JEPA2 vision embeddings to universal space."""
    def __init__(self, vision_dim=1408, hidden_dim=256, universal_dim=2048, dropout=0.1):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, universal_dim)
        )
        
    def forward(self, x):
        # Global average pooling across spatial and temporal dimensions
        x_pooled = x.mean(dim=(1, 2, 3))  # (batch, 1408)
        return self.projection(x_pooled)


class LanguageMLP(nn.Module):
    """MLP to project DeepSeek language embeddings to universal space."""
    def __init__(self, language_dim=7168, hidden_dim=256, universal_dim=2048, dropout=0.1):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(language_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, universal_dim)
        )
        
    def forward(self, x):
        return self.projection(x)


class LanguageDecoder(nn.Module):
    """MLP decoder to reconstruct language embeddings from universal space."""
    def __init__(self, universal_dim=2048, hidden_dim=256, language_dim=7168, dropout=0.1):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(universal_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, language_dim)
        )
        
    def forward(self, x):
        return self.decoder(x)


class MultimodalMaskingModel(nn.Module):
    """Complete multimodal system for masked reconstruction."""
    def __init__(self, vision_dim=1408, language_dim=7168, 
                 hidden_dim=256, universal_dim=2048, dropout=0.1):
        super().__init__()
        
        self.vision_mlp = VisionMLP(vision_dim, hidden_dim, universal_dim, dropout)
        self.language_mlp = LanguageMLP(language_dim, hidden_dim, universal_dim, dropout)
        self.language_decoder = LanguageDecoder(universal_dim, hidden_dim, language_dim, dropout)
        
    def forward(self, vision_emb, language_emb, mask_language=True, mask_prob=0.5):
        batch_size = vision_emb.shape[0]
        device = vision_emb.device
        
        # Encode both modalities
        vision_universal = self.vision_mlp(vision_emb)
        language_universal = self.language_mlp(language_emb)
        
        # Create mask
        if mask_language:
            mask = torch.rand(batch_size, device=device) < mask_prob
            language_reconstructed = self.language_decoder(vision_universal)
        else:
            mask = torch.rand(batch_size, device=device) < mask_prob
            language_reconstructed = self.language_decoder(language_universal)
        
        return {
            'vision_universal': vision_universal,
            'language_universal': language_universal,
            'language_reconstructed': language_reconstructed,
            'mask': mask
        }


class DeepEarthDataset(Dataset):
    """Optimized Dataset with vision embedding cache."""
    
    def __init__(self, 
                 data_dir: Path,
                 observation_ids: Optional[List[int]] = None,
                 taxon_filter: Optional[List[str]] = None,
                 max_samples: Optional[int] = None,
                 split: str = 'train',
                 force_split: bool = False,
                 cache_embeddings: bool = True):
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.cache_embeddings = cache_embeddings
        self.vision_cache = {}  # Cache for vision embeddings
        
        # Load observations
        logger.info("Loading observations...")
        obs_path = self.data_dir / "observations.parquet"
        self.observations_df = pd.read_parquet(obs_path)
        
        # Check available splits
        if 'split' in self.observations_df.columns:
            available_splits = self.observations_df['split'].unique()
            logger.info(f"Available splits: {available_splits}")
            
            if split in available_splits:
                self.observations_df = self.observations_df[self.observations_df['split'] == split]
                logger.info(f"Filtered to {len(self.observations_df)} observations in {split} split")
            elif not force_split:
                logger.warning(f"Split '{split}' not found. Using all data.")
        
        # Filter by specific observation IDs if provided
        if observation_ids is not None:
            self.observations_df = self.observations_df[self.observations_df['gbif_id'].isin(observation_ids)]
        
        # Filter by taxa if specified
        if taxon_filter:
            self.observations_df = self.observations_df[self.observations_df['taxon_name'].isin(taxon_filter)]
            logger.info(f"Filtered to {len(self.observations_df)} observations for specified taxa")
            
            # Show taxon counts
            taxon_counts = self.observations_df['taxon_name'].value_counts()
            for taxon in taxon_filter:
                count = taxon_counts.get(taxon, 0)
                logger.info(f"  {taxon}: {count} observations")
        
        # Limit samples if specified
        if max_samples and len(self.observations_df) > max_samples:
            self.observations_df = self.observations_df.sample(n=max_samples, random_state=42)
            logger.info(f"Limited to {max_samples} samples")
        
        # Filter to only observations with vision data
        if 'has_vision' in self.observations_df.columns:
            self.observations_df = self.observations_df[self.observations_df['has_vision']]
            logger.info(f"Filtered to {len(self.observations_df)} observations with vision data")
        
        # Load vision index for efficient lookup
        logger.info("Loading vision index...")
        self.vision_index = pd.read_parquet(self.data_dir / "vision_index.parquet")
        
        # Reset index for consistent indexing
        self.observations_df = self.observations_df.reset_index(drop=True)
        
        # Pre-cache vision embeddings if requested
        if self.cache_embeddings:
            self._cache_vision_embeddings()
        
        logger.info(f"Dataset initialized with {len(self.observations_df)} observations")
    
    def _cache_vision_embeddings(self):
        """Pre-load all required vision embedding files into memory."""
        # Find which files we need
        needed_gbif_ids = set(self.observations_df['gbif_id'])
        vision_info_needed = self.vision_index[self.vision_index['gbif_id'].isin(needed_gbif_ids)]
        unique_files = vision_info_needed['filename'].unique()
        
        logger.info(f"Pre-loading {len(unique_files)} vision embedding files...")
        
        for filename in tqdm(unique_files, desc="Loading vision files"):
            if filename not in self.vision_cache:
                emb_path = self.data_dir / f"vision_embeddings/{filename}"
                start_time = time.time()
                self.vision_cache[filename] = pd.read_parquet(emb_path)
                logger.debug(f"Loaded {filename} in {time.time() - start_time:.2f}s")
        
        logger.info(f"Vision embedding cache ready with {len(self.vision_cache)} files")
        
    def __len__(self):
        return len(self.observations_df)
    
    def __getitem__(self, idx):
        # Get observation
        obs = self.observations_df.iloc[idx]
        
        # Get vision embedding
        vision_embedding = self._load_vision_embedding(obs['gbif_id'])
        
        # Get language embedding
        language_embedding = self._load_language_embedding(obs)
        
        return {
            'vision_embedding': torch.tensor(vision_embedding, dtype=torch.float32),
            'language_embedding': torch.tensor(language_embedding, dtype=torch.float32),
            'taxon_name': obs['taxon_name'],
            'gbif_id': obs['gbif_id'],
            'latitude': obs['latitude'],
            'longitude': obs['longitude'],
            'timestamp': self._create_timestamp(obs)
        }
    
    def _load_vision_embedding(self, gbif_id):
        """Load vision embedding from cache or parquet files."""
        # Find the file containing this observation
        vision_info = self.vision_index[self.vision_index['gbif_id'] == gbif_id]
        
        if len(vision_info) == 0:
            # Return random embedding if not found
            logger.warning(f"No vision embedding found for gbif_id {gbif_id}")
            return np.random.randn(8, 24, 24, 1408).astype(np.float32)
        
        # Get file info from vision index
        filename = vision_info.iloc[0]['filename']
        
        # Get embedding dataframe from cache or load it
        if self.cache_embeddings and filename in self.vision_cache:
            emb_df = self.vision_cache[filename]
        else:
            emb_path = self.data_dir / f"vision_embeddings/{filename}"
            emb_df = pd.read_parquet(emb_path)
        
        # Find the specific embedding for this gbif_id
        embedding_row = emb_df[emb_df['gbif_id'] == gbif_id]
        
        if len(embedding_row) == 0:
            logger.warning(f"No embedding found for gbif_id {gbif_id} in {filename}")
            return np.random.randn(8, 24, 24, 1408).astype(np.float32)
        
        # Get the embedding from the 'embedding' column
        embedding = np.array(embedding_row.iloc[0]['embedding'], dtype=np.float32)
        
        # The embedding is stored as a flattened array
        # Original shape is [4608, 1408] which is [8*24*24, 1408]
        # We need to reshape it to [8, 24, 24, 1408]
        expected_size = 8 * 24 * 24 * 1408  # 6488064
        
        if embedding.size == expected_size:
            # First reshape to [4608, 1408] then to [8, 24, 24, 1408]
            embedding = embedding.reshape(4608, 1408)  # [8*24*24, 1408]
            embedding = embedding.reshape(8, 24, 24, 1408)  # Final shape
        else:
            logger.warning(f"Embedding size mismatch for gbif_id {gbif_id}. Expected {expected_size}, got {embedding.size}")
            # Fall back to random
            embedding = np.random.randn(8, 24, 24, 1408).astype(np.float32)
        
        return embedding
    
    def _load_language_embedding(self, obs):
        """Load or create language embedding."""
        # Check if language embedding exists
        if 'language_embedding' in self.observations_df.columns:
            lang_emb = obs['language_embedding']
            
            # Check if it's not None and has content
            if lang_emb is not None:
                try:
                    # Handle different possible formats
                    if isinstance(lang_emb, np.ndarray):
                        embedding = lang_emb
                    elif isinstance(lang_emb, list):
                        embedding = np.array(lang_emb)
                    else:
                        # Try to convert to array
                        embedding = np.array(lang_emb)
                    
                    # Check if it's the right size
                    if embedding.size == 7168:
                        return embedding.astype(np.float32)
                    else:
                        logger.debug(f"Language embedding has wrong size: {embedding.size} (expected 7168)")
                except Exception as e:
                    logger.debug(f"Error loading language embedding: {e}")
        
        # For now, create a deterministic mock embedding based on taxon
        np.random.seed(hash(str(obs['taxon_name'])) % 2**32)
        embedding = np.random.randn(7168).astype(np.float32)
        np.random.seed()  # Reset random seed
        
        return embedding
    
    def _create_timestamp(self, obs):
        """Create timestamp from date components."""
        try:
            year = int(obs.get('year', 2023))
            month = int(obs.get('month', 1)) if pd.notna(obs.get('month')) else 1
            day = int(obs.get('day', 1)) if pd.notna(obs.get('day')) else 1
            hour = int(obs.get('hour', 0)) if pd.notna(obs.get('hour')) else 0
            
            dt = datetime(year, month, day, hour)
            return dt.timestamp()
        except:
            # Return a default timestamp if parsing fails
            return datetime(2023, 1, 1).timestamp()


def main():
    parser = argparse.ArgumentParser(description='DeepEarth Multimodal Masking Training')
    parser.add_argument('--data-dir', type=str, default='../dashboard/huggingface_dataset/hf_download/',
                       help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device: cuda, cpu, or auto')
    parser.add_argument('--max-samples', type=int, default=1000,
                       help='Maximum samples to load (for testing)')
    parser.add_argument('--no-species-filter', action='store_true',
                       help='Use all taxa without filtering')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio if no test split exists')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable vision embedding caching (saves memory but slower)')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("🌍 DeepEarth Multimodal Training (Optimized)")
    print(f"Device: {device}")
    print(f"Vision caching: {'Disabled' if args.no_cache else 'Enabled'}")
    
    # Target Florida taxa that exist in the dataset
    if args.no_species_filter:
        target_taxa = None
        print("Using all available taxa")
    else:
        target_taxa = [
            'Callicarpa americana',  # Most common
            'Serenoa repens',        # Saw palmetto
            'Sabal palmetto',        # Cabbage palm
            'Quercus virginiana',    # Southern live oak
            'Pinus elliottii',       # Slash pine
            'Magnolia grandiflora',  # Southern magnolia
            'Ilex cassine',          # Dahoon holly
            'Taxodium distichum',    # Bald cypress
            'Lyonia lucida',         # Fetterbush
            'Acer rubrum'            # Red maple
        ]
        print(f"Filtering to {len(target_taxa)} target taxa")
    
    # Create datasets
    print("\n📦 Creating datasets...")
    
    # First, try to load train/test splits
    train_dataset = DeepEarthDataset(
        data_dir=args.data_dir,
        taxon_filter=target_taxa,
        max_samples=args.max_samples,
        split='train',
        cache_embeddings=not args.no_cache
    )
    
    # Check if test split exists
    try:
        val_dataset = DeepEarthDataset(
            data_dir=args.data_dir,
            taxon_filter=target_taxa,
            max_samples=None,  # Don't limit validation set
            split='test',
            cache_embeddings=not args.no_cache
        )
        
        if len(val_dataset) == 0:
            raise ValueError("No test data")
            
    except Exception as e:
        logger.info("No test split found, creating train/val split from training data")
        
        # Create a validation split from training data
        total_len = len(train_dataset)
        val_len = int(total_len * args.val_split)
        train_len = total_len - val_len
        
        # Note: When using random_split, the underlying dataset is shared
        # so the cache will be shared between train and val
        train_dataset, val_dataset = random_split(
            train_dataset, 
            [train_len, val_len],
            generator=torch.Generator().manual_seed(42)
        )
    
    if len(train_dataset) == 0:
        print("❌ No training data available after filtering!")
        return 1
    
    # Use more workers for faster data loading
    num_workers = min(4, len(train_dataset) // args.batch_size)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False
    )
    
    print(f"\nDataset statistics:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"DataLoader workers: {num_workers}")
    
    # Initialize model
    print("\n🧠 Initializing model...")
    model = MultimodalMaskingModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in progress_bar:
            vision_emb = batch['vision_embedding'].to(device)
            language_emb = batch['language_embedding'].to(device)
            
            # Forward pass
            outputs = model(vision_emb, language_emb, mask_language=True)
            
            # Compute loss only on masked samples
            mask = outputs['mask']
            if mask.any():
                loss = F.mse_loss(
                    outputs['language_reconstructed'][mask],
                    language_emb[mask]
                )
            else:
                loss = 0.1 * F.mse_loss(
                    outputs['language_reconstructed'],
                    language_emb
                )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                vision_emb = batch['vision_embedding'].to(device)
                language_emb = batch['language_embedding'].to(device)
                
                outputs = model(vision_emb, language_emb, mask_language=True)
                
                mask = outputs['mask']
                if mask.any():
                    loss = F.mse_loss(
                        outputs['language_reconstructed'][mask],
                        language_emb[mask]
                    )
                else:
                    loss = 0.1 * F.mse_loss(
                        outputs['language_reconstructed'],
                        language_emb
                    )
                
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            output_dir = Path("models")
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_dir / "multimodal_model_best.pth")
            print(f"  → Saved best model (val_loss={best_val_loss:.4f})")
    
    print("\n✅ Training complete!")
    
    # Save final model
    output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "multimodal_model_final.pth")
    print(f"Model saved to {output_dir / 'multimodal_model_final.pth'}")
    print(f"Best model saved to {output_dir / 'multimodal_model_best.pth'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
