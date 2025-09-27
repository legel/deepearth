#!/usr/bin/env python3
"""
Earth4D to AlphaEarth Training Pipeline
========================================

Train a DeepEarth model that uses Earth4D spacetime encoder to predict
AlphaEarth 64D embeddings from (latitude, longitude, elevation, time) coordinates.

This pipeline automatically downloads the AlphaEarth dataset if not present locally.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import sys
import os
import json
import pickle
from datetime import datetime
from tqdm import tqdm
import subprocess
import threading
import warnings
import urllib.request
import hashlib
warnings.filterwarnings('ignore')

# Add current directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from earth4d import Earth4D

# Configure matplotlib for better quality
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Default data directory and URLs
DATA_DIR = Path(SCRIPT_DIR) / 'data' / 'alphaearth'
METADATA_URL = 'https://storage.googleapis.com/deepearth/datasets/alphaearth/20250927_172450_alphaearth_metadata.csv'
EMBEDDINGS_URL = 'https://storage.googleapis.com/deepearth/datasets/alphaearth/20250927_172519_alphaearth_embeddings.pt'

# Expected file checksums for validation (SHA256)
METADATA_SHA256 = None  # Will be computed on first download
EMBEDDINGS_SHA256 = None  # Will be computed on first download


def download_with_progress(url: str, destination: Path, description: str = "Downloading"):
    """Download a file with progress bar."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Get file size
    with urllib.request.urlopen(url) as response:
        total_size = int(response.headers.get('Content-Length', 0))

    # Download with progress bar
    with tqdm(total=total_size, unit='B', unit_scale=True, desc=description) as pbar:
        def download_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if downloaded < total_size:
                pbar.update(block_size)

        urllib.request.urlretrieve(url, destination, reporthook=download_hook)

    return destination


def ensure_alphaearth_data(data_dir: Path = DATA_DIR, force_download: bool = False) -> tuple:
    """
    Ensure AlphaEarth dataset is available locally.

    Downloads the dataset from Google Cloud Storage if not present.

    Args:
        data_dir: Directory to store/load data
        force_download: Force re-download even if files exist

    Returns:
        Tuple of (metadata_path, embeddings_path)
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    # Expected file paths
    metadata_path = data_dir / 'alphaearth_metadata.csv'
    embeddings_path = data_dir / 'alphaearth_embeddings.pt'

    # Check if files exist and are valid
    if not force_download and metadata_path.exists() and embeddings_path.exists():
        # Validate file sizes if we have expected values
        metadata_size = metadata_path.stat().st_size
        embeddings_size = embeddings_path.stat().st_size

        print(f"Found existing AlphaEarth data:")
        print(f"  Metadata: {metadata_path} ({metadata_size / 1024**2:.1f} MB)")
        print(f"  Embeddings: {embeddings_path} ({embeddings_size / 1024**2:.1f} MB)")

        # Basic validation - files should not be empty
        if metadata_size > 1000 and embeddings_size > 1000000:  # Metadata > 1KB, Embeddings > 1MB
            return metadata_path, embeddings_path
        else:
            print("  Warning: Files appear to be corrupted, re-downloading...")

    # Download files
    print("\n" + "="*60)
    print("DOWNLOADING ALPHAEARTH DATASET")
    print("="*60)
    print(f"Downloading to: {data_dir}")
    print()

    try:
        # Download metadata
        print("1. Downloading metadata...")
        download_with_progress(
            METADATA_URL,
            metadata_path,
            "Metadata"
        )
        metadata_size = metadata_path.stat().st_size
        print(f"   Downloaded: {metadata_size / 1024**2:.1f} MB")

        # Download embeddings
        print("\n2. Downloading embeddings (this may take a while)...")
        download_with_progress(
            EMBEDDINGS_URL,
            embeddings_path,
            "Embeddings"
        )
        embeddings_size = embeddings_path.stat().st_size
        print(f"   Downloaded: {embeddings_size / 1024**2:.1f} MB")

        # Validate downloads
        print("\n3. Validating downloads...")

        # Check metadata is valid CSV
        try:
            test_df = pd.read_csv(metadata_path, nrows=5)
            print(f"   ✓ Metadata valid: {len(test_df)} test rows loaded")
        except Exception as e:
            raise ValueError(f"Downloaded metadata file is invalid: {e}")

        # Check embeddings are valid PyTorch tensor
        try:
            test_tensor = torch.load(embeddings_path, map_location='cpu')
            print(f"   ✓ Embeddings valid: shape {test_tensor.shape}, dtype {test_tensor.dtype}")
            del test_tensor  # Free memory
        except Exception as e:
            raise ValueError(f"Downloaded embeddings file is invalid: {e}")

        print("\n✓ AlphaEarth dataset downloaded successfully!")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n✗ Error downloading AlphaEarth data: {e}")
        print("Please check your internet connection and try again.")
        print("You can also manually download the files from:")
        print(f"  - {METADATA_URL}")
        print(f"  - {EMBEDDINGS_URL}")
        print(f"And place them in: {data_dir}")
        raise

    return metadata_path, embeddings_path


class AlphaEarthGPUDataset(Dataset):
    """GPU-preloaded dataset for maximum training speed.
    
    Loads all coordinates and embeddings directly onto GPU memory.
    Eliminates CPU->GPU transfer bottleneck during training.
    """
    
    def __init__(self, 
                 metadata_path: str,
                 embeddings_path: str,
                 device: str = 'cuda',
                 max_samples: int = None,
                 time_range: tuple = (1900, 2100),
                 random_seed: int = 42,
                 test_for_nan: bool = True,
                 model: nn.Module = None):
        """
        Initialize GPU-preloaded dataset.
        
        Args:
            metadata_path: Path to metadata CSV
            embeddings_path: Path to embeddings .pt file
            device: Device to load tensors on ('cuda' or 'cpu')
            max_samples: Maximum number of samples to load
            time_range: (min_year, max_year) for time normalization
            random_seed: Random seed for sampling
        """
        self.device = device
        self.time_range = time_range
        
        print(f"Loading metadata from {metadata_path}...")
        metadata = pd.read_csv(metadata_path)
        
        print(f"Loading embeddings from {embeddings_path}...")
        embeddings = torch.load(embeddings_path)
        
        # Check if embeddings are int8 or float32
        if embeddings.dtype == torch.int8:
            print(f"  Converting int8 embeddings to float32...")
            embeddings = embeddings.float() / 127.0  # Convert once
        
        print(f"  Embeddings: {embeddings.shape} {embeddings.dtype} ({embeddings.nbytes / 1024**2:.1f} MB)")
        
        # Filter out samples with NaN elevation values
        nan_mask = metadata.elevation_m.isna()
        if nan_mask.any():
            print(f"Filtering out {nan_mask.sum():,} samples with NaN elevation values")
            valid_indices = np.where(~nan_mask)[0]
            metadata = metadata.iloc[valid_indices].reset_index(drop=True)
            embeddings = embeddings[valid_indices]
            print(f"Retained {len(metadata):,} valid samples")
        
        # Sample if needed
        if max_samples and max_samples < len(metadata):
            np.random.seed(random_seed)
            indices = np.random.choice(len(metadata), max_samples, replace=False)
            metadata = metadata.iloc[indices].reset_index(drop=True)
            embeddings = embeddings[indices]
            print(f"Sampled {max_samples:,} samples")
        
        # Parse dates and normalize time
        time_norm = self._prepare_temporal_data(metadata)
        
        print(f"Converting coordinates to GPU tensors...")
        
        # Create coordinate tensor [N, 4] with [lat, lon, elev, time]
        coords_array = torch.tensor([
            metadata['latitude'].values,
            metadata['longitude'].values,
            metadata['elevation_m'].values,
            time_norm.values
        ], dtype=torch.float32).T
        
        # Filter out any samples with extreme values that might cause NaN
        valid_mask = (
            (torch.abs(coords_array[:, 0]) <= 90) &  # Valid latitude
            (torch.abs(coords_array[:, 1]) <= 180) &  # Valid longitude
            (torch.abs(coords_array[:, 2]) <= 10000) &  # Reasonable elevation
            (coords_array[:, 3] >= 0) & (coords_array[:, 3] <= 1) &  # Valid time
            torch.isfinite(coords_array).all(dim=1) &  # No inf/nan
            torch.isfinite(embeddings).all(dim=1)  # No inf/nan in embeddings
        )
        
        if not valid_mask.all():
            invalid_count = (~valid_mask).sum().item()
            print(f"Filtering out {invalid_count} samples with extreme/invalid values")
            coords_array = coords_array[valid_mask]
            embeddings = embeddings[valid_mask]
        
        # Additional filtering for problematic elevation values
        # Some elevations might be technically valid but cause numerical issues
        elev_mask = (
            (coords_array[:, 2] >= -500) &  # No deep ocean trenches
            (coords_array[:, 2] <= 9000) &  # No extreme mountain peaks
            (torch.abs(coords_array[:, 2]) != 9999) &  # Common missing data value
            (torch.abs(coords_array[:, 2]) != 32767) &  # Int16 max (error value)
            (torch.abs(coords_array[:, 2]) != -32768)  # Int16 min (error value)
        )
        
        if not elev_mask.all():
            invalid_count = (~elev_mask).sum().item()
            print(f"Filtering out {invalid_count} samples with problematic elevation values")
            coords_array = coords_array[elev_mask]
            embeddings = embeddings[elev_mask]
        
        # Move to GPU
        self.coords = coords_array.to(device)
        self.embeddings = embeddings.to(device)
        
        # Test for NaN-causing samples using actual model if provided
        if test_for_nan and model is not None:
            print("\nTesting samples for NaN with actual model forward pass...")
            nan_indices = self._test_for_nan_samples(model)
            if len(nan_indices) > 0:
                print(f"Found {len(nan_indices)} NaN-causing samples, removing them...")
                # Create mask for valid samples
                all_indices = torch.arange(len(self.coords), device=device)
                nan_mask = torch.zeros(len(self.coords), dtype=torch.bool, device=device)
                nan_mask[nan_indices] = True
                valid_mask = ~nan_mask
                
                # Filter out NaN-causing samples
                self.coords = self.coords[valid_mask]
                self.embeddings = self.embeddings[valid_mask]
                print(f"Dataset reduced to {len(self.coords):,} samples after NaN filtering")
        
        print(f"GPU Dataset ready: {len(self):,} samples")
        print(f"  Coords: {self.coords.shape} on {self.coords.device}")
        print(f"  Embeddings: {self.embeddings.shape} on {self.embeddings.device}")
        print(f"  GPU memory: {(self.coords.nbytes + self.embeddings.nbytes) / 1024**3:.2f} GB")
        
        # Print ranges
        print(f"Coordinate ranges:")
        print(f"  Latitude: [{self.coords[:, 0].min():.2f}, {self.coords[:, 0].max():.2f}]")
        print(f"  Longitude: [{self.coords[:, 1].min():.2f}, {self.coords[:, 1].max():.2f}]")
        print(f"  Elevation: [{self.coords[:, 2].min():.1f}, {self.coords[:, 2].max():.1f}] m")
        print(f"  Time: [{self.coords[:, 3].min():.3f}, {self.coords[:, 3].max():.3f}] (normalized)")
        
        # Final verification - no NaN/Inf should remain
        coords_has_nan = torch.isnan(self.coords).any() or torch.isinf(self.coords).any()
        embeds_has_nan = torch.isnan(self.embeddings).any() or torch.isinf(self.embeddings).any()
        if coords_has_nan or embeds_has_nan:
            print("⚠️  WARNING: Dataset still contains NaN/Inf after filtering!")
            print(f"    Coords has NaN/Inf: {coords_has_nan}")
            print(f"    Embeddings has NaN/Inf: {embeds_has_nan}")
        else:
            print("✓ Dataset verified: No NaN/Inf values")
    
    def _prepare_temporal_data(self, metadata):
        """Parse dates and normalize to [0, 1] range."""
        dates = pd.to_datetime(metadata.event_date, errors='coerce', format='ISO8601', utc=True)
        
        # For any that failed, try without format specification
        if dates.isna().any():
            failed_mask = dates.isna()
            dates[failed_mask] = pd.to_datetime(
                metadata.event_date[failed_mask], 
                errors='coerce',
                infer_datetime_format=True,
                utc=True
            )
        
        # For any remaining NaT, fill with default date
        if dates.isna().any():
            n_failed = dates.isna().sum()
            if n_failed > 0:
                print(f"Warning: {n_failed} dates could not be parsed, using default year 2000")
                dates = dates.fillna(pd.Timestamp('2000-01-01', tz='UTC'))
        
        # Convert to DatetimeIndex and remove timezone for compatibility
        dates = pd.DatetimeIndex(dates.dt.tz_localize(None))
        
        # Calculate fractional year
        years = dates.year.astype(float)
        day_of_year = dates.dayofyear.astype(float)
        
        # Handle leap years properly
        is_leap = dates.is_leap_year
        days_in_year = np.where(is_leap, 366.0, 365.0)
        
        fractional_years = years + (day_of_year - 1) / days_in_year
        
        # Normalize to [0, 1] based on time_range
        min_year, max_year = self.time_range
        time_norm = pd.Series((fractional_years - min_year) / (max_year - min_year))
        time_norm = time_norm.clip(0, 1)
        
        return time_norm
    
    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self, idx):
        """Get a single sample - direct GPU tensor access."""
        return self.coords[idx], self.embeddings[idx]
    
    def _test_for_nan_samples(self, model: nn.Module, batch_size: int = 1000):
        """Test which samples cause NaN in model forward pass."""
        model.eval()
        nan_indices = []
        
        with torch.no_grad():
            for start_idx in range(0, len(self.coords), batch_size):
                end_idx = min(start_idx + batch_size, len(self.coords))
                batch_coords = self.coords[start_idx:end_idx]
                
                try:
                    # Test forward pass
                    predictions = model(batch_coords)
                    
                    # Check for NaN in this batch
                    if torch.isnan(predictions).any():
                        # Test each sample individually
                        for i in range(len(batch_coords)):
                            single_coord = batch_coords[i:i+1]
                            single_pred = model(single_coord)
                            
                            if torch.isnan(single_pred).any():
                                global_idx = start_idx + i
                                nan_indices.append(global_idx)
                                
                                if len(nan_indices) <= 10:  # Print first 10 problematic samples
                                    coord = single_coord[0].cpu().numpy()
                                    print(f"  NaN sample {global_idx}: lat={coord[0]:.2f}, lon={coord[1]:.2f}, "
                                          f"elev={coord[2]:.1f}, time={coord[3]:.3f}")
                
                except RuntimeError as e:
                    # Entire batch causes error
                    print(f"  Batch {start_idx}-{end_idx} causes error: {e}")
                    nan_indices.extend(range(start_idx, end_idx))
                
                # Progress indicator
                if start_idx % 50000 == 0 and start_idx > 0:
                    print(f"  Tested {start_idx:,}/{len(self.coords):,} samples, found {len(nan_indices)} NaN-causing")
        
        model.train()  # Reset to training mode
        return nan_indices


class AlphaEarthPredictor(nn.Module):
    """DeepEarth model combining Earth4D encoder with MLP for embedding prediction."""
    
    def __init__(self,
                 spatial_levels: int = 24,
                 temporal_levels: int = 19,
                 spatial_log2_hashmap_size: int = 22,
                 temporal_log2_hashmap_size: int = 18,
                 mlp_hidden_dims: list = [256, 256, 256],
                 output_dim: int = 64,
                 dropout: float = 0.1):
        """
        Initialize DeepEarth model.
        
        Args:
            spatial_levels: Number of spatial hash levels
            temporal_levels: Number of temporal hash levels
            mlp_hidden_dims: List of hidden dimensions for MLP
            output_dim: Output embedding dimension (64 for AlphaEarth)
            dropout: Dropout probability
        """
        super().__init__()
        
        # Earth4D spacetime encoder
        self.earth4d = Earth4D(
            spatial_levels=spatial_levels,
            temporal_levels=temporal_levels,
            spatial_log2_hashmap_size=spatial_log2_hashmap_size,
            temporal_log2_hashmap_size=18,
            verbose=False
        )
        
        # Get encoder output dimension
        encoder_dim = self.earth4d.get_output_dim()
        
        # Initialize Earth4D embeddings with smaller, more stable values
        with torch.no_grad():
            for name, param in self.earth4d.named_parameters():
                if 'embeddings' in name:
                    # Initialize embeddings with smaller range to prevent overflow
                    torch.nn.init.uniform_(param, -0.1, 0.1)
        
        # Pre-allocate GPU memory to avoid fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Reserve some GPU memory
            dummy = torch.zeros((10000, encoder_dim), device='cuda')
            del dummy
        
        # Input normalization layer to scale up Earth4D features
        # Earth4D outputs are very small (~0.1 range), need to scale up
        self.input_norm = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, mlp_hidden_dims[0])
        )
        
        # Build MLP layers
        layers = []
        prev_dim = mlp_hidden_dims[0]
        
        for hidden_dim in mlp_hidden_dims[1:]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer with tanh to match AlphaEarth range [-1, 1]
        layers.extend([
            nn.Linear(prev_dim, output_dim),
            nn.Tanh()  # Output range [-1, 1] to match normalized AlphaEarth
        ])
        
        self.mlp = nn.Sequential(*layers)
        
        # Store dimensions
        self.encoder_dim = encoder_dim
        self.output_dim = output_dim
        
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            coords: Input coordinates [batch, 4] with [lat, lon, elev, time]
            
        Returns:
            Predicted embeddings [batch, 64]
        """
        # Check inputs for NaN
        if torch.isnan(coords).any():
            print(f"NaN in model input coords! Shape: {coords.shape}")
            print(f"  Coords stats: min={coords.min():.2f}, max={coords.max():.2f}")
            print(f"  NaN count: {torch.isnan(coords).sum().item()}")
        
        # Encode spatiotemporal coordinates
        spacetime_features = self.earth4d(coords)
        
        # Check Earth4D output
        if torch.isnan(spacetime_features).any():
            print(f"NaN in Earth4D output! Shape: {spacetime_features.shape}")
            print(f"  Features stats: min={spacetime_features.min():.2f}, max={spacetime_features.max():.2f}")
            print(f"  NaN count: {torch.isnan(spacetime_features).sum().item()}")
        
        # Normalize and scale up features
        normalized_features = self.input_norm(spacetime_features)
        
        # Check after normalization
        if torch.isnan(normalized_features).any():
            print(f"NaN after input normalization! Shape: {normalized_features.shape}")
            print(f"  Features stats: min={normalized_features.min():.2f}, max={normalized_features.max():.2f}")
            print(f"  NaN count: {torch.isnan(normalized_features).sum().item()}")
        
        # Predict embedding through MLP
        embedding = self.mlp(normalized_features)
        
        # Check final output
        if torch.isnan(embedding).any():
            print(f"NaN in final MLP output! Shape: {embedding.shape}")
            print(f"  Embedding stats: min={embedding.min():.2f}, max={embedding.max():.2f}")
            print(f"  NaN count: {torch.isnan(embedding).sum().item()}")
        
        return embedding
    
    def get_params_count(self) -> dict:
        """Get parameter counts for different components."""
        earth4d_params = sum(p.numel() for p in self.earth4d.parameters())
        mlp_params = sum(p.numel() for p in self.mlp.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            'earth4d': earth4d_params,
            'mlp': mlp_params,
            'total': total_params,
            'earth4d_mb': earth4d_params * 4 / 1024 / 1024,
            'mlp_mb': mlp_params * 4 / 1024 / 1024,
            'total_mb': total_params * 4 / 1024 / 1024
        }


class Trainer:
    """Training pipeline for AlphaEarthPredictor model."""
    
    def __init__(self, 
                 model: AlphaEarthPredictor,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 device: str = 'cuda',
                 output_dir: str = './outputs'):
        """Initialize trainer."""
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.train_losses = []
        self.train_mapes = []
        self.test_losses = []
        self.test_mapes = []
        self.epoch_times = []
        
        # Fixed test sample for tracking
        self.fixed_test_batch = None
        self.fixed_test_predictions = []
        
        # For async visualization
        self.viz_thread = None
        self.autoencoder_model_path = None
        self.bay_area_indices = None
        
    def compute_mape(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute Mean Absolute Percentage Error as percentage of dimension range.
        
        Since AlphaEarth embeddings are normalized to roughly [-1, 1] range,
        we compute the error as a percentage of the total possible range (2.0).
        This gives us the average percentage deviation per dimension.
        """
        # Handle NaN values
        if torch.isnan(pred).any() or torch.isnan(target).any():
            return float('nan')
        
        # Compute absolute differences
        abs_diff = torch.abs(pred - target)
        
        # Normalize by the expected range of the embeddings (roughly 2.0 for [-1, 1])
        # This gives us the fraction of the total range that the error represents
        dimension_range = 2.0
        normalized_error = abs_diff / dimension_range
        
        # Return as percentage
        return (normalized_error.mean() * 100).item()
    
    def train_epoch(self, model: nn.Module, loader: DataLoader, 
                   optimizer: optim.Optimizer, criterion: nn.Module, 
                   epoch_num: int = 1) -> tuple:
        """Train for one epoch with direct GPU indexing."""
        model.train()
        total_loss = 0
        total_mape = 0
        num_batches = 0
        
        # Exponential moving average for display
        ema_loss = None
        ema_mape = None
        ema_alpha = 0.1
        
        # For fractional epoch saves - after specific batches (1, 5, 10, 20)
        # Maps batch index to label for saving
        fractional_batch_checkpoints = {
            1: "01_03",   # After 1st batch 
            5: "01_17",   # After 5th batch
            10: "01_33",  # After 10th batch
            20: "01_67"   # After 20th batch
        } if epoch_num == 1 else {}
        
        # Get the dataset directly for GPU indexing
        # Handle Subset wrapper from random_split
        if hasattr(loader.dataset, 'dataset'):  # This is a Subset
            base_dataset = loader.dataset.dataset
            subset_indices = torch.tensor(loader.dataset.indices, device=self.device)
            n_samples = len(loader.dataset)
        else:
            base_dataset = loader.dataset
            subset_indices = None
            n_samples = len(loader.dataset)
        
        batch_size = loader.batch_size
        n_batches = n_samples // batch_size
        
        # Generate shuffled indices on GPU
        indices = torch.randperm(n_samples, device=self.device)
        
        # Direct GPU iteration - no DataLoader overhead
        pbar = tqdm(range(n_batches), desc="Training", leave=False)
        for batch_idx in pbar:
            # Get batch indices
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_local_indices = indices[start_idx:end_idx]
            
            # Map to actual dataset indices if using subset
            if subset_indices is not None:
                batch_indices = subset_indices[batch_local_indices]
            else:
                batch_indices = batch_local_indices
            
            # Direct GPU indexing - no CPU involvement
            coords = base_dataset.coords[batch_indices]
            targets = base_dataset.embeddings[batch_indices]
            
            # Forward pass
            predictions = model(coords)
            loss = criterion(predictions, targets)
            
            # Check for NaN in forward pass
            if torch.isnan(loss) or torch.isnan(predictions).any():
                print(f"\n  WARNING: NaN detected in batch {batch_idx+1}/{len(loader)}")
                print(f"    Loss is NaN: {torch.isnan(loss).item()}")
                print(f"    Predictions have NaN: {torch.isnan(predictions).any().item()}")
                print(f"    Inputs have NaN: {torch.isnan(coords).any().item()}")
                print(f"    Targets have NaN: {torch.isnan(targets).any().item()}")
                
                # Debug specific values
                if torch.isnan(coords).any():
                    nan_mask = torch.isnan(coords).any(dim=1)
                    print(f"    NaN coords samples: {nan_mask.sum().item()}")
                    print(f"    First NaN coord: {coords[nan_mask][0] if nan_mask.any() else 'None'}")
                
                # Skip this batch
                continue
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)  # More efficient
            loss.backward()
            
            # Gradient clipping to prevent NaN
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Check for NaN gradients
            if torch.isnan(grad_norm):
                print(f"\n  WARNING: NaN gradient in batch {batch_idx+1}")
                continue
            
            optimizer.step()
            
            # Check for NaN in parameters after update and reset if needed
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    print(f"\n  WARNING: NaN in parameter {name}, resetting...")
                    with torch.no_grad():
                        if 'embeddings' in name:
                            torch.nn.init.uniform_(param, -0.1, 0.1)
                        else:
                            torch.nn.init.xavier_uniform_(param)
            
            # Metrics (only compute if no NaN)
            batch_loss = loss.item()
            batch_mape = self.compute_mape(predictions, targets)
            
            total_loss += batch_loss
            total_mape += batch_mape
            num_batches += 1
            
            # Update EMA
            if ema_loss is None:
                ema_loss = batch_loss
                ema_mape = batch_mape
            else:
                ema_loss = (1 - ema_alpha) * ema_loss + ema_alpha * batch_loss
                ema_mape = (1 - ema_alpha) * ema_mape + ema_alpha * batch_mape
            
            # Update progress bar every 10 batches for larger batch sizes
            if (batch_idx + 1) % 10 == 0:
                pbar.set_postfix({
                    'batch': f'{batch_idx+1}/{n_batches}',
                    'loss': f'{ema_loss:.4f}',
                    'mape': f'{ema_mape:.2f}%'
                })
            
            # Check for fractional epoch checkpoints (first epoch only)
            if (batch_idx + 1) in fractional_batch_checkpoints:
                epoch_str = fractional_batch_checkpoints[batch_idx + 1]
                fraction = float(f"0.{epoch_str.split('_')[1]}")
                # Save fractional epoch visualization
                self.save_fractional_epoch_predictions(epoch_num, fraction, model, epoch_str)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
        avg_mape = total_mape / num_batches if num_batches > 0 else float('nan')
        
        return avg_loss, avg_mape
    
    @torch.no_grad()
    def evaluate(self, model: nn.Module, loader: DataLoader, 
                criterion: nn.Module, subset_size: int = None) -> tuple:
        """Evaluate model on test set with direct GPU indexing."""
        model.eval()
        total_loss = 0
        total_mape = 0
        num_batches = 0
        num_valid_batches = 0
        
        # Get the dataset directly for GPU indexing
        # Handle Subset wrapper from random_split
        if hasattr(loader.dataset, 'dataset'):  # This is a Subset
            base_dataset = loader.dataset.dataset
            subset_indices = torch.tensor(loader.dataset.indices, device=self.device)
            n_total_samples = len(loader.dataset)
        else:
            base_dataset = loader.dataset
            subset_indices = None
            n_total_samples = len(loader.dataset)
        
        batch_size = loader.batch_size
        n_samples = min(n_total_samples, subset_size) if subset_size else n_total_samples
        
        # Fix: Handle case where n_samples < batch_size (was causing 0 batches)
        if n_samples == 0:
            return float('nan'), float('nan')
        
        # Calculate number of batches - at least 1 if we have any samples
        n_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division
        
        # Direct GPU iteration - no DataLoader overhead
        for batch_idx in range(n_batches):
            # Get batch indices
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_local_indices = torch.arange(start_idx, end_idx, device=self.device)
            
            # Map to actual dataset indices if using subset
            if subset_indices is not None:
                batch_indices = subset_indices[batch_local_indices]
            else:
                batch_indices = batch_local_indices
            
            # Direct GPU indexing
            coords = base_dataset.coords[batch_indices]
            targets = base_dataset.embeddings[batch_indices]
            
            predictions = model(coords)
            
            # Check for NaN in predictions
            if torch.isnan(predictions).any():
                print(f"  WARNING: NaN in test predictions at batch {batch_idx}")
                print(f"    Input coords range: [{coords.min():.2f}, {coords.max():.2f}]")
                print(f"    Predictions range: [{predictions.min():.2f}, {predictions.max():.2f}]")
                print(f"    NaN count: {torch.isnan(predictions).sum().item()}")
                continue
            
            loss = criterion(predictions, targets)
            
            # Check for NaN loss - skip this batch entirely
            if torch.isnan(loss):
                print(f"\n  ⚠️ NaN DETECTED in test batch {batch_idx}/{n_batches}")
                print(f"  Batch size: {len(coords)}, Loss value: {loss}")
                
                # Find which specific samples cause NaN
                nan_sample_indices = []
                for i in range(len(coords)):
                    single_coord = coords[i:i+1]
                    single_target = targets[i:i+1]
                    single_pred = model(single_coord)
                    single_loss = criterion(single_pred, single_target)
                    
                    if torch.isnan(single_pred).any() or torch.isnan(single_loss):
                        nan_sample_indices.append(i)
                        
                        if len(nan_sample_indices) <= 5:  # Detail first 5
                            coord = single_coord[0].cpu().numpy()
                            print(f"\n  Sample {i} (batch idx {batch_idx}, global idx {start_idx + i}) causes NaN:")
                            print(f"    Coordinates: lat={coord[0]:.6f}, lon={coord[1]:.6f}, "
                                  f"elev={coord[2]:.3f}m, time={coord[3]:.6f}")
                            
                            # Trace through model layers to find where NaN originates
                            with torch.no_grad():
                                # Earth4D encoding
                                earth4d_out = model.earth4d(single_coord)
                                has_nan_earth4d = torch.isnan(earth4d_out).any()
                                print(f"    Earth4D output: shape={earth4d_out.shape}, "
                                      f"has_nan={has_nan_earth4d}, "
                                      f"range=[{earth4d_out.min():.3f}, {earth4d_out.max():.3f}]")
                                
                                if not has_nan_earth4d:
                                    # Input normalization
                                    norm_out = model.input_norm(earth4d_out)
                                    has_nan_norm = torch.isnan(norm_out).any()
                                    print(f"    LayerNorm output: has_nan={has_nan_norm}, "
                                          f"range=[{norm_out.min():.3f}, {norm_out.max():.3f}]")
                                    
                                    if not has_nan_norm:
                                        # MLP layers
                                        x = norm_out
                                        for layer_idx, layer in enumerate(model.mlp):
                                            x = layer(x)
                                            if torch.isnan(x).any():
                                                print(f"    MLP layer {layer_idx} ({layer.__class__.__name__}) produces NaN")
                                                print(f"      Input range: [{norm_out.min():.3f}, {norm_out.max():.3f}]")
                                                print(f"      Output: {x}")
                                                break
                            
                            # Check target
                            target_val = single_target[0].cpu().numpy()
                            print(f"    Target: has_nan={torch.isnan(single_target).any()}, "
                                  f"first_5_dims={target_val[:5]}")
                            
                            # Check prediction
                            if not torch.isnan(single_pred).all():
                                pred_val = single_pred[0].cpu().numpy()
                                print(f"    Prediction: first_5_dims={pred_val[:5]}")
                
                print(f"\n  Summary: {len(nan_sample_indices)}/{len(coords)} samples in batch cause NaN")
                
                # Analyze patterns in NaN-causing samples
                if nan_sample_indices:
                    nan_coords = coords[nan_sample_indices].cpu().numpy()
                    print(f"  NaN sample statistics:")
                    print(f"    Latitude:  min={nan_coords[:,0].min():.2f}, max={nan_coords[:,0].max():.2f}, "
                          f"mean={nan_coords[:,0].mean():.2f}")
                    print(f"    Longitude: min={nan_coords[:,1].min():.2f}, max={nan_coords[:,1].max():.2f}, "
                          f"mean={nan_coords[:,1].mean():.2f}")
                    print(f"    Elevation: min={nan_coords[:,2].min():.1f}, max={nan_coords[:,2].max():.1f}, "
                          f"mean={nan_coords[:,2].mean():.1f}")
                    print(f"    Time:      min={nan_coords[:,3].min():.3f}, max={nan_coords[:,3].max():.3f}, "
                          f"mean={nan_coords[:,3].mean():.3f}")
                
                continue
            
            # Only accumulate valid losses and MAPEs
            total_loss += loss.item()
            num_batches += 1
            
            mape = self.compute_mape(predictions, targets)
            if not np.isnan(mape):
                total_mape += mape
                num_valid_batches += 1
        
        # Average only over valid batches
        avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
        avg_mape = total_mape / num_valid_batches if num_valid_batches > 0 else float('nan')
        
        return avg_loss, avg_mape
    
    def save_fractional_epoch_predictions(self, epoch_num: int, fraction: float, model: nn.Module, epoch_str: str = None):
        """Save predictions at fractional epoch points."""
        if self.fixed_test_batch is None:
            # Initialize if not already done
            self.save_fixed_test_predictions(epoch_num, initialize_only=True)
        
        if self.fixed_test_batch:
            # Get predictions
            model.eval()
            with torch.no_grad():
                predictions = model(self.fixed_test_coords)
            model.train()  # Switch back to training
            
            # Save predictions with fractional epoch naming
            if epoch_str is None:
                epoch_str = f"{epoch_num:02d}_{int(fraction*100):02d}"  # e.g., "01_25" for epoch 1.25
            pred_path = self.output_dir / 'predictions' / f'epoch_{epoch_str}.csv'
            pred_df = pd.DataFrame(predictions.cpu().numpy())
            pred_df.columns = [f'dim_{i}' for i in range(pred_df.shape[1])]
            
            # Compute MAPE
            abs_diff = torch.abs(self.fixed_test_targets - predictions)
            dimension_range = 2.0
            normalized_error = abs_diff / dimension_range
            mape_per_sample = (normalized_error.mean(dim=1) * 100).cpu().numpy()
            pred_df['mape'] = mape_per_sample
            
            pred_df.to_csv(pred_path, index=False)
            
            # Trigger visualization with correct epoch value (0.03, 0.17, etc)
            self.trigger_fractional_visualization(epoch_num, fraction, epoch_str)
            
            print(f"  Saved fractional epoch {epoch_num}.{int(fraction*100):02d} (MAPE: {mape_per_sample.mean():.2f}%)")
    
    def trigger_fractional_visualization(self, epoch_num: int, fraction: float, epoch_str: str):
        """Create visualization for fractional epoch."""
        try:
            from visualize_bay_area_comparison_v2 import create_dual_visualizations
            
            # Load data
            pred_path = self.output_dir / 'predictions' / f'epoch_{epoch_str}.csv'
            gt_path = self.output_dir / 'predictions' / 'ground_truth.csv'
            coords_path = self.output_dir / 'predictions' / 'test_coordinates.csv'
            
            if not pred_path.exists() or not gt_path.exists():
                return
            
            pred_df = pd.read_csv(pred_path)
            gt_df = pd.read_csv(gt_path)
            coords_df = pd.read_csv(coords_path)
            
            # Extract embeddings
            pred_embeddings = pred_df.iloc[:, :-1].values if 'mape' in pred_df.columns else pred_df.values
            gt_embeddings = gt_df.iloc[:, :-1].values if 'mape' in gt_df.columns else gt_df.values
            
            # Find autoencoder model if not already found
            if self.autoencoder_model_path is None:
                autoencoder_search_paths = [
                    Path('autoencoder_models/best_model.pt'),
                    Path('best_model.pt'),
                    Path('../autoencoder_models/best_model.pt'),
                    Path('./autoencoder_models/best_model.pt')
                ]
                
                for autoencoder_path in autoencoder_search_paths:
                    if autoencoder_path.exists():
                        self.autoencoder_model_path = str(autoencoder_path)
                        break
            
            if self.autoencoder_model_path is None:
                print(f"  Warning: Could not find autoencoder model for fractional visualization")
                return
            
            # Create visualization - will generate two files: _rgb.png and _error.png
            viz_base = self.output_dir / 'visualizations' / f'bay_area_epoch_{epoch_str}'
            viz_base.parent.mkdir(parents=True, exist_ok=True)
            
            create_dual_visualizations(
                gt_embeddings,
                pred_embeddings,
                coords_df,
                self.autoencoder_model_path,
                str(viz_base),
                epoch=fraction  # Display as 0.03, 0.17, 0.33, 0.67
            )
            
            print(f"  Saved fractional visualizations: {viz_base.name}_rgb.png and _error.png")
            
        except Exception as e:
            print(f"  Fractional visualization error: {e}")
    
    def save_fixed_test_predictions(self, epoch: int, n_samples: int = 10000, initialize_only: bool = False):
        """Save predictions for fixed test samples to track evolution."""
        if self.fixed_test_batch is None:
            # Initialize fixed test batch
            coords_list = []
            targets_list = []
            
            for batch_idx, (coords, targets) in enumerate(self.test_loader):
                coords_list.append(coords)
                targets_list.append(targets)
                
                if sum(len(c) for c in coords_list) >= n_samples:
                    break
            
            self.fixed_test_coords = torch.cat(coords_list)[:n_samples].to(self.device)
            self.fixed_test_targets = torch.cat(targets_list)[:n_samples].to(self.device)
            self.fixed_test_batch = True
            
            # Find Bay Area samples for fast visualization
            coords_np = self.fixed_test_coords.cpu().numpy()
            bay_area_mask = (
                (coords_np[:, 0] >= 36.9) & (coords_np[:, 0] <= 38.9) &  # Latitude
                (coords_np[:, 1] >= -123.5) & (coords_np[:, 1] <= -121.0)  # Longitude
            )
            self.bay_area_indices = np.where(bay_area_mask)[0]
            print(f"Found {len(self.bay_area_indices)} Bay Area samples for visualization")
            
            # Save ground truth and metadata
            gt_path = self.output_dir / 'predictions' / 'ground_truth.csv'
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            
            gt_df = pd.DataFrame(self.fixed_test_targets.cpu().numpy())
            gt_df.columns = [f'dim_{i}' for i in range(gt_df.shape[1])]
            gt_df.to_csv(gt_path, index=False)
            print(f"Saved ground truth to {gt_path}")
            
            # Save test coordinates for visualization
            coords_path = self.output_dir / 'predictions' / 'test_coordinates.csv'
            coords_np = self.fixed_test_coords.cpu().numpy()
            coords_df = pd.DataFrame({
                'latitude': coords_np[:, 0],
                'longitude': coords_np[:, 1],
                'elevation_m': coords_np[:, 2],
                'time_norm': coords_np[:, 3]
            })
            coords_df.to_csv(coords_path, index=False)
        
        # If only initializing, return early
        if initialize_only:
            return
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(self.fixed_test_coords)
        
        # Save predictions
        pred_path = self.output_dir / 'predictions' / f'epoch_{epoch:03d}.csv'
        pred_df = pd.DataFrame(predictions.cpu().numpy())
        pred_df.columns = [f'dim_{i}' for i in range(pred_df.shape[1])]
        
        # Compute MAPE per sample using dimension range
        abs_diff = torch.abs(self.fixed_test_targets - predictions)
        dimension_range = 2.0  # For [-1, 1] range
        normalized_error = abs_diff / dimension_range
        mape_per_sample = (normalized_error.mean(dim=1) * 100).cpu().numpy()
        pred_df['mape'] = mape_per_sample
        
        pred_df.to_csv(pred_path, index=False)
        
        # Store for visualization
        self.fixed_test_predictions.append({
            'epoch': epoch,
            'predictions': predictions.cpu().numpy(),
            'mape': mape_per_sample.mean()
        })
        
        # Trigger async visualization
        self.trigger_visualization(epoch)
    
    def trigger_visualization(self, epoch: int):
        """Trigger asynchronous visualization generation using autoencoder."""
        if self.autoencoder_model_path is None:
            # Check multiple possible autoencoder paths
            autoencoder_paths = [
                Path('./autoencoder_models/best_model.pt'),
                Path('./autoencoder_models/final_model.pt'),
                Path('./autoencoder_models/alphaearth_autoencoder.pth'),
                Path('../autoencoder_models/best_model.pt'),
                Path('/opt/ecodash/deepearth/encoders/xyzt/autoencoder_models/best_model.pt')
            ]
            
            for autoencoder_path in autoencoder_paths:
                if autoencoder_path.exists():
                    self.autoencoder_model_path = str(autoencoder_path)
                    print(f"Found autoencoder at {autoencoder_path}")
                    break
            else:
                print(f"Autoencoder not found in any of these paths:")
                for path in autoencoder_paths:
                    print(f"  {path}")
                print("Skipping visualization")
                return
        
        # Don't start new thread if previous one is still running
        if self.viz_thread is not None and self.viz_thread.is_alive():
            return
        
        # Run visualization in background thread
        def run_viz():
            try:
                # Use the new comparison visualization
                from visualize_bay_area_comparison_v2 import create_dual_visualizations
                
                # Load predictions and ground truth
                pred_path = self.output_dir / 'predictions' / f'epoch_{epoch:03d}.csv'
                gt_path = self.output_dir / 'predictions' / 'ground_truth.csv'
                coords_path = self.output_dir / 'predictions' / 'test_coordinates.csv'
                
                if not pred_path.exists() or not gt_path.exists():
                    return
                
                # Load data
                pred_df = pd.read_csv(pred_path)
                gt_df = pd.read_csv(gt_path)
                coords_df = pd.read_csv(coords_path)
                
                # Extract embeddings
                pred_embeddings = pred_df.iloc[:, :-1].values if 'mape' in pred_df.columns else pred_df.values
                gt_embeddings = gt_df.iloc[:, :-1].values if 'mape' in gt_df.columns else gt_df.values
                
                # Create dual visualizations - RGB and error
                viz_base = self.output_dir / 'visualizations' / f'bay_area_epoch_{epoch:03d}'
                viz_base.parent.mkdir(parents=True, exist_ok=True)
                
                create_dual_visualizations(
                    gt_embeddings,
                    pred_embeddings,
                    coords_df,
                    self.autoencoder_model_path,
                    str(viz_base),
                    epoch=epoch
                )
                
                print(f"  Saved Bay Area visualizations: {viz_base.name}_rgb.png and _error.png")
                
            except Exception as e:
                print(f"  Visualization error for epoch {epoch}: {e}")
        
        self.viz_thread = threading.Thread(target=run_viz, daemon=True)
        self.viz_thread.start()
    
    def print_epoch_insights(self, epoch: int, train_loss: float, train_mape: float,
                            test_loss: float, test_mape: float):
        """Print insights after each epoch."""
        print("\n" + "="*60)
        print(f"EPOCH {epoch} COMPLETE")
        print("="*60)
        
        # Format MAPE with appropriate precision
        if not np.isnan(train_mape):
            train_mape_str = f"{train_mape:.2f}%"
        else:
            train_mape_str = "N/A"
            
        if not np.isnan(test_mape):
            test_mape_str = f"{test_mape:.2f}%"
        else:
            test_mape_str = "N/A"
        
        print(f"Train: Loss={train_loss:.4f}, MAPE={train_mape_str}")
        print(f"Test:  Loss={test_loss:.4f}, MAPE={test_mape_str}")
        
        # Overfitting analysis
        if not np.isnan(train_mape) and not np.isnan(test_mape):
            overfit_ratio = test_mape / train_mape if train_mape > 0 else float('inf')
            if overfit_ratio > 2:
                print(f"⚠️  High overfitting detected (ratio: {overfit_ratio:.2f})")
            elif overfit_ratio > 1.5:
                print(f"⚠️  Moderate overfitting (ratio: {overfit_ratio:.2f})")
            else:
                print(f"✓ Good generalization (ratio: {overfit_ratio:.2f})")
        else:
            print("⚠️  Cannot compute generalization ratio due to NaN values")
        
        # Show sample predictions if we have fixed test batch
        if self.fixed_test_batch:
            with torch.no_grad():
                sample_pred = self.model(self.fixed_test_coords[:5])
                sample_target = self.fixed_test_targets[:5]
                
                print("\nSample predictions (first 5 test samples, first 10 dims):")
                print("Target:    ", sample_target[0, :10].cpu().numpy().round(3))
                print("Predicted: ", sample_pred[0, :10].cpu().numpy().round(3))
                sample_mape = self.compute_mape(sample_pred[0:1], sample_target[0:1])
                print(f"Sample MAPE: {sample_mape:.2f}%")
    
    def generate_final_visualization(self):
        """Generate final visualization using all test predictions."""
        try:
            # Get the final epoch number
            final_epoch = len(self.train_losses)
            
            # Use the new comparison visualization for the final output
            from visualize_bay_area_comparison_v2 import create_dual_visualizations
            
            # Paths to data
            pred_path = self.output_dir / 'predictions' / f'epoch_{final_epoch:03d}.csv'
            gt_path = self.output_dir / 'predictions' / 'ground_truth.csv'
            coords_path = self.output_dir / 'predictions' / 'test_coordinates.csv'
            
            if not pred_path.exists() or not gt_path.exists() or not coords_path.exists():
                print("Warning: Could not generate final visualization - missing data files")
                return
            
            # Load data
            pred_df = pd.read_csv(pred_path)
            gt_df = pd.read_csv(gt_path)
            coords_df = pd.read_csv(coords_path)
            
            # Extract embeddings
            pred_embeddings = pred_df.iloc[:, :-1].values if 'mape' in pred_df.columns else pred_df.values
            gt_embeddings = gt_df.iloc[:, :-1].values if 'mape' in gt_df.columns else gt_df.values
            
            # Find autoencoder model
            autoencoder_path = './autoencoder_models/best_model.pt'
            if not Path(autoencoder_path).exists():
                autoencoder_path = './autoencoder_models/final_model.pt'
                if not Path(autoencoder_path).exists():
                    print("Warning: Autoencoder model not found for visualization")
                    return
            
            # Create final dual visualizations
            final_viz_base = self.output_dir / 'final_bay_area'
            
            avg_mape = create_dual_visualizations(
                gt_embeddings,
                pred_embeddings,
                coords_df,
                autoencoder_path,
                str(final_viz_base),
                epoch=final_epoch
            )
            
            print(f"✓ Final visualizations saved:")
            print(f"  - {final_viz_base}_rgb.png")
            print(f"  - {final_viz_base}_error.png")
            print(f"  Final Bay Area MAPE: {avg_mape:.2f}%")
            
        except Exception as e:
            print(f"Error generating final visualization: {e}")
    
    def plot_training_curves(self):
        """Plot training and test curves."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss plot
        axes[0].plot(epochs, self.train_losses, 'b-', label='Train Loss')
        axes[0].plot(epochs, self.test_losses, 'r-', label='Test Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MSE Loss')
        axes[0].set_title('Training and Test Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAPE plot
        axes[1].plot(epochs, self.train_mapes, 'b-', label='Train MAPE')
        axes[1].plot(epochs, self.test_mapes, 'r-', label='Test MAPE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAPE (%)')
        axes[1].set_title('Training and Test MAPE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def train(self, epochs: int, learning_rate: float = 1e-3,
             test_every: int = 5, save_every: int = 10):
        """Main training loop."""
        # Setup optimizer and loss
        # Use smaller learning rate for Earth4D parameters
        earth4d_params = list(self.model.earth4d.parameters())
        other_params = [p for n, p in self.model.named_parameters() if 'earth4d' not in n]
        
        optimizer = optim.Adam([
            {'params': earth4d_params, 'lr': learning_rate * 0.1},  # 10x smaller lr for Earth4D
            {'params': other_params, 'lr': learning_rate}
        ], eps=1e-6)  # Larger epsilon for stability
        
        # Add learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        criterion = nn.MSELoss()
        
        # Print model info
        params = self.model.get_params_count()
        print("\n" + "="*60)
        print("DEEPEARTH MODEL CONFIGURATION")
        print("="*60)
        print(f"Earth4D params: {params['earth4d']:,} ({params['earth4d_mb']:.2f} MB)")
        print(f"MLP params: {params['mlp']:,} ({params['mlp_mb']:.2f} MB)")
        print(f"Total params: {params['total']:,} ({params['total_mb']:.2f} MB)")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Learning rate: {learning_rate}")
        print("="*60 + "\n")
        
        # Training loop
        for epoch in range(1, epochs + 1):
            epoch_start = datetime.now()
            
            # Train
            train_loss, train_mape = self.train_epoch(
                self.model, self.train_loader, optimizer, criterion, epoch_num=epoch
            )
            self.train_losses.append(train_loss)
            self.train_mapes.append(train_mape)
            
            # Quick test evaluation (subset)
            test_loss, test_mape = self.evaluate(
                self.model, self.test_loader, criterion, subset_size=10000
            )
            self.test_losses.append(test_loss)
            self.test_mapes.append(test_mape)
            
            # Update learning rate based on test loss
            if 'scheduler' in locals():
                scheduler.step(test_loss)
            
            # Print insights
            self.print_epoch_insights(epoch, train_loss, train_mape, test_loss, test_mape)
            
            # Full test evaluation every N epochs
            if epoch % test_every == 0:
                print(f"\nRunning full test set evaluation...")
                full_test_loss, full_test_mape = self.evaluate(
                    self.model, self.test_loader, criterion
                )
                print(f"Full Test: Loss={full_test_loss:.4f}, MAPE={full_test_mape:.2f}%")
            
            # Save predictions for fixed test samples
            self.save_fixed_test_predictions(epoch)
            
            # Save model checkpoint
            if epoch % save_every == 0:
                checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch:03d}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
            
            epoch_time = (datetime.now() - epoch_start).total_seconds()
            self.epoch_times.append(epoch_time)
            print(f"Epoch time: {epoch_time:.1f}s")
        
        # Final plots
        self.plot_training_curves()
        
        # Save final model
        final_path = self.output_dir / 'final_model.pt'
        torch.save(self.model.state_dict(), final_path)
        print(f"\n✓ Training complete. Final model saved to {final_path}")
        
        # Generate final full visualization
        print("\nGenerating final full-scale visualization...")
        self.generate_final_visualization()


def main():
    parser = argparse.ArgumentParser(description='Train Earth4D to AlphaEarth')
    parser.add_argument('--metadata', type=str, default=None,
                       help='Path to metadata CSV (auto-downloads if not specified)')
    parser.add_argument('--embeddings', type=str, default=None,
                       help='Path to embeddings .pt file (auto-downloads if not specified)')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Directory for storing/loading AlphaEarth data')
    parser.add_argument('--force-download', action='store_true',
                       help='Force re-download of AlphaEarth data')
    parser.add_argument('--output-dir', type=str, default='./earth4d_alphaearth_outputs')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=10000)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--train-split', type=float, default=0.95)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-nan-test', action='store_true', help='Skip NaN testing (faster but risky)')
    
    args = parser.parse_args()

    # Handle data paths
    if args.metadata is None or args.embeddings is None:
        # Auto-download AlphaEarth data if paths not specified
        data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
        metadata_path, embeddings_path = ensure_alphaearth_data(
            data_dir,
            force_download=args.force_download
        )

        # Use downloaded paths if not specified
        if args.metadata is None:
            args.metadata = str(metadata_path)
        if args.embeddings is None:
            args.embeddings = str(embeddings_path)
    else:
        # Validate provided paths exist
        if not Path(args.metadata).exists():
            print(f"Error: Metadata file not found: {args.metadata}")
            print("Run without --metadata to auto-download the AlphaEarth dataset")
            sys.exit(1)
        if not Path(args.embeddings).exists():
            print(f"Error: Embeddings file not found: {args.embeddings}")
            print("Run without --embeddings to auto-download the AlphaEarth dataset")
            sys.exit(1)

    # Enable GPU optimizations
    if args.device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("GPU optimizations enabled: CuDNN benchmark, TF32")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("\n" + "="*80)
    print("EARTH4D TO ALPHAEARTH TRAINING PIPELINE")
    print("="*80)
    
    # Create model first for NaN testing
    model = AlphaEarthPredictor(
        spatial_levels=24,  # Default planetary scale
        temporal_levels=19,  # Default for 200 years
        spatial_log2_hashmap_size=24,
        mlp_hidden_dims=[256, 256, 256],
        output_dim=64,
        dropout=0.1
    )
    model = model.to(args.device)
    
    # Load dataset - preload everything to GPU and test for NaN
    dataset = AlphaEarthGPUDataset(
        metadata_path=args.metadata,
        embeddings_path=args.embeddings,
        device=args.device,
        max_samples=args.max_samples,
        time_range=(1900, 2100),
        random_seed=args.seed,
        test_for_nan=(not args.skip_nan_test),
        model=model
    )
    
    # Split dataset
    train_size = int(args.train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_dataset):,} samples")
    print(f"  Test: {len(test_dataset):,} samples")
    
    # Create dataloaders - must use num_workers=0 for GPU tensors
    # GPU tensors cannot be accessed from worker processes
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Must be 0 for GPU-resident data
        pin_memory=False,  # Data already on GPU
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Must be 0 for GPU-resident data
        pin_memory=False,  # Data already on GPU
        drop_last=False
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=args.device,
        output_dir=args.output_dir
    )
    
    # Train
    trainer.train(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        test_every=5,
        save_every=10
    )
    
    # Create training progression video if visualizations exist
    viz_dir = Path(args.output_dir) / 'visualizations'
    if viz_dir.exists() and list(viz_dir.glob('*epoch*.png')):
        print("\n" + "="*80)
        print("Creating training progression video...")
        print("="*80)
        try:
            import subprocess
            # Create two videos - one for RGB, one for error
            rgb_video = viz_dir / 'bay_area_rgb_progression.mp4'
            error_video = viz_dir / 'bay_area_error_progression.mp4'
            
            # RGB video
            cmd_rgb = [
                'python3', 'create_training_video.py',
                '--input-dir', str(viz_dir),
                '--output', str(rgb_video),
                '--pattern', 'bay_area_*_rgb.png',
                '--fps', '2.0'
            ]
            
            # Error video  
            cmd_error = [
                'python3', 'create_training_video.py',
                '--input-dir', str(viz_dir),
                '--output', str(error_video),
                '--pattern', 'bay_area_*_error.png',
                '--fps', '2.0'
            ]
            # Run both video creations
            result_rgb = subprocess.run(cmd_rgb, capture_output=True, text=True, cwd=Path(__file__).parent)
            result_error = subprocess.run(cmd_error, capture_output=True, text=True, cwd=Path(__file__).parent)
            
            videos_created = []
            if result_rgb.returncode == 0 and rgb_video.exists():
                size_mb = rgb_video.stat().st_size / (1024 * 1024)
                videos_created.append((rgb_video, size_mb))
            
            if result_error.returncode == 0 and error_video.exists():
                size_mb = error_video.stat().st_size / (1024 * 1024)
                videos_created.append((error_video, size_mb))
            
            if videos_created:
                print(f"✓ Training videos created successfully!")
                for video_path, size_mb in videos_created:
                    print(f"  - {video_path.name} ({size_mb:.1f} MB)")
            else:
                print(f"Warning: Could not create videos")
                if result_rgb.stderr:
                    print(f"  RGB Error: {result_rgb.stderr[:200]}")
                if result_error.stderr:
                    print(f"  Error video Error: {result_error.stderr[:200]}")
        except Exception as e:
            print(f"Warning: Could not create video: {e}")
    
    print("\n✓ Pipeline complete!")

if __name__ == "__main__":
    main()
