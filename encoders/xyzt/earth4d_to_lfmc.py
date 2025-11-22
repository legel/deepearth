#!/usr/bin/env python3
"""
Earth4D LFMC - Species-Aware Version
=====================================
Supports both learnable embeddings and pre-trained BioCLIP 2 embeddings.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
from pathlib import Path
import argparse
import sys
import os
import time
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from earth4d import Earth4D

# LFMC constants following allenai/lfmc approach
MAX_LFMC_VALUE = 302  # 99.9th percentile of the LFMC values

# AI2 official data URL
AI2_LFMC_CSV_URL = "https://raw.githubusercontent.com/allenai/lfmc/refs/heads/main/data/labels/lfmc_data_conus.csv"
DEFAULT_AI2_CSV_PATH = "data/labels/lfmc_data_conus.csv"

# AI2 default fold configuration (70-15-15 split)
# Uses seed 42 for reproducibility, matching allenai/lfmc
DEFAULT_NUM_FOLDS = 100
_rng = random.Random(42)
DEFAULT_VALIDATION_FOLDS, DEFAULT_TEST_FOLDS = map(
    frozenset, (lambda lst: (lst[:15], lst[15:]))(lst := _rng.sample(range(DEFAULT_NUM_FOLDS), 30))
)


def download_ai2_csv(output_path: str = DEFAULT_AI2_CSV_PATH):
    """Download AI2's official LFMC CSV if not present."""
    import urllib.request
    output_path = Path(output_path)

    if output_path.exists():
        print(f"LFMC CSV already exists at {output_path}", flush=True)
        return output_path

    print(f"Downloading LFMC CSV from {AI2_LFMC_CSV_URL}...", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(AI2_LFMC_CSV_URL, output_path)
        print(f"Downloaded to {output_path}", flush=True)
        return output_path
    except Exception as e:
        print(f"Error downloading LFMC CSV: {e}", flush=True)
        raise


def assign_ai2_folds(df: pd.DataFrame, id_column: str = 'sorting_id', num_folds: int = DEFAULT_NUM_FOLDS) -> pd.DataFrame:
    """
    Assign folds using AI2's hash-based deterministic approach.

    This matches their approach in lfmc/lfmc/core/splits.py:
    - Uses SHA256 hash of sorting_id
    - Converts to deterministic fold assignment
    - Ensures reproducibility across runs
    """
    import hashlib

    def create_prob(value: int) -> float:
        hash_val = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
        return int(hash_val[:8], 16) / 0xFFFFFFFF

    probs = df[id_column].apply(create_prob)
    df['fold'] = (probs * num_folds).astype(int)
    return df


def assign_splits_from_folds(
    df: pd.DataFrame,
    validation_folds: frozenset[int],
    test_folds: frozenset[int],
) -> pd.DataFrame:
    """Assign train/val/test splits from fold assignments."""
    def map_split(row: pd.Series) -> str:
        if row['fold'] in validation_folds:
            return 'validation'
        elif row['fold'] in test_folds:
            return 'test'
        else:
            return 'train'

    df['split'] = df.apply(map_split, axis=1)
    return df


class ExponentialMovingAverage:
    """Track exponential moving average of metrics."""

    def __init__(self, alpha=0.1):
        """Initialize EMA with smoothing factor alpha (0 < alpha <= 1)."""
        self.alpha = alpha
        self.ema = None

    def update(self, value):
        """Update EMA with new value."""
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema
        return self.ema

    def get(self):
        """Get current EMA value."""
        return self.ema if self.ema is not None else 0.0


class MetricsEMA:
    """Track EMAs for all metrics."""

    def __init__(self, alpha=0.1):
        self.emas = defaultdict(lambda: ExponentialMovingAverage(alpha))
        self.sample_predictions = defaultdict(list)  # Store sample predictions for visualization

    def update(self, metrics_dict):
        """Update all EMAs with new metrics."""
        ema_dict = {}
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                ema_dict[f"{key}_ema"] = self.emas[key].update(value)
        return ema_dict

    def get_all(self):
        """Get all current EMA values."""
        return {key: ema.get() for key, ema in self.emas.items()}


class FullyGPUDataset:
    """
    Everything on GPU from the start, with species encoding.

    Loads AI2's official LFMC CSV and uses their fold-based train/test splits.
    """

    def __init__(self, data_path: str | None = None, device: str = 'cuda', use_ai2_splits: bool = True):
        """
        Load dataset with AI2 official splits.

        Args:
            data_path: Path to CSV file. If None, downloads AI2 official CSV.
            device: Device for GPU tensors
            use_ai2_splits: If True, use AI2's hash-based fold splits (70-15-15)
        """
        # Download AI2 CSV if no path provided
        if data_path is None:
            data_path = download_ai2_csv()
            print(f"Using official LFMC CSV", flush=True)

        # Load CSV
        print(f"Loading data from {data_path}...", flush=True)
        df = pd.read_csv(data_path)

        # Check if this is AI2 format or custom format
        if 'sorting_id' in df.columns:
            # AI2 format
            print("Detected official CSV format", flush=True)
            df = df.rename(columns={
                'latitude': 'lat',
                'longitude': 'lon',
                'elevation': 'elev',
                'lfmc_value': 'lfmc',
                'species_collected': 'species'
            })

            # Parse sampling_date to date components
            df['sampling_date'] = pd.to_datetime(df['sampling_date'])
            df['date_str'] = df['sampling_date'].dt.strftime('%Y%m%d')
            df['time_str'] = '0000'  # Placeholder

            # Apply AI2 fold-based splits if requested
            if use_ai2_splits:
                print("Applying hash-based fold splits...", flush=True)
                df = assign_ai2_folds(df, 'sorting_id', DEFAULT_NUM_FOLDS)
                df = assign_splits_from_folds(df, DEFAULT_VALIDATION_FOLDS, DEFAULT_TEST_FOLDS)

                # Store split info
                self.has_splits = True
                print(f"  Train folds: {DEFAULT_NUM_FOLDS - len(DEFAULT_VALIDATION_FOLDS) - len(DEFAULT_TEST_FOLDS)}", flush=True)
                print(f"  Validation folds: {len(DEFAULT_VALIDATION_FOLDS)}", flush=True)
                print(f"  Test folds: {len(DEFAULT_TEST_FOLDS)}", flush=True)
            else:
                self.has_splits = False
        else:
            # Custom format - assume columns already named correctly
            print("Detected custom CSV format", flush=True)
            df.columns = ['lat', 'lon', 'elev', 'date_str', 'time_str', 'lfmc', 'species']
            self.has_splits = False

        # Filter - clip at 99.9th percentile (302) following allenai/lfmc approach
        df = df[(df['lfmc'] >= 0) & (df['lfmc'] <= MAX_LFMC_VALUE) &
                df['lat'].notna() & df['lon'].notna() &
                df['elev'].notna()].copy()

        n_total = len(df)
        print(f"Loaded {n_total:,} samples after filtering", flush=True)

        # Show split breakdown if available
        if self.has_splits and 'split' in df.columns:
            for split_name in ['train', 'validation', 'test']:
                n_split = len(df[df['split'] == split_name])
                pct = 100 * n_split / n_total
                print(f"  {split_name.capitalize()}: {n_split:,} samples ({pct:.1f}%)", flush=True)

        n = len(df)
        print(f"Total samples to process: {n:,}", flush=True)

        # Parse dates to floats (for GPU sorting later)
        date_floats = np.zeros(n)
        for i, d in enumerate(df['date_str'].values):
            d = str(d)
            if len(d) == 8:
                year = int(d[:4])
                month = int(d[4:6])
                day = int(d[6:8])
                date_floats[i] = year + (month - 1) / 12.0 + day / 365.0
            else:
                date_floats[i] = 2020.0

        # Normalize time to [0, 1]
        time_norm = (date_floats - 2015) / 10.0
        time_norm = np.clip(time_norm, 0, 1)

        # SPECIES ENCODING
        # Create species vocabulary
        unique_species = df['species'].unique()
        self.species_to_idx = {species: idx for idx, species in enumerate(unique_species)}
        self.idx_to_species = {idx: species for species, idx in self.species_to_idx.items()}
        self.n_species = len(unique_species)

        # Convert species to indices
        species_indices = np.array([self.species_to_idx[s] for s in df['species'].values])

        print(f"\nSpecies Statistics:", flush=True)
        print(f"  Unique species: {self.n_species}", flush=True)

        # Show top 10 most common species
        species_counts = defaultdict(int)
        for s in df['species'].values:
            species_counts[s] += 1
        top_species = sorted(species_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"  Top 10 species:", flush=True)
        for i, (species, count) in enumerate(top_species):
            print(f"    {i+1:2d}. {species}: {count:,} samples ({100*count/n:.1f}%)", flush=True)

        # DEGENERACY ANALYSIS WITH SPECIES
        # Create unique coordinate keys (lat, lon, elev, time)
        coord_keys = []
        coord_to_indices = defaultdict(list)
        coord_species_lfmc = defaultdict(lambda: defaultdict(list))

        for i in range(n):
            # Round to reasonable precision to handle floating point issues
            key = (
                round(df['lat'].values[i], 6),
                round(df['lon'].values[i], 6),
                round(df['elev'].values[i], 2),
                round(time_norm[i], 6)
            )
            coord_keys.append(key)
            coord_to_indices[key].append(i)
            coord_species_lfmc[key][df['species'].values[i]].append(df['lfmc'].values[i])

        # Create degeneracy flags (True if coordinate has multiple LFMC values for DIFFERENT species)
        is_degenerate = np.zeros(n, dtype=bool)
        degenerate_groups = []

        for key, indices in coord_to_indices.items():
            if len(indices) > 1:
                # Check if we have different species at this coordinate
                species_at_coord = df['species'].values[indices]
                unique_species_at_coord = np.unique(species_at_coord)

                if len(unique_species_at_coord) > 1:
                    # This is a true multi-species degeneracy
                    is_degenerate[indices] = True
                    degenerate_groups.append({
                        'coord': key,
                        'indices': indices,
                        'lfmc_values': df['lfmc'].values[indices].tolist(),
                        'species': species_at_coord.tolist()
                    })

        # Calculate statistics
        n_unique_coords = len(coord_to_indices)
        n_degenerate_coords = len(degenerate_groups)
        n_degenerate_samples = np.sum(is_degenerate)

        print(f"\nDegeneracy Analysis:", flush=True)
        print(f"  Total samples: {n:,}", flush=True)
        print(f"  Unique spatiotemporal coordinates: {n_unique_coords:,}", flush=True)
        print(f"  Multi-species degenerate coordinates: {n_degenerate_coords:,} ({100*n_degenerate_coords/n_unique_coords:.1f}% of unique coords)", flush=True)
        print(f"  Multi-species degenerate samples: {n_degenerate_samples:,} ({100*n_degenerate_samples/n:.1f}% of all samples)", flush=True)

        # Show a few examples of degeneracies
        if degenerate_groups:
            print(f"\n  Example multi-species degeneracies (showing first 3):", flush=True)
            for i, group in enumerate(degenerate_groups[:3]):
                lat, lon, elev, t = group['coord']
                print(f"    {i+1}. Lat={lat:.4f}, Lon={lon:.4f}, Elev={elev:.1f}m, Time={t:.4f}", flush=True)
                # Group by species to show variation
                species_lfmc = defaultdict(list)
                for lfmc, species in zip(group['lfmc_values'], group['species']):
                    species_lfmc[species].append(lfmc)
                for species, lfmcs in list(species_lfmc.items())[:5]:
                    avg_lfmc = np.mean(lfmcs)
                    if len(lfmcs) > 1:
                        print(f"       → Species={species}: LFMC={lfmcs} (avg={avg_lfmc:.0f}%)", flush=True)
                    else:
                        print(f"       → Species={species}: LFMC={lfmcs[0]:.0f}%", flush=True)

        # FINAL CPU->GPU transfer
        self.coords = torch.tensor(
            np.column_stack([df['lat'].values, df['lon'].values,
                           df['elev'].values, time_norm]),
            dtype=torch.float32, device=device
        )
        # Normalize targets to [0, 1] following allenai/lfmc approach
        normalized_lfmc = df['lfmc'].values / MAX_LFMC_VALUE
        self.targets = torch.tensor(normalized_lfmc, dtype=torch.float32, device=device)
        self.species_idx = torch.tensor(species_indices, dtype=torch.long, device=device)

        # Degeneracy flag on GPU
        self.is_degenerate = torch.tensor(is_degenerate, dtype=torch.bool, device=device)

        # For temporal split - convert dates to GPU tensor
        self.date_values = torch.tensor(date_floats, dtype=torch.float32, device=device)

        # Store raw dataframe columns for split creation
        self.df = df

        # Create split indices on GPU if splits are available
        if self.has_splits and 'split' in df.columns:
            # Create indices for train (train+validation) and test
            train_mask = df['split'].isin(['train', 'validation']).values
            test_mask = (df['split'] == 'test').values

            self.train_indices = torch.tensor(np.where(train_mask)[0], dtype=torch.long, device=device)
            self.test_indices = torch.tensor(np.where(test_mask)[0], dtype=torch.long, device=device)

            print(f"\nSplit Indices Created:", flush=True)
            print(f"  Train+Validation: {len(self.train_indices):,} samples", flush=True)
            print(f"  Test: {len(self.test_indices):,} samples", flush=True)

            # Report degeneracy breakdown per split
            print(f"\nDegeneracy breakdown by split:", flush=True)
            for name, indices in [('Train+Val', self.train_indices), ('Test', self.test_indices)]:
                n_degen = self.is_degenerate[indices].sum().item()
                n_unique = len(indices) - n_degen
                print(f"  {name:10s}: {n_unique:5d} unique, {n_degen:5d} multi-species degenerate ({100*n_degen/len(indices):.1f}%)", flush=True)
        else:
            self.train_indices = None
            self.test_indices = None

        self.n = n
        self.device = device
        self.n_degenerate_samples = n_degenerate_samples
        self.n_unique_samples = n - n_degenerate_samples

        print(f"\nGPU dataset ready: {n:,} samples", flush=True)


class SpeciesAwareLFMCModel(nn.Module):
    """LFMC model with learnable species embeddings."""

    def __init__(self, n_species, species_dim=32):
        super().__init__()

        self.earth4d = Earth4D(
            verbose=True
        )

        earth4d_dim = self.earth4d.get_output_dim()

        # Learnable species embeddings
        self.species_embeddings = nn.Embedding(n_species, species_dim)
        nn.init.normal_(self.species_embeddings.weight, mean=0.0, std=0.1)
        print(f"  Using learnable species embeddings: ({n_species}, {species_dim})", flush=True)

        # MLP that takes concatenated Earth4D features and species embedding
        input_dim = earth4d_dim + species_dim
        print(f"  MLP input dimension: {input_dim} (Earth4D: {earth4d_dim} + Species: {species_dim})", flush=True)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Constrain output to [0, 1] to match normalized targets
        )

        # Initialize Earth4D parameters
        with torch.no_grad():
            for p in self.earth4d.parameters():
                if p.dim() > 1:
                    nn.init.uniform_(p, -0.1, 0.1)

        self.n_species = n_species
        self.species_dim = species_dim

    def forward(self, coords, species_idx):
        # Get Earth4D spatiotemporal features
        earth4d_features = self.earth4d(coords)

        # Get species embeddings
        species_features = self.species_embeddings(species_idx)

        # Concatenate features
        combined_features = torch.cat([earth4d_features, species_features], dim=-1)

        # Predict LFMC
        return self.mlp(combined_features).squeeze(-1)


def get_ai2_splits(dataset):
    """
    Return AI2 train/test splits from the dataset.

    Uses the hash-based fold assignment (70% train, 15% validation, 15% test).
    For training, we combine train+validation following AI2's approach.
    """
    if not dataset.has_splits or dataset.train_indices is None:
        raise ValueError("Dataset does not have AI2 splits. Ensure use_ai2_splits=True when creating dataset.")

    # Check species coverage
    train_species = set(dataset.df.iloc[dataset.train_indices.cpu().numpy()]['species'].unique())
    test_species = set(dataset.df.iloc[dataset.test_indices.cpu().numpy()]['species'].unique())

    print(f"\nSpecies Coverage:", flush=True)
    print(f"  Train species: {len(train_species)}", flush=True)
    print(f"  Test species: {len(test_species)} ({100*len(test_species & train_species)/len(test_species):.1f}% in train)", flush=True)

    # Report species NOT in training set
    test_novel = test_species - train_species
    if test_novel:
        print(f"\n  WARNING: {len(test_novel)} species in test NOT in training:", flush=True)
        for s in list(test_novel)[:5]:
            print(f"    - {s}", flush=True)
        if len(test_novel) > 5:
            print(f"    ... and {len(test_novel) - 5} more", flush=True)

    return {
        'train': dataset.train_indices,
        'test': dataset.test_indices
    }


def compute_metrics_gpu(preds, targets, is_degenerate=None):
    """
    Compute metrics on GPU using absolute errors (LFMC percentage points).

    Following allenai/lfmc approach: denormalize predictions and targets
    before computing metrics.

    Metrics align with standard LFMC literature (Rao et al. 2020, Miller et al. 2023):
    - RMSE: Root Mean Squared Error in LFMC percentage points
    - MAE: Mean Absolute Error in LFMC percentage points
    - R²: Coefficient of determination
    """
    def calc_metrics(p, t):
        if len(p) == 0 or len(t) == 0:
            # If no samples, return zeros
            return {
                'mse': 0.0,
                'rmse': 0.0,
                'mae': 0.0,
                'r2': 0.0,
                'n_samples': 0
            }

        # Denormalize to LFMC percentage points (following allenai/lfmc)
        p_denorm = p * MAX_LFMC_VALUE
        t_denorm = t * MAX_LFMC_VALUE

        # Compute errors in LFMC percentage point space
        errors = p_denorm - t_denorm  # Signed errors
        abs_errors = torch.abs(errors)  # Absolute errors

        # Absolute metrics (in LFMC percentage points)
        mse = (errors ** 2).mean()
        rmse = torch.sqrt(mse)
        mae = abs_errors.mean()

        # R² score (coefficient of determination)
        # R² = 1 - (SS_res / SS_tot)
        # where SS_res = sum of squared residuals, SS_tot = total sum of squares
        ss_res = ((p_denorm - t_denorm) ** 2).sum()
        ss_tot = ((t_denorm - t_denorm.mean()) ** 2).sum()

        if ss_tot > 0:
            r2 = 1 - (ss_res / ss_tot)
        else:
            r2 = torch.tensor(0.0, device=p.device)

        return {
            'mse': mse.item(),
            'rmse': rmse.item(),
            'mae': mae.item(),
            'r2': r2.item(),
            'n_samples': len(p)
        }

    # Overall metrics
    overall = calc_metrics(preds, targets)

    # Split by degeneracy if provided
    if is_degenerate is not None:
        unique_mask = ~is_degenerate
        degen_mask = is_degenerate

        # Default empty metrics dict
        empty_metrics = {
            'mse': 0, 'rmse': 0, 'mae': 0, 'r2': 0, 'n_samples': 0
        }

        unique_metrics = calc_metrics(preds[unique_mask], targets[unique_mask]) if unique_mask.sum() > 0 else empty_metrics
        degen_metrics = calc_metrics(preds[degen_mask], targets[degen_mask]) if degen_mask.sum() > 0 else empty_metrics

        return overall, unique_metrics, degen_metrics

    return overall


def train_epoch_gpu(model, dataset, indices, optimizer, batch_size=20000):
    """Ultra-fast training - all GPU."""
    model.train()
    n = len(indices)

    # Shuffle ON GPU
    perm = torch.randperm(n, device=indices.device)
    indices = indices[perm]

    criterion = nn.MSELoss()

    # Accumulate for metrics
    all_preds = []
    all_targets = []
    all_degens = []

    # Process batches
    n_batches = (n + batch_size - 1) // batch_size

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]

        # Pure GPU ops
        coords = dataset.coords[batch_idx]
        targets = dataset.targets[batch_idx]
        species = dataset.species_idx[batch_idx]

        preds = model(coords, species)

        # FIX #2: Use compute_loss() with entropy regularization if available
        if hasattr(model.earth4d, 'compute_loss'):
            loss_dict = model.earth4d.compute_loss(
                preds, targets,
                enable_probe_entropy_loss=True,
                probe_entropy_weight=0.5
            )
            loss = loss_dict['_total_loss_tensor']
        else:
            loss = criterion(preds, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # FIX #1: Update probe indices after optimizer step (for learned probing)
        if hasattr(model.earth4d.encoder.xyz_encoder, 'update_probe_indices'):
            model.earth4d.encoder.xyz_encoder.update_probe_indices()
            model.earth4d.encoder.xyt_encoder.update_probe_indices()
            model.earth4d.encoder.yzt_encoder.update_probe_indices()
            model.earth4d.encoder.xzt_encoder.update_probe_indices()

        # Store for metrics
        all_preds.append(preds.detach())
        all_targets.append(targets)
        all_degens.append(dataset.is_degenerate[batch_idx])

    # Compute metrics on full epoch predictions
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_degens = torch.cat(all_degens)

    overall, unique, degen = compute_metrics_gpu(all_preds, all_targets, all_degens)

    return overall, unique, degen


@torch.no_grad()
def evaluate_split(model, dataset, indices):
    """Fast evaluation of a single split."""
    model.eval()

    empty_metrics = {
        'mse': 0, 'rmse': 0, 'mae': 0, 'median_ae': 0, 'median_rmse': 0, 'r2': 0, 'n_samples': 0
    }

    if len(indices) == 0:
        return empty_metrics, empty_metrics, empty_metrics, [], [], []

    coords = dataset.coords[indices]
    targets = dataset.targets[indices]
    species = dataset.species_idx[indices]
    degens = dataset.is_degenerate[indices]

    # Single forward pass
    preds = model(coords, species)

    # Metrics
    overall, unique, degen = compute_metrics_gpu(preds, targets, degens)

    # Get 5 sample predictions for monitoring (mix of unique and degenerate)
    unique_idx = torch.where(~degens)[0]
    degen_idx = torch.where(degens)[0]

    sample_preds = []
    sample_trues = []
    sample_types = []

    # Get up to 3 unique and 2 degenerate samples
    if len(unique_idx) > 0:
        n_unique_samples = min(3, len(unique_idx))
        for i in range(n_unique_samples):
            idx = unique_idx[i * len(unique_idx) // n_unique_samples]
            sample_preds.append(preds[idx].item())
            sample_trues.append(targets[idx].item())
            sample_types.append('U')

    if len(degen_idx) > 0:
        n_degen_samples = min(2, len(degen_idx))
        for i in range(n_degen_samples):
            idx = degen_idx[i * len(degen_idx) // n_degen_samples]
            sample_preds.append(preds[idx].item())
            sample_trues.append(targets[idx].item())
            sample_types.append('D')

    return overall, unique, degen, sample_trues, sample_preds, sample_types


def print_predictions_table(test_gt, test_pred, test_types):
    """Print predictions in a clean table format with degeneracy indicators for test set."""
    print("  ┌───────────────────────────────────────────────────────────────┐", flush=True)
    print("  │  Sample Test Predictions (UNIQUE vs MULTI-species)          │", flush=True)
    print("  ├───────────────────────────────────────────────────────────────┤", flush=True)

    for i in range(min(5, len(test_gt))):
        # Denormalize for display (multiply by MAX_LFMC_VALUE)
        gt_val = test_gt[i] * MAX_LFMC_VALUE
        pred_val = test_pred[i] * MAX_LFMC_VALUE
        err = abs(gt_val - pred_val)
        tag = "UNIQUE" if test_types[i] == 'U' else "MULTI"

        line = f"  │  {i+1}. {tag:6s}: {gt_val:6.1f}% → {pred_val:6.1f}% (Δ{err:5.1f}pp)  │"
        print(line, flush=True)

    print("  └───────────────────────────────────────────────────────────────┘", flush=True)
    print("    (UNIQUE=Single species, MULTI=Multiple species at same location)", flush=True)




def run_training_session(args, run_name=""):
    """Run a single training session."""
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = 'cuda'
    print(f"Random seed: {args.seed}", flush=True)

    # Create output directory with run name suffix
    output_suffix = f"_{run_name}" if run_name else ""
    output_dir = Path(args.output_dir + output_suffix)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80, flush=True)
    print(f"EARTH4D LFMC BENCHMARK", flush=True)
    print(f"Using learnable {args.species_dim}D species embeddings", flush=True)
    print("="*80, flush=True)

    # Load dataset with AI2 splits
    dataset = FullyGPUDataset(args.data_path, device, use_ai2_splits=True)
    splits = get_ai2_splits(dataset)

    # Model with learnable species embeddings
    model = SpeciesAwareLFMCModel(
        dataset.n_species,
        species_dim=args.species_dim
    ).to(device)

    # Count parameters
    earth4d_params = sum(p.numel() for p in model.earth4d.parameters())
    species_params = sum(p.numel() for p in model.species_embeddings.parameters())
    mlp_params = sum(p.numel() for p in model.mlp.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel architecture:", flush=True)
    print(f"  Earth4D parameters: {earth4d_params:,}", flush=True)
    print(f"  Species embedding parameters: {species_params:,} (Learnable, {dataset.n_species} species × {args.species_dim} dims)", flush=True)
    print(f"  MLP parameters: {mlp_params:,}", flush=True)
    print(f"  Total parameters: {total_params:,}", flush=True)
    print(f"  Trainable parameters: {trainable_params:,}", flush=True)

    # FIX #3: Use differential learning rates if learned probing is enabled
    if hasattr(model.earth4d.encoder.xyz_encoder, 'index_logits'):
        index_lr_multiplier = 10.0
        optimizer_params = [
            # Earth4D embedding parameters (base LR)
            {'params': model.earth4d.encoder.xyz_encoder.embeddings, 'lr': args.lr},
            {'params': model.earth4d.encoder.xyt_encoder.embeddings, 'lr': args.lr},
            {'params': model.earth4d.encoder.yzt_encoder.embeddings, 'lr': args.lr},
            {'params': model.earth4d.encoder.xzt_encoder.embeddings, 'lr': args.lr},
            # Species embeddings (base LR)
            {'params': model.species_embeddings.parameters(), 'lr': args.lr},
            # MLP parameters (base LR)
            {'params': model.mlp.parameters(), 'lr': args.lr},
            # Index logits (10× HIGHER LR - critical for learned probing!)
            {'params': model.earth4d.encoder.xyz_encoder.index_logits, 'lr': args.lr * index_lr_multiplier},
            {'params': model.earth4d.encoder.xyt_encoder.index_logits, 'lr': args.lr * index_lr_multiplier},
            {'params': model.earth4d.encoder.yzt_encoder.index_logits, 'lr': args.lr * index_lr_multiplier},
            {'params': model.earth4d.encoder.xzt_encoder.index_logits, 'lr': args.lr * index_lr_multiplier},
        ]
        optimizer = optim.AdamW(optimizer_params, weight_decay=0.001)
        print(f"\n✓ Using {index_lr_multiplier}× higher LR for index_logits: {args.lr * index_lr_multiplier:.6f}", flush=True)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)

    # Tracking metrics
    metrics_history = []
    metrics_ema = MetricsEMA(alpha=0.1)  # EMA with smoothing factor 0.1

    print("\n" + "="*80, flush=True)
    print("Training with species embeddings (UNIQUE=Single species, MULTI=Multi-species):", flush=True)
    print("-"*80, flush=True)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        trn_overall, trn_unique, trn_degen = train_epoch_gpu(
            model, dataset, splits['train'], optimizer, args.batch_size
        )

        # Evaluate test split
        test_overall, test_unique, test_degen, test_gt, test_pred, test_types = evaluate_split(
            model, dataset, splits['test']
        )

        dt = time.time() - t0

        # Update EMAs
        current_metrics = {
            'epoch': epoch,
            'time': dt,
            # Train overall metrics
            'train_rmse': trn_overall['rmse'],
            'train_mae': trn_overall['mae'],
            'train_r2': trn_overall['r2'],
            # Train unique metrics
            'train_unique_mae': trn_unique['mae'],
            'train_unique_r2': trn_unique['r2'],
            # Train degenerate metrics
            'train_degen_mae': trn_degen['mae'],
            'train_degen_r2': trn_degen['r2'],
            # Test metrics
            'test_rmse': test_overall['rmse'],
            'test_mae': test_overall['mae'],
            'test_r2': test_overall['r2'],
            'test_unique_mae': test_unique['mae'],
            'test_unique_r2': test_unique['r2'],
            'test_degen_mae': test_degen['mae'],
            'test_degen_r2': test_degen['r2']
        }

        # Update EMAs and store metrics
        ema_metrics = metrics_ema.update(current_metrics)
        current_metrics.update(ema_metrics)
        metrics_history.append(current_metrics)

        # Store sample predictions for EMA tracking
        if epoch == 1:
            metrics_ema.sample_predictions['test_gt'] = test_gt
        metrics_ema.sample_predictions[f'test_pred_{epoch}'] = test_pred

        # Print clean formatted metrics with absolute errors (percentage points) and R²
        print(f"\nEPOCH {epoch:3d} ({dt:.1f}s)", flush=True)
        print(f"  TRAIN ALL: [RMSE: {trn_overall['rmse']:5.1f}pp, MAE: {trn_overall['mae']:5.1f}pp, R²: {trn_overall['r2']:.3f}]", flush=True)
        print(f"        UNIQUE: MAE={trn_unique['mae']:5.1f}pp, R²={trn_unique['r2']:.3f}  |  MULTI: MAE={trn_degen['mae']:5.1f}pp, R²={trn_degen['r2']:.3f}", flush=True)

        print(f"\n  TEST: [RMSE: {test_overall['rmse']:5.1f}pp, MAE: {test_overall['mae']:5.1f}pp, R²: {test_overall['r2']:.3f}]", flush=True)
        print(f"        UNIQUE: MAE={test_unique['mae']:5.1f}pp, R²={test_unique['r2']:.3f}  |  MULTI: MAE={test_degen['mae']:5.1f}pp, R²={test_degen['r2']:.3f}", flush=True)

        # Show predictions table every epoch
        print_predictions_table(test_gt, test_pred, test_types)

        # LR decay -  reduction per epoch 
        for g in optimizer.param_groups:
            g['lr'] *= 0.9995
        print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.6f}", flush=True)

    print("="*80, flush=True)

    # Save final model
    final_model_path = output_dir / 'final_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_species': dataset.n_species,
        'species_dim': args.species_dim,
        'species_to_idx': dataset.species_to_idx,
        'idx_to_species': dataset.idx_to_species
    }, final_model_path)
    print(f"\nFinal model saved to: {final_model_path}", flush=True)

    # Save metrics history to CSV
    metrics_df = pd.DataFrame(metrics_history)
    metrics_path = output_dir / f'training_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}", flush=True)

    # Print final summary with absolute errors (percentage points) and R²
    final = metrics_history[-1]
    print(f"\nFINAL RESULTS (Epoch {args.epochs}) - Absolute Errors (LFMC Percentage Points):", flush=True)
    print(f"  Comparison with AllenAI LFMC (Rao et al. 2020): RMSE=18.91pp, MAE=12.58pp, R²=0.72", flush=True)
    print(f"\n  Overall Performance:", flush=True)
    print(f"    Training:  RMSE={final['train_rmse']:.1f}pp, MAE={final['train_mae']:.1f}pp, R²={final['train_r2']:.3f}", flush=True)
    print(f"    Test:      RMSE={final['test_rmse']:.1f}pp, MAE={final['test_mae']:.1f}pp, R²={final['test_r2']:.3f}", flush=True)

    print(f"\n  Unique Species Locations Only:", flush=True)
    print(f"    Training:  MAE={final['train_unique_mae']:.1f}pp, R²={final['train_unique_r2']:.3f}", flush=True)
    print(f"    Test:      MAE={final['test_unique_mae']:.1f}pp, R²={final['test_unique_r2']:.3f}", flush=True)

    print(f"\n  Multi-Species Locations Only:", flush=True)
    print(f"    Training:  MAE={final['train_degen_mae']:.1f}pp, R²={final['train_degen_r2']:.3f}", flush=True)
    print(f"    Test:      MAE={final['test_degen_mae']:.1f}pp, R²={final['test_degen_r2']:.3f}", flush=True)

    print("\nTraining complete!", flush=True)

    # Create visualizations using final predictions
    print("\nGenerating visualizations...", flush=True)

    # Get final predictions for all test sets
    with torch.no_grad():
        model.eval()

        # Collect predictions from AI2 test set
        if len(splits['test']) > 0:
            coords = dataset.coords[splits['test']]
            targets = dataset.targets[splits['test']]
            species = dataset.species_idx[splits['test']]
            test_preds = model(coords, species)

            test_preds_np = test_preds.cpu().numpy()
            test_gts_np = targets.cpu().numpy()
            test_indices = splits['test']
            train_count = len(splits['train'])

            # Create temporal visualization with AI2 test set
            # Package in dict format for compatibility with visualization function
            all_preds = {'test': test_preds_np}
            all_gts = {'test': test_gts_np}
            all_indices = {'test': test_indices}

            temp_errors = create_temporal_visualization(
                dataset,
                all_preds,
                all_gts,
                all_indices,
                output_dir,
                epoch=args.epochs,
                total_epochs=args.epochs,
                train_samples=train_count
            )

            # Create geospatial visualization with AI2 test set
            grid_errors, grid_counts = create_geospatial_visualization(
                dataset,
                test_preds_np,
                test_gts_np,
                test_indices,
                output_dir,
                epoch=args.epochs,
                train_samples=train_count,
                spatial_indices=None  # No spatial-specific visualization
            )

            # Create combined scientific figure (geospatial + temporal)
            print("\nCreating combined scientific figure...", flush=True)
            create_combined_scientific_figure(
                dataset,
                test_preds_np,
                test_gts_np,
                test_indices,
                all_preds,
                all_gts,
                all_indices,
                output_dir,
                epoch=args.epochs,
                train_samples=train_count
            )

    print(f"Visualizations saved to {output_dir}", flush=True)

    # Export test predictions to CSV
    print("\nExporting test predictions to CSV...", flush=True)
    csv_path = export_test_predictions_csv(
        dataset,
        test_preds_np,
        test_gts_np,
        test_indices,
        output_dir
    )

    # Create error distribution histogram
    print("\nCreating error distribution histogram...", flush=True)
    create_error_histogram(csv_path, output_dir)

    # Return final metrics for comparison
    return final


def export_test_predictions_csv(dataset, test_predictions, test_ground_truth, test_indices, output_dir):
    """
    Export test predictions and ground truth to CSV for analysis.

    Args:
        dataset: Dataset object
        test_predictions: Predictions for test samples (normalized [0, 1])
        test_ground_truth: Ground truth for test samples (normalized [0, 1])
        test_indices: Indices of test samples
        output_dir: Output directory
    """
    # Denormalize to LFMC percentage points
    preds_denorm = test_predictions * MAX_LFMC_VALUE
    gts_denorm = test_ground_truth * MAX_LFMC_VALUE

    # Extract coordinates and metadata
    coords = dataset.coords[test_indices].cpu().numpy()
    lats = coords[:, 0]
    lons = coords[:, 1]
    elevs = coords[:, 2]
    times_norm = coords[:, 3]

    # Get species information
    species_indices = dataset.species_idx[test_indices].cpu().numpy()
    species_names = [dataset.idx_to_species[idx] for idx in species_indices]

    # Get degeneracy flags
    is_degenerate = dataset.is_degenerate[test_indices].cpu().numpy()

    # Calculate errors
    errors = preds_denorm - gts_denorm
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2

    # Create DataFrame
    export_df = pd.DataFrame({
        'latitude': lats,
        'longitude': lons,
        'elevation_m': elevs,
        'time_normalized': times_norm,
        'species': species_names,
        'is_multi_species_location': is_degenerate,
        'ground_truth_lfmc_pct': gts_denorm,
        'predicted_lfmc_pct': preds_denorm,
        'error_pp': errors,
        'absolute_error_pp': abs_errors,
        'squared_error': squared_errors
    })

    # Sort by absolute error (descending) to see worst predictions first
    export_df = export_df.sort_values('absolute_error_pp', ascending=False)

    # Save to CSV
    csv_path = output_dir / 'test_predictions.csv'
    export_df.to_csv(csv_path, index=False, float_format='%.4f')

    print(f"  Exported {len(export_df):,} test predictions to: {csv_path}", flush=True)

    # Print summary statistics
    print(f"\n  CSV Summary Statistics:", flush=True)
    print(f"    LFMC range: [{gts_denorm.min():.1f}%, {gts_denorm.max():.1f}%]", flush=True)
    print(f"    Absolute Error percentiles:", flush=True)
    for p in [50, 75, 90, 95, 99]:
        val = np.percentile(abs_errors, p)
        print(f"      {p:2d}th: {val:5.1f}pp", flush=True)

    # Show worst predictions
    worst_10 = sorted(abs_errors, reverse=True)[:10]
    print(f"    Worst 10 absolute errors (pp):", flush=True)
    print(f"      {', '.join([f'{e:.1f}' for e in worst_10])}", flush=True)

    # Compute overall metrics
    mean_squared_error = np.mean(squared_errors)
    mean_rmse = np.sqrt(mean_squared_error)
    mean_mae = np.mean(abs_errors)

    print(f"\n  Overall Metrics on Test Set:", flush=True)
    print(f"    RMSE: {mean_rmse:.1f}pp", flush=True)
    print(f"    MAE:  {mean_mae:.1f}pp", flush=True)

    return csv_path


def create_error_histogram(csv_path, output_dir):
    """
    Create histogram showing distribution of absolute errors across test dataset.

    Args:
        csv_path: Path to test_predictions.csv
        output_dir: Output directory for saving histogram
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Read the CSV
    df = pd.read_csv(csv_path)

    # Get absolute errors and predictions/ground truth
    abs_errors_original = df['absolute_error_pp'].values
    predictions = df['predicted_lfmc_pct'].values
    ground_truth = df['ground_truth_lfmc_pct'].values

    # Compute R²
    ss_res = np.sum((ground_truth - predictions) ** 2)
    ss_tot = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Cap errors at 100pp for histogram display
    abs_errors = np.clip(abs_errors_original, 0, 100)
    n_over_100 = np.sum(abs_errors_original > 100)

    # Compute statistics (on original uncapped data)
    mean_error = np.mean(abs_errors_original)
    median_error = np.median(abs_errors_original)
    rmse = np.sqrt(np.mean(df['squared_error'].values))
    mae = np.mean(abs_errors_original)
    max_error = np.max(abs_errors_original)
    min_error = np.min(abs_errors_original)

    # Create histogram with 100 bins from 0-100pp
    n_bins = 100
    bins = np.linspace(0, 100, n_bins + 1)

    # Create figure (height = 1/5 of width for compressed vertical display)
    fig, ax = plt.subplots(1, 1, figsize=(16, 3.2))

    # Plot histogram with Turbo colormap (matching geospatial plot)
    counts, bins_out, patches = ax.hist(abs_errors, bins=bins, edgecolor='black', linewidth=0.5)

    # Color bars using Turbo colormap (same as geospatial plot: 0-50pp scale)
    norm = plt.Normalize(vmin=0, vmax=50)
    cmap = cm.get_cmap('turbo')

    for i, patch in enumerate(patches):
        # Get the center value of this bin
        bin_center = (bins[i] + bins[i+1]) / 2
        # Clip to 50pp max (same as geospatial)
        bin_value = min(bin_center, 50.0)
        # Set color based on normalized value
        patch.set_facecolor(cmap(norm(bin_value)))

    # Calculate MAE percentiles for subtitle
    mae_25th = np.percentile(abs_errors_original, 25)
    mae_median = np.median(abs_errors_original)
    mae_75th = np.percentile(abs_errors_original, 75)

    # Labels and title (matching font sizes from other plots)
    ax.set_xlabel('Absolute Error (percentage points)', fontsize=12)
    ax.set_ylabel('Number of Test Samples', fontsize=12)

    # Title with LFMC definition for readers unfamiliar with LFMC
    title_text = 'Distribution of Live Fuel Moisture Content (LFMC) Prediction Errors on Test Set\n'
    ax.text(0.5, 1.08, title_text.strip(), transform=ax.transAxes,
            fontsize=13, ha='center', va='bottom')

    # Second line with metrics (smaller font, italicized)
    subtitle_text = f'LFMC% = (wet mass - dry mass) / dry mass × 100  |  '
    subtitle_text += f'Total Samples: {len(abs_errors_original):,} | RMSE: {rmse:.1f}pp | '
    subtitle_text += f'MAE: {mae:.1f}pp (25th: {mae_25th:.1f}pp, Median: {mae_median:.1f}pp, 75th: {mae_75th:.1f}pp) | '
    subtitle_text += f'R²: {r2:.3f}'
    if n_over_100 > 0:
        subtitle_text += f' | {n_over_100} samples >100pp'
    ax.text(0.5, 1.02, subtitle_text, transform=ax.transAxes,
            fontsize=9, ha='center', va='bottom', style='italic')

    # Set x-axis limits and tick marks every 5 percentage points
    ax.set_xlim(0, 100)
    ax.set_xticks([i for i in range(0, 101, 5)])

    # No gridlines - clean white background
    ax.grid(False)

    # Format y-axis with comma separators
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    # Save metrics to JSON file
    metrics_dict = {
        'summary_metrics': {
            'rmse_pp': float(rmse),
            'mae_pp': float(mae),
            'r2': float(r2),
            'total_samples': int(len(abs_errors_original)),
            'samples_over_100pp': int(n_over_100)
        },
        'error_distribution': {
            'min_error_pp': float(min_error),
            'percentile_25_pp': float(np.percentile(abs_errors_original, 25)),
            'median_pp': float(median_error),
            'mean_pp': float(mean_error),
            'percentile_75_pp': float(np.percentile(abs_errors_original, 75)),
            'percentile_95_pp': float(np.percentile(abs_errors_original, 95)),
            'max_error_pp': float(max_error)
        },
        'mae_distribution': {
            'mae_25th_percentile_pp': float(mae_25th),
            'mae_median_pp': float(mae_median),
            'mae_75th_percentile_pp': float(mae_75th)
        },
        'lfmc_definition': {
            'formula': 'LFMC% = (wet mass - dry mass) / dry mass × 100',
            'error_calculation': 'Percentage Points (pp) = |Predicted LFMC% - Ground Truth LFMC%|',
            'histogram_note': 'Histogram capped at 100pp for display'
        }
    }

    import json
    metrics_json_path = output_dir / 'error_histogram_metrics.json'
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"  Metrics saved to: {metrics_json_path}", flush=True)

    # Tight layout
    plt.tight_layout()

    # Save figure
    histogram_path = output_dir / 'error_distribution_histogram.png'
    plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Error histogram saved to: {histogram_path}", flush=True)
    print(f"  Error range: [{min_error:.1f}pp, {max_error:.1f}pp]", flush=True)
    print(f"  Errors >100pp: {n_over_100} samples (binned at 100+)", flush=True)
    print(f"  Number of bins: {n_bins} (0-100pp)", flush=True)


def create_geospatial_visualization(dataset, test_predictions, test_ground_truth, test_indices, output_dir, epoch="final", train_samples=None, spatial_indices=None):
    """Create geospatial visualization of LFMC prediction errors across CONUS.

    Args:
        dataset: Dataset object
        test_predictions: Predictions for test samples
        test_ground_truth: Ground truth for test samples
        test_indices: Indices of test samples
        output_dir: Output directory
        epoch: Epoch number
        train_samples: Number of training samples
        spatial_indices: Indices of spatial holdout samples (for marking regions)
    """
    # Import geopandas for US boundaries (required)
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError("geopandas is required for US state boundaries. Install with: pip install geopandas")

    # Extract coordinates for test samples
    coords = dataset.coords[test_indices].cpu().numpy()
    lats = coords[:, 0]
    lons = coords[:, 1]

    # Denormalize predictions and targets to LFMC percentage points
    test_predictions_denorm = test_predictions * MAX_LFMC_VALUE
    test_ground_truth_denorm = test_ground_truth * MAX_LFMC_VALUE

    # Calculate errors in LFMC percentage points
    errors = np.abs(test_predictions_denorm - test_ground_truth_denorm)

    # Define CONUS boundaries
    lon_min, lon_max = -125, -66
    lat_min, lat_max = 24, 50

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    # Load and plot US shapefile
    shapefile_path = Path(SCRIPT_DIR) / 'shapefiles' / 'cb_2018_us_state_20m.shp'
    if not shapefile_path.exists():
        raise FileNotFoundError(f"Shapefile not found at {shapefile_path}. Please ensure shapefiles are downloaded.")

    # Read shapefile
    states = gpd.read_file(shapefile_path)
    # Filter to continental US (exclude Alaska, Hawaii, territories)
    states_conus = states[
        ~states['NAME'].isin(['Alaska', 'Hawaii', 'Puerto Rico',
                             'United States Virgin Islands', 'Guam',
                             'American Samoa', 'Northern Mariana Islands'])
    ]
    # Plot state boundaries with light grey, thin lines
    states_conus.boundary.plot(ax=ax, color='#CCCCCC', linewidth=0.4, alpha=0.8)
    # Plot overall US boundary slightly darker
    states_conus.dissolve().boundary.plot(ax=ax, color='#AAAAAA', linewidth=0.5, alpha=0.9)

    # Create 50x50 km grid (approximately 0.45 degrees)
    grid_size = 0.45
    lon_bins = np.arange(lon_min, lon_max, grid_size)
    lat_bins = np.arange(lat_min, lat_max, grid_size)

    # Bin the data
    grid_errors = np.zeros((len(lat_bins)-1, len(lon_bins)-1))
    grid_counts = np.zeros((len(lat_bins)-1, len(lon_bins)-1))

    for i in range(len(lats)):
        if lon_min <= lons[i] <= lon_max and lat_min <= lats[i] <= lat_max:
            lon_idx = np.digitize(lons[i], lon_bins) - 1
            lat_idx = np.digitize(lats[i], lat_bins) - 1
            if 0 <= lon_idx < len(lon_bins)-1 and 0 <= lat_idx < len(lat_bins)-1:
                grid_errors[lat_idx, lon_idx] += errors[i]
                grid_counts[lat_idx, lon_idx] += 1

    # Calculate average errors per bin
    with np.errstate(divide='ignore', invalid='ignore'):
        grid_avg_errors = grid_errors / grid_counts
        grid_avg_errors[grid_counts == 0] = np.nan

    # Plot binned data
    valid_bins = ~np.isnan(grid_avg_errors)
    if np.any(valid_bins):
        # Get non-zero counts for sizing
        nonzero_counts = grid_counts[grid_counts > 0]
        if len(nonzero_counts) > 0:
            # Log scale for sizes
            log_counts = np.log1p(grid_counts)  # log(counts + 1) to handle zeros
            log_counts[grid_counts == 0] = 0

            # Normalize sizes
            min_log = np.min(log_counts[log_counts > 0]) if np.any(log_counts > 0) else 1
            max_log = np.max(log_counts)

            # Size range: ensure max size doesn't exceed bin size (50km grid)
            # Scale sizes to prevent overlap - max diameter should be ~80% of grid spacing
            size_min, size_max = 10, 68  # Reduced by 15% to prevent overlap

            # Plot each grid cell
            for i in range(len(lat_bins)-1):
                for j in range(len(lon_bins)-1):
                    if not np.isnan(grid_avg_errors[i, j]) and grid_counts[i, j] > 0:
                        # Calculate position (center of bin)
                        lon_center = (lon_bins[j] + lon_bins[j+1]) / 2
                        lat_center = (lat_bins[i] + lat_bins[i+1]) / 2

                        # Calculate size based on log count
                        if max_log > min_log:
                            size_norm = (log_counts[i, j] - min_log) / (max_log - min_log)
                            size = size_min + (size_max - size_min) * size_norm
                        else:
                            size = size_max

                        # Color based on error (TURBO colormap) - clip to 50pp max
                        error_value = min(grid_avg_errors[i, j], 50.0)  # Clip to 50pp

                        scatter = ax.scatter(lon_center, lat_center, s=size,
                                           c=[error_value], cmap='turbo',
                                           vmin=0, vmax=50,  # Fixed scale 0-50pp
                                           alpha=1.0, edgecolors='black', linewidth=0.5)  # Alpha=1.0

    # Add colorbar with fixed 0-50 scale
    if 'scatter' in locals():
        # Create a proper colorbar with fixed scale
        import matplotlib.cm as cm
        norm = plt.Normalize(vmin=0, vmax=50)
        sm = plt.cm.ScalarMappable(cmap='turbo', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Average LFMC Error (pp)')
        cbar.set_ticks([0, 10, 20, 30, 40, 50])
        cbar.set_ticklabels(['0', '10', '20', '30', '40', '50+'])  # Add + to indicate clipping

    # Create size legend
    # Calculate example sizes for legend
    if len(nonzero_counts) > 0:
        count_min = int(np.min(nonzero_counts))
        count_max = int(np.max(nonzero_counts))
        count_25 = int(np.percentile(nonzero_counts, 25))
        count_75 = int(np.percentile(nonzero_counts, 75))

        # Create legend handles
        legend_sizes = [count_min, count_25, count_75, count_max]
        legend_labels = [f'{c:,} samples' for c in legend_sizes]

        # Calculate corresponding marker sizes
        legend_handles = []
        for count in legend_sizes:
            log_count = np.log1p(count)
            if max_log > min_log:
                size_norm = (log_count - min_log) / (max_log - min_log)
                size = size_min + (size_max - size_min) * size_norm
            else:
                size = size_max
            # Create grey circle for legend
            legend_handles.append(plt.scatter([], [], s=size, c='grey', alpha=1.0,
                                             edgecolors='black', linewidth=0.5))

        # Add size legend in upper right
        size_legend = ax.legend(legend_handles, legend_labels,
                              title='Sample Count', loc='upper right',
                              frameon=True, fancybox=True, shadow=True)
        ax.add_artist(size_legend)  # Add the legend separately to not interfere with other legends

    # Set labels and title
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    # Determine title text with sample counts
    test_count = len(test_indices) if hasattr(test_indices, '__len__') else 0
    if train_samples and str(epoch).isdigit():
        title_text = f'Test Performance on {test_count:,} Samples after training for {epoch} epochs on {train_samples:,} samples'
    elif str(epoch).isdigit():
        title_text = f'Test Performance on {test_count:,} Samples after {epoch} epochs'
    else:
        title_text = f'Test Performance on {test_count:,} Samples'

    ax.set_title(f'Earth4D LFMC Prediction Error - Geospatial Distribution\n'
                 f'100km × 100km Grid Bins\n'
                 f'{title_text}', fontsize=14)

    # Set CONUS boundaries
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    # Remove grid (no gridlines)
    ax.grid(False)

    # Add overall statistics in a neat bubble
    # Calculate mean and median RMSE and MAE across entire dataset (absolute errors in pp)
    if len(test_predictions) > 0 and len(test_ground_truth) > 0:
        # Denormalize if not already done
        if test_predictions.max() <= 1.0:
            # Values are normalized, denormalize
            preds_denorm = test_predictions * MAX_LFMC_VALUE
            gts_denorm = test_ground_truth * MAX_LFMC_VALUE
        else:
            # Already denormalized
            preds_denorm = test_predictions
            gts_denorm = test_ground_truth

        # Compute absolute errors
        errors = preds_denorm - gts_denorm
        abs_errors = np.abs(errors)

        # Compute absolute metrics (in LFMC percentage points)
        rmse = np.sqrt(np.mean(errors ** 2))
        mae = np.mean(abs_errors)

        # Compute R²
        ss_res = np.sum((preds_denorm - gts_denorm) ** 2)
        ss_tot = np.sum((gts_denorm - np.mean(gts_denorm)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Display statistics in top-left corner
        stats_text = (f'Test Dataset:\n'
                     f'RMSE: {rmse:.1f}pp\n'
                     f'MAE: {mae:.1f}pp\n'
                     f'R²: {r2:.3f}')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               ha='left', va='top', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9,
                        edgecolor='black', linewidth=0.5))

    # Save figure
    plt.tight_layout()
    plt.savefig(output_dir / f'geospatial_error_map_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
    plt.close()

    return grid_avg_errors, grid_counts


def create_temporal_visualization(dataset, all_predictions, all_ground_truth, all_indices, output_dir, epoch="final", total_epochs=None, train_samples=None):
    """Create temporal visualization of LFMC predictions vs ground truth with weekly binning.

    Args:
        dataset: The dataset object
        all_predictions: Dictionary with keys 'temporal', 'spatial', 'random' containing predictions
        all_ground_truth: Dictionary with keys 'temporal', 'spatial', 'random' containing ground truth
        all_indices: Dictionary with keys 'temporal', 'spatial', 'random' containing indices
        output_dir: Output directory for saving plots
        epoch: Epoch number or 'final'
        total_epochs: Total number of epochs trained (for displaying in title)
        train_samples: Number of training samples
    """
    from datetime import datetime, timedelta
    import matplotlib.dates as mdates

    # Combine all test sets
    all_preds = []
    all_gts = []
    all_times = []
    all_sources = []  # Track which test set each sample comes from

    for split_name in ['test']:  # AI2 test split only
        if split_name in all_predictions and len(all_predictions[split_name]) > 0:
            preds = all_predictions[split_name]
            gts = all_ground_truth[split_name]
            indices = all_indices[split_name]

            # Extract temporal information
            coords = dataset.coords[indices].cpu().numpy()
            times = coords[:, 3]  # Normalized time [0, 1]

            all_preds.extend(preds)
            all_gts.extend(gts)
            all_times.extend(times)
            all_sources.extend([split_name] * len(preds))

    if len(all_preds) == 0:
        print("No test data available for temporal visualization", flush=True)
        return None

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_gts = np.array(all_gts)
    all_times = np.array(all_times)

    # Denormalize to LFMC percentage points
    all_preds = all_preds * MAX_LFMC_VALUE
    all_gts = all_gts * MAX_LFMC_VALUE

    # Convert normalized times to actual dates
    # Assuming 2015-2025 range, convert to datetime objects
    base_date = datetime(2015, 1, 1)
    end_date = datetime(2025, 1, 1)
    total_days = (end_date - base_date).days

    dates = [base_date + timedelta(days=int(t * total_days)) for t in all_times]

    # Create monthly bins
    # Find the range of dates
    min_date = min(dates)
    max_date = max(dates)

    # Align to start of month
    start_month = datetime(min_date.year, min_date.month, 1)
    # Move to next month for end
    if max_date.month == 12:
        end_month = datetime(max_date.year + 1, 1, 1)
    else:
        end_month = datetime(max_date.year, max_date.month + 1, 1)

    # Generate monthly bins
    monthly_bins = []
    current_month = start_month
    while current_month <= end_month:
        monthly_bins.append(current_month)
        # Move to next month
        if current_month.month == 12:
            current_month = datetime(current_month.year + 1, 1, 1)
        else:
            current_month = datetime(current_month.year, current_month.month + 1, 1)

    # Bin the data by month
    monthly_data = defaultdict(lambda: {'preds': [], 'gts': [], 'sources': []})

    for i, date in enumerate(dates):
        # Find which month this belongs to
        month_start = datetime(date.year, date.month, 1)
        monthly_data[month_start]['preds'].append(all_preds[i])
        monthly_data[month_start]['gts'].append(all_gts[i])
        monthly_data[month_start]['sources'].append(all_sources[i])

    # Prepare data for plotting with side-by-side violin plots
    months = sorted(monthly_data.keys())

    # Filter months with at least 5 samples for violin plots
    months_filtered = []
    month_predictions = []
    month_ground_truths = []
    month_pred_medians = []
    month_gt_medians = []
    month_positions = []  # For x-axis positioning

    for month in months:
        preds = np.array(monthly_data[month]['preds'])
        gts = np.array(monthly_data[month]['gts'])

        if len(preds) >= 5:  # Minimum 5 samples for violin plot
            months_filtered.append(month)
            month_predictions.append(preds)
            month_ground_truths.append(gts)
            month_pred_medians.append(np.median(preds))
            month_gt_medians.append(np.median(gts))
            month_positions.append(mdates.date2num(month))

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))

    if len(months_filtered) > 0:
        # Offset for side-by-side plotting (in days) - reduced for tighter grouping
        offset = 5  # Reduced from 7 to 5 days for tighter pairing

        # Lists to store positions for median dots
        gt_positions_offset = []
        pred_positions_offset = []

        for i, pos in enumerate(month_positions):
            # Ground truth violin (left side, ember red)
            gt_parts = ax.violinplot([month_ground_truths[i]],
                                     positions=[pos - offset],
                                     widths=8.4,  # Reduced to 70% of 12 (was 12, now 8.4)
                                     showmeans=False, showmedians=False, showextrema=False)

            # Style ground truth violins (ember/magma red)
            for pc in gt_parts['bodies']:
                pc.set_facecolor('#B22222')  # Ember red
                pc.set_edgecolor('#8B0000')  # Darker red edge
                pc.set_alpha(0.6)
                pc.set_linewidth(0.5)

            # Prediction violin (right side, navy blue)
            pred_parts = ax.violinplot([month_predictions[i]],
                                       positions=[pos + offset],
                                       widths=8.4,  # Reduced to 70% of 12 (was 12, now 8.4)
                                       showmeans=False, showmedians=False, showextrema=False)

            # Style prediction violins (navy blue)
            for pc in pred_parts['bodies']:
                pc.set_facecolor('#000080')  # Navy blue
                pc.set_edgecolor('#000050')  # Darker blue edge
                pc.set_alpha(0.6)
                pc.set_linewidth(0.5)

            # Store positions for median dots
            gt_positions_offset.append(pos - offset)
            pred_positions_offset.append(pos + offset)

        # Plot median ground truth as ember red dots
        for pos, gt_median in zip(gt_positions_offset, month_gt_medians):
            ax.scatter(pos, gt_median, c='#B22222', s=40, zorder=6,
                      edgecolors='#8B0000', linewidth=0.5)

        # Plot median predictions as navy blue dots
        for pos, pred_median in zip(pred_positions_offset, month_pred_medians):
            ax.scatter(pos, pred_median, c='#000080', s=40, zorder=6,
                      edgecolors='#000050', linewidth=0.5)

        # Add thin lines connecting median values for continuity
        ax.plot(gt_positions_offset, month_gt_medians, color='#B22222', linewidth=0.5,
                alpha=0.4, zorder=2, linestyle='-')
        ax.plot(pred_positions_offset, month_pred_medians, color='#000080', linewidth=0.5,
                alpha=0.4, zorder=2, linestyle='-')

    # Convert x-axis back to dates
    ax.xaxis_date()

    # Add legend with all 4 plot types
    gt_violin = mpatches.Patch(color='#B22222', alpha=0.6,
                               label='Ground Truth Distribution')
    gt_median = plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='#B22222', markeredgecolor='#8B0000',
                          markersize=6, label='Ground Truth Median')
    pred_violin = mpatches.Patch(color='#000080', alpha=0.6,
                                 label='Prediction Distribution')
    pred_median = plt.Line2D([0], [0], marker='o', color='w',
                            markerfacecolor='#000080', markeredgecolor='#000050',
                            markersize=6, label='Prediction Median')

    ax.legend(handles=[gt_violin, gt_median, pred_violin, pred_median],
             loc='upper right', frameon=True, fancybox=True, shadow=True)

    # Format x-axis with dates - simple year labels only
    ax.xaxis.set_major_locator(mdates.YearLocator())  # Major ticks every year
    ax.xaxis.set_minor_locator(mdates.MonthLocator())  # Minor ticks every month
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Set labels and title
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('LFMC (%)', fontsize=12)
    # Count total test samples
    total_test_samples = sum(len(indices) if hasattr(indices, '__len__') else 0
                            for indices in all_indices.values())

    # Determine title text with sample counts
    if train_samples and str(epoch).isdigit():
        title_text = f'Test Performance on {total_test_samples:,} Samples after training for {epoch} epochs on {train_samples:,} samples'
    elif str(epoch).isdigit():
        title_text = f'Test Performance on {total_test_samples:,} Samples after {epoch} epochs'
    else:
        title_text = f'Test Performance on {total_test_samples:,} Samples'

    ax.set_title(f'Earth4D LFMC Predictions - Monthly Temporal Evolution (Test Set)\n'
                 f'Ground Truth and Prediction Distributions\n'
                 f'{title_text}', fontsize=14)

    # Remove grid (no gridlines as requested)
    ax.grid(False)

    # Set y-axis limits based on all values
    if len(months_filtered) > 0:
        all_values = np.concatenate(month_predictions + month_ground_truths)
        y_min = max(0, np.min(all_values) - 20)
        y_max = min(600, np.max(all_values) + 20)
        ax.set_ylim(y_min, y_max)

    # Add overall statistics - calculate from all predictions (absolute errors in pp)
    if len(months_filtered) > 0:
        all_predictions = np.concatenate(month_predictions)
        all_ground_truths = np.concatenate(month_ground_truths)

        # Compute absolute errors
        errors = all_predictions - all_ground_truths
        abs_errors = np.abs(errors)

        # Compute absolute metrics (in LFMC percentage points)
        rmse = np.sqrt(np.mean(errors ** 2))
        mae = np.mean(abs_errors)

        # Compute R²
        ss_res = np.sum((all_predictions - all_ground_truths) ** 2)
        ss_tot = np.sum((all_ground_truths - np.mean(all_ground_truths)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        stats_text = (f'Test Dataset:\n'
                     f'RMSE: {rmse:.1f}pp\n'
                     f'MAE: {mae:.1f}pp\n'
                     f'R²: {r2:.3f}')
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
                ha='left', va='top', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9,
                         edgecolor='black', linewidth=0.5))

    # Save figure
    plt.tight_layout()
    plt.savefig(output_dir / f'temporal_predictions_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Return absolute errors if we have valid data
    if len(months_filtered) > 0:
        all_predictions_full = np.concatenate(month_predictions)
        all_ground_truths_full = np.concatenate(month_ground_truths)
        abs_errors_full = np.abs(all_predictions_full - all_ground_truths_full)
        return abs_errors_full
    return None


def create_combined_scientific_figure(dataset, test_predictions, test_ground_truth, test_indices,
                                     all_predictions, all_ground_truth, all_indices,
                                     output_dir, epoch="final", train_samples=None):
    """Create combined geospatial + temporal scientific figure with (A) and (B) labels.

    Args:
        dataset: Dataset object
        test_predictions: Test predictions for geospatial plot
        test_ground_truth: Test ground truth for geospatial plot
        test_indices: Test indices for geospatial plot
        all_predictions: Dict of all predictions for temporal plot
        all_ground_truth: Dict of all ground truth for temporal plot
        all_indices: Dict of all indices for temporal plot
        output_dir: Output directory
        epoch: Epoch number
        train_samples: Number of training samples
    """
    from datetime import datetime, timedelta
    import matplotlib.dates as mdates
    from matplotlib.gridspec import GridSpec

    # Import geopandas for US boundaries (required)
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError("geopandas is required for US state boundaries. Install with: pip install geopandas")

    # ========== STEP 1: DETERMINE CORRECT DIMENSIONS ==========
    lon_min, lon_max = -125, -66
    lat_min, lat_max = 24, 50

    # For mid-latitude maps, need to account for cos(lat) correction
    # At 37°N (mid-CONUS), 1° longitude ≈ 0.8 × 1° latitude in physical distance
    lat_center = (lat_min + lat_max) / 2
    cos_lat = np.cos(np.radians(lat_center))  # ~0.799 at 37°N

    # Physical aspect ratio corrected for latitude
    lon_span = lon_max - lon_min  # 59°
    lat_span = lat_max - lat_min  # 26°
    # Physical width/height ratio
    geo_aspect_physical = (lon_span * cos_lat) / lat_span  # ~1.81

    # Set desired geospatial WIDTH and compute natural HEIGHT
    geo_width_inches = 10.0
    geo_height_inches = geo_width_inches / geo_aspect_physical  # ~5.52 inches

    # Temporal plot: match the geospatial height, use original aspect ratio
    temp_aspect = 18.0 / 8.0  # 2.25
    temp_height_inches = geo_height_inches  # Match geo height
    temp_width_inches = temp_height_inches * temp_aspect  # ~12.4 inches

    # Combined figure dimensions - add extra space on right for y-axis labels
    total_width = geo_width_inches + temp_width_inches + 2.0  # Extra space for gap and right y-axis
    total_height = geo_height_inches

    # ========== STEP 2: CREATE FIGURE WITH CORRECT LAYOUT ==========
    fig = plt.figure(figsize=(total_width, total_height))
    gs = GridSpec(1, 2, figure=fig,
                 width_ratios=[geo_width_inches, temp_width_inches],
                 wspace=0.08, hspace=0)

    ax_geo = fig.add_subplot(gs[0])

    # Extract geospatial data
    coords = dataset.coords[test_indices].cpu().numpy()
    lats, lons = coords[:, 0], coords[:, 1]
    test_preds_denorm = test_predictions * MAX_LFMC_VALUE
    test_gts_denorm = test_ground_truth * MAX_LFMC_VALUE
    errors = np.abs(test_preds_denorm - test_gts_denorm)

    # Grid and bin data - 150x150km bins (approximately 1.35 degrees)
    grid_size = 1.35
    lon_bins = np.arange(lon_min, lon_max, grid_size)
    lat_bins = np.arange(lat_min, lat_max, grid_size)
    grid_errors = np.zeros((len(lat_bins)-1, len(lon_bins)-1))
    grid_counts = np.zeros((len(lat_bins)-1, len(lon_bins)-1))

    for i in range(len(lats)):
        if lon_min <= lons[i] <= lon_max and lat_min <= lats[i] <= lat_max:
            lon_idx = np.digitize(lons[i], lon_bins) - 1
            lat_idx = np.digitize(lats[i], lat_bins) - 1
            if 0 <= lon_idx < len(lon_bins)-1 and 0 <= lat_idx < len(lat_bins)-1:
                grid_errors[lat_idx, lon_idx] += errors[i]
                grid_counts[lat_idx, lon_idx] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        grid_avg_errors = grid_errors / grid_counts
        grid_avg_errors[grid_counts == 0] = np.nan

    # Plot grid cells with larger circles for 150km bins
    if np.any(~np.isnan(grid_avg_errors)):
        log_counts = np.log1p(grid_counts)
        log_counts[grid_counts == 0] = 0
        min_log = np.min(log_counts[log_counts > 0]) if np.any(log_counts > 0) else 1
        max_log = np.max(log_counts)
        # Larger circles for 150km grid - reduced by 25% from 150 to 112
        size_min, size_max = 30, 112

        for i in range(len(lat_bins)-1):
            for j in range(len(lon_bins)-1):
                if not np.isnan(grid_avg_errors[i, j]) and grid_counts[i, j] > 0:
                    lon_center = (lon_bins[j] + lon_bins[j+1]) / 2
                    lat_center = (lat_bins[i] + lat_bins[i+1]) / 2
                    if max_log > min_log:
                        size_norm = (log_counts[i, j] - min_log) / (max_log - min_log)
                        size = size_min + (size_max - size_min) * size_norm
                    else:
                        size = size_max
                    error_value = min(grid_avg_errors[i, j], 50.0)
                    ax_geo.scatter(lon_center, lat_center, s=size, c=[error_value],
                                 cmap='turbo', vmin=0, vmax=50, alpha=1.0,
                                 edgecolors='black', linewidth=0.5)

    # Plot US boundaries AFTER circles so boundaries are on top
    shapefile_path = Path(SCRIPT_DIR) / 'shapefiles' / 'cb_2018_us_state_20m.shp'
    if not shapefile_path.exists():
        raise FileNotFoundError(f"Shapefile not found at {shapefile_path}. Please ensure shapefiles are downloaded.")

    states = gpd.read_file(shapefile_path)
    states_conus = states[
        ~states['NAME'].isin(['Alaska', 'Hawaii', 'Puerto Rico',
                             'United States Virgin Islands', 'Guam',
                             'American Samoa', 'Northern Mariana Islands'])
    ]
    states_conus.boundary.plot(ax=ax_geo, color='#CCCCCC', linewidth=0.4, alpha=0.8)
    states_conus.dissolve().boundary.plot(ax=ax_geo, color='#AAAAAA', linewidth=0.5, alpha=0.9)

    # Add colorbar
    norm = plt.Normalize(vmin=0, vmax=50)
    sm = plt.cm.ScalarMappable(cmap='turbo', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_geo, label='Avg Error (pp)', fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 10, 20, 30, 40, 50])
    cbar.set_ticklabels(['0', '10', '20', '30', '40', '50+'])
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label('Avg Error (pp)', fontsize=14)

    # Set axis limits and aspect ratio (preserve geographic proportions)
    # Expand y-limits to match the vertical extent of temporal plot while maintaining aspect
    # Add padding to top and bottom to fill the subplot area
    lat_center_point = (lat_min + lat_max) / 2  # 37
    lat_range_current = lat_max - lat_min  # 26

    # Expand by 35% to fill more vertical space and align with temporal plot height
    lat_range_expanded = lat_range_current * 1.35
    lat_min_expanded = lat_center_point - lat_range_expanded / 2  # ~19.4
    lat_max_expanded = lat_center_point + lat_range_expanded / 2  # ~54.6

    ax_geo.set_xlim(lon_min, lon_max)
    ax_geo.set_ylim(lat_min_expanded, lat_max_expanded)
    # Set aspect ratio accounting for latitude: 1° lon = cos(lat) × 1° lat in physical space
    ax_geo.set_aspect(1.0 / cos_lat)  # This makes the map look geographically correct
    ax_geo.grid(False)

    # Add axis labels with consistent font size
    ax_geo.set_xlabel('Longitude', fontsize=14)
    ax_geo.set_ylabel('Latitude', fontsize=14)
    ax_geo.tick_params(axis='both', labelsize=13)

    # Panel label - just (A)
    ax_geo.text(0.02, 0.98, '(A)',
               transform=ax_geo.transAxes, fontsize=14, weight='bold',
               ha='left', va='top',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))

    # ========== PANEL B: TEMPORAL ==========
    ax_temp = fig.add_subplot(gs[1])

    # Combine test data
    all_preds, all_gts, all_times = [], [], []
    for split_name in ['test']:
        if split_name in all_predictions and len(all_predictions[split_name]) > 0:
            preds = all_predictions[split_name]
            gts = all_ground_truth[split_name]
            indices = all_indices[split_name]
            coords = dataset.coords[indices].cpu().numpy()
            times = coords[:, 3]
            all_preds.extend(preds)
            all_gts.extend(gts)
            all_times.extend(times)

    if len(all_preds) > 0:
        all_preds = np.array(all_preds) * MAX_LFMC_VALUE
        all_gts = np.array(all_gts) * MAX_LFMC_VALUE

        # Convert to dates
        base_date = datetime(2015, 1, 1)
        end_date = datetime(2025, 1, 1)
        total_days = (end_date - base_date).days
        dates = [base_date + timedelta(days=int(t * total_days)) for t in all_times]

        # Monthly binning
        monthly_data = defaultdict(lambda: {'preds': [], 'gts': []})
        for i, date in enumerate(dates):
            month_start = datetime(date.year, date.month, 1)
            monthly_data[month_start]['preds'].append(all_preds[i])
            monthly_data[month_start]['gts'].append(all_gts[i])

        months = sorted(monthly_data.keys())
        month_predictions, month_ground_truths, month_positions = [], [], []

        for month in months:
            preds = np.array(monthly_data[month]['preds'])
            gts = np.array(monthly_data[month]['gts'])
            if len(preds) >= 5:
                month_predictions.append(preds)
                month_ground_truths.append(gts)
                month_positions.append(mdates.date2num(month))

        # Plot violins
        if len(month_predictions) > 0:
            offset = 5
            gt_medians_list = []
            pred_medians_list = []
            gt_positions_list = []
            pred_positions_list = []

            for i, pos in enumerate(month_positions):
                # Ground truth (red)
                gt_parts = ax_temp.violinplot([month_ground_truths[i]],
                                            positions=[pos - offset],
                                            widths=8.4, showmeans=False,
                                            showmedians=False, showextrema=False)
                for pc in gt_parts['bodies']:
                    pc.set_facecolor('#B22222')
                    pc.set_edgecolor('#8B0000')
                    pc.set_alpha(0.6)
                    pc.set_linewidth(0.5)

                # Prediction (blue)
                pred_parts = ax_temp.violinplot([month_predictions[i]],
                                              positions=[pos + offset],
                                              widths=8.4, showmeans=False,
                                              showmedians=False, showextrema=False)
                for pc in pred_parts['bodies']:
                    pc.set_facecolor('#000080')
                    pc.set_edgecolor('#000050')
                    pc.set_alpha(0.6)
                    pc.set_linewidth(0.5)

                # Store medians and positions
                gt_median = np.median(month_ground_truths[i])
                pred_median = np.median(month_predictions[i])
                gt_medians_list.append(gt_median)
                pred_medians_list.append(pred_median)
                gt_positions_list.append(pos - offset)
                pred_positions_list.append(pos + offset)

            # Plot median points
            for pos, gt_median in zip(gt_positions_list, gt_medians_list):
                ax_temp.scatter(pos, gt_median, c='#B22222', s=40, zorder=6,
                              edgecolors='#8B0000', linewidth=0.5)
            for pos, pred_median in zip(pred_positions_list, pred_medians_list):
                ax_temp.scatter(pos, pred_median, c='#000080', s=40, zorder=6,
                              edgecolors='#000050', linewidth=0.5)

            # Format temporal axis
            ax_temp.xaxis_date()
            ax_temp.xaxis.set_major_locator(mdates.YearLocator())
            ax_temp.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax_temp.set_xlabel('Date', fontsize=14)
            ax_temp.set_ylabel('LFMC (%)', fontsize=14)

            # Move y-axis to the right side
            ax_temp.yaxis.tick_right()
            ax_temp.yaxis.set_label_position('right')

            # Set tick label sizes to match geospatial plot
            ax_temp.tick_params(axis='both', labelsize=13)

            ax_temp.grid(False)

            # Full legend with distributions and medians - larger font
            gt_violin = mpatches.Patch(color='#B22222', alpha=0.6,
                                      label='Ground Truth Distribution')
            gt_median_marker = plt.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor='#B22222', markeredgecolor='#8B0000',
                                        markersize=6, label='Ground Truth Median')
            pred_violin = mpatches.Patch(color='#000080', alpha=0.6,
                                        label='Prediction Distribution')
            pred_median_marker = plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor='#000080', markeredgecolor='#000050',
                                          markersize=6, label='Prediction Median')

            ax_temp.legend(handles=[gt_violin, gt_median_marker, pred_violin, pred_median_marker],
                         loc='upper right', frameon=True, fancybox=True, shadow=False, fontsize=12)

    # Panel label - just (B)
    ax_temp.text(0.02, 0.98, '(B)',
                transform=ax_temp.transAxes, fontsize=14, weight='bold',
                ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))

    # Save combined figure
    plt.savefig(output_dir / f'combined_scientific_figure_epoch_{epoch}.png',
                dpi=200, bbox_inches='tight')
    plt.close()

    print(f"  Combined scientific figure saved to: combined_scientific_figure_epoch_{epoch}.png", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default=None,
                       help='Path to LFMC CSV. If not provided, downloads AI2 official CSV.')
    parser.add_argument('--epochs', type=int, default=2500)
    parser.add_argument('--batch-size', type=int, default=30000)
    parser.add_argument('--lr', type=float, default=0.0125)
    parser.add_argument('--output-dir', type=str, default='./outputs')
    parser.add_argument('--species-dim', type=int, default=768,
                       help='Dimension of learnable species embeddings')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed for reproducibility (default: 0)')
    args = parser.parse_args()

    device = 'cuda'
    # Note: cudnn.benchmark is set to False in run_training_session for determinism
    torch.backends.cuda.matmul.allow_tf32 = True

    # Run training session
    run_training_session(args)


if __name__ == "__main__":
    main()
