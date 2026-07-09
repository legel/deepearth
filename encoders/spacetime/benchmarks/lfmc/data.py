"""
LFMC Dataset and data loading utilities.

Implements FullyGPUDataset with AI2 official splits following allenai/lfmc approach.
"""

import torch
import pandas as pd
import numpy as np
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional, Any

from .constants import (
    MAX_LFMC_VALUE,
    AI2_LFMC_CSV_URL,
    DEFAULT_AI2_CSV_PATH,
    DEFAULT_NUM_FOLDS,
    DEFAULT_VALIDATION_FOLDS,
    DEFAULT_TEST_FOLDS,
)


def download_ai2_csv(output_path: str = DEFAULT_AI2_CSV_PATH) -> Path:
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


def assign_ai2_folds(
    df: pd.DataFrame,
    id_column: str = 'sorting_id',
    num_folds: int = DEFAULT_NUM_FOLDS
) -> pd.DataFrame:
    """
    Assign folds using AI2's hash-based deterministic approach.

    This matches their approach in lfmc/lfmc/core/splits.py:
    - Uses SHA256 hash of sorting_id
    - Converts to deterministic fold assignment
    - Ensures reproducibility across runs
    """
    def create_prob(value: int) -> float:
        hash_val = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
        return int(hash_val[:8], 16) / 0xFFFFFFFF

    probs = df[id_column].apply(create_prob)
    df['fold'] = (probs * num_folds).astype(int)
    return df


def assign_splits_from_folds(
    df: pd.DataFrame,
    validation_folds: frozenset,
    test_folds: frozenset,
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


class FullyGPUDataset:
    """
    LFMC Dataset with everything on GPU from the start.

    Implements TrainableDataset protocol with get_batch_data() method.
    Loads AI2's official LFMC CSV and uses their fold-based train/test splits.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        device: str = 'cuda',
        use_ai2_splits: bool = True
    ):
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
                        print(f"       -> Species={species}: LFMC={lfmcs} (avg={avg_lfmc:.0f}%)", flush=True)
                    else:
                        print(f"       -> Species={species}: LFMC={lfmcs[0]:.0f}%", flush=True)

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
        self.is_sorted = False  # Track if spatiotemporal sorting has been applied

        print(f"\nGPU dataset ready: {n:,} samples", flush=True)

    def get_batch_data(self, indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get all data needed for a batch given indices.

        Implements TrainableDataset protocol.

        Args:
            indices: (B,) tensor of sample indices

        Returns:
            Dictionary with coords, targets, species_idx, is_degenerate
        """
        return {
            'coords': self.coords[indices],
            'targets': self.targets[indices],
            'species_idx': self.species_idx[indices],
            'is_degenerate': self.is_degenerate[indices]
        }

    def apply_spatiotemporal_sort(self):
        """
        Sort all data by 4D Morton code for spatial locality optimization.

        After sorting, spatiotemporally nearby points are adjacent in memory.
        This can improve cache locality for all Earth4D grids.

        Call this once before training starts.
        """
        # Import here to avoid circular imports
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from sorting import spatiotemporal_sort_indices

        if self.is_sorted:
            print("Dataset already sorted, skipping.", flush=True)
            return

        print("\nApplying spatiotemporal sort for locality optimization...", flush=True)

        # Compute sort indices from coordinates
        sort_idx = spatiotemporal_sort_indices(self.coords)

        # Sort all data tensors
        self.coords = self.coords[sort_idx]
        self.targets = self.targets[sort_idx]
        self.species_idx = self.species_idx[sort_idx]
        self.is_degenerate = self.is_degenerate[sort_idx]
        self.date_values = self.date_values[sort_idx]

        # Update split indices to point to sorted positions
        # Create inverse mapping: inverse_idx[sort_idx[i]] = i
        inverse_idx = torch.empty_like(sort_idx)
        inverse_idx[sort_idx] = torch.arange(len(sort_idx), device=sort_idx.device)

        if self.train_indices is not None:
            self.train_indices = inverse_idx[self.train_indices]
        if self.test_indices is not None:
            self.test_indices = inverse_idx[self.test_indices]

        self.is_sorted = True
        print(f"  Sorted {self.n:,} samples by 4D Morton code", flush=True)


def get_ai2_splits(dataset: FullyGPUDataset) -> Dict[str, torch.Tensor]:
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


def compute_lfmc_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    batch_data: Dict[str, torch.Tensor]
) -> Dict[str, Any]:
    """
    Compute LFMC metrics with denormalization and degeneracy tracking.

    This is the LFMC-specific metrics callback for use with generic training.

    Args:
        predictions: Model predictions (normalized [0, 1])
        targets: Ground truth targets (normalized [0, 1])
        batch_data: Dictionary containing 'is_degenerate' mask

    Returns:
        Dictionary with overall, unique, and degenerate metrics
    """
    # Denormalize to LFMC percentage points
    preds_pp = predictions * MAX_LFMC_VALUE
    targets_pp = targets * MAX_LFMC_VALUE

    def calc_metrics(p: torch.Tensor, t: torch.Tensor) -> Dict[str, float]:
        if len(p) == 0:
            return {'mse': 0, 'rmse': 0, 'mae': 0, 'r2': 0, 'n_samples': 0}

        errors = p - t
        mse = (errors ** 2).mean()
        rmse = torch.sqrt(mse)
        mae = torch.abs(errors).mean()

        ss_res = (errors ** 2).sum()
        ss_tot = ((t - t.mean()) ** 2).sum()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else torch.tensor(0.0, device=p.device)

        return {
            'mse': mse.item(),
            'rmse': rmse.item(),
            'mae': mae.item(),
            'r2': r2.item(),
            'n_samples': len(p)
        }

    # Overall metrics
    overall = calc_metrics(preds_pp, targets_pp)

    # Split by degeneracy
    is_degenerate = batch_data.get('is_degenerate')
    if is_degenerate is not None:
        unique_mask = ~is_degenerate
        unique = calc_metrics(preds_pp[unique_mask], targets_pp[unique_mask])
        degen = calc_metrics(preds_pp[is_degenerate], targets_pp[is_degenerate])
        return {
            'overall': overall,
            'unique': unique,
            'degenerate': degen
        }

    return overall
