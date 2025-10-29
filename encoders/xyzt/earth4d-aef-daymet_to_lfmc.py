#!/usr/bin/env python3
"""
Earth4D + Species + AEF + Daymet LFMC Prediction
================================================
Combines Earth4D spatiotemporal encoding + Species embeddings + AlphaEarth Features (AEF) + Daymet weather.
Supports modular feature selection with auto-download from cloud storage.

Features:
- Modular dataset loading (AEF and Daymet are optional)
- Auto-download missing datasets from Google Cloud Storage
- Learnable or BioCLIP species embeddings
- Daymet weather normalization (0-1 based on min/max)
- Automatic MLP input dimension adjustment
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

# Import Earth4D encoder
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'deepearth-earth-observation', 'encoders', 'xyzt'))
from earth4d import Earth4D

# Import modular dataset loader
from lfmc_dataset import LFMCDataset


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
        self.sample_predictions = defaultdict(list)

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


def load_bioclip_embeddings(bioclip_path='./species_embeddings',
                           lfmc_species=None,
                           cache_path='./lfmc_bioclip_embeddings_cache.pt'):
    """Load BioCLIP 2 species embeddings, using cache if available."""

    # Check if we have a cached version for LFMC species
    if lfmc_species is not None and os.path.exists(cache_path):
        print(f"\nLoading cached BioCLIP embeddings from {cache_path}", flush=True)
        cache_data = torch.load(cache_path, weights_only=False)

        # Verify cache is still valid
        if set(cache_data['species']) == set(lfmc_species):
            print(f"  Cache valid: {len(cache_data['embeddings'])} species, {cache_data['embedding_dim']}D embeddings", flush=True)
            return cache_data['embeddings']
        else:
            print(f"  Cache invalid (species mismatch), regenerating...", flush=True)

    # Load from original BioCLIP files
    mapping_path = os.path.join(bioclip_path, 'species_occurrence_counts_with_embeddings.csv')
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"BioCLIP mapping file not found: {mapping_path}")

    mapping_df = pd.read_csv(mapping_path)

    print(f"\nLoading BioCLIP 2 embeddings from {bioclip_path}", flush=True)
    print(f"  Total species in BioCLIP database: {len(mapping_df):,}", flush=True)

    # If we know which species we need, filter the dataframe
    if lfmc_species is not None:
        relevant_df = mapping_df[mapping_df['species'].isin(lfmc_species)]
        print(f"  Filtering to {len(relevant_df)} relevant species for LFMC dataset", flush=True)
    else:
        relevant_df = mapping_df

    # Create species name to embedding mapping
    species_to_embedding = {}
    loaded_chunks = {}

    for _, row in relevant_df.iterrows():
        species_name = row['species']
        chunk_file = row['embedding_file']
        emb_index = row['embedding_index']

        # Load chunk if not already loaded
        if chunk_file not in loaded_chunks:
            chunk_path = os.path.join(bioclip_path, chunk_file)
            if not os.path.exists(chunk_path):
                raise FileNotFoundError(f"BioCLIP chunk file not found: {chunk_path}")
            chunk_data = torch.load(chunk_path, weights_only=False)
            loaded_chunks[chunk_file] = chunk_data['embeddings']
            print(f"  Loaded {chunk_file}: {chunk_data['embeddings'].shape}", flush=True)

        # Get the embedding for this species
        if chunk_file in loaded_chunks:
            chunk_embeddings = loaded_chunks[chunk_file]
            local_idx = emb_index % chunk_embeddings.shape[0]
            species_to_embedding[species_name] = chunk_embeddings[local_idx].float()

    print(f"  Loaded embeddings for {len(species_to_embedding):,} species", flush=True)

    # Check embedding dimension
    if not species_to_embedding:
        raise ValueError("No BioCLIP embeddings loaded")

    sample_emb = next(iter(species_to_embedding.values()))
    print(f"  Embedding dimension: {sample_emb.shape[0]}", flush=True)

    # Save cache if we filtered to LFMC species
    if lfmc_species is not None and cache_path:
        print(f"  Saving cache to {cache_path}", flush=True)
        cache_data = {
            'species': list(species_to_embedding.keys()),
            'embeddings': species_to_embedding,
            'embedding_dim': sample_emb.shape[0]
        }
        torch.save(cache_data, cache_path)
        print(f"  Cache saved successfully", flush=True)

    return species_to_embedding


class ModularGPUDataset:
    """Modular GPU dataset with auto-download and optional features (AEF, Daymet)."""

    def __init__(self, data_dir: str = "./data", device: str = 'cuda',
                 use_aef: bool = True, use_daymet: bool = True,
                 use_bioclip: bool = False, auto_download: bool = True):
        """
        Load LFMC dataset with modular features.

        Args:
            data_dir: Directory containing Parquet files
            device: 'cuda' or 'cpu'
            use_aef: Include AlphaEarth Features (64D)
            use_daymet: Include Daymet weather features
            use_bioclip: Use BioCLIP species embeddings
            auto_download: Auto-download missing files from cloud storage
        """
        print("\nLoading LFMC dataset with modular features...", flush=True)
        print(f"  AEF enabled: {use_aef}", flush=True)
        print(f"  Daymet enabled: {use_daymet}", flush=True)
        print(f"  Auto-download: {auto_download}", flush=True)

        # Load dataset using modular loader
        self.dataset = LFMCDataset(
            data_dir=data_dir,
            use_aef=use_aef,
            use_daymet=use_daymet,
            auto_download=auto_download,
            use_cache=True,
            verbose=True
        )

        self.use_aef = use_aef
        self.use_daymet = use_daymet
        self.use_bioclip = use_bioclip
        self.device = device

        # Get statistics
        stats = self.dataset.get_statistics()
        self.n = stats['n_samples']

        print(f"\nDataset loaded: {self.n:,} samples", flush=True)

        # Extract required columns
        df = self.dataset.data

        # Coordinates and targets
        coords_np = np.column_stack([
            df['lat'].values,
            df['lon'].values,
            df['elevation_m'].values,
            df['date'].astype('int64').values / 1e9  # Convert to float
        ])

        # Normalize time to [0, 1] (2015-2025 range)
        time_values = coords_np[:, 3]
        time_norm = (time_values - time_values.min()) / (time_values.max() - time_values.min())
        coords_np[:, 3] = time_norm

        # Transfer to GPU
        self.coords = torch.tensor(coords_np, dtype=torch.float32, device=device)
        self.targets = torch.tensor(df['lfmc_percent'].values, dtype=torch.float32, device=device)

        # Species encoding
        unique_species = df['species'].unique()
        self.species_to_idx = {species: idx for idx, species in enumerate(unique_species)}
        self.idx_to_species = {idx: species for species, idx in self.species_to_idx.items()}
        self.n_species = len(unique_species)

        species_indices = np.array([self.species_to_idx[s] for s in df['species'].values])
        self.species_idx = torch.tensor(species_indices, dtype=torch.long, device=device)

        print(f"\nSpecies Statistics:", flush=True)
        print(f"  Unique species: {self.n_species}", flush=True)

        # Show top 10 most common species
        species_counts = df['species'].value_counts()
        print(f"  Top 10 species:", flush=True)
        for i, (species, count) in enumerate(species_counts.head(10).items()):
            print(f"    {i+1:2d}. {species}: {count:,} samples ({100*count/self.n:.1f}%)", flush=True)

        # AEF features (optional)
        if use_aef:
            aef_cols = [col for col in df.columns if col.startswith('aef_')]
            if not aef_cols:
                raise ValueError("AEF features requested but not found in dataset")
            aef_np = df[aef_cols].values.astype(np.float32)
            self.aef_features = torch.tensor(aef_np, dtype=torch.float32, device=device)
            self.aef_dim = aef_np.shape[1]
            print(f"\nAEF features loaded: {self.aef_dim}D embeddings", flush=True)
        else:
            self.aef_features = None
            self.aef_dim = 0

        # Daymet features (optional)
        if use_daymet:
            # Get Daymet columns (exclude datetime columns and metadata)
            daymet_cols = [col for col in df.columns
                          if any(x in col for x in ['prcp', 'tmin', 'tmax', 'srad', 'vp', 'dayl', 'swe'])
                          and 'start' not in col and 'end' not in col]

            if not daymet_cols:
                raise ValueError("Daymet features requested but not found in dataset")

            daymet_np = df[daymet_cols].values.astype(np.float32)

            # Normalize Daymet features to [0, 1] based on column min/max
            print(f"\nNormalizing Daymet features ({len(daymet_cols)} columns)...", flush=True)
            daymet_normalized = np.zeros_like(daymet_np)
            for i, col in enumerate(daymet_cols):
                col_min = daymet_np[:, i].min()
                col_max = daymet_np[:, i].max()
                if col_max > col_min:
                    daymet_normalized[:, i] = (daymet_np[:, i] - col_min) / (col_max - col_min)
                else:
                    daymet_normalized[:, i] = 0.5  # Constant column
                print(f"  {col:20s}: min={col_min:8.2f}, max={col_max:8.2f}", flush=True)

            self.daymet_features = torch.tensor(daymet_normalized, dtype=torch.float32, device=device)
            self.daymet_dim = daymet_normalized.shape[1]
            print(f"\nDaymet features normalized: {self.daymet_dim}D vector", flush=True)
        else:
            self.daymet_features = None
            self.daymet_dim = 0

        # BioCLIP embeddings (optional)
        self.bioclip_embeddings = None
        self.bioclip_dim = 768

        if use_bioclip:
            lfmc_species = list(unique_species)
            cache_path = os.path.join(data_dir, 'lfmc_bioclip_embeddings_cache.pt')
            bioclip_dict = load_bioclip_embeddings(lfmc_species=lfmc_species, cache_path=cache_path)

            # Check matching
            bioclip_species = set(bioclip_dict.keys())
            matched_species = set(lfmc_species) & bioclip_species
            missing_species = set(lfmc_species) - bioclip_species

            print(f"\nSpecies matching:", flush=True)
            print(f"  LFMC species: {len(lfmc_species)}", flush=True)
            print(f"  Matched with BioCLIP: {len(matched_species)} ({100*len(matched_species)/len(lfmc_species):.1f}%)", flush=True)

            if missing_species:
                raise ValueError(f"BioCLIP embeddings missing for {len(missing_species)} species. "
                               f"Set use_bioclip=False or ensure all species have embeddings.")

            # Create embedding matrix
            embedding_matrix = torch.zeros((self.n_species, self.bioclip_dim), dtype=torch.float32)
            for species, idx in self.species_to_idx.items():
                embedding_matrix[idx] = bioclip_dict[species]

            self.species_embeddings = embedding_matrix.to(device)
            print(f"  BioCLIP embeddings prepared: {embedding_matrix.shape}", flush=True)

        # Degeneracy analysis
        coord_keys = []
        coord_to_indices = defaultdict(list)

        for i in range(self.n):
            key = (
                round(coords_np[i, 0], 6),  # lat
                round(coords_np[i, 1], 6),  # lon
                round(coords_np[i, 2], 2),  # elev
                round(coords_np[i, 3], 6)   # time
            )
            coord_keys.append(key)
            coord_to_indices[key].append(i)

        # Create degeneracy flags
        is_degenerate = np.zeros(self.n, dtype=bool)
        degenerate_groups = []

        for key, indices in coord_to_indices.items():
            if len(indices) > 1:
                species_at_coord = df['species'].values[indices]
                unique_species_at_coord = np.unique(species_at_coord)

                if len(unique_species_at_coord) > 1:
                    is_degenerate[indices] = True
                    degenerate_groups.append({
                        'coord': key,
                        'indices': indices,
                        'lfmc_values': df['lfmc_percent'].values[indices].tolist(),
                        'species': species_at_coord.tolist()
                    })

        self.is_degenerate = torch.tensor(is_degenerate, dtype=torch.bool, device=device)

        n_unique_coords = len(coord_to_indices)
        n_degenerate_coords = len(degenerate_groups)
        n_degenerate_samples = np.sum(is_degenerate)

        print(f"\nDegeneracy Analysis:", flush=True)
        print(f"  Total samples: {self.n:,}", flush=True)
        print(f"  Unique spatiotemporal coordinates: {n_unique_coords:,}", flush=True)
        print(f"  Multi-species degenerate coordinates: {n_degenerate_coords:,} ({100*n_degenerate_coords/n_unique_coords:.1f}%)", flush=True)
        print(f"  Multi-species degenerate samples: {n_degenerate_samples:,} ({100*n_degenerate_samples/self.n:.1f}%)", flush=True)

        # Store for splitting
        self.date_values = torch.tensor(coords_np[:, 3], dtype=torch.float32, device=device)
        self.df = df

        self.n_degenerate_samples = n_degenerate_samples
        self.n_unique_samples = self.n - n_degenerate_samples

        print(f"\nGPU dataset ready: {self.n:,} samples", flush=True)
        print(f"  AEF: {self.aef_dim}D", flush=True)
        print(f"  Daymet: {self.daymet_dim}D", flush=True)
        print(f"  Total feature dims: {self.aef_dim + self.daymet_dim}", flush=True)


class ModularLFMCModel(nn.Module):
    """LFMC model with fully modular features (all optional: Earth4D, Species, AEF, Daymet)."""

    def __init__(self, n_species, species_dim=32, aef_dim=0, daymet_dim=0,
                 use_earth4d=True, use_species=True,
                 use_bioclip=False, bioclip_embeddings=None, freeze_embeddings=False):
        super().__init__()

        self.use_earth4d = use_earth4d
        self.use_species = use_species
        self.use_bioclip = use_bioclip
        self.aef_dim = aef_dim
        self.daymet_dim = daymet_dim

        earth4d_dim = 0
        if use_earth4d:
            self.earth4d = Earth4D(
                spatial_levels=24,
                temporal_levels=19,
                spatial_log2_hashmap_size=22,
                temporal_log2_hashmap_size=18,
                verbose=False
            )
            earth4d_dim = self.earth4d.get_output_dim()

            # Initialize Earth4D parameters
            with torch.no_grad():
                for p in self.earth4d.parameters():
                    if p.dim() > 1:
                        nn.init.uniform_(p, -0.1, 0.1)
        else:
            self.earth4d = None

        # Species embeddings (optional)
        if use_species:
            if use_bioclip:
                if bioclip_embeddings is None:
                    raise ValueError("BioCLIP embeddings must be provided when use_bioclip=True")
                self.species_embeddings = nn.Embedding.from_pretrained(bioclip_embeddings, freeze=freeze_embeddings)
                species_dim = bioclip_embeddings.shape[1]
                status = "FROZEN" if freeze_embeddings else "TRAINABLE"
                print(f"  Using {status} BioCLIP embeddings: {bioclip_embeddings.shape}", flush=True)
            else:
                self.species_embeddings = nn.Embedding(n_species, species_dim)
                nn.init.normal_(self.species_embeddings.weight, mean=0.0, std=0.1)
                if freeze_embeddings:
                    self.species_embeddings.weight.requires_grad = False
                status = "FROZEN" if freeze_embeddings else "TRAINABLE"
                print(f"  Using {status} random embeddings: ({n_species}, {species_dim})", flush=True)
        else:
            self.species_embeddings = None
            species_dim = 0

        # MLP input = Earth4D + Species + AEF + Daymet (all modular)
        input_dim = earth4d_dim + species_dim + aef_dim + daymet_dim

        if input_dim == 0:
            raise ValueError("At least one feature type must be enabled (Earth4D, Species, AEF, or Daymet)")

        print(f"  MLP input dimension: {input_dim} = Earth4D({earth4d_dim}) + Species({species_dim}) + AEF({aef_dim}) + Daymet({daymet_dim})", flush=True)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

        self.n_species = n_species
        self.species_dim = species_dim

    def forward(self, coords, species_idx, aef_features=None, daymet_features=None):
        feature_list = []

        # Add Earth4D if enabled
        if self.use_earth4d:
            earth4d_features = self.earth4d(coords)
            feature_list.append(earth4d_features)

        # Add species embeddings if enabled
        if self.use_species:
            species_features = self.species_embeddings(species_idx)
            feature_list.append(species_features)

        # Add AEF if provided
        if aef_features is not None and self.aef_dim > 0:
            feature_list.append(aef_features)

        # Add Daymet if provided
        if daymet_features is not None and self.daymet_dim > 0:
            feature_list.append(daymet_features)

        # Concatenate all features
        if len(feature_list) == 0:
            raise RuntimeError("No features available for forward pass")

        combined_features = torch.cat(feature_list, dim=-1)

        # Predict LFMC
        return self.mlp(combined_features).squeeze(-1)


def create_gpu_splits(dataset, device='cuda'):
    """Create train/test splits (temporal, spatial, random)."""
    n = dataset.n

    # All indices on GPU
    all_idx = torch.arange(n, device=device)

    # 1. Temporal: sort by date and take last 5%
    n_temp = int(n * 0.05)
    _, date_order = torch.sort(dataset.date_values)
    temp_idx = date_order[-n_temp:]

    # 2. Create mask for used indices
    used = torch.zeros(n, dtype=torch.bool, device=device)
    used[temp_idx] = True

    # 3. Spatial: Create 5 spatial clusters
    n_spat = int(n * 0.05)
    available = all_idx[~used]

    available_coords = dataset.coords[available][:, :2]  # lat, lon

    n_available = len(available)
    center_indices = torch.randperm(n_available, device=device)[:5]
    cluster_centers = available_coords[center_indices]

    spat_indices = []
    samples_per_cluster = n_spat // 5
    remaining_samples = n_spat % 5

    for i, center in enumerate(cluster_centers):
        distances = torch.sum((available_coords - center.unsqueeze(0)) ** 2, dim=1)
        n_samples = samples_per_cluster + (1 if i < remaining_samples else 0)
        _, nearest_indices = torch.topk(distances, n_samples, largest=False)
        cluster_samples = available[nearest_indices]
        spat_indices.append(cluster_samples)

    spat_idx = torch.cat(spat_indices)
    used[spat_idx] = True

    dataset.spatial_cluster_centers = cluster_centers.cpu().numpy()

    # 4. Random: 5% from remaining
    n_rand = int(n * 0.05)
    available = all_idx[~used]
    perm = torch.randperm(len(available), device=device)
    rand_idx = available[perm[:n_rand]]
    used[rand_idx] = True

    # 5. Train: everything else
    train_idx = all_idx[~used]

    print(f"\nSplits: Train={len(train_idx)}, Temporal={len(temp_idx)}, "
          f"Spatial={len(spat_idx)}, Random={len(rand_idx)}", flush=True)

    # Species coverage
    train_species = set(dataset.df.iloc[train_idx.cpu().numpy()]['species'].unique())
    temp_species = set(dataset.df.iloc[temp_idx.cpu().numpy()]['species'].unique())
    spat_species = set(dataset.df.iloc[spat_idx.cpu().numpy()]['species'].unique())
    rand_species = set(dataset.df.iloc[rand_idx.cpu().numpy()]['species'].unique())

    print(f"\nSpecies coverage:", flush=True)
    print(f"  Train species: {len(train_species)}", flush=True)
    print(f"  Temporal test: {len(temp_species)} ({100*len(temp_species & train_species)/len(temp_species):.1f}% in train)", flush=True)
    print(f"  Spatial test: {len(spat_species)} ({100*len(spat_species & train_species)/len(spat_species):.1f}% in train)", flush=True)
    print(f"  Random test: {len(rand_species)} ({100*len(rand_species & train_species)/len(rand_species):.1f}% in train)", flush=True)

    # Degeneracy breakdown
    print(f"\nDegeneracy breakdown by split:", flush=True)
    for name, idx in [('Train', train_idx), ('Temporal', temp_idx),
                       ('Spatial', spat_idx), ('Random', rand_idx)]:
        n_degen = dataset.is_degenerate[idx].sum().item()
        n_unique = len(idx) - n_degen
        print(f"  {name:8s}: {n_unique:5d} unique, {n_degen:5d} degenerate ({100*n_degen/len(idx):.1f}%)", flush=True)

    return {
        'train': train_idx,
        'temporal': temp_idx,
        'spatial': spat_idx,
        'random': rand_idx
    }


def compute_metrics_gpu(preds, targets, is_degenerate=None):
    """Compute metrics on GPU."""
    def calc_metrics(p, t):
        errors = p - t
        abs_errors = torch.abs(errors)

        mse = (errors ** 2).mean()
        rmse = torch.sqrt(mse)
        mae = abs_errors.mean()
        median_ae = torch.median(abs_errors)
        error_var = errors.var()
        error_std = errors.std()

        return {
            'mse': mse.item(),
            'rmse': rmse.item(),
            'mae': mae.item(),
            'median_ae': median_ae.item(),
            'error_var': error_var.item(),
            'error_std': error_std.item()
        }

    overall = calc_metrics(preds, targets)

    if is_degenerate is not None:
        unique_mask = ~is_degenerate
        degen_mask = is_degenerate

        empty_metrics = {'mse': 0, 'rmse': 0, 'mae': 0, 'median_ae': 0, 'error_var': 0, 'error_std': 0}

        unique_metrics = calc_metrics(preds[unique_mask], targets[unique_mask]) if unique_mask.sum() > 0 else empty_metrics
        degen_metrics = calc_metrics(preds[degen_mask], targets[degen_mask]) if degen_mask.sum() > 0 else empty_metrics

        return overall, unique_metrics, degen_metrics

    return overall


def train_epoch_gpu(model, dataset, indices, optimizer, batch_size=20000):
    """Train for one epoch."""
    model.train()
    n = len(indices)

    # Shuffle on GPU
    perm = torch.randperm(n, device=indices.device)
    indices = indices[perm]

    criterion = nn.MSELoss()

    all_preds = []
    all_targets = []
    all_degens = []

    n_batches = (n + batch_size - 1) // batch_size

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]

        coords = dataset.coords[batch_idx]
        targets = dataset.targets[batch_idx]
        species = dataset.species_idx[batch_idx]

        # Optional features
        aef_features = dataset.aef_features[batch_idx] if dataset.aef_features is not None else None
        daymet_features = dataset.daymet_features[batch_idx] if dataset.daymet_features is not None else None

        preds = model(coords, species, aef_features, daymet_features)
        loss = criterion(preds, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        all_preds.append(preds.detach())
        all_targets.append(targets)
        all_degens.append(dataset.is_degenerate[batch_idx])

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_degens = torch.cat(all_degens)

    overall, unique, degen = compute_metrics_gpu(all_preds, all_targets, all_degens)

    return overall, unique, degen


@torch.no_grad()
def evaluate_split(model, dataset, indices):
    """Evaluate a single split."""
    model.eval()

    empty_metrics = {'mse': 0, 'rmse': 0, 'mae': 0, 'median_ae': 0, 'error_var': 0, 'error_std': 0}

    if len(indices) == 0:
        return empty_metrics, empty_metrics, empty_metrics, [], [], []

    coords = dataset.coords[indices]
    targets = dataset.targets[indices]
    species = dataset.species_idx[indices]
    degens = dataset.is_degenerate[indices]

    # Optional features
    aef_features = dataset.aef_features[indices] if dataset.aef_features is not None else None
    daymet_features = dataset.daymet_features[indices] if dataset.daymet_features is not None else None

    preds = model(coords, species, aef_features, daymet_features)

    overall, unique, degen = compute_metrics_gpu(preds, targets, degens)

    # Sample predictions
    unique_idx = torch.where(~degens)[0]
    degen_idx = torch.where(degens)[0]

    sample_preds = []
    sample_trues = []
    sample_types = []

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


def print_predictions_table(tmp_gt, tmp_pred, tmp_types, spt_gt, spt_pred, spt_types, rnd_gt, rnd_pred, rnd_types):
    """Print predictions table."""
    print("  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐", flush=True)

    max_len = max(len(tmp_gt), len(spt_gt), len(rnd_gt))

    for i in range(min(5, max_len)):
        line = "  │ "

        if i < len(tmp_gt):
            tmp_err = abs(tmp_gt[i] - tmp_pred[i])
            tmp_tag = "UNIQUE" if tmp_types[i] == 'U' else "MULTI"
            line += f"TEMPORAL-{tmp_tag}: {tmp_gt[i]:3.0f}%→{tmp_pred[i]:3.0f}% (Δ{tmp_err:3.0f}%) │ "
        else:
            line += " " * 36 + " │ "

        if i < len(spt_gt):
            spt_err = abs(spt_gt[i] - spt_pred[i])
            spt_tag = "UNIQUE" if spt_types[i] == 'U' else "MULTI"
            line += f"SPATIAL-{spt_tag}: {spt_gt[i]:3.0f}%→{spt_pred[i]:3.0f}% (Δ{spt_err:3.0f}%) │ "
        else:
            line += " " * 35 + " │ "

        if i < len(rnd_gt):
            rnd_err = abs(rnd_gt[i] - rnd_pred[i])
            rnd_tag = "UNIQUE" if rnd_types[i] == 'U' else "MULTI"
            line += f"RANDOM-{rnd_tag}: {rnd_gt[i]:3.0f}%→{rnd_pred[i]:3.0f}% (Δ{rnd_err:3.0f}%) │"
        else:
            line += " " * 34 + " │"

        print(line, flush=True)

    print("  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘", flush=True)
    print("    (UNIQUE=Single species at location, MULTI=Multiple species at location)", flush=True)


def run_training_session(args, run_name=""):
    """Run training session."""
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = 'cuda'
    print(f"Random seed: {args.seed}", flush=True)

    # Create output directory
    output_suffix = f"_{run_name}" if run_name else ""
    output_dir = Path(args.output_dir + output_suffix)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80, flush=True)
    feature_desc = []
    if args.use_earth4d:
        feature_desc.append("Earth4D")
    if args.use_species:
        if args.use_bioclip:
            feature_desc.append("BioCLIP")
        else:
            feature_desc.append("Species")
    if args.use_aef:
        feature_desc.append("AEF")
    if args.use_daymet:
        feature_desc.append("Daymet")
    features_str = " + ".join(feature_desc) if feature_desc else "NONE"

    print(f"LFMC PREDICTION - {features_str}", flush=True)
    print("="*80, flush=True)

    # Validate at least one feature
    if not any([args.use_earth4d, args.use_species, args.use_aef, args.use_daymet]):
        raise ValueError("At least one feature type must be enabled (--use-earth4d, --use-species, --use-aef, or --use-daymet)")

    # Load dataset
    dataset = ModularGPUDataset(
        data_dir=args.data_dir,
        device=device,
        use_aef=args.use_aef,
        use_daymet=args.use_daymet,
        use_bioclip=args.use_bioclip if args.use_species else False,
        auto_download=args.auto_download
    )

    splits = create_gpu_splits(dataset, device)

    # Create model
    if args.use_bioclip and args.use_species:
        model = ModularLFMCModel(
            dataset.n_species,
            aef_dim=dataset.aef_dim,
            daymet_dim=dataset.daymet_dim,
            use_earth4d=args.use_earth4d,
            use_species=args.use_species,
            use_bioclip=True,
            bioclip_embeddings=dataset.species_embeddings,
            freeze_embeddings=args.freeze_embeddings
        ).to(device)
    else:
        model = ModularLFMCModel(
            dataset.n_species,
            species_dim=args.species_dim,
            aef_dim=dataset.aef_dim,
            daymet_dim=dataset.daymet_dim,
            use_earth4d=args.use_earth4d,
            use_species=args.use_species,
            use_bioclip=False,
            freeze_embeddings=args.freeze_embeddings
        ).to(device)

    # Count parameters
    earth4d_params = sum(p.numel() for p in model.earth4d.parameters()) if model.earth4d is not None else 0
    species_params = sum(p.numel() for p in model.species_embeddings.parameters()) if model.species_embeddings is not None else 0
    mlp_params = sum(p.numel() for p in model.mlp.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel architecture:", flush=True)
    if args.use_earth4d:
        print(f"  Earth4D parameters: {earth4d_params:,}", flush=True)

    if args.use_species:
        embedding_type = "BioCLIP" if args.use_bioclip else "Random"
        embedding_status = "FROZEN" if args.freeze_embeddings else "TRAINABLE"
        embedding_dim = 768 if args.use_bioclip else args.species_dim
        print(f"  Species embedding: {species_params:,} ({embedding_type} {embedding_status}, {dataset.n_species} × {embedding_dim}D)", flush=True)

    print(f"  MLP parameters: {mlp_params:,}", flush=True)
    print(f"  Total parameters: {total_params:,}", flush=True)
    print(f"  Trainable parameters: {trainable_params:,}", flush=True)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)

    # Tracking
    metrics_history = []
    metrics_ema = MetricsEMA(alpha=0.1)

    print("\n" + "="*80, flush=True)
    print("Training (UNIQUE=Single species, MULTI=Multi-species):", flush=True)
    print("-"*80, flush=True)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        trn_overall, trn_unique, trn_degen = train_epoch_gpu(
            model, dataset, splits['train'], optimizer, args.batch_size
        )

        # Evaluate
        tmp_overall, tmp_unique, tmp_degen, tmp_gt, tmp_pred, tmp_types = evaluate_split(
            model, dataset, splits['temporal']
        )
        spt_overall, spt_unique, spt_degen, spt_gt, spt_pred, spt_types = evaluate_split(
            model, dataset, splits['spatial']
        )
        rnd_overall, rnd_unique, rnd_degen, rnd_gt, rnd_pred, rnd_types = evaluate_split(
            model, dataset, splits['random']
        )

        dt = time.time() - t0

        # Metrics
        current_metrics = {
            'epoch': epoch,
            'time': dt,
            'train_mse': trn_overall['mse'],
            'train_rmse': trn_overall['rmse'],
            'train_mae': trn_overall['mae'],
            'train_median_ae': trn_overall['median_ae'],
            'train_error_var': trn_overall['error_var'],
            'train_error_std': trn_overall['error_std'],
            'train_unique_mae': trn_unique['mae'],
            'train_unique_median_ae': trn_unique['median_ae'],
            'train_unique_std': trn_unique['error_std'],
            'train_degen_mae': trn_degen['mae'],
            'train_degen_median_ae': trn_degen['median_ae'],
            'train_degen_std': trn_degen['error_std'],
            'temporal_mse': tmp_overall['mse'],
            'temporal_mae': tmp_overall['mae'],
            'temporal_median_ae': tmp_overall['median_ae'],
            'temporal_error_std': tmp_overall['error_std'],
            'temporal_unique_mae': tmp_unique['mae'],
            'temporal_unique_median_ae': tmp_unique['median_ae'],
            'temporal_unique_std': tmp_unique['error_std'],
            'temporal_degen_mae': tmp_degen['mae'],
            'temporal_degen_median_ae': tmp_degen['median_ae'],
            'temporal_degen_std': tmp_degen['error_std'],
            'spatial_mse': spt_overall['mse'],
            'spatial_mae': spt_overall['mae'],
            'spatial_median_ae': spt_overall['median_ae'],
            'spatial_error_std': spt_overall['error_std'],
            'spatial_unique_mae': spt_unique['mae'],
            'spatial_unique_median_ae': spt_unique['median_ae'],
            'spatial_unique_std': spt_unique['error_std'],
            'spatial_degen_mae': spt_degen['mae'],
            'spatial_degen_median_ae': spt_degen['median_ae'],
            'spatial_degen_std': spt_degen['error_std'],
            'random_mse': rnd_overall['mse'],
            'random_mae': rnd_overall['mae'],
            'random_median_ae': rnd_overall['median_ae'],
            'random_error_std': rnd_overall['error_std'],
            'random_unique_mae': rnd_unique['mae'],
            'random_unique_median_ae': rnd_unique['median_ae'],
            'random_unique_std': rnd_unique['error_std'],
            'random_degen_mae': rnd_degen['mae'],
            'random_degen_median_ae': rnd_degen['median_ae'],
            'random_degen_std': rnd_degen['error_std']
        }

        ema_metrics = metrics_ema.update(current_metrics)
        current_metrics.update(ema_metrics)
        metrics_history.append(current_metrics)

        # Print
        print(f"\nEPOCH {epoch:3d} ({dt:.1f}s)", flush=True)
        print(f"  TRAIN ALL: [MSE: {trn_overall['mse']:7.1f}, MAE: {trn_overall['mae']:5.1f}pp, Median: {trn_overall['median_ae']:5.1f}pp, Std: {trn_overall['error_std']:5.1f}pp]", flush=True)
        print(f"        UNIQUE: MAE={trn_unique['mae']:5.1f}pp, Med={trn_unique['median_ae']:5.1f}pp  |  MULTI: MAE={trn_degen['mae']:5.1f}pp, Med={trn_degen['median_ae']:5.1f}pp", flush=True)

        print(f"\n  TEST TEMPORAL: [MSE: {tmp_overall['mse']:7.1f}, MAE: {tmp_overall['mae']:5.1f}pp, Median: {tmp_overall['median_ae']:5.1f}pp, Std: {tmp_overall['error_std']:5.1f}pp]", flush=True)
        print(f"        UNIQUE: MAE={tmp_unique['mae']:5.1f}pp, Med={tmp_unique['median_ae']:5.1f}pp  |  MULTI: MAE={tmp_degen['mae']:5.1f}pp, Med={tmp_degen['median_ae']:5.1f}pp", flush=True)

        print(f"  TEST SPATIAL:  [MSE: {spt_overall['mse']:7.1f}, MAE: {spt_overall['mae']:5.1f}pp, Median: {spt_overall['median_ae']:5.1f}pp, Std: {spt_overall['error_std']:5.1f}pp]", flush=True)
        print(f"        UNIQUE: MAE={spt_unique['mae']:5.1f}pp, Med={spt_unique['median_ae']:5.1f}pp  |  MULTI: MAE={spt_degen['mae']:5.1f}pp, Med={spt_degen['median_ae']:5.1f}pp", flush=True)

        print(f"  TEST RANDOM:   [MSE: {rnd_overall['mse']:7.1f}, MAE: {rnd_overall['mae']:5.1f}pp, Median: {rnd_overall['median_ae']:5.1f}pp, Std: {rnd_overall['error_std']:5.1f}pp]", flush=True)
        print(f"        UNIQUE: MAE={rnd_unique['mae']:5.1f}pp, Med={rnd_unique['median_ae']:5.1f}pp  |  MULTI: MAE={rnd_degen['mae']:5.1f}pp, Med={rnd_degen['median_ae']:5.1f}pp", flush=True)

        print_predictions_table(tmp_gt, tmp_pred, tmp_types, spt_gt, spt_pred, spt_types, rnd_gt, rnd_pred, rnd_types)

        # LR decay
        for g in optimizer.param_groups:
            g['lr'] *= 0.999
        print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.6f}", flush=True)

    print("="*80, flush=True)

    # Save model
    final_model_path = output_dir / 'final_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_species': dataset.n_species,
        'species_dim': args.species_dim,
        'species_to_idx': dataset.species_to_idx,
        'idx_to_species': dataset.idx_to_species,
        'use_aef': args.use_aef,
        'use_daymet': args.use_daymet,
        'aef_dim': dataset.aef_dim,
        'daymet_dim': dataset.daymet_dim
    }, final_model_path)
    print(f"\nFinal model saved to: {final_model_path}", flush=True)

    # Save metrics
    metrics_df = pd.DataFrame(metrics_history)
    metrics_path = output_dir / f'training_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}", flush=True)

    # Print final summary
    final = metrics_history[-1]
    print(f"\nFINAL RESULTS (Epoch {args.epochs}):", flush=True)
    print(f"  Overall Performance:", flush=True)
    print(f"    Training:      MAE={final['train_mae']:.1f}pp, Median={final['train_median_ae']:.1f}pp, Std={final['train_error_std']:.1f}pp", flush=True)
    print(f"    Temporal Test: MAE={final['temporal_mae']:.1f}pp, Median={final['temporal_median_ae']:.1f}pp, Std={final['temporal_error_std']:.1f}pp", flush=True)
    print(f"    Spatial Test:  MAE={final['spatial_mae']:.1f}pp, Median={final['spatial_median_ae']:.1f}pp, Std={final['spatial_error_std']:.1f}pp", flush=True)
    print(f"    Random Test:   MAE={final['random_mae']:.1f}pp, Median={final['random_median_ae']:.1f}pp, Std={final['random_error_std']:.1f}pp", flush=True)

    print("\nTraining complete!", flush=True)

    return final


def main():
    parser = argparse.ArgumentParser(description="Earth4D LFMC Prediction with Modular Features")
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory containing Parquet files (default: ./data)')
    parser.add_argument('--epochs', type=int, default=500,
                       help='Number of training epochs (default: 500)')
    parser.add_argument('--batch-size', type=int, default=30000,
                       help='Batch size (default: 30000)')
    parser.add_argument('--lr', type=float, default=0.03,
                       help='Learning rate (default: 0.03)')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Output directory (default: ./outputs)')
    parser.add_argument('--species-dim', type=int, default=768,
                       help='Dimension of learnable species embeddings (default: 768)')
    parser.add_argument('--use-earth4d', action='store_true', default=True,
                       help='Use Earth4D spatiotemporal features (default: True)')
    parser.add_argument('--no-earth4d', dest='use_earth4d', action='store_false',
                       help='Disable Earth4D features')
    parser.add_argument('--use-species', action='store_true', default=True,
                       help='Use species embeddings (default: True)')
    parser.add_argument('--no-species', dest='use_species', action='store_false',
                       help='Disable species embeddings')
    parser.add_argument('--use-bioclip', action='store_true',
                       help='Use BioCLIP 2 embeddings instead of learnable')
    parser.add_argument('--freeze-embeddings', action='store_true',
                       help='Freeze species embeddings (no training)')
    parser.add_argument('--use-aef', action='store_true',
                       help='Use AlphaEarth Features (64D)')
    parser.add_argument('--use-daymet', action='store_true',
                       help='Use Daymet weather features')
    parser.add_argument('--auto-download', action='store_true', default=True,
                       help='Auto-download missing datasets from cloud (default: True)')
    parser.add_argument('--no-auto-download', dest='auto_download', action='store_false',
                       help='Disable auto-download')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed (default: 0)')
    args = parser.parse_args()

    device = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = True

    # Run training
    run_training_session(args)


if __name__ == "__main__":
    main()
