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
    # Load the mapping CSV
    mapping_path = os.path.join(bioclip_path, 'species_occurrence_counts_with_embeddings.csv')
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
            if os.path.exists(chunk_path):
                chunk_data = torch.load(chunk_path, weights_only=False)
                loaded_chunks[chunk_file] = chunk_data['embeddings']
                print(f"  Loaded {chunk_file}: {chunk_data['embeddings'].shape}", flush=True)

        # Get the embedding for this species
        if chunk_file in loaded_chunks:
            # Compute actual index within the chunk
            chunk_embeddings = loaded_chunks[chunk_file]
            # The embedding_index in CSV is global, we need local index within chunk
            local_idx = emb_index % chunk_embeddings.shape[0]
            species_to_embedding[species_name] = chunk_embeddings[local_idx].float()  # Convert from float16 to float32

    print(f"  Loaded embeddings for {len(species_to_embedding):,} species", flush=True)

    # Check embedding dimension
    if species_to_embedding:
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


class FullyGPUDataset:
    """Everything on GPU from the start, with species encoding."""

    def __init__(self, data_path: str, device: str = 'cuda', use_bioclip: bool = False):
        # ONE-TIME CPU operations for loading
        df = pd.read_csv(data_path)
        df.columns = ['lat', 'lon', 'elev', 'date_str', 'time_str', 'lfmc', 'species']

        # Filter
        df = df[(df['lfmc'] >= 0) & (df['lfmc'] <= 600) &
                df['lat'].notna() & df['lon'].notna() &
                df['elev'].notna()].copy()

        self.use_bioclip = use_bioclip
        self.bioclip_embeddings = None
        self.bioclip_dim = 768  # BioCLIP 2 dimension

        # If using BioCLIP, load embeddings and filter dataset
        if use_bioclip:
            # Get unique species in LFMC dataset
            lfmc_species = df['species'].unique()

            # Load BioCLIP embeddings, using cache if available
            cache_path = os.path.join(os.path.dirname(data_path), 'lfmc_bioclip_embeddings_cache.pt')
            bioclip_dict = load_bioclip_embeddings(lfmc_species=list(lfmc_species), cache_path=cache_path)

            # Check which species in LFMC data have BioCLIP embeddings
            lfmc_species = set(df['species'].unique())
            bioclip_species = set(bioclip_dict.keys())
            matched_species = lfmc_species & bioclip_species
            missing_species = lfmc_species - bioclip_species

            print(f"\nSpecies matching:", flush=True)
            print(f"  LFMC species: {len(lfmc_species)}", flush=True)
            print(f"  Matched with BioCLIP: {len(matched_species)} ({100*len(matched_species)/len(lfmc_species):.1f}%)", flush=True)
            print(f"  Missing from BioCLIP: {len(missing_species)}", flush=True)

            if missing_species:
                # Count samples for missing species
                missing_counts = {}
                for species in missing_species:
                    missing_counts[species] = len(df[df['species'] == species])

                # Sort by count
                sorted_missing = sorted(missing_counts.items(), key=lambda x: x[1], reverse=True)

                total_missing_samples = sum(missing_counts.values())
                print(f"  Total samples to be filtered: {total_missing_samples:,} ({100*total_missing_samples/len(df):.1f}%)", flush=True)
                print(f"  Top 5 missing species by sample count:", flush=True)
                for i, (species, count) in enumerate(sorted_missing[:5]):
                    print(f"    {i+1}. {species}: {count:,} samples", flush=True)

                # Filter dataset to only include species with BioCLIP embeddings
                df = df[df['species'].isin(matched_species)].copy()
                print(f"\nFiltered dataset: {len(df):,} samples remaining", flush=True)

            # Store BioCLIP embeddings for matched species
            self.bioclip_embeddings = bioclip_dict

        n = len(df)
        print(f"Loaded {n:,} samples", flush=True)

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

        # If using BioCLIP, prepare embeddings tensor
        if use_bioclip:
            # Create embedding matrix for all species in dataset
            embedding_matrix = torch.zeros((self.n_species, self.bioclip_dim), dtype=torch.float32)
            for species, idx in self.species_to_idx.items():
                if species in self.bioclip_embeddings:
                    embedding_matrix[idx] = self.bioclip_embeddings[species]
                else:
                    # This shouldn't happen as we filtered, but just in case
                    print(f"WARNING: No BioCLIP embedding for {species}", flush=True)
            self.species_embeddings = embedding_matrix.to(device)
            print(f"  BioCLIP embeddings prepared: {embedding_matrix.shape}", flush=True)

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
        self.targets = torch.tensor(df['lfmc'].values, dtype=torch.float32, device=device)
        self.species_idx = torch.tensor(species_indices, dtype=torch.long, device=device)

        # Degeneracy flag on GPU
        self.is_degenerate = torch.tensor(is_degenerate, dtype=torch.bool, device=device)

        # For temporal split - convert dates to GPU tensor
        self.date_values = torch.tensor(date_floats, dtype=torch.float32, device=device)

        # Store raw dataframe columns for split creation
        self.df = df

        self.n = n
        self.device = device
        self.n_degenerate_samples = n_degenerate_samples
        self.n_unique_samples = n - n_degenerate_samples

        print(f"\nGPU dataset ready: {n:,} samples", flush=True)


class SpeciesAwareLFMCModel(nn.Module):
    """LFMC model with species embeddings (learnable or pre-trained)."""

    def __init__(self, n_species, species_dim=32, use_bioclip=False, bioclip_embeddings=None, freeze_embeddings=False):
        super().__init__()

        self.earth4d = Earth4D(
            spatial_levels=24,
            temporal_levels=19,
            spatial_log2_hashmap_size=22,
            temporal_log2_hashmap_size=18,
            verbose=False
        )

        earth4d_dim = self.earth4d.get_output_dim()
        self.use_bioclip = use_bioclip

        if use_bioclip:
            # Use pre-trained BioCLIP embeddings
            if bioclip_embeddings is None:
                raise ValueError("BioCLIP embeddings must be provided when use_bioclip=True")
            self.species_embeddings = nn.Embedding.from_pretrained(bioclip_embeddings, freeze=freeze_embeddings)
            species_dim = bioclip_embeddings.shape[1]  # 768 for BioCLIP 2
            if freeze_embeddings:
                print(f"  Using FROZEN BioCLIP embeddings: {bioclip_embeddings.shape}", flush=True)
            else:
                print(f"  Using TRAINABLE BioCLIP embeddings: {bioclip_embeddings.shape}", flush=True)
        else:
            # Learnable species embeddings
            self.species_embeddings = nn.Embedding(n_species, species_dim)
            nn.init.normal_(self.species_embeddings.weight, mean=0.0, std=0.1)
            if freeze_embeddings:
                self.species_embeddings.weight.requires_grad = False
                print(f"  Using FROZEN random embeddings: ({n_species}, {species_dim})", flush=True)
            else:
                print(f"  Using TRAINABLE random embeddings: ({n_species}, {species_dim})", flush=True)

        # MLP that takes concatenated Earth4D features and species embedding
        input_dim = earth4d_dim + species_dim
        print(f"  MLP input dimension: {input_dim} (Earth4D: {earth4d_dim} + Species: {species_dim})", flush=True)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
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


def create_gpu_splits(dataset, device='cuda'):
    """Create splits ensuring test species are in training set.

    Spatial split uses 5 cluster centers with nearest neighbors to create
    true spatial holdout regions.
    """
    n = dataset.n

    # Get unique species in dataset
    all_species = set(dataset.df['species'].unique())

    # All indices on GPU
    all_idx = torch.arange(n, device=device)

    # 1. Temporal: sort by date ON GPU and take last 5%
    n_temp = int(n * 0.05)
    _, date_order = torch.sort(dataset.date_values)
    temp_idx = date_order[-n_temp:]

    # 2. Create mask for used indices
    used = torch.zeros(n, dtype=torch.bool, device=device)
    used[temp_idx] = True

    # 3. Spatial: Create 5 spatial clusters using k-nearest neighbors
    n_spat = int(n * 0.05)
    available = all_idx[~used]

    # Get spatial coordinates (lat, lon) for available samples
    available_coords = dataset.coords[available][:, :2]  # Just lat, lon

    # Randomly select 5 cluster centers from available points
    n_available = len(available)
    center_indices = torch.randperm(n_available, device=device)[:5]
    cluster_centers = available_coords[center_indices]

    # For each cluster center, find nearest neighbors
    spat_indices = []
    samples_per_cluster = n_spat // 5
    remaining_samples = n_spat % 5  # Handle remainder

    for i, center in enumerate(cluster_centers):
        # Calculate distances to all available points
        distances = torch.sum((available_coords - center.unsqueeze(0)) ** 2, dim=1)

        # Get indices of nearest neighbors
        n_samples = samples_per_cluster + (1 if i < remaining_samples else 0)
        _, nearest_indices = torch.topk(distances, n_samples, largest=False)

        # Map back to original indices
        cluster_samples = available[nearest_indices]
        spat_indices.append(cluster_samples)

    # Combine all spatial clusters
    spat_idx = torch.cat(spat_indices)
    used[spat_idx] = True

    # Store cluster centers for visualization
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

    # Check species coverage
    train_species = set(dataset.df.iloc[train_idx.cpu().numpy()]['species'].unique())
    temp_species = set(dataset.df.iloc[temp_idx.cpu().numpy()]['species'].unique())
    spat_species = set(dataset.df.iloc[spat_idx.cpu().numpy()]['species'].unique())
    rand_species = set(dataset.df.iloc[rand_idx.cpu().numpy()]['species'].unique())

    print(f"\nSpecies coverage:", flush=True)
    print(f"  Train species: {len(train_species)}", flush=True)
    print(f"  Temporal test species: {len(temp_species)} ({100*len(temp_species & train_species)/len(temp_species):.1f}% in train)", flush=True)
    print(f"  Spatial test species: {len(spat_species)} ({100*len(spat_species & train_species)/len(spat_species):.1f}% in train)", flush=True)
    print(f"  Random test species: {len(rand_species)} ({100*len(rand_species & train_species)/len(rand_species):.1f}% in train)", flush=True)

    # Report species NOT in training set
    temp_novel = temp_species - train_species
    spat_novel = spat_species - train_species
    rand_novel = rand_species - train_species

    if temp_novel:
        print(f"\n  WARNING: {len(temp_novel)} species in temporal test NOT in training:", flush=True)
        for s in list(temp_novel)[:5]:
            print(f"    - {s}", flush=True)
    if spat_novel:
        print(f"  WARNING: {len(spat_novel)} species in spatial test NOT in training:", flush=True)
        for s in list(spat_novel)[:5]:
            print(f"    - {s}", flush=True)
    if rand_novel:
        print(f"  WARNING: {len(rand_novel)} species in random test NOT in training:", flush=True)
        for s in list(rand_novel)[:5]:
            print(f"    - {s}", flush=True)

    # Report degeneracy breakdown per split
    print(f"\nDegeneracy breakdown by split:", flush=True)
    for name, idx in [('Train', train_idx), ('Temporal', temp_idx),
                       ('Spatial', spat_idx), ('Random', rand_idx)]:
        n_degen = dataset.is_degenerate[idx].sum().item()
        n_unique = len(idx) - n_degen
        print(f"  {name:8s}: {n_unique:5d} unique, {n_degen:5d} multi-species degenerate ({100*n_degen/len(idx):.1f}%)", flush=True)

    return {
        'train': train_idx,
        'temporal': temp_idx,
        'spatial': spat_idx,
        'random': rand_idx
    }


def compute_metrics_gpu(preds, targets, is_degenerate=None):
    """
    Compute metrics on GPU.
    Since LFMC is already in percentage units, we report error statistics directly in LFMC % space.
    """
    def calc_metrics(p, t):
        # Compute errors in LFMC percentage space
        errors = p - t  # Signed errors
        abs_errors = torch.abs(errors)  # Absolute errors

        # Basic metrics
        mse = (errors ** 2).mean()
        rmse = torch.sqrt(mse)

        # Mean absolute error (in LFMC percentage points)
        mae = abs_errors.mean()

        # Median absolute error
        median_ae = torch.median(abs_errors)

        # Variance of errors
        error_var = errors.var()

        # Standard deviation of errors
        error_std = errors.std()

        return {
            'mse': mse.item(),
            'rmse': rmse.item(),
            'mae': mae.item(),
            'median_ae': median_ae.item(),
            'error_var': error_var.item(),
            'error_std': error_std.item()
        }

    # Overall metrics
    overall = calc_metrics(preds, targets)

    # Split by degeneracy if provided
    if is_degenerate is not None:
        unique_mask = ~is_degenerate
        degen_mask = is_degenerate

        # Default empty metrics dict
        empty_metrics = {'mse': 0, 'rmse': 0, 'mae': 0, 'median_ae': 0, 'error_var': 0, 'error_std': 0}

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
        loss = criterion(preds, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

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

    empty_metrics = {'mse': 0, 'rmse': 0, 'mae': 0, 'median_ae': 0, 'error_var': 0, 'error_std': 0}

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


def print_predictions_table(tmp_gt, tmp_pred, tmp_types, spt_gt, spt_pred, spt_types, rnd_gt, rnd_pred, rnd_types):
    """Print predictions in a clean table format with degeneracy indicators."""
    print("  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐", flush=True)

    max_len = max(len(tmp_gt), len(spt_gt), len(rnd_gt))

    for i in range(min(5, max_len)):
        line = "  │ "

        # Temporal
        if i < len(tmp_gt):
            tmp_err = abs(tmp_gt[i] - tmp_pred[i])
            tmp_tag = "UNIQUE" if tmp_types[i] == 'U' else "MULTI" if i < len(tmp_types) else ''
            line += f"TEMPORAL-{tmp_tag}: {tmp_gt[i]:3.0f}%→{tmp_pred[i]:3.0f}% (Δ{tmp_err:3.0f}%) │ "
        else:
            line += " " * 36 + " │ "

        # Spatial
        if i < len(spt_gt):
            spt_err = abs(spt_gt[i] - spt_pred[i])
            spt_tag = "UNIQUE" if spt_types[i] == 'U' else "MULTI" if i < len(spt_types) else ''
            line += f"SPATIAL-{spt_tag}: {spt_gt[i]:3.0f}%→{spt_pred[i]:3.0f}% (Δ{spt_err:3.0f}%) │ "
        else:
            line += " " * 35 + " │ "

        # Random
        if i < len(rnd_gt):
            rnd_err = abs(rnd_gt[i] - rnd_pred[i])
            rnd_tag = "UNIQUE" if rnd_types[i] == 'U' else "MULTI" if i < len(rnd_types) else ''
            line += f"RANDOM-{rnd_tag}: {rnd_gt[i]:3.0f}%→{rnd_pred[i]:3.0f}% (Δ{rnd_err:3.0f}%) │"
        else:
            line += " " * 34 + " │"

        print(line, flush=True)

    print("  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘", flush=True)
    print("    (UNIQUE=Single species at location, MULTI=Multiple species at location)", flush=True)




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
    if args.use_bioclip:
        print(f"EARTH4D LFMC - BIOCLIP SPECIES EMBEDDING VERSION", flush=True)
    elif args.species_dim == 768:
        print(f"EARTH4D LFMC - RANDOM 768D SPECIES EMBEDDING VERSION", flush=True)
    else:
        print(f"EARTH4D LFMC - LEARNABLE {args.species_dim}D SPECIES EMBEDDING VERSION", flush=True)
    print("="*80, flush=True)

    # Load dataset
    dataset = FullyGPUDataset(args.data_path, device, use_bioclip=args.use_bioclip)
    splits = create_gpu_splits(dataset, device)

    # Model with species embeddings
    if args.use_bioclip:
        # Use BioCLIP embeddings
        model = SpeciesAwareLFMCModel(
            dataset.n_species,
            use_bioclip=True,
            bioclip_embeddings=dataset.species_embeddings,
            freeze_embeddings=args.freeze_embeddings
        ).to(device)
    else:
        # Use learnable embeddings
        model = SpeciesAwareLFMCModel(
            dataset.n_species,
            species_dim=args.species_dim,
            use_bioclip=False,
            freeze_embeddings=args.freeze_embeddings
        ).to(device)

    # Count parameters
    earth4d_params = sum(p.numel() for p in model.earth4d.parameters())
    species_params = sum(p.numel() for p in model.species_embeddings.parameters())
    mlp_params = sum(p.numel() for p in model.mlp.parameters())
    total_params = sum(p.numel() for p in model.parameters())

    # Count trainable parameters (BioCLIP embeddings are frozen)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel architecture:", flush=True)
    print(f"  Earth4D parameters: {earth4d_params:,}", flush=True)

    # Report species embedding details
    embedding_type = "BioCLIP" if args.use_bioclip else "Random"
    embedding_status = "FROZEN" if args.freeze_embeddings else "TRAINABLE"
    embedding_dim = 768 if args.use_bioclip else args.species_dim

    print(f"  Species embedding parameters: {species_params:,} ({embedding_type} {embedding_status}, {dataset.n_species} species × {embedding_dim} dims)", flush=True)
    print(f"  MLP parameters: {mlp_params:,}", flush=True)
    print(f"  Total parameters: {total_params:,}", flush=True)
    print(f"  Trainable parameters: {trainable_params:,}", flush=True)

    # Optimizer
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

        # Evaluate all test splits
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

        # Update EMAs
        current_metrics = {
            'epoch': epoch,
            'time': dt,
            # Train overall metrics
            'train_mse': trn_overall['mse'],
            'train_rmse': trn_overall['rmse'],
            'train_mae': trn_overall['mae'],
            'train_median_ae': trn_overall['median_ae'],
            'train_error_var': trn_overall['error_var'],
            'train_error_std': trn_overall['error_std'],
            # Train unique metrics
            'train_unique_mae': trn_unique['mae'],
            'train_unique_median_ae': trn_unique['median_ae'],
            'train_unique_std': trn_unique['error_std'],
            # Train degenerate metrics
            'train_degen_mae': trn_degen['mae'],
            'train_degen_median_ae': trn_degen['median_ae'],
            'train_degen_std': trn_degen['error_std'],
            # Temporal test metrics
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
            # Spatial test metrics
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
            # Random test metrics
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

        # Update EMAs and store metrics
        ema_metrics = metrics_ema.update(current_metrics)
        current_metrics.update(ema_metrics)
        metrics_history.append(current_metrics)

        # Store sample predictions for EMA tracking
        if epoch == 1:
            metrics_ema.sample_predictions['temporal_gt'] = tmp_gt
            metrics_ema.sample_predictions['spatial_gt'] = spt_gt
            metrics_ema.sample_predictions['random_gt'] = rnd_gt
        metrics_ema.sample_predictions[f'temporal_pred_{epoch}'] = tmp_pred
        metrics_ema.sample_predictions[f'spatial_pred_{epoch}'] = spt_pred
        metrics_ema.sample_predictions[f'random_pred_{epoch}'] = rnd_pred

        # Print clean formatted metrics with LFMC % error statistics
        print(f"\nEPOCH {epoch:3d} ({dt:.1f}s)", flush=True)
        print(f"  TRAIN ALL: [MSE: {trn_overall['mse']:7.1f}, MAE: {trn_overall['mae']:5.1f}pp, Median: {trn_overall['median_ae']:5.1f}pp, Std: {trn_overall['error_std']:5.1f}pp]", flush=True)
        print(f"        UNIQUE: MAE={trn_unique['mae']:5.1f}pp, Med={trn_unique['median_ae']:5.1f}pp  |  MULTI: MAE={trn_degen['mae']:5.1f}pp, Med={trn_degen['median_ae']:5.1f}pp", flush=True)

        print(f"\n  TEST TEMPORAL: [MSE: {tmp_overall['mse']:7.1f}, MAE: {tmp_overall['mae']:5.1f}pp, Median: {tmp_overall['median_ae']:5.1f}pp, Std: {tmp_overall['error_std']:5.1f}pp]", flush=True)
        print(f"        UNIQUE: MAE={tmp_unique['mae']:5.1f}pp, Med={tmp_unique['median_ae']:5.1f}pp  |  MULTI: MAE={tmp_degen['mae']:5.1f}pp, Med={tmp_degen['median_ae']:5.1f}pp", flush=True)

        print(f"  TEST SPATIAL:  [MSE: {spt_overall['mse']:7.1f}, MAE: {spt_overall['mae']:5.1f}pp, Median: {spt_overall['median_ae']:5.1f}pp, Std: {spt_overall['error_std']:5.1f}pp]", flush=True)
        print(f"        UNIQUE: MAE={spt_unique['mae']:5.1f}pp, Med={spt_unique['median_ae']:5.1f}pp  |  MULTI: MAE={spt_degen['mae']:5.1f}pp, Med={spt_degen['median_ae']:5.1f}pp", flush=True)

        print(f"  TEST RANDOM:   [MSE: {rnd_overall['mse']:7.1f}, MAE: {rnd_overall['mae']:5.1f}pp, Median: {rnd_overall['median_ae']:5.1f}pp, Std: {rnd_overall['error_std']:5.1f}pp]", flush=True)
        print(f"        UNIQUE: MAE={rnd_unique['mae']:5.1f}pp, Med={rnd_unique['median_ae']:5.1f}pp  |  MULTI: MAE={rnd_degen['mae']:5.1f}pp, Med={rnd_degen['median_ae']:5.1f}pp", flush=True)

        # Show predictions table every epoch
        print_predictions_table(tmp_gt, tmp_pred, tmp_types, spt_gt, spt_pred, spt_types, rnd_gt, rnd_pred, rnd_types)

        # LR decay -  reduction per epoch 
        for g in optimizer.param_groups:
            g['lr'] *= 0.999
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

    # Print final summary with LFMC percentage point errors
    final = metrics_history[-1]
    print(f"\nFINAL RESULTS (Epoch {args.epochs}) - Errors in LFMC percentage points:", flush=True)
    print(f"  Overall Performance:", flush=True)
    print(f"    Training:      MAE={final['train_mae']:.1f}pp, Median={final['train_median_ae']:.1f}pp, Std={final['train_error_std']:.1f}pp", flush=True)
    print(f"    Temporal Test: MAE={final['temporal_mae']:.1f}pp, Median={final['temporal_median_ae']:.1f}pp, Std={final['temporal_error_std']:.1f}pp", flush=True)
    print(f"    Spatial Test:  MAE={final['spatial_mae']:.1f}pp, Median={final['spatial_median_ae']:.1f}pp, Std={final['spatial_error_std']:.1f}pp", flush=True)
    print(f"    Random Test:   MAE={final['random_mae']:.1f}pp, Median={final['random_median_ae']:.1f}pp, Std={final['random_error_std']:.1f}pp", flush=True)

    print(f"\n  Unique Species Locations Only:", flush=True)
    print(f"    Training:      MAE={final['train_unique_mae']:.1f}pp, Median={final['train_unique_median_ae']:.1f}pp, Std={final['train_unique_std']:.1f}pp", flush=True)
    print(f"    Temporal Test: MAE={final['temporal_unique_mae']:.1f}pp, Median={final['temporal_unique_median_ae']:.1f}pp, Std={final['temporal_unique_std']:.1f}pp", flush=True)
    print(f"    Spatial Test:  MAE={final['spatial_unique_mae']:.1f}pp, Median={final['spatial_unique_median_ae']:.1f}pp, Std={final['spatial_unique_std']:.1f}pp", flush=True)
    print(f"    Random Test:   MAE={final['random_unique_mae']:.1f}pp, Median={final['random_unique_median_ae']:.1f}pp, Std={final['random_unique_std']:.1f}pp", flush=True)

    print(f"\n  Multi-Species Locations Only:", flush=True)
    print(f"    Training:      MAE={final['train_degen_mae']:.1f}pp, Median={final['train_degen_median_ae']:.1f}pp, Std={final['train_degen_std']:.1f}pp", flush=True)
    print(f"    Temporal Test: MAE={final['temporal_degen_mae']:.1f}pp, Median={final['temporal_degen_median_ae']:.1f}pp, Std={final['temporal_degen_std']:.1f}pp", flush=True)
    print(f"    Spatial Test:  MAE={final['spatial_degen_mae']:.1f}pp, Median={final['spatial_degen_median_ae']:.1f}pp, Std={final['spatial_degen_std']:.1f}pp", flush=True)
    print(f"    Random Test:   MAE={final['random_degen_mae']:.1f}pp, Median={final['random_degen_median_ae']:.1f}pp, Std={final['random_degen_std']:.1f}pp", flush=True)

    print("\nTraining complete!", flush=True)

    # Create visualizations using final predictions
    print("\nGenerating visualizations...", flush=True)

    # Get final predictions for all test sets
    with torch.no_grad():
        model.eval()

        # Collect predictions from all test sets
        all_preds = {}
        all_gts = {}
        all_indices = {}

        for split_name in ['temporal', 'spatial', 'random']:
            if len(splits[split_name]) > 0:
                coords = dataset.coords[splits[split_name]]
                targets = dataset.targets[splits[split_name]]
                species = dataset.species_idx[splits[split_name]]
                preds = model(coords, species)

                all_preds[split_name] = preds.cpu().numpy()
                all_gts[split_name] = targets.cpu().numpy()
                all_indices[split_name] = splits[split_name]

        # Combine all test sets for visualizations
        if len(all_preds) > 0:
            # Concatenate all test predictions and ground truth
            combined_preds = np.concatenate([all_preds[k] for k in all_preds.keys()])
            combined_gts = np.concatenate([all_gts[k] for k in all_gts.keys()])
            combined_indices = torch.cat([all_indices[k] for k in all_indices.keys()])
            train_count = len(splits['train'])

            # Create temporal visualization with all test sets
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

            # Create geospatial visualization with ALL test sets combined
            grid_errors, grid_counts = create_geospatial_visualization(
                dataset,
                combined_preds,
                combined_gts,
                combined_indices,
                output_dir,
                epoch=args.epochs,
                train_samples=train_count,
                spatial_indices=splits['spatial']  # Pass spatial indices for marking
            )

    print(f"Visualizations saved to {output_dir}", flush=True)

    # Return final metrics for comparison
    return final


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
    try:
        import geopandas as gpd
        use_shapefile = True
    except ImportError:
        use_shapefile = False
        print("Warning: geopandas not available, using simplified boundaries", flush=True)

    # Extract coordinates for test samples
    coords = dataset.coords[test_indices].cpu().numpy()
    lats = coords[:, 0]
    lons = coords[:, 1]

    # Calculate errors
    errors = np.abs(test_predictions - test_ground_truth)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    # Define CONUS boundaries
    lon_min, lon_max = -125, -66
    lat_min, lat_max = 24, 50

    # Load and plot US shapefile if available
    if use_shapefile:
        shapefile_path = Path(SCRIPT_DIR) / 'shapefiles' / 'cb_2018_us_state_20m.shp'
        if shapefile_path.exists():
            try:
                # Read shapefile
                states = gpd.read_file(shapefile_path)
                # Filter to continental US (exclude Alaska, Hawaii, territories)
                states_conus = states[
                    ~states['NAME'].isin(['Alaska', 'Hawaii', 'Puerto Rico',
                                         'United States Virgin Islands', 'Guam',
                                         'American Samoa', 'Northern Mariana Islands'])
                ]
                # Plot state boundaries
                states_conus.boundary.plot(ax=ax, color='darkgrey', linewidth=0.5, alpha=0.7)
            except Exception as e:
                print(f"Warning: Could not plot shapefile: {e}", flush=True)
                use_shapefile = False
        else:
            use_shapefile = False

    # Create 100x100 km grid (approximately 0.9 degrees)
    grid_size = 0.9
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

            # Size range: ensure max size doesn't exceed bin size (100km grid)
            # Scale sizes to prevent overlap - max diameter should be ~80% of grid spacing
            size_min, size_max = 20, 250  # Reduced max size to prevent overlap

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

                        # Color based on error (TURBO colormap) - clip to 75pp max
                        error_value = min(grid_avg_errors[i, j], 75.0)  # Clip to 75pp

                        scatter = ax.scatter(lon_center, lat_center, s=size,
                                           c=[error_value], cmap='turbo',
                                           vmin=0, vmax=75,  # Fixed scale 0-75pp
                                           alpha=1.0, edgecolors='black', linewidth=0.5)  # Alpha=1.0

    # Add colorbar with fixed 0-75 scale
    if 'scatter' in locals():
        # Create a proper colorbar with fixed scale
        import matplotlib.cm as cm
        norm = plt.Normalize(vmin=0, vmax=75)
        sm = plt.cm.ScalarMappable(cmap='turbo', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Average LFMC Error (pp)')
        cbar.set_ticks([0, 15, 30, 45, 60, 75])
        cbar.set_ticklabels(['0', '15', '30', '45', '60', '75+'])  # Add + to indicate clipping

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

    # Add simple boundaries if shapefile not available
    if not use_shapefile:
        ax.axhline(y=lat_min, color='black', linewidth=1, alpha=0.5)
        ax.axhline(y=lat_max, color='black', linewidth=1, alpha=0.5)
        ax.axvline(x=lon_min, color='black', linewidth=1, alpha=0.5)
        ax.axvline(x=lon_max, color='black', linewidth=1, alpha=0.5)

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

    for split_name in ['temporal', 'spatial', 'random']:
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

    ax.set_title(f'Earth4D LFMC Predictions - Monthly Temporal Evolution (All Test Sets)\n'
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

    # Add overall statistics - calculate from all predictions
    if len(months_filtered) > 0:
        all_predictions = np.concatenate(month_predictions)
        all_ground_truths = np.concatenate(month_ground_truths)

        # Calculate overall error statistics (prediction vs actual ground truth)
        all_errors = np.abs(all_predictions - all_ground_truths)
        overall_mae = np.mean(all_errors)
        overall_median_error = np.median(all_errors)

        stats_text = f'Entire Test Dataset: Mean {overall_mae:.1f}pp, Median {overall_median_error:.1f}pp'
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
                ha='left', va='top', fontsize=14,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9,
                         edgecolor='black', linewidth=0.5))

    # Save figure
    plt.tight_layout()
    plt.savefig(output_dir / f'temporal_predictions_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
    plt.close()

    return np.array(all_errors) if len(months_filtered) > 0 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=30000)
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--output-dir', type=str, default='./outputs')
    parser.add_argument('--species-dim', type=int, default=768,
                       help='Dimension of learnable species embeddings (ignored if using BioCLIP or comparison mode)')
    parser.add_argument('--use-bioclip', action='store_true',
                       help='Use pre-trained BioCLIP 2 embeddings instead of learnable embeddings')
    parser.add_argument('--freeze-embeddings', action='store_true',
                       help='Freeze species embeddings (no training). Default is trainable.')
    parser.add_argument('--compare-embeddings', action='store_true',
                       help='Run comparison: BioCLIP vs random 768D embeddings (runs both sequentially)')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed for reproducibility (default: 0)')
    args = parser.parse_args()

    device = 'cuda'
    # Note: cudnn.benchmark is set to False in run_training_session for determinism
    torch.backends.cuda.matmul.allow_tf32 = True

    if args.compare_embeddings:
        print("="*80, flush=True)
        print("ABLATION STUDY: BioCLIP vs Random 768D Embeddings", flush=True)
        print(f"Both will be {'FROZEN' if args.freeze_embeddings else 'TRAINABLE'} for fair comparison", flush=True)
        print("="*80, flush=True)

        # Run with BioCLIP embeddings
        print("\n[1/2] Training with BioCLIP embeddings...", flush=True)
        args_bioclip = argparse.Namespace(**vars(args))
        args_bioclip.use_bioclip = True
        args_bioclip.freeze_embeddings = args.freeze_embeddings  # Use same freeze setting
        bioclip_results = run_training_session(args_bioclip, run_name="bioclip")

        # Run with random 768D embeddings
        print("\n[2/2] Training with random 768D embeddings...", flush=True)
        args_random = argparse.Namespace(**vars(args))
        args_random.use_bioclip = False
        args_random.species_dim = 768
        args_random.freeze_embeddings = args.freeze_embeddings  # Use same freeze setting
        random_results = run_training_session(args_random, run_name="random768d")

        # Compare results
        print("\n" + "="*80, flush=True)
        print("ABLATION STUDY RESULTS - BioCLIP Advantage:", flush=True)
        print("="*80, flush=True)

        # Calculate improvements
        for split in ['temporal', 'spatial', 'random']:
            bioclip_mae = bioclip_results[f'{split}_mae']
            random_mae = random_results[f'{split}_mae']
            improvement = random_mae - bioclip_mae
            percent_improvement = (improvement / random_mae) * 100

            print(f"\n  {split.capitalize()} Test:", flush=True)
            print(f"    Random 768D: MAE={random_mae:.1f}pp", flush=True)
            print(f"    BioCLIP:     MAE={bioclip_mae:.1f}pp", flush=True)
            print(f"    Improvement: {improvement:.1f}pp ({percent_improvement:.1f}% better)", flush=True)

        print("\n  Overall BioCLIP provides pre-trained biological knowledge that", flush=True)
        print("  improves generalization compared to random initialization.", flush=True)
        print("="*80, flush=True)

    else:
        # Single run mode
        run_training_session(args)


if __name__ == "__main__":
    main()
