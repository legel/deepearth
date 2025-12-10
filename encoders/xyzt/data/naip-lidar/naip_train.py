#!/usr/bin/env python3
"""
NAIP-3DEP RGB Reconstruction Training with Earth4D
====================================================

Trains Earth4D to reconstruct NAIP RGB imagery from (latitude, longitude, elevation, timestamp).
Uses learned hash probing (state-of-the-art) with hyperparameters tuned from Globe-LFMC benchmark.

Key Features:
- Float64 ECEF coordinate system for full geographic precision
- Learned hash probing enabled by default (state-of-the-art)
- Cross-chip batch sampling for improved generalization
- Batch-based training with verbose per-batch logging
- 95% train / 5% test split (configurable)
- MLP head (2 hidden layers x 256 units)
- Visualization chips forced to train set (memorization tracking)
- Visual PNG reconstructions saved periodically

Usage:
    # Default: 1000 batches, 10K pixels/batch, learned probing ON
    python naip_train.py --data-dir stanford

    # Track memorization of specific locations (forced to train set, visualized every --test-every batches)
    python naip_train.py --data-dir stanford \\
        --visualize-locations "Eucalyptus_Grove:37.436,-122.165" \\
                              "Sequoia_HQ:37.421,-122.212" \\
                              "Golf_Course:37.411,-122.231"

    # Full configuration
    python naip_train.py \\
        --data-dir stanford \\
        --num-batches 1000 \\
        --batch-size 10000 \\
        --lr 1e-3 \\
        --index-lr-multiplier 10.0 \\
        --probe-entropy-weight 0.5 \\
        --test-every 10 \\
        --probing-range 32

    # Disable learned probing (baseline)
    python naip_train.py --data-dir stanford --no-learned-probing

Author: Earth4D Team
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import argparse
import sys
import os
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from PIL import Image

# Add parent directory for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', '..'))

from earth4d import Earth4D
from naip_utils import latlon_to_utm


class ExponentialMovingAverage:
    """Track exponential moving average of metrics."""
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.ema = None

    def update(self, value):
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema
        return self.ema

    def get(self):
        return self.ema if self.ema is not None else 0.0


class MetricsEMA:
    """Track EMAs for all metrics."""
    def __init__(self, alpha=0.1):
        self.emas = defaultdict(lambda: ExponentialMovingAverage(alpha))

    def update(self, metrics_dict):
        ema_dict = {}
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                ema_dict[f"{key}_ema"] = self.emas[key].update(value)
        return ema_dict

    def get_all(self):
        return {key: ema.get() for key, ema in self.emas.items()}


class RGBReconstructionModel(nn.Module):
    """
    Earth4D + MLP for RGB reconstruction.

    Architecture:
    - Earth4D encoder (state-of-the-art spatiotemporal encoding)
    - 2 hidden layers: 256 -> 256 -> 3
    - ReLU activations, Sigmoid output
    """

    def __init__(self, enable_learned_probing=True, probing_range=32,
                 probe_entropy_weight=0.5, verbose=False,
                 spatial_levels=24, temporal_levels=24,
                 spatial_log2_hashmap_size=22, temporal_log2_hashmap_size=22):
        super().__init__()

        self.earth4d = Earth4D(
            verbose=verbose,
            enable_learned_probing=enable_learned_probing,
            probing_range=probing_range,
            probe_entropy_weight=probe_entropy_weight,
            spatial_levels=spatial_levels,
            temporal_levels=temporal_levels,
            spatial_log2_hashmap_size=spatial_log2_hashmap_size,
            temporal_log2_hashmap_size=temporal_log2_hashmap_size
        )

        earth4d_dim = self.earth4d.get_output_dim()

        # MLP head (2 hidden layers)
        self.rgb_head = nn.Sequential(
            nn.Linear(earth4d_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3),
            nn.Sigmoid()  # RGB in [0, 1]
        )

        # Initialize Earth4D parameters
        with torch.no_grad():
            for p in self.earth4d.parameters():
                if p.dim() > 1:
                    nn.init.uniform_(p, -0.1, 0.1)

    def forward(self, coords):
        """
        Forward pass.

        Args:
            coords: (B, 4) tensor of [lat, lon, elevation, timestamp]
                    Note: Earth4D handles ECEF conversion internally

        Returns:
            (B, 3) RGB predictions in [0, 1]
        """
        features = self.earth4d(coords)
        return self.rgb_head(features)


def find_closest_chip(lat, lon, chip_metadata):
    """
    Find the chip closest to a given lat/lon coordinate.

    Args:
        lat: Target latitude
        lon: Target longitude
        chip_metadata: List of chip metadata dicts with UTM coordinates

    Returns:
        Index of closest chip, distance in meters
    """
    # Convert target to UTM
    target_zone, target_x, target_y = latlon_to_utm(lat, lon)

    best_idx = None
    best_dist = float('inf')

    for i, chip in enumerate(chip_metadata):
        chip_zone = chip['utm_zone']
        chip_x = chip['x']
        chip_y = chip['y']

        # Only compare chips in same UTM zone
        if chip_zone != target_zone:
            continue

        # Euclidean distance in UTM coordinates (meters)
        dist = np.sqrt((chip_x - target_x)**2 + (chip_y - target_y)**2)

        if dist < best_dist:
            best_dist = dist
            best_idx = i

    return best_idx, best_dist


def parse_visualization_locations(location_strings):
    """
    Parse visualization location strings into (name, lat, lon) tuples.

    Format: "Name:lat,lon" e.g. "Eucalyptus_Grove:37.436,-122.165"

    Args:
        location_strings: List of location specification strings

    Returns:
        List of (name, lat, lon) tuples
    """
    locations = []
    for loc_str in location_strings:
        try:
            name, coords = loc_str.split(':')
            lat, lon = map(float, coords.split(','))
            locations.append((name, lat, lon))
        except ValueError:
            print(f"Warning: Could not parse location '{loc_str}', expected format 'Name:lat,lon'")
    return locations


class NAIPDataset:
    """
    GPU-resident NAIP-3DEP dataset with cross-chip sampling.

    Features:
    - All data on GPU from the start
    - Float64 coordinates for ECEF precision
    - Chip-level train/test splits
    - Cross-chip batch sampling
    - Per-chip data access for visualization
    - Named visualization chips (forced into train set for memorization tracking)
    """

    def __init__(self, data_path, metadata_path=None, device='cuda', train_ratio=0.95,
                 chip_split=True, seed=42, visualize_locations=None):
        """
        Load NAIP dataset.

        Args:
            data_path: Path to parsed_xyztrgb.pt
            metadata_path: Path to chip_metadata.json (auto-detected if None)
            device: GPU device
            train_ratio: Fraction for training (default 95%)
            chip_split: If True, split by chips; if False, random pixel split
            seed: Random seed for reproducibility
            visualize_locations: List of (name, lat, lon) tuples for visualization chips
                                 (forced into TRAIN set to track memorization)
        """
        print(f"Loading dataset from {data_path}...", flush=True)

        # Load data
        data_dict = torch.load(data_path, weights_only=False)
        data_tensor = data_dict['data']
        self.chip_sizes = data_dict['chip_sizes']
        self.n_chips = data_dict['n_chips']

        n_total = len(data_tensor)
        print(f"  Total pixels: {n_total:,}", flush=True)
        print(f"  Total chips: {self.n_chips}", flush=True)

        # Load chip metadata for location-based selection
        if metadata_path is None:
            metadata_path = Path(data_path).parent / 'chip_metadata.json'

        self.chip_metadata = None
        self.visualization_chips = {}  # Maps name -> chip_idx (for train set visualization)

        if metadata_path.exists():
            with open(metadata_path) as f:
                self.chip_metadata = json.load(f)
            print(f"  Loaded chip metadata: {len(self.chip_metadata)} chips", flush=True)

        # Separate coordinates and RGB
        # CRITICAL: Use float64 for coordinates to maintain ECEF precision
        coords = data_tensor[:, :4].to(dtype=torch.float64, device=device)
        rgb = data_tensor[:, 4:].to(dtype=torch.float32, device=device)

        self.coords = coords
        self.rgb = rgb
        self.device = device
        self.n_total = n_total

        # Compute chip boundaries (pixel index ranges for each chip)
        self.chip_boundaries = np.cumsum([0] + self.chip_sizes)

        # Create train/test split
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Find visualization chips (these will be forced into TRAIN set)
        forced_train_chips = set()
        if visualize_locations and self.chip_metadata:
            print(f"\nSelecting visualization chips (forced to TRAIN set):", flush=True)
            for name, lat, lon in visualize_locations:
                chip_idx, dist = find_closest_chip(lat, lon, self.chip_metadata)
                if chip_idx is not None:
                    forced_train_chips.add(chip_idx)
                    self.visualization_chips[name] = chip_idx
                    chip_meta = self.chip_metadata[chip_idx]
                    print(f"  {name}: chip {chip_idx} (dist={dist:.0f}m, "
                          f"UTM {chip_meta['utm_zone']} {chip_meta['x']},{chip_meta['y']})", flush=True)
                else:
                    print(f"  {name}: No chip found in same UTM zone", flush=True)

        if chip_split:
            # Split by chips, ensuring visualization chips go to train
            remaining_chips = [i for i in range(self.n_chips) if i not in forced_train_chips]
            np.random.shuffle(remaining_chips)

            # Calculate test/train split from remaining chips
            n_target_test = int(self.n_chips * (1 - train_ratio))
            n_test = min(n_target_test, len(remaining_chips))

            test_chips = set(remaining_chips[:n_test])
            train_chips = forced_train_chips | set(remaining_chips[n_test:])

            # Build index lists
            train_indices = []
            test_indices = []

            for i in range(self.n_chips):
                start, end = self.chip_boundaries[i], self.chip_boundaries[i + 1]
                if i in train_chips:
                    train_indices.extend(range(start, end))
                else:
                    test_indices.extend(range(start, end))

            self.train_indices = torch.tensor(train_indices, dtype=torch.long, device=device)
            self.test_indices = torch.tensor(test_indices, dtype=torch.long, device=device)
            self.train_chip_set = train_chips
            self.test_chip_set = test_chips

            print(f"\n  Chip-level split:", flush=True)
            print(f"    Train chips: {len(train_chips)} ({len(forced_train_chips)} visualization)", flush=True)
            print(f"    Test chips: {len(test_chips)}", flush=True)
        else:
            # Random pixel split
            all_indices = torch.randperm(n_total, device=device)
            n_train = int(n_total * train_ratio)

            self.train_indices = all_indices[:n_train]
            self.test_indices = all_indices[n_train:]
            self.train_chip_set = set()
            self.test_chip_set = set()

            print(f"  Random pixel split", flush=True)

        print(f"  Train pixels: {len(self.train_indices):,} ({100*len(self.train_indices)/n_total:.1f}%)", flush=True)
        print(f"  Test pixels: {len(self.test_indices):,} ({100*len(self.test_indices)/n_total:.1f}%)", flush=True)

        # Print coordinate ranges
        print(f"\nCoordinate ranges:", flush=True)
        print(f"  Latitude:  {coords[:, 0].min():.6f} to {coords[:, 0].max():.6f}", flush=True)
        print(f"  Longitude: {coords[:, 1].min():.6f} to {coords[:, 1].max():.6f}", flush=True)
        print(f"  Elevation: {coords[:, 2].min():.2f} to {coords[:, 2].max():.2f} m", flush=True)
        print(f"  Timestamp: {coords[:, 3].min():.6f} to {coords[:, 3].max():.6f}", flush=True)

        print(f"\nGPU dataset ready", flush=True)

    def sample_train_batch(self, batch_size):
        """Sample a random batch from training set (cross-chip)."""
        perm = torch.randperm(len(self.train_indices), device=self.device)[:batch_size]
        indices = self.train_indices[perm]
        return self.coords[indices], self.rgb[indices]

    def get_test_data(self):
        """Get full test set."""
        return self.coords[self.test_indices], self.rgb[self.test_indices]

    def get_chip_data(self, chip_idx):
        """
        Get all data for a specific chip.

        Args:
            chip_idx: Chip index (0 to n_chips-1)

        Returns:
            coords: (N, 4) tensor of [lat, lon, elev, time]
            rgb: (N, 3) tensor of RGB values
        """
        start = self.chip_boundaries[chip_idx]
        end = self.chip_boundaries[chip_idx + 1]
        return self.coords[start:end], self.rgb[start:end]

    def get_visualization_chip(self, name):
        """
        Get chip data for a named visualization location.

        Args:
            name: Location name (e.g., "Eucalyptus_Grove")

        Returns:
            chip_idx, coords, rgb or None if not found
        """
        if name not in self.visualization_chips:
            return None
        chip_idx = self.visualization_chips[name]
        coords, rgb = self.get_chip_data(chip_idx)
        return chip_idx, coords, rgb


@torch.no_grad()
def render_chip_reconstruction(model, dataset, chip_idx, name=None, output_path=None):
    """
    Render a full chip reconstruction as side-by-side ground truth vs prediction.

    Args:
        model: Trained model
        dataset: NAIPDataset instance
        chip_idx: Index of chip to render
        name: Optional name for the chip (used in filename)
        output_path: Path to save the PNG (if None, returns the image array)

    Returns:
        numpy array of the combined image if output_path is None
    """
    model.eval()

    # Get chip data
    coords, rgb_gt = dataset.get_chip_data(chip_idx)
    n_pixels = len(coords)

    # Predict in batches to avoid OOM
    batch_size = 50000
    all_preds = []
    for i in range(0, n_pixels, batch_size):
        end = min(i + batch_size, n_pixels)
        preds = model(coords[i:end])
        all_preds.append(preds)

    rgb_pred = torch.cat(all_preds, dim=0)

    # Convert to numpy
    rgb_gt_np = rgb_gt.cpu().numpy()
    rgb_pred_np = rgb_pred.cpu().numpy()

    # Chips are 256x256
    size = int(np.sqrt(n_pixels))
    if size * size != n_pixels:
        print(f"Warning: Chip has {n_pixels} pixels, not a perfect square")
        size = int(np.ceil(np.sqrt(n_pixels)))

    # Reshape to images
    gt_img = (rgb_gt_np * 255).astype(np.uint8).reshape(size, size, 3)
    pred_img = (np.clip(rgb_pred_np, 0, 1) * 255).astype(np.uint8).reshape(size, size, 3)

    # Create side-by-side image (GT left, Pred right)
    combined = np.concatenate([gt_img, pred_img], axis=1)

    if output_path is not None:
        Image.fromarray(combined).save(output_path)
        return output_path
    else:
        return combined


@torch.no_grad()
def save_visualization_chips(model, dataset, output_dir, batch_idx, batch_size=50000):
    """
    Save PNG visualizations for all named visualization chips (train set memorization).

    Output: 3-column image [Ground Truth RGB | Predicted RGB | LiDAR Elevation (Viridis)]

    Args:
        model: Trained model
        dataset: NAIPDataset with visualization_chips
        output_dir: Directory to save images
        batch_idx: Current batch number (used in filename)
        batch_size: Batch size for inference

    Returns:
        List of saved file paths
    """
    import matplotlib.cm as cm

    model.eval()
    saved_files = []
    vis_dir = Path(output_dir) / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)

    for name, chip_idx in dataset.visualization_chips.items():
        # Get chip data
        coords, rgb_gt = dataset.get_chip_data(chip_idx)
        n_pixels = len(coords)

        # Extract elevation (3rd column of coords)
        elevation = coords[:, 2].cpu().numpy()

        # Compute min/max elevation rounded to nearest 5m
        elev_min_raw = elevation.min()
        elev_max_raw = elevation.max()
        elev_min = int(np.floor(elev_min_raw / 5) * 5)
        elev_max = int(np.ceil(elev_max_raw / 5) * 5)

        # Predict in batches
        all_preds = []
        for i in range(0, n_pixels, batch_size):
            end = min(i + batch_size, n_pixels)
            preds = model(coords[i:end])
            all_preds.append(preds)

        rgb_pred = torch.cat(all_preds, dim=0)

        # Convert to numpy
        rgb_gt_np = rgb_gt.cpu().numpy()
        rgb_pred_np = rgb_pred.cpu().numpy()

        # Chips are 256x256
        size = int(np.sqrt(n_pixels))
        if size * size != n_pixels:
            size = int(np.ceil(np.sqrt(n_pixels)))

        # Reshape to images
        gt_img = (rgb_gt_np * 255).astype(np.uint8).reshape(size, size, 3)
        pred_img = (np.clip(rgb_pred_np, 0, 1) * 255).astype(np.uint8).reshape(size, size, 3)

        # Create LiDAR elevation image with Viridis colormap
        # Normalize elevation to [0, 1] using rounded min/max
        if elev_max > elev_min:
            elev_norm = (elevation - elev_min) / (elev_max - elev_min)
        else:
            elev_norm = np.zeros_like(elevation)
        elev_norm = np.clip(elev_norm, 0, 1)

        # Apply Viridis colormap
        viridis = cm.get_cmap('viridis')
        elev_rgba = viridis(elev_norm.reshape(size, size))
        elev_img = (elev_rgba[:, :, :3] * 255).astype(np.uint8)

        # Create 3-column image (GT | Pred | LiDAR)
        combined = np.concatenate([gt_img, pred_img, elev_img], axis=1)

        # Save with name, batch number, and elevation range
        filename = f"{name}_{elev_min}-{elev_max}m_batch{batch_idx:05d}.png"
        filepath = vis_dir / filename
        Image.fromarray(combined).save(filepath)
        saved_files.append(str(filepath))

    return saved_files


def compute_metrics_gpu(preds, targets):
    """Compute RGB reconstruction metrics on GPU."""
    # MSE per channel and overall
    mse = ((preds - targets) ** 2).mean()
    mse_r = ((preds[:, 0] - targets[:, 0]) ** 2).mean()
    mse_g = ((preds[:, 1] - targets[:, 1]) ** 2).mean()
    mse_b = ((preds[:, 2] - targets[:, 2]) ** 2).mean()

    # RMSE
    rmse = torch.sqrt(mse)

    # MAE
    mae = (preds - targets).abs().mean()

    # PSNR (assuming max value 1.0)
    psnr = 10 * torch.log10(1.0 / mse) if mse > 0 else torch.tensor(float('inf'))

    return {
        'mse': mse.item(),
        'rmse': rmse.item(),
        'mae': mae.item(),
        'psnr': psnr.item(),
        'mse_r': mse_r.item(),
        'mse_g': mse_g.item(),
        'mse_b': mse_b.item(),
        'n_samples': len(preds)
    }


def format_metrics(metrics, prefix=""):
    """Format metrics for display."""
    parts = [
        f"MSE={metrics['mse']:.6f}",
        f"RMSE={metrics['rmse']:.4f}",
        f"MAE={metrics['mae']:.4f}",
        f"PSNR={metrics['psnr']:.2f}dB"
    ]
    return f"{prefix}{' | '.join(parts)}"


@torch.no_grad()
def evaluate(model, dataset, batch_size=50000):
    """Evaluate model on test set."""
    model.eval()

    test_coords, test_rgb = dataset.get_test_data()
    n_test = len(test_coords)

    if n_test == 0:
        return {'mse': 0, 'rmse': 0, 'mae': 0, 'psnr': 0, 'n_samples': 0}

    # Process in batches to avoid OOM
    all_preds = []
    for i in range(0, n_test, batch_size):
        end = min(i + batch_size, n_test)
        coords_batch = test_coords[i:end]
        preds_batch = model(coords_batch)
        all_preds.append(preds_batch)

    all_preds = torch.cat(all_preds, dim=0)
    metrics = compute_metrics_gpu(all_preds, test_rgb)

    return metrics


def train(args):
    """Main training function."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("=" * 80, flush=True)
    print("NAIP-3DEP RGB RECONSTRUCTION WITH EARTH4D", flush=True)
    print("=" * 80, flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Learned hash probing: {'ENABLED (state-of-the-art)' if args.enable_learned_probing else 'DISABLED (baseline)'}", flush=True)
    if args.enable_learned_probing:
        print(f"  Probing range (N_p): {args.probing_range}", flush=True)
        print(f"  Entropy weight: {args.probe_entropy_weight}", flush=True)
        print(f"  Index LR multiplier: {args.index_lr_multiplier}x", flush=True)
    print(f"Batch size: {args.batch_size:,} pixels", flush=True)
    print(f"Number of batches: {args.num_batches:,}", flush=True)
    print(f"Learning rate: {args.lr}", flush=True)
    print(f"Chip-level split: {args.chip_split}", flush=True)
    print()

    # Parse visualization locations if provided
    visualize_locations = None
    if args.visualize_locations:
        visualize_locations = parse_visualization_locations(args.visualize_locations)
        print(f"Visualization locations (forced to train): {len(visualize_locations)}", flush=True)
        for name, lat, lon in visualize_locations:
            print(f"  {name}: ({lat}, {lon})", flush=True)
        print()

    # Load dataset
    data_path = Path(args.data_dir) / 'parsed_xyztrgb.pt'
    dataset = NAIPDataset(
        data_path,
        device=device,
        train_ratio=args.train_ratio,
        chip_split=args.chip_split,
        seed=args.seed,
        visualize_locations=visualize_locations
    )

    # Create model
    print("\nInitializing model...", flush=True)
    model = RGBReconstructionModel(
        enable_learned_probing=args.enable_learned_probing,
        probing_range=args.probing_range,
        probe_entropy_weight=args.probe_entropy_weight,
        verbose=True,
        spatial_levels=args.spatial_levels,
        temporal_levels=args.temporal_levels,
        spatial_log2_hashmap_size=args.spatial_log2_hashmap_size,
        temporal_log2_hashmap_size=args.temporal_log2_hashmap_size
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    earth4d_params = sum(p.numel() for p in model.earth4d.parameters())
    mlp_params = sum(p.numel() for p in model.rgb_head.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel parameters:", flush=True)
    print(f"  Earth4D: {earth4d_params:,}", flush=True)
    print(f"  MLP head: {mlp_params:,}", flush=True)
    print(f"  Total: {total_params:,}", flush=True)
    print(f"  Trainable: {trainable_params:,}", flush=True)

    # Setup optimizer with differential learning rates for learned probing
    if args.enable_learned_probing and hasattr(model.earth4d.encoder.xyz_encoder, 'index_logits'):
        index_lr = args.lr * args.index_lr_multiplier
        optimizer_params = [
            # Earth4D embeddings (base LR)
            {'params': model.earth4d.encoder.xyz_encoder.embeddings, 'lr': args.lr},
            {'params': model.earth4d.encoder.xyt_encoder.embeddings, 'lr': args.lr},
            {'params': model.earth4d.encoder.yzt_encoder.embeddings, 'lr': args.lr},
            {'params': model.earth4d.encoder.xzt_encoder.embeddings, 'lr': args.lr},
            # MLP parameters (base LR)
            {'params': model.rgb_head.parameters(), 'lr': args.lr},
            # Index logits (higher LR - critical for learned probing)
            {'params': model.earth4d.encoder.xyz_encoder.index_logits, 'lr': index_lr},
            {'params': model.earth4d.encoder.xyt_encoder.index_logits, 'lr': index_lr},
            {'params': model.earth4d.encoder.yzt_encoder.index_logits, 'lr': index_lr},
            {'params': model.earth4d.encoder.xzt_encoder.index_logits, 'lr': index_lr},
        ]
        optimizer = optim.AdamW(optimizer_params, weight_decay=args.weight_decay)
        print(f"\nUsing {args.index_lr_multiplier}x higher LR for index_logits: {index_lr:.6f}", flush=True)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training metrics
    metrics_ema = MetricsEMA(alpha=0.1)
    history = {
        'batch': [], 'train_mse': [], 'train_rmse': [], 'train_mae': [], 'train_psnr': [],
        'test_mse': [], 'test_rmse': [], 'test_mae': [], 'test_psnr': [], 'time': []
    }

    # Training loop
    print("\n" + "=" * 80, flush=True)
    print("TRAINING", flush=True)
    print("-" * 80, flush=True)

    model.train()
    start_time = time.time()
    batch_times = []

    for batch_idx in range(1, args.num_batches + 1):
        batch_start = time.time()

        # Sample batch
        coords, rgb_gt = dataset.sample_train_batch(args.batch_size)

        # Forward pass
        rgb_pred = model(coords)

        # Compute loss with optional entropy regularization
        if hasattr(model.earth4d, 'compute_loss') and args.enable_learned_probing:
            loss_dict = model.earth4d.compute_loss(
                rgb_pred, rgb_gt,
                enable_probe_entropy_loss=True,
                probe_entropy_weight=args.probe_entropy_weight
            )
            loss = loss_dict['_total_loss_tensor']
        else:
            loss = nn.functional.mse_loss(rgb_pred, rgb_gt)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Update probe indices (for learned probing)
        if args.enable_learned_probing and hasattr(model.earth4d.encoder.xyz_encoder, 'update_probe_indices'):
            model.earth4d.encoder.xyz_encoder.update_probe_indices()
            model.earth4d.encoder.xyt_encoder.update_probe_indices()
            model.earth4d.encoder.yzt_encoder.update_probe_indices()
            model.earth4d.encoder.xzt_encoder.update_probe_indices()

        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        # Compute train metrics
        with torch.no_grad():
            train_metrics = compute_metrics_gpu(rgb_pred, rgb_gt)

        # Update EMA
        ema_metrics = metrics_ema.update(train_metrics)

        # Log every batch (verbose)
        elapsed = time.time() - start_time
        avg_batch_time = sum(batch_times[-100:]) / len(batch_times[-100:])
        samples_per_sec = args.batch_size / avg_batch_time

        print(f"[{batch_idx:5d}/{args.num_batches}] "
              f"MSE={train_metrics['mse']:.6f} (EMA={ema_metrics.get('mse_ema', 0):.6f}) | "
              f"RMSE={train_metrics['rmse']:.4f} | "
              f"PSNR={train_metrics['psnr']:.1f}dB | "
              f"{samples_per_sec:.0f} px/s | "
              f"{batch_time*1000:.1f}ms", flush=True)

        # Evaluate on test set periodically
        if batch_idx % args.test_every == 0 or batch_idx == args.num_batches:
            model.eval()
            test_metrics = evaluate(model, dataset)

            history['batch'].append(batch_idx)
            history['train_mse'].append(train_metrics['mse'])
            history['train_rmse'].append(train_metrics['rmse'])
            history['train_mae'].append(train_metrics['mae'])
            history['train_psnr'].append(train_metrics['psnr'])
            history['test_mse'].append(test_metrics['mse'])
            history['test_rmse'].append(test_metrics['rmse'])
            history['test_mae'].append(test_metrics['mae'])
            history['test_psnr'].append(test_metrics['psnr'])
            history['time'].append(elapsed)

            print(f"  >> TEST ({test_metrics['n_samples']:,} px): "
                  f"MSE={test_metrics['mse']:.6f} | RMSE={test_metrics['rmse']:.4f} | "
                  f"MAE={test_metrics['mae']:.4f} | PSNR={test_metrics['psnr']:.2f}dB", flush=True)

            # Save visualizations for named visualization chips (train set memorization)
            if dataset.visualization_chips:
                saved_files = save_visualization_chips(model, dataset, output_dir, batch_idx)
                if saved_files:
                    print(f"  >> Saved {len(saved_files)} visualization(s)", flush=True)

            model.train()

    # Final summary
    total_time = time.time() - start_time
    print("\n" + "=" * 80, flush=True)
    print("TRAINING COMPLETE", flush=True)
    print("=" * 80, flush=True)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.2f} min)", flush=True)
    print(f"Total batches: {args.num_batches:,}", flush=True)
    print(f"Total pixels trained: {args.num_batches * args.batch_size:,}", flush=True)

    # Final test evaluation
    model.eval()
    final_test = evaluate(model, dataset)
    print(f"\nFinal Test Results:", flush=True)
    print(f"  MSE:  {final_test['mse']:.6f}", flush=True)
    print(f"  RMSE: {final_test['rmse']:.4f}", flush=True)
    print(f"  MAE:  {final_test['mae']:.4f}", flush=True)
    print(f"  PSNR: {final_test['psnr']:.2f} dB", flush=True)

    # Save checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': vars(args),
        'history': history,
        'final_test_metrics': final_test,
        'total_time': total_time
    }
    checkpoint_path = output_dir / 'checkpoint_final.pt'
    torch.save(checkpoint, checkpoint_path)
    print(f"\nCheckpoint saved to {checkpoint_path}", flush=True)

    # Save history
    history_path = output_dir / 'history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {history_path}", flush=True)

    return final_test


def main():
    parser = argparse.ArgumentParser(
        description='NAIP-3DEP RGB Reconstruction with Earth4D',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data
    parser.add_argument('--data-dir', '-d', type=str, default='stanford',
                       help='Path to dataset directory containing parsed_xyztrgb.pt')
    parser.add_argument('--output-dir', '-o', type=str, default='runs/naip_earth4d',
                       help='Output directory for checkpoints')

    # Training
    parser.add_argument('--num-batches', '-n', type=int, default=1000,
                       help='Number of training batches')
    parser.add_argument('--batch-size', '-b', type=int, default=10000,
                       help='Batch size (pixels per batch)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.001,
                       help='Weight decay for AdamW')
    parser.add_argument('--test-every', type=int, default=10,
                       help='Evaluate test set every N batches')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Data split
    parser.add_argument('--train-ratio', type=float, default=0.95,
                       help='Training set ratio (default: 95%% train, 5%% test)')
    parser.add_argument('--chip-split', action='store_true', default=True,
                       help='Split train/test by chips (recommended)')
    parser.add_argument('--no-chip-split', dest='chip_split', action='store_false',
                       help='Use random pixel split instead of chip split')

    # Visualization chip selection (forced to train set for memorization tracking)
    parser.add_argument('--visualize-locations', nargs='+', type=str, default=None,
                       help='Visualize memorization of specific locations (forced to train set). '
                            'Format: "Name:lat,lon" e.g. "Eucalyptus_Grove:37.436,-122.165"')

    # Earth4D model size configuration
    parser.add_argument('--spatial-levels', type=int, default=24,
                       help='Number of spatial resolution levels (default: 24)')
    parser.add_argument('--temporal-levels', type=int, default=24,
                       help='Number of temporal resolution levels (default: 24)')
    parser.add_argument('--spatial-log2-hashmap-size', type=int, default=22,
                       help='Log2 of spatial hash table size (default: 22 = 4M entries). '
                            'Use 15 for ~5M params, 19 for ~50M, 22 for ~800M')
    parser.add_argument('--temporal-log2-hashmap-size', type=int, default=22,
                       help='Log2 of temporal hash table size (default: 22)')

    # Learned hash probing (state-of-the-art, ON by default)
    parser.add_argument('--enable-learned-probing', action='store_true', default=True,
                       help='Enable learned hash probing (state-of-the-art)')
    parser.add_argument('--no-learned-probing', dest='enable_learned_probing', action='store_false',
                       help='Disable learned hash probing (baseline)')
    parser.add_argument('--probing-range', type=int, default=32,
                       help='Probing range N_p (optimal: 32, must be power-of-2)')
    parser.add_argument('--probe-entropy-weight', type=float, default=0.5,
                       help='Entropy regularization weight (optimal: 0.5)')
    parser.add_argument('--index-lr-multiplier', type=float, default=10.0,
                       help='Learning rate multiplier for index_logits (optimal: 10x)')

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
