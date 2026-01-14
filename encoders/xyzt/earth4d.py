"""
Earth4D: Planetary (X,Y,Z,T) Positional Encoder
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from typing import Optional, Literal

from hashencoder.hashgrid import HashEncoder
from coordinates import to_ecef, ECEF_NORM_FACTOR, AdaptiveRange, GeoAdaptiveRange
from ops import compute_loss, print_resolution_info
import tracking


class Earth4D(nn.Module):

    def __init__(self,
                 spatial_levels: int = 24,
                 temporal_levels: int = 24,
                 features_per_level: int = 2,
                 spatial_log2_hashmap_size: int = 22,
                 temporal_log2_hashmap_size: int = 22,
                 base_spatial_resolution: float = 32.0,
                 base_temporal_resolution: float = 32.0,
                 growth_factor: float = 2.0,
                 temporal_growth_factor: float = None,  # If None, uses growth_factor
                 latlon_growth_factor: float = None,    # If None, uses encoder-specific default
                 elev_growth_factor: float = None,      # If None, uses encoder-specific default
                 verbose: bool = True,
                 enable_collision_tracking: bool = False,
                 max_tracked_examples: int = 1000000,
                 enable_learned_probing: bool = True,
                 probing_range: int = 32,
                 index_codebook_size: int = 512,
                 probe_entropy_weight: float = 0.0,
                 use_adaptive_range: bool = False,
                 adaptive_range: Optional[AdaptiveRange] = None,
                 max_precision_level: Optional[int] = None,
                 coordinate_system: Literal['geographic', 'ecef'] = 'ecef',
                 geo_range: Optional[GeoAdaptiveRange] = None,
                 resolution_mode: str = 'balanced'):
        """
        Args:
            spatial_levels: Number of spatial resolution levels
            temporal_levels: Number of temporal resolution levels
            features_per_level: Features per level (default: 2)
            spatial_log2_hashmap_size: Log2 of spatial hash table size
            temporal_log2_hashmap_size: Log2 of temporal hash table size
            base_spatial_resolution: Base resolution for spatial encoder
            base_temporal_resolution: Base resolution for temporal encoder
            growth_factor: Growth factor between levels (default: 2.0)
            latlon_growth_factor: Growth factor for lat/lon dimensions (default: growth_factor for xyz, temporal_growth_factor for others)
            elev_growth_factor: Growth factor for elevation dimension (default: growth_factor for xyz, temporal_growth_factor for others)
            verbose: Print resolution table on initialization
            enable_collision_tracking: Track hash indices for collision analysis
            max_tracked_examples: Maximum number of examples to track
            enable_learned_probing: Enable learned hash probing (default: True)
            probing_range: Number of probe candidates (default: 32, must be power-of-2)
            index_codebook_size: Size of learned probe table (default: 512)
            probe_entropy_weight: Entropy regularization weight (default: 0.5)
            use_adaptive_range: Fit normalization to data extent (default: False)
            adaptive_range: Pre-computed AdaptiveRange object (optional)
            max_precision_level: Cap finest level (optional, for compute savings)
        """
        super().__init__()

        self.verbose = verbose

        # Collision tracking configuration
        self.enable_collision_tracking = enable_collision_tracking
        self.max_tracked_examples = max_tracked_examples
        self.collision_tracking_data = None

        # Learned probing configuration
        self.enable_learned_probing = enable_learned_probing
        self.probing_range = probing_range
        self.index_codebook_size = index_codebook_size
        self.probe_entropy_weight = probe_entropy_weight

        # Adaptive range configuration
        self.use_adaptive_range = use_adaptive_range
        self.adaptive_range = adaptive_range
        self.max_precision_level = max_precision_level

        # Store base parameters for level analysis
        self.base_spatial_resolution = base_spatial_resolution
        self.base_temporal_resolution = base_temporal_resolution
        self.spatial_levels = spatial_levels
        self.temporal_levels = temporal_levels
        self.growth_factor = growth_factor
        self.temporal_growth_factor = temporal_growth_factor if temporal_growth_factor is not None else growth_factor
        self.features_per_level = features_per_level

        # Decoupled growth factors for lat/lon vs elevation
        # For xyz (spatial-only): default to growth_factor (2.0)
        # For xyt/yzt/xzt (spatiotemporal): default to temporal_growth_factor
        self.latlon_growth_factor = latlon_growth_factor
        self.elev_growth_factor = elev_growth_factor

        # Coordinate system configuration
        self.coordinate_system = coordinate_system
        self.resolution_mode = resolution_mode
        self.geo_range = geo_range

        # Initialize geo_range for geographic mode if not provided
        if coordinate_system == 'geographic' and geo_range is None:
            self.geo_range = GeoAdaptiveRange.global_range()
            self._geo_range_is_default = True
        else:
            self._geo_range_is_default = False

        # Check HashEncoder is available
        if HashEncoder is None:
            raise ImportError(
                "HashEncoder is required for Earth4D functionality. "
                "Please install the hash encoding library."
            )

        # Store hashmap sizes for reporting
        self.spatial_log2_hashmap_size = spatial_log2_hashmap_size
        self.temporal_log2_hashmap_size = temporal_log2_hashmap_size

        # Store dimensions for output
        self.spatial_dim = spatial_levels * features_per_level
        self.spatiotemporal_dim = temporal_levels * features_per_level * 3  # 3 projections
        self.output_dim = self.spatial_dim + self.spatiotemporal_dim

        # Calculate max resolutions
        spatial_max_res = int(base_spatial_resolution * (growth_factor ** (spatial_levels - 1)))
        temporal_base_res = [int(base_temporal_resolution)] * 3
        tgf = self.temporal_growth_factor

        # Determine per-axis growth factors for each encoder
        # xyz (lat, lon, elev): uses growth_factor by default, or explicit overrides
        # xyt (lat, lon, time): uses temporal_growth_factor by default
        # yzt (lon, elev, time): uses temporal_growth_factor by default
        # xzt (lat, elev, time): uses temporal_growth_factor by default
        llgf = latlon_growth_factor  # May be None
        egf = elev_growth_factor     # May be None

        # xyz encoder: [lat, lon, elev]
        if llgf is not None or egf is not None:
            # Per-axis growth specified - use list
            xyz_llgf = llgf if llgf is not None else growth_factor
            xyz_egf = egf if egf is not None else growth_factor
            xyz_scale = [xyz_llgf, xyz_llgf, xyz_egf]
            xyz_max_res = [
                int(base_spatial_resolution * (xyz_llgf ** (spatial_levels - 1))),
                int(base_spatial_resolution * (xyz_llgf ** (spatial_levels - 1))),
                int(base_spatial_resolution * (xyz_egf ** (spatial_levels - 1))),
            ]
        else:
            # Default: uniform growth_factor=2.0
            xyz_scale = growth_factor
            xyz_max_res = spatial_max_res

        # Spatiotemporal encoders: use temporal_growth_factor as baseline
        if llgf is not None or egf is not None:
            # Per-axis growth specified
            st_llgf = llgf if llgf is not None else tgf  # Default to tgf for spatiotemporal
            st_egf = egf if egf is not None else tgf

            # xyt: [lat, lon, time]
            xyt_scale = [st_llgf, st_llgf, tgf]
            xyt_max_res = [
                int(base_temporal_resolution * (st_llgf ** (temporal_levels - 1))),
                int(base_temporal_resolution * (st_llgf ** (temporal_levels - 1))),
                int(base_temporal_resolution * (tgf ** (temporal_levels - 1))),
            ]

            # yzt: [lon, elev, time]
            yzt_scale = [st_llgf, st_egf, tgf]
            yzt_max_res = [
                int(base_temporal_resolution * (st_llgf ** (temporal_levels - 1))),
                int(base_temporal_resolution * (st_egf ** (temporal_levels - 1))),
                int(base_temporal_resolution * (tgf ** (temporal_levels - 1))),
            ]

            # xzt: [lat, elev, time]
            xzt_scale = [st_llgf, st_egf, tgf]
            xzt_max_res = [
                int(base_temporal_resolution * (st_llgf ** (temporal_levels - 1))),
                int(base_temporal_resolution * (st_egf ** (temporal_levels - 1))),
                int(base_temporal_resolution * (tgf ** (temporal_levels - 1))),
            ]
        else:
            # Default: uniform tgf for all dimensions
            xyt_scale = yzt_scale = xzt_scale = tgf
            temporal_max_res = [int(base_temporal_resolution * (tgf ** (temporal_levels - 1)))] * 3
            xyt_max_res = yzt_max_res = xzt_max_res = temporal_max_res

        # Spatial encoder (xyz) - [lat, lon, elev]
        self.xyz_encoder = HashEncoder(
            input_dim=3,
            num_levels=spatial_levels,
            level_dim=features_per_level,
            per_level_scale=xyz_scale,
            base_resolution=int(base_spatial_resolution),
            log2_hashmap_size=spatial_log2_hashmap_size,
            desired_resolution=xyz_max_res,
            enable_learned_probing=enable_learned_probing,
            probing_range=probing_range,
            index_codebook_size=index_codebook_size
        )

        # Spatiotemporal encoders with per-axis growth factors
        # xyt: [lat, lon, time]
        self.xyt_encoder = HashEncoder(
            input_dim=3,
            num_levels=temporal_levels,
            level_dim=features_per_level,
            per_level_scale=xyt_scale,
            base_resolution=temporal_base_res,
            log2_hashmap_size=temporal_log2_hashmap_size,
            desired_resolution=xyt_max_res,
            enable_learned_probing=enable_learned_probing,
            probing_range=probing_range,
            index_codebook_size=index_codebook_size
        )

        # yzt: [lon, elev, time]
        self.yzt_encoder = HashEncoder(
            input_dim=3,
            num_levels=temporal_levels,
            level_dim=features_per_level,
            per_level_scale=yzt_scale,
            base_resolution=temporal_base_res,
            log2_hashmap_size=temporal_log2_hashmap_size,
            desired_resolution=yzt_max_res,
            enable_learned_probing=enable_learned_probing,
            probing_range=probing_range,
            index_codebook_size=index_codebook_size
        )

        # xzt: [lat, elev, time]
        self.xzt_encoder = HashEncoder(
            input_dim=3,
            num_levels=temporal_levels,
            level_dim=features_per_level,
            per_level_scale=xzt_scale,
            base_resolution=temporal_base_res,
            log2_hashmap_size=temporal_log2_hashmap_size,
            desired_resolution=xzt_max_res,
            enable_learned_probing=enable_learned_probing,
            probing_range=probing_range,
            index_codebook_size=index_codebook_size
        )

        # Initialize collision tracking if enabled
        if self.enable_collision_tracking:
            self._init_collision_tracking()

        if self.verbose and not self.use_adaptive_range:
            self._print_resolution_info()
    
    def _init_collision_tracking(self):
        """Initialize collision tracking tensors."""
        self.collision_tracking_data = tracking.init(self, self.max_tracked_examples)

    def to(self, *args, **kwargs):
        """Override to() to also move collision tracking tensors."""
        super().to(*args, **kwargs)
        if self.collision_tracking_data is not None:
            self.collision_tracking_data = tracking.move(self.collision_tracking_data, *args, **kwargs)
        return self

    def cuda(self, device=None):
        """Override cuda() to also move collision tracking tensors to GPU."""
        super().cuda(device)
        if self.collision_tracking_data is not None:
            device_str = 'cuda' if device is None else f'cuda:{device}'
            self.collision_tracking_data = tracking.move(self.collision_tracking_data, device=device_str)
        return self

    def _print_resolution_info(self):
        """Print detailed resolution information."""
        config = {
            'use_adaptive_range': self.use_adaptive_range,
            'enable_learned_probing': self.enable_learned_probing,
            'probing_range': self.probing_range,
            'max_precision_level': self.max_precision_level,
            'spatial_levels': self.spatial_levels,
            'coordinate_system': self.coordinate_system,
        }
        adaptive_range = self.geo_range if self.coordinate_system == 'geographic' else self.adaptive_range
        print_resolution_info(self, config, adaptive_range)

    def _encode_spatial(self, xyz: torch.Tensor, ct_data: dict = None) -> torch.Tensor:
        """Encode spatial xyz coordinates."""
        xyz_tracking = ct_data.get('xyz') if ct_data else None
        return self.xyz_encoder(xyz, size=1.0, collision_tracking=xyz_tracking)

    def _encode_spatiotemporal(self, xyzt: torch.Tensor, ct_data: dict = None) -> torch.Tensor:
        """Encode spatiotemporal projections (xyt, yzt, xzt)."""
        # Scale time dimension
        t_scaled = (xyzt[..., 3:] * 2 - 1) * 0.9
        xyzt_scaled = torch.cat([xyzt[..., :3], t_scaled], dim=-1)

        # Create 3D projections
        xyt = torch.cat([xyzt_scaled[..., :2], xyzt_scaled[..., 3:]], dim=-1)
        yzt = xyzt_scaled[..., 1:]
        xzt = torch.cat([xyzt_scaled[..., :1], xyzt_scaled[..., 2:]], dim=-1)

        xyt_tracking = ct_data.get('xyt') if ct_data else None
        yzt_tracking = ct_data.get('yzt') if ct_data else None
        xzt_tracking = ct_data.get('xzt') if ct_data else None
        xyt_features = self.xyt_encoder(xyt, size=1.0, collision_tracking=xyt_tracking)
        yzt_features = self.yzt_encoder(yzt, size=1.0, collision_tracking=yzt_tracking)
        xzt_features = self.xzt_encoder(xzt, size=1.0, collision_tracking=xzt_tracking)

        return torch.cat([xyt_features, yzt_features, xzt_features], dim=-1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Earth4D encoder.

        Args:
            coords: Input coordinates tensor (..., 4)
                    Format: [latitude, longitude, elevation_m, time_normalized]

        Returns:
            Concatenated features tensor
        """
        if self.coordinate_system == 'geographic':
            # Geographic mode: normalize lat/lon/elev directly
            lat, lon, elev, time = coords[..., 0], coords[..., 1], coords[..., 2], coords[..., 3]
            x_norm, y_norm, z_norm, time_norm = self.geo_range.normalize(lat, lon, elev, time)
            norm_coords = torch.stack([x_norm, y_norm, z_norm, time_norm], dim=-1)
        else:
            # ECEF mode: convert to Earth-centered coordinates
            x, y, z = to_ecef(coords[..., 0], coords[..., 1], coords[..., 2])
            time = coords[..., 3]

            # Normalize ECEF coordinates
            if self.use_adaptive_range and self.adaptive_range is not None:
                x_norm, y_norm, z_norm, time_norm = self.adaptive_range.normalize_ecef(x, y, z, time)
            else:
                x_norm = x / ECEF_NORM_FACTOR
                y_norm = y / ECEF_NORM_FACTOR
                z_norm = z / ECEF_NORM_FACTOR
                time_norm = time

            norm_coords = torch.stack([x_norm, y_norm, z_norm, time_norm], dim=-1)

        ct_data = None
        if self.enable_collision_tracking and self.collision_tracking_data is not None:
            offset = tracking.save_coords(self.collision_tracking_data, coords, norm_coords, self.max_tracked_examples)
            tracking.set_offsets(self.collision_tracking_data, offset)
            ct_data = self.collision_tracking_data

        spatial_features = self._encode_spatial(norm_coords[..., :3], ct_data)
        spatiotemporal_features = self._encode_spatiotemporal(norm_coords, ct_data)

        return torch.cat([spatial_features, spatiotemporal_features], dim=-1)

    def get_output_dim(self) -> int:
        """Return total output dimension."""
        return self.output_dim

    def compute_loss(self, predictions, targets, criterion=None,
                    enable_probe_entropy_loss=None, probe_entropy_weight=None,
                    enable_gradient_validation=False):
        """Compute loss with optional regularization for learned hash probing."""
        return compute_loss(
            predictions, targets, self, criterion,
            self.enable_learned_probing,
            probe_entropy_weight or self.probe_entropy_weight,
            enable_probe_entropy_loss, enable_gradient_validation
        )
    
    def export_collision_data(self, output_dir: str = "collision_analysis", fmt: str = 'csv'):
        """Export collision tracking data for scientific analysis."""
        if not self.enable_collision_tracking:
            raise RuntimeError("Collision tracking not enabled")
        return tracking.export(self.collision_tracking_data, self, self.max_tracked_examples, output_dir, fmt)

    def fit_range(self, coords: torch.Tensor, buffer_fraction: float = 0.25) -> 'Earth4D':
        """Fit adaptive range from training coordinates for improved hash utilization."""
        x, y, z = to_ecef(coords[..., 0], coords[..., 1], coords[..., 2])
        ecef_coords = torch.stack([x, y, z], dim=-1)

        self.adaptive_range = AdaptiveRange.from_ecef_coordinates(
            ecef_coords, coords[..., 3], orig_coords=coords, buffer_fraction=buffer_fraction
        )
        self.use_adaptive_range = True

        if self.verbose:
            self._print_resolution_info()

        return self

    def fit_geo_range(self, coords: torch.Tensor,
                      lat_coverage: float = 0.25,
                      lon_coverage: float = 0.25,
                      elev_coverage: float = 0.15,
                      time_coverage: float = 1.0) -> 'Earth4D':
        """Fit geographic range from training coordinates.

        Args:
            coords: Training coordinates (N, 4) - [lat, lon, elev, time]
            lat_coverage: Target coverage fraction for latitude (smaller = tighter fit)
            lon_coverage: Target coverage fraction for longitude
            elev_coverage: Target coverage fraction for elevation
            time_coverage: Target coverage fraction for time

        Returns:
            self for chaining
        """
        self.geo_range = GeoAdaptiveRange.balanced_regional(
            coords, lat_coverage, lon_coverage, elev_coverage, time_coverage
        )
        self._geo_range_is_default = False

        if self.verbose:
            self._print_resolution_info()

        return self

    def precompute(self, coords: torch.Tensor) -> dict:
        """
        Precompute hash indices and weights for a fixed set of coordinates.

        Call once before training. Enables forward_precomputed() for faster training.

        Args:
            coords: Input coordinates tensor (N, 4) - [lat, lon, elev, time]

        Returns:
            dict with memory usage statistics for all 4 encoders
        """
        # Convert coordinates to normalized form
        if self.coordinate_system == 'geographic':
            lat, lon, elev, time = coords[..., 0], coords[..., 1], coords[..., 2], coords[..., 3]
            x_norm, y_norm, z_norm, time_norm = self.geo_range.normalize(lat, lon, elev, time)
        else:
            x, y, z = to_ecef(coords[..., 0], coords[..., 1], coords[..., 2])
            time = coords[..., 3]

            if self.use_adaptive_range and self.adaptive_range is not None:
                x_norm, y_norm, z_norm, time_norm = self.adaptive_range.normalize_ecef(x, y, z, time)
            else:
                x_norm = x / ECEF_NORM_FACTOR
                y_norm = y / ECEF_NORM_FACTOR
                z_norm = z / ECEF_NORM_FACTOR
                time_norm = time

        norm_coords = torch.stack([x_norm, y_norm, z_norm, time_norm], dim=-1)

        # Scale time dimension
        t_scaled = (norm_coords[..., 3:] * 2 - 1) * 0.9
        xyzt_scaled = torch.cat([norm_coords[..., :3], t_scaled], dim=-1)

        # Create 3D projections
        xyz = norm_coords[..., :3]
        xyt = torch.cat([xyzt_scaled[..., :2], xyzt_scaled[..., 3:]], dim=-1)
        yzt = xyzt_scaled[..., 1:]
        xzt = torch.cat([xyzt_scaled[..., :1], xyzt_scaled[..., 2:]], dim=-1)

        # Precompute for each encoder
        stats = {}
        stats['xyz'] = self.xyz_encoder.precompute(xyz, size=1.0)
        stats['xyt'] = self.xyt_encoder.precompute(xyt, size=1.0)
        stats['yzt'] = self.yzt_encoder.precompute(yzt, size=1.0)
        stats['xzt'] = self.xzt_encoder.precompute(xzt, size=1.0)

        # Total memory
        total_bytes = sum(s['total_bytes'] for s in stats.values())
        total_mb = total_bytes / (1024 * 1024)

        self._precomputed = True
        self._precomp_coords_count = coords.shape[0]

        if self.verbose:
            print(f"\nPrecomputation complete:")
            print(f"  Coordinates: {coords.shape[0]:,}")
            print(f"  Total memory: {total_mb:.1f} MB ({total_bytes / coords.shape[0]:.1f} bytes/coord)")
            for name, s in stats.items():
                print(f"    {name}: {s['total_mb']:.1f} MB")

        return {
            'total_bytes': total_bytes,
            'total_mb': total_mb,
            'bytes_per_coord': total_bytes / coords.shape[0],
            'num_coords': coords.shape[0],
            'encoders': stats
        }

    def forward_precomputed(self, batch_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using precomputed indices and weights.

        Must call precompute() first with all training coordinates.

        Args:
            batch_indices: Tensor of indices into precomputed coordinates

        Returns:
            Concatenated features tensor (B, output_dim)
        """
        if not hasattr(self, '_precomputed') or not self._precomputed:
            raise RuntimeError("Must call precompute() before forward_precomputed()")

        # Get features from each encoder
        spatial_features = self.xyz_encoder.forward_precomputed(batch_indices)
        xyt_features = self.xyt_encoder.forward_precomputed(batch_indices)
        yzt_features = self.yzt_encoder.forward_precomputed(batch_indices)
        xzt_features = self.xzt_encoder.forward_precomputed(batch_indices)

        spatiotemporal_features = torch.cat([xyt_features, yzt_features, xzt_features], dim=-1)
        return torch.cat([spatial_features, spatiotemporal_features], dim=-1)

    def clear_precomputed(self):
        """Clear precomputed buffers to free memory."""
        self._precomputed = False
        self._precomp_coords_count = 0
        self.xyz_encoder.clear_precomputed()
        self.xyt_encoder.clear_precomputed()
        self.yzt_encoder.clear_precomputed()
        self.xzt_encoder.clear_precomputed()


# Example usage and testing
if __name__ == "__main__":
    print("Earth4D: Planetary Spatiotemporal Positional Encoder")
    print("=" * 60)
    
    # Check device availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Using device: {device.upper()}")

    encoder = Earth4D(
        spatial_levels=24,
        temporal_levels=24,
        spatial_log2_hashmap_size=22,
        temporal_log2_hashmap_size=22,
        verbose=True
    ).to(device)

    # Example coordinates: [lat, lon, elev_m, time_norm]
    coords = torch.tensor([
        [37.7749, -122.4194, 50.0, 0.5],   # San Francisco
        [40.7128, -74.0060, 100.0, 0.7],   # New York
        [-33.8688, 151.2093, 20.0, 0.3],   # Sydney
    ], device=device)

    features = encoder(coords)
    print(f"\nInput shape: {coords.shape}")
    print(f"Output shape: {features.shape}")
