"""
Energy4D: Relative Spatiotemporal Positional Encoder

Energy4D encodes RELATIVE spatiotemporal offsets instead of absolute coordinates.
This enables better generalization across different locations and times by learning
spatiotemporal dynamics rather than memorizing specific coordinates.

Key Difference from Earth4D:
- Earth4D: Encodes absolute (lat, lon, elev, time) → learns "what happens at 56.7°N"
- Energy4D: Encodes relative (Δx, Δy, Δz, Δt) → learns "what happens 10km north"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "xyzt"))

import torch
import torch.nn as nn
from typing import Optional, Literal, Tuple

from hashencoder.hashgrid import HashEncoder
from coordinates import to_ecef, ECEF_NORM_FACTOR, AdaptiveRange, GeoAdaptiveRange
from ops import compute_loss, print_resolution_info


class Energy4D(nn.Module):
    """
    Energy4D: Relative spatiotemporal encoder using multi-scale hash encoding.

    Instead of encoding absolute coordinates, Energy4D encodes RELATIVE offsets
    from reference points, enabling translational invariance and better generalization.

    Architecture identical to Earth4D but operates on (Δx, Δy, Δz, Δt).
    """

    def __init__(self,
                 spatial_levels: int = 24,
                 temporal_levels: int = 24,
                 features_per_level: int = 2,
                 spatial_log2_hashmap_size: int = 22,
                 temporal_log2_hashmap_size: int = 22,
                 base_spatial_resolution: float = 32.0,
                 base_temporal_resolution: float = 32.0,
                 growth_factor: float = 2.0,
                 temporal_growth_factor: float = None,
                 latlon_growth_factor: float = None,
                 elev_growth_factor: float = None,
                 verbose: bool = True,
                 enable_learned_probing: bool = False,  # Disabled by default (stability)
                 probing_range: int = 32,
                 index_codebook_size: int = 512,
                 use_adaptive_range: bool = False,
                 coordinate_system: Literal['geographic', 'ecef'] = 'geographic',
                 resolution_mode: str = 'balanced'):
        """
        Initialize Energy4D relative spatiotemporal encoder.

        Args:
            spatial_levels: Number of spatial resolution levels (default: 24)
            temporal_levels: Number of temporal resolution levels (default: 24)
            features_per_level: Features per level (default: 2)
            ... (same parameters as Earth4D)

        Note:
            enable_learned_probing is disabled by default due to gradient computation
            issues with PyTorch autograd in-place operations.
        """
        super().__init__()

        self.verbose = verbose
        self.enable_learned_probing = enable_learned_probing
        self.probing_range = probing_range
        self.index_codebook_size = index_codebook_size
        self.use_adaptive_range = use_adaptive_range
        self.coordinate_system = coordinate_system
        self.resolution_mode = resolution_mode

        # Store parameters
        self.base_spatial_resolution = base_spatial_resolution
        self.base_temporal_resolution = base_temporal_resolution
        self.spatial_levels = spatial_levels
        self.temporal_levels = temporal_levels
        self.growth_factor = growth_factor
        self.temporal_growth_factor = temporal_growth_factor if temporal_growth_factor is not None else growth_factor
        self.features_per_level = features_per_level
        self.latlon_growth_factor = latlon_growth_factor
        self.elev_growth_factor = elev_growth_factor

        # Initialize geo_range for geographic mode
        if coordinate_system == 'geographic':
            self.geo_range = GeoAdaptiveRange.global_range()
        else:
            self.geo_range = None

        # Output dimensions
        self.spatial_dim = spatial_levels * features_per_level
        self.spatiotemporal_dim = temporal_levels * features_per_level * 3
        self.output_dim = self.spatial_dim + self.spatiotemporal_dim

        # Calculate max resolutions
        spatial_max_res = int(base_spatial_resolution * (growth_factor ** (spatial_levels - 1)))
        temporal_base_res = [int(base_temporal_resolution)] * 3
        tgf = self.temporal_growth_factor

        llgf = latlon_growth_factor
        egf = elev_growth_factor

        # Configure encoders (same as Earth4D)
        if llgf is not None or egf is not None:
            xyz_llgf = llgf if llgf is not None else growth_factor
            xyz_egf = egf if egf is not None else growth_factor
            xyz_scale = [xyz_llgf, xyz_llgf, xyz_egf]
            xyz_max_res = [
                int(base_spatial_resolution * (xyz_llgf ** (spatial_levels - 1))),
                int(base_spatial_resolution * (xyz_llgf ** (spatial_levels - 1))),
                int(base_spatial_resolution * (xyz_egf ** (spatial_levels - 1))),
            ]
        else:
            xyz_scale = growth_factor
            xyz_max_res = spatial_max_res

        if llgf is not None or egf is not None:
            st_llgf = llgf if llgf is not None else tgf
            st_egf = egf if egf is not None else tgf
            xyt_scale = [st_llgf, st_llgf, tgf]
            yzt_scale = [st_llgf, st_egf, tgf]
            xzt_scale = [st_llgf, st_egf, tgf]
            xyt_max_res = [
                int(base_temporal_resolution * (st_llgf ** (temporal_levels - 1))),
                int(base_temporal_resolution * (st_llgf ** (temporal_levels - 1))),
                int(base_temporal_resolution * (tgf ** (temporal_levels - 1))),
            ]
            yzt_max_res = xzt_max_res = xyt_max_res
        else:
            xyt_scale = yzt_scale = xzt_scale = tgf
            temporal_max_res = [int(base_temporal_resolution * (tgf ** (temporal_levels - 1)))] * 3
            xyt_max_res = yzt_max_res = xzt_max_res = temporal_max_res

        # Initialize hash encoders (same architecture as Earth4D)
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

        if self.verbose:
            print(f"\n{'='*60}")
            print("Energy4D: Relative Spatiotemporal Encoder")
            print(f"{'='*60}")
            print(f"  Mode: RELATIVE offsets (Δx, Δy, Δz, Δt)")
            print(f"  Spatial levels: {spatial_levels}")
            print(f"  Temporal levels: {temporal_levels}")
            print(f"  Features per level: {features_per_level}")
            print(f"  Output dimension: {self.output_dim}")
            print(f"  Learned probing: {enable_learned_probing}")
            print(f"  Coordinate system: {coordinate_system}")
            print(f"{'='*60}\n")

    def forward(
        self,
        coords: torch.Tensor,
        reference_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode relative spatiotemporal coordinates.

        Parameters
        ----------
        coords : torch.Tensor
            Coordinates to encode, shape (N, 4) - [lat, lon, elev, time]
        reference_coords : torch.Tensor, optional
            Reference coordinates for relative encoding, shape (N, 4)
            If None, treats all coords as centered at origin (relative to global mean)

        Returns
        -------
        features : torch.Tensor
            Encoded features, shape (N, output_dim=192)

        Key Difference from Earth4D:
        ----------------------------
        Earth4D: forward(coords) → encodes coords directly
        Energy4D: forward(coords, reference_coords) → encodes (coords - reference_coords)
        """
        # Compute relative offsets
        if reference_coords is not None:
            # Relative mode: encode offsets from reference points
            relative_coords = coords - reference_coords
        else:
            # Fallback: encode absolute (same as Earth4D)
            relative_coords = coords

        # Normalize based on coordinate system
        if self.coordinate_system == 'geographic':
            lat, lon, elev, time = relative_coords[..., 0], relative_coords[..., 1], relative_coords[..., 2], relative_coords[..., 3]
            x_norm, y_norm, z_norm, time_norm = self.geo_range.normalize(lat, lon, elev, time)
            norm_coords = torch.stack([x_norm, y_norm, z_norm, time_norm], dim=-1)
        else:
            # ECEF mode
            x, y, z = to_ecef(relative_coords[..., 0], relative_coords[..., 1], relative_coords[..., 2])
            time = relative_coords[..., 3]
            x_norm = x / ECEF_NORM_FACTOR
            y_norm = y / ECEF_NORM_FACTOR
            z_norm = z / ECEF_NORM_FACTOR
            time_norm = time
            norm_coords = torch.stack([x_norm, y_norm, z_norm, time_norm], dim=-1)

        # Encode spatial (same as Earth4D)
        spatial_features = self.xyz_encoder(norm_coords[..., :3], size=1.0)

        # Encode spatiotemporal (same as Earth4D)
        t_scaled = (norm_coords[..., 3:] * 2 - 1) * 0.9
        xyzt_scaled = torch.cat([norm_coords[..., :3], t_scaled], dim=-1)

        xyt = torch.cat([xyzt_scaled[..., :2], xyzt_scaled[..., 3:]], dim=-1)
        yzt = xyzt_scaled[..., 1:]
        xzt = torch.cat([xyzt_scaled[..., :1], xyzt_scaled[..., 2:]], dim=-1)

        xyt_features = self.xyt_encoder(xyt, size=1.0)
        yzt_features = self.yzt_encoder(yzt, size=1.0)
        xzt_features = self.xzt_encoder(xzt, size=1.0)

        spatiotemporal_features = torch.cat([xyt_features, yzt_features, xzt_features], dim=-1)

        return torch.cat([spatial_features, spatiotemporal_features], dim=-1)

    def get_output_dim(self) -> int:
        """Return total output dimension."""
        return self.output_dim

    def fit_geo_range(self, coords: torch.Tensor,
                      lat_coverage: float = 0.25,
                      lon_coverage: float = 0.25,
                      elev_coverage: float = 0.15,
                      time_coverage: float = 1.0) -> 'Energy4D':
        """Fit geographic range from training coordinates."""
        self.geo_range = GeoAdaptiveRange.balanced_regional(
            coords, lat_coverage, lon_coverage, elev_coverage, time_coverage
        )
        return self


def check_availability() -> Tuple[bool, Optional[str]]:
    """Check if Energy4D is available."""
    try:
        from encoders.xyzt.hashencoder.hashgrid import HashEncoder
        return True, None
    except ImportError as e:
        return False, f"HashEncoder not available: {e}"
