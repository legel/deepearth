"""
Geographic coordinate utilities for Earth4D.

Supports two coordinate systems:
- ECEF (Earth-Centered, Earth-Fixed): Traditional WGS84 ellipsoid conversion
- Geographic: Direct lat/lon/elev mapping that preserves latitude relationships
  - x = latitude (-90 to +90 degrees)
  - y = longitude (-180 to +180 degrees)
  - z = elevation (meters above MSL)
"""

import warnings
import torch
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, asdict, field

# WGS84 ellipsoid parameters
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2 * WGS84_F - WGS84_F**2

ECEF_NORM_FACTOR = 6400000.0


def to_ecef(lat: torch.Tensor, lon: torch.Tensor, elev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert lat/lon/elev to ECEF coordinates using WGS84 ellipsoid."""
    lat_rad = torch.deg2rad(lat)
    lon_rad = torch.deg2rad(lon)
    sin_lat, cos_lat = torch.sin(lat_rad), torch.cos(lat_rad)
    sin_lon, cos_lon = torch.sin(lon_rad), torch.cos(lon_rad)
    N = WGS84_A / torch.sqrt(1 - WGS84_E2 * sin_lat * sin_lat)
    x = (N + elev) * cos_lat * cos_lon
    y = (N + elev) * cos_lat * sin_lon
    z = (N * (1 - WGS84_E2) + elev) * sin_lat
    return x, y, z


@dataclass
class AdaptiveRange:
    """Data-driven ECEF coordinate normalization for regional datasets."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    time_min: float
    time_max: float
    buffer_fraction: float = 0.25
    lat_min: float = 0.0
    lat_max: float = 0.0
    lon_min: float = 0.0
    lon_max: float = 0.0
    elev_min: float = 0.0
    elev_max: float = 0.0

    @classmethod
    def from_ecef_coordinates(cls, ecef_coords: torch.Tensor, time: torch.Tensor,
                               orig_coords: torch.Tensor = None,
                               buffer_fraction: float = 0.25) -> 'AdaptiveRange':
        lat_min = lon_min = elev_min = lat_max = lon_max = elev_max = 0.0
        if orig_coords is not None:
            lat_min, lat_max = orig_coords[:, 0].min().item(), orig_coords[:, 0].max().item()
            lon_min, lon_max = orig_coords[:, 1].min().item(), orig_coords[:, 1].max().item()
            elev_min, elev_max = orig_coords[:, 2].min().item(), orig_coords[:, 2].max().item()
        return cls(
            x_min=ecef_coords[:, 0].min().item(), x_max=ecef_coords[:, 0].max().item(),
            y_min=ecef_coords[:, 1].min().item(), y_max=ecef_coords[:, 1].max().item(),
            z_min=ecef_coords[:, 2].min().item(), z_max=ecef_coords[:, 2].max().item(),
            time_min=time.min().item(), time_max=time.max().item(),
            buffer_fraction=buffer_fraction,
            lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,
            elev_min=elev_min, elev_max=elev_max,
        )

    def get_effective_range(self, dim: str) -> Tuple[float, float]:
        ranges = {'x': (self.x_min, self.x_max), 'y': (self.y_min, self.y_max),
                  'z': (self.z_min, self.z_max), 'time': (self.time_min, self.time_max)}
        data_min, data_max = ranges[dim]
        data_range = max(data_max - data_min, 1.0)
        buffer = data_range * self.buffer_fraction
        return data_min - buffer, data_max + buffer

    def normalize_ecef(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                       time: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_min, x_max = self.get_effective_range('x')
        y_min, y_max = self.get_effective_range('y')
        z_min, z_max = self.get_effective_range('z')
        t_min, t_max = self.get_effective_range('time')
        x_norm = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
        y_norm = 2.0 * (y - y_min) / (y_max - y_min) - 1.0
        z_norm = 2.0 * (z - z_min) / (z_max - z_min) - 1.0
        t_norm = (time - t_min) / (t_max - t_min)
        return x_norm, y_norm, z_norm, t_norm

    def get_coordinate_coverage(self) -> Dict[str, float]:
        global_range = 6400000.0 * 2
        return {
            'x': (self.x_max - self.x_min) / global_range,
            'y': (self.y_max - self.y_min) / global_range,
            'z': (self.z_max - self.z_min) / global_range,
        }

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'AdaptiveRange':
        return cls(**d)


@dataclass
class GeoAdaptiveRange:
    """
    Geographic coordinate normalization with direct lat/lon/elev semantics.

    Unlike ECEF (AdaptiveRange), this preserves latitude relationships:
    - Points at the same latitude share x-coordinate values
    - Points at the same elevation share z-coordinate values
    - The xzt grid (lat, elev, time) enables ecological prior transfer

    Coordinate mapping:
        x = latitude  (-90 to +90 degrees)
        y = longitude (-180 to +180 degrees)
        z = elevation (meters above Mean Sea Level)
        t = time (normalized)

    Example usage:
        # Global coverage (full Earth)
        geo_range = GeoAdaptiveRange.global_range()

        # Fit to training data
        geo_range = GeoAdaptiveRange.from_coordinates(train_coords, buffer_fraction=0.25)

        # Custom regional range
        geo_range = GeoAdaptiveRange(
            lat_min=30.0, lat_max=50.0,
            lon_min=-10.0, lon_max=40.0,
            elev_min=0.0, elev_max=3000.0
        )
    """
    # Latitude range (degrees, -90 to +90)
    lat_min: float = -90.0
    lat_max: float = 90.0

    # Longitude range (degrees, -180 to +180)
    lon_min: float = -180.0
    lon_max: float = 180.0

    # Elevation range (meters above MSL)
    elev_min: float = -500.0   # Below Dead Sea (-430m)
    elev_max: float = 9000.0   # Above Everest (8849m)

    # Time range (normalized)
    time_min: float = 0.0
    time_max: float = 1.0

    # Buffer fraction for generalization beyond training distribution
    buffer_fraction: float = 0.25

    # Mode: 'global' uses full Earth ranges, 'adaptive' fits to data
    mode: str = 'global'

    # Cosine correction for uniform physical resolution across latitude/longitude
    # When enabled, longitude is scaled by cos(mid_lat) to account for the fact that
    # 1° longitude covers less physical distance at higher latitudes
    use_cos_correction: bool = False
    cos_correction_factor: float = 1.0  # cos(mid_lat), computed from range

    # Uniform collision scale - compresses ALL coordinates toward center
    # This forces more hash collisions, pushing more work to learned probing
    collision_scale: float = 1.0  # 1.0 = no compression, 0.5 = compress to 50%

    # Time output scale - compresses time dimension (since time range is fixed [0,1])
    time_output_scale: float = 1.0  # 1.0 = full range, 0.05 = compress to 5%

    @classmethod
    def global_range(cls, elev_min: float = -500.0, elev_max: float = 9000.0,
                     time_min: float = 0.0, time_max: float = 1.0) -> 'GeoAdaptiveRange':
        """
        Factory for global Earth coverage.

        Uses full lat/lon ranges with no buffer (already covers entire Earth).
        Elevation and time ranges can be customized.

        Args:
            elev_min: Minimum elevation in meters (default: -500, below Dead Sea)
            elev_max: Maximum elevation in meters (default: 9000, above Everest)
            time_min: Minimum normalized time (default: 0.0)
            time_max: Maximum normalized time (default: 1.0)
        """
        return cls(
            lat_min=-90.0, lat_max=90.0,
            lon_min=-180.0, lon_max=180.0,
            elev_min=elev_min, elev_max=elev_max,
            time_min=time_min, time_max=time_max,
            buffer_fraction=0.0,  # No buffer for global coverage
            mode='global'
        )

    @classmethod
    def from_coordinates(cls, coords: torch.Tensor,
                        buffer_fraction: float = 0.25,
                        clip_to_globe: bool = True) -> 'GeoAdaptiveRange':
        """
        Fit range to data extent with buffer for generalization.

        Args:
            coords: (N, 4) tensor of [lat, lon, elev, time]
            buffer_fraction: Fractional buffer around data extent (default: 0.25)
            clip_to_globe: If True, clip lat to [-90,90] and lon to [-180,180]

        Returns:
            GeoAdaptiveRange fitted to data with buffer applied
        """
        coords_flat = coords.view(-1, 4)
        lat = coords_flat[:, 0]
        lon = coords_flat[:, 1]
        elev = coords_flat[:, 2]
        time = coords_flat[:, 3]

        # Compute data extent
        lat_data_min, lat_data_max = lat.min().item(), lat.max().item()
        lon_data_min, lon_data_max = lon.min().item(), lon.max().item()
        elev_data_min, elev_data_max = elev.min().item(), elev.max().item()
        time_data_min, time_data_max = time.min().item(), time.max().item()

        # Apply buffer
        lat_range = max(lat_data_max - lat_data_min, 1.0)  # Minimum 1 degree
        lat_buffer = lat_range * buffer_fraction

        lon_range = max(lon_data_max - lon_data_min, 1.0)  # Minimum 1 degree
        lon_buffer = lon_range * buffer_fraction

        elev_range = max(elev_data_max - elev_data_min, 100.0)  # Minimum 100m
        elev_buffer = elev_range * buffer_fraction

        time_range = max(time_data_max - time_data_min, 0.01)
        time_buffer = time_range * buffer_fraction

        # Compute final ranges with buffer
        lat_min = lat_data_min - lat_buffer
        lat_max = lat_data_max + lat_buffer
        lon_min = lon_data_min - lon_buffer
        lon_max = lon_data_max + lon_buffer
        elev_min = elev_data_min - elev_buffer
        elev_max = elev_data_max + elev_buffer
        time_min = max(0.0, time_data_min - time_buffer)
        time_max = min(1.0, time_data_max + time_buffer)

        # Optionally clip to valid globe ranges
        if clip_to_globe:
            lat_min = max(-90.0, lat_min)
            lat_max = min(90.0, lat_max)
            lon_min = max(-180.0, lon_min)
            lon_max = min(180.0, lon_max)

        return cls(
            lat_min=lat_min, lat_max=lat_max,
            lon_min=lon_min, lon_max=lon_max,
            elev_min=elev_min, elev_max=elev_max,
            time_min=time_min, time_max=time_max,
            buffer_fraction=buffer_fraction,
            mode='adaptive'
        )

    @classmethod
    def balanced_regional(cls, coords: torch.Tensor,
                          target_lat_coverage: float = 0.35,
                          target_lon_coverage: float = 0.20,
                          target_elev_coverage: float = 0.15,
                          target_time_coverage: float = 1.0) -> 'GeoAdaptiveRange':
        """
        Create a balanced regional range that matches ECEF-like per-dimension coverage.

        This factory creates ranges where each dimension has similar effective grid
        resolution as ECEF mode, enabling fair comparison and optimal hash utilization.

        The key insight: ECEF concentrates data in ~35% of x, ~17% of y, ~16% of z.
        To match this, we compute buffers that achieve similar coverage for lat/lon/elev.

        Args:
            coords: (N, 4) tensor of [lat, lon, elev, time]
            target_lat_coverage: Target fraction of [-1,1] for latitude (default: 0.35)
            target_lon_coverage: Target fraction of [-1,1] for longitude (default: 0.20)
            target_elev_coverage: Target fraction of [-1,1] for elevation (default: 0.15)
            target_time_coverage: Target fraction of [0,1] for time (default: 1.0 = full range)

        Returns:
            GeoAdaptiveRange with balanced per-dimension coverage
        """
        coords_flat = coords.view(-1, 4)
        lat = coords_flat[:, 0]
        lon = coords_flat[:, 1]
        elev = coords_flat[:, 2]
        time = coords_flat[:, 3]

        # Compute data extent
        lat_data_min, lat_data_max = lat.min().item(), lat.max().item()
        lon_data_min, lon_data_max = lon.min().item(), lon.max().item()
        elev_data_min, elev_data_max = elev.min().item(), elev.max().item()
        time_data_min, time_data_max = time.min().item(), time.max().item()

        lat_data_range = max(lat_data_max - lat_data_min, 1.0)
        lon_data_range = max(lon_data_max - lon_data_min, 1.0)
        elev_data_range = max(elev_data_max - elev_data_min, 100.0)

        # Compute required total ranges to achieve target coverage
        # coverage = data_range / (2 * total_half_range) where total spans [-1, 1]
        # For lat: coverage = lat_data_range / total_lat_range
        # So: total_lat_range = lat_data_range / target_lat_coverage
        lat_total_range = lat_data_range / target_lat_coverage
        lon_total_range = lon_data_range / target_lon_coverage
        elev_total_range = elev_data_range / target_elev_coverage

        # Compute buffer on each side
        lat_buffer = (lat_total_range - lat_data_range) / 2
        lon_buffer = (lon_total_range - lon_data_range) / 2
        elev_buffer = (elev_total_range - elev_data_range) / 2

        # Apply buffers
        lat_min = lat_data_min - lat_buffer
        lat_max = lat_data_max + lat_buffer
        lon_min = lon_data_min - lon_buffer
        lon_max = lon_data_max + lon_buffer
        elev_min = elev_data_min - elev_buffer
        elev_max = elev_data_max + elev_buffer

        # Clip to valid ranges
        lat_min = max(-90.0, lat_min)
        lat_max = min(90.0, lat_max)
        lon_min = max(-180.0, lon_min)
        lon_max = min(180.0, lon_max)

        # Time: store coverage for output scaling (can't expand [0,1] range)
        # Unlike spatial dims, time is pre-normalized to [0,1] so we compress output instead
        time_min = 0.0
        time_max = 1.0
        time_output_scale = target_time_coverage  # Will be applied in normalize()

        return cls(
            lat_min=lat_min, lat_max=lat_max,
            lon_min=lon_min, lon_max=lon_max,
            elev_min=elev_min, elev_max=elev_max,
            time_min=time_min, time_max=time_max,
            buffer_fraction=0.0,  # Buffers already applied via target coverage
            mode='balanced',
            time_output_scale=time_output_scale,
        )

    @classmethod
    def cosine_corrected(cls, coords: torch.Tensor,
                         target_lat_coverage: float = 0.35,
                         target_lon_coverage: float = 0.20,
                         target_elev_coverage: float = 0.15) -> 'GeoAdaptiveRange':
        """
        Create a geographic range with per-sample cosine-corrected longitude.

        This applies two optimizations:
        1. Balanced coverage for optimal hash utilization (like balanced_regional)
        2. Per-sample cosine correction: each point's longitude is scaled by cos(lat)

        The per-sample cosine correction accounts for the fact that 1° longitude
        covers less physical distance at higher latitudes. This makes the normalized
        coordinate space physically accurate (like ECEF) while preserving the
        latitude semantic structure that enables ecological prior transfer.

        Unlike a fixed mid_lat correction, per-sample correction properly handles
        datasets spanning wide latitude ranges where cos(lat) varies significantly.

        Args:
            coords: (N, 4) tensor of [lat, lon, elev, time]
            target_lat_coverage: Target fraction of [-1,1] for latitude (default: 0.35)
            target_lon_coverage: Target fraction of [-1,1] for longitude (default: 0.20)
            target_elev_coverage: Target fraction of [-1,1] for elevation (default: 0.15)

        Returns:
            GeoAdaptiveRange with per-sample cosine correction enabled
        """
        import math

        # Create balanced regional range first
        geo_range = cls.balanced_regional(
            coords,
            target_lat_coverage=target_lat_coverage,
            target_lon_coverage=target_lon_coverage,
            target_elev_coverage=target_elev_coverage
        )

        # Compute mid_lat for reference (actual correction is per-sample)
        mid_lat = (geo_range.lat_min + geo_range.lat_max) / 2.0
        cos_factor = math.cos(math.radians(mid_lat))

        # Enable per-sample cosine correction
        geo_range.use_cos_correction = True
        geo_range.cos_correction_factor = cos_factor  # For reference only
        geo_range.mode = 'cosine_corrected'

        return geo_range

    def get_effective_range(self, dim: str) -> Tuple[float, float]:
        """Get the effective range for a dimension (with buffer already applied in from_coordinates)."""
        ranges = {
            'lat': (self.lat_min, self.lat_max),
            'lon': (self.lon_min, self.lon_max),
            'elev': (self.elev_min, self.elev_max),
            'time': (self.time_min, self.time_max),
            # Aliases for compatibility
            'x': (self.lat_min, self.lat_max),
            'y': (self.lon_min, self.lon_max),
            'z': (self.elev_min, self.elev_max),
        }
        return ranges[dim]

    def normalize(self, lat: torch.Tensor, lon: torch.Tensor,
                  elev: torch.Tensor, time: torch.Tensor,
                  warn_on_out_of_bounds: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Normalize geographic coordinates to [-1, 1] range for spatial, [0, 1] for time.

        Args:
            lat: Latitude in degrees (-90 to +90)
            lon: Longitude in degrees (-180 to +180)
            elev: Elevation in meters above MSL
            time: Normalized time
            warn_on_out_of_bounds: If True, warn when coordinates exceed fitted range

        Returns:
            Tuple of (lat_norm, lon_norm, elev_norm, time_norm) as x, y, z, t
        """
        # Check for out-of-bounds coordinates
        if warn_on_out_of_bounds:
            self._check_bounds(lat, lon, elev, time)

        # Normalize to [-1, 1] for spatial dimensions
        lat_range = max(self.lat_max - self.lat_min, 1e-6)
        lon_range = max(self.lon_max - self.lon_min, 1e-6)
        elev_range = max(self.elev_max - self.elev_min, 1e-6)
        time_range = max(self.time_max - self.time_min, 1e-6)

        lat_norm = 2.0 * (lat - self.lat_min) / lat_range - 1.0
        lon_norm = 2.0 * (lon - self.lon_min) / lon_range - 1.0
        elev_norm = 2.0 * (elev - self.elev_min) / elev_range - 1.0
        time_norm = (time - self.time_min) / time_range

        # Apply uniform collision scale to ALL coordinates
        # This compresses data toward center, forcing more hash collisions
        # and pushing more learning into the learned probing mechanism
        if self.collision_scale != 1.0:
            lat_norm = lat_norm * self.collision_scale
            lon_norm = lon_norm * self.collision_scale
            elev_norm = elev_norm * self.collision_scale
            time_norm = time_norm * self.collision_scale

        # Apply time-specific output scale (for time coverage parameter)
        # This compresses time to use fewer resolution levels effectively
        if self.time_output_scale != 1.0:
            time_norm = time_norm * self.time_output_scale

        return lat_norm, lon_norm, elev_norm, time_norm

    def _check_bounds(self, lat: torch.Tensor, lon: torch.Tensor,
                      elev: torch.Tensor, time: torch.Tensor) -> None:
        """Check for out-of-bounds coordinates and issue warnings with details."""
        lat_min_data, lat_max_data = lat.min().item(), lat.max().item()
        lon_min_data, lon_max_data = lon.min().item(), lon.max().item()
        elev_min_data, elev_max_data = elev.min().item(), elev.max().item()

        oob_messages = []

        if lat_min_data < self.lat_min:
            delta_km = (self.lat_min - lat_min_data) * 111.0  # ~111km per degree
            oob_messages.append(f"Lat below range by {delta_km:.1f} km ({lat_min_data:.2f}° < {self.lat_min:.2f}°)")
        if lat_max_data > self.lat_max:
            delta_km = (lat_max_data - self.lat_max) * 111.0
            oob_messages.append(f"Lat above range by {delta_km:.1f} km ({lat_max_data:.2f}° > {self.lat_max:.2f}°)")

        if lon_min_data < self.lon_min:
            delta_km = (self.lon_min - lon_min_data) * 111.0 * 0.7  # Approximate
            oob_messages.append(f"Lon below range by ~{delta_km:.1f} km ({lon_min_data:.2f}° < {self.lon_min:.2f}°)")
        if lon_max_data > self.lon_max:
            delta_km = (lon_max_data - self.lon_max) * 111.0 * 0.7
            oob_messages.append(f"Lon above range by ~{delta_km:.1f} km ({lon_max_data:.2f}° > {self.lon_max:.2f}°)")

        if elev_min_data < self.elev_min:
            delta_m = self.elev_min - elev_min_data
            oob_messages.append(f"Elev below range by {delta_m:.0f} m ({elev_min_data:.0f}m < {self.elev_min:.0f}m)")
        if elev_max_data > self.elev_max:
            delta_m = elev_max_data - self.elev_max
            oob_messages.append(f"Elev above range by {delta_m:.0f} m ({elev_max_data:.0f}m > {self.elev_max:.0f}m)")

        if oob_messages:
            warnings.warn(
                f"Coordinates outside fitted range:\n  " + "\n  ".join(oob_messages) +
                f"\nConsider re-fitting range with fit_range() including all data.",
                UserWarning
            )

    def get_resolution_in_units(self, grid_resolution: int) -> Dict[str, float]:
        """
        Get physical resolution at a given grid resolution.

        Args:
            grid_resolution: Number of grid cells along each dimension

        Returns:
            Dict with 'lat_deg', 'lat_km', 'lon_deg', 'elev_m' per cell
        """
        lat_range = self.lat_max - self.lat_min
        lon_range = self.lon_max - self.lon_min
        elev_range = self.elev_max - self.elev_min

        lat_deg_per_cell = lat_range / grid_resolution
        lon_deg_per_cell = lon_range / grid_resolution
        elev_m_per_cell = elev_range / grid_resolution

        # Convert latitude to km (~111km per degree)
        lat_km_per_cell = lat_deg_per_cell * 111.0

        return {
            'lat_deg': lat_deg_per_cell,
            'lat_km': lat_km_per_cell,
            'lon_deg': lon_deg_per_cell,
            'elev_m': elev_m_per_cell
        }

    def get_coordinate_coverage(self) -> Dict[str, float]:
        """
        Get fraction of global range covered by this range.

        Returns:
            Dict with coverage fractions for lat, lon, elev
        """
        return {
            'lat': (self.lat_max - self.lat_min) / 180.0,  # Full range is 180 degrees
            'lon': (self.lon_max - self.lon_min) / 360.0,  # Full range is 360 degrees
            'elev': (self.elev_max - self.elev_min) / 9500.0,  # Assume ~9500m typical range
        }

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'GeoAdaptiveRange':
        """Create from dictionary."""
        return cls(**d)

    def __repr__(self) -> str:
        return (
            f"GeoAdaptiveRange(mode='{self.mode}', "
            f"lat=[{self.lat_min:.2f}, {self.lat_max:.2f}]°, "
            f"lon=[{self.lon_min:.2f}, {self.lon_max:.2f}]°, "
            f"elev=[{self.elev_min:.0f}, {self.elev_max:.0f}]m)"
        )
