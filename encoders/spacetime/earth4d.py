"""Earth4D: planetary (X,Y,Z,T) hash-grid positional encoder."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Literal, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn

from hashencoder.hashgrid import HashEncoder

# WGS84 ellipsoid
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2 * WGS84_F - WGS84_F**2
ECEF_NORM_FACTOR = 6400000.0


def to_ecef(lat: torch.Tensor, lon: torch.Tensor, elev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert lat/lon/elev to ECEF coordinates using the WGS84 ellipsoid."""
    lat_rad, lon_rad = torch.deg2rad(lat), torch.deg2rad(lon)
    sin_lat, cos_lat = torch.sin(lat_rad), torch.cos(lat_rad)
    sin_lon, cos_lon = torch.sin(lon_rad), torch.cos(lon_rad)
    N = WGS84_A / torch.sqrt(1 - WGS84_E2 * sin_lat * sin_lat)
    return (N + elev) * cos_lat * cos_lon, (N + elev) * cos_lat * sin_lon, (N * (1 - WGS84_E2) + elev) * sin_lat


@dataclass
class AdaptiveRange:
    """Data-driven ECEF coordinate normalization for regional datasets."""
    x_min: float; x_max: float; y_min: float; y_max: float; z_min: float; z_max: float
    time_min: float; time_max: float
    buffer_fraction: float = 0.25
    lat_min: float = 0.0; lat_max: float = 0.0; lon_min: float = 0.0; lon_max: float = 0.0
    elev_min: float = 0.0; elev_max: float = 0.0

    @classmethod
    def from_ecef_coordinates(cls, ecef_coords: torch.Tensor, time: torch.Tensor,
                               orig_coords: torch.Tensor = None, buffer_fraction: float = 0.25) -> 'AdaptiveRange':
        lat_min = lon_min = elev_min = lat_max = lon_max = elev_max = 0.0
        if orig_coords is not None:
            lat_min, lat_max = orig_coords[:, 0].min().item(), orig_coords[:, 0].max().item()
            lon_min, lon_max = orig_coords[:, 1].min().item(), orig_coords[:, 1].max().item()
            elev_min, elev_max = orig_coords[:, 2].min().item(), orig_coords[:, 2].max().item()
        return cls(
            x_min=ecef_coords[:, 0].min().item(), x_max=ecef_coords[:, 0].max().item(),
            y_min=ecef_coords[:, 1].min().item(), y_max=ecef_coords[:, 1].max().item(),
            z_min=ecef_coords[:, 2].min().item(), z_max=ecef_coords[:, 2].max().item(),
            time_min=time.min().item(), time_max=time.max().item(), buffer_fraction=buffer_fraction,
            lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max, elev_min=elev_min, elev_max=elev_max)

    def get_effective_range(self, dim: str) -> Tuple[float, float]:
        ranges = {'x': (self.x_min, self.x_max), 'y': (self.y_min, self.y_max),
                  'z': (self.z_min, self.z_max), 'time': (self.time_min, self.time_max)}
        data_min, data_max = ranges[dim]
        buffer = max(data_max - data_min, 1.0) * self.buffer_fraction
        return data_min - buffer, data_max + buffer

    def normalize_ecef(self, x, y, z, time) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_min, x_max = self.get_effective_range('x'); y_min, y_max = self.get_effective_range('y')
        z_min, z_max = self.get_effective_range('z'); t_min, t_max = self.get_effective_range('time')
        x_norm = 2.0 * (x - x_min) / (x_max - x_min) - 1.0; y_norm = 2.0 * (y - y_min) / (y_max - y_min) - 1.0
        z_norm = 2.0 * (z - z_min) / (z_max - z_min) - 1.0; t_norm = (time - t_min) / (t_max - t_min)
        return x_norm, y_norm, z_norm, t_norm

    def get_coordinate_coverage(self) -> Dict[str, float]:
        gr = 6400000.0 * 2
        return {'x': (self.x_max - self.x_min) / gr, 'y': (self.y_max - self.y_min) / gr, 'z': (self.z_max - self.z_min) / gr}

    def to_dict(self) -> dict: return asdict(self)
    @classmethod
    def from_dict(cls, d: dict) -> 'AdaptiveRange': return cls(**d)


@dataclass
class GeoAdaptiveRange:
    """Geographic normalization with direct lat/lon/elev semantics (x=lat, y=lon, z=elev, t=time)."""
    lat_min: float = -90.0; lat_max: float = 90.0
    lon_min: float = -180.0; lon_max: float = 180.0
    elev_min: float = -500.0; elev_max: float = 9000.0    # below Dead Sea .. above Everest
    time_min: float = 0.0; time_max: float = 1.0
    buffer_fraction: float = 0.25
    mode: str = 'global'
    use_cos_correction: bool = False              # scale lon by cos(mid_lat) for uniform physical resolution
    cos_correction_factor: float = 1.0
    collision_scale: float = 1.0                  # <1 compresses coords toward center, forcing hash collisions
    time_output_scale: float = 1.0                # <1 compresses time dimension

    @classmethod
    def global_range(cls, elev_min: float = -500.0, elev_max: float = 9000.0,
                     time_min: float = 0.0, time_max: float = 1.0) -> 'GeoAdaptiveRange':
        """Full Earth coverage, no buffer."""
        return cls(lat_min=-90.0, lat_max=90.0, lon_min=-180.0, lon_max=180.0,
                   elev_min=elev_min, elev_max=elev_max, time_min=time_min, time_max=time_max,
                   buffer_fraction=0.0, mode='global')

    @classmethod
    def from_coordinates(cls, coords: torch.Tensor, buffer_fraction: float = 0.25,
                        clip_to_globe: bool = True) -> 'GeoAdaptiveRange':
        """Fit range to data extent with buffer for generalization."""
        c = coords.view(-1, 4); lat, lon, elev, time = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
        lat_dmin, lat_dmax = lat.min().item(), lat.max().item(); lon_dmin, lon_dmax = lon.min().item(), lon.max().item()
        elev_dmin, elev_dmax = elev.min().item(), elev.max().item(); time_dmin, time_dmax = time.min().item(), time.max().item()
        lat_buf = max(lat_dmax - lat_dmin, 1.0) * buffer_fraction; lon_buf = max(lon_dmax - lon_dmin, 1.0) * buffer_fraction
        elev_buf = max(elev_dmax - elev_dmin, 100.0) * buffer_fraction; time_buf = max(time_dmax - time_dmin, 0.01) * buffer_fraction
        lat_min, lat_max = lat_dmin - lat_buf, lat_dmax + lat_buf; lon_min, lon_max = lon_dmin - lon_buf, lon_dmax + lon_buf
        elev_min, elev_max = elev_dmin - elev_buf, elev_dmax + elev_buf
        time_min, time_max = max(0.0, time_dmin - time_buf), min(1.0, time_dmax + time_buf)
        if clip_to_globe:
            lat_min, lat_max = max(-90.0, lat_min), min(90.0, lat_max); lon_min, lon_max = max(-180.0, lon_min), min(180.0, lon_max)
        return cls(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,
                   elev_min=elev_min, elev_max=elev_max, time_min=time_min, time_max=time_max,
                   buffer_fraction=buffer_fraction, mode='adaptive')

    @classmethod
    def balanced_regional(cls, coords: torch.Tensor, target_lat_coverage: float = 0.35,
                          target_lon_coverage: float = 0.20, target_elev_coverage: float = 0.15,
                          target_time_coverage: float = 1.0) -> 'GeoAdaptiveRange':
        """Regional range whose per-dim [-1,1] coverage matches ECEF-like utilization (coverage = data_range/total_range)."""
        c = coords.view(-1, 4); lat, lon, elev = c[:, 0], c[:, 1], c[:, 2]
        lat_dmin, lat_dmax = lat.min().item(), lat.max().item(); lon_dmin, lon_dmax = lon.min().item(), lon.max().item()
        elev_dmin, elev_dmax = elev.min().item(), elev.max().item()
        lat_dr = max(lat_dmax - lat_dmin, 1.0); lon_dr = max(lon_dmax - lon_dmin, 1.0); elev_dr = max(elev_dmax - elev_dmin, 100.0)
        lat_buf = (lat_dr / target_lat_coverage - lat_dr) / 2; lon_buf = (lon_dr / target_lon_coverage - lon_dr) / 2
        elev_buf = (elev_dr / target_elev_coverage - elev_dr) / 2
        lat_min, lat_max = max(-90.0, lat_dmin - lat_buf), min(90.0, lat_dmax + lat_buf)
        lon_min, lon_max = max(-180.0, lon_dmin - lon_buf), min(180.0, lon_dmax + lon_buf)
        elev_min, elev_max = elev_dmin - elev_buf, elev_dmax + elev_buf
        return cls(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,
                   elev_min=elev_min, elev_max=elev_max, time_min=0.0, time_max=1.0,
                   buffer_fraction=0.0, mode='balanced', time_output_scale=target_time_coverage)

    @classmethod
    def cosine_corrected(cls, coords: torch.Tensor, target_lat_coverage: float = 0.35,
                         target_lon_coverage: float = 0.20, target_elev_coverage: float = 0.15) -> 'GeoAdaptiveRange':
        """balanced_regional plus per-sample cos(lat) longitude correction for physical accuracy."""
        import math
        gr = cls.balanced_regional(coords, target_lat_coverage, target_lon_coverage, target_elev_coverage)
        mid_lat = (gr.lat_min + gr.lat_max) / 2.0
        gr.use_cos_correction = True; gr.cos_correction_factor = math.cos(math.radians(mid_lat)); gr.mode = 'cosine_corrected'
        return gr

    def get_effective_range(self, dim: str) -> Tuple[float, float]:
        ranges = {'lat': (self.lat_min, self.lat_max), 'lon': (self.lon_min, self.lon_max),
                  'elev': (self.elev_min, self.elev_max), 'time': (self.time_min, self.time_max),
                  'x': (self.lat_min, self.lat_max), 'y': (self.lon_min, self.lon_max), 'z': (self.elev_min, self.elev_max)}
        return ranges[dim]

    def normalize(self, lat, lon, elev, time, warn_on_out_of_bounds: bool = True):
        """Normalize to [-1,1] spatial, [0,1] time; apply collision_scale and time_output_scale."""
        if warn_on_out_of_bounds:
            self._check_bounds(lat, lon, elev, time)
        lat_range = max(self.lat_max - self.lat_min, 1e-6); lon_range = max(self.lon_max - self.lon_min, 1e-6)
        elev_range = max(self.elev_max - self.elev_min, 1e-6); time_range = max(self.time_max - self.time_min, 1e-6)
        lat_norm = 2.0 * (lat - self.lat_min) / lat_range - 1.0; lon_norm = 2.0 * (lon - self.lon_min) / lon_range - 1.0
        elev_norm = 2.0 * (elev - self.elev_min) / elev_range - 1.0; time_norm = (time - self.time_min) / time_range
        if self.collision_scale != 1.0:
            lat_norm, lon_norm = lat_norm * self.collision_scale, lon_norm * self.collision_scale
            elev_norm, time_norm = elev_norm * self.collision_scale, time_norm * self.collision_scale
        if self.time_output_scale != 1.0:
            time_norm = time_norm * self.time_output_scale
        return lat_norm, lon_norm, elev_norm, time_norm

    def _check_bounds(self, lat, lon, elev, time) -> None:
        msgs = []
        if lat.min().item() < self.lat_min:
            msgs.append(f"Lat below range by {(self.lat_min - lat.min().item()) * 111.0:.1f} km")
        if lat.max().item() > self.lat_max:
            msgs.append(f"Lat above range by {(lat.max().item() - self.lat_max) * 111.0:.1f} km")
        if lon.min().item() < self.lon_min:
            msgs.append(f"Lon below range by ~{(self.lon_min - lon.min().item()) * 111.0 * 0.7:.1f} km")
        if lon.max().item() > self.lon_max:
            msgs.append(f"Lon above range by ~{(lon.max().item() - self.lon_max) * 111.0 * 0.7:.1f} km")
        if elev.min().item() < self.elev_min:
            msgs.append(f"Elev below range by {self.elev_min - elev.min().item():.0f} m")
        if elev.max().item() > self.elev_max:
            msgs.append(f"Elev above range by {elev.max().item() - self.elev_max:.0f} m")
        if msgs:
            warnings.warn("Coordinates outside fitted range:\n  " + "\n  ".join(msgs) +
                          "\nConsider re-fitting range with fit_range() including all data.", UserWarning)

    def get_resolution_in_units(self, grid_resolution: int) -> Dict[str, float]:
        """Physical resolution per grid cell at a given resolution."""
        lat_deg = (self.lat_max - self.lat_min) / grid_resolution
        return {'lat_deg': lat_deg, 'lat_km': lat_deg * 111.0,
                'lon_deg': (self.lon_max - self.lon_min) / grid_resolution,
                'elev_m': (self.elev_max - self.elev_min) / grid_resolution}

    def get_coordinate_coverage(self) -> Dict[str, float]:
        return {'lat': (self.lat_max - self.lat_min) / 180.0, 'lon': (self.lon_max - self.lon_min) / 360.0,
                'elev': (self.elev_max - self.elev_min) / 9500.0}

    def to_dict(self) -> dict: return asdict(self)
    @classmethod
    def from_dict(cls, d: dict) -> 'GeoAdaptiveRange': return cls(**d)

    def __repr__(self) -> str:
        return (f"GeoAdaptiveRange(mode='{self.mode}', lat=[{self.lat_min:.2f}, {self.lat_max:.2f}]°, "
                f"lon=[{self.lon_min:.2f}, {self.lon_max:.2f}]°, elev=[{self.elev_min:.0f}, {self.elev_max:.0f}]m)")


def parse_timestamp(timestamp: str, time_range: Tuple[int, int] = (1900, 2100)) -> float:
    """Parse a human-readable timestamp (e.g. '1941-06-01 09:00 GMT') into normalized time [0, 1]."""
    tz_offsets = {'UTC': 0, 'GMT': 0, 'Z': 0, 'ET': -5, 'EST': -5, 'EDT': -4, 'CT': -6, 'CST': -6, 'CDT': -5,
                  'MT': -7, 'MST': -7, 'MDT': -6, 'PT': -8, 'PST': -8, 'PDT': -7, 'CET': 1, 'CEST': 2,
                  'JST': 9, 'KST': 9, 'IST': 5.5, 'AEST': 10, 'AEDT': 11}
    parts = timestamp.strip().split()
    tz_str = parts[-1].upper() if parts else 'UTC'
    if tz_str in tz_offsets:
        tz_offset, timestamp_clean = tz_offsets[tz_str], ' '.join(parts[:-1])
    else:
        tz_offset, timestamp_clean = 0, timestamp
    formats = ["%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M",
               "%Y/%m/%d", "%d-%m-%Y %H:%M", "%d/%m/%Y %H:%M"]
    dt = None
    for fmt in formats:
        try:
            dt = datetime.strptime(timestamp_clean.strip(), fmt); break
        except ValueError:
            continue
    if dt is None:
        raise ValueError(f"Could not parse timestamp: {timestamp}")
    unix_ts = dt.timestamp() - (tz_offset * 3600)
    start_ts = datetime(time_range[0], 1, 1).timestamp(); end_ts = datetime(time_range[1], 1, 1).timestamp()
    return max(0.0, min(1.0, (unix_ts - start_ts) / (end_ts - start_ts)))


class Earth4D(nn.Module):
    """Multi-resolution hash-grid encoder over (lat, lon, elev, time): one xyz spatial encoder plus three
    spatiotemporal projections (xyt, yzt, xzt), with optional absolute and relative (offset) channels."""

    def __init__(self,
                 spatial_levels: int = 24,
                 temporal_levels: int = 24,
                 features_per_level: int = 2,
                 spatial_log2_hashmap_size: int = 22,
                 temporal_log2_hashmap_size: int = 22,
                 base_spatial_resolution: float = 32.0,
                 base_temporal_resolution: float = 32.0,
                 growth_factor: float = 2.0,
                 temporal_growth_factor: float = None,   # None -> growth_factor
                 latlon_growth_factor: float = None,     # None -> encoder-specific default
                 elev_growth_factor: float = None,       # None -> encoder-specific default
                 verbose: bool = True,
                 enable_learned_probing: bool = True,
                 probing_range: int = 32,
                 index_codebook_size: int = 512,
                 probe_entropy_weight: float = 0.0,
                 use_adaptive_range: bool = False,
                 adaptive_range: Optional[AdaptiveRange] = None,
                 max_precision_level: Optional[int] = None,
                 coordinate_system: Literal['geographic', 'ecef'] = 'ecef',
                 geo_range: Optional[GeoAdaptiveRange] = None,
                 resolution_mode: str = 'balanced',
                 enable_absolute: bool = True,
                 freq_log_scale_init: float = 0.0,
                 enable_relative: bool = False,
                 relative_levels: int = 24,
                 relative_log2_hashmap_size: int = 20,
                 relative_window: Tuple[float, float, float, float] = (8000.0, 8000.0, 300.0, 130.0),
                 relative_finest: Tuple[float, float, float, float] = (15.0, 15.0, 3.0, 1.0)):
        super().__init__()
        self.verbose = verbose; self.enable_learned_probing = enable_learned_probing
        self.probing_range = probing_range; self.index_codebook_size = index_codebook_size
        self.probe_entropy_weight = probe_entropy_weight; self.use_adaptive_range = use_adaptive_range
        self.adaptive_range = adaptive_range; self.max_precision_level = max_precision_level
        self.base_spatial_resolution = base_spatial_resolution; self.base_temporal_resolution = base_temporal_resolution
        self.spatial_levels = spatial_levels; self.temporal_levels = temporal_levels; self.growth_factor = growth_factor
        self.temporal_growth_factor = temporal_growth_factor if temporal_growth_factor is not None else growth_factor
        self.features_per_level = features_per_level
        # decoupled growth for lat/lon vs elevation (xyz -> growth_factor, spatiotemporal -> temporal_growth_factor)
        self.latlon_growth_factor = latlon_growth_factor; self.elev_growth_factor = elev_growth_factor
        self.coordinate_system = coordinate_system; self.resolution_mode = resolution_mode; self.geo_range = geo_range
        if coordinate_system == 'geographic' and geo_range is None:
            self.geo_range = GeoAdaptiveRange.global_range(); self._geo_range_is_default = True
        else:
            self._geo_range_is_default = False
        if HashEncoder is None:
            raise ImportError("HashEncoder is required for Earth4D. Please install the hash encoding library.")
        self.spatial_log2_hashmap_size = spatial_log2_hashmap_size; self.temporal_log2_hashmap_size = temporal_log2_hashmap_size
        self.spatial_dim = spatial_levels * features_per_level
        self.spatiotemporal_dim = temporal_levels * features_per_level * 3    # 3 projections
        self.output_dim = self.spatial_dim + self.spatiotemporal_dim

        spatial_max_res = int(base_spatial_resolution * (growth_factor ** (spatial_levels - 1)))
        temporal_base_res = [int(base_temporal_resolution)] * 3
        tgf = self.temporal_growth_factor
        llgf, egf = latlon_growth_factor, elev_growth_factor    # may be None
        # xyz encoder [lat, lon, elev]: per-axis growth if lat/lon or elev decoupled, else uniform growth_factor
        if llgf is not None or egf is not None:
            xyz_llgf = llgf if llgf is not None else growth_factor; xyz_egf = egf if egf is not None else growth_factor
            xyz_scale = [xyz_llgf, xyz_llgf, xyz_egf]
            xyz_max_res = [int(base_spatial_resolution * (xyz_llgf ** (spatial_levels - 1))),
                           int(base_spatial_resolution * (xyz_llgf ** (spatial_levels - 1))),
                           int(base_spatial_resolution * (xyz_egf ** (spatial_levels - 1)))]
        else:
            xyz_scale, xyz_max_res = growth_factor, spatial_max_res
        # spatiotemporal encoders (xyt, yzt, xzt): temporal_growth_factor baseline
        if llgf is not None or egf is not None:
            st_llgf = llgf if llgf is not None else tgf; st_egf = egf if egf is not None else tgf
            xyt_scale = [st_llgf, st_llgf, tgf]
            xyt_max_res = [int(base_temporal_resolution * (st_llgf ** (temporal_levels - 1))),
                           int(base_temporal_resolution * (st_llgf ** (temporal_levels - 1))),
                           int(base_temporal_resolution * (tgf ** (temporal_levels - 1)))]
            yzt_scale = xzt_scale = [st_llgf, st_egf, tgf]
            yzt_max_res = xzt_max_res = [int(base_temporal_resolution * (st_llgf ** (temporal_levels - 1))),
                                         int(base_temporal_resolution * (st_egf ** (temporal_levels - 1))),
                                         int(base_temporal_resolution * (tgf ** (temporal_levels - 1)))]
        else:
            xyt_scale = yzt_scale = xzt_scale = tgf
            temporal_max_res = [int(base_temporal_resolution * (tgf ** (temporal_levels - 1)))] * 3
            xyt_max_res = yzt_max_res = xzt_max_res = temporal_max_res

        self.xyz_encoder = HashEncoder(
            input_dim=3, num_levels=spatial_levels, level_dim=features_per_level, per_level_scale=xyz_scale,
            base_resolution=int(base_spatial_resolution), log2_hashmap_size=spatial_log2_hashmap_size,
            desired_resolution=xyz_max_res, enable_learned_probing=enable_learned_probing,
            probing_range=probing_range, index_codebook_size=index_codebook_size)
        self.xyt_encoder = HashEncoder(
            input_dim=3, num_levels=temporal_levels, level_dim=features_per_level, per_level_scale=xyt_scale,
            base_resolution=temporal_base_res, log2_hashmap_size=temporal_log2_hashmap_size,
            desired_resolution=xyt_max_res, enable_learned_probing=enable_learned_probing,
            probing_range=probing_range, index_codebook_size=index_codebook_size)
        self.yzt_encoder = HashEncoder(
            input_dim=3, num_levels=temporal_levels, level_dim=features_per_level, per_level_scale=yzt_scale,
            base_resolution=temporal_base_res, log2_hashmap_size=temporal_log2_hashmap_size,
            desired_resolution=yzt_max_res, enable_learned_probing=enable_learned_probing,
            probing_range=probing_range, index_codebook_size=index_codebook_size)
        self.xzt_encoder = HashEncoder(
            input_dim=3, num_levels=temporal_levels, level_dim=features_per_level, per_level_scale=xzt_scale,
            base_resolution=temporal_base_res, log2_hashmap_size=temporal_log2_hashmap_size,
            desired_resolution=xzt_max_res, enable_learned_probing=enable_learned_probing,
            probing_range=probing_range, index_codebook_size=index_codebook_size)
        # a relative-only field never reads the four absolute projections, so it can opt out of carrying them
        self.enable_absolute = enable_absolute
        if not enable_absolute:
            self.xyz_encoder = self.xyt_encoder = self.yzt_encoder = self.xzt_encoder = None

        # RELATIVE (translation-equivariant): encode the OFFSET between two observations, so a pattern learned at one
        # place/time applies everywhere. Window fitted to the offset distribution; four 3D projections, per-axis res.
        self.enable_relative = enable_relative
        if enable_relative:
            self._rel_projections = ((0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3))    # axes: 0=N,1=E,2=elev,3=time
            self.register_buffer('_rel_window', torch.tensor(list(relative_window), dtype=torch.float32))
            base_r = 16
            rel_res = [max(int(round(2.0 * relative_window[a] / relative_finest[a])), base_r + 1) for a in range(4)]
            self.rel_encoders = nn.ModuleList(
                HashEncoder(input_dim=3, num_levels=relative_levels, level_dim=features_per_level,
                            base_resolution=base_r, desired_resolution=[rel_res[a] for a in axes],
                            log2_hashmap_size=relative_log2_hashmap_size,
                            enable_learned_probing=enable_learned_probing, probing_range=probing_range,
                            index_codebook_size=index_codebook_size)
                for axes in self._rel_projections)
            self.relative_output_dim = 4 * relative_levels * features_per_level

        # learnable per-axis frequency (scale + center), tanh-bounded to [-0.9, 0.9]: picks each axis's effective
        # resolution range. Axes are (N/x, E/y, elev/z, time); initialized to identity.
        self.freq_log_scale = nn.Parameter(torch.full((4,), float(freq_log_scale_init)))
        self.freq_center = nn.Parameter(torch.zeros(4))
        if self.verbose and not self.use_adaptive_range:
            self._print_resolution_info()

    def _print_resolution_info(self):
        config = {'use_adaptive_range': self.use_adaptive_range, 'enable_learned_probing': self.enable_learned_probing,
                  'probing_range': self.probing_range, 'max_precision_level': self.max_precision_level,
                  'spatial_levels': self.spatial_levels, 'coordinate_system': self.coordinate_system}
        adaptive_range = self.geo_range if self.coordinate_system == 'geographic' else self.adaptive_range
        print_resolution_info(self, config, adaptive_range)

    def _encode_spatial(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.xyz_encoder(xyz, size=1.0)

    def _encode_spatiotemporal(self, xyzt: torch.Tensor) -> torch.Tensor:
        t_scaled = (xyzt[..., 3:] * 2 - 1) * 0.9
        xyzt_scaled = torch.cat([xyzt[..., :3], t_scaled], dim=-1)
        xyt = torch.cat([xyzt_scaled[..., :2], xyzt_scaled[..., 3:]], dim=-1)
        yzt = xyzt_scaled[..., 1:]
        xzt = torch.cat([xyzt_scaled[..., :1], xyzt_scaled[..., 2:]], dim=-1)
        return torch.cat([self.xyt_encoder(xyt, size=1.0), self.yzt_encoder(yzt, size=1.0),
                          self.xzt_encoder(xzt, size=1.0)], dim=-1)

    def _learnable_freq(self, norm: torch.Tensor) -> torch.Tensor:
        """Learnable per-axis frequency (scale + center), bounded to [-0.9, 0.9] by tanh."""
        return 0.9 * torch.tanh((norm - self.freq_center) * self.freq_log_scale.exp())

    def pyramid(self, coords: torch.Tensor) -> torch.Tensor:
        """Per-projection, per-level features ``[..., 4, n_levels, features_per_level]`` (coarse->fine) for scale
        mixing; four projections (xyz, xyt, yzt, xzt). ``coords`` is ``[..., 4]`` = (lat, lon, elev, time in [0,1])."""
        assert self.spatial_levels == self.temporal_levels, "pyramid() needs spatial_levels == temporal_levels"
        L, F = self.spatial_levels, self.features_per_level
        if self.coordinate_system == 'geographic':
            x_norm, y_norm, z_norm, time_norm = self.geo_range.normalize(
                coords[..., 0], coords[..., 1], coords[..., 2], coords[..., 3])
        else:
            x, y, z = to_ecef(coords[..., 0], coords[..., 1], coords[..., 2])
            if self.use_adaptive_range and self.adaptive_range is not None:
                x_norm, y_norm, z_norm, time_norm = self.adaptive_range.normalize_ecef(x, y, z, coords[..., 3])
            else:
                x_norm, y_norm, z_norm = x / ECEF_NORM_FACTOR, y / ECEF_NORM_FACTOR, z / ECEF_NORM_FACTOR
                time_norm = coords[..., 3]
        norm = torch.stack([x_norm, y_norm, z_norm, time_norm * 2 - 1], dim=-1)     # time to [-1,1] like the rest
        norm = self._learnable_freq(norm); t = norm[..., 3:]
        projections = [norm[..., :3],
                       torch.cat([norm[..., :2], t], dim=-1),
                       torch.cat([norm[..., 1:3], t], dim=-1),
                       torch.cat([norm[..., :1], norm[..., 2:3], t], dim=-1)]
        encoders = [self.xyz_encoder, self.xyt_encoder, self.yzt_encoder, self.xzt_encoder]
        feats = [enc(proj, size=1.0).reshape(*proj.shape[:-1], L, F) for enc, proj in zip(encoders, projections)]
        return torch.stack(feats, dim=-3)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Encode coordinates. Accepts a tensor ``(..., 4)`` of [lat, lon, elev, time_norm], or coordinate tuples
        ``(lat, lon, elev, "timestamp string")``. Returns concatenated features ``(N, output_dim)``."""
        if len(args) > 0 and isinstance(args[0], (tuple, list)):
            coords = self._parse_coordinate_tuples(args)
        elif len(args) == 1 and isinstance(args[0], torch.Tensor):
            coords = args[0]
        else:
            raise ValueError("Expected either a tensor or coordinate tuples. "
                             "Example: model((lat, lon, elev, 'timestamp'), ...)")
        return self._forward_tensor(coords)

    def _parse_coordinate_tuples(self, tuples) -> torch.Tensor:
        coords_list = []
        for t in tuples:
            if len(t) != 4:
                raise ValueError(f"Each coordinate tuple must have 4 elements (lat, lon, elev, time), got {len(t)}")
            lat, lon, elev, time_val = t
            time_norm = parse_timestamp(time_val) if isinstance(time_val, str) else float(time_val)
            coords_list.append([float(lat), float(lon), float(elev), time_norm])
        device = next(self.parameters()).device
        return torch.tensor(coords_list, dtype=torch.float32, device=device)

    def _normalize_coords(self, coords: torch.Tensor):
        """Shared coordinate normalization -> stacked [x,y,z,time] in the encoder's units."""
        if self.coordinate_system == 'geographic':
            x_norm, y_norm, z_norm, time_norm = self.geo_range.normalize(
                coords[..., 0], coords[..., 1], coords[..., 2], coords[..., 3])
        else:
            x, y, z = to_ecef(coords[..., 0], coords[..., 1], coords[..., 2]); time = coords[..., 3]
            if self.use_adaptive_range and self.adaptive_range is not None:
                x_norm, y_norm, z_norm, time_norm = self.adaptive_range.normalize_ecef(x, y, z, time)
            else:
                x_norm, y_norm, z_norm = x / ECEF_NORM_FACTOR, y / ECEF_NORM_FACTOR, z / ECEF_NORM_FACTOR
                time_norm = time
        return torch.stack([x_norm, y_norm, z_norm, time_norm], dim=-1)

    def _forward_tensor(self, coords: torch.Tensor) -> torch.Tensor:
        norm_coords = self._normalize_coords(coords)
        return torch.cat([self._encode_spatial(norm_coords[..., :3]),
                          self._encode_spatiotemporal(norm_coords)], dim=-1)

    def get_output_dim(self) -> int:
        return self.output_dim

    def encode_relative(self, offset: torch.Tensor) -> torch.Tensor:
        """Translation-equivariant encoding of a metric offset ``(dN, dE, dElev, dTime)`` (requires enable_relative=True).
        Offset normalized by ``relative_window``, clamped to [-0.9, 0.9], encoded by four 3D projections; invariant to
        absolute position. Returns ``(..., relative_output_dim)``."""
        if not self.enable_relative:
            raise RuntimeError("Earth4D was constructed with enable_relative=False")
        lead = offset.shape[:-1]
        norm = self._learnable_freq((offset / self._rel_window).clamp(-1.0, 1.0)).reshape(-1, 4)
        feats = torch.cat([enc(norm[:, list(axes)].contiguous(), size=1.0)
                           for enc, axes in zip(self.rel_encoders, self._rel_projections)], dim=-1)
        return feats.reshape(*lead, -1)

    @staticmethod
    def fit_relative_window(offsets: torch.Tensor, quantile: float = 0.99) -> list:
        """Suggest a per-axis relative window (half-extent) from a high quantile of metric offsets ``(..., 4)``."""
        return torch.quantile(offsets.reshape(-1, 4).abs(), quantile, dim=0).clamp_min(1e-6).tolist()

    def fit_range(self, coords: torch.Tensor, buffer_fraction: float = 0.25) -> 'Earth4D':
        """Fit adaptive ECEF range from training coordinates for improved hash utilization."""
        x, y, z = to_ecef(coords[..., 0], coords[..., 1], coords[..., 2])
        self.adaptive_range = AdaptiveRange.from_ecef_coordinates(
            torch.stack([x, y, z], dim=-1), coords[..., 3], orig_coords=coords, buffer_fraction=buffer_fraction)
        self.use_adaptive_range = True
        if self.verbose:
            self._print_resolution_info()
        return self

    def fit_geo_range(self, coords: torch.Tensor, lat_coverage: float = 0.25, lon_coverage: float = 0.25,
                      elev_coverage: float = 0.15, time_coverage: float = 1.0) -> 'Earth4D':
        """Fit geographic range from training coordinates (N, 4) = [lat, lon, elev, time]."""
        self.geo_range = GeoAdaptiveRange.balanced_regional(coords, lat_coverage, lon_coverage, elev_coverage, time_coverage)
        self._geo_range_is_default = False
        if self.verbose:
            self._print_resolution_info()
        return self

    def precompute(self, coords: torch.Tensor) -> dict:
        """Precompute hash indices/weights for a fixed coordinate set (N, 4). Enables forward_precomputed()."""
        norm_coords = self._normalize_coords(coords)
        t_scaled = (norm_coords[..., 3:] * 2 - 1) * 0.9
        xyzt_scaled = torch.cat([norm_coords[..., :3], t_scaled], dim=-1)
        xyz = norm_coords[..., :3]
        xyt = torch.cat([xyzt_scaled[..., :2], xyzt_scaled[..., 3:]], dim=-1)
        yzt = xyzt_scaled[..., 1:]
        xzt = torch.cat([xyzt_scaled[..., :1], xyzt_scaled[..., 2:]], dim=-1)
        stats = {'xyz': self.xyz_encoder.precompute(xyz, size=1.0),
                 'xyt': self.xyt_encoder.precompute(xyt, size=1.0),
                 'yzt': self.yzt_encoder.precompute(yzt, size=1.0),
                 'xzt': self.xzt_encoder.precompute(xzt, size=1.0)}
        total_bytes = sum(s['total_bytes'] for s in stats.values()); total_mb = total_bytes / (1024 * 1024)
        self._precomputed = True; self._precomp_coords_count = coords.shape[0]
        if self.verbose:
            print(f"\nPrecomputation complete:\n  Coordinates: {coords.shape[0]:,}\n"
                  f"  Total memory: {total_mb:.1f} MB ({total_bytes / coords.shape[0]:.1f} bytes/coord)")
            for name, s in stats.items():
                print(f"    {name}: {s['total_mb']:.1f} MB")
        return {'total_bytes': total_bytes, 'total_mb': total_mb,
                'bytes_per_coord': total_bytes / coords.shape[0], 'num_coords': coords.shape[0], 'encoders': stats}

    def forward_precomputed(self, batch_indices: torch.Tensor) -> torch.Tensor:
        """Forward using precomputed indices/weights (call precompute() first)."""
        if not hasattr(self, '_precomputed') or not self._precomputed:
            raise RuntimeError("Must call precompute() before forward_precomputed()")
        spatial_features = self.xyz_encoder.forward_precomputed(batch_indices)
        spatiotemporal_features = torch.cat([self.xyt_encoder.forward_precomputed(batch_indices),
                                             self.yzt_encoder.forward_precomputed(batch_indices),
                                             self.xzt_encoder.forward_precomputed(batch_indices)], dim=-1)
        return torch.cat([spatial_features, spatiotemporal_features], dim=-1)

    def clear_precomputed(self):
        """Clear precomputed buffers to free memory."""
        self._precomputed = False; self._precomp_coords_count = 0
        self.xyz_encoder.clear_precomputed(); self.xyt_encoder.clear_precomputed()
        self.yzt_encoder.clear_precomputed(); self.xzt_encoder.clear_precomputed()


def print_resolution_info(encoder, config: Dict[str, Any], adaptive_range: Optional[Any] = None):
    """Print resolution scale table and parameter footprint for an Earth4D encoder."""
    is_geographic = config.get('coordinate_system', 'ecef') == 'geographic'
    results = (_calculate_resolution_scales_geographic(encoder, adaptive_range)
               if is_geographic and adaptive_range is not None else _calculate_resolution_scales(encoder))

    print("\n" + "=" * 80 + "\nEARTH4D INITIALIZATION REPORT\n" + "=" * 80)
    print("\n┌─ COORDINATE SYSTEM ─────────────────────────────────────────────────────┐")
    if is_geographic:
        print("│  Mode: GEOGRAPHIC (lat/lon/elev) - preserves latitude relationships     │\n"
              "│    x = Latitude   (-90° to +90°)                                        │\n"
              "│    y = Longitude  (-180° to +180°)                                      │\n"
              "│    z = Elevation  (meters above MSL)                                    │")
        if adaptive_range is not None:
            print(f"│  Range: lat [{adaptive_range.lat_min:.1f}°, {adaptive_range.lat_max:.1f}°], " +
                  f"lon [{adaptive_range.lon_min:.1f}°, {adaptive_range.lon_max:.1f}°]    │")
    else:
        print("│  Mode: ECEF (Earth-Centered Earth-Fixed) - WGS84 ellipsoid             │\n"
              "│    x, y, z = Cartesian coordinates centered at Earth's center          │")
    print("└─────────────────────────────────────────────────────────────────────────┘")

    print("\n┌─ ENHANCEMENT CONFIGURATION ─────────────────────────────────────────────┐")
    if is_geographic:
        range_str = f"GEOGRAPHIC ({adaptive_range.mode})" if adaptive_range else "GEOGRAPHIC (global)"
        print(f"│  Coordinate Range:   {range_str:24}              │")
    else:
        print(f"│  Adaptive Range:     {'ENABLED' if config.get('use_adaptive_range') else 'disabled':12}                                  │")
    lp_str = f"ENABLED (N_p={config.get('probing_range', 0)})" if config.get('enable_learned_probing') else 'disabled'
    print(f"│  Learned Probing:    {lp_str:24}              │")
    print("└─────────────────────────────────────────────────────────────────────────┘")

    print("\n" + "-" * 80 + "\nRESOLUTION SCALE TABLE\n" + "-" * 80)
    if is_geographic:
        print("\nSPATIAL ENCODER (LAT, LON, ELEV):\n"
              f"{'Level':<6} {'Lat Grid':<10} {'Lon Grid':<10} {'Elev Grid':<10} {'Lat/Cell':<12} {'Lon/Cell':<12} {'Elev/Cell':<12}\n"
              + "-" * 82)
        for item in results['spatial']:
            def fmt(m):
                return f"{m/1000:.2f} km" if m >= 1000 else (f"{m:.1f} m" if m >= 1 else f"{m:.3f} m")
            print(f"{item['level']:<6} {item.get('lat_grid', 0):<10} {item.get('lon_grid', 0):<10} "
                  f"{item.get('elev_grid', 0):<10} {fmt(item.get('lat_m', 0)):<12} {fmt(item.get('lon_m', 0)):<12} "
                  f"{fmt(item.get('elev_m', 0)):<12}")
    else:
        print("\nSPATIAL ENCODER (XYZ):\n"
              f"{'Level':<6} {'Grid Res':<12} {'Meters/Cell':<15} {'KM/Cell':<12}\n" + "-" * 70)
        for item in results['spatial']:
            meters = item['meters_per_cell']
            meters_str = f"{meters/1000:.1f}km" if meters >= 1000 else (f"{meters:.2f}m" if meters >= 1 else f"{meters:.3f}m")
            km_str = f"{item['km_per_cell']:.3f}" if item['km_per_cell'] < 1 else f"{item['km_per_cell']:.2f}"
            print(f"{item['level']:<6} {item['grid_resolution']:<12} {meters_str:<15} {km_str:<12}")

    if is_geographic:
        print("\nSPATIOTEMPORAL ENCODERS:\n  xyt = (lat, lon, time)    - Surface dynamics over time\n"
              "  yzt = (lon, elev, time)   - Continental altitude-time patterns\n"
              "  xzt = (lat, elev, time)   - KEY: Same latitudes share cells!")

    print("\nTEMPORAL RESOLUTION:")
    time_scale = results.get('time_output_scale', 1.0)
    if time_scale != 1.0:
        print(f"  (time_output_scale={time_scale:.2f} - effective range compressed to {time_scale*100:.0f}%)")
    print(f"{'Level':<6} {'Grid Res':<12} {'Time/Cell':<15} {'Days/Cell':<12}\n" + "-" * 70)
    temporal_data = results['temporal'] if isinstance(results['temporal'], list) else results['temporal'].get('xyt', [])
    for item in temporal_data:
        secs, days = item['seconds_per_cell'], item['days_per_cell']
        if secs >= 86400 * 30:
            time_str = f"{days:.1f} days"
        elif secs >= 86400:
            time_str = f"{days:.2f} days"
        elif secs >= 3600:
            time_str = f"{secs/3600:.1f} hrs"
        elif secs >= 60:
            time_str = f"{secs/60:.1f} min"
        else:
            time_str = f"{secs:.1f} sec"
        print(f"{item['level']:<6} {item['grid_resolution']:<12} {time_str:<15} {days:<12.2f}")

    spatial_params = encoder.xyz_encoder.embeddings.numel()
    temporal_params = sum(getattr(encoder, f'{n}_encoder').embeddings.numel() for n in ['xyt', 'yzt', 'xzt'])
    total_params = spatial_params + temporal_params; total_memory = total_params * 4 / (1024 * 1024)
    spatial_hash_entries = 2 ** encoder.spatial_log2_hashmap_size
    temporal_hash_entries = 2 ** encoder.temporal_log2_hashmap_size
    print(f"\nHASH TABLE CONFIGURATION:\n"
          f"  Spatial: 2^{encoder.spatial_log2_hashmap_size} = {spatial_hash_entries:,} entries\n"
          f"  Spatiotemporal: 2^{encoder.temporal_log2_hashmap_size} = {temporal_hash_entries:,} entries\n"
          f"  Total capacity: {spatial_hash_entries + temporal_hash_entries*3:,} entries")
    print(f"\nACTUAL PARAMETERS (MEMORY FOOTPRINT):\n"
          f"  Spatial encoders: {spatial_params:,} params = {spatial_params * 4 / (1024*1024):.2f} MB\n"
          f"  Spatiotemporal encoders: {temporal_params:,} params = {temporal_params * 4 / (1024*1024):.2f} MB\n"
          f"  Total: {total_params:,} params = {total_memory:.2f} MB\n"
          f"  During training (4x): ~{total_memory * 4:.2f} MB")


def _calculate_resolution_scales(encoder) -> Dict:
    """Resolution scales for all encoders (ECEF mode)."""
    physical_range = 2 * 6371000.0
    results = {'spatial': [], 'temporal': {'xyt': [], 'yzt': [], 'xzt': []}}
    se = encoder.xyz_encoder
    for level in range(se.num_levels):
        grid = np.ceil(se.base_resolution[0].item() * (se.per_level_scale[0].item() ** level)); mpc = physical_range / grid
        results['spatial'].append({'level': level, 'grid_resolution': int(grid),
                                   'meters_per_cell': mpc, 'km_per_cell': mpc / 1000})
    seconds_per_year = 365.25 * 24 * 3600
    for name, enc in [('xyt', encoder.xyt_encoder), ('yzt', encoder.yzt_encoder), ('xzt', encoder.xzt_encoder)]:
        for level in range(enc.num_levels):
            grid = np.ceil(enc.base_resolution[0].item() * (enc.per_level_scale[0].item() ** level)); spc = seconds_per_year / grid
            results['temporal'][name].append({'level': level, 'grid_resolution': int(grid),
                                              'seconds_per_cell': spc, 'days_per_cell': spc / 86400})
    return results


def _calculate_resolution_scales_geographic(encoder, geo_range) -> Dict:
    """Resolution scales for geographic mode: per-axis [lat, lon, elev] spatial, time from xzt encoder."""
    results = {'spatial': [], 'temporal': []}
    mid_lat = (geo_range.lat_max + geo_range.lat_min) / 2.0
    lat_range_m = (geo_range.lat_max - geo_range.lat_min) * 111000.0
    lon_range_m = (geo_range.lon_max - geo_range.lon_min) * 111000.0 * np.cos(np.radians(mid_lat))
    elev_range_m = geo_range.elev_max - geo_range.elev_min
    se = encoder.xyz_encoder
    for level in range(se.num_levels):
        lat_grid = np.ceil(se.base_resolution[0].item() * (se.per_level_scale[0].item() ** level))
        lon_grid = np.ceil(se.base_resolution[1].item() * (se.per_level_scale[1].item() ** level))
        elev_grid = np.ceil(se.base_resolution[2].item() * (se.per_level_scale[2].item() ** level))
        results['spatial'].append({'level': level, 'lat_grid': int(lat_grid), 'lon_grid': int(lon_grid),
                                   'elev_grid': int(elev_grid), 'lat_m': lat_range_m / lat_grid,
                                   'lon_m': lon_range_m / lon_grid, 'elev_m': elev_range_m / elev_grid})
    seconds_per_year = 365.25 * 24 * 3600
    time_scale_factor = getattr(geo_range, 'time_output_scale', 1.0)
    enc = encoder.xzt_encoder
    for level in range(enc.num_levels):
        time_grid = np.ceil(enc.base_resolution[2].item() * (enc.per_level_scale[2].item() ** level))
        spc = seconds_per_year / (time_grid * time_scale_factor)
        results['temporal'].append({'level': level, 'grid_resolution': int(time_grid),
                                    'seconds_per_cell': spc, 'days_per_cell': spc / 86400})
    results['time_output_scale'] = time_scale_factor
    return results


if __name__ == "__main__":
    # https://github.com/legel/deepearth
    from deepearth.encoders.spacetime.earth4d import Earth4D
    world_model = Earth4D()
    embeddings = world_model(
        (51.9976, -0.7416, 110, "1941-06-01 09:00 GMT"),      # Bletchley Park (Turing breaks Enigma, 1941)
        (40.4433, -79.9436, 270, "1985-01-15 10:00 ET"),      # Carnegie Mellon (Hinton, Boltzmann Machines, 1985)
        (46.2330, 6.0557, 430, "1989-03-12 10:00 CET"),       # CERN (Berners-Lee invents WWW, 1989)
        (45.5308, -73.6128, 63, "2026-02-04 11:00 ET"),       # Mila, Quebec (World Modeling Workshop 2026)
    )
    print(f"embeddings.shape: {list(embeddings.shape)} -- trainable space-time features")
