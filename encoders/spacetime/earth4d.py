"""Earth4D: planetary (X,Y,Z,T) hash-grid positional encoder."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import warnings
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Literal, Tuple, Dict, Any
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
                 relative_finest: Tuple[float, float, float, float] = (15.0, 15.0, 3.0, 1.0),
                 fourier_features: int = 0,
                 fourier_scale: float = 10.0,
                 time_harmonics: int = 0,
                 time_film: int = 0,
                 spatial_cline: int = 0,
                 cline_scale: float = 1.0,
                 spatial_siren: int = 0,
                 siren_layers: int = 2,
                 siren_w0: float = 30.0,
                 causal_lags: int = 0,
                 causal_lag_span: float = 0.25):
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

        # Relative (translation-equivariant) channel: encode the OFFSET between two observations so a learned
        # pattern transfers across absolute position/time. Window fits the offset distribution; four 3D projections.
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

        # Learnable per-axis frequency (scale + center) selecting each axis's effective resolution range;
        # tanh-bounded to [-0.9, 0.9]. Axes are (N/x, E/y, elev/z, time); initialized to identity.
        self.freq_log_scale = nn.Parameter(torch.full((4,), float(freq_log_scale_init)))
        self.freq_center = nn.Parameter(torch.zeros(4))

        # Optional random-Fourier-features branch (default OFF -> champion path byte-identical). sin/cos of a random
        # linear projection of the normalized 4D coords: smooth GLOBAL features (Rahimi-Recht RFF) that complement
        # the LOCAL multi-resolution hash, which underperforms a generic RFF PE on smooth/static signals. Concatenated
        # onto the hash output; only the dense forward() path carries it (forward_precomputed stays hash-only).
        self.fourier_features = int(fourier_features)
        if self.fourier_features > 0:
            # SPATIAL-only RFF (projects xyz, NOT time) so the temporal projections stay undiluted -> fixes the
            # hash's static/env weakness without hurting the forecast/temporal path.
            self.register_buffer("_fourier_B", torch.randn(3, self.fourier_features) * float(fourier_scale))
            self.output_dim = self.output_dim + 2 * self.fourier_features

        # Optional learnable temporal-harmonic path (default OFF -> champion byte-identical). The (xy·t / yz·t / xz·t)
        # hash MEMORIZES discrete time cells and carries NO smooth/periodic temporal inductive bias; smooth_geo
        # (fusion.py) is SPATIAL-only, so this is NOT redundant with it. A learnable multi-scale sin/cos basis over
        # normalized time gives the encoder an internal seasonal/persistence prior (the forecast objective's actual
        # signal is seasonal persistence). Appended to the dense forward() output; forward_precomputed stays hash-only.
        self.time_harmonics = int(time_harmonics)
        if self.time_harmonics > 0:
            init = torch.tensor([2.0 ** k for k in range(self.time_harmonics)], dtype=torch.float32)
            self.time_freqs = nn.Parameter(init)   # cycles over the normalized time span; learnable timescale
            self.output_dim = self.output_dim + 2 * self.time_harmonics

        # Optional gated SPATIAL-CLINE branch (default OFF -> champion byte-identical). Diagnosis: the multiresolution
        # hash MEMORIZES position and carries no monotone spatial GRADIENT (the lat->flowering-DOY Hopkins cline is
        # measurably absent from pos_s), and --fourier is a FIXED HIGH-frequency (scale 10) random projection, which
        # cannot represent a degree-1 cline either. This branch adds (a) the normalized xyz themselves = the explicit
        # linear/cline term, and (b) a LEARNABLE LOW-frequency sin/cos band (init scale ~1 cycle over the domain, and
        # B is a Parameter so training can shrink it toward a pure gradient). Spatial-only; dense forward() path only.
        self.spatial_cline = int(spatial_cline)
        if self.spatial_cline > 0:
            self._cline_B = nn.Parameter(torch.randn(3, self.spatial_cline) * float(cline_scale))
            self.output_dim = self.output_dim + 3 + 2 * self.spatial_cline

        # Optional gated SIREN spatial branch (default OFF -> champion byte-identical). Diagnosis: on STATIC/spatial
        # held-out-BLOCK tasks the multiresolution hash LOSES to a plain RFF (family_from_env: Earth4D 0.074-0.083 vs
        # RFF 0.113-0.116) because it memorizes position locally and has nothing to say across a block boundary,
        # while RFF's smooth global basis extrapolates. --fourier is a FIXED random projection (one bandwidth, not
        # trained) and an env-reconstruction objective did not fix it either. A SIREN (sinusoidal-activation MLP,
        # Sitzmann et al.) is smooth and extrapolative BY CONSTRUCTION and its frequencies are LEARNED at every
        # layer -- a genuinely different spatial representation rather than another fixed basis. Spatial-only
        # (xyz, not time) so the temporal projections that already win stay untouched. Dense forward() path only.
        self.spatial_siren = int(spatial_siren)
        if self.spatial_siren > 0:
            self.siren_w0 = float(siren_w0)
            dims = [3] + [self.spatial_siren] * int(siren_layers)
            self.siren = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))
            with torch.no_grad():   # SIREN init: first layer U(-1/in, 1/in), rest U(-sqrt(6/in)/w0, +same)
                for i, lin in enumerate(self.siren):
                    b = 1.0 / lin.in_features if i == 0 else math.sqrt(6.0 / lin.in_features) / self.siren_w0
                    lin.weight.uniform_(-b, b); lin.bias.zero_()
            self.output_dim = self.output_dim + self.spatial_siren

        # Optional gated CAUSAL TEMPORAL STATE (default OFF -> champion byte-identical). The encoder is evaluated
        # POINTWISE: the (xy.t / yz.t / xz.t) hash reads the cell AT time t and carries NO history, so the
        # representation at t knows nothing about what happened before t -- every temporal structure the probes
        # exploit has had to be rebuilt OUTSIDE the encoder by a propagator (GNN/LSTM/attention over neighbours).
        # This gives the encoder its own causal memory: the spatiotemporal features are ALSO evaluated at K learned
        # causal LAGS (t - lag_k, strictly backward, clamped at the domain start) and combined by learned softmax
        # weights, so the output at t is a function of the field's own past. Lags are sigmoid-bounded to
        # [0, causal_lag_span] of the normalized time span and are learned, not fixed. Costs K extra hash lookups.
        self.causal_lags = int(causal_lags)
        if self.causal_lags > 0:
            self.causal_lag_span = float(causal_lag_span)
            # init lags spread across the span (inverse-sigmoid of evenly spaced fractions)
            fr = torch.linspace(0.25, 0.9, self.causal_lags)
            self._lag_raw = nn.Parameter(torch.log(fr / (1.0 - fr)))
            self._lag_w = nn.Parameter(torch.zeros(self.causal_lags))    # softmax -> uniform at init
            self.output_dim = self.output_dim + self.spatiotemporal_dim

        # Optional gated space x time FiLM (default OFF -> champion byte-identical). The concatenated spatial-hash +
        # time-harmonic features are ADDITIVE; a linear head cannot form space*time PRODUCTS. This modulates the
        # spatial-hash block by a learned function of a multi-scale time basis -> explicit seasonal-spatial interaction
        # (species range shifts with season). Zero-init W/b -> gate = 1 + tanh(0) = 1 -> identity at start.
        self.time_film = int(time_film)
        if self.time_film > 0:
            self.register_buffer("_film_freqs", torch.tensor([2.0 ** k for k in range(self.time_film)], dtype=torch.float32))
            self._film_W = nn.Parameter(torch.zeros(2 * self.time_film, self.spatial_dim))
            self._film_b = nn.Parameter(torch.zeros(self.spatial_dim))

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

    def _fourier(self, norm_coords: torch.Tensor) -> torch.Tensor:
        """Spatial-only random Fourier features (xyz, not time): [sin(2π B xyz), cos(2π B xyz)]."""
        proj = 2.0 * math.pi * (norm_coords[..., :3] @ self._fourier_B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def _cline(self, norm_coords: torch.Tensor) -> torch.Tensor:
        """Smooth spatial-gradient band: [xyz, sin(2π B xyz), cos(2π B xyz)] with LEARNABLE low-frequency B.
        The linear xyz term is the cline (degree-1 gradient) the hash and the fixed high-freq RFF both destroy."""
        xyz = norm_coords[..., :3]
        proj = 2.0 * math.pi * (xyz @ self._cline_B)
        return torch.cat([xyz, torch.sin(proj), torch.cos(proj)], dim=-1)

    def _causal_state(self, norm_coords: torch.Tensor) -> torch.Tensor:
        """Causal memory: the spatiotemporal field re-read at K learned BACKWARD lags, softmax-combined.
        Strictly causal (t - lag, lag >= 0) and clamped to the domain start, so no future information enters."""
        lags = torch.sigmoid(self._lag_raw) * self.causal_lag_span      # (K,) in [0, span]
        w = torch.softmax(self._lag_w, dim=0)
        out = None
        for k in range(self.causal_lags):
            shifted = norm_coords.clone()
            shifted[..., 3] = torch.clamp(shifted[..., 3] - lags[k], min=0.0)
            fk = self._encode_spatiotemporal(shifted) * w[k]
            out = fk if out is None else out + fk
        return out

    def _siren(self, norm_coords: torch.Tensor) -> torch.Tensor:
        """Sinusoidal-activation MLP over normalized xyz: sin(w0 * W_k ... sin(w0 * W_1 xyz)).
        Smooth and globally extrapolative by construction, with LEARNED per-layer frequencies."""
        h = norm_coords[..., :3]
        for lin in self.siren:
            h = torch.sin(self.siren_w0 * lin(h))
        return h

    def _film_harmonics(self, norm_coords: torch.Tensor) -> torch.Tensor:
        t = norm_coords[..., 3:]
        proj = 2.0 * math.pi * t * self._film_freqs
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def _time_harmonics(self, norm_coords: torch.Tensor) -> torch.Tensor:
        """Learnable multi-scale periodic time basis [sin(2π f t), cos(2π f t)] over normalized time -> a smooth
        seasonal/persistence prior the discrete hash lacks (temporal analogue of the spatial RFF; not smooth_geo)."""
        t = norm_coords[..., 3:]                          # (..., 1) normalized time in [0,1]
        proj = 2.0 * math.pi * t * self.time_freqs        # (..., K) broadcast over learnable frequencies
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def _forward_tensor(self, coords: torch.Tensor) -> torch.Tensor:
        norm_coords = self._normalize_coords(coords)
        spatial = self._encode_spatial(norm_coords[..., :3])
        if self.time_film > 0:
            gate = 1.0 + torch.tanh(self._film_harmonics(norm_coords) @ self._film_W + self._film_b)
            spatial = spatial * gate                             # space x time interaction (FiLM)
        feats = torch.cat([spatial, self._encode_spatiotemporal(norm_coords)], dim=-1)
        if self.fourier_features > 0:
            feats = torch.cat([feats, self._fourier(norm_coords)], dim=-1)
        if self.time_harmonics > 0:
            feats = torch.cat([feats, self._time_harmonics(norm_coords)], dim=-1)
        if self.spatial_cline > 0:
            feats = torch.cat([feats, self._cline(norm_coords)], dim=-1)
        if self.spatial_siren > 0:
            feats = torch.cat([feats, self._siren(norm_coords)], dim=-1)
        if self.causal_lags > 0:
            feats = torch.cat([feats, self._causal_state(norm_coords)], dim=-1)
        return feats

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
        return self

    def fit_geo_range(self, coords: torch.Tensor, lat_coverage: float = 0.25, lon_coverage: float = 0.25,
                      elev_coverage: float = 0.15, time_coverage: float = 1.0) -> 'Earth4D':
        """Fit geographic range from training coordinates (N, 4) = [lat, lon, elev, time]."""
        self.geo_range = GeoAdaptiveRange.balanced_regional(coords, lat_coverage, lon_coverage, elev_coverage, time_coverage)
        self._geo_range_is_default = False
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

    def forward_precomputed(self, batch_indices: torch.Tensor, return_dydx: bool = False):
        """Forward using precomputed indices/weights (call precompute() first).

        When ``return_dydx`` is True, also return per-sub-encoder ``dy_dx`` and the normalized ``inputs`` each used
        (order xyz, xyt, yzt, xzt), so the sparse path can form each sub-encoder's per_level_scale gradient without
        materializing dense embedding grads. Returns ``(features, [dy_dx...], [inputs...])`` in that case."""
        if not hasattr(self, '_precomputed') or not self._precomputed:
            raise RuntimeError("Must call precompute() before forward_precomputed()")
        encs = [self.xyz_encoder, self.xyt_encoder, self.yzt_encoder, self.xzt_encoder]
        if not return_dydx:
            spatial_features = self.xyz_encoder.forward_precomputed(batch_indices)
            spatiotemporal_features = torch.cat([self.xyt_encoder.forward_precomputed(batch_indices),
                                                 self.yzt_encoder.forward_precomputed(batch_indices),
                                                 self.xzt_encoder.forward_precomputed(batch_indices)], dim=-1)
            return torch.cat([spatial_features, spatiotemporal_features], dim=-1)
        outs, dydx, inputs = [], [], []
        for en in encs:
            o, dd, inp = en.forward_precomputed(batch_indices, return_dydx=True)
            outs.append(o); dydx.append(dd); inputs.append(inp)
        return torch.cat(outs, dim=-1), dydx, inputs

    def clear_precomputed(self):
        """Clear precomputed buffers to free memory."""
        self._precomputed = False; self._precomp_coords_count = 0
        self.xyz_encoder.clear_precomputed(); self.xyt_encoder.clear_precomputed()
        self.yzt_encoder.clear_precomputed(); self.xzt_encoder.clear_precomputed()


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
