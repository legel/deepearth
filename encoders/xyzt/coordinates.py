"""
Geographic coordinate utilities for Earth4D.
"""

import torch
from typing import Dict, Tuple
from dataclasses import dataclass, asdict

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
