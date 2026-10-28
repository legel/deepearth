"""Grid-cell selection at a tower: the nearest cell and the bilinear value, both recorded.

A regular latitude-longitude grid is defined by the centre of its first cell (lat0, lon0), its spacing and its size.
Cell (i, j) has its centre at (lat0 + i * dlat, lon0 + j * dlon). Longitudes wrap.

The bilinear value uses the four cell centres around the tower. A corner without a value (sea in a land-only
product such as ERA5-Land) drops out and the remaining weights are renormalised. The nearest cell is the corner
closest to the tower that has a value.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class RegularGrid:
    lat0: float
    lon0: float
    dlat: float
    dlon: float
    ny: int
    nx: int

    def nearest(self, lat: float, lon: float) -> Tuple[int, int]:
        i = int(np.clip(np.rint((lat - self.lat0) / self.dlat), 0, self.ny - 1))
        j = int(np.rint((lon - self.lon0) / self.dlon)) % self.nx
        return i, j

    def centre(self, i: int, j: int) -> Tuple[float, float]:
        lon = self.lon0 + j * self.dlon
        lon = (lon + 180.0) % 360.0 - 180.0
        return self.lat0 + i * self.dlat, lon

    def corners(self, lat: float, lon: float):
        """(i0, j0, j1, wy, wx): the 2 x 2 block [i0, i0 + 1] x [j0, j1] and the fractional position in it.
        j1 is j0 + 1 except across the date line, where it wraps to 0."""
        fi = (lat - self.lat0) / self.dlat
        i0 = int(np.clip(np.floor(fi), 0, self.ny - 2))
        wy = float(np.clip(fi - i0, 0.0, 1.0))
        fj = (lon - self.lon0) / self.dlon
        j0f = int(np.floor(fj))
        wx = float(fj - j0f)
        j0 = j0f % self.nx
        j1 = (j0 + 1) % self.nx
        return i0, j0, j1, wy, wx


def corner_weights(wy: float, wx: float) -> np.ndarray:
    """Bilinear weights of the corners ordered [(i0, j0), (i0, j1), (i0 + 1, j0), (i0 + 1, j1)]."""
    return np.array([(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx])


def corner_offsets() -> np.ndarray:
    """(di, dj) of the four corners, in the order of `corner_weights`."""
    return np.array([[0, 0], [0, 1], [1, 0], [1, 1]])


def combine(block: np.ndarray, wy: float, wx: float, valid_corners: np.ndarray = None):
    """Nearest and bilinear series from a (4, nt) corner block.

    valid_corners: which corners are valid cells (land-sea mask). Default: corners with any finite value.
    Returns (nearest (nt,), bilinear (nt,), index of the nearest corner used).
    """
    w = corner_weights(wy, wx)
    if valid_corners is None:
        valid_corners = np.isfinite(block).any(axis=1)
    valid_corners = np.asarray(valid_corners, bool)
    if not valid_corners.any():
        nt = block.shape[1]
        return np.full(nt, np.nan, np.float32), np.full(nt, np.nan, np.float32), -1
    # Nearest valid corner by distance in grid units.
    pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    d = np.hypot(pos[:, 0] - wy, pos[:, 1] - wx)
    d[~valid_corners] = np.inf
    k = int(np.argmin(d))
    nearest = block[k].astype(np.float32)
    wv = np.where(valid_corners, w, 0.0)
    vals = np.where(np.isfinite(block), block, 0.0)
    wt = np.where(np.isfinite(block), wv[:, None], 0.0)
    s = wt.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        bil = np.where(s > 0, (vals * wt).sum(axis=0) / s, np.nan).astype(np.float32)
    return nearest, bil, k


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))
