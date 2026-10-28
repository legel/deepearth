"""Chip geometry: a 300 m x 300 m window centred on a tower, on the native pixel grid of each band.

Sentinel-2 tiles are 109.8 km squares in the UTM zone of the MGRS tile, with upper-left corners on a 60 m grid, so the
10, 20 and 60 m grids of one tile share an origin. A chip of n pixels at resolution r starts at the column

    c0 = round((x - x0) / r - n / 2)

(and the row likewise from y0 - y), which puts the chip centre within r / 2 of the tower on each axis: 5 m for the
10 m bands, 10 m for the 20 m bands and 30 m for the 60 m bands. Each resolution group keeps its own window origin, and
every chip records the tower offset from the 10 m chip centre.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from pyproj import Transformer


@lru_cache(maxsize=256)
def _to_utm(epsg: int) -> Transformer:
    return Transformer.from_crs(4326, int(epsg), always_xy=True)


@lru_cache(maxsize=256)
def _to_lonlat(epsg: int) -> Transformer:
    return Transformer.from_crs(int(epsg), 4326, always_xy=True)


def lonlat_to_utm(lon: float, lat: float, epsg: int) -> tuple:
    return _to_utm(epsg).transform(lon, lat)


def utm_to_lonlat(x: float, y: float, epsg: int) -> tuple:
    return _to_lonlat(epsg).transform(x, y)


@dataclass(frozen=True)
class Window:
    """A pixel window in a raster with north-up transform (x0, res, 0, y0, 0, -res)."""
    col: int
    row: int
    size: int
    res: float
    x0: float          # raster upper-left x
    y0: float          # raster upper-left y

    @property
    def ul(self) -> tuple:
        """Map coordinates of the window's upper-left corner."""
        return self.x0 + self.col * self.res, self.y0 - self.row * self.res

    @property
    def centre(self) -> tuple:
        x, y = self.ul
        half = self.size * self.res / 2.0
        return x + half, y - half


def chip_window(x: float, y: float, x0: float, y0: float, res: float, size: int) -> Window:
    """The size x size window on the raster grid whose centre is nearest to (x, y)."""
    col = int(np.floor((x - x0) / res - size / 2.0 + 0.5))
    row = int(np.floor((y0 - y) / res - size / 2.0 + 0.5))
    return Window(col, row, size, float(res), float(x0), float(y0))


def read_window(read, win: Window, width: int, height: int, dtype, fill=0) -> np.ndarray:
    """Read `win` from a raster of width x height pixels, padding with `fill` outside the raster.

    `read(col_off, row_off, w, h)` returns the in-bounds block as a 2-D array (rasterio `DatasetReader.read(1,
    window=...)`). Reading only the in-bounds part avoids GDAL's boundless VRT path.
    """
    out = np.full((win.size, win.size), fill, dtype=dtype)
    c0, r0 = max(win.col, 0), max(win.row, 0)
    c1, r1 = min(win.col + win.size, width), min(win.row + win.size, height)
    if c1 <= c0 or r1 <= r0:
        return out
    block = read(c0, r0, c1 - c0, r1 - r0)
    out[r0 - win.row:r1 - win.row, c0 - win.col:c1 - win.col] = block
    return out


def upsample_to(src: np.ndarray, src_win: Window, dst_win: Window) -> np.ndarray:
    """Nearest-neighbour resampling of a coarser chip (..., n, n) onto the pixel centres of a finer chip.

    Both windows come from the same tile grid, so each fine pixel centre falls inside exactly one coarse pixel; fine
    pixels whose centre falls outside the coarse window take the nearest edge pixel.
    """
    ux, uy = dst_win.ul
    centres = (np.arange(dst_win.size) + 0.5) * dst_win.res
    sx, sy = src_win.ul
    cols = np.clip(np.floor((ux + centres - sx) / src_win.res).astype(int), 0, src_win.size - 1)
    rows = np.clip(np.floor((sy - (uy - centres)) / src_win.res).astype(int), 0, src_win.size - 1)
    return src[..., rows[:, None], cols[None, :]]
