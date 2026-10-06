"""Climate over the whole land domain: a shoreline location takes the climate of the nearest place that has one.

The gap. WorldClim 2.1 at 30 arc-seconds (about 1 km) has no value in a pixel whose centre falls in the sea or a
large lake, and the 240 m CONUS stack warped from it inherits the gaps. Beaches, dunes, salt marshes and mangroves
therefore have no climate in the raw layers: a model reading them cannot score such a place, and a map built from
them leaves it blank, although it is land and exactly where shoreline species live (in the VegBank benchmark, 48% of
Uniola paniculata's and 53% of Croton punctatus' presence plots; 0.76% of all VegBank plots and 169,719 land cells of
the CONUS grid, about 9,800 km²).

The rule. The climate a few hundred metres inland is the shoreline's climate. A location without climate takes
every climate band of the nearest pixel that has climate, if that pixel lies within ``max_km`` (5 km); farther
locations stay missing. Soil and fine terrain keep their own missing-value flags.

* Points (training records and evaluation plots; ``fill_points``): nearest by great-circle distance between the
  point and the centres of the 30" pixels of the global stack.
* The 240 m map grid (``grid_fill_index``, ``build_grid_fill``, ``GridFill``): nearest cell by exact Euclidean
  distance on the equal-area grid, for the land cells of the mapped domain only.

``fill_points`` processes points in batches (each reads the same-sized pixel window around every point of the batch,
wide enough in longitude for its most poleward point); ``fill_points_reference`` is its one-point-at-a-time
definition, and the two agree exactly (tests/test_climate_fill.py).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch

CELL = 1.0 / 120.0                                   # 30 arc-seconds
N_CLIMATE = 20                                       # WorldClim bio1-19 and elevation, the stack's first bands
EARTH_KM = 6371.0088                                 # mean Earth radius (IUGG)


def great_circle_km(lat1, lon1, lat2, lon2):
    """Haversine distance (km) between points given in degrees (float64 tensors, broadcast)."""
    p1, p2 = torch.deg2rad(lat1), torch.deg2rad(lat2)
    a = torch.sin((p2 - p1) / 2) ** 2 + torch.cos(p1) * torch.cos(p2) * torch.sin(torch.deg2rad(lon2 - lon1) / 2) ** 2
    return 2 * EARTH_KM * torch.asin(a.clamp(0, 1).sqrt())


def _half_widths(lat_abs_max: float, cell: float, max_km: float, ncol: int) -> tuple[int, int]:
    """Window half-widths in pixels: k north-south covers max_km; east-west widens as 1 / cos(latitude), evaluated at
    the window's poleward edge."""
    k = math.ceil(max_km / (EARTH_KM * math.radians(cell))) + 1
    kc = min(ncol // 2, math.ceil(k / max(math.cos(math.radians(lat_abs_max + k * cell)), 1e-6)))
    return k, kc


def fill_points(X: np.ndarray, lat, lon, data, cell: float = CELL, n_climate: int = N_CLIMATE,
                max_km: float = 5.0, batch: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Fill the climate of points that have none.

    ``X`` [n, >= n_climate]: predictors whose first ``n_climate`` columns are the climate bands in the order of
    ``data``, a global pixel-interleaved array [rows, cols, >= n_climate] whose row 0 starts at 90 N and column 0 at
    180 W, pixels ``cell`` degrees wide (``predictors.GlobalStack(...).data``). A point lacks climate when any of its
    climate columns is missing. Returns a filled copy of ``X`` and the fill distance in km per point (0 where the
    point had climate, NaN where no pixel with climate lies within ``max_km``)."""
    X = np.array(X, np.float32, copy=True)
    dist = np.zeros(len(X), np.float32)
    miss = np.nonzero(~np.isfinite(X[:, :n_climate]).all(1))[0]
    if len(miss) == 0:
        return X, dist
    nrow, ncol = data.shape[:2]
    lat_m, lon_m = np.asarray(lat, np.float64)[miss], np.asarray(lon, np.float64)[miss]
    o = np.argsort(np.abs(lat_m), kind="stable")                    # similar latitudes share a window width
    for i in range(0, len(o), batch):
        j = o[i:i + batch]
        la, lo = torch.from_numpy(lat_m[j]), torch.from_numpy(lon_m[j])
        k, kc = _half_widths(float(la.abs().max()), cell, max_km, ncol)
        r0, c0 = torch.floor((90.0 - la) / cell).long(), torch.floor((lo + 180.0) / cell).long()
        rows = (r0[:, None] + torch.arange(-k, k + 1)).clamp(0, nrow - 1)                       # [B, h]
        cols = torch.remainder(c0[:, None] + torch.arange(-kc, kc + 1), ncol)                    # [B, w]
        blk = torch.from_numpy(np.asarray(data[rows.numpy()[:, :, None], cols.numpy()[:, None, :], :n_climate],
                                          np.float32))                                          # [B, h, w, bands]
        ok = torch.isfinite(blk).all(-1)
        plat = 90.0 - (rows.double() + 0.5) * cell
        plon = -180.0 + (cols.double() + 0.5) * cell
        d = great_circle_km(la[:, None, None], lo[:, None, None], plat[:, :, None], plon[:, None, :])
        d = d.masked_fill(~ok, float("inf")).reshape(len(j), -1)
        dmin, arg = d.min(1)
        hit = dmin <= max_km
        vals = blk.reshape(len(j), -1, n_climate)[torch.arange(len(j)), arg]
        X[miss[j[hit.numpy()]], :n_climate] = vals[hit].numpy()
        dist[miss[j]] = np.where(hit.numpy(), dmin.float().numpy(), np.nan)
    return X, dist


def fill_points_reference(X: np.ndarray, lat, lon, data, cell: float = CELL, n_climate: int = N_CLIMATE,
                          max_km: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """``fill_points``, one point at a time over its own window (the definition the batched version is checked
    against). Ties between equally distant pixels go to the first in row-major window order, as in the batch."""
    X = np.array(X, np.float32, copy=True)
    dist = np.zeros(len(X), np.float32)
    miss = np.nonzero(~np.isfinite(X[:, :n_climate]).all(1))[0]
    nrow, ncol = data.shape[:2]
    for m in miss:
        la, lo = float(np.float64(lat[m])), float(np.float64(lon[m]))
        k, kc = _half_widths(abs(la), cell, max_km, ncol)
        r0, c0 = math.floor((90.0 - la) / cell), math.floor((lo + 180.0) / cell)
        rows = np.clip(np.arange(r0 - k, r0 + k + 1), 0, nrow - 1)
        cols = np.mod(np.arange(c0 - kc, c0 + kc + 1), ncol)
        blk = torch.from_numpy(np.asarray(data[rows[:, None], cols[None, :], :n_climate], np.float32))
        ok = torch.isfinite(blk).all(-1)
        plat = torch.from_numpy(90.0 - (rows + 0.5) * cell)[:, None]
        plon = torch.from_numpy(-180.0 + (cols + 0.5) * cell)[None, :]
        d = great_circle_km(torch.tensor(la, dtype=torch.float64), torch.tensor(lo, dtype=torch.float64), plat, plon)
        d = d.masked_fill(~ok, float("inf")).reshape(-1)
        i = int(torch.argmin(d))
        if not float(d[i]) <= max_km:
            dist[m] = np.nan
            continue
        X[m, :n_climate] = blk.reshape(-1, n_climate)[i].numpy()
        dist[m] = float(d[i])
    return X, dist


# ------------------------------------------------------------------------------------------------- map grid
def grid_fill_index(valid: np.ndarray, land: np.ndarray, cell_km: float = 0.24,
                    max_km: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Fill of an equal-area grid. ``valid`` [H, W]: the cell has climate; ``land`` [H, W]: the cell belongs to the
    mapped land domain. Returns flat row-major indices (dst, src): every land cell without climate whose nearest cell
    with climate (exact Euclidean distance in cells, by scipy's distance transform) lies within ``max_km``, and that
    cell."""
    from scipy.ndimage import distance_transform_edt
    d, (ri, ci) = distance_transform_edt(~valid, return_indices=True)
    dst = np.nonzero((land & ~valid & (d * cell_km <= max_km)).ravel())[0]
    src = ri.ravel()[dst].astype(np.int64) * valid.shape[1] + ci.ravel()[dst]
    return dst, src


WATER_CLASSES = (18, 19)                     # NALCMS 2020: water, snow and ice


def water_fraction(nalcms_30m: str | Path, shape: tuple[int, int], rows: int = 64) -> np.ndarray:
    """Share of water (and snow/ice) among the classified NALCMS 2020 pixels of each 240 m cell, float32 [H, W].

    ``nalcms_30m``: NALCMS warped (nearest neighbour) to the 30 m EPSG:5070 grid whose origin is the 240 m grid's,
    so each 240 m cell is exactly an 8 x 8 block of pixels. Classes 1-19 are classified; 0 (outside the mapped
    extent, including open sea) and 127 (no data) are not. NaN where a cell has no classified pixel."""
    import rasterio
    H, W = shape
    out = np.full((H, W), np.nan, np.float32)
    with rasterio.open(nalcms_30m) as src:
        if (src.height, src.width) != (H * 8, W * 8):
            raise ValueError(f"{nalcms_30m}: {src.height} x {src.width} pixels, expected {H * 8} x {W * 8}")
        for r0 in range(0, H, rows):
            r1 = min(H, r0 + rows)
            px = src.read(1, window=((r0 * 8, r1 * 8), (0, W * 8)))
            blk = px.reshape(r1 - r0, 8, W, 8).transpose(0, 2, 1, 3).reshape(r1 - r0, W, 64)
            classified = (blk >= 1) & (blk <= 19)
            n = classified.sum(-1)
            water = np.isin(blk, WATER_CLASSES).sum(-1)
            out[r0:r1] = np.where(n > 0, water / np.maximum(n, 1), np.nan)
    return out


def build_grid_fill(stack_dir: str | Path, land: np.ndarray, out: str | Path, cell_km: float = 0.24,
                    max_km: float = 5.0, log: Callable = print) -> dict:
    """Write the fill index of a 240 m region grid (``out``, e.g. <grid>/climate_fill_conus240.npz: dst, src, km,
    shape). A cell has climate when all 20 WorldClim bands of the grid's stack are finite; ``land`` [H, W] marks the
    cells to fill when they lack it (e.g. inside the country and under half water by NALCMS: an administrative mask
    follows state boundaries ~3 nautical miles offshore, so without the water test it would "fill" coastal sea)."""
    from ..predictors import VARIABLES, ConusStack
    st = ConusStack(stack_dir)
    H, W = st.shape
    valid = np.ones((H, W), bool)
    for k in st.band_index(VARIABLES):
        valid &= np.isfinite(np.asarray(st.data[k]))
    dst, src = grid_fill_index(valid, np.asarray(land, bool), cell_km, max_km)
    km = np.hypot(*np.subtract(np.unravel_index(dst, (H, W)), np.unravel_index(src, (H, W)))) * cell_km
    np.savez(out, dst=dst, src=src, km=km.astype(np.float32), shape=np.array([H, W]))
    info = {"land_cells_without_climate": int((np.asarray(land, bool) & ~valid).sum()), "filled": int(len(dst)),
            "median_km": float(np.median(km)) if len(km) else 0.0}
    log(f"{info['land_cells_without_climate']:,} land cells without climate; {len(dst):,} filled within {max_km} km "
        f"(median {info['median_km']:.2f} km) -> {out}")
    return info


class GridFill:
    """The fill index of a region grid (``build_grid_fill``): which cells take their climate from which cell. Every
    filled cell carries all 20 WorldClim bands of its source cell."""

    def __init__(self, dst: np.ndarray, src: np.ndarray, shape: tuple[int, int]):
        order = np.argsort(dst, kind="stable")
        self.dst, self.src = np.asarray(dst, np.int64)[order], np.asarray(src, np.int64)[order]
        self.shape = (int(shape[0]), int(shape[1]))

    @classmethod
    def load(cls, path: str | Path) -> "GridFill":
        z = np.load(path)
        return cls(z["dst"], z["src"], tuple(z["shape"]))

    def window(self, r0: int, r1: int, c0: int, c1: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filled cells of the window [r0, r1) x [c0, c1): (their positions in the window's row-major order, source
        rows, source columns)."""
        W = self.shape[1]
        lo, hi = np.searchsorted(self.dst, [r0 * W, (r1 - 1) * W + c1])
        d, s = self.dst[lo:hi], self.src[lo:hi]
        dr, dc = d // W, d % W
        inw = (dc >= c0) & (dc < c1)
        d, s, dr, dc = d[inw], s[inw], dr[inw], dc[inw]
        return (dr - r0) * (c1 - c0) + (dc - c0), s // W, s % W


# --------------------------------------------------------------------------------------------- plot sets
def fill_plot_sets(data_dir: str | Path, data, cell: float = CELL, max_km: float = 5.0,
                   log: Callable = print) -> dict:
    """Fill the predictors of every plot set of a joint data directory (VegBank in joint_data.npz, the others in
    plots_<source>.npz) and write <data_dir>/plot_fill_<source>.npz: ``X`` (filled predictors), ``dist`` (km, 0
    where the plot had climate, NaN beyond ``max_km``), ``n`` (number of plots, to detect a stale file)."""
    data_dir = Path(data_dir)
    sets = [("vegbank", data_dir / "joint_data.npz")] + [(f.stem.split("_", 1)[1], f)
                                                         for f in sorted(data_dir.glob("plots_*.npz"))]
    out = {}
    for name, f in sets:
        z = np.load(f)
        X, dist = fill_points(np.asarray(z["plot_X"]), z["plot_lat"], z["plot_lon"], data, cell, max_km=max_km)
        np.savez(data_dir / f"plot_fill_{name}.npz", X=X, dist=dist, n=len(X))
        filled = dist > 0
        out[name] = {"plots": len(X), "filled": int(np.isfinite(dist[filled]).sum()),
                     "beyond": int(np.isnan(dist).sum())}
        log(f"{name}: {out[name]['filled']} plots without climate filled (median "
            f"{float(np.nanmedian(dist[filled])) if filled.any() and np.isfinite(dist[filled]).any() else 0:.2f} km), "
            f"{out[name]['beyond']} beyond {max_km} km")
    (data_dir / "plot_fill.json").write_text(json.dumps({"max_km": max_km, "sets": out}, indent=1))
    return out
