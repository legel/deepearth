"""Sampling-bias grid and background points (Daru 2024 step 5).

Daru estimated vascular-plant sampling effort with ``spatialEco::sp.kde``, standardized to 0–1, and drew
10,000 background points per species with probability proportional to it inside the calibration area
(``phyloregion::backg``; if the area has <= 10,000 cells, every cell is used).

``sp.kde`` evaluates a Gaussian product kernel on a projected grid. Its per-axis bandwidth is
``4 * 1.06 * min(sd, IQR / 1.34) * n^(-1/5)``, divided by 4 inside the kernel, i.e. the Gaussian standard
deviation is Silverman's rule ``1.06 * min(sd, IQR / 1.34) * n^(-1/5)``. Here the records arrive aggregated to
~9 km cells (a GBIF SQL download), so the kernel sum is computed as an FFT convolution of the count grid with the
same Gaussian, and the bandwidth statistics are computed over all records via count-weighted moments and
quantiles. Behrmann equal-area projection (ESRI:54017), the projection of Daru's world map.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.signal import fftconvolve

BEHRMANN = "ESRI:54017"


def _weighted_quantile(v: np.ndarray, w: np.ndarray, q: Sequence[float]) -> np.ndarray:
    o = np.argsort(v)
    cw = np.cumsum(w[o]) / w.sum()
    return np.interp(q, cw, v[o])


def silverman_bandwidth(v: np.ndarray, w: np.ndarray) -> float:
    """sp.kde's bandwidth (after its /4): 1.06 * min(sd, IQR/1.34) * n^(-1/5), n = total records."""
    n = w.sum()
    mean = np.average(v, weights=w)
    sd = np.sqrt(np.average((v - mean) ** 2, weights=w) * n / (n - 1))
    q25, q75 = _weighted_quantile(v, w, [0.25, 0.75])
    return 1.06 * min(sd, (q75 - q25) / 1.34) * n ** (-0.2)


def bias_grid(counts: pd.DataFrame, cell_km: float = 10.0) -> dict:
    """Standardized (0–1) sampling-density grid in Behrmann from per-~9 km-cell record counts.

    ``counts`` has columns xi, yi (floor(lon*12), floor(lat*12)) and n (records).
    """
    lon = (counts.xi.values + 0.5) / 12.0
    lat = (counts.yi.values + 0.5) / 12.0
    x, y = Transformer.from_crs(4326, BEHRMANN, always_xy=True).transform(lon, lat)
    w = counts.n.values.astype(np.float64)
    hx, hy = silverman_bandwidth(x, w), silverman_bandwidth(y, w)
    res = cell_km * 1000.0
    x0, x1 = -17_367_530.45, 17_367_530.45            # Behrmann world extent
    y0, y1 = -7_342_230.14, 7_342_230.14
    nx, ny = int(np.ceil((x1 - x0) / res)), int(np.ceil((y1 - y0) / res))
    grid, _, _ = np.histogram2d(y, x, bins=[ny, nx], range=[[y0, y0 + ny * res], [x0, x0 + nx * res]], weights=w)
    kx = np.arange(-int(4 * hx / res) - 1, int(4 * hx / res) + 2) * res
    ky = np.arange(-int(4 * hy / res) - 1, int(4 * hy / res) + 2) * res
    kern = np.exp(-0.5 * (ky[:, None] / hy) ** 2) * np.exp(-0.5 * (kx[None, :] / hx) ** 2)
    dens = fftconvolve(grid, kern / (2 * np.pi * hx * hy * w.sum()), mode="same")
    dens = np.clip(dens, 0, None)
    std = (dens - dens.min()) / (dens.max() - dens.min())
    return {"grid": std[::-1], "x0": x0, "y1": y0 + ny * res, "res": res, "bandwidth_m": (hx, hy),
            "n_records": int(w.sum())}


def sample_background(cell_xy: np.ndarray, cell_bias: np.ndarray, size: int = 10_000, seed: int = 0) -> np.ndarray:
    """phyloregion::backg: all calibration cells if there are <= size, else ``size`` cells drawn without
    replacement with probability proportional to the bias value."""
    return cell_xy[sample_background_index(len(cell_xy), cell_bias, size, seed)]


def sample_background_index(n: int, cell_bias: np.ndarray, size: int = 10_000, seed: int = 0,
                            inplace: bool = False) -> np.ndarray:
    """Indices of the cells ``sample_background`` draws, so callers need not hold every cell's coordinates.
    ``inplace``: the bias array may be overwritten (it becomes the sampling probabilities), which saves two copies of
    a per-cell array for calibration areas of ~2e8 cells; the draw is unchanged."""
    if n <= size:
        return np.arange(n)
    p = np.clip(cell_bias, 0, None, out=cell_bias if inplace else None).astype(np.float64, copy=False)
    total = p.sum()
    if total > 0:
        p /= total
    else:
        p = None
    return np.random.default_rng(seed).choice(n, size=size, replace=False, p=p)
