"""Light through a tree canopy, per 2 m column, from a LiDAR survey's first returns and MODIS leaf area.

Each column's nadir optical depth is its first-return gap fraction (a pulse's first return reaching within 1 m of
the ground is a gap):

    gap = (first returns within 1 m of the ground + 0.5) / (first returns + 1),   tau0 = -ln(gap)

NaN under 8 first returns. The optical depth above a height h is the same count from h up, as a share F(h) of tau0.
Leaves through the year are added to (or taken from) the survey's own, allocated by each column's share of the site's
optical depth:

    tau(t) = tau0 + G Omega (dL(t) - dL_survey) tau0 / mean(tau0),   dL = max(LAI(t) - LAI_floor, 0)

with G = 0.5 (spherical leaf angles), Omega = 0.8 (deciduous broadleaf clumping; Chen et al. 2005) and the floor the
record's 5th-percentile LAI. Beer-Lambert: beam passes as exp(-tau / cos z), isotropic diffuse as 2 E3(tau).
"""

from typing import Optional

import numpy as np

OCCLUDE_M = 1.0
MIN_FIRST = 8
PROFILE_M = tuple(range(0, 32, 4))   # heights above the column's ground the profile is given at
G = 0.5
OMEGA = 0.8
LEAF_FALL_LAG_D = 10
"""Days a leaf stays on the tree after it loses its green: 50 % color to 50 % fall, the mean of 661 species-years of
the Harvard Forest phenology record (HF003)."""


def columns(xyz: np.ndarray, cls: np.ndarray, return_number: np.ndarray, cell: float = 2.0) -> dict:
    """tau0 per column and its profile F(h) from points (x, y, z) [m], their ASPRS class and return numbers. The
    column's ground is its lowest ground return (class 2), else its neighbors'."""
    x0, y0 = float(np.floor(xyz[:, 0].min())), float(np.floor(xyz[:, 1].min()))
    ix = ((xyz[:, 0] - x0) // cell).astype(np.int64)
    iy = ((xyz[:, 1] - y0) // cell).astype(np.int64)
    nx, ny = int(ix.max()) + 1, int(iy.max()) + 1
    col = iy * nx + ix
    zg = np.full(nx * ny, np.inf)
    g = cls == 2
    np.minimum.at(zg, col[g], xyz[g, 2])
    grid = zg.reshape(ny, nx)
    pad = np.pad(grid, 1, constant_values=np.inf)
    near = np.min([pad[1 + dy:ny + 1 + dy, 1 + dx:nx + 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=0)
    ground = np.where(np.isfinite(grid), grid, near).ravel()
    first = (return_number == 1) & np.isfinite(ground[col])
    h = xyz[:, 2] - ground[col]
    allf = np.bincount(col[first], minlength=nx * ny)
    lowf = np.bincount(col[first & (h < OCCLUDE_M)], minlength=nx * ny)
    tau = -np.log((lowf + 0.5) / (allf + 1.0))
    tau[allf < MIN_FIRST] = np.nan
    prof = np.zeros((len(PROFILE_M), nx * ny), np.float32)
    for i, hm in enumerate(PROFILE_M):
        below = np.bincount(col[first & (h < hm + OCCLUDE_M)], minlength=nx * ny)
        t_h = -np.log((below + 0.5) / (allf + 1.0))
        prof[i] = np.where(tau > 0, np.clip(t_h / np.where(tau > 0, tau, 1.0), 0.0, 1.0), 0.0)
    prof[:, allf < MIN_FIRST] = np.nan
    return {"x0": x0, "y0": y0, "cell": cell, "nx": nx, "ny": ny, "tau": tau.reshape(ny, nx),
            "profile": prof.reshape(len(PROFILE_M), ny, nx)}


def seasonal_tau(tau0: np.ndarray, lai_site: float, floor: float, lai_survey: Optional[float] = None) -> np.ndarray:
    """The nadir optical depth on a day with site leaf area `lai_site`: the day's leaves added, those the survey
    was flown with (`lai_survey`; None: the floor) taken out."""
    t0 = np.asarray(tau0, np.float64)
    fin = np.isfinite(t0)
    w = np.where(fin, t0, 0.0) / max(float(t0[fin].mean()) if fin.any() else 0.0, 1e-9)
    had = max((lai_survey if lai_survey is not None else floor) - floor, 0.0)
    return np.where(fin, np.maximum(t0 + G * OMEGA * (max(lai_site - floor, 0.0) - had) * w, 0.0), np.nan)


def leaves_on_trees(green: np.ndarray, lag: int = LEAF_FALL_LAG_D) -> np.ndarray:
    """MODIS counts green leaf area; a leaf that has turned keeps shading until it falls, so each day holds the
    most green leaf area of the `lag` days before it."""
    s = np.asarray(green, np.float64)
    x = np.concatenate([np.full(lag, s[0]), s])
    return np.array([x[i:i + lag + 1].max() for i in range(len(s))])


def beam_transmittance(tau: np.ndarray, cos_zenith: np.ndarray) -> np.ndarray:
    """exp(-tau / cos z): the direct beam through a column of nadir optical depth tau."""
    return np.exp(-np.asarray(tau) / np.maximum(np.asarray(cos_zenith), 1e-6))


def diffuse_transmittance(tau: np.ndarray) -> np.ndarray:
    """2 E3(tau): isotropic sky light through a layer of nadir optical depth tau (E3 the exponential integral)."""
    from scipy.special import expn
    return 2.0 * expn(3, np.asarray(tau, np.float64))
