"""A year's wind run at every LiDAR return, and the year's typical hour for the flow drawn over it.

The solver is linear in the reference speed, so a return's speed in hour t is U_ref(t) times the length of its unit
field blended between the two solved headings about the hour's direction. Hours are grouped by heading pair and blend
fraction (1/16 of 22.5 degrees), so each group's unit speed is computed once per return:

    R(p) = 3.6 sum_g |(1 - a_g) S_k(g)(p) + a_g S_k(g)+1(p)| sum_{t in g} U_ref(t)     (km)

S: the unit (u, v, w) per heading at each return, read from the solver's 3D field (`read_points_basis`); a return the
solve does not reach takes the flow beside it (`fill_solids`). The speed is the norm of every component S carries. PyTorch, any device.
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch

HEADINGS = 16                  # solved headings, 22.5 degrees apart (the direction the wind blows from)
STEP = 360.0 / HEADINGS
LEVELS = 16                    # blend fractions per heading pair
KM_PER_MS_H = 3.6              # 1 m/s for an hour
TOP, NONE = 254, 255           # codes 0..254 over [0, hi]; 255 no value
PLANT_HEIGHT_M = 2.0           # a return under this over the bare earth is filled as if this high (fill_solids' log law)
FILL_MAX = 3.0                 # the log-law ratio a filled return may take from its neighbor, either way
CALM_MS = 0.5                  # at or under this an hour is calm


# ------------------------------------------------------------------------------------------------ the run

def groups(theta_deg: np.ndarray) -> np.ndarray:
    """Hour -> group = heading k * LEVELS + blend level; NaN direction -> -1."""
    ok = ~np.isnan(theta_deg)
    x = np.where(ok, np.mod(theta_deg, 360.0) / STEP, 0.0)
    k = np.floor(x).astype(int) % HEADINGS
    lvl = np.minimum(np.rint((x - np.floor(x)) * LEVELS).astype(int), LEVELS)
    k = np.where(lvl == LEVELS, (k + 1) % HEADINGS, k)
    lvl = np.where(lvl == LEVELS, 0, lvl)
    return np.where(ok, k * LEVELS + lvl, -1)


def unit_field(S: torch.Tensor, g: int) -> torch.Tensor:
    """The unit field at every return for group g, blended between its two headings. S: [16, N, 2 or 3]."""
    k, lvl = divmod(int(g), LEVELS)
    a = lvl / LEVELS
    return (1.0 - a) * S[k].float() + a * S[(k + 1) % HEADINGS].float()


def run_km(S: torch.Tensor, uref: np.ndarray, theta: np.ndarray) -> torch.Tensor:
    """A year's wind run per return, km. S: [16, N, 3] unit (u, v, w) (or [16, N, 2]), NaN where the solve does not reach; uref: hourly
    reference speed, m/s; theta: hourly direction the wind blows from, degrees. A NaN hour counts for nothing."""
    ok = ~np.isnan(uref) & ~np.isnan(theta)
    g_of = groups(np.where(ok, theta, np.nan))
    total = torch.zeros(S.shape[1], device=S.device)
    for g in np.unique(g_of[g_of >= 0]):
        f = unit_field(S, int(g))
        total += torch.linalg.vector_norm(f, dim=1) * float(np.nansum(uref[g_of == g]))
    return total * KM_PER_MS_H


def nice_ceil(x: float) -> float:
    """The next 1, 2 or 5 times a power of ten at or above x: the scale's top."""
    if not x > 0:
        return 1.0
    e = 10.0 ** math.floor(math.log10(x))
    for m in (1.0, 2.0, 5.0, 10.0):
        if m * e >= x:
            return m * e
    return 10.0 * e


def quantize(run: np.ndarray, hi: float) -> np.ndarray:
    """One byte a return: 0..TOP over [0, hi] (clipped at hi), NONE where not a number. hi: the record's p98 over every
    year and return, `nice_ceil`, one scale for every year."""
    q = np.clip(np.rint(np.nan_to_num(run, nan=0.0) / hi * TOP), 0, TOP).astype(np.uint8)
    q[~np.isfinite(run)] = NONE
    return q


# ------------------------------------------------------------------------------------------------ S at each return
# Every return is read from the solver's 3D field in the same `levels` run that makes the published levels
# (`cli.py levels --points`, points.py): at its own z, CLEARANCE_M off every solid face it stands on or beside,
# trilinear over the fluid cell centres, one rule for every class. Its (u, v, w) per unit heading is
# points_basis_f16.bin, [16, N, 3] float16, with points_basis.json beside it.

POINTS_BASIS, POINTS_META = "points_basis_f16.bin", "points_basis.json"


def read_points_basis(path, n: int):
    """(S [16, N, 3] float16 unit (u, v, w) per heading at each return, sidecar) from `cli.py levels --points`, or
    (None, why) when the file is absent, holds another point set, or is not the 16 unit headings the run blends."""
    import json
    from pathlib import Path
    path = Path(path)
    if not path.is_file() or not (path.parent / POINTS_META).is_file():
        return None, "no points basis"
    meta = json.loads((path.parent / POINTS_META).read_text())
    k, m, c = meta["shape"]
    if m != n:
        return None, f"the points basis holds {m:,} points, the set {n:,}"
    if [float(h) for h in meta["headings_deg"]] != [i * STEP for i in range(HEADINGS)] or c != 3:
        return None, f"the points basis is not {HEADINGS} headings of (u, v, w): {meta['headings_deg']}, {c}"
    return np.fromfile(path, "<f2").reshape(k, m, c), meta


def fill_solids(S: np.ndarray, xy: np.ndarray, h: np.ndarray, z0: np.ndarray) -> Tuple[np.ndarray, int]:
    """A return the solve does not reach (inside a building's solid cells, above the top level) takes the nearest
    reached return in (x, y, height), carried to its own height by the log law,
    u(h) / u(h_n) = ln(h / z0) / ln(h_n / z0), clipped to FILL_MAX either way. Returns (S, returns filled)."""
    miss = ~np.isfinite(S).all(axis=(0, 2))
    if not miss.any() or miss.all():
        return S, 0
    from scipy.spatial import cKDTree
    fin, mi = np.nonzero(~miss)[0], np.nonzero(miss)[0]
    _, j = cKDTree(np.c_[xy[fin, 0], xy[fin, 1], h[fin]]).query(np.c_[xy[mi, 0], xy[mi, 1], h[mi]])
    nn = fin[j]
    zz = np.maximum(z0[mi].astype(np.float64), 1e-3)
    f = np.log(np.maximum(h[mi], 2 * zz) / zz) / np.log(np.maximum(h[nn], 2 * zz) / zz)
    f = np.clip(np.nan_to_num(f, nan=1.0), 1.0 / FILL_MAX, FILL_MAX)
    S = np.array(S)
    S[:, mi, :] = (S[:, nn, :].astype(np.float32) * f[None, :, None].astype(np.float32)).astype(S.dtype)
    return S, int(len(mi))


# ------------------------------------------------------------------------------------------------ the typical hour

def typical_hour(uref: np.ndarray, wind_from: np.ndarray) -> Optional[Tuple[int, float, float]]:
    """The year's typical hour: of the 16 sectors the one the wind came from most often (calm hours aside), and in it
    the hour at its median speed. Returns (hour, U_ref, direction) or None."""
    ok = (uref > CALM_MS) & np.isfinite(wind_from)
    if not ok.any():
        return None
    sec = np.rint(np.mod(np.nan_to_num(wind_from), 360.0) / STEP).astype(int) % HEADINGS
    k = int(np.bincount(sec[ok], minlength=HEADINGS).argmax())
    hrs = np.nonzero(ok & (sec == k))[0]
    h = int(hrs[np.argsort(uref[hrs], kind="stable")][len(hrs) // 2])
    return h, float(uref[h]), float(wind_from[h])


def ribbon_field(levels: np.ndarray, uref: float, wind_from: float) -> Tuple[np.ndarray, np.ndarray]:
    """The flow drawn over the Year: the lowest level blended between the two headings about the direction, times U_ref.
    levels: [16, 2, nz, ny, nx]. Returns (U, V) m/s, NaN where the solve does not reach."""
    x = (wind_from % 360.0) / STEP
    k0 = int(np.floor(x)) % HEADINGS
    w, k1 = x - np.floor(x), (k0 + 1) % HEADINGS
    a, b = levels[k0, :, 0], levels[k1, :, 0]
    f = uref * ((1.0 - w) * a + w * b)
    return f[0], f[1]


def year(S: torch.Tensor, uref: np.ndarray, theta: np.ndarray) -> Dict[str, np.ndarray]:
    """One year: the run per return (km) and the typical hour."""
    return {"run_km": run_km(S, uref, theta).cpu().numpy(), "typical_hour": typical_hour(uref, theta)}


def ribbon_over_top(over_top: np.ndarray, height: np.ndarray, uref: float,
                    wind_from: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The flow drawn over the Year where the levels run read it over each column's measured top (`cli.py levels
    --over-top-out`, view.over_top): over_top [16, 2, ny, nx] unit (u, v) per heading, height [ny, nx] over the bare
    earth. Returns (U, V) m/s and the height each is drawn at: the flow over crowns and roofs, so a depth test hides a
    ribbon only behind something taller than its own column."""
    U, V = ribbon_field(np.asarray(over_top, np.float32)[:, :, None], uref, wind_from)
    return U, V, np.asarray(height, np.float32)
