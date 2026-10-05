"""Each ground cell's representative return: the LiDAR return whose sunlight and wind the cell's water balance takes.

The balance runs on a ground grid; its shortwave Rs and 2 m wind u2 are read at one return per cell:

  - a canopy cell: the crown surface, the highest return within 2 m (a canopy height model's surface). A 0.5 m cell's
    own highest return is often a lower branch in a crown gap, shaded by its neighbors' crowns;
  - an open cell: its highest up-facing return within 1 m of the bare earth. A lawn beside a building otherwise takes an
    eave or roof edge meters above it, and the roof's light and wind;
  - an open cell with no such return (under an eave): its nearest open neighbor's ground return;
  - any other cell: its highest up-facing return, else its highest return; an empty cell, the nearest cell's.

Each hour, a canopy cell's Rs and u2 are the mean over the canopy cells within 2 m (`crown_smoother`): the crown
surface shares one return per 2 m window, which would otherwise print 2 m squares into the soil water.
"""

from typing import Callable, Optional, Tuple

import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import cKDTree

UP_NZ = 0.7                    # a return faces up within 45 degrees of vertical
GROUND_MAX_M = 1.0             # an open cell's return lies within this of its bare earth
CROWN_M = 2.0                  # the crown surface's window, and the hourly smoothing's


def representative(z: np.ndarray, height: np.ndarray, up: np.ndarray, cell: np.ndarray, shape: Tuple[int, int],
                   dx: float, canopy: np.ndarray, roof: np.ndarray) -> np.ndarray:
    """The representative return per cell, an index into the returns.

    Args:
        z: [N] elevation of each return, m.
        height: [N] height above the bare earth, m.
        up: [N] bool: faces up (n_z >= UP_NZ) or is a leaf.
        cell: [N] the cell each return falls in, row-major over `shape`; -1 outside.
        shape: (ny, nx) of the ground grid. dx: its cell size, m.
        canopy, roof: [ny * nx] bool per cell, from the surface classes.
    Returns:
        [ny * nx] int64.
    """
    ny, nx = shape
    n_cells = ny * nx
    ok = cell >= 0
    low = height <= GROUND_MAX_M
    open_cell = ~np.asarray(roof, bool) & ~np.asarray(canopy, bool)
    rep = np.full(n_cells, -1, np.int64)
    for m in (ok & up & (low | ~open_cell[np.maximum(cell, 0)]), ok & up, ok):
        zi = np.nonzero(m)[0]
        zi = zi[np.argsort(-z[zi], kind="stable")]                                    # highest first
        ucell, first = np.unique(cell[zi], return_index=True)
        fresh = rep[ucell] < 0
        rep[ucell[fresh]] = zi[first][fresh]
    # an open cell whose every return is high takes the nearest open cell's ground return
    bad = open_cell & (rep >= 0) & ~low[np.maximum(rep, 0)]
    good = open_cell & (rep >= 0) & low[np.maximum(rep, 0)]
    if bad.any() and good.any():
        g, b = np.nonzero(good)[0], np.nonzero(bad)[0]
        _, j = cKDTree(np.c_[g // nx, g % nx]).query(np.c_[b // nx, b % nx])
        rep[b] = rep[g[j]]
    # a canopy cell: the highest return within CROWN_M, found as the window maximum of (height, cell) packed in one key
    assert n_cells < 1 << 23, "the crown key packs a cell index in 23 bits"
    zi = np.nonzero(ok)[0]
    zi = zi[np.argsort(-z[zi], kind="stable")]
    ucell, first = np.unique(cell[zi], return_index=True)
    top = np.full(n_cells, -1, np.int64)
    top[ucell] = zi[first]
    zq = np.where(top >= 0, np.round((z[np.maximum(top, 0)] - z.min()) * 100.0), -1)
    key = np.where(top >= 0, (zq.astype(np.int64) << 23) | np.arange(n_cells, dtype=np.int64), -1)
    w = max(1, int(round(CROWN_M / abs(dx))))
    kmax = ndimage.maximum_filter(key.reshape(ny, nx), size=w, mode="nearest").ravel()
    src = np.where(kmax >= 0, kmax & ((1 << 23) - 1), 0)
    cm = np.asarray(canopy, bool) & (kmax >= 0) & (top[src] >= 0)
    rep[cm] = top[src[cm]]
    # an empty cell takes the nearest cell's
    have = rep >= 0
    if (~have).any() and have.any():
        hv, miss = np.nonzero(have)[0], np.nonzero(~have)[0]
        _, j = cKDTree(np.c_[hv // nx, hv % nx]).query(np.c_[miss // nx, miss % nx])
        rep[miss] = rep[hv[j]]
    return rep


def crown_smoother(canopy: np.ndarray, shape: Tuple[int, int], dx: float, device="cpu") -> Optional[Callable]:
    """A function taking an hour's per-cell field [lanes, cells] to its mean over the canopy cells within CROWN_M (a
    masked box mean) on canopy cells, unchanged elsewhere; None when no cell is canopy."""
    if not np.asarray(canopy).any():
        return None
    ny, nx = shape
    w = max(1, int(round(CROWN_M / abs(dx))))
    m = torch.as_tensor(np.asarray(canopy, np.float32).reshape(1, 1, ny, nx), device=device)

    def pool(x):
        return torch.nn.functional.avg_pool2d(x, w, stride=1, padding=w // 2, count_include_pad=False)[..., :ny, :nx]

    den = torch.clamp(pool(m), min=1e-6)
    on = m.reshape(1, -1) > 0

    def smooth(v: torch.Tensor) -> torch.Tensor:
        b = v.shape[0]
        s = pool(v.reshape(b, 1, ny, nx) * m) / den
        return torch.where(on, s.reshape(b, -1), v)
    return smooth
