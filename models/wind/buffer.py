"""The controlled comparison a boundary buffer is judged by.

Two solves of one bundle that differ only in the buffer: the grid of the first covers the display
disc, the grid of the second the disc plus its buffer. Every other setting is shared, including
the vertical grid, whose floor both take from the bundle's terrain raster. Inside the display
disc the two scenes are the same cells but for the few edge columns whose gaps fill from beyond
it, which `identical_inside` counts.

A buffer should move the flow near the disc edge and leave the centre almost untouched, so the
change is reported against distance from the edge, in bands. For a building the disc cuts, the
volume flux entering its footprint and the speed just beyond the cut are reported for both arms;
through a solid both are zero.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from domain import Scene, radius
from solver import Result


@dataclass(frozen=True)
class BufferConfig:
    """What the comparison reports.

    Attributes:
        bands_m: Lower edges of the distance-from-edge bands [m]; the last runs to the centre.
        pedestrian_m: Height above the terrain of the pedestrian level [m].
        column_top_m: Height above the terrain the column mean runs to [m].
        erode_m: Inset of a mapped footprint before it is used, covering the metre-level gap
            between a mapped outline and the surveyed wall [m].
    """

    bands_m: Tuple[float, ...] = (0.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0)
    pedestrian_m: float = 2.0
    column_top_m: float = 30.0
    erode_m: float = 1.5


def window(small: Scene, large: Scene) -> Tuple[slice, slice]:
    """Rows and columns of `large` that `small`'s grid occupies; the two lattices must coincide."""
    g, h = small.grid, large.grid
    assert g.dx == h.dx and np.array_equal(g.zf, h.zf), "the arms do not share one grid"
    ox = (small.origin[0] - large.origin[0]) / g.dx
    oy = (small.origin[1] - large.origin[1]) / g.dx
    assert ox == round(ox) and oy == round(oy) and small.origin[2] == large.origin[2], (
        "the arms' cells are not the same cells")
    return slice(int(oy), int(oy) + g.ny), slice(int(ox), int(ox) + g.nx)


def disc(scene: Scene, r: float) -> np.ndarray:
    """(ny, nx) columns whose centres lie within `r` of the scene origin."""
    return radius(scene.grid, scene.origin[:2]) <= r


def identical_inside(before: Scene, after: Scene, r: float) -> Dict[str, int]:
    """Cells inside radius `r` whose solid, drag or roughness differ between the arms."""
    rows, cols = window(before, after)
    inner = disc(before, r)
    return {"solid": int((before.solid != after.solid[:, rows, cols])[:, inner].sum()),
            "sink": int((before.sink != after.sink[:, rows, cols])[:, inner].sum()),
            "z0": int((before.z0 != after.z0[rows, cols])[inner].sum()),
            "columns_compared": int(inner.sum())}


def terrain_level(scene: Scene) -> np.ndarray:
    """(ny, nx) index of the lowest cell above the terrain of each column, buildings aside."""
    zt = np.zeros(scene.z0.shape) if scene.terrain is None else scene.terrain
    return np.minimum((scene.grid.zc[:, None, None] < zt[None]).sum(axis=0), scene.grid.nz - 1)


def level_above(scene: Scene, height_m: float) -> np.ndarray:
    """(ny, nx) level whose centre is nearest `height_m` above each column's terrain."""
    zc, zf = scene.grid.zc, scene.grid.zf
    target = zf[terrain_level(scene)] + height_m
    return np.abs(zc[:, None, None] - target[None]).argmin(axis=0)


def band_table(delta: np.ndarray, base: np.ndarray, dist: np.ndarray,
               edges: Sequence[float], limit: float) -> List[Dict[str, float]]:
    """Change against distance from the disc edge.

    Args:
        delta: After minus before [m/s], one value per compared sample.
        base: Before [m/s], same samples.
        dist: Distance of each sample inside the disc edge [m].
        edges: Lower band edges [m].
        limit: Upper edge of the last band [m].
    Returns:
        Per band: samples, mean and 95th percentile of |change|, the mean change, and the mean
        |change| over the mean before speed.
    """
    hi = list(edges[1:]) + [limit]
    out = []
    for lo, up in zip(edges, hi):
        m = (dist >= lo) & (dist < up)
        a = np.abs(delta[m])
        row = {"from_edge_m": [lo, up], "samples": int(m.sum())}
        if m.any():
            row.update(mean_abs_change_m_s=float(a.mean()), p95_abs_change_m_s=float(np.percentile(a, 95)),
                       mean_change_m_s=float(delta[m].mean()),
                       relative_mean_abs_change=float(a.mean() / base[m].mean()))
        out.append(row)
    return out


def edge_bands(before: Tuple[Scene, Result], after: Tuple[Scene, Result], r: float,
               cfg: BufferConfig) -> Dict[str, List[Dict[str, float]]]:
    """Speed change against distance from the disc edge, at pedestrian height and over the column.

    Samples are cells fluid in both arms, inside the disc.
    """
    (sb, rb), (sa, ra) = before, after
    rows, cols = window(sb, sa)
    speed_b, speed_a = rb.speed, ra.speed[:, rows, cols]
    fluid = ~sb.solid & ~sa.solid[:, rows, cols]
    inner = disc(sb, r)
    dist = r - radius(sb.grid, sb.origin[:2])
    k = level_above(sb, cfg.pedestrian_m)
    iy, ix = np.nonzero(inner & np.take_along_axis(fluid, k[None], 0)[0])
    ped = band_table(speed_a[k[iy, ix], iy, ix] - speed_b[k[iy, ix], iy, ix], speed_b[k[iy, ix], iy, ix],
                     dist[iy, ix], cfg.bands_m, r)
    top = sb.grid.zf[terrain_level(sb)] + cfg.column_top_m
    col = fluid & inner[None] & (sb.grid.zc[:, None, None] < top[None])
    kk, yy, xx = np.nonzero(col)
    column = band_table(speed_a[kk, yy, xx] - speed_b[kk, yy, xx], speed_b[kk, yy, xx], dist[yy, xx],
                        cfg.bands_m, r)
    return {"pedestrian": ped, "column_to_30m": column}


def prism(scene: Scene, footprint: np.ndarray, ring: int = 3) -> np.ndarray:
    """(nz, ny, nx) cells inside the footprint from the ground around it to each column's top solid.

    Under a building the surface raster's lowest crossing is the roof, so the footprint's own
    terrain is not ground; the ground is the median terrain level of the columns within `ring`
    cells outside the footprint.
    """
    from scipy.ndimage import binary_dilation

    around = binary_dilation(footprint, iterations=ring) & ~footprint
    ground = int(np.median(terrain_level(scene)[around])) if around.any() else 0
    solid_top = np.where(scene.solid.any(axis=0), scene.grid.nz - np.argmax(scene.solid[::-1], axis=0), 0)
    levels = np.arange(scene.grid.nz)[:, None, None]
    return footprint[None] & (levels >= ground) & (levels < solid_top[None])


def entering_flux(res: Result, scene: Scene, cells: np.ndarray) -> float:
    """Volume flux [m^3/s] entering a set of cells through the faces that bound it."""
    dx, dz = scene.grid.dx, scene.grid.dz[:, None, None]
    total = 0.0
    for axis, face, area in ((2, res.faces[0], dx * dz), (1, res.faces[1], dx * dz), (0, res.faces[2], dx * dx)):
        p = np.pad(cells, [(1, 1) if a == axis else (0, 0) for a in range(3)])
        n = p.shape[axis]
        hi, lo = np.take(p, np.arange(1, n), axis=axis), np.take(p, np.arange(n - 1), axis=axis)
        total += float(((np.clip(face, 0, None) * (hi & ~lo) + np.clip(-face, 0, None) * (lo & ~hi)) * area).sum())
    return total


def cut_face(footprint: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """(ny, nx) footprint columns beyond the disc that touch the footprint inside it."""
    part = footprint & inner
    touch = np.zeros_like(part)
    touch[1:] |= part[:-1]
    touch[:-1] |= part[1:]
    touch[:, 1:] |= part[:, :-1]
    touch[:, :-1] |= part[:, 1:]
    return touch & footprint & ~inner


def building(before: Tuple[Scene, Result], after: Tuple[Scene, Result], footprint: np.ndarray,
             r: float) -> Dict[str, object]:
    """Flux into a cut building and speed just beyond its cut, in both arms.

    Args:
        before, after: (scene, result) of each arm.
        footprint: (ny, nx) footprint on the after arm's grid.
        r: Display radius [m].
    Returns:
        For each arm: volume flux entering the footprint's cells below the roof the after arm
        sees, the fluid fraction and mean speed of the part beyond the disc, and the mean speed
        in the first column beyond the cut.
    """
    (sb, rb), (sa, ra) = before, after
    rows, cols = window(sb, sa)
    fp = footprint[rows, cols]
    inner = disc(sb, r)
    cells = prism(sa, footprint)[:, rows, cols]
    beyond = cells & ~inner[None]
    face = cells & cut_face(fp, inner)[None]
    out = {"footprint_columns": int(fp.sum()), "footprint_columns_beyond_disc": int((fp & ~inner).sum()),
           "footprint_columns_without_solid_in_after_arm": int((fp & ~sa.solid[:, rows, cols].any(axis=0)).sum()),
           "cells_below_roof": int(cells.sum()), "cells_below_roof_beyond_disc": int(beyond.sum()),
           "cut_face_cells": int(face.sum())}
    for name, scene, res, sub in (("before", sb, rb, (slice(None), slice(None))), ("after", sa, ra, (rows, cols))):
        full = np.zeros(scene.solid.shape, bool)
        full[:, sub[0], sub[1]] = cells
        speed = res.speed[:, sub[0], sub[1]]
        solid = scene.solid[:, sub[0], sub[1]]
        out[name] = {"flux_into_footprint_m3_s": entering_flux(res, scene, full),
                     "fluid_fraction_beyond_disc": float((~solid[beyond]).mean()) if beyond.any() else None,
                     "mean_speed_beyond_disc_m_s": float(speed[beyond].mean()) if beyond.any() else None,
                     "mean_speed_on_cut_face_m_s": float(speed[face].mean()) if face.any() else None,
                     "max_speed_on_cut_face_m_s": float(speed[face].max()) if face.any() else None}
    return out


def footprint(scene: Scene, outline: np.ndarray, erode_m: float) -> np.ndarray:
    """(ny, nx) columns whose centres lie inside a scene-frame outline and `erode_m` from its edge.

    Args:
        scene: Grid the footprint is wanted on.
        outline: (n, 2) polygon vertices, scene metres.
        erode_m: Inset [m].
    Returns:
        Boolean footprint.
    """
    from scipy.ndimage import distance_transform_edt

    x, y = np.meshgrid(scene.origin[0] + scene.grid.xc, scene.origin[1] + scene.grid.yc)
    inside = np.zeros(x.shape, bool)
    for (x1, y1), (x2, y2) in zip(outline, np.roll(outline, -1, axis=0)):
        if y1 != y2:
            inside ^= ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)
    return distance_transform_edt(inside) * scene.grid.dx > erode_m
