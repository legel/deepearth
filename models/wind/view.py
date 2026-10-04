"""The viewer's wind product from the 3D solve: heights above the ground the viewer draws on, on
the viewer's grid, shown only inside the display disc.

The viewer draws a few horizontal slices, each a fixed height above the local terrain, on its own
2 m grid over the display disc. A buffered solve covers more than that; everything beyond the
display radius is written as NaN, so the buffer moves the boundary without being displayed.

A day of frames is composed from one unit-speed solve per heading: every steady term of the
momentum balance is homogeneous of degree two in velocity, so a field at speed s is s times the
unit field, and a heading between two solved ones is their angular blend.

`levels` samples the solve a fixed height above the bare earth: a level never follows a roof. A cell
whose structure stands taller than the level is solid there and carries no wind; every other cell
carries a finite value, trilinear over the fluid cell centres around it. Where the level lies below
its column's lowest fluid centre, that centre's value is carried down by the log law over the
column's roughness, and the cell is labelled filled.
"""

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from domain import Scene, pockets, radius


@dataclass(frozen=True)
class ViewConfig:
    """The viewer's grid and slices.

    Attributes:
        heights_m: Slice heights above the terrain [m].
        cell_m: Viewer cell [m].
        half_m: Half-width of the viewer's square, centred on the scene origin [m].
        spacing_deg: Heading spacing of the unit-speed solves [deg].
        box: On a site that follows its polygon, the viewer grid's bounds (x0, y0, x1, y1) [scene m], the
            display box snapped outward to the cell (`for_box`); None is the square of `half_m`.
        display: That site's display box (x0, y0, x1, y1): cells outside it are NaN, as beyond the disc.
    """

    heights_m: Tuple[float, ...] = (2.0, 10.0, 30.0)
    cell_m: float = 2.0
    half_m: float = 120.0
    spacing_deg: float = 22.5
    box: Optional[Tuple[float, float, float, float]] = None
    display: Optional[Tuple[float, float, float, float]] = None

    @classmethod
    def for_box(cls, display: Sequence[float], cell_m: float = 2.0, **kw) -> "ViewConfig":
        """The viewer grid over a display box, its edges snapped outward to multiples of `cell_m`."""
        e = 1e-9
        b = (math.floor(display[0] / cell_m + e) * cell_m, math.floor(display[1] / cell_m + e) * cell_m,
             math.ceil(display[2] / cell_m - e) * cell_m, math.ceil(display[3] / cell_m - e) * cell_m)
        return cls(cell_m=cell_m, box=b, display=tuple(float(v) for v in display), **kw)

    @property
    def faces(self) -> List[float]:
        """Level faces whose centres are `heights_m`, the first starting at the terrain."""
        out = [0.0]
        for h in self.heights_m:
            out.append(2 * h - out[-1])
        return out

    @property
    def n(self) -> int:
        """Viewer cells across the square (east-west on a box)."""
        return self.nx if self.box else int(round(2 * self.half_m / self.cell_m))

    @property
    def nx(self) -> int:
        return int(round((self.box[2] - self.box[0]) / self.cell_m)) if self.box else self.n

    @property
    def ny(self) -> int:
        return int(round((self.box[3] - self.box[1]) / self.cell_m)) if self.box else self.n

    @property
    def origin(self) -> Tuple[float, float, float]:
        """The product's north-west corner [scene m], as `frames.Header` takes it."""
        return (self.box[0], self.box[3], 0.0) if self.box else (-self.half_m, self.half_m, 0.0)

    def centres(self) -> Tuple[np.ndarray, np.ndarray]:
        """(ny, nx) scene x and y [m] of the viewer's cell centres, rows running north."""
        if self.box:
            return np.meshgrid(self.box[0] + (np.arange(self.nx) + 0.5) * self.cell_m,
                               self.box[1] + (np.arange(self.ny) + 0.5) * self.cell_m)
        c = -self.half_m + (np.arange(self.n) + 0.5) * self.cell_m
        return np.meshgrid(c, c)

    def shown(self, x: np.ndarray, y: np.ndarray, display_radius: float) -> np.ndarray:
        """True where a point is shown: inside the display box on a box site, else within the display radius."""
        if self.display:
            d = self.display
            return (x >= d[0]) & (x <= d[2]) & (y >= d[1]) & (y <= d[3])
        return np.hypot(x, y) <= display_radius


def half_for(display_radius_m: float, cell_m: float = 2.0, least_m: float = 120.0) -> float:
    """The viewer square's half-width for a site: its display radius rounded up to the viewer cell, never under the
    Campanile's `least_m`. The square was 120 m for every site, so a site's 301.52 m disc published a 240 m square,
    a fifth of its area (2026-09-14)."""
    return max(float(least_m), math.ceil(float(display_radius_m) / cell_m - 1e-9) * cell_m)


def viewer_ground(path: Path, scene: Scene, cfg: ViewConfig, cell_m: float = 1.0) -> np.ndarray:
    """(ny, nx) scene-frame elevation [m] of the surface the viewer draws its slices over.

    The viewer reads a float32 raster of `cell_m` cells over its square, rows running north, and
    area-averages it onto the product's grid; this places the same average on the solver grid, with
    the solver's own terrain wherever the viewer's square does not reach.
    """
    assert cfg.box is None, "the viewer ground raster is a disc site's square"
    n, f = int(round(2 * cfg.half_m / cell_m)), int(round(cfg.cell_m / cell_m))
    raw = np.fromfile(path, "<f4").reshape(n, n)
    avg = raw.reshape(n // f, f, n // f, f).mean(axis=(1, 3))
    ground = scene.origin[2] + (np.zeros(scene.z0.shape) if scene.terrain is None else scene.terrain)
    c0 = int(round((-cfg.half_m - scene.origin[0]) / cfg.cell_m))
    r0 = int(round((-cfg.half_m - scene.origin[1]) / cfg.cell_m))
    ground[r0:r0 + n // f, c0:c0 + n // f] = avg
    return ground


def slices(velocity: np.ndarray, scene: Scene, heights: Sequence[float], ground: np.ndarray) -> np.ndarray:
    """(2, len(heights), ny, nx) u and v `heights` above `ground` [scene m], NaN inside solids."""
    out = np.full((2, len(heights)) + velocity.shape[2:], np.nan)
    iy, ix = np.indices(velocity.shape[2:])
    zc = scene.origin[2] + scene.grid.zc
    for j, h in enumerate(heights):
        k = np.abs(zc[:, None, None] - (ground + h)[None]).argmin(axis=0)
        out[:, j] = np.where(~scene.solid[k, iy, ix], velocity[:2, k, iy, ix], np.nan)
    return out


def crop(fields: np.ndarray, scene: Scene, cfg: ViewConfig, display_radius: float) -> np.ndarray:
    """The viewer's square, rows running north for `frames.Writer`, NaN beyond the display radius."""
    assert scene.grid.dx == cfg.cell_m, "solve on the viewer's cell"
    assert cfg.box is None, "crop is a disc site's square"
    c0 = int(round((-cfg.half_m - scene.origin[0]) / cfg.cell_m))
    r0 = int(round((-cfg.half_m - scene.origin[1]) / cfg.cell_m))
    n = int(round(2 * cfg.half_m / cfg.cell_m))
    beyond = radius(scene.grid, scene.origin[:2])[r0:r0 + n, c0:c0 + n] > display_radius
    return np.where(beyond, np.nan, fields[..., r0:r0 + n, c0:c0 + n])


def headings(directions: Sequence[float], spacing: float) -> List[float]:
    """Solved headings bracketing every direction of the record."""
    lo = np.floor((np.asarray(directions, float) % 360.0) / spacing) * spacing
    return sorted({float(v % 360.0) for v in np.r_[lo, lo + spacing]})


def share(headings_: Sequence[float], part: str) -> List[float]:
    """Part "i/k" of the headings: every k-th from the i-th, so k processes, one per GPU, solve them
    all between them and none twice."""
    i, k = (int(v) for v in part.split("/"))
    assert 0 <= i < k, f"part {part}: i must be in [0, k)"
    return list(headings_)[i::k]


def blend(basis: Dict[float, np.ndarray], direction: float, spacing: float) -> np.ndarray:
    """The unit field at `direction`, linear in angle between the two solved headings around it."""
    d = direction % 360.0
    lo = np.floor(d / spacing) * spacing
    w = (d - lo) / spacing
    return (1 - w) * basis[float(lo % 360.0)] + w * basis[float((lo + spacing) % 360.0)]


def walkable(scene: Scene) -> np.ndarray:
    """(ny, nx) scene-frame elevation [m] of the walkable top."""
    top = scene.terrain if scene.top is None else scene.top
    return scene.origin[2] + (np.zeros(scene.z0.shape) if top is None else top)


def first_fluid(scene: Scene) -> Tuple[np.ndarray, np.ndarray]:
    """(ny, nx) height [m] of each column's lowest fluid cell centre above its walkable top, and
    that cell's thickness [m]. Columns are solid from the floor up, so the count of solid cells is
    the index of the first fluid one."""
    top = walkable(scene) - scene.origin[2]
    k = np.minimum(scene.solid.sum(axis=0), scene.grid.nz - 1)
    return scene.grid.zc[k] - top, scene.grid.dz[k]


def trilinear(field: np.ndarray, scene: Scene, x: np.ndarray, y: np.ndarray,
              z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The cell-centred (c, nz, ny, nx) `field` at scene points, trilinear between cell centres.

    Args:
        field: Cell-centred values.
        scene: The scene `field` was solved on.
        x, y, z: (n,) scene-frame points [m].

    Returns:
        (c, n) values, and (n,) True where every centre carrying weight is fluid and the point is
        not below the lowest level of centres.
    """
    g = scene.grid
    f = [np.interp(z - scene.origin[2], g.zc, np.arange(g.nz)),
         np.clip((y - scene.origin[1]) / g.dx - 0.5, 0, g.ny - 1),
         np.clip((x - scene.origin[0]) / g.dx - 0.5, 0, g.nx - 1)]
    i0 = [np.minimum(np.floor(a).astype(int), n - 2) for a, n in zip(f, g.shape)]
    w1 = [a - i for a, i in zip(f, i0)]
    out = np.zeros((field.shape[0], len(x)))
    ok = z - scene.origin[2] >= g.zc[0]
    for corner in itertools.product((0, 1), repeat=3):
        w = np.prod([wi if c else 1 - wi for wi, c in zip(w1, corner)], axis=0)
        k, j, i = (a + c for a, c in zip(i0, corner))
        out += w * field[:, k, j, i]
        ok &= (w <= 0) | ~scene.solid[k, j, i]
    return out, ok


def trilinear_fluid(field: np.ndarray, scene: Scene, x: np.ndarray, y: np.ndarray,
                    z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """`trilinear` over the fluid centres alone, weights renormalised: a solid centre holds no wind.

    Returns:
        (c, n) values, NaN where no fluid centre carries weight, and (n,) that fluid weight.
    """
    g = scene.grid
    f = [np.interp(z - scene.origin[2], g.zc, np.arange(g.nz)),
         np.clip((y - scene.origin[1]) / g.dx - 0.5, 0, g.ny - 1),
         np.clip((x - scene.origin[0]) / g.dx - 0.5, 0, g.nx - 1)]
    i0 = [np.minimum(np.floor(a).astype(int), n - 2) for a, n in zip(f, g.shape)]
    w1 = [a - i for a, i in zip(f, i0)]
    out, wsum = np.zeros((field.shape[0], len(x))), np.zeros(len(x))
    for corner in itertools.product((0, 1), repeat=3):
        w = np.prod([wi if c else 1 - wi for wi, c in zip(w1, corner)], axis=0)
        k, j, i = (a + c for a, c in zip(i0, corner))
        w = np.where(scene.solid[k, j, i], 0.0, w)
        out += w * field[:, k, j, i]
        wsum += w
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(wsum > 1e-12, out / wsum, np.nan), wsum


def reroot(v: np.ndarray, z_from: np.ndarray, z_to: np.ndarray, z0: np.ndarray) -> np.ndarray:
    """`v` at `z_from` above a surface carried to `z_to` by the log law ln(1 + z / z0), the forcing's own
    profile: zero at the surface, direction held, every component scaled alike."""
    return v * (np.log1p(np.maximum(z_to, 0.0) / z0) / np.log1p(z_from / z0))


def ground(scene: Scene, cfg: ViewConfig) -> np.ndarray:
    """(n, n) scene-frame bare-earth elevation under each viewer cell, rows running north."""
    t = np.zeros(scene.z0.shape) if scene.terrain is None else scene.terrain
    return under(scene, cfg, scene.origin[2] + t)


def solid_at(scene: Scene, cfg: ViewConfig, h: float) -> np.ndarray:
    """(n, n) True where a solid column's top stands more than `h` above the bare earth under it."""
    t = np.zeros(scene.z0.shape) if scene.terrain is None else scene.terrain
    return under(scene, cfg, walkable(scene) - scene.origin[2] - t > h)


def levels(field: np.ndarray, scene: Scene, cfg: ViewConfig,
           display_radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`field` at `cfg.heights_m` above the bare earth on the viewer's square.

    A structure taller than a level makes that cell solid; every other cell in the disc is finite:
    trilinear over the fluid centres, and below the column's lowest fluid centre that centre's value
    re-rooted down by the log law over the column's roughness (filled).

    Returns:
        (c, len(heights), n, n) values, rows running north for `frames.Writer`, NaN beyond
        `display_radius` and in solid cells; the (len(heights), n, n) solid mask and filled mask,
        both inside the radius only.
    """
    x, y = cfg.centres()
    base, top = ground(scene, cfg), surface(scene, cfg)
    above = under(scene, cfg, first_fluid(scene)[0])
    z0 = under(scene, cfg, scene.z0)
    beyond = ~cfg.shown(x, y, display_radius)
    shape = (len(cfg.heights_m),) + x.shape
    out = np.full((field.shape[0],) + shape, np.nan)
    solid, filled = np.zeros(shape, bool), np.zeros(shape, bool)
    for j, h in enumerate(cfg.heights_m):
        solid[j] = solid_at(scene, cfg, h) & ~beyond
        solid[j] |= pockets(~solid[j]) & ~beyond
        rise = base + h - top                      # height above the walkable top under the level
        low = rise < above
        z = np.where(low, top + above, base + h)
        v, _ = trilinear_fluid(field, scene, x.ravel(), y.ravel(), z.ravel())
        v = v.reshape((-1,) + x.shape)
        with np.errstate(invalid="ignore", divide="ignore"):   # solid cells divide by zero; they are dropped
            v = np.where(low, reroot(v, above, rise, z0), v)
        keep = ~solid[j] & ~beyond
        out[:, j] = np.where(keep, v, np.nan)
        filled[j] = low & keep
    return out, solid, filled


OVER_TOP_M = 2.0
"""The clearance the ribbons fly at over each column's measured top: the flow over crowns and roofs, so the page's depth
test hides a ribbon only behind something taller than its own column (2026-10-01). Under Harvard's closed canopy
the ribbons on the lowest level over the bare earth (4 m) were almost all hidden inside the crowns."""


def crown_top(scene: Scene) -> np.ndarray:
    """(ny, nx) scene-frame elevation [m] of each column's measured top: its walkable top (a roof, or the bare earth), or
    where the column carries canopy drag, its crown: the survey's own top (`scene.canopy_top`, the DSM), else the top
    face of its highest drag cell."""
    has = np.asarray(scene.sink) > 1e-9
    plant = has.any(axis=0)
    if getattr(scene, "canopy_top", None) is not None:
        crown = np.where(plant, scene.origin[2] + np.asarray(scene.canopy_top, float), -np.inf)
    else:
        nz = scene.grid.nz
        k = np.where(plant, nz - 1 - np.argmax(has[::-1], axis=0), 0)
        crown = np.where(plant, scene.origin[2] + scene.grid.zf[k + 1], -np.inf)
    return np.maximum(walkable(scene), crown)


RAISED_M = 2.0
"""A column whose measured top stands this far over its bare earth is raised: a crown or a structure."""


def envelope(scene: Scene, where: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, float]]:
    """(ny, nx) height [m] over the bare earth of the surface the wind passes over: the canopy's envelope, at or above
    every measured top under it. Over a canopy the flow forms a shear layer at the crown envelope and skims gaps narrower
    than about the canopy height h_c, separating at a crown and reattaching beyond, rather than entering every gap (the
    mixing-layer analogy, Raupach, Finnigan and Brunet 1996, Boundary-Layer Meteorol. 78: 351; skimming flow where gap
    width W < h_c roughly, Oke 1988, Energy and Buildings 11: 103).

    h_c is the median top of the raised columns (RAISED_M and over), r = h_c / 2. A top more than r over its local
    canopy (the median top within h_c) is emergent, a tower or a lone crown standing over its neighbours: it is capped
    there and pierces the envelope, hiding the ribbons behind it. The capped tops are closed with a disc of radius r (a
    gap narrower than h_c is filled, a clearing wider is not), dilated by r and smoothed with a Gaussian of sigma r / 2,
    and the envelope is that or the capped top, whichever is higher: a Gaussian alone sat under the crowns' peaks and the
    crowns hid the ribbons over most of Harvard's near half (2026-10-01). The radii, the gaps' widths (twice each gap
    cell's distance to a raised column, gaps under r) and the emergent share are in the returned record. `where` ((ny,
    nx), the shown site) is what h_c and the gaps are measured over: the buffer beyond the survey is ground."""
    from scipy import ndimage
    base = scene.origin[2] + (np.zeros(scene.z0.shape) if scene.terrain is None else np.asarray(scene.terrain, float))
    top = np.maximum(crown_top(scene) - base, 0.0)
    dx = float(scene.grid.dx)
    site = np.ones(top.shape, bool) if where is None else np.asarray(where, bool)
    raised = (top >= RAISED_M) & site
    if not raised.any():
        return top, {"canopy_height_m": 0.0, "closing_radius_m": 0.0, "sigma_m": 0.0}
    hc = float(np.median(top[raised]))
    r = max(1, int(round(0.5 * hc / dx)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disc = (xx * xx + yy * yy) <= r * r
    big = max(1, int(round(hc / dx)))
    local = ndimage.median_filter(top, size=2 * big + 1, mode="nearest")
    cap = local + r * dx
    emergent = top > cap
    capped = np.where(emergent, cap, top)
    gap = top < 0.5 * hc
    width = 2.0 * ndimage.distance_transform_edt(gap & site) * dx   # bounded by the site's edge
    w = width[gap & site]
    closed = ndimage.grey_closing(capped, footprint=disc, mode="nearest")
    sigma = max(1.0, 0.5 * r)
    smooth = ndimage.gaussian_filter(ndimage.grey_dilation(closed, footprint=disc, mode="nearest"), sigma, mode="nearest")
    env = np.maximum(smooth, capped)
    return env, {"canopy_height_m": round(hc, 2), "closing_radius_m": round(r * dx, 2), "sigma_m": round(sigma * dx, 2),
                 "dilation_radius_m": round(r * dx, 2), "emergent_above_local_m": round(r * dx, 2),
                 "emergent_share": round(float(emergent[site].mean()), 4),
                 "gap_width_p50_p90_m": [round(float(np.percentile(w, q)), 1) for q in (50, 90)] if w.size else None,
                 "gap_share": round(float(gap[site].mean()), 4)}


def over_top(field: np.ndarray, scene: Scene, cfg: ViewConfig, display_radius: float,
             clearance: float = OVER_TOP_M, surface: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """`field` at `clearance` over the surface the wind passes over (`envelope`, given as `surface` or made here):
    ((c, n, n) values, trilinear over the fluid centres, and (n, n) that point's height above the bare earth), rows as
    `levels` gives them, NaN beyond `display_radius` and where a structure taller than the envelope stands (the point
    inside its solid): such a structure rises above the ribbons and hides them behind it."""
    x, y = cfg.centres()
    env = envelope(scene)[0] if surface is None else surface
    z = ground(scene, cfg) + under(scene, cfg, env) + clearance
    v, _ = trilinear_fluid(field, scene, x.ravel(), y.ravel(), z.ravel())
    v = v.reshape((-1,) + x.shape)
    beyond = ~cfg.shown(x, y, display_radius)
    inside = under(scene, cfg, walkable(scene)) > z           # within a solid taller than the envelope
    v[:, beyond | inside] = np.nan
    return v, np.where(beyond | inside, np.nan, z - ground(scene, cfg))


def under(scene: Scene, cfg: ViewConfig, a: np.ndarray) -> np.ndarray:
    """(n, n) the per-column (ny, nx) `a` under each viewer cell centre, rows running north."""
    x, y = cfg.centres()
    g = scene.grid
    i = np.clip(np.floor((x - scene.origin[0]) / g.dx).astype(int), 0, g.nx - 1)
    j = np.clip(np.floor((y - scene.origin[1]) / g.dx).astype(int), 0, g.ny - 1)
    return a[j, i]


def surface(scene: Scene, cfg: ViewConfig) -> np.ndarray:
    """(n, n) scene-frame elevation [m] of the walkable top under each viewer cell, rows running north."""
    return under(scene, cfg, walkable(scene))


def roofs(scene: Scene, cfg: ViewConfig) -> np.ndarray:
    """(n, n) True under each viewer cell standing on a building."""
    return under(scene, cfg, np.zeros(scene.z0.shape, bool) if scene.roof is None else scene.roof)


def resolution(scene: Scene, cfg: ViewConfig, display_radius: float,
               heights: Sequence[float]) -> Dict[str, object]:
    """How far above the walkable top the solve first holds a value of its own, over the display
    disc, and how many cells each candidate height leaves below that."""
    x, y = cfg.centres()
    above, dz = (under(scene, cfg, a) for a in first_fluid(scene))
    roof = roofs(scene, cfg)
    inside = cfg.shown(x, y, display_radius)
    # A disc with no roof (a beach, a park) has no roof cells to take percentiles of (WS22's F15,
    # Miami Beach, 2026-09-13: `levels` died here on np.percentile of an empty array).
    pct = lambda a: ({f"p{q}": float(np.percentile(a, q)) for q in (50, 90, 99)}  # noqa: E731
                     | {"max": float(a.max())}) if a.size else {}
    out: Dict[str, object] = {}
    for name, m in (("roof", inside & roof), ("ground", inside & ~roof)):
        out[name] = {"cells": int(m.sum()), "first_centre_above_top_m": pct(above[m]),
                     "first_cell_dz_m": pct(dz[m])}
    out["unresolved_cells_by_height_m"] = {
        f"{h:g}": {"roof": int((above > h)[inside & roof].sum()),
                   "ground": int((above > h)[inside & ~roof].sum())} for h in heights}
    beside = {h: under(scene, cfg, beside_wall(scene, h)) & inside for h in heights}
    out["beside_wall_cells_by_height_m"] = {
        f"{h:g}": {"roof": int((b & roof).sum()), "ground": int((b & ~roof).sum())} for h, b in beside.items()}
    return out


def beside_wall(scene: Scene, h: float) -> np.ndarray:
    """(ny, nx) True where an adjacent column's walkable top stands more than `h` above this one's:
    a point `h` above it is one horizontal cell from a wall the grid cannot resolve across."""
    t = walkable(scene)
    p = np.pad(t, 1, mode="edge")
    return np.max([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]], axis=0) > t + h
