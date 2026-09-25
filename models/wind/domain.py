"""The grid, the scene on it, and the ways a scene is built: synthetic shapes or bundle rasters.

Axes are (z, y, x) everywhere: x east, y north, z up, cell centres at (i + 0.5) dx and at the
midpoints of the stretched levels. Row 0 of every (ny, nx) array is the southern edge.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np

from physics import CLASSES, GROUND_CLASS
import sites
from sites import SiteConfig

if TYPE_CHECKING:
    from params import CellParams


@dataclass(frozen=True)
class Grid:
    """A horizontally uniform, vertically stretched cell grid.

    Attributes:
        dx: Horizontal cell size [m].
        nx: Cells east-west.
        ny: Cells north-south.
        zf: Level face heights [m], nz + 1 values starting at 0.
    """

    dx: float
    nx: int
    ny: int
    zf: np.ndarray

    @classmethod
    def stretched(cls, dx: float, nx: int, ny: int, nz: int, dz0: float,
                  ratio: float) -> "Grid":
        """Levels growing geometrically from `dz0` at the surface by `ratio` per level."""
        dz = dz0 * ratio ** np.arange(nz)
        return cls(dx=dx, nx=nx, ny=ny, zf=np.concatenate([[0.0], np.cumsum(dz)]))

    @classmethod
    def banded(cls, dx: float, nx: int, ny: int, dz: float, band: float, ratio: float, top: float,
               multiple: int = 8) -> "Grid":
        """Levels of `dz` from the floor to `band`, then growing by `ratio` per level until the top
        clears `top`, the level count rounded up to a multiple of `multiple` for the multigrid."""
        steps = [dz] * int(np.ceil(band / dz - 1e-9))
        while sum(steps) < top or len(steps) % multiple:
            steps.append(steps[-1] * ratio)
        return cls(dx=dx, nx=nx, ny=ny, zf=np.concatenate([[0.0], np.cumsum(steps)]))

    @classmethod
    def uniform(cls, dx: float, nx: int, ny: int, nz: int) -> "Grid":
        """Cubic cells."""
        return cls(dx=dx, nx=nx, ny=ny, zf=dx * np.arange(nz + 1, dtype=float))

    @property
    def nz(self) -> int:
        return len(self.zf) - 1

    @property
    def zc(self) -> np.ndarray:
        """Cell-centre heights [m]."""
        return 0.5 * (self.zf[:-1] + self.zf[1:])

    @property
    def dz(self) -> np.ndarray:
        """Level thicknesses [m]."""
        return np.diff(self.zf)

    @property
    def top(self) -> float:
        return float(self.zf[-1])

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.nz, self.ny, self.nx

    @property
    def cells(self) -> int:
        return self.nz * self.ny * self.nx

    @property
    def xc(self) -> np.ndarray:
        return (np.arange(self.nx) + 0.5) * self.dx

    @property
    def yc(self) -> np.ndarray:
        return (np.arange(self.ny) + 0.5) * self.dx


@dataclass
class Scene:
    """Static per-cell fields the solver reads.

    Attributes:
        grid: The grid.
        solid: True where flow is blocked.
        sink: Volumetric drag coefficient cd * a [1/m], zero outside canopy.
        z0: Roughness length [m] of each column's solid surfaces, including the ground.
        label: What this scene is.
        origin: Scene metres (x, y, z) of the south-west corner of cell (0, 0) at the floor.
        terrain: (ny, nx) bare-earth height above the floor [m]; None is flat.
        top: (ny, nx) height above the floor [m] of the walkable top: a solid column's top, bare
            earth under an open one, canopy included; None is the terrain.
        roof: (ny, nx) True where the column is a building; None is none.
    """

    grid: Grid
    solid: np.ndarray
    sink: np.ndarray
    z0: np.ndarray
    label: str
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    terrain: Optional[np.ndarray] = None
    top: Optional[np.ndarray] = None
    roof: Optional[np.ndarray] = None
    closed_cells: int = 0

    def summary(self) -> Dict[str, float]:
        return {
            "label": self.label, "cells": self.grid.cells, "dx_m": self.grid.dx,
            "top_m": self.grid.top, "solid_fraction": float(self.solid.mean()),
            "canopy_fraction": float((self.sink > 0).mean()),
        }


def _empty(grid: Grid, z0: float, label: str) -> Scene:
    return Scene(grid=grid, solid=np.zeros(grid.shape, bool), sink=np.zeros(grid.shape),
                 z0=np.full((grid.ny, grid.nx), z0), label=label,
                 origin=(-grid.nx * grid.dx / 2, -grid.ny * grid.dx / 2, 0.0))


def _footprint(grid: Grid, centre: Optional[Tuple[float, float]], half: float,
               round_: bool = False, subsamples: int = 1) -> np.ndarray:
    """(ny, nx) coverage of columns within `half` of `centre`, square or round.

    With `subsamples` of 1 this is a boolean mask at cell centres; more give the fraction of
    each cell's area covered, sampled on a `subsamples` x `subsamples` sub-grid.
    """
    cx, cy = centre if centre is not None else (grid.nx * grid.dx / 2, grid.ny * grid.dx / 2)
    offsets = ((np.arange(subsamples) + 0.5) / subsamples - 0.5) * grid.dx
    cover = np.zeros((grid.ny, grid.nx))
    for oy in offsets:
        for ox in offsets:
            x, y = np.meshgrid(grid.xc + ox - cx, grid.yc + oy - cy)
            cover += np.hypot(x, y) <= half if round_ else (np.abs(x) <= half) & (np.abs(y) <= half)
    return cover > 0 if subsamples == 1 else cover / subsamples ** 2


def flat(grid: Grid, z0: float = 0.03) -> Scene:
    """Open ground of uniform roughness."""
    return _empty(grid, z0, "flat")


def box(grid: Grid, width: float, height: float, centre: Optional[Tuple[float, float]] = None,
        z0: float = 0.03, z0_wall: float = 0.03) -> Scene:
    """A solid square tower of side `width` and height `height` standing on the ground."""
    scene = _empty(grid, z0, f"box {width:g} x {height:g} m")
    fp = _footprint(grid, centre, width / 2)
    scene.solid[:] = fp[None] & (grid.zc < height)[:, None, None]
    scene.z0[fp] = z0_wall
    scene.top, scene.roof = np.where(fp, height, 0.0), fp
    return scene


def cube(grid: Grid, size: float, centre: Optional[Tuple[float, float]] = None,
         z0: float = 0.03, z0_wall: float = 0.03) -> Scene:
    """A solid cube of edge `size` standing on the ground."""
    return box(grid, size, size, centre, z0, z0_wall)


def cylinder(grid: Grid, radius: float, height: float,
             centre: Optional[Tuple[float, float]] = None, z0: float = 0.03) -> Scene:
    """A solid round tower."""
    scene = _empty(grid, z0, f"cylinder r={radius:g} h={height:g} m")
    fp = _footprint(grid, centre, radius, round_=True)
    scene.solid[:] = fp[None] & (grid.zc < height)[:, None, None]
    return scene


def porous_block(grid: Grid, size: float, lai: float, cd: float = 0.2,
                 centre: Optional[Tuple[float, float]] = None, z0: float = 0.03) -> Scene:
    """A canopy cube of edge `size` with uniform leaf area density LAI / size."""
    scene = _empty(grid, z0, f"canopy {size:g} m, LAI {lai:g}")
    fp = _footprint(grid, centre, size / 2)
    inside = fp[None] & (grid.zc < size)[:, None, None]
    scene.sink[inside] = cd * lai / size
    return scene


def ridge(grid: Grid, height: float, half_width: float, z0: float = 0.03) -> Scene:
    """A cosine ridge running north-south across the whole domain."""
    scene = _empty(grid, z0, f"ridge h={height:g} m")
    x = grid.xc - grid.nx * grid.dx / 2
    profile = np.where(np.abs(x) < half_width,
                       0.5 * height * (1 + np.cos(np.pi * x / half_width)), 0.0)
    scene.solid[:] = grid.zc[:, None, None] < profile[None, None, :]
    return scene


def _fill_nodata(a: np.ndarray) -> np.ndarray:
    """Replace NaN with the nearest finite value."""
    from scipy.ndimage import distance_transform_edt

    if not np.isnan(a).any():
        return a
    idx = distance_transform_edt(np.isnan(a), return_distances=False, return_indices=True)
    return a[tuple(idx)]


POCKET_ACROSS = 3
"""Cells across the smallest open square an enclosed pocket must hold for the grid to carry flow in it."""


def pockets(fluid: np.ndarray, across: int = POCKET_ACROSS) -> np.ndarray:
    """(ny, nx) True in each 4-connected open component of `fluid` that touches no edge of the array and holds no
    `across` x `across` open square: a shaft inside a structure, sealed on every side, too narrow to carry flow."""
    from scipy import ndimage

    lab, n = ndimage.label(fluid)
    if n == 0:
        return np.zeros(fluid.shape, bool)
    edge = np.zeros(fluid.shape, bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    keep = np.zeros(n + 1, bool)
    keep[0] = True
    keep[np.unique(lab[edge])] = True
    keep[np.unique(lab[ndimage.binary_opening(fluid, structure=np.ones((across, across), bool))])] = True
    return ~keep[lab]


def close_pockets(scene: Scene) -> int:
    """Make solid every cell of each layer's `pockets`, raising the top of each column that takes one to the face
    above its highest; returns the cells closed. Solid layers shrink upward, so a pocket's column below it is solid
    or a pocket too, and every column stays solid from the floor up."""
    closed = np.stack([pockets(~layer) for layer in scene.solid])
    if not closed.any():
        return 0
    scene.solid |= closed
    scene.sink[closed] = 0.0
    hit = closed.any(axis=0)
    k = scene.grid.nz - 1 - np.argmax(closed[::-1], axis=0)
    face = scene.grid.zf[np.minimum(k + 1, scene.grid.nz)]
    base = scene.terrain if scene.top is None else scene.top
    if base is not None:
        scene.top = np.where(hit, np.maximum(base, face), base)
    return int(closed.sum())


def voxelize(columns: "CellParams", dtm: np.ndarray, dsm: np.ndarray, grid: Grid,
             label: str, floor: Optional[float] = None) -> Scene:
    """Solids, drag and roughness from per-column parameters and south-up height rasters.

    Args:
        columns: Per-column class parameters on `grid`.
        dtm: Bare-earth elevation [m], NaN where unknown.
        dsm: Surface elevation [m], NaN where unknown.
        grid: Target grid.
        label: Scene label.
        floor: Elevation [m] of grid z = 0; the lowest terrain when not given. Terrain below it
            is clamped up to it.

    Returns:
        Solid where a cell centre lies below the terrain or inside a solid column below its
        top, or in an enclosed pocket too narrow to carry flow (`close_pockets`, counted in
        `scene.closed_cells`); drag cd * LAI / height, or cd * blockage / dx without foliage,
        inside an open column; roughness from the column's class.
    """
    assert dtm.shape == dsm.shape == (grid.ny, grid.nx), (
        f"rasters {dtm.shape} do not match the grid {(grid.ny, grid.nx)}")
    dtm, dsm = _fill_nodata(dtm.astype(float)), _fill_nodata(dsm.astype(float))
    floor = float(np.nanmin(dtm)) if floor is None else floor
    zt = np.clip(dtm - floor, 0.0, None)
    zo = np.maximum(dsm - floor, zt)
    zc = grid.zc[:, None, None]

    scene = _empty(grid, CLASSES[GROUND_CLASS].z0_m, label)
    heights = zo - zt
    density = np.where(columns.lai > 0, columns.lai / np.maximum(heights, 1e-9),
                       columns.blockage / grid.dx)
    sink_col = np.where((heights > 0) & ~columns.solid, columns.cd * density, 0.0)
    scene.z0[:] = np.where(columns.solid, columns.z0, CLASSES[GROUND_CLASS].z0_m)
    scene.solid[:] = (zc < zt[None]) | (columns.solid[None] & (zc < zo[None]))
    in_column = (zc >= zt[None]) & (zc < zo[None]) & ~scene.solid
    scene.sink[:] = np.where(in_column, sink_col[None], 0.0)
    scene.origin = (scene.origin[0], scene.origin[1], floor)
    scene.terrain = zt
    scene.top = np.where(columns.solid, zo, zt)
    scene.roof = np.asarray(columns.building, bool)
    scene.closed_cells = close_pockets(scene)
    return scene


def from_rasters(dtm: np.ndarray, dsm: np.ndarray, classes: np.ndarray,
                 legend: Dict[int, str], grid: Grid, label: str = "rasters") -> Scene:
    """Voxelise from an integer class raster and its legend, on the built-in class table."""
    import params

    assert classes.shape == (grid.ny, grid.nx), (
        f"rasters {classes.shape} do not match the grid {(grid.ny, grid.nx)}")
    return voxelize(params.from_legend(classes, legend, grid), dtm, dsm, grid, label)


def radius(grid: Grid, origin: Tuple[float, float]) -> np.ndarray:
    """(ny, nx) horizontal distance [m] of every column centre from the scene origin."""
    return np.hypot(*np.meshgrid(origin[0] + grid.xc, origin[1] + grid.yc))


def in_box(grid: Grid, origin: Tuple[float, float], box: Tuple[float, float, float, float]) -> np.ndarray:
    """(ny, nx) True where a column centre lies in the box (x0, y0, x1, y1) [scene m]."""
    X, Y = np.meshgrid(origin[0] + grid.xc, origin[1] + grid.yc)
    return (X >= box[0]) & (X <= box[2]) & (Y >= box[1]) & (Y <= box[3])


def flat_declared(flat_ok: Optional[bool] = None) -> bool:
    """Whether this run may stand on flat ground where a bundle holds no terrain: `flat_ok` when given, else
    WIND_FLAT_TERRAIN=1 (`--flat-terrain`). A verification case only; a customer's site never declares it."""
    return bool(flat_ok) if flat_ok is not None else os.environ.get("WIND_FLAT_TERRAIN") == "1"


def from_bundle(site: SiteConfig, dx: float, bundle: Path, data_radius: Optional[float] = None,
                band: Optional[Tuple[float, float]] = None, cells: Optional[int] = None,
                flat_ok: Optional[bool] = None,
                ground_band: Optional[Tuple[float, float, float]] = None) -> Tuple[Scene, Dict[str, object]]:
    """The parcel scene from a bundle directory at `dx`, and the parameter receipt.

    The grid covers `site.fetch_radius_m`. Columns come from `bundle/semantics` and heights
    from the surface rasters. A bundle with no surface/ is solved over flat ground only when that
    is declared (`flat_ok`, or WIND_FLAT_TERRAIN=1 from `--flat-terrain`: a verification case, never
    a customer's site), and the receipt says so; otherwise the run fails (params.TerrainMissing). Every column beyond the fetch radius is treated as outside the bundle, so a site
    whose buffer is overridden to 0 sees exactly what a bundle ending at the display disc would
    give it. The terrain is bare earth: the surface raster's lowest crossing is the roof under a
    building and, where the mesh holds a dense crown down to the ground, the crown top under a
    tree, so under every building and every drag column it is the nearest ground column's. Beyond
    the data the same ground continues and nothing stands on it. The floor is taken from the terrain
    raster itself, so every buffer over one bundle shares one vertical grid. `data_radius` ends the
    data short of the grid, which separates what a wider grid does from what the data in it does.
    `band`, (dz, height), replaces the levels up to `height` above the floor with cells of `dz`,
    stretching above as before to at least the same top. `cells` widens the square grid with open
    ground, which moves its prescribed sides away from the same data. `ground_band`, (dz, above, cap),
    is `band` with its height the highest terrain in the fetch disc plus `above`, at most `cap` above
    the floor: every level over every ground and every structure it can stand on then lies in cells of `dz`.
    Terrain below the floor is the mesh's closure skirt or a stray low crossing, not ground: left
    in, each such column is a shaft one cell wide down to the floor. They are treated as unknown
    and filled.
    """
    import params

    doc = sites.aoi_document(site, bundle)
    boxes = sites.box_of(site, bundle) if data_radius is None else None
    box = boxes["fetch"] if boxes else None
    if box:
        # A site that follows its ordered polygon: nx x ny over its fetch box, centred on it. A tuple `cells` is a
        # coarse solve over the fine grid's own extent (`cli.coarse_cells`); an int (--width-cells) is a disc square.
        nx, ny = cells if isinstance(cells, tuple) else sites.box_cells(box, dx)
        origin = ((box[0] + box[2]) / 2 - nx * dx / 2, (box[1] + box[3]) / 2 - ny * dx / 2)
    else:
        nx = ny = (cells[0] if isinstance(cells, tuple) else cells) or site.cells_across(dx)
        origin = (-nx * dx / 2, -ny * dx / 2)
    grid = Grid.stretched(dx, nx, ny, site.levels(dx), dx, site.stretch)
    if band is not None:
        grid = Grid.banded(dx, nx, ny, band[0], band[1], site.stretch, grid.top)
    outside = (~in_box(grid, origin, box) if box else
               radius(grid, origin) > (site.fetch_radius_m if data_radius is None else data_radius))
    columns = params.beyond(params.from_bundle(bundle, grid, origin, site.anchor_utm[:2]), outside)
    heights = params.surface_rasters(bundle, dx, grid, origin)       # raises TerrainMissing for an unreadable surface/
    if heights is None and not flat_declared(flat_ok):
        raise params.TerrainMissing(f"{bundle}: no surface/ and no DTM or DSM, and flat terrain was not declared "
                                    f"(--flat-terrain)")
    over = np.zeros(outside.shape, bool)
    if heights is not None:
        columns, over = params.canopy_over_surfaces(columns, heights[0], heights[1], ~outside)
    receipt = columns.receipt(dx)
    receipt["surface_columns_to_canopy"] = int(over.sum())
    if box:
        receipt["box_scene_m"] = doc["aoi"]["box_scene_m"]
    receipt.update(display_radius_m=site.display_radius_m, fetch_radius_m=site.fetch_radius_m,
                   buffer_m=site.buffer_m, bundle_buffer_m=doc["aoi"]["buffer_m"],
                   columns_beyond_fetch_radius=int(outside.sum()),
                   grid_convergence_deg=doc["scene_frame"]["grid_convergence_deg"])
    if heights is None:
        receipt["surface"] = "flat: declared (--flat-terrain); the bundle holds no dtm/dsm rasters"
        scene = voxelize(columns, np.zeros((ny, nx)), np.zeros((ny, nx)), grid, f"{site.name} {dx:g} m")
    else:
        reach = doc["aoi"]["radius_m"] + doc["aoi"]["buffer_m"]
        floor = params.terrain_floor(bundle, dx, reach, box)
        pits = heights[0] < floor
        bare = ~columns.building & columns.solid & ~outside & ~pits
        ground = _fill_nodata(np.where(bare, heights[0], np.nan))
        dsm = np.where(outside, ground, np.where(pits, np.nan, heights[1]))
        receipt.update(surface=str(params.nearest_raster(bundle, "dtm", dx)), terrain_floor_scene_m=floor,
                       below_floor_columns_filled=int(pits.sum()),
                       terrain_filled_from_ground_columns=int((~bare & ~outside).sum()),
                       beyond_data="open ground at the height of the nearest ground column inside")
        dtm = ground
        if ground_band is not None:
            dz, above, cap = ground_band
            height = min(float(cap), float(np.ceil(np.nanmax(np.maximum(ground[~outside], floor)) - floor + above)))
            grid = Grid.banded(dx, nx, ny, dz, height, site.stretch, grid.top)
            receipt["ground_band_dz_height_m"] = [float(dz), height]
        scene = voxelize(columns, dtm, dsm, grid, f"{site.name} {dx:g} m", floor)
        receipt["enclosed_pocket_cells_closed"] = scene.closed_cells
    scene.origin = (origin[0], origin[1], scene.origin[2])
    return scene, receipt
