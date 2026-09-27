"""Per-column aerodynamic parameters, from the semantics bundle or the built-in class table.

The bundle's `semantics/class_top_<res>.tif` names the class seen from above in every pixel as
a uint8 row id into `semantics/parameters.json`; 46 is unobserved and 255 nodata. Wind reads
z0_m, cd, LAI and closure from each row and uses `volume_category` and `closed` to tell solid
columns from drag columns. Cells the bundle does not describe take `FALLBACK_CLASS` and are
counted in the receipt.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from domain import Grid
from physics import CLASSES, GROUND_CLASS

FALLBACK_CLASS = GROUND_CLASS
UNOBSERVED = 46
NODATA = 255


@dataclass(frozen=True)
class ClassTable:
    """`parameters.json` as arrays indexed by row id.

    Attributes:
        names: Class id per row.
        z0: Roughness length [m] per row.
        cd: Drag coefficient per row.
        lai: Leaf area index [m^2/m^2] per row.
        blockage: Foliage-free area fraction an open structure presents, per row.
        solid: True where a row blocks flow.
        building: True where a row is a building, whose surface raster's lowest crossing is its roof.
    """

    names: Tuple[str, ...]
    z0: np.ndarray
    cd: np.ndarray
    lai: np.ndarray
    blockage: np.ndarray
    solid: np.ndarray
    building: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "ClassTable":
        """Read `parameters.json`; solid rows are buildings and closed classes."""
        rows = json.loads(Path(path).read_text())["rows"]
        assert [r["id"] for r in rows] == list(range(len(rows))), "row ids are not 0..n-1"
        p = lambda k: np.array([r["params"][k] for r in rows], float)  # noqa: E731
        return cls(names=tuple(r["class_id"] for r in rows), z0=p("z0_m"), cd=p("cd"),
                   lai=p("LAI"), blockage=p("closure"),
                   solid=np.array([r["volume_category"] == "building" or r["closed"] == "yes"
                                   for r in rows]),
                   building=np.array([r["volume_category"] == "building" for r in rows]))

    @classmethod
    def builtin(cls) -> "ClassTable":
        """The class table in `physics`, blockage taken as one minus porosity."""
        names = tuple(CLASSES)
        return cls(names=names, z0=np.array([CLASSES[n].z0_m for n in names]),
                   cd=np.array([CLASSES[n].cd for n in names]),
                   lai=np.array([CLASSES[n].lai for n in names]),
                   blockage=np.array([1.0 - CLASSES[n].poro for n in names]),
                   solid=np.array([CLASSES[n].solid for n in names]),
                   building=np.array([CLASSES[n].solid for n in names]))

    @property
    def fallback(self) -> int:
        return self.names.index(FALLBACK_CLASS)


@dataclass
class CellParams:
    """Per-column parameters on the solver grid, (ny, nx) south-up.

    Attributes:
        row: Class row of every column after fallback.
        table: The rows' parameters.
        unobserved: Columns the bundle marked unobserved.
        nodata: Columns outside the bundle's coverage.
        source: Where the rows came from.
    """

    row: np.ndarray
    table: ClassTable
    unobserved: np.ndarray
    nodata: np.ndarray
    source: str

    @property
    def z0(self) -> np.ndarray:
        return self.table.z0[self.row]

    @property
    def cd(self) -> np.ndarray:
        return self.table.cd[self.row]

    @property
    def lai(self) -> np.ndarray:
        return self.table.lai[self.row]

    @property
    def blockage(self) -> np.ndarray:
        return self.table.blockage[self.row]

    @property
    def solid(self) -> np.ndarray:
        return self.table.solid[self.row]

    @property
    def building(self) -> np.ndarray:
        return self.table.building[self.row]

    @property
    def observed(self) -> np.ndarray:
        return ~(self.unobserved | self.nodata)

    def receipt(self, dx: float) -> Dict[str, object]:
        """What the run saw: coverage, the fallback counts and the area-weighted roughness."""
        cells = self.row.size
        counts = np.bincount(self.row[self.observed], minlength=len(self.table.names))
        return {
            "source": self.source, "cells": int(cells), "cell_m": dx,
            "unobserved_cells": int(self.unobserved.sum()), "nodata_cells": int(self.nodata.sum()),
            "fallback_class": FALLBACK_CLASS,
            "observed_area_m2": float(self.observed.sum() * dx * dx),
            "mean_z0_observed_m": float(self.z0[self.observed].mean()) if self.observed.any() else None,
            "class_area_m2": {n: float(c * dx * dx) for n, c in zip(self.table.names, counts) if c},
        }


def from_rows(rows: np.ndarray, table: ClassTable, source: str) -> CellParams:
    """Resolve a row raster against a table, sending unobserved and nodata to the fallback."""
    unobserved, nodata = rows == UNOBSERVED, rows == NODATA
    row = np.where(unobserved | nodata | (rows >= len(table.names)), table.fallback, rows)
    return CellParams(row=row.astype(int), table=table, unobserved=unobserved, nodata=nodata,
                      source=source)


def from_legend(classes: np.ndarray, legend: Dict[int, str], grid: Grid) -> CellParams:
    """Columns from an integer class raster and its legend, on the built-in table."""
    table = ClassTable.builtin()
    rows = np.full(classes.shape, NODATA, np.uint8)
    for code, name in legend.items():
        rows[classes == code] = table.names.index(name)
    return from_rows(rows, table, "builtin")


def resample_nearest(raster: np.ndarray, transform: Tuple[float, ...], grid: Grid,
                     origin: Tuple[float, float], anchor: Tuple[float, float],
                     nodata: float = NODATA) -> np.ndarray:
    """Nearest-neighbour sample of a north-down raster onto the solver grid, south-up.

    Args:
        raster: (rows, cols), row 0 at the north edge.
        transform: GDAL geotransform (x0, dx, 0, y1, 0, -dy) in the raster's projection.
        grid: Solver grid.
        origin: Scene metres (x, y) of the south-west corner of solver cell (0, 0).
        anchor: Projected metres (E, N) of the scene origin.
        nodata: Value for solver cells the raster does not cover.

    Returns:
        (ny, nx) values, row 0 at the south edge.
    """
    x0, dx, _, y1, _, dy = transform
    e = origin[0] + anchor[0] + grid.xc
    n = origin[1] + anchor[1] + grid.yc
    col = np.floor((e - x0) / dx).astype(int)
    row = np.floor((y1 - n) / -dy).astype(int)
    inside = (col >= 0) & (col < raster.shape[1])
    inside = inside[None, :] & ((row >= 0) & (row < raster.shape[0]))[:, None]
    out = np.full((grid.ny, grid.nx), nodata, raster.dtype)
    rr, cc = np.meshgrid(np.clip(row, 0, raster.shape[0] - 1), np.clip(col, 0, raster.shape[1] - 1),
                         indexing="ij")
    out[inside] = raster[rr, cc][inside]
    return out


def read_raster(path: Path) -> Tuple[np.ndarray, Tuple[float, ...], float]:
    """(array, geotransform, nodata) of a single-band GeoTIFF."""
    import rasterio

    with rasterio.open(path) as src:
        return src.read(1), tuple(src.transform.to_gdal()), src.nodata


class TerrainMissing(RuntimeError):
    """A bundle that holds terrain (a `surface/` directory) yields no DTM or DSM the solver can read. The wind is
    never solved over flat ground in its place: on 2026-09-13 a bundle whose `surface/` was a symlink was read as
    holding none (pathlib's `**` does not enter a symlinked directory before Python 3.13), and the wind came out
    0.90 m/s rms off the deployed day over flat ground, with only a note in its sidecar and exit code 0."""


def _find(root: Path, pattern: str) -> list:
    """Files matching `pattern` anywhere under `root`, symlinked directories followed (each real directory once)."""
    import fnmatch
    import os

    out, seen = [], set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen:
            dirnames[:] = []
            continue
        seen.add(real)
        out += [Path(dirpath) / f for f in fnmatch.filter(filenames, pattern)]
    return sorted(out)


def nearest_raster(root: Path, prefix: str, dx: float) -> Optional[Path]:
    """The `<prefix>_<res>m.tif` under `root` whose resolution is nearest `dx`, or None. Symlinked directories are
    followed (`_find`): a symlinked `surface/` once read as empty.

    A raster grid and a solver grid are independent; `resample_nearest` bridges them, so a 1 m
    solve reads the 0.2 m rasters rather than refusing to run.
    """
    files = _find(Path(root), f"{prefix}_*m.tif")
    res = lambda p: float(p.stem.split("_")[-1][:-1].replace("p", "."))  # noqa: E731
    return min(files, key=lambda p: abs(res(p) - dx)) if files else None


def class_raster(bundle: Path, dx: float) -> Path:
    """The `class_top` raster whose resolution is nearest the solver's."""
    path = nearest_raster(bundle / "semantics", "class_top", dx)
    assert path is not None, f"no class_top raster under {bundle / 'semantics'}"
    return path


def from_bundle(bundle: Path, grid: Grid, origin: Tuple[float, float],
                anchor: Tuple[float, float]) -> CellParams:
    """Columns from the bundle's top-of-column class raster and parameter table."""
    table = ClassTable.load(bundle / "semantics" / "parameters.json")
    path = class_raster(bundle, grid.dx)
    raster, transform, nodata = read_raster(path)
    assert nodata == NODATA, f"{path} nodata is {nodata}, not {NODATA}"
    rows = resample_nearest(raster, transform, grid, origin, anchor)
    return from_rows(rows, table, str(path))


SURFACE_CLASSES = frozenset({"asphalt_pavement", "brick_paving", "concrete_pavement", "dirt_soil", "gravel_surface",
                             "pavement_generic", "pavement_marking", "stone_paving", "synthetic_sports_surface",
                             "planting_bed", "turf_grass", "open_water"})
"""Classes that are a ground surface, not a volume: nothing of theirs stands above the terrain."""

OVERHANG_M = 1.0
"""DSM above the terrain over a surface class beyond which what stands there is a crown, not the surface."""


ROOF_EDGE_M = 1.0
"""DSM above the terrain within which a surface-class column beside a building stands at that roof: its edge, which the
class raster places one cell off the DSM's."""


def roof_edge(building: np.ndarray, hag: np.ndarray) -> np.ndarray:
    """(ny, nx) True where one of the 8 neighbouring columns is a building whose height above its terrain is within
    ROOF_EDGE_M of this column's."""
    b, h = np.pad(building, 1), np.pad(hag, 1, constant_values=np.nan)
    ny, nx = hag.shape
    out = np.zeros(hag.shape, bool)
    with np.errstate(invalid="ignore"):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    s = (slice(1 + dy, 1 + dy + ny), slice(1 + dx, 1 + dx + nx))
                    out |= b[s] & (np.abs(h[s] - hag) <= ROOF_EDGE_M)
    return out


def canopy_over_surfaces(columns: CellParams, dtm: np.ndarray, dsm: np.ndarray,
                         where: np.ndarray) -> Tuple[CellParams, np.ndarray]:
    """The columns with each surface-class column in `where` whose DSM stands more than OVERHANG_M above its terrain
    turned to canopy, and that mask. Solid to its DSM, such a column made a tree over a lawn a tower. A roof's edge
    (`roof_edge`) stays solid: made porous, it let wind through the building's rim."""
    names = columns.table.names
    if "tree_canopy" not in names:
        return columns, np.zeros(columns.row.shape, bool)
    surface = np.array([n in SURFACE_CLASSES for n in names])
    hag = np.nan_to_num(dsm - dtm, nan=0.0)
    is_surface = surface[columns.row]
    roof = np.asarray(columns.building, bool) & ~is_surface
    with np.errstate(invalid="ignore"):
        over = is_surface & (hag > OVERHANG_M) & where & ~roof_edge(roof, hag)
    row = np.where(over, names.index("tree_canopy"), columns.row).astype(columns.row.dtype)
    return CellParams(row=row, table=columns.table, unobserved=columns.unobserved, nodata=columns.nodata,
                      source=columns.source), over


def beyond(columns: CellParams, outside: np.ndarray) -> CellParams:
    """The columns with every `outside` column turned to nodata, as if the bundle ended there."""
    return CellParams(row=np.where(outside, columns.table.fallback, columns.row), table=columns.table,
                      unobserved=columns.unobserved & ~outside, nodata=columns.nodata | outside,
                      source=columns.source)


def surface_rasters(bundle: Path, dx: float, grid: Grid,
                    origin: Tuple[float, float]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(dtm, dsm) on the solver grid from the scene-frame surface rasters nearest `dx` in
    resolution; None only for a bundle with no `surface/` at all (terrain it never had).

    Raises:
        TerrainMissing: the bundle has a `surface/` and it yields no DTM or no DSM.
    """
    paths = [nearest_raster(bundle, kind, dx) for kind in ("dtm", "dsm")]
    if not all(paths):
        surface = Path(bundle) / "surface"
        if surface.exists() or surface.is_symlink():
            raise TerrainMissing(f"{bundle}: its surface/ ({surface.resolve() if surface.exists() else 'a broken link'}) "
                                 f"yields no {' or '.join(k for k, p in zip(('dtm', 'dsm'), paths) if not p)} raster")
        return None
    out = []
    for path in paths:
        raster, transform, nodata = read_raster(path)
        a = resample_nearest(raster.astype(float), transform, grid, origin, (0.0, 0.0), np.nan)
        out.append(np.where(a == nodata, np.nan, a) if nodata is not None else a)
    return out[0], out[1]


FLOOR_PERCENTILE = 0.1
"""Percentile of the bundle's terrain that becomes grid z = 0.

Not the minimum: the watertight mesh closes the disc with a skirt below the ground, and a
handful of stray low crossings survive inside it. Cells below the floor are clamped up to it."""

FLOOR_RIM_M = 3.0
"""Terrain within this distance of the bundle's disc edge is the skirt and never sets the floor."""


def terrain_floor(bundle: Path, dx: float, reach_m: float, box: Optional[Tuple[float, ...]] = None) -> float:
    """Elevation [m] that becomes grid z = 0, from the terrain raster itself.

    Read from the raster rather than from the solver grid, so that two grids of different
    extent over one bundle, the arms of a buffer comparison, share one vertical discretisation.

    Args:
        bundle: Bundle directory.
        dx: Solver cell [m]; picks the raster.
        reach_m: Radius the bundle's surface covers about the scene origin [m].
    Returns:
        Scene-frame elevation [m].
    """
    raster, (x0, cell, _, y1, _, ncell), _ = read_raster(nearest_raster(bundle, "dtm", dx))
    x = x0 + (np.arange(raster.shape[1]) + 0.5) * cell
    y = y1 + (np.arange(raster.shape[0]) + 0.5) * ncell
    X, Y = np.meshgrid(x, y)
    if box:        # a site following its polygon: its fetch box, less the rim, as the disc less the rim
        r = FLOOR_RIM_M
        inside = (X > box[0] + r) & (X < box[2] - r) & (Y > box[1] + r) & (Y < box[3] - r)
    else:
        inside = np.hypot(X, Y) < reach_m - FLOOR_RIM_M
    return float(np.nanpercentile(np.where(inside, raster, np.nan), FLOOR_PERCENTILE))
