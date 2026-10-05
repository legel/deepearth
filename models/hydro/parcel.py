"""A sub-metre parcel: terrain sampled from the surface mesh, hydrology from a class raster.

The solver reads a DTM raster and a class-code raster on the same grid. Each class carries a
Manning's n, a saturated conductivity, a surface storage and an impervious fraction; this
module turns those into the per-cell fields `solver.Surface` takes.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from solver import Surface

MM_HR = 1.0 / 1000.0 / 3600.0
"""mm/hr -> m/s."""


@dataclass(frozen=True)
class ClassParams:
    """Hydrology of one surface class.

    Args:
        manning_n: Roughness [s/m^(1/3)].
        ksat_mm_hr: Saturated conductivity [mm/hr].
        smax_mm: Surface storage [mm].
        impervious_frac: Fraction of the cell that admits no infiltration.
    """

    manning_n: float
    ksat_mm_hr: float
    smax_mm: float
    impervious_frac: float

    @property
    def infiltration_m_s(self) -> float:
        """Constant infiltration capacity [m/s] of the pervious fraction."""
        return self.ksat_mm_hr * MM_HR * (1.0 - self.impervious_frac)


ClassTable = Dict[int, ClassParams]
"""Class code -> parameters. Code 0 is unlabelled."""

SYNTHETIC_CLASSES: ClassTable = {
    1: ClassParams(manning_n=0.014, ksat_mm_hr=0.05, smax_mm=1.25, impervious_frac=0.99),
    2: ClassParams(manning_n=0.225, ksat_mm_hr=30.0, smax_mm=3.5, impervious_frac=0.0),
    3: ClassParams(manning_n=0.25, ksat_mm_hr=55.0, smax_mm=5.5, impervious_frac=0.0),
    4: ClassParams(manning_n=0.035, ksat_mm_hr=10.0, smax_mm=2.0, impervious_frac=0.0),
}
"""Pavement, turf, planting bed and channel, for synthetic terrain."""


def load_class_table(path: Path) -> Tuple[ClassTable, Dict[str, int]]:
    """Class parameters from an ontology JSON with a `classes` list of `{class_id, params}`.

    Returns:
        (code -> parameters, class_id -> code), codes numbered from 1 in list order.
    """
    doc = json.loads(Path(path).read_text())
    table, codes = {}, {}
    for i, cls in enumerate(doc["classes"]):
        p = cls["params"]
        codes[cls["class_id"]] = i + 1
        table[i + 1] = ClassParams(
            manning_n=float(p["manning_n"]), ksat_mm_hr=float(p["ksat_mm_hr"]),
            smax_mm=float(p["smax_mm"]), impervious_frac=float(p["impervious_frac"]))
    return table, codes


def fields(classes: np.ndarray, table: ClassTable, default: ClassParams) -> Dict[str, np.ndarray]:
    """Per-cell manning_n, infiltration [m/s] and smax [m] from a class-code raster."""
    n = np.full(classes.shape, default.manning_n, dtype=np.float32)
    inf = np.full(classes.shape, default.infiltration_m_s, dtype=np.float32)
    smax = np.full(classes.shape, default.smax_mm / 1000.0, dtype=np.float32)
    for code, p in table.items():
        m = classes == code
        n[m], inf[m], smax[m] = p.manning_n, p.infiltration_m_s, p.smax_mm / 1000.0
    return {"manning_n": n, "infiltration": inf, "smax_m": smax}


def build_surface(z: np.ndarray, classes: np.ndarray, table: ClassTable, default: ClassParams,
                  deficit_mm: Optional[float] = None) -> Surface:
    """A solver `Surface` from a DTM and a class raster on the same grid.

    Args:
        z: Elevation [m], NaN outside the parcel.
        classes: Class codes, 0 unlabelled.
        table: Code -> parameters.
        default: Parameters for unlabelled cells.
        deficit_mm: Soil storage [mm] for every cell; None is unbounded.
    """
    assert z.shape == classes.shape, f"{z.shape} != {classes.shape}"
    f = fields(classes, table, default)
    deficit = None if deficit_mm is None else np.full(z.shape, deficit_mm / 1000.0, dtype=np.float32)
    return Surface(z=z.astype(np.float32), f0=f["infiltration"], fc=f["infiltration"],
                   k=np.zeros(z.shape, dtype=np.float32), max_deficit_m=deficit,
                   manning_n=f["manning_n"], smax_m=f["smax_m"])


def read_dem(path: Path) -> Tuple[np.ndarray, float, object]:
    """A metric DEM.

    Returns:
        (elevation with NaN at nodata, cell size [m], rasterio transform).
    """
    import rasterio

    with rasterio.open(path) as src:
        assert src.crs is not None and src.crs.is_projected, f"{path}: CRS must be projected"
        z = src.read(1).astype(np.float32)
        if src.nodata is not None:
            z[z == src.nodata] = np.nan
        dx = abs(src.transform.a)
        assert abs(abs(src.transform.e) - dx) < 1e-6 * dx, f"{path}: cells are not square"
        return z, float(dx), src.transform


def read_rasters(dtm: Path, classes: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    """A DTM and a class raster on one grid.

    Returns:
        (elevation with NaN at nodata, class codes, cell size [m]).
    """
    import rasterio

    z, dx, transform = read_dem(dtm)
    with rasterio.open(classes) as src:
        assert src.shape == z.shape and src.transform == transform, (
            f"{classes} is not on the grid of {dtm}")
        codes = src.read(1).astype(np.int32)
    return z, codes, dx


def disc_mask(shape: Tuple[int, int], dx: float, radius_m: float) -> np.ndarray:
    """True inside a disc centred on the grid."""
    r = (np.arange(shape[0]) + 0.5 - shape[0] / 2) * dx
    c = (np.arange(shape[1]) + 0.5 - shape[1] / 2) * dx
    return (r[:, None] ** 2 + c[None, :] ** 2) <= radius_m ** 2


def synthetic(dx: float, radius_m: float = 112.32, slope: float = 0.03) -> Tuple[np.ndarray, np.ndarray]:
    """A disc parcel with a slope, a channel, a pit and a flat, plus its class raster.

    Args:
        dx: Cell size [m].
        radius_m: Disc radius [m]; cells outside are NaN.
        slope: Regional gradient, falling southward.

    Returns:
        (elevation [m] with NaN outside the disc, class codes on `SYNTHETIC_CLASSES`).
    """
    n = int(np.ceil(2 * radius_m / dx))
    y = (np.arange(n) + 0.5) * dx
    x = (np.arange(n) + 0.5) * dx
    yy, xx = np.meshgrid(y, x, indexing="ij")
    z = 100.0 - slope * yy
    channel = np.abs(xx - 0.6 * n * dx) < 0.03 * n * dx
    z = z - 0.4 * np.exp(-((xx - 0.6 * n * dx) / (0.03 * n * dx)) ** 2)
    pit = np.hypot(xx - 0.3 * n * dx, yy - 0.4 * n * dx) < 0.08 * n * dx
    z = z - 1.0 * np.clip(1.0 - np.hypot(xx - 0.3 * n * dx, yy - 0.4 * n * dx) / (0.08 * n * dx), 0.0, 1.0)
    flat = (yy > 0.7 * n * dx) & (xx < 0.45 * n * dx)
    z = np.where(flat, 100.0 - slope * 0.7 * n * dx, z)
    classes = np.where(xx < 0.2 * n * dx, 1, 2).astype(np.int32)
    classes[pit] = 3
    classes[channel] = 4
    z = np.where(disc_mask((n, n), dx, radius_m), z, np.nan)
    return z.astype(np.float32), classes


def fill_sinks(z: np.ndarray) -> Tuple[np.ndarray, float, int]:
    """Raise closed depressions to their spill elevation.

    A DEM resampled from a coarser one carries sinks that the ground does not have, and the
    solver conserves mass into them faithfully. Filling is reported, never silent.

    Args:
        z: Elevation [m], NaN outside the domain.

    Returns:
        (filled elevation, volume raised [m^3] per unit cell area, cells raised).
    """
    from inflow import fill_and_route

    valid = np.isfinite(z)
    # Nodata is an OUTLET, not a wall. Seeding it low lets the flood reach every valid cell
    # beside it at that cell's own elevation, so nothing is filled behind a hole in the raster.
    # Walling it instead dammed the ground around each dropped building and raised 10,442 m3
    # against the 205 m3 the terrain actually holds.
    work = np.where(valid, z, -1e6).astype(np.float64)
    filled, _, _ = fill_and_route(work)
    raised = np.where(valid, filled - work, 0.0)
    out = np.where(valid, filled, np.nan).astype(z.dtype)
    return out, float(raised.sum()), int((raised > 1e-3).sum())


def radii(bundle: Path) -> Tuple[float, float]:
    """(display radius, buffer) [m] from the bundle's `surface/aoi_<site>.json`, the one place
    the buffer is defined; a document without one was fetched with none."""
    doc = json.loads(next((Path(bundle) / "surface").glob("aoi_*.json")).read_text())
    return float(doc["aoi"]["radius_m"]), float(doc["aoi"].get("buffer_m", 0.0))


def window(z: np.ndarray, transform: object, centre: Tuple[float, float], reach_m: float,
           dx: float) -> Tuple[np.ndarray, object]:
    """The DTM cropped to the square a bundle reaching `reach_m` is gridded on.

    The square is centred on `centre` with a half-width of `reach_m` rounded up to whole cells, as
    `semantics/common.grid_for` grids a bundle; its edges snap to the DTM's own cells. Where the
    square runs past the DTM, as the buffered mesh's rasters do on one side because they are centred
    on the extract's disc rather than on the anchor, the cells are NaN, outside the domain.

    Returns:
        (cropped elevation, its affine transform).
    """
    from affine import Affine

    half = dx * np.ceil(reach_m / dx - 1e-9)
    n = int(round(2 * half / dx))
    c0 = int(round((centre[0] - half - transform.c) / transform.a))
    r0 = int(round((transform.f - centre[1] - half) / -transform.e))
    pad = max(0, -c0, -r0, c0 + n - z.shape[1], r0 + n - z.shape[0])
    zp = np.pad(z, pad, constant_values=np.nan)
    t = transform
    return zp[r0 + pad:r0 + pad + n, c0 + pad:c0 + pad + n], Affine(t.a, t.b, t.c + c0 * t.a, t.d, t.e,
                                                                    t.f + r0 * t.e)


def box(bundle: Path) -> Optional[Dict[str, Tuple[float, float, float, float]]]:
    """The bundle's boxes, {"fetch", "display"} as (x0, y0, x1, y1) [m], when its site follows its ordered
    polygon; None for a disc site, which then runs exactly as before."""
    doc = json.loads(next((Path(bundle) / "surface").glob("aoi_*.json")).read_text())
    b = doc["aoi"].get("box_scene_m")
    return {k: tuple(float(v) for v in b[k]) for k in ("fetch", "display")} if b else None


def window_box(z: np.ndarray, transform: object, fetch: Tuple[float, float, float, float],
               dx: float) -> Tuple[np.ndarray, object]:
    """The DTM cropped to a fetch box snapped outward to the `dx` lattice, as `window` crops to the disc's square,
    NaN where the DTM runs short. Each side is made even, one cell further east or south, because the viewer
    halves the grid (`water_viewer.py build --k 2`)."""
    from affine import Affine

    e = 1e-9
    x0, x1 = dx * np.floor(fetch[0] / dx + e), dx * np.ceil(fetch[2] / dx - e)
    y0, y1 = dx * np.floor(fetch[1] / dx + e), dx * np.ceil(fetch[3] / dx - e)
    nc, nr = int(round((x1 - x0) / dx)), int(round((y1 - y0) / dx))
    nc, nr = nc + nc % 2, nr + nr % 2
    c0 = int(round((x0 - transform.c) / transform.a))
    r0 = int(round((transform.f - y1) / -transform.e))
    pad = max(0, -c0, -r0, c0 + nc - z.shape[1], r0 + nr - z.shape[0])
    zp = np.pad(z, pad, constant_values=np.nan)
    t = transform
    return zp[r0 + pad:r0 + pad + nr, c0 + pad:c0 + pad + nc], Affine(t.a, t.b, t.c + c0 * t.a, t.d, t.e,
                                                                      t.f + r0 * t.e)


def outside_box(shape: Tuple[int, int], transform: object, b: Tuple[float, float, float, float]) -> np.ndarray:
    """(rows, cols) True where a cell centre lies outside the box (x0, y0, x1, y1) [m]."""
    rows, cols = np.indices(shape)
    x, y = transform.c + (cols + 0.5) * transform.a, transform.f + (rows + 0.5) * transform.e
    return ~((x >= b[0]) & (x <= b[2]) & (y >= b[1]) & (y <= b[3]))


def distance(shape: Tuple[int, int], transform: object, centre: Tuple[float, float]) -> np.ndarray:
    """(rows, cols) horizontal distance [m] of every cell centre from `centre`."""
    rows, cols = np.indices(shape)
    return np.hypot(transform.c + (cols + 0.5) * transform.a - centre[0],
                    transform.f + (rows + 0.5) * transform.e - centre[1])
