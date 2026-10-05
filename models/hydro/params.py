"""Per-cell hydrology from the semantics bundle, the parameterisation every solver shares.

The bundle carries a class raster per resolution (uint8, pixel = row id, 46 unobserved, 255
nodata) and `parameters.json`, whose `rows[id].params` hold the class physics. Nearest
resampling onto the solver grid keeps the classes categorical; unobserved and nodata cells take
the bare-soil row and are counted.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from infiltration import Soil
from solver import Surface

MM_HR = 1.0 / 1000.0 / 3600.0
"""mm/hr -> m/s."""

FIELDS = ("manning_n", "ksat_mm_hr", "smax_mm", "impervious_frac", "poro")
"""Parameters the water solver takes from the class table."""

BARE_SOIL = "dirt_soil"
UNOBSERVED = 46
NODATA = 255
ZONE = "UTM zone 10N"


@dataclass(frozen=True)
class Table:
    """Class physics indexed by row id.

    Args:
        names: class_id per row.
        values: {field: array[n_rows]} for every entry of `FIELDS`.
        bare_soil: Row id substituted for unobserved and nodata pixels.
    """

    names: Tuple[str, ...]
    values: Dict[str, np.ndarray]
    bare_soil: int

    @property
    def infiltration_m_s(self) -> np.ndarray:
        """Constant infiltration capacity [m/s] of each row's pervious fraction."""
        return self.values["ksat_mm_hr"] * MM_HR * (1.0 - self.values["impervious_frac"])


def load_table(path: Path) -> Table:
    """Read `parameters.json`.

    Returns:
        A `Table` whose arrays are indexed by row id.
    """
    doc = json.loads(Path(path).read_text())
    rows = sorted(doc["rows"], key=lambda r: r["id"])
    assert [r["id"] for r in rows] == list(range(len(rows))), "row ids must be 0..n-1"
    names = tuple(r["class_id"] for r in rows)
    values = {f: np.array([float(r["params"][f]) for r in rows]) for f in FIELDS}
    return Table(names=names, values=values, bare_soil=names.index(BARE_SOIL))


def read_classes(path: Path) -> Tuple[np.ndarray, object]:
    """A class raster and its affine transform, checked to be on UTM zone 10N."""
    import rasterio

    with rasterio.open(path) as src:
        assert ZONE in src.crs.to_wkt() or src.crs.to_epsg() == 32610, f"{path}: {src.crs}"
        assert src.nodata == NODATA, f"{path}: nodata {src.nodata}"
        return src.read(1).astype(np.uint8), src.transform


def resample_nearest(codes: np.ndarray, transform: object, shape: Tuple[int, int],
                     grid: object) -> np.ndarray:
    """Class codes at the centres of another grid's cells; outside the raster is nodata.

    Args:
        codes: Class raster.
        transform: Its affine transform.
        shape: (rows, cols) of the target grid.
        grid: The target grid's affine transform, same projected frame.
    """
    rows, cols = shape
    x = grid.c + (np.arange(cols) + 0.5) * grid.a
    y = grid.f + (np.arange(rows) + 0.5) * grid.e
    inv = ~transform
    cc = np.floor(inv.a * x + inv.c).astype(np.int64)
    rr = np.floor(inv.e * y + inv.f).astype(np.int64)
    out = np.full(shape, NODATA, dtype=np.uint8)
    ok_r = (rr >= 0) & (rr < codes.shape[0])
    ok_c = (cc >= 0) & (cc < codes.shape[1])
    out[np.ix_(ok_r, ok_c)] = codes[np.ix_(rr[ok_r], cc[ok_c])]
    return out


def fields(codes: np.ndarray, table: Table) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Per-cell parameters from class codes.

    Returns:
        ({field: array} with `infiltration_m_s` and `smax_m` added, counts of substituted cells).
    """
    counts = {"unobserved": int((codes == UNOBSERVED).sum()), "nodata": int((codes == NODATA).sum())}
    ids = np.where((codes == UNOBSERVED) | (codes == NODATA), table.bare_soil, codes).astype(np.int64)
    assert ids.max() < len(table.names), f"class id {ids.max()} outside the table"
    out = {f: table.values[f][ids].astype(np.float32) for f in FIELDS}
    out["infiltration_m_s"] = table.infiltration_m_s[ids].astype(np.float32)
    out["smax_m"] = (out["smax_mm"] / 1000.0).astype(np.float32)
    return out, counts


def class_raster_name(dx: float) -> str:
    """`class_ground_0p2m.tif` for 0.2 m."""
    return f"class_ground_{f'{dx:g}'.replace('.', 'p')}m.tif"


CELL_BANDS = ("infiltration_mm_hr", "deficit_mm", "manning_n", "smax_mm")
"""`--cells`: per-cell hydrology from the caller's own model, overriding the class table (NaN keeps the table's
value; a NaN deficit is unbounded). Infiltration is the cell's whole capacity, its impervious share already out."""

GAR_BANDS = ("psi_f_mm", "theta_s", "theta_r", "lambda", "theta_i")
"""Optional `--cells` bands. When all are present the storm runs Green-Ampt with redistribution (`infiltration`):
K_s is `infiltration_mm_hr`, F_max is `deficit_mm` (the root zone's room above theta_i), and theta_i is the
balance's own state at the storm's start. A cell with any of them NaN, or K_s 0, takes nothing in."""


def read_cells(path: Path) -> Tuple[Dict[str, np.ndarray], object]:
    """The per-cell parameter raster: one float band per CELL_BANDS entry (and GAR_BANDS when given), band
    descriptions naming them."""
    import rasterio

    with rasterio.open(path) as src:
        names = list(src.descriptions)
        assert set(CELL_BANDS) <= set(names), f"{path}: bands {names}, want {CELL_BANDS}"
        want = CELL_BANDS + (GAR_BANDS if set(GAR_BANDS) <= set(names) else ())
        return {n: src.read(names.index(n) + 1).astype(np.float32) for n in want}, src.transform


def gar_soil(on: Dict[str, np.ndarray]) -> Soil:
    """The GAR soil of `--cells` values already on the solver grid; a cell missing any value is sealed."""
    ok = np.logical_and.reduce([np.isfinite(on[n]) for n in GAR_BANDS + ("infiltration_mm_hr",)])
    val = lambda n, fill: np.where(ok, on[n], fill).astype(np.float64)  # noqa: E731
    f_max = np.where(np.isfinite(on["deficit_mm"]), on["deficit_mm"] / 1000.0, np.inf)
    return Soil(ks=val("infiltration_mm_hr", 0.0) * MM_HR, psi_f=val("psi_f_mm", 0.1) / 1000.0,
                theta_s=val("theta_s", 0.45), theta_r=val("theta_r", 0.05), lam=val("lambda", 0.3),
                theta_i=val("theta_i", 0.2), f_max=f_max)


def resample_values(values: np.ndarray, transform: object, shape: Tuple[int, int], grid: object) -> np.ndarray:
    """Float values at the centres of another grid's cells (nearest); outside the raster is NaN."""
    rows, cols = shape
    x = grid.c + (np.arange(cols) + 0.5) * grid.a
    y = grid.f + (np.arange(rows) + 0.5) * grid.e
    inv = ~transform
    cc = np.floor(inv.a * x + inv.c).astype(np.int64)
    rr = np.floor(inv.e * y + inv.f).astype(np.int64)
    out = np.full(shape, np.nan, dtype=np.float32)
    ok_r = (rr >= 0) & (rr < values.shape[0])
    ok_c = (cc >= 0) & (cc < values.shape[1])
    out[np.ix_(ok_r, ok_c)] = values[np.ix_(rr[ok_r], cc[ok_c])]
    return out


def build_surface(z: np.ndarray, grid: object, bundle: Path, dx: float,
                  deficit_mm: Optional[float] = None,
                  beyond: Optional[np.ndarray] = None,
                  cells: Optional[Path] = None) -> Tuple[Surface, Dict[str, object]]:
    """A solver `Surface` for a DTM on `grid` from a bundle directory.

    Args:
        z: Elevation [m], NaN outside the parcel.
        grid: Affine transform of `z`.
        bundle: Bundle root holding `semantics/`.
        dx: Cell size [m]; picks `class_ground_0p1m.tif` or `class_ground_0p2m.tif`.
        deficit_mm: Soil storage [mm] for every cell; None is unbounded.

    Returns:
        (Surface, receipt with the class raster used and the substituted-cell counts).
    """
    raster = Path(bundle) / "semantics" / class_raster_name(dx)
    table = load_table(Path(bundle) / "semantics" / "parameters.json")
    codes, transform = read_classes(raster)
    on_grid = resample_nearest(codes, transform, z.shape, grid)
    on_grid = on_grid if beyond is None else np.where(beyond, NODATA, on_grid).astype(np.uint8)
    f, counts = fields(on_grid, table)
    deficit = None if deficit_mm is None else np.full(z.shape, deficit_mm / 1000.0, dtype=np.float32)
    given = {}
    if cells is not None:
        vals, t = read_cells(Path(cells))
        on = {n: resample_values(v, t, z.shape, grid) for n, v in vals.items()}
        take = lambda n: np.isfinite(on[n])  # noqa: E731
        f["infiltration_m_s"] = np.where(take("infiltration_mm_hr"), on["infiltration_mm_hr"] * MM_HR,
                                         f["infiltration_m_s"]).astype(np.float32)
        f["manning_n"] = np.where(take("manning_n"), on["manning_n"], f["manning_n"]).astype(np.float32)
        f["smax_m"] = np.where(take("smax_mm"), on["smax_mm"] / 1000.0, f["smax_m"]).astype(np.float32)
        if take("deficit_mm").any():
            base = deficit if deficit is not None else np.full(z.shape, np.inf, dtype=np.float32)
            deficit = np.where(take("deficit_mm"), on["deficit_mm"] / 1000.0, base).astype(np.float32)
        given = {"cell_parameters": Path(cells).name,
                 "cells_given": {n: int((take(n) & np.isfinite(z)).sum()) for n in on},
                 "infiltration": "gar" if set(GAR_BANDS) <= set(on) else "horton"}
    surface = Surface(z=z.astype(np.float32), f0=f["infiltration_m_s"], fc=f["infiltration_m_s"],
                      k=np.zeros(z.shape, dtype=np.float32), max_deficit_m=deficit,
                      manning_n=f["manning_n"], smax_m=f["smax_m"],
                      soil=gar_soil(on) if cells is not None and set(GAR_BANDS) <= set(on) else None)
    valid = np.isfinite(z)
    receipt = {"class_raster": raster.name, "bare_soil_row": table.bare_soil, **given,
               "substituted_cells": counts,
               "substituted_valid_cells": int(((on_grid == UNOBSERVED) | (on_grid == NODATA))[valid].sum()),
               "manning_n_mean": float(f["manning_n"][valid].mean()),
               "impervious_fraction_mean": float(f["impervious_frac"][valid].mean())}
    return surface, receipt


def building_mask(codes: np.ndarray, path: Path) -> np.ndarray:
    """True where the class raster labels a building surface.

    A mapped roof is not flood ground: rain landing on it leaves through downpipes the surface
    model does not carry, so a terrain raster that includes the building ponds water on top of
    it. Marking those cells nodata routes that water off the footprint instead.
    """
    doc = json.loads(Path(path).read_text())
    ids = [r["id"] for r in doc["rows"] if r.get("volume_category") == "building"]
    return np.isin(codes, ids)
