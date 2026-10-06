"""SoilGrids 2.0 predictors (ISRIC; 250 m; ledger L10): pH (H2O), clay, sand and soil organic carbon at 5–15 cm.

Two copies, both built by ``scripts/fetch_soilgrids.sh`` + ``scripts/build_soil_stacks.py`` from SoilGrids by
area averaging:
* training: North America ~230 m grid of ``fine.py`` (int16 memmap per variable, SoilGrids integer units);
* rendering: CONUS 240 m grid (float32 memmap per variable, physical units), read by ``ConusStack(extra=...)``.
Values in physical units: pH, clay and sand in %, SOC in g/kg. SoilGrids is itself predicted partly from
WorldClim covariates, so these layers are not independent of climate; their added content is terrain- and
geology-driven variation at 250 m.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .fine import CELL, LAT1, LON0, NCOL, NROW

VARS = ["soil_phh2o", "soil_clay", "soil_sand", "soil_soc"]
SCALE = {"soil_phh2o": 0.1, "soil_clay": 0.1, "soil_sand": 0.1, "soil_soc": 0.1}   # SoilGrids mapped units
NODATA = -32768


class SoilPoints:
    """Point sampling of the North America ~230 m soil layers (nearest cell; NaN outside or where no soil)."""

    def __init__(self, directory: str | Path):
        d = Path(directory)
        self.layers = {v: np.memmap(d / f"{v}_na7p5s.i16", dtype=np.int16, mode="r", shape=(NROW, NCOL)) for v in VARS}

    def at(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        lon, lat = np.asarray(lon, float), np.asarray(lat, float)
        r = np.floor((LAT1 - lat) / CELL).astype(np.int64)
        c = np.floor((lon - LON0) / CELL).astype(np.int64)
        ok = (r >= 0) & (r < NROW) & (c >= 0) & (c < NCOL)
        out = np.full((len(lon), len(VARS)), np.nan)
        for j, v in enumerate(VARS):
            x = self.layers[v][r[ok], c[ok]].astype(np.float64)
            out[ok, j] = np.where(x == NODATA, np.nan, x * SCALE[v])
        return out
