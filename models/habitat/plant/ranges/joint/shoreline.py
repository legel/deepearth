"""Shoreline records: the presences the per-species preparation dropped for lacking climate, restored.

Why they were dropped. The per-species pipeline (``pipeline.run_species``, step 6a) draws a species' presences from
its cleaned records, ``occurrences_clean.sample(min(n, 5000), random_state=seed)``, and reads the 30" WorldClim
predictors at each. WorldClim has no value in a pixel whose centre lies in the sea or a large lake, so a record on a
beach, dune, salt marsh or mangrove shore often has no climate and was dropped (Uniola paniculata, sea oats: 224 of
its 683 drawn presences). The species that live only there lose most of their evidence exactly where they live.

How they are restored. The draw is seeded, so replaying it reproduces the drawn presences exactly, and the ones
without climate are exactly those dropped (checked: Uniola paniculata 224, Croton punctatus 178, Abies concolor 0).
Each takes the climate of the nearest 30" pixel with climate within 5 km (``climate_fill.fill_points``; farther ones
stay dropped), SoilGrids at the point as every training point, and its fractional position on the 240 m grid
(``geodesy``, the transformation the rasters were warped with) for the environmental field and the calibration area.

Output: <data_dir>/shore_records.npz with ``lat``, ``lon`` (float32), ``sid`` (int32, row of species.csv), ``X``
[n, n_predictors] (float32, the columns of names.npy), ``fill_km``, ``rc`` [n, 2] (float32 fractional row, column on
the 240 m grid; -1e6 off it) and ``names``, sorted by species. Training appends these rows as presences after the
prepared rows.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from . import climate_fill
from .geodesy import WGS84ToConusAlbers, grid_positions
from .prepare import _usable


def dropped_presences(species_dir: str | Path, stack, seed: int = 0, max_occurrences: int = 5000) -> np.ndarray:
    """(lon, lat) [n, 2] of the presences ``pipeline.run_species`` drew for the species in ``species_dir`` and then
    dropped because their 30" pixel of ``stack`` (a ``predictors.GlobalStack``) has no climate."""
    occ = Path(species_dir) / "occurrences_clean.csv.gz"
    if not occ.exists():
        return np.empty((0, 2))
    df = pd.read_csv(occ, usecols=["decimalLongitude", "decimalLatitude"])
    o = df.sample(min(len(df), max_occurrences), random_state=seed)      # pipeline.run_species step 6a, verbatim
    xy = o[["decimalLongitude", "decimalLatitude"]].values
    ok = np.isfinite(stack.at(xy[:, 0], xy[:, 1])).all(1)
    return xy[~ok]


def _dropped(args) -> tuple[int, np.ndarray]:
    s, d, stack_path, seed, max_occurrences = args
    from ..predictors import GlobalStack
    if d is None:
        return s, np.empty((0, 2))
    return s, dropped_presences(d, GlobalStack(stack_path), seed, max_occurrences)


def restore_records(data_dir: str | Path, products: Sequence[str | Path], worldclim_stack: str | Path,
                    soil_dir: str | Path, transform: Sequence[float], proj_grids: str | Path, seed: int = 0,
                    max_occurrences: int = 5000, max_km: float = 5.0, workers: int = 12,
                    log: Callable = print) -> dict:
    """Write <data_dir>/shore_records.npz for the species of <data_dir>/species.csv. ``products``: per-species
    directories searched in order (the first holding usable products is the one ``prepare.training_data`` read);
    ``transform``: affine (a, b, c, d, e, f) of the 240 m EPSG:5070 grid; ``proj_grids``: PROJ grid directory
    (``geodesy.fetch_grids``)."""
    from ..predictors import GlobalStack
    from ..soil import VARS as SOIL_VARS, SoilPoints
    t0 = time.time()
    data_dir = Path(data_dir)
    sp = pd.read_csv(data_dir / "species.csv")
    jobs = []
    for s, label in enumerate(sp.species):
        d = next((Path(b) / label for b in products if _usable(Path(b) / label)), None)
        jobs.append((s, d, str(worldclim_stack), seed, max_occurrences))
    lon, lat, sid = [], [], []
    with ProcessPoolExecutor(workers) as ex:
        for k, (s, xy) in enumerate(ex.map(_dropped, jobs, chunksize=16)):
            if len(xy):
                lon.append(xy[:, 0])
                lat.append(xy[:, 1])
                sid.append(np.full(len(xy), s, np.int32))
            if k % 2000 == 0:
                log(f"  {k:,}/{len(jobs):,} species ({time.time() - t0:.0f} s)")
    lon = np.concatenate(lon) if lon else np.empty(0)
    lat = np.concatenate(lat) if lat else np.empty(0)
    sid = np.concatenate(sid) if sid else np.empty(0, np.int32)
    log(f"{len(lon):,} dropped presences in {len(np.unique(sid)):,} species")

    names = [str(x) for x in (np.load(data_dir / "names.npy") if (data_dir / "names.npy").exists()
                              else np.load(data_dir / "joint_data.npz")["names"])]
    g = GlobalStack(worldclim_stack)
    n_clim = len(g.variables)
    if names[:n_clim] != list(g.variables):
        raise ValueError("the first predictors of the data directory are not the WorldClim stack's bands")
    X = np.full((len(lon), len(names)), np.nan, np.float32)
    X, dist = climate_fill.fill_points(X, lat, lon, g.data, climate_fill.CELL, n_clim, max_km)
    soil_cols = [names.index(v) for v in SOIL_VARS if v in names]
    if soil_cols:
        X[:, soil_cols] = SoilPoints(soil_dir).at(lon, lat)[:, [SOIL_VARS.index(names[c]) for c in soil_cols]]
    keep = np.isfinite(dist)
    log(f"filled {int(keep.sum()):,} within {max_km} km (median "
        f"{float(np.median(dist[keep])) if keep.any() else 0:.2f} km); {int((~keep).sum()):,} farther stay dropped")
    lon, lat, sid, X, dist = lon[keep], lat[keep], sid[keep], X[keep], dist[keep]
    rc = grid_positions(lat, lon, transform, WGS84ToConusAlbers(proj_grids))
    o = np.argsort(sid, kind="stable")
    np.savez(data_dir / "shore_records.npz", lat=lat[o].astype(np.float32), lon=lon[o].astype(np.float32),
             sid=sid[o].astype(np.int32), X=X[o], fill_km=dist[o].astype(np.float32), rc=rc[o], names=np.array(names))
    info = {"records": int(len(sid)), "species": int(len(np.unique(sid))), "beyond_km": int((~keep).sum()),
            "max_km": max_km, "seconds": round(time.time() - t0)}
    (data_dir / "shore_records.json").write_text(json.dumps(info, indent=1))
    log(json.dumps(info))
    return info
