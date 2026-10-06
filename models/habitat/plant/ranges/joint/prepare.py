"""Training and evaluation data of the joint model, assembled from the per-species products.

Training points. Each species contributes exactly what its own MaxEnt saw (or would see): its thinned presences
(``maxent/samples.csv``) and its effort-weighted background points in its calibration ecoregions
(``maxent/background.csv``), as written by ``pipeline.run_species`` (``scripts/run_fit.py``, with or without
``--prepare-only``). Every point gets one common predictor vector, so that species can share a representation of the
environment: the 19 WorldClim 2.1 bioclimatic variables and elevation at 30 arc-seconds (about 1 km) and four
SoilGrids 2.0 topsoil properties at 7.5 arc-seconds (about 230 m; NaN on water, rock, built land and outside North
America).

Scale. About 200 million points share far fewer distinct places: background points are ~1 km cell centres shared by
many species. Every predictor is a function of the point's 7.5" cell and its 30" cell, so points are reduced to their
distinct (7.5", 30") cell pairs, each pair is sampled once, in file order, at the first point that fell in it (its
values are exactly those of every point in it), and the points index the cell table. Nothing national is held in
memory beyond the 8-byte cell key of each point.

Evaluation plots. Independent presence/absence plots (VegBank, BLM AIM, FIA; ``validate.PlotTruth``), never used
in training, with their predictors, their ecoregion on the 240 m grid (the calibration-area test the maps use) and,
where a per-species MaxEnt range card exists, its suitability at the plots (for a paired comparison).

Output layout (the directory ``data.JointData`` reads): ``species.csv``; per point ``lat.npy``, ``lon.npy``
(float32), ``sid.npy`` (int32 species row), ``pres.npy`` (int8, 1 = presence) and ``train_points_cache.npy``
(float32 [N, 24]); ``names.npy`` (the 24 predictor names); ``joint_data.npz`` (names and the VegBank plots) and
``plots_<source>.npz`` (further plot sets); ``tree/natives.dated.nwk``; ``meta.json``.
"""
from __future__ import annotations

import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .. import fine as grid7
from ..fine import FineStack
from ..predictors import GlobalStack
from ..soil import VARS as SOIL_VARS, SoilPoints

N7 = grid7.NROW * grid7.NCOL                  # 7.5" cells of the North America grid (< 2^31)


def cell_keys(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Exact cell key of each point: (7.5" cell, 30" cell) packed in one int64 (31 + 30 bits). Outside the 7.5"
    domain the first part is N7. Both parts are kept because floating-point rounding can put a point within ~1e-12
    degree of an edge in a 30" cell its 7.5" cell does not nominally belong to."""
    r30, c30 = GlobalStack.rowcol(lon, lat)
    r7, c7 = FineStack.rowcol(lon, lat)
    inside = (r7 >= 0) & (r7 < grid7.NROW) & (c7 >= 0) & (c7 < grid7.NCOL)
    return (np.where(inside, r7 * grid7.NCOL + c7, N7) << 30) | (r30 * 43200 + c30)


class Predictors:
    """The 24 predictors at points: WorldClim (``GlobalStack``) and SoilGrids (``SoilPoints``). The memory maps are
    reopened per chunk: pages a map has touched stay in the process's resident set until it is unmapped."""

    def __init__(self, worldclim: str | Path, soil_dir: str | Path):
        self.worldclim, self.soil_dir = Path(worldclim), Path(soil_dir)
        self.names = list(GlobalStack(self.worldclim).variables) + list(SOIL_VARS)

    def sample(self, lon: np.ndarray, lat: np.ndarray, out: np.ndarray, chunk: int = 250_000, log=None) -> None:
        for i in range(0, len(lon), chunk):
            g, soil = GlobalStack(self.worldclim), SoilPoints(self.soil_dir)
            x, y = lon[i:i + chunk], lat[i:i + chunk]
            out[i:i + len(x), :20] = g.at(x, y)
            out[i:i + len(x), 20:24] = soil.at(x, y)
            del g, soil
            if log and (i // chunk) % 40 == 0:
                log(f"  cells sampled {i + len(x):,}/{len(lon):,}")


def _read_species(args) -> dict:
    """One species' presences and background (coordinates in file order) and its summary fields."""
    sp, d = args
    s = json.loads((d / "summary.json").read_text())
    p = pd.read_csv(d / "maxent/samples.csv", usecols=["x", "y"])
    b = pd.read_csv(d / "maxent/background.csv", usecols=["x", "y"])
    return {"species": sp, "lon": np.concatenate([p.x.values, b.x.values]),
            "lat": np.concatenate([p.y.values, b.y.values]), "n_p": len(p), "n_b": len(b),
            "family": s.get("family"), "n_buffer_points": s.get("n_buffer_points", 0), "beta": s.get("beta"),
            "calibration_ecoregions": " ".join(map(str, s.get("calibration_ecoregions", []))),
            "predictors": " ".join(s.get("predictors", [])), "source": "fitted" if "beta" in s else "prepared"}


def _usable(d: Path) -> bool:
    """A species directory with model inputs: a finished summary (fitted, or prepared only) and its samples."""
    try:
        s = json.loads((d / "summary.json").read_text())
    except (OSError, ValueError):
        return False
    return (d / "maxent/samples.csv").exists() and ("beta" in s or s.get("stage") == "prepared")


class NpyWriter:
    """Write a .npy file sequentially with ordinary file writes (filling a writable memory map instead keeps every
    written page in the process's resident set)."""

    def __init__(self, path, dtype, shape):
        self.f = open(path, "wb")
        self.dtype = np.dtype(dtype)
        np.lib.format.write_array_header_1_0(self.f, {"descr": np.lib.format.dtype_to_descr(self.dtype),
                                                      "fortran_order": False, "shape": tuple(shape)})

    def write(self, a):
        self.f.write(np.ascontiguousarray(a, dtype=self.dtype).tobytes())

    def close(self):
        self.f.close()


def training_data(out: str | Path, tree_dir: str | Path, species_table: str | Path, products: Sequence[str | Path],
                  predictors: Predictors, workers: int = 12, log=print) -> dict:
    """Write the training points of every species placed on the tree (``tree_dir``: ``natives.dated.nwk`` and
    ``tree_placement.csv``) that has products in one of ``products`` (species directories, searched in order)."""
    t0 = time.time()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    place = pd.read_csv(Path(tree_dir) / "tree_placement.csv")
    table = pd.read_csv(species_table)
    table["species"] = table.wcvp_accepted_name.str.replace(" ", "_")
    cand = sorted(set(place.loc[place.placement != "dropped", "species"]))
    jobs = []
    for sp in cand:
        for base in map(Path, products):
            if _usable(base / sp):
                jobs.append((sp, base / sp))
                break
    log(f"{len(jobs)} of {len(cand)} species on the tree have products")

    # pass 1: points -> cell keys; per-point arrays go straight to disk, only the int64 keys stay in memory
    S = len(jobs)
    tmp64 = {k: out / f".{k}64.tmp" for k in ("lon", "lat")}
    f64 = {k: open(v, "wb") for k, v in tmp64.items()}
    parts = {k: open(out / f".{k}.part", "wb") for k in ("lon", "lat", "sid", "pres")}
    keys, meta = [], []
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(_read_species, jobs, chunksize=16)):
            n = r["n_p"] + r["n_b"]
            keys.append(cell_keys(r["lon"], r["lat"]))
            f64["lon"].write(r["lon"].astype(np.float64).tobytes())
            f64["lat"].write(r["lat"].astype(np.float64).tobytes())
            parts["lon"].write(r["lon"].astype(np.float32).tobytes())
            parts["lat"].write(r["lat"].astype(np.float32).tobytes())
            parts["sid"].write(np.full(n, i, np.int32).tobytes())
            parts["pres"].write(np.r_[np.ones(r["n_p"], np.int8), np.zeros(r["n_b"], np.int8)].tobytes())
            meta.append({"species": r["species"], "n_presence": r["n_p"], "n_background": r["n_b"]} |
                        {k: r[k] for k in ("family", "calibration_ecoregions", "beta", "predictors",
                                           "n_buffer_points", "source")})
            if len(meta) % 1000 == 0:
                log(f"  read {len(meta):,}/{S:,} species")
    for f in list(f64.values()) + list(parts.values()):
        f.close()
    sp_tab = pd.DataFrame(meta)
    offsets = np.zeros(S + 1, np.int64)
    offsets[1:] = np.cumsum((sp_tab.n_presence + sp_tab.n_background).values)
    N = int(offsets[-1])
    for k, dt in (("lon", np.float32), ("lat", np.float32), ("sid", np.int32), ("pres", np.int8)):
        w = NpyWriter(out / f"{k}.npy", dt, (N,))
        with open(out / f".{k}.part", "rb") as f:
            while chunk := f.read(1 << 26):
                w.f.write(chunk)
        w.close()
        (out / f".{k}.part").unlink()
    log(f"{N:,} points ({int(sp_tab.n_presence.sum()):,} presences); {time.time() - t0:.0f} s")

    # pass 2: distinct cells, each represented by one of its points
    K = np.concatenate(keys)
    del keys
    cells, inv = np.unique(K, return_inverse=True)
    del K
    U = len(cells)
    inv = inv.astype(np.int32)
    first = np.empty(U, np.int64)
    first[inv] = np.arange(N, dtype=np.int64)
    first.sort()                                                   # file order
    clon = np.fromfile(tmp64["lon"], np.float64)[first]
    clat = np.fromfile(tmp64["lat"], np.float64)[first]
    for v in tmp64.values():
        v.unlink()
    o = np.searchsorted(cells, cell_keys(clon, clat))             # back to the order of ``cells``
    lon_c, lat_c = np.empty(U), np.empty(U)
    lon_c[o], lat_c[o] = clon, clat
    del first, o, clon, clat
    if not np.array_equal(cell_keys(lon_c, lat_c), cells):
        raise RuntimeError("cell representatives do not reproduce their cell keys")
    log(f"{U:,} distinct cells for {N:,} points ({N / U:.1f} points per cell); {time.time() - t0:.0f} s")

    # pass 3: predictors per cell, then per point
    X = np.empty((U, len(predictors.names)), np.float32)
    predictors.sample(lon_c, lat_c, X, log=log)
    w = NpyWriter(out / "train_points_cache.npy", np.float32, (N, X.shape[1]))
    for i in range(0, N, 5_000_000):
        w.write(X[inv[i:i + 5_000_000]])
    w.close()
    np.save(out / "names.npy", np.array(predictors.names))
    n_bad = int((~np.isfinite(X[:, :20]).all(1))[inv].sum())
    del X, inv

    sp_tab["offset"] = offsets[:-1]
    t = table.set_index("species")
    for col in ("us_regions", "listed"):
        if col in t:
            sp_tab[col] = sp_tab.species.map(t[col])
    sp_tab["placement"] = sp_tab.species.map(place.set_index("species").placement)
    sp_tab.to_csv(out / "species.csv", index=False)
    info = {"species": S, "points": N, "presences": int(sp_tab.n_presence.sum()), "cells": U,
            "points_in_cells_without_climate": n_bad, "fitted": int((sp_tab.source == "fitted").sum()),
            "prepared": int((sp_tab.source == "prepared").sum()), "seconds": round(time.time() - t0)}
    (out / "meta.json").write_text(json.dumps(info, indent=1))
    (out / "tree").mkdir(exist_ok=True)                            # the trainer reads <dir>/tree/natives.dated.nwk
    for f in ("natives.dated.nwk", "tree_placement.csv"):
        shutil.copy(Path(tree_dir) / f, out / "tree" / f)
    log(json.dumps(info))
    return info


def ecoregion_ids(eco_raster: str | Path, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(RESOLVE ecoregion id of the 240 m cell under each point, -1 off the grid; the cell's row; its column)."""
    from pyproj import Transformer                                 # before rasterio (its bundled PROJ)
    import rasterio
    with rasterio.open(eco_raster) as r:
        crs, t, a = r.crs.to_wkt(), r.transform, r.read(1)
    x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):    # a failed transformation must fail loudly
        raise RuntimeError("projecting plot coordinates returned non-finite values")
    rows = np.floor((t.f - y) / -t.e).astype(np.int64)
    cols = np.floor((x - t.c) / t.a).astype(np.int64)
    ok = (rows >= 0) & (rows < a.shape[0]) & (cols >= 0) & (cols < a.shape[1])
    out = np.full(len(lon), -1, np.int32)
    out[ok] = a[rows[ok], cols[ok]]
    return out, rows, cols


def plot_sets(data_dir: str | Path, truth, predictors: Predictors, grid_dir: str | Path,
              cards: dict[str, Sequence[str | Path]] | None = None, sources: Sequence[str] = ("vegbank", "aim", "fia"),
              region: str = "CONUS", min_presence: int = 20, device: str = "cpu", log=print) -> None:
    """Write the plot sets of a training data directory: ``joint_data.npz`` (VegBank, with the predictor names) and
    ``plots_<source>.npz`` for the others. ``truth``: a ``validate.PlotTruth``. A species is scored on a source when
    it is native to ``region`` (species table ``us_regions``) and has >= ``min_presence`` presence plots.
    ``cards``: per-species MaxEnt range-card directories by mode ({"occurrences": [...], "daru": [...]}); the
    card's suitability at the plots, (value - 1) / 254 and 0 outside its calibration area, is stored as
    ``maxent_occ`` / ``maxent_daru`` (NaN without a card)."""
    from .. import codec
    from ..predictors import ConusStack
    data_dir = Path(data_dir)
    sp_tab = pd.read_csv(data_dir / "species.csv")
    species = sp_tab.species.tolist()
    native = [region in str(r).split(",") for r in sp_tab.us_regions]
    stack = ConusStack(grid_dir) if cards else None
    attrs = {"vegbank": ("vb_locs", "vb_sp"), "aim": ("aim_locs", "aim_sp"), "fia": ("fia_locs", "fia_sp")}
    from .. import validate
    for src in sources:
        locs, hits = (getattr(truth, a) for a in attrs[src])
        if locs is None:
            log(f"{src}: no plots")
            continue
        locs = locs[validate.in_domain(locs.lat, locs.lon, "conus")]
        lon, lat = locs.lon.values.astype(np.float64), locs.lat.values.astype(np.float64)
        pos = {k: i for i, k in enumerate(locs.index)}
        sids, ys = [], []
        for i, s in enumerate(species):
            hit = hits.get(s.replace("_", " ")) if native[i] else None
            if not hit:
                continue
            idx = [pos[k] for k in hit if k in pos]
            if len(idx) < min_presence:
                continue
            y = np.zeros(len(lon), np.int8)
            y[idx] = 1
            sids.append(i)
            ys.append(y)
        order = np.lexsort((lon, -lat))                            # read the predictor maps in file order
        X = np.empty((len(lon), len(predictors.names)), np.float32)
        Xo = np.empty_like(X)
        predictors.sample(lon[order], lat[order], Xo)
        X[order] = Xo
        eco, rows, cols = ecoregion_ids(Path(grid_dir) / "ecoregion_id_conus240.tif", lon, lat)
        Y = np.stack(ys) if ys else np.zeros((0, len(lon)), np.int8)
        base = {}
        for mode, key in (("occurrences", "maxent_occ"), ("daru", "maxent_daru")):
            b = np.full(Y.shape, np.nan, np.float16)
            for k, s_ in enumerate(sids):
                label = species[s_]
                card = next((Path(d) / label / f"{label}.rangecard" for d in (cards or {}).get(mode, [])
                             if (Path(d) / label / f"{label}.rangecard").exists()), None)
                if card is not None:
                    v = codec.decode_cells(card, stack, rows, cols, device=device)
                    b[k] = np.where(v["suitability"] > 0, (v["suitability"].astype(np.float32) - 1) / 254, 0.0)
            base[key] = b
        members = dict(plot_lon=lon.astype(np.float32), plot_lat=lat.astype(np.float32), plot_X=X, plot_eco=eco,
                       eval_sid=np.array(sids, np.int32), eval_y=Y, **base)
        if src == "vegbank":
            np.savez(data_dir / "joint_data.npz", names=np.array(predictors.names), **members)
        else:
            np.savez(data_dir / f"plots_{src}.npz", **members)
        log(f"{src}: {len(lon):,} plots, {len(sids)} species with >= {min_presence} presence plots, "
            f"{int(np.isfinite(base['maxent_occ'][:, 0]).sum()) if len(sids) else 0} with MaxEnt cards")
