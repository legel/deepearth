"""One species through the Daru (2024) pipeline, end to end.

Each stage writes its products into the species' work directory, so any stage can be inspected or rerun, and
``summary.json`` records the counts and parameters behind every decision.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely import vectorized
from shapely.geometry import mapping
from shapely.ops import transform as shp_transform, unary_union

from . import background, modelling, occurrences
from .predictors import CELL, GlobalStack


@dataclass
class Resources:
    """Shared inputs, loaded once per process."""
    root: Path
    stack: GlobalStack
    wgsrpd: gpd.GeoDataFrame
    land: gpd.GeoDataFrame
    lakes: gpd.GeoDataFrame
    ecoregions: gpd.GeoDataFrame
    wcvp_native: pd.DataFrame          # plant_name_id, taxon_name, family, area_code_l3 (native, not extinct/doubtful)
    sbm: pd.DataFrame                  # clade, km_per_year (family rows + 'ALL')
    bias: dict                         # from background.bias_grid
    fine: object = None                # fine.FineStack, loaded when predictors == "fine"
    soil: object = None                # soil.SoilPoints, loaded when the soil layers exist
    _family_cache: dict = field(default_factory=dict)

    @classmethod
    def load(cls, root: Path, sbm_csv: Path, bias_npz: Path) -> "Resources":
        geo = root / "raw/geo"
        wcvp = root / "raw/wcvp"
        names = pd.read_csv(wcvp / "wcvp_names.csv", sep="|", dtype=str,
                            usecols=["plant_name_id", "taxon_rank", "taxon_status", "family", "taxon_name"])
        acc = names[(names.taxon_status == "Accepted") & (names.taxon_rank == "Species")]
        dist = pd.read_csv(wcvp / "wcvp_distribution.csv", sep="|", dtype=str,
                           usecols=["plant_name_id", "area_code_l3", "introduced", "extinct", "location_doubtful"])
        nat = dist[(dist.introduced == "0") & (dist.extinct == "0") & (dist.location_doubtful == "0")]
        nat = nat.merge(acc[["plant_name_id", "taxon_name", "family"]], on="plant_name_id")
        b = np.load(bias_npz)
        return cls(root=root, stack=GlobalStack(root / "work/global30s/worldclim30s_stack.f32"),
                   wgsrpd=gpd.read_file(geo / "wgsrpd_level3.geojson"),
                   land=gpd.read_file(geo / "ne_land/ne_10m_land.shp"),
                   lakes=gpd.read_file(geo / "ne_lakes/ne_10m_lakes.shp"),
                   ecoregions=gpd.read_file(geo / "ecoregions2017/Ecoregions2017.shp")[["ECO_ID", "ECO_NAME", "geometry"]],
                   wcvp_native=nat, sbm=pd.read_csv(sbm_csv),
                   bias={k: b[k] for k in b.files},
                   fine=_load_fine(root), soil=_load_soil(root))

    def family_range(self, family: str):
        if family not in self._family_cache:
            codes = self.wcvp_native.loc[self.wcvp_native.family == family, "area_code_l3"].unique()
            # Morphological closing (+/- 0.02°): WGSRPD neighbours do not share vertices exactly, so a plain
            # union leaves hairline slivers along borders that would cut 240 m lines through every hull.
            parts = self.wgsrpd[self.wgsrpd.LEVEL3_COD.isin(codes)].geometry.values
            self._family_cache[family] = unary_union([g.buffer(0.02) for g in parts]).buffer(-0.02)
        return self._family_cache[family]

    def km_per_year(self, family: str) -> tuple[float, str]:
        row = self.sbm[self.sbm.clade == family]
        if len(row):
            return float(row.km_per_year.iloc[0]), family
        return float(self.sbm.loc[self.sbm.clade == "ALL", "km_per_year"].iloc[0]), "ALL"

    def cell_bias(self, rr: np.ndarray, cc: np.ndarray, chunk: int = 4_000_000) -> np.ndarray:
        """``bias_at`` the centres of ~1 km cells, computed in chunks (calibration areas reach ~2e8 cells)."""
        out = np.empty(len(rr), self.bias["grid"].dtype)
        for i in range(0, len(rr), chunk):
            xy = GlobalStack.cell_centres(rr[i:i + chunk], cc[i:i + chunk])
            out[i:i + chunk] = self.bias_at(xy[:, 0], xy[:, 1])
        return out

    def bias_at(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        x, y = Transformer.from_crs(4326, background.BEHRMANN, always_xy=True).transform(lon, lat)
        # PROJ signals a failed transformation with inf, which would silently become an edge cell of the grid
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise RuntimeError("projection to the bias grid returned non-finite coordinates")
        res = float(self.bias["res"])
        c = np.clip(((x - float(self.bias["x0"])) / res).astype(int), 0, self.bias["grid"].shape[1] - 1)
        r = np.clip(((float(self.bias["y1"]) - y) / res).astype(int), 0, self.bias["grid"].shape[0] - 1)
        return self.bias["grid"][r, c]


def _load_fine(root: Path):
    p = root / "work/fine/fine_7p5s_na.i16"
    if not p.exists():
        return None
    from .fine import FineStack
    return FineStack(p)


def _load_soil(root: Path):
    d = root / "work/soil"
    from . import soil
    if not all((d / f"{v}_na7p5s.i16").exists() for v in soil.VARS):
        return None
    return soil.SoilPoints(d)


def buffer_km(geom, km: float):
    """Buffer a lon/lat geometry by ``km`` in a local azimuthal equidistant projection."""
    if km <= 0:
        return geom
    c = geom.centroid
    aeqd = CRS.from_proj4(f"+proj=aeqd +lat_0={c.y} +lon_0={c.x} +datum=WGS84 +units=m")
    fwd = Transformer.from_crs(4326, aeqd, always_xy=True).transform
    inv = Transformer.from_crs(aeqd, 4326, always_xy=True).transform
    return shp_transform(inv, shp_transform(fwd, geom).buffer(km * 1000.0))


def _buffer_points(df: pd.DataFrame, res: "Resources", km: float, total: int, seed: int) -> pd.DataFrame:
    """Random points on land within ``km`` of the records, added until ``total`` points exist."""
    from shapely.geometry import MultiPoint
    area = buffer_km(MultiPoint(list(zip(df.decimalLongitude, df.decimalLatitude))), km)
    area = area.intersection(unary_union(res.land.cx[area.bounds[0]:area.bounds[2], area.bounds[1]:area.bounds[3]].geometry.values))
    rng = np.random.default_rng(seed + 11)
    minx, miny, maxx, maxy = area.bounds
    need, pts = total - len(df), []
    while need > 0 and not area.is_empty:
        x, y = rng.uniform(minx, maxx, 4 * need), rng.uniform(miny, maxy, 4 * need)
        keep = vectorized.contains(area, x, y)
        pts += list(zip(x[keep], y[keep]))[:need]
        need = total - len(df) - len(pts)
    add = pd.DataFrame(pts, columns=["decimalLongitude", "decimalLatitude"])
    add["species"] = df.species.iloc[0] if "species" in df else None
    out = pd.concat([df.assign(source="record"), add.assign(source="buffer")], ignore_index=True)
    return out


LARGE_AREA_CELLS = 30_000_000


@contextlib.contextmanager
def _large_area_slot(geom):
    """At most LARGE_AREA_SLOTS (environment; 0 or unset = no limit) worker processes at a time hold the per-cell
    arrays of a calibration area whose bounding box exceeds LARGE_AREA_CELLS ~1 km cells (circumboreal species reach
    ~2e8 cells and ~8 GB), coordinated through lock files, so that many workers fit in a fixed memory budget."""
    n = int(os.environ.get("LARGE_AREA_SLOTS", "0"))
    minx, miny, maxx, maxy = geom.bounds
    if n <= 0 or (maxx - minx) * (maxy - miny) / CELL ** 2 < LARGE_AREA_CELLS:
        yield
        return
    lockdir = Path(os.environ.get("LARGE_AREA_LOCKDIR", "/tmp"))
    while True:
        for i in range(n):
            fh = open(lockdir / f"deepearth_large_area_slot_{i}.lock", "w")
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                fh.close()
                continue
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
                fh.close()
            return
        time.sleep(1.0)


def calibration_cells(geom, res: Resources) -> tuple[np.ndarray, np.ndarray]:
    """Row/col of every ~1 km cell whose centre lies in ``geom`` (lakes excluded)."""
    minx, miny, maxx, maxy = geom.bounds
    r0, c0 = GlobalStack.rowcol(minx, maxy)
    r1, c1 = GlobalStack.rowcol(maxx, miny)
    h, w = int(r1 - r0 + 1), int(c1 - c0 + 1)
    tr = from_origin(-180.0 + c0 * CELL, 90.0 - r0 * CELL, CELL, CELL)
    lakes = res.lakes.cx[minx:maxx, miny:maxy]
    shapes = [(mapping(geom), 1)] + [(mapping(g), 0) for g in lakes.geometry.values]
    m = rasterize(shapes, out_shape=(h, w), transform=tr, fill=0, dtype="uint8")
    # by blocks of rows, as int32: np.nonzero over the whole mask makes two int64 arrays, 16 bytes per cell, and
    # circumboreal calibration areas hold ~2e8 cells (same cells, same row-major order)
    rr, cc = [], []
    for i in range(0, h, 1024):
        r_, c_ = np.nonzero(m[i:i + 1024])
        rr.append((r_ + (i + r0)).astype(np.int32)); cc.append((c_ + c0).astype(np.int32))
    return np.concatenate(rr), np.concatenate(cc)


@dataclass
class Cleaned:
    """CoordinateCleaner flags of one species' records in both cleaning orders (provenance D11)."""
    is_native: pd.Series               # record inside the species' WCVP native L3 areas
    all_records: pd.DataFrame          # every record, cleaned together (Daru's order), with cc_valid
    native_records: pd.DataFrame       # the native records cleaned alone, with cc_valid


def clean_species(items: list[tuple[str, pd.DataFrame]], res: Resources) -> list[Cleaned]:
    """Daru step 2b-c for several (WCVP name, GBIF records) pairs: native-area filter, then CoordinateCleaner on all
    records and on the native records alone, every record set in one R session (``clean_coordinates_many``)."""
    natives, frames = [], []
    for name, gbif in items:
        l3 = sorted(res.wcvp_native.loc[res.wcvp_native.taxon_name == name, "area_code_l3"].unique())
        is_native = occurrences.native_filter(gbif, l3, res.wgsrpd)
        natives.append(is_native)
        frames += [gbif, gbif[is_native]]
    flagged = occurrences.clean_coordinates_many(frames)
    return [Cleaned(n, flagged[2 * i], flagged[2 * i + 1]) for i, n in enumerate(natives)]


def run_species(name: str, gbif: pd.DataFrame, res: Resources, workdir: Path, seed: int = 0,
                presence_mode: str = "hull", max_occurrences: int = 5000, predictors: str = "worldclim",
                background_mode: str = "bias", prepare_only: bool = False, cleaned: "Cleaned | None" = None) -> dict:
    """``name`` is the WCVP accepted name (native areas, family); ``gbif`` holds that species' GBIF records,
    selected by the caller under GBIF's own name, which differs from WCVP's for ~3% of species.

    prepare_only: stop before the range hull and MaxEnt. Everything a model fitted to the species' presences and
    background needs is written exactly as a full run writes it (cleaned records, calibration ecoregions, the
    presences and bias-weighted background with their predictors in maxent/samples.csv and maxent/background.csv,
    the VIF selection), and summary.json gains ``"stage": "prepared"`` (a full run's summary holds ``beta``). Only
    for presence_mode "occurrences": the hull is used by no step of that mode (calibration = whole occupied
    ecoregions, D8), so its products are identical to the full run's.

    cleaned: this species' entry of ``clean_species`` when the caller cleaned several species in one R session;
    computed here otherwise (same result).

    presence_mode: "hull" = Daru (2024): 500 points on a regular lattice over the alpha hull (default);
    "occurrences" = the cleaned, native, ~1 km-thinned records themselves (at most ``max_occurrences``, seeded
    random sample), see future ledger L8.

    predictors: "worldclim" = Daru (2024): WorldClim 2.1 bio1-19 + elevation at ~1 km; "fine" = plus
    Copernicus-derived terrain and lapse-rate-corrected temperatures at ~230 m (``fine.py``, ledger L3/L10);
    points outside the fine domain (North/Central America) are dropped and counted.

    background_mode: "bias" = Daru's text: 10,000 cells drawn with probability proportional to the sampling-bias
    KDE; "uniform" = what his published models used (phyloregion ``sdm`` lets ``predicts::MaxEnt`` draw a
    uniform random background; provenance D9)."""
    t0 = time.time()
    if prepare_only and presence_mode != "occurrences":
        raise ValueError("prepare_only needs presence_mode 'occurrences' (hull presences need the hull)")
    workdir.mkdir(parents=True, exist_ok=True)
    s = {"species": name}
    nat = res.wcvp_native[res.wcvp_native.taxon_name == name]
    if nat.empty:
        raise ValueError(f"{name}: no WCVP native distribution")
    if gbif.empty:
        raise ValueError(f"{name}: no GBIF records under any name that resolves to it in WCVP")
    family = nat.family.iloc[0]
    s.update(family=family, native_l3=sorted(nat.area_code_l3.unique()), n_gbif=len(gbif))

    # step 2: cleaning, native filter, thinning
    # Daru's order: CoordinateCleaner on all records, then the native-area filter. Its outlier test is relative
    # to the species' own records, so for species recorded mostly where they are planted (Abies procera: 79%
    # of records in Europe) it flags the native cluster itself. Cleaning only the native records is used
    # instead when Daru's order keeps under half as many native records (cleaning_order_study.py: the orders
    # differ by < 4% for 58 of 60 random species; Abies procera 0 vs 1,274, Yucca gloriosa 0 vs 180).
    c = cleaned or clean_species([(name, gbif)], res)[0]
    is_native = c.is_native
    df = c.all_records
    s["n_coordinatecleaner_valid"] = int(df.cc_valid.sum())
    df = df[df.cc_valid & is_native]
    s["n_native_daru_order"] = len(df)
    s["cleaning_order"] = "daru"
    if is_native.any():
        first = c.native_records
        s["n_native_native_first"] = int(first.cc_valid.sum())
        if len(df) < 0.5 * s["n_native_native_first"]:
            df, s["cleaning_order"] = first[first.cc_valid], "native_first"
    s["n_native"] = len(df)
    df = occurrences.grid_thin(df, CELL, seed=seed)
    s["n_thinned"] = len(df)
    if 1 <= len(df) < 5:
        # Daru's rule for very few localities (provenance D3): Fig. S1 adds a 30 km buffer for N <= 5, and
        # phyloregion's sdm() tops a species up to `size` = 50 points sampled inside a buffer around its records.
        # Here: points drawn uniformly on land within 30 km of the records until 50 presences exist (seeded); they
        # are flagged source = "buffer" and used throughout, as in Daru's code.
        df = _buffer_points(df, res, km=30.0, total=50, seed=seed)
        s["n_buffer_points"] = int((df.source == "buffer").sum())
    df.to_csv(workdir / "occurrences_clean.csv.gz", index=False)
    if len(df) < 5:
        raise ValueError(f"{name}: only {len(df)} cleaned native records")

    # step 3: alpha hull, land, family range. Only hull presences use it: in mode "occurrences" no step reads the
    # hull (calibration = whole occupied ecoregions, D8), so it is not built (it cost up to ~25 min per widespread
    # species in rangeBuilder; national run, provenance 2026-10-04)
    if presence_mode == "hull":
        hull, alpha = occurrences.alpha_hull(df, res.land)
        from shapely import make_valid
        hull = make_valid(hull).intersection(make_valid(res.family_range(family)))   # antimeridian-crossing hulls
        s["alpha"] = alpha

    # step 4: dispersal rate and occupied ecoregions
    km, clade = res.km_per_year(family)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(df.decimalLongitude, df.decimalLatitude), crs=4326)
    occ_eco = gpd.sjoin(pts, res.ecoregions, predicate="within").ECO_ID.unique()
    # Calibration area = the whole ecoregions occupied by the species' records. Daru's text reads as a
    # geometric intersection of the buffered hull with those ecoregions, but his published rasters match
    # whole occupied ecoregions (Echinacea purpurea footprint IoU 0.83 vs 0.47; provenance D8). The dispersal
    # buffer (~1-4 km/yr) does not change the footprint and is kept only in the provenance record.
    calib = unary_union(res.ecoregions[res.ecoregions.ECO_ID.isin(occ_eco)].geometry.values)
    s.update(sbm_clade=clade, dispersal_km_per_year=km, n_ecoregions=len(occ_eco),
             calibration_ecoregions=sorted(int(i) for i in occ_eco))
    if presence_mode == "hull":
        gpd.GeoDataFrame({"part": ["hull"]}, geometry=[hull], crs=4326).to_file(workdir / "hull.gpkg", driver="GPKG")

    # step 6a: presences
    if presence_mode == "hull":                     # Daru: 500 regular points from the hull at predictor resolution
        pres_xy = occurrences.regular_hull_sample(hull, CELL, 500)
    elif presence_mode == "occurrences":
        o = df.sample(min(len(df), max_occurrences), random_state=seed)
        pres_xy = o[["decimalLongitude", "decimalLatitude"]].values
    else:
        raise ValueError(f"unknown presence_mode {presence_mode!r}")
    s["presence_mode"] = presence_mode
    pres_env = res.stack.at(pres_xy[:, 0], pres_xy[:, 1])
    ok = np.isfinite(pres_env).all(1)
    pres_xy, pres_env = pres_xy[ok], pres_env[ok]

    # step 5: background in the calibration area, weighted by sampling bias
    with _large_area_slot(calib):
        rr, cc = calibration_cells(calib, res)
        if background_mode == "bias":
            weights = res.cell_bias(rr, cc)
        elif background_mode == "uniform":
            weights = np.ones(len(rr))
        else:
            raise ValueError(f"unknown background_mode {background_mode!r}")
        idx = background.sample_background_index(len(rr), weights, 10_000, seed, inplace=True)
        del weights
        bg_xy = GlobalStack.cell_centres(rr[idx], cc[idx])
        n_cells = len(rr)
        if background_mode != "uniform":            # the uniform mode draws its evaluation background from them later
            del rr, cc
    if predictors == "soil":
        # fine layers vary inside a 30" cell: place each background point uniformly within its cell
        bg_xy = bg_xy + np.random.default_rng(seed + 7).uniform(-CELL / 2, CELL / 2, bg_xy.shape)
    s["background_mode"] = background_mode
    bg_env = res.stack.at(bg_xy[:, 0], bg_xy[:, 1])
    ok = np.isfinite(bg_env).all(1)
    bg_xy, bg_env = bg_xy[ok], bg_env[ok]
    s.update(n_presence=len(pres_xy), n_calibration_cells_30s=n_cells, n_background=len(bg_xy))

    names = list(res.stack.variables)
    if predictors == "fine":
        from . import fine
        pres_env, names_f = fine.fine_predictors(pres_env, names, res.fine.at(pres_xy[:, 0], pres_xy[:, 1]))
        bg_env, _ = fine.fine_predictors(bg_env, names, res.fine.at(bg_xy[:, 0], bg_xy[:, 1]))
        okp, okb = np.isfinite(pres_env).all(1), np.isfinite(bg_env).all(1)
        s.update(n_presence_outside_fine_domain=int((~okp).sum()), n_background_outside_fine_domain=int((~okb).sum()))
        pres_xy, pres_env, bg_xy, bg_env, names = pres_xy[okp], pres_env[okp], bg_xy[okb], bg_env[okb], names_f
    elif predictors == "soil":
        from . import soil
        pres_env = np.column_stack([pres_env, res.soil.at(pres_xy[:, 0], pres_xy[:, 1])])
        bg_env = np.column_stack([bg_env, res.soil.at(bg_xy[:, 0], bg_xy[:, 1])])
        okp, okb = np.isfinite(pres_env).all(1), np.isfinite(bg_env).all(1)
        s.update(n_presence_without_soil=int((~okp).sum()), n_background_without_soil=int((~okb).sum()))
        pres_xy, pres_env, bg_xy, bg_env, names = pres_xy[okp], pres_env[okp], bg_xy[okb], bg_env[okb], names + soil.VARS
    elif predictors == "decomposed":
        # each WorldClim variable split into its ~8 km neighbourhood mean and the point's deviation from it (L16)
        fp, fb = res.stack.focal(pres_xy[:, 0], pres_xy[:, 1]), res.stack.focal(bg_xy[:, 0], bg_xy[:, 1])
        pres_env = np.column_stack([fp, pres_env - fp])
        bg_env = np.column_stack([fb, bg_env - fb])
        okp, okb = np.isfinite(pres_env).all(1), np.isfinite(bg_env).all(1)
        pres_xy, pres_env, bg_xy, bg_env = pres_xy[okp], pres_env[okp], bg_xy[okb], bg_env[okb]
        names = ["focal9_" + n for n in names] + ["dev9_" + n for n in names]
    elif predictors == "multiscale":
        # each WorldClim variable at the point and as its ~8 km neighbourhood mean (ledger L16)
        fn = ["focal9_" + n for n in names]
        pres_env = np.column_stack([pres_env, res.stack.focal(pres_xy[:, 0], pres_xy[:, 1])])
        bg_env = np.column_stack([bg_env, res.stack.focal(bg_xy[:, 0], bg_xy[:, 1])])
        okp, okb = np.isfinite(pres_env).all(1), np.isfinite(bg_env).all(1)
        pres_xy, pres_env, bg_xy, bg_env, names = pres_xy[okp], pres_env[okp], bg_xy[okb], bg_env[okb], names + fn
    elif predictors != "worldclim":
        raise ValueError(f"unknown predictors {predictors!r}")
    s["predictor_set"] = predictors

    # step 6b: VIF screening on background predictor values
    bg_df = pd.DataFrame(bg_env, columns=names)
    keep = modelling.vifstep(bg_df, 5.0, seed=seed)
    s["predictors"] = keep

    # step 6c: MaxEnt
    label = name.replace(" ", "_")
    idx = [names.index(v) for v in keep]
    if prepare_only:
        modelling.write_swd_inputs(workdir / "maxent", label, pres_xy, pd.DataFrame(pres_env[:, idx], columns=keep),
                                   bg_xy, pd.DataFrame(bg_env[:, idx], columns=keep))
        s.update(stage="prepared", seconds=round(time.time() - t0, 1))
        _write_summary(workdir, s)
        return s
    fit = modelling.fit_species(workdir / "maxent", label, pres_xy, pd.DataFrame(pres_env[:, idx], columns=keep),
                                bg_xy, pd.DataFrame(bg_env[:, idx], columns=keep))
    if background_mode == "uniform":
        # phyloregion sdm(): MaxEnt fits its own uniform background, but the model is evaluated against the
        # bias-weighted background Daru supplied; keep that set so his metrics are reproducible (evaluate.py).
        ev_idx = background.sample_background_index(len(rr), res.cell_bias(rr, cc), 10_000, seed + 1, inplace=True)
        ev_xy = GlobalStack.cell_centres(rr[ev_idx], cc[ev_idx])
        ev = res.stack.at(ev_xy[:, 0], ev_xy[:, 1])
        if predictors == "fine":
            ev, _ = fine.fine_predictors(ev, list(res.stack.variables), res.fine.at(ev_xy[:, 0], ev_xy[:, 1]))
        elif predictors == "soil":
            ev = np.column_stack([ev, res.soil.at(ev_xy[:, 0], ev_xy[:, 1])])
        elif predictors == "multiscale":
            ev = np.column_stack([ev, res.stack.focal(ev_xy[:, 0], ev_xy[:, 1])])
        elif predictors == "decomposed":
            fe = res.stack.focal(ev_xy[:, 0], ev_xy[:, 1])
            ev = np.column_stack([fe, ev - fe])
        ok = np.isfinite(ev).all(1)
        e = pd.DataFrame(ev[ok][:, idx], columns=keep)
        e.insert(0, "y", ev_xy[ok, 1]); e.insert(0, "x", ev_xy[ok, 0]); e.insert(0, "species", "background")
        e.to_csv(workdir / "maxent/evaluation_background.csv", index=False)
    s.update(beta=fit.beta, cv_auc=fit.cv_auc, seconds=round(time.time() - t0, 1))
    _write_summary(workdir, s)
    return s


def _write_summary(workdir: Path, s: dict) -> None:
    tmp = workdir / "summary.json.tmp"          # written last and atomically: its presence marks a finished species
    tmp.write_text(json.dumps(s, indent=1, default=str))
    os.sync()                                   # everything written above reaches disk before the summary marks it done
    tmp.replace(workdir / "summary.json")
