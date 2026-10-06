"""Occurrence preparation following Daru (2024) steps 1–3 and 6a.

1. GBIF records (SIMPLE_PARQUET downloads, read by scripts/run_fit.py).
2. Reconcile names to WCVP accepted species; clean coordinates with CoordinateCleaner (R); keep records that
   fall inside the species' WCVP native TDWG level-3 areas; thin species with >= 5 unique localities to one
   record per predictor cell (dismo::gridSample with n = 1).
3. Alpha hull (rangeBuilder, R), cropped to land.
6a. Regular sample of the hull at predictor resolution, thinned to 500 points.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import vectorized
from shapely.ops import unary_union

R_DIR = Path(__file__).parent / "r"


def rscript(script: str, *args: str, env: str | None = None) -> str:
    """Run one of the package's R scripts inside the conda env that holds Daru's R toolchain."""
    conda_sh = os.environ.get("CONDA_SH", "~/miniconda3/etc/profile.d/conda.sh")
    env = env or os.environ.get("R_ENV", "daru")
    cmd = f"source {conda_sh} && conda activate {env} && Rscript {R_DIR / script} " + \
          " ".join(f"'{a}'" for a in args)
    res = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{script} failed:\n{res.stderr[-2000:]}")
    return res.stdout


def clean_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """CoordinateCleaner::clean_coordinates (Daru's tests, inst_rad = 100 m). Returns df with a ``cc_valid`` flag.
    Set CC_SEAS_REF to a local Natural Earth 50 m land shapefile to avoid the run-time download of the sea test."""
    return clean_coordinates_many([df])[0]


def clean_coordinates_many(frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """``clean_coordinates`` for several record sets in one R session: each set is cleaned by its own
    CoordinateCleaner call (identical flags to separate sessions), and R and its reference layers load once."""
    out = [f.assign(cc_valid=np.zeros(len(f), bool)) for f in frames]
    jobs = [i for i, f in enumerate(frames) if len(f)]
    if not jobs:
        return out
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.csv", Path(tmp) / "out.csv"
        pd.concat([frames[i][["species", "decimalLongitude", "decimalLatitude"]].assign(group=i) for i in jobs],
                  ignore_index=True).to_csv(src, index=False)
        rscript("clean_coordinates.R", str(src), str(dst))
        kept = pd.read_csv(dst)[".summary"].values.astype(bool)
    k = 0
    for i in jobs:
        out[i]["cc_valid"] = kept[k:k + len(frames[i])]
        k += len(frames[i])
    return out


def native_filter(df: pd.DataFrame, native_l3: list[str], wgsrpd: gpd.GeoDataFrame) -> pd.Series:
    """True for records inside the union of the species' WCVP native L3 areas."""
    area = unary_union(wgsrpd[wgsrpd.LEVEL3_COD.isin(native_l3)].geometry.values)
    return pd.Series(vectorized.contains(area, df.decimalLongitude.values, df.decimalLatitude.values), index=df.index)


def grid_thin(df: pd.DataFrame, cell_deg: float, seed: int = 0) -> pd.DataFrame:
    """One record per predictor cell (dismo::gridSample, n = 1), applied when there are >= 5 unique localities."""
    xy = df[["decimalLongitude", "decimalLatitude"]].round(6).drop_duplicates()
    if len(xy) < 5:
        return df.loc[xy.index]
    cell = (np.floor((df.decimalLongitude + 180) / cell_deg).astype(np.int64) * 10**7 +
            np.floor((df.decimalLatitude + 90) / cell_deg).astype(np.int64))
    return df.assign(_c=cell).sample(frac=1, random_state=seed).drop_duplicates("_c").drop(columns="_c")


HULL_MAX_POINTS = 20_000


def alpha_hull(df: pd.DataFrame, land: gpd.GeoDataFrame) -> tuple:
    """rangeBuilder::getDynamicAlphaHull (initialAlpha = 2, fraction = 0.99), cropped to land.

    rangeBuilder builds an all-pairs structure, so very large record sets (e.g. circumboreal species: ~280k
    points needed 624 GB) are thinned on progressively coarser grids (~1.8 km, ~4.6 km, ~9 km — Daru's predictor scale,
    ...) until at most HULL_MAX_POINTS remain; the presences used for modelling are unaffected."""
    cell = 1 / 60
    while len(df) > HULL_MAX_POINTS:
        df = grid_thin(df, cell)
        cell *= 2.5
    from shapely import make_valid
    try:
        hull, alpha = _robust_hull(df, cell)
    except RuntimeError as e:
        lon = df.decimalLongitude
        if not ("TopologyException" in str(e) and (lon > 150).any() and (lon < -150).any()):
            raise
        # Circumboreal species: a hull crossing the antimeridian is invalid in longitude/latitude (Deschampsia
        # cespitosa fails at 179.7°E, 68.6°N). Hull three groups separately, cut where no land joins them: the
        # Americas (169°W–30°W), the Old World (30°W–180°) and far-eastern Chukotka (180°–169°W). Coordinates are
        # never shifted past ±180°: the Old World set shifted to 0–191° failed with degenerate rings in all six
        # attempts (~25 min each), unshifted it succeeds in 140 s.
        groups = [(lon >= -169) & (lon < -30), lon >= -30, lon < -169]
        parts, alpha = [], np.nan
        for k, g in enumerate(groups):
            if g.sum() >= 5:
                h, a_ = _robust_hull(df[g], cell)
                parts.append(h)
                if k == 0:
                    alpha = a_                         # the Americas hull is the one the CONUS product uses
        hull = unary_union(parts)
    geom = make_valid(hull).intersection(unary_union(land.geometry.values))
    return geom, alpha


def _robust_hull(df: pd.DataFrame, cell: float, tries: int = 3) -> tuple:
    """``_rangebuilder``, retried on progressively coarser grid thinning when GEOS rejects a degenerate ring that
    jitter does not remove (Deschampsia cespitosa, Old World: "LinearRing found 2 points" inside rangeBuilder's
    buffer). Thinning is the step ``alpha_hull`` already applies to large record sets; it changes the
    triangulation and shortens each attempt."""
    for k in range(tries):
        try:
            return _rangebuilder(df)
        except RuntimeError as e:
            if "IllegalArgumentException" not in str(e) or k == tries - 1:
                raise
            df = grid_thin(df, cell)
            cell *= 2.5


def _rangebuilder(df: pd.DataFrame) -> tuple:
    """One rangeBuilder hull with the recoverable failures handled (see ``alpha_hull``)."""
    from shapely import make_valid
    xy = df[["decimalLongitude", "decimalLatitude"]]
    engine = []
    jittered = False
    for attempt in range(3):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "pts.csv", Path(tmp) / "hull.geojson"
            xy.to_csv(src, index=False)
            try:
                log = rscript("alpha_hull.R", str(src), str(dst), *engine)
            except RuntimeError as e:
                if attempt == 2:
                    raise
                if "s2_geography" in str(e) and not engine:
                    # s2 (sf's spherical engine) rejects some self-touching hull loops of very large point sets;
                    # planar GEOS accepts them
                    engine = ["planar"]
                    continue
                if "IllegalArgumentException" not in str(e) or jittered:
                    raise
                jittered = True
                # GEOS rejects a degenerate ring (collinear or coincident vertices) that the triangulation produced
                # for some point sets; a seeded jitter of ~1 m breaks the degeneracy without moving any record
                # out of its 30" (~1 km) predictor cell.
                xy = df[["decimalLongitude", "decimalLatitude"]] + np.random.default_rng(attempt).uniform(-1e-5, 1e-5, (len(df), 2))
                continue
            hull = gpd.read_file(dst).to_crs(4326)
            break
    num = re.findall(r"[-+]?\d*\.?\d+", log.split("alpha", 1)[-1]) if "alpha" in log else []
    alpha = float(num[0]) if num else np.nan          # rangeBuilder reports e.g. "alpha2"
    return make_valid(unary_union(make_valid(hull.geometry.values))), alpha


def regular_hull_sample(hull, cell_deg: float, n: int = 500) -> np.ndarray:
    """Cell centres of the hull rasterized at predictor resolution, regularly thinned to ``n`` points."""
    minx, miny, maxx, maxy = hull.bounds
    xs = np.arange(np.floor(minx / cell_deg) * cell_deg + cell_deg / 2, maxx, cell_deg)
    ys = np.arange(np.floor(miny / cell_deg) * cell_deg + cell_deg / 2, maxy, cell_deg)
    step = 1
    while True:                                         # coarsen the lattice until <= n cells remain
        gx, gy = np.meshgrid(xs[::step], ys[::step])
        inside = vectorized.contains(hull, gx.ravel(), gy.ravel())
        if inside.sum() <= n or step > 10_000:
            break
        step = max(step + 1, int(step * np.sqrt(inside.sum() / n)))
    pts = np.column_stack([gx.ravel()[inside], gy.ravel()[inside]])
    if len(pts) < n and step > 1:                       # refine between the last two lattices to reach n
        gx2, gy2 = np.meshgrid(xs[:: step - 1], ys[:: step - 1])
        in2 = vectorized.contains(hull, gx2.ravel(), gy2.ravel())
        cand = np.column_stack([gx2.ravel()[in2], gy2.ravel()[in2]])
        idx = np.linspace(0, len(cand) - 1, min(n, len(cand))).round().astype(int)
        pts = cand[idx]
    return pts
