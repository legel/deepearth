"""
Hurricane Ian Flood Simulation — CFX SR417 Corridor
====================================================
2D raster-based flood model (LISFLOOD-FP) for Hurricane Ian (2022-09-28/30).
Single historical case — no design-storm scenarios, no return periods.

Precipitation input: ASOS MCO hourly (Sep 26–Oct 1 2022, UTC).
  Sep 28 total: 73.5 mm  |  Sep 29 total: 263 mm (peak 66.8 mm/hr at 06:00 UTC)
  Sep 30 total: 0.0 mm   |  Total: 336 mm over 72 hours

Calibration: NWIS 02263800 Shingle Creek peak 3,500 cfs / 11.43 ft (2022-09-30)
(gauge is 7.2 km south; calibration is indirect — AOI boundary outflow alignment)

Physics: Bates et al. (2010) local inertia SWE (LISFLOOD-FP) + Horton infiltration

Usage:
    python3 simulation/flood_sim_ian.py                    # run, save outputs
    python3 simulation/flood_sim_ian.py --save-frames      # + save animation frames
    python3 simulation/flood_sim_ian.py --dry-run          # check setup only
    python3 simulation/flood_sim_ian.py --cell-size 5 --dt 20 --save-frames
"""

import os, sys, json, time, argparse, struct, warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.features import shapes as rio_shapes
from PIL import Image
import geopandas as gpd
from shapely.geometry import shape

warnings.filterwarnings("ignore")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR  = os.path.dirname(BASE_DIR)
OUT_DIR   = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

DEM_COND     = os.path.join(PROJ_DIR, "dem", "data", "hydro", "dem_conditioned.tif")
# 2026-07-21 alignment fix: dem_conditioned.tif (derived from the independently-downloaded 1m
# DEM) has a real, measured ~1.3-1.6m bounding-box offset on every edge relative to the raw 3m
# DEM below — the two were fetched from USGS 3DEP as separate requests and snapped to slightly
# different native tile grids. viewer/preprocess/export_dem.py's geo_meta.json (the canonical
# scene-space reference every other viewer layer aligns to — terrain, NAIP, point clouds) is
# built from THIS file, not dem_conditioned.tif. The solver itself still runs on the
# hydro-conditioned DEM (that's the right surface for D8/breaching), but the final export grid
# must use these bounds so the Ian flood-depth texture lines up with everything else when
# draped onto the terrain.
SR417_DEM_RAW = os.path.join(PROJ_DIR, "dem", "data", "sr417_corridor_dem.tif")
# Downsampling operator used when the solver grid is coarser than the conditioned DEM
# (load_dem_for_sim below). NOT cosmetic — measured on site3, 0.875m conditioned DEM -> 5m grid:
#
#   operator    trapped depression storage after resampling
#   ---------   -------------------------------------------
#   bilinear         3.710e6 m^3   (79.4 mm domain-average, 20.3% of the Ian storm total)
#   average          3.614e6 m^3   (77.3 mm)
#   nearest          4.604e6 m^3   (98.5 mm)
#   min              0.000e6 m^3   (0.0 mm)
#
# The native conditioned DEM has exactly zero trapped volume — richdem's breaching carves
# least-cost drainage paths one cell (~0.9m) wide. Any averaging kernel blends that trench back
# into the surrounding high ground at 5m, re-sealing every outlet the conditioning opened and
# silently reintroducing ~a full storm's worth of depression storage. min() is the standard
# channel-preserving choice for hydrologic downsampling: a coarse cell containing a drainage
# path should convey at that path's elevation, since conveyance is set by the lowest flow line
# through the cell, not its mean surface. Cost is a downward elevation bias (mean 0.134 m,
# median 0.091 m against 27.9 m of relief) and an unchanged median slope (1.830% -> 1.848%).
DEM_RESAMPLING = Resampling.min

SOIL_JSON    = os.path.join(PROJ_DIR, "soil", "data", "soil_parameters.json")
MUKEY_MAP    = os.path.join(PROJ_DIR, "soil", "data", "mukey_map.tif")
MUKEY_LEGEND = os.path.join(PROJ_DIR, "soil", "data", "mukey_map_legend.csv")
SOIL_STORAGE_CSV = os.path.join(PROJ_DIR, "soil", "data", "soil_storage.csv")
ROADS_PATH     = os.path.join(PROJ_DIR, "infrastructure", "data", "roads.geojson")
BUILDINGS_PATH = os.path.join(PROJ_DIR, "infrastructure", "data", "buildings.geojson")
NLCD_IMPERVIOUS_PATH = os.path.join(PROJ_DIR, "soil", "data", "nlcd_impervious.tif")
# Road buffer widths, gravity and the friction exponent come from the shared physics module,
# so the "impervious" mask used by the solver is by construction the same footprint
# viewer/preprocess/export_overlays.py draws.
_PROGRAM_DIR = os.path.dirname(PROJ_DIR)   # .../models/flood_hydrology (holds floodtwin/)
if _PROGRAM_DIR not in sys.path:
    sys.path.insert(0, _PROGRAM_DIR)
from floodtwin.physics import (   # noqa: E402
    ROAD_BUFFER_M, ROAD_BUFFER_DEFAULT_M, IMPERVIOUS_FC_MM_HR,
)
ASOS_CSV     = os.path.join(PROJ_DIR, "precipitation", "data", "asos_hourly_MCO.csv")

G          = 9.81
CFL_ALPHA  = 0.15     # Was 0.3, which is UNSTABLE on this terrain once water actually
                      # accumulates. Measured 2026-08-29 at the main AOI over 7,000 steps with
                      # the storm properly delivered: alpha=0.30 gave a -517.8 % mass residual
                      # and h_max of 8.99 m on terrain with 14 m of total relief, with depths
                      # oscillating between 5.5 and 8 m under ZERO rainfall. alpha=0.15 and
                      # alpha=0.05 agree to four figures (-12.6 % residual, h_max 2.03 m), so
                      # the solution is timestep-converged at 0.15 and tightening further only
                      # costs sub-steps. The instability was unreachable before the 2026-08-28
                      # sub-stepping fix because the solver never accumulated enough water.
MIN_DEPTH  = 1e-4     # m wet/dry threshold
DEPTH_THR  = 0.05     # m "flooded" for outputs
MANNING_N  = 0.040    # flatwoods grassland / mixed cover

# Exponent on the flow depth in the Bates et al. (2010) semi-implicit friction denominator:
#     q^(n+1) = [q - g*hf*dt*d(eta)/dx] / [1 + g*dt*n^2*|q| / hf^MANNING_EXP]
# MUST be 7/3 when q is UNIT DISCHARGE (m^2/s), which it is here — the depth update divides
# the flux difference by dx, so q carries m^2/s, not velocity.
#
# Corrected 2026-08-04 from a long-standing 4/3. Two independent confirmations, neither guessed:
#  1. Dimensional analysis — the friction term must be dimensionless:
#       [g][dt][n^2][q] / [hf^p] = (m/s^2)(s)(s^2*m^-2/3)(m^2/s) * m^-p = m^(7/3 - p) * s^0
#     which is dimensionless only at p = 7/3.
#  2. Steady state — setting q^(n+1) = q^n and solving gives q = hf^((p+1)/2) * sqrt(-S)/n.
#     Manning's equation is q = hf^(5/3) * sqrt(-S)/n, so (p+1)/2 = 5/3 -> p = 7/3.
#     Verified numerically by iterating the update to a fixed point on a uniform slope:
#     p=7/3 reproduces Manning to -0.00% at every depth tested (0.02-1.0 m); p=4/3
#     OVER-predicts discharge by +216% at h=0.10 m and +607% at h=0.02 m (the two agree only
#     at h=1 m, where hf^0 = 1 makes the exponent irrelevant).
# The error was therefore worst in exactly this project's normal operating regime — the
# documented median wet depth here is 7-8 cm, p90 ~14 cm (see RESOLUTION_ANALYSIS.md) — and
# under-stated friction, making water route to the domain boundary faster than the specified
# Manning's n implies. That directly affects peak TIMING, which is the metric the Gee Creek
# gauge comparison rests on.
# Both flood_hydrology solvers (flood_sim.py, torch_swe_benchmark.py) already used 7/3
# correctly; the 4/3 was introduced only in this project's own solvers.
MANNING_EXP = 7.0 / 3.0

# AMC-III reduction factor for antecedent-wet B/D spodosol soils.
# Most AOI soils (Immokalee, Ona, Basinger) are HSG B/D — dual-rated because
# their sandy A-horizon Ksat (~9–92 μm/s → 32–331 mm/hr) is fast, but the
# restrictive spodic Bh horizon at 50–80 cm depth limits actual drainage to
# ~0.1–5 μm/s during flood events.  Ian followed 2 weeks of above-avg rain,
# so the profile was pre-saturated (AMC III) and the water table was near the
# surface → effective fc is a small fraction of the tabulated Ksat.
AMC3_FACTOR = 0.07   # effective Ksat fraction for pre-saturated spodosol profile


def _load_horton_params():
    """Load SSURGO-derived Horton params with AMC-III wet-antecedent correction.

    Raw SSURGO fc values represent A-horizon Ksat (dry conditions, HSG B side).
    For Ian (HSG D conditions, pre-saturated), we apply AMC3_FACTOR to bring
    effective fc into the physically realistic 5–15 mm/hr range for spodosols.
    Falls back to hardcoded wet-season preset if soil_parameters.json is absent.
    """
    if os.path.exists(SOIL_JSON):
        with open(SOIL_JSON) as fh:
            soil = json.load(fh)
        units = [v for v in soil.values()
                 if "fc_mm_hr" in v and "Water" not in v.get("muname", "")]
        if units:
            fc_raw  = float(np.mean([v["fc_mm_hr"] for v in units]))
            k_mean  = float(np.mean([v["k_hr"]     for v in units]))
            fc_eff  = fc_raw * AMC3_FACTOR
            f0_eff  = fc_eff * 2.5   # small initial excess for near-saturated conditions
            print(f"  SSURGO Ksat mean={fc_raw:.0f} mm/hr × AMC3={AMC3_FACTOR} "
                  f"→ fc_eff={fc_eff:.1f}  f0_eff={f0_eff:.1f}  k={k_mean:.2f} hr⁻¹")
            return {"f0": round(f0_eff, 1), "fc": round(fc_eff, 1), "k": round(k_mean, 2)}
    print("  WARNING: soil_parameters.json not found — using Ian wet-season preset")
    return {"f0": 76.0, "fc": 25.0, "k": 2.0}


HORTON = _load_horton_params()


# Drainable (air-filled) porosity between field capacity and saturation for the fine sands that
# dominate this landscape. The water a saturating profile can still accept is the depth to the
# seasonal-high water table times this, NOT total porosity — the profile below field capacity is
# already wet. 0.25 is mid-range for fine sand; it is the one tunable in this term.
DRAINABLE_POROSITY = 0.25

# Depth assumed for soils SSURGO reports no water table for (excessively drained, water table
# below the 200 cm observation limit). Deeper than any storm can fill, so the exact value only
# has to be large.
NO_WATER_TABLE_DEPTH_CM = 150.0


def load_soil_storage_capacity(z_shape, dst_transform, dst_crs, storage_csv=None):
    """Per-cell finite soil storage [m] — the "maximum deficit" of the Deficit and Constant method.

    Derived from SSURGO's muaggatt.wtdepannmin (depth to the seasonal-high water table, cm) as

        storage = water_table_depth * DRAINABLE_POROSITY

    which is the standard construction (effective porosity x active layer depth). Depressional
    soils report a water table at the surface and therefore get zero storage: they generate
    runoff immediately, which is correct for them.

    Returns None if the storage table has not been fetched, in which case the solver falls back
    to unbounded infiltration and behaves exactly as before.
    """
    path = storage_csv or SOIL_STORAGE_CSV
    if not (os.path.exists(path) and os.path.exists(MUKEY_MAP) and os.path.exists(MUKEY_LEGEND)):
        return None

    import csv as _csv
    wt_cm = {}
    with open(path, newline="") as fh:
        for row in _csv.DictReader(fh):
            raw = (row.get("wtdepannmin") or "").strip()
            wt_cm[str(row["mukey"])] = float(raw) if raw else NO_WATER_TABLE_DEPTH_CM

    key_to_int = {}
    with open(MUKEY_LEGEND, newline="") as fh:
        for row in _csv.DictReader(fh):
            key_to_int[str(row["mukey"])] = int(row["mukey_int"])

    with rasterio.open(MUKEY_MAP) as src:
        mk = np.zeros(z_shape, dtype=np.float32)
        reproject(src.read(1).astype(np.float32), mk,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=dst_transform, dst_crs=dst_crs,
                  resampling=Resampling.nearest)
    mk = mk.astype(np.int32)

    # Domain mean is the sensible default for cells whose map unit has no storage record.
    known = [wt_cm[k] * 0.01 * DRAINABLE_POROSITY for k in wt_cm]
    out = np.full(z_shape, float(np.mean(known)) if known else 0.0, dtype=np.float32)
    for mukey, ival in key_to_int.items():
        if mukey in wt_cm:
            out[mk == ival] = wt_cm[mukey] * 0.01 * DRAINABLE_POROSITY   # cm -> m
    return out


def load_spatial_horton(z_shape, dst_transform, dst_crs):
    """Build per-cell Horton f0/fc/k (SI units) from mukey_map.tif reprojected onto the sim grid.

    soil/data/mukey_map.tif already exists (per-cell SSURGO map-unit codes) but was previously
    never read by the solver — _load_horton_params() collapsed all 9 units into one scalar mean.
    Returns None (caller falls back to the scalar HORTON dict) if the mukey raster, its legend,
    or soil_parameters.json is missing.
    """
    if not (os.path.exists(MUKEY_MAP) and os.path.exists(MUKEY_LEGEND) and os.path.exists(SOIL_JSON)):
        return None

    with rasterio.open(MUKEY_MAP) as src:
        mukey_native   = src.read(1)
        native_transform = src.transform
        native_crs     = src.crs

    mukey_grid = np.zeros(z_shape, dtype=np.int32)
    reproject(mukey_native, mukey_grid,
              src_transform=native_transform, src_crs=native_crs,
              dst_transform=dst_transform, dst_crs=dst_crs,
              resampling=Resampling.nearest)

    legend = pd.read_csv(MUKEY_LEGEND).set_index("mukey_int")["mukey"].astype(str).to_dict()
    with open(SOIL_JSON) as fh:
        soil = json.load(fh)

    fc_mm_hr = np.full(z_shape, HORTON["fc"], dtype=np.float32)
    f0_mm_hr = np.full(z_shape, HORTON["f0"], dtype=np.float32)
    k_arr    = np.full(z_shape, HORTON["k"],  dtype=np.float32)

    codes = np.unique(mukey_grid)
    for code in codes:
        mukey_str = legend.get(int(code))
        params    = soil.get(mukey_str) if mukey_str else None
        if not params or "fc_mm_hr" not in params:
            continue   # leave the scalar-mean fallback value for unmapped codes
        mask = mukey_grid == code
        fc_mm_hr[mask] = params["fc_mm_hr"] * AMC3_FACTOR
        f0_mm_hr[mask] = fc_mm_hr[mask][0] * 2.5
        k_arr[mask]    = params["k_hr"]

    print(f"  Spatial Horton: {len(codes)} soil units mapped onto {z_shape[0]}×{z_shape[1]} grid  "
          f"(fc_eff range {fc_mm_hr.min():.1f}–{fc_mm_hr.max():.1f} mm/hr)")

    return {
        "f0": f0_mm_hr / 1000 / 3600,
        "fc": fc_mm_hr / 1000 / 3600,
        "k":  k_arr / 3600,
    }


def apply_impervious_mask(horton_arrays, z_shape, dst_transform, dst_crs):
    """Zero out infiltration wherever OSM roads/buildings cover a cell — hard surfaces don't
    infiltrate at all, regardless of the SSURGO soil unit underneath them. This is a real,
    directly-known impervious mask from vector data already on disk (infrastructure/data/),
    not a substitute for the DINOv2/SAM classification work (that's for classifying NAIP
    imagery into surface types generally — this only covers what OSM already maps as roads/
    buildings, which is not the same as "every hard surface," e.g. driveways/patios/parking
    lots not tagged as buildings in OSM are not covered here).
    Returns horton_arrays unchanged if roads/buildings data isn't available yet.
    """
    if horton_arrays is None or not (os.path.exists(ROADS_PATH) and os.path.exists(BUILDINGS_PATH)):
        return horton_arrays

    import geopandas as gpd
    from rasterio.features import rasterize

    roads = gpd.read_file(ROADS_PATH).to_crs(dst_crs)
    buildings = gpd.read_file(BUILDINGS_PATH).to_crs(dst_crs)

    shapes = []
    for _, row in roads.iterrows():
        buf_m = ROAD_BUFFER_M.get(str(row.get("highway")), ROAD_BUFFER_DEFAULT_M)
        shapes.append((row.geometry.buffer(buf_m), 1))
    for geom in buildings.geometry:
        shapes.append((geom, 1))

    mask = rasterize(shapes, out_shape=z_shape, transform=dst_transform, fill=0, dtype=np.uint8)
    impervious = mask.astype(bool)
    n_cells = int(impervious.sum())

    fc_si = IMPERVIOUS_FC_MM_HR / 1000 / 3600
    horton_arrays["fc"] = np.where(impervious, fc_si, horton_arrays["fc"])
    horton_arrays["f0"] = np.where(impervious, fc_si, horton_arrays["f0"])
    print(f"  Impervious mask: {n_cells}/{z_shape[0]*z_shape[1]} cells "
          f"({100*n_cells/(z_shape[0]*z_shape[1]):.1f}%) forced to zero infiltration "
          f"(OSM roads + buildings)")
    return horton_arrays


def apply_nlcd_graded_impervious(horton_arrays, z_shape, dst_transform, dst_crs):
    """Added 2026-07-24: apply_impervious_mask() above is a hard binary cut — 100% or 0%,
    only where OSM explicitly maps a road/building polygon. NLCD's impervious-surface layer
    (soil/data/nlcd_impervious.tif, already fetched, never previously used by either solver)
    gives a continuous 0-100% impervious fraction per 30m cell, catching driveways/parking lots/
    compacted soil near structures that OSM's own polygons miss entirely. Scales infiltration
    CAPACITY (f0/fc) down proportionally by (1 - impervious_fraction) everywhere NOT already
    zeroed by the binary OSM mask above — cells OSM already forced to true impervious keep that
    unchanged (a real road IS 100% impervious regardless of what NLCD's 30m footprint reports for
    that pixel). k (the Horton decay rate, a soil-structure property) is left untouched — imperv-
    iousness reduces how MUCH can infiltrate, not the shape of the decay curve for what still can.
    Returns horton_arrays unchanged if nlcd_impervious.tif isn't available.
    """
    if horton_arrays is None or not os.path.exists(NLCD_IMPERVIOUS_PATH):
        return horton_arrays

    with rasterio.open(NLCD_IMPERVIOUS_PATH) as src:
        nlcd_native = src.read(1)
        native_transform = src.transform
        native_crs = src.crs

    nlcd_grid = np.full(z_shape, np.nan, dtype=np.float32)
    reproject(nlcd_native, nlcd_grid,
              src_transform=native_transform, src_crs=native_crs,
              dst_transform=dst_transform, dst_crs=dst_crs,
              resampling=Resampling.bilinear)

    imperv_frac = np.clip(np.nan_to_num(nlcd_grid, nan=0.0) / 100.0, 0.0, 1.0)
    already_hard = np.isclose(horton_arrays["fc"], IMPERVIOUS_FC_MM_HR / 1000 / 3600)
    grade = np.where(already_hard, 1.0, 1.0 - imperv_frac)

    horton_arrays["fc"] = horton_arrays["fc"] * grade
    horton_arrays["f0"] = horton_arrays["f0"] * grade
    print(f"  NLCD graded impervious: mean {100*imperv_frac[~already_hard].mean():.1f}% "
          f"impervious fraction applied to {int((~already_hard).sum())} non-OSM-masked cells "
          f"(mean infiltration-capacity reduction {100*(1-grade[~already_hard]).mean():.1f}%)")
    return horton_arrays


# Ian storm window in the ASOS MCO CSV (UTC)
IAN_START = "2022-09-28 00:00"
IAN_END   = "2022-09-30 23:00"


# ── Raster I/O ────────────────────────────────────────────────────────────────

def load_raster(path, dtype=np.float32):
    with rasterio.open(path) as src:
        return src.read(1).astype(dtype), src.profile.copy()

def save_raster(arr, profile, path, dtype=np.float32):
    p = profile.copy()
    p.update(count=1, dtype=dtype, compress="lzw", nodata=np.nan)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype(dtype), 1)


# ── Hyetograph ────────────────────────────────────────────────────────────────

def _synthetic_ian_hyetograph(dt_s):
    """Fallback synthetic based on GHCND daily totals."""
    rain_mm = np.zeros(72)
    rain_mm[0:24]  = 14.0 / 24
    rain_mm[24:48] = 345.0 / 24
    rain_mm[48:72] = 47.0  / 24
    hours     = np.arange(72)
    rain_m_s  = rain_mm / 1000 / 3600
    t_sim     = np.arange(0, 72 * 3600, dt_s)
    rain_interp = np.interp(t_sim, hours * 3600.0, rain_m_s)
    print("  Using synthetic Ian hyetograph (GHCND daily fallback)")
    return rain_interp, hours, rain_mm


def load_ian_hyetograph(dt_s):
    """Load ASOS MCO hourly data for Ian window → per-step m/s array."""
    if not os.path.exists(ASOS_CSV):
        return _synthetic_ian_hyetograph(dt_s)

    df = pd.read_csv(ASOS_CSV)
    df.columns = [c.strip().lower() for c in df.columns]
    time_col   = next((c for c in df.columns if 'datetime' in c or 'valid' in c or 'time' in c), df.columns[0])
    precip_col = next((c for c in df.columns if 'prcp' in c or 'p01' in c or 'precip' in c), None)

    if precip_col is None:
        return _synthetic_ian_hyetograph(dt_s)

    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors='coerce')
    df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()

    start = pd.Timestamp(IAN_START, tz='UTC')
    end   = pd.Timestamp(IAN_END,   tz='UTC')
    ian   = df[start:end]

    if ian.empty:
        return _synthetic_ian_hyetograph(dt_s)

    rain_mm  = pd.to_numeric(ian[precip_col], errors='coerce').fillna(0.0).values  # mm/hr
    hours    = np.arange(len(rain_mm))
    rain_m_s = rain_mm / 1000 / 3600

    t_sim    = np.arange(0, len(rain_mm) * 3600, dt_s)
    rain_sim = np.interp(t_sim, hours * 3600.0, rain_m_s, right=0.0)

    print(f"  ASOS MCO Ian: {len(rain_mm)} hrs  "
          f"total={rain_mm.sum():.0f} mm  peak={rain_mm.max():.1f} mm/hr  "
          f"→ {len(rain_sim)} sim steps @ {dt_s}s")
    return rain_sim, hours, rain_mm


# ── Horton infiltration ───────────────────────────────────────────────────────

def horton_rate(t_s, f0_si, fc_si, k_si):
    """Horton decay. f0_si/fc_si/k_si may be scalars (uniform) or 2D arrays (spatial)."""
    return fc_si + (f0_si - fc_si) * np.exp(-k_si * t_s)


# ── DEM loading ───────────────────────────────────────────────────────────────

def load_dem_for_sim(cell_size_m):
    if not os.path.exists(DEM_COND):
        raise FileNotFoundError(f"dem_conditioned.tif not found: {DEM_COND}\n"
                                "Run: python3 dem/dem_hydro.py")
    with rasterio.open(DEM_COND) as src:
        native_res    = abs(src.transform.a)
        native_crs    = src.crs
        native_bounds = src.bounds
        dem_native    = src.read(1).astype(np.float32)
        # `src.nodata or -9999.0` would swap a legitimate nodata value of 0.0 for -9999.0,
        # since 0.0 is falsy — leaving real nodata cells as valid ground at elevation zero.
        nodata        = src.nodata if src.nodata is not None else -9999.0
        profile       = src.profile.copy()

    print(f"  DEM: {dem_native.shape[0]}×{dem_native.shape[1]} @ {native_res:.2f}m  CRS: {native_crs}")

    if cell_size_m <= native_res * 1.1:
        z = dem_native.copy()
        z[z == nodata] = np.nan
        print(f"  Sim grid: {z.shape[0]}×{z.shape[1]} @ {native_res:.2f}m native")
        return z, profile, native_res
    else:
        # Normalize orientation before computing width/height — a standard north-up raster
        # has bounds.bottom < bounds.top, but rasterio's .bounds just reports whatever a
        # raster's own (possibly inverted) affine transform produces without enforcing that
        # ordering. site3's DEM (unlike the original AOI's) has a positive y-resolution
        # (transform.e > 0), so bounds.bottom > bounds.top — the un-normalized subtraction
        # below produced a negative new_h and crashed with "negative dimensions are not
        # allowed" the first time this ran against site3. Same underlying issue already fixed
        # in lidar/droplet_flow_test.py's build_ground_surface() for the mesh-building
        # pipeline; fixed here the same way (min/max instead of assuming an order) since this
        # is a real orientation bug, not site3-specific behavior.
        true_left, true_right = sorted([native_bounds.left, native_bounds.right])
        true_bottom, true_top = sorted([native_bounds.bottom, native_bounds.top])
        new_h = int((true_top - true_bottom) / cell_size_m)
        new_w = int((true_right - true_left) / cell_size_m)
        dst_tf = from_bounds(true_left, true_bottom, true_right, true_top, new_w, new_h)
        z_c = np.zeros((new_h, new_w), dtype=np.float32)
        # src/dst nodata must be declared explicitly. Without them GDAL treats the nodata
        # sentinel as ordinary data: verified that Resampling.min then SILENTLY FILLS a NaN
        # hole with the surrounding minimum (0 NaN cells out of a 36-cell hole) where bilinear
        # correctly propagates it. Neither conditioned DEM on disk has any NaN today, so this
        # changes nothing here — but the pipeline's premise is "any coordinate", and a DEM with
        # holes would otherwise get invented terrain at exactly the elevations that matter most.
        reproject(dem_native, z_c,
                  src_transform=profile["transform"], src_crs=native_crs,
                  dst_transform=dst_tf, dst_crs=native_crs,
                  src_nodata=nodata, dst_nodata=nodata,
                  resampling=DEM_RESAMPLING)
        z_c[z_c == nodata] = np.nan
        prof = profile.copy()
        prof.update(height=new_h, width=new_w, transform=dst_tf)
        print(f"  Sim grid: {new_h}×{new_w} @ {cell_size_m:.1f}m  ({z_c.size/1e3:.1f}k cells)")
        return z_c, prof, float(cell_size_m)


# ── LISFLOOD-FP solver with optional frame capture ────────────────────────────

def run_sim(z, dx, rain_sim, dt_s, frame_interval_min=30, verbose=True,
            use_infiltration=True, horton_arrays=None, max_deficit_m=None, gauge_rc=None,
            initial_h=None, manning_n=None):
    """
    Bates et al. (2010) local inertia solver.
    use_infiltration=False sets Horton inf=0 so Pe=rain (all rain stays on surface).
    horton_arrays: optional {'f0','fc','k'} dict of 2D arrays (SI units) for spatially-varying
    infiltration (see load_spatial_horton); falls back to the scalar global HORTON mean if None.
    max_deficit_m: optional 2D array of finite soil storage [m]. Once a cell's cumulative
    infiltration reaches it, infiltration stops and further rain becomes runoff (saturation
    excess). None reproduces the previous unbounded-infiltration behaviour exactly.
    manning_n: optional 2D array of Manning's n, one value per CELL (averaged onto faces
    internally). None uses the scalar MANNING_N, which is the original code path. Built by
    segmentation/rasterize_parameters.py from SAM-style surface classes over NAIP + a LiDAR
    canopy-height model.
    Returns:
      h_max, cum_infil, flooded_ha_ts, frame_data
      where frame_data = {'frames': [...], 'infil_frames': [...], 'times_min': [...],
      'outflow_south_cms': [...], 'outflow_total_cms': [...]} — per-step domain-boundary
      discharge (m^3/s), added 2026-07-24 as the real "civil engineering hydrograph" signal
      for comparing against the observed NWIS 02263800 gauge record. South-edge-only is the
      physically relevant direction (both Shingle Creek gauges sit south of the AOI); total
      (all 4 edges) is also tracked as a mass-balance sanity check. This is a shape/timing
      comparison, not a magnitude one: the gauge's contributing watershed (231 km^2) is ~44x
      this AOI's own area (5.24 km^2).
    """
    nrows, ncols = z.shape
    valid = np.isfinite(z)

    if use_infiltration and horton_arrays is not None:
        f0_si, fc_si, k_si = horton_arrays["f0"], horton_arrays["fc"], horton_arrays["k"]
    elif use_infiltration:
        f0_si = HORTON["f0"] / 1000 / 3600
        fc_si = HORTON["fc"] / 1000 / 3600
        k_si  = HORTON["k"]  / 3600
    else:
        f0_si = fc_si = k_si = 0.0

    z_work = z.copy()
    z_work[~valid] = np.nanmax(z_work) + 100.0

    # Initial water. Default (None) starts the whole domain bone-dry, which is wrong for any
    # perennial channel and became the dominant error once stream burning gave site3 a real
    # carved channel: the creek was carrying 45.2 cfs of baseflow when Ian made landfall, i.e.
    # it was already full, whereas the model made 72 h of runoff fill 22 km of empty channel
    # first. Measured, that channel held 63 % of everything passing the gauge all event, and
    # gauge-cell discharge rose monotonically for the entire run without ever peaking.
    h     = np.zeros_like(z_work) if initial_h is None else initial_h.astype(np.float32).copy()
    if initial_h is not None:
        h[~valid] = 0.0
        print(f"  Initial water: {float(h.sum()) * dx * dx / 1e3:.1f} thousand m³ "
              f"in {int((h > 0).sum()):,} cells (mean depth {float(h[h > 0].mean()):.3f} m)")
    initial_volume_m3 = float(h.sum()) * dx * dx
    qx    = np.zeros((nrows, ncols + 1), dtype=np.float32)
    qy    = np.zeros((nrows + 1, ncols), dtype=np.float32)

    h_max     = np.zeros_like(h)
    cum_infil = np.zeros_like(h)

    flooded_ha_ts   = []    # one entry per sim step
    outflow_south_cms_ts = []   # m^3/s leaving the south domain edge only — the direction
                                 # both Shingle Creek gauges sit in; the signal to compare
                                 # shape/timing against (not magnitude — the gauge's
                                 # watershed is ~44x this AOI, see run_sim's docstring)
    outflow_total_cms_ts = []   # m^3/s leaving all 4 domain edges combined — mass-balance check
    gauge_cms_ts = []           # m^3/s through the gauge cell itself, when one is given.
                                # Domain-boundary outflow is NOT what a stream gauge measures:
                                # the gauge sits INSIDE the domain, so boundary outflow charges
                                # the comparison for the extra travel time from the gauge out
                                # to the box edge. Measured on site3 that is the difference
                                # between a +5 h and a +29 h lag against an observed +4.5 h,
                                # entirely from which cross-section is read — not from physics.
    rain_mm_hr_ts   = []
    Pe_mm_hr_ts     = []
    infil_mm_hr_ts  = []
    mean_depth_ts   = []    # instantaneous mean depth of flooded cells per step

    frame_depths    = []   # depth snapshots at frame_interval_min intervals
    frame_infil     = []   # cumulative infiltration snapshots [m]
    frame_times_min = []

    # Manning's n at each FACE, not each cell: the friction term below acts on the flux across
    # a cell boundary, so it needs the roughness the water actually crosses. Averaging the two
    # adjacent cells is the standard face reconstruction, and it keeps the array shapes matching
    # hf_x (nrows, ncols-1) and hf_y (nrows-1, ncols) exactly.
    #
    # manning_n=None takes the scalar branch, which is the ORIGINAL expression unchanged — every
    # existing caller and every previously-recorded run is bit-identical. Passing an array is
    # what segmentation/rasterize_parameters.py produces (Track A): the promotion of MANNING_N
    # from a single domain-wide constant to a measured spatial field.
    #
    # A uniform array is NOT bit-identical to the scalar, and the reason is numpy casting, not
    # physics: a float64 SCALAR times a float32 array stays float32 (value-based casting), while
    # a float64 ARRAY times a float32 array promotes to float64, so the two round differently.
    # Measured on a 60x60 test grid over 200 steps: max depth difference 8.9e-08 m (90 nm),
    # outflow-volume difference 1.1e-07 relative. Negligible, but stated rather than assumed —
    # and it is why the A/B runs below use the scalar branch for the baseline, so the baseline
    # is the untouched code path rather than a re-derivation of it.
    if manning_n is None:
        n2_x = n2_y = MANNING_N ** 2
    else:
        # float64: the friction denominator is the one place n enters, and carrying the face
        # values at full precision costs 15 MB at site3's grid size. See the note above for why
        # this still does not reproduce the scalar branch bit-for-bit.
        mn = np.asarray(manning_n, dtype=np.float64)
        if mn.shape != z.shape:
            raise ValueError(f"manning_n shape {mn.shape} != grid shape {z.shape}")
        n2_x = (0.5 * (mn[:, :-1] + mn[:, 1:])) ** 2
        n2_y = (0.5 * (mn[:-1, :] + mn[1:, :])) ** 2
    cell_ha = dx * dx / 1e4
    n_steps = len(rain_sim)
    frame_interval_s = frame_interval_min * 60

    t0  = time.time()
    t_s = 0.0
    n_substeps = 0
    substep_cap_hits = 0
    last_frame_t = -1e9    # force a frame at t=0

    # Sub-stepping. The physics is integrated with a CFL-limited dt, and one hyetograph
    # interval (dt_s long) generally needs SEVERAL such steps. This loop previously took a
    # single CFL-limited step and then advanced the clock by the full dt_s, so the model
    # applied `dt` worth of rain and routing while recording `dt_s` of elapsed time.
    #
    # Measured 2026-08-28 at site3, infiltration off: CFL bound 94.6 % of steps (median real
    # dt 2.97 s against dt_s = 60 s), a nominal 72-hour run integrated 9.59 h of physics, and
    # 7.4 % of the specified rain ever entered the domain. At the production 5 m / dt_s = 20 s
    # configuration it was 11 %. The solver's numerics were never at fault — recomputing the
    # mass balance on the real dt closes to 0.1 %, and a synthetic sloped domain conserves
    # exactly. The clock was wrong, so the storm was never delivered.
    #
    # The fix is to advance to each interval's end with as many CFL-limited sub-steps as it
    # takes. Cost is real: a correct run takes roughly dt_s/dt times more steps.
    SUBSTEP_CAP = 20000      # guard against dt collapsing; exceeded => report, do not hang

    for step in range(n_steps):
        P   = rain_sim[step]
        t_target = t_s + dt_s

        # The per-step series are RATES that get integrated downstream against a dt_s-spaced
        # axis, so each must be this interval's TIME-WEIGHTED MEAN, not whatever the final
        # sub-step happened to leave behind.
        acc_out_s = acc_out_t = acc_gauge = 0.0
        acc_Pe = acc_inf = 0.0
        sub_elapsed = 0.0
        n_sub = 0

        while t_s < t_target - 1e-9 and n_sub < SUBSTEP_CAP:
            h_max_local = float(h.max())
            dt = min(dt_s, CFL_ALPHA * dx / np.sqrt(G * h_max_local)) \
                 if h_max_local > MIN_DEPTH else dt_s
            # Never overshoot the hyetograph interval: the last sub-step is trimmed to land
            # exactly on t_target, so each interval receives exactly dt_s of its own rain rate
            # and the clock stays aligned with the forcing.
            dt = min(dt, t_target - t_s)

            inf = horton_rate(t_s, f0_si, fc_si, k_si) if use_infiltration else 0.0

            # Storage-limited infiltration (saturation excess).
            #
            # Horton alone gives a RATE that decays to fc and then stays there indefinitely, so a
            # cell can absorb water forever. Over a 72-hour event at site3's fc_eff of 23.3 mm/hr
            # that is 1,678 mm of capacity against 392 mm of rain, and essentially all rainfall
            # infiltrates: the simulated runoff coefficient came out at 1.0% against 19.6% measured
            # at the Gee Creek gauge for Hurricane Ian.
            #
            # Real soil has finite storage. Once the profile fills, infiltration stops and further
            # rain becomes runoff — saturation excess, which dominates over infiltration excess on
            # flat terrain with a shallow water table, i.e. exactly this landscape.
            #
            # This is the "maximum deficit" of HEC-RAS/HEC-HMS's Deficit and Constant loss method,
            # conventionally taken as effective porosity x active layer depth and then calibrated.
            # Here max_deficit_m is derived per cell from SSURGO (depth to seasonal-high water table
            # x drainable porosity); see load_soil_storage_capacity().
            if use_infiltration and max_deficit_m is not None:
                remaining = np.maximum(max_deficit_m - cum_infil, 0.0)
                inf = np.minimum(inf, remaining / dt)   # cannot infiltrate more than is left

            Pe  = np.maximum(P - inf, 0.0)

            eta = z_work + h

            # X fluxes
            hf_x    = np.maximum(eta[:, 1:], eta[:, :-1]) - np.maximum(z_work[:, 1:], z_work[:, :-1])
            hf_x    = np.maximum(hf_x, 0.0)
            num_x   = qx[:, 1:-1] - G * hf_x * dt * (eta[:, 1:] - eta[:, :-1]) / dx
            denom_x = 1.0 + G * dt * n2_x * np.abs(qx[:, 1:-1]) / (hf_x ** MANNING_EXP + 1e-10)
            qx[:, 1:-1] = np.where(hf_x > MIN_DEPTH, num_x / denom_x, 0.0)
            # Froude limiter: cap unit discharge to subcritical (Fr ≤ 0.9) at each face.
            # Prevents supercritical flow instability at steep road-embankment cells.
            q_cap_x = 0.9 * hf_x * np.sqrt(G * np.maximum(hf_x, MIN_DEPTH))
            qx[:, 1:-1] = np.clip(qx[:, 1:-1], -q_cap_x, q_cap_x)

            # Y fluxes
            hf_y    = np.maximum(eta[1:, :], eta[:-1, :]) - np.maximum(z_work[1:, :], z_work[:-1, :])
            hf_y    = np.maximum(hf_y, 0.0)
            num_y   = qy[1:-1, :] - G * hf_y * dt * (eta[1:, :] - eta[:-1, :]) / dx
            denom_y = 1.0 + G * dt * n2_y * np.abs(qy[1:-1, :]) / (hf_y ** MANNING_EXP + 1e-10)
            qy[1:-1, :] = np.where(hf_y > MIN_DEPTH, num_y / denom_y, 0.0)
            q_cap_y = 0.9 * hf_y * np.sqrt(G * np.maximum(hf_y, MIN_DEPTH))
            qy[1:-1, :] = np.clip(qy[1:-1, :], -q_cap_y, q_cap_y)

            # Open (transmissive) boundary conditions — allow outflow at all domain edges.
            # Water exits when it reaches the grid boundary; no inflow from outside.
            # This physically represents the 2×2km AOI being embedded in a larger watershed
            # rather than a closed box.  Sign convention: qx > 0 = eastward, qy > 0 = southward.
            qx[:, 0]  = np.minimum(qx[:, 1],  0.0)   # west edge: outflow (westward) only
            qx[:, -1] = np.maximum(qx[:, -2], 0.0)   # east edge: outflow (eastward) only
            qy[0, :]  = np.minimum(qy[1, :],  0.0)   # north edge: outflow (northward) only
            qy[-1, :] = np.maximum(qy[-2, :], 0.0)   # south edge: outflow (southward) only

            # Boundary discharge (m^3/s) — q here is unit discharge (m^2/s per Bates convention),
            # so multiplying each face's flux by dx (the cell width that face spans) and summing
            # across the edge gives the actual volumetric rate crossing it. All four terms made
            # positive (leaving), regardless of each edge's own sign convention above.
            out_west  = float(-qx[:, 0].sum())  * dx
            out_east  = float( qx[:, -1].sum()) * dx
            out_north = float(-qy[0, :].sum())  * dx
            out_south = float( qy[-1, :].sum()) * dx
            outflow_south_ts_step = out_south
            outflow_total_ts_step = out_west + out_east + out_north + out_south

            # Depth update
            h += dt / dx * (qx[:, :-1] - qx[:, 1:] + qy[:-1, :] - qy[1:, :]) + dt * Pe
            h  = np.maximum(h, 0.0)
            h[~valid] = 0.0

            # Charge the soil only for water that ACTUALLY infiltrated. `inf` is a capacity
            # RATE; the water available to satisfy it is the rain falling this sub-step, since
            # Pe = max(P - inf, 0) draws infiltration from rainfall only. Charging the full
            # capacity regardless of supply let the profile fill without absorbing anything:
            # over site3's first ~8 h Ian delivers <1 mm/hr while Horton offers 25-58 mm/hr, so
            # cum_infil accrued ~194 mm of PHANTOM infiltration and hit the 206 mm cap before
            # the storm arrived. Infiltration then stopped for the rest of the event — measured
            # 2.9 % of rain infiltrating against the ~50 % the cap implies, and a 92.8 % runoff
            # coefficient against an observed 28.9-31.4 %.
            #
            # This was invisible until the sub-stepping fix of 2026-08-28: the clock previously
            # ran ~10x ahead of the physics, so cum_infil accrued over a fraction of the real
            # time and never reached the cap. One bug was masking the other.
            cum_infil += np.minimum(inf, P) * dt
            h_max = np.maximum(h_max, h)

            # advance the clock by what was ACTUALLY integrated
            t_s += dt
            sub_elapsed += dt
            n_sub += 1
            n_substeps += 1

            acc_out_s += outflow_south_ts_step * dt
            acc_out_t += outflow_total_ts_step * dt
            acc_Pe    += float(np.mean(Pe[valid]) if isinstance(Pe, np.ndarray) else Pe) * dt
            acc_inf   += float(np.mean(inf[valid]) if isinstance(inf, np.ndarray) else inf) * dt
            if gauge_rc is not None:
                gr, gc = gauge_rc
                gqx = 0.5 * (qx[gr, gc] + qx[gr, gc + 1])
                gqy = 0.5 * (qy[gr, gc] + qy[gr + 1, gc])
                acc_gauge += float(np.hypot(gqx, gqy)) * dx * dt

        if n_sub >= SUBSTEP_CAP:
            substep_cap_hits += 1
        w = sub_elapsed if sub_elapsed > 0 else 1.0
        outflow_south_ts_step = acc_out_s / w
        outflow_total_ts_step = acc_out_t / w

        n_flooded = int((h[valid] > DEPTH_THR).sum())
        flooded_ha_ts.append(n_flooded * cell_ha)
        outflow_south_cms_ts.append(outflow_south_ts_step)
        outflow_total_cms_ts.append(outflow_total_ts_step)
        if gauge_rc is not None:
            gauge_cms_ts.append(acc_gauge / w)
        rain_mm_hr_ts.append(P * 3600 * 1000)
        # Pe/inf may be per-cell arrays under spatial infiltration — log the domain mean.
        Pe_mm_hr_ts.append(acc_Pe / w * 3600 * 1000)
        infil_mm_hr_ts.append(acc_inf / w * 3600 * 1000)
        wet_h = h[valid][h[valid] > DEPTH_THR]
        mean_depth_ts.append(float(wet_h.mean()) if len(wet_h) else 0.0)

        # ── Frame snapshot ────────────────────────────────────────────────────
        if t_s - last_frame_t >= frame_interval_s:
            frame_depths.append(h.copy().astype(np.float32))
            frame_infil.append(cum_infil.copy().astype(np.float32) * 1000)  # m → mm
            frame_times_min.append(t_s / 60.0)
            last_frame_t = t_s

        if verbose and step % 360 == 0:
            print(f"  t={t_s/3600:.1f}h  rain={rain_mm_hr_ts[-1]:.1f}mm/hr  "
                  f"Pe={Pe_mm_hr_ts[-1]:.1f}mm/hr  "
                  f"depth_max={float(h.max()):.3f}m  "
                  f"flooded={n_flooded*cell_ha:.1f}ha  "
                  f"[{(step+1)/n_steps*100:.0f}% {time.time()-t0:.0f}s]")


    # Always capture a final frame
    # Always land a frame on the true end of the run. The old guard
    #     if frame_times_min and abs(frame_times_min[-1] - t_s/60) > frame_interval_min/2
    # skipped the final frame whenever the run ended less than half an interval after the last
    # periodic one, and captured NOTHING at all when frame_interval_min exceeded the run length
    # (the leading `and` short-circuits on an empty list). Either way frames[-1] is then a
    # snapshot from BEFORE the end, which silently breaks any mass balance computed against it:
    # outflow integrated to the end vs storage sampled earlier reads as created mass. That cost
    # real diagnostic time on 2026-08-29 — it produced a convincing -12.6 % "structural mass
    # error" that did not exist.
    if (not frame_times_min) or frame_times_min[-1] < t_s / 60.0 - 1e-9:
        frame_depths.append(h.copy().astype(np.float32))
        frame_infil.append(cum_infil.copy().astype(np.float32) * 1000)
        frame_times_min.append(t_s / 60.0)

    return (
        h_max, cum_infil,
        np.array(flooded_ha_ts),
        np.array(rain_mm_hr_ts),
        np.array(Pe_mm_hr_ts),
        np.array(mean_depth_ts),
        {
            "frames":      frame_depths,
            "infil_frames": frame_infil,
            "times_min":   frame_times_min,
            "outflow_south_cms": np.array(outflow_south_cms_ts),
            "outflow_total_cms": np.array(outflow_total_cms_ts),
            "gauge_cms": np.array(gauge_cms_ts) if gauge_rc is not None else None,
            "initial_volume_m3": initial_volume_m3,
        },
    )


# ── SIML binary writer (same format as flood_hydrology) ──────────────────────

TARGET = 256   # SIML output resolution (viewer uses 256×256 DataTexture)

def _downsample_frame(arr, profile, dem_bounds):
    """Bilinear downsample a 2D float32 array to TARGET×TARGET in DEM extent."""
    dst_tf = from_bounds(dem_bounds.left, dem_bounds.bottom,
                         dem_bounds.right, dem_bounds.top, TARGET, TARGET)
    dst = np.zeros((TARGET, TARGET), dtype=np.float32)
    reproject(arr, dst,
              src_transform=profile["transform"], src_crs=profile["crs"],
              dst_transform=dst_tf, dst_crs=profile["crs"],
              resampling=Resampling.bilinear)
    return np.maximum(dst, 0.0)


def write_siml_bin(out_path, frames, profile, dem_bounds, times_min_arr):
    """Write SIML binary: magic + header + times + downsampled frames."""
    n = len(frames)
    small = np.stack([_downsample_frame(f, profile, dem_bounds) for f in frames])
    with open(out_path, "wb") as fh:
        fh.write(b"SIML")
        fh.write(struct.pack("<I", n))
        fh.write(struct.pack("<I", TARGET))
        fh.write(struct.pack("<I", TARGET))
        fh.write(np.array(times_min_arr, dtype=np.float32).tobytes())
        fh.write(small.astype(np.float32).tobytes())
    kb = os.path.getsize(out_path) / 1024
    print(f"  {os.path.basename(out_path)}: {n} frames × {TARGET}×{TARGET}  ({kb:.0f} KB)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-size", type=float, default=5.0,
                    help="Simulation grid resolution [m]  (default 5m)")
    ap.add_argument("--dt", type=float, default=20.0,
                    help="Max simulation timestep [s]  (default 20s)")
    ap.add_argument("--frame-interval", type=float, default=30.0,
                    help="Minutes between animation frames  (default 30)")
    ap.add_argument("--save-frames", action="store_true",
                    help="Save per-timestep depth frames for viewer animation")
    ap.add_argument("--dem-resample", default="min",
                    choices=["min", "bilinear", "average", "nearest"],
                    help="Downsampling operator for the conditioned DEM (default min). "
                         "See DEM_RESAMPLING above — averaging kernels destroy the breaching.")
    ap.add_argument("--no-infiltration", action="store_true",
                    help="Disable Horton infiltration — Pe = rain (water stays on surface)")
    ap.add_argument("--uniform-infiltration", action="store_true",
                    help="Use the single global-mean Horton params instead of the per-cell "
                         "SSURGO-derived mukey_map.tif (legacy behavior)")
    ap.add_argument("--no-soil-storage", action="store_true",
                    help="Disable the finite soil-storage cap, restoring the previous "
                         "unbounded-infiltration behaviour (for comparison runs)")
    ap.add_argument("--no-impervious-mask", action="store_true",
                    help="Don't force zero infiltration under OSM roads/buildings "
                         "(legacy behavior: hard surfaces get soil-derived infiltration too)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print setup info and exit")
    args = ap.parse_args()

    global DEM_RESAMPLING
    DEM_RESAMPLING = getattr(Resampling, args.dem_resample)

    use_infiltration = not args.no_infiltration
    tag = "_noinfil" if not use_infiltration else ("_uniforminfil" if args.uniform_infiltration else "")

    print("=" * 62)
    print("Hurricane Ian Flood Simulation — CFX SR417 Corridor")
    print("=" * 62)
    print(f"  Solver: LISFLOOD-FP local inertia  (Bates et al. 2010)")
    print(f"  Manning n={MANNING_N}  (flatwoods mixed cover)")

    # [1] Load DEM
    print(f"\n[1/4] DEM  (target cell size = {args.cell_size}m) …")
    z, profile, dx = load_dem_for_sim(args.cell_size)

    # Bounds for the FINAL export grid — deliberately the raw 3m DEM's bounds (same file
    # export_dem.py's geo_meta.json uses), not dem_conditioned.tif's own bounds. See
    # SR417_DEM_RAW's comment above for why these two files' bounds differ.
    with rasterio.open(SR417_DEM_RAW) as src:
        dem_bounds = src.bounds

    # Horton infiltration params — spatial (per-cell SSURGO mukey) by default, or the legacy
    # single global mean if --uniform-infiltration is passed or mukey_map.tif is unavailable.
    horton_arrays = None
    if use_infiltration:
        if args.uniform_infiltration:
            print(f"  Horton: uniform (SSURGO mean)  "
                  f"f0={HORTON['f0']} mm/hr  fc={HORTON['fc']} mm/hr  k={HORTON['k']} hr⁻¹")
        else:
            horton_arrays = load_spatial_horton(z.shape, profile["transform"], profile["crs"])
            if horton_arrays is None:
                print(f"  Horton: mukey_map.tif not found — falling back to uniform mean  "
                      f"f0={HORTON['f0']} mm/hr  fc={HORTON['fc']} mm/hr  k={HORTON['k']} hr⁻¹")
            elif not args.no_impervious_mask:
                horton_arrays = apply_impervious_mask(horton_arrays, z.shape,
                                                       profile["transform"], profile["crs"])
                horton_arrays = apply_nlcd_graded_impervious(horton_arrays, z.shape,
                                                              profile["transform"], profile["crs"])
    else:
        print(f"  Infiltration: DISABLED — all rain becomes runoff (Pe = P)")

    # [2] Hyetograph
    print(f"\n[2/4] Ian hyetograph (ASOS MCO hourly, {IAN_START} – {IAN_END} UTC) …")
    rain_sim, hours, rain_mm = load_ian_hyetograph(args.dt)
    total_rain = rain_mm.sum()
    n_steps    = len(rain_sim)
    print(f"  Sep 28: {rain_mm[:24].sum():.0f} mm  "
          f"Sep 29: {rain_mm[24:48].sum():.0f} mm  "
          f"Sep 30: {rain_mm[48:].sum():.0f} mm  "
          f"total={total_rain:.0f} mm")
    print(f"  Simulation: {n_steps} steps × {args.dt:.0f}s = {n_steps*args.dt/3600:.1f} hrs")
    if args.save_frames:
        expected_frames = int(n_steps * args.dt / 60 / args.frame_interval) + 2
        print(f"  Frame snapshots: every {args.frame_interval:.0f} min → ~{expected_frames} frames")

    if args.dry_run:
        print("\nDry run — exiting.")
        return

    # [3] Solver
    print(f"\n[3/4] Running solver …")
    t0 = time.time()
    # Finite soil storage, so infiltration stops once the profile fills (saturation excess).
    # Absent the SSURGO storage table this stays None and behaviour is unchanged.
    max_deficit_m = None
    if use_infiltration and not getattr(args, "no_soil_storage", False):
        max_deficit_m = load_soil_storage_capacity(z.shape, profile["transform"], profile["crs"])
        if max_deficit_m is not None:
            print(f"  Soil storage cap: mean {1000*float(max_deficit_m.mean()):.0f} mm, "
                  f"range {1000*float(max_deficit_m.min()):.0f}-{1000*float(max_deficit_m.max()):.0f} mm, "
                  f"{100*float((max_deficit_m == 0).mean()):.0f}% of cells depressional (zero storage)")
        else:
            print("  Soil storage table absent — infiltration UNBOUNDED "
                  "(run soil/fetch_soil_storage.py to enable the cap)")

    h_max, cum_infil, flooded_ha_ts, rain_ts, Pe_ts, mean_depth_ts, frame_data = run_sim(
        z, dx, rain_sim, args.dt,
        frame_interval_min=args.frame_interval,
        use_infiltration=use_infiltration,
        horton_arrays=horton_arrays,
        max_deficit_m=max_deficit_m,
    )
    elapsed = time.time() - t0

    peak_ha = float(flooded_ha_ts.max())
    peak_h  = float(h_max.max())
    print(f"\n  Solver: {elapsed:.0f}s  |  peak depth={peak_h:.3f}m  |  peak flooded={peak_ha:.1f}ha")
    print(f"  Calibration: NWIS 02263800 peak = 3500 cfs / 11.43 ft (2022-09-30)")

    # [4] Outputs
    print(f"\n[4/4] Saving outputs →  simulation/outputs/ …")
    step_hrs = np.arange(n_steps) * args.dt / 3600.0

    # Hydrograph CSV (column names match flood_hydrology export_simulation.py)
    CMS_TO_CFS = 35.3147
    outflow_south_cms = frame_data["outflow_south_cms"]
    outflow_total_cms = frame_data["outflow_total_cms"]
    hydro_df = pd.DataFrame({
        "time_min":    step_hrs * 60,
        "rain_mm_hr":  rain_ts,
        "Pe_mm_hr":    Pe_ts,
        "flooded_ha":  flooded_ha_ts,
        "lake_rise_m": 0.0,      # no lake — keep column for API compatibility
        "mean_depth_m": mean_depth_ts,
        # Added 2026-07-24 — the real "civil engineering hydrograph" signal for comparing
        # shape/timing (NOT magnitude — see the module-level run_sim docstring for why) against
        # the observed NWIS 02263800 gauge record.
        "outflow_south_cms": outflow_south_cms,
        "outflow_south_cfs": outflow_south_cms * CMS_TO_CFS,
        "outflow_total_cms": outflow_total_cms,
        "outflow_total_cfs": outflow_total_cms * CMS_TO_CFS,
    })
    hydro_path = os.path.join(OUT_DIR, "hydrograph_ian.csv")
    hydro_df.to_csv(hydro_path, index=False)
    print(f"  hydrograph_ian.csv  ({n_steps} rows)")
    peak_outflow_south_cfs = float(hydro_df["outflow_south_cfs"].max())
    peak_outflow_idx = int(hydro_df["outflow_south_cfs"].idxmax())
    peak_outflow_time_hr = float(hydro_df["time_min"].iloc[peak_outflow_idx]) / 60.0
    print(f"  Simulated south-edge outflow peak: {peak_outflow_south_cfs:.1f} cfs "
          f"at t={peak_outflow_time_hr:.1f}h into the sim window "
          f"(AOI is only ~1/44th of gauge 02263800's watershed area — compare SHAPE/TIMING "
          f"against the observed record, not this magnitude directly)")

    # Peak inundation TIF
    save_raster(h_max, profile, os.path.join(OUT_DIR, "inundation_depth_ian.tif"))
    print(f"  inundation_depth_ian.tif")

    # Infiltration TIF
    save_raster(cum_infil, profile, os.path.join(OUT_DIR, "infiltration_ian.tif"))
    print(f"  infiltration_ian.tif")

    # Flood extent GeoJSON
    flooded_mask = (h_max > DEPTH_THR) & np.isfinite(z)
    geoms = [(shape(g), int(v)) for g, v in
             rio_shapes(flooded_mask.astype(np.uint8),
                        mask=flooded_mask.astype(np.uint8),
                        transform=profile["transform"]) if int(v) == 1]
    if geoms:
        gdf = gpd.GeoDataFrame(
            [{"event":"Ian","date":"2022-09-30","peak_depth_m":round(peak_h,3),
              "peak_ha":round(peak_ha,1)} for _ in geoms],
            geometry=[g for g,_ in geoms], crs=profile["crs"]
        ).dissolve().to_crs("epsg:4326")
        gdf.to_file(os.path.join(OUT_DIR, "flood_extent_ian.geojson"), driver="GeoJSON")
        print(f"  flood_extent_ian.geojson")

    # Viewer PNG (static snapshot)
    SIZE = 512
    dst_tf = from_bounds(dem_bounds.left, dem_bounds.bottom,
                         dem_bounds.right, dem_bounds.top, SIZE, SIZE)
    h_sm = np.zeros((SIZE, SIZE), dtype=np.float32)
    reproject(h_max, h_sm,
              src_transform=profile["transform"], src_crs=profile["crs"],
              dst_transform=dst_tf, dst_crs=profile["crs"],
              resampling=Resampling.max)
    rgba = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
    wet  = h_sm > DEPTH_THR
    norm = np.clip(h_sm / max(peak_h, 0.5), 0, 1)
    rgba[wet, 0] = (20  + norm[wet] * 40).astype(np.uint8)
    rgba[wet, 1] = (80  + norm[wet] * 60).astype(np.uint8)
    rgba[wet, 2] = (180 + norm[wet] * 75).astype(np.uint8)
    rgba[wet, 3] = (140 + norm[wet] * 80).astype(np.uint8)
    Image.fromarray(rgba).save(os.path.join(OUT_DIR, "ian_flood_viewer.png"))
    print(f"  ian_flood_viewer.png  (512×512 static peak)")

    # SIML animation frames
    if args.save_frames and frame_data["frames"]:
        n_frames = len(frame_data["frames"])
        print(f"\n  Saving {n_frames} animation frames …")

        write_siml_bin(
            os.path.join(OUT_DIR, f"depth_frames_ian{tag}.bin"),
            frame_data["frames"], profile, dem_bounds,
            frame_data["times_min"],
        )

        # Hydrograph JSON for viewer (per frame, not per step)
        f_idxs = [min(int(tm * 60 / args.dt), n_steps - 1) for tm in frame_data["times_min"]]
        frame_max_depths = [
            float(f[f > DEPTH_THR].max()) if (f > DEPTH_THR).any() else 0.0
            for f in frame_data["frames"]
        ]
        scenario_id = f"ian{tag}"
        hydro_json = {
            "scenario_id":    scenario_id,
            "use_infiltration": use_infiltration,
            "times_min":      frame_data["times_min"],
            "rain_mm_hr":     [float(rain_ts[i]) for i in f_idxs],
            "Pe_mm_hr":       [float(Pe_ts[i])   for i in f_idxs],
            "flooded_ha":     [float(flooded_ha_ts[i]) for i in f_idxs],
            "lake_rise_m":    [0.0] * n_frames,
            "max_depth_m":    frame_max_depths,
            "data_source":    "ASOS MCO · Orlando Intl Airport · 14 km",
            "total_rain_mm":  round(total_rain, 1),
            "peak_flooded_ha": round(peak_ha, 2),
            "peak_lake_rise_m": 0.0,
        }
        jpath = os.path.join(OUT_DIR, f"simulation_ian{tag}_hydrograph.json")
        with open(jpath, "w") as f:
            json.dump(hydro_json, f, indent=2)
        print(f"  simulation_ian{tag}_hydrograph.json  ({n_frames} frames)")

        # Summary JSON
        summary = {
            "id":              scenario_id,
            "label":           "Hurricane Ian · Sep 28–30 2022",
            "data_source":     "ASOS MCO · 336 mm total",
            "use_infiltration": use_infiltration,
            "total_rain_mm":   round(total_rain, 1),
            "peak_flooded_ha": round(peak_ha, 2),
            "peak_lake_rise_m": 0.0,
            "n_frames":        n_frames,
            "frame_interval_min": args.frame_interval,
        }
        with open(os.path.join(OUT_DIR, f"ian{tag}_sim_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    print(f"\n══ COMPLETE ══════════════════════════════════════════════")
    print(f"  Peak depth:   {peak_h:.3f} m")
    print(f"  Peak flooded: {peak_ha:.1f} ha  ({peak_ha/400*100:.1f}% of 4 km² AOI)")
    if args.save_frames and frame_data["frames"]:
        print(f"  Animation:    {len(frame_data['frames'])} frames @ {args.frame_interval:.0f}min")


if __name__ == "__main__":
    main()
