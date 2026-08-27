#!/usr/bin/env python3
"""
Flood PROBABILITY from a design-storm ensemble  —  P(flood | x, y, z, t, Δt, C) ∈ [0, 1]
=========================================================================================

Why this exists
---------------
Every solver in this project produces a DETERMINISTIC DEPTH FIELD for one storm: "given this
rainfall, this cell ends up 0.28 m deep." That is a hydraulic answer, not a risk answer.

The DeepEarth Flood Risk API proposal (§1) specifies a fundamentally different output type:

    P(flood | x, y, z, t, Δt, 𝓒) ∈ [0, 1]

— a probability, over a stated time horizon Δt. Nothing in either project produced that. This
module closes that gap using the standard hydrologic/actuarial route (design-storm ensemble →
annual exceedance probability), so the output TYPE now matches the proposal even though the
learned foundation model that would eventually replace this does not exist yet.

Method (each step is standard practice, not invented here)
----------------------------------------------------------
1. NOAA Atlas 14 gives, for this exact coordinate, the rainfall DEPTH for a given duration at
   each return period T ∈ {1, 2, 5, 10, 25, 50, 100, 200, 500} years. A return period T has by
   definition an annual exceedance probability AEP = 1/T.

2. Run the calibrated physics solver once per return period, driven by that period's SCS Type II
   design hyetograph. This yields, per cell, a peak-depth curve  h_T(x, y)  that is monotonically
   increasing in T (a rarer storm never floods a cell less).

3. INVERT that curve per cell. For a flood threshold d*, the annual exceedance probability of
   flooding at that cell is the AEP of the smallest storm that reaches d*:

       AEP(x, y) = 1 / T*    where T* = min{ T : h_T(x, y) ≥ d* }

   Between the simulated return periods we interpolate depth against log(T) — the standard
   assumption, since Atlas 14 depth is close to linear in log return period (verified for this
   AOI in `_report_loglinearity()` below, printed at run time rather than asserted).

4. Convert to the proposal's time horizon Δt. Treating annual exceedances as independent
   Bernoulli trials (the standard assumption behind "1% annual chance" language):

       P(at least one flood in N years) = 1 − (1 − AEP)^N

   So Δt enters exactly where the proposal puts it, and Δt = 1 recovers the plain AEP.

What this is NOT
----------------
- **Not a learned model.** This is the physics solver run 9 times, wrapped in frequency analysis.
  It gives the right output type and a defensible number; it is not the GraphCast-style
  foundation model the proposal's §4 describes.
- **Not nonstationary.** Atlas 14 is a fixed historical frequency analysis with no climate trend,
  so "t" and "Δt" here carry no climate-change signal. The proposal asks for weather events
  "through to 2100"; delivering that needs a projection dataset this project does not have.
  A future-year query is therefore answered under a stationarity assumption, stated explicitly
  in the output rather than hidden.
- **Not calibrated on observed flood frequency.** The depth→probability mapping inherits every
  caveat already documented for the solver itself — notably that this AOI has no inflow boundary
  condition, so it reproduces direct rainfall-runoff ponding, NOT channel overtopping driven by
  the 231 km² upstream watershed. Probabilities here are for PLUVIAL (rain-driven
  surface) flooding only. That is a real scope limit, not a bug.

Usage
-----
    # Build the ensemble + probability surface (runs the solver once per return period)
    python3 analysis/flood_probability.py --duration-hr 24 --threshold-m 0.15

    # Query a point, with a time horizon (this is the proposal's API shape)
    python3 analysis/flood_probability.py --query 28.3669 -81.4330 --horizon-years 30

Outputs (analysis/data/):
    flood_aep.tif                 per-cell annual exceedance probability [0,1]
    flood_aep_summary.json        run metadata, ensemble table, log-linearity check
    flood_depth_by_return_period.tif   per-cell peak depth, one band per return period
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

sys.path.insert(0, PROJ_DIR)
sys.path.insert(0, os.path.join(PROJ_DIR, "simulation"))
sys.path.insert(0, os.path.join(PROJ_DIR, "precipitation"))

# Import with stdout redirected to stderr: flood_sim_ian computes and PRINTS its Horton
# parameter banner at import time, which would otherwise corrupt --query mode's stdout so it no
# longer parses as JSON. An API-shaped endpoint has to emit clean, machine-readable stdout; the
# banner is still visible on stderr for interactive runs.
import contextlib                     # noqa: E402
with contextlib.redirect_stdout(sys.stderr):
    import flood_sim_ian as fsi       # noqa: E402  the calibrated solver, reused unchanged
    import noaa_atlas14 as a14        # noqa: E402  IDF curves + SCS Type II hyetograph

# Return periods to simulate. 1-yr anchors the frequent end (without it, every cell that floods
# even in a common storm saturates at the 2-yr AEP of 0.5 and the map loses all resolution in
# the high-probability range, which is exactly the range that matters for insurance pricing).
RETURN_PERIODS_YR = [1, 2, 5, 10, 25, 50, 100, 200, 500]

AEP_TIF   = os.path.join(DATA_DIR, "flood_aep.tif")
DEPTH_TIF = os.path.join(DATA_DIR, "flood_depth_by_return_period.tif")
SUMMARY   = os.path.join(DATA_DIR, "flood_aep_summary.json")


def site_paths(site_name):
    """Site-suffixed output paths, so a site3 run cannot overwrite the main AOI's surface.

    Same failure mode the fetch scripts had before site_registry.py (both AOIs sharing e.g.
    naip_2021_RGB.tif) — avoided here up front rather than after it bites.
    """
    if site_name in (None, "main_aoi"):
        return AEP_TIF, DEPTH_TIF, SUMMARY
    sfx = f"_{site_name}"
    return (os.path.join(DATA_DIR, f"flood_aep{sfx}.tif"),
            os.path.join(DATA_DIR, f"flood_depth_by_return_period{sfx}.tif"),
            os.path.join(DATA_DIR, f"flood_aep_summary{sfx}.json"))


# ── 1. Design storms ─────────────────────────────────────────────────────────────────────
def load_idf(lat, lon, duration_hr):
    """Atlas 14 depth [mm] at `duration_hr` for each return period in RETURN_PERIODS_YR."""
    df = a14.fetch_atlas14(lat, lon)
    depths = {}
    for T in RETURN_PERIODS_YR:
        sub = df[(df.return_period_yr == T)]
        if sub.empty:
            raise SystemExit(f"Atlas 14 table has no return period {T}yr")
        # nearest available duration to the one requested
        i = (sub.duration_hr - duration_hr).abs().idxmin()
        row = sub.loc[i]
        if abs(row.duration_hr - duration_hr) > 1e-6:
            print(f"  note: {T}yr — nearest available duration {row.duration_hr}hr "
                  f"(requested {duration_hr}hr)")
        depths[T] = float(row.depth_mm)
    return depths


def _report_loglinearity(depths):
    """Print how close depth-vs-log(T) is to linear — the interpolation assumption in step 3.

    Reported, not asserted: if this AOI's curve were strongly nonlinear the interpolation would
    still run, and the reader deserves to know how good the assumption actually is here.
    """
    T = np.array(RETURN_PERIODS_YR, float)
    d = np.array([depths[t] for t in RETURN_PERIODS_YR], float)
    A = np.vstack([np.log(T), np.ones_like(T)]).T
    coef, *_ = np.linalg.lstsq(A, d, rcond=None)
    pred = A @ coef
    ss_res = float(((d - pred) ** 2).sum())
    ss_tot = float(((d - d.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"  depth vs log(return period): R² = {r2:.4f}  "
          f"(interpolation assumption; 1.0 = perfectly log-linear)")
    return r2


# ── 2. Run the ensemble ──────────────────────────────────────────────────────────────────
def run_ensemble(depths, duration_hr, cell_size_m, dt_s):
    """Run the solver once per return period. Returns (depth_stack, profile, meta)."""
    z, dx, profile, dem_bounds = _load_dem(cell_size_m)

    horton = fsi.load_spatial_horton(z.shape, profile["transform"], profile["crs"])
    if horton is not None:
        horton = fsi.apply_impervious_mask(horton, z.shape, profile["transform"], profile["crs"])
        horton = fsi.apply_nlcd_graded_impervious(horton, z.shape,
                                                  profile["transform"], profile["crs"])
    else:
        print("  WARNING: mukey_map.tif unavailable — falling back to the uniform Horton mean")

    stack, meta = [], []
    for T in RETURN_PERIODS_YR:
        depth_mm = depths[T]
        # SCS Type II design hyetograph, the same generator flood_hydrology's scenarios use.
        # Returns a DataFrame: time_min, cumulative_depth_mm, incremental_depth_mm.
        HY_DT_MIN = 5
        hy = a14.make_design_hyetograph(depth_mm, duration_hr, dt_min=HY_DT_MIN)
        step_s = HY_DT_MIN * 60.0
        # incremental_depth_mm is the depth falling WITHIN each step -> convert to a rate [m/s]
        rate_ms = np.asarray(hy["incremental_depth_mm"], dtype=float) / 1000.0 / step_s
        t_hy = np.asarray(hy["time_min"], dtype=float) * 60.0
        # Resample onto the solver's own timestep, then append a drain-down tail so the peak
        # depth is a real peak and not just "wherever it happened to be when the rain stopped"
        # (the solver keeps routing water after rainfall ends).
        tail_s = duration_hr * 3600.0
        t_sim = np.arange(0.0, t_hy[-1] + tail_s, dt_s)
        rain_sim = np.interp(t_sim, t_hy, rate_ms, left=0.0, right=0.0)
        # Conserve the storm's total depth through the resampling (interpolating a rate onto a
        # different grid does not preserve the integral in general).
        applied_mm = float(rain_sim.sum() * dt_s * 1000.0)
        if applied_mm > 1e-9:
            rain_sim *= depth_mm / applied_mm

        t0 = time.time()
        h_max, _, flooded_ha_ts, *_ = fsi.run_sim(
            z, dx, rain_sim, dt_s, frame_interval_min=10 ** 9,   # no frames needed
            verbose=False, use_infiltration=True, horton_arrays=horton)
        el = time.time() - t0
        stack.append(h_max.astype(np.float32))
        meta.append(dict(return_period_yr=T, aep=1.0 / T, rain_mm=depth_mm,
                         peak_depth_m=float(h_max.max()),
                         peak_flooded_ha=float(np.max(flooded_ha_ts)),
                         wall_s=round(el, 1)))
        print(f"  T={T:>4}yr  AEP={1/T:6.4f}  rain={depth_mm:6.1f}mm  "
              f"peak_depth={h_max.max():5.3f}m  peak_flooded={np.max(flooded_ha_ts):6.1f}ha  "
              f"[{el:.0f}s]")
    return np.stack(stack), profile, meta


def repoint_solver_to_site(site_name):
    """Repoint flood_sim_ian's module-level path constants at `site_name`'s own data tree.

    WITHOUT this, passing --site site3 would fetch site3's Atlas 14 IDF but run the solver on
    the MAIN AOI's DEM and soil — a silently wrong answer with no error, exactly the class of
    coordinate/data mismatch site_registry.py exists to prevent. Same monkey-patch pattern
    run_site3_ian.py already uses (that script is the proven reference for this).
    """
    if site_name in (None, "main_aoi"):
        return                                   # module defaults already are the main AOI
    import site_registry
    site = site_registry.get_site(site_name)
    needed = ("dem_cond_path", "soil_json_path", "mukey_map_path", "mukey_legend_path",
              "roads_path", "buildings_path", "nlcd_path")
    missing = [k for k in needed if k not in site]
    if missing:
        raise SystemExit(
            f"\n  --site {site_name} has no registered {missing} in the site registry, so the\n"
            f"  solver cannot be repointed at its data. Refusing to run rather than silently\n"
            f"  simulating the main AOI under this site's rainfall.\n")
    fsi.DEM_COND             = site["dem_cond_path"]
    fsi.SOIL_JSON            = site["soil_json_path"]
    fsi.MUKEY_MAP            = site["mukey_map_path"]
    fsi.MUKEY_LEGEND         = site["mukey_legend_path"]
    fsi.ROADS_PATH           = site["roads_path"]
    fsi.BUILDINGS_PATH       = site["buildings_path"]
    fsi.NLCD_IMPERVIOUS_PATH = site["nlcd_path"]
    # fsi.HORTON was computed at IMPORT time from the main AOI's SOIL_JSON — patching the path
    # afterwards does not retroactively fix that global, so re-derive it (same gotcha and same
    # fix as run_site3_ian.py documents).
    fsi.HORTON = fsi._load_horton_params()
    # SR417_DEM_RAW is the export-grid georeference; for a non-main site use its own raw DEM.
    raw = os.path.join(site["data_root"], "dem", "data", f"{site_name}_dem.tif")
    if os.path.exists(raw):
        fsi.SR417_DEM_RAW = raw
    print(f"  [flood_probability] solver repointed to {site_name}: "
          f"{os.path.relpath(site['dem_cond_path'], PROJ_DIR)}")


def _load_dem(cell_size_m):
    # load_dem_for_sim returns (z, profile, dx) — note the order.
    z, profile, dx = fsi.load_dem_for_sim(cell_size_m)
    with rasterio.open(fsi.SR417_DEM_RAW) as src:
        bounds = src.bounds
    return z, dx, profile, bounds


# ── 3. Invert to annual exceedance probability ───────────────────────────────────────────
def depth_stack_to_aep(stack, threshold_m):
    """Per-cell AEP that peak depth reaches `threshold_m`.

    stack[i] is the peak depth under the return period RETURN_PERIODS_YR[i], increasing in i.

    Three cases per cell:
      * threshold reached even by the most frequent storm  -> AEP = 1/T_min (clamped, cannot
        resolve anything more frequent than the shortest return period simulated)
      * threshold never reached even by the rarest storm   -> AEP = 0 (below resolvable risk)
      * otherwise -> log-linear interpolation in T between the bracketing return periods
    """
    T = np.array(RETURN_PERIODS_YR, float)
    logT = np.log(T)
    nT = len(T)
    ny, nx = stack.shape[1:]

    # Enforce monotonicity in T. Physically guaranteed, but the solver is nonlinear (adaptive dt,
    # Froude cap), so a handful of cells can invert by a hair; cumulative max makes the inversion
    # below well-posed everywhere instead of failing on those cells.
    mono = np.maximum.accumulate(stack, axis=0)
    n_fixed = int((mono != stack).any(axis=0).sum())
    if n_fixed:
        print(f"  monotonicity enforced on {n_fixed:,} cells "
              f"({100*n_fixed/(ny*nx):.3f}% — nonlinear-solver jitter, expected to be tiny)")

    exceeds = mono >= threshold_m                       # (nT, ny, nx) bool
    ever = exceeds.any(axis=0)
    first = np.argmax(exceeds, axis=0)                  # index of smallest T that exceeds

    aep = np.zeros((ny, nx), dtype=np.float32)

    # Case A: exceeded already at the most frequent storm -> clamp at its AEP
    at_floor = ever & (first == 0)
    aep[at_floor] = 1.0 / T[0]

    # Case B: interpolate between first-1 and first
    interp = ever & (first > 0)
    if interp.any():
        i1 = first[interp]
        i0 = i1 - 1
        yy, xx = np.nonzero(interp)
        d0 = mono[i0, yy, xx]
        d1 = mono[i1, yy, xx]
        # fraction of the way from d0 to threshold, in depth
        denom = np.where((d1 - d0) > 1e-12, d1 - d0, np.nan)
        frac = np.clip((threshold_m - d0) / denom, 0.0, 1.0)
        frac = np.nan_to_num(frac, nan=0.0)
        logT_star = logT[i0] + frac * (logT[i1] - logT[i0])
        aep[yy, xx] = (1.0 / np.exp(logT_star)).astype(np.float32)

    # Case C (never exceeded) stays 0.0
    return aep


def aep_to_horizon(aep, years):
    """P(at least one exceedance in `years`) = 1 - (1 - AEP)^years."""
    return 1.0 - np.power(1.0 - np.clip(aep, 0.0, 1.0), float(years))


# ── Point query — the proposal's API shape ───────────────────────────────────────────────
def query_point(lat, lon, horizon_years=1, aep_path=AEP_TIF):
    """P(flood | x, y, t, Δt) at one coordinate. Returns a dict ready to serialize as JSON."""
    if not os.path.exists(aep_path):
        raise SystemExit(f"{aep_path} not found — run without --query first to build it.")
    with rasterio.open(aep_path) as src:
        tx = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x, y = tx.transform(lon, lat)
        if not (src.bounds.left <= x <= src.bounds.right and
                src.bounds.bottom <= y <= src.bounds.top):
            raise SystemExit(f"({lat}, {lon}) falls outside the modeled AOI "
                             f"— this surface covers only the CFX SR417 2x2km box.")
        r, c = rowcol(src.transform, x, y)
        aep = float(src.read(1)[r, c])
        meta = src.tags()
    p = float(aep_to_horizon(np.array(aep), horizon_years))
    return {
        "lat": lat, "lon": lon,
        "annual_exceedance_probability": round(aep, 6),
        "return_period_yr": (round(1.0 / aep, 1) if aep > 0 else None),
        "horizon_years": horizon_years,
        "P_flood_within_horizon": round(p, 6),
        "flood_threshold_m": float(meta.get("threshold_m", "nan")),
        "flood_type": "pluvial (direct rainfall-runoff) only — NOT channel/riverine overtopping",
        "stationarity": "NOAA Atlas 14 historical frequency; no climate trend applied",
    }


# ── Main ─────────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lat", type=float, default=28.36687)
    ap.add_argument("--lon", type=float, default=-81.43299)
    import site_registry
    site_registry.add_site_arg(ap)
    ap.add_argument("--duration-hr", type=float, default=24.0,
                    help="Design-storm duration (default 24hr, the standard for flood design)")
    ap.add_argument("--threshold-m", type=float, default=0.15,
                    help="Depth defining 'flooded' (default 0.15m — roughly the depth at which "
                         "water enters a slab-on-grade FL home / stalls a passenger vehicle)")
    ap.add_argument("--cell-size", type=float, default=5.0)
    ap.add_argument("--dt", type=float, default=20.0)
    ap.add_argument("--query", nargs=2, type=float, metavar=("LAT", "LON"),
                    help="Query an existing probability surface instead of rebuilding it")
    ap.add_argument("--horizon-years", type=float, default=1.0,
                    help="Δt in the proposal's P(flood | ..., Δt). 1 = plain annual probability")
    # site_registry.resolve() prints its own diagnostic lines (which --site resolved to, the
    # coordinates/data_root it picked) -- real, useful output for a human running the ensemble
    # build, but it runs on real stdout and corrupts --query mode's JSON the exact same way the
    # solver import banner did (see the redirect_stdout import above). Same fix, same reason:
    # redirect it here too, found by actually testing `--site site3 --query ...` end to end
    # rather than assuming the import-time fix covered every print path.
    with contextlib.redirect_stdout(sys.stderr):
        args = site_registry.resolve(ap.parse_args())

    if args.query:
        qa, _, _ = site_paths(getattr(args, "site", None))
        print(json.dumps(query_point(args.query[0], args.query[1], args.horizon_years,
                                     aep_path=qa), indent=2))
        return

    print("═" * 74)
    print("  Flood probability from a design-storm ensemble")
    print("═" * 74)
    # Must happen BEFORE any solver call, or the ensemble would run the main AOI's terrain
    # under this site's rainfall.
    repoint_solver_to_site(getattr(args, "site", None))
    aep_tif, depth_tif, summary_path = site_paths(getattr(args, "site", None))
    print(f"\n[1/4] NOAA Atlas 14 IDF for ({args.lat}, {args.lon}), "
          f"{args.duration_hr:.0f}hr duration …")
    depths = load_idf(args.lat, args.lon, args.duration_hr)
    r2 = _report_loglinearity(depths)
    for T in RETURN_PERIODS_YR:
        print(f"    {T:>4}yr (AEP {1/T:6.4f}): {depths[T]:6.1f} mm")

    print(f"\n[2/4] Running the solver once per return period "
          f"({len(RETURN_PERIODS_YR)} runs) …")
    stack, profile, meta = run_ensemble(depths, args.duration_hr, args.cell_size, args.dt)

    print(f"\n[3/4] Inverting depth→probability at threshold {args.threshold_m:.2f} m …")
    aep = depth_stack_to_aep(stack, args.threshold_m)

    prof = dict(profile)
    prof.update(dtype="float32", count=1, compress="deflate", nodata=None)
    with rasterio.open(aep_tif, "w", **prof) as dst:
        dst.write(aep, 1)
        dst.update_tags(threshold_m=str(args.threshold_m),
                        duration_hr=str(args.duration_hr),
                        return_periods=",".join(map(str, RETURN_PERIODS_YR)),
                        method="design-storm ensemble -> annual exceedance probability",
                        flood_type="pluvial only")
    prof_stack = dict(prof); prof_stack.update(count=len(RETURN_PERIODS_YR))
    with rasterio.open(depth_tif, "w", **prof_stack) as dst:
        for i, T in enumerate(RETURN_PERIODS_YR):
            dst.write(stack[i], i + 1)
            dst.set_band_description(i + 1, f"{T}yr peak depth [m]")

    valid = aep[np.isfinite(aep)]
    nz = valid[valid > 0]
    cell_ha = (args.cell_size ** 2) / 1e4
    summary = dict(
        generated="2026-08-04",
        method="NOAA Atlas 14 design-storm ensemble -> per-cell annual exceedance probability",
        site=getattr(args, "site", None) or "main_aoi",
        lat=args.lat, lon=args.lon,
        duration_hr=args.duration_hr, threshold_m=args.threshold_m,
        cell_size_m=args.cell_size, dt_s=args.dt,
        depth_vs_logT_r2=round(r2, 4),
        return_periods=meta,
        cells_total=int(valid.size),
        cells_with_nonzero_risk=int(nz.size),
        area_nonzero_risk_ha=round(float(nz.size * cell_ha), 2),
        area_gt_1pct_annual_ha=round(float((valid >= 0.01).sum() * cell_ha), 2),
        area_gt_10pct_annual_ha=round(float((valid >= 0.10).sum() * cell_ha), 2),
        max_aep=round(float(valid.max()), 6),
        caveats=[
            "Pluvial (direct rainfall-runoff) flooding only. The solver has no inflow boundary "
            "condition, so channel/riverine overtopping driven by the 231 km^2 upstream Shingle "
            "Creek watershed is NOT represented — see the FEMA extent cross-reference (11% "
            "overlap).",
            "Stationary: Atlas 14 is a historical frequency analysis with no climate trend, so "
            "a future-dated query carries no climate-change signal.",
            "AEP is clamped at 1/%d (the most frequent return period simulated); cells flooding "
            "more often than that cannot be resolved." % RETURN_PERIODS_YR[0],
            "Inherits every solver caveat, including the sub-grid pit-trapping noted in "
            "simulation/outputs/RESOLUTION_ANALYSIS.md at fine cell sizes.",
        ],
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[4/4] Saved:")
    print(f"    {os.path.relpath(aep_tif, PROJ_DIR)}")
    print(f"    {os.path.relpath(depth_tif, PROJ_DIR)}")
    print(f"    {os.path.relpath(summary_path, PROJ_DIR)}")
    print(f"\n  Area with any resolvable pluvial flood risk : "
          f"{summary['area_nonzero_risk_ha']:.1f} ha")
    print(f"  Area with >=1% annual chance (FEMA SFHA-equiv): "
          f"{summary['area_gt_1pct_annual_ha']:.1f} ha")
    print(f"  Area with >=10% annual chance                 : "
          f"{summary['area_gt_10pct_annual_ha']:.1f} ha")
    print(f"\n  Query a point:  python3 analysis/flood_probability.py "
          f"--query {args.lat} {args.lon} --horizon-years 30")


if __name__ == "__main__":
    main()
