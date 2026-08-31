"""
A/B the segmentation-derived surface parameters against the scalar, on the real Ian event
=========================================================================================
Runs Hurricane Ian at site3 three times in ONE process and scores each against USGS 02234400
using `analysis/validate_gauge_site3.py`'s own metric functions:

    baseline   scalar MANNING_N        + NLCD 30 m impervious     <- the model as it stands today
    manning    segmented Manning's n   + NLCD 30 m impervious     <- roughness alone
    full       segmented Manning's n   + 0.6 m segmentation impervious

Three arms rather than two because the segmentation supplies two things that act on different
parts of the physics — roughness changes conveyance, imperviousness changes runoff generation —
and a single combined run could not tell you which one moved the metric.

Every arm shares the same DEM, the same SSURGO Horton arrays, the same soil-storage cap and the
same hyetograph, because they are built once and handed to each call. Only the arguments named
in the table above differ.

Why this script and not `simulation/run_site3_ian.py`
-----------------------------------------------------
Two reasons, both practical:
  * That script is being edited concurrently by the connectivity investigation, and it
    writes `simulation/outputs/hydrograph_ian_site3.csv` — the file that investigation's own
    runs produce. Everything here writes to `segmentation/data/` instead, so the two efforts
    cannot clobber each other's results.
  * A baseline recorded by someone else's run is not a controlled baseline. Running both arms
    here, back to back, on the same loaded inputs, removes every confounder except the one
    under test.

The baseline arm passes `manning_n=None`, which takes the solver's original scalar expression
untouched — so the baseline is the existing model, not a re-derivation of it.

Usage:
    python3 segmentation/run_site3_ian_segmented.py                       # all three + table
    python3 segmentation/run_site3_ian_segmented.py --arms baseline full
    python3 segmentation/run_site3_ian_segmented.py --dry-run
"""
import os
import sys
import json
import time
import argparse
import warnings

import numpy as np
import pandas as pd
import rasterio

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
sys.path.insert(0, os.path.join(PROJ_DIR, "simulation"))
sys.path.insert(0, os.path.join(PROJ_DIR, "lidar"))
sys.path.insert(0, os.path.join(PROJ_DIR, "analysis"))

import flood_sim_ian as fsi            # noqa: E402
from test_sites import get_site        # noqa: E402

SITE = "site3"
site = get_site(SITE)
SITE3_DIR = os.path.join(PROJ_DIR, "site3_gee_creek")

# Same monkey-patching every other site3 script uses — flood_sim_ian.py's own path constants
# repointed at site3's data tree. See simulation/run_site3_ian.py for the original.
fsi.DEM_COND = site["dem_cond_path"]
fsi.SOIL_JSON = site["soil_json_path"]
fsi.MUKEY_MAP = site["mukey_map_path"]
fsi.MUKEY_LEGEND = site["mukey_legend_path"]
fsi.ROADS_PATH = site["roads_path"]
fsi.BUILDINGS_PATH = site["buildings_path"]
fsi.NLCD_IMPERVIOUS_PATH = site["nlcd_path"]
fsi.SOIL_STORAGE_CSV = os.path.join(os.path.dirname(site["soil_json_path"]), "soil_storage.csv")
fsi.ASOS_CSV = os.path.join(SITE3_DIR, "precipitation", "data", "asos_hourly_SFB.csv")
# HORTON was computed at import time from the ORIGINAL AOI's soil file; re-derive it now that
# SOIL_JSON points at site3's own.
fsi.HORTON = fsi._load_horton_params()

CMS_TO_CFS = 35.3147


def load_manning(shape, cell_size, tag=""):
    path = os.path.join(DATA_DIR, f"manning_n_{cell_size:g}m_{SITE}{tag}.tif")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}\n  run segmentation/rasterize_parameters.py first")
    with rasterio.open(path) as src:
        mn = src.read(1).astype(np.float32)
    if mn.shape != shape:
        raise SystemExit(f"manning raster {mn.shape} != solver grid {shape}; "
                         f"re-run rasterize_parameters.py at --cell-size {cell_size}")
    return mn


def apply_segmentation_impervious(horton_arrays, cell_size, shape):
    """Substitute the 0.6 m segmentation impervious fraction for NLCD's 30 m one.

    Deliberately the SAME operation as fsi.apply_nlcd_graded_impervious — scale infiltration
    capacity f0/fc by (1 - impervious fraction), leave k alone because imperviousness changes how
    much can infiltrate, not the shape of the decay curve, and leave cells the binary OSM mask
    already forced hard untouched. Only the source of the fraction differs, which is the whole
    point: NLCD resolves a 30 m pixel, NAIP resolves a driveway.

    This is a substitution at an interface the solver already has, so it needs no solver change —
    the same property that lets Smax and Ks be substituted without touching the solver.
    """
    path = os.path.join(DATA_DIR, f"impervious_frac_{cell_size:g}m_{SITE}.tif")
    if horton_arrays is None or not os.path.exists(path):
        raise SystemExit(f"missing {path}; run rasterize_parameters.py first")
    with rasterio.open(path) as src:
        frac = np.clip(src.read(1).astype(np.float32), 0.0, 1.0)
    if frac.shape != shape:
        raise SystemExit(f"impervious raster {frac.shape} != grid {shape}")
    from floodtwin.physics import IMPERVIOUS_FC_MM_HR
    already_hard = np.isclose(horton_arrays["fc"], IMPERVIOUS_FC_MM_HR / 1000 / 3600)
    grade = np.where(already_hard, 1.0, 1.0 - frac)
    out = {"f0": horton_arrays["f0"] * grade, "fc": horton_arrays["fc"] * grade,
           "k": horton_arrays["k"]}
    print(f"  segmentation impervious: mean {100*frac[~already_hard].mean():.1f} % over "
          f"{int((~already_hard).sum())} non-OSM-masked cells "
          f"(capacity reduction {100*(1-grade[~already_hard]).mean():.1f} %)")
    return out


# Horton decay rate for the vision route. A textbook value for sandy profiles, NOT borrowed from
# SSURGO — borrowing it would smuggle survey information into the arm that is meant to do without
# one. Over a 72-hour event fc dominates the decay term anyway.
VISION_HORTON_K_PER_HR = 2.0


def load_vision_soil(shape, cell_size):
    """Horton arrays and a storage cap built from the CLASS MAP instead of SSURGO.

    Estimates Ks and Smax from surface cover alone. It exists so the two routes can be measured
    against the same gauge rather than argued about, and it matters past this site: SSURGO is a
    United States product, so the gap between these two arms is what decides whether the
    "any coordinate -> a twin" premise can leave the US.

    FAIRNESS. The class table stores DRY saturated conductivity. The solver already multiplies
    SSURGO Ksat by AMC3_FACTOR (0.07) for Ian's pre-saturated antecedent conditions, and the same
    factor is applied here. Both sides are dry Ks; correcting only one would pit a wet soil
    against a dry one and let this arm infiltrate ~14x too much.
    """
    ks_p = os.path.join(DATA_DIR, f"vision_ks_dry_mm_hr_{cell_size:g}m_{SITE}.tif")
    sm_p = os.path.join(DATA_DIR, f"vision_soil_storage_{cell_size:g}m_{SITE}.tif")
    for q in (ks_p, sm_p):
        if not os.path.exists(q):
            raise SystemExit(f"missing {q}; run rasterize_parameters.py first")
    with rasterio.open(ks_p) as src:
        ks_dry = src.read(1).astype(np.float32)
    with rasterio.open(sm_p) as src:
        smax = src.read(1).astype(np.float32)
    if ks_dry.shape != shape or smax.shape != shape:
        raise SystemExit(f"vision soil rasters {ks_dry.shape}/{smax.shape} != grid {shape}")

    fc_mm_hr = ks_dry * fsi.AMC3_FACTOR
    horton = {
        "fc": (fc_mm_hr / 1000.0 / 3600.0).astype(np.float32),
        "f0": (fc_mm_hr * 2.5 / 1000.0 / 3600.0).astype(np.float32),
        "k": np.full(shape, VISION_HORTON_K_PER_HR / 3600.0, dtype=np.float32),
    }
    print(f"  vision soil: Ks dry mean {ks_dry.mean():.1f} mm/hr x AMC3={fsi.AMC3_FACTOR} "
          f"-> fc_eff mean {fc_mm_hr.mean():.1f} mm/hr")
    print(f"               storage cap mean {1000*smax.mean():.0f} mm, "
          f"{100*float((smax == 0).mean()):.0f} % of cells zero-storage")
    return horton, smax


def snap_gauge(z, profile, dx):
    """Grid cell for USGS 02234400, snapped onto the burned channel.

    Mirrors simulation/run_site3_ian.py: the gauge coordinate can land a cell or two off the
    burned centreline, and a dry floodplain cell next to the creek reports no discharge.
    """
    from pyproj import Transformer
    tr = Transformer.from_crs("epsg:4326", profile["crs"], always_xy=True)
    gx, gy = tr.transform(site["gauge_lon"], site["gauge_lat"])
    gc, gr = ~profile["transform"] * (gx, gy)
    gr, gc = int(round(gr)), int(round(gc))
    rad = max(1, int(round(25.0 / dx)))
    r0, r1 = max(0, gr - rad), min(z.shape[0], gr + rad + 1)
    c0, c1 = max(0, gc - rad), min(z.shape[1], gc + rad + 1)
    sub = z[r0:r1, c0:c1]
    fl = int(np.nanargmin(np.where(np.isfinite(sub), sub, np.inf)))
    return (r0 + fl // sub.shape[1], c0 + fl % sub.shape[1])


def run_arm(tag, z, dx, rain_sim, dt, profile, horton_arrays, max_deficit_m, gauge_rc,
            manning_n, n_steps):
    print(f"\n{'='*74}\n[{tag}] solving {z.shape[0]}x{z.shape[1]} @ {dx:.1f} m …")
    if manning_n is None:
        print(f"       Manning's n: scalar {fsi.MANNING_N} (original code path)")
    else:
        print(f"       Manning's n: spatial field, mean {manning_n.mean():.4f}, "
              f"range {manning_n.min():.4f}-{manning_n.max():.4f}")
    t0 = time.time()
    kw = dict(frame_interval_min=60.0, use_infiltration=True,
              horton_arrays=horton_arrays, max_deficit_m=max_deficit_m, manning_n=manning_n)
    if gauge_rc is not None:
        kw["gauge_rc"] = gauge_rc
    out = fsi.run_sim(z, dx, rain_sim, dt, **kw)
    h_max, cum_infil, flooded_ha_ts, rain_ts, Pe_ts, mean_depth_ts, frame_data = out
    elapsed = time.time() - t0

    step_hrs = np.arange(n_steps) * dt / 3600.0
    cols = {
        "time_min": step_hrs * 60,
        "rain_mm_hr": rain_ts,
        "Pe_mm_hr": Pe_ts,
        "flooded_ha": flooded_ha_ts,
        "mean_depth_m": mean_depth_ts,
        "outflow_total_cms": frame_data["outflow_total_cms"],
        "outflow_total_cfs": frame_data["outflow_total_cms"] * CMS_TO_CFS,
        "outflow_south_cms": frame_data["outflow_south_cms"],
        "outflow_south_cfs": frame_data["outflow_south_cms"] * CMS_TO_CFS,
    }
    if frame_data.get("gauge_cms") is not None:
        cols["gauge_cms"] = frame_data["gauge_cms"]
        cols["gauge_cfs"] = frame_data["gauge_cms"] * CMS_TO_CFS
    df = pd.DataFrame(cols)
    path = os.path.join(DATA_DIR, f"hydrograph_ian_{SITE}_{tag}.csv")
    df.to_csv(path, index=False)

    peak_ha = float(flooded_ha_ts.max())
    print(f"       {elapsed:.0f}s | peak depth {float(h_max.max()):.3f} m | "
          f"peak flooded {peak_ha:.1f} ha | peak outflow {df['outflow_total_cfs'].max():.1f} cfs")
    return df, {"wall_s": round(elapsed, 1), "peak_depth_m": float(h_max.max()),
                "peak_flooded_ha": peak_ha,
                "peak_outflow_total_cfs": float(df["outflow_total_cfs"].max()),
                "hydrograph": os.path.relpath(path, PROJ_DIR)}


def score(df, rain_mm, domain_m2):
    """The two numbers every change here is scored against, via the validator's own functions."""
    import validate_gauge_site3 as vg
    ts = df["time_min"].to_numpy(float) / 60.0
    qs = df["outflow_total_cms"].to_numpy(float)
    to, qo = vg.load_observed()
    qo_x = (qo - vg.BASEFLOW_CFS).clip(0) * vg.CFS_TO_CMS

    limb_sim = vg.rising_limb_50(ts, qs)
    limb_obs = vg.rising_limb_50(to, qo_x)
    vol = np.trapz(qs, ts * 3600.0)
    rc = vol / (rain_mm / 1000.0 * domain_m2)

    # Like-for-like observed range over the simulated window, both baseflow choices.
    lo_h, hi_h = float(ts.min()), float(ts.max())
    m = (to >= lo_h) & (to <= hi_h)
    obs_rc = []
    for bf in (0.0, vg.BASEFLOW_CFS):
        v = np.trapz((qo[m] - bf).clip(0) * vg.CFS_TO_CMS, to[m] * 3600.0)
        obs_rc.append(v / (rain_mm / 1000.0 * vg.GAUGE_AREA_M2))

    out = {
        "rising_limb_sim_h": round(float(limb_sim), 3),
        "rising_limb_obs_h": round(float(limb_obs), 3),
        "rising_limb_diff_h": round(abs(float(limb_sim) - float(limb_obs)), 3),
        "runoff_coefficient_sim_pct": round(100 * rc, 3),
        "runoff_coefficient_obs_pct_range": [round(100 * min(obs_rc), 2), round(100 * max(obs_rc), 2)],
        "runoff_shortfall_x": [round(min(obs_rc) / rc, 2), round(max(obs_rc) / rc, 2)],
        "outflow_volume_Mm3": round(vol / 1e6, 4),
    }
    if "gauge_cms" in df.columns:
        qg = df["gauge_cms"].to_numpy(float)
        if np.nanmax(qg) > 0:
            out["gauge_cell_rising_limb_sim_h"] = round(float(vg.rising_limb_50(ts, qg)), 3)
            out["gauge_cell_peak_cfs"] = round(float(np.nanmax(qg) * CMS_TO_CFS), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-size", type=float, default=5.0)
    ap.add_argument("--dt", type=float, default=20.0)
    ap.add_argument("--param-tag", default="",
                    help="suffix on the parameter rasters to run, e.g. _sam3 to use the SAM3 "
                         "backend's field instead of the spectral one")
    ap.add_argument("--arms", default="all", nargs="+",
                    choices=["all", "baseline", "manning", "full", "vision_soil"],
                    help="baseline = scalar n + NLCD impervious (the model as it stands today); "
                         "manning = segmented n only; full = segmented n + segmentation "
                         "impervious; vision_soil = segmented n + Ks/Smax from the class map "
                         "instead of SSURGO")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("=" * 74)
    print(f"Hurricane Ian at {SITE} — scalar vs segmentation-derived Manning's n")
    print("=" * 74)

    print(f"\n[1/4] DEM @ {args.cell_size} m …")
    z, profile, dx = fsi.load_dem_for_sim(args.cell_size)

    print("\n[2/4] Shared inputs (identical for both arms) …")
    horton_arrays = fsi.load_spatial_horton(z.shape, profile["transform"], profile["crs"])
    if horton_arrays is not None:
        horton_arrays = fsi.apply_impervious_mask(horton_arrays, z.shape,
                                                  profile["transform"], profile["crs"])
        horton_arrays = fsi.apply_nlcd_graded_impervious(horton_arrays, z.shape,
                                                         profile["transform"], profile["crs"])
    max_deficit_m = fsi.load_soil_storage_capacity(z.shape, profile["transform"], profile["crs"])
    if max_deficit_m is not None:
        print(f"  soil storage cap: mean {1000*float(max_deficit_m.mean()):.0f} mm, "
              f"{100*float((max_deficit_m == 0).mean()):.0f} % depressional")

    rain_sim, hours, rain_mm = fsi.load_ian_hyetograph(args.dt)
    total_rain = float(rain_mm.sum())
    n_steps = len(rain_sim)
    domain_m2 = z.shape[0] * z.shape[1] * dx * dx
    print(f"  hyetograph: {total_rain:.0f} mm over {n_steps*args.dt/3600:.0f} h  "
          f"({n_steps} steps)   domain {domain_m2/1e6:.2f} km2")

    gauge_rc = None
    try:
        gauge_rc = snap_gauge(z, profile, dx)
        print(f"  gauge {site['gauge_site_no']} at grid {gauge_rc}, bed {z[gauge_rc]:.2f} m")
    except Exception as e:
        print(f"  gauge snapping unavailable ({e}); boundary outflow only")

    manning = load_manning(z.shape, args.cell_size, args.param_tag)
    print(f"  Manning field{args.param_tag or ' (spectral)'}: mean {manning.mean():.4f}  "
          f"vs scalar {fsi.MANNING_N}  ({100*(manning.mean()/fsi.MANNING_N - 1):+.1f} %)")

    if args.dry_run:
        print("\nDry run — exiting.")
        return

    arms = (["baseline", "manning", "full", "vision_soil"] if "all" in args.arms
            else list(args.arms))
    results = {}
    for tag in arms:
        mn = None if tag == "baseline" else manning
        ha, mdef = horton_arrays, max_deficit_m
        soil = "SSURGO"
        imperv = "NLCD 30 m"
        if tag == "full":
            ha = apply_segmentation_impervious(horton_arrays, args.cell_size, z.shape)
            imperv = "segmentation 0.6 m"
        elif tag == "vision_soil":
            ha, mdef = load_vision_soil(z.shape, args.cell_size)
            # The hard OSM road/building footprint is applied to BOTH routes — it is a mapped
            # fact, not a soil property, and both should respect it. NLCD grading is applied only
            # to the SSURGO route: it exists to tell SSURGO about imperviousness SSURGO cannot
            # know, and the class map already carries that information directly, so adding it
            # here would double-count. Each route therefore gets its own complete and
            # self-consistent treatment of imperviousness, which is the like-for-like comparison.
            ha = fsi.apply_impervious_mask(ha, z.shape, profile["transform"], profile["crs"])
            soil, imperv = "vision (class map)", "class map"
        df, meta = run_arm(tag, z, dx, rain_sim, args.dt, profile, ha,
                           mdef, gauge_rc, mn, n_steps)
        meta["scores"] = score(df, total_rain, domain_m2)
        meta["config"] = {"manning": "scalar" if mn is None else "segmented",
                          "soil": soil, "impervious": imperv}
        results[tag] = meta

    summary = {
        "site": SITE, "cell_size_m": float(dx), "grid_shape": list(z.shape),
        "dt_s": args.dt, "n_steps": n_steps, "total_rain_mm": total_rain,
        "domain_km2": round(domain_m2 / 1e6, 3),
        "manning_scalar": float(fsi.MANNING_N),
        "manning_field_mean": float(manning.mean()),
        "manning_field_min_max": [float(manning.min()), float(manning.max())],
        "arms": results,
    }
    sp = os.path.join(DATA_DIR, f"ab_summary_{SITE}.json")
    with open(sp, "w") as fh:
        json.dump(summary, fh, indent=2)

    # ── comparison ────────────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("SCORED AGAINST USGS 02234400 (Hurricane Ian)")
    print("=" * 74)
    rows = [
        ("rising limb, sim [h]", "scores", "rising_limb_sim_h", "{:.2f}"),
        ("rising limb vs observed [h]", "scores", "rising_limb_diff_h", "{:.2f}"),
        ("runoff coefficient [%]", "scores", "runoff_coefficient_sim_pct", "{:.3f}"),
        ("outflow volume [Mm3]", "scores", "outflow_volume_Mm3", "{:.4f}"),
        ("peak flooded [ha]", None, "peak_flooded_ha", "{:.1f}"),
        ("peak depth [m]", None, "peak_depth_m", "{:.3f}"),
        ("peak outflow [cfs]", None, "peak_outflow_total_cfs", "{:.1f}"),
    ]
    names = [a for a in arms if a in results]
    hdr = "  {:<32}".format("metric") + "".join(f"{n:>15}" for n in names)
    print(hdr)
    print("  " + "-" * (32 + 15 * len(names)))
    for label, sub, key, fmt in rows:
        vals = []
        for n in names:
            v = results[n][sub][key] if sub else results[n][key]
            vals.append(v)
        line = "  {:<32}".format(label) + "".join(fmt.format(v).rjust(15) for v in vals)
        if len(names) > 1 and vals[0]:
            line += "   " + " ".join(f"{100*(v-vals[0])/vals[0]:+.1f}%" for v in vals[1:])
        print(line)
    print("  " + "-" * (32 + 15 * len(names)))
    ref = results[names[0]]["scores"]
    print(f"  observed rising limb            {ref['rising_limb_obs_h']:.2f} h")
    print(f"  observed runoff coefficient     {ref['runoff_coefficient_obs_pct_range'][0]:.1f}"
          f" - {ref['runoff_coefficient_obs_pct_range'][1]:.1f} %  (like-for-like window)")
    print("  arms:")
    for n in names:
        c = results[n]["config"]
        print(f"    {n:<12} n={c['manning']:<10} soil={c['soil']:<18} impervious={c['impervious']}")
    print(f"\n  wrote {os.path.relpath(sp, PROJ_DIR)}")


if __name__ == "__main__":
    main()
