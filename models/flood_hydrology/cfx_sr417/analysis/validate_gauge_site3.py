"""
Gauge validation — site3 / Gee Creek (USGS 02234400) vs. Hurricane Ian
======================================================================
Computes the two metrics every solver change at site3 is scored against, from the raw NWIS
instantaneous-values record and the solver's own boundary-outflow hydrograph.

Why this script exists
----------------------
Both metrics were previously computed ad hoc and only their results recorded. One of them,
the observed runoff coefficient of 19.6 %, could not be reproduced from the on-disk gauge
record under any standard baseflow/window choice — the defensible range is 16-36 %. An
unreproducible target is not a target, so this reports the full sensitivity instead of a
single number, and prints the assumptions that produced each one.

The two metrics
---------------
1. RISING LIMB — time at which flow first crosses 50 % of its own peak. Robust to the
   magnitude gap (the model captures 11.65 of the gauge's 33.15 km²) because it is a timing
   statistic on each hydrograph's own scale. NOTE the gauge samples at 15 min = 0.25 h, so
   any difference below that is not resolved by the observation and must not be claimed as
   accuracy — the same limit that makes the peak argmax unusable here.

2. RUNOFF COEFFICIENT — runoff volume / rainfall volume, each over its own contributing
   area. The differing denominators are correct, not a mismatch: the coefficient is
   dimensionless and area-normalises each side independently.

   Simulated: boundary outflow summed over all four domain edges / (P x domain area).
   Observed:  gauge discharge above baseflow / (P x documented drainage area).

   The simulated side is structurally approximate: it sums every edge, while the gauge
   measures one channel. It is a magnitude check, not an identity.

Usage:
    python3 analysis/validate_gauge_site3.py
    python3 analysis/validate_gauge_site3.py --hydrograph <path-to-another-run.csv>
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)

SIM_CSV  = os.path.join(PROJ_DIR, "simulation", "outputs", "hydrograph_ian_site3.csv")
OBS_CSV  = os.path.join(PROJ_DIR, "site3_gee_creek", "infrastructure", "data",
                        "gee_creek_ian_discharge.csv")
SUMMARY  = os.path.join(PROJ_DIR, "simulation", "outputs", "ian_sim_summary_site3.json")

CFS_TO_CMS   = 0.0283168466      # exact, ft^3 -> m^3
GAUGE_AREA_M2 = 33.15e6          # USGS documented drainage area for 02234400
BASEFLOW_CFS  = 45.2             # pre-storm flow, 2022-09-27 19:30 UTC
GAUGE_DT_H    = 0.25             # 15-minute instantaneous-values record

# Ian window used by run_site3_ian.py's hyetograph (KSFB ASOS, UTC)
IAN_T0 = pd.Timestamp("2022-09-28 00:00:00", tz="UTC")


def rising_limb_50(t_h, q):
    """Time at which the rising limb first reaches 50 % of the peak (linear interpolation)."""
    t, q = np.asarray(t_h, float), np.asarray(q, float)
    ip = int(np.argmax(q))
    half = 0.5 * q[ip]
    seg_q, seg_t = q[:ip + 1], t[:ip + 1]
    hit = np.where(seg_q >= half)[0]
    if len(hit) == 0:
        return np.nan
    i = hit[0]
    if i == 0:
        return seg_t[0]
    q0, q1, t0, t1 = seg_q[i - 1], seg_q[i], seg_t[i - 1], seg_t[i]
    return t0 + (half - q0) / (q1 - q0) * (t1 - t0) if q1 > q0 else t1


def load_observed():
    df = pd.read_csv(OBS_CSV, parse_dates=["dateTime"])
    t = (df["dateTime"] - IAN_T0).dt.total_seconds().to_numpy() / 3600.0
    return t, df["discharge_cfs"].to_numpy(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hydrograph", default=SIM_CSV, help="simulated hydrograph CSV")
    ap.add_argument("--rain-mm", type=float, default=None,
                    help="storm total [mm]; default reads ian_sim_summary_site3.json")
    args = ap.parse_args()

    for p in (args.hydrograph, OBS_CSV):
        if not os.path.exists(p):
            sys.exit(f"missing input: {p}")

    sim = pd.read_csv(args.hydrograph)
    ts  = sim["time_min"].to_numpy(float) / 60.0
    qs  = sim["outflow_total_cms"].to_numpy(float)

    # Discharge AT the gauge cell, when the run recorded it. This is the only apples-to-apples
    # cross-section: domain-boundary outflow makes water traverse from the gauge out to the box
    # edge, time the real gauge never sees. Its contributing area is the delineated catchment,
    # not the whole domain, so it carries its own denominator.
    qg = sim["gauge_cms"].to_numpy(float) if "gauge_cms" in sim.columns else None
    catch_km2 = None
    ws = os.path.join(PROJ_DIR, "site3_gee_creek", "dem", "data", "hydro", "watershed.geojson")
    if qg is not None and os.path.exists(ws):
        try:
            import geopandas as gpd
            catch_km2 = float(gpd.read_file(ws).to_crs("epsg:5070").area.sum()) / 1e6
        except Exception:
            catch_km2 = None

    rain_mm = args.rain_mm
    if rain_mm is None:
        if not os.path.exists(SUMMARY):
            sys.exit("ian_sim_summary_site3.json not found — pass --rain-mm explicitly")
        meta = json.load(open(SUMMARY))
        rain_mm = float(meta["total_rain_mm"])
        rows, cols = meta["grid_shape"]
        cell = float(meta["cell_size_m"])
        domain_m2 = rows * cols * cell * cell
    else:
        sys.exit("--rain-mm given but domain area still comes from the summary; keep both")

    to, qo = load_observed()

    print("=" * 74)
    print("Gauge validation — site3 / Gee Creek 02234400, Hurricane Ian")
    print("=" * 74)
    print(f"  simulated : {os.path.relpath(args.hydrograph, PROJ_DIR)}")
    print(f"  observed  : {os.path.relpath(OBS_CSV, PROJ_DIR)}  ({len(to)} pts, "
          f"{np.median(np.diff(to)) * 60:.0f} min sampling)")
    print(f"  storm     : {rain_mm:.1f} mm   domain {domain_m2 / 1e6:.2f} km²   "
          f"gauge watershed {GAUGE_AREA_M2 / 1e6:.2f} km²")

    # ── 1. rising limb ────────────────────────────────────────────────────────
    qo_x = (qo - BASEFLOW_CFS).clip(0) * CFS_TO_CMS
    # Baseflow separation must be SYMMETRIC. With a baseflow initial condition the simulated
    # series also starts non-zero, and the pre-filled channel dumps out at the domain edge on
    # step 1 — leaving that in put the "rising limb" at t=0.00 h, an initial-condition
    # transient rather than a storm response. Subtract each series' own pre-storm level.
    qs_x = (qs - float(qs[0])).clip(0)
    ls, lo = rising_limb_50(ts, qs_x), rising_limb_50(to, qo_x)
    diff = abs(ls - lo)
    print("\n[1] RISING LIMB (50 % of peak)")
    print(f"  simulated  t = {ls:6.2f} h")
    print(f"  observed   t = {lo:6.2f} h")
    print(f"  difference   = {diff:6.2f} h", end="")
    if diff < GAUGE_DT_H:
        print(f"   -- BELOW the gauge's own {GAUGE_DT_H:.2f} h sampling interval;")
        print("                 agreement to within one sample. Report as 'preserved', "
              "not as an accuracy figure.")
    else:
        print(f"   ({diff / GAUGE_DT_H:.1f}x the gauge sampling interval — resolved)")

    # ── 2. runoff coefficient ─────────────────────────────────────────────────
    vol_s = np.trapz(qs, ts * 3600.0)
    rc_s  = vol_s / (rain_mm / 1000.0 * domain_m2)
    print("\n[2] RUNOFF COEFFICIENT")
    print(f"  simulated : {vol_s / 1e6:6.3f} of {rain_mm / 1000 * domain_m2 / 1e6:6.3f} "
          f"million m³  =  {rc_s * 100:5.2f} %   (all four domain edges)")
    # The simulation integrates a fixed window; the gauge record is longer. Only the matching
    # window is a like-for-like comparison — a longer observed window collects recession the
    # simulation was never run far enough to produce, which inflates the apparent shortfall.
    sim_lo, sim_hi = float(ts.min()), float(ts.max())
    windows = {f"{sim_lo:.0f}-{sim_hi:.0f} h  <-- sim window": (sim_lo, sim_hi),
               "full gauge record": (float(to.min()), float(to.max())),
               "0-48 h": (0.0, 48.0)}
    print(f"\n  observed, by baseflow separation and integration window:")
    print(f"    {'baseflow':>9} {'window':>26} {'volume Mm³':>12} {'RC':>9}")
    primary = []
    for bf in (0.0, BASEFLOW_CFS):
        for lab, (lo_h, hi_h) in windows.items():
            m = (to >= lo_h) & (to <= hi_h)
            v = np.trapz((qo[m] - bf).clip(0) * CFS_TO_CMS, to[m] * 3600.0)
            rc = v / (rain_mm / 1000.0 * GAUGE_AREA_M2)
            if "sim window" in lab:
                primary.append(rc)
            print(f"    {bf:>8.1f}c {lab:>26} {v / 1e6:>12.3f} {rc * 100:>8.1f} %")
    lo_rc, hi_rc = min(primary), max(primary)
    print(f"\n  LIKE-FOR-LIKE ({sim_lo:.0f}-{sim_hi:.0f} h, the simulated window):")
    print(f"    observed  : {lo_rc * 100:.1f} – {hi_rc * 100:.1f} %  "
          f"(range spans baseflow separated / not separated)")
    print(f"    simulated : {rc_s * 100:.2f} %")
    print(f"    shortfall : {lo_rc / rc_s:.1f}x – {hi_rc / rc_s:.1f}x")
    print("\n  The 19.6 % in earlier write-ups could not be reproduced from this record under")
    print("  any of these choices. Quote the like-for-like range, not a single figure.")

    # ── 3. drainage behaviour ────────────────────────────────────────────────
    rain = sim["rain_mm_hr"].to_numpy(float)
    wet = ts[rain > 0.05]
    if qg is not None:
        qg_x = (qg - float(qg[0])).clip(0)          # same symmetric separation
        lg = rising_limb_50(ts, qg_x)
        print("\n[2b] AT THE GAUGE CELL (apples-to-apples cross-section)")
        print(f"  peak         : {qg.max() / CFS_TO_CMS:8.1f} cfs at t={ts[qg.argmax()]:.1f} h"
              f"   (observed {qo.max():.0f} cfs at t={to[qo.argmax()]:.1f} h)")
        print(f"  lag from rain peak (t=33.0 h): simulated {ts[qg.argmax()] - 33.0:+.1f} h, "
              f"observed {to[qo.argmax()] - 33.0:+.1f} h")
        print(f"  rising limb  : sim t={lg:.2f} h vs obs {lo:.2f} h -> {abs(lg - lo):.2f} h"
              + ("  (below gauge resolution)" if abs(lg - lo) < GAUGE_DT_H else ""))
        if float(qg[0]) > 0:
            print(f"  simulated pre-storm baseflow at the gauge cell: "
                  f"{float(qg[0]) / CFS_TO_CMS:.1f} cfs  (observed {BASEFLOW_CFS} cfs)")
        if catch_km2:
            vg = np.trapz(qg_x, ts * 3600.0)        # storm response only, baseflow removed
            rcg = vg / (rain_mm / 1000.0 * catch_km2 * 1e6)
            print(f"  runoff coeff : {rcg * 100:5.2f} % (storm only) over the {catch_km2:.2f} km² delineated "
                  f"catchment  (observed {lo_rc * 100:.1f}-{hi_rc * 100:.1f} % over 33.15 km²)")
            print(f"  NOTE the model captures {catch_km2 / 33.15 * 100:.0f} % of the gauge's "
                  f"documented area, so magnitude remains structurally low.")

    print("\n[3] DRAINAGE AFTER RAIN STOPS")
    if len(wet):
        t_end = wet.max()
        fa = sim["flooded_ha"].to_numpy(float)
        # With a baseflow initial condition the channel is already wet at t=0, so flooded_ha
        # carries a constant channel baseline. Report storm-driven area net of it, otherwise
        # peak-area is not comparable against runs that started dry.
        fa0 = float(fa[0])
        if fa0 > 0.1:
            print(f"  channel already wet at t=0: {fa0:.1f} ha baseline (baseflow IC) — "
                  f"storm-driven area below is net of it")
            fa = fa - fa0
        tail = ts >= ts.max() - 24
        slope = np.polyfit(ts[tail], fa[tail], 1)[0]
        print(f"  rain ends t = {t_end:.1f} h; peak flooded {fa.max():.1f} ha, "
              f"final {fa[-1]:.1f} ha")
        print(f"  trend over the last 24 h: {slope:+.3f} ha/hr", end="")
        if slope > 0:
            print("   -- STILL RISING. Water is not reaching an outlet.")
        else:
            print(f"   -- draining ({abs(fa[-1] / slope) / 24:.0f} days to clear at this rate)")
    print("=" * 74)


if __name__ == "__main__":
    main()
