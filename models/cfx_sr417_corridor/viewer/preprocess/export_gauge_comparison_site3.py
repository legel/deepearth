"""
Export the real-vs-simulated Ian discharge comparison for site3 -> viewer/data/
==================================================================================
Combines simulation/outputs/hydrograph_ian_site3.csv (this project's own simulated
total-boundary outflow, from run_site3_ian.py) with the real USGS NWIS instantaneous-values
record for Gee Creek gauge 02234400
(site3_gee_creek/infrastructure/data/gee_creek_ian_discharge.csv, fetched 2026-07-27) into one
JSON the site3.html viewer page renders as a live chart -- the actual validation deliverable
this whole site3 effort was built for. See CLAUDE.md's 2026-07-27 "real Hurricane Ian event"
entry for the full comparison writeup and honest caveats (outflow_total sums all 4 box edges,
not a single channel; site3's mesh captures 35.1% of the gauge's documented drainage area).

Usage:
    python3 viewer/preprocess/export_gauge_comparison_site3.py
"""
import os, sys, json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # viewer/
PROJ_DIR = os.path.dirname(BASE_DIR)
OUT_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(OUT_DIR, exist_ok=True)

SIM_CSV = os.path.join(PROJ_DIR, "simulation", "outputs", "hydrograph_ian_site3.csv")
REAL_CSV = os.path.join(PROJ_DIR, "site3_gee_creek", "infrastructure", "data",
                         "gee_creek_ian_discharge.csv")
SIM_START = pd.Timestamp("2022-09-28 00:00", tz="UTC")   # matches flood_sim_ian.IAN_START


def main():
    if not os.path.exists(SIM_CSV):
        print(f"MISSING: {SIM_CSV} -- run simulation/run_site3_ian.py first")
        return
    if not os.path.exists(REAL_CSV):
        print(f"MISSING: {REAL_CSV} -- fetch the real gauge record first")
        return

    sim = pd.read_csv(SIM_CSV)
    real = pd.read_csv(REAL_CSV, parse_dates=["dateTime"])

    # Downsample the simulated series (12,960 rows @ 20s) for a smooth-enough, browser-light
    # chart -- every 10th row is still a 200s effective resolution, far finer than needed to
    # see the shape.
    sim_ds = sim.iloc[::10].reset_index(drop=True)

    real["time_min"] = (real["dateTime"] - SIM_START).dt.total_seconds() / 60.0
    # Keep a window around the sim (-12h to +84h) -- plenty of context either side.
    real_w = real[(real["time_min"] >= -12 * 60) & (real["time_min"] <= 84 * 60)].reset_index(drop=True)

    sim_peak_idx = int(sim["outflow_total_cfs"].idxmax())
    real_peak_idx = int(real_w["discharge_cfs"].idxmax())
    rain_peak_idx = int(sim["rain_mm_hr"].idxmax())

    out = {
        "sim_time_min": sim_ds["time_min"].round(2).tolist(),
        "sim_outflow_cfs": sim_ds["outflow_total_cfs"].round(3).tolist(),
        "sim_rain_mm_hr": sim_ds["rain_mm_hr"].round(2).tolist(),
        "sim_flooded_ha": sim_ds["flooded_ha"].round(3).tolist(),
        "real_time_min": real_w["time_min"].round(2).tolist(),
        "real_discharge_cfs": real_w["discharge_cfs"].round(1).tolist(),
        "summary": {
            "sim_peak_cfs": float(sim["outflow_total_cfs"].iloc[sim_peak_idx]),
            "sim_peak_time_min": float(sim["time_min"].iloc[sim_peak_idx]),
            "real_peak_cfs": float(real_w["discharge_cfs"].iloc[real_peak_idx]),
            "real_peak_time_min": float(real_w["time_min"].iloc[real_peak_idx]),
            "rain_peak_time_min": float(sim["time_min"].iloc[rain_peak_idx]),
            "area_capture_frac": 11.65 / 33.15,
            "gauge_site_no": "02234400",
            "precip_station": "KSFB (Orlando Sanford Intl, 10.8km)",
        },
    }
    out_path = os.path.join(OUT_DIR, "gauge_comparison_site3.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh)
    print(f"gauge_comparison_site3.json written "
          f"({len(out['sim_time_min'])} sim pts, {len(out['real_time_min'])} real pts)")
    print(f"  sim peak {out['summary']['sim_peak_cfs']:.1f} cfs @ "
          f"t={out['summary']['sim_peak_time_min']/60:.2f}h")
    print(f"  real peak {out['summary']['real_peak_cfs']:.1f} cfs @ "
          f"t={out['summary']['real_peak_time_min']/60:.2f}h")


if __name__ == "__main__":
    main()
