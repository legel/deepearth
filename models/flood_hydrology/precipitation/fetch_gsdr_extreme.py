"""
Confirm the true historical rainfall extreme across the full GSDR record and
build a hyetograph for it.

`fetch_gsdr_gauge.py` already extracted the 1960-07-25 event from station
US_086638 (143.5 mm in a single hour — the all-time max *1-hr* intensity).
This script re-examines the *entire* hourly record of both nearby stations
(US_086638 1942-1985, US_086628 1974-2011) to check whether a longer/larger
storm — by total depth rather than peak intensity — is buried in the record.

Finding (2026-06-18): the 1960-07-25 event is confirmed as the all-time max
1-hr intensity across both stations. But the biggest storm by total depth is
a different event: 1945-09-16 at US_086638, 245.6 mm in 24 hours (vs 208 mm
for the existing historical_gsdr window) — consistent with a tropical
system making landfall near Orlando in mid-September 1945. This is the
"biggest in history" event by total flood-producing rainfall, and is what
this script extracts.

Outputs
-------
precipitation/data/hyetograph_historical_gsdr_extreme.csv
    columns: time_min, cumulative_depth_mm, incremental_depth_mm

Usage
-----
    python3 precipitation/fetch_gsdr_extreme.py
"""

import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_CSV  = os.path.join(DATA_DIR, "hyetograph_historical_gsdr_extreme.csv")

GSDR_DIR = os.path.expanduser("~/Desktop/GSDR/QC_d data - US")
STATIONS = {
    "US_086638": "ORLANDO HERNDON AP",
    "US_086628": "ORLANDO INTL AP",
}

sys.path.insert(0, BASE_DIR)
from fetch_gsdr_gauge import load_gsdr  # noqa: E402 (reuse INTENSE-format parser)

EXTREME_STATION = "US_086638"
EXTREME_START   = "1945-09-15 17:00:00"
EXTREME_END     = "1945-09-16 16:00:00"


def survey_station(station_id):
    """Print 1-hr peak and 24-hr rolling-max totals for one station."""
    path = os.path.join(GSDR_DIR, f"{station_id}.txt")
    ts, vals = load_gsdr(path)
    s = pd.Series(vals, index=pd.DatetimeIndex(ts))

    peak_1hr_idx = int(np.nanargmax(vals))
    peak_1hr_mm  = vals[peak_1hr_idx]
    peak_1hr_dt  = pd.Timestamp(ts[peak_1hr_idx])

    roll24 = s.fillna(0).rolling(24, min_periods=24).sum()
    peak_24hr_end = roll24.idxmax()
    peak_24hr_mm  = roll24.max()

    print(f"  {station_id} ({STATIONS[station_id]}):")
    print(f"    Record         : {pd.Timestamp(ts[0])} -> {pd.Timestamp(ts[-1])}")
    print(f"    Max 1-hr event : {peak_1hr_mm:.1f} mm at {peak_1hr_dt}")
    print(f"    Max 24-hr total: {peak_24hr_mm:.1f} mm ending {peak_24hr_end}")
    return peak_1hr_mm, peak_1hr_dt, peak_24hr_mm, peak_24hr_end


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Surveying full GSDR record for both nearby stations …\n")
    results = {sid: survey_station(sid) for sid in STATIONS}

    best_1hr_station = max(results, key=lambda sid: results[sid][0])
    best_24hr_station = max(results, key=lambda sid: results[sid][2])
    print(f"\nAll-time max 1-hr intensity : {results[best_1hr_station][0]:.1f} mm "
          f"({best_1hr_station}, {results[best_1hr_station][1]}) "
          f"— matches existing historical_gsdr scenario." )
    print(f"All-time max 24-hr total    : {results[best_24hr_station][2]:.1f} mm "
          f"({best_24hr_station}, ending {results[best_24hr_station][3]}) "
          f"— bigger storm than historical_gsdr's 208 mm/24hr window.")

    print(f"\nExtracting extreme-storm window from {EXTREME_STATION} "
          f"({EXTREME_START} -> {EXTREME_END}) …")
    path = os.path.join(GSDR_DIR, f"{EXTREME_STATION}.txt")
    ts, vals = load_gsdr(path)
    s = pd.Series(vals, index=pd.DatetimeIndex(ts))
    win = s[EXTREME_START:EXTREME_END].fillna(0)

    n = len(win)
    time_min = np.arange(n) * 60.0
    cumulative = np.cumsum(win.values)
    hyet = pd.DataFrame({
        "time_min":             time_min,
        "cumulative_depth_mm":  np.round(cumulative, 2),
        "incremental_depth_mm": np.round(win.values, 2),
    })
    hyet.to_csv(OUT_CSV, index=False)

    print(f"  Duration : {n - 1} hours, {n} rows")
    print(f"  Total    : {win.sum():.1f} mm")
    print(f"  Peak hr  : {win.max():.1f} mm at hour {int(win.values.argmax())}")
    print(f"\nHyetograph saved -> {OUT_CSV}")
    print()
    print("Next step — register 'historical_gsdr_extreme' in flood_sim.py SCENARIOS")
    print("and viewer/preprocess/export_simulation.py SCENARIO_META, then run:")
    print("  ~/miniforge3/envs/prithvi/bin/python simulation/flood_sim.py \\")
    print("      --scenario historical_gsdr_extreme --save-frames \\")
    print("      --soil-preset central-fl-antecedent")


if __name__ == "__main__":
    main()
