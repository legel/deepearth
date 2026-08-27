"""
Build hyetograph for the historical GSDR extreme event from station US_086638.

Station US_086638 — ORLANDO HERNDON AP
  Location : 28.55°N, 81.33°W  (31.8 km from Johns Lake)
  Record   : 1942-01-01 to 1985-01-01 (hourly, mm)
  Source   : INTENSE / GSDR QC_d dataset

Identifies the all-time maximum 1-hr event (143.5 mm, 1960) from the raw gauge
record, extracts a ±12-hr window around the peak hour, and writes a hyetograph
in the same format as the Atlas 14 design storm files so flood_sim.py can run
the 'historical_gsdr' scenario.

Outputs
-------
precipitation/data/hyetograph_historical_gsdr.csv
    columns: time_min, cumulative_depth_mm, incremental_depth_mm

Usage
-----
    python3 precipitation/fetch_gsdr_gauge.py

Requirements
------------
Raw GSDR QC_d file must be present at:
    ~/Desktop/GSDR/QC_d data - US/US_086638.txt
This is the standard INTENSE-format GSDR distribution path.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
GSDR_FILE   = os.path.expanduser(
    "~/Desktop/GSDR/QC_d data - US/US_086638.txt"
)
OUT_CSV     = os.path.join(DATA_DIR, "hyetograph_historical_gsdr.csv")

STATION_ID  = "US_086638"
STATION_NAME = "ORLANDO HERNDON AP"
MISSING_VAL  = -999.0
HEADER_LINES = 21


def load_gsdr(filepath):
    """
    Parse INTENSE-format GSDR file.
    Returns (timestamps, values_mm) as numpy arrays.
    Start datetime is read from header line 8 (0-indexed line 7): 'Start datetime: YYYYMMDDHH'
    """
    start_dt = None
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            if i < HEADER_LINES:
                if line.startswith("Start datetime:"):
                    s = line.split(":", 1)[1].strip()
                    start_dt = datetime(int(s[0:4]), int(s[4:6]), int(s[6:8]), int(s[8:10]))
                continue
            # stop reading header
            break

    if start_dt is None:
        raise RuntimeError("Could not parse 'Start datetime' from GSDR header.")

    # Read all hourly values (one per line, after header)
    raw = []
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            if i < HEADER_LINES:
                continue
            v = line.strip()
            if v == "":
                continue
            raw.append(float(v))

    values = np.array(raw, dtype=np.float64)
    values[values == MISSING_VAL] = np.nan

    n = len(values)
    timestamps = np.array([start_dt + timedelta(hours=h) for h in range(n)],
                          dtype="datetime64[s]")
    return timestamps, values


def find_peak_event(timestamps, values, window_hr=24):
    """
    Find the hour with the maximum 1-hr accumulation.
    Returns (peak_idx, peak_mm, peak_dt).
    """
    peak_idx = int(np.nanargmax(values))
    peak_mm  = values[peak_idx]
    peak_dt  = pd.Timestamp(timestamps[peak_idx])
    return peak_idx, peak_mm, peak_dt


def extract_window(timestamps, values, peak_idx, before_hr=12, after_hr=12):
    """
    Extract [peak_idx - before_hr, peak_idx + after_hr] inclusive.
    Clamps to array bounds. Missing values → 0 (dry hour).
    Returns (window_ts, window_mm).
    """
    n     = len(values)
    start = max(0, peak_idx - before_hr)
    end   = min(n - 1, peak_idx + after_hr)
    win_ts  = timestamps[start : end + 1]
    win_mm  = np.where(np.isnan(values[start : end + 1]), 0.0,
                       values[start : end + 1])
    return win_ts, win_mm


def build_hyetograph(win_ts, win_mm):
    """
    Convert window arrays to hyetograph DataFrame with hourly time steps.
    time_min: minutes from start of window (0, 60, 120, …)
    """
    n = len(win_mm)
    time_min = np.arange(n) * 60.0
    cumulative = np.cumsum(win_mm)
    hyet = pd.DataFrame({
        "time_min":             time_min,
        "cumulative_depth_mm":  np.round(cumulative, 2),
        "incremental_depth_mm": np.round(win_mm, 2),
    })
    return hyet


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(GSDR_FILE):
        print(f"ERROR: GSDR file not found: {GSDR_FILE}")
        print("Expected INTENSE QC_d format file for station US_086638.")
        sys.exit(1)

    print(f"Loading GSDR station {STATION_ID} ({STATION_NAME}) …")
    timestamps, values = load_gsdr(GSDR_FILE)
    n_valid = int(np.sum(~np.isnan(values)))
    print(f"  Records : {len(values):,} hours  ({n_valid:,} non-missing)")
    print(f"  Period  : {pd.Timestamp(timestamps[0])} → {pd.Timestamp(timestamps[-1])}")

    peak_idx, peak_mm, peak_dt = find_peak_event(timestamps, values)
    print(f"\nPeak 1-hr event:")
    print(f"  Depth    : {peak_mm:.1f} mm")
    print(f"  Datetime : {peak_dt}  (local time, UTC-5)")
    print(f"  Index    : hour {peak_idx:,} of record")

    BEFORE_HR = 12
    AFTER_HR  = 12
    win_ts, win_mm = extract_window(timestamps, values, peak_idx,
                                    before_hr=BEFORE_HR, after_hr=AFTER_HR)
    duration_hr = (len(win_mm) - 1)
    total_mm    = float(win_mm.sum())
    print(f"\nExtracted {duration_hr}-hr window (±{BEFORE_HR} hr around peak):")
    print(f"  Start   : {pd.Timestamp(win_ts[0])}")
    print(f"  End     : {pd.Timestamp(win_ts[-1])}")
    print(f"  Total   : {total_mm:.1f} mm over {duration_hr} hours")
    print(f"  Peak hr : {win_mm.max():.1f} mm at hour {int(win_mm.argmax())}")

    hyet = build_hyetograph(win_ts, win_mm)
    hyet.to_csv(OUT_CSV, index=False)
    print(f"\nHyetograph saved → {OUT_CSV}  ({len(hyet)} rows)")
    print()
    print("Next step — run the historical_gsdr simulation scenario:")
    print(f"  ~/miniforge3/envs/prithvi/bin/python simulation/flood_sim.py \\")
    print(f"      --scenario historical_gsdr --save-frames --soil-preset central-fl-antecedent")


if __name__ == "__main__":
    main()
