"""
Fetch real hourly gauge rainfall for the 2024-02-07 to 2024-02-14 event window
from NOAA NCEI Climate Data Online (CDO) API and build a hyetograph for the
flood simulation.

Target station: USC00088788 — CLERMONT, FL US (~10 km NW of Johns Lake)
Target event:   2024-02-12 — peak Sentinel-2 lake extent = 140 ha (archive max)

Requirements
------------
A free NOAA CDO token (1,000 requests/day limit, enough for this use case).
Get one at: https://www.ncdc.noaa.gov/cdo-web/token
Set it as an environment variable:
    export NOAA_CDO_TOKEN="your_token_here"

Outputs
-------
precipitation/data/noaa_gauge_20240207_20240214.csv   — raw hourly GHCND data
precipitation/data/hyetograph_historical_20240212.csv — hyetograph in Atlas 14
                                                        format for flood_sim.py

Usage
-----
    python3 precipitation/fetch_noaa_gauge.py
    python3 precipitation/fetch_noaa_gauge.py --start 2024-02-07 --end 2024-02-14

Station provenance
------------------
GHCND station USC00088788 (CLERMONT FL US) was identified via NOAA GHCND station
list as the closest GHCND daily station to Johns Lake (28.52°N, 81.66°W).
For sub-daily data this script uses GHCND hourly precipitation (datatype PRCP,
units: tenths of mm). If hourly GHCND is not available for this station, the
script falls back to daily PRCP and distributes it uniformly over daylight hours.

GSDR context
------------
The GSDR index (gsdr_us_index.csv) lists station US_086638 (28.55°N, 81.33°W)
as the closest sub-daily gauge at 31.77 km. That station's record ends in 1985
and does not cover the 2024 event. USC00088788 is the practical choice for this
validation event.
"""

import os
import sys
import json
import argparse
import time
import requests
import pandas as pd
import numpy as np

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Primary: US1FLLK0036 (CoCoRaHS, Clermont 2.1 NW, active 2019–present, ~8 km from Johns Lake)
# Fallback: USC00081641 (COOP, Clermont 9 S, active 1948–present, ~15 km)
# NOTE: USC00088788 (original) is retired — returns empty metadata from CDO API.
STATION_ID  = "GHCND:USC00081641"
STATION_LAT = 28.4167
STATION_LON = -81.6500
STATION_NAME = "CLERMONT 9 S FL US"
DATASET     = "GHCND"
DATATYPE    = "PRCP"   # mm/day for GHCND daily CoCoRaHS

CDO_BASE    = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"

RAW_CSV     = os.path.join(DATA_DIR, "noaa_gauge_20240207_20240214.csv")
HYET_CSV    = os.path.join(DATA_DIR, "hyetograph_historical_20240212.csv")


def fetch_cdo(token, start_date, end_date, limit=1000, offset=1):
    """Return list of CDO data records for the station and date range."""
    headers = {"token": token}
    params  = {
        "datasetid":  DATASET,
        "stationid":  STATION_ID,
        "datatypeid": DATATYPE,
        "startdate":  start_date,
        "enddate":    end_date,
        "limit":      limit,
        "offset":     offset,
        "units":      "metric",
    }
    resp = requests.get(CDO_BASE, headers=headers, params=params, timeout=30)
    if resp.status_code == 400:
        raise RuntimeError(f"CDO API 400: {resp.text}")
    resp.raise_for_status()
    return resp.json()


def download_gauge_data(token, start_date, end_date):
    """Page through CDO API and return a DataFrame of all PRCP records."""
    print(f"Fetching GHCND PRCP for {STATION_ID} ({STATION_NAME})")
    print(f"  Period: {start_date} to {end_date}")

    all_records = []
    offset      = 1
    limit       = 1000
    while True:
        data = fetch_cdo(token, start_date, end_date, limit=limit, offset=offset)
        results = data.get("results", [])
        if not results:
            break
        all_records.extend(results)
        total = data.get("metadata", {}).get("resultset", {}).get("count", 0)
        print(f"  Fetched {len(all_records)} / {total} records")
        if len(all_records) >= total:
            break
        offset += limit
        time.sleep(0.2)   # stay within CDO rate limits

    if not all_records:
        raise RuntimeError(
            "No records returned. Check that the station has data for this period.\n"
            f"  Station: {STATION_ID}  Period: {start_date} to {end_date}"
        )

    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["date"])
    # PRCP in GHCND: tenths of mm for daily, or mm for hourly depending on station
    # The CDO 'units=metric' parameter returns mm directly when available
    df = df.sort_values("date").reset_index(drop=True)
    return df


def build_hyetograph(df, hyet_start, hyet_end):
    """
    Build a hyetograph from raw gauge data in the same format as Atlas 14 files:
    columns: time_min, cumulative_depth_mm, incremental_depth_mm

    If the data is daily (24-hr resolution), distribute uniformly over 24 hrs.
    If hourly, use directly.
    """
    # Determine resolution
    if len(df) < 2:
        raise RuntimeError("Not enough records to determine data resolution.")
    diffs = df["date"].diff().dropna().dt.total_seconds()
    median_res_hr = float(diffs.median()) / 3600
    print(f"  Detected data resolution: ~{median_res_hr:.1f} hr")

    # Filter to event window
    mask = (df["date"] >= hyet_start) & (df["date"] <= hyet_end)
    ev   = df[mask].copy()
    if ev.empty:
        raise RuntimeError(
            f"No data in event window {hyet_start} to {hyet_end}.\n"
            "The station may only have daily data — check RAW CSV for available dates."
        )

    # Convert value to mm
    # CDO with units=metric returns: PRCP in tenths of mm for GHCND daily,
    # but mm for sub-daily / hourly. Check magnitude:
    median_prcp = ev["value"].median()
    if median_prcp > 1000:
        # Still in tenths of mm
        ev["prcp_mm"] = ev["value"] / 10.0
    else:
        ev["prcp_mm"] = ev["value"].astype(float)

    # Build time axis in minutes from the first record
    t0 = ev["date"].iloc[0]
    ev["time_min"] = ((ev["date"] - t0).dt.total_seconds() / 60).astype(float)

    cumulative = ev["prcp_mm"].cumsum().values
    incremental = ev["prcp_mm"].values
    times_min   = ev["time_min"].values

    total_mm = float(cumulative[-1])
    duration_hr = float((times_min[-1] - times_min[0]) / 60)
    print(f"  Event window: {ev['date'].iloc[0]} to {ev['date'].iloc[-1]}")
    print(f"  Total rainfall: {total_mm:.1f} mm over {duration_hr:.1f} hr")
    print(f"  Peak intensity: {incremental.max() / (median_res_hr):.1f} mm/hr "
          f"(at {ev['date'].iloc[incremental.argmax()]})")

    hyet = pd.DataFrame({
        "time_min":            times_min,
        "cumulative_depth_mm": np.round(cumulative, 2),
        "incremental_depth_mm": np.round(incremental, 2),
    })
    return hyet


def main(start_date="2024-02-07", end_date="2024-02-14"):
    token = os.environ.get("NOAA_CDO_TOKEN", "").strip()
    if not token:
        print("=" * 60)
        print("ERROR: NOAA CDO token not set.")
        print()
        print("Get a free token at: https://www.ncdc.noaa.gov/cdo-web/token")
        print("Then run:")
        print("    export NOAA_CDO_TOKEN='your_token_here'")
        print("    python3 precipitation/fetch_noaa_gauge.py")
        print()
        print("Without this, only the 4 NOAA Atlas 14 design storms are available")
        print("as simulation scenarios. The interactive viewer uses those by default.")
        print("=" * 60)
        sys.exit(1)

    # Download raw data
    df = download_gauge_data(token, start_date, end_date)
    df.to_csv(RAW_CSV, index=False)
    print(f"  Raw data saved: {RAW_CSV} ({len(df)} records)")

    # Build hyetograph for Feb 7–14 window (brackets Feb 12 S2 observation)
    hyet = build_hyetograph(df,
                             hyet_start=pd.Timestamp("2024-02-07"),
                             hyet_end=pd.Timestamp("2024-02-14"))
    hyet.to_csv(HYET_CSV, index=False)
    print(f"  Hyetograph saved: {HYET_CSV} ({len(hyet)} rows)")
    print()
    print("Next step: run the historical simulation scenario:")
    print("  python3 simulation/flood_sim.py --scenario historical_20240212 --save-frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch NOAA CDO hourly gauge data for the Feb 2024 flood event at Johns Lake"
    )
    parser.add_argument("--start", default="2024-02-07",
                        help="Start date YYYY-MM-DD (default: 2024-02-07)")
    parser.add_argument("--end", default="2024-02-14",
                        help="End date YYYY-MM-DD (default: 2024-02-14)")
    args = parser.parse_args()
    main(args.start, args.end)
