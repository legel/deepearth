"""
Fetch USGS NWIS instantaneous-values discharge for Gee Creek (02234400), any date range.

The existing gee_creek_ian_discharge.csv / gee_creek_milton_discharge.csv on disk were both
fetched ad hoc (see cfx_sr417/CLAUDE.md's 2026-07-27 entry) with no saved script -- this is the
first reusable version, needed now to extend the Ian observed record past its current Oct 3
2022 cutoff for the window-truncation follow-up test (see ../../NEXT_STEPS.md, 2026-09-03/04
entries: neither the Ian nor Milton extended-window comparison had reached a fully-drained
endpoint on either side).

Usage:
    python3 site3_gee_creek/infrastructure/fetch_gauge_discharge_site3.py --start 2022-09-26 --end 2022-10-15 --out gee_creek_ian_discharge_extended.csv
"""
import os
import argparse
import requests
import pandas as pd

SITE_NO = "02234400"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NWIS_IV_URL = "https://waterservices.usgs.gov/nwis/iv/"


def fetch(start, end):
    params = {
        "format": "json",
        "sites": SITE_NO,
        "startDT": start,
        "endDT": end,
        "parameterCd": "00060",  # discharge, cfs
    }
    r = requests.get(NWIS_IV_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    ts = data["value"]["timeSeries"][0]["values"][0]["value"]
    df = pd.DataFrame(ts)
    df["dateTime"] = pd.to_datetime(df["dateTime"], utc=True)
    df["discharge_cfs"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["dateTime", "discharge_cfs", "qualifiers"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = fetch(args.start, args.end)
    out_path = os.path.join(OUT_DIR, args.out)
    df.to_csv(out_path, index=False)
    print(f"Fetched {len(df)} rows, {df['dateTime'].min()} to {df['dateTime'].max()}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
