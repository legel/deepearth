#!/usr/bin/env python3
"""
KSFB (ASOS, Orlando Sanford Intl) reliability cross-check for site3.
================================================================================
Closes a real, documented gap (CLAUDE.md's 2026-07-27 dataset-selection-logic audit):
site3's precipitation source (KSFB) was picked on proximity alone (10.8km, vs the main AOI's
own KMCO at 29.1km) and never got the reliability cross-check the main AOI's own precipitation
choice DID get. The main AOI's MCO-vs-ISM cross-check found a REAL problem via this method —
ISM looked closer (8.6km) but had r=0.166 vs GHCND and reported 0.0mm during Hurricane Ian
itself (sensor offline exactly when it mattered) — proximity alone is not sufficient evidence
of reliability. See precipitation/DATA_SOURCES.md for the main AOI's own methodology, which
this script reproduces for site3, reusing its functions directly rather than re-implementing.

Method (identical to the main AOI's MCO/ISM check):
1. Find a nearby NOAA GHCND daily CoOp station (independent instrument type/network from ASOS)
   via fetch_precip_seasonality.py's own discover_stations()/filter_candidates().
2. Fetch KSFB's FULL multi-year hourly ASOS record (previously only ever fetched for the 6-day
   Ian window — see site3_gee_creek/precipitation/data/asos_hourly_SFB.csv, 121 rows) and
   aggregate to daily totals, reusing fetch_asos_hourly.py's fetch_iem_asos()/extract_hourly().
3. Correlate KSFB daily vs the GHCND daily record over their real overlap period.
4. Specifically check what KSFB reported during Hurricane Ian — the exact failure mode that
   sank ISM.

Usage: python3 site3_gee_creek/precipitation/verify_ksfb_reliability.py
Requires: NOAA_CDO_TOKEN env var (same token flood_hydrology/.env already has).
"""
import os
import sys
import json

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_SITE3_DIR = os.path.dirname(_HERE)
_PROJ_DIR = os.path.dirname(_SITE3_DIR)
sys.path.insert(0, os.path.join(_PROJ_DIR, "precipitation"))

from fetch_precip_seasonality import (          # noqa: E402
    discover_stations, filter_candidates, download_daily_prcp, require_token,
)
from fetch_asos_hourly import fetch_iem_asos, extract_hourly   # noqa: E402

DATA_DIR = os.path.join(_HERE, "data")
os.makedirs(DATA_DIR, exist_ok=True)

SITE3_LAT, SITE3_LON = 28.690514, -81.287539
KSFB_STATION = "SFB"
KSFB_DIST_KM = 10.8
START, END = "2021-01-01", "2026-06-22"     # same window fetch_asos_hourly.py's main() uses

# Hurricane Ian's real UTC window (same dates every other Ian analysis in this project uses).
IAN_WINDOW = ("2022-09-27", "2022-09-30")


def find_reference_station(token):
    """Nearest reliable-looking GHCND daily station to site3 — an independent cross-check
    instrument (CoOp daily gauge, not ASOS), same role Kissimmee-2 plays for the main AOI."""
    stations = discover_stations(token, SITE3_LAT, SITE3_LON, search_radius_km=25)
    candidates = filter_candidates(stations, SITE3_LAT, SITE3_LON)
    if not candidates:
        raise SystemExit("No GHCND candidates found within 25km of site3.")
    best = candidates[0]
    print(f"  Reference station: {best['id']} ({best['name']}), {best['distance_km']}km, "
          f"{best['record_years']}yr record, {best['datacoverage']*100:.1f}% coverage")
    return best


def main():
    token = require_token()

    print("═" * 70)
    print("  Site3 precipitation reliability cross-check: KSFB vs. an independent gauge")
    print("═" * 70)

    print("\n[1/4] Finding a reference GHCND daily station near site3 …")
    ref = find_reference_station(token)

    print(f"\n[2/4] Fetching {ref['id']} daily record ({START} to {END}) …")
    ghcnd = download_daily_prcp(token, ref["id"], START, END)
    ghcnd_path = os.path.join(DATA_DIR, f"ghcnd_daily_{ref['id'].replace(':', '_')}.csv")
    ghcnd.to_csv(ghcnd_path, index=False)
    ghcnd_s = ghcnd.set_index("date")["prcp_mm"]
    print(f"  {len(ghcnd_s)} daily records saved → {ghcnd_path}")

    print(f"\n[3/4] Fetching KSFB FULL hourly ASOS record ({START} to {END}) — "
          f"previously only ever fetched for the 6-day Ian window …")
    raw = fetch_iem_asos(KSFB_STATION, START, END)
    hourly = extract_hourly(raw)
    hourly_path = os.path.join(DATA_DIR, "asos_hourly_SFB_full.csv")
    hourly.to_csv(hourly_path, header=True)
    ksfb_daily = hourly.resample("D").sum()
    ksfb_daily.index = ksfb_daily.index.tz_localize(None)
    print(f"  {len(hourly)} hourly records ({hourly.index.min()} to {hourly.index.max()}) "
          f"→ {hourly_path}")

    print(f"\n[4/4] Cross-validating KSFB daily totals against {ref['id']} …")
    both = pd.DataFrame({"ksfb": ksfb_daily, "ghcnd": ghcnd_s}).dropna()
    r = float(np.corrcoef(both["ksfb"], both["ghcnd"])[0, 1])
    bias = float((both["ksfb"] - both["ghcnd"]).mean())
    rmse = float(np.sqrt(((both["ksfb"] - both["ghcnd"]) ** 2).mean()))

    print(f"\n  Overlap: n={len(both)} days ({both.index.min().date()} to {both.index.max().date()})")
    print(f"  r = {r:.3f}   bias = {bias:+.2f} mm/day   RMSE = {rmse:.2f} mm/day")

    # The specific failure mode that sank ISM: did the sensor go quiet during a real storm?
    ian_ksfb = ksfb_daily[IAN_WINDOW[0]:IAN_WINDOW[1]].sum()
    ian_ghcnd = ghcnd_s[IAN_WINDOW[0]:IAN_WINDOW[1]].sum()
    print(f"\n  Hurricane Ian window ({IAN_WINDOW[0]} to {IAN_WINDOW[1]}):")
    print(f"    KSFB total:  {ian_ksfb:.1f} mm")
    print(f"    {ref['id']} total: {ian_ghcnd:.1f} mm")
    ian_ratio = ian_ksfb / ian_ghcnd if ian_ghcnd > 0 else float("nan")
    print(f"    ratio: {ian_ratio:.2f}")

    # Verdict, same evidentiary bar the main AOI's own MCO/ISM check used.
    verdict = "RELIABLE" if (r >= 0.4 and ian_ksfb > 0.3 * ian_ghcnd) else "QUESTIONABLE"
    if r < 0.2 or ian_ksfb < 0.05 * ian_ghcnd:
        verdict = "DO NOT USE"
    print(f"\n  VERDICT: {verdict}")
    if verdict == "RELIABLE":
        print(f"  KSFB's r={r:.3f} is comparable to or better than the main AOI's accepted MCO "
              f"(r=0.578), and it captured Ian ({ian_ksfb:.0f}mm, {ian_ratio:.2f}x the reference "
              f"station) rather than going quiet the way ISM did (0.0mm during the same storm). "
              f"No evidence of the ISM-style sensor-outage failure mode.")
    elif verdict == "QUESTIONABLE":
        print(f"  Correlation or Ian capture is weaker than the main AOI's accepted MCO "
              f"threshold — worth a closer look before treating KSFB with the same confidence.")
    else:
        print(f"  Same evidentiary pattern that sank ISM for the main AOI — recommend NOT "
              f"using KSFB without a replacement source.")

    summary = dict(
        generated="2026-08-04",
        purpose="Closes the site3 precipitation-reliability gap flagged in CLAUDE.md's "
                "2026-07-27 dataset-selection-logic audit",
        ksfb_station="SFB", ksfb_dist_km=KSFB_DIST_KM,
        reference_station=ref["id"], reference_name=ref["name"],
        reference_dist_km=ref["distance_km"], reference_record_years=ref["record_years"],
        overlap_days=int(len(both)),
        overlap_start=str(both.index.min().date()), overlap_end=str(both.index.max().date()),
        correlation_r=round(r, 4), bias_mm_day=round(bias, 3), rmse_mm_day=round(rmse, 3),
        ian_window=IAN_WINDOW,
        ian_ksfb_total_mm=round(float(ian_ksfb), 1),
        ian_reference_total_mm=round(float(ian_ghcnd), 1),
        ian_ratio=round(float(ian_ratio), 3),
        main_aoi_reference_mco_r=0.578,
        main_aoi_reference_ism_r=0.166,
        verdict=verdict,
    )
    summary_path = os.path.join(DATA_DIR, "ksfb_reliability_check.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {summary_path}")


if __name__ == "__main__":
    main()
