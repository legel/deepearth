"""
GSDR US — Sub-Daily Intensity Matrix
======================================
For a given coordinate, finds all QC'd GSDR hourly gauge stations within
each search radius (default: 10, 50, 100 km) and returns the all-time
observed maximum precipitation accumulation for each duration window:

    Durations : 1, 3, 6, 12, 24 hours
    Radii     : 10, 50, 100 km  (cumulative, ≤r)

Output is the raw observed maximum — no curve fitting, no distribution,
consistent with the HYADES all-time-max approach in hyades_flood_risk_matrix.py.

The 24-hr row can be directly compared to HYADES as a cross-validation.

Values are in mm (total accumulation over the duration window).
Divide by hours to get average intensity in mm/hr if needed.

Environment variable:
    GSDR_QC_DIR : path to the QC_d data - US folder
                  default: ~/Desktop/GSDR/QC_d data - US

Usage:
    python3 gsdr/gsdr_intensity_matrix.py --lat 28.5652 --lon -81.5868
    python3 gsdr_intensity_matrix.py --lat 37.38575 --lon -122.00037

Prerequisite:
    Run gsdr_build_index.py once to generate gsdr_us_index.csv.
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
QC_DIR     = os.environ.get("GSDR_QC_DIR", os.path.expanduser("~/Desktop/GSDR/QC_d data - US"))
INDEX_PATH = os.path.join(BASE_DIR, "gsdr_us_index.csv")
MM_TO_IN   = 1.0 / 25.4
HEADER_LINES = 21
MISSING_VAL  = -999.0


# ── Haversine ──────────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
         * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ── Parse start datetime from INTENSE header ───────────────────────────────

def parse_start_dt(start_str):
    """Parse YYYYMMDDHH string → datetime."""
    s = str(start_str).strip()
    if len(s) < 10:
        return None
    try:
        return datetime(int(s[0:4]), int(s[4:6]), int(s[6:8]), int(s[8:10]))
    except Exception:
        return None


# ── Load one gauge file → numpy array of hourly mm values ─────────────────

def load_gauge(filepath):
    """
    Returns (data_mm, start_dt) where data_mm is a float32 array of hourly
    precipitation values in mm (-999 already replaced with NaN).
    """
    header = {}
    data_lines = []
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            if i < HEADER_LINES:
                if ":" in line:
                    k, _, v = line.partition(":")
                    header[k.strip()] = v.strip()
            else:
                data_lines.append(line.strip())

    data = np.array(data_lines, dtype=np.float32)
    data[data == MISSING_VAL] = np.nan
    start_dt = parse_start_dt(header.get("Start datetime", ""))
    return data, start_dt


# ── Max rolling accumulation for one gauge ─────────────────────────────────

def max_rolling(data, start_dt, durations):
    """
    For each duration (hours), compute the rolling sum and return:
        (max_mm, year_of_max)
    using pandas rolling with min_periods=duration (all hours must be valid).
    """
    s = pd.Series(data)
    results = {}
    for d in durations:
        rolled = s.rolling(window=d, min_periods=d).sum()
        idx    = rolled.idxmax()
        if pd.isna(idx) or pd.isna(rolled[idx]):
            results[d] = (np.nan, "N/A")
        else:
            max_val = float(rolled[idx])
            if start_dt is not None:
                year = (start_dt + timedelta(hours=int(idx))).year
            else:
                year = "N/A"
            results[d] = (max_val, year)
    return results


# ── Core function ──────────────────────────────────────────────────────────

def gsdr_intensity_matrix(
    lat,
    lon,
    index_path=INDEX_PATH,
    qc_dir=QC_DIR,
    radii_km=None,
    durations_hr=None,
    max_pct_missing=50.0,
    save_csv=True,
    output_dir=None,
    verbose=True,
):
    """
    All-time maximum sub-daily precipitation accumulation by radius and duration.

    Parameters
    ----------
    lat, lon         : float — WGS-84 decimal degrees
    index_path       : str   — path to gsdr_us_index.csv
    qc_dir           : str   — path to QC_d data - US folder
    radii_km         : list  — search radii in km (default: [10, 50, 100])
    durations_hr     : list  — accumulation windows in hours (default: [1,3,6,12,24])
    max_pct_missing  : float — exclude stations with more than this % missing data
    save_csv         : bool  — save result to outputs/
    output_dir       : str   — folder for CSV output
    verbose          : bool  — print diagnostics

    Returns
    -------
    pd.DataFrame — rows=durations, columns=radii, values=max mm observed
    """
    if radii_km is None:
        radii_km = [10, 50, 100]
    if durations_hr is None:
        durations_hr = [1, 3, 6, 12, 24]

    # ── Load index ─────────────────────────────────────────────────────────
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Index not found: {index_path}\n"
            "Run gsdr_build_index.py first."
        )
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "outputs")
    index = pd.read_csv(index_path)
    index = index[index["PCT_MISSING"] <= max_pct_missing].copy()

    # ── Compute distances ──────────────────────────────────────────────────
    index["dist_km"] = haversine(lat, lon,
                                 index["LAT"].values,
                                 index["LON"].values)
    nearest = index.nsmallest(1, "dist_km").iloc[0]

    if verbose:
        print(f"Target            : lat={lat}, lon={lon}")
        print(f"Nearest station   : {nearest['ID']}  ({nearest['dist_km']:.2f} km away)")
        print(f"Stations after missing-data filter (≤{max_pct_missing}%): {len(index):,}\n")

    # ── Identify stations per radius ───────────────────────────────────────
    radius_ids = {}
    for r in radii_km:
        ids = index.loc[index["dist_km"] <= r, "ID"].tolist()
        radius_ids[r] = ids
        if verbose:
            print(f"  ≤ {r:>3} km — {len(ids)} station(s)")
            sub = index[index["ID"].isin(ids)].sort_values("dist_km")
            for _, row in sub.iterrows():
                print(f"      {row['ID']}  lat={row['LAT']:.4f}  lon={row['LON']:.4f}"
                      f"  dist={row['dist_km']:.1f} km  missing={row['PCT_MISSING']:.1f}%")
    if verbose:
        print()

    # ── Load gauge data and compute rolling maxima ─────────────────────────
    # Collect all unique station IDs needed across all radii
    all_ids = set(sid for ids in radius_ids.values() for sid in ids)

    if verbose:
        print(f"Loading {len(all_ids)} unique gauge file(s) …")

    gauge_results = {}   # station_id → {duration: (max_mm, year)}
    for sid in sorted(all_ids):
        fpath = os.path.join(qc_dir, f"{sid}.txt")
        if not os.path.exists(fpath):
            if verbose:
                print(f"  WARNING: file not found for {sid}, skipping.")
            continue
        try:
            data, start_dt = load_gauge(fpath)
            gauge_results[sid] = max_rolling(data, start_dt, durations_hr)
            if verbose:
                start_yr = start_dt.year if start_dt else "?"
                print(f"  {sid}  {len(data):,} hours  start={start_yr}"
                      f"  1hr-max={gauge_results[sid][1][0]:.1f}mm"
                      f"  24hr-max={gauge_results[sid][24][0]:.1f}mm"
                      if 24 in durations_hr else "")
        except Exception as e:
            if verbose:
                print(f"  WARNING: error loading {sid}: {e}")

    if verbose:
        print()

    # ── Assemble result matrix ─────────────────────────────────────────────
    # For each (duration, radius): find the station with the highest max
    dur_labels = {1: "1-hr", 3: "3-hr", 6: "6-hr", 12: "12-hr", 24: "24-hr"}

    col_mm  = [f"≤{r} km (mm)" for r in radii_km]
    col_in  = [f"≤{r} km (in)" for r in radii_km]
    col_st  = [f"Station ≤{r}km" for r in radii_km]
    col_yr  = [f"Year ≤{r}km"    for r in radii_km]
    col_cnt = [f"Count ≤{r}km"   for r in radii_km]

    rows = []
    row_labels = []
    provenance = {}  # for verbose printing

    for d in durations_hr:
        row = {}
        prov = {}
        for r in radii_km:
            ids = radius_ids[r]
            best_mm, best_yr, best_st = np.nan, "N/A", "N/A"
            for sid in ids:
                if sid not in gauge_results:
                    continue
                mm, yr = gauge_results[sid][d]
                if not np.isnan(mm) and (np.isnan(best_mm) or mm > best_mm):
                    best_mm, best_yr, best_st = mm, yr, sid
            row[f"≤{r} km (mm)"]   = round(best_mm, 1) if not np.isnan(best_mm) else np.nan
            row[f"≤{r} km (in)"]   = round(best_mm * MM_TO_IN, 2) if not np.isnan(best_mm) else np.nan
            row[f"Station ≤{r}km"] = best_st
            row[f"Year ≤{r}km"]    = best_yr
            row[f"Count ≤{r}km"]   = len(ids)
            prov[r] = (best_mm, best_yr, best_st)
        rows.append(row)
        row_labels.append(dur_labels.get(d, f"{d}-hr"))
        provenance[d] = prov

    result = pd.DataFrame(rows, index=row_labels)
    result.index.name = "Duration"

    # ── Print table ────────────────────────────────────────────────────────
    if verbose:
        w = 14
        print("=" * 72)
        print("ALL-TIME MAXIMUM SUB-DAILY PRECIPITATION  (mm accumulated)")
        print("(raw observed maximum — no curve fitting)")
        print("=" * 72)
        header = f"{'Duration':<10}" + "".join(f"{'≤'+str(r)+' km':>{w}}" for r in radii_km)
        sep    = "-" * (10 + w * len(radii_km))
        print(header + "   [mm]")
        print(sep)
        for d in durations_hr:
            row_s = f"{dur_labels[d]:<10}"
            for r in radii_km:
                v = provenance[d][r][0]
                row_s += (f"{v:.1f} mm" if not np.isnan(v) else "N/A").rjust(w)
            print(row_s)
        print()
        print(header + "   [inches]")
        print(sep)
        for d in durations_hr:
            row_s = f"{dur_labels[d]:<10}"
            for r in radii_km:
                v = provenance[d][r][0]
                v_in = v * MM_TO_IN if not np.isnan(v) else np.nan
                row_s += (f"{v_in:.2f} in" if not np.isnan(v_in) else "N/A").rjust(w)
            print(row_s)
        print(sep)
        print("\nProvenance (station that recorded each maximum):")
        print(sep)
        for d in durations_hr:
            row_s = f"{dur_labels[d]:<10}"
            for r in radii_km:
                st = provenance[d][r][2]
                row_s += str(st).rjust(w)
            print(row_s)
        for d in durations_hr:
            row_s = f"{dur_labels[d]+' yr':<10}"
            for r in radii_km:
                yr = provenance[d][r][1]
                row_s += str(yr).rjust(w)
            print(row_s)
        print("=" * 72)
        print()

    # ── Save CSV ───────────────────────────────────────────────────────────
    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        lat_tag = f"{lat:.4f}".replace("-", "S").replace(".", "p")
        lon_tag = f"{lon:.4f}".replace("-", "W").replace(".", "p")
        fpath   = os.path.join(output_dir, f"gsdr_intensity_{lat_tag}_{lon_tag}.csv")

        meta = (
            f"# GSDR US — All-Time Maximum Sub-Daily Precipitation\n"
            f"# Location        : lat={lat}, lon={lon}\n"
            f"# Radii (km)      : {radii_km}\n"
            f"# Durations (hr)  : {durations_hr}\n"
            f"# Method          : raw observed rolling maximum (no curve fitting)\n"
            f"# Units           : mm (total accumulation over duration window)\n"
            f"# Dataset         : GSDR QC_d US (INTENSE format, hourly, ~1948–2014)\n"
            f"# Missing filter  : stations with >{max_pct_missing}% missing excluded\n"
            f"# Nearest station : {nearest['ID']} ({nearest['dist_km']:.2f} km)\n"
            f"# Cross-check     : 24-hr row should be comparable to HYADES all-time max\n"
            f"#\n"
        )
        with open(fpath, "w") as f:
            f.write(meta)
            result.to_csv(f)
        if verbose:
            print(f"CSV saved → {fpath}")

    return result


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="All-time max sub-daily precip by duration and search radius (GSDR)."
    )
    parser.add_argument("--lat",      type=float, required=True)
    parser.add_argument("--lon",      type=float, required=True)
    parser.add_argument("--radii",    type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--durations",type=int, nargs="+", default=[1, 3, 6, 12, 24])
    parser.add_argument("--index",    default=INDEX_PATH)
    parser.add_argument("--qcdir",    default=QC_DIR)
    parser.add_argument("--outdir",   default="outputs")
    parser.add_argument("--no-csv",   action="store_true")
    args = parser.parse_args()

    gsdr_intensity_matrix(
        lat=args.lat,
        lon=args.lon,
        index_path=args.index,
        qc_dir=args.qcdir,
        radii_km=args.radii,
        durations_hr=args.durations,
        save_csv=not args.no_csv,
        output_dir=args.outdir,
        verbose=True,
    )
