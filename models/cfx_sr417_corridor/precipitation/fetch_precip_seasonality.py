"""
Precipitation Wet/Dry Seasonality — SR417 Corridor AOI
==========================================================
Builds a monthly precipitation time series for the SR417 corridor AOI
(28.36687N, -81.43299W) and labels each month wet/dry, to characterize
the AOI's rainfall seasonality ahead of hydrology/erosion modeling.

Unlike flood_hydrology/precipitation/fetch_noaa_gauge.py (Johns Lake),
which used a single hand-picked station for a one-week event window, this
script DISCOVERS a nearby long-record station programmatically via NOAA
CDO's /stations endpoint — there is no pre-identified gauge for this AOI.

Wet/dry convention: wet = Jun-Sep, dry = Oct-May, matching the existing
flood_hydrology/ground_truth/seasonal_area_split.py convention (kept for
cross-project consistency; note the official NWS Florida wet season is
May-Oct — this repo's Jun-Sep/Oct-May convention is a deliberate choice
to stay comparable with the Johns Lake project, not an independent
re-derivation).

Requires a free NOAA CDO token:
    export NOAA_CDO_TOKEN="your_token_here"
    (get one at https://www.ncdc.noaa.gov/cdo-web/token)

Outputs (saved under precipitation/data/):
    daily_precip_raw.csv             — raw daily GHCND PRCP records
    monthly_precip_timeseries.csv    — monthly totals + wet/dry label
    monthly_precip_timeseries.png    — bar chart, colored by season
    station_metadata.json            — chosen station provenance + candidates

Usage:
    python3 precipitation/fetch_precip_seasonality.py
    python3 precipitation/fetch_precip_seasonality.py --lat 28.36687 --lon -81.43299 --radius_km 1.0
"""

import os
import sys
import json
import time
import argparse
import requests
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_LAT = 28.36687   # CFX SR417 corridor test-landscape AOI (Lake Nona / south Orlando)
DEFAULT_LON = -81.43299

STATION_SEARCH_RADIUS_KM = 25   # wider than the 2km AOI box — daily GHCND stations are sparse
WET_MONTHS = {5, 6, 7, 8, 9, 10}  # NWS official Florida wet season (May–Oct)
MIN_RECORD_YEARS = 20            # filter floor for "long-record" station
MIN_DATACOVERAGE = 0.7
RECENT_YEARS_FLOOR = 5           # station's maxdate must be within this many years of today

CDO_STATIONS_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2/stations"
CDO_DATA_URL     = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"

DAILY_RAW_CSV  = os.path.join(DATA_DIR, "daily_precip_raw.csv")
MONTHLY_CSV    = os.path.join(DATA_DIR, "monthly_precip_timeseries.csv")
MONTHLY_PNG    = os.path.join(DATA_DIR, "monthly_precip_timeseries.png")
STATION_JSON   = os.path.join(DATA_DIR, "station_metadata.json")


def bbox_from_center(lat, lon, radius_km):
    """Return (west, south, east, north) bounding box in EPSG:4326."""
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(lat))
    dlat = radius_km / km_per_deg_lat
    dlon = radius_km / km_per_deg_lon
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def discover_stations(token, lat, lon, search_radius_km):
    """GET NOAA CDO /stations for GHCND stations intersecting a search bbox around the AOI."""
    west, south, east, north = bbox_from_center(lat, lon, search_radius_km)
    headers = {"token": token}
    stations = []
    offset = 1
    limit = 1000
    print(f"Discovering GHCND stations within {search_radius_km} km of ({lat}, {lon}) …")
    while True:
        params = {
            "datasetid": "GHCND",
            "extent": f"{south},{west},{north},{east}",
            "limit": limit,
            "offset": offset,
        }
        resp = requests.get(CDO_STATIONS_URL, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            break
        stations.extend(results)
        total = data.get("metadata", {}).get("resultset", {}).get("count", 0)
        if len(stations) >= total:
            break
        offset += limit
        time.sleep(0.2)
    print(f"  Found {len(stations)} candidate station(s)")
    return stations


def filter_candidates(stations, lat, lon):
    """
    Rank candidate stations by distance, after filtering on record length,
    recency, and data coverage. Returns a list of dicts sorted by distance
    ascending (best candidate first) — caller should sanity-check the top
    few rather than trust the filter blindly.
    """
    now_year = pd.Timestamp.now().year
    candidates = []
    for s in stations:
        try:
            mindate = pd.Timestamp(s["mindate"])
            maxdate = pd.Timestamp(s["maxdate"])
        except (KeyError, ValueError):
            continue
        record_years = (maxdate - mindate).days / 365.25
        if record_years < MIN_RECORD_YEARS:
            continue
        if (now_year - maxdate.year) > RECENT_YEARS_FLOOR:
            continue
        coverage = s.get("datacoverage", 0)
        if coverage < MIN_DATACOVERAGE:
            continue
        dist_km = haversine_km(lat, lon, s["latitude"], s["longitude"])
        candidates.append({
            "id": s["id"], "name": s.get("name", ""),
            "latitude": s["latitude"], "longitude": s["longitude"],
            "mindate": s["mindate"], "maxdate": s["maxdate"],
            "record_years": round(record_years, 1),
            "datacoverage": coverage,
            "distance_km": round(dist_km, 2),
        })
    candidates.sort(key=lambda c: c["distance_km"])
    return candidates


def print_candidate_table(candidates, n=5):
    print("\n── Top station candidates ───────────────────────────────────")
    print(f"{'id':<18} {'name':<30} {'dist_km':>8} {'years':>6} {'coverage':>9} {'range'}")
    for c in candidates[:n]:
        print(f"{c['id']:<18} {c['name'][:30]:<30} {c['distance_km']:>8} "
              f"{c['record_years']:>6} {c['datacoverage']:>9} "
              f"{c['mindate'][:10]}–{c['maxdate'][:10]}")


def fetch_cdo_page(token, station_id, start_date, end_date, limit=1000, offset=1, max_retries=4):
    headers = {"token": token}
    params = {
        "datasetid": "GHCND",
        "stationid": station_id,
        "datatypeid": "PRCP",
        "startdate": start_date,
        "enddate": end_date,
        "limit": limit,
        "offset": offset,
        "units": "metric",
    }
    for attempt in range(max_retries):
        try:
            resp = requests.get(CDO_DATA_URL, headers=headers, params=params, timeout=30)
            if resp.status_code == 400:
                raise RuntimeError(f"CDO API 400: {resp.text}")
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as exc:
            if attempt == max_retries - 1:
                raise
            backoff = 2 ** attempt
            print(f"    ⚠ transient network error ({exc.__class__.__name__}), "
                  f"retrying in {backoff}s …")
            time.sleep(backoff)


def download_daily_prcp(token, station_id, start_date, end_date):
    """
    Download full daily PRCP record for a station, chunked by calendar year
    (CDO single-request results can silently truncate over long multi-decade
    ranges — chunking avoids relying on an unverified single-request ceiling).
    """
    start_year = pd.Timestamp(start_date).year
    end_year = pd.Timestamp(end_date).year
    all_records = []

    for year in range(start_year, end_year + 1):
        year_start = max(pd.Timestamp(start_date), pd.Timestamp(f"{year}-01-01")).strftime("%Y-%m-%d")
        year_end = min(pd.Timestamp(end_date), pd.Timestamp(f"{year}-12-31")).strftime("%Y-%m-%d")

        offset = 1
        limit = 1000
        year_records = []
        try:
            while True:
                data = fetch_cdo_page(token, station_id, year_start, year_end, limit=limit, offset=offset)
                results = data.get("results", [])
                if not results:
                    break
                year_records.extend(results)
                total = data.get("metadata", {}).get("resultset", {}).get("count", 0)
                if len(year_records) >= total:
                    break
                offset += limit
                time.sleep(0.2)
        except requests.exceptions.RequestException as exc:
            # Retries inside fetch_cdo_page already exhausted — skip this year
            # rather than discarding every other year already downloaded.
            print(f"  {year}: SKIPPED after retries exhausted ({exc.__class__.__name__})")
            time.sleep(0.2)
            continue

        if year_records:
            all_records.extend(year_records)
            print(f"  {year}: {len(year_records)} record(s)")
        time.sleep(0.2)

    if not all_records:
        raise RuntimeError(f"No PRCP records returned for {station_id} over {start_date} to {end_date}.")

    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["date"])
    df["prcp_mm"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["date", "prcp_mm"]].dropna().sort_values("date").reset_index(drop=True)
    return df


def aggregate_monthly(daily_df):
    """
    Sum daily PRCP to monthly totals; flag low-completeness months rather
    than silently treating gaps as low rainfall (gaps would otherwise bias
    the wet/dry comparison).
    """
    df = daily_df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    grouped = df.groupby(["year", "month"]).agg(
        monthly_total_mm=("prcp_mm", "sum"),
        n_days_reported=("prcp_mm", "count"),
    ).reset_index()

    grouped["days_in_month"] = grouped.apply(
        lambda r: pd.Timestamp(year=int(r["year"]), month=int(r["month"]), day=1).days_in_month, axis=1
    )
    grouped["complete"] = grouped["n_days_reported"] >= 20
    grouped["season"] = grouped["month"].apply(lambda m: "wet" if m in WET_MONTHS else "dry")
    grouped["date"] = pd.to_datetime(dict(year=grouped["year"], month=grouped["month"], day=1))
    return grouped.sort_values("date").reset_index(drop=True)


def summarize_seasonality(monthly_df):
    """Mirror ground_truth/seasonal_area_split.py's summarize()/print style."""
    complete = monthly_df[monthly_df["complete"]]
    wet = complete.loc[complete["season"] == "wet", "monthly_total_mm"]
    dry = complete.loc[complete["season"] == "dry", "monthly_total_mm"]

    print("\n── Wet vs Dry Season Summary (complete months only) ──────────")
    print(f"Total months: {len(monthly_df)}  (complete={len(complete)}, "
          f"incomplete={len(monthly_df) - len(complete)})")
    print()
    print(f"{'season':6} {'n':>4} {'mean_mm':>9} {'median_mm':>10} {'stdev_mm':>9} {'min_mm':>8} {'max_mm':>8}")
    for label, vals in [("wet", wet), ("dry", dry)]:
        if len(vals) == 0:
            print(f"{label:6} {0:>4}  (no complete months)")
            continue
        print(f"{label:6} {len(vals):>4} {vals.mean():>9.1f} {vals.median():>10.1f} "
              f"{vals.std():>9.1f} {vals.min():>8.1f} {vals.max():>8.1f}")

    if len(wet) and len(dry):
        diff = wet.mean() - dry.mean()
        print()
        print(f"Wet-minus-dry mean difference: {diff:+.1f} mm "
              f"({100 * diff / dry.mean():+.1f}% relative to dry-season mean)")


def plot_timeseries(monthly_df, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = monthly_df["season"].map({"wet": "#2b6cb0", "dry": "#c2935c"})
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(monthly_df["date"], monthly_df["monthly_total_mm"], width=20, color=colors)
    ax.set_ylabel("Monthly precipitation (mm)")
    ax.set_title("SR417 Corridor AOI — Monthly Precipitation (blue=wet Jun-Sep, tan=dry Oct-May)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved plot → {out_path}")


def require_token():
    token = os.environ.get("NOAA_CDO_TOKEN", "").strip()
    if not token:
        print("=" * 60)
        print("ERROR: NOAA CDO token not set.")
        print()
        print("Get a free token at: https://www.ncdc.noaa.gov/cdo-web/token")
        print("Then run:")
        print("    export NOAA_CDO_TOKEN='your_token_here'")
        print("    python3 precipitation/fetch_precip_seasonality.py")
        print("=" * 60)
        sys.exit(1)
    return token


def main(lat=DEFAULT_LAT, lon=DEFAULT_LON, radius_km=1.0):
    token = require_token()

    stations = discover_stations(token, lat, lon, STATION_SEARCH_RADIUS_KM)
    candidates = filter_candidates(stations, lat, lon)
    if not candidates:
        sys.exit(f"No qualifying long-record GHCND station found within "
                 f"{STATION_SEARCH_RADIUS_KM} km. Try increasing STATION_SEARCH_RADIUS_KM.")
    print_candidate_table(candidates)

    chosen = candidates[0]
    print(f"\nChosen station: {chosen['id']} ({chosen['name']}), "
          f"{chosen['distance_km']} km away, {chosen['record_years']} yr record")

    daily = download_daily_prcp(token, chosen["id"], chosen["mindate"], chosen["maxdate"])
    daily.to_csv(DAILY_RAW_CSV, index=False)
    print(f"\nSaved raw daily data → {DAILY_RAW_CSV} ({len(daily)} records)")

    monthly = aggregate_monthly(daily)
    monthly.to_csv(MONTHLY_CSV, index=False)
    print(f"Saved monthly time series → {MONTHLY_CSV} ({len(monthly)} months)")

    summarize_seasonality(monthly)
    plot_timeseries(monthly, MONTHLY_PNG)

    with open(STATION_JSON, "w") as f:
        json.dump({"chosen": chosen, "candidates": candidates[:10]}, f, indent=2)
    print(f"Saved station metadata → {STATION_JSON}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a wet/dry monthly precipitation time series for the SR417 corridor AOI")
    parser.add_argument("--lat",       type=float, default=DEFAULT_LAT)
    parser.add_argument("--lon",       type=float, default=DEFAULT_LON)
    parser.add_argument("--radius_km", type=float, default=1.0, help="AOI radius (informational; station search uses STATION_SEARCH_RADIUS_KM)")
    args = parser.parse_args()
    main(args.lat, args.lon, args.radius_km)
