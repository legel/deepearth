"""
ASOS Hourly Precipitation — IEM Iowa State via Free API
=========================================================
Fetches 5-minute ASOS observations (IEM p01i field, inches) for the two
nearest ASOS stations to the SR417 corridor AOI, extracts true hourly totals
from the standard METAR :53 observation cycle, converts to mm, and compares
against our existing NOAA GHCND daily record.

Stations (by distance to AOI center 28.36687N, -81.43299W):
  ISM  Kissimmee Muni Airport/Orlando   8.6 km  — closest ASOS
  MCO  Orlando International Airport   14.0 km  — larger/better-instrumented

IEM p01i format (critical for correct aggregation):
  - Observations are 5-minute ASOS/METAR METARs; p01i is the running
    accumulation for the CURRENT hour (inches), not an instantaneous rate.
  - It resets to a small value (~0.00–0.02 in) at the top of each new hour.
  - The :53 METAR observation is the AUTHORITATIVE HOURLY TOTAL for that
    UTC hour — the last accumulated value before the reset.
  - Never sum all rows; instead take max(p01i) per hour (= the :53 value
    before the reset, robust to duplicate observations) then convert to mm.

IEM API: https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py
No authentication required.

Outputs (saved under precipitation/data/):
  asos_hourly_{station}.csv     — hourly totals in mm (ISM, MCO)
  asos_daily_{station}.csv      — daily totals aggregated from hourly
  precip_comparison_ian.png     — Hurricane Ian (Sep 27–Oct 5, 2022):
                                  GHCND daily vs ASOS daily (bar) +
                                  MCO/ISM hourly intensity (line)
  precip_comparison_monthly.png — monthly totals: GHCND vs ASOS (2021-present)

Usage:
    python3 precipitation/fetch_asos_hourly.py
    python3 precipitation/fetch_asos_hourly.py --start 2021-01-01 --end 2026-06-22
"""

import os
import io
import sys
import time
import argparse
from datetime import datetime
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

STATIONS = {
    "ISM": {"name": "Kissimmee Muni (ISM)", "dist_km": 8.6,  "color": "#e07020"},
    "MCO": {"name": "Orlando Intl (MCO)",   "dist_km": 14.0, "color": "#2060c0"},
}
IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

GHCND_CSV = os.path.join(DATA_DIR, "daily_precip_raw.csv")
INCHES_TO_MM = 25.4


def fetch_iem_asos(station, start_date, end_date, max_retries=4):
    """Fetch 5-minute ASOS p01i (inches) from IEM for a date range."""
    params = dict(
        station=station, data="p01i",
        year1=start_date[:4], month1=start_date[5:7], day1=start_date[8:10],
        year2=end_date[:4],   month2=end_date[5:7],   day2=end_date[8:10],
        tz="UTC", format="comma", latlon="no", trace="T", direct="no",
    )
    for attempt in range(max_retries):
        try:
            r = requests.get(IEM_URL, params=params, timeout=60)
            r.raise_for_status()
            lines = [l for l in r.text.splitlines()
                     if l.strip() and not l.startswith("#")]
            if len(lines) < 2:
                raise RuntimeError(f"No data returned for {station}: {r.text[:200]}")
            df = pd.read_csv(io.StringIO("\n".join(lines)), parse_dates=["valid"])
            return df
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"    ⚠ IEM fetch error ({exc.__class__.__name__}), retry in {wait}s …")
            time.sleep(wait)


def extract_hourly(raw_df):
    """
    Extract true hourly totals from raw 5-min ASOS observations.

    IEM p01i is a running-accumulator for the CURRENT hour, reset each UTC
    hour. The :53 METAR observation is authoritative. Taking max(p01i) per
    UTC hour (which equals the :53 value for normal hours) is robust to
    duplicate rows and missing :53 obs.

    Returns a Series indexed by UTC timestamp (top-of-hour), values in mm.
    """
    df = raw_df.copy()
    # Convert trace ("T") → 0.001 in; coerce other non-numerics to NaN → 0
    df["p01i"] = pd.to_numeric(
        df["p01i"].replace("T", "0.001"), errors="coerce"
    ).fillna(0.0)
    df = df.set_index("valid").sort_index()

    # Max p01i per UTC hour = the accumulated value just before the reset
    hourly_in = df["p01i"].resample("h").max().fillna(0.0)
    hourly_mm = hourly_in * INCHES_TO_MM
    hourly_mm.name = "prcp_mm"
    return hourly_mm


def load_ghcnd():
    """Load existing NOAA GHCND daily record (Kissimmee 2, station_metadata.json)."""
    if not os.path.exists(GHCND_CSV):
        print(f"  ⚠ GHCND CSV not found at {GHCND_CSV}; run fetch_precip_seasonality.py first")
        return None
    df = pd.read_csv(GHCND_CSV, parse_dates=["date"])
    df = df.rename(columns={"prcp_mm": "mm"})
    df["mm"] = pd.to_numeric(df["mm"], errors="coerce").fillna(0.0)
    return df.set_index("date")["mm"]


def main(start="2021-01-01", end="2026-06-22"):
    ghcnd = load_ghcnd()
    asos_hourly = {}
    asos_daily = {}

    for stn, info in STATIONS.items():
        print(f"\nFetching ASOS from IEM: {stn} ({info['name']}, {info['dist_km']} km) …")
        try:
            raw = fetch_iem_asos(stn, start, end)
        except Exception as exc:
            print(f"  ✗ Failed: {exc}")
            continue
        print(f"  {len(raw):,} raw 5-min obs downloaded")

        hourly = extract_hourly(raw)
        asos_hourly[stn] = hourly

        # Daily totals (UTC day)
        daily = hourly.resample("D").sum()
        daily.index = daily.index.normalize()
        asos_daily[stn] = daily

        hourly_path = os.path.join(DATA_DIR, f"asos_hourly_{stn}.csv")
        daily_path  = os.path.join(DATA_DIR, f"asos_daily_{stn}.csv")
        hourly.reset_index().rename(columns={"valid": "datetime_utc"}).to_csv(hourly_path, index=False)
        daily.reset_index().rename(columns={"valid": "date"}).to_csv(daily_path, index=False)
        print(f"  Saved: {os.path.basename(hourly_path)}, {os.path.basename(daily_path)}")

    if not asos_daily:
        sys.exit("No ASOS data retrieved.")

    _plot_ian_comparison(ghcnd, asos_daily, asos_hourly)
    _plot_monthly_comparison(ghcnd, asos_daily)
    _print_comparison_summary(ghcnd, asos_daily)


def _plot_ian_comparison(ghcnd, asos_daily, asos_hourly):
    """Hurricane Ian window: GHCND daily vs ASOS daily + hourly intensity."""
    ian_start, ian_end = "2022-09-26", "2022-10-06"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    # — Top panel: daily comparison —
    ax1.set_title("Hurricane Ian · Gauge Comparison · Sep 26 – Oct 5, 2022", fontsize=13)

    dates = pd.date_range(ian_start, ian_end, freq="D")
    bar_w = 0.25
    x = np.arange(len(dates))

    if ghcnd is not None:
        ghcnd_ian = ghcnd.reindex(dates, fill_value=0.0)
        ax1.bar(x - bar_w, ghcnd_ian.values, width=bar_w, label="GHCND daily (Kissimmee 2, CoOp, 10km)",
                color="#3a6060", alpha=0.85)

    colors_bar = ["#e07020", "#2060c0"]
    for i, (stn, daily) in enumerate(asos_daily.items()):
        daily_ian = daily.reindex(dates, fill_value=0.0)
        info = STATIONS[stn]
        ax1.bar(x + i * bar_w, daily_ian.values, width=bar_w,
                label=f"ASOS daily ({info['name']}, {info['dist_km']} km)",
                color=colors_bar[i], alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels([d.strftime("%m/%d") for d in dates], fontsize=9)
    ax1.set_ylabel("Daily precipitation (mm)")
    ax1.legend(fontsize=9)
    ax1.set_xlim(-0.5, len(dates) - 0.5)

    # — Bottom panel: MCO/ISM hourly intensity during Ian —
    ax2.set_title("Hourly Intensity at ASOS Stations · Sep 28–30, 2022 UTC", fontsize=13)
    ax2.axvline(pd.Timestamp("2022-09-28 12:00"), color="gray", ls="--", lw=0.8, label="Ian landfall ~12Z Sep28")

    hourly_start = pd.Timestamp("2022-09-28 00:00")
    hourly_end   = pd.Timestamp("2022-09-30 06:00")

    for stn, hourly in asos_hourly.items():
        info = STATIONS[stn]
        window = hourly.loc[hourly_start:hourly_end]
        ax2.plot(window.index, window.values, label=f"{stn} ({info['dist_km']} km)", color=info["color"], lw=1.8)

    ax2.set_ylabel("Precipitation (mm/hr)")
    ax2.set_xlabel("Date/time (UTC)")
    ax2.legend(fontsize=9)

    # Annotate peak hour
    for stn, hourly in asos_hourly.items():
        window = hourly.loc[hourly_start:hourly_end]
        if len(window) == 0:
            continue
        peak_idx = window.idxmax()
        peak_val = window.max()
        ax2.annotate(f"{stn} peak\n{peak_val:.1f} mm/hr",
                     xy=(peak_idx, peak_val),
                     xytext=(10, 8), textcoords="offset points",
                     fontsize=8, color=STATIONS[stn]["color"],
                     arrowprops=dict(arrowstyle="->", color=STATIONS[stn]["color"], lw=0.8))

    fig.tight_layout()
    out = os.path.join(DATA_DIR, "precip_comparison_ian.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nSaved Ian comparison plot → {out}")


def _plot_monthly_comparison(ghcnd, asos_daily):
    """Monthly totals comparison: GHCND vs ASOS (2021-present)."""
    start_month = pd.Timestamp("2021-01-01")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_title("Monthly Precipitation Comparison · GHCND (Kissimmee 2) vs ASOS (ISM, MCO) · 2021–present")

    if ghcnd is not None:
        monthly_ghcnd = ghcnd[ghcnd.index >= start_month].resample("ME").sum()
        months = monthly_ghcnd.index
        ax.plot(months, monthly_ghcnd.values, "o-", lw=2, ms=4, color="#3a6060",
                label="GHCND daily (Kissimmee 2 CoOp, 10km)", zorder=3)

    for stn, daily in asos_daily.items():
        info = STATIONS[stn]
        monthly = daily[daily.index >= start_month].resample("ME").sum()
        ax.plot(monthly.index, monthly.values, "s--", lw=1.5, ms=3,
                color=info["color"], label=f"ASOS ({info['name']}, {info['dist_km']} km)", alpha=0.85)

    ax.set_ylabel("Monthly total (mm)")
    ax.set_xlabel("Month")
    ax.legend(fontsize=9)
    ax.axhline(0, color="#aaa", lw=0.5)

    # Shade wet seasons (Jun-Sep)
    for year in range(2021, datetime.now().year + 1):
        ax.axvspan(pd.Timestamp(f"{year}-06-01"), pd.Timestamp(f"{year}-09-30"),
                   alpha=0.07, color="blue", label="_nolegend_")

    fig.tight_layout()
    out = os.path.join(DATA_DIR, "precip_comparison_monthly.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved monthly comparison plot → {out}")


def _print_comparison_summary(ghcnd, asos_daily):
    """Print a side-by-side daily comparison for the Ian event and a correlation check."""
    ian_dates = pd.date_range("2022-09-26", "2022-10-05", freq="D")
    print("\n── Daily totals comparison (Hurricane Ian window) ─────────────────")
    header = f"{'Date':12}"
    if ghcnd is not None:
        header += f"  {'GHCND(mm)':>10}"
    for stn in asos_daily:
        header += f"  {stn+' ASOS(mm)':>12}"
    print(header)
    for d in ian_dates:
        row = f"{str(d.date()):12}"
        if ghcnd is not None:
            row += f"  {ghcnd.get(d, 0.0):>10.1f}"
        for stn, daily in asos_daily.items():
            row += f"  {daily.get(d, 0.0):>12.1f}"
        print(row)

    # Full-record daily correlation
    if ghcnd is None:
        return
    print("\n── Correlation (all 2021-present days) ─────────────────────────────")
    for stn, daily in asos_daily.items():
        shared = pd.concat([ghcnd.rename("ghcnd"), daily.rename("asos")], axis=1).dropna()
        shared = shared[shared.index >= pd.Timestamp("2021-01-01")]
        if len(shared) < 10:
            print(f"  {stn}: insufficient overlap")
            continue
        corr = shared["ghcnd"].corr(shared["asos"])
        bias = (shared["asos"] - shared["ghcnd"]).mean()
        rmse = float(np.sqrt(((shared["asos"] - shared["ghcnd"])**2).mean()))
        print(f"  {stn} vs GHCND:  r={corr:.3f}  bias={bias:+.1f}mm/day  "
              f"RMSE={rmse:.1f}mm  n={len(shared)} days")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch & compare ASOS hourly vs GHCND daily for SR417 AOI")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end",   default="2026-06-22")
    args = parser.parse_args()
    main(args.start, args.end)
