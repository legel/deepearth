"""
Target-Date Generator for PlanetScope Imagery Selection
========================================================
Reads existing downloaded precipitation data and identifies specific dates
for Lance Legel (team lead) to use as PlanetScope imagery pull targets:

  MAX  — top single-day events + multi-day wet sequences (2021-present)
  AVG  — representative weeks near the seasonal mean (2021-present)
  MIN  — driest 30-day windows + longest consecutive dry streaks (2021-present)

Also generates two plots for the project update email:
  full_timeseries_ghcnd.png    — monthly totals 1948–2026 with trend line
  timeseries_2021_present.png  — daily detail (GHCND + MCO ASOS), annotated
                                  with MAX/AVG/MIN candidate dates

Inputs (must already exist in precipitation/data/):
  daily_precip_raw.csv          GHCND daily, 1948-2026
  monthly_precip_timeseries.csv GHCND monthly with wet/dry labels
  asos_daily_MCO.csv            MCO ASOS daily, 2021-present

Outputs (written to precipitation/data/):
  full_timeseries_ghcnd.png
  timeseries_2021_present.png
  target_dates_planetscope.json
  target_dates_all_time.json

Usage:
    python3 precipitation/generate_target_dates.py
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

PLANETSCOPE_START = "2021-01-01"
# NWS official Florida wet season: May–Oct / dry: Nov–Apr
WET_MONTHS = {5, 6, 7, 8, 9, 10}

# Seasonal means under NWS definition (computed from complete months, full record)
WET_MEAN_MM_MO = 149.5   # May–Oct
DRY_MEAN_MM_MO = 59.5    # Nov–Apr

# Tolerance for "average" months (±15%)
AVG_TOLERANCE = 0.15

# Trace threshold — MCO ASOS values < this treated as 0 for dry-day counting
TRACE_MM = 0.1


# ── Data loaders ─────────────────────────────────────────────────────────────

def load_ghcnd_daily():
    path = os.path.join(DATA_DIR, "daily_precip_raw.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    df["prcp_mm"] = pd.to_numeric(df["prcp_mm"], errors="coerce").fillna(0.0)
    return df


def load_ghcnd_monthly():
    path = os.path.join(DATA_DIR, "monthly_precip_timeseries.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    df["monthly_total_mm"] = pd.to_numeric(df["monthly_total_mm"], errors="coerce")
    return df


def load_asos_mco():
    path = os.path.join(DATA_DIR, "asos_daily_MCO.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    df["prcp_mm"] = pd.to_numeric(df["prcp_mm"], errors="coerce").fillna(0.0)
    return df


# ── Target-date computation ───────────────────────────────────────────────────

def find_max_dates(ghcnd, mco, n_single=10, n_sequence=5):
    """
    MAX category: biggest single-day events and wettest 5-day windows
    since PLANETSCOPE_START, using GHCND as primary and MCO as cross-check.
    """
    g = ghcnd[ghcnd.index >= PLANETSCOPE_START].copy()
    m = mco[mco.index >= PLANETSCOPE_START].copy()

    # Align on common dates
    common = g.index.intersection(m.index)
    combined = pd.DataFrame({
        "ghcnd_mm": g.loc[common, "prcp_mm"],
        "mco_mm":   m.loc[common, "prcp_mm"],
    })
    # Primary sort by GHCND; MCO provided for context
    combined["primary_mm"] = combined["ghcnd_mm"]

    # Top single-day events
    top_single = combined.nlargest(n_single, "primary_mm")
    single_records = []
    for date, row in top_single.iterrows():
        single_records.append({
            "date": date.strftime("%Y-%m-%d"),
            "ghcnd_mm": round(float(row["ghcnd_mm"]), 1),
            "mco_mm":   round(float(row["mco_mm"]), 1),
            "note": "Hurricane Ian peak" if date.strftime("%Y-%m-%d") in ("2022-09-28", "2022-09-29") else "",
        })

    # Top 5-day wet sequences (rolling sum on GHCND)
    roll5 = g["prcp_mm"].rolling(5, min_periods=5).sum()
    top_seq_ends = roll5.nlargest(n_sequence * 3).index  # oversample, then deduplicate
    sequences = []
    used_dates = set()
    for end_date in top_seq_ends:
        start_date = end_date - pd.Timedelta(days=4)
        # Skip if overlaps an already-selected sequence
        overlap = any(
            abs((end_date - pd.Timestamp(s["end_date"])).days) < 5
            for s in sequences
        )
        if overlap:
            continue
        sequences.append({
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date":   end_date.strftime("%Y-%m-%d"),
            "5day_total_mm": round(float(roll5[end_date]), 1),
        })
        if len(sequences) >= n_sequence:
            break

    # Always include Ian explicitly if not already in single records
    ian_dates = ["2022-09-28", "2022-09-29"]
    existing = {r["date"] for r in single_records}
    ian_extras = []
    for d in ian_dates:
        if d not in existing and d in combined.index.strftime("%Y-%m-%d"):
            ts = pd.Timestamp(d)
            if ts in combined.index:
                ian_extras.append({
                    "date": d,
                    "ghcnd_mm": round(float(combined.loc[ts, "ghcnd_mm"]), 1),
                    "mco_mm":   round(float(combined.loc[ts, "mco_mm"]), 1),
                    "note": "Hurricane Ian",
                })
    single_records = ian_extras + single_records

    return {"single_day_events": single_records, "5day_sequences": sequences}


def find_avg_dates(ghcnd_monthly, ghcnd_daily, n_per_season=2):
    """
    AVG category: months within ±AVG_TOLERANCE of the seasonal mean,
    one representative 7-day window per qualifying month (2021-present).
    """
    mo = ghcnd_monthly[
        (ghcnd_monthly.index >= PLANETSCOPE_START) & ghcnd_monthly["complete"]
    ].copy()

    results = []
    for date, row in mo.iterrows():
        # Derive season from month number (NWS May–Oct / Nov–Apr), not the CSV label
        season = "wet" if date.month in WET_MONTHS else "dry"
        target = WET_MEAN_MM_MO if season == "wet" else DRY_MEAN_MM_MO
        deviation = abs(row["monthly_total_mm"] - target) / target
        if deviation > AVG_TOLERANCE:
            continue

        year, month = date.year, date.month
        # Find the 7-day window in this month whose total is closest to target/4.3
        target_week = target / 4.3
        daily_month = ghcnd_daily[
            (ghcnd_daily.index.year == year) & (ghcnd_daily.index.month == month)
        ]["prcp_mm"]
        roll7 = daily_month.rolling(7, min_periods=7).sum()
        if roll7.dropna().empty:
            continue
        best_end = (roll7 - target_week).abs().idxmin()
        best_start = best_end - pd.Timedelta(days=6)
        results.append({
            "month": date.strftime("%Y-%m"),
            "season": season,
            "monthly_total_mm": round(float(row["monthly_total_mm"]), 1),
            "deviation_pct": round(deviation * 100, 1),
            "representative_window": {
                "start": best_start.strftime("%Y-%m-%d"),
                "end":   best_end.strftime("%Y-%m-%d"),
                "7day_total_mm": round(float(roll7[best_end]), 1),
            },
        })

    # Sort by deviation (closest to mean first), keep n_per_season per season per year
    results.sort(key=lambda r: r["deviation_pct"])
    wet_by_year = {}
    dry_by_year = {}
    final = []
    for r in results:
        yr = int(r["month"][:4])
        if r["season"] == "wet":
            if wet_by_year.get(yr, 0) < n_per_season:
                final.append(r)
                wet_by_year[yr] = wet_by_year.get(yr, 0) + 1
        else:
            if dry_by_year.get(yr, 0) < n_per_season:
                final.append(r)
                dry_by_year[yr] = dry_by_year.get(yr, 0) + 1

    return final


def find_min_dates(ghcnd, n_windows=8, n_streaks=5):
    """
    MIN category: driest 30-day windows and longest consecutive dry streaks
    (< TRACE_MM/day) since PLANETSCOPE_START.
    """
    g = ghcnd[ghcnd.index >= PLANETSCOPE_START].copy()

    # Driest 30-day windows
    roll30 = g["prcp_mm"].rolling(30, min_periods=28).sum()
    windows = []
    used = []
    for end_date in roll30.nsmallest(n_windows * 4).index:
        start_date = end_date - pd.Timedelta(days=29)
        overlap = any(abs((end_date - pd.Timestamp(s["end_date"])).days) < 15 for s in windows)
        if overlap:
            continue
        windows.append({
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date":   end_date.strftime("%Y-%m-%d"),
            "30day_total_mm": round(float(roll30[end_date]), 1),
        })
        if len(windows) >= n_windows:
            break

    # Longest consecutive dry stretches (daily prcp < TRACE_MM)
    is_dry = (g["prcp_mm"] < TRACE_MM)
    streaks = []
    streak_start = None
    streak_len = 0
    for date, dry in is_dry.items():
        if dry:
            if streak_start is None:
                streak_start = date
            streak_len += 1
        else:
            if streak_start is not None and streak_len >= 5:
                streaks.append({
                    "start_date": streak_start.strftime("%Y-%m-%d"),
                    "end_date":   (date - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    "dry_days": streak_len,
                })
            streak_start = None
            streak_len = 0
    if streak_start is not None and streak_len >= 5:
        streaks.append({
            "start_date": streak_start.strftime("%Y-%m-%d"),
            "end_date":   g.index[-1].strftime("%Y-%m-%d"),
            "dry_days": streak_len,
        })
    streaks.sort(key=lambda s: -s["dry_days"])

    return {"driest_30day_windows": windows, "longest_dry_streaks": streaks[:n_streaks]}


def find_all_time_context(ghcnd_monthly):
    """
    ALL-TIME context: top wettest/driest months in the full 77.8-yr record.
    """
    mo = ghcnd_monthly[ghcnd_monthly["complete"]].copy()
    # Re-derive season from month number (NWS May–Oct / Nov–Apr)
    mo["nws_season"] = mo.index.month.map(lambda m: "wet" if m in WET_MONTHS else "dry")

    wet_mo = mo[mo["nws_season"] == "wet"]
    dry_mo = mo[mo["nws_season"] == "dry"]

    def top_months(subset, n, ascending):
        subset = subset.nsmallest(n, "monthly_total_mm") if ascending else subset.nlargest(n, "monthly_total_mm")
        return [
            {"month": d.strftime("%Y-%m"), "total_mm": round(float(r["monthly_total_mm"]), 1),
             "season": r["nws_season"]}
            for d, r in subset.iterrows()
        ]

    return {
        "top10_wettest_wet_season_months": top_months(wet_mo, 10, ascending=False),
        "top10_driest_wet_season_months":  top_months(wet_mo, 10, ascending=True),
        "top10_wettest_dry_season_months": top_months(dry_mo, 10, ascending=False),
        "top10_driest_dry_season_months":  top_months(dry_mo, 10, ascending=True),
    }


# ── Plot 1: Full time series 1948-2026 ───────────────────────────────────────

def _apply_clean_style(ax, fig):
    """Pure white background, no grid, minimal spines."""
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#bbbbbb")
    ax.spines["bottom"].set_color("#bbbbbb")
    ax.tick_params(colors="#444444")


def plot_full_timeseries(ghcnd_monthly):
    mo = ghcnd_monthly.copy()

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(18, 5))
    _apply_clean_style(ax, fig)

    WET_COLOR = "#2166ac"
    DRY_COLOR = "#d6604d"
    bar_colors = [WET_COLOR if s == "wet" else DRY_COLOR for s in mo["season"]]
    ax.bar(mo.index, mo["monthly_total_mm"], width=20, color=bar_colors, alpha=0.75,
           linewidth=0, zorder=2)

    # 12-month rolling mean
    roll12 = mo["monthly_total_mm"].rolling(12, center=True, min_periods=6).mean()
    ax.plot(mo.index, roll12, color="#333333", lw=1.6, alpha=0.9,
            label="12-mo rolling mean", zorder=3)

    # Trend line fit to the 12-month rolling mean
    roll12_clean = roll12.dropna()
    roll12_clean = roll12_clean[roll12_clean.index >= pd.Timestamp("1959-01-01")]
    x_num = np.array([(d - pd.Timestamp("1959-01-01")).days for d in roll12_clean.index], dtype=float)
    y_num = roll12_clean.values
    slope_day, intercept, r, p, _ = stats.linregress(x_num, y_num)
    slope_decade = slope_day * 3650
    trend_y = slope_day * x_num + intercept
    ax.plot(roll12_clean.index, trend_y, color="#e08020", lw=1.8, ls="--", alpha=0.9,
            label=f"Trend ({slope_decade:+.1f} mm/decade, p = {p:.2f})", zorder=4)

    # Annotate 1952-1958 data gap
    ax.axvspan(pd.Timestamp("1952-01-01"), pd.Timestamp("1958-12-31"),
               color="#cccccc", alpha=0.50, zorder=1, label="_nolegend_")
    ymax = mo["monthly_total_mm"].max()
    ax.text(pd.Timestamp("1955-07-01"), ymax * 0.96,
            "Data gap\n1952–1958", color="#666666", ha="center", va="top",
            fontsize=8, style="italic",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#bbbbbb", alpha=0.9))

    wet_patch = mpatches.Patch(color=WET_COLOR, alpha=0.75, label="Wet season (May–Oct, NWS)")
    dry_patch = mpatches.Patch(color=DRY_COLOR, alpha=0.75, label="Dry season (Nov–Apr, NWS)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=[wet_patch, dry_patch] + handles,
              labels=["Wet season (Jun–Sep)", "Dry season (Oct–May)"] + labels,
              fontsize=8.5, loc="upper right", framealpha=0.95,
              edgecolor="#cccccc")

    ax.set_xlabel("Year", fontsize=10, color="#444444")
    ax.set_ylabel("Monthly precipitation (mm)", fontsize=10, color="#444444")
    ax.set_title(
        "Monthly Precipitation — Kissimmee 2, FL (GHCND USC00084625, 1948–2026)\n"
        "SR417 Corridor AOI · Station 10.1 km from 28.37°N, 81.43°W",
        fontsize=11)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=9)

    fig.tight_layout()
    out = os.path.join(DATA_DIR, "full_timeseries_ghcnd.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out}")


# ── Plot 2: 2021-present daily detail with annotated target dates ─────────────

def plot_2021_present(ghcnd_daily, mco_daily=None, max_dates=None, avg_dates=None, min_dates=None):
    """
    Clean daily bar chart of GHCND for the PlanetScope window (2021-present).
    Y-axis is clipped at Y_CLIP to keep typical events readable; any bars that
    exceed the clip are drawn to the clip height and annotated with their true value.
    """
    g = ghcnd_daily[ghcnd_daily.index >= PLANETSCOPE_START].copy()

    from datetime import datetime as dt

    Y_CLIP = 85.0   # axis ceiling; bars above this are clipped and labeled

    # Identify days that exceed the clip (need explicit annotation)
    outliers = g[g["prcp_mm"] > Y_CLIP].copy()
    # Cap bar heights at Y_CLIP for plotting
    g_plot = g["prcp_mm"].clip(upper=Y_CLIP)

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(16, 6))
    _apply_clean_style(ax, fig)

    WET_COLOR = "#2166ac"
    DRY_COLOR = "#d6604d"

    bar_colors = [WET_COLOR if d.month in WET_MONTHS else DRY_COLOR for d in g.index]
    ax.bar(g.index, g_plot, width=1.2, color=bar_colors, alpha=0.80,
           linewidth=0, zorder=2)

    # Mark clipped bars: triangle marker + staggered labels so they never overlap.
    # Compute stagger: sort outliers by date; if two are within 90 days, alternate rows.
    Y_TOP = Y_CLIP * 1.22   # axis ceiling
    label_rows = [Y_CLIP * 1.03, Y_CLIP * 1.13]   # two vertical levels
    prev_date = None
    row_idx = 0
    for date, row in outliers.sort_index().iterrows():
        true_val = row["prcp_mm"]
        bar_color = WET_COLOR if date.month in WET_MONTHS else DRY_COLOR
        ax.plot(date, Y_CLIP, marker="^", color=bar_color, ms=7, zorder=5,
                clip_on=False)
        if prev_date is not None and (date - prev_date).days < 90:
            row_idx = 1 - row_idx   # flip to the other row
        else:
            row_idx = 0
        ax.text(date, label_rows[row_idx], f"{true_val:.0f} mm",
                ha="center", va="bottom", fontsize=8, color=bar_color,
                fontweight="bold", zorder=6)
        prev_date = date

    # Wet season shading (NWS May–Oct)
    for yr in range(2021, dt.now().year + 1):
        ax.axvspan(pd.Timestamp(f"{yr}-05-01"), pd.Timestamp(f"{yr}-10-31"),
                   alpha=0.07, color=WET_COLOR, label="_nolegend_", zorder=1)

    wet_patch  = mpatches.Patch(color=WET_COLOR, alpha=0.80, label="Wet season (May–Oct, NWS)")
    dry_patch  = mpatches.Patch(color=DRY_COLOR, alpha=0.80, label="Dry season (Nov–Apr, NWS)")
    shade_patch = mpatches.Patch(color=WET_COLOR, alpha=0.15, label="May–Oct window")
    if not outliers.empty:
        clip_marker = plt.Line2D([0], [0], marker="^", color="#555555", lw=0,
                                 ms=7, label="Bar clipped — true value labeled above")
        legend_handles = [wet_patch, dry_patch, shade_patch, clip_marker]
    else:
        legend_handles = [wet_patch, dry_patch, shade_patch]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper right",
              framealpha=0.95, edgecolor="#cccccc")

    ax.set_xlabel("Date", fontsize=10, color="#444444")
    ax.set_ylabel("Daily precipitation (mm)", fontsize=10, color="#444444")
    ax.set_title(
        "Daily Precipitation (2021–present) — SR417 Corridor AOI\n"
        "GHCND USC00084625 Kissimmee 2, FL  ·  Station 10.1 km from AOI center",
        fontsize=11)
    ax.set_ylim(0, Y_TOP)
    ax.set_yticks([20, 40, 60, 80])
    ax.tick_params(labelsize=9)

    fig.tight_layout()
    out = os.path.join(DATA_DIR, "timeseries_2021_present.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate PlanetScope target dates")
    parser.add_argument("--lat", type=float, default=28.36687)
    parser.add_argument("--lon", type=float, default=-81.43299)
    args = parser.parse_args()

    print("Loading precipitation data ...")
    ghcnd_daily   = load_ghcnd_daily()
    ghcnd_monthly = load_ghcnd_monthly()
    mco_daily     = load_asos_mco()

    print("Computing MAX / AVG / MIN target dates (2021-present) ...")
    max_dates = find_max_dates(ghcnd_daily, mco_daily)
    avg_dates = find_avg_dates(ghcnd_monthly, ghcnd_daily)
    min_dates = find_min_dates(ghcnd_daily)

    # PlanetScope target dates JSON
    planetscope_out = {
        "aoi": {"lat": args.lat, "lon": args.lon},
        "planetscope_window": f"{PLANETSCOPE_START} to present",
        "ghcnd_station": "USC00084625 (Kissimmee 2, FL, 10.1 km)",
        "asos_station":  "MCO (Orlando Intl, 14.0 km)",
        "categories": {
            "MAX": max_dates,
            "AVG": avg_dates,
            "MIN": min_dates,
        },
    }
    ps_path = os.path.join(DATA_DIR, "target_dates_planetscope.json")
    with open(ps_path, "w") as f:
        json.dump(planetscope_out, f, indent=2, default=str)
    print(f"Saved → {ps_path}")

    # All-time context JSON
    all_time_out = {
        "aoi": {"lat": args.lat, "lon": args.lon},
        "record": "1948-09-02 to 2026-06-22 (GHCND USC00084625)",
        "note": "Real data gap 1952-1958; complete months only used in extremes",
        **find_all_time_context(ghcnd_monthly),
    }
    at_path = os.path.join(DATA_DIR, "target_dates_all_time.json")
    with open(at_path, "w") as f:
        json.dump(all_time_out, f, indent=2, default=str)
    print(f"Saved → {at_path}")

    print("Generating plots ...")
    plot_full_timeseries(ghcnd_monthly)
    plot_2021_present(ghcnd_daily, mco_daily, max_dates, avg_dates, min_dates)

    # Print summary table for Lance's email
    print("\n" + "=" * 60)
    print("SUMMARY: Target dates for PlanetScope imagery")
    print("=" * 60)
    print(f"\nMAX — Top single-day events (2021–present):")
    for r in max_dates["single_day_events"][:5]:
        note = f"  ← {r['note']}" if r.get("note") else ""
        print(f"  {r['date']}  GHCND {r['ghcnd_mm']:>6.1f} mm  MCO {r['mco_mm']:>6.1f} mm{note}")

    print(f"\nMAX — Top 5-day wet sequences (2021–present):")
    for r in max_dates["5day_sequences"]:
        print(f"  {r['start_date']} – {r['end_date']}  5-day total: {r['5day_total_mm']:.1f} mm")

    print(f"\nAVG — Representative weeks (closest to seasonal mean):")
    for r in avg_dates[:6]:
        w = r["representative_window"]
        print(f"  {w['start']} – {w['end']}  ({r['season']}, {r['monthly_total_mm']:.0f} mm/mo, "
              f"{r['deviation_pct']:.0f}% from mean)")

    print(f"\nMIN — Longest consecutive dry streaks (2021–present):")
    for r in min_dates["longest_dry_streaks"][:3]:
        print(f"  {r['start_date']} – {r['end_date']}  ({r['dry_days']} dry days)")

    print(f"\nMIN — Driest 30-day windows (2021–present):")
    for r in min_dates["driest_30day_windows"][:3]:
        print(f"  {r['start_date']} – {r['end_date']}  30-day total: {r['30day_total_mm']:.1f} mm")
    print()


if __name__ == "__main__":
    main()
