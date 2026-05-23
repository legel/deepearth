"""
GSDR US — Visualization Suite
==============================
Produces 6 figures saved to gsdr/outputs/:

  fig01_us_stations.html      — Interactive US station map (colored by record length)
  fig02_record_length.png     — Distribution of record length (years) per station
  fig03_temporal_coverage.png — Active stations per year across full record
  fig04_missing_data.png      — Missing data rate distribution across stations
  fig05_intensity_scatter.png — 1-hr vs 24-hr all-time max scatter (flash vs riverine)
  fig06_peak_event_year.png   — Year each station recorded its all-time 1-hr maximum

Prerequisite: run gsdr_build_index.py first to generate gsdr_us_index.csv.
The index provides station metadata (lat/lon, dates, missing %).
For Figs 5 and 6 the script loads each gauge's full time series to compute
per-station maxima — this takes 10–20 minutes for all 6,605 stations.

Environment variable:
    GSDR_QC_DIR : path to the QC_d data - US folder
                  default: ~/Desktop/GSDR/QC_d data - US

Usage:
    python3 gsdr/gsdr_visualizations.py

Dependencies: pandas, numpy, matplotlib, seaborn, plotly
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "gsdr_us_index.csv")
QC_DIR     = os.environ.get("GSDR_QC_DIR", os.path.expanduser("~/Desktop/GSDR/QC_d data - US"))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADER_LINES = 21
MISSING_VAL  = -999.0


# ── Load index ─────────────────────────────────────────────────────────────
print("Loading station index …")
index = pd.read_csv(INDEX_PATH)
index["start_year"] = index["START"].astype(str).str[:4].astype(int, errors="ignore")
index["end_year"]   = index["END"].astype(str).str[:4].astype(int, errors="ignore")
index["record_len"] = index["end_year"] - index["start_year"]
index = index[index["record_len"] > 0]
print(f"  {len(index):,} stations loaded.\n")


# ── Helper: parse INTENSE start datetime ──────────────────────────────────
def parse_start_dt(start_str):
    s = str(start_str).strip()
    try:
        return datetime(int(s[0:4]), int(s[4:6]), int(s[6:8]), int(s[8:10]))
    except Exception:
        return None


# ── Helper: load gauge and compute per-station maxima ─────────────────────
def station_maxima(station_id, durations=(1, 24)):
    fpath = os.path.join(QC_DIR, f"{station_id}.txt")
    if not os.path.exists(fpath):
        return None
    try:
        header = {}
        lines  = []
        with open(fpath) as f:
            for i, line in enumerate(f):
                if i < HEADER_LINES:
                    if ":" in line:
                        k, _, v = line.partition(":")
                        header[k.strip()] = v.strip()
                else:
                    lines.append(line.strip())
        data = np.array(lines, dtype=np.float32)
        data[data == MISSING_VAL] = np.nan
        start_dt = parse_start_dt(header.get("Start datetime", ""))
        s = pd.Series(data)
        result = {}
        for d in durations:
            rolled = s.rolling(window=d, min_periods=d).sum()
            idx = rolled.idxmax()
            if pd.isna(idx) or pd.isna(rolled[idx]):
                result[d] = (np.nan, np.nan)
            else:
                year = (start_dt + timedelta(hours=int(idx))).year if start_dt else np.nan
                result[d] = (float(rolled[idx]), year)
        return result
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Fig 1 — US station map
# ═══════════════════════════════════════════════════════════════════════════
print("Fig 1 — US station map …")

fig1 = px.scatter_geo(
    index,
    lat="LAT",
    lon="LON",
    color="record_len",
    color_continuous_scale="Viridis",
    range_color=[index["record_len"].quantile(0.05),
                 index["record_len"].quantile(0.95)],
    hover_name="ID",
    hover_data={
        "LAT": ":.3f",
        "LON": ":.3f",
        "record_len": True,
        "start_year": True,
        "end_year": True,
        "PCT_MISSING": ":.1f",
    },
    labels={"record_len": "Record length (yr)", "PCT_MISSING": "Missing (%)"},
    title=f"GSDR US Station Network — {len(index):,} hourly gauges",
    scope="usa",
    opacity=0.65,
)
fig1.update_traces(marker=dict(size=4))
fig1.update_layout(
    coloraxis_colorbar=dict(title="Record length (yr)"),
    geo=dict(showland=True, landcolor="lightgray",
             showlakes=True, lakecolor="lightblue",
             showcoastlines=True, coastlinecolor="gray"),
    margin=dict(l=0, r=0, t=40, b=0),
)
out1 = os.path.join(OUTPUT_DIR, "fig01_us_stations.html")
fig1.write_html(out1)
print(f"  Saved → {out1}")

# ═══════════════════════════════════════════════════════════════════════════
# Fig 2 — Record length distribution
# ═══════════════════════════════════════════════════════════════════════════
print("Fig 2 — Record length distribution …")

fig2, ax = plt.subplots(figsize=(11, 5))
med = index["record_len"].median()
ax.hist(index["record_len"].clip(0, 80), bins=40,
        color="#4C72B0", edgecolor="white", linewidth=0.4)
ax.axvline(med, color="orange", linewidth=1.8, linestyle="--",
           label=f"Median = {med:.0f} yr")
ax.axvline(1, color="red", linewidth=1.2, linestyle=":",
           label="Min threshold (1 yr)")
ax.set_xlabel("Record length (years, clipped at 80)", fontsize=11)
ax.set_ylabel("Number of stations", fontsize=11)
ax.set_title("GSDR US — Record Length Distribution (6,605 hourly gauges)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
note = ("Note: GSDR records are shorter on average than HYADES (~13 yr vs ~47 yr)\n"
        "because sub-daily archiving began later at most stations.")
fig2.text(0.5, -0.04, note, ha="center", fontsize=9, color="gray")
plt.tight_layout()
out2 = os.path.join(OUTPUT_DIR, "fig02_record_length.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {out2}")

# ═══════════════════════════════════════════════════════════════════════════
# Fig 3 — Temporal coverage (active stations per year)
# ═══════════════════════════════════════════════════════════════════════════
print("Fig 3 — Temporal coverage …")

all_years = range(
    int(index["start_year"].min()),
    int(index["end_year"].max()) + 1
)
active = {yr: ((index["start_year"] <= yr) & (index["end_year"] >= yr)).sum()
          for yr in all_years}
active_df = pd.DataFrame(list(active.items()), columns=["YEAR", "n_stations"])
peak_row  = active_df.loc[active_df["n_stations"].idxmax()]

fig3, ax = plt.subplots(figsize=(13, 5))
ax.fill_between(active_df["YEAR"], active_df["n_stations"],
                alpha=0.25, color="#C44E52")
ax.plot(active_df["YEAR"], active_df["n_stations"],
        color="#C44E52", linewidth=1.2)
ax.axvline(2014, color="navy", linewidth=1.0, linestyle="--",
           label="2014 — dataset end (most stations)")
ax.annotate(
    f"Peak: {int(peak_row['n_stations']):,} stations\n({int(peak_row['YEAR'])})",
    xy=(peak_row["YEAR"], peak_row["n_stations"]),
    xytext=(peak_row["YEAR"] - 25, peak_row["n_stations"] * 0.88),
    arrowprops=dict(arrowstyle="->", color="black"),
    fontsize=9,
)
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Active stations", fontsize=11)
ax.set_title("GSDR US — Active Hourly Gauge Count Per Year",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
out3 = os.path.join(OUTPUT_DIR, "fig03_temporal_coverage.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {out3}")

# ═══════════════════════════════════════════════════════════════════════════
# Fig 4 — Missing data rate distribution
# ═══════════════════════════════════════════════════════════════════════════
print("Fig 4 — Missing data distribution …")

fig4, ax = plt.subplots(figsize=(11, 5))
med_miss = index["PCT_MISSING"].median()
ax.hist(index["PCT_MISSING"].clip(0, 100), bins=40,
        color="#DD8452", edgecolor="white", linewidth=0.4)
ax.axvline(med_miss, color="navy", linewidth=1.8, linestyle="--",
           label=f"Median = {med_miss:.1f}%")
ax.axvline(50, color="red", linewidth=1.2, linestyle=":",
           label="50% threshold (query filter)")
n_excluded = (index["PCT_MISSING"] > 50).sum()
ax.set_xlabel("Missing data (%)", fontsize=11)
ax.set_ylabel("Number of stations", fontsize=11)
ax.set_title("GSDR US — Missing Data Rate Per Station",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
fig4.text(0.5, -0.03,
          f"{n_excluded:,} stations ({100*n_excluded/len(index):.1f}%) excluded by default "
          f"50% missing filter in gsdr_intensity_matrix.py",
          ha="center", fontsize=9, color="gray")
plt.tight_layout()
out4 = os.path.join(OUTPUT_DIR, "fig04_missing_data.png")
plt.savefig(out4, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {out4}")

# ═══════════════════════════════════════════════════════════════════════════
# Figs 5 & 6 — require loading all gauge time series (slow)
# ═══════════════════════════════════════════════════════════════════════════
print("\nComputing per-station 1-hr and 24-hr maxima (this takes 10–20 min) …")
print("Progress printed every 500 stations.\n")

max1hr, max24hr, peak1hr_year = [], [], []
valid_ids = index["ID"].tolist()

for i, sid in enumerate(valid_ids):
    res = station_maxima(sid, durations=(1, 24))
    if res is None:
        max1hr.append(np.nan)
        max24hr.append(np.nan)
        peak1hr_year.append(np.nan)
    else:
        max1hr.append(res[1][0])
        max24hr.append(res[24][0])
        peak1hr_year.append(res[1][1])
    if (i + 1) % 500 == 0:
        print(f"  {i+1:,} / {len(valid_ids):,} stations processed …")

index["max_1hr"]      = max1hr
index["max_24hr"]     = max24hr
index["peak_1hr_year"] = peak1hr_year

print(f"\n  Done. Valid 1-hr maxima: {(~pd.isna(index['max_1hr'])).sum():,}")
print(f"         Valid 24-hr maxima: {(~pd.isna(index['max_24hr'])).sum():,}\n")

# ═══════════════════════════════════════════════════════════════════════════
# Fig 5 — 1-hr vs 24-hr max scatter (flash vs riverine signal)
# ═══════════════════════════════════════════════════════════════════════════
print("Fig 5 — 1-hr vs 24-hr max scatter …")

plot_df = index.dropna(subset=["max_1hr", "max_24hr"]).copy()
plot_df["ratio"] = plot_df["max_1hr"] / plot_df["max_24hr"]

fig5, ax = plt.subplots(figsize=(10, 8))
sc = ax.scatter(
    plot_df["max_1hr"],
    plot_df["max_24hr"],
    c=plot_df["ratio"],
    cmap="RdYlBu_r",
    alpha=0.35,
    s=8,
    vmin=0.1,
    vmax=0.8,
)
cb = plt.colorbar(sc, ax=ax)
cb.set_label("1-hr / 24-hr ratio\n(high = flash-dominated, low = slow-rain dominated)",
             fontsize=9)

# Reference lines
x = np.linspace(0, plot_df["max_1hr"].max(), 100)
for frac, label in [(0.5, "50% in 1 hr"), (0.25, "25% in 1 hr"), (0.1, "10% in 1 hr")]:
    ax.plot(x, x / frac, linestyle="--", linewidth=0.8, alpha=0.5, label=label)

ax.set_xlabel("All-time max 1-hr accumulation (mm)", fontsize=11)
ax.set_ylabel("All-time max 24-hr accumulation (mm)", fontsize=11)
ax.set_title("GSDR US — 1-hr vs 24-hr All-Time Maximum\n"
             "High ratio = convective flash flood regime  |  Low ratio = atmospheric river / slow-rain regime",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.2)
ax.set_xlim(0, plot_df["max_1hr"].quantile(0.995))
ax.set_ylim(0, plot_df["max_24hr"].quantile(0.995))
plt.tight_layout()
out5 = os.path.join(OUTPUT_DIR, "fig05_intensity_scatter.png")
plt.savefig(out5, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {out5}")

# ═══════════════════════════════════════════════════════════════════════════
# Fig 6 — Year of all-time 1-hr maximum per station
# ═══════════════════════════════════════════════════════════════════════════
print("Fig 6 — Year of all-time 1-hr peak event …")

peak_years = index["peak_1hr_year"].dropna()
peak_years = peak_years[(peak_years >= 1900) & (peak_years <= 2020)]

fig6, ax = plt.subplots(figsize=(13, 5))
bins = range(1900, 2016, 5)
ax.hist(peak_years, bins=bins, color="#C44E52", edgecolor="white", linewidth=0.4)

ax.axvspan(1950, 1960, alpha=0.12, color="steelblue", label="1950s (active hurricane decade)")
ax.axvspan(2000, 2010, alpha=0.12, color="orange",    label="2000s (active hurricane decade)")
med_yr = peak_years.median()
ax.axvline(med_yr, color="navy", linewidth=1.6, linestyle="--",
           label=f"Median = {int(med_yr)}")
ax.axvline(2014, color="gray", linewidth=1.0, linestyle=":",
           label="2014 — dataset end")

ax.set_xlabel("Year of all-time maximum 1-hr accumulation", fontsize=11)
ax.set_ylabel("Number of stations", fontsize=11)
ax.set_title("GSDR US — When Did Each Station Record Its Worst 1-Hour Rainfall?",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
out6 = os.path.join(OUTPUT_DIR, "fig06_peak_event_year.png")
plt.savefig(out6, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {out6}")

# ── Summary ────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("All GSDR visualizations complete. Output files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.startswith("fig"):
        path = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(path)
        print(f"  {f:<42}  {size/1024:.0f} KB")
