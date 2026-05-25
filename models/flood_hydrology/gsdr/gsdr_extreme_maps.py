"""
GSDR US — Extreme Rainfall Grid Maps
======================================
Generates 5 interactive HTML maps and 5 static PNG exports,
one per duration (1, 3, 6, 12, 24 hr):

  Circle position : 1°×1° grid cell centre
  Circle SIZE     : number of GSDR stations in that cell
  Circle COLOR    : all-time max accumulation (inches), log-normalised

Colour design
-------------
  Palette  : ColorBrewer YlGnBu 9-class — colorblind-verified, journal standard
             (yellow-green = low  →  dark navy = extreme)
  Scale    : shared log₁ₚ normalization across all 5 plots
             - log₁ₚ(value) used internally for colour mapping
             - colorbar tick labels show ACTUAL inches (not log values)
             - hover tooltip shows ACTUAL inches and mm
  Why log  : compresses extreme outliers so mid-range variation is visible
             on all 5 plots simultaneously while remaining directly comparable

Output
------
  gsdr/outputs/extreme_grid_{dur}hr.html  — interactive (zoom, hover)
  gsdr/outputs/extreme_grid_{dur}hr.png   — static 1400×800 px, 150 dpi

Workflow
--------
1. Loads station index (gsdr_us_index.csv).
2. Loads cached per-station maxima (gsdr_station_maxima.csv) if present,
   otherwise reads all gauge files (~15 min) and saves the cache.
3. Generates HTML and PNG for each duration.

Environment variable
--------------------
  GSDR_QC_DIR : path to the QC_d data - US folder
                default: ~/Desktop/GSDR/QC_d data - US

Usage:
    python3 gsdr/gsdr_extreme_maps.py

Dependencies: pandas, numpy, plotly, kaleido
"""

import os
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH  = os.path.join(BASE_DIR, "gsdr_us_index.csv")
CACHE_PATH  = os.path.join(BASE_DIR, "gsdr_station_maxima.csv")
QC_DIR      = os.environ.get("GSDR_QC_DIR", os.path.expanduser("~/Desktop/GSDR/QC_d data - US"))
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DURATIONS    = [1, 3, 6, 12, 24]
DUR_LABELS   = {1: "1-hr", 3: "3-hr", 6: "6-hr", 12: "12-hr", 24: "24-hr"}
MM_TO_IN     = 1.0 / 25.4
HEADER_LINES = 21
MISSING_VAL  = -999.0
GRID_DEG     = 1.0

# ── ColorBrewer YlGnBu 9-class (colorblind-verified, multi-hue) ───────────
# Yellow → green → teal → blue → dark navy across 5 hue families.
# Far more perceptually distinct than single-hue Blues.
# Standard in Nature/GRL/JHM precipitation maps.
COLORSCALE = [
    [0.000, "#ffffd9"],   # pale yellow   — trace/no rain
    [0.125, "#edf8b1"],   # yellow-green  — very low
    [0.250, "#c7e9b4"],   # light green   — low
    [0.375, "#7fcdbb"],   # teal-green    — moderate-low
    [0.500, "#41b6c4"],   # teal          — moderate
    [0.625, "#1d91c0"],   # sky blue      — moderate-high
    [0.750, "#225ea8"],   # medium blue   — high
    [0.875, "#253494"],   # dark blue     — very high
    [1.000, "#081d58"],   # dark navy     — extreme
]

# Colorbar tick values shown to the user (actual inches)
TICK_INCHES = [0.5, 1, 2, 4, 6, 8, 10, 15]


# ── Helpers ────────────────────────────────────────────────────────────────

def load_and_compute(station_id, durations):
    fpath = os.path.join(QC_DIR, f"{station_id}.txt")
    if not os.path.exists(fpath):
        return None
    try:
        lines = []
        with open(fpath) as f:
            for i, line in enumerate(f):
                if i >= HEADER_LINES:
                    lines.append(line.strip())
        data = np.array(lines, dtype=np.float32)
        data[data == MISSING_VAL] = np.nan
        s = pd.Series(data)
        return {d: float(v) if not np.isnan(v := s.rolling(d, min_periods=d).sum().max()) else np.nan
                for d in durations}
    except Exception:
        return None


def build_grid(df, dur_col):
    df = df.dropna(subset=[dur_col, "LAT", "LON"]).copy()
    df["cell_lat"] = (np.floor(df["LAT"] / GRID_DEG) * GRID_DEG + GRID_DEG / 2).round(1)
    df["cell_lon"] = (np.floor(df["LON"] / GRID_DEG) * GRID_DEG + GRID_DEG / 2).round(1)
    return (
        df.groupby(["cell_lat", "cell_lon"])
        .agg(
            max_val=(dur_col, "max"),
            n_stations=("ID", "count"),
            best_station=("ID", lambda x: x.iloc[df.loc[x.index, dur_col].argmax()]),
        )
        .reset_index()
    )


# ── Step 1: Load station index ─────────────────────────────────────────────

print("Loading station index …")
index = pd.read_csv(INDEX_PATH)
index["start_year"] = index["START"].astype(str).str[:4].astype(float)
index["end_year"]   = index["END"].astype(str).str[:4].astype(float)
index["record_len"] = (index["end_year"] - index["start_year"]).clip(lower=0)
index = index[(index["record_len"] > 0) & (index["PCT_MISSING"] <= 50)].copy()
print(f"  {len(index):,} stations after quality filter.\n")


# ── Step 2: Per-station maxima ─────────────────────────────────────────────

if os.path.exists(CACHE_PATH):
    print(f"Loading cached maxima → {CACHE_PATH}")
    maxima = pd.read_csv(CACHE_PATH)
    maxima = maxima[maxima["ID"].isin(index["ID"])].copy()
    print(f"  {len(maxima):,} stations.\n")
else:
    print("Cache not found — reading all gauge files (~15 min) …")
    rows = []
    for i, sid in enumerate(index["ID"]):
        res = load_and_compute(sid, DURATIONS)
        row = {"ID": sid}
        for d in DURATIONS:
            row[f"max_{d}hr_mm"] = res[d] if res else np.nan
        rows.append(row)
        if (i + 1) % 500 == 0:
            print(f"  {i+1:,} / {len(index):,} …")
    maxima = pd.DataFrame(rows)
    maxima.to_csv(CACHE_PATH, index=False)
    print(f"Cache saved → {CACHE_PATH}\n")

maxima = maxima.merge(index[["ID", "LAT", "LON", "record_len"]], on="ID", how="inner")

for d in DURATIONS:
    maxima[f"max_{d}hr_in"] = maxima[f"max_{d}hr_mm"] * MM_TO_IN

print("Per-station maxima (inches):")
for d in DURATIONS:
    v = maxima[f"max_{d}hr_in"].dropna()
    print(f"  {DUR_LABELS[d]:>5}  n={len(v):,}  "
          f"median={v.median():.2f}  p99={v.quantile(0.99):.2f}  max={v.max():.2f} in")
print()


# ── Step 3: Log normalization (shared across all 5 plots) ─────────────────

# Upper bound: 99th percentile of 24-hr values
color_max_in  = maxima["max_24hr_in"].quantile(0.99)
log_cmin      = 0.0
log_cmax      = float(np.log1p(color_max_in))

# Colorbar: tick positions in log space, labels in actual inches
tick_vals  = [float(np.log1p(t)) for t in TICK_INCHES if t <= color_max_in * 1.05]
tick_text  = [f"{t} in" for t in TICK_INCHES if t <= color_max_in * 1.05]
# Always include the max tick
if not any(abs(t - log_cmax) < 0.05 for t in tick_vals):
    tick_vals.append(log_cmax)
    tick_text.append(f"{color_max_in:.1f} in")

colorbar_cfg = dict(
    title=dict(text="Max precip (inches)", font=dict(size=11)),
    tickvals=tick_vals,
    ticktext=tick_text,
    tickfont=dict(size=10),
    thickness=15,
    len=0.65,
)

print(f"Shared log-normalised colour scale:")
print(f"  Actual range : 0 – {color_max_in:.2f} in (99th pct of 24-hr)")
print(f"  Log range    : {log_cmin:.3f} – {log_cmax:.3f}")
print(f"  Colorbar ticks (actual inches): {[t for t in TICK_INCHES if t <= color_max_in*1.05]}\n")


# ── Step 4: Map layout ─────────────────────────────────────────────────────

geo_layout = dict(
    scope="usa",
    showland=True,
    landcolor="rgb(235,235,235)",
    showlakes=True,
    lakecolor="rgb(210,228,255)",
    showcoastlines=True,
    coastlinecolor="rgb(130,130,130)",
    coastlinewidth=0.8,
    showsubunits=True,
    subunitcolor="rgb(170,170,170)",
    subunitwidth=0.5,
    bgcolor="white",
    projection_type="albers usa",
)


# ── Step 5: Generate 5 grid cell maps ─────────────────────────────────────

print("Generating maps …")

for d in DURATIONS:
    dur_label = DUR_LABELS[d]
    col_in    = f"max_{d}hr_in"
    grid      = build_grid(maxima, col_in)

    # Log-transform values for colour mapping
    log_vals = np.log1p(grid["max_val"].clip(lower=0))

    # Circle size: log scale of station count, 8–36 px
    max_n = grid["n_stations"].max()
    sizes = (6 + 18 * np.log1p(grid["n_stations"] - 1) /
             np.log1p(max(max_n - 1, 1))).clip(6, 24)

    hover_text = [
        (f"<b>{row.cell_lat:.1f}°N, {row.cell_lon:.1f}°W</b><br>"
         f"Max {dur_label}: <b>{row.max_val:.2f} in</b> ({row.max_val * 25.4:.1f} mm)<br>"
         f"Stations in cell: {int(row.n_stations)}<br>"
         f"Best station: {row.best_station}")
        for _, row in grid.iterrows()
    ]

    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lat=grid["cell_lat"],
        lon=grid["cell_lon"],
        mode="markers",
        marker=dict(
            size=sizes,
            color=log_vals,
            colorscale=COLORSCALE,
            cmin=log_cmin,
            cmax=log_cmax,
            colorbar=colorbar_cfg,
            opacity=0.82,
            line=dict(width=0.5, color="rgba(0,0,0,0.25)"),
        ),
        text=hover_text,
        hoverinfo="text",
        name="",
    ))

    fig.update_layout(
        title=dict(
            text=(
                f"GSDR US — All-Time Maximum {dur_label} Rainfall<br>"
                f"<sup>"
                f"Colour: max inches per 1°×1° cell (log-scaled for visual clarity — "
                f"hover shows actual value)  |  "
                f"Size: station count in cell  |  "
                f"Colour scale shared across all 5 duration maps"
                f"</sup>"
            ),
            font=dict(size=14),
            x=0.5,
            xanchor="center",
        ),
        geo=geo_layout,
        margin=dict(l=0, r=0, t=75, b=10),
        paper_bgcolor="white",
    )

    out_html = os.path.join(OUTPUT_DIR, f"extreme_grid_{d}hr.html")
    out_png  = os.path.join(OUTPUT_DIR, f"extreme_grid_{d}hr.png")
    fig.write_html(out_html)
    fig.write_image(out_png, width=1400, height=800, scale=2)
    sz_html = os.path.getsize(out_html) / 1024
    sz_png  = os.path.getsize(out_png)  / 1024
    print(f"  extreme_grid_{d}hr.html  ({sz_html:.0f} KB)  |  extreme_grid_{d}hr.png  ({sz_png:.0f} KB)")

print()
print("=" * 65)
print("10 files saved to gsdr/outputs/:")
for d in DURATIONS:
    print(f"  extreme_grid_{d}hr.html  +  extreme_grid_{d}hr.png")
print()
print(f"Colour palette : ColorBrewer YlGnBu 9-class (colorblind-verified)")
print(f"Normalisation  : shared log₁ₚ scale  →  0 – {color_max_in:.2f} in")
print(f"Hover values   : actual inches and mm (not log-transformed)")
print(f"PNG size       : 1400×800 px, scale=2 (high resolution)")
print("=" * 65)
