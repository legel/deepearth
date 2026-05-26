"""
GSDR US — Monthly Maximum Rainfall Grid Maps
==============================================
Generates 24 static PNG maps showing the spatial distribution of
monthly-maximum sub-daily rainfall across the US for two durations:

    Durations : 1-hour, 12-hour
    Months    : January – December  (24 maps total)

For each month and duration, each 1°×1° grid cell is coloured by the
all-time maximum rolling accumulation observed in that month across all
years of record at stations in the cell.  This reveals the seasonal
geography of extreme sub-daily rainfall.

Design
------
  Circle position : centre of 1°×1° grid cell
  Circle SIZE     : number of GSDR stations in that cell
  Circle COLOR    : monthly max accumulation (inches), log₁ₚ-normalised

  Colour palette  : ColorBrewer YlGnBu 9-class (colorblind-verified)
  Colour scale    : shared log₁ₚ normalisation within each duration
                    (the 12 monthly maps for a given duration share the
                    same scale, enabling direct month-to-month comparison)
  Size legend     : inset legend shows 4 reference circles (1, 5, 10, 20
                    stations) so readers can interpret circle size

Output
------
  outputs/max_1_hour_<month>_rainfall_us.png   — 12 files
  outputs/max_12_hour_<month>_rainfall_us.png  — 12 files

Cache
-----
  gsdr_station_monthly_maxima.csv
  Per-station, per-month all-time maximum rolling accumulation for 1-hr
  and 12-hr windows.  Auto-generated from raw gauge files if absent
  (~15 min); can be reused across runs.

Usage
-----
    python3 gsdr/gsdr_monthly_maps.py

Environment variable
--------------------
  GSDR_QC_DIR : path to the QC_d data - US folder
                default: ~/Desktop/GSDR/QC_d data - US

Dependencies: pandas, numpy, plotly, kaleido
"""

import os
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "gsdr_us_index.csv")
CACHE_PATH = os.path.join(BASE_DIR, "gsdr_station_monthly_maxima.csv")
QC_DIR     = os.environ.get("GSDR_QC_DIR", os.path.expanduser("~/Desktop/GSDR/QC_d data - US"))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DURATIONS    = [1, 12]
MM_TO_IN     = 1.0 / 25.4
HEADER_LINES = 21
MISSING_VAL  = -999.0
GRID_DEG     = 1.0

MONTHS = [
    (1,  "january"),   (2,  "february"),  (3,  "march"),
    (4,  "april"),     (5,  "may"),       (6,  "june"),
    (7,  "july"),      (8,  "august"),    (9,  "september"),
    (10, "october"),   (11, "november"),  (12, "december"),
]
MONTH_ABBR  = {num: name[:3] for num, name in MONTHS}
MONTH_TITLE = {name: name.capitalize() for _, name in MONTHS}
DUR_LABEL   = {1: "1-hour", 12: "12-hour"}

COLORSCALE = [
    [0.000, "#ffffd9"],
    [0.125, "#edf8b1"],
    [0.250, "#c7e9b4"],
    [0.375, "#7fcdbb"],
    [0.500, "#41b6c4"],
    [0.625, "#1d91c0"],
    [0.750, "#225ea8"],
    [0.875, "#253494"],
    [1.000, "#081d58"],
]

TICK_INCHES = [0.5, 1, 2, 4, 6, 8, 10, 15]

GEO_LAYOUT = dict(
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


# ── Helpers ────────────────────────────────────────────────────────────────

def load_gauge(filepath):
    header = {}
    data_lines = []
    with open(filepath) as f:
        for i, line in enumerate(f):
            if i < HEADER_LINES:
                if ":" in line:
                    k, _, v = line.partition(":")
                    header[k.strip()] = v.strip()
            else:
                data_lines.append(line.strip())
    data = np.array(data_lines, dtype=np.float32)
    data[data == MISSING_VAL] = np.nan
    s = str(header.get("Start datetime", "")).strip()
    if len(s) < 10:
        return data, None
    try:
        start_dt = datetime(int(s[0:4]), int(s[4:6]), int(s[6:8]), int(s[8:10]))
    except Exception:
        start_dt = None
    return data, start_dt


def monthly_maxima(data, start_dt, durations):
    """Per-month all-time maximum rolling accumulation for each duration."""
    if start_dt is None:
        return None
    idx = pd.date_range(start=start_dt, periods=len(data), freq="h")
    s   = pd.Series(data, index=idx)
    results = {}
    for d in durations:
        rolled   = s.rolling(window=d, min_periods=d).sum()
        by_month = rolled.groupby(rolled.index.month).max()
        results[d] = {m: float(v) if not pd.isna(v) else np.nan
                      for m, v in by_month.items()}
    return results


def build_grid(df, col):
    df = df.dropna(subset=[col, "LAT", "LON"]).copy()
    df["cell_lat"] = (np.floor(df["LAT"] / GRID_DEG) * GRID_DEG + GRID_DEG / 2).round(1)
    df["cell_lon"] = (np.floor(df["LON"] / GRID_DEG) * GRID_DEG + GRID_DEG / 2).round(1)
    return (
        df.groupby(["cell_lat", "cell_lon"])
        .agg(
            max_val=(col, "max"),
            n_stations=("ID", "count"),
            best_station=("ID", lambda x: x.iloc[df.loc[x.index, col].argmax()]),
        )
        .reset_index()
    )


def marker_size(n_stations, global_max_n):
    return float(np.clip(
        6 + 18 * np.log1p(n_stations - 1) / np.log1p(max(global_max_n - 1, 1)),
        6, 24,
    ))


def add_size_legend(fig, global_max_n, legend_counts=(1, 5, 10, 20)):
    """Inset circle-size legend placed in the lower-left corner of the figure."""
    fig.add_shape(
        type="rect", xref="paper", yref="paper",
        x0=0.010, y0=0.060, x1=0.182, y1=0.282,
        fillcolor="rgba(255,255,255,0.88)",
        line=dict(color="rgba(80,80,80,0.45)", width=0.8),
        layer="above",
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=0.096, y=0.258,
        text="<b>Circle size</b>",
        showarrow=False, font=dict(size=10, color="#333"),
        xanchor="center",
    )

    y_positions = [0.222, 0.178, 0.134, 0.090]
    x_circle, x_label = 0.042, 0.082

    for i, count in enumerate(legend_counts):
        sz  = marker_size(count, global_max_n)
        r_x = sz / 2 / 1400
        r_y = sz / 2 / 800
        y   = y_positions[i]
        fig.add_shape(
            type="circle", xref="paper", yref="paper",
            x0=x_circle - r_x, y0=y - r_y,
            x1=x_circle + r_x, y1=y + r_y,
            fillcolor="rgba(80,80,80,0.65)",
            line=dict(color="rgba(0,0,0,0.5)", width=0.5),
            layer="above",
        )
        label = "1 station" if count == 1 else f"{count} stations"
        fig.add_annotation(
            xref="paper", yref="paper",
            x=x_label, y=y,
            text=label, showarrow=False,
            font=dict(size=9, color="#333"),
            xanchor="left", yanchor="middle",
        )


# ── Step 1: Load station index ─────────────────────────────────────────────

print("Loading station index …")
index = pd.read_csv(INDEX_PATH)
index = index[index["PCT_MISSING"] <= 50].copy()
print(f"  {len(index):,} stations after quality filter.\n")


# ── Step 2: Monthly maxima cache ───────────────────────────────────────────

if os.path.exists(CACHE_PATH):
    print(f"Loading cached monthly maxima → {CACHE_PATH}")
    maxima = pd.read_csv(CACHE_PATH)
    maxima = maxima[maxima["ID"].isin(index["ID"])].copy()
    print(f"  {len(maxima):,} stations.\n")
else:
    print("Cache not found — reading all gauge files (~15 min) …")
    rows = []
    for i, sid in enumerate(index["ID"]):
        fpath = os.path.join(QC_DIR, f"{sid}.txt")
        if not os.path.exists(fpath):
            continue
        try:
            data, start_dt = load_gauge(fpath)
            res = monthly_maxima(data, start_dt, DURATIONS)
        except Exception:
            res = None
        row = {"ID": sid}
        for d in DURATIONS:
            for m_num, m_name in MONTHS:
                col = f"max_{d}hr_{MONTH_ABBR[m_num]}_mm"
                row[col] = res[d].get(m_num, np.nan) if res and d in res else np.nan
        rows.append(row)
        if (i + 1) % 500 == 0:
            print(f"  {i+1:,} / {len(index):,} …")
    maxima = pd.DataFrame(rows)
    maxima.to_csv(CACHE_PATH, index=False)
    print(f"Cache saved → {CACHE_PATH}\n")

maxima = maxima.merge(index[["ID", "LAT", "LON"]], on="ID", how="inner")

for d in DURATIONS:
    for m_num, m_name in MONTHS:
        col_mm = f"max_{d}hr_{MONTH_ABBR[m_num]}_mm"
        col_in = f"max_{d}hr_{MONTH_ABBR[m_num]}_in"
        if col_mm in maxima.columns:
            maxima[col_in] = maxima[col_mm] * MM_TO_IN


# ── Step 3: Pre-build grids and global circle-size scale ───────────────────

print("Building grid cells …")
grids = {}
for d in DURATIONS:
    for m_num, m_name in MONTHS:
        col_in = f"max_{d}hr_{MONTH_ABBR[m_num]}_in"
        if col_in in maxima.columns:
            grids[(d, m_name)] = build_grid(maxima, col_in)

global_max_n = max(
    int(g["n_stations"].max()) for g in grids.values() if len(g) > 0
)
print(f"  Global max stations per cell: {global_max_n}\n")


# ── Step 4: Shared colour scale per duration ───────────────────────────────

print("Computing shared colour scales …")
dur_color_max = {}
for d in DURATIONS:
    vals = []
    for _, m_name in MONTHS:
        col_in = f"max_{d}hr_{m_name[:3]}_in"
        if col_in in maxima.columns:
            vals.extend(maxima[col_in].dropna().tolist())
    dur_color_max[d] = float(np.percentile(vals, 99)) if vals else 15.0
    print(f"  {d:>2}-hr : colour upper bound = {dur_color_max[d]:.2f} in "
          f"(99th pct across all months)")
print()


# ── Step 5: Generate 24 PNG maps ──────────────────────────────────────────

print("Generating maps …")

for d in DURATIONS:
    color_max_in = dur_color_max[d]
    log_cmin     = 0.0
    log_cmax     = float(np.log1p(color_max_in))

    tick_vals = [float(np.log1p(t)) for t in TICK_INCHES if t <= color_max_in * 1.05]
    tick_text = [f"{t} in" for t in TICK_INCHES if t <= color_max_in * 1.05]
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

    for m_num, m_name in MONTHS:
        grid = grids.get((d, m_name))
        if grid is None or grid.empty:
            print(f"  Skipping {d}hr {m_name}: no data.")
            continue

        log_vals = np.log1p(grid["max_val"].clip(lower=0))
        sizes    = np.array([marker_size(n, global_max_n) for n in grid["n_stations"]])

        hover_text = [
            (f"<b>{row.cell_lat:.1f}°N, {row.cell_lon:.1f}°W</b><br>"
             f"Max {DUR_LABEL[d]} ({MONTH_TITLE[m_name]}): "
             f"<b>{row.max_val:.2f} in</b> ({row.max_val * 25.4:.1f} mm)<br>"
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

        add_size_legend(fig, global_max_n)

        fig.update_layout(
            title=dict(
                text=(
                    f"GSDR US — Maximum {DUR_LABEL[d]} Rainfall: {MONTH_TITLE[m_name]}<br>"
                    f"<sup>"
                    f"Colour: monthly max inches per 1°×1° cell (log-scaled — "
                    f"hover shows actual value)  |  "
                    f"Size: station count in cell  |  "
                    f"Colour scale shared across all 12 monthly maps for this duration"
                    f"</sup>"
                ),
                font=dict(size=14),
                x=0.5,
                xanchor="center",
            ),
            geo=GEO_LAYOUT,
            margin=dict(l=0, r=0, t=75, b=10),
            paper_bgcolor="white",
        )

        out_png = os.path.join(OUTPUT_DIR, f"max_{d}_hour_{m_name}_rainfall_us.png")
        fig.write_image(out_png, width=1400, height=800, scale=2)
        sz = os.path.getsize(out_png) / 1024
        print(f"  max_{d}_hour_{m_name}_rainfall_us.png  ({sz:.0f} KB)")

print()
print("=" * 65)
print("24 PNG files saved to gsdr/outputs/")
print()
for d in DURATIONS:
    for _, m_name in MONTHS:
        print(f"  max_{d}_hour_{m_name}_rainfall_us.png")
print()
print(f"Colour palette : ColorBrewer YlGnBu 9-class (colorblind-verified)")
print(f"Normalisation  : shared log₁ₚ scale per duration")
print(f"  1-hr  upper bound : {dur_color_max[1]:.2f} in")
print(f"  12-hr upper bound : {dur_color_max[12]:.2f} in")
print(f"PNG size       : 1400×800 px, scale=2 (high resolution)")
print("=" * 65)
