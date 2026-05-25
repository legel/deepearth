# GSDR Sub-Daily Precipitation Extremes — Data Description and Usage Guide

This package contains analysis scripts, pre-computed station indices, and pre-generated visualizations derived from the Global Sub-Daily Rainfall dataset (GSDR) for the US. The outputs characterise historical extreme sub-daily rainfall at any geospatial coordinate and are designed as input features for property-level flood risk modelling.

---

## Contents

```
models/flood_hydrology/
├── gsdr/
│   ├── gsdr_build_index.py               One-time station index builder
│   ├── gsdr_intensity_matrix.py          Query: sub-daily max precip by radius/duration
│   ├── gsdr_visualizations.py            Dataset exploration figures
│   ├── gsdr_extreme_maps.py              US extreme rainfall maps (5 durations)
│   ├── gsdr_us_index.csv                 Pre-built station metadata index
│   ├── gsdr_station_maxima.csv           Pre-computed per-station maxima cache
│   └── outputs/
│       ├── fig01_us_stations.html        Interactive US station map
│       ├── fig02_record_length.png       Record length distribution
│       ├── fig03_temporal_coverage.png   Active stations per year
│       ├── fig04_missing_data.png        Missing data rate distribution
│       ├── fig05_intensity_scatter.png   1-hr vs 24-hr max scatter
│       ├── fig06_peak_event_year.png     Year of all-time 1-hr max per station
│       ├── extreme_grid_1hr.html/.png    Extreme rainfall map — 1-hour duration
│       ├── extreme_grid_3hr.html/.png    Extreme rainfall map — 3-hour duration
│       ├── extreme_grid_6hr.html/.png    Extreme rainfall map — 6-hour duration
│       ├── extreme_grid_12hr.html/.png   Extreme rainfall map — 12-hour duration
│       └── extreme_grid_24hr.html/.png   Extreme rainfall map — 24-hour duration
└── README.md
```

---

## Dataset — GSDR (Global Sub-Daily Rainfall, US open-access subset)

| Item | Detail |
|---|---|
| Full name | Global Sub-Daily Rainfall dataset (INTENSE project) |
| Source | NOAA + partner agencies; Lewis et al., *Journal of Climate*, 2019 |
| US gauges | 6,605 quality-controlled hourly stations |
| Time span | ~1948–2014 |
| Unit | **mm** — continuous hourly precipitation accumulation |
| Format | INTENSE format: 21-line header + one mm value per line; missing = −999 |
| Download | [Zenodo record 8369987](https://zenodo.org/records/8369987) — download `QC_d data - US.zip` |

### Column structure of `gsdr_us_index.csv`

| Column | Description |
|---|---|
| `ID` | Station identifier (e.g. `US_086638`) |
| `LAT` | Latitude (decimal degrees, WGS-84) |
| `LON` | Longitude (decimal degrees, WGS-84) |
| `START` | Record start (YYYYMMDDHH) |
| `END` | Record end (YYYYMMDDHH) |
| `N_RECORDS` | Total number of hourly records |
| `PCT_MISSING` | Percentage of missing values |

### Column structure of `gsdr_station_maxima.csv`

Pre-computed all-time maximum rolling accumulation per station for each duration. Used by `gsdr_extreme_maps.py` to avoid reprocessing all gauge files.

| Column | Description |
|---|---|
| `ID` | Station identifier |
| `max_1hr_mm` | All-time max 1-hour accumulation (mm) |
| `max_3hr_mm` | All-time max 3-hour accumulation (mm) |
| `max_6hr_mm` | All-time max 6-hour accumulation (mm) |
| `max_12hr_mm` | All-time max 12-hour accumulation (mm) |
| `max_24hr_mm` | All-time max 24-hour accumulation (mm) |

---

## Setup

### Python dependencies

```bash
pip install pandas numpy matplotlib seaborn plotly kaleido
```

### GSDR raw data (required for `gsdr_intensity_matrix.py` and `gsdr_visualizations.py`)

1. Download `QC_d data - US.zip` from [Zenodo record 8369987](https://zenodo.org/records/8369987)
2. Extract to a local folder (≈ 4.4 GB uncompressed)
3. Set the environment variable:

```bash
export GSDR_QC_DIR="/path/to/QC_d data - US"
```

If `GSDR_QC_DIR` is not set, scripts default to `~/Desktop/GSDR/QC_d data - US`.

> `gsdr_extreme_maps.py` uses the pre-built cache files included in this package and does **not** require raw gauge files.

---

## Scripts

All scripts use `__file__`-based absolute paths and can be run from any directory.

---

### `gsdr_build_index.py`

Scans the 21-line header of every gauge file and builds `gsdr_us_index.csv`. A pre-built index is included — only rerun if you have a different GSDR download.

```bash
python3 gsdr/gsdr_build_index.py
```

Runtime: ~45 seconds. Requires raw GSDR data.

---

### `gsdr_intensity_matrix.py`

Given a geospatial coordinate, finds all GSDR stations within each search radius and returns the **all-time observed maximum precipitation accumulation** for each duration window. Values are raw historical observations — no statistical distribution fitting.

```bash
python3 gsdr/gsdr_intensity_matrix.py --lat 28.5652 --lon -81.5868
python3 gsdr/gsdr_intensity_matrix.py --lat 37.38575 --lon -122.00037
```

| Argument | Default | Description |
|---|---|---|
| `--lat` | required | Latitude (decimal degrees, WGS-84) |
| `--lon` | required | Longitude (decimal degrees, WGS-84) |
| `--radii` | `10 50 100` | Search radii in km |
| `--durations` | `1 3 6 12 24` | Accumulation windows in hours |
| `--no-csv` | — | Print results only, skip saving CSV |

**Output:** `gsdr/outputs/gsdr_intensity_{lat}_{lon}.csv`

Example output (Winter Garden, FL):

| Duration | ≤10 km | ≤50 km | ≤100 km |
|---|---|---|---|
| 1-hr | N/A | 143.5 mm (5.65 in) | 143.5 mm (5.65 in) |
| 6-hr | N/A | 232.4 mm (9.15 in) | 248.4 mm (9.78 in) |
| 24-hr | N/A | 363.0 mm (14.29 in) | 363.0 mm (14.29 in) |

N/A indicates no station within that radius. Values increase monotonically across radii since each larger radius is a superset of the smaller.

Requires raw GSDR data.

---

### `gsdr_visualizations.py`

Generates 6 dataset exploration figures saved to `gsdr/outputs/`. Pre-generated outputs are included. Figs 1–4 run in seconds; Figs 5–6 load all gauge files and take 10–20 minutes.

```bash
python3 gsdr/gsdr_visualizations.py
```

| Figure | Description |
|---|---|
| `fig01_us_stations.html` | Interactive US map — 6,605 hourly gauges coloured by record length |
| `fig02_record_length.png` | Distribution of years of data per station |
| `fig03_temporal_coverage.png` | Active stations per year across full record |
| `fig04_missing_data.png` | Missing data rate per station; shows the 50% filter threshold |
| `fig05_intensity_scatter.png` | 1-hr vs 24-hr all-time max — distinguishes flash-flood vs slow-rain regimes |
| `fig06_peak_event_year.png` | Year each station recorded its all-time 1-hr maximum |

Figs 5–6 require raw GSDR data.

---

### `gsdr_extreme_maps.py`

Generates 5 interactive HTML maps and 5 static PNG exports showing the spatial distribution of extreme sub-daily rainfall across the US, one per duration. Uses the pre-built cache — does **not** require raw GSDR data.

```bash
python3 gsdr/gsdr_extreme_maps.py
```

| Design element | Detail |
|---|---|
| Circle position | Centre of 1°×1° grid cell |
| Circle size | Number of GSDR stations in that cell |
| Circle colour | All-time maximum accumulation in inches |
| Colour palette | ColorBrewer YlGnBu 9-class (colorblind-verified, journal standard) |
| Colour scale | Shared log₁ₚ normalisation across all 5 maps (0–15 in) |
| Hover tooltip | Actual inches and mm, station count, best station ID |
| PNG resolution | 1400×800 px, 2× scale |
