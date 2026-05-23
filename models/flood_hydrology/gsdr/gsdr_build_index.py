"""
GSDR US — Station Metadata Index Builder
=========================================
One-time script. Scans the first 21 header lines of all 6,605 QC'd GSDR
gauge files and saves a compact station index CSV:

    gsdr_us_index.csv  →  ID, LAT, LON, START, END, N_RECORDS, PCT_MISSING

This index lets the query script (gsdr_intensity_matrix.py) identify
which stations fall within a search radius without loading any time series.

Environment variable:
    GSDR_QC_DIR : path to the QC_d data - US folder
                  default: ~/Desktop/GSDR/QC_d data - US

Usage:
    python3 gsdr/gsdr_build_index.py

Runtime: ~30–60 seconds (reads headers only, no time series loaded).
Dependencies: pandas
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QC_DIR   = os.environ.get("GSDR_QC_DIR", os.path.expanduser("~/Desktop/GSDR/QC_d data - US"))
OUT_PATH = os.path.join(BASE_DIR, "gsdr_us_index.csv")
HEADER_LINES = 21  # all QC_d files have exactly 21 header lines before data

def parse_header(filepath):
    """Extract metadata from the 21-line INTENSE format header."""
    meta = {}
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            if i >= HEADER_LINES:
                break
            if ":" in line:
                key, _, val = line.partition(":")
                meta[key.strip()] = val.strip()
    return meta

print(f"Scanning {QC_DIR} …")
files = sorted(os.listdir(QC_DIR))
print(f"  Found {len(files):,} gauge files.\n")

records = []
for i, fname in enumerate(files):
    if not fname.endswith(".txt"):
        continue
    path = os.path.join(QC_DIR, fname)
    try:
        m = parse_header(path)
        records.append({
            "ID":          m.get("Station ID", fname.replace(".txt", "")),
            "LAT":         float(m.get("Latitude", "nan")),
            "LON":         float(m.get("Longitude", "nan")),
            "START":       str(m.get("Start datetime", "")),
            "END":         str(m.get("End datetime", "")),
            "N_RECORDS":   int(m.get("Number of records", 0)),
            "PCT_MISSING": float(m.get("Percent missing data", 100)),
        })
    except Exception as e:
        print(f"  WARNING: could not parse {fname}: {e}")

    if (i + 1) % 500 == 0:
        print(f"  {i+1:,} / {len(files):,} files scanned …")

index = pd.DataFrame(records)
index.to_csv(OUT_PATH, index=False)
print(f"\nIndex saved → {OUT_PATH}")
print(f"  {len(index):,} stations")
print(f"  LAT range : {index['LAT'].min():.2f} – {index['LAT'].max():.2f}")
print(f"  LON range : {index['LON'].min():.2f} – {index['LON'].max():.2f}")
print(f"  START year range : {index['START'].str[:4].min()} – {index['START'].str[:4].max()}")
print(f"  END year range   : {index['END'].str[:4].min()} – {index['END'].str[:4].max()}")
print(f"  Median missing   : {index['PCT_MISSING'].median():.1f}%")
