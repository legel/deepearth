"""FIA (USDA Forest Inventory and Analysis) national plots -> live-tree species presence per plot location.
Source: https://apps.fs.usda.gov/fia/datamart/CSV/ (ENTIRE_PLOT.csv, ENTIRE_TREE.csv, REF_SPECIES.csv).

Streams ENTIRE_TREE.csv (13.7 GB) keeping PLT_CN, SPCD, STATUSCD (1 = live), and joins plot coordinates
(public FIA coordinates are perturbed by up to ~1.6 km and swapped on some private plots).
"""
import sys
from pathlib import Path

import pandas as pd

out = Path(sys.argv[1])
BASE = str(out)                                   # files fetched beforehand with aria2c (16 connections)
plots = pd.read_csv(f"{BASE}/ENTIRE_PLOT.csv", usecols=["CN", "LAT", "LON", "INVYR", "STATECD", "PLOT_STATUS_CD"],
                    low_memory=False)
plots = plots[plots.PLOT_STATUS_CD == 1]                      # sampled, forested plots
pres = []
for chunk in pd.read_csv(f"{BASE}/ENTIRE_TREE.csv", usecols=["PLT_CN", "SPCD", "STATUSCD"], chunksize=5_000_000,
                         low_memory=False):
    c = chunk[chunk.STATUSCD == 1].drop_duplicates(["PLT_CN", "SPCD"])
    pres.append(c[["PLT_CN", "SPCD"]])
    print("chunk", len(pres), flush=True)
p = pd.concat(pres).drop_duplicates()
ref = pd.read_csv(f"{BASE}/REF_SPECIES.csv",
                  usecols=["SPCD", "GENUS", "SPECIES", "COMMON_NAME"], low_memory=False)
ref["scientificName"] = ref.GENUS.str.strip() + " " + ref.SPECIES.str.strip()
p = p.merge(ref[["SPCD", "scientificName"]], on="SPCD", how="left").merge(
    plots.rename(columns={"CN": "PLT_CN"}), on="PLT_CN", how="inner")
p.to_csv(out / "fia_live_tree_presence.csv", index=False)
plots.to_csv(out / "fia_plots.csv", index=False)
print("plots", plots.CN.nunique(), "plot x species", len(p), "species", p.scientificName.nunique())
