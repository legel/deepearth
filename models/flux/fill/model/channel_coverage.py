"""Coverage of the tower channels the model does not use yet (E30 step 1): per channel and split, the sites with any
measured value, the fraction of half-hours measured at those sites, and the non-overlapping 180-day windows with the
channel measured in at least 5 % of their half-hours.

Measured-only values under the dataset's QC (towers/hh tables; data/towers/README.md). Coverage counts only: no value
of a test site enters any model. *_RANDUNC (ONEFlux random-uncertainty estimates) and derived products are not
candidates.

Usage: python channel_coverage.py [--out /home/photon/4tb/fluxfill/model/eval/channel_coverage.csv]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HH = Path("/home/photon/4tb/fluxfill/towers/hh")
SPLIT = Path("/home/photon/4tb/fluxfill/towers/split.csv")
CANDIDATES = (["SW_OUT", "LW_OUT", "NETRAD", "PPFD_IN", "PPFD_OUT", "RH", "WD", "CO2", "WTD"]
              + [f"SWC_{i}" for i in range(2, 6)] + [f"TS_{i}" for i in range(2, 6)])
CURRENT = ["NEE", "SWC_1", "TS_1"]                     # references (in the model)
WIN = 180 * 48
MIN_FRAC = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/photon/4tb/fluxfill/model/eval/channel_coverage.csv")
    a = ap.parse_args()
    split = pd.read_csv(SPLIT).set_index("site_id").split
    rows = []
    for f in sorted(HH.glob("*.parquet")):
        site = f.stem
        names = set(pq.read_schema(f).names)
        cols = [c for c in CANDIDATES + CURRENT if c in names]
        t = pq.read_table(f, columns=cols).to_pandas() if cols else pd.DataFrame()
        T = pq.ParquetFile(f).metadata.num_rows
        for c in CANDIDATES + CURRENT:
            m = np.isfinite(t[c].to_numpy(np.float64)) if c in t else np.zeros(T, bool)
            n_win = int(sum(m[i:i + WIN].mean() >= MIN_FRAC for i in range(0, max(T - WIN + 1, 1), WIN))) if m.any() else 0
            rows.append({"site": site, "split": split.get(site, "none"), "channel": c, "n_steps": T,
                         "n_measured": int(m.sum()), "n_windows": n_win})
    d = pd.DataFrame(rows)
    pd.DataFrame(rows).to_parquet(Path(a.out).with_suffix(".parquet"))
    g = d[d.n_measured > 0].groupby(["channel", "split"]).agg(
        sites=("site", "nunique"), windows=("n_windows", "sum"), measured=("n_measured", "sum"), steps=("n_steps", "sum"))
    g["frac_measured_at_sites"] = (g.measured / g.steps).round(3)
    tab = g[["sites", "windows", "frac_measured_at_sites"]].unstack("split")
    tab = tab.reindex(CANDIDATES + CURRENT)
    tab.to_csv(a.out)
    pd.set_option("display.width", 200)
    print(tab.to_string())


if __name__ == "__main__":
    main()
