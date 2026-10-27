"""Truth statistics of every scored cell, for pooled R2 and the weighted absolute percentage error.

Score parts written before bench/run.py carried the truth sums hold per cell only n, MAE, RMSE and bias. Pooled MSE
follows from those (sum n rmse^2 / sum n); R2 and WAPE also need the truth's sum, sum of squares and sum of absolute
values. This script recomputes them from the same windows and masks the runner uses (`run.masks_for`,
`windows.load_window`): per cell (site, window, family, variable, kind, gap class), over the hidden measured values
(mask & measured) of each tower row, the same set `run.score` scores when a method fills the cell completely
(fill_rate = 1). For sum kinds, the statistics are of the block sums of the truth over those values.

Usage: python truth_stats.py --windows windows_val_list.json --out truth_stats.parquet [--families M2]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from metrics import GAP_CLASSES, block_sums, gap_class, gap_runs  # noqa: E402
from run import FLUXES, SUM_BLOCKS, masks_for  # noqa: E402
from windows import load_window  # noqa: E402


def stats(y: np.ndarray) -> dict:
    y = np.asarray(y, np.float64)
    return dict(n=int(y.size), sum_y=float(y.sum()), sum_y2=float((y ** 2).sum()), sum_abs_y=float(np.abs(y).sum()))


def window_stats(w, masks) -> list:
    rows = []
    for fam, mask in masks.items():
        cls = gap_class(gap_runs(mask))
        for i, k in enumerate(w.keys):
            if w.provenance[i] != "tower":
                continue
            target = mask[i] & w.measured[i]
            if not target.any():
                continue
            base = dict(site=w.site, window=str(w.start.date()), family=fam, variable=k)
            for c in range(-1, len(GAP_CLASSES)):
                sel = target if c == -1 else target & (cls[i] == c)
                if sel.any():
                    g = "all" if c == -1 else f"{GAP_CLASSES[c][0]}-{GAP_CLASSES[c][1]}"
                    rows.append(dict(base, kind="values", gap_class=g, **stats(w.values[i, sel])))
            if k in FLUXES:
                for bname, blk in SUM_BLOCKS.items():
                    _, ts = block_sums(np.zeros((1, w.values.shape[1])), w.values[i], target, blk)
                    if ts.size:
                        rows.append(dict(base, kind=f"sum-{bname}", gap_class="all", **stats(ts)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True, help="JSON list of {site, start, table} (or {'windows': [...]})")
    ap.add_argument("--families", nargs="+", default=["M2"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    specs = json.loads(Path(a.windows).read_text())
    if isinstance(specs, dict):
        specs = specs["windows"]
    rows, tables = [], {}
    for s in specs:
        if s["site"] not in tables:
            tables = {s["site"]: pd.read_parquet(s["table"])}      # one site table in memory at a time
        w = load_window(tables[s["site"]], s["site"], s["start"])
        rows += window_stats(w, masks_for(w, a.families))
        print(f"{s['site']} {s['start']}: {len(rows)} rows", flush=True)
    df = pd.DataFrame(rows).drop_duplicates(["site", "window", "family", "variable", "kind", "gap_class"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f"wrote {a.out}: {len(df)} cells")


if __name__ == "__main__":
    main()
