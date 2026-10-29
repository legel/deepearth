"""Methane validation windows: every usable, non-overlapping 180-day window with measured FCH4 at the validation
sites that measure FCH4 (validation sites only; never test sites).

Rule, fixed before any method is scored on these windows:
1. Sites: the validation sites of `data.load_split` (the frozen split) with any measured FCH4 half-hour.
2. Windows: the site's usable starts (`data.valid_starts`, validation rule), taken greedily from the first, each at
   least 180 days after the previous kept one (non-overlapping), kept when FCH4 is measured in at least 5 % of the
   window's 8,640 half-hours (432, nine days' worth).
3. Every kept window is scored; nothing is selected on scores. Each window records its FCH4 count, the FCH4 history
   (measured FCH4 half-hours in the site's other calendar years, the context a per-site method or the site memory
   sees) and the NEE history, so results can be split by "with" and "without FCH4 history".
Note: BR-SM1, NZ-Kop, US-ORv and US-YK1 are calibration sites of the baseline development set (windows_val_list.json,
rule 5: spreads and early stopping of SAITS and CSDI); scores of those imputers there are not independent.

Usage: python methane_windows.py [--out /home/photon/4tb/fluxfill/bench/val/windows_val_fch4.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import (STEPS_PER_DAY, T_WIN, VARS, WINDOW_DAYS, load_records, load_split, sites_table,  # noqa: E402
                  table_dir, valid_starts)

MIN_FCH4_FRAC = 0.05
RULE = __doc__.split("Usage:")[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/photon/4tb/fluxfill/bench/val/windows_val_fch4.json")
    a = ap.parse_args()
    sites = sorted(p.stem for p in table_dir().glob("*.parquet"))
    split, name = load_split(sites)
    meta = sites_table()
    n_ch4 = pd.to_numeric(meta["fch4_measured_hh"], errors="coerce").fillna(0)
    val = [s for s, v in split.items() if v == "val" and s in meta.index and n_ch4.loc[s] > 0]
    recs = load_records(val, verbose=False)
    ich4, inee = VARS.index("FCH4"), VARS.index("NEE")
    out = []
    for s in val:
        r = recs[s]
        t = r.t0 + pd.to_timedelta(np.arange(r.values.shape[1]) * 30, unit="m")
        last = -WINDOW_DAYS
        for d in valid_starts(r, exclude_heldout=False):
            if d < last + WINDOW_DAYS:
                continue
            seg = r.values[:, d * STEPS_PER_DAY:d * STEPS_PER_DAY + T_WIN]
            n_ch4 = int(np.isfinite(seg[ich4]).sum())
            if n_ch4 < MIN_FCH4_FRAC * T_WIN:
                continue
            start = r.t0 + pd.Timedelta(days=int(d))
            years = set(range(start.year, (start + pd.Timedelta(days=WINDOW_DAYS - 1)).year + 1))
            other = ~np.isin(t.year, list(years))
            out.append({"site": s, "start": str(start.date()), "table": str(table_dir() / f"{s}.parquet"),
                        "n_fch4": n_ch4, "fch4_history_hh": int(np.isfinite(r.values[ich4, other]).sum()),
                        "nee_history_hh": int(np.isfinite(r.values[inee, other]).sum())})
            last = int(d)
    doc = {"split": name, "rule": RULE, "sites": val, "windows": out}
    Path(a.out).write_text(json.dumps(doc, indent=1))
    df = pd.DataFrame(out)
    print(df.groupby("site").agg(windows=("start", "size"), n_fch4=("n_fch4", "sum"),
                                 with_history=("fch4_history_hh", lambda x: int((x > 0).sum()))))
    print(f"{len(out)} windows at {df.site.nunique()} sites -> {a.out}")


if __name__ == "__main__":
    main()
