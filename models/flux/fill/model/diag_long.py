"""Long-gap NEE against climatology: which part of the CRPS gap is location and which is shape.

For each window of a list and the M2 all-tower blocks of 90 and 180 days (benchmark seeds), every hidden, measured NEE
value where climatology predicts gets: the truth, the site-memory mean, R0's samples (marginal, `--samples`), and
climatology's 64 samples. Scores (CRPS from samples, fair estimator):
- ff: the model as is;
- ff@mem: the model's samples shifted so their median is the memory mean (location from memory, shape from the model);
- ff@clim_median: shifted so their median is climatology's median;
- clim: climatology as is;
- clim@ff_median: climatology's samples shifted so their median is the model's median (location from the model, shape
  from climatology).
ff vs ff@clim_median is the location effect with the model's shape; ff@clim_median vs clim is the shape effect at the
same location; clim@ff_median vs ff is the shape effect at the model's location.

Usage: python diag_long.py --ckpt CK --windows list.json --out rows.parquet [--samples 256] [--imagery-sites ...]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "bench"))

from data import VARS  # noqa: E402
from masks import contiguous_block, seed_for  # noqa: E402
from method import FluxFill  # noqa: E402
from metrics import crps_samples  # noqa: E402
from windows import Window, load_window, lst_index  # noqa: E402
import run as bench  # noqa: E402


def main():
    torch.set_num_threads(2)
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--days", type=int, nargs="+", default=[90, 180])
    ap.add_argument("--imagery-sites", default="")
    a = ap.parse_args()
    specs = list({(s["site"], s["start"]): s for s in json.loads(Path(a.windows).read_text())}.values())
    m = FluxFill(a.ckpt, mode="marginal", n_samples=a.samples, device="cpu",
                 imagery_sites=a.imagery_sites.split(",") if a.imagery_sites else None)
    cl = bench.Climatology(64)
    iN = VARS.index("NEE")
    rows = []
    for spec in specs:
        tab = pd.read_parquet(spec["table"])
        w = load_window(tab, spec["site"], spec["start"])
        if "NEE" not in w.keys:
            continue
        idx = lst_index(tab)
        years = list(range(w.t[0].year, w.t[-1].year + 1))
        ctx = {"table": tab[~idx.year.isin(years)], "index": idx[~idx.year.isin(years)]}
        V, T = w.values.shape
        k = w.keys.index("NEE")
        for d in a.days:
            mk = contiguous_block(w.rows("tower"), V, T, d, np.random.default_rng(seed_for(w.site, spec["start"], "M2", d)))
            vis = Window(w.site, w.start, w.t, w.keys, np.where(mk, np.nan, w.values), w.measured & ~mk, w.provenance)
            s_ff = m.predict(vis, mk, ctx)[:, k]
            ex = m.example(vis, ctx)
            mem = np.where(ex["mem_h"][iN].reshape(-1), ex["mem_m"][iN].reshape(-1) * m.stats.std[iN] + m.stats.mean[iN], np.nan)
            s_cl = cl.predict(vis, mk, ctx)[:, k]
            sel = mk[k] & w.measured[k] & np.isfinite(s_cl).all(0) & np.isfinite(mem)
            j = np.flatnonzero(sel)
            if j.size == 0:
                continue
            y, F, C = w.values[k, j], s_ff[:, j], s_cl[:, j]
            fmed, cmed = np.median(F, 0), np.median(C, 0)
            r = dict(site=w.site, window=spec["start"], mask=f"M2-{d}d", j=j, truth=y, mem=mem[j], ff_mean=F.mean(0),
                     ff_median=fmed, clim_mean=C.mean(0), clim_median=cmed,
                     sw=w.values[w.keys.index("ERA5_SW_IN"), j] if "ERA5_SW_IN" in w.keys else np.nan,
                     crps_ff=crps_samples(F, y), crps_ff_at_mem=crps_samples(F - fmed + mem[j], y),
                     crps_ff_at_clim_median=crps_samples(F - fmed + cmed, y), crps_clim=crps_samples(C, y),
                     crps_clim_at_ff_median=crps_samples(C - cmed + fmed, y),
                     ff_iqr=np.subtract(*np.quantile(F, [0.75, 0.25], 0)), clim_iqr=np.subtract(*np.quantile(C, [0.75, 0.25], 0)))
            rows.append(pd.DataFrame(r))
        print(w.site, spec["start"], flush=True)
    pd.concat(rows, ignore_index=True).to_parquet(a.out, index=False)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
