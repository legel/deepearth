"""Diagnose v1 joint samples: bias and interval coverage of the Student-t marginals, of the flow head with one round
(every hidden token drawn from the first encoding: no feedback) and with K rounds (masked generation with feedback),
on the same hidden values of the same windows and masks.

Writes one row per (window, mask, variable, sampler): n, bias, MAE of the sample mean, CRPS, 50/80/95 % coverage, and
the ratio of the sample spread to the RMSE of the sample mean (1 for a calibrated spread).

Usage: python diagnose_v1.py --ckpt CK --windows list.json --out rows.parquet [--rounds 1 16] [--samples 32]
       [--masks 7 30] [--device cuda] [--max-windows 8]
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

from masks import contiguous_block, seed_for  # noqa: E402
from method import FluxFill  # noqa: E402
from metrics import coverage_samples, crps_samples  # noqa: E402
from windows import Window, load_window, lst_index  # noqa: E402

VARS_OUT = ("NEE", "LE", "H", "TA", "SW_IN")


def main():
    torch.set_num_threads(2)
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--variants", nargs="*", default=[], help="rounds:ode_steps pairs, e.g. 1:24 16:24 1:48 (overrides --rounds)")
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--masks", type=int, nargs="+", default=[7, 30])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-windows", type=int, default=8)
    ap.add_argument("--imagery-sites", default="", help="comma list: Clay imagery only at these sites")
    ap.add_argument("--resume", action="store_true", help="keep the rows already in --out and skip their windows")
    a = ap.parse_args()
    isites = a.imagery_sites.split(",") if a.imagery_sites else None
    specs = json.loads(Path(a.windows).read_text())[: a.max_windows]
    samplers = {"student-t": FluxFill(a.ckpt, mode="marginal", n_samples=a.samples, device=a.device, imagery_sites=isites)}
    variants = [tuple(map(int, v.split(":"))) for v in a.variants] or [(r, 24) for r in a.rounds]
    for r, fs in variants:
        samplers[f"flow-{r}r" + ("" if fs == 24 else f"-{fs}s")] = FluxFill(a.ckpt, mode="joint", n_samples=a.samples,
                                                                            steps=r, flow_steps=fs, chunk=8, device=a.device, imagery_sites=isites)
    rows, done = [], set()
    if a.resume and Path(a.out).exists():
        prev = pd.read_parquet(a.out)
        rows = prev.to_dict("records")
        done = set(map(tuple, prev[["site", "window"]].drop_duplicates().to_numpy()))
    from gpu_pause import wait_if_paused
    for spec in specs:
        if (spec["site"], spec["start"]) in done:
            continue
        wait_if_paused(f"diagnose_v1 {spec['site']} {spec['start']}")
        tab = pd.read_parquet(spec["table"])
        w = load_window(tab, spec["site"], spec["start"])
        idx = lst_index(tab)
        years = list(range(w.t[0].year, w.t[-1].year + 1))
        ctx = {"table": tab[~idx.year.isin(years)], "index": idx[~idx.year.isin(years)]}
        V, T = w.values.shape
        for d in a.masks:
            mk = contiguous_block(w.rows("tower"), V, T, d, np.random.default_rng(seed_for(w.site, spec["start"], "M2", d)))
            vis = Window(w.site, w.start, w.t, w.keys, np.where(mk, np.nan, w.values), w.measured & ~mk, w.provenance)
            for name, m in samplers.items():
                s = m.predict(vis, mk, ctx)
                for k in VARS_OUT:
                    if k not in w.keys:
                        continue
                    i = w.keys.index(k)
                    sel = mk[i] & w.measured[i]
                    if not sel.any():
                        continue
                    y, ss = w.values[i, sel], s[:, i, sel]
                    mu = ss.mean(0)
                    cov = coverage_samples(ss, y, (0.5, 0.8, 0.95))
                    rmse = float(np.sqrt(((mu - y) ** 2).mean()))
                    rows.append(dict(site=w.site, window=spec["start"], mask=f"M2-{d}d", variable=k, sampler=name,
                                     n=int(sel.sum()), bias=float((mu - y).mean()), mae=float(np.abs(mu - y).mean()),
                                     crps=float(crps_samples(ss, y).mean()), cov50=cov[0.5], cov80=cov[0.8],
                                     cov95=cov[0.95], spread_ratio=float(ss.std(0).mean() / rmse) if rmse > 0 else np.nan))
            print(spec["site"], spec["start"], d, flush=True)
        pd.DataFrame(rows).to_parquet(a.out, index=False)
    print(f"wrote {a.out}: {len(rows)} rows")


if __name__ == "__main__":
    main()
