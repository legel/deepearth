"""Short-gap diagnosis: FluxFill against Chronos-2 and climatology per hidden flux half-hour, by distance from the gap
edge.

For each window of a list and each M2 all-tower block (1 and 7 days by default, benchmark seeds), every hidden,
measured NEE, LE and H value is written with: the distance (half-hours) to the nearest visible step on either side of
the block, the offset from the block start, the truth, and for each method the sample mean and the per-value CRPS from
64 samples. Chronos-2 forecasts each block forward from the history before it (baselines/chronos2.py); its flux
forecast uses only the flux history and the ERA5 covariates, so it is run with only the flux rows hidden, which
gives the same flux forecast as the all-tower mask without forecasting the meteorology.

Usage: python diagnose_short.py --ckpt CK --windows list.json --out rows.parquet [--days 1 7] [--imagery-sites ...]
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
sys.path.insert(0, str(HERE.parent / "baselines"))

from masks import contiguous_block, seed_for  # noqa: E402
from method import FluxFill  # noqa: E402
from metrics import crps_samples  # noqa: E402
from windows import Window, load_window, lst_index  # noqa: E402
import run as bench  # noqa: E402

FLUXES = ("NEE", "LE", "H")


def main():
    torch.set_num_threads(2)
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", type=int, nargs="+", default=[1, 7])
    ap.add_argument("--imagery-sites", default="")
    ap.add_argument("--chronos-from", default="", help="reuse Chronos-2 columns of an earlier output (same windows)")
    a = ap.parse_args()
    specs = json.loads(Path(a.windows).read_text())
    specs = list({(s["site"], s["start"]): s for s in specs}.values())        # distinct windows
    ff = FluxFill(a.ckpt, mode="marginal", n_samples=64, device="cpu",
                  imagery_sites=a.imagery_sites.split(",") if a.imagery_sites else None)
    if a.chronos_from:
        ch = None
        prev = pd.read_parquet(a.chronos_from)[["site", "window", "mask", "variable", "j", "ch_mean", "ch_crps"]]
    else:
        from chronos2 import Chronos2
        ch = Chronos2(n_samples=64, device="cpu", dtype="float32")
    cl = bench.Climatology(64)
    rows = []
    for spec in specs:
        tab = pd.read_parquet(spec["table"])
        w = load_window(tab, spec["site"], spec["start"])
        idx = lst_index(tab)
        years = list(range(w.t[0].year, w.t[-1].year + 1))
        ctx = {"table": tab[~idx.year.isin(years)], "index": idx[~idx.year.isin(years)]}
        V, T = w.values.shape
        tower = w.rows("tower")
        flux = [w.keys.index(k) for k in FLUXES if k in w.keys]
        for d in a.days:
            mk = contiguous_block(tower, V, T, d, np.random.default_rng(seed_for(w.site, spec["start"], "M2", d)))
            vis = Window(w.site, w.start, w.t, w.keys, np.where(mk, np.nan, w.values), w.measured & ~mk, w.provenance)
            mflux = np.zeros_like(mk)
            mflux[flux] = mk[flux]
            visf = Window(w.site, w.start, w.t, w.keys, np.where(mflux, np.nan, w.values), w.measured & ~mflux,
                          w.provenance)
            s_ff = ff.predict(vis, mk, ctx)
            s_ch = ch.predict(visf, mflux, ctx) if ch is not None else np.full((64,) + mk.shape, np.nan)
            s_cl = cl.predict(vis, mk, ctx)
            hid = mk[tower].any(0)
            e = np.flatnonzero(np.diff(np.concatenate(([0], hid.astype(np.int8), [0]))))
            for a0, b0 in zip(e[::2], e[1::2]):
                for i in flux:
                    j = a0 + np.flatnonzero(mk[i, a0:b0] & w.measured[i, a0:b0])
                    if j.size == 0:
                        continue
                    y = w.values[i, j]
                    r = dict(site=w.site, window=spec["start"], mask=f"M2-{d}d", variable=w.keys[i], j=j,
                             from_start=j - a0 + 1, to_end=b0 - j, edge=np.minimum(j - a0 + 1, b0 - j),
                             hour=w.t[j].hour + w.t[j].minute / 60, truth=y, sw_in_era5=w.values[w.keys.index("ERA5_SW_IN"), j]
                             if "ERA5_SW_IN" in w.keys else np.nan)
                    for name, s in (("ff", s_ff), ("ch", s_ch), ("cl", s_cl)):
                        ss = s[:, i, j]
                        ok = np.isfinite(ss).all(0)
                        c = np.full(j.size, np.nan)
                        if ok.any():
                            c[ok] = crps_samples(ss[:, ok], y[ok])
                        r[f"{name}_mean"] = np.where(ok, np.nanmean(np.where(np.isfinite(ss), ss, np.nan), 0), np.nan)
                        r[f"{name}_crps"] = c
                    rows.append(pd.DataFrame(r))
            print(w.site, spec["start"], d, flush=True)
        out = pd.concat(rows, ignore_index=True)
        if ch is None:
            out = out.drop(columns=["ch_mean", "ch_crps"]).merge(prev, on=["site", "window", "mask", "variable", "j"],
                                                                  how="left")
        out.to_parquet(a.out, index=False)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
