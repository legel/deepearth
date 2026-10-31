"""Data of report Figure 4: the final FluxFill on ten test windows, one hidden station outage each (M2, all tower
channels), at four gap lengths.

Selection rule, fixed before any fill is seen: among the release-0.2 test windows, for each slot of the plan below,
the window with the largest measured share of hidden half-hours in NEE, LE and H (each at least 0.5), at a tower not
yet chosen, preferring a vegetation class not yet chosen; the first four slots also require measured FCH4 (share at
least 0.3). The masks are the benchmark's (run.masks_for, release seeds). FluxFill runs as scored: FLUXFILL_CKPT, 300
steps of site adaptation, 64 samples coupled in time by the temporal copula (FLUXFILL_COPULA).

Output: bench/dashboard/report/fig4/<site>_<start>_<mask>.npz with the flux rows' values, the hidden span, and the
half-hourly and daily-mean mean and 95% interval of the samples. Run in the fluxfill-clay environment on one GPU.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "bench"), str(HERE.parent / "baselines"), str(HERE.parent / "model")]

from run import fluxfill_method, masks_for  # noqa: E402
from windows import Window, load_window, lst_index  # noqa: E402

R = Path("/home/photon/4tb/fluxfill")
OUT = R / "bench" / "dashboard" / "report" / "fig4"
FLUX = ("NEE", "LE", "H", "FCH4")
PLAN = ["M2-90d", "M2-30d", "M2-7d", "M2-1d", "M2-90d", "M2-30d", "M2-30d", "M2-7d", "M2-7d", "M2-1d"]


def candidates():
    wins = json.load(open(R / "bench" / "test_r02" / "windows_test.json"))
    sites = {x["id"]: x for x in json.load(open(R / "dashboard_build" / "site" / "data" / "sites.json"))["sites"]}
    tabs, rows = {}, []
    for e in wins:
        if e["site"] not in tabs:
            tabs = {e["site"]: pd.read_parquet(e["table"])}               # one tower in memory at a time
        w = load_window(tabs[e["site"]], e["site"], e["start"])
        ms = masks_for(w, ["M2"])
        for name in set(PLAN):
            m = ms.get(name)
            if m is None:
                continue
            share = {}
            for k in FLUX:
                if k in w.keys:
                    i = w.keys.index(k)
                    share[k] = float((w.measured[i] & m[i]).sum() / max(m[i].sum(), 1))
            rows.append({"site": e["site"], "start": e["start"], "table": e["table"], "mask": name,
                         "igbp": sites.get(e["site"], {}).get("igbp", ""), **{f"s_{k}": share.get(k, 0.0) for k in FLUX}})
    return pd.DataFrame(rows)


def select(c):
    c = c[(c.s_NEE >= 0.5) & (c.s_LE >= 0.5) & (c.s_H >= 0.5)].assign(score=lambda d: (d.s_NEE + d.s_LE + d.s_H) / 3)
    pick, towers, classes = [], set(), set()
    for j, name in enumerate(PLAN):
        d = c[(c["mask"] == name) & ~c.site.isin(towers)]
        if j < 4:
            d = d[d.s_FCH4 >= 0.3]
        d = d.sort_values(["score", "site", "start"], ascending=[False, True, True])
        fresh = d[~d.igbp.isin(classes)]
        r = (fresh if len(fresh) else d).iloc[0]
        pick.append(r); towers.add(r.site); classes.add(r.igbp)
    return pd.DataFrame(pick)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sel = select(candidates())
    sel.to_csv(OUT / "selection.csv", index=False)
    print(sel[["site", "start", "mask", "igbp", "score", "s_FCH4"]].to_string(), flush=True)
    ff = fluxfill_method()
    for r in sel.itertuples():
        tab = pd.read_parquet(r.table)
        w = load_window(tab, r.site, r.start)
        m = masks_for(w, ["M2"])[r.mask]
        idx = lst_index(tab)
        years = list(range(w.t[0].year, w.t[-1].year + 1))
        ctx = {"table": tab[~idx.year.isin(years)], "index": idx[~idx.year.isin(years)]}
        visible = Window(w.site, w.start, w.t, w.keys, np.where(m, np.nan, w.values), w.measured & ~m, w.provenance, w.meta)
        s = ff.predict(visible, m, ctx)
        keys = [k for k in FLUX if k in w.keys]
        ii = [w.keys.index(k) for k in keys]
        x, y = s[:, ii], np.where(w.measured[ii], w.values[ii], np.nan)
        a, b = np.flatnonzero(m[ii[0]])[[0, -1]] + [0, 1]
        off = a % 48
        n = (x.shape[-1] - off) // 48
        day = x[..., off:off + n * 48].reshape(*x.shape[:2], n, 48).mean(-1)          # daily means of each sample
        with np.errstate(all="ignore"):
            yd = np.nanmean(y[:, off:off + n * 48].reshape(len(ii), n, 48), -1)
        np.savez_compressed(OUT / f"{r.site}_{r.start}_{r.mask}.npz", keys=np.array(keys), y=y.astype(np.float32),
                            span=np.array([a, b]), off=off, igbp=r.igbp,
                            hh=np.stack([x.mean(0), *np.quantile(x, [0.025, 0.975], 0)]).astype(np.float32),
                            yd=yd.astype(np.float32),
                            dd=np.stack([day.mean(0), *np.quantile(day, [0.025, 0.975], 0)]).astype(np.float32))
        print(f"{r.site} {r.start} {r.mask}: done", flush=True)


if __name__ == "__main__":
    main()
