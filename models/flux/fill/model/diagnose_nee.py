"""Diagnose the NEE bias of a FluxFill checkpoint on validation windows (CPU), against climatology.

For each window (one per validation site, chosen as in evaluate.py) and each mask (M2 all-tower blocks of 7, 30 and
180 days, and an NEE-only 30-day block at the M2 position, which keeps tower SW_IN visible), every hidden, measured NEE
half-hour is written with: FluxFill Student-t location, scale and degrees of freedom; FluxFill sample mean (64 draws);
climatology sample mean and median; the site-memory mean at that half-hour; the truth; tower SW_IN (truth, visible or
not), ERA5 SW_IN, hour, month, IGBP, mask and whether tower SW_IN was visible to the model.

Usage: python diagnose_nee.py --ckpt CK --out rows.parquet [--sites N]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "bench"))
sys.path.insert(0, str(HERE.parent / "baselines"))

from data import V_TAB, VARS, table_dir  # noqa: E402
from evaluate import val_windows  # noqa: E402
from masks import contiguous_block, seed_for  # noqa: E402
from metrics import crps_samples, crps_student_t  # noqa: E402
from method import FluxFill  # noqa: E402
from net import sample_marginal_params, sample_student_t  # noqa: E402
from windows import Window, load_window, lst_index  # noqa: E402
import climatology  # noqa: E402


def crps_or_nan(samples, y):
    ok = np.isfinite(samples).all(0)
    out = np.full(y.shape, np.nan)
    if ok.any():
        out[ok] = crps_samples(samples[:, ok], y[ok])
    return out


def main():
    torch.set_num_threads(2)
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sites", type=int, default=12)
    ap.add_argument("--windows", default="", help="JSON list of {site, start, table}: use exactly these windows")
    ap.add_argument("--m2", type=int, nargs="+", default=[7, 30, 180], help="M2 all-tower block lengths (days)")
    ap.add_argument("--imagery-sites", default="", help="comma list: Clay imagery only at these sites")
    a = ap.parse_args()
    isites = a.imagery_sites.split(",") if a.imagery_sites else None
    sp = pd.read_csv("/home/photon/4tb/fluxfill/towers/split.csv").set_index("site_id")
    if a.windows:
        import json as _json
        specs = _json.loads(Path(a.windows).read_text())
        a.sites = len(specs)
    else:
        specs, split = val_windows(1)
        assert split == "frozen"
    specs = [s for s in specs if sp.loc[s["site"], "split"] == "val"]
    # spread the chosen sites over IGBP classes: round-robin over classes
    by = {}
    for s in specs:
        by.setdefault(sp.loc[s["site"], "igbp"], []).append(s)
    chosen = []
    if a.windows:
        by = {}
        chosen = specs
    while len(chosen) < a.sites and any(by.values()):
        for k in sorted(by):
            if by[k] and len(chosen) < a.sites:
                chosen.append(by[k].pop(0))
    m = FluxFill(a.ckpt, mode="marginal", n_samples=64, device="cpu", imagery_sites=isites)
    iN, iS, iE = VARS.index("NEE"), VARS.index("SW_IN"), VARS.index("ERA5_SW_IN")
    rows = []
    for spec in chosen:
        site, start = spec["site"], spec["start"]
        tab = pd.read_parquet(spec["table"])
        w = load_window(tab, site, start)
        if "NEE" not in w.keys:
            continue
        idx = lst_index(tab)
        years = list(range(w.t[0].year, w.t[-1].year + 1))
        ctx = {"table": tab[~idx.year.isin(years)], "index": idx[~idx.year.isin(years)]}
        V, T = w.values.shape
        tower = w.rows("tower")
        kN = w.keys.index("NEE")
        masks = {}
        for d in a.m2:
            masks[f"all-{d}d"] = contiguous_block(tower, V, T, d, np.random.default_rng(seed_for(site, start, "M2", d)))
        masks["nee-only-30d"] = contiguous_block([kN], V, T, 30, np.random.default_rng(seed_for(site, start, "M2", 30)))
        nee_ctx = ctx["table"]["NEE"].notna() & (ctx["table"]["NEE_flag"] == 0)
        hist_days = float(nee_ctx.sum()) / 48.0                    # measured NEE in the other years, in days
        clim = climatology.predict(pd.Series(tab["NEE"].to_numpy(np.float64), index=idx),
                                   pd.Series(tab["NEE_flag"].to_numpy() == 0, index=idx) & tab["NEE"].notna().to_numpy(),
                                   w.t, n_samples=64, seed=0)
        sw_true = w.values[w.keys.index("SW_IN")] if "SW_IN" in w.keys else np.full(T, np.nan)
        era = w.values[w.keys.index("ERA5_SW_IN")] if "ERA5_SW_IN" in w.keys else np.full(T, np.nan)
        for name, mk in masks.items():
            vis = Window(w.site, w.start, w.t, w.keys, np.where(mk, np.nan, w.values), w.measured & ~mk, w.provenance)
            ex = m.example(vis, ctx)
            b = m._batch(ex, 1)
            with torch.no_grad():
                _, params = m.net(b)
            sd = m.stats.std[iN]
            mu = m.stats.mean[iN]
            if len(params) == 3:
                loc, scale, df = (x[0, iN].reshape(-1).numpy().astype(np.float64) for x in params)
                samp = sample_student_t(torch.tensor(loc), torch.tensor(scale), torch.tensor(df), 64).numpy()
            else:                       # mixture head: point = sample median; CRPS from the draws
                pn = tuple(x[0, iN].reshape(-1, *x.shape[4:]) if x.dim() == 5 else x[0, iN].reshape(-1) for x in params)
                samp = sample_marginal_params(pn, 64).numpy().astype(np.float64)
                loc = np.median(samp, 0)
                scale = np.full(loc.shape, np.nan); df = np.full(loc.shape, np.nan)
            sel = mk[kN] & w.measured[kN]
            sw_vis = ("SW_IN" in w.keys) and not mk[w.keys.index("SW_IN")].any()
            j = np.flatnonzero(sel)
            rows.append(pd.DataFrame(dict(
                site=site, window=start, igbp=sp.loc[site, "igbp"], mask=name, sw_visible=sw_vis, t=w.t[j],
                hour=w.t[j].hour + w.t[j].minute / 60, month=w.t[j].month, truth=w.values[kN, j],
                ff_loc=loc[j] * sd + mu, ff_scale=scale[j] * sd, ff_df=df[j], ff_mean=samp[:, j].mean(0) * sd + mu,
                mem_mean=np.where(ex["mem_h"][iN].reshape(-1)[j], ex["mem_m"][iN].reshape(-1)[j] * sd + mu, np.nan),
                clim_mean=np.nanmean(clim[:, j], 0), clim_median=np.nanmedian(clim[:, j], 0),
                clim_crps=crps_or_nan(clim[:, j], w.values[kN, j]),
                ff_crps=(crps_student_t(loc[j] * sd + mu, scale[j] * sd, df[j], w.values[kN, j]) if len(params) == 3
                         else crps_samples(samp[:, j] * sd + mu, w.values[kN, j])), hist_days=hist_days,
                sw_in=sw_true[j], era5_sw_in=era[j])))
        print(site, start, sp.loc[site, "igbp"], flush=True)
    df = pd.concat(rows, ignore_index=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f"{len(df)} rows, {df.site.nunique()} sites")


if __name__ == "__main__":
    main()
