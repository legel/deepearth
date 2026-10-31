"""Data of report Figure 1: the final FluxFill on validation window US-YK2 2019-07-15 under one mask that hides part of
every reconstructed modality.

The mask is the benchmark's 30-day station outage of this window (M2, all tower channels, days 99 to 129), plus 20
days of the reanalysis rows and the Sentinel-2 acquisitions of 30 days, each placed outside the outage. FluxFill runs
as scored: checkpoint FLUXFILL_CKPT, 300 steps of site adaptation on the visible values and the tower's other years,
64 samples coupled in time by the temporal copula (FLUXFILL_COPULA). Hidden values never enter adaptation.

Output (bench/dashboard/report/fig1_all.npz): the window's values and keys, the mask, samples of every row, and for
every day with a Sentinel-2 acquisition the layer-normalised Clay embedding, whether it was hidden, and the imagery
head's mean and log-scale. Run in the fluxfill-clay environment on one GPU.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "bench"), str(HERE.parent / "baselines"), str(HERE.parent / "model")]

from method import FluxFill  # noqa: E402
from data import V_TAB  # noqa: E402
from run import masks_for  # noqa: E402
from windows import Window, load_window, lst_index  # noqa: E402

R = Path("/home/photon/4tb/fluxfill")
SITE, START = "US-YK2", "2019-07-15"
ERA_DAYS, IMG_DAYS = 20, 30


def free_span(busy, length, score):
    """The start day of a span of `length` days that avoids `busy` (day flags) and maximises `score` over the span."""
    ok = [d for d in range(0, 180 - length + 1) if not busy[d:d + length].any()]
    return max(ok, key=lambda d: score[d:d + length].sum())


def main():
    tab = pd.read_parquet(R / "towers" / "hh" / f"{SITE}.parquet")
    w = load_window(tab, SITE, START)
    m = masks_for(w, ["M2"])["M2-30d"].copy()
    idx = lst_index(tab)
    years = list(range(w.t[0].year, w.t[-1].year + 1))
    ctx = {"table": tab[~idx.year.isin(years)], "index": idx[~idx.year.isin(years)]}
    ff = FluxFill(os.environ["FLUXFILL_CKPT"], name="fluxfill", adapt_steps=300, adapt_lr=3e-4,
                  adapt_row_weight={"FCH4": 4.0}, copula=os.environ.get("FLUXFILL_COPULA"))

    busy = m.any(0).reshape(180, 48).any(1)
    busy = np.convolve(busy, np.ones(11), "same") > 0                     # 5 days apart from the outage
    rean = w.rows("reanalysis")
    e0 = free_span(busy, ERA_DAYS, -np.abs(np.arange(180) - 155.0))     # the free span nearest day 155
    m[rean, e0 * 48:(e0 + ERA_DAYS) * 48] = True
    busy[max(0, e0 - 5):e0 + ERA_DAYS + 5] = True

    visible = Window(w.site, w.start, w.t, w.keys, np.where(m, np.nan, w.values), w.measured & ~m, w.provenance, w.meta)
    ex = ff.example(visible, ctx)
    has = ex["img_obs"].copy()
    meta = ex["img_meta"]
    i0 = free_span(busy, IMG_DAYS, has - np.abs(np.arange(180) - 50.0) / 200)   # many acquisitions, near day 50
    img_hide = np.zeros(180, bool)
    img_hide[i0:i0 + IMG_DAYS] = has[i0:i0 + IMG_DAYS]
    ex["img_obs"] = has & ~img_hide
    ex["img_cls"] = np.where(img_hide[:, None], 0.0, ex["img_cls"]).astype(np.float32)

    ff.adapt(visible, ex, ctx)
    rows, pos = ff.window_rows(w)
    with torch.no_grad():
        z = ff.sample_marginal(ex)
        h, _ = ff.net(ff._batch(ex, 1))
        o = ff.net.head_img(h[0, V_TAB].float()).float()
    x = ff.stats.denorm(z.reshape(ff.n * V_TAB, -1).astype(np.float64), np.tile(np.arange(V_TAB), ff.n)).reshape(ff.n, V_TAB, -1)
    samples = np.full((ff.n,) + w.values.shape, np.nan, np.float32)
    samples[:, pos] = x[:, rows]

    # Clay embeddings of every acquisition day of the window (the hidden ones from the unmasked example), layer-normalised
    full = ff.example(w, ctx)
    with torch.no_grad():
        ln = ff.net.img_norm(torch.as_tensor(full["img_cls"], device=ff.device).float()).cpu().numpy()
    mu = o[:, :1024].cpu().numpy()
    log_s = o[:, 1024:].clamp(-7, 5).cpu().numpy() if o.shape[1] > 1024 else np.full_like(mu, np.nan)
    cos = [float(np.dot(mu[d], ln[d]) / (np.linalg.norm(mu[d]) * np.linalg.norm(ln[d]))) for d in np.flatnonzero(img_hide)]
    print(f"tower outage days {m[w.rows('tower')].any(0).nonzero()[0][[0, -1]] / 48}, reanalysis days {e0}-{e0 + ERA_DAYS}, "
          f"imagery days {i0}-{i0 + IMG_DAYS} ({img_hide.sum()} acquisition days hidden), cosine of hidden embeddings "
          f"{np.round(cos, 3).tolist()}", flush=True)
    np.savez_compressed(R / "bench" / "dashboard" / "report" / "fig1_all.npz", values=w.values.astype(np.float32),
                        measured=w.measured, keys=np.array(w.keys), mask=m, samples=samples, img_days=np.flatnonzero(has),
                        img_hidden=img_hide[has], img_ln=ln[has], img_mu=mu[has], img_log_s=log_s[has],
                        img_span=np.array([i0, i0 + IMG_DAYS]), era_span=np.array([e0, e0 + ERA_DAYS]), img_meta=meta[has])
    json.dump({"img_cos": cos}, open(R / "bench" / "dashboard" / "report" / "fig1_all.json", "w"))


if __name__ == "__main__":
    main()
