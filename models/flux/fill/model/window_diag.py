"""Per-window distribution diagnostics: predicted against real distributions, saved for the dashboard.

For each window of a list and each mask (M2 all-tower blocks, benchmark seeds), FluxFill's marginal predictive
distribution of every hidden, measured value is compared with the truth:
- exact PIT (from the predictive CDF, `net.mixture_cdf`) and exact 50 / 80 / 95 % coverage;
- CRPS (fair estimator from samples) and MAE of the median;
- spread against error (mean predictive standard deviation / RMSE of the mean);
- tails: CRPS on the 5 % of values with the largest |truth|;
- sums: daily sums of the hidden steps, coverage and bias from the marginal samples (independent half-hours, labelled).
Arrays per window (npz, float16): for every tower variable the truth, measured flag, mask, predictive mean, the 2.5,
10, 25, 50, 75, 90 and 97.5 % quantiles (from `--samples` draws) and the exact PIT, on the window's 8,640 steps.
Summary per window and variable (JSON), and a ranking table of windows by CRPS skill against climatology when the
benchmark rows are given.

Usage: python window_diag.py --ckpt CK --windows list.json --out DIR [--days 7 30 180] [--samples 256]
       [--imagery-sites ...] [--device cpu]
       python window_diag.py --rank DIR --bench merged.parquet   (ranking.json in DIR: CRPS skill per window and flux
       against climatology and the best other method, worst first)
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

QS = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
FLUXES = ("NEE", "LE", "H", "FCH4")


def summarise(y, s, pit):
    """Scores of one variable over its hidden, measured values. y (n,), s (S, n) samples, pit (n,)."""
    mu = s.mean(0)
    rmse = float(np.sqrt(((mu - y) ** 2).mean()))
    crps = crps_samples(s, y)
    tail = np.abs(y) >= np.quantile(np.abs(y), 0.95) if y.size >= 20 else np.zeros(y.size, bool)
    return {"n": int(y.size), "crps": float(crps.mean()), "mae_median": float(np.abs(np.median(s, 0) - y).mean()),
            "bias": float((mu - y).mean()), "spread_over_rmse": float(s.std(0).mean() / rmse) if rmse > 0 else None,
            **{f"cov{int(q * 100)}": float(np.mean((pit >= (1 - q) / 2) & (pit <= (1 + q) / 2))) for q in (0.5, 0.8, 0.95)},
            "pit_hist10": np.histogram(pit, bins=10, range=(0, 1))[0].tolist(),
            "crps_tail5": float(crps[tail].mean()) if tail.any() else None}


def rank(diag_dir: Path, bench: Path):
    """CRPS skill of every (window, mask, flux) in `diag_dir` against climatology and the best other method of the
    benchmark rows (values, all gap classes, R1), with the window's calibration summary; worst skill first."""
    b = pd.read_parquet(bench)
    b = b[(b.kind == "values") & (b.gap_class == "all") & (b.regime == "R1") & ~b.method.str.startswith("fluxfill")]
    b = b.set_index(["site", "window", "family", "variable", "method"]).crps.sort_index()
    rows = []
    for f in sorted(diag_dir.glob("*.json")):
        if f.name in ("index.json", "ranking.json"):
            continue
        j = json.loads(f.read_text())
        for k, v in j["variables"].items():
            if k not in FLUXES:
                continue
            try:
                ref = b.loc[(j["site"], j["window"], j["mask"], k)]
            except KeyError:
                continue
            clim = ref.get("climatology", np.nan)
            other = ref.drop("climatology", errors="ignore")
            h = np.asarray(v["pit_hist10"], float)
            rows.append({"site": j["site"], "window": j["window"], "mask": j["mask"], "variable": k, "n": v["n"],
                         "crps": v["crps"], "crps_climatology": float(clim),
                         "skill_vs_climatology": float(1 - v["crps"] / clim) if clim > 0 else None,
                         "best_other": other.idxmin() if other.size else None,
                         "crps_best_other": float(other.min()) if other.size else None,
                         "cov95": v["cov95"], "spread_over_rmse": v["spread_over_rmse"],
                         "bias_over_crps": v["bias"] / v["crps"] if v["crps"] > 0 else None,
                         "pit_outer_frac": float((h[0] + h[-1]) / h.sum()) if h.sum() else None,
                         "daily_sum_cov95_marginal": v.get("daily_sum_cov95_marginal")})
    rows.sort(key=lambda r: (r["skill_vs_climatology"] is None, r["skill_vs_climatology"] or 0))
    (diag_dir / "ranking.json").write_text(json.dumps(rows, indent=1))
    return rows


def main():
    torch.set_num_threads(2)
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", default="")
    ap.add_argument("--bench", default="")
    ap.add_argument("--ckpt")
    ap.add_argument("--windows")
    ap.add_argument("--out")
    ap.add_argument("--days", type=int, nargs="+", default=[7, 30, 180])
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--imagery-sites", default="")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.rank:
        rows = rank(Path(a.rank), Path(a.bench))
        print(pd.DataFrame(rows).to_string(max_rows=60, float_format=lambda x: f"{x:.3g}"))
        return
    specs = list({(s["site"], s["start"]): s for s in json.loads(Path(a.windows).read_text())}.values())
    m = FluxFill(a.ckpt, mode="marginal", n_samples=a.samples, device=a.device,
                 imagery_sites=a.imagery_sites.split(",") if a.imagery_sites else None)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    index = []
    for spec in specs:
        tab = pd.read_parquet(spec["table"])
        w = load_window(tab, spec["site"], spec["start"])
        idx = lst_index(tab)
        years = list(range(w.t[0].year, w.t[-1].year + 1))
        ctx = {"table": tab[~idx.year.isin(years)], "index": idx[~idx.year.isin(years)]}
        V, T = w.values.shape
        tower = w.rows("tower")
        for d in a.days:
            mk = contiguous_block(tower, V, T, d, np.random.default_rng(seed_for(w.site, spec["start"], "M2", d)))
            vis = Window(w.site, w.start, w.t, w.keys, np.where(mk, np.nan, w.values), w.measured & ~mk, w.provenance)
            s = m.predict(vis, mk, ctx)
            pit_all = m.pit(w.values)
            arrays, summ = {}, {}
            for i in tower:
                k = w.keys[i]
                sel = mk[i] & w.measured[i]
                q = np.quantile(s[:, i], QS, axis=0)
                arrays[f"{k}/truth"] = np.where(w.measured[i], w.values[i], np.nan).astype(np.float16)
                arrays[f"{k}/mask"] = mk[i]
                arrays[f"{k}/mean"] = s[:, i].mean(0).astype(np.float16)
                for qq, row in zip(QS, q):
                    arrays[f"{k}/q{qq * 1000:04.0f}"] = row.astype(np.float16)
                arrays[f"{k}/pit"] = np.where(sel, pit_all[i], np.nan).astype(np.float16)
                if sel.sum() >= 5:
                    summ[k] = summarise(w.values[i, sel], s[:, i, sel], pit_all[i, sel])
                    if k in FLUXES:                                   # daily sums of the hidden steps (marginal)
                        days_ = np.flatnonzero(sel.reshape(-1, 48).any(1))
                        tr = np.array([np.where(sel[j * 48:(j + 1) * 48], w.values[i, j * 48:(j + 1) * 48], 0).sum() for j in days_])
                        sm = np.stack([np.where(sel[j * 48:(j + 1) * 48], s[:, i, j * 48:(j + 1) * 48], 0).sum(1) for j in days_], 1)
                        lo, hi = np.quantile(sm, [0.025, 0.975], 0)
                        summ[k]["daily_sum_cov95_marginal"] = float(np.mean((tr >= lo) & (tr <= hi)))
                        summ[k]["daily_sum_bias"] = float((sm.mean(0) - tr).mean())
            name = f"{w.site}_{spec['start']}_M2-{d}d"
            np.savez_compressed(out / f"{name}.npz", **arrays)
            (out / f"{name}.json").write_text(json.dumps({"site": w.site, "window": spec["start"], "mask": f"M2-{d}d",
                                                           "checkpoint": a.ckpt, "variables": summ}, indent=1))
            index.append({"file": name, "site": w.site, "window": spec["start"], "mask": f"M2-{d}d",
                          **{f"crps_{k}": v["crps"] for k, v in summ.items() if k in FLUXES},
                          **{f"cov95_{k}": v["cov95"] for k, v in summ.items() if k in FLUXES}})
        print(w.site, spec["start"], flush=True)
        (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"wrote {len(index)} window-masks to {out}")


if __name__ == "__main__":
    main()
