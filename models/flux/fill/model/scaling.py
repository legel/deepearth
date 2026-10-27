"""FluxFill scaling study: collect runs into scaling.json and fit L(x) = a * x^(-alpha) + c.

Runs are registered in `scaling/runs.json` as {run: axis}, axis "data", "model" or "data,model" (the reference run,
which is a point of both). "data": fixed model, nested stratified subsets of the
training sites, `train.py --n-train-sites`) or "model" (all training sites, model width and depth varied). For each run
this reads `<run>/config.json`, `<run>/log.jsonl` and the per-window validation scores `<run>/val_windows/stepN.json`
written by train.py at every evaluation, and takes the evaluation with the lowest validation CRPS.

Fit: for each axis and each score (all tower variables; fluxes only), non-linear least squares of
L(x) = a x^(-alpha) + c, x = number of training sites (data axis) or number of parameters (model axis). Uncertainty:
1,000 bootstrap resamples of the validation windows (the same resample for every run, since all runs share the 60 fixed
validation windows); each run's score is recomputed as the value-weighted mean over the resampled windows and the law
refitted. Reported: point estimates, bootstrap median and 2.5-97.5 % interval of alpha and c, and the residuals.

Usage: python scaling.py [--register RUN AXIS] [--fit]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

MODEL = Path("/home/photon/4tb/fluxfill/model")
OUT = MODEL / "scaling"


def run_record(run: str, axis: str):
    d = MODEL / run
    if not (d / "log.jsonl").exists():
        return None
    cfg = json.loads((d / "config.json").read_text())
    ev = [json.loads(l) for l in open(d / "log.jsonl") if '"eval"' in l]
    ev = [e for e in ev if "val_crps" in e]
    if not ev:
        return None
    best = min(ev, key=lambda e: e["val_crps"])
    last = max(e["step"] for e in ev)
    tps = cfg.get("tokens_per_step", cfg["batch"] * cfg.get("accum", 1) * 31 * 180)
    return {"run": run, "axis": axis, "n_params": cfg.get("n_params"), "d": cfg["d"], "depth": cfg["depth"],
            "heads": cfg["heads"], "n_train_sites": len(cfg["train_sites"]),
            "n_train_sites_used": cfg.get("n_train_sites_used"), "n_train_windows": cfg.get("n_train_windows"),
            "steps": last, "best_step": best["step"], "tokens_seen": int(last * tps),
            "tokens_seen_at_best": int(best["step"] * tps), "val_crps": best["val_crps"],
            "val_crps_flux": best["val_crps_flux"], "val_cov95": best.get("val_cov95"),
            "complete": last >= cfg["steps"] or (cfg.get("stop_at") and last >= cfg["stop_at"]),
            "flags": {k: cfg.get(k) for k in ("flow", "memory", "mem_residual", "gated_base", "local_base",
                                               "components", "img_nll", "imagery_sites")}}


def window_scores(run: str, step: int):
    f = MODEL / run / "val_windows" / f"step{step:06d}.json"
    return json.loads(f.read_text()) if f.exists() else None


def power_law(x, a, alpha, c):
    return a * np.power(x, -alpha) + c


def fit(x, y):
    from scipy.optimize import curve_fit
    x, y = np.asarray(x, float), np.asarray(y, float)
    p0 = (max(y[0] - y[-1], 1e-3) * x[0] ** 0.3, 0.3, max(y.min() * 0.9, 0.0))
    p, _ = curve_fit(power_law, x, y, p0=p0, bounds=([0, 0, 0], [np.inf, 3.0, np.inf]), maxfev=20000)
    return p


def fit_axis(recs, axis: str, key: str, n_boot: int = 1000, seed: int = 0):
    rs = sorted([r for r in recs if axis in r["axis"].split(",") and r["complete"]],
                key=lambda r: r["n_train_sites"] if axis == "data" else r["n_params"])
    if len(rs) < 4:
        return {"axis": axis, "score": key, "n_runs": len(rs), "note": "fewer than 4 complete runs: no fit"}
    xs = [r["n_train_sites"] if axis == "data" else r["n_params"] for r in rs]
    ys = [r[key] for r in rs]
    p = fit(xs, ys)
    out = {"axis": axis, "score": key, "n_runs": len(rs), "x": xs, "y": ys, "a": p[0], "alpha": p[1], "c": p[2],
           "residuals": list(np.asarray(ys) - power_law(np.asarray(xs, float), *p))}
    W = [window_scores(r["run"], r["best_step"]) for r in rs]
    if all(W):
        ids = sorted(set.intersection(*[{w["id"] for w in ws} for ws in W]))
        col = "crps" if key == "val_crps" else "crps_flux"
        wcol = "n" if key == "val_crps" else "n_flux"
        M = np.array([[next(w for w in ws if w["id"] == i)[col] for i in ids] for ws in W], float)
        N = np.array([[next(w for w in ws if w["id"] == i)[wcol] for i in ids] for ws in W], float)
        M = np.nan_to_num(M); N = np.where(np.isfinite(M), N, 0)
        rng = np.random.default_rng(seed)
        al, cs = [], []
        for _ in range(n_boot):
            k = rng.integers(0, len(ids), len(ids))
            yb = (M[:, k] * N[:, k]).sum(1) / np.maximum(N[:, k].sum(1), 1)
            try:
                pb = fit(xs, yb)
                al.append(pb[1]); cs.append(pb[2])
            except Exception:
                continue
        if al:
            out.update(alpha_boot_median=float(np.median(al)), alpha_ci95=[float(np.quantile(al, 0.025)),
                       float(np.quantile(al, 0.975))], c_ci95=[float(np.quantile(cs, 0.025)), float(np.quantile(cs, 0.975))],
                       n_boot_ok=len(al), n_windows=len(ids))
    return {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", nargs=2, metavar=("RUN", "AXIS"), action="append", default=[])
    ap.add_argument("--fit", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    reg_f = OUT / "runs.json"
    reg = json.loads(reg_f.read_text()) if reg_f.exists() else {}
    for run, axis in a.register:
        assert set(axis.split(",")) <= {"data", "model"}, axis
        reg[run] = axis
    reg_f.write_text(json.dumps(reg, indent=1))
    recs = [r for r in (run_record(run, axis) for run, axis in reg.items()) if r]
    (OUT / "scaling.json").write_text(json.dumps(recs, indent=1, default=str))
    print(f"scaling.json: {len(recs)} runs")
    if a.fit:
        fits = [fit_axis(recs, ax, k) for ax in ("data", "model") for k in ("val_crps", "val_crps_flux")]
        (OUT / "scaling_fit.json").write_text(json.dumps(fits, indent=1, default=str))
        for f_ in fits:
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in f_.items() if k not in ("x", "y", "residuals")})


if __name__ == "__main__":
    sys.exit(main())
