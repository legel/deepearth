"""Fit the temporal Gaussian copula of FluxFill (method.FluxFill.sample_copula) to validation PITs.

Input: PIT files written by bench/run.py with FLUXFILL_PIT_DUMP (one per window and mask, the exact predictive CDF of
every hidden measured value). For each flux the PITs are mapped to normal scores, and the autocorrelation at lag k
(half-hours) is pooled over all hidden blocks: sum of s_t s_{t+k} over pairs with both values measured, divided by
the number of such pairs (normal scores have mean 0 and variance 1 under calibration, so no centring is applied).
The two-scale model corr(k) = w exp(-k / L_slow) + (1 - w) exp(-k / L_fast) is fitted by least squares on lags from
1 half-hour to 30 days. Variables other than the four fluxes use the mean of the flux parameters.

    python copula_fit.py PIT_DIR OUT.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import optimize, stats

LAGS = np.array([1, 2, 4, 8, 16, 24, 48, 96, 144, 240, 336, 480, 720, 1008, 1440])
FLUXES = ["NEE", "LE", "H", "FCH4"]


def acf(series, lags):
    num = np.zeros(len(lags)); cnt = np.zeros(len(lags))
    for s in series:
        ok = np.isfinite(s)
        z = np.where(ok, s, 0.0)
        for i, k in enumerate(lags):
            if k >= len(s):
                continue
            both = ok[:-k] & ok[k:]
            num[i] += (z[:-k] * z[k:])[both].sum(); cnt[i] += both.sum()
    return num / np.maximum(cnt, 1), cnt


def main(pit_dir: str, out: str):
    files = sorted(Path(pit_dir).glob("*.npz"))
    series = {v: [] for v in FLUXES}
    for f in files:
        d = np.load(f)
        keys = list(d["keys"])
        for v in FLUXES:
            if v in keys:
                p = d["pit"][keys.index(v)].astype(np.float64)
                if np.isfinite(p).sum() > 48:
                    series[v].append(stats.norm.ppf(np.clip(p, 1e-6, 1 - 1e-6)))
    model = lambda k, w, ls, lf: w * np.exp(-k / ls) + (1 - w) * np.exp(-k / lf)
    res, report = {}, {}
    for v in FLUXES:
        if not series[v]:
            continue
        r, n = acf(series[v], LAGS)
        keep = n > 100
        (w, ls, lf), _ = optimize.curve_fit(model, LAGS[keep], r[keep], p0=[0.5, 480, 4], bounds=([0, 48, 0.1], [1, 20000, 48]))
        res[v] = [float(w), float(ls), float(lf)]
        report[v] = {"blocks": len(series[v]), "acf": {int(k): round(float(x), 3) for k, x in zip(LAGS, r)}, "fit": res[v]}
    res["_default"] = [float(np.mean([res[v][i] for v in res])) for i in range(3)]
    Path(out).write_text(json.dumps(res, indent=1))
    Path(out).with_suffix(".report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
