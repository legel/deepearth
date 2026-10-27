"""Random forest ensemble of Irvin et al. 2021 as a FluxFill-Bench method.

Irvin, J., Zhou, S., McNicol, G., et al. 2021. Gap-filling eddy covariance methane fluxes: comparison of machine
learning model predictions and uncertainties at FLUXNET-CH4 wetlands. Agricultural and Forest Meteorology
308-309:108528. https://doi.org/10.1016/j.agrformet.2021.108528
Released harness: github.com/stanfordmlgroup/methane-gapfill-ml, commit 473db3e7711f71a1ed8151a3ef4b0e08c70a5a08
(local clone /home/photon/ecological/flux_gapfill/methane-gapfill-ml; reproduction report
flux_gapfill/HARNESS_REPRODUCTION.md).

What is ported, with the harness lines:
- Predictors: the harness's `knox` set, temporal + TA, SW_IN, WS, PA (predictors/knox.txt). Temporal =
  yearly_sin, yearly_cos = sin, cos(2 pi (doy - 1) / 365) and delta (predictors/process.py:32-42).
- Missing predictors imputed with the training mean (models/base.py:43-50, SimpleImputer(strategy='mean')).
- Model: sklearn RandomForestRegressor (models/models.py:63-75).
- Ensemble of 10 members, predictive Laplace with location = member mean and scale = member std / 2
  (models/ensemble.py:77-116, line 96).
- Platt-style scaling: factor = mean(|y - mean| / scale), the Laplace maximum-likelihood factor of the scale
  (models/ensemble.py:125-132).

Changes, each stated:
1. `delta` fix. The harness computes delta as days since the first row of whatever frame it is given
   (process.py:40: `timestamp - timestamp.iloc[0]`), so fit and prediction frames see different origins
   (HARNESS_REPRODUCTION.md section 10). Here delta = days since 2000-01-01 00:00 local standard time for every
   frame, so a time has the same delta at fit and at prediction.
2. No hyperparameter search. The harness runs RandomizedSearchCV (20 configurations, 5-fold CV) per member
   (base.py:52-70); at this benchmark's scale and machine load that costs thousands of core-hours
   (HARNESS_REPRODUCTION.md section 5). One fixed configuration from the harness's own grid is used for every
   site: n_estimators=50, max_features=1.0 (the released 'auto'), min_samples_leaf=4, max_depth=None,
   bootstrap=True (`RF_PARAMS`). Each tree draws a bootstrap sample of 4,000 rows (sklearn `max_samples=4000`, not in
   the harness grid; the forest of a member still sees 200,000 draws), which keeps the cost per tree constant across
   sites at the machine load of 2026-10-26 (about 13 min per window at 20 % of the rows on the smallest site).
3. Members. The harness's members are the 10 train/validation splits of its protocol. Here each member is fit on
   a bootstrap resample of whole days of the training rows (common.day_bootstrap), with its own seed.
4. Training rows: the site's other calendar years (runner `context`) plus the visible part of the window, where the
   flux is measured. Drivers hidden by the mask, or missing, are replaced by ERA5_<VAR> (common.py).
5. Calibration on held-out blocks of the context, never on the window. Per window and flux: 10 % of the context's
   measured values in 7-day blocks are held out; a 10-member ensemble fit on the rest of the context predicts them,
   with every driver in those blocks replaced by ERA5 as in a masked station outage; the Platt factor is fitted on
   those predictions. The harness fits the factor on its test set (in-sample; HARNESS_REPRODUCTION.md section 7).
6. Scale floor: a member std of zero gives scale 0 in the harness (division by zero in the factor); here the scale
   is floored at 1e-3 times the standard deviation of the training target.

Samples: Laplace draws, independent per half-hour. Sums from these samples have the independence spread of the
marginals: the method has no joint uncertainty.
"""

from typing import Dict

import numpy as np
import pandas as pd

from common import (EPOCH, Context, context_keys, day_bootstrap, fill_output, flux_rows, holdout_blocks,
                    laplace_samples, place, seed_of, window_driver)

DRIVERS = ("TA", "SW_IN", "WS", "PA")
RF_PARAMS = dict(n_estimators=50, max_features=1.0, min_samples_leaf=4, max_depth=None, bootstrap=True, max_samples=4000)
N_MEMBERS = 10


def temporal(t: pd.DatetimeIndex) -> np.ndarray:
    """yearly_sin, yearly_cos (process.py:35-37) and delta in days since the fixed EPOCH (fix of process.py:40)."""
    doy = t.dayofyear.to_numpy()
    delta = ((t - EPOCH) / pd.Timedelta(days=1)).to_numpy()
    return np.stack([np.sin(2 * np.pi * (doy - 1) / 365), np.cos(2 * np.pi * (doy - 1) / 365), delta], 1)


def temporal_released(t: pd.DatetimeIndex) -> np.ndarray:
    """The released delta (days since the frame's first row), kept for the test that documents the fix."""
    doy = t.dayofyear.to_numpy()
    delta = ((t - t[0]) / pd.Timedelta(days=1)).to_numpy()
    return np.stack([np.sin(2 * np.pi * (doy - 1) / 365), np.cos(2 * np.pi * (doy - 1) / 365), delta], 1)


class IrvinRF:
    name = "irvin_rf"

    def __init__(self, n_samples: int = 64, n_members: int = N_MEMBERS, rf_params: Dict = None, n_jobs: int = 2,
                 seed: int = 0):
        self.n, self.m, self.n_jobs, self.seed = n_samples, n_members, n_jobs, seed
        self.rf_params = dict(RF_PARAMS if rf_params is None else rf_params)
        self._platt: Dict = {}
        self.log = []

    # ------------------------------------------------------------------ ensemble
    def _ensemble(self, X: np.ndarray, y: np.ndarray, t: pd.DatetimeIndex, Xp: np.ndarray, seed: int) -> np.ndarray:
        """(members, len(Xp)) member predictions. Mean imputation of predictors as base.py:43-50."""
        from sklearn.ensemble import RandomForestRegressor
        mean = np.nanmean(X, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        X = np.where(np.isfinite(X), X, mean)
        Xp = np.where(np.isfinite(Xp), Xp, mean)
        rng = np.random.default_rng(seed)
        out = np.empty((self.m, len(Xp)))
        p = dict(self.rf_params)
        if isinstance(p.get("max_samples"), int) and p["max_samples"] >= len(X):
            p["max_samples"] = None                      # small training sets: the full bootstrap
        for k in range(self.m):
            rows = day_bootstrap(t, rng)
            rf = RandomForestRegressor(**p, n_jobs=self.n_jobs, random_state=int(rng.integers(2**31)))
            rf.fit(X[rows], y[rows])
            out[k] = rf.predict(Xp)
        return out

    @staticmethod
    def _dist(preds: np.ndarray, floor: float):
        """Laplace location and scale of the harness: mean and std / 2 of the members (ensemble.py:96)."""
        return preds.mean(0), np.maximum(preds.std(0) / 2, floor)

    def _features(self, drv: Dict[str, np.ndarray], t: pd.DatetimeIndex) -> np.ndarray:
        return np.column_stack([temporal(t)] + [drv[d] for d in DRIVERS])

    def platt(self, ctx: Context, key: str, site: str, start: str) -> float:
        """Platt factor of one (window, flux), fitted on held-out 7-day blocks of the context (cached per window)."""
        ck = (site, start, key)
        if ck in self._platt:
            return self._platt[ck]
        y = ctx.get(key)
        rng = np.random.default_rng(seed_of(self.seed, site, start, key, "platt"))
        held = holdout_blocks(ctx.t, y, rng)
        fit = np.isfinite(y) & ~held
        tgt = np.isfinite(y) & held
        f = 1.0
        if fit.sum() > 48 * 30 and tgt.sum() > 48:
            Xfit = self._features({d: ctx.driver(d) for d in DRIVERS}, ctx.t)[fit]
            Xh = self._features({d: ctx.driver(d, era5_rows=held) for d in DRIVERS}, ctx.t)[tgt]
            preds = self._ensemble(Xfit, y[fit], ctx.t[fit], Xh, seed_of(self.seed, site, start, key, "cal"))
            mu, b = self._dist(preds, 1e-3 * np.nanstd(y[fit]))
            f = float(np.mean(np.abs(y[tgt] - mu) / b))          # ensemble.py:130-132
        self._platt[ck] = f
        self.log.append(dict(site=site, window=start, variable=key, platt=f, n_cal=int(tgt.sum())))
        return f

    def predict(self, w, mask: np.ndarray, context: Dict) -> np.ndarray:
        out = fill_output(w, mask, self.n)
        rows = flux_rows(w, mask)
        if not rows:
            return out
        start = str(w.start.date())
        ctx = Context.from_runner(context, context_keys(DRIVERS))
        Xc = self._features({d: ctx.driver(d) for d in DRIVERS}, ctx.t)
        Xw = self._features({d: window_driver(w, mask, d) for d in DRIVERS}, w.t)
        mname = f"{int(mask.any(0).sum())}"
        for i in rows:
            k = w.keys[i]
            yc = ctx.get(k)
            yw = np.where(mask[i], np.nan, w.values[i])
            okc, okw = np.isfinite(yc), np.isfinite(yw)
            if okc.sum() + okw.sum() < 48 * 7:
                continue                                    # under a week of training data: left unfilled
            X = np.vstack([Xc[okc], Xw[okw]])
            y = np.concatenate([yc[okc], yw[okw]])
            t = ctx.t[okc].append(w.t[okw])
            hid = mask[i]
            preds = self._ensemble(X, y, t, Xw[hid], seed_of(self.seed, w.site, start, k, mname))
            mu, b = self._dist(preds, 1e-3 * np.nanstd(y))
            b = b * self.platt(ctx, k, w.site, start)
            s = np.full((self.n, w.values.shape[1]), np.nan)
            s[:, hid] = laplace_samples(mu, b, self.n, np.random.default_rng(seed_of(self.seed, w.site, start, k, mname, "s")))
            place(out, i, hid, s)
        return out
