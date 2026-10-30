"""XGBoost gap filling of Liu et al. 2025 as a FluxFill-Bench method, with Clay embeddings in place of PhenoCam GCC.

Liu, Y., Lucas, B., Bergl, D. D., Richardson, A. D. 2025. Robust filling of extra-long gaps in eddy covariance CO2
flux measurements from a temperate deciduous forest using eXtreme Gradient Boosting. Agricultural and Forest
Meteorology 364:110438. https://doi.org/10.1016/j.agrformet.2025.110438
Code: github.com/YujieLiu666/fluxgapfill_XGB_vs_MDS, commit a9a6eeb8d13820de105c859673c55a0873e2f283 (local clone
/home/photon/ecological/flux_gapfill/baselines_src/fluxgapfill_XGB_vs_MDS).

Ported, with the released notebook cells (train_XGB.ipynb):
- Predictors, "02_Base" (cell 5): TIMESTAMP_END (time as a number), SWC_1_f, SWC_4_f, Tair_f, Tsoil, VPD_f,
  PPFD_DIF, PPFD_DIR, PPFD_f, DOY_sin, DOY_cos. Here: time in days since 2000-01-01 LST (a monotone transform of
  TIMESTAMP_END, so the trees' splits are the same), SWC_1, TS_1, TA, VPD, SW_IN (in place of the three PPFD
  columns, as the task brief fixes; most towers have no diffuse/direct PPFD), DOY_sin, DOY_cos. The site's SWC_4 is
  not in the benchmark tables.
- Vegetation state "03_GCC+Base" (cell 5) adds daily PhenoCam GCC. PhenoCam is not available at most towers; the
  Sentinel-2 Clay v1.5 CLS embedding stands in where the site has one (`CLAY_SITES` records which windows did):
  the first 8 principal components of the site's clear chips (SCL clear fraction >= 0.7), interpolated linearly in
  time between acquisitions, missing outside the acquisitions and across gaps longer than 60 days. Imagery is
  never masked in M2, so this uses no hidden value.
- Model: XGBRegressor(booster='gbtree', tree_method='hist', random_state=42) (cell 11).
- Fit on every row where the flux is measured (cell 15: `dropna(subset=[y_col])`, then `reg.fit(X, y)`); XGBoost
  routes missing predictors natively.

Changes, each stated:
1. Training rows: the site's other calendar years (runner `context`) plus the visible part of the window. Drivers
   with an ERA5 counterpart (TA, VPD, SW_IN) take ERA5 where the tower value is hidden or missing; TS_1 and SWC_1 have
   none and stay missing (XGBoost's default branch).
2. No hyperparameter search. The release runs BayesSearchCV with up to 5,100 trials per predictor set (cell 11).
   One fixed configuration inside its search space (cell 7) is used for every site: learning_rate 0.04 (the top
   of the space), max_depth 8, subsample 0.7, colsample_bytree 0.8, reg_lambda 1, reg_alpha 0, n_estimators 200,
   min_child_weight 5, gamma 0.1 (`XGB_PARAMS`). The target is standardised on the training rows, so gamma and
   lambda act on a unit scale at every site.
3. Uncertainty (the release has none): XGBoost multi-quantile regression (`reg:quantileerror`) at the levels in
   `LEVELS`, one model with the same configuration. Samples are inverse-CDF draws from the predicted quantiles
   (common.quantile_samples), independent per half-hour. The quantiles are fitted in-sample on the training rows,
   without calibration. Sums from these samples assume independence. Seven levels and 200 rounds keep the cost at
   about 1,400 trees per fit on 1 CPU core (one multi-quantile tree per level per round).
4. DOY_sin, DOY_cos = sin, cos(2 pi (doy - 1) / 365); the release imports them from `function_train_xgb`, which is
   not in the repository.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from common import EPOCH, Context, context_keys, fill_output, flux_rows, place, quantile_samples, seed_of, window_driver

DRIVERS = ("SWC_1", "TS_1", "TA", "VPD", "SW_IN")
XGB_PARAMS = dict(learning_rate=0.04, max_depth=8, subsample=0.7, colsample_bytree=0.8, reg_lambda=1.0,
                  reg_alpha=0.0, n_estimators=200, min_child_weight=5, gamma=0.1)
LEVELS = np.array([0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975])
CLAY_ROOT = Path("/home/photon/4tb/fluxfill/clay/sites")
N_PC, CLEAR_MIN, MAX_INTERP_DAYS = 8, 0.7, 60
CLAY_SITES = {}
"""site -> number of clear Clay chips used; filled as windows are run (the record of which sites had imagery)."""


def base_time(t: pd.DatetimeIndex) -> np.ndarray:
    doy = t.dayofyear.to_numpy()
    days = ((t - EPOCH) / pd.Timedelta(days=1)).to_numpy()
    return np.stack([days, np.sin(2 * np.pi * (doy - 1) / 365), np.cos(2 * np.pi * (doy - 1) / 365)], 1)


class ClayPCs:
    """The site's Clay CLS embeddings as N_PC principal components on any LST time axis (None without a store)."""

    def __init__(self, site: str, utc_offset_h: float, root: Path = CLAY_ROOT):
        self.ok = False
        p = root / f"{site}.zarr"
        if not p.exists():
            return
        import zarr
        g = zarr.open(str(p), mode="r")
        n = int(g.attrs.get("n_committed", g["cls"].shape[0]))
        if n == 0:
            return
        clear = np.nan_to_num(np.asarray(g["meta"]["clear_frac"][:n], np.float64), nan=0.0)
        keep = clear >= CLEAR_MIN
        if keep.sum() < N_PC + 2:
            return
        cls = np.asarray(g["cls"][:n], np.float64)[keep]
        t_us = np.asarray(g["time_us"][:n], np.int64)[keep] + int(utc_offset_h * 3600e6)
        days = (t_us / 86_400e6) - (EPOCH - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1)
        order = np.argsort(days)
        cls, days = cls[order], days[order]
        mu = cls.mean(0)
        u, s, vt = np.linalg.svd(cls - mu, full_matrices=False)
        self.pcs = (cls - mu) @ vt[:N_PC].T
        self.days, self.n, self.ok = days, int(keep.sum()), True

    def at(self, t: pd.DatetimeIndex) -> np.ndarray:
        d = ((t - EPOCH) / pd.Timedelta(days=1)).to_numpy()
        out = np.column_stack([np.interp(d, self.days, self.pcs[:, j]) for j in range(N_PC)])
        j = np.searchsorted(self.days, d)
        lo = self.days[np.clip(j - 1, 0, len(self.days) - 1)]
        hi = self.days[np.clip(j, 0, len(self.days) - 1)]
        bad = (d < self.days[0]) | (d > self.days[-1]) | ((hi - lo) > MAX_INTERP_DAYS)
        out[bad] = np.nan
        return out


def utc_offset(table: pd.DataFrame) -> float:
    if "t_lst" in table and "t_utc" in table:
        return float(np.median((table["t_lst"].to_numpy()[:1000] - table["t_utc"].to_numpy()[:1000])) / 60.0)
    if "time_lst" in table and "time_utc" in table:
        utc = pd.DatetimeIndex(table["time_utc"][:1000])
        utc = utc.tz_convert(None) if utc.tz is not None else utc    # final tables: time_utc tz-aware, time_lst naive
        d = pd.DatetimeIndex(table["time_lst"][:1000]) - utc
        return float(np.median(d / pd.Timedelta(hours=1)))
    return 0.0


class LiuXGB:
    name = "liu_xgb"

    def __init__(self, n_samples: int = 64, params: Dict = None, use_clay: bool = True, device: str = "cpu",
                 n_jobs: int = 2, seed: int = 0):
        self.p = dict(XGB_PARAMS if params is None else params)
        self.n, self.use_clay, self.device, self.n_jobs, self.seed = n_samples, use_clay, device, n_jobs, seed
        self._clay: Dict[str, Optional[ClayPCs]] = {}
        self.log = []

    def clay(self, site: str, table: pd.DataFrame) -> Optional[ClayPCs]:
        if site not in self._clay:
            off = utc_offset(table)
            if not np.isfinite(off):                  # no time columns to read it from: the runner's site offset
                off = float(getattr(self, "tz", 0.0) or 0.0)
            c = ClayPCs(site, off) if self.use_clay else None
            self._clay[site] = c if (c is not None and c.ok) else None
            if self._clay[site] is not None:
                CLAY_SITES[site] = self._clay[site].n
        return self._clay[site]

    def predict(self, w, mask: np.ndarray, context: Dict) -> np.ndarray:
        import xgboost as xgb
        out = fill_output(w, mask, self.n)
        rows = flux_rows(w, mask)
        if not rows:
            return out
        start = str(w.start.date())
        ctx = Context.from_runner(context, context_keys(DRIVERS))
        Xc = np.column_stack([base_time(ctx.t)] + [ctx.driver(d) for d in DRIVERS])
        Xw = np.column_stack([base_time(w.t)] + [window_driver(w, mask, d) for d in DRIVERS])
        c = self.clay(w.site, context["table"])
        if c is not None:
            Xc, Xw = np.column_stack([Xc, c.at(ctx.t)]), np.column_stack([Xw, c.at(w.t)])
        for i in rows:
            k = w.keys[i]
            yc, yw = ctx.get(k), np.where(mask[i], np.nan, w.values[i])
            okc, okw = np.isfinite(yc), np.isfinite(yw)
            if okc.sum() + okw.sum() < 48 * 7:
                continue
            X = np.vstack([Xc[okc], Xw[okw]])
            y = np.concatenate([yc[okc], yw[okw]])
            my, sy = y.mean(), y.std() or 1.0
            reg = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=LEVELS, booster="gbtree",
                                   tree_method="hist", random_state=42, n_jobs=self.n_jobs, device=self.device,
                                   **self.p)
            reg.fit(X, (y - my) / sy)
            hid = mask[i]
            q = reg.predict(Xw[hid]).T * sy + my                     # (Q, n_hidden)
            s = np.full((self.n, w.values.shape[1]), np.nan)
            s[:, hid] = quantile_samples(LEVELS, q, self.n, np.random.default_rng(
                seed_of(self.seed, w.site, start, k, int(mask.any(0).sum()))))
            place(out, i, hid, s)
            self.log.append(dict(site=w.site, window=start, variable=k, n_train=int(len(y)), clay=c is not None))
        return out
