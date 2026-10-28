"""Climatology: the floor reference of FluxFill-Bench.

For each masked half-hour, the predictive distribution is the empirical distribution of the same variable's measured
values at the same site, at the same half-hour of the day, within +/- 15 days of the same day of year, in every
other year of the record (the target window's own year excluded, so no value inside the window is used). Samples
are drawn from that pool. It uses no drivers; any method that does not beat it has learned nothing beyond the
mean seasonal and diurnal cycle.
"""

import numpy as np
import pandas as pd

HALF_WIDTH_DAYS = 15


def pools(series: pd.Series, measured: pd.Series, exclude: pd.DatetimeIndex):
    """Measured values indexed by (day of year, half-hour slot), with the excluded span removed."""
    ok = measured.to_numpy() & ~series.index.isin(exclude)
    s = series[ok]
    doy = s.index.dayofyear.to_numpy()
    slot = (s.index.hour * 2 + s.index.minute // 30).to_numpy()
    return s.to_numpy(np.float64), doy, slot


def predict(series: pd.Series, measured: pd.Series, target: pd.DatetimeIndex, n_samples: int = 64,
            seed: int = 0) -> np.ndarray:
    """(n_samples, len(target)) samples. Columns with no pooled value are NaN."""
    excl = series.index[(series.index.year >= target[0].year) & (series.index.year <= target[-1].year)]
    vals, doy, slot = pools(series, measured, excl)
    rng = np.random.default_rng(seed)
    out = np.full((n_samples, len(target)), np.nan)
    tdoy = target.dayofyear.to_numpy()
    tslot = (target.hour * 2 + target.minute // 30).to_numpy()
    by_slot = {k: np.flatnonzero(slot == k) for k in range(48)}
    for j in range(len(target)):
        idx = by_slot[tslot[j]]
        d = np.abs(doy[idx] - tdoy[j])
        d = np.minimum(d, 366 - d)
        pool = vals[idx[d <= HALF_WIDTH_DAYS]]
        if pool.size:
            out[:, j] = rng.choice(pool, size=n_samples, replace=True)
    return out
