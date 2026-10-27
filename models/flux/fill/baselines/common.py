"""Shared pieces of the per-site FluxFill-Bench baselines (Irvin RF, Vekuri deep ensemble, Liu XGBoost).

Every per-site baseline fills the tower fluxes (NEE, LE, H, FCH4) of one 180-day window from regressions on
drivers, fitted on the site's other calendar years (the runner's `context`) plus the visible part of the window.

Drivers. Where the tower driver is hidden by the mask or missing, the ONEFlux-downscaled ERA5 value (`ERA5_<VAR>`)
stands in, as FLUXNET fills its drivers (Vuichard and Papale 2015, Earth Syst. Sci. Data 7:157). The context uses
the same rule: tower values with flag 0, else ERA5. Drivers without an ERA5 counterpart (TS_1, SWC_1) stay missing
where the tower value is missing.

Leakage. Only `window.values` entries outside `mask` are read; the hidden entries are never indexed (tests check that
a window with sentinel values under the mask gives the same output as the NaN copy the runner passes).
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

FLUXES = ("NEE", "LE", "H", "FCH4")
EPOCH = pd.Timestamp("2000-01-01")
"""Fixed anchor of every `days since` time feature (local standard time), identical at fit and predict time."""


def flux_rows(w, mask: np.ndarray, fluxes: Sequence[str] = FLUXES) -> List[int]:
    """Rows of the window that are fluxes with at least one hidden value."""
    return [i for i, k in enumerate(w.keys) if k in fluxes and w.provenance[i] == "tower" and mask[i].any()]


def visible(w, mask: np.ndarray, key: str) -> np.ndarray:
    """The window's tower values of `key` outside the mask (NaN under the mask or where not measured)."""
    T = w.values.shape[1]
    if key not in w.keys:
        return np.full(T, np.nan)
    i = w.keys.index(key)
    x = np.full(T, np.nan)
    keep = ~mask[i]
    x[keep] = w.values[i, keep]
    return x


def window_driver(w, mask: np.ndarray, key: str) -> np.ndarray:
    """Tower driver where visible and measured, else the ERA5 value (reanalysis rows are visible in M2 masks)."""
    x = visible(w, mask, key)
    e = "ERA5_" + key
    if e in w.keys:
        j = w.keys.index(e)
        era = np.where(mask[j], np.nan, w.values[j])
        x = np.where(np.isfinite(x), x, era)
    return x


@dataclass
class Context:
    """The site's other calendar years on their own half-hourly LST index: measured values per variable."""
    t: pd.DatetimeIndex
    values: Dict[str, np.ndarray]

    @classmethod
    def from_runner(cls, context: Dict, keys: Sequence[str]) -> "Context":
        tab, idx = context["table"], pd.DatetimeIndex(context["index"])
        out = {}
        for k in keys:
            if k not in tab:
                continue
            x = tab[k].to_numpy(np.float64)
            ok = np.isfinite(x)
            f = tab.get(f"{k}_flag")
            if f is not None:
                ok &= f.to_numpy() == 0
            out[k] = np.where(ok, x, np.nan)
        order = np.argsort(idx.asi8, kind="stable")
        return cls(idx[order], {k: v[order] for k, v in out.items()})

    def get(self, key: str) -> np.ndarray:
        return self.values.get(key, np.full(len(self.t), np.nan))

    def driver(self, key: str, era5_rows: Optional[np.ndarray] = None) -> np.ndarray:
        """Tower value where measured, else ERA5. `era5_rows` (bool) forces ERA5 on those rows (held-out blocks
        that imitate a masked station outage)."""
        x = self.get(key).copy()
        era = self.values.get("ERA5_" + key)
        if era is not None:
            if era5_rows is not None:
                x[era5_rows] = np.nan
            x = np.where(np.isfinite(x), x, era)
        elif era5_rows is not None:
            x[era5_rows] = np.nan
        return x


def context_keys(drivers: Sequence[str], targets: Sequence[str] = FLUXES) -> List[str]:
    return list(dict.fromkeys(list(targets) + list(drivers) + ["ERA5_" + d for d in drivers]))


def holdout_blocks(t: pd.DatetimeIndex, y: np.ndarray, rng: np.random.Generator, frac: float = 0.1,
                   block_days: int = 7) -> np.ndarray:
    """Rows of non-overlapping `block_days` blocks of the context, drawn until they hold `frac` of the measured
    values of y. Used to fit calibration (variance scaling) on the context, never on the window."""
    n = len(t)
    held = np.zeros(n, bool)
    if n == 0:
        return held
    day = ((t - EPOCH) // pd.Timedelta(days=1)).to_numpy()
    blk = (day - day.min()) // block_days
    meas = np.isfinite(y)
    per = np.bincount(blk, weights=meas, minlength=blk.max() + 1)
    cand = np.flatnonzero(per > 0)
    rng.shuffle(cand)
    target, got, chosen = frac * meas.sum(), 0.0, []
    for b in cand:
        if got >= target:
            break
        chosen.append(b)
        got += per[b]
    held[np.isin(blk, chosen)] = True
    return held


def day_bootstrap(t: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Row indices of a bootstrap resample of whole days (days drawn with replacement, all their rows kept), so a
    member sees intact diurnal cycles."""
    day = ((t - EPOCH) // pd.Timedelta(days=1)).to_numpy()
    u, inv = np.unique(day, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(u)))
    ends = np.append(starts[1:], len(order))
    pick = rng.integers(0, len(u), len(u))
    return np.concatenate([order[starts[d]:ends[d]] for d in pick]) if len(u) else np.zeros(0, np.int64)


def laplace_samples(mu: np.ndarray, b: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """(n, T) draws from Laplace(mu, b), independent per step."""
    return mu[None] + b[None] * rng.laplace(size=(n, mu.shape[-1]))


def quantile_samples(levels: np.ndarray, q: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """(n, T) draws from predicted quantiles q (Q, T) at increasing `levels` (Q,), independent per step.

    The quantiles are sorted per step (no crossing), the quantile function is linear between levels, and beyond the
    outer levels it continues with the slope of the outer segment (linear tails). Draw u ~ U(0, 1), return Q(u)."""
    levels = np.asarray(levels, np.float64)
    q = np.sort(np.asarray(q, np.float64), axis=0)
    u = rng.uniform(size=(n, q.shape[1]))
    j = np.clip(np.searchsorted(levels, u) - 1, 0, len(levels) - 2)
    l0, l1 = levels[j], levels[j + 1]
    cols = np.arange(q.shape[1])[None]
    q0, q1 = q[j, cols], q[j + 1, cols]
    return q0 + (q1 - q0) * (u - l0) / (l1 - l0)


def seed_of(*parts) -> int:
    import hashlib
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") >> 1


def fill_output(w, mask: np.ndarray, n: int) -> np.ndarray:
    """The (n, V, T) output array: NaN everywhere (rows a method does not fill stay NaN, counted as unfilled)."""
    return np.full((n,) + w.values.shape, np.nan)


def place(out: np.ndarray, i: int, mask_row: np.ndarray, samples: np.ndarray) -> None:
    """Write samples (n, T) into row i under the mask only; outside the mask the row stays NaN (never scored)."""
    out[:, i, mask_row] = samples[:, mask_row]
