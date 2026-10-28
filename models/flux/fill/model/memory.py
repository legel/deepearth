"""Site memory: what the site's other years say about each variable at each half-hour of each day of the window.

For variable v, day-of-year j and half-hour k, the memory is the mean and standard deviation of the site's measured
values at half-hour k within +/- 15 days of day j (circular in the year), over the site's other years, and the
number of values behind them. These are the statistics of the climatology floor (baselines/climatology.py), given to
the model as three extra channels of every day-token (docs/MODEL.md section 5 specifies site memory as pooled encoder
summaries of other years; this is its first, closed-form version).

No value of the window's own calendar years enters: in training the window's years (and a training site's held-out
year) are subtracted from the site totals; in the benchmark the runner's context table already excludes them.
"""

from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from data import STEPS_PER_DAY, V_TAB, WINDOW_DAYS, SiteRecord, Stats

HALF_WIDTH_DAYS = 15
MIN_COUNT = 3
K = STEPS_PER_DAY


def _day_doy(rec: SiteRecord) -> Tuple[np.ndarray, np.ndarray]:
    days = rec.t0 + pd.to_timedelta(np.arange(rec.n_days), unit="D")
    return days.year.to_numpy(), days.day_of_year.to_numpy() - 1


def year_sums(rec: SiteRecord, year: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum, sum of squares and count of measured values of one calendar year by (V, 366, 48)."""
    yr, doy = _day_doy(rec)
    sel = np.flatnonzero(yr == year)
    S1 = np.zeros((V_TAB, 366, K)); S2 = np.zeros((V_TAB, 366, K)); N = np.zeros((V_TAB, 366, K))
    if sel.size == 0:
        return S1, S2, N
    x = rec.values[:, sel[0] * K:(sel[-1] + 1) * K].reshape(V_TAB, sel.size, K).astype(np.float64)
    ok = np.isfinite(x)
    x0 = np.where(ok, x, 0.0)
    d = doy[sel]                                   # unique within a year
    S1[:, d] = x0; S2[:, d] = x0 * x0; N[:, d] = ok
    return S1, S2, N


def site_totals(rec: SiteRecord, skip_years: Iterable[int] = ()) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    yr, _ = _day_doy(rec)
    skip = set(int(y) for y in skip_years)
    tot = [np.zeros((V_TAB, 366, K)) for _ in range(3)]
    for y in np.unique(yr):
        if int(y) in skip:
            continue
        for a, b in zip(tot, year_sums(rec, int(y))):
            a += b
    return tuple(tot)


def smooth_doy(x: np.ndarray, half: int = HALF_WIDTH_DAYS) -> np.ndarray:
    """Circular moving sum over the day-of-year axis (axis 1) with half-width `half`."""
    pad = np.concatenate([x[:, -half:], x, x[:, :half]], 1)
    c = np.cumsum(pad, 1)
    c = np.concatenate([np.zeros_like(c[:, :1]), c], 1)
    w = 2 * half + 1
    return c[:, w:] - c[:, :-w]


def memory_from_sums(S1, S2, N, stats: Stats) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardised mean, standardised standard deviation and a has-memory flag, each (V, 366, 48)."""
    s1, s2, n = smooth_doy(S1), smooth_doy(S2), smooth_doy(N)
    has = n >= MIN_COUNT
    m = np.where(has, s1 / np.maximum(n, 1), 0.0)
    sd = np.sqrt(np.maximum(np.where(has, s2 / np.maximum(n, 1), 0.0) - m * m, 0.0))
    mz = np.where(has, (m - stats.mean[:, None, None]) / stats.std[:, None, None], 0.0)
    sz = np.where(has, sd / stats.std[:, None, None], 0.0)
    return mz.astype(np.float32), sz.astype(np.float32), has


def window_memory(mz, sz, has, start: pd.Timestamp, n_days: int = WINDOW_DAYS):
    """Memory channels of a window, (V, D, 48) each, from (V, 366, 48) day-of-year tables."""
    d = pd.date_range(pd.Timestamp(start), periods=n_days, freq="D").day_of_year.to_numpy() - 1
    return mz[:, d], sz[:, d], has[:, d]


class SiteMemory:
    """Per-site totals over all usable years, from which a window's own years are subtracted on demand."""

    def __init__(self, rec: SiteRecord, never_years: Iterable[int] = ()):
        self.rec = rec
        self.never = [int(y) for y in never_years]
        # stored as float32 (RAM over ~500 sites); differences and moments are taken in float64
        self.S1, self.S2, self.N = (x.astype(np.float32) for x in site_totals(rec, self.never))

    def for_window(self, start_day: int, stats: Stats, drop: bool = False):
        start = self.rec.t0 + pd.Timedelta(days=start_day)
        if drop:
            z = np.zeros((V_TAB, WINDOW_DAYS, K), np.float32)
            return z, z, np.zeros(z.shape, bool)
        end = start + pd.Timedelta(days=WINDOW_DAYS - 1)
        S1, S2, N = (x.astype(np.float64) for x in (self.S1, self.S2, self.N))
        for y in range(start.year, end.year + 1):
            if y in self.never:
                continue
            a, b, c = year_sums(self.rec, y)
            S1 -= a; S2 -= b; N -= c
        return window_memory(*memory_from_sums(S1, S2, np.maximum(N, 0), stats), start)


def memory_from_table(values: np.ndarray, t0: pd.Timestamp, stats: Stats, start: pd.Timestamp):
    """Memory channels for a window from a context record (V, T) that already excludes the window's years."""
    rec = SiteRecord("ctx", t0, values)
    S1, S2, N = site_totals(rec)
    return window_memory(*memory_from_sums(S1, S2, N, stats), start)


# ------------------------------------------------------------------------------------- skewness channel (E23, flag)
# Separate functions, used only when the memory skewness is requested, so runs without it execute the code above
# unchanged. Power sums are taken of values minus a per-(variable, half-hour) centre of the site; the centre is the
# same for every day of the year, so within each (day-of-year, half-hour) pool it is a constant and the central
# moments do not depend on it. Removing the diurnal offset keeps the float32 cancellation of the third moment small
# (what remains is the seasonal offset of the pool from the centre).

def slot_shift(rec: SiteRecord) -> np.ndarray:
    """(V, 48) mean of each variable at each half-hour of the day over the whole record (0 where none)."""
    x = rec.values[:, : rec.n_days * K].reshape(V_TAB, rec.n_days, K).astype(np.float64)
    with np.errstate(all="ignore"):
        c = np.nanmean(x, 1)
    return np.nan_to_num(c)


def year_sums_skew(rec: SiteRecord, year: int, shift: np.ndarray):
    """Sums of the first three powers of (value - shift) and the count, one calendar year, by (V, 366, 48)."""
    yr, doy = _day_doy(rec)
    sel = np.flatnonzero(yr == year)
    S1, S2, S3, N = (np.zeros((V_TAB, 366, K)) for _ in range(4))
    if sel.size == 0:
        return S1, S2, S3, N
    x = rec.values[:, sel[0] * K:(sel[-1] + 1) * K].reshape(V_TAB, sel.size, K).astype(np.float64) - shift[:, None, :]
    ok = np.isfinite(x)
    x0 = np.where(ok, x, 0.0)
    d = doy[sel]
    S1[:, d] = x0; S2[:, d] = x0 * x0; S3[:, d] = x0 * x0 * x0; N[:, d] = ok
    return S1, S2, S3, N


def site_totals_skew(rec: SiteRecord, skip_years: Iterable[int], shift: np.ndarray):
    yr, _ = _day_doy(rec)
    skip = set(int(y) for y in skip_years)
    tot = [np.zeros((V_TAB, 366, K)) for _ in range(4)]
    for y in np.unique(yr):
        if int(y) not in skip:
            for a_, b_ in zip(tot, year_sums_skew(rec, int(y), shift)):
                a_ += b_
    return tuple(tot)


def memory_from_sums_skew(S1, S2, S3, N, shift: np.ndarray, stats: Stats):
    """Standardised mean, standardised sd, skewness (clipped to +/- 5, 0 where the spread is negligible) and has."""
    s1, s2, s3, n = smooth_doy(S1), smooth_doy(S2), smooth_doy(S3), smooth_doy(N)
    has = n >= MIN_COUNT
    nn = np.maximum(n, 1)
    mc = np.where(has, s1 / nn, 0.0)
    e2 = np.where(has, s2 / nn, 0.0)
    var = np.maximum(e2 - mc * mc, 0.0)
    m3 = np.where(has, s3 / nn - 3 * mc * e2 + 2 * mc ** 3, 0.0)
    m = mc + shift[:, None, :]
    small = var <= (1e-4 * stats.std[:, None, None]) ** 2
    skew = np.where(has & ~small, np.clip(m3 / np.maximum(var, 1e-30) ** 1.5, -5.0, 5.0), 0.0)
    mz = np.where(has, (m - stats.mean[:, None, None]) / stats.std[:, None, None], 0.0)
    sz = np.where(has, np.sqrt(var) / stats.std[:, None, None], 0.0)
    return mz.astype(np.float32), sz.astype(np.float32), skew.astype(np.float32), has


def window_memory_skew(mz, sz, kz, has, start: pd.Timestamp, n_days: int = WINDOW_DAYS):
    d = pd.date_range(pd.Timestamp(start), periods=n_days, freq="D").day_of_year.to_numpy() - 1
    return mz[:, d], sz[:, d], kz[:, d], has[:, d]


class SiteMemorySkew:
    """As SiteMemory, with the third power sum, for the skewness channel."""

    def __init__(self, rec: SiteRecord, never_years: Iterable[int] = ()):
        self.rec = rec
        self.never = [int(y) for y in never_years]
        self.shift = slot_shift(rec)
        self.S1, self.S2, self.S3, self.N = (x.astype(np.float32) for x in site_totals_skew(rec, self.never, self.shift))

    def tables(self, years: Iterable[int], stats: Stats):
        """(mean, sd, skew, has) day-of-year tables without the given calendar years."""
        S = [x.astype(np.float64) for x in (self.S1, self.S2, self.S3, self.N)]
        for y in years:
            if y not in self.never:
                for a_, b_ in zip(S, year_sums_skew(self.rec, y, self.shift)):
                    a_ -= b_
        S[3] = np.maximum(S[3], 0)
        return memory_from_sums_skew(*S, self.shift, stats)


def memory_from_table_skew(values: np.ndarray, t0: pd.Timestamp, stats: Stats, start: pd.Timestamp):
    """(mean, sd, skew, has) channels for a window from a context record without the window's years."""
    rec = SiteRecord("ctx", t0, values)
    sh = slot_shift(rec)
    return window_memory_skew(*memory_from_sums_skew(*site_totals_skew(rec, (), sh), sh, stats), start)
