"""Quality rules for tower measurements. Each rule is a pure function on arrays and returns a boolean array of the
values it removes; `build.py` combines them into one flag word per value.

A value is a target only if it was measured (never gap-filled at the source) and no rule removed it. The flag word
says why a value is absent, one bit per rule:

| bit | name | rule |
|---|---|---|
| 1 | FILLED | the source gap-filled this step (ONEFlux QC 1 to 3); the fill is not stored |
| 2 | RANGE | outside the variable's physical range (AmeriFlux FP limits) |
| 4 | SPIKE | double-difference spike (Papale et al. 2006), fluxes only |
| 8 | STUCK | inside a run of identical values at least the variable's stuck length |
| 16 | USTAR | night-time flux below the site-year u* threshold (applied where the source did not) |
| 32 | SITE | the variable failed a whole-record test at this site (sign convention, units) |
| 64 | QUALITY | the source's own steady-state and turbulence test failed (SSITC flag 2, Foken et al. 2004) |
"""

from typing import Dict, Optional, Sequence

import numpy as np

FILLED, RANGE, SPIKE, STUCK, USTAR, SITE, QUALITY = 1, 2, 4, 8, 16, 32, 64
FLAG_NAMES = {FILLED: "filled", QUALITY: "quality", RANGE: "range", STUCK: "stuck", SPIKE: "spike", USTAR: "ustar",
              SITE: "site"}
"""In the order the rules run; each value is counted under the first rule that removes it."""
MISSING = -9999.0
"""The FP-standard missing value."""

SPIKE_Z = 5.5
"""The middle of the three z values of Papale et al. (2006) (4, 5.5, 7). ONEFlux already despiked the fluxes it
publishes; this second pass at 5.5 removes what remains and is the only despiking for FLUXNET-CH4, BASE and ICOS
columns."""
SPIKE_BLOCK_DAYS = 13
"""Papale et al. (2006) compute the median and MAD in 13-day blocks, separately for day and night."""


def clean(values) -> np.ndarray:
    """float64 copy with the FP missing value (-9999, and anything at or below -9000) as NaN."""
    v = np.asarray(values, np.float64).copy()
    v[v <= -9000.0] = np.nan
    return v


def measured(values, qc=None) -> np.ndarray:
    """True where the source measured the step: a value, and QC 0 where a QC column exists (ONEFlux: 0 = measured;
    1 = good-quality gap-fill; 2 = medium; 3 = poor)."""
    v = clean(values)
    ok = np.isfinite(v)
    if qc is not None:
        q = clean(qc)
        ok &= np.isfinite(q) & (q == 0)
    return ok


def out_of_range(v: np.ndarray, lo: float, hi: float, tol: float = 0.0) -> np.ndarray:
    """True where a finite value lies outside [lo - tol, hi + tol]."""
    v = np.asarray(v, np.float64)
    with np.errstate(invalid="ignore"):
        return np.isfinite(v) & ((v < lo - tol) | (v > hi + tol))


def stuck(v: np.ndarray, min_steps: Optional[int], exempt: Sequence[float] = (),
          dark: Optional[np.ndarray] = None, atol: float = 1e-9) -> np.ndarray:
    """True on every value inside a run of at least `min_steps` consecutive identical finite values. Missing steps
    break a run. Runs of an exempt value (within `atol` of it: 0 hPa VPD, 100 % humidity, 0 deg C soil in the
    zero curtain) are kept. With `dark`, darkness repeats legitimately: a run wholly in `dark` steps is kept, and in a
    run that reaches daylight only its daylight steps are removed (a PAR sensor reading 0 in daylight has failed; its
    zeros at night are right)."""
    v = np.asarray(v, np.float64)
    if min_steps is None or v.size == 0:
        return np.zeros(v.shape, bool)
    fin = np.isfinite(v)
    same = np.zeros(v.shape, bool)
    same[1:] = fin[1:] & fin[:-1] & (v[1:] == v[:-1])
    run = np.cumsum(~same)
    length = np.bincount(run)[run]
    out = fin & (length >= min_steps)
    if dark is not None:
        out &= ~np.asarray(dark, bool)
    for x in exempt:
        out &= ~(np.abs(v - x) <= atol)
    return out


def spikes(v: np.ndarray, night: np.ndarray, steps_per_day: int, z: float = SPIKE_Z,
           block_days: int = SPIKE_BLOCK_DAYS) -> np.ndarray:
    """Papale et al. (2006) spike test. For each value with both neighbours, d_i = (v_i - v_{i-1}) - (v_{i+1} - v_i).
    In each block of `block_days`, separately for day and night, a value is a spike if
    d_i < Md - z MAD / 0.6745 or d_i > Md + z MAD / 0.6745, with Md the median of d and MAD its median absolute
    deviation. Of the three values a spike displaces, only the centre (largest |d|) is flagged."""
    v = np.asarray(v, np.float64)
    night = np.asarray(night, bool)
    d = np.full(v.shape, np.nan)
    d[1:-1] = (v[1:-1] - v[:-2]) - (v[2:] - v[1:-1])
    out = np.zeros(v.shape, bool)
    block = block_days * steps_per_day
    for b0 in range(0, v.size, block):
        sl = slice(b0, min(b0 + block, v.size))
        for part in (night[sl], ~night[sl]):
            dd = d[sl][part]
            fin = np.isfinite(dd)
            if fin.sum() < 20:
                continue
            md = np.median(dd[fin])
            mad = np.median(np.abs(dd[fin] - md))
            if mad == 0:
                continue
            lim = z * mad / 0.6745
            hit = np.zeros(part.sum(), bool)
            hit[fin] = (dd[fin] < md - lim) | (dd[fin] > md + lim)
            idx = np.flatnonzero(part) + b0
            out[idx[hit]] = True
    # A spike at i also displaces d at i-1 and i+1 by half its size; keep only the centre, where |d| peaks.
    a = np.nan_to_num(np.abs(d), nan=0.0)
    peak = np.ones(v.shape, bool)
    peak[1:] &= a[1:] >= a[:-1]
    peak[:-1] &= a[:-1] >= a[1:]
    return out & peak


def low_ustar(ustar: np.ndarray, threshold: np.ndarray, only: Optional[np.ndarray] = None) -> np.ndarray:
    """True at steps whose u* is below the step's threshold, or whose u* is missing while a threshold exists (the
    turbulence cannot be verified). Every step is filtered, day and night, as ONEFlux does for NEE_VUT_REF: in the
    ONEFlux v1.3 archives no measured NEE_VUT_REF half-hour, day or night, has u* below its year's VUT threshold
    (`oneflux_nee_below_ustar_threshold` in each site's sidecar). `only` restricts the filter to a subset (for
    example night only, the Papale et al. 2006 practice)."""
    u = np.asarray(ustar, np.float64)
    t = np.asarray(threshold, np.float64)
    has_t = np.isfinite(t)
    with np.errstate(invalid="ignore"):
        low = has_t & (~np.isfinite(u) | (u < t))
    return low if only is None else low & np.asarray(only, bool)


def ustar_mpt(nee: np.ndarray, ustar: np.ndarray, ta: np.ndarray, night: np.ndarray, month: np.ndarray,
              min_per_season: int = 600) -> Optional[float]:
    """One year's u* threshold by the moving point test (Reichstein et al. 2005; Papale et al. 2006), for records that
    no network has u*-filtered and that ONEFlux does not cover.

    Night-time NEE is split by season (JFM, AMJ, JAS, OND), each season into 6 air-temperature classes of equal size
    and each class into 20 u* classes of equal size. In a temperature class the threshold is the mean u* of the first
    u* class whose mean NEE reaches 99 % of the mean NEE of the next 10 classes; classes where u* and temperature
    correlate (|r| >= 0.4) are skipped. A season's threshold is the median over its temperature classes, and the
    year's threshold the largest seasonal one. None when no season has `min_per_season` night values."""
    nee, u, ta = (np.asarray(x, np.float64) for x in (nee, ustar, ta))
    ok = np.isfinite(nee) & np.isfinite(u) & np.isfinite(ta) & np.asarray(night, bool)
    season = (np.asarray(month) - 1) // 3
    seasonal = []
    for s in range(4):
        sel = ok & (season == s)
        if sel.sum() < min_per_season:
            continue
        n_s, u_s, t_s = nee[sel], u[sel], ta[sel]
        found = []
        for tc in np.array_split(np.argsort(t_s), 6):
            if tc.size < 100:
                continue
            if abs(np.corrcoef(u_s[tc], t_s[tc])[0, 1]) >= 0.4:
                continue
            order = tc[np.argsort(u_s[tc])]
            cls = np.array_split(order, 20)
            mf = np.array([n_s[c].mean() for c in cls])
            mu = np.array([u_s[c].mean() for c in cls])
            for i in range(19):
                ahead = mf[i + 1:i + 11].mean()
                if ahead > 0 and mf[i] >= 0.99 * ahead:
                    found.append(mu[i])
                    break
        if found:
            seasonal.append(float(np.median(found)))
    return max(seasonal) if seasonal else None


# ---- Solar geometry, for night masks and the clock test ----------------------------------------------------------

def solar_elevation(lat: float, lon: float, minutes_utc: np.ndarray) -> np.ndarray:
    """Solar elevation in degrees at the given UTC instants (minutes since 1970-01-01), NOAA solar-position equations
    (Spencer 1971 series for declination and the equation of time)."""
    t = np.asarray(minutes_utc, np.float64)
    doy = np.floor((t / 1440.0) % 365.2425)
    g = 2 * np.pi / 365.0 * (doy + ((t % 1440) / 60.0 - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g)
                    - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = (t % 1440) + eqt + 4 * lon
    ha = np.radians(tst / 4 - 180)
    phi = np.radians(lat)
    cosz = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)
    return np.degrees(np.arcsin(np.clip(cosz, -1, 1)))


def step_midpoints_utc(start_minutes_lst: np.ndarray, step_minutes: int, utc_offset_h: float) -> np.ndarray:
    """UTC minutes of each step's midpoint, from TIMESTAMP_START in local standard time."""
    return np.asarray(start_minutes_lst, np.float64) + step_minutes / 2 - utc_offset_h * 60


def clock_offset_minutes(sw_in: np.ndarray, elevation: np.ndarray, step_minutes: int,
                         max_shift_steps: int = 6) -> Optional[float]:
    """The shift (minutes) that best aligns measured SW_IN with the sun, from the mean diurnal cycles.

    Positive means the record's timestamps are late: the measured noon falls after the solar noon implied by the
    stated time zone. Returns None with fewer than 30 days of daytime SW_IN. A correct record gives a shift within
    one step."""
    sw = np.asarray(sw_in, np.float64)
    el = np.asarray(elevation, np.float64)
    per_day = 1440 // step_minutes
    n = (sw.size // per_day) * per_day
    if n == 0:
        return None
    sw, el = sw[:n].reshape(-1, per_day), el[:n].reshape(-1, per_day)
    good = np.isfinite(sw).sum(axis=1) >= per_day * 0.9
    if good.sum() < 30:
        return None
    a = np.nanmean(sw[good], axis=0)
    b = np.nanmean(np.clip(np.sin(np.radians(el[good])), 0, None), axis=0)
    best, score = 0, -np.inf
    for s in range(-max_shift_steps, max_shift_steps + 1):
        c = np.corrcoef(np.roll(a, -s), b)[0, 1]
        if c > score:
            best, score = s, c
    return float(best * step_minutes)


# ---- Whole-record tests ------------------------------------------------------------------------------------------

def sign_check(flux: np.ndarray, driver: np.ndarray, expect: int, min_n: int = 500) -> Dict:
    """Pearson correlation of a flux with its driver over day-time steps where both are measured, and whether its
    sign contradicts the convention. `expect` is the sign the FLUXNET convention implies: NEE vs SW_IN is negative
    at vegetated sites (uptake in light); H and LE vs NETRAD are positive."""
    f, d = np.asarray(flux, np.float64), np.asarray(driver, np.float64)
    ok = np.isfinite(f) & np.isfinite(d)
    if ok.sum() < min_n:
        return {"n": int(ok.sum()), "r": None, "flipped": False}
    r = float(np.corrcoef(f[ok], d[ok])[0, 1])
    return {"n": int(ok.sum()), "r": r, "flipped": bool(np.sign(r) == -expect and abs(r) > 0.3)}


def fch4_unit_check(fch4: np.ndarray) -> Dict:
    """FCH4 must be in nmol m-2 s-1. A record in umol m-2 s-1 has a median absolute flux about 1000 times smaller:
    measured wetland and upland fluxes span about 0.1 to 1000 nmol m-2 s-1, so a median |FCH4| below 0.05 nmol
    m-2 s-1 with a 99th percentile below 1 is flagged as a likely unit error."""
    f = np.asarray(fch4, np.float64)
    f = np.abs(f[np.isfinite(f)])
    if f.size < 500:
        return {"n": int(f.size), "median_abs": None, "suspect_umol": False}
    med, p99 = float(np.median(f)), float(np.percentile(f, 99))
    return {"n": int(f.size), "median_abs": med, "p99_abs": p99, "suspect_umol": bool(med < 0.05 and p99 < 1.0)}


def regular(start_minutes: np.ndarray) -> Dict:
    """The record's step in minutes (the most common difference), and counts of duplicate and irregular
    TIMESTAMP_START values."""
    t = np.asarray(start_minutes, np.int64)
    d = np.diff(t)
    if d.size == 0:
        return {"step": None, "duplicates": 0, "irregular": 0}
    vals, counts = np.unique(d[d > 0], return_counts=True)
    step = int(vals[np.argmax(counts)]) if vals.size else None
    return {"step": step, "duplicates": int((d == 0).sum()), "irregular": int(((d != step) & (d != 0)).sum())}
