"""Flood PROBABILITY from a design-storm ensemble — P(flood) ∈ [0, 1].

Every solver in these projects produces a deterministic depth field for one storm: "given this
rainfall, this cell ends up 0.28 m deep." That is a hydraulic answer, not a risk answer. This
module converts an ensemble of such answers into an annual exceedance probability, and then to
any time horizon.

It lives in the shared library because the method must be IDENTICAL at every site for
cross-site comparison to mean anything. The inputs differ per site (its own Atlas 14 depths,
its own DEM, its own solver); the frequency analysis does not.

Method — each step is standard practice, not invented here
----------------------------------------------------------
1. NOAA Atlas 14 gives, for a coordinate, the rainfall depth at each return period T. By
   definition T has annual exceedance probability AEP = 1/T.
2. Run the calibrated solver once per return period, driven by that period's design
   hyetograph. This yields a per-cell peak-depth curve h_T(x, y), monotonically increasing
   in T — a rarer storm never floods a cell less.
3. Invert per cell: for a flood threshold d*, the AEP is that of the smallest storm reaching
   d*,  AEP = 1/T*  where  T* = min{T : h_T ≥ d*}. Between simulated return periods,
   interpolate depth against log(T) — Atlas 14 depth is close to linear in log T (verify per
   site with loglinearity_r2() rather than assuming).
4. Convert to a horizon by treating annual exceedances as independent Bernoulli trials, the
   standard assumption behind "1% annual chance" language.

What this is NOT
----------------
A learned model. This is the physics solver run once per return period, wrapped in frequency
analysis. It gives the right output type and a defensible number.

It is also NOT nonstationary: Atlas 14 carries no climate trend, so these are present-day
probabilities.
"""
import numpy as np

# Return periods to simulate. The 1-yr anchors the frequent end: without it, every cell that
# floods even in a common storm saturates at the 2-yr AEP of 0.5 and the map loses all
# resolution in the high-probability range — exactly the range that matters most.
RETURN_PERIODS_YR = [1, 2, 5, 10, 25, 50, 100, 200, 500]

# Defaults shared by every site so surfaces are directly comparable. Override per run only
# with a stated reason.
DEFAULT_DURATION_HR = 24.0   # standard for flood design
DEFAULT_THRESHOLD_M = 0.15   # depth at which a cell counts as "flooded"


def loglinearity_r2(depths_by_T):
    """R² of Atlas 14 depth against log(T) — reported, not asserted.

    Step 3 interpolates in log T. This quantifies how well that holds at a given site, so a
    site where it does not can be spotted rather than silently mis-interpolated.

    depths_by_T : mapping {return_period_yr: depth_mm}
    """
    T = np.array(sorted(depths_by_T), dtype=float)
    d = np.array([depths_by_T[t] for t in sorted(depths_by_T)], dtype=float)
    x = np.log(T)
    slope, intercept = np.polyfit(x, d, 1)
    pred = slope * x + intercept
    ss_res = float(((d - pred) ** 2).sum())
    ss_tot = float(((d - d.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def depth_stack_to_aep(stack, threshold_m, return_periods_yr=None, verbose=True):
    """Per-cell annual exceedance probability that peak depth reaches `threshold_m`.

    stack[i] is the peak depth field under return_periods_yr[i], increasing in i.

    Three cases per cell:
      * threshold reached even by the most frequent storm -> AEP = 1/T_min, clamped (nothing
        more frequent than the shortest return period simulated can be resolved)
      * threshold never reached even by the rarest storm  -> AEP = 0 (below resolvable risk)
      * otherwise -> log-linear interpolation in T between the bracketing return periods

    Returns a float32 array shaped like one depth field.
    """
    T = np.array(return_periods_yr or RETURN_PERIODS_YR, dtype=float)
    if stack.shape[0] != len(T):
        raise ValueError(f"stack has {stack.shape[0]} layers but {len(T)} return periods given")
    logT = np.log(T)
    ny, nx = stack.shape[1:]

    # Enforce monotonicity in T. Physically guaranteed, but the solvers are nonlinear (adaptive
    # dt, Froude cap), so a few cells can invert by a hair; a cumulative max makes the inversion
    # well-posed everywhere instead of failing on those cells.
    mono = np.maximum.accumulate(stack, axis=0)
    n_fixed = int((mono != stack).any(axis=0).sum())
    if n_fixed and verbose:
        print(f"  monotonicity enforced on {n_fixed:,} cells "
              f"({100 * n_fixed / (ny * nx):.3f}% — nonlinear-solver jitter, expected to be tiny)")

    exceeds = mono >= threshold_m
    ever = exceeds.any(axis=0)
    first = np.argmax(exceeds, axis=0)          # index of the smallest T that exceeds

    aep = np.zeros((ny, nx), dtype=np.float32)

    # Case A — already exceeded at the most frequent storm; clamp at its AEP.
    aep[ever & (first == 0)] = 1.0 / T[0]

    # Case B — interpolate between the bracketing return periods.
    interp = ever & (first > 0)
    if interp.any():
        i1 = first[interp]
        i0 = i1 - 1
        yy, xx = np.nonzero(interp)
        d0, d1 = mono[i0, yy, xx], mono[i1, yy, xx]
        denom = np.where((d1 - d0) > 1e-12, d1 - d0, np.nan)
        frac = np.nan_to_num(np.clip((threshold_m - d0) / denom, 0.0, 1.0), nan=0.0)
        logT_star = logT[i0] + frac * (logT[i1] - logT[i0])
        aep[yy, xx] = (1.0 / np.exp(logT_star)).astype(np.float32)

    # Case C — never exceeded; stays 0.0.
    return aep


def aep_to_horizon(aep, years):
    """P(at least one exceedance in `years`) = 1 - (1 - AEP)^years.

    Sanity check: a 1%-AEP cell over 30 years gives 0.2603, matching FEMA's published
    "26% chance over a 30-year mortgage".
    """
    return 1.0 - np.power(1.0 - np.clip(aep, 0.0, 1.0), float(years))


def summarize_aep(aep, cell_area_m2):
    """Headline areas from an AEP surface, as a dict ready to serialize."""
    ha = cell_area_m2 / 10_000.0
    return {
        "cells_total": int(aep.size),
        "cells_with_nonzero_risk": int((aep > 0).sum()),
        "area_nonzero_risk_ha": round(float((aep > 0).sum() * ha), 2),
        "area_gt_1pct_annual_ha": round(float((aep >= 0.01).sum() * ha), 2),
        "area_gt_10pct_annual_ha": round(float((aep >= 0.10).sum() * ha), 2),
    }
