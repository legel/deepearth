"""Design-storm ensemble to per-cell flood probability.

Atlas 14 gives, per coordinate, the rainfall depth at each return period T, so a storm of
return period T has annual exceedance probability 1/T. Run the solver once per T, and each cell
gets a peak depth that rises monotonically with T. Inverting that curve at a depth threshold
gives the AEP of the smallest storm that floods the cell, and

    P(at least one in N years) = 1 - (1 - AEP)^N

converts it to any horizon. This is the physics solver wrapped in frequency analysis, not a
learned model, and it is stationary: Atlas 14 carries no climate trend, so these are
present-day probabilities.

Because Atlas 14 is queried per coordinate, the same ensemble runs unchanged anywhere -- which
is what makes comparing one site against another mean anything.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from forcing import RETURN_PERIODS_YR


@dataclass
class AEPSummary:
    """Areas at risk, in hectares, plus the diagnostics needed to trust them."""

    area_any_risk_ha: float
    area_gt_1pct_ha: float
    area_gt_10pct_ha: float
    cells_total: int
    cells_at_risk: int
    monotonicity_fixed_cells: int
    loglinearity_r2: float
    threshold_m: float
    duration_hr: float

    def as_dict(self) -> Dict[str, float]:
        """Plain numbers, for writing alongside a run."""
        return {k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
                for k, v in self.__dict__.items()}


def loglinearity_r2(depths_by_t: Dict[float, float]) -> float:
    """R^2 of depth against log(T), the assumption the interpolation below rests on.

    Report it. If it is poor, interpolating between simulated return periods is not justified
    and the ensemble needs more of them.
    """
    t = np.array(sorted(depths_by_t), dtype=float)
    d = np.array([depths_by_t[k] for k in sorted(depths_by_t)], dtype=float)
    slope, intercept = np.polyfit(np.log(t), d, 1)
    resid = d - (slope * np.log(t) + intercept)
    ss_res, ss_tot = float((resid ** 2).sum()), float(((d - d.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def depth_stack_to_aep(stack: np.ndarray, threshold_m: float,
                       return_periods_yr: Optional[Sequence[float]] = None
                       ) -> Tuple[np.ndarray, int]:
    """Invert a peak-depth-by-return-period stack to a per-cell annual exceedance probability.

    Args:
        stack: Peak depth [m], shape (n_return_periods, rows, cols), ordered as
            `return_periods_yr`.
        threshold_m: Depth at or above which a cell counts as flooded.
        return_periods_yr: Return periods matching the stack's first axis.

    Returns:
        (aep array, number of cells whose depth curve needed monotonicity enforced).
    """
    t = np.array(return_periods_yr or RETURN_PERIODS_YR, dtype=float)
    assert stack.shape[0] == len(t), f"stack has {stack.shape[0]} layers, {len(t)} return periods"
    log_t = np.log(t)
    _, ny, nx = stack.shape

    # Depth must rise with T for the inversion to be well posed. The solver is nonlinear and a
    # few cells invert by a hair; a cumulative max fixes those rather than failing on them. The
    # count is returned because a large one means something real is wrong.
    mono = np.maximum.accumulate(stack, axis=0)
    fixed = int((mono != stack).any(axis=0).sum())

    exceeds = mono >= threshold_m
    ever = exceeds.any(axis=0)
    first = np.argmax(exceeds, axis=0)
    aep = np.zeros((ny, nx), dtype=np.float32)

    # Already flooded by the most frequent storm simulated: clamp at its AEP. Nothing more
    # frequent than the shortest return period in the ensemble can be resolved.
    aep[ever & (first == 0)] = 1.0 / t[0]

    interp = ever & (first > 0)
    if interp.any():
        i1 = first[interp]
        i0 = i1 - 1
        yy, xx = np.nonzero(interp)
        d0, d1 = mono[i0, yy, xx], mono[i1, yy, xx]
        denom = np.where((d1 - d0) > 1e-12, d1 - d0, np.nan)
        frac = np.nan_to_num(np.clip((threshold_m - d0) / denom, 0.0, 1.0), nan=0.0)
        aep[yy, xx] = (1.0 / np.exp(log_t[i0] + frac * (log_t[i1] - log_t[i0]))).astype(np.float32)

    return aep, fixed


def aep_to_horizon(aep: np.ndarray, years: float) -> np.ndarray:
    """P(at least one exceedance in `years`) = 1 - (1 - AEP)^years.

    A 1 %-AEP cell returns 0.2603 over 30 years, which is FEMA's own published "26 % chance
    over a 30-year mortgage" -- an independent check that the frequency arithmetic is right.
    """
    return 1.0 - np.power(1.0 - np.clip(aep, 0.0, 1.0), float(years))


def summarize(aep: np.ndarray, cell_area_m2: float, threshold_m: float, duration_hr: float,
              monotonicity_fixed_cells: int = 0, r2: float = float("nan")) -> AEPSummary:
    """Aggregate an AEP surface into the areas worth quoting."""
    ha = cell_area_m2 / 1e4
    return AEPSummary(
        area_any_risk_ha=float((aep > 0).sum()) * ha,
        area_gt_1pct_ha=float((aep >= 0.01).sum()) * ha,
        area_gt_10pct_ha=float((aep >= 0.10).sum()) * ha,
        cells_total=int(aep.size),
        cells_at_risk=int((aep > 0).sum()),
        monotonicity_fixed_cells=monotonicity_fixed_cells,
        loglinearity_r2=r2,
        threshold_m=threshold_m,
        duration_hr=duration_hr,
    )
