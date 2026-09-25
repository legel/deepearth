"""Precipitation from gauges: one rule for any site with one or several.

    1. The reference gauge (a WMO double-fence or shielded weighing gauge, final QC 0) where it measured the hour.
    2. Otherwise the tower's gauge (QC 0), vetted against the site's independent sources (the other gauges, the
       gridded analysis, the reference):
         an hour is kept only if  p <= RAIN_HOUR_FACTOR x max(independent, 3 h window) + RAIN_HOUR_FLOOR_MM,
         a month is kept only if  |sum(p) - median(sums)| <= RAIN_MONTH_BAND x median(sums)
         (sums over the same hours, each independent source present in more than 80 % of them).
    3. Otherwise the gridded analysis (AORC).

A site with one gauge and no reference is vetted against the analysis alone; with no independent source the gauge
stands. Neither check needs a gauge's own flag: at Harvard Forest the tower-top bucket was flagged good throughout
while it undercaught a summer and overcaught convective half hours (README).
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from qc import FILL, REFERENCE, REANALYSIS, TOWER, layer, measured

RAIN_HOUR_FACTOR = 2.0
RAIN_HOUR_FLOOR_MM = 1.0
RAIN_MONTH_BAND = 0.35
RAIN_MONTH_MIN_MM = 20.0
RAIN_MONTH_COVER = 0.8


def window_max(a: np.ndarray, half: int = 1) -> np.ndarray:
    """Each hour's largest value over the hours within `half` of it (NaN where none has a value)."""
    pad = np.pad(np.nan_to_num(np.asarray(a, np.float64), nan=-1.0), half, constant_values=-1.0)
    w = np.max(np.stack([pad[k:len(pad) - 2 * half + k] for k in range(2 * half + 1)]), axis=0)
    return np.where(w < 0, np.nan, w)


def vet_rain(p: np.ndarray, others: Sequence[np.ndarray], month: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """A gauge's hourly rain (mm) with every hour and month its independent sources do not support removed.

    Args:
        p: The gauge's hourly rain, its measured hours only (NaN elsewhere).
        others: The site's independent sources on the same hours (mm; NaN where absent).
        month: Each hour's calendar month, any integer label.
    Returns:
        (vetted rain, {"hours_out": n, "months_out": [labels]}).
    """
    ind = [np.asarray(o, np.float64) for o in others if o is not None and np.isfinite(o).any()]
    p = np.asarray(p, np.float64).copy()
    if not ind:
        return p, {"hours_out": 0, "months_out": []}
    win = np.stack([window_max(o) for o in ind])
    have = np.isfinite(win).any(axis=0)
    cap = RAIN_HOUR_FACTOR * np.nanmax(np.where(np.isfinite(win), win, -np.inf), axis=0) + RAIN_HOUR_FLOOR_MM
    spike = have & np.isfinite(p) & (p > cap)
    p[spike] = np.nan
    out = []
    for m in np.unique(month):
        sel = (month == m) & np.isfinite(p)
        if not sel.any():
            continue
        sums = [float(np.sum(o[sel & np.isfinite(o)])) for o in ind if np.isfinite(o[sel]).mean() > RAIN_MONTH_COVER]
        if not sums:
            continue
        ref, mine = float(np.median(sums)), float(np.sum(p[sel]))
        if max(ref, mine) >= RAIN_MONTH_MIN_MM and abs(mine - ref) > RAIN_MONTH_BAND * max(ref, 1e-9):
            p[month == m] = np.nan
            out.append(int(m))
    return p, {"hours_out": int(spike.sum()), "months_out": out}


def rain(gauge: Optional[np.ndarray], gauge_qc: Optional[np.ndarray], month: np.ndarray,
         reference: Optional[np.ndarray] = None, analysis: Optional[np.ndarray] = None,
         other_gauges: Sequence[np.ndarray] = ()) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """The site's hourly rain under the rule: (mm, source code per hour, the vetting record)."""
    n = len(month)
    rec: Dict = {"hours_out": 0, "months_out": []}
    p_t = None
    if gauge is not None:
        p_t, rec = vet_rain(measured(gauge, gauge_qc), [*other_gauges, analysis, reference], month)
    p, code = layer(n, [(reference, REFERENCE), (p_t, TOWER),
                        (analysis, FILL if (p_t is not None or reference is not None) else REANALYSIS)])
    return p, code, rec
