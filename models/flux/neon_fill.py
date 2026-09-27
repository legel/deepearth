"""The months a tower's FLUXNET release has not published, from the same tower's NEON measurements.

A NEON tower (US-xHA is NEON's Harvard tower) keeps measuring after its FLUXNET release ends. Where the release has no
measured step (absent, or gap-filled QC 1 to 3), NEON's own measurement of the same quantity at the same position takes
it, QC 0; a measured step is never replaced. Only instruments at the release's own height qualify: the sunshine
pyranometer (39.25 m beside FLUXNET's 39.2) and the tipping bucket (39.28, FLUXNET's own P). Wind waits for the 3D sonic:
NEON's 2D wind tops out at 28.91 m and is never merged into the 39.16 m series.
"""

from typing import Tuple

import numpy as np

from qc import MEASURED_QC


def fill(t: np.ndarray, values: np.ndarray, qc: np.ndarray, t_neon: np.ndarray, v_neon: np.ndarray
         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """(steps, values, QC, steps taken): the release's series on its own steps, grown to NEON's steps past it, with
    every step it did not measure taken from NEON where NEON measured it (finite, its final flag already 0).

    Args:
        t: The release's step starts (seconds), sorted.
        values, qc: Its values and QC (255 or NaN where absent).
        t_neon, v_neon: NEON's step starts at the same interval, and its measured values (NaN where flagged).
    """
    ok = np.isfinite(v_neon)
    t_neon, v_neon = np.asarray(t_neon)[ok], np.asarray(v_neon, np.float64)[ok]
    steps = np.union1d(t, t_neon)
    v = np.full(len(steps), np.nan)
    q = np.full(len(steps), 255.0)
    pos = np.searchsorted(steps, t)
    v[pos], q[pos] = values, qc
    i = np.searchsorted(steps, t_neon)
    take = ~(np.isfinite(v[i]) & (q[i] == MEASURED_QC))
    v[i[take]], q[i[take]] = v_neon[take], MEASURED_QC
    return steps, v, q, int(take.sum())
