"""Source codes and the measured-only rule every driver follows.

A flux tower's hour carries a QC flag: FLUXNET's 0 is measured; 1 to 3 are its own gap-fills (MDS, then ERA5
downscaled). A gap-filled hour is shown as the tower published it, but it never counts as measured: the forcing
takes the next source for it, and the hour carries that source's code.
"""

from typing import Iterable, Optional, Tuple

import numpy as np

TOWER, FILL, OBSERVED, REANALYSIS, TOWER_PAR, REFERENCE, STATION_CALM = 0, 1, 2, 3, 5, 6, 7
NONE = 255
CODES = {TOWER: "tower", FILL: "fill: station, satellite or analysis in the tower's gap", OBSERVED: "observed",
         REANALYSIS: "reanalysis", TOWER_PAR: "tower, SW_IN from its PAR sensor at a fixed ratio",
         REFERENCE: "reference gauge (WMO double-fence or shielded weighing gauge)",
         STATION_CALM: "station calm: below its anemometer's 3 kt threshold, read as 0 m/s", NONE: "none"}
"""Each hour's source, as published beside the value."""

MEASURED_QC = 0


def measured(values: np.ndarray, qc: Optional[np.ndarray]) -> np.ndarray:
    """`values` where the tower measured the hour (QC 0 and a value), NaN elsewhere. No QC column: every value counts."""
    v = np.asarray(values, np.float64)
    if qc is None:
        return np.where(np.isfinite(v), v, np.nan)
    return np.where(np.isfinite(v) & (np.nan_to_num(np.asarray(qc, np.float64), nan=9.0) == MEASURED_QC), v, np.nan)


def layer(n: int, sources: Iterable[Tuple[Optional[np.ndarray], int]]) -> Tuple[np.ndarray, np.ndarray]:
    """(values, codes): each hour from the first source in order that has a value, with that source's code; NONE where
    none has."""
    out = np.full(n, np.nan)
    code = np.full(n, NONE, np.uint8)
    for arr, c in sources:
        if arr is None:
            continue
        a = np.asarray(arr, np.float64)
        take = np.isnan(out) & np.isfinite(a)
        out[take], code[take] = a[take], c
    return out, code
