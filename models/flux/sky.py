"""Shortwave from the tower: its clear-sky index against the measured atmosphere's clear sky.

k_c = SW_IN / GHI_cs, with GHI_cs the REST2 clear sky driven by MERRA-2 aerosol and water vapour (NSRDB's
clearsky_ghi). Night and horizon hours (GHI_cs <= KC_MIN_GHI_CS) have no index. A daytime hour no source measured is
filled by interpolating k_c, not SW_IN, between the nearest measured hours, so the fill follows the sun's course through
the day; night stays 0. A filled hour is marked, never counted as measured.
"""

from typing import Tuple

import numpy as np

KC_MIN_GHI_CS = 10.0
"""W m-2: below this clear-sky irradiance an hour has no sky coefficient."""


def clear_sky_index(ghi: np.ndarray, ghi_cs: np.ndarray) -> np.ndarray:
    """k_c per hour; NaN at night and where no irradiance was measured."""
    ghi, ghi_cs = np.asarray(ghi, np.float64), np.asarray(ghi_cs, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ghi_cs > KC_MIN_GHI_CS, ghi / ghi_cs, np.nan)


def fill_ghi(ghi: np.ndarray, ghi_cs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(SW_IN with every daytime gap filled by k_c interpolation, the filled hours)."""
    ghi, ghi_cs = np.asarray(ghi, np.float64), np.asarray(ghi_cs, np.float64)
    day = ghi_cs > KC_MIN_GHI_CS
    have = np.isfinite(ghi) & day
    gap = ~np.isfinite(ghi) & day
    out = np.where(np.isfinite(ghi), ghi, 0.0)
    if gap.any():
        if not have.any():
            raise ValueError("no daytime hour has irradiance from any source")
        i = np.arange(len(ghi))
        out[gap] = np.interp(i[gap], i[have], ghi[have] / ghi_cs[have]) * ghi_cs[gap]
    return out, gap
