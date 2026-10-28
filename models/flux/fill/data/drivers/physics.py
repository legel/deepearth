"""Conversions from source variables to FLUXNET variables and units (Pastorello et al. 2020, FP standard).

FLUXNET units: TA deg C, VPD hPa, RH %, SW_IN and LW_IN W m-2, P mm per time step, WS m s-1, PA kPa, SWC %,
TS deg C. Saturation vapour pressure over water follows Alduchov and Eskridge (1996), the Magnus form used by the
ECMWF documentation for 2 m humidity diagnostics.
"""

import numpy as np

G = 9.80665          # m s-2
RD = 287.05          # J kg-1 K-1
LAPSE = 0.0065       # K m-1, standard atmosphere
T0 = 273.15


def es_hpa(t_c):
    """Saturation vapour pressure over water (hPa) at temperature t_c (deg C)."""
    t_c = np.asarray(t_c, dtype=np.float64)
    return 6.1094 * np.exp(17.625 * t_c / (t_c + 243.04))


def vpd_hpa(ta_c, td_c):
    """Vapour pressure deficit (hPa) from air and dew-point temperature (deg C). Negative values (dew point above
    air temperature from interpolation) are set to 0."""
    v = es_hpa(ta_c) - es_hpa(np.minimum(td_c, ta_c))
    return np.maximum(v, 0.0).astype(np.float32)


def rh_pct(ta_c, td_c):
    return np.clip(100.0 * es_hpa(td_c) / es_hpa(ta_c), 0, 100).astype(np.float32)


def surface_pressure_kpa(msl_hpa, ta_c, z_m):
    """Surface pressure (kPa) at height z_m from mean-sea-level pressure, with a standard lapse rate below the
    surface: p = msl * (T / (T + L z)) ** (g / (Rd L)), T the 2 m temperature in K."""
    t_k = np.asarray(ta_c, dtype=np.float64) + T0
    p = np.asarray(msl_hpa, dtype=np.float64) * (t_k / (t_k + LAPSE * z_m)) ** (G / (RD * LAPSE))
    return (p / 10.0).astype(np.float32)


def pressure_to_height_kpa(pa_kpa, ta_c, z_from, z_to):
    """Move a surface pressure from height z_from to z_to (m) hydrostatically, standard lapse rate."""
    t_k = np.asarray(ta_c, dtype=np.float64) + T0          # temperature at z_from
    return (np.asarray(pa_kpa) * ((t_k - LAPSE * (z_to - z_from)) / t_k) ** (G / (RD * LAPSE))).astype(np.float32)


def wind_speed(u, v):
    return np.hypot(u, v).astype(np.float32)


def kelvin_to_c(t):
    return (np.asarray(t) - T0).astype(np.float32)


def fraction_to_pct(x):
    return (100.0 * np.asarray(x)).astype(np.float32)


def lapse_adjust_c(ta_c, z_cell, z_site, lapse=LAPSE):
    """Air temperature moved from the grid-cell orography to the tower elevation with a fixed lapse rate."""
    return (np.asarray(ta_c) - lapse * (z_site - z_cell)).astype(np.float32)
