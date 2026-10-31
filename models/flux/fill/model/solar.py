"""Solar geometry of a tower: sine of the solar altitude and normalised top-of-atmosphere radiation per half-hour.

Port to numpy of WeatherNext (google-deepmind/graphcast, weathernext/utils/solar_radiation.py, commit in
docs/REFERENCES.md): `_get_orbital_parameters`, `_get_solar_sin_altitude`, `_get_radiation_flux`, without the
integration over the time step and with the mean total solar irradiance, so the flux is
max(sin altitude, 0) / distance^2, in units of the solar constant.
"""

import numpy as np
import pandas as pd

J2000_EPOCH = 2451545.0                  # Julian date of 2000-01-01 12:00
JULIAN_YEAR = 365.25
SECONDS_PER_DAY = 86400.0


def orbital(j2000_days: np.ndarray):
    """Sine and cosine of the solar declination, equation of time (s), distance (au), rotational phase."""
    theta = j2000_days / JULIAN_YEAR
    phase = j2000_days % 1.0
    rel = 1.7535 + 6.283076 * theta
    rem = 6.240041 + 6.283020 * theta
    rlls = 4.8951 + 6.283076 * theta
    one = np.ones_like(theta)
    sin_rel, cos_rel = np.sin(rel), np.cos(rel)
    rllls = (np.stack([one, theta, sin_rel, cos_rel, np.sin(2 * rel), np.cos(2 * rel)], -1)
             @ np.array([4.8952, 6.283320, -0.0075, -0.0326, -0.0003, 0.0002]))
    sin_dec = np.sin(0.409093) * np.sin(rllls)
    cos_dec = np.sqrt(1.0 - sin_dec ** 2)
    sin_rem = np.sin(rem)
    eq_time = (np.stack([np.sin(2 * rlls), sin_rem, sin_rem * np.cos(2 * rlls), np.sin(4 * rlls), np.sin(2 * rem)], -1)
               @ np.array([591.8, -459.4, 39.5, -12.7, -4.8]))
    dist = np.stack([one, sin_rel, cos_rel], -1) @ np.array([1.0001, -0.0163, 0.0037])
    return sin_dec, cos_dec, eq_time, dist, phase


def sun_features(start_lst: pd.Timestamp, n_days: int, lat: float, lon: float, utc_offset_h: float) -> np.ndarray:
    """(n_days, 96) float32: sine of the solar altitude (48 values) and normalised top-of-atmosphere radiation (48
    values) at the middle of each half-hour of the days from `start_lst` (local standard time). Zeros without a
    location."""
    if not np.isfinite(lat) or not np.isfinite(lon):
        return np.zeros((n_days, 96), np.float32)
    day0 = pd.Timestamp(start_lst).to_julian_date() - utc_offset_h / 24.0 - J2000_EPOCH
    days = day0 + (np.arange(n_days * 48) + 0.5) / 48.0
    sin_dec, cos_dec, eq_time, dist, phase = orbital(days)
    la, lo = np.radians(lat), np.radians(lon)
    hour_angle = 2.0 * np.pi * (phase + eq_time / SECONDS_PER_DAY) + lo
    sin_alt = np.cos(la) * cos_dec * np.cos(hour_angle) + np.sin(la) * sin_dec
    toa = np.maximum(sin_alt, 0.0) / dist ** 2
    return np.concatenate([sin_alt.reshape(n_days, 48), toa.reshape(n_days, 48)], -1).astype(np.float32)
