"""The tower's wind speed at the site's reference height, and a station's calm read as what it is.

The sonic's speed is carried to the reference (10 m over the site's own roughness) through the blending height
(Wieringa 1986), with the tower fetch's roughness per 30-degree sector fitted from its own near-neutral hours and
d = 2/3 of its canopy height. A station's calm report (ASOS: below 3 kt) in the tower's gap is kept at the station's
0 but carries its own code, never "measured still air".
"""

from typing import Optional

import numpy as np

from qc import FILL, OBSERVED, STATION_CALM

KAPPA = 0.41
BLEND_M, REF_M = 60.0, 10.0
CALM_KT, KNOT = 3.0, 0.514444
"""ASOS reports speeds below 3 kt (1.54 m/s) as calm, 0 kt."""


def obukhov(ustar: np.ndarray, h: np.ndarray, ta: np.ndarray, pa_kpa: np.ndarray) -> np.ndarray:
    """Obukhov length, m (sensible heat only)."""
    rho = pa_kpa * 1000.0 / (287.05 * (ta + 273.15))
    with np.errstate(divide="ignore", invalid="ignore"):
        return -(ustar ** 3) * rho * 1005.0 * (ta + 273.15) / (KAPPA * 9.81 * h)


def z0_by_sector(ws, ustar, wd, h, ta, pa, z_m: float, d_m: float, sectors: int = 12, min_hours: int = 50) -> np.ndarray:
    """Fetch roughness per sector from the tower's near-neutral hours (|z/L| < 0.05, u* > 0.2):
    z0 = (z - d) exp(-kappa U / u*), the median per sector; a sector with too few hours takes the median of the rest."""
    L = obukhov(ustar, h, ta, pa)
    neutral = (np.abs((z_m - d_m) / L) < 0.05) & (ustar > 0.2) & ~np.isnan(ws) & ~np.isnan(wd)
    z0 = (z_m - d_m) * np.exp(-KAPPA * ws / ustar)
    sec = (np.floor(np.mod(wd, 360.0) / (360.0 / sectors)).astype(int)) % sectors
    out = np.full(sectors, np.nan)
    for s in range(sectors):
        sel = neutral & (sec == s)
        if sel.sum() >= min_hours:
            out[s] = float(np.median(z0[sel]))
    if np.isnan(out).all():
        return out
    return np.where(np.isnan(out), np.nanmedian(out), out)


def transfer(ws: np.ndarray, z_m: float, z0_tower, d_tower: float, z0_site: float, d_site: float,
             z_blend: float = BLEND_M, z_ref: float = REF_M) -> np.ndarray:
    """Tower speed at z_m to the reference height over the site's roughness, through the blending height:
    u_b = u ln((z_b - d_t)/z0_t) / ln((z_m - d_t)/z0_t),  u_ref = u_b ln((z_ref - d_s)/z0_s) / ln((z_b - d_s)/z0_s)."""
    ub = ws * np.log((z_blend - d_tower) / z0_tower) / np.log((z_m - d_tower) / z0_tower)
    return ub * np.log((z_ref - d_site) / z0_site) / np.log((z_blend - d_site) / z0_site)


def calm_codes(ws: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """`codes` with every station hour (observed or fill) whose speed reads exactly 0 re-coded STATION_CALM."""
    calm = np.isin(codes, (OBSERVED, FILL)) & (np.asarray(ws, np.float64) == 0.0)
    return np.where(calm, STATION_CALM, codes).astype(np.asarray(codes).dtype)


def calm_ceiling_m_s(ratio_to_station: Optional[float] = None) -> float:
    """The speed a calm hour is below: 3 kt at the station, times the station-to-point ratio where one is given."""
    return CALM_KT * KNOT * (ratio_to_station if ratio_to_station is not None else 1.0)
