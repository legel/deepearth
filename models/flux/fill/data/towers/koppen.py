"""Koppen-Geiger class of each site: the network's own value where it gives one; else the Beck et al. (2023) 1-km map
for 1991-2020 (doi:10.6084/m9.figshare.21789074); else a class computed from the site's ERA5 normals.

Criteria of Beck et al. (2018, Sci. Data 5:180214, Table 1), with polar (E) taking precedence over arid (B), and B
over A, C and D (Kottek et al. 2006). Inputs are the 1991-2020 monthly means of the site-downscaled ERA5 air
temperature and monthly precipitation totals from the ONEFlux ERA5 MM file. Summer is April to September in the
northern hemisphere and October to March in the southern.
"""

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

BECK_CODES = ["Af", "Am", "Aw", "BWh", "BWk", "BSh", "BSk", "Csa", "Csb", "Csc", "Cwa", "Cwb", "Cwc", "Cfa", "Cfb", "Cfc",
              "Dsa", "Dsb", "Dsc", "Dsd", "Dwa", "Dwb", "Dwc", "Dwd", "Dfa", "Dfb", "Dfc", "Dfd", "ET", "EF"]
"""Legend of the Beck et al. (2023) maps: pixel value k is BECK_CODES[k - 1]; 0 is no data (sea)."""


def normalize(code: Optional[str]) -> Optional[str]:
    """Canonical capitalisation of a network-reported class ("Bsh" -> "BSh", "cfb" -> "Cfb"); None if invalid."""
    if not code or not isinstance(code, str):
        return None
    c = code.strip()
    if not c or c[0].upper() not in "ABCDE":
        return None
    c = c[0].upper() + (c[1:2].upper() if c[0].upper() in "BE" else c[1:2].lower()) + c[2:3].lower()
    return c if c in BECK_CODES else None


def from_map(tif: Path, lat: np.ndarray, lon: np.ndarray, search_px: int = 3) -> list:
    """Classes at the given points from a Beck et al. (2023) 1-km GeoTIFF. A point on a no-data pixel (a coastal or
    lake tower) takes the most common class within `search_px` pixels."""
    import rasterio
    out = []
    with rasterio.open(tif) as src:
        for la, lo in zip(lat, lon):
            r, c = src.index(lo, la)
            win = rasterio.windows.Window(c - search_px, r - search_px, 2 * search_px + 1, 2 * search_px + 1)
            a = src.read(1, window=win, boundless=True, fill_value=0)
            v = int(a[search_px, search_px])
            if v == 0:
                vals = a[a > 0]
                v = int(np.bincount(vals).argmax()) if vals.size else 0
            out.append(BECK_CODES[v - 1] if 0 < v <= len(BECK_CODES) else None)
    return out


def classify(t_month: Sequence[float], p_month: Sequence[float], lat: float) -> str:
    """Koppen-Geiger code (e.g. "Cfb", "BSk", "ET") from 12 monthly temperatures (deg C) and totals (mm)."""
    t, p = np.asarray(t_month, float), np.asarray(p_month, float)
    if t.size != 12 or p.size != 12 or not (np.isfinite(t).all() and np.isfinite(p).all()):
        raise ValueError("need 12 finite monthly values of temperature and precipitation")
    summer = np.zeros(12, bool)
    summer[3:9] = True
    if lat < 0:
        summer = ~summer
    mat, map_ = t.mean(), p.sum()
    t_hot, t_cold = t.max(), t.min()
    t_mon10 = int((t > 10).sum())
    p_dry = p.min()
    p_sdry, p_swet = p[summer].min(), p[summer].max()
    p_wdry, p_wwet = p[~summer].min(), p[~summer].max()
    if p[~summer].sum() > 0.7 * map_:
        p_thr = 2 * mat
    elif p[summer].sum() > 0.7 * map_:
        p_thr = 2 * mat + 28
    else:
        p_thr = 2 * mat + 14
    if t_hot < 10:
        return "ET" if t_hot > 0 else "EF"
    if map_ < 10 * p_thr:
        return ("BW" if map_ < 5 * p_thr else "BS") + ("h" if mat >= 18 else "k")
    if t_cold >= 18:
        if p_dry >= 60:
            return "Af"
        return "Am" if p_dry >= 100 - map_ / 25 else "Aw"
    first = "C" if t_cold > 0 else "D"
    if p_sdry < 40 and p_sdry < p_wwet / 3:
        second = "s"
    elif p_wdry < p_swet / 10:
        second = "w"
    else:
        second = "f"
    if t_hot >= 22:
        third = "a"
    elif t_mon10 >= 4:
        third = "b"
    elif first == "D" and t_cold < -38:
        third = "d"
    else:
        third = "c"
    return first + second + third
