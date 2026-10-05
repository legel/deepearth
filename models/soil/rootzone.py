"""The woody root zone's depth from the site's measured water storage.

A tree's roots reach far below FAO-56's lower bound (1.0 m): at Tonzi Ranch blue oaks take up to 80 % of summer ET from
groundwater 8 to 10.5 m down (Miller, Chen, Rubin, Ma and Baldocchi 2010, Water Resour. Res. 46, W10503), and a 1 m
bucket left them nothing by July. The storage the vegetation actually draws is measured from space: S_CWDX80, the largest
cumulative water deficit (ET less rain) the vegetation sustains, 80th percentile of yearly maxima, at 0.05 degrees
(Stocker et al. 2023, Nat. Geosci. 16:250-256; Zenodo 10885724, CC BY 4.0). A woody cell's root zone holds that storage
on the cell's own soil,

    z_r = clip(max(z_class, S / (1000 (fc - wp))), z_class, WOODY_ROOT_MAX_M),

never shallower than its cover, never deeper than the trees' mean maximum rooting depth (Canadell et al. 1996, Oecologia
108:583-595: 7.0 +/- 1.2 m). Nothing is fitted to a tower. Grass and shrubs keep their cover's depth. An urban or masked
pixel (UC Berkeley's campus: 8 mm) leaves the cover's depth.
"""
from typing import Dict, Optional

import numpy as np

WOODY_ROOT_MAX_M = 7.0


def storage_mm(lat: float, lon: float, lat_axis: np.ndarray, lon_axis: np.ndarray, value: np.ndarray) -> Dict:
    """The site's 0.05-degree pixel of S_CWDX80 (mm) from a grid (lat_axis x lon_axis), or {"mm": None, "why": ...}."""
    lat_a, lon_a, v = np.asarray(lat_axis), np.asarray(lon_axis), np.asarray(value)
    if not (lat_a.min() - 0.025 <= lat <= lat_a.max() + 0.025 and lon_a.min() - 0.025 <= lon <= lon_a.max() + 0.025):
        return {"mm": None, "why": "outside the grid's extent"}
    i, j = int(np.argmin(np.abs(lat_a - lat))), int(np.argmin(np.abs(lon_a - lon)))
    s = float(v[i, j])
    if not np.isfinite(s) or s <= 0:
        return {"mm": None, "why": "the site's pixel has no value"}
    return {"mm": round(s, 1), "pixel": [round(float(lat_a[i]), 3), round(float(lon_a[j]), 3)],
            "source": "Stocker et al. 2023 S_CWDX80 (Zenodo 10885724)"}


def woody_depth(z_class: np.ndarray, fc: np.ndarray, wp: np.ndarray, storage: Optional[float]) -> np.ndarray:
    """Each woody cell's root depth (m) holding `storage` mm on its own soil, within its cover's depth and
    WOODY_ROOT_MAX_M. No storage: the cover's depth."""
    z_class = np.asarray(z_class, np.float64)
    if not storage:
        return z_class
    held = storage / (1000.0 * np.maximum(np.asarray(fc, np.float64) - np.asarray(wp, np.float64), 1e-3))
    return np.clip(np.maximum(z_class, held), z_class, WOODY_ROOT_MAX_M)
