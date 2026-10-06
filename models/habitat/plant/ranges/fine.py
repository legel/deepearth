"""Fine-scale predictors (future ledger L3 + L10): terrain from Copernicus GLO-90 and lapse-rate-corrected
WorldClim temperatures, at ~230 m over North/Central America for training and at 240 m on the CONUS grid
for rendering.

Stored layers (int16, scaled): elevation (m), slope (0.1°), northness and eastness (x1000; cos/sin of aspect
weighted by sin(slope)), topographic position index TPI (m; elevation minus the mean within ±4 cells, ~2 km).
Derived on the fly: for the WorldClim temperature variables bio1, 5, 6, 8, 9, 10, 11,
    T_fine = T_wc + LAPSE * (z_fine - z_wc),   LAPSE = -0.0065 °C/m (standard environmental lapse rate),
where T_wc and z_wc are WorldClim's ~1 km temperature and elevation at the point. Precipitation and the
temperature-range variables (bio2, 3, 4, 7) are not elevation-adjusted.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

LAPSE = -0.0065
TEMP_VARS = {f"wc2.1_30s_bio_{i}" for i in (1, 5, 6, 8, 9, 10, 11)}
FINE_VARS = ["fine_elev", "fine_slope", "fine_northness", "fine_eastness", "fine_tpi"]
SCALE = {"fine_elev": 1.0, "fine_slope": 0.1, "fine_northness": 0.001, "fine_eastness": 0.001, "fine_tpi": 1.0}
CELL = 7.5 / 3600.0
LON0, LAT1, NCOL, NROW = -170.0, 75.0, 57600, 33600       # 170°W–50°W, 5°N–75°N


def terrain(z: np.ndarray, dx_m: np.ndarray | float, dy_m: float) -> dict[str, np.ndarray]:
    """Slope (degrees), northness, eastness, TPI from an elevation block (rows north->south)."""
    zp = np.pad(z, 1, mode="edge")
    gx = (zp[1:-1, 2:] - zp[1:-1, :-2]) / (2 * np.asarray(dx_m).reshape(-1, 1) if np.ndim(dx_m) else 2 * dx_m)
    gy = (zp[:-2, 1:-1] - zp[2:, 1:-1]) / (2 * dy_m)            # north minus south
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    aspect = np.arctan2(-gx, -gy)                              # downslope direction, 0 = north
    s = np.sin(np.radians(slope))
    k = 4
    zp2 = np.pad(z, k, mode="edge")
    c = np.cumsum(np.cumsum(zp2, 0), 1)
    c = np.pad(c, ((1, 0), (1, 0)))
    w = 2 * k + 1
    box = (c[w:, w:] - c[:-w, w:] - c[w:, :-w] + c[:-w, :-w]) / (w * w)
    return {"fine_slope": slope, "fine_northness": np.cos(aspect) * s, "fine_eastness": np.sin(aspect) * s,
            "fine_tpi": z - box}


class FineStack:
    """North/Central America ~230 m fine layers (pixel-interleaved int16 memmap)."""

    def __init__(self, path: str | Path):
        self.data = np.memmap(path, dtype=np.int16, mode="r", shape=(NROW, NCOL, len(FINE_VARS)))

    @staticmethod
    def rowcol(lon, lat):
        r = np.floor((LAT1 - np.asarray(lat)) / CELL).astype(np.int64)
        c = np.floor((np.asarray(lon) - LON0) / CELL).astype(np.int64)
        return r, c

    def at(self, lon, lat) -> np.ndarray:
        """(N, 5) float64 fine layers; NaN outside the domain or over no-data."""
        r, c = self.rowcol(lon, lat)
        ok = (r >= 0) & (r < NROW) & (c >= 0) & (c < NCOL)
        out = np.full((len(r), len(FINE_VARS)), np.nan)
        v = np.asarray(self.data[r[ok], c[ok], :]).astype(np.float64)
        v[v == -32768] = np.nan
        out[ok] = v * np.array([SCALE[n] for n in FINE_VARS])
        return out


def fine_predictors(wc_values: np.ndarray, wc_names: list[str], fine_values: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Combine WorldClim values (N, V incl. wc2.1_30s_elev) with fine layers: lapse-rate-correct the temperature
    variables, replace WorldClim elevation by fine elevation, append terrain. Returns (N, V + 4) and names."""
    x = np.array(wc_values, dtype=np.float64, copy=True)
    zi = wc_names.index("wc2.1_30s_elev")
    dz = fine_values[:, 0] - x[:, zi]
    names = []
    for j, n in enumerate(wc_names):
        if n in TEMP_VARS:
            x[:, j] = x[:, j] + LAPSE * dz
            names.append(n.replace("wc2.1_30s_", "fine_"))
        elif n == "wc2.1_30s_elev":
            x[:, j] = fine_values[:, 0]
            names.append("fine_elev")
        else:
            names.append(n)
    return np.column_stack([x, fine_values[:, 1:]]), names + FINE_VARS[1:]
