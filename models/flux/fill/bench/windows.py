"""A 180-day window of one tower as a (V, T) field: values, a measured flag and a provenance per variable.

The tower table is the dataset's per-site half-hourly table (data/towers): `<VAR>` values with `<VAR>_flag`, where
flag 0 is a measured value that passed every quality rule; `ERA5_<VAR>` are reanalysis drivers, measured wherever
finite. Times are local standard time (`t_lst`), the convention of FLUXNET products, so a day's 48 steps are a
solar day at the site.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from masks import STEPS_PER_DAY, WINDOW_DAYS

TOWER_VARS = ("NEE", "LE", "H", "FCH4", "G", "TA", "SW_IN", "LW_IN", "VPD", "PA", "P", "WS", "USTAR", "SWC_1", "TS_1")
REANALYSIS_VARS = ("ERA5_TA", "ERA5_SW_IN", "ERA5_LW_IN", "ERA5_VPD", "ERA5_PA", "ERA5_P", "ERA5_WS")
UNITS = {"NEE": "µmol m⁻² s⁻¹", "LE": "W m⁻²", "H": "W m⁻²", "FCH4": "nmol m⁻² s⁻¹", "G": "W m⁻²", "TA": "°C",
         "SW_IN": "W m⁻²", "LW_IN": "W m⁻²", "VPD": "hPa", "PA": "kPa", "P": "mm", "WS": "m s⁻¹", "USTAR": "m s⁻¹",
         "SWC_1": "%", "TS_1": "°C"}
LABELS = {"NEE": "Net ecosystem exchange", "LE": "Latent heat", "H": "Sensible heat", "FCH4": "Methane flux",
          "G": "Ground heat", "TA": "Air temperature", "SW_IN": "Shortwave in", "LW_IN": "Longwave in",
          "VPD": "Vapour pressure deficit", "PA": "Air pressure", "P": "Precipitation", "WS": "Wind speed",
          "USTAR": "Friction velocity", "SWC_1": "Soil water content", "TS_1": "Soil temperature"}


def lst_index(table: pd.DataFrame) -> pd.DatetimeIndex:
    """Local standard time of each row: `time_lst` (final tables) or `t_lst` in minutes since 1970 (interim)."""
    if "time_lst" in table:
        return pd.DatetimeIndex(table["time_lst"])
    x = table["t_lst"]
    return pd.DatetimeIndex(pd.to_datetime(x, unit="m") if np.issubdtype(x.dtype, np.integer) else x)


@dataclass
class Window:
    site: str
    start: pd.Timestamp
    t: pd.DatetimeIndex
    keys: List[str]
    values: np.ndarray             # (V, T) float64, NaN where no value
    measured: np.ndarray           # (V, T) bool
    provenance: List[str]          # "tower" or "reanalysis" per row
    meta: Dict = field(default_factory=dict)

    def rows(self, provenance: str) -> List[int]:
        return [i for i, p in enumerate(self.provenance) if p == provenance]


def load_window(table: pd.DataFrame, site: str, start, tower_vars: Sequence[str] = TOWER_VARS,
                reanalysis_vars: Sequence[str] = REANALYSIS_VARS) -> Window:
    """Cut the window [start, start + 180 d) from a site table on its regular half-hourly grid. Variables the site
    never measures inside the window are left out."""
    start = pd.Timestamp(start)
    T = WINDOW_DAYS * STEPS_PER_DAY
    t = pd.date_range(start, periods=T, freq="30min")
    tab = table.set_index(lst_index(table)).reindex(t)
    keys, vals, meas, prov = [], [], [], []
    for v in tower_vars:
        if v not in tab:
            continue
        x = tab[v].to_numpy(np.float64)
        flag = tab.get(f"{v}_flag")
        ok = np.isfinite(x) & ((flag.to_numpy() == 0) if flag is not None else True)
        if ok.any():
            keys.append(v); vals.append(np.where(ok, x, np.nan)); meas.append(ok); prov.append("tower")
    for v in reanalysis_vars:
        if v in tab:
            x = tab[v].to_numpy(np.float64)
            if np.isfinite(x).any():
                keys.append(v); vals.append(x); meas.append(np.isfinite(x)); prov.append("reanalysis")
    return Window(site, start, t, keys, np.array(vals), np.array(meas), prov)
