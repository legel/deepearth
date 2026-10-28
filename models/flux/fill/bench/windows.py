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

# Further measured tower channels (E30; opt-in, not rows of release 0.2): radiation components, humidity, wind
# direction, CO2 concentration, water-table depth and the deeper soil layers. Wind direction enters as the sine and
# cosine of its angle (WD_SIN, WD_COS; `with_extra_columns`), so the row is continuous at north.
EXTRA_TOWER_SOURCE = ("SW_OUT", "LW_OUT", "NETRAD", "PPFD_IN", "PPFD_OUT", "RH", "WD", "CO2", "WTD",
                      "SWC_2", "SWC_3", "SWC_4", "SWC_5", "TS_2", "TS_3", "TS_4", "TS_5")
EXTRA_TOWER_VARS = tuple(c for v in EXTRA_TOWER_SOURCE for c in (("WD_SIN", "WD_COS") if v == "WD" else (v,)))
UNITS.update({"SW_OUT": "W m⁻²", "LW_OUT": "W m⁻²", "NETRAD": "W m⁻²", "PPFD_IN": "µmol m⁻² s⁻¹",
              "PPFD_OUT": "µmol m⁻² s⁻¹", "RH": "%", "WD_SIN": "", "WD_COS": "", "CO2": "µmol mol⁻¹", "WTD": "m",
              **{f"SWC_{i}": "%" for i in range(2, 6)}, **{f"TS_{i}": "°C" for i in range(2, 6)}})
LABELS.update({"SW_OUT": "Shortwave out", "LW_OUT": "Longwave out", "NETRAD": "Net radiation",
               "PPFD_IN": "PPFD in", "PPFD_OUT": "PPFD out", "RH": "Relative humidity",
               "WD_SIN": "Wind direction (sine)", "WD_COS": "Wind direction (cosine)", "CO2": "CO2 concentration",
               "WTD": "Water-table depth", **{f"SWC_{i}": f"Soil water content, layer {i}" for i in range(2, 6)},
               **{f"TS_{i}": f"Soil temperature, layer {i}" for i in range(2, 6)}})


def with_extra_columns(table: pd.DataFrame) -> pd.DataFrame:
    """The table with WD_SIN and WD_COS (and their flags, those of WD) where it has WD; unchanged otherwise."""
    if "WD" not in table or "WD_SIN" in table:
        return table
    a = np.radians(table["WD"].to_numpy(np.float64))
    extra = {"WD_SIN": np.sin(a), "WD_COS": np.cos(a)}
    if "WD_flag" in table:
        extra.update({"WD_SIN_flag": table["WD_flag"].to_numpy(), "WD_COS_flag": table["WD_flag"].to_numpy()})
    return table.assign(**extra)


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
