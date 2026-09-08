"""Rainfall forcing: observed storms from ASOS, and NOAA Atlas 14 design storms.

Both routes produce the same thing -- rainfall rate [m/s] on a `dt_s`-spaced axis -- so the
solver never knows which it is running.
"""

import json
from typing import Tuple

import numpy as np
import pandas as pd

from sites import SiteConfig, Storm

SCS_II_T = np.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
SCS_II_P = np.array([0, .04, .1, .2, .35, .63, .78, .87, .93, .97, 1.0])
"""SCS Type II dimensionless cumulative rainfall, the Florida standard. Peak intensity lands at
about 60 % of the duration, which suits convective storms here."""

RETURN_PERIODS_YR = (1, 2, 5, 10, 25, 50, 100, 200, 500)
"""Design-storm ensemble. Atlas 14 is queried per coordinate, so this runs unchanged anywhere,
which is what makes cross-site comparison mean anything."""


def observed_hyetograph(site: SiteConfig, storm: Storm, dt_s: float,
                        extend_hours: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Rainfall rate [m/s] from the site's ASOS record over the storm window.

    Args:
        site: Site whose ASOS station to read.
        storm: Storm defining the UTC window.
        dt_s: Output timestep [s].
        extend_hours: Zero-rain hours appended so the solver's own drainage tail is not cut
            off. A magnitude comparison that truncates the model but not the gauge is unfair in
            a way that has been measured, not assumed.

    Returns:
        (rate [m/s] per dt_s step, hourly depth [mm] as recorded).
    """
    path = site.asos(storm.name)
    assert path.exists(), f"{path} missing; run `python3 cli.py fetch --site {site.name}`"

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    time_col = next((c for c in df.columns if "datetime" in c or "valid" in c or "time" in c),
                    df.columns[0])
    precip_col = next((c for c in df.columns if "prcp" in c or "p01" in c or "precip" in c), None)
    assert precip_col is not None, f"{path} has no recognisable precipitation column"

    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()
    window = df[pd.Timestamp(storm.start, tz="UTC"):pd.Timestamp(storm.end, tz="UTC")]
    assert not window.empty, f"{path} has no records inside {storm.start}..{storm.end}"

    rain_mm = pd.to_numeric(window[precip_col], errors="coerce").fillna(0.0).values
    hours = np.arange(len(rain_mm), dtype=float)
    total_s = (len(rain_mm) + extend_hours) * 3600.0
    t = np.arange(0.0, total_s, dt_s)
    return np.interp(t, hours * 3600.0, rain_mm / 1000 / 3600, right=0.0), rain_mm


def design_hyetograph(depth_mm: float, duration_hr: float, dt_s: float) -> np.ndarray:
    """SCS Type II rainfall rate [m/s] for a total depth over a duration."""
    assert depth_mm > 0 and duration_hr > 0, "design storm needs a positive depth and duration"
    total_s = duration_hr * 3600.0
    edges = np.arange(0.0, total_s + dt_s, dt_s)
    cumulative = np.interp(edges / total_s, SCS_II_T, SCS_II_P) * (depth_mm / 1000.0)
    return np.diff(cumulative) / dt_s


def atlas14_depth_mm(site: SiteConfig, return_period_yr: int, duration_hr: float) -> float:
    """Design rainfall depth [mm] for one return period and duration.

    Reads the fetched Atlas 14 table. It carries a provenance field because this query silently
    fell back to hardcoded county defaults for months -- a stale URL plus a parser expecting the
    wrong JSON shape -- printing a plausible table the whole time.
    """
    assert site.atlas14.exists(), (
        f"{site.atlas14} missing; run `python3 cli.py fetch --site {site.name}`")
    table = json.loads(site.atlas14.read_text())
    assert table.get("source") == "pfds", (
        f"{site.atlas14} was not fetched from NOAA PFDS (source={table.get('source')!r}); "
        f"re-run the fetch rather than modelling on fallback values")

    key = f"{duration_hr:g}hr"
    assert key in table["depths_mm"], f"duration {key} not in {sorted(table['depths_mm'])}"
    row = table["depths_mm"][key]
    assert str(return_period_yr) in row, f"return period {return_period_yr} not in {sorted(row)}"
    return float(row[str(return_period_yr)])
