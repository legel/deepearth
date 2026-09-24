"""Boundary forcing: the upwind log profile, and the ASOS record that sets it.

Everything here produces the same thing, a `LogProfile` and a direction, so the solver never
knows whether it is running one observed hour or one sector of the annual rose.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from physics import ASOS_ANEMOMETER_M, KAPPA
from sites import Day, SiteConfig

SECTORS = 16
"""Direction sectors in the wind rose, 22.5 degrees each, centred on north."""

CALM_M_S = 0.5
"""Speed below which an hour is calm and carries no direction."""


@dataclass(frozen=True)
class LogProfile:
    """Neutral surface-layer profile u(z) = (u* / kappa) ln((z - d) / z0).

    Attributes:
        u_star: Friction velocity [m/s].
        z0: Roughness length [m].
        d: Displacement height [m].
    """

    u_star: float
    z0: float
    d: float = 0.0

    @classmethod
    def from_reference(cls, u_ref: float, z_ref: float, z0: float, d: float = 0.0) -> "LogProfile":
        """The profile passing through `u_ref` at height `z_ref`."""
        assert z_ref - d > z0, f"reference height {z_ref} m is inside the roughness sublayer"
        return cls(u_star=KAPPA * u_ref / math.log((z_ref - d) / z0), z0=z0, d=d)

    def speed(self, z: np.ndarray) -> np.ndarray:
        """Speed [m/s] at heights `z` [m], zero at and below d + z0."""
        return self.u_star / KAPPA * np.log(np.maximum((np.asarray(z) - self.d) / self.z0, 1.0))


def transfer(profile: LogProfile, z0: float, d: float, z_blend: float) -> LogProfile:
    """Re-root a profile onto a different fetch, matching speed at the blending height.

    Args:
        profile: Profile over the station's fetch.
        z0: Roughness length of the destination fetch [m].
        d: Displacement height of the destination fetch [m].
        z_blend: Height at which both fetches see the same speed [m].

    Returns:
        The destination profile.
    """
    return LogProfile.from_reference(float(profile.speed(z_blend)), z_blend, z0, d)


def inflow(site: SiteConfig, speed: float, z_ref: float = ASOS_ANEMOMETER_M) -> LogProfile:
    """The parcel's upwind profile from a station speed at `z_ref`."""
    station = LogProfile.from_reference(speed, z_ref, site.station_z0_m)
    return transfer(station, site.z0_m, site.d_m, site.z_blend_m)


def wind_vector(speed: float, direction_deg: float) -> Tuple[float, float]:
    """(east, north) components of a wind blowing FROM `direction_deg`, clockwise from north."""
    theta = math.radians(direction_deg)
    return -speed * math.sin(theta), -speed * math.cos(theta)


def station_record(site: SiteConfig, year: int) -> "pd.DataFrame":  # noqa: F821
    """Hourly speed [m/s], direction [deg] and gust [m/s] for one year, indexed by UTC hour."""
    import pandas as pd       # the station record alone needs it; the solver imports this module

    path = site.asos(year)
    assert path.exists(), f"{path} missing; run `python3 cli.py fetch --site {site.name}`"
    df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
    assert df.index.tz is not None, f"{path} timestamps are not timezone-aware"
    return df


def day_series(site: SiteConfig, day: Day) -> np.ndarray:
    """(24, 3) array of speed [m/s], direction [deg], gust [m/s] for each UTC hour of `day`.

    Hours without a report take the nearest reported hour; direction on calm hours is NaN.
    """
    import pandas as pd

    df = station_record(site, day.year).dropna(subset=["speed_m_s"])
    start = pd.Timestamp(day.date, tz="UTC")
    hours = pd.date_range(start, periods=24, freq="1h")
    window = df.reindex(hours, method="nearest", tolerance=pd.Timedelta(hours=3))
    assert window["speed_m_s"].notna().all(), f"{day.date} has hours with no report within 3 h"
    return window[["speed_m_s", "direction_deg", "gust_m_s"]].to_numpy(dtype=float)


def sector_of(direction_deg: np.ndarray) -> np.ndarray:
    """Sector index 0..SECTORS-1 of each direction; sector 0 is centred on north."""
    width = 360.0 / SECTORS
    return (np.floor((np.asarray(direction_deg) + width / 2) / width) % SECTORS).astype(int)


def sector_centre(sector: int) -> float:
    """Direction [deg] at the centre of a sector."""
    return sector * 360.0 / SECTORS


def year_table(site: SiteConfig, year: int) -> List[Optional[List[float]]]:
    """[speed m/s, direction deg] for every hour of the year, None where nothing was reported."""
    df = station_record(site, year)
    rows = df[["speed_m_s", "direction_deg"]].to_numpy(float)
    return [None if not np.isfinite(r[0]) else [round(float(r[0]), 2), None if not np.isfinite(r[1])
                                                 else float(r[1])] for r in rows]


def rose(site: SiteConfig, year: int) -> Dict[str, object]:
    """Annual wind climatology by sector.

    Returns:
        Sector frequency (fraction of non-calm hours), mean and 90th-percentile speed [m/s]
        per sector, the calm fraction and the hour count.
    """
    df = station_record(site, year)
    speed, drct = df["speed_m_s"].to_numpy(float), df["direction_deg"].to_numpy(float)
    blowing = (speed >= CALM_M_S) & np.isfinite(drct)
    sectors = sector_of(drct[blowing])
    counts = np.bincount(sectors, minlength=SECTORS)
    mean = np.bincount(sectors, weights=speed[blowing], minlength=SECTORS) / np.maximum(counts, 1)
    p90 = [float(np.percentile(speed[blowing][sectors == s], 90)) if counts[s] else 0.0
           for s in range(SECTORS)]
    return {
        "year": year, "station": site.asos_station, "hours": int(len(df)),
        "calm_fraction": float(1.0 - blowing.mean()),
        "sector_deg": [sector_centre(s) for s in range(SECTORS)],
        "frequency": (counts / max(counts.sum(), 1)).tolist(),
        "mean_m_s": mean.tolist(), "p90_m_s": p90,
    }
