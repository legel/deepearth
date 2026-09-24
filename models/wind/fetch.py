"""The public data a wind twin needs: the nearest ASOS station's hourly record, via IEM."""

import io
import time
from typing import Dict, Optional, Sequence

import pandas as pd
import requests

from physics import M_S_PER_KNOT
from sites import SiteConfig

USER_AGENT = "DeepEarth-wind/1.0 (research; https://github.com/legel/deepearth)"
IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


def _get(url: str, params: Optional[Dict] = None, retries: int = 4,
         timeout: int = 120) -> requests.Response:
    """GET with exponential backoff."""
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout,
                             headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
            return r
        except requests.RequestException as exc:
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last}")


def asos(site: SiteConfig, year: int) -> Dict[str, float]:
    """Hourly wind for one year from the site's ASOS station.

    Routine hourly reports only (`report_type=3`), one row per UTC hour: speed and gust in
    knots converted to m/s, direction in degrees from which the wind blows. Calm reports carry
    speed 0 and no direction.
    """
    r = _get(IEM_ASOS_URL, {
        "station": site.asos_station, "data": "sknt,drct,gust", "tz": "UTC",
        "format": "onlycomma", "missing": "empty", "trace": "0.0001", "latlon": "no",
        "report_type": "3", "year1": year, "month1": 1, "day1": 1,
        "year2": year + 1, "month2": 1, "day2": 1,
    })
    df = pd.read_csv(io.StringIO(r.text))
    df["valid"] = pd.to_datetime(df["valid"], utc=True, errors="coerce")
    df = df.dropna(subset=["valid"]).set_index("valid").sort_index()
    for col in ("sknt", "drct", "gust"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    hourly = df.resample("1h").last()
    hourly = hourly[hourly.index.year == year]

    out = pd.DataFrame({
        "speed_m_s": hourly["sknt"] * M_S_PER_KNOT,
        "direction_deg": hourly["drct"],
        "gust_m_s": hourly["gust"] * M_S_PER_KNOT,
    })
    out.index.name = "datetime"
    out.to_csv(site.asos(year), float_format="%.3f")
    reported = out["speed_m_s"].notna()
    return {"hours": int(len(out)), "reported_hours": int(reported.sum()),
            "mean_m_s": float(out["speed_m_s"].mean()),
            "max_gust_m_s": float(out["gust_m_s"].max())}


def all_sources(site: SiteConfig, years: Sequence[int]) -> Dict[str, object]:
    """Fetch everything the pipeline needs for a site."""
    return {f"asos_{year}": asos(site, year) for year in years}
