"""STAC search over Earth Search (Sentinel-2 Collection 1 L2A) and Microsoft Planetary Computer (Sentinel-2 L2A).

Every item intersecting the tower point is kept. Items are reduced to one per acquisition, where an acquisition is
one MGRS tile imaged by one datatake (`s2:datatake_id` without its baseline suffix). Overlapping tiles and orbits
are different acquisitions and are all kept. When a tile-datatake exists in several processings, the chosen one is,
in order: Collection 1 (baseline 05.00 or later), Earth Search over Planetary Computer, the higher baseline, the later
generation time.
"""

import json
import random
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import requests

from bands import ASSETS, baseline_tuple, boa_add_offset, offset_from_raster_bands

EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
PLANETARY = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"
SOURCES = {
    "earth-search-c1": (EARTH_SEARCH, "sentinel-2-c1-l2a"),
    "planetary-computer": (PLANETARY, "sentinel-2-l2a"),
}
MISSION_START = "2015-06-23T00:00:00Z"
USER_AGENT = "FluxFill-Sentinel2/1.0 (UC Berkeley QED lab; legel@berkeley.edu)"


def http_retry(fn, tries: int = 7, base: float = 2.0, cap: float = 120.0):
    """Call fn() with exponential backoff and jitter on network errors, 429 and 5xx."""
    for k in range(tries):
        try:
            r = fn()
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            r.raise_for_status()
            return r
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status is not None and status < 500 and status != 429:
                raise
            if k == tries - 1:
                raise
            time.sleep(min(cap, base * 2 ** k) * (0.5 + random.random()))


_session = threading.local()


def session() -> requests.Session:
    if not hasattr(_session, "s"):
        s = requests.Session()
        s.headers["User-Agent"] = USER_AGENT
        _session.s = s
    return _session.s


def search(source: str, lon: float, lat: float, start: str = MISSION_START, end: str | None = None,
           page: int = 200) -> list:
    """All STAC features of `source` whose footprint contains (lon, lat), following `next` links."""
    url, coll = SOURCES[source]
    end = end or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    body = {"collections": [coll], "intersects": {"type": "Point", "coordinates": [lon, lat]},
            "datetime": f"{start}/{end}", "limit": page}
    href, feats = f"{url}/search", []
    while True:
        r = http_retry(lambda: session().post(href, json=body, timeout=120))
        js = r.json()
        feats.extend(js.get("features", []))
        nxt = [ln for ln in js.get("links", []) if ln.get("rel") == "next"]
        if not nxt or not js.get("features"):
            return feats
        ln = nxt[0]
        href = ln["href"]
        if ln.get("method", "GET").upper() == "POST":
            body = {**body, **ln.get("body", {})} if ln.get("merge") else ln.get("body", body)
        else:   # GET pagination: the token is in the href
            r = None
            while href:
                r = http_retry(lambda: session().get(href, timeout=120))
                js = r.json()
                feats.extend(js.get("features", []))
                nx = [x for x in js.get("links", []) if x.get("rel") == "next"]
                href = nx[0]["href"] if nx and js.get("features") else None
            return feats


@dataclass
class Scene:
    """One Sentinel-2 acquisition (tile x datatake) and the metadata stored with its chip."""
    source: str
    item_id: str
    datetime_utc: str
    platform: str
    mgrs_tile: str
    epsg: int
    relative_orbit: int
    datatake_id: str
    acq_key: str
    processing_baseline: str
    boa_add_offset: int
    generation_time: str
    sun_azimuth: float
    sun_elevation: float
    view_azimuth: float
    view_incidence: float
    scene_cloud_cover: float
    hrefs: dict = field(default_factory=dict)

    def meta(self) -> dict:
        d = asdict(self)
        d.pop("hrefs")
        return d


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _rel_orbit(props: dict) -> int:
    if props.get("sat:relative_orbit") is not None:
        return int(props["sat:relative_orbit"])
    m = re.search(r"_R(\d{3})_", props.get("s2:product_uri", ""))
    return int(m.group(1)) if m else -1


def parse(source: str, feat: dict) -> Scene:
    p = feat["properties"]
    tile = p.get("s2:mgrs_tile") or (p.get("grid:code", "").replace("MGRS-", ""))
    dt_id = p.get("s2:datatake_id", "")
    core = re.sub(r"_N\d{2}\.\d{2}$", "", dt_id) or p.get("s2:product_uri", feat["id"])[:26]
    baseline = str(p.get("s2:processing_baseline", ""))
    offset = boa_add_offset(baseline)
    rb = (feat["assets"].get(ASSETS[source]["B04"], {}).get("raster:bands") or [None])[0]
    declared = offset_from_raster_bands(rb)
    if declared is not None and declared != offset:
        raise ValueError(f"{feat['id']}: raster:bands offset {declared} disagrees with baseline {baseline}")
    if "view:sun_elevation" in p:
        sun_az, sun_el = _f(p.get("view:sun_azimuth")), _f(p.get("view:sun_elevation"))
    else:
        sun_az, sun_el = _f(p.get("s2:mean_solar_azimuth")), 90.0 - _f(p.get("s2:mean_solar_zenith"))
    hrefs = {b: feat["assets"][a]["href"] for b, a in ASSETS[source].items() if a in feat["assets"]}
    return Scene(
        source=source, item_id=feat["id"], datetime_utc=p["datetime"], platform=str(p.get("platform", "")).lower(),
        mgrs_tile=tile, epsg=int(p.get("proj:epsg") or str(p.get("proj:code", "0")).split(":")[-1]),
        relative_orbit=_rel_orbit(p), datatake_id=dt_id, acq_key=f"{tile}|{core}", processing_baseline=baseline,
        boa_add_offset=offset, generation_time=str(p.get("s2:generation_time", p.get("updated", ""))),
        sun_azimuth=sun_az, sun_elevation=sun_el, view_azimuth=_f(p.get("view:azimuth")),
        view_incidence=_f(p.get("view:incidence_angle")), scene_cloud_cover=_f(p.get("eo:cloud_cover")),
        hrefs=hrefs)


SOURCE_RANK = {"earth-search-c1": 1, "planetary-computer": 0}


def rank(s: Scene) -> tuple:
    return (baseline_tuple(s.processing_baseline) >= (5, 0), SOURCE_RANK[s.source],
            baseline_tuple(s.processing_baseline), s.generation_time)


def select(scenes: list) -> list:
    """One scene per acquisition key, by `rank`; sorted by time."""
    best = {}
    for s in scenes:
        if s.acq_key not in best or rank(s) > rank(best[s.acq_key]):
            best[s.acq_key] = s
    return sorted(best.values(), key=lambda s: (s.datetime_utc, s.mgrs_tile))


def scenes_for(lon: float, lat: float, sources=("earth-search-c1", "planetary-computer")) -> tuple:
    """(selected scenes, per-source item counts) for a tower."""
    allsc, counts = [], {}
    for src in sources:
        feats = search(src, lon, lat)
        counts[src] = len(feats)
        allsc.extend(parse(src, f) for f in feats)
    return select(allsc), counts


class PCToken:
    """Anonymous Planetary Computer SAS token for the sentinel-2-l2a container, refreshed before expiry."""

    def __init__(self):
        self._lock, self._token, self._expiry = threading.Lock(), None, 0.0

    def get(self) -> str:
        with self._lock:
            if self._token is None or time.time() > self._expiry - 600:
                js = http_retry(lambda: session().get(PC_TOKEN, timeout=60)).json()
                self._token = js["token"]
                exp = js.get("msft:expiry", "")
                self._expiry = (datetime.strptime(exp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
                                if exp else time.time() + 1800)
            return self._token

    def sign(self, href: str) -> str:
        return f"{href}?{self.get()}"


def dump_scenes(scenes: list, path: str) -> None:
    with open(path, "w") as f:
        for s in scenes:
            f.write(json.dumps(asdict(s)) + "\n")
