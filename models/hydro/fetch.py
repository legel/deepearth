"""Every public dataset a twin needs, for any coordinate.

One function per source, all driven by `SiteConfig`. On the old branch this layer was ~2,815
lines duplicated across two sites at 83-100 % similarity -- one pair was byte-identical -- plus
sixteen wrapper scripts that existed only because the fetch functions could not take a site.

Anything that cannot be fetched raises. The one soft case is Atlas 14, which records its own
provenance instead: it silently served hardcoded county defaults for months behind a stale URL,
so a fallback is written to disk but marked, and `forcing.atlas14_depth_mm` refuses to model on
it.
"""

import io
import json
import subprocess
import time
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests

from sites import SiteConfig, Storm

USER_AGENT = "DeepEarth-hydro/1.0 (research; https://github.com/legel/deepearth)"

SDA_URL = "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest"
THREEDHP_URL = "https://3dhp.nationalmap.gov/arcgis/rest/services/usgs_3dhp_all/FeatureServer"
NFHL_URL = "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer"
MRLC_WCS = "https://www.mrlc.gov/geoserver/mrlc_display/ows"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
NWIS_IV_URL = "https://waterservices.usgs.gov/nwis/iv/"
PFDS_URL = "https://hdsc.nws.noaa.gov/cgi-bin/new/cgi_readH5.py"
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

PFDS_DURATIONS_HR = [5 / 60, 10 / 60, 15 / 60, 30 / 60, 1, 2, 3, 6, 12, 24,
                     48, 72, 96, 168, 240, 480, 720, 1080, 1440]
"""The 19 duration rows PFDS returns, in the order it returns them."""

PFDS_RETURN_PERIODS_YR = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]
"""The 10 return-period columns PFDS returns.

Deliberately NOT `forcing.RETURN_PERIODS_YR`, which is the 9-storm ensemble this project
actually runs and stops at 500. Conflating the two makes the shape guard below reject a
perfectly good response and silently write a fallback table -- which is the exact failure mode
the provenance field exists to catch.
"""

HORTON_K_BY_TEXTURE = {
    "sand": 1.5, "loamy sand": 1.8, "sandy loam": 2.0, "loam": 2.5, "silt loam": 3.0,
    "silt": 3.0, "sandy clay loam": 3.5, "clay loam": 4.0, "silty clay loam": 4.0,
    "sandy clay": 4.5, "silty clay": 5.0, "clay": 5.0, "default": 3.0,
}
"""Horton decay constant [1/hr] by USDA texture class."""

F0_KSAT_RATIO = {
    "sand": 5.0, "loamy sand": 4.5, "sandy loam": 4.0, "loam": 3.5, "silt loam": 3.5,
    "silt": 3.0, "sandy clay loam": 3.0, "clay loam": 2.8, "silty clay loam": 2.5,
    "sandy clay": 2.5, "silty clay": 2.3, "clay": 2.0, "default": 3.0,
}
"""Initial-to-final infiltration ratio by texture, after Rawls et al. (1983) for dry soil."""


def _get(url: str, params: Optional[Dict] = None, retries: int = 4,
         timeout: int = 45) -> requests.Response:
    """GET with exponential backoff. A multi-decade download is a large surface for one blip."""
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


# ── Terrain ──────────────────────────────────────────────────────────────────────────────

def dem(site: SiteConfig, resolution_m: int = 1) -> None:
    """USGS 3DEP elevation, falling back through 3 m and 10 m if the finest is unavailable."""
    import py3dep
    import rioxarray  # noqa: F401  imported for its side effect: the .rio accessor

    attempts = []
    for res in sorted({resolution_m, 3, 10}):
        try:
            raster = py3dep.get_dem(site.bbox(), crs="epsg:4326", resolution=res)
        except Exception as exc:  # 3DEP raises several unrelated types on an unavailable tile
            attempts.append(f"{res} m: {type(exc).__name__}: {exc}")
            continue
        raster.rio.to_raster(site.dem)
        return
    raise RuntimeError(f"3DEP returned no DEM for {site.name}. Tried:\n  "
                       + "\n  ".join(attempts))


# ── Hydrography, floodplain, infrastructure ──────────────────────────────────────────────

def _arcgis_layer(base: str, layer: int, bbox: Tuple[float, float, float, float]) -> Dict:
    """Query one ArcGIS REST layer over a WGS84 envelope, as GeoJSON."""
    west, south, east, north = bbox
    return _get(f"{base}/{layer}/query", {
        "where": "1=1",
        "geometry": f"{west},{south},{east},{north}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326, "outSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*", "returnGeometry": "true", "f": "geojson",
    }).json()


def hydrography(site: SiteConfig) -> Dict[str, int]:
    """USGS 3DHP flowlines (layer 50) and waterbodies (layer 60).

    The flowlines are what `terrain.burn_streams` carves, and 3DHP names reaches that NHD
    leaves unnamed. Note the Catchment layer is empty across this region -- USGS is populating
    it as elevation-derived hydrography rolls out -- so watersheds are delineated here, not
    fetched.
    """
    counts = {}
    for layer, path, key in ((50, site.flowlines, "flowlines"),
                             (60, site.waterbodies, "waterbodies")):
        fc = _arcgis_layer(THREEDHP_URL, layer, site.bbox())
        path.write_text(json.dumps(fc))
        counts[key] = len(fc.get("features", []))
    assert counts["flowlines"] > 0, (
        f"3DHP returned no flowlines for {site.name}. Stream burning cannot run without them, "
        f"and a silently empty burn is exactly the failure this asserts against.")
    return counts


def floodplain(site: SiteConfig) -> int:
    """FEMA National Flood Hazard Layer zones, including the regulatory floodway (layer 28)."""
    fc = _arcgis_layer(NFHL_URL, 28, site.bbox())
    site.fema_zones.write_text(json.dumps(fc))
    return len(fc.get("features", []))


def roads_and_buildings(site: SiteConfig) -> Dict[str, int]:
    """OpenStreetMap highways and building footprints, via Overpass.

    These become the impervious mask: `physics.ROAD_BUFFER_M` turns width-less centrelines into
    real surfaces, and the solver and the viewer read the same table so the drawn footprint is
    the one the physics used.
    """
    import geopandas as gpd
    from shapely.geometry import LineString, Polygon

    west, south, east, north = site.bbox()
    query = (f"[out:json][timeout:90];("
             f'way["highway"]({south},{west},{north},{east});'
             f'way["building"]({south},{west},{north},{east});'
             f'relation["building"]({south},{west},{north},{east});'
             f");out body;>;out skel qt;")
    data = _get(OVERPASS_URL, {"data": query}, timeout=120).json()

    nodes = {e["id"]: (e["lon"], e["lat"]) for e in data["elements"] if e["type"] == "node"}
    roads, buildings = [], []
    for e in data["elements"]:
        if e["type"] != "way" or "nodes" not in e:
            continue
        coords = [nodes[n] for n in e["nodes"] if n in nodes]
        tags = e.get("tags", {})
        if "highway" in tags and len(coords) >= 2:
            roads.append({"highway": tags["highway"], "geometry": LineString(coords)})
        elif "building" in tags and len(coords) >= 4:
            buildings.append({"building": tags["building"], "geometry": Polygon(coords)})

    for records, path in ((roads, site.roads), (buildings, site.buildings)):
        gdf = gpd.GeoDataFrame(records, crs="epsg:4326") if records else \
            gpd.GeoDataFrame({"geometry": []}, crs="epsg:4326")
        gdf.to_file(path, driver="GeoJSON")
    return {"roads": len(roads), "buildings": len(buildings)}


# ── Soil and land cover ──────────────────────────────────────────────────────────────────

def _sda(sql: str) -> Optional[List[List]]:
    """POST a query to USDA Soil Data Access; returns rows with a header row first."""
    r = requests.post(SDA_URL, data={"query": sql, "format": "JSON+COLUMNNAME"},
                      timeout=90, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    return r.json().get("Table")


def _texture(sand_pct: object, clay_pct: object) -> str:
    """USDA texture class from sand and clay percentages, coarse but sufficient for Horton."""
    try:
        s, c = float(sand_pct), float(clay_pct)
    except (TypeError, ValueError):
        return "loam"
    if s >= 85:
        return "sand"
    if s >= 70 and c <= 15:
        return "loamy sand"
    if s >= 50 and c <= 20:
        return "sandy loam"
    if c >= 40:
        return "clay"
    if c >= 28:
        return "clay loam"
    return "loam"


def _horton(ksat_umps: object, texture: str) -> Dict[str, float]:
    """Horton f0/fc/k [mm/hr, mm/hr, 1/hr] from saturated conductivity and texture."""
    try:
        ksat = float(ksat_umps) * 3.6
        if ksat != ksat:  # NaN
            ksat = None
    except (TypeError, ValueError):
        ksat = None
    ksat = max(ksat if ksat is not None else 10.0, 0.1)
    key = next((k for k in HORTON_K_BY_TEXTURE if k in texture), "default")
    return {"fc_mm_hr": round(ksat, 2),
            "f0_mm_hr": round(ksat * F0_KSAT_RATIO[key], 2),
            "k_hr": round(HORTON_K_BY_TEXTURE[key], 2)}


def _orient_wfs(geo, bbox: Tuple[float, float, float, float]):
    """Tag and, if necessary, un-swap the SSURGO WFS response.

    Two quirks, both silent. The service does not set a CRS, so geopandas sees naive geometries
    and `to_crs` refuses them. And it emits coordinates as (lat, lon) while declaring EPSG:4326
    -- the mirror of its lon,lat BBOX expectation. Reprojected as-is, the polygons land some
    16,000 km from the domain and rasterise to an empty map unit raster, which reads downstream
    as "uniform soil everywhere" rather than as an error.

    Rather than assume the swap, this checks which orientation actually overlaps the box that
    was requested, so an upstream fix cannot silently mirror the map back.
    """
    from shapely.ops import transform as shapely_transform

    if geo.crs is None:
        geo = geo.set_crs("epsg:4326")

    west, south, east, north = bbox

    def overlaps(minx, miny, maxx, maxy) -> bool:
        return not (maxx < west or minx > east or maxy < south or miny > north)

    if overlaps(*geo.total_bounds):
        return geo
    swapped = geo.copy()
    swapped["geometry"] = geo.geometry.map(
        lambda g: shapely_transform(lambda x, y, z=None: (y, x), g))
    assert overlaps(*swapped.total_bounds), (
        f"SSURGO polygons do not overlap the requested box in either axis order; "
        f"got {tuple(round(v, 4) for v in geo.total_bounds)} against {bbox}")
    return swapped


def soil(site: SiteConfig) -> Dict[str, int]:
    """SSURGO map units, their Horton parameters, the map-unit raster and the storage table.

    Writes `soil_parameters.json`, `mukey_map.tif`, `mukey_map_legend.csv` and
    `soil_storage.csv`. The storage table is what bounds infiltration; without it the solver
    absorbs essentially any storm, so it is fetched here rather than left optional.
    """
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize

    west, south, east, north = site.bbox()
    wkt = (f"POLYGON(({west} {south},{east} {south},{east} {north},"
           f"{west} {north},{west} {south}))")

    rows = _sda(f"SELECT DISTINCT mukey FROM "
                f"SDA_Get_Mukey_from_intersection_with_WktWgs84('{wkt}')")
    assert rows and len(rows) > 1, f"SDA returned no soil map units for {site.name}"
    mukeys = [str(r[0]) for r in rows[1:]]
    in_list = ",".join(f"'{m}'" for m in mukeys)

    comp = _sda(
        f"SELECT mu.mukey, mu.muname, co.cokey, co.comppct_r, co.hydgrp "
        f"FROM mapunit mu JOIN component co ON co.mukey = mu.mukey "
        f"WHERE mu.mukey IN ({in_list})")
    horizon = _sda(
        f"SELECT ch.cokey, ch.ksat_r, ch.sandtotal_r, ch.claytotal_r, ch.hzdept_r "
        f"FROM chorizon ch JOIN component co ON ch.cokey = co.cokey "
        f"WHERE co.mukey IN ({in_list})")
    assert comp and horizon, f"SDA component/horizon query returned nothing for {site.name}"

    cdf = pd.DataFrame(comp[1:], columns=comp[0])
    hdf = pd.DataFrame(horizon[1:], columns=horizon[0])
    for col in ("ksat_r", "sandtotal_r", "claytotal_r", "hzdept_r"):
        hdf[col] = pd.to_numeric(hdf[col], errors="coerce")
    surface_h = hdf.sort_values("hzdept_r").groupby("cokey", as_index=False).first()
    cdf["comppct_r"] = pd.to_numeric(cdf["comppct_r"], errors="coerce").fillna(0)
    cdf = cdf.merge(surface_h, on="cokey", how="left")

    params: Dict[str, Dict] = {}
    for mukey in mukeys:
        sub = cdf[cdf["mukey"].astype(str) == mukey]
        if sub.empty:
            continue
        dom = sub.sort_values("comppct_r", ascending=False).iloc[0]
        hsg_raw = (str(dom.get("hydgrp") or "B")).strip() or "B"
        # Dual ratings like "B/D" mean drained/undrained. Keep the drained side as primary and
        # record both -- an empty rating on open water once mapped to B, giving a water body a
        # residential curve number and a nonzero infiltration rate.
        drained = (hsg_raw.split("/")[0].strip().upper() or "B")[0]
        wet = (hsg_raw.split("/")[1].strip().upper()[0] if "/" in hsg_raw else drained)
        drained = drained if drained in "ABCD" else "B"
        wet = wet if wet in "ABCD" else drained

        muname = str(dom.get("muname") or "")
        if "water" in muname.lower():
            horton = {"fc_mm_hr": 0.0, "f0_mm_hr": 0.0, "k_hr": HORTON_K_BY_TEXTURE["default"]}
        else:
            horton = _horton(dom.get("ksat_r"), _texture(dom.get("sandtotal_r"),
                                                         dom.get("claytotal_r")))
        params[mukey] = {"muname": muname, "hsg": drained, "hsg_wet": wet, **horton}

    site.soil_params.write_text(json.dumps(params, indent=1))

    geo = gpd.read_file(
        f"https://sdmdataaccess.sc.egov.usda.gov/Spatial/SDMWGS84Geographic.wfs?"
        f"SERVICE=WFS&VERSION=1.1.0&REQUEST=GetFeature&TYPENAME=MapunitPoly&"
        f"SRSNAME=EPSG:4326&BBOX={west},{south},{east},{north}")
    # BBOX is lon,lat here. WFS 1.1.0 specifies lat,lon for EPSG:4326 and this service ignores
    # that: measured on one box, lat,lon returns 0 features and lon,lat returns 18. It answers
    # HTTP 200 either way, so the wrong order reads as "no soil here" rather than as an error.
    assert not geo.empty, f"SSURGO WFS returned no polygons for {site.name}"
    geo = _orient_wfs(geo, site.bbox())
    geo["mukey"] = geo["mukey"].astype(str)

    with rasterio.open(site.dem) as src:
        profile, shape, transform, crs = src.profile.copy(), src.shape, src.transform, src.crs
    geo = geo.to_crs(crs)
    codes = {m: i + 1 for i, m in enumerate(sorted(params))}
    shapes = [(g.__geo_interface__, codes[m]) for m, g in
              zip(geo["mukey"], geo.geometry) if m in codes and g is not None]
    assert shapes, f"no SSURGO polygons overlap {site.name}'s DEM"

    raster = rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype="int32")
    profile.update(dtype="int32", count=1, nodata=0, compress="deflate")
    with rasterio.open(site.mukey_map, "w", **profile) as dst:
        dst.write(raster, 1)
    pd.DataFrame({"mukey": list(codes), "mukey_int": list(codes.values())}).to_csv(
        site.mukey_legend, index=False)

    storage = _sda(f"SELECT mukey, wtdepannmin FROM muaggatt WHERE mukey IN ({in_list})")
    assert storage and len(storage) > 1, (
        f"SDA muaggatt returned no water-table depths for {site.name}. Without them soil "
        f"storage is unbounded and the solver absorbs essentially any storm.")
    pd.DataFrame(storage[1:], columns=storage[0]).to_csv(site.soil_storage, index=False)
    return {"map_units": len(params), "storage_rows": len(storage) - 1}


def nlcd(site: SiteConfig) -> None:
    """NLCD 2021 impervious-surface percentage, ~30 m, via the MRLC WCS."""
    west, south, east, north = site.bbox()
    pad = 0.005
    w, s, e, n = west - pad, south - pad, east + pad, north + pad
    width = max(100, int((e - w) / 0.00027))
    height = max(100, int((n - s) / 0.00027))
    url = (f"{MRLC_WCS}?SERVICE=WCS&VERSION=1.0.0&REQUEST=GetCoverage"
           f"&COVERAGE=NLCD_2021_Impervious_L48&BBOX={w},{s},{e},{n}"
           f"&CRS=EPSG:4326&RESPONSE_CRS=EPSG:4326&FORMAT=GeoTIFF"
           f"&WIDTH={width}&HEIGHT={height}")
    with urllib.request.urlopen(url, timeout=120) as resp:
        site.nlcd_impervious.write_bytes(resp.read())


def naip(site: SiteConfig) -> Optional[str]:
    """NAIP aerial imagery, mosaicked, via the Planetary Computer STAC API.

    The most recent year available is taken, so the ground resolution is whatever NAIP flew last
    and improves over time: 0.6 m over this site in 2021, 0.3 m in 2023. That is a 965 MB mosaic
    built in memory by `merge`, ~18 minutes and ~2.7 GB of RSS for a 6 km box -- by far the most
    expensive fetch here, and the reason this one is worth running last.

    Feeds the surface parameterisation and the viewer's aerial drape; the solver never reads it.

    Returns:
        The NAIP year used, or None when the catalogue has no imagery for this box.
    """
    import planetary_computer
    import pystac_client
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import transform_bounds

    catalog = pystac_client.Client.open(PC_STAC_URL, modifier=planetary_computer.sign_inplace)
    items = list(catalog.search(collections=["naip"], bbox=site.bbox()).items())
    if not items:
        return None
    year = max(i.properties["datetime"][:4] for i in items)
    latest = [i for i in items if i.properties["datetime"][:4] == year]

    srcs = [rasterio.open(i.assets["image"].href) for i in latest]
    # merge() interprets `bounds` in the SOURCE crs. NAIP ships in UTM, so handing it the site's
    # WGS84 box silently intersects to nothing and rasterio then refuses to write a 0x0 raster.
    bounds = transform_bounds("epsg:4326", srcs[0].crs, *site.bbox())
    mosaic, transform = merge(srcs, bounds=bounds)
    profile = srcs[0].profile.copy()
    profile.update(height=mosaic.shape[1], width=mosaic.shape[2],
                   transform=transform, count=3, compress="deflate")
    with rasterio.open(site.naip_rgb, "w", **profile) as dst:
        dst.write(mosaic[:3])
    if mosaic.shape[0] >= 4:
        profile.update(count=1)
        with rasterio.open(site.naip_nir, "w", **profile) as dst:
            dst.write(mosaic[3:4])
    for s in srcs:
        s.close()
    return year


# ── Forcing and observations ─────────────────────────────────────────────────────────────

def asos(site: SiteConfig, storm: Storm) -> Dict[str, float]:
    """Hourly rainfall [mm] for the storm window from the site's ASOS station, via IEM.

    IEM's `p01i` is a running hourly accumulation reset each hour, so the hourly total is the
    LAST report in each hour, not a sum of the 5-minute rows.
    """
    start = pd.Timestamp(storm.start) - pd.Timedelta(days=1)
    end = pd.Timestamp(storm.end) + pd.Timedelta(days=2)
    r = _get(IEM_ASOS_URL, {
        "station": site.asos_station, "data": "p01i", "tz": "UTC", "format": "onlycomma",
        "missing": "empty", "trace": "0.0001", "latlon": "no", "report_type": "3",
        "year1": start.year, "month1": start.month, "day1": start.day,
        "year2": end.year, "month2": end.month, "day2": end.day,
    }, timeout=120)

    df = pd.read_csv(io.StringIO(r.text))
    df["valid"] = pd.to_datetime(df["valid"], utc=True, errors="coerce")
    df["p01i"] = pd.to_numeric(df["p01i"], errors="coerce")
    df = df.dropna(subset=["valid"]).set_index("valid").sort_index()
    hourly = df["p01i"].resample("1h").last().fillna(0.0) * 25.4  # inches -> mm

    out = hourly.reset_index()
    out.columns = ["datetime", "precip_mm"]
    out.to_csv(site.asos(storm.name), index=False)
    return {"hours": len(out), "total_mm": float(out["precip_mm"].sum()),
            "peak_mm_hr": float(out["precip_mm"].max())}


def discharge(site: SiteConfig, storm: Storm) -> Dict[str, float]:
    """Observed 15-minute discharge [cfs] from USGS NWIS instantaneous values."""
    assert site.gauge is not None, f"site {site.name} has no gauge"
    r = _get(NWIS_IV_URL, {"format": "json", "sites": site.gauge.site_no,
                           "startDT": storm.gauge_start, "endDT": storm.gauge_end,
                           "parameterCd": "00060"}, timeout=120)
    series = r.json()["value"]["timeSeries"]
    assert series, f"NWIS returned no discharge for {site.gauge.site_no} over the storm window"

    df = pd.DataFrame(series[0]["values"][0]["value"])
    df["dateTime"] = pd.to_datetime(df["dateTime"], utc=True)
    df["discharge_cfs"] = pd.to_numeric(df["value"], errors="coerce")
    df[["dateTime", "discharge_cfs"]].to_csv(site.discharge(storm.name), index=False)
    return {"samples": len(df), "peak_cfs": float(df["discharge_cfs"].max())}


def atlas14(site: SiteConfig) -> str:
    """NOAA Atlas 14 depth-duration-frequency table for the site's coordinate.

    Python 3.9 links LibreSSL 2.8.3 and cannot complete the TLS handshake with the PFDS host
    (`verify=False` skips certificate validation, not the handshake), so system curl is the
    fallback transport. Provenance is written into the file: this query returned hardcoded
    county defaults for months behind a stale URL and a parser expecting a dict where PFDS
    returns an array, printing a plausible table throughout.

    Returns:
        "pfds" when the real grid was parsed, "fallback" otherwise. `forcing` refuses to model
        on a fallback table.
    """
    import re

    params = {"lat": site.lat, "lon": site.lon, "type": "pf", "data": "depth",
              "units": "english", "series": "pds"}
    body = None
    try:
        body = _get(PFDS_URL, params, retries=2, timeout=60).text
    except Exception:
        query = "&".join(f"{k}={v}" for k, v in params.items())
        proc = subprocess.run(["curl", "-sL", "--max-time", "60", f"{PFDS_URL}?{query}"],
                              capture_output=True, text=True)
        body = proc.stdout if proc.returncode == 0 and proc.stdout else None

    depths: Dict[str, Dict[str, float]] = {}
    source = "fallback"
    match = re.search(r"quantiles\s*=\s*(\[\[.*?\]\])\s*;", body, re.S) if body else None
    if match:
        grid = json.loads(match.group(1).replace("'", '"'))
        if len(grid) == len(PFDS_DURATIONS_HR) and len(grid[0]) == len(PFDS_RETURN_PERIODS_YR):
            for dur_hr, row in zip(PFDS_DURATIONS_HR, grid):
                depths[f"{dur_hr:g}hr"] = {str(rp): float(v) * 25.4
                                           for rp, v in zip(PFDS_RETURN_PERIODS_YR, row)}
            source = "pfds"

    site.atlas14.write_text(json.dumps(
        {"source": source, "lat": site.lat, "lon": site.lon, "units": "mm",
         "return_periods_yr": PFDS_RETURN_PERIODS_YR, "depths_mm": depths}, indent=1))
    return source


def all_sources(site: SiteConfig, storms: Sequence[Storm] = ()) -> Dict[str, object]:
    """Fetch everything the pipeline needs for a site, in dependency order."""
    summary: Dict[str, object] = {}
    dem(site)
    summary["hydrography"] = hydrography(site)
    summary["soil"] = soil(site)          # needs the DEM grid to rasterise onto
    nlcd(site)
    summary["naip_year"] = naip(site)
    summary["osm"] = roads_and_buildings(site)
    summary["fema_features"] = floodplain(site)
    summary["atlas14"] = atlas14(site)
    for storm in storms:
        summary[f"asos_{storm.name}"] = asos(site, storm)
        if site.gauge is not None:
            summary[f"discharge_{storm.name}"] = discharge(site, storm)
    return summary
