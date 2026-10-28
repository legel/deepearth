"""Point sampling of the gridded soil products we benchmark, all read remotely (no CONUS downloads):

  SOLUS100   USDA-NRCS, 100 m, Nauman et al. 2024; public COGs in gs://solus100pub ({prop}_{depth}_cm_p.tif, EPSG:5070),
             values x 1/scalar from Final_Layer_Table_20231215.csv. Depth = point depth (0, 5, 15 ... cm).
  POLARIS    Chaney et al. 2019, 30 m, 1-degree EPSG:4326 tiles (hydrology.cee.duke.edu), interval depths (0_5, 5_15 cm);
             om is log10(%).
  SoilGrids  ISRIC 2.0, 250 m Homolosine VRTs (files.isric.org); pH x10, clay/sand g/kg, soc dg/kg; urban areas are masked (nodata).
  SSURGO     NRCS Soil Data Access: map unit at the point, horizons overlapping the depth interval, component-percent
             weighted mean of the representative values (ph1to1h2o_r, claytotal_r, sandtotal_r, om_r).

Every function returns surface-layer values for a depth interval [top, bottom) in common units:
pH (1:1 water), clay %, sand %, organic carbon % (OM = 1.724 x OC where a product reports OM)."""
import json, math, time, urllib.parse, urllib.request, functools, concurrent.futures as cf
import numpy as np, rasterio, rasterio.errors
from rasterio.warp import transform

ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.vrt",
           GDAL_HTTP_MAX_RETRY=4, GDAL_HTTP_RETRY_DELAY=2, VSI_CACHE=True, GDAL_CACHEMAX=512)
PROPS = ("ph", "clay", "sand", "oc")
OM_TO_OC = 1 / 1.724

SOLUS = {"ph": ("ph1to1h2o", 100), "clay": ("claytotal", 1), "sand": ("sandtotal", 1), "oc": ("soc", 1000)}
POLARIS = {"ph": "ph", "clay": "clay", "sand": "sand", "oc": "om"}
SOILGRIDS = {"ph": ("phh2o", 10), "clay": ("clay", 10), "sand": ("sand", 10), "oc": ("soc", 100)}


def _sample_chunk(path, lon, lat, tries=5):
    """Values at the points; a remote read that times out is retried with backoff (nan after `tries` failures)."""
    for k in range(tries):
        try:
            with rasterio.Env(**ENV), rasterio.open(path) as s:
                xs, ys = (lon, lat) if s.crs.to_epsg() == 4326 else transform("EPSG:4326", s.crs, list(lon), list(lat))
                v = np.array([x[0] for x in s.sample(zip(xs, ys))], dtype="float64")
                if s.nodata is not None:
                    v[v == s.nodata] = np.nan
            return v
        except rasterio.errors.RasterioIOError as e:
            if "404" in str(e) or "does not exist" in str(e) or k == tries - 1:
                if k == tries - 1 and "404" not in str(e):
                    return np.full(len(lon), np.nan)
                raise
            time.sleep(3 * 2 ** k)


def _sample(path, lon, lat, workers=8, chunk=16):
    """Values of the first band at (lon, lat) arrays; nodata -> nan. Scattered points each fetch their own remote
    block, so chunks are read concurrently (one dataset handle per thread)."""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    if len(lon) <= chunk:
        return _sample_chunk(path, lon, lat)
    parts = [slice(i, i + chunk) for i in range(0, len(lon), chunk)]
    with cf.ThreadPoolExecutor(workers) as ex:
        return np.concatenate(list(ex.map(lambda sl: _sample_chunk(path, lon[sl], lat[sl]), parts)))


def _overlap_weights(top, bottom, layers):
    """Thickness overlap of [top, bottom) with each (a, b) layer."""
    return np.array([max(0.0, min(bottom, b) - max(top, a)) for a, b in layers])


LAYERS = ((0, 5), (5, 15), (15, 30))          # the profile every product is reduced to (0-30 cm covers all tests)


def profile(product, prop, lon, lat):
    """(n, 3) array of `prop` in LAYERS at the points: one remote read per raster for all points."""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    if product == "solus":                      # point depths -> layer means (trapezoid)
        name, scal = SOLUS[prop]
        d = {k: _sample(f"/vsicurl/https://storage.googleapis.com/solus100pub/{name}_{k}_cm_p.tif", lon, lat) / scal
             for k in (0, 5, 15, 30)}
        return np.stack([(d[a] + d[b]) / 2 for a, b in LAYERS], 1)
    if product == "soilgrids":                  # soc: dg/kg / 100 = %
        name, scal = SOILGRIDS[prop]
        return np.stack([_sample(f"/vsicurl/https://files.isric.org/soilgrids/latest/data/{name}/{name}_{a}-{b}cm_mean.vrt",
                                 lon, lat) / scal for a, b in LAYERS], 1)
    if product == "polaris":
        out = np.full((len(lon), len(LAYERS)), np.nan)
        tiles = {}
        for i, (x, y) in enumerate(zip(lon, lat)):
            tiles.setdefault((math.floor(y), math.floor(x)), []).append(i)
        for (la, lo), idx in tiles.items():
            for j, (a, b) in enumerate(LAYERS):
                url = (f"/vsicurl/http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/{POLARIS[prop]}/mean/{a}_{b}/"
                       f"lat{la}{la + 1}_lon{lo}{lo + 1}.tif")
                try:
                    v = _sample(url, lon[idx], lat[idx])
                except Exception:
                    continue                    # outside POLARIS coverage (CONUS only)
                out[idx, j] = 10 ** v * OM_TO_OC if prop == "oc" else v
        return out
    raise ValueError(product)


def interval_mean(prof, top, bottom):
    """Thickness-weighted mean of a (n, 3) LAYERS profile over per-point [top, bottom)."""
    n = prof.shape[0]
    top, bottom = np.broadcast_to(np.asarray(top, float), (n,)), np.broadcast_to(np.asarray(bottom, float), (n,))
    w = np.stack([np.clip(np.minimum(bottom, b) - np.maximum(top, a), 0, None) for a, b in LAYERS], 1)
    w = np.where(np.isnan(prof), 0, w)
    with np.errstate(invalid="ignore"):
        return np.where(w.sum(1) > 0, np.nansum(prof * w, 1) / np.maximum(w.sum(1), 1e-9), np.nan)


def solus(lon, lat, prop, top=0, bottom=5):
    return interval_mean(profile("solus", prop, lon, lat), top, bottom)


def polaris(lon, lat, prop, top=0, bottom=5):
    return interval_mean(profile("polaris", prop, lon, lat), top, bottom)


def soilgrids(lon, lat, prop, top=0, bottom=5):
    return interval_mean(profile("soilgrids", prop, lon, lat), top, bottom)


@functools.lru_cache(maxsize=100000)
def _sda_point(lon, lat, top, bottom):
    q = f"""SELECT c.cokey, c.comppct_r, h.hzdept_r, h.hzdepb_r, h.ph1to1h2o_r, h.claytotal_r, h.sandtotal_r, h.om_r
    FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('point({lon:.6f} {lat:.6f})') mu
    JOIN component c ON c.mukey = mu.mukey JOIN chorizon h ON h.cokey = c.cokey
    WHERE h.hzdept_r < {bottom} AND h.hzdepb_r > {top}"""
    d = urllib.parse.urlencode({"query": q, "format": "JSON+COLUMNNAME"}).encode()
    for k in range(4):
        try:
            r = json.load(urllib.request.urlopen("https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest", d, timeout=90))
            break
        except Exception:
            if k == 3:
                return None
    t = r.get("Table") or []
    if len(t) < 2:
        return {p: np.nan for p in PROPS}
    cols = t[0]; rows = [dict(zip(cols, x)) for x in t[1:]]
    out = {}
    for p, col in (("ph", "ph1to1h2o_r"), ("clay", "claytotal_r"), ("sand", "sandtotal_r"), ("oc", "om_r")):
        num = den = 0.0
        for r_ in rows:
            if r_[col] is None or r_["comppct_r"] is None:
                continue
            ov = max(0.0, min(bottom, float(r_["hzdepb_r"])) - max(top, float(r_["hzdept_r"])))
            w = float(r_["comppct_r"]) * ov
            num += w * float(r_[col]); den += w
        v = num / den if den else np.nan
        out[p] = v * OM_TO_OC if p == "oc" else v
    return out


def ssurgo(lon, lat, top=0, bottom=5):
    """{prop: array} for all four properties (one SDA query per point)."""
    res = [_sda_point(round(float(x), 6), round(float(y), 6), top, bottom) for x, y in zip(lon, lat)]
    return {p: np.array([np.nan if r is None else r[p] for r in res]) for p in PROPS}
