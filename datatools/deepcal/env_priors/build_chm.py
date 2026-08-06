"""Sample NAIP-CHM 0.6m canopy-height-and-structure (trees+buildings, height-above-ground) at every observation.
Openly-hosted rasters (rangeland.ntsg.umt.edu); per-obs +-150m window -> structure descriptors. Foundation for the
solar/microclimate channels (DEM+CHM net height). Resumable checkpoint. Output: gbif_chm_tokens.npz {gbifID, chm[N,D], has_chm}."""
import os, time, pickle, warnings, json, urllib.request
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
from shapely.geometry import shape, Point
from shapely.strtree import STRtree
warnings.filterwarnings("ignore")
os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", VSI_CACHE="TRUE")

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_URL = "https://rangeland.ntsg.umt.edu/data/naip-chm/index.geojson"
INDEX = os.path.join(HERE, "naip_chm_index.geojson")
CKPT = os.path.join(HERE, "chm_ckpt.pkl")
HALF = 150                       # meters window half-width -> 500x500 px @ 0.6m
WORKERS = 12
_tf = {}
def transformer(epsg):
    if epsg not in _tf: _tf[epsg] = Transformer.from_crs(4326, epsg, always_xy=True)
    return _tf[epsg]

def descriptors(h):
    """h: meters, nan where nodata. Invariant structure (height leaks nothing; structure is transferable)."""
    v = h[np.isfinite(h)]
    if v.size < h.size * 0.4: return None
    mean, std, mx = float(v.mean()), float(v.std()), float(v.max())
    p50, p90, p95 = [float(np.percentile(v, p)) for p in (50, 90, 95)]
    cover = float((v > 2).mean()); gap = float((v < 0.5).mean()); shrub = float(((v >= 0.5) & (v <= 2)).mean())
    hf = np.nan_to_num(h, nan=float(np.nanmean(h)))
    gy, gx = np.gradient(hf, 0.6)                                    # rumple: canopy-surface / planar area
    rumple = float(np.mean(np.sqrt(1 + gx * gx + gy * gy)))
    het = float(std / (mean + 1e-3))                                # vertical heterogeneity (CV)
    return np.array([mean, std, mx, p50, p90, p95, cover, gap, shrub, rumple, het], np.float32)
NFEAT = 11

def load_index():
    if not os.path.exists(INDEX):
        print("downloading index.geojson (299MB)...", flush=True)
        urllib.request.urlretrieve(INDEX_URL, INDEX)
    gj = json.load(open(INDEX))
    geoms, urls, years = [], [], []
    for f in gj["features"]:
        p = f["properties"]; lon0 = f["geometry"]["coordinates"][0][0][0]
        # CA bbox prefilter on centroid-ish first ring vertex
        geoms.append(shape(f["geometry"])); urls.append(p.get("chm_url", "")); years.append(int(float(p.get("year", 0) or 0)))
    return geoms, np.array(urls, object), np.array(years)

def process_tile(url, obs):
    """obs: list of (gid, lat, lon). Returns {gid: feat}."""
    u = url.replace("http://", "https://")
    vsi = "/vsicurl/" + u
    out = {}
    try:
        with rasterio.open(vsi) as ds:
            tf = transformer(ds.crs.to_epsg())
            for gid, la, lo in obs:
                x, y = tf.transform(lo, la)
                try:
                    patch = ds.read(1, window=from_bounds(x - HALF, y - HALF, x + HALF, y + HALF, ds.transform))
                except Exception:
                    continue
                h = patch.astype(np.float32); h[patch == 65535] = np.nan; h /= 100.0
                d = descriptors(h)
                if d is not None: out[gid] = d
    except Exception:
        return {}
    return out

def main():
    z = np.load(os.path.join(HERE, "obs_coords.npz"))
    gid, lat, lon = z["gbifID"], z["lat"], z["lon"]
    geoms, urls, years = load_index()
    tree = STRtree(geoms)
    print(f"index: {len(geoms)} tiles; assigning {len(gid)} obs to newest covering tile...", flush=True)
    tile_obs = {}                                                   # url -> [(gid,lat,lon)]
    for i in range(len(gid)):
        pt = Point(float(lon[i]), float(lat[i]))
        cand = tree.query(pt)
        best_url, best_year = None, -1
        for j in cand:
            if geoms[j].contains(pt) and years[j] > best_year:
                best_year, best_url = years[j], urls[j]
        if best_url: tile_obs.setdefault(best_url, []).append((int(gid[i]), float(lat[i]), float(lon[i])))
    done = pickle.load(open(CKPT, "rb")) if os.path.exists(CKPT) else {}
    todo = [u for u in tile_obs if not all(o[0] in done for o in tile_obs[u])]
    print(f"{len(tile_obs)} tiles, {len(todo)} to fetch, {sum(len(v) for v in tile_obs.values())} obs covered", flush=True)
    t0 = time.time(); nok = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(process_tile, u, tile_obs[u]): u for u in todo}
        for i, fut in enumerate(as_completed(futs)):
            r = fut.result()
            if r: done.update(r); nok += 1
            if (i + 1) % 100 == 0:
                pickle.dump(done, open(CKPT, "wb")); el = time.time() - t0
                print(f"  {i+1}/{len(todo)} tiles | {len(done)} obs | {el:.0f}s | eta {el/(i+1)*(len(todo)-i-1)/60:.1f}min", flush=True)
    pickle.dump(done, open(CKPT, "wb"))
    chm = np.zeros((len(gid), NFEAT), np.float32); have = np.zeros(len(gid), bool)
    for i in range(len(gid)):
        g = int(gid[i])
        if g in done: chm[i] = done[g]; have[i] = True
    np.savez(os.path.join(HERE, "..", "gbif_chm_tokens.npz"), gbifID=gid, chm=chm, has_chm=have)
    print(f"DONE: {have.sum()}/{len(gid)} obs have chm ({100*have.mean():.1f}%)", flush=True)

if __name__ == "__main__":
    main()
