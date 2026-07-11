"""Sample USGS 3DEP 1m topographic derivatives at every observation (microtopography drives fine-scale species
distribution: west-facing microslopes, local drainage, ruggedness). One 1m-DEM fetch per ~2.2km cell (17.6k cells
cover 207k obs), compute slope/aspect/ruggedness/curvature rasters in UTM meters, sample all obs in the cell.

Output: gbif_topo_tokens.npz {gbifID, topo[N,6], has_topo[N]} with features
[elev, slope_deg, northness, eastness, TRI, curvature]. Resumable via a checkpoint pickle.
"""
import os, io, sys, time, pickle, warnings, math
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import rasterio
from rasterio.io import MemoryFile
from scipy import ndimage
from pyproj import Transformer
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
CELL = 0.002                     # ~220 m dedup cells; fetch a fixed 512 m 1m patch centered on each (fast render)
PATCH_HALF = 256                 # meters; 512x512 px @ 1m -> TPI up to ~140 m radius fits with margin
WORKERS = 16
CKPT = os.path.join(HERE, "topo_ckpt.pkl")
_wgs = {}                        # per-utm-epsg transformer cache

def utm_epsg(lon):
    return 26910 if lon < -120.0 else 26911   # CA UTM zone 10N / 11N

def transformer(epsg):
    if epsg not in _wgs:
        _wgs[epsg] = Transformer.from_crs(4326, epsg, always_xy=True)
    return _wgs[epsg]

def fetch_dem(xmin, ymin, xmax, ymax, epsg, nx, ny, retries=5):
    url = (f"{BASE}?bbox={xmin},{ymin},{xmax},{ymax}&bboxSR={epsg}&size={nx},{ny}"
           f"&imageSR={epsg}&format=tiff&pixelType=F32&noData=-9999&adjustAspectRatio=false"
           f"&interpolation=RSP_BilinearInterpolation&f=image")
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                data = r.read()
            with MemoryFile(data) as mf, mf.open() as ds:
                return ds.read(1).astype(np.float32)
        except Exception:
            if k == retries - 1:
                return None
            time.sleep(2.0 * (k + 1))
    return None

TPI_RADII = (8, 24, 72, 140)                    # meters(px at 1m): micro -> hillslope -> local (fit in 512m patch)
# feature order per obs: [elev, slope, northness, eastness, tri, curv, vrm, hli, *TPI]  (elev leaks location; the
# INVARIANT form descriptors -- TPI/VRM/HLI/aspect -- are the transferable signal per Ploton 2020 / the terrain research)
NFEAT = 8 + len(TPI_RADII)                         # 12

def _std_filter(a, size):
    """Fast vectorized local std via uniform_filter (E[x^2]-E[x]^2), C-optimized (not generic_filter)."""
    m = ndimage.uniform_filter(a, size, mode="nearest")
    m2 = ndimage.uniform_filter(a * a, size, mode="nearest")
    return np.sqrt(np.clip(m2 - m * m, 0, None))

def derivatives(dem):
    """Multi-scale invariant surrounding-topography bank at 1m. Returns rasters:
    slope(deg), northness, eastness, TRI, curvature, VRM (Sappington ruggedness decoupled from slope), aspect_deg
    (for per-point HLI), and TPI at each radius (point vs neighborhood mean: ridge>0, valley<0). All vectorized."""
    dem = np.where(np.isfinite(dem) & (dem > -1e4), dem, np.nan)
    dem = _fill(dem)
    dzdy, dzdx = np.gradient(dem, 1.0)          # dzdy: down-rows (south+), dzdx: right-cols (east+)
    slope = np.arctan(np.hypot(dzdx, dzdy))     # radians
    asp = np.arctan2(dzdx, -dzdy)               # geographic aspect: 0=N, pi/2=E (north-up)
    northness = np.cos(asp); eastness = np.sin(asp)
    tri = _std_filter(dem, 3)
    curv = ndimage.laplace(dem, mode="nearest")
    # VRM (Sappington 2007): resultant of unit surface-normal vectors over a 9x9 window; 0=flat/planar, 1=rugged.
    xn = np.sin(slope) * np.sin(asp); yn = np.sin(slope) * np.cos(asp); zn = np.cos(slope)
    mx = ndimage.uniform_filter(xn, 9); my = ndimage.uniform_filter(yn, 9); mz = ndimage.uniform_filter(zn, 9)
    vrm = 1.0 - np.sqrt(mx * mx + my * my + mz * mz)
    aspect_deg = (np.degrees(asp)) % 360.0
    tpis = [dem - ndimage.uniform_filter(dem, 2 * r + 1, mode="nearest") for r in TPI_RADII]
    return dict(slope=np.degrees(slope), northness=northness, eastness=eastness, tri=tri, curv=curv,
                vrm=vrm, aspect_deg=aspect_deg, slope_rad=slope, tpis=tpis)

def _hli(slope_rad, aspect_deg, lat_deg):
    """Heat Load Index (McCune & Keon 2002): folds aspect+slope+latitude into a topoclimate insolation load
    (SW slopes hottest). The single most-used plant-SDM topoclimate predictor; invariant physical forcing."""
    lat = math.radians(lat_deg)
    fold = math.radians(abs(180.0 - abs(aspect_deg - 225.0)))
    s = slope_rad
    return math.exp(-1.467 + 1.582 * math.cos(lat) * math.cos(s)
                    - 1.500 * math.cos(fold) * math.sin(s) * math.sin(lat)
                    - 0.262 * math.sin(lat) * math.sin(s) + 0.607 * math.sin(fold) * math.sin(s))

def _fill(a):
    if not np.isnan(a).any():
        return a
    m = np.isnan(a)
    if m.all():
        return np.zeros_like(a)
    idx = ndimage.distance_transform_edt(m, return_distances=False, return_indices=True)
    return a[tuple(idx)]

def process_cell(cell, obs):
    """obs: list of (gbifID, lat, lon). Return {gbifID: feat[6]} or None on failure."""
    (clat, clon) = cell
    latc, lonc = (clat + 0.5) * CELL, (clon + 0.5) * CELL     # cell CENTER (cell key = round(coord/CELL))
    epsg = utm_epsg(lonc)
    tf = transformer(epsg)
    cx, cy = tf.transform(lonc, latc)
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return None
    xmin, xmax = cx - PATCH_HALF, cx + PATCH_HALF             # fixed 512 m patch centered on the cell
    ymin, ymax = cy - PATCH_HALF, cy + PATCH_HALF
    nx = ny = 2 * PATCH_HALF
    dem = fetch_dem(xmin, ymin, xmax, ymax, epsg, nx, ny)
    if dem is None or dem.shape != (ny, nx):
        return None
    d = derivatives(dem)
    res_x = (xmax - xmin) / nx; res_y = (ymax - ymin) / ny
    out = {}
    for gid, la, lo in obs:
        x, y = tf.transform(lo, la)
        col = (x - xmin) / res_x; row = (ymax - y) / res_y
        r = int(np.clip(round(row), 0, ny - 1)); c = int(np.clip(round(col), 0, nx - 1))
        hli = _hli(float(d["slope_rad"][r, c]), float(d["aspect_deg"][r, c]), la)   # per-point (uses obs latitude)
        vec = [dem[r, c], d["slope"][r, c], d["northness"][r, c], d["eastness"][r, c],
               d["tri"][r, c], d["curv"][r, c], d["vrm"][r, c], hli] + [t[r, c] for t in d["tpis"]]
        out[gid] = np.array(vec, dtype=np.float32)
    return out

def main():
    z = np.load(os.path.join(HERE, "obs_coords.npz"))
    gid, lat, lon = z["gbifID"], z["lat"], z["lon"]
    cells = {}
    for i in range(len(gid)):
        key = (int(np.floor(lat[i] / CELL)), int(np.floor(lon[i] / CELL)))
        cells.setdefault(key, []).append((int(gid[i]), float(lat[i]), float(lon[i])))
    done = {}
    if os.path.exists(CKPT):
        done = pickle.load(open(CKPT, "rb"))
        print(f"resume: {len(done)} obs already sampled", flush=True)
    todo = [c for c in cells if not all(o[0] in done for o in cells[c])]
    print(f"{len(cells)} cells, {len(todo)} to fetch, {len(gid)} obs", flush=True)
    t0 = time.time(); n_ok = 0; n_fail = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(process_cell, c, cells[c]): c for c in todo}
        for i, fut in enumerate(as_completed(futs)):
            res = fut.result()
            if res:
                done.update(res); n_ok += 1
            else:
                n_fail += 1
            if (i + 1) % 500 == 0:
                pickle.dump(done, open(CKPT, "wb"))
                el = time.time() - t0
                print(f"  {i+1}/{len(todo)} cells | ok {n_ok} fail {n_fail} | {len(done)} obs | "
                      f"{el:.0f}s | eta {el/(i+1)*(len(todo)-i-1)/60:.1f}min", flush=True)
    pickle.dump(done, open(CKPT, "wb"))
    # assemble aligned arrays
    topo = np.zeros((len(gid), NFEAT), np.float32); have = np.zeros(len(gid), bool)
    for i in range(len(gid)):
        g = int(gid[i])
        if g in done:
            topo[i] = done[g]; have[i] = True
    np.savez(os.path.join(HERE, "..", "gbif_topo_tokens.npz"), gbifID=gid, topo=topo, has_topo=have)
    print(f"DONE: {have.sum()}/{len(gid)} obs have topo ({100*have.mean():.1f}%), fail cells {n_fail}", flush=True)

if __name__ == "__main__":
    main()
