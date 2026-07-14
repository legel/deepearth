"""Sample USGS 3DEP 1m topographic derivatives at every observation (microtopography drives fine-scale species
distribution: west-facing microslopes, local drainage, ruggedness). One 1m-DEM window per ~220 m cell; compute
slope/aspect/ruggedness/curvature rasters in native UTM meters, sample all obs in the cell.

Source = 3DEP 1m staged COGs on AWS (prd-tnm), resolved per cell via the TNM Access product API and read with
windowed /vsicurl range requests (NO whole-tile download, no auth, no anonymous-IP throttling -- unlike the
elevation.nationalmap.gov ImageServer, which is rate-limited and returns dynamically-mosaicked patches with no
stable id). Each obs is SCENE-PINNED to the exact 1m tile filename (topo_scene) -> reproducible provenance.

Output: gbif_topo_tokens.npz {gbifID, topo[N,12], has_topo[N], topo_scene[N]} with features
[elev, slope_deg, northness, eastness, TRI, curvature, VRM, HLI, *TPI(4)]. Resumable via a checkpoint pickle.
"""
import os, sys, time, pickle, warnings, math, json
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import rasterio
from rasterio.windows import from_bounds
from scipy import ndimage
from pyproj import Transformer
warnings.filterwarnings("ignore")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("VSI_CACHE", "TRUE")

HERE = os.path.dirname(os.path.abspath(__file__))
TNM = "https://tnmaccess.nationalmap.gov/api/v1/products"
DATASET = "Digital Elevation Model (DEM) 1 meter"
CELL = 0.002                     # ~220 m dedup cells; fetch a fixed 512 m 1m patch centered on each
PATCH_HALF = 256                 # meters; 512x512 px @ 1m -> TPI up to ~140 m radius fits with margin
WORKERS = 16                     # parallel tile readers (S3 scales; each opens one /vsicurl COG)
IDXW = 8                         # parallel TNM Access index queries
CKPT = os.path.join(HERE, "topo_ckpt.pkl")
IDXP = os.path.join(HERE, "topo_tileidx.json")
_tf = {}                         # per-crs transformer cache

def transformer(crs):
    k = str(crs)
    if k not in _tf:
        _tf[k] = Transformer.from_crs(4326, crs, always_xy=True)
    return _tf[k]

def tnm_tiles(w, s, e, n, retries=4):
    """TNM Access -> list of (tiff_url, (minlon,minlat,maxlon,maxlat)) for 1m DEM tiles intersecting the bbox."""
    for k in range(retries):
        try:
            r = requests.get(TNM, params={"datasets": DATASET, "bbox": f"{w},{s},{e},{n}",
                                          "outputFormat": "JSON", "max": 250}, timeout=60)
            r.raise_for_status()
            out = []
            for it in r.json().get("items", []):
                u = (it.get("urls") or {}).get("TIFF") or it.get("downloadURL")
                bb = it.get("boundingBox") or {}
                if u and bb:
                    out.append((u, (bb["minX"], bb["minY"], bb["maxX"], bb["maxY"])))
            return out
        except Exception:
            if k == retries - 1:
                return []
            time.sleep(2.0 * (k + 1))
    return []

def build_index(cell_list):
    """Resolve every cell to its covering 1m tiles. Query TNM over a ~0.1 deg grid (few hundred calls), collect
    tiles + lon/lat bounds, map each cell to ALL bbox-containing tiles (CA has OVERLAPPING 1m projects across years,
    and any single tile can have a nodata gap where another fills it -> keep every candidate, try in order at read
    time). Per-cell fallback for gaps. Persisted as cell -> [urls]."""
    idx = json.load(open(IDXP)) if os.path.exists(IDXP) else {}
    assigned = {tuple(map(int, k.split("_"))): v for k, v in idx.get("cells", {}).items()}
    todo = [c for c in cell_list if c not in assigned]
    if not todo:
        print(f"tile index: {len(assigned)} cells cached", flush=True)
        return assigned
    grid = defaultdict(list)
    for c in todo:
        latc, lonc = (c[0] + .5) * CELL, (c[1] + .5) * CELL
        grid[(round(latc / 0.1), round(lonc / 0.1))].append((c, latc, lonc))
    print(f"tile index: {len(todo)} cells -> {len(grid)} TNM grid queries", flush=True)
    keys = list(grid)
    def q(gk):
        latc, lonc = gk[0] * 0.1, gk[1] * 0.1
        return gk, tnm_tiles(lonc - 0.06, latc - 0.06, lonc + 0.06, latc + 0.06)
    done_q = 0
    with ThreadPoolExecutor(max_workers=IDXW) as ex:
        for gk, ts in ex.map(q, keys):
            for (c, latc, lonc) in grid[gk]:
                cands = [u for u, bb in ts if bb[0] <= lonc <= bb[2] and bb[1] <= latc <= bb[3]]
                if cands:
                    assigned[c] = cands
            done_q += 1
            if done_q % 50 == 0:
                print(f"  index {done_q}/{len(keys)} queries | {len(assigned)} cells assigned", flush=True)
    # per-cell fallback for cells no grid query contained (tile-edge / sparse coverage)
    miss = [c for c in todo if c not in assigned]
    if miss:
        print(f"  fallback: {len(miss)} unassigned cells -> per-cell TNM", flush=True)
        def qc(c):
            latc, lonc = (c[0] + .5) * CELL, (c[1] + .5) * CELL
            return c, [u for u, _ in tnm_tiles(lonc - 1e-4, latc - 1e-4, lonc + 1e-4, latc + 1e-4)]
        with ThreadPoolExecutor(max_workers=IDXW) as ex:
            for c, us in ex.map(qc, miss):
                if us:
                    assigned[c] = us
    json.dump({"cells": {f"{c[0]}_{c[1]}": u for c, u in assigned.items()}}, open(IDXP, "w"))
    print(f"tile index built: {len(assigned)}/{len(cell_list)} cells assigned", flush=True)
    return assigned

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

def process_tile(url, cell_obs):
    """Open one 3DEP 1m COG (windowed /vsicurl); for every cell assigned to it, read a 512 m patch, compute
    derivatives in native UTM meters, sample all its obs. Returns {gbifID: (feat[12], tile_filename)} or {}."""
    fname = url.rsplit("/", 1)[-1]
    try:
        with rasterio.open("/vsicurl/" + url) as ds:
            tf = transformer(ds.crs); out = {}
            for cell, obs in cell_obs.items():
                latc, lonc = (cell[0] + 0.5) * CELL, (cell[1] + 0.5) * CELL
                cx, cy = tf.transform(lonc, latc)
                if not (np.isfinite(cx) and np.isfinite(cy)):
                    continue
                w = from_bounds(cx - PATCH_HALF, cy - PATCH_HALF, cx + PATCH_HALF, cy + PATCH_HALF, ds.transform)
                dem = ds.read(1, window=w, boundless=True, fill_value=np.nan).astype(np.float32)
                if dem.shape[0] < 16 or dem.shape[1] < 16 or np.isnan(dem).all():
                    continue
                d = derivatives(dem)
                wt = ds.window_transform(w)                 # affine of the window's upper-left
                ny, nx = dem.shape
                for gid, la, lo in obs:
                    x, y = tf.transform(lo, la)
                    col = (x - wt.c) / wt.a; row = (y - wt.f) / wt.e   # wt.e < 0 (north-up)
                    r = int(np.clip(round(row), 0, ny - 1)); c = int(np.clip(round(col), 0, nx - 1))
                    if not np.isfinite(dem[r, c]) or dem[r, c] <= -1e4:  # nodata at the obs pixel -> retry alt tile
                        continue
                    hli = _hli(float(d["slope_rad"][r, c]), float(d["aspect_deg"][r, c]), la)
                    vec = [dem[r, c], d["slope"][r, c], d["northness"][r, c], d["eastness"][r, c],
                           d["tri"][r, c], d["curv"][r, c], d["vrm"][r, c], hli] + [t[r, c] for t in d["tpis"]]
                    out[gid] = (np.array(vec, np.float32), fname)
            return out
    except Exception as e:
        print(f"  tile fail {fname}: {type(e).__name__} {e}", flush=True)
        return {}

def main():
    z = np.load(os.path.join(HERE, "obs_coords.npz"))
    gid, lat, lon = z["gbifID"], z["lat"], z["lon"]
    cells = {}
    for i in range(len(gid)):
        key = (int(np.floor(lat[i] / CELL)), int(np.floor(lon[i] / CELL)))
        cells.setdefault(key, []).append((int(gid[i]), float(lat[i]), float(lon[i])))
    done, scene = {}, {}
    if os.path.exists(CKPT):
        ck = pickle.load(open(CKPT, "rb"))
        done = ck.get("done", ck); scene = ck.get("scene", {})    # tolerate the old {gid:vec} checkpoint
        print(f"resume: {len(done)} obs already sampled", flush=True)
    assigned = build_index(list(cells))                    # cell -> [candidate tile urls]
    n_unres = sum(1 for c in cells if c not in assigned)
    # remaining[cell] = candidate tiles not yet tried; obs get recorded when a tile has real (non-nodata) data there
    remaining = {c: list(assigned[c]) for c in cells
                 if c in assigned and not all(o[0] in done for o in cells[c])}
    print(f"{len(cells)} cells ({n_unres} unresolved), {len(remaining)} to fetch, {len(gid)} obs", flush=True)
    t0 = time.time(); rnd = 0
    while remaining and rnd < 6:                            # rounds: overlapping-project fallback for nodata gaps
        rnd += 1
        by_tile = defaultdict(dict)
        for c, cands in remaining.items():
            by_tile[cands[0]][c] = cells[c]
        n_ok = n_fail = 0
        print(f"round {rnd}: {len(remaining)} cells across {len(by_tile)} tiles", flush=True)
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(process_tile, u, co): u for u, co in by_tile.items()}
            for i, fut in enumerate(as_completed(futs)):
                res = fut.result()
                if res:
                    for g, (v, fn) in res.items():
                        done[g] = v; scene[g] = fn
                    n_ok += 1
                else:
                    n_fail += 1
                if (i + 1) % 25 == 0:
                    pickle.dump({"done": done, "scene": scene}, open(CKPT, "wb"))
                    el = time.time() - t0
                    print(f"  r{rnd} {i+1}/{len(by_tile)} tiles | ok {n_ok} fail {n_fail} | {len(done)} obs | "
                          f"{el:.0f}s | eta {el/(i+1)*(len(by_tile)-i-1)/60:.1f}min", flush=True)
        pickle.dump({"done": done, "scene": scene}, open(CKPT, "wb"))
        remaining = {c: cands[1:] for c, cands in remaining.items()          # unsatisfied obs -> next candidate tile
                     if len(cands) > 1 and any(o[0] not in done for o in cells[c])}
    # assemble aligned arrays
    topo = np.zeros((len(gid), NFEAT), np.float32); have = np.zeros(len(gid), bool)
    scn = np.empty(len(gid), object)
    for i in range(len(gid)):
        g = int(gid[i])
        if g in done:
            topo[i] = done[g]; have[i] = True; scn[i] = scene.get(g, "")
    np.savez(os.path.join(HERE, "..", "gbif_topo_tokens.npz"), gbifID=gid, topo=topo, has_topo=have,
             topo_scene=scn.astype(object))
    print(f"DONE: {have.sum()}/{len(gid)} obs have topo ({100*have.mean():.1f}%), fail tiles {n_fail}", flush=True)

if __name__ == "__main__":
    main()
