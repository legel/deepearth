"""2024 NAIP per-observation imagery + DINOv3-SAT493M embeddings via USGS EROS M2M (2024-EXCLUSIVE).

Planetary Computer only serves 2022 NAIP for CA; the 2024 CNIR acquisition lives on USGS M2M. For each NAIP-2024
scene that covers >=1 observation (mapped from the cached statewide catalog naip2024_tiles.json), download the scene
ONCE via M2M unless the catalog provides local_path/url, window-read a 300x300 m patch CENTERED on every observation the scene covers -> 512x512 4-band uint8,
then (a) optionally accumulate the raw imagery patch -> per-scene npz streamed to NERSC and deleted locally, and
(b) embed with DINOv3-SAT493M. The pooled cache is still emitted for backward compatibility; NAIP_SAVE_PATCH32=1
also emits full 32x32x1024 patch-token shards for train/test-aligned Earth4D readers. Scene GeoTIFF is deleted
after tiling. Scene-pinned: naip_scene = M2M entityId, naip_year =
acquisition year. Streaming keeps the working set small (BATCH_TILES scenes at a time); resumable via a checkpoint
of finished gbifIDs.

Output: gbif_naip_tokens/chunk*.npz {gbifID, naip_year, naip_scene, rgb_pool[N,1024]f32, ir_pool[N,1024]f32}
Patch output: gbif_naip_dinov3_patch32_v1/manifest.npz plus chunk*.npz
  {gbifID, naip_year, naip_scene, patch[N,32,32,1024], patch_lat[N,32,32], patch_lon[N,32,32], has_naip}
Raw imagery (optional, NAIP_SAVE_IMAGERY=1): NERSC <NERSC_DIR>/<entityId>.npz {gbifID, patch[n,512,512,4]uint8}
env: USGS_M2M_TOKEN (default ~/.usgs_m2m_token), M2M_USER (ecological), NAIP_BATCH_TILES, NAIP_DLW,
     NAIP_SAVE_IMAGERY, NAIP_SAVE_PATCH32, NAIP_PATCH_DTYPE, NAIP_PATCH_VIEW, NAIP_EMBED_BATCH,
     NAIP_PATCH_ROWS, NAIP_NERSC_DIR, NAIP_TILES_JSON, HF cache. Low-memory default favors resumability over throughput.
"""
import os, sys, io, time, json, pickle, zipfile, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, requests, rasterio, matplotlib
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from pyproj import Transformer
from scipy.spatial import cKDTree
from dinov3_patch32 import DINOv3Patch32
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
CACHE = Path(os.environ.get("DEEPCAL_CACHE", HERE.parent)).expanduser()
TILES_JSON = Path(os.environ.get("NAIP_TILES_JSON", str(CACHE / "env_priors" / "naip2024_tiles.json")))
TOKENS = CACHE / "gbif_tokens"                                  # train/test shards: {gbifID, lat, lon, ...}
COORDS = CACHE / "env_priors" / "obs_coords.npz"                # fallback: {gbifID, lat, lon}
TOK = CACHE / "gbif_naip_tokens"; TOK.mkdir(parents=True, exist_ok=True)
PATCH = CACHE / "gbif_naip_dinov3_patch32_v1"; PATCH.mkdir(parents=True, exist_ok=True)
IMG = CACHE / "env_priors" / "_naip2024_imagery"; IMG.mkdir(parents=True, exist_ok=True)
SCENES = CACHE / "env_priors" / "_naip2024_scenes"; SCENES.mkdir(parents=True, exist_ok=True)
CKPT = CACHE / "env_priors" / "naip_m2m2024_ckpt.pkl"
PATCH_CKPT = CACHE / "env_priors" / "naip_m2m2024_patch32_ckpt.pkl"
BASE = "https://m2m.cr.usgs.gov/api/api/json/stable"
USERNAME = os.environ.get("M2M_USER", "ecological")
TOKEN_PATH = Path(os.environ.get("USGS_M2M_TOKEN", str(Path.home() / ".usgs_m2m_token")))
DINO_SAT = "facebook/dinov3-vitl16-pretrain-sat493m"
EXT, PX = 300.0, 512                                            # 300 m patch centered on the obs, resampled to 512 px
INFERNO = matplotlib.colormaps["inferno"]
BATCH_TILES = int(os.environ.get("NAIP_BATCH_TILES", 4))        # scenes per M2M download-request (working-set bound)
DLW = int(os.environ.get("NAIP_DLW", 2))                        # parallel scene downloads
EMBED_BATCH = int(os.environ.get("NAIP_EMBED_BATCH", 2))        # DINOv3 ViT-L patch forward microbatch
PATCH_ROWS = int(os.environ.get("NAIP_PATCH_ROWS", 16))         # obs per scene chunk held through DINO + write buffer
SAVE_IMAGERY = os.environ.get("NAIP_SAVE_IMAGERY", "0") == "1"
SAVE_PATCH32 = os.environ.get("NAIP_SAVE_PATCH32", "1") == "1"
PATCH_VIEW = os.environ.get("NAIP_PATCH_VIEW", "rgb")
PATCH_DTYPE = np.float16 if os.environ.get("NAIP_PATCH_DTYPE", "float16") == "float16" else np.float32
NERSC_DIR = os.environ.get("NAIP_NERSC_DIR", "/global/cfs/cdirs/m5239/deepearth/naip2024_imagery")
if PATCH_VIEW not in {"rgb", "ir"}:
    raise ValueError("NAIP_PATCH_VIEW must be 'rgb' or 'ir'")


# ---------------- USGS M2M ----------------
def m2m(key, ep, body=None):
    h = {"X-Auth-Token": key} if key else {}
    r = requests.post(f"{BASE}/{ep}", headers=h, json=(body or {}), timeout=300); r.raise_for_status()
    j = r.json()
    if j.get("errorCode"): raise RuntimeError(f"M2M {ep}: {j['errorCode']} {j.get('errorMessage')}")
    return j["data"]

def login():
    return m2m(None, "login-token", {"username": USERNAME, "token": TOKEN_PATH.read_text().strip()})

def scene_urls(key, entity_ids):
    """scene-list-add -> download-options -> download-request -> poll download-retrieve -> {entityId: url}."""
    lst = f"naip2024_{int(time.time())}_{entity_ids[0]}"
    m2m(key, "scene-list-add", {"listId": lst, "datasetName": "naip", "idField": "entityId",
                                "entityIds": list(map(str, entity_ids))})
    opts = m2m(key, "download-options", {"datasetName": "naip", "listId": lst})
    dls = []
    for o in opts:                                              # first available product per entity (o or a secondaryDownload)
        for p in [o] + (o.get("secondaryDownloads") or []):
            if p.get("available") and p.get("id"):
                dls.append({"entityId": o["entityId"], "productId": p["id"]}); break
    if not dls:
        return {}
    label = f"naip2024_{int(time.time())}"
    resp = m2m(key, "download-request", {"downloads": dls, "label": label})
    urls = {str(d.get("entityId")): d["url"] for d in resp.get("availableDownloads", []) if d.get("url")}
    t0 = time.time()
    while len(urls) < len(dls) and time.time() - t0 < 900:      # poll for the ones still staging
        time.sleep(15)
        rr = m2m(key, "download-retrieve", {"label": label})
        for d in rr.get("available", []):
            if d.get("url"): urls[str(d.get("entityId"))] = d["url"]
    return urls

def fetch_scene(url, entity):
    """Stream-download a scene; unwrap the M2M zip -> a GeoTIFF path (or None on failure)."""
    raw = SCENES / f"{entity}.bin"; scene = SCENES / f"{entity}.tif"
    if scene.exists() and scene.stat().st_size > 1_000_000:
        return scene
    try:
        with requests.get(url, stream=True, timeout=1800) as r:
            r.raise_for_status()
            with open(raw, "wb") as f:
                for c in r.iter_content(1 << 20): f.write(c)
        if open(raw, "rb").read(2) == b"PK":
            with zipfile.ZipFile(raw) as zf:
                tifs = [n for n in zf.namelist() if n.lower().endswith((".tif", ".tiff", ".jp2"))]
                if not tifs: raw.unlink(); return None
                with zf.open(tifs[0]) as s, open(scene, "wb") as d: d.write(s.read())
            raw.unlink()
        else:
            raw.rename(scene)
        return scene
    except Exception as e:
        print(f"  fetch fail {entity}: {e}", flush=True)
        for p in (raw, scene):
            if p.exists(): p.unlink()
        return None


def catalog_scene_path(tile):
    path = tile.get("local_path")
    if not path:
        return None
    path = Path(path).expanduser()
    return path if path.exists() else None


def read_patch(src, lon, lat):
    """300 m box centered on (lon,lat) from an open scene -> (4,512,512) uint8, or None."""
    cx, cy = Transformer.from_crs(4326, src.crs, always_xy=True).transform(lon, lat)
    h = EXT / 2
    b = src.bounds
    if not (b.left + 30 <= cx - h and cx + h <= b.right - 30 and b.bottom + 30 <= cy - h and cy + h <= b.top - 30):
        return None                                            # obs too close to the scene edge (partial coverage)
    win = from_bounds(cx - h, cy - h, cx + h, cy + h, src.transform)
    arr = src.read(indexes=[1, 2, 3, 4], window=win, out_shape=(4, PX, PX), resampling=Resampling.bilinear).astype(np.uint8)
    return arr if np.isfinite(arr).all() and arr.max() > 0 else None


def iter_scene_patches(src, idxs, lon, lat):
    patches, keep = [], []
    for i in idxs:
        a = read_patch(src, lon[i], lat[i])
        if a is not None:
            patches.append(a)
            keep.append(i)
        if len(patches) >= PATCH_ROWS:
            yield patches, keep
            patches, keep = [], []
    if patches:
        yield patches, keep


def patch_latlon(lat, lon, patch_offset_m):
    lat = np.asarray(lat, np.float32)
    lon = np.asarray(lon, np.float32)
    north = patch_offset_m[..., 1]
    east = patch_offset_m[..., 0]
    dlat = north[None, :, :] / 111_320.0
    dlon = east[None, :, :] / (111_320.0 * np.cos(np.deg2rad(lat))[:, None, None] + 1e-6)
    return (lat[:, None, None] + dlat).astype(np.float32), (lon[:, None, None] + dlon).astype(np.float32)


def load_elevation(gid):
    path = CACHE / "gbif_elev.npz"
    if not path.exists():
        return np.full(len(gid), np.nan, np.float32)
    z = np.load(path)
    lut = dict(zip(z["gbifID"].astype(np.int64).tolist(), z["elev"].astype(np.float32).tolist()))
    return np.array([lut.get(int(g), np.nan) for g in gid], np.float32)


def load_event_day(gid):
    path = CACHE / "gbif_eventtime.npz"
    if not path.exists():
        return np.full(len(gid), np.nan, np.float32)
    z = np.load(path)
    key = "days" if "days" in z else ("event_day" if "event_day" in z else None)
    if key is None:
        return np.full(len(gid), np.nan, np.float32)
    lut = dict(zip(z["gbifID"].astype(np.int64).tolist(), z[key].astype(np.float32).tolist()))
    return np.array([lut.get(int(g), np.nan) for g in gid], np.float32)


_SF = {}
def _nersc_dir():
    """Lazy SFAPI handle to NERSC_DIR (client_id + secret from ~/.sfapi/sfapi.json), reused across uploads."""
    if "d" not in _SF:
        from sfapi_client import Client
        cfg = json.load(open(os.environ.get("SFAPI_JSON", str(Path.home() / ".sfapi/sfapi.json"))))
        pm = Client(cfg["client_id"], cfg["secret"]).compute("perlmutter")
        pm.run(f"mkdir -p {NERSC_DIR}")
        _SF["d"] = pm.ls(NERSC_DIR, directory=True)[0]
    return _SF["d"]

def nersc_put(local_path, remote_name):
    """Upload one file into NERSC_DIR via SFAPI. On the first failure (e.g. expired SFAPI secret) disable further
    attempts so imagery is simply kept LOCAL (uploaded later once creds refresh) instead of retrying auth per scene."""
    if _SF.get("disabled"):
        return False
    try:
        b = io.BytesIO(local_path.read_bytes()); b.filename = remote_name
        _nersc_dir().upload(b); return True
    except Exception as e:
        _SF["disabled"] = True
        print(f"  NERSC upload DISABLED ({repr(e)[:100]}); imagery kept local for later upload", flush=True); return False


def main():
    if not TILES_JSON.exists():
        raise SystemExit(f"{TILES_JSON} not found; set NAIP_TILES_JSON to the 2024 M2M tile catalog")
    tiles = json.load(open(TILES_JSON))
    tb = np.array([t["bbox"] for t in tiles], float)            # lon0,lat0,lon1,lat1
    cen = np.stack([(tb[:, 0] + tb[:, 2]) / 2, (tb[:, 1] + tb[:, 3]) / 2], 1)
    tree = cKDTree(cen)
    token_files = sorted(TOKENS.glob("*.npz"))
    if token_files:
        gid, lat, lon, obs_ord = [], [], [], []
        for file in token_files:
            z = np.load(file)
            gid.append(z["gbifID"].astype(np.int64))
            lat.append(z["lat"].astype(float))
            lon.append(z["lon"].astype(float))
            obs_ord.append(z["ord"].astype(np.int32) if "ord" in z else np.full(len(z["gbifID"]), -1, np.int32))
        gid, lat, lon, obs_ord = np.concatenate(gid), np.concatenate(lat), np.concatenate(lon), np.concatenate(obs_ord)
    else:
        z = np.load(COORDS)
        gid, lat, lon = z["gbifID"].astype(np.int64), z["lat"].astype(float), z["lon"].astype(float)
        obs_ord = np.full(len(gid), -1, np.int32)
    elev = load_elevation(gid)
    event_day = load_event_day(gid)
    # map each obs -> covering tile index (check nearest 8 tile centers for bbox containment)
    _, ii = tree.query(np.stack([lon, lat], 1), k=8)
    tile_of = np.full(len(gid), -1)
    for k in range(ii.shape[1]):
        ti = ii[:, k]; b = tb[ti]
        inside = (tile_of < 0) & (lon >= b[:, 0]) & (lon <= b[:, 2]) & (lat >= b[:, 1]) & (lat <= b[:, 3])
        tile_of[inside] = ti[inside]
    pool_done = pickle.load(open(CKPT, "rb")) if CKPT.exists() else set()
    for f in TOK.glob("chunk*.npz"):
        try: pool_done |= set(int(x) for x in np.load(f)["gbifID"])
        except Exception: pass
    patch_done = pickle.load(open(PATCH_CKPT, "rb")) if PATCH_CKPT.exists() else set()
    for f in PATCH.glob("chunk*.npz"):
        try: patch_done |= set(int(x) for x in np.load(f)["gbifID"])
        except Exception: pass
    done = patch_done if SAVE_PATCH32 else pool_done
    # scenes to process = covering tiles with >=1 undone obs
    by_scene = {}
    for i in range(len(gid)):
        if tile_of[i] >= 0 and int(gid[i]) not in done:
            by_scene.setdefault(tile_of[i], []).append(i)
    todo = sorted(by_scene, key=lambda t: -len(by_scene[t]))
    _max = int(os.environ.get("NAIP_MAX_SCENES", 0))           # >0 limits scenes (validation runs)
    if _max: todo = todo[:_max]
    print(f"{len(gid)} obs | {int((tile_of>=0).sum())} covered | {len(done)} done | {len(todo)} scenes to fetch", flush=True)
    if SAVE_PATCH32:
        centers = (np.arange(32, dtype=np.float32) + 0.5) / 32.0 - 0.5
        off_y, off_x = np.meshgrid(centers * EXT, centers * EXT, indexing="ij")
        patch_offset_m = np.stack([off_x, -off_y], -1).astype(np.float32)
        np.savez(PATCH / "manifest.npz",
            gbifID=gid.astype(np.int64),
            lat=lat.astype(np.float32),
            lon=lon.astype(np.float32),
            elev_m=elev.astype(np.float32),
            event_day=event_day.astype(np.float32),
            obs_ord=obs_ord.astype(np.int32),
            has_candidate_tile=(tile_of >= 0),
            patch_shape=np.array([32, 32, 1024], np.int16),
            patch_offset_m=patch_offset_m,
            dtype=np.array(str(np.dtype(PATCH_DTYPE))))
        with open(PATCH / "metadata.json", "w") as f:
            json.dump({
                "model": DINO_SAT,
                "view": PATCH_VIEW,
                "source_patch": [4, PX, PX],
                "patch_extent_m": EXT,
                "patch_offset_m": "manifest.npz:patch_offset_m [32,32,2], east/north meters from observation center",
                "patch_latlon": "chunk*.npz:patch_lat/patch_lon [N,32,32], derived from obs center + patch_offset_m",
                "center_elev_m": "manifest.npz:elev_m, copied from gbif_elev.npz when present; true per-patch DEM elevation is not generated here",
                "timestamp": "manifest.npz:event_day when gbif_eventtime.npz is present; shared by all patches for the observation",
                "patch_shape": [32, 32, 1024],
                "dtype": str(np.dtype(PATCH_DTYPE)),
                "row_key": "gbifID",
                "row_manifest": "manifest.npz",
                "missing_policy": "rows without a readable NAIP patch are absent from chunks and masked by gbifID",
            }, f, indent=2)
    emb = DINOv3Patch32(DINO_SAT, batch=EMBED_BATCH)
    buf = {k: [] for k in ("gbifID", "naip_year", "naip_scene", "rgb_pool", "ir_pool")}
    patch_buf = {k: [] for k in ("gbifID", "naip_year", "naip_scene", "patch", "patch_lat", "patch_lon")}
    chunk = len(list(TOK.glob("chunk*.npz"))); n_ok = 0; t0 = time.time()
    patch_chunk = len(list(PATCH.glob("chunk*.npz")))

    def flush_tokens():
        nonlocal chunk, pool_done
        if not buf["gbifID"]: return
        ids = list(buf["gbifID"])
        np.savez_compressed(TOK / f"chunk{chunk:04d}.npz",
            gbifID=np.array(buf["gbifID"], np.int64), naip_year=np.array(buf["naip_year"], np.int16),
            naip_scene=np.array(buf["naip_scene"], object),
            rgb_pool=np.stack(buf["rgb_pool"]).astype(np.float32), ir_pool=np.stack(buf["ir_pool"]).astype(np.float32))
        chunk += 1
        pool_done |= set(ids)
        pickle.dump(pool_done, open(CKPT, "wb"))
        for k in buf: buf[k] = []

    def flush_patch_tokens():
        nonlocal patch_chunk, patch_done
        if not SAVE_PATCH32 or not patch_buf["gbifID"]: return
        ids = list(patch_buf["gbifID"])
        np.savez_compressed(PATCH / f"chunk{patch_chunk:04d}.npz",
            gbifID=np.array(patch_buf["gbifID"], np.int64),
            naip_year=np.array(patch_buf["naip_year"], np.int16),
            naip_scene=np.array(patch_buf["naip_scene"], object),
            patch=np.stack(patch_buf["patch"]).astype(PATCH_DTYPE),
            patch_lat=np.stack(patch_buf["patch_lat"]).astype(np.float32),
            patch_lon=np.stack(patch_buf["patch_lon"]).astype(np.float32),
            has_naip=np.ones(len(patch_buf["gbifID"]), bool))
        patch_chunk += 1
        patch_done |= set(ids)
        pickle.dump(patch_done, open(PATCH_CKPT, "wb"))
        for k in patch_buf: patch_buf[k] = []

    key = None
    for w in range(0, len(todo), BATCH_TILES):
        batch = todo[w:w + BATCH_TILES]
        urls, paths, cleanup = {}, {}, {}
        m2m_batch = []
        for t in batch:
            local = catalog_scene_path(tiles[t])
            if local is not None:
                paths[t] = local
                cleanup[t] = False
            elif tiles[t].get("url"):
                urls[tiles[t]["entityId"]] = tiles[t]["url"]
            else:
                m2m_batch.append(t)
        if m2m_batch:
            ents = [tiles[t]["entityId"] for t in m2m_batch]
            if key is None:
                key = login()
            try:
                urls.update(scene_urls(key, ents))
            except Exception as e:                              # apiKey expires ~2h -> re-login
                print(f"  re-login ({e})", flush=True); key = login(); urls.update(scene_urls(key, ents))
        # download scenes in parallel
        with ThreadPoolExecutor(max_workers=DLW) as ex:
            futs = {ex.submit(fetch_scene, urls[tiles[t]["entityId"]], tiles[t]["entityId"]): t
                    for t in batch if t not in paths and tiles[t]["entityId"] in urls}
            for fut in as_completed(futs):
                p = fut.result()
                if p is not None:
                    paths[futs[fut]] = p
                    cleanup[futs[fut]] = True
        # tile + embed each scene, upload imagery, delete
        for t in batch:
            if t not in paths: continue
            ent = tiles[t]["entityId"]; yr = int(tiles[t]["displayId"].split("_")[-1][:4])
            idxs = by_scene[t]
            with rasterio.open(paths[t]) as src:
                for patches, keep in iter_scene_patches(src, idxs, lon, lat):
                    rgb_img = [a[:3] for a in patches]
                    need_pool = any(int(gid[i]) not in pool_done for i in keep)
                    rgb_patch = emb.patch32(rgb_img)
                    rgb = rgb_patch.reshape(rgb_patch.shape[0], -1, rgb_patch.shape[-1]).mean(1)
                    ir_patch = irp = None
                    if need_pool or PATCH_VIEW == "ir":
                        ir_img = [(INFERNO((a[3].astype(np.float32) - a[3].min()) / (np.ptp(a[3]) + 1e-6))[:, :, :3] * 255)
                                  .astype(np.uint8).transpose(2, 0, 1) for a in patches]
                        ir_patch = emb.patch32(ir_img)
                        irp = ir_patch.reshape(ir_patch.shape[0], -1, ir_patch.shape[-1]).mean(1)
                    patch_lat, patch_lon = (None, None)
                    if SAVE_PATCH32:
                        keep_idx = np.array(keep)
                        patch_lat, patch_lon = patch_latlon(lat[keep_idx], lon[keep_idx], patch_offset_m)
                    for j, i in enumerate(keep):
                        gid_i = int(gid[i])
                        if gid_i not in pool_done:
                            buf["gbifID"].append(gid_i); buf["naip_year"].append(yr); buf["naip_scene"].append(ent)
                            buf["rgb_pool"].append(rgb[j]); buf["ir_pool"].append(irp[j])
                        if SAVE_PATCH32:
                            patch_buf["gbifID"].append(gid_i); patch_buf["naip_year"].append(yr); patch_buf["naip_scene"].append(ent)
                            patch_buf["patch"].append(rgb_patch[j] if PATCH_VIEW == "rgb" else ir_patch[j])
                            patch_buf["patch_lat"].append(patch_lat[j]); patch_buf["patch_lon"].append(patch_lon[j])
                    n_ok += len(keep)
                    if SAVE_IMAGERY:                            # raw imagery -> per-chunk npz -> NERSC -> delete
                        imp = IMG / f"{ent}_{n_ok:08d}.npz"
                        np.savez_compressed(imp, gbifID=np.array([int(gid[i]) for i in keep], np.int64),
                                            patch=np.stack(patches).astype(np.uint8))
                        if nersc_put(imp, imp.name): imp.unlink()
                    if len(patch_buf["gbifID"]) >= 64:
                        flush_patch_tokens()
            if cleanup.get(t, True):
                paths[t].unlink(missing_ok=True)                # delete downloaded scene GeoTIFFs; keep catalog local_path files
        if len(buf["gbifID"]) >= 4000: flush_tokens()
        if len(patch_buf["gbifID"]) >= 64: flush_patch_tokens()
        print(f"  scenes {min(w+BATCH_TILES,len(todo))}/{len(todo)} | {n_ok} obs embedded | {n_ok/max(time.time()-t0,1):.1f} obs/s", flush=True)
    flush_tokens()
    flush_patch_tokens()
    print(f"DONE: {n_ok} obs on 2024 NAIP (scene-pinned).", flush=True)


if __name__ == "__main__":
    main()
