"""2024 NAIP per-observation imagery + DINOv3-SAT493M embeddings via USGS EROS M2M (2024-EXCLUSIVE).

Planetary Computer only serves 2022 NAIP for CA; the 2024 CNIR acquisition lives on USGS M2M. For each NAIP-2024
scene that covers >=1 observation (mapped from the cached statewide catalog naip2024_tiles.json), download the scene
ONCE via M2M, window-read a 300x300 m patch CENTERED on every observation the scene covers -> 512x512 4-band uint8,
then (a) accumulate the raw imagery patch -> per-scene npz streamed to NERSC and deleted locally, and (b) embed with
DINOv3-SAT493M (NIR rendered through the inferno colormap) -> rgb_pool + ir_pool (1024-d each, mean over patch
tokens) matching the pooled-NAIP schema. Scene GeoTIFF is deleted after tiling. Scene-pinned: naip_scene = M2M
entityId, naip_year = acquisition year. Streaming keeps the working set small (BATCH_TILES scenes at a time);
resumable via a checkpoint of finished gbifIDs.

Output: gbif_naip_tokens/chunk*.npz {gbifID, naip_year, naip_scene, rgb_pool[N,1024]f32, ir_pool[N,1024]f32}
Raw imagery (optional, NAIP_SAVE_IMAGERY=1): NERSC <NERSC_DIR>/<entityId>.npz {gbifID, patch[n,512,512,4]uint8}
env: USGS_M2M_TOKEN (default ~/.usgs_m2m_token), M2M_USER (ecological), NAIP_BATCH_TILES, NAIP_SAVE_IMAGERY,
     NAIP_NERSC_DIR, HF cache. Run on the H200: python build_naip_m2m2024.py
"""
import os, sys, io, time, json, pickle, zipfile, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, requests, torch, rasterio, matplotlib
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from pyproj import Transformer
from scipy.spatial import cKDTree
from transformers import AutoModel, AutoImageProcessor
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
TILES_JSON = HERE / "naip2024_tiles.json"                       # {entityId, displayId, bbox:[lon0,lat0,lon1,lat1]} statewide catalog
COORDS = HERE / "obs_coords.npz"                                # {gbifID, lat, lon}
TOK = HERE / "gbif_naip_tokens"; TOK.mkdir(exist_ok=True)
IMG = HERE / "_naip2024_imagery"; IMG.mkdir(exist_ok=True)      # transient per-scene imagery npz -> NERSC -> deleted
SCENES = HERE / "_naip2024_scenes"; SCENES.mkdir(exist_ok=True) # transient scene GeoTIFFs -> deleted after tiling
CKPT = HERE / "naip_m2m2024_ckpt.pkl"
BASE = "https://m2m.cr.usgs.gov/api/api/json/stable"
USERNAME = os.environ.get("M2M_USER", "ecological")
TOKEN_PATH = Path(os.environ.get("USGS_M2M_TOKEN", str(Path.home() / ".usgs_m2m_token")))
DINO_SAT = "facebook/dinov3-vitl16-pretrain-sat493m"
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
EXT, PX = 300.0, 512                                            # 300 m patch centered on the obs, resampled to 512 px
INFERNO = matplotlib.colormaps["inferno"]
BATCH_TILES = int(os.environ.get("NAIP_BATCH_TILES", 24))       # scenes per M2M download-request (working-set bound)
DLW = int(os.environ.get("NAIP_DLW", 8))                        # parallel scene downloads
SAVE_IMAGERY = os.environ.get("NAIP_SAVE_IMAGERY", "1") == "1"
NERSC_DIR = os.environ.get("NAIP_NERSC_DIR", "/global/cfs/cdirs/m5239/deepearth/naip2024_imagery")


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


# ---------------- DINOv3-SAT embedding ----------------
class Embedder:
    def __init__(self):
        self.proc = AutoImageProcessor.from_pretrained(DINO_SAT)
        self.mdl = AutoModel.from_pretrained(DINO_SAT).eval().to(DEV)
        self.nreg = self.mdl.config.num_register_tokens
        self.mean = np.array(self.proc.image_mean)[:, None, None]; self.std = np.array(self.proc.image_std)[:, None, None]

    @torch.no_grad()
    def pool(self, ims):                                        # list of [3,512,512] uint8 -> [N,1024] mean over patch tokens
        x = np.stack([(im.astype(np.float32) / 255.0 - self.mean) / self.std for im in ims])
        out = []
        for i in range(0, len(x), 64):
            xt = torch.tensor(x[i:i + 64], dtype=torch.float32, device=DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                h = self.mdl(pixel_values=xt).last_hidden_state[:, 1 + self.nreg:]
            out.append(h.float().mean(1).cpu().numpy())
        return np.concatenate(out)


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
    """Upload one file into NERSC_DIR via SFAPI (returns True on success)."""
    try:
        b = io.BytesIO(local_path.read_bytes()); b.filename = remote_name
        _nersc_dir().upload(b); return True
    except Exception as e:
        print(f"  NERSC upload fail {remote_name}: {e}", flush=True); return False


def main():
    tiles = json.load(open(TILES_JSON))
    tb = np.array([t["bbox"] for t in tiles], float)            # lon0,lat0,lon1,lat1
    cen = np.stack([(tb[:, 0] + tb[:, 2]) / 2, (tb[:, 1] + tb[:, 3]) / 2], 1)
    tree = cKDTree(cen)
    z = np.load(COORDS)
    gid, lat, lon = z["gbifID"].astype(np.int64), z["lat"].astype(float), z["lon"].astype(float)
    # map each obs -> covering tile index (check nearest 8 tile centers for bbox containment)
    _, ii = tree.query(np.stack([lon, lat], 1), k=8)
    tile_of = np.full(len(gid), -1)
    for k in range(ii.shape[1]):
        ti = ii[:, k]; b = tb[ti]
        inside = (tile_of < 0) & (lon >= b[:, 0]) & (lon <= b[:, 2]) & (lat >= b[:, 1]) & (lat <= b[:, 3])
        tile_of[inside] = ti[inside]
    done = pickle.load(open(CKPT, "rb")) if CKPT.exists() else set()
    for f in TOK.glob("chunk*.npz"):
        try: done |= set(int(x) for x in np.load(f)["gbifID"])
        except Exception: pass
    # scenes to process = covering tiles with >=1 undone obs
    by_scene = {}
    for i in range(len(gid)):
        if tile_of[i] >= 0 and int(gid[i]) not in done:
            by_scene.setdefault(tile_of[i], []).append(i)
    todo = sorted(by_scene, key=lambda t: -len(by_scene[t]))
    _max = int(os.environ.get("NAIP_MAX_SCENES", 0))           # >0 limits scenes (validation runs)
    if _max: todo = todo[:_max]
    print(f"{len(gid)} obs | {int((tile_of>=0).sum())} covered | {len(done)} done | {len(todo)} scenes to fetch", flush=True)
    emb = Embedder()
    buf = {k: [] for k in ("gbifID", "naip_year", "naip_scene", "rgb_pool", "ir_pool")}
    chunk = len(list(TOK.glob("chunk*.npz"))); n_ok = 0; t0 = time.time()

    def flush_tokens():
        nonlocal chunk
        if not buf["gbifID"]: return
        np.savez_compressed(TOK / f"chunk{chunk:04d}.npz",
            gbifID=np.array(buf["gbifID"], np.int64), naip_year=np.array(buf["naip_year"], np.int16),
            naip_scene=np.array(buf["naip_scene"], object),
            rgb_pool=np.stack(buf["rgb_pool"]).astype(np.float32), ir_pool=np.stack(buf["ir_pool"]).astype(np.float32))
        chunk += 1
        for k in buf: buf[k] = []

    key = login()
    for w in range(0, len(todo), BATCH_TILES):
        batch = todo[w:w + BATCH_TILES]
        ents = [tiles[t]["entityId"] for t in batch]
        try:
            urls = scene_urls(key, ents)
        except Exception as e:                                  # apiKey expires ~2h -> re-login
            print(f"  re-login ({e})", flush=True); key = login(); urls = scene_urls(key, ents)
        # download scenes in parallel
        paths = {}
        with ThreadPoolExecutor(max_workers=DLW) as ex:
            futs = {ex.submit(fetch_scene, urls[tiles[t]["entityId"]], tiles[t]["entityId"]): t
                    for t in batch if tiles[t]["entityId"] in urls}
            for fut in as_completed(futs):
                p = fut.result()
                if p is not None: paths[futs[fut]] = p
        # tile + embed each scene, upload imagery, delete
        for t in batch:
            if t not in paths: continue
            ent = tiles[t]["entityId"]; yr = int(tiles[t]["displayId"].split("_")[-1][:4])
            idxs = by_scene[t]
            patches, keep = [], []
            with rasterio.open(paths[t]) as src:
                for i in idxs:
                    a = read_patch(src, lon[i], lat[i])
                    if a is not None: patches.append(a); keep.append(i)
            if patches:
                rgb = emb.pool([a[:3] for a in patches])
                irp = emb.pool([(INFERNO((a[3].astype(np.float32) - a[3].min()) / (np.ptp(a[3]) + 1e-6))[:, :, :3] * 255)
                                .astype(np.uint8).transpose(2, 0, 1) for a in patches])
                for j, i in enumerate(keep):
                    buf["gbifID"].append(int(gid[i])); buf["naip_year"].append(yr); buf["naip_scene"].append(ent)
                    buf["rgb_pool"].append(rgb[j]); buf["ir_pool"].append(irp[j])
                n_ok += len(keep)
                if SAVE_IMAGERY:                                # raw imagery -> per-scene npz -> NERSC -> delete
                    imp = IMG / f"{ent}.npz"
                    np.savez_compressed(imp, gbifID=np.array([int(gid[i]) for i in keep], np.int64),
                                        patch=np.stack(patches).astype(np.uint8))
                    if nersc_put(imp, f"{ent}.npz"): imp.unlink()
            paths[t].unlink(missing_ok=True)                    # delete the scene GeoTIFF (streaming)
        if len(buf["gbifID"]) >= 4000: flush_tokens()
        pickle.dump(done | set(buf["gbifID"]), open(CKPT, "wb"))
        print(f"  scenes {min(w+BATCH_TILES,len(todo))}/{len(todo)} | {n_ok} obs embedded | {n_ok/max(time.time()-t0,1):.1f} obs/s", flush=True)
    flush_tokens()
    print(f"DONE: {n_ok} obs on 2024 NAIP (scene-pinned).", flush=True)


if __name__ == "__main__":
    main()
