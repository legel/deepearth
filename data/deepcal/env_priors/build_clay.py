"""Fetch a cloud-free 2025 Sentinel-2 L2A patch per observation and embed it with the Clay v1.5 foundation model
(geospatial MAE, https://github.com/Clay-foundation/model). One STAC fetch per ~1.1km cell; each obs embedded with
its own lat/lon token so co-located obs still differ. Output gbif_clay_tokens.npz {gbifID, clay[N,1024] f16,
clay_year[N] i16, has_clay[N]} -- matches the existing cache exactly (Clay large -> 1024-d CLS, RAW: the autoresearch
loader normalizes). Resumable via clay_ckpt.pkl. On any failure has_clay=False. env: CLAY_CKPT, CLAY_METADATA (Clay
configs/metadata.yaml), CLAY_MAX (limit obs, testing), CLAY_OUT (output path override, testing), STAC_API/STAC_SIGN."""
import os, sys, time, pickle, math, warnings
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "x") != "" and __import__("torch").cuda.is_available() else "cpu"
CELL = 0.01                 # ~1.1km dedup cell; a 224px@10m (2.24km) patch on the cell center covers all its obs
SIZE, GSD = 224, 10         # Clay chip: 224x224 px @ 10 m ground sample distance (patch_size 8 -> 28x28 tokens)
YEAR = 2025                 # prefer a cloud-free 2025 scene (existing cache is clay_year=2025 throughout)
CLOUD_LT = 20               # % max scene cloud cover
WORKERS = 12
DIM = 1024                  # Clay v1.5 large encoder embedding dim (== existing cache D)
CKPT = os.path.join(HERE, "clay_ckpt.pkl")
STAC_API = os.environ.get("STAC_API", "https://planetarycomputer.microsoft.com/api/stac/v1")
# Clay sentinel-2-l2a band order -> STAC asset keys (Planetary Computer). Element84 earth-search uses lowercase
# names (blue,green,...,swir22); flip STAC_SIGN=0 + STAC_API to earth-search and swap this map if used.
CLAY_BANDS = ["blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16", "swir22"]
ASSET = {"blue": "B02", "green": "B03", "red": "B04", "rededge1": "B05", "rededge2": "B06",
         "rededge3": "B07", "nir": "B08", "nir08": "B8A", "swir16": "B11", "swir22": "B12"}
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF",
                GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="1", VSI_CACHE="TRUE")


def utm_epsg(lon, lat):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) / 6) + 1


# ---- Clay datacube position encodings (verbatim from the Clay v1.5 embeddings tutorial) ----
def enc_time(d):
    wk = d.isocalendar().week * 2 * math.pi / 52
    hr = d.hour * 2 * math.pi / 24
    return [math.sin(wk), math.cos(wk), math.sin(hr), math.cos(hr)]

def enc_latlon(lat, lon):
    la, lo = math.radians(lat), math.radians(lon)
    return [math.sin(la), math.cos(la), math.sin(lo), math.cos(lo)]


# ---- Sentinel-2 L2A patch via STAC (rasterio windowed/warped reads; no stackstac dependency) ----
def stac_client():
    from pystac_client import Client
    if os.environ.get("STAC_SIGN", "1") == "1" and "planetarycomputer" in STAC_API:
        import planetary_computer as pc
        return Client.open(STAC_API, modifier=pc.sign_inplace)
    return Client.open(STAC_API)

def best_item(cat, lon, lat):
    s = cat.search(collections=["sentinel-2-l2a"], intersects={"type": "Point", "coordinates": [lon, lat]},
                   datetime=f"{YEAR}-01-01/{YEAR}-12-31", query={"eo:cloud_cover": {"lt": CLOUD_LT}},
                   sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}], max_items=1)
    items = list(s.items())
    return items[0] if items else None

def fetch_patch(cat, lon, lat):
    """Return (pixels[10,SIZE,SIZE] float32 raw DN, scene datetime) or None. Reprojects each band COG onto a
    SIZE x SIZE, 10 m UTM grid centered on (lon,lat) via a WarpedVRT (resamples 20 m red-edge/SWIR bands to 10 m)."""
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling
    from rasterio.crs import CRS
    from affine import Affine
    from pyproj import Transformer
    item = best_item(cat, lon, lat)
    if item is None:
        return None
    epsg = item.properties.get("proj:epsg") or utm_epsg(lon, lat)
    cx, cy = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    half = SIZE * GSD / 2.0
    dst_t = Affine(GSD, 0, cx - half, 0, -GSD, cy + half)                 # north-up 10 m grid on the point
    dst_crs = CRS.from_epsg(epsg)
    bands = []
    with rasterio.Env(**GDAL_ENV):
        for b in CLAY_BANDS:
            a = item.assets.get(ASSET[b]) or item.assets.get(b)
            if a is None:
                return None
            with rasterio.open(a.href) as src, WarpedVRT(src, crs=dst_crs, transform=dst_t, width=SIZE,
                                                         height=SIZE, resampling=Resampling.bilinear) as vrt:
                bands.append(vrt.read(1).astype(np.float32))
    px = np.stack(bands)                                                  # (10,SIZE,SIZE), raw DN (Clay norms in DN)
    if not np.isfinite(px).any() or (px <= 0).all():
        return None
    return px, item.datetime, item.id                                     # item.id = exact S2 scene, for pinned provenance


# ---- Clay v1.5 model (lazy; needs the claymodel package + checkpoint + metadata.yaml) ----
def load_clay():
    ckpt = os.environ.get("CLAY_CKPT", os.path.join(HERE, "clay-v1.5.ckpt"))
    meta_path = os.environ.get("CLAY_METADATA", os.path.join(HERE, "clay_metadata.yaml"))
    if not (os.path.exists(ckpt) and os.path.exists(meta_path)):
        raise FileNotFoundError(f"Clay checkpoint/metadata missing (ckpt={ckpt}, metadata={meta_path})")
    import torch, yaml
    from claymodel.module import ClayMAEModule
    model = ClayMAEModule.load_from_checkpoint(ckpt, model_size="large", metadata_path=meta_path,
            dolls=[16, 32, 64, 128, 256, 768, 1024], doll_weights=[1] * 7, mask_ratio=0.0,
            shuffle=False).eval().to(DEV)
    s2 = yaml.safe_load(open(meta_path))["sentinel-2-l2a"]["bands"]
    meta = dict(mean=np.array([s2["mean"][b] for b in CLAY_BANDS], np.float32),
                std=np.array([s2["std"][b] for b in CLAY_BANDS], np.float32),
                waves=[s2["wavelength"][b] for b in CLAY_BANDS])
    return model, meta

def embed(model, meta, px, date, obs):
    """px shared by all `obs` (gid,lat,lon) in a cell; encode with per-obs latlon token. Returns {gid: f16[1024]}."""
    import torch
    p = (px - meta["mean"][:, None, None]) / meta["std"][:, None, None]
    pix = torch.from_numpy(p).float().unsqueeze(0).repeat(len(obs), 1, 1, 1)
    cube = dict(pixels=pix.to(DEV),
                time=torch.tensor([enc_time(date)] * len(obs)).float().to(DEV),
                latlon=torch.tensor([enc_latlon(la, lo) for _, la, lo in obs]).float().to(DEV),
                waves=torch.tensor(meta["waves"]).float().to(DEV),
                gsd=torch.tensor(float(GSD)).to(DEV))
    with torch.no_grad():
        enc, *_ = model.model.encoder(cube)                              # (B, 1+ntok, 1024); [:,0] is the CLS token
    e = enc[:, 0, :].cpu().numpy().astype(np.float16)
    return {int(g): e[i] for i, (g, _, _) in enumerate(obs)}


# ---- driver ----
def load_coords():
    """obs_coords.npz union with pollinator obs (gbif_pollinator_obs/*.npz); obs_coords wins on duplicate gbifID."""
    import glob
    z = np.load(os.path.join(HERE, "obs_coords.npz"))
    gid = list(map(int, z["gbifID"])); lat = list(map(float, z["lat"])); lon = list(map(float, z["lon"]))
    seen = set(gid)
    for f in sorted(glob.glob(os.path.join(HERE, "..", "gbif_pollinator_obs", "*.npz"))):
        p = np.load(f, allow_pickle=True)
        for g, la, lo in zip(p["gbifID"], p["lat"], p["lon"]):
            if int(g) not in seen:
                seen.add(int(g)); gid.append(int(g)); lat.append(float(la)); lon.append(float(lo))
    return np.array(gid, np.int64), np.array(lat, np.float32), np.array(lon, np.float32)

def save(out, gid, done):
    clay = np.zeros((len(gid), DIM), np.float16); year = np.zeros(len(gid), np.int16)
    scene = np.array([""] * len(gid), object); have = np.zeros(len(gid), bool)
    for i, g in enumerate(gid):
        v = done.get(int(g))
        if v is not None:
            clay[i], year[i] = v[0], v[1]; scene[i] = v[2] if len(v) > 2 else ""; have[i] = True
    np.savez(out, gbifID=gid, clay=clay, clay_year=year, clay_scene=scene, has_clay=have)  # clay_scene = S2 item id (provenance)
    return have.sum()

def main():
    gid, lat, lon = load_coords()
    if os.environ.get("CLAY_MAX"):
        n = int(os.environ["CLAY_MAX"]); gid, lat, lon = gid[:n], lat[:n], lon[:n]
    out = os.environ.get("CLAY_OUT", os.path.join(HERE, "..", "gbif_clay_tokens.npz"))
    cells = {}
    for i in range(len(gid)):
        cells.setdefault((int(math.floor(lat[i] / CELL)), int(math.floor(lon[i] / CELL))), []).append(
            (int(gid[i]), float(lat[i]), float(lon[i])))
    done = pickle.load(open(CKPT, "rb")) if os.path.exists(CKPT) else {}
    print(f"{len(gid)} obs, {len(cells)} cells, {len(done)} done | out={out} | dev={DEV}", flush=True)

    try:
        model, meta = load_clay()
    except Exception as e:
        n = save(out, gid, done)
        print(f"BLOCKED: Clay unavailable ({e}). Wrote schema-only {out}: clay[{len(gid)},{DIM}] f16, "
              f"has_clay={n} True. Install claymodel + checkpoint/metadata + planetary-computer, then rerun.",
              flush=True)
        return

    cat = stac_client()
    todo = [c for c in cells if not all(o[0] in done for o in cells[c])]
    t0 = time.time(); ok = fail = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch_patch, cat, (c[1] + 0.5) * CELL, (c[0] + 0.5) * CELL): c for c in todo}
        for i, fut in enumerate(as_completed(futs)):
            obs = cells[futs[fut]]
            try:
                r = fut.result()
                if r is not None:
                    px, date, sid = r
                    for g, v in embed(model, meta, px, date, obs).items():
                        done[g] = (v, date.year, sid)                     # record scene id per obs (provenance)
                    ok += 1
                else:
                    fail += 1
            except Exception:
                fail += 1
            if (i + 1) % 200 == 0:
                pickle.dump(done, open(CKPT, "wb"))
                el = time.time() - t0
                print(f"  {i+1}/{len(todo)} cells | ok {ok} fail {fail} | {len(done)} obs | {el:.0f}s | "
                      f"eta {el/(i+1)*(len(todo)-i-1)/60:.1f}min", flush=True)
    pickle.dump(done, open(CKPT, "wb"))
    n = save(out, gid, done)
    print(f"DONE: {n}/{len(gid)} obs have clay ({100*n/len(gid):.1f}%) -> {out}", flush=True)


if __name__ == "__main__":
    main()
