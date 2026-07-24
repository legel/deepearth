"""Embed NAIP aerial imagery per observation with DINOv3-SAT493M -> gbif_naip_tokens/ shards.

Fetch a 320px (~190m @ 0.6m) NAIP patch centered on each obs from Microsoft Planetary Computer STAC
(collection 'naip', 4-band R,G,B,NIR COGs, signed via the SAS REST API -- no planetary_computer pkg needed).
Two 3-band composites are mean-patch-pooled through facebook/dinov3-vitl16-pretrain-sat493m (ViT-L/16, 1024-d):
  rgb_pool = pool(R,G,B)   ir_pool = pool(R,G,NIR).   One STAC search + COG open per ~2km cell; windowed read
per obs; GPU-batched embed. Output schema EXACTLY matches the existing cache: per shard chunkNNNN.npz with
{gbifID[int64], naip_year[int16], rgb_pool[N,1024 f32], ir_pool[N,1024 f32]}. Failures are omitted (like the
existing cache). Resumable via processed_ids ckpt + already-present shard IDs.  `--test N` = sanity check vs cache.
"""
import os, sys, json, glob, time, pickle, socket, urllib.request, urllib.parse, warnings
import numpy as np, torch
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")
socket.setdefaulttimeout(30)                                    # bound every STAC/SAS call (PC hangs otherwise)
os.environ.setdefault("HF_HOME", "/home/photon/4tb/hf")
for _k, _v in {"GDAL_HTTP_TIMEOUT": "20", "GDAL_HTTP_CONNECTTIMEOUT": "10", "GDAL_HTTP_MAX_RETRY": "2",
               "GDAL_HTTP_RETRY_DELAY": "1", "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "VSI_CACHE": "TRUE"}.items():
    os.environ.setdefault(_k, _v)                               # bound /vsicurl reads so a stalled COG can't hang

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "..", "gbif_naip_tokens")
CKPT = os.path.join(HERE, "naip_ckpt.pkl")
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
SIGN = "https://planetarycomputer.microsoft.com/api/sas/v1/sign?href="
MODEL = "facebook/dinov3-vitl16-pretrain-sat493m"
DEV   = "cuda" if torch.cuda.is_available() else "cpu"
HALF, CELL, WORKERS, BATCH, SHARD = 160, 0.02, 12, 128, 4000     # 320px window; ~2km dedup cell
_sig = {}                                                        # item.id -> signed href (SAS ~valid 1h)

def sign(item):
    if item.id not in _sig:
        u = SIGN + urllib.parse.quote(item.assets["image"].href, safe="")
        _sig[item.id] = json.load(urllib.request.urlopen(u, timeout=30))["href"]
    return _sig[item.id]

def open_stac():
    from pystac_client import Client
    from pystac_client.stac_api_io import StacApiIO
    for k in range(6):
        try: return Client.open(STAC, stac_io=StacApiIO(timeout=30, max_retries=2))
        except Exception: time.sleep(3 * (k + 1))
    raise RuntimeError("cannot reach Planetary Computer STAC")

_iopool = ThreadPoolExecutor(max_workers=64)                    # bounds GDAL /vsicurl reads (see cell_patches)

def _read_cell(cand, obs):
    """cand: newest-year NAIP items overlapping the cell. Per obs: pick the quad that contains the point with the
    most edge-margin (best-centered patch), read a 320px window. Opens each needed COG once."""
    import rasterio
    from rasterio.warp import transform as rio_transform
    from rasterio.windows import Window
    bb = {i.id: i.bbox for i in cand}                          # [minlon,minlat,maxlon,maxlat] (WGS84)
    ds_cache, out = {}, {}
    try:
        for gid, la, lo in obs:
            best, bestm = None, -1e9
            for i in cand:
                x0, y0, x1, y1 = bb[i.id]
                if x0 <= lo <= x1 and y0 <= la <= y1:
                    m = min(lo - x0, x1 - lo, la - y0, y1 - la)
                    if m > bestm: bestm, best = m, i
            if best is None:
                best = min(cand, key=lambda i: ((bb[i.id][0]+bb[i.id][2])/2-lo)**2 + ((bb[i.id][1]+bb[i.id][3])/2-la)**2)
            try:
                if best.id not in ds_cache: ds_cache[best.id] = rasterio.open(sign(best))
                ds = ds_cache[best.id]
                xs, ys = rio_transform("EPSG:4326", ds.crs, [lo], [la]); r, c = ds.index(xs[0], ys[0])
                a = ds.read(window=Window(c - HALF, r - HALF, 2 * HALF, 2 * HALF), boundless=True, fill_value=0)
                if a.shape == (4, 2 * HALF, 2 * HALF) and a.any():
                    out[gid] = (a, best.datetime.year, best.id)          # best.id = exact NAIP tile/item id (pinned provenance)
            except Exception:
                pass
    finally:
        for ds in ds_cache.values():
            try: ds.close()
            except Exception: pass
    return out

def cell_patches(cat, key, obs):
    """obs: [(gid, lat, lon)] in one ~2km cell. One STAC search over the whole cell polygon; per-obs read of the
    best-centered newest-year NAIP quad. Returns {gid: (patch[4,H,W] uint8, year)}. The read runs in a bounded
    pool so a stalled Azure /vsicurl connection (GDAL ignores socket timeouts) is capped and the cell skipped."""
    x0, y0 = key[1] * CELL, key[0] * CELL; x1, y1 = x0 + CELL, y0 + CELL
    poly = {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}
    try:
        items = list(cat.search(collections=["naip"], intersects=poly).items())
        if not items: return {}
        yr = max(i.datetime.year for i in items)
        cand = [i for i in items if i.datetime.year == yr]           # newest year (2022 in CA); may be several quads
    except Exception:
        return {}
    try:
        return _iopool.submit(_read_cell, cand, obs).result(timeout=90)
    except Exception:
        return {}                                                    # timed-out read leaks its thread; skip cell

def load_coords():
    """obs_coords.npz plus any gbif_pollinator_obs/*.npz. Dedup by gbifID."""
    seen, gid, lat, lon = set(), [], [], []
    srcs = [os.path.join(HERE, "obs_coords.npz")] + sorted(glob.glob(os.path.join(HERE, "..", "gbif_pollinator_obs", "*.npz")))
    for f in srcs:
        if not os.path.exists(f): continue
        z = np.load(f, allow_pickle=True)
        zg, zla, zlo = z["gbifID"], z["lat"], z["lon"]          # materialize once (NpzFile re-reads per index)
        for i in range(len(zg)):
            g = int(zg[i])
            if g in seen: continue
            seen.add(g); gid.append(g); lat.append(float(zla[i])); lon.append(float(zlo[i]))
    return np.array(gid, np.int64), np.array(lat, np.float32), np.array(lon, np.float32)

def existing_ids():
    ids = set()
    for f in glob.glob(os.path.join(OUT, "chunk*.npz")):
        ids.update(int(x) for x in np.load(f)["gbifID"])
    return ids

def main():
    from transformers import AutoImageProcessor, AutoModel
    from PIL import Image
    test_n = int(sys.argv[sys.argv.index("--test") + 1]) if "--test" in sys.argv else 0
    os.makedirs(OUT, exist_ok=True)
    proc = AutoImageProcessor.from_pretrained(MODEL)
    mdl  = AutoModel.from_pretrained(MODEL).eval().to(DEV)
    nreg = mdl.config.num_register_tokens
    print(f"model loaded on {DEV}, opening STAC...", flush=True)
    cat  = open_stac()
    print("STAC ready", flush=True)

    def embed(patches):                                          # [(4,H,W) uint8] -> (rgb[N,1024], ir[N,1024]) f32
        rgb = [Image.fromarray(np.transpose(a[[0, 1, 2]], (1, 2, 0))) for a in patches]
        ir  = [Image.fromarray(np.transpose(a[[0, 1, 3]], (1, 2, 0))) for a in patches]
        def pool(imgs):
            px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(DEV)
            with torch.no_grad():
                h = mdl(px).last_hidden_state[:, 1 + nreg:].mean(1)   # mean over patch tokens
            return h.float().cpu().numpy()
        return pool(rgb), pool(ir)

    gid, lat, lon = load_coords()
    if test_n:                                                   # sanity: rebuild ids present in existing cache
        cache = {}
        for f in sorted(glob.glob(os.path.join(OUT, "chunk*.npz")))[:1]:
            z = np.load(f); zg, zr, zi, zy = z["gbifID"], z["rgb_pool"], z["ir_pool"], z["naip_year"]
            for i in range(len(zg)):
                cache[int(zg[i])] = (zr[i], zi[i], int(zy[i]))
        pos = {int(g): i for i, g in enumerate(gid)}
        pick = [g for g in cache if g in pos][:test_n]
        cells = {}
        for g in pick:
            i = pos[g]; k = (int(np.floor(lat[i] / CELL)), int(np.floor(lon[i] / CELL)))
            cells.setdefault(k, []).append((g, float(lat[i]), float(lon[i])))
        got = {}
        for ci, (k, o) in enumerate(cells.items()):
            got.update(cell_patches(cat, k, o)); print(f"  fetched cell {ci+1}/{len(cells)}", flush=True)
        pick = [g for g in pick if g in got]
        rr, ii = embed([got[g][0] for g in pick])
        cos = lambda a, b: float((a.astype(np.float64) * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        print(f"\n=== SANITY: rebuilt vs existing cache ({len(pick)} obs) ===")
        print(f"{'gbifID':>12} {'year':>5} {'rgb_cos':>8} {'ir_cos':>7}")
        rc, ic = [], []
        for j, g in enumerate(pick):
            cr = cos(rr[j], cache[g][0]); ci = cos(ii[j], cache[g][1]); rc.append(cr); ic.append(ci)
            print(f"{g:>12} {got[g][1]:>5} {cr:>8.3f} {ci:>7.3f}")
        print(f"{'MEAN':>12} {'':>5} {np.mean(rc):>8.3f} {np.mean(ic):>7.3f}  (random baseline ~0.28)")
        print(f"rebuilt rgb_pool[0][:5] {np.round(rr[0][:5],4)}  norm {np.linalg.norm(rr[0]):.2f}")
        print(f"cache   rgb_pool[0][:5] {np.round(cache[pick[0]][0][:5],4)}  norm {np.linalg.norm(cache[pick[0]][0]):.2f}")
        return

    done = pickle.load(open(CKPT, "rb")) if os.path.exists(CKPT) else set()
    done |= existing_ids()
    pos = {int(g): i for i, g in enumerate(gid)}
    cells = {}
    for i, g in enumerate(gid):
        if int(g) in done: continue
        k = (int(np.floor(lat[i] / CELL)), int(np.floor(lon[i] / CELL)))
        cells.setdefault(k, []).append((int(g), float(lat[i]), float(lon[i])))
    todo = list(cells)
    nshard = len(glob.glob(os.path.join(OUT, "chunk*.npz")))
    print(f"{len(gid)} obs, {len(done)} done, {len(todo)} cells to fetch, next shard chunk{nshard:04d} on {DEV}", flush=True)

    buf = {"gbifID": [], "naip_year": [], "naip_scene": [], "rgb_pool": [], "ir_pool": []}
    pend_g, pend_y, pend_s, pend_p = [], [], [], []
    t0 = time.time(); n_ok = n_fail = 0

    def flush_embed():
        if not pend_p: return
        rr, ii = embed(pend_p)
        buf["gbifID"] += pend_g; buf["naip_year"] += pend_y; buf["naip_scene"] += pend_s
        buf["rgb_pool"] += list(rr); buf["ir_pool"] += list(ii)
        pend_g.clear(); pend_y.clear(); pend_s.clear(); pend_p.clear()

    def write_shard():
        nonlocal nshard
        while len(buf["gbifID"]) >= SHARD:
            sl = slice(0, SHARD)
            np.savez(os.path.join(OUT, f"chunk{nshard:04d}.npz"),
                     gbifID=np.array(buf["gbifID"][sl], np.int64),
                     naip_year=np.array(buf["naip_year"][sl], np.int16),
                     naip_scene=np.array(buf["naip_scene"][sl], object),   # exact NAIP tile id per obs (pinned provenance)
                     rgb_pool=np.array(buf["rgb_pool"][sl], np.float32),
                     ir_pool=np.array(buf["ir_pool"][sl], np.float32))
            for k in buf: del buf[k][sl]
            nshard += 1; print(f"  wrote chunk{nshard-1:04d}", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(cell_patches, cat, k, cells[k]): k for k in todo}
        for n, fut in enumerate(as_completed(futs)):
            k = futs[fut]; res = fut.result()
            for gid_, (patch, yr, scene) in res.items():
                pend_g.append(gid_); pend_y.append(yr); pend_s.append(scene); pend_p.append(patch)
            done.update(g for g, _, _ in cells[k]); n_ok += len(res); n_fail += len(cells[k]) - len(res)
            if len(pend_p) >= BATCH: flush_embed(); write_shard()
            if (n + 1) % 200 == 0:
                pickle.dump(done, open(CKPT, "wb"))
                el = time.time() - t0
                print(f"  {n+1}/{len(todo)} cells | ok {n_ok} fail {n_fail} | {el:.0f}s | "
                      f"eta {el/(n+1)*(len(todo)-n-1)/60:.1f}min", flush=True)
    flush_embed(); write_shard()
    if buf["gbifID"]:                                            # final partial shard
        np.savez(os.path.join(OUT, f"chunk{nshard:04d}.npz"),
                 gbifID=np.array(buf["gbifID"], np.int64), naip_year=np.array(buf["naip_year"], np.int16),
                 rgb_pool=np.array(buf["rgb_pool"], np.float32), ir_pool=np.array(buf["ir_pool"], np.float32))
        print(f"  wrote chunk{nshard:04d} (partial {len(buf['gbifID'])})", flush=True)
    pickle.dump(done, open(CKPT, "wb"))
    print(f"DONE: ok {n_ok} fail {n_fail}", flush=True)

if __name__ == "__main__":
    main()
