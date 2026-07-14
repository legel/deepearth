"""GPU hydrology (TWI/HAND/catchment) + Winstral Sx wind exposure from 3DEP 1m DEM, batched in PyTorch.
Flow accumulation = iterative upstream propagation (D8 -> a forest of in-trees, so exact, no double-count),
converging in O(flow-path) cheap GPU steps over a whole batch of patches. Fetch (threaded, hardened) feeds GPU
batches, decoupling I/O from compute. Output: gbif_hydro_tokens.npz {gbifID, hydro[N,6], has_hydro, hydro_scene}.

DEM source = 3DEP 1m staged COGs on AWS (prd-tnm), resolved per cell via TNM Access and read with windowed
/vsicurl range requests decimated to 2m -- robust, unthrottled, scene-pinned (see build_topo.py; reuses its tile
index topo_tileidx.json since hydro shares the same cell grid + obs_coords). Resumable."""
import os, time, pickle, warnings, math, json
import numpy as np, torch, torch.nn.functional as F
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import rasterio
from rasterio.windows import from_bounds
from affine import Affine
from build_topo import tnm_tiles, transformer, build_index, IDXP   # reuse tile resolution (same cells)
warnings.filterwarnings("ignore")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
HERE = os.path.dirname(os.path.abspath(__file__)); DEV = "cuda"
CELL = 0.002; HALF = 256; N = 256; RES = 2.0; WORKERS = 24; GPU_BATCH = 96      # 512 m patch @ 2 m (256 px)
CKPT = os.path.join(HERE, "hydrowind_ckpt.pkl")
DIRS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
OPP = [7,6,5,4,3,2,1,0]
DIST = torch.tensor([2**.5,1,2**.5,1,1,2**.5,1,2**.5])

def read_tile(url, cell_list):
    """Open one 3DEP 1m COG ONCE (amortizing the /vsicurl open) and read a 512 m patch decimated to N=256 px (2 m)
    for every cell assigned to it. Returns (patches, failed) where patches=[(cell,crs,out_tf,dem,fname)] and
    failed=[cells whose patch is mostly nodata -> retry an alternate overlapping-project tile]."""
    fname = url.rsplit("/", 1)[-1]; patches = []; failed = []
    try:
        with rasterio.open("/vsicurl/" + url) as ds:
            tf = transformer(ds.crs)
            for c in cell_list:
                latc, lonc = (c[0] + .5) * CELL, (c[1] + .5) * CELL
                cx, cy = tf.transform(lonc, latc)
                if not (np.isfinite(cx) and np.isfinite(cy)):
                    failed.append(c); continue
                w = from_bounds(cx - HALF, cy - HALF, cx + HALF, cy + HALF, ds.transform)
                dem = ds.read(1, window=w, out_shape=(N, N), boundless=True, fill_value=np.nan).astype(np.float32)
                out_tf = ds.window_transform(w) * Affine.scale(w.width / N, w.height / N)   # native->decimated grid
                dem = np.where(np.isfinite(dem) & (dem > -1e4), dem, np.nan)
                if np.isnan(dem).mean() > 0.5:
                    failed.append(c); continue
                patches.append((c, ds.crs, out_tf, np.nan_to_num(dem, nan=np.nanmean(dem)).astype(np.float32), fname))
        return patches, failed
    except Exception:
        return [], list(cell_list)                          # whole-tile failure -> all cells retry alternates

def neighbor(x, k):                                    # shift x[B,H,W] to bring direction-k neighbor onto each cell
    dr, dc = DIRS[k]; return torch.roll(x, shifts=(-dr, -dc), dims=(1, 2))

@torch.no_grad()
def physics(dem, dist):                                # dem [B,H,W] on GPU -> per-cell rasters
    B, H, W = dem.shape; dev = dem.device
    z = dem.clone()
    for _ in range(60):                                # Planchon-Darboux-style fill: raise cells below their lowest neighbor
        mn = torch.stack([neighbor(z, k) for k in range(8)], 0).amin(0)
        z = torch.maximum(dem, torch.minimum(z, mn + 1e-3))
    slope_nb = torch.stack([(z - neighbor(z, k)) / dist[k] for k in range(8)], 0)   # [8,B,H,W]
    rd = slope_nb.argmax(0)                            # receiver direction (steepest descent)
    acc = torch.ones_like(z)
    for _ in range(400):
        inflow = torch.zeros_like(z)
        for k in range(8):
            inflow = inflow + neighbor(acc, k) * (neighbor(rd, k) == OPP[k]).float()
        new = 1.0 + inflow
        if torch.max((new - acc).abs()) < 0.5: acc = new; break
        acc = new
    dzdy = (neighbor(z, 6) - neighbor(z, 1)) / (2 * RES); dzdx = (neighbor(z, 4) - neighbor(z, 3)) / (2 * RES)
    slope = torch.atan(torch.hypot(dzdx, dzdy)); tanb = torch.clamp(torch.tan(slope), min=1e-3)
    sca = acc * RES; twi = torch.log(sca / tanb)
    stream = acc > 2000
    se = z.clone()                                     # HAND: propagate downstream stream elevation upstream along D8
    for _ in range(400):
        recv_se = torch.zeros_like(z)
        for k in range(8):                             # a cell's receiver is neighbor k where rd==k; pull its se
            recv_se = recv_se + neighbor(se, k) * (rd == k).float()
        new_se = torch.where(stream, z, recv_se)
        if torch.max((new_se - se).abs()) < 1e-2: se = new_se; break
        se = new_se
    hand = torch.clamp(z - se, min=0)
    return z, twi, hand, torch.log(sca), slope, dzdx, dzdy

@torch.no_grad()
def winstral_sx(dem, wind_from_deg, dmax=100.0, step=4.0):
    B, H, W = dem.shape; dev = dem.device
    az = math.radians(wind_from_deg); drow = -math.cos(az); dcol = math.sin(az)
    ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                            torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    sx = torch.full_like(dem, -1e9)
    for kk in range(1, int(dmax / step) + 1):
        d = kk * step
        gy = (ys + drow * d) / (H - 1) * 2 - 1; gx = (xs + dcol * d) / (W - 1) * 2 - 1
        grid = torch.stack([gx, gy], -1).unsqueeze(0).expand(B, H, W, 2)
        zv = F.grid_sample(dem.unsqueeze(1), grid, align_corners=True, padding_mode="border").squeeze(1)
        sx = torch.maximum(sx, torch.rad2deg(torch.atan((zv - dem) / (d * RES))))
    return sx

@torch.no_grad()
def process_batch(dems, dist):                         # dems: list of [H,W] np -> [B,6] descriptors sampled at center
    x = torch.from_numpy(np.stack(dems)).to(DEV)
    z, twi, hand, lnsca, slope, dzdx, dzdy = physics(x, dist)
    tpi = z - F.avg_pool2d(F.pad(z.unsqueeze(1), (50,50,50,50), mode="replicate"), 101, 1).squeeze(1)
    sx_w = torch.stack([winstral_sx(x, 270 + 30 * (k - 1) / 2) for k in range(3)]).mean(0)
    sx_m = torch.stack([winstral_sx(x, a) for a in (0, 90, 180, 270)]).mean(0)
    return twi, hand, lnsca, sx_w, sx_m, tpi

def main():
    import threading
    from queue import Queue
    zc = np.load(os.path.join(HERE, "obs_coords.npz")); gid, lat, lon = zc["gbifID"], zc["lat"], zc["lon"]
    cells = {}
    for i in range(len(gid)):
        cells.setdefault((int(np.floor(lat[i] / CELL)), int(np.floor(lon[i] / CELL))), []).append((int(gid[i]), float(lat[i]), float(lon[i])))
    ck = pickle.load(open(CKPT, "rb")) if os.path.exists(CKPT) else {}
    done = ck.get("done", ck); scene = ck.get("scene", {})    # tolerate old {gid:vec} checkpoint
    assigned = build_index(list(cells))                       # cell -> [candidate tile urls] (reuses topo_tileidx.json)
    remaining = {c: list(assigned[c]) for c in cells
                 if c in assigned and not all(o[0] in done for o in cells[c])}
    total = len(remaining)
    print(f"{len(cells)} cells, {total} to fetch, GPU_BATCH={GPU_BATCH}", flush=True)
    dist = DIST.to(DEV); t0 = time.time(); n_cell = 0; rnd = 0
    while remaining and rnd < 6:                              # rounds: overlapping-project fallback for nodata gaps
        rnd += 1
        by_tile = defaultdict(list)
        for c, cands in remaining.items():
            by_tile[cands[0]].append(c)
        print(f"round {rnd}: {len(remaining)} cells across {len(by_tile)} tiles", flush=True)
        q = Queue(maxsize=8); failed_all = []
        def producer():
            with ThreadPoolExecutor(WORKERS) as ex:          # each thread opens ONE tile, reads all its cells
                for res in ex.map(lambda u: read_tile(u, by_tile[u]), list(by_tile)):
                    q.put(res)
            q.put(None)
        th = threading.Thread(target=producer, daemon=True); th.start()
        buf = []
        while True:
            item = q.get()
            if item is None:
                break
            patches, failed = item; failed_all += failed
            for p in patches:
                buf.append(p)
                if len(buf) >= GPU_BATCH:
                    _flush(buf, dist, done, scene, cells); n_cell += len(buf); buf = []
                    el = time.time() - t0
                    print(f"  r{rnd} {n_cell}/{total} cells | {len(done)} obs | {el:.0f}s | "
                          f"eta {el/max(n_cell,1)*(total-n_cell)/60:.1f}min", flush=True)
                    pickle.dump({"done": done, "scene": scene}, open(CKPT, "wb"))
        if buf:
            _flush(buf, dist, done, scene, cells); n_cell += len(buf); buf = []
        th.join()
        pickle.dump({"done": done, "scene": scene}, open(CKPT, "wb"))
        remaining = {c: remaining[c][1:] for c in failed_all if len(remaining.get(c, [])) > 1}
    hy = np.zeros((len(gid), 6), np.float32); have = np.zeros(len(gid), bool); scn = np.empty(len(gid), object)
    for i in range(len(gid)):
        g = int(gid[i])
        if g in done: hy[i] = done[g]; have[i] = True; scn[i] = scene.get(g, "")
    np.savez(os.path.join(HERE, "..", "gbif_hydro_tokens.npz"), gbifID=gid, hydro=hy, has_hydro=have,
             hydro_scene=scn.astype(object))
    print(f"DONE: {have.sum()}/{len(gid)} ({100*have.mean():.1f}%)", flush=True)

def _flush(buf, dist, done, scene, cells):
    dems = [b[3] for b in buf]
    twi, hand, lnsca, sxw, sxm, tpi = process_batch(dems, dist)
    twi, hand, lnsca, sxw, sxm, tpi = [t.cpu().numpy() for t in (twi, hand, lnsca, sxw, sxm, tpi)]
    for bi, (c, crs, out_tf, d, fname) in enumerate(buf):
        tf = transformer(crs); ia = ~out_tf                    # world -> (col,row)
        for gid_, la, lo in cells[c]:
            x, y = tf.transform(lo, la); col, row = ia * (x, y)
            col = int(np.clip(round(col), 0, N - 1)); row = int(np.clip(round(row), 0, N - 1))
            done[gid_] = np.array([twi[bi, row, col], hand[bi, row, col], lnsca[bi, row, col],
                                   sxw[bi, row, col], sxm[bi, row, col], tpi[bi, row, col]], np.float32)
            scene[gid_] = fname

if __name__ == "__main__":
    main()
