"""GPU hydrology (TWI/HAND/catchment) + Winstral Sx wind exposure from 3DEP 1m DEM, batched in PyTorch.
Flow accumulation = iterative upstream propagation (D8 -> a forest of in-trees, so exact, no double-count),
converging in O(flow-path) cheap GPU steps over a whole batch of patches. Fetch (threaded, hardened) feeds GPU
batches, decoupling I/O from compute. Output: gbif_hydro_tokens.npz {gbifID, hydro[N,6], has_hydro}. Resumable."""
import os, time, pickle, warnings, math, urllib.request
import numpy as np, torch, torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from rasterio.io import MemoryFile
from pyproj import Transformer
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); DEV = "cuda"
BASE = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
CELL = 0.002; HALF = 256; N = 256; RES = 2.0; WORKERS = 32; GPU_BATCH = 96
CKPT = os.path.join(HERE, "hydrowind_ckpt.pkl")
DIRS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
OPP = [7,6,5,4,3,2,1,0]
DIST = torch.tensor([2**.5,1,2**.5,1,1,2**.5,1,2**.5])
_tf = {}
def utm_epsg(lon): return 26910 if lon < -120.0 else 26911
def tform(e):
    if e not in _tf: _tf[e] = Transformer.from_crs(4326, e, always_xy=True)
    return _tf[e]

def fetch(cx, cy, epsg):
    url = (f"{BASE}?bbox={cx-HALF},{cy-HALF},{cx+HALF},{cy+HALF}&bboxSR={epsg}&size={N},{N}&imageSR={epsg}"
           f"&format=tiff&pixelType=F32&noData=-9999&adjustAspectRatio=false&interpolation=RSP_BilinearInterpolation&f=image")
    for k in range(2):
        try:
            with urllib.request.urlopen(url, timeout=20) as r: data = r.read()
            with MemoryFile(data) as mf, mf.open() as ds: return ds.read(1).astype(np.float32)
        except Exception:
            if k: return None
            time.sleep(1.0)
    return None

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
    zc = np.load(os.path.join(HERE, "obs_coords.npz")); gid, lat, lon = zc["gbifID"], zc["lat"], zc["lon"]
    cells = {}
    for i in range(len(gid)):
        cells.setdefault((int(np.floor(lat[i] / CELL)), int(np.floor(lon[i] / CELL))), []).append((int(gid[i]), float(lat[i]), float(lon[i])))
    done = pickle.load(open(CKPT, "rb")) if os.path.exists(CKPT) else {}
    todo = [c for c in cells if not all(o[0] in done for o in cells[c])]
    print(f"{len(cells)} cells, {len(todo)} to fetch, GPU_BATCH={GPU_BATCH}", flush=True)
    dist = DIST.to(DEV); t0 = time.time(); n_cell = 0
    def fetch_cell(c):
        clat, clon = c; latc, lonc = (clat + .5) * CELL, (clon + .5) * CELL; e = utm_epsg(lonc)
        cx, cy = tform(e).transform(lonc, latc)
        if not (np.isfinite(cx) and np.isfinite(cy)): return None
        d = fetch(cx, cy, e)
        if d is None or d.shape != (N, N): return None
        d = np.where(np.isfinite(d) & (d > -1e4), d, np.nan)
        if np.isnan(d).mean() > 0.5: return None
        return (c, e, cx, cy, np.nan_to_num(d, nan=np.nanmean(d)).astype(np.float32))
    with ThreadPoolExecutor(WORKERS) as ex:
        buf = []
        for res in ex.map(fetch_cell, todo):
            if res is None: continue
            buf.append(res)
            if len(buf) >= GPU_BATCH:
                _flush(buf, dist, done, cells); n_cell += len(buf); buf = []
                el = time.time() - t0
                print(f"  {n_cell}/{len(todo)} cells | {len(done)} obs | {el:.0f}s | eta {el/max(n_cell,1)*(len(todo)-n_cell)/60:.1f}min", flush=True)
                pickle.dump(done, open(CKPT, "wb"))
        if buf: _flush(buf, dist, done, cells); n_cell += len(buf)
    pickle.dump(done, open(CKPT, "wb"))
    hy = np.zeros((len(gid), 6), np.float32); have = np.zeros(len(gid), bool)
    for i in range(len(gid)):
        g = int(gid[i])
        if g in done: hy[i] = done[g]; have[i] = True
    np.savez(os.path.join(HERE, "..", "gbif_hydro_tokens.npz"), gbifID=gid, hydro=hy, has_hydro=have)
    print(f"DONE: {have.sum()}/{len(gid)} ({100*have.mean():.1f}%)", flush=True)

def _flush(buf, dist, done, cells):
    dems = [b[4] for b in buf]
    twi, hand, lnsca, sxw, sxm, tpi = process_batch(dems, dist)
    twi, hand, lnsca, sxw, sxm, tpi = [t.cpu().numpy() for t in (twi, hand, lnsca, sxw, sxm, tpi)]
    for bi, (c, e, cx, cy, d) in enumerate(buf):
        tf = tform(e)
        for gid_, la, lo in cells[c]:
            x, y = tf.transform(lo, la); col = int(np.clip(round((x - (cx - HALF)) / RES), 0, N - 1)); row = int(np.clip(round(((cy + HALF) - y) / RES), 0, N - 1))
            done[gid_] = np.array([twi[bi, row, col], hand[bi, row, col], lnsca[bi, row, col], sxw[bi, row, col], sxm[bi, row, col], tpi[bi, row, col]], np.float32)

if __name__ == "__main__":
    main()
