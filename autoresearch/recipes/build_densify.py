import numpy as np, glob
D = "/workspace/deepearth/data/deepcal"
chunks = sorted(glob.glob(f"{D}/gbif_tokens/chunk0*.npz"))
lat0 = np.concatenate([np.load(f)["lat"] for f in chunks])
lon0 = np.concatenate([np.load(f)["lon"] for f in chunks])
def cellhash(lat, lon): return np.floor(lat/0.5).astype(np.int64)*10007 + np.floor(lon/0.5).astype(np.int64)
cells = np.unique(cellhash(lat0, lon0))
rng = np.random.default_rng(0); rng.shuffle(cells)
ntest = max(1, int(len(cells)/6))
train_cells = cells[ntest:]                                   # existing TRAIN cells only -> split preserved exactly
print(f"original: {len(lat0)} obs, {len(cells)} cells ({ntest} test / {len(train_cells)} train)", flush=True)
b = np.load(f"{D}/gbif_densify_bulk.npz")
cb = cellhash(b["lat"], b["lon"])
keep = np.isin(cb, train_cells)                               # drop obs in test cells or brand-new cells
zc = np.load(chunks[0]); dd = zc["dino"].shape[1]; bd = zc["bio"].shape[1]
n = int(keep.sum())
print(f"densify: kept {n} of {len(cb)} bulk obs (in existing train cells) -> new total {len(lat0)+n}", flush=True)
np.savez_compressed(f"{D}/gbif_tokens/chunk_densify_bulk.npz",
    gbifID=b["gbifID"][keep].astype(np.int64), species_local=b["species_local"][keep].astype(np.int32),
    lat=b["lat"][keep].astype(np.float32), lon=b["lon"][keep].astype(np.float32),
    ord=np.zeros(n, np.float32),
    dino=np.zeros((n, dd), np.float32), bio=np.zeros((n, bd), np.float32))   # zeroed vision -> has_vision=False -> masked
print("wrote gbif_tokens/chunk_densify_bulk.npz", flush=True)
