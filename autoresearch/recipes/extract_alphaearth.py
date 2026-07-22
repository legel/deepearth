#!/usr/bin/env python3
"""Extract AlphaEarth (Google Satellite Embedding V1) 64-dim annual embeddings @10m for every GBIF observation.

AlphaEarth is REQUIRED to reproduce the DeepCal champion (arith 0.6074): the model consumes it as a
SatCLIP-style learned geo prior (``model.alphaearth_geo: true`` in ``autoresearch/champion.yaml``), added to
the spatial position that every head reads. See ``autoresearch/recipes/README.md``.

Prerequisites
-------------
- An Earth Engine account registered for the project you pass as EE_PROJECT (``earthengine authenticate``).
- ``pip install earthengine-api numpy``
- A coords file (npz) with per-observation ``gbifID``, ``lat``, ``lon`` (extract these from your prepared cache).

Output
------
``gbif_alphaearth_tokens.npz`` with ``{gbifID: [N], ae: [N,64] float32}`` (NaN rows = no coverage).
Drop it into your DeepCal cache dir; ``autoresearch/data.py`` loads it automatically as the ``alphaearth`` modality
(aligned by gbifID, z-scored on the train split).

Performance note
----------------
Earth Engine ``getInfo`` scales SUPERLINEARLY with FeatureCollection size (server-side list.map blows up:
~36 ms/pt at 50 pts vs ~160 ms/pt at 500 pts). Use SMALL batches (200) with high client concurrency (16 workers)
on the high-volume endpoint. ~25-45 min for 620k points. Resumable via the ``.partial.npy`` sidecar.
"""
import numpy as np, ee, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT = os.environ.get("EE_PROJECT", "your-ee-project")
YEAR = int(os.environ.get("AE_YEAR", "2024"))
COORDS = os.environ.get("AE_COORDS", "ae_coords.npz")            # {gbifID, lat, lon} from your prepared cache
OUT = os.environ.get("AE_OUT", "gbif_alphaearth_tokens.npz")
BATCH = 200
WORKERS = 16

ee.Initialize(project=PROJECT, opt_url="https://earthengine-highvolume.googleapis.com")
z = np.load(COORDS, allow_pickle=True)
gid, lat, lon = z["gbifID"], z["lat"].astype(float), z["lon"].astype(float)
N = len(lat); print(f"{N} obs, year {YEAR}, {WORKERS} workers, batch {BATCH}", flush=True)
img = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").filterDate(f"{YEAR}-01-01", f"{YEAR}-12-31").mosaic()
BANDS = [f"A{i:02d}" for i in range(64)]
img = img.select(BANDS)

emb = np.full((N, 64), np.nan, dtype=np.float32)
part = OUT + ".partial.npy"
if os.path.exists(part):
    prev = np.load(part)
    if len(prev) == N: emb[:] = prev; print(f"resume: {int(np.isfinite(emb[:,0]).sum())} already sampled", flush=True)

def do_batch(b):
    j = min(b + BATCH, N)
    if np.isfinite(emb[b:j, 0]).all():
        return b, None                                           # already done (resume)
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(lon[k]), float(lat[k])]), {"i": k - b})
                               for k in range(b, j)])
    samp = img.sampleRegions(collection=fc, scale=10, geometries=False)
    for _ in range(5):
        try:
            return b, samp.getInfo()["features"]
        except Exception:
            time.sleep(5)
    return b, "FAIL"

batches = list(range(0, N, BATCH))
t0 = time.time(); done = 0; fails = 0
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futs = [ex.submit(do_batch, b) for b in batches]
    for fut in as_completed(futs):
        b, feats = fut.result(); done += 1
        if feats and feats != "FAIL":
            for f in feats:
                p = f["properties"]; emb[b + int(p["i"])] = [p.get(x, np.nan) for x in BANDS]
        elif feats == "FAIL":
            fails += 1
        if done % 100 == 0:
            np.save(part, emb); eta = (time.time() - t0) / done * (len(batches) - done)
            print(f"  {done}/{len(batches)} ({100*done//len(batches)}%)  "
                  f"{int(np.isfinite(emb[:,0]).sum())} sampled  {fails} fails  eta {eta/60:.1f}min", flush=True)

np.save(part, emb)
np.savez(OUT, gbifID=gid, ae=emb)
print(f"WROTE {OUT}: {int(np.isfinite(emb[:,0]).sum())}/{N} sampled, {fails} failed batches", flush=True)
