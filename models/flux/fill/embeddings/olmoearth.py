"""OlmoEarth v1.3 annual embeddings around every flux tower, read from the public store (Ai2, source.coop/olmoearth,
olmoearth-v1-3-embeddings; OlmoEarth Artifact License, cite Ai2). Each tower's file holds, per year with data:
- `center` (Y, 128) float32: the mean over 300 m around the tower;
- `grid` (Y, 32, 32, 64) int8: the first 64 dimensions, mean-pooled to 40 m over 1.28 km, re-quantized as the store is
  (signed square root, -128 is nodata), for footprint sums.

Pretraining used imagery and maps (Sentinel-1, Sentinel-2, Landsat, OpenStreetMap, WorldCover, CDL, SRTM, canopy height,
WorldCereal); no flux data. An embedding is computed from all Sentinel-1, Sentinel-2 and Landsat of one calendar year, the
same Sentinel-2 FluxFill reconstructs as Clay embeddings, and Landsat thermal and radar follow reconstructed tower and
reanalysis rows. So only embeddings of other calendar years may enter a window, the rule for a tower's own context.

  python olmoearth.py SITES.json OUT_DIR [--limit N]
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zarr
from obstore.store import S3Store
from pyproj import Transformer

STORE = "s3://olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr"
GRID, POOL, CENTER = 64, 4, 15           # half-widths and pool in 10 m pixels: 1.28 km to 40 m; 300 m


def dequantize(q):
    v = np.where(q == -128, np.nan, q).astype("float32")
    return (np.abs(v) / 127.5) ** 2.0 * np.sign(v)


def requantize(v):
    q = np.sign(v) * np.sqrt(np.abs(v)) * 127.5
    return np.where(np.isfinite(q), np.clip(np.round(q), -127, 127), -128).astype(np.int8)


def tower(root, site, out):
    path = out / f"{site['id']}.npz"
    if path.exists():
        return site["id"], "done"
    lon, lat = site["lon"], site["lat"]
    zone = root[f"utm{int((lon + 180) // 6) + 1:02d}"]
    x, y = Transformer.from_crs("EPSG:4326", zone.attrs["proj:code"], always_xy=True).transform(lon, lat)
    a, _, c, _, e, f = zone.attrs["spatial:transform"]
    col, row = int((x - c) / a), int((y - f) / e)
    years, center, grid = [], [], []
    arr = zone["embeddings"]
    for t, year in enumerate(zone["time"][:]):
        mid = dequantize(arr[t, :, row - CENTER:row + CENTER, col - CENTER:col + CENTER])
        if not np.isfinite(mid).any():
            continue
        g = dequantize(arr[t, 0:64, row - GRID:row + GRID, col - GRID:col + GRID])
        n = 2 * GRID // POOL
        with np.errstate(invalid="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pooled = np.nanmean(g.reshape(64, n, POOL, n, POOL), (2, 4))
                years.append(int(year)); center.append(np.nanmean(mid, (1, 2)))
        grid.append(requantize(pooled).transpose(1, 2, 0))
    if not years:
        return site["id"], "no data"
    np.savez_compressed(path, years=np.array(years), center=np.stack(center).astype(np.float32), grid=np.stack(grid), row=row, col=col,
                        zone=zone.path)
    return site["id"], f"{len(years)} years"


def main(sites_json, out, limit=0, workers=4):
    zarr.config.set({"array.sharding_coalesce_max_gap_bytes": 0})
    s3 = S3Store.from_url(STORE, endpoint="https://data.source.coop", skip_signature=True)
    root = zarr.open_group(zarr.storage.ObjectStore(s3, read_only=True), mode="r")
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    sites = [s for s in json.load(open(sites_json))["sites"] if s.get("split") and s.get("lat") is not None]
    sites = sites[:limit] if limit else sites

    def one(s):
        try:
            return tower(root, s, out)
        except Exception as err:                       # one tower's failure is reported and retried on the next run
            return s["id"], f"failed: {err!r}"[:200]

    with ThreadPoolExecutor(workers) as ex:
        for k, (name, msg) in enumerate(ex.map(one, sites)):
            print(k, name, msg, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[4]) if len(sys.argv) > 4 else 0)
