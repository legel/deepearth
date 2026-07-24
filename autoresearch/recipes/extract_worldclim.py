#!/usr/bin/env python3
"""WorldClim v2.1 bioclim channel: sample the 19 standard bioclimatic normals (1970-2000, 10-arcmin
climatology) at each observation's coordinates. Transferable climate-niche signal for species-from-env.

One-time raster download (public, ~50MB):
    curl -L https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_bio.zip -o /tmp/wc.zip
    mkdir -p $WC_DIR && unzip -o /tmp/wc.zip -d $WC_DIR      # WC_DIR default /tmp/worldclim
Then run this to write gbif_worldclim_tokens.npz. Output holds RAW bio values [N,19] (the data.py loader
z-scores on load) plus a has_worldclim presence mask; missing cells are median-imputed so absence is not a
location cue (their has_worldclim stays False and the loader masks them)."""
import numpy as np, glob, os, rasterio
from rasterio.transform import rowcol

CACHE = os.environ.get("CACHE", "/workspace/deepearth/data/deepcal")
WC_DIR = os.environ.get("WC_DIR", "/tmp/worldclim")
OUT = os.environ.get("OUT", f"{CACHE}/gbif_worldclim_tokens.npz")

gid, lat, lon = [], [], []
for f in sorted(glob.glob(f"{CACHE}/gbif_tokens/chunk*.npz")):
    d = np.load(f); gid.append(d["gbifID"]); lat.append(d["lat"]); lon.append(d["lon"])
gid = np.concatenate(gid); lat = np.concatenate(lat).astype(np.float64); lon = np.concatenate(lon).astype(np.float64)
N = len(gid); print(f"{N} observations", flush=True)

bio = np.full((N, 19), np.nan, np.float32)
for b in range(1, 20):
    with rasterio.open(f"{WC_DIR}/wc2.1_10m_bio_{b}.tif") as ds:
        rows, cols = rowcol(ds.transform, lon, lat)          # array-friendly (x=lon, y=lat) -> (row, col)
        rows = np.asarray(rows); cols = np.asarray(cols)
        ok = (rows >= 0) & (rows < ds.height) & (cols >= 0) & (cols < ds.width)
        arr = ds.read(1); vals = np.full(N, np.nan, np.float32)
        vals[ok] = arr[rows[ok], cols[ok]]
        if ds.nodata is not None: vals[vals == ds.nodata] = np.nan
        bio[:, b - 1] = vals
    print(f"  bio_{b}: coverage {np.isfinite(bio[:, b-1]).mean():.3f}", flush=True)

has = np.isfinite(bio).all(1)
med = np.nanmedian(bio, axis=0)
miss = ~np.isfinite(bio); bio[miss] = np.take(med, np.where(miss)[1])
print(f"has_worldclim coverage {has.mean():.3f}", flush=True)
np.savez_compressed(OUT, gbifID=gid.astype(np.int64), worldclim=bio.astype(np.float32), has_worldclim=has)
print(f"wrote {OUT}", flush=True)
