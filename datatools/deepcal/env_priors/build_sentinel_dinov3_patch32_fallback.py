"""Fill no-NAIP rows with Sentinel-2 RGB DINOv3 patch32 embeddings.

This builder is intentionally narrow: it reads the NAIP patch32 manifest,
selects rows where public NAIP/STAC has no candidate, and writes compatible
`chunk*.npz` files under a separate fallback directory. The primary NAIP cache
remains untouched while long extraction is running.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests

from dinov3_patch32 import DINOv3Patch32


PATCH_SHAPE = (32, 32, 1024)
SIZE = 512
EXT = 320.0
GSD = EXT / SIZE
YEAR = int(os.environ.get("SENTINEL_PATCH_YEAR", "2025"))
CLOUD_LT = float(os.environ.get("SENTINEL_PATCH_CLOUD_LT", "20"))
WORKERS = int(os.environ.get("SENTINEL_PATCH_WORKERS", "8"))
EMBED_BATCH = int(os.environ.get("SENTINEL_EMBED_BATCH", os.environ.get("NAIP_EMBED_BATCH", "4")))
PATCH_ROWS = int(os.environ.get("SENTINEL_PATCH_ROWS", "16"))
STAC_API = os.environ.get("STAC_API", "https://earth-search.aws.element84.com/v1")
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "VSI_CACHE": "TRUE",
    "AWS_NO_SIGN_REQUEST": "YES",
}


def patch_latlon(lat, lon):
    centers = (np.arange(32, dtype=np.float32) + 0.5) / 32.0 - 0.5
    off_y, off_x = np.meshgrid(centers * EXT, centers * EXT, indexing="ij")
    north = -off_y
    east = off_x
    lat = np.asarray(lat, np.float32)
    lon = np.asarray(lon, np.float32)
    dlat = north[None] / 111_320.0
    dlon = east[None] / (111_320.0 * np.cos(np.deg2rad(lat))[:, None, None] + 1e-6)
    return (
        (lat[:, None, None] + dlat).astype(np.float32),
        (lon[:, None, None] + dlon).astype(np.float32),
    )


def patch_offsets():
    centers = (np.arange(32, dtype=np.float32) + 0.5) / 32.0 - 0.5
    off_y, off_x = np.meshgrid(centers * EXT, centers * EXT, indexing="ij")
    return np.stack([off_x, -off_y], -1).astype(np.float32)


def utm_epsg(lon, lat):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) / 6) + 1


def best_item(lon, lat):
    body = {
        "collections": ["sentinel-2-l2a"],
        "intersects": {"type": "Point", "coordinates": [float(lon), float(lat)]},
        "datetime": f"{YEAR}-01-01T00:00:00Z/{YEAR}-12-31T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": CLOUD_LT}},
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        "limit": 1,
    }
    response = requests.post(f"{STAC_API.rstrip('/')}/search", json=body, timeout=60)
    response.raise_for_status()
    features = response.json().get("features", [])
    return features[0] if features else None


def fetch_rgb(row):
    import rasterio
    from affine import Affine
    from pyproj import Transformer
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    item = best_item(row["lon"], row["lat"])
    if item is None:
        return None, "no_item"
    epsg = item.get("properties", {}).get("proj:epsg") or utm_epsg(row["lon"], row["lat"])
    cx, cy = Transformer.from_crs(4326, epsg, always_xy=True).transform(row["lon"], row["lat"])
    half = EXT / 2.0
    transform = Affine(GSD, 0, cx - half, 0, -GSD, cy + half)
    dst_crs = CRS.from_epsg(epsg)
    assets = [
        item.get("assets", {}).get(k) or item.get("assets", {}).get(v)
        for k, v in (("red", "B04"), ("green", "B03"), ("blue", "B02"))
    ]
    if any(asset is None for asset in assets):
        return None, "missing_rgb_asset"
    bands = []
    with rasterio.Env(**GDAL_ENV):
        for asset in assets:
            with rasterio.open(asset["href"]) as src:
                with WarpedVRT(
                    src,
                    crs=dst_crs,
                    transform=transform,
                    width=SIZE,
                    height=SIZE,
                    resampling=Resampling.bilinear,
                ) as vrt:
                    bands.append(vrt.read(1).astype(np.float32))
    rgb = np.stack(bands)
    if not np.isfinite(rgb).any() or (rgb <= 0).all():
        return None, "empty_pixels"
    hi = np.nanpercentile(rgb, 99)
    lo = np.nanpercentile(rgb, 1)
    rgb = np.clip((rgb - lo) / max(hi - lo, 1.0), 0.0, 1.0)
    return ((rgb * 255).astype(np.uint8), item.get("id", "")), None


def load_rows(manifest):
    mask = ~manifest["has_candidate_tile"].astype(bool)
    rows = []
    for i in np.flatnonzero(mask):
        rows.append({
            "gbifID": int(manifest["gbifID"][i]),
            "lat": float(manifest["lat"][i]),
            "lon": float(manifest["lon"][i]),
            "elev_m": float(manifest["elev_m"][i]),
            "event_day": float(manifest["event_day"][i]),
            "obs_ord": int(manifest["obs_ord"][i]),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=os.environ.get("DEEPCAL_CACHE", "."))
    parser.add_argument("--patch-dir", default="gbif_naip_dinov3_patch32_v1")
    parser.add_argument("--out-dir", default="gbif_sentinel2_dinov3_patch32_fallback_v1")
    parser.add_argument("--max-rows", type=int, default=int(os.environ.get("SENTINEL_PATCH_MAX_ROWS", "0")))
    args = parser.parse_args()

    root = Path(args.cache).expanduser()
    primary = root / args.patch_dir
    out = root / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = np.load(primary / "manifest.npz")
    rows_all = load_rows(manifest)
    rows = rows_all
    if args.max_rows:
        rows = rows[:args.max_rows]
    done_path = out / "sentinel_patch32_ckpt.pkl"
    done = pickle.load(open(done_path, "rb")) if done_path.exists() else {}
    rows = [row for row in rows if row["gbifID"] not in done]

    np.savez(
        out / "manifest.npz",
        gbifID=np.array([row["gbifID"] for row in rows_all], np.int64),
        lat=np.array([row["lat"] for row in rows_all], np.float32),
        lon=np.array([row["lon"] for row in rows_all], np.float32),
        elev_m=np.array([row["elev_m"] for row in rows_all], np.float32),
        event_day=np.array([row["event_day"] for row in rows_all], np.float32),
        obs_ord=np.array([row["obs_ord"] for row in rows_all], np.int32),
        has_candidate_tile=np.ones(len(rows_all), bool),
        patch_shape=np.array(PATCH_SHAPE, np.int16),
        patch_offset_m=patch_offsets(),
    )
    with open(out / "metadata.json", "w") as f:
        json.dump({
            "model": "facebook/dinov3-vitl16-pretrain-sat493m",
            "source": "Sentinel-2 L2A RGB fallback for rows without public NAIP/STAC candidate",
            "source_year": YEAR,
            "cloud_lt": CLOUD_LT,
            "patch_extent_m": EXT,
            "source_patch": [3, SIZE, SIZE],
            "patch_shape": list(PATCH_SHAPE),
            "dtype": "float16",
            "row_key": "gbifID",
            "primary_patch_dir": args.patch_dir,
        }, f, indent=2)

    print(f"{len(rows)} Sentinel fallback rows remaining | out={out}", flush=True)
    if not rows:
        return

    emb = DINOv3Patch32(batch=EMBED_BATCH)
    chunk_id = len(list(out.glob("chunk[0-9]*.npz")))
    t0 = time.time()
    ok = fail = 0
    fail_reasons = {}

    for start in range(0, len(rows), PATCH_ROWS):
        batch = rows[start:start + PATCH_ROWS]
        fetched = []
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {executor.submit(fetch_rgb, row): row for row in batch}
            for future in as_completed(futures):
                row = futures[future]
                try:
                    result, reason = future.result()
                except Exception as exc:
                    result, reason = None, type(exc).__name__
                if result is None:
                    fail += 1
                    fail_reasons[reason or "unknown"] = fail_reasons.get(reason or "unknown", 0) + 1
                    continue
                image, scene = result
                fetched.append((row, image, scene))
        if not fetched:
            continue
        patches = emb.patch32([image for _, image, _ in fetched]).astype(np.float16)
        lat_grid, lon_grid = patch_latlon(
            [row["lat"] for row, _, _ in fetched],
            [row["lon"] for row, _, _ in fetched],
        )
        np.savez(
            out / f"chunk{chunk_id:04d}.npz",
            gbifID=np.array([row["gbifID"] for row, _, _ in fetched], np.int64),
            naip_year=np.full(len(fetched), YEAR, np.int16),
            naip_scene=np.array([f"sentinel2:{scene}" for _, _, scene in fetched], object),
            source_sensor=np.array(["sentinel-2-l2a"] * len(fetched), object),
            patch=patches,
            patch_lat=lat_grid,
            patch_lon=lon_grid,
            has_naip=np.ones(len(fetched), bool),
        )
        for row, _, scene in fetched:
            done[row["gbifID"]] = scene
        pickle.dump(done, open(done_path, "wb"))
        chunk_id += 1
        ok += len(fetched)
        elapsed = max(time.time() - t0, 1.0)
        print(
            f"  sentinel chunk {chunk_id} | ok {ok} fail {fail} | "
            f"{ok / elapsed:.2f} obs/s",
            flush=True,
        )
    print(f"DONE sentinel fallback ok {ok} fail {fail} reasons {fail_reasons}", flush=True)


if __name__ == "__main__":
    main()
