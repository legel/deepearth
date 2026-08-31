"""Build a tiny DINOv3 patch32 cache from synthetic chips and real row metadata."""
import argparse
import json
from pathlib import Path

import numpy as np

from dinov3_patch32 import DINOv3Patch32


def load_rows(root, n):
    files = sorted((root / "gbif_tokens").glob("*.npz"))
    if not files:
        raise SystemExit(f"missing train/test shards under {root / 'gbif_tokens'}")
    ids, lat, lon, obs_ord = [], [], [], []
    for file in files:
        z = np.load(file)
        take = min(n - len(ids), len(z["gbifID"]))
        if take <= 0:
            break
        ids.extend(z["gbifID"][:take].astype(np.int64))
        lat.extend(z["lat"][:take].astype(np.float32))
        lon.extend(z["lon"][:take].astype(np.float32))
        if "ord" in z:
            obs_ord.extend(z["ord"][:take].astype(np.int32))
        else:
            obs_ord.extend(np.full(take, -1, np.int32))
    return map(np.asarray, (ids, lat, lon, obs_ord))


def patch_offsets(ext_m=300.0):
    centers = (np.arange(32, dtype=np.float32) + 0.5) / 32.0 - 0.5
    off_y, off_x = np.meshgrid(centers * ext_m, centers * ext_m, indexing="ij")
    return np.stack([off_x, -off_y], -1).astype(np.float32)


def patch_latlon(lat, lon, patch_offset_m):
    north = patch_offset_m[..., 1]
    east = patch_offset_m[..., 0]
    dlat = north[None, :, :] / 111_320.0
    dlon = east[None, :, :] / (111_320.0 * np.cos(np.deg2rad(lat))[:, None, None] + 1e-6)
    return (lat[:, None, None] + dlat).astype(np.float32), (lon[:, None, None] + dlon).astype(np.float32)


def synthetic_chips(lat, lon):
    chips = np.empty((len(lat), 3, 512, 512), np.uint8)
    yy, xx = np.mgrid[:512, :512]
    for i, (la, lo) in enumerate(zip(lat, lon)):
        chips[i, 0] = ((xx + int(abs(lo) * 10)) % 256).astype(np.uint8)
        chips[i, 1] = ((yy + int(abs(la) * 10)) % 256).astype(np.uint8)
        chips[i, 2] = (((xx // 8) ^ (yy // 8) ^ i) % 256).astype(np.uint8)
    return chips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=".")
    ap.add_argument("--out-name", default="gbif_naip_dinov3_patch32_smoke")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    root = Path(args.cache).expanduser()
    out = root / args.out_name
    out.mkdir(parents=True, exist_ok=True)
    gid, lat, lon, obs_ord = load_rows(root, args.n)
    offset = patch_offsets()
    plat, plon = patch_latlon(lat.astype(np.float32), lon.astype(np.float32), offset)
    patch = DINOv3Patch32(batch=args.batch).patch32(synthetic_chips(lat, lon)).astype(np.float16)

    np.savez(out / "manifest.npz",
        gbifID=gid.astype(np.int64), lat=lat.astype(np.float32), lon=lon.astype(np.float32),
        elev_m=np.full(len(gid), np.nan, np.float32),
        event_day=np.full(len(gid), np.nan, np.float32),
        obs_ord=obs_ord.astype(np.int32),
        has_candidate_tile=np.ones(len(gid), bool),
        patch_shape=np.array([32, 32, 1024], np.int16),
        patch_offset_m=offset,
        dtype=np.array("float16"))
    with open(out / "metadata.json", "w") as f:
        json.dump({
            "model": "facebook/dinov3-vitl16-pretrain-sat493m",
            "source": "synthetic smoke chips, not scientific NAIP data",
            "patch_shape": [32, 32, 1024],
            "dtype": "float16",
            "patch_offset_m": "manifest.npz:patch_offset_m [32,32,2], east/north meters from observation center",
            "patch_latlon": "chunk*.npz:patch_lat/patch_lon [N,32,32], derived from obs center + patch_offset_m",
        }, f, indent=2)
    np.savez_compressed(out / "chunk0000.npz",
        gbifID=gid.astype(np.int64),
        naip_year=np.zeros(len(gid), np.int16),
        naip_scene=np.array(["synthetic"] * len(gid), object),
        patch=patch,
        patch_lat=plat,
        patch_lon=plon,
        has_naip=np.ones(len(gid), bool))
    print(f"wrote {out} rows={len(gid)} patch_shape={patch.shape} dtype={patch.dtype}")


if __name__ == "__main__":
    main()
