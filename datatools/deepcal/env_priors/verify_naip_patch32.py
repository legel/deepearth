"""Verify the DINOv3 NAIP patch32 cache.

Checks the schema requested by Lance: per-observation DINOv3 patch embeddings
shaped (32, 32, 1024), keyed by gbifID, with an all-row manifest for train/test
alignment.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

PATCH_CHUNK_GLOB = "chunk[0-9]*.npz"


def _load_token_ids(path: Path) -> np.ndarray:
    files = sorted(path.glob("*.npz"))
    if not files:
        return np.array([], np.int64)
    return np.concatenate([_load_ids(file) for file in files])


def _load_ids(path: Path) -> np.ndarray:
    z = np.load(path)
    if "gbifID" not in z:
        raise SystemExit(f"{path} is missing gbifID")
    return z["gbifID"].astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=".")
    ap.add_argument("--patch-dir", default="gbif_naip_dinov3_patch32_v1")
    ap.add_argument("--allow-prefix", action="store_true")
    ap.add_argument("--require-complete", action="store_true")
    ap.add_argument("--estimate-only", action="store_true")
    args = ap.parse_args()

    root = Path(args.cache).expanduser()
    patch = root / args.patch_dir
    manifest_path = patch / "manifest.npz"
    metadata_path = patch / "metadata.json"
    if args.estimate_only and not manifest_path.exists():
        token_ids = _load_token_ids(root / "gbif_tokens")
        if not len(token_ids):
            raise SystemExit("estimate needs either patch manifest.npz or gbif_tokens/*.npz")
        values = len(token_ids) * 32 * 32 * 1024
        print(
            f"rows={len(token_ids):,} rgb_fp16={values*2/1e12:.2f}TB "
            f"rgb_fp32={values*4/1e12:.2f}TB"
        )
        return
    if not manifest_path.exists():
        raise SystemExit(f"missing {manifest_path}")
    if not metadata_path.exists():
        raise SystemExit(f"missing {metadata_path}")

    manifest = np.load(manifest_path)
    all_ids = manifest["gbifID"].astype(np.int64)
    if all_ids.ndim != 1 or len(all_ids) == 0:
        raise SystemExit("manifest gbifID must be a non-empty 1D array")
    if len(np.unique(all_ids)) != len(all_ids):
        raise SystemExit("manifest gbifID contains duplicates")
    if tuple(manifest["patch_shape"].tolist()) != (32, 32, 1024):
        raise SystemExit(f"manifest patch_shape is {manifest['patch_shape']}, expected (32,32,1024)")
    for key in ("lat", "lon", "elev_m", "event_day", "obs_ord", "has_candidate_tile"):
        if key not in manifest:
            raise SystemExit(f"manifest is missing {key}")
        if manifest[key].shape[0] != len(all_ids):
            raise SystemExit(f"manifest {key} length {manifest[key].shape[0]}, expected {len(all_ids)}")
    if "patch_offset_m" not in manifest:
        raise SystemExit("manifest is missing patch_offset_m")
    if manifest["patch_offset_m"].shape != (32, 32, 2):
        raise SystemExit(f"patch_offset_m shape is {manifest['patch_offset_m'].shape}, expected (32,32,2)")

    token_ids = _load_token_ids(root / "gbif_tokens")
    if len(token_ids):
        expected = token_ids
        expected_source = root / "gbif_tokens"
    else:
        expected_ids_path = root / "env_priors" / "obs_coords.npz"
        if not expected_ids_path.exists():
            expected_ids_path = root / "obs_coords.npz"
        expected = _load_ids(expected_ids_path) if expected_ids_path.exists() else None
        expected_source = expected_ids_path
    expected_ok = expected is None or np.array_equal(all_ids, expected)
    if not expected_ok and args.allow_prefix and expected is not None:
        expected_ok = len(all_ids) <= len(expected) and np.array_equal(all_ids, expected[:len(all_ids)])
    if not expected_ok:
        raise SystemExit(f"manifest gbifID does not match {expected_source}")

    with open(metadata_path) as f:
        metadata = json.load(f)
    if metadata.get("patch_shape") != [32, 32, 1024]:
        raise SystemExit("metadata patch_shape must be [32, 32, 1024]")
    if metadata.get("patch_offset_m") != "manifest.npz:patch_offset_m [32,32,2], east/north meters from observation center":
        raise SystemExit("metadata patch_offset_m contract is missing")
    if "patch_latlon" not in metadata:
        raise SystemExit("metadata patch_latlon contract is missing")
    if args.estimate_only:
        values = len(all_ids) * 32 * 32 * 1024
        print(
            f"rows={len(all_ids):,} rgb_fp16={values*2/1e12:.2f}TB "
            f"rgb_fp32={values*4/1e12:.2f}TB"
        )
        return

    files = sorted(glob.glob(str(patch / PATCH_CHUNK_GLOB)))
    if not files:
        raise SystemExit(f"no patch chunks under {patch}")

    seen, rows, bytes_total = set(), 0, 0
    manifest_set = set(map(int, all_ids))
    for file in files:
        z = np.load(file, allow_pickle=True)
        for key in ("gbifID", "naip_year", "naip_scene", "patch", "patch_lat", "patch_lon", "has_naip"):
            if key not in z:
                raise SystemExit(f"{file} missing {key}")
        gid = z["gbifID"].astype(np.int64)
        patch_tensor = z["patch"]
        if patch_tensor.shape != (len(gid), 32, 32, 1024):
            raise SystemExit(f"{file} patch shape {patch_tensor.shape}")
        if z["patch_lat"].shape != (len(gid), 32, 32):
            raise SystemExit(f"{file} patch_lat shape {z['patch_lat'].shape}")
        if z["patch_lon"].shape != (len(gid), 32, 32):
            raise SystemExit(f"{file} patch_lon shape {z['patch_lon'].shape}")
        if not (np.isfinite(z["patch_lat"]).all() and np.isfinite(z["patch_lon"]).all()):
            raise SystemExit(f"{file} patch coordinates must be finite")
        if patch_tensor.dtype not in (np.float16, np.float32):
            raise SystemExit(f"{file} patch dtype must be float16 or float32")
        if not np.all(z["has_naip"].astype(bool)):
            raise SystemExit(f"{file} has_naip must be true for every stored row")
        for g in gid:
            gi = int(g)
            if gi not in manifest_set:
                raise SystemExit(f"{file} gbifID {gi} is absent from manifest")
            if gi in seen:
                raise SystemExit(f"duplicate chunk gbifID {gi}")
            seen.add(gi)
        rows += len(gid)
        bytes_total += patch_tensor.nbytes

    missing = len(all_ids) - len(seen)
    if args.require_complete and missing:
        raise SystemExit(f"patch cache incomplete: {missing}/{len(all_ids)} manifest rows missing")
    print(
        f"OK patch32 rows={rows:,}/{len(all_ids):,} missing={missing:,} "
        f"dtype={metadata.get('dtype')} payload={bytes_total/1e9:.2f}GB"
    )


if __name__ == "__main__":
    main()
