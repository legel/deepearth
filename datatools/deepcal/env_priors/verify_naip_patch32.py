"""Verify the DINOv3 NAIP patch32 cache.

Checks the schema requested by Lance: per-observation DINOv3 patch embeddings
shaped (32, 32, 1024), keyed by gbifID, with an all-row manifest for train/test
alignment.
"""
import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

PATCH_CHUNK_GLOB = "chunk[0-9]*.npz"


def _finite_scalar(value, fallback=0.0) -> np.float32:
    value = np.float32(value)
    if np.isfinite(value):
        return value
    return np.float32(fallback)


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
    ap.add_argument("--fallback-dir", action="append", default=[])
    ap.add_argument("--allow-prefix", action="store_true")
    ap.add_argument("--require-complete", action="store_true")
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--latest", action="store_true")
    ap.add_argument("--coverage-only", action="store_true")
    ap.add_argument("--split-summary", action="store_true")
    ap.add_argument("--holdout-fraction", type=float, default=1 / 6)
    ap.add_argument("--write-missing")
    ap.add_argument("--write-no-candidate")
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
    if args.write_missing and args.max_chunks:
        raise SystemExit("--write-missing requires a full scan; do not combine with --max-chunks")

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
    if args.write_no_candidate:
        has_candidate = manifest["has_candidate_tile"].astype(bool)
        with open(args.write_no_candidate, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["gbifID", "lat", "lon", "elev_m", "event_day", "obs_ord"])
            for i in np.flatnonzero(~has_candidate):
                writer.writerow([
                    int(all_ids[i]),
                    float(manifest["lat"][i]),
                    float(manifest["lon"][i]),
                    float(manifest["elev_m"][i]),
                    float(manifest["event_day"][i]),
                    int(manifest["obs_ord"][i]),
                ])
        print(
            f"OK no-candidate rows={int((~has_candidate).sum()):,}/{len(all_ids):,} "
            f"wrote={args.write_no_candidate}"
        )
        return

    files = [(Path(file), False) for file in sorted(glob.glob(str(patch / PATCH_CHUNK_GLOB)))]
    for fallback in args.fallback_dir:
        fpath = Path(fallback)
        if not fpath.is_absolute():
            fpath = root / fpath
        files.extend((Path(file), True) for file in sorted(glob.glob(str(fpath / PATCH_CHUNK_GLOB))))
    if not files:
        raise SystemExit(f"no patch chunks under {patch}")
    total_files = len(files)
    if args.max_chunks:
        files = files[-args.max_chunks:] if args.latest else files[:args.max_chunks]

    seen, rows, bytes_total = set(), 0, 0
    manifest_set = set(map(int, all_ids))
    manifest_row = {int(g): i for i, g in enumerate(all_ids)}
    for file, is_fallback in files:
        z = np.load(file, allow_pickle=True)
        if args.coverage_only:
            if "gbifID" not in z:
                raise SystemExit(f"{file} missing gbifID")
            gid = z["gbifID"].astype(np.int64)
            added = 0
            for g in gid:
                gi = int(g)
                if gi not in manifest_set:
                    raise SystemExit(f"{file} gbifID {gi} is absent from manifest")
                if gi in seen:
                    if is_fallback:
                        continue
                    raise SystemExit(f"duplicate chunk gbifID {gi}")
                seen.add(gi)
                added += 1
            rows += added
            continue
        for key in ("gbifID", "naip_year", "naip_scene", "patch", "patch_lat", "patch_lon", "has_naip"):
            if key not in z:
                raise SystemExit(f"{file} missing {key}")
        gid = z["gbifID"].astype(np.int64)
        if is_fallback and all(int(g) in seen for g in gid):
            continue
        patch_tensor = z["patch"]
        if patch_tensor.shape != (len(gid), 32, 32, 1024):
            raise SystemExit(f"{file} patch shape {patch_tensor.shape}")
        if z["patch_lat"].shape != (len(gid), 32, 32):
            raise SystemExit(f"{file} patch_lat shape {z['patch_lat'].shape}")
        if z["patch_lon"].shape != (len(gid), 32, 32):
            raise SystemExit(f"{file} patch_lon shape {z['patch_lon'].shape}")
        if "patch_elev" in z and z["patch_elev"].shape != (len(gid), 32, 32):
            raise SystemExit(f"{file} patch_elev shape {z['patch_elev'].shape}")
        if not np.isfinite(patch_tensor).all():
            raise SystemExit(f"{file} patch tensor must be finite")
        if not (np.isfinite(z["patch_lat"]).all() and np.isfinite(z["patch_lon"]).all()):
            raise SystemExit(f"{file} patch coordinates must be finite")
        if patch_tensor.dtype not in (np.float16, np.float32):
            raise SystemExit(f"{file} patch dtype must be float16 or float32")
        if not np.all(z["has_naip"].astype(bool)):
            raise SystemExit(f"{file} has_naip must be true for every stored row")
        for row, g in enumerate(gid):
            gi = int(g)
            if gi not in manifest_set:
                raise SystemExit(f"{file} gbifID {gi} is absent from manifest")
            if gi in seen:
                if is_fallback:
                    continue
                raise SystemExit(f"duplicate chunk gbifID {gi}")
            seen.add(gi)
            rows += 1
            m = manifest_row[gi]
            if "patch_elev" in z:
                elev = z["patch_elev"][row].astype(np.float32)
                elev = np.where(np.isfinite(elev), elev, _finite_scalar(manifest["elev_m"][m]))
            else:
                elev = np.full((32, 32), _finite_scalar(manifest["elev_m"][m]), np.float32)
            day = np.full((32, 32), _finite_scalar(manifest["event_day"][m]), np.float32)
            coords = np.stack([
                z["patch_lat"][row].astype(np.float32),
                z["patch_lon"][row].astype(np.float32),
                elev,
                day,
            ], axis=-1)
            if not np.isfinite(coords).all():
                raise SystemExit(f"{file} Earth4D coords must be finite")
        bytes_total += patch_tensor.nbytes

    missing = len(all_ids) - len(seen)
    if args.split_summary:
        cell = (
            np.floor(manifest["lat"].astype(np.float32) / 0.5).astype(np.int64) * 10007
            + np.floor(manifest["lon"].astype(np.float32) / 0.5).astype(np.int64)
        )
        cells = np.unique(cell)
        np.random.default_rng(0).shuffle(cells)
        test_cells = cells[: max(1, int(len(cells) * args.holdout_fraction))]
        test_mask = np.isin(cell, test_cells)
        covered = np.array([int(g) in seen for g in all_ids], bool)
        train_mask = ~test_mask
        print(
            "SPLIT COVERAGE "
            f"train={int((covered & train_mask).sum()):,}/{int(train_mask.sum()):,} "
            f"test={int((covered & test_mask).sum()):,}/{int(test_mask.sum()):,}"
        )
    if args.write_missing:
        with open(args.write_missing, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["gbifID", "lat", "lon", "has_candidate_tile"])
            for i, gid in enumerate(all_ids):
                if int(gid) not in seen:
                    writer.writerow([
                        int(gid),
                        float(manifest["lat"][i]),
                        float(manifest["lon"][i]),
                        bool(manifest["has_candidate_tile"][i]),
                    ])
    if args.require_complete and missing:
        raise SystemExit(f"patch cache incomplete: {missing}/{len(all_ids)} manifest rows missing")
    scanned = f" chunks={len(files):,}/{total_files:,}" if args.max_chunks else ""
    print(
        f"OK patch32 rows={rows:,}/{len(all_ids):,} missing={missing:,}{scanned} "
        f"dtype={metadata.get('dtype')} payload={bytes_total/1e9:.2f}GB"
    )


if __name__ == "__main__":
    main()
