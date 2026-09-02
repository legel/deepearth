"""Pack DINOv3 patch32 chunks into row-ordered mmap arrays."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap


def chunk_files(root, patch_dir, fallback_dirs):
    primary = root / patch_dir
    for file in sorted(primary.glob("chunk[0-9]*.npz")):
        yield file, False
    for directory in fallback_dirs:
        path = Path(directory)
        if not path.is_absolute():
            path = root / path
        for file in sorted(path.glob("chunk[0-9]*.npz")):
            yield file, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--patch-dir", default="gbif_naip_dinov3_patch32_v1")
    parser.add_argument("--out-dir", default="gbif_naip_dinov3_patch32_packed_v1")
    parser.add_argument("--fallback-dir", action="append", default=[])
    args = parser.parse_args()

    root = Path(args.cache).expanduser()
    src = root / args.patch_dir
    out = root / args.out_dir
    manifest_path = src / "manifest.npz"
    if not manifest_path.exists():
        raise SystemExit(f"missing {manifest_path}")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, out / "manifest.npz")

    manifest = np.load(manifest_path)
    ids = manifest["gbifID"].astype(np.int64)
    row_for_id = {int(g): i for i, g in enumerate(ids)}
    n = len(ids)

    patch = open_memmap(
        out / "patch.npy", mode="w+", dtype=np.float16,
        shape=(n, 32, 32, 1024),
    )
    coords = open_memmap(
        out / "coords.npy", mode="w+", dtype=np.float32,
        shape=(n, 32, 32, 4),
    )
    valid = open_memmap(out / "valid.npy", mode="w+", dtype=np.bool_, shape=(n,))
    valid[:] = False

    rows_written = 0
    chunks_seen = 0
    for file, is_fallback in chunk_files(root, args.patch_dir, args.fallback_dir):
        z = np.load(file, allow_pickle=True)
        gids = z["gbifID"].astype(np.int64)
        p = z["patch"]
        lat = z["patch_lat"].astype(np.float32)
        lon = z["patch_lon"].astype(np.float32)
        elev = z["patch_elev"].astype(np.float32) if "patch_elev" in z else None
        chunks_seen += 1
        for j, gid in enumerate(gids):
            row = row_for_id.get(int(gid))
            if row is None:
                raise SystemExit(f"{file} gbifID {int(gid)} is absent from manifest")
            if valid[row]:
                if is_fallback:
                    continue
                raise SystemExit(f"duplicate primary gbifID {int(gid)} in {file}")
            patch[row] = p[j]
            coords[row, ..., 0] = lat[j]
            coords[row, ..., 1] = lon[j]
            if elev is None:
                coords[row, ..., 2] = np.float32(manifest["elev_m"][row])
            else:
                fallback_elev = np.float32(manifest["elev_m"][row])
                coords[row, ..., 2] = np.where(np.isfinite(elev[j]), elev[j], fallback_elev)
            coords[row, ..., 3] = np.float32(manifest["event_day"][row])
            valid[row] = np.isfinite(lat[j]).all() and np.isfinite(lon[j]).all()
            rows_written += 1
        if chunks_seen % 100 == 0:
            patch.flush()
            coords.flush()
            valid.flush()
            print(f"packed chunks={chunks_seen:,} rows={int(valid.sum()):,}/{n:,}", flush=True)

    patch.flush()
    coords.flush()
    valid.flush()
    meta = {
        "source_patch_dir": args.patch_dir,
        "fallback_dirs": args.fallback_dir,
        "rows": n,
        "valid_rows": int(valid.sum()),
        "patch_shape": [32, 32, 1024],
        "patch_dtype": "float16",
        "coords_shape": [32, 32, 4],
        "coords_dtype": "float32",
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"OK packed patch32 rows={meta['valid_rows']:,}/{n:,} chunks={chunks_seen:,} -> {out}", flush=True)


if __name__ == "__main__":
    main()
