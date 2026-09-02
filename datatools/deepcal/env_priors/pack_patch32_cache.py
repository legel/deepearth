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
        try:
            gids = z["gbifID"].astype(np.int64)
            p = z["patch"]
            lat = z["patch_lat"].astype(np.float32)
            lon = z["patch_lon"].astype(np.float32)
            elev = z["patch_elev"].astype(np.float32) if "patch_elev" in z else None
            chunks_seen += 1
            rows = np.array([row_for_id.get(int(g), -1) for g in gids], np.int64)
            if (rows < 0).any():
                bad = int(gids[int(np.flatnonzero(rows < 0)[0])])
                raise SystemExit(f"{file} gbifID {bad} is absent from manifest")
            keep = ~np.asarray(valid[rows], dtype=bool)
            if not is_fallback and not keep.all():
                bad = int(gids[int(np.flatnonzero(~keep)[0])])
                raise SystemExit(f"duplicate primary gbifID {bad} in {file}")
            if not keep.any():
                continue
            src_rows = np.flatnonzero(keep)
            dst_rows = rows[keep]
            patch[dst_rows] = p[src_rows]
            coords[dst_rows, ..., 0] = lat[src_rows]
            coords[dst_rows, ..., 1] = lon[src_rows]
            if elev is None:
                coords[dst_rows, ..., 2] = manifest["elev_m"][dst_rows, None, None].astype(np.float32)
            else:
                fallback_elev = manifest["elev_m"][dst_rows, None, None].astype(np.float32)
                coords[dst_rows, ..., 2] = np.where(
                    np.isfinite(elev[src_rows]), elev[src_rows], fallback_elev
                )
            coords[dst_rows, ..., 3] = manifest["event_day"][dst_rows, None, None].astype(np.float32)
            valid[dst_rows] = np.isfinite(lat[src_rows]).all(axis=(1, 2)) \
                & np.isfinite(lon[src_rows]).all(axis=(1, 2))
            rows_written += len(src_rows)
        finally:
            z.close()
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
