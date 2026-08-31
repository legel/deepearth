"""Smoke-test DINOv3 patch32 extraction without NAIP."""
import argparse

import numpy as np

from dinov3_patch32 import DINOv3Patch32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    chips = rng.integers(0, 256, size=(args.n, 3, 512, 512), dtype=np.uint8)
    embedder = DINOv3Patch32(batch=args.batch)
    patch = embedder.patch32(chips)
    if patch.shape != (args.n, 32, 32, 1024):
        raise SystemExit(f"bad patch shape {patch.shape}")
    if not np.isfinite(patch).all():
        raise SystemExit("patch contains non-finite values")
    print(f"OK DINOv3 patch32 shape={patch.shape} dtype={patch.dtype}")


if __name__ == "__main__":
    main()
