"""Preflight checks for the NAIP 2024 DINOv3 patch32 build."""
import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import requests


MODEL = "facebook/dinov3-vitl16-pretrain-sat493m"
PATCH_VALUES = 32 * 32 * 1024


def load_token_ids(root):
    files = sorted((root / "gbif_tokens").glob("*.npz"))
    if not files:
        raise SystemExit(f"missing train/test shards under {root / 'gbif_tokens'}")
    ids = []
    for file in files:
        z = np.load(file)
        if "gbifID" not in z:
            raise SystemExit(f"{file} is missing gbifID")
        ids.append(z["gbifID"].astype(np.int64))
    return np.concatenate(ids)


def hf_token_path():
    explicit = os.environ.get("HF_TOKEN_PATH")
    if explicit:
        return Path(explicit).expanduser()
    return Path.home() / ".cache" / "huggingface" / "token"


def check_hf():
    token = hf_token_path()
    if not token.exists():
        return False, f"missing HF token at {token}"
    headers = {"Authorization": "Bearer " + token.read_text().strip()}
    try:
        r = requests.get(f"https://huggingface.co/api/models/{MODEL}", headers=headers, timeout=30)
    except Exception as e:
        return False, f"HF request failed: {type(e).__name__}: {e}"
    if r.status_code != 200:
        return False, f"HF model access failed with status {r.status_code}"
    return True, f"HF model access ok for {MODEL}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.environ.get("DEEPCAL_CACHE", "."))
    ap.add_argument("--dtype", choices=("float16", "float32"), default=os.environ.get("NAIP_PATCH_DTYPE", "float16"))
    ap.add_argument("--require-catalog", action="store_true")
    ap.add_argument("--require-usgs", action="store_true")
    ap.add_argument("--require-hf", action="store_true")
    args = ap.parse_args()

    root = Path(args.cache).expanduser()
    ids = load_token_ids(root)
    bytes_per = 2 if args.dtype == "float16" else 4
    needed = len(ids) * PATCH_VALUES * bytes_per
    usage = shutil.disk_usage(root)
    print(f"cache={root}")
    print(f"rows={len(ids):,}")
    print(f"patch_shape=(32,32,1024)")
    print(f"dtype={args.dtype}")
    print(f"payload_estimate={needed/1e12:.2f}TB")
    print(f"disk_free={usage.free/1e12:.2f}TB")
    print(f"disk_ok={usage.free > needed * 1.10}")

    catalog = root / "env_priors" / "naip2024_tiles.json"
    print(f"catalog_present={catalog.exists()} path={catalog}")
    if args.require_catalog and not catalog.exists():
        raise SystemExit(2)

    usgs = Path(os.environ.get("USGS_M2M_TOKEN", str(Path.home() / ".usgs_m2m_token"))).expanduser()
    print(f"usgs_token_present={usgs.exists()} path={usgs}")
    if args.require_usgs and not usgs.exists():
        raise SystemExit(3)

    ok, msg = check_hf()
    print(f"hf_ok={ok} {msg}")
    if args.require_hf and not ok:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
