"""Preflight checks for the NAIP 2024 DINOv3 patch32 build."""
import argparse
import json
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


def hf_cache_has_model():
    home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))).expanduser()
    hub = home / "hub"
    pattern = "models--facebook--dinov3-vitl16-pretrain-sat493m*"
    return any(hub.glob(pattern))


def read_hf_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip(), "env"
    path = hf_token_path()
    if path.exists():
        return path.read_text().strip(), str(path)
    return None, str(path)


def check_hf():
    if hf_cache_has_model():
        return True, f"HF cache contains {MODEL}"
    token, source = read_hf_token()
    if not token:
        return False, f"missing HF token; checked env and {source}"
    headers = {"Authorization": "Bearer " + token}
    file_url = f"https://huggingface.co/{MODEL}/resolve/main/model.safetensors"
    try:
        who = requests.get("https://huggingface.co/api/whoami-v2", headers=headers, timeout=30)
        r = requests.get(file_url, headers={**headers, "Range": "bytes=0-0"}, stream=True, timeout=30)
        r.close()
    except Exception as e:
        return False, f"HF request failed: {type(e).__name__}: {e}"
    if who.status_code != 200:
        return False, f"HF token from {source} is not valid; whoami status {who.status_code}"
    if r.status_code not in (200, 206):
        return False, f"HF token from {source} lacks gated file access to {MODEL}; file status {r.status_code}"
    return True, f"HF gated file access ok for {MODEL} via {source}"


def catalog_summary(path):
    if not path.exists():
        return None
    with open(path) as f:
        tiles = json.load(f)
    local = 0
    valid_local = 0
    url = 0
    missing_local = 0
    for tile in tiles:
        if tile.get("url"):
            url += 1
        if tile.get("local_path"):
            local += 1
            if not Path(tile["local_path"]).expanduser().exists():
                missing_local += 1
            else:
                valid_local += 1
    return {
        "entries": len(tiles),
        "local_path": local,
        "valid_local_path": valid_local,
        "url": url,
        "missing_local_path": missing_local,
        "direct": valid_local + url,
    }


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

    catalog = Path(os.environ.get("NAIP_TILES_JSON", str(root / "env_priors" / "naip2024_tiles.json"))).expanduser()
    summary = catalog_summary(catalog)
    print(f"catalog_present={summary is not None} path={catalog}")
    if summary is not None:
        print(
            "catalog_entries={entries} local_path={local_path} valid_local_path={valid_local_path} url={url} "
            "missing_local_path={missing_local_path} direct_sources={direct}".format(**summary)
        )
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
