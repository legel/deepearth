"""Clay v1.5 embeddings of every FluxFill Sentinel-2 chip: the CLS token for all chips, patch tokens for a pilot.

Reads `<s2 root>/sites/<site>.zarr` (see data/sentinel2/store.py) and writes `<out>/sites/<site>.zarr`:

    cls       (N, 1024) float16    encoder CLS token (the first output token)
    patches   (N, 16, 1024) float16  the 4 x 4 patch tokens, row-major (pilot sites only)
    time_us   (N,) int64           acquisition time, microseconds since the Unix epoch, UTC
    meta/acq_key, meta/item_id, meta/mgrs_tile (N,) str
    meta/cloud_frac, shadow_frac, snow_frac, invalid_frac, clear_frac, nodata_frac, cld_mean (N,) float32

Row i of the embedding store is row i of the chip store. Rows are appended as the chip store grows (resumable).
Every chip is embedded, cloudy or not; the SCL fractions are kept so the downstream model can learn reliability.

    python embed.py --s2 <s2 root> --out <clay root> --ckpt <encoder .pt> [--pilot US-Ha1,...] [--watch 600]
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numcodecs
import numpy as np
import torch
import zarr

from inputs import (PATCH, WAVES_UM, clay_stack, latlon_features, normalise, time_features, to_size,
                    upsample_r20)

COMPRESSOR = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)
META = ("cloud_frac", "shadow_frac", "snow_frac", "invalid_frac", "clear_frac", "nodata_frac", "cld_mean")
STR_META = ("acq_key", "item_id", "mgrs_tile")
ENCODER_ARGS = dict(mask_ratio=0.0, patch_size=PATCH, shuffle=False, dim=1024, depth=24, heads=16, dim_head=64,
                    mlp_ratio=4.0)


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def sha256(path: Path, block: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(block):
            h.update(b)
    return h.hexdigest()


def export_encoder(ckpt: Path, out: Path) -> dict:
    """Save the encoder weights of the Clay v1.5 Lightning checkpoint (keys `model.encoder.*`) as a plain state dict."""
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    enc = {k[len("model.encoder."):]: v for k, v in sd.items() if k.startswith("model.encoder.")}
    torch.save(enc, out)
    info = {"source_checkpoint": str(ckpt), "source_sha256": sha256(ckpt), "encoder_sha256": sha256(out),
            "n_tensors": len(enc), "n_params": int(sum(v.numel() for v in enc.values()))}
    out.with_suffix(".json").write_text(json.dumps(info, indent=1))
    return info


def load_encoder(path: Path, device: str):
    from claymodel.model import Encoder
    enc = Encoder(**ENCODER_ARGS)
    enc.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return enc.eval().to(device)


@torch.no_grad()
def encode(enc, pixels: np.ndarray, time_f: np.ndarray, latlon_f: np.ndarray, gsd: float, device: str,
           batch: int = 256, pause: float = 0.0) -> tuple:
    """(CLS (N, 1024), patch tokens (N, L, 1024)) as float16 numpy arrays."""
    waves = torch.tensor(WAVES_UM, device=device)
    g = torch.tensor(gsd, device=device)
    cls, pt = [], []
    for i in range(0, len(pixels), batch):
        cube = {"pixels": torch.from_numpy(np.ascontiguousarray(pixels[i:i + batch])).float().to(device),
                "time": torch.from_numpy(time_f[i:i + batch]).to(device),
                "latlon": torch.from_numpy(latlon_f[i:i + batch]).to(device), "gsd": g, "waves": waves}
        out, *_ = enc(cube)
        cls.append(out[:, 0].half().cpu().numpy())
        pt.append(out[:, 1:].half().cpu().numpy())
        if pause:
            time.sleep(pause)
    return np.concatenate(cls), np.concatenate(pt)


def chip_inputs(src, i0: int, i1: int, mode: str = "edge", harmonise: bool = True) -> tuple:
    """Clay pixels (n, 10, s, s), GSD, time and lat/lon features for chip rows [i0, i1) of a chip store."""
    m = src["meta"]
    r20 = upsample_r20(src["r20"][i0:i1], m["x_ul_r10"][i0:i1], m["y_ul_r10"][i0:i1], m["x_ul_r20"][i0:i1],
                       m["y_ul_r20"][i0:i1])
    off = m["boa_add_offset"][i0:i1].astype(np.int32) if harmonise else np.zeros(i1 - i0, np.int32)
    x, _ = clay_stack(src["r10"][i0:i1], r20, off)
    pix, gsd = to_size(normalise(x), mode)
    tf = time_features(src["time_us"][i0:i1])
    ll = latlon_features(src.attrs["tower_lat"], src.attrs["tower_lon"], i1 - i0)
    return pix, gsd, tf, ll


def open_out(path: Path, src, with_patches: bool, info: dict):
    g = zarr.open_group(str(path), mode="a")
    if "cls" not in g:
        g.create_dataset("cls", shape=(0, 1024), chunks=(1024, 1024), dtype="float16", compressor=COMPRESSOR)
        g.create_dataset("time_us", shape=(0,), chunks=(4096,), dtype="int64", compressor=COMPRESSOR)
        mg = g.require_group("meta")
        for k in META:
            mg.create_dataset(k, shape=(0,), chunks=(4096,), dtype="float32", compressor=COMPRESSOR)
        for k in STR_META:
            mg.create_dataset(k, shape=(0,), chunks=(4096,), dtype=object, object_codec=numcodecs.VLenUTF8(),
                              compressor=COMPRESSOR)
        g.attrs.update(site_id=src.attrs["site_id"], tower_lat=src.attrs["tower_lat"],
                       tower_lon=src.attrs["tower_lon"], n_committed=0, model="Clay v1.5", **info)
    if with_patches and "patches" not in g:
        n = int(g.attrs["n_committed"])
        if n:
            raise RuntimeError(f"{path}: patch tokens requested after {n} rows were written without them")
        g.create_dataset("patches", shape=(0, 16, 1024), chunks=(64, 16, 1024), dtype="float16",
                         compressor=COMPRESSOR)
    n = int(g.attrs["n_committed"])
    if n and g.attrs.get("input_mode") != info["input_mode"]:
        raise RuntimeError(f"{path}: written with input mode {g.attrs.get('input_mode')!r}, not {info['input_mode']!r}")
    for a in [g["cls"], g["time_us"], *(g["meta"][k] for k in META + STR_META)] + ([g["patches"]] if "patches" in g
                                                                                   else []):
        if a.shape[0] != n:
            a.resize((n,) + a.shape[1:])
    return g


def embed_site(enc, s2_path: Path, out_path: Path, with_patches: bool, info: dict, device: str, batch: int,
               pause: float, mode: str = "edge", block: int = 2048) -> int:
    src = zarr.open_group(str(s2_path), mode="r")
    n_src = int(src.attrs["n_committed"])
    g = open_out(out_path, src, with_patches, info)
    done = int(g.attrs["n_committed"])
    for i0 in range(done, n_src, block):
        i1 = min(n_src, i0 + block)
        pix, gsd, tf, ll = chip_inputs(src, i0, i1, mode)
        cls, pt = encode(enc, pix, tf, ll, gsd, device, batch, pause)
        g["cls"].append(cls)
        if "patches" in g:
            g["patches"].append(pt)
        g["time_us"].append(src["time_us"][i0:i1])
        for k in META:
            g["meta"][k].append(src["meta"][k][i0:i1].astype(np.float32))
        for k in STR_META:
            g["meta"][k].append(np.asarray(src["meta"][k][i0:i1], dtype=object))
        g.attrs["n_committed"] = i1
    return n_src - done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True, help="encoder state dict from export_encoder")
    ap.add_argument("--export-from", type=Path, help="Clay Lightning checkpoint to export the encoder from")
    ap.add_argument("--pilot", default="", help="comma-separated sites that also store patch tokens")
    ap.add_argument("--only", default="")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--pause", type=float, default=0.0, help="seconds to sleep between batches (GPU power cap)")
    ap.add_argument("--mode", default="edge", help="inputs.to_size mode; edge was closest to a true 320 m chip")
    ap.add_argument("--watch", type=float, default=0.0, help="re-scan every N seconds until gathering is done")
    a = ap.parse_args()
    if a.export_from and not a.ckpt.exists():
        log("export", export_encoder(a.export_from, a.ckpt))
    info = json.loads(a.ckpt.with_suffix(".json").read_text())
    info = {"encoder_sha256": info["encoder_sha256"], "source_sha256": info["source_sha256"],
            "input_mode": f"{a.mode} pad/resize 30->32 px (inputs.to_size), 20 m bands nearest to 10 m, "
                          "DN + BOA_ADD_OFFSET, no-data to band mean"}
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    enc = load_encoder(a.ckpt, device)
    pilot = set(a.pilot.split(",")) if a.pilot else set()
    (a.out / "sites").mkdir(parents=True, exist_ok=True)
    while True:
        paths = sorted((a.s2 / "sites").glob("*.zarr"))
        if a.only:
            paths = [p for p in paths if p.stem in a.only.split(",")]
        total, t0 = 0, time.time()
        for p in paths:
            try:
                n = embed_site(enc, p, a.out / "sites" / p.name, p.stem in pilot, info, device, a.batch, a.pause,
                               a.mode)
            except Exception as e:
                log(f"{p.stem}: failed {e!r}")
                continue
            if n:
                total += n
                log(f"{p.stem}: +{n} chips embedded ({total / (time.time() - t0):.0f} chips/s this pass)")
        if not a.watch:
            break
        gather_done = (a.s2 / "GATHER_DONE").exists()
        if gather_done and total == 0:
            break
        time.sleep(a.watch)
    log("DONE")


if __name__ == "__main__":
    sys.exit(main())
