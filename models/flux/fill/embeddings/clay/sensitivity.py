"""Sensitivity of the Clay CLS embedding to how a 30 x 30 px chip is fed to the model.

For a sample of chips at a few sites, embeds each chip under every `inputs.to_size` mode and, as a reference, a
true 32 x 32 px (320 m) chip read again from the source COGs around the same tower. Reports the cosine similarity of
each mode's CLS to the reference and to the chosen mode, and the effect of harmonising the baseline offset on
chips from baseline 04.00 and later.

    python sensitivity.py --s2 <s2 root> --ckpt <encoder .pt> --sites US-Ha1,DE-Tha,AU-How --n 40 --out <json>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import zarr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "data" / "sentinel2"))

from embed import encode, load_encoder  # noqa: E402
from inputs import clay_stack, latlon_features, normalise, time_features, to_size, upsample_r20  # noqa: E402

MODES = ("reflect", "edge", "zero", "bilinear", "crop24")


def reread_320(scene_meta: dict, hrefs: dict, lon: float, lat: float, token):
    """(r10 (4, 32, 32), r20 on the 10 m grid (6, 32, 32)) for a 320 m window around the tower."""
    import gather
    from geometry import lonlat_to_utm, upsample_to
    x, y = lonlat_to_utm(lon, lat, scene_meta["epsg"])
    tk = token if scene_meta["source"] == "planetary-computer" else None
    r10, w10 = [], None
    for b in ("B02", "B03", "B04", "B08"):
        h = tk.sign(hrefs[b]) if tk else hrefs[b]
        a, w10 = gather._open_read(h, x, y, 320, 10, np.uint16)
        r10.append(a)
    r20 = []
    for b in ("B05", "B06", "B07", "B8A", "B11", "B12"):
        h = tk.sign(hrefs[b]) if tk else hrefs[b]
        a, w20 = gather._open_read(h, x, y, 340, 20, np.uint16)
        r20.append(upsample_to(a, w20, w10))
    return np.stack(r10), np.stack(r20)


def cos(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--sites", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from stac import PCToken
    token = PCToken()
    enc = load_encoder(a.ckpt, "cuda")
    rng = np.random.default_rng(0)
    report = {"modes": MODES, "sites": {}}
    allres = {m: [] for m in MODES}
    allres_ref = {m: [] for m in MODES}
    harm = []
    for sid in a.sites.split(","):
        src = zarr.open_group(str(a.s2 / "sites" / f"{sid}.zarr"), mode="r")
        n = int(src.attrs["n_committed"])
        idx = np.sort(rng.choice(n, size=min(a.n, n), replace=False))
        m = src["meta"]
        lat, lon = src.attrs["tower_lat"], src.attrs["tower_lon"]
        r20 = upsample_r20(src["r20"].oindex[idx], m["x_ul_r10"][:][idx], m["y_ul_r10"][:][idx],
                           m["x_ul_r20"][:][idx], m["y_ul_r20"][:][idx])
        off = m["boa_add_offset"][:][idx].astype(np.int32)
        x, _ = clay_stack(src["r10"].oindex[idx], r20, off)
        xn = normalise(x)
        tf = time_features(src["time_us"][:][idx])
        ll = latlon_features(lat, lon, len(idx))
        cls = {}
        for mode in MODES:
            pix, gsd = to_size(xn, mode)
            cls[mode], _ = encode(enc, pix, tf, ll, gsd, "cuda", batch=64)
        # Reference: the true 320 m chip from the source.
        scenes = {json.loads(l)["acq_key"]: json.loads(l) for l in open(a.s2 / "scenes" / f"{sid}.jsonl")}
        keys = m["acq_key"][:][idx]
        ref10, ref20 = zip(*[reread_320(scenes[k], scenes[k]["hrefs"], lon, lat, token) for k in keys])
        xr, _ = clay_stack(np.stack(ref10), np.stack(ref20), off)
        cref, _ = encode(enc, normalise(xr), tf, ll, 10.0, "cuda", batch=64)
        # Harmonisation: raw DN versus DN + offset, for chips with a non-zero offset.
        nz = off != 0
        if nz.any():
            xraw, _ = clay_stack(src["r10"].oindex[idx[nz]], r20[nz], np.zeros(nz.sum(), np.int32))
            craw, _ = encode(enc, to_size(normalise(xraw), "reflect")[0], tf[nz], ll[nz], 10.0, "cuda", batch=64)
            harm.extend(cos(craw, cls["reflect"][nz]).tolist())
        site = {}
        for mode in MODES:
            c_ref = cos(cls[mode], cref)
            c_ch = cos(cls[mode], cls["reflect"])
            allres_ref[mode].extend(c_ref.tolist())
            allres[mode].extend(c_ch.tolist())
            site[mode] = {"cos_to_true320_mean": float(c_ref.mean()), "cos_to_true320_min": float(c_ref.min()),
                          "cos_to_reflect_mean": float(c_ch.mean())}
        report["sites"][sid] = {"n": int(len(idx)), **site}
        print(sid, json.dumps(site), flush=True)
    report["all"] = {mode: {"cos_to_true320_mean": float(np.mean(allres_ref[mode])),
                            "cos_to_true320_p05": float(np.percentile(allres_ref[mode], 5)),
                            "cos_to_reflect_mean": float(np.mean(allres[mode]))} for mode in MODES}
    if harm:
        report["harmonise_offset_cos_raw_vs_harmonised"] = {"n": len(harm), "mean": float(np.mean(harm)),
                                                            "min": float(np.min(harm))}
    a.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report["all"], indent=1))
    print(json.dumps(report.get("harmonise_offset_cos_raw_vs_harmonised")))


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
