"""Standalone spacetime-encoder probe -- train + evaluate Earth4D IN ISOLATION.

No fusion model, no 790M backbone, no full benchmark suite -- just Earth4D + a linear head over a
subsample of observation coordinates. Measures the encoder's science (science.md rules 1-6, 24): does the
Earth4D positional field make space-time PREDICTIVE of biology at HELD-OUT locations (spatial generalization,
the SDM task B1/B5/B8)? Fast.

Objective (standalone `st_gain`): held-out-block family accuracy from Earth4D(coords) MINUS from raw
normalized coordinates. >0 ⟹ the multi-resolution positional encoder adds spatial-biology structure a raw
coordinate cannot. Reuses Earth4D unchanged (no core edit).

  python -m deepearth.autoresearch.programs.spacetime.probe --cache_dir data/deepcal --steps 800
"""
import argparse
import csv
import glob
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepearth.encoders.spacetime.earth4d import Earth4D


def load_obs(cache: str, n_shards: int):
    """(lat, lon) + family-per-observation from a subsample of token shards -- fast, no full build."""
    cachep = Path(cache)
    vocab = np.load(cachep / "gbif_vocab.npz", allow_pickle=True)
    gidx = vocab["global_idx"]
    rows = list(csv.DictReader(open(cachep / "derived/species_index.csv")))
    family = np.array([rows[i]["family"] for i in gidx])
    fam_id = np.unique(family, return_inverse=True)[1]          # species-local -> family id
    lat, lon, sp = [], [], []
    for f in sorted(glob.glob(str(cachep / "gbif_tokens/*.npz")))[:n_shards]:
        z = np.load(f)
        lat.append(z["lat"]); lon.append(z["lon"]); sp.append(z["species_local"])
    lat = np.concatenate(lat).astype(np.float32)
    lon = np.concatenate(lon).astype(np.float32)
    sp = np.concatenate(sp).astype(np.int64)
    fam = fam_id[sp].astype(np.int64)                           # family per observation
    n_fam = int(fam_id.max()) + 1
    return lat, lon, fam, n_fam


def spatial_holdout(lat, lon, frac=0.2, block=0.5, seed=0):
    """Hold out whole 0.5-degree spatial blocks (tests generalization to UNSEEN locations, not memorization)."""
    blk = (np.floor(lat / block).astype(np.int64) * 100000 + np.floor(lon / block).astype(np.int64))
    ublk = np.unique(blk)
    rng = np.random.default_rng(seed); rng.shuffle(ublk)
    held = set(ublk[: int(len(ublk) * frac)].tolist())
    return np.array([b in held for b in blk])                  # bool [N_obs], True = held-out location


def evaluate(feats, fam, test, n_fam, dev, steps, lr, tag, head_hidden=0):
    """Train a linear head feats->family on TRAIN locations; report held-out-block accuracy."""
    train = ~test
    Xtr, ytr = feats[train].to(dev), fam[train].to(dev)
    Xte, yte = feats[test].to(dev), fam[test].to(dev)
    if head_hidden > 0:
        head = nn.Sequential(nn.Linear(feats.shape[1], head_hidden), nn.ReLU(),
                             nn.Linear(head_hidden, n_fam)).to(dev)
    else:
        head = nn.Linear(feats.shape[1], n_fam).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(steps):
        idx = torch.randint(0, Xtr.shape[0], (4096,), device=dev)
        loss = F.cross_entropy(head(Xtr[idx]), ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        logits = head(Xte)
        acc = (logits.argmax(-1) == yte).float().mean().item()
        top5 = (logits.topk(5, -1).indices == yte[:, None]).any(-1).float().mean().item()
    return acc, top5


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="data/deepcal")
    ap.add_argument("--n_shards", type=int, default=8)         # ~65k obs; the lever for coverage/speed
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spatial_levels", type=int, default=18)   # S3: expose Earth4D capacity
    ap.add_argument("--temporal_levels", type=int, default=18)
    ap.add_argument("--log2_hashmap", type=int, default=20)
    ap.add_argument("--head_hidden", type=int, default=0)       # 0=linear head; >0=MLP head width
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    dev = a.device if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    lat, lon, fam, n_fam = load_obs(a.cache_dir, a.n_shards)
    test = spatial_holdout(lat, lon, a.holdout, seed=a.seed)
    fam_t = torch.tensor(fam)
    coords = torch.tensor(np.stack([lat, lon, np.zeros_like(lat), np.zeros_like(lat)], 1))  # [N,4]=(lat,lon,elev=0,t=0)

    enc = Earth4D(verbose=False, spatial_levels=a.spatial_levels, temporal_levels=a.temporal_levels,   # S3: exposed capacity
                  spatial_log2_hashmap_size=a.log2_hashmap, temporal_log2_hashmap_size=a.log2_hashmap, freq_log_scale_init=-2.5).to(dev)
    with torch.no_grad():
        e4d = enc(coords.to(dev)).cpu()                          # [N, output_dim] Earth4D positional features
    rn = np.stack([lat / 90.0, lon / 180.0], 1).astype(np.float32)
    raw = torch.tensor(rn)                                        # raw normalized coords baseline
    # Random Fourier Features of (lat,lon): fair nonlinear positional-encoding control vs Earth4D
    rff_rng = np.random.default_rng(0)
    proj = rn @ (rff_rng.normal(0, 8.0, (2, e4d.shape[1] // 2)).astype(np.float32))
    rff = torch.tensor(np.concatenate([np.sin(proj), np.cos(proj)], 1).astype(np.float32))

    raw_acc, raw_t5 = evaluate(raw, fam_t, test, n_fam, dev, a.steps, a.lr, "raw", a.head_hidden)
    rff_acc, rff_t5 = evaluate(rff, fam_t, test, n_fam, dev, a.steps, a.lr, "rff", a.head_hidden)
    e4d_acc, e4d_t5 = evaluate(e4d, fam_t, test, n_fam, dev, a.steps, a.lr, "earth4d", a.head_hidden)
    dt = time.time() - t0
    print(f"=== SPACETIME encoder (standalone) | obs={len(lat)} held-out-blocks={int(test.sum())} families={n_fam} ===")
    print(f"  held-out family acc | raw-coords {raw_acc:.4f} | RFF {rff_acc:.4f} | Earth4D {e4d_acc:.4f}   st_gain(vs raw) {e4d_acc - raw_acc:+.4f}  st_gain(vs RFF) {e4d_acc - rff_acc:+.4f}")
    print(f"  held-out top5 acc   | raw-coords {raw_t5:.4f} | RFF {rff_t5:.4f} | Earth4D {e4d_t5:.4f}   top5_gain(vs raw) {e4d_t5 - raw_t5:+.4f}")
    print(f"  [profile] earth4d_dim={e4d.shape[1]}  frac_held={test.mean():.3f}  spatial_levels={a.spatial_levels} temporal_levels={a.temporal_levels} log2_hashmap={a.log2_hashmap}")
    print(f"  {len(lat)} obs, {a.steps}-step probe in {dt:.1f}s")
    return {"st_gain": e4d_acc - raw_acc, "earth4d_acc": e4d_acc, "raw_acc": raw_acc,
            "obs": len(lat), "seconds": dt}


if __name__ == "__main__":
    main()
