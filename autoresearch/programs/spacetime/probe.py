"""Standalone spacetime-encoder probe -- train + evaluate Earth4D IN ISOLATION.

No fusion model, no 790M backbone, no full benchmark suite -- just Earth4D + a linear head over a
subsample of observation coordinates. Measures the encoder's science (science.md rules 1-6, 24): does the
Earth4D positional field make space-time PREDICTIVE of biology at HELD-OUT locations (spatial generalization,
the SDM task B1/B5/B8)? Fast.

Objective (standalone `st_gain`): held-out-block family accuracy from Earth4D(coords) MINUS from raw
normalized coordinates. >0 ⟹ the multi-resolution positional encoder adds spatial-biology structure a raw
coordinate cannot. Reuses Earth4D unchanged (no core edit).

  python -m deepearth.autoresearch.programs.spacetime.probe --cache_dir data/deepcal --steps 800

FORECAST mode (--forecast, science.md rule 1: causal auto-regressive):
  Real event-time (gbif_eventtime.npz, joined by gbifID) is placed into Earth4D coord slot 3 (t), which the
  default path leaves at 0. The held-out set becomes the LATEST `holdout` fraction of observations BY TIME
  (train on the past, forecast the future) instead of held-out spatial blocks. The raw-coords and RFF
  baselines receive the identical normalized-time feature, so the st_gain isolates whether Earth4D's 4D
  multi-resolution field forecasts biology forward better than a plain space-time code. Default path
  (no --forecast) is byte-identical to before: t=0, spatial-block holdout.

ENVIRONMENT modes (science.md rules 1-6, 24 done right -- the positional field should represent the ENVIRONMENT,
biology follows; a coordinate is not the science, the environment at that coordinate is):
  --env         (Move 1) held-out-block family acc from real ENVIRONMENT covariates (worldclim+soil+elev,
                joined by gbifID) vs the best coordinate-PE (Earth4D / RFF / raw), plus an Earth4D+env fused
                head. Answers: does real environment >> any coordinate positional encoding? If yes the
                encoder's job is to REPRESENT environment, not index coordinates.
  --env_decode  (Move 2, rule 24 done right) train Earth4D END-TO-END to decode the physically-real,
                spatially-smooth ENVIRONMENT field (worldclim, standardized) at TRAIN obs as an auxiliary
                regression target, THEN predict biology from the learned field at the strict held-out set.
                Fair control = coord-MLP / RFF given the identical env-decode auxiliary. Answers: does an
                env-supervised field finally beat a generic PE, where the family-supervised field failed
                (-0.10)? A smooth environment target is the field rule-24 actually asks for.
Both default-off; the no-flag path is byte-identical.
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


def load_obs(cache: str, n_shards: int, with_time: bool = False, with_gid: bool = False):
    """(lat, lon) + family-per-observation from a subsample of token shards -- fast, no full build.

    If with_time, also joins per-observation event day (gbif_eventtime.npz, keyed by gbifID) and returns it
    as `days` [N] float32; otherwise returns days=None (default path unchanged). If with_gid, also returns
    the per-observation gbifID [N] int64 (for joining environment covariates); else gid=None."""
    cachep = Path(cache)
    vocab = np.load(cachep / "gbif_vocab.npz", allow_pickle=True)
    gidx = vocab["global_idx"]
    rows = list(csv.DictReader(open(cachep / "derived/species_index.csv")))
    family = np.array([rows[i]["family"] for i in gidx])
    fam_id = np.unique(family, return_inverse=True)[1]          # species-local -> family id
    id2day = None
    if with_time:
        et = np.load(cachep / "gbif_eventtime.npz")
        id2day = dict(zip(et["gbifID"].tolist(), et["days"].tolist()))
    lat, lon, sp, day, gid = [], [], [], [], []
    for f in sorted(glob.glob(str(cachep / "gbif_tokens/*.npz")))[:n_shards]:
        z = np.load(f)
        lat.append(z["lat"]); lon.append(z["lon"]); sp.append(z["species_local"])
        if with_time:
            day.append(np.array([id2day.get(int(i), np.nan) for i in z["gbifID"]], dtype=np.float32))
        if with_gid:
            gid.append(z["gbifID"].astype(np.int64))
    lat = np.concatenate(lat).astype(np.float32)
    lon = np.concatenate(lon).astype(np.float32)
    sp = np.concatenate(sp).astype(np.int64)
    fam = fam_id[sp].astype(np.int64)                           # family per observation
    n_fam = int(fam_id.max()) + 1
    days = np.concatenate(day).astype(np.float32) if with_time else None
    gids = np.concatenate(gid).astype(np.int64) if with_gid else None
    return lat, lon, fam, n_fam, days, gids


def load_env(cache: str, gid):
    """Join real ENVIRONMENT covariates to each observation by gbifID (science.md rule 24: environment field).

    Returns env [N, D] float32, standardized (zero-mean/unit-std per column over the covered obs), missing
    values imputed to 0 (= the column mean post-standardization). D = 19 worldclim + 9 soil + 1 elev = 29.
    A per-column present-mask is folded in as the impute so absent env reads as neutral, never leaks NaN."""
    cachep = Path(cache)
    wc = np.load(cachep / "gbif_worldclim_tokens.npz")
    so = np.load(cachep / "gbif_soil_tokens.npz")
    el = np.load(cachep / "gbif_elev.npz")
    wcmap = dict(zip(wc["gbifID"].tolist(), wc["worldclim"]))
    somap = dict(zip(so["gbifID"].tolist(), so["soil"]))
    elmap = dict(zip(el["gbifID"].tolist(), el["elev"].tolist()))
    D = 19 + 9 + 1
    env = np.full((len(gid), D), np.nan, np.float32)
    for i, g in enumerate(gid):
        g = int(g)
        if g in wcmap: env[i, :19] = wcmap[g]
        if g in somap: env[i, 19:28] = somap[g]
        if g in elmap: env[i, 28] = elmap[g]
    # standardize per column over present values, then impute missing to the (post-standardization) mean = 0
    mu = np.nanmean(env, 0); sd = np.nanstd(env, 0); sd[sd < 1e-6] = 1.0
    env = (env - mu) / sd
    env = np.nan_to_num(env, nan=0.0).astype(np.float32)
    return env


def spatial_holdout(lat, lon, frac=0.2, block=0.5, seed=0):
    """Hold out whole 0.5-degree spatial blocks (tests generalization to UNSEEN locations, not memorization)."""
    blk = (np.floor(lat / block).astype(np.int64) * 100000 + np.floor(lon / block).astype(np.int64))
    ublk = np.unique(blk)
    rng = np.random.default_rng(seed); rng.shuffle(ublk)
    held = set(ublk[: int(len(ublk) * frac)].tolist())
    return np.array([b in held for b in blk])                  # bool [N_obs], True = held-out location


def temporal_holdout(days, frac=0.2):
    """Hold out the LATEST `frac` of observations by event time (train past -> forecast future).

    Causal split (science.md rule 1): the test set is strictly in the future of the train set, so a passing
    st_gain means Earth4D forecasts forward, it does not interpolate a location it has already seen."""
    thr = np.nanquantile(days, 1.0 - frac)
    return days >= thr                                          # bool [N_obs], True = future (held out)


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
    ap.add_argument("--forecast", action="store_true")          # S1/rule1: causal past->future temporal split + live time coord
    ap.add_argument("--forecast_spatial", action="store_true")  # rule1 strict: future time AND held-out place (no location-recall shortcut)
    ap.add_argument("--recurrence", action="store_true")        # rule2b: 4D-LSTM rollout PROPAGATES past->future (replaces static lookup head)
    ap.add_argument("--rec_k", type=int, default=16)            # causal context-window size (K nearest past neighbours)
    ap.add_argument("--rec_hidden", type=int, default=256)      # LSTM hidden width
    ap.add_argument("--rec_time_cond", action="store_true")     # rule24+2b: per-step token = QUERY cell re-encoded FORWARD in time (propagate encoder STATE, not a static neighbour code)
    ap.add_argument("--gnn", action="store_true")               # rule1+2b: GraphCast/GenCast-style message-passing propagator (learned space-time edges, multi-hop) vs LSTM vs no-propagation; reports ABSOLUTE forecast skill
    ap.add_argument("--gnn_hops", type=int, default=2)          # message-passing rounds (multi-hop)
    ap.add_argument("--phenology", action="store_true")         # DECISIVE non-stationary control: predict day-of-year (seasonal timing) a static coord map CANNOT capture; static vs GNN vs LSTM propagator over Earth4D/RFF/raw
    ap.add_argument("--pheno_tol", type=float, default=15.0)    # within-+/-N-days circular accuracy tolerance
    ap.add_argument("--pheno_attn", action="store_true")        # ROUND-1: add temporal self-attention propagator (AttnDOY) alongside static/GNN/LSTM
    ap.add_argument("--attn_heads", type=int, default=4)        # attention heads for AttnDOY
    ap.add_argument("--attn_layers", type=int, default=2)       # self-attention encoder layers over the past window
    ap.add_argument("--pheno_species", action="store_true")     # ROUND-2: species-conditioned LSTM propagator (neighbour species emb + query species + match bit)
    ap.add_argument("--first_arrival", action="store_true")     # dynamic target: per (0.5deg cell, species) EARLIEST DOY (seasonal onset, leading edge); static vs GNN vs LSTM over Earth4D/RFF/raw
    ap.add_argument("--abundance", action="store_true")         # dynamic target: log obs-count in query cell over trailing window (activity a static climatology cannot forecast); static vs GNN vs LSTM
    ap.add_argument("--abund_win", type=float, default=90.0)    # trailing window (days) for the abundance count
    ap.add_argument("--field_decode", action="store_true")      # rule24: TRAIN the encoder end-to-end to decode the dense family field between sparse obs; fair control = trainable-head-on-RFF / coord-MLP
    ap.add_argument("--env", action="store_true")               # Move1: real ENVIRONMENT covariates (worldclim+soil+elev) vs coordinate-PE; + Earth4D+env fused
    ap.add_argument("--env_decode", action="store_true")        # Move2/rule24: TRAIN encoder to decode the smooth ENVIRONMENT field (aux), then predict biology from the learned field
    ap.add_argument("--env_aux_weight", type=float, default=1.0) # weight on the env-reconstruction auxiliary loss
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    dev = a.device if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    need_gid = a.env or a.env_decode
    lat, lon, fam, n_fam, days, gid = load_obs(a.cache_dir, a.n_shards, with_time=a.forecast, with_gid=need_gid)
    fam_t = torch.tensor(fam)

    if a.forecast:
        # science.md rule 1: causal auto-regressive -- test = future, train = past; time is a LIVE coordinate.
        test = temporal_holdout(days, a.holdout)
        if a.forecast_spatial:
            # Remove the same-location recall shortcut: future test set must ALSO be a held-out spatial block,
            # so raw lat/lon cannot memorize place->family. This is the true rule-1 forecast: new place, future time.
            sp_held = spatial_holdout(lat, lon, a.holdout, seed=a.seed)
            test = test & sp_held
        tmin, tspan = np.nanmin(days), max(np.nanmax(days) - np.nanmin(days), 1e-6)
        tnorm = ((days - tmin) / tspan).astype(np.float32)      # [0,1] event time
        coords = torch.tensor(np.stack([lat, lon, np.zeros_like(lat), tnorm], 1))  # [N,4]=(lat,lon,elev=0,t=REAL)
    else:
        test = spatial_holdout(lat, lon, a.holdout, seed=a.seed)
        coords = torch.tensor(np.stack([lat, lon, np.zeros_like(lat), np.zeros_like(lat)], 1))  # [N,4] t=0

    enc = Earth4D(verbose=False, spatial_levels=a.spatial_levels, temporal_levels=a.temporal_levels,   # S3: exposed capacity
                  spatial_log2_hashmap_size=a.log2_hashmap, temporal_log2_hashmap_size=a.log2_hashmap, freq_log_scale_init=-2.5).to(dev)

    if a.env or a.env_decode:
        # science.md rules 1-6, 24 done RIGHT: the positional field should represent the ENVIRONMENT; biology
        # follows. Real env covariates (worldclim+soil+elev) joined by gbifID -> the science-aligned question.
        env = load_env(a.cache_dir, gid)                         # [N, 29] standardized, NaN-imputed to 0
        rn = np.stack([lat / 90.0, lon / 180.0], 1).astype(np.float32)
        if a.forecast:
            rn = np.concatenate([rn, tnorm[:, None]], 1)

    if a.env:
        # ---- Move 1: is real ENVIRONMENT >> any coordinate positional encoding at held-out biology? ----
        # Fair controls: Earth4D(coords), RFF(coords), raw(coords) -- the best coordinate-PE. Plus Earth4D+env
        # fused. All share the SAME head (linear or MLP), steps, lr. st_gain reported as env-or-fused MINUS the
        # best coordinate-PE control -> if env >> best coord-PE, the encoder's job is to REPRESENT environment.
        with torch.no_grad():
            e4d = enc(coords.to(dev)).cpu()
        env_t = torch.tensor(env)
        raw = torch.tensor(rn)
        rff_rng = np.random.default_rng(0)
        proj = rn @ (rff_rng.normal(0, 8.0, (rn.shape[1], e4d.shape[1] // 2)).astype(np.float32))
        rff = torch.tensor(np.concatenate([np.sin(proj), np.cos(proj)], 1).astype(np.float32))
        fused = torch.cat([e4d, env_t], 1)                       # Earth4D coords ++ real environment
        raw_acc, raw_t5 = evaluate(raw, fam_t, test, n_fam, dev, a.steps, a.lr, "raw", a.head_hidden)
        rff_acc, rff_t5 = evaluate(rff, fam_t, test, n_fam, dev, a.steps, a.lr, "rff", a.head_hidden)
        e4d_acc, e4d_t5 = evaluate(e4d, fam_t, test, n_fam, dev, a.steps, a.lr, "earth4d", a.head_hidden)
        env_acc, env_t5 = evaluate(env_t, fam_t, test, n_fam, dev, a.steps, a.lr, "env", a.head_hidden)
        fus_acc, fus_t5 = evaluate(fused, fam_t, test, n_fam, dev, a.steps, a.lr, "fused", a.head_hidden)
        dt = time.time() - t0
        best_coord = max(raw_acc, rff_acc, e4d_acc)              # best coordinate-only PE
        mode = ("FORECAST(future+newplace)" if a.forecast_spatial else "FORECAST(past->future)") if a.forecast else "spatial-block"
        print(f"=== SPACETIME encoder (standalone) | mode=ENV({mode}) obs={len(lat)} held-out={int(test.sum())} families={n_fam} env_dim={env.shape[1]} ===")
        print(f"  held-out family acc | raw {raw_acc:.4f} | RFF {rff_acc:.4f} | Earth4D {e4d_acc:.4f} || ENV {env_acc:.4f} | Earth4D+ENV {fus_acc:.4f}")
        print(f"    st_gain(ENV vs best-coord-PE) {env_acc - best_coord:+.4f}   st_gain(fused vs best-coord-PE) {fus_acc - best_coord:+.4f}   (best-coord-PE={best_coord:.4f})")
        print(f"  held-out top5 acc   | raw {raw_t5:.4f} | RFF {rff_t5:.4f} | Earth4D {e4d_t5:.4f} || ENV {env_t5:.4f} | Earth4D+ENV {fus_t5:.4f}")
        print(f"  [profile] earth4d_dim={e4d.shape[1]} env_dim={env.shape[1]} frac_held={test.mean():.3f} head_hidden={a.head_hidden} steps={a.steps} forecast={a.forecast}")
        print(f"  {len(lat)} obs in {dt:.1f}s")
        return {"st_gain": env_acc - best_coord, "st_gain_fused": fus_acc - best_coord,
                "env_acc": env_acc, "fused_acc": fus_acc, "earth4d_acc": e4d_acc, "rff_acc": rff_acc,
                "raw_acc": raw_acc, "best_coord_pe": best_coord, "obs": len(lat), "seconds": dt, "env": True}

    if a.env_decode:
        # ---- Move 2 (rule 24 done right): env-supervised field. Train each encoder to ALSO decode the smooth ----
        # real ENVIRONMENT field (worldclim, standardized) at TRAIN obs as an aux regression target, THEN predict
        # biology from the learned field at the strict held-out set. Fair controls (mlp/rff) get the identical
        # aux. st_gain = env-supervised-Earth4D biology-acc MINUS best generic control -> does a physically-real
        # smooth target make the 4D field finally beat a generic PE (family-supervised field failed -0.10)?
        from deepearth.autoresearch.programs.spacetime.env_field import run_env_decode
        with torch.no_grad():
            fdim = enc(coords[:8].to(dev)).shape[1]
        env_tgt = env[:, :19]                                    # worldclim = the smooth, physically-real field
        e4d_acc, e4d_t5, n_te, e4d_er = run_env_decode("earth4d", coords, rn, env_tgt, fam, test, n_fam, dev,
                                                       enc=enc, feat_dim=fdim, steps=a.steps, lr=a.lr,
                                                       head_hidden=max(a.head_hidden, 256), aux_w=a.env_aux_weight)
        mlp_acc, mlp_t5, _, mlp_er = run_env_decode("mlp", coords, rn, env_tgt, fam, test, n_fam, dev,
                                                    feat_dim=fdim, steps=a.steps, lr=a.lr,
                                                    head_hidden=max(a.head_hidden, 256), aux_w=a.env_aux_weight)
        rff_acc, rff_t5, _, rff_er = run_env_decode("rff", coords, rn, env_tgt, fam, test, n_fam, dev,
                                                    feat_dim=fdim, steps=a.steps, lr=a.lr,
                                                    head_hidden=max(a.head_hidden, 256), aux_w=a.env_aux_weight)
        dt = time.time() - t0
        ctrl = max(mlp_acc, rff_acc)
        mode = ("FORECAST(future+newplace)" if a.forecast_spatial else "FORECAST(past->future)") if a.forecast else "spatial-block"
        print(f"=== SPACETIME encoder (standalone) | mode=ENV-DECODE({mode}) obs={len(lat)} held-out={n_te} families={n_fam} aux_w={a.env_aux_weight} ===")
        print(f"  held-out family acc | mlp {mlp_acc:.4f} | RFF {rff_acc:.4f} | Earth4D {e4d_acc:.4f}   st_gain(vs mlp) {e4d_acc - mlp_acc:+.4f}  st_gain(vs best-ctrl) {e4d_acc - ctrl:+.4f}")
        print(f"  env-recon val R2ish | mlp {mlp_er:.4f} | RFF {rff_er:.4f} | Earth4D {e4d_er:.4f}   (aux env-field fit quality)")
        print(f"  held-out top5 acc   | mlp {mlp_t5:.4f} | RFF {rff_t5:.4f} | Earth4D {e4d_t5:.4f}")
        print(f"  [profile] earth4d_dim={fdim} held={n_te} steps={a.steps} env_decode=True aux_w={a.env_aux_weight}")
        print(f"  {len(lat)} obs, {a.steps}-step env-decode in {dt:.1f}s")
        return {"st_gain": e4d_acc - mlp_acc, "st_gain_bestctrl": e4d_acc - ctrl, "earth4d_acc": e4d_acc,
                "mlp_acc": mlp_acc, "rff_acc": rff_acc, "earth4d_envR2": e4d_er, "obs": len(lat),
                "seconds": dt, "env_decode": True}

    if a.field_decode:
        # science.md rule 24: TRAIN the encoder end-to-end to decode the dense family field between sparse obs,
        # then forecast the strict held-out (future+new-place) set. Fair controls under the identical decode:
        # a trainable coord-MLP (generic learned PE, matched capacity) and fixed-RFF+trainable-head.
        # st_gain = trained-Earth4D forecast MINUS the best generic learned control -> isolates whether the
        # 4D hash field learns propagatable field structure a plain learned coordinate map lacks.
        from deepearth.autoresearch.programs.spacetime.recurrence import run_field_decode
        with torch.no_grad():
            fdim = enc(coords[:8].to(dev)).shape[1]
        rn_fd = np.stack([lat / 90.0, lon / 180.0], 1).astype(np.float32)
        if a.forecast:
            rn_fd = np.concatenate([rn_fd, tnorm[:, None]], 1)
        else:
            rn_fd = np.concatenate([rn_fd, np.zeros((len(lat), 1), np.float32)], 1)
        e4d_acc, e4d_t5, n_te = run_field_decode("earth4d", coords, rn_fd, fam, test, n_fam, dev,
                                                 enc=enc, feat_dim=fdim, steps=a.steps, lr=a.lr, head_hidden=max(a.head_hidden, 256))
        mlp_acc, mlp_t5, _ = run_field_decode("mlp", coords, rn_fd, fam, test, n_fam, dev,
                                              feat_dim=fdim, steps=a.steps, lr=a.lr, head_hidden=max(a.head_hidden, 256))
        rff_acc, rff_t5, _ = run_field_decode("rff", coords, rn_fd, fam, test, n_fam, dev,
                                              feat_dim=fdim, steps=a.steps, lr=a.lr, head_hidden=max(a.head_hidden, 256))
        dt = time.time() - t0
        ctrl = max(mlp_acc, rff_acc)
        mode = "FIELD-DECODE(future+newplace)" if a.forecast_spatial else ("FIELD-DECODE(past->future)" if a.forecast else "FIELD-DECODE(spatial-block)")
        print(f"=== SPACETIME encoder (standalone) | mode={mode} obs={len(lat)} held-out={n_te} families={n_fam} ===")
        print(f"  held-out family acc | mlp {mlp_acc:.4f} | RFF {rff_acc:.4f} | Earth4D {e4d_acc:.4f}   st_gain(vs mlp) {e4d_acc - mlp_acc:+.4f}  st_gain(vs RFF) {e4d_acc - rff_acc:+.4f}  st_gain(vs best-ctrl) {e4d_acc - ctrl:+.4f}")
        print(f"  held-out top5 acc   | mlp {mlp_t5:.4f} | RFF {rff_t5:.4f} | Earth4D {e4d_t5:.4f}   top5_gain(vs mlp) {e4d_t5 - mlp_t5:+.4f}  top5_gain(vs RFF) {e4d_t5 - rff_t5:+.4f}")
        print(f"  [profile] earth4d_dim={fdim}  held={n_te}  spatial_levels={a.spatial_levels} steps={a.steps} field_decode=True trained_encoder=True")
        print(f"  {len(lat)} obs, {a.steps}-step decode in {dt:.1f}s")
        return {"st_gain": e4d_acc - mlp_acc, "st_gain_rff": e4d_acc - rff_acc, "st_gain_bestctrl": e4d_acc - ctrl,
                "earth4d_acc": e4d_acc, "mlp_acc": mlp_acc, "rff_acc": rff_acc, "obs": len(lat), "seconds": dt, "field_decode": True}

    with torch.no_grad():
        e4d = enc(coords.to(dev)).cpu()                          # [N, output_dim] Earth4D positional features
    rn = np.stack([lat / 90.0, lon / 180.0], 1).astype(np.float32)
    if a.forecast:
        rn = np.concatenate([rn, tnorm[:, None]], 1)             # fair: baselines get the SAME time feature
    raw = torch.tensor(rn)                                        # raw normalized coords (+time) baseline
    # Random Fourier Features of (lat,lon[,t]): fair nonlinear positional-encoding control vs Earth4D
    rff_rng = np.random.default_rng(0)
    proj = rn @ (rff_rng.normal(0, 8.0, (rn.shape[1], e4d.shape[1] // 2)).astype(np.float32))
    rff = torch.tensor(np.concatenate([np.sin(proj), np.cos(proj)], 1).astype(np.float32))

    if a.phenology:
        # DECISIVE non-stationary control (science.md rule 1+2b). Prior family-presence forecasting showed
        # propagator_gain ~0 because a STATIONARY spatial climatology fit the target. Here the target is the
        # DAY-OF-YEAR (phenology / seasonal timing) -- non-stationary: a static (lat,lon) map explains ~3% of
        # it, so a real propagator that carries WHEN nearby species were recently seen should finally win.
        # static no-propagation floor vs GNN vs LSTM, each over Earth4D / RFF / raw, on the SAME future+new-
        # place query set. propagator_gain = propagator MAE improvement over the static floor.
        assert a.forecast, "--phenology requires --forecast (needs live event-time + past->future split)"
        from deepearth.autoresearch.programs.spacetime.phenology import run_phenology_all
        coords_ll = torch.tensor(np.stack([lat, lon], 1).astype(np.float32))
        # CRITICAL leak-guard: the phenology TARGET is the query's own day-of-year, which is derivable from the
        # query timestamp. So the QUERY-POINT features here must be SPACE-ONLY (lat,lon) -- time stripped -- else
        # a static head reads the answer off its own time coordinate (smoke test: RFF+time -> MAE 1.3d, cheating).
        # Neighbours legitimately carry their OBSERVED past DOY as explicit node state (that IS the propagation).
        rn_sp = np.stack([lat / 90.0, lon / 180.0], 1).astype(np.float32)
        raw_sp = torch.tensor(rn_sp)
        _rng = np.random.default_rng(0)
        _proj = rn_sp @ (_rng.normal(0, 8.0, (2, e4d.shape[1] // 2)).astype(np.float32))
        rff_sp = torch.tensor(np.concatenate([np.sin(_proj), np.cos(_proj)], 1).astype(np.float32))
        coords_sp = torch.tensor(np.stack([lat, lon, np.zeros_like(lat), np.zeros_like(lat)], 1).astype(np.float32))  # t=0: no time leak
        with torch.no_grad():
            e4d_sp = enc(coords_sp.to(dev)).cpu()
        fd = {"e4d": e4d_sp.shape[1], "rff": rff_sp.shape[1], "raw": raw_sp.shape[1]}
        sp_all = None
        if a.pheno_species:
            import glob as _glob
            from pathlib import Path as _Path
            _sp = []
            for _f in sorted(_glob.glob(str(_Path(a.cache_dir) / "gbif_tokens/*.npz")))[:a.n_shards]:
                _sp.append(np.load(_f)["species_local"])
            sp_all = np.concatenate(_sp).astype(np.int64)
        r = run_phenology_all(e4d_sp, rff_sp, raw_sp, fd, days, coords_ll, test, dev,
                              K=a.rec_k, steps=a.steps, lr=a.lr, hidden=a.rec_hidden, hops=a.gnn_hops, tol_days=a.pheno_tol,
                              attn=a.pheno_attn, attn_heads=a.attn_heads, attn_layers=a.attn_layers, sp=sp_all)
        dt = time.time() - t0
        n_te = r["raw"]["n_te"]
        def pg(ft, prop):
            return r[ft]["static_mae"] - r[ft][prop + "_mae"], r[ft][prop + "_acc"] - r[ft]["static_acc"]
        pg_raw_gnn_mae, pg_raw_gnn_acc = pg("raw", "gnn")
        pg_raw_lstm_mae, pg_raw_lstm_acc = pg("raw", "lstm")
        pg_e4d_gnn_mae, _ = pg("e4d", "gnn")
        pg_rff_gnn_mae, _ = pg("rff", "gnn")
        best_prop_raw_mae = max(pg_raw_gnn_mae, pg_raw_lstm_mae)
        pg_raw_attn_mae = pg_raw_attn_acc = float("nan")
        if a.pheno_attn:
            pg_raw_attn_mae, pg_raw_attn_acc = pg("raw", "attn")
            best_prop_raw_mae = max(best_prop_raw_mae, pg_raw_attn_mae)
        pg_raw_sp_mae = pg_raw_sp_acc = float("nan")
        if a.pheno_species:
            pg_raw_sp_mae, pg_raw_sp_acc = pg("raw", "sp")
            best_prop_raw_mae = max(best_prop_raw_mae, pg_raw_sp_mae)
        print(f"=== SPACETIME encoder (standalone) | mode=PHENOLOGY(day-of-year, non-stationary, future+newplace) obs={len(lat)} forecast-queries={n_te} tol=+/-{a.pheno_tol:.0f}d K={a.rec_k} hops={a.gnn_hops} attn={a.pheno_attn} ===")
        for ft in ("raw", "rff", "e4d"):
            d = r[ft]
            attn_s = f" | ATTN MAE {d.get('attn_mae', float('nan')):6.2f}d acc {d.get('attn_acc', float('nan')):.4f} (prop {d['static_mae']-d.get('attn_mae', float('nan')):+.2f}d)" if a.pheno_attn else ""
            sp_s = f" | SP MAE {d.get('sp_mae', float('nan')):6.2f}d acc {d.get('sp_acc', float('nan')):.4f} (prop {d['static_mae']-d.get('sp_mae', float('nan')):+.2f}d)" if a.pheno_species else ""
            print(f"  {ft:>4} | static MAE {d['static_mae']:6.2f}d acc {d['static_acc']:.4f} -> GNN MAE {d['gnn_mae']:6.2f}d acc {d['gnn_acc']:.4f} (prop {d['static_mae']-d['gnn_mae']:+.2f}d) | LSTM MAE {d['lstm_mae']:6.2f}d acc {d['lstm_acc']:.4f} (prop {d['static_mae']-d['lstm_mae']:+.2f}d){attn_s}{sp_s}")
        print(f"  BEST propagator_gain (raw features, MAE reduction in days; POSITIVE=propagation helps) GNN {pg_raw_gnn_mae:+.2f}d  LSTM {pg_raw_lstm_mae:+.2f}d  ATTN {pg_raw_attn_mae:+.2f}d  SP {pg_raw_sp_mae:+.2f}d  best {best_prop_raw_mae:+.2f}d")
        print(f"  propagator_gain(within-tol acc, raw) GNN {pg_raw_gnn_acc:+.4f}  LSTM {pg_raw_lstm_acc:+.4f}  ATTN {pg_raw_attn_acc:+.4f}  SP {pg_raw_sp_acc:+.4f}")
        print(f"  ENCODER control (GNN MAE reduction vs static, per PE): raw {pg_raw_gnn_mae:+.2f}d | RFF {pg_rff_gnn_mae:+.2f}d | Earth4D {pg_e4d_gnn_mae:+.2f}d  (Earth4D-vs-raw GNN MAE {r['raw']['gnn_mae']-r['e4d']['gnn_mae']:+.2f}d: +=E4D better)")
        print(f"  [profile] forecast_queries={n_te} K={a.rec_k} hidden={a.rec_hidden} hops={a.gnn_hops} steps={a.steps}")
        print(f"  {len(lat)} obs, {a.steps}-step phenology in {dt:.1f}s")
        return {"static_mae_raw": r["raw"]["static_mae"], "gnn_mae_raw": r["raw"]["gnn_mae"], "lstm_mae_raw": r["raw"]["lstm_mae"],
                "attn_mae_raw": r["raw"].get("attn_mae", float("nan")), "sp_mae_raw": r["raw"].get("sp_mae", float("nan")),
                "propagator_gain_mae": best_prop_raw_mae, "propagator_gain_gnn_mae": pg_raw_gnn_mae, "propagator_gain_lstm_mae": pg_raw_lstm_mae,
                "propagator_gain_attn_mae": pg_raw_attn_mae, "propagator_gain_attn_acc": pg_raw_attn_acc,
                "propagator_gain_sp_mae": pg_raw_sp_mae, "propagator_gain_sp_acc": pg_raw_sp_acc,
                "propagator_gain_acc": pg_raw_gnn_acc, "propagator_gain_e4d_mae": pg_e4d_gnn_mae, "propagator_gain_rff_mae": pg_rff_gnn_mae,
                "static_acc_raw": r["raw"]["static_acc"], "gnn_acc_raw": r["raw"]["gnn_acc"], "lstm_acc_raw": r["raw"]["lstm_acc"],
                "obs": len(lat), "seconds": dt, "phenology": True, "n_te": n_te}

    if a.first_arrival or a.abundance:
        # GENERALITY test of the phenology unlock (LOOP-spacetime-nonstationary-phenology-dayofyear): does the
        # propagator carry large value on OTHER temporally-dynamic targets, and does Earth4D remain neutral?
        # SAME leak-guards as --phenology: query features SPACE-ONLY (t stripped), graph edges SPATIAL-only
        # (no dt-to-query). Neighbours carry their OWN observed state (past DOY / past activity).
        assert a.forecast, "--first_arrival/--abundance require --forecast (needs live event-time + past->future split)"
        rn_sp = np.stack([lat / 90.0, lon / 180.0], 1).astype(np.float32)
        raw_sp = torch.tensor(rn_sp)
        _rng = np.random.default_rng(0)
        _proj = rn_sp @ (_rng.normal(0, 8.0, (2, e4d.shape[1] // 2)).astype(np.float32))
        rff_sp = torch.tensor(np.concatenate([np.sin(_proj), np.cos(_proj)], 1).astype(np.float32))
        coords_sp = torch.tensor(np.stack([lat, lon, np.zeros_like(lat), np.zeros_like(lat)], 1).astype(np.float32))  # t=0: no time leak
        with torch.no_grad():
            e4d_sp = enc(coords_sp.to(dev)).cpu()
        fd = {"e4d": e4d_sp.shape[1], "rff": rff_sp.shape[1], "raw": raw_sp.shape[1]}
        coords_ll = torch.tensor(np.stack([lat, lon], 1).astype(np.float32))

        def _report(name, unit, r, tol_line):
            n_te = r["raw"]["n_te"]
            def pg(ft, prop):
                return r[ft]["static_mae"] - r[ft][prop + "_mae"]
            g_raw_gnn = pg("raw", "gnn"); g_raw_lstm = pg("raw", "lstm")
            g_e4d_gnn = pg("e4d", "gnn"); g_rff_gnn = pg("rff", "gnn")
            best = max(g_raw_gnn, g_raw_lstm)
            print(f"=== SPACETIME encoder (standalone) | mode={name}(dynamic, future+newplace) obs={len(lat)} forecast-queries={n_te} K={a.rec_k} hops={a.gnn_hops} {tol_line} ===")
            for ft in ("raw", "rff", "e4d"):
                d = r[ft]
                print(f"  {ft:>4} | static {unit} {d['static_mae']:7.3f} acc/R2 {d['static_acc']:+.4f} -> GNN {unit} {d['gnn_mae']:7.3f} acc/R2 {d['gnn_acc']:+.4f} (prop {d['static_mae']-d['gnn_mae']:+.3f}) | LSTM {unit} {d['lstm_mae']:7.3f} acc/R2 {d['lstm_acc']:+.4f} (prop {d['static_mae']-d['lstm_mae']:+.3f})")
            print(f"  BEST propagator_gain (raw features, {unit} reduction; POSITIVE=propagation helps) GNN {g_raw_gnn:+.3f}  LSTM {g_raw_lstm:+.3f}  best {best:+.3f}")
            print(f"  ENCODER control (GNN {unit} reduction vs static, per PE): raw {g_raw_gnn:+.3f} | RFF {g_rff_gnn:+.3f} | Earth4D {g_e4d_gnn:+.3f}  (Earth4D-vs-raw GNN {unit} {r['raw']['gnn_mae']-r['e4d']['gnn_mae']:+.3f}: +=E4D better)")
            print(f"  [profile] forecast_queries={n_te} K={a.rec_k} hidden={a.rec_hidden} hops={a.gnn_hops} steps={a.steps}")
            return {"target": name, "static_mae_raw": r["raw"]["static_mae"], "gnn_mae_raw": r["raw"]["gnn_mae"],
                    "lstm_mae_raw": r["raw"]["lstm_mae"], "propagator_gain_mae": best,
                    "propagator_gain_gnn_mae": g_raw_gnn, "propagator_gain_lstm_mae": g_raw_lstm,
                    "propagator_gain_e4d_mae": g_e4d_gnn, "propagator_gain_rff_mae": g_rff_gnn,
                    "static_acc_raw": r["raw"]["static_acc"], "gnn_acc_raw": r["raw"]["gnn_acc"],
                    "lstm_acc_raw": r["raw"]["lstm_acc"], "obs": len(lat), "n_te": n_te}

        if a.first_arrival:
            import glob as _glob
            from pathlib import Path as _Path
            _sp = []
            for _f in sorted(_glob.glob(str(_Path(a.cache_dir) / "gbif_tokens/*.npz")))[:a.n_shards]:
                _sp.append(np.load(_f)["species_local"])
            sp_all = np.concatenate(_sp).astype(np.int64)
            from deepearth.autoresearch.programs.spacetime.dyntargets import run_first_arrival_all
            r = run_first_arrival_all(e4d_sp, rff_sp, raw_sp, fd, days, coords_ll, sp_all, test, dev,
                                      K=a.rec_k, steps=a.steps, lr=a.lr, hidden=a.rec_hidden, hops=a.gnn_hops, tol_days=a.pheno_tol)
            res = _report("FIRST_ARRIVAL(onset-DOY)", "MAEd", r, f"tol=+/-{a.pheno_tol:.0f}d")
            dt = time.time() - t0
            print(f"  {len(lat)} obs, {a.steps}-step first-arrival in {dt:.1f}s")
            return res | {"seconds": dt, "first_arrival": True}

        if a.abundance:
            from deepearth.autoresearch.programs.spacetime.dyntargets import run_abundance_all
            r = run_abundance_all(e4d_sp, rff_sp, raw_sp, fd, days, coords_ll, test, dev,
                                  K=a.rec_k, steps=a.steps, lr=a.lr, hidden=a.rec_hidden, hops=a.gnn_hops, win=a.abund_win)
            res = _report("ABUNDANCE(log-count)", "MAE", r, f"win={a.abund_win:.0f}d")
            dt = time.time() - t0
            print(f"  {len(lat)} obs, {a.steps}-step abundance in {dt:.1f}s")
            return res | {"seconds": dt, "abundance": True}

    if a.gnn:
        # science.md rule 1+2b: GraphCast/GenCast-style message-passing propagator. Per held-out (future+new-
        # place) query, build a graph of its K strictly-earlier past observations with LEARNED space-time edges
        # (dlat,dlon,dt), run multi-hop message passing, decode the query -> family forecast. Report ABSOLUTE
        # forecast skill for: GNN propagator vs the no-propagation static head (same query set) vs -- for the
        # mechanism control -- the SAME GNN over Earth4D / RFF / raw node features. propagator-vs-none isolates
        # whether causal propagation forecasts forward at all; Earth4D-vs-RFF isolates mechanism vs encoder.
        assert a.forecast, "--gnn requires --forecast (needs live event-time + past->future split)"
        from deepearth.autoresearch.programs.spacetime.gnn import run_gnn
        coords_ll = torch.tensor(np.stack([lat, lon], 1).astype(np.float32))
        e4d_r = run_gnn(e4d, e4d.shape[1], fam, days, coords_ll, test, n_fam, dev,
                        K=a.rec_k, steps=a.steps, lr=a.lr, hidden=a.rec_hidden, hops=a.gnn_hops)
        rff_r = run_gnn(rff, rff.shape[1], fam, days, coords_ll, test, n_fam, dev,
                        K=a.rec_k, steps=a.steps, lr=a.lr, hidden=a.rec_hidden, hops=a.gnn_hops)
        raw_r = run_gnn(raw, raw.shape[1], fam, days, coords_ll, test, n_fam, dev,
                        K=a.rec_k, steps=a.steps, lr=a.lr, hidden=a.rec_hidden, hops=a.gnn_hops)
        dt = time.time() - t0
        n_te = e4d_r["n_te"]
        # propagator-vs-none PER feature type (apples-to-apples: each GNN vs its OWN static floor).
        pg_raw = raw_r["gnn_acc"] - raw_r["static_acc"]; pg_raw5 = raw_r["gnn_top5"] - raw_r["static_top5"]
        pg_e4d = e4d_r["gnn_acc"] - e4d_r["static_acc"]; pg_rff = rff_r["gnn_acc"] - rff_r["static_acc"]
        print(f"=== SPACETIME encoder (standalone) | mode=GNN(message-passing propagator, future+newplace) obs={len(lat)} forecast-queries={n_te} families={n_fam} K={a.rec_k} hops={a.gnn_hops} ===")
        print(f"  ABSOLUTE top1 | raw: static {raw_r['static_acc']:.4f} -> GNN {raw_r['gnn_acc']:.4f} (prop {pg_raw:+.4f}) | RFF: static {rff_r['static_acc']:.4f} -> GNN {rff_r['gnn_acc']:.4f} (prop {pg_rff:+.4f}) | E4D: static {e4d_r['static_acc']:.4f} -> GNN {e4d_r['gnn_acc']:.4f} (prop {pg_e4d:+.4f})")
        print(f"  ABSOLUTE top5 | raw: static {raw_r['static_top5']:.4f} -> GNN {raw_r['gnn_top5']:.4f} (prop {pg_raw5:+.4f}) | RFF: static {rff_r['static_top5']:.4f} -> GNN {rff_r['gnn_top5']:.4f} | E4D: static {e4d_r['static_top5']:.4f} -> GNN {e4d_r['gnn_top5']:.4f}")
        print(f"  BEST propagator_gain (raw features: GNN vs its no-prop floor) top1 {pg_raw:+.4f}  top5 {pg_raw5:+.4f}   (does causal propagation forecast forward?)")
        print(f"  st_gain(GNN Earth4D vs RFF) {e4d_r['gnn_acc'] - rff_r['gnn_acc']:+.4f}  (GNN Earth4D vs raw) {e4d_r['gnn_acc'] - raw_r['gnn_acc']:+.4f}   (mechanism vs encoder control)")
        print(f"  [profile] earth4d_dim={e4d.shape[1]} forecast_queries={n_te} K={a.rec_k} hidden={a.rec_hidden} hops={a.gnn_hops} steps={a.steps}")
        print(f"  {len(lat)} obs, {a.steps}-step GNN in {dt:.1f}s")
        return {"gnn_acc_raw": raw_r["gnn_acc"], "gnn_top5_raw": raw_r["gnn_top5"],
                "static_acc_raw": raw_r["static_acc"], "static_top5_raw": raw_r["static_top5"],
                "propagator_gain": pg_raw, "propagator_gain_top5": pg_raw5,
                "propagator_gain_e4d": pg_e4d, "propagator_gain_rff": pg_rff,
                "gnn_acc_e4d": e4d_r["gnn_acc"], "gnn_acc_rff": rff_r["gnn_acc"],
                "st_gain": e4d_r["gnn_acc"] - rff_r["gnn_acc"], "st_gain_raw": e4d_r["gnn_acc"] - raw_r["gnn_acc"],
                "obs": len(lat), "seconds": dt, "gnn": True, "n_te": n_te}

    if a.recurrence:
        # science.md rule 2b: physics-inspired 4D recurrence. Instead of a static per-point lookup head,
        # a causal LSTM rollout PROPAGATES local past state forward to each held-out (future+new-place) query.
        # Same rollout is run on Earth4D / raw / RFF features -> st_gain isolates whether Earth4D's 4D field
        # carries structure that PROPAGATES past->future, not just structure that indexes a cell.
        assert a.forecast, "--recurrence requires --forecast (needs live event-time + past->future split)"
        coords_ll = torch.tensor(np.stack([lat, lon], 1).astype(np.float32))
        if a.rec_time_cond:
            # rule24+2b: instead of feeding each neighbour its OWN static code, re-encode the QUERY cell
            # (lat_q,lon_q) FORWARD to each step's event day so the encoder's time axis carries state the LSTM
            # propagates. featurize(lat,lon,day) reproduces the exact Earth4D / raw / RFF normalizations.
            from deepearth.autoresearch.programs.spacetime.recurrence import run_recurrence_timecond
            def _tn(day_arr):
                return ((np.asarray(day_arr, dtype=np.float32) - tmin) / tspan).astype(np.float32)
            def feat_e4d(la, lo, dy):
                c = torch.tensor(np.stack([np.asarray(la, np.float32), np.asarray(lo, np.float32),
                                           np.zeros_like(la, np.float32), _tn(dy)], 1))
                with torch.no_grad():
                    return enc(c.to(dev)).cpu()
            def _rn3(la, lo, dy):
                return np.stack([np.asarray(la, np.float32) / 90.0, np.asarray(lo, np.float32) / 180.0, _tn(dy)], 1).astype(np.float32)
            def feat_raw(la, lo, dy):
                return torch.tensor(_rn3(la, lo, dy))
            _P = np.random.default_rng(0).normal(0, 8.0, (3, e4d.shape[1] // 2)).astype(np.float32)  # frozen RFF projection
            def feat_rff(la, lo, dy):
                p = _rn3(la, lo, dy) @ _P
                return torch.tensor(np.concatenate([np.sin(p), np.cos(p)], 1).astype(np.float32))
            raw_acc, raw_t5, n_te = run_recurrence_timecond(feat_raw, 3, fam, days, coords_ll, test, n_fam, dev,
                                                            K=a.rec_k, steps=a.steps, lr=a.lr, hidden=a.rec_hidden, tag="raw")
            rff_acc, rff_t5, _ = run_recurrence_timecond(feat_rff, e4d.shape[1], fam, days, coords_ll, test, n_fam, dev,
                                                         K=a.rec_k, steps=a.steps, lr=a.lr, hidden=a.rec_hidden, tag="rff")
            e4d_acc, e4d_t5, _ = run_recurrence_timecond(feat_e4d, e4d.shape[1], fam, days, coords_ll, test, n_fam, dev,
                                                         K=a.rec_k, steps=a.steps, lr=a.lr, hidden=a.rec_hidden, tag="earth4d")
            dt = time.time() - t0
            print(f"=== SPACETIME encoder (standalone) | mode=RECURRENCE-TIMECOND(query-cell forward-in-time) obs={len(lat)} rollout-queries={n_te} families={n_fam} K={a.rec_k} ===")
            print(f"  forecast family acc | raw-coords {raw_acc:.4f} | RFF {rff_acc:.4f} | Earth4D {e4d_acc:.4f}   st_gain(vs raw) {e4d_acc - raw_acc:+.4f}  st_gain(vs RFF) {e4d_acc - rff_acc:+.4f}")
            print(f"  forecast top5 acc   | raw-coords {raw_t5:.4f} | RFF {rff_t5:.4f} | Earth4D {e4d_t5:.4f}   top5_gain(vs raw) {e4d_t5 - raw_t5:+.4f}  top5_gain(vs RFF) {e4d_t5 - rff_t5:+.4f}")
            print(f"  [profile] earth4d_dim={e4d.shape[1]}  rollout_queries={n_te}  K={a.rec_k} hidden={a.rec_hidden} spatial_levels={a.spatial_levels} steps={a.steps} time_cond=True")
            print(f"  {len(lat)} obs, {a.steps}-step rollout in {dt:.1f}s")
            return {"st_gain": e4d_acc - raw_acc, "st_gain_rff": e4d_acc - rff_acc, "earth4d_acc": e4d_acc,
                    "raw_acc": raw_acc, "rff_acc": rff_acc, "obs": len(lat), "seconds": dt, "recurrence": True, "time_cond": True}
        from deepearth.autoresearch.programs.spacetime.recurrence import run_recurrence
        raw_acc, raw_t5, n_te = run_recurrence(raw, fam, days, coords_ll, test, n_fam, dev,
                                               K=a.rec_k, steps=a.steps, lr=a.lr, hidden=a.rec_hidden, tag="raw")
        rff_acc, rff_t5, _ = run_recurrence(rff, fam, days, coords_ll, test, n_fam, dev,
                                            K=a.rec_k, steps=a.steps, lr=a.lr, hidden=a.rec_hidden, tag="rff")
        e4d_acc, e4d_t5, _ = run_recurrence(e4d, fam, days, coords_ll, test, n_fam, dev,
                                            K=a.rec_k, steps=a.steps, lr=a.lr, hidden=a.rec_hidden, tag="earth4d")
        dt = time.time() - t0
        print(f"=== SPACETIME encoder (standalone) | mode=RECURRENCE(4D-LSTM rollout past->future) obs={len(lat)} rollout-queries={n_te} families={n_fam} K={a.rec_k} ===")
        print(f"  forecast family acc | raw-coords {raw_acc:.4f} | RFF {rff_acc:.4f} | Earth4D {e4d_acc:.4f}   st_gain(vs raw) {e4d_acc - raw_acc:+.4f}  st_gain(vs RFF) {e4d_acc - rff_acc:+.4f}")
        print(f"  forecast top5 acc   | raw-coords {raw_t5:.4f} | RFF {rff_t5:.4f} | Earth4D {e4d_t5:.4f}   top5_gain(vs raw) {e4d_t5 - raw_t5:+.4f}")
        print(f"  [profile] earth4d_dim={e4d.shape[1]}  rollout_queries={n_te}  K={a.rec_k} hidden={a.rec_hidden} spatial_levels={a.spatial_levels} steps={a.steps}")
        print(f"  {len(lat)} obs, {a.steps}-step rollout in {dt:.1f}s")
        return {"st_gain": e4d_acc - raw_acc, "st_gain_rff": e4d_acc - rff_acc, "earth4d_acc": e4d_acc,
                "raw_acc": raw_acc, "rff_acc": rff_acc, "obs": len(lat), "seconds": dt, "recurrence": True}

    raw_acc, raw_t5 = evaluate(raw, fam_t, test, n_fam, dev, a.steps, a.lr, "raw", a.head_hidden)
    rff_acc, rff_t5 = evaluate(rff, fam_t, test, n_fam, dev, a.steps, a.lr, "rff", a.head_hidden)
    e4d_acc, e4d_t5 = evaluate(e4d, fam_t, test, n_fam, dev, a.steps, a.lr, "earth4d", a.head_hidden)
    dt = time.time() - t0
    mode = ("FORECAST(future+newplace)" if a.forecast_spatial else "FORECAST(past->future)") if a.forecast else "spatial-block"
    print(f"=== SPACETIME encoder (standalone) | mode={mode} obs={len(lat)} held-out={int(test.sum())} families={n_fam} ===")
    print(f"  held-out family acc | raw-coords {raw_acc:.4f} | RFF {rff_acc:.4f} | Earth4D {e4d_acc:.4f}   st_gain(vs raw) {e4d_acc - raw_acc:+.4f}  st_gain(vs RFF) {e4d_acc - rff_acc:+.4f}")
    print(f"  held-out top5 acc   | raw-coords {raw_t5:.4f} | RFF {rff_t5:.4f} | Earth4D {e4d_t5:.4f}   top5_gain(vs raw) {e4d_t5 - raw_t5:+.4f}")
    print(f"  [profile] earth4d_dim={e4d.shape[1]}  frac_held={test.mean():.3f}  spatial_levels={a.spatial_levels} temporal_levels={a.temporal_levels} log2_hashmap={a.log2_hashmap} forecast={a.forecast}")
    print(f"  {len(lat)} obs, {a.steps}-step probe in {dt:.1f}s")
    return {"st_gain": e4d_acc - raw_acc, "st_gain_rff": e4d_acc - rff_acc, "earth4d_acc": e4d_acc, "raw_acc": raw_acc,
            "rff_acc": rff_acc, "obs": len(lat), "seconds": dt, "forecast": a.forecast}


if __name__ == "__main__":
    main()
