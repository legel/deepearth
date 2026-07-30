"""Standalone spacetime-encoder probe -- train + evaluate Earth4D IN ISOLATION.

No fusion model, no 790M backbone, no full benchmark suite -- just Earth4D + a linear head over a
subsample of observation coordinates. Measures the encoder's science (science.md rules 1-6, 24): does the
Earth4D positional field make space-time PREDICTIVE of biology at HELD-OUT locations (spatial generalization,
the SDM task B1/B5/B8)? Fast.

Objective (standalone `st_gain`): held-out-block family accuracy from Earth4D(coords) MINUS from raw
normalized coordinates. >0 ⟹ the multi-resolution positional encoder adds spatial-biology structure a raw
coordinate cannot. Reuses Earth4D unchanged (no core edit).

  python -m deepearth.autoresearch.programs.spacetime.probe --cache_dir data/deepcal --steps 800

FORECAST mode (--forecast, chronological discovery probe):
  Real event-time (gbif_eventtime.npz, joined by gbifID) is placed into Earth4D coord slot 3 (t), which the
  default path leaves at 0. The held-out set becomes the LATEST `holdout` fraction of observations BY TIME
  (train on the past, forecast the future) instead of held-out spatial blocks. The raw-coords and RFF
  baselines receive the identical normalized-time feature, so the st_gain isolates whether Earth4D's 4D
  multi-resolution field predicts later observations better than a plain space-time code. This is not
  autoregression unless a separate mechanism consumes observed past state and rolls it forward. Default path
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
PROBE_MODULE = "deepearth.autoresearch.programs.spacetime.probe"
# Must match agents/earth4d/trace.py PROTOCOL. Bump both when a change alters what a run MEASURES.
PROTOCOL_VERSION = "v2-leakfix"
_TRACE_AUTHORIZED = False
if __name__ == "__main__":
    import sys as _entry_sys
    from deepearth.autoresearch.programs.spacetime.recurrence import (
        require_recorded_entrypoint as _require_recorded,
    )

    _require_recorded(
        "probe.py",
        module=PROBE_MODULE,
        argv=_entry_sys.argv[1:],
    )
    _TRACE_AUTHORIZED = True

import argparse
import csv
import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepearth.encoders.spacetime.earth4d import Earth4D
from deepearth.autoresearch.programs.spacetime.recurrence import (
    DEFAULT_TIME_HORIZON,
    normalize_forecast_time,
    normalize_time_from_train,
    phenology_feature_set,
    phenology_mode,
    strict_spatiotemporal_masks,
    validate_dynamic_target_causality,
    require_recorded_entrypoint,
)


PHENO_RAW_REASON = (
    "this phenology direction runs on RAW spatial features only (Earth4D settled neutral here), "
    "so its numbers cannot speak to the encoder"
)

RAW_PE_REASON = (
    "this mode evaluates propagator architectures on RAW coordinate features only -- Earth4D is "
    "not in the comparison, so its numbers cannot speak to the encoder"
)

_RESULT_SINK = {"path": "", "capability": "", "protocol": "", "flags": "", "seed": None,
                "steps": None, "n_shards": None, "trained_encoder": False}


def _set_result_sink(path, capability, protocol, args):
    """Arm the result contract for this run. Called once, right after parse_args."""
    _RESULT_SINK.update({
        "path": path or "", "capability": capability or "", "protocol": protocol,
        "flags": " ".join(sys.argv[1:]), "seed": getattr(args, "seed", None),
        "steps": getattr(args, "steps", None), "n_shards": getattr(args, "n_shards", None),
        "trained_encoder": bool(getattr(args, "train_encoder", False)),
    })


def declare(capability, mode, metric, value, gains=None, baselines=None, split="",
            trained_encoder=None, diagnostic=False, diagnostic_reason="", **extras):
    """Declare WHAT this run measured, in the contract's terms.

    A mode calls this immediately before returning. Fields the run already knows (seed, steps, shard
    count, protocol, whether the encoder was trained) come from the armed sink rather than being
    re-derived, so they cannot drift from the actual invocation.

    `--capability` from the harness wins over the mode's natural default when both are present: the
    harness declared the objective, and any mismatch is the harness's to detect.

    `trained_encoder` defaults to the --train_encoder FLAG, but some modes (FIELD-DECODE, ENV-DECODE)
    train the encoder end-to-end unconditionally, so they pass it explicitly. Only the trained protocol
    can support a claim about learned hash state, so this field must describe what actually happened
    rather than what was requested.
    """
    from deepearth.autoresearch.programs.spacetime.probe_contract import Primary, ProbeResult

    result = ProbeResult(
        capability=_RESULT_SINK["capability"] or capability,
        mode=mode,
        primary=Primary(metric, float(value)),
        protocol=_RESULT_SINK["protocol"],
        split=split,
        n_shards=_RESULT_SINK["n_shards"],
        seed=_RESULT_SINK["seed"],
        steps=_RESULT_SINK["steps"],
        trained_encoder=(_RESULT_SINK["trained_encoder"] if trained_encoder is None
                         else bool(trained_encoder)),
        gains=dict(gains or {}),
        baselines=dict(baselines or {}),
        flags=_RESULT_SINK["flags"],
        extras=dict(extras),
        diagnostic=bool(diagnostic),
        diagnostic_reason=diagnostic_reason,
    ).validate()
    print(result.render(), flush=True)          # the ONE human-readable block, derived from the result
    if _RESULT_SINK["path"]:
        result.write(_RESULT_SINK["path"])
        print(f"[probe] result -> {_RESULT_SINK['path']}  identity={result.identity_digest()}",
              flush=True)
    return result


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
    return lat, lon, fam, n_fam, days, gids, sp


def load_species(cache: str, n_shards: int):
    """Per-observation species_local id aligned to the SAME shard subsample load_obs uses (for community /
    per-species breadth targets). Additive; used only by --breadth_target."""
    import glob as _g
    from pathlib import Path as _P
    sp = []
    for f in sorted(_g.glob(str(_P(cache) / "gbif_tokens/*.npz")))[:n_shards]:
        z = np.load(f)
        sp.append(z["species_local"])
    return np.concatenate(sp).astype(np.int64)


def load_env(cache: str, gid, channels: str = "wcsoil", fit_mask=None):
    """Join real ENVIRONMENT covariates to each observation by gbifID (science.md rule 24: environment field).

    Returns env [N, D] float32, standardized (zero-mean/unit-std per column over the covered obs), missing
    values imputed to 0 (= the column mean post-standardization). A per-column present-mask is folded in as
    the impute so absent env reads as neutral, never leaks NaN.

    DATA LEVER (`channels`) -- this used to be hard-wired to worldclim+soil+elev, so `--env_channels` had NO
    effect on the family_from_env path and every "channel swap" fed the identical 29 columns:
      wcsoil (default, unchanged) = 19 worldclim + 9 soil + 1 elev  = 29
      worldclim                   = 19 worldclim                    = 19
      alphaearth                  = 64 AlphaEarth satellite embed   = 64  (learned, NOT physical climate)
      all                         = wcsoil ++ alphaearth            = 93
      modis                       = 12 MODIS phenology bands        = 12  (per-obs SEASONAL greenness time-series)
      all+modis                   = wcsoil ++ alphaearth ++ modis   = 105
    MODIS is a genuinely different modality from AlphaEarth: a within-year greenness TRAJECTORY (vegetation
    seasonality at the observation) rather than a static annual scene embedding. It has only ever fed the
    phenology probe, never the env->biology path."""
    cachep = Path(cache)
    wc = np.load(cachep / "gbif_worldclim_tokens.npz")
    so = np.load(cachep / "gbif_soil_tokens.npz")
    el = np.load(cachep / "gbif_elev.npz")
    wcmap = dict(zip(wc["gbifID"].tolist(), wc["worldclim"]))
    somap = dict(zip(so["gbifID"].tolist(), so["soil"]))
    elmap = dict(zip(el["gbifID"].tolist(), el["elev"].tolist()))
    aemap = AE = None
    if channels in ("alphaearth", "all", "all+modis"):
        _ae = np.load(cachep / "gbif_alphaearth_tokens.npz")
        aemap = {int(g): i for i, g in enumerate(_ae["gbifID"])}; AE = _ae["ae"]
    phmap = PH = None
    if channels in ("modis", "all+modis"):
        _ph = np.load(cachep / "gbif_phenology_tokens.npz")
        phmap = {int(g): i for i, g in enumerate(_ph["gbifID"])}; PH = _ph["phenology"]
    n_ae = 0 if AE is None else AE.shape[1]
    n_ph = 0 if PH is None else PH.shape[1]
    _base = 0 if channels in ("alphaearth", "modis") else (19 if channels == "worldclim" else 29)
    D = _base + n_ae + n_ph
    env = np.full((len(gid), D), np.nan, np.float32)
    for i, g in enumerate(gid):
        g = int(g)
        o = 0
        if channels not in ("alphaearth", "modis"):
            if g in wcmap: env[i, :19] = wcmap[g]
            o = 19
            if channels != "worldclim":
                if g in somap: env[i, 19:28] = somap[g]
                if g in elmap: env[i, 28] = elmap[g]
                o = 29
        if aemap is not None:
            j = aemap.get(g)
            if j is not None: env[i, o:o + n_ae] = AE[j]
            o += n_ae
        if phmap is not None:
            j = phmap.get(g)
            if j is not None: env[i, o:o + n_ph] = PH[j]
    # Fit transforms on train only when a split mask is supplied.
    fit = env if fit_mask is None else env[np.asarray(fit_mask, dtype=bool)]
    mu = np.nanmean(fit, 0); sd = np.nanstd(fit, 0); sd[sd < 1e-6] = 1.0
    env = (env - mu) / sd
    env = np.nan_to_num(env, nan=0.0).astype(np.float32)
    return env


def load_vision(cache: str, gid, feat: str = "dino", n_shards: int = 999, fit_mask=None):
    """Per-obs PLANT vision (DINO/BioCLIP) joined by gbifID from gbif_tokens/ -- tests whether convergent
    MORPHOLOGY carries family where environment does not (perception law). Leak-free per-obs image emb,
    standardized; densify obs with zeroed vision impute to 0."""
    import glob as _glob
    cachep = Path(cache); dmap = {}
    for f in sorted(_glob.glob(str(cachep / "gbif_tokens/*.npz")))[:n_shards]:
        z = np.load(f); cols = []
        if feat in ("dino", "both"): cols.append(z["dino"])
        if feat in ("bio", "both"):  cols.append(z["bio"])
        V = np.concatenate(cols, 1).astype(np.float32)
        for g, v in zip(z["gbifID"].tolist(), V): dmap[int(g)] = v
    D = len(next(iter(dmap.values())))
    X = np.zeros((len(gid), D), np.float32)
    for i, g in enumerate(gid):
        v = dmap.get(int(g))
        if v is not None: X[i] = v
    fit = X if fit_mask is None else X[np.asarray(fit_mask, dtype=bool)]
    mu = fit.mean(0); sd = fit.std(0); sd[sd < 1e-6] = 1.0
    return ((X - mu) / sd).astype(np.float32)


def load_env_species(cache: str, extra_channels: bool = True, temporal: bool = False):
    """Species-aggregated ENVIRONMENT features for ENV->NICHE-TRAIT routing (Ensue ROUTING-soil-ph/MAP-*).

    Joins every observation to its worldclim(19) + AlphaEarth(64) [+ soil(9) + elev(1)] covariates by gbifID,
    then aggregates per species (species_local index, aligned to the 2141-species vocab used by traitprobe).
    Returns (envmean [S,D], envmedoid [S,D], n_per_species [S]). Aggregation levers exposed to the caller:
      mean   -- per-column mean over the species' observations (smooth, the route_map baseline)
      medoid -- the single observation whose feature vector is closest (L2) to the species mean (a real,
                non-averaged niche exemplar; robust to multi-modal / vagrant records)
    Columns are z-scored over covered species AFTER aggregation, missing imputed to 0 (= column mean)."""
    from pathlib import Path
    cachep = Path(cache)
    vocab = np.load(cachep / "gbif_vocab.npz", allow_pickle=True)
    S = len(vocab["global_idx"])
    wc = np.load(cachep / "gbif_worldclim_tokens.npz"); wcm = {int(g): i for i, g in enumerate(wc["gbifID"])}; WC = wc["worldclim"]
    ae = np.load(cachep / "gbif_alphaearth_tokens.npz"); aem = {int(g): i for i, g in enumerate(ae["gbifID"])}; AE = ae["ae"]
    id2day = {}
    if temporal:
        et = np.load(cachep / "gbif_eventtime.npz"); id2day = {int(g): float(d) for g, d in zip(et["gbifID"], et["days"])}
    doy_by_sp = [[] for _ in range(S)]  # per-species observed day-of-year radians (seasonal niche timing)
    if extra_channels:
        so = np.load(cachep / "gbif_soil_tokens.npz"); som = {int(g): i for i, g in enumerate(so["gbifID"])}; SO = so["soil"]
        el = np.load(cachep / "gbif_elev.npz"); elm = {int(g): float(v) for g, v in zip(el["gbifID"], el["elev"])}
        D = 19 + 64 + 9 + 1
    else:
        D = 19 + 64
    # First pass: per-obs feature rows grouped by species (for both mean and medoid).
    rows_by_sp = [[] for _ in range(S)]
    for f in sorted(glob.glob(str(cachep / "gbif_tokens/*.npz"))):
        z = np.load(f); sl = z["species_local"].astype(np.int64); gid = z["gbifID"]
        for s, g in zip(sl, gid):
            g = int(g)
            if g not in wcm or g not in aem:
                continue
            v = np.empty(D, np.float32)
            v[:19] = WC[wcm[g]]; v[19:83] = AE[aem[g]]
            if extra_channels:
                v[83:92] = SO[som[g]] if g in som else np.nan
                v[92] = elm.get(g, np.nan)
            rows_by_sp[int(s)].append(v)
            if temporal and g in id2day:
                doy_by_sp[int(s)].append((id2day[g] % 365.25) / 365.25 * 2.0 * np.pi)
    envmean = np.full((S, D), np.nan, np.float32)
    envmedoid = np.full((S, D), np.nan, np.float32)
    envstd = np.full((S, D), np.nan, np.float32)
    envlo = np.full((S, D), np.nan, np.float32)
    envhi = np.full((S, D), np.nan, np.float32)
    envmin = np.full((S, D), np.nan, np.float32)
    envmax = np.full((S, D), np.nan, np.float32)
    n = np.zeros(S, np.int64)
    for s in range(S):
        r = rows_by_sp[s]
        if not r:
            continue
        M = np.stack(r, 0)                                        # [k, D]
        n[s] = len(r)
        mu = np.nanmean(M, 0)
        envmean[s] = mu
        envstd[s] = np.nanstd(M, 0) if len(r) > 1 else 0.0
        envlo[s] = np.nanpercentile(M, 10, 0); envhi[s] = np.nanpercentile(M, 90, 0)
        envmin[s] = np.nanmin(M, 0); envmax[s] = np.nanmax(M, 0)  # explicit realized niche envelope (boundary extremes)
        # medoid = obs nearest (L2 over non-nan cols) to the species mean
        good = ~np.isnan(M).any(1)
        if good.sum() >= 1:
            Mg = M[good]
            d = np.linalg.norm(Mg - np.nan_to_num(mu), axis=1)
            envmedoid[s] = Mg[int(d.argmin())]
        else:
            envmedoid[s] = mu
    # per-species seasonal-timing features: circular mean sin/cos of observed DOY + resultant length R
    # (R in [0,1] = seasonal SPECIALIZATION; high R = tight active season, a temporal niche axis)
    envtime = np.full((S, 3), np.nan, np.float32)
    if temporal:
        for s in range(S):
            th = doy_by_sp[s]
            if not th:
                continue
            th = np.asarray(th, np.float64)
            c = np.cos(th).mean(); si = np.sin(th).mean()
            envtime[s] = (si, c, np.hypot(c, si))                 # mean-sin, mean-cos, resultant length R
    # per-species PHENOLOGICAL-BREADTH: season DURATION + multimodality (distinct from the unimodal mean+R above)
    #   col0 = active-month occupancy fraction (# distinct DOY-months with >=1 obs / 12) -> longer/broader season
    #   col1 = 2nd-harmonic resultant length R2 (bimodality; high when two activity peaks ~ multivoltine)
    envpheno = np.full((S, 2), np.nan, np.float32)
    if temporal:
        for s in range(S):
            th = doy_by_sp[s]
            if not th:
                continue
            th = np.asarray(th, np.float64)
            months = np.unique(np.floor((th % (2.0 * np.pi)) / (2.0 * np.pi) * 12.0).astype(np.int64))
            occ = len(months) / 12.0
            c2 = np.cos(2.0 * th).mean(); s2 = np.sin(2.0 * th).mean()
            envpheno[s] = (occ, np.hypot(c2, s2))
    # z-score per column over covered species, impute missing to 0
    def _z(X):
        m = np.nanmean(X, 0); sd = np.nanstd(X, 0); sd[sd < 1e-6] = 1.0
        return np.nan_to_num((X - m) / sd, nan=0.0).astype(np.float32)
    return _z(envmean), _z(envmedoid), n, _z(envstd), _z(envlo), _z(envhi), _z(envmin), _z(envmax), _z(envtime), _z(envpheno)


def load_env_obs(cache: str):
    """Per-OBSERVATION env rows (worldclim19+AlphaEarth64) with species_local, for per-obs niche-map training.

    Returns (Xobs [M,83] z-scored, sp_obs [M] species_local int64). Same columns/standardization convention as
    load_env_species' worldclim+alphaearth block, so a per-obs-trained head is comparable to the per-species one."""
    from pathlib import Path
    cachep = Path(cache)
    wc = np.load(cachep / "gbif_worldclim_tokens.npz"); wcm = {int(g): i for i, g in enumerate(wc["gbifID"])}; WC = wc["worldclim"]
    ae = np.load(cachep / "gbif_alphaearth_tokens.npz"); aem = {int(g): i for i, g in enumerate(ae["gbifID"])}; AE = ae["ae"]
    X, S = [], []
    for f in sorted(glob.glob(str(cachep / "gbif_tokens/*.npz"))):
        z = np.load(f); sl = z["species_local"].astype(np.int64); gid = z["gbifID"]
        for s, g in zip(sl, gid):
            g = int(g)
            if g in wcm and g in aem:
                X.append(np.concatenate([WC[wcm[g]], AE[aem[g]]]).astype(np.float32)); S.append(int(s))
    X = np.stack(X, 0); S = np.array(S, np.int64)
    m = np.nanmean(X, 0); sd = np.nanstd(X, 0); sd[sd < 1e-6] = 1.0
    X = np.nan_to_num((X - m) / sd, nan=0.0)
    return X.astype(np.float32), S


def _ridge_spearman(X, y, tr, te, alphas=(0.1, 1.0, 10.0, 100.0)):
    """RidgeCV fit on train species, Spearman rho on held-out species (route_map's linear baseline)."""
    from sklearn.linear_model import RidgeCV
    from scipy.stats import spearmanr
    r = RidgeCV(alphas=list(alphas)).fit(X[tr], y[tr])
    pr = r.predict(X[te])
    rho = spearmanr(y[te], pr).correlation
    return float(rho if rho == rho else 0.0)


def _mlp_spearman(X, y, tr, te, dev, hidden=128, steps=1500, lr=3e-3):
    """Nonlinear head: 1-hidden-layer MLP trained MSE on train species, Spearman rho on held-out species."""
    from scipy.stats import spearmanr
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    yt = torch.tensor(y, dtype=torch.float32, device=dev)
    ym, ys = yt[tr].mean(), yt[tr].std().clamp_min(1e-6)
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    tri = torch.tensor(tr, device=dev)
    for _ in range(steps):
        b = tri[torch.randint(0, len(tri), (min(512, len(tri)),), device=dev)]
        pred = net(Xt[b]).squeeze(-1)
        loss = F.mse_loss(pred, (yt[b] - ym) / ys)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pr = net(Xt[torch.tensor(te, device=dev)]).squeeze(-1).cpu().numpy()
    rho = spearmanr(y[te], pr).correlation
    return float(rho if rho == rho else 0.0)


def spatial_holdout(lat, lon, frac=0.2, block=0.5, seed=0):
    """Hold out whole 0.5-degree spatial blocks (tests generalization to UNSEEN locations, not memorization)."""
    lat, lon = np.asarray(lat), np.asarray(lon)
    if lat.ndim != 1 or lon.ndim != 1 or len(lat) != len(lon):
        raise ValueError("latitude and longitude must be aligned 1D arrays")
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError("spatial holdout coordinates must be finite")
    if not 0.0 < frac < 1.0 or not np.isfinite(block) or block <= 0:
        raise ValueError("holdout fraction and block size must be valid")
    blk = (np.floor(lat / block).astype(np.int64) * 100000 + np.floor(lon / block).astype(np.int64))
    ublk = np.unique(blk)
    rng = np.random.default_rng(seed); rng.shuffle(ublk)
    held = set(ublk[: int(len(ublk) * frac)].tolist())
    return np.array([b in held for b in blk])                  # bool [N_obs], True = held-out location


def temporal_holdout(days, frac=0.2):
    """Hold out the LATEST `frac` of observations by event time (train past -> forecast future).

    The split is chronological, but that alone does not make a pointwise model
    causal or autoregressive; those claims additionally require observed history
    and a rollout."""
    days = np.asarray(days)
    if days.ndim != 1 or not np.isfinite(days).all():
        raise ValueError("temporal holdout days must be a finite 1D array")
    if not 0.0 < frac < 1.0:
        raise ValueError("holdout fraction must be between zero and one")
    thr = np.quantile(days, 1.0 - frac)
    return days >= thr                                          # bool [N_obs], True = future (held out)


def strict_spatiotemporal_holdout(lat, lon, days, frac=0.2, block=0.5, seed=0):
    """Return disjoint train/test/embargo masks for a future-at-unseen-place test.

    ``~(future & held_place)`` is not a valid training mask: it admits future rows
    from seen places and past rows from held places.  A confirmatory partition
    trains only on past rows at seen places, tests only on future rows at held
    places, and excludes the two cross-quadrants.
    """
    future = temporal_holdout(days, frac)
    held_place = spatial_holdout(lat, lon, frac, block=block, seed=seed)
    return strict_spatiotemporal_masks(
        lat, lon, days, future, held_place, block=block
    )


def evaluate(feats, fam, test, n_fam, dev, steps, lr, tag, head_hidden=0, seed=0):
    """Train a linear head feats->family on TRAIN locations; report held-out-block accuracy."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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


def evaluate_trainable(enc, coords, fam, test, n_fam, dev, steps, lr, tag, head_hidden=0,
                       enc_lr_mult=0.05, warmup=0.15, c2f=0.5, clip=1.0, seed=0):
    """Train the ENCODER end-to-end with the head, instead of reading a frozen random hash table.

    Every other probe path calls enc(coords) under no_grad on a freshly-initialized Earth4D, so its hash table
    stays RANDOM: the reported fair-gains compare architectural priors as fixed random feature maps, not a
    trained encoder. For an architecture whose premise is a LEARNED table that is close to its worst case.

    A hash grid does not train stably by default here (a bare table on a Poisson objective returned +0.88 /
    +0.44 / +0.35 across seeds), so three standard stabilizers are on:
      * the encoder gets its OWN param group at lr*enc_lr_mult with no weight decay,
      * linear LR WARMUP over the first `warmup` fraction of steps (project memory: off-champion configs
        NaN/collapse without it),
      * COARSE-TO-FINE level unmasking -- fine hash levels are zeroed early and released over the first `c2f`
        fraction, the standard remedy for hash-grid overfitting/instability.
    Returns (acc, top5) with the SAME protocol as evaluate() so the numbers stay comparable."""
    torch.manual_seed(seed)
    train = ~test
    Ctr, ytr = coords[train].to(dev), fam[train].to(dev)
    Cte, yte = coords[test].to(dev), fam[test].to(dev)
    with torch.no_grad():
        fdim = enc(Ctr[:8]).shape[1]
    head = (nn.Sequential(nn.Linear(fdim, head_hidden), nn.ReLU(), nn.Linear(head_hidden, n_fam))
            if head_hidden > 0 else nn.Linear(fdim, n_fam)).to(dev)
    opt = torch.optim.Adam([{"params": head.parameters(), "lr": lr},
                            {"params": list(enc.parameters()), "lr": lr * enc_lr_mult, "weight_decay": 0.0}])
    # per-level feature mask over the SPATIAL block (levels are contiguous, features_per_level each)
    fpl = getattr(enc, "features_per_level", 2); sdim = getattr(enc, "spatial_dim", fdim)
    n_lv = max(int(sdim // max(fpl, 1)), 1)
    lvl_of = (torch.arange(fdim, device=dev) // max(fpl, 1)).clamp(max=n_lv - 1)
    warm_n, c2f_n = max(int(steps * warmup), 1), max(int(steps * c2f), 1)
    _p0 = {n: q.detach().clone() for n, q in enc.named_parameters()}   # sanity: did the encoder ACTUALLY move?
    for it in range(steps):
        for gi, base in enumerate((lr, lr * enc_lr_mult)):
            opt.param_groups[gi]["lr"] = base * min(1.0, (it + 1) / warm_n)      # linear warmup
        keep = n_lv if it >= c2f_n else max(1, int(n_lv * (it + 1) / c2f_n))     # coarse-to-fine
        idx = torch.randint(0, Ctr.shape[0], (4096,), device=dev)
        f = enc(Ctr[idx])
        if keep < n_lv:
            f = f * (lvl_of < keep).to(f.dtype)
        loss = F.cross_entropy(head(f), ytr[idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), clip)
        opt.step()
    with torch.no_grad():
        moved = {n: (q - _p0[n]).norm().item() / max(_p0[n].norm().item(), 1e-9) for n, q in enc.named_parameters()}
        tot = sum(1 for v in moved.values() if v > 1e-6)
        print(f"  [train_encoder] {tot}/{len(moved)} encoder tensors moved; "
              f"rel-delta " + ", ".join(f"{n.split('.')[-1]}={v:.3g}" for n, v in list(moved.items())[:6]), flush=True)
        logits = torch.cat([head(enc(Cte[i:i + 8192])) for i in range(0, Cte.shape[0], 8192)])
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
    ap.add_argument("--fourier", type=int, default=0)           # ARCH LEVER: add a random-Fourier-features branch of this width to Earth4D (0=off) -- tests hash+Fourier vs the pure-RFF baseline it currently loses to
    ap.add_argument("--fourier_scale", type=float, default=10.0)  # RFF bandwidth (freq scale) for the --fourier branch
    ap.add_argument("--time_harmonics", type=int, default=0)      # ARCH LEVER: internal learnable multi-scale sin/cos time basis (0=off) -- seasonal/persistence prior the discrete hash lacks; NOT redundant with spatial smooth_geo
    ap.add_argument("--train_encoder", action="store_true")   # TRAIN the encoder end-to-end instead of reading a frozen RANDOM hash table (every other path is frozen -- see evaluate_trainable)
    ap.add_argument("--enc_lr_mult", type=float, default=0.05)  # encoder lr = lr * this (own param group, wd=0)
    ap.add_argument("--enc_warmup", type=float, default=0.15)   # fraction of steps for linear LR warmup
    ap.add_argument("--enc_c2f", type=float, default=0.5)       # fraction of steps to fully unmask hash levels
    ap.add_argument("--target", default="family", choices=["family", "species"])  # CAPABILITY LEVER: classification target. The paths only ever predicted family (166-way); "species" switches to the 2141-way species vocab, which is what species_from_spacetime / species_from_env actually name
    ap.add_argument("--time_horizon", type=float, default=DEFAULT_TIME_HORIZON)   # train time compressed into [0,1/h] so the held-out FUTURE stays inside the encoder's representable range (it saturates past t~1.1). Design constant; never derived from test dates
    ap.add_argument("--causal_lags", type=int, default=0)         # ARCH LEVER: delayed positional basis (K learned backward coordinate reads; 0=off). It consumes no observed state, so it is not memory or autoregression
    ap.add_argument("--causal_lag_span", type=float, default=0.25)  # max lag as a fraction of the normalized time span
    ap.add_argument("--spatial_siren", type=int, default=0)      # ARCH LEVER: gated SIREN spatial branch (width; 0=off) -- sinusoidal-activation MLP over xyz, smooth+extrapolative BY CONSTRUCTION with LEARNED per-layer frequencies; aimed at the hash's held-out-spatial-block weakness (loses to a fixed RFF on static tasks)
    ap.add_argument("--siren_layers", type=int, default=2)
    ap.add_argument("--siren_w0", type=float, default=30.0)       # SIREN frequency scale (Sitzmann et al. default)
    ap.add_argument("--spatial_cline", type=int, default=0)      # ARCH LEVER: gated smooth spatial-CLINE band (0=off) -- linear xyz + LEARNABLE low-freq sin/cos; the monotone spatial gradient (lat->flowering-DOY Hopkins cline) the hash memorizes away and the fixed high-freq --fourier cannot form
    ap.add_argument("--cline_scale", type=float, default=1.0)     # init bandwidth of the learnable cline band (LOW by design; ~1 cycle over the domain)
    ap.add_argument("--time_film", type=int, default=0)        # ARCH LEVER: gated space x time FiLM (0=off) -- modulate spatial hash by a learned time basis; explicit seasonal-spatial interaction the additive features cannot form
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
    ap.add_argument("--rec_block_deg", type=float, default=2.0)  # ROUND-3: spatial neighbour-search block width (deg); widen with large K to feed more past-DOY samples
    ap.add_argument("--rec_fast", action="store_true")           # ROUND-4: vectorized cKDTree causal-window builder (true K-nearest, no block-ring cap, no per-query loop) -- decouples receptive-field breadth from O(block_area) cost so large K is affordable
    ap.add_argument("--pheno_nofair", action="store_true")   # opt OUT of auto-training the RFF fair control alongside Earth4D (default: always train it, so no phenology record is set without a fair baseline)
    ap.add_argument("--pheno_feats", default="e4d,rff,raw")  # HARD-RULE fast path: comma list subset of e4d,rff,raw to TRAIN (isolate ONE feature-type per run)
    # ---- LOOP-spacetime NEW DIRECTIONS on the mean-DOY graduation target (additive, default-off) ----
    ap.add_argument("--pheno_spatial", action="store_true")     # (1) SPATIAL generalization: query set = held-out 0.5deg BLOCKS (unseen geography), neighbours from train blocks; MAE-gain over static floor in new places
    ap.add_argument("--pheno_env", action="store_true")         # (2) ENV-conditioning: join per-obs worldclim(+soil+elev) as a propagator INPUT; neighbour-only vs neighbour+env vs env-only(static)
    ap.add_argument("--pheno_disttarget", default="")           # (3) distributional-timing target class: phase_centroid | peak_week | mean_doy (static vs GNN vs LSTM)
    ap.add_argument("--pheno_taxon", default="")                # (4) per-taxon breakdown of mean-DOY propagator gain: order | family (per-group static/LSTM MAE-gain)
    ap.add_argument("--pheno_densefield", action="store_true")  # rule-24 dense-field: mean-DOY at query cells whose OWN cell is EXCLUDED from the window (pure spatial interpolation from surrounding occupied cells); reports EMPTY-cell vs OCCUPIED-cell MAE-gain over static. leak-guard: query cell contributes nothing to itself.
    ap.add_argument("--densefield_drop", type=float, default=0.0)  # sparsity stress: drop this fraction of pool CELLS before interpolating (0.25/0.5/0.75) -- bounds where dense-field interpolation breaks.
    ap.add_argument("--densefield_block", type=float, default=0.5)  # cell size (deg) defining "same cell" for the exclusion + empty/occupied labelling.
    ap.add_argument("--first_arrival", action="store_true")     # dynamic target: per (0.5deg cell, species) EARLIEST DOY (seasonal onset, leading edge); static vs GNN vs LSTM over Earth4D/RFF/raw
    ap.add_argument("--abundance", action="store_true")         # dynamic target: log obs-count in query cell over trailing window (activity a static climatology cannot forecast); static vs GNN vs LSTM
    ap.add_argument("--abund_win", type=float, default=90.0)    # trailing window (days) for the abundance count
    ap.add_argument("--abund_lead", type=float, default=0.0)    # FORECAST-AHEAD horizon (days): target = activity in [d+lead-win, d+lead]; neighbours see only past<=d, so lead>0 = genuine lead-time forecast
    ap.add_argument("--abund_delta", action="store_true")       # DELTA-DYNAMICS target: future log-activity MINUS trailing-past log-activity (removes stationary seasonal mean; pure forward change)
    ap.add_argument("--field_decode", action="store_true")      # rule24: TRAIN the encoder end-to-end to decode the dense family field between sparse obs; fair control = trainable-head-on-RFF / coord-MLP
    ap.add_argument("--env", action="store_true")               # Move1: real ENVIRONMENT covariates (worldclim+soil+elev) vs coordinate-PE; + Earth4D+env fused
    ap.add_argument("--env_decode", action="store_true")        # Move2/rule24: TRAIN encoder to decode the smooth ENVIRONMENT field (aux), then predict biology from the learned field
    ap.add_argument("--env_aux_weight", type=float, default=1.0) # weight on the env-reconstruction auxiliary loss
    ap.add_argument("--env_trait", action="store_true")        # ROUTING (Ensue ROUTING-soil-ph/MAP-*): species-aggregated ENV->NICHE-TRAIT regression (held-out species). Predict num_soil_ph_max/rain_max/rain_min/elev_max/elev_min from per-species env (worldclim+AlphaEarth[+soil+elev]).
    ap.add_argument("--env_agg", default="mean", choices=["mean", "medoid"])  # per-species env aggregation lever: smooth column-mean vs a real non-averaged niche exemplar
    ap.add_argument("--env_extra", action="store_true")         # add soil(9)+elev(1) channels on top of worldclim(19)+AlphaEarth(64)
    ap.add_argument("--env_head", default="ridge", choices=["ridge", "mlp"])  # linear RidgeCV vs 1-hidden MLP niche head
    ap.add_argument("--env_mlp_hidden", type=int, default=128)
    ap.add_argument("--env_channels", default="all", choices=["all","worldclim","alphaearth","wcsoil","modis","all+modis"])  # ("wcsoil" = the legacy hard-wired 19wc+9soil+1elev stack the --env path used to force regardless of this flag) # channel-family ablation: which env source carries the niche-trait routing
    ap.add_argument("--vision", action="store_true")             # DATA LEVER: family from per-obs PLANT vision (DINO/BioCLIP) instead of env -- perception-law test
    ap.add_argument("--vision_feats", default="dino", choices=["dino","bio","both"])
    ap.add_argument("--pheno_channel", action="store_true")      # DATA LEVER: join per-obs MODIS phenology (gbif_phenology_tokens, 12 feats) onto the phenology forecaster query features
    ap.add_argument("--env_spread", action="store_true")
    ap.add_argument("--env_quantiles", action="store_true")      # concat per-species p10/p90 env (explicit distribution EDGES) on top of mean(+std) -- most direct match to max/min niche-boundary traits         # concat per-species column STD (niche breadth) to the mean -- directly informs max/min envelope traits
    ap.add_argument("--env_extremes", action="store_true")      # concat per-species column MIN/MAX env (realized niche ENVELOPE boundaries) -- the most literal match to num_*_max / num_*_min niche-boundary traits
    ap.add_argument("--env_extra_traits", action="store_true")  # extend the env-niche routing panel with num_soil_ph_min (the min counterpart already env-routed, 366 species) -- broader niche-trait coverage test
    ap.add_argument("--env_morph_traits", action="store_true")  # extend the routing panel with MORPHOLOGICAL traits (num_height_max 1768sp, num_width_max 420sp) -- tests whether env-niche routing carries non-climate (structural) traits, not just climate-envelope
    ap.add_argument("--env_biotic_trait", action="store_true")  # extend panel with num_lep_support (lepidopteran host-support, full 2141sp) -- a BIOTIC-interaction niche axis: does env routing carry biotic niche, not just abiotic climate/structure
    ap.add_argument("--env_trait_phylo", action="store_true")   # REROUTE VERDICT: alongside the env->trait Spearman, compute the PHYLO-SEED baseline (E1 text/tree species seed -> RidgeCV -> same held-out species split) for each numeric trait, and print the per-axis winner (env vs phylo). Isolates which encoder each numeric trait routes to.
    ap.add_argument("--env_temporal", action="store_true")     # concat per-species SEASONAL-TIMING features (circular mean sin/cos of observed DOY + resultant length R = seasonal specialization) -- a temporal niche axis on top of static env aggregates
    ap.add_argument("--env_phenobreadth", action="store_true")  # concat per-species PHENOLOGICAL-BREADTH temporal features (active-month occupancy fraction + 2nd-harmonic resultant R2 = bimodality/multivoltinism) -- a season-DURATION/multimodality axis (distinct from env_temporal mean+R) aimed at the biotic lep-support axis
    ap.add_argument("--env_perobs", action="store_true")        # train the linear niche map on PER-OBSERVATION env rows (train species only), predict held-out species from their species-mean env -- more training rows, same species holdout
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--abund_prop_arch", action="store_true")  # LOOP-spacetime: alt propagator ARCHITECTURES on the LEVEL-abundance target (rule-2b recurrence depth): single-LSTM vs deep-LSTM(2/3L) vs attention-over-neighbour-history; report ABSOLUTE R2 vs static floor. raw features only, one PE.
    ap.add_argument("--prop_arch", default="lstm1,lstm2,lstm3,attn")  # comma list subset of {lstm1,lstm2,lstm3,attn,mv} to run
    ap.add_argument("--prop_attn_heads", type=int, default=4)
    ap.add_argument("--prop_attn_layers", type=int, default=2)
    ap.add_argument("--abund_multivar", action="store_true")  # LOOP-spacetime: neighbour PAST state = joint [past-abundance || past-DOY(sin,cos) || past-occupancy] (rule24 dense-field cross-signal) forecasting abundance LEVEL; mv head in prop_arch
    ap.add_argument("--breadth_target", default="", choices=["", "occupancy", "richness", "community_activity"])
    ap.add_argument("--breadth_sub", type=float, default=30.0)
    # ---- LOOP-spacetime rule-1 AR ROLLOUT (this turn) ----------------------------------------------------
    ap.add_argument("--ar_rollout", action="store_true")        # rule-1 CAUSAL autoregressive rollout: predict one Delta-step ahead, FEED prediction back as the query's own current-state, roll forward to the final horizon; compare absR2 vs a single-shot DIRECT predictor at the SAME final horizon. Community-activity (default) or single-species abundance.
    ap.add_argument("--ar_target", default="community_activity", choices=["community_activity", "abundance", "richness"])  # AR rollout target: strong LEVEL signals only
    ap.add_argument("--ar_final", type=float, default=540.0)     # final forecast horizon (days) reached by the rollout AND by the direct baseline (matched)
    ap.add_argument("--ar_step", type=float, default=180.0)      # rollout step size Delta (days); n_steps = round(ar_final/ar_step). overlap regime keeps win = ar_step + 180 unless overridden
    ap.add_argument("--ar_cond_lead", action="store_true")      # PIVOT: continuous-lead conditioning -- feed the target lead as an input so ONE model spans all horizons; compare vs per-lead direct specialists
    ap.add_argument("--cooccur", action="store_true")            # CROSS-ENCODER ROUTING: predict per-species co-occurrence PARTNER-SET from ENV/SPACE (held-out species); micro-AP + gain over non-spatial prevalence baseline.
    ap.add_argument("--cooccur_mech", default="env", choices=["env","space","both"])
    ap.add_argument("--cooccur_thresh", type=int, default=2)
    ap.add_argument("--cooccur_file", default="cooccur_count_005.npy")
    ap.add_argument("--cooccur_channels", default="all", choices=["all","worldclim","alphaearth"])
    ap.add_argument("--sdm_presence", action="store_true")       # SDM env->biology (rules 1-6, B1/B5/B6/B8): predict which SPECIES occur at a held-out 0.5deg CELL from env+space.
    ap.add_argument("--sdm_hard", action="store_true")            # HARDENED SDM presence: finer grid + spatial-block CV -> many held-out cells; per-channel env decomp; optional seasonal-time feature.
    ap.add_argument("--sdm_cell_deg", type=float, default=0.1)     # grid cell size (deg); smaller => many more cells.
    ap.add_argument("--sdm_holdout_mode", default="block", choices=["block","random"])  # spatial-block CV vs random cell holdout.
    ap.add_argument("--sdm_block_deg", type=float, default=2.0)    # super-block width (deg) for spatial-block CV.
    ap.add_argument("--sdm_channels", default="all", choices=["all","worldclim","alphaearth","soil","elev"])  # per-channel env decomposition.
    ap.add_argument("--sdm_time", action="store_true")            # append per-cell seasonal timing (sin/cos mean-DOY + R).
    ap.add_argument("--sdm_seeds", type=int, default=1)            # run seeds seed..seed+n-1; report mean +/- std.
    # ---- LOOP-spacetime ENV-DERIVABLE CONSTRUCT test (rarity=range-size, ease=climate-breadth) ----
    ap.add_argument("--env_construct", action="store_true")
    ap.add_argument("--construct", default="rarity", choices=["rarity","ease","ns_grank","crpr"])
    ap.add_argument("--construct_feature", default="range", choices=["range","breadth","both","nichebreadth","nichebreadth_env","allbreadth"])
    ap.add_argument("--construct_shuffle", action="store_true")
    ap.add_argument("--construct_only", default="")
    # The result contract (probe_contract.py). --capability is what the harness DECLARED as its
    # objective; a mode supplies its own natural capability when the probe is run standalone. The
    # harness asserts the two agree, so a probe cannot quietly answer a different question.
    ap.add_argument("--result-json", dest="result_json", default="",
                    help="write a ProbeResult here; the harness reads this instead of parsing stdout")
    ap.add_argument("--capability", default="",
                    help="the capability the harness declared as its objective")
    a = ap.parse_args(argv)
    _set_result_sink(a.result_json, a.capability, PROTOCOL_VERSION, a)
    if not _TRACE_AUTHORIZED:
        authorization_argv = sys.argv[1:] if argv is None else list(argv)
        require_recorded_entrypoint(
            "probe.py",
            module=PROBE_MODULE,
            argv=authorization_argv,
        )
    validate_dynamic_target_causality(
        ar_rollout=a.ar_rollout,
        ar_cond_lead=a.ar_cond_lead,
        abundance=a.abundance,
        abund_prop_arch=a.abund_prop_arch,
        breadth_target=a.breadth_target,
        lead=a.abund_lead,
    )
    dev = a.device if torch.cuda.is_available() else "cpu"
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    if a.env_construct:
        r = env_construct(a.cache_dir, seed=a.seed, construct=a.construct,
                          feature=a.construct_feature, holdout=a.holdout, shuffle=a.construct_shuffle, only=a.construct_only)
        print(f"  n_labeled_used={r['n_labeled_used']} n_classes={r['n_classes']} held_out={r['held_out']}")
        print(f"  FLOOR acc {r['floor_acc']:.4f} bacc {r['floor_bacc']:.4f}  |  FEAT acc {r['acc']:.4f} bacc {r['bacc']:.4f}  |  Spearman(ord) {r['spearman_ord']:+.4f}")
        print(f"  univar Spearman: {r['univar_spearman']}")
        declare(
            capability="", mode=f"ENV-CONSTRUCT({r['construct']}<-{r['feature']})", metric="acc",
            value=r["acc"],
            diagnostic=True,
            diagnostic_reason=f"{r['construct']} is a species-level construct, not a scorecard capability",
            floor_acc=r["floor_acc"], floor_bacc=r["floor_bacc"], bacc=r["bacc"],
            spearman_ord=r["spearman_ord"], n_labeled_used=r["n_labeled_used"],
            n_classes=r["n_classes"], held_out=r["held_out"], shuffle_null=r["shuffle_null"],
        )
        return r

    if a.cooccur:
        import sys as _sys; _sys.path.insert(0, '/workspace')
        from deepearth.autoresearch.programs.spacetime.dyntargets import cooccur_routing
        r = cooccur_routing(a.cache_dir, thresh=a.cooccur_thresh, seed=a.seed,
                            mechanism=a.cooccur_mech, cooccur_file=a.cooccur_file,
                            env_channels=a.cooccur_channels)
        print(f"  query_sp={r['n_query_sp']} cand_sp={r['n_cand_sp']} feat_dim={r['feat_dim']} base_rate={r['micro_AP_baserate']:.4f}")
        print(f"  micro-AP(feat) {r['micro_AP_feat']:.4f} | micro-AP(prevalence-baseline) {r['micro_AP_prevalence']:.4f} | GAIN {r['gain_over_prevalence']:+.4f} | lift-over-baserate {r['lift_over_baserate']:.2f}x")
        print(f"  [leak-guard] {r['leak_guard']}")
        declare(
            capability="community_from_env",
            mode="COOCCUR-ROUTING",
            metric="micro_AP_feat",
            value=r["micro_AP_feat"],
            split=f"mech={r['mechanism']}",
            gains={"GAIN": r["gain_over_prevalence"]},
            baselines={"prevalence": r["micro_AP_prevalence"], "baserate": r["micro_AP_baserate"]},
            mechanism=r["mechanism"], thresh=r["thresh"], cooccur_file=r["cooccur_file"],
            n_query_sp=r["n_query_sp"], n_cand_sp=r["n_cand_sp"], feat_dim=r["feat_dim"],
            lift_over_baserate=r["lift_over_baserate"], leak_guard=r["leak_guard"],
        )
        return r

    if a.sdm_presence:
        import sys as _sys; _sys.path.insert(0, '/workspace')
        from deepearth.autoresearch.programs.spacetime.dyntargets import sdm_presence
        r = sdm_presence(a.cache_dir, seed=a.seed, mechanism=a.cooccur_mech, cooccur_file=a.cooccur_file)
        print(f"  query_cells={r['n_query_cells']} cand_sp={r['n_cand_sp']} feat_dim={r['feat_dim']} base_rate={r['micro_AP_baserate']:.4f}")
        print(f"  micro-AP(feat) {r['micro_AP_feat']:.4f} | micro-AP(prevalence-baseline) {r['micro_AP_prevalence']:.4f} | GAIN {r['gain_over_prevalence']:+.4f} | lift-over-baserate {r['lift_over_baserate']:.2f}x")
        print(f"  [leak-guard] {r['leak_guard']}")
        declare(
            capability="species_from_env",
            mode="SDM-PRESENCE",
            metric="micro_AP_feat",
            value=r["micro_AP_feat"],
            split=f"mech={r['mechanism']}",
            gains={"GAIN": r["gain_over_prevalence"]},
            baselines={"prevalence": r["micro_AP_prevalence"], "baserate": r["micro_AP_baserate"]},
            mechanism=r["mechanism"], n_query_cells=r["n_query_cells"], n_cand_sp=r["n_cand_sp"],
            feat_dim=r["feat_dim"], lift_over_baserate=r["lift_over_baserate"],
            leak_guard=r["leak_guard"],
        )
        return r

    if a.sdm_hard:
        import sys as _sys; _sys.path.insert(0, '/workspace')
        from deepearth.autoresearch.programs.spacetime.dyntargets import sdm_presence_hard
        import numpy as _np
        runs = []
        for sd in range(a.seed, a.seed + a.sdm_seeds):
            r = sdm_presence_hard(a.cache_dir, seed=sd, mechanism=a.cooccur_mech,
                                  cell_deg=a.sdm_cell_deg, holdout_mode=a.sdm_holdout_mode,
                                  block_deg=a.sdm_block_deg, env_channels=a.sdm_channels,
                                  add_time=a.sdm_time, cooccur_file=a.cooccur_file)
            runs.append(r)
        aps = _np.array([x['micro_AP_feat'] for x in runs])
        gns = _np.array([x['gain_over_prevalence'] for x in runs])
        r0 = runs[0]
        print(f"  query_cells={r0['n_query_cells']} train_cells={r0['n_train_cells']} cand_sp={r0['n_cand_sp']} "
              f"feat_dim={r0['feat_dim']} base_rate={r0['micro_AP_baserate']:.4f}")
        print(f"  micro-AP(feat) {aps.mean():.4f} +/- {aps.std():.4f} | prevalence {r0['micro_AP_prevalence']:.4f} | "
              f"GAIN {gns.mean():+.4f} +/- {gns.std():.4f} | lift {r0['lift_over_baserate']:.2f}x")
        print(f"  [leak-guard] {r0['leak_guard']}")
        declare(
            capability="species_from_env",
            mode="SDM-HARD",
            metric="micro_AP_feat",
            value=float(aps.mean()),
            split=f"{r0['holdout_mode']}({r0['block_deg']}deg)/grid{r0['cell_deg']}deg",
            gains={"GAIN": float(gns.mean())},
            baselines={"prevalence": r0["micro_AP_prevalence"], "baserate": r0["micro_AP_baserate"]},
            mechanism=r0["mechanism"], env_channels=r0["env_channels"], add_time=r0["add_time"],
            sdm_seeds=a.sdm_seeds, ap_std=float(aps.std()), gain_std=float(gns.std()),
            n_query_cells=r0["n_query_cells"], n_train_cells=r0["n_train_cells"],
            n_cand_sp=r0["n_cand_sp"], feat_dim=r0["feat_dim"], leak_guard=r0["leak_guard"],
        )
        return {'runs': runs, 'ap_mean': float(aps.mean()), 'ap_std': float(aps.std()),
                'gain_mean': float(gns.mean()), 'gain_std': float(gns.std())}

    t0 = time.time()
    need_gid = a.env or a.env_decode or a.pheno_env
    lat, lon, fam, n_fam, days, gid, sp_obs = load_obs(a.cache_dir, a.n_shards, with_time=a.forecast, with_gid=need_gid)
    obs_index = np.arange(len(lat), dtype=np.int64)
    if a.forecast:
        valid_time = np.isfinite(days)
        if not valid_time.all():
            lat, lon, fam, days, sp_obs = (
                x[valid_time] for x in (lat, lon, fam, days, sp_obs)
            )
            if gid is not None:
                gid = gid[valid_time]
            obs_index = obs_index[valid_time]
    if a.target == "species":
        # CAPABILITY LEVER: the classification paths only ever predicted FAMILY, which is why
        # species_from_spacetime / species_from_env were never probeable from this probe at all. Species is a
        # strictly harder target (2141-way vocab vs 166 families) and is the capability the scorecard names.
        _u, fam = np.unique(sp_obs, return_inverse=True)         # compact the species ids actually present
        fam = fam.astype(np.int64); n_fam = int(fam.max()) + 1
    if a.forecast:
        # Temporal holdout is a forecast split, but direct coordinate classification is
        # still a static forecast probe, not an autoregressive model.
        test = temporal_holdout(days, a.holdout)
        if a.pheno_spatial:
            # LOOP-spacetime (1) SPATIAL generalization: the mean-DOY graduation head must forecast timing in
            # UNSEEN geography, not just future time. Swap the query set to held-out 0.5deg spatial blocks;
            # neighbours are drawn from TRAIN (seen) blocks. Tests generalization to new places, not memorization.
            test = spatial_holdout(lat, lon, a.holdout, seed=a.seed)
        if a.forecast_spatial:
            # Keep only the two valid quadrants.  The old ``test=future&held; train=~test``
            # leaked future-seen-place and past-held-place rows into training.
            train, test, _embargo = strict_spatiotemporal_holdout(
                lat, lon, days, a.holdout, seed=a.seed
            )
            keep = train | test
            lat, lon, fam, days, sp_obs = (
                x[keep] for x in (lat, lon, fam, days, sp_obs)
            )
            if gid is not None:
                gid = gid[keep]
            obs_index = obs_index[keep]
            test = test[keep]  # complement is now exactly past + seen-place
        # Reserve room for the held-out future inside the encoder's [0,1] time grid. Without it the future
        # lands above t=1.0, where the hash grid SATURATES (t=1.2/1.5/2.0/3.0 all return identical features),
        # so every test row would be temporally indistinguishable and the forecast probe would lose its time
        # axis. A DESIGN constant -- no test date is consulted, so the train-only fit stays leak-free.
        # 1/(1-holdout) is NOT enough: temporal_holdout splits by row COUNT and observations are denser in the
        # past, so the last 20% of rows spans ~52% of the calendar range (measured test t max 1.52 at h=1.0).
        tnorm, tmin, tspan = normalize_time_from_train(days, ~test, horizon=a.time_horizon)
        coords = torch.tensor(np.stack([lat, lon, np.zeros_like(lat), tnorm], 1))  # [N,4]=(lat,lon,elev=0,t=REAL)
    else:
        test = spatial_holdout(lat, lon, a.holdout, seed=a.seed)
        coords = torch.tensor(np.stack([lat, lon, np.zeros_like(lat), np.zeros_like(lat)], 1))  # [N,4] t=0
    fam_t = torch.tensor(fam)

    enc = Earth4D(verbose=False, spatial_levels=a.spatial_levels, temporal_levels=a.temporal_levels,   # S3: exposed capacity
                  spatial_log2_hashmap_size=a.log2_hashmap, temporal_log2_hashmap_size=a.log2_hashmap, freq_log_scale_init=-2.5,
                  fourier_features=a.fourier, fourier_scale=a.fourier_scale,
                  time_harmonics=a.time_harmonics, time_film=a.time_film,
                  spatial_cline=a.spatial_cline, cline_scale=a.cline_scale,
                  spatial_siren=a.spatial_siren, siren_layers=a.siren_layers, siren_w0=a.siren_w0,
                  causal_lags=a.causal_lags, causal_lag_span=a.causal_lag_span).to(dev)   # RFF + temporal-harmonic + space x time FiLM (arch levers)

    # These flags only ever reached the env->TRAIT routing path. On the --env classification path they were
    # silently inert, so a "DATA lever" run changed nothing while still reporting a score -- which is how
    # family_from_env read data-limited for 53 runs. Fail loudly instead of lying.
    _inert = [n for n, v in (("--env_extra", a.env_extra), ("--env_temporal", a.env_temporal),
                             ("--env_perobs", a.env_perobs), ("--env_quantiles", a.env_quantiles),
                             ("--env_extremes", a.env_extremes), ("--env_spread", a.env_spread))
              if v and not a.env_trait]
    if _inert and (a.env or a.env_decode):
        raise SystemExit(f"[probe] {' '.join(_inert)} has NO effect on the --env/--env_decode path "
                         f"(it only applies to --env_trait). Use --env_channels to change what --env loads, "
                         f"or add --env_trait. Refusing to run a lever that would silently do nothing.")

    if a.env or a.env_decode:
        # science.md rules 1-6, 24 done RIGHT: the positional field should represent the ENVIRONMENT; biology
        # follows. Real env covariates (worldclim+soil+elev) joined by gbifID -> the science-aligned question.
        env = load_env(a.cache_dir, gid, channels=a.env_channels, fit_mask=~test)  # train-fit transform
        if a.vision:
            env = load_vision(a.cache_dir, gid, a.vision_feats, a.n_shards, fit_mask=~test)
        rn = np.stack([lat / 90.0, lon / 180.0], 1).astype(np.float32)
        if a.forecast:
            rn = np.concatenate([rn, tnorm[:, None]], 1)

    if a.env_trait:
        # ---- ENV->NICHE-TRAIT ROUTING (Ensue ROUTING-soil-ph-*/ROUTING-MAP-*): environmental-niche traits are ----
        # routed to THIS (spacetime/environment) encoder. Aggregate env per species, predict each numeric niche
        # trait on HELD-OUT species. Reference (worldclim+AlphaEarth, mean, ridge): soil_ph 0.59 rain_min 0.77
        # rain_max 0.76 elev_min 0.78 elev_max 0.52; vs phylo-graph ~0.1. Levers: --env_agg, --env_extra, --env_head.
        import sys as _sys
        _sys.path.insert(0, "/workspace")
        from deepearth.autoresearch.programs.biological.probe import load_trait as _load_trait
        vocab = np.load(Path(a.cache_dir) / "gbif_vocab.npz", allow_pickle=True)
        gidx = vocab["global_idx"]
        emean, emedoid, npsp, estd, elo, ehi, emin, emax, etime, epheno = load_env_species(a.cache_dir, extra_channels=a.env_extra, temporal=(a.env_temporal or a.env_phenobreadth))
        ENV = emedoid if a.env_agg == "medoid" else emean
        if a.env_spread:
            ENV = np.concatenate([ENV, estd], 1)                 # mean niche center ++ per-species breadth
        if a.env_quantiles:
            ENV = np.concatenate([ENV, elo, ehi], 1)             # ++ per-species p10/p90 distribution edges
        if a.env_extremes:
            ENV = np.concatenate([ENV, emin, emax], 1)           # ++ per-species realized min/max envelope (literal niche boundaries)
        if a.env_temporal:
            ENV = np.concatenate([ENV, etime], 1)                # ++ per-species seasonal-timing (DOY circular mean + specialization R)
        if a.env_phenobreadth:
            ENV = np.concatenate([ENV, epheno], 1)               # ++ per-species phenological-breadth (active-month occupancy + 2nd-harmonic R2 bimodality)
        if a.env_channels == "worldclim":
            ENV = ENV[:, :19]                                     # physical climate only (WorldClim 19-band)
        elif a.env_channels == "alphaearth":
            ENV = ENV[:, 19:83]                                   # learned satellite embedding only (AlphaEarth 64d)
        _XOBS = _SPOBS = None
        if a.env_perobs:
            _XOBS, _SPOBS = load_env_obs(a.cache_dir)
            if a.env_channels == "worldclim": _XOBS = _XOBS[:, :19]
            elif a.env_channels == "alphaearth": _XOBS = _XOBS[:, 19:83]
        keys = ["num_soil_ph_max", "num_rain_max", "num_rain_min", "num_elev_max", "num_elev_min"]
        if a.env_extra_traits:
            keys = keys + ["num_soil_ph_min"]
        if a.env_morph_traits:
            keys = keys + ["num_height_max", "num_width_max"]
        if a.env_biotic_trait:
            keys = keys + ["num_lep_support"]
        PHY = None
        if a.env_trait_phylo:
            # REROUTE VERDICT: load the PHYLO/text species seed (E1, the SAME BioCLIP text prior the biological
            # graph refines) aligned to the 2141-vocab, and predict each trait from it via the identical RidgeCV
            # + same held-out species split as the env side. Fair head-for-head: only the FEATURE source differs
            # (env aggregates vs phylo seed), so the winner is the honest per-axis routing verdict.
            from deepearth.autoresearch.programs.biological.probe import load_species as _load_species
            E1, _famid, _tree, _tiprow, _gidxb = _load_species(a.cache_dir)
            PHY = np.asarray(E1.detach().cpu()).astype(np.float32)   # [2141, seed_dim] text/tree species seed
        dt0 = time.time()
        results = {}
        rows = []
        for key in keys:
            _, Y, obs, _ = _load_trait(a.cache_dir, gidx, key, "cpu")
            Y = np.asarray(Y).astype(np.float32); obs = np.asarray(obs).astype(bool)
            sp = obs & (npsp > 0) & np.isfinite(ENV).all(1)
            idx = np.where(sp)[0]
            if len(idx) < 20:
                rows.append((key, float("nan"), len(idx))); results[key] = float("nan"); continue
            rng = np.random.default_rng(a.seed); rng.shuffle(idx)
            cut = len(idx) // 5
            te, tr = idx[:cut], idx[cut:]
            if a.env_perobs:
                # train linear niche map on PER-OBS env rows of the TRAIN species; predict held-out species from species-mean ENV
                from sklearn.linear_model import RidgeCV as _RCV
                from scipy.stats import spearmanr as _sr
                tr_set = set(tr.tolist())
                mo = np.array([sp0 in tr_set for sp0 in _SPOBS]) & np.isfinite(Y[_SPOBS]) & np.isfinite(_XOBS).all(1)
                r = _RCV(alphas=[0.1,1.0,10.0,100.0]).fit(_XOBS[mo], Y[_SPOBS[mo]])
                pr = r.predict(ENV[te]); _rho = _sr(Y[te], pr).correlation
                rho = float(_rho if _rho==_rho else 0.0)
            elif a.env_head == "mlp":
                rho = _mlp_spearman(ENV, Y, tr, te, dev, hidden=a.env_mlp_hidden, steps=a.steps)
            else:
                rho = _ridge_spearman(ENV, Y, tr, te)
            phy_rho = float("nan")
            if PHY is not None:
                # phylo baseline on the SAME te/tr species split; drop any species with a non-finite seed row
                fin = np.isfinite(PHY).all(1)
                tr_p = tr[fin[tr]]; te_p = te[fin[te]]
                if len(tr_p) >= 5 and len(te_p) >= 5:
                    phy_rho = _ridge_spearman(PHY, Y, tr_p, te_p)
                results[key + "_phylo"] = phy_rho
            rows.append((key, rho, len(te), phy_rho)); results[key] = rho
        dt = time.time() - dt0
        vals = [r[1] for r in rows if r[1] == r[1]]
        mean_rho = float(np.mean(vals)) if vals else float("nan")
        if PHY is not None:
            phyvals = [r[3] for r in rows if r[3] == r[3]]
            phy_mean = float(np.mean(phyvals)) if phyvals else float("nan")
            print(f"  {'axis':18s} {'env':>8s} {'phylo':>8s} {'winner':>7s}  (n)   phylo_seed_dim={PHY.shape[1]}")
            for key, rho, nte, prho in rows:
                win = "ENV" if (rho == rho and (prho != prho or rho >= prho)) else "PHYLO"
                d = (rho - prho) if (rho == rho and prho == prho) else float("nan")
                print(f"  {key:18s} {rho:+8.3f} {prho:+8.3f} {win:>7s}  n={nte}  env-phylo={d:+.3f}")
            print(f"  mean_over_traits  env {mean_rho:+.4f}  phylo {phy_mean:+.4f}  ({dt:.1f}s)")
            results["phylo_mean_spearman"] = phy_mean
        else:
            for key, rho, nte, prho in rows:
                print(f"  {key:18s} spearman {rho:+.3f}   held-out_species_n={nte}")
            print(f"  mean_spearman_over_traits {mean_rho:+.4f}   ({dt:.1f}s)")
        results["mean_spearman"] = mean_rho
        results["env_dim"] = int(ENV.shape[1]); results["agg"] = a.env_agg
        results["head"] = a.env_head; results["extra"] = bool(a.env_extra); results["seconds"] = dt
        declare(
            capability="", mode=f"ENV->NICHE-TRAIT(agg={a.env_agg})", metric="mean_spearman",
            value=mean_rho,
            diagnostic=True,
            diagnostic_reason="trait Spearman over species aggregates is not a scorecard capability",
            env_dim=int(ENV.shape[1]), agg=a.env_agg, head=a.env_head, extra=bool(a.env_extra),
            seconds=dt,
        )
        return results

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
        raw_acc, raw_t5 = evaluate(raw, fam_t, test, n_fam, dev, a.steps, a.lr, "raw", a.head_hidden, a.seed)
        rff_acc, rff_t5 = evaluate(rff, fam_t, test, n_fam, dev, a.steps, a.lr, "rff", a.head_hidden, a.seed)
        e4d_acc, e4d_t5 = (evaluate_trainable(enc, coords, fam_t, test, n_fam, dev, a.steps, a.lr, "earth4d",
                                              a.head_hidden, a.enc_lr_mult, a.enc_warmup, a.enc_c2f, seed=a.seed)
                           if a.train_encoder else
                           evaluate(e4d, fam_t, test, n_fam, dev, a.steps, a.lr, "earth4d", a.head_hidden, a.seed))
        env_acc, env_t5 = evaluate(env_t, fam_t, test, n_fam, dev, a.steps, a.lr, "env", a.head_hidden, a.seed)
        fus_acc, fus_t5 = evaluate(fused, fam_t, test, n_fam, dev, a.steps, a.lr, "fused", a.head_hidden, a.seed)
        dt = time.time() - t0
        best_coord = max(raw_acc, rff_acc, e4d_acc)              # best coordinate-only PE
        mode = ("FORECAST(future+newplace)" if a.forecast_spatial else "FORECAST(past->future)") if a.forecast else "spatial-block"
        print(f"  held-out family acc | raw {raw_acc:.4f} | RFF {rff_acc:.4f} | Earth4D {e4d_acc:.4f} || ENV {env_acc:.4f} | Earth4D+ENV {fus_acc:.4f}")
        # The record's primary for family_from_env is the FUSED Earth4D+ENV accuracy; the old harness
        # recovered it by matching r"Earth4D\+ENV\s+([\d.]+)" against the first line that happened to
        # contain it, which is the top1 row only because top1 prints before top5.
        declare(
            capability="family_from_env",
            mode=f"ENV({mode})",
            metric="family_top1_accuracy",
            value=fus_acc,
            split=mode,
            # "ENV vs best-coord-PE" is the CHANNEL's advantage over coordinates, not the encoder's.
            # Without an explicit Earth4D-vs-generic-PE entry the harness's fair-baseline preference
            # matched "best-coord" and read +0.0411 as an encoder gain -- diagnosing ENCODER-LIMITED
            # when Earth4D alone (0.0938) actually LOSES to RFF (0.1010) and the true read is
            # INPUT-LIMITED. The encoder-vs-PE gain has to be stated for the diagnosis to be right.
            gains={"Earth4D vs RFF": e4d_acc - rff_acc,
                   "ENV vs best-coord-PE": env_acc - best_coord,
                   "fused vs best-coord-PE": fus_acc - best_coord},
            baselines={"raw": raw_acc, "RFF": rff_acc, "earth4d": e4d_acc, "env": env_acc,
                       "best-coord-PE": best_coord},
            obs=len(lat), held_out=int(test.sum()), families=n_fam, env_dim=int(env.shape[1]),
            earth4d_dim=int(e4d.shape[1]), seconds=dt,
            top5={"raw": raw_t5, "rff": rff_t5, "earth4d": e4d_t5, "env": env_t5, "fused": fus_t5},
        )
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
        print(f"  env-recon val R2ish | mlp {mlp_er:.4f} | RFF {rff_er:.4f} | Earth4D {e4d_er:.4f}   (aux env-field fit quality)")
        print(f"  {len(lat)} obs, {a.steps}-step env-decode in {dt:.1f}s")
        declare(
            capability="family_from_env",
            mode=f"ENV-DECODE({mode})",
            metric="family_top1_accuracy",
            trained_encoder=True,          # trains the encoder against an auxiliary env field
            value=e4d_acc,
            split=mode,
            gains={"vs mlp": e4d_acc - mlp_acc, "vs best-ctrl": e4d_acc - ctrl},
            baselines={"mlp": mlp_acc, "RFF": rff_acc, "best-ctrl": ctrl},
            obs=len(lat), held_out=n_te, earth4d_dim=fdim, seconds=dt,
            env_aux_weight=a.env_aux_weight,
            env_recon_r2={"mlp": mlp_er, "rff": rff_er, "earth4d": e4d_er},
            top5={"mlp": mlp_t5, "rff": rff_t5, "earth4d": e4d_t5},
        )
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
        print(f"  {len(lat)} obs, {a.steps}-step decode in {dt:.1f}s")
        declare(
            capability="family_from_spacetime",
            mode=mode,                     # already reads "FIELD-DECODE(...)"; do not wrap it again
            metric="family_top1_accuracy",
            trained_encoder=True,          # run_field_decode trains end-to-end, flag or no flag
            value=e4d_acc,
            split=mode,
            gains={"vs mlp": e4d_acc - mlp_acc, "vs RFF": e4d_acc - rff_acc,
                   "vs best-ctrl": e4d_acc - ctrl},
            baselines={"mlp": mlp_acc, "RFF": rff_acc, "best-ctrl": ctrl},
            obs=len(lat), held_out=n_te, families=n_fam, earth4d_dim=fdim, seconds=dt,
            top5={"mlp": mlp_t5, "rff": rff_t5, "earth4d": e4d_t5},
        )
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
        # static no-propagation floor vs GNN vs LSTM, each over Earth4D / RFF / raw, on the declared split.
        # propagator_gain = propagator MAE improvement over the static floor.
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
        if a.pheno_channel:
            _ph = np.load(Path(a.cache_dir) / "gbif_phenology_tokens.npz")
            _pm = {int(g): i for i, g in enumerate(_ph["gbifID"])}; _PH = _ph["phenology"]
            import glob as _g2
            _gg=[np.load(_f)["gbifID"] for _f in sorted(_g2.glob(str(Path(a.cache_dir)/"gbif_tokens/*.npz")))[:a.n_shards]]
            _gid=np.concatenate(_gg).astype(np.int64)[obs_index]
            _px = np.zeros((len(lat), _PH.shape[1]), np.float32)
            for _i, _g in enumerate(_gid):
                _j = _pm.get(int(_g))
                if _j is not None: _px[_i] = _PH[_j]
            _fit = _px[~test]
            _mu = _fit.mean(0); _sd = _fit.std(0); _sd[_sd < 1e-6] = 1.0
            _pt = torch.tensor(((_px - _mu) / _sd).astype(np.float32))
            raw_sp = torch.cat([raw_sp, _pt], 1); e4d_sp = torch.cat([e4d_sp, _pt], 1)
            fd = {"e4d": e4d_sp.shape[1], "rff": rff_sp.shape[1], "raw": raw_sp.shape[1]}
        sp_all = None
        if a.pheno_species:
            import glob as _glob
            from pathlib import Path as _Path
            _sp = []
            for _f in sorted(_glob.glob(str(_Path(a.cache_dir) / "gbif_tokens/*.npz")))[:a.n_shards]:
                _sp.append(np.load(_f)["species_local"])
            sp_all = np.concatenate(_sp).astype(np.int64)[obs_index]
        _feats = phenology_feature_set(a.pheno_feats, a.pheno_nofair)
        # FAIR-BASELINE GUARD: a single-feature run (e.g. --pheno_feats e4d) left the RFF control untrained, so the
        # trace could report NO fair gain at all and still set a record -- this capability's records were being
        # gated on nothing. Whenever Earth4D is trained, train raw and generic-PE controls too
        # (opt out: --pheno_nofair).
        r = run_phenology_all(e4d_sp, rff_sp, raw_sp, fd, days, coords_ll, test, dev,
                              K=a.rec_k, steps=a.steps, lr=a.lr, hidden=a.rec_hidden, hops=a.gnn_hops, tol_days=a.pheno_tol,
                              attn=a.pheno_attn, attn_heads=a.attn_heads, attn_layers=a.attn_layers, sp=sp_all,
                              block_deg=a.rec_block_deg, fast=a.rec_fast,
                              feats=_feats)
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
        pheno_mode = phenology_mode(a.forecast_spatial, a.pheno_spatial)
        for ft in ("raw", "rff", "e4d"):
            d = r[ft]
            attn_s = f" | ATTN MAE {d.get('attn_mae', float('nan')):6.2f}d acc {d.get('attn_acc', float('nan')):.4f} (prop {d['static_mae']-d.get('attn_mae', float('nan')):+.2f}d)" if a.pheno_attn else ""
            sp_s = f" | SP MAE {d.get('sp_mae', float('nan')):6.2f}d acc {d.get('sp_acc', float('nan')):.4f} (prop {d['static_mae']-d.get('sp_mae', float('nan')):+.2f}d)" if a.pheno_species else ""
            print(f"  {ft:>4} | static MAE {d['static_mae']:6.2f}d acc {d['static_acc']:.4f} -> GNN MAE {d['gnn_mae']:6.2f}d acc {d['gnn_acc']:.4f} (prop {d['static_mae']-d['gnn_mae']:+.2f}d) | LSTM MAE {d['lstm_mae']:6.2f}d acc {d['lstm_acc']:.4f} (prop {d['static_mae']-d['lstm_mae']:+.2f}d){attn_s}{sp_s}")
        print(f"  BEST propagator_gain (raw features, MAE reduction in days; POSITIVE=propagation helps) GNN {pg_raw_gnn_mae:+.2f}d  LSTM {pg_raw_lstm_mae:+.2f}d  ATTN {pg_raw_attn_mae:+.2f}d  SP {pg_raw_sp_mae:+.2f}d  best {best_prop_raw_mae:+.2f}d")
        print(f"  propagator_gain(within-tol acc, raw) GNN {pg_raw_gnn_acc:+.4f}  LSTM {pg_raw_lstm_acc:+.4f}  ATTN {pg_raw_attn_acc:+.4f}  SP {pg_raw_sp_acc:+.4f}")
        print(f"  ENCODER control (GNN MAE reduction vs static, per PE): raw {pg_raw_gnn_mae:+.2f}d | RFF {pg_rff_gnn_mae:+.2f}d | Earth4D {pg_e4d_gnn_mae:+.2f}d  (Earth4D-vs-raw GNN MAE {r['raw']['gnn_mae']-r['e4d']['gnn_mae']:+.2f}d: +=E4D better)")
        # THE fair gain for this capability: Earth4D's best head vs the GENERIC TRAINED PE's best head, on the
        # native within-tol acc. What used to be reported as the "fair gain" here was propagator_gain (propagation
        # vs static, on RAW features) -- a propagator quantity, not an encoder-vs-PE one, so it never gated the
        # encoder at all. Printed in the st_gain(...) form the trace's fair-baseline parser prefers.
        def _best_acc(ft):
            d = r[ft]
            vs = [d.get(k) for k in ("static_acc", "gnn_acc", "lstm_acc", "attn_acc", "sp_acc")]
            vs = [v for v in vs if v is not None and v == v]
            return max(vs) if vs else float("nan")
        _e4d_best, _rff_best = _best_acc("e4d"), _best_acc("rff")
        if _e4d_best == _e4d_best and _rff_best == _rff_best:
            print(f"  st_gain(Earth4D vs RFF, best-head within-tol acc) {_e4d_best - _rff_best:+.4f}   (Earth4D {_e4d_best:.4f}  RFF {_rff_best:.4f})")
        print(f"  {len(lat)} obs, {a.steps}-step phenology in {dt:.1f}s")
        # The record metric here is Earth4D's BEST-HEAD within-tolerance accuracy vs the generic trained
        # PE's -- NOT propagator_gain, which is a propagation-vs-static quantity on RAW features and so
        # never gated the encoder at all. _best_acc() is nan-safe when a head was not run.
        declare(
            capability="flowering_peak_month",
            mode=pheno_mode,
            metric="within_tol_accuracy",
            value=_e4d_best,
            split=pheno_mode,
            gains=({"Earth4D vs RFF, best-head within-tol acc": _e4d_best - _rff_best}
                   if _e4d_best == _e4d_best and _rff_best == _rff_best else {}),
            baselines={"RFF_best_head": _rff_best, "raw_best_head": _best_acc("raw")},
            forecast_queries=n_te, tol_days=a.pheno_tol, K=a.rec_k, hops=a.gnn_hops,
            attn=a.pheno_attn, obs=len(lat), seconds=dt,
            propagator_gain_acc_raw=pg_raw_gnn_acc, propagator_gain_mae_raw=best_prop_raw_mae,
            static_mae_raw=r["raw"]["static_mae"], gnn_mae_raw=r["raw"]["gnn_mae"],
        )
        return {"static_mae_raw": r["raw"]["static_mae"], "gnn_mae_raw": r["raw"]["gnn_mae"], "lstm_mae_raw": r["raw"]["lstm_mae"],
                "attn_mae_raw": r["raw"].get("attn_mae", float("nan")), "sp_mae_raw": r["raw"].get("sp_mae", float("nan")),
                "propagator_gain_mae": best_prop_raw_mae, "propagator_gain_gnn_mae": pg_raw_gnn_mae, "propagator_gain_lstm_mae": pg_raw_lstm_mae,
                "propagator_gain_attn_mae": pg_raw_attn_mae, "propagator_gain_attn_acc": pg_raw_attn_acc,
                "propagator_gain_sp_mae": pg_raw_sp_mae, "propagator_gain_sp_acc": pg_raw_sp_acc,
                "propagator_gain_acc": pg_raw_gnn_acc, "propagator_gain_e4d_mae": pg_e4d_gnn_mae, "propagator_gain_rff_mae": pg_rff_gnn_mae,
                "static_acc_raw": r["raw"]["static_acc"], "gnn_acc_raw": r["raw"]["gnn_acc"], "lstm_acc_raw": r["raw"]["lstm_acc"],
                "obs": len(lat), "seconds": dt, "phenology": True, "n_te": n_te}

    if a.pheno_densefield:
        # ===== LOOP-spacetime rule-24 DENSE-FIELD interpolation (mean-DOY, empty vs occupied query cells) =====
        # Reuses ALL phenology leak-guards: query features SPACE-ONLY (t=0 baked into raw_sp), spatial-only
        # edge (no dt-to-query), neighbours carry OWN observed DOY. ADDITIONAL leak-guard: the query cell is
        # excluded from its own neighbour window (contributes nothing to itself). raw features only.
        assert a.forecast, "--pheno_densefield requires --forecast"
        import numpy as _np
        from deepearth.autoresearch.programs.spacetime.dyntargets import run_pheno_densefield
        coords_ll = torch.tensor(_np.stack([lat, lon], 1).astype(_np.float32))
        rn_sp = _np.stack([lat / 90.0, lon / 180.0], 1).astype(_np.float32)
        raw_sp = torch.tensor(rn_sp)
        SPLIT = "SPATIAL(unseen-geo)" if a.pheno_spatial else "TEMPORAL(future)"
        r = run_pheno_densefield(raw_sp, raw_sp.shape[1], days, coords_ll, test, dev,
                                 block=a.densefield_block, drop_cell_frac=a.densefield_drop,
                                 K=a.rec_k, steps=a.steps, lr=a.lr, hidden=a.rec_hidden, tol_days=a.pheno_tol, seed=a.seed)
        dt = time.time() - t0
        print("=== SPACETIME | mode=PHENO-DENSEFIELD(mean-DOY, same-cell-EXCLUDED) split=%s obs=%d queries=%d block=%.2fdeg drop_cells=%.2f pool=%d K=%d ===" % (SPLIT, len(lat), r.get("n_te", 0), a.densefield_block, a.densefield_drop, r.get("pool_n", 0), a.rec_k))
        for _lab, _key in (("ALL      ", "all"), ("EMPTY-cell", "empty"), ("OCCUPIED ", "occ")):
            d = r.get(_key, {})
            print("  %s n=%6d | static MAE %6.2fd  LSTM MAE %6.2fd  gain %+.2fd" % (_lab, d.get("n", 0), d.get("static_mae", float("nan")), d.get("lstm_mae", float("nan")), d.get("gain", float("nan"))))
        print("  LEAK-GUARD: query feat SPACE-ONLY(t=0); edge SPATIAL-only(no dt); query cell EXCLUDED from own window (surrounding cells only)")
        print("  %d obs in %.1fs" % (len(lat), dt))
        declare(
            capability="", mode="PHENO-DENSEFIELD(mean-DOY, same-cell-EXCLUDED)", metric="MAEd",
            value=float((r.get("all") or {}).get("static_mae", float("nan"))),
            diagnostic=True, diagnostic_reason=PHENO_RAW_REASON,
            split=SPLIT, obs=len(lat), queries=r.get("n_te"), pool_n=r.get("pool_n"),
            block=a.densefield_block, drop_cell_frac=a.densefield_drop, K=a.rec_k, seconds=dt,
            cells={k: r.get(k) for k in ("all", "empty", "occ")},
        )
        return {"pheno_densefield": True, "split": SPLIT, "block": a.densefield_block, "drop_cell_frac": a.densefield_drop,
                "n_te": r.get("n_te", 0), "pool_n": r.get("pool_n", 0),
                "all": r.get("all"), "empty": r.get("empty"), "occ": r.get("occ"), "seconds": dt}

    if a.pheno_env or a.pheno_disttarget or a.pheno_taxon:
        # ============ LOOP-spacetime NEW DIRECTIONS on the mean-DOY graduation target ============
        # All reuse phenology leak-guards: query feature SPACE-ONLY (t=0); edge = spatial offset only
        # (no dt-to-query); neighbours carry OWN observed DOY. raw features only (Earth4D settled neutral).
        assert a.forecast, "--pheno_env/--pheno_disttarget/--pheno_taxon require --forecast"
        import numpy as _np
        coords_ll = torch.tensor(_np.stack([lat, lon], 1).astype(_np.float32))
        rn_sp = _np.stack([lat / 90.0, lon / 180.0], 1).astype(_np.float32)
        raw_sp = torch.tensor(rn_sp)
        fdim = raw_sp.shape[1]
        SPLIT = "SPATIAL(unseen-geo)" if a.pheno_spatial else "TEMPORAL(future)"
        res = {"obs": len(lat), "split": SPLIT}

        if a.pheno_env:
            from deepearth.autoresearch.programs.spacetime.dyntargets import run_pheno_env
            env = load_env(a.cache_dir, gid, fit_mask=~test)
            r = run_pheno_env(raw_sp, fdim, days, coords_ll, env, test, dev,
                              K=a.rec_k, steps=a.steps, lr=a.lr, hidden=a.rec_hidden, tol_days=a.pheno_tol)
            dt = time.time() - t0
            gain_env = r["static_mae"] - r["neighbourenv_mae"]
            gain_nbr = r["static_mae"] - r["neighbour_mae"]
            gain_only = r["static_mae"] - r["envonly_mae"]
            env_lift = r["neighbour_mae"] - r["neighbourenv_mae"]
            print("=== SPACETIME | mode=PHENO-ENV(mean-DOY) split=%s obs=%d queries=%d env_dim=%d tol=+/-%.0fd K=%d ===" % (SPLIT, len(lat), r["n_te"], r["env_dim"], a.pheno_tol, a.rec_k))
            print("  static           MAE %6.2fd acc %.4f  (no propagation, no env floor)" % (r["static_mae"], r["static_acc"]))
            print("  neighbour-only   MAE %6.2fd acc %.4f  gain %+.2fd" % (r["neighbour_mae"], r["neighbour_acc"], gain_nbr))
            print("  neighbour+env    MAE %6.2fd acc %.4f  gain %+.2fd  (env-lift-over-neighbour %+.2fd)" % (r["neighbourenv_mae"], r["neighbourenv_acc"], gain_env, env_lift))
            print("  env-only(static) MAE %6.2fd acc %.4f  gain %+.2fd" % (r["envonly_mae"], r["envonly_acc"], gain_only))
            print("  %d obs in %.1fs" % (len(lat), dt))
            res.update({"pheno_env": True, "n_te": r["n_te"], "env_dim": r["env_dim"],
                    "static_mae": r["static_mae"], "neighbour_mae": r["neighbour_mae"],
                    "neighbourenv_mae": r["neighbourenv_mae"], "envonly_mae": r["envonly_mae"],
                    "gain_neighbour": gain_nbr, "gain_neighbourenv": gain_env, "gain_envonly": gain_only,
                    "env_lift_over_neighbour": env_lift, "seconds": dt})
            declare(
                capability="", mode="PHENO-ENV(mean-DOY)", metric="MAEd",
                value=float(r.get("static_mae", float("nan"))),
                diagnostic=True, diagnostic_reason=PHENO_RAW_REASON,
                split=SPLIT, obs=len(lat), queries=r.get("n_te"), env_dim=r.get("env_dim"),
                tol_days=a.pheno_tol, K=a.rec_k, lstm_mae=r.get("lstm_mae"), gain=r.get("gain"),
                seconds=dt,
            )
            return res

        if a.pheno_disttarget:
            from deepearth.autoresearch.programs.spacetime.dyntargets import run_pheno_disttarget
            r = run_pheno_disttarget(raw_sp, fdim, days, coords_ll, test, dev, target=a.pheno_disttarget,
                                     K=a.rec_k, steps=a.steps, lr=a.lr, hidden=a.rec_hidden, hops=a.gnn_hops, tol_days=a.pheno_tol)
            dt = time.time() - t0
            gnn_gain = r["static_mae"] - r["gnn_mae"]; lstm_gain = r["static_mae"] - r["lstm_mae"]
            print("=== SPACETIME | mode=PHENO-DISTTARGET(%s) split=%s obs=%d queries=%d tol=+/-%.0fd K=%d ===" % (a.pheno_disttarget, SPLIT, len(lat), r["n_te"], a.pheno_tol, a.rec_k))
            print("  static MAE %6.2fd acc %.4f -> GNN MAE %6.2fd (gain %+.2fd) | LSTM MAE %6.2fd (gain %+.2fd)" % (r["static_mae"], r["static_acc"], r["gnn_mae"], gnn_gain, r["lstm_mae"], lstm_gain))
            print("  %d obs in %.1fs" % (len(lat), dt))
            res.update({"pheno_disttarget": a.pheno_disttarget, "n_te": r["n_te"],
                    "static_mae": r["static_mae"], "gnn_mae": r["gnn_mae"], "lstm_mae": r["lstm_mae"],
                    "gnn_gain": gnn_gain, "lstm_gain": lstm_gain, "seconds": dt})
            declare(
                capability="", mode=f"PHENO-DISTTARGET({a.pheno_disttarget})", metric="MAEd",
                value=float(r.get("static_mae", float("nan"))),
                diagnostic=True, diagnostic_reason=PHENO_RAW_REASON,
                split=SPLIT, obs=len(lat), queries=r.get("n_te"), tol_days=a.pheno_tol, K=a.rec_k,
                lstm_mae=r.get("lstm_mae"), gain=r.get("gain"), seconds=dt,
            )
            return res

        if a.pheno_taxon:
            from deepearth.autoresearch.programs.spacetime.dyntargets import run_pheno_by_taxon
            import csv as _csv
            from pathlib import Path as _P
            rows = list(_csv.DictReader(open(_P(a.cache_dir) / "derived/species_index.csv")))
            vocab = _np.load(_P(a.cache_dir) / "gbif_vocab.npz", allow_pickle=True)["global_idx"]
            col = a.pheno_taxon
            taxon_str = _np.array([rows[i][col] for i in vocab])
            names, gid_of_species = _np.unique(taxon_str, return_inverse=True)
            sp_all = load_species(a.cache_dir, a.n_shards)[obs_index]
            group = gid_of_species[sp_all].astype(_np.int64)
            r = run_pheno_by_taxon(raw_sp, fdim, days, coords_ll, group, test, dev,
                                   K=a.rec_k, steps=a.steps, lr=a.lr, hidden=a.rec_hidden, tol_days=a.pheno_tol)
            dt = time.time() - t0
            print("=== SPACETIME | mode=PHENO-BY-TAXON(%s) split=%s obs=%d queries=%d tol=+/-%.0fd K=%d ===" % (col, SPLIT, len(lat), r["n_te"], a.pheno_tol, a.rec_k))
            for row in r["groups"][:15]:
                print("  %-22s n_te %5d | static %6.2fd -> LSTM %6.2fd  gain %+.2fd" % (str(names[row["group"]])[:22], row["n_te"], row["static_mae"], row["lstm_mae"], row["gain"]))
            print("  %d obs in %.1fs" % (len(lat), dt))
            res.update({"pheno_taxon": col, "n_te": r["n_te"],
                    "groups": [{"name": str(names[row["group"]]), "n_te": row["n_te"], "static_mae": row["static_mae"], "lstm_mae": row["lstm_mae"], "gain": row["gain"]} for row in r["groups"]],
                    "seconds": dt})
            declare(
                capability="", mode=f"PHENO-BY-TAXON({col})", metric="MAEd",
                value=float(r["groups"][0]["static_mae"]) if r.get("groups") else float("nan"),
                diagnostic=True, diagnostic_reason=PHENO_RAW_REASON,
                split=SPLIT, obs=len(lat), queries=r.get("n_te"), tol_days=a.pheno_tol, K=a.rec_k,
                n_groups=len(r.get("groups", [])), seconds=dt,
            )
            return res

    if a.ar_rollout:
        # ================= LOOP-spacetime: rule-1 CAUSAL AUTOREGRESSIVE ROLLOUT ==========================
        # The forecast head so far is DIRECT single-shot lead prediction. Rule 1 demands a causal AR model:
        # predict ONE Delta-step ahead, FEED the prediction back as the query's OWN current-state, and roll
        # forward to the final horizon. Compare absR2 of the ROLLED prediction vs a single-shot DIRECT
        # predictor at the SAME final horizon. Strong LEVEL targets only (community-activity / abundance).
        #
        # LEAK-GUARDS (identical discipline to breadth/abundance):
        #   * query positional feature is SPACE-ONLY (t stripped); the query timestamp is never a feature.
        #   * edge carries ONLY the spatial offset (dlat,dlon) -- no dt-to-query.
        #   * neighbour state = each neighbour's OWN TRAILING-PAST activity over [d-win, d] (lead=0): purely
        #     historical, observed <= the neighbour's own day; NEVER anything relative to the query or future.
        #   * the fed-back value is the model's OWN prediction (seeded 0), so the rollout consumes only its
        #     predictions + past-window features. A separate query-state channel carries it (1 extra dim).
        #   * time-only static smoke test at the FINAL horizon reported for every run.
        assert a.forecast, "--ar_rollout requires --forecast"
        import numpy as _np
        from deepearth.autoresearch.programs.spacetime.dyntargets import (
            _windows, _assemble, _reg_skill, _community_activity_target, _abundance_target, _richness_target)
        lat_a = lat.astype(_np.float32); lon_a = lon.astype(_np.float32)
        rn_sp = _np.stack([lat / 90.0, lon / 180.0], 1).astype(_np.float32)
        raw_feat = torch.tensor(rn_sp)                                 # SPACE-ONLY query feature
        fdim = raw_feat.shape[1]; K = a.rec_k; H = a.rec_hidden; out_dim = 1
        win = a.abund_win                                              # overlap regime: win = step + 180 (set by caller)
        n_steps = max(1, int(round(a.ar_final / a.ar_step)))
        leads = [float(j * a.ar_step) for j in range(1, n_steps + 1)]  # intermediate + final leads
        assert abs(leads[-1] - a.ar_final) < 1e-6, f"ar_final {a.ar_final} not a multiple of ar_step {a.ar_step}"
        _sp_arr = load_species(a.cache_dir, a.n_shards)[obs_index] if a.ar_target == "richness" else None

        def _tgt_at(lead):
            if a.ar_target == "abundance":
                return _abundance_target(lat_a, lon_a, days, win=win, lead=lead, delta=False).astype(_np.float32)
            if a.ar_target == "richness":
                return _richness_target(lat_a, lon_a, days, _sp_arr, win=win, lead=lead)[0].astype(_np.float32)
            return _community_activity_target(lat_a, lon_a, days, win=win, lead=lead)[0].astype(_np.float32)

        # neighbour PAST state = its OWN target-lead activity (SAME convention as the settled breadth/direct
        # baseline, whose leak guard passes): each neighbour carries only its own observed quantity, never
        # anything relative to the query. This makes AR-vs-direct a fair matched comparison; the ONLY AR
        # addition is the fed-back query-state channel.
        past_state = _tgt_at(a.ar_final).reshape(-1, 1).astype(_np.float32)
        S = past_state.shape[1]

        _test = test
        tr_idx = _np.where(~_test)[0]; te_idx = _np.where(_test)[0]
        _rng = _np.random.default_rng(0)
        q_tr = tr_idx if len(tr_idx) <= 6000 else _rng.choice(tr_idx, 6000, replace=False)
        g_tr, v_tr = _windows(lat_a, lon_a, days, q_tr, tr_idx, K)
        g_te, v_te = _windows(lat_a, lon_a, days, te_idx, tr_idx, K)
        # assemble the neighbour tensors ONCE (windows are lead-independent: causal past<=d); only the target
        # scalar changes per lead. Use a dummy target to grab the leak-safe (nfeat,nstate,edge,mask,len) tensors.
        dummy = _np.zeros(len(lat_a), _np.float32)
        tr0 = _assemble(raw_feat, past_state, days, lat_a, lon_a, q_tr, g_tr, v_tr, dummy, K, out_dim)
        te0 = _assemble(raw_feat, past_state, days, lat_a, lon_a, te_idx, g_te, v_te, dummy, K, out_dim)
        _to = lambda ts: [t.to(dev) for t in ts]
        nftr, nstr, qftr, etr, mtr, ltr, _, _ = _to(tr0)
        nfte, nste, qfte, ete, mte, lte, _, _ = _to(te0)
        # per-lead targets aligned to the SAME ok-masked rows _assemble kept (ok = window has >=1 valid nb)
        ok_tr = torch.tensor(v_tr).any(1).numpy(); ok_te = torch.tensor(v_te).any(1).numpy()
        Ytr = {ld: torch.tensor(_tgt_at(ld)[q_tr][ok_tr]).unsqueeze(-1).to(dev) for ld in leads}
        Yte = {ld: torch.tensor(_tgt_at(ld)[te_idx][ok_te]).unsqueeze(-1).to(dev) for ld in leads}
        Btr = int(nftr.shape[0]); n_te = int(nfte.shape[0])
        if Btr == 0 or n_te == 0:
            print("=== ar_rollout: EMPTY window set, abort ==="); return {"ar_rollout": True, "n_te": n_te}
        bs = min(2048, Btr)

        # single-step propagator g: (neighbour window, edge, query-prev-state ŷ) -> level at this step's lead.
        # The +1 input dim is the fed-back query state channel (0 at step 1, then the model's own prediction).
        class _ARStep(nn.Module):
            def __init__(s):
                super().__init__()
                s.lstm = nn.LSTM(fdim + S + 2, H, num_layers=2, batch_first=True)
                s.head = nn.Sequential(nn.Linear(H + 1, H), nn.GELU(), nn.Linear(H, out_dim))
            def forward(s, nf, ns, edge, lengths, qprev):
                x = torch.cat([nf, ns, edge], -1)
                packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu().clamp(min=1),
                                                           batch_first=True, enforce_sorted=False)
                _, (h, _) = s.lstm(packed)
                return s.head(torch.cat([h[-1], qprev], -1))
        # direct single-shot: identical capacity, no fed-back channel (qprev fixed 0), trained ONLY on final lead
        class _Direct(nn.Module):
            def __init__(s):
                super().__init__()
                s.lstm = nn.LSTM(fdim + S + 2, H, num_layers=2, batch_first=True)
                s.head = nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.Linear(H, out_dim))
            def forward(s, nf, ns, edge, lengths):
                x = torch.cat([nf, ns, edge], -1)
                packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu().clamp(min=1),
                                                           batch_first=True, enforce_sorted=False)
                _, (h, _) = s.lstm(packed)
                return s.head(h[-1])

        loss_fn = lambda p, y: F.smooth_l1_loss(p, y)
        # ---- train the AR step model: shared weights over ALL steps; teacher-forced qprev = prev-lead target,
        #      plus scheduled feed-back of the model's own prediction (0.5 mix) so eval-time rollout matches.
        arm = _ARStep().to(dev); opt = torch.optim.Adam(arm.parameters(), lr=a.lr); arm.train()
        for it in range(a.steps):
            sidx = torch.randint(0, Btr, (bs,), device=dev)
            qprev = torch.zeros(bs, 1, device=dev)
            loss = 0.0
            for j, ld in enumerate(leads):
                pred = arm(nftr[sidx], nstr[sidx], etr[sidx], ltr[sidx], qprev)
                loss = loss + loss_fn(pred, Ytr[ld][sidx])
                # feed BACK: mix model prediction with teacher (prev-lead truth) -> AR consistency w/o drift
                tf = Ytr[ld][sidx]
                qprev = (0.5 * pred.detach() + 0.5 * tf).detach()
            opt.zero_grad(); (loss / len(leads)).backward(); opt.step()
        arm.eval()
        # ---- ROLLOUT at eval: seed qprev=0, feed the model's OWN prediction forward. NO future obs consumed.
        roll_r2 = {}; qprev = torch.zeros(n_te, 1, device=dev)
        with torch.no_grad():
            for ld in leads:
                pred = arm(nfte, nste, ete, lte, qprev)
                roll_r2[ld] = _reg_skill(pred, Yte[ld], Yte[ld])
                qprev = pred                                          # pure AR: its own prediction only
        # ---- DIRECT single-shot at the final horizon (matched capacity, one prediction) ----
        dm = _Direct().to(dev); opt = torch.optim.Adam(dm.parameters(), lr=a.lr); dm.train()
        yfin_tr = Ytr[leads[-1]]
        for it in range(a.steps):
            sidx = torch.randint(0, Btr, (bs,), device=dev)
            loss = loss_fn(dm(nftr[sidx], nstr[sidx], etr[sidx], ltr[sidx]), yfin_tr[sidx])
            opt.zero_grad(); loss.backward(); opt.step()
        dm.eval()
        with torch.no_grad():
            direct_fin = _reg_skill(dm(nfte, nste, ete, lte), Yte[leads[-1]], Yte[leads[-1]])

        # ---- STATIC FLOOR at final horizon (query space-only feat, no propagation) for dR2 context ----
        class _StaticH(nn.Module):
            def __init__(s):
                super().__init__(); s.net = nn.Sequential(nn.Linear(fdim, H), nn.GELU(), nn.Linear(H, out_dim))
            def forward(s, qf): return s.net(qf)
        sh = _StaticH().to(dev); opt = torch.optim.Adam(sh.parameters(), lr=a.lr); sh.train()
        for _ in range(a.steps):
            sidx = torch.randint(0, Btr, (bs,), device=dev)
            loss = loss_fn(sh(qftr[sidx]), yfin_tr[sidx]); opt.zero_grad(); loss.backward(); opt.step()
        sh.eval()
        with torch.no_grad(): static_fin = _reg_skill(sh(qfte), Yte[leads[-1]], Yte[leads[-1]])

        # ---- LEAK-GUARD: time-only static head at the FINAL horizon must NOT solve it ----
        tnorm = ((days - days.min()) / (days.max() - days.min() + 1e-9)).astype(_np.float32)
        tq_tr = torch.tensor(tnorm[q_tr][ok_tr]).unsqueeze(-1).to(dev)[: Btr]
        tq_te = torch.tensor(tnorm[te_idx][ok_te]).unsqueeze(-1).to(dev)[: n_te]
        yl_tr = Ytr[leads[-1]][: tq_tr.shape[0]]; yl_te = Yte[leads[-1]][: tq_te.shape[0]]
        class _TimeOnly(nn.Module):
            def __init__(s):
                super().__init__(); s.net = nn.Sequential(nn.Linear(1, H), nn.GELU(), nn.Linear(H, 1))
            def forward(s, x): return s.net(x)
        tom = _TimeOnly().to(dev); opt = torch.optim.Adam(tom.parameters(), lr=a.lr)
        for _ in range(a.steps):
            sidx = torch.randint(0, tq_tr.shape[0], (min(2048, tq_tr.shape[0]),), device=dev)
            loss = loss_fn(tom(tq_tr[sidx]), yl_tr[sidx]); opt.zero_grad(); loss.backward(); opt.step()
        tom.eval()
        with torch.no_grad(): leak_mae, leak_r2 = _reg_skill(tom(tq_te), yl_te, yl_te)

        print(f"=== SPACETIME AR-ROLLOUT | target={a.ar_target} | raw PE | obs={len(lat_a)} q={n_te} K={K} win={win:.0f}d step={a.ar_step:.0f}d final={a.ar_final:.0f}d n_steps={n_steps} Sdim={S} ===")
        for ld in leads:
            m, r2 = roll_r2[ld]
            tag = " <FINAL>" if abs(ld - leads[-1]) < 1e-6 else ""
            print(f"  rollout  lead {ld:6.0f}d | MAE {m:7.4f}  absR2 {r2:+.4f}{tag}")
        dmae, dr2 = direct_fin
        rmae, rr2 = roll_r2[leads[-1]]
        smae, sr2 = static_fin
        print(f"  DIRECT   lead {leads[-1]:6.0f}d | MAE {dmae:7.4f}  absR2 {dr2:+.4f}  (single-shot, matched horizon; dR2 vs static {dr2 - sr2:+.4f})")
        print(f"  STATIC   lead {leads[-1]:6.0f}d | MAE {smae:7.4f}  absR2 {sr2:+.4f}  (no-propagation floor)")
        print(f"  AR final dR2 vs static           | {rr2 - sr2:+.4f}")
        print(f"  AR - DIRECT (final absR2)        | {rr2 - dr2:+.4f}   (POSITIVE = rollout holds skill better)")
        print(f"  LEAK-GUARD time-only (final)     | MAE {leak_mae:7.4f}  absR2 {leak_r2:+.4f}  (must be ~0/neg = no time leak)")
        dt = time.time() - t0
        print(f"  [profile] q={n_te} K={K} hidden={H} steps={a.steps} n_steps={n_steps}")
        print(f"  {len(lat_a)} obs, {a.steps}-step AR-rollout in {dt:.1f}s")
        return {"ar_rollout": True, "target": a.ar_target, "final_lead": leads[-1], "step": a.ar_step,
                "n_steps": n_steps, "win": win, "K": K, "n_te": n_te,
                "rollout_absR2": {ld: roll_r2[ld][1] for ld in leads},
                "direct_final_absR2": dr2, "static_final_absR2": sr2,
                "ar_minus_direct": rr2 - dr2, "ar_dR2_vs_static": rr2 - sr2,
                "leak_absR2": leak_r2, "seconds": dt}

    if a.ar_cond_lead:
        # ================= PIVOT: CONTINUOUS-LEAD CONDITIONING ===========================================
        # Feed the target lead as an INPUT so ONE model spans all horizons; compare vs per-lead DIRECT
        # specialists at each horizon. Same leak-guards (space-only query feat, spatial-only edge, neighbour
        # OWN trailing-past state). The lead scalar is a MODEL CONTROL, not an observation -> no leak (it says
        # WHICH horizon to predict, carries no future data). Time-only smoke test still reported.
        assert a.forecast, "--ar_cond_lead requires --forecast"
        import numpy as _np
        from deepearth.autoresearch.programs.spacetime.dyntargets import (
            _windows, _assemble, _reg_skill, _community_activity_target)
        lat_a = lat.astype(_np.float32); lon_a = lon.astype(_np.float32)
        rn_sp = _np.stack([lat / 90.0, lon / 180.0], 1).astype(_np.float32)
        raw_feat = torch.tensor(rn_sp); fdim = raw_feat.shape[1]
        K = a.rec_k; H = a.rec_hidden; out_dim = 1; win = a.abund_win
        n_steps = max(1, int(round(a.ar_final / a.ar_step)))
        leads = [float(j * a.ar_step) for j in range(1, n_steps + 1)]
        past_state = _community_activity_target(lat_a, lon_a, days, win=win, lead=0.0)[0].reshape(-1, 1).astype(_np.float32)
        S = past_state.shape[1]
        _test = test
        tr_idx = _np.where(~_test)[0]; te_idx = _np.where(_test)[0]
        _rng = _np.random.default_rng(0)
        q_tr = tr_idx if len(tr_idx) <= 6000 else _rng.choice(tr_idx, 6000, replace=False)
        g_tr, v_tr = _windows(lat_a, lon_a, days, q_tr, tr_idx, K)
        g_te, v_te = _windows(lat_a, lon_a, days, te_idx, tr_idx, K)
        dummy = _np.zeros(len(lat_a), _np.float32)
        tr0 = _assemble(raw_feat, past_state, days, lat_a, lon_a, q_tr, g_tr, v_tr, dummy, K, out_dim)
        te0 = _assemble(raw_feat, past_state, days, lat_a, lon_a, te_idx, g_te, v_te, dummy, K, out_dim)
        _to = lambda ts: [t.to(dev) for t in ts]
        nftr, nstr, qftr, etr, mtr, ltr, _, _ = _to(tr0)
        nfte, nste, qfte, ete, mte, lte, _, _ = _to(te0)
        ok_tr = torch.tensor(v_tr).any(1).numpy(); ok_te = torch.tensor(v_te).any(1).numpy()
        Ytr = {ld: torch.tensor(_community_activity_target(lat_a, lon_a, days, win=win, lead=ld)[0][q_tr][ok_tr]).unsqueeze(-1).to(dev) for ld in leads}
        Yte = {ld: torch.tensor(_community_activity_target(lat_a, lon_a, days, win=win, lead=ld)[0][te_idx][ok_te]).unsqueeze(-1).to(dev) for ld in leads}
        Btr = int(nftr.shape[0]); n_te = int(nfte.shape[0])
        if Btr == 0 or n_te == 0:
            print("=== ar_cond_lead: EMPTY window set, abort ==="); return {"ar_cond_lead": True, "n_te": n_te}
        bs = min(2048, Btr); lscale = float(max(leads))

        class _CondLead(nn.Module):
            def __init__(s):
                super().__init__()
                s.lstm = nn.LSTM(fdim + S + 2, H, num_layers=2, batch_first=True)
                s.head = nn.Sequential(nn.Linear(H + 1, H), nn.GELU(), nn.Linear(H, out_dim))
            def forward(s, nf, ns, edge, lengths, lead_scalar):
                x = torch.cat([nf, ns, edge], -1)
                packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu().clamp(min=1),
                                                           batch_first=True, enforce_sorted=False)
                _, (h, _) = s.lstm(packed)
                return s.head(torch.cat([h[-1], lead_scalar], -1))
        class _Direct(nn.Module):
            def __init__(s):
                super().__init__()
                s.lstm = nn.LSTM(fdim + S + 2, H, num_layers=2, batch_first=True)
                s.head = nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.Linear(H, out_dim))
            def forward(s, nf, ns, edge, lengths):
                x = torch.cat([nf, ns, edge], -1)
                packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu().clamp(min=1),
                                                           batch_first=True, enforce_sorted=False)
                _, (h, _) = s.lstm(packed)
                return s.head(h[-1])
        loss_fn = lambda p, y: F.smooth_l1_loss(p, y)
        # one conditioned model over ALL leads (lead sampled each step)
        cm = _CondLead().to(dev); opt = torch.optim.Adam(cm.parameters(), lr=a.lr); cm.train()
        for it in range(a.steps):
            ld = leads[torch.randint(0, len(leads), (1,)).item()]
            sidx = torch.randint(0, Btr, (bs,), device=dev)
            ls = torch.full((bs, 1), ld / lscale, device=dev)
            loss = loss_fn(cm(nftr[sidx], nstr[sidx], etr[sidx], ltr[sidx], ls), Ytr[ld][sidx])
            opt.zero_grad(); loss.backward(); opt.step()
        cm.eval()
        cond_r2 = {}
        with torch.no_grad():
            for ld in leads:
                ls = torch.full((n_te, 1), ld / lscale, device=dev)
                cond_r2[ld] = _reg_skill(cm(nfte, nste, ete, lte, ls), Yte[ld], Yte[ld])
        # per-lead DIRECT specialists
        spec_r2 = {}
        for ld in leads:
            dm = _Direct().to(dev); opt = torch.optim.Adam(dm.parameters(), lr=a.lr); dm.train()
            for it in range(a.steps):
                sidx = torch.randint(0, Btr, (bs,), device=dev)
                loss = loss_fn(dm(nftr[sidx], nstr[sidx], etr[sidx], ltr[sidx]), Ytr[ld][sidx])
                opt.zero_grad(); loss.backward(); opt.step()
            dm.eval()
            with torch.no_grad(): spec_r2[ld] = _reg_skill(dm(nfte, nste, ete, lte), Yte[ld], Yte[ld])

        print(f"=== SPACETIME CONTINUOUS-LEAD conditioning | community-activity | raw PE | obs={len(lat_a)} q={n_te} K={K} win={win:.0f}d leads={[int(l) for l in leads]} ===")
        for ld in leads:
            cm_, cr2 = cond_r2[ld]; sm_, sr2 = spec_r2[ld]
            print(f"  lead {ld:6.0f}d | 1-model-cond absR2 {cr2:+.4f}  vs  specialist absR2 {sr2:+.4f}  (cond-spec {cr2 - sr2:+.4f})")
        mean_gap = float(_np.mean([cond_r2[ld][1] - spec_r2[ld][1] for ld in leads]))
        print(f"  mean (cond - specialist) absR2 over leads | {mean_gap:+.4f}   (>=0 = one model matches per-lead specialists)")
        dt = time.time() - t0
        print(f"  [profile] q={n_te} K={K} hidden={H} steps={a.steps} n_leads={len(leads)}")
        print(f"  {len(lat_a)} obs, {a.steps}-step cond-lead in {dt:.1f}s")
        return {"ar_cond_lead": True, "leads": leads, "win": win, "K": K, "n_te": n_te,
                "cond_absR2": {ld: cond_r2[ld][1] for ld in leads},
                "spec_absR2": {ld: spec_r2[ld][1] for ld in leads},
                "mean_cond_minus_spec": mean_gap, "seconds": dt}

    if a.breadth_target:
        # LOOP-spacetime-target-breadth: reuse the settled propagator scaffold (space-only query feat,
        # SPATIAL-only edge, neighbour carries OWN past state) but swap the dynamic LEVEL target. raw PE only,
        # deepLSTM-2L propagator, K<=32. Reports propagator absR2 vs static-floor absR2. Same leak-guards.
        assert a.forecast, "--breadth_target requires --forecast"
        import numpy as _np
        from deepearth.autoresearch.programs.spacetime.dyntargets import (
            _windows, _assemble, _reg_skill, _occupancy_target, _richness_target, _community_activity_target)
        coords_ll = torch.tensor(_np.stack([lat, lon], 1).astype(_np.float32))
        lat_a = coords_ll[:, 0].numpy(); lon_a = coords_ll[:, 1].numpy()
        rn_sp = _np.stack([lat / 90.0, lon / 180.0], 1).astype(_np.float32)
        raw_feat = torch.tensor(rn_sp)                                 # SPACE-ONLY query feature (t stripped)
        fdim = raw_feat.shape[1]
        sp_arr = load_species(a.cache_dir, a.n_shards)[obs_index]
        win, lead = a.abund_win, a.abund_lead
        if a.breadth_target == "occupancy":
            tgt, past = _occupancy_target(lat_a, lon_a, days, sp_arr, win=win, lead=lead, sub=a.breadth_sub)
            _tn = "OCCUPANCY-LEVEL(detect-frac)"
        elif a.breadth_target == "richness":
            tgt, past = _richness_target(lat_a, lon_a, days, sp_arr, win=win, lead=lead)
            _tn = "COMMUNITY-RICHNESS-LEVEL(log-nspp)"
        else:
            tgt, past = _community_activity_target(lat_a, lon_a, days, win=win, lead=lead)
            _tn = "COMMUNITY-ACTIVITY-LEVEL(log-count-all)"
        tgt = tgt.astype(_np.float32)
        nstate = past.astype(_np.float32) if past is not None else tgt.reshape(-1, 1).astype(_np.float32)
        S = nstate.shape[1]; K = a.rec_k; H = a.rec_hidden; out_dim = 1

        _test = test
        tr_idx = _np.where(~_test)[0]; te_idx = _np.where(_test)[0]
        _rng = _np.random.default_rng(0)
        q_tr = tr_idx if len(tr_idx) <= 6000 else _rng.choice(tr_idx, 6000, replace=False)
        g_tr, v_tr = _windows(lat_a, lon_a, days, q_tr, tr_idx, K)
        g_te, v_te = _windows(lat_a, lon_a, days, te_idx, tr_idx, K)
        tr = _assemble(raw_feat, nstate, days, lat_a, lon_a, q_tr, g_tr, v_tr, tgt, K, out_dim)
        te = _assemble(raw_feat, nstate, days, lat_a, lon_a, te_idx, g_te, v_te, tgt, K, out_dim)
        _to = lambda ts: [t.to(dev) for t in ts]
        nftr, nstr, qftr, etr, mtr, ltr, ytr, _ = _to(tr)
        nfte, nste, qfte, ete, mte, lte, yte, _ = _to(te)
        n_te = int(nfte.shape[0]); Btr = int(nftr.shape[0])
        if Btr == 0 or n_te == 0:
            print("=== breadth_target: EMPTY window set, abort ==="); return {"breadth_target": a.breadth_target, "n_te": n_te}
        bs = min(2048, Btr)

        class _StaticH(nn.Module):
            def __init__(s):
                super().__init__(); s.net = nn.Sequential(nn.Linear(fdim, H), nn.GELU(), nn.Linear(H, out_dim))
            def forward(s, qf): return s.net(qf)

        class _DeepLSTM(nn.Module):
            def __init__(s, layers):
                super().__init__()
                s.lstm = nn.LSTM(fdim + S + 2, H, num_layers=layers, batch_first=True)
                s.head = nn.Linear(H, out_dim)
            def forward(s, nf, ns, edge, lengths):
                x = torch.cat([nf, ns, edge], -1)
                packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu().clamp(min=1),
                                                           batch_first=True, enforce_sorted=False)
                _, (h, _) = s.lstm(packed)
                return s.head(h[-1])

        loss_fn = lambda p, y: F.smooth_l1_loss(p, y)
        def _train(model, fwd):
            opt = torch.optim.Adam(model.parameters(), lr=a.lr)
            model.train()
            for _ in range(a.steps):
                sidx = torch.randint(0, Btr, (bs,), device=dev)
                loss = loss_fn(fwd(model, sidx), ytr[sidx])
                opt.zero_grad(); loss.backward(); opt.step()
            model.eval()

        results = {}
        sh = _StaticH().to(dev); _train(sh, lambda m, s: m(qftr[s]))
        with torch.no_grad(): results["static"] = _reg_skill(sh(qfte), yte, yte)
        lstm = _DeepLSTM(2).to(dev)
        _train(lstm, lambda m, s: m(nftr[s], nstr[s], etr[s], ltr[s]))
        with torch.no_grad(): results["lstm2"] = _reg_skill(lstm(nfte, nste, ete, lte), yte, yte)

        # LEAK-GUARD SMOKE TEST: a static head reading ONLY the query's own normalized time coord must NOT
        # solve it. Low absR2 => the propagator result is not a time-arithmetic leak. Uses the SAME ok-masked
        # test rows so it is directly comparable to the propagator absR2 above.
        tnorm = ((days - days.min()) / (days.max() - days.min() + 1e-9)).astype(_np.float32)
        okmask_te = torch.tensor(v_te).any(1).numpy()
        tq_te = torch.tensor(tnorm[te_idx][okmask_te]).unsqueeze(-1).to(dev)
        okmask_tr = torch.tensor(v_tr).any(1).numpy()
        tq_tr = torch.tensor(tnorm[q_tr][okmask_tr]).unsqueeze(-1).to(dev)
        n_lk = min(tq_tr.shape[0], ytr.shape[0]); tq_tr = tq_tr[:n_lk]; yl_tr = ytr[:n_lk]
        n_lke = min(tq_te.shape[0], yte.shape[0]); tq_te = tq_te[:n_lke]; yl_te = yte[:n_lke]
        class _TimeOnly(nn.Module):
            def __init__(s):
                super().__init__(); s.net = nn.Sequential(nn.Linear(1, H), nn.GELU(), nn.Linear(H, 1))
            def forward(s, x): return s.net(x)
        to_m = _TimeOnly().to(dev); opt = torch.optim.Adam(to_m.parameters(), lr=a.lr)
        for _ in range(a.steps):
            sidx = torch.randint(0, tq_tr.shape[0], (min(2048, tq_tr.shape[0]),), device=dev)
            loss = loss_fn(to_m(tq_tr[sidx]), yl_tr[sidx]); opt.zero_grad(); loss.backward(); opt.step()
        to_m.eval()
        with torch.no_grad(): leak_mae, leak_r2 = _reg_skill(to_m(tq_te), yl_te, yl_te)

        s_mae, s_r2 = results["static"]; l_mae, l_r2 = results["lstm2"]
        print(f"  static-floor        | MAE {s_mae:7.4f}  absR2 {s_r2:+.4f}")
        print(f"  deepLSTM-2L         | MAE {l_mae:7.4f}  absR2 {l_r2:+.4f}  (dR2 vs static {l_r2 - s_r2:+.4f})")
        print(f"  LEAK-GUARD time-only| MAE {leak_mae:7.4f}  absR2 {leak_r2:+.4f}  (must be ~0/negative = no time leak)")
        dt = time.time() - t0
        print(f"  {len(lat_a)} obs, {a.steps}-step breadth in {dt:.1f}s")
        declare(
            capability="", mode=f"BREADTH({_tn})", metric="absR2", value=s_r2,
            diagnostic=True,
            diagnostic_reason=f"{_tn} is not a scorecard capability; " + RAW_PE_REASON,
            obs=len(lat_a), queries=n_te, K=K, win=win, lead=lead, seconds=dt,
            static_mae=s_mae, lstm_absR2=l_r2, leak_guard_absR2=leak_r2,
        )
        return {"breadth_target": a.breadth_target, "target": _tn, "static_absR2": s_r2,
                "lstm2_absR2": l_r2, "leak_absR2": leak_r2, "win": win, "lead": lead, "K": K,
                "n_te": n_te, "seconds": dt}

    if a.abund_prop_arch:
        # ensure space-only query feature + coords are available (mirror the first_arrival/abundance setup)
        import numpy as _np0
        _rn = _np0.stack([lat / 90.0, lon / 180.0], 1).astype(_np0.float32)
        raw_sp = torch.tensor(_rn)
        coords_ll = torch.tensor(_np0.stack([lat, lon], 1).astype(_np0.float32))
        # LOOP-spacetime propagator-ARCHITECTURE probe on the LEVEL abundance target (settled forecastable:
        # LSTM abs R2 up to +0.76). ONE structural change: swap the causal propagator head. Reuse dyntargets'
        # leak-guarded window builder + target; define deeper/attention heads LOCALLY (additive, probe-only).
        assert a.forecast, "--abund_prop_arch requires --forecast"
        import numpy as _np
        from deepearth.autoresearch.programs.spacetime.dyntargets import (
            _abundance_target, _windows, _assemble, _reg_skill, doy_of, doy_to_vec)
        lat_a = coords_ll[:, 0].numpy(); lon_a = coords_ll[:, 1].numpy()
        raw_feat = raw_sp                                              # SPACE-ONLY query feature (t stripped)
        fdim = raw_feat.shape[1]
        tgt = _abundance_target(lat_a, lon_a, days, win=a.abund_win, lead=a.abund_lead, delta=a.abund_delta)
        # neighbour PAST state: abundance-only, or joint multivariate [abund || DOY sin,cos || occupancy bit]
        if a.abund_multivar:
            doyv = doy_to_vec(doy_of(days))                           # [N,2] each neighbour's own past DOY phase
            occ = (tgt > 0).astype(_np.float32).reshape(-1, 1)        # past occupancy (was cell active)
            nstate = _np.concatenate([tgt.reshape(-1, 1), doyv, occ], 1).astype(_np.float32)
        else:
            nstate = tgt.reshape(-1, 1).astype(_np.float32)
        S = nstate.shape[1]; K = a.rec_k; H = a.rec_hidden; out_dim = 1

        # build the SAME leak-guarded train/test window tensors used by dyntargets._fit_eval
        _test = test
        tr_idx = _np.where(~_test)[0]; te_idx = _np.where(_test)[0]
        _rng = _np.random.default_rng(0)
        q_tr = tr_idx if len(tr_idx) <= 6000 else _rng.choice(tr_idx, 6000, replace=False)
        g_tr, v_tr = _windows(lat_a, lon_a, days, q_tr, tr_idx, K)
        g_te, v_te = _windows(lat_a, lon_a, days, te_idx, tr_idx, K)
        tr = _assemble(raw_feat, nstate, days, lat_a, lon_a, q_tr, g_tr, v_tr, tgt, K, out_dim)
        te = _assemble(raw_feat, nstate, days, lat_a, lon_a, te_idx, g_te, v_te, tgt, K, out_dim)
        _to = lambda ts: [t.to(dev) for t in ts]
        nftr, nstr, qftr, etr, mtr, ltr, ytr, _ = _to(tr)
        nfte, nste, qfte, ete, mte, lte, yte, _ = _to(te)
        n_te = int(nfte.shape[0]); Btr = int(nftr.shape[0])
        if Btr == 0 or n_te == 0:
            print("=== abund_prop_arch: EMPTY window set, abort ==="); return {"abund_prop_arch": True, "n_te": n_te}
        bs = min(2048, Btr)

        class _StaticH(nn.Module):
            def __init__(s):
                super().__init__(); s.net = nn.Sequential(nn.Linear(fdim, H), nn.GELU(), nn.Linear(H, out_dim))
            def forward(s, qf): return s.net(qf)

        class _DeepLSTM(nn.Module):
            def __init__(s, layers):
                super().__init__()
                s.lstm = nn.LSTM(fdim + S + 2, H, num_layers=layers, batch_first=True,
                                 dropout=0.0)
                s.head = nn.Linear(H, out_dim)
            def forward(s, nf, ns, edge, lengths):
                x = torch.cat([nf, ns, edge], -1)
                packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu().clamp(min=1),
                                                           batch_first=True, enforce_sorted=False)
                _, (h, _) = s.lstm(packed)
                return s.head(h[-1])

        class _AttnH(nn.Module):
            # attention-over-neighbour-history: query token attends over K past-neighbour tokens
            def __init__(s, heads, layers):
                super().__init__()
                s.tok = nn.Linear(fdim + S + 2, H)
                s.q = nn.Linear(fdim, H)
                enc = nn.TransformerEncoderLayer(H, heads, H * 2, batch_first=True, activation="gelu")
                s.tr = nn.TransformerEncoder(enc, layers)
                s.head = nn.Linear(H, out_dim)
            def forward(s, nf, ns, edge, mask):
                x = s.tok(torch.cat([nf, ns, edge], -1))              # [B,K,H]
                pad = ~mask.bool()                                    # True where padded
                x = s.tr(x, src_key_padding_mask=pad)
                x = x.masked_fill(pad.unsqueeze(-1), 0.0)
                pooled = x.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
                return s.head(pooled)

        loss_fn = lambda p, y: F.smooth_l1_loss(p, y)
        def _train(model, fwd):
            opt = torch.optim.Adam(model.parameters(), lr=a.lr)
            model.train()
            for _ in range(a.steps):
                sidx = torch.randint(0, Btr, (bs,), device=dev)
                loss = loss_fn(fwd(model, sidx), ytr[sidx])
                opt.zero_grad(); loss.backward(); opt.step()
            model.eval()

        want = set(x for x in a.prop_arch.split(",") if x)
        results = {}
        # static floor (shared)
        sh = _StaticH().to(dev); _train(sh, lambda m, s: m(qftr[s]))
        with torch.no_grad():
            results["static"] = _reg_skill(sh(qfte), yte, yte)
        arch_defs = {
            "lstm1": ("deepLSTM-1L", lambda: _DeepLSTM(1), "seq"),
            "lstm2": ("deepLSTM-2L", lambda: _DeepLSTM(2), "seq"),
            "lstm3": ("deepLSTM-3L", lambda: _DeepLSTM(3), "seq"),
            "lstm4": ("deepLSTM-4L", lambda: _DeepLSTM(4), "seq"),
            "attn":  ("attn-hist",   lambda: _AttnH(a.prop_attn_heads, a.prop_attn_layers), "attn"),
            "mv":    ("deepLSTM-2L", lambda: _DeepLSTM(2), "seq"),
        }
        for key in ("lstm1", "lstm2", "lstm3", "lstm4", "attn", "mv"):
            if key not in want: continue
            nm, ctor, kind = arch_defs[key]
            model = ctor().to(dev)
            if kind == "seq":
                _train(model, lambda m, s: m(nftr[s], nstr[s], etr[s], ltr[s]))
                with torch.no_grad():
                    results[key] = _reg_skill(model(nfte, nste, ete, lte), yte, yte)
            else:
                _train(model, lambda m, s: m(nftr[s], nstr[s], etr[s], mtr[s]))
                with torch.no_grad():
                    results[key] = _reg_skill(model(nfte, nste, ete, mte), yte, yte)

        _tgtn = "ABUND-DELTA(dlog)" if a.abund_delta else "ABUND-LEVEL(log-count)"
        _mv = " MULTIVAR-nstate[abund|doy|occ]" if a.abund_multivar else ""
        s_mae, s_r2 = results["static"]
        print(f"  static-floor        | MAE {s_mae:7.3f}  absR2 {s_r2:+.4f}")
        for key in ("lstm1", "lstm2", "lstm3", "lstm4", "attn", "mv"):
            if key in results and key != "static":
                nm = arch_defs[key][0]
                mae, r2 = results[key]
                print(f"  {nm:<18}| MAE {mae:7.3f}  absR2 {r2:+.4f}  (dR2 vs static {r2 - s_r2:+.4f})")
        dt = time.time() - t0
        print(f"  {len(lat_a)} obs, {a.steps}-step prop-arch in {dt:.1f}s")
        declare(
            capability="", mode=f"PROPAGATOR-ARCH({_tgtn})", metric="absR2", value=s_r2,
            diagnostic=True,
            diagnostic_reason="propagator-architecture comparison; " + RAW_PE_REASON,
            obs=len(lat_a), queries=n_te, K=K, hidden=H, seconds=dt,
            static_mae=s_mae, prop_arch=a.prop_arch,
        )
        return {"abund_prop_arch": True, "target": _tgtn, "static_absR2": s_r2,
                "results": {k: {"mae": v[0], "absR2": v[1]} for k, v in results.items()},
                "abund_lead": a.abund_lead, "abund_win": a.abund_win, "abund_delta": a.abund_delta,
                "multivar": a.abund_multivar, "K": K, "n_te": n_te, "seconds": dt}

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
            for ft in ("raw", "rff", "e4d"):
                d = r[ft]
                print(f"  {ft:>4} | static {unit} {d['static_mae']:7.3f} acc/R2 {d['static_acc']:+.4f} -> GNN {unit} {d['gnn_mae']:7.3f} acc/R2 {d['gnn_acc']:+.4f} (prop {d['static_mae']-d['gnn_mae']:+.3f}) | LSTM {unit} {d['lstm_mae']:7.3f} acc/R2 {d['lstm_acc']:+.4f} (prop {d['static_mae']-d['lstm_mae']:+.3f})")
            print(f"  BEST propagator_gain (raw features, {unit} reduction; POSITIVE=propagation helps) GNN {g_raw_gnn:+.3f}  LSTM {g_raw_lstm:+.3f}  best {best:+.3f}")
            print(f"  ENCODER control (GNN {unit} reduction vs static, per PE): raw {g_raw_gnn:+.3f} | RFF {g_rff_gnn:+.3f} | Earth4D {g_e4d_gnn:+.3f}  (Earth4D-vs-raw GNN {unit} {r['raw']['gnn_mae']-r['e4d']['gnn_mae']:+.3f}: +=E4D better)")
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
            sp_all = np.concatenate(_sp).astype(np.int64)[obs_index]
            from deepearth.autoresearch.programs.spacetime.dyntargets import run_first_arrival_all
            r = run_first_arrival_all(e4d_sp, rff_sp, raw_sp, fd, days, coords_ll, sp_all, test, dev,
                                      feats=tuple(x for x in a.pheno_feats.split(",") if x),
                                      K=a.rec_k, steps=a.steps, lr=a.lr, hidden=a.rec_hidden, hops=a.gnn_hops, tol_days=a.pheno_tol)
            res = _report("FIRST_ARRIVAL(onset-DOY)", "MAEd", r, f"tol=+/-{a.pheno_tol:.0f}d")
            dt = time.time() - t0
            print(f"  {len(lat)} obs, {a.steps}-step first-arrival in {dt:.1f}s")
            declare(
                capability="", mode="FIRST-ARRIVAL(onset-DOY)", metric="MAE",
                value=res.get("static_mae_raw", float("nan")),
                diagnostic=True,
                diagnostic_reason="first-arrival is not a scorecard capability; " + RAW_PE_REASON,
                obs=len(lat), seconds=dt, win=a.abund_win, lead=a.abund_lead,
            )
            return res | {"seconds": dt, "first_arrival": True}

        if a.abundance:
            from deepearth.autoresearch.programs.spacetime.dyntargets import run_abundance_all
            r = run_abundance_all(e4d_sp, rff_sp, raw_sp, fd, days, coords_ll, test, dev,
                                  feats=tuple(x for x in a.pheno_feats.split(",") if x),
                                  K=a.rec_k, steps=a.steps, lr=a.lr, hidden=a.rec_hidden, hops=a.gnn_hops, win=a.abund_win, lead=a.abund_lead, delta=a.abund_delta)
            _nm = "ABUNDANCE-DELTA(dlog)" if a.abund_delta else "ABUNDANCE(log-count)"
            res = _report(_nm, "MAE", r, f"win={a.abund_win:.0f}d lead={a.abund_lead:.0f}d delta={a.abund_delta}")
            res = res | {"abund_lead": a.abund_lead, "abund_win": a.abund_win, "abund_delta": a.abund_delta}
            dt = time.time() - t0
            print(f"  {len(lat)} obs, {a.steps}-step abundance in {dt:.1f}s")
            declare(
                capability="", mode=_nm,   # _nm already reads "ABUNDANCE(...)"
                metric="MAE",
                value=res.get("static_mae_raw", float("nan")),
                diagnostic=True,
                diagnostic_reason="abundance is not a scorecard capability; " + RAW_PE_REASON,
                obs=len(lat), seconds=dt, win=a.abund_win, lead=a.abund_lead, delta=a.abund_delta,
            )
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
        print(f"  ABSOLUTE top1 | raw: static {raw_r['static_acc']:.4f} -> GNN {raw_r['gnn_acc']:.4f} (prop {pg_raw:+.4f}) | RFF: static {rff_r['static_acc']:.4f} -> GNN {rff_r['gnn_acc']:.4f} (prop {pg_rff:+.4f}) | E4D: static {e4d_r['static_acc']:.4f} -> GNN {e4d_r['gnn_acc']:.4f} (prop {pg_e4d:+.4f})")
        print(f"  ABSOLUTE top5 | raw: static {raw_r['static_top5']:.4f} -> GNN {raw_r['gnn_top5']:.4f} (prop {pg_raw5:+.4f}) | RFF: static {rff_r['static_top5']:.4f} -> GNN {rff_r['gnn_top5']:.4f} | E4D: static {e4d_r['static_top5']:.4f} -> GNN {e4d_r['gnn_top5']:.4f}")
        print(f"  BEST propagator_gain (raw features: GNN vs its no-prop floor) top1 {pg_raw:+.4f}  top5 {pg_raw5:+.4f}   (does causal propagation forecast forward?)")
        print(f"  {len(lat)} obs, {a.steps}-step GNN in {dt:.1f}s")
        # The mechanism (GNN propagation) and the encoder are separate questions. propagator_gain
        # measures propagation-vs-static; the ENCODER gain is Earth4D's GNN accuracy against the
        # generic PE's GNN accuracy. Declaring both keeps them from being confused for one another.
        declare(
            capability="family_from_spacetime",
            mode="GNN(message-passing propagator)",
            metric="family_top1_accuracy",
            value=e4d_r["gnn_acc"],
            split="FORECAST(past->future)",
            gains={"GNN Earth4D vs RFF": e4d_r["gnn_acc"] - rff_r["gnn_acc"],
                   "GNN Earth4D vs raw": e4d_r["gnn_acc"] - raw_r["gnn_acc"]},
            baselines={"raw_gnn": raw_r["gnn_acc"], "RFF_gnn": rff_r["gnn_acc"],
                       "raw_static": raw_r["static_acc"]},
            obs=len(lat), forecast_queries=n_te, K=a.rec_k, hops=a.gnn_hops,
            earth4d_dim=int(e4d.shape[1]), seconds=dt,
            propagator_gain_raw=pg_raw, propagator_gain_e4d=pg_e4d, propagator_gain_rff=pg_rff,
        )
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
        print(f"  {len(lat)} obs, {a.steps}-step rollout in {dt:.1f}s")
        declare(
            capability="family_from_spacetime",
            mode="RECURRENCE(4D-LSTM rollout past->future)",
            metric="family_top1_accuracy",
            value=e4d_acc,
            split="FORECAST(past->future)",
            gains={"vs raw": e4d_acc - raw_acc, "vs RFF": e4d_acc - rff_acc},
            baselines={"raw": raw_acc, "RFF": rff_acc},
            obs=len(lat), rollout_queries=n_te, families=n_fam, K=a.rec_k, hidden=a.rec_hidden,
            earth4d_dim=int(e4d.shape[1]), seconds=dt,
            top5={"raw": raw_t5, "rff": rff_t5, "earth4d": e4d_t5},
        )
        return {"st_gain": e4d_acc - raw_acc, "st_gain_rff": e4d_acc - rff_acc, "earth4d_acc": e4d_acc,
                "raw_acc": raw_acc, "rff_acc": rff_acc, "obs": len(lat), "seconds": dt, "recurrence": True}

    raw_acc, raw_t5 = evaluate(raw, fam_t, test, n_fam, dev, a.steps, a.lr, "raw", a.head_hidden, a.seed)
    rff_acc, rff_t5 = evaluate(rff, fam_t, test, n_fam, dev, a.steps, a.lr, "rff", a.head_hidden, a.seed)
    e4d_acc, e4d_t5 = (evaluate_trainable(enc, coords, fam_t, test, n_fam, dev, a.steps, a.lr, "earth4d",
                                          a.head_hidden, a.enc_lr_mult, a.enc_warmup, a.enc_c2f, seed=a.seed)
                       if a.train_encoder else
                       evaluate(e4d, fam_t, test, n_fam, dev, a.steps, a.lr, "earth4d", a.head_hidden, a.seed))
    dt = time.time() - t0
    mode = ("FORECAST(future+newplace)" if a.forecast_spatial else "FORECAST(past->future)") if a.forecast else "spatial-block"
    print(f"  {len(lat)} obs, {a.steps}-step probe in {dt:.1f}s")
    # The shared coordinate/forecast tail. --target selects WHICH capability this is; the old harness
    # could not tell family_from_spacetime from species_from_spacetime here because both print the same
    # header and both were matched by the same r"\bEarth4D\s+([\d.]+)" pattern.
    _target_capability = ("species_from_spacetime" if a.target == "species" else "family_from_spacetime")
    declare(
        capability=_target_capability,
        mode=mode,
        metric=f"{a.target}_top1_accuracy",
        value=e4d_acc,
        split=mode,
        gains={"vs raw": e4d_acc - raw_acc, "vs RFF": e4d_acc - rff_acc},
        baselines={"raw": raw_acc, "RFF": rff_acc},
        obs=len(lat), held_out=int(test.sum()), n_classes=n_fam, earth4d_dim=int(e4d.shape[1]),
        seconds=dt, target=a.target,
        top5={"raw": raw_t5, "rff": rff_t5, "earth4d": e4d_t5},
    )
    return {"st_gain": e4d_acc - raw_acc, "st_gain_rff": e4d_acc - rff_acc, "earth4d_acc": e4d_acc, "raw_acc": raw_acc,
            "rff_acc": rff_acc, "obs": len(lat), "seconds": dt, "forecast": a.forecast}



# ============================================================================================
# LOOP-spacetime: ENV-DERIVABLE CONSTRUCT test (flag-gated, default-off, additive).
# Question: cat_rarity / cat_ease_of_care categorical LABELS are degenerate proxies (rarity floor
# 0.885, ease 0.772). But rarity and ease-of-care are NICHE/SPATIAL constructs. Does the on-box
# occurrence+env data carry them DIRECTLY, via RANGE-SIZE (rarity) and CLIMATE-TOLERANCE BREADTH
# (ease)? If yes -> route rarity/ease to spacetime/env encoder (range/breadth features), not phylo.
# Leak-guard: held-out species; a species' features come only from ITS OWN occurrences; a species'
# own label is never a feature; label-shuffle null reported.
# ============================================================================================
def _range_features_per_species(cache, S):
    """Per-species RANGE-SIZE features from the 621k occurrences (gbif_tokens lat/lon/species_local,
    the SAME 2141-vocab join used by --cooccur). Returns dict name->[S] float32 (raw, un-z-scored):
      n_obs             total occurrence count
      n_cells_05        # distinct occupied 0.5deg cells (ECOLOGICAL RANGE SIZE)
      n_cells_10        # distinct occupied 1.0deg cells (coarser range)
      lat_span/lon_span geographic extent (max-min) in degrees
      hull_area         convex-hull area of occurrence points (deg^2 geographic extent)
      max_gc            max pairwise great-circle-ish extent (deg) = range diameter proxy
    """
    import numpy as _np, glob as _glob
    from pathlib import Path as _P
    lats = [[] for _ in range(S)]; lons = [[] for _ in range(S)]
    for f in sorted(_glob.glob(str(_P(cache) / "gbif_tokens/*.npz"))):
        z = _np.load(f); sl = z["species_local"].astype(_np.int64); la = z["lat"]; lo = z["lon"]
        for s, a, b in zip(sl, la, lo):
            s = int(s)
            if 0 <= s < S:
                lats[s].append(float(a)); lons[s].append(float(b))
    n_obs = _np.zeros(S, _np.float32); n05 = _np.full(S, _np.nan, _np.float32); n10 = _np.full(S, _np.nan, _np.float32)
    lat_span = _np.full(S, _np.nan, _np.float32); lon_span = _np.full(S, _np.nan, _np.float32)
    hull = _np.full(S, _np.nan, _np.float32); maxgc = _np.full(S, _np.nan, _np.float32)
    try:
        from scipy.spatial import ConvexHull as _CH
    except Exception:
        _CH = None
    for s in range(S):
        if not lats[s]:
            continue
        la = _np.asarray(lats[s]); lo = _np.asarray(lons[s]); n_obs[s] = len(la)
        n05[s] = len(set(zip(_np.floor(la / 0.5).astype(int).tolist(), _np.floor(lo / 0.5).astype(int).tolist())))
        n10[s] = len(set(zip(_np.floor(la / 1.0).astype(int).tolist(), _np.floor(lo / 1.0).astype(int).tolist())))
        lat_span[s] = la.max() - la.min(); lon_span[s] = lo.max() - lo.min()
        maxgc[s] = _np.hypot(lat_span[s], lon_span[s])
        pts = _np.stack([la, lo], 1)
        if _CH is not None and len(_np.unique(pts, axis=0)) >= 3:
            try: hull[s] = _CH(pts).volume  # 2D volume == area
            except Exception: hull[s] = 0.0
        else:
            hull[s] = 0.0
    return {"n_obs": n_obs, "n_cells_05": n05, "n_cells_10": n10, "lat_span": lat_span,
            "lon_span": lon_span, "hull_area": hull, "max_gc": maxgc}



def _niche_breadth_features_per_species(cache, S):
    """Nature-derived MULTIVARIATE niche-BREADTH (tolerance) descriptor per species, from ITS OWN
    occurrences + on-box env only (NO human labels). Returns dict name->[S] float32 (raw):
      elev_span     elevation range (max-min, m) occupied  -> topographic tolerance
      elev_iqr      elevation p90-p10 (robust vertical breadth)
      lat_span_abs  latitudinal extent (deg)                -> thermal-latitude tolerance
      lon_span      longitudinal extent (deg)
      hyper_logdet  log-det of the (wc19+ae64) env covariance = climate-envelope HYPERVOLUME
      hyper_trace   trace of that covariance = total per-channel variance (additive niche width)
      ae_effrank    AlphaEarth-64 covariance effective rank (participation ratio) = HABITAT DIVERSITY
      ae_meanpair   mean pairwise L2 among a species' AlphaEarth vectors (habitat spread, subsampled)
    Leak-safe: purely a species' own occurrence env distribution."""
    import numpy as _np, glob as _glob
    from pathlib import Path as _P
    cachep = _P(cache)
    wc = _np.load(cachep / "gbif_worldclim_tokens.npz"); wcm = {int(g): i for i, g in enumerate(wc["gbifID"])}; WC = wc["worldclim"]
    ae = _np.load(cachep / "gbif_alphaearth_tokens.npz"); aem = {int(g): i for i, g in enumerate(ae["gbifID"])}; AE = ae["ae"]
    el = _np.load(cachep / "gbif_elev.npz"); elm = {int(g): float(v) for g, v in zip(el["gbifID"], el["elev"])}
    # gather per-species: env rows (wc+ae, z-scored per-channel globally so det/trace comparable), lat, elev
    # global per-channel standardization of wc+ae so hypervolume is unit-free & channels comparable
    D = 19 + 64
    WCz = (WC - _np.nanmean(WC, 0)) / (_np.nanstd(WC, 0) + 1e-6)
    AEz = (AE - _np.nanmean(AE, 0)) / (_np.nanstd(AE, 0) + 1e-6)
    rows = [[] for _ in range(S)]; aerows = [[] for _ in range(S)]
    lats = [[] for _ in range(S)]; lons = [[] for _ in range(S)]; elevs = [[] for _ in range(S)]
    for f in sorted(_glob.glob(str(cachep / "gbif_tokens/*.npz"))):
        z = _np.load(f); sl = z["species_local"].astype(_np.int64); gid = z["gbifID"]; la = z["lat"]; lo = z["lon"]
        for s, g, a, b in zip(sl, gid, la, lo):
            s = int(s); g = int(g)
            if not (0 <= s < S):
                continue
            lats[s].append(float(a)); lons[s].append(float(b))
            if g in elm:
                elevs[s].append(elm[g])
            if g in wcm and g in aem:
                v = _np.empty(D, _np.float32); v[:19] = WCz[wcm[g]]; v[19:] = AEz[aem[g]]
                rows[s].append(v); aerows[s].append(AEz[aem[g]])
    elev_span = _np.full(S, _np.nan, _np.float32); elev_iqr = _np.full(S, _np.nan, _np.float32)
    lat_span_abs = _np.full(S, _np.nan, _np.float32); lon_span = _np.full(S, _np.nan, _np.float32)
    hyper_logdet = _np.full(S, _np.nan, _np.float32); hyper_trace = _np.full(S, _np.nan, _np.float32)
    ae_effrank = _np.full(S, _np.nan, _np.float32); ae_meanpair = _np.full(S, _np.nan, _np.float32)
    _rng = _np.random.RandomState(0)
    for s in range(S):
        if lats[s]:
            la = _np.asarray(lats[s]); lo = _np.asarray(lons[s])
            lat_span_abs[s] = la.max() - la.min(); lon_span[s] = lo.max() - lo.min()
        if elevs[s]:
            ev = _np.asarray(elevs[s]); ev = ev[~_np.isnan(ev)]
            if ev.size:
                elev_span[s] = float(ev.max() - ev.min())
                elev_iqr[s] = float(_np.percentile(ev, 90) - _np.percentile(ev, 10))
        r = rows[s]
        if len(r) >= 3:
            M = _np.stack(r, 0)
            M = M[~_np.isnan(M).any(1)]
            if M.shape[0] >= D + 1:
                C = _np.cov(M, rowvar=False)
                ev = _np.linalg.eigvalsh(C); ev = _np.clip(ev, 1e-8, None)
                hyper_logdet[s] = float(_np.log(ev).sum())        # env-envelope hypervolume (log)
                hyper_trace[s] = float(_np.trace(C))
            elif M.shape[0] >= 3:
                hyper_trace[s] = float(_np.var(M, 0).sum())        # trace still defined w/o full-rank cov
        a = aerows[s]
        if len(a) >= 3:
            A = _np.stack(a, 0); A = A[~_np.isnan(A).any(1)]
            if A.shape[0] >= 3:
                Ca = _np.cov(A, rowvar=False); eva = _np.clip(_np.linalg.eigvalsh(Ca), 0, None)
                ssum = eva.sum()
                if ssum > 1e-9:
                    ae_effrank[s] = float((ssum ** 2) / (_np.square(eva).sum() + 1e-12))  # participation ratio
                idx = _rng.permutation(A.shape[0])[:min(60, A.shape[0])]
                As = A[idx]
                dif = As[:, None, :] - As[None, :, :]
                dm = _np.sqrt(_np.square(dif).sum(-1))
                iu = _np.triu_indices(As.shape[0], 1)
                if iu[0].size:
                    ae_meanpair[s] = float(dm[iu].mean())
    return {"elev_span": elev_span, "elev_iqr": elev_iqr, "lat_span_abs": lat_span_abs,
            "lon_span": lon_span, "hyper_logdet": hyper_logdet, "hyper_trace": hyper_trace,
            "ae_effrank": ae_effrank, "ae_meanpair": ae_meanpair}


def env_construct(cache, seed=0, construct="rarity", feature="range", holdout=0.3, shuffle=False, only=""):
    """Test whether a construct label is predictable from occurrence/env-derived features on held-out species.

    construct: 'rarity' (cat_rarity ordinal) or 'ease' (cat_ease_of_care ordinal). Only LABELED species
              (raw label>0; 0 == unlabeled/missing) enter the supervised task.
    feature:  'range'   -> range-size features (n_cells_05/10, n_obs, spans, hull, max_gc)  [log1p heavy-tailed]
              'breadth' -> per-species env-tolerance breadth (worldclim+alphaearth std/p10-p90/min-max)
              'both'    -> concat
    Reports, on held-out species: balanced accuracy vs a MAJORITY floor, Spearman(feature-score, ordinal),
    per-feature univariate Spearman (channel decomposition), and a label-shuffle null (same pipeline, y permuted).
    """
    import numpy as _np
    from pathlib import Path as _P
    from scipy.stats import spearmanr as _sp
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.metrics import balanced_accuracy_score as _bacc
    import sys as _sys; _sys.path.insert(0, "/workspace")
    from deepearth.autoresearch.programs.biological.probe import load_trait as _load_trait

    vocab = _np.load(_P(cache) / "gbif_vocab.npz", allow_pickle=True); gidx = vocab["global_idx"]; S = len(gidx)
    if construct in ("ns_grank", "crpr"):
        # AUTHORITATIVE EXTERNAL rarity ordinal from derived/species_rarity.jsonl (idx aligned to 2141 vocab).
        import json as _json, re as _re
        rows = [_json.loads(l) for l in open(_P(cache) / "derived/species_rarity.jsonl")]
        yord = _np.full(S, -1, _np.int64)
        for r in rows:
            ii = int(r["idx"])
            if not (0 <= ii < S):
                continue
            if construct == "ns_grank":  # NatureServe Global rank G1(rarest)..G5(secure) -> rarity ordinal 4..0
                m = _re.match(r"G(\d)", r.get("ns_g_rank", "") or "")
                if m:
                    yord[ii] = 5 - int(m.group(1))   # G1->4 (rarest), G5->0 (common)
            else:                          # CNPS Rare Plant Rank ordinal (0 = not-ranked/common .. higher = rarer)
                yord[ii] = int(r.get("crpr_ordinal", 0))
        labeled = yord >= 0
    else:
        key = "cat_rarity" if construct == "rarity" else "cat_ease_of_care"
        _, yt, _, _ = _load_trait(cache, gidx, key, "cpu")
        yraw = yt.cpu().numpy().astype(_np.int64)
        # ordinal maps: raw label -> rarity/difficulty ordinal (higher = rarer / harder). 0 == missing.
        if construct == "rarity":  # vocab [Abundant,Common,Rare,Uncommon]; raw = vocab_idx+1
            ordmap = {1: 0, 2: 1, 4: 2, 3: 3}   # Abundant<Common<Uncommon<Rare
        else:                       # vocab [Challenging,Easy,Moderate]; raw = vocab_idx+1; higher=harder
            ordmap = {2: 0, 3: 1, 1: 2}         # Easy<Moderate<Challenging
        labeled = _np.array([v in ordmap for v in yraw])
        yord = _np.array([ordmap.get(v, -1) for v in yraw], _np.int64)

    # ---- feature construction ----
    feats = {}
    if feature in ("range", "both"):
        rf = _range_features_per_species(cache, S)
        for k, v in rf.items():
            vv = v.copy()
            if k in ("n_obs", "n_cells_05", "n_cells_10", "hull_area"):
                vv = _np.log1p(_np.nan_to_num(vv, nan=0.0))
            feats[k] = vv
    if feature in ("breadth", "both", "nichebreadth_env", "allbreadth"):
        emean, emedoid, npsp, estd, elo, ehi, emin, emax, etime, epheno = load_env_species(cache, extra_channels=False, temporal=False)
        # per-species scalar breadth summaries per channel-group (worldclim 0:19, alphaearth 19:83)
        iqr = ehi - elo
        feats["breadth_wc_std"]  = _np.nanmean(estd[:, :19], 1)
        feats["breadth_ae_std"]  = _np.nanmean(estd[:, 19:83], 1)
        feats["breadth_wc_iqr"]  = _np.nanmean(iqr[:, :19], 1)
        feats["breadth_ae_iqr"]  = _np.nanmean(iqr[:, 19:83], 1)
        feats["breadth_wc_range"] = _np.nanmean((emax - emin)[:, :19], 1)
        feats["breadth_ae_range"] = _np.nanmean((emax - emin)[:, 19:83], 1)
    if feature in ("nichebreadth", "allbreadth", "nichebreadth_env"):
        nb = _niche_breadth_features_per_species(cache, S)
        for k, v in nb.items():
            vv = v.copy()
            if k in ("elev_span", "elev_iqr"):
                vv = _np.log1p(_np.nan_to_num(vv, nan=0.0))   # heavy-tailed elevation
            feats[k] = vv
    if feature == "allbreadth":
        rf = _range_features_per_species(cache, S)
        for k, v in rf.items():
            vv = v.copy()
            if k in ("n_obs", "n_cells_05", "n_cells_10", "hull_area"):
                vv = _np.log1p(_np.nan_to_num(vv, nan=0.0))
            feats[k] = vv

    if only:
        keep=[k for k in feats if any(t in k for t in only.split(","))]
        feats={k:feats[k] for k in keep}
    fnames = list(feats.keys())
    X = _np.stack([feats[k] for k in fnames], 1).astype(_np.float32)
    # a species with zero occurrences has all-nan features -> drop from labeled set (no self-data)
    have = ~_np.isnan(X).any(1)
    use = labeled & have
    Xu = X[use]; yu = yord[use]
    # z-score features over used species
    mu = Xu.mean(0); sd = Xu.std(0); sd[sd < 1e-9] = 1.0; Xz = (Xu - mu) / sd
    n = len(yu)

    rng = _np.random.RandomState(seed)
    if shuffle:
        yu = yu[rng.permutation(n)]
    perm = rng.permutation(n); ncut = int(round((1 - holdout) * n))
    tr = perm[:ncut]; te = perm[ncut:]
    # majority floor on the held-out split
    vals, cts = _np.unique(yu[tr], return_counts=True); maj = vals[cts.argmax()]
    floor_acc = float((yu[te] == maj).mean())
    floor_bacc = float(_bacc(yu[te], _np.full(len(te), maj)))

    # multinomial logistic (balanced) on train features -> held-out
    clf = _LR(max_iter=2000, class_weight="balanced", C=1.0).fit(Xz[tr], yu[tr])
    pred = clf.predict(Xz[te])
    acc = float((pred == yu[te]).mean()); bacc = float(_bacc(yu[te], pred))
    # ordinal signal: Spearman between a 1-D risk score (LR decision projected to ordinal expectation) and truth
    proba = clf.predict_proba(Xz[te])
    classes = clf.classes_.astype(_np.float32)
    score = (proba * classes[None, :]).sum(1)  # expected ordinal
    rho = _sp(yu[te], score).correlation
    rho = float(rho if rho == rho else 0.0)

    # per-feature univariate Spearman over ALL used labeled species (channel decomposition)
    uni = {}
    for j, name in enumerate(fnames):
        r = _sp(yu, Xz[:, j]).correlation
        uni[name] = round(float(r if r == r else 0.0), 3)

    return {"construct": construct, "feature": feature, "n_labeled_used": int(n), "n_classes": int(len(vals)),
            "held_out": int(len(te)), "floor_acc": round(floor_acc, 4), "floor_bacc": round(floor_bacc, 4),
            "acc": round(acc, 4), "bacc": round(bacc, 4), "spearman_ord": round(rho, 4),
            "univar_spearman": dict(sorted(uni.items(), key=lambda kv: -abs(kv[1]))),
            "shuffle_null": shuffle, "seed": seed}


if __name__ == "__main__":
    main()
