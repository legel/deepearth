"""The Earth4D probe — the instrument this loop edits.

One capability at a time, measured against fair baselines in minutes. `harness.py` decides what a
number means; this file decides what gets computed.

THE LEVERS. Two families, and the fair-gain tells you which one you are on:

  fair-gain ~ 0 or negative  →  INPUT-limited   →  DATA lever: change the channel
  fair-gain positive, score low →  ENCODER-limited →  ARCHITECTURE lever: change the mechanism

  DATA ─────────────────────────────────────────────────────────────────────────────────
    --env --env_channels {all,worldclim,alphaearth,wcsoil,...}   which environment
    --env_extra                                                 + soil and elevation
    --sdm_channels ...                                          channels for the SDM modes
    --cooccur_channels ...                                      channels for co-occurrence
    --vision --vision_feats {dino,bio,both}                     borrowed morphology (LABEL IT)
    --pheno_channel                                             remote-sensing phenology
    new sources: editable_files/data/, loaders in editable_files/lib/

  ARCHITECTURE ─────────────────────────────────────────────────────────────────────────
    mechanism      --recurrence [--rec_k --rec_hidden --rec_time_cond]   4D-LSTM rollout
                   --gnn [--gnn_hops]                                   message passing
                   --field_decode                                       dense-field decode
                   --env_decode [--env_aux_weight]                      env-supervised field
    priors         --fourier[_scale] --spatial_siren[--siren_*] --spatial_cline[--cline_scale]
                   --time_harmonics --time_film --causal_lags[--causal_lag_span]
    encoder itself autoresearch/probes/spacetime/editable_files/earth4d.py  ← the real architecture lever
    end-to-end     --train_encoder [--enc_lr_mult --enc_warmup --enc_c2f]

  CAPACITY (tunes a winner; not a move) ────────────────────────────────────────────────
    --spatial_levels --temporal_levels --log2_hashmap --head_hidden --steps --lr

  WHAT IS MEASURED (identity — changing these makes a DIFFERENT measurement, not a better score)
    --forecast [--forecast_spatial] · --target {family,species} · --phenology · --sdm_presence
    --sdm_hard · --cooccur · --n_shards · --holdout · --seed

An experiment is an EDIT to this file or to earth4d.py, on a branch. Not a new flag: a flag is what a
change earns when it GRADUATES, and a dead flag is a bug. Gating at conception is what grew this file
to 113 flags and 19 modes; the diagnostics that could never set a record have been deleted.
"""



PROBE_MODULE = "deepearth.autoresearch.probes.spacetime.editable_files.probe"
# Must match autoresearch/probes/spacetime/editable_files/harness.py PROTOCOL. Bump both when a change alters what a run MEASURES.
PROTOCOL_VERSION = "v3-fairbaseline"

_TRACE_AUTHORIZED = False
if __name__ == "__main__":
    import sys as _entry_sys
    from deepearth.autoresearch.probes.spacetime.editable_files.lib.recurrence import (
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
from dataclasses import dataclass
import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepearth.autoresearch.probes.spacetime.editable_files.earth4d import Earth4D
from deepearth.autoresearch.probes.spacetime.editable_files.lib.fair_baseline import (
    _rff_features,
    fair_rff,
)
from deepearth.autoresearch.probes.spacetime.editable_files.harness import (
    _set_result_sink,
    declare,
)

# ===================================================================================================
# CONFIG — THE EXPERIMENT. Edit this, on a branch. There is no command-line flag for any of it.
# ===================================================================================================
#
# These 33 values used to be CLI flags. That was the problem: when every lever is a flag, every
# "experiment" is a toggle, and the code takes the shape of its toggles -- 113 flags, 19 modes, a
# 1,552-line main(), and gated-default-off branches nobody ever removed. A flag is also a promise that
# the default path still works, which is why nothing was ever deleted.
#
# Now an experiment is a DIFF of this block (and of earth4d.py). Change a value, commit it on a branch,
# run it. The branch is the isolation and git is the record; if it fails, delete the branch and the
# change is gone. Nothing accumulates.
#
# The CLI keeps only what decides WHICH MEASUREMENT is being made -- capability/mode, data size, split,
# seed, device -- because those must be comparable across experiments and belong in the record's
# identity. Everything below is how the encoder is BUILT and TRAINED, which is exactly what an
# experiment is supposed to change.
#
# The identity carries config_digest (see harness.py), so two runs with different CONFIG are never
# treated as the same measurement even though their command lines are identical.
CONFIG = {
    # Starts at the STANDING CHAMPION for species_from_spacetime (0.0787, tag tilecode_cmac4). A run with no edits must
    # reproduce the record; if the defaults measured some other mode, every experiment would silently be
    # compared against a different measurement.
    "lr": 3e-3,
    "spatial_levels": 18,
    "temporal_levels": 18,
    "log2_hashmap": 20,
    "head_hidden": 512,
    "fourier": 1024,
    "fourier_scale": 6400.0,
    "time_harmonics": 8,
    "train_encoder": False,
    "enc_lr_mult": 0.05,
    "enc_warmup": 0.15,
    "enc_c2f": 0.5,
    # literal, not the imported constant: CONFIG is read at module load, before that import.
    # Must stay equal to lib/recurrence.py DEFAULT_TIME_HORIZON.
    "time_horizon": 2.0,
    # ---- WHAT IS MEASURED (these were mode-selector and data flags on the CLI) ----
    # Changing any of these changes the measurement, which is exactly why they belong in the
    # experiment rather than on a command line. config_digest makes the gate see the difference.
    "cache_dir": "autoresearch/data/deepcal",
    "n_shards": 12,
    "steps": 800,
    "holdout": 0.2,
    "target": "species",
    "forecast": True,
    "forecast_spatial": False,
    "recurrence": False,
    "gnn": False,
    "phenology": True,
    "pheno_nofair": False,
    "pheno_feats": "e4d,rff,raw",
    "pheno_spatial": False,
    "field_decode": False,
    "env": False,
    "env_decode": False,
    "env_extra": False,
    "env_channels": "all",
    "vision": False,
    "vision_feats": "dino",
    "pheno_channel": False,
    "cooccur": False,
    "cooccur_mech": "env",
    "cooccur_thresh": 2,
    "cooccur_file": "cooccur_count_005.npy",
    "cooccur_channels": "all",
    "sdm_presence": False,
    "sdm_hard": False,
    "sdm_cell_deg": 0.1,
    "sdm_holdout_mode": "block",
    "sdm_block_deg": 2.0,
    "sdm_channels": "all",
    "sdm_time": False,
    "sdm_seeds": 1,
    "causal_lags": 0,
    "causal_lag_span": 0.25,
    "spatial_siren": 0,
    "siren_layers": 2,
    "siren_w0": 30.0,
    "spatial_cline": 64,
    "cline_scale": 1.0,
    "time_film": 0,
    "rec_k": 16,
    "rec_hidden": 256,
    "rec_time_cond": False,
    "gnn_hops": 2,
    "pheno_tol": 15.0,
    "pheno_attn": False,
    "attn_heads": 4,
    "attn_layers": 2,
    "pheno_species": False,
    "rec_block_deg": 2.0,
    "rec_fast": False,
    "env_aux_weight": 1.0,
    # ENV-CHANNEL REPRESENTATION (data lever, not head capacity): 0 = the raw standardized channel.
    # >0 appends that many frozen random Fourier features OF THE ENV VECTOR, so the LINEAR head can carve
    # a niche (a region of env space) instead of a halfspace. A trained MLP head over the same channel
    # (head_hidden=256) overfits at 800 steps and LOST (0.1340 vs 0.1423); a frozen kernel expansion with
    # a linear readout is the regularized version of the same hypothesis. Every arm that consumes env
    # (env, fused) sees the identical expansion; the coordinate baselines are untouched, so the ENV vs
    # best-coord-PE comparison stays fair.
    "env_rff": 0,
    "env_rff_scale": 1.0,
    # ---- ARCHITECTURE ARMS (earth4d.py). Default-off: the champion path is byte-identical. ----
    "seasonal_time": 0,        # 1 = space-time planes hash annual PHASE; 2 = absolute planes + seasonal planes
    "drop_spatiotemporal": False,
    "extent_fit": False,       # scale the ECEF axes to the TRAIN extent, not to the globe
    "nystrom": 0,              # RBF features against N train-drawn space-time anchors
    "conj": 0,                 # explicit space x time degree-2 sketch
    "elm": 0,                  # frozen random nonlinear expansion of the encoder's own output
    "elm_scale": 1.0,
    "stencil": 0,              # local field average of the spatial lookup instead of a point sample
    "stencil_radius": 0.002,
    "coord_shrink": 1.0,       # <1 coarsens every hash level at once (directed follow-up to extent_fit)
    "spatial_ensemble": 0,     # spend the space-time budget on three purely SPATIAL tables instead
    "whiten": False,           # PCA-whiten the encoder output (fit on train rows)
    "standardize": False,      # per-dimension mean/std of the encoder output (fit on train rows)
    "tile": 64,                 # sparse tile coding: per-level one-hot cell code of this width
    "tile_replace": False,     # tile coding REPLACES the hash blocks instead of joining them
    "tile_time": False,        # tile the SPACE-TIME cell rather than the spatial cell
    "tile_offsets": 4,         # CMAC-style overlapping tilings per level
    "tile_quantile": False,    # equal-occupancy tiles (train empirical CDF warp)
    "geographic": False,       # hash (lat, lon, elev) directly instead of ECEF
}
@dataclass
class _DynamicsCtx:
    """What the dynamics/AR diagnostics need from main()'s loaded state."""
    a: object; dev: object; t0: float
    lat: object; lon: object; days: object; test: object
    obs_index: object = None
    load_species: object = None
    enc: object = None          # arrival/abundance compare their propagator against the encoder
    e4d: object = None

from deepearth.autoresearch.probes.spacetime.editable_files.lib.recurrence import (
    DEFAULT_TIME_HORIZON,
    normalize_forecast_time,
    normalize_time_from_train,
    phenology_feature_set,
    phenology_mode,
    strict_spatiotemporal_masks,
    require_recorded_entrypoint,
)













# Per-capability CONFIG. CONFIG above holds the defaults; these override it for the capability the
# harness declared, and are applied before anything runs.
#
# Without this, CONFIG starts at ONE capability's champion and every other capability's control is
# silently wrong. Measured: with CONFIG at the species_from_spacetime champion, a flowering_peak_month
# control scored 0.00285 against a standing record of 0.0521 — a 20x discrepancy that looks like a
# catastrophic regression and is really just the wrong experiment. Any sweep run that way measures
# nothing, and its arms would have been published as dead-ends that never happened.
#
# A run with no edits must reproduce ITS OWN capability's record. That is the invariant.
CAPABILITY_CONFIG = {
    "species_from_spacetime": {
        "forecast": True, "target": "species", "phenology": False,
        "head_hidden": 512, "fourier": 1024, "fourier_scale": 6400.0, "time_harmonics": 8,
        "spatial_cline": 64, "n_shards": 12, "tile": 64, "tile_offsets": 4,
    },
    "family_from_spacetime": {
        "forecast": True, "target": "family", "phenology": False,
        "head_hidden": 256, "fourier": 1024, "time_harmonics": 8, "n_shards": 12,
    },
    # The standing record (0.0521, tag v2_exact_migration_phenology) was set by the PRE-REFACTOR CLI as
    #     --phenology --forecast --pheno_env --pheno_feats e4d --n_shards 12
    # and NOTHING else, so every other lever took that CLI's argparse DEFAULT -- not the values CONFIG
    # now carries, which are the species_from_spacetime champion. (--pheno_env was one of the inert
    # flags: the `if phenology:` branch returns before any pheno_env code is reached, so the record was
    # set on the plain PHENOLOGY-FUTURE path.) Reproducing the record therefore means restoring those
    # defaults explicitly; a partial block is why the control was reading 0.0028-0.017 instead of 0.0521.
    # Verbatim from probe.py at e2b062c^ (the commit that turned the 33 flags into CONFIG).
    "flowering_peak_month": {
        "forecast": True, "phenology": True, "pheno_feats": "e4d", "pheno_nofair": False,
        "target": "species", "n_shards": 12, "steps": 800, "lr": 3e-3, "holdout": 0.2,
        "spatial_levels": 18, "temporal_levels": 18, "log2_hashmap": 20,
        "head_hidden": 0,            # old default: LINEAR head (CONFIG's 512 is the species champion)
        "fourier": 0, "fourier_scale": 10.0, "time_harmonics": 0,
        "spatial_cline": 0, "cline_scale": 1.0, "spatial_siren": 0, "time_film": 0, "causal_lags": 0,
        "tile": 0, "tile_offsets": 1, "tile_replace": False, "tile_time": False, "tile_quantile": False,
        "rec_k": 16, "rec_hidden": 256, "gnn_hops": 2, "pheno_tol": 15.0,
        "pheno_attn": False, "pheno_species": False, "pheno_spatial": False,
        "forecast_spatial": False, "recurrence": False, "gnn": False, "train_encoder": False,
    },
    # The standing record (0.1423, tag v2_baseline_famenv) was set by the PRE-REFACTOR CLI as
    #     --env --env_channels alphaearth --n_shards 12
    # and NOTHING else, so every unstated lever took that CLI's argparse DEFAULT. The reconstructed
    # block below was missing four of them, and one of those -- `target` -- silently changed WHAT IS
    # MEASURED: CONFIG's default is the species_from_spacetime champion's "species", so the env path
    # compacted sp_obs into a 1364-way SPECIES target while still declaring metric
    # family_top1_accuracy. The control scored 0.0556 against a 166-family record of 0.1423; that is a
    # different target, not a regression. Defaults verified verbatim against probe.py at e2b062c^
    # (the commit that turned the 33 flags into CONFIG): --target family, --head_hidden 0 (LINEAR
    # head), --fourier 0, --time_harmonics 0, --fourier_scale 10.0, --spatial_cline 0, --tile 0.
    "family_from_env": {
        "env": True, "env_channels": "alphaearth", "forecast": False, "phenology": False,
        "n_shards": 12, "tile": 0, "spatial_cline": 0, "fourier_scale": 10.0,
        "target": "family",          # old CLI default; CONFIG's "species" is a DIFFERENT capability
        "head_hidden": 0,            # old default: LINEAR head (CONFIG's 512 is the species champion)
        "fourier": 0, "time_harmonics": 0,
        "env_channels": "ae_wb_ph",   # ARM aewbph
    },
    "species_from_env": {
        "sdm_presence": True, "sdm_hard": True, "sdm_channels": "alphaearth", "n_shards": 16,
        "forecast": False, "phenology": False,
    },
    "community_from_env": {
        "cooccur": True, "cooccur_mech": "both", "cooccur_channels": "all", "n_shards": 12,
        "forecast": False, "phenology": False,
    },
}


def apply_capability_config(capability: str) -> None:
    """Point CONFIG at the declared capability's champion before the run starts."""
    for k, v in CAPABILITY_CONFIG.get(capability, {}).items():
        CONFIG[k] = v


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
    phenology probe, never the env->biology path.

    A `+terrain` suffix on any of the above appends the PHYSICAL-STRUCTURE stack, which has never fed the
    env->biology path at all (it exists in the corpus and only the infer_* decoders ever read it):
      topo  (12, gbif_topo_tokens)   elevation/slope/aspect/roughness terrain descriptors
      hydro ( 6, gbif_hydro_tokens)  surface-water / flow-accumulation context
      chm   (11, gbif_chm_tokens)    canopy-height structure
    e.g. alphaearth+terrain = 64 + 29 = 93, terrain = 29 alone. The claim being tested is that landform and
    vegetation STRUCTURE carry niche information that neither a spectral scene embedding nor gridded climate
    resolves -- a different physical modality, not more columns of the same one."""
    cachep = Path(cache)
    terrain = channels.endswith("+terrain")
    if terrain:
        channels = channels[: -len("+terrain")] or "none"
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
    elif channels in ("ae_wb", "ae_wb_ph"):
        # Prepared channel fusions that already exist in the corpus and have NEVER fed this path:
        #   ae_wb    = AlphaEarth ++ water-balance bands (78)
        #   ae_wb_ph = AlphaEarth ++ water balance ++ phenology (larger)
        _ae = np.load(cachep / ("gbif_ae_wb.npz" if channels == "ae_wb" else "gbif_ae_wb_ph.npz"))
        aemap = {int(g): i for i, g in enumerate(_ae["gbifID"])}; AE = _ae["ae"]
    phmap = PH = None
    if channels in ("modis", "all+modis"):
        _ph = np.load(cachep / "gbif_phenology_tokens.npz")
        phmap = {int(g): i for i, g in enumerate(_ph["gbifID"])}; PH = _ph["phenology"]
    n_ae = 0 if AE is None else AE.shape[1]
    n_ph = 0 if PH is None else PH.shape[1]
    _base = 0 if channels in ("alphaearth", "modis", "none", "ae_wb", "ae_wb_ph") else (19 if channels == "worldclim" else 29)
    D = _base + n_ae + n_ph
    env = np.full((len(gid), D), np.nan, np.float32)
    for i, g in enumerate(gid):
        g = int(g)
        o = 0
        if channels not in ("alphaearth", "modis", "none", "ae_wb", "ae_wb_ph"):
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
    if terrain:
        cols = []
        for fn, key in (("gbif_topo_tokens", "topo"), ("gbif_hydro_tokens", "hydro"),
                        ("gbif_chm_tokens", "chm")):
            z = np.load(cachep / f"{fn}.npz", allow_pickle=True)
            idx = {int(g): i for i, g in enumerate(z["gbifID"])}
            M = z[key]
            X = np.full((len(gid), M.shape[1]), np.nan, np.float32)
            for i, g in enumerate(gid):
                j = idx.get(int(g))
                if j is not None: X[i] = M[j]
            cols.append(X)
        env = np.concatenate([env] + cols, 1) if env.shape[1] else np.concatenate(cols, 1)
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
                       enc_lr_mult=0.05, warmup=0.15, c2f=0.5, clip=1.0, seed=0, side=None):
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
    # `side` = static per-observation features (the ENV channel) concatenated to the encoder output, so the
    # FUSED arm can be trained end-to-end. Without it the fused arm always reads a frozen random hash table:
    # train_encoder moved the Earth4D-alone arm 0.0938 -> 0.1105 and left the fused primary byte-identical at
    # 0.142318, because `fused` was built from the frozen features before any training happened.
    Str = Ste = None
    if side is not None:
        Str, Ste = side[train].to(dev), side[test].to(dev)
    with torch.no_grad():
        edim = enc(Ctr[:8]).shape[1]
        fdim = edim + (0 if side is None else Str.shape[1])
    head = (nn.Sequential(nn.Linear(fdim, head_hidden), nn.ReLU(), nn.Linear(head_hidden, n_fam))
            if head_hidden > 0 else nn.Linear(fdim, n_fam)).to(dev)
    opt = torch.optim.Adam([{"params": head.parameters(), "lr": lr},
                            {"params": list(enc.parameters()), "lr": lr * enc_lr_mult, "weight_decay": 0.0}])
    # per-level feature mask over the SPATIAL block (levels are contiguous, features_per_level each)
    fpl = getattr(enc, "features_per_level", 2); sdim = getattr(enc, "spatial_dim", edim)
    n_lv = max(int(sdim // max(fpl, 1)), 1)
    lvl_of = (torch.arange(edim, device=dev) // max(fpl, 1)).clamp(max=n_lv - 1)   # ENCODER dims only
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
        if Str is not None:
            f = torch.cat([f, Str[idx]], 1)
        loss = F.cross_entropy(head(f), ytr[idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), clip)
        opt.step()
    with torch.no_grad():
        moved = {n: (q - _p0[n]).norm().item() / max(_p0[n].norm().item(), 1e-9) for n, q in enc.named_parameters()}
        tot = sum(1 for v in moved.values() if v > 1e-6)
        print(f"  [train_encoder] {tot}/{len(moved)} encoder tensors moved; "
              f"rel-delta " + ", ".join(f"{n.split('.')[-1]}={v:.3g}" for n, v in list(moved.items())[:6]), flush=True)
        logits = torch.cat([
            head(enc(Cte[i:i + 8192]) if Ste is None
                 else torch.cat([enc(Cte[i:i + 8192]), Ste[i:i + 8192]], 1))
            for i in range(0, Cte.shape[0], 8192)])
        acc = (logits.argmax(-1) == yte).float().mean().item()
        top5 = (logits.topk(5, -1).indices == yte[:, None]).any(-1).float().mean().item()
    return acc, top5


def cooccur_mode(a):
    import sys as _sys; _sys.path.insert(0, '/workspace')
    from deepearth.autoresearch.probes.spacetime.editable_files.lib.dyntargets import cooccur_routing
    r = cooccur_routing(CONFIG["cache_dir"], thresh=CONFIG["cooccur_thresh"], seed=a.seed,
                        mechanism=CONFIG["cooccur_mech"], cooccur_file=CONFIG["cooccur_file"],
                        env_channels=CONFIG["cooccur_channels"])
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

def sdm_presence_mode(a):
    import sys as _sys; _sys.path.insert(0, '/workspace')
    from deepearth.autoresearch.probes.spacetime.editable_files.lib.dyntargets import sdm_presence
    r = sdm_presence(CONFIG["cache_dir"], seed=a.seed, mechanism=CONFIG["cooccur_mech"], cooccur_file=CONFIG["cooccur_file"])
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

def sdm_hard_mode(a):
    import sys as _sys; _sys.path.insert(0, '/workspace')
    from deepearth.autoresearch.probes.spacetime.editable_files.lib.dyntargets import sdm_presence_hard
    import numpy as _np
    runs = []
    for sd in range(a.seed, a.seed + CONFIG["sdm_seeds"]):
        r = sdm_presence_hard(CONFIG["cache_dir"], seed=sd, mechanism=CONFIG["cooccur_mech"],
                              cell_deg=CONFIG["sdm_cell_deg"], holdout_mode=CONFIG["sdm_holdout_mode"],
                              block_deg=CONFIG["sdm_block_deg"], env_channels=CONFIG["sdm_channels"],
                              add_time=CONFIG["sdm_time"], cooccur_file=CONFIG["cooccur_file"])
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
        sdm_seeds=CONFIG["sdm_seeds"], ap_std=float(aps.std()), gain_std=float(gns.std()),
        n_query_cells=r0["n_query_cells"], n_train_cells=r0["n_train_cells"],
        n_cand_sp=r0["n_cand_sp"], feat_dim=r0["feat_dim"], leak_guard=r0["leak_guard"],
    )
    return {'runs': runs, 'ap_mean': float(aps.mean()), 'ap_std': float(aps.std()),
            'gain_mean': float(gns.mean()), 'gain_std': float(gns.std())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    # ---- LOOP-spacetime NEW DIRECTIONS on the mean-DOY graduation target (additive, default-off) ----
    ap.add_argument("--device", default="cuda")
    # ---- LOOP-spacetime rule-1 AR ROLLOUT (this turn) ----------------------------------------------------
    # ---- LOOP-spacetime ENV-DERIVABLE CONSTRUCT test (rarity=range-size, ease=climate-breadth) ----
    # The result contract (probe_contract.py). --capability is what the harness DECLARED as its
    # objective; a mode supplies its own natural capability when the probe is run standalone. The
    # harness asserts the two agree, so a probe cannot quietly answer a different question.
    ap.add_argument("--result-json", dest="result_json", default="",
                    help="write a ProbeResult here; the harness reads this instead of parsing stdout")
    ap.add_argument("--capability", default="",
                    help="the capability the harness declared as its objective")
    a = ap.parse_args(argv)
    apply_capability_config(a.capability)
    _set_result_sink(a.result_json, a.capability, PROTOCOL_VERSION, a, config=CONFIG)
    if not _TRACE_AUTHORIZED:
        authorization_argv = sys.argv[1:] if argv is None else list(argv)
        require_recorded_entrypoint(
            "probe.py",
            module=PROBE_MODULE,
            argv=authorization_argv,
        )
    dev = a.device if torch.cuda.is_available() else "cpu"
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    # Modes that never build Earth4D: env -> identity from precomputed tables (DATA lever only).
    for _flag, _mode in ((CONFIG["cooccur"], cooccur_mode), (CONFIG["sdm_presence"], sdm_presence_mode),
                         (CONFIG["sdm_hard"], sdm_hard_mode)):
        if _flag:
            return _mode(a)





    t0 = time.time()
    need_gid = CONFIG["env"] or CONFIG["env_decode"]
    lat, lon, fam, n_fam, days, gid, sp_obs = load_obs(CONFIG["cache_dir"], CONFIG["n_shards"], with_time=CONFIG["forecast"], with_gid=need_gid)
    obs_index = np.arange(len(lat), dtype=np.int64)
    if CONFIG["forecast"]:
        valid_time = np.isfinite(days)
        if not valid_time.all():
            lat, lon, fam, days, sp_obs = (
                x[valid_time] for x in (lat, lon, fam, days, sp_obs)
            )
            if gid is not None:
                gid = gid[valid_time]
            obs_index = obs_index[valid_time]
    if CONFIG["target"] == "species":
        # CAPABILITY LEVER: the classification paths only ever predicted FAMILY, which is why
        # species_from_spacetime / species_from_env were never probeable from this probe at all. Species is a
        # strictly harder target (2141-way vocab vs 166 families) and is the capability the scorecard names.
        _u, fam = np.unique(sp_obs, return_inverse=True)         # compact the species ids actually present
        fam = fam.astype(np.int64); n_fam = int(fam.max()) + 1
    if CONFIG["forecast"]:
        # Temporal holdout is a forecast split, but direct coordinate classification is
        # still a static forecast probe, not an autoregressive model.
        test = temporal_holdout(days, CONFIG["holdout"])
        if CONFIG["pheno_spatial"]:
            # LOOP-spacetime (1) SPATIAL generalization: the mean-DOY graduation head must forecast timing in
            # UNSEEN geography, not just future time. Swap the query set to held-out 0.5deg spatial blocks;
            # neighbours are drawn from TRAIN (seen) blocks. Tests generalization to new places, not memorization.
            test = spatial_holdout(lat, lon, CONFIG["holdout"], seed=a.seed)
        if CONFIG["forecast_spatial"]:
            # Keep only the two valid quadrants.  The old ``test=future&held; train=~test``
            # leaked future-seen-place and past-held-place rows into training.
            train, test, _embargo = strict_spatiotemporal_holdout(
                lat, lon, days, CONFIG["holdout"], seed=a.seed
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
        tnorm, tmin, tspan = normalize_time_from_train(days, ~test, horizon=CONFIG["time_horizon"])
        coords = torch.tensor(np.stack([lat, lon, np.zeros_like(lat), tnorm], 1))  # [N,4]=(lat,lon,elev=0,t=REAL)
    else:
        test = spatial_holdout(lat, lon, CONFIG["holdout"], seed=a.seed)
        coords = torch.tensor(np.stack([lat, lon, np.zeros_like(lat), np.zeros_like(lat)], 1))  # [N,4] t=0
    fam_t = torch.tensor(fam)

    enc = Earth4D(verbose=False, spatial_levels=CONFIG["spatial_levels"], temporal_levels=CONFIG["temporal_levels"],   # S3: exposed capacity
                  spatial_log2_hashmap_size=CONFIG["log2_hashmap"], temporal_log2_hashmap_size=CONFIG["log2_hashmap"], freq_log_scale_init=-2.5,
                  fourier_features=CONFIG["fourier"], fourier_scale=CONFIG["fourier_scale"],
                  time_harmonics=CONFIG["time_harmonics"], time_film=CONFIG["time_film"],
                  spatial_cline=CONFIG["spatial_cline"], cline_scale=CONFIG["cline_scale"],
                  spatial_siren=CONFIG["spatial_siren"], siren_layers=CONFIG["siren_layers"], siren_w0=CONFIG["siren_w0"],
                  causal_lags=CONFIG["causal_lags"], causal_lag_span=CONFIG["causal_lag_span"],
                  seasonal_time=CONFIG["seasonal_time"], drop_spatiotemporal=CONFIG["drop_spatiotemporal"],
                  nystrom=CONFIG["nystrom"], conj=CONFIG["conj"], elm=CONFIG["elm"],
                  elm_scale=CONFIG["elm_scale"], stencil=CONFIG["stencil"],
                  stencil_radius=CONFIG["stencil_radius"],
                  # one year in normalized-time units. tspan is fit on TRAIN rows only (see
                  # normalize_time_from_train), so the seasonal period carries no test information.
                  time_period=(365.25 / tspan if CONFIG["forecast"] and CONFIG["seasonal_time"] else 0.0),
                  coord_shrink=CONFIG["coord_shrink"], spatial_ensemble=CONFIG["spatial_ensemble"],
                  whiten=CONFIG["whiten"], standardize=CONFIG["standardize"],
                  tile=CONFIG["tile"], tile_replace=CONFIG["tile_replace"], tile_time=CONFIG["tile_time"], tile_offsets=CONFIG["tile_offsets"], tile_quantile=CONFIG["tile_quantile"],
                  coordinate_system=("geographic" if CONFIG["geographic"] else "ecef"),
                  ).to(dev)   # RFF + temporal-harmonic + space x time FiLM (arch levers)
    if CONFIG["tile_quantile"]:
        enc.fit_tile_quantiles(coords[torch.tensor(~test)].to(dev))
    if CONFIG["whiten"]:
        enc.fit_whiten(coords[torch.tensor(~test)].to(dev))
    if CONFIG["standardize"]:
        enc.fit_standardize(coords[torch.tensor(~test)].to(dev))
    if CONFIG["extent_fit"] or CONFIG["nystrom"] > 0:
        # Fit on TRAIN rows only. Using every row would leak the evaluation period's extent (and, for the
        # anchors, its actual coordinates) into the feature map.
        _train_coords = coords[torch.tensor(~test)].to(dev)
        if CONFIG["extent_fit"]:
            enc.fit_extent(_train_coords)
        if CONFIG["nystrom"] > 0:
            enc.fit_anchors(_train_coords, seed=a.seed)

    # (The old --env_temporal/--env_perobs/--env_quantiles/--env_extremes/--env_spread guard is gone with
    # those flags: they only ever affected the deleted --env_trait diagnostic and were silently inert on
    # the --env path, which is how family_from_env read data-limited for 53 runs. A flag that cannot act
    # is worse than no flag.)

    if CONFIG["env"] or CONFIG["env_decode"]:
        # science.md rules 1-6, 24 done RIGHT: the positional field should represent the ENVIRONMENT; biology
        # follows. Real env covariates (worldclim+soil+elev) joined by gbifID -> the science-aligned question.
        env = load_env(CONFIG["cache_dir"], gid, channels=CONFIG["env_channels"], fit_mask=~test)  # train-fit transform
        if CONFIG["env_rff"] > 0:
            _rng = np.random.default_rng(a.seed)
            _W = _rng.normal(0, CONFIG["env_rff_scale"], (env.shape[1], CONFIG["env_rff"])).astype(np.float32)
            _b = _rng.uniform(0, 2 * np.pi, CONFIG["env_rff"]).astype(np.float32)
            _p = env @ _W + _b
            env = np.concatenate([env, np.cos(_p).astype(np.float32)], 1)
        if CONFIG["vision"]:
            env = load_vision(CONFIG["cache_dir"], gid, CONFIG["vision_feats"], CONFIG["n_shards"], fit_mask=~test)
        rn = np.stack([lat / 90.0, lon / 180.0], 1).astype(np.float32)
        if CONFIG["forecast"]:
            rn = np.concatenate([rn, tnorm[:, None]], 1)


    if CONFIG["env"]:
        # ---- Move 1: is real ENVIRONMENT >> any coordinate positional encoding at held-out biology? ----
        # Fair controls: Earth4D(coords), RFF(coords), raw(coords) -- the best coordinate-PE. Plus Earth4D+env
        # fused. All share the SAME head (linear or MLP), steps, lr. st_gain reported as env-or-fused MINUS the
        # best coordinate-PE control -> if env >> best coord-PE, the encoder's job is to REPRESENT environment.
        with torch.no_grad():
            e4d = enc(coords.to(dev)).cpu()
        # The Earth4D arm trains `enc` IN PLACE under train_encoder, so a fused arm that reused it would
        # inherit an already-trained table -- twice the encoder budget of the arm it is compared against.
        # The fused arm gets its own freshly-initialized copy.
        import copy as _copy
        enc_fused = _copy.deepcopy(enc)
        env_t = torch.tensor(env)
        raw = torch.tensor(rn)
        rff_rng = np.random.default_rng(0)
        proj = rn @ (rff_rng.normal(0, 8.0, (rn.shape[1], e4d.shape[1] // 2)).astype(np.float32))
        rff = torch.tensor(np.concatenate([np.sin(proj), np.cos(proj)], 1).astype(np.float32))
        fused = torch.cat([e4d, env_t], 1)                       # Earth4D coords ++ real environment
        raw_acc, raw_t5 = evaluate(raw, fam_t, test, n_fam, dev, CONFIG["steps"], CONFIG["lr"], "raw", CONFIG["head_hidden"], a.seed)
        rff_acc, rff_t5 = evaluate(rff, fam_t, test, n_fam, dev, CONFIG['steps'], CONFIG['lr'], "rff", CONFIG['head_hidden'], a.seed)
        e4d_acc, e4d_t5 = (evaluate_trainable(enc, coords, fam_t, test, n_fam, dev, CONFIG["steps"], CONFIG["lr"], "earth4d",
                                              CONFIG['head_hidden'], CONFIG['enc_lr_mult'], CONFIG['enc_warmup'], CONFIG['enc_c2f'], seed=a.seed)
                           if CONFIG["train_encoder"] else
                           evaluate(e4d, fam_t, test, n_fam, dev, CONFIG["steps"], CONFIG["lr"], "earth4d", CONFIG["head_hidden"], a.seed))
        env_acc, env_t5 = evaluate(env_t, fam_t, test, n_fam, dev, CONFIG["steps"], CONFIG["lr"], "env", CONFIG["head_hidden"], a.seed)
        fus_acc, fus_t5 = (evaluate_trainable(enc_fused, coords, fam_t, test, n_fam, dev, CONFIG["steps"], CONFIG["lr"], "fused",
                                              CONFIG['head_hidden'], CONFIG['enc_lr_mult'], CONFIG['enc_warmup'], CONFIG['enc_c2f'],
                                              seed=a.seed, side=env_t)
                           if CONFIG["train_encoder"] else
                           evaluate(fused, fam_t, test, n_fam, dev, CONFIG["steps"], CONFIG["lr"], "fused", CONFIG["head_hidden"], a.seed))
        dt = time.time() - t0
        best_coord = max(raw_acc, rff_acc, e4d_acc)              # best coordinate-only PE
        mode = ("FORECAST(future+newplace)" if CONFIG["forecast_spatial"] else "FORECAST(past->future)") if CONFIG["forecast"] else "spatial-block"
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
            # The fused primary is only frozen-random when train_encoder is off. Declaring it honestly is
            # what stops a trained-encoder number from being compared like-for-like with a frozen record.
            trained_encoder=CONFIG["train_encoder"],
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

    if CONFIG["env_decode"]:
        # ---- Move 2 (rule 24 done right): env-supervised field. Train each encoder to ALSO decode the smooth ----
        # real ENVIRONMENT field (worldclim, standardized) at TRAIN obs as an aux regression target, THEN predict
        # biology from the learned field at the strict held-out set. Fair controls (mlp/rff) get the identical
        # aux. st_gain = env-supervised-Earth4D biology-acc MINUS best generic control -> does a physically-real
        # smooth target make the 4D field finally beat a generic PE (family-supervised field failed -0.10)?
        from deepearth.autoresearch.probes.spacetime.editable_files.lib.env_field import run_env_decode
        with torch.no_grad():
            fdim = enc(coords[:8].to(dev)).shape[1]
        env_tgt = env[:, :19]                                    # worldclim = the smooth, physically-real field
        e4d_acc, e4d_t5, n_te, e4d_er = run_env_decode("earth4d", coords, rn, env_tgt, fam, test, n_fam, dev,
                                                       enc=enc, feat_dim=fdim, steps=CONFIG["steps"], lr=CONFIG["lr"],
                                                       head_hidden=max(CONFIG["head_hidden"], 256), aux_w=CONFIG["env_aux_weight"])
        mlp_acc, mlp_t5, _, mlp_er = run_env_decode("mlp", coords, rn, env_tgt, fam, test, n_fam, dev,
                                                    feat_dim=fdim, steps=CONFIG["steps"], lr=CONFIG["lr"],
                                                    head_hidden=max(CONFIG["head_hidden"], 256), aux_w=CONFIG["env_aux_weight"])
        rff_acc, rff_t5, _, rff_er = run_env_decode("rff", coords, rn, env_tgt, fam, test, n_fam, dev,
                                                    feat_dim=fdim, steps=CONFIG["steps"], lr=CONFIG["lr"],
                                                    head_hidden=max(CONFIG["head_hidden"], 256), aux_w=CONFIG["env_aux_weight"])
        dt = time.time() - t0
        ctrl = max(mlp_acc, rff_acc)
        mode = ("FORECAST(future+newplace)" if CONFIG["forecast_spatial"] else "FORECAST(past->future)") if CONFIG["forecast"] else "spatial-block"
        print(f"  env-recon val R2ish | mlp {mlp_er:.4f} | RFF {rff_er:.4f} | Earth4D {e4d_er:.4f}   (aux env-field fit quality)")
        print(f"  {len(lat)} obs, {CONFIG['steps']}-step env-decode in {dt:.1f}s")
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
            env_aux_weight=CONFIG["env_aux_weight"],
            env_recon_r2={"mlp": mlp_er, "rff": rff_er, "earth4d": e4d_er},
            top5={"mlp": mlp_t5, "rff": rff_t5, "earth4d": e4d_t5},
        )
        return {"st_gain": e4d_acc - mlp_acc, "st_gain_bestctrl": e4d_acc - ctrl, "earth4d_acc": e4d_acc,
                "mlp_acc": mlp_acc, "rff_acc": rff_acc, "earth4d_envR2": e4d_er, "obs": len(lat),
                "seconds": dt, "env_decode": True}

    if CONFIG["field_decode"]:
        # science.md rule 24: TRAIN the encoder end-to-end to decode the dense family field between sparse obs,
        # then forecast the strict held-out (future+new-place) set. Fair controls under the identical decode:
        # a trainable coord-MLP (generic learned PE, matched capacity) and fixed-RFF+trainable-head.
        # st_gain = trained-Earth4D forecast MINUS the best generic learned control -> isolates whether the
        # 4D hash field learns propagatable field structure a plain learned coordinate map lacks.
        from deepearth.autoresearch.probes.spacetime.editable_files.lib.recurrence import run_field_decode
        with torch.no_grad():
            fdim = enc(coords[:8].to(dev)).shape[1]
        rn_fd = np.stack([lat / 90.0, lon / 180.0], 1).astype(np.float32)
        if CONFIG["forecast"]:
            rn_fd = np.concatenate([rn_fd, tnorm[:, None]], 1)
        else:
            rn_fd = np.concatenate([rn_fd, np.zeros((len(lat), 1), np.float32)], 1)
        e4d_acc, e4d_t5, n_te = run_field_decode("earth4d", coords, rn_fd, fam, test, n_fam, dev,
                                                 enc=enc, feat_dim=fdim, steps=CONFIG["steps"], lr=CONFIG["lr"], head_hidden=max(CONFIG["head_hidden"], 256))
        mlp_acc, mlp_t5, _ = run_field_decode("mlp", coords, rn_fd, fam, test, n_fam, dev,
                                              feat_dim=fdim, steps=CONFIG["steps"], lr=CONFIG["lr"], head_hidden=max(CONFIG["head_hidden"], 256))
        rff_acc, rff_t5, _ = run_field_decode("rff", coords, rn_fd, fam, test, n_fam, dev,
                                              feat_dim=fdim, steps=CONFIG["steps"], lr=CONFIG["lr"], head_hidden=max(CONFIG["head_hidden"], 256))
        dt = time.time() - t0
        ctrl = max(mlp_acc, rff_acc)
        mode = "FIELD-DECODE(future+newplace)" if CONFIG["forecast_spatial"] else ("FIELD-DECODE(past->future)" if CONFIG["forecast"] else "FIELD-DECODE(spatial-block)")
        print(f"  {len(lat)} obs, {CONFIG['steps']}-step decode in {dt:.1f}s")
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
    if CONFIG["forecast"]:
        rn = np.concatenate([rn, tnorm[:, None]], 1)             # fair: baselines get the SAME time feature
    raw = torch.tensor(rn)                                        # raw normalized coords (+time) baseline
    # Random Fourier Features of (lat,lon[,t]): the fair nonlinear positional-encoding control.
    # Train-extent normalized and bandwidth-selected — see fair_rff(). The previous fixed sigma=8 on
    # globe-normalized coords was degenerate on a regional corpus and scored BELOW raw coordinates.
    _rff_scaled, _rff_base, _rff_sigmas = fair_rff(rn, e4d.shape[1], train_mask=~test, seed=a.seed)
    _best = (-1.0, _rff_sigmas[0], None)
    for _sig in _rff_sigmas:
        _cand = _rff_features(_rff_scaled, _rff_base, _sig)
        _acc, _ = evaluate(_cand, fam_t, test, n_fam, dev, CONFIG['steps'], CONFIG['lr'], f"rff_s{_sig:g}",
                           CONFIG["head_hidden"], a.seed)
        if _acc > _best[0]:
            _best = (_acc, _sig, _cand)
    rff = _best[2]
    print(f"  [fair-baseline] RFF bandwidth selected: sigma={_best[1]:g} "
          f"(acc {_best[0]:.4f} over {[f'{x:g}' for x in _rff_sigmas]}) — the control gets its best shot")

    if CONFIG["phenology"]:
        # DECISIVE non-stationary control (science.md rule 1+2b). Prior family-presence forecasting showed
        # propagator_gain ~0 because a STATIONARY spatial climatology fit the target. Here the target is the
        # DAY-OF-YEAR (phenology / seasonal timing) -- non-stationary: a static (lat,lon) map explains ~3% of
        # it, so a real propagator that carries WHEN nearby species were recently seen should finally win.
        # static no-propagation floor vs GNN vs LSTM, each over Earth4D / RFF / raw, on the declared split.
        # propagator_gain = propagator MAE improvement over the static floor.
        assert CONFIG["forecast"], "--phenology requires --forecast (needs live event-time + past->future split)"
        from deepearth.autoresearch.probes.spacetime.editable_files.lib.phenology import run_phenology_all
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
        if CONFIG["pheno_channel"]:
            _ph = np.load(Path(CONFIG["cache_dir"]) / "gbif_phenology_tokens.npz")
            _pm = {int(g): i for i, g in enumerate(_ph["gbifID"])}; _PH = _ph["phenology"]
            import glob as _g2
            _gg=[np.load(_f)["gbifID"] for _f in sorted(_g2.glob(str(Path(CONFIG["cache_dir"])/"gbif_tokens/*.npz")))[:CONFIG["n_shards"]]]
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
        if CONFIG["pheno_species"]:
            import glob as _glob
            from pathlib import Path as _Path
            _sp = []
            for _f in sorted(_glob.glob(str(_Path(CONFIG["cache_dir"]) / "gbif_tokens/*.npz")))[:CONFIG["n_shards"]]:
                _sp.append(np.load(_f)["species_local"])
            sp_all = np.concatenate(_sp).astype(np.int64)[obs_index]
        _feats = phenology_feature_set(CONFIG["pheno_feats"], CONFIG["pheno_nofair"])
        # FAIR-BASELINE GUARD: a single-feature run (e.g. --pheno_feats e4d) left the RFF control untrained, so the
        # trace could report NO fair gain at all and still set a record -- this capability's records were being
        # gated on nothing. Whenever Earth4D is trained, train raw and generic-PE controls too
        # (opt out: --pheno_nofair).
        r = run_phenology_all(e4d_sp, rff_sp, raw_sp, fd, days, coords_ll, test, dev,
                              K=CONFIG["rec_k"], steps=CONFIG["steps"], lr=CONFIG["lr"], hidden=CONFIG["rec_hidden"], hops=CONFIG["gnn_hops"], tol_days=CONFIG["pheno_tol"],
                              attn=CONFIG["pheno_attn"], attn_heads=CONFIG["attn_heads"], attn_layers=CONFIG["attn_layers"], sp=sp_all,
                              block_deg=CONFIG["rec_block_deg"], fast=CONFIG["rec_fast"],
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
        if CONFIG["pheno_attn"]:
            pg_raw_attn_mae, pg_raw_attn_acc = pg("raw", "attn")
            best_prop_raw_mae = max(best_prop_raw_mae, pg_raw_attn_mae)
        pg_raw_sp_mae = pg_raw_sp_acc = float("nan")
        if CONFIG["pheno_species"]:
            pg_raw_sp_mae, pg_raw_sp_acc = pg("raw", "sp")
            best_prop_raw_mae = max(best_prop_raw_mae, pg_raw_sp_mae)
        pheno_mode = phenology_mode(CONFIG["forecast_spatial"], CONFIG["pheno_spatial"])
        for ft in ("raw", "rff", "e4d"):
            d = r[ft]
            attn_s = f" | ATTN MAE {d.get('attn_mae', float('nan')):6.2f}d acc {d.get('attn_acc', float('nan')):.4f} (prop {d['static_mae']-d.get('attn_mae', float('nan')):+.2f}d)" if CONFIG['pheno_attn'] else ""
            sp_s = f" | SP MAE {d.get('sp_mae', float('nan')):6.2f}d acc {d.get('sp_acc', float('nan')):.4f} (prop {d['static_mae']-d.get('sp_mae', float('nan')):+.2f}d)" if CONFIG['pheno_species'] else ""
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
        print(f"  {len(lat)} obs, {CONFIG['steps']}-step phenology in {dt:.1f}s")
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
            forecast_queries=n_te, tol_days=CONFIG["pheno_tol"], K=CONFIG["rec_k"], hops=CONFIG["gnn_hops"],
            attn=CONFIG["pheno_attn"], obs=len(lat), seconds=dt,
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









    if CONFIG["gnn"]:
        # science.md rule 1+2b: GraphCast/GenCast-style message-passing propagator. Per held-out (future+new-
        # place) query, build a graph of its K strictly-earlier past observations with LEARNED space-time edges
        # (dlat,dlon,dt), run multi-hop message passing, decode the query -> family forecast. Report ABSOLUTE
        # forecast skill for: GNN propagator vs the no-propagation static head (same query set) vs -- for the
        # mechanism control -- the SAME GNN over Earth4D / RFF / raw node features. propagator-vs-none isolates
        # whether causal propagation forecasts forward at all; Earth4D-vs-RFF isolates mechanism vs encoder.
        assert CONFIG["forecast"], "--gnn requires --forecast (needs live event-time + past->future split)"
        from deepearth.autoresearch.probes.spacetime.editable_files.lib.gnn import run_gnn
        coords_ll = torch.tensor(np.stack([lat, lon], 1).astype(np.float32))
        e4d_r = run_gnn(e4d, e4d.shape[1], fam, days, coords_ll, test, n_fam, dev,
                        K=CONFIG["rec_k"], steps=CONFIG["steps"], lr=CONFIG["lr"], hidden=CONFIG["rec_hidden"], hops=CONFIG["gnn_hops"])
        rff_r = run_gnn(rff, rff.shape[1], fam, days, coords_ll, test, n_fam, dev,
                        K=CONFIG["rec_k"], steps=CONFIG["steps"], lr=CONFIG["lr"], hidden=CONFIG["rec_hidden"], hops=CONFIG["gnn_hops"])
        raw_r = run_gnn(raw, raw.shape[1], fam, days, coords_ll, test, n_fam, dev,
                        K=CONFIG["rec_k"], steps=CONFIG["steps"], lr=CONFIG["lr"], hidden=CONFIG["rec_hidden"], hops=CONFIG["gnn_hops"])
        dt = time.time() - t0
        n_te = e4d_r["n_te"]
        # propagator-vs-none PER feature type (apples-to-apples: each GNN vs its OWN static floor).
        pg_raw = raw_r["gnn_acc"] - raw_r["static_acc"]; pg_raw5 = raw_r["gnn_top5"] - raw_r["static_top5"]
        pg_e4d = e4d_r["gnn_acc"] - e4d_r["static_acc"]; pg_rff = rff_r["gnn_acc"] - rff_r["static_acc"]
        print(f"  ABSOLUTE top1 | raw: static {raw_r['static_acc']:.4f} -> GNN {raw_r['gnn_acc']:.4f} (prop {pg_raw:+.4f}) | RFF: static {rff_r['static_acc']:.4f} -> GNN {rff_r['gnn_acc']:.4f} (prop {pg_rff:+.4f}) | E4D: static {e4d_r['static_acc']:.4f} -> GNN {e4d_r['gnn_acc']:.4f} (prop {pg_e4d:+.4f})")
        print(f"  ABSOLUTE top5 | raw: static {raw_r['static_top5']:.4f} -> GNN {raw_r['gnn_top5']:.4f} (prop {pg_raw5:+.4f}) | RFF: static {rff_r['static_top5']:.4f} -> GNN {rff_r['gnn_top5']:.4f} | E4D: static {e4d_r['static_top5']:.4f} -> GNN {e4d_r['gnn_top5']:.4f}")
        print(f"  BEST propagator_gain (raw features: GNN vs its no-prop floor) top1 {pg_raw:+.4f}  top5 {pg_raw5:+.4f}   (does causal propagation forecast forward?)")
        print(f"  {len(lat)} obs, {CONFIG['steps']}-step GNN in {dt:.1f}s")
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
            obs=len(lat), forecast_queries=n_te, K=CONFIG["rec_k"], hops=CONFIG["gnn_hops"],
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

    if CONFIG["recurrence"]:
        # science.md rule 2b: physics-inspired 4D recurrence. Instead of a static per-point lookup head,
        # a causal LSTM rollout PROPAGATES local past state forward to each held-out (future+new-place) query.
        # Same rollout is run on Earth4D / raw / RFF features -> st_gain isolates whether Earth4D's 4D field
        # carries structure that PROPAGATES past->future, not just structure that indexes a cell.
        assert CONFIG["forecast"], "--recurrence requires --forecast (needs live event-time + past->future split)"
        coords_ll = torch.tensor(np.stack([lat, lon], 1).astype(np.float32))
        if CONFIG["rec_time_cond"]:
            # rule24+2b: instead of feeding each neighbour its OWN static code, re-encode the QUERY cell
            # (lat_q,lon_q) FORWARD to each step's event day so the encoder's time axis carries state the LSTM
            # propagates. featurize(lat,lon,day) reproduces the exact Earth4D / raw / RFF normalizations.
            from deepearth.autoresearch.probes.spacetime.editable_files.lib.recurrence import run_recurrence_timecond
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
                                                            K=CONFIG["rec_k"], steps=CONFIG["steps"], lr=CONFIG["lr"], hidden=CONFIG["rec_hidden"], tag="raw")
            rff_acc, rff_t5, _ = run_recurrence_timecond(feat_rff, e4d.shape[1], fam, days, coords_ll, test, n_fam, dev,
                                                         K=CONFIG['rec_k'], steps=CONFIG['steps'], lr=CONFIG['lr'], hidden=CONFIG['rec_hidden'], tag="rff")
            e4d_acc, e4d_t5, _ = run_recurrence_timecond(feat_e4d, e4d.shape[1], fam, days, coords_ll, test, n_fam, dev,
                                                         K=CONFIG["rec_k"], steps=CONFIG["steps"], lr=CONFIG["lr"], hidden=CONFIG["rec_hidden"], tag="earth4d")
            dt = time.time() - t0
            print(f"  {len(lat)} obs, {CONFIG['steps']}-step rollout in {dt:.1f}s")
            return {"st_gain": e4d_acc - raw_acc, "st_gain_rff": e4d_acc - rff_acc, "earth4d_acc": e4d_acc,
                    "raw_acc": raw_acc, "rff_acc": rff_acc, "obs": len(lat), "seconds": dt, "recurrence": True, "time_cond": True}
        from deepearth.autoresearch.probes.spacetime.editable_files.lib.recurrence import run_recurrence
        raw_acc, raw_t5, n_te = run_recurrence(raw, fam, days, coords_ll, test, n_fam, dev,
                                               K=CONFIG["rec_k"], steps=CONFIG["steps"], lr=CONFIG["lr"], hidden=CONFIG["rec_hidden"], tag="raw")
        rff_acc, rff_t5, _ = run_recurrence(rff, fam, days, coords_ll, test, n_fam, dev,
                                            K=CONFIG['rec_k'], steps=CONFIG['steps'], lr=CONFIG['lr'], hidden=CONFIG['rec_hidden'], tag="rff")
        e4d_acc, e4d_t5, _ = run_recurrence(e4d, fam, days, coords_ll, test, n_fam, dev,
                                            K=CONFIG["rec_k"], steps=CONFIG["steps"], lr=CONFIG["lr"], hidden=CONFIG["rec_hidden"], tag="earth4d")
        dt = time.time() - t0
        print(f"  {len(lat)} obs, {CONFIG['steps']}-step rollout in {dt:.1f}s")
        declare(
            capability="family_from_spacetime",
            mode="RECURRENCE(4D-LSTM rollout past->future)",
            metric="family_top1_accuracy",
            value=e4d_acc,
            split="FORECAST(past->future)",
            gains={"vs raw": e4d_acc - raw_acc, "vs RFF": e4d_acc - rff_acc},
            baselines={"raw": raw_acc, "RFF": rff_acc},
            obs=len(lat), rollout_queries=n_te, families=n_fam, K=CONFIG["rec_k"], hidden=CONFIG["rec_hidden"],
            earth4d_dim=int(e4d.shape[1]), seconds=dt,
            top5={"raw": raw_t5, "rff": rff_t5, "earth4d": e4d_t5},
        )
        return {"st_gain": e4d_acc - raw_acc, "st_gain_rff": e4d_acc - rff_acc, "earth4d_acc": e4d_acc,
                "raw_acc": raw_acc, "rff_acc": rff_acc, "obs": len(lat), "seconds": dt, "recurrence": True}

    raw_acc, raw_t5 = evaluate(raw, fam_t, test, n_fam, dev, CONFIG["steps"], CONFIG["lr"], "raw", CONFIG["head_hidden"], a.seed)
    rff_acc, rff_t5 = evaluate(rff, fam_t, test, n_fam, dev, CONFIG['steps'], CONFIG['lr'], "rff", CONFIG['head_hidden'], a.seed)
    e4d_acc, e4d_t5 = (evaluate_trainable(enc, coords, fam_t, test, n_fam, dev, CONFIG["steps"], CONFIG["lr"], "earth4d",
                                          CONFIG['head_hidden'], CONFIG['enc_lr_mult'], CONFIG['enc_warmup'], CONFIG['enc_c2f'], seed=a.seed)
                       if CONFIG["train_encoder"] else
                       evaluate(e4d, fam_t, test, n_fam, dev, CONFIG["steps"], CONFIG["lr"], "earth4d", CONFIG["head_hidden"], a.seed))
    dt = time.time() - t0
    mode = ("FORECAST(future+newplace)" if CONFIG["forecast_spatial"] else "FORECAST(past->future)") if CONFIG["forecast"] else "spatial-block"
    print(f"  {len(lat)} obs, {CONFIG['steps']}-step probe in {dt:.1f}s")
    # The shared coordinate/forecast tail. --target selects WHICH capability this is; the old harness
    # could not tell family_from_spacetime from species_from_spacetime here because both print the same
    # header and both were matched by the same r"\bEarth4D\s+([\d.]+)" pattern.
    _target_capability = ("species_from_spacetime" if CONFIG["target"] == "species" else "family_from_spacetime")
    declare(
        capability=_target_capability,
        mode=mode,
        metric=f"{CONFIG['target']}_top1_accuracy",
        value=e4d_acc,
        split=mode,
        gains={"vs raw": e4d_acc - raw_acc, "vs RFF": e4d_acc - rff_acc},
        baselines={"raw": raw_acc, "RFF": rff_acc},
        obs=len(lat), held_out=int(test.sum()), n_classes=n_fam, earth4d_dim=int(e4d.shape[1]),
        seconds=dt, target=CONFIG["target"],
        top5={"raw": raw_t5, "rff": rff_t5, "earth4d": e4d_t5},
    )
    return {"st_gain": e4d_acc - raw_acc, "st_gain_rff": e4d_acc - rff_acc, "earth4d_acc": e4d_acc, "raw_acc": raw_acc,
            "rff_acc": rff_acc, "obs": len(lat), "seconds": dt, "forecast": CONFIG["forecast"]}



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




if __name__ == "__main__":
    main()
