"""Editable Earth4D candidate profiles and scientific channel declarations.

The fixed probe owns measurement identity and scoring. Candidate architecture,
readout, channel, and optimizer choices live here so autoresearch never edits
its judge.
"""

CHANNELS = {
    # name           files (relative to CONFIG["cache_dir"])            dims  switch          what it is
    "worldclim":    (("gbif_worldclim_tokens.npz",),                     19, "env_channels", "19 bioclim variables: temperature and precipitation normals"),
    "soil":         (("gbif_soil_tokens.npz",),                           9, "env_channels", "SSURGO soil properties"),
    "elev":         (("gbif_elev.npz",),                                  1, "env_channels", "elevation; always joined to worldclim/soil"),
    "alphaearth":   (("gbif_alphaearth_tokens.npz",),                    64, "env_channels", "AlphaEarth learned geo embedding -- a foundation-model prior, not a measurement"),
    "ae_wb":        (("gbif_ae_wb.npz",),                                64, "env_channels", "AlphaEarth + water balance"),
    "ae_wb_ph":     (("gbif_ae_wb_ph.npz",),                             64, "env_channels", "AlphaEarth + water balance + soil pH"),
    "modis":        (("gbif_phenology_tokens.npz",),                     16, "env_channels", "MODIS phenology: greenup / senescence timing"),
    "terrain":      (("gbif_topo_tokens.npz", "gbif_hydro_tokens.npz"),  32, "env_channels", "3DEP microtopography + HydroSHEDS drainage; '+terrain' suffix"),
    "vision_dino":  (("gbif_tokens/",),                                 768, "vision",       "DINOv3 embedding of the iNaturalist photo. BORROWED: a win here is the vision model's, not the encoder's"),
    "vision_bio":   (("gbif_tokens/",),                                 768, "vision_feats", "BioCLIP-2 embedding of the same photo; same attribution warning"),
    "pheno":        (("gbif_flower.npz",),                                1, "pheno_channel","observed flowering state"),
    # NOTE: gbif_species_dist.npz is NOT read by this probe. The SDM modes build their target from
    # cooccur_count_005 + the env channels; gbif_species_dist belongs to fusion (B29/B39/B40). It was
    # listed here for one commit and that was wrong -- a table that claims a channel the probe cannot
    # use is worse than no table.
    "cooccur":      (("derived/cooccur_count_005.npy",),                  0, "cooccur_file", "species co-occurrence counts at 0.5 deg -- the community target"),
}

# The corpus carries repaired versions under derived/*_rebuilt.*. A data-integrity audit
# (main/program/GRADUATION_BLUEPRINT.md) found 2 mislabeled arrays and 5 missing files, rebuilt them,
# and made "activate the 6 rebuilt data files" graduation step 1. Activating one means copying it onto
# its live name -- a champion-pipeline change, an operator decision. `harness.py --channels` reports
# which are still un-activated, because a run against the un-repaired file measures the wrong corpus.
REPAIRED = {
    "gbif_species_dist.npz":      "derived/gbif_species_dist_rebuilt.npz",
    "gbif_plant_dist.npz":        "derived/gbif_plant_dist_rebuilt.npz",
    "gbif_mycorrhiza.npz":        "derived/gbif_mycorrhiza_rebuilt.npz",
    "gbif_lfmc.npz":              "derived/gbif_lfmc_rebuilt.npz",
    "bioclip_taxon_text_emb.npy": "derived/bioclip_taxon_text_emb_rebuilt.npy",
}


CONFIG = {
    # THE ENCODER, AS fusion.py INSTANTIATES IT. Nothing bolted on.
    #
    # These defaults used to be the tile-coding champion, and that made the probe measure the wrong
    # object. Counted at tag sp2_cmac16_dropst_tau01: the head received 20,663 features, of which
    # Earth4D's hash grid was 36 -- 0.17%. CMAC tile coding was 18,432 (89.2%), the RFF another 2,048
    # (9.9%), and drop_spatiotemporal deleted the tri-planes outright. So `fair_gain vs RFF` was scoring
    # a tile coder; "dropping the inert tri-planes" was free because they were 108 dims out of 20,663;
    # and the question "does the hash grid earn anything" was unanswerable by construction.
    #
    # fusion.py:302 builds Earth4D(spatial_levels=18, temporal_levels=18, log2_hashmap=20,
    # freq_log_scale_init=-2.5) and NOTHING ELSE. The probe now builds exactly that -- 36 spatial + 108
    # tri-plane = 144 dims -- trained end to end, so a probe number is about the object fusion runs.
    #
    # train_encoder is ON. It was off because the trained path was nondeterministic at fixed seed (five
    # seed-0 runs: 0.1873/0.1925/0.1867/0.1872/0.1952) and an irreproducible number cannot set a record.
    # EARTH4D_DETERMINISTIC=1 fixes that in the kernel -- verified bit-identical on all four encoders,
    # and 4.5% FASTER -- so the frozen-random workaround is retired. Set that env var for every run.
    #
    # The bolt-on bases (fourier, time_harmonics, spatial_cline, nystrom, tile*) all still exist and all
    # default OFF. They are legitimate experiments; they are not the encoder. A run that turns one on is
    # measuring the encoder PLUS that basis, and its record must say so.
    "lr": 3e-3,
    "spatial_levels": 18,
    "temporal_levels": 18,
    "log2_hashmap": 20,
    "head_hidden": 512,
    "fourier": 0,
    "fourier_scale": 6400.0,
    "time_harmonics": 0,
    "train_encoder": True,
    # Encoder-training hyperparameters, restored: the arm-deletion purge removed them along with the
    # dead bolt-on keys, and probe.py:1533 reads all three. Values are evaluate_trainable's own
    # defaults, so behaviour is unchanged from before the purge.
    "enc_lr_mult": 0.05,       # the encoder gets its own param group at lr * this, no weight decay
    "enc_warmup": 0.15,        # fraction of steps before the encoder starts moving
    "enc_c2f": 0.5,            # coarse-to-fine: fraction of steps over which fine levels come online
    # literal, not the imported constant: CONFIG is read at module load, before that import.
    # Must stay equal to lib/recurrence.py DEFAULT_TIME_HORIZON.
    "time_horizon": 2.0,
    # ---- WHAT IS MEASURED (these were mode-selector and data flags on the CLI) ----
    # Changing any of these changes the measurement, which is exactly why they belong in the
    # experiment rather than on a command line. config_digest makes the gate see the difference.
    "cache_dir": "autoresearch/data/deepcal",
    "n_shards": 12,
    # R21: equal WALL-CLOCK per arm is what makes a speedup convert to score -- but flipping the budget
    # to time in the same protocol change that reshapes WHAT is measured would confound the v5
    # re-baseline with a compute-budget change. So v5 keeps the historical 800-step budget exactly, and
    # `time_budget_s` stays OFF until one run measures what 800 steps actually costs on the box. Set it
    # to that measured number, in its own commit, and speed starts converting to score.
    "steps": 800,
    "time_budget_s": 0.0,      # 0 = disabled; budget is `steps`. See budgeted().
    "holdout": 0.2,
    "target": "species",
    "forecast": True,
    "forecast_spatial": False,
    "recurrence": False,
    "phenology": True,
    "pheno_nofair": False,
    "pheno_feats": "e4d,rff,raw",
    "pheno_spatial": False,
    "env": False,
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
    "spatial_cline": 0,
    "cline_scale": 1.0,
    "rec_k": 16,
    "rec_hidden": 256,
    "gnn_hops": 2,          # hops for the phenology propagator (the only remaining consumer)
    "rec_time_cond": False,
    "pheno_tol": 15.0,
    "pheno_attn": False,
    "attn_heads": 4,
    "attn_layers": 2,
    "pheno_species": False,
    "rec_block_deg": 2.0,
    "rec_fast": False,
    # ENV-CHANNEL REPRESENTATION (data lever, not head capacity): 0 = the raw standardized channel.
    # >0 appends that many frozen random Fourier features OF THE ENV VECTOR, so the LINEAR head can carve
    # a niche (a region of env space) instead of a halfspace. A trained MLP head over the same channel
    # (head_hidden=256) overfits at 800 steps and LOST (0.1340 vs 0.1423); a frozen kernel expansion with
    # a linear readout is the regularized version of the same hypothesis. Every arm that consumes env
    # (env, fused) sees the identical expansion; the coordinate baselines are untouched, so the ENV vs
    # best-coord-PE comparison stays fair.
    "env_rff": 0,
    "env_rff_scale": 1.0,
    "knn_readout": 0,          # k for the non-parametric local-frequency readout (0 = trained linear head)
    # Soft k-NN class log-vote ADDED to the head logits in evaluate() (cosine, K, tau). 1 = on, the
    # species_from_spacetime champion (0.0787 -> 0.0860). It was introduced UNGATED, which silently
    # changed the estimator for every other capability -- including capabilities whose standing record
    # was set before it existed, whose control can then never reproduce. Gated here, default = champion.
    "knn_vote": 1,
    "knn_vote_k": 256,
    "knn_vote_tau": 0.02,
    # ---- ARCHITECTURE ARMS (earth4d.py). Default-off: the champion path is byte-identical. ----
    "drop_spatiotemporal": False,
    "nystrom": 0,              # RBF features against N train-drawn space-time anchors
    "tile": 0,                 # sparse tile coding: per-level one-hot cell code of this width
    "tile_offsets": 1,         # CMAC-style overlapping tilings per level
    "geographic": False,       # hash (lat, lon, elev) directly instead of ECEF
    "pheno_ntime": False,      # ARM flw_ntime: encode NEIGHBOURS at their observed event time
    # Bandwidth for the phenology RFF control. PINNED, not selected: the classification path sweeps
    # `bandwidths` and lets the control keep its best score, but the phenology head costs ~31s per fit,
    # so sweeping it every run is not affordable. 4.0 won a manual sweep over {1,4,16,64} on THIS corpus
    # at THIS output dim (144): RFF 0.1906 at sigma=4 vs 0.1854 at sigma=8. Change the corpus, the
    # region, or e4d's output width and this number is stale -- and a stale bandwidth silently turns the
    # control back into the handicap the fix above removed. Re-sweep before trusting a gain after any
    # such change.
    "pheno_rff_sigma": 4.0,
}


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
        "head_hidden": 512, "fourier": 0, "time_harmonics": 0,
        "spatial_cline": 0, "n_shards": 12, "tile": 0,
        # CHAMPION 0.095463 (was 0.085953, barrier 0.0020), tag sp2_cmac16_dropst_tau01.
        # tile_offsets 4 -> 16 is the whole effect and it is NOT an output-width artifact: at an
        # identical dim of 20771, offsets=16 scores 0.0925 against tile=128/offsets=8 at 0.0884 and
        # tile=256/offsets=4 at 0.0881. What buys the gain is the NUMBER of overlapping CMAC tilings,
        # not the column count.
        "tile_offsets": 1,
        "knn_vote_tau": 0.01,
        # The space-time tri-planes measure inert on this row as they do on family_from_spacetime:
        # deleting all 108 of those dims is free (+0.0003) and positive in combination. This FORECAST
        # row is effectively a spatial model.
        "drop_spatiotemporal": False,
    },
    # The standing record (0.1769) was set by the PRE-REFACTOR CLI as
    #     --forecast --head_hidden 256 --fourier 1024 --time_harmonics 8 --n_shards 12
    # and NOTHING else, so every unstated lever took that CLI's argparse DEFAULT. The reconstructed
    # block below carried only the five STATED levers, so it silently inherited CONFIG's defaults --
    # which are the species_from_spacetime champion: fourier_scale 6400 (old default 10), spatial_cline
    # 64 (old default 0), tile 64 / tile_offsets 4 (did not exist pre-refactor), and, after the k-NN
    # class-vote landed ungated, a DIFFERENT ESTIMATOR than the one that set the record. Five unstated
    # changes stacked on a control is not a control. Defaults verified verbatim against probe.py at
    # e2b062c^ (the commit that turned the 33 flags into CONFIG): --fourier_scale 10.0, --spatial_cline
    "family_from_spacetime": {
        "forecast": True, "target": "family", "phenology": False, "head_hidden": 256, "fourier": 0,
        "time_harmonics": 0, "n_shards": 12, "fourier_scale": 10.0, "spatial_cline": 0,
        "cline_scale": 1.0, "train_encoder": True, "tile": 0, "tile_offsets": 1, },
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
        "spatial_cline": 0, "cline_scale": 1.0, "tile": 0, "tile_offsets": 1, "rec_k": 16, "rec_hidden": 256, "gnn_hops": 2, "pheno_tol": 15.0,
        "pheno_attn": False, "pheno_species": False, "pheno_spatial": False,
        "forecast_spatial": False, "recurrence": False, "train_encoder": True,
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
        "env": True, "forecast": False, "phenology": False,
        "n_shards": 12, "tile": 0, "spatial_cline": 0, "fourier_scale": 10.0,
        "target": "family",          # old CLI default; CONFIG's "species" is a DIFFERENT capability
        "head_hidden": 0,            # old default: LINEAR head (CONFIG's 512 is the species champion)
        "fourier": 0, "time_harmonics": 0,
        # ARM knn200_terrain. The linear-head champion was env_channels="alphaearth" + knn_readout=0
        # (0.142318). Under the LINEAR head the terrain stack was a dead-end (0.1410); under the
        # non-parametric readout it is the win (0.1489) -- the channel's value was masked by the readout.
        "env_channels": "alphaearth+terrain", "knn_readout": 200,
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
    """Point CONFIG at the declared capability's champion before the run starts.

    THE PRESET WINS OVER CONFIG, AND THAT SILENTLY VOIDS EXPERIMENTS. Since the flags were removed, an
    experiment IS a diff of the CONFIG block -- but this runs after CONFIG is defined, so editing CONFIG
    for any key the capability's preset also pins does nothing at all. An agent sweeping
    time_harmonics on species_from_spacetime got the control's exact score AND the control's exact
    identity_digest, because the preset overwrote the edit before the encoder was built.

    The precedence is not changed here: flipping it would silently alter every capability's champion
    mid-campaign. Instead the override is made LOUD, so a voided edit is impossible to mistake for a
    null result. To experiment on a key the preset pins, edit the PRESET for that capability -- that
    block IS the champion definition, and it registers in config_digest.
    """
    preset = CAPABILITY_CONFIG.get(capability, {})
    overridden = [(k, CONFIG[k], v) for k, v in preset.items()
                  if k in CONFIG and CONFIG[k] != v]
    for k, v in preset.items():
        CONFIG[k] = v
    if overridden:
        print(f"  [capability-preset] {capability}: the preset OVERRODE {len(overridden)} CONFIG "
              f"value(s). If you edited any of these in CONFIG, YOUR EDIT DID NOT TAKE EFFECT -- "
              f"edit the CAPABILITY_CONFIG preset instead:")
        for k, was, now in overridden:
            print(f"      {k}: CONFIG={was!r} -> preset={now!r}")



# Candidate-owned observation and channel transforms.
import csv
import glob
from pathlib import Path

import numpy as np


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


def load_historical_gbif_support(cache: str):
    """Prepared, ID-disjoint 2019--2024 GBIF range support."""
    z = np.load(Path(cache) / "gbif_densify_bulk.npz")
    return (z["lat"].astype(np.float32), z["lon"].astype(np.float32),
            z["species_local"].astype(np.int64), z["gbifID"].astype(np.int64))


def load_dated_gbif_support(cache: str, shard_start: int = 12, shard_stop: int = 26):
    """Load the disjoint native shards after the fixed 12-shard corpus, with exact event days."""
    cachep = Path(cache)
    shards = sorted(
        p for p in (cachep / "gbif_tokens").glob("chunk*.npz")
        if p.stem.removeprefix("chunk").isdigit()
    )[shard_start:shard_stop]
    if len(shards) != shard_stop - shard_start:
        raise RuntimeError(f"expected {shard_stop - shard_start} dated support shards, found {len(shards)}")
    event_time = np.load(cachep / "gbif_eventtime.npz")
    id2day = dict(zip(event_time["gbifID"].tolist(), event_time["days"].tolist()))
    lat, lon, species, gid, days = [], [], [], [], []
    for path in shards:
        z = np.load(path)
        ids = z["gbifID"].astype(np.int64)
        lat.append(z["lat"]); lon.append(z["lon"]); species.append(z["species_local"])
        gid.append(ids)
        days.append(np.array([id2day.get(int(i), np.nan) for i in ids], dtype=np.float32))
    return (np.concatenate(lat).astype(np.float32), np.concatenate(lon).astype(np.float32),
            np.concatenate(species).astype(np.int64), np.concatenate(gid).astype(np.int64),
            np.concatenate(days).astype(np.float32))



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
