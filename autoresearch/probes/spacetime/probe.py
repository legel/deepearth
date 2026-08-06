"""Fixed Earth4D validation probe.

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
    encoder itself autoresearch/probes/spacetime/editable_files/earth4d.py  ← the real architecture lever
    end-to-end     --train_encoder [--enc_lr_mult --enc_warmup --enc_c2f]

  CAPACITY (tunes a winner; not a move) ────────────────────────────────────────────────
    --spatial_levels --temporal_levels --log2_hashmap --head_hidden --steps --lr

  WHAT IS MEASURED (identity — changing these makes a DIFFERENT measurement, not a better score)
    --forecast [--forecast_spatial] · --target {family,species} · --phenology · --sdm_presence
    --sdm_hard · --cooccur · --n_shards · --holdout · --seed

Scientific experiments edit the ``editable_files`` package or its local ``lib/``. This module owns
only protocol identity, canonical splits, leak guards, fair controls, budget enforcement, scoring,
and result recording.
"""


PROBE_MODULE = "deepearth.autoresearch.probes.spacetime.probe"
# Must match autoresearch/probes/spacetime/harness.py PROTOCOL. Bump both when a change alters what a run MEASURES.
PROTOCOL_VERSION = "v5-encoder-only"

import os
import argparse
import csv
from functools import partial
import sys
import time
from pathlib import Path

import numpy as np
import torch

from deepearth.autoresearch.probes.spacetime.editable_files import (
    CONFIG,
    DEFAULT_TIME_HORIZON,
    GROUP_DRO_TEMPERATURE,
    apply_capability_config,
    build_candidate_encoder,
    cooccur_routing,
    evaluate_candidate,
    load_dated_gbif_support,
    load_env,
    load_historical_gbif_support,
    load_obs,
    load_vision,
    nearest_dated_conspecific,
    normalize_forecast_time,
    normalize_time_from_train,
    phenology_feature_set,
    phenology_mode,
    run_phenology_all,
    run_recurrence,
    run_recurrence_timecond,
    sdm_presence,
    sdm_presence_hard,
    strict_spatiotemporal_masks,
    train_candidate,
)
from deepearth.autoresearch.probes.spacetime.harness import (
    FAIR_CONTROL_DIM,
    _rff_features,
    fair_rff,
)
from deepearth.autoresearch.probes.spacetime.harness import (
    _set_result_sink,
    declare as _declare_raw,
)
# Measurement definitions live in the non-editable scoring module, not here.
from deepearth.autoresearch.probes.spacetime.harness.diagnostics import (
    autoregressive_rollout, enforce_determinism, field_interpolation, relative_transfer, science_axes,
    signal_capture)

# Canonical split construction stays fixed below. Candidate data transforms,
# models, objectives, and training behavior enter through the package API above.


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


# ==================================================================================================
# SCIENCE AXES — one number per science.md rule, emitted by EVERY run
# ==================================================================================================
#
# The probe used to report ONE number (accuracy vs a fair control) as a proxy for everything, so an
# edit to earth4d.py produced a single scalar and you could not tell WHICH property you had changed.
# That is how drop_spatiotemporal and CMAC tile coding scored as wins while moving away from rule 24.
#
# Each rule gets its own cheap measurement, and every run emits all of them, so one edit -> one command
# -> a VECTOR: "+0.03 on the field axis, -0.01 on autoregression, 2.1x on throughput". That is the
# granular feedback loop. Axes with no instrument yet are absent rather than faked -- see
# scoring.definitions --coverage for which those are.

# ------------------------------------------------------------------------------------------------
# Every declare() carries the science axes. ONE place, not seven.
#
# These were first wired into the forecast declare only, so the capacity and throughput axes appeared
# on two rows of the board and were silently absent from the other five. An axis reported on one
# capability and missing on five is not a measurement, it is an anecdote -- the same failure mode that
# once let mode=None make four unrelated runs mutually "comparable".
#
# _SCIENCE_AXES is filled once, right after the encoder is built, and merged into every result. A mode
# that runs before the encoder exists carries an empty dict rather than a wrong number.
_SCIENCE_AXES: dict = {}


def declare(**kw):
    """harness.declare, with this run's science axes attached to every result."""
    return _declare_raw(**{**kw, **_SCIENCE_AXES})


# R21 — THE BUDGET IS TIME, NOT STEPS.
#
# science.md rule 21: "wall-clock throughput converts directly into training steps and therefore into
# net_score: any acceleration of the algorithm that does not change its per-step mathematics MUST score
# at least as high, and under the budget, strictly higher."
#
# Under a STEP budget it cannot. A 2x faster kernel runs the same 800 steps and produces an identical
# number, so the one loop that owns the CUDA kernel was structurally unable to score a kernel speedup --
# the determinism fix measured 4.5% faster and could not have shown up anywhere on this board.
#
# `steps` is now a safety cap. `time_budget_s` is the budget. Every arm in a comparison runs through
# this same generator, so they get identical wall-clock and the comparison stays fair; a faster encoder
# simply fits more steps into it, which is exactly what rule 21 says should convert to score.
_STEPS_DONE: dict = {}


def budgeted(steps, tag="arm"):
    """Yield step indices until `steps` OR CONFIG['time_budget_s'] runs out, whichever is first."""
    budget = CONFIG.get("time_budget_s", 0.0)
    t0 = time.time()
    n = 0
    for i in range(steps):
        if budget > 0 and (time.time() - t0) > budget:
            break
        n = i + 1
        yield i
    _STEPS_DONE[tag] = n


evaluate = partial(evaluate_candidate, stepper=budgeted)
evaluate_trainable = partial(train_candidate, stepper=budgeted)

def coord_encoders(dev, dim=None):
    """(earth4d, rff) coordinate encoders for the env modes, matched by construction.

    The env modes build their spatial feature from a raw (lat, lon) centroid, which is why they could
    only ever be compared against the class prior. These give them the same encoder-vs-encoder arm every
    other row reports: identical inputs, identical width, differing only in the encoder.

    Elevation and time are pinned to constants -- a species/cell centroid has no single elevation or
    timestamp -- so this measures the SPATIAL encoder, which is what these targets are about.
    """
    e4d = build_candidate_encoder(enable_relative=False).to(dev).eval()

    def _e4d(lat, lon):
        import numpy as _np
        c = torch.tensor(_np.stack([lat, lon, _np.zeros_like(lat), _np.full_like(lat, 0.5)], 1),
                         dtype=torch.float32, device=dev)
        with torch.no_grad():
            return e4d(c).cpu().numpy()

    def _rff(lat, lon):
        import numpy as _np
        rn = _np.stack([lat, lon], 1).astype(_np.float32)
        sc, base, sigmas = fair_rff(rn, FAIR_CONTROL_DIM, seed=0)
        return _rff_features(sc, base, sigmas[len(sigmas) // 2]).numpy()

    return _e4d, _rff


def cooccur_mode(a):
    import sys as _sys; _sys.path.insert(0, '/workspace')
    _dev = torch.device(a.device)
    _e4d_enc, _rff_enc = coord_encoders(_dev)
    _kw = dict(thresh=CONFIG["cooccur_thresh"], seed=a.seed, mechanism=CONFIG["cooccur_mech"],
               cooccur_file=CONFIG["cooccur_file"], env_channels=CONFIG["cooccur_channels"])
    r = cooccur_routing(CONFIG["cache_dir"], coord_encoder=_e4d_enc, **_kw)
    r_rff = cooccur_routing(CONFIG["cache_dir"], coord_encoder=_rff_enc, **_kw)
    _e4d_ap, _rff_ap = r["micro_AP_feat"], r_rff["micro_AP_feat"]
    print(f"  Earth4D {_e4d_ap:.4f} | RFF {_rff_ap:.4f} | vs RFF {_e4d_ap - _rff_ap:+.4f}")
    print(f"  query_sp={r['n_query_sp']} cand_sp={r['n_cand_sp']} feat_dim={r['feat_dim']} base_rate={r['micro_AP_baserate']:.4f}")
    print(f"  micro-AP(feat) {r['micro_AP_feat']:.4f} | micro-AP(prevalence-baseline) {r['micro_AP_prevalence']:.4f} | GAIN {r['gain_over_prevalence']:+.4f} | lift-over-baserate {r['lift_over_baserate']:.2f}x")
    print(f"  [leak-guard] {r['leak_guard']}")
    declare(
        capability="community_from_env",
        mode="COOCCUR-ROUTING",
        metric="micro_AP_feat",
        value=r["micro_AP_feat"],
        split=f"mech={r['mechanism']}",
        gains={"vs RFF": _e4d_ap - _rff_ap, "vs prior": r["gain_over_prevalence"]},
        baselines={"RFF": _rff_ap, "prior": r["micro_AP_prevalence"], "baserate": r["micro_AP_baserate"]},
        mechanism=r["mechanism"], thresh=r["thresh"], cooccur_file=r["cooccur_file"],
        n_query_sp=r["n_query_sp"], n_cand_sp=r["n_cand_sp"], feat_dim=r["feat_dim"],
        lift_over_baserate=r["lift_over_baserate"], leak_guard=r["leak_guard"],
    )
    return r

def sdm_presence_mode(a):
    import sys as _sys; _sys.path.insert(0, '/workspace')
    _e4d_enc, _rff_enc = coord_encoders(torch.device(a.device))
    _kw = dict(seed=a.seed, mechanism=CONFIG["cooccur_mech"], cooccur_file=CONFIG["cooccur_file"])
    r = sdm_presence(CONFIG["cache_dir"], coord_encoder=_e4d_enc, **_kw)
    r_rff = sdm_presence(CONFIG["cache_dir"], coord_encoder=_rff_enc, **_kw)
    _e4d_ap, _rff_ap = r["micro_AP_feat"], r_rff["micro_AP_feat"]
    print(f"  Earth4D {_e4d_ap:.4f} | RFF {_rff_ap:.4f} | vs RFF {_e4d_ap - _rff_ap:+.4f}")
    print(f"  query_cells={r['n_query_cells']} cand_sp={r['n_cand_sp']} feat_dim={r['feat_dim']} base_rate={r['micro_AP_baserate']:.4f}")
    print(f"  micro-AP(feat) {r['micro_AP_feat']:.4f} | micro-AP(prevalence-baseline) {r['micro_AP_prevalence']:.4f} | GAIN {r['gain_over_prevalence']:+.4f} | lift-over-baserate {r['lift_over_baserate']:.2f}x")
    print(f"  [leak-guard] {r['leak_guard']}")
    declare(
        capability="species_from_env",
        mode="SDM-PRESENCE",
        metric="micro_AP_feat",
        value=r["micro_AP_feat"],
        split=f"mech={r['mechanism']}",
        gains={"vs RFF": _e4d_ap - _rff_ap, "vs prior": r["gain_over_prevalence"]},
        baselines={"RFF": _rff_ap, "prior": r["micro_AP_prevalence"], "baserate": r["micro_AP_baserate"]},
        mechanism=r["mechanism"], n_query_cells=r["n_query_cells"], n_cand_sp=r["n_cand_sp"],
        feat_dim=r["feat_dim"], lift_over_baserate=r["lift_over_baserate"],
        leak_guard=r["leak_guard"],
    )
    return r

def sdm_hard_mode(a):
    import sys as _sys; _sys.path.insert(0, '/workspace')
    import numpy as _np
    runs = []
    runs_rff = []
    for sd in range(a.seed, a.seed + CONFIG["sdm_seeds"]):
        _e4d_enc, _rff_enc = coord_encoders(torch.device(a.device))
        r = sdm_presence_hard(CONFIG["cache_dir"], seed=sd, mechanism=CONFIG["cooccur_mech"], coord_encoder=_e4d_enc,
                              cell_deg=CONFIG["sdm_cell_deg"], holdout_mode=CONFIG["sdm_holdout_mode"],
                              block_deg=CONFIG["sdm_block_deg"], env_channels=CONFIG["sdm_channels"],
                              add_time=CONFIG["sdm_time"], cooccur_file=CONFIG["cooccur_file"])
        r_rff = sdm_presence_hard(CONFIG["cache_dir"], seed=sd, mechanism=CONFIG["cooccur_mech"], coord_encoder=_rff_enc,
                              cell_deg=CONFIG["sdm_cell_deg"], holdout_mode=CONFIG["sdm_holdout_mode"],
                              block_deg=CONFIG["sdm_block_deg"], env_channels=CONFIG["sdm_channels"],
                              add_time=CONFIG["sdm_time"], cooccur_file=CONFIG["cooccur_file"])
        runs.append(r); runs_rff.append(r_rff)
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
        gains={"vs RFF": float(aps.mean()) - float(_np.mean([x["micro_AP_feat"] for x in runs_rff])),
               "vs prior": float(gns.mean())},
        baselines={"RFF": float(_np.mean([x["micro_AP_feat"] for x in runs_rff])),
                   "prior": r0["micro_AP_prevalence"], "baserate": r0["micro_AP_baserate"]},
        mechanism=r0["mechanism"], env_channels=r0["env_channels"], add_time=r0["add_time"],
        sdm_seeds=CONFIG["sdm_seeds"], ap_std=float(aps.std()), gain_std=float(gns.std()),
        n_query_cells=r0["n_query_cells"], n_train_cells=r0["n_train_cells"],
        n_cand_sp=r0["n_cand_sp"], feat_dim=r0["feat_dim"], leak_guard=r0["leak_guard"],
    )
    return {'runs': runs, 'ap_mean': float(aps.mean()), 'ap_std': float(aps.std()),
            'gain_mean': float(gns.mean()), 'gain_std': float(gns.std())}


def main(argv=None):
    # A run that cannot be reproduced cannot set a record. The hash-kernel fix is necessary and not
    # sufficient -- cuBLAS, TF32, cuDNN autotune and torch's scatter kernels all sit between the encoder
    # and the loss. Pinned here, before anything allocates a CUDA context.
    _DETERMINISM = enforce_determinism(0)
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
    historical_support = a.capability in ("species_from_spacetime", "family_from_spacetime")
    need_gid = CONFIG["env"] or historical_support
    lat, lon, fam, n_fam, days, gid, sp_obs = load_obs(CONFIG["cache_dir"], CONFIG["n_shards"], with_time=CONFIG["forecast"], with_gid=need_gid)
    obs_index = np.arange(len(lat), dtype=np.int64)
    target_species = None
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
        target_species, fam = np.unique(sp_obs, return_inverse=True)  # compact species ids actually present
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
    temporal_phase = None
    if CONFIG["target"] == "family" and CONFIG["forecast"] and not CONFIG["phenology"]:
        phase_fit = np.asarray(days, dtype=np.float32)[~test]
        phase_origin = float(phase_fit.min())
        phase_span = max(float(phase_fit.max()) - phase_origin, 1.0)
        phase_values = (np.asarray(days, dtype=np.float32) - phase_origin) / phase_span
        if not np.isfinite(phase_values).all():
            raise ValueError("temporal transport phase must be finite after the forecast split")
        temporal_phase = torch.tensor(phase_values, dtype=torch.float32)
        print(f"  [orthogonal-temporal-head] train-fitted phase [0,1]; held-out phase "
              f"[{phase_values[test].min():.3f},{phase_values[test].max():.3f}]; "
              "norm-preserving pair rotations", flush=True)

    # OBJECTIVE swing: future robustness across collection regimes.  Partition TRAIN ONLY into four
    # equal-count chronological domains, then every coordinate arm uses the same balanced sampler and
    # smooth worst-domain loss.  No held-out timestamp, label, feature, or prediction enters the cutpoints.
    _train_domains = None
    _group_dro_diag = {}
    if CONFIG["target"] == "family" and CONFIG["forecast"] and not CONFIG["phenology"]:
        _train_days = np.asarray(days)[~test]
        _order = np.argsort(_train_days, kind="stable")
        _domains = np.empty(len(_order), dtype=np.int64)
        _domains[_order] = np.minimum(np.arange(len(_order)) * 4 // len(_order), 3)
        _train_domains = torch.tensor(_domains, dtype=torch.long)
        _counts = np.bincount(_domains, minlength=4)
        _group_dro_diag = {
            "group_dro_domains": 4,
            "group_dro_counts": _counts.tolist(),
            "group_dro_temperature": GROUP_DRO_TEMPERATURE,
            "group_dro_fit": "train timestamps only",
        }
        print(f"  [objective/chrono-groupdro] train-domain counts={_counts.tolist()} "
              f"temperature={GROUP_DRO_TEMPERATURE:g}", flush=True)

    # DATA intervention: historical occurrences enter only the spatial support
    # bank used by the existing soft k-NN range decoder. Optimization and the
    # fixed 2025 train/test corpus are untouched. Unknown per-row dates are
    # conservatively right-censored at the last pre-corpus day.
    support_coords = support_fam_t = support_raw = support_rn = support_partner_indices = None
    support_rows = dated_support_rows = paired_rows = 0
    if historical_support:
        hlat, hlon, hsp, hgid = load_historical_gbif_support(CONFIG["cache_dir"])
        dated = None
        if CONFIG["target"] == "species":
            xlat, xlon, xsp, xgid, xday = load_dated_gbif_support(CONFIG["cache_dir"])
            class_map = np.full(max(int(hsp.max()), int(xsp.max()), int(target_species.max())) + 1, -1,
                                dtype=np.int64)
            class_map[target_species] = np.arange(len(target_species), dtype=np.int64)
            mapped = class_map[hsp]
            keep = mapped >= 0
            xmapped = class_map[xsp]
            forecast_origin = float(np.min(days[test]))
            xkeep = np.isfinite(xday) & (xday < forecast_origin) & (xmapped >= 0)
            dated = (xlat[xkeep], xlon[xkeep], xgid[xkeep], xday[xkeep], xmapped[xkeep])
        else:
            xlat, xlon, xsp, xgid, xday = load_dated_gbif_support(CONFIG["cache_dir"])
            vocab = np.load(Path(CONFIG["cache_dir"]) / "gbif_vocab.npz", allow_pickle=True)
            global_idx = vocab["global_idx"]
            taxonomy = list(csv.DictReader(open(
                Path(CONFIG["cache_dir"]) / "derived/species_index.csv")))
            family_name = np.array([taxonomy[i]["family"] for i in global_idx])
            species_to_family = np.unique(family_name, return_inverse=True)[1].astype(np.int64)
            if max(int(hsp.max()), int(xsp.max())) >= len(species_to_family):
                raise ValueError("historical support species ids exceed the fixed taxonomy vocabulary")
            mapped = species_to_family[hsp]
            keep = mapped < n_fam
            xmapped = species_to_family[xsp]
            forecast_origin = float(np.min(days[test]))
            xkeep = np.isfinite(xday) & (xday < forecast_origin) & (xmapped < n_fam)
            dated = (xlat[xkeep], xlon[xkeep], xgid[xkeep], xday[xkeep], xmapped[xkeep])
        hlat, hlon, hgid, mapped = hlat[keep], hlon[keep], hgid[keep], mapped[keep]
        if len(np.intersect1d(gid, hgid)):
            raise ValueError("historical GBIF support overlaps the fixed 2025 corpus")
        source_last_day = float((np.datetime64("2024-12-31") - np.datetime64("1970-01-01"))
                                / np.timedelta64(1, "D"))
        if not source_last_day < float(np.min(days)):
            raise ValueError("historical GBIF support can cross the 2025 corpus origin")
        source_time = np.float32((source_last_day - tmin) / tspan)
        support_coord_np = np.stack([
            hlat, hlon, np.zeros_like(hlat), np.full_like(hlat, source_time)
        ], 1)
        if dated is not None:
            xlat, xlon, xgid, xday, xmapped = dated
            if len(np.intersect1d(gid, xgid)):
                raise ValueError("dated strict-past support overlaps the fixed corpus")
            if len(np.intersect1d(hgid, xgid)):
                raise ValueError("dated strict-past support overlaps the undated historical bank")
            if not np.all(xday < float(np.min(days[test]))):
                raise ValueError("dated support crosses the forecast origin")
            xtime = ((xday - tmin) / tspan).astype(np.float32)
            support_coord_np = np.concatenate([support_coord_np, np.stack([
                xlat, xlon, np.zeros_like(xlat), xtime
            ], 1)])
            mapped = np.concatenate([mapped, xmapped])
            dated_support_rows = len(xmapped)
        support_coords = torch.tensor(support_coord_np)
        support_fam_t = torch.tensor(mapped)
        support_rows = len(mapped)
        print(f"  [historical-range-support] {support_rows:,} disjoint pre-forecast occurrences across "
              f"{len(np.unique(mapped))} {CONFIG['target']} classes; includes {dated_support_rows:,} "
              "exact-dated strict-past rows; soft k-NN bank expanded" +
              ("; local cross-era objective enabled" if CONFIG["target"] == "species" else
               "; classifier training unchanged"),
              flush=True)
        if CONFIG["target"] == "species" and dated_support_rows:
            support_partner_indices = nearest_dated_conspecific(
                coords[~test], fam_t[~test], support_coords, support_fam_t, dated_support_rows)
            paired_rows = int((support_partner_indices >= 0).sum())
            print(f"  [local-cross-era] nearest exact-dated conspecific partner for "
                  f"{paired_rows:,}/{len(support_partner_indices):,} current training rows; "
                  "other conspecific modes masked from contrastive negatives", flush=True)

    enc = build_candidate_encoder().to(dev)
    # Capacity (R5) and throughput (R4/R21) are properties of the ENCODER, not of any one capability,
    # so they are measured once here and ride along on every declare() below.
    _SCIENCE_AXES.update(science_axes(enc, coords, dev))
    # R24 (does the encoder infer where nothing was observed?) and R2b (does the relative channel
    # transfer across absolute position?). Both are properties of the ENCODER, so like capacity and
    # throughput they are measured once here and ride on every declare() below. Each reports
    # `measurable: False` rather than a wrong number when its precondition is absent.
    try:
        _SCIENCE_AXES.update(relative_transfer(enc, coords, fam_t, dev))
    except Exception as _exc:
        _SCIENCE_AXES["axis_R2b_measurable"] = False
        _SCIENCE_AXES["axis_R2b_reason"] = f"{type(_exc).__name__}: {_exc}"[:120]
    # R1 — does consuming observed past state help, and does it SURVIVE being fed the model's own
    # output? A delayed positional basis collapses to the control once its input is synthetic.
    try:
        _SCIENCE_AXES.update(autoregressive_rollout(enc, coords, fam_t, days, test, dev))
    except Exception as _exc:
        _SCIENCE_AXES["axis_R1_measurable"] = False
        _SCIENCE_AXES["axis_R1_reason"] = f"{type(_exc).__name__}: {_exc}"[:120]
    # R5 is a FLOOR, not a readout: "small models must have no less than 100M parameters". The v4
    # champion ran ~37.7M -- one hash table, tri-planes dropped -- and nothing said so, because nothing
    # checked. A run below the floor measures a model science.md does not permit, so it declares itself
    # unfit rather than quietly setting a record.
    if not _SCIENCE_AXES["axis_R5_meets_100M_floor"]:
        print(f"[R5] *** BELOW THE science.md FLOOR: {_SCIENCE_AXES['axis_R5_params_M']}M < 100M — "
              f"not a permitted model; this run is diagnostic-only.", flush=True)
        _SCIENCE_AXES["diagnostic"] = True
        _SCIENCE_AXES["diagnostic_reason"] = (
            f"encoder has {_SCIENCE_AXES['axis_R5_params_M']}M parameters, below science.md rule 5's "
            f"100M floor")

    if CONFIG["nystrom"] > 0:
        # Fit on TRAIN rows only: the anchors would otherwise carry the evaluation set's coordinates
        # into the feature map.
        enc.fit_anchors(coords[torch.tensor(~test)].to(dev), seed=a.seed)

    # (The old --env_temporal/--env_perobs/--env_quantiles/--env_extremes/--env_spread guard is gone with
    # those flags: they only ever affected the deleted --env_trait diagnostic and were silently inert on
    # the --env path, which is how family_from_env read data-limited for 53 runs. A flag that cannot act
    # is worse than no flag.)

    if CONFIG["env"]:
        # science.md rules 1-6, 24 done RIGHT: the positional field should represent the ENVIRONMENT; biology
        # follows. Real env covariates (worldclim+soil+elev) joined by gbifID -> the science-aligned question.
        env = load_env(CONFIG["cache_dir"], gid, channels=CONFIG["env_channels"], fit_mask=~test)  # train-fit transform
        # R24 — the dense-field axis. Needs an always-available target at any coordinate, which is what
        # an env channel is (species are sparse). Held-out CELLS, so a test point has no training
        # observation nearby and the encoder has to generalise across the gap instead of looking it up.
        try:
            _SCIENCE_AXES.update(field_interpolation(enc, coords, env, dev))
        except Exception as _exc:
            _SCIENCE_AXES["axis_R24_measurable"] = False
            _SCIENCE_AXES["axis_R24_reason"] = f"{type(_exc).__name__}: {_exc}"[:120]
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
            gains={"vs RFF": e4d_acc - rff_acc,
                   "vs raw": e4d_acc - raw_acc,
                   # Channel gains, deliberately OUTSIDE the fair vocabulary: they are the env
                   # channel's advantage, not the encoder's, and when they were eligible the harness
                   # read +0.0411 as an encoder gain while Earth4D alone (0.0938) LOST to RFF (0.1010).
                   "env_channel_over_coord": env_acc - best_coord,
                   "env_channel_fused_over_coord": fus_acc - best_coord},
            baselines={"raw": raw_acc, "RFF": rff_acc, "earth4d": e4d_acc, "env": env_acc,
                       "best-coord-PE": best_coord},
            obs=len(lat), held_out=int(test.sum()), families=n_fam, env_dim=int(env.shape[1]),
            earth4d_dim=int(e4d.shape[1]), seconds=dt,
            top5={"raw": raw_t5, "rff": rff_t5, "earth4d": e4d_t5, "env": env_t5, "fused": fus_t5},
        )
        return {"st_gain": env_acc - best_coord, "st_gain_fused": fus_acc - best_coord,
                "env_acc": env_acc, "fused_acc": fus_acc, "earth4d_acc": e4d_acc, "rff_acc": rff_acc,
                "raw_acc": raw_acc, "best_coord_pe": best_coord, "obs": len(lat), "seconds": dt, "env": True}

    with torch.no_grad():
        e4d = enc(coords.to(dev)).cpu()                          # [N, output_dim] Earth4D positional features
    rn = np.stack([lat / 90.0, lon / 180.0], 1).astype(np.float32)
    if CONFIG["forecast"]:
        rn = np.concatenate([rn, tnorm[:, None]], 1)             # fair: baselines get the SAME time feature
    raw = torch.tensor(rn)                                        # raw normalized coords (+time) baseline
    if support_coords is not None:
        support_rn = np.stack([
            support_coords[:, 0].numpy() / 90.0,
            support_coords[:, 1].numpy() / 180.0,
            support_coords[:, 3].numpy(),
        ], 1).astype(np.float32)
        support_raw = torch.tensor(support_rn)
    # Random Fourier Features of (lat,lon[,t]): the fair nonlinear positional-encoding control.
    # Train-extent normalized and bandwidth-selected — see fair_rff(). The previous fixed sigma=8 on
    # globe-normalized coords was degenerate on a regional corpus and scored BELOW raw coordinates.
    _rff_scaled, _rff_base, _rff_sigmas = fair_rff(rn, FAIR_CONTROL_DIM, train_mask=~test, seed=a.seed)
    # THE ZERO-PAD NULL, asserted at runtime rather than in a test file. The control must not depend on
    # the encoder's output width: build it again as if the encoder were 25% wider and require the
    # features to be bit-identical. This is the exact check whose absence let zero-information padding
    # move a row's share from 20.7% to 27.2%.
    _pad_scaled, _pad_base, _ = fair_rff(rn, FAIR_CONTROL_DIM, train_mask=~test, seed=a.seed)
    assert np.array_equal(_rff_scaled, _pad_scaled) and np.array_equal(_rff_base, _pad_base), (
        "fair control is not width-independent — it must not read e4d.shape[1]")
    _fit_rn = rn[~test]
    _support_lo = _fit_rn.min(0)
    _support_span = np.maximum(_fit_rn.max(0) - _support_lo, 1e-6)
    _support_scaled = (None if support_rn is None else
                       ((support_rn - _support_lo) / _support_span * 2.0 - 1.0).astype(np.float32))
    _best = (-1.0, _rff_sigmas[0], None, None)
    for _sig in _rff_sigmas:
        _cand = _rff_features(_rff_scaled, _rff_base, _sig)
        _support_cand = (None if _support_scaled is None else
                         _rff_features(_support_scaled, _rff_base, _sig))
        _acc, _ = evaluate(_cand, fam_t, test, n_fam, dev, CONFIG['steps'], CONFIG['lr'], f"rff_s{_sig:g}",
                           CONFIG["head_hidden"], a.seed,
                           train_domains=_train_domains,
                           support_feats=_support_cand, support_fam=support_fam_t,
                           support_partner_indices=support_partner_indices,
                           temporal_phase=temporal_phase)
        if _acc > _best[0]:
            _best = (_acc, _sig, _cand, _support_cand)
    rff, support_rff = _best[2], _best[3]
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
        coords_ll = torch.tensor(np.stack([lat, lon], 1).astype(np.float32))
        # CRITICAL leak-guard: the phenology TARGET is the query's own day-of-year, which is derivable from the
        # query timestamp. So the QUERY-POINT features here must be SPACE-ONLY (lat,lon) -- time stripped -- else
        # a static head reads the answer off its own time coordinate (smoke test: RFF+time -> MAE 1.3d, cheating).
        # Neighbours legitimately carry their OBSERVED past DOY as explicit node state (that IS the propagation).
        rn_sp = np.stack([lat / 90.0, lon / 180.0], 1).astype(np.float32)
        raw_sp = torch.tensor(rn_sp)
        # FAIR-BASELINE FIX (phenology path). This control was `rn_sp @ N(0, 8)` on GLOBE-normalized
        # (lat/90, lon/180) -- exactly the configuration lib/fair_baseline.py documents as degenerate on a
        # regional corpus ("not a control; it is a handicap"). The fix landed in the CLASSIFICATION path
        # (fair_rff: train-extent normalization + bandwidth selection) but never here, so every
        # flowering_peak_month `vs RFF` gain was measured against the known-broken control.
        _sc, _bs, _ = fair_rff(rn_sp, FAIR_CONTROL_DIM, train_mask=~test, seed=a.seed)
        rff_sp = _rff_features(_sc, _bs, CONFIG["pheno_rff_sigma"])
        print("  [fair-baseline/pheno] RFF sigma=%g on train-extent coords (was sigma=8 on globe coords)"
              % CONFIG["pheno_rff_sigma"])
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
        # ARM flw_ntime -----------------------------------------------------------------------------
        # The record-setting head is LSTMDOY, which consumes ONLY neighbour features -- and every bank
        # above encodes neighbours at t=0, so Earth4D space-time tri-planes are switched off on the one
        # path that sets the score. A neighbour event time is PAST-ONLY under the causal window and is
        # already handed to the propagator explicitly as ndoy, so putting it into the encoder basis adds
        # no information the model did not already have -- it only lets Earth4D represent WHERE-AND-WHEN.
        # The QUERY bank stays space-only: the query own day-of-year IS the target.
        nfeat_src = None
        if CONFIG["pheno_ntime"]:
            coords_nb = torch.tensor(np.stack([lat, lon, np.zeros_like(lat), tnorm], 1).astype(np.float32))
            with torch.no_grad():
                e4d_nb = enc(coords_nb.to(dev)).cpu()
            rn_nb = np.concatenate([rn_sp, tnorm[:, None].astype(np.float32)], 1)
            raw_nb = torch.tensor(rn_nb)
            _rng2 = np.random.default_rng(0)
            _proj2 = rn_nb @ (_rng2.normal(0, 8.0, (3, e4d.shape[1] // 2)).astype(np.float32))
            rff_nb = torch.tensor(np.concatenate([np.sin(_proj2), np.cos(_proj2)], 1).astype(np.float32))
            nfeat_src = {"e4d": e4d_nb, "rff": rff_nb, "raw": raw_nb}
            print("  [flw_ntime] neighbour banks carry event time: e4d %s rff %s raw %s; queries stay space-only"
                  % (tuple(e4d_nb.shape), tuple(rff_nb.shape), tuple(raw_nb.shape)))
        _feats = phenology_feature_set(CONFIG["pheno_feats"], CONFIG["pheno_nofair"])
        # FAIR-BASELINE GUARD: a single-feature run (e.g. --pheno_feats e4d) left the RFF control untrained, so the
        # trace could report NO fair gain at all and still set a record -- this capability's records were being
        # gated on nothing. Whenever Earth4D is trained, train raw and generic-PE controls too
        # (opt out: --pheno_nofair).
        r = run_phenology_all(e4d_sp, rff_sp, raw_sp, fd, days, coords_ll, test, dev,
                              K=CONFIG["rec_k"], steps=CONFIG["steps"], lr=CONFIG["lr"], hidden=CONFIG["rec_hidden"], hops=CONFIG["gnn_hops"], tol_days=CONFIG["pheno_tol"],
                              attn=CONFIG["pheno_attn"], attn_heads=CONFIG["attn_heads"], attn_layers=CONFIG["attn_layers"], sp=sp_all,
                              block_deg=CONFIG["rec_block_deg"], fast=CONFIG["rec_fast"],
                              feats=_feats, nfeat_src=nfeat_src)
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
            gains=({"vs RFF": _e4d_best - _rff_best}
                   if _e4d_best == _e4d_best and _rff_best == _rff_best else {}),
            baselines={"RFF": _rff_best, "raw": _best_acc("raw")},
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

    raw_acc, raw_t5 = evaluate(raw, fam_t, test, n_fam, dev, CONFIG["steps"], CONFIG["lr"], "raw",
                               CONFIG["head_hidden"], a.seed,
                               train_domains=_train_domains,
                               support_feats=support_raw, support_fam=support_fam_t,
                               support_partner_indices=support_partner_indices,
                               temporal_phase=temporal_phase)
    rff_acc, rff_t5 = evaluate(rff, fam_t, test, n_fam, dev, CONFIG['steps'], CONFIG['lr'], "rff",
                               CONFIG['head_hidden'], a.seed,
                               train_domains=_train_domains,
                               support_feats=support_rff, support_fam=support_fam_t,
                               support_partner_indices=support_partner_indices,
                               temporal_phase=temporal_phase)
    e4d_acc, e4d_t5 = (evaluate_trainable(enc, coords, fam_t, test, n_fam, dev, CONFIG["steps"], CONFIG["lr"], "earth4d",
                                          CONFIG['head_hidden'], CONFIG['enc_lr_mult'], CONFIG['enc_warmup'], CONFIG['enc_c2f'],
                                          seed=a.seed, train_domains=_train_domains,
                                          support_coords=support_coords, support_fam=support_fam_t,
                                          support_partner_indices=support_partner_indices,
                                          temporal_phase=temporal_phase)
                       if CONFIG["train_encoder"] else
                       evaluate(e4d, fam_t, test, n_fam, dev, CONFIG["steps"], CONFIG["lr"], "earth4d",
                                CONFIG["head_hidden"], a.seed, train_domains=_train_domains,
                                support_feats=None, support_fam=None,
                                temporal_phase=temporal_phase))
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
        seconds=dt, target=CONFIG["target"], historical_range_support_rows=support_rows,
        dated_strict_past_support_rows=dated_support_rows,
        local_cross_era_paired_rows=paired_rows,
        local_cross_era_objective=("nearest exact-dated conspecific; disjoint modes masked"
                                   if paired_rows else "none"),
        historical_range_support="undated GBIF right-censored 2024-12-31 + exact-dated strict-past GBIF; retrieval only",
        temporal_transport_head=("shared trunk + exact block-orthogonal hidden rotation + static classifier"
                                 if temporal_phase is not None else "off"),
        temporal_phase_fit=("training observation dates only" if temporal_phase is not None else "off"),
        temporal_transport_pairs=(CONFIG["head_hidden"] // 2 if temporal_phase is not None else 0),
        top5={"raw": raw_t5, "rff": rff_t5, "earth4d": e4d_t5},
        # ...and how much of the signal actually PRESENT in the coordinates the architecture captured.
        # `captured` near 1.0 => the coordinates are exhausted, stop tuning architecture and add a
        # channel. Low `captured` with a high `ceiling` => the signal is there and the architecture is
        # failing to represent it. Low `ceiling` => the coordinates do not carry this target at all.
        **signal_capture(lat, lon, days, fam_t, test, n_fam, e4d_acc),
        **_group_dro_diag,
    )
    return {"st_gain": e4d_acc - raw_acc, "st_gain_rff": e4d_acc - rff_acc, "earth4d_acc": e4d_acc, "raw_acc": raw_acc,
            "rff_acc": rff_acc, "obs": len(lat), "seconds": dt, "forecast": CONFIG["forecast"]}


if __name__ == "__main__":
    main()
