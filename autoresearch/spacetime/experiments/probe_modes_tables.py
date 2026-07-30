"""Probe modes that never build Earth4D.

All four measure env -> identity from precomputed species/environment tables, so the encoder is not
in the comparison at all and only the DATA lever can move them. Keeping them together makes that
explicit: an agent picking species_from_env or community_from_env is choosing a channel, not an
architecture.

Each function returns the mode's native result dict and calls `declare` before returning, so the
harness reads a contract rather than the printed text.
"""
from __future__ import annotations

from deepearth.autoresearch.spacetime.harness.probe_emit import declare


def env_construct_mode(a, env_construct=None):
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

def cooccur_mode(a, env_construct=None):
    import sys as _sys; _sys.path.insert(0, '/workspace')
    from deepearth.autoresearch.spacetime.experiments.dyntargets import cooccur_routing
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

def sdm_presence_mode(a, env_construct=None):
    import sys as _sys; _sys.path.insert(0, '/workspace')
    from deepearth.autoresearch.spacetime.experiments.dyntargets import sdm_presence
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

def sdm_hard_mode(a, env_construct=None):
    import sys as _sys; _sys.path.insert(0, '/workspace')
    from deepearth.autoresearch.spacetime.experiments.dyntargets import sdm_presence_hard
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
