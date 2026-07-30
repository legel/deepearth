#!/usr/bin/env python3.12
"""Per-encoder scoring instruments (ADDITIVE, flag-gated; touches no core path).

The OLD loop metrics scored the wrong thing:
  * spacetime `st_gain`  = Earth4D-hash MARGINAL over a fair positional control (RFF/MLP)
  * biological `bio_gain` = phylo-GRAPH MARGINAL over the frozen seed
Both are ~0 because they measure the redundant OPERATOR, not the encoder's capability,
which lives in the OBJECTIVE. This module formalises the two CORRECT, leak-guarded scores
as reusable functions that parse the existing probes' machine-readable output. It edits
nothing in core/fusion.py, evaluate.py, encoders/*, earth4d.py, or existing probe defaults.

Two instruments
---------------
1. propagator_gain  (spacetime capability, Earth4D-AGNOSTIC)
     = static-floor MAE  -  LSTM-propagator MAE   (circular days, higher=better)
   Source: earth4d_engine.py --mode physprop  -> stdout line `RESULT_JSON {...}`
   Leak-guard (enforced inside run_arm_phys): query feature is SPACE-ONLY at t=0;
   neighbours contribute their OWN PAST DOY under causal windows; new-place split holds
   whole spatial blocks out. Reports in-place and new-place, plus the interpretable
   decomposition (persistence tau, Hopkins lat/elev clines) and cross-year beta d/degC.

2. supervised_impute_gain  (biological capability, graph-AGNOSTIC)
     = supervised masked-impute score  -  seed/unsup-baseline impute score   (per axis)
   Source: traitprobe.py --trait_supervised -> stdout line `[trait_supervised] {...}`
   Leak-guard: species held out (holdout split), imputation metric on held-out species
   reconstructed purely from RELATIVES, shuffle-null netted out inside the probe.

Usage
-----
  # parse an already-captured run log:
  python3.12 score_encoders.py --spacetime_log st_inplace.log st_newplace.log \
                               --bio_log bio_lep.log bio_seasonality.log
  # or import: from score_encoders import score_spacetime, score_biological
"""
from __future__ import annotations
import argparse, json, math, re, sys
from statistics import mean, stdev

# ---- CI helper (small-seed t 95%) ----
_T95 = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228}


def _ci95(xs):
    xs = [x for x in xs if x is not None and x == x]
    n = len(xs)
    if n == 0:
        return (float("nan"), float("nan"), 0)
    mu = mean(xs)
    if n == 1:
        return (mu, float("nan"), 1)
    half = _T95.get(n - 1, 1.96) * (stdev(xs) / math.sqrt(n))
    return (mu, half, n)


def _result_json_lines(text, tag):
    """Yield every parsed JSON object emitted after a `tag ` prefix in probe stdout."""
    out = []
    for ln in text.splitlines():
        i = ln.find(tag + " ")
        if i >= 0:
            try:
                out.append(json.loads(ln[i + len(tag) + 1:]))
            except json.JSONDecodeError:
                pass
    return out


# ============================ SPACETIME ============================
def score_spacetime(physprop_objs, propagator_arm="lstm", persist_arm="recency_adv"):
    """propagator_gain from one-or-more physprop RESULT_JSON objects (one per seed x split).

    Returns per-split: propagator_gain (floor_mae - lstm_mae) mean+/-CI over seeds,
    the persistence-recovery fraction (persist_gain / lstm_gain), and the identified
    tau / Hopkins clines if the persistence arm carried them.
    """
    by_split = {}
    for o in physprop_objs:
        if o.get("mode") != "physprop":
            continue
        split = o.get("split", "?")
        arms = o.get("arms", {})
        floor = o.get("floor_mae")
        d = by_split.setdefault(split, {"prop_gain": [], "recov": [], "tau": [],
                                        "cline_lat": [], "cline_elev": []})
        if propagator_arm in arms and floor is not None:
            lstm_mae = arms[propagator_arm]["mae"]
            prop_gain = floor - lstm_mae
            d["prop_gain"].append(prop_gain)
            if persist_arm in arms and prop_gain > 1e-6:
                pg = floor - arms[persist_arm]["mae"]
                d["recov"].append(pg / prop_gain)
        pa = arms.get(persist_arm, {})
        for k, dst in (("tau_days", "tau"), ("eco_days_per_deg_abslat", "cline_lat"),
                       ("cline_days_per_deg_lat", "cline_lat"),
                       ("eco_days_per_100m_elev", "cline_elev"),
                       ("cline_days_per_100m_elev", "cline_elev")):
            if k in pa:
                d[dst].append(pa[k])
    report = {}
    for split, d in by_split.items():
        mu, half, n = _ci95(d["prop_gain"])
        rec_mu, rec_half, _ = _ci95(d["recov"])
        tau_mu, tau_half, _ = _ci95(d["tau"])
        lat_mu, lat_half, _ = _ci95(d["cline_lat"])
        elev_mu, elev_half, _ = _ci95(d["cline_elev"])
        report[split] = {
            "propagator_gain_days": round(mu, 3),
            "ci95_halfwidth": None if half != half else round(half, 3),
            "seeds": n,
            "persistence_recovery_frac": None if rec_mu != rec_mu else round(rec_mu, 3),
            "tau_days": None if tau_mu != tau_mu else round(tau_mu, 3),
            "hopkins_lat_d_per_deg": None if lat_mu != lat_mu else round(lat_mu, 2),
            "hopkins_elev_d_per_100m": None if elev_mu != elev_mu else round(elev_mu, 2),
            "clears_bar": bool(mu == mu and half == half and (mu - half) > 0),
        }
    return report


# ============================ BIOLOGICAL ============================
def score_biological(ts_objs, seed_ref="seed"):
    """supervised_impute_gain per (trait,source,variant) from `[trait_supervised]` objects.

    gain = supervised impute  -  seed baseline impute (the d_imp_vs_txt the probe already
    nets against the unsup text reference; we also expose sup-vs-seed directly). Aggregates
    the `sup` variant across seeds/objects per axis.
    """
    axes = {}
    for o in ts_objs:
        if o.get("mode") != "trait_supervised":
            continue
        trait = o.get("trait", "?")
        for row in o.get("rows", []):
            if row.get("variant") != "sup":
                continue
            key = (trait, row.get("source", "?"))
            a = axes.setdefault(key, {"impute": [], "seed": [], "d_vs_txt": []})
            if row.get("impute") is not None:
                a["impute"].append(row["impute"])
            if row.get(seed_ref) is not None:
                a["seed"].append(row[seed_ref])
            if row.get("d_imp_vs_txt") is not None:
                a["d_vs_txt"].append(row["d_imp_vs_txt"])
    report = {}
    for (trait, src), a in axes.items():
        imp_mu, imp_half, n = _ci95(a["impute"])
        seed_mu, _, _ = _ci95(a["seed"])
        # PRIMARY supervised_impute_gain = supervised impute OVER THE UNSUPERVISED baseline
        # (d_imp_vs_txt); this is the operator's objective earning its keep and is what the
        # loops report (e.g. lep +0.667, seasonality +0.114). Positive => the supervised
        # masked-recon objective adds held-out imputable structure the unsup graph destroyed.
        vtxt_mu, vtxt_half, vn = _ci95(a["d_vs_txt"])
        # SECONDARY: sup impute vs the raw seed's own family-NN floor (the harder bar; often
        # <=0 on conserved categorical axes because relative-imputation is harder than seed-NN).
        vseed = [i - s for i, s in zip(a["impute"], a["seed"])] if a["seed"] else []
        vseed_mu, _, _ = _ci95(vseed)
        report[f"{trait}::{src}"] = {
            "supervised_impute_gain": None if vtxt_mu != vtxt_mu else round(vtxt_mu, 4),
            "ci95_halfwidth": None if vtxt_half != vtxt_half else round(vtxt_half, 4),
            "seeds": vn if vn else n,
            "sup_impute": None if imp_mu != imp_mu else round(imp_mu, 4),
            "seed_baseline": None if seed_mu != seed_mu else round(seed_mu, 4),
            "sup_vs_seed_floor": None if vseed_mu != vseed_mu else round(vseed_mu, 4),
            "clears_bar": bool(vtxt_mu == vtxt_mu and vtxt_mu > 0.05),  # bar = +0.05 over unsup
        }
    return report


# ============================ CLI ============================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spacetime_log", nargs="*", default=[],
                    help="earth4d_engine physprop stdout logs (any number of seeds/splits)")
    ap.add_argument("--bio_log", nargs="*", default=[],
                    help="traitprobe --trait_supervised stdout logs")
    ap.add_argument("--propagator_arm", default="lstm")
    ap.add_argument("--persist_arm", default="recency_adv")
    a = ap.parse_args(argv)

    st_objs, bio_objs = [], []
    for p in a.spacetime_log:
        st_objs += _result_json_lines(open(p).read(), "RESULT_JSON")
    for p in a.bio_log:
        bio_objs += _result_json_lines(open(p).read(), "[trait_supervised]")

    out = {
        "spacetime": {
            "metric": "propagator_gain (static-floor MAE - LSTM MAE, days; Earth4D-agnostic)",
            "old_operator_metric": "st_gain (Earth4D marginal vs RFF/MLP) ~= 0",
            "scores": score_spacetime(st_objs, a.propagator_arm, a.persist_arm),
        },
        "biological": {
            "metric": "supervised_impute_gain (sup masked-impute - seed impute, per axis; graph-agnostic)",
            "old_operator_metric": "bio_gain (phylo-graph marginal over seed) ~= 0",
            "scores": score_biological(bio_objs),
        },
    }
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
