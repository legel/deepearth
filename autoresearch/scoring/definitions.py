"""definitions.py — what every number in this repo MEANS. One owner, outside every loop.

READ THIS BEFORE EDITING. This module is deliberately NOT inside any loop's `editable_files/`.
Every scorer used to be: `main/harness/evaluate.py` computes the net,
`main/harness/score.py` HAND-COPIES `_net_value` and `is_diagnostic` under a comment
saying "keep byte-identical", and `probes/spacetime/harness.py` defines its own
`noise_barrier` and fair-baseline selection that no other loop can see. Three definitions of the
north star, two of them in directories named for the fact that agents edit them, and nothing checking
that they agree.

science.md rule 19 already calls evaluate.py "the fixed harness that is not subject to change ... the
immutable ground truth". It was stored in the editable tree anyway. This module is that intent, given
a location that enforces it: **an experiment may not edit this file.** Changing it changes what every
past record meant, so it moves only as its own commit, with a test that fails before and passes after,
and a protocol bump on every board it re-baselines.

Two things live here:

  1. NUMERIC PRIMITIVES -- net_value / is_diagnostic / net_score / arithmetic_net (the champion suite),
     noise_barrier and strongest_fair_gain (the probe boards). Semantics are byte-identical to what
     they replace; this is a move, not a redefinition.

  2. SCIENCE_AXES -- the registry that answers "does the scoring accomplish science.md?" in code
     instead of prose. Every rule that constrains an encoder gets a row naming the instrument that
     measures it, or `None` with the reason it is unmeasured. A rule with no instrument is then a
     VISIBLE hole. Today four of six axes are holes, and the board has been climbing for months
     without that being sayable.

    python -m deepearth.autoresearch.scoring.definitions --coverage
"""
from __future__ import annotations

import argparse
import ast
import builtins
import json
import math
import os
import re
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ==================================================================================================
# 1. NUMERIC PRIMITIVES — moved verbatim; changing any of these re-baselines every board
# ==================================================================================================

SCORE_FLOOR = 1e-3    # keeps the harmonic mean finite if a benchmark reads ~0 (a zero would nuke it)


def is_diagnostic(k: str) -> bool:
    """A DERIVED difference benchmark: `B24_geo_information_gain` = B2-B1, `B56..B62_*_phylo_graph_gain`
    = capability WITH minus WITHOUT the species graph. It isolates a MECHANISM rather than measuring a
    capability, and it lives on a compressed scale, so it enters the net through `net_value` instead of
    raw."""
    return k.endswith("_gain")


def net_value(k: str, v: float) -> float:
    """The safe [0,1] contribution of benchmark `k` to the harmonic net.

    100% of the suite is included (science.md rule 32: every benchmark exists to be measured AND
    optimized). A capability metric is already in [0,1], floored so a genuine ~0 does not nuke the mean.

    An ablation-delta is mapped AFFINELY: 0.5 = neutral, 1.0 at a full +1, 0.0 at a full -1. Affine and
    not logistic, deliberately -- a logistic at scale 0.1 puts a 0.5 gain at 0.99, i.e. no headroom left
    to improve, which is the opposite of what a north star should do. NOTE FOR THE OPERATOR: science.md
    rule 32 still describes this as "(logistic, evaluate._net_value)" and `evaluate.format_benchmarks`
    still PRINTS a logistic contribution next to each delta. Both describe a formula that has not been
    the live one; the affine map below is what every stored record was computed under, so it is what
    moved here. Reconcile the prose to the code, never the reverse.
    """
    if is_diagnostic(k):
        return 0.5 + 0.5 * float(max(-1.0, min(1.0, v)))
    return max(v, SCORE_FLOOR)


def normalized(raw: Dict[str, float]) -> Dict[str, float]:
    """Each benchmark's score clipped to its OWN natural range, NaNs dropped. No baseline/target remap,
    because a hand-set target below the attainable maximum is an artificial ceiling.

    A capability metric is naturally [0,1]. An ablation-delta is a DIFFERENCE of two of them, so its
    natural range is [-1,1] -- and this used to clip every key to [0,1] regardless, on a docstring that
    asserted "every benchmark in this suite is already defined on a naturally-[0,1] metric". That was
    false for all 13 `_gain` keys, and it rectified every one of them before `net_value` ever saw them:

      * the entire negative half of `net_value`'s affine map (0.0 at a full -1) was unreachable, so a
        mechanism that HURT a capability scored identically to one that did nothing;
      * it silently defeated the deliberately-signed spacetime gains in `main/harness/hooks.py`, because
        `format_benchmarks` re-prints the clipped value and `score.parse_log`'s dict overwrite lets the
        later, clipped row win;
      * `bio_gain` -- the mean of B56..B62, and the biological loop's whole objective -- could only ever
        be biased upward, while `program.md` decides keep/discard on whether it rose.

    Clipping by kind is what makes a regression representable. `net_value` does its own [-1,1] clamp, so
    nothing downstream needs a pre-clamp; this only stops one from destroying the sign first."""
    return {k: float(min(1.0, max(-1.0 if is_diagnostic(k) else 0.0, v))) for k, v in raw.items()
            if not (isinstance(v, float) and math.isnan(v))}


def net_score(raw: Dict[str, float], suite: Optional[Iterable[str]] = None) -> float:
    """North star: HARMONIC mean (power mean p=-1) over the DECLARED suite. Harmonic so lifting the
    WEAKEST helps most and none can be traded away.

    `suite` is the declared benchmark set. Pass it. Averaging over whatever keys happen to be present
    makes the suite composition part of the result: `hooks.instrument(spacetime_gain=True)` writes six
    `*_spacetime_gain` keys into the same dict, none of them declared. Each maps through `net_value` to
    ~0.5, and a harmonic mean dominated by near-zero terms RISES when six 0.5s are added -- so a CLI
    flag moved the north star. Undeclared keys are still reported by the caller; they just cannot score.

    `suite=None` keeps the old permissive behaviour for callers that have no declared set (the probe
    boards score one capability at a time and never build a suite dict).
    """
    normed = normalized(raw)
    keys = normed.keys() if suite is None else [k for k in normed if k in set(suite)]
    vals = [net_value(k, normed[k]) for k in keys]
    if not vals:
        return 0.0
    return float(len(vals) / sum(1.0 / v for v in vals))


def arithmetic_net(raw: Dict[str, float]) -> float:
    """Arithmetic mean over CAPABILITY benchmarks only (deltas excluded). Reported alongside the
    harmonic north star: it moves when any benchmark improves, whereas the harmonic mean is pinned by
    the current weakest. autoresearch.md operator-authorizes SELECTING on this, because the harmonic net
    is single-seed noise (B20-B22 swing it 0.02<->0.13)."""
    vals = [v for k, v in normalized(raw).items() if not is_diagnostic(k)]
    return float(sum(vals) / len(vals)) if vals else 0.0


def suite_mismatch(before: Dict[str, float], after: Dict[str, float]) -> Tuple[List[str], List[str]]:
    """(added, missing) keys between two runs. Non-empty => their nets are NOT comparable (see
    net_score's caution). Every before/after report should call this first."""
    return sorted(set(after) - set(before)), sorted(set(before) - set(after))


# -- probe boards ----------------------------------------------------------------------------------

# Calibrated against noise OBSERVED in a probe, not invented. Moved here from
# `probes/spacetime/harness.py`, which kept its own copy of both the constants and the function -- the
# duplicate `noise_barrier` the audit has been warning about. The evidence:
#
#   * a second agent walked family_from_spacetime 0.1769 -> 0.19143 in seven accepted single-seed steps
#     of +0.0007 / +0.0008 / +0.0112 / +0.0005 / +0.0002 / +0.0006 / +0.0006;
#   * a verification run took flowering_peak_month 0.0521 -> 0.052131, a delta of +0.000031.
#
# MIN_CONFIRMATION_SEEDS deliberately did NOT move: it is a probe's operator policy for when to call a
# result confirmed, not a definition of what a number means, and nothing here consumes it.
MIN_REL_IMPROVEMENT = 0.015     # 1.5% of the standing record
MIN_ABS_IMPROVEMENT = 0.002     # ...and never less than this in absolute terms
SEED_SIGMA_MULTIPLE = 2.0       # with >=3 seeds, must also clear 2 sigma of the seed spread


def noise_barrier(prev: Optional[float], seed_std: Optional[float] = None, n_seeds: int = 1) -> float:
    """How much a new probe score must beat the standing record BY to count as a record at all.

    Two regimes. With >=3 seeds the spread is measurable, so the barrier is the larger of the fixed
    floor and 2 sigma -- the run has to land outside its own noise. Without them there is no spread to
    measure, so only the fixed floor applies and the result stays provisional: a single seed cannot
    distinguish a lever from a draw, whatever it scores. This exists because "beats" once meant "is a
    larger float", which is how seven consecutive +0.0006 steps were each accepted as a new best on
    family_from_spacetime before the walk was invalidated."""
    if prev is None:
        return 0.0
    floor = max(MIN_REL_IMPROVEMENT * abs(prev), MIN_ABS_IMPROVEMENT)
    if seed_std is not None and n_seeds >= 3:
        return max(floor, SEED_SIGMA_MULTIPLE * float(seed_std))
    return floor


def strongest_fair_gain(gains: Dict[str, float], order: Sequence[str]) -> Tuple[Optional[float], Optional[str]]:
    """The honest marginal: the gain against the STRONGEST fair baseline present.

    `order` decides which labels COUNT as fair; it belongs to the harness, not the probe, so a mode
    cannot nominate a flattering control for itself. Among those, the strongest baseline is the one with
    the SMALLEST gain (every gain is `encoder - baseline`, so min gain <=> max baseline). Returning the
    first label in preference order instead is a much friendlier quantity and was a real bug:
    flowering_peak_month published "+0.0128 vs RFF" while raw coordinates beat Earth4D outright, so the
    row read ENCODER-LIMITED when the honest read was INPUT-LIMITED."""
    fair = [(v, label) for label, v in gains.items()
            if v is not None and any(p.lower() in label.lower() for p in order)]
    if not fair:
        return (None, None)
    value, label = min(fair, key=lambda t: t[0])
    return (value, label)


# ==================================================================================================
# 2. THE METRIC REGISTRY — keyed by metric, because everything else is a view on it
# ==================================================================================================
#
# One row per metric. The metric NAME is the primary key, and it carries everything that used to be
# scattered: what it measures, which science.md rule demands it, which editable file moves it, and
# which probe capability estimates it cheaply.
#
# Before this, the same relationships lived in four places that disagreed -- `scorecard.md`'s
# capability table (prose), `program.md`'s LEVER_SITES (prose, and still pointing at a lib/gnn.py that
# was deleted), `score.py`'s ST_CAP/BIO_CAP partitions, and `graduation.py`'s own CAPABILITY_BENCH dict.
# Four maps, no owner. This is the owner; the others become lookups.
#
# `surface` is the routing an agent needs: pick a metric, get the file. It is declared HERE, in the
# harness, so an experiment cannot widen its own scope by editing its routing.

@dataclass(frozen=True)
class Metric:
    """One scored quantity, and everything that hangs off it."""

    name: str                             # benchmark id in evaluate.BENCHMARKS — the primary key
    measures: str                         # what the number means, in one line
    rule: str                             # the science.md rule that demands it
    surface: Tuple[str, ...] = ()         # editable file(s) that move it, relative to autoresearch/
    capability: Optional[str] = None      # probe-board row that estimates it cheaply, if any
    question: str = ""                    # the scientific question the surface is answering

    @property
    def kind(self) -> str:
        return "gain" if is_diagnostic(self.name) else "capability"

    @property
    def probed(self) -> bool:
        return self.capability is not None


_SPACETIME = "probes/spacetime/editable_files"
_BIOLOGICAL = "probes/biological/editable_files"
_MAIN = "main/editable_files"

METRICS: Tuple[Metric, ...] = (
    # ---- the coordinate encoder earns these from space and time ---------------------------------
    Metric("B5_species_from_spacetime_top10", "species identity from bare (lat,lon,elev,t), top-10",
           rule="R1 causal autoregressive forecast; R24 dense 4D field",
           surface=(_SPACETIME,), capability="species_from_spacetime",
           question="What must a coordinate become for a species to be predictable from it?"),
    Metric("B8_family_from_spacetime", "family from bare space-time, accuracy",
           rule="R1 causal autoregressive forecast; R2b relative encoder",
           surface=(_SPACETIME,), capability="family_from_spacetime",
           question="Does the encoder carry structure across a forecast boundary?"),
    Metric("B1_species_from_env_top10", "species from the environment vector, top-10",
           rule="R18 all data must lift induction",
           surface=(_SPACETIME, _MAIN), capability="species_from_env",
           question="Which environment makes a species present?"),
    Metric("B6_family_from_env", "family from environment (niche determinism), accuracy",
           rule="R18 all data must lift induction",
           surface=(_SPACETIME, _MAIN), capability="family_from_env",
           question="Is a family's niche recoverable from environment alone?"),
    Metric("B20_community_from_env_recall", "the whole local species set from environment, recall@10",
           rule="R24 dense 4D field; R18 all data must lift",
           surface=(_SPACETIME, _MAIN), capability="community_from_env",
           question="Which species co-occur here, as a set rather than one at a time?"),
    Metric("B28_flowering_peak_month_mrr", "true peak-flowering month over a 12-month sweep, MRR",
           rule="R1 causal autoregressive forecast",
           surface=(_SPACETIME,), capability="flowering_peak_month",
           question="When does a plant flower, from where and when it is?"),

    # ---- the species graph earns these from phylogeny --------------------------------------------
    #
    # Graduation targets are the B64-B67 masked-species endpoints below.  The older observation-row
    # scores remain useful capability floors, but they do not reproduce the probes' held-out-species
    # question and therefore carry no probe capability mapping.
    Metric("B7_family_from_phylo", "family from the phylogenomic embedding, accuracy",
           rule="R7 one embedding per species; R8 self-supervised on a dated tree",
           surface=(_BIOLOGICAL,),
           question="Does the embedding preserve evolutionary structure?"),
    Metric("B56_family_phylo_graph_gain", "masked-family accuracy gained from graph imputation over the seed",
           rule="R25 maskable phylo embedding; R29 exact O(N) two-pass OU-GP",
           surface=(_BIOLOGICAL,), question="Does relative reconstruction add over the raw seed?"),
    Metric("B61_trait_phylo_graph_gain", "trait macro-F1 gained from the species graph",
           rule="R25 maskable phylo embedding",
           surface=(_BIOLOGICAL,), question="Which traits are conserved enough to impute?"),
    Metric("B62_mycorrhiza_phylo_graph_gain", "mycorrhiza macro-F1 gained from the species graph",
           rule="R29 exact O(N) two-pass OU-GP",
           surface=(_BIOLOGICAL,), question="Is symbiosis phylogenetically conserved?"),
    Metric("B63_myco_from_species_f1", "mycorrhiza type given species identity, macro-F1",
           rule="R28 no fuzzy science",
           surface=(_BIOLOGICAL,),
           question="Can symbiosis be imputed from relatives?"),
    Metric("B55_pollinator_phylo_transfer_recall",
           "a plant's pollinators from its relatives' pollinators, recall@10",
           rule="R27 interactions across two trees",
           surface=(_BIOLOGICAL, _MAIN),
           question="Does interaction signal travel along phylogeny?"),
    Metric("B21_community_from_species_recall",
           "the co-occurring species set given a species identity, recall@10",
           rule="R10-12 an observation of A updates its in-context neighbours",
           surface=(_BIOLOGICAL, _MAIN),
           question="Does co-occurrence travel along phylogeny, or only along space?"),
    Metric("B41_pollinator_from_species_recall", "a plant's pollinators from its identity, recall@10",
           rule="R27 interactions across two trees",
           surface=(_BIOLOGICAL, _MAIN), capability="pollinator_from_species",
           question="Which pollinators does a plant identity imply?"),

    # ---- biological rows with no probe capability: scored, owned, but not cheaply estimable ---------
    # These are in `score.BIO_GAIN` / `score.BIO_CAP` -- they move `bio_gain` and the no-regression
    # floor -- but had no registry row at all, so `unowned()` reported them and nothing said which file
    # was responsible for moving them. Owning them without a `capability` is the honest state: the
    # surface is known, the cheap probe estimate is not.
    Metric("B57_flowering_phylo_graph_gain", "flowering AUC gained from the species graph",
           rule="R29 exact O(N) two-pass OU-GP",
           surface=(_BIOLOGICAL, _SPACETIME), question="Is phenology phylogenetically conserved?"),
    Metric("B58_lfmc_phylo_graph_gain", "LFMC correlation gained from the species graph",
           rule="R29 exact O(N) two-pass OU-GP",
           surface=(_BIOLOGICAL, _MAIN), question="Is ecophysiology phylogenetically conserved?"),
    Metric("B59_pollinator_phylo_graph_gain", "pollinator recall gained from the species graph",
           rule="R27 interactions across two trees",
           surface=(_BIOLOGICAL, _MAIN), question="Does the graph carry interaction signal?"),
    Metric("B60_community_phylo_graph_gain", "env->community recall gained from the species graph",
           rule="R24 dense 4D field; R10-12 neighbours update together",
           surface=(_BIOLOGICAL, _MAIN), question="Does the graph carry niche/community signal?"),
    Metric("B53_pollinator_calibration_mrr", "pollinator posterior calibration given species, MRR",
           rule="R28 no fuzzy science",
           surface=(_BIOLOGICAL, _MAIN), question="Is the interaction posterior honest, not just ranked?"),
    Metric("B54_pollinator_dist_kl", "pollinator visitation distribution given species, KL",
           rule="R28 no fuzzy science",
           surface=(_BIOLOGICAL, _MAIN), question="Does the predicted visitation mass match observed?"),
    # ---- biological probe->fusion bridge: same held-out-species intervention, production readouts ----
    Metric("B64_family_phylo_masked_imputation", "family of a seed-masked species from relatives, NN accuracy",
           rule="R25 maskable phylo embedding", surface=(_BIOLOGICAL, _MAIN),
           capability="family_from_phylo",
           question="Can the production graph impute a held-out species' family from relatives?"),
    Metric("B65_myco_phylo_masked_imputation_f1", "mycorrhiza of a seed-masked species from relatives, macro-F1",
           rule="R25 maskable phylo embedding; R29 exact O(N) two-pass OU-GP",
           surface=(_BIOLOGICAL, _MAIN), capability="myco_from_species",
           question="Can the production graph impute conserved symbiosis without the species seed?"),
    Metric("B66_community_phylo_masked_recall", "community of a seed-masked species from relatives, recall@10",
           rule="R10-12 neighbours update together; R25 maskable phylo embedding",
           surface=(_BIOLOGICAL, _MAIN), capability="community_from_species",
           question="Does relative reconstruction carry co-occurrence structure?"),
    Metric("B67_pollinator_phylo_masked_recall", "pollinators of a seed-masked plant from relatives, recall@10",
           rule="R27 interactions across two trees", surface=(_BIOLOGICAL, _MAIN),
           capability="pollinator_transfer",
           question="Does interaction signal reach a plant when its own species seed is hidden?"),

    # ---- production Earth4D marginal: all Earth4D channels WITH minus WITHOUT ----------------------
    # This is deliberately distinct from the standalone probe's matched-RFF architectural control.
    Metric("B1_species_spacetime_gain", "species-from-env accuracy gained FROM Earth4D",
           rule="R18 all data must lift induction", surface=(_SPACETIME, _MAIN),
           question="Does the coordinate encoder add anything the env channel does not already carry?"),
    Metric("B6_family_spacetime_gain", "family-from-env accuracy gained FROM Earth4D",
           rule="R18 all data must lift induction", surface=(_SPACETIME, _MAIN)),
    Metric("B34_lfmc_spacetime_gain", "LFMC correlation gained FROM Earth4D",
           rule="R18 all data must lift induction", surface=(_SPACETIME, _MAIN)),
    Metric("B42_mycorrhiza_spacetime_gain", "mycorrhiza macro-F1 gained FROM Earth4D",
           rule="R18 all data must lift induction", surface=(_SPACETIME, _MAIN)),
    Metric("B51_pollinator_spacetime_gain", "pollinator recall gained FROM Earth4D",
           rule="R18 all data must lift induction", surface=(_SPACETIME, _MAIN)),
    Metric("B23_calibration_spacetime_gain", "species-posterior calibration gained FROM Earth4D",
           rule="R18 all data must lift induction", surface=(_SPACETIME, _MAIN)),

    # ---- remaining spacetime score ownership -------------------------------------------------------
    Metric("B23_species_calibration_mrr", "species posterior calibration from environment, MRR",
           rule="R18 all data must lift induction", surface=(_SPACETIME, _MAIN)),
    Metric("B29_species_dist_30m_skill", "30 m species distribution skill",
           rule="R24 dense 4D field", surface=(_SPACETIME, _MAIN)),
    Metric("B39_species_dist_3km_skill", "3 km species distribution skill",
           rule="R24 dense 4D field", surface=(_SPACETIME, _MAIN)),
    Metric("B40_species_dist_300m_skill", "300 m species distribution skill",
           rule="R24 dense 4D field", surface=(_SPACETIME, _MAIN)),
    Metric("B34_lfmc_from_env", "live fuel moisture from environment",
           rule="R18 all data must lift induction", surface=(_SPACETIME, _MAIN)),
    Metric("B42_mycorrhiza_from_env", "mycorrhiza from environment, macro-F1",
           rule="R18 all data must lift induction", surface=(_SPACETIME, _MAIN)),
    Metric("B50_pollinator_from_spacetime_recall", "pollinators from bare spacetime, recall@10",
           rule="R24 dense 4D field", surface=(_SPACETIME, _MAIN)),
    Metric("B51_pollinator_from_env_recall", "pollinators from environment, recall@10",
           rule="R18 all data must lift induction", surface=(_SPACETIME, _MAIN)),
    Metric("B26_flowering_auc", "flowering from environment, ROC-AUC",
           rule="R1 causal forecast; R18 all data must lift", surface=(_SPACETIME, _MAIN)),
    Metric("B27_flowering_fidelity", "flowering agreement between imagined and observed vision",
           rule="R1 causal forecast", surface=(_SPACETIME, _MAIN)),
    Metric("B16_infer_clay_cos", "clay field reconstruction, anomaly cosine",
           rule="R24 dense 4D field", surface=(_SPACETIME, _MAIN)),
    Metric("B17_infer_soil_cos", "soil field reconstruction, anomaly cosine",
           rule="R24 dense 4D field", surface=(_SPACETIME, _MAIN)),
    Metric("B18_infer_climate_cos", "climate field reconstruction, anomaly cosine",
           rule="R24 dense 4D field", surface=(_SPACETIME, _MAIN)),
    Metric("B43_infer_hydro_cos", "hydrology field reconstruction, anomaly cosine",
           rule="R24 dense 4D field", surface=(_SPACETIME, _MAIN)),
    Metric("B44_infer_topo_cos", "topography field reconstruction, anomaly cosine",
           rule="R24 dense 4D field", surface=(_SPACETIME, _MAIN)),
    Metric("B46_infer_chm_cos", "canopy-height field reconstruction, anomaly cosine",
           rule="R24 dense 4D field", surface=(_SPACETIME, _MAIN)),
    Metric("B47_infer_naip_ir_cos", "aerial infrared reconstruction, anomaly cosine",
           rule="R24 dense 4D field", surface=(_SPACETIME, _MAIN)),
)

_BY_NAME = {m.name: m for m in METRICS}


def metric(name: str) -> Optional[Metric]:
    return _BY_NAME.get(name)


def metrics_for_surface(path_fragment: str) -> Tuple[Metric, ...]:
    """Every metric owned by a surface containing a matching file or directory.

    Registry rows name stable editable roots, so internal science files may be
    renamed or replaced without changing the fixed metric contract.
    """
    matches = {
        str(path.relative_to(AUTORESEARCH))
        for path in AUTORESEARCH.rglob("*")
        if path_fragment in str(path.relative_to(AUTORESEARCH))
    }
    return tuple(
        m
        for m in METRICS
        if any(
            path_fragment in surface
            or any(path == surface or path.startswith(surface + "/") for path in matches)
            for surface in m.surface
        )
    )


def metrics_for_capability(capability: str) -> Tuple[Metric, ...]:
    return tuple(m for m in METRICS if m.capability == capability)


def capability_to_benchmark() -> Dict[str, str]:
    """The probe->champion join, DERIVED from the registry so it cannot drift from it."""
    grouped = {cap: metrics_for_capability(cap) for cap in {m.capability for m in METRICS if m.capability}}
    duplicate = {cap: rows for cap, rows in grouped.items() if len(rows) != 1}
    if duplicate:
        detail = ", ".join(f"{cap}={[m.name for m in rows]}" for cap, rows in sorted(duplicate.items()))
        raise ValueError(f"every probe capability must have exactly one graduation benchmark: {detail}")
    return {cap: rows[0].name for cap, rows in grouped.items()}


def unowned(declared_suite: Iterable[str]) -> Tuple[str, ...]:
    """Declared benchmarks with no registry row: scored, but nobody is responsible for moving them
    and nothing says which file would. Every one is a metric the loop cannot deliberately improve."""
    return tuple(sorted(set(declared_suite) - set(_BY_NAME)))


# ==================================================================================================
# 3. SCIENCE AXES — does the scoring accomplish science.md?
# ==================================================================================================

@dataclass(frozen=True)
class Axis:
    """One science.md demand on an encoder, and the instrument that measures it.

    `instrument=None` means NOTHING measures this rule today. That is the point of the registry: a rule
    with no instrument is a hole an agent can see, rather than a silent gap the board climbs around."""
    rule: str                 # science.md rule number(s)
    demand: str               # what the rule requires of the encoder
    instrument: Optional[str] # the measurement that satisfies it, or None
    status: str               # "measured" | "unmeasured" | "unmeasurable-here"
    note: str = ""

    @property
    def is_hole(self) -> bool:
        return self.instrument is None


SCIENCE_AXES: Tuple[Axis, ...] = (
    Axis("R1", "learn spatio-temporal distributions via a CAUSAL AUTOREGRESSIVE forecaster that "
               "consumes observed past state and rolls its own predictions forward",
         instrument="autoregressive_rollout", status="measured",
         note="Three arms: positional-only (control), + observed strictly-past neighbour state, and "
              "ROLLED where that state is the model's own prediction fed back `horizon` times. A "
              "delayed basis collapses to the control once its input is synthetic; memory does not. "
              "Previously: the probe's FORECAST mode is a SPLIT (train t<0.5, test t>0.5), not "
              "autoregression. "
              "earth4d.py's own _causal_state docstring: 'does not consume observed history and "
              "therefore is not state memory or an autoregressive mechanism.' program.md's evidence "
              "standard #3 already defines the correct test; no mode implements it."),

    Axis("R2b", "a RELATIVE encoder over limited context windows going back in time "
                "(translation-equivariant, physics-inspired 4D LSTM)",
         instrument="relative_transfer", status="measured",
         note="Train on one spatial half, test on the other. An absolute encoding of a coordinate in "
              "the held region was never seen; an encoding of the OFFSET between two nearby "
              "observations is the same vector in both. axis_R2b_gain is the difference. Reports "
              "measurable=False when the encoder is built with enable_relative=False."),

    Axis("R4, R21", "remain at least as fast as it is now; speed converts directly into score under a "
                    "fixed budget, so a non-compromising speedup MUST score strictly higher",
         instrument="science_axes", status="unmeasurable-here",
         note="axis_R21_fwd_bwd_ms_per_1k_coords is measured and reported every run, so a speedup is "
              "VISIBLE. It does not yet CONVERT to score, which is what rule 21 actually demands: the "
              "budget is still CONFIG[steps]=800. budgeted() and CONFIG[time_budget_s] are wired and "
              "switched OFF, waiting on one run to measure what 800 steps costs -- flipping the budget "
              "in the same change that reshaped WHAT is measured would confound the v5 re-baseline."
              "every probe number. main uses time_budget_s=600. The one loop that owns the CUDA kernel "
              "is the one loop structurally unable to score a kernel speedup."),

    Axis("R5", "small models no less than 100M parameters",
         instrument="science_axes", status="measured",
         note="axis_R5_params_M is counted from the live module every run, and the floor BINDS: below "
              "100M the run marks itself diagnostic and cannot set a record. The v4 champion ran "
              "~37.7M -- one hash table, tri-planes dropped -- and nothing checked."
              "skips the min(max_params, prod(resolution)) cap): 18 levels x 2^20 x 2 = 37.7M per "
              "table. The species_from_spacetime champion sets drop_spatiotemporal=True, leaving ONE "
              "table: ~37.7M, below the rule-5 floor."),

    Axis("R24", "model the DENSE 4D field -- infer every variable at every space-time patch, "
                "SAMPLING BETWEEN sparse observations",
         instrument="field_interpolation", status="measured",
         note="fair_gain is discriminative accuracy at held-out observation ROWS. The split does hold "
              "out places (strict_spatiotemporal_masks), but nothing ever queries a coordinate with no "
              "observation. run_field_decode exists in lib/recurrence.py and is orphaned -- no caller. "
              "This is why the two moves that earned the last records -- drop_spatiotemporal (delete "
              "the space-time planes) and CMAC tile coding (a one-hot cell indicator that cannot "
              "interpolate BY CONSTRUCTION) -- scored as wins while moving away from the rule."),

    Axis("R32", "score AND optimize 100% of the benchmark suite; nothing excluded",
         instrument="net_score", status="measured",
         note="Satisfied on the champion suite, and the denominator is now the DECLARED suite, so an "
              "injected key set can no longer move the north star."),

    # The axis this whole three-loop design exists for.
    Axis("probe->fusion", "a probe measures what an encoder CONTRIBUTES, so that measurement must be "
                          "the same quantity fusion scores -- only cheaper",
         instrument="hooks.ablate_spacetime + B*_spacetime_gain", status="measured",
         note="The probe's matched-RFF gain is an architecture screen, while the canonical champion "
              "suite always measures a distinct production marginal: capability with all Earth4D "
              "channels minus capability with absolute and relative Earth4D removed. Graduation maps "
              "the screen to an absolute capability prediction; the ledger measures whether that "
              "prediction transfers instead of pretending the two controls are numerically identical."),

    Axis("R30", "every champion improvement reported before->after, no metric regressing",
         instrument="champion_report.format_commit", status="measured",
         note="The regression guard is real. Its record path was broken until 2026-07-31 (it pointed "
              "at a file that never existed, so every report read BASELINE)."),

    Axis("R18", "all data must lift induction; a modality that HURTS is a bug to be found",
         instrument="hooks.ST_GAIN_MAP", status="unmeasurable-here",
         note="hooks.py clamps the delta with max(0.0, ...), so an ablation that HURTS records 0.000 -- "
              "identical to one that does nothing. The instrument that would find the bug rule 18 "
              "describes is the one discarding the sign."),
)


def holes() -> Tuple[Axis, ...]:
    return tuple(a for a in SCIENCE_AXES if a.is_hole)


def coverage_report() -> str:
    lines = ["science.md axes — what the scoring can and cannot see", "=" * 78, ""]
    for a in SCIENCE_AXES:
        mark = {"measured": "OK  ", "unmeasured": "HOLE", "unmeasurable-here": "BENT"}[a.status]
        lines.append(f"{mark} {a.rule:<9} {a.demand}")
        lines.append(f"          instrument: {a.instrument or '(none)'}")
        for chunk in _wrap(a.note, 84):
            lines.append(f"          {chunk}")
        lines.append("")
    n_hole = len(holes())
    lines.append("=" * 78)
    lines.append(f"{len(SCIENCE_AXES) - n_hole}/{len(SCIENCE_AXES)} axes have an instrument; "
                 f"{n_hole} are holes.")
    lines.append("")
    lines.append("A hole is not a missing feature -- it is a direction the board CANNOT tell it is "
                 "moving in.")
    lines.append("R24 is the load-bearing one: it is the axis that would have scored "
                 "drop_spatiotemporal and CMAC tile coding as regressions instead of records.")
    return "\n".join(lines)


def _wrap(text: str, width: int) -> List[str]:
    out, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


AUTORESEARCH = next(p for p in Path(__file__).resolve().parents if p.name == "autoresearch")




def describe(m: Metric) -> str:
    out = [f"  {m.name}   [{m.kind}]",
           f"      measures    {m.measures}",
           f"      science.md  {m.rule}"]
    if m.question:
        out.append(f"      question    {m.question}")
    out.append(f"      probe row   {m.capability or '(none — champion-only, no cheap estimator)'}")
    out.append("      EDIT:")
    for s in m.surface:
        exists = (AUTORESEARCH / s).exists()
        out.append(f"        {s}{'' if exists else '   *** MISSING ON DISK ***'}")
    return "\n".join(out)


def _routing_report() -> str:
    lines = ["routing audit — the registry against the tree", "=" * 74, ""]

    missing = sorted({s for m in METRICS for s in m.surface if not (AUTORESEARCH / s).exists()})
    lines.append(f"surfaces named by a metric but absent on disk: {len(missing)}")
    lines += [f"    {s}" for s in missing]

    leaks = sorted({s for m in METRICS for s in m.surface if "/harness" in s or s.startswith("harness")})
    lines += ["", f"metrics routed INTO the harness (must be 0): {len(leaks)}"]
    lines += [f"    {s}" for s in leaks]

    # Editable files no metric claims: an agent opening one has no idea what it is scored on.
    claimed = {s.rstrip("/") for m in METRICS for s in m.surface}
    orphan_files = []
    for p in sorted(AUTORESEARCH.rglob("editable_files/**/*.py")):
        if "__pycache__" in str(p) or p.name == "__init__.py":
            continue
        rel = str(p.relative_to(AUTORESEARCH))
        if not any(rel == c or rel.startswith(c + "/") for c in claimed):
            orphan_files.append(rel)
    lines += ["", f"editable files NO metric routes to: {len(orphan_files)}"]
    lines += [f"    {o}" for o in orphan_files]

    try:
        from deepearth.autoresearch.main.harness.evaluate import BENCHMARKS
        bad = [m.name for m in METRICS if m.name not in BENCHMARKS]
        lines += ["", f"registry rows that are not declared benchmarks (must be 0): {len(bad)}"]
        lines += [f"    {b}" for b in bad]
        orphan_metrics = unowned(BENCHMARKS)
        lines += ["", f"declared benchmarks with NO registry row: {len(orphan_metrics)}/{len(BENCHMARKS)}",
                  "    (scored, but nothing says which file moves them or which rule demands them)"]
        lines += [f"    {b}" for b in orphan_metrics]
    except Exception as exc:
        lines.append(f"\n(benchmark cross-check skipped: {exc})")

    # score.py still carries its own ST_CAP/BIO_CAP/ST_GAIN partitions of the suite. They are RICHER
    # than this registry (36 benchmarks vs 12 rows), so deleting them would lose information -- but two
    # maps of the same relationships is exactly how LEVER_SITES came to point at a deleted file. Until
    # the registry absorbs them, the drift is at least VISIBLE here.
    try:
        from deepearth.autoresearch.main.harness import score as _score
        from deepearth.autoresearch.main.harness.evaluate import BENCHMARKS as _B
        lines += ["", "score.py partitions vs this registry:"]
        for part in ("ST_GAIN", "ST_CAP", "ST_SECONDARY", "BIO_GAIN", "BIO_CAP"):
            ids = getattr(_score, part, [])
            ghost = [x for x in ids if x not in _B]
            unreg = [x for x in ids if x in _B and x not in _BY_NAME]
            flag = "  *** NOT REAL BENCHMARKS" if ghost else ""
            lines.append(f"    {part:<14} {len(ids):>2} ids · {len(ghost)} nonexistent · "
                         f"{len(unreg)} real-but-unregistered{flag}")
            for x in ghost:
                lines.append(f"        ghost: {x}")
    except Exception as exc:
        lines.append(f"\n(score.py partition check skipped: {exc})")

    probed = [m for m in METRICS if m.probed]
    lines += ["", "=" * 74,
              f"{len(probed)}/{len(METRICS)} registry metrics have a cheap probe estimator.",
              "A metric with no probe row can only be moved by a 10-minute champion run."]
    return "\n".join(lines)


# ==================================================================================================
# 3b. THE AUDIT — the one command that PROVES this repo is sane
# ==================================================================================================
#
# `--audit` used to be a routing REPORT: counts a human had to read and judge. A report is not a
# check. The v5 surgery that cut earth4d.py from 977 to 719 lines shipped `if spatial_ensemble:` --
# an unconditional NameError in Earth4D.__init__ -- straight past a green report, because nothing
# below the counts ASSERTED anything. Everything here is an assertion with a PASS/FAIL line that
# names the file and the symbol, and `--audit` exits 1 when any of them fails, so a human and an
# agent get the same verdict from the same command.
#
# Nothing in this section may import torch or touch a GPU: the audit has to run on a laptop while the
# box is busy with a baseline. The two GPU-only modules (earth4d.py, probe.py) are therefore checked
# STATICALLY, by AST -- which is strictly better anyway, since a successful `import earth4d` would
# not have caught `spatial_ensemble` either. It is a constructor-time name, not an import-time one.

CHAMPION_NET = 0.32413703851749265   # net_score(main/records/champion_scores.json) — moving it re-baselines every board

# The arms and methods the v5 regex surgery removed. A survivor in CODE is a NameError waiting for the
# next run; a survivor in PROSE is a lever an agent will read, try to pull, and find does not exist.
DELETED_ARMS = ("time_film", "spatial_siren", "siren_layers", "siren_w0", "causal_lags",
                "causal_lag_span", "elm_scale", "stencil_radius", "coord_shrink",
                "spatial_ensemble", "tile_replace", "tile_time", "tile_quantile")
DELETED_METHODS = ("_siren", "_causal_state", "_film_harmonics", "fit_whiten", "fit_standardize",
                   "fit_tile_quantiles", "fit_extent")

# What each loop's editable tree is allowed to weigh. CEILINGS, checked as an inequality: this repo
# was cleaned specifically to stop file proliferation, so a directory that grows past its budget has
# to raise the number in the same commit that grows it, where a reviewer sees it.
EDITABLE_BUDGET = {"probes/spacetime/editable_files": 12,
                   "probes/biological/editable_files": 12,
                   "main/editable_files": 13}

# A scoring definition inside an editable_files/ is an experiment grading its own homework.
SCORING_DEFS = ("net_value", "net_score", "noise_barrier", "signal_capture", "science_axes",
                "is_diagnostic")

# Modules that must import on any box. earth4d/probe are excluded on purpose — they pull the CUDA
# extension in at import time, so off-box they are covered by the static checks instead.
CORE_MODULES = ("deepearth.autoresearch.scoring.definitions",
                "deepearth.autoresearch.scoring.contract",
                "deepearth.autoresearch.scoring.graduation",
                "deepearth.autoresearch.main.harness.evaluate",
                "deepearth.autoresearch.main.harness.score",
                "deepearth.autoresearch.main.harness.hooks",
                "deepearth.autoresearch.main.harness.champion_report",
                "deepearth.autoresearch.probes.spacetime.harness",
                "deepearth.autoresearch.probes.biological.harness.board")

_E4D_REL = "probes/spacetime/editable_files/earth4d.py"
_PROBE_REL = "probes/spacetime/probe.py"

# Each loop's fixed evaluator and the ONE gain label that counts as fair on its board. Every check
# below iterates this rather than naming the spacetime probe, because an audit that covers one of two
# loops is exactly as good as no audit for the loop it skips -- and the biological loop is the one
# that just grew declare() sites, a fair control and a board to protect.
# A loop may have MORE THAN ONE fixed evaluator -- the biological loop splits its capabilities across
# probe.py (family, interaction) and traitprobe.py (trait, community, symbiosis). Scanning only the
# first would report the others as unproduced, which is what "one probe file per loop" did.
LOOP_PROBES = {
    "spacetime": (("probes/spacetime/probe.py",), "vs RFF",
                  "deepearth.autoresearch.probes.spacetime.harness"),
    "biological": (("probes/biological/harness/probe.py",
                    "probes/biological/harness/traitprobe.py"), "vs null-tree",
                   "deepearth.autoresearch.probes.biological.harness.board"),
}
# Discover each loop's whole editable surface. Autoresearch may replace or rename internal science
# modules without teaching the fixed audit their filenames.
def _editable_py(rel: str) -> tuple:
    return tuple(str(path.relative_to(AUTORESEARCH))
                 for path in sorted((AUTORESEARCH / rel).rglob("*.py"))
                 if "__pycache__" not in path.parts)


_SPACETIME_PY = _editable_py(_SPACETIME)
_BIOLOGICAL_PY = _editable_py(_BIOLOGICAL)
CHANGED_FILES = tuple(dict.fromkeys((
    *(rel for rels, _l, _m in LOOP_PROBES.values() for rel in rels),
    "scoring/definitions.py", "scoring/contract.py", "scoring/graduation.py",
    "main/harness/evaluate.py", "main/harness/score.py", "probes/spacetime/harness.py",
    "probes/biological/harness/board.py", "probes/biological/harness/nulltree.py",
    *_SPACETIME_PY, *_BIOLOGICAL_PY,
)))


class _Audit:
    """PASS/FAIL accumulator. A failure prints the file and the symbol, never just a count."""

    def __init__(self) -> None:
        self.lines: List[str] = []
        self.failures: List[str] = []

    def section(self, title: str) -> None:
        self.lines += ["", title, "-" * len(title)]

    def check(self, ok: bool, name: str, detail: str = "") -> None:
        self.lines.append(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  —  {detail}" if detail else ""))
        if not ok:
            self.failures.append(name)

    def info(self, name: str, detail: str = "") -> None:
        self.lines.append(f"  ....  {name}" + (f"  —  {detail}" if detail else ""))

    def warn(self, name: str, detail: str = "") -> None:
        self.lines.append(f"  WARN  {name}" + (f"  —  {detail}" if detail else ""))

    def bullet(self, text: str) -> None:
        self.lines.append(f"          {text}")


# -- static analysis helpers (AST only: no import, no torch, no GPU) -------------------------------

def _tree(rel: str) -> ast.AST:
    return ast.parse((AUTORESEARCH / rel).read_text(), rel)


def _scope_bindings(node) -> set:
    """Every name a scope binds, flow-insensitively. Deliberately over-generous: the point is zero
    false alarms on a check whose FAIL line accuses a specific symbol of not existing."""
    out = set()
    args = getattr(node, "args", None)
    if args is not None:
        out |= {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
        out |= {a.arg for a in (args.vararg, args.kwarg) if a}
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out |= {(al.asname or al.name).split(".")[0] for al in n.names}
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            out |= set(n.names)
    return out


def undefined_names(rel: str) -> List[Tuple[int, str]]:
    """Names read in `rel` that no enclosing scope, module global, or builtin binds.

    This is the check that would have caught `spatial_ensemble` -- the deleted constructor parameter
    that regex surgery left behind as a live `if` test in Earth4D.__init__. It runs without importing
    anything, which is the only way to check a CUDA-linked module from a laptop.
    """
    hits, tree = [], _tree(rel)

    def visit(node, enclosing: set) -> None:
        scope = enclosing | _scope_bindings(node)
        stack = list(ast.iter_child_nodes(node))
        while stack:
            n = stack.pop()
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                visit(n, scope)
                continue
            if (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                    and n.id not in scope and not hasattr(builtins, n.id)):
                hits.append((n.lineno, n.id))
            stack.extend(ast.iter_child_nodes(n))

    visit(tree, {"__file__", "__name__", "__doc__"})
    return sorted(set(hits))


def _literal_dict(rel: str, name: str) -> Dict[str, object]:
    """A module-level dict literal, read WITHOUT importing the module."""
    for n in _tree(rel).body:
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == name for t in n.targets):
            return {k.value: ast.literal_eval(v) for k, v in zip(n.value.keys, n.value.values)
                    if isinstance(k, ast.Constant)}
    return {}


def _science_literal_dict(name: str) -> Dict[str, object]:
    """Discover one editable declaration by responsibility, independent of its filename."""
    found = [(rel, value) for rel in _SPACETIME_PY if (value := _literal_dict(rel, name))]
    if len(found) != 1:
        raise ValueError(f"expected one space-time declaration of {name}, found {[rel for rel, _ in found]}")
    return found[0][1]


def _e4d_defaults() -> Dict[str, object]:
    """Earth4D.__init__'s keyword defaults, by AST — the encoder's shape without constructing it."""
    cls = next(c for c in _tree(_E4D_REL).body if isinstance(c, ast.ClassDef) and c.name == "Earth4D")
    init = next(m for m in cls.body if getattr(m, "name", None) == "__init__")
    a = init.args
    names = [x.arg for x in a.posonlyargs + a.args][-len(a.defaults):] if a.defaults else []
    out = {}
    for k, d in zip(names, a.defaults):
        try:
            out[k] = ast.literal_eval(d)
        except ValueError:
            out[k] = None
    return out


def encoder_output_dim(cfg: Dict[str, object]) -> int:
    """output_dim Earth4D would report under `cfg`, computed from source rather than from a GPU.

    Mirrors __init__'s accounting exactly: one xyz block, three space-time projections, then each
    bolt-on basis. It exists so FAIR_CONTROL_DIM can be asserted equal to the encoder's real width --
    the control must be the same size as the thing it controls, and nothing checked that either.
    """
    d = _e4d_defaults()
    fpl = int(d.get("features_per_level", 2))
    get = lambda k, dflt=0: int(cfg.get(k, dflt) or 0)
    dim = get("spatial_levels") * fpl
    if not cfg.get("drop_spatiotemporal"):
        dim += get("temporal_levels") * fpl * 3
    dim += 2 * get("fourier") + 2 * get("time_harmonics")
    if get("spatial_cline"):
        dim += 3 + 2 * get("spatial_cline")
    dim += get("nystrom") * len(d.get("nystrom_scales") or (0.25, 1.0, 4.0))
    dim += get("tile") * int(cfg.get("tile_levels", d.get("tile_levels", 18))) * get("tile_offsets", 1)
    return dim


def _guarded_by(fn, needle: str) -> set:
    """`self.X` attributes guarding an `if` whose body mentions `needle`."""
    out = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.If) and needle in ast.dump(ast.Module(body=n.body, type_ignores=[])):
            out |= {a.attr for a in ast.walk(n.test) if isinstance(a, ast.Attribute)}
    return out


def _declare_sites(tree) -> List[Tuple[int, object, List[str], bool]]:
    """(line, capability, gain labels, declares itself diagnostic) for every declare() in probe.py."""
    out = []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and getattr(n.func, "id", None) == "declare"):
            continue
        kw = {k.arg: k.value for k in n.keywords}
        cap = kw.get("capability")
        cap = cap.value if isinstance(cap, ast.Constant) else None
        labels = [k.value for g in ast.walk(kw["gains"]) if isinstance(g, ast.Dict)
                  for k in g.keys if isinstance(k, ast.Constant)] if "gains" in kw else []
        diag = isinstance(kw.get("diagnostic"), ast.Constant) and kw["diagnostic"].value is True
        out.append((n.lineno, cap, labels, diag))
    return out


def _py_files(rel: str) -> List[Path]:
    return [p for p in sorted((AUTORESEARCH / rel).rglob("*.py")) if "__pycache__" not in str(p)]


_PATH_RE = re.compile(r"(?<![\w./-])((?:[\w.-]+/)+[\w.-]+\.(?:py|md|json|yaml|yml|cu|cuh|h))")


def _md_path_exists(md: Path, ref: str) -> bool:
    """A path named in prose resolves if it resolves from ANY root a reader would try."""
    stripped = ref[len("autoresearch/"):] if ref.startswith("autoresearch/") else ref
    # Every loop contributes both its own root and its editable root, derived rather than listed: the
    # hand-written list omitted `probes/biological`, so a correct reference in that loop's program.md
    # to `editable_files/phylogenomic.py` was reported as a broken path.
    roots = [AUTORESEARCH, AUTORESEARCH / "main/editable_files", md.parent]
    for loop in LOOP_PROBES:
        roots += [AUTORESEARCH / "probes" / loop, AUTORESEARCH / "probes" / loop / "editable_files"]
    return any((r / c).exists() for r in roots for c in (ref, stripped))


# -- the sections ----------------------------------------------------------------------------------

def _audit_core(A: _Audit) -> None:
    A.section("CORE — does it load, and does the north star still read the same number?")
    import importlib
    for mod in CORE_MODULES:
        try:
            importlib.import_module(mod)
            A.check(True, f"import {mod}")
        except Exception as exc:                                  # noqa: BLE001 — any failure is a failure
            A.check(False, f"import {mod}", f"{type(exc).__name__}: {exc}")
    # earth4d/probe link the CUDA extension at import; off-box that is expected, not a regression.
    for rel in (_E4D_REL, _PROBE_REL):
        A.info(f"import {rel}", "GPU-linked — covered by the static checks below, not by import")

    rec = AUTORESEARCH / "main/records/champion_scores.json"
    if not rec.exists():
        A.check(False, "champion record present", f"{rec} is missing")
        return
    scores = json.loads(rec.read_text()).get("scores", {})
    got = net_score(scores, BENCHMARKS_or_none())
    A.check(got == CHAMPION_NET, "net_score(champion_scores.json)",
            f"{got!r}" + ("" if got == CHAMPION_NET else f" != {CHAMPION_NET!r} — scoring has been redefined"))


def BENCHMARKS_or_none():
    """evaluate.BENCHMARKS if it imports, else None (net_score's permissive mode)."""
    try:
        from deepearth.autoresearch.main.harness.evaluate import BENCHMARKS
        return BENCHMARKS
    except Exception:                                             # noqa: BLE001
        return None


def _audit_scoring(A: _Audit) -> None:
    A.section("SCORING CONSISTENCY — one owner for every number")
    # ONE fair label per loop. A preference list over several is how three different quantities came to
    # share a single column of the spacetime board; the biological loop inherits the same rule so that
    # `vs seed` -- the operator measured against its own input -- can never be read as a fair gain.
    import importlib
    for loop, (_rel, fair_label, module) in sorted(LOOP_PROBES.items()):
        try:
            board = importlib.import_module(module)
            A.check(list(board.FAIR_ORDER) == [fair_label],
                    f"{loop}: FAIR_ORDER == [{fair_label!r}]", repr(board.FAIR_ORDER))
            A.check(board.PROTOCOL in board.PROTOCOL_HISTORY,
                    f"{loop}: PROTOCOL {board.PROTOCOL!r} is in PROTOCOL_HISTORY")
        except Exception as exc:                                  # noqa: BLE001
            A.check(False, f"{loop}: board constants readable", f"{type(exc).__name__}: {exc}")
    try:
        from deepearth.autoresearch.probes.spacetime import harness as ph
        cfg = _science_literal_dict("CONFIG")
        want = encoder_output_dim(cfg)
        A.check(ph.FAIR_CONTROL_DIM == want, "FAIR_CONTROL_DIM == encoder output_dim under v5 CONFIG",
                f"control {ph.FAIR_CONTROL_DIM} vs encoder {want}")
    except Exception as exc:                                      # noqa: BLE001
        A.check(False, "probe harness constants readable", f"{type(exc).__name__}: {exc}")

    # The gate, the board writer and the noise barrier must be the SHARED objects, not per-loop copies.
    try:
        from deepearth.autoresearch.probes.spacetime import harness as ph
        from deepearth.autoresearch.probes.biological.harness import board as bb
        from deepearth.autoresearch.scoring import contract as K
        from deepearth.autoresearch.scoring import definitions as D
        for loop, mod in (("spacetime", ph), ("biological", bb)):
            A.check(mod.noise_barrier is D.noise_barrier,
                    f"{loop}: noise_barrier IS definitions.noise_barrier")
            A.check(mod.ProbeResult is K.ProbeResult, f"{loop}: ProbeResult IS contract.ProbeResult")
            A.check(mod.declare is K.declare, f"{loop}: declare IS contract.declare")
    except Exception as exc:                                      # noqa: BLE001
        A.check(False, "both loops share one contract", f"{type(exc).__name__}: {exc}")

    # score.py used to HAND-COPY these under a comment saying "keep byte-identical". Identity, not
    # equality: a copy that agrees today is a copy that drifts tomorrow.
    try:
        from deepearth.autoresearch.main.harness import score as sc
        from deepearth.autoresearch.main.harness import evaluate as ev
        # Compare against the IMPORTED module, not this module's globals: under `python -m` this file
        # is `__main__`, a second module object whose functions would never be `is` the imported ones.
        from deepearth.autoresearch.scoring import definitions as D
        A.check(sc.is_diagnostic is D.is_diagnostic, "score.is_diagnostic IS definitions.is_diagnostic")
        A.check(sc._net_value is D.net_value, "score._net_value IS definitions.net_value")
        A.check(ev.is_diagnostic is D.is_diagnostic, "evaluate.is_diagnostic IS definitions.is_diagnostic")
        A.check(ev._net_value is D.net_value, "evaluate._net_value IS definitions.net_value")
    except Exception as exc:                                      # noqa: BLE001
        A.check(False, "champion scorers import the shared definitions", f"{type(exc).__name__}: {exc}")

    # A SECOND module-level def of a primitive is the drift this module exists to end. Not fatal --
    # some are deliberate suite-pinned wrappers -- but it must never be invisible.
    here = Path(__file__).resolve()
    for p in AUTORESEARCH.rglob("*.py"):
        if "__pycache__" in str(p) or p.resolve() == here:
            continue
        for n in _tree(str(p.relative_to(AUTORESEARCH))).body:
            if isinstance(n, ast.FunctionDef) and n.name in SCORING_DEFS:
                A.warn(f"second definition of {n.name}()",
                       f"{p.relative_to(AUTORESEARCH)}:{n.lineno} — definitions.py is the owner")

    # Every probe row must publish exactly one fair gain: `fair_gain` picks the MINIMUM over labels
    # matching FAIR_ORDER, so two matching entries silently change which baseline a record was gated
    # on. A site may declare itself diagnostic ONLY with a computed flag -- a literal `diagnostic=True`
    # is a row that can never record and should not be pretending to be a capability.
    for loop, (rels, fair_label, _module) in sorted(LOOP_PROBES.items()):
        for rel in rels:
            for line, cap, labels, diag in sorted(_declare_sites(_tree(rel))):
                name = f"{loop} {Path(rel).name}:{line} declare({cap or '<computed>'})"
                A.check(labels.count(fair_label) == 1,
                        f"{name} has exactly one {fair_label!r} gain", f"gains={labels}")
                A.check(not diag, f"{name} is not literally diagnostic=True")


def _excluded_capabilities(loop: str) -> dict:
    """A loop's declared-and-refused capabilities, with their reasons. Empty if it declares none."""
    import importlib
    try:
        return dict(getattr(importlib.import_module(LOOP_PROBES[loop][2]),
                            "EXCLUDED_CAPABILITIES", {}) or {})
    except Exception:                                             # noqa: BLE001
        return {}


def _audit_propagation(A: _Audit) -> None:
    A.section("PROPAGATION — can a probe finding actually reach a champion score?")
    try:
        from deepearth.autoresearch.main.harness.evaluate import BENCHMARKS
        from deepearth.autoresearch.scoring import graduation as gr
    except Exception as exc:                                      # noqa: BLE001
        A.check(False, "graduation/evaluate import", f"{type(exc).__name__}: {exc}")
        return

    known = {c for caps in gr.CAPABILITY_BENCH.values() for c in caps}

    def _declared_by(rels) -> set:
        """Capabilities a fixed evaluator actually emits.

        Some declare() sites compute their capability from a config value rather than writing a
        literal, so the module's own string constants are consulted too -- resolving the row rather
        than pretending it does not exist.
        """
        out = set()
        for rel in rels:
            tree = _tree(rel)
            literal = {c for _, c, _, _ in _declare_sites(tree) if c}
            strings = {n.value for n in ast.walk(tree)
                       if isinstance(n, ast.Constant) and isinstance(n.value, str)}
            out |= literal | (strings & known)
        return out

    # Both loops are checked. This used to be guarded by `if loop == "spacetime"`, so the biological
    # rows were only ever tested for "does the benchmark exist" -- never for whether anything actually
    # produces them, which was the state that let all five sit unreachable without a single failure.
    declared = {}
    for loop, (rels, _label, _module) in LOOP_PROBES.items():
        try:
            declared[loop] = _declared_by(rels)
        except Exception as exc:                                  # noqa: BLE001
            A.check(False, f"{loop}: fixed evaluator is parseable", f"{type(exc).__name__}: {exc}")
            declared[loop] = set()

    for loop, caps in gr.CAPABILITY_BENCH.items():
        for cap, bench in sorted(caps.items()):
            A.check(bench in BENCHMARKS, f"{loop}:{cap} -> {bench} is a declared benchmark")
            if loop in declared:
                emitted = cap in declared[loop]
                # A capability the loop explicitly refuses is not a hole -- it is a documented refusal.
                refused = cap in _excluded_capabilities(loop)
                A.check(emitted or refused,
                        f"{loop}:{cap} is produced by a declare() site, or refused with a reason",
                        "" if emitted or refused else "nothing emits it and nothing explains why")

    ghosts = [m.name for m in METRICS if m.name not in BENCHMARKS]
    A.check(not ghosts, "every Metric row names a real benchmark", ", ".join(ghosts))
    orphans = unowned(BENCHMARKS)
    A.info("benchmarks with NO Metric row", f"{len(orphans)}/{len(BENCHMARKS)} — nothing says which file moves them")


def _audit_assumptions(A: _Audit) -> None:
    A.section("ASSUMPTIONS — does the prose still describe the tree?")
    bad = []
    for loop in sorted(LOOP_PROBES):
        for md in sorted(AUTORESEARCH.glob(f"probes/{loop}/**/*.md")):
            for ref in sorted(set(_PATH_RE.findall(md.read_text()))):
                if not _md_path_exists(md, ref):
                    bad.append(f"{md.relative_to(AUTORESEARCH)} -> {ref}")
    A.check(not bad, "every path named in probes/*/**/*.md exists on disk")
    for b in bad:
        A.bullet(b)


def _audit_conciseness(A: _Audit) -> None:
    A.section("CONCISENESS — the anti-bloat budget")
    for rel, budget in EDITABLE_BUDGET.items():
        files = _py_files(rel)
        loc = sum(len(p.read_text().splitlines()) for p in files)
        A.check(len(files) <= budget, f"{rel} within its file budget",
                f"{len(files)} .py files (budget {budget}), {loc:,} lines")
    for rel in ("scoring", "main/harness", "probes/spacetime"):
        files = [p for p in _py_files(rel) if "editable_files" not in str(p)]
        A.info(f"{rel} (harness side)",
               f"{len(files)} .py files, {sum(len(p.read_text().splitlines()) for p in files):,} lines")

    # A scoring definition inside an editable_files/ is an experiment grading its own homework.
    strays = []
    for p in AUTORESEARCH.rglob("editable_files/**/*.py"):
        if "__pycache__" in str(p):
            continue
        for n in ast.walk(_tree(str(p.relative_to(AUTORESEARCH)))):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in SCORING_DEFS:
                strays.append(f"{p.relative_to(AUTORESEARCH)}:{n.lineno} def {n.name}()")
    A.check(not strays, "no scoring definition lives under an editable_files/")
    for s in strays:
        A.bullet(s)


def _audit_structure(A: _Audit) -> None:
    A.section("DIRECTORY STRUCTURE — the judge and the judged stay apart")
    leaks = sorted({str(p.relative_to(AUTORESEARCH)) for p in AUTORESEARCH.rglob("editable_files/**/harness*")
                    if "__pycache__" not in str(p)
                    and ((p.is_dir() and any(p.iterdir())) or p.suffix == ".py")})
    A.check(not leaks, "no harness/ directory or module inside any editable_files/")
    for l in leaks:
        A.bullet(f"{l} — an editable tree cannot contain its own judge")

    probe_words = ("earth4d", "phenology", "dyntargets", "traitprobe", "phylogenomic", "probe", "fusion")
    stray = [p.name for p in _py_files("scoring") if any(w in p.stem.lower() for w in probe_words)]
    A.check(not stray, "scoring/ contains no probe-specific module", ", ".join(stray))


def _audit_hygiene(A: _Audit) -> None:
    A.section("HYGIENE — what the regex surgery left behind")
    for rel in CHANGED_FILES:
        hits = undefined_names(rel)
        A.check(not hits, f"{rel} has no undefined names",
                ", ".join(f"line {ln}: {nm}" for ln, nm in hits))

    # Deleted arms/methods, in code and in prose. A dead flag in a docstring is a lever an agent will
    # try to pull; a dead `self.X` is the next NameError.
    dead = []
    for p in sorted(AUTORESEARCH.rglob("*")):
        if p.suffix not in (".py", ".md", ".yaml", ".yml") or "__pycache__" in str(p):
            continue
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            for sym in DELETED_ARMS:
                if f"--{sym}" in line or f"self.{sym}" in line or f'CONFIG["{sym}"]' in line:
                    dead.append(f"{p.relative_to(AUTORESEARCH)}:{i}  {sym}")
            for sym in DELETED_METHODS:
                if f"self.{sym}(" in line or f"def {sym}(" in line:
                    dead.append(f"{p.relative_to(AUTORESEARCH)}:{i}  {sym}")
    A.check(not dead, "no reference to a deleted arm or method survives")
    for d in dead:
        A.bullet(d)

    # Every CONFIG key the probe READS must be a key the probe DEFINES. Regex surgery deleted flags
    # and left their reads; CONFIG["enc_lr_mult"] is a KeyError on a path train_encoder=True always takes.
    cfg = _science_literal_dict("CONFIG")
    missing = sorted({(rel, n.lineno, n.slice.value)
                      for rel in (_PROBE_REL, *_SPACETIME_PY)
                      for n in ast.walk(_tree(rel))
                      if isinstance(n, ast.Subscript) and getattr(n.value, "id", None) == "CONFIG"
                      and isinstance(n.slice, ast.Constant) and n.slice.value not in cfg})
    A.check(not missing, "every CONFIG[...] read in probe.py is a defined CONFIG key")
    for rel, ln, k in missing:
        A.bullet(f"{rel}:{ln}  CONFIG[{k!r}] — KeyError at runtime")
    preset_bad = sorted({(cap, k) for cap, d in _science_literal_dict("CAPABILITY_CONFIG").items()
                         for k in d if k not in cfg})
    A.check(not preset_bad, "every CAPABILITY_CONFIG preset key exists in CONFIG",
            ", ".join(f"{c}:{k}" for c, k in preset_bad))

    # output_dim accounting vs what _forward_tensor actually concatenates. These two drifted apart
    # once already (the `self.tile*` delete), and a mismatch is a silent shape bug at the head.
    cls = next(c for c in _tree(_E4D_REL).body if isinstance(c, ast.ClassDef) and c.name == "Earth4D")
    init = next(m for m in cls.body if getattr(m, "name", None) == "__init__")
    fwd = next(m for m in cls.body if getattr(m, "name", None) == "_forward_tensor")
    counted, concatenated = _guarded_by(init, "output_dim"), _guarded_by(fwd, "'cat'")
    A.check(counted == concatenated, "earth4d output_dim accounting == _forward_tensor concatenation",
            f"counted-not-concatenated {sorted(counted - concatenated)}, "
            f"concatenated-not-counted {sorted(concatenated - counted)}")
    A.info("earth4d arms accounted", ", ".join(sorted(counted)))
    A.check(encoder_output_dim(cfg) > 0, "encoder output_dim under v5 CONFIG is computable",
            f"{encoder_output_dim(cfg)} dims")


_SECTIONS = (_audit_core, _audit_scoring, _audit_propagation, _audit_assumptions,
             _audit_conciseness, _audit_structure, _audit_hygiene)


def audit() -> Tuple[str, List[str]]:
    """The whole verdict: the routing report, then every machine-checked assertion.

    Returns (text, failures). A check that RAISES is a failure -- an audit that silently skips its
    own broken section is exactly the green report this replaced.
    """
    A = _Audit()
    for fn in _SECTIONS:
        try:
            fn(A)
        except Exception as exc:                                  # noqa: BLE001
            A.check(False, f"{fn.__name__} raised", f"{type(exc).__name__}: {exc}")
    tail = ["", "=" * 74,
            f"AUDIT: {len(A.failures)} FAILURE(S)" if A.failures else "AUDIT: ALL CHECKS PASS",
            "=" * 74]
    tail += [f"  FAIL  {f}" for f in A.failures]
    return _routing_report() + "\n" + "\n".join(A.lines + tail), A.failures


# ==================================================================================================
# 4. ENCODER MEASUREMENTS — generic, and NOT editable by the experiments they judge
# ==================================================================================================
#
# These define what a number MEANS, so they belong here rather than in a probe's editable_files/ --
# the same reason net_value and noise_barrier live here. They are also encoder-agnostic: nothing below
# knows about Earth4D, only about "a torch module" and "a categorical target over coordinates", so the
# biological loop can call them unchanged.


def enforce_determinism(seed: int = 0) -> dict:
    """Make a whole RUN reproducible, not just the hash backward.

    EARTH4D_DETERMINISTIC=1 makes the hash-grid gradient bit-identical -- verified on the box, all four
    encoders. It is not sufficient. Measured after that fix, four seed-0 runs of species_from_spacetime
    still gave 0.032906 / 0.033466 / 0.034178 / 0.036721: a spread of 0.0038, which is LARGER than the
    0.002 noise-barrier floor a record has to clear. The frozen RFF control was bit-identical across the
    same runs (0.041806530207395554), so the nondeterminism is in the TRAINED path, not the data or the
    split.

    What is left after the kernel: cuBLAS split-k reductions pick different orders per launch, TF32 lets
    matmuls use a lower-precision path chosen at runtime, cuDNN autotunes an algorithm per shape, and
    torch's scatter/index kernels have nondeterministic variants. Every one of those sits between the
    encoder output and the loss.

    This pins all of them. It costs some throughput, which is the correct trade: a number nobody can
    reproduce cannot set a record, and that is exactly why the trained protocol went unused for so long.
    Returns what it set, so a run can record the guarantee it was made under.
    """
    import os
    import random
    import torch
    # cuBLAS needs this set BEFORE the first CUDA context or the workspace is already allocated.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False        # no per-shape autotune
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False  # TF32 picks precision at runtime
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    return {
        "determinism_hash_kernel": os.environ.get("EARTH4D_DETERMINISTIC", "") in ("1", "true", "True"),
        "determinism_cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "determinism_tf32_off": True,
        "determinism_torch_algorithms": True,
    }


def _as_device(dev):
    """Accept "cuda:0" or torch.device. probe.py passes the raw --device STRING, and an axis that
    assumed the object died with AttributeError: 'str' has no attribute 'type'. Normalise once here so
    every axis takes either."""
    import torch
    return dev if hasattr(dev, "type") else torch.device(dev)


def science_axes(enc, coords, dev, warmup: int = 5, iters: int = 20) -> dict:
    """R5 (capacity) and R4/R21 (throughput). Both are cheap and both are currently unscored.

    R5: science.md says a small model is "no less than 100M parameters". Nothing has ever asserted it.
    Counted from the live module, so a config that quietly shrinks the encoder is visible immediately
    (the v4 champion ran ~37.7M -- one table, tri-planes dropped -- and nothing said so).

    R21: "speed is a first-class score lever ... a non-compromising speedup MUST score strictly
    higher". The probe budget is CONFIG["steps"], so throughput cannot move the primary metric at all.
    Reporting fwd+bwd wall-clock per 1k coords at least makes a speedup VISIBLE while the budget is
    still counted in steps.
    """
    dev = _as_device(dev)
    import time
    import torch
    n_params = sum(p.numel() for p in enc.parameters())
    hash_params = sum(p.numel() for n, p in enc.named_parameters() if n.endswith("embeddings"))

    x = coords[: min(len(coords), 65536)].to(dev)
    for _ in range(warmup):
        enc(x).sum().backward()
    if torch.device(dev).type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        enc(x).sum().backward()
    if torch.device(dev).type == "cuda":
        torch.cuda.synchronize()
    ms_per_1k = (time.time() - t0) / iters / (len(x) / 1000.0) * 1000.0
    enc.zero_grad(set_to_none=True)

    return {
        "axis_R5_params_M": round(n_params / 1e6, 2),
        "axis_R5_hash_params_M": round(hash_params / 1e6, 2),
        "axis_R5_meets_100M_floor": bool(n_params >= 100_000_000),
        "axis_R21_fwd_bwd_ms_per_1k_coords": round(ms_per_1k, 4),
        "axis_R21_deterministic": os.environ.get("EARTH4D_DETERMINISTIC", "") in ("1", "true", "True"),
    }

def signal_capture(lat, lon, days, fam, test, n_fam, encoder_acc: float, cells=(0.05, 0.1, 0.25, 0.5)) -> dict:
    """R-signal: what FRACTION of the signal present in the coordinates does the architecture capture?

    `fair_gain vs RFF` answers "did we beat this particular competitor". It cannot answer "is the
    architecture leaving signal on the table", because RFF is an arbitrary reference with no relation
    to how much structure the coordinates actually contain. So the board could not distinguish an
    encoder that had exhausted the available signal from one capturing a third of it, and there was no
    way to know when to stop pushing architecture and start adding data channels.

    This brackets the encoder between two non-parametric references, both fit on TRAIN and scored on
    TEST under the identical split:

      FLOOR    predict the train marginal argmax, ignoring position entirely.
               = the score with ZERO coordinate information.
      CEILING  the empirical conditional p(family | spatial cell), at the finest cell size that still
               has train support, backing off to coarser cells for test points whose cell is unseen.
               This is a direct estimate of the Bayes-optimal predictor given position -- a perfect
               memorizer of the training distribution -- so no function of the coordinates can
               reliably beat it on this split.

      captured = (encoder - floor) / (ceiling - floor)

    Read it as: 1.0 means the architecture has extracted everything the coordinates hold and further
    architecture work is wasted -- go get another data channel. A low value with a high ceiling means
    the signal IS there and the architecture is failing to represent it, which is the case that
    justifies more architecture. A LOW ceiling means the coordinates are simply uninformative for this
    target on this split, and no encoder will fix that.

    Cell sizes are in degrees; the backoff makes the ceiling honest rather than a memorization artifact
    (a cell containing exactly one train point would otherwise "predict" it perfectly and inflate the
    ceiling toward 1.0 while being pure overfit).
    """
    tr, te = ~test, test
    fam_np = fam.numpy() if hasattr(fam, "numpy") else np.asarray(fam)

    counts = np.bincount(fam_np[tr], minlength=n_fam)
    floor = float((fam_np[te] == int(counts.argmax())).mean())

    # finest-first backoff: each test point is predicted by the finest cell that had train support
    pred = np.full(te.sum(), -1, dtype=np.int64)
    unfilled = np.ones(te.sum(), dtype=bool)
    for deg in sorted(cells):
        key_tr = (np.floor(lat[tr] / deg).astype(np.int64) * 100003
                  + np.floor(lon[tr] / deg).astype(np.int64))
        key_te = (np.floor(lat[te] / deg).astype(np.int64) * 100003
                  + np.floor(lon[te] / deg).astype(np.int64))
        table = {}
        order = np.argsort(key_tr, kind="stable")
        ks, fs = key_tr[order], fam_np[tr][order]
        bounds = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
        for i in range(len(bounds) - 1):
            lo, hi = bounds[i], bounds[i + 1]
            if hi - lo >= 3:                       # >=3 train points, else it is memorization
                table[int(ks[lo])] = int(np.bincount(fs[lo:hi], minlength=n_fam).argmax())
        for j in np.flatnonzero(unfilled):
            hit = table.get(int(key_te[j]))
            if hit is not None:
                pred[j] = hit
                unfilled[j] = False
        if not unfilled.any():
            break
    pred[unfilled] = int(counts.argmax())          # never seen at any resolution -> the floor
    ceiling = float((pred == fam_np[te]).mean())

    # The span must clear SAMPLING NOISE before a fraction of it means anything. With span > 1e-9 a
    # target carrying NO signal read floor 0.180 / ceiling 0.210 -- a 0.03 gap that is pure binomial
    # noise on 800 test points -- and an encoder sitting at 0.21 scored "captured 1.0, headroom 0.0",
    # i.e. the loop would have been told the coordinates were exhausted when they were empty. That is
    # the exact misreading this measurement exists to prevent.
    #
    # Require the span to exceed 2 standard errors of the ceiling estimate. Below that, floor and
    # ceiling are indistinguishable and the honest answer is "no measurable signal here", not a ratio.
    # The standard error of the DIFFERENCE, not of one proportion: floor and ceiling are two estimates
    # on the same test set, and the quantity that has to clear noise is the gap between them. A single-
    # proportion SE was too permissive -- a signal-free target (floor 0.180, ceiling 0.210 on 800 test
    # points) passed a 2-SE test at 0.0288 with a span of 0.030 and reported "captured 1.0".
    #
    # The ceiling is also a MAXIMUM over four cell sizes with backoff, i.e. a selected statistic, so its
    # upward bias is real. Requiring 2 SE of the difference is the minimum honest bar; anything below it
    # means floor and ceiling are the same number and no fraction of the gap is meaningful.
    n_te = max(int(te.sum()), 1)
    se = float(np.sqrt((max(ceiling * (1 - ceiling), 0.0) + max(floor * (1 - floor), 0.0)) / n_te))
    span = ceiling - floor
    measurable = span > 2.0 * se
    captured = (encoder_acc - floor) / span if measurable else float("nan")
    return {
        "axis_signal_floor": round(floor, 6),
        "axis_signal_ceiling": round(ceiling, 6),
        "axis_signal_captured": None if captured != captured else round(captured, 4),
        "axis_signal_headroom": None if captured != captured else round(1.0 - captured, 4),
        # span <= 2*SE => floor and ceiling are the same number within noise; no fraction is reportable.
        "axis_signal_measurable": bool(measurable),
    }



def field_interpolation(enc, coords, env, dev, cell_deg: float = 0.25, steps: int = 400,
                        lr: float = 3e-3, seed: int = 0) -> dict:
    """R24 — does the encoder infer a variable at a coordinate where NOTHING was observed?

    science.md rule 24: model the dense 4D field, "sampling between sparse observations in space and
    time". Every other measurement on this board scores at held-out OBSERVATION points, so a code that
    memorises observed positions perfectly and interpolates not at all scores identically to a genuine
    field. That is exactly how CMAC tile coding -- a one-hot cell indicator that CANNOT interpolate by
    construction -- came to hold the Earth4D record, and why deleting the space-time tri-planes read as
    free.

    Whole spatial CELLS are held out, so a test coordinate has no training observation anywhere near it
    and the encoder must generalise across the gap rather than look the answer up. The target is a dense
    env channel (always available at any coordinate, unlike species), reconstructed from encoder
    features by a linear head fit on train cells only.

    Control is NEAREST-NEIGHBOUR from the train cells: the interpolation any method gets for free. A
    gain over it is evidence of a learned field; at or below it, the encoder is a lookup table.
    """
    dev = _as_device(dev)
    import torch
    torch.manual_seed(seed)
    lat, lon = np.asarray(coords[:, 0].cpu()), np.asarray(coords[:, 1].cpu())
    cell = (np.floor(lat / cell_deg).astype(np.int64) * 100003
            + np.floor(lon / cell_deg).astype(np.int64))
    uniq = np.unique(cell)
    rng = np.random.default_rng(seed)
    held = set(rng.choice(uniq, max(1, len(uniq) // 5), replace=False).tolist())
    te = np.array([c in held for c in cell])
    tr = ~te
    if tr.sum() < 32 or te.sum() < 32:
        return {"axis_R24_measurable": False}

    Y = torch.as_tensor(np.asarray(env), dtype=torch.float32, device=dev)
    m, sd = Y[torch.as_tensor(tr)].mean(0), Y[torch.as_tensor(tr)].std(0).clamp_min(1e-6)
    Y = (Y - m) / sd

    with torch.no_grad():
        F = enc(coords.to(dev)).float()
    Ftr, Fte = F[torch.as_tensor(tr)], F[torch.as_tensor(te)]
    Ytr, Yte = Y[torch.as_tensor(tr)], Y[torch.as_tensor(te)]
    head = torch.nn.Linear(F.shape[1], Y.shape[1]).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.mse_loss(head(Ftr), Ytr).backward()
        opt.step()
    with torch.no_grad():
        enc_mse = float(torch.nn.functional.mse_loss(head(Fte), Yte))

    # nearest train observation in raw space -- the free interpolation
    ll_tr = torch.as_tensor(np.stack([lat[tr], lon[tr]], 1), dtype=torch.float32, device=dev)
    ll_te = torch.as_tensor(np.stack([lat[te], lon[te]], 1), dtype=torch.float32, device=dev)
    nn_idx = torch.cdist(ll_te, ll_tr).argmin(1)
    with torch.no_grad():
        nn_mse = float(torch.nn.functional.mse_loss(Ytr[nn_idx], Yte))
    var = float(Yte.var())
    return {
        "axis_R24_measurable": True,
        "axis_R24_encoder_r2": round(1.0 - enc_mse / max(var, 1e-9), 4),
        "axis_R24_nearest_r2": round(1.0 - nn_mse / max(var, 1e-9), 4),
        "axis_R24_vs_nearest": round((nn_mse - enc_mse) / max(var, 1e-9), 4),
        "axis_R24_held_cells": len(held),
    }


def relative_transfer(enc, coords, fam, dev, steps: int = 400, lr: float = 3e-3, seed: int = 0) -> dict:
    """R2b — does the relative encoder carry a pattern ACROSS absolute position?

    science.md rule 2B: a relative encoder over "limited context windows, focused on a limited spatial
    region, going back in time". `earth4d.py` implements it (`encode_relative`) and `fusion.py:54` calls
    it; no probe mode ever has, so half of rule 2 has never been measured.

    The test is transfer. Train on one spatial half, evaluate on the other. An ABSOLUTE encoding of a
    coordinate in region B was never seen during training, so it should collapse. An encoding of the
    OFFSET between two nearby observations is the same vector in both regions, so if the relative
    channel works it should hold up. Returns the gap.

    Requires enable_relative=True; without it the axis reports unmeasurable rather than a wrong number.
    """
    dev = _as_device(dev)
    import torch
    if not getattr(enc, "enable_relative", False):
        return {"axis_R2b_measurable": False,
                "axis_R2b_reason": "encoder built with enable_relative=False"}
    torch.manual_seed(seed)
    lat = np.asarray(coords[:, 0].cpu())
    tr = lat < np.median(lat)
    te = ~tr
    y = torch.as_tensor(np.asarray(fam), dtype=torch.long, device=dev)
    n_cls = int(y.max()) + 1

    # pair each observation with its nearest neighbour INSIDE its own half, encode the offset
    def _pairs(mask):
        idx = np.flatnonzero(mask)
        ll = torch.as_tensor(np.stack([lat[idx], np.asarray(coords[:, 1].cpu())[idx]], 1),
                             dtype=torch.float32, device=dev)
        dist = torch.cdist(ll, ll)
        dist.fill_diagonal_(float("inf"))
        j = dist.argmin(1)
        a = coords.to(dev)[torch.as_tensor(idx, device=dev)]
        b = a[j]
        return a, b, y[torch.as_tensor(idx, device=dev)]

    a_tr, b_tr, y_tr = _pairs(tr)
    a_te, b_te, y_te = _pairs(te)
    with torch.no_grad():
        rel_tr = enc.encode_relative(a_tr - b_tr).float()
        rel_te = enc.encode_relative(a_te - b_te).float()
        abs_tr, abs_te = enc(a_tr).float(), enc(a_te).float()

    def _fit(Xtr, Xte):
        head = torch.nn.Linear(Xtr.shape[1], n_cls).to(dev)
        opt = torch.optim.Adam(head.parameters(), lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(head(Xtr), y_tr).backward()
            opt.step()
        with torch.no_grad():
            return float((head(Xte).argmax(1) == y_te).float().mean())

    rel, absol = _fit(rel_tr, rel_te), _fit(abs_tr, abs_te)
    return {
        "axis_R2b_measurable": True,
        "axis_R2b_relative_transfer": round(rel, 4),
        "axis_R2b_absolute_transfer": round(absol, 4),
        "axis_R2b_gain": round(rel - absol, 4),
    }



def autoregressive_rollout(enc, coords, fam, days, test, dev, K: int = 16, horizon: int = 2,
                           steps: int = 400, lr: float = 3e-3, seed: int = 0) -> dict:
    """R1 — does the model CONSUME observed past state and roll its own predictions forward?

    science.md rule 1: "a causal auto-regressive model trained to forecast future states from past
    states". program.md's evidence standard #3 sharpens it: "Consume observed past state; roll your own
    predictions forward. A positional lookup at t-lag is a delayed basis, not memory."

    Every capability on this board is a past->future SPLIT, which is not the same thing. A split asks
    "does a coordinate in the future decode?"; autoregression asks "does knowing what happened nearby
    BEFORE help, and does that survive being fed the model's own output?" Nothing has ever asked the
    second question, so rule 1 has been unmeasured while the loop reported a forecast metric.

    Three arms, identical head capacity and budget:

      POSITIONAL  encoder(query coordinate) alone.               the control -- no state consumed
      OBSERVED    + aggregated state of the K nearest STRICTLY-PAST train neighbours
      ROLLED      same, but the state is the model's OWN prediction from the previous step,
                  applied `horizon` times. This is the part that separates memory from a delayed basis:
                  a delayed basis degrades to the control as soon as its input is synthetic.

    axis_R1_gain_observed  = OBSERVED - POSITIONAL   (does history help at all?)
    axis_R1_gain_rolled    = ROLLED   - POSITIONAL   (does it survive self-feeding?)

    Neighbours are drawn from TRAIN rows only and are strictly earlier in time, so no test label and no
    future information can enter the state.
    """
    dev = _as_device(dev)
    import torch
    torch.manual_seed(seed)
    lat = np.asarray(coords[:, 0].cpu()); lon = np.asarray(coords[:, 1].cpu())
    dy = np.asarray(days); te = np.asarray(test); tr = ~te
    y = torch.as_tensor(np.asarray(fam), dtype=torch.long, device=dev)
    n_cls = int(y.max()) + 1
    if tr.sum() < 64 or te.sum() < 64:
        return {"axis_R1_measurable": False, "axis_R1_reason": "split too small"}

    with torch.no_grad():
        P = enc(coords.to(dev)).float()

    tr_idx = np.flatnonzero(tr)
    ll_tr = torch.as_tensor(np.stack([lat[tr_idx], lon[tr_idx]], 1), dtype=torch.float32, device=dev)
    d_tr = torch.as_tensor(dy[tr_idx], dtype=torch.float32, device=dev)

    def _past_state(idx, state_src):
        """Mean one-hot state of the K nearest TRAIN neighbours strictly earlier in time."""
        ll = torch.as_tensor(np.stack([lat[idx], lon[idx]], 1), dtype=torch.float32, device=dev)
        dq = torch.as_tensor(dy[idx], dtype=torch.float32, device=dev)
        dist = torch.cdist(ll, ll_tr)
        dist = dist.masked_fill(d_tr.unsqueeze(0) >= dq.unsqueeze(1), float("inf"))  # strictly past
        k = min(K, ll_tr.shape[0])
        nn = dist.topk(k, largest=False).indices
        valid = torch.isfinite(dist.gather(1, nn)).float().unsqueeze(-1)
        return (state_src[nn] * valid).sum(1) / valid.sum(1).clamp_min(1.0)

    onehot_tr = torch.nn.functional.one_hot(y[torch.as_tensor(tr_idx, device=dev)], n_cls).float()
    tr_state, te_state = _past_state(tr_idx, onehot_tr), _past_state(np.flatnonzero(te), onehot_tr)
    Ptr, Pte = P[torch.as_tensor(tr)], P[torch.as_tensor(te)]
    ytr, yte = y[torch.as_tensor(tr)], y[torch.as_tensor(te)]

    def _fit(Xtr, Xte):
        head = torch.nn.Linear(Xtr.shape[1], n_cls).to(dev)
        opt = torch.optim.Adam(head.parameters(), lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(head(Xtr), ytr).backward()
            opt.step()
        with torch.no_grad():
            return head, float((head(Xte).argmax(1) == yte).float().mean())

    _, positional = _fit(Ptr, Pte)
    head, observed = _fit(torch.cat([Ptr, tr_state], 1), torch.cat([Pte, te_state], 1))

    # ROLLED: replace the observed state with the model's own output, `horizon` times.
    rolled_state = te_state
    with torch.no_grad():
        for _ in range(horizon):
            pred = torch.softmax(head(torch.cat([Pte, rolled_state], 1)), dim=1)
            rolled_state = pred
        rolled = float((head(torch.cat([Pte, rolled_state], 1)).argmax(1) == yte).float().mean())

    return {
        "axis_R1_measurable": True,
        "axis_R1_positional": round(positional, 4),
        "axis_R1_observed": round(observed, 4),
        "axis_R1_rolled": round(rolled, 4),
        "axis_R1_gain_observed": round(observed - positional, 4),
        "axis_R1_gain_rolled": round(rolled - positional, 4),
        "axis_R1_horizon": horizon,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--metric", default="", help="what moves this metric, and what science it serves")
    ap.add_argument("--capability", default="", help="probe row -> its champion metric -> its files")
    ap.add_argument("--file", default="", help="what an editable file is responsible for")
    ap.add_argument("--audit", action="store_true", help="orphan metrics, orphan files, harness leaks")
    ap.add_argument("--coverage", action="store_true",
                    help="which science.md axes have an instrument, and which are holes")
    a = ap.parse_args(argv)
    if a.coverage:
        print(coverage_report())
        return
    if a.audit:
        text, failures = audit()
        print(text)
        # A non-zero exit is what makes this usable from a hook, a CI step, or an agent that cannot
        # read prose. "Is the repo sane?" has to be answerable without a human in the loop.
        if failures:
            raise SystemExit(1)
        return
    picked: List[Metric] = []
    if a.metric:
        m = metric(a.metric)
        if not m:
            near = [x.name for x in METRICS if a.metric.lower() in x.name.lower()]
            print(f"no registry row named {a.metric!r}."
                  + (f" did you mean: {', '.join(near)}" if near else ""))
            return
        picked = [m]
    elif a.capability:
        picked = list(metrics_for_capability(a.capability))
        if not picked:
            print(f"no metric is estimated by capability {a.capability!r}. known: "
                  + ", ".join(sorted(capability_to_benchmark())))
            return
    elif a.file:
        picked = list(metrics_for_surface(a.file))
        if not picked:
            print(f"no metric routes to a file matching {a.file!r}. Run --audit.")
            return
    if picked:
        for m in picked:
            print(describe(m))
            print()
        return

    print(f"{len(METRICS)} metrics in the registry, {len(SCIENCE_AXES)} science axes "
          f"({len(holes())} without an instrument).\n"
          f"Every file below is EDITABLE science; everything in a harness/ is the judge.\n")
    for m in METRICS:
        print(f"  {m.name:<40} {m.capability or '-':<24} {m.surface[0] if m.surface else '-'}")
    print("\n--metric X | --capability X | --file X | --audit | --coverage")


if __name__ == "__main__":
    main()
