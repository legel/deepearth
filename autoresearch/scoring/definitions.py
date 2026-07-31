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
import math
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
    """Each benchmark's score clipped to [0,1], NaNs dropped. Every benchmark in this suite is already
    defined on a naturally-[0,1] metric, so the score IS the raw value -- no baseline/target remap,
    because a hand-set target below the attainable maximum is an artificial ceiling."""
    return {k: float(min(1.0, max(0.0, v))) for k, v in raw.items()
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

def noise_barrier(prev: Optional[float], seed_std: Optional[float] = None, n_seeds: int = 1) -> float:
    """How much a new probe score must beat the standing record BY to count as a record at all.

    2% of the record, never below 0.002; with >=3 seeds the measured spread also has to clear 2 sigma.
    Without >=3 seeds there is no spread to measure, so only the fixed floor applies and the result
    stays provisional -- a single seed cannot distinguish a lever from a draw. This exists because
    "beats" once meant "is a larger float", which is how seven consecutive +0.0006 steps were each
    accepted as a new best on family_from_spacetime before the walk was invalidated."""
    if prev is None:
        return 0.0
    floor = max(0.02 * abs(prev), 0.002)
    if seed_std is not None and n_seeds >= 3:
        return max(floor, 2.0 * seed_std)
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


_E4D = "probes/spacetime/editable_files/earth4d.py"
_PROBE = "probes/spacetime/editable_files/probe.py"
_REC = "probes/spacetime/editable_files/lib/recurrence.py"
_PHEN = "probes/spacetime/editable_files/lib/phenology.py"
_DYN = "probes/spacetime/editable_files/lib/dyntargets.py"
_PHYLO = "probes/biological/editable_files/phylogenomic.py"
_TRAIT = "probes/biological/editable_files/lib/traitprobe.py"
_FUSION = "main/editable_files/fusion/fusion.py"
_DATA = "main/editable_files/lib/data.py"

METRICS: Tuple[Metric, ...] = (
    # ---- the coordinate encoder earns these from space and time ---------------------------------
    Metric("B5_species_from_spacetime_top10", "species identity from bare (lat,lon,elev,t), top-10",
           rule="R1 causal autoregressive forecast; R24 dense 4D field",
           surface=(_E4D, _PROBE, _REC), capability="species_from_spacetime",
           question="What must a coordinate become for a species to be predictable from it?"),
    Metric("B8_family_from_spacetime", "family from bare space-time, accuracy",
           rule="R1 causal autoregressive forecast; R2b relative encoder",
           surface=(_E4D, _PROBE, _REC), capability="family_from_spacetime",
           question="Does the encoder carry structure across a forecast boundary?"),
    Metric("B1_species_from_env_top10", "species from the environment vector, top-10",
           rule="R18 all data must lift induction",
           surface=(_DYN, _PROBE, _DATA), capability="species_from_env",
           question="Which environment makes a species present?"),
    Metric("B6_family_from_env", "family from environment (niche determinism), accuracy",
           rule="R18 all data must lift induction",
           surface=(_DYN, _PROBE, _DATA), capability="family_from_env",
           question="Is a family's niche recoverable from environment alone?"),
    Metric("B20_community_from_env_recall", "the whole local species set from environment, recall@10",
           rule="R24 dense 4D field; R18 all data must lift",
           surface=(_DYN, _DATA), capability="community_from_env",
           question="Which species co-occur here, as a set rather than one at a time?"),
    Metric("B28_flowering_peak_month_mrr", "true peak-flowering month over a 12-month sweep, MRR",
           rule="R1 causal autoregressive forecast",
           surface=(_PHEN, _E4D), capability="flowering_peak_month",
           question="When does a plant flower, from where and when it is?"),

    # ---- the species graph earns these from phylogeny --------------------------------------------
    Metric("B7_family_from_phylo", "family from the phylogenomic embedding, accuracy",
           rule="R7 one embedding per species; R8 self-supervised on a dated tree",
           surface=(_PHYLO,), question="Does the embedding preserve evolutionary structure?"),
    Metric("B56_family_phylo_graph_gain", "family accuracy gained FROM the species graph",
           rule="R29 exact O(N) two-pass OU-GP",
           surface=(_PHYLO,), question="Does graph refinement add over the raw seed?"),
    Metric("B61_trait_phylo_graph_gain", "trait macro-F1 gained from the species graph",
           rule="R25 maskable phylo embedding",
           surface=(_PHYLO, _TRAIT), question="Which traits are conserved enough to impute?"),
    Metric("B62_mycorrhiza_phylo_graph_gain", "mycorrhiza macro-F1 gained from the species graph",
           rule="R29 exact O(N) two-pass OU-GP",
           surface=(_PHYLO, _TRAIT), question="Is symbiosis phylogenetically conserved?"),
    Metric("B63_myco_from_species_f1", "mycorrhiza type given species identity, macro-F1",
           rule="R28 no fuzzy science",
           surface=(_TRAIT,), question="Can symbiosis be imputed from relatives?"),
    Metric("B55_pollinator_phylo_transfer_recall",
           "a plant's pollinators from its relatives' pollinators, recall@10",
           rule="R27 interactions across two trees",
           surface=(_PHYLO, _FUSION), question="Does interaction signal travel along phylogeny?"),
)

_BY_NAME = {m.name: m for m in METRICS}


def metric(name: str) -> Optional[Metric]:
    return _BY_NAME.get(name)


def metrics_for_surface(path_fragment: str) -> Tuple[Metric, ...]:
    """Every metric a given editable file is responsible for. The routing an agent needs."""
    return tuple(m for m in METRICS if any(path_fragment in s for s in m.surface))


def metrics_for_capability(capability: str) -> Tuple[Metric, ...]:
    return tuple(m for m in METRICS if m.capability == capability)


def capability_to_benchmark() -> Dict[str, str]:
    """The probe->champion join, DERIVED from the registry so it cannot drift from it."""
    return {m.capability: m.name for m in METRICS if m.capability}


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
         instrument=None, status="unmeasured",
         note="The probe's FORECAST mode is a SPLIT (train t<0.5, test t>0.5), not autoregression. "
              "earth4d.py's own _causal_state docstring: 'does not consume observed history and "
              "therefore is not state memory or an autoregressive mechanism.' program.md's evidence "
              "standard #3 already defines the correct test; no mode implements it."),

    Axis("R2b", "a RELATIVE encoder over limited context windows going back in time "
                "(translation-equivariant, physics-inspired 4D LSTM)",
         instrument=None, status="unmeasured",
         note="earth4d.py implements enable_relative / encode_relative / fit_relative_window. "
              "probe.py never references any of them. The capability exists and is unscored."),

    Axis("R4, R21", "remain at least as fast as it is now; speed converts directly into score under a "
                    "fixed budget, so a non-compromising speedup MUST score strictly higher",
         instrument=None, status="unmeasured",
         note="The probe budget is CONFIG['steps']=800, not wall-clock, so throughput is invisible to "
              "every probe number. main uses time_budget_s=600. The one loop that owns the CUDA kernel "
              "is the one loop structurally unable to score a kernel speedup."),

    Axis("R5", "small models no less than 100M parameters",
         instrument=None, status="unmeasured",
         note="Nothing asserts a floor. Computed from hashgrid.py's allocation rule (learned probing "
              "skips the min(max_params, prod(resolution)) cap): 18 levels x 2^20 x 2 = 37.7M per "
              "table. The species_from_spacetime champion sets drop_spatiotemporal=True, leaving ONE "
              "table: ~37.7M, below the rule-5 floor."),

    Axis("R24", "model the DENSE 4D field -- infer every variable at every space-time patch, "
                "SAMPLING BETWEEN sparse observations",
         instrument=None, status="unmeasured",
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

    # The axis this whole three-loop design exists for, and the one with no instrument.
    Axis("probe->fusion", "a probe measures what an encoder CONTRIBUTES, so that measurement must be "
                          "the same quantity fusion scores -- only cheaper",
         instrument=None, status="unmeasured",
         note="The probe scores fair_gain = Earth4D minus the strongest fair baseline (a MARGINAL). "
              "The champion scores B5/B8 = absolute accuracy of the full model, with no encoder "
              "ablation in the declared suite. Those are different quantities, so a probe gain has no "
              "defined relationship to a benchmark score and `graduation.py` is currently comparing a "
              "marginal against an absolute. The fusion-side counterpart already exists half-built: "
              "hooks.ablate_spacetime computes capability WITH Earth4D minus WITHOUT, which IS "
              "fair_gain at full-model scale. It is not in BENCHMARKS, it is flag-gated behind "
              "--st-gain, and it is clamped at max(0.0, ...). Declare those six keys, drop the clamp, "
              "and the probe becomes a cheap ESTIMATOR of the expensive quantity -- which is the only "
              "thing that would justify running it at all."),

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


def audit() -> str:
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
# 4. ENCODER MEASUREMENTS — generic, and NOT editable by the experiments they judge
# ==================================================================================================
#
# These define what a number MEANS, so they belong here rather than in a probe's editable_files/ --
# the same reason net_value and noise_barrier live here. They are also encoder-agnostic: nothing below
# knows about Earth4D, only about "a torch module" and "a categorical target over coordinates", so the
# biological loop can call them unchanged.

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
    import time
    import torch
    import time
    import torch
    n_params = sum(p.numel() for p in enc.parameters())
    hash_params = sum(p.numel() for n, p in enc.named_parameters() if n.endswith("embeddings"))

    x = coords[: min(len(coords), 65536)].to(dev)
    for _ in range(warmup):
        enc(x).sum().backward()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        enc(x).sum().backward()
    if dev.type == "cuda":
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
        print(audit())
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
