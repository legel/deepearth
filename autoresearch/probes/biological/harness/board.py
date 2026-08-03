"""The biological loop's judge: what may be optimized, what a record must beat, and where it is written.

This is the biological counterpart of `probes/spacetime/harness.py`. Until it existed, this loop had a
probe that printed numbers and nothing else -- no result contract, no gate, no board, no generated
scorecard -- which is why `program/scorecard.txt` said in plain text that every biological number in the
repository should be treated as unverified. It asked for three things:

    1. a result contract, so a number carries its own identity (capability, mode, split, metric)
    2. a record gate, so a score is only compared like-for-like
    3. a generated scorecard, regenerated from the board after every run

All three now come from `autoresearch/scoring/contract.py`, shared with the spacetime loop rather than
reimplemented here. What this file owns is only what is genuinely biological: which capabilities are
reachable, which are refused and why, the protocol version, the board paths, and -- the substantive
one -- what counts as a FAIR baseline for a phylogenetic operator.

THE FAIR CONTROL. `FAIR_ORDER = ["vs null-tree"]`, one entry, for the same reason the spacetime loop
collapsed its own to one: a preference list over several labels is how three different quantities came
to share a single column of that board. The null tree is defined and built in `nulltree.py`; the short
version is that the control is the SAME operator with the SAME parameters and budget, run on a tree
that is not the phylogeny. `vs seed` -- the old `bio_gain` -- is still reported, but it is not fair and
cannot set a record: it is the operator measured against its own input.

Editing this file changes what a biological number means. Do that as its own commit, with a test that
fails before and passes after -- never inside an experiment.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

# Paths by NAME, never by counting parents: a parents[N] off-by-one is what once pointed a board
# outside its own loop, at a file that did not exist, which the harness then created empty, found no
# prior record in, and reported "RECORD = YES (new best!)" against for every capability.
_HERE = Path(__file__).resolve()
AUTORESEARCH = next(p for p in _HERE.parents if p.name == "autoresearch")
REPO = AUTORESEARCH.parent
LOOP = AUTORESEARCH / "probes" / "biological"
assert (LOOP / "program").is_dir(), f"biological loop not found at {LOOP}"
assert REPO.name == "deepearth", f"expected the deepearth package root, resolved {REPO}"
sys.path.insert(0, str(REPO.parent))

from deepearth.autoresearch.scoring import contract                              # noqa: E402
from deepearth.autoresearch.scoring.contract import (                            # noqa: E402,F401
    ContractError, Primary, ProbeResult, code_provenance, declare, noise_barrier,
)

RECORDS = LOOP / "records" / "records.json"
SCORECARD_TXT = LOOP / "program" / "scorecard.txt"
PROBE_MODULE = "deepearth.autoresearch.probes.biological.harness.probe"


# ============================================================================================================
# 1. CAPABILITIES — what may be optimized, and what is refused, with the reason
# ============================================================================================================
#
# These are the rows `graduation.LOOP_CAPABILITIES["biological"]` names, minus the one the standalone
# probe cannot honestly measure. The list and the metric registry are one contract: a capability here
# must have a `Metric(..., capability=...)` row in `scoring/definitions.py`, or it can never graduate.
CAPABILITIES = [
    "family_from_phylo",        # family from relatives, held-out species' own seed masked
    "myco_from_species",        # mycorrhiza type from species identity (traitprobe --myco_supervised)
    "community_from_species",   # co-occurrence partner set from species identity (--cooccur)
    "pollinator_transfer",      # a plant's pollinators from its relatives' (probe --objective interaction)
]

EXCLUDED_CAPABILITIES = {
    "pollinator_from_species":
        "not separable from pollinator_transfer under this probe, and the probe has no env channel. "
        "The only readout of a plant's pollinator set here is run_interaction's held-out-plant "
        "reconstruction, which IS transfer-from-relatives; the unmasked variant scores a plant whose "
        "own identity trained the bilinear head, i.e. memorization rather than an encoder effect. Its "
        "benchmark B41 is 'plant identity + env -> pollinators', and env is exactly what the standalone "
        "probe does not have -- the same reason the spacetime loop refuses pollinator_from_env",
}


# ============================================================================================================
# 2. PROTOCOL — what a run MEASURES, versioned
# ============================================================================================================
#
# Bump whenever a change alters what is measured rather than how well it does. A run under a new
# protocol RE-BASELINES a capability instead of "beating" it.
#
#   v0-unverified : everything before this file existed. The probe printed `bio_gain = graph(ON) -
#                   graph._seed()` and nothing gated, stored or regenerated it. Those numbers have no
#                   identity, no fair control and no board; they are not records and cannot be migrated.
#   v1-nulltree   : the first measured protocol. Fair gain is against the strongest member of the
#                   null-tree family (nulltree.py), not against the seed. Holdout is random over
#                   species (`split="species-random"`).
#
# ON THE ROADMAP, and it will re-baseline every row when it lands: `v2-cladeblock`. The holdout is
# currently random over species, so a held-out species' congeners stay in TRAIN -- on a family-NN
# metric that is close to memorizing the sister species, and much of the seed's ~0.89 is exactly that.
# The biological analogue of the spacetime spatiotemporal-block split is to hold out whole genera or
# families. Declaring `split="species-random"` now is what makes the eventual change visible as a
# re-baseline rather than an unexplained drop.
PROTOCOL = "v1-nulltree"
PROTOCOL_HISTORY = ("v0-unverified", "v1-nulltree")
assert PROTOCOL in PROTOCOL_HISTORY, f"PROTOCOL {PROTOCOL!r} missing from PROTOCOL_HISTORY"
REBASELINE_PROTOCOLS = frozenset(PROTOCOL_HISTORY[:PROTOCOL_HISTORY.index(PROTOCOL)])

# ONE entry. See the module docstring, and nulltree.py for what the control actually is.
FAIR_ORDER = ["vs null-tree"]

MIN_CONFIRMATION_SEEDS = 2      # operator policy: one seed screens, two matched seeds confirm


# ============================================================================================================
# 3. LOOP BINDINGS — the shared contract, told which board it is judging
# ============================================================================================================

def _set_result_sink(path, capability, protocol, args, config=None):
    return contract._set_result_sink(path, capability, protocol, args, config, loop="biological")


def _read_records(path=None):
    return contract._read_records(path or RECORDS)


def _commit_records_if_unchanged(expected_raw, records, path=None):
    return contract._commit_records_if_unchanged(expected_raw, records, path or RECORDS)


def retire_record(capability, reason, path=None):
    return contract.retire_record(capability, reason, path or RECORDS)


def _bottleneck(fair, primary, seed_score=0.0, refined_seed_norm=None) -> str:
    """Which lever the measurement points at, with the SEED as the floor.

    `seed_score` is the part of `primary` that the real tree and every null tree inherit identically --
    neither can earn it, so it must come out of the denominator. Family-NN accuracy is ~0.89 here and
    almost all of it is the frozen text prior, leaving ~0.11 for the operator to contest; the plain
    `fair / primary` ratio the spacetime loop uses would therefore cap the share at ~12% and stamp
    EVERY biological row ENCODER-LIMITED however good the operator was. That is the same failure as the
    old absolute `primary < 0.20` rule, re-entering through the denominator.

    OPERATOR-INERT is its own verdict rather than a flavour of ENCODER-LIMITED because it is a
    different failure and points at a different fix. It says the graph did not move the embedding AT
    ALL over its own seed -- `refined_seed_norm` at zero, the operator having learned the identity map.
    That is the state `program.md` records as today's, and no change to the CONTROL can affect it: the
    mechanism is not engaging, so there is nothing yet to compare against a null tree.
    """
    if refined_seed_norm is not None and abs(float(refined_seed_norm)) <= 1e-8:
        return ("OPERATOR-INERT: the graph moved nothing over its own seed (check [profile] "
                "refined_seed_norm ≈ 0) → ARCHITECTURE lever, the operator is not engaging")
    return contract._bottleneck(
        fair, primary, floor=seed_score, encoder="the real phylogeny",
        input_lever=("change the SEED (rule 26: reseed orthogonal to tree topology) or change the "
                     "TARGET to an axis the seed does not already saturate"))


def _read_of(record: dict, gain, score) -> str:
    return contract._read_of(record, gain, score,
                             fair_keywords=("null-tree", "null", "dendrogram", "perm"),
                             stale=(("vs seed", "null-tree"),),
                             floor=record.get("seed_score") or 0.0,
                             encoder="the real phylogeny")


def write_scorecard(recs: dict, path: Path | None = None) -> Path:
    return contract.write_scorecard(
        recs, path or SCORECARD_TXT, CAPABILITIES, PROTOCOL,
        title="BIOLOGICAL PROBE — SCORECARD",
        legend=("read:  INPUT-LIMITED    loses to a tree of the same shape carrying no phylogeny",
                "                        -> DATA lever, change the seed or the target",
                "       ENCODER-LIMITED  wins, but <25% of the contested headroom",
                "                        -> ARCHITECTURE lever, change the operator",
                "       EARNING          >=25% of the contested headroom is the real tree",
                "       OPERATOR-INERT   the graph moved nothing at all over its own seed"),
        read_of=_read_of)


def _print_net_scorecard(recs: dict, current: str = "") -> None:
    contract._print_net_scorecard(recs, current, CAPABILITIES,
                                  "biological encoder-probe records so far")


def _aggregate_results(paths) -> ProbeResult:
    """Combine matched per-seed results without selecting the best rerun."""
    results = [ProbeResult.read(path) for path in paths]
    if not results:
        raise ContractError("at least one result is required")
    first = results[0]
    for result in results[1:]:
        if result.identity() != first.identity():
            raise ContractError(
                f"cannot aggregate unlike biological results: {first.identity()} != {result.identity()}")
    if len(results) == 1:
        return first

    def mean_map(name):
        maps = [getattr(result, name) for result in results]
        keys = set.intersection(*(set(mapping) for mapping in maps)) if maps else set()
        out = {}
        for key in keys:
            values = [mapping[key] for mapping in maps]
            out[key] = None if any(value is None for value in values) else sum(values) / len(values)
        return out

    primary_values = [result.primary.value for result in results]
    extras = dict(first.extras)
    extras["primary_seed_values"] = primary_values
    return replace(
        first,
        primary=Primary(first.primary.name, sum(primary_values) / len(primary_values)),
        gains=mean_map("gains"),
        baselines=mean_map("baselines"),
        extras=extras,
        seed=None,
        diagnostic=any(result.diagnostic for result in results),
        diagnostic_reason="; ".join(
            result.diagnostic_reason for result in results if result.diagnostic_reason),
    )


# ============================================================================================================
# 4. CLI
# ============================================================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="biological probe board")
    ap.add_argument("--capability", default="", help=f"one of: {', '.join(CAPABILITIES)}")
    ap.add_argument("--scorecard", action="store_true", help="regenerate scorecard.txt and exit")
    ap.add_argument("--list-capabilities", action="store_true")
    ap.add_argument("--result-json", nargs="+", default=[],
                    help="gate one screen or aggregate matched per-seed ProbeResults")
    ap.add_argument("--tag", default="bio_run")
    a = ap.parse_args(argv)

    if a.list_capabilities:
        for cap in CAPABILITIES:
            print(f"  {cap}")
        for cap, why in sorted(EXCLUDED_CAPABILITIES.items()):
            print(f"  {cap}  REFUSED — {why}")
        return 0

    if a.scorecard:
        print(f"[board] wrote {write_scorecard(_read_records()[1])}")
        return 0

    if a.result_json:
        if a.capability not in CAPABILITIES:
            raise SystemExit(f"[board] {a.capability!r} is not recordable. "
                             f"Legal: {', '.join(CAPABILITIES)}"
                             + (f"\n[board] REFUSED: {EXCLUDED_CAPABILITIES[a.capability]}"
                                if a.capability in EXCLUDED_CAPABILITIES else ""))
        try:
            result = _aggregate_results(a.result_json)
        except ContractError as exc:
            raise SystemExit(f"[board] {exc}") from exc
        raw, recs = _read_records()
        seed_score = float(result.baselines.get("seed") or 0.0)
        refined_seed_norm = result.extras.get("refined_seed_norm")
        primary_seed_values = result.extras.get("primary_seed_values") or ()
        fair, _ = result.fair_gain(FAIR_ORDER)
        decision = contract.commit_result(
            recs, a.capability, result, fair_order=FAIR_ORDER, protocol=PROTOCOL,
            rebaseline_protocols=REBASELINE_PROTOCOLS, repo=REPO, tag=a.tag,
            bottleneck=_bottleneck(fair, result.primary.value, seed_score, refined_seed_norm),
            seed_values=primary_seed_values,
            min_confirmation_seeds=MIN_CONFIRMATION_SEEDS)
        recs[a.capability].setdefault("seed_score", seed_score)
        if not _commit_records_if_unchanged(raw, recs):
            raise SystemExit("[board] records.json changed while gating — rerun")
        print(json.dumps(decision, indent=2))
        print(f"[board] wrote {write_scorecard(recs)}")
        _print_net_scorecard(recs, a.capability)
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
