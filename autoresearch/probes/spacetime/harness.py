"""The spacetime loop, in one file.

This is the harness: it declares what a run measures, runs the probe, decides whether the result may
become a record, writes the ledger, and publishes to the swarm. It was six files
(probe_contract, probe_emit, probe_registry, trace, plus two mode modules) which meant six places to
look for one question. A harness should be one file you can read start to finish.

    READ ──► PICK ──► DIAGNOSE ──► RUN ──► MEASURE ──► DECIDE ──► WRITE

Sections, in that order:

  1. CAPABILITIES   what may be optimized, and what is refused, with the reason
  2. CONTRACT       ProbeResult: identity, validation, fair-gain, rendering
  3. EMIT           declare() -- the ONE path by which a number becomes recordable
  4. REGISTRY       capability -> modes -> what each needs -> where to edit
  5. GATE           like-for-like comparison, re-baseline, atomic board commit
  6. PUBLISH        Ensue upsert
  7. CLI            the loop itself

What the harness enforces is measurement identity, not a menu of permitted edits: a run is comparable
to a record only when capability, mode, split, n_shards and protocol all match. Change anything the
hypothesis needs; a mismatch is recorded as a re-baseline or withheld, never as a win.

Editing THIS file changes what a number means. Do that as its own commit, with a test that fails before
and passes after -- never inside an experiment.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

# Paths are derived by NAME, not by counting parents. Every restructure this file has been through broke
# a parents[N]: REPO once resolved to "/" and the board once resolved outside its own loop, which made
# trace.py mint a record against an empty file. Named anchors cannot drift when a directory moves.
_HERE = Path(__file__).resolve()
AUTORESEARCH = next(p for p in _HERE.parents if p.name == "autoresearch")
REPO = AUTORESEARCH.parent                  # the deepearth package root
# The loop this harness JUDGES, named outright. It used to be `_HERE.parents[1]`, which was correct only
# while the harness lived inside the loop it scored. Moving it to autoresearch/scoring/ -- so that an
# experiment cannot edit its own judge -- made parents[1] resolve to autoresearch/, i.e. RECORDS pointed
# at autoresearch/records/records.json: a file that does not exist, which the harness would have created
# empty, found no prior record in, and reported "RECORD = YES (new best!)" against for every capability.
# Exactly the failure this comment block was already warning about. A judge that lives outside its
# subject must name its subject.
LOOP = AUTORESEARCH / "probes" / "spacetime"
assert (LOOP / "program").is_dir(), f"spacetime loop not found at {LOOP}"
assert REPO.name == "deepearth", f"expected the deepearth package root, resolved {REPO}"
sys.path.insert(0, str(REPO.parent))        # dir holding the deepearth package



# ============================================================================================================
# 1b. FAIR BASELINE — what every `vs RFF` gain is measured against
# ============================================================================================================
#
# The fair baseline — what every `vs RFF` gain is measured against.
#
# This is not a helper. It defines what "fair-gain" MEANS on this board, and therefore what every EARNING
# or ENCODER-LIMITED read is asserting. It lives apart from probe.py so it can be tested without CUDA,
# because a baseline nobody can test is how the previous one stayed broken.
#
# Folded in from lib/fair_baseline.py: 46 lines in their own module, imported by exactly one
# caller, is not a boundary — it is a file to keep in sync.
import numpy as np
import torch

# The control's width is now FIXED instead of tracking the encoder's.
#
# Both call sites used to pass `e4d.shape[1]`, so the baseline's width moved whenever any arm changed
# the encoder's output width -- and RFF accuracy is non-monotone in width. Measured: padding the encoder
# with columns of LITERAL ZEROS, adding no information at all, moved family_from_spacetime's share from
# 20.7% (dim 2592) to 27.2% (dim 3024) to 15.1% (dim 3744). Since `share = fair_gain / score` is the
# number that chooses DATA vs ARCHITECTURE, an unknown fraction of every EARNING / ENCODER-LIMITED read
# on this board was an artifact of output width.
#
# v4 set this to 2592, the width family_from_spacetime's record was set at, on the reasoning that ONE
# control everywhere makes gains comparable across rows. That reasoning was sound while encoder widths
# varied wildly per capability (144 to 20,663).
#
# Under v5 they no longer vary. The probe builds exactly what fusion.py:302 builds -- 36 spatial + 108
# tri-plane = 144 dims -- for EVERY capability, with the bolt-on bases off. So the two goals stop
# competing: 144 is both a fixed protocol constant AND matched to the encoder, and Earth4D vs RFF is
# finally a head-to-head at equal width.
#
# Leaving it at 2592 would have handed the control 18x the encoder's capacity and crushed every gain --
# the same class of error as the old dimension-MATCHED control, pointing the other way.
#
# If an experiment turns a bolt-on basis back on, the encoder gets wider and the control does NOT
# follow. That is correct: the control is a fixed reference, and an arm that buys width has to earn it.
FAIR_CONTROL_DIM = 144


def fair_rff(rn: np.ndarray, out_dim: int, train_mask=None, seed: int = 0,
             bandwidths=(1.0, 4.0, 16.0, 64.0, 256.0, 1024.0)):
    """A random-Fourier control that is actually FAIR.

    The old control was `rn @ N(0, 8)` where rn is (lat/90, lon/180): coordinates normalized to the
    GLOBE. On a regional corpus that is degenerate. California spans ~9.5 deg of latitude, i.e. ~0.05 of
    the normalized range, so at sigma=8 the projection varies ~0.04 CYCLES across the entire dataset --
    every sample gets nearly the same feature. Measured: it scored 0.008, BELOW the raw-coordinate
    baseline at 0.0166. A nonlinear control that loses to raw coordinates is not a control; it is a
    handicap, and every `vs RFF` gain computed against it was inflated by that handicap.

    Two fixes, both of which just extend to the baseline the courtesy the encoder already gets:

      1. Normalize to the TRAIN extent, exactly as the encoder's GeoAdaptiveRange does, so the control
         sees the data at its actual scale rather than as a speck of the globe. Fit on train rows only —
         using the full extent would leak the evaluation range into the control's features.
      2. Select the bandwidth. The encoder gets its hyperparameters chosen; a baseline pinned to one
         arbitrary sigma is not the strongest fair baseline, it is a straw man. Pick the sigma that
         maximizes the control's own held-out fit, and let the encoder beat THAT.

    Returns (features, chosen_sigma). Selection is on the same split the arms are scored on, which is
    generous to the baseline by design: an encoder gain that survives a baseline tuned on the evaluation
    split is not an artifact of the baseline being weak.
    """
    rn = np.asarray(rn, dtype=np.float32)
    fit = np.ones(len(rn), dtype=bool) if train_mask is None else np.asarray(train_mask, dtype=bool)
    lo = rn[fit].min(0)
    span = np.maximum(rn[fit].max(0) - lo, 1e-6)
    scaled = ((rn - lo) / span * 2.0 - 1.0).astype(np.float32)      # train extent -> [-1, 1]
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, (scaled.shape[1], max(out_dim // 2, 1))).astype(np.float32)
    return scaled, base, tuple(bandwidths)


def _rff_features(scaled: np.ndarray, base: np.ndarray, sigma: float) -> torch.Tensor:
    proj = scaled @ (base * float(sigma))
    return torch.tensor(np.concatenate([np.sin(proj), np.cos(proj)], 1).astype(np.float32))


# ============================================================================================================
# 2. CONTRACT — what a probe must declare
# ============================================================================================================
#
# The probe -> harness result contract.
# 
# Every probe mode already computes a structured dict and then `return`s it. Nothing consumed that
# return value: `if __name__ == "__main__": main()` discarded it, and `trace.py` reconstructed the same
# facts by regex-scraping the prose printed alongside it -- guessing the mode from one of 22 hand-written
# `=== SPACETIME ...` header formats, the primary score from a per-capability regex table with two
# hand-patched special cases, and the fair baseline from a preference list over whatever gain labels
# happened to appear.
# 
# Every historical mis-record came from that interface rather than from a logic error:
# 
#   * `--pheno_disttarget peak_week` published 0.067 -> 0.683 because a DIFFERENT target printed an `acc`
#     the regex accepted.
#   * A run where the RFF control beat Earth4D recorded the CONTROL's accuracy as the Earth4D record,
#     because the scan took the max over every `acc` in the output.
#   * `calibration` maxed over four different uncertainty signals, so a run using a different signal could
#     "beat" a record set on max-softmax.
#   * Four mode paths print no `mode=` at all, so they all read `mode=None` and were therefore mutually
#     "like-for-like" -- defeating the very gate that exists to stop cross-target comparison.
# 
# Each was patched with another regex or another gate while the surface producing the ambiguity kept
# growing. This module removes the ambiguity instead: the probe DECLARES its identity and its numbers, the
# harness validates and consumes them, and anything missing is a hard error rather than a silent None.
# 
# Identity is what makes two runs comparable (see autoresearch/probes/spacetime/program/program.md):
# 
#     capability . mode . split . n_shards . protocol . code hash
# 
# `mode` and `primary` are REQUIRED. A mode that cannot say what it measured cannot set a record.

CONTRACT_VERSION = 1


class ContractError(ValueError):
    """Raised when a probe result cannot establish what it measured."""


@dataclass
class Primary:
    """The capability's absolute score, and the native metric it is expressed in.

    `name` exists so a record carries its own units. The board previously stored bare floats, which is
    how a within-tolerance accuracy, a micro-AP and an AUROC all came to live in one `score` column with
    nothing distinguishing them.
    """

    name: str
    value: float


@dataclass
class ProbeResult:
    capability: str
    mode: str                                  # e.g. "FORECAST(past->future)", "SDM-PRESENCE"
    primary: Primary
    protocol: str
    split: str = ""                            # e.g. "spatiotemporal-block", "temporal-future"
    n_shards: Optional[int] = None
    seed: Optional[int] = None
    steps: Optional[int] = None
    trained_encoder: bool = False              # frozen random hash vs end-to-end trained
    gains: Dict[str, float] = field(default_factory=dict)     # baseline label -> Earth4D minus baseline
    baselines: Dict[str, float] = field(default_factory=dict)  # baseline label -> its absolute score
    flags: str = ""
    # The probe's CONFIG block: the levers that used to be flags. Part of identity — see config_digest.
    config: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)
    # A diagnostic measures something that is NOT a scorecard capability, or measures it without
    # Earth4D in the comparison at all (several dynamics modes run on raw PE only). It is legitimate
    # research output, but it can never set a record, and saying so here is better than inventing a
    # capability for it so that it fits a slot on the board.
    diagnostic: bool = False
    diagnostic_reason: str = ""
    contract_version: int = CONTRACT_VERSION

    # -- identity ---------------------------------------------------------------------------------
    def config_digest(self) -> str:
        """Hash of the probe's CONFIG block — what the encoder was actually built and trained as.

        With the levers moved out of the CLI and into the file, two experiments run with the SAME
        command line. Identity therefore has to include what was built, or the gate would compare a
        rewired encoder against the control as though they were the same measurement.
        """
        return hashlib.sha256(
            json.dumps(self.config, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def identity(self) -> Dict[str, Any]:
        """The tuple that decides comparability. Two runs whose identities differ measure different
        things, however similar their scores look."""
        return {
            "capability": self.capability,
            "mode": self.mode,
            "split": self.split,
            "n_shards": self.n_shards,
            "protocol": self.protocol,
            "metric": self.primary.name,
            "config": self.config_digest(),
        }

    def identity_digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.identity(), sort_keys=True).encode()
        ).hexdigest()[:16]

    def comparable_to(self, other: Dict[str, Any]) -> bool:
        """Compare against a stored record's identity fields, tolerating older records that predate a
        field. A field the stored record never had cannot be asserted equal, so it is skipped rather
        than treated as a mismatch -- but `mode` and `metric` are never skipped, because those are the
        two that let cross-target comparisons through."""
        mine = self.identity()
        for key in ("mode", "metric"):
            if mine[key] != other.get(key):
                return False
        # `config` is skipped only when the stored record predates it — a legacy record cannot assert
        # what it was built as. When both sides have one, a different build is a different measurement.
        for key in ("capability", "split", "n_shards", "protocol", "config"):
            if key in other and other[key] is not None and mine[key] != other[key]:
                return False
        return True

    # -- fair gain --------------------------------------------------------------------------------
    def fair_gain(self, order) -> tuple:
        """The honest marginal: the gain against the STRONGEST fair baseline present.

        Returns (value, label) or (None, None). `order` decides which labels COUNT as fair; it is the
        harness's list, not the probe's, so a mode cannot nominate a flattering baseline for itself.

        Among those, the strongest baseline is the one with the SMALLEST gain -- every gain is
        `earth4d_score - baseline_score`, so min gain <=> max baseline. This used to return the first
        label matching the preference order instead, which is a different and much friendlier quantity:
        with "RFF" ahead of "raw" in the order, a mode that reported both got its gain-vs-RFF even when
        raw coordinates beat Earth4D outright. flowering_peak_month published `+0.0128 vs RFF` while
        raw scored 0.19933 against Earth4D's 0.19728 -- a real gain of -0.0021. The row read
        ENCODER-LIMITED when the honest read was INPUT-LIMITED, and the same silent flattery was
        available to every capability that declared more than one control.
        """
        fair = [(value, label) for label, value in self.gains.items()
                if any(preference.lower() in label.lower() for preference in order)
                and value is not None]
        if not fair:
            return (None, None)
        return min(fair, key=lambda pair: pair[0])

    # -- serialization ----------------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["identity_digest"] = self.identity_digest()
        return payload

    def write(self, path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True))
        return target

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ProbeResult":
        data = dict(payload)
        data.pop("identity_digest", None)
        version = data.pop("contract_version", None)
        if version != CONTRACT_VERSION:
            raise ContractError(
                f"probe result declares contract_version {version!r}, this harness speaks "
                f"{CONTRACT_VERSION}. Refusing to interpret a result written by a different contract."
            )
        primary = data.pop("primary", None)
        if not isinstance(primary, dict):
            raise ContractError("probe result has no primary block")
        return cls(primary=Primary(**primary), contract_version=CONTRACT_VERSION, **data)

    @classmethod
    def read(cls, path) -> "ProbeResult":
        text = Path(path).read_text()
        try:
            payload = json.loads(text)
        except ValueError as exc:
            raise ContractError(f"probe result at {path} is not valid JSON: {exc}") from exc
        return cls.from_dict(payload)

    # -- validation -------------------------------------------------------------------------------
    def validate(self) -> "ProbeResult":
        problems = []
        if self.diagnostic and not self.diagnostic_reason:
            problems.append("a diagnostic must say WHY it cannot set a record")
        if not self.capability and not self.diagnostic:
            problems.append("capability is empty")
        if not self.mode:
            # The old harness let this through as mode=None, which made unrelated runs mutually
            # comparable. A mode that will not name itself cannot set a record.
            problems.append("mode is empty -- a run that cannot name its target cannot be compared")
        if not self.primary or not self.primary.name:
            problems.append("primary.name is empty -- a score with no metric has no units")
        if self.primary is None or self.primary.value is None:
            problems.append("primary.value is missing")
        elif self.primary.value != self.primary.value:          # NaN
            problems.append("primary.value is NaN")
        if not self.protocol:
            problems.append("protocol is empty")
        for label, value in self.gains.items():
            if value != value:
                problems.append(f"gain {label!r} is NaN")
        if problems:
            raise ContractError(
                "probe result cannot establish what it measured: " + "; ".join(problems)
            )
        return self

    # -- rendering --------------------------------------------------------------------------------
    def records(self) -> bool:
        """Whether this result is eligible to be compared against a record at all."""
        return not self.diagnostic

    def render(self) -> str:
        """The single human-readable block, DERIVED from the result.

        Replaces 22 hand-written header formats. Because the harness no longer reads this text, a change
        here can never change what gets recorded -- which was the whole hazard of the old design.
        """
        lines = [
            f"=== SPACETIME | capability={self.capability or 'DIAGNOSTIC'} | mode={self.mode} "
            f"| split={self.split or 'n/a'} | n_shards={self.n_shards} | protocol={self.protocol} "
            f"| encoder={'trained' if self.trained_encoder else 'frozen-random'} ===",
            f"  {self.primary.name} = {self.primary.value:.6f}",
        ]
        if self.baselines:
            lines.append("  baselines: " + "  ".join(
                f"{label} {value:.4f}" for label, value in sorted(self.baselines.items())))
        if self.gains:
            lines.append("  gains:     " + "  ".join(
                f"{label} {value:+.4f}" for label, value in sorted(self.gains.items())))
        if self.diagnostic:
            lines.append(f"  DIAGNOSTIC (cannot set a record): {self.diagnostic_reason}")
        lines.append(f"  identity={self.identity_digest()}  seed={self.seed}  steps={self.steps}")
        return "\n".join(lines)


# ============================================================================================================
# 3. EMIT — the one path by which a number becomes recordable
# ============================================================================================================

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


def _set_result_sink(path, capability, protocol, args, config=None):
    """Arm the result contract for this run. Called once, right after parse_args.

    `config` is the probe's CONFIG block — the levers that used to be CLI flags. It must reach the
    identity: with the levers in the file, two experiments have IDENTICAL command lines, so without a
    digest of what was actually built the gate would treat a rewired encoder as the same measurement as
    the control and let one 'beat' the other.
    """
    _RESULT_SINK.update({
        "config": dict(config or {}),
        "path": path or "", "capability": capability or "", "protocol": protocol,
        # These moved out of argv and into the probe's CONFIG, so read them from there first. Falling
        # back to argv keeps any module that still declares them as flags working.
        "flags": " ".join(sys.argv[1:]), "seed": getattr(args, "seed", None),
        "steps": (config or {}).get("steps", getattr(args, "steps", None)),
        "n_shards": (config or {}).get("n_shards", getattr(args, "n_shards", None)),
        "trained_encoder": bool((config or {}).get("train_encoder",
                                                   getattr(args, "train_encoder", False))),
    })


def declare(capability, mode, metric, value, gains=None, baselines=None, split="",
            trained_encoder=None, diagnostic=False, diagnostic_reason="", **extras):
    """Declare WHAT this run measured, in the contract's terms.

    A mode calls this immediately before returning. Fields the run already knows (seed, steps, shard
    count, protocol, whether the encoder was trained) come from the armed sink rather than being
    re-derived, so they cannot drift from the actual invocation.

    `--capability` from the harness wins over the mode's natural default when both are present: the
    harness declared the objective, and any mismatch is the harness's to detect.

    `trained_encoder` defaults to the --train_encoder FLAG, but some modes
    train the encoder end-to-end unconditionally, so they pass it explicitly. Only the trained protocol
    can support a claim about learned hash state, so this field must describe what actually happened
    rather than what was requested.
    """
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
        config=dict(_RESULT_SINK.get("config") or {}),
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


# ============================================================================================================
# 4. REGISTRY — capability -> modes -> where to edit
# ============================================================================================================
#
# What an agent needs to know after it picks a capability to improve.
# 
# The loop's step ② is "pick one capability from scorecard.md, with intention". Everything after that
# used to require reading a 1,500-line `main()` to answer three questions:
# 
#     which probe modes measure this capability?
#     what flags select each one, and what do they REQUIRE to run?
#     where do I edit to change the mechanism vs the data channel?
# 
# Those answers were only discoverable by grep, and getting them wrong is cheap-looking and expensive:
# eight of nineteen modes silently require `--forecast` via a bare `assert` buried mid-function, and
# `--phenology` shadows `--pheno_env`/`--pheno_taxon`/`--pheno_densefield` entirely because its branch
# returns first. This module states all of it in one place.
# 
# A `records=False` mode is a legitimate diagnostic that can never set a record -- either because its
# target is not on the scorecard, or because it evaluates on raw coordinate features with Earth4D absent
# from the comparison.
# 
# The five dynamics/AR modes that used to appear here (BREADTH, PROPAGATOR-ARCH, FIRST-ARRIVAL,
# ABUNDANCE, AR-ROLLOUT, CONTINUOUS-LEAD) were DELETED: ~1,300 lines of instrument/ that could never
# move a scorecard capability, plus 17 flags for them. They are in git history if a real target ever
# needs them.
# 
# Usage:
#     python -m deepearth.autoresearch.probes.spacetime.harness --capability family_from_env --list-modes
#     python -m deepearth.autoresearch.probes.spacetime.harness.harness.py --list-modes

DATA = "DATA"
ARCH = "ARCHITECTURE"
BOTH = "DATA+ARCHITECTURE"


@dataclass(frozen=True)
class Mode:
    """One probe mode: what it measures, how to select it, and where to change it."""

    mode: str                      # the declared mode string; part of measurement identity
    flags: str                      # flags that select this mode
    capability: str = ""            # "" means diagnostic
    requires: Tuple[str, ...] = ()  # flags this mode asserts on, currently as bare asserts
    lever: str = BOTH               # which lever family this mode can actually move
    records: bool = True
    reason: str = ""                # why it cannot record, when records is False
    notes: str = ""

    @property
    def is_diagnostic(self) -> bool:
        return not self.records


MODES: Tuple[Mode, ...] = (
    # ---- family_from_env -------------------------------------------------------------------------
    Mode("ENV(<split>)", "--env [--env_channels {all,worldclim,alphaearth,wcsoil,...}] [--env_extra]",
         capability="family_from_env", lever=BOTH,
         notes="Primary is the FUSED Earth4D+ENV accuracy. Earth4D alone currently LOSES to RFF "
               "(0.0938 vs 0.1010), so the record is carried by the env channel -- label it."),

    # ---- family_from_spacetime -------------------------------------------------------------------
    Mode("FORECAST(past->future)", "--forecast [--target family]",
         capability="family_from_spacetime", lever=ARCH,
         notes="The default coordinate path. --forecast_spatial switches to future+newplace."),
    Mode("RECURRENCE(4D-LSTM rollout past->future)", "--recurrence [--rec_k K] [--rec_hidden H]",
         capability="family_from_spacetime", requires=("--forecast",), lever=ARCH,
         notes="science.md rule 2b. Currently negative vs RFF."),

    # ---- species_from_spacetime ------------------------------------------------------------------
    Mode("FORECAST(past->future)", "--forecast --target species",
         capability="species_from_spacetime", lever=ARCH,
         notes="Same tail as family; --target selects the capability and the metric name."),

    # ---- species_from_env ------------------------------------------------------------------------
    Mode("SDM-PRESENCE", "--sdm_presence [--cooccur_mech {env,space,both}]",
         capability="species_from_env", lever=DATA),
    Mode("SDM-HARD", "--sdm_hard [--sdm_channels ...] [--sdm_cell_deg D] [--sdm_holdout_mode ...]",
         capability="species_from_env", lever=DATA,
         notes="A DIFFERENT measurement from SDM-PRESENCE (0.3336 vs 0.6275 at 12 shards). "
               "Identity keeps them apart; never compare the two."),

    # ---- community_from_env ----------------------------------------------------------------------
    Mode("COOCCUR-ROUTING", "--cooccur [--cooccur_mech ...] [--cooccur_channels ...] [--cooccur_thresh N]",
         capability="community_from_env", lever=DATA),

    # ---- flowering_peak_month --------------------------------------------------------------------
    Mode("PHENOLOGY-FUTURE / -FUTURE-HELD / -HELD", "--phenology [--pheno_feats e4d,rff,raw] [--pheno_tol D]",
         capability="flowering_peak_month", requires=("--forecast",), lever=ARCH,
         notes="Record metric is Earth4D's best-head within-tolerance accuracy vs the generic PE's — "
               "NOT propagator_gain, which measures propagation vs static on raw features."),

)

# Where to make each kind of change. An agent that has picked a capability needs this more than it
# needs the file layout.
LEVER_SITES = {
    DATA: [
        "autoresearch/probes/spacetime/editable_files/probe.py: load_env / load_vision / load_env_species "
        "(which channel feeds the head)",
        "flags: --env_channels, --env_extra, --sdm_channels, --vision --vision_feats, --pheno_channel",
        "data prep: occurrence densification, channel fusion, per-entity aggregation",
    ],
    ARCH: [
        "autoresearch/probes/spacetime/editable_files/earth4d.py: __init__, forward, training objective (the encoder itself)",
        "autoresearch/probes/spacetime/editable_files/lib/recurrence.py: run_recurrence, propagators",
        "CONFIG levers: recurrence, forecast, fourier, "
        "--time_harmonics",
    ],
}


def for_capability(capability: str) -> Tuple[Mode, ...]:
    """Every mode that can set a record for this capability."""
    return tuple(m for m in MODES if m.capability == capability and m.records)


def diagnostics() -> Tuple[Mode, ...]:
    return tuple(m for m in MODES if not m.records)


def capabilities() -> Tuple[str, ...]:
    seen = []
    for m in MODES:
        if m.capability and m.capability not in seen:
            seen.append(m.capability)
    return tuple(seen)


def describe(capability: str) -> str:
    modes = for_capability(capability)
    if not modes:
        known = "\n  ".join(capabilities())
        return (f"no recording mode measures {capability!r}.\n"
                f"capabilities with modes:\n  {known}")
    out = [f"=== {capability} — {len(modes)} mode(s) can set this record ==="]
    for m in modes:
        out.append(f"\n  mode     {m.mode}")
        out.append(f"  select   {m.flags}")
        if m.requires:
            out.append(f"  REQUIRES {' '.join(m.requires)}")
        out.append(f"  lever    {m.lever}")
        if m.notes:
            out.append(f"  note     {m.notes}")
    levers = {m.lever for m in modes}
    out.append("\n--- where to change things ---")
    for lever in (DATA, ARCH):
        if any(lever in l for l in levers):
            out.append(f"  {lever}:")
            out.extend(f"    - {site}" for site in LEVER_SITES[lever])
    out.append("\nA run is comparable to the record only if capability, mode, split, n_shards and "
               "protocol all match. Changing mode is a new measurement, not a better score.")
    return "\n".join(out)


def _list_modes(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--capability", default="")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args(argv)
    if a.capability:
        print(describe(a.capability))
        return
    if a.all:
        for capability in capabilities():
            print(describe(capability))
            print()
        print("=== diagnostics — cannot set any record ===")
        for m in diagnostics():
            print(f"  {m.mode:52} {m.reason}")
        return
    print(f"capabilities: {', '.join(capabilities())}")
    print(f"{len(diagnostics())} diagnostic modes cannot record. Use --capability X or --all.")


# ============================================================================================================
# 5-7. GATE, PUBLISH, CLI — the loop
# ============================================================================================================

RECORDS = LOOP / "records" / "records.json"  # the machine record (fill scorecard by breaking these)
PROBE_MODULE = "deepearth.autoresearch.probes.spacetime.editable_files.probe"

# The encoder-probeable capabilities (scorecard.md Layer 2). The objective must be one of these; the
# probe MODE and the architecture are the agent's choice. This list and scorecard.md Layer 2/3 are one
# contract -- change both together.
CAPABILITIES = [
    "species_from_env", "species_from_spacetime", "family_from_env", "family_from_spacetime",
    "community_from_env", "flowering_peak_month",
]

# Declared-and-refused, with the reason (scorecard.md Layer 3). These used to sit in CAPABILITIES with
# no PRIMARY_RE entry, so a run would fall through to the generic r"\bEarth4D\s+([\d.]+)" pattern and
# record whatever number matched first -- a legal --metric that measured nothing in particular. An
# explicit refusal is the honest version: the capability is real on the full-model board, it is simply
# not reachable through the encoder probe.
EXCLUDED_CAPABILITIES = {
    "calibration": "cannot resolve an encoder effect: the paired Earth4D-vs-RFF difference has a "
                   "per-seed spread of +/-0.055 against a 0.0118 barrier, so a single run says "
                   "nothing. Its record was also never a real measurement -- the probe imported a "
                   "module as a package, fell silently into a synthetic surrogate that ignores the "
                   "feature argument, and ran an unseeded encoder (0.5375/0.5682/0.6062 on three "
                   "identical commands). Retired, and its probe deleted with it",
    "family_from_vision": "borrowed frozen DINO/BioCLIP, and the stored record has no mode or shard "
                          "identity; it is not an Earth4D probe record",
    "lfmc_from_env": "non-encoder head: the capability lives in a downstream head",
    "mycorrhiza_from_env": "non-encoder head: the capability lives in a downstream head",
    "pollinator_from_env": "non-encoder head: the capability lives in a downstream head",
    "flowering_auc": "measured on the fusion model's flowering head, not the encoder",
    "flowering_fidelity": "measured on the fusion model's flowering head, not the encoder",
    "infer_clay": "env->env reconstruction runs through the field decoder, not the encoder probe",
    "infer_soil": "env->env reconstruction runs through the field decoder, not the encoder probe",
    "infer_climate": "env->env reconstruction runs through the field decoder, not the encoder probe",
    "infer_hydro": "env->env reconstruction runs through the field decoder, not the encoder probe",
}

# PROTOCOL VERSION. Bump this whenever a change alters what a run MEASURES rather than how well it does:
#   v5-encoder-only : 2026-07-31. The probe was not measuring Earth4D. At the v4 champion the head
#                     received 20,663 features and the hash grid was 36 of them (0.17%); CMAC tile
#                     coding was 89.2%, the RFF 9.9%, and drop_spatiotemporal had deleted the
#                     tri-planes. Every v4 and earlier `fair_gain vs RFF` therefore scored a tile coder
#                     with a hash-shaped residue attached, and no record on this board has ever been a
#                     measurement of the encoder. v5 builds exactly what fusion.py:302 builds -- 36
#                     spatial + 108 tri-plane = 144 dims -- and TRAINS it (the frozen-random protocol
#                     existed only because the trained backward was nondeterministic; the kernel fix
#                     lands that at bit-identical and 4.5% faster). Every row re-baselines. Expect the
#                     numbers to FALL: 0.0955 was the tile coder's.
# a leak fix, a split change, a target/normalization change. Records carry the protocol they were set under,
# and a run under a different protocol RE-BASELINES the capability instead of "beating" it -- mode and shard
# count both match across such a change, so neither of those gates catches it.
#   v1-prefix     : everything up to 2026-07-29. Leaked in three ways (train mask admitted future-at-seen-place
#                   and past-at-held-place rows; time normalization fitted its span on test dates; env/vision
#                   standardization fitted mu/sd over test rows) and normalized time so the held-out future
#                   landed where the hash grid saturates.
#   v2-leakfix    : strict spatiotemporal split, train-only time normalization with horizon headroom,
#                   train-only feature standardization, deterministic seeding.
#   v3-fairbaseline: the RFF control is no longer degenerate. It was `rn @ N(0, 8)` on GLOBE-normalized
#                   coords, which across a regional corpus varies ~0.04 cycles end to end -- it scored
#                   0.008, BELOW raw coordinates at 0.0166. Every `vs RFF` gain measured against it was
#                   inflated by that handicap, and every EARNING share on the board rested on it. The
#                   control now gets train-extent normalization (the same courtesy the encoder gets) and
#                   its bandwidth selected over a sweep. This changes what fair-gain MEANS, so v2 numbers
#                   are not comparable to v3 ones and must re-baseline rather than be beaten.
PROTOCOL = "v5-encoder-only"
# Only explicitly identified, audited protocols may be migrated automatically.
# Absence of a protocol is not evidence that a hand-restored or pre-gate record
# belongs to the known v1 measurement regime.
# Every protocol this board has ever run under, oldest first. APPEND on each bump; never reorder.
PROTOCOL_HISTORY = ("v1-prefix", "v2-leakfix", "v3-fairbaseline", "v4-fixedcontrol", "v5-encoder-only")
assert PROTOCOL in PROTOCOL_HISTORY, f"PROTOCOL {PROTOCOL!r} missing from PROTOCOL_HISTORY"

# A record under any SUPERSEDED protocol may be re-baselined by the current one -- that is what a
# protocol bump means. This was a hand-maintained allowlist frozen at {"v1-prefix", "v2-leakfix"}, so
# after the v3 bump nothing could migrate a v3 record: the v5 baseline run scored 0.0367 against a
# stored 0.0955 that measured a different object entirely, and was WITHHELD for "protocol migration
# mismatch" instead of re-baselining. The board could never have moved off v3. Derived from the history
# now, so bumping PROTOCOL is a one-line change that cannot half-apply.
REBASELINE_PROTOCOLS = frozenset(PROTOCOL_HISTORY[:PROTOCOL_HISTORY.index(PROTOCOL)])

# Fair-baseline preference: Earth4D must beat a TRAINED generic PE, not just raw coords.
# THE fair control. One entry, because there is one encoder question: does Earth4D beat a
# matched-width generic coordinate encoder on the same data, split and head?
#
# This was a 7-entry preference list, and the list is what let three different quantities share one
# column of the board. "GAIN" matched a gain over the CLASS PRIOR, so species_from_env published
# +0.4000 -- beating the base rate -- next to species_from_spacetime's +0.0608 against a real encoder,
# and the two were sorted against each other. "best-coord" matched the ENV CHANNEL's advantage over
# coordinates, which is how family_from_env read as an encoder gain of +0.0411 while Earth4D alone
# (0.0938) was losing to RFF (0.1010).
#
# A row that cannot produce "vs RFF" now yields fair_gain=None and is declared diagnostic at the call
# site, rather than silently scoring against whatever else it happened to report.
FAIR_ORDER = ["vs RFF"]


def _run(device: str, log_path: str, result_path: str, capability: str, seed=None) -> int:
    """Invoke the probe.

    There is ONE probe and it takes four arguments — capability, seed, device, result-json — because
    every lever that used to be a flag lives in its CONFIG block, and an experiment is a diff of that
    block. `--probe`/`--probe-module` existed only for lib/calib_probe, which declared its own flags and
    served the one capability now excluded; both are gone with it.
    """
    probe_argv = ["--device", device, "--result-json", result_path, "--capability", capability]
    if seed is not None:
        probe_argv += ["--seed", str(seed)]
    cmd = [sys.executable, "-m", PROBE_MODULE] + probe_argv
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO.parent) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    print(f"[trace] $ {' '.join(cmd)}  (cwd={REPO})", flush=True)
    with open(log_path, "w") as lf:
        return subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                              env=env, cwd=str(REPO)).returncode










def _same_mode(mode, prev_mode):
    """Is this run measuring the same MODE as the stored record?

    Exact string equality plus one provenance rule. Records set BEFORE the result contract stored the
    mode FAMILY only ("ENV"); the contract later appended the split as a submode, so the identical
    measurement now declares "ENV(spatial-block)". family_from_env has been unrecordable ever since:
    every run -- including a no-edit control that reproduces the record to six decimals (0.142318
    against a stored 0.1423) -- was refused as "not like-for-like", and those refusals are still
    sitting in that capability's dead-end ledger (contract_cutover_famenv, dag_verify, leaf_verify...).

    The rule is narrow on purpose: it applies only when the STORED mode carries no submode, and only to
    a run whose mode is that same family with a submode appended. It cannot merge two submodes of one
    family, so FORECAST(past->future) and FORECAST(future+newplace) stay distinct targets -- and no
    pre-contract record stored a bare "FORECAST" anyway, because the forecast paths already carried
    their quadrant inside the mode string before the contract existed.
    """
    if mode == prev_mode:
        return True
    if not mode or not prev_mode or "(" in prev_mode:
        return False
    return mode.startswith(prev_mode + "(") and mode.endswith(")")


def _same_probe(probe, prev_probe):
    """Compare a migration command after shell-token normalization."""
    if not probe or not prev_probe:
        return False
    try:
        return shlex.split(probe) == shlex.split(prev_probe)
    except ValueError:
        return False


DEADEND_CAP = 40


def _next_seq(ledger):
    """Monotonic insertion counter for dead-ends."""
    seq = int(ledger.get("deadend_seq", 0)) + 1
    ledger["deadend_seq"] = seq
    return seq


def _evict_oldest_deadends(ledger, cap=DEADEND_CAP):
    """Bound the dead-end ledger by AGE, never by name.

    The eviction used to be `dict(list(deadends.items())[-cap:])`, i.e. "keep the last cap in dict
    order". That is only recency if the dict preserves insertion order -- and it does not survive the
    round trip, because the board is written with `json.dumps(..., sort_keys=True)`. After a reload the
    order is ALPHABETICAL, so the trim silently kept the alphabetically-last entries and deleted
    everything earlier. A deliberately written provenance note tagged
    `exp58_..._INVALIDATED` was destroyed that way, along with 73 other dead-ends: the swarm's record
    of what not to retry, evicted by first letter.

    Entries carry an explicit `seq`; the lowest go first. Pre-existing entries without one are treated
    as oldest, which is true of them.
    """
    dead = ledger.get("deadends") or {}
    if len(dead) <= cap:
        return
    ranked = sorted(dead.items(), key=lambda kv: (kv[1] or {}).get("seq", 0))
    ledger["deadends"] = dict(ranked[-cap:])


# ---------------------------------------------------------------------------------------------------
# The noise barrier — what a record must actually beat
# ---------------------------------------------------------------------------------------------------
# Calibrated against noise observed in THIS probe, not invented:
#
#   * a second agent walked family_from_spacetime 0.1769 -> 0.19143 in seven accepted single-seed steps
#     of +0.0007 / +0.0008 / +0.0112 / +0.0005 / +0.0002 / +0.0006 / +0.0006;
#   * a verification run here took flowering_peak_month 0.0521 -> 0.052131, a delta of +0.000031;
MIN_REL_IMPROVEMENT = 0.02      # 2% of the standing record
MIN_ABS_IMPROVEMENT = 0.002     # ...and never less than this in absolute terms
SEED_SIGMA_MULTIPLE = 2.0       # with >=3 seeds, must also clear 2 sigma of the seed spread
MIN_CONFIRMATION_SEEDS = 2      # operator policy: one seed screens, two matched seeds confirm


def noise_barrier(prev, seed_std=None, n_seeds=1):
    """How much a new score must exceed the standing record by to count.

    Two regimes. With enough seeds the spread is measurable, so the barrier is the larger of the fixed
    floor and 2 sigma -- the run has to be outside its own noise. With a single seed there IS no spread
    to measure, so only the fixed floor applies and the result stays provisional: a single seed can
    never be confirmatory under the evidence standard, whatever it scores.
    """
    if prev is None:
        return 0.0
    floor = max(abs(prev) * MIN_REL_IMPROVEMENT, MIN_ABS_IMPROVEMENT)
    if seed_std is not None and n_seeds >= 3:
        return max(floor, SEED_SIGMA_MULTIPLE * float(seed_std))
    return floor


def _record_gate(
    key_val,
    prev,
    prev_proto,
    mode,
    prev_mode,
    shards,
    prev_shards,
    probe=None,
    prev_probe=None,
    seed_std=None,
    n_seeds=1,
):
    """Return the like-for-like record decision and its component checks.

    `beats` now means "beats the standing record BY MORE THAN THE NOISE", not "is a larger float".
    The old meaning is why seven consecutive +0.0006 steps could each be accepted as a new best.
    """
    mode_ok = (prev is None) or _same_mode(mode, prev_mode)
    shards_ok = (prev is None) or (shards == prev_shards)
    barrier = noise_barrier(prev, seed_std, n_seeds)
    beats = key_val is not None and (prev is None or key_val > prev + barrier)
    rebaseline = (
        prev is not None
        and prev_proto in REBASELINE_PROTOCOLS
        and prev_proto != PROTOCOL
        and mode_ok
        and shards_ok
        # The probe STRING used to carry a run's identity, so a migration had to match it exactly.
        # Under the four-argument CLI there is no probe string — every lever lives in CONFIG and
        # config_digest carries the identity instead — so `probe` is always '' and this condition could
        # never be satisfied. The effect was total: NO v3 run could migrate a v2 record, so a 3.8x
        # improvement on flowering_peak_month was withheld for "protocol migration mismatch" while
        # species_from_spacetime only got past it because I hand-restored that record.
        #
        # When the new run carries no probe string, fall back to the identity that does exist: same mode
        # and same shard count, already required above.
        and (_same_probe(probe, prev_probe) or not (probe or "").strip())
    )
    current_comparison = prev is None or prev_proto == PROTOCOL
    is_record = (
        beats and mode_ok and shards_ok and current_comparison
    ) or (rebaseline and key_val is not None)
    return is_record, rebaseline, beats, mode_ok, shards_ok


def _read_records(path=RECORDS):
    """Read one exact board snapshot for optimistic concurrency control."""
    raw = path.read_bytes() if path.exists() else b""
    return raw, json.loads(raw or b"{}")


def _commit_records_if_unchanged(expected_raw, records, path=RECORDS):
    """Atomically replace a board only if it has not changed since preflight."""
    lock_path = path.with_name("records.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current_raw = path.read_bytes() if path.exists() else b""
        if current_raw != expected_raw:
            return False
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("w") as stream:
                stream.write(json.dumps(records, indent=2, sort_keys=True))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
    return True




def retire_record(capability, reason, path=RECORDS):
    """Withdraw a record that was never a valid measurement, keeping the full audit trail.

    A record can be invalid rather than merely beaten: the probe silently measured a synthetic
    surrogate, the encoder was unseeded, the split leaked. Such a number is not a high bar to clear,
    it is a false statement about the encoder -- and because the gate compares against it, it also
    blocks every subsequent honest run on that row.

    Deleting it by hand is what the doctrine forbids, and rightly: the board would lose the fact that
    a wrong number was ever believed. So retirement is a harness operation that MOVES the record into
    the ledger under `retired` with its reason and the sequence number of the retirement, clears the
    live score so the next run establishes a fresh baseline, and leaves every dead-end untouched.
    Nothing is destroyed; the row simply stops asserting something false.
    """
    raw, records = _read_records(path)
    record = records.get(capability)
    if record is None:
        raise KeyError(f"{capability!r} has no record to retire")
    ledger = record.setdefault("ledger", {})
    retired = ledger.setdefault("retired", [])
    retired.append({
        "seq": _next_seq(ledger),
        "score": record.get("score"),
        "gain": record.get("gain"),
        "protocol": record.get("protocol"),
        "mode": record.get("mode"),
        "tag": (ledger.get("records") or [{}])[-1].get("tag"),
        "why": reason,
    })
    # Clear only what asserts a result. Run counts, dead-ends and the retired history all survive,
    # so the row's cost and its mistakes stay visible.
    for field in ("score", "gain", "fair_baseline", "fair_st_gain", "read", "protocol", "mode"):
        record[field] = None
    if not _commit_records_if_unchanged(raw, records, path):
        raise RuntimeError("records.json changed during retirement — rerun")
    return retired[-1]


ENCODER_SHARE_FLOOR = 0.25      # below this, the encoder is adding little over a generic PE


def _bottleneck(fair, primary) -> str:
    """Diagnose which lever family the measurement points at (program.md, section 3).

    This string is written into records.json AND published to Ensue as the swarm's reason-to-move, so
    it decides what every agent reaches for next. It has to mean the same thing for every capability.

    It used to read `fair_gain > 0 and primary < 0.20 -> ENCODER-LIMITED`. That 0.20 was an absolute
    constant applied regardless of how hard the target is. species_from_spacetime has ~2,009 classes
    (chance ~0.0005, so 0.0512 is ~100x chance); family_from_spacetime has 166 (chance ~0.006, so
    0.1769 is ~30x chance). Both tripped `< 0.20` and both were told ARCHITECTURE, though they are not
    remotely in the same position relative to their own baselines. Acting on that sent four consecutive
    mechanism changes at species_from_spacetime -- --recurrence -0.0180, --gnn -0.0261, --train_encoder
    +0.0037, a tri-plane conjunction edit -0.0001 -- when the encoder was already contributing 84% of
    the score and the mechanism was not the weak part.

    The scale-free quantity is the encoder's SHARE of what was achieved:

        share = fair_gain / score      how much of the score is Earth4D over the strongest fair baseline

    A fraction is comparable across targets of any difficulty, needs nothing the contract does not
    already carry, and answers the question the lever choice actually turns on: is the encoder the part
    doing the work, or is it barely beating a generic positional encoding?

    Note what this does NOT claim. A high share does not mean the capability is finished, and a low
    absolute score is not evidence of a ceiling -- where the ceiling is, is the thing being discovered.
    """
    if fair is None:
        return "NO-FAIR-BASELINE (probe reported no vs-generic-PE gain — check output)"
    if fair <= 0:
        return ("INPUT-LIMITED: Earth4D does not beat a generic trained PE, so the coordinate/current "
                "channel lacks the signal → DATA lever, change the channel")
    if primary is None or primary <= 0:
        return f"EARNING: positive fair-gain {fair:+.4f} but no absolute score to weigh it against"
    share = fair / primary
    if share < ENCODER_SHARE_FLOOR:
        return (f"ENCODER-LIMITED: Earth4D contributes only {share:.0%} of the score over a generic PE "
                f"→ ARCHITECTURE lever, change the mechanism")
    return (f"EARNING: Earth4D contributes {share:.0%} of the score over a generic PE → the mechanism "
            f"is carrying real signal, push it further")



# ---------------------------------------------------------------------------------------------------
# scorecard.txt — the git-visible view of the campaign
# ---------------------------------------------------------------------------------------------------
SCORECARD_TXT = LOOP / "program" / "scorecard.txt"


def _read_of(record: dict, gain, score) -> str:
    """The diagnosis, flagged when the stored gain is not an encoder-vs-PE comparison.

    A gain labelled e.g. "ENV vs best-coord-PE" measures the CHANNEL's advantage over coordinates, not
    Earth4D's over a generic PE, so reading it as ENCODER-LIMITED is wrong. family_from_env is stored that
    way: the board says +0.0411 while a live run reports -0.0072 vs RFF and the honest read is
    INPUT-LIMITED. Until such a record is re-measured, say so rather than print a confident wrong word.
    """
    if gain is None:
        return "NO-FAIR-BASELINE"
    baseline = (record.get("fair_baseline") or "").lower()
    if baseline and not any(k in baseline for k in ("rff", "mlp", "ctrl", "gain", "raw", "pe")):
        return "CHECK-BASELINE"
    if "vs best-coord-pe" in baseline and "earth4d" not in baseline:
        return "STALE-GAIN(channel)"
    return _bottleneck(gain, score).split(":")[0]


def write_scorecard(recs: dict, path: Path = SCORECARD_TXT) -> Path:
    """Render the board as plain text for fast skimming, and for git.

    `records.json` is gitignored and lives on the box, so nobody reading the repository can see where
    the campaign actually stands. This file is that view: every metric, its current best, the fair gain
    against the strongest fair baseline, and the diagnosis. It is GENERATED after every run, so it
    cannot drift from the board -- do not hand-edit it. `scorecard.md` next to it explains what each
    row means and how a record is earned; this one is only the numbers.
    """
    # ONE SCREEN. The previous layout was 10 columns and ~160 chars, so it wrapped in any normal
    # terminal and the board -- the thing an agent reads before every pick -- was unreadable. Anything
    # that is not a number you act on moved to scorecard.md or the ledger.
    rows = []
    for cap in CAPABILITIES:
        r = recs.get(cap) or {}
        score, gain = r.get("score"), r.get("fair_st_gain")
        seeds = r.get("n_seeds")
        rows.append((
            cap,
            f"{score:.4f}" if isinstance(score, (int, float)) else "—",
            f"{gain:+.4f}" if isinstance(gain, (int, float)) else "—",
            _read_of(r, gain, score),
            str((r.get("ledger") or {}).get("runs", "—")),
            (f"{seeds}s" if seeds else "1s?"),
            (r.get("protocol") or "—"),
        ))
    head = ("capability", "record", "fair-gain", "read", "runs", "seeds", "protocol")
    w = [max(len(str(r[i])) for r in rows + [head]) for i in range(len(head))]
    lines = [
        "EARTH4D SPACETIME PROBE — SCORECARD",
        f"protocol {PROTOCOL}   ·   generated by harness.py   ·   DO NOT HAND-EDIT",
        "",
        "  ".join(h.ljust(w[i]) for i, h in enumerate(head)).rstrip(),
        "  ".join("-" * w[i] for i in range(len(head))),
    ]
    for r in sorted(rows, key=lambda x: (x[1] == "—", -float(x[1]) if x[1] != "—" else 0)):
        lines.append("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)).rstrip())
    earning = sum(1 for r in rows if r[2] != "—" and float(r[2]) > 0)
    probed = sum(1 for r in rows if r[1] != "—")
    lines += [
        "  ".join("-" * w[i] for i in range(len(head))),
        f"probed {probed}/{len(CAPABILITIES)}   ·   earning (fair-gain > 0): {earning}",
        "",
        "read:  INPUT-LIMITED    loses to a matched-width RFF   -> DATA lever",
        "       ENCODER-LIMITED  wins but contributes <25%      -> ARCHITECTURE lever",
        "       EARNING          contributes >=25%              -> push the mechanism",
        "a record under a superseded protocol is VOID -- it measured something else. See scorecard.md.",
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def _print_net_scorecard(recs: dict, current: str) -> None:
    """The NET scorecard: every capability's current best encoder-probe record. Printed after every run."""
    print("\n" + "#" * 76)
    print("# NET SCORECARD  —  Earth4D encoder-probe records so far")
    print("#" * 76)
    print(f"{'capability':<26}{'record':>9}{'fair_gain':>11}  best-lever")
    earning = 0
    for cap in CAPABILITIES:
        r = recs.get(cap)
        mark = "  <— this run" if cap == current else ""
        if not r:
            print(f"{cap:<26}{'—':>9}{'—':>11}  —{mark}")
            continue
        fg = r.get("fair_st_gain")
        if fg is not None and fg > 0:
            earning += 1
        sc = r.get("score")
        print(f"{cap:<26}{(f'{sc:.3f}' if sc is not None else '—'):>9}"
              f"{(f'{fg:+.3f}' if fg is not None else '—'):>11}  {r.get('tag', '')}{mark}")
    probed = sum(1 for c in CAPABILITIES if recs.get(c))
    print("-" * 76)
    print(f"probed {probed}/{len(CAPABILITIES)}   |   earning (fair_gain > 0): {earning}")
    print("#" * 76, flush=True)


ENSUE_ENV_FILE = AUTORESEARCH / ".env"   # autoresearch/.env, gitignored


def _read_env_file(path: Path, key: str) -> str:
    """Read one KEY=value from a dotenv-style file. Never logs the value."""
    if not path.exists():
        return ""
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() == key:
            return value.strip().strip('"').strip("'")
    return ""


def _ensue_token() -> str:
    """Resolve the Ensue token: environment first, then `autoresearch/.env`.

    The credential belongs to the autoresearch tree (gitignored there), not to a machine-specific
    absolute path. `/workspace/.env` is still honoured last so an existing box keeps working, but it is
    a fallback, not the home.
    """
    token = os.environ.get("ENSUE_API_TOKEN")
    if token:
        return token.strip()
    return (_read_env_file(ENSUE_ENV_FILE, "ENSUE_API_TOKEN")
            or _read_env_file(Path("/workspace/.env"), "ENSUE_API_TOKEN"))


def _code_provenance() -> dict:
    """Commit SHA + dirty flag for the tree this run measured.

    The evidence standard says a record from an unpushed commit is discovery-only, because nobody else
    can reproduce it. That rule was unenforceable: nothing recorded WHICH commit produced a number. It
    is not hypothetical -- a foreign agent's record on this board claims a `trained_rff` baseline that
    exists in no reachable tree, and a run of this loop was contaminated for an hour by an uncommitted
    edit to earth4d.py that nothing in the record would have revealed.
    """
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=str(REPO), capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return ""
    # records/ is EXCLUDED: the harness writes the board and the scorecard on every run, so counting
    # them would make every result permanently "dirty" and the flag would mean nothing. Dirty here means
    # the CODE that produced the number is uncommitted.
    changed = [l for l in git("status", "--porcelain").splitlines()
               if l.strip() and "/records/" not in l and "/program/scorecard.txt" not in l]
    return {"commit": git("rev-parse", "HEAD")[:12],
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(changed)}


def post_ensue(trace: dict) -> None:
    tok = _ensue_token()
    if not tok:
        # A silent skip here means the swarm never learns this run happened, and the next agent pays to
        # rediscover the same dead-end. --ensue was explicitly requested, so a missing token is an error.
        sys.exit("[trace] --ensue was requested but no ENSUE_API_TOKEN is available (env or "
                 "/workspace/.env). The record was written locally; the swarm was NOT updated. "
                 "Export the token and re-publish rather than leaving the board stale.")
    o = trace["objective"]
    led = trace.get("ledger", {}) or {}
    hist = led.get("records", [])
    best = hist[-1] if hist else {"tag": trace["tag"], "score": o.get("record_value"), "gain": o.get("fair_st_gain")}
    dead = led.get("deadends", {})
    rec_str = " -> ".join(f"{r['tag']}:{r['score']}" for r in hist[-8:]) or "(none)"
    dead_str = "; ".join(f"{t}={d['score']}({(d.get('why') or '')[:34]})" for t, d in list(dead.items())[-12:]) or "(none)"
    # ONE upserted key per capability (LOOP-<program>-<capability> taxonomy): running best + record history +
    # this run's outcome + deduped dead-ends WITH their bottleneck reason. Win or dead-end, every run captured.
    # Evidence and provenance travel WITH the number. Without them the swarm cannot tell a two-seed
    # result from a single-seed one, nor reproduce the tree that produced it -- and both of those
    # ambiguities have already put a noise-mined record on this board.
    ev = trace.get("evidence", {})
    prov = trace.get("code", {})
    ev_str = (f"{ev.get('n_seeds', '?')} seeds"
              + (f" sd {ev['seed_std']:.6f}" if ev.get("seed_std") is not None else " (sd needs >=3)")
              + (" PROVISIONAL" if ev.get("provisional", True) else " CONFIRMED-ELIGIBLE"))
    prov_str = (f"{prov.get('branch', '?')}@{prov.get('commit', '?')}"
                + (" DIRTY-TREE" if prov.get("dirty") else ""))
    val = (f"LOOP-earth4d {trace['metric']}: BEST {best.get('score')} (gain {best.get('gain')}, {o.get('fair_baseline')}) "
           f"via '{best.get('tag')}'. runs={led.get('runs')}. record-history: {rec_str}. "
           f"THIS RUN '{trace['tag']}': primary={o['primary']} gain={o['fair_st_gain']} "
           f"decision={o.get('decision', 'legacy')} evidence={ev_str} code={prov_str} "
           f"bottleneck={trace['bottleneck']}. dead-ends-tried: {dead_str}.")
    # create_memory does NOT overwrite an existing key -- the API has a separate update_memory. This
    # loop upserts ONE key per capability, so every run after the first was a silent no-op: the server
    # returned 200, the harness printed "Ensue logged", and the stored value stayed at whatever the
    # first run wrote. Checked directly: the key still read "BEST 0.0474 ... runs=1" from 2026-07-29
    # while the local board had moved to 0.0787 over ~30 runs. The swarm was reading a stale board and
    # would have re-bought every dead-end published since.
    #
    # Update first, create only if the key does not exist yet.
    def _call(tool: str) -> tuple:
        # The two tools take DIFFERENT argument shapes: create_memory batches under "items", while
        # update_memory takes one flat object. Sending the batch form to update_memory fails, which is
        # why the earlier fallback still ended at create_memory's duplicate-key error.
        args = ({"items": [{"key_name": key, "value": val, "description": desc}]} if tool == "create_memory"
                else {"key_name": key, "value": val, "description": desc})
        payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                   "params": {"name": tool, "arguments": args}}
        req = urllib.request.Request("https://api.ensue-network.ai/", data=json.dumps(payload).encode(),
                                     headers={"Authorization": f"Bearer {tok}",
                                              "Content-Type": "application/json",
                                              "Accept": "application/json, text/event-stream"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read().decode()
            failed = '"failed":1' in body or '"error"' in body
            return (not failed), body
        except Exception as exc:
            return False, str(exc)

    key = f"LOOP-earth4d-{trace['metric']}"
    desc = (f"Earth4D encoder-probe loop {trace['metric']}: best {best.get('score')} "
            f"gain {best.get('gain')} over {led.get('runs')} runs")
    ok, body = _call("update_memory")
    if not ok:
        ok, body = _call("create_memory")
    if ok:
        print(f"[trace] Ensue upserted {key}", flush=True)
    else:
        # A silent failure here is how the swarm went stale for a whole session.
        sys.exit(f"[trace] ENSUE WRITE FAILED for {key}: {body[:300]}\n"
                 f"        The local board is correct but the swarm was NOT updated. Fix and re-publish.")



# ============================================================================================================
# DETERMINISM CHECK  (harness.py --determinism)
# ============================================================================================================
#
# The trained protocol was unusable for years because the backward is a storm of colliding float atomics:
# five seed-0 runs gave 0.1873/0.1925/0.1867/0.1872/0.1952 (sd 0.0038), as large as the whole across-seed
# spread. An irreproducible number cannot set a record, which is why every record before v5 was frozen-
# encoder. utils.cuh::atomicAddFixed replaces those atomics with order-independent int64 accumulation.
#
# This is the gate for that fix, and the regression test for it: exit 1 while the backward diverges,
# exit 0 once it is bit-identical. It also ATTRIBUTES the divergence per level -- concentrated in coarse
# levels (many points per cell, many colliding atomics) confirms atomicAdd; a flat profile would mean the
# cause is elsewhere and the kernel is the wrong target. Measured on 2x RTX PRO 6000: coarse levels ran
# 5-13x above a flat ~2.5e-7 fine-level plateau, and EARTH4D_DETERMINISTIC=1 made all four encoders
# bit-identical at 4.5% FASTER.



def _det_one_backward(enc: Earth4D, coords: torch.Tensor, seed: int):
    """One forward+backward at a fixed seed. Returns (output, per-encoder embedding grads)."""
    torch.manual_seed(seed)
    for p in enc.parameters():
        if p.grad is not None:
            p.grad = None
    out = enc(coords)
    # A fixed, seed-independent scalar objective: no randomness anywhere except the kernel itself.
    loss = (out * torch.linspace(1.0, 2.0, out.shape[1], device=out.device)).sum()
    loss.backward()
    grads = {name: p.grad.detach().clone()
             for name, p in enc.named_parameters()
             if p.grad is not None and name.endswith("embeddings")}
    return out.detach().clone(), grads


def _det_report(name: str, a: torch.Tensor, b: torch.Tensor) -> bool:
    same = torch.equal(a, b)
    if same:
        print(f"  {name:<34} BIT-IDENTICAL")
        return True
    d = (a - b).abs()
    scale = a.abs().max().clamp_min(1e-30)
    n_diff = int((d > 0).sum())
    print(f"  {name:<34} DIVERGES   max|d|={d.max():.3e}  rel={d.max() / scale:.3e}  "
          f"elems={n_diff}/{a.numel()}")
    return False


def _determinism_check(device: str = "cuda:0", n: int = 200_000, repeats: int = 3,
                       levels: int = 18, log2_hashmap: int = 20) -> int:
    import torch
    from deepearth.autoresearch.probes.spacetime.editable_files.earth4d import Earth4D
    if not torch.cuda.is_available() and device.startswith("cuda"):
        print("[determinism] no CUDA device — this must run on the box.")
        return 2
    dev = torch.device(device)
    torch.manual_seed(0)
    coords = torch.stack([
        torch.rand(n, device=dev) * 9.5 + 32.5, torch.rand(n, device=dev) * 10.0 - 124.0,
        torch.rand(n, device=dev) * 2000.0, torch.rand(n, device=dev)], dim=1)
    enc = Earth4D(verbose=False, spatial_levels=levels, temporal_levels=levels,
                  spatial_log2_hashmap_size=log2_hashmap,
                  temporal_log2_hashmap_size=log2_hashmap).to(dev)
    print(f"[determinism] {n:,} coords · {levels} levels · 2^{log2_hashmap} hashmap · {repeats} repeats\n")
    outs, grads = [], []
    for _ in range(repeats):
        o, g = _det_one_backward(enc, coords, seed=0)
        outs.append(o); grads.append(g)
    print("1. FORWARD (gather — expected bit-identical)")
    for r in range(1, repeats):
        _det_report(f"output[0 vs {r}]", outs[0], outs[r])
    print("\n2. BACKWARD (scatter — atomicAdd lives here)")
    ok = True
    for name in grads[0]:
        for r in range(1, repeats):
            ok &= _det_report(f"{name}[0 vs {r}]", grads[0][name], grads[r][name])
    if ok:
        print("\nBackward is deterministic on this build.")
        return 0
    print("\n3. WHERE — per-level divergence (coarse-concentrated => atomicAdd confirmed)")
    named = {nm: m for nm, m in enc.named_modules() if hasattr(m, "offsets")}
    for name, g0 in grads[0].items():
        mod = named.get(name.rsplit(".", 1)[0])
        if mod is None:
            continue
        offs = mod.offsets.tolist()
        print(f"\n   {name.rsplit('.', 1)[0]}")
        for lvl in range(len(offs) - 1):
            lo, hi = offs[lvl], offs[lvl + 1]
            b0, b1 = g0[lo:hi], grads[1][name][lo:hi]
            rel = (b0 - b1).abs().max().item() / max(b0.abs().max().item(), 1e-30)
            print(f"     level {lvl:>2}  entries {hi-lo:>9,}  rel={rel:.3e}  " + "#" * min(int(rel * 2e4), 40))
    print("\nEXIT 1: backward is nondeterministic. Rerun with EARTH4D_DETERMINISTIC=1.")
    return 1


# ============================================================================================================
# PRIOR LEARNINGS  (harness.py --insights)
# ============================================================================================================
#
# 2,531 recorded runs and 123 dead-ends with their reasons sit in this board, and until now NOTHING read
# them back. Step (1) of the loop says "pull Ensue keys + records.json, skip logged dead-ends" and there
# was no command that did it, so every agent re-derived what the last one had already paid for.
#
# v5 voids the SCORES -- they measured a tile coder at 0.17% encoder content, against a fair-gain column
# that mixed three different quantities. It does NOT void the HYPOTHESES. "--recurrence -0.0180",
# "--gnn -0.0261", "extent_fit -0.0199", "tri-plane conjunction ~0", "17 capacity-sweep rows bought
# nothing" are all still information about which levers are worth a run, and re-buying them under the
# new regime is pure waste.
#
# So this prints the attempts and their reasons, with every pre-v5 number explicitly marked void so
# nobody compares against one. Ensue is queried too when a token is present, because the swarm's board
# carries attempts this checkout never ran.
def _ensue_fetch(key: str):
    """Read one Ensue key. Returns its value string, or None if unavailable."""
    tok = _ensue_token()
    if not tok:
        return None
    for tool, args in (("get_memory", {"key_name": key}),
                       ("search_memories", {"query": key})):
        payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                   "params": {"name": tool, "arguments": args}}
        req = urllib.request.Request("https://api.ensue-network.ai/", data=json.dumps(payload).encode(),
                                     headers={"Authorization": f"Bearer {tok}",
                                              "Content-Type": "application/json",
                                              "Accept": "application/json, text/event-stream"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                body = r.read().decode()
            if '"error"' not in body and key in body:
                return body
        except Exception:
            continue
    return None


def _insights(capability: str = "", ensue: bool = True) -> int:
    _raw, recs = _read_records()
    caps = [capability] if capability else sorted(recs)
    if capability and capability not in recs:
        print(f"[insights] no ledger for {capability!r}. known: {', '.join(sorted(recs))}")
        return 1

    print("=" * 108)
    print("PRIOR LEARNINGS — what has already been tried on this board")
    print("=" * 108)
    print("Every SCORE below predates protocol v5 and is VOID: it measured a feature vector that was")
    print("0.17% encoder, scored against a fair-gain column that mixed encoder / env-channel / class-prior")
    print("gains. Do not compare a v5 number to one. The HYPOTHESES and their reasons are still valid --")
    print("that is the point of reading this before you pick.\n")

    for cap in caps:
        v = recs.get(cap) or {}
        led = v.get("ledger") or {}
        dead = led.get("deadends", {})
        hist = led.get("records", [])
        print("-" * 108)
        print(f"{cap}   runs={led.get('runs', 0)}   dead-ends={len(dead)}   "
              f"pre-v5 record={v.get('score')} (VOID)   protocol={v.get('protocol')}")
        if hist:
            print("  record history (void): " + " -> ".join(f"{r.get('tag')}:{r.get('score')}" for r in hist[-6:]))
        if dead:
            print("  attempts and why they stopped:")
            for tag, x in sorted(dead.items(), key=lambda kv: -(kv[1].get("seq") or 0))[:24]:
                why = (x.get("why") or "").replace("\n", " ")
                print(f"    {tag[:44]:46} {str(x.get('score'))[:8]:9} {why[:52]}")
        if ensue:
            got = _ensue_fetch(f"LOOP-earth4d-{cap}")
            print(f"  ensue: {'fetched — swarm board may carry attempts this checkout never ran' if got else 'unavailable (no token, or key absent)'}")
    print("-" * 108)
    print("Before picking: skip anything above whose REASON still applies under v5. A lever that failed")
    print("because it was drowned in 18,432 tile-code dims is NOT settled -- it was never measured on the")
    print("encoder. A lever that failed on its own mechanics (extent_fit -0.0199, capacity sweeps) is.")
    return 0



# ============================================================================================================
# THE DATA LEVER  (harness.py --channels)
# ============================================================================================================
#
# program.md calls DATA co-equal with ARCHITECTURE, and until now an agent had no way to see it: the
# corpus is one flat bag shared with the biological loop and fusion, and nothing said which files THIS
# encoder can consume. Measured consequence -- five of six capabilities run on coordinates alone, two
# advertised levers (`env_extra`, `densify`) are not CONFIG keys at all, and two channels that exist on
# disk (topo, hydro) have no loader.
#
# probe.py declares CHANNELS; this derives their state from disk, so the table cannot claim a channel
# the corpus does not have.
def _channels(cap: str = "") -> int:
    import ast as _ast
    import re as _re
    src = (LOOP / "editable_files" / "probe.py").read_text()
    CH = _ast.literal_eval(_re.search(r"CHANNELS = (\{.*?\n\})", src, _re.S).group(1))
    RP = _ast.literal_eval(_re.search(r"REPAIRED = (\{.*?\n\})", src, _re.S).group(1))
    CFG = _ast.literal_eval(_re.search(r"CONFIG = (\{.*?\n\})", src, _re.S).group(1))
    CAP = _ast.literal_eval(_re.search(r"CAPABILITY_CONFIG = (\{.*?\n\})", src, _re.S).group(1))
    cache = Path(CFG["cache_dir"])
    if not cache.is_absolute():
        cache = REPO.parent / cache if (REPO.parent / cache).exists() else AUTORESEARCH.parent / cache

    # a channel is LIVE if any capability preset (or CONFIG, for a bare switch) actually selects it
    def live_in(name, switch):
        for c, v in CAP.items():
            if cap and c != cap:
                continue
            m = {**CFG, **v}
            val = m.get(switch)
            if val is True or (isinstance(val, str) and (name in val or val == "all")):
                if switch != "env_channels" or m.get("env"):
                    return c
                if switch == "env_channels" and m.get("env"):
                    return c
        return ""

    print("=" * 100)
    print(f"DATA LEVER — channels this encoder can consume{f'  (capability: {cap})' if cap else ''}")
    print("=" * 100)
    print(f"corpus: {cache}\n")
    print(f"  {'channel':<14} {'state':<10} {'dims':>5}  {'switch':<14} files")
    n_unwired = n_missing = 0
    for name, (files, dims, switch, what) in CH.items():
        on_disk = all((cache / f).exists() for f in files)
        wired = f'"{name}"' in src or f"'{name}'" in src
        if not on_disk:
            state = "MISSING"; n_missing += 1
        elif live_in(name, switch):
            state = "LIVE"
        elif not wired:
            state = "UNWIRED"; n_unwired += 1
        else:
            state = "AVAILABLE"
        print(f"  {name:<14} {state:<10} {dims:>5}  {switch:<14} {', '.join(files)}")
        print(f"  {'':14} {what}")
    print()
    print("  LIVE       a capability preset selects it today")
    print("  AVAILABLE  on disk and a loader reads it -- flip the switch")
    print("  UNWIRED    on disk and NO loader reads it. A real lever nobody can pull.")
    print("  MISSING    declared but absent on disk")

    print("\n" + "-" * 100)
    print("REPAIRED CORPUS FILES — a run against an un-activated one measures the wrong data")
    stale = 0
    for live, rebuilt in RP.items():
        lp, rp = cache / live, cache / rebuilt
        if not rp.exists():
            print(f"  {live:<30} no rebuilt version on disk")
            continue
        if not lp.exists():
            print(f"  {live:<30} *** LIVE FILE MISSING, repair exists at {rebuilt}"); stale += 1
        elif lp.stat().st_size != rp.stat().st_size:
            print(f"  {live:<30} *** NOT ACTIVATED  live={lp.stat().st_size:,}  repaired={rp.stat().st_size:,}")
            stale += 1
        else:
            print(f"  {live:<30} activated")
    print("-" * 100)
    print(f"{n_unwired} unwired · {n_missing} missing · {stale} un-activated repair(s)")
    if stale:
        print("Activating one is `cp derived/<x>_rebuilt.<ext> <x>.<ext>` -- a champion-pipeline change,")
        print("so it is the OPERATOR's call, not an agent's. Until then those rows measure the old corpus.")
    return 1 if (n_missing or stale) else 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Earth4D legacy probe ledger — exact audited protocol migrations only"
    )
    ap.add_argument("--metric", required=True, help="objective capability (one of the scorecard capabilities)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, default=1,
                    help="run the probe this many times with matched seeds; >=3 makes the seed spread "
                         "measurable, so the noise barrier becomes 2 sigma instead of a fixed floor")
    ap.add_argument("--ensue", action="store_true")
    ap.add_argument("--log", default=None)
    ap.add_argument("--retire", default="", metavar="REASON",
                    help="OPERATOR ACTION, not an experiment: withdraw this capability's record "
                         "because it was never a valid measurement, giving the reason. The old score "
                         "moves into the ledger under `retired` and the row reopens. Use when a record "
                         "is provably wrong (surrogate data, unseeded encoder, leaked split) — never "
                         "because a run failed to beat it.")
    a = ap.parse_args()

    if a.metric in EXCLUDED_CAPABILITIES:
        sys.exit("[trace] --metric %r is excluded: %s\n"
                 "        See autoresearch/probes/spacetime/program/scorecard.md Layer 3."
                 % (a.metric, EXCLUDED_CAPABILITIES[a.metric]))
    if a.metric not in CAPABILITIES:
        sys.exit("[trace] --metric %r is not an encoder-probeable capability. one of:\n  %s"
                 % (a.metric, "\n  ".join(CAPABILITIES)))
    if a.retire:
        entry = retire_record(a.metric, a.retire)
        print(f"[trace] RETIRED {a.metric}: score {entry['score']} withdrawn — {a.retire}")
        print(f"[trace] the row is reopened; the next run sets a fresh baseline.")
        write_scorecard(_read_records()[1])
        return

    modes = for_capability(a.metric)
    if not modes:
        sys.exit(f"[trace] no recording probe mode measures {a.metric!r}. "
                 f"See harness.py --list-modes.")
    print(f"[trace] {a.metric}: {len(modes)} mode(s) can set this record — "
          + ", ".join(m.mode for m in modes), flush=True)
    records_snapshot, preflight_records = _read_records()

    tag = a.tag or f"e4d_{a.metric}"
    log_path = a.log or str(LOOP / "records" / "traces" / f"{tag}.log")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[trace] OBJECTIVE={a.metric}  tag={tag}", flush=True)
    # One run per seed. Seeds are matched across arms by construction: the probe seeds numpy and torch
    # from --seed, so seed k is the same initialization for every configuration compared here.
    seed_results, seed_values = [], []
    for k in range(max(1, a.seeds)):
        seed_log = log_path if a.seeds == 1 else str(Path(log_path).with_suffix(f".seed{k}.log"))
        result_path = str(Path(seed_log).with_suffix(".result.json"))
        rc = _run(a.device, seed_log, result_path, a.metric, seed=k)
        text = Path(seed_log).read_text(errors="ignore")
        if rc != 0:
            print(text[-1800:])
            sys.exit(f"[trace] probe FAILED on seed {k} (rc={rc}); see {seed_log}")
        try:
            seed_results.append(ProbeResult.read(result_path))
        except (ContractError, OSError) as exc:
            sys.exit(f"[trace] seed {k} emitted no usable result contract: {exc}")
        seed_values.append(seed_results[-1].primary.value)
        if a.seeds > 1:
            print(f"[trace] seed {k}: {seed_results[-1].primary.name} = {seed_values[-1]:.6f}", flush=True)

    log_path = str(Path(log_path).with_suffix(".seed0.log")) if a.seeds > 1 else log_path
    result_path = str(Path(log_path).with_suffix(".result.json"))
    seed_std = (statistics.pstdev(seed_values) if len(seed_values) > 2 else None)
    seed_mean = sum(seed_values) / len(seed_values)
    if a.seeds > 1:
        print(f"[trace] {a.seeds} seeds: mean {seed_mean:.6f}"
              + (f"  sd {seed_std:.6f}" if seed_std is not None else "  (sd needs >=3 seeds)"), flush=True)
    text = Path(log_path).read_text(errors="ignore")

    # The probe DECLARES what it measured. Nothing here parses stdout: a mode that does not emit a
    # contract cannot set a record, which is the point -- the old parser always produced *something*.
    result = seed_results[0]
    if result.diagnostic:
        sys.exit(f"[trace] {result.mode} is a DIAGNOSTIC and cannot set a record: "
                 f"{result.diagnostic_reason}\n        log preserved at {log_path}")
    if result.capability != a.metric:
        sys.exit(f"[trace] probe measured {result.capability!r} but --metric declared {a.metric!r}; "
                 f"refusing to record a different question's answer")

    known = {m.mode for m in modes}
    if result.mode not in known and not any(result.mode.startswith(k.split("(")[0]) for k in known):
        print(f"[trace] *** UNREGISTERED MODE {result.mode!r} for {a.metric}. Registered: "
              f"{sorted(known)}. Recording it, but add it to MODES in this file so the next agent can "
              f"find it.", flush=True)

    primary = seed_mean          # the mean across seeds — never the max of reruns
    fair, fair_base = result.fair_gain(FAIR_ORDER)
    bottleneck = _bottleneck(fair, primary)
    mode = result.mode
    shards = result.n_shards
    header = result.render()
    gains = dict(result.gains)
    metrics = [f"{k} = {v}" for k, v in sorted(result.baselines.items())]

    # RECORD tracking + full run LEDGER (taxonomy: never lose a run's result; publish win OR dead-end w/ reason) --
    recs = preflight_records
    key_val = primary if primary is not None else fair
    cur = recs.get(a.metric, {})
    prev = cur.get("score")
    # RECORD GATE. A record used to fire on any parsed number that beat the stored one -- no check that the run
    # measured the SAME THING. That is how --pheno_disttarget peak_week (a different target, and a leaked one)
    # took flowering_peak_month 0.067 -> 0.683 and published it. A capability's record may only be beaten by a
    # run in the SAME probe mode; a different mode is a different target and gets flagged for review instead.
    prev_mode, prev_shards = cur.get("mode"), cur.get("n_shards")
    prev_proto = cur.get("protocol")
    # An UNSTAMPED record (pre-gate, or hand-restored) is treated as unknown-mode and does NOT auto-pass:
    # that is exactly how the leaked peak-week run slipped through on its second attempt.
    is_record, rebaseline, beats, mode_ok, shards_ok = _record_gate(
        key_val,
        prev,
        prev_proto,
        mode,
        prev_mode,
        shards,
        prev_shards,
        cur.get("probe"),
        seed_std=seed_std,
        n_seeds=len(seed_values),
    )
    # A DIRTY TREE CANNOT SET A RECORD. With the flags gone, the experiment IS the CONFIG/earth4d.py
    # diff, so a record measured on uncommitted code has a configuration nobody can ever recover -- the
    # number survives and the thing that produced it does not. All three coordinate records carry
    # dirty=True, and that is the mechanism behind the family_from_spacetime noise-walk: seven accepted
    # single-seed steps from a second tree, invalidated, then immediately re-set at one dirty seed.
    # The run still measures, still publishes to Ensue, still writes its trace. It just cannot take the
    # record.
    _dirty = _code_provenance().get("dirty")
    if _dirty and is_record:
        print("[trace] *** RECORD WITHHELD: DIRTY TREE. The run measured uncommitted changes, so its "
              "configuration is not recoverable by anyone else. Commit, then re-run to claim it.",
              flush=True)
        is_record = rebaseline = False
    _barrier = noise_barrier(prev, seed_std, len(seed_values))
    if prev is not None and key_val is not None and key_val > prev and not beats:
        print(f"[trace] *** WITHIN NOISE: {key_val:.6f} beats {prev} by {key_val - prev:+.6f}, under the "
              f"barrier of {_barrier:.6f}"
              + (f" (2 sd of {len(seed_values)} seeds)" if seed_std is not None and len(seed_values) >= 3
                 else " (fixed floor; seed spread needs >=3)") +
              ".\n[trace]     Not a record. Seven steps of this size are how a record was walked "
              "0.1769 -> 0.1914 overnight.", flush=True)
    migration_withheld = prev is not None and prev_proto != PROTOCOL and not rebaseline
    if rebaseline and key_val is not None:
        print(f"[trace] *** RE-BASELINE: record was set under protocol {prev_proto!r}, this run is {PROTOCOL!r}.\n"
              f"[trace]     {prev} and {key_val} measure different things, so this is not a comparison —\n"
              f"[trace]     the capability's baseline is being RESET to {key_val}. Prior record archived in the ledger.",
              flush=True)
    elif prev is not None and prev_proto != PROTOCOL and prev_proto not in REBASELINE_PROTOCOLS:
        print(f"[trace] *** PROTOCOL MIGRATION WITHHELD: prior protocol {prev_proto!r} is not an "
              f"explicitly audited migration source.\n"
              f"[trace]     The protected record remains unchanged; migrate it deliberately after provenance review.",
              flush=True)
    elif prev is not None and prev_proto in REBASELINE_PROTOCOLS and prev_proto != PROTOCOL and not rebaseline:
        print(f"[trace] *** PROTOCOL MIGRATION WITHHELD: the old probe command, mode, and shard count "
              f"must match exactly.\n"
              f"[trace]     old={cur.get('probe')!r} mode={prev_mode!r} shards={prev_shards!r}\n"
              f"[trace]     new mode={mode!r} shards={shards!r}",
              flush=True)
    if beats and not (mode_ok and shards_ok):
        why = ("mode %r != record mode %r" % (mode, prev_mode) if not mode_ok
               else "n_shards %r != record n_shards %r" % (shards, prev_shards))
        print(f"[trace] *** RECORD WITHHELD: {why}.\n"
              f"[trace]     {key_val} vs {prev} is not a like-for-like comparison. Match the record's protocol,\n"
              f"[trace]     or verify the new one is sound and set the record deliberately.", flush=True)
    ledger = cur.get("ledger", {"runs": 0, "records": [], "deadends": {}})
    ledger["runs"] = ledger.get("runs", 0) + 1
    if is_record:
        cur = {"score": key_val, "primary": primary, "fair_st_gain": fair,
               "code": _code_provenance(),
               "n_seeds": len(seed_values), "seed_values": [float(v) for v in seed_values],
               "seed_std": (float(seed_std) if seed_std is not None else None),
               "provisional": len(seed_values) < MIN_CONFIRMATION_SEEDS,
               "fair_baseline": fair_base, "tag": tag, "mode": mode, "n_shards": shards,
               "protocol": PROTOCOL}
        ledger["records"] = (ledger.get("records", []) + [{"tag": tag, "score": key_val, "gain": fair,
                                                           "protocol": PROTOCOL,
                                                           "rebaseline_from": prev if rebaseline else None}])[-20:]
    elif migration_withheld and key_val is not None:
        ledger.setdefault("deadends", {})[tag] = {
            "score": key_val,
            "gain": fair,
            "why": (
                f"PROTOCOL MIGRATION WITHHELD (old protocol={prev_proto!r}, mode={prev_mode!r}, "
                f"n_shards={prev_shards!r}, probe={cur.get('probe')!r}; new mode={mode!r}, "
                f"n_shards={shards!r})"
            ),
        }
    elif beats and not (mode_ok and shards_ok):
        ledger.setdefault("deadends", {})[tag] = {
            "score": key_val, "gain": fair,
            "why": (f"RECORD WITHHELD (mode {mode!r} vs {prev_mode!r}, n_shards {shards!r} vs {prev_shards!r}) "
                    f"-- not like-for-like; needs a deliberate check before it can count")}
    elif key_val is not None:
        # dead-end: a lever below record — kept WITH its reason, deduped by tag (no noise-floor spam)
        ledger.setdefault("deadends", {})[tag] = {
            "score": key_val, "gain": fair, "why": bottleneck, "seq": _next_seq(ledger),
        }
        _evict_oldest_deadends(ledger)
    cur["ledger"] = ledger
    recs[a.metric] = cur
    if not _commit_records_if_unchanged(records_snapshot, recs):
        sys.exit(
            "[trace] WORKFLOW WITHHELD: records.json changed while the probe ran; "
            f"the probe log is preserved at {log_path}, but no record was written"
        )

    decision = (
        "rebaseline" if rebaseline
        else "record" if is_record
        else "migration_withheld" if migration_withheld
        else "no_record"
    )
    objective = {"primary": primary, "fair_st_gain": fair, "fair_baseline": fair_base,
                 "record": bool(is_record), "rebaseline": bool(rebaseline), "decision": decision,
                 "prev_record": prev, "record_value": key_val}
    code = _code_provenance()
    if code.get("dirty"):
        print("[trace] *** DIRTY TREE: this run measured uncommitted changes. The result is not "
              "reproducible by anyone else and is discovery-only. Commit before recording a claim.",
              flush=True)
    trace = {"metric": a.metric, "tag": tag,
             "code": code,
             "evidence": {"n_seeds": len(seed_values),
                          "seed_values": [float(v) for v in seed_values],
                          "seed_std": (float(seed_std) if seed_std is not None else None),
                          "provisional": len(seed_values) < MIN_CONFIRMATION_SEEDS},
             "objective": objective, "gains": gains, "header": header, "metrics": metrics,
             "bottleneck": bottleneck, "rc": rc, "ledger": ledger}

    # one-screen consistent summary ---------------------------------------------------------------------
    print("\n" + "=" * 76)
    print(f"OBJECTIVE  {a.metric}")
    print(header or "(no '=== SPACETIME' header parsed — check the log)")
    print("-" * 76)
    print(f"  primary(score) = {primary}   fair_st_gain = {fair} (vs {fair_base})   all_gains = {gains}")
    record_text = (
        "RE-BASELINE (not a comparable win)" if rebaseline
        else "YES (new best!)" if is_record
        else "WITHHELD (protocol migration mismatch)" if migration_withheld
        else "no"
    )
    print(f"  RECORD = {record_text}   prev_record = {prev}")
    print(f"  BOTTLENECK: {bottleneck}")
    print("  metrics:")
    for m in metrics:
        print("    " + m)
    print("=" * 76)
    out = Path(log_path).with_suffix(".trace.json")
    out.write_text(json.dumps(trace, indent=2))
    print(f"[trace] wrote {out}" + ("  |  RECORDS.json updated" if is_record else ""))

    _print_net_scorecard(recs, a.metric)   # show the whole board after every run
    print(f"[trace] scorecard -> {write_scorecard(recs)}", flush=True)

    if a.ensue:
        post_ensue(trace)


if __name__ == "__main__":
    # Two entry points, one file: `--list-modes` answers "what can move this capability, and where do I
    # edit?" without running anything; anything else runs the loop.
    if "--channels" in sys.argv:
        sys.argv.remove("--channels")
        _c = ""
        if "--metric" in sys.argv:
            _i = sys.argv.index("--metric"); _c = sys.argv[_i + 1]; del sys.argv[_i:_i + 2]
        raise SystemExit(_channels(_c))
    if "--insights" in sys.argv:
        sys.argv.remove("--insights")
        _cap = ""
        if "--metric" in sys.argv:
            _i = sys.argv.index("--metric"); _cap = sys.argv[_i + 1]; del sys.argv[_i:_i + 2]
        raise SystemExit(_insights(_cap, ensue="--no-ensue" not in sys.argv))
    if "--determinism" in sys.argv:
        sys.argv.remove("--determinism")
        raise SystemExit(_determinism_check())
    if "--list-modes" in sys.argv:
        sys.argv.remove("--list-modes")
        _list_modes()
    else:
        main()
