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
import shlex
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
LOOP = _HERE.parents[1]                     # autoresearch/<loop>
AUTORESEARCH = LOOP.parent                  # autoresearch
REPO = AUTORESEARCH.parent                  # the deepearth package root
assert REPO.name == "deepearth", f"expected the deepearth package root, resolved {REPO}"
sys.path.insert(0, str(REPO.parent))        # dir holding the deepearth package



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
# Identity is what makes two runs comparable (see autoresearch/spacetime/program/program.md):
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
    extras: Dict[str, Any] = field(default_factory=dict)
    # A diagnostic measures something that is NOT a scorecard capability, or measures it without
    # Earth4D in the comparison at all (several dynamics modes run on raw PE only). It is legitimate
    # research output, but it can never set a record, and saying so here is better than inventing a
    # capability for it so that it fits a slot on the board.
    diagnostic: bool = False
    diagnostic_reason: str = ""
    contract_version: int = CONTRACT_VERSION

    # -- identity ---------------------------------------------------------------------------------
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
        for key in ("capability", "split", "n_shards", "protocol"):
            if key in other and other[key] is not None and mine[key] != other[key]:
                return False
        return True

    # -- fair gain --------------------------------------------------------------------------------
    def fair_gain(self, order) -> tuple:
        """The honest marginal: the gain against the STRONGEST fair baseline present.

        Returns (value, label) or (None, None). The preference order is the harness's, not the probe's,
        so a mode cannot nominate a flattering baseline for itself.
        """
        for preference in order:
            for label, value in self.gains.items():
                if preference.lower() in label.lower():
                    return value, label
        return (None, None)

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
#     python -m deepearth.autoresearch.spacetime.editable_files.harness --capability family_from_env --list-modes
#     python -m deepearth.autoresearch.spacetime.editable_files.harness.harness.py --list-modes

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
    Mode("ENV-DECODE(<split>)", "--env_decode [--env_aux_weight W]",
         capability="family_from_env", lever=ARCH,
         notes="Trains the encoder end-to-end against an auxiliary env field. NOT reproducible "
               "run-to-run (gains move ~0.005), so a single-seed record here means nothing."),

    # ---- family_from_spacetime -------------------------------------------------------------------
    Mode("FORECAST(past->future)", "--forecast [--target family]",
         capability="family_from_spacetime", lever=ARCH,
         notes="The default coordinate path. --forecast_spatial switches to future+newplace."),
    Mode("FIELD-DECODE(<split>)", "--field_decode",
         capability="family_from_spacetime", lever=ARCH,
         notes="Trains the encoder end-to-end (rule 24). Also not reproducible run-to-run."),
    Mode("GNN(message-passing propagator)", "--gnn [--gnn_hops H] [--rec_k K]",
         capability="family_from_spacetime", requires=("--forecast",), lever=ARCH,
         notes="Declares the ENCODER gain (Earth4D GNN vs generic-PE GNN). propagator_gain, in "
               "extras, is propagation-vs-static on RAW features and gates nothing about Earth4D."),
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

    # ---- calibration -----------------------------------------------------------------------------
    Mode("CALIBRATION", "--feature earth4d --ensemble N   (module: calib_probe, not probe)",
         capability="calibration", lever=ARCH,
         notes="Lives in calib_probe.py and reports conf_auroc (0.5 = useless). The live 0.591 "
               "record has NO fair baseline, so its bottleneck is undiagnosable. Not yet on the "
               "result contract."),

)

# Where to make each kind of change. An agent that has picked a capability needs this more than it
# needs the file layout.
LEVER_SITES = {
    DATA: [
        "autoresearch/spacetime/editable_files/probe.py: load_env / load_vision / load_env_species "
        "(which channel feeds the head)",
        "flags: --env_channels, --env_extra, --sdm_channels, --vision --vision_feats, --pheno_channel",
        "data prep: occurrence densification, channel fusion, per-entity aggregation",
    ],
    ARCH: [
        "encoders/spacetime/earth4d.py: __init__, forward, training objective (the encoder itself)",
        "autoresearch/spacetime/editable_files/lib/recurrence.py: run_recurrence, run_field_decode, propagators",
        "autoresearch/spacetime/editable_files/lib/gnn.py: message passing",
        "flags: --recurrence, --gnn, --forecast, --env_decode, --field_decode, --fourier, "
        "--spatial_siren, --time_harmonics",
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
DEFAULT_PROBE_MODULE = "deepearth.autoresearch.spacetime.editable_files.probe"
TRACE_AUTH_FD_ENV = "EARTH4D_TRACE_AUTH_FD"

# The encoder-probeable capabilities (scorecard.md Layer 2). The objective must be one of these; the
# probe MODE and the architecture are the agent's choice. This list and scorecard.md Layer 2/3 are one
# contract -- change both together.
CAPABILITIES = [
    "species_from_env", "species_from_spacetime", "family_from_env", "family_from_spacetime",
    "community_from_env", "calibration", "flowering_peak_month",
]

# Declared-and-refused, with the reason (scorecard.md Layer 3). These used to sit in CAPABILITIES with
# no PRIMARY_RE entry, so a run would fall through to the generic r"\bEarth4D\s+([\d.]+)" pattern and
# record whatever number matched first -- a legal --metric that measured nothing in particular. An
# explicit refusal is the honest version: the capability is real on the full-model board, it is simply
# not reachable through the encoder probe.
EXCLUDED_CAPABILITIES = {
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
# a leak fix, a split change, a target/normalization change. Records carry the protocol they were set under,
# and a run under a different protocol RE-BASELINES the capability instead of "beating" it -- mode and shard
# count both match across such a change, so neither of those gates catches it.
#   v1-prefix     : everything up to 2026-07-29. Leaked in three ways (train mask admitted future-at-seen-place
#                   and past-at-held-place rows; time normalization fitted its span on test dates; env/vision
#                   standardization fitted mu/sd over test rows) and normalized time so the held-out future
#                   landed where the hash grid saturates.
#   v2-leakfix    : strict spatiotemporal split, train-only time normalization with horizon headroom,
#                   train-only feature standardization, deterministic seeding.
PROTOCOL = "v2-leakfix"
# Only explicitly identified, audited protocols may be migrated automatically.
# Absence of a protocol is not evidence that a hand-restored or pre-gate record
# belongs to the known v1 measurement regime.
REBASELINE_PROTOCOLS = frozenset({"v1-prefix"})

# Fair-baseline preference: Earth4D must beat a TRAINED generic PE, not just raw coords.
FAIR_ORDER = ["best-ctrl", "RFF", "mlp", "GAIN", "prop_acc", "best-coord", "raw"]


def _run(module: str, probe_args: str, device: str, log_path: str, result_path: str,
         capability: str) -> int:
    probe_argv = shlex.split(probe_args) + ["--device", device,
                                            "--result-json", result_path,
                                            "--capability", capability]
    cmd = [sys.executable, "-m", module] + probe_argv
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO.parent) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    read_fd, write_fd = os.pipe()
    try:
        authorization = (
            json.dumps(
                {"module": module, "argv": probe_argv},
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        os.write(write_fd, authorization)
        os.close(write_fd)
        write_fd = -1
        env[TRACE_AUTH_FD_ENV] = str(read_fd)
        print(f"[trace] $ {' '.join(cmd)}  (cwd={REPO})", flush=True)
        with open(log_path, "w") as lf:
            return subprocess.run(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(REPO),
                pass_fds=(read_fd,),
            ).returncode
    finally:
        if write_fd >= 0:
            os.close(write_fd)
        os.close(read_fd)










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
):
    """Return the like-for-like record decision and its component checks."""
    mode_ok = (prev is None) or (mode == prev_mode)
    shards_ok = (prev is None) or (shards == prev_shards)
    beats = key_val is not None and (prev is None or key_val > prev)
    rebaseline = (
        prev is not None
        and prev_proto in REBASELINE_PROTOCOLS
        and prev_proto != PROTOCOL
        and mode_ok
        and shards_ok
        and _same_probe(probe, prev_probe)
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




def _bottleneck(fair, primary) -> str:
    """Diagnose which lever family the fair-gain points at (program.md, section 3 Diagnose).

    This string is written into records.json AND pushed to Ensue as the swarm's reason-to-move, so it
    has to agree with the program. It previously read a flat/negative fair-gain as ARCHITECTURE-LIMITED
    and told the agent to "swing bigger on the architecture" -- the exact inverse of the program, which
    reads a flat gain as the INPUT being signal-limited. Under the old string every flat-gain run
    advised the whole swarm to do the one thing the program forbids ("Don't default to architecture").
    """
    if fair is None:
        return "NO-FAIR-BASELINE (probe reported no vs-generic-PE gain — check output)"
    if fair <= 0:
        return ("INPUT-LIMITED: Earth4D does not beat a generic trained PE, so the coordinate/current "
                "channel lacks the signal → DATA lever, change the channel")
    if primary is not None and primary < 0.20:
        return ("ENCODER-LIMITED: the encoder beats the PE but the absolute score is low → ARCHITECTURE "
                "lever, change the mechanism")
    return "EARNING: the architecture is carrying real signal → push it further"



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
    rows = []
    for cap in CAPABILITIES:
        r = recs.get(cap) or {}
        score, gain = r.get("score"), r.get("fair_st_gain")
        rows.append((
            cap,
            f"{score:.4f}" if isinstance(score, (int, float)) else "—",
            (r.get("probe_metric") or r.get("primary_metric") or "pre-contract"),
            f"{gain:+.4f}" if isinstance(gain, (int, float)) else "—",
            (r.get("fair_baseline") or "—"),
            (r.get("mode") or "—"),
            str(r.get("n_shards") or "—"),
            _read_of(r, gain, score),
            (r.get("ledger") or {}).get("runs", "—"),
        ))
    w = [max(len(str(r[i])) for r in rows + [("capability", "record", "metric", "fair-gain",
                                              "vs baseline", "mode", "shards", "read", "runs")])
         for i in range(9)]
    head = ("capability", "record", "metric", "fair-gain", "vs baseline", "mode", "shards", "read", "runs")
    lines = [
        "EARTH4D SPACETIME PROBE — SCORECARD",
        f"protocol {PROTOCOL}   ·   generated from records.json by harness.py   ·   DO NOT HAND-EDIT",
        "",
        "  ".join(h.ljust(w[i]) for i, h in enumerate(head)).rstrip(),
        "  ".join("-" * w[i] for i in range(9)),
    ]
    for r in sorted(rows, key=lambda x: (x[1] == "—", -float(x[1]) if x[1] != "—" else 0)):
        lines.append("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)).rstrip())
    earning = sum(1 for r in rows if r[3] != "—" and float(r[3]) > 0)
    probed = sum(1 for r in rows if r[1] != "—")
    lines += [
        "  ".join("-" * w[i] for i in range(9)),
        f"probed {probed}/{len(CAPABILITIES)}   ·   earning (fair-gain > 0): {earning}",
        "",
        "READ:  INPUT-LIMITED     Earth4D does not beat a generic trained PE -> DATA lever, change the channel",
        "       ENCODER-LIMITED   beats the PE but the absolute score is low -> ARCHITECTURE lever",
        "       EARNING           the architecture is carrying real signal -> push it further",
        "       STALE-GAIN        the stored gain is a CHANNEL advantage, not encoder-vs-PE. Re-measure",
        "                         before trusting the read (family_from_env: board +0.0411, live -0.0072)",
        "       pre-contract      record predates the result contract, so its metric name was never stored",
        "",
        "A record here is a PROBE record: discovery, not science. It becomes a claim only by clearing the",
        "evidence standard in program.md (>=5 matched seeds, block bootstrap, no regression, reproducible",
        "from a committed tree). See scorecard.md for what each row means and what is excluded.",
        "",
        "EXCLUDED — real on the full-model board, not reachable through this probe:",
    ]
    for cap, why in sorted(EXCLUDED_CAPABILITIES.items()):
        lines.append(f"  {cap:24} {why}")
    path.parent.mkdir(parents=True, exist_ok=True)
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
    val = (f"LOOP-earth4d {trace['metric']}: BEST {best.get('score')} (gain {best.get('gain')}, {o.get('fair_baseline')}) "
           f"via '{best.get('tag')}'. runs={led.get('runs')}. record-history: {rec_str}. "
           f"THIS RUN '{trace['tag']}': primary={o['primary']} gain={o['fair_st_gain']} "
           f"decision={o.get('decision', 'legacy')} "
           f"bottleneck={trace['bottleneck']}. dead-ends-tried: {dead_str}.")
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "create_memory", "arguments": {
        "items": [{"key_name": f"LOOP-earth4d-{trace['metric']}", "value": val,
                   "description": f"Earth4D encoder-probe loop {trace['metric']}: best {best.get('score')} "
                                  f"gain {best.get('gain')} over {led.get('runs')} runs"}]}}}
    req = urllib.request.Request("https://api.ensue-network.ai/", data=json.dumps(payload).encode(),
                                 headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json",
                                          "Accept": "application/json, text/event-stream"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            print(f"[trace] Ensue logged LOOP-earth4d-{trace['metric']} ({r.status})", flush=True)
    except Exception as e:
        print(f"[trace] Ensue POST failed: {e}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Earth4D legacy probe ledger — exact audited protocol migrations only"
    )
    ap.add_argument("--metric", required=True, help="objective capability (one of the scorecard capabilities)")
    ap.add_argument("--probe", required=True, help="probe flags = the architectural lever (quote the whole string)")
    ap.add_argument("--probe-module", default=DEFAULT_PROBE_MODULE)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ensue", action="store_true")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()

    if a.metric in EXCLUDED_CAPABILITIES:
        sys.exit("[trace] --metric %r is excluded: %s\n"
                 "        See autoresearch/spacetime/program/scorecard.md Layer 3."
                 % (a.metric, EXCLUDED_CAPABILITIES[a.metric]))
    if a.metric not in CAPABILITIES:
        sys.exit("[trace] --metric %r is not an encoder-probeable capability. one of:\n  %s"
                 % (a.metric, "\n  ".join(CAPABILITIES)))
    modes = for_capability(a.metric)
    if not modes:
        sys.exit(f"[trace] no recording probe mode measures {a.metric!r}. "
                 f"See harness.py --list-modes.")
    print(f"[trace] {a.metric}: {len(modes)} mode(s) can set this record — "
          + ", ".join(m.mode for m in modes), flush=True)
    records_snapshot, preflight_records = _read_records()

    tag = a.tag or ("e4d_" + re.sub(r"\W+", "_", a.probe)[:24].strip("_"))
    log_path = a.log or str(LOOP / "records" / "traces" / f"{tag}.log")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[trace] OBJECTIVE={a.metric}  probe='{a.probe}'  tag={tag}", flush=True)
    result_path = str(Path(log_path).with_suffix(".result.json"))
    rc = _run(a.probe_module, a.probe, a.device, log_path, result_path, a.metric)
    text = Path(log_path).read_text(errors="ignore")
    if rc != 0:
        print(text[-1800:])
        sys.exit(f"[trace] probe FAILED (rc={rc}); see {log_path}")

    # The probe DECLARES what it measured. Nothing here parses stdout: a mode that does not emit a
    # contract cannot set a record, which is the point -- the old parser always produced *something*.
    try:
        result = ProbeResult.read(result_path)
    except (ContractError, OSError) as exc:
        sys.exit(f"[trace] probe emitted no usable result contract: {exc}\n"
                 f"        log preserved at {log_path}; no record was written")
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

    primary = result.primary.value
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
        a.probe,
        cur.get("probe"),
    )
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
              f"[trace]     new={a.probe!r} mode={mode!r} shards={shards!r}",
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
               "fair_baseline": fair_base, "tag": tag, "probe": a.probe, "mode": mode, "n_shards": shards,
               "probe_module": a.probe_module, "protocol": PROTOCOL}
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
                f"n_shards={shards!r}, probe={a.probe!r})"
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
    trace = {"metric": a.metric, "tag": tag, "probe": a.probe, "probe_module": a.probe_module,
             "objective": objective, "gains": gains, "header": header, "metrics": metrics,
             "bottleneck": bottleneck, "rc": rc, "ledger": ledger}

    # one-screen consistent summary ---------------------------------------------------------------------
    print("\n" + "=" * 76)
    print(f"OBJECTIVE  {a.metric}   probe='{a.probe}'")
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
    if "--list-modes" in sys.argv:
        sys.argv.remove("--list-modes")
        _list_modes()
    else:
        main()