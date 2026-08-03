"""The probe result contract — what a number must carry before it can become a record.

Shared by every probe loop, and deliberately OUTSIDE all of them: a loop that can edit its own contract
can define its way to a record. `definitions.py` owns what a number MEANS; this owns what a number must
PROVE about itself before that meaning applies.

    declare()  ->  ProbeResult (identity, validation, fair-gain)  ->  gate  ->  board  ->  scorecard

Everything here was written for, and paid for by, the spacetime loop -- it is lifted verbatim rather
than rewritten, because the comments record failures that cost something to find and should not be
rediscovered by the next loop. What changed in the lift is only parameterization: the pieces that named
Earth4D, the spacetime capability list, its protocol or its board path now take those as arguments.

A loop supplies its own capability list, PROTOCOL and history, records path, scorecard path, and
FAIR_ORDER. It does NOT supply the rules -- comparability, the noise barrier, the atomic board commit,
and what a fair gain IS are fixed here, where an experiment cannot reach them.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shlex
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from deepearth.autoresearch.scoring.definitions import noise_barrier



# ============================================================================================================
# 1. CONTRACT — what a probe must declare
# ============================================================================================================
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
    # Which loop measured this. Only `render` reads it -- the header used to be the literal string
    # "SPACETIME" back when there was one probe. It is NOT part of identity: two loops cannot produce
    # the same (capability, mode, metric) anyway, and making it identity would silently prevent a
    # re-baseline if a capability ever moved between loops.
    loop: str = ""

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
            f"=== {(self.loop or 'PROBE').upper()} | capability={self.capability or 'DIAGNOSTIC'} | mode={self.mode} "
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
# 2. EMIT — the one path by which a number becomes recordable
# ============================================================================================================


_RESULT_SINK = {"path": "", "capability": "", "protocol": "", "flags": "", "seed": None,
                "steps": None, "n_shards": None, "trained_encoder": False, "loop": ""}


def _set_result_sink(path, capability, protocol, args, config=None, loop=""):
    """Arm the result contract for this run. Called once, right after parse_args.

    `config` is the probe's CONFIG block — the levers that used to be CLI flags. It must reach the
    identity: with the levers in the file, two experiments have IDENTICAL command lines, so without a
    digest of what was actually built the gate would treat a rewired encoder as the same measurement as
    the control and let one 'beat' the other.
    """
    _RESULT_SINK.update({
        "config": dict(config or {}),
        "path": path or "", "capability": capability or "", "protocol": protocol, "loop": loop or "",
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
        loop=_RESULT_SINK.get("loop") or "",
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
# 3. GATE — like-for-like comparison, the noise barrier, the atomic board commit
# ============================================================================================================


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
    *,
    protocol,
    rebaseline_protocols=frozenset(),
):
    """Return the like-for-like record decision and its component checks.

    `beats` now means "beats the standing record BY MORE THAN THE NOISE", not "is a larger float".
    The old meaning is why seven consecutive +0.0006 steps could each be accepted as a new best.

    `protocol` and `rebaseline_protocols` are the CALLING loop's, passed rather than read from a
    module global, because each loop versions its measurement independently. They are keyword-only so
    a caller cannot supply them positionally and silently swap the two.
    """
    mode_ok = (prev is None) or _same_mode(mode, prev_mode)
    shards_ok = (prev is None) or (shards == prev_shards)
    barrier = noise_barrier(prev, seed_std, n_seeds)
    beats = key_val is not None and (prev is None or key_val > prev + barrier)
    rebaseline = (
        prev is not None
        and prev_proto in rebaseline_protocols
        and prev_proto != protocol
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
    current_comparison = prev is None or prev_proto == protocol
    is_record = (
        beats and mode_ok and shards_ok and current_comparison
    ) or (rebaseline and key_val is not None)
    return is_record, rebaseline, beats, mode_ok, shards_ok


def _read_records(path):
    """Read one exact board snapshot for optimistic concurrency control."""
    raw = path.read_bytes() if path.exists() else b""
    return raw, json.loads(raw or b"{}")


def _commit_records_if_unchanged(expected_raw, records, path):
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




def retire_record(capability, reason, path):
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


# ============================================================================================================
# 4. DIAGNOSE — which lever family the measurement points at
# ============================================================================================================


ENCODER_SHARE_FLOOR = 0.25      # below this, the encoder is adding little over a generic PE


def _bottleneck(fair, primary, floor=0.0, encoder="Earth4D", input_lever="") -> str:
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

    `floor` is the part of `primary` that BOTH arms inherit and neither can earn, and it exists because
    the plain `fair / primary` ratio does not survive the move to a second loop. The biological probe
    scores family-NN accuracy ~0.89, of which nearly all is the frozen text seed that the real tree and
    the null tree receive identically; the most the operator could ever contest is ~0.11, so the maximum
    attainable share is ~0.12 and EVERY biological row would print ENCODER-LIMITED by construction --
    the same failure as the old absolute `primary < 0.20` rule, re-entering through the denominator.
    Passing the seed score as `floor` makes the ratio "of the headroom the encoder could actually
    contest, how much did it take", which is scale-free in the way the original intended. The default
    0.0 is the spacetime case -- its head sees only encoder features, so the shared floor is already
    zero and the arithmetic is unchanged.

    `encoder` and `input_lever` only name things in the prose. A loop whose DATA lever is not "change
    the coordinate channel" has to be able to say what its own is.
    """
    if fair is None:
        return "NO-FAIR-BASELINE (probe reported no vs-generic-control gain — check output)"
    if fair <= 0:
        return (f"INPUT-LIMITED: {encoder} does not beat a matched generic control, so the channel "
                f"lacks the signal → DATA lever, "
                + (input_lever or "change the channel"))
    contested = None if primary is None else primary - floor
    if contested is None or contested <= 0:
        return f"EARNING: positive fair-gain {fair:+.4f} but no contested headroom to weigh it against"
    share = fair / contested
    if share < ENCODER_SHARE_FLOOR:
        return (f"ENCODER-LIMITED: {encoder} contributes only {share:.0%} of the contested headroom "
                f"over a generic control → ARCHITECTURE lever, change the mechanism")
    return (f"EARNING: {encoder} contributes {share:.0%} of the contested headroom over a generic "
            f"control → the mechanism is carrying real signal, push it further")


# ============================================================================================================
# 5. PUBLISH — the git-visible view of the campaign
# ============================================================================================================


def _read_of(record: dict, gain, score, fair_keywords=("rff", "mlp", "ctrl", "gain", "raw", "pe"),
             stale=(("vs best-coord-pe", "earth4d"),), **bottleneck_kw) -> str:
    """The diagnosis, flagged when the stored gain is not an encoder-vs-control comparison.

    A gain labelled e.g. "ENV vs best-coord-PE" measures the CHANNEL's advantage over coordinates, not
    Earth4D's over a generic PE, so reading it as ENCODER-LIMITED is wrong. family_from_env is stored that
    way: the board says +0.0411 while a live run reports -0.0072 vs RFF and the honest read is
    INPUT-LIMITED. Until such a record is re-measured, say so rather than print a confident wrong word.

    `fair_keywords` and `stale` are the calling loop's vocabulary for the same two questions -- is this
    baseline label one of MY fair controls, and is it a known-stale one? Both are per-loop because the
    labels are: the biological board's fair control is "vs null-tree", which none of the spacetime
    keywords match, so an unparameterized version would flag every biological row CHECK-BASELINE.
    """
    if gain is None:
        return "NO-FAIR-BASELINE"
    baseline = (record.get("fair_baseline") or "").lower()
    if baseline and not any(k in baseline for k in fair_keywords):
        return "CHECK-BASELINE"
    for marker, absent in stale:
        if marker in baseline and absent not in baseline:
            return "STALE-GAIN(channel)"
    return _bottleneck(gain, score, **bottleneck_kw).split(":")[0]


def write_scorecard(recs: dict, path: Path, capabilities, protocol: str, title: str,
                    legend=(), read_of=_read_of) -> Path:
    """Render the board as plain text for fast skimming, and for git.

    `records.json` is gitignored and lives on the box, so nobody reading the repository can see where
    the campaign actually stands. This file is that view: every metric, its current best, the fair gain
    against the strongest fair baseline, and the diagnosis. It is GENERATED after every run, so it
    cannot drift from the board -- do not hand-edit it. `scorecard.md` next to it explains what each
    row means and how a record is earned; this one is only the numbers.

    `legend` is the loop's own three-line reading key -- what INPUT-LIMITED means depends on what the
    fair control was, so the loop that chose the control writes the words. `read_of` lets a loop bind
    its own baseline vocabulary and bottleneck floor without re-implementing the row layout.
    """
    # ONE SCREEN. The previous layout was 10 columns and ~160 chars, so it wrapped in any normal
    # terminal and the board -- the thing an agent reads before every pick -- was unreadable. Anything
    # that is not a number you act on moved to scorecard.md or the ledger.
    rows = []
    for cap in capabilities:
        r = recs.get(cap) or {}
        score, gain = r.get("score"), r.get("fair_st_gain")
        seeds = r.get("n_seeds")
        rows.append((
            cap,
            f"{score:.4f}" if isinstance(score, (int, float)) else "—",
            f"{gain:+.4f}" if isinstance(gain, (int, float)) else "—",
            read_of(r, gain, score),
            str((r.get("ledger") or {}).get("runs", "—")),
            (f"{seeds}s" if seeds else "1s?"),
            (r.get("protocol") or "—"),
        ))
    head = ("capability", "record", "fair-gain", "read", "runs", "seeds", "protocol")
    w = [max(len(str(r[i])) for r in rows + [head]) for i in range(len(head))]
    lines = [
        title,
        f"protocol {protocol}   ·   generated by the harness   ·   DO NOT HAND-EDIT",
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
        f"probed {probed}/{len(capabilities)}   ·   earning (fair-gain > 0): {earning}",
        "",
        *legend,
        "a record under a superseded protocol is VOID -- it measured something else. See scorecard.md.",
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def _print_net_scorecard(recs: dict, current: str, capabilities, title: str) -> None:
    """The NET scorecard: every capability's current best encoder-probe record. Printed after every run."""
    print("\n" + "#" * 76)
    print(f"# NET SCORECARD  —  {title}")
    print("#" * 76)
    print(f"{'capability':<26}{'record':>9}{'fair_gain':>11}  best-lever")
    earning = 0
    for cap in capabilities:
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
    probed = sum(1 for c in capabilities if recs.get(c))
    print("-" * 76)
    print(f"probed {probed}/{len(capabilities)}   |   earning (fair_gain > 0): {earning}")
    print("#" * 76, flush=True)


# ============================================================================================================
# 6. COMMIT — provenance, the gate decision, and the ledger, in one place
# ============================================================================================================
#
# The record SHAPE is not a per-loop choice. `graduation.py` reads `score`, `primary`, `code.dirty`,
# `code.commit`, `provisional` and `n_seeds` off every board to decide whether a finding may cross into
# the champion, so a loop that invents its own field names produces records that can never graduate --
# which is exactly the state the biological loop was in (its board did not exist at all). This is the
# one writer, so the shape is the same by construction rather than by two authors agreeing.


def code_provenance(repo) -> dict:
    """Commit SHA + dirty flag for the tree this run measured.

    The evidence standard says a record from an unpushed commit is discovery-only, because nobody else
    can reproduce it. That rule was unenforceable: nothing recorded WHICH commit produced a number. It
    is not hypothetical -- a foreign agent's record on this board claims a `trained_rff` baseline that
    exists in no reachable tree, and a run of this loop was contaminated for an hour by an uncommitted
    edit to earth4d.py that nothing in the record would have revealed.
    """
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:                                          # noqa: BLE001
            return ""
    # records/ is EXCLUDED: the harness writes the board and the scorecard on every run, so counting
    # them would make every result permanently "dirty" and the flag would mean nothing. Dirty here means
    # the CODE that produced the number is uncommitted.
    changed = [l for l in git("status", "--porcelain").splitlines()
               if l.strip() and "/records/" not in l and "/program/scorecard.txt" not in l]
    return {"commit": git("rev-parse", "HEAD")[:12],
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(changed)}


def commit_result(records: dict, capability: str, result: "ProbeResult", *, fair_order, protocol,
                  rebaseline_protocols, repo, tag: str, bottleneck: str, seed_values=(),
                  min_confirmation_seeds: int = 2) -> dict:
    """Gate one result against the standing record, update the ledger, and return the decision.

    Mutates `records` in place; the caller commits it atomically. Every refusal is printed with its
    reason and recorded as a dead-end, because a lever that failed is a result -- the thing that must
    never happen is a number quietly becoming a record it did not earn.
    """
    seed_values = [float(v) for v in (seed_values or [result.primary.value])]
    primary = sum(seed_values) / len(seed_values)          # the MEAN across seeds, never the max
    seed_std = (statistics.stdev(seed_values) if len(seed_values) >= 2 else None)
    fair, fair_base = result.fair_gain(fair_order)
    cur = records.get(capability, {})
    prev, prev_proto = cur.get("score"), cur.get("protocol")
    prev_mode, prev_shards = cur.get("mode"), cur.get("n_shards")

    is_record, rebaseline, beats, mode_ok, shards_ok = _record_gate(
        primary, prev, prev_proto, result.mode, prev_mode, result.n_shards, prev_shards,
        cur.get("probe"), seed_std=seed_std, n_seeds=len(seed_values),
        protocol=protocol, rebaseline_protocols=rebaseline_protocols)

    # A DIRTY TREE CANNOT SET A RECORD. With the levers in the file rather than the CLI, the experiment
    # IS the code diff, so a record measured on uncommitted code has a configuration nobody can ever
    # recover -- the number survives and the thing that produced it does not.
    code = code_provenance(repo)
    if code.get("dirty") and is_record:
        print("[trace] *** RECORD WITHHELD: DIRTY TREE. The run measured uncommitted changes, so its "
              "configuration is not recoverable by anyone else. Commit, then re-run to claim it.",
              flush=True)
        is_record = rebaseline = False
    barrier = noise_barrier(prev, seed_std, len(seed_values))
    if prev is not None and primary > prev and not beats:
        print(f"[trace] *** WITHIN NOISE: {primary:.6f} beats {prev} by {primary - prev:+.6f}, under "
              f"the barrier of {barrier:.6f}. Not a record.", flush=True)
    if beats and not (mode_ok and shards_ok):
        why = (f"mode {result.mode!r} != record mode {prev_mode!r}" if not mode_ok
               else f"n_shards {result.n_shards!r} != record n_shards {prev_shards!r}")
        print(f"[trace] *** RECORD WITHHELD: {why} — not a like-for-like comparison.", flush=True)

    ledger = cur.get("ledger", {"runs": 0, "records": [], "deadends": {}})
    ledger["runs"] = ledger.get("runs", 0) + 1
    if is_record:
        cur = {"score": primary, "primary": primary, "fair_st_gain": fair, "code": code,
               "n_seeds": len(seed_values), "seed_values": seed_values,
               "seed_std": (float(seed_std) if seed_std is not None else None),
               "provisional": len(seed_values) < min_confirmation_seeds,
               "fair_baseline": fair_base, "tag": tag, "mode": result.mode,
               "n_shards": result.n_shards, "protocol": protocol}
        ledger["records"] = (ledger.get("records", []) + [
            {"tag": tag, "score": primary, "gain": fair, "protocol": protocol,
             "rebaseline_from": prev if rebaseline else None}])[-20:]
    else:
        ledger.setdefault("deadends", {})[tag] = {
            "score": primary, "gain": fair, "why": bottleneck, "seq": _next_seq(ledger)}
        _evict_oldest_deadends(ledger)
    cur["ledger"] = ledger
    records[capability] = cur
    return {"record": bool(is_record), "rebaseline": bool(rebaseline), "primary": primary,
            "fair_gain": fair, "fair_baseline": fair_base, "prev_record": prev,
            "decision": ("rebaseline" if rebaseline else "record" if is_record else "no_record")}
