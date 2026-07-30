"""The probe -> harness result contract.

Every probe mode already computes a structured dict and then `return`s it. Nothing consumed that
return value: `if __name__ == "__main__": main()` discarded it, and `trace.py` reconstructed the same
facts by regex-scraping the prose printed alongside it -- guessing the mode from one of 22 hand-written
`=== SPACETIME ...` header formats, the primary score from a per-capability regex table with two
hand-patched special cases, and the fair baseline from a preference list over whatever gain labels
happened to appear.

Every historical mis-record came from that interface rather than from a logic error:

  * `--pheno_disttarget peak_week` published 0.067 -> 0.683 because a DIFFERENT target printed an `acc`
    the regex accepted.
  * A run where the RFF control beat Earth4D recorded the CONTROL's accuracy as the Earth4D record,
    because the scan took the max over every `acc` in the output.
  * `calibration` maxed over four different uncertainty signals, so a run using a different signal could
    "beat" a record set on max-softmax.
  * Four mode paths print no `mode=` at all, so they all read `mode=None` and were therefore mutually
    "like-for-like" -- defeating the very gate that exists to stop cross-target comparison.

Each was patched with another regex or another gate while the surface producing the ambiguity kept
growing. This module removes the ambiguity instead: the probe DECLARES its identity and its numbers, the
harness validates and consumes them, and anything missing is a hard error rather than a silent None.

Identity is what makes two runs comparable (see agents/earth4d/program.md):

    capability . mode . split . n_shards . protocol . code hash

`mode` and `primary` are REQUIRED. A mode that cannot say what it measured cannot set a record.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

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
        if not self.capability:
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
    def render(self) -> str:
        """The single human-readable block, DERIVED from the result.

        Replaces 22 hand-written header formats. Because the harness no longer reads this text, a change
        here can never change what gets recorded -- which was the whole hazard of the old design.
        """
        lines = [
            f"=== SPACETIME | capability={self.capability} | mode={self.mode} "
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
        lines.append(f"  identity={self.identity_digest()}  seed={self.seed}  steps={self.steps}")
        return "\n".join(lines)
