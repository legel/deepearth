"""Emitting a ProbeResult: the one place a probe mode says what it measured.

Separate from probe.py so mode modules can import `declare` without importing the probe back --
probe.py imports the mode modules, so the dependency has to point this way.

`declare` is the ONLY way a number becomes recordable. A mode that does not call it produces no
contract, and trace.py then refuses to write a record rather than falling back to parsing stdout.
"""
from __future__ import annotations

import sys

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
    from deepearth.autoresearch.spacetime.editable_files.harness.probe_contract import Primary, ProbeResult

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
