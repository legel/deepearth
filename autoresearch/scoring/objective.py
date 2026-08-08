"""The scoring primitives every loop reports.

`val_bpb` is held-out masked reconstruction scored as a proper likelihood, in bits per revealed dimension.
It shares the model's data, split, masking and decoder path but NOT its loss functions: training uses
centered cosine and log-C-normalized cross-entropy, which are not log-likelihoods. A change can therefore
improve one and worsen the other. What it does guarantee is one number at every scale.
It is additive over variables, so the aggregate aligns the loops and the per-variable decomposition
gives each loop its granular target.

Human-interpretable capabilities decide promotion. ``val_bpb`` and its decomposition remain the
likelihood lens used to understand where a change landed; they do not decide whether a champion ships.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Optional, Sequence

SCORE_FLOOR = 1e-3          # keeps the capability harmonic finite when a benchmark reads ~0
LN2 = math.log(2.0)

# Quarantine is evidence-based, not a way to hide a weak score. B55 predicts from focal identity +
# environment, then scores against the neighbors' pollinator union; relatives' pollinators are never
# inputs. Keep reporting it until repaired, but do not call it phylogenetic transfer.
QUARANTINED_BENCHMARKS = {
    "B55_pollinator_phylo_transfer_recall":
        "focal identity + environment is scored against neighbors' pollinators; relatives are not inputs",
}


# ---------------------------------------------------------------- the objective

def bits_per_dim(total_nats: float, dims: int) -> float:
    """Nats summed over revealed dimensions -> bits per dimension."""
    return total_nats / max(dims, 1) / LN2


def aggregate(per_variable: Mapping[str, tuple]) -> float:
    """Aggregate val_bpb from ``{variable: (nats, dims)}``.

    The sum, not the mean of per-variable rates: a variable with more revealed dimensions carries
    proportionally more of the objective, exactly as it does in training.
    """
    nats = sum(float(n) for n, _ in per_variable.values())
    dims = sum(int(d) for _, d in per_variable.values())
    return bits_per_dim(nats, dims)


def decompose(per_variable: Mapping[str, tuple]) -> Dict[str, float]:
    """Per-variable bits/dim. The granular target -- a loop steers by the variables it owns, and cannot
    win by trading another variable's budget away."""
    return {k: bits_per_dim(float(n), int(d)) for k, (n, d) in per_variable.items()}


def improved(before: float, after: float, floor: float) -> bool:
    """val_bpb is a loss: lower is better. ``floor`` is the noise measured AT THIS SCALE, not a constant.

    Fixed thresholds are what let the campaign promote inside its own noise -- champion steps of
    +0.0013 to +0.0034 against two-seed spreads of 0.0033 (172.6M), 0.0167 (21.8M) and 0.027 (796M).
    """
    return (before - after) > floor


def noise_floor(values: Sequence[float]) -> float:
    """Full spread across matched seeds of one configuration. Needs >= 2 seeds; there is no default."""
    vals = [float(v) for v in values]
    if len(vals) < 2:
        raise ValueError("a noise floor needs at least two matched seeds")
    return max(vals) - min(vals)


def macro(per_variable: Mapping[str, tuple]) -> float:
    """Unweighted mean of per-variable bits/dim -- every scientific capability counts equally.

    `aggregate` weights by revealed dimensions, and measurement shows that is degenerate: `climate`
    carries 95.3% of it, every other capability about 0.07-0.9%, `identity` -- species, the headline
    capability -- 0.076%. Directional variables are scored by retrieval against a frozen bank, so they
    contribute ONE dimension each regardless of native width. That measures reconstruction efficiency,
    not scientific coverage. Report both as likelihood diagnostics.
    """
    d = decompose(per_variable)
    return sum(d.values()) / max(len(d), 1)


def diagnose_likelihood(before: Mapping[str, tuple], after: Mapping[str, tuple],
                        floors: Mapping[str, float], weak: Sequence[str],
                        owned: Sequence[str] = ()) -> dict:
    """Diagnose whether a likelihood change landed as hypothesized.

    This is not the champion gate. It explains whether a likelihood change landed where expected.

    Three conditions, all required:

    1. **Reconstruction signal** -- the MACRO mean improves by more than its floor. Macro, not the
       dimension-weighted aggregate, because training macro-averages: `_decode_loss` adds
       (err*w).sum()/w.sum() per variable, so each variable already contributes equally to the gradient.
       Gating on dimension weight would optimize one thing and judge another.
    2. **No owned regression** -- no variable this experiment claims may get worse by more than its own
       measured floor. This is causal evidence about where the likelihood change landed.
    3. **Coverage** -- at least one variable in `weak` must improve. Without it, moving one variable
       satisfies the aggregate alone: `climate` measures at 95.3% of it, so the model can get narrower.
    """
    b, a = decompose(before), decompose(after)
    shared = [k for k in b if k in a]
    if not shared:
        return {"keep": False, "reason": "no shared variables between the two runs"}

    if not weak:
        return {"keep": False, "reason": "no weak set supplied; derive it from benchmark scores"}

    agg_floor = floors.get("__aggregate__")
    if agg_floor is None:
        return {"keep": False, "reason": "no measured floor for macro; measure before judging"}
    agg_gain = macro(before) - macro(after)

    regressions = []
    for k in (owned or shared):
        if k not in shared:
            continue
        floor = floors.get(k)
        if floor is None:
            return {"keep": False, "reason": f"no measured floor for {k}; measure before judging"}
        if (a[k] - b[k]) > floor:
            regressions.append((k, a[k] - b[k], floor))

    improved_weak = [k for k in weak
                     if k in shared and (b[k] - a[k]) > floors.get(k, float("inf"))]
    keep = agg_gain > agg_floor and not regressions and bool(improved_weak)
    return {
        "keep": keep,
        "macro_gain": agg_gain,
        "aggregate_before": aggregate(before), "aggregate_after": aggregate(after),
        "macro_before": macro(before), "macro_after": macro(after),
        "regressions": regressions,
        "improved_weak": improved_weak,
        "reason": ("regressed: " + ", ".join(f"{k} +{d:.4f} > {f:.4f}" for k, d, f in regressions))
                  if regressions else ("no weak capability improved" if not improved_weak
                                       else ("macro gain inside noise" if agg_gain <= agg_floor else "keep")),
    }


# ---------------------------------------------------------------- diagnostics

def is_diagnostic(k: str) -> bool:
    """A derived difference benchmark (`*_gain`): capability with minus without a mechanism."""
    return k.endswith("_gain")


def is_uncalibrated(k: str) -> bool:
    """A representation score whose raw scale has no human meaning without an empirical null."""
    return k.endswith("_cos")


def capability_suite(raw: Mapping[str, float]) -> tuple[str, ...]:
    """Comparable human capabilities present in one run, in stable order."""
    return tuple(sorted(k for k in normalized(raw)
                        if not is_diagnostic(k) and not is_uncalibrated(k)
                        and k not in QUARANTINED_BENCHMARKS))


def normalized(raw: Mapping[str, float]) -> Dict[str, float]:
    """Clip each benchmark to its own natural range, drop NaNs. A capability is [0,1]; an ablation
    delta is a difference of two, so [-1,1] -- clipping deltas to [0,1] would make a regression
    unrepresentable."""
    return {k: float(min(1.0, max(-1.0 if is_diagnostic(k) else 0.0, v))) for k, v in raw.items()
            if not (isinstance(v, float) and math.isnan(v))}


def harmonic(raw: Mapping[str, float], suite: Optional[Iterable[str]] = None) -> float:
    """Harmonic mean over a declared human-capability suite.

    Callers must bind ``suite`` for a promotion comparison. That keeps optional diagnostics, quarantine
    entries and CLI-dependent outputs from moving the primary score.
    """
    normed = normalized(raw)
    declared = set(capability_suite(normed) if suite is None else suite)
    keys = [k for k in normed if k in declared and not is_diagnostic(k) and not is_uncalibrated(k)
            and k not in QUARANTINED_BENCHMARKS]
    vals = [max(normed[k], SCORE_FLOOR) for k in keys]
    return float(len(vals) / sum(1.0 / v for v in vals)) if vals else 0.0


def arithmetic(raw: Mapping[str, float], suite: Optional[Iterable[str]] = None) -> float:
    """Arithmetic breadth guard over the same declared human-capability suite."""
    normed = normalized(raw)
    declared = set(capability_suite(normed) if suite is None else suite)
    vals = [v for k, v in normed.items()
            if k in declared and not is_diagnostic(k) and not is_uncalibrated(k)
            and k not in QUARANTINED_BENCHMARKS]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _run_summary(runs: Sequence[Mapping[str, float]], suite: Sequence[str]) -> dict:
    if len(runs) != 2:
        raise ValueError("a promotion decision needs exactly two benchmark seeds")
    declared = tuple(sorted(suite))
    if not declared:
        raise ValueError("the capability suite is empty")
    for i, run in enumerate(runs):
        observed = capability_suite(run)
        if observed != declared:
            raise ValueError(f"run {i} capability suite is {observed}, expected {declared}")
    harmonics = [harmonic(run, declared) for run in runs]
    arithmetics = [arithmetic(run, declared) for run in runs]
    return {
        "harmonic": sum(harmonics) / len(harmonics),
        "arithmetic": sum(arithmetics) / len(arithmetics),
        "harmonic_floor": noise_floor(harmonics),
        "arithmetic_floor": noise_floor(arithmetics),
    }


def judge(before_runs: Sequence[Mapping[str, float]], after_runs: Sequence[Mapping[str, float]], *,
          before_suite: Sequence[str], after_suite: Sequence[str],
          before_protocol: str, after_protocol: str) -> dict:
    """Judge promotion on human capabilities measured over matched two-seed runs.

    Harmonic improvement must beat the incumbent seed spread. Arithmetic may not regress beyond its
    incumbent spread. Likelihood metrics are deliberately absent: they are scorecard diagnostics.
    """
    if before_protocol != after_protocol:
        return {"keep": False, "reason": "benchmark protocols differ"}
    if tuple(sorted(before_suite)) != tuple(sorted(after_suite)):
        return {"keep": False, "reason": "capability suites differ"}
    try:
        before = _run_summary(before_runs, before_suite)
        after = _run_summary(after_runs, after_suite)
    except ValueError as exc:
        return {"keep": False, "reason": str(exc)}

    harmonic_gain = after["harmonic"] - before["harmonic"]
    arithmetic_regression = before["arithmetic"] - after["arithmetic"]
    keep = (harmonic_gain > before["harmonic_floor"] and
            arithmetic_regression <= before["arithmetic_floor"])
    if harmonic_gain <= before["harmonic_floor"]:
        reason = (f"harmonic gain {harmonic_gain:+.6f} does not beat "
                  f"its {before['harmonic_floor']:.6f} two-seed floor")
    elif arithmetic_regression > before["arithmetic_floor"]:
        reason = (f"arithmetic regressed {arithmetic_regression:.6f}, beyond "
                  f"its {before['arithmetic_floor']:.6f} two-seed floor")
    else:
        reason = "keep"
    return {
        "keep": keep,
        "harmonic_gain": harmonic_gain,
        "arithmetic_regression": arithmetic_regression,
        "before": before,
        "after": after,
        "reason": reason,
    }
