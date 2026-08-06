"""The single objective every loop reports.

`val_bpb` is held-out masked reconstruction scored as a proper likelihood, in bits per revealed dimension.
It shares the model's data, split, masking and decoder path but NOT its loss functions: training uses
centered cosine and log-C-normalized cross-entropy, which are not log-likelihoods. A change can therefore
improve one and worsen the other. What it does guarantee is one number at every scale.
It is additive over variables, so the aggregate aligns the loops and the per-variable decomposition
gives each loop its granular target.

Benchmarks are kept as diagnostics. They are not a promotion gate: the harmonic mean cannot resolve
model size (24.0M and 796M tie at 0.332 vs 0.319-0.325) because it is dominated by the near-zero
benchmarks neither model solves.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Optional, Sequence

SCORE_FLOOR = 1e-3          # keeps a harmonic diagnostic finite when a benchmark reads ~0
LN2 = math.log(2.0)


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

    `aggregate` weights by revealed dimensions, so six large embeddings carry 97.8% of it (clay alone
    30.1%) while `identity` -- species, the headline capability -- carries 0.029%. That measures
    reconstruction efficiency, not scientific coverage. Report both: aggregate is the reconstruction
    gate, macro is the balance number.
    """
    d = decompose(per_variable)
    return sum(d.values()) / max(len(d), 1)


def judge(before: Mapping[str, tuple], after: Mapping[str, tuple], floors: Mapping[str, float],
          weak: Sequence[str], owned: Sequence[str] = ()) -> dict:
    """Decide keep or discard on coverage, not just on the aggregate.

    Three conditions, all required:

    1. **Reconstruction gate** -- the MACRO mean improves by more than its floor. Macro, not the
       dimension-weighted aggregate, because training macro-averages: `_decode_loss` adds
       (err*w).sum()/w.sum() per variable, so each variable already contributes equally to the gradient.
       Gating on dimension weight would optimize one thing and judge another.
    2. **No owned regression** -- no variable this experiment claims may get worse by more than its own
       measured floor. An aggregate win paid for by a regression elsewhere is a trade, and rule 32
       forbids trades.
    3. **Coverage** -- at least one variable in `weak` must improve. Without it, moving one large
       embedding satisfies the aggregate alone: six embeddings carry 97.8% of it and clay carries 30.1%,
       so the model can get narrower while the number goes up.

    `weak` must be supplied by the caller from the BENCHMARK scores, which are commensurable in [0,1].
    It cannot be derived from bits/dim: that is a differential entropy whose scale reflects a variable's
    target variance, so ranking variables by it says nothing about which capability is behind.

    `floors` is per-variable and measured from matched seeds. There is no default -- a threshold you did
    not measure is a threshold that admits noise.
    """
    b, a = decompose(before), decompose(after)
    shared = [k for k in b if k in a]
    if not shared:
        return {"keep": False, "reason": "no shared variables between the two runs"}
    if not weak:
        return {"keep": False, "reason": "no weak set supplied; derive it from the benchmark scores"}

    agg_floor = floors.get("__aggregate__")
    if agg_floor is None:
        return {"keep": False, "reason": "no measured floor for the aggregate; measure before judging"}
    # Training macro-averages: `_decode_loss` adds (err*w).sum()/w.sum() per variable, so each variable
    # contributes equally regardless of width. Gate on the same view, or optimization pressure and the
    # gate disagree -- the dimension-weighted aggregate is reported alongside as efficiency.
    agg_gain = macro(before) - macro(after)                          # a loss: positive means improved

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
        "reason": ("regressed: " + ", ".join(f"{k} +{d:.4f} > {f:.4f}" for k, d, f in regressions)) if regressions
                  else ("no weak capability improved -- the model got narrower" if not improved_weak
                        else ("macro gain inside noise" if agg_gain <= agg_floor else "keep")),
    }


# ---------------------------------------------------------------- diagnostics

def is_diagnostic(k: str) -> bool:
    """A derived difference benchmark (`*_gain`): capability with minus without a mechanism."""
    return k.endswith("_gain")


def normalized(raw: Mapping[str, float]) -> Dict[str, float]:
    """Clip each benchmark to its own natural range, drop NaNs. A capability is [0,1]; an ablation
    delta is a difference of two, so [-1,1] -- clipping deltas to [0,1] would make a regression
    unrepresentable."""
    return {k: float(min(1.0, max(-1.0 if is_diagnostic(k) else 0.0, v))) for k, v in raw.items()
            if not (isinstance(v, float) and math.isnan(v))}


def suite_mismatch(before: Mapping[str, float], after: Mapping[str, float]):
    """(added, missing) keys between two runs. Non-empty means their aggregates are not comparable."""
    return sorted(set(after) - set(before)), sorted(set(before) - set(after))


def net_value(k: str, v: float) -> float:
    """Safe [0,1] contribution of one benchmark. Deltas map affinely: 0.5 neutral, 1.0 at +1, 0.0 at -1."""
    if is_diagnostic(k):
        return 0.5 + 0.5 * float(max(-1.0, min(1.0, v)))
    return max(v, SCORE_FLOOR)


def harmonic(raw: Mapping[str, float], suite: Optional[Iterable[str]] = None) -> float:
    """Harmonic mean over the declared suite. Diagnostic only -- never a gate. Pass ``suite`` or an
    undeclared key can move it (a CLI flag once did, by adding six ~0.5 terms)."""
    normed = normalized(raw)
    keys = normed.keys() if suite is None else [k for k in normed if k in set(suite)]
    vals = [net_value(k, normed[k]) for k in keys]
    return float(len(vals) / sum(1.0 / v for v in vals)) if vals else 0.0


def arithmetic(raw: Mapping[str, float]) -> float:
    """Arithmetic mean over capability benchmarks only. Moves when any benchmark improves."""
    vals = [v for k, v in normalized(raw).items() if not is_diagnostic(k)]
    return float(sum(vals) / len(vals)) if vals else 0.0
