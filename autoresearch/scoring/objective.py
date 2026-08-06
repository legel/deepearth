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
