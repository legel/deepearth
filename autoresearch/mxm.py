"""M x M reconstruction matrix: hide each modality, reconstruct each, to expose information flow.

For every choice of observed variables, condition on them plus the always-available space-time context and predict
every reconstructable variable on held-out data. The resulting matrix shows which inputs inform which targets: a
strong entry means one variable carries information about another. Two views are produced:

  - single-source: condition on one variable at a time (the rows), predict every target (the columns);
  - leave-one-out: condition on all variables but one, predict the held-out one, to measure each variable's unique
    contribution beyond the rest.

Scores are accuracy for categorical variables and cosine similarity for continuous ones, always on the held-out
split so they measure transfer rather than memorization.
"""
from __future__ import annotations
from typing import Dict, List, Sequence
import numpy as np
import torch
import torch.nn.functional as F


def _score(kind: str, pred: torch.Tensor, target: torch.Tensor, observed: torch.Tensor):
    """Return ``(sum_of_scores, count)`` over the observed rows only, so unobserved targets never poison the mean."""
    if kind == "categorical":
        s = (pred.argmax(-1) == target).float()
    else:
        s = F.cosine_similarity(pred, target, dim=-1)
    m = observed.to(s.dtype)
    return (s * m).sum().item(), m.sum().item()


@torch.no_grad()
def reconstruction_matrix(model, source, indices: np.ndarray, device: str, batch: int = 4096) -> dict:
    """Condition on each single variable and predict every reconstructable target over ``indices``.

    Returns ``{"variables", "targets", "single": [given][target] -> score, "leave_one_out": {target -> score}}``.
    ``model`` is a trained :class:`DeepEarth`; ``source`` a data adapter; ``indices`` the held-out rows.
    """
    model.eval()
    kinds = {v.name: v.kind for v in model.variables}
    names = [v.name for v in model.variables]
    targets = [v.name for v in model.variables if v.reconstruct]

    single = {g: {t: [0.0, 0.0] for t in targets if t != g} for g in names}   # [sum, count]
    loo = {t: [0.0, 0.0] for t in targets}
    for c0 in range(0, len(indices), batch):
        idx = torch.tensor(indices[c0:c0 + batch], device=device)
        values, observed, coords, nbr_coords, manifold_positions, nbr_values = source.batch(idx)
        ctx = model.context(coords, nbr_coords, manifold_positions, nbr_values)
        for g in names:                                          # single-source rows
            preds = model.infer(values, [g], [t for t in targets if t != g], ctx)
            for t, p in preds.items():
                s, c = _score(kinds[t], p, values[t], observed[t]); single[g][t][0] += s; single[g][t][1] += c
        for t in targets:                                        # leave-one-out
            preds = model.infer(values, [x for x in names if x != t], [t], ctx)
            s, c = _score(kinds[t], preds[t], values[t], observed[t]); loo[t][0] += s; loo[t][1] += c
    reduce = lambda sc: sc[0] / sc[1] if sc[1] > 0 else float("nan")
    single = {g: {t: reduce(sc) for t, sc in row.items()} for g, row in single.items()}
    return {"variables": names, "targets": targets, "single": single, "leave_one_out": {t: reduce(sc) for t, sc in loo.items()}}


def format_matrix(result: dict, width: int = 9) -> str:
    """Render the single-source matrix and the leave-one-out row as aligned text."""
    names, targets = result["variables"], result["targets"]
    lines = ["M x M single-source transfer (row = given, col = predicted target):"]
    header = " " * 14 + "".join(f"{t[:width-1]:>{width}}" for t in targets)
    lines.append(header)
    for g in names:
        row = result["single"][g]
        cells = "".join((f"{row[t]:>{width}.3f}" if t in row else f"{'--':>{width}}") for t in targets)
        lines.append(f"{g[:13]:<14}{cells}")
    lines.append("")
    lines.append("leave-one-out (predict each from all others): " +
                 " | ".join(f"{t} {result['leave_one_out'][t]:.3f}" for t in targets))
    return "\n".join(lines)


# --------------------------------------------------------------------------------------- standalone unit test
def _test():
    # scoring logic on synthetic predictions (masked sum, count)
    cat_pred = torch.tensor([[0.1, 0.9], [0.8, 0.2]]); cat_true = torch.tensor([1, 0]); obs = torch.tensor([True, True])
    s, c = _score("categorical", cat_pred, cat_true, obs); assert abs(s / c - 1.0) < 1e-6
    con_pred = torch.tensor([[1.0, 0.0]]); con_true = torch.tensor([[1.0, 0.0]])
    s, c = _score("continuous", con_pred, con_true, torch.tensor([True])); assert abs(s / c - 1.0) < 1e-6
    # an unobserved row contributes nothing (no NaN poisoning)
    s, c = _score("continuous", con_pred, torch.zeros(1, 2), torch.tensor([False])); assert c == 0
    # formatter on a tiny synthetic result
    result = {"variables": ["a", "b"], "targets": ["a", "b"],
              "single": {"a": {"b": 0.5}, "b": {"a": 0.7}}, "leave_one_out": {"a": 0.6, "b": 0.8}}
    text = format_matrix(result)
    assert "single-source" in text and "leave-one-out" in text and "0.500" in text
    print("mxm.py: all unit tests passed")


if __name__ == "__main__":
    _test()
