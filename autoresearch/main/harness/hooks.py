"""Independent runtime hooks for the two encoder loops -- monkeypatch a live model, NEVER edit core.

Measures an encoder's marginal contribution + its bottleneck without touching fusion.py / evaluate.py.
Everything here is reversible and lives in the harness so the champion path stays byte-identical;
consolidate into core later. Minimal by design -- this is the fast-feedback measurement layer only.
"""
from __future__ import annotations
import contextlib
from typing import Dict

import torch
import torch.nn.functional as F


def refinement_norm(model) -> Dict[str, float]:
    """BIOLOGICAL bottleneck: how far the species-graph moves species OFF their seed.
    ``‖refined − seed‖ ≈ 0`` ⟹ the graph is inert (learns identity) -- the core thing to fix.
    Direct call of the graph module only (no full forward) -> cheap."""
    g = getattr(model, "species_graph", None)
    if g is None:
        return {}
    with torch.no_grad():
        refined = g()          # [n_species, d_model]  (mask=None)
        seed = g._seed()       # [n_species, d_model]
        d = (refined - seed).norm(dim=-1)
        out = {
            "refined_seed_norm": float(d.mean()),
            "refined_seed_norm_p90": float(torch.quantile(d, 0.9)),
            "seed_norm": float(seed.norm(dim=-1).mean()),
        }
        for name, p in g.named_parameters():          # OU decay rate(s): softplus(theta / log_rate)
            leaf = name.split(".")[-1]
            if leaf in ("theta", "log_rate"):
                out[f"ou_rate_{leaf}"] = float(F.softplus(p.detach()).mean())
    return out


@contextlib.contextmanager
def ablate_earth4d(model):
    """Production Earth4D ablation: remove both absolute and relative coordinate encodings.

    This is deliberately *not* the probe's trained matched-RFF control.  It answers the production
    question "what does Earth4D contribute to fusion?" by leaving environmental/value tokens and
    non-Earth4D priors intact while removing every Earth4D-derived position vector.
    """
    patched = []
    for attr in ("absolute_proj_s", "absolute_proj_t"):
        m = getattr(model, attr, None)
        if m is None:
            continue
        orig = m.forward
        m.forward = (lambda o: (lambda *a, **k: torch.zeros_like(o(*a, **k))))(orig)
        patched.append((m, orig))

    # The neighbor field owns the relative Earth4D channel.  Zero its encoded position while leaving
    # NeighborContext's value features and field marker untouched, so this remains an encoder ablation
    # rather than a neighbor-data ablation.
    neighbors = getattr(model, "neighbors", None)
    relative = getattr(neighbors, "space_time", None)
    if relative is not None:
        orig = relative.forward
        relative.forward = (lambda o: (
            lambda query_coords, neighbor_coords, *a, **k:
                torch.zeros((*neighbor_coords.shape[:2], model.d_model),
                            device=neighbor_coords.device,
                            dtype=next(relative.parameters()).dtype)
        ))(orig)
        patched.append((relative, orig))
    try:
        yield
    finally:
        for m, orig in patched:
            m.forward = orig


# Compatibility name for existing callers.  The semantics are now the honest total-Earth4D ablation.
ablate_spacetime = ablate_earth4d


@contextlib.contextmanager
def ablate_species(model):
    """BIOLOGICAL control: graph OFF via the model's EXISTING ``_ablate_species`` flag (used upstream to
    swap ``species_graph()`` for ``species_graph._seed()``). Provided for symmetry; the B56-B62 gains
    already use this internally, so bio_gain needs no run-time ablation -- it is read straight off the log."""
    prev = getattr(model, "_ablate_species", False)
    model._ablate_species = True
    try:
        yield
    finally:
        model._ablate_species = prev


# capability -> production Earth4D-ablation gain name (mirrors score.ST_GAIN)
ST_GAIN_MAP = {
    "B1_species_from_env_top10": "B1_species_spacetime_gain",
    "B6_family_from_env": "B6_family_spacetime_gain",
    "B34_lfmc_from_env": "B34_lfmc_spacetime_gain",
    "B42_mycorrhiza_from_env": "B42_mycorrhiza_spacetime_gain",
    "B51_pollinator_from_env_recall": "B51_pollinator_spacetime_gain",
    "B23_species_calibration_mrr": "B23_calibration_spacetime_gain",
}


def instrument(spacetime_gain: bool = True) -> None:
    """Wrap ``evaluate.evaluate_benchmarks`` (in-process monkeypatch) so a run emits the encoder feedback
    signal INTO its own log -- with NO edit to evaluate.py / fusion.py:
      - always: the biological bottleneck  ``[profile] refined_seed_norm=...`` (cheap: one graph call).
      - Earth4D gains are part of ``evaluate_benchmarks`` itself, so every run emits the same suite.

    ``spacetime_gain`` remains in the signature for old launchers but is ignored.  A CLI flag must not
    be able to change which scientific quantities a champion run measures.
    Install before training so the ablation deltas are available to the evaluator."""
    from deepearth.autoresearch.main.harness import evaluate as ev
    if getattr(ev, "_programs_instrumented", False):
        return
    orig = ev.evaluate_benchmarks

    def wrapped(model, source, device, *a, **k):
        for key, val in refinement_norm(model).items():
            print(f"[profile] {key}={val:.6g}", flush=True)
        return orig(model, source, device, *a, **k)

    ev.evaluate_benchmarks = wrapped
    ev._programs_instrumented = True
