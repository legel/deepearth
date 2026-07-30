"""Independent runtime hooks for the two encoder loops -- monkeypatch a live model, NEVER edit core.

Measures an encoder's marginal contribution + its bottleneck without touching fusion.py / evaluate.py.
Everything here is reversible and lives in programs/ so the champion path stays byte-identical; consolidate
into core later. Minimal by design -- this is the fast-feedback measurement layer only.
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
def ablate_spacetime(model):
    """SPACETIME 'null spatial prior' control: zero the ABSOLUTE Earth4D position contribution at runtime
    (the projected pos_s / pos_t) so coordinates carry no spatial signal. Relative + value tokens remain.
    This is the S0 instrument -- st_gain = capability WITH this off vs ON. No core edit."""
    patched = []
    for attr in ("absolute_proj_s", "absolute_proj_t"):
        m = getattr(model, attr, None)
        if m is None:
            continue
        orig = m.forward
        m.forward = (lambda o: (lambda *a, **k: torch.zeros_like(o(*a, **k))))(orig)
        patched.append((m, orig))
    try:
        yield
    finally:
        for m, orig in patched:
            m.forward = orig


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


# capability -> spacetime-gain name (mirrors score.ST_GAIN); the S0 instrument built by monkeypatch, no core edit
ST_GAIN_MAP = {
    "B1_species_from_env_top10": "B1_species_spacetime_gain",
    "B6_family_from_env": "B6_family_spacetime_gain",
    "B34_lfmc_from_env": "B34_lfmc_spacetime_gain",
    "B42_mycorrhiza_from_env": "B42_mycorrhiza_spacetime_gain",
    "B51_pollinator_from_env_recall": "B51_pollinator_spacetime_gain",
    "B23_species_calibration_mrr": "B23_calibration_spacetime_gain",
}


def instrument(spacetime_gain: bool = False) -> None:
    """Wrap ``evaluate.evaluate_benchmarks`` (in-process monkeypatch) so a run emits the encoder feedback
    signal INTO its own log -- with NO edit to evaluate.py / fusion.py:
      - always: the biological bottleneck  ``[profile] refined_seed_norm=...`` (cheap: one graph call).
      - if spacetime_gain: a SECOND eval under ``ablate_spacetime`` and the ``*_spacetime_gain`` deltas
        (this is the S0 instrument -- it *creates* st_gain). Doubles eval time only when requested.
    Launch experiments through ``programs/run_experiment.py`` so this is installed before training."""
    from deepearth.autoresearch import evaluate as ev
    if getattr(ev, "_programs_instrumented", False):
        return
    orig = ev.evaluate_benchmarks

    def wrapped(model, source, device, *a, **k):
        for key, val in refinement_norm(model).items():
            print(f"[profile] {key}={val:.6g}", flush=True)
        raw = orig(model, source, device, *a, **k)
        if spacetime_gain:
            with torch.no_grad(), ablate_spacetime(model):
                abl = orig(model, source, device, *a, **k)
            for cap, gain in ST_GAIN_MAP.items():
                if cap in raw and cap in abl:
                    raw[gain] = max(0.0, float(raw[cap]) - float(abl[cap]))
                    print(f"  {gain:<34} {raw[gain]:.3f}  (spacetime-gain: WITH {raw[cap]:.3f} - WITHOUT {abl[cap]:.3f})", flush=True)
        return raw

    ev.evaluate_benchmarks = wrapped
    ev._programs_instrumented = True
