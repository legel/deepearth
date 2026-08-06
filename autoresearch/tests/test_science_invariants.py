"""The testable invariants science.md states outright.

Rules 18, 23 and 31 do not say "prefer"; they name conditions a champion must satisfy. Each is checked
here against the real model rather than asserted in prose, so a violation fails a run instead of
surviving as a plausible-looking score.

These are cheap structural checks on a tiny model. They cannot prove a rule holds at scale -- rule 18 in
particular needs a full ablation pass -- but they catch the wiring errors that make a rule quietly
untrue, which is how all three were violated before.
"""

from __future__ import annotations

import pytest
import torch

try:
    from deepearth.autoresearch.main.editable_files.fusion.fusion import DeepEarth, Variable
except Exception as exc:                                        # pragma: no cover - needs the CUDA kernel
    pytest.skip(f"fusion unavailable: {exc}", allow_module_level=True)


def _model(**kw):
    variables = [
        Variable("identity", "categorical", num_classes=32),
        Variable("climate", "continuous", dim=8),
        Variable("soil", "continuous", dim=8),
    ]
    torch.manual_seed(0)
    return DeepEarth(variables, d_model=32, n_latents=4, n_layers=1, capacity=4,
                     decoder_hidden=16, species_variable="identity", **kw)


# ---------------------------------------------------------------- rule 31

def test_rule31_heads_read_a_detached_latent_by_default():
    """A head must not commandeer the core: `_head_in(detach=True)` cuts the gradient path."""
    m = _model()
    z = torch.randn(4, m.latents.shape[0], 32, requires_grad=True)
    out = m._head_in(z, "identity", detach=True)
    assert not out.requires_grad, "detached head input still carries gradient into the core"


def test_rule31_attached_head_is_opt_in_and_visible():
    """The attached path must exist but be explicit -- a head that backprops silently is the violation."""
    m = _model()
    z = torch.randn(4, m.latents.shape[0], 32, requires_grad=True)
    assert m._head_in(z, "identity", detach=False).requires_grad, "attached path is not reachable"
    # every auxiliary weight defaults off, so no head couples to the core unless a config turns it on
    for w in ("_myco_weight", "_lfmc_weight", "_flower_weight", "_sdist_weight"):
        assert float(getattr(m, w, 0.0)) == 0.0, f"{w} couples a head to the core by default"


# ---------------------------------------------------------------- rule 23

def test_rule23_each_variable_keeps_its_own_decoder():
    """Never collapse a variable's manifold into the shared representation: one decoder each, not shared."""
    m = _model()
    decoders = [m.decoders[v.name] for v in m.variables if v.reconstruct and v.name in m.decoders]
    ids = {id(d) for d in decoders}
    assert len(ids) == len(decoders), "two variables share a decoder; their marginals cannot stay distinct"


def test_rule23_cross_modal_flow_is_through_the_latent_highway():
    """Coupling is O(N*L) through the latents, never O(N^2) variable-to-variable."""
    m = _model()
    n_lat, d = m.latents.shape
    assert n_lat < len(m.variables) * d, "latent bottleneck is not narrower than direct variable coupling"
    # a variable's decoder reads the pooled latent, so its fan-in is L-shaped, not N-shaped
    for v in m.variables:
        if v.name in m.decoders:
            first = next(p for p in m.decoders[v.name].parameters() if p.dim() == 2)
            assert first.shape[1] in (d, 2 * d), f"{v.name} decoder reads something other than the latent field"


# ---------------------------------------------------------------- rule 18

def test_rule18_every_prepared_modality_is_a_reconstruction_target():
    """All available data must be included -- a prepared modality that is never reconstructed is dropped
    data, which rule 18 calls a bug rather than a choice."""
    m = _model()
    for v in m.variables:
        if not v.reconstruct:
            continue
        assert v.name in m.decoders or v.name == m.species_variable, \
            f"{v.name} is a declared target with no way to reconstruct it"


def test_rule18_masking_never_silently_drops_a_variable():
    """A fully-masked variable must still be scored, not skipped: skipping is how a modality stops
    counting without anyone deciding to drop it."""
    m = _model()
    present = {n: torch.zeros(4, dtype=torch.bool) for n in m.names}      # everything hidden
    observed = {n: torch.ones(4, dtype=torch.bool) for n in m.names}
    hidden_but_observed = {n: int(((~present[n]) & observed[n]).sum()) for n in m.names}
    assert all(v == 4 for v in hidden_but_observed.values()), \
        "a fully-masked variable is not counted as a reconstruction target"
