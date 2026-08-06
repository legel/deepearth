import importlib.util
from pathlib import Path

import torch


_SPEC = importlib.util.spec_from_file_location(
    "resolution", Path(__file__).parents[1] / "encoders" / "spacetime" / "hashencoder" / "resolution.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
resolution_grad = _MODULE.resolution_grad

_LN2 = 0.6931471805599453


def _reference(per_level_scale, base_resolution, contrib):
    """The unguarded formula, as written before the domain clamp."""
    L, D = contrib.shape
    scale = torch.exp2(per_level_scale.view(L, D).float()) * base_resolution.view(1, D).float() - 1.0
    return (_LN2 * (scale + 1.0) / scale.clamp_min(1e-6) * contrib).to(per_level_scale.dtype)


def test_matches_unguarded_formula_on_the_geometric_ladder():
    base = torch.full((3,), 16.0)
    pls = torch.stack([i * torch.ones(3) for i in range(5)])          # exp2 ladder, scale >= 15
    contrib = torch.randn(5, 3)

    assert torch.allclose(resolution_grad(pls, base, contrib), _reference(pls, base, contrib), atol=1e-6)


def test_bounds_the_factor_when_resolution_collapses():
    base = torch.full((2,), 16.0)
    near_singular = float(torch.log2(torch.tensor((1.0 + 1e-4) / 16.0)))   # scale ~ 1e-4, the divergence
    pls = torch.stack([torch.full((2,), near_singular), torch.full((2,), -12.0)])   # and scale < 0
    contrib = torch.ones(2, 2)

    guarded = resolution_grad(pls, base, contrib)
    assert torch.isfinite(guarded).all()
    assert (guarded.abs() <= 2.0 * _LN2).all()
    assert _reference(pls, base, contrib).abs().max() > 1e3          # the unguarded factor explodes


def test_never_emits_non_finite_values():
    base = torch.full((2,), 16.0)
    pls = torch.tensor([[-4.0, 0.0], [40.0, -1e9], [float("inf"), float("nan")]])
    contrib = torch.ones(3, 2)

    assert torch.isfinite(resolution_grad(pls, base, contrib)).all()


def test_preserves_dtype_and_shape():
    base = torch.full((3,), 16.0)
    pls = torch.zeros(4, 3, dtype=torch.float16)
    contrib = torch.randn(4, 3)

    out = resolution_grad(pls, base, contrib)
    assert out.shape == (4, 3)
    assert out.dtype == torch.float16
