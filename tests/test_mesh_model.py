import pytest
import torch

try:
    from deepearth.core.fusion import DeepEarth, NestedProjection, signal_lens
except Exception as exc:  # pragma: no cover - requires the CUDA extension
    pytest.skip(f"mesh unavailable: {exc}", allow_module_level=True)


def test_modalities_write_to_distinct_lenses():
    assert signal_lens("climate") == "abiotic"
    assert signal_lens("vision_dino") == "visual"
    assert signal_lens("identity", "categorical") == "biological"
    assert signal_lens("phenology") == "ecological"


def test_nested_field_preserves_the_base_path():
    base = torch.nn.Linear(2, 4, bias=False)
    residual = torch.nn.Linear(3, 4, bias=False)
    projection = NestedProjection(base, residual, base_dim=2, levels=5)
    values = torch.randn(2, 5, 5)
    expected = base(values[..., :2]) + torch.sigmoid(projection.gate)[None, :, None] * residual(values[..., 2:])
    torch.testing.assert_close(projection(values), expected)


def test_public_model_interface_is_complete():
    for method in ("context", "encode", "infer", "reconstruction_loss"):
        assert callable(getattr(DeepEarth, method))
