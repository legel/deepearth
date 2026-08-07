import pytest
import torch

try:
    from deepearth.core.fusion import DeepEarth, Variable
except Exception as exc:                                    # pragma: no cover - needs the CUDA kernel
    pytest.skip(f"fusion unavailable: {exc}", allow_module_level=True)


def _model(cal):
    variables = [Variable("identity", "categorical", num_classes=16),
                 Variable("climate", "continuous", dim=8)]
    torch.manual_seed(0)
    return DeepEarth(variables, d_model=32, n_latents=4, n_layers=1, capacity=4,
                     decoder_hidden=16, species_variable="identity", continuous_calibration=cal)


def test_off_allocates_nothing():
    m = _model(False)
    assert len(m.cal_gain) == 0 and len(m.cal_bias) == 0
    assert not any("cal_" in n for n, _ in m.named_parameters())


def test_on_does_not_disturb_the_rest_of_initialization():
    """ones/zeros draw no RNG, so every other parameter must match the flag-off model exactly."""
    a, b = _model(False), _model(True)
    pa = dict(a.named_parameters())
    for n, p in b.named_parameters():
        if n.startswith(("cal_gain", "cal_bias")):
            continue
        assert torch.equal(p, pa[n]), f"{n} differs: the flag re-rolled initialization"


def test_calibration_is_identity_until_trained():
    m = _model(True)
    x = torch.randn(4, 8)
    assert torch.equal(m._calibrated("climate", x), x)


def test_calibration_applies_once_trained():
    m = _model(True)
    with torch.no_grad():
        m.cal_gain["climate"].fill_(2.0); m.cal_bias["climate"].fill_(1.0)
    x = torch.randn(4, 8)
    assert torch.allclose(m._calibrated("climate", x), x * 2.0 + 1.0)


def test_categorical_heads_are_never_calibrated():
    m = _model(True)
    assert "identity" not in m.cal_gain
