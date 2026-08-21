import pytest
import torch
import torch.nn as nn

try:
    from deepearth.core.fusion import DeepEarth, Variable
except Exception as exc:                                    # pragma: no cover - needs the CUDA kernel
    pytest.skip(f"fusion unavailable: {exc}", allow_module_level=True)


def _model():
    torch.manual_seed(0)
    model = DeepEarth(
        [Variable("signal", "continuous", dim=4)],
        d_model=8,
        n_latents=2,
        n_layers=1,
        n_heads=2,
        capacity=2,
        rounds=2,
        diffusion=True,
        absolute_log2_hashmap_size=8,
        absolute_levels=2,
        relative_log2_hashmap_size=8,
    )
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    return model


def _inputs(model):
    batch = 3
    values = {"signal": torch.randn(batch, 4)}
    present = {"signal": torch.zeros(batch, dtype=torch.bool)}
    zeros = torch.zeros(batch, model.d_model)
    context = {
        "position_s": zeros,
        "position_t": zeros,
        "position": zeros,
        "tokens": zeros[:, None, :],
        "cls_tokens": None,
        "experience": None,
    }
    return values, present, context


def test_diffusion_noise_is_training_only():
    model = _model()
    values, present, context = _inputs(model)

    model.eval()
    torch.manual_seed(1)
    first = model.encode(values, present, context)
    torch.manual_seed(2)
    second = model.encode(values, present, context)
    assert torch.equal(first, second)

    model.train()
    torch.manual_seed(1)
    first = model.encode(values, present, context)
    torch.manual_seed(2)
    second = model.encode(values, present, context)
    assert not torch.equal(first, second)
