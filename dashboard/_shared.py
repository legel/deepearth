"""Shared production-model loader for dashboard tools."""
import torch


def load_checkpoint(cache, checkpoint, device):
    from deepearth.core.fusion import build_model
    from deepearth.core.train import EXPERIMENT, load_data

    source, variables, always = load_data(cache, device)
    model = build_model(source, variables, always, device, EXPERIMENT)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model, source
