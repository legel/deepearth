import sys
import types

import torch
import torch.nn as nn

hashencoder = types.ModuleType("hashencoder")
hashgrid = types.ModuleType("hashencoder.hashgrid")
hashgrid.HashEncoder = object
hashencoder.hashgrid = hashgrid
sys.modules.setdefault("hashencoder", hashencoder)
sys.modules.setdefault("hashencoder.hashgrid", hashgrid)

from encoders.spacetime.earth4d import Earth4D


class _CoefficientField(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = offset

    def forward(self, xyz, size=1.0):
        return xyz[..., :2] + self.offset

    def precompute(self, xyz, size=1.0):
        self.cached = self(xyz, size)
        size_bytes = self.cached.numel() * self.cached.element_size()
        return {"total_bytes": size_bytes, "total_mb": size_bytes / 2**20}

    def forward_precomputed(self, indices):
        return self.cached[indices]


def _earth4d_stub(temporal_basis="polynomial"):
    field = Earth4D.__new__(Earth4D)
    nn.Module.__init__(field)
    field.temporal_basis = temporal_basis
    field.spatial_dim = 2
    field.temporal_levels = 1
    field.features_per_level = 2
    field.verbose = False
    field.xyz_encoder = _CoefficientField(-1.0)
    field.xyt_encoder = _CoefficientField(1.0)
    field.yzt_encoder = _CoefficientField(2.0)
    field.xzt_encoder = _CoefficientField(3.0)
    field._normalize_coords = lambda coords: coords
    return field


def test_precomputed_coefficients_match_direct_temporal_field():
    field = _earth4d_stub()
    coords = torch.tensor([[0.2, -0.4, 0.7, 0.25], [-0.1, 0.3, 0.6, 0.8]])
    spatial = coords[:, :2] - 1.0
    raw = torch.cat([
        spatial,
        field.xyt_encoder(coords[:, :3]),
        field.yzt_encoder(coords[:, :3]),
        field.xzt_encoder(coords[:, :3]),
    ], dim=-1)

    transformed = field.transform_precomputed(raw, coords)
    direct = torch.cat([spatial, field._encode_spatiotemporal(coords)], dim=-1)

    torch.testing.assert_close(transformed, direct)


def test_temporal_modes_remain_continuous_across_forecast_boundary():
    left = torch.tensor([[0.5 - 1e-5]])
    right = torch.tensor([[0.5 + 1e-5]])

    for before, after in zip(Earth4D._temporal_modes(left), Earth4D._temporal_modes(right)):
        assert torch.max(torch.abs(after - before)) < 1e-3


def test_hash_basis_keeps_precomputed_features_unchanged():
    field = _earth4d_stub("hash")
    flat = torch.randn(3, 8)
    coords = torch.randn(3, 4)

    assert field.transform_precomputed(flat, coords) is flat


def test_polynomial_precompute_matches_forward():
    field = _earth4d_stub()
    coords = torch.tensor([[0.2, -0.4, 0.7, 0.25], [-0.1, 0.3, 0.6, 0.8]])
    field.precompute(coords)

    direct = torch.cat([field._encode_spatial(coords[:, :3]), field._encode_spatiotemporal(coords)], dim=-1)
    cached = field.forward_precomputed(torch.arange(len(coords)))

    torch.testing.assert_close(cached, direct)
