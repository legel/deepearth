"""The landscape field (ranges/joint/field.py): Entropy3D ring harmonics of the raw 240 m field.

* orientation: on fields that rise to the east and to the north, the ring's first harmonic is a pure sine (east) and
  a pure cosine (north) term equal to the rise across the ring, on every level the ring reads, and the radial profile
  and higher orders vanish (rows count southward, angles clockwise from north);
* the radius gradient of the PyTorch reference equals a central finite difference of its sums;
* the fused CUDA kernel equals the PyTorch reference: sums (forward) and the gradient with respect to the ring radii,
  with missing cells, grid edges and radii between pyramid levels (needs a GPU);
* ring_features (every ring at once) equals the per-ring token construction it replaced;
* the module: zero output at the start (its last layer is zero), rows independent of the chunking, the gradient
  reaches the radii, sizes recovered from a state dict;
* the pyramid builder: int8 steps of 1/16 standard deviation, missing cells -128, level k the mean of the valid level-0
  cells beneath it.
"""
import json
import math

import numpy as np
import pytest
import torch

from ranges.joint import field as FL


def _pyramid(H=96, W=128, C=5, levels=4, missing=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    out, h, w = [], H, W
    for _ in range(levels):
        v = torch.randint(-127, 128, (h, w, C), generator=g, dtype=torch.int16)
        v[torch.rand(h, w, C, generator=g) < missing] = FL.MISSING
        out.append(v.to(torch.int8))
        h, w = (h + 1) // 2, (w + 1) // 2
    return out


def _positions(n, H, W, seed=1):
    g = torch.Generator().manual_seed(seed)
    rc = torch.rand(n, 2, generator=g) * torch.tensor([H + 8.0, W + 8.0]) - 4.0   # some near or off the edges
    return rc


def test_orientation_east_and_north_ramps():
    """Channel 0 rises to the east (increasing column), channel 1 to the north (decreasing row), one int8 step
    (1/16 SD) per cell: the first harmonic is sine (east) for one, cosine (north) for the other, slope x radius."""
    H, W, slope = 64, 64, FL.STEP
    levels = []
    for k in range(4):                                                   # each level samples the same planes
        h, w = H >> k, W >> k
        x = (torch.arange(w, dtype=torch.float64) + 0.5) * 2 ** k        # cell centres in level-0 units
        y = (torch.arange(h, dtype=torch.float64) + 0.5) * 2 ** k
        east = (slope * (x - 0.5))[None, :].expand(h, w)
        north = (slope * (H - 0.5 - y))[:, None].expand(h, w)
        levels.append(torch.round(torch.stack([east, north], -1) / FL.STEP).clamp(-127, 127).to(torch.int8))
    rc = torch.tensor([[32.0, 32.0], [30.5, 33.25]])
    radii = torch.tensor([6.0, 12.0])
    vc = FL.bilinear(levels[0], rc[:, 0], rc[:, 1])[0]
    S = FL.ring_sums_reference(levels, rc, radii, vc, angles=8, orders=3)
    n = S[..., -1]
    a0, a1, b1 = S[..., 0] / n, S[..., 1] * 2 / n, S[..., 4] * 2 / n
    for j, r in enumerate(radii.tolist()):
        assert torch.allclose(b1[:, j, 0], torch.full((2,), slope * r), atol=0.05), b1[:, j]   # east = sine
        assert torch.allclose(a1[:, j, 1], torch.full((2,), slope * r), atol=0.05), a1[:, j]   # north = cosine
        assert a1[:, j, 0].abs().max() < 0.05 and b1[:, j, 1].abs().max() < 0.05
        assert a0[:, j].abs().max() < 0.05                                                     # no radial term
        assert (S[:, j, :, [2, 3, 5, 6]].abs() * 2 / n[:, j, :, None]).max() < 0.05           # no higher orders


def test_reference_radius_gradient_is_the_derivative():
    levels = _pyramid(missing=0.0)
    rc = _positions(40, 96, 128).clamp(10, 80)
    radii = torch.tensor([1.3, 2.7, 5.5, 11.0], dtype=torch.float64)
    vc = FL.bilinear(levels[0], rc[:, 0], rc[:, 1])[0].double()
    w = torch.randn(40, 4, 5, 8, dtype=torch.float64)

    def total(r):
        return (FL.ring_sums_reference(levels, rc.double(), r, vc, 8, 3) * w).sum()
    r = radii.clone().requires_grad_(True)
    total(r).backward()
    eps = 1e-5
    for j in range(len(radii)):
        e = torch.zeros_like(radii)
        e[j] = eps
        fd = (total(radii + e) - total(radii - e)) / (2 * eps)
        assert torch.isclose(r.grad[j], fd, rtol=1e-4, atol=1e-6), (j, float(r.grad[j]), float(fd))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused kernel needs a CUDA GPU")
def test_kernel_equals_reference():
    dev = torch.device("cuda")
    levels = _pyramid(H=300, W=420, C=12, levels=7, missing=0.05)
    rc = _positions(3000, 300, 420)
    radii = torch.tensor([1.0, 1.7, 2.9, 4.0, 9.3, 16.0, 40.5, 128.0])   # on and between pyramid levels
    vc = FL.bilinear(levels[0], rc[:, 0], rc[:, 1])[0]
    g = torch.randn(3000, 8, 12, 8)
    r_ref = radii.clone().requires_grad_(True)
    S_ref = FL.ring_sums_reference(levels, rc, r_ref, vc, 8, 3)
    (S_ref * g).sum().backward()
    lev_d = [v.to(dev) for v in levels]
    r_k = radii.to(dev).requires_grad_(True)
    S_k = FL.RingSums.apply(r_k, rc.to(dev), vc.to(dev), lev_d, 8, 3)
    (S_k * g.to(dev)).sum().backward()
    S_k = S_k.detach().cpu()
    scale = S_ref.abs().max()
    assert (S_k[..., -1] == S_ref[..., -1].detach()).all()                       # valid counts exactly
    assert (S_k - S_ref.detach()).abs().max() <= 1e-4 * scale, float((S_k - S_ref.detach()).abs().max())
    gr_k, gr_ref = r_k.grad.cpu(), r_ref.grad
    assert torch.allclose(gr_k, gr_ref, rtol=2e-3, atol=2e-3 * gr_ref.abs().max()), (gr_k, gr_ref)


def _per_ring_tokens(f, S):
    """The per-ring token construction ring_features replaced (one ring_in call per ring)."""
    B, M = S.shape[0], f.M
    toks = []
    for j in range(S.shape[1]):
        Sj = S[:, j].permute(1, 0, 2)                                            # [C, B, 2M + 2]
        nn_ = Sj[..., -1]
        n = nn_.clamp(min=1)
        a0 = Sj[..., 0] / n
        ac = Sj[..., 1:M + 1] * (2 / n)[..., None]
        as_ = Sj[..., M + 1:2 * M + 1] * (2 / n)[..., None]
        mag = torch.sqrt(ac * ac + as_ * as_ + 1e-8)
        miss = 1 - (nn_ / f.A).mean(0)
        x = torch.cat([a0.T, ac.permute(1, 0, 2).reshape(B, -1), as_.permute(1, 0, 2).reshape(B, -1),
                       mag.permute(1, 0, 2).reshape(B, -1), miss[:, None]], 1)
        toks.append(f.ring_in(x))
    return torch.stack(toks, 1)


def test_ring_features_equal_the_per_ring_loop():
    torch.manual_seed(0)
    f = FL.HarmonicField(32, n_channels=12, d_c=64, angles=8, orders=3)
    B, R, C, M = 513, 8, 12, 3
    S = torch.randn(B, R, C, 2 * M + 2)
    S[..., -1] = torch.randint(0, f.A + 1, (B, R, C)).float()
    with torch.no_grad():
        new, ref = f.ring_in(f.ring_features(S)), _per_ring_tokens(f, S)
    assert (new - ref).abs().max() < 1e-6


def test_module_chunking_gradient_and_spec():
    torch.manual_seed(0)
    levels = _pyramid(C=5)
    f = FL.HarmonicField(16, n_channels=5, d_c=32, heads=4, layers=2, radii_cells=(1, 2, 4, 8, 16), angles=8,
                         orders=3).attach(FL.FieldPyramid.from_levels(levels, [f"c{i}" for i in range(5)]))
    rc = _positions(300, 96, 128)
    with torch.no_grad():
        assert f(rc).abs().max() == 0                                            # starts as no change
        torch.nn.init.normal_(f.out[-1].weight, std=0.1)
        a, b = f(rc, chunk=1000), f(rc, chunk=37)
    assert torch.allclose(a, b, atol=1e-5)
    f.train()
    (f(rc, chunk=64) ** 2).sum().backward()
    assert f.log_r.grad is not None and f.log_r.grad.abs().sum() > 0
    spec = FL.HarmonicField.spec_from_state({f"field.{k}": v for k, v in f.state_dict().items()})
    assert spec == dict(n_channels=5, d_c=32, heads=4, layers=2, radii_cells=spec["radii_cells"], angles=8, orders=3)
    assert np.allclose(spec["radii_cells"], [1, 2, 4, 8, 16], rtol=1e-6)


def test_build_pyramid(tmp_path):
    rng = np.random.default_rng(0)
    C, H, W = 3, 37, 50
    x = rng.normal(5, 2, (C, H, W)).astype(np.float32)
    x[1] = np.exp(x[1] / 4)                                                      # a precipitation-like channel
    x[:, :4, :6] = np.nan                                                        # a lake
    x.tofile(tmp_path / "stack.f32")
    (tmp_path / "stack.json").write_text(json.dumps({"variables": ["a", "wc_bio_12", "b"], "shape": [C, H, W],
                                                     "transform": [240, 0, 0, 0, -240, 0], "crs": "EPSG:5070"}))
    info = FL.build_pyramid(tmp_path / "stack.f32", ["b", "wc_bio_12"], tmp_path / "field", levels=3,
                            log_channels=("bio_12",), log=lambda m: None)
    assert info["log"] == ["wc_bio_12"] and [tuple(v) for v in info["shapes"]] == [(37, 50), (19, 25), (10, 13)]
    L0 = np.load(tmp_path / "field" / "field_L0.npy")
    z = np.log1p(np.clip(x[1], 0, None))
    z = (z - np.nanmean(z.astype(np.float64))) / (np.nanstd(z.astype(np.float64), ddof=1) + 1e-6)
    q = np.where(np.isfinite(z), np.clip(np.round(z * 16), -127, 127), -128).astype(np.int8)
    assert (np.abs(L0[1].astype(int) - q.astype(int)) <= 1).all() and (L0[1][:4, :6] == -128).all()
    L1 = np.load(tmp_path / "field" / "field_L1.npy")
    zb = (x[2] - np.nanmean(x[2].astype(np.float64))) / (np.nanstd(x[2].astype(np.float64), ddof=1) + 1e-6)
    blk = zb[:2, 6:8]                                                            # a 2 x 2 block, all valid
    assert abs(int(L1[0, 0, 3]) - round(float(blk.mean()) * 16)) <= 1
    assert L1[0, 0, 0] == -128 and L1[0, 1, 2] == -128                           # all four cells missing
    p = FL.FieldPyramid(tmp_path / "field")
    assert p.levels[0].shape == (37, 50, 2) and p.channels == ["b", "wc_bio_12"]
    assert math.isclose(p.meta["step"], 1 / 16)
