"""The hand-fused sub-step (`fused.Step`) against the compiled one it replaces: one sub-step from the same state to the
bit, field by field, and whole storms on terrain with pits, nodata brinks, edge inflow, a rim source and a split soil to
round-off and its growth (a storm's wetting fronts carry a last-bit difference forward)."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from inflow import Inflow
from solver import Kernel, Probes, SolverConfig, Surface, _build, _runtime, _set_forcing, _substep, simulate

MM_HR = 1.0 / 1000.0 / 3600.0
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused sub-step is CUDA's")
# the compiled soil's CUDA-graph tree captures an empty graph as it sets itself up, and says so
pytestmark = pytest.mark.filterwarnings("ignore:The CUDA Graph is empty:UserWarning")


def terrain(rows: int = 160, cols: int = 150, seed: int = 3) -> np.ndarray:
    """A rough southward slope with pits and a nodata bite out of the north-east corner."""
    rng = np.random.default_rng(seed)
    r = np.arange(rows, dtype=np.float64)[:, None]
    z = 30.0 - 0.02 * r + 0.3 * rng.standard_normal((rows, cols))
    z[:20, -25:] = np.nan
    return z.astype(np.float32)


def inputs(rim: bool):
    z = terrain()
    rows, cols = z.shape
    rng = np.random.default_rng(7)
    rain = [float(v) * MM_HR for v in rng.uniform(0.0, 90.0, 40)]
    inflow = Inflow.uniform(z.shape, {"west": 0.02, "north": 0.01}, 60.0 * len(rain))
    if rim:
        idx = np.flatnonzero(np.isfinite(z[-1]))[:20] + (rows - 1) * cols
        inflow = Inflow(inflow.times_s, inflow.west, inflow.east, inflow.north, inflow.south, rim_index=idx,
                        rim=np.full((len(inflow.times_s), len(idx)), 2e-4))
    surface = Surface(z=z, f0=2e-6, fc=4e-7, k=1e-3, smax_m=np.full(z.shape, 0.002, dtype=np.float32),
                      manning_n=(0.03 + 0.02 * rng.random(z.shape)).astype(np.float32))
    return surface, rain, inflow


def config(fused: bool, cfl_depth: str) -> SolverConfig:
    return SolverConfig(dx=0.5, dt_s=60.0, cfl_alpha=0.5, cfl_depth=cfl_depth, frame_interval_min=5.0, device="cuda",
                        compile=True, soil_dt_s=120.0, fused=fused)

def differ(a, b, keys) -> str:
    """Every named field of `a` and `b` that is not the same to the bit: how many cells, the largest gap, where."""
    said = []
    for k in keys:
        x, y = getattr(a, k), getattr(b, k)
        x, y = (torch.as_tensor(x), torch.as_tensor(y))
        diff = x != y
        if bool(diff.any()):
            at = torch.nonzero(diff)[:3].tolist()
            said.append(f"{k}: {int(diff.sum())} of {diff.numel()} differ, max {float((x - y).abs().max()):.3g}, "
                        f"first at {at}: {[float(x[tuple(p)]) for p in at]} against {[float(y[tuple(p)]) for p in at]}")
    return "; ".join(said)


@cuda
@pytest.mark.parametrize("rim", [True, False])
def test_one_fused_sub_step_is_the_compiled_one_to_the_bit(rim):
    """From one wet state, one sub-step each way: the clock, both discharges and the depth to the bit; the mass terms to
    round-off (their sums run in another order)."""
    import fused
    surface, rain, inflow = inputs(rim)
    cfg = config(True, "face")
    device, dtype, _, block, _ = _runtime(cfg, surface.z.size)
    grid, state, forcing = _build(surface, cfg, Probes(), device, dtype)
    kern = Kernel(dx=cfg.dx, dt_s=cfg.dt_s, alpha=cfg.cfl_alpha, cell_cfl=False, split=True)
    _set_forcing(forcing, 80 * MM_HR, 60.0, inflow, 0.0)
    rng = np.random.default_rng(11)
    state.h.copy_(torch.as_tensor(rng.uniform(0.0, 0.3, surface.z.shape), dtype=dtype, device=device) * (~grid.invalid))
    state.qx.copy_(torch.as_tensor(rng.normal(0.0, 0.01, tuple(state.qx.shape)), dtype=dtype, device=device))
    state.qy.copy_(torch.as_tensor(rng.normal(0.0, 0.01, tuple(state.qy.shape)), dtype=dtype, device=device))
    a, b = copy.deepcopy(state), copy.deepcopy(state)
    torch.compile(_substep, dynamic=False)(a, grid, forcing, kern)
    fused.Step(b, grid, forcing, kern, False)(b, grid, forcing, kern)
    torch.cuda.synchronize()
    said = differ(a, b, ("t", "qx", "qy", "h"))
    assert not said, said
    np.testing.assert_allclose(b.acc.cpu().numpy(), a.acc.cpu().numpy(), rtol=1e-12, atol=1e-15)


@cuda
@pytest.mark.parametrize("rim", [True, False])
def test_a_fused_storm_is_the_compiled_storm_to_the_bit(rim):
    """Forty intervals, graphed blocks and the split soil between them: every frame and field to the bit, the sub-steps
    the same, the mass terms to round-off."""
    surface, rain, inflow = inputs(rim)
    a = simulate(surface, rain, config(False, "face"), inflow=inflow, verbose=False)
    b = simulate(surface, rain, config(True, "face"), inflow=inflow, verbose=False)
    assert a.n_substeps == b.n_substeps
    said = differ(a, b, ("h_final", "h_max", "qx_final", "qy_final", "cum_infil"))
    assert not said, said
    assert all(np.array_equal(fa, fb) for fa, fb in zip(a.frames, b.frames))
    for k in ("outflow", "inflow", "created", "infiltrated", "abstracted", "stored"):
        assert getattr(b.mass, k) == pytest.approx(getattr(a.mass, k), rel=1e-9, abs=1e-12), k


def test_it_says_why_it_does_not_apply():
    pytest.importorskip("triton")
    from fused import usable
    grid = SimpleNamespace(gauge=None, ws_sx=None, z=torch.empty(160, 150))
    kern = SimpleNamespace(split=True, dx=0.5)
    gpu = torch.device("cuda")
    assert usable(grid, kern, torch.float32, gpu, 8) is None
    assert usable(grid, kern, torch.float32, torch.device("cpu"), 8) == "not on CUDA"
    assert usable(grid, kern, torch.float64, gpu, 8) == "float64 storm"
    assert usable(grid, SimpleNamespace(split=False, dx=0.5), torch.float32, gpu, 8) == "soil inside the sub-step"
    assert "power of two" in usable(grid, SimpleNamespace(split=True, dx=0.3), torch.float32, gpu, 8)
    assert "probe" in usable(SimpleNamespace(gauge=(1, 1), ws_sx=None, z=grid.z), kern, torch.float32, gpu, 8)
    assert "cells" in usable(SimpleNamespace(gauge=None, ws_sx=None, z=torch.empty(97, 83)), kern, torch.float32, gpu, 8)
    assert "cell CFL" in usable(grid, SimpleNamespace(split=True, dx=0.5, cell_cfl=True), torch.float32, gpu, 8)
