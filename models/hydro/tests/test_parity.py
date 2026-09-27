"""The torch solver against the v1 NumPy solver on the same domain, storm and settings."""

import numpy as np
import pytest

import reference_v1
from solver import Probes, SolverConfig, Surface, simulate

MM_HR = 1.0 / 1000.0 / 3600.0


def domain(nrows: int = 90, ncols: int = 100, dx: float = 5.0, seed: int = 23) -> dict:
    """Regional slope, a burned channel, micro-relief, spatial Horton fields, a soil store."""
    rng = np.random.default_rng(seed)
    rows = np.arange(nrows)[:, None] * dx
    cols = np.arange(ncols)[None, :] * dx
    z = 20.0 - 0.002 * rows - 0.0005 * cols + 0.05 * rng.standard_normal((nrows, ncols))
    z -= 1.5 * np.exp(-((cols - 0.6 * ncols * dx) / (3 * dx)) ** 2)
    f0 = rng.uniform(2.0, 8.0, (nrows, ncols)) * MM_HR
    fc = f0 * 0.4
    k = rng.uniform(1.0, 3.0, (nrows, ncols)) / 3600.0
    deficit = rng.uniform(0.01, 0.05, (nrows, ncols))
    h0 = np.where(z < 19.0, 0.02, 0.0)
    return {"z": z, "f0": f0, "fc": fc, "k": k, "max_deficit_m": deficit, "initial_h": h0}


@pytest.mark.parametrize("dtype, tol_h, tol_q", [("float32", 1e-5, 1e-4), ("float64", 1e-6, 1e-6)])
def test_matches_v1_state_and_substep_count(dtype, tol_h, tol_q):
    np_dtype = {"float32": np.float32, "float64": np.float64}[dtype]
    d = {k: v.astype(np_dtype) for k, v in domain().items()}
    rain = [40.0 * MM_HR] * 30 + [0.0] * 10
    gauge = (45, 50)
    v1 = reference_v1.simulate(
        reference_v1.Surface(**d), rain,
        reference_v1.SolverConfig(dx=5.0, dt_s=20.0, cfl_alpha=0.15, frame_interval_min=1e9),
        reference_v1.Probes(gauge_rc=gauge), verbose=False)
    v2 = simulate(
        Surface(**d), rain,
        SolverConfig(dx=5.0, dt_s=20.0, cfl_alpha=0.15, frame_interval_min=1e9, dtype=dtype,
                     device="cpu"),
        Probes(gauge_rc=gauge), verbose=False)
    dh = np.abs(v1.h_final.astype(np.float64) - v2.h_final).max()
    dhmax = np.abs(v1.h_max.astype(np.float64) - v2.h_max).max()
    dcum = np.abs(v1.cum_infil.astype(np.float64) - v2.cum_infil).max()
    dq = abs(v1.mass.outflow - v2.mass.outflow) / v1.mass.outflow
    dg = np.abs(v1.series["gauge_cms"] - v2.series["gauge_cms"]).max() / v1.series["gauge_cms"].max()
    print(f"\nparity {dtype}: substeps {v1.n_substeps} vs {v2.n_substeps}, max|dh| {dh:.3e} m, "
          f"max|dh_max| {dhmax:.3e} m, max|dcum| {dcum:.3e} m, outflow rel {dq:.3e}, gauge rel {dg:.3e}")
    assert v1.n_substeps == v2.n_substeps
    assert v1.h_max.max() > 0.2
    assert dh < tol_h and dhmax < tol_h and dcum < tol_h
    assert dq < tol_q and dg < tol_q
