"""The multigrid kernel: symmetry, convergence, and the coarsening rule."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from poisson import Operator, Solver, cg


def _operator(nz=8, ny=16, nx=16, seed=0, helmholtz=False) -> Operator:
    """A random-coefficient 7-point operator with a blocked pocket and Dirichlet sides."""
    rng = np.random.default_rng(seed)
    t = lambda a: torch.as_tensor(a, dtype=torch.float64)  # noqa: E731
    ax, ay, az = [t(rng.uniform(0.5, 2.0, s)) for s in
                  ((nz, ny, nx + 1), (nz, ny + 1, nx), (nz + 1, ny, nx))]
    ax[:, :, 0] = 0.0
    az[0] = 0.0
    ax[2:5, 3:6, 4:7] = 0.0  # a solid pocket: every face into it is blocked
    ident = t(rng.uniform(1.0, 3.0, (nz, ny, nx))) if helmholtz else torch.zeros(nz, ny, nx, dtype=torch.float64)
    return Operator(ident, ax, ay, az)


def test_operator_is_symmetric():
    op = _operator()
    rng = np.random.default_rng(1)
    x, y = [torch.as_tensor(rng.standard_normal((2, *op.shape))) for _ in range(2)]
    lhs, rhs = (op.apply(x) * y).sum(), (x * op.apply(y)).sum()
    assert float(lhs) == pytest.approx(float(rhs), rel=1e-12)


def test_operator_is_positive_on_nonconstant_fields():
    op = _operator()
    x = torch.as_tensor(np.random.default_rng(2).standard_normal((1, *op.shape)))
    assert float((op.apply(x) * x).sum()) > 0


@pytest.mark.parametrize("helmholtz", [False, True])
def test_pcg_reaches_the_requested_residual(helmholtz):
    op = _operator(helmholtz=helmholtz)
    x_true = torch.as_tensor(np.random.default_rng(3).standard_normal((3, *op.shape)))
    b = op.apply(x_true)
    x, it, rel = Solver(op).solve(b, tol=1e-8, max_iter=100)
    assert rel <= 1e-8 and it < 40
    norm = lambda a: torch.linalg.vector_norm(a, dim=(1, 2, 3))  # noqa: E731
    assert float((norm(b - op.apply(x)) / norm(b)).max()) <= 1e-8


def test_multigrid_beats_plain_conjugate_gradients():
    op = _operator(nz=16, ny=32, nx=32)
    b = torch.as_tensor(np.random.default_rng(4).standard_normal((1, *op.shape)))
    _, it_plain, _ = cg(op, b, torch.zeros_like(b), 1e-6, 500)
    _, it_mg, _ = Solver(op).solve(b, tol=1e-6, max_iter=500)
    assert it_mg < it_plain / 3, (it_mg, it_plain)


def test_apply_matches_the_flux_form():
    """diag p - sum c p_nb equals ident p + sum c (p - p_nb) with zero across boundary faces."""
    op = _operator(helmholtz=True)
    p = torch.as_tensor(np.random.default_rng(5).standard_normal((1, *op.shape)))
    ax, ay, az = op.ax, op.ay, op.az
    fx = ax[:, :, 1:-1] * (p[..., 1:] - p[..., :-1])
    fy = ay[:, 1:-1, :] * (p[..., 1:, :] - p[..., :-1, :])
    fz = az[1:-1] * (p[..., 1:, :, :] - p[..., :-1, :, :])
    flux = op.ident * p
    flux = flux - F.pad(fx, (0, 1)) + F.pad(fx, (1, 0))
    flux = flux - F.pad(fy, (0, 0, 0, 1)) + F.pad(fy, (0, 0, 1, 0))
    flux = flux - F.pad(fz, (0, 0, 0, 0, 0, 1)) + F.pad(fz, (0, 0, 0, 0, 1, 0))
    flux[..., 0] += ax[:, :, 0] * p[..., 0]
    flux[..., -1] += ax[:, :, -1] * p[..., -1]
    flux[..., 0, :] += ay[:, 0, :] * p[..., 0, :]
    flux[..., -1, :] += ay[:, -1, :] * p[..., -1, :]
    flux[..., 0, :, :] += az[0] * p[..., 0, :, :]
    flux[..., -1, :, :] += az[-1] * p[..., -1, :, :]
    assert torch.allclose(op.apply(p), flux, atol=1e-12)


def test_coarsening_sums_the_diagonal_and_halves_summed_face_coefficients():
    op = _operator()
    coarse = op.coarsen((2, 2, 2))
    assert coarse.shape == (4, 8, 8)
    assert float(coarse.ident.sum()) == pytest.approx(float(op.ident.sum()))
    assert float(coarse.ax[:, :, 0].sum()) == pytest.approx(float(op.ax[:, :, 0].sum()) / 2)
    assert float(coarse.ax[1, 2, 3]) == pytest.approx(float(op.ax[2:4, 4:6, 6].sum()) / 2)


def test_z_coarsening_stops_where_the_vertical_coupling_is_weak():
    strong, weak = _operator(), _operator()
    weak.az *= 1e-3
    assert strong.factors() == (2, 2, 2)
    assert weak.factors() == (1, 2, 2)


def test_a_zero_right_hand_side_returns_immediately():
    op = _operator()
    x, it, rel = Solver(op).solve(torch.zeros(1, *op.shape, dtype=torch.float64))
    assert it == 0 and rel == 0.0 and float(x.abs().max()) == 0.0


def test_flat_cells_coarsen_in_z_alone_until_they_are_not():
    flat = _operator()
    flat.az *= 4.0
    assert flat.factors() == (2, 1, 1)
    assert flat.coarsen((2, 1, 1)).factors() == (2, 2, 2)
