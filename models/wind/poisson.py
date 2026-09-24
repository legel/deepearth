"""Multigrid-preconditioned conjugate gradients for 7-point operators on the cell grid.

One kernel serves the projection (Poisson) and the implicit diffusion (Helmholtz). Both are
L p = ident p + sum_f c_f (p - p_nb), with p_nb = 0 across a boundary face: symmetric positive
semi-definite in flux form, so a coarse level is the same operator with its face coefficients
summed over the coarse face and divided by the coarsening factor across it.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F

Tensor = torch.Tensor
Factors = Tuple[int, int, int]


def _pool(a: Tensor, k: Factors) -> Tensor:
    """Sum over blocks of size `k`, batching any leading dimensions."""
    if k == (1, 1, 1):
        return a
    lead = a.shape[:-3]
    out = F.avg_pool3d(a.reshape(-1, 1, *a.shape[-3:]), k, stride=k, divisor_override=1)
    return out.reshape(*lead, *out.shape[-3:])


def _spread(a: Tensor, k: Factors, trilinear: bool) -> Tensor:
    """Prolongation by factors `k`: cell-centred trilinear, or piecewise constant."""
    if trilinear:
        return F.interpolate(a[None], scale_factor=tuple(float(f) for f in k), mode="trilinear",
                             align_corners=False)[0]
    for dim, f in zip((-3, -2, -1), k):
        if f > 1:
            a = a.repeat_interleave(f, dim=dim)
    return a


@dataclass
class Operator:
    """L p = ident p + sum over the six faces of c_f (p - p_nb).

    Attributes:
        ident: Diagonal term, (nz, ny, nx).
        ax: x-face coefficients including both boundary faces, (nz, ny, nx + 1).
        ay: y-face coefficients, (nz, ny + 1, nx).
        az: z-face coefficients, (nz + 1, ny, nx).
    """

    ident: Tensor
    ax: Tensor
    ay: Tensor
    az: Tensor
    diag: Tensor = field(init=False)

    def __post_init__(self):
        self.diag = (self.ident + self.ax[:, :, 1:] + self.ax[:, :, :-1]
                     + self.ay[:, 1:, :] + self.ay[:, :-1, :] + self.az[1:] + self.az[:-1])

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.ident.shape)

    def apply(self, p: Tensor) -> Tensor:
        """L p = diag p - sum over neighbours of c_f p_nb, for p of shape (C, nz, ny, nx)."""
        ax, ay, az = self.ax[:, :, 1:-1], self.ay[:, 1:-1, :], self.az[1:-1]
        out = self.diag * p
        out[..., 1:] -= ax * p[..., :-1]
        out[..., :-1] -= ax * p[..., 1:]
        out[..., 1:, :] -= ay * p[..., :-1, :]
        out[..., :-1, :] -= ay * p[..., 1:, :]
        out[..., 1:, :, :] -= az * p[..., :-1, :, :]
        out[..., :-1, :, :] -= az * p[..., 1:, :, :]
        return out

    def factors(self) -> Factors:
        """Coarsening factors: 2 where the level is even, and in z only while the vertical
        coupling is within a decade of the horizontal. Where the median level couples vertically
        more than twice as strongly as across, its cells are flatter than they are wide and
        point smoothing leaves the across-error alone, so z is coarsened alone until they are not."""
        nz, ny, nx = self.shape
        coupled = bool(self.az[1:-1].mean() >= 0.1 * self.ax[:, :, 1:-1].mean())
        fz = 2 if nz % 2 == 0 and nz >= 4 and coupled else 1
        flat = fz == 2 and bool(self.az[1:-1].mean(dim=(1, 2)).median()
                                > 2.0 * self.ax[:, :, 1:-1].mean(dim=(1, 2)).median())
        fx = 2 if nx % 2 == 0 and nx >= 4 and not flat else 1
        fy = 2 if ny % 2 == 0 and ny >= 4 and not flat else 1
        return fz, fy, fx

    def coarsen(self, k: Factors) -> "Operator":
        fz, fy, fx = k
        return Operator(
            ident=_pool(self.ident, k),
            ax=_pool(self.ax[:, :, ::fx], (fz, fy, 1)) / fx,
            ay=_pool(self.ay[:, ::fy, :], (fz, 1, fx)) / fy,
            az=_pool(self.az[::fz], (1, fy, fx)) / fz,
        )


@dataclass
class Level:
    op: Operator
    diag: Tensor
    red: Tensor
    factors: Optional[Factors]


def _dot(a: Tensor, b: Tensor) -> Tensor:
    return (a * b).sum(dim=(-3, -2, -1))


def _norm(a: Tensor) -> Tensor:
    return _dot(a, a).sqrt()


def cg(op: Operator, b: Tensor, x: Tensor, tol: float, max_iter: int,
       precond: Optional[Callable[[Tensor], Tensor]] = None) -> Tuple[Tensor, int, float]:
    """Conjugate gradients on every channel of `b` at once.

    Args:
        op: The operator.
        b: Right-hand sides, (C, nz, ny, nx).
        x: Initial guess, same shape.
        tol: Stop when every channel's residual is below `tol` times its right-hand side.
        max_iter: Iteration cap.
        precond: Optional preconditioner applied to the residual.

    Returns:
        (solution, iterations, worst relative residual).
    """
    view = (-1, 1, 1, 1)
    bnorm = _norm(b)
    r = b - op.apply(x)
    rel = torch.where(bnorm > 0, _norm(r) / bnorm, torch.zeros_like(bnorm))
    if bool((rel <= tol).all()):
        return x, 0, float(rel.max())
    z = precond(r) if precond else r
    p, rz = z.clone(), _dot(r, z)
    it = 0
    for it in range(1, max_iter + 1):
        Ap = op.apply(p)
        pAp = _dot(p, Ap)
        alpha = torch.where(pAp > 0, rz / pAp, torch.zeros_like(pAp))
        x = x + alpha.view(view) * p
        r = r - alpha.view(view) * Ap
        rel = torch.where(bnorm > 0, _norm(r) / bnorm, torch.zeros_like(bnorm))
        if bool((rel <= tol).all()):
            break
        z = precond(r) if precond else r
        rz_new = _dot(r, z)
        beta = torch.where(rz > 0, rz_new / rz, torch.zeros_like(rz))
        p, rz = z + beta.view(view) * p, rz_new
    return x, it, float(rel.max())


class Solver:
    """A multigrid hierarchy for one operator, reused across solves.

    Args:
        op: The finest operator.
        min_cells: Stop coarsening below this many cells.
        sweeps: Red-black Gauss-Seidel sweeps before and after each coarse correction.
        coarse_iter: Conjugate-gradient iterations on the coarsest level.
        trilinear: Trilinear prolongation; False is piecewise constant.
    """

    def __init__(self, op: Operator, min_cells: int = 512, sweeps: int = 2,
                 coarse_iter: int = 20, trilinear: bool = True):
        self.sweeps, self.coarse_iter, self.trilinear = sweeps, coarse_iter, trilinear
        self.levels: List[Level] = []
        while True:
            k = op.factors()
            last = k == (1, 1, 1) or op.ident.numel() <= min_cells or len(self.levels) >= 15
            self.levels.append(Level(op, self._safe_diag(op), self._parity(op), None if last else k))
            if last:
                break
            op = op.coarsen(k)

    @staticmethod
    def _safe_diag(op: Operator) -> Tensor:
        return torch.where(op.diag > 0, op.diag, torch.ones_like(op.diag))

    @staticmethod
    def _parity(op: Operator) -> Tensor:
        z, y, x = [torch.arange(n, device=op.ident.device) for n in op.shape]
        return (z[:, None, None] + y[None, :, None] + x[None, None, :]) % 2 == 0

    def _smooth(self, lvl: Level, b: Tensor, x: Tensor, reverse: bool) -> Tensor:
        """Red-black Gauss-Seidel; `reverse` orders black then red so a V-cycle is symmetric."""
        for _ in range(self.sweeps):
            for colour in ((~lvl.red, lvl.red) if reverse else (lvl.red, ~lvl.red)):
                x = x + torch.where(colour, (b - lvl.op.apply(x)) / lvl.diag, 0.0)
        return x

    def vcycle(self, b: Tensor, i: int = 0) -> Tensor:
        """One V-cycle from level `i` on right-hand side `b`, starting from zero."""
        lvl = self.levels[i]
        if lvl.factors is None:
            return cg(lvl.op, b, torch.zeros_like(b), 1e-8, self.coarse_iter)[0]
        x = self._smooth(lvl, b, torch.zeros_like(b), False)
        r = b - lvl.op.apply(x)
        x = x + _spread(self.vcycle(_pool(r, lvl.factors), i + 1), lvl.factors, self.trilinear)
        return self._smooth(lvl, b, x, True)

    def solve(self, b: Tensor, x0: Optional[Tensor] = None, tol: float = 1e-6,
              max_iter: int = 200) -> Tuple[Tensor, int, float]:
        """Solve L x = b to a relative residual `tol` on every channel.

        Returns:
            (solution, PCG iterations, worst relative residual).
        """
        x = torch.zeros_like(b) if x0 is None else x0.clone()
        return cg(self.levels[0].op, b, x, tol, max_iter, precond=self.vcycle)
