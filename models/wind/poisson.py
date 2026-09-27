"""Multigrid-preconditioned conjugate gradients for 7-point operators on the cell grid.

One kernel serves the projection (Poisson) and the implicit diffusion (Helmholtz). Both are
L p = ident p + sum_f c_f (p - p_nb), with p_nb = 0 across a boundary face: symmetric positive
semi-definite in flux form, so a coarse level is the same operator with its face coefficients
summed over the coarse face and divided by the coarsening factor across it.

The stencil and the smoother are single fused kernels on CUDA (`torch.compile`), and the V-cycle may run in a
lower precision than the Krylov iteration it preconditions (`Solver(precond_dtype=...)`): the conjugate gradients
keep their residual and search directions in the operator's own precision and take the flexible (Polak-Ribiere)
step, so the preconditioner's rounding changes the iteration count, never the tolerance reached.
"""

import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F

Tensor = torch.Tensor
Factors = Tuple[int, int, int]

COMPILE = os.environ.get("WIND_COMPILE", "1") == "1"
"""Fuse the stencil and the smoother on CUDA. WIND_COMPILE=0 runs them eagerly (identical arithmetic)."""


def _stencil(diag: Tensor, ax: Tensor, ay: Tensor, az: Tensor, p: Tensor) -> Tensor:
    """diag p - sum over neighbors of c_f p_nb; ax, ay, az are the interior faces only."""
    out = diag * p
    out = out - F.pad(ax * p[..., :-1], (1, 0)) - F.pad(ax * p[..., 1:], (0, 1))
    out = out - F.pad(ay * p[..., :-1, :], (0, 0, 1, 0)) - F.pad(ay * p[..., 1:, :], (0, 0, 0, 1))
    out = out - F.pad(az * p[..., :-1, :, :], (0, 0, 0, 0, 1, 0)) - F.pad(az * p[..., 1:, :, :], (0, 0, 0, 0, 0, 1))
    return out


def _relax(diag: Tensor, ax: Tensor, ay: Tensor, az: Tensor, safe: Tensor, color: Tensor, b: Tensor,
           x: Tensor) -> Tensor:
    """One color of a red-black Gauss-Seidel sweep."""
    return x + torch.where(color, (b - _stencil(diag, ax, ay, az, x)) / safe, 0.0)


def _stencil6(diag: Tensor, cw: Tensor, ce: Tensor, cs: Tensor, cn: Tensor, cd: Tensor, cu: Tensor,
              p: Tensor) -> Tensor:
    """diag p - sum over the six neighbors of c_nb p_nb, one coefficient array per neighbor (nonsymmetric)."""
    out = diag * p
    out = out - cw * F.pad(p[..., :-1], (1, 0)) - ce * F.pad(p[..., 1:], (0, 1))
    out = out - cs * F.pad(p[..., :-1, :], (0, 0, 1, 0)) - cn * F.pad(p[..., 1:, :], (0, 0, 0, 1))
    out = out - cd * F.pad(p[..., :-1, :, :], (0, 0, 0, 0, 1, 0)) - cu * F.pad(p[..., 1:, :, :], (0, 0, 0, 0, 0, 1))
    return out


def _relax6(diag: Tensor, cw: Tensor, ce: Tensor, cs: Tensor, cn: Tensor, cd: Tensor, cu: Tensor, safe: Tensor,
            color: Tensor, b: Tensor, x: Tensor) -> Tensor:
    return x + torch.where(color, (b - _stencil6(diag, cw, ce, cs, cn, cd, cu, x)) / safe, 0.0)


_KERNELS = {}


def _kernel(name: str, fn: Callable, device: torch.device) -> Callable:
    if not (COMPILE and device.type == "cuda"):
        return fn
    if name not in _KERNELS:
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 256)
        _KERNELS[name] = torch.compile(fn, dynamic=False)
    return _KERNELS[name]


def _pool(a: Tensor, k: Factors) -> Tensor:
    """Sum over blocks of size `k`, batching any leading dimensions."""
    if k == (1, 1, 1):
        return a
    lead = a.shape[:-3]
    out = F.avg_pool3d(a.reshape(-1, 1, *a.shape[-3:]), k, stride=k, divisor_override=1)
    return out.reshape(*lead, *out.shape[-3:])


def _spread(a: Tensor, k: Factors, trilinear: bool) -> Tensor:
    """Prolongation by factors `k`: cell-centered trilinear, or piecewise constant."""
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
        self.inner = (self.ax[:, :, 1:-1].contiguous(), self.ay[:, 1:-1, :].contiguous(), self.az[1:-1].contiguous())

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.ident.shape)

    def to(self, dtype: torch.dtype) -> "Operator":
        return self if dtype == self.ident.dtype else Operator(self.ident.to(dtype), self.ax.to(dtype),
                                                               self.ay.to(dtype), self.az.to(dtype))

    symmetric = True

    def apply(self, p: Tensor) -> Tensor:
        """L p = diag p - sum over neighbors of c_f p_nb, for p of shape (C, nz, ny, nx)."""
        return _kernel("stencil", _stencil, p.device)(self.diag, *self.inner, p)

    def relax(self, safe: Tensor, color: Tensor, b: Tensor, x: Tensor) -> Tensor:
        return _kernel("relax", _relax, b.device)(self.diag, *self.inner, safe, color, b, x)

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
class Convective(Operator):
    """M p = ident p + sum_f c_f (p - p_nb) + sum_f F_f p_upwind(f): implicit diffusion with first-order upwind
    transport by the face volume fluxes `fx`, `fy`, `fz` [m^3/s, positive along +x, +y, +z, boundary faces
    included]. An M-matrix for divergence-free fluxes; a coarse level pools the fluxes over each coarse face (sums,
    so the coarse fluxes stay divergence-free) and the conductances as `Operator` does."""

    fx: Tensor = None
    fy: Tensor = None
    fz: Tensor = None
    symmetric = False

    def __post_init__(self):
        pos, neg = (lambda f: f.clamp(min=0.0)), (lambda f: (-f).clamp(min=0.0))  # noqa: E731
        fx, fy, fz = self.fx, self.fy, self.fz
        self.diag = (self.ident + self.ax[:, :, 1:] + self.ax[:, :, :-1] + self.ay[:, 1:, :] + self.ay[:, :-1, :]
                     + self.az[1:] + self.az[:-1]
                     + pos(fx[:, :, 1:]) + neg(fx[:, :, :-1]) + pos(fy[:, 1:, :]) + neg(fy[:, :-1, :])
                     + pos(fz[1:]) + neg(fz[:-1]))
        # Neighbor coefficients; the outermost of each is multiplied by a zero pad (a boundary value's increment).
        self.coeffs = ((self.ax[:, :, :-1] + pos(fx[:, :, :-1])).contiguous(),
                       (self.ax[:, :, 1:] + neg(fx[:, :, 1:])).contiguous(),
                       (self.ay[:, :-1, :] + pos(fy[:, :-1, :])).contiguous(),
                       (self.ay[:, 1:, :] + neg(fy[:, 1:, :])).contiguous(),
                       (self.az[:-1] + pos(fz[:-1])).contiguous(),
                       (self.az[1:] + neg(fz[1:])).contiguous())

    def release(self) -> None:
        self.ax = self.ay = self.az = self.fx = self.fy = self.fz = None

    def to(self, dtype: torch.dtype) -> "Convective":
        if dtype == self.ident.dtype:
            return self
        return Convective(self.ident.to(dtype), self.ax.to(dtype), self.ay.to(dtype), self.az.to(dtype),
                          self.fx.to(dtype), self.fy.to(dtype), self.fz.to(dtype))

    def apply(self, p: Tensor) -> Tensor:
        return _kernel("stencil6", _stencil6, p.device)(self.diag, *self.coeffs, p)

    def relax(self, safe: Tensor, color: Tensor, b: Tensor, x: Tensor) -> Tensor:
        return _kernel("relax6", _relax6, b.device)(self.diag, *self.coeffs, safe, color, b, x)

    def coarsen(self, k: Factors) -> "Convective":
        fz_, fy_, fx_ = k
        base = Operator.coarsen(self, k)
        return Convective(base.ident, base.ax, base.ay, base.az,
                          fx=_pool(self.fx[:, :, ::fx_], (fz_, fy_, 1)),
                          fy=_pool(self.fy[:, ::fy_, :], (fz_, 1, fx_)),
                          fz=_pool(self.fz[::fz_], (1, fy_, fx_)))


def bicgstab(op: Operator, b: Tensor, x: Tensor, tol: float, max_iter: int,
             precond: Optional[Callable[[Tensor], Tensor]] = None) -> Tuple[Tensor, int, float]:
    """Right-preconditioned BiCGSTAB (van der Vorst 1992) on every channel of `b` at once, for a nonsymmetric `op`.

    Returns:
        (solution, iterations, worst relative residual).
    """
    view = (-1, 1, 1, 1)
    M = precond or (lambda v: v)  # noqa: E731
    bnorm = _norm(b)
    r = b - op.apply(x)
    rel = torch.where(bnorm > 0, _norm(r) / bnorm, torch.zeros_like(bnorm))
    if bool((rel <= tol).all()):
        return x, 0, float(rel.max())
    r0, p, v = r.clone(), torch.zeros_like(r), torch.zeros_like(r)
    one = torch.ones_like(bnorm)
    rho, alpha, omega = one, one, one
    safe = lambda n, d: torch.where(d != 0, n / torch.where(d != 0, d, one), torch.zeros_like(n))  # noqa: E731
    it = 0
    for it in range(1, max_iter + 1):
        rho_new = _dot(r0, r)
        beta = safe(rho_new, rho) * safe(alpha, omega)
        p = r + beta.view(view) * (p - omega.view(view) * v)
        ph = M(p)
        v = op.apply(ph)
        alpha = safe(rho_new, _dot(r0, v))
        s = r - alpha.view(view) * v
        rel = torch.where(bnorm > 0, _norm(s) / bnorm, torch.zeros_like(bnorm))
        if bool((rel <= tol).all()):
            x = x + alpha.view(view) * ph
            break
        sh = M(s)
        t = op.apply(sh)
        omega = safe(_dot(t, s), _dot(t, t))
        x = x + alpha.view(view) * ph + omega.view(view) * sh
        r = s - omega.view(view) * t
        rho = rho_new
        rel = torch.where(bnorm > 0, _norm(r) / bnorm, torch.zeros_like(bnorm))
        if bool((rel <= tol).all()):
            break
    return x, it, float(rel.max())


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
       precond: Optional[Callable[[Tensor], Tensor]] = None, flexible: bool = False) -> Tuple[Tensor, int, float]:
    """Conjugate gradients on every channel of `b` at once.

    Args:
        op: The operator.
        b: Right-hand sides, (C, nz, ny, nx).
        x: Initial guess, same shape.
        tol: Stop when every channel's residual is below `tol` times its right-hand side.
        max_iter: Iteration cap.
        precond: Optional preconditioner applied to the residual.
        flexible: The Polak-Ribiere step, for a preconditioner that is not exactly the same linear map every call
            (one rounded to a lower precision).

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
        z_old = z
        z = precond(r) if precond else r
        rz_new = _dot(r, z)
        num = rz_new - _dot(r, z_old) if flexible else rz_new
        beta = torch.where(rz > 0, num / rz, torch.zeros_like(rz))
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
        precond_dtype: The V-cycle's precision; None is the operator's own.
        factors: Coarsening factors per level from an earlier hierarchy of the same grid, reused so a rebuilt
            operator costs no host synchronisation; None derives them (`Operator.factors`).
    """

    def __init__(self, op: Operator, min_cells: int = 512, sweeps: int = 2,
                 coarse_iter: int = 20, trilinear: bool = True, precond_dtype: Optional[torch.dtype] = None,
                 factors: Optional[List[Optional[Factors]]] = None):
        self.sweeps, self.coarse_iter, self.trilinear = sweeps, coarse_iter, trilinear
        self.op, self.dtype = op, op.ident.dtype
        self.precond_dtype = precond_dtype or self.dtype
        self.levels: List[Level] = []
        level = op.to(self.precond_dtype)
        while True:
            if factors is not None:
                k = factors[len(self.levels)]
                last = k is None
            else:
                k = level.factors()
                last = k == (1, 1, 1) or level.ident.numel() <= min_cells or len(self.levels) >= 15
            self.levels.append(Level(level, self._safe_diag(level), self._parity(level), None if last else k))
            if last:
                break
            level = level.coarsen(k)
        for lvl in self.levels:                      # the stencils are built; the face arrays are not needed again
            getattr(lvl.op, "release", lambda: None)()

    @property
    def factor_list(self) -> List[Optional[Factors]]:
        return [lvl.factors for lvl in self.levels]

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
            for color in ((~lvl.red, lvl.red) if reverse else (lvl.red, ~lvl.red)):
                x = lvl.op.relax(lvl.diag, color, b, x)
        return x

    def vcycle(self, b: Tensor, i: int = 0) -> Tensor:
        """One V-cycle from level `i` on right-hand side `b`, starting from zero."""
        lvl = self.levels[i]
        if lvl.factors is None:
            krylov = cg if lvl.op.symmetric else bicgstab
            return krylov(lvl.op, b, torch.zeros_like(b), 1e-8, self.coarse_iter)[0]
        x = self._smooth(lvl, b, torch.zeros_like(b), False)
        r = b - lvl.op.apply(x)
        x = x + _spread(self.vcycle(_pool(r, lvl.factors), i + 1), lvl.factors, self.trilinear)
        return self._smooth(lvl, b, x, True)

    def precondition(self, r: Tensor) -> Tensor:
        return self.vcycle(r.to(self.precond_dtype)).to(self.dtype)

    def solve(self, b: Tensor, x0: Optional[Tensor] = None, tol: float = 1e-6,
              max_iter: int = 200) -> Tuple[Tensor, int, float]:
        """Solve L x = b to a relative residual `tol` on every channel.

        Returns:
            (solution, PCG iterations, worst relative residual).
        """
        x = torch.zeros_like(b) if x0 is None else x0.clone()
        mixed = self.precond_dtype != self.dtype
        pre = self.precondition if mixed else self.vcycle
        if not self.op.symmetric:
            return bicgstab(self.op, b, x, tol, max_iter, precond=pre)
        return cg(self.op, b, x, tol, max_iter, precond=pre, flexible=mixed)
