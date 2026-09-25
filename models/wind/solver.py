"""Mass-consistent wind over a parcel, in PyTorch.

Face velocities on a staggered grid are projected onto the divergence-free space by one Poisson
solve for a Lagrange multiplier: the minimal correction to the background profile that conserves
mass around terrain and buildings. Pseudo-time momentum iterations then add what the projection
alone has no term for: advection, mixing-length diffusion, canopy drag cd a |u| u, and a log-law
wall stress carrying each class's roughness length. Every implicit operator is a 7-point solve on
the multigrid kernel in `poisson`.

The finite-volume scheme (`SolverConfig.scheme = "fv"`, the default) steps the steady residual in
delta form, with MUSCL convection by the divergence-free face fluxes and the projection's pressure
carried between steps, so the field it converges to does not depend on the pseudo-time step. The
semi-Lagrangian scheme ("sl") is kept for comparison: its steady state does.
"""

import copy
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from domain import Scene
from forcing import LogProfile, wind_vector
from physics import KAPPA, NU_AIR

PROGRESS: Optional[Tuple[int, int]] = None
"""(this solve's index, the solves in the run), set by `cli` around each heading's solve: every PROGRESS_EVERY momentum
steps the run prints `PROGRESS wind k/n` over all its solves' steps, which a caller can read for its progress.
None prints nothing. It changes no number the solver computes."""

PROGRESS_EVERY = 10
from poisson import Convective, Operator, Solver as Poisson

# The k-l closure's constants, each from the literature and none fitted here.
KL_C_MU = 0.09          # Launder and Spalding (1974): nu_t = C_mu^(1/4) l sqrt(k), eps = C_mu^(3/4) k^(3/2) / l
KL_SIGMA_K = 1.0        # Launder and Spalding (1974): k diffuses with nu_t / sigma_k
KL_BETA_P = 1.0         # Katul et al. (2004): the share of the canopy's drag work that becomes wake turbulence
KL_BETA_D = 5.1         # Katul et al. (2004): the canopy's short-circuit of the cascade, beta_d c_d a |u| k
KL_D_OVER_H = 2.0 / 3.0  # displacement height d = 2 h_c / 3 (Raupach 1994); l = kappa (h_c - d) within the canopy
KL_WALL = 1.0e3         # the pseudo-time rate (per step) holding a wall cell at k = u*^2 / sqrt(C_mu) (Richards and Hoxey 1993)

Tensor = torch.Tensor
Faces = Tuple[Tensor, Tensor, Tensor]


@dataclass
class SolverConfig:
    """Numerical settings for one run.

    Attributes:
        steps: Pseudo-time momentum iterations after the projection; 0 is the projection alone. With
            `settle_tol`, the fewest taken.
        settle_tol: Step until the field near the ground has settled. The run is averaged over windows of
            `settle_every` steps (a canopy's pseudo-time state wobbles about its mean); the horizontal speed of each
            window mean's fluid cells 2 to 30 m above their surface is compared with the last window's, and the run
            ends once its median and p95 moved under this fraction and its RMS change under SETTLE_RMS times it, two
            windows running. The delivered field is the last window's mean, projected. None takes exactly `steps`.
        settle_every: Steps a settling window holds.
        max_steps: The most steps a settling run takes; it then ends unsettled and says so (`Result.settled`).
        cfl: Pseudo-time step as a multiple of dx / u_top.
        tol: Relative residual the projection is driven to.
        tol_momentum: Relative residual of the implicit momentum solve.
        max_iter: PCG iteration cap.
        tol_final: Relative residual of one more projection after the last momentum step, so the
            delivered field meets a divergence the relaxation does not need; None ends on the
            last step's projection.
        max_iter_final: PCG iteration cap of that projection.
        lateral: Side boundaries. "profile": the profile is prescribed on all four sides and
            displaced mass leaves through the top. "open": the profile is prescribed only on
            the sides the background enters through and the others let flow leave.
        device: torch device.
        dtype: torch floating type.
        verbose: Print progress every 50 steps.
        work_dtype: The momentum phase's precision (advection, mixing, the implicit solve); None is `dtype`.
        precond_dtype: The projection's V-cycle precision under its `dtype` conjugate gradients; None is `dtype`.
        warm_projection: Start each semi-Lagrangian projection from the last multiplier.
        scheme: "fv", the steady finite-volume momentum whose fixed point does not depend on `cfl`; "sl", the
            semi-Lagrangian pseudo-time step, whose steady state does.
        limiter: The finite volumes' face reconstruction (`LIMITERS`).
        relax_nu: Under-relaxation of the eddy viscosity between steps; 1 takes each step's own. Below 1 it damps
            the odd-even oscillation the mixing-length feedback drives at large steps, and leaves the fixed point alone.
        settle_rule: "tail" ends a settling run once the change still to come, estimated from the geometric decay of
            the last three window-to-window changes (`tail_bound`), is at most `settle_tol` for the median and p95 of
            every published level (`LEVELS_M`); "window" once the last window's own change is under it, two windows
            running. The eddy viscosity relaxes per step, so the field converges per step, not per unit of pseudo-time,
            and a per-window change understates what is left when the decay is slow.
        inflow: "canopy" prescribes on the sides the steady column over the site's mean canopy
            (`Model.equilibrium_column`), so the fetch the domain holds does not change the field; "log" the upwind
            log law, which the canopy keeps slowing downwind. A unit's boundary overrides both.
        anderson: Depth of Anderson acceleration of the finite volumes' outer fixed point (faces, multiplier and relaxed
            eddy viscosity); 0 is none. It changes how fast the steps arrive, not where.
        closure: The eddy viscosity. "mixing": the mixing length (kappa h)^2 |S|. "k-l": a transported turbulent
            kinetic energy k with a prescribed length (Katul et al. 2004), nu_t = C_mu^(1/4) l sqrt(k), whose canopy
            makes turbulence in the wakes of its leaves and loses it to the cascade (`KL_*`). Finite volumes only.
    """

    steps: int = 200
    settle_tol: Optional[float] = None
    settle_every: int = 20
    max_steps: Optional[int] = None
    cfl: float = 2.0
    tol: float = 1e-6
    tol_momentum: float = 1e-5
    max_iter: int = 200
    tol_final: Optional[float] = None
    max_iter_final: int = 2000
    lateral: str = "profile"
    device: str = "cpu"
    dtype: torch.dtype = torch.float64
    verbose: bool = False
    work_dtype: Optional[torch.dtype] = None
    precond_dtype: Optional[torch.dtype] = None
    warm_projection: bool = False
    scheme: str = "fv"
    limiter: str = "vanleer"
    relax_nu: float = 0.5
    settle_rule: str = "window"
    inflow: str = "log"
    anderson: int = 0
    anderson_store: Optional[str] = None
    inflow_ring_m: float = 0.0
    closure: str = "mixing"
    drive: str = "shear"

    def fast(self) -> "SolverConfig":
        """The production numerics: advection, mixing and the implicit momentum solve in float32, the projection's
        conjugate gradients in `dtype` preconditioned by a float32 V-cycle, each projection started from the last
        one's multiplier. The faces, the divergence and every projection's tolerance stay in `dtype`."""
        self.work_dtype, self.precond_dtype, self.warm_projection = torch.float32, torch.float32, True
        return self


@dataclass
class Result:
    """Everything one run produces.

    Attributes:
        velocity: Cell-centered (u, v, w) [m/s], (3, nz, ny, nx).
        vorticity: Cell-centered curl [1/s], (3, nz, ny, nx).
        faces: Staggered (ux, uy, uz) [m/s] on x-, y- and z-faces; the divergence-free field.
        divergence_rel: Face divergence after the last projection over that before it.
        divergence_max: Largest face divergence over u_top dx^2.
        divergence_max_1_s: Largest cell divergence per unit volume [1/s].
        flux_in_m3_s: Volume flux entering the domain.
        flux_out_m3_s: Volume flux leaving it.
        poisson_iterations: PCG iterations of every projection.
        change: Largest velocity change per momentum step over u_top.
        wall_s: Wall time.
        cells: Grid cells.
        steps: Momentum steps taken.
    """

    velocity: np.ndarray
    vorticity: np.ndarray
    faces: Tuple[np.ndarray, np.ndarray, np.ndarray]
    divergence_rel: float
    divergence_max: float
    divergence_max_1_s: float
    flux_in_m3_s: float
    flux_out_m3_s: float
    poisson_iterations: List[int]
    change: List[float]
    wall_s: float
    cells: int
    steps: int
    settled: Optional[bool] = None
    settle: Optional[List[dict]] = None
    state: Optional[dict] = None
    """The finite-volume state to continue from (`solve(initial=...)`): faces and pressure [m^2/s^2], and k under k-l."""
    tke: Optional[np.ndarray] = None
    """Turbulent kinetic energy k [m^2/s^2], (nz, ny, nx), under the k-l closure."""

    @property
    def speed(self) -> np.ndarray:
        return np.linalg.norm(self.velocity, axis=0)

    @property
    def flux_balance(self) -> float:
        """|in - out| / in."""
        return abs(self.flux_in_m3_s - self.flux_out_m3_s) / self.flux_in_m3_s


@dataclass
class Boundary:
    """Velocities prescribed on a geographic unit's four sides and top, taken from a coarser solve of
    the whole domain (one-way nesting): what the unit's sides would see in one domain, where the
    single domain's own sides see the upwind profile.

    Attributes:
        west, east: (3, nz, ny) [m/s] on the unit's west and east faces; None where the side is the whole
            domain's own, which sees the profile as a single solve's side does.
        south, north: (3, nz, nx) on its south and north faces, likewise.
        top: (3, ny, nx) above its top; None: the profile, as above a single solve.
        initial: (3, nz, ny, nx) the coarse field on the unit's cells, the momentum's starting point;
            None starts from the profile.
    """

    west: Optional[np.ndarray] = None
    east: Optional[np.ndarray] = None
    south: Optional[np.ndarray] = None
    north: Optional[np.ndarray] = None
    top: Optional[np.ndarray] = None
    initial: Optional[np.ndarray] = None


def json_row(row: dict) -> str:
    """A settle check on one line."""
    return " ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items())


def tail_bound(h: List[float]) -> float:
    """The change still to come of a converging sequence, relative to its last value, from its last four values.

    Successive changes d1, d2, d3 decaying by a ratio r sum to d3 r / (1 - r) beyond the last (Aitken). r is the
    larger of the two measured ratios; a ratio of 1 or more is no decay yet (infinite); changes that alternate in sign
    bound the limit within the last change.
    """
    if len(h) < 4:
        return float("inf")
    d1, d2, d3 = h[-3] - h[-4], h[-2] - h[-3], h[-1] - h[-2]
    if d3 == 0.0:
        return 0.0
    ratio = [b / a if a != 0.0 else float("inf") for a, b in ((d1, d2), (d2, d3))]
    r = max(ratio)
    if r >= 1.0:
        return float("inf")
    rest = abs(d3) if r <= 0.0 else abs(d3) * r / (1.0 - r)
    return rest / max(abs(h[-1]), 1e-12)


class _Anderson:
    """Anderson acceleration (type II) of a fixed-point map x -> G(x) over tuples of tensors.

    The next state is G(x_k) - dG gamma, gamma minimizing |f_k - dF gamma| over the last `depth` changes of the
    residual f = G(x) - x (its first `residual_parts` tensors: the faces). Every column of dG is a difference of two
    states, so a mix keeps the prescribed boundary values, and of divergence-free states is divergence-free. The
    history is kept in float32 on `store` (the host, when the GPU has no room for it), its Gram matrix updated a column
    at a time; it restarts when the residual grows tenfold over its smallest.
    """

    def __init__(self, depth: int, store: Optional[str] = None):
        self.depth, self.store = depth, store
        self.dF, self.dG, self.gram, self.f, self.g, self.best = [], [], [], None, None, float("inf")

    @staticmethod
    def _dot(a, b) -> float:
        return float(sum((x * y).sum(dtype=torch.float64) for x, y in zip(a, b)))

    def _keep(self, t: Tensor) -> Tensor:
        t = t.to(torch.float32)
        return t.to(self.store) if self.store else t

    def step(self, x, g, residual_parts: int):
        f = tuple(self._keep(b - a) for a, b in zip(x[:residual_parts], g[:residual_parts]))
        norm = self._dot(f, f) ** 0.5
        if norm > 10.0 * self.best:
            self.dF, self.dG, self.gram = [], [], []
        self.best = min(self.best, norm)
        if self.f is not None:
            col = tuple(a - b for a, b in zip(f, self.f))
            self.dF.append(col)
            self.dG.append(tuple(self._keep(a) - b for a, b in zip(g, self.g)))
            row = [self._dot(c, col) for c in self.dF]
            for r, v in zip(self.gram, row[:-1]):
                r.append(v)
            self.gram.append(row)
            if len(self.dF) > self.depth:
                self.dF.pop(0), self.dG.pop(0)
                self.gram = [r[1:] for r in self.gram[1:]]
        self.f, self.g = f, tuple(self._keep(a) for a in g)
        if not self.dF:
            return g
        k = len(self.dF)
        a = np.array(self.gram)
        b = np.array([self._dot(self.dF[i], f) for i in range(k)])
        gamma = np.linalg.solve(a + 1e-10 * np.trace(a) * np.eye(k) + 1e-300 * np.eye(k), b)
        out = []
        for i, gi in enumerate(g):
            mix = gi.clone()
            for j in range(k):
                mix -= float(gamma[j]) * self.dG[j][i].to(device=gi.device, dtype=gi.dtype)
            out.append(mix)
        return tuple(out)


def _tridiagonal(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a tridiagonal system (Thomas); `lower` and `upper` are the off-diagonals, one shorter than `diag`."""
    n = diag.size
    c, d = np.zeros(n), np.zeros(n)
    c[0], d[0] = (upper[0] / diag[0] if n > 1 else 0.0), rhs[0] / diag[0]
    for i in range(1, n):
        m = diag[i] - lower[i - 1] * c[i - 1]
        c[i] = upper[i] / m if i < n - 1 else 0.0
        d[i] = (rhs[i] - lower[i - 1] * d[i - 1]) / m
    x = np.zeros(n)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def _logmean(a: Tensor, b: Tensor) -> Tensor:
    """Logarithmic mean of two positive tensors."""
    return torch.where(a == b, a, (b - a) / torch.log(b / a))


LIMITERS = {
    "vanleer": lambda r: (r + r.abs()) / (1.0 + r.abs()),
    "vanalbada": lambda r: torch.where(r > 0, (r * r + r) / (r * r + 1.0), torch.zeros_like(r)),
    "upwind": lambda r: torch.zeros_like(r),
}
"""Flux limiters psi(r) of the face reconstruction: van Leer (1974), van Albada et al. (1982), and first-order upwind."""


def _muscl(far: Tensor, up: Tensor, down: Tensor, limiter: str = "vanleer") -> Tensor:
    """The face value between `up` (upwind) and `down`: up + psi(r) (down - up) / 2, r = (up - far) / (down - up)."""
    jump = down - up
    r = (up - far) / torch.where(jump == 0, torch.full_like(jump, 1e-30), jump)
    return up + 0.5 * LIMITERS[limiter](r) * jump


def _wall(delta: Tensor, z0: Tensor, size: Tensor) -> Tensor:
    """Log-law stress coefficient per unit volume [1/m]: (kappa / ln(delta / z0))^2 / size."""
    z0 = torch.minimum(z0, delta * math.exp(-1.0))
    return (KAPPA / torch.log(delta / z0)) ** 2 / size


def _kl_length(hf, hc):
    """The k-l closure's length: kappa (h_c - d) within a canopy of height h_c, kappa (h - d) above it (d = 2 h_c / 3),
    never more than kappa h near the ground; kappa h without a canopy."""
    d = KL_D_OVER_H * hc
    if torch.is_tensor(hf):
        return torch.minimum(KAPPA * hf, torch.maximum(KAPPA * (hc - d), KAPPA * (hf - d)))
    return np.minimum(KAPPA * hf, np.maximum(KAPPA * (hc - d), KAPPA * (hf - d)))


class Model:
    """A scene and its forcing, discretized and ready to run.

    Args:
        scene: Geometry, drag and roughness.
        profile: Upwind log profile.
        direction_deg: Direction the wind blows from, clockwise from north.
        cfg: Numerical settings.
    """

    def __init__(self, scene: Scene, profile: LogProfile, direction_deg: float,
                 cfg: SolverConfig, boundary: Optional[Boundary] = None):
        g, self.cfg = scene.grid, cfg
        kw = dict(device=cfg.device, dtype=cfg.dtype)
        t = lambda a: torch.as_tensor(np.asarray(a), **kw)  # noqa: E731
        assert boundary is None or cfg.lateral == "profile", "a unit's boundary is prescribed on every side"
        self.initial = None if boundary is None or boundary.initial is None else t(boundary.initial)
        self.nz, self.ny, self.nx = g.shape
        self.dx = float(g.dx)
        self.zc, self.zf, self.dz = t(g.zc), t(g.zf), t(g.dz)
        self.xc, self.yc = t(g.xc), t(g.yc)
        self.dzc = self.zc[1:] - self.zc[:-1]
        self.d_top = float(g.zf[-1] - g.zc[-1])
        self.vol = (self.dx * self.dx * self.dz)[:, None, None]
        self.area_x = (self.dx * self.dz)[:, None, None]
        self.area_z = self.dx * self.dx

        self.solid = torch.as_tensor(scene.solid, device=cfg.device)
        fluid = ~self.solid
        self.sink = t(scene.sink)
        z0 = t(scene.z0)[None].expand(self.nz, -1, -1)
        self.open_x = fluid[..., :-1] & fluid[..., 1:]
        self.open_y = fluid[:, :-1, :] & fluid[:, 1:, :]
        self.open_z = fluid[:-1] & fluid[1:]
        self.edge = [fluid[..., 0], fluid[..., -1], fluid[:, 0, :], fluid[:, -1, :], fluid[-1]]

        ex, ey = wind_vector(1.0, direction_deg)
        assert cfg.lateral in ("open", "profile"), f"unknown lateral policy {cfg.lateral!r}"
        self.inflow = ([ex > 1e-12, ex < -1e-12, ey > 1e-12, ey < -1e-12]
                       if cfg.lateral == "open" else [True] * 4)
        self.u0 = t(profile.speed(g.zc))[None] * t([ex, ey, 0.0])[:, None]
        self.u_top = float(profile.speed(np.array([g.top]))[0])
        self.u0_top = t([ex, ey, 0.0]) * self.u_top
        self.dt = cfg.cfl * self.dx / self.u_top
        self.bc = None
        if boundary is not None:
            side = lambda a, n: self.u0[:, :, None].expand(3, self.nz, n).clone() if a is None else t(a)  # noqa: E731
            self.bc = {"west": side(boundary.west, self.ny), "east": side(boundary.east, self.ny),
                       "south": side(boundary.south, self.nx), "north": side(boundary.north, self.nx),
                       "top": (self.u0_top[:, None, None].expand(3, self.ny, self.nx).clone() if boundary.top is None
                               else t(boundary.top))}

        below = torch.cummax(torch.where(self.solid, self.zf[1:, None, None].expand_as(self.sink),
                                         torch.zeros_like(self.sink)), dim=0).values
        self.height = torch.clamp(self.zc[:, None, None] - below, min=1e-3 * self.dx)
        self.wall = self._wall_coefficient(z0)
        self.column = None
        self.k, self.k_bc = None, None
        assert cfg.closure in ("mixing", "k-l"), f"unknown closure {cfg.closure!r}"
        if cfg.closure == "k-l":
            assert cfg.scheme == "fv" and not cfg.anderson, "the k-l closure steps with the finite volumes, alone"
            assert boundary is None, "a unit's boundary carries no k yet: the k-l closure runs a whole domain"
            self._kl_geometry(z0)
            self.k_log = profile.u_star ** 2 / math.sqrt(KL_C_MU)     # the log layer's k (Richards and Hoxey 1993)
            self.k_floor = 1e-6 * self.u_top ** 2
        self.body_vec = None
        assert cfg.drive in ("shear", "pressure"), f"unknown drive {cfg.drive!r}"
        if cfg.inflow == "canopy" and boundary is None:
            self._equilibrium_inflow(z0, ex, ey)
        if cfg.drive == "pressure":
            assert cfg.closure == "k-l" and self.column is not None, "the pressure drive runs k-l over the canopy column"
            self.body_vec = torch.tensor([ex, ey, 0.0], **kw) * float(self.column["body"])
        if cfg.closure == "k-l" and self.k is None:
            self.k = torch.where(self.solid, torch.zeros_like(self.sink), torch.full_like(self.sink, self.k_log))
            self.k_bc = {"west": self.k[..., 0].clone(), "east": self.k[..., -1].clone(),
                         "south": self.k[:, 0, :].clone(), "north": self.k[:, -1, :].clone()}
        self.poisson = Poisson(self._projection_operator(), precond_dtype=cfg.precond_dtype)
        self._lam: Optional[Tensor] = None
        self._momentum_factors = None
        self._work = self if (cfg.work_dtype or cfg.dtype) == cfg.dtype else self._copy(cfg.work_dtype)

    def _copy(self, dtype: torch.dtype) -> "Model":
        """This model with every floating tensor (and a unit's boundary) in `dtype`, for the momentum phase."""
        w = copy.copy(self)
        for k, v in vars(self).items():
            if torch.is_tensor(v) and v.is_floating_point():
                setattr(w, k, v.to(dtype))
        if self.bc is not None:
            w.bc = {k: v.to(dtype) for k, v in self.bc.items()}
        if self.k_bc is not None:
            w.k_bc = {k: v.to(dtype) for k, v in self.k_bc.items()}
        w.poisson, w._work = None, w
        return w

    # ── Inflow in equilibrium with the canopy ────────────────────────────────────────────

    COLUMN_TOL = 1e-10
    """Relative change of the precursor column's speed at which its iteration stops."""

    def equilibrium_column(self, z0: Tensor, iterations: int = 5000, where: Optional[Tensor] = None) -> Dict[str, np.ndarray]:
        """The steady, horizontally uniform wind over this site's mean canopy: the precursor of the inflow.

        The model's own vertical balance on its own levels, with nothing varying across: mixing-length diffusion
        d/dh(nu dU/dh) = (s + w) U^2, where s is the canopy drag density averaged over the fluid cells at each height
        above their ground, w the ground's log-law stress in the first cell (the domain's mean z0), and the top held at
        the profile's speed as the 3D model holds it. A log-law inflow is not a steady state of that canopy, so the
        canopy slows it for hundreds of meters downwind; this column is, so the fetch inside the domain does not
        change the field.

        Returns:
            {"h": cell-center heights above ground [m], "u": speed there [m/s], "h_top", "u_top"}.
        """
        zf = self.zf.double().cpu().numpy()
        h = self.zc.double().cpu().numpy() - zf[0]
        dz = np.diff(zf)
        dzc = np.diff(h)
        n = h.size
        fluid = ~self.solid if where is None else (~self.solid) & where[None]
        cell = torch.bucketize(self.height, self.zf[1:] - self.zf[0]).clamp(max=n - 1)
        num = torch.zeros(n, dtype=torch.float64, device=self.sink.device)
        cnt = torch.zeros(n, dtype=torch.float64, device=self.sink.device)
        num.index_add_(0, cell[fluid], self.sink[fluid].double())
        cnt.index_add_(0, cell[fluid], torch.ones_like(self.sink[fluid], dtype=torch.float64))
        s = (num / cnt.clamp(min=1)).cpu().numpy()
        z0m = float(z0[0][fluid[0]].double().mean()) if bool(fluid[0].any()) else float(z0.double().mean())
        wall0 = float(_wall(torch.tensor(dz[0] / 2), torch.tensor(z0m), torch.tensor(dz[0])))
        drag = s.copy()
        drag[0] += wall0
        h_top, u_top = h[-1] + self.d_top, self.u_top
        lm = lambda a, b: np.where(np.isclose(a, b), a, (b - a) / np.log(b / a))  # noqa: E731
        u = np.maximum(u_top * np.log(np.maximum(h / z0m, 1.0 + 1e-9)) / np.log(h_top / z0m), 1e-3 * u_top)
        if self.cfg.closure == "k-l":
            hc = float(self.canopy_height[fluid[0]].double().mean()) if bool(fluid[0].any()) else 0.0
            hf = np.concatenate([lm(h[:-1], h[1:]), lm(h[-1:], np.array([h_top]))])
            return self._kl_column(h, dz, dzc, hf, s, drag, z0m, hc, u, iterations)
        l_int = (KAPPA * lm(h[:-1], h[1:])) ** 2
        l_top = (KAPPA * lm(h[-1:], np.array([h_top]))) ** 2
        nu_int = l_int * np.abs(np.diff(u)) / dzc + NU_AIR
        nu_top = l_top * abs(u_top - u[-1]) / self.d_top + NU_AIR
        for it in range(iterations):
            nu_int = 0.5 * nu_int + 0.5 * (l_int * np.abs(np.diff(u)) / dzc + NU_AIR)
            nu_top = 0.5 * nu_top + 0.5 * (l_top * abs(u_top - u[-1]) / self.d_top + NU_AIR)
            c_lo = np.concatenate([[0.0], nu_int / dzc])          # conductance to the cell below
            c_hi = np.concatenate([nu_int / dzc, nu_top / self.d_top])
            diag = c_lo + c_hi + dz * drag * np.abs(u)
            rhs = np.zeros(n)
            rhs[-1] = c_hi[-1] * u_top
            new = _tridiagonal(-c_lo[1:], diag, -c_hi[:-1], rhs)
            done = np.max(np.abs(new - u)) <= self.COLUMN_TOL * u_top
            u = new
            if done:
                break
        return {"h": h, "u": u, "h_top": h_top, "u_top": u_top, "iterations": it + 1, "drag": s, "z0": z0m}

    def _kl_column(self, h: np.ndarray, dz: np.ndarray, dzc: np.ndarray, hf: np.ndarray, s: np.ndarray,
                   drag: np.ndarray, z0m: float, hc: float, u: np.ndarray, iterations: int) -> Dict[str, np.ndarray]:
        """The precursor column under the k-l closure: U from d/dh(nu dU/dh) = (s + w) U^2 with nu = C_mu^(1/4) l sqrt(k)
        on each face, and k from d/dh(nu / sigma_k dk/dh) + nu (dU/dh)^2 - eps + s (beta_p U^3 - beta_d U k) = 0, the
        ground cell held at the log law's k and no k through the top, as in the 3D model. `hf`: the heights of the
        faces above each cell (the log means the mixing length uses), `hc` the site's mean canopy height."""
        n, u_top, d_top = h.size, self.u_top, self.d_top
        h_top = h[-1] + d_top
        ell = _kl_length(hf, hc)                                      # on the faces above cells 0..n-1
        ell_inv = 0.5 * (1.0 / np.concatenate([ell[:1], ell[:-1]]) + 1.0 / ell)
        delta = dz[0] / 2
        fr0 = (KAPPA / math.log(delta / min(z0m, delta * math.exp(-1.0)))) ** 2
        ust0 = KAPPA * u_top / math.log(h_top / z0m)
        k = np.full(n, ust0 ** 2 / math.sqrt(KL_C_MU))
        pressure = self.cfg.drive == "pressure"
        body = ust0 ** 2 / h_top if pressure else 0.0     # -dp/dx / rho: the column's stress u*^2 over its depth
        nu = None
        for it in range(iterations):
            kf = np.concatenate([0.5 * (k[:-1] + k[1:]), k[-1:]])
            new = KL_C_MU ** 0.25 * ell * np.sqrt(kf) + NU_AIR
            nu = new if nu is None else 0.5 * nu + 0.5 * new
            c_lo = np.concatenate([[0.0], nu[:-1] / dzc])
            c_hi = np.concatenate([nu[:-1] / dzc, [0.0] if pressure else nu[-1:] / d_top])   # pressure: no stress on top
            rhs = dz * body
            rhs[-1] += c_hi[-1] * u_top
            if pressure:                 # drag alone restrains the speed: Newton on U|U|, half a step
                u_new = _tridiagonal(-c_lo[1:], c_lo + c_hi + 2.0 * dz * drag * np.abs(u), -c_hi[:-1],
                                     rhs + dz * drag * np.abs(u) * u)
                u_new = 0.5 * u + 0.5 * u_new
            else:
                u_new = _tridiagonal(-c_lo[1:], c_lo + c_hi + dz * drag * np.abs(u), -c_hi[:-1], rhs)
            top_shear = 0.0 if pressure else (u_top - u_new[-1]) / d_top
            prod_f = (nu - NU_AIR) * np.concatenate([np.diff(u_new) / dzc, [top_shear]]) ** 2
            prod = 0.5 * (np.concatenate([prod_f[:1], prod_f[:-1]]) + prod_f)
            sp = np.abs(u_new)
            rate = KL_C_MU ** 0.75 * np.sqrt(k) * ell_inv                   # eps / k
            kc_lo = np.concatenate([[0.0], nu[:-1] / KL_SIGMA_K / dzc])
            kc_hi = np.concatenate([nu[:-1] / KL_SIGMA_K / dzc, [0.0]])
            diag = kc_lo + kc_hi + dz * (1.5 * rate + KL_BETA_D * s * sp)   # eps linearized about the last k
            kr = dz * (prod + KL_BETA_P * s * sp ** 3 + 0.5 * rate * k)
            lo, hi = -kc_lo[1:], -kc_hi[:-1].copy()
            diag[0], kr[0], hi[0] = 1.0, fr0 * sp[0] ** 2 / math.sqrt(KL_C_MU), 0.0
            k_new = 0.5 * k + 0.5 * np.maximum(_tridiagonal(lo, diag, hi, kr), 1e-6 * u_top ** 2)
            done = (np.max(np.abs(u_new - u)) <= self.COLUMN_TOL * u_top
                    and np.max(np.abs(k_new - k)) <= self.COLUMN_TOL * np.max(k))
            u, k = u_new, k_new
            if done:
                break
        if pressure:                     # every term is quadratic in the speed: scaled so the top cell has u_top
            c = u_top / max(u[-1], 1e-9)
            u, k, body = u * c, k * c * c, body * c * c
        return {"h": h, "u": u, "k": k, "h_top": h_top, "u_top": u_top, "iterations": it + 1, "drag": s, "z0": z0m,
                "canopy_height": hc, "body": body}

    def _kl_geometry(self, z0: Tensor) -> None:
        """The k-l closure's fixed geometry: each column's canopy height (the top of its highest drag cell above its
        ground), the length on every z-face, the mean inverse length of each cell, and each wall cell's log-law
        (u* / |u|)^2."""
        h, dz = self.height, self.dz[:, None, None]
        self.canopy_height = torch.where(self.sink > 0, h + 0.5 * dz, torch.zeros_like(h)).max(dim=0).values
        hf = torch.cat([_logmean(h[:1], h[1:2]), _logmean(h[:-1], h[1:]), _logmean(h[-1:], h[-1:] + self.d_top)])
        self.ell_z = _kl_length(hf, self.canopy_height[None])
        self.ell_inv = 0.5 * (1.0 / self.ell_z[:-1] + 1.0 / self.ell_z[1:])
        s = self.solid
        half = torch.full_like(self.sink, self.dx / 2)
        dzh = (dz / 2).expand_as(self.sink)
        coef = lambda delta, z: (KAPPA / torch.log(delta / torch.minimum(z, delta * math.exp(-1.0)))) ** 2  # noqa: E731
        fr = torch.zeros_like(self.sink)
        fr[..., :-1] = torch.maximum(fr[..., :-1], s[..., 1:] * coef(half[..., 1:], z0[..., 1:]))
        fr[..., 1:] = torch.maximum(fr[..., 1:], s[..., :-1] * coef(half[..., :-1], z0[..., :-1]))
        fr[:, :-1, :] = torch.maximum(fr[:, :-1, :], s[:, 1:, :] * coef(half[:, 1:, :], z0[:, 1:, :]))
        fr[:, 1:, :] = torch.maximum(fr[:, 1:, :], s[:, :-1, :] * coef(half[:, :-1, :], z0[:, :-1, :]))
        fr[:-1] = torch.maximum(fr[:-1], s[1:] * coef(dzh[:-1], z0[1:]))
        fr[1:] = torch.maximum(fr[1:], s[:-1] * coef(dzh[1:], z0[:-1]))
        fr[0] = torch.maximum(fr[0], coef(dzh[0], z0[0]))
        self.wall_friction = torch.where(s, torch.zeros_like(fr), fr)
        self.wall_mask = (self.wall_friction > 0).to(self.sink.dtype)

    def _equilibrium_inflow(self, z0: Tensor, ex: float, ey: float) -> None:
        """Prescribe the equilibrium column on every side and start from it, at each cell's height above its ground."""
        kw = dict(device=self.sink.device, dtype=self.sink.dtype)
        height = self.height.double().cpu().numpy()
        vec = torch.tensor([ex, ey, 0.0], **kw)[:, None, None, None]

        def field(col):
            hh = np.concatenate([[0.0], col["h"], [col["h_top"]]])
            uu = np.concatenate([[0.0], col["u"], [col["u_top"]]])
            u = torch.as_tensor(np.interp(height, hh, uu), **kw)[None] * vec
            u[:, self.solid] = 0.0
            return u
        def kfield(col):
            hh = np.concatenate([[0.0], col["h"], [col["h_top"]]])
            kk = np.concatenate([col["k"][:1], col["k"], col["k"][-1:]])
            return torch.where(self.solid, torch.zeros_like(self.sink),
                               torch.as_tensor(np.interp(height, hh, kk), **kw))[None]
        kl = self.cfg.closure == "k-l"
        self.column = self.equilibrium_column(z0)
        u = field(self.column)
        self.initial = u
        sides = {"west": u[..., 0].clone(), "east": u[..., -1].clone(), "south": u[:, :, 0, :].clone(),
                 "north": u[:, :, -1, :].clone()}
        if kl:
            self.k = kfield(self.column)[0]
            k_sides = {"west": self.k[..., 0].clone(), "east": self.k[..., -1].clone(),
                       "south": self.k[:, 0, :].clone(), "north": self.k[:, -1, :].clone()}
        ring = int(round(self.cfg.inflow_ring_m / self.dx))
        if ring > 0:            # each side from the canopy of its own strip: the canopy the wind crosses to arrive
            strips = {"west": (slice(None), slice(0, ring)), "east": (slice(None), slice(-ring, None)),
                      "south": (slice(0, ring), slice(None)), "north": (slice(-ring, None), slice(None))}
            take = {"west": lambda a: a[..., 0], "east": lambda a: a[..., -1], "south": lambda a: a[:, :, 0, :],
                    "north": lambda a: a[:, :, -1, :]}
            for side, (sy, sx) in strips.items():
                where = torch.zeros(self.ny, self.nx, dtype=torch.bool, device=self.sink.device)
                where[sy, sx] = True
                col = self.equilibrium_column(z0, where=where)
                sides[side] = take[side](field(col)).clone()
                if kl:
                    k_sides[side] = take[side](kfield(col))[0].clone()
        self.bc = dict(sides, top=self.u0_top[:, None, None].expand(3, self.ny, self.nx).clone())
        if kl:
            self.k_bc = k_sides

    # ── Geometry ─────────────────────────────────────────────────────────────────────────

    def _wall_coefficient(self, z0: Tensor) -> Tensor:
        """Sum of log-law stress coefficients over each fluid cell's solid neighbors."""
        s, dx = self.solid, self.dx
        half = torch.full_like(self.sink, dx / 2)
        size = torch.full_like(self.sink, dx)
        dz = self.dz[:, None, None].expand_as(self.sink)
        w = torch.zeros_like(self.sink)
        w[..., :-1] += s[..., 1:] * _wall(half[..., 1:], z0[..., 1:], size[..., 1:])
        w[..., 1:] += s[..., :-1] * _wall(half[..., :-1], z0[..., :-1], size[..., :-1])
        w[:, :-1, :] += s[:, 1:, :] * _wall(half[:, 1:, :], z0[:, 1:, :], size[:, 1:, :])
        w[:, 1:, :] += s[:, :-1, :] * _wall(half[:, :-1, :], z0[:, :-1, :], size[:, :-1, :])
        w[:-1] += s[1:] * _wall(dz[:-1] / 2, z0[1:], dz[:-1])
        w[1:] += s[:-1] * _wall(dz[1:] / 2, z0[:-1], dz[1:])
        w[0] += _wall(dz[0] / 2, z0[0], dz[0])
        return torch.where(s, torch.zeros_like(w), w)

    def _operator(self, nu_x: Tensor, nu_y: Tensor, nu_z: Tensor, ident: Tensor,
                  dirichlet_inflow: bool) -> Operator:
        """The 7-point operator with face conductances nu * area / distance on open faces.

        Boundary faces are Dirichlet (conductance over half a cell) where `edge` is open and
        the side is an outflow, or an inflow when `dirichlet_inflow`; otherwise Neumann.
        """
        west, east, south, north, top = self.edge
        side = lambda i: (self.inflow[i] == dirichlet_inflow)  # noqa: E731
        ax = torch.zeros(self.nz, self.ny, self.nx + 1, dtype=self.sink.dtype, device=self.sink.device)
        ay = torch.zeros(self.nz, self.ny + 1, self.nx, dtype=self.sink.dtype, device=self.sink.device)
        az = torch.zeros(self.nz + 1, self.ny, self.nx, dtype=self.sink.dtype, device=self.sink.device)
        ax[..., 1:-1] = self.open_x * nu_x[..., 1:-1] * self.area_x / self.dx
        ay[:, 1:-1, :] = self.open_y * nu_y[:, 1:-1, :] * self.area_x / self.dx
        az[1:-1] = self.open_z * nu_z[1:-1] * self.area_z / self.dzc[:, None, None]
        ax[..., 0] = west * side(0) * nu_x[..., 0] * self.area_x[:, :, 0] / (self.dx / 2)
        ax[..., -1] = east * side(1) * nu_x[..., -1] * self.area_x[:, :, 0] / (self.dx / 2)
        ay[:, 0, :] = south * side(2) * nu_y[:, 0, :] * self.area_x[:, :, 0] / (self.dx / 2)
        ay[:, -1, :] = north * side(3) * nu_y[:, -1, :] * self.area_x[:, :, 0] / (self.dx / 2)
        az[-1] = top * nu_z[-1] * self.area_z / self.d_top
        return Operator(ident, ax, ay, az)

    def _projection_operator(self) -> Operator:
        one = torch.ones_like(self.sink)
        ones = lambda pad: F.pad(one, pad, value=1.0)  # noqa: E731
        return self._operator(ones((0, 1)), ones((0, 0, 0, 1)), ones((0, 0, 0, 0, 0, 1)),
                              torch.zeros_like(one), False)

    # ── Fields ───────────────────────────────────────────────────────────────────────────

    def background(self) -> Tensor:
        """The upwind profile everywhere (a unit's coarse field, when it has one), zero inside solids,
        (3, nz, ny, nx)."""
        if self.initial is not None:
            u = self.initial.clone()
        else:
            u = self.u0[:, :, None, None].expand(3, self.nz, self.ny, self.nx).clone()
        u[:, self.solid] = 0.0
        return u

    def _sides(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """(3, nz, n) velocities prescribed on the west, east, south and north sides: the profile, or a
        unit's boundary."""
        if self.bc is not None:
            return self.bc["west"], self.bc["east"], self.bc["south"], self.bc["north"]
        u0 = self.u0[:, :, None]
        return u0, u0, u0, u0

    def faces(self, u: Tensor) -> Faces:
        """Face velocities from cell velocities: averages inside, the profile (or a unit's boundary) on
        inflow sides, the cell value on outflow sides and the top, zero on blocked faces."""
        west, east, south, north, top = self.edge
        ux = F.pad(0.5 * (u[0, ..., :-1] + u[0, ..., 1:]), (1, 1))
        uy = F.pad(0.5 * (u[1, :, :-1, :] + u[1, :, 1:, :]), (0, 0, 1, 1))
        uz = F.pad(0.5 * (u[2, :-1] + u[2, 1:]), (0, 0, 0, 0, 1, 1))
        bw, be, bs, bn = self._sides()
        ux[..., 0] = west * (bw[0] if self.inflow[0] else u[0, ..., 0])
        ux[..., -1] = east * (be[0] if self.inflow[1] else u[0, ..., -1])
        uy[:, 0, :] = south * (bs[1] if self.inflow[2] else u[1, :, 0, :])
        uy[:, -1, :] = north * (bn[1] if self.inflow[3] else u[1, :, -1, :])
        uz[-1] = top * u[2, -1]
        ux[..., 1:-1] *= self.open_x
        uy[:, 1:-1, :] *= self.open_y
        uz[1:-1] *= self.open_z
        return ux, uy, uz

    def cells(self, faces: Faces) -> Tensor:
        ux, uy, uz = faces
        return torch.stack([0.5 * (ux[..., :-1] + ux[..., 1:]),
                            0.5 * (uy[:, :-1, :] + uy[:, 1:, :]),
                            0.5 * (uz[:-1] + uz[1:])])

    def divergence(self, faces: Faces) -> Tensor:
        """Net outward volume flux of every cell [m^3/s]."""
        ux, uy, uz = faces
        return (self.area_x * (ux[..., 1:] - ux[..., :-1])
                + self.area_x * (uy[:, 1:, :] - uy[:, :-1, :])
                + self.area_z * (uz[1:] - uz[:-1]))

    def project(self, faces: Faces, tol: Optional[float] = None,
                max_iter: Optional[int] = None) -> Tuple[Faces, int, float]:
        """The minimal correction making `faces` divergence-free.

        Returns:
            (corrected faces, PCG iterations, relative residual).
        """
        div = self.divergence(faces)
        x0 = self._lam if self.cfg.warm_projection and self.cfg.scheme != "fv" else None
        lam, it, rel = self.poisson.solve(div[None], x0=x0, tol=tol or self.cfg.tol,
                                          max_iter=max_iter or self.cfg.max_iter)
        self._lam = lam
        return tuple(f + g for f, g in zip(faces, self.face_gradient(lam[0]))), it, rel

    def face_gradient(self, lam: Tensor) -> Faces:
        """The face velocities a multiplier `lam` adds: its gradient on open faces, and across the boundary faces
        where the projection holds it at zero (outflow sides and the open top)."""
        op = self.poisson.levels[0].op
        gx = torch.zeros(self.nz, self.ny, self.nx + 1, dtype=lam.dtype, device=lam.device)
        gy = torch.zeros(self.nz, self.ny + 1, self.nx, dtype=lam.dtype, device=lam.device)
        gz = torch.zeros(self.nz + 1, self.ny, self.nx, dtype=lam.dtype, device=lam.device)
        gx[..., 1:-1] = self.open_x * (lam[..., 1:] - lam[..., :-1]) / self.dx
        gy[:, 1:-1, :] = self.open_y * (lam[:, 1:, :] - lam[:, :-1, :]) / self.dx
        gz[1:-1] = self.open_z * (lam[1:] - lam[:-1]) / self.dzc[:, None, None]
        gx[..., 0] = (op.ax[..., 0] > 0) * lam[..., 0] / (self.dx / 2)
        gx[..., -1] = -((op.ax[..., -1] > 0) * lam[..., -1]) / (self.dx / 2)
        gy[:, 0, :] = (op.ay[:, 0, :] > 0) * lam[:, 0, :] / (self.dx / 2)
        gy[:, -1, :] = -((op.ay[:, -1, :] > 0) * lam[:, -1, :]) / (self.dx / 2)
        gz[-1] = -((op.az[-1] > 0) * lam[-1]) / self.d_top
        return gx, gy, gz

    def boundary_flux(self, faces: Faces) -> Tuple[float, float]:
        """(entering, leaving) volume flux through the domain boundary [m^3/s]."""
        ux, uy, uz = faces
        area = self.area_x[:, :, 0]
        entering = torch.cat([
            (area * ux[..., 0]).ravel(), (-area * ux[..., -1]).ravel(),
            (area * uy[:, 0, :]).ravel(), (-area * uy[:, -1, :]).ravel(),
            (-self.area_z * uz[-1]).ravel(), (self.area_z * uz[0]).ravel()])
        return float(entering.clamp(min=0).sum()), float((-entering).clamp(min=0).sum())

    # ── Momentum ─────────────────────────────────────────────────────────────────────────

    def _padded(self, field: Tensor) -> Tensor:
        """`field` with a one-cell halo: the profile (or a unit's boundary) on the sides and top, zero at
        the ground."""
        pad = F.pad(field, (1, 1, 1, 1, 1, 1))
        if self.bc is not None:
            b = self.bc
            pad[:, 1:-1, 1:-1, 0], pad[:, 1:-1, 1:-1, -1] = b["west"], b["east"]
            pad[:, 1:-1, 0, 1:-1], pad[:, 1:-1, -1, 1:-1] = b["south"], b["north"]
            pad[:, 1:-1, 0, 0], pad[:, 1:-1, 0, -1] = b["south"][..., 0], b["south"][..., -1]
            pad[:, 1:-1, -1, 0], pad[:, 1:-1, -1, -1] = b["north"][..., 0], b["north"][..., -1]
            pad[:, 0] = 0.0
            pad[:, -1] = F.pad(b["top"][None], (1, 1, 1, 1), mode="replicate")[0]
            return pad
        prof = torch.cat([torch.zeros_like(self.u0[:, :1]), self.u0, self.u0[:, -1:]], dim=1)
        pad[:, :, :, 0] = pad[:, :, :, -1] = prof[:, :, None]
        pad[:, :, 0, :] = pad[:, :, -1, :] = prof[:, :, None]
        pad[:, 0], pad[:, -1] = 0.0, self.u0_top[:, None, None]
        return pad

    def _departures(self, u: Tensor, sign: float) -> Tensor:
        """Sampling grid for the points x - sign dt u, in the halo's normalized index space."""
        nz, ny, nx, dt = self.nz, self.ny, self.nx, sign * self.dt
        ix = torch.arange(nx, device=u.device, dtype=u.dtype) + 1 - dt * u[0] / self.dx
        iy = (torch.arange(ny, device=u.device, dtype=u.dtype) + 1)[:, None] - dt * u[1] / self.dx
        zd = self.zc[:, None, None] - dt * u[2]
        j = torch.searchsorted(self.zc, zd.reshape(-1)).reshape(zd.shape).clamp(1, nz - 1)
        z0, z1 = self.zc[j - 1], self.zc[j]
        kz = (j - 1) + (zd - z0) / (z1 - z0) + 1
        return torch.stack([2 * ix / (nx + 1) - 1, 2 * iy / (ny + 1) - 1, 2 * kz / (nz + 1) - 1],
                           dim=-1)[None]

    @staticmethod
    def _sample(padded: Tensor, grid: Tensor, mode: str = "bilinear") -> Tensor:
        return F.grid_sample(padded[None], grid, mode=mode, padding_mode="border",
                             align_corners=True)[0]

    def advect(self, u: Tensor) -> Tensor:
        """Semi-Lagrangian advection with a MacCormack correction, limited to the range of the
        departure point's neighbors."""
        back, fore = self._departures(u, 1.0), self._departures(u, -1.0)
        forward = self._sample(self._padded(u), back)
        out = forward + 0.5 * (u - self._sample(self._padded(forward), fore))
        lo = -F.max_pool3d(-self._padded(u)[None], 3, 1, 1)[0]
        hi = F.max_pool3d(self._padded(u)[None], 3, 1, 1)[0]
        out = torch.minimum(torch.maximum(out, self._sample(lo, back, "nearest")),
                            self._sample(hi, back, "nearest"))
        out[:, self.solid] = 0.0
        return out

    def strain_rest(self, u: Tensor) -> Tensor:
        """2 S_ij S_ij at cell centres less its vertical-shear terms (du/dz)^2 + (dv/dz)^2."""
        (uz, uy, ux), (vz, vy, vx), (wz, wy, wx) = [
            torch.gradient(u[c], spacing=[self.zc, self.yc, self.xc], dim=(0, 1, 2))
            for c in range(3)]
        return (2 * (ux ** 2 + vy ** 2 + wz ** 2) + (uy + vx) ** 2
                + wx ** 2 + 2 * uz * wx + wy ** 2 + 2 * vz * wy)

    def viscosity(self, u: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Mixing-length eddy viscosity (kappa h)^2 |S| on x-, y- and z-faces.

        On a z-face the vertical shear is the difference across that face and the mixing
        length uses the logarithmic mean of the two heights above the local solid surface, so
        the discrete log profile is an exact steady state.
        """
        h = self.height
        rest = self.strain_rest(u)
        shear = (u[:2, 1:] - u[:2, :-1]).norm(dim=0) / self.dzc[:, None, None]
        s_int = torch.clamp(shear ** 2 + 0.5 * (rest[:-1] + rest[1:]), min=0).sqrt()
        h_ghost = h[-1] + self.d_top
        above = self.bc["top"][:2] if self.bc is not None else self.u0_top[:2, None, None]
        shear_top = (above - u[:2, -1]).norm(dim=0) / self.d_top
        s_top = torch.clamp(shear_top ** 2 + rest[-1], min=0).sqrt()
        if self.k is not None:          # k-l: nu_t = C_mu^(1/4) l sqrt(k) on each z-face, |S|^2 kept for k's production
            self._s2 = torch.cat([s_int[:1], s_int, s_top[None]]) ** 2
            if self.body_vec is not None:
                self._s2[-1] = 0.0            # no stress through a pressure-driven top
            k = self.k
            kf = torch.cat([k[:1], 0.5 * (k[:-1] + k[1:]), k[-1:]]).clamp(min=0)
            nu_z = KL_C_MU ** 0.25 * self.ell_z * kf.sqrt() + NU_AIR
        else:
            nu_int = (KAPPA * _logmean(h[:-1], h[1:])) ** 2 * s_int + NU_AIR
            nu_top = (KAPPA * _logmean(h[-1], h_ghost)) ** 2 * s_top + NU_AIR
            nu_z = torch.cat([nu_int[:1], nu_int, nu_top[None]])
        nu_c = 0.5 * (nu_z[:-1] + nu_z[1:])
        padded = F.pad(nu_c[None], (1, 1, 1, 1), mode="replicate")[0]
        nu_x = 0.5 * (padded[:, 1:-1, :-1] + padded[:, 1:-1, 1:])
        nu_y = 0.5 * (padded[:, :-1, 1:-1] + padded[:, 1:, 1:-1])
        return nu_x, nu_y, nu_z

    def diffuse(self, u_adv: Tensor, u: Tensor) -> Tuple[Tensor, int]:
        """Implicit diffusion, canopy drag and wall stress, linearized about `u`."""
        nu_x, nu_y, nu_z = self.viscosity(u)
        speed = u.norm(dim=0)
        ident = self.vol * (1.0 + self.dt * (self.wall + self.sink) * speed)
        op = self._operator(self.dt * nu_x, self.dt * nu_y, self.dt * nu_z, ident, True)
        rhs = self.vol * u_adv
        bw, be, bs, bn = self._sides()
        rhs[..., 0] += op.ax[..., 0] * bw
        rhs[..., -1] += op.ax[..., -1] * be
        rhs[:, :, 0, :] += op.ay[:, 0, :] * bs
        rhs[:, :, -1, :] += op.ay[:, -1, :] * bn
        rhs[:, -1] += op.az[-1] * (self.bc["top"] if self.bc is not None else self.u0_top[:, None, None])
        mg = Poisson(op, factors=self._momentum_factors)
        self._momentum_factors = mg.factor_list      # the grid's; rebuilding the hierarchy then needs no host sync
        out, it, _ = mg.solve(rhs, x0=u, tol=self.cfg.tol_momentum, max_iter=self.cfg.max_iter)
        out[:, self.solid] = 0.0
        return out, it

    # ── Steady finite volume (cfg.scheme == "fv") ────────────────────────────────────────

    def _boundary_terms(self, op: Operator) -> Tensor:
        """What a diffusion operator's boundary faces carry in from the prescribed side and top velocities."""
        out = torch.zeros(3, self.nz, self.ny, self.nx, dtype=self.sink.dtype, device=self.sink.device)
        bw, be, bs, bn = self._sides()
        out[..., 0] += op.ax[..., 0] * bw
        out[..., -1] += op.ax[..., -1] * be
        out[:, :, 0, :] += op.ay[:, 0, :] * bs
        out[:, :, -1, :] += op.ay[:, -1, :] * bn
        out[:, -1] += op.az[-1] * (self.bc["top"] if self.bc is not None else self.u0_top[:, None, None])
        return out

    def convection(self, u: Tensor, fx: Tensor, fy: Tensor, fz: Tensor, padded: Optional[Tensor] = None) -> Tensor:
        """Net outflow sum_f F_f u_f of each cell [m^4/s^2]: the face volume flux times the upwind MUSCL face value,
        van Leer limited (van Leer 1979), the halo carrying the prescribed sides and top and zero at the ground
        (`padded`: another field's own halo)."""
        p = F.pad(self._padded(u) if padded is None else padded, (1, 1, 1, 1, 1, 1), mode="replicate")

        def flux(q: Tensor, f: Tensor, dim: int) -> Tensor:
            n = q.shape[dim]
            ll, left, right, rr = (q.narrow(dim, s, n - 3) for s in (0, 1, 2, 3))
            lim = self.cfg.limiter
            face = torch.where(f > 0, _muscl(ll, left, right, lim), _muscl(rr, right, left, lim))
            return f * face

        out = torch.empty_like(u)
        for c in range(u.shape[0]):                 # one component at a time: a third of the temporaries
            q = p[c:c + 1]
            fxu = flux(q[:, 2:-2, 2:-2, :], fx, -1)
            fyu = flux(q[:, 2:-2, :, 2:-2], fy, -2)
            fzu = flux(q[:, :, 2:-2, 2:-2], fz, -3)
            out[c] = ((fxu[..., 1:] - fxu[..., :-1]) + (fyu[:, :, 1:, :] - fyu[:, :, :-1, :]) + (fzu[:, 1:] - fzu[:, :-1]))[0]
        return out

    def fv_increment(self, faces: Faces, u: Tensor, pressure: Tensor) -> Tuple[Tensor, int]:
        """One implicit pseudo-time step in delta form:
            M du = dt R(u) + V G(Pi),   M = V (1 + dt c |u|) + dt (D + A_upwind)
        R is the steady momentum residual (MUSCL convection, mixing-length diffusion, canopy drag and wall stress),
        V G(Pi) the accumulated projection gradient (`pressure`, cell-centered). M only sets how fast the iteration
        moves: where du = 0 the steady equations hold whatever dt, so the delivered field does not depend on it."""
        ux, uy, uz = faces
        fx, fy, fz = ux * self.area_x, uy * self.area_x, uz * self.area_z
        nu_x, nu_y, nu_z = self.viscosity(u)
        a = self.cfg.relax_nu
        if a < 1.0 and getattr(self, "_nu", None) is not None:     # the eddy viscosity under-relaxed (Picard)
            nu_x, nu_y, nu_z = (a * n + (1.0 - a) * o for n, o in zip((nu_x, nu_y, nu_z), self._nu))
        self._nu = (nu_x, nu_y, nu_z)
        if self.body_vec is not None:      # driven by the mean pressure gradient: no stress through the top
            nu_z = nu_z.clone()
            nu_z[-1] = 0.0
        diff_op = self._operator(nu_x, nu_y, nu_z, torch.zeros_like(self.sink), True)
        speed = u.norm(dim=0)
        drag = self.wall + self.sink
        residual = (self._boundary_terms(diff_op) - diff_op.apply(u) - self.convection(u, fx, fy, fz)
                    - self.vol * drag * speed * u)
        if self.body_vec is not None:
            residual = residual + self.vol * self.body_vec[:, None, None, None] * (~self.solid)
        dt = self.dt
        m = Convective(ident=self.vol * (1.0 + dt * drag * speed), ax=dt * diff_op.ax, ay=dt * diff_op.ay,
                       az=dt * diff_op.az, fx=dt * fx, fy=dt * fy, fz=dt * fz)
        mg = Poisson(m, factors=self._momentum_factors)
        del m
        self._momentum_factors = mg.factor_list
        rhs = dt * residual + self.vol * pressure
        del residual, diff_op
        du, its = torch.empty_like(rhs), []
        for c in range(rhs.shape[0]):               # one component at a time: a third of the Krylov vectors
            d, i, _ = mg.solve(rhs[c:c + 1], tol=self.cfg.tol_momentum, max_iter=self.cfg.max_iter)
            du[c] = d[0]
            its.append(i)
        it = max(its)
        du[:, self.solid] = 0.0
        self.last_residual = rhs.norm() / dt                  # the steady momentum residual, pressure included
        return du, it

    def _padded_k(self, k: Tensor) -> Tensor:
        """k with a one-cell halo, (1, nz+2, ny+2, nx+2): the inflow's k on the sides, no gradient at the ground and top."""
        pad = F.pad(k, (1, 1, 1, 1, 1, 1))
        b = self.k_bc
        pad[1:-1, 1:-1, 0], pad[1:-1, 1:-1, -1] = b["west"], b["east"]
        pad[1:-1, 0, 1:-1], pad[1:-1, -1, 1:-1] = b["south"], b["north"]
        pad[1:-1, 0, 0], pad[1:-1, 0, -1] = b["south"][..., 0], b["south"][..., -1]
        pad[1:-1, -1, 0], pad[1:-1, -1, -1] = b["north"][..., 0], b["north"][..., -1]
        pad[0], pad[-1] = pad[1], pad[-2]
        return pad[None]

    def k_step(self, faces: Faces, u: Tensor) -> int:
        """One implicit pseudo-time step of the turbulent kinetic energy, in delta form as the momentum's:
            dk/dt + div(u k) = div(nu_t / sigma_k grad k) + nu_t |S|^2 - eps + c_d a (beta_p |u|^3 - beta_d |u| k),
            eps = C_mu^(3/4) k^(3/2) / l          (Katul et al. 2004),
        a wall cell held at the log law's k = u*^2 / sqrt(C_mu), the inflow's k on the sides, and no flux of k through
        the top. Takes this step's divergence-free faces and the eddy viscosity the momentum step used."""
        k, dt, vol = self.k, self.dt, self.vol
        ux, uy, uz = faces
        fx, fy, fz = ux * self.area_x, uy * self.area_x, uz * self.area_z
        nu_x, nu_y, nu_z = self._nu
        nu_zk = nu_z.clone()
        nu_zk[-1] = 0.0
        op = self._operator(nu_x / KL_SIGMA_K, nu_y / KL_SIGMA_K, nu_zk / KL_SIGMA_K, torch.zeros_like(k), True)
        b = self.k_bc
        inflow = torch.zeros_like(k)
        inflow[..., 0] += op.ax[..., 0] * b["west"]
        inflow[..., -1] += op.ax[..., -1] * b["east"]
        inflow[:, 0, :] += op.ay[:, 0, :] * b["south"]
        inflow[:, -1, :] += op.ay[:, -1, :] * b["north"]
        speed = u.norm(dim=0)
        s2 = self._s2
        production = 0.5 * ((nu_z[:-1] - NU_AIR) * s2[:-1] + (nu_z[1:] - NU_AIR) * s2[1:])
        rate = KL_C_MU ** 0.75 * k.clamp(min=0).sqrt() * self.ell_inv          # eps / k
        canopy = self.sink * speed
        k_wall = self.wall_friction * speed ** 2 / math.sqrt(KL_C_MU)
        source = (production + KL_BETA_P * canopy * speed ** 2 - (rate + KL_BETA_D * canopy) * k
                  + (KL_WALL / dt) * self.wall_mask * (k_wall - k))
        residual = inflow - op.apply(k[None])[0] - self.convection(k[None], fx, fy, fz, self._padded_k(k))[0] + vol * source
        m = Convective(ident=vol * (1.0 + dt * (1.5 * rate + KL_BETA_D * canopy) + KL_WALL * self.wall_mask),
                       ax=dt * op.ax, ay=dt * op.ay, az=dt * op.az, fx=dt * fx, fy=dt * fy, fz=dt * fz)
        del op
        mg = Poisson(m, factors=self._momentum_factors)
        dk, it, _ = mg.solve((dt * residual)[None], tol=self.cfg.tol_momentum, max_iter=self.cfg.max_iter)
        k = (k + dk[0]).clamp(min=self.k_floor)
        k[self.solid] = 0.0
        self.k = k
        return it

    def add_increment(self, faces: Faces, du: Tensor) -> Faces:
        """Faces plus the cell increment interpolated onto them; prescribed faces keep their values."""
        west, east, south, north, top = self.edge
        ux, uy, uz = (f.clone() for f in faces)
        ux[..., 1:-1] += 0.5 * (du[0, ..., :-1] + du[0, ..., 1:]) * self.open_x
        uy[:, 1:-1, :] += 0.5 * (du[1, :, :-1, :] + du[1, :, 1:, :]) * self.open_y
        uz[1:-1] += 0.5 * (du[2, :-1] + du[2, 1:]) * self.open_z
        if not self.inflow[0]:
            ux[..., 0] += west * du[0, ..., 0]
        if not self.inflow[1]:
            ux[..., -1] += east * du[0, ..., -1]
        if not self.inflow[2]:
            uy[:, 0, :] += south * du[1, :, 0, :]
        if not self.inflow[3]:
            uy[:, -1, :] += north * du[1, :, -1, :]
        uz[-1] += top * du[2, -1]
        return ux, uy, uz

    def vorticity(self, u: Tensor) -> Tensor:
        """Curl of the cell-centered velocity, (3, nz, ny, nx) [1/s]."""
        d = [torch.gradient(u[c], spacing=[self.zc, self.yc, self.xc], dim=(0, 1, 2))
             for c in range(3)]
        return torch.stack([d[2][1] - d[1][0], d[0][0] - d[2][2], d[1][2] - d[0][1]])

    # ── Settling ─────────────────────────────────────────────────────────────────────────

    SETTLE_RMS = 5.0
    """A settled window mean's RMS change may be this many times `settle_tol`: canopy cells keep moving about a
    steady median (Harvard at CFL 8, 40-step means: RMS 3.2 to 4.3 % of the median while it held within 0.4 %)."""

    NEAR_GROUND_M = (2.0, 30.0)
    """The cells a settling run watches: fluid centres this high above their surface, the product's levels (4 to 25 m).
    Watching 2 to 8 m alone stopped Harvard while its 25 m level still fell 8 % (0.598 at 1,013 s, 0.550 at 2,432 s)."""

    def near_ground(self, u: Tensor) -> Tensor:
        """Horizontal speed of the fluid cells NEAR_GROUND_M above the highest solid below them, as one vector."""
        lo, hi = self.NEAR_GROUND_M
        if not hasattr(self, "_near"):
            self._near = (~self.solid) & (self.height >= lo) & (self.height <= hi)
        return u[:2, self._near].norm(dim=0)

    LEVELS_M = (4.0, 5.0, 10.0, 25.0)
    """The published levels, each watched on its own band of cells within half a cell of that height."""

    def level_stats(self, u: Tensor) -> dict:
        """Median and p95 of the horizontal speed on each LEVELS_M band, {"4": [median, p95], ...}."""
        if not hasattr(self, "_bands"):
            half = max(0.5 * self.dx, 0.5)
            self._bands = {f"{h:g}": (~self.solid) & ((self.height - h).abs() <= half) for h in self.LEVELS_M}
        out = {}
        for k, m in self._bands.items():
            s = u[:2, m].norm(dim=0)
            if s.numel() == 0:
                continue
            s = s[::max(1, -(-s.numel() // (1 << 24)))]
            q = torch.quantile(s, torch.tensor([0.5, 0.95], dtype=s.dtype, device=s.device))
            out[k] = [float(q[0]), float(q[1])]
        return out

    @staticmethod
    def settle_change(a: Tensor, b: Tensor) -> dict:
        """How far the near-ground speed moved from `a` to `b`: its median and p95, relative, and the RMS change
        over the median. Quantiles over at most 2^24 cells (torch's limit), every k-th cell beyond."""
        k = max(1, -(-b.numel() // (1 << 24)))
        qa = torch.quantile(a[::k], torch.tensor([0.5, 0.95], dtype=a.dtype, device=a.device))
        qb = torch.quantile(b[::k], torch.tensor([0.5, 0.95], dtype=b.dtype, device=b.device))
        med, p95 = float(qb[0]), float(qb[1])
        return {"median": med, "p95": p95, "d_median": abs(med - float(qa[0])) / max(med, 1e-12),
                "d_p95": abs(p95 - float(qa[1])) / max(p95, 1e-12),
                "rms_rel": float(((b - a) ** 2).mean().sqrt()) / max(med, 1e-12)}

    # ── Run ──────────────────────────────────────────────────────────────────────────────

    def run(self, initial=None) -> Result:
        """Project the background (or `initial`: cell velocities, or a finite-volume `Result.state`) and relax it."""
        t0, cfg = time.time(), self.cfg
        state = initial if isinstance(initial, dict) else None
        u = self.background() if initial is None or state is not None else torch.as_tensor(
            initial, device=cfg.device, dtype=cfg.dtype)
        if self.column is not None:
            self.initial = None                     # the canopy inflow's start, used: a whole field freed
        faces = self.faces(u) if state is None else tuple(torch.as_tensor(f, device=cfg.device, dtype=cfg.dtype)
                                                            for f in state["faces"])
        div0 = float(self.divergence(faces).norm())
        faces, it, rel = self.project(faces)
        u = self.cells(faces)
        iterations, change = [it], []
        settling = cfg.settle_tol is not None
        last = cfg.max_steps if settling and cfg.max_steps else cfg.steps
        settle, calm, settled, step, hist = [], 0, None, -1, {}
        ref, acc = None, None
        fv = cfg.scheme == "fv"
        pi = torch.zeros_like(u[0])                 # the accumulated multiplier: -dt times the pressure
        if isinstance(initial, dict):               # a finite-volume state: its faces and pressure carry on
            pi = -torch.as_tensor(initial["pressure"], device=cfg.device, dtype=cfg.dtype) * self.dt
            if self._work.k is not None and initial.get("k") is not None:
                self._work.k = torch.as_tensor(initial["k"], device=cfg.device, dtype=self._work.sink.dtype)
        aa = _Anderson(cfg.anderson, cfg.anderson_store) if fv and cfg.anderson else None
        for step in range(last):
            prev = u
            w, wd = self._work, self._work.sink.dtype
            x0 = (*faces, pi, *w._nu) if aa is not None and getattr(w, "_nu", None) is not None else None
            if fv:
                pressure = self.cells(self.face_gradient(pi))
                du, _ = w.fv_increment(tuple(f.to(wd) for f in faces), u.to(wd), pressure.to(wd))
                res_norm = w.last_residual
                res0 = res_norm if step == 0 else res0
                faces = self.add_increment(faces, du.to(u.dtype))
            else:
                u_star = w.diffuse(w.advect(u.to(wd)), u.to(wd))[0].to(u.dtype)
                faces = self.faces(u_star)
            div0 = float(self.divergence(faces).norm())
            faces, it, rel = self.project(faces)
            if fv:
                pi = pi + self._lam[0]
            if x0 is not None:              # the step's map mixed with the last ones' (faces, multiplier, viscosity)
                mixed = aa.step(x0, (*faces, pi, *w._nu), residual_parts=3)
                faces, pi, w._nu = tuple(mixed[:3]), mixed[3], tuple(mixed[4:])
            u = self.cells(faces)
            if w.k is not None:                     # k-l: k follows the step's divergence-free field
                w.k_step(tuple(f.to(wd) for f in faces), u.to(wd))
            iterations.append(it)
            change.append(float((u - prev).abs().max()) / self.u_top)
            if PROGRESS is not None and ((step + 1) % PROGRESS_EVERY == 0 or step + 1 == last):
                print(f"PROGRESS wind {PROGRESS[0] * last + step + 1}/{PROGRESS[1] * last}", flush=True)
            if cfg.verbose and (step + 1) % 50 == 0:
                print(f"  step {step + 1:5d}  change {change[-1]:.2e}  poisson {it:3d} it  "
                      f"[{time.time() - t0:.0f}s]")
            if settling:
                acc = [f.clone() for f in faces] if acc is None else [a.add_(f) for a, f in zip(acc, faces)]
            if settling and (step + 1) % cfg.settle_every == 0:
                mean_faces = tuple(a / cfg.settle_every for a in acc)
                mean = self.cells(mean_faces)
                acc, now, levels = None, self.near_ground(mean), self.level_stats(mean)
                for k, v in levels.items():
                    for j, name in enumerate(("median", "p95")):
                        hist.setdefault(f"{k}m {name}", []).append(v[j])
                if ref is not None:
                    row = dict(self.settle_change(ref, now), step=step + 1, pseudo_s=round((step + 1) * self.dt, 1),
                               levels=levels)
                    if fv:
                        row["residual"] = float(res_norm / res0)
                    tail = max(tail_bound(h) for h in hist.values()) if hist else float("inf")
                    row["tail"] = tail if math.isfinite(tail) else None          # None: no decay measured yet
                    settle.append(row)
                    if cfg.settle_rule == "tail":
                        calm = 2 if tail <= cfg.settle_tol else 0
                    else:
                        calm = calm + 1 if (row["d_median"] < cfg.settle_tol and row["d_p95"] < cfg.settle_tol
                                            and row["rms_rel"] < self.SETTLE_RMS * cfg.settle_tol) else 0
                    if cfg.verbose:
                        print(f"  settle {json_row(row)}", flush=True)
                    if calm >= 2 and step + 1 >= cfg.steps:
                        settled = True
                        # the window's mean, divergence-free (faces averaged as faces under the finite volumes)
                        faces, it, rel = self.project(mean_faces if fv else self.faces(mean))
                        u = self.cells(faces)
                        break
                ref = now
        if settling and settled is None:
            settled = False
            print(f"  NOT SETTLED after {step + 1} steps: {json_row(settle[-1]) if settle else 'no check'}", flush=True)
        if PROGRESS is not None and settling and step + 1 < last:
            print(f"PROGRESS wind {(PROGRESS[0] + 1) * last}/{PROGRESS[1] * last}", flush=True)
        if cfg.tol_final is not None:
            faces, it, rel = self.project(faces, cfg.tol_final, cfg.max_iter_final)
            u = self.cells(faces)
            iterations.append(it)
        div = self.divergence(faces)
        flux_in, flux_out = self.boundary_flux(faces)
        tke = self._work.k.double().cpu().numpy() if self._work.k is not None else None
        return Result(
            velocity=u.cpu().numpy(),
            vorticity=self.vorticity(u).cpu().numpy(),
            faces=tuple(f.cpu().numpy() for f in faces),
            divergence_rel=float(div.norm()) / div0 if div0 > 0 else 0.0,
            divergence_max=float(div.abs().max()) / (self.u_top * self.dx * self.dx),
            divergence_max_1_s=float((div / self.vol).abs().max()),
            flux_in_m3_s=flux_in, flux_out_m3_s=flux_out,
            poisson_iterations=iterations, change=change,
            wall_s=time.time() - t0, cells=self.nz * self.ny * self.nx,
            steps=step + 1 if settling else cfg.steps, settled=settled, settle=settle or None,
            state=(dict({"faces": tuple(f.cpu().numpy() for f in faces), "pressure": (-pi / self.dt).cpu().numpy()},
                        **({"k": tke} if tke is not None else {})) if fv else None),
            tke=tke,
        )


def solve(scene: Scene, profile: LogProfile, direction_deg: float, cfg: SolverConfig,
          initial: Optional[np.ndarray] = None, boundary: Optional[Boundary] = None) -> Result:
    """Build the model for one forcing and run it; with `boundary`, a unit forced by a coarser solve."""
    return Model(scene, profile, direction_deg, cfg, boundary).run(initial)
