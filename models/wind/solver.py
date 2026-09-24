"""Mass-consistent wind over a parcel, in PyTorch.

Face velocities on a staggered grid are projected onto the divergence-free space by one Poisson
solve for a Lagrange multiplier: the minimal correction to the background profile that conserves
mass around terrain and buildings. Pseudo-time momentum iterations then add what the projection
alone has no term for: semi-Lagrangian advection, mixing-length diffusion, canopy drag
cd a |u| u, and a log-law wall stress carrying each class's roughness length. Every implicit
operator is a 7-point solve on the multigrid kernel in `poisson`.
"""

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

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
from poisson import Operator, Solver as Poisson

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


@dataclass
class Result:
    """Everything one run produces.

    Attributes:
        velocity: Cell-centred (u, v, w) [m/s], (3, nz, ny, nx).
        vorticity: Cell-centred curl [1/s], (3, nz, ny, nx).
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


def _logmean(a: Tensor, b: Tensor) -> Tensor:
    """Logarithmic mean of two positive tensors."""
    return torch.where(a == b, a, (b - a) / torch.log(b / a))


def _wall(delta: Tensor, z0: Tensor, size: Tensor) -> Tensor:
    """Log-law stress coefficient per unit volume [1/m]: (kappa / ln(delta / z0))^2 / size."""
    z0 = torch.minimum(z0, delta * math.exp(-1.0))
    return (KAPPA / torch.log(delta / z0)) ** 2 / size


class Model:
    """A scene and its forcing, discretised and ready to run.

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
        self.poisson = Poisson(self._projection_operator())

    # ── Geometry ─────────────────────────────────────────────────────────────────────────

    def _wall_coefficient(self, z0: Tensor) -> Tensor:
        """Sum of log-law stress coefficients over each fluid cell's solid neighbours."""
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
        ux, uy, uz = faces
        div = self.divergence(faces)
        lam, it, rel = self.poisson.solve(div[None], tol=tol or self.cfg.tol, max_iter=max_iter or self.cfg.max_iter)
        lam = lam[0]
        op = self.poisson.levels[0].op
        ux, uy, uz = ux.clone(), uy.clone(), uz.clone()
        ux[..., 1:-1] += self.open_x * (lam[..., 1:] - lam[..., :-1]) / self.dx
        uy[:, 1:-1, :] += self.open_y * (lam[:, 1:, :] - lam[:, :-1, :]) / self.dx
        uz[1:-1] += self.open_z * (lam[1:] - lam[:-1]) / self.dzc[:, None, None]
        ux[..., 0] += (op.ax[..., 0] > 0) * lam[..., 0] / (self.dx / 2)
        ux[..., -1] -= (op.ax[..., -1] > 0) * lam[..., -1] / (self.dx / 2)
        uy[:, 0, :] += (op.ay[:, 0, :] > 0) * lam[:, 0, :] / (self.dx / 2)
        uy[:, -1, :] -= (op.ay[:, -1, :] > 0) * lam[:, -1, :] / (self.dx / 2)
        uz[-1] -= (op.az[-1] > 0) * lam[-1] / self.d_top
        return (ux, uy, uz), it, rel

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
        """Sampling grid for the points x - sign dt u, in the halo's normalised index space."""
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
        departure point's neighbours."""
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
        nu_int = (KAPPA * _logmean(h[:-1], h[1:])) ** 2 * s_int + NU_AIR
        h_ghost = h[-1] + self.d_top
        above = self.bc["top"][:2] if self.bc is not None else self.u0_top[:2, None, None]
        shear_top = (above - u[:2, -1]).norm(dim=0) / self.d_top
        s_top = torch.clamp(shear_top ** 2 + rest[-1], min=0).sqrt()
        nu_top = (KAPPA * _logmean(h[-1], h_ghost)) ** 2 * s_top + NU_AIR
        nu_z = torch.cat([nu_int[:1], nu_int, nu_top[None]])
        nu_c = 0.5 * (nu_z[:-1] + nu_z[1:])
        padded = F.pad(nu_c[None], (1, 1, 1, 1), mode="replicate")[0]
        nu_x = 0.5 * (padded[:, 1:-1, :-1] + padded[:, 1:-1, 1:])
        nu_y = 0.5 * (padded[:, :-1, 1:-1] + padded[:, 1:, 1:-1])
        return nu_x, nu_y, nu_z

    def diffuse(self, u_adv: Tensor, u: Tensor) -> Tuple[Tensor, int]:
        """Implicit diffusion, canopy drag and wall stress, linearised about `u`."""
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
        out, it, _ = Poisson(op).solve(rhs, x0=u, tol=self.cfg.tol_momentum,
                                       max_iter=self.cfg.max_iter)
        out[:, self.solid] = 0.0
        return out, it

    def vorticity(self, u: Tensor) -> Tensor:
        """Curl of the cell-centred velocity, (3, nz, ny, nx) [1/s]."""
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

    def run(self, initial: Optional[np.ndarray] = None) -> Result:
        """Project the background (or `initial`) and relax it for `cfg.steps` iterations."""
        t0, cfg = time.time(), self.cfg
        u = self.background() if initial is None else torch.as_tensor(
            initial, device=cfg.device, dtype=cfg.dtype)
        faces = self.faces(u)
        div0 = float(self.divergence(faces).norm())
        faces, it, rel = self.project(faces)
        u = self.cells(faces)
        iterations, change = [it], []
        settling = cfg.settle_tol is not None
        last = cfg.max_steps if settling and cfg.max_steps else cfg.steps
        settle, calm, settled, step = [], 0, None, -1
        ref, acc = None, None
        for step in range(last):
            prev = u
            u_star, _ = self.diffuse(self.advect(u), u)
            faces = self.faces(u_star)
            div0 = float(self.divergence(faces).norm())
            faces, it, rel = self.project(faces)
            u = self.cells(faces)
            iterations.append(it)
            change.append(float((u - prev).abs().max()) / self.u_top)
            if PROGRESS is not None and ((step + 1) % PROGRESS_EVERY == 0 or step + 1 == last):
                print(f"PROGRESS wind {PROGRESS[0] * last + step + 1}/{PROGRESS[1] * last}", flush=True)
            if cfg.verbose and (step + 1) % 50 == 0:
                print(f"  step {step + 1:5d}  change {change[-1]:.2e}  poisson {it:3d} it  "
                      f"[{time.time() - t0:.0f}s]")
            if settling:
                acc = u.clone() if acc is None else acc.add_(u)
            if settling and (step + 1) % cfg.settle_every == 0:
                mean = acc / cfg.settle_every
                acc, now = None, self.near_ground(mean)
                if ref is not None:
                    row = dict(self.settle_change(ref, now), step=step + 1, pseudo_s=round((step + 1) * self.dt, 1))
                    settle.append(row)
                    calm = calm + 1 if (row["d_median"] < cfg.settle_tol and row["d_p95"] < cfg.settle_tol
                                        and row["rms_rel"] < self.SETTLE_RMS * cfg.settle_tol) else 0
                    if cfg.verbose:
                        print(f"  settle {json_row(row)}", flush=True)
                    if calm >= 2 and step + 1 >= cfg.steps:
                        settled = True
                        faces, it, rel = self.project(self.faces(mean))      # the window's mean, divergence-free
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
        )


def solve(scene: Scene, profile: LogProfile, direction_deg: float, cfg: SolverConfig,
          initial: Optional[np.ndarray] = None, boundary: Optional[Boundary] = None) -> Result:
    """Build the model for one forcing and run it; with `boundary`, a unit forced by a coarser solve."""
    return Model(scene, profile, direction_deg, cfg, boundary).run(initial)
