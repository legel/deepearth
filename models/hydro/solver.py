"""Bates et al. (2010) local-inertial shallow-water solver on a raster grid, in PyTorch.

Rain and prescribed edge inflow enter, water infiltrates against a finite soil store and fills
surface storage, the rest routes under gravity and Manning friction and leaves through open
edges. One sub-step is one function of tensors; the same code runs eagerly on CPU and compiled
on CUDA. Mass balance is computed every run.
"""

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

import infiltration
from infiltration import Soil
from inflow import Inflow
from physics import FLOODED_DEPTH_THR_M, FROUDE_CAP, G, MANNING_EXP, MIN_DEPTH

PROGRESS_LINES = False
"""When set (the command line sets it for every run it makes), each forcing interval integrated prints
`PROGRESS hydro k/n`, which a caller can read for its progress. It changes no
number the solver computes."""

Field = Union[float, np.ndarray]
FIELDS = (("depth", "m"), ("u", "m/s"), ("v", "m/s"))
"""Per-frame output fields with units: depth, eastward velocity, southward velocity."""

EDGE_DROP_M = 1.0
"""How far below its lowest valid neighbour a nodata cell sits. Water reaching the face falls
over a brink at the Froude cap and leaves the domain."""

SOURCE_STEP_M = 0.05
"""Most depth a rim source may add to a cell in one sub-step [m]. On a dry domain the CFL step is
the whole interval, and the capped Campanile rim inflow, 2.32 m/s of depth on its busiest cells,
put 139 m into them in that one step before this bound."""


@dataclass
class SolverConfig:
    """Numerical settings for one run.

    Args:
        dx: Cell size [m].
        dt_s: Forcing interval [s]; each is integrated with as many CFL sub-steps as it takes.
        cfl_alpha: CFL safety factor on the wave speed sqrt(g h).
        manning_n: Scalar Manning's n [s/m^(1/3)] used when `Surface.manning_n` is None.
        frame_interval_min: Spacing of saved frames [min].
        substep_cap: Sub-steps per interval before the interval is abandoned and counted.
        cfl_depth: Depth the CFL step is taken from: "cell" for the deepest cell, "face" for
            the deepest flow depth across any face.
        dtype: "float32" or "float64".
        device: torch device name; None picks CUDA when available.
        compile: torch.compile the sub-step; None enables it on CUDA only.
        block: Sub-steps between host synchronisations; None picks 8 on CUDA, 1 on CPU.
    """

    dx: float
    dt_s: float
    cfl_alpha: float = 0.15
    manning_n: float = 0.040
    frame_interval_min: float = 30.0
    substep_cap: int = 20000
    cfl_depth: str = "cell"
    dtype: str = "float32"
    device: Optional[str] = None
    compile: Optional[bool] = None
    block: Optional[int] = None


@dataclass
class Surface:
    """Static per-cell fields the solver reads.

    Args:
        z: Ground elevation [m], NaN outside the domain.
        f0: Horton initial infiltration capacity [m/s], scalar or per-cell.
        fc: Horton final infiltration capacity [m/s], scalar or per-cell.
        k: Horton decay constant [1/s], scalar or per-cell.
        max_deficit_m: Soil storage [m] per cell; infiltration stops when it is full. None is
            unbounded.
        manning_n: Per-cell Manning's n, averaged onto faces. None uses the config scalar.
        initial_h: Initial depth [m]. None starts dry.
        smax_m: Surface storage [m] per cell, filled from standing water at any rate before
            it can leave the cell. None is zero.
        soil: Green-Ampt with redistribution (`infiltration`). When given it replaces Horton and
            the deficit, and the soil's state bank is carried through every sub-step.
        soil_state: The bank an earlier run ended with (`Result.soil_state`). None starts with no
            wetting fronts, at the soil's theta_i.
    """

    z: np.ndarray
    f0: Field = 0.0
    fc: Field = 0.0
    k: Field = 0.0
    max_deficit_m: Optional[np.ndarray] = None
    manning_n: Optional[np.ndarray] = None
    initial_h: Optional[np.ndarray] = None
    smax_m: Optional[np.ndarray] = None
    soil: Optional[Soil] = None
    soil_state: Optional[np.ndarray] = None

    @property
    def valid(self) -> np.ndarray:
        """True where the cell carries real elevation."""
        return np.isfinite(self.z)


@dataclass
class Probes:
    """Optional cross-sections to measure discharge through.

    Args:
        gauge_rc: (row, col) of a cell whose discharge magnitude is recorded.
        watershed_mask: True inside a catchment; net flux across its boundary is recorded.
    """

    gauge_rc: Optional[Tuple[int, int]] = None
    watershed_mask: Optional[np.ndarray] = None


@dataclass
class MassBalance:
    """Volumes [m^3] over the whole run."""

    rain: float
    initial: float
    inflow: float
    created: float
    infiltrated: float
    abstracted: float
    stored: float
    outflow: float

    @property
    def supplied(self) -> float:
        """Everything that entered, `created` included: the volume the positivity clamp
        invents when a cell would go negative in one sub-step. It is reported rather than
        absorbed, because silently absorbing it is what lets a residual look clean."""
        return self.rain + self.initial + self.inflow + self.created

    @property
    def residual(self) -> float:
        """Unaccounted volume as a fraction of everything that entered."""
        if self.supplied <= 0.0:
            return 0.0
        closed = self.infiltrated + self.abstracted + self.stored + self.outflow
        return (self.supplied - closed) / self.supplied

    @property
    def residual_pct(self) -> float:
        return self.residual * 100.0


@dataclass
class Result:
    """Everything one run produces. Frames are [3, rows, cols] arrays ordered as `FIELDS`."""

    h_final: np.ndarray
    h_max: np.ndarray
    cum_infil: np.ndarray
    qx_final: np.ndarray
    qy_final: np.ndarray
    u_final: np.ndarray
    v_final: np.ndarray
    frames: List[np.ndarray]
    frame_times_s: List[float]
    series: Dict[str, np.ndarray]
    mass: MassBalance
    n_substeps: int
    substep_cap_hits: int
    wall_s: float
    device: str
    soil_state: Optional[np.ndarray] = None
    """GAR's state bank at the storm's end, [5, rows, cols] as `infiltration.BANK`: where the next storm starts."""


@dataclass
class Grid:
    """Static tensors for one run."""

    z: Tensor
    zmax_x: Tensor
    zmax_y: Tensor
    invalid: Tensor
    valid_f: Tensor
    f0: Tensor
    fc: Tensor
    k: Tensor
    deficit: Optional[Tensor]
    smax: Optional[Tensor]
    n2_x: Union[float, Tensor]
    n2_y: Union[float, Tensor]
    ws_sx: Optional[Tensor]
    ws_sy: Optional[Tensor]
    gauge: Optional[Tuple[int, int]]
    gar: Optional[Dict[str, Tensor]] = None


@dataclass
class State:
    """Tensors the sub-step advances in place. `acc` holds outflow, inflow, infiltrated,
    abstracted, gauge and watershed volumes and the sub-step count."""

    h: Tensor
    qx: Tensor
    qy: Tensor
    h_max: Tensor
    cum: Tensor
    abstracted: Tensor
    t: Tensor
    acc: Tensor
    bank: Optional[Tensor] = None
    """GAR's per-cell soil state, [5, rows, cols] as `infiltration.BANK`; None under Horton."""


@dataclass
class Forcing:
    """Per-interval inputs, overwritten in place so compiled code sees fixed addresses."""

    rain: Tensor
    t_end: Tensor
    west: Tensor
    east: Tensor
    north: Tensor
    south: Tensor
    source: Tensor
    source_dt: Tensor


@dataclass
class Kernel:
    """Compile-time constants of the sub-step."""

    dx: float
    dt_s: float
    alpha: float
    cell_cfl: bool


def horton_rate(t_s: float, f0: Field, fc: Field, k: Field) -> Field:
    """Horton infiltration capacity [m/s] at time `t_s`, scalar or per-cell."""
    return fc + (f0 - fc) * np.exp(-k * t_s)


def _cfl_dt(peak: Tensor, kern: Kernel, dt_max: Tensor) -> Tensor:
    """CFL-limited step [s] from the governing depth, never past the interval end."""
    wave = kern.alpha * kern.dx / torch.sqrt(G * peak)
    dt = torch.where(peak > MIN_DEPTH, wave.clamp(max=kern.dt_s), torch.full_like(peak, kern.dt_s))
    return torch.minimum(dt, dt_max).clamp(min=0.0)


def _face_flux(q: Tensor, hf: Tensor, deta: Tensor, n2: Union[float, Tensor], dt: Tensor,
               dx: float) -> Tensor:
    """Semi-implicit momentum update across one face family, Froude-capped."""
    num = q - G * hf * dt * deta / dx
    den = 1.0 + G * dt * n2 * q.abs() / (hf ** MANNING_EXP + 1e-10)
    q = torch.where(hf > MIN_DEPTH, num / den, 0.0)
    cap = FROUDE_CAP * hf * torch.sqrt(G * hf.clamp(min=MIN_DEPTH))
    return torch.clamp(q, -cap, cap)


def _edge(free: Tensor, forced: Tensor) -> Tensor:
    """Prescribed inflow where given, free outflow elsewhere."""
    return torch.where(forced != 0.0, forced, free)


def _substep(s: State, g: Grid, f: Forcing, kern: Kernel) -> None:
    """One CFL-limited sub-step, advancing `s` in place."""
    h = s.h
    eta = g.z + h
    hf_x = (torch.maximum(eta[:, 1:], eta[:, :-1]) - g.zmax_x).clamp(min=0.0)
    hf_y = (torch.maximum(eta[1:, :], eta[:-1, :]) - g.zmax_y).clamp(min=0.0)
    peak = h.max() if kern.cell_cfl else torch.maximum(hf_x.max(), hf_y.max())
    dt = torch.minimum(_cfl_dt(peak.double(), kern, f.t_end - s.t), f.source_dt)

    if g.gar is None:
        inf = g.fc + (g.f0 - g.fc) * torch.exp(-g.k * s.t)
        if g.deficit is not None:
            inf = torch.minimum(inf, (g.deficit - s.cum).clamp(min=0.0) / dt.clamp(min=1e-30))

    qxi = _face_flux(s.qx[:, 1:-1], hf_x, eta[:, 1:] - eta[:, :-1], g.n2_x, dt, kern.dx)
    qyi = _face_flux(s.qy[1:-1, :], hf_y, eta[1:, :] - eta[:-1, :], g.n2_y, dt, kern.dx)
    qx = torch.cat([_edge(qxi[:, :1].clamp(max=0.0), f.west), qxi,
                    _edge(qxi[:, -1:].clamp(min=0.0), -f.east)], dim=1)
    qy = torch.cat([_edge(qyi[:1, :].clamp(max=0.0), f.north), qyi,
                    _edge(qyi[-1:, :].clamp(min=0.0), -f.south)], dim=0)

    div = dt / kern.dx * (qx[:, :-1] - qx[:, 1:] + qy[:-1, :] - qy[1:, :])
    # Rain lands only where the terrain is mapped, matching what `MassBalance.rain` counts.
    wet = h + (div + (dt * f.rain) * g.valid_f + dt * f.source)
    created = wet.clamp(max=0.0)
    h = wet - created
    # Whatever reaches a nodata cell has left the domain: a brink at the edge of the mapped
    # ground, or an edge inflow prescribed onto ground the terrain raster does not cover.
    # It is zeroed here, so it is measured here or it is lost from the mass balance.
    spilled = torch.where(g.invalid, h, 0.0)
    h = h - spilled
    if g.gar is None:
        inf_amount = torch.minimum(inf * dt, h)
    else:
        inf_amount, bank = infiltration.step(h, s.bank, dt.to(h.dtype), g.gar, MIN_DEPTH)
        s.bank.copy_(bank)
    h = (h - inf_amount).clamp(min=0.0)
    if g.smax is not None:
        take = torch.minimum(h, (g.smax - s.abstracted).clamp(min=0.0))
        h = h - take
        s.abstracted.add_(take)
    else:
        take = inf_amount * 0.0

    leaving = ((-qx[:, 0]).clamp(min=0.0).sum(dtype=torch.float64) + qx[:, -1].clamp(min=0.0).sum(dtype=torch.float64)
               + (-qy[0, :]).clamp(min=0.0).sum(dtype=torch.float64) + qy[-1, :].clamp(min=0.0).sum(dtype=torch.float64)
               + spilled.sum(dtype=torch.float64) * kern.dx / dt.clamp(min=1e-30))
    entering = (qx[:, 0].clamp(min=0.0).sum(dtype=torch.float64) + (-qx[:, -1]).clamp(min=0.0).sum(dtype=torch.float64)
                + qy[0, :].clamp(min=0.0).sum(dtype=torch.float64) + (-qy[-1, :]).clamp(min=0.0).sum(dtype=torch.float64)
                + f.source.sum(dtype=torch.float64) * kern.dx)
    gauge = torch.zeros((), dtype=torch.float64, device=h.device)
    if g.gauge is not None:
        gr, gc = g.gauge
        gauge = torch.hypot(0.5 * (qx[gr, gc] + qx[gr, gc + 1]),
                            0.5 * (qy[gr, gc] + qy[gr + 1, gc])).double()
    ws = torch.zeros((), dtype=torch.float64, device=h.device)
    if g.ws_sx is not None:
        ws = ((g.ws_sx * qxi).sum(dtype=torch.float64) + (g.ws_sy * qyi).sum(dtype=torch.float64))

    s.acc[:6].add_(torch.stack([leaving * kern.dx, entering * kern.dx, inf_amount.sum(dtype=torch.float64),
                                take.sum(dtype=torch.float64), gauge * kern.dx, ws * kern.dx]) * dt)
    s.acc[6].add_((dt > 0.0).double())
    s.acc[7].sub_(created.sum(dtype=torch.float64))
    s.h.copy_(h)
    s.qx.copy_(qx)
    s.qy.copy_(qy)
    s.h_max.copy_(torch.maximum(s.h_max, h))
    s.cum.add_(inf_amount)
    s.t.add_(dt)


def _velocity(h: Tensor, qx: Tensor, qy: Tensor) -> Tuple[Tensor, Tensor]:
    """Cell-centred velocity [m/s] from face discharges, zero on dry cells."""
    wet = h > MIN_DEPTH
    hd = h.clamp(min=MIN_DEPTH)
    u = torch.where(wet, 0.5 * (qx[:, :-1] + qx[:, 1:]) / hd, 0.0)
    v = torch.where(wet, 0.5 * (qy[:-1, :] + qy[1:, :]) / hd, 0.0)
    return u, v


def _cells(a: Optional[np.ndarray], dtype: torch.dtype, device: torch.device) -> Optional[Tensor]:
    """Per-cell tensor, or None."""
    return None if a is None else torch.as_tensor(np.asarray(a), dtype=dtype, device=device)


def _field(x: Field, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Scalar fields become 0-d float64 tensors, arrays become device tensors of `dtype`."""
    if isinstance(x, np.ndarray) and x.ndim > 0:
        return torch.as_tensor(np.ascontiguousarray(x), dtype=dtype, device=device)
    return torch.tensor(float(x), dtype=torch.float64, device=device)


def _face_roughness(cfg: SolverConfig, manning_n: Optional[np.ndarray], dtype: torch.dtype,
                    device: torch.device) -> Tuple[Union[float, Tensor], Union[float, Tensor]]:
    """Squared Manning's n at x- and y-faces, the mean of the two adjacent cells."""
    if manning_n is None:
        return cfg.manning_n ** 2, cfg.manning_n ** 2
    mn = torch.as_tensor(np.asarray(manning_n, dtype=np.float32), dtype=dtype, device=device)
    return (0.5 * (mn[:, :-1] + mn[:, 1:])) ** 2, (0.5 * (mn[:-1, :] + mn[1:, :])) ** 2


def _neighbour_min(z: np.ndarray) -> np.ndarray:
    """Lowest finite 4-neighbour elevation per cell, NaN where there is none."""
    p = np.pad(z, 1, constant_values=np.nan)
    stack = np.stack([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
    return np.nanmin(np.where(np.isnan(stack), np.inf, stack), axis=0).astype(z.dtype)


def _brink(z: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Nodata cells that share a face with a valid cell: the open boundary."""
    return ~valid & np.isfinite(_neighbour_min(z))


def _watershed_signs(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-face +1/-1/0: which way across an interior face counts as leaving the watershed."""
    wm = mask.astype(bool)
    sx = wm[:, :-1].astype(np.float32) - wm[:, 1:].astype(np.float32)
    sy = wm[:-1, :].astype(np.float32) - wm[1:, :].astype(np.float32)
    return sx, sy


def _runtime(cfg: SolverConfig, n_cells: int) -> Tuple[torch.device, torch.dtype, bool, int, str]:
    """Resolve device, dtype, compilation, block length and compile mode."""
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = {"float32": torch.float32, "float64": torch.float64}[cfg.dtype]
    cuda = device.type == "cuda"
    compiled = cuda if cfg.compile is None else cfg.compile
    block = cfg.block or (8 if cuda else 1)
    mode = "reduce-overhead" if cuda and n_cells < 2 ** 20 else "default"
    return device, dtype, compiled, block, mode


def _gar(soil: Optional[Soil], shape: Tuple[int, int],
         device: torch.device) -> Optional[Dict[str, Tensor]]:
    """Per-cell GAR soil tensors, or None when the run uses Horton."""
    if soil is None:
        return None
    cell = lambda a: torch.as_tensor(np.broadcast_to(np.asarray(a, dtype=np.float64), shape).copy(),  # noqa: E731
                                     dtype=torch.float64, device=device)
    g = {k: cell(getattr(soil, k)) for k in ("ks", "psi_f", "theta_s", "theta_r", "lam", "theta_i")}
    g["theta_i"] = torch.minimum(torch.maximum(g["theta_i"], g["theta_r"]), g["theta_s"])
    g["ks"] = torch.nan_to_num(g["ks"], nan=0.0)
    g["f_max"] = cell(np.inf if soil.f_max is None else soil.f_max)
    return g


def _build(surface: Surface, cfg: SolverConfig, probes: Probes, device: torch.device,
           dtype: torch.dtype) -> Tuple[Grid, State, Forcing]:
    """Tensors for one run from the NumPy inputs."""
    valid = surface.valid
    z = surface.z.astype(np.float32 if dtype == torch.float32 else np.float64).copy()
    if valid.any():
        z[~valid] = np.nanmax(z) + 100.0
        brink = _brink(surface.z, valid)
        z[brink] = _neighbour_min(surface.z)[brink] - EDGE_DROP_M
    else:
        z[:] = 0.0
    zt = torch.as_tensor(z, dtype=dtype, device=device)
    rows, cols = z.shape
    n2_x, n2_y = _face_roughness(cfg, surface.manning_n, dtype, device)
    ws = (_watershed_signs(probes.watershed_mask) if probes.watershed_mask is not None
          else (None, None))
    grid = Grid(
        z=zt, zmax_x=torch.maximum(zt[:, 1:], zt[:, :-1]), zmax_y=torch.maximum(zt[1:, :], zt[:-1, :]),
        invalid=torch.as_tensor(~valid, device=device),
        valid_f=torch.as_tensor(valid, dtype=dtype, device=device),
        f0=_field(surface.f0, dtype, device), fc=_field(surface.fc, dtype, device),
        k=_field(surface.k, dtype, device),
        deficit=_cells(surface.max_deficit_m, dtype, device), smax=_cells(surface.smax_m, dtype, device),
        n2_x=n2_x, n2_y=n2_y, ws_sx=_cells(ws[0], dtype, device), ws_sy=_cells(ws[1], dtype, device),
        gauge=probes.gauge_rc, gar=_gar(surface.soil, z.shape, device),
    )
    h = np.zeros(z.shape, dtype=np.float64) if surface.initial_h is None else surface.initial_h.astype(np.float64)
    h = np.where(valid, h, 0.0)
    ht = torch.as_tensor(h, dtype=dtype, device=device)
    state = State(
        h=ht, qx=torch.zeros((rows, cols + 1), dtype=dtype, device=device),
        qy=torch.zeros((rows + 1, cols), dtype=dtype, device=device),
        h_max=torch.zeros_like(ht), cum=torch.zeros_like(ht), abstracted=torch.zeros_like(ht),
        t=torch.zeros((), dtype=torch.float64, device=device),
        acc=torch.zeros(8, dtype=torch.float64, device=device),
    )
    if grid.gar is not None:
        bank = surface.soil.bank(z.shape) if surface.soil_state is None else surface.soil_state
        assert bank.shape == (len(infiltration.BANK),) + z.shape, bank.shape
        state.bank = torch.as_tensor(np.ascontiguousarray(bank), dtype=torch.float64, device=device)
    forcing = Forcing(
        rain=torch.zeros((), dtype=torch.float64, device=device),
        t_end=torch.zeros((), dtype=torch.float64, device=device),
        west=torch.zeros((rows, 1), dtype=dtype, device=device),
        east=torch.zeros((rows, 1), dtype=dtype, device=device),
        north=torch.zeros((1, cols), dtype=dtype, device=device),
        south=torch.zeros((1, cols), dtype=dtype, device=device),
        source=torch.zeros((rows, cols), dtype=dtype, device=device),
        source_dt=torch.full((), 1e30, dtype=torch.float64, device=device),
    )
    for obj in (grid, state, forcing):
        for v in vars(obj).values():
            if isinstance(v, Tensor):
                torch._dynamo.mark_static_address(v)
    return grid, state, forcing


def _set_forcing(f: Forcing, rain: float, t_end: float, inflow: Optional[Inflow], t_s: float) -> None:
    """Load one interval's rain, end time and edge inflow into the static buffers."""
    f.rain.fill_(rain)
    f.t_end.fill_(t_end)
    if inflow is None:
        return
    q = inflow.at(t_s)
    f.west.copy_(torch.as_tensor(q["west"], dtype=f.west.dtype).view(-1, 1))
    f.east.copy_(torch.as_tensor(q["east"], dtype=f.east.dtype).view(-1, 1))
    f.north.copy_(torch.as_tensor(q["north"], dtype=f.north.dtype).view(1, -1))
    f.south.copy_(torch.as_tensor(q["south"], dtype=f.south.dtype).view(1, -1))
    if "rim" in q:
        f.source.view(-1)[torch.as_tensor(inflow.rim_index, device=f.source.device)] = torch.as_tensor(
            q["rim"], dtype=f.source.dtype, device=f.source.device)
        f.source_dt.fill_(SOURCE_STEP_M / max(float(np.max(q["rim"])), 1e-30))


def simulate(
    surface: Surface,
    rain: Sequence[float],
    cfg: SolverConfig,
    probes: Probes = Probes(),
    inflow: Optional[Inflow] = None,
    sink: Optional[Callable[[float, np.ndarray], None]] = None,
    verbose: bool = True,
) -> Result:
    """Integrate the storm.

    Args:
        surface: Terrain, soil and roughness fields.
        rain: Rainfall rate [m/s], one entry per `cfg.dt_s` interval.
        cfg: Numerical settings.
        probes: Optional gauge cell and watershed mask.
        inflow: Optional prescribed edge inflow.
        sink: Receives (seconds since start, [3, rows, cols] frame) for every saved frame; when
            None the frames are kept in memory on `Result.frames`.
        verbose: Print progress every 360 intervals.

    Returns:
        A `Result` whose `mass.residual` states how well volume closed.
    """
    assert len(rain) > 0, "rain series is empty"
    assert cfg.cfl_depth in ("cell", "face"), cfg.cfl_depth
    valid = surface.valid
    n_valid = int(valid.sum())
    device, dtype, compiled, block, mode = _runtime(cfg, surface.z.size)
    grid, state, forcing = _build(surface, cfg, probes, device, dtype)
    kern = Kernel(dx=cfg.dx, dt_s=cfg.dt_s, alpha=cfg.cfl_alpha, cell_cfl=cfg.cfl_depth == "cell")
    step = torch.compile(_substep, dynamic=False, mode=mode) if compiled else _substep

    dx, cell_ha = cfg.dx, cfg.dx * cfg.dx / 1e4
    frame_interval_s = cfg.frame_interval_min * 60.0
    series: Dict[str, List[float]] = {k: [] for k in (
        "flooded_ha", "rain_mm_hr", "pe_mm_hr", "infil_mm_hr", "mean_depth_m",
        "outflow_total_cms", "inflow_total_cms", "gauge_cms", "watershed_outflow_cms")}
    frames: List[np.ndarray] = []
    frame_times: List[float] = []
    initial_volume = float(state.h.sum(dtype=torch.float64).item()) * dx * dx

    def capture(t_s: float) -> None:
        u, v = _velocity(state.h, state.qx, state.qy)
        frame = torch.stack([state.h, u, v]).float().cpu().numpy()
        frame_times.append(t_s)
        (frames.append(frame) if sink is None else sink(t_s, frame))

    t0 = time.time()
    t_s, last_frame_t, cap_hits, n_substeps = 0.0, -1e9, 0, 0
    acc_prev = np.zeros(8)
    for i, P in enumerate(rain):
        t_target = t_s + cfg.dt_s
        _set_forcing(forcing, float(P), t_target, inflow, t_s)
        n_sub = 0
        while True:
            for _ in range(block):
                step(state, grid, forcing, kern)
            n_sub += block
            t_s = float(state.t.item())
            if t_s >= t_target - 1e-9 or n_sub >= cfg.substep_cap:
                break
        cap_hits += n_sub >= cfg.substep_cap
        acc = np.array(state.acc.tolist())
        d, w = acc - acc_prev, max(t_s - (t_target - cfg.dt_s), 1e-30)
        acc_prev = acc
        n_substeps = int(round(acc[6]))

        h = state.h
        deep = h[torch.as_tensor(valid, device=device) & (h > FLOODED_DEPTH_THR_M)]
        infil_mm_hr = d[2] / max(n_valid, 1) / w * 3.6e6
        series["flooded_ha"].append(deep.numel() * cell_ha)
        series["rain_mm_hr"].append(P * 3.6e6)
        series["pe_mm_hr"].append(P * 3.6e6 - infil_mm_hr)
        series["infil_mm_hr"].append(infil_mm_hr)
        series["mean_depth_m"].append(float(deep.mean().item()) if deep.numel() else 0.0)
        series["outflow_total_cms"].append(d[0] / w)
        series["inflow_total_cms"].append(d[1] / w)
        if probes.gauge_rc is not None:
            series["gauge_cms"].append(d[4] / w)
        if probes.watershed_mask is not None:
            series["watershed_outflow_cms"].append(d[5] / w)

        if t_s - last_frame_t >= frame_interval_s:
            capture(t_s)
            last_frame_t = t_s
        if PROGRESS_LINES:
            print(f"PROGRESS hydro {i + 1}/{len(rain)}", flush=True)
        if verbose and i % 360 == 0:
            print(f"  t={t_s / 3600:6.1f}h  rain={P * 3.6e6:6.1f}mm/hr  "
                  f"h_max={float(h.max().item()):.3f}m  flooded={series['flooded_ha'][-1]:8.1f}ha  "
                  f"[{(i + 1) / len(rain) * 100:3.0f}% {time.time() - t0:.0f}s]", flush=True)

    if not frame_times or frame_times[-1] < t_s - 1e-9:
        capture(t_s)
    if device.type == "cuda":
        torch.cuda.synchronize()

    area = dx * dx
    acc = np.array(state.acc.tolist())
    mass = MassBalance(
        rain=float(np.sum(np.asarray(rain, dtype=np.float64))) * cfg.dt_s * n_valid * area,
        initial=initial_volume,
        inflow=float(acc[1]),
        created=float(acc[7]) * area,
        infiltrated=float(state.cum.sum(dtype=torch.float64).item()) * area,
        abstracted=float(state.abstracted.sum(dtype=torch.float64).item()) * area,
        stored=float(state.h.sum(dtype=torch.float64).item()) * area,
        outflow=float(acc[0]),
    )
    u, v = _velocity(state.h, state.qx, state.qy)
    return Result(
        h_final=state.h.cpu().numpy(), h_max=state.h_max.cpu().numpy(),
        cum_infil=state.cum.cpu().numpy(), qx_final=state.qx.cpu().numpy(),
        qy_final=state.qy.cpu().numpy(), u_final=u.cpu().numpy(), v_final=v.cpu().numpy(),
        frames=frames, frame_times_s=frame_times,
        series={k: np.asarray(v) for k, v in series.items() if v}, mass=mass,
        n_substeps=n_substeps, substep_cap_hits=cap_hits, wall_s=time.time() - t0,
        device=str(device),
        soil_state=None if state.bank is None else state.bank.double().cpu().numpy(),
    )
