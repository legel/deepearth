"""The v1 NumPy solver, verbatim, as the reference for `test_parity`.

Rain falls, some infiltrates against a finite soil store, the rest routes downhill under
gravity and Manning friction and leaves through open domain boundaries.

The numerics here are load-bearing and were each established by measurement, not preference.
Four defects in an earlier version of this loop are called out at their fix sites, because all
four produced plausible-looking output while being wrong:

  * the clock advanced by the forcing interval while the physics integrated a shorter
    CFL-limited step, so a "72-hour" run delivered 7-11 % of the storm;
  * infiltration was charged at the Horton *capacity* rate regardless of available rain, so
    the soil profile filled on paper without absorbing anything;
  * `CFL_ALPHA = 0.30` was unstable once water accumulated, giving a -517.8 % mass residual
    and 9 m depths oscillating under zero rainfall;
  * the final-frame guard could capture no end-of-run frame, which broke any mass balance
    measured against the last frame and manufactured a convincing -12.6 % "structural error".

Mass balance is the diagnostic that found all four, so `simulate` computes it every run rather
than leaving it to callers.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from physics import G, MANNING_EXP, MIN_DEPTH, FLOODED_DEPTH_THR_M

Field = Union[float, np.ndarray]


@dataclass
class SolverConfig:
    """Numerical settings for one run.

    Attributes:
        dx: Cell size [m].
        dt_s: Hyetograph interval [s]. Each interval is integrated with as many CFL-limited
            sub-steps as it takes; this is the forcing resolution, not the physics timestep.
        cfl_alpha: CFL safety factor. 0.15 -- 0.30 is unstable on this terrain once water
            accumulates, and 0.15 agrees with 0.05 to four figures, so the solution is
            timestep-converged here and tightening further only buys sub-steps.
        manning_n: Scalar Manning's n [s/m^(1/3)] for flatwoods grassland and mixed cover,
            used when `Surface.manning_n` is None.
        frame_interval_min: Spacing of saved depth snapshots [min].
        ponded_infiltration: Draw infiltration from standing depth after routing rather than
            from the instantaneous rain rate, so ponded water keeps soaking in once rain
            stops. False restores the rainfall-limited formulation for comparison.
        substep_cap: Guard against dt collapsing. Exceeding it is reported, never hung on.
    """

    dx: float
    dt_s: float
    cfl_alpha: float = 0.15
    manning_n: float = 0.040
    frame_interval_min: float = 30.0
    ponded_infiltration: bool = True
    substep_cap: int = 20000


@dataclass
class Surface:
    """Static per-cell fields the solver reads.

    Attributes:
        z: Ground elevation [m], NaN outside the domain.
        f0: Horton initial infiltration capacity [m/s], scalar or per-cell.
        fc: Horton final infiltration capacity [m/s], scalar or per-cell.
        k: Horton decay constant [1/s], scalar or per-cell.
        max_deficit_m: Finite soil storage [m] per cell. Once cumulative infiltration reaches
            it, infiltration stops and further rain becomes runoff -- saturation excess, which
            dominates over infiltration excess on flat terrain with a shallow water table.
            None gives unbounded infiltration, which absorbs essentially any storm and is
            almost never what you want.
        manning_n: Per-cell Manning's n, averaged onto faces internally. None uses the scalar
            in `SolverConfig`.
        initial_h: Initial water depth [m]. None starts bone-dry, which is wrong for a
            perennial channel: a carved creek that was already carrying baseflow otherwise
            spends the storm filling 22 km of empty channel first.
    """

    z: np.ndarray
    f0: Field = 0.0
    fc: Field = 0.0
    k: Field = 0.0
    max_deficit_m: Optional[np.ndarray] = None
    manning_n: Optional[np.ndarray] = None
    initial_h: Optional[np.ndarray] = None

    @property
    def valid(self) -> np.ndarray:
        """True where the cell carries real elevation."""
        return np.isfinite(self.z)


@dataclass
class Probes:
    """Optional cross-sections to measure discharge through.

    Attributes:
        gauge_rc: (row, col) of the cell holding a streamgauge. Domain-boundary outflow is not
            what a gauge measures -- the gauge sits inside the domain, so boundary outflow
            charges the comparison for travel time out to the box edge. On site3 that alone is
            the difference between a +5 h and a +29 h lag against an observed +4.5 h.
        watershed_mask: True inside the delineated watershed. Net flux across the mask's own
            internal boundary is tracked, because the domain box is much larger than the
            hydrologically connected catchment.
    """

    gauge_rc: Optional[Tuple[int, int]] = None
    watershed_mask: Optional[np.ndarray] = None


@dataclass
class MassBalance:
    """Volumes [m^3] over the whole run. `residual_pct` is the correctness signal."""

    rain: float
    initial: float
    infiltrated: float
    stored: float
    outflow: float

    @property
    def residual_pct(self) -> float:
        """Unaccounted volume as a percentage of everything that entered."""
        supplied = self.rain + self.initial
        if supplied <= 0.0:
            return 0.0
        closed = self.infiltrated + self.stored + self.outflow
        return (supplied - closed) / supplied * 100.0


@dataclass
class Result:
    """Everything one run produces."""

    h_final: np.ndarray
    h_max: np.ndarray
    cum_infil: np.ndarray
    frames: List[np.ndarray]
    frame_times_min: List[float]
    series: Dict[str, np.ndarray]
    mass: MassBalance
    n_substeps: int
    substep_cap_hits: int
    wall_s: float


def horton_rate(t_s: float, f0: Field, fc: Field, k: Field) -> Field:
    """Horton infiltration capacity [m/s] at time `t_s`, scalar or per-cell."""
    return fc + (f0 - fc) * np.exp(-k * t_s)


def _face_roughness(
    cfg: SolverConfig, manning_n: Optional[np.ndarray], shape: Tuple[int, int]
) -> Tuple[Field, Field]:
    """Squared Manning's n at x- and y-faces.

    Friction acts on flux across a cell boundary, so it needs the roughness the water actually
    crosses; averaging the two adjacent cells is the standard face reconstruction.

    A uniform array is not bit-identical to the equivalent scalar, and the reason is numpy
    casting rather than physics: a float64 scalar times a float32 array stays float32, while a
    float64 array promotes. Measured over 200 steps on a 60x60 grid the difference is 8.9e-08 m
    in depth. The scalar branch is kept exactly as-is so baselines are the untouched path.
    """
    if manning_n is None:
        return cfg.manning_n ** 2, cfg.manning_n ** 2
    mn = np.asarray(manning_n, dtype=np.float64)
    assert mn.shape == shape, f"manning_n shape {mn.shape} != grid shape {shape}"
    return (0.5 * (mn[:, :-1] + mn[:, 1:])) ** 2, (0.5 * (mn[:-1, :] + mn[1:, :])) ** 2


def _mean_over(field: Field, valid: np.ndarray) -> float:
    """Domain mean of a rate over valid cells, accepting either a scalar or a per-cell array."""
    return float(np.mean(field[valid]) if isinstance(field, np.ndarray) else field)


def _watershed_signs(
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-face +1/-1/0 signs marking which way across a face counts as leaving the watershed.

    +1 where the lower-index cell is inside and the higher-index cell is outside, -1 the other
    way, 0 where both agree and the face is not on the boundary at all.
    """
    wm = mask.astype(bool)
    left, right = wm[:, :-1], wm[:, 1:]
    sx = np.where(left & ~right, 1.0, np.where(~left & right, -1.0, 0.0)).astype(np.float32)
    top, bottom = wm[:-1, :], wm[1:, :]
    sy = np.where(top & ~bottom, 1.0, np.where(~top & bottom, -1.0, 0.0)).astype(np.float32)
    return sx, sy


def simulate(
    surface: Surface,
    rain: Sequence[float],
    cfg: SolverConfig,
    probes: Probes = Probes(),
    verbose: bool = True,
) -> Result:
    """Integrate the storm.

    Args:
        surface: Terrain and soil fields.
        rain: Rainfall rate [m/s], one entry per `cfg.dt_s` interval.
        cfg: Numerical settings.
        probes: Optional gauge cell and watershed mask.
        verbose: Print progress every 360 intervals.

    Returns:
        A `Result` whose `mass.residual_pct` should sit within a few thousandths of a percent.
    """
    z, valid = surface.z, surface.valid
    nrows, ncols = z.shape
    dx, dt_s = cfg.dx, cfg.dt_s
    assert len(rain) > 0, "rain series is empty"

    z_work = z.copy()
    z_work[~valid] = np.nanmax(z_work) + 100.0

    h = (np.zeros_like(z_work) if surface.initial_h is None
         else surface.initial_h.astype(np.float32).copy())
    h[~valid] = 0.0
    initial_volume = float(h.sum()) * dx * dx

    qx = np.zeros((nrows, ncols + 1), dtype=np.float32)
    qy = np.zeros((nrows + 1, ncols), dtype=np.float32)
    h_max = np.zeros_like(h)
    cum_infil = np.zeros_like(h)

    n2_x, n2_y = _face_roughness(cfg, surface.manning_n, z.shape)
    ws_sx, ws_sy = (_watershed_signs(probes.watershed_mask)
                    if probes.watershed_mask is not None else (None, None))

    f0, fc, k = surface.f0, surface.fc, surface.k
    cell_ha = dx * dx / 1e4
    frame_interval_s = cfg.frame_interval_min * 60.0

    series: Dict[str, List[float]] = {
        "flooded_ha": [], "rain_mm_hr": [], "pe_mm_hr": [], "infil_mm_hr": [],
        "mean_depth_m": [], "outflow_south_cms": [], "outflow_total_cms": [],
        "gauge_cms": [], "watershed_outflow_cms": [],
    }
    frames: List[np.ndarray] = []
    frame_times: List[float] = []

    t0, t_s = time.time(), 0.0
    n_substeps = cap_hits = 0
    last_frame_t = -1e9
    outflow_volume = 0.0

    for step, P in enumerate(rain):
        t_target = t_s + dt_s
        # Each per-interval series is a RATE integrated downstream against a dt_s-spaced axis,
        # so it must be this interval's time-weighted mean, not whatever the last sub-step left.
        acc = dict(south=0.0, total=0.0, gauge=0.0, ws=0.0, pe=0.0, inf=0.0)
        sub_elapsed = 0.0
        n_sub = 0

        # Advance to the interval's end in as many CFL-limited sub-steps as it takes. Taking one
        # short step and then advancing the clock by the full interval is what delivered 7-11 %
        # of the storm; the cost of doing it correctly is roughly dt_s/dt more steps.
        while t_s < t_target - 1e-9 and n_sub < cfg.substep_cap:
            h_peak = float(h.max())
            dt = min(dt_s, cfg.cfl_alpha * dx / np.sqrt(G * h_peak)) if h_peak > MIN_DEPTH else dt_s
            # Never overshoot: the last sub-step lands exactly on t_target, so each interval
            # receives exactly dt_s of its own rain rate and the clock stays aligned with forcing.
            dt = min(dt, t_target - t_s)

            inf = horton_rate(t_s, f0, fc, k)
            if surface.max_deficit_m is not None:
                remaining = np.maximum(surface.max_deficit_m - cum_infil, 0.0)
                inf = np.minimum(inf, remaining / dt)
            if not cfg.ponded_infiltration:
                Pe = np.maximum(P - inf, 0.0)

            eta = z_work + h

            hf_x = np.maximum(eta[:, 1:], eta[:, :-1]) - np.maximum(z_work[:, 1:], z_work[:, :-1])
            hf_x = np.maximum(hf_x, 0.0)
            num_x = qx[:, 1:-1] - G * hf_x * dt * (eta[:, 1:] - eta[:, :-1]) / dx
            den_x = 1.0 + G * dt * n2_x * np.abs(qx[:, 1:-1]) / (hf_x ** MANNING_EXP + 1e-10)
            qx[:, 1:-1] = np.where(hf_x > MIN_DEPTH, num_x / den_x, 0.0)
            # Froude limiter: cap unit discharge to subcritical at each face, which is what
            # keeps steep road embankments from going unstable.
            cap_x = 0.9 * hf_x * np.sqrt(G * np.maximum(hf_x, MIN_DEPTH))
            qx[:, 1:-1] = np.clip(qx[:, 1:-1], -cap_x, cap_x)

            hf_y = np.maximum(eta[1:, :], eta[:-1, :]) - np.maximum(z_work[1:, :], z_work[:-1, :])
            hf_y = np.maximum(hf_y, 0.0)
            num_y = qy[1:-1, :] - G * hf_y * dt * (eta[1:, :] - eta[:-1, :]) / dx
            den_y = 1.0 + G * dt * n2_y * np.abs(qy[1:-1, :]) / (hf_y ** MANNING_EXP + 1e-10)
            qy[1:-1, :] = np.where(hf_y > MIN_DEPTH, num_y / den_y, 0.0)
            cap_y = 0.9 * hf_y * np.sqrt(G * np.maximum(hf_y, MIN_DEPTH))
            qy[1:-1, :] = np.clip(qy[1:-1, :], -cap_y, cap_y)

            # Open boundaries: the domain is embedded in a larger watershed, so water leaves at
            # every edge and none enters. qx > 0 is eastward, qy > 0 is southward.
            qx[:, 0] = np.minimum(qx[:, 1], 0.0)
            qx[:, -1] = np.maximum(qx[:, -2], 0.0)
            qy[0, :] = np.minimum(qy[1, :], 0.0)
            qy[-1, :] = np.maximum(qy[-2, :], 0.0)

            # q is unit discharge [m^2/s], so a face's volumetric rate is q * dx. All four made
            # positive-as-leaving regardless of each edge's own sign convention.
            out_south = float(qy[-1, :].sum()) * dx
            out_total = (float(-qx[:, 0].sum()) * dx + float(qx[:, -1].sum()) * dx
                         + float(-qy[0, :].sum()) * dx + out_south)

            flux_div = dt / dx * (qx[:, :-1] - qx[:, 1:] + qy[:-1, :] - qy[1:, :])

            if cfg.ponded_infiltration:
                # All rainfall reaches the surface; infiltration then draws from whatever depth
                # is standing. Under the rainfall-limited form, 138 of 206 mm of capacity was
                # structurally unreachable once rain stopped even though the profile had room.
                h += flux_div + dt * P
                h = np.maximum(h, 0.0)
                h[~valid] = 0.0
                inf_amount = np.minimum(inf * dt, h)
                h -= inf_amount
                h = np.maximum(h, 0.0)
            else:
                h += flux_div + dt * Pe
                h = np.maximum(h, 0.0)
                h[~valid] = 0.0
                inf_amount = np.minimum(inf, P) * dt

            cum_infil += inf_amount
            h_max = np.maximum(h_max, h)

            inf_applied = inf_amount / dt
            pe_rate = P - inf_applied

            t_s += dt
            sub_elapsed += dt
            n_sub += 1
            n_substeps += 1

            acc["south"] += out_south * dt
            acc["total"] += out_total * dt
            acc["pe"] += _mean_over(pe_rate, valid) * dt
            acc["inf"] += _mean_over(inf_applied, valid) * dt
            if probes.gauge_rc is not None:
                gr, gc = probes.gauge_rc
                gqx = 0.5 * (qx[gr, gc] + qx[gr, gc + 1])
                gqy = 0.5 * (qy[gr, gc] + qy[gr + 1, gc])
                acc["gauge"] += float(np.hypot(gqx, gqy)) * dx * dt
            if probes.watershed_mask is not None:
                acc["ws"] += float((ws_sx * qx[:, 1:-1]).sum() * dx
                                   + (ws_sy * qy[1:-1, :]).sum() * dx) * dt

        if n_sub >= cfg.substep_cap:
            cap_hits += 1
        w = sub_elapsed if sub_elapsed > 0 else 1.0
        outflow_volume += acc["total"]

        wet = h[valid]
        deep = wet[wet > FLOODED_DEPTH_THR_M]
        series["flooded_ha"].append(len(deep) * cell_ha)
        series["outflow_south_cms"].append(acc["south"] / w)
        series["outflow_total_cms"].append(acc["total"] / w)
        series["rain_mm_hr"].append(P * 3600 * 1000)
        series["pe_mm_hr"].append(acc["pe"] / w * 3600 * 1000)
        series["infil_mm_hr"].append(acc["inf"] / w * 3600 * 1000)
        series["mean_depth_m"].append(float(deep.mean()) if len(deep) else 0.0)
        if probes.gauge_rc is not None:
            series["gauge_cms"].append(acc["gauge"] / w)
        if probes.watershed_mask is not None:
            series["watershed_outflow_cms"].append(acc["ws"] / w)

        if t_s - last_frame_t >= frame_interval_s:
            frames.append(h.copy().astype(np.float32))
            frame_times.append(t_s / 60.0)
            last_frame_t = t_s

        if verbose and step % 360 == 0:
            print(f"  t={t_s/3600:6.1f}h  rain={series['rain_mm_hr'][-1]:6.1f}mm/hr  "
                  f"h_max={float(h.max()):.3f}m  flooded={series['flooded_ha'][-1]:8.1f}ha  "
                  f"[{(step+1)/len(rain)*100:3.0f}% {time.time()-t0:.0f}s]")

    # Always land a frame on the true end of the run. A guard that skipped it whenever the run
    # ended less than half an interval after the last periodic frame left frames[-1] sampled
    # before the end, which silently breaks any mass balance measured against it.
    if (not frame_times) or frame_times[-1] < t_s / 60.0 - 1e-9:
        frames.append(h.copy().astype(np.float32))
        frame_times.append(t_s / 60.0)

    cell_area = dx * dx
    mass = MassBalance(
        rain=float(np.sum(rain)) * dt_s * int(valid.sum()) * cell_area,
        initial=initial_volume,
        infiltrated=float(cum_infil.sum()) * cell_area,
        stored=float(h.sum()) * cell_area,
        outflow=outflow_volume,
    )
    return Result(
        h_final=h,
        h_max=h_max,
        cum_infil=cum_infil,
        frames=frames,
        frame_times_min=frame_times,
        series={k: np.asarray(v) for k, v in series.items() if v},
        mass=mass,
        n_substeps=n_substeps,
        substep_cap_hits=cap_hits,
        wall_s=time.time() - t0,
    )
