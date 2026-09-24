"""Physics checks on synthetic scenes, in physical units.

Each returns the numbers a test asserts on and `cli.py verify` records, so the receipt in
`docs/` and the test suite are one code path.
"""

from typing import Dict, Optional, Sequence

import numpy as np

import domain
from domain import Grid, Scene
from forcing import LogProfile
from solver import Result, SolverConfig, solve

WEST = 270.0
"""Direction a westerly blows from: flow along +x."""

Z0 = 0.03
"""Roughness length of open ground in every check [m]."""


def profile(u_ref: float = 5.0, z_ref: float = 10.0) -> LogProfile:
    return LogProfile.from_reference(u_ref, z_ref, Z0)


def grid(dx: float = 1.0, nx: int = 64, ny: int = 32, nz: int = 24) -> Grid:
    return Grid.stretched(dx, nx, ny, nz, dx, 1.06)


def _conservation(res: Result) -> Dict[str, float]:
    return {"divergence_rel": res.divergence_rel, "divergence_max": res.divergence_max,
            "divergence_max_1_s": res.divergence_max_1_s,
            "flux_balance": res.flux_balance, "flux_in_m3_s": res.flux_in_m3_s,
            "poisson_iterations_max": max(res.poisson_iterations), "wall_s": res.wall_s}


def flat_profile(cfg: SolverConfig, g: Optional[Grid] = None) -> Dict[str, float]:
    """An unobstructed domain must return the inflow profile unchanged."""
    g = g or grid()
    p = profile()
    res = solve(domain.flat(g, Z0), p, WEST, cfg)
    expected = p.speed(g.zc)[:, None, None]
    return {"max_profile_error_rel": float(np.abs(res.velocity[0] - expected).max() / expected.max()),
            "max_crossflow_rel": float(np.abs(res.velocity[1:]).max() / expected.max()),
            "max_change_rel": max(res.change) if res.change else 0.0, **_conservation(res)}


def cube_wake(cfg: SolverConfig, g: Optional[Grid] = None, size: float = 8.0) -> Dict[str, float]:
    """A cube must shed a wake with reversed flow and speed the flow up over its roof."""
    g = g or grid()
    p = profile()
    cx, cy = 2.5 * size, g.ny * g.dx / 2
    scene = domain.cube(g, size, centre=(cx, cy), z0=Z0)
    res = solve(scene, p, WEST, cfg)
    potential = solve(scene, p, WEST, SolverConfig(steps=0, device=cfg.device, dtype=cfg.dtype))
    u_h = float(p.speed(np.array([size]))[0])
    x, y, z = g.xc[None, None, :], g.yc[None, :, None], g.zc[:, None, None]
    wake = (x > cx + size / 2) & (x < cx + 2.5 * size) & (np.abs(y - cy) < size / 2) & (z < size)
    roof = (np.abs(x - cx) < size / 2) & (np.abs(y - cy) < size / 2) & (z > size) & (z < 2 * size)
    side = (np.abs(x - cx) < size / 2) & (np.abs(y - cy) > size / 2) & (np.abs(y - cy) < size) & (z < size)
    speedup = res.speed / p.speed(g.zc)[:, None, None]
    return {"wake_min_u_over_u_h": float(res.velocity[0][wake].min() / u_h),
            "wake_reversed_fraction": float((res.velocity[0][wake] < 0).mean()),
            "roof_max_speedup": float(speedup[roof].max()),
            "side_max_speedup": float(speedup[side].max()),
            "roof_max_speedup_projection_only": float(
                (potential.speed / p.speed(g.zc)[:, None, None])[roof].max()),
            "solid_max_speed": float(res.speed[scene.solid].max()),
            "vorticity_max_1_s": float(np.abs(res.vorticity).max()),
            "u_h_m_s": u_h, **_conservation(res)}


def canopy_lai(cfg: SolverConfig, g: Optional[Grid] = None, size: float = 8.0,
               lai: Sequence[float] = (0.5, 2.0, 8.0)) -> Dict[str, object]:
    """Speed at a canopy block's downwind face must fall monotonically with LAI."""
    g = g or grid()
    p = profile()
    cx, cy = 2.5 * size, g.ny * g.dx / 2
    ix = int((cx + size / 2) / g.dx) - 1
    iy, iz = int(cy / g.dx), int(np.searchsorted(g.zc, size / 2))
    x, y, z = g.xc[None, None, :], g.yc[None, :, None], g.zc[:, None, None]
    inside = (z < size) & (np.abs(y - cy) < size / 2) & (np.abs(x - cx) < size / 2)
    open_speed = float(solve(domain.flat(g, Z0), p, WEST, cfg).speed[inside].mean())
    exit_speed, inside_speed = [], []
    for value in lai:
        res = solve(domain.porous_block(g, size, value, centre=(cx, cy), z0=Z0), p, WEST, cfg)
        exit_speed.append(float(res.speed[iz, iy, ix]))
        inside_speed.append(float(res.speed[inside].mean()))
    ratio = np.asarray(exit_speed) / float(p.speed(np.array([g.zc[iz]]))[0])
    return {"lai": list(lai), "exit_speed_ratio": ratio.tolist(),
            "monotonic": bool(np.all(np.diff(ratio) < 0)),
            "mean_speed_in_canopy_m_s": inside_speed,
            "mean_speed_same_cells_no_canopy_m_s": open_speed,
            "canopy_over_open": [s / open_speed for s in inside_speed]}


def rotation(cfg: SolverConfig, n: int = 32, nz: int = 16, size: float = 6.0) -> Dict[str, float]:
    """A westerly over a cube and a southerly over the same cube are one field rotated 90 deg."""
    g = grid(1.0, n, n, nz)
    p = profile()
    scene = domain.cube(g, size, z0=Z0)
    west, south = solve(scene, p, WEST, cfg), solve(scene, p, 180.0, cfg)
    expected = west.speed[:, ::-1, :].transpose(0, 2, 1)
    return {"max_speed_difference_rel": float(np.abs(south.speed - expected).max() / west.speed.max())}


def orientation(cfg: SolverConfig, n: int = 96, nz: int = 24, radius: float = 4.0,
                lai: float = 4.0, dx: float = 0.5) -> Dict[str, float]:
    """A porous tower's wake must not depend on whether the wind crosses the grid at 0 or 45 deg.

    The tower is rasterised by area coverage and the wake is sampled by interpolation, so the
    comparison measures the solver rather than the pixelation of either.
    """
    from scipy.ndimage import map_coordinates

    g = grid(dx, n, n, nz)
    p = profile()
    centre = (n * dx / 2, n * dx / 2)
    scene = domain._empty(g, Z0, "porous cylinder")
    cover = domain._footprint(g, centre, radius, round_=True, subsamples=8)
    scene.sink[:] = cover[None] * (g.zc < 2 * radius)[:, None, None] * 0.2 * lai / (2 * radius)

    def deficit(direction: float) -> float:
        res = solve(scene, p, direction, cfg)
        e = np.array([np.sin(np.radians(direction)), np.cos(np.radians(direction))])
        z = radius / 2
        k = np.interp(z, g.zc, np.arange(g.nz))
        points = np.array([np.array(centre) - s * radius * e for s in (2.0, 2.5, 3.0)])
        coords = [np.full(len(points), k), points[:, 1] / g.dx - 0.5, points[:, 0] / g.dx - 0.5]
        speed = map_coordinates(res.speed, coords, order=1)
        return float((speed / p.speed(np.array([z]))[0]).mean())

    axis, diagonal = deficit(WEST), deficit(225.0)
    return {"wake_speed_ratio_axis": axis, "wake_speed_ratio_diagonal": diagonal,
            "difference_rel": abs(axis - diagonal) / axis}


def ridge_speedup(cfg: SolverConfig, g: Optional[Grid] = None, height: float = 6.0,
                  half_width: float = 16.0) -> Dict[str, float]:
    """Speed-up over a ridge crest relative to the inflow at the same height above ground."""
    g = g or grid(1.0, 64, 16, 24)
    p = profile()
    res = solve(domain.ridge(g, height, half_width, z0=Z0), p, WEST, cfg)
    ix, iy = g.nx // 2, g.ny // 2
    k = int(np.searchsorted(g.zc, height + 2.0))
    above = g.zc[k] - height
    return {"crest_speedup": float(res.speed[k, iy, ix] / p.speed(np.array([above]))[0]),
            "height_above_crest_m": float(above), **_conservation(res)}


def linearity(cfg: SolverConfig, g: Optional[Grid] = None, size: float = 8.0,
              speeds: Sequence[float] = (2.0, 5.0, 10.0),
              spacings_deg: Sequence[float] = (22.5, 10.0)) -> Dict[str, object]:
    """One heading at three speeds must be one field scaled, and the mean of two headings a
    spacing apart is compared with a solve midway between them, for each spacing."""
    g = g or grid()
    cx, cy = 2.5 * size, g.ny * g.dx / 2
    scene = domain.cube(g, size, centre=(cx, cy), z0=Z0)
    fields = {s: solve(scene, LogProfile.from_reference(s, 10.0, Z0), WEST, cfg).velocity / s
              for s in speeds}
    ref = fields[speeds[1]]
    scale = float(np.abs(ref).max())
    deviation = {f"{s:g}": float(np.abs(fields[s] - ref).max() / scale) for s in speeds}
    p = profile()
    right = ref * speeds[1]
    midway = {}
    for spacing in spacings_deg:
        left = solve(scene, p, WEST - spacing, cfg).velocity
        mid = solve(scene, p, WEST - spacing / 2, cfg).velocity
        midway[f"{spacing:g}"] = float(np.abs(mid - 0.5 * (left + right)).max() / np.abs(mid).max())
    return {"speeds_m_s": list(speeds), "deviation_from_linear_rel": deviation,
            "max_deviation_from_linear_rel": max(deviation.values()),
            "midway_heading_interpolation_error_rel": midway}


def vorticity_of_rotation(g: Optional[Grid] = None, omega: float = 0.1) -> Dict[str, float]:
    """The curl of solid-body rotation about z at rate omega is 2 omega everywhere."""
    import torch

    from solver import Model

    g = g or grid(1.0, 16, 16, 8)
    cfg = SolverConfig(steps=0)
    model = Model(domain.flat(g, Z0), profile(), WEST, cfg)
    x, y = torch.as_tensor(g.xc - g.nx / 2), torch.as_tensor(g.yc - g.ny / 2)
    u = torch.stack([(-omega * y[:, None]).expand(g.ny, g.nx), (omega * x[None, :]).expand(g.ny, g.nx),
                     torch.zeros(g.ny, g.nx)])[:, None].expand(3, g.nz, g.ny, g.nx).to(torch.float64)
    w = model.vorticity(u.contiguous())
    return {"omega_z_over_2omega": float(w[2].mean() / (2 * omega)),
            "omega_z_spread": float((w[2].max() - w[2].min()) / (2 * omega)),
            "omega_xy_max": float(w[:2].abs().max())}


ALL = {"flat": flat_profile, "cube": cube_wake, "canopy": canopy_lai, "rotation": rotation,
       "orientation": orientation, "ridge": ridge_speedup, "linearity": linearity}
