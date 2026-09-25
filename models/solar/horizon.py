"""Horizon profiles by ray march over a height field, and the sky-view factor they leave, batched in PyTorch.

From each observer, rays leave in `bins` azimuth wedges (`subrays` each) at geometrically growing distances; a
wedge's horizon is the highest elevation angle any of its samples subtends. Where columns transmit light (a canopy
of transmittance t), a ray entering such a run above the opaque horizon is attenuated once by the column it enters.
The sky-view factor of a surface with unit normal n is the cosine-weighted visible sky:

    V = (1 / pi) sum over wedges of the integral from the zenith to the horizon of max(n.omega, 0) sin(theta) dtheta dphi

1 for an open horizontal surface and 1/2 at the foot of an infinite wall.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

Tensor = torch.Tensor


@dataclass(frozen=True)
class Grid:
    """A square lattice: `cell` [m], nx by ny cells, lower-left corner (x0, y0)."""

    cell: float
    nx: int
    ny: int
    x0: float = 0.0
    y0: float = 0.0


@dataclass
class MarchConfig:
    """bins: azimuth wedges; subrays: rays per wedge; r0: first sample [m]; rmax: last [m]; growth: ratio between
    successive sample distances; chunk: samples per tensor operation."""

    bins: int = 64
    subrays: int = 3
    r0: float = 1.0
    rmax: float = 280.0
    growth: float = 1.035
    chunk: int = 32


def subray_azimuths(cfg: MarchConfig) -> Tensor:
    """Azimuths [rad] of every subray, wedge-major [bins * subrays], clockwise from north."""
    b = torch.arange(cfg.bins, dtype=torch.float64)[:, None]
    j = (torch.arange(cfg.subrays, dtype=torch.float64)[None, :] + 0.5) / cfg.subrays
    return torch.deg2rad((b + j) * (360.0 / cfg.bins)).flatten()


def _samples(z: Tensor, tau: Optional[Tensor], grid: Grid, obs: Tensor, r: Tensor, sa: Tensor, ca: Tensor,
             rmax: float) -> Tuple[Tensor, Optional[Tensor]]:
    """Tangent of each sample's elevation and its column's transmittance; -1 and 0 off the grid."""
    fx = (obs[:, 0, None, None] + r * sa - grid.x0) / grid.cell
    fy = (obs[:, 1, None, None] + r * ca - grid.y0) / grid.cell
    ok = (fx >= 0) & (fx < grid.nx) & (fy >= 0) & (fy < grid.ny) & (r <= rmax)
    idx = (fy.long() * grid.nx + fx.long()).clamp_(0, grid.ny * grid.nx - 1)
    t = ((z.flatten()[idx] - obs[:, 2, None, None]) / r).masked_fill_(~ok, -1.0)
    return t, (tau.flatten()[idx].masked_fill_(~ok, 0.0) if tau is not None else None)


def march(z: Tensor, grid: Grid, obs: Tensor, cfg: MarchConfig = MarchConfig(),
          tau: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
    """Horizons for observers `obs` (x, y, z) [M, 3] over obstacle heights `z` [ny, nx] (-inf where rays pass).

    Returns:
        (horizon, opaque horizon [deg], transmittance of the band between them), each [M, bins].
    """
    az = subray_azimuths(cfg).to(z.device, torch.float32)
    sa, ca = torch.sin(az)[None, :, None], torch.cos(az)[None, :, None]
    n = int(math.ceil(math.log(cfg.rmax / cfg.r0) / math.log(cfg.growth))) + 1
    radii = cfg.r0 * cfg.growth ** torch.arange(n, device=z.device, dtype=torch.float32)
    radii = radii[None, None, :].expand(obs.shape[0], 1, n)
    best = torch.full((obs.shape[0], az.numel()), -1.0, device=z.device)
    opaque = best.clone()
    for j in range(0, n, cfg.chunk):
        t, ti = _samples(z, tau, grid, obs, radii[..., j:j + cfg.chunk], sa, ca, cfg.rmax)
        best = torch.maximum(best, t.amax(-1))
        opaque = torch.maximum(opaque, (t if ti is None else torch.where(ti > 0, -1.0, t)).amax(-1))
    band = torch.zeros_like(best)
    if tau is not None:
        inside = torch.zeros_like(best, dtype=torch.bool)[:, :, None]
        for j in range(0, n, cfg.chunk):
            t, ti = _samples(z, tau, grid, obs, radii[..., j:j + cfg.chunk], sa, ca, cfg.rmax)
            crossed = (ti > 0) & (t > opaque[:, :, None])
            entered = crossed & ~torch.cat([inside, crossed[:, :, :-1]], dim=2)
            band += torch.where(entered, torch.log(ti.clamp_min(1e-6)), 0.0).sum(-1)
            inside = crossed[:, :, -1:]
    shape = (-1, cfg.bins, cfg.subrays)
    deg = lambda x: torch.rad2deg(torch.atan(x.clamp_min(0.0))).view(shape).mean(-1)  # noqa: E731
    return deg(best), deg(opaque), torch.exp(band).view(shape).mean(-1)


def sky_view_bins(horizon_deg: Tensor, normal: Tensor, n_theta: int = 64) -> Tensor:
    """Each wedge's share of the cosine-weighted visible sky for a unit normal [M, 3]; [M, bins], summing to V."""
    m, b = horizon_deg.shape
    phi = (torch.arange(b, device=horizon_deg.device, dtype=torch.float32) + 0.5) * (2 * math.pi / b)
    tmax = (math.pi / 2 - torch.deg2rad(horizon_deg))[:, :, None]
    u = (torch.arange(n_theta, device=horizon_deg.device, dtype=torch.float32) + 0.5) / n_theta
    theta = tmax * u
    st, ct = torch.sin(theta), torch.cos(theta)
    nd = (normal[:, 0, None, None] * st * torch.sin(phi)[None, :, None]
          + normal[:, 1, None, None] * st * torch.cos(phi)[None, :, None]
          + normal[:, 2, None, None] * ct)
    return (nd.clamp_min(0.0) * st).sum(-1) * tmax[:, :, 0] / n_theta * (2 / b)


def sky_view(horizon_deg: Tensor, normal: Tensor, opaque_deg: Optional[Tensor] = None,
             band: Optional[Tensor] = None) -> Tensor:
    """Sky-view factor [M], seeing through the band between the two horizons with its transmittance."""
    svf = sky_view_bins(horizon_deg, normal)
    if opaque_deg is not None:
        svf = svf + band * (sky_view_bins(opaque_deg, normal) - svf)
    return svf.sum(-1)
