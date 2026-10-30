"""Wind exposure per point from the unit basis and a measured hourly record (NET_EXPOSURE.md s.2).

The solver is linear in inflow speed, so a point's hourly speed is U_ref(t) m(theta(t)), where m is the
point's unit speed for that direction. Hours are grouped by heading pair and blend fraction (LEVELS a pair):
each group's unit speed and local direction are computed once per point, and every metric follows from per-group
sums and sorted speeds. The basis holds 16 headings (22.5 degrees, blended linearly as before) or, from 2026-10-11,
8 (45 degrees, blended polar, to cut the wind's cost 10 to 30x; 0.5 to 2.8 % median against 16, scripts/wind/heading_cut.py).
Its heading count is read from the basis itself, so a held 16-heading basis still blends as it was solved.
PyTorch, any device.
"""

import math
from typing import Dict, Optional, Sequence

import numpy as np
import torch

SECTORS = 16                      # the prevailing-direction histogram's sectors: the output's, not the basis's
SECTOR_STEP = 360.0 / SECTORS
POLAR_BELOW = 16                  # a basis of fewer headings blends polar (speed blended, direction along the vector)
LEVELS = 16                       # blend fractions per heading pair
STRONG = 10.8                     # m/s hourly mean, Beaufort 6
GALE_GUST = 17.2                  # m/s gust, Beaufort 8
PEAK_C = 7.0                      # EN 1991-1-4 peak velocity: G = U sqrt(1 + 7 I)
KAPPA = 0.41
EULER = 0.5772156649


def groups(theta_deg: np.ndarray, headings: int) -> np.ndarray:
    """Hour -> group id = heading k * LEVELS + blend level on a basis of `headings`; NaN direction -> -1 (calm or
    missing)."""
    ok = ~np.isnan(theta_deg)
    x = np.where(ok, np.mod(theta_deg, 360.0) / (360.0 / headings), 0.0)
    k = np.floor(x).astype(int) % headings
    lvl = np.minimum(np.rint((x - np.floor(x)) * LEVELS).astype(int), LEVELS)
    k = np.where(lvl == LEVELS, (k + 1) % headings, k)
    lvl = np.where(lvl == LEVELS, 0, lvl)
    return np.where(ok, k * LEVELS + lvl, -1)


def unit_field(S: torch.Tensor, g: int) -> torch.Tensor:
    """The blended unit (u, v, w) at every point for group g. S: [headings, N, 3] (headings the wind blows FROM). On a
    basis of fewer than POLAR_BELOW headings the horizontal speed is the blend of the two speeds, its direction the
    vector blend's: a chord of two vectors 45 degrees apart runs up to 7.6 % short."""
    h = S.shape[0]
    k, lvl = divmod(g, LEVELS)
    a = lvl / LEVELS
    lo, hi = S[k].float(), S[(k + 1) % h].float()
    f = (1.0 - a) * lo + a * hi
    if h >= POLAR_BELOW or lvl == 0:
        return f
    s = (1.0 - a) * torch.linalg.vector_norm(lo[:, :2], dim=1) + a * torch.linalg.vector_norm(hi[:, :2], dim=1)
    n = torch.linalg.vector_norm(f[:, :2], dim=1)
    scale = torch.where(n > 1e-6, s / torch.clamp(n, min=1e-6), torch.zeros_like(n))
    return torch.cat([f[:, :2] * scale[:, None], f[:, 2:]], dim=1)


def year(S: torch.Tensor, uref: np.ndarray, theta: np.ndarray, gust_factor: torch.Tensor, open_speed: torch.Tensor) -> Dict[str, torch.Tensor]:
    """One year's metrics per point.

    Args:
        S: [headings, N, 3] unit basis per point (fp16 or fp32), 16 or 8 headings.
        uref: hourly reference speed at the basis reference (10 m over the inflow fetch), m/s; NaN missing.
        theta: hourly direction the wind blows from, degrees; NaN missing.
        gust_factor: [N] sqrt(1 + c I_p).
        open_speed: [N] the open-ground unit speed at each point's height (the inflow log profile, per 1 m/s).
    """
    dev = S.device
    n = S.shape[1]
    ok = ~np.isnan(uref) & ~np.isnan(theta)
    g_of = groups(np.where(ok, theta, np.nan), S.shape[0])
    hours = int(ok.sum())
    total = torch.zeros(n, device=dev)
    strong = torch.zeros(n, device=dev)
    gale = torch.zeros(n, device=dev)
    peak = torch.zeros(n, device=dev)
    shelter = torch.zeros(n, device=dev)
    hist = torch.zeros(SECTORS, n, device=dev)
    dmg = torch.zeros(2, n, device=dev)
    top = torch.zeros(2, n, device=dev)
    for g in np.unique(g_of[g_of >= 0]):
        u = torch.as_tensor(np.sort(uref[g_of == g]), device=dev, dtype=torch.float32)
        f = unit_field(S, int(g))
        m = torch.linalg.vector_norm(f[:, :2], dim=1)
        total += m * u.sum()
        # hours at or above a threshold: count of u >= threshold / m (u sorted ascending)
        strong += len(u) - torch.searchsorted(u, STRONG / torch.clamp(m, min=1e-6))
        gm = m * gust_factor
        over = len(u) - torch.searchsorted(u, GALE_GUST / torch.clamp(gm, min=1e-6))
        gale += over
        peak = torch.maximum(peak, gm * u[-1])
        shelter += (m / torch.clamp(open_speed, min=1e-6)) * len(u)
        # local direction the wind blows FROM at the point: opposite its velocity
        d = torch.remainder(torch.rad2deg(torch.atan2(-f[:, 0], -f[:, 1])), 360.0)
        b = torch.remainder(torch.round(d / SECTOR_STEP), SECTORS).long()
        hist.scatter_add_(0, b[None], torch.full((1, n), float(len(u)), device=dev))
        # damaging direction: gust-energy-weighted mean of hours past the gale gust (vector mean of G^2)
        e_over = (gm ** 2) * _sum_sq_above(u, GALE_GUST / torch.clamp(gm, min=1e-6))
        rad = torch.deg2rad(d)
        dmg[0] += e_over * torch.sin(rad)
        dmg[1] += e_over * torch.cos(rad)
        e_top = (gm * u[-1]) ** 2
        top[0] += e_top * torch.sin(rad)
        top[1] += e_top * torch.cos(rad)
    has_gale = dmg.abs().sum(0) > 0
    dvec = torch.where(has_gale[None], dmg, top)
    return {"mean_speed": total / max(hours, 1), "strong_hours": strong, "gale_gust_hours": gale, "peak_gust": peak,
            "shelter": shelter / max(hours, 1), "prevailing_deg": hist.argmax(0).float() * SECTOR_STEP,
            "damaging_deg": torch.remainder(torch.rad2deg(torch.atan2(dvec[0], dvec[1])), 360.0), "hours": torch.tensor(hours)}


def _sum_sq_above(u_sorted: torch.Tensor, thr: torch.Tensor) -> torch.Tensor:
    """Per point, the sum of u^2 over hours with u >= thr (u sorted ascending)."""
    c = torch.cumsum(torch.flip(u_sorted, [0]) ** 2, 0)
    idx = torch.searchsorted(u_sorted, thr)
    cnt = len(u_sorted) - idx
    return torch.where(cnt > 0, c[torch.clamp(cnt - 1, min=0)], torch.zeros_like(thr))


def return_gust(annual_peaks: torch.Tensor, period_years: float = 10.0) -> torch.Tensor:
    """Gumbel by moments over the record's annual maxima [years, N]: mu + y_T beta."""
    mean, std = annual_peaks.mean(0), annual_peaks.std(0, unbiased=True)
    beta = std * math.sqrt(6.0) / math.pi
    mu = mean - EULER * beta
    return mu - beta * math.log(-math.log(1.0 - 1.0 / period_years))


def gust_factor(height_m: torch.Tensor, z0_m: torch.Tensor, c: float = PEAK_C) -> torch.Tensor:
    """sqrt(1 + c I), I = 1 / ln(z / z0), with z at least 2 z0 (EN 1991-1-4 peak velocity)."""
    z = torch.maximum(height_m, 2.0 * z0_m)
    return torch.sqrt(1.0 + c / torch.log(z / z0_m))


def open_speed(height_m: torch.Tensor, z0_open: float, z_ref: float = 10.0) -> torch.Tensor:
    """Unit inflow log profile at each height over open ground: ln(z / z0) / ln(z_ref / z0)."""
    z = torch.clamp(height_m, min=2.0 * z0_open)
    return torch.log(z / z0_open) / math.log(z_ref / z0_open)


# ---------------------------------------------------------------- the reference speed from a tower

def obukhov(ustar: np.ndarray, h: np.ndarray, ta: np.ndarray, pa_kpa: np.ndarray) -> np.ndarray:
    """Obukhov length, m (sensible heat only)."""
    rho = pa_kpa * 1000.0 / (287.05 * (ta + 273.15))
    with np.errstate(divide="ignore", invalid="ignore"):
        return -(ustar ** 3) * rho * 1005.0 * (ta + 273.15) / (KAPPA * 9.81 * h)


def z0_by_sector(ws: np.ndarray, ustar: np.ndarray, wd: np.ndarray, h: np.ndarray, ta: np.ndarray, pa: np.ndarray,
                 z_m: float, d_m: float, sectors: int = 12, min_hours: int = 50) -> np.ndarray:
    """Fetch roughness per sector from near-neutral hours (|z/L| < 0.05, u* > 0.2): z0 = (z - d) exp(-kappa U / u*)."""
    L = obukhov(ustar, h, ta, pa)
    neutral = (np.abs((z_m - d_m) / L) < 0.05) & (ustar > 0.2) & ~np.isnan(ws) & ~np.isnan(wd)
    z0 = (z_m - d_m) * np.exp(-KAPPA * ws / ustar)
    sec = (np.floor(np.mod(wd, 360.0) / (360.0 / sectors)).astype(int)) % sectors
    out = np.full(sectors, np.nan)
    for s in range(sectors):
        sel = neutral & (sec == s)
        if sel.sum() >= min_hours:
            out[s] = float(np.median(z0[sel]))
    if np.isnan(out).all():
        return out
    return np.where(np.isnan(out), np.nanmedian(out), out)


def transfer(ws: np.ndarray, z_m: float, z0_tower: np.ndarray, d_tower: float, z0_site: float, d_site: float,
             z_blend: float = 60.0, z_ref: float = 10.0) -> np.ndarray:
    """Tower speed to the basis reference over the site's inflow fetch through the blending height (Wieringa 1986)."""
    ub = ws * np.log((z_blend - d_tower) / z0_tower) / np.log((z_m - d_tower) / z0_tower)
    return ub * np.log((z_ref - d_site) / z0_site) / np.log((z_blend - d_site) / z0_site)
