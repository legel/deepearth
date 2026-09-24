"""Green-Ampt infiltration with redistribution (GAR), two wetting fronts per cell, in PyTorch.

Ogden, F. L. and Saghafian, B. (1997), Green and Ampt infiltration with redistribution, J. Irrig. Drain. Eng.
123(5), 386-393, with the redistribution equation of Smith, Corradini and Melone (1993), Water Resour. Res. 29(1),
133-144. This is the single-layer case of the multi-front scheme NOAA's Next Generation Water Resources Modeling
Framework runs as LGAR (github.com/NOAA-OWP/LGAR-C). Capacity is Green and Ampt's (1911) with Mein and Larson's
(1973) ponding by construction: water the surface does not hold is supply-limited, standing water capacity-limited.
Soil hydraulics are Brooks and Corey (1964); parameters by texture from Rawls, Brakensiek and Miller (1983).

The state bank, per cell, carried through every sub-step and from one storm to the next (`BANK`):
    F1, theta1   the deep front: water F1 [m] above theta_i, at content theta1, down to Z1 = F1 / (theta1 - theta_i)
    F2, theta2   the surface front a later pulse starts over the redistributed deep front: water F2 above theta1,
                 down to Z2 = F2 / (theta2 - theta1)
    hiatus       1 after a sub-step in which nothing entered the cell

Equations, per sub-step dt, on the surface front (the deep one while there is no surface front):
    capacity             f = K_s (1 + S / F),   S = G(theta_b, theta_s) (theta_s - theta_b)   theta_b: content beneath
    infiltration         i = min(d, h, F_max - F1 - F2),   d - S ln(1 + d / (F + S)) = K_s dt   h: water on the cell
                         (d is Green-Ampt's exact ponded increment over the step, not f dt)
    redistribution       Z dtheta/dt = r - (K(theta) - K(theta_b)) - p K_s G(theta_b, theta) / Z    (no ponding)
                         p = 1.7 while nothing enters (r = 0), 1.0 while rain enters unponded; theta = theta_s ponded
    conductivity         K(theta) = K_s S_e^(3 + 2/lambda),   S_e = (theta - theta_r) / (theta_s - theta_r)
    capillary drive      G(theta_b, theta) = psi_f (S_e^c - S_eb^c) / (1 - S_eb^c),   c = 3 + 1/lambda

G is Morel-Seytoux and Khanji's (1974) effective drive, the integral of K/K_s over suction, whose Brooks-Corey form
grows as S_e^(3 + 1/lambda); it is scaled so G(theta_i, theta_s) is the texture's measured wetting-front suction psi_f.

A surface front starts when water returns after a hiatus faster than the deep front conducts it (r > K(theta1)), and
merges into the deep front, conserving F1 + F2, once it reaches the same depth or has drained to the same content. The
deep front holds still while a surface front sits on it. The redistribution ODE is stiff for a thin front, so it is
advanced linearly implicitly (Rosenbrock-Euler), stable at any sub-step the flow solver takes.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor

P_REDISTRIBUTE, P_INFILTRATE = 1.7, 1.0
"""Ogden and Saghafian's (1997) shape factor p: 1.7 during redistribution, 1.0 while water enters."""

F_MIN_M = 1e-6
"""Front water [m] below which a front does not exist."""

BANK = ("F1", "theta1", "F2", "theta2", "hiatus")
"""The rows of the per-cell state bank, a [5, rows, cols] tensor; hiatus is 1 after a sub-step nothing entered."""

# Rawls, Brakensiek and Miller (1983), Table 1: total porosity theta_s, residual theta_r, Brooks-Corey lambda
# (geometric mean), wetting-front suction psi_f [cm, geometric mean], saturated conductivity K_s [cm/h].
RAWLS_1983: Dict[str, Dict[str, float]] = {
    "sand":            {"theta_s": 0.437, "theta_r": 0.020, "lam": 0.592, "psi_f_cm": 4.95, "ks_cm_h": 11.78},
    "loamy sand":      {"theta_s": 0.437, "theta_r": 0.035, "lam": 0.474, "psi_f_cm": 6.13, "ks_cm_h": 2.99},
    "sandy loam":      {"theta_s": 0.453, "theta_r": 0.041, "lam": 0.322, "psi_f_cm": 11.01, "ks_cm_h": 1.09},
    "loam":            {"theta_s": 0.463, "theta_r": 0.027, "lam": 0.220, "psi_f_cm": 8.89, "ks_cm_h": 0.34},
    "silt loam":       {"theta_s": 0.501, "theta_r": 0.015, "lam": 0.211, "psi_f_cm": 16.68, "ks_cm_h": 0.65},
    "sandy clay loam": {"theta_s": 0.398, "theta_r": 0.068, "lam": 0.250, "psi_f_cm": 21.85, "ks_cm_h": 0.15},
    "clay loam":       {"theta_s": 0.464, "theta_r": 0.075, "lam": 0.194, "psi_f_cm": 20.88, "ks_cm_h": 0.10},
    "silty clay loam": {"theta_s": 0.471, "theta_r": 0.040, "lam": 0.151, "psi_f_cm": 27.30, "ks_cm_h": 0.10},
    "sandy clay":      {"theta_s": 0.430, "theta_r": 0.109, "lam": 0.168, "psi_f_cm": 23.90, "ks_cm_h": 0.06},
    "silty clay":      {"theta_s": 0.479, "theta_r": 0.056, "lam": 0.127, "psi_f_cm": 29.22, "ks_cm_h": 0.05},
    "clay":            {"theta_s": 0.475, "theta_r": 0.090, "lam": 0.131, "psi_f_cm": 31.63, "ks_cm_h": 0.03},
}


def usda_texture(sand_pct: float, clay_pct: float) -> str:
    """The USDA soil texture class of a sand and clay percentage (Soil Survey Manual, 2017), in `RAWLS_1983`'s names.

    Silt, which Rawls et al. do not tabulate, falls to silt loam, its nearest class.
    """
    sand, clay = float(sand_pct), float(clay_pct)
    silt = 100.0 - sand - clay
    if silt + 1.5 * clay < 15:
        return "sand"
    if silt + 2 * clay < 30:
        return "loamy sand"
    if clay >= 40:
        return "clay" if sand <= 45 and silt < 40 else ("silty clay" if silt >= 40 else "sandy clay")
    if clay >= 35 and sand > 45:
        return "sandy clay"
    if clay >= 27:
        return "silty clay loam" if sand <= 20 else ("clay loam" if sand <= 45 else "sandy clay loam")
    if clay >= 20 and silt < 28 and sand > 45:
        return "sandy clay loam"
    if silt >= 50:
        return "silt loam"
    if clay >= 7 and silt >= 28 and sand <= 52:
        return "loam"
    return "sandy loam"


@dataclass
class Soil:
    """Per-cell soil for GAR, NumPy arrays of the grid's shape (or scalars).

    Args:
        ks: Saturated hydraulic conductivity [m/s]; 0 where the surface is sealed.
        psi_f: Wetting-front suction head [m], positive.
        theta_s, theta_r: Saturated and residual water content [-].
        lam: Brooks-Corey pore-size index [-].
        theta_i: Antecedent water content below every front [-].
        f_max: Most water the soil above its restrictive layer takes [m]; None is unbounded.
    """

    ks: np.ndarray
    psi_f: np.ndarray
    theta_s: np.ndarray
    theta_r: np.ndarray
    lam: np.ndarray
    theta_i: np.ndarray
    f_max: Optional[np.ndarray] = None

    @classmethod
    def texture(cls, name: str, theta_i: float, shape=(), f_max: Optional[float] = None) -> "Soil":
        """A uniform soil of one Rawls et al. (1983) texture class."""
        r = RAWLS_1983[name]
        full = lambda v: np.full(shape, v, dtype=np.float64)  # noqa: E731
        return cls(ks=full(r["ks_cm_h"] / 3.6e5), psi_f=full(r["psi_f_cm"] / 100.0), theta_s=full(r["theta_s"]),
                   theta_r=full(r["theta_r"]), lam=full(r["lam"]), theta_i=full(theta_i),
                   f_max=None if f_max is None else full(f_max))

    def bank(self, shape) -> np.ndarray:
        """The state bank of a soil with no fronts: [5, *shape] as `BANK`."""
        ti = np.broadcast_to(np.asarray(self.theta_i, dtype=np.float64), shape)
        zero = np.zeros(shape)
        return np.stack([zero, ti, zero, ti, zero])


def effective_saturation(theta: Tensor, theta_s: Tensor, theta_r: Tensor) -> Tensor:
    """S_e = (theta - theta_r) / (theta_s - theta_r), in [0, 1]."""
    return ((theta - theta_r) / (theta_s - theta_r)).clamp(0.0, 1.0)


def conductivity(theta: Tensor, ks: Tensor, theta_s: Tensor, theta_r: Tensor, lam: Tensor) -> Tensor:
    """Brooks-Corey unsaturated conductivity K(theta) = K_s S_e^(3 + 2/lambda) [m/s]."""
    return ks * effective_saturation(theta, theta_s, theta_r) ** (3.0 + 2.0 / lam)


def capillary_drive(theta_b: Tensor, theta: Tensor, psi_f: Tensor, theta_s: Tensor, theta_r: Tensor,
                    lam: Tensor) -> Tensor:
    """G(theta_b, theta) [m]: psi_f from theta_i to theta_s, zero as theta reaches theta_b."""
    c = 3.0 + 1.0 / lam
    seb = effective_saturation(theta_b, theta_s, theta_r) ** c
    se = effective_saturation(theta, theta_s, theta_r) ** c
    return psi_f * ((se - seb) / (1.0 - seb).clamp(min=1e-9)).clamp(min=0.0)


def suction_storage(theta_b: Tensor, g: Dict[str, Tensor]) -> Tensor:
    """S = G(theta_b, theta_s) (theta_s - theta_b) [m], the product Green-Ampt's capacity turns on."""
    ts = g["theta_s"]
    return capillary_drive(theta_b, ts, g["psi_f"], ts, g["theta_r"], g["lam"]) * (ts - theta_b).clamp(min=0.0)


def capacity(F: Tensor, theta_b: Tensor, g: Dict[str, Tensor]) -> Tensor:
    """Green-Ampt ponded capacity f = K_s (1 + S / F) [m/s]."""
    return g["ks"] * (1.0 + suction_storage(theta_b, g) / F.clamp(min=F_MIN_M))


def ponded_increment(F: Tensor, S: Tensor, ks: Tensor, dt: Tensor, iterations: int = 8) -> Tensor:
    """Water [m] a front of F takes in dt under ponding: the exact Green-Ampt solution, not f dt.

    Solves d - S ln(1 + d / (F + S)) = K_s dt (the implicit Green-Ampt equation from F to F + d) by Newton's method
    from d0 = sqrt(2 S K_s dt) + K_s dt, an upper bound the convex residual descends from monotonically. Exact at
    any dt, including the first step of a front, where f dt is unbounded.
    """
    kdt = ks * dt
    d = (2.0 * S * kdt).sqrt() + kdt
    base = F.clamp(min=0.0) + S
    for _ in range(iterations):
        res = d - S * torch.log1p(d / base.clamp(min=1e-30)) - kdt
        d = (d - res * (base + d) / (F.clamp(min=0.0) + d).clamp(min=1e-30)).clamp(min=0.0)
    return d


def _rate(theta: Tensor, theta_b: Tensor, F: Tensor, r: Tensor, p: Tensor, g: Dict[str, Tensor]) -> Tensor:
    """dtheta/dt of a front of water F over content theta_b (Smith et al. 1993)."""
    ks, ts, tr, lam = g["ks"], g["theta_s"], g["theta_r"], g["lam"]
    Z = F.clamp(min=F_MIN_M) / (theta - theta_b).clamp(min=1e-9)
    gravity = conductivity(theta, ks, ts, tr, lam) - conductivity(theta_b, ks, ts, tr, lam)
    return (r - gravity - p * ks * capillary_drive(theta_b, theta, g["psi_f"], ts, tr, lam) / Z) / Z


def _redistribute(theta: Tensor, theta_b: Tensor, F: Tensor, r: Tensor, dt: Tensor, g: Dict[str, Tensor]) -> Tensor:
    """One linearly implicit (Rosenbrock-Euler) step of the redistribution ODE, kept in [theta_b, theta_s]."""
    p = torch.where(r > 0.0, torch.full_like(r, P_INFILTRATE), torch.full_like(r, P_REDISTRIBUTE))
    eps = 1e-4 * (g["theta_s"] - g["theta_r"])
    f0 = _rate(theta, theta_b, F, r, p, g)
    jac = ((_rate(theta - eps, theta_b, F, r, p, g) - f0) / -eps).clamp(max=0.0)
    return torch.minimum(torch.maximum(theta + dt * f0 / (1.0 - dt * jac), theta_b), g["theta_s"])


def step(h: Tensor, bank: Tensor, dt: Tensor, g: Dict[str, Tensor], min_depth: float):
    """One sub-step of GAR on every cell: (water infiltrated [m], the new bank).

    Args:
        h: Water on each cell after this sub-step's rain and routing [m].
        bank: [5, rows, cols] F1, theta1, F2, theta2, hiatus (`BANK`).
        dt: Sub-step [s].
        g: Soil tensors: ks, psi_f, theta_s, theta_r, lam, theta_i, f_max (inf where unbounded).
        min_depth: Depth below which a cell counts as dry (no ponding).
    """
    out_dtype, h = h.dtype, h.double()
    dt = dt.double()
    F1, th1, F2, th2, hiatus = bank.unbind(0)
    ks, ts, ti = g["ks"], g["theta_s"], g["theta_i"]
    dt_ = dt.clamp(min=1e-30)
    has1, has2 = F1 > F_MIN_M, F2 > F_MIN_M
    k1 = conductivity(th1, ks, ts, g["theta_r"], g["lam"])
    # A later pulse wets over the drained deep front once it arrives faster than that front conducts.
    top2 = has2 | (has1 & (hiatus > 0.5) & (h / dt_ > k1) & (th1 < ts - 1e-6))
    below = torch.where(top2, th1, ti)
    F_top = torch.where(top2, F2, F1)
    room = (g["f_max"] - F1 - F2).clamp(min=0.0)
    potential = ponded_increment(F_top, suction_storage(below, g), ks, dt)
    inf = torch.minimum(torch.minimum(potential, h), room)
    inf = torch.where(ks > 0.0, inf, torch.zeros_like(inf))
    r = inf / dt_
    ponded = (h - inf) > min_depth

    F_new = F_top + inf
    fresh = (F_top <= F_MIN_M) & (inf > 0.0)
    th_top = torch.where(fresh, ts, torch.where(top2, th2, th1))
    th_top = torch.where(ponded, ts, _redistribute(th_top, below, F_new, r, dt, g))
    th_top = torch.where(F_new > F_MIN_M, th_top, below)
    F1n = torch.where(top2, F1, F_new)
    th1n = torch.where(top2, th1, th_top)
    F2n = torch.where(top2, F_new, F2)
    th2n = torch.where(top2, th_top, th2)

    # Merge the surface front into the deep one at equal depth or equal content, conserving F1 + F2.
    Z1 = F1n / (th1n - ti).clamp(min=1e-9)
    Z2 = F2n / (th2n - th1n).clamp(min=1e-9)
    merge = (F2n > F_MIN_M) & ((Z2 >= Z1) | (th2n <= th1n + 1e-6))
    Fm = F1n + F2n
    thm = (ti + Fm / torch.maximum(Z1, Z2).clamp(min=1e-9)).clamp(max=ts)
    F1n = torch.where(merge, Fm, F1n)
    th1n = torch.where(merge, thm, th1n)
    F2n = torch.where(merge, torch.zeros_like(F2n), F2n)
    th2n = torch.where(merge | (F2n <= F_MIN_M), th1n, th2n)
    return inf.to(out_dtype), torch.stack([F1n, th1n, F2n, th2n, (inf <= 0.0).to(F1n.dtype)])
