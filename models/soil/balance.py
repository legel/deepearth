"""The hourly water balance of every ground cell, in PyTorch: the soil water and standing water a site shows at any hour.

Per cell, a canopy store, water standing on the surface and the soil in layers: the evaporation layer 0 to ZE, the
surface root layer ZE to z_s = min(z_r, Z_S) and, where the roots reach deeper, the deep root layer z_s to z_r (three
layers: `Cells.z_s` and `State.theta3` set). Without them the root zone below ZE is one layer to z_r (two layers). Each
hour:

  1. rain hours: interception, Green-Ampt capacity, and run-on over the terrain (`rain_step`, `routing.cascade`);
  2. every hour: standing water soaks in, net radiation, hourly reference ET (ASCE-EWRI 2005), the FAO-56 dual crop
     coefficient split into interception loss, open-water loss, transpiration (the understory grass's under a crown
     beside the tree's) and soil evaporation, then Brooks-Corey drainage down the layers and out of the root zone
     (`local_step`);
  3. a crown's transpiration drawn from the soil its lateral roots reach (`RootShare`, `root_share_step`);
  4. lateral flow in the surface root layer down the terrain's multiple flow directions (`lateral_step`).

Units: water in mm, contents in m3 m-3, depths in m, fluxes per hour, radiation W m-2, temperature degC, pressure and
vapor pressure kPa, wind m/s at 2 m. Any tensor shape [cells] or [lanes, cells]; any device.
"""

from dataclasses import dataclass, fields
from typing import Callable, Dict, Optional

import numpy as np
import torch

import routing

SIGMA = 5.670374e-8            # Stefan-Boltzmann, W m-2 K-4
EMISSIVITY = 0.98
ZE = 0.10                      # m: the evaporation layer (FAO-56 Z_e)
P_STRESS = 0.5                 # FAO-56 p: the share of TAW used before transpiration falls
F_RESET_H = 6                  # dry hours that end a Green-Ampt event (F back to 0)

# land cover: (infiltrates, has soil, roof, Kcb, Kc_max, root depth m, albedo, f_ew)
SURFACES = {
    "built_surface":    (False, False, True,  0.00, 1.20, 0.00, 0.20, 0.0),
    "sealed_hardscape": (False, False, False, 0.00, 1.20, 0.00, 0.15, 0.0),
    "pervious_ground":  (True,  True,  False, 0.85, 1.15, 0.50, 0.23, 0.3),
    "tall_vegetation":  (True,  True,  False, 0.95, 1.20, 1.00, 0.15, 0.2),
    "vegetated_roof":   (True,  True,  True,  0.75, 1.15, 0.15, 0.20, 0.3),
}
"""Kcb and Kc_max: FAO-56 Table 17, mid-season (turf 0.85, trees 0.95, meadow 0.75). Root depth: FAO-56 Table 22 lower
bounds (turf 0.5 m, trees 1.0 m) and a 0.15 m green-roof substrate (FLL 2018). Albedo: FAO-56 grass reference 0.23,
forest 0.15, paving 0.15, roofing 0.20. f_ew = 1 - cover (FAO-56 eq. 75) at typical cover."""


@dataclass
class Cells:
    """Per-cell constants, each a tensor over cells."""
    alpha: torch.Tensor        # shortwave albedo
    fc: torch.Tensor           # field capacity (33 kPa)
    wp: torch.Tensor           # wilting point (1500 kPa)
    theta_sat: torch.Tensor
    theta_r: torch.Tensor
    ksat: torch.Tensor         # mm/h
    lam: torch.Tensor          # Brooks-Corey pore-size index
    psi_f: torch.Tensor        # Green-Ampt wetting-front suction, mm
    z_r: torch.Tensor          # root depth, m (> ZE)
    kcb: torch.Tensor
    kc_max: torch.Tensor
    rew: torch.Tensor          # readily evaporable water, mm
    f_ew: torch.Tensor         # exposed and wetted fraction
    s_max: torch.Tensor        # canopy storage capacity, mm
    no_soil: torch.Tensor      # bool: no rooting soil (roofs, paving)
    perv: torch.Tensor         # 1 where rain and run-on can soak in, else 0
    roof: torch.Tensor         # bool: rain leaves by the storm sewer unless downspouts are disconnected
    # three layers (`root_fractions`): z_s ends layer 2, layer 3 runs z_s to z_r (empty where the class depth is all), and
    # r1 to r3 are each layer's share of the roots. None: two layers.
    z_s: Optional[torch.Tensor] = None
    r1: Optional[torch.Tensor] = None
    r2: Optional[torch.Tensor] = None
    r3: Optional[torch.Tensor] = None
    kcb_u: Optional[torch.Tensor] = None   # the understory grass's Kcb under a crown (`phenology.understory_kcb`), or None


@dataclass
class State:
    c: torch.Tensor            # canopy store, mm
    w: torch.Tensor            # standing water, mm
    theta1: torch.Tensor       # water content, 0 to ZE
    theta2: torch.Tensor       # water content, ZE to z_r (two layers) or ZE to z_s (three)
    F: torch.Tensor            # Green-Ampt cumulative infiltration of the current event, mm
    dry: torch.Tensor          # hours since water last arrived
    theta3: Optional[torch.Tensor] = None   # water content, z_s to z_r (three layers)

    def clone(self) -> "State":
        return State(**{f.name: (None if getattr(self, f.name) is None else getattr(self, f.name).clone())
                        for f in fields(self)})


Z_S = 0.5
"""m: the surface root layer every class shares, the grass's class depth (FAO-56 Table 22, turf), and the depth SOIL
MOISTURE reports for every class. Read as one well-mixed 2.82 m layer, an oak's deep store was what the soil water showed
for the top half-meter under every crown at Tonzi Ranch: a 4.5-point step at the drip line, where shallow soil water under
blue oak and in the open grass differs little through the season (Jackson et al. 1990, Agric. Ecosyst. Environ.
32:89-105)."""
BETA_WOODY, BETA_HERB = 0.966, 0.943
"""Jackson et al. 1996 (Oecologia 108:389-411) cumulative root fraction Y(d) = 1 - beta^d, d in cm: temperate deciduous
forest 0.966, temperate grassland 0.943."""


def root_fractions(z_r: np.ndarray, canopy: np.ndarray):
    """(z_s, r1, r2, r3) per cell for three layers: layer 2 ends at min(z_r, Z_S). A root zone that ends there keeps its
    roots uniform with depth, so its uptake splits by available water as two layers do; one deeper takes Jackson et
    al.'s (1996) profile for its cover, renormalized over 0 to z_r."""
    z_r = np.asarray(z_r, np.float64)
    z_s = np.maximum(np.minimum(z_r, Z_S), ZE + 0.01)
    deep = z_r > z_s + 1e-6
    beta = np.where(np.asarray(canopy, bool), BETA_WOODY, BETA_HERB)
    Y = lambda d: 1.0 - beta ** (100.0 * d)  # noqa: E731
    tot = np.maximum(Y(z_r), 1e-9)
    r1 = np.where(deep, Y(ZE) / tot, ZE / z_r)
    r2 = np.where(deep, (Y(z_s) - Y(ZE)) / tot, (z_s - ZE) / z_r)
    r3 = np.where(deep, (Y(z_r) - Y(z_s)) / tot, 0.0)
    return z_s, r1, r2, r3


def three_layers(k: Cells, canopy: np.ndarray) -> Cells:
    """`k` with its root zone in three layers (`root_fractions` from its z_r and the canopy cells)."""
    import dataclasses
    zs, r1, r2, r3 = root_fractions(k.z_r.detach().cpu().numpy(), canopy)
    t = lambda a: torch.as_tensor(a, dtype=k.z_r.dtype, device=k.z_r.device)  # noqa: E731
    return dataclasses.replace(k, z_s=t(zs), r1=t(r1), r2=t(r2), r3=t(r3))


def _layers(s: State, k: Cells):
    """(dz2, dz3 or None): layer 2's thickness and, with three layers, layer 3's (0 where the class depth is all)."""
    if s.theta3 is None:
        return k.z_r - ZE, None
    return k.z_s - ZE, torch.clamp(k.z_r - k.z_s, min=0.0)


def root_zone(s: State, k: Cells) -> torch.Tensor:
    """The soil water a site shows (SOIL MOISTURE), m3/m3. Three layers: every cover read over the same surface layer,
    0 to z_s, (theta1 ZE + theta2 (z_s - ZE)) / z_s. Two layers: the root zone's mean, (theta1 ZE + theta2 (z_r - ZE)) / z_r."""
    z = k.z_r if s.theta3 is None else k.z_s
    return (s.theta1 * ZE + s.theta2 * (z - ZE)) / z


def column(s: State, k: Cells) -> torch.Tensor:
    """The whole root zone's mean water content over 0 to z_r, every layer (waterlogging and the dry spell read it)."""
    dz2, dz3 = _layers(s, k)
    deep = 0.0 if dz3 is None else s.theta3 * dz3
    return (s.theta1 * ZE + s.theta2 * dz2 + deep) / k.z_r


# ---------------------------------------------------------------- reference ET and the local step

def es_kpa(t: torch.Tensor) -> torch.Tensor:
    """Saturation vapor pressure (Tetens, FAO-56 eq. 11), kPa."""
    return 0.6108 * torch.exp(17.27 * t / (t + 237.3))


def et0_hourly(rn_w, rs_w, t, ea, pa, u2) -> torch.Tensor:
    """ASCE-EWRI (2005) standardized short-reference ET, hourly, mm/h; negative (dew) set to 0."""
    rn = rn_w * 0.0036                                              # W m-2 to MJ m-2 h-1
    g = torch.where(rs_w > 0, 0.1 * rn, 0.5 * rn)
    es = es_kpa(t)
    delta = 4098.0 * es / (t + 237.3) ** 2
    gamma = 0.000665 * pa
    cd = torch.where(rn > 0, torch.full_like(rn, 0.24), torch.full_like(rn, 0.96))
    num = 0.408 * delta * (rn - g) + gamma * (37.0 / (t + 273.0)) * u2 * (es - ea)
    return torch.clamp(num / (delta + gamma * (1.0 + cd * u2)), min=0.0)


def drain(theta, theta_r, theta_sat, ksat, lam, fc, dz) -> torch.Tensor:
    """Unit-gradient Brooks-Corey drainage over one hour, integrated exactly, never below field capacity, mm.

    d theta / dt = -K_sat S_e^b / (1000 dz),  b = 3 + 2 / lambda, has the closed form
    S_e(1 h)^(1 - b) = S_e0^(1 - b) + (b - 1) K_sat / (1000 dz (theta_s - theta_r)), taken in logs (b reaches 20 in
    clay). An explicit hourly step would drain a wet 10 cm layer at K_sat and empty it to field capacity in one hour."""
    span = torch.clamp(theta_sat - theta_r, min=1e-6)
    se0 = torch.clamp((theta - theta_r) / span, 1e-3, 1.0)
    b = 3.0 + 2.0 / lam
    rate = (b - 1.0) * ksat / (1000.0 * dz * span)
    log_se1 = torch.logaddexp((1.0 - b) * torch.log(se0), torch.log(torch.clamp(rate, min=1e-30))) / (1.0 - b)
    after = theta_r + torch.exp(log_se1) * span
    return 1000.0 * dz * torch.clamp(theta - torch.maximum(after, fc), min=0.0)


def local_step(s: State, k: Cells, rs, u2, t, ea, pa, lw_in) -> Dict[str, torch.Tensor]:
    """Advance `s` in place by one hour of soaking in, drying and drainage; returns the hour's fluxes, mm.
    rs: shortwave reaching the cell (W m-2); lw_in: incoming longwave; t, ea, pa, u2 the hour's air."""
    t, ea, pa, lw_in = (torch.as_tensor(v, dtype=rs.dtype, device=rs.device) for v in (t, ea, pa, lw_in))
    if s.theta3 is not None:
        return _local_step3(s, k, rs, u2, t, ea, pa, lw_in)
    soil = ~k.no_soil
    th1, th2, c, w = s.theta1, s.theta2, s.c, s.w
    dz2 = k.z_r - ZE
    # (a) standing water soaks in, at most K_sat on the pervious share and what the two layers hold
    room = 1000.0 * (torch.clamp(k.theta_sat - th1, min=0.0) * ZE + torch.clamp(k.theta_sat - th2, min=0.0) * dz2)
    fp = torch.where(soil & (k.perv > 0), torch.minimum(torch.minimum(w, k.ksat * k.perv), room), torch.zeros_like(w))
    w = w - fp
    add1 = torch.minimum(fp, 1000.0 * torch.clamp(k.theta_sat - th1, min=0.0) * ZE)
    th1 = th1 + add1 / (1000.0 * ZE)
    th2 = th2 + (fp - add1) / (1000.0 * dz2)
    # (b) net radiation and (c) reference ET
    rn = (1.0 - k.alpha) * rs + EMISSIVITY * (lw_in - SIGMA * (t + 273.15) ** 4)
    et0 = et0_hourly(rn, rs, t, ea, pa, u2)
    # (d) the canopy store evaporates first, then standing water
    ec = torch.minimum(c, et0)
    c = c - ec
    e = et0 - ec
    ew = torch.minimum(w, e)
    w = w - ew
    e = e - ew
    # (e) transpiration, K_s K_cb ET0 (FAO-56 eq. 84), from each layer in proportion to its water above wp
    taw = 1000.0 * (k.fc - k.wp) * k.z_r
    dr = 1000.0 * (k.fc * k.z_r - th1 * ZE - th2 * dz2)
    ks = torch.clamp((taw - dr) / ((1.0 - P_STRESS) * taw), 0.0, 1.0)
    tr = torch.where(soil, ks * k.kcb * e, torch.zeros_like(e))
    a1 = 1000.0 * torch.clamp(th1 - k.wp, min=0.0) * ZE
    a2 = 1000.0 * torch.clamp(th2 - k.wp, min=0.0) * dz2
    tot = a1 + a2
    share1 = torch.where(tot > 0, a1 / torch.clamp(tot, min=1e-12), torch.zeros_like(tot))
    t1 = torch.minimum(tr * share1, a1)
    t2 = torch.minimum(tr * (1.0 - share1), a2)
    th1 = th1 - t1 / (1000.0 * ZE)
    th2 = th2 - t2 / (1000.0 * dz2)
    # (f) soil evaporation, K_e ET0 (FAO-56 eqs. 71 to 74), from layer 1 down to wp / 2
    tew = 1000.0 * (k.fc - 0.5 * k.wp) * ZE
    de = torch.clamp(1000.0 * (k.fc - th1) * ZE, min=0.0)
    kr = torch.clamp((tew - de) / torch.clamp(tew - k.rew, min=1e-6), 0.0, 1.0)
    ke = torch.minimum(kr * (k.kc_max - k.kcb), k.f_ew * k.kc_max)
    avail = 1000.0 * torch.clamp(th1 - 0.5 * k.wp, min=0.0) * ZE
    es = torch.where(soil, torch.minimum(torch.clamp(ke, min=0.0) * e, avail), torch.zeros_like(e))
    th1 = th1 - es / (1000.0 * ZE)
    # (g) drainage: layer 1 into layer 2 (as much as it holds), then out of the root zone
    q1 = drain(th1, k.theta_r, k.theta_sat, k.ksat, k.lam, k.fc, ZE)
    q1 = torch.where(soil, torch.minimum(q1, 1000.0 * torch.clamp(k.theta_sat - th2, min=0.0) * dz2), torch.zeros_like(q1))
    th1 = th1 - q1 / (1000.0 * ZE)
    th2 = th2 + q1 / (1000.0 * dz2)
    q2 = torch.where(soil, drain(th2, k.theta_r, k.theta_sat, k.ksat, k.lam, k.fc, dz2), torch.zeros_like(q1))
    th2 = th2 - q2 / (1000.0 * dz2)
    s.c, s.w, s.theta1, s.theta2 = c, w, th1, th2
    return {"et0": et0, "aet": ec + ew + t1 + t2 + es, "drain": q2, "ec": ec, "ew": ew, "tr": t1 + t2, "es": es,
            "pond_in": fp, "t1": t1, "t2": t2}


def _local_step3(s: State, k: Cells, rs, u2, t, ea, pa, lw_in) -> Dict[str, torch.Tensor]:
    """`local_step` in three layers: 0 to ZE, ZE to z_s, z_s to z_r. Water enters from the top and drains down layer by
    layer. Transpiration is drawn from each layer by its share of the roots times its relative available water (Feddes,
    Kowalik and Zaradny 1978, with compensation), so the top dries first and the deep store carries the dry season. The
    understory grass under a crown (`k.kcb_u`) transpires from layers 1 and 2 by its own stress, within what the tree
    leaves of Kc_max. Where z_r = z_s, layer 3 is empty and every term is the two-layer step's."""
    soil = ~k.no_soil
    th1, th2, th3, c, w = s.theta1, s.theta2, s.theta3, s.c, s.w
    dz2 = k.z_s - ZE
    dz3 = k.z_r - k.z_s
    has3 = dz3 > 1e-6
    dz3s = torch.clamp(dz3, min=1e-6)
    zero = torch.zeros_like(th1)
    # (a) standing water soaks in, from the top: layer 1, then 2, then 3
    room1 = 1000.0 * torch.clamp(k.theta_sat - th1, min=0.0) * ZE
    room2 = 1000.0 * torch.clamp(k.theta_sat - th2, min=0.0) * dz2
    room3 = torch.where(has3, 1000.0 * torch.clamp(k.theta_sat - th3, min=0.0) * dz3, zero)
    fp = torch.where(soil & (k.perv > 0), torch.minimum(torch.minimum(w, k.ksat * k.perv), room1 + room2 + room3), zero)
    w = w - fp
    add1 = torch.minimum(fp, room1)
    add2 = torch.minimum(fp - add1, room2)
    th1 = th1 + add1 / (1000.0 * ZE)
    th2 = th2 + add2 / (1000.0 * dz2)
    th3 = th3 + (fp - add1 - add2) / (1000.0 * dz3s)
    # (b) net radiation, (c) reference ET, (d) the canopy store then standing water
    rn = (1.0 - k.alpha) * rs + EMISSIVITY * (lw_in - SIGMA * (t + 273.15) ** 4)
    et0 = et0_hourly(rn, rs, t, ea, pa, u2)
    ec = torch.minimum(c, et0)
    c = c - ec
    e = et0 - ec
    ew = torch.minimum(w, e)
    w = w - ew
    e = e - ew
    # (e) the tree's (or the cover's) transpiration, K_s over the whole root zone, split by roots x relative water
    deep = torch.where(has3, th3 * dz3, zero)
    taw = 1000.0 * (k.fc - k.wp) * k.z_r
    dr = 1000.0 * (k.fc * k.z_r - th1 * ZE - th2 * dz2 - deep)
    ks = torch.clamp((taw - dr) / ((1.0 - P_STRESS) * taw), 0.0, 1.0)
    tr = torch.where(soil, ks * k.kcb * e, zero)
    a1 = 1000.0 * torch.clamp(th1 - k.wp, min=0.0) * ZE
    a2 = 1000.0 * torch.clamp(th2 - k.wp, min=0.0) * dz2
    a3 = torch.where(has3, 1000.0 * torch.clamp(th3 - k.wp, min=0.0) * dz3, zero)
    span = torch.clamp(1000.0 * (k.fc - k.wp), min=1e-6)
    g1 = k.r1 * a1 / (span * ZE)
    g2 = k.r2 * a2 / (span * dz2)
    g3 = k.r3 * a3 / (span * dz3s)
    gt = g1 + g2 + g3
    inv = torch.where(gt > 0, 1.0 / torch.clamp(gt, min=1e-12), zero)
    t1 = torch.minimum(tr * g1 * inv, a1)
    t2 = torch.minimum(tr * g2 * inv, a2)
    t3 = torch.minimum(tr * g3 * inv, a3)
    th1 = th1 - t1 / (1000.0 * ZE)
    th2 = th2 - t2 / (1000.0 * dz2)
    th3 = th3 - t3 / (1000.0 * dz3s)
    # (e') the understory grass: its roots in layers 1 and 2, its stress over them; it stays on its own cell
    kcb_u = k.kcb_u if k.kcb_u is not None else torch.zeros_like(k.kcb)
    kcb_ue = torch.minimum(kcb_u, torch.clamp(k.kc_max - k.kcb, min=0.0))
    taw_s = 1000.0 * (k.fc - k.wp) * k.z_s
    dr_s = torch.clamp(1000.0 * (k.fc * k.z_s - th1 * ZE - th2 * dz2), min=0.0)
    ks_u = torch.clamp((taw_s - dr_s) / ((1.0 - P_STRESS) * taw_s), 0.0, 1.0)
    tr_u = torch.where(soil, ks_u * kcb_ue * e, zero)
    b1 = 1000.0 * torch.clamp(th1 - k.wp, min=0.0) * ZE
    b2 = 1000.0 * torch.clamp(th2 - k.wp, min=0.0) * dz2
    share_u = torch.where(b1 + b2 > 0, b1 / torch.clamp(b1 + b2, min=1e-12), zero)
    tu1 = torch.minimum(tr_u * share_u, b1)
    tu2 = torch.minimum(tr_u * (1.0 - share_u), b2)
    th1 = th1 - tu1 / (1000.0 * ZE)
    th2 = th2 - tu2 / (1000.0 * dz2)
    tu = tu1 + tu2
    # (f) soil evaporation from layer 1, in what neither the tree nor the grass takes of Kc_max
    tew = 1000.0 * (k.fc - 0.5 * k.wp) * ZE
    de = torch.clamp(1000.0 * (k.fc - th1) * ZE, min=0.0)
    kr = torch.clamp((tew - de) / torch.clamp(tew - k.rew, min=1e-6), 0.0, 1.0)
    ke = torch.minimum(kr * torch.clamp(k.kc_max - k.kcb - kcb_ue, min=0.0), k.f_ew * k.kc_max)
    avail = 1000.0 * torch.clamp(th1 - 0.5 * k.wp, min=0.0) * ZE
    es = torch.where(soil, torch.minimum(torch.clamp(ke, min=0.0) * e, avail), zero)
    th1 = th1 - es / (1000.0 * ZE)
    # (g) drainage down the layers, each into the next as much as it holds, the last out of the root zone
    q1 = drain(th1, k.theta_r, k.theta_sat, k.ksat, k.lam, k.fc, ZE)
    q1 = torch.where(soil, torch.minimum(q1, 1000.0 * torch.clamp(k.theta_sat - th2, min=0.0) * dz2), zero)
    th1 = th1 - q1 / (1000.0 * ZE)
    th2 = th2 + q1 / (1000.0 * dz2)
    q2 = torch.where(soil, drain(th2, k.theta_r, k.theta_sat, k.ksat, k.lam, k.fc, dz2), zero)
    q2 = torch.where(has3, torch.minimum(q2, 1000.0 * torch.clamp(k.theta_sat - th3, min=0.0) * dz3s), q2)
    th2 = th2 - q2 / (1000.0 * dz2)
    th3 = th3 + torch.where(has3, q2, zero) / (1000.0 * dz3s)
    q3 = torch.where(soil & has3, drain(th3, k.theta_r, k.theta_sat, k.ksat, k.lam, k.fc, dz3s), zero)
    th3 = th3 - q3 / (1000.0 * dz3s)
    s.c, s.w, s.theta1, s.theta2, s.theta3 = c, w, th1, th2, th3
    return {"et0": et0, "aet": ec + ew + t1 + t2 + t3 + tu + es, "drain": torch.where(has3, q3, q2), "ec": ec, "ew": ew,
            "tr": t1 + t2 + t3 + tu, "es": es, "pond_in": fp, "t1": t1, "t2": t2, "t3": t3, "tu": tu}


# ---------------------------------------------------------------- rain hours

def green_ampt_capacity(F: torch.Tensor, k: Cells, theta1: torch.Tensor, iters: int = 20) -> torch.Tensor:
    """What a cell can take in over one hour with water standing from the start (Green and Ampt 1911; Mein and Larson
    1973), mm: F1 = F0 + K_sat + psi_f dtheta ln((F1 + psi_f dtheta) / (F0 + psi_f dtheta)), dtheta = theta_s - theta1,
    solved by Newton from Philip's dry-start estimate."""
    pd = k.psi_f * torch.clamp(k.theta_sat - theta1, min=1e-4)
    f1 = F + k.ksat + torch.sqrt(2.0 * k.ksat * pd)
    for _ in range(iters):
        g = f1 - F - k.ksat - pd * torch.log((f1 + pd) / (F + pd))
        dg = 1.0 - pd / (f1 + pd)
        f1 = torch.clamp(f1 - g / torch.clamp(dg, min=1e-6), min=F + k.ksat)
    return f1 - F


def rain_step(s: State, k: Cells, rain_mm: float, net: routing.Network,
              downspouts_disconnected: bool = False) -> Dict[str, torch.Tensor]:
    """An hour's rain on 1-D state: interception, Green-Ampt capacity capped by the soil's room, then run-on over the
    terrain (`routing.cascade`), where each cell infiltrates what it can, fills its depression and passes the rest
    downslope. A roof's rain goes to the storm sewer unless its downspouts are disconnected; a vegetated roof first
    fills its substrate. Only pervious cells take water in; the rest shed all that reaches them."""
    caught = torch.minimum(torch.clamp(k.s_max - s.c, min=0.0), torch.full_like(s.c, rain_mm))
    s.c = s.c + caught
    through = rain_mm - caught
    cap = torch.where(k.perv > 0, torch.minimum(green_ampt_capacity(s.F, k, s.theta1), _room(s, k)), torch.zeros_like(s.c))
    roof_in = torch.where(k.roof, torch.minimum(through * k.perv, cap), torch.zeros_like(s.c))
    sewer = torch.where(k.roof, through - roof_in, torch.zeros_like(s.c))
    ground = torch.where(k.roof, sewer if downspouts_disconnected else torch.zeros_like(s.c), through)
    sewer = torch.zeros_like(sewer) if downspouts_disconnected else sewer
    cap = torch.where(k.roof, torch.zeros_like(cap), cap)
    np_ = lambda a: a.detach().double().cpu().numpy()                                          # noqa: E731
    infil, pond, inflow, lost = (torch.as_tensor(a, dtype=s.c.dtype, device=s.c.device)
                                 for a in routing.cascade(net, np_(ground + s.w), np_(cap), np_(k.perv)))
    infil = infil + roof_in
    s.w = pond
    s.F = s.F + infil
    _soak(s, k, infil)
    return {"through": through, "infil": infil, "arrived": through + inflow, "runon": inflow, "lost": lost + sewer}


def _room(s: State, k: Cells) -> torch.Tensor:
    """mm the root zone can still take, every layer to saturation."""
    dz2, dz3 = _layers(s, k)
    room = 1000.0 * (torch.clamp(k.theta_sat - s.theta1, min=0.0) * ZE + torch.clamp(k.theta_sat - s.theta2, min=0.0) * dz2)
    if dz3 is not None:
        room = room + 1000.0 * torch.clamp(k.theta_sat - s.theta3, min=0.0) * dz3
    return room


def _soak(s: State, k: Cells, infil: torch.Tensor) -> None:
    """Infiltrated water into the layers from the top: layer 1 to saturation, then layer 2, then (three layers) 3."""
    dz2, dz3 = _layers(s, k)
    add1 = torch.minimum(infil, 1000.0 * torch.clamp(k.theta_sat - s.theta1, min=0.0) * ZE)
    s.theta1 = s.theta1 + add1 / (1000.0 * ZE)
    if dz3 is None:
        s.theta2 = s.theta2 + (infil - add1) / (1000.0 * dz2)
        return
    add2 = torch.minimum(infil - add1, 1000.0 * torch.clamp(k.theta_sat - s.theta2, min=0.0) * dz2)
    s.theta2 = s.theta2 + add2 / (1000.0 * dz2)
    s.theta3 = s.theta3 + (infil - add1 - add2) / (1000.0 * torch.clamp(dz3, min=1e-6))


# ---------------------------------------------------------------- a crown drinks from the soil its roots reach

ROOT_RADIUS_M = 10.0
"""How far a tree's lateral roots draw water. Framework roots span four to seven times the area under the crown (Perry
1982, J. Arboric. 8:197-211), a radius of 2 to 2.6 crown radii, and root systems one to two tree heights across are
common (Stout 1956; Lyford and Wilson 1964, Harvard Forest Paper 10, red maple at Harvard Forest). A mature oak's or
maple's crown radius of 4 to 5 m gives 8 to 13 m: 10 m."""


class RootShare:
    """A crown's transpiration drawn evenly from the soil cells within ROOT_RADIUS_M of it (a disk), mass kept exactly:
    each canopy cell's uptake divided by the soil cells in its reach, then summed at every soil cell over the crowns
    that reach it. Grass and shrub cells, and a crown with no soil in reach, keep their own cell's soil. On a grid of
    rows x cols cells (row-major), by FFT convolution per lane, in float64."""

    def __init__(self, shape, dx_m: float, canopy: np.ndarray, soil: np.ndarray, device="cpu",
                 radius_m: float = ROOT_RADIUS_M):
        th, tw = shape
        self.shape, self.r = (th, tw), max(1, int(round(radius_m / dx_m)))
        r = self.r
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        disk = ((xx ** 2 + yy ** 2) <= r * r).astype(np.float64)
        self.fs = (th + 2 * r, tw + 2 * r)                     # linear, not circular: room for the reach both ways
        kern = np.zeros(self.fs)
        kern[:2 * r + 1, :2 * r + 1] = disk
        kern = np.roll(kern, (-r, -r), axis=(0, 1))            # the disk centered on (0, 0)
        self.K = torch.fft.rfft2(torch.as_tensor(kern, dtype=torch.float64, device=device))
        self.soil = torch.as_tensor(np.asarray(soil, bool).astype(np.float64), device=device)
        den = self.conv(self.soil[None])[0]
        can = torch.as_tensor(np.asarray(canopy, bool), device=device)
        spread = can & (den > 0.5)
        self.inv = torch.where(spread, 1.0 / torch.clamp(den, min=1.0), torch.zeros_like(den))
        self.local = (~spread).double()

    def conv(self, x: torch.Tensor) -> torch.Tensor:
        """[lanes, cells] summed over each cell's disk."""
        th, tw = self.shape
        X = torch.zeros((x.shape[0],) + self.fs, dtype=torch.float64, device=x.device)
        X[:, :th, :tw] = x.double().reshape(-1, th, tw)
        y = torch.fft.irfft2(torch.fft.rfft2(X) * self.K, s=self.fs)
        return y[:, :th, :tw].reshape(x.shape[0], th * tw)

    def uptake(self, tr: torch.Tensor) -> torch.Tensor:
        """mm each cell's soil gives for the crowns' transpiration `tr` ([cells] or [lanes, cells])."""
        one = tr.dim() == 1
        x = tr[None] if one else tr
        out = (self.soil * self.conv(x.double() * self.inv) + x.double() * self.local).to(tr.dtype)
        return out[0] if one else out


def root_share_step(s: State, k: Cells, flux: Dict[str, torch.Tensor], share: RootShare) -> Dict[str, torch.Tensor]:
    """The hour's transpiration taken from the soil its roots reach rather than the crown's own cell. What the local step
    took from layers 1 and 2 is put back, and each soil cell then gives its share of the crowns' uptake from those two
    layers by their available water, never below wilting point (with three layers the deep layer stays the cell's own).
    Mass is kept: `root_out` left each crown's cell, `root_in` came from each soil cell, and `root_unmet` (a reach whose
    soil is at wilting point) is transpiration its soil could not give, taken off the crowns' AET in proportion."""
    dz2 = torch.clamp(_layers(s, k)[0], min=1e-3)
    t1, t2 = flux["t1"], flux["t2"]
    own = t1 + t2
    s.theta1 = s.theta1 + t1 / (1000.0 * ZE)
    s.theta2 = s.theta2 + t2 / (1000.0 * dz2)
    want = share.uptake(own)
    a1 = 1000.0 * torch.clamp(s.theta1 - k.wp, min=0.0) * ZE
    a2 = 1000.0 * torch.clamp(s.theta2 - k.wp, min=0.0) * dz2
    tot = a1 + a2
    take = torch.minimum(want, tot)
    f1 = torch.where(tot > 0, a1 / torch.clamp(tot, min=1e-12), torch.zeros_like(tot))
    s.theta1 = s.theta1 - take * f1 / (1000.0 * ZE)
    s.theta2 = s.theta2 - take * (1.0 - f1) / (1000.0 * dz2)
    unmet = want - take
    lost = unmet.sum(dim=-1, keepdim=True)                     # given back to the crowns' AET pro rata
    out = own.sum(dim=-1, keepdim=True)
    frac = torch.where(out > 0, lost / torch.clamp(out, min=1e-12), torch.zeros_like(lost))
    return {"aet": flux["aet"] - own * frac, "root_out": own, "root_in": take, "root_unmet": unmet}


# ---------------------------------------------------------------- lateral flow in the root zone

LATERAL_ANISOTROPY = 1.0
"""K_lateral / K_sat of the root zone. 1 is the isotropic floor; forest soils run higher along the slope (macropores,
layered till), which this does not assume."""


class LateralGraph:
    """`routing.Lateral` on a device: a sparse [dst, src] matrix of MFD weights, tan(beta) per cell, dx in mm."""

    def __init__(self, lat: routing.Lateral, n: int, device="cpu"):
        idx = torch.as_tensor(np.stack([lat.dst, lat.src]), dtype=torch.int64, device=device)
        self.M = torch.sparse_coo_tensor(idx, torch.as_tensor(lat.w, dtype=torch.float32, device=device), (n, n), check_invariants=False).coalesce()
        self.tanb = torch.as_tensor(lat.tanb, dtype=torch.float32, device=device)
        self.dx_mm = 1000.0 * float(lat.dx_m)


def lateral_step(s: State, k: Cells, g: LateralGraph) -> Dict[str, torch.Tensor]:
    """An hour of downslope flow through layer 2: Darcy's flux on the land surface's gradient at the layer's own
    Brooks-Corey conductivity, the K(theta2) its vertical drainage uses,

        Q = min(W, a K(theta2) tan(beta) dz2 / dx),   K = K_sat S_e^(3 + 2/lambda),   W = 1000 (theta2 - fc)+ dz2,

    handed to the MFD receivers by weight, filling their layer 2, then layer 1, the rest returning to the surface
    (return flow). Only water above field capacity moves: a dry slope passes nothing, a hollow gathers what its slopes
    pass. Every Q has receivers (tan(beta) is 0 without them), so mass is conserved."""
    one = s.theta2.dim() == 1
    if one:
        s2 = State(**{f.name: (None if getattr(s, f.name) is None else getattr(s, f.name)[None]) for f in fields(s)})
        out = lateral_step(s2, k, g)
        for f in fields(s):
            if getattr(s2, f.name) is not None:
                setattr(s, f.name, getattr(s2, f.name)[0])
        return {n: v[0] for n, v in out.items()}
    soil = ~k.no_soil
    dz2 = torch.clamp(_layers(s, k)[0], min=1e-3)     # three layers: the surface root layer moves
    W = torch.where(soil, 1000.0 * torch.clamp(s.theta2 - k.fc, min=0.0) * dz2, torch.zeros_like(s.theta2))
    se = torch.clamp((s.theta2 - k.theta_r) / torch.clamp(k.theta_sat - k.theta_r, min=1e-6), 0.0, 1.0)
    kth = k.ksat * se ** (3.0 + 2.0 / k.lam)
    q = torch.minimum(W, LATERAL_ANISOTROPY * kth * g.tanb * 1000.0 * dz2 / g.dx_mm)
    s.theta2 = s.theta2 - q / (1000.0 * dz2)
    inflow = torch.sparse.mm(g.M, q.T.to(g.M.dtype)).T.to(q.dtype)
    a2 = torch.minimum(inflow, torch.where(soil, 1000.0 * torch.clamp(k.theta_sat - s.theta2, min=0.0) * dz2,
                                           torch.zeros_like(W)))
    s.theta2 = s.theta2 + a2 / (1000.0 * dz2)
    rest = inflow - a2
    a1 = torch.minimum(rest, torch.where(soil, 1000.0 * torch.clamp(k.theta_sat - s.theta1, min=0.0) * ZE,
                                         torch.zeros_like(W)))
    s.theta1 = s.theta1 + a1 / (1000.0 * ZE)
    back = rest - a1
    s.w = s.w + back
    return {"lateral_out": q, "lateral_in": inflow, "return_flow": back}


# ---------------------------------------------------------------- an hour and a year

def hour(s: State, k: Cells, rain_mm: float, net: routing.Network, rs, u2, t, ea, pa, lw_in,
         lateral: Optional[LateralGraph] = None, downspouts_disconnected: bool = False,
         roots: Optional[RootShare] = None) -> Dict[str, torch.Tensor]:
    """One hour on 1-D state: `rain_step` if it rains, `local_step`, `root_share_step` if a RootShare is given, then
    `lateral_step` if a graph is given."""
    if rain_mm > 0:
        r = rain_step(s, k, rain_mm, net, downspouts_disconnected)
        s.dry = torch.zeros_like(s.dry)
    else:
        r = {"infil": torch.zeros_like(s.c), "lost": torch.zeros_like(s.c), "arrived": torch.zeros_like(s.c)}
        s.dry = s.dry + 1
    s.F = torch.where(s.dry >= F_RESET_H, torch.zeros_like(s.F), s.F)
    flux = local_step(s, k, rs, u2, t, ea, pa, lw_in)
    if roots is not None:
        flux.update(root_share_step(s, k, flux, roots))
    if lateral is not None:
        flux.update(lateral_step(s, k, lateral))
    return dict(flux, infil=r["infil"] + flux["pond_in"], lost=r["lost"], arrived=r["arrived"])


class Year:
    """A year's metrics per cell: climatic water deficit, actual ET, water received, waterlogging and dry spells,
    ponding."""

    AIR_FILLED = 0.10          # waterlogged: less than this air-filled porosity in the root zone
    PONDED_MM = 5.0
    FLOW_MM = 0.01             # an hour in which less than this reaches the soil has no water flow

    def __init__(self, n: int, device="cpu"):
        z = lambda: torch.zeros(n, device=device)                                               # noqa: E731
        self.cwd, self.aet, self.received, self.runoff, self.drain = z(), z(), z(), z(), z()
        self.wet_h, self.wet_run, self.wet_max, self.dry_run, self.dry_max = z(), z(), z(), z(), z()
        self.noflow_run, self.noflow_max, self.flow_h, self.pond_max, self.pond_h = z(), z(), z(), z(), z()

    @staticmethod
    def _run(run, best, on):
        run = (run + 1.0) * on.float()
        return run, torch.maximum(best, run)

    def add(self, s: State, k: Cells, f: Dict[str, torch.Tensor], roots: Optional[RootShare] = None) -> None:
        soil = ~k.no_soil
        deficit = torch.clamp(f["et0"] - f["aet"], min=0.0)
        if roots is not None:
            # the deficit follows the soil the roots draw: a crown's unmet demand is charged to the soil cells its roots
            # reach, as its uptake is. Charged to the crown's own cell, the crown-top demand stepped against the ground's
            # at every drip line (Harvard DROUGHT: 82 % of edge pairs jumped), where a crown's transpiration does not
            # change with distance from a gap's edge (Arteman et al. 2025, Agric. For. Meteorol. 373:110754) and gap soil
            # dries from the edge as roots encroach (Gray et al. 2002, Can. J. For. Res. 32:332-343). Grass and shrubs
            # keep their own cell; the site's total deficit is unchanged.
            deficit = roots.uptake(deficit)
        self.cwd += deficit
        self.aet += f["aet"]
        self.received += f["infil"]
        self.runoff += f["lost"]
        self.drain += f["drain"]
        wet = soil & (k.theta_sat - column(s, k) < self.AIR_FILLED)
        self.wet_h += wet.float()
        self.wet_run, self.wet_max = self._run(self.wet_run, self.wet_max, wet)
        dr = 1000.0 * (k.fc - column(s, k)) * k.z_r
        self.dry_run, self.dry_max = self._run(self.dry_run, self.dry_max, soil & (dr > P_STRESS * 1000.0 * (k.fc - k.wp) * k.z_r))
        nof = f["infil"] < self.FLOW_MM
        self.noflow_run, self.noflow_max = self._run(self.noflow_run, self.noflow_max, nof)
        self.flow_h += (~nof).float()
        self.pond_max = torch.maximum(self.pond_max, s.w)
        self.pond_h += (s.w > self.PONDED_MM).float()

    def metrics(self, k: Cells) -> Dict[str, torch.Tensor]:
        """Soil metrics are NaN where the cell has no soil; ponding stays everywhere (water stands on paving too)."""
        m = lambda a: torch.where(~k.no_soil, a, torch.full_like(a, float("nan")))                # noqa: E731
        return {"cwd_mm": m(self.cwd), "aet_mm": m(self.aet), "received_mm": m(self.received),
                "waterlogged_h": m(self.wet_h), "waterlogged_spell_h": m(self.wet_max),
                "dry_spell_days": m(self.dry_max / 24.0), "no_flow_h": m(self.noflow_max), "flow_h": m(self.flow_h),
                "pond_peak_mm": self.pond_max, "ponded_h": self.pond_h}


def run(s: State, k: Cells, net: routing.Network, hours: int, rain, rs_of: Callable, u2_of: Callable, air: Callable,
        lateral: Optional[LateralGraph] = None, on_hour: Optional[Callable] = None,
        roots: Optional[RootShare] = None, kcb_of: Optional[Callable] = None) -> Year:
    """Advance `s` through `hours`: rain[h] mm, rs_of(h) and u2_of(h) per-cell tensors, air(h) = (t, ea, pa, lw_in);
    on_hour(h, s) sees each hour's state (a site stores the root zone's water and the standing water from it); roots
    shares the crowns' uptake and deficit (`RootShare`); kcb_of(h) returns the hour's cells (each cover's season,
    `phenology`), else `k` throughout."""
    y = Year(s.c.shape[0], s.c.device)
    for h in range(hours):
        kh = k if kcb_of is None else kcb_of(h)
        f = hour(s, kh, float(rain[h]), net, rs_of(h), u2_of(h), *air(h), lateral=lateral, roots=roots)
        y.add(s, kh, f, roots=roots)
        if on_hour is not None:
            on_hour(h, s)
    return y
