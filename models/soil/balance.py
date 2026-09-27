"""The hourly water balance of every ground cell, in PyTorch: the soil water and standing water a site shows at any hour.

Per cell, two soil layers (the evaporation layer 0 to ZE and the root zone ZE to z_r), a canopy store and water standing
on the surface. Each hour:

  1. rain hours: interception, Green-Ampt capacity, and run-on over the terrain (`rain_step`, `routing.cascade`);
  2. every hour: standing water soaks in, net radiation, hourly reference ET (ASCE-EWRI 2005), the FAO-56 dual crop
     coefficient split into interception loss, open-water loss, transpiration and soil evaporation, then Brooks-Corey
     drainage from layer 1 to layer 2 and out of the root zone (`local_step`);
  3. lateral flow in the root zone down the terrain's multiple flow directions (`lateral_step`).

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


@dataclass
class State:
    c: torch.Tensor            # canopy store, mm
    w: torch.Tensor            # standing water, mm
    theta1: torch.Tensor       # water content, 0 to ZE
    theta2: torch.Tensor       # water content, ZE to z_r
    F: torch.Tensor            # Green-Ampt cumulative infiltration of the current event, mm
    dry: torch.Tensor          # hours since water last arrived

    def clone(self) -> "State":
        return State(**{f.name: getattr(self, f.name).clone() for f in fields(self)})


def root_zone(s: State, k: Cells) -> torch.Tensor:
    """The root zone's mean water content, what a site shows as soil water: (theta1 ZE + theta2 (z_r - ZE)) / z_r."""
    return (s.theta1 * ZE + s.theta2 * (k.z_r - ZE)) / k.z_r


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
            "pond_in": fp}


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
    dz2 = k.z_r - ZE
    room = 1000.0 * (torch.clamp(k.theta_sat - s.theta1, min=0.0) * ZE + torch.clamp(k.theta_sat - s.theta2, min=0.0) * dz2)
    cap = torch.where(k.perv > 0, torch.minimum(green_ampt_capacity(s.F, k, s.theta1), room), torch.zeros_like(s.c))
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
    add1 = torch.minimum(infil, 1000.0 * torch.clamp(k.theta_sat - s.theta1, min=0.0) * ZE)
    s.theta1 = s.theta1 + add1 / (1000.0 * ZE)
    s.theta2 = s.theta2 + (infil - add1) / (1000.0 * dz2)
    return {"through": through, "infil": infil, "arrived": through + inflow, "runon": inflow, "lost": lost + sewer}


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
        s2 = State(**{f.name: getattr(s, f.name)[None] for f in fields(s)})
        out = lateral_step(s2, k, g)
        for f in fields(s):
            setattr(s, f.name, getattr(s2, f.name)[0])
        return {n: v[0] for n, v in out.items()}
    soil = ~k.no_soil
    dz2 = torch.clamp(k.z_r - ZE, min=1e-3)
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
         lateral: Optional[LateralGraph] = None, downspouts_disconnected: bool = False) -> Dict[str, torch.Tensor]:
    """One hour on 1-D state: `rain_step` if it rains, `local_step`, then `lateral_step` if a graph is given."""
    if rain_mm > 0:
        r = rain_step(s, k, rain_mm, net, downspouts_disconnected)
        s.dry = torch.zeros_like(s.dry)
    else:
        r = {"infil": torch.zeros_like(s.c), "lost": torch.zeros_like(s.c), "arrived": torch.zeros_like(s.c)}
        s.dry = s.dry + 1
    s.F = torch.where(s.dry >= F_RESET_H, torch.zeros_like(s.F), s.F)
    flux = local_step(s, k, rs, u2, t, ea, pa, lw_in)
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

    def add(self, s: State, k: Cells, f: Dict[str, torch.Tensor]) -> None:
        soil = ~k.no_soil
        self.cwd += torch.clamp(f["et0"] - f["aet"], min=0.0)
        self.aet += f["aet"]
        self.received += f["infil"]
        self.runoff += f["lost"]
        self.drain += f["drain"]
        wet = soil & (k.theta_sat - root_zone(s, k) < self.AIR_FILLED)
        self.wet_h += wet.float()
        self.wet_run, self.wet_max = self._run(self.wet_run, self.wet_max, wet)
        dr = 1000.0 * (k.fc * k.z_r - s.theta1 * ZE - s.theta2 * (k.z_r - ZE))
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
        lateral: Optional[LateralGraph] = None, on_hour: Optional[Callable] = None) -> Year:
    """Advance `s` through `hours`: rain[h] mm, rs_of(h) and u2_of(h) per-cell tensors, air(h) = (t, ea, pa, lw_in);
    on_hour(h, s) sees each hour's state (a site stores the root zone's water and the standing water from it)."""
    y = Year(s.c.shape[0], s.c.device)
    for h in range(hours):
        f = hour(s, k, float(rain[h]), net, rs_of(h), u2_of(h), *air(h), lateral=lateral)
        y.add(s, k, f)
        if on_hour is not None:
            on_hour(h, s)
    return y
