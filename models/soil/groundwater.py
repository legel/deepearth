"""The shallow water table under the root zone, from the soil survey and the terrain (EXPOSURE_GROUNDWATER, on by default).

Without one the balance drained every root zone into nothing, so a valley bottom and a ridge dried at the same rate
through a dry summer (an urban forest site, 2026-10-05: SOIL MOISTURE and DROUGHT rank-correlated with the topographic
wetness index at -0.04 and -0.03, with the soil unit's theta_s at 0.82 and -0.67).

The table is a lower boundary the survey measures, distributed by the terrain the way TOPMODEL distributes it (Beven and
Kirkby 1979), held in the soil above its restrictive layer (Saulnier, Beven and Obled 1997, J. Hydrol. 202:158: TOPMODEL's
saturated zone confined to an effective soil depth):

    zbar(month)  the site's mean table: the median over its soil cells of the survey's top of the Wet layer that month
                 (SSURGO cosoilmoist, the normal year; soils.survey_water_table); a cell whose unit has none that month
                 counts DRY_BELOW_M under its restrictive layer, the perched table gone into the regolith
    z_i          zbar + SENS_M (lambda - TWI_i): TOPMODEL's local depth, the wetness index ln(a / tan beta) on the 3DEP
                 DEM around the site (exposure.twi), lambda its mean over the site's soil cells; SENS_M is m / phi
    D_i          the restrictive layer (SSURGO corestrictions: a fragipan, dense till, bedrock; else the profile's
                 bottom): the table perches on it, so where z_i >= D_i there is no saturated zone in the soil

Each hour the root zone's layers between the capillary fringe's top, z_i - h_b (Brooks-Corey air-entry head), and D_i
are held at saturation from the table, and what that takes is the hour's gw_in; `drain` becomes the root zone's net
exchange with the table, so the root zone's own closure is unchanged. The table is a boundary condition, not a store:
the first form (a TOPMODEL store whose mean deficit moved with recharge and baseflow) rested on m and T0 nobody had
measured, kept Harvard's table at 0.6 m all summer, saturated its uplands (NEON 6 cm sensors 0.07 to 0.15 m3/m3
wetter) and put woody roots 2 to 4.5 m deep under that site's ridges into a table 2 to 3 m down, erasing the summer
deficit everywhere. Here a ridge's table lies under its restrictive layer and gives nothing; a valley bottom whose
wetness index is SENS_M per metre above the mean stays wet; and the season comes from the survey.
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

ENV = "EXPOSURE_GROUNDWATER"
SENS_M = 0.25
"""TOPMODEL's m / phi (m of table depth per unit of the wetness index): m 0.02 to 0.05 m in humid catchments (Beven 2012,
Rainfall-Runoff Modelling, s.6.3), drainable porosity 0.1 to 0.2 (Johnson 1967, USGS Water-Supply Paper 1662-D)."""
DRY_BELOW_M = 0.5
"""Where the survey records no Wet layer in a month, the table counts this far under the restrictive layer."""


def enabled() -> bool:
    return os.environ.get(ENV, "1") != "0"


@dataclass
class Store:
    """One block's water-table constants on its tile: per cell [n] tensors and the site's scalars."""
    twi: torch.Tensor
    hb: torch.Tensor           # Brooks-Corey air-entry head, m: the capillary fringe's height
    restrict: torch.Tensor     # D_i, m
    soilf: torch.Tensor        # 1 on soil cells, 0 elsewhere
    lam_bar: float
    sens: float


def build(k, twi: np.ndarray, h_b_mm: np.ndarray, restrict_m: np.ndarray, lam_bar: float, device,
          sens: float = SENS_M) -> Store:
    """The Store for a tile from its Cells `k` (on `device`), its TWI, the soil's air-entry head (mm) and restrictive
    depth (m; NaN where no survey: the profile is taken as soils.SOIL_DEPTH_MAX_M)."""
    from exposure.soils import SOIL_DEPTH_MAX_M
    soil = (~k.no_soil & ~k.roof).float()                 # a planted roof holds no water table
    t = lambda a: torch.as_tensor(np.asarray(a, np.float64), dtype=torch.float32, device=device)  # noqa: E731
    return Store(twi=t(np.where(np.isfinite(twi), twi, lam_bar)), hb=torch.clamp(t(np.nan_to_num(h_b_mm, nan=100.0)) / 1000.0, min=1e-3),
                 restrict=t(np.where(np.isfinite(restrict_m), restrict_m, SOIL_DEPTH_MAX_M)), soilf=soil,
                 lam_bar=float(lam_bar), sens=float(sens))


def site_monthly_mean(wet_month_m: Optional[np.ndarray], restrict_m: Optional[np.ndarray], soil: np.ndarray
                      ) -> Tuple[np.ndarray, Dict]:
    """zbar for each month (12, m): the median over the soil cells of the survey's Wet layer top, a cell with none that month
    at DRY_BELOW_M under its restrictive layer. Without the survey fields every month is the dry case."""
    from exposure.soils import SOIL_DEPTH_MAX_M
    soil = np.asarray(soil, bool)
    n = soil.size
    D = np.where(np.isfinite(restrict_m), restrict_m, SOIL_DEPTH_MAX_M) if restrict_m is not None else np.full(n, SOIL_DEPTH_MAX_M)
    W = wet_month_m if wet_month_m is not None else np.full((12, n), np.nan)
    if not soil.any():
        return np.full(12, SOIL_DEPTH_MAX_M + DRY_BELOW_M), {"zbar_m": None, "why": "no soil cells"}
    zb = np.array([float(np.median(np.where(np.isfinite(W[mo][soil]), W[mo][soil], D[soil] + DRY_BELOW_M))) for mo in range(12)])
    return zb, {"zbar_m_by_month": [round(float(v), 3) for v in zb],
                "wet_share_by_month": [round(float(np.isfinite(W[mo][soil]).mean()), 3) for mo in range(12)],
                "restrict_m_p10_p50_p90": [round(float(x), 2) for x in np.percentile(D[soil], [10, 50, 90])],
                "source": "SSURGO cosoilmoist (Wet layer top by month) and corestrictions (restrictive depth)"}


def unit_twi(twi: np.ndarray, wet_month_m: Optional[np.ndarray], restrict_m: Optional[np.ndarray], zbar12: np.ndarray,
             sens: float = SENS_M, chunk: int = 1 << 20) -> Tuple[np.ndarray, Dict]:
    """The wetness index the table's depth reads (`depth`), shifted so each cell's table stands at its own soil unit's
    survey table, not the site's: depth_i = zbar + sens (lambda - twi_eff_i) = W_i + sens (lambda - twi_i), W_i the unit's
    Wet layer top that month (DRY_BELOW_M under its restrictive layer where it has none), the shift the median over the
    months of W_i - zbar. The site's median of a swamp's 0 m and an upland's 2.5 m flips between the two at half the soil:
    Tom Swamp's rule 13 made its swamp soil, the wet share went 0.475 to 0.528, zbar 2.15 to 0 m, and every upland root
    zone sat saturated all year against US-Ha1's autumn dry-down to 0.13 m3/m3 (UC Berkeley, 2026-10-08). Returns
    (twi_eff, record)."""
    from exposure.soils import SOIL_DEPTH_MAX_M
    twi = np.asarray(twi, np.float32)
    n = twi.size
    zb = np.asarray(zbar12, np.float32)[:, None]
    out = np.empty(n, np.float32)
    dz_mean = np.empty(n, np.float32)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        D = (np.where(np.isfinite(restrict_m[a:b]), restrict_m[a:b], SOIL_DEPTH_MAX_M) if restrict_m is not None
             else np.full(b - a, SOIL_DEPTH_MAX_M)).astype(np.float32)
        W = np.asarray(wet_month_m[:, a:b], np.float32) if wet_month_m is not None else np.full((12, b - a), np.nan, np.float32)
        Wc = np.where(np.isfinite(W), W, D[None, :] + DRY_BELOW_M)
        off = np.median(Wc - zb, axis=0)
        out[a:b] = twi[a:b] - off / sens
        dz_mean[a:b] = np.median(Wc, axis=0)
    ok = np.isfinite(twi)
    rec = {"table": "each cell's own soil unit's survey table, shaped by the terrain about it (groundwater.unit_twi)",
           "cell_table_m_p10_p50_p90": [round(float(v), 2) for v in np.percentile(dz_mean[ok], [10, 50, 90])] if ok.any()
           else None}
    return np.where(ok, out, twi), rec


def month_of_hours(year: int, hours: np.ndarray) -> np.ndarray:
    """The calendar month (0 to 11, UTC) of each hour of `year`."""
    t = np.datetime64(f"{int(year)}-01-01T00", "h") + np.asarray(hours, np.int64).astype("timedelta64[h]")
    return (t.astype("datetime64[M]").astype(np.int64) % 12).astype(np.int64)


def lane_zbar(zbar12: np.ndarray, hours: np.ndarray, phase: np.ndarray, group) -> np.ndarray:
    """zbar for every step of every lane [T, B] (m): each hour's month in its own year, the spin-up's for phase 1."""
    zb = np.asarray(zbar12, np.float64)
    out = np.full(hours.shape, float(np.nanmedian(zb)), np.float32)
    for b, (y, s) in enumerate(group):
        for ph, yr in ((1, s if s is not None else y), (2, y)):
            sel = phase[:, b] == ph
            if sel.any():
                out[sel, b] = zb[month_of_hours(yr, hours[sel, b])]
    return out


def _layers(s, k):
    """[(top m, bottom m, state field, theta_s)] of the root zone's layers, top down (water.State, water.Cells)."""
    from exposure.water import ZE, surface
    tsat1 = surface(k)[2]
    if s.theta3 is not None:
        return [(0.0, ZE, "theta1", tsat1), (ZE, k.z_s, "theta2", k.theta_sat), (k.z_s, k.z_r, "theta3", k.theta_sat)]
    return [(0.0, ZE, "theta1", tsat1), (ZE, k.z_r, "theta2", k.theta_sat)]


def depth(zbar: torch.Tensor, g: Store) -> torch.Tensor:
    """The table's depth below the ground per cell, m ([lanes, n] from zbar [lanes, 1])."""
    return torch.clamp(zbar + g.sens * (g.lam_bar - g.twi), min=0.0)


_SCALARS: Dict[Tuple[float, torch.dtype, str], torch.Tensor] = {}


def _on(x, like: torch.Tensor) -> torch.Tensor:
    """`torch.as_tensor(x, dtype, device)` of `like`, a Python number made on the device once and kept: made inside a CUDA
    graph's capture it is a host-to-device copy, which invalidated the dry hour's graph (water.DryHour) on every order once
    the table was on, every hour then eager (Fisher's water block 68 to 98 s on a g4, 2026-10-05). Same values."""
    if isinstance(x, torch.Tensor):
        return torch.as_tensor(x, dtype=like.dtype, device=like.device)
    key = (float(x), like.dtype, str(like.device))
    t = _SCALARS.get(key)
    if t is None:
        t = _SCALARS[key] = torch.as_tensor(x, dtype=like.dtype, device=like.device)
    return t


def _fixed_terms(sbar: torch.Tensor, s, k, g: Store, roots=None):
    """What an hour of the table reads that moves only with zbar: each layer's (1000 frac dz, dz), and the roots' reach."""
    z = depth(sbar, g)
    top = z - g.hb                                     # the capillary fringe's top
    perched = (z < g.restrict).float() * g.soilf       # a table under the restrictive layer is not in the soil
    layers = []
    for a, b, f, tsat in _layers(s, k):
        lo = torch.maximum(_on(a, z), top)
        hi = torch.minimum(_on(b, z), g.restrict)
        dz = torch.clamp(_on(b, z) - a, min=1e-6)
        frac = torch.clamp((hi - lo) / dz, 0.0, 1.0) * perched
        layers.append((1000.0 * frac * dz, dz))
    # roots in the saturated zone transpire unstressed (Fan et al. 2017, PNAS 114:10572): where the fringe reaches the root
    # zone, the hour's unmet basal transpiration Kcb E' - Tr is drawn from the table. A crown 2 to 4.5 m deep over a creek
    # whose table stands above its pan otherwise kept FAO-56's stress from its dry deep layers (the urban forest site)
    reach = perched * (top < k.z_r).float()
    if roots is not None:       # a crown drinks from the soil within its reach (water.RootShare): the share of it the table feeds
        reach = (roots.conv(reach.double() * roots.soil) * roots.inv + reach.double() * roots.local).to(reach.dtype)
    return layers, reach


class Fixed:
    """`_fixed_terms` held while every lane's zbar stays (a month), in buffers a captured hour reads (water.DryHour): they
    were made every hour, about a third of the table's hour and one of the root share's three FFTs. The same values."""

    def __init__(self):
        self.key, self.layers, self.reach, self.mode, self.zero = None, None, None, None, None

    def update(self, key, sbar: torch.Tensor, s, k, g: Store, roots=None) -> None:
        if key == self.key:
            return
        layers, reach = _fixed_terms(sbar, s, k, g, roots)
        if self.reach is None or self.reach.shape != reach.shape:
            self.layers, self.reach = layers, reach
            self.zero = torch.zeros_like(reach)
        else:
            for (p, d), (p1, d1) in zip(self.layers, layers):
                p.copy_(p1)
                d.copy_(d1)
            self.reach.copy_(reach)
        # which terms the table touches this month: a layer it holds nowhere adds 0 everywhere and a reach of 0
        # draws 0, so those terms are skipped (the same values); a captured hour is captured again when this changes
        self.mode = tuple(bool(p.any()) for p, _ in self.layers) + (bool(self.reach.any()),)
        self.key = key


def step(s, k, flux: Dict[str, torch.Tensor], g: Store, roots=None) -> Dict[str, torch.Tensor]:
    """One hour of the table against the root zone, in place on `s` (s.sbar holds the hour's zbar, m [lanes, 1]); with
    `g.fixed` (a `Fixed` the caller keeps current) its zbar terms are read from there."""
    fixed = getattr(g, "fixed", None)
    if fixed is not None and fixed.reach is not None:
        layers, reach, mode = fixed.layers, fixed.reach, fixed.mode
    else:
        layers, reach = _fixed_terms(s.sbar, s, k, g, roots)
        mode = (True,) * (len(layers) + 1)
    from exposure import water as W
    if W._fusing(s.theta1.device):        # the layers and the tail as one compiled pass (water.FUSE_ENV)
        on = [(f, tsat, fdz, dz) for (a, b, f, tsat), (fdz, dz), m in zip(_layers(s, k), layers, mode) if m]
        ths, filled, unmet, drain, aet, tr = W._fused(f"gw{len(on)}{int(mode[-1])}", _gw_apply)(
            [getattr(s, f) for f, _, _, _ in on], [x[2] for x in on], [x[3] for x in on], [x[1] for x in on],
            flux["et0"], flux["ec"], flux["ew"], k.kcb, flux["tr"], reach, flux["drain"], flux["aet"], bool(mode[-1]))
        for (f, _, _, _), th in zip(on, ths):
            setattr(s, f, th)
        return {"drain": drain, "gw_in": filled + unmet, "drain_out": flux["drain"], "aet": aet, "tr": tr,
                "gw_uptake": unmet}
    filled = torch.zeros_like(flux["drain"])
    for (a, b, f, tsat), (fdz, dz), on in zip(_layers(s, k), layers, mode):
        if not on:
            continue
        th = getattr(s, f)
        add = fdz * torch.clamp(tsat - th, min=0.0)
        setattr(s, f, th + add / (1000.0 * dz))
        filled = filled + add
    if mode[-1]:
        e = torch.clamp(flux["et0"] - flux["ec"] - flux["ew"], min=0.0)
        unmet = torch.clamp(k.kcb * e - flux["tr"], min=0.0) * reach
    else:
        unmet = fixed.zero
    return {"drain": flux["drain"] - filled - unmet, "gw_in": filled + unmet, "drain_out": flux["drain"],
            "aet": flux["aet"] + unmet, "tr": flux["tr"] + unmet, "gw_uptake": unmet}


def _gw_apply(ths, fdzs, dzs, tsats, et0, ec, ew, kcb, tr, reach, drain, aet, reaches: bool):
    filled = torch.zeros_like(drain)
    out = []
    for th, fdz, dz, tsat in zip(ths, fdzs, dzs, tsats):
        add = fdz * torch.clamp(tsat - th, min=0.0)
        out.append(th + add / (1000.0 * dz))
        filled = filled + add
    unmet = (torch.clamp(kcb * torch.clamp(et0 - ec - ew, min=0.0) - tr, min=0.0) * reach if reaches
             else torch.zeros_like(drain))
    return out, filled, unmet, drain - filled - unmet, aet + unmet, tr + unmet


def lam_bar_of(twi: np.ndarray) -> float:
    v = twi[np.isfinite(twi)]
    return float(v.mean()) if v.size else 0.0


TWI_CELL_M = 2.0
"""The DEM's cell for the index: the landform's scale, not the survey's micro-relief."""


def site_twi(dem_tif, grid_transform, shape) -> Optional[Dict]:
    """TWI on the 3DEP DEM around the site (sims/dem.py's dem_enu_<cell>m.tif, scene ENU metres), block-averaged to
    TWI_CELL_M, sampled bilinearly at the water grid's cell centres. None without a DEM."""
    import rasterio
    from scipy import ndimage
    from exposure import twi as TW
    if dem_tif is None:
        return None
    with rasterio.open(dem_tif) as d:
        z, T = d.read(1).astype(np.float64), d.transform
        nod = d.nodata
    bad = ~np.isfinite(z) | (z <= -9000) | ((z == nod) if nod is not None else False)
    z = np.where(bad, np.nan, z)
    f = max(1, int(round(TWI_CELL_M / abs(T.a))))
    ny, nx = (z.shape[0] // f) * f, (z.shape[1] // f) * f
    zc = np.nanmean(z[:ny, :nx].reshape(ny // f, f, nx // f, f), axis=(1, 3))
    dx = abs(T.a) * f
    w = TW.twi(zc, np.isfinite(zc), dx)
    lam = lam_bar_of(w)
    gy, gx = shape
    t = grid_transform
    xs = t.c + (np.arange(gx) + 0.5) * t.a
    ys = t.f + (np.arange(gy) + 0.5) * t.e
    X, Y = np.meshgrid(xs, ys)
    col = (X - T.c) / (T.a * f) - 0.5
    row = (Y - T.f) / (T.e * f) - 0.5
    fill = np.where(np.isfinite(w), w, lam)
    out = ndimage.map_coordinates(fill, [row.ravel(), col.ravel()], order=1, mode="nearest")
    return {"twi": out.astype(np.float32), "lambda": lam, "cell_m": dx, "dem": str(dem_tif)}
