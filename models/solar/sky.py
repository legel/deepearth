"""The measured sky, hourly: clear-sky reference, the clear-sky index k_c, beam and diffuse from a tower's SW_IN.

Clear sky: the measured atmosphere's where it is known, REST2 (Gueymard 2008) driven by MERRA-2 aerosol, water vapor
and ozone, as NSRDB v4 publishes it hourly (`clearsky_ghi`, `clearsky_dni`, `clearsky_dhi`); each hour's 5-minute
Ineichen-Perez shape (pvlib, NREL SPA geometry) is scaled to it (`with_clear_sky`). Elsewhere Ineichen-Perez with
SoDa's monthly Linke turbidity. Diffuse: measured where the tower measures it; otherwise the Engerer2 separation,
refit on the US towers that measure diffuse (`ENGERER2_US`, `fit`). Beam: whatever of the tower's GHI the diffuse
leaves, so DNI cos z + DHI is the tower's GHI in every hour.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

SUB = 12                     # 5-min samples per hour
KC_MIN_GHI_CS = 10.0         # W m-2: below this the hour has no sky coefficient (night, sun on the horizon)
CS_MIN = 5.0
"""W m-2: an hour Ineichen puts below this keeps Ineichen's clear sky (the sun on the horizon has no shape to scale)."""

ENGERER2_US = (-0.1844909229993944, -2.443171814299747, 4.261545108345926, 0.0020934596424676843,
               0.0038915519506463204, -3.847387131650481, 1.8559664817403347)
"""Engerer2 (c, b0..b5) refit on 518,865 hours of measured diffuse at 51 US towers on REST2's clear sky (`fit`).
Leave-one-site-out RMSE of DHI 53.0 W m-2 (MBE -1.6), against 57.6 for Erbs and 72.4 for NSRDB's own diffuse."""

ENGERER2_START = (0.042336, -3.7912, 7.5479, -0.010036, 0.003148, -5.3146, 1.7073)
"""Engerer (2015)'s published 1-min set: the fit's starting point only."""


def geometry(year: int, lat: float, lon: float, elev_m: float,
             cs: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
    """5-min sun geometry and clear sky of a UTC year, sample k at hour start + 2.5 + 5k minutes. `cs`: the hourly
    measured-atmosphere clear sky (ghi, dni, dhi W m-2; NaN where absent), to which each hour's 5-min clear sky is
    scaled (`with_clear_sky`)."""
    import pandas as pd
    import pvlib
    t = pd.date_range(f"{year}-01-01 00:02:30", f"{year + 1}-01-01 00:00:00", freq="5min", tz="UTC")
    sp = pvlib.solarposition.get_solarposition(t, lat, lon, altitude=elev_m, method="nrel_numpy")
    e0 = pvlib.irradiance.get_extra_radiation(t).to_numpy()
    am = pvlib.atmosphere.get_relative_airmass(sp["apparent_zenith"]).to_numpy()
    ama = pvlib.atmosphere.get_absolute_airmass(am, pvlib.atmosphere.alt2pres(elev_m))
    tl = pvlib.clearsky.lookup_linke_turbidity(t, lat, lon).to_numpy()
    ine = pvlib.clearsky.ineichen(sp["apparent_zenith"].to_numpy(), ama, tl, altitude=elev_m, dni_extra=e0)
    z = sp["zenith"].to_numpy()
    g = {"zenith": z, "azimuth": sp["azimuth"].to_numpy(), "e0": e0, "linke": tl,
         "ghi_cs": np.nan_to_num(np.asarray(ine["ghi"], dtype=float)),
         "dni_cs": np.nan_to_num(np.asarray(ine["dni"], dtype=float)),
         "dhi_cs": np.nan_to_num(np.asarray(ine["dhi"], dtype=float)),
         "cosz": np.clip(np.cos(np.radians(z)), 0.0, None), "eot": sp["equation_of_time"].to_numpy()}
    return with_clear_sky(g, cs)


def with_clear_sky(g: Dict[str, np.ndarray], cs: Optional[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """`g` with each hour's 5-min ghi_cs, dni_cs, dhi_cs scaled to the measured-atmosphere hourly `cs` where it has
    all three, keeping Ineichen's shape within the hour; `cs_measured` marks those hours (False: the Linke climatology)."""
    n = len(g["ghi_cs"]) // SUB
    if cs is None:
        return dict(g, cs_measured=np.zeros(n, bool))
    got = np.isfinite(cs["ghi"]) & np.isfinite(cs["dni"]) & np.isfinite(cs["dhi"])
    out = dict(g, cs_measured=got)
    for k in ("ghi", "dni", "dhi"):
        h = hourly(g[f"{k}_cs"])
        use = got & (h > CS_MIN)
        ratio = np.ones(n)
        ratio[use] = np.maximum(np.asarray(cs[k], np.float64)[use], 0.0) / h[use]
        out[f"{k}_cs"] = g[f"{k}_cs"] * np.repeat(ratio, SUB)
    return out


def hourly(a: np.ndarray) -> np.ndarray:
    return a.reshape(-1, SUB).mean(axis=1)


def predictors(g: Dict[str, np.ndarray], ghi: np.ndarray, lon: float) -> Dict[str, np.ndarray]:
    """Hourly separation inputs: clearness k_t, clear-sky clearness k_tc, their difference, the clear-sky excess
    k_de, zenith [deg] and apparent solar time [h]."""
    ext = hourly(g["e0"] * g["cosz"])
    ghi_cs = hourly(g["ghi_cs"])
    with np.errstate(divide="ignore", invalid="ignore"):
        kt = np.where(ext > 1.0, ghi / ext, np.nan)
        ktc = np.where(ext > 1.0, ghi_cs / ext, np.nan)
        kde = np.where(ghi > 0, np.maximum(0.0, 1.0 - ghi_cs / ghi), 0.0)
    zen = np.degrees(np.arccos(np.clip(hourly(g["cosz"]), 0, 1)))
    utc_mid = (np.arange(len(ghi)) % 24) + 0.5
    ast = np.mod(utc_mid + lon / 15.0 + hourly(g["eot"]) / 60.0, 24.0)
    return {"kt": np.clip(kt, 0, 1.5), "ktc": ktc, "dktc": ktc - kt, "kde": kde, "zen": zen, "ast": ast, "ext": ext}


def engerer2(p: Sequence[float], x: Dict[str, np.ndarray]) -> np.ndarray:
    """Diffuse fraction, Engerer2 (Engerer 2015; Bright and Engerer 2019):
    k_d = c + (1 - c) / (1 + exp(b0 + b1 k_t + b2 AST + b3 z + b4 dk_tc)) + b5 k_de."""
    c, b0, b1, b2, b3, b4, b5 = p
    kd = c + (1 - c) / (1 + np.exp(b0 + b1 * x["kt"] + b2 * x["ast"] + b3 * x["zen"] + b4 * x["dktc"])) + b5 * x["kde"]
    return np.clip(kd, 0.0, 1.0)


def erbs(x: Dict[str, np.ndarray]) -> np.ndarray:
    """Diffuse fraction, Erbs, Klein and Duffie (1982): the fallback where a predictor is missing."""
    kt = x["kt"]
    return np.where(kt <= 0.22, 1.0 - 0.09 * kt,
                    np.where(kt <= 0.80, 0.9511 - 0.1604 * kt + 4.388 * kt ** 2 - 16.638 * kt ** 3 + 12.336 * kt ** 4,
                             0.165))


def fill_ghi(ghi: np.ndarray, ghi_cs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(GHI with every daytime gap filled, the filled hours). A gap is interpolated in k_c, not in GHI, so the fill
    follows the sun's own course through the day; night hours stay 0. Daytime irradiance is never 0 for want of data."""
    ghi = np.asarray(ghi, np.float64)
    day = ghi_cs > KC_MIN_GHI_CS
    have = np.isfinite(ghi) & day
    gap = ~np.isfinite(ghi) & day
    out = np.where(np.isfinite(ghi), ghi, 0.0)
    if gap.any():
        if not have.any():
            raise ValueError("no daytime hour of the year has irradiance from any source")
        i = np.arange(len(ghi))
        kc = np.interp(i[gap], i[have], ghi[have] / ghi_cs[have])
        out[gap] = kc * ghi_cs[gap]
    return out, gap


def split(ghi: np.ndarray, dhi_meas: np.ndarray, g: Dict[str, np.ndarray], x: Dict[str, np.ndarray],
          params: Sequence[float] = ENGERER2_US) -> Dict[str, np.ndarray]:
    """Hourly k_c, k_b, DHI (measured first) and DNI from a tower's GHI. The beam is capped at 1.1 clear-sky DNI and
    k_b at 1.3; what the caps withhold goes to diffuse, so DNI cos z + DHI = GHI exactly in every hour."""
    ghi_cs, dhi_cs = hourly(g["ghi_cs"]), hourly(g["dhi_cs"])
    kd = engerer2(params, x)
    kd = np.where(np.isnan(kd), erbs(x), kd)
    dhi = np.where(~np.isnan(dhi_meas), np.minimum(dhi_meas, ghi), kd * ghi)
    dhi = np.where(x["ext"] > 1.0, dhi, 0.0)
    bh, bh_cs = np.maximum(ghi - dhi, 0.0), np.maximum(ghi_cs - dhi_cs, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        kc = np.where(ghi_cs > KC_MIN_GHI_CS, ghi / ghi_cs, np.nan)
        kb = np.where(bh_cs > 1.0, np.minimum(bh / bh_cs, 1.3), 0.0)
        cz = hourly(g["cosz"])
        dni = np.where(cz > 0.0175, bh / np.maximum(cz, 0.0175), 0.0)          # below 1 degree the beam is 0
        dni = np.minimum(dni, 1.1 * hourly(g["dni_cs"]) + 1e-9)
        kept = np.minimum(dni * cz, np.where(bh_cs > 1.0, kb * bh_cs, 0.0))
        dhi = np.where(x["ext"] > 1.0, ghi - kept, ghi)
        dni = np.where(cz > 0.0175, kept / np.maximum(cz, 0.0175), 0.0)
        kb = np.where(bh_cs > 1.0, kept / bh_cs, 0.0)
    return {"kc": kc, "kb": kb, "dhi": dhi, "dni": dni, "cosz": cz}


def sky(year: int, lat: float, lon: float, elev_m: float, ghi: np.ndarray, dhi_meas: np.ndarray,
        cs: Optional[Dict[str, np.ndarray]] = None, params: Sequence[float] = ENGERER2_US) -> Dict[str, np.ndarray]:
    """One UTC year of the tower's sky: `ghi` [W m-2] hourly (NaN where nothing measured the hour: a daytime gap is
    filled by `fill_ghi` and marked in `ghi_filled`), `dhi_meas` its diffuse (NaN where not measured)."""
    g = geometry(year, lat, lon, elev_m, cs)
    ghi, filled = fill_ghi(ghi, hourly(g["ghi_cs"]))
    x = predictors(g, ghi, lon)
    s = split(ghi, dhi_meas, g, x, params)
    return dict(s, ghi=ghi, ghi_filled=filled, ghi_cs=hourly(g["ghi_cs"]),
                zenith=np.degrees(np.arccos(np.clip(s["cosz"], 0, 1))), e0=hourly(g["e0"]),
                cs_measured=g["cs_measured"])


def fit(samples: Dict[str, Dict[str, np.ndarray]]) -> Dict:
    """Leave-one-site-out comparison of Engerer2 (refit), Erbs and a satellite diffuse on hourly measured diffuse:
    per-model RMSE and MBE of DHI (W m-2), and Engerer2's parameters fitted on every site. `samples[site]` holds
    kt, ast, zen, dktc, kde, ghi, dhi and sat_kd (NaN where no satellite diffuse)."""
    from scipy.optimize import least_squares

    def stack(sites):
        return {k: np.concatenate([samples[s][k] for s in sites])
                for k in ("kt", "ast", "zen", "dktc", "kde", "ghi", "dhi", "sat_kd")}

    def resid(p, x):
        return (engerer2(p, x) * x["ghi"] - x["dhi"]) / 100.0

    sites = sorted(samples)
    err = {"engerer2": [], "erbs": [], "satellite": []}
    for s in sites:
        tr, te = stack([o for o in sites if o != s]), stack([s])
        p = least_squares(resid, ENGERER2_START, args=(tr,), loss="soft_l1").x
        err["engerer2"].append(engerer2(p, te) * te["ghi"] - te["dhi"])
        err["erbs"].append(erbs(te) * te["ghi"] - te["dhi"])
        err["satellite"].append(np.where(np.isnan(te["sat_kd"]), np.nan, te["sat_kd"] * te["ghi"] - te["dhi"]))
    stats = {}
    for m, e in err.items():
        e = np.concatenate(e)
        ok = ~np.isnan(e)
        stats[m] = ({"rmse": float(np.sqrt(np.mean(e[ok] ** 2))), "mbe": float(np.mean(e[ok])), "n": int(ok.sum())}
                    if ok.any() else {"rmse": float("inf"), "mbe": None, "n": 0})
    p_all = least_squares(resid, ENGERER2_START, args=(stack(sites),), loss="soft_l1").x
    return {"engerer2": [float(v) for v in p_all], "loso": stats, "sites": sites}
