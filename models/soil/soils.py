"""Soil hydraulic properties per cell, and the canopy's storage from LiDAR.

Per cell: theta_sat, theta_r, field capacity (33 kPa), wilting point (1500 kPa), K_sat (mm/h), the Brooks-Corey lambda
and bubbling pressure, and the Green-Ampt wetting-front suction. A survey's own measured values win (SSURGO's 1/3-bar and
15-bar water contents, K_sat, saturated content; POLARIS's Brooks-Corey fit); the texture's row fills the rest.
Texture parameters: Rawls, Brakensiek and Miller (1983), Table 1 (geometric means).
"""

from typing import Dict, Optional, Sequence

import numpy as np

# class: theta_s, theta_r, psi_f (mm), lambda, h_b (mm), K_sat (mm/h)
RAWLS = {
    "sand": (0.437, 0.020, 49.5, 0.694, 72.6, 117.8),
    "loamy sand": (0.437, 0.035, 61.3, 0.553, 86.9, 29.9),
    "sandy loam": (0.453, 0.041, 110.1, 0.378, 146.6, 10.9),
    "loam": (0.463, 0.027, 88.9, 0.252, 111.5, 3.4),
    "silt loam": (0.501, 0.015, 166.8, 0.234, 207.6, 6.5),
    "sandy clay loam": (0.398, 0.068, 218.5, 0.319, 280.8, 1.5),
    "clay loam": (0.464, 0.075, 208.8, 0.242, 258.9, 1.0),
    "silty clay loam": (0.471, 0.040, 273.0, 0.177, 325.6, 1.0),
    "sandy clay": (0.430, 0.109, 239.0, 0.223, 291.7, 0.6),
    "silty clay": (0.479, 0.056, 292.2, 0.150, 341.9, 0.5),
    "clay": (0.475, 0.090, 316.3, 0.165, 373.0, 0.3),
}
H_FC_MM = 3366.0          # 33 kPa
H_WP_MM = 152957.0        # 1500 kPa

ROOF_SUBSTRATE = {"theta_sat": 0.55, "fc": 0.35, "wp": 0.08, "theta_r": 0.02, "ksat": 36.0, "lam": 0.5, "psi_f": 50.0}
"""An engineered green-roof substrate at the FLL (2018) guideline values (water capacity 35 %, permeability 0.6 mm/min)."""


def texture(sand: float, clay: float) -> str:
    """USDA texture class from sand and clay percent (silt the rest); silt maps to silt loam (no Rawls row)."""
    silt = 100.0 - sand - clay
    if silt + 1.5 * clay < 15:
        return "sand"
    if silt + 2 * clay < 30:
        return "loamy sand"
    if clay >= 40:
        return "silty clay" if silt >= 40 else ("sandy clay" if sand > 45 else "clay")
    if clay >= 35 and sand > 45:
        return "sandy clay"
    if clay >= 27:
        return "silty clay loam" if sand <= 20 else ("clay loam" if sand <= 45 else "sandy clay loam")
    if clay >= 20 and silt < 28 and sand > 45:
        return "sandy clay loam"
    if silt >= 50 and (clay >= 12 or silt < 80):
        return "silt loam"
    if silt >= 80:
        return "silt loam"
    if clay >= 7 and sand <= 52 and silt >= 28:
        return "loam"
    return "sandy loam"


def brooks_corey(h_mm: float, theta_s: float, theta_r: float, lam: float, h_b: float) -> float:
    """Water content at suction h (mm): theta_r + (theta_s - theta_r) (h_b / h)^lambda above the bubbling pressure."""
    if h_mm <= h_b:
        return theta_s
    return theta_r + (theta_s - theta_r) * (h_b / h_mm) ** lam


def hydraulics(sand: float, clay: float, fc: Optional[float] = None, wp: Optional[float] = None,
               theta_s: Optional[float] = None, ksat_mm_h: Optional[float] = None, lam: Optional[float] = None,
               h_b: Optional[float] = None, theta_r: Optional[float] = None) -> Dict[str, float]:
    """A complete hydraulic set: what the source measured, the texture's Rawls row for the rest."""
    cls = texture(sand, clay)
    ts, tr, psi, lm, hb, ks = RAWLS[cls]
    ts = theta_s if theta_s is not None else ts
    tr = theta_r if theta_r is not None else tr
    lm = lam if lam is not None else lm
    hb = h_b if h_b is not None else hb
    out = {"texture": cls, "theta_sat": ts, "theta_r": tr, "lam": lm, "h_b": hb, "psi_f": psi,
           "ksat": ksat_mm_h if ksat_mm_h is not None else ks,
           "fc": fc if fc is not None else brooks_corey(H_FC_MM, ts, tr, lm, hb),
           "wp": wp if wp is not None else brooks_corey(H_WP_MM, ts, tr, lm, hb)}
    out["wp"] = min(out["wp"], out["fc"] - 0.01)
    out["fc"] = min(out["fc"], ts - 0.01)
    return out


def depth_mean(horizons: Sequence[Dict], key: str, z_m: float) -> Optional[float]:
    """Thickness-weighted mean of a horizon property over 0 to z_m (SSURGO depths in cm)."""
    num = den = 0.0
    for h in horizons:
        v = h.get(key)
        top, bot = h.get("hzdept_r"), h.get("hzdepb_r")
        if v is None or top is None or bot is None:
            continue
        t = max(0.0, min(float(bot), z_m * 100) - float(top))
        if t > 0:
            num += float(v) * t
            den += t
    return num / den if den else None


def from_ssurgo_horizons(horizons: Sequence[Dict], z_m: float = 1.0) -> Optional[Dict[str, float]]:
    """A SSURGO map unit's dominant component over the root zone; None for a unit with no profile (Urban land)."""
    sand, clay = depth_mean(horizons, "sandtotal_r", z_m), depth_mean(horizons, "claytotal_r", z_m)
    if sand is None or clay is None:
        return None
    pct = lambda k: (lambda v: v / 100.0 if v is not None else None)(depth_mean(horizons, k, z_m))   # noqa: E731
    bd = depth_mean(horizons, "dbthirdbar_r", z_m)
    ts = pct("wsatiated_r") or (1.0 - bd / 2.65 if bd else None)
    k = depth_mean(horizons, "ksat_r", z_m)
    return hydraulics(sand, clay, fc=pct("wthirdbar_r"), wp=pct("wfifteenbar_r"), theta_s=ts,
                      ksat_mm_h=k * 3.6 if k is not None else None)


def from_polaris(layers: Dict[str, float]) -> Dict[str, float]:
    """POLARIS depth layers over the root zone (already averaged): sand, clay %, thetas, thetar, ksat cm/h, lambda,
    hb kPa."""
    return hydraulics(layers["sand"], layers["clay"], theta_s=layers["thetas"], theta_r=layers["thetar"],
                      ksat_mm_h=layers["ksat"] * 10.0, lam=layers["lambda"], h_b=layers["hb"] * 101.97)


def s_max(lai: np.ndarray) -> np.ndarray:
    """Canopy storage capacity, mm (von Hoyningen-Huene 1981)."""
    return np.where(lai > 0, 0.935 + 0.498 * lai - 0.00575 * lai ** 2, 0.0)


def lai_from_gap(p_gap: np.ndarray, g: float = 0.5) -> np.ndarray:
    """Effective LAI from a LiDAR column's gap fraction (ground returns over all returns), Beer-Lambert."""
    return -np.log(np.clip(p_gap, 1e-3, 1.0)) / g
