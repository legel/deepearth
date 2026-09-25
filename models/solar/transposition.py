"""Irradiance on a surface point from the hour's sky: Hay-Davies transposition and the point's own geometry.

For a point with unit normal n, sky-view factor V, lit fraction L of the sun's direction s (1 in the open, 0 in
shadow, the canopy's transmittance through a crown), and albedo rho of the ground around it:

    E = B L max(n.s, 0) + D V + rho GHI (1 - n_z) / 2

where B and D are the Hay and Davies (1980) anisotropic sky folded into a beam and an isotropic part:

    A = DNI / E0,   B = DNI + A DHI / cos z,   D = (1 - A) DHI

On open level ground (V = 1, n_z = 1, L = 1) E reduces to DNI cos z + DHI, the tower's GHI (`sky.split`).
"""

from typing import Dict

import numpy as np

ALBEDO = 0.2
MIN_COSZ = 0.0175                     # the sun one degree up: below it the circumsolar beam is 0


def hay_davies(dni: np.ndarray, dhi: np.ndarray, zenith_deg: np.ndarray, e0: np.ndarray) -> Dict[str, np.ndarray]:
    """The anisotropy index A and the beam and isotropic diffuse the point sums [W m-2]."""
    cz = np.cos(np.radians(zenith_deg))
    a = np.where(e0 > 0, np.clip(dni / np.maximum(e0, 1e-6), 0, 1), 0.0)
    circ = np.where(cz > MIN_COSZ, a * dhi / np.maximum(cz, MIN_COSZ), 0.0)
    return {"a": a, "beam": np.where(cz > 0, dni + circ, 0.0), "diffuse": (1.0 - a) * dhi}


def irradiance(beam: np.ndarray, diffuse: np.ndarray, ghi: np.ndarray, cos_incidence: np.ndarray,
               lit: np.ndarray, svf: np.ndarray, normal_z: np.ndarray, albedo: float = ALBEDO) -> np.ndarray:
    """E [W m-2] on a point: `cos_incidence` n.s, `lit` the share of the sun's beam reaching it, `svf` its
    sky-view factor, `normal_z` its normal's vertical component."""
    return (beam * lit * np.maximum(cos_incidence, 0.0) + diffuse * svf
            + albedo * ghi * (1.0 - normal_z) / 2.0)
