"""Soil hydraulics: the texture triangle, Brooks-Corey retention, and what each survey supplies."""

import numpy as np

import soils as S


def test_texture_triangle():
    cases = {(90, 5): "sand", (80, 5): "loamy sand", (65, 10): "sandy loam", (40, 20): "loam", (20, 15): "silt loam",
             (30, 33): "clay loam", (10, 33): "silty clay loam", (60, 25): "sandy clay loam", (50, 38): "sandy clay",
             (5, 45): "silty clay", (20, 60): "clay"}
    for (sand, clay), want in cases.items():
        assert S.texture(sand, clay) == want, (sand, clay)


def test_brooks_corey_field_capacity_and_wilting_point_of_loam():
    h = S.hydraulics(40, 20)
    assert h["texture"] == "loam" and abs(h["fc"] - 0.2117) < 1e-3 and abs(h["wp"] - 0.0976) < 1e-3
    assert h["ksat"] == 3.4 and h["psi_f"] == 88.9


def test_measured_values_win_over_the_texture_row():
    h = S.hydraulics(40, 20, fc=0.30, wp=0.14, theta_s=0.48, ksat_mm_h=20.0)
    assert (h["fc"], h["wp"], h["theta_sat"], h["ksat"]) == (0.30, 0.14, 0.48, 20.0)


def test_ssurgo_root_zone_mean_and_urban_land():
    hz = [{"hzdept_r": 0, "hzdepb_r": 20, "sandtotal_r": 40, "claytotal_r": 20, "wthirdbar_r": 30, "wfifteenbar_r": 14,
           "ksat_r": 9.0, "dbthirdbar_r": 1.4, "wsatiated_r": 45},
          {"hzdept_r": 20, "hzdepb_r": 150, "sandtotal_r": 30, "claytotal_r": 30, "wthirdbar_r": 34, "wfifteenbar_r": 20,
           "ksat_r": 3.0, "dbthirdbar_r": 1.5, "wsatiated_r": 42}]
    h = S.from_ssurgo_horizons(hz, z_m=1.0)
    assert abs(h["fc"] - (0.30 * 20 + 0.34 * 80) / 100) < 1e-9 and abs(h["ksat"] - (9 * 20 + 3 * 80) / 100 * 3.6) < 1e-9
    assert S.from_ssurgo_horizons([{"hzdept_r": 0, "hzdepb_r": 150}]) is None


def test_polaris_units_and_canopy_storage():
    h = S.from_polaris({"sand": 40, "clay": 20, "thetas": 0.45, "thetar": 0.05, "ksat": 1.2, "lambda": 0.3, "hb": 1.0})
    assert h["ksat"] == 12.0 and abs(h["h_b"] - 101.97) < 1e-9
    assert abs(float(S.s_max(np.array([3.0]))[0]) - (0.935 + 1.494 - 0.05175)) < 1e-9
    assert abs(float(S.lai_from_gap(np.array([np.exp(-1.5)]))[0]) - 3.0) < 1e-9


def test_polaris_is_continuous_across_its_pixel_edges():
    """Taken whole per pixel, POLARIS drew straight 30 m divides in the soil water; between pixel centers it blends."""
    step = S.POLARIS_STEP_DEG

    def pixel(clon, clat):
        v = np.round(clon / step - 0.5) * 0.01 + 0.2                  # a property rising 0.01 per pixel eastward
        return {k: v.copy() for k in S.KEYS}

    lon = (np.array([10.0, 10.49, 10.51, 11.0]) + 0.5) * step          # 10 and 11 are pixel centers
    lat = np.full(4, 40.5 * step)
    out = S.polaris_bilinear(lon, lat, pixel)
    assert np.allclose(out["fc"][[0, 3]], [0.3, 0.31], atol=1e-9), "a pixel center keeps its pixel's value"
    assert abs(out["fc"][2] - out["fc"][1]) < 1e-3, "no step at the pixel edge"
    assert np.allclose(out["ksat"][[0, 3]], [0.3, 0.31], atol=1e-9), "logs at the centers return the value"


def test_polaris_renormalizes_over_pixels_with_a_value():
    step = S.POLARIS_STEP_DEG

    def pixel(clon, clat):
        v = np.where(np.round(clon / step - 0.5) >= 11, np.nan, 0.25)
        return {k: v.copy() for k in S.KEYS}

    out = S.polaris_bilinear(np.array([10.7 * step]), np.array([40.5 * step]), pixel)
    assert np.allclose(out["fc"], 0.25)
