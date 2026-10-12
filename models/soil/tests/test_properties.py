"""Offline tests of the soil-property model: the USDA texture triangle and the terrain covariates' point / grid
equivalence (no network: a synthetic hillslope stands in for 3DEP)."""
import math

import numpy as np
import pytest

from properties import terrain as T
from properties.products import interval_mean, LAYERS
from properties.texture import texture_class, CLASSES

TEXTBOOK = {(92, 3): "sand", (82, 5): "loamy sand", (65, 10): "sandy loam", (40, 20): "loam", (20, 15): "silt loam",
            (5, 5): "silt", (60, 25): "sandy clay loam", (32, 33): "clay loam", (10, 33): "silty clay loam",
            (50, 40): "sandy clay", (5, 45): "silty clay", (20, 60): "clay", (43, 24.5): "loam"}


def test_texture_textbook_points():
    got = texture_class([k[0] for k in TEXTBOOK], [k[1] for k in TEXTBOOK])
    assert list(got) == list(TEXTBOOK.values())


def test_texture_covers_the_whole_triangle_once():
    s, c = np.meshgrid(np.arange(0, 100.5, 0.5), np.arange(0, 100.5, 0.5))
    m = s + c <= 100
    got = texture_class(s[m], c[m])
    assert (got != "").all() and set(got) == set(CLASSES)


def test_texture_unknown_is_blank():
    assert list(texture_class([np.nan, 40], [20, np.nan])) == ["", ""]


def test_interval_mean_weights_by_thickness():
    prof = np.array([[1.0, 2.0, 3.0]])                     # 0-5, 5-15, 15-30 cm
    assert interval_mean(prof, 0, 30)[0] == pytest.approx((5 * 1 + 10 * 2 + 15 * 3) / 30)
    assert interval_mean(prof, 0, 5)[0] == pytest.approx(1.0)
    assert np.isnan(interval_mean(np.full((1, len(LAYERS)), np.nan), 0, 30)[0])


def _hillslope(lat=37.0, seed=0):
    n = 2 * T.HALF + 1
    dy = T.PIX_DEG * 111320.0
    dx = dy * math.cos(math.radians(lat))
    yy, xx = np.mgrid[0:n, 0:n]
    rng = np.random.default_rng(seed)
    return 300 + 0.08 * yy * dy + 0.03 * xx * dx + 4 * np.sin(xx / 17) * np.cos(yy / 23) + rng.normal(0, 0.2, (n, n))


@pytest.mark.parametrize("lat", [30.0, 37.0, 46.0])
def test_point_covariates_equal_grid_centre(lat):
    z = _hillslope(lat)
    p = T.point_covariates(z, lat)
    g = T.grid_covariates(z, lat)
    for k in T.NAMES:
        if k == "twi":       # flow routing over the central 1 km in both paths
            sub = z[T.HALF - 50:T.HALF + 51, T.HALF - 50:T.HALF + 51]
            ref = T.grid_covariates(sub, lat)["twi"][50, 50]
        else:
            ref = g[k][T.HALF, T.HALF]
        assert p[k] == pytest.approx(float(ref), rel=1e-6, abs=1e-9), k


def test_point_covariates_nodata_centre():
    z = _hillslope()
    z[T.HALF, T.HALF] = np.nan
    assert all(np.isnan(v) for v in T.point_covariates(z, 37.0).values())


def test_d8_area_conserves_mass_on_a_plane():
    z = np.add.outer(np.arange(20.0)[::-1], np.zeros(10))  # tilted plane draining to the last row
    a = T.d8_area(z, 10.0)
    assert a[-1].sum() == pytest.approx(z.size * 100.0)
