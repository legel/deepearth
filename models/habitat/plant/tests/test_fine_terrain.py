"""Terrain derivatives on analytic surfaces."""
import numpy as np

from ranges.fine import fine_predictors, terrain


def test_plane_rising_east():
    dx = dy = 230.0
    x = np.arange(60) * dx
    z = np.tile(x * np.tan(np.radians(10.0)), (40, 1))
    t = terrain(z, dx, dy)
    inner = (slice(5, -5), slice(5, -5))
    assert np.allclose(t["fine_slope"][inner], 10.0, atol=1e-6)
    assert np.allclose(t["fine_eastness"][inner], -np.sin(np.radians(10.0)), atol=1e-6)   # faces west
    assert np.allclose(t["fine_northness"][inner], 0.0, atol=1e-6)
    assert np.allclose(t["fine_tpi"][inner], 0.0, atol=1e-6)                                # planar: no relief


def test_lapse_rate_adjusts_only_temperature():
    names = ["wc2.1_30s_bio_1", "wc2.1_30s_bio_12", "wc2.1_30s_elev"]
    wc = np.array([[10.0, 500.0, 1000.0]])
    fine = np.array([[1500.0, 5.0, 0.1, 0.0, 20.0]])
    x, n = fine_predictors(wc, names, fine)
    assert n[:3] == ["fine_bio_1", "wc2.1_30s_bio_12", "fine_elev"]
    assert np.isclose(x[0, 0], 10.0 - 0.0065 * 500.0) and x[0, 1] == 500.0 and x[0, 2] == 1500.0
