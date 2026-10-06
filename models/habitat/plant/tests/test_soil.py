"""SoilGrids point sampling: physical units on land, NaN over open water and outside the North America grid."""

import numpy as np
import pytest

from ranges.config import data_root

D = data_root() / "work/soil"
pytestmark = pytest.mark.skipif(not (D / "soil_phh2o_na7p5s.i16").exists(), reason="needs the soil layers")


def test_soil_points():
    from ranges import soil
    s = soil.SoilPoints(D)
    v = s.at(np.array([-93.6, -70.0, 10.0]), np.array([42.0, 35.0, 45.0]))   # Iowa farmland, open Atlantic, Europe
    ph, clay, sand, soc = v[0]
    assert 5.0 < ph < 8.0 and 5 < clay < 50 and 5 < sand < 80 and 1 < soc < 100
    assert np.isnan(v[1]).all() and np.isnan(v[2]).all()
