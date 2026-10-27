"""Koppen classes of known climates from their monthly normals."""

import pytest

from koppen import classify


def test_mediterranean_twitchell_island():
    # The ONEFlux ERA5 1991-2020 normals of US-Twt (Sacramento-San Joaquin Delta); network metadata says Csa.
    ta = [7.854, 9.87, 12.023, 14.112, 16.92, 19.561, 20.457, 20.4, 19.647, 16.434, 11.409, 8.006]
    p = [74.87, 73.24, 55.94, 26.15, 13.79, 3.15, 0.35, 1.31, 1.2, 17.66, 36.37, 69.77]
    # 88 % of the 374 mm falls in winter, so the arid threshold is 10 x 2 x MAT = 294 mm: not B. The ERA5 hottest
    # month is 20.5 deg C (< 22), so "b"; the network's Csa reflects the tower's own warmer record.
    assert classify(ta, p, 38.1) == "Csb"


def test_temperate_oceanic_mieming():
    ta = [-1.044, 0.392, 3.865, 7.317, 11.565, 14.988, 16.669, 16.484, 12.803, 9.161, 3.838, -0.214]
    p = [43.37, 41.36, 55.63, 57.17, 89.24, 97.56, 99.82, 101.58, 70.22, 56.44, 51.89, 48.31]
    assert classify(ta, p, 47.3) == "Dfb"  # the ERA5 cell's coldest month is below 0 deg C; the network says Cfb


def test_tropical_polar_and_southern_hemisphere():
    assert classify([27] * 12, [250] * 12, 1.0) == "Af"
    assert classify([-20, -18, -15, -8, 0, 5, 8, 6, 1, -6, -14, -18], [10] * 12, 70.0) == "ET"
    # Southern-hemisphere summer is October to March: a dry austral winter is "w" there, not "s".
    ta = [24, 24, 22, 19, 16, 13, 13, 15, 18, 20, 22, 23]
    p = [150, 140, 120, 60, 20, 5, 5, 10, 30, 80, 120, 150]
    assert classify(ta, p, -25.0)[:2] == "Cw"


def test_bad_input_is_refused():
    with pytest.raises(ValueError):
        classify([1] * 11, [1] * 12, 0)


def test_network_codes_are_normalised():
    from koppen import normalize
    assert normalize("Bsh") == "BSh" and normalize("cfb") == "Cfb" and normalize("ET") == "ET"
    assert normalize("Xyz") is None and normalize(None) is None and normalize("Cfq") is None
