"""The DEM must arrive on a metric grid of the requested size, or the fetch must say so."""

import pytest

from fetch import check_dem, pixel_size_m


def test_geographic_pixels_are_measured_in_metres_at_the_site_latitude():
    dx, dy = pixel_size_m("epsg:4326", (1.135655e-5, -9.0083e-6), lat=37.87217)
    assert dx == pytest.approx(1.0, abs=0.01) and dy == pytest.approx(1.0, abs=0.01)


def test_projected_pixels_pass_through():
    assert pixel_size_m("epsg:32610", (0.2, -0.2), lat=37.9) == (0.2, 0.2)


def test_a_one_metre_request_delivered_at_three_metres_fails():
    with pytest.raises(AssertionError, match="requested 1.0 m, delivered 2.99"):
        check_dem("epsg:4326", (3.40697e-5, -2.7025e-5), requested_m=1.0, lat=37.87217)


def test_a_projected_crs_in_feet_fails():
    with pytest.raises(AssertionError, match="not in metres"):
        check_dem("epsg:2227", (3.0, -3.0), requested_m=1.0, lat=37.87217)


def test_the_delivered_grid_is_accepted_within_tolerance():
    check_dem("epsg:4326", (1.135655e-5, -9.0083e-6), requested_m=1.0, lat=37.87217)
    check_dem("epsg:5070", (25.0, -25.0), requested_m=25.0, lat=28.69)
