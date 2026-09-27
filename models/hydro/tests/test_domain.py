"""Gauge snapping: the one piece of domain assembly that needs no raster on disk.

The published gauge coordinate can land a cell or two off the burned centreline, and reading a
dry floodplain cell beside the creek reports almost no discharge. On site3 the gauge probe is
the difference between a +5 h and a +29 h lag against an observed +4.5 h, so where this lands
is load-bearing.
"""

import numpy as np
import pytest
import rasterio
from affine import Affine

import domain
from sites import get_site


def _profile(transform, crs="epsg:5070"):
    return {"transform": transform, "crs": rasterio.crs.CRS.from_string(crs)}


def _grid_with_channel_at(shape, rc, floor=1.0, ground=10.0):
    z = np.full(shape, ground, dtype=np.float32)
    z[rc] = floor
    return z


def _site_at(lat, lon):
    """site3 with the gauge moved to a chosen coordinate, leaving the registry untouched."""
    import dataclasses
    site = get_site("site3")
    return dataclasses.replace(site, gauge=dataclasses.replace(site.gauge, lat=lat, lon=lon))


def _transform_centred_on(x, y, dx, shape):
    """North-up affine placing projected (x, y) at the centre of the grid."""
    rows, cols = shape
    return Affine(dx, 0.0, x - cols / 2 * dx, 0.0, -dx, y + rows / 2 * dx)


def _project(site):
    from pyproj import Transformer
    return Transformer.from_crs("epsg:4326", "epsg:5070", always_xy=True).transform(
        site.gauge.lon, site.gauge.lat)


def test_snaps_onto_the_lowest_cell_within_the_search_radius():
    site = _site_at(28.7041629, -81.2906221)
    dx, shape = 5.0, (41, 41)
    x, y = _project(site)
    tf = _transform_centred_on(x, y, dx, shape)
    z = _grid_with_channel_at(shape, (20, 18))  # channel two cells west of the published point
    rc = domain.snap_gauge(site, z, _profile(tf), dx, search_m=25.0)
    assert rc == (20, 18)


def test_does_not_reach_beyond_the_search_radius():
    site = _site_at(28.7041629, -81.2906221)
    dx, shape = 5.0, (41, 41)
    x, y = _project(site)
    tf = _transform_centred_on(x, y, dx, shape)
    z = _grid_with_channel_at(shape, (20, 2))  # 18 cells = 90 m away, outside a 25 m search
    rc = domain.snap_gauge(site, z, _profile(tf), dx, search_m=25.0)
    assert rc != (20, 2)
    assert abs(rc[1] - 20) <= 5


def test_stays_inside_the_grid_at_an_edge():
    """The search window is clipped, so a gauge near the boundary must not index out of bounds."""
    site = _site_at(28.7041629, -81.2906221)
    dx, shape = 5.0, (21, 21)
    x, y = _project(site)
    tf = Affine(dx, 0.0, x, 0.0, -dx, y)  # puts the gauge at the very top-left corner
    z = np.full(shape, 10.0, dtype=np.float32)
    r, c = domain.snap_gauge(site, z, _profile(tf), dx, search_m=25.0)
    assert 0 <= r < shape[0] and 0 <= c < shape[1]


def test_ignores_nodata_when_choosing_the_lowest_cell():
    """NaN outside the domain must never win the argmin and put the probe off the map."""
    site = _site_at(28.7041629, -81.2906221)
    dx, shape = 5.0, (41, 41)
    x, y = _project(site)
    tf = _transform_centred_on(x, y, dx, shape)
    z = _grid_with_channel_at(shape, (20, 21))
    z[19, 19] = np.nan
    rc = domain.snap_gauge(site, z, _profile(tf), dx, search_m=25.0)
    assert rc == (20, 21)


def test_a_site_without_a_gauge_is_refused():
    with pytest.raises(AssertionError, match="no gauge"):
        domain.snap_gauge(get_site("main_aoi"), np.zeros((4, 4), np.float32),
                          _profile(Affine.identity()), 5.0)


def test_manning_by_land_cover_takes_the_published_flood_plain_values(monkeypatch):
    """Chow (1959) Table 5-6 normal values by NLCD class; an unlisted class keeps the scalar."""
    import domain
    codes = np.array([[90, 95, 42], [11, 81, 99]], dtype=np.int32)
    monkeypatch.setattr(domain, "_warp_onto", lambda path, shape, profile, resampling, dtype=None, fill=0: codes)
    n = domain.manning_nlcd(type("S", (), {"nlcd_landcover": None})(), codes.shape, {}, default=0.04)
    assert np.allclose(n, [[0.15, 0.10, 0.10], [0.03, 0.035, 0.04]])
    assert all(0.1 <= domain.MANNING_NLCD[c] <= 0.2 for c in (41, 42, 43, 52, 90, 95)), "Arcement and Schneider's range"
