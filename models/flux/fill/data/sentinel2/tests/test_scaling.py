"""Band scaling: BOA_ADD_OFFSET per processing baseline, reflectance, SCL fractions, scene selection."""

import numpy as np

from bands import GROUP, boa_add_offset, offset_from_raster_bands, reflectance, scl_fractions
from stac import Scene, select


def test_offset_by_baseline():
    assert boa_add_offset("02.12") == 0
    assert boa_add_offset("03.01") == 0
    assert boa_add_offset("04.00") == -1000
    assert boa_add_offset("05.00") == -1000
    assert boa_add_offset("05.10") == -1000
    assert boa_add_offset("") == 0


def test_offset_from_stac_raster_bands_matches_collection1():
    assert offset_from_raster_bands({"scale": 0.0001, "offset": -0.1}) == -1000
    assert offset_from_raster_bands({"scale": 0.0001, "offset": 0}) == 0
    assert offset_from_raster_bands(None) is None


def test_reflectance_and_nodata():
    dn = np.array([0, 1000, 2000, 11000], dtype=np.uint16)
    r_new = reflectance(dn, -1000)
    assert np.isnan(r_new[0])
    np.testing.assert_allclose(r_new[1:], [0.0, 0.1, 1.0])
    r_old = reflectance(dn, 0)
    np.testing.assert_allclose(r_old[1:], [0.1, 0.2, 1.1])


def test_same_surface_same_reflectance_across_baselines():
    # A pixel of reflectance 0.25 is DN 2500 before 04.00 and DN 3500 from 04.00 on.
    assert reflectance(np.array([2500], np.uint16), boa_add_offset("02.07"))[0] == np.float32(0.25)
    assert reflectance(np.array([3500], np.uint16), boa_add_offset("05.00"))[0] == np.float32(0.25)


def test_group_sizes():
    assert GROUP["r10"].size == 30 and GROUP["r20"].size == 15 and GROUP["r60"].size == 5
    assert len(GROUP["r10"].bands) + len(GROUP["r20"].bands) + len(GROUP["r60"].bands) == 12


def test_scl_fractions():
    scl = np.full((15, 15), 4, np.uint8)
    scl[0, :] = 9
    scl[1, :] = 3
    scl[2, :5] = 0
    f = scl_fractions(scl)
    assert np.isclose(f["cloud_frac"], 15 / 225) and np.isclose(f["shadow_frac"], 15 / 225)
    assert np.isclose(f["invalid_frac"], 5 / 225)
    assert np.isclose(f["clear_frac"], 1 - 35 / 225)


def _scene(source, baseline, gen, key="T1|GS2A_20200101T000000_000001"):
    return Scene(source=source, item_id=f"{source}-{baseline}", datetime_utc="2020-01-01T00:00:00Z",
                 platform="sentinel-2a", mgrs_tile="T1", epsg=32610, relative_orbit=1, datatake_id="",
                 acq_key=key, processing_baseline=baseline, boa_add_offset=boa_add_offset(baseline),
                 generation_time=gen, sun_azimuth=0, sun_elevation=0, view_azimuth=0, view_incidence=0,
                 scene_cloud_cover=0)


def test_select_prefers_collection1_then_earth_search():
    a = _scene("planetary-computer", "04.00", "2022")
    b = _scene("planetary-computer", "05.10", "2024")
    c = _scene("earth-search-c1", "05.00", "2023")
    assert select([a, b, c])[0].item_id == "earth-search-c1-05.00"
    assert select([a, b])[0].item_id == "planetary-computer-05.10"
    assert select([a])[0].item_id == "planetary-computer-04.00"
    d = _scene("planetary-computer", "02.12", "2021", key="T2|other")
    assert len(select([a, b, c, d])) == 2
