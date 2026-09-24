"""The registry is the only source of coordinates, so its invariants are worth asserting."""

import pytest

from domain import Grid
from sites import DAYS, SITES, get_day, get_site


def test_unknown_names_fail_with_the_valid_options():
    with pytest.raises(AssertionError, match="valid sites"):
        get_site("nope")
    with pytest.raises(AssertionError, match="valid days"):
        get_day("nope")


def test_names_match_their_registry_keys():
    assert all(k == v.name for k, v in SITES.items())
    assert all(k == v.name for k, v in DAYS.items())


def test_campanile_station_is_oakland_at_17_km():
    s = get_site("campanile")
    assert s.asos_station == "OAK"
    assert s.station_km() == pytest.approx(17.3, abs=0.1)


@pytest.mark.parametrize("dx", get_site("campanile").cell_sizes_m)
def test_grid_covers_the_disc_and_clears_the_tower(dx):
    s = get_site("campanile")
    n, nz = s.cells_across(dx), s.levels(dx)
    assert n * dx >= 2 * s.radius_m
    assert n % 128 == 0 and nz % 8 == 0
    g = Grid.stretched(dx, n, n, nz, dx, s.stretch)
    assert g.top >= 2 * s.tallest_m
    assert g.dz[0] == pytest.approx(dx)


def test_production_grid_sizes_are_what_the_readme_quotes():
    s = get_site("campanile")
    assert (s.cells_across(0.2), s.levels(0.2)) == (1152, 72)
    assert (s.cells_across(0.1), s.levels(0.1)) == (2304, 88)


def test_paths_live_under_the_site_root():
    s = get_site("campanile")
    for path in (s.asos(2025), s.bundle):
        assert s.root in path.parents, f"{path} escapes {s.root}"
    assert s.out_path("x.bin").parent == s.out


@pytest.mark.parametrize("name", sorted(SITES))
def test_registry_matches_the_bundle_aoi_exactly(name):
    """Five copies of the anchor hold the three products in register; these are copied
    constants, so the comparison is exact, not a tolerance."""
    import json

    import frames

    s = get_site(name)
    path = s.bundle / "surface" / f"aoi_{name}.json"
    if not path.exists():
        pytest.skip(f"{path} not present")
    aoi = json.loads(path.read_text())
    assert tuple(aoi["scene_frame"]["anchor_utm"]) == s.anchor_utm
    assert aoi["aoi"]["centre_utm10n"]["epsg"] == frames.EPSG
    assert (aoi["aoi"]["centre_wgs84"]["lat"], aoi["aoi"]["centre_wgs84"]["lon"]) == (s.lat, s.lon)
    assert aoi["aoi"]["radius_m"] == s.radius_m


def test_anchor_is_the_parcel_centre_in_utm_10n():
    """UTM 10N central meridian is -123; the anchor easting sits 0.74 deg east of it."""
    s = get_site("campanile")
    east, north, h = s.anchor_utm
    assert 565_000 < east < 565_500 and 4_191_500 < north < 4_192_500 and 100 < h < 140


def test_days_carry_their_year():
    assert all(d.year == int(d.date[:4]) for d in DAYS.values())
