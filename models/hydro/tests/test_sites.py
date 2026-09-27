"""The registry is the only source of coordinates, so its invariants are worth asserting."""

import json
import math
import os
from pathlib import Path

import pytest

from sites import KM_PER_DEG_LAT, SITES, STORMS, get_site, get_storm


def test_unknown_names_fail_with_the_valid_options():
    with pytest.raises(AssertionError, match="valid sites"):
        get_site("nope")
    with pytest.raises(AssertionError, match="valid storms"):
        get_storm("nope")


@pytest.mark.parametrize("name", sorted(SITES))
def test_bbox_is_centred_and_square_in_kilometres(name):
    s = get_site(name)
    west, south, east, north = s.bbox()
    assert (west + east) / 2 == pytest.approx(s.lon)
    assert (south + north) / 2 == pytest.approx(s.lat)
    height_km = (north - south) * KM_PER_DEG_LAT
    width_km = (east - west) * KM_PER_DEG_LAT * math.cos(math.radians(s.lat))
    assert height_km == pytest.approx(2 * s.radius_km, rel=1e-9)
    assert width_km == pytest.approx(2 * s.radius_km, rel=1e-9)


def test_km_per_deg_is_the_pinned_legacy_value():
    """Changing this shifts every fetched raster's grid; it is not a geodesy improvement."""
    assert KM_PER_DEG_LAT == 111.0


@pytest.mark.parametrize("name", sorted(SITES))
def test_paths_live_under_the_site_root(name):
    s = get_site(name)
    for path in (s.dem, s.dem_conditioned, s.mukey_map, s.soil_storage, s.roads,
                 s.nlcd_impervious, s.atlas14, s.asos("ian"), s.discharge("ian")):
        assert s.root in path.parents, f"{path} escapes {s.root}"
    assert s.out_path("x.tif").parent == s.out


def test_site_names_match_their_registry_keys():
    assert all(k == v.name for k, v in SITES.items())
    assert all(k == v.name for k, v in STORMS.items())


def test_gauge_capture_fraction_is_reported_not_assumed():
    """site3 validates against less than half the gauge's documented area; the caveat travels.

    The value moves with the DELIVERED DEM, not only with the terrain: 3.72 km2 on the 0.88 m
    DEM this site was first built on, 15.27 km2 on the 3 m DEM 3DEP returns for a 1 m request
    today. Both are `cli.py terrain` on a fresh fetch. What must not come back is the
    superseded 11.65 km2 figure, which predates the stream-burn and threshold fixes.
    """
    g = get_site("site3").gauge
    assert g is not None
    assert g.delineated_area_km2 == pytest.approx(15.27)
    assert g.capture_fraction == pytest.approx(15.27 / 33.15, rel=1e-9)
    assert g.capture_fraction < 0.5


def test_delineated_area_matches_the_recorded_terrain_run():
    """The registry value must be what the shipped terrain stage actually produces."""
    import json
    from pathlib import Path

    recorded = json.loads((Path(__file__).resolve().parents[1]
                           / "docs" / "terrain_site3.json").read_text())
    assert recorded["pour_point"] == "gauge"
    assert recorded["catchment_km2"] == pytest.approx(
        get_site("site3").gauge.delineated_area_km2, abs=0.005)


def test_main_aoi_declares_no_gauge():
    """The nearest gauge drains 44x this domain; pretending otherwise is the error to avoid."""
    assert get_site("main_aoi").gauge is None


def test_storm_windows_are_ordered():
    for storm in STORMS.values():
        assert storm.start < storm.end
        assert storm.gauge_start <= storm.start[:10]
        assert storm.gauge_end >= storm.end[:10]


BUNDLE = Path(os.environ.get("HYDRO_BUNDLE", "/nonexistent"))  # a bundle directory, when one is at hand


@pytest.mark.parametrize("name", [n for n in SITES if (BUNDLE / "surface" / f"aoi_{n}.json").exists()]
                         or [pytest.param("none", marks=pytest.mark.skip(reason="no bundle"))])
def test_registry_matches_the_bundle_aoi_exactly(name):
    """The anchor is a copied constant held by every solver; the copies must agree to the digit."""
    doc = json.loads((BUNDLE / "surface" / f"aoi_{name}.json").read_text())
    site = get_site(name)
    assert list(site.anchor_m) == doc["scene_frame"]["anchor_utm"]
    assert site.epsg == doc["aoi"]["centre_utm10n"]["epsg"]
    assert site.lat == doc["aoi"]["centre_wgs84"]["lat"] and site.lon == doc["aoi"]["centre_wgs84"]["lon"]
    assert site.anchor_m[0] == doc["aoi"]["centre_utm10n"]["easting"]
    assert site.anchor_m[1] == doc["aoi"]["centre_utm10n"]["northing"]
