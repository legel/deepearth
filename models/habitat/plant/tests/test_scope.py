"""The mapped region (ranges/joint/scope.py): which training points leave, and which calibration areas are extended.

* points inside the excluded level-3 areas (with their buffer, across 180 degrees) or inside an excluded box leave,
  every other point stays;
* on the real WGSRPD polygons (when present): Alaska with the Aleutians on both sides of 180 degrees and Hawaii leave,
  the contiguous US, Canada and Siberia stay;
* restriction of a small national data directory keeps the region's species, renumbers them and subsets the plots;
* a species whose calibration ecoregions all miss the grid gains the ecoregions of its native areas in the region.
"""
import json

import numpy as np
import pandas as pd
import pytest
import shapely

from ranges.config import data_root
from ranges.joint import scope

GEO = data_root() / "raw/geo/wgsrpd_level3.geojson"
HAWAII = [{"lat": [18.5, 22.6], "lon": [-160.6, -154.6]}]


def _area(tmp_path):
    """Two level-3 areas: AAA spans 180 degrees (170 E to 170 W), BBB lies at 10-20 E."""
    import geopandas as gpd
    g = gpd.GeoDataFrame({"LEVEL3_COD": ["AAA", "AAA", "BBB"]},
                         geometry=[shapely.box(170, 50, 180, 55), shapely.box(-180, 50, -170, 55), shapely.box(10, 0, 20, 5)],
                         crs=4326)
    f = tmp_path / "l3.geojson"
    g.to_file(f, driver="GeoJSON")
    return f


def test_points_in_excluded_areas(tmp_path):
    area = scope.excluded_area(_area(tmp_path), ["AAA"], 0.05)
    lat = np.array([52, 52, 52, 52, 49.9, 55.04, 52, 2, 20, 21])
    lon = np.array([175, -175, 179.99, -169.97, 175, -175, -160, 15, -158, -150])
    out = scope.in_excluded(lat, lon, area, HAWAII)
    assert out.tolist() == [True, True, True, True, False, True, False, False, True, False]


@pytest.mark.skipif(not GEO.exists(), reason="needs the WGSRPD level-3 polygons (fetch_sources.sh geo)")
def test_alaska_and_hawaii_leave_the_contiguous_us_stays():
    area = scope.excluded_area(GEO, ["ASK", "ALU"], 0.05)
    pts = {"Anchorage": (61.2, -149.9, True), "Juneau": (58.3, -134.4, True), "Attu, 172.9 E": (52.9, 172.9, True),
           "Adak": (51.88, -176.65, True), "Honolulu": (21.3, -157.86, True), "Seattle": (47.6, -122.3, False),
           "Prince Rupert, Canada": (54.3, -130.3, False), "Kamchatka": (53.0, 158.6, False),
           "Miami": (25.8, -80.2, False)}
    lat, lon, want = (np.array(v) for v in zip(*pts.values()))
    assert scope.in_excluded(lat, lon, area, HAWAII).tolist() == want.tolist()


def test_restrict_keeps_the_region(tmp_path):
    src, dst = tmp_path / "national", tmp_path / "conus"
    src.mkdir()
    (src / "tree").mkdir()
    pd.DataFrame({"species": ["A_a", "B_b", "C_c"], "us_regions": ["ASK,CONUS", "HAW", "CONUS"],
                  "calibration_ecoregions": ["1", "2", "3"]}).to_csv(src / "species.csv", index=False)
    lat = np.array([40, 61, 40, 21, 45, 50], np.float32)            # A: CONUS, Alaska, CONUS; B: Hawaii; C: CONUS, Canada
    lon = np.array([-100, -150, -90, -157, -95, -100], np.float32)
    sid = np.array([0, 0, 0, 1, 2, 2], np.int32)
    for k, v in (("lat", lat), ("lon", lon), ("sid", sid), ("pres", np.array([1, 1, 0, 1, 1, 0], np.int8)),
                 ("train_points_cache", np.arange(12, dtype=np.float32).reshape(6, 2))):
        np.save(src / f"{k}.npy", v)
    np.savez(src / "joint_data.npz", names=np.array(["x", "y"]), eval_sid=np.array([0, 1, 2], np.int32),
             eval_y=np.eye(3, dtype=np.int8))
    np.savez(src / "plots_hawaii.npz", eval_sid=np.array([1], np.int32))
    l3 = tmp_path / "l3.geojson"
    import geopandas as gpd
    gpd.GeoDataFrame({"LEVEL3_COD": ["ASK"]}, geometry=[shapely.box(-170, 55, -130, 72)], crs=4326).to_file(l3, driver="GeoJSON")
    meta = scope.restrict(src, dst, "CONUS", l3, ["ASK"], 0.05, HAWAII, ["hawaii"], log=lambda m: None)
    assert pd.read_csv(dst / "species.csv").species.tolist() == ["A_a", "C_c"]
    assert np.load(dst / "sid.npy").tolist() == [0, 0, 1, 1]
    assert np.load(dst / "lat.npy").tolist() == [40, 40, 45, 50]          # Canada stays: native range abroad
    assert np.load(dst / "train_points_cache.npy")[:, 0].tolist() == [0, 4, 8, 10]
    z = np.load(dst / "joint_data.npz")
    assert z["eval_sid"].tolist() == [0, 1] and z["eval_y"].shape == (2, 3)
    assert not (dst / "plots_hawaii.npz").exists() and meta["species"] == 2 and meta["points"] == 4
    assert json.loads((dst / "meta.json").read_text())["scope"] == "CONUS"


def test_calibration_extended_only_off_the_grid(tmp_path):
    inv = pd.DataFrame({"wcvp_accepted_name": ["A a", "B b", "C c"], "native_l3": ["ONT,MIN", "MIN", "XXX"]})
    out = scope.extend_calibration(["A_a", "B_b", "C_c"], ["900", "5 900", "901"], inv, on_grid={5, 6, 7},
                                   l3eco={"MIN": [6, 900], "ONT": [7]}, region_l3=["MIN"], l3_geojson=None,
                                   ecoregions_shp=None, log=lambda m: None)
    # A: off the grid -> its region area MIN adds 6 (900 is off the grid; ONT is outside the region);
    # B: already on the grid -> unchanged; C: no native area in the region -> unchanged
    assert out == ["6 900", "5 900", "901"]
