"""Class rasters to solver fields, and the synthetic parcel."""

import json

import numpy as np
import pytest

import parcel
from parcel import ClassParams, SYNTHETIC_CLASSES


def test_synthetic_parcel_has_a_disc_a_channel_a_pit_and_a_flat():
    dx, r = 2.0, 112.32
    z, classes = parcel.synthetic(dx, radius_m=r)
    valid = np.isfinite(z)
    assert valid.sum() == pytest.approx(np.pi * r * r / dx / dx, rel=0.01)
    assert z.shape == classes.shape and set(np.unique(classes)) == {1, 2, 3, 4}
    n = z.shape[0]
    centre, bank = int(0.6 * n), int(0.6 * n) + int(0.15 * n)
    rows = slice(int(0.4 * n), int(0.6 * n))
    assert (z[rows, bank] - z[rows, centre]).min() > 0.3
    pr, pc = int(0.4 * n), int(0.3 * n)
    assert classes[pr, pc] == 3
    assert z[pr, pc + int(0.1 * n)] - z[pr, pc] > 0.6 and z[pr, pc - int(0.1 * n)] - z[pr, pc] > 0.6
    assert z[pr + int(0.09 * n), pc] - z[pr, pc] > 0.3
    flat = z[int(0.75 * n):int(0.95 * n), int(0.05 * n):int(0.4 * n)]
    assert np.nanmax(flat) - np.nanmin(flat) == 0.0


def test_fields_map_codes_and_leave_unlabelled_at_the_default():
    codes = np.array([[0, 1], [2, 4]])
    default = ClassParams(manning_n=0.05, ksat_mm_hr=10.0, smax_mm=1.0, impervious_frac=0.5)
    f = parcel.fields(codes, SYNTHETIC_CLASSES, default)
    assert f["manning_n"][0, 0] == pytest.approx(0.05) and f["manning_n"][0, 1] == pytest.approx(0.014)
    assert f["infiltration"][0, 0] == pytest.approx(10.0 * 0.5 / 3.6e6)
    assert f["infiltration"][0, 1] == pytest.approx(0.05 * 0.01 / 3.6e6)
    assert f["smax_m"][1, 0] == pytest.approx(0.0035)


def test_build_surface_carries_constant_rate_infiltration():
    z, classes = parcel.synthetic(4.0)
    s = parcel.build_surface(z, classes, SYNTHETIC_CLASSES, SYNTHETIC_CLASSES[2], deficit_mm=50.0)
    assert np.array_equal(s.f0, s.fc) and s.k.max() == 0.0
    assert s.max_deficit_m.max() == pytest.approx(0.05) and s.smax_m.shape == z.shape
    assert s.manning_n[classes == 4].max() == pytest.approx(0.035)


def test_load_class_table_numbers_codes_in_list_order(tmp_path):
    doc = {"classes": [
        {"class_id": "asphalt", "params": {"manning_n": 0.014, "ksat_mm_hr": 0.05, "smax_mm": 1.25, "impervious_frac": 0.99}},
        {"class_id": "turf", "params": {"manning_n": 0.225, "ksat_mm_hr": 30.0, "smax_mm": 3.5, "impervious_frac": 0.0}},
    ]}
    (tmp_path / "t.json").write_text(json.dumps(doc))
    table, codes = parcel.load_class_table(tmp_path / "t.json")
    assert codes == {"asphalt": 1, "turf": 2}
    assert table[2].infiltration_m_s == pytest.approx(30.0 / 3.6e6)


def _write(path, arr, crs, nodata=None):
    import rasterio
    from affine import Affine

    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype=arr.dtype, crs=crs, transform=Affine(0.5, 0.0, 0.0, 0.0, -0.5, 0.0),
                       nodata=nodata) as dst:
        dst.write(arr, 1)


def test_read_rasters_rejects_a_geographic_dtm(tmp_path):
    z = np.ones((4, 5), dtype=np.float32)
    _write(tmp_path / "z.tif", z, "epsg:4326")
    _write(tmp_path / "c.tif", np.ones((4, 5), dtype=np.uint8), "epsg:4326")
    with pytest.raises(AssertionError, match="projected"):
        parcel.read_rasters(tmp_path / "z.tif", tmp_path / "c.tif")


def test_read_rasters_returns_nan_at_nodata_and_the_cell_size(tmp_path):
    z = np.arange(20, dtype=np.float32).reshape(4, 5)
    z[0, 0] = -9999.0
    _write(tmp_path / "z.tif", z, "epsg:32610", nodata=-9999.0)
    _write(tmp_path / "c.tif", np.full((4, 5), 2, dtype=np.uint8), "epsg:32610")
    got, codes, dx = parcel.read_rasters(tmp_path / "z.tif", tmp_path / "c.tif")
    assert np.isnan(got[0, 0]) and got[1, 1] == 6.0 and dx == 0.5 and codes.max() == 2


def test_fill_sinks_raises_closed_depressions_and_reports_the_volume():
    z = np.zeros((20, 20), dtype=np.float32)
    z += np.arange(20, dtype=np.float32)[:, None] * -0.01
    z[9:11, 9:11] -= 0.5
    filled, raised, cells = parcel.fill_sinks(z)
    assert cells == 4 and raised == pytest.approx(4 * 0.5, rel=0.05)
    assert (filled >= z - 1e-6).all() and filled[9, 9] > z[9, 9] + 0.4
    assert np.isfinite(filled).all()


def test_fill_sinks_leaves_nodata_and_a_sinkless_plane_alone():
    z = np.repeat(np.arange(10, dtype=np.float32)[:, None] * -0.1, 10, axis=1)
    z[0, 0] = np.nan
    filled, raised, cells = parcel.fill_sinks(z)
    assert raised == pytest.approx(0.0, abs=1e-9) and cells == 0
    assert np.isnan(filled[0, 0]) and np.allclose(filled[1:], z[1:])


def test_fill_sinks_treats_nodata_as_an_outlet_not_a_wall():
    """A hole in the raster drains; filling behind it would invent storage that is not there."""
    z = np.repeat(np.arange(12, dtype=np.float32)[:, None] * -0.1, 12, axis=1)
    z[5:7, 5:7] -= 0.4
    walled, _, _ = parcel.fill_sinks(z)
    holed = z.copy()
    holed[8, :] = np.nan
    filled, raised, cells = parcel.fill_sinks(holed)
    assert raised == pytest.approx(1.0, rel=0.02), "the real pit is still filled to its spill"
    assert cells == 4 and np.isnan(filled[8, :]).all()
    assert np.allclose(filled[9:], holed[9:]), "ground below the hole must not be dammed by it"
    assert filled[5, 5] == pytest.approx(walled[5, 5], rel=1e-6)


def test_the_buffer_window_is_the_square_a_bundle_of_that_reach_is_gridded_on():
    from affine import Affine

    z = np.arange(100 * 100, dtype=np.float32).reshape(100, 100)
    t = Affine(0.5, 0.0, 1000.0, 0.0, -0.5, 2050.0)
    w, tw = parcel.window(z, t, (1025.0, 2025.0), 10.1, 0.5)
    assert w.shape == (42, 42) and (tw.c, tw.f) == (1014.5, 2035.5)
    assert parcel.distance(w.shape, tw, (1025.0, 2025.0)).min() == pytest.approx(np.hypot(0.25, 0.25))


def test_the_soils_step_is_a_cells_crossing_by_sheet_flow_or_what_is_given():
    import cli
    assert cli.soil_dt(0.2, "cell") == pytest.approx(0.5) and cli.soil_dt(0.5, "cell") == pytest.approx(1.25)
    assert cli.soil_dt(0.2, "2") == 2.0 and cli.soil_dt(0.2, None) == 0.0
