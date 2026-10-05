"""The shared parameterisation: class rasters and tables to per-cell fields."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from affine import Affine

import params

BUNDLE = Path(os.environ.get("HYDRO_BUNDLE", "bundle"))


def _table(tmp_path):
    rows = [{"id": i, "class_id": name, "params": {
        "manning_n": 0.01 * (i + 1), "ksat_mm_hr": 10.0 * i, "smax_mm": 1.0 + i,
        "impervious_frac": 0.5 if i == 0 else 0.0, "poro": 0.3}}
        for i, name in enumerate(("asphalt", "dirt_soil", "turf"))]
    (tmp_path / "parameters.json").write_text(json.dumps({"rows": rows}))
    return params.load_table(tmp_path / "parameters.json")


def test_table_indexes_rows_by_id_and_finds_bare_soil(tmp_path):
    t = _table(tmp_path)
    assert t.names == ("asphalt", "dirt_soil", "turf") and t.bare_soil == 1
    assert t.values["manning_n"].tolist() == [0.01, 0.02, 0.03]
    assert t.infiltration_m_s[0] == pytest.approx(0.0) and t.infiltration_m_s[2] == pytest.approx(20.0 / 3.6e6)


def test_unobserved_and_nodata_take_bare_soil_and_are_counted(tmp_path):
    t = _table(tmp_path)
    codes = np.array([[0, 2], [params.UNOBSERVED, params.NODATA]], dtype=np.uint8)
    f, counts = params.fields(codes, t)
    assert counts == {"unobserved": 1, "nodata": 1}
    assert f["manning_n"].ravel().tolist() == pytest.approx([0.01, 0.03, 0.02, 0.02])
    assert f["smax_m"][0, 1] == pytest.approx(0.003) and f["poro"].max() == pytest.approx(0.3)


def test_nearest_resampling_picks_the_containing_pixel():
    codes = np.arange(6, dtype=np.uint8).reshape(2, 3)
    coarse = Affine(1.0, 0.0, 10.0, 0.0, -1.0, 20.0)
    fine = Affine(0.5, 0.0, 10.0, 0.0, -0.5, 20.0)
    got = params.resample_nearest(codes, coarse, (4, 6), fine)
    assert got[0, :2].tolist() == [0, 0] and got[3, 4:].tolist() == [5, 5] and got[1, 3] == 1
    outside = params.resample_nearest(codes, coarse, (2, 2), Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0))
    assert (outside == params.NODATA).all()


@pytest.mark.skipif(not (BUNDLE / "semantics" / "areas.json").exists(), reason="no bundle")
def test_grid_mean_manning_n_equals_the_class_area_sum():
    """Area-weighted Manning's n over the 0.2 m grid against areas.json times the table."""
    table = params.load_table(BUNDLE / "semantics" / "parameters.json")
    codes, transform = params.read_classes(BUNDLE / "semantics" / "class_ground_0p2m.tif")
    on_grid = params.resample_nearest(codes, transform, codes.shape, transform)
    assert np.array_equal(on_grid, codes)
    f, counts = params.fields(on_grid, table)
    inside = codes != params.NODATA
    grid_mean = float((f["manning_n"][inside].astype(np.float64) * 0.04).sum() / (inside.sum() * 0.04))
    areas = json.loads((BUNDLE / "semantics" / "areas.json").read_text())["by_resolution"]["0p2m"]
    n = dict(zip(table.names, table.values["manning_n"]))
    weighted = sum(a * n[c] for c, a in areas["ground_surface"].items())
    weighted += areas.get("unobserved", 0.0) * n[params.BARE_SOIL]
    total = sum(areas["ground_surface"].values()) + areas.get("unobserved", 0.0)
    assert inside.sum() * 0.04 == pytest.approx(total, rel=1e-6)
    assert grid_mean == pytest.approx(weighted / total, rel=1e-6)
    assert counts["unobserved"] == 0


def test_class_raster_name_follows_the_bundle_convention():
    assert params.class_raster_name(0.2) == "class_ground_0p2m.tif"
    assert params.class_raster_name(0.1) == "class_ground_0p1m.tif"
    assert params.class_raster_name(1.0) == "class_ground_1m.tif"


def _raster(path, bands, transform, names=None, dtype="float32", nodata=None):
    import rasterio
    from rasterio.crs import CRS
    with rasterio.open(path, "w", driver="GTiff", height=bands.shape[1], width=bands.shape[2], count=bands.shape[0],
                       dtype=dtype, crs=CRS.from_epsg(32610), transform=transform, nodata=nodata) as w:
        w.write(bands)
        for i, n in enumerate(names or []):
            w.set_band_description(i + 1, n)


def test_cells_override_the_class_table_cell_by_cell_and_nan_keeps_the_table(tmp_path):
    t = Affine(0.2, 0.0, 500000.0, 0.0, -0.2, 4000000.4)
    sem = tmp_path / "semantics"
    sem.mkdir()
    _table(sem)
    _raster(sem / "class_ground_0p2m.tif", np.array([[[2, 2], [0, 2]]], np.uint8), t, dtype="uint8", nodata=params.NODATA)
    nan = np.nan
    cells = np.array([[[5.0, nan], [0.0, 12.0]],        # infiltration mm/h
                      [[40.0, nan], [nan, 0.0]],       # deficit mm: a saturated cell at 0
                      [[0.3, nan], [nan, nan]],        # manning n
                      [[nan, nan], [7.0, nan]]], np.float32)
    _raster(tmp_path / "cells.tif", cells, t, names=params.CELL_BANDS)
    z = np.zeros((2, 2), np.float32)
    surf, rec = params.build_surface(z, t, tmp_path, 0.2, cells=tmp_path / "cells.tif")
    assert surf.f0[0, 0] == pytest.approx(5.0 / 3.6e6) and surf.f0[0, 1] == pytest.approx(20.0 / 3.6e6)
    assert surf.f0[1, 0] == 0.0 and surf.max_deficit_m[0, 0] == pytest.approx(0.04) and surf.max_deficit_m[1, 1] == 0.0
    assert np.isinf(surf.max_deficit_m[0, 1]) and surf.manning_n[0, 0] == pytest.approx(0.3)
    assert surf.manning_n[1, 1] == pytest.approx(0.03) and surf.smax_m[1, 0] == pytest.approx(0.007)
    assert rec["cells_given"]["infiltration_mm_hr"] == 3
    plain, _ = params.build_surface(z, t, tmp_path, 0.2)
    assert plain.max_deficit_m is None and plain.f0[0, 0] == pytest.approx(20.0 / 3.6e6)


def test_cells_with_the_gar_bands_run_green_ampt_from_the_balance_state(tmp_path):
    """K_s from `infiltration_mm_hr`, F_max from `deficit_mm`, theta_i the balance's own; a cell missing a band is
    sealed; without the GAR bands the run stays Horton."""
    t = Affine(0.2, 0.0, 500000.0, 0.0, -0.2, 4000000.4)
    sem = tmp_path / "semantics"
    sem.mkdir()
    _table(sem)
    _raster(sem / "class_ground_0p2m.tif", np.array([[[2, 2], [0, 2]]], np.uint8), t, dtype="uint8", nodata=params.NODATA)
    nan = np.nan
    base = [[[10.0, 10.0], [0.0, 10.0]], [[30.0, nan], [0.0, 30.0]], [[0.05] * 2] * 2, [[2.0] * 2] * 2]
    gar = [[[89.0, 89.0], [89.0, nan]], [[0.46] * 2] * 2, [[0.03] * 2] * 2, [[0.22] * 2] * 2, [[0.25] * 2] * 2]
    _raster(tmp_path / "cells.tif", np.array(base + gar, np.float32), t, names=params.CELL_BANDS + params.GAR_BANDS)
    surf, rec = params.build_surface(np.zeros((2, 2), np.float32), t, tmp_path, 0.2, cells=tmp_path / "cells.tif")
    s = surf.soil
    assert rec["infiltration"] == "gar" and s is not None
    assert s.ks[0, 0] == pytest.approx(10.0 / 3.6e6) and s.ks[1, 1] == 0.0 and s.ks[1, 0] == 0.0
    assert s.f_max[0, 0] == pytest.approx(0.03) and np.isinf(s.f_max[0, 1]) and s.psi_f[0, 0] == pytest.approx(0.089)
    assert s.theta_i[0, 0] == pytest.approx(0.25) and s.lam[0, 0] == pytest.approx(0.22)
    _raster(tmp_path / "plain.tif", np.array(base, np.float32), t, names=params.CELL_BANDS)
    plain, rec2 = params.build_surface(np.zeros((2, 2), np.float32), t, tmp_path, 0.2, cells=tmp_path / "plain.tif")
    assert plain.soil is None and rec2["infiltration"] == "horton"
