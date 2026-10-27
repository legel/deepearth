"""Coordinate selection, unit conversion and time alignment of the driver extraction and validation."""

import numpy as np
import pandas as pd
import pytest

import physics
from dem import tile_name
from grid import RegularGrid, combine, corner_weights
from hrrr import lcc_corners
from openmeteo import SOURCES
from validate import metrics, tower_hourly

E5L = SOURCES["era5_land"].grid
E5 = SOURCES["era5"].grid


# Coordinate selection ------------------------------------------------------------------------------------------

def test_nearest_cell_centres():
    # ERA5-Land cell centres are multiples of 0.1 deg; index 0 is lat -90, lon -180.
    i, j = E5L.nearest(44.06, -71.29)
    assert E5L.centre(i, j) == pytest.approx((44.1, -71.3))
    i, j = E5.nearest(44.06, -71.29)
    assert E5.centre(i, j) == pytest.approx((44.0, -71.25))
    assert E5L.nearest(-90, -180) == (0, 0)


def test_corners_and_weights_reproduce_a_linear_field():
    lat, lon = 44.0646, -71.2881
    i0, j0, j1, wy, wx = E5L.corners(lat, lon)
    lats = np.array([E5L.centre(i0, j0)[0], E5L.centre(i0, j1)[0], E5L.centre(i0 + 1, j0)[0], E5L.centre(i0 + 1, j1)[0]])
    lons = np.array([E5L.centre(i0, j0)[1], E5L.centre(i0, j1)[1], E5L.centre(i0 + 1, j0)[1], E5L.centre(i0 + 1, j1)[1]])
    assert lats.min() <= lat <= lats.max() and lons.min() <= lon <= lons.max()
    field = 3.0 * lats - 2.0 * lons + 7.0  # bilinear interpolation is exact for a linear field
    block = np.repeat(field[:, None], 5, axis=1)
    nn, bl, k = combine(block, wy, wx)
    assert bl == pytest.approx(np.full(5, 3.0 * lat - 2.0 * lon + 7.0), abs=1e-4)
    assert corner_weights(wy, wx).sum() == pytest.approx(1.0)
    # The nearest corner is the one whose centre is closest to the tower.
    d = np.hypot(lats - lat, (lons - lon) * np.cos(np.radians(lat)))
    assert k == int(np.argmin(d))


def test_sea_corner_dropped_and_weights_renormalised():
    block = np.array([[1.0, 1.0], [np.nan, np.nan], [3.0, 3.0], [5.0, 5.0]])
    nn, bl, k = combine(block, wy=0.5, wx=0.5)
    assert bl == pytest.approx([3.0, 3.0])           # (1 + 3 + 5) / 3 at equal weights
    nn, bl, k = combine(block, wy=0.1, wx=0.9)       # nearest corner (0, 1) is sea: next nearest with a value
    assert k in (0, 3) and np.isfinite(nn).all()
    nn, bl, k = combine(np.full((4, 3), np.nan), 0.5, 0.5)
    assert k == -1 and np.isnan(bl).all()


def test_dateline_wrap():
    i0, j0, j1, wy, wx = E5L.corners(10.0, 179.97)
    assert j0 == 3599 and j1 == 0
    assert E5L.centre(0, 3599)[1] == pytest.approx(179.9)


def test_hrrr_corners_inside_and_outside():
    xs = np.arange(5) * 3000.0
    ys = 12000.0 - np.arange(5) * 3000.0  # descending, as in the HRRR store
    iy0, ix0, wy, wx = lcc_corners(4500.0, 10500.0, xs, ys)
    assert (iy0, ix0) == (0, 1) and wy == pytest.approx(0.5) and wx == pytest.approx(0.5)
    assert lcc_corners(-10.0, 6000.0, xs, ys) is None
    assert lcc_corners(6000.0, 13000.0, xs, ys) is None


def test_dem_tile_names():
    assert tile_name(44.06, -71.29) == "Copernicus_DSM_COG_10_N44_00_W072_00_DEM"
    assert tile_name(-37.76, 145.0) == "Copernicus_DSM_COG_10_S38_00_E145_00_DEM"
    assert tile_name(0.5, -0.5) == "Copernicus_DSM_COG_10_N00_00_W001_00_DEM"


# Unit conversion -----------------------------------------------------------------------------------------------

def test_saturation_vapour_pressure_and_vpd():
    assert physics.es_hpa(0.0) == pytest.approx(6.1094)
    assert physics.es_hpa(20.0) == pytest.approx(23.39, abs=0.07)    # WMO tables 23.39 hPa; Magnus form 23.33
    assert physics.vpd_hpa(20.0, 20.0) == pytest.approx(0.0)
    assert physics.vpd_hpa(25.0, 10.0) == pytest.approx(31.67 - 12.27, abs=0.1)
    assert physics.vpd_hpa(10.0, 10.5) == 0.0                        # dew point above air temperature
    assert physics.rh_pct(20.0, 20.0) == pytest.approx(100.0)


def test_surface_pressure_from_msl():
    # Standard atmosphere: 101.325 kPa at sea level, 89.87 kPa at 1000 m, 79.50 kPa at 2000 m.
    assert physics.surface_pressure_kpa(1013.25, 15.0 - 6.5, 1000.0) == pytest.approx(89.87, abs=0.02)
    assert physics.surface_pressure_kpa(1013.25, 15.0 - 13.0, 2000.0) == pytest.approx(79.50, abs=0.03)
    assert physics.surface_pressure_kpa(1013.25, 15.0, 0.0) == pytest.approx(101.325)
    # Moving a pressure up and back down returns it.
    p = physics.pressure_to_height_kpa(95.0, 10.0, 500.0, 800.0)
    assert physics.pressure_to_height_kpa(p, 10.0 - 0.0065 * 300, 800.0, 500.0) == pytest.approx(95.0, abs=1e-3)


def test_other_conversions():
    assert physics.wind_speed(3.0, -4.0) == pytest.approx(5.0)
    assert physics.kelvin_to_c(273.15) == pytest.approx(0.0)
    assert physics.fraction_to_pct(0.31) == pytest.approx(31.0)
    assert physics.lapse_adjust_c(10.0, z_cell=500.0, z_site=300.0) == pytest.approx(11.3)


# Time alignment ------------------------------------------------------------------------------------------------

def _tower(values, var="TA", t0="2019-06-01T00:00"):
    t0 = int(pd.Timestamp(t0).value // 60_000_000_000)
    t = t0 + 30 * np.arange(len(values))
    return pd.DataFrame({"t_utc": t, var: np.asarray(values, float), var + "_flag": np.zeros(len(values), np.uint8)})


def test_instant_and_preceding_hour_matching():
    # Half-hours starting 00:00, 00:30, 01:00, 01:30, 02:00 with values 0, 1, 2, 3, 4.
    th = tower_hourly(_tower([0, 1, 2, 3, 4]), "TA")
    t1 = pd.Timestamp("2019-06-01T01:00")
    assert th.loc[t1, "inst"] == pytest.approx(1.5)   # half-hours 00:30 and 01:00: the hour centred on 01:00
    assert th.loc[t1, "prev"] == pytest.approx(0.5)   # half-hours 00:00 and 00:30: the hour ending at 01:00


def test_precipitation_sums_and_missing_half_hour():
    th = tower_hourly(_tower([0.2, 0.4, 0.0, 1.0], "P"), "P")
    assert th.loc[pd.Timestamp("2019-06-01T01:00"), "prev"] == pytest.approx(0.6)
    tw = _tower([1.0, 2.0, 3.0, 4.0])
    tw.loc[1, "TA_flag"] = 4                           # a removed half-hour invalidates both hours it touches
    th = tower_hourly(tw, "TA")
    assert np.isnan(th.loc[pd.Timestamp("2019-06-01T01:00"), "inst"])
    assert np.isnan(th.loc[pd.Timestamp("2019-06-01T01:00"), "prev"])


def test_local_standard_time_offset_is_not_applied_to_utc_matching():
    # A tower at UTC-8 stores t_utc = t_lst + 480 min; matching uses t_utc only, so a driver at 20:00 UTC meets the
    # tower half-hours that started 19:30 and 20:00 UTC (11:30 and 12:00 local standard time).
    tw = _tower(np.arange(48.0), t0="2019-06-01T00:00")
    th = tower_hourly(tw, "TA")
    assert th.loc[pd.Timestamp("2019-06-01T20:00"), "inst"] == pytest.approx((39 + 40) / 2)


def test_metrics():
    x = np.array([1.0, 2.0, 3.0, 4.0] * 5)
    m = metrics(x + 1.0, x)
    assert m["bias"] == pytest.approx(1.0) and m["rmse"] == pytest.approx(1.0) and m["r"] == pytest.approx(1.0)


def test_open_meteo_pressure_is_pascal():
    # The Open-Meteo files hold pressure_msl in Pa (median 101,538 at the validation towers); assemble divides by
    # 100 before the hPa-based height move. A missed factor shows as PA near 10,000 kPa.
    assert SOURCES["era5"].variables["pressure_msl"] == "Pa"
    assert physics.surface_pressure_kpa(101325.0 / 100.0, 15.0, 0.0) == pytest.approx(101.325)
