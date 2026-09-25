"""Rainfall forcing: depth conservation, and the provenance guard on Atlas 14."""

import json

import numpy as np
import pytest

import fetch
from forcing import RETURN_PERIODS_YR, atlas14_depth_mm, design_hyetograph


@pytest.mark.parametrize("depth_mm,duration_hr,dt_s", [
    (96.3, 24.0, 1200.0), (266.7, 24.0, 300.0), (98.0, 1.0, 60.0),
])
def test_design_storm_conserves_depth(depth_mm, duration_hr, dt_s):
    rate = design_hyetograph(depth_mm, duration_hr, dt_s)
    assert float(rate.sum() * dt_s * 1000.0) == pytest.approx(depth_mm, rel=1e-9)
    assert np.all(rate >= 0.0)


def test_design_storm_peaks_in_the_scs_type_ii_window():
    """SCS Type II front-loads the peak at roughly 40-60 % of the duration."""
    rate = design_hyetograph(266.7, 24.0, 60.0)
    assert 0.35 <= float(np.argmax(rate)) / len(rate) <= 0.65


def test_design_storm_rejects_nonsense():
    with pytest.raises(AssertionError):
        design_hyetograph(0.0, 24.0, 60.0)
    with pytest.raises(AssertionError):
        design_hyetograph(100.0, 0.0, 60.0)


def _write_table(site, source, depths):
    site.atlas14.write_text(json.dumps({"source": source, "depths_mm": depths}))


def test_atlas14_refuses_a_fallback_table(tmp_path, monkeypatch):
    """A fallback table must never be modelled on -- it served county defaults for months."""
    import sites

    site = sites.get_site("site3")
    monkeypatch.setattr(type(site), "atlas14", property(lambda s: tmp_path / "atlas14.json"))
    _write_table(site, "fallback", {"24hr": {"100": 266.7}})
    with pytest.raises(AssertionError, match="not fetched from NOAA PFDS"):
        atlas14_depth_mm(site, 100, 24.0)

    _write_table(site, "pfds", {"24hr": {"100": 266.7}})
    assert atlas14_depth_mm(site, 100, 24.0) == pytest.approx(266.7)


def test_atlas14_reports_missing_durations_and_periods(tmp_path, monkeypatch):
    import sites

    site = sites.get_site("site3")
    monkeypatch.setattr(type(site), "atlas14", property(lambda s: tmp_path / "atlas14.json"))
    _write_table(site, "pfds", {"24hr": {"100": 266.7}})
    with pytest.raises(AssertionError, match="duration"):
        atlas14_depth_mm(site, 100, 12.0)
    with pytest.raises(AssertionError, match="return period"):
        atlas14_depth_mm(site, 5, 24.0)


def test_pfds_grid_shape_is_not_the_ensemble_shape():
    """PFDS returns 10 return periods; the ensemble runs 9. Conflating them writes a fallback.

    This is not hypothetical -- checking the response against the ensemble list rejects a
    perfectly good PFDS grid and silently degrades every design storm downstream.
    """
    assert len(fetch.PFDS_RETURN_PERIODS_YR) == 10
    assert len(RETURN_PERIODS_YR) == 9
    assert 1000 in fetch.PFDS_RETURN_PERIODS_YR and 1000 not in RETURN_PERIODS_YR
    assert list(RETURN_PERIODS_YR) == fetch.PFDS_RETURN_PERIODS_YR[:9]


def test_pfds_duration_rows_match_the_documented_service_shape():
    assert len(fetch.PFDS_DURATIONS_HR) == 19
    assert 24 in fetch.PFDS_DURATIONS_HR and 1 in fetch.PFDS_DURATIONS_HR


def test_tower_rain_is_the_forcing_read_hour_by_hour_over_the_storm(tmp_path):
    """The flux tower's hourly rain under the P rule (models/flux) drives a storm: its hours inside the window, each
    hour's depth conserved on the solver's steps."""
    from forcing import tower_hyetograph
    from sites import Storm
    csv = tmp_path / "rain.csv"
    csv.write_text("time_utc,p\n2024-07-09T21:00:00Z,0.0\n2024-07-09T22:00:00Z,12.0\n"
                   "2024-07-09T23:00:00Z,39.9\n2024-07-10T00:00:00Z,3.0\n2024-07-10T01:00:00Z,0.0\n")
    storm = Storm("t", "t", "2024-07-09 22:00", "2024-07-10 00:00", "2024-07-09", "2024-07-10")
    rate, hourly = tower_hyetograph(csv, storm, 600.0)
    assert hourly.tolist() == [12.0, 39.9, 3.0] and np.all(rate >= 0.0)
    assert float(rate[0]) == pytest.approx(12.0 / 1000 / 3600)
    assert float(rate.sum() * 600.0 * 1000.0) == pytest.approx(54.9), "every hour's depth, exactly"
