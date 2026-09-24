"""The log profile, its transfer between fetches, and the ASOS record's reduction."""

import numpy as np
import pandas as pd
import pytest

import forcing
import sites
from forcing import LogProfile, day_series, inflow, rose, sector_of, transfer, wind_vector
from physics import KAPPA


def test_profile_passes_through_its_reference():
    p = LogProfile.from_reference(5.0, 10.0, 0.03)
    assert p.speed(np.array([10.0]))[0] == pytest.approx(5.0)
    assert p.u_star == pytest.approx(KAPPA * 5.0 / np.log(10.0 / 0.03))
    assert p.speed(np.array([0.03, 0.01]))[0] == 0.0 and p.speed(np.array([0.01]))[0] == 0.0


def test_profile_rejects_a_reference_inside_the_roughness_sublayer():
    with pytest.raises(AssertionError, match="roughness sublayer"):
        LogProfile.from_reference(5.0, 4.0, 0.03, d=4.0)


def test_transfer_matches_speed_at_the_blending_height():
    station = LogProfile.from_reference(6.0, 10.0, 0.03)
    parcel = transfer(station, 0.8, 5.0, 60.0)
    assert parcel.speed(np.array([60.0]))[0] == pytest.approx(station.speed(np.array([60.0]))[0])
    assert parcel.speed(np.array([10.0]))[0] < station.speed(np.array([10.0]))[0]


def test_inflow_uses_the_site_fetch():
    site = sites.get_site("campanile")
    p = inflow(site, 6.0)
    assert (p.z0, p.d) == (site.z0_m, site.d_m)


def test_wind_vector_convention():
    """Direction is where the wind blows FROM: a westerly moves air eastward."""
    assert wind_vector(5.0, 270.0) == pytest.approx((5.0, 0.0), abs=1e-12)
    assert wind_vector(5.0, 180.0) == pytest.approx((0.0, 5.0), abs=1e-12)
    assert wind_vector(5.0, 0.0) == pytest.approx((0.0, -5.0), abs=1e-12)


def test_sectors_are_centred_on_north():
    assert sector_of(np.array([0.0, 11.2, 11.3, 348.8, 348.7, 270.0])).tolist() == [0, 0, 1, 0, 15, 12]


def _write_record(monkeypatch, tmp_path, year=2025):
    idx = pd.date_range(f"{year}-01-01", periods=24 * 365, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    speed = rng.gamma(2.0, 2.0, len(idx))
    drct = np.where(speed < forcing.CALM_M_S, np.nan, rng.choice([270.0, 300.0, 180.0], len(idx)))
    df = pd.DataFrame({"speed_m_s": speed, "direction_deg": drct, "gust_m_s": speed * 1.5}, index=idx)
    df.index.name = "datetime"
    df.iloc[24 * 100 + 5] = np.nan  # one missing hour
    path = tmp_path / "asos.csv"
    df.to_csv(path)
    monkeypatch.setattr(sites.SiteConfig, "asos", lambda self, y: path)
    return df


def test_day_series_fills_a_missing_hour_from_its_neighbour(monkeypatch, tmp_path):
    df = _write_record(monkeypatch, tmp_path)
    site = sites.get_site("campanile")
    day = sites.Day("d", "d", "2025-04-11")  # day 100 of a non-leap year
    hours = day_series(site, day)
    assert hours.shape == (24, 3)
    assert np.isfinite(hours[:, 0]).all()
    assert hours[5, 0] in (df.iloc[24 * 100 + 4, 0], df.iloc[24 * 100 + 6, 0])


def test_rose_frequencies_sum_to_one(monkeypatch, tmp_path):
    _write_record(monkeypatch, tmp_path)
    r = rose(sites.get_site("campanile"), 2025)
    assert sum(r["frequency"]) == pytest.approx(1.0)
    assert len(r["mean_m_s"]) == forcing.SECTORS == 16
    assert 0.0 < r["calm_fraction"] < 0.2
    assert r["frequency"][12] > 0 and r["frequency"][3] == 0
