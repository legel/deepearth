"""Each quality rule on a small fixture with a known answer."""

import numpy as np

import qc as Q
from variables import VARIABLES, base, columns


def test_a_gap_filled_value_is_never_measured():
    v = np.array([1.0, 2.0, -9999.0, 4.0, 5.0])
    qc = np.array([0, 1, 0, 3, -9999])
    assert Q.measured(v, qc).tolist() == [True, False, False, False, False]
    assert Q.measured(v).tolist() == [True, True, False, True, True], "a raw column without QC: every value"


def test_the_fp_missing_value_is_nan():
    assert np.isnan(Q.clean([-9999, -9999.0, -9000.5])).all() and Q.clean([0.0])[0] == 0.0


def test_physical_range_with_a_radiation_offset_tolerance():
    v = np.array([-12.0, -5.0, 0.0, 1300.0, 1311.0, np.nan])
    assert Q.out_of_range(v, 0, 1300, tol=10).tolist() == [True, False, False, False, True, False]
    assert Q.out_of_range(v, 0, 1300).tolist() == [True, True, False, False, True, False]


def test_a_long_constant_run_is_stuck_and_a_short_one_is_not():
    v = np.r_[np.arange(5.0), np.full(12, 7.5), np.arange(3.0), np.full(11, 2.25)]
    s = Q.stuck(v, 12)
    assert s[5:17].all() and not s[:5].any() and not s[17:].any()


def test_a_missing_step_breaks_a_run():
    v = np.r_[np.full(6, 1.0), np.nan, np.full(6, 1.0)]
    assert not Q.stuck(v, 12).any()


def test_exempt_values_and_dark_repeats_are_not_stuck():
    rh = np.r_[np.full(30, 100.0), np.arange(5.0)]
    assert not Q.stuck(rh, 24, exempt=(100.0,)).any()
    sw = np.r_[np.zeros(20), np.arange(1.0, 6.0)]
    dark = np.r_[np.ones(20, bool), np.zeros(5, bool)]
    assert not Q.stuck(sw, 12, dark=dark).any(), "zeros wholly at night are darkness"
    dark2 = np.r_[np.ones(10, bool), np.zeros(15, bool)]
    s = Q.stuck(sw, 12, dark=dark2)
    assert s[10:20].all() and not s[:10].any(), "zeros in daylight are a failed sensor; its night zeros are right"
    ts = np.r_[np.full(200, 0.1), np.arange(5.0)]
    assert not Q.stuck(ts, 96, exempt=(0.0,), atol=0.5).any(), "soil held at 0 deg C while it freezes"


def test_papale_spike_test_finds_an_isolated_spike():
    rng = np.random.default_rng(0)
    n = 48 * 13
    v = np.sin(np.arange(n) * 2 * np.pi / 48) * 5 + rng.normal(0, 0.3, n)
    v[100] += 15.0
    v[400] -= 12.0
    night = (np.arange(n) % 48) < 24
    s = Q.spikes(v, night, 48)
    assert s[100] and s[400] and s.sum() <= 4


def test_spike_test_leaves_a_smooth_series_alone():
    n = 48 * 26
    v = np.sin(np.arange(n) * 2 * np.pi / 48) * 10 + np.random.default_rng(1).normal(0, 0.5, n)
    assert Q.spikes(v, (np.arange(n) % 48) < 24, 48).sum() <= 2


def test_low_ustar_filters_day_and_night_and_unverifiable_steps():
    u = np.array([0.05, 0.3, np.nan, 0.1, 0.5])
    thr = np.array([0.2, 0.2, 0.2, np.nan, 0.2])
    assert Q.low_ustar(u, thr).tolist() == [True, False, True, False, False]
    night = np.array([False, True, True, True, True])
    assert Q.low_ustar(u, thr, only=night).tolist() == [False, False, True, False, False]


def test_solar_noon_and_the_clock_test():
    lat, lon, off = 38.1, -121.65, -8.0
    day0 = int((np.datetime64("2016-06-21T00:00") - np.datetime64("1970-01-01T00:00")) // np.timedelta64(1, "m"))
    t = day0 + np.arange(0, 1440 * 60, 30)
    el = Q.solar_elevation(lat, lon, Q.step_midpoints_utc(t, 30, off))
    first = el[:48]
    noon_lst = (np.argmax(first) * 30 + 15) / 60
    assert 12.0 <= noon_lst <= 12.75, noon_lst  # solar noon near 12:06 PST at 121.65 W in June
    assert 74 < first.max() < 76  # 90 - 38.1 + 23.44
    sw = np.clip(np.sin(np.radians(el)), 0, None) * 900
    assert Q.clock_offset_minutes(sw, el, 30) == 0.0
    late = np.roll(sw, 2)  # the record's stamps are one hour late
    assert Q.clock_offset_minutes(late, el, 30) == 60.0


def test_sign_and_unit_checks():
    rng = np.random.default_rng(2)
    sw = rng.uniform(50, 900, 2000)
    nee = -0.02 * sw + rng.normal(0, 2, 2000)
    assert not Q.sign_check(nee, sw, -1)["flipped"]
    assert Q.sign_check(-nee, sw, -1)["flipped"]
    assert Q.fch4_unit_check(rng.lognormal(3, 1, 2000))["suspect_umol"] is False
    assert Q.fch4_unit_check(rng.lognormal(3, 1, 2000) / 1000)["suspect_umol"] is True


def test_timestamp_regularity():
    t = np.r_[np.arange(0, 300, 30), 270, 330, 345]
    r = Q.regular(t)
    assert r["step"] == 30 and r["duplicates"] == 1 and r["irregular"] == 2


def test_the_registry_names_columns_consistently():
    assert columns("TS", 2) == ("TS_F_MDS_2", "TS_F_MDS_2_QC")
    assert base("TS_2") == "TS" and base("SW_IN") == "SW_IN" and base("NEE") == "NEE"
    assert VARIABLES["NEE"].oneflux == ("NEE_VUT_REF", "NEE_VUT_REF_QC")
    assert VARIABLES["FCH4"].oneflux is None, "the Shuttle carries no FCH4"
    for v in VARIABLES.values():
        assert v.lo < v.hi and v.unit


def test_moving_point_test_recovers_a_known_threshold():
    rng = np.random.default_rng(3)
    n = 20000
    u = rng.uniform(0.02, 0.8, n)
    ta = rng.uniform(0, 20, n)
    resp = 2.0 * np.exp(0.07 * ta)
    nee = resp * np.clip(u / 0.25, 0, 1) + rng.normal(0, 0.3, n)  # flux lost below u* = 0.25
    month = np.repeat(np.arange(1, 13), n // 12 + 1)[:n]
    thr = Q.ustar_mpt(nee, u, ta, np.ones(n, bool), month)
    assert thr is not None and 0.2 < thr < 0.32, thr
