"""The three samples of the record, and the SOIL MOISTURE and DROUGHT maps."""

import numpy as np
import pytest
import torch

import balance as B
import routing
import water_modes as WM
from test_balance import _plane, cells, state


def _hours(y0: int, n: int) -> np.ndarray:
    return np.datetime64(f"{y0}-01-01T00", "s").astype(np.int64) + 3600 * np.arange(n, dtype=np.int64)


def test_the_sites_et0_is_the_balances_formula():
    rng = np.random.default_rng(0)
    rn, rs = rng.uniform(-80, 700, 200), rng.uniform(0, 900, 200)
    t, ea = rng.uniform(-5, 35, 200), rng.uniform(0.3, 2.5, 200)
    pa, u2 = rng.uniform(95, 101, 200), rng.uniform(0, 6, 200)
    rs[:50] = 0.0
    want = B.et0_hourly(*(torch.as_tensor(v, dtype=torch.float64) for v in (rn, rs, t, ea, pa, u2))).numpy()
    assert np.allclose(WM.et0_hourly_np(rn, rs, t, ea, pa, u2), want, atol=1e-9)


def test_air_gaps_are_filled_as_the_balance_fills_them():
    n = 48
    ea = np.full(n, 1.2)
    ea[10:20] = np.nan
    pet = WM.site_pet(np.full(n, 300.0), np.full(n, 20.0), ea, np.full(n, 100.0), np.full(n, 330.0), np.full(n, 2.0))
    assert np.isfinite(pet).all() and np.allclose(pet, pet[0])


def test_the_storm_is_the_deepest_event():
    h = _hours(2021, 2000)
    p = np.zeros(2000)
    p[10:14] = 5.0                  # 20 mm
    p[100:103] = 2.0                # 6 mm
    p[103 + 5] = 30.0               # within 6 dry hours: the same event, 36 mm
    p[500:502] = 9.0                # 18 mm
    ev = WM.storm_events(p, h)
    s, refuted = WM.five_year_storm(ev, lambda e: None, [2021])
    assert len(ev) == 3 and not refuted
    assert s["total_mm"] == 36.0 and s["start_s"] == h[100] and s["end_s"] == h[108] + 3600 and s["tail_h"] == 24


def test_a_storm_an_independent_gauge_refutes_gives_way_to_the_next():
    h = _hours(2019, 24 * 365 * 7)
    t = lambda s: int(np.searchsorted(h, np.datetime64(s, "s").astype(np.int64)))  # noqa: E731
    p = np.zeros(len(h))
    p[t("2019-03-01T00:00:00"):t("2019-03-01T10:00:00")] = 20.0      # the biggest, but before the five years
    p[t("2024-07-05T21:00:00"):t("2024-07-06T03:00:00")] = 12.0      # 72 mm: another gauge saw a tenth
    p[t("2022-09-05T22:00:00"):t("2022-09-06T09:00:00")] = 6.0       # 66 mm: another gauge saw it
    other = p * 0.1
    other[t("2022-09-05T22:00:00"):t("2022-09-06T09:00:00")] = 5.0
    ev = WM.storm_events(p, h)
    years = WM.complete_years(range(2019, 2027), 2026)
    s, refuted = WM.five_year_storm(ev, lambda e: WM.corroborate(e, h, p, {"ground gauge": other}), years)
    assert years == [2021, 2022, 2023, 2024, 2025]
    assert s["total_mm"] == 66.0 and [e["total_mm"] for e in refuted] == [72.0]


def test_the_recession_stays_inside_72_hours():
    assert WM.recession_h({"start_s": 0, "end_s": 6 * 3600}) == 24
    assert WM.recession_h({"start_s": 0, "end_s": 55 * 3600}) == 17


def test_the_drought_is_the_largest_deficit_window_from_local_midnight():
    n = 24 * 400
    h = _hours(2021, n)
    pet = np.full(n, 0.1)
    rain = np.zeros(n)
    rain[::24] = 3.0                              # 3 mm a day against 2.4 of ET0: no deficit
    rain[24 * 200:24 * 291] = 0.0                 # 91 dry days from day 200
    d = WM.drought_window(h, rain, pet, 0.0)
    assert d["start_s"] == h[24 * 200] and d["days"] == 91
    assert d["rain_mm"] == 0.0 and d["deficit_mm"] == pytest.approx(91 * 2.4, abs=0.2)
    d8 = WM.drought_window(h, rain, pet, -8.0)
    assert (d8["start_s"] - 8 * 3600) % 86400 == 0


def test_the_drought_window_survives_gaps_and_says_when_none_is_whole():
    h = _hours(2022, 24 * 400)
    pet = np.full(h.size, 0.1)
    pet[24 * 200:24 * 291] = 0.5
    pet[5::97] = np.nan                           # scattered gaps, a few hours a day at most
    d = WM.drought_window(h, np.zeros(h.size), pet, 0.0)
    assert 190 <= (d["start_s"] - int(h[0])) // 86400 <= 210 and np.isfinite(d["pet_mm"])
    pet[::3] = np.nan                             # every day short of 20 hours
    with pytest.raises(ValueError, match="no 91-day window"):
        WM.drought_window(h, np.zeros(h.size), pet, 0.0)


def test_the_typical_year_is_the_median():
    h = np.concatenate([_hours(y, 10) for y in range(2021, 2026)])
    rain = np.concatenate([np.full(10, v) for v in (50, 10, 30, 40, 20)])
    assert WM.median_year(h, rain, range(2021, 2026))["year"] == 2023


def test_sampled_lanes_take_each_samples_year_and_its_spin_up():
    y = lambda s: int(np.datetime64(s, "s").astype(np.int64))  # noqa: E731
    per = {"typical": {"year": 2023},
           "drought": {"start_s": y("2024-05-23T08:00:00"), "end_s": y("2024-08-22T08:00:00")},
           "storm": {"start_s": y("2022-12-30T00:00:00"), "end_s": y("2023-01-01T07:00:00")}}
    assert WM.sampled_lanes(per, range(2020, 2026)) == [(2022, 2021), (2023, 2022), (2024, 2023)]
    assert WM.sampled_lanes(per, range(2022, 2026)) == [(2022, 2022), (2023, 2022), (2024, 2023)]


def test_the_typical_theta_weights_months_by_hours():
    m = np.zeros((12, 2))
    m[0] = 0.6
    assert np.allclose(WM.typical_theta(m, [744] + [0] * 10 + [744]), 0.3)


def test_the_drought_is_the_balances_deficit():
    z = _plane(2, 4)
    n = z.size
    net = routing.network(z, np.ones_like(z, bool), 0.5)
    k = cells(n)
    rain = [0.0] * 48
    one = lambda v: (lambda h: torch.full((n,), v, dtype=torch.float64))              # noqa: E731
    air = lambda h: (25.0, 1.0, 100.0, 330.0)                                          # noqa: E731
    cwd = WM.drought_cwd(state(n, 0.12), k, net, rain, one(600.0), one(3.0), air, len(rain))
    want = B.run(state(n, 0.12), k, net, len(rain), rain, one(600.0), one(3.0), air).metrics(k)["cwd_mm"].numpy()
    assert np.allclose(cwd, want) and (cwd > 0).all()


def test_codes_round_trip_and_reserve_the_top():
    c = WM.encode(np.array([0.1, 0.2, 0.35, np.nan, 0.5]), 0.15, 0.40)
    assert c.tolist() == [0, 50, 200, 255, 250]
    assert np.allclose(WM.decode(c, 0.15, 0.40)[[1, 2]], [0.2, 0.35])


def test_only_plantable_ground_is_drawn():
    theta = np.array([0.20, 0.21, 0.05, 0.22, 0.30, 0.25])
    no_soil = np.array([False, False, False, True, False, False])
    roof = np.array([False, False, False, False, True, False])
    perv = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0])                  # cell 2: a crown over paving
    site = np.array([True, True, True, True, True, False])
    codes, (lo, hi) = WM.soil_moisture_map(theta, no_soil, roof, perv, site)
    assert codes[2] == codes[3] == WM.CODE_SEALED and codes[4] == WM.CODE_BUILDING and codes[5] == WM.CODE_NONE
    assert lo > 0.05, "a sealed crown is not on the scale"
    dc, _ = WM.drought_map(np.array([10.0, 20.0, 400.0, 0.0, 5.0, 1.0]), no_soil, roof, perv, site)
    assert dc[2] == dc[3] == dc[4] == WM.CODE_SEALED


def test_a_near_uniform_site_reads_near_uniform():
    rng = np.random.default_rng(1)
    theta = 0.208 + 0.002 * rng.standard_normal(1000)
    z, one = np.zeros(1000, bool), np.ones(1000)
    codes, (lo, hi) = WM.soil_moisture_map(theta, z, z, one, ~z, wp=0.051, theta_s=0.387)
    assert hi - lo >= WM.MIN_SPAN * (0.387 - 0.051) - 1e-9 and 0.051 <= lo < 0.208 < hi <= 0.387
    assert np.ptp(codes) < 45
