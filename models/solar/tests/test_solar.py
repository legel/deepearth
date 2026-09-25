"""The sky's separation, its identity with the tower, the horizon and sky-view factor, and the canopy's light."""

import math

import numpy as np
import pytest
import torch

import canopy
import horizon
import sky
import transposition


def _x(kt):
    kt = np.asarray(kt, dtype=float)
    return {"kt": kt, "ast": np.full(kt.shape, 12.0), "zen": np.full(kt.shape, 40.0), "dktc": 0.75 - kt,
            "kde": np.zeros(kt.shape)}


def test_erbs_is_the_published_piecewise_fit():
    kd = sky.erbs(_x([0.1, 0.5, 0.9]))
    assert abs(kd[0] - 0.991) < 1e-9 and kd[2] == 0.165
    assert abs(kd[1] - (0.9511 - 0.1604 * 0.5 + 4.388 * 0.25 - 16.638 * 0.125 + 12.336 * 0.0625)) < 1e-9


@pytest.mark.parametrize("params", [sky.ENGERER2_START, sky.ENGERER2_US])
def test_engerer2_is_a_bounded_falling_fraction(params):
    kd = sky.engerer2(params, _x(np.linspace(0.05, 0.85, 17)))
    assert np.all((kd >= 0) & (kd <= 1)) and kd[0] > 0.8 and kd[-1] < 0.3 and np.all(np.diff(kd) <= 1e-9)


def test_fit_recovers_the_model_that_generated_the_data_and_ranks_it_first():
    rng = np.random.default_rng(0)
    truth = (0.08, -4.0, 8.0, -0.01, 0.002, -5.0, 0.5)
    samples = {}
    for s in ("A", "B", "C"):
        x = {"kt": rng.uniform(0.05, 0.85, 800), "ast": rng.uniform(7, 17, 800), "zen": rng.uniform(20, 80, 800),
             "dktc": rng.uniform(-0.1, 0.6, 800), "kde": np.zeros(800)}
        ghi = rng.uniform(100, 900, 800)
        samples[s] = dict(x, ghi=ghi, dhi=sky.engerer2(truth, x) * ghi, sat_kd=np.full(800, np.nan))
    res = sky.fit(samples)
    assert res["loso"]["engerer2"]["rmse"] < 1.0 and res["loso"]["erbs"]["rmse"] > 5 * res["loso"]["engerer2"]["rmse"]


def _hours(n=24 * 30, seed=3):
    """A month of synthetic 5-minute geometry and clear sky at a mid-latitude, with the sun's daily course."""
    t = np.arange(n * sky.SUB) / sky.SUB
    cosz = np.clip(np.sin(np.pi * ((t % 24) - 6) / 12) * 0.8, 0, None)
    e0 = np.full_like(t, 1361.0)
    dni_cs = np.where(cosz > 0, 900 * cosz ** 0.3, 0.0)
    dhi_cs = 80 * cosz
    g = {"cosz": cosz, "e0": e0, "dni_cs": dni_cs, "dhi_cs": dhi_cs, "ghi_cs": dni_cs * cosz + dhi_cs,
         "eot": np.zeros_like(t)}
    rng = np.random.default_rng(seed)
    cs = sky.hourly(g["ghi_cs"])
    ghi = cs * rng.uniform(0.1, 1.35, n)                          # up to 35 % above clear sky
    dhi = np.where(rng.random(n) < 0.5, ghi * rng.uniform(0.05, 0.9, n), np.nan)
    return g, ghi, dhi


def test_beam_and_diffuse_sum_back_to_the_tower_s_ghi_every_hour():
    """At an unobstructed horizontal sensor the simulated SW_IN is the tower's, exactly, including broken-cloud hours
    above clear sky: what the beam caps withhold is diffuse."""
    g, ghi, dhi = _hours()
    x = sky.predictors(g, ghi, -72.17)
    s = sky.split(ghi, dhi, g, x)
    day = x["ext"] > 1.0
    assert np.allclose((s["dni"] * np.clip(s["cosz"], 0, None) + s["dhi"])[day], ghi[day], atol=1e-6)
    z = np.degrees(np.arccos(np.clip(s["cosz"], 0, 1)))
    hd = transposition.hay_davies(s["dni"], s["dhi"], z, np.full_like(z, 1361.0))
    e = transposition.irradiance(hd["beam"], hd["diffuse"], ghi, s["cosz"], 1.0, 1.0, 1.0)
    lit = day & (s["cosz"] > transposition.MIN_COSZ)
    assert np.allclose(e[lit], ghi[lit], atol=1e-6), "open level ground returns GHI through Hay-Davies"


def test_a_daytime_hour_no_source_measured_is_filled_in_clear_sky_index_and_night_stays_dark():
    cs = np.array([0.0, 100.0, 400.0, 800.0, 400.0, 100.0, 0.0])
    ghi = np.array([0.0, 50.0, np.nan, np.nan, np.nan, 90.0, np.nan])
    out, filled = sky.fill_ghi(ghi, cs)
    assert list(filled) == [False, False, True, True, True, False, False] and out[-1] == 0.0
    kc = out[1:6] / cs[1:6]
    assert np.allclose(kc, np.linspace(0.5, 0.9, 5))


def test_the_measured_atmosphere_scales_each_hour_and_keeps_its_shape():
    g, _, _ = _hours(48)
    cs = {k: sky.hourly(g[f"{k}_cs"]) * 1.1 for k in ("ghi", "dni", "dhi")}
    cs["ghi"][5] = np.nan
    out = sky.with_clear_sky(g, cs)
    h = sky.hourly(out["ghi_cs"])
    day = sky.hourly(g["ghi_cs"]) > sky.CS_MIN
    keep = day & np.isfinite(cs["ghi"])
    assert np.allclose(h[keep], cs["ghi"][keep]) and not out["cs_measured"][5]


def test_an_open_horizontal_surface_sees_the_whole_sky_and_a_wall_s_foot_half():
    n = torch.tensor([[0.0, 0.0, 1.0]])
    assert float(horizon.sky_view(torch.zeros(1, 64), n)) == pytest.approx(1.0, abs=2e-3)
    wall = torch.zeros(1, 64)
    wall[0, :32] = 90.0                                      # an infinite wall filling the eastern half
    assert float(horizon.sky_view(wall, n)) == pytest.approx(0.5, abs=2e-3)


def test_the_march_finds_a_tower_s_elevation_and_nothing_where_there_is_nothing():
    g = horizon.Grid(1.0, 200, 200, -100.0, -100.0)
    z = torch.full((200, 200), float("-inf"))
    z[90:110, 140:160] = 30.0                                 # a 30 m block 40 to 60 m east
    obs = torch.tensor([[0.0, 0.0, 0.0]])
    h, o, _ = horizon.march(z, g, obs, horizon.MarchConfig(bins=64))
    east = 16                                                 # the wedge centered on 90 degrees
    assert float(h[0, east]) == pytest.approx(math.degrees(math.atan(30.0 / 40.0)), abs=1.5)
    assert float(h[0, 48]) == 0.0 and torch.equal(h, o)


def test_a_transmitting_canopy_band_lets_through_its_transmittance():
    g = horizon.Grid(1.0, 200, 200, -100.0, -100.0)
    z = torch.full((200, 200), float("-inf"))
    z[90:110, 140:160] = 30.0
    tau = torch.zeros(200, 200)
    tau[90:110, 140:160] = 0.4
    h, o, band = horizon.march(z, g, torch.tensor([[0.0, 0.0, 0.0]]), horizon.MarchConfig(bins=64), tau)
    assert float(o[0, 16]) == 0.0 and float(band[0, 16]) == pytest.approx(0.4, abs=1e-6)
    svf = horizon.sky_view(h, torch.tensor([[0.0, 0.0, 1.0]]), o, band)
    assert float(horizon.sky_view(h, torch.tensor([[0.0, 0.0, 1.0]]))) < float(svf) < 1.0


def test_beer_lambert_through_the_canopy():
    assert canopy.beam_transmittance(0.0, 0.5) == 1.0
    assert canopy.beam_transmittance(1.0, 1.0) == pytest.approx(math.exp(-1.0))
    assert canopy.beam_transmittance(1.0, 0.5) == pytest.approx(math.exp(-2.0)), "a low sun crosses more canopy"
    d = canopy.diffuse_transmittance(np.array([0.0, 0.5, 2.0]))
    assert d[0] == pytest.approx(1.0) and 0 < d[2] < d[1] < 1 and d[1] < math.exp(-0.5), "diffuse is attenuated more"


def test_the_first_return_gap_and_the_season_s_leaves():
    rng = np.random.default_rng(1)
    n = 400
    xyz = np.column_stack([rng.uniform(0, 2, n), rng.uniform(0, 2, n), np.zeros(n)])
    xyz[:100, 2] = 0.2                                        # a quarter of first returns reach the ground
    xyz[100:, 2] = rng.uniform(10, 20, n - 100)
    cls = np.where(xyz[:, 2] < 1, 2, 5)
    c = canopy.columns(xyz, cls, np.ones(n, int), cell=2.0)
    assert float(c["tau"][0, 0]) == pytest.approx(-math.log(100.5 / 401.0))
    t0 = np.array([[1.0, 2.0], [np.nan, 3.0]])
    summer = canopy.seasonal_tau(t0, lai_site=4.0, floor=1.0)
    assert np.isnan(summer[1, 0]) and summer[0, 0] == pytest.approx(1.0 + 0.5 * 0.8 * 3.0 * 0.5)
    assert np.allclose(canopy.seasonal_tau(t0, 4.0, 1.0, lai_survey=4.0)[np.isfinite(t0)], t0[np.isfinite(t0)])
    green = np.r_[np.full(20, 5.0), np.full(20, 1.0)]
    on = canopy.leaves_on_trees(green, lag=10)
    assert on[25] == 5.0 and on[31] == 1.0, "a turned leaf stays ten days"
