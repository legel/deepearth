"""Each cover its own season, and the trees their measured storage: Tonzi Ranch's oaks went dry by July on a 1 m bucket
and spent it in February and March on the grass's leaf area."""
import numpy as np
import torch

import phenology as P
import rootzone as R

CYC = {"2021": {"Greenup": "2020-11-01", "Maturity": "2021-01-01", "Senescence": "2021-04-01", "Dormancy": "2021-06-01"},
       "2022": {"Greenup": "2021-11-01", "Maturity": "2022-01-01", "Senescence": "2022-04-01", "Dormancy": "2022-06-01"}}


def _air(n_days, wet=0.5):
    t = np.arange(n_days * 24)
    ta = 15.0 - 8.0 * np.cos(2 * np.pi * (t / 24.0 - 15) / 365.0) + 8.0 * np.sin(2 * np.pi * (t % 24 - 9) / 24.0)
    return ta, wet * 0.6108 * np.exp(17.27 * ta / (ta + 237.3))


def test_the_stage_curve_is_fao56s_from_the_cycle_dates():
    g = P.stage_daily(CYC, 2022)
    assert len(g) == 365
    assert g[0] == 1.0 and g[80] == 1.0, "mature from January to the start of April"
    assert g[200] == 0.0, "dormant in July"
    assert 0.4 < g[(np.datetime64("2022-05-01") - np.datetime64("2022-01-01")).astype(int)] < 0.6, "half way down"
    nov = (np.datetime64("2022-12-01") - np.datetime64("2022-01-01")).astype(int)
    assert 0.3 < g[nov] < 0.7, "the next cycle's green-up from the median dates"
    assert np.all(P.stage_daily({}, 2022) == 1.0)


def test_the_growing_season_index_rises_with_warmth_and_light_and_falls_with_dry_air():
    t = np.arange(365 * 24)
    ta = 12.0 - 10.0 * np.cos(2 * np.pi * (t / 24.0 - 15) / 365.0) + 6.0 * np.sin(2 * np.pi * (t % 24 - 9) / 24.0)
    es = 0.6108 * np.exp(17.27 * ta / (ta + 237.3))
    wet, dry = P.gsi_daily(ta, 0.9 * es, 42.5), P.gsi_daily(ta, 0.2 * es, 42.5)
    assert wet[10] < 0.05 and wet[180] > 0.9, "bare in January, full in June (humid air)"
    assert dry[200] < wet[200], "dry summer air closes the trees down"
    assert np.all((wet >= 0) & (wet <= 1))


def test_a_grass_pixel_gives_the_trees_their_own_season_and_a_tree_pixel_keeps_the_sites():
    g = P.stage_daily(CYC, 2022)
    site = 0.5 + 1.5 * g + 0.6 * (1.0 - g)
    years = {"2022": site.tolist()}
    ta, ea = _air(365)
    grass = {"dominant": "herbaceous", "lat": 38.4, "cycles": {"herbaceous": CYC, "woody": {}}}
    woody, herb = P.class_daily(grass, years, 0.5, 2022, ta, ea)
    assert site[40] > site[200] and woody[40] < woody[200], "the oaks are bare when the grass is green"
    assert herb[40] == 1.0 and herb[200] == 0.0
    w2, _ = P.class_daily(dict(grass, dominant="woody"), years, 0.5, 2022, ta, ea)
    np.testing.assert_allclose(w2, site, err_msg="a tree-dominated pixel's LAI is the trees'")


def test_grass_kcb_follows_its_stage_and_the_trees_follow_their_leaves():
    k = torch.tensor([0.85, 0.0])
    np.testing.assert_allclose(P.grass_kcb(k, 0.0).numpy(), [P.KC_MIN, 0.0], rtol=1e-6)
    np.testing.assert_allclose(P.grass_kcb(k, 1.0).numpy(), [0.85, 0.0], rtol=1e-6)
    lo, hi = P.kcb(torch.tensor(0.95), torch.tensor(0.0)), P.kcb(torch.tensor(0.95), torch.tensor(5.0))
    assert abs(float(lo) - P.KC_MIN) < 1e-6 and 0.9 < float(hi) < 0.95


def test_the_woody_root_zone_holds_the_sites_storage_within_its_cover_and_the_trees_maximum():
    z = R.woody_depth(np.array([1.0, 1.0, 1.0]), np.array([0.26, 0.26, 0.40]), np.array([0.13, 0.13, 0.39]), 366.0)
    np.testing.assert_allclose(z[0], 366.0 / 130.0, rtol=1e-6)
    assert z[2] == R.WOODY_ROOT_MAX_M, "a soil holding little is capped at the trees' mean maximum"
    assert np.all(R.woody_depth(np.array([1.0]), np.array([0.3]), np.array([0.1]), 50.0) == 1.0), "never shallower"
    assert np.all(R.woody_depth(np.array([1.0]), np.array([0.3]), np.array([0.1]), None) == 1.0)
    lat, lon = np.array([38.40, 38.45]), np.array([-121.0, -120.95])
    v = np.array([[366.0, np.nan], [10.0, 20.0]], np.float32)
    assert R.storage_mm(38.401, -120.999, lat, lon, v)["mm"] == 366.0
    assert R.storage_mm(38.401, -120.951, lat, lon, v)["mm"] is None, "a masked pixel: the cover's depth"
    assert R.storage_mm(45.0, -100.0, lat, lon, v)["mm"] is None
