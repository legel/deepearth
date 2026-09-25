"""Each tower-to-forcing rule on a small fixture."""

import numpy as np

import neon_fill
import rain as R
import sky
import wind as W
from qc import FILL, NONE, OBSERVED, REANALYSIS, REFERENCE, STATION_CALM, TOWER, layer, measured

N = 24 * 60
MONTH = np.repeat([0, 1], N // 2)


def test_a_gap_filled_hour_never_counts_as_measured():
    v, q = np.array([1.0, 2.0, np.nan, 4.0]), np.array([0, 2, 0, np.nan])
    assert np.allclose(measured(v, q), [1.0, np.nan, np.nan, np.nan], equal_nan=True)
    vals, code = layer(4, [(measured(v, q), TOWER), (np.full(4, 9.0), REANALYSIS)])
    assert vals.tolist() == [1.0, 9.0, 9.0, 9.0] and code.tolist() == [TOWER, REANALYSIS, REANALYSIS, REANALYSIS]
    assert layer(2, [(None, TOWER)])[1].tolist() == [NONE, NONE]


def test_an_unsupported_spike_hour_is_not_rain():
    p, other, grid = np.zeros(N), np.zeros(N), np.zeros(N)
    p[10], other[10], grid[11] = 57.5, 20.0, 18.0          # 57.5 > 2 x 20 + 1: the spike goes
    p[40], other[40], grid[40] = 30.0, 16.0, 16.0          # 30 <= 2 x 16 + 1: kept
    got, rec = R.vet_rain(p, [other, grid], MONTH)
    assert np.isnan(got[10]) and got[40] == 30.0 and rec["hours_out"] == 1


def test_a_month_far_from_its_sources_is_not_rain_and_a_lone_gauge_stands():
    p, other, grid = np.zeros(N), np.zeros(N), np.zeros(N)
    p[N // 2::24], other[N // 2::24], grid[N // 2::24] = 1.0, 7.0, 6.5  # an undercatching month
    got, rec = R.vet_rain(p, [other, grid], MONTH)
    assert rec["months_out"] == [1] and np.isnan(got[N // 2:]).all() and np.isfinite(got[: N // 2]).all()
    same, rec0 = R.vet_rain(p, [], MONTH)
    assert np.array_equal(same, p) and rec0 == {"hours_out": 0, "months_out": []}


def test_the_reference_first_then_the_vetted_gauge_then_the_analysis():
    gauge, qc = np.full(N, 0.5), np.zeros(N)
    ref = np.full(N, np.nan)
    ref[:100] = 0.4
    gauge[200], qc[300] = 40.0, 2.0                        # a spike, and a gap-filled hour
    grid = np.full(N, 0.45)
    p, code, rec = R.rain(gauge, qc, MONTH, reference=ref, analysis=grid)
    assert (code[:100] == REFERENCE).all() and p[0] == 0.4
    assert code[150] == TOWER and p[150] == 0.5
    assert code[200] == FILL and p[200] == 0.45 and code[300] == FILL and rec["hours_out"] == 1


def test_a_station_calm_carries_its_own_code():
    ws = np.array([0.0, 2.572, 0.0, 3.0])
    codes = np.array([FILL, FILL, TOWER, OBSERVED], np.uint8)
    assert W.calm_codes(ws, codes).tolist() == [STATION_CALM, FILL, TOWER, OBSERVED]
    assert abs(W.calm_ceiling_m_s() - 1.543) < 1e-3 and abs(W.calm_ceiling_m_s(0.54) - 0.833) < 1e-3


def test_the_sonic_is_carried_to_the_reference_height_and_back():
    u = np.array([4.0])
    ref = W.transfer(u, 39.16, 2.0, 17.0, 0.03, 0.0)
    back = W.transfer(ref, 10.0, 0.03, 0.0, 2.0, 17.0, z_ref=39.16)
    assert np.allclose(back, u) and ref[0] > 0
    assert np.isnan(W.z0_by_sector(*(np.zeros(5),) * 6, z_m=39.0, d_m=17.0)).all(), "no neutral hours, no z0"


def test_k_c_against_the_clear_sky_and_a_daytime_gap_filled_along_it():
    cs = np.array([0.0, 200.0, 400.0, 600.0, 400.0, 0.0])
    ghi = np.array([0.0, 100.0, np.nan, 300.0, 200.0, 0.0])
    kc = sky.clear_sky_index(ghi, cs)
    assert np.isnan(kc[0]) and kc[1] == 0.5 and np.isnan(kc[2])
    out, gap = sky.fill_ghi(ghi, cs)
    assert gap.tolist() == [False, False, True, False, False, False] and out[2] == 200.0 and out[0] == 0.0


def test_neon_fills_what_the_release_did_not_measure_and_never_a_measured_step():
    t = np.array([0, 1800, 3600])
    v, q = np.array([10.0, 20.0, np.nan]), np.array([0.0, 2.0, 255.0])
    tn, vn = np.array([0, 1800, 3600, 5400, 7200]), np.array([99.0, 21.0, 30.0, 40.0, np.nan])
    steps, got, qc, took = neon_fill.fill(t, v, q, tn, vn)
    assert steps.tolist() == [0, 1800, 3600, 5400] and got.tolist() == [10.0, 21.0, 30.0, 40.0]
    assert qc.tolist() == [0.0, 0.0, 0.0, 0.0] and took == 3
