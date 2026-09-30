"""The Year's wind run: the grouped sum against each hour taken alone, its codes, the fill, the typical hour."""

import numpy as np
import torch

import year_run as Y


def per_hour(S, uref, theta):
    """Each hour exactly: U_ref x |unit (u, v) blended at the hour's own fraction between its two headings|, km."""
    out = np.zeros(S.shape[1])
    for u, th in zip(uref, theta):
        if np.isnan(u) or np.isnan(th):
            continue
        x = (th % 360.0) / 22.5
        k0, t = int(np.floor(x)) % 16, x - np.floor(x)
        f = (1 - t) * S[k0] + t * S[(k0 + 1) % 16]
        out += u * np.hypot(f[:, 0], f[:, 1])
    return out * Y.KM_PER_MS_H


def test_the_grouped_run_is_every_hour_summed():
    rng = np.random.default_rng(3)
    S = rng.normal(0.0, 1.0, (16, 500, 2)).astype(np.float32)
    uref = rng.gamma(2.0, 2.0, 8760)
    theta = rng.uniform(0, 360, 8760)
    uref[::97] = np.nan
    got = Y.run_km(torch.as_tensor(S), uref, theta).numpy()
    rel = np.abs(got - per_hour(S, uref, theta)) / per_hour(S, uref, theta)
    assert np.median(rel) < 0.002 and rel.max() < 0.01


def test_codes_on_the_fixed_scale_and_none():
    assert Y.quantize(np.array([0.0, 50.0, 100.0, 250.0, np.nan]), 100.0).tolist() == [0, 127, 254, 254, 255]
    assert [Y.nice_ceil(v) for v in (73.0, 120.0, 0.3, 100.0, 51_234.0)] == [100.0, 200.0, 0.5, 100.0, 100_000.0]


def test_levels_are_read_in_log_height_and_the_log_law_below():
    lv = np.zeros((16, 2, 2, 1, 1), np.float32)
    lv[:, 0, 0] = 1.0                                     # u 1 at 4 m
    lv[:, 0, 1] = 2.0                                     # u 2 at 16 m
    x, y = np.array([0.5, 0.5]), np.array([-0.5, -0.5])
    S = Y.point_basis(lv, 1.0, (0.0, 0.0), [4.0, 16.0], x, y, np.array([8.0, 0.0]), np.array([False, True]),
                      np.array([1.0, 0.03]))
    assert np.isclose(S[0, 0, 0], 1.5), "a crown at 8 m: halfway in ln(height) between 4 and 16 m"
    assert np.isclose(S[0, 1, 0], np.log(2 / 0.03) / np.log(4 / 0.03), rtol=1e-5), "ground: 2 m up, log law from 4 m"


def test_a_return_the_solve_does_not_reach_takes_the_flow_beside_it():
    S = np.full((16, 4, 2), np.nan, np.float32)
    S[:, 0] = [1.0, 0.0]                                  # reached, 2 m up
    S[:, 1] = [2.0, 0.0]                                  # reached, 30 m up
    xy = np.array([[0.0, 0.0], [0.0, 0.0], [0.5, 0.0], [0.2, 0.0]], np.float32)
    h = np.array([2.0, 30.0, 2.5, 60.0])
    out, filled = Y.fill_solids(S, xy, h, np.full(4, 0.03))
    assert np.isfinite(out).all() and filled == 2
    assert np.isclose(out[0, 2, 0], np.log(2.5 / 0.03) / np.log(2.0 / 0.03), rtol=1e-4)
    assert np.isclose(out[0, 3, 0], 2.0 * np.log(60 / 0.03) / np.log(30 / 0.03), rtol=1e-4)


def test_the_typical_hour_is_the_prevailing_sectors_median_speed():
    u = np.array([0.2, 3.0, 5.0, 7.0, 9.0, 4.0])
    f = np.array([90.0, 270.0, 271.0, 268.0, 90.0, 272.0])
    assert Y.typical_hour(u, f) == (2, 5.0, 271.0)


def test_the_ribbons_blend_the_two_headings_about_the_direction():
    lv = np.zeros((16, 2, 3, 2, 2), np.float32)
    lv[4, 0, 0] = 1.0
    lv[5, 1, 0] = 1.0
    U, V = Y.ribbon_field(lv, 2.0, 90.0 + 22.5 / 4)
    assert np.allclose(U, 1.5) and np.allclose(V, 0.5)
