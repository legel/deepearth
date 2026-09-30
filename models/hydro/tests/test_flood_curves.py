"""FLOOD's per-cell curves hold the solver's depth; a return below the peak surface is flooded."""

import numpy as np

import flood_curves as FC


def _frames():
    t = np.arange(0, 3600 * 6 + 1, 120.0)
    tri = np.interp(t, [0, 1800, 5400, 18000, t[-1]], [0, 0, 0.12, 0.0, 0.0])
    two = np.interp(t, [0, 600, 2400, 3600, 6000, 9000, t[-1]], [0, 0, 0.05, 0.02, 0.09, 0.0, 0.0])
    return t, np.stack([tri, two, np.zeros_like(t)])


def test_eight_knots_rebuild_a_hydrograph():
    t, D = _frames()
    tk, hk = FC.fit_knots(D, t)
    assert tk.shape == (3, 8) and np.all(np.diff(tk, axis=1) >= 0)
    for i in range(3):
        rec = np.array([FC.depth_at(tk[i:i + 1], hk[i:i + 1], x)[0] for x in t])
        assert np.abs(rec - D[i]).max() < 2e-3
    err = FC.error(D, t, tk, hk)
    assert err["rmse_m"] < 1e-3 and err["iou"]["0.003"] > 0.95 and err["volume_rel_err_at_fullest"] < 0.02


def test_knots_quantize_within_their_units():
    t, D = _frames()
    tk, hk = FC.fit_knots(D, t)
    q = FC.quantize(tk, hk)
    assert q.shape == (2, 8, 3) and q.dtype == np.dtype("<u2")
    tk2, hk2 = FC.dequantize(q)
    assert np.abs(tk2 - tk).max() <= FC.T_UNIT_S / 2 and np.abs(hk2 - hk).max() <= FC.H_UNIT_M / 2 + 1e-12


def test_a_return_below_the_peak_surface_is_flooded():
    _, D = _frames()
    surf = FC.peak_surface(np.array([10.0, 10.0, 10.0]), D)
    assert np.allclose(surf, [10.12, 10.09, 10.0])
    wet, depth = FC.flooded(np.array([10.02, 10.5, np.nan]), np.array([surf[0], surf[1], np.nan]))
    assert wet.tolist() == [True, False, False] and np.isclose(depth[0], 0.10)
