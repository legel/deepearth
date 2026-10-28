"""The runner's score rows: point scores from the sample mean and the truth sums used for pooled R2 and WAPE."""

import numpy as np
import pandas as pd

from run import score
from windows import Window


def _window(T=96):
    t = pd.date_range("2020-01-01", periods=T, freq="30min")
    y = np.sin(np.arange(T) / 7.0) * 3.0 - 0.5
    vals = np.stack([y, y + 1.0])
    return Window("XX-Tst", t[0], t, ["NEE", "ERA5_TA"], vals, np.isfinite(vals), ["tower", "reanalysis"])


def test_score_rows_carry_truth_sums_and_mean_based_point_scores():
    w = _window()
    mask = np.zeros(w.values.shape, bool)
    mask[0, 10:58] = True
    rng = np.random.default_rng(0)
    samples = w.values[None] + rng.normal(0.4, 1.0, size=(64,) + w.values.shape)
    rows = pd.DataFrame(score("m", "R1", "M2-1d", w, mask, samples))
    r = rows[(rows.kind == "values") & (rows.gap_class == "all")].iloc[0]
    y = w.values[0, 10:58]
    mu = samples[:, 0, 10:58].mean(0)
    assert r.n == 48
    assert np.isclose(r.sum_y, y.sum()) and np.isclose(r.sum_y2, (y ** 2).sum()) and np.isclose(r.sum_abs_y, np.abs(y).sum())
    assert np.isclose(r.mae, np.abs(mu - y).mean()) and np.isclose(r.rmse, np.sqrt(((mu - y) ** 2).mean()))
    # pooled R2 and WAPE from the row alone equal the direct computation
    sse = r.n * r.rmse ** 2
    sst = r.sum_y2 - r.sum_y ** 2 / r.n
    assert np.isclose(1 - sse / sst, 1 - ((mu - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    assert np.isclose(100 * r.n * r.mae / r.sum_abs_y, 100 * np.abs(mu - y).sum() / np.abs(y).sum())
    s = rows[rows.kind == "sum-daily"].iloc[0]
    assert {"sum_y", "sum_y2", "sum_abs_y"} <= set(s.index)


def test_extra_tower_rows_leave_release_masks_unchanged():
    """--extra-tower (E30): the masks of the release rows are identical with and without extra tower rows; M2 hides
    the extra rows with the other tower rows; WD enters as sine and cosine with WD's flag."""
    from run import masks_for
    from windows import EXTRA_TOWER_VARS, TOWER_VARS, load_window, with_extra_columns
    T = 180 * 48
    t = pd.date_range("2020-01-01", periods=T, freq="30min")
    rng = np.random.default_rng(0)
    cols = {"time_lst": t}
    for v in ("NEE", "LE", "H", "TA", "SW_IN", "VPD", "SW_OUT", "RH", "WD", "TS_2"):
        cols[v] = rng.normal(size=T) if v != "WD" else rng.uniform(0, 360, T)
        cols[f"{v}_flag"] = np.zeros(T, int)
    cols["ERA5_TA"] = rng.normal(size=T)
    tab = with_extra_columns(pd.DataFrame(cols))
    assert np.allclose(tab.WD_SIN ** 2 + tab.WD_COS ** 2, 1.0) and (tab.WD_SIN_flag == 0).all()
    w0 = load_window(tab, "XX-Tst", "2020-01-01")
    w1 = load_window(tab, "XX-Tst", "2020-01-01", tower_vars=TOWER_VARS + EXTRA_TOWER_VARS)
    assert set(w1.keys) - set(w0.keys) == {"SW_OUT", "RH", "WD_SIN", "WD_COS", "TS_2"}
    fams = ["M2", "M2-sub", "M5", "M4-reanalysis"]
    m0, m1 = masks_for(w0, fams), masks_for(w1, fams)
    idx = [w1.keys.index(k) for k in w0.keys]
    ext = [i for i, k in enumerate(w1.keys) if k in EXTRA_TOWER_VARS]
    assert m0.keys() == m1.keys()
    for k in m0:
        assert np.array_equal(m0[k], m1[k][idx]), k
    tow = [i for i, p in enumerate(w1.provenance) if p == "tower" and i not in ext]
    assert np.array_equal(m1["M2-30d"][ext], np.repeat(m1["M2-30d"][tow[:1]], len(ext), 0))
