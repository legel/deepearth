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
