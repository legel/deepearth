"""Mask families: shape, extent, determinism."""

import numpy as np

from masks import (STEPS_PER_DAY, contiguous_block, cut_outage, modality_dropout, seed_for, transplanted_outage)

V, T = 6, 180 * STEPS_PER_DAY


def test_seeds_are_stable_and_distinct():
    assert seed_for("US-Twt", "2018-04-01", "M2") == seed_for("US-Twt", "2018-04-01", "M2")
    assert seed_for("US-Twt", "2018-04-01", "M2") != seed_for("US-Twt", "2018-04-01", "M3")


def test_a_block_covers_exactly_its_days_on_its_rows():
    for days in (1, 7, 30, 90, 180):
        m = contiguous_block([0, 2], V, T, days, np.random.default_rng(seed_for("s", "w", "M2", days)))
        assert m[0].sum() == m[2].sum() == days * STEPS_PER_DAY and m[[1, 3, 4, 5]].sum() == 0
        idx = np.flatnonzero(m[0])
        assert idx[-1] - idx[0] + 1 == idx.size


def test_a_cut_outage_always_masks_something_and_is_deterministic():
    rec = np.zeros(3 * T, bool)
    rec[2 * T + 100: 2 * T + 400] = True
    a = cut_outage(rec, T, np.random.default_rng(1))
    b = cut_outage(rec, T, np.random.default_rng(1))
    assert a.any() and (a == b).all() and a.shape == (T,)
    m = transplanted_outage([0, 1, 2], V, a)
    assert (m[0] == a).all() and m[3:].sum() == 0


def test_modality_dropout_hides_the_whole_modality():
    m = modality_dropout([4, 5], V, T)
    assert m[4:].all() and not m[:4].any()


def test_m5_and_subdaily_families():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from masks import contiguous_steps, mixed_blocks, M5_MIN_STEPS, M5_MAX_STEPS
    V, T = 6, 8640
    rng = np.random.default_rng(3)
    m = contiguous_steps([0, 2], V, T, 12, rng)
    assert m.sum() == 24 and m[0].sum() == 12 and not m[1].any()
    runs = np.flatnonzero(np.diff(np.concatenate(([0], m[0].astype(int), [0]))))
    assert len(runs) == 2                               # one contiguous block
    a, pa = mixed_blocks(range(V), V, T, np.random.default_rng(7))
    b, pb = mixed_blocks(range(V), V, T, np.random.default_rng(7))
    assert (a == b).all() and pa == pb                  # deterministic per seed
    starts = {p[0] for p in pa.values()}
    assert len(starts) > 1                              # placement differs across variables
    for r, (s0, n) in pa.items():
        assert M5_MIN_STEPS <= n <= min(M5_MAX_STEPS, T) and a[r].sum() == n and a[r, s0:s0 + n].all()
    lens = [mixed_blocks([0], 1, T, np.random.default_rng(k))[1][0][1] for k in range(400)]
    assert min(lens) < 48 < 48 * 30 < max(lens)          # spans sub-daily to monthly


def test_window_from_record_hourly_and_outside():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from masks import window_from_record
    rt = np.arange("2020-01-01T00:00", "2020-01-02T00:00", np.timedelta64(60, "m"), dtype="datetime64[m]")
    rm = np.zeros(rt.size, bool); rm[3] = True          # 03:00-04:00 hidden in an hourly record
    wt = np.arange("2019-12-31T23:00", "2020-01-01T05:00", np.timedelta64(30, "m"), dtype="datetime64[m]")
    out = window_from_record(rm, rt, wt)
    hid = wt[out]
    assert list(hid.astype(str)) == ["2020-01-01T03:00", "2020-01-01T03:30"]
    assert not out[:2].any()                            # before the record start
