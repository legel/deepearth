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
