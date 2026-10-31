"""The short mask forms of masking.regime_mask, and that their shares at 0 leave the reference draw unchanged."""

import numpy as np

import masking
from data import IS_FLUX, IS_REAN, IS_TOWER, IS_TOWER_MET, T_WIN, V_TAB


def test_reference_draw_unchanged():
    present = np.ones(V_TAB, bool)
    a, b = np.random.default_rng(7), np.random.default_rng(7)
    p = np.array([0.35, 0.20, 0.15, 0.10, 0.20])
    for _ in range(50):
        assert masking.FAMILIES[int(a.choice(8, p=np.r_[p, 0, 0, 0]))] == masking.FAMILIES[int(b.choice(5, p=p))]


def test_sub_all_hides_every_tower_row_for_one_block():
    h = masking.regime_mask("sub_all", np.random.default_rng(0), np.ones(V_TAB, bool), 4)
    assert h[IS_TOWER].sum(1).tolist() == [4] * int(IS_TOWER.sum()) and not h[~IS_TOWER].any()


def test_flux_short_leaves_meteorology_visible():
    h = masking.regime_mask("flux_short", np.random.default_rng(0), np.ones(V_TAB, bool))
    assert h[IS_FLUX].any() and not h[~IS_FLUX].any()


def test_m5_blocks_every_tower_and_reanalysis_row():
    h = masking.regime_mask("m5", np.random.default_rng(0), np.ones(V_TAB, bool))
    assert h[IS_TOWER | IS_REAN].any(1).all() and h.shape == (V_TAB, T_WIN)
    assert any(h[r].all() for r in (IS_TOWER_MET, IS_REAN, IS_FLUX))


def test_draw_reaches_the_new_families():
    cfg = masking.MaskConfig(p_block_all=0, p_block_var=0, p_short=0, p_r3=0, p_mixed=0, p_sub_all=1, p_reveal=0)
    assert masking.draw(np.random.default_rng(1), np.ones(V_TAB, bool), cfg)[2].startswith("sub_all")
