"""AEP inversion, pinned by an independently published number.

FEMA states a 1 %-annual-chance location has a 26 % chance of flooding over a 30-year mortgage.
That is the same arithmetic this module does, from a source that has never seen this code.
"""

import numpy as np
import pytest

from forcing import RETURN_PERIODS_YR
from probability import aep_to_horizon, depth_stack_to_aep, loglinearity_r2, summarize


def monotone_stack(shape=(4, 4), base=0.0, step=0.05):
    """Depth rising linearly with return-period index, identical in every cell."""
    return np.stack([np.full(shape, base + i * step, dtype=np.float32)
                     for i in range(len(RETURN_PERIODS_YR))])


def test_matches_fema_thirty_year_mortgage_figure():
    """1 - 0.99**30 = 0.26030, i.e. FEMA's published 26 % over a 30-year mortgage.

    Earlier project notes recorded this as 0.2606. That was a transcription slip, not a code
    defect -- the exact value is 0.26030 and always was. Pinned here so it cannot drift again.
    """
    aep = np.array([[0.01]], dtype=np.float32)
    assert float(aep_to_horizon(aep, 30)[0, 0]) == pytest.approx(0.26030, abs=5e-5)


def test_horizon_probability_is_monotone_and_bounded():
    aep = np.array([[0.0, 0.01, 0.5, 1.0]], dtype=np.float32)
    p1, p30 = aep_to_horizon(aep, 1), aep_to_horizon(aep, 30)
    assert np.all(p30 >= p1)
    assert np.all((p30 >= 0.0) & (p30 <= 1.0))
    assert float(p1[0, 0]) == 0.0 and float(p1[0, 3]) == 1.0


def test_cell_flooded_by_the_most_frequent_storm_clamps_at_its_aep():
    stack = monotone_stack(base=1.0, step=0.0)
    aep, fixed = depth_stack_to_aep(stack, threshold_m=0.15)
    assert fixed == 0
    assert np.allclose(aep, 1.0 / RETURN_PERIODS_YR[0])


def test_cell_never_flooded_has_zero_aep():
    aep, _ = depth_stack_to_aep(monotone_stack(base=0.0, step=0.001), threshold_m=10.0)
    assert np.all(aep == 0.0)


def test_threshold_at_a_simulated_return_period_recovers_its_aep():
    """A threshold landing exactly on the 100-yr depth must invert to AEP = 1/100."""
    idx = RETURN_PERIODS_YR.index(100)
    stack = monotone_stack(base=0.0, step=0.05)
    aep, _ = depth_stack_to_aep(stack, threshold_m=float(stack[idx, 0, 0]))
    assert np.allclose(aep, 1.0 / 100.0, rtol=1e-5)


def test_aep_decreases_as_the_threshold_rises():
    stack = monotone_stack(base=0.0, step=0.05)
    a = depth_stack_to_aep(stack, threshold_m=0.10)[0]
    b = depth_stack_to_aep(stack, threshold_m=0.30)[0]
    assert np.all(b <= a)
    assert b.max() < a.max()


def test_monotonicity_is_enforced_and_counted():
    """Solver jitter can invert a cell's depth curve by a hair; that is fixed, and reported."""
    stack = monotone_stack(base=0.1, step=0.05)
    stack[4, 0, 0] = stack[3, 0, 0] - 0.02  # one non-monotone cell
    aep, fixed = depth_stack_to_aep(stack, threshold_m=0.25)
    assert fixed == 1
    assert np.isfinite(aep).all()


def test_stack_and_return_periods_must_agree():
    with pytest.raises(AssertionError):
        depth_stack_to_aep(monotone_stack()[:3], threshold_m=0.15)


def test_loglinearity_is_one_for_a_perfect_log_fit():
    depths = {t: 2.0 * np.log(t) + 1.0 for t in RETURN_PERIODS_YR}
    assert loglinearity_r2(depths) == pytest.approx(1.0, abs=1e-9)


def test_summary_areas_are_nested():
    aep = np.array([[0.0, 0.005, 0.05, 0.5]], dtype=np.float32)
    s = summarize(aep, cell_area_m2=25.0, threshold_m=0.15, duration_hr=24.0)
    assert s.area_any_risk_ha >= s.area_gt_1pct_ha >= s.area_gt_10pct_ha
    assert s.cells_at_risk == 3
    assert s.as_dict()["threshold_m"] == 0.15
