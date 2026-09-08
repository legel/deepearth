"""The friction exponent, pinned by the property that fixed it.

A 4/3 shipped here for months. It is wrong because `q` in the Bates et al. (2010) update is
unit discharge, not velocity, and it is hard to spot because the two exponents agree exactly at
h = 1 m where hf**0 = 1 -- while diverging most in this project's own operating regime, a
median wet depth of 7-8 cm.
"""

import numpy as np
import pytest

from physics import G, MANNING_EXP


def steady_unit_discharge(h: float, slope: float, n: float, exponent: float) -> float:
    """Iterate the semi-implicit friction update to its fixed point on a uniform slope.

    This is the solver's own momentum update with the flux divergence removed, so whatever it
    converges to is what the solver believes Manning's equation is.
    """
    q, dt = 0.0, 0.5
    for _ in range(200_000):
        num = q + G * h * dt * slope
        den = 1.0 + G * dt * n**2 * abs(q) / (h**exponent + 1e-30)
        q_next = num / den
        if abs(q_next - q) < 1e-15:
            return q_next
        q = q_next
    return q


def manning_unit_discharge(h: float, slope: float, n: float) -> float:
    """Manning's equation for unit discharge [m^2/s]: q = h**(5/3) * sqrt(S) / n."""
    return h ** (5.0 / 3.0) * np.sqrt(slope) / n


DEPTHS = [0.02, 0.05, 0.10, 0.25, 0.50, 1.00]


def test_manning_exp_is_seven_thirds():
    assert MANNING_EXP == pytest.approx(7.0 / 3.0)


@pytest.mark.parametrize("h", DEPTHS)
def test_steady_state_reproduces_manning(h):
    """At 7/3 the fixed point IS Manning's equation, at every depth."""
    slope, n = 1.94e-3, 0.040
    got = steady_unit_discharge(h, slope, n, MANNING_EXP)
    want = manning_unit_discharge(h, slope, n)
    assert got == pytest.approx(want, rel=1e-3), f"h={h}: {got:.6g} vs Manning {want:.6g}"


def test_four_thirds_over_predicts_in_the_shallow_regime():
    """The regression guard: 4/3 must be visibly wrong where this project actually operates."""
    slope, n = 1.94e-3, 0.040
    err = {h: (steady_unit_discharge(h, slope, n, 4.0 / 3.0)
               / manning_unit_discharge(h, slope, n) - 1.0)
           for h in (0.02, 0.10)}
    assert err[0.10] > 2.0, f"expected >200% over-prediction at h=0.10 m, got {err[0.10]:.1%}"
    assert err[0.02] > 6.0, f"expected >600% over-prediction at h=0.02 m, got {err[0.02]:.1%}"


def test_exponents_agree_only_at_unit_depth():
    """Why the bug survived review: hf**0 = 1 makes the exponent irrelevant at h = 1 m."""
    slope, n = 1.94e-3, 0.040
    a = steady_unit_discharge(1.0, slope, n, MANNING_EXP)
    b = steady_unit_discharge(1.0, slope, n, 4.0 / 3.0)
    assert a == pytest.approx(b, rel=1e-6)
