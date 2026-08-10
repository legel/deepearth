from deepearth.autoresearch.champion_report import passes


def test_promotion_requires_harmonic_gain_and_arithmetic_breadth():
    old = {"harmonic": 0.30, "arithmetic": 0.60}
    assert passes(old, {"harmonic": 0.31, "arithmetic": 0.60})
    assert not passes(old, {"harmonic": 0.30, "arithmetic": 0.61})
    assert not passes(old, {"harmonic": 0.31, "arithmetic": 0.59})
