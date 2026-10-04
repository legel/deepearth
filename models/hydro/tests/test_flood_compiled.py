"""The compiled priority flood and accumulation are the Python ones, number for number (inflow.py, 2026-10-04)."""
import numpy as np
import pytest

import inflow

numba = pytest.importorskip("numba")


@pytest.mark.parametrize("seed,levels", [(0, None), (1, 4), (2, 1)])
def test_the_compiled_flood_is_the_python_flood_on_ties_and_flats(seed, levels):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(37, 53)).cumsum(0) + rng.normal(size=(37, 53)).cumsum(1)
    if levels:                                       # many equal levels: the heap's order turns on the cell index
        z = np.round(z / np.ptp(z) * levels)
    flat = np.ascontiguousarray(z, np.float64).ravel()
    jit = inflow._flood_jit(flat, *z.shape, inflow.OFFSETS[:, 0].astype(np.int64), inflow.OFFSETS[:, 1].astype(np.int64))
    py = inflow._flood_py(z)
    for a, b in zip(jit, py):
        assert np.array_equal(a, b)


def test_the_compiled_accumulation_adds_in_the_same_order():
    rng = np.random.default_rng(3)
    z = rng.normal(size=(41, 29)).cumsum(0)
    filled, recv, order = inflow.fill_and_route(z)
    got = inflow.accumulate(filled, recv, order, 0.7)
    acc, rf = np.full(z.size, 0.7 * 0.7), recv.ravel()
    for c in np.lexsort((-order.ravel(), -filled.ravel())):
        if rf[c] >= 0:
            acc[rf[c]] += acc[c]
    assert np.array_equal(got.ravel(), acc)
