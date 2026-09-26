"""Run-on: fills, flow weights, and a mass-conserving cascade in processing order."""

import numpy as np

import routing as R


def plane(ny=6, nx=8, slope=0.05, dx=0.5):
    return np.tile(10.0 - slope * np.arange(nx) * dx, (ny, 1))


def test_plain_fill_raises_a_pit_to_its_spill_and_epsilon_makes_it_drain():
    z = plane()
    z[3, 4] -= 0.2
    valid = np.ones_like(z, bool)
    plain = R.fill(z, valid, 0.0)
    assert abs(plain[3, 4] - z[3, 5]) < 1e-9
    assert np.array_equal(np.delete(plain.ravel(), 3 * 8 + 4), np.delete(z.ravel(), 3 * 8 + 4))
    assert R.fill(z, valid)[3, 4] > plain[3, 4]


def test_the_network_runs_downslope_in_order_with_unit_weights():
    net = R.network(plane(), np.ones((6, 8), bool), 0.5)
    for p in range(len(net.order)):
        a, b = net.ptr[p], net.ptr[p + 1]
        if b > a:
            assert (net.rcv[a:b] > p).all() and abs(float(net.w[a:b].sum()) - 1.0) < 1e-6


def test_everything_leaves_a_sealed_plane_downhill():
    z = plane()
    net = R.network(z, np.ones_like(z, bool), 0.5)
    n = z.size
    infil, pond, inflow, lost = R.cascade(net, np.ones(n), np.zeros(n), np.ones(n))
    assert abs(lost.sum() - n) < 1e-5 and infil.sum() == 0 and pond.sum() == 0
    assert np.abs(lost.reshape(z.shape)[:, :-1]).max() < 1e-5


def test_a_depression_holds_water_and_mass_is_conserved():
    z = plane()
    z[2:4, 3:5] -= 0.1
    net = R.network(z, np.ones_like(z, bool), 0.5)
    supply = np.full(z.size, 2.0)
    infil, pond, inflow, lost = R.cascade(net, supply, np.zeros(z.size), np.ones(z.size))
    assert pond.sum() > 0 and np.all(pond <= net.room + 1e-5)
    assert abs(infil.sum() + pond.sum() + lost.sum() - supply.sum()) < 1e-5 * supply.sum()


def test_run_on_grows_downslope_and_feeds_infiltration():
    z = plane()
    net = R.network(z, np.ones_like(z, bool), 0.5)
    n = z.size
    infil, pond, inflow, lost = R.cascade(net, np.ones(n), np.full(n, 0.5), np.ones(n))
    runon = inflow.reshape(z.shape).mean(0)
    assert np.all(np.diff(runon) >= -1e-6) and runon[-1] > runon[1]
    assert np.allclose(infil, 0.5) and abs(infil.sum() + lost.sum() - n) < 1e-5 * n


def test_a_cell_that_does_not_infiltrate_sheds_everything_that_reaches_it():
    z = plane()
    net = R.network(z, np.ones_like(z, bool), 0.5)
    n = z.size
    perv = np.ones(n)
    perv[z.ravel() > 9.95] = 0.0
    infil, pond, inflow, lost = R.cascade(net, np.ones(n), np.full(n, 100.0), perv)
    assert np.all(infil[perv == 0] == 0) and np.allclose(infil[perv == 1], 1.0 + inflow[perv == 1])
