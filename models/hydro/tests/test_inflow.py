"""Edge inflow: sampling in time, and the D8 estimate from a coarse DEM."""

import numpy as np
import pytest

from inflow import Inflow, accumulate, fill_and_route, rim_cells, rim_inflow, watershed_inflow


def test_uniform_puts_discharge_on_the_named_edges_only():
    q = Inflow.uniform((4, 6), {"north": 0.5, "east": 0.25}, 100.0).at(50.0)
    assert q["north"].tolist() == [0.5] * 6 and q["east"].tolist() == [0.25] * 4
    assert q["west"].max() == 0.0 and q["south"].max() == 0.0


def test_at_interpolates_and_holds_the_ends():
    inflow = Inflow(times_s=np.array([0.0, 10.0, 20.0]), west=np.array([[0.0], [1.0], [3.0]]),
                    east=np.zeros((3, 1)), north=np.zeros((3, 2)), south=np.zeros((3, 2)))
    assert inflow.at(5.0)["west"][0] == pytest.approx(0.5)
    assert inflow.at(15.0)["west"][0] == pytest.approx(2.0)
    assert inflow.at(-5.0)["west"][0] == 0.0 and inflow.at(99.0)["west"][0] == 3.0


def plane(nrows: int = 20, ncols: int = 10, dx: float = 10.0, slope: float = 0.01) -> np.ndarray:
    rows = np.arange(nrows, dtype=np.float64)[:, None]
    return np.repeat(50.0 - rows * dx * slope, ncols, axis=1)


def test_d8_on_a_plane_drains_straight_downslope():
    z = plane()
    filled, recv, _ = fill_and_route(z)
    assert np.array_equal(filled, z)
    rows, cols = z.shape
    for r in range(rows - 1):
        for c in range(1, cols - 1):
            assert recv[r, c] == (r + 1) * cols + c
    assert (recv[-1, :] == -1).all()


def test_accumulation_on_a_plane_is_the_column_above():
    z = plane()
    filled, recv, order = fill_and_route(z)
    area = accumulate(filled, recv, order, 10.0)
    for r in range(z.shape[0]):
        assert area[r, 5] == pytest.approx((r + 1) * 100.0)
    assert area.sum() > 0 and area[-1, 1:-1].sum() + area[:, 0].sum() + area[:, -1].sum() >= z.size * 100.0 / 2


def test_a_pit_is_filled_to_its_spill_and_still_drains():
    z = plane()
    z[8:11, 4:7] -= 2.0
    filled, recv, order = fill_and_route(z)
    assert (filled >= z).all()
    assert filled[8:11, 4:7].max() == pytest.approx(z[11, 5])
    area = accumulate(filled, recv, order, 10.0)
    outlets = recv == -1
    assert outlets.sum() > 0 and area[outlets].sum() == pytest.approx(z.size * 100.0)


def test_watershed_inflow_on_a_plane_is_the_upstream_strip():
    P, dx = 1e-5, 10.0
    inflow = watershed_inflow(plane(), dx, window=(10, 15, 2, 8), shape=(25, 30),
                              rain=[P, 2 * P], dt_s=60.0)
    assert inflow.times_s.tolist() == [0.0, 60.0]
    assert inflow.north.shape == (2, 30) and inflow.west.shape == (2, 25)
    assert np.allclose(inflow.north[0], 10 * dx * dx * P / dx)
    assert np.allclose(inflow.north[1], 2 * 10 * dx * dx * P / dx)
    assert inflow.south.max() == 0.0 and inflow.west.max() == 0.0 and inflow.east.max() == 0.0


def test_watershed_inflow_conserves_volume_on_a_fine_grid_that_does_not_divide():
    """1124 fine cells over 6 coarse ones: no cell is dropped and the rate is preserved."""
    P, dx, n = 1e-5, 10.0, 1124
    inflow = watershed_inflow(plane(), dx, window=(10, 15, 2, 8), shape=(37, n),
                              rain=[P], dt_s=60.0)
    assert inflow.north.shape == (1, n) and (inflow.north[0] > 0).all()
    fine_dx = dx * 6 / n
    assert inflow.north[0].sum() * fine_dx == pytest.approx(6 * 10 * dx * dx * P, rel=1e-12)


def test_cap_unit_discharge_conserves_volume_and_removes_the_slot():
    """A whole catchment through one cell forces metres of depth; the cap spreads it."""
    from inflow import cap_unit_discharge

    q = np.zeros((2, 200))
    q[:, 70] = 27.8
    q[1] *= 0.5
    capped = cap_unit_discharge(q, max_depth_m=0.3)
    ceiling = 0.9 * 0.3 * np.sqrt(9.81 * 0.3)
    assert capped.max() <= ceiling * (1 + 1e-12)
    assert capped.sum(axis=1) == pytest.approx(q.sum(axis=1), rel=1e-12)
    assert (capped[0] > 0).sum() == 200


def test_cap_unit_discharge_leaves_a_flow_under_the_ceiling_alone():
    from inflow import cap_unit_discharge

    q = np.full((1, 5), 0.02)
    assert cap_unit_discharge(q, max_depth_m=0.3) == pytest.approx(q)


def test_cap_unit_discharge_saturates_when_the_edge_cannot_carry_it():
    """Beyond what the whole edge can convey at the ceiling, volume cannot be preserved."""
    from inflow import cap_unit_discharge

    ceiling = 0.9 * 0.3 * np.sqrt(9.81 * 0.3)
    q = np.full((1, 4), 10 * ceiling)
    capped = cap_unit_discharge(q, max_depth_m=0.3)
    assert capped == pytest.approx(np.full((1, 4), ceiling))
    assert capped.sum() < q.sum()


def test_rim_inflow_delivers_each_row_upstream_of_a_disc_and_the_budget_closes():
    """On a plane falling east, each coarse row's cells west of the disc drain into it."""
    from affine import Affine

    from solver import Surface, SolverConfig, simulate

    x = np.arange(40) + 0.5
    zc = np.tile(100.0 - 0.05 * x, (40, 1))
    tc, tf = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 40.0), Affine(0.5, 0.0, 10.0, 0.0, -0.5, 30.0)
    r, c = np.indices((40, 40))
    domain = np.hypot(10.25 + 0.5 * c - 20.0, 29.75 - 0.5 * r - 20.0) <= 9.0
    rain = [1e-5] * 4
    flow = rim_inflow(zc, tc, domain, domain, tf, rain, 60.0, runoff=0.5)
    cy, cx = np.indices(zc.shape)
    fx, fy = 2 * cx - 19, 2 * cy - 19
    on = (fx >= 0) & (fx < 40) & (fy >= 0) & (fy < 40)
    inside = np.zeros(zc.shape, bool)
    inside[on] = domain[fy[on], fx[on]]
    entry = ~inside[:, :-1] & inside[:, 1:]
    expected = rain[0] * 0.5 * (np.nonzero(entry)[1] + 1).sum()
    assert flow.rim[0].sum() * 0.25 == pytest.approx(expected, rel=1e-9) and expected > 0
    assert domain.ravel()[flow.rim_index].all() and not (~rim_cells(domain)).ravel()[flow.rim_index].any()
    z = np.where(domain, 0.0, np.nan)
    res = simulate(Surface(z=z), rain, SolverConfig(dx=0.5, dt_s=60.0, frame_interval_min=2.0, device="cpu"),
                   inflow=flow, verbose=False)
    assert res.mass.inflow == pytest.approx(expected * 240.0, rel=1e-3)
    assert abs(res.mass.residual_pct) < 1e-3
