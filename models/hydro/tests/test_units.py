"""Geographic units of a storm run (WS18 M3): the split, the seams the coarse pass records, the discharge a
unit takes across its edges, and a whole run against its units joined, in numbers. No network, no site
data."""

import numpy as np
import pytest
from affine import Affine

import units
from inflow import Inflow
from solver import SolverConfig, Surface, simulate

MM_HR = 1.0 / 1000.0 / 3600.0


def test_the_split_covers_the_square_once_and_its_windows_reach_past_their_cores():
    s = units.split((1000.0, 2000.0), 100.0, 2, 3, 20.0)
    assert len(s) == 6 and s[0]["core"] == (900.0, 900.0 + 200.0 / 3, 2000.0, 2100.0), "north-west first"
    t = Affine(1.0, 0.0, 900.0, 0.0, -1.0, 2100.0)                    # 200 x 200 cells of 1 m
    seen = np.zeros((200, 200), int)
    for u in s:
        r0, r1, c0, c1 = units.cells(t, (200, 200), u["core"])
        seen[r0:r1, c0:c1] += 1
        w = units.cells(t, (200, 200), u["window"])
        assert w[0] <= r0 and w[1] >= r1 and w[2] <= c0 and w[3] >= c1
    assert (seen == 1).all()
    xs, ys = units.seam_lines(s, (1000.0, 2000.0), 100.0)
    assert all(900.0 < x < 1100.0 for x in xs) and len(xs) == 4 and len(ys) == 2, "each window edge inside, once"


def test_a_unit_takes_across_each_edge_only_what_flows_into_it():
    t = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0)                        # 10 x 10 cells, x and y in [0, 10]
    rows_y = 10.0 - (np.arange(10) + 0.5)
    cols_x = np.arange(10) + 0.5
    seams = units.Seams(times_s=np.array([0.0, 60.0]), xs=np.array([2.0, 8.0]), ys=np.array([3.0]),
                        row_y=rows_y, col_x=cols_x,
                        qx=np.stack([np.full((2, 10), 0.4), np.full((2, 10), 0.3)]),   # both eastward
                        qy=np.stack([np.full((2, 10), 0.2)]))              # southward, as the solver's v
    flow = units.unit_inflow(seams, (2.0, 8.0, 3.0, 10.0), Affine(1.0, 0.0, 2.0, 0.0, -1.0, 10.0), (7, 6))
    assert np.allclose(flow.west, 0.4) and np.allclose(flow.east, 0.0), "eastward enters the west edge only"
    assert np.allclose(flow.north, 0.0), "the north edge is the domain's own: no line there"
    assert np.allclose(flow.south, 0.0), "southward flow leaves across the south edge"
    other = units.unit_inflow(seams, (0.0, 2.0, 0.0, 3.0), Affine(1.0, 0.0, 0.0, 0.0, -1.0, 3.0), (3, 2))
    assert np.allclose(other.north, 0.2) and np.allclose(other.east, 0.0), "southward enters from the north"


def test_the_recorder_reads_the_solvers_own_directions_between_the_cells_either_side_of_a_line():
    """The solver's v is southward (`solver.FIELDS`); a line on a face takes the mean of its two cells."""
    n = 24
    t = Affine(1.0, 0.0, 0.0, 0.0, -1.0, float(n))
    rec = units.SeamRecorder(t, (n, n), [12.0], [12.0])
    ramp = np.arange(n, dtype=np.float64)
    rec(0.0, np.stack([np.ones((n, n)), np.tile(ramp, (n, 1)), np.tile(ramp[:, None], (1, n))]))
    s = rec.seams()
    assert np.allclose(s.qx[0, 0], 11.5) and np.allclose(s.qy[0, 0], 11.5), "x = 12 m and y = 12 m lie on faces"
    z = (10.0 - 0.02 * np.indices((n, n))[0]).astype(np.float32)          # falling south
    rec = units.SeamRecorder(t, (n, n), [12.0], [12.0])
    cfg = SolverConfig(dx=1.0, dt_s=60.0, frame_interval_min=1.0, device="cpu", dtype="float64")
    simulate(Surface(z=z), [60.0 * MM_HR] * 5, cfg, inflow=None, sink=rec, verbose=False)
    s = rec.seams()
    assert s.qy[0, -1].mean() > 0.0, "water running south crosses y = 12 m southward, positive"
    assert np.abs(s.qx[0, -1]).max() < 0.1 * s.qy[0, -1].max()


def _slope(n: int = 48, dx: float = 1.0) -> np.ndarray:
    """A plane falling south-east with a shallow bowl, every cell modelled."""
    r, c = np.indices((n, n)).astype(np.float64)
    bowl = -0.3 * np.exp(-((r - 30) ** 2 + (c - 20) ** 2) / 40.0)
    return (10.0 - 0.01 * dx * r - 0.005 * dx * c + bowl).astype(np.float32)


def test_units_joined_follow_the_whole_run_away_from_their_seams_and_close_their_mass():
    """2 x 2 units with a 10 m overlap, fed by a pass over the whole domain at the same cell: their cores
    joined against the whole run's peak depth, far from the seams and within one overlap of them."""
    n, dx = 48, 1.0
    z = _slope(n, dx)
    t = Affine(dx, 0.0, 0.0, 0.0, -dx, n * dx)
    rain = [60.0 * MM_HR] * 15
    cfg = SolverConfig(dx=dx, dt_s=60.0, frame_interval_min=0.5, device="cpu", dtype="float64")
    centre, reach = (n * dx / 2.0, n * dx / 2.0), n * dx / 2.0
    split = units.split(centre, reach, 2, 2, 10.0)
    xs, ys = units.seam_lines(split, centre, reach)
    rec = units.SeamRecorder(t, (n, n), xs, ys)
    whole = simulate(Surface(z=z), rain, cfg, inflow=None, sink=rec, verbose=False)
    seams = rec.seams()
    peak = np.full((n, n), np.nan)
    residuals = []
    for u in split:
        r0, r1, c0, c1 = units.cells(t, (n, n), u["window"])
        ut = Affine(dx, 0.0, t.c + c0 * dx, 0.0, -dx, t.f - r0 * dx)
        flow = units.unit_inflow(seams, u["window"], ut, (r1 - r0, c1 - c0))
        res = simulate(units.crop_surface(Surface(z=z), (r0, r1, c0, c1)), rain, cfg, inflow=flow, verbose=False)
        k0, k1, j0, j1 = units.cells(t, (n, n), u["core"])
        peak[k0:k1, j0:j1] = res.h_max[k0 - r0:k1 - r0, j0 - c0:j1 - c0]
        residuals.append(abs(res.mass.residual_pct))
    assert np.isfinite(peak).all()
    seam = np.zeros((n, n), bool)
    for x in (24,):
        seam[:, x - 3:x + 3] = True
        seam[x - 3:x + 3, :] = True
    d = np.abs(peak - whole.h_max)
    print(f"units against whole: away {d[~seam].max():.2e} m, seams {d[seam].max():.2e} m, "
          f"peak {whole.h_max.max():.3f} m, residuals {residuals}")
    assert max(residuals) < 1e-3, residuals
    assert d[~seam].max() < 2e-3 and d[seam].max() < 5e-3, (d[~seam].max(), d[seam].max())
    assert whole.h_max.max() > 0.01, "the storm makes water worth comparing"


def test_a_units_core_reports_its_peak_and_the_rim_inflow_into_it_for_the_joined_receipt():
    """`join` writes the whole run's receipt from its units: the peak depth over the cores, and the rim
    inflow summed over the cores at each forcing interval, as the whole run loads it."""
    from types import SimpleNamespace

    import cli
    h = np.zeros((4, 5))
    h[1, 2], h[3, 4] = 0.3, 0.9                                          # (3, 4) lies outside the core
    flow = Inflow(times_s=np.array([0.0, 120.0]), west=np.zeros((2, 4)), east=np.zeros((2, 4)),
                  north=np.zeros((2, 5)), south=np.zeros((2, 5)), rim_index=np.array([2, 7, 19]),
                  rim=np.array([[1e-3, 2e-3, 5e-3], [3e-3, 4e-3, 5e-3]]))
    out = cli._core(SimpleNamespace(h_max=h), flow, (0, 3, 0, 4), 3, 60.0, 0.5)
    assert out["core_peak_depth_m"] == 0.3
    assert np.allclose(out["core_rim_inflow_cms"], [0.25 * 3e-3, 0.25 * 5e-3, 0.25 * 7e-3]), "cells 2 and 7 only"


def test_the_inflow_merges_a_units_edges_with_the_rim_inside_it():
    edges = Inflow.uniform((3, 2), {"west": 0.1}, 120.0)
    both = units.combine(edges, np.array([1, 4]), np.array([[1e-5, 2e-5], [3e-5, 4e-5]]), np.array([0.0, 60.0]))
    assert list(both.times_s) == [0.0, 60.0, 120.0]
    assert np.allclose(both.at(30.0)["rim"], [2e-5, 3e-5]) and np.allclose(both.at(90.0)["west"], 0.1)
    rim = Inflow(times_s=np.array([0.0]), west=np.zeros((1, 4)), east=np.zeros((1, 4)), north=np.zeros((1, 4)),
                 south=np.zeros((1, 4)), rim_index=np.array([0, 5, 15]), rim=np.array([[1.0, 2.0, 3.0]]))
    idx, rates = units.crop_rim(rim, (4, 4), (1, 4, 1, 3))
    assert list(idx) == [0] and list(rates[0]) == [2.0], "cell (1, 1) is the window's first; (3, 3) lies outside"
    assert units.crop_rim(None, (4, 4), (0, 4, 0, 4)) == (None, None)
