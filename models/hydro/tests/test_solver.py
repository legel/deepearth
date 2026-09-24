"""Solver invariants on synthetic domains, in physical units. No network, no site data."""

import numpy as np
import pytest

from inflow import Inflow
from typing import Tuple

from physics import FROUDE_CAP, G, MIN_DEPTH
from solver import FIELDS, MassBalance, Probes, Result, SolverConfig, Surface, horton_rate, simulate

MM_HR = 1.0 / 1000.0 / 3600.0


def plane(nrows: int = 40, ncols: int = 40, dx: float = 5.0, slope: float = 0.002,
          walls: bool = False) -> np.ndarray:
    """A plane falling southward; `walls` raises the west and east columns 100 m."""
    rows = np.arange(nrows, dtype=np.float64)[:, None]
    z = np.repeat(20.0 - rows * dx * slope, ncols, axis=1)
    if walls:
        z[:, 0] = z[:, -1] = z[:, 1] + 100.0
    return z.astype(np.float32)


def run(z: np.ndarray, rain, dt_s: float = 60.0, dx: float = 5.0, dtype: str = "float64",
        **kw) -> Result:
    """Run quietly on the CPU; Surface fields come from `kw`, the rest configures the solver."""
    fields = [k for k in kw if k in Surface.__dataclass_fields__
              and not (k == "manning_n" and np.isscalar(kw[k]))]
    surface = Surface(z=z, **{k: kw.pop(k) for k in fields})
    inflow, probes = kw.pop("inflow", None), kw.pop("probes", Probes())
    kw.setdefault("frame_interval_min", 10.0)
    cfg = SolverConfig(dx=dx, dt_s=dt_s, dtype=dtype, device="cpu", **kw)
    return simulate(surface, rain, cfg, probes, inflow=inflow, verbose=False)


def manning_depth(q: float, n: float, slope: float) -> float:
    """Normal depth [m] for unit discharge q on a slope."""
    return (q * n / np.sqrt(slope)) ** 0.6


def closure(mass: MassBalance) -> float:
    """|in - out| relative to everything that entered."""
    out = mass.infiltrated + mass.abstracted + mass.stored + mass.outflow
    return abs(mass.supplied - out) / mass.supplied


def test_mass_balance_closes_to_1e6_relative():
    z = plane()
    rain = [80.0 * MM_HR] * 60 + [0.0] * 120
    res = run(z, rain, f0=25.0 * MM_HR, fc=10.0 * MM_HR, k=2.0 / 3600.0,
              max_deficit_m=np.full(z.shape, 0.02, dtype=np.float32),
              smax_m=np.full(z.shape, 0.002, dtype=np.float32),
              inflow=Inflow.uniform(z.shape, {"north": 0.004}, 180 * 60.0))
    assert res.mass.inflow > 0 and res.mass.infiltrated > 0 and res.mass.abstracted > 0
    assert closure(res.mass) < 1e-6, res.mass


def test_mass_balance_closes_in_float32():
    z = plane()
    rain = [80.0 * MM_HR] * 60 + [0.0] * 120
    res = run(z, rain, dtype="float32", f0=25.0 * MM_HR, fc=10.0 * MM_HR, k=2.0 / 3600.0,
              max_deficit_m=np.full(z.shape, 0.02, dtype=np.float32))
    assert closure(res.mass) < 2e-5, res.mass


def test_infiltrated_plus_abstracted_plus_stored_plus_outflow_equals_input():
    z = plane()
    res = run(z, [60.0 * MM_HR] * 90, f0=20.0 * MM_HR, fc=20.0 * MM_HR, k=0.0,
              max_deficit_m=np.full(z.shape, 0.01, dtype=np.float32),
              smax_m=np.full(z.shape, 0.003, dtype=np.float32),
              initial_h=np.full(z.shape, 0.005, dtype=np.float32),
              inflow=Inflow.uniform(z.shape, {"west": 0.002, "north": 0.001}, 90 * 60.0))
    m = res.mass
    total_out = m.infiltrated + m.abstracted + m.stored + m.outflow
    assert total_out == pytest.approx(m.rain + m.initial + m.inflow, rel=1e-6)
    assert m.abstracted == pytest.approx(0.003 * 40 * 40 * 25.0, rel=1e-6)
    assert m.infiltrated == pytest.approx(0.01 * 40 * 40 * 25.0, rel=1e-6)


def test_no_negative_depth_ever():
    z = plane(slope=0.05)
    z[15:20, 10:20] -= 0.5
    rain = [400.0 * MM_HR] * 30 + [0.0] * 30
    res = run(z, rain, dtype="float32", frame_interval_min=1.0)
    assert res.h_final.min() >= 0.0 and res.h_max.min() >= 0.0
    assert min(float(f[0].min()) for f in res.frames) >= 0.0
    assert all(np.isfinite(f).all() for f in res.frames)


def walled(z: np.ndarray) -> np.ndarray:
    """Raise the outer ring 100 m so nothing can leave."""
    z = z.copy()
    z[0, :] = z[-1, :] = z[:, 0] = z[:, -1] = np.nanmax(z) + 100.0
    return z


def test_water_at_rest_on_flat_ground_stays_at_rest():
    z = walled(np.full((30, 30), 10.0, dtype=np.float32))
    h0 = np.where(z < 50.0, 0.1, 0.0).astype(np.float32)
    res = run(z, [0.0] * 60, dtype="float32", initial_h=h0)
    assert np.abs(res.h_final - h0).max() == 0.0
    assert np.abs(res.u_final).max() == 0.0 and np.abs(res.v_final).max() == 0.0
    assert res.mass.outflow == 0.0


def test_lake_at_rest_over_an_uneven_bed_stays_at_rest():
    rng = np.random.default_rng(1)
    z = walled((10.0 + rng.random((30, 30))).astype(np.float32))
    h0 = np.where(z < 50.0, 12.0 - z, 0.0).astype(np.float32)
    res = run(z, [0.0] * 60, dtype="float32", initial_h=h0)
    assert np.abs(res.h_final - h0).max() == 0.0
    assert np.abs(res.qx_final).max() == 0.0 and np.abs(res.qy_final).max() == 0.0


def flume(nrows: int, dx: float, slope: float) -> Tuple[np.ndarray, np.ndarray]:
    """A walled plane whose last row is nodata: an open brink at the downstream end.

    Returns:
        (elevation, infiltration capacity [m/s] that swallows the rain landing on the walls).
    """
    z = plane(nrows=nrows, ncols=10, dx=dx, slope=slope, walls=True)
    z[-1, :] = np.nan
    return z, np.where(z > 50.0, 1.0, 0.0).astype(np.float32)


def test_uniform_slope_reaches_manning_steady_state():
    slope, n, P, dx = 0.01, 0.03, 50.0 * MM_HR, 5.0
    z, walls = flume(60, dx, slope)
    res = run(z, [P] * 120, dx=dx, manning_n=n, f0=walls, fc=walls, k=np.zeros_like(walls))
    rows = np.arange(10, 54)
    got = res.h_final[rows, 5]
    want = manning_depth(P * (rows + 1) * dx, n, slope)
    assert np.abs(got / want - 1.0).max() < 0.01, np.abs(got / want - 1.0).max()
    assert closure(res.mass) < 1e-6


def test_uniform_inflow_on_a_plane_matches_manning_exactly():
    slope, n, q, dx = 0.01, 0.03, 0.02, 5.0
    z, _ = flume(60, dx, slope)
    inflow = Inflow.uniform(z.shape, {"north": q}, 3600.0)
    inflow.north[:, 0] = inflow.north[:, -1] = 0.0
    res = run(z, [0.0] * 60, dx=dx, manning_n=n, inflow=inflow)
    want = manning_depth(q, n, slope)
    got = res.h_final[5:-6, 1:-1]
    assert np.abs(got / want - 1.0).max() < 1e-4, np.abs(got / want - 1.0).max()
    assert res.mass.inflow == pytest.approx(q * 8 * dx * 3600.0, rel=1e-9)
    assert res.series["outflow_total_cms"][-1] == pytest.approx(q * 8 * dx, rel=1e-3)
    assert closure(res.mass) < 1e-6


def test_pit_fills_to_its_rim_then_spills():
    slope, q, dx = 0.01, 0.01, 5.0
    z, _ = flume(40, dx, slope)
    rim = float(z[33, 5])
    z[30:33, 1:-1] = rim - 0.3
    inflow = Inflow.uniform(z.shape, {"north": q}, 7200.0)
    inflow.north[:, 0] = inflow.north[:, -1] = 0.0
    res = run(z, [0.0] * 120, dx=dx, inflow=inflow, frame_interval_min=1.0)
    capacity = 0.3 * 3 * 8 * dx * dx
    fill_time_s = capacity / (q * 8 * dx)
    outflow = res.series["outflow_total_cms"]
    t_min = np.arange(1, 121)
    assert outflow[t_min * 60 < 0.9 * fill_time_s].max() == 0.0
    assert outflow[t_min * 60 > 3 * fill_time_s].min() > 0.0
    assert outflow[-1] == pytest.approx(q * 8 * dx, rel=0.02)
    eta = res.h_final[30:33, 1:-1] + z[30:33, 1:-1]
    assert eta.min() >= rim - 1e-3 and eta.max() <= rim + 0.1
    assert closure(res.mass) < 1e-6


def test_velocity_never_exceeds_the_froude_cap():
    z = plane(slope=0.2)
    rain = [300.0 * MM_HR] * 40
    res = run(z, rain, dtype="float32", manning_n=0.01)
    eta = z + res.h_final
    hf_x = np.maximum(np.maximum(eta[:, 1:], eta[:, :-1]) - np.maximum(z[:, 1:], z[:, :-1]), 0.0)
    hf_y = np.maximum(np.maximum(eta[1:, :], eta[:-1, :]) - np.maximum(z[1:, :], z[:-1, :]), 0.0)
    froude = []
    for q, hf in ((res.qx_final[:, 1:-1], hf_x), (res.qy_final[1:-1, :], hf_y)):
        wet = hf > MIN_DEPTH
        froude.append(np.abs(q[wet]) / hf[wet] / np.sqrt(G * hf[wet]))
        assert np.abs(np.where(wet, 0.0, q)).max() == 0.0
    fr = np.concatenate(froude)
    assert fr.max() <= FROUDE_CAP * 1.01, fr.max()
    assert fr.max() > 0.85 * FROUDE_CAP


def test_face_cfl_ignores_water_trapped_in_single_cell_pits():
    z = plane(slope=0.005)
    pits = [(10, 10), (20, 25), (30, 15)]
    h0 = np.zeros(z.shape, dtype=np.float32)
    for r, c in pits:
        z[r, c] -= 2.0
        h0[r, c] = 2.0
    rain = [40.0 * MM_HR] * 30
    cell = run(z, rain, dtype="float32", initial_h=h0, cfl_depth="cell")
    face = run(z, rain, dtype="float32", initial_h=h0, cfl_depth="face")
    assert face.n_substeps < cell.n_substeps / 3, (face.n_substeps, cell.n_substeps)
    assert closure(face.mass) < 2e-5 and closure(cell.mass) < 2e-5
    assert np.abs(face.h_final - cell.h_final).max() < 1e-3
    assert face.h_final.min() >= 0.0


def test_nodata_cells_are_open_brinks_that_drain_the_domain():
    z = plane(slope=0.0)
    z[15:25, 15:25] = np.nan
    h0 = np.where(np.isfinite(z), 0.05, 0.0).astype(np.float32)
    res = run(z, [0.0] * 60, initial_h=h0)
    assert res.mass.outflow > 0.5 * res.mass.initial
    assert res.h_final[15:25, 15:25].max() == 0.0
    assert closure(res.mass) < 1e-6


def test_frames_carry_seconds_and_three_fields():
    res = run(plane(), [50.0 * MM_HR] * 25, frame_interval_min=10.0)
    assert res.frame_times_s == [60.0, 660.0, 1260.0, 1500.0]
    assert len(FIELDS) == 3 and all(f.shape == (3, 40, 40) for f in res.frames)
    assert np.array_equal(res.frames[-1][0], res.h_final.astype(np.float32))
    assert np.array_equal(res.frames[-1][1], res.u_final.astype(np.float32))


def test_sink_receives_every_frame_and_nothing_is_kept():
    got = []
    res = simulate(Surface(z=plane()), [50.0 * MM_HR] * 25,
                   SolverConfig(dx=5.0, dt_s=60.0, frame_interval_min=10.0, device="cpu"),
                   sink=lambda t, f: got.append((t, f.copy())), verbose=False)
    assert [t for t, _ in got] == res.frame_times_s and res.frames == []
    assert np.array_equal(got[-1][1][0], res.h_final)


def test_final_frame_lands_on_the_end_of_the_run():
    res = run(plane(), [50.0 * MM_HR] * 3, frame_interval_min=1e6)
    assert res.frame_times_s[-1] == pytest.approx(180.0)
    assert np.array_equal(res.frames[-1][0], res.h_final.astype(np.float32))


def test_nodata_cells_never_hold_water():
    z = plane()
    z[0:5, 0:5] = np.nan
    res = run(z, [80.0 * MM_HR] * 60)
    assert res.h_final[0:5, 0:5].max() == 0.0 and res.cum_infil[0:5, 0:5].max() == 0.0
    assert closure(res.mass) < 1e-6


def test_no_rain_produces_no_water():
    res = run(plane(), [0.0] * 30)
    assert res.h_max.max() == 0.0 and res.mass.outflow == 0.0


def test_substepping_delivers_the_whole_storm():
    dt_s, n, rate = 60.0, 90, 50.0 * MM_HR
    res = run(plane(), [rate] * n, dt_s=dt_s)
    assert res.mass.rain == pytest.approx(rate * dt_s * n * 40 * 40 * 25.0, rel=1e-9)
    assert res.n_substeps > n


def test_water_drains_rather_than_ponding_forever():
    res = run(plane(), [60.0 * MM_HR] * 60 + [0.0] * 600)
    assert res.mass.stored / res.mass.rain < 0.10
    assert res.mass.outflow / res.mass.rain > 0.85


def test_storage_cap_bounds_infiltration():
    z, cap = plane(), 0.01
    res = run(z, [100.0 * MM_HR] * 240, f0=200.0 * MM_HR, fc=200.0 * MM_HR, k=0.0,
              max_deficit_m=np.full(z.shape, cap, dtype=np.float32))
    assert res.cum_infil.max() <= cap + 1e-9 and res.cum_infil.max() == pytest.approx(cap, rel=1e-3)


def test_scalar_and_uniform_manning_agree():
    z, rain = plane(), [60.0 * MM_HR] * 60
    a = run(z, rain, dtype="float32")
    b = run(z, rain, dtype="float32", manning_n=np.full(z.shape, 0.040))
    assert np.abs(a.h_final - b.h_final).max() < 1e-5


def test_float32_and_float64_agree():
    z, rain = plane(), [60.0 * MM_HR] * 60 + [0.0] * 30
    a = run(z, rain, dtype="float32")
    b = run(z, rain, dtype="float64")
    assert np.abs(a.h_final - b.h_final).max() < 1e-4
    assert a.mass.outflow == pytest.approx(b.mass.outflow, rel=1e-4)


def test_horton_rate_decays_from_f0_to_fc():
    f0, fc, k = 76.0 * MM_HR, 25.0 * MM_HR, 2.0 / 3600.0
    assert horton_rate(0.0, f0, fc, k) == pytest.approx(f0)
    assert horton_rate(1e9, f0, fc, k) == pytest.approx(fc)


def test_gauge_and_watershed_probes_report_flow():
    z = plane()
    mask = np.zeros(z.shape, dtype=bool)
    mask[:20, :] = True
    res = run(z, [60.0 * MM_HR] * 90, probes=Probes(gauge_rc=(30, 20), watershed_mask=mask))
    assert len(res.series["watershed_outflow_cms"]) == 90 and res.series["watershed_outflow_cms"].max() > 0
    assert res.series["gauge_cms"].max() == pytest.approx(
        res.series["outflow_total_cms"].max() * 31 / 40 / 40, rel=0.1)


def test_mass_balance_residual_is_zero_when_nothing_supplied():
    mass = MassBalance(rain=0.0, initial=0.0, inflow=0.0, created=0.0, infiltrated=0.0, abstracted=0.0,
                       stored=0.0, outflow=0.0)
    assert mass.residual == 0.0


def test_inflow_onto_nodata_is_counted_as_leaving_not_lost():
    """The disc-parcel bug: water prescribed onto ground the raster does not cover.

    A nodata border zeroes whatever reaches it. If that volume is not measured it vanishes
    from the mass balance, which is how a 49.6 % residual reached a run receipt.
    """
    z = plane(nrows=30, ncols=30, slope=0.002)
    z[:4, :] = z[-4:, :] = z[:, :4] = z[:, -4:] = np.nan
    inflow = Inflow.uniform(z.shape, {"north": 0.01, "west": 0.01}, 1800.0)
    res = run(z, [0.0] * 30, inflow=inflow)
    assert res.mass.inflow == pytest.approx((0.01 * 30 + 0.01 * 30) * 5.0 * 1800.0, rel=1e-9)
    assert res.mass.outflow == pytest.approx(res.mass.inflow, rel=1e-3)
    assert closure(res.mass) < 1e-6, res.mass


def test_a_fully_masked_domain_loses_nothing():
    """Every cell nodata: all rain is zeroed, and all of it is accounted as leaving."""
    z = np.full((10, 10), np.nan, dtype=np.float32)
    res = run(z, [50.0 * MM_HR] * 10)
    assert res.mass.rain == 0.0 and res.mass.outflow == 0.0 and res.mass.residual == 0.0


def test_the_positivity_clamp_reports_the_volume_it_invents():
    """A cell that would go negative in one sub-step is clamped; that volume is measured."""
    z = plane(slope=0.05)
    res = run(z, [200.0 * MM_HR] * 30, dtype="float32", cfl_alpha=0.7)
    assert res.mass.created >= 0.0
    assert closure(res.mass) < 2e-5, res.mass
    tame = run(z, [200.0 * MM_HR] * 30, dtype="float32", cfl_alpha=0.15)
    assert tame.mass.created <= res.mass.created


def test_progress_lines_say_each_interval_and_change_no_number(capsys, monkeypatch):
    """A caller reads `PROGRESS hydro k/n` for its progress: one line per
    forcing interval while `solver.PROGRESS_LINES` is set, as the command line sets it, the run untouched."""
    import solver

    quiet = run(plane(), [50.0 * MM_HR] * 5, frame_interval_min=2.0)
    assert "PROGRESS" not in capsys.readouterr().out
    monkeypatch.setattr(solver, "PROGRESS_LINES", True)
    told = run(plane(), [50.0 * MM_HR] * 5, frame_interval_min=2.0)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("PROGRESS")]
    assert lines == [f"PROGRESS hydro {k}/5" for k in range(1, 6)]
    assert all(np.array_equal(a, b) for a, b in zip(quiet.frames, told.frames)) and len(quiet.frames) == len(told.frames)
    assert np.array_equal(quiet.h_final, told.h_final) and quiet.frame_times_s == told.frame_times_s


def test_the_command_line_turns_the_progress_lines_on(monkeypatch):
    import cli
    import solver

    monkeypatch.setattr(solver, "PROGRESS_LINES", False)
    seen = []
    monkeypatch.setattr(cli, "cmd_segment", lambda args: seen.append(solver.PROGRESS_LINES))
    try:
        cli.main(["segment", "--site", "campanile"])
    except SystemExit:
        pass
    assert seen == [True], seen
