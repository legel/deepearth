"""Geographic units of the wind solve: a unit whose boundary is the profile is the single solve
itself; the split covers the grid once; and units forced across their seams by a coarser solve of the
whole domain, joined and projected once, follow the single solve and are divergence-free. No network,
no site data."""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import checks
import cli
import domain
from domain import Grid
from solver import Boundary, Model, SolverConfig, solve

FAST = SolverConfig(steps=40, tol=1e-6)


def test_a_boundary_that_is_the_profile_reproduces_the_single_solve_byte_for_byte():
    g = checks.grid(1.0, 48, 24, 20)
    scene = domain.cube(g, 8.0, centre=(16.0, 12.0))
    p = checks.profile()
    single = solve(scene, p, checks.WEST, FAST)
    assert np.array_equal(solve(scene, p, checks.WEST, FAST, boundary=Boundary()).velocity, single.velocity)
    m = Model(scene, p, checks.WEST, FAST)
    u0 = m.u0.cpu().numpy()
    given = Boundary(west=np.repeat(u0[:, :, None], g.ny, axis=2), south=np.repeat(u0[:, :, None], g.nx, axis=2),
                     top=np.broadcast_to(m.u0_top.cpu().numpy()[:, None, None], (3, g.ny, g.nx)).copy())
    assert np.array_equal(solve(scene, p, checks.WEST, FAST, boundary=given).velocity, single.velocity), \
        "the profile given explicitly on sides and top"


def test_the_split_covers_the_grid_once_and_each_window_holds_its_core():
    scene = domain.flat(checks.grid(2.0, 50, 30, 8))
    specs = cli._split(scene, 3, 2, 10.0)
    seen = np.zeros((30, 50), int)
    for s in specs:
        r0, r1, c0, c1 = s["core"]
        w0, w1, v0, v1 = s["window"]
        seen[r0:r1, c0:c1] += 1
        assert w0 == max(r0 - 5, 0) and w1 == min(r1 + 5, 30) and v0 == max(c0 - 5, 0) and v1 == min(c1 + 5, 50)
    assert (seen == 1).all() and len(specs) == 6
    unit = cli._crop(scene, specs[4]["window"])
    assert unit.grid.shape == (8, specs[4]["window"][1] - specs[4]["window"][0],
                               specs[4]["window"][3] - specs[4]["window"][2])
    assert unit.origin[0] == scene.origin[0] + specs[4]["window"][2] * 2.0


def test_units_forced_by_a_coarse_solve_join_to_the_single_solve_divergence_free():
    """2 x 2 units at 1 m with an 8 m overlap, forced across their seams by a 2 m solve of the whole
    domain, a tower in the middle of the seams: joined, projected once, against the single 1 m solve."""
    fine = checks.grid(1.0, 64, 32, 24)
    coarse_g = Grid(dx=2.0, nx=32, ny=16, zf=fine.zf)
    centre = (32.0, 16.0)
    scene, coarse = domain.box(fine, 8.0, 10.0, centre=centre), domain.box(coarse_g, 8.0, 10.0, centre=centre)
    p = checks.profile()
    cfg = SolverConfig(steps=60, tol=1e-6, tol_final=1e-8)
    single = solve(scene, p, checks.WEST, cfg)
    c = solve(coarse, p, checks.WEST, cfg)
    specs = cli._split(scene, 2, 2, 8.0)
    parts = [(s, cli._unit_solve(scene, coarse, c.velocity, s, p, checks.WEST, cfg)[0]) for s in specs]
    vel, _, stats = cli._join_project(scene, p, checks.WEST, cfg, parts)
    axes = (coarse.grid.zc, coarse.origin[1] + coarse.grid.yc, coarse.origin[0] + coarse.grid.xc)
    pts = np.stack(np.meshgrid(fine.zc, scene.origin[1] + fine.yc, scene.origin[0] + fine.xc, indexing="ij"), -1)
    coarse_on_fine = np.stack([RegularGridInterpolator(axes, c.velocity[k], bounds_error=False, fill_value=None)(pts)
                               for k in range(3)])
    u_h = float(p.speed(np.array([10.0]))[0])
    fluid = ~scene.solid
    d = np.linalg.norm(vel - single.velocity, axis=0)[fluid] / u_h
    dc = np.linalg.norm(coarse_on_fine - single.velocity, axis=0)[fluid] / u_h
    rms, p99, rms_c = float(np.sqrt(np.mean(d ** 2))), float(np.percentile(d, 99)), float(np.sqrt(np.mean(dc ** 2)))
    print(f"units against single: rms {rms:.4f}, p99 {p99:.4f}, max {d.max():.4f} of u_h; coarse alone rms "
          f"{rms_c:.4f}; divergence {stats['divergence_max_1_s']:.2e} 1/s (single {single.divergence_max_1_s:.2e}), "
          f"seams before the projection {stats['seam_divergence_max_1_s']:.2e}")
    assert stats["divergence_max_1_s"] <= max(10.0 * single.divergence_max_1_s, 1e-6)
    assert stats["seam_divergence_max_1_s"] > stats["divergence_max_1_s"], "the projection closes the seams"
    assert rms < 0.5 * rms_c, "the units resolve what the coarse solve cannot"
    assert rms < 0.05 and p99 < 0.2


def test_blocks_keep_their_cores_levels_and_follow_the_single_solve_without_a_join(tmp_path):
    """Blocks (2026-09-24: no site outgrows a GPU): each unit's core levels come from its own window's solve on
    the published grid, its buffer discarded, with no whole-site field and no projection across a seam. Every published
    cell belongs to exactly one core, and the levels follow the single solve's."""
    import json

    import view

    fine = checks.grid(1.0, 64, 32, 24)
    coarse_g = Grid(dx=2.0, nx=32, ny=16, zf=fine.zf)
    centre = (32.0, 16.0)
    scene, coarse = domain.box(fine, 8.0, 10.0, centre=centre), domain.box(coarse_g, 8.0, 10.0, centre=centre)
    p = checks.profile()
    cfg = SolverConfig(steps=60, tol=1e-6, tol_final=1e-8)
    single = solve(scene, p, checks.WEST, cfg)
    c = solve(coarse, p, checks.WEST, cfg)
    vcfg = view.ViewConfig(heights_m=(4.0, 10.0), cell_m=1.0, half_m=15.0)
    ref, ref_solid, _ = view.levels(np.concatenate([single.velocity[:2], single.vorticity[2][None]]), scene, vcfg, 15.0)
    specs = cli._split(scene, 2, 2, 8.0)
    for i, s in enumerate(specs):
        _, res = cli._unit_solve(scene, coarse, c.velocity, s, p, checks.WEST, cfg)
        lv, solid, filled, rect = cli._block_levels(scene, s, res, vcfg, 15.0)
        np.savez(tmp_path / f"unit_270.tile{i}of2x2.npz", levels=lv, solid=solid, filled=filled, rect=np.asarray(rect),
                 stats=json.dumps(cli._stats(res)))
    lv, solid, filled, stats = cli._blocks(scene, vcfg, 270.0, tmp_path, "2x2", specs)
    assert lv.shape == ref.shape and np.array_equal(solid, ref_solid), "the tower's solid cells, block by block"
    assert np.array_equal(np.isnan(lv), np.isnan(ref)), "finite exactly where the single solve's levels are"
    ok = np.isfinite(ref[:2])
    u_h = float(p.speed(np.array([10.0]))[0])
    d = np.abs(lv[:2][ok] - ref[:2][ok]) / u_h
    print(f"blocks against single: rms {np.sqrt(np.mean(d ** 2)):.4f}, max {d.max():.4f} of u_h; "
          f"worst block divergence {stats['divergence_max_1_s']:.2e} 1/s")
    assert float(np.sqrt(np.mean(d ** 2))) < 0.05 and len(stats["unit_stats"]) == 4
    assert stats["divergence_max_1_s"] <= max(10.0 * single.divergence_max_1_s, 1e-6)
