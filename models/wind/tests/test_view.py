"""Levels above the bare earth: the sampler, the solid and filled masks, and where the solve holds no value."""

import numpy as np

import domain
import view
from domain import Grid


def test_trilinear_is_exact_for_a_linear_field_on_a_stretched_grid():
    g = Grid.stretched(2.0, 24, 20, 16, 2.0, 1.06)
    scene = domain.flat(g)
    zc = scene.origin[2] + g.zc
    yc, xc = scene.origin[1] + g.yc, scene.origin[0] + g.xc
    field = (0.3 * zc[:, None, None] - 0.2 * yc[None, :, None] + 0.1 * xc[None, None, :])[None]
    rng = np.random.default_rng(0)
    x, y = rng.uniform(xc[0], xc[-1], 200), rng.uniform(yc[0], yc[-1], 200)
    z = rng.uniform(zc[0], zc[-1], 200)
    out, ok = view.trilinear(field, scene, x, y, z)
    assert ok.all()
    np.testing.assert_allclose(out[0], 0.3 * z - 0.2 * y + 0.1 * x, atol=1e-12)


def test_the_walkable_top_is_the_roof_on_a_building_and_the_ground_beside_it():
    g = Grid.uniform(2.0, 40, 40, 20)
    scene = domain.box(g, 10.0, 13.5)
    cfg = view.ViewConfig(heights_m=(1.0, 5.0), cell_m=2.0, half_m=30.0)
    top, roof = view.surface(scene, cfg), view.roofs(scene, cfg)
    assert roof.sum() == 36 and np.all(top[roof] == 13.5) and np.all(top[~roof] == 0.0)
    above, dz = (view.under(scene, cfg, a) for a in view.first_fluid(scene))
    np.testing.assert_allclose(above[roof], 1.5)
    np.testing.assert_allclose(above[~roof], 1.0)
    assert np.all(dz == 2.0)


def test_a_level_passes_through_a_building_taller_than_it_and_is_finite_everywhere_else():
    g = Grid.uniform(2.0, 40, 40, 20)
    scene = domain.box(g, 10.0, 13.5)
    cfg = view.ViewConfig(heights_m=(1.0, 5.0, 25.0), cell_m=2.0, half_m=30.0)
    out, solid, filled = view.levels(np.ones((1,) + g.shape), scene, cfg, display_radius=25.0)
    roof = view.roofs(scene, cfg)
    x, y = cfg.centres()
    disc = np.hypot(x, y) <= 25.0
    assert roof.sum() == 36
    for j in (0, 1):
        assert np.array_equal(solid[j], roof & disc), "13.5 m of building stands through the 1 m and 5 m levels"
    assert not solid[2].any(), "nothing pierces 25 m"
    for j in range(3):
        fluid = disc & ~solid[j]
        np.testing.assert_allclose(out[0, j][fluid], 1.0)
        assert np.isnan(out[0, j][solid[j]]).all() and np.isnan(out[0, j][~disc]).all()
    assert not filled.any(), "the first centre is 1 m up on a 2 m grid, at or below every level"


def _site(stretch: float = 1.06):
    """A sloping lot on a stretched grid: a 12 m hall, a 40 m tower, a tree and open pavement."""
    n, dx = 48, 2.0
    g = Grid.stretched(dx, n, n, 32, dx, stretch)
    X, Y = np.meshgrid(np.arange(n), np.arange(n))
    dtm = 100.0 + 0.4 * X + 0.15 * Y                      # 19 m and 7 m of fall across the lot
    dsm, classes = dtm.copy(), np.full((n, n), 3)
    classes[10:20, 8:22] = 1
    dsm[10:20, 8:22] += 12.0
    classes[30:34, 30:34] = 1
    dsm[30:34, 30:34] += 40.0
    classes[24:30, 12:18] = 2
    dsm[24:30, 12:18] += 9.0
    legend = {1: "facade_masonry", 2: "tree_canopy", 3: "asphalt_pavement"}
    scene = domain.from_rasters(dtm, dsm, classes, legend, g)
    scene.origin = (-n * dx / 2, -n * dx / 2, scene.origin[2])
    return scene, dtm, dsm, classes == 1


def test_every_fluid_cell_is_finite_and_the_solid_mask_is_the_structure_against_the_level():
    scene, dtm, dsm, building = _site()
    rng = np.random.default_rng(3)
    field = rng.normal(size=(3,) + scene.grid.shape)
    cfg = view.ViewConfig(heights_m=(4.0, 5.0, 10.0, 25.0), cell_m=2.0, half_m=46.0)
    out, solid, filled = view.levels(field, scene, cfg, display_radius=44.0)
    x, y = cfg.centres()
    disc = np.hypot(x, y) <= 44.0
    for j, h in enumerate(cfg.heights_m):
        want = view.under(scene, cfg, building & (dsm - dtm > h)) & disc
        assert np.array_equal(solid[j], want), f"{h} m"
        fluid = disc & ~solid[j]
        assert fluid.sum() > 0 and np.isfinite(out[:, j][:, fluid]).all(), f"{h} m: every fluid cell is finite"
        assert np.isnan(out[:, j][:, solid[j]]).all()
    assert solid[0].sum() > solid[3].sum() > 0, "the hall stands through 4 m, only the tower through 25 m"
    assert not (filled & solid).any()


def test_on_open_ground_the_level_is_the_trilinear_sample_it_always_was():
    scene, dtm, dsm, building = _site()
    rng = np.random.default_rng(4)
    field = rng.normal(size=(2,) + scene.grid.shape)
    cfg = view.ViewConfig(heights_m=(4.0, 10.0), cell_m=2.0, half_m=46.0)
    out, solid, filled = view.levels(field, scene, cfg, display_radius=44.0)
    x, y = cfg.centres()
    top = view.surface(scene, cfg)
    open_ = (np.hypot(x, y) <= 44.0) & ~view.under(scene, cfg, dsm > dtm)
    for j, h in enumerate(cfg.heights_m):
        old, ok = view.trilinear(field, scene, x.ravel(), y.ravel(), (top + h).ravel())
        ok = ok.reshape(x.shape) & open_
        assert ok.sum() > 100
        np.testing.assert_allclose(out[:, j][:, ok], old.reshape((-1,) + x.shape)[:, ok], atol=1e-12)


def test_below_the_lowest_fluid_centre_the_log_law_carries_that_centre_down():
    g = Grid.uniform(4.0, 20, 20, 10)
    scene = domain.flat(g, z0=0.1)
    cfg = view.ViewConfig(heights_m=(1.0, 3.0), cell_m=4.0, half_m=36.0)
    field = np.broadcast_to(g.zc[:, None, None], g.shape)[None].astype(float)
    out, solid, filled = view.levels(field, scene, cfg, display_radius=30.0)
    disc = np.hypot(*cfg.centres()) <= 30.0
    assert filled[0][disc].all() and not filled[1].any(), "the first centre is 2 m up"
    np.testing.assert_allclose(out[0, 0][disc], 2.0 * np.log1p(1.0 / 0.1) / np.log1p(2.0 / 0.1))
    np.testing.assert_allclose(out[0, 1][disc], 3.0)
    assert float(view.reroot(np.array(5.0), np.array(2.0), np.array(0.0), np.array(0.1))) == 0.0


def test_a_disc_with_no_roof_reads_its_resolution():
    """WS22's F15: a beach lot with no building in its display disc crashed `levels` here."""
    g = Grid.uniform(2.0, 40, 40, 20)
    scene = domain.flat(g)
    cfg = view.ViewConfig(heights_m=(1.0, 5.0), cell_m=2.0, half_m=30.0)
    table = view.resolution(scene, cfg, 25.0, [1.0, 5.0])
    assert table["roof"] == {"cells": 0, "first_centre_above_top_m": {}, "first_cell_dz_m": {}}
    assert table["ground"]["cells"] > 0 and table["ground"]["first_centre_above_top_m"]["max"] == 1.0
    assert table["unresolved_cells_by_height_m"]["1"] == {"roof": 0, "ground": 0}


def test_k_shares_of_the_headings_solve_each_once():
    """`levels --part i/k`: k processes, one per GPU, solve every heading between them, none twice."""
    hs = [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0]
    parts = [view.share(hs, f"{i}/3") for i in range(3)]
    assert sorted(h for p in parts for h in p) == hs and sum(map(len, parts)) == len(hs)
    assert view.share(hs, "0/1") == hs and view.share(hs, "3/4") == [67.5]


def test_a_banded_grid_is_uniform_to_the_band_then_stretches_to_the_top():
    g = Grid.banded(2.0, 8, 8, 1.0, 100.0, 1.06, 309.5)
    dz = g.dz
    assert np.all(dz[:100] == 1.0) and np.allclose(dz[101:] / dz[100:-1], 1.06)
    assert g.top >= 309.5 and g.nz % 8 == 0 and g.nz == 144


def test_a_level_beside_a_taller_column_is_flagged_as_beside_a_wall():
    g = Grid.uniform(2.0, 40, 40, 20)
    scene = domain.box(g, 10.0, 13.5)
    wall = view.beside_wall(scene, 1.0)
    fp = scene.roof
    ring = ~fp & (np.pad(fp, 1)[:-2, 1:-1] | np.pad(fp, 1)[2:, 1:-1] | np.pad(fp, 1)[1:-1, :-2] | np.pad(fp, 1)[1:-1, 2:])
    assert np.array_equal(wall, ring) and not wall[fp].any() and not view.beside_wall(scene, 13.5).any()


def test_a_level_no_fluid_cell_resolves_is_recorded_as_null_not_a_crash():
    """A site's 4 m level (2026-09-13): the blend's per-level percentile of an all-NaN level raised IndexError."""
    import json

    import cli

    assert cli._pct(np.array([np.nan, np.nan]), 50) is None
    assert cli._pct(np.array([]), 99) is None
    assert cli._pct(np.array([1.0, np.nan, 3.0]), 50) == 2.0
    assert json.dumps({"speed_p50_m_s": cli._pct(np.full(4, np.nan), 50)}) == '{"speed_p50_m_s": null}'


def test_the_viewer_square_reaches_every_sites_display_disc():
    """a site's 301.52 m disc published a 240 m square, a fifth of its area, when the square was 120 m for every site
    (2026-09-14): it is now the display radius rounded up to the viewer cell, never under the Campanile's."""
    assert view.half_for(301.52) == 302.0 and view.ViewConfig(half_m=view.half_for(301.52)).n == 302
    assert view.half_for(300.0) == 300.0, "a radius on the grid is not grown a cell"
    assert view.half_for(43.6) == 120.0 and view.half_for(112.32) == 120.0, "no site's square shrinks"
    assert view.half_for(1301.0, cell_m=2.0) == 1302.0


def test_no_level_carries_wind_in_a_sealed_shaft_and_a_courtyard_carries_it_everywhere():
    n, dx = 40, 2.0
    g = Grid.uniform(dx, n, n, 16)
    dtm = np.full((n, n), 100.0)
    dsm, classes = dtm.copy(), np.full((n, n), 3)
    classes[8:32, 8:32] = 1
    dsm[8:32, 8:32] = 112.0                     # a 12 m hall
    classes[12, 12] = 2                         # a crown pixel read inside it
    classes[14:24, 18:28] = 3
    dsm[14:24, 18:28] = 100.0                   # a 20 m courtyard
    scene = domain.from_rasters(dtm, dsm, classes, {1: "facade_masonry", 2: "tree_canopy", 3: "asphalt_pavement"}, g)
    scene.origin = (-n * dx / 2, -n * dx / 2, scene.origin[2])
    cfg = view.ViewConfig(heights_m=(4.0, 10.0), cell_m=dx, half_m=40.0)
    out, solid, filled = view.levels(np.ones((1,) + g.shape), scene, cfg, display_radius=38.0)
    disc = np.hypot(*cfg.centres()) <= 38.0
    court = view.under(scene, cfg, np.pad(np.ones((10, 10), bool), ((14, n - 24), (18, n - 28))))
    for j in range(2):
        assert not domain.pockets(~solid[j] | ~disc).any(), "no sealed shaft carries wind"
        fluid = disc & ~solid[j]
        assert (court & fluid).sum() == 100 and np.isfinite(out[0, j][fluid]).all()
        assert np.isnan(out[0, j][solid[j]]).all()


def test_the_published_levels_keep_the_solvers_cell():
    """A 1 m solve is shown on 1 m cells (2026-09-24): `levels` sampled every solve onto a 2 m grid, whatever
    its --dx. --view-cell overrides; a 2 m solve's grid is what it always was."""
    import argparse

    import cli
    import sites

    site = sites.get_site("campanile")
    ns = lambda dx, vc=None: argparse.Namespace(dx=dx, view_cell=vc, heights=[4.0, 5.0, 10.0, 25.0])  # noqa: E731
    fine, old = cli.view_config(ns(1.0), site, None), cli.view_config(ns(2.0), site, None)
    assert (fine.cell_m, fine.half_m, fine.n) == (1.0, 120.0, 240)
    assert old == view.ViewConfig(heights_m=(4.0, 5.0, 10.0, 25.0), half_m=view.half_for(site.display_radius_m))
    assert cli.view_config(ns(1.0, 2.0), site, None).cell_m == 2.0
    box = {"display": (-100.5, -40.0, 60.2, 30.0)}
    assert cli.view_config(ns(1.0), site, box) == view.ViewConfig.for_box(box["display"], 1.0, heights_m=(4.0, 5.0, 10.0, 25.0))
    assert cli.view_config(ns(1.0), site, box).nx == 162


def test_a_unit_run_rebuilds_the_coarse_grid_its_coarse_solve_was_made_on():
    """A 1 m unit run forced by a 2 m coarse solve in 2 m cubes rebuilds that coarse grid, not one banded in 1 m."""
    import argparse

    import cli

    ns = lambda **kw: argparse.Namespace(**{"ground_band": [1.0, 30.0, 150.0], "coarse_band_dz": None, **kw})  # noqa: E731
    assert cli._coarse_band(ns()) == (1.0, 30.0, 150.0), "the old runs: the unit's own band"
    assert cli._coarse_band(ns(coarse_band_dz=2.0)) == (2.0, 30.0, 150.0)
    assert cli._coarse_band(ns(ground_band=None, coarse_band_dz=2.0)) is None


def test_the_ribbons_fly_over_the_crowns_and_roofs():
    """The ribbons' field (view.over_top): OVER_TOP_M over each column's measured top, so a depth test hides a ribbon
    only behind something taller than its own column. On the lowest level over the bare earth, Harvard's ribbons ran
    inside the crowns and the page hid almost all of them (2026-10-01)."""
    scene, dtm, dsm, building = _site()
    cfg = view.ViewConfig(heights_m=(4.0,), cell_m=2.0, half_m=40.0)
    v, h = view.over_top(np.ones((2,) + scene.grid.shape), scene, cfg, display_radius=40.0)
    x, y = cfg.centres()
    disc = cfg.shown(x, y, 40.0)
    assert np.isfinite(v[:, disc]).all() and np.isnan(v[:, ~disc]).all() and np.isnan(h[~disc]).all()
    np.testing.assert_allclose(v[:, disc], 1.0)
    raised = view.under(scene, cfg, dsm - dtm)
    roof, tree = view.under(scene, cfg, building) & disc, view.under(scene, cfg, (dsm - dtm > 0) & ~building) & disc
    bare = disc & (raised == 0)
    assert roof.any() and tree.any() and bare.any()
    np.testing.assert_allclose(h[roof], raised[roof] + view.OVER_TOP_M, atol=1e-6)   # the hall's 14 m, the tower's 42
    np.testing.assert_allclose(h[bare], view.OVER_TOP_M, atol=1e-6)
    np.testing.assert_allclose(h[tree], raised[tree] + view.OVER_TOP_M, atol=1e-6)    # the crown's own top, 9 m


def test_the_over_top_file_holds_every_heading_then_the_height(tmp_path):
    import json

    import cli
    cfg = view.ViewConfig(heights_m=(4.0,), cell_m=2.0, half_m=8.0)
    hs = [0.0, 22.5, 45.0]
    ot = {h: (np.full((2, cfg.ny, cfg.nx), i + 1.0), np.full((cfg.ny, cfg.nx), 3.5)) for i, h in enumerate(hs)}
    cli._write_over_top(tmp_path, ot, hs, cfg, 2.0, None)
    doc = json.loads((tmp_path / cli.OVER_TOP_META).read_text())
    raw = np.frombuffer((tmp_path / doc["file"]).read_bytes(), "<f2")
    n = cfg.ny * cfg.nx
    assert raw.size == len(hs) * 2 * n + n and doc["headings"] == hs and doc["clearance_m"] == 2.0
    uv = raw[: len(hs) * 2 * n].reshape(len(hs), 2, cfg.ny, cfg.nx)
    assert (uv[1] == 2.0).all() and (raw[-n:] == 3.5).all()
    ramp = {h: (np.zeros((2, cfg.ny, cfg.nx)), np.arange(cfg.ny, dtype=float)[:, None] * np.ones(cfg.nx)) for h in hs}
    cli._write_over_top(tmp_path / "r", ramp, hs, cfg, 2.0, None)
    hr = np.frombuffer((tmp_path / "r" / cli.OVER_TOP_FILE).read_bytes(), "<f2")[-n:].reshape(cfg.ny, cfg.nx)
    assert hr[0, 0] == cfg.ny - 1 and hr[-1, 0] == 0, "rows run south, as the frames store them"
    cli._write_over_top(tmp_path / "b", {}, hs, cfg, 2.0, "blocks keep no whole field to read over the top")
    assert "skipped" in json.loads((tmp_path / "b" / cli.OVER_TOP_META).read_text())
