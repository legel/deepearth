"""The boundary buffer: one bundle, two grids, and the same cells inside the display disc."""

import json
from pathlib import Path

import numpy as np
import pytest

import buffer
import checks
import domain
import params
import sites
from physics import CLASSES
from solver import SolverConfig, solve

SITE = sites.get_site("campanile")
RING_M = 30.0


def _tif(path: Path, a: np.ndarray, x0: float, y1: float, cell: float, nodata: float) -> None:
    import rasterio
    from affine import Affine

    with rasterio.open(path, "w", driver="GTiff", height=a.shape[0], width=a.shape[1], count=1,
                       dtype=a.dtype, transform=Affine(cell, 0.0, x0, 0.0, -cell, y1), nodata=nodata) as dst:
        dst.write(a, 1)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory) -> Path:
    """Data out to 30 m past the display disc, and a 12 m slab building straddling its east edge."""
    root = tmp_path_factory.mktemp("bundle")
    (root / "surface").mkdir()
    (root / "semantics").mkdir()
    doc = {"aoi": {"radius_m": SITE.radius_m, "buffer_m": RING_M},
           "scene_frame": {"anchor_utm": list(SITE.anchor_utm), "grid_convergence_deg": 0.0}}
    (root / "surface" / "aoi_campanile.json").write_text(json.dumps(doc))
    names = [params.FALLBACK_CLASS, "facade_masonry"]
    rows = [{"id": i, "class_id": n, "volume_category": "building" if n == "facade_masonry" else "plantable",
             "closed": "yes", "params": {"z0_m": CLASSES[n].z0_m, "cd": CLASSES[n].cd, "LAI": CLASSES[n].lai,
                                        "closure": 1 - CLASSES[n].poro}} for i, n in enumerate(names)]
    (root / "semantics" / "parameters.json").write_text(json.dumps({"rows": rows}))
    r = SITE.radius_m + RING_M
    x = -r + np.arange(int(2 * r)) + 0.5
    X, Y = np.meshgrid(x, -x)
    inside, slab = np.hypot(X, Y) <= r, (np.abs(X - SITE.radius_m) < 15) & (np.abs(Y) < 20)
    dtm = np.where(inside, 0.02 * X, np.nan).astype(np.float32)
    ax, ay = SITE.anchor_utm[:2]
    _tif(root / "semantics" / "class_top_1m.tif", np.where(inside, slab, 255).astype(np.uint8), ax - r, ay + r, 1.0, 255)
    _tif(root / "surface" / "dtm_1m.tif", dtm, -r, r, 1.0, np.nan)
    _tif(root / "surface" / "dsm_1m.tif", np.where(slab, dtm + 12.0, dtm).astype(np.float32), -r, r, 1.0, np.nan)
    return root


def test_the_buffer_comes_from_the_bundle_and_widens_only_the_grid(bundle):
    before, after = sites.bundled(SITE, bundle, 0.0), sites.bundled(SITE, bundle)
    assert after.buffer_m == RING_M and after.fetch_radius_m == SITE.radius_m + RING_M
    assert before.display_radius_m == after.display_radius_m == SITE.radius_m
    assert after.cells_across(2.0) > before.cells_across(2.0) and after.levels(2.0) == before.levels(2.0)


def test_both_arms_are_the_same_cells_inside_the_display_disc(bundle):
    sb, rb = domain.from_bundle(sites.bundled(SITE, bundle, 0.0), 2.0, bundle)
    sa, ra = domain.from_bundle(sites.bundled(SITE, bundle), 2.0, bundle)
    assert rb["terrain_floor_scene_m"] == ra["terrain_floor_scene_m"]
    diff = buffer.identical_inside(sb, sa, SITE.radius_m)
    assert diff["solid"] == diff["sink"] == diff["z0"] == 0 and diff["columns_compared"] > 3000
    assert rb["nodata_cells"] == (~buffer.disc(sb, SITE.radius_m)).sum() > ra["nodata_cells"] - (~buffer.disc(sa, SITE.radius_m)).sum()
    iy, ix = (int((v - o) / 2.0) for v, o in ((0.0, sa.origin[1]), (120.0, sa.origin[0])))
    wall = sa.grid.zc[sa.solid[:, iy, ix]].max() - sa.terrain[iy, ix]
    assert 9.0 < wall < 12.0


def test_no_flux_enters_a_solid_and_some_enters_open_air():
    g = checks.grid()
    scene = domain.cube(g, 8.0)
    res = solve(scene, checks.profile(), checks.WEST, SolverConfig(steps=0))
    cube = buffer.prism(scene, scene.solid.any(axis=0))
    air = np.zeros_like(scene.solid)
    air[:4, :, :8] = True
    assert cube.sum() == scene.solid.sum() and buffer.entering_flux(res, scene, cube) == 0.0
    assert buffer.entering_flux(res, scene, air) > 0.0


def test_bands_partition_the_disc_by_distance_from_its_edge():
    dist = np.array([1.0, 7.0, 15.0, 90.0, 111.0])
    rows = buffer.band_table(np.ones(5), np.full(5, 2.0), dist, (0.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0),
                             SITE.radius_m)
    assert [r["samples"] for r in rows] == [1, 1, 1, 0, 0, 0, 1, 1]
    assert rows[0]["relative_mean_abs_change"] == 0.5


def test_the_viewer_slices_and_the_heading_blend():
    import view

    assert view.ViewConfig().faces == [0.0, 4.0, 16.0, 44.0]
    hs = view.headings([60.0, 249.0], 22.5)
    assert hs == [45.0, 67.5, 247.5, 270.0]
    assert view.blend({h: np.array([h]) for h in hs}, 60.0, 22.5)[0] == pytest.approx(60.0)


def test_a_final_projection_meets_a_tighter_divergence():
    scene = domain.cube(checks.grid(), 8.0)
    loose = solve(scene, checks.profile(), checks.WEST, SolverConfig(steps=3))
    tight = solve(scene, checks.profile(), checks.WEST, SolverConfig(steps=3, tol_final=1e-10))
    assert tight.divergence_max_1_s < loose.divergence_max_1_s / 100


def test_the_viewer_crop_runs_north_and_blanks_beyond_the_display_disc():
    import view
    from domain import Grid

    scene = domain.flat(Grid.uniform(2.0, 128, 128, 4))
    north = np.broadcast_to((scene.origin[1] + scene.grid.yc)[:, None], (128, 128))[None, None]
    out = view.crop(north, scene, view.ViewConfig(), SITE.radius_m)
    assert out.shape == (1, 1, 120, 120) and out[0, 0, 10, 60] < out[0, 0, 109, 60]
    assert np.isnan(out[0, 0, 0, 0]) and np.isfinite(out[0, 0, 60, 60])
