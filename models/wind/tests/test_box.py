"""A site that follows its ordered polygon: the grid over its fetch box, nx x ny and centred on it, the masks
on the box, the coarse solve over the fine grid's own extent, the viewer over the display box snapped to its
cell. Without a box every site keeps its disc."""

import json

import numpy as np

import cli
import domain
import params
import sites
import view

DISPLAY = (-100.0, -40.0, 60.0, 30.0)
FETCH = (-120.0, -60.0, 80.0, 50.0)          # the display box grown by the 20 m buffer


def _tif(path, a, x0, y1, cell, nodata):
    import rasterio
    from affine import Affine

    with rasterio.open(path, "w", driver="GTiff", height=a.shape[0], width=a.shape[1], count=1,
                       dtype=a.dtype, transform=Affine(cell, 0.0, x0, 0.0, -cell, y1), nodata=nodata) as dst:
        dst.write(a, 1)


def _bundle(tmp, box=True):
    site = sites.get_site("campanile")
    (tmp / "surface").mkdir(parents=True)
    (tmp / "semantics").mkdir()
    aoi = {"radius_m": site.radius_m, "buffer_m": 20.0}
    if box:
        aoi["box_scene_m"] = {"fetch": list(FETCH), "display": list(DISPLAY)}
    doc = {"aoi": aoi, "scene_frame": {"anchor_utm": list(site.anchor_utm), "grid_convergence_deg": 0.0}}
    (tmp / "surface" / "aoi_campanile.json").write_text(json.dumps(doc))
    rows = [{"id": 0, "class_id": params.FALLBACK_CLASS, "volume_category": "plantable", "closed": "yes",
             "params": {"z0_m": 0.03, "cd": 0.0, "LAI": 0.0, "closure": 1.0}}]
    (tmp / "semantics" / "parameters.json").write_text(json.dumps({"rows": rows}))
    r = site.radius_m + 20.0
    x = -r + np.arange(int(2 * r)) + 0.5
    X, _ = np.meshgrid(x, -x)
    dtm = (0.05 * X).astype(np.float32)
    ax, ay = site.anchor_utm[:2]
    _tif(tmp / "semantics" / "class_top_1m.tif", np.zeros(dtm.shape, np.uint8), ax - r, ay + r, 1.0, 255)
    _tif(tmp / "surface" / "dtm_1m.tif", dtm, -r, r, 1.0, np.nan)
    _tif(tmp / "surface" / "dsm_1m.tif", dtm, -r, r, 1.0, np.nan)
    return sites.bundled(site, tmp)


def test_a_box_site_solves_nx_by_ny_centred_on_its_fetch_box(tmp_path):
    site = _bundle(tmp_path)
    assert sites.box_of(site, tmp_path) == {"fetch": FETCH, "display": DISPLAY}
    assert sites.box_cells(FETCH, 1.0) == (256, 128)
    scene, rx = domain.from_bundle(site, 1.0, tmp_path)
    assert (scene.grid.nx, scene.grid.ny) == (256, 128)
    assert scene.origin[:2] == (-20.0 - 128.0, -5.0 - 64.0)
    inside = domain.in_box(scene.grid, scene.origin[:2], FETCH)
    assert inside.sum() == 200 * 110 and rx["columns_beyond_fetch_radius"] == int((~inside).sum())
    assert rx["box_scene_m"]["fetch"] == list(FETCH)
    # the coarse solve covers the fine extent with the same centre, as a unit run rebuilds it
    cx, cy = cli.coarse_cells(scene, 4.0)
    coarse, _ = domain.from_bundle(site, 4.0, tmp_path, cells=(cx, cy))
    assert (coarse.grid.nx, coarse.grid.ny) == (64, 32) and coarse.origin[:2] == scene.origin[:2]
    fx, fy = sites.box_cells(FETCH, 1.0)                 # what `levels --extent-dx 1 --dx 4` asks for
    assert (int(round(fx * 1.0 / 4.0)), int(round(fy * 1.0 / 4.0))) == (cx, cy)
    # an int --width-cells is a disc site's square: ignored on a box site
    same, _ = domain.from_bundle(site, 1.0, tmp_path, cells=512)
    assert (same.grid.nx, same.grid.ny) == (256, 128)


def test_without_a_box_the_site_keeps_its_disc_square(tmp_path):
    site = _bundle(tmp_path, box=False)
    assert sites.box_of(site, tmp_path) is None
    scene, rx = domain.from_bundle(site, 2.0, tmp_path)
    n = site.cells_across(2.0)
    assert (scene.grid.nx, scene.grid.ny) == (n, n) and scene.origin[:2] == (-n, -n) and "box_scene_m" not in rx
    coarse, _ = domain.from_bundle(site, 4.0, tmp_path, cells=cli.coarse_cells(scene, 4.0))
    assert coarse.grid.nx == coarse.grid.ny == n // 2


def test_the_terrain_floor_reads_the_fetch_box_less_its_rim(tmp_path):
    _bundle(tmp_path)
    floor = params.terrain_floor(tmp_path, 1.0, 132.32, FETCH)
    lo = 0.05 * (FETCH[0] + params.FLOOR_RIM_M)
    assert lo - 0.1 <= floor <= lo + 5.0
    assert params.terrain_floor(tmp_path, 1.0, 132.32) < floor, "the disc reaches further west, lower"


def test_the_viewer_grid_is_the_display_box_snapped_out_to_its_cell():
    cfg = view.ViewConfig.for_box((-100.5, -40.0, 60.2, 30.0), 2.0, heights_m=(1.0, 5.0))
    assert cfg.box == (-102.0, -40.0, 62.0, 30.0) and (cfg.nx, cfg.ny) == (82, 35)
    assert cfg.origin == (-102.0, 30.0, 0.0)
    x, y = cfg.centres()
    assert x.shape == (35, 82) and x[0, 0] == -101.0 and y[0, 0] == -39.0 and y[-1, 0] == 29.0
    shown = cfg.shown(x, y, 1.0)
    assert not shown[:, 0].any() and shown[:, 1].all() and not shown[:, -1].any()
    disc = view.ViewConfig(half_m=30.0)
    assert disc.box is None and disc.nx == disc.ny == disc.n == 30 and disc.origin == (-30.0, 30.0, 0.0)


def test_levels_on_a_box_view_are_nan_outside_the_display_box():
    g = domain.Grid.uniform(2.0, 80, 40, 20)
    scene = domain.box(g, 10.0, 13.5)
    scene.origin = (-80.0, -40.0, 0.0)
    cfg = view.ViewConfig.for_box((-50.0, -20.0, 30.0, 10.0), 2.0, heights_m=(1.0, 25.0))
    out, solid, filled = view.levels(np.ones((1,) + g.shape), scene, cfg, display_radius=1.0)
    assert out.shape == (1, 2, 15, 40)
    assert np.isfinite(out[0, 1]).all(), "every cell of the display box carries the level above the tower"
