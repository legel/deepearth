"""Grid geometry and scene construction, synthetic and from rasters."""

import numpy as np
import pytest

import domain
from domain import Grid
from physics import CLASSES, GROUND_CLASS


def test_stretched_levels_start_at_dz0_and_grow_geometrically():
    g = Grid.stretched(0.2, 8, 8, 10, 0.2, 1.06)
    assert g.zf[0] == 0.0 and g.dz[0] == pytest.approx(0.2)
    assert np.allclose(g.dz[1:] / g.dz[:-1], 1.06)
    assert g.zc[0] == pytest.approx(0.1) and g.top == pytest.approx(0.2 * (1.06 ** 10 - 1) / 0.06)
    assert g.shape == (10, 8, 8) and g.cells == 640


def test_cube_occupies_its_footprint_up_to_its_height():
    g = Grid.uniform(1.0, 16, 16, 8)
    s = domain.cube(g, 4.0, centre=(8.0, 8.0), z0=0.03, z0_wall=0.5)
    assert s.solid.sum() == 4 * 4 * 4
    assert s.z0[6:10, 6:10].min() == 0.5 and s.z0[0, 0] == 0.03
    assert s.sink.max() == 0.0


def test_porous_block_carries_leaf_area_per_volume():
    g = Grid.uniform(1.0, 16, 16, 8)
    s = domain.porous_block(g, 4.0, lai=6.0, cd=0.2, centre=(8.0, 8.0))
    assert (s.sink > 0).sum() == 64 and s.sink.max() == pytest.approx(0.2 * 6.0 / 4.0)
    assert not s.solid.any()


def test_ridge_spans_the_domain_and_peaks_at_its_height():
    g = Grid.uniform(1.0, 32, 8, 12)
    s = domain.ridge(g, 6.0, 8.0)
    assert s.solid[:, 0, :].sum() == s.solid[:, -1, :].sum()
    assert s.solid[:, 0, 16].sum() == 6 and s.solid[:, 0, 0].sum() == 0


def test_rasters_voxelise_terrain_buildings_canopy_and_fill_nodata():
    g = Grid.uniform(1.0, 12, 10, 12)
    dtm = np.full((10, 12), 100.0)
    dtm[:, 6:] = 102.0  # a 2 m step in the terrain
    dsm = dtm.copy()
    classes = np.full((10, 12), 3)  # asphalt everywhere
    dsm[2:4, 2:4] = 105.0  # a 5 m building on the low side
    classes[2:4, 2:4] = 1
    dsm[6:9, 8:11] = 108.0  # a 6 m tree on the high side
    classes[6:9, 8:11] = 2
    dtm[0, 0] = dsm[0, 0] = np.nan  # one unknown pixel
    legend = {1: "facade_masonry", 2: "tree_canopy", 3: "asphalt_pavement"}
    s = domain.from_rasters(dtm, dsm, classes, legend, g)

    assert s.solid[:, 5, 0].sum() == 0 and s.solid[:, 5, 8].sum() == 2, "terrain is 0 and 2 m"
    assert s.solid[:, 2, 2].sum() == 5, "building columns are solid to the DSM"
    assert s.solid[:, 7, 9].sum() == 2 and (s.sink[:, 7, 9] > 0).sum() == 6, "canopy is drag above ground"
    tree = CLASSES["tree_canopy"]
    assert s.sink[4, 7, 9] == pytest.approx(tree.cd * tree.lai / 6.0)
    assert s.z0[2, 2] == CLASSES["facade_masonry"].z0_m
    assert s.z0[7, 9] == CLASSES[GROUND_CLASS].z0_m, "ground under canopy is the default class"
    assert s.z0[5, 5] == CLASSES["asphalt_pavement"].z0_m
    assert s.solid[:, 0, 0].sum() == 0, "the unknown pixel took its neighbour's terrain"
    assert s.summary()["solid_fraction"] > 0


def test_rasters_must_match_the_grid():
    g = Grid.uniform(1.0, 4, 4, 4)
    with pytest.raises(AssertionError, match="do not match"):
        domain.from_rasters(np.zeros((3, 4)), np.zeros((3, 4)), np.zeros((3, 4), int), {}, g)


def test_a_crown_over_a_surface_class_is_canopy_not_a_tower():
    import params

    g = Grid.uniform(1.0, 12, 10, 16)
    dtm = np.full((10, 12), 100.0)
    dsm, classes = dtm.copy(), np.full((10, 12), 3)
    dsm[4:7, 4:7] = 110.0                       # a crown over asphalt, in a top-class raster that says asphalt
    dsm[1, 1] = 100.5                           # a kerb: under the overhang line, still the surface
    classes[8:10, 8:10] = 1
    dsm[8:10, 8:10] = 106.0                     # a building stays solid
    cols = params.from_legend(classes, {1: "facade_masonry", 3: "asphalt_pavement"}, g)
    same, none = params.canopy_over_surfaces(cols, dtm, dsm, np.zeros((10, 12), bool))
    assert not none.any() and np.array_equal(same.row, cols.row)
    cols, over = params.canopy_over_surfaces(cols, dtm, dsm, np.ones((10, 12), bool))
    assert over.sum() == 9 and over[4:7, 4:7].all() and not over[8:10, 8:10].any() and not over[1, 1]
    s = domain.voxelize(cols, dtm, dsm, g, "crown")
    assert s.solid[:, 5, 5].sum() == 0 and (s.sink[:, 5, 5] > 0).sum() == 10, "drag through the crown, no solid"
    assert s.solid[:, 8, 8].sum() == 6 and s.solid[:, 1, 1].sum() == 0


def test_a_sealed_shaft_too_narrow_for_flow_is_a_pocket_and_a_courtyard_is_not():
    solid = np.ones((20, 20), bool)
    solid[:, :3] = False                        # open ground reaching the edge
    solid[5, 8] = False                         # a one-cell shaft
    solid[10:12, 8:11] = False                  # a 2 x 3 light well
    solid[4:10, 13:19] = False                  # a 6 x 6 courtyard
    got = domain.pockets(~solid)
    assert got[5, 8] and got[10:12, 8:11].all() and got.sum() == 7
    assert not got[4:10, 13:19].any() and not got[:, :3].any()


def test_a_crown_inside_a_building_is_closed_to_its_roof_and_a_courtyard_stays_open():
    g = Grid.uniform(1.0, 30, 30, 16)
    dtm = np.full((30, 30), 100.0)
    dsm, classes = dtm.copy(), np.full((30, 30), 3)
    classes[5:25, 5:25] = 1
    dsm[5:25, 5:25] = 110.0                     # a 10 m hall
    classes[9, 9] = 2                           # a crown pixel the class raster reads inside it
    classes[14:16, 9:12] = 3
    dsm[14:16, 9:12] = 100.0                    # a 2 x 3 light well to the ground
    classes[10:16, 15:21] = 3
    dsm[10:16, 15:21] = 100.0                   # a 6 x 6 courtyard
    s = domain.from_rasters(dtm, dsm, classes, {1: "facade_masonry", 2: "tree_canopy", 3: "asphalt_pavement"}, g)
    assert s.solid[:, 9, 9].sum() == 10 and s.sink[:10, 9, 9].max() == 0.0, "the crown pixel is the hall to its roof"
    assert (s.solid[:, 14:16, 9:12].sum(axis=0) == 10).all() and np.all(s.top[14:16, 9:12] == 10.0)
    assert s.solid[:, 10:16, 15:21].sum() == 0 and s.solid[:, 2, 2].sum() == 0
    assert s.closed_cells == 7 * 10
    assert (np.diff(s.solid.astype(int), axis=0) <= 0).all(), "every column is solid from the floor up"


def test_a_roof_edge_the_class_raster_calls_pavement_stays_solid():
    import params

    g = Grid.uniform(1.0, 14, 12, 16)
    dtm = np.full((12, 14), 100.0)
    dsm, classes = dtm.copy(), np.full((12, 14), 3)
    classes[2:6, 2:6] = 1
    dsm[2:6, 2:6] = 110.0                       # a 10 m building
    dsm[2:6, 6] = 110.4                         # its east rim, which the class raster calls asphalt
    dsm[6, 2:6] = 104.0                         # a 4 m crown beside it, over asphalt
    dsm[9:11, 10:12] = 108.0                    # a crown in the open
    cols = params.from_legend(classes, {1: "facade_masonry", 3: "asphalt_pavement"}, g)
    cols, over = params.canopy_over_surfaces(cols, dtm, dsm, np.ones((12, 14), bool))
    assert not over[2:6, 6].any(), "the rim stands at the roof"
    assert over[6, 2:6].all() and over[9:11, 10:12].all() and over.sum() == 8
    s = domain.voxelize(cols, dtm, dsm, g, "rim")
    assert (s.solid[:, 2:6, 6].sum(axis=0) == 10).all()


def _tif(path, a, x0, y1, cell, nodata):
    import rasterio
    from affine import Affine

    with rasterio.open(path, "w", driver="GTiff", height=a.shape[0], width=a.shape[1], count=1,
                       dtype=a.dtype, transform=Affine(cell, 0.0, x0, 0.0, -cell, y1), nodata=nodata) as dst:
        dst.write(a, 1)


def test_the_ground_band_holds_every_level_over_the_highest_terrain_in_fine_cells(tmp_path):
    import json

    import params
    import sites

    site = sites.get_site("campanile")
    (tmp_path / "surface").mkdir()
    (tmp_path / "semantics").mkdir()
    doc = {"aoi": {"radius_m": site.radius_m, "buffer_m": 20.0},
           "scene_frame": {"anchor_utm": list(site.anchor_utm), "grid_convergence_deg": 0.0}}
    (tmp_path / "surface" / "aoi_campanile.json").write_text(json.dumps(doc))
    rows = [{"id": 0, "class_id": params.FALLBACK_CLASS, "volume_category": "plantable", "closed": "yes",
             "params": {"z0_m": 0.03, "cd": 0.0, "LAI": 0.0, "closure": 1.0}}]
    (tmp_path / "semantics" / "parameters.json").write_text(json.dumps({"rows": rows}))
    r = site.radius_m + 20.0
    x = -r + np.arange(int(2 * r)) + 0.5
    X, Y = np.meshgrid(x, -x)
    inside = np.hypot(X, Y) <= r
    dtm = np.where(inside, 0.2 * X, np.nan).astype(np.float32)          # about 53 m of fall across the disc
    ax, ay = site.anchor_utm[:2]
    _tif(tmp_path / "semantics" / "class_top_1m.tif", np.where(inside, 0, 255).astype(np.uint8), ax - r, ay + r, 1.0, 255)
    _tif(tmp_path / "surface" / "dtm_1m.tif", dtm, -r, r, 1.0, np.nan)
    _tif(tmp_path / "surface" / "dsm_1m.tif", dtm, -r, r, 1.0, np.nan)
    s0, _ = domain.from_bundle(sites.bundled(site, tmp_path), 2.0, tmp_path)
    s, rx = domain.from_bundle(sites.bundled(site, tmp_path), 2.0, tmp_path, ground_band=(1.0, 30.0, 150.0))
    dz, height = rx["ground_band_dz_height_m"]
    assert dz == 1.0 and 80.0 < height < 85.0 and s.grid.top >= s0.grid.top and s.grid.nz % 8 == 0
    highest = float(np.max(s.terrain))
    assert np.all(s.grid.dz[s.grid.zc <= highest + 26.0] == 1.0), "every level to 25 m over every ground"
    assert s0.grid.dz[np.searchsorted(s0.grid.zc, highest)] > 2.0, "the stretched grid was coarser up there"
    _, capped = domain.from_bundle(sites.bundled(site, tmp_path), 2.0, tmp_path, ground_band=(1.0, 30.0, 20.0))
    assert capped["ground_band_dz_height_m"] == [1.0, 20.0]
