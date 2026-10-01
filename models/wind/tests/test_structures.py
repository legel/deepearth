"""A raised surface the survey finds no plants on is a structure, solid to its DSM: the scene follows the measured
surface whatever the class raster calls its top (a third site's decks, classed concrete, had been solved
as a porous crown). No network, no site data."""

import numpy as np

import canopy
import domain
import params
from domain import Grid


def _site():
    g = Grid.uniform(1.0, 12, 10, 32)
    dtm = np.full((10, 12), 100.0)
    dsm, classes = dtm.copy(), np.full((10, 12), 3)
    dsm[2:5, 2:5] = 110.0                       # a crown over asphalt, in a top-class raster that says asphalt
    dsm[6:9, 6:10] = 125.0                      # a 25 m deck whose top the classes read as asphalt too
    cols = params.from_legend(classes, {3: "asphalt_pavement"}, g)
    return g, dtm, dsm, cols


def test_only_a_raised_surface_with_plants_becomes_canopy_and_a_deck_stays_solid():
    g, dtm, dsm, cols = _site()
    veg = np.zeros((10, 12), bool)
    veg[2:5, 2:5] = True                        # the survey's returns hold a crown here and nowhere else
    old, over_old = params.canopy_over_surfaces(cols, dtm, dsm, np.ones((10, 12), bool))
    assert over_old[6:9, 6:10].all(), "by the class rule alone the deck turned to canopy"
    new, over = params.canopy_over_surfaces(cols, dtm, dsm, np.ones((10, 12), bool), veg)
    assert over[2:5, 2:5].all() and not over[6:9, 6:10].any() and over.sum() == 9
    s = domain.voxelize(new, dtm, dsm, g, "deck")
    assert s.solid[:, 7, 7].sum() == 25, "the deck is solid up to its measured surface"
    assert s.solid[:, 3, 3].sum() == 0 and (s.sink[:, 3, 3] > 0).sum() == 10, "the crown is drag, not a tower"


def test_the_vegetation_mask_reads_the_survey_s_plant_area_on_the_solver_grid():
    ny, nx = 5, 6
    tau0 = np.zeros((ny, nx))
    tau0[1, 2] = 2.0                            # one 2 m profile cell with a crown's optical depth
    tau0[4, 5] = np.nan                         # too few first returns: no evidence
    heights = np.array([0.0, 5.0, 10.0])
    f = np.ones((3, ny, nx))
    p = canopy.Profile(tau0=tau0, f=f, f_leaf=f.copy(), heights=heights, x0=0.0, y0=0.0, cell=2.0, leaf={})
    g = Grid.uniform(1.0, 12, 10, 4)
    m = canopy.vegetation_mask(p, g, (0.0, 0.0))
    assert m[2:4, 4:6].all() and m.sum() == 4, "the 2 m profile cell covers four 1 m columns"


def test_ground_beyond_the_data_has_no_radial_steps_and_stays_on_the_data_inside():
    """A survey disc whose ground stands 10 m higher in one half: carried out along the rays by the nearest measured
    column, the unmeasured ring steps 10 m between the halves; filled smoothly it slopes, and inside it is unchanged."""
    n = 120
    yy, xx = np.mgrid[:n, :n] - n / 2 + 0.5
    ground = np.where(xx > 0, 110.0, 100.0)
    a = np.where(np.hypot(xx, yy) < 35, ground, np.nan)
    got = domain.smooth_fill(a, 8.0)
    near = domain._fill_nodata(a)
    ring = np.hypot(xx, yy) > 55                         # past two sigmas from the data
    step = lambda z: np.abs(np.diff(z, axis=1))[ring[:, 1:]].max()  # noqa: E731
    assert np.array_equal(got[~np.isnan(a)], a[~np.isnan(a)])
    assert step(near) >= 9.9 and step(got) < 2.5, "the fill's step between the halves is a slope, not a wall"


def _evidence_site():
    """Asphalt ground; a crown over it that the top-class raster reads as asphalt; a 25 m deck read as asphalt; a
    white-roofed box the raster calls a crown; and a true crown the raster calls a crown."""
    g = Grid.uniform(1.0, 12, 10, 32)
    dtm = np.full((10, 12), 100.0)
    dsm, cls = dtm.copy(), np.full((10, 12), 3)
    echo, green = np.full((10, 12), np.nan), np.full((10, 12), 0.05)
    dsm[2:5, 2:5], echo[2:5, 2:5] = 110.0, 0.6            # crown over paving: many echoes
    dsm[6:9, 6:10], echo[6:9, 6:10] = 125.0, 0.02         # deck: one echo a pulse
    dsm[1:3, 8:11], echo[1:3, 8:11], green[1:3, 8:11], cls[1:3, 8:11] = 118.0, 0.02, -0.05, 2  # white box, "crown"
    dsm[4:6, 9:11], echo[4:6, 9:11], green[4:6, 9:11], cls[4:6, 9:11] = 111.0, 0.03, 0.10, 2   # a green crown, few echoes
    dsm[7:9, 1:3], echo[7:9, 1:3], green[7:9, 1:3], cls[7:9, 1:3] = 112.0, 0.7, 0.1, 2        # a true crown
    cols = params.from_legend(cls, {1: "roof_sealed", 2: "tree_canopy", 3: "asphalt_pavement"}, g)
    heights = np.array([0.0, 5.0, 10.0, 20.0])
    ones = np.ones((4, 10, 12))
    p = canopy.Profile(tau0=np.full((10, 12), 2.0), f=ones, f_leaf=ones.copy(), heights=heights, x0=0.0, y0=0.0,
                       cell=1.0, leaf={}, echo=echo, green=green)
    return g, dtm, dsm, cols, p


def test_echoes_keep_decks_and_white_boxes_solid_and_crowns_porous_with_drag_only_on_plants():
    g, dtm, dsm, cols, p = _evidence_site()
    ev = canopy.evidence(p, g, (0.0, 0.0))
    assert (ev[2:5, 2:5] == canopy.PLANTS).all() and (ev[7:9, 1:3] == canopy.PLANTS).all()
    assert (ev[6:9, 6:10] != canopy.PLANTS).all(), "one echo a pulse: no plants on the deck"
    assert (ev[4:6, 9:11] == canopy.NONE).all(), "a green crown returning few echoes is not proved a structure"
    assert (ev[1:3, 8:11] == canopy.STRUCTURE).all(), "one echo a pulse under a white roof: a structure"
    assert (ev[0, 0] == canopy.NONE), "nothing raised to judge"
    everywhere = np.ones((10, 12), bool)
    cols, over = params.canopy_over_surfaces(cols, dtm, dsm, everywhere, ev == canopy.PLANTS)
    cols, solid = params.structures_classed_as_crowns(cols, dtm, dsm, everywhere, ev == canopy.STRUCTURE)
    assert over[2:5, 2:5].all() and not over[6:9, 6:10].any() and over.sum() == 9
    assert solid[1:3, 8:11].all() and not solid[7:9, 1:3].any() and not solid[4:6, 9:11].any() and solid.sum() == 6
    s = domain.voxelize(cols, dtm, dsm, g, "evidence")
    assert s.solid[:, 7, 7].sum() == 25 and s.solid[:, 2, 9].sum() == 18, "the deck and the white box are solid"
    assert s.solid[:, 3, 3].sum() == 0 and s.solid[:, 8, 2].sum() == 0, "both crowns are porous"
    crown = ~np.asarray(cols.solid, bool) & (np.asarray(cols.lai) > 0)
    s.plants = (ev == canopy.PLANTS) | ((ev == canopy.NONE) & crown)
    canopy.apply(s, p, None)
    drag = s.sink.sum(axis=0)
    assert drag[3, 3] > 0 and drag[8, 2] > 0, "drag in the crowns"
    assert drag[0, 0] == 0 and drag[5, 0] == 0, "no drag over the paving the survey saw nothing raised on"


def test_a_planar_top_that_is_not_green_is_a_structure_though_glass_splits_its_pulses():
    """The press box: a glass front splits pulses (echo share 0.6, like a crown) but its top is a plane and it is not
    green, so it is a structure; a green crown with the same echoes and a rough top stays plants, and so does a trimmed
    green hedge whose top is flat."""
    g = Grid.uniform(1.0, 6, 3, 8)
    ones = np.ones((4, 3, 6))
    echo = np.full((3, 6), 0.6)
    green = np.array([[0.0, 0.0, 0.10, 0.10, 0.10, 0.0]] * 3)
    flat = np.array([[0.002, 0.002, 0.15, 0.15, 0.002, 0.2]] * 3)
    p = canopy.Profile(tau0=np.full((3, 6), 2.0), f=ones, f_leaf=ones.copy(), heights=np.array([0.0, 5.0, 10.0, 20.0]),
                       x0=0.0, y0=0.0, cell=1.0, leaf={}, echo=echo, green=green, flat=flat)
    ev = canopy.evidence(p, g, (0.0, 0.0))[0]
    assert list(ev) == [canopy.STRUCTURE, canopy.STRUCTURE, canopy.PLANTS, canopy.PLANTS, canopy.PLANTS, canopy.PLANTS]
