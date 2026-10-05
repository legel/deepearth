"""The canopy's drag with height from a survey profile: plant area per volume from the optical depth above each height."""

import numpy as np
import pytest

import canopy
import checks
import domain


def _profile(tau0=2.0, leaf=None):
    hs = np.arange(0.0, 32.0, 4.0)
    f = np.clip((24.0 - hs) / 12.0, 0.0, 1.0)            # all of the plant area between 12 and 24 m: an open trunk space
    grid = lambda v: np.broadcast_to(np.asarray(v, float)[:, None, None], (len(hs), 4, 4)).copy()  # noqa: E731
    return canopy.Profile(tau0=np.full((4, 4), tau0), f=grid(f), f_leaf=grid(f), heights=hs, x0=-8.0, y0=-8.0,
                          cell=4.0, leaf=leaf or {"G": 0.5, "omega": 0.8})


def test_plant_area_follows_the_optical_depth_above_each_height():
    p = _profile()
    a = canopy.density(p, 196)[:, 0, 0]
    pai = 2.0 / 0.4                                       # tau0 / (G omega)
    assert a.sum() * 4.0 == pytest.approx(pai), "the bands add up to the column's plant area"
    assert a[:3].sum() == 0.0 and a[3:6].sum() > 0.0, "nothing in the trunk space, all in the crown"


def test_the_seasons_leaves_are_added_where_the_crown_is():
    lai = [0.5] * 100 + [4.5] * 200 + [0.5] * 66           # leaf-off, leaf-on, leaf-off
    p = _profile(leaf={"G": 0.5, "omega": 0.8, "lai_doy": lai, "floor": 0.5, "lai_survey": 0.5})
    assert canopy.plant_area_above(p, 196)[0, 0, 0] == pytest.approx(5.0 + 4.0)
    assert canopy.plant_area_above(p, 15)[0, 0, 0] == pytest.approx(5.0)


def test_the_scene_takes_the_profile_where_the_survey_covers_it():
    g = checks.grid(1.0, 16, 16, 40)
    scene = domain.flat(g, checks.Z0)
    rec = canopy.apply(scene, _profile(), 196, cd=0.2)
    k_trunk, k_crown = int(np.searchsorted(g.zc, 6.0)), int(np.searchsorted(g.zc, 18.0))
    assert rec["columns_from_survey"] == 16 * 16
    assert scene.sink[k_trunk, 8, 8] == 0.0 and scene.sink[k_crown, 8, 8] == pytest.approx(0.2 * 5.0 / 12.0)
