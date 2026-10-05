"""Each cell's representative return: the crown surface for canopy, the bare earth's own return for open ground."""

import numpy as np
import torch

import representative as R


def _cell(r, c, nx):
    return r * nx + c


def test_open_ground_takes_its_own_ground_return_not_the_eave_above():
    ny, nx, dx = 4, 4, 0.5
    canopy, roof = np.zeros(ny * nx, bool), np.zeros(ny * nx, bool)
    # cell (1, 1): a ground return and an eave 6 m up; cell (1, 2): only the eave; cell (1, 3): ground
    z = np.array([0.1, 6.0, 6.1, 0.2])
    height = z.copy()
    up = np.ones(4, bool)
    cell = np.array([_cell(1, 1, nx), _cell(1, 1, nx), _cell(1, 2, nx), _cell(1, 3, nx)])
    rep = R.representative(z, height, up, cell, (ny, nx), dx, canopy, roof)
    assert rep[_cell(1, 1, nx)] == 0, "the ground return, not the higher eave"
    assert rep[_cell(1, 2, nx)] in (0, 3), "an open cell with only an eave takes its nearest open ground return"
    assert rep[_cell(0, 0, nx)] in (0, 3), "an empty cell takes the nearest cell's"


def test_a_canopy_cell_takes_the_highest_return_within_2_m():
    ny, nx, dx = 8, 8, 0.5
    canopy = np.zeros(ny * nx, bool)
    canopy[_cell(4, 4, nx)] = canopy[_cell(4, 5, nx)] = True
    roof = np.zeros(ny * nx, bool)
    z = np.array([12.0, 20.0, 0.1])               # a low branch in a crown gap, the crown top 1 m away, ground
    height = z.copy()
    up = np.array([True, True, True])
    cell = np.array([_cell(4, 4, nx), _cell(4, 3, nx), _cell(0, 0, nx)])
    rep = R.representative(z, height, up, cell, (ny, nx), dx, canopy, roof)
    assert rep[_cell(4, 4, nx)] == 1 and rep[_cell(4, 5, nx)] == 1


def test_the_crowns_light_is_smoothed_over_2_m_and_open_ground_is_not():
    canopy = np.zeros((8, 8), bool)
    canopy[:, :6] = True
    assert R.crown_smoother(np.zeros(64, bool), (8, 8), 0.5) is None
    sm = R.crown_smoother(canopy.ravel(), (8, 8), 0.5)
    v = np.zeros((8, 8), np.float32)
    v[:, :3] = 100.0                              # a 2 m step inside the crown
    v[:, 6:] = 7.0                                # open ground
    out = sm(torch.as_tensor(v.reshape(1, -1))).numpy().reshape(8, 8)
    assert np.allclose(out[:, 6:], 7.0)
    assert np.abs(np.diff(out[:, :6], axis=1)).max() < 60.0 and 0.0 <= out[:, :6].min() and out[:, :6].max() <= 100.0
