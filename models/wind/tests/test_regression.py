"""The final solver on open ground: one fixed scene whose 4 m level every physics change must answer for."""

import numpy as np
import pytest

import checks
import domain
import params
import view
from domain import Grid
from solver import SolverConfig, solve

OPEN_4M_MEAN, OPEN_4M_P90 = 3.7183, 5.1667
"""Speed [m/s] over the scene's open ground at 4 m at this solver's physics (bench, float64, 40 steps of the finite-volume
scheme; the semi-Lagrangian gave 3.6539 and 4.8125)."""


def test_open_ground_at_4_m_is_what_this_solver_gives():
    """A 12 m hall and a crown over asphalt on flat ground, a 5 m/s westerly at 10 m."""
    nx, ny, dx = 48, 40, 2.0
    g = Grid.uniform(dx, nx, ny, 16)
    dtm = np.zeros((ny, nx))
    dsm, classes = dtm.copy(), np.full((ny, nx), 3)
    classes[14:26, 18:24] = 1
    dsm[14:26, 18:24] = 12.0
    dsm[6:10, 30:36] = 8.0
    cols = params.from_legend(classes, {1: "facade_masonry", 3: "asphalt_pavement"}, g)
    cols, over = params.canopy_over_surfaces(cols, dtm, dsm, np.ones((ny, nx), bool))
    assert over.sum() == 24, "the crown is canopy"
    scene = domain.voxelize(cols, dtm, dsm, g, "regression")
    scene.origin = (-nx * dx / 2, -ny * dx / 2, 0.0)
    res = solve(scene, checks.profile(), checks.WEST, SolverConfig(steps=40, tol=1e-6))
    cfg = view.ViewConfig(heights_m=(4.0,), cell_m=dx, half_m=38.0)
    out, solid, filled = view.levels(res.velocity[:2], scene, cfg, display_radius=38.0)
    x, y = cfg.centres()
    open_ = (np.hypot(x, y) <= 38.0) & ~solid[0] & ~filled[0] & ~view.under(scene, cfg, dsm > dtm)
    speed = np.hypot(out[0, 0], out[1, 0])[open_]
    assert open_.sum() > 500
    got = (float(speed.mean()), float(np.percentile(speed, 90)))
    assert got == pytest.approx((OPEN_4M_MEAN, OPEN_4M_P90), rel=5e-3), got
