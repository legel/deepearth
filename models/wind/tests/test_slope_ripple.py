"""The cut-cell ground (domain.voxelize partial): a slope in 1 m cubes is a ramp to the flow, not a staircase of risers.
UC's southwest lawns (6 %) waved at the 17 m step period, 16.5 to 27 km of Year run within a step.

Conditions, none of which can be passed by taking friction away:
  (a) on every cell over the ground, the wall stress is the rough-wall log law at the open part's mid height d over the
      true surface, over the surface's own area, analytically, for a range of slopes and open fractions; the closure
      reads the same d; a cut cell thinner than the roughness sublayer (d < e z0) has its surface read by the cell
      above, at that cell's true distance;
  (b) flat ground carried in cut cells (open 1, 0.9, 0.5 and 0.2, the last thinner than the roughness sublayer) gives a
      surface u* within USTAR_MAX of the log law's from the same solve's 10 m speed;
  (c) on the 6 and 15 % slopes the surface u* holds within CONVERGE_MAX as the cells shrink from 1 m to 0.25 m;
  (d) the 2 m speed has no ripple at the step period (RIPPLE_MAX), where whole cubes do (the test has teeth);
  (e) what enters the domain leaves it, through the cut faces' open shares;
  (f) under a slope-parallel stream (flat ground rotated) a slope's u* is flat ground's within USTAR_MAX.
On a slope, (b)'s form does not apply: the box's free stream is level, so the flow over the plane is a slope under level
air, not flat ground rotated, and over terrain the log law with the local surface stress holds only within an inner layer
of depth l, (l / L) ln(l / z0) = 2 kappa^2 (Jackson and Hunt 1975, QJRMS 101: 929), about 9 m here. The slopes' u* stands
+18 % (6 %) and +43 % (15 %) over the 10 m log law's, steady along a 480 m plane, with no pressure gradient to explain it;
turned onto the slope (f), the same model reads flat ground's u* within 2.2 and 4.2 %: it was the level air, not the
ground. Carried whole, cut cells read those slopes' u* 27 to 54 % over; with FAVOR but read inside the roughness
sublayer, it moved 10 % as the cells shrank.
(b) to (f) are settled as production settles; a GPU when there is one."""

import math
from functools import lru_cache

import numpy as np
import pytest
import torch

import checks
import domain
import params
import view
from domain import Grid
from solver import KAPPA, Model, SolverConfig, solve

RIPPLE_MAX = 0.02
"""Largest departure of the 2 m along-slope speed from a smooth (quadratic) trend over the slope's middle, as a
fraction of its mean."""
USTAR_MAX = 0.05
"""Largest departure, on flat ground, of the u* the solve's surface stress gives from the log law's,
kappa U(10 m) / ln(10 m / z0)."""
CONVERGE_MAX = 0.05
"""Largest change of a slope's surface u* from 1 m to 0.25 m cells."""

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SETTLE = dict(steps=40, cfl=16.0, settle_tol=0.002, settle_every=40, max_steps=1200, tol=1e-3, tol_final=1e-10,
              tol_momentum=0.1, device=DEV, lateral="open")
CONFIGS = {"mixing": SolverConfig(**SETTLE),
           "k-l": SolverConfig(closure="k-l", drive="pressure", inflow="canopy", **SETTLE)}
"""The mixing length the tests use elsewhere, and the production closure (k-l, driven by the mean pressure gradient),
with open sides: a side prescribed by height above the floor is not the slope's."""


def _scene(slope: float, partial: bool, n: int, ny: int, dz: float, band: float, lid: float, rise0: float = 0.0,
           floor=None):
    g = Grid.banded(1.0, n, ny, dz, band, 1.08, lid)
    x = np.arange(n) + 0.5
    dtm = np.broadcast_to(100.0 + rise0 + slope * x, (ny, n)).copy()
    cols = params.from_legend(np.full((ny, n), 3), {3: "turf_grass"}, g)
    return domain.voxelize(cols, dtm, dtm.copy(), g, f"plane {slope:g}", floor, partial=partial), g


@pytest.mark.parametrize("slope,rise0", [(0.0, 0.1), (0.0, 0.6), (0.0, 0.9), (0.03, 0.0), (0.06, 0.37), (0.15, 0.0),
                                         (0.3, 0.2)])
@pytest.mark.parametrize("dz", [1.0, 0.5])
def test_a_cut_cell_s_wall_stress_is_the_log_law_over_the_true_surface(slope, rise0, dz):
    scene, g = _scene(slope, True, 48, 4, dz, 24.0, 32.0, rise0, floor=99.0)
    m = Model(scene, checks.profile(), checks.WEST, SolverConfig(closure="mixing"))
    solid = np.asarray(scene.solid)
    over = ~solid[1:] & solid[:-1]                                 # every fluid cell with ground under it
    assert over.any()
    k = np.nonzero(over)[0] + 1
    dzk = np.diff(g.zf)[k]
    open_ = np.asarray(scene.cut_open)[1:][over]
    assert (open_ < 1).any() and (open_ >= 0.049).all()
    area = math.sqrt(1.0 + slope ** 2)
    z0 = np.broadcast_to(np.asarray(scene.z0), solid.shape)[:-1][over]
    d = np.maximum(open_ * dzk / 2, 1e-3)
    thin = (open_ < 1) & (d < math.e * z0)                         # inside the roughness sublayer: no log layer
    want = np.where(thin, 0.0, (KAPPA / np.log(d / np.minimum(z0, d / math.e))) ** 2 * area / dzk)
    wall = m.wall.cpu().numpy()
    np.testing.assert_allclose(wall[1:][over], want, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(m.height.cpu().numpy()[1:][over], d, rtol=1e-6)
    assert thin.any() or not slope, "a slope's cut cells include ones thinner than the roughness sublayer"
    # a thin cut cell's surface is read by the cell above, at that cell's centre's true distance
    kt, jt, it = (a[thin] for a in np.nonzero(over))
    d2 = g.zc[kt + 2] - np.asarray(scene.top)[jt, it]
    want2 = (KAPPA / np.log(d2 / z0[thin])) ** 2 * area / np.diff(g.zf)[kt + 2]
    np.testing.assert_allclose(wall[kt + 2, jt, it], want2, rtol=1e-6)


N, NY, BAND, LID = 240, 8, 24.0, 200.0


def _flow(slope: float, partial: bool, closure: str, dz: float = 1.0, rise0: float = 0.0, floor=None) -> dict:
    """Over the middle of a 240 m plane, the lid 200 m over its top: the step-period ripple of the 2 m speed, the u* of
    the solve's own surface stress, the log law's from its 10 m speed, and the domain's flux balance. `rise0` and
    `floor` stand flat ground at a fraction of a cell over the floor, so its first cells are cut."""
    rise = slope * N + (1.0 if floor is not None else 0.0)
    scene, g = _scene(slope, partial, N, NY, dz, rise + BAND, rise + LID, rise0, floor)
    cfg = CONFIGS[closure]
    res = solve(scene, checks.profile(), checks.WEST, cfg)
    mid = (g.xc > 0.5 * N) & (g.xc < 0.85 * N)
    xs = scene.origin[0] + g.xc[mid]
    y = np.full(len(xs), scene.origin[1] + 0.5 * NY)
    zt = scene.terrain[NY // 2][mid]
    sp = {}
    for h in (2.0, 10.0):                    # h normal to the slope; the speed includes w
        v, _ = view.trilinear_fluid(res.velocity, scene, xs, y, scene.origin[2] + zt + h * math.sqrt(1 + slope ** 2))
        sp[h] = np.sqrt((v ** 2).sum(axis=0))
    trend = np.polyval(np.polyfit(xs, sp[2.0], 2), xs)
    wall = Model(scene, checks.profile(), checks.WEST, SolverConfig(closure=cfg.closure, device="cpu")).wall.numpy()
    speed2 = (np.asarray(res.velocity) ** 2).sum(axis=0)
    tau = (wall * speed2 * np.diff(g.zf)[:, None, None]).sum(axis=0)[:, mid].mean(axis=0) / math.sqrt(1 + slope ** 2)
    z0 = float(np.asarray(scene.z0)[NY // 2, N // 2])
    return {"ripple": float(np.max(np.abs(sp[2.0] - trend)) / np.mean(sp[2.0])),
            "ustar": float(np.sqrt(np.mean(tau))),
            "ustar_log": float(KAPPA * np.mean(sp[10.0]) / math.log(10.0 / z0)),
            "balance": abs(res.flux_in_m3_s - res.flux_out_m3_s) / res.flux_in_m3_s}


@pytest.mark.parametrize("closure", sorted(CONFIGS))
@pytest.mark.parametrize("rise0", [0.0, 0.1, 0.5, 0.8])
def test_flat_ground_in_cut_cells_carries_the_log_law_s_stress(rise0, closure):
    f = _flow(0.0, True, closure, rise0=rise0, floor=99.0)
    off = f["ustar"] / f["ustar_log"] - 1.0
    print(f"FLAT open {1.0 - rise0:g} {closure}: u* {f['ustar']:.4f} vs log law {f['ustar_log']:.4f} ({off:+.4f})")
    assert f["balance"] < 1e-6, "the open faces carry the flux: what enters leaves"
    assert abs(off) < USTAR_MAX


@pytest.mark.parametrize("closure", sorted(CONFIGS))
@pytest.mark.parametrize("slope", [0.06, 0.15])
def test_a_slope_s_stress_holds_as_its_cells_shrink_and_it_does_not_ripple(slope, closure):
    coarse, fine = _flow(slope, True, closure), _flow(slope, True, closure, dz=0.25)
    stair = _flow(slope, False, closure)
    held = coarse["ustar"] / fine["ustar"] - 1.0
    print(f"SLOPE {slope:g} {closure}: u* {coarse['ustar']:.4f} at 1 m, {fine['ustar']:.4f} at 0.25 m ({held:+.4f}); "
          f"against the 10 m log law {coarse['ustar'] / coarse['ustar_log'] - 1:+.4f} (the level air, (f)); ripple "
          f"{coarse['ripple']:.4f}, whole cubes {stair['ripple']:.4f}")
    assert coarse["balance"] < 1e-6 and fine["balance"] < 1e-6, "the open faces carry the flux: what enters leaves"
    assert abs(held) < CONVERGE_MAX
    assert coarse["ripple"] < RIPPLE_MAX
    assert stair["ripple"] > RIPPLE_MAX, "the staircase ripples (the test has teeth)"


def _tilt(v: torch.Tensor, s: float) -> torch.Tensor:
    """(3, ...) horizontal (u, 0, 0) velocities turned onto a slope s = dz/dx, speed kept."""
    c = 1.0 / math.sqrt(1.0 + s * s)
    return torch.stack([v[0] * c, torch.zeros_like(v[0]), v[0] * s * c])


@lru_cache(maxsize=None)
def _rotated(slope: float) -> float:
    """The plane's surface u* under a slope-parallel stream: the model built as production builds it (k-l, the
    pressure drive, the canopy inflow), then its sides, top, start and drive turned onto the slope, each carrying the flat
    equilibrium column's speed at its height above the local ground. Flat ground rotated, as near as a level lid allows."""
    rise = slope * N
    scene, g = _scene(slope, True, N, NY, 1.0, rise + BAND, rise + LID)
    m = Model(scene, checks.profile(), checks.WEST, CONFIGS["k-l"])
    if slope:
        col = m.column
        hh = np.concatenate([[0.0], col["h"], [col["h_top"]]])
        uu = np.concatenate([[0.0], col["u"], [col["u_top"]]])
        for k in ("west", "east", "south", "north"):
            m.bc[k] = _tilt(m.bc[k], slope)
        m.initial = _tilt(m.initial, slope)
        ht = float(g.zf[-1]) - np.asarray(scene.terrain)
        ut = torch.as_tensor(np.interp(np.clip(ht, 0, hh[-1]), hh, uu), dtype=m.sink.dtype, device=m.sink.device)
        m.bc["top"] = _tilt(torch.stack([ut, torch.zeros_like(ut), torch.zeros_like(ut)]), slope)
        c, b = 1.0 / math.sqrt(1.0 + slope * slope), float(m.body_vec[0])
        m.body_vec = torch.tensor([b * c, 0.0, b * slope * c], dtype=m.body_vec.dtype, device=m.body_vec.device)
    res = m.run()
    mid = (g.xc > 0.5 * N) & (g.xc < 0.85 * N)
    speed2 = (np.asarray(res.velocity) ** 2).sum(axis=0)
    tau = (m.wall.cpu().numpy() * speed2 * np.diff(g.zf)[:, None, None]).sum(axis=0)[:, mid].mean(axis=0)
    return float(np.sqrt(np.mean(tau / math.sqrt(1 + slope ** 2))))


@pytest.mark.parametrize("slope", [0.06, 0.15])
def test_a_slope_under_a_slope_parallel_stream_carries_flat_ground_s_stress(slope):
    """Under level air the slopes' u* stands +18 and +43 % over the 10 m log law's (the box, not the ground). Turned onto
    the slope, the same model's u* is flat ground's within USTAR_MAX: 2.2 and 4.2 % (2026-10-01)."""
    flat, rot = _rotated(0.0), _rotated(slope)
    print(f"ROTATED {slope:g}: u* {rot:.4f} against flat ground's {flat:.4f} ({rot / flat - 1:+.4f})")
    assert abs(rot / flat - 1.0) < USTAR_MAX
