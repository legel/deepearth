"""A crown drinks from all the soil its lateral roots reach (balance.RootShare): mass kept, spread evenly over the soil in
reach, grass and shrubs on their own cell, no water drawn from paving; DROUGHT's deficit follows the same roots."""
import numpy as np
import torch

import balance as B


def _share(canopy, soil, dx=0.5, radius=2.0):
    return B.RootShare(canopy.shape, dx, canopy.ravel(), soil.ravel(), "cpu", radius_m=radius)


def _cells(n):
    f = lambda v: torch.full((n,), float(v))  # noqa: E731
    return B.Cells(alpha=f(0.2), fc=f(0.3), wp=f(0.1), theta_sat=f(0.45), theta_r=f(0.05), ksat=f(10), lam=f(0.3),
                   psi_f=f(100), z_r=f(1.0), kcb=f(1.0), kc_max=f(1.2), rew=f(9), f_ew=f(0.5), s_max=f(1),
                   no_soil=torch.zeros(n, dtype=torch.bool), perv=f(1), roof=torch.zeros(n, dtype=torch.bool))


def test_a_crowns_uptake_is_spread_evenly_over_the_soil_in_reach():
    shape = (30, 40)
    canopy = np.zeros(shape, bool)
    canopy[15, 20] = True
    sh = _share(canopy, np.ones(shape, bool))
    tr = torch.zeros(30 * 40, dtype=torch.float64)
    tr[15 * 40 + 20] = 1.0
    u = sh.uptake(tr).reshape(shape).numpy()
    yy, xx = np.mgrid[:30, :40]
    inside = (yy - 15) ** 2 + (xx - 20) ** 2 <= 16          # 2 m at 0.5 m cells: a disk of radius 4 cells
    np.testing.assert_allclose(u[inside], 1.0 / inside.sum(), rtol=1e-9)
    assert np.abs(u[~inside]).max() < 1e-12
    np.testing.assert_allclose(u.sum(), 1.0, rtol=1e-12)


def test_uptake_is_conserved_and_never_drawn_from_paving_and_grass_keeps_its_own():
    rng = np.random.default_rng(0)
    shape = (40, 50)
    canopy = rng.random(shape) < 0.5
    soil = rng.random(shape) < 0.8
    soil[:, :3] = False
    canopy[:, :3] = True                                     # a crown over paving beside open soil
    sh = _share(canopy, soil)
    tr = torch.as_tensor(rng.random((3, 40 * 50)) * soil.ravel(), dtype=torch.float64)
    u = sh.uptake(tr)
    np.testing.assert_allclose(u.sum(1).numpy(), tr.sum(1).numpy(), rtol=1e-10)
    assert (u[:, ~soil.ravel()].abs() < 1e-12).all(), "paving gives no water"
    grass = (~canopy & soil).ravel()
    lone = u - sh.soil * sh.conv(tr * sh.inv)
    np.testing.assert_allclose(lone[:, grass].numpy(), tr[:, grass].numpy(), rtol=1e-12)


def test_the_step_keeps_water_and_takes_off_the_crowns_aet_what_dry_soil_cannot_give():
    shape = (20, 20)
    n = 400
    canopy = np.zeros(shape, bool)
    canopy[8:12, 8:12] = True
    sh = _share(canopy, np.ones(shape, bool), radius=1.5)
    k = _cells(n)
    th = torch.full((1, n), 0.25)
    th[0, :200] = 0.1                                        # half the grid at wilting point
    s = B.State(c=torch.zeros(1, n), w=torch.zeros(1, n), theta1=th.clone(), theta2=th.clone(), F=torch.zeros(1, n),
                dry=torch.zeros(1, n))
    t1 = torch.zeros(1, n)
    t2 = torch.where(torch.as_tensor(canopy.ravel())[None], torch.tensor(2.0), torch.tensor(0.0))
    s.theta2 = s.theta2 - t2 / 900.0                         # what each crown's local step took from its own cell
    before = (s.theta1 * 100.0 + s.theta2 * 900.0).sum()
    out = B.root_share_step(s, k, {"t1": t1, "t2": t2, "aet": t2.clone()}, sh)
    after = (s.theta1 * 100.0 + s.theta2 * 900.0).sum()
    took = out["root_in"].sum()
    np.testing.assert_allclose(float(before + t2.sum() - took), float(after), rtol=1e-6)
    np.testing.assert_allclose(float(out["aet"].sum()), float(took), rtol=1e-5)
    assert float(out["root_unmet"].sum()) > 0, "the dry half's share was not given"
    assert (s.theta2 >= k.wp - 1e-6).all() and (s.theta1 >= k.wp - 1e-6).all()


def test_the_deficit_follows_the_soil_the_roots_draw_and_a_gap_edge_does_not_step():
    """A forest half and a gap half: crowns demand 4 mm and get 1, the gap's grass demands 2 and gets 2. Charged to each
    crown's own cell the deficit steps 3 to 0 at the drip line; charged to the soil the roots draw it ramps over the
    root reach, the site's total kept."""
    shape = (20, 40)
    n = 800
    canopy = np.zeros(shape, bool)
    canopy[:, :20] = True
    sh = _share(canopy, np.ones(shape, bool), radius=2.0)
    k = _cells(n)
    s = B.State(c=torch.zeros(n), w=torch.zeros(n), theta1=torch.full((n,), 0.2), theta2=torch.full((n,), 0.2),
                F=torch.zeros(n), dry=torch.zeros(n))
    can = torch.as_tensor(canopy.ravel())
    z = torch.zeros(n)
    flux = {"et0": torch.where(can, torch.tensor(4.0), torch.tensor(2.0)),
            "aet": torch.where(can, torch.tensor(1.0), torch.tensor(2.0)), "drain": z, "infil": z, "lost": z}
    own, shared = B.Year(n), B.Year(n)
    own.add(s, k, flux)
    shared.add(s, k, flux, roots=sh)
    a = own.cwd.reshape(shape).numpy()
    b = shared.cwd.reshape(shape).numpy()
    np.testing.assert_allclose(b.sum(), a.sum(), rtol=1e-5)
    assert abs(a[10, 19] - a[10, 20]) == 3.0, "the step at the drip line"
    row = b[10]
    assert np.all(np.abs(np.diff(row[12:28])) < 1.0), "no step over 1 mm between neighbors across the edge"
    assert row[5] > row[19] > row[24] > row[35] - 1e-9 and row[35] < 1e-6, "a ramp from the forest into the gap"
    np.testing.assert_allclose(shared.aet, own.aet)


def test_an_hour_with_roots_keeps_a_sites_water():
    shape = (10, 10)
    n = 100
    canopy = np.zeros(shape, bool)
    canopy[3:7, 3:7] = True
    sh = _share(canopy, np.ones(shape, bool), radius=1.0)
    k = _cells(n)
    k.z_r = torch.where(torch.as_tensor(canopy.ravel()), torch.tensor(2.0), torch.tensor(0.5))
    k = B.three_layers(k, canopy.ravel())
    s = B.State(c=torch.zeros(n), w=torch.zeros(n), theta1=torch.full((n,), 0.28), theta2=torch.full((n,), 0.28),
                F=torch.zeros(n), dry=torch.full((n,), 24.0), theta3=torch.full((n,), 0.28))

    def held():
        return float((1000.0 * (s.theta1 * B.ZE + s.theta2 * (k.z_s - B.ZE) + s.theta3 * (k.z_r - k.z_s))).sum())
    before, out = held(), 0.0
    for _ in range(48):
        f = B.hour(s, k, 0.0, None, torch.full((n,), 600.0), torch.full((n,), 2.0), 25.0, 1.0, 101.0, 350.0, roots=sh)
        out += float((f["aet"] + f["drain"]).sum())
    np.testing.assert_allclose(before - held(), out, rtol=1e-4)
