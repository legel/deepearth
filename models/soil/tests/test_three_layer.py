"""Three soil layers where the root zone runs deeper than the surface layer, and the understory grass under a savanna's
crowns. The top half-meter is its own store: it fills first, dries first, is shared by the crowns' roots and is what
SOIL MOISTURE reads for every cover; a shallow cover is the two-layer model unchanged."""
import numpy as np
import torch

import balance as B
import phenology as P

AIR = (25.0, 1.0, 101.0, 350.0)          # t degC, ea kPa, pa kPa, lw W m-2: a dry summer hour


def _cells(z_r, canopy, three=True, kcb=0.9):
    n = len(z_r)
    f = lambda v: torch.full((n,), float(v))  # noqa: E731
    k = B.Cells(alpha=f(0.2), fc=f(0.30), wp=f(0.13), theta_sat=f(0.45), theta_r=f(0.05), ksat=f(10.0), lam=f(0.3),
                psi_f=f(100.0), z_r=torch.as_tensor(np.asarray(z_r, np.float64), dtype=torch.float32), kcb=f(kcb),
                kc_max=f(1.2), rew=f(9.0), f_ew=f(0.3), s_max=f(1.0), no_soil=torch.zeros(n, dtype=torch.bool),
                perv=f(1.0), roof=torch.zeros(n, dtype=torch.bool))
    return B.three_layers(k, np.asarray(canopy)) if three else k


def _state(n, theta, three=True, lanes=1):
    f = lambda v: torch.full((lanes, n), float(v))  # noqa: E731
    return B.State(c=f(0), w=f(0), theta1=f(theta), theta2=f(theta), F=f(0), dry=f(24), theta3=f(theta) if three else None)


def _dry_hours(s, k, hours, roots=None):
    rs, u2 = torch.full_like(s.c, 600.0), torch.full_like(s.c, 2.0)
    for _ in range(hours):
        flux = B.local_step(s, k, rs, u2, *AIR)
        if roots is not None:
            flux.update(B.root_share_step(s, k, flux, roots))
    return s


def water_mm(s, k):
    return s.c + s.w + 1000.0 * (s.theta1 * B.ZE + s.theta2 * (k.z_s - B.ZE) + s.theta3 * (k.z_r - k.z_s))


def test_root_fractions_are_uniform_in_a_cover_depth_and_jacksons_below_it():
    zs, r1, r2, r3 = B.root_fractions(np.array([0.5, 0.3, 2.82, 1.79]), np.array([False, False, True, True]))
    np.testing.assert_allclose(zs, [0.5, 0.3, 0.5, 0.5])
    np.testing.assert_allclose(r1[:2], [0.2, 1 / 3], rtol=1e-9)
    assert r3[0] == 0.0 and r3[1] == 0.0, "a cover depth at or above Z_S has no layer 3"
    np.testing.assert_allclose(r1 + r2 + r3, 1.0, rtol=1e-9)
    y = lambda d: 1 - 0.966 ** (100 * d)  # noqa: E731
    np.testing.assert_allclose(r1[2], y(0.1) / y(2.82), rtol=1e-9)
    assert r2[2] > r3[2] > 0.1, "most of an oak's roots in the top half-meter, a real share below it"


def test_a_shallow_cover_is_the_two_layer_model_unchanged():
    z = [0.5, 0.3, 0.5]
    k2, k3 = _cells(z, [False] * 3, three=False), _cells(z, [False] * 3, three=True)
    s2, s3 = _state(3, 0.28, three=False), _state(3, 0.28, three=True)
    s2.w[:] = 3.0
    s3.w[:] = 3.0
    rs, u2 = torch.full((1, 3), 400.0), torch.full((1, 3), 2.0)
    for _ in range(48):
        f2 = B.local_step(s2, k2, rs, u2, *AIR)
        f3 = B.local_step(s3, k3, rs, u2, *AIR)
        for key in ("aet", "drain", "t1", "t2", "es"):
            np.testing.assert_allclose(f3[key].numpy(), f2[key].numpy(), atol=1e-6, err_msg=key)
    np.testing.assert_allclose(s3.theta1.numpy(), s2.theta1.numpy(), atol=1e-6)
    np.testing.assert_allclose(s3.theta2.numpy(), s2.theta2.numpy(), atol=1e-6)
    np.testing.assert_allclose(s3.theta3.numpy(), 0.28, atol=1e-7, err_msg="an empty layer 3 never moves")
    np.testing.assert_allclose(B.root_zone(s3, k3).numpy(), B.root_zone(s2, k2).numpy(), atol=1e-6)


def test_a_deep_root_zone_dries_from_the_top_and_its_deep_store_carries_the_dry_season():
    k = _cells([2.82], [True])
    s = _dry_hours(_state(1, 0.30), k, 24 * 40)
    rel = lambda th: (float(th) - 0.13) / 0.17  # noqa: E731
    assert rel(s.theta2[0, 0]) < rel(s.theta3[0, 0]) - 0.1
    one = _dry_hours(_state(1, 0.30, three=False), _cells([2.82], [True], three=False), 24 * 40)
    np.testing.assert_allclose(float(B.column(s, k)[0, 0]), float(B.column(one, _cells([2.82], [True], False))[0, 0]),
                               atol=0.02, err_msg="the column's water stays close to the one-store model's")


def test_water_soaks_in_from_the_top_and_every_millimeter_is_kept():
    k = _cells([2.82, 0.5], [True, False])
    s = _state(2, 0.14)
    s.w[:] = 120.0
    before = water_mm(s, k)
    rs, u2 = torch.full((1, 2), 0.0), torch.full((1, 2), 1.0)
    out = 0.0
    for _ in range(24):
        f = B.local_step(s, k, rs, u2, 10.0, 1.2, 101.0, 300.0)
        out = out + f["aet"] + f["drain"]
    np.testing.assert_allclose((before - water_mm(s, k) - out).numpy(), 0.0, atol=1e-3)
    assert float(s.theta2[0, 0]) > float(s.theta3[0, 0]), "the surface layer fills before the deep store"


def test_rain_soaks_into_three_layers_from_the_top():
    k = _cells([2.82], [True])
    s = _state(1, 0.2)
    s.F = torch.zeros(1, 1)
    before = float(water_mm(s, k).sum())
    B._soak(s, k, torch.tensor([[200.0]]))
    np.testing.assert_allclose(float(water_mm(s, k).sum()) - before, 200.0, rtol=1e-5)
    assert abs(float(s.theta1[0, 0]) - 0.45) < 1e-6 and abs(float(s.theta2[0, 0]) - 0.45) < 1e-6
    assert float(s.theta3[0, 0]) > 0.2


def test_the_crown_edge_grades_where_one_well_mixed_store_stepped():
    """A strip of 20 crown cells (2.82 m) beside 20 grass cells (0.5 m) at 0.5 m, roots shared over 3 m, 30 dry days:
    SOIL MOISTURE (0 to z_s) across the drip line steps a third or less of what the two-layer reading stepped."""
    n = 40
    canopy = np.arange(n) < 20
    z = np.where(canopy, 2.82, 0.5)
    steps = {}
    for three in (False, True):
        k = _cells(z, canopy, three=three)
        roots = B.RootShare((1, n), 0.5, canopy, np.ones(n, bool), "cpu", radius_m=3.0)
        s = _dry_hours(_state(n, 0.30, three=three), k, 24 * 30, roots=roots)
        zr = k.z_s if three else torch.as_tensor(np.where(canopy, 1.0, 0.5), dtype=torch.float32)
        th = ((s.theta1[0] * B.ZE + s.theta2[0] * (zr - B.ZE)) / zr).numpy()
        steps[three] = abs(float(th[19] - th[20]))
    assert steps[True] <= steps[False] / 3.0, steps


def test_the_understory_follows_the_sites_own_grass_cover_and_none_under_a_forest():
    can = np.array([True, True, False, True])
    perv = np.array([True, False, True, True])
    u, rec = P.understory_share({"dominant": "herbaceous", "tree_cover": 14.5, "nontree_cover": 73.0}, can, perv)
    np.testing.assert_allclose(u, [73.0 / 85.5, 0.0, 0.0, 73.0 / 85.5], rtol=1e-9)
    assert rec["share"] == round(73.0 / 85.5, 3)
    u, rec = P.understory_share({"dominant": "woody", "tree_cover": 72.5, "nontree_cover": 20.5}, can, perv)
    assert not u.any(), "Harvard Forest grows no grass under its crowns"
    np.testing.assert_allclose(P.understory_kcb(torch.tensor([0.8, 0.0])).numpy(), [0.8 * 0.85, 0.0], rtol=1e-6)
    np.testing.assert_allclose(P.understory_kcb(torch.tensor([0.8]), 0.0).numpy(), [0.8 * P.KC_MIN], rtol=1e-6)


def test_an_understory_draws_the_surface_layers_only_and_keeps_the_water():
    k = _cells([2.82, 2.82], [True, True])
    k.kcb_u = torch.tensor([0.0, 0.6])
    s = _state(2, 0.30)
    before = water_mm(s, k)
    rs, u2 = torch.full((1, 2), 600.0), torch.full((1, 2), 2.0)
    out = 0.0
    for _ in range(24 * 10):
        f = B.local_step(s, k, rs, u2, *AIR)
        out = out + f["aet"] + f["drain"]
        assert float((f["tr"] / torch.clamp(f["et0"] - f["ec"] - f["ew"], min=1e-9)).max()) <= 1.2 + 1e-5
    np.testing.assert_allclose((before - water_mm(s, k) - out).numpy(), 0.0, atol=1e-3)
    assert float(s.theta2[0, 1]) < float(s.theta2[0, 0]) - 0.01, "the grass under the crown dries the surface layer"
    assert abs(float(s.theta3[0, 1]) - float(s.theta3[0, 0])) < 0.01, "and leaves the deep store to the tree"
