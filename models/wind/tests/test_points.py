"""Survey points read from the 3D solve (points.py): exact for uniform and linear fields, a wall face read 2 m out,
no contour lines along a stepped slope, no jumps across a crown's edge, and every point owned by one block. No
network, no site data, no solve."""

import numpy as np
from scipy.spatial import cKDTree

import cli
import domain
import points as P
import view
from domain import Grid


def _jump_share(xyz: np.ndarray, v: np.ndarray, k: int = 8, radius: float = 1.0, rel: float = 0.3) -> float:
    """The continuity measure (simulation exposure.continuity): neighbour pairs within `radius` whose values differ by
    more than `rel` of their mean."""
    t = cKDTree(xyz)
    d, j = t.query(xyz, k=k + 1, distance_upper_bound=radius)
    i = np.repeat(np.arange(len(xyz)), k + 1).reshape(len(xyz), k + 1)
    keep = np.isfinite(d) & (j < len(xyz)) & (i < j)
    a, b = v[i[keep]], v[j[keep]]
    return float((np.abs(a - b) > rel * 0.5 * (np.abs(a) + np.abs(b))).mean())


def _cell_centres(scene):
    g, (ox, oy, _) = scene.grid, scene.origin
    return g.zc[:, None, None], (oy + g.yc)[None, :, None], (ox + g.xc)[None, None, :]


def test_a_uniform_field_is_returned_exactly_at_every_point():
    g = Grid.uniform(1.0, 40, 40, 24)
    scene = domain.box(g, 8.0, 9.0)
    vel = np.stack([np.full(g.shape, 1.3), np.full(g.shape, -0.7), np.full(g.shape, 0.2)])
    rng = np.random.default_rng(1)
    x, y = rng.uniform(-18, 18, 3000), rng.uniform(-18, 18, 3000)
    h = rng.uniform(0.0, 14.0, 3000)
    placed = P.place(scene, x, y, h)
    out = P.sample(vel, placed)
    assert placed.resolved.all()
    np.testing.assert_allclose(out, np.array([[1.3], [-0.7], [0.2]]) * np.ones((1, 3000)), rtol=1e-6)


def test_a_linear_field_is_returned_exactly_at_fluid_points():
    g = Grid.stretched(1.0, 40, 36, 30, 1.0, 1.05)
    scene = domain.flat(g)
    z, y, x = _cell_centres(scene)
    f = lambda xx, yy, zz: 0.3 * zz - 0.2 * yy + 0.1 * xx  # noqa: E731
    vel = np.stack([f(x, y, z) + 0 * z, 2 * f(x, y, z) + 0 * z, -f(x, y, z) + 0 * z])
    rng = np.random.default_rng(2)
    px, py = rng.uniform(-17, 17, 2000), rng.uniform(-15, 15, 2000)
    ph = rng.uniform(P.CLEARANCE_M, 20.0, 2000)          # in the fluid, off the ground by at least the clearance
    out = P.sample(vel, P.place(scene, px, py, ph))
    want = f(px, py, ph)
    np.testing.assert_allclose(out[0], want, atol=2e-5)
    np.testing.assert_allclose(out[1], 2 * want, atol=4e-5)
    np.testing.assert_allclose(out[2], -want, atol=2e-5)


def test_a_point_on_the_ground_is_read_at_the_clearance_above_it():
    g = Grid.uniform(1.0, 30, 30, 20)
    scene = domain.flat(g)
    z, y, x = _cell_centres(scene)
    vel = np.stack([np.log1p(z / 0.03) + 0 * x + 0 * y] * 3)
    placed = P.place(scene, np.array([0.3, -4.2]), np.array([1.1, 2.6]), np.array([0.0, 0.4]))
    out = P.sample(vel, placed)
    np.testing.assert_allclose(placed.z_eval, P.CLEARANCE_M)
    assert placed.raised.all()
    lo, hi = np.log1p(1.5 / 0.03), np.log1p(2.5 / 0.03)
    np.testing.assert_allclose(out[0], 0.5 * (lo + hi), rtol=1e-6)


def test_a_wall_face_matches_its_fluid_neighbour_two_metres_out():
    """A facade return on a tower's east face, and one 0.3 m inside it (a footprint rounded to the cell), both read
    the field 2 m out from the wall at their own height; the one inside records how far it searched."""
    g = Grid.uniform(1.0, 40, 40, 24)
    scene = domain.box(g, 10.0, 12.0)                    # faces at x = +-5 about the origin, 12 m tall
    z, y, x = _cell_centres(scene)
    f = lambda xx, yy, zz: 0.4 * xx + 0.05 * yy + 0.2 * zz  # noqa: E731
    vel = np.stack([f(x, y, z) + 0 * z] * 3)
    vel[:, scene.solid] = 0.0                            # a solve holds nothing in a solid cell
    placed = P.place(scene, np.array([5.0, 4.7]), np.array([0.3, 0.3]), np.array([6.0, 6.0]))
    out = P.sample(vel, placed)
    ref, ok = view.trilinear(vel[:1], scene, np.array([7.0]), np.array([0.3]), np.array([6.0]))
    assert ok.all()
    np.testing.assert_allclose(out[0], ref[0, 0], rtol=1e-6)
    np.testing.assert_allclose(out[0], f(7.0, 0.3, 6.0), rtol=1e-6)
    assert placed.search_m[0] == 0 and 0.29 < placed.search_m[1] < 0.31
    assert (placed.pushed_m > 1.9).all()


def test_a_stepped_slope_draws_no_contour_lines():
    """Ground returns on a 15 % slope voxelised in 1 m cubes. The solve's flow near the ground follows each column's
    own top, so reading every column at the same height above its own top is continuous. A plain trilinear at the
    points' heights over the true ground would draw a line at every step, and it does here (the test has teeth)."""
    g = Grid.uniform(1.0, 60, 30, 30)
    scene = domain.flat(g)
    X = scene.origin[0] + g.xc
    zt = np.broadcast_to(0.15 * (X - X.min()), (g.ny, g.nx)).copy()
    scene.solid[:] = g.zc[:, None, None] < zt[None]
    scene.terrain, scene.top = zt, zt
    kf, T = P.column_tops(scene)
    z = g.zc[:, None, None] - T[None]                    # height above each column's own solid top
    prof = np.log1p(np.clip(z, 0.0, None) / 0.03)
    vel = np.stack([prof, 0.1 * prof, 0.0 * prof])
    vel[:, scene.solid] = 0.0
    rng = np.random.default_rng(3)
    px, py = rng.uniform(-27, 27, 20000), rng.uniform(-12, 12, 20000)
    ph = rng.uniform(0.0, 0.3, 20000)
    out = P.sample(vel, P.place(scene, px, py, ph))
    ground = P.bilinear(zt, scene, px, py)
    xyz = np.c_[px, py, ground + ph]
    assert _jump_share(xyz, np.hypot(out[0], out[1]), rel=0.02) == 0.0
    naive, _ = view.trilinear_fluid(vel[:1], scene, px, py, scene.origin[2] + ground + P.CLEARANCE_M)
    assert _jump_share(xyz, np.abs(naive[0]), rel=0.02) > 0.001, "absolute heights on the staircase do jump"


def test_a_cut_slope_reads_every_ground_return_at_the_clearance_over_the_true_ground():
    """The ground's cut cells stay open (domain.PARTIAL_GROUND), so the solid top under a cut cell is the step below the
    ground. Clearance measured from that step read ground returns 1 to 2 m over the true ground with its height modulo
    the cell: bands along the contours of every sloped forest site (2026-10-03). Measured from the true ground, every
    return is read at the clearance over the ground under it."""
    g = Grid.uniform(1.0, 60, 30, 30)
    scene = domain.flat(g)
    X = scene.origin[0] + g.xc
    zt = np.broadcast_to(0.15 * (X - X.min()) + 0.37, (g.ny, g.nx)).copy()
    scene.solid[:] = g.zf[1:, None, None] <= zt[None]          # whole cells under the ground; its cut cells open
    scene.terrain, scene.top, scene.partial = zt, zt, True
    rng = np.random.default_rng(5)
    px, py = rng.uniform(-27, 27, 5000), rng.uniform(-12, 12, 5000)
    ground = P.bilinear(zt, scene, px, py)
    over = P.place(scene, px, py, np.zeros(5000)).z_eval - ground
    np.testing.assert_allclose(over, P.CLEARANCE_M, atol=1e-6)
    scene.partial = False                                      # the stepped tops as the surface: the test has teeth
    stepped = P.place(scene, px, py, np.zeros(5000)).z_eval - ground
    assert np.ptp(stepped) > 0.8 and stepped.min() < P.CLEARANCE_M - 0.5


def test_no_jumps_across_a_crowns_edge():
    """Returns over a porous crown (its top and sides) and the open ground around it, in a smooth field: neighbours
    within 1 m never differ by more than 30 %. The old rules read a low crown return at its own height and the
    ground beside it 2 m up."""
    g = Grid.uniform(1.0, 40, 40, 24)
    scene = domain.porous_block(g, 10.0, 3.0)
    z, y, x = _cell_centres(scene)
    r = np.hypot(x, y)
    vel = np.stack([np.log1p(z / 0.5) * (1.0 - 0.3 * np.exp(-(r / 6.0) ** 2))] * 2 + [np.zeros(g.shape)])
    rng = np.random.default_rng(4)
    n = 30000
    px, py = rng.uniform(-12, 12, n), rng.uniform(-12, 12, n)
    in_crown = (np.abs(px) <= 5) & (np.abs(py) <= 5)
    ph = np.where(in_crown, rng.uniform(0.0, 10.0, n), rng.uniform(0.0, 0.2, n))
    out = P.sample(vel, P.place(scene, px, py, ph))
    assert np.isfinite(out).all()
    assert _jump_share(np.c_[px, py, ph], np.hypot(out[0], out[1])) == 0.0


def test_every_point_is_owned_by_one_block_and_read_there_as_in_the_whole_scene():
    g = Grid.uniform(1.0, 48, 40, 20)
    scene = domain.box(g, 8.0, 9.0, centre=(20.0, 22.0))
    z, y, x = _cell_centres(scene)
    vel = np.stack([np.log1p(z / 0.03) * (1 + 0.01 * x) + 0 * y, 0.02 * y + 0 * z + 0 * x, 0.01 * x + 0 * z + 0 * y])
    vel[:, scene.solid] = 0.0
    rng = np.random.default_rng(5)
    px, py = rng.uniform(-24, 24, 4000), rng.uniform(-20, 20, 4000)
    ph = rng.uniform(0.0, 12.0, 4000)
    whole = P.sample(vel, P.place(scene, px, py, ph))
    specs = cli._split(scene, 2, 2, 8.0)
    seen = np.zeros(len(px), int)
    got = np.full_like(whole, np.nan)
    for s in specs:
        mine = P.core_points(scene, px, py, s["core"])
        seen[mine] += 1
        w0, w1, v0, v1 = s["window"]
        unit = cli._crop(scene, s["window"])
        got[:, mine] = P.sample(vel[:, :, w0:w1, v0:v1], P.place(unit, px[mine], py[mine], ph[mine]))
    assert (seen == 1).all()
    np.testing.assert_allclose(got, whole, rtol=1e-6, atol=1e-7)


def test_the_writer_streams_heading_major_and_leaves_missing_blocks_empty(tmp_path):
    w = P.Writer(tmp_path, [0.0, 22.5], 5)
    w.put(0.0, np.arange(15, dtype=np.float32).reshape(3, 5))
    w.put(22.5, np.ones((3, 2), np.float32), np.array([1, 3]))
    meta = w.close({"points": 5}, np.zeros(5, np.uint8), {})
    a = np.fromfile(tmp_path / P.Writer.NAME, "<f2").reshape(2, 5, 3)
    np.testing.assert_array_equal(a[0].T, np.arange(15).reshape(3, 5))
    assert np.isnan(a[1, [0, 2, 4]]).all() and (a[1, [1, 3]] == 1).all()
    assert meta["shape"] == [2, 5, 3] and "w" in meta["speed"]


def test_the_points_file_round_trips(tmp_path):
    x, y, z, h = (np.arange(4, dtype=np.float32) + k for k in range(4))
    got = P.read_points(P.write_points(tmp_path / "p.bin", x, y, z, h))
    for a, b in zip(got, (x, y, z, h)):
        np.testing.assert_array_equal(a, b)


def test_datum_placement_keeps_a_roof_return_on_its_roof_where_the_dtms_differ():
    """A roof return whose height above the survey's bare earth is 5 m short (the survey's DTM under the building is
    5 m higher than the solver's) stands inside the roof by that placement and on it by its own z."""
    g = Grid.uniform(1.0, 40, 40, 24)
    scene = domain.box(g, 10.0, 12.0)
    x, y, z, h = np.array([0.3, 12.0]), np.array([0.2, 12.0]), np.array([12.0, 3.0]), np.array([7.0, 3.0])
    z_off = z + 0.4                                      # the survey's frame 0.4 m above the solver's
    off = P.datum_offset(scene, x[1:], y[1:], z_off[1:], h[1:])
    assert abs(off - 0.4) < 1e-9
    by_h = P.place(scene, x, y, h)
    by_z = P.place(scene, x, y, h, z=z_off - off)
    assert 4.9 < by_h.search_m[0] < 5.1 and by_z.search_m[0] == 0
    np.testing.assert_allclose(by_z.z_eval[0], 12.0 + P.CLEARANCE_M)


def test_a_roof_wall_edge_reads_continuously_along_the_returns_own_normals():
    """Dense returns over a tower's roof, round its edge and down its wall, their normals turning from up to out over
    1.5 m of the profile as a surface fit's do. Along those normals the reads meet round the edge; by the vertical
    rule for the roof and the horizontal rule for the wall they stand 2 m apart and jump (the test has teeth)."""
    g = Grid.uniform(1.0, 40, 40, 24)
    scene = domain.box(g, 10.0, 12.0)                    # east face at x = 5, roof at 12 m
    z, y, x = _cell_centres(scene)
    vel = np.stack([np.exp(0.35 * (z - 12.0)) + 0 * x + 0 * y] * 2 + [np.zeros(g.shape)])
    vel[:, scene.solid] = 0.0
    step = 0.05
    roof_x = np.arange(2.0, 5.0, step)
    wall_z = np.arange(12.0 - step, 8.0, -step)
    px = np.r_[roof_x, np.full(len(wall_z), 5.0)]
    pz = np.r_[np.full(len(roof_x), 12.0), wall_z]
    s = np.r_[roof_x - 5.0, 12.0 - wall_z]               # distance along the profile from the edge
    th = np.radians(90.0 * np.clip((s + 0.75) / 1.5, 0.0, 1.0))
    nrm = np.c_[np.sin(th), np.zeros_like(th), np.cos(th)]
    py = np.full(len(px), 0.3)
    xyz = np.c_[px, py, pz]
    new = P.place(scene, px, py, pz, z=pz, normals=nrm)
    old = P.place(scene, px, py, pz, z=pz)
    speed = lambda pl: np.hypot(*P.sample(vel, pl)[:2])  # noqa: E731
    assert np.isfinite(speed(new)).all() and (new.offset_m > 1.9).all()
    assert _jump_share(xyz, speed(new)) == 0.0
    assert _jump_share(xyz, speed(old)) > 0.0, "the two faces' rules jump at the edge"


def test_the_solids_own_normal_rounds_the_edge_and_follows_a_stepped_slope():
    """The default normal is the solve's own (points.GeometricNormals): no survey normal needed, and no noise. Round a
    tower's roof-wall edge and along a 15 % slope in 1 m cubes, neighbours never jump."""
    g = Grid.uniform(1.0, 40, 40, 24)
    scene = domain.box(g, 10.0, 12.0)
    z, y, x = _cell_centres(scene)
    vel = np.stack([np.exp(0.35 * (z - 12.0)) + 0 * x + 0 * y] * 2 + [np.zeros(g.shape)])
    vel[:, scene.solid] = 0.0
    step = 0.05
    roof_x, wall_z = np.arange(2.0, 5.0, step), np.arange(12.0 - step, 8.0, -step)
    px = np.r_[roof_x, np.full(len(wall_z), 5.0)]
    pz = np.r_[np.full(len(roof_x), 12.0), wall_z]
    py = np.full(len(px), 0.3)
    geo = P.place(scene, px, py, pz, z=pz, normals="geometric")
    sp = np.hypot(*P.sample(vel, geo)[:2])
    assert np.isfinite(sp).all() and _jump_share(np.c_[px, py, pz], sp) == 0.0
    g2 = Grid.uniform(1.0, 60, 30, 30)
    slope = domain.flat(g2)
    X = slope.origin[0] + g2.xc
    zt = np.broadcast_to(0.15 * (X - X.min()), (g2.ny, g2.nx)).copy()
    slope.solid[:] = g2.zc[:, None, None] < zt[None]
    slope.terrain, slope.top = zt, zt
    _, T = P.column_tops(slope)
    prof = np.log1p(np.clip(g2.zc[:, None, None] - T[None], 0.0, None) / 0.03)
    v2 = np.stack([prof, 0.1 * prof, 0.0 * prof])
    v2[:, slope.solid] = 0.0
    rng = np.random.default_rng(6)
    qx, qy, qh = rng.uniform(-25, 25, 20000), rng.uniform(-10, 10, 20000), rng.uniform(0.0, 0.3, 20000)
    s2 = np.hypot(*P.sample(v2, P.place(slope, qx, qy, qh, normals="geometric"))[:2])
    assert _jump_share(np.c_[qx, qy, P.bilinear(zt, slope, qx, qy) + qh], s2, rel=0.02) == 0.0
