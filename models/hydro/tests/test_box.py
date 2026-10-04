"""A site that follows its ordered polygon: the DTM cropped to its fetch box on the cell lattice, even on both
sides, the classes cut outside the box, and the units cut along the box's lines at every pass's cell."""

import argparse

import numpy as np
from affine import Affine

import cli
import parcel
import units


def test_the_box_window_is_snapped_out_to_the_lattice_and_even():
    z = np.arange(100 * 100, dtype=np.float32).reshape(100, 100)
    t = Affine(0.5, 0.0, -25.0, 0.0, -0.5, 25.0)
    fetch = (-10.2, -5.0, 7.3, 4.1)
    w, tw = parcel.window_box(z, t, fetch, 0.5)
    assert w.shape == (20, 36), "19 rows made even by one more to the south"
    assert (tw.c, tw.f) == (-10.5, 4.5) and w[0, 0] == z[41, 29]
    out = parcel.outside_box(w.shape, tw, fetch)
    assert (~out).sum() == 35 * 18 and out[0].all() and out[:, 0].all() and out[-1].all()
    far, _ = parcel.window_box(z, t, (20.0, 20.0, 30.0, 30.0), 0.5)
    assert far.shape == (20, 20) and np.isnan(far[:10]).all() and np.isfinite(far[10:, :10]).all()


def test_the_box_split_cuts_the_same_lines_whatever_the_grid():
    fetch = (-10.0, -6.0, 30.0, 14.0)
    s = units.split_box(fetch, 2, 2, 5.0)
    assert [u["core"][1] for u in s[:1]] == [10.0] and s[0]["core"][2] == 4.0
    assert s[0]["window"] == (-units.OUTER_M, 15.0, -1.0, units.OUTER_M)
    assert units.seam_lines_box(s, fetch) == ([5.0, 15.0], [-1.0, 9.0])
    for dx, x0, y1, nc, nr in ((0.5, -10.0, 14.0, 80, 40), (1.0, -11.0, 15.0, 42, 22)):
        t = Affine(dx, 0.0, x0, 0.0, -dx, y1)
        seen = np.zeros((nr, nc), int)
        for u in s:
            r0, r1, c0, c1 = units.cells(t, (nr, nc), u["core"])
            w0, w1, v0, v1 = units.cells(t, (nr, nc), u["window"])
            assert w0 <= r0 and r1 <= w1 and v0 <= c0 and c1 <= v1
            seen[r0:r1, c0:c1] += 1
        assert (seen == 1).all(), f"the cores cover the {dx:g} m grid once"


def test_a_box_split_matches_the_disc_split_where_the_lines_agree():
    n, dx = 48, 1.0
    t = Affine(dx, 0.0, 0.0, 0.0, -dx, n * dx)
    disc = units.split((24.0, 24.0), 24.0, 2, 2, 10.0)
    box = units.split_box((0.5, 0.5, 47.5, 47.5), 2, 2, 10.0)
    for a, b in zip(disc, box):
        for k in ("core", "window"):
            assert units.cells(t, (n, n), a[k]) == units.cells(t, (n, n), b[k])
    assert units.seam_lines(disc, (24.0, 24.0), 24.0) == units.seam_lines_box(box, (0.5, 0.5, 47.5, 47.5))


def test_a_box_run_splits_its_fetch_box():
    fetch = [-10.0, -6.0, 30.0, 14.0]
    args = argparse.Namespace(units="2x2", overlap_m=5.0)
    specs, lines = cli._units(None, args, {"box_scene_m": {"fetch": fetch, "display": [0, 0, 1, 1]}})
    assert lines == tuple(fetch) and specs == units.split_box(tuple(fetch), 2, 2, 5.0)
