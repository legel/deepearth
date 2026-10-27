"""Clay inputs: band order, offset harmonisation, resampling, padding, time and location features."""

import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from inputs import (CLAY_BANDS, MEAN, STD, clay_stack, latlon_features, normalise, time_features, to_size,
                    upsample_r20)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "data" / "sentinel2"))
from geometry import Window, upsample_to  # noqa: E402
from store import to_us  # noqa: E402


def _chips(n=2):
    r10 = np.stack([np.full((30, 30), 1000 * (i + 1), np.uint16) for i in range(4)])[None].repeat(n, 0)
    r20 = np.stack([np.full((15, 15), 100 * (i + 1) + 5000, np.uint16) for i in range(6)])[None].repeat(n, 0)
    return r10, r20


def test_band_order_matches_clay_metadata():
    assert CLAY_BANDS == ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
    r10, r20 = _chips()
    up = upsample_r20(r20, [0, 0], [0, 0], [0, 0], [0, 0])
    x, nodata = clay_stack(r10, up, np.zeros(2, np.int32))
    expect = [1000, 2000, 3000, 5100, 5200, 5300, 4000, 5400, 5500, 5600]
    assert [float(x[0, c, 0, 0]) for c in range(10)] == expect
    assert not nodata.any()


def test_offset_harmonisation_and_nodata():
    r10, r20 = _chips()
    r10[1, :, 0, 0] = 0
    r20[1, :, 0, 0] = 0
    up = upsample_r20(r20, [0, 0], [0, 0], [0, 0], [0, 0])
    x, nodata = clay_stack(r10, up, np.array([0, -1000], np.int32))
    assert x[1, 0, 5, 5] == x[0, 0, 5, 5] - 1000
    assert nodata[1, 0, 0] and not nodata[0, 0, 0]
    np.testing.assert_allclose(normalise(x)[1, :, 0, 0], 0.0, atol=1e-6)   # no-data -> band mean


def test_upsample_r20_agrees_with_geometry():
    rng = np.random.default_rng(1)
    r20 = rng.integers(1, 10000, (1, 6, 15, 15)).astype(np.uint16)
    for dx, dy in ((0, 0), (10, 0), (0, -10), (-10, 10)):
        x10, y10, x20, y20 = 500000.0 + dx, 4000000.0 + dy, 500000.0, 4000000.0
        a = upsample_r20(r20, [x10], [y10], [x20], [y20])[0]
        s = Window(col=int(x20 / 20), row=0, size=15, res=20, x0=0.0, y0=y20)
        d = Window(col=int(x10 / 10), row=0, size=30, res=10, x0=0.0, y0=y10)
        assert (a == upsample_to(r20[0], s, d)).all()


@pytest.mark.parametrize("mode,side,gsd", [("reflect", 32, 10.0), ("edge", 32, 10.0), ("zero", 32, 10.0),
                                           ("bilinear", 32, 9.375), ("crop24", 24, 10.0)])
def test_sizes_are_patch_multiples(mode, side, gsd):
    x = np.random.default_rng(0).normal(size=(3, 10, 30, 30)).astype(np.float32)
    y, g = to_size(x, mode)
    assert y.shape == (3, 10, side, side) and side % 8 == 0 and g == gsd
    if mode == "reflect":
        assert (y[:, :, 1:31, 1:31] == x).all() and (y[:, :, 0, 1:31] == x[:, :, 1, :]).all()


def test_time_features_week_and_utc_hour():
    t = to_us("2020-07-01T10:30:00Z")
    f = time_features(np.array([t]))[0]
    week = datetime(2020, 7, 1, tzinfo=timezone.utc).isocalendar().week * 2 * math.pi / 52
    np.testing.assert_allclose(f, [math.sin(week), math.cos(week), math.sin(10 * 2 * math.pi / 24),
                                   math.cos(10 * 2 * math.pi / 24)], rtol=1e-6)


def test_latlon_features():
    f = latlon_features(42.5378, -72.1715, 3)
    assert f.shape == (3, 4)
    np.testing.assert_allclose(f[0], [math.sin(math.radians(42.5378)), math.cos(math.radians(42.5378)),
                                      math.sin(math.radians(-72.1715)), math.cos(math.radians(-72.1715))], rtol=1e-6)


def test_metadata_constants_match_clay_yaml():
    import yaml
    for p in (Path("/home/photon/4tb/envs/fluxfill-clay/clay_configs/metadata.yaml"),):
        if p.exists():
            md = yaml.safe_load(open(p))["sentinel-2-l2a"]
            names = md["band_order"]
            assert names == ["blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16",
                             "swir22"]
            assert [md["bands"]["mean"][b] for b in names] == MEAN.tolist()
            assert [md["bands"]["std"][b] for b in names] == STD.tolist()
            return
    pytest.skip("Clay metadata.yaml not installed")
