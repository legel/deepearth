"""Chip geometry: the chip centre maps back to the tower, windows pad at raster edges, resampling is exact."""

import numpy as np
import pytest

from geometry import Window, chip_window, lonlat_to_utm, read_window, upsample_to, utm_to_lonlat

# Real tile upper-left corners (Earth Search items) and towers inside them.
CASES = [
    (-72.1715, 42.5378, 32618, 699960.0, 4800000.0),    # US-Ha1, tile 18TYN
    (13.5651, 50.9626, 32633, 300000.0, 5700000.0),     # DE-Tha, tile 33UUS
    (131.1523, -12.4943, 32752, 699960.0, 8700040.0),   # AU-How, tile 52LGL-like origin
]


@pytest.mark.parametrize("lon,lat,epsg,x0,y0", CASES)
def test_chip_centre_within_half_pixel_of_tower(lon, lat, epsg, x0, y0):
    x, y = lonlat_to_utm(lon, lat, epsg)
    for res, n in ((10, 30), (20, 15), (60, 5)):
        w = chip_window(x, y, x0, y0, res, n)
        cx, cy = w.centre
        assert abs(cx - x) <= res / 2 + 1e-6 and abs(cy - y) <= res / 2 + 1e-6
        assert w.size * w.res == 300
    w10 = chip_window(x, y, x0, y0, 10, 30)
    clon, clat = utm_to_lonlat(*w10.centre, epsg)
    back_x, back_y = lonlat_to_utm(clon, clat, epsg)
    assert np.hypot(back_x - x, back_y - y) <= 5 * np.sqrt(2) + 1e-6
    assert abs(back_x - x) <= 5 + 1e-6 and abs(back_y - y) <= 5 + 1e-6


def test_random_towers_within_5_m_per_axis():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        x0 = 60.0 * rng.integers(1000, 12000)
        y0 = 60.0 * rng.integers(1000, 160000)
        x, y = x0 + rng.uniform(0, 109800), y0 - rng.uniform(0, 109800)
        w = chip_window(x, y, x0, y0, 10, 30)
        cx, cy = w.centre
        assert abs(cx - x) <= 5 + 1e-9 and abs(cy - y) <= 5 + 1e-9
        # window lies on the grid
        assert (w.ul[0] - x0) % 10 == 0 and (y0 - w.ul[1]) % 10 == 0


def test_read_window_pads_outside_raster():
    data = np.arange(100, dtype=np.uint16).reshape(10, 10) + 1
    read = lambda c, r, w, h: data[r:r + h, c:c + w]
    w = Window(col=-2, row=7, size=5, res=10, x0=0, y0=0)
    out = read_window(read, w, 10, 10, np.uint16)
    assert out.shape == (5, 5)
    assert (out[:, :2] == 0).all() and (out[3:, :] == 0).all()
    assert (out[:3, 2:] == data[7:10, 0:3]).all()
    far = Window(col=50, row=50, size=5, res=10, x0=0, y0=0)
    assert (read_window(read, far, 10, 10, np.uint16) == 0).all()


def test_upsample_20_to_10_is_block_replication_when_aligned():
    src = np.arange(15 * 15).reshape(15, 15)
    s = Window(col=100, row=200, size=15, res=20, x0=0, y0=0)
    d = Window(col=200, row=400, size=30, res=10, x0=0, y0=0)
    up = upsample_to(src, s, d)
    assert up.shape == (30, 30)
    assert (up == np.kron(src, np.ones((2, 2), dtype=int))).all()


def test_upsample_offset_window_clamps_to_edge():
    src = np.arange(15 * 15).reshape(15, 15)
    s = Window(col=100, row=200, size=15, res=20, x0=0, y0=0)
    d = Window(col=201, row=400, size=30, res=10, x0=0, y0=0)  # shifted one 10 m pixel east
    up = upsample_to(src, s, d)
    ref = np.kron(src, np.ones((2, 2), dtype=int))
    assert (up[:, :29] == ref[:, 1:]).all()
    assert (up[:, 29] == src[np.arange(30) // 2, 14]).all()
