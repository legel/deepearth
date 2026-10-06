"""Climate fill (ranges/joint/climate_fill.py): shoreline locations take the climate of the nearest place with one.

* points: the batched fill equals its one-point-at-a-time definition exactly (values; distances to 1e-4 km) on a
  coarse synthetic global grid with sea gaps, at any batch size, near the antimeridian and at high latitudes; points
  with climate are untouched (distance 0); points with nothing within max_km stay missing (NaN);
* the map grid: every filled cell's source is a nearest cell with climate (brute force), exactly the land cells
  without climate within max_km are filled, and ``GridFill.window`` applied to any window reproduces the fill of the
  whole grid and leaves every other cell unchanged;
* the NALCMS water share of each 240 m cell (8 x 8 pixels of 30 m) on a small synthetic raster.
"""
import numpy as np
import pytest

from ranges.joint import climate_fill as CF

CELL = 0.5                                  # degrees: a 360 x 720 global grid
MAX_KM = 150.0                              # about 2.7 cells north-south


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(0)
    nrow, ncol, bands = int(180 / CELL), int(360 / CELL), 3
    data = rng.standard_normal((nrow, ncol, bands)).astype(np.float32)
    sea = np.zeros((nrow, ncol), bool)
    for _ in range(400):                    # rectangular seas of 1-12 cells
        r, c = rng.integers(0, nrow), rng.integers(0, ncol)
        sea[r:r + rng.integers(1, 13), c:c + rng.integers(1, 13)] = True
    sea[:, :6] = sea[:, -6:] = True         # an ocean across the antimeridian
    sea[:8] = True                          # and around the north pole
    data[sea] = np.nan
    data[rng.random((nrow, ncol)) < 0.02, 1] = np.nan        # one band missing: no climate either
    return data


def _points(world, rng, n):
    lat = np.r_[rng.uniform(-89.9, 89.9, n), rng.uniform(80, 89.99, 50), rng.uniform(-60, 60, 50)]
    lon = np.r_[rng.uniform(-180, 180, n), rng.uniform(-180, 180, 50), rng.choice([-179.99, 179.99], 50)]
    r = np.clip(np.floor((90 - lat) / CELL).astype(int), 0, world.shape[0] - 1)
    c = np.clip(np.floor((lon + 180) / CELL).astype(int), 0, world.shape[1] - 1)
    X = np.concatenate([world[r, c], rng.standard_normal((len(lat), 2)).astype(np.float32)], 1)   # + 2 other columns
    return X, lat, lon


def test_points_batched_equal_reference(world):
    X, lat, lon = _points(world, np.random.default_rng(1), 3000)
    miss = ~np.isfinite(X[:, :3]).all(1)
    assert miss.sum() > 300
    ref, dref = CF.fill_points_reference(X, lat, lon, world, CELL, 3, MAX_KM)
    for batch in (1, 7, 100_000):
        got, d = CF.fill_points(X, lat, lon, world, CELL, 3, MAX_KM, batch)
        assert np.array_equal(got, ref, equal_nan=True), batch
        assert np.allclose(d, dref, atol=1e-4, equal_nan=True), batch
    assert np.array_equal(got[~miss], X[~miss]) and not d[~miss].any()              # climate kept as it was
    assert np.array_equal(got[:, 3:], X[:, 3:], equal_nan=True)                     # other columns untouched
    far = np.isnan(d)
    assert far.any() and (~np.isfinite(got[far, :3])).any(1).all()                  # nothing near: stays missing
    filled = miss & ~far
    assert filled.sum() > 100 and np.isfinite(got[filled, :3]).all() and (d[filled] <= MAX_KM).all()
    # a filled point carries the bands of an existing pixel with climate, at the reported distance
    i = np.flatnonzero(filled)[0]
    hit = np.flatnonzero((world.reshape(-1, 3) == got[i, :3]).all(1))
    rr, cc = np.unravel_index(hit[0], world.shape[:2])
    import torch
    dd = CF.great_circle_km(*(torch.tensor(v, dtype=torch.float64) for v in
                              (lat[i], lon[i], 90 - (rr + 0.5) * CELL, -180 + (cc + 0.5) * CELL)))
    assert abs(float(dd) - d[i]) < 1e-3


def test_grid_fill_index_is_the_nearest_cell():
    rng = np.random.default_rng(2)
    H, W = 40, 55
    valid = rng.random((H, W)) > 0.35
    valid[10:25, 20:40] = False                                          # a lake
    land = rng.random((H, W)) > 0.2
    dst, src = CF.grid_fill_index(valid, land, cell_km=1.0, max_km=4.0)
    vr, vc = np.nonzero(valid)
    expect = []
    for k in range(H * W):
        r, c = divmod(k, W)
        if not land[r, c] or valid[r, c]:
            continue
        dmin = np.sqrt(((vr - r) ** 2 + (vc - c) ** 2).min())
        if dmin <= 4.0:
            expect.append((k, dmin))
    assert dst.tolist() == [k for k, _ in expect]
    for k, (kk, dmin) in zip(src, expect):
        r, c = divmod(int(kk), W)
        sr, sc = divmod(int(k), W)
        assert valid[sr, sc] and np.isclose(np.hypot(sr - r, sc - c), dmin)


def test_grid_fill_window_reproduces_the_fill():
    rng = np.random.default_rng(3)
    H, W, B = 60, 70, 4
    stack = rng.standard_normal((B, H, W)).astype(np.float32)
    valid = rng.random((H, W)) > 0.3
    stack[:, ~valid] = np.nan
    dst, src = CF.grid_fill_index(valid, np.ones((H, W), bool), cell_km=1.0, max_km=3.0)
    full = stack.reshape(B, -1).copy()
    full[:, dst] = full[:, src]
    fill = CF.GridFill(dst[::-1].copy(), src[::-1].copy(), (H, W))                 # any order on input
    for (r0, r1, c0, c1) in [(0, H, 0, W), (5, 33, 7, 61), (59, 60, 0, 1), (10, 11, 3, 70)]:
        X = stack[:, r0:r1, c0:c1].reshape(B, -1).T.copy()
        pos, sr, sc = fill.window(r0, r1, c0, c1)
        X[pos] = stack[:, sr, sc].T
        want = full.reshape(B, H, W)[:, r0:r1, c0:c1].reshape(B, -1).T
        assert np.array_equal(X, want, equal_nan=True)


def test_water_fraction(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    px = np.zeros((16, 24), np.uint8)                       # 2 x 3 cells of 8 x 8 pixels
    px[:8, :8] = 18                                         # all water
    px[:8, 8:16] = 1
    px[:8, 8:12] = 19                                       # half snow/ice
    px[8:, :8] = 0                                          # unclassified: NaN
    px[8:, 8:16] = 127                                      # no data: NaN
    px[8:, 16:24] = 5
    px[8:10, 16:24] = 18                                    # 16 water pixels of 64
    px[10, 16] = 0                                          # one unclassified pixel: 16 of 63
    px[:8, 16:24] = 15
    f = tmp_path / "nalcms.tif"
    with rasterio.open(f, "w", driver="GTiff", height=16, width=24, count=1, dtype="uint8", crs="EPSG:5070",
                       transform=from_origin(0, 0, 30, 30)) as d:
        d.write(px, 1)
    w = CF.water_fraction(f, (2, 3), rows=1)
    assert np.allclose(w, [[1.0, 0.5, 0.0], [np.nan, np.nan, 16 / 63]], equal_nan=True)
    with pytest.raises(ValueError):
        CF.water_fraction(f, (3, 3))
