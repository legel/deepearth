"""Geodesy (ranges/joint/geodesy.py): WGS 84 -> EPSG:5070 exactly as PROJ (and so gdalwarp) transforms points.

* the Albers projection alone (NAD83 -> EPSG:5070, no datum shift) against pyproj, needing no grid;
* the full transformation against pyproj to 1e-6 m: random CONUS points, the edges of every NOAA HARN grid (PROJ's
  extent tolerance), the NRC Quebec grids (nested subgrids placed by PROJ's bounding-box rule) and exact integer
  coordinates (e.g. 48 N, 85 W on a grid edge). Needs the PROJ grid files (<data root>/raw/proj_grids or pyproj's
  data directories; geodesy.fetch_grids); skipped without them;
* grid positions: fractional rows and columns from the raster's affine transform, points outside the box off grid.
"""
import numpy as np
import pytest
import torch
from pyproj import Transformer

from ranges import config
from ranges.joint import geodesy as G


def test_albers_matches_pyproj():
    rng = np.random.default_rng(0)
    lat, lon = rng.uniform(15, 60, 100_000), rng.uniform(-170, -50, 100_000)
    x, y = G.albers_5070(torch.from_numpy(lat), torch.from_numpy(lon))
    xr, yr = Transformer.from_crs(4269, 5070, always_xy=True).transform(lon, lat)
    assert max(np.abs(x.numpy() - xr).max(), np.abs(y.numpy() - yr).max()) < 1e-6


def test_grid_positions_box_and_affine():
    t = (240.0, 0.0, -2493045.0, 0.0, -240.0, 3310005.0)
    lat = np.array([37.87, 45.0, 10.0, 55.0, 37.0])
    lon = np.array([-122.27, -100.0, -100.0, -100.0, -140.0])
    rc = G.grid_positions(lat, lon, t, G.albers_5070, chunk=2)
    x, y = G.albers_5070(torch.from_numpy(lat), torch.from_numpy(lon))
    assert rc.dtype == np.float32 and rc.shape == (5, 2)
    assert np.allclose(rc[:2, 0], ((y.numpy() - t[5]) / t[4])[:2])
    assert np.allclose(rc[:2, 1], ((x.numpy() - t[2]) / t[0])[:2])
    assert (rc[2:] == -1e6).all()


@pytest.fixture(scope="module")
def to5070():
    try:
        return G.WGS84ToConusAlbers(config.data_root() / "raw" / "proj_grids")
    except FileNotFoundError as e:
        pytest.skip(f"PROJ grids not available: {e}")


def test_matches_pyproj_to_a_micrometre(to5070):
    ref = Transformer.from_crs(4326, 5070, always_xy=True)
    rng = np.random.default_rng(0)
    cases = {"random CONUS": (rng.uniform(24.5, 49.4, 200_000), rng.uniform(-124.8, -66.9, 200_000))}
    lat_e, lon_e = [], []
    for o in to5070.ops:
        if o["grid"] is None:
            continue
        for g in o["grid"].top:
            w, s, e, n = g.extent()
            t = np.linspace(0, 1, 50)
            lat_e += [np.full(50, s), np.full(50, n), s + (n - s) * t, s + (n - s) * t]
            lon_e += [w + (e - w) * t, w + (e - w) * t, np.full(50, w), np.full(50, e)]
    cases["grid edges"] = (np.concatenate(lat_e), np.concatenate(lon_e))
    cases["Quebec nested grids"] = (rng.uniform(45.0, 47.5, 50_000), rng.uniform(-74.5, -70.0, 50_000))
    la, lo = np.meshgrid(np.arange(25.0, 50.0), np.arange(-125.0, -66.0))
    cases["integer coordinates"] = (la.ravel(), lo.ravel())
    worst = {}
    for name, (lat, lon) in cases.items():
        x, y = to5070(torch.from_numpy(lat), torch.from_numpy(lon))
        xr, yr = ref.transform(lon, lat)
        worst[name] = float(max(np.abs(x.numpy() - xr).max(), np.abs(y.numpy() - yr).max()))
    assert len(cases["grid edges"][0]) > 1000
    assert all(v < 1e-6 for v in worst.values()), worst
