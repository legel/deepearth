"""Per-column parameters: the built-in table, nearest resampling, and the bundle when present."""

import json
from pathlib import Path

import numpy as np
import pytest

import params
import sites
from domain import Grid
from params import NODATA, UNOBSERVED, ClassTable, from_rows, resample_nearest

BUNDLE = sites.get_site("campanile").bundle


def test_builtin_table_has_every_class_and_a_fallback():
    t = ClassTable.builtin()
    assert len(t.names) == 46 and t.names[t.fallback] == params.FALLBACK_CLASS
    assert t.solid[t.names.index("facade_masonry")] and not t.solid[t.names.index("tree_canopy")]


def test_unobserved_and_nodata_take_the_fallback_and_are_counted():
    t = ClassTable.builtin()
    canopy = t.names.index("tree_canopy")
    rows = np.array([[canopy, UNOBSERVED], [NODATA, canopy]], np.uint8)
    c = from_rows(rows, t, "test")
    assert c.row.tolist() == [[canopy, t.fallback], [t.fallback, canopy]]
    assert c.unobserved.sum() == 1 and c.nodata.sum() == 1 and c.observed.sum() == 2
    r = c.receipt(0.5)
    assert r["unobserved_cells"] == 1 and r["nodata_cells"] == 1
    assert r["class_area_m2"] == {"tree_canopy": 0.5}
    assert r["mean_z0_observed_m"] == pytest.approx(t.z0[canopy])


def test_nearest_resampling_places_pixels_by_projected_coordinate():
    """A 4x4 raster at 1 m, north-down, read onto a 0.5 m grid offset by the anchor."""
    raster = np.arange(16, dtype=np.uint8).reshape(4, 4)
    transform = (100.0, 1.0, 0.0, 204.0, 0.0, -1.0)  # west 100, north 204
    g = Grid.uniform(0.5, 8, 8, 2)
    out = resample_nearest(raster, transform, g, origin=(0.0, 0.0), anchor=(100.0, 200.0))
    assert out.shape == (8, 8)
    assert out[0, 0] == raster[3, 0] and out[7, 7] == raster[0, 3], "south-up from north-down"
    assert out[1, 1] == out[0, 0] and out[2, 2] == raster[2, 1], "each pixel covers 2x2 cells"
    outside = resample_nearest(raster, transform, g, origin=(10.0, 0.0), anchor=(100.0, 200.0))
    assert (outside == NODATA).all()


@pytest.mark.skipif(not (BUNDLE / "semantics" / "areas.json").exists(),
                    reason="semantics bundle not fetched")
def test_bundle_mean_z0_equals_the_class_area_sum():
    """Area-weighted z0 over the observed grid must equal areas.json times the table, to 1e-6."""
    t = ClassTable.load(BUNDLE / "semantics" / "parameters.json")
    assert len(t.names) == 47 and t.names[UNOBSERVED] == "unobserved"
    site = sites.get_site("campanile")
    raster, transform, nodata = params.read_raster(params.class_raster(BUNDLE, 0.2))
    dx, rows, cols = transform[1], raster.shape[0], raster.shape[1]
    g = Grid.uniform(dx, cols, rows, 2)
    origin = (transform[0] - site.anchor_utm[0], transform[3] + transform[5] * rows - site.anchor_utm[1])
    c = params.from_bundle(BUNDLE, g, origin, site.anchor_utm[:2])
    assert c.nodata.sum() == (raster == NODATA).sum(), "the solver grid is the raster grid"

    areas = json.loads((BUNDLE / "semantics" / "areas.json").read_text())["by_resolution"]
    top = next(v for v in areas.values() if v["res_m"] == dx)["top_surface"]
    weighted = sum(a * t.z0[t.names.index(n)] for n, a in top.items() if n != "unobserved")
    total = sum(a for n, a in top.items() if n != "unobserved")
    assert c.z0[c.observed].mean() == pytest.approx(weighted / total, rel=1e-6)
    assert c.observed.sum() * dx * dx == pytest.approx(total, rel=1e-6)


def _surface(d: Path, z: float = 7.0) -> None:
    """dtm_0.2m.tif and dsm_0.2m.tif, 10 x 10 cells about the origin, as a bundle's surface/ holds them."""
    import rasterio
    from rasterio.transform import Affine

    d.mkdir(parents=True, exist_ok=True)
    for kind, h in (("dtm", z), ("dsm", z + 3.0)):
        with rasterio.open(d / f"{kind}_0.2m.tif", "w", driver="GTiff", width=10, height=10, count=1, dtype="float32",
                           transform=Affine(0.2, 0.0, -1.0, 0.0, -0.2, 1.0), nodata=-9999.0) as ds:
            ds.write(np.full((10, 10), h, np.float32), 1)


def test_a_symlinked_surface_yields_its_terrain(tmp_path):
    """2026-09-13: a bundle whose surface/ was a symlink read as holding no rasters (pathlib's ** does not enter
    a symlinked directory before Python 3.13), and the wind was solved over flat ground with exit code 0."""
    _surface(tmp_path / "elsewhere" / "surface")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "surface").symlink_to(tmp_path / "elsewhere" / "surface", target_is_directory=True)
    g = Grid.uniform(0.2, 10, 10, 2)
    dtm, dsm = params.surface_rasters(bundle, 0.2, g, (-1.0, -1.0))
    assert np.nanmax(np.abs(dtm - 7.0)) == 0.0 and np.nanmax(np.abs(dsm - 10.0)) == 0.0
    assert params.nearest_raster(bundle, "dsm", 0.2).name == "dsm_0.2m.tif"


def test_a_bundle_whose_terrain_cannot_be_read_fails_the_run(tmp_path):
    """A surface/ that yields no DTM or DSM, or points nowhere, is never flat ground: the run fails, named."""
    g = Grid.uniform(0.2, 10, 10, 2)
    empty = tmp_path / "empty" / "surface"
    empty.mkdir(parents=True)
    with pytest.raises(params.TerrainMissing, match="dtm or dsm"):
        params.surface_rasters(empty.parent, 0.2, g, (-1.0, -1.0))
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "surface").symlink_to(tmp_path / "nowhere", target_is_directory=True)
    with pytest.raises(params.TerrainMissing, match="broken link"):
        params.surface_rasters(broken, 0.2, g, (-1.0, -1.0))
    assert params.surface_rasters(tmp_path / "no_terrain_at_all", 0.2, g, (-1.0, -1.0)) is None


def test_flat_ground_only_where_it_is_declared(monkeypatch):
    """A bundle with no surface/ at all is flat ground only under --flat-terrain (WIND_FLAT_TERRAIN=1), a
    verification case: a customer's site never declares it, and without it the scene is refused
    (domain.from_bundle raises params.TerrainMissing)."""
    import domain

    monkeypatch.delenv("WIND_FLAT_TERRAIN", raising=False)
    assert domain.flat_declared() is False and domain.flat_declared(True) is True
    monkeypatch.setenv("WIND_FLAT_TERRAIN", "1")
    assert domain.flat_declared() is True and domain.flat_declared(False) is False, "an explicit refusal wins"
