"""Assemble a solver `Surface` for a site: terrain onto one grid, soil onto the same grid.

Where the predecessor degraded silently, this module asserts. Two cases mattered in practice:
a missing soil-storage table left infiltration unbounded, which absorbs essentially any storm
and quietly produces a different model; and a missing map-unit raster fell back to a single
domain-wide Horton mean, which invalidates a nine-run probability ensemble without saying so.
Both printed a warning and carried on.
"""

import csv
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds, rowcol
from rasterio.warp import reproject

from physics import IMPERVIOUS_FC_MM_HR, road_buffer_m
from infiltration import RAWLS_1983, Soil, usda_texture
from sites import SiteConfig
from solver import Surface

DEM_RESAMPLING = Resampling.min
"""Downsampling rule for the conditioned DEM, and not an arbitrary choice.

richdem's breaching carves least-cost drainage one cell (~0.9 m) wide. Any averaging kernel
blends that trench back into the surrounding high ground at 5 m, re-sealing every outlet the
conditioning opened and reintroducing 3.6-4.6e6 m^3 of depression storage -- a full storm's
worth. `min` is the channel-preserving choice: a coarse cell containing a drainage path should
convey at that path's elevation, since conveyance is set by the lowest flow line through the
cell rather than its mean surface. It costs a downward bias of 0.134 m mean against 27.9 m of
relief, and leaves median slope essentially unchanged (1.830 % -> 1.848 %).
"""

AMC3_FACTOR = 0.07
"""Effective-Ksat fraction for a pre-saturated spodosol profile.

Most soils here are HSG B/D: dual-rated because the sandy A horizon conducts fast (32-331
mm/hr) while the restrictive spodic Bh horizon at 50-80 cm limits real drainage. Ian followed
two weeks of above-average rain, so the profile was wet and the water table near the surface.
"""

DRAINABLE_POROSITY = 0.25
"""Air-filled porosity between field capacity and saturation for these fine sands.

What a saturating profile can still accept is the depth to the seasonal-high water table times
this, not total porosity -- the profile below field capacity is already wet.
"""

NO_WATER_TABLE_DEPTH_CM = 150.0
"""Assumed water-table depth where SSURGO reports none, i.e. below its 200 cm observation
limit. Deeper than any storm can fill, so only the magnitude matters."""


def load_dem(site: SiteConfig, cell_size_m: float) -> Tuple[np.ndarray, Dict, float]:
    """Read the conditioned DEM, optionally downsampled onto a coarser square grid.

    Args:
        site: Site whose conditioned DEM to read.
        cell_size_m: Target resolution. At or below native resolution the raster is used as-is.

    Returns:
        (elevation with NaN outside the domain, rasterio profile for the grid, cell size [m]).
    """
    assert site.dem_conditioned.exists(), (
        f"{site.dem_conditioned} missing; run `python3 cli.py terrain --site {site.name}`")

    with rasterio.open(site.dem_conditioned) as src:
        native_res = abs(src.transform.a)
        crs, bounds = src.crs, src.bounds
        dem = src.read(1).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else -9999.0
        profile = src.profile.copy()

    if cell_size_m <= native_res * 1.1:
        z = dem.copy()
        z[z == nodata] = np.nan
        return z, profile, native_res

    # Normalise orientation before sizing. A north-up raster has bottom < top, but rasterio
    # reports whatever the affine produces, and site3's DEM has a positive y-resolution -- the
    # un-normalised subtraction yields a negative height and raises "negative dimensions".
    left, right = sorted([bounds.left, bounds.right])
    bottom, top = sorted([bounds.bottom, bounds.top])
    height = int((top - bottom) / cell_size_m)
    width = int((right - left) / cell_size_m)
    transform = from_bounds(left, bottom, right, top, width, height)

    out = np.zeros((height, width), dtype=np.float32)
    # nodata must be declared on both sides. Without it GDAL treats the sentinel as data, and
    # Resampling.min then fills a NaN hole with the surrounding minimum -- inventing terrain at
    # exactly the elevations that matter most.
    reproject(dem, out,
              src_transform=profile["transform"], src_crs=crs,
              dst_transform=transform, dst_crs=crs,
              src_nodata=nodata, dst_nodata=nodata,
              resampling=DEM_RESAMPLING)
    out[out == nodata] = np.nan

    profile.update(height=height, width=width, transform=transform)
    return out, profile, float(cell_size_m)


def _warp_onto(path: Path, shape: Tuple[int, int], profile: Dict, resampling: Resampling,
               dtype: type = np.float32, fill: float = 0.0) -> np.ndarray:
    """Reproject a raster onto the solver grid."""
    with rasterio.open(path) as src:
        native, src_transform, src_crs = src.read(1), src.transform, src.crs
    out = np.full(shape, fill, dtype=dtype)
    reproject(native.astype(dtype), out,
              src_transform=src_transform, src_crs=src_crs,
              dst_transform=profile["transform"], dst_crs=profile["crs"],
              resampling=resampling)
    return out


def spatial_horton(site: SiteConfig, shape: Tuple[int, int],
                   profile: Dict) -> Dict[str, np.ndarray]:
    """Per-cell Horton f0/fc/k [SI] from the SSURGO map-unit raster.

    fc is the survey's A-horizon Ksat scaled by `AMC3_FACTOR`; f0 is 2.5x fc, a small initial
    excess appropriate to near-saturated conditions.
    """
    for p in (site.mukey_map, site.mukey_legend, site.soil_params):
        assert p.exists(), f"{p} missing; run `python3 cli.py fetch --site {site.name}`"

    mukey = _warp_onto(site.mukey_map, shape, profile,
                       Resampling.nearest, np.int32).astype(np.int32)
    covered = float((mukey != 0).mean())
    assert covered > 0.5, (
        f"{site.mukey_map} covers only {covered:.1%} of the solver grid. Below half the domain "
        f"the per-cell soil field is mostly the domain mean, which is the uniform-infiltration "
        f"behaviour this pipeline exists to avoid -- and it is what an empty raster looks like.")
    with open(site.mukey_legend, newline="") as fh:
        legend = {int(r["mukey_int"]): str(r["mukey"]) for r in csv.DictReader(fh)}
    soil = json.loads(site.soil_params.read_text())

    units = [v for v in soil.values() if "fc_mm_hr" in v and "Water" not in v.get("muname", "")]
    assert units, f"{site.soil_params} has no usable map units"
    # Rounded to match the values every recorded run was produced with. These are only the
    # fill for map-unit codes the survey does not describe; mapped cells overwrite them below.
    fc_default = round(float(np.mean([v["fc_mm_hr"] for v in units])) * AMC3_FACTOR, 1)
    k_default = round(float(np.mean([v["k_hr"] for v in units])), 2)

    fc = np.full(shape, fc_default, dtype=np.float32)
    f0 = np.full(shape, round(fc_default * 2.5, 1), dtype=np.float32)
    k = np.full(shape, k_default, dtype=np.float32)

    for code in np.unique(mukey):
        params = soil.get(legend.get(int(code), ""), None)
        if not params or "fc_mm_hr" not in params:
            continue  # unmapped code keeps the domain-mean fallback
        mask = mukey == code
        fc[mask] = params["fc_mm_hr"] * AMC3_FACTOR
        f0[mask] = params["fc_mm_hr"] * AMC3_FACTOR * 2.5
        k[mask] = params["k_hr"]

    return {"f0": f0 / 1000 / 3600, "fc": fc / 1000 / 3600, "k": k / 3600}


def impervious(site: SiteConfig, shape: Tuple[int, int], profile: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """(OSM roads and buildings as a hard mask, NLCD impervious fraction) on the solver grid."""
    import geopandas as gpd
    from rasterio.features import rasterize

    for p in (site.roads, site.buildings, site.nlcd_impervious):
        assert p.exists(), f"{p} missing; run `python3 cli.py fetch --site {site.name}`"

    crs = profile["crs"]
    roads = gpd.read_file(site.roads).to_crs(crs)
    buildings = gpd.read_file(site.buildings).to_crs(crs)
    shapes = [(row.geometry.buffer(road_buffer_m(str(row.get("highway")))), 1)
              for _, row in roads.iterrows()]
    shapes += [(geom, 1) for geom in buildings.geometry]
    hard = rasterize(shapes, out_shape=shape, transform=profile["transform"],
                     fill=0, dtype=np.uint8).astype(bool)
    nlcd = _warp_onto(site.nlcd_impervious, shape, profile, Resampling.bilinear)
    return hard, np.clip(np.nan_to_num(nlcd, nan=0.0) / 100.0, 0.0, 1.0)


def apply_impervious(site: SiteConfig, horton: Dict[str, np.ndarray], shape: Tuple[int, int],
                     profile: Dict) -> Dict[str, np.ndarray]:
    """Zero infiltration under OSM roads and buildings, then grade the rest by NLCD.

    The binary OSM mask is a hard cut: a real road is fully impervious whatever NLCD's 30 m
    pixel reports. Everywhere else, capacity is scaled by (1 - impervious fraction), which
    catches driveways and compacted ground that OSM does not map. `k` is untouched --
    imperviousness changes how much can infiltrate, not the shape of the decay curve.
    """
    hard, frac = impervious(site, shape, profile)
    fc_hard = IMPERVIOUS_FC_MM_HR / 1000 / 3600
    horton["fc"] = np.where(hard, fc_hard, horton["fc"])
    horton["f0"] = np.where(hard, fc_hard, horton["f0"])
    grade = np.where(np.isclose(horton["fc"], fc_hard), 1.0, 1.0 - frac)
    horton["fc"] = horton["fc"] * grade
    horton["f0"] = horton["f0"] * grade
    return horton


def soil_storage(site: SiteConfig, shape: Tuple[int, int], profile: Dict) -> np.ndarray:
    """Per-cell finite soil storage [m] -- the maximum deficit of the Deficit-and-Constant method.

    Built as depth to the seasonal-high water table times drainable porosity. Depressional soils
    report a water table at the surface and so get zero storage: they generate runoff
    immediately, which is correct for them.
    """
    assert site.soil_storage.exists(), (
        f"{site.soil_storage} missing. Without it infiltration is unbounded, which absorbs "
        f"essentially any storm; run `python3 cli.py fetch --site {site.name}`")

    with open(site.soil_storage, newline="") as fh:
        wt_cm = {str(r["mukey"]): float(r["wtdepannmin"]) if (r.get("wtdepannmin") or "").strip()
                 else NO_WATER_TABLE_DEPTH_CM for r in csv.DictReader(fh)}
    with open(site.mukey_legend, newline="") as fh:
        key_to_int = {str(r["mukey"]): int(r["mukey_int"]) for r in csv.DictReader(fh)}

    mukey = _warp_onto(site.mukey_map, shape, profile,
                       Resampling.nearest, np.int32).astype(np.int32)
    known = [v * 0.01 * DRAINABLE_POROSITY for v in wt_cm.values()]
    out = np.full(shape, float(np.mean(known)) if known else 0.0, dtype=np.float32)
    for mukey_str, code in key_to_int.items():
        if mukey_str in wt_cm:
            out[mukey == code] = wt_cm[mukey_str] * 0.01 * DRAINABLE_POROSITY
    return out


def gar_soil(site: SiteConfig, shape: Tuple[int, int], profile: Dict) -> Soil:
    """Per-cell Green-Ampt soil from SSURGO, at field capacity, above its seasonal-high water table.

    K_s is the dominant component's surface-horizon `ksat_r`, zero-graded by impervious cover as the Horton path
    is; theta_s is `wsatiated_r` and the antecedent theta_i `wthirdbar_r` (field capacity: Ian followed two weeks
    of above-average rain). theta_r, lambda and psi_f come from the Rawls et al. (1983) row of the horizon's
    USDA texture. What the profile can take, F_max, is the water-table depth times (theta_s - theta_i): the
    measured pore space, where the Horton path multiplies by a fixed `DRAINABLE_POROSITY`. A map unit the
    survey leaves blank takes the Rawls row of loam.
    """
    for p in (site.mukey_map, site.mukey_legend, site.soil_hydraulics, site.soil_storage):
        assert p.exists(), f"{p} missing; run `python3 cli.py fetch --site {site.name}`"
    hyd = json.loads(site.soil_hydraulics.read_text())
    names = json.loads(site.soil_params.read_text()) if site.soil_params.exists() else {}
    with open(site.soil_storage, newline="") as fh:
        wt_cm = {str(r["mukey"]): float(r["wtdepannmin"]) if (r.get("wtdepannmin") or "").strip()
                 else NO_WATER_TABLE_DEPTH_CM for r in csv.DictReader(fh)}
    with open(site.mukey_legend, newline="") as fh:
        legend = {int(r["mukey_int"]): str(r["mukey"]) for r in csv.DictReader(fh)}
    mukey = _warp_onto(site.mukey_map, shape, profile, Resampling.nearest, np.int32).astype(np.int32)

    fields = {k: np.zeros(shape) for k in ("ks", "psi_f", "theta_s", "theta_r", "lam", "theta_i", "f_max")}
    for code in np.unique(mukey):
        key = legend.get(int(code), "")
        h = hyd.get(key, {})
        sand, clay = h.get("sandtotal_r"), h.get("claytotal_r")
        row = RAWLS_1983[usda_texture(sand, clay) if sand is not None and clay is not None else "loam"]
        theta_s = (h.get("wsatiated_r") or 100.0 * row["theta_s"]) / 100.0
        theta_i = min((h.get("wthirdbar_r") or 100.0 * (row["theta_r"] + 0.5 * (row["theta_s"] - row["theta_r"])))
                      / 100.0, theta_s)
        water = "water" in str(names.get(key, {}).get("muname", "")).lower()
        ks = 0.0 if water else (h["ksat_r"] * 1e-6 if h.get("ksat_r") is not None else row["ks_cm_h"] / 3.6e5)
        m = mukey == code
        for k, v in (("ks", ks), ("psi_f", row["psi_f_cm"] / 100.0), ("theta_s", theta_s),
                     ("theta_r", min(row["theta_r"], theta_i)), ("lam", row["lam"]), ("theta_i", theta_i),
                     ("f_max", wt_cm.get(key, NO_WATER_TABLE_DEPTH_CM) * 0.01 * (theta_s - theta_i))):
            fields[k][m] = v

    hard, frac = impervious(site, shape, profile)
    fields["ks"] = np.where(hard, IMPERVIOUS_FC_MM_HR / 1000 / 3600, fields["ks"] * (1.0 - frac))
    return Soil(**fields)


def snap_gauge(site: SiteConfig, z: np.ndarray, profile: Dict, dx: float,
               search_m: float = 25.0) -> Tuple[int, int]:
    """Locate the streamgauge on the solver grid, snapped onto the channel.

    The published coordinate can land a cell or two off the burned centreline, and reading a dry
    floodplain cell beside the creek reports almost no discharge. The lowest-bed cell within
    `search_m` is the channel on a burned DEM.
    """
    from pyproj import Transformer

    assert site.gauge is not None, f"site {site.name} has no gauge"
    x, y = Transformer.from_crs("epsg:4326", profile["crs"], always_xy=True).transform(
        site.gauge.lon, site.gauge.lat)
    row, col = rowcol(profile["transform"], x, y, op=round)
    row, col = int(row), int(col)

    rad = max(1, int(round(search_m / dx)))
    r0, r1 = max(0, row - rad), min(z.shape[0], row + rad + 1)
    c0, c1 = max(0, col - rad), min(z.shape[1], col + rad + 1)
    sub = z[r0:r1, c0:c1]
    flat = int(np.nanargmin(np.where(np.isfinite(sub), sub, np.inf)))
    return r0 + flat // sub.shape[1], c0 + flat % sub.shape[1]


def build_surface(site: SiteConfig, cell_size_m: float, infiltration: str = "horton") -> Tuple[Surface, Dict, float]:
    """Terrain, soil and impervious cover for one site on one grid.

    Roughness is deliberately not a parameter here. A caller wanting the segmentation-derived
    field sets `Surface.manning_n` afterwards, using the profile this returns -- taking it as an
    argument meant the caller had to load the DEM first to know the grid, and then this function
    loaded the same 60-million-cell raster a second time to use it.

    Args:
        site: Site to assemble.
        cell_size_m: Solver resolution.
        infiltration: "horton" (spatial Horton against a finite store) or "gar" (Green-Ampt with
            redistribution from the survey's hydraulics, `gar_soil`).

    Returns:
        (Surface, rasterio profile for the grid, cell size [m]).
    """
    z, profile, dx = load_dem(site, cell_size_m)
    if infiltration == "gar":
        return Surface(z=z, soil=gar_soil(site, z.shape, profile)), profile, dx
    assert infiltration == "horton", infiltration
    horton = apply_impervious(site, spatial_horton(site, z.shape, profile), z.shape, profile)
    return (
        Surface(z=z, f0=horton["f0"], fc=horton["fc"], k=horton["k"],
                max_deficit_m=soil_storage(site, z.shape, profile)),
        profile,
        dx,
    )
