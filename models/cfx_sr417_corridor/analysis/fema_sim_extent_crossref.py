"""
Simulated Hurricane Ian Flood Extent × FEMA NFHL Flood Zones — Spatial Cross-Reference
===========================================================================================
A real, direct ground-truth check for the main AOI's Ian simulation that sidesteps this
project's long-standing watershed-area-mismatch problem entirely: FEMA's mapped flood-hazard
zones are a SPATIAL product (where flooding is expected/regulated), not a discharge/gauge
record, so comparing them against the simulator's own flood-EXTENT polygon (flood_extent_ian
.geojson) needs no watershed-area scaling at all — apples-to-apples in a way the Shingle Creek
gauge comparison structurally can't be.

This is NOT a perfect ground truth either, and that's stated plainly in the output, not hidden:
FEMA NFHL zones represent a regulatory 1%-annual-chance (100-yr) design-storm floodplain, a
different (and differently-derived) standard than "cells this project's own LISFLOOD-FP solver
predicted as wet during one specific real storm, Hurricane Ian." Agreement is suggestive, not
proof of either being correct — but it's a real, available, previously-unbuilt cross-check.

Outputs:
    analysis/data/fema_sim_extent_overlay.png   — viewer overlay (512x512, RGBA)
    analysis/data/fema_sim_extent_summary.json  — real areas + overlap percentages

Usage:
    python3 analysis/fema_sim_extent_crossref.py
"""
import os, json
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
OUT_DIR  = os.path.join(BASE_DIR, "data")
os.makedirs(OUT_DIR, exist_ok=True)

SIM_EXTENT_GEOJSON  = os.path.join(PROJ_DIR, "simulation", "outputs", "flood_extent_ian.geojson")
FEMA_GEOJSON        = os.path.join(PROJ_DIR, "floodplain", "data", "fema_flood_zones.geojson")
# Same bounds source viewer/preprocess/export_overlays.py's get_dem_bounds() uses, for pixel-
# for-pixel alignment with every other viewer overlay PNG (hydrography.png, ssurgo.png, etc.)
# should this ever be wired into the viewer.
DEM_TIF              = os.path.join(PROJ_DIR, "dem", "data", "sr417_corridor_dem.tif")
OVERLAY_PNG          = os.path.join(OUT_DIR, "fema_sim_extent_overlay.png")
SUMMARY_JSON         = os.path.join(OUT_DIR, "fema_sim_extent_summary.json")

WORK_CRS = "EPSG:5070"   # this project's standard equal-area-ish CRS for DEM-derived products
SIZE = 512

FLOODWAY_SUBTYPES = {
    "FLOODWAY", "ADMINISTRATIVE FLOODWAY", "COMMUNITY ENCROACHMENT AREA",
    "FLOWAGE EASEMENT AREA", "NARROW FLOODWAY",
}


def load_and_project(path):
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs(WORK_CRS)


def main():
    print("Loading simulated Ian flood extent + FEMA flood zones …")
    sim = load_and_project(SIM_EXTENT_GEOJSON)
    fema = load_and_project(FEMA_GEOJSON)

    sim_union = sim.union_all()
    sim_area_ha = sim_union.area / 10_000
    print(f"  Simulated Ian flood extent: {sim_area_ha:.2f} ha")

    # Real bug caught before trusting the numbers: fetch_fema_nfhl.py (like this project's own
    # NHD fetch) returns each FEMA feature's FULL,
    # un-clipped geometry if any part intersects the query bbox -- some AE/floodway polygons here
    # run the full length of Shingle Creek (~7km), and Zone X "AREA OF MINIMAL FLOOD HAZARD" is
    # 53,715 ha, vastly larger than the 523.75 ha AOI. Comparing simulated (AOI-only) area against
    # these RAW areas would be meaningless -- clip every FEMA geometry to the AOI/DEM box first,
    # same box the simulation itself runs inside.
    with rasterio.open(DEM_TIF) as src:
        b = src.bounds
    from shapely.geometry import box
    aoi_box = box(b.left, b.bottom, b.right, b.top)
    fema = fema.copy()
    fema["geometry"] = fema.geometry.intersection(aoi_box)
    fema = fema[~fema.geometry.is_empty]

    is_sfha = fema["SFHA_TF"].astype(str).str.upper() == "T"
    is_floodway = fema["ZONE_SUBTY"].astype(str).str.upper().isin(FLOODWAY_SUBTYPES)

    sfha_union = fema[is_sfha].union_all() if is_sfha.any() else None
    floodway_union = fema[is_floodway].union_all() if is_floodway.any() else None
    sfha_area_ha = sfha_union.area / 10_000 if sfha_union is not None else 0.0
    floodway_area_ha = floodway_union.area / 10_000 if floodway_union is not None else 0.0
    print(f"  FEMA SFHA (AE zones), AOI-clipped:     {sfha_area_ha:.2f} ha")
    print(f"  FEMA regulatory floodway, AOI-clipped: {floodway_area_ha:.2f} ha")

    # ── Real overlap computation ────────────────────────────────────────────────
    sim_and_sfha = sim_union.intersection(sfha_union) if sfha_union is not None else None
    sim_and_floodway = sim_union.intersection(floodway_union) if floodway_union is not None else None
    sim_and_sfha_ha = sim_and_sfha.area / 10_000 if sim_and_sfha is not None else 0.0
    sim_and_floodway_ha = sim_and_floodway.area / 10_000 if sim_and_floodway is not None else 0.0
    sim_outside_fema_ha = sim_area_ha - sim_and_sfha_ha

    pct_sim_inside_sfha = 100 * sim_and_sfha_ha / sim_area_ha if sim_area_ha > 0 else 0.0
    pct_sfha_captured_by_sim = 100 * sim_and_sfha_ha / sfha_area_ha if sfha_area_ha > 0 else 0.0
    pct_sim_inside_floodway = 100 * sim_and_floodway_ha / sim_area_ha if sim_area_ha > 0 else 0.0
    pct_floodway_captured_by_sim = 100 * sim_and_floodway_ha / floodway_area_ha if floodway_area_ha > 0 else 0.0

    print(f"\n── Real overlap ──────────────────────────────────────────────")
    print(f"  Simulated ∩ FEMA SFHA:      {sim_and_sfha_ha:.2f} ha "
          f"({pct_sim_inside_sfha:.1f}% of simulated area falls inside a mapped SFHA)")
    print(f"  FEMA SFHA captured by sim:  {pct_sfha_captured_by_sim:.1f}% of the mapped SFHA "
          f"area was also predicted flooded by the Ian simulation")
    print(f"  Simulated ∩ FEMA floodway:  {sim_and_floodway_ha:.2f} ha "
          f"({pct_sim_inside_floodway:.1f}% of simulated area)")
    print(f"  FEMA floodway captured:     {pct_floodway_captured_by_sim:.1f}% of the regulatory "
          f"floodway area was also predicted flooded")
    print(f"  Simulated area OUTSIDE any FEMA SFHA: {sim_outside_fema_ha:.2f} ha "
          f"({100 - pct_sim_inside_sfha:.1f}%)")

    # ── Viewer overlay PNG (matches this project's DEM-bounds/SIZE convention) ────
    with rasterio.open(DEM_TIF) as src:
        dem_bounds = src.bounds
        dem_crs = src.crs
    dst_transform = from_bounds(dem_bounds.left, dem_bounds.bottom, dem_bounds.right,
                                 dem_bounds.top, SIZE, SIZE)
    rgba = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)

    if sfha_union is not None:
        mask = rasterize([(sfha_union, 1)], out_shape=(SIZE, SIZE), transform=dst_transform,
                          fill=0, dtype=np.uint8).astype(bool)
        rgba[mask] = (230, 180, 40, 130)  # amber — FEMA SFHA, semi-transparent
    if floodway_union is not None:
        mask = rasterize([(floodway_union, 1)], out_shape=(SIZE, SIZE), transform=dst_transform,
                          fill=0, dtype=np.uint8).astype(bool)
        rgba[mask] = (220, 60, 40, 170)  # red — regulatory floodway, on top of SFHA fill
    sim_mask = rasterize([(sim_union, 1)], out_shape=(SIZE, SIZE), transform=dst_transform,
                          fill=0, dtype=np.uint8).astype(bool)
    # Simulated extent drawn as a cyan OUTLINE effect via a distinct fill wherever it does NOT
    # already have a FEMA color, plus a brighter tint wherever it overlaps FEMA — makes both
    # "agreement" and "simulated-only" areas visually distinguishable at a glance.
    overlap_mask = sim_mask & (rgba[:, :, 3] > 0)
    sim_only_mask = sim_mask & ~overlap_mask
    rgba[sim_only_mask] = (40, 200, 220, 150)     # cyan — simulated flood, no FEMA overlap
    rgba[overlap_mask] = (80, 230, 140, 210)      # green — simulated AND FEMA agree

    Image.fromarray(rgba).save(OVERLAY_PNG)
    print(f"\nSaved overlay PNG → {OVERLAY_PNG}")

    summary = {
        "simulated_flood_ha": round(sim_area_ha, 3),
        "fema_sfha_ha": round(sfha_area_ha, 3),
        "fema_floodway_ha": round(floodway_area_ha, 3),
        "simulated_and_sfha_overlap_ha": round(sim_and_sfha_ha, 3),
        "simulated_and_floodway_overlap_ha": round(sim_and_floodway_ha, 3),
        "simulated_outside_fema_sfha_ha": round(sim_outside_fema_ha, 3),
        "pct_simulated_area_inside_sfha": round(pct_sim_inside_sfha, 2),
        "pct_sfha_area_captured_by_simulation": round(pct_sfha_captured_by_sim, 2),
        "pct_simulated_area_inside_floodway": round(pct_sim_inside_floodway, 2),
        "pct_floodway_area_captured_by_simulation": round(pct_floodway_captured_by_sim, 2),
        "aoi_box_area_ha": round((b.right - b.left) * (b.top - b.bottom) / 10_000, 2),
        "caveat": ("FEMA NFHL zones represent a regulatory 1%-annual-chance design-storm "
                   "floodplain, a different standard than one specific real storm's simulated "
                   "extent -- agreement/disagreement is suggestive plausibility evidence, not "
                   "proof either is correct. This comparison sidesteps the watershed-area-"
                   "mismatch problem that makes the Shingle Creek gauge comparison invalid, "
                   "since both are spatial extents at the SAME AOI, not discharge magnitudes "
                   "integrated over different watershed areas. FEMA zone geometries were "
                   "clipped to the AOI/DEM box before any area math -- the raw fetched features "
                   "are unclipped and some run the full length of Shingle Creek (~7km) or, for "
                   "Zone X, cover 53,715 ha nationally; using their raw areas would have made "
                   "every percentage below meaningless."),
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary → {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
