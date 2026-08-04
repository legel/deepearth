"""
Export overlay textures aligned to the DEM grid → viewer/data/

Outputs (512x512 PNG, north-up, covering the full DEM extent):
  naip_rgb.png     — NAIP true-color aerial imagery (imagery/fetch_naip.py output),
                      a directly georeferenced GeoTIFF reprojected onto the DEM grid
                      (Task 3, 2026-06-29 Lance meeting notes; higher-res land-surface
                      counterpart to PlanetScope, which stays focused on water/flood extent)
  ssurgo.png       — SSURGO soil map units, colorized by canonical soil series
  hydrography.png  — USGS 3DHP flowlines (line, buffered for visibility) +
                      waterbodies (polygon)
  floodplain.png   — FEMA NFHL flood hazard zones, with the regulatory
                      floodway drawn on top in red
  roads_buildings.png — OSM roads (dark gray, buffered by highway type) +
                      building footprints (warm tan) — geometric mask for
                      separating built surfaces from natural ground

Usage:
    python3 viewer/preprocess/export_overlays.py
"""
import os, json
import numpy as np
import rasterio
import pandas as pd
import geopandas as gpd
import matplotlib.cm as mpl_cm
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt, binary_erosion
from PIL import Image

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # viewer/
PROJ_DIR     = os.path.dirname(BASE_DIR)
DEM_DIR      = os.path.join(PROJ_DIR, "dem", "data")
SOIL_DIR     = os.path.join(PROJ_DIR, "soil", "data")
HYDRO_DIR    = os.path.join(PROJ_DIR, "hydrography", "data")
FLOOD_DIR    = os.path.join(PROJ_DIR, "floodplain", "data")
BOUND_DIR    = os.path.join(PROJ_DIR, "boundary", "data")
INFRA_DIR    = os.path.join(PROJ_DIR, "infrastructure", "data")
IMAGERY_DIR  = os.path.join(PROJ_DIR, "imagery", "data")
TERRAIN_DIR  = os.path.join(PROJ_DIR, "dem", "data", "terrain")
PLANET_DIR   = os.path.join(PROJ_DIR, "Florida_Hydrology_PlanetScope")
OUT_DIR      = os.path.join(BASE_DIR, "data")
# Bumped 512 -> 2048 (2026-07-28): at the main AOI's ~2.3km box, 512px was only ~4.5m/px --
# coarser than even the original 0.6m NAIP source, let alone the 0.3m 2023 imagery being
# fetched to replace it. 2048px -> ~1.1m/px, same precedent already established for site3
# (which went 512->2048 for the same reason, and ended up sharper than this constant used to
# produce). Affects every overlay this script exports (NAIP/SSURGO/hydrography/floodplain/
# roads+buildings), not just NAIP -- consistent with how site3's own monkey-patch of this same
# constant already applies project-wide for that site, not NAIP-only.
SIZE         = 2048

os.makedirs(OUT_DIR, exist_ok=True)

# Regulatory-floodway subtypes per FEMA NFHL data dictionary (matches
# floodplain/fetch_fema_nfhl.py's FLOODWAY_SUBTYPES)
FLOODWAY_SUBTYPES = {
    "FLOODWAY", "ADMINISTRATIVE FLOODWAY", "COMMUNITY ENCROACHMENT AREA",
    "FLOWAGE EASEMENT AREA", "NARROW FLOODWAY",
}


def get_dem_bounds():
    with rasterio.open(os.path.join(DEM_DIR, "sr417_corridor_dem.tif")) as src:
        return src.bounds, src.crs


def _find_naip_rgb():
    """Return the most recent naip_{year}_RGB.tif in imagery/data/, or None."""
    if not os.path.isdir(IMAGERY_DIR):
        return None
    candidates = sorted(
        f for f in os.listdir(IMAGERY_DIR)
        if f.startswith("naip_") and f.endswith("_RGB.tif")
    )
    return os.path.join(IMAGERY_DIR, candidates[-1]) if candidates else None


def export_naip(bounds, dem_crs):
    """NAIP true-color aerial imagery — directly georeferenced GeoTIFF, so (unlike the
    PlanetScope derived PNGs) this reprojects straight from the source raster via
    rasterio.warp.reproject, the same pattern export_ssurgo()/export_floodplain() use
    for their real georeferenced sources.

    Areas outside the NAIP tile coverage (or outside imagery/fetch_naip.py's fetch
    radius) are left transparent rather than filled, so the terrain/wireframe shows
    through at the AOI edges instead of a false black border.
    """
    naip_path = _find_naip_rgb()
    if naip_path is None:
        print("  NAIP not found — skipping naip_rgb.png "
              "(run: python3 imagery/fetch_naip.py)")
        return

    dst_transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, SIZE, SIZE)
    rgb_dst = np.zeros((3, SIZE, SIZE), dtype=np.float32)

    with rasterio.open(naip_path) as src:
        for band in range(3):
            band_data = src.read(band + 1).astype(np.float32)
            reproject(
                band_data, rgb_dst[band],
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=dst_transform, dst_crs=dem_crs,
                resampling=Resampling.bilinear,
            )

        # Reproject the source's valid-data mask (nearest-neighbour — a mask is
        # categorical) so pixels outside NAIP's actual tile coverage stay transparent.
        mask_src = src.dataset_mask().astype(np.float32)
        mask_dst = np.zeros((SIZE, SIZE), dtype=np.float32)
        reproject(
            mask_src, mask_dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=dst_transform, dst_crs=dem_crs,
            resampling=Resampling.nearest,
        )

    rgba = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
    rgba[:, :, :3] = np.clip(np.moveaxis(rgb_dst, 0, -1), 0, 255).astype(np.uint8)
    rgba[:, :, 3] = np.where(mask_dst > 127, 235, 0).astype(np.uint8)

    img = Image.fromarray(rgba)
    img.save(os.path.join(OUT_DIR, "naip_rgb.png"))
    print(f"  naip_rgb.png saved ({SIZE}x{SIZE})  source={os.path.basename(naip_path)}")


def export_ssurgo(bounds, dem_crs):
    mukey_path = os.path.join(SOIL_DIR, "mukey_map.tif")
    legend_path = os.path.join(SOIL_DIR, "mukey_map_legend.csv")
    if not os.path.exists(mukey_path):
        print("  SSURGO mukey_map not found — skipping ssurgo.png")
        return

    def _canonical(muname):
        base = muname.split(',')[0].strip()
        if 'water' in base.lower():
            return 'Water'
        return base

    groups = {}
    if os.path.exists(legend_path):
        df = pd.read_csv(legend_path)
        for _, row in df.iterrows():
            key = int(row["mukey_int"])
            label = _canonical(str(row.get("muname", "")).strip())
            groups.setdefault(label, []).append(key)

    key_to_rgba = {}
    legend_entries = []
    color_idx = 1
    for label, mukeys in groups.items():
        if label == 'Water':
            r, g, b = 40, 120, 200
        else:
            rgba_f = mpl_cm.tab10(color_idx / 10)
            r, g, b = int(rgba_f[0] * 255), int(rgba_f[1] * 255), int(rgba_f[2] * 255)
            color_idx += 1
        for mk in mukeys:
            key_to_rgba[mk] = (r, g, b, 200)
        legend_entries.append({"label": label, "rgba": [r, g, b, 255]})

    with rasterio.open(mukey_path) as src:
        mukey = src.read(1).astype(np.int16)
        dst_transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, SIZE, SIZE)
        mukey_dst = np.zeros((SIZE, SIZE), dtype=np.float32)
        reproject(
            mukey.astype(np.float32), mukey_dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=dst_transform, dst_crs=dem_crs,
            resampling=Resampling.nearest,
        )
    mukey_dst = mukey_dst.astype(np.int16)

    nodata_mask = mukey_dst == 0
    if nodata_mask.any():
        nearest = distance_transform_edt(nodata_mask, return_distances=False, return_indices=True)
        mukey_dst[nodata_mask] = mukey_dst[nearest[0][nodata_mask], nearest[1][nodata_mask]]
        print(f"  Filled {int(nodata_mask.sum())} nodata pixels via nearest-neighbour")

    rgba = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
    for key, color in key_to_rgba.items():
        rgba[mukey_dst == key] = color

    img = Image.fromarray(rgba)
    img.save(os.path.join(OUT_DIR, "ssurgo.png"))
    print(f"  ssurgo.png saved ({SIZE}x{SIZE})")

    legend_out = os.path.join(OUT_DIR, "ssurgo_legend.json")
    with open(legend_out, "w") as f:
        json.dump(legend_entries, f, indent=2)
    print(f"  ssurgo_legend.json saved ({len(legend_entries)} entries)")


def _read_geojson_or_empty(path):
    if not os.path.exists(path):
        return gpd.GeoDataFrame(geometry=[], crs="epsg:4326")
    try:
        gdf = gpd.read_file(path)
    except Exception:
        return gpd.GeoDataFrame(geometry=[], crs="epsg:4326")
    if gdf.crs is None:
        gdf = gdf.set_crs("epsg:4326")
    return gdf


def export_hydrography(bounds, dem_crs):
    """3DHP waterbodies (polygon, blue) + flowlines (buffered line, cyan) on top."""
    flow_path = os.path.join(HYDRO_DIR, "3dhp_flowlines.geojson")
    water_path = os.path.join(HYDRO_DIR, "3dhp_waterbodies.geojson")

    flow_gdf = _read_geojson_or_empty(flow_path)
    water_gdf = _read_geojson_or_empty(water_path)

    if flow_gdf.empty and water_gdf.empty:
        print("  No 3DHP flowlines/waterbodies — skipping hydrography.png")
        return

    dst_transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, SIZE, SIZE)
    rgba = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)

    if not water_gdf.empty:
        water_proj = water_gdf.to_crs(dem_crs)
        mask = rasterize(
            [(geom, 1) for geom in water_proj.geometry if geom is not None],
            out_shape=(SIZE, SIZE), transform=dst_transform, fill=0, dtype=np.uint8,
        ).astype(bool)
        rgba[mask] = (40, 120, 200, 190)  # blue, semi-transparent — matches SSURGO water color

    if not flow_gdf.empty:
        flow_proj = flow_gdf.to_crs(dem_crs)
        # Buffer so a thin line is visible at 512x512 resolution over a ~2.6km AOI
        # (cell size at this output res is ~5m/px; a 12m buffer gives a ~5px-wide line).
        buffered = flow_proj.geometry.buffer(12)
        mask = rasterize(
            [(geom, 1) for geom in buffered if geom is not None],
            out_shape=(SIZE, SIZE), transform=dst_transform, fill=0, dtype=np.uint8,
        ).astype(bool)
        rgba[mask] = (80, 220, 220, 220)  # cyan, drawn on top of waterbody fill

    img = Image.fromarray(rgba)
    img.save(os.path.join(OUT_DIR, "hydrography.png"))
    print(f"  hydrography.png saved ({SIZE}x{SIZE})  "
          f"flowlines={len(flow_gdf)} waterbodies={len(water_gdf)}")


def export_floodplain(bounds, dem_crs):
    """FEMA NFHL flood hazard zones — X (skip, uninformative) under AE (blue) under FLOODWAY (red)."""
    zones_path = os.path.join(FLOOD_DIR, "fema_flood_zones.geojson")
    gdf = _read_geojson_or_empty(zones_path)
    if gdf.empty:
        print("  No FEMA flood zones — skipping floodplain.png")
        return

    gdf_proj = gdf.to_crs(dem_crs)
    dst_transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, SIZE, SIZE)
    rgba = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)

    def zone_subty(row):
        return str(row.get("ZONE_SUBTY") or "").upper()

    is_floodway = gdf_proj.apply(lambda r: zone_subty(r) in FLOODWAY_SUBTYPES, axis=1)
    is_sfha = gdf_proj.get("SFHA_TF", pd.Series(["F"] * len(gdf_proj))) == "T"

    # Draw order: general SFHA (AE etc.) first, floodway on top — floodway is a subset of SFHA.
    sfha_only = gdf_proj[is_sfha & ~is_floodway]
    floodway = gdf_proj[is_floodway]

    if not sfha_only.empty:
        mask = rasterize(
            [(geom, 1) for geom in sfha_only.geometry if geom is not None],
            out_shape=(SIZE, SIZE), transform=dst_transform, fill=0, dtype=np.uint8,
        ).astype(bool)
        rgba[mask] = (50, 110, 200, 130)  # blue, semi-transparent — general SFHA / AE zone

    if not floodway.empty:
        mask = rasterize(
            [(geom, 1) for geom in floodway.geometry if geom is not None],
            out_shape=(SIZE, SIZE), transform=dst_transform, fill=0, dtype=np.uint8,
        ).astype(bool)
        rgba[mask] = (220, 50, 50, 170)  # red, semi-transparent — regulatory floodway

    img = Image.fromarray(rgba)
    img.save(os.path.join(OUT_DIR, "floodplain.png"))
    print(f"  floodplain.png saved ({SIZE}x{SIZE})  "
          f"sfha={int(is_sfha.sum())} floodway={int(is_floodway.sum())} total={len(gdf)}")


def export_boundary(bounds, dem_crs):
    """CFX SR417 corridor estimated boundary — bright amber outline with faint fill."""
    bound_path = os.path.join(BOUND_DIR, "cfx_sr417_corridor_estimated.geojson")
    gdf = _read_geojson_or_empty(bound_path)
    if gdf.empty:
        print("  No corridor boundary — skipping boundary.png")
        return

    gdf_proj = gdf.to_crs(dem_crs)
    dst_transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, SIZE, SIZE)
    rgba = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)

    filled = rasterize(
        [(geom, 1) for geom in gdf_proj.geometry if geom is not None],
        out_shape=(SIZE, SIZE), transform=dst_transform, fill=0, dtype=np.uint8,
    ).astype(bool)

    # Separate the edge outline (5 px wide) from the interior fill
    interior = binary_erosion(filled, iterations=5)
    outline  = filled & ~interior

    rgba[interior] = (255, 200, 30, 35)   # faint amber fill — barely visible, preserves underlying layers
    rgba[outline]  = (255, 200, 30, 210)  # bright amber outline

    img = Image.fromarray(rgba)
    img.save(os.path.join(OUT_DIR, "boundary.png"))
    print(f"  boundary.png saved ({SIZE}x{SIZE})  polygons={len(gdf)}")


_ROAD_BUFFER_M = {
    "motorway": 16, "motorway_link": 12, "trunk": 14, "trunk_link": 10,
    "primary": 10, "primary_link": 8, "secondary": 8, "secondary_link": 6,
    "tertiary": 6, "tertiary_link": 5, "residential": 5, "unclassified": 5,
    "service": 3, "track": 3, "path": 2, "footway": 2, "pedestrian": 3,
    "proposed": 3, "construction": 3,
}


def export_roads_buildings(bounds, dem_crs):
    """OSM roads (dark gray, buffered by highway type) + building footprints (warm
    tan) — the geometric mask for separating built surfaces from natural ground
    (Task 1 from the 2026-06-29 Lance meeting notes)."""
    roads_path = os.path.join(INFRA_DIR, "roads.geojson")
    buildings_path = os.path.join(INFRA_DIR, "buildings.geojson")
    roads_gdf = _read_geojson_or_empty(roads_path)
    buildings_gdf = _read_geojson_or_empty(buildings_path)

    if roads_gdf.empty and buildings_gdf.empty:
        print("  No roads/buildings — skipping roads_buildings.png "
              "(run: python3 infrastructure/fetch_roads_buildings.py)")
        return

    dst_transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, SIZE, SIZE)
    rgba = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)

    if not roads_gdf.empty:
        roads_proj = roads_gdf.to_crs(dem_crs)
        buffer_m = roads_proj.get("highway", pd.Series(["residential"] * len(roads_proj))).apply(
            lambda h: _ROAD_BUFFER_M.get(str(h), 4)
        )
        buffered = [geom.buffer(w) for geom, w in zip(roads_proj.geometry, buffer_m) if geom is not None]
        mask = rasterize(
            [(geom, 1) for geom in buffered],
            out_shape=(SIZE, SIZE), transform=dst_transform, fill=0, dtype=np.uint8,
        ).astype(bool)
        rgba[mask] = (70, 70, 78, 210)  # dark gray — roads/paved surfaces

    if not buildings_gdf.empty:
        buildings_proj = buildings_gdf.to_crs(dem_crs)
        mask = rasterize(
            [(geom, 1) for geom in buildings_proj.geometry if geom is not None],
            out_shape=(SIZE, SIZE), transform=dst_transform, fill=0, dtype=np.uint8,
        ).astype(bool)
        rgba[mask] = (205, 155, 95, 225)  # warm tan/amber — building footprints (roofs)

    img = Image.fromarray(rgba)
    img.save(os.path.join(OUT_DIR, "roads_buildings.png"))
    print(f"  roads_buildings.png saved ({SIZE}x{SIZE})  "
          f"roads={len(roads_gdf)} buildings={len(buildings_gdf)}")


def _warp_png_to_dem_grid(png_path, tif_path, bounds, dem_crs, out_path, opacity=230):
    """Warp a derived RGBA PNG onto the DEM grid using the GeoTIFF for spatial registration.

    The derived PNGs (rgb_truecolor.png, ndvi.png, etc.) are plain images with no embedded
    georeferencing.  Their pixel grid is identical to their source GeoTIFF, so we read the
    GeoTIFF bounds/CRS once and use them as the PNG's registration.
    """
    if not os.path.exists(png_path):
        print(f"  {os.path.basename(out_path)} — {os.path.basename(png_path)} not found")
        return False

    # Read GeoTIFF spatial metadata (bounds + CRS only — we load pixels from the PNG)
    with rasterio.open(tif_path) as src:
        src_crs       = src.crs
        src_h, src_w  = src.height, src.width
        src_transform = src.transform

    # Load derived PNG as RGBA
    img_arr = np.array(Image.open(png_path).convert("RGBA"))   # (H, W, 4)
    if img_arr.shape[:2] != (src_h, src_w):
        # Resize to match GeoTIFF pixel grid if dimensions differ
        img_arr = np.array(
            Image.fromarray(img_arr).resize((src_w, src_h), Image.BILINEAR)
        )

    dst_transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, SIZE, SIZE)
    dst_rgba = np.zeros((4, SIZE, SIZE), dtype=np.float32)

    # Reproject each RGBA channel independently
    for ch in range(4):
        src_ch = img_arr[:, :, ch].astype(np.float32)
        reproject(
            src_ch, dst_rgba[ch],
            src_transform=src_transform, src_crs=src_crs,
            dst_transform=dst_transform, dst_crs=dem_crs,
            resampling=Resampling.bilinear,
        )

    # Where original alpha was 0 (nodata corners), keep transparent
    alpha = dst_rgba[3]
    alpha_out = np.where(alpha > 10, opacity, 0).astype(np.uint8)

    out_arr = np.stack([
        np.clip(dst_rgba[0], 0, 255).astype(np.uint8),
        np.clip(dst_rgba[1], 0, 255).astype(np.uint8),
        np.clip(dst_rgba[2], 0, 255).astype(np.uint8),
        alpha_out,
    ], axis=-1)

    Image.fromarray(out_arr).save(out_path)
    return True


def _find_scene_tif(scene_folder):
    """Return the path to the 8-band SR GeoTIFF inside a PSScene/ subfolder."""
    ps_dir = os.path.join(scene_folder, "PSScene")
    if not os.path.isdir(ps_dir):
        return None
    for f in os.listdir(ps_dir):
        if f.endswith("_SR_8b_clip.tif"):
            return os.path.join(ps_dir, f)
    return None


def export_all_planetscope_imagery(bounds, dem_crs):
    """Warp the derived RGB and NDVI PNGs for all 10 scenes onto the DEM grid.

    Outputs (per scene):
      ps_{key}_rgb.png  — true-colour composite (R=Red, G=Green, B=Blue)
      ps_{key}_ndvi.png — NDVI (RdYlGn: red=bare/water, yellow=sparse, green=healthy veg)
    """
    print("\nPlanetScope imagery overlays (RGB + NDVI, warped to DEM grid):")
    rgb_ok = ndvi_ok = 0
    for folder, key, label, _, note in PLANETSCOPE_SCENES:
        scene_dir = os.path.join(PLANET_DIR, folder)
        tif_path  = _find_scene_tif(scene_dir)
        if tif_path is None:
            print(f"  {key}: GeoTIFF not found — skipping")
            continue

        deriv = os.path.join(scene_dir, "derived")

        # RGB true-colour
        ok = _warp_png_to_dem_grid(
            os.path.join(deriv, "rgb_truecolor.png"), tif_path,
            bounds, dem_crs,
            os.path.join(OUT_DIR, f"{key}_rgb.png"),
            opacity=245,
        )
        if ok:
            print(f"  {key}_rgb.png  [{note}]")
            rgb_ok += 1

        # NDVI
        ok = _warp_png_to_dem_grid(
            os.path.join(deriv, "ndvi.png"), tif_path,
            bounds, dem_crs,
            os.path.join(OUT_DIR, f"{key}_ndvi.png"),
            opacity=220,
        )
        if ok:
            print(f"  {key}_ndvi.png")
            ndvi_ok += 1

    print(f"  Done: {rgb_ok} RGB + {ndvi_ok} NDVI scenes exported")


def export_hydro_overlays(bounds=None, dem_crs=None):
    """Copy/render hydrological overlays into viewer/data/.

    HAND and flow-accumulation PNGs are copied from the pre-computed dem/data/hydro/ files.
    The stream network is re-rendered from the vectorized GeoJSON so it gets proper
    anti-aliasing and configurable line width (the raw raster PNG looks pixelated at 512×512
    because the D8 grid cells are only ~1 px each after resize).
    """
    import shutil
    HYDRO_OUT = os.path.join(PROJ_DIR, "dem", "data", "hydro")

    # Build a waterbody mask once and reuse for both HAND and stream masking.
    # These are unnamed engineered stormwater retention ponds; D8 routing runs
    # through them and HAND LiDAR artifacts create patchy colour inside them.
    # Masking them to transparent keeps the layer scientifically cleaner.
    wb_mask = None
    if bounds is not None and dem_crs is not None:
        wb_path = os.path.join(PROJ_DIR, "hydrography", "data", "3dhp_waterbodies.geojson")
        if os.path.exists(wb_path):
            wb_gdf = _read_geojson_or_empty(wb_path)
            if not wb_gdf.empty:
                wb_proj = wb_gdf.to_crs(dem_crs)
                dst_transform = from_bounds(
                    bounds.left, bounds.bottom, bounds.right, bounds.top, SIZE, SIZE
                )
                wb_mask = rasterize(
                    [(g, 1) for g in wb_proj.geometry if g is not None],
                    out_shape=(SIZE, SIZE), transform=dst_transform, fill=0, dtype=np.uint8,
                ).astype(bool)

    # HAND: copy pre-computed colourised PNG, then zero-out waterbody pixels so
    # LiDAR artifacts inside ponds don't show partial HAND colouring.
    hand_src = os.path.join(HYDRO_OUT, "hand.png")
    hand_dst = os.path.join(OUT_DIR, "hydro_hand.png")
    if os.path.exists(hand_src):
        hand_img = np.array(Image.open(hand_src).convert("RGBA"))
        if wb_mask is not None:
            hand_img[wb_mask] = 0   # fully transparent over water bodies
        Image.fromarray(hand_img).save(hand_dst)
        print(f"  hydro_hand.png (waterbody mask applied: {int(wb_mask.sum()) if wb_mask is not None else 0} px cleared)")
    else:
        print("  hand.png not found — run dem/dem_hydro.py first")

    # Flow accumulation: copy as-is (stormwater ponds have high accumulation — correct).
    fa_src = os.path.join(HYDRO_OUT, "flow_accum.png")
    fa_dst = os.path.join(OUT_DIR, "hydro_flow_accum.png")
    if os.path.exists(fa_src):
        shutil.copy2(fa_src, fa_dst)
        print("  hydro_flow_accum.png ← dem/data/hydro/")
    else:
        print("  flow_accum.png not found — run dem/dem_hydro.py first")

    # Stream network: render from the vectorized GeoJSON, then mask out waterbody pixels.
    # The D8 stream enters ponds/lakes (correct routing) but the cyan line inside a
    # lake polygon looks wrong visually — the whole lake is already drawn by hydrography.png.
    stream_geojson = os.path.join(HYDRO_OUT, "stream_network.geojson")
    dst_stream = os.path.join(OUT_DIR, "hydro_streams.png")
    if bounds is not None and dem_crs is not None and os.path.exists(stream_geojson):
        gdf = _read_geojson_or_empty(stream_geojson)
        if not gdf.empty:
            gdf_proj = gdf.to_crs(dem_crs)
            dst_transform = from_bounds(
                bounds.left, bounds.bottom, bounds.right, bounds.top, SIZE, SIZE
            )
            buffered = gdf_proj.geometry.buffer(12)
            stream_mask = rasterize(
                [(geom, 1) for geom in buffered if geom is not None],
                out_shape=(SIZE, SIZE), transform=dst_transform, fill=0, dtype=np.uint8,
            ).astype(bool)
            n_before = int(stream_mask.sum())
            if wb_mask is not None:
                stream_mask = stream_mask & ~wb_mask  # erase stream pixels inside ponds
            rgba = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
            rgba[stream_mask] = (0, 230, 255, 230)
            Image.fromarray(rgba).save(dst_stream)
            removed = n_before - int(stream_mask.sum())
            print(f"  hydro_streams.png: {int(stream_mask.sum())} stream px "
                  f"({removed} inside-pond px removed)")
        else:
            print("  stream_network.geojson is empty — skipping hydro_streams.png")
    else:
        src_raster = os.path.join(HYDRO_OUT, "stream_network.png")
        if os.path.exists(src_raster):
            shutil.copy2(src_raster, dst_stream)
            print("  hydro_streams.png ← raster fallback")
        else:
            print("  stream_network.png not found — run dem/dem_hydro.py first")


def export_terrain_overlays():
    """Copy pre-computed terrain PNGs from dem/data/terrain/ into viewer/data/.
    dem_terrain.py already exports 512×512 PNGs aligned to the DEM grid."""
    copies = [
        ("terrain_elevation.png", "terrain_elevation.png"),
        ("terrain_slope.png",     "terrain_slope.png"),
        ("terrain_hillshade.png", "terrain_hillshade.png"),
        ("terrain_tpi.png",       "terrain_tpi.png"),
        ("terrain_curvature.png", "terrain_curvature.png"),
        ("terrain_tri.png",       "terrain_tri.png"),
    ]
    import shutil
    any_copied = False
    for src_name, dst_name in copies:
        src = os.path.join(TERRAIN_DIR, src_name)
        dst = os.path.join(OUT_DIR, dst_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  {dst_name} ← dem/data/terrain/")
            any_copied = True
        else:
            print(f"  {src_name} not found — run dem/dem_terrain.py first")
    if not any_copied:
        print("  No terrain PNGs found — run: python3 dem/dem_terrain.py")


PLANETSCOPE_SCENES = [
    # (folder_name, short_key, label, color_rgb, note)
    ("Florida_Hydrology_Max_Precipitation_1", "ps_max1", "PlanetScope: Ian Flood 2022-09-30 (MAX)",    (0, 180, 255), "Hurricane Ian peak; 25.9 ha; NWIS 3500 cfs"),
    ("Florida_Hydrology_Max_Precipitation_2", "ps_max2", "PlanetScope: 2025-05-29 (MAX)",              (0, 210, 180), "Strong event; only 3.8 ha water"),
    ("Florida_Hydrology_Max_Precipitation_3", "ps_max3", "PlanetScope: 2024-06-30 (MAX)",              (0, 220, 140), "Early wet season; 11.6 ha"),
    ("Florida_Hydrology_Avg_Precipitation_1", "ps_avg1", "PlanetScope: 2025-06-17 (AVG wet)",          (80, 160, 255), "Typical wet season; 7.3 ha"),
    ("Florida_Hydrology_Avg_Precipitation_2", "ps_avg2", "PlanetScope: 2023-09-17 (AVG wet)",          (100, 170, 255), "Typical wet season; 25.8 ha"),
    ("Florida_Hydrology_Avg_Precipitation_3", "ps_avg3", "PlanetScope: 2025-12-04 (AVG dry)",          (130, 180, 220), "Typical dry season; 24.4 ha"),
    ("Florida_Hydrology_Avg_Precipitation_4", "ps_avg4", "PlanetScope: 2023-12-23 (AVG dry)",          (140, 185, 210), "Typical dry season; 20.0 ha"),
    ("Florida_Hydrology_Min_Precipitation_1", "ps_min1", "PlanetScope: 2021-03-17 (MIN 43-day dry)",   (180, 200, 160), "43-day dry streak; baseline 17.0 ha"),
    ("Florida_Hydrology_Min_Precipitation_2", "ps_min2", "PlanetScope: 2024-12-09 (MIN 34-day dry)",   (190, 205, 155), "34-day dry streak; 20.5 ha"),
    ("Florida_Hydrology_Min_Precipitation_3", "ps_min3", "PlanetScope: 2022-04-19 (MIN 33-day dry)",   (200, 210, 150), "33-day dry streak; 15.5 ha"),
]


def export_planetscope_ian(bounds, dem_crs):
    """Backwards-compat alias — exports the Ian scene (Max-1) as ps_ian_water.png."""
    ian_entry = PLANETSCOPE_SCENES[0]
    _export_one_ps_scene(ian_entry, bounds, dem_crs, out_name="ps_ian_water.png")


def _export_one_ps_scene(scene_entry, bounds, dem_crs, out_name=None):
    folder, key, label, color_rgb, note = scene_entry
    out_file = out_name or f"{key}_water.png"
    wb_path = os.path.join(PLANET_DIR, folder, "derived", "water_boundaries_wgs84.geojson")
    if not os.path.exists(wb_path):
        print(f"  {out_file} — water boundaries not found in {folder}/derived/")
        return
    gdf = _read_geojson_or_empty(wb_path)
    if gdf.empty:
        print(f"  {out_file} — water boundaries empty")
        return
    gdf_proj = gdf.to_crs(dem_crs)
    dst_transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, SIZE, SIZE)
    rgba = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
    mask = rasterize(
        [(geom, 1) for geom in gdf_proj.geometry if geom is not None],
        out_shape=(SIZE, SIZE), transform=dst_transform, fill=0, dtype=np.uint8,
    ).astype(bool)
    r, g, b = color_rgb
    rgba[mask] = (r, g, b, 200)
    img = Image.fromarray(rgba)
    img.save(os.path.join(OUT_DIR, out_file))
    print(f"  {out_file} saved  water={len(gdf)} bodies  [{note}]")


def export_all_planetscope_scenes(bounds, dem_crs):
    """Rasterize all 10 PlanetScope water boundaries as individually-toggleable viewer PNGs.
    MAX scenes use bright cyan-blue; AVG mid-blue; MIN muted green-blue.
    Ian (Max-1) is also exported as ps_ian_water.png for backward compatibility."""
    for scene in PLANETSCOPE_SCENES:
        _export_one_ps_scene(scene, bounds, dem_crs)
    # Keep ps_ian_water.png for backward compat with server REQUIRED list
    _export_one_ps_scene(PLANETSCOPE_SCENES[0], bounds, dem_crs, out_name="ps_ian_water.png")


def export_fema_hand_risk():
    """Copy the FEMA×HAND risk PNG produced by analysis/fema_hand_crossref.py."""
    src = os.path.join(PROJ_DIR, "analysis", "data", "fema_hand_risk.png")
    dst = os.path.join(OUT_DIR, "fema_hand_risk.png")
    if not os.path.exists(src):
        print("  fema_hand_risk.png not found — run: python3 analysis/fema_hand_crossref.py")
        return
    import shutil
    shutil.copy2(src, dst)
    print(f"  fema_hand_risk.png  ({os.path.getsize(dst)//1024} KB)")


def export_fema_sim_extent():
    """Copy the FEMA-vs-simulated-Ian-extent overlay PNG (analysis/fema_sim_extent_crossref.py)."""
    src = os.path.join(PROJ_DIR, "analysis", "data", "fema_sim_extent_overlay.png")
    dst = os.path.join(OUT_DIR, "fema_sim_extent_overlay.png")
    if not os.path.exists(src):
        print("  fema_sim_extent_overlay.png not found — run: python3 analysis/fema_sim_extent_crossref.py")
        return
    import shutil
    shutil.copy2(src, dst)
    print(f"  fema_sim_extent_overlay.png  ({os.path.getsize(dst)//1024} KB)")


def export_ian_flood():
    """Copy the Hurricane Ian flood simulation viewer PNG."""
    src = os.path.join(PROJ_DIR, "simulation", "outputs", "ian_flood_viewer.png")
    dst = os.path.join(OUT_DIR, "ian_flood_viewer.png")
    if not os.path.exists(src):
        print("  ian_flood_viewer.png not found — run: python3 simulation/flood_sim_ian.py")
        return
    import shutil
    shutil.copy2(src, dst)
    print(f"  ian_flood_viewer.png  ({os.path.getsize(dst)//1024} KB)")


def main():
    bounds, dem_crs = get_dem_bounds()
    print(f"DEM bounds: {bounds}  CRS: {dem_crs}")
    export_naip(bounds, dem_crs)
    export_ssurgo(bounds, dem_crs)
    export_hydrography(bounds, dem_crs)
    export_floodplain(bounds, dem_crs)
    export_boundary(bounds, dem_crs)
    export_roads_buildings(bounds, dem_crs)
    export_hydro_overlays(bounds, dem_crs)
    export_terrain_overlays()
    export_fema_hand_risk()
    export_fema_sim_extent()
    export_ian_flood()
    export_all_planetscope_imagery(bounds, dem_crs)
    export_all_planetscope_scenes(bounds, dem_crs)
    print("Done — viewer/data/ overlays ready")


if __name__ == "__main__":
    main()
