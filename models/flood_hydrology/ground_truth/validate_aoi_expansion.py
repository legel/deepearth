"""
AOI-expansion scoping/validation — see CLAUDE.md "Future work" item 2.

1. Fetch Johns Lake's real shoreline geometry from USGS NHDPlus HR (via pynhd)
   and derive a bounding box (+250 m buffer) that should fully contain it.
2. Download a DEM at that bbox (separate file, does NOT touch the production
   dem/data/winter_garden_dem.tif) and rasterize the NHD polygon onto it to
   build a preliminary NHD-only lake mask (no OWM majority-vote).
3. Check the 4 grid edges for wet-pixel clipping, the same method that found
   the current AOI's 502/450/127/5 edge-clipping problem.

Does NOT touch Sentinel-2, water segmentation, the lake mask consensus, or
any production file. Outputs land in ground_truth/aoi_expansion_test_data/.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import rasterio
from rasterio.features import rasterize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "aoi_expansion_test_data")
os.makedirs(OUT_DIR, exist_ok=True)

SEARCH_BBOX_4326 = (-81.70, 28.49, -81.62, 28.55)  # generous search window around Johns Lake
BUFFER_M = 250  # watershed-context buffer, matches the current AOI's rationale
RESOLUTION_M = 3


def fetch_johns_lake_polygon():
    import pynhd
    wb = pynhd.NHD("waterbody_hr")
    gdf = wb.bygeom(SEARCH_BBOX_4326, geo_crs="epsg:4326")
    johns = gdf.sort_values("AREASQKM", ascending=False).iloc[0]
    assert johns["GNIS_NAME"] == "Johns Lake", f"largest waterbody in search box was {johns['GNIS_NAME']!r}, not Johns Lake"
    print(f"NHD Johns Lake: {johns['AREASQKM']:.3f} km^2 = {johns['AREASQKM']*100:.1f} ha "
          f"= {johns['AREASQKM']*247.105:.1f} acres, {len(johns.geometry.geoms)} parts")
    return gdf.loc[[johns.name]]  # single-row GeoDataFrame, geographic CRS (4326)


def bbox_with_buffer(geom_gdf, buffer_m):
    west, south, east, north = geom_gdf.total_bounds
    lat_c = (south + north) / 2
    dlat = buffer_m / 111_000
    dlon = buffer_m / (111_000 * np.cos(np.radians(lat_c)))
    return west - dlon, south - dlat, east + dlon, north + dlat


def download_dem(bbox_4326, out_path, resolution=RESOLUTION_M):
    import py3dep
    import rioxarray  # noqa: F401
    print(f"Requesting DEM at {resolution}m for bbox {bbox_4326} ...")
    dem = py3dep.get_dem(bbox_4326, crs="epsg:4326", resolution=resolution)
    dem.rio.to_raster(out_path)
    print(f"  saved {out_path}  shape={dem.shape}  crs={dem.rio.crs}")
    return out_path


def build_nhd_mask(dem_path, lake_gdf, out_path):
    with rasterio.open(dem_path) as src:
        dem_t, dem_crs, dem_shape = src.transform, src.crs, src.shape

    lake_proj = lake_gdf.to_crs(dem_crs)
    mask = rasterize(
        [(geom, 1) for geom in lake_proj.geometry],
        out_shape=dem_shape, transform=dem_t, fill=0, dtype=np.uint8,
    ).astype(bool)

    with rasterio.open(dem_path) as src:
        profile = src.profile
    profile.update(dtype=rasterio.uint8, count=1, nodata=0)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)

    return mask, dem_t


def edge_check(mask):
    n, s, w, e = mask[0, :].sum(), mask[-1, :].sum(), mask[:, 0].sum(), mask[:, -1].sum()
    print(f"Edge wet-pixel counts — north: {n}, south: {s}, west: {w}, east: {e}  "
          f"(grid shape {mask.shape})")
    return dict(north=int(n), south=int(s), west=int(w), east=int(e))


def main():
    lake_gdf = fetch_johns_lake_polygon()
    bbox = bbox_with_buffer(lake_gdf, BUFFER_M)
    west, south, east, north = bbox
    km_ew = (east - west) * 111 * np.cos(np.radians((south + north) / 2))
    km_ns = (north - south) * 111
    print(f"Buffered bbox (4326): {bbox}")
    print(f"Box size: {km_ew:.2f} km (E-W) x {km_ns:.2f} km (N-S)  (current AOI: 2.0 x 2.0 km)")

    dem_path = os.path.join(OUT_DIR, "dem_aoi_expansion_test.tif")
    download_dem(bbox, dem_path)

    mask_path = os.path.join(OUT_DIR, "lake_mask_nhd_aoi_expansion_test.tif")
    mask, dem_t = build_nhd_mask(dem_path, lake_gdf, mask_path)

    res_x, res_y = dem_t.a, -dem_t.e
    area_ha = mask.sum() * res_x * res_y / 1e4
    grid_area_ha = mask.size * res_x * res_y / 1e4
    print(f"NHD mask wet area on new grid: {area_ha:.1f} ha  (grid total: {grid_area_ha:.1f} ha)")

    edges = edge_check(mask)
    clipped = any(v > 5 for v in edges.values())  # >5 px ~ noise floor, matches current AOI's "south: 5 = clear" baseline
    print()
    print("RESULT:", "STILL CLIPPED on >=1 edge" if clipped else "NO EDGE CLIPPING — extent fully contains the lake")


if __name__ == "__main__":
    main()
