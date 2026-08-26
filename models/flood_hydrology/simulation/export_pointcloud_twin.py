#!/usr/bin/env python3
"""Export the FULL raw LiDAR point cloud (every real point in the 25x25m box, all
classes except noise) for 17801 Champagne Dr — a direct visual cross-check layer for
confirming that the solved ground/roof MESH lines up with the raw scattered LiDAR
returns it was built from.

Uses the exact same coordinate pipeline as build_mesh_twin.py (same LAZ file, same
property-center transform, same HALF_M box, same FT2M, same x=east/y=north/z=up local
axes) so this point cloud is guaranteed to sit in the identical local frame as the
already-exported mesh/tracers -- not re-derived or approximated.

Only 23,725 real points fall in this box at native density (measured directly), so no
decimation is needed -- every real point is included.
"""
import os, json, base64
import numpy as np
import rasterio
from pyproj import Transformer

PROP_LAT, PROP_LON = 28.5217321, -81.6570725
LAZ = "/Users/hqqq422/Desktop/deepearth/models/flood_hydrology/lidar/data/raw/USGS_LPC_FL_Peninsular_FDEM_2018_D19_DRRA_LID2019_258656_E.laz"
HALF_M = 12.5
FT2M = 0.3048006096012192
OUT_DIR = "/Users/hqqq422/Desktop/deepearth/models/flood_hydrology/simulation/outputs"
VIEWER_HTML = "/Users/hqqq422/Desktop/deepearth/models/flood_hydrology/simulation/twin_mesh_viewer.html"
NAIP_PATH = "/Users/hqqq422/Desktop/deepearth/models/flood_hydrology/soil/data/naip_2021_RGB.tif"

# fallback only, if NAIP doesn't cover a point for some reason
CLASS_COLOR_FALLBACK = {
    1: (140, 140, 148),   # unclassified -- cool gray
    2: (196, 172, 108),   # ground -- tan
    6: (214, 92, 62),     # building -- warm red-orange
}


def color_by_naip(x_native, y_native, native_crs):
    """Real aerial-photo color per point -- same technique this project's own established
    full-point-cloud convention already uses (lidar/export_full_pointcloud.py, both here and
    in the sibling cfx_sr417_corridor project), instead of a flat per-class color. Matters a
    lot for this specific site: this LAZ tile's vendor classification never split vegetation
    into its own ASPRS classes (3/4/5) -- real tree canopy returns are lumped into class 1
    "unclassified" along with everything else, so a flat single-color fill for that whole
    class reads as a formless gray haze. NAIP coloring gives canopy points their real green,
    roof points their real color, etc., regardless of what classification bucket they landed in."""
    with rasterio.open(NAIP_PATH) as src:
        img = src.read()  # (3, H, W) uint8
        transform, crs = src.transform, src.crs
        H, W = src.height, src.width
    tr = Transformer.from_crs(native_crs, crs, always_xy=True)
    nx, ny = tr.transform(x_native, y_native)
    rows, cols = rasterio.transform.rowcol(transform, nx, ny)
    rows = np.clip(np.asarray(rows), 0, H - 1)
    cols = np.clip(np.asarray(cols), 0, W - 1)
    return img[:, rows, cols].T.astype(np.uint8)  # (n, 3)


def load_full_cloud():
    import laspy
    las = laspy.read(LAZ)
    crs = las.header.parse_crs()
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    cx, cy = tr.transform(PROP_LON, PROP_LAT)
    x = np.asarray(las.x); y = np.asarray(las.y); z = np.asarray(las.z)
    cl = np.asarray(las.classification)
    xm = (x - cx) * FT2M
    ym = (y - cy) * FT2M
    zm = z * FT2M
    in_box = (np.abs(xm) < HALF_M) & (np.abs(ym) < HALF_M) & (cl != 7)  # drop noise (class 7)

    x, y, xm, ym, zm, cl = x[in_box], y[in_box], xm[in_box], ym[in_box], zm[in_box], cl[in_box]
    print(f"    full raw cloud: {len(xm):,} real points in the 25x25m box "
          f"(classes: {dict(zip(*np.unique(cl, return_counts=True)))})")

    rgb = color_by_naip(x, y, crs)
    # only true zero (rasterio nodata / off-raster) falls back to a per-class flat color
    missing = (rgb.sum(axis=1) == 0)
    if missing.any():
        for c, col in CLASS_COLOR_FALLBACK.items():
            rgb[missing & (cl == c)] = col
        print(f"    {missing.sum():,} points fell outside NAIP coverage -> classification fallback color")

    pts = np.stack([xm, ym, zm], axis=1).astype(np.float32)
    return pts, rgb


def main():
    pts, rgb = load_full_cloud()
    pts_b64 = base64.b64encode(pts.tobytes()).decode()
    col_b64 = base64.b64encode(rgb.tobytes()).decode()

    with open(VIEWER_HTML, "r") as f:
        content = f.read()

    marker = "const DEPTH = "
    idx = content.find(marker)
    end = content.find(";\nconst OBJ_TEXT", idx)
    assert idx >= 0 and end > idx, "could not locate DEPTH block in viewer HTML"
    depth = json.loads(content[idx + len(marker):end])

    depth["cloud_n"] = int(len(pts))
    depth["cloud_pts_b64"] = pts_b64
    depth["cloud_col_b64"] = col_b64

    new_block = marker + json.dumps(depth)
    content = content[:idx] + new_block + content[end:]
    with open(VIEWER_HTML, "w") as f:
        f.write(content)

    print(f"    injected cloud_n={depth['cloud_n']:,} into {VIEWER_HTML}")
    # also keep a standalone copy on disk for reference / reuse by other sites later
    out = os.path.join(OUT_DIR, "pointcloud_twin.json")
    with open(out, "w") as f:
        json.dump(dict(cloud_n=depth["cloud_n"], cloud_pts_b64=pts_b64, cloud_col_b64=col_b64,
                        site=dict(lat=PROP_LAT, lon=PROP_LON, half_m=HALF_M)), f)
    print(f"    wrote {out} ({os.path.getsize(out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
