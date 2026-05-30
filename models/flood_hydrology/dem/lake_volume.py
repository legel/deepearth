"""
Lake Volume Calculation — Voxelization + Marching Cubes Mesh
=============================================================
Explicitly calculates lake volume using a voxel-counting approach on the DEM
and the estimated lake-bed DEM, then visualizes the 3D lake body.

Method (as described by team lead)
-----------------------------------
1. Define lake extent from dem/data/lake_mask.tif
2. Water surface elevation = mean elevation of lake cells in the hydro-flattened DEM
3. Lake bed elevation = from lake_bed_dem.tif (extrapolated from shoreline slope)
4. For each lake grid cell:
      depth(x,y) = water_surface_elev − lake_bed_elev(x,y)
      volume contribution = depth(x,y) × cell_area_m²
5. Total volume = Σ depth(x,y) × cell_area_m²
   (equivalent to filling the lake with unit voxels cell_m × cell_m × 1m and
    stacking them to the water surface — same as a 3D integral)
6. Marching cubes (skimage.measure) creates a triangulated mesh of the lake
   volume for 3D visualization.

Seasonal volume check
---------------------
Uses Sentinel-2 MNDWI water masks to adjust lake extent per season,
then recalculates volume at each season to check for variation.

Outputs
-------
    dem/lake_volume.png        — 6-panel visualization
    dem/data/lake_volume.csv   — volume per lake region + seasonal table

Usage:
    python3 dem/lake_volume.py
"""

import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors
import rasterio
import pandas as pd

try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("  skimage not found — marching cubes mesh skipped (pip install scikit-image)")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
S2_DIR   = os.path.join(BASE_DIR, "..", "sentinel2", "data")
OUT_PNG  = os.path.join(BASE_DIR, "lake_volume.png")
OUT_CSV  = os.path.join(DATA_DIR, "lake_volume.csv")


def load_tif(fname, data_dir=DATA_DIR):
    path = os.path.join(data_dir, fname)
    if not os.path.exists(path):
        return None, None
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nd  = src.nodata
        transform = src.transform
        crs = src.crs
    if nd is not None:
        arr[arr == nd] = np.nan
    return arr, (transform, abs(transform.a))


def voxel_volume(lake_mask_bool, dem, lake_bed, cell_m, label=""):
    """Compute lake volume by integrating depth over lake area."""
    valid = lake_mask_bool & np.isfinite(dem) & np.isfinite(lake_bed)
    if valid.sum() == 0:
        return 0.0, 0.0, 0.0

    water_surface = float(np.nanmean(dem[lake_mask_bool & np.isfinite(dem)]))
    depth = np.where(valid, np.maximum(water_surface - lake_bed, 0.0), 0.0)
    cell_area_m2 = cell_m ** 2
    vol_m3  = float(depth.sum() * cell_area_m2)
    area_ha = float(valid.sum() * cell_area_m2 / 1e4)
    mean_depth = float(depth[valid].mean()) if valid.any() else 0.0

    if label:
        print(f"  {label}:")
        print(f"    Area          : {area_ha:.2f} ha")
        print(f"    Mean depth    : {mean_depth:.2f} m")
        print(f"    Max depth     : {float(depth.max()):.2f} m")
        print(f"    Volume        : {vol_m3:,.0f} m³  ({vol_m3/1e6:.4f} km³  |  {vol_m3*264.172:.0f} gal)")
    return vol_m3, area_ha, mean_depth


def build_voxel_grid(lake_mask_bool, dem, lake_bed, cell_m, z_resolution=0.5):
    """
    Build a 3D boolean voxel grid of the lake volume.
    Each voxel is cell_m × cell_m × z_resolution m.
    Returns voxel_grid[row, col, z_layer] = True if inside lake water body.
    """
    valid = lake_mask_bool & np.isfinite(dem) & np.isfinite(lake_bed)
    water_surface = float(np.nanmean(dem[lake_mask_bool & np.isfinite(dem)]))
    depth_map = np.where(valid, np.maximum(water_surface - lake_bed, 0.0), 0.0)
    max_depth = float(depth_map.max())

    if max_depth < 0.1:
        print("  ⚠ Max depth < 0.1 m — voxel grid trivially empty")
        return None, water_surface, 0

    n_zlayers = max(1, int(np.ceil(max_depth / z_resolution)))
    z_levels = np.arange(0, n_zlayers) * z_resolution  # depth below surface

    rows, cols = np.where(valid)
    voxels = np.zeros((dem.shape[0], dem.shape[1], n_zlayers), dtype=bool)
    for z_idx, z_depth in enumerate(z_levels):
        voxels[:, :, z_idx] = valid & (depth_map > z_depth)

    total_voxels = int(voxels.sum())
    voxel_vol_m3 = total_voxels * cell_m * cell_m * z_resolution
    print(f"  Voxel grid     : {dem.shape[0]}×{dem.shape[1]}×{n_zlayers} layers")
    print(f"  Voxel size     : {cell_m:.1f}m × {cell_m:.1f}m × {z_resolution:.1f}m")
    print(f"  Filled voxels  : {total_voxels:,}")
    print(f"  Voxel volume   : {voxel_vol_m3:,.0f} m³")
    return voxels, water_surface, n_zlayers


def main():
    print("Loading rasters …")
    dem,      (transform, cell_m) = load_tif("winter_garden_dem.tif")
    # Prefer S2-derived lake mask (135k cells, true open-water surface confirmed by
    # optical imagery) over the DEM-based mask (216k cells, includes misclassified
    # flat suburban terrain). Fall back to DEM mask if S2 mask not available.
    _mask_src = ("lake_mask_s2.tif"
                 if os.path.exists(os.path.join(DATA_DIR, "lake_mask_s2.tif"))
                 else "lake_mask.tif")
    lake_mask, _                  = load_tif(_mask_src)
    # Prefer FWC bathymetric survey; fall back to shoreline-slope estimate
    _bed_src = ("lake_bed_dem_fwc.tif"
                if os.path.exists(os.path.join(DATA_DIR, "lake_bed_dem_fwc.tif"))
                else "lake_bed_dem_estimated.tif")
    lake_bed, _                   = load_tif(_bed_src)
    print(f"  Lake mask source: {_mask_src}")
    print(f"  Lake bed source : {_bed_src}")

    if dem is None or lake_mask is None or lake_bed is None:
        sys.exit("Missing DEM files. Run dem_download.py and dem_process.py first.")

    lake_bool = (lake_mask == 1) & np.isfinite(dem)
    if lake_bool.sum() == 0:
        sys.exit("No lake cells found in lake_mask.tif")

    # ── 1. Baseline volume (full lake mask from DEM) ─────────────────────
    print("\n── Baseline Lake Volume (DEM lake mask) ─────────────────────")
    vol_m3, area_ha, mean_depth = voxel_volume(lake_bool, dem, lake_bed, cell_m, label="All lake regions")
    water_surface = float(np.nanmean(dem[lake_bool & np.isfinite(dem)]))
    max_depth = float(np.nanmax(np.maximum(water_surface - lake_bed[lake_bool & np.isfinite(lake_bed)], 0)))

    # ── 2. Build voxel grid for marching cubes ───────────────────────────
    print("\n── Voxel Grid Construction ──────────────────────────────────")
    voxels, ws_elev, n_z = build_voxel_grid(lake_bool, dem, lake_bed, cell_m, z_resolution=0.5)

    # ── 3. Seasonal volumes from Sentinel-2 MNDWI masks ─────────────────
    print("\n── Seasonal Volume Check (Sentinel-2 water masks) ───────────")
    s2_files = sorted(glob.glob(os.path.join(S2_DIR, "s2_*_B03.tif")))
    seasonal_rows = []

    for b03_path in s2_files:
        date = os.path.basename(b03_path).replace("s2_", "").replace("_B03.tif", "")
        b11_path = b03_path.replace("_B03.tif", "_B11.tif")
        if not os.path.exists(b11_path):
            continue
        try:
            with rasterio.open(b03_path) as src:
                b03 = src.read(1).astype(np.float32)
            with rasterio.open(b11_path) as src_11:
                b11_raw = src_11.read(1).astype(np.float32)
                if b11_raw.shape != b03.shape:
                    from scipy.ndimage import zoom
                    scale = (b03.shape[0]/b11_raw.shape[0], b03.shape[1]/b11_raw.shape[1])
                    b11_raw = zoom(b11_raw, scale, order=1)
            b11 = b11_raw
            with np.errstate(divide="ignore", invalid="ignore"):
                mndwi = np.where((b03 + b11) > 0, (b03 - b11) / (b03 + b11), 0)

            # S2 water mask at this date (10m resolution)
            s2_water = mndwi > 0
            s2_area_ha = float(s2_water.sum() * 10 * 10 / 1e4)

            # Project S2 water mask back onto DEM grid for volume calc
            # Use DEM lake mask refined by S2: intersection approach
            # Resize S2 mask to DEM resolution
            from scipy.ndimage import zoom as nd_zoom
            scale_r = dem.shape[0] / s2_water.shape[0]
            scale_c = dem.shape[1] / s2_water.shape[1]
            s2_on_dem = nd_zoom(s2_water.astype(np.float32), (scale_r, scale_c), order=0) > 0.5

            # Volume for this date: use S2-derived extent × DEM-derived depth
            s2_lake_for_vol = s2_on_dem & np.isfinite(dem) & np.isfinite(lake_bed)
            if s2_lake_for_vol.sum() < 10:
                s2_lake_for_vol = lake_bool  # fallback

            v, a, d = voxel_volume(s2_lake_for_vol, dem, lake_bed, cell_m)
            month = int(date[4:6])
            season = ("dry" if month in [1,2,3,4,5] else
                      "wet" if month in [6,7,8,9] else "shoulder")
            seasonal_rows.append({
                "date": date, "season": season,
                "s2_area_ha": round(s2_area_ha, 2),
                "dem_volume_m3": round(v, 0),
                "mean_depth_m": round(d, 3),
            })
            print(f"  {date} ({season}): S2 area={s2_area_ha:.1f}ha, vol={v:,.0f}m³, depth={d:.2f}m")
        except Exception as e:
            print(f"  {date}: skipped ({e})")

    seasonal_df = pd.DataFrame(seasonal_rows) if seasonal_rows else pd.DataFrame()

    # ── 4. Save CSV ──────────────────────────────────────────────────────
    rows_csv = [{"region": "all_lakes", "area_ha": round(area_ha,2),
                 "volume_m3": round(vol_m3,0), "mean_depth_m": round(mean_depth,3),
                 "max_depth_m": round(max_depth,3), "water_surface_m": round(water_surface,2)}]
    pd.DataFrame(rows_csv).to_csv(OUT_CSV, index=False)
    if not seasonal_df.empty:
        seasonal_df.to_csv(OUT_CSV.replace(".csv","_seasonal.csv"), index=False)
    print(f"\nSaved volume CSV → {OUT_CSV}")

    # ── 5. Visualize ─────────────────────────────────────────────────────
    print("\nBuilding visualization …")
    fig = plt.figure(figsize=(20, 13))
    fig.suptitle(
        "Lake Volume Analysis — 17801 Champagne Dr, Winter Garden FL\n"
        f"Voxelization: {cell_m:.1f}m×{cell_m:.1f}m grid | "
        f"Baseline volume = {vol_m3:,.0f} m³ ({area_ha:.1f} ha, mean depth {mean_depth:.2f} m)",
        fontsize=12, fontweight="bold"
    )

    # Panel 1: Lake depth map (2D)
    ax1 = fig.add_subplot(2, 3, 1)
    depth_map = np.where(lake_bool & np.isfinite(lake_bed),
                         np.maximum(water_surface - lake_bed, 0), np.nan)
    im1 = ax1.imshow(depth_map, cmap="Blues", origin="upper", vmin=0, vmax=max(max_depth, 1.0))
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.set_label("Depth [m] (dark = deeper)", fontsize=8)
    ax1.set_title(f"Lake depth map\n(water surface {water_surface:.1f}m − lake bed)\n"
                  f"Max depth: {max_depth:.2f}m", fontsize=9)
    ax1.set_xlabel("col"); ax1.set_ylabel("row")
    ax1.text(0.02, 0.02,
             "⚠ Depth from shoreline slope extrapolation\n"
             "FL karst lakes typically 2–6 m deep at centre\n"
             "FWC/FDEP bathymetry survey data pending",
             transform=ax1.transAxes, fontsize=6.5, color="darkred", va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.85))

    # Panel 2: Depth histogram
    ax2 = fig.add_subplot(2, 3, 2)
    depths_valid = depth_map[np.isfinite(depth_map) & (depth_map > 0)]
    if len(depths_valid) > 0:
        ax2.hist(depths_valid, bins=50, color="steelblue", edgecolor="none", alpha=0.8)
        ax2.axvline(mean_depth, color="red", lw=2, label=f"Mean: {mean_depth:.2f}m")
        ax2.axvline(max_depth, color="orange", lw=1.5, linestyle="--", label=f"Max: {max_depth:.2f}m")
        ax2.set_xlabel("Depth [m]")
        ax2.set_ylabel("Cell count")
        ax2.legend(fontsize=9)
    ax2.set_title(f"Depth distribution\n"
                  f"Volume = Σ depth × {cell_m:.1f}² m² = {vol_m3:,.0f} m³\n"
                  f"({vol_m3/1e6:.4f} km³ | {area_ha:.1f} ha total area)", fontsize=9)

    # Panel 3: VAE curve (Volume-Area-Elevation)
    ax3 = fig.add_subplot(2, 3, 3)
    z_steps = np.linspace(water_surface - max(max_depth, 0.5), water_surface, 60)
    vae_vols = []
    vae_areas = []
    for z in z_steps:
        submerged = lake_bool & np.isfinite(lake_bed) & (lake_bed < z)
        depth_at_z = np.where(submerged, z - lake_bed, 0.0)
        vae_vols.append(float(depth_at_z.sum() * cell_m**2))
        vae_areas.append(float(submerged.sum() * cell_m**2 / 1e4))
    ax3.plot(vae_vols, z_steps, color="steelblue", lw=2)
    ax3.axhline(water_surface, color="royalblue", lw=1.5, linestyle="--",
                label=f"Current surface: {water_surface:.1f}m")
    ax3.fill_betweenx(z_steps, vae_vols, alpha=0.3, color="steelblue")
    ax3.set_xlabel("Cumulative volume [m³]")
    ax3.set_ylabel("Water surface elevation [m NAVD88]")
    ax3.set_title("Volume–Elevation (VAE) curve\n"
                  "(How volume changes with lake level rise/fall)", fontsize=9)
    ax3.legend(fontsize=8)

    # Panel 4: 3D lake volume — z-axis = depth below surface (0=surface, negative=deeper)
    ax4 = fig.add_subplot(2, 3, 4, projection="3d")
    rows_idx, cols_idx = np.where(lake_bool & np.isfinite(lake_bed))
    if len(rows_idx) > 0:
        step = max(1, len(rows_idx) // 3000)
        rs = rows_idx[::step]; cs = cols_idx[::step]
        xs = cs * cell_m / 1000   # km E-W
        ys = rs * cell_m / 1000   # km N-S
        zs_bed   = lake_bed[rs, cs]
        valid_3d = np.isfinite(zs_bed)
        # Depth below surface: 0 at water surface, negative going down to lake bed
        zs_depth_bed = -(water_surface - zs_bed)
        if valid_3d.sum() > 0:
            depth_vals = np.abs(zs_depth_bed[valid_3d])   # positive = metres deep
            dmax = float(depth_vals.max()) if len(depth_vals) else 1.0
            # Lake bed coloured by depth: deeper = darker blue
            # Blues: 0=white (surface/shallow), max=dark blue (deepest) — correct direction
            sc = ax4.scatter(xs[valid_3d], ys[valid_3d], zs_depth_bed[valid_3d],
                             c=depth_vals, cmap="Blues",
                             vmin=0, vmax=dmax, s=3, alpha=0.85,
                             label=f"Lake bed (FWC survey, max {dmax:.1f} m)")
            # Water surface at z=0 as a semi-transparent overlay
            ax4.scatter(xs[valid_3d], ys[valid_3d], np.zeros(valid_3d.sum()),
                        c="lightcyan", s=1, alpha=0.15, label="Water surface (z=0)")
            # Vertical depth lines every ~100th point
            step2 = max(1, valid_3d.sum() // 80)
            for i in range(0, valid_3d.sum(), step2):
                idx = np.where(valid_3d)[0][i]
                ax4.plot([xs[idx], xs[idx]], [ys[idx], ys[idx]],
                         [zs_depth_bed[idx], 0],
                         color="steelblue", alpha=0.15, lw=0.4)
            try:
                plt.colorbar(sc, ax=ax4, label="Depth [m]", shrink=0.5, pad=0.12)
            except Exception:
                pass
        ax4.set_xlabel("E–W [km]", fontsize=7); ax4.set_ylabel("N–S [km]", fontsize=7)
        ax4.set_zlabel("Depth below surface [m]", fontsize=7)
        # Z-axis: surface at 0, extend below actual max depth
        ax4.set_zlim(-(max_depth * 1.15), 0.3)
        bed_src_label = _bed_src.replace("lake_bed_dem_", "").replace(".tif", "")
        ax4.set_title(f"3D lake volume (depth below surface)\n"
                      f"Max depth: {max_depth:.1f} m | Source: {bed_src_label}\n"
                      f"Colour: darker blue = deeper", fontsize=8)
        ax4.view_init(elev=30, azim=-60)

    # Panel 5: Marching cubes mesh (if skimage available)
    ax5 = fig.add_subplot(2, 3, 5, projection="3d")
    if HAS_SKIMAGE and voxels is not None and voxels.shape[2] > 1:
        try:
            # Pad voxel grid for clean mesh boundaries
            vpad = np.pad(voxels.astype(np.uint8), pad_width=1, mode="constant", constant_values=0)
            verts, faces, normals, _ = measure.marching_cubes(vpad.astype(np.float32), level=0.5)
            # Scale vertices to real coordinates
            verts_km = verts.copy()
            verts_km[:, 0] = (verts[:, 0] - 1) * cell_m / 1000  # row → N-S km
            verts_km[:, 1] = (verts[:, 1] - 1) * cell_m / 1000  # col → E-W km
            verts_km[:, 2] = (verts[:, 2] - 1) * 0.5            # z_layer → depth m (from surface down)
            verts_km[:, 2] = water_surface - verts_km[:, 2]       # flip: surface at top

            mesh = Poly3DCollection(verts_km[faces], alpha=0.3, linewidth=0)
            mesh.set_facecolor("steelblue")
            mesh.set_edgecolor("none")
            ax5.add_collection3d(mesh)
            ax5.set_xlim(verts_km[:,0].min(), verts_km[:,0].max())
            ax5.set_ylim(verts_km[:,1].min(), verts_km[:,1].max())
            ax5.set_zlim(verts_km[:,2].min(), water_surface + 0.5)
            ax5.set_xlabel("N–S [km]"); ax5.set_ylabel("E–W [km]"); ax5.set_zlabel("Elev [m]")
            ax5.set_title(f"Marching cubes mesh\n"
                          f"({len(faces):,} triangles from voxel grid)\n"
                          f"Triangulated lake volume surface", fontsize=8)
            ax5.view_init(elev=25, azim=45)
        except Exception as e:
            ax5.text(0.5, 0.5, 0.5, f"Mesh failed:\n{e}", transform=ax5.transAxes,
                     ha="center", fontsize=8)
    else:
        ax5.text(0.5, 0.5, f"{'scikit-image not installed' if not HAS_SKIMAGE else 'voxel grid empty'}\n"
                 "pip install scikit-image",
                 transform=ax5.transAxes, ha="center", va="center", fontsize=9)
        ax5.set_title("Marching cubes mesh", fontsize=9)

    # Panel 6: Seasonal volume bar chart
    ax6 = fig.add_subplot(2, 3, 6)
    if not seasonal_df.empty:
        colors_s = {"dry": "sandybrown", "wet": "steelblue", "shoulder": "mediumseagreen"}
        bar_colors = [colors_s.get(s, "gray") for s in seasonal_df["season"]]
        bars = ax6.bar(range(len(seasonal_df)), seasonal_df["dem_volume_m3"],
                       color=bar_colors, edgecolor="white", linewidth=0.5)
        ax6.axhline(vol_m3, color="black", lw=1.5, linestyle="--",
                    label=f"DEM baseline: {vol_m3:,.0f} m³")
        ax6.set_xticks(range(len(seasonal_df)))
        ax6.set_xticklabels(seasonal_df["date"], rotation=45, ha="right", fontsize=7)
        ax6.set_ylabel("Lake volume [m³]")
        ax6.set_title("Seasonal lake volume\n"
                      "(S2 water mask extent × DEM-estimated depth)\n"
                      "Sandy=dry, Blue=wet, Green=shoulder", fontsize=9)
        ax6.legend(fontsize=8)
        # Volume range
        if len(seasonal_df) > 1:
            vmin, vmax_s = seasonal_df["dem_volume_m3"].min(), seasonal_df["dem_volume_m3"].max()
            ax6.text(0.98, 0.98, f"Seasonal swing:\n{vmax_s-vmin:,.0f} m³\n({100*(vmax_s-vmin)/vol_m3:.1f}%)",
                     transform=ax6.transAxes, ha="right", va="top", fontsize=8,
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    else:
        ax6.text(0.5, 0.5, "No S2 seasonal data\nRun s2_download.py first",
                 transform=ax6.transAxes, ha="center", va="center", fontsize=10)
        ax6.set_title("Seasonal volume", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {OUT_PNG}")
    print(f"\n── Volume Summary ──────────────────────────────────────────")
    print(f"  Method         : voxelization (depth integral over lake cells)")
    print(f"  Lake area      : {area_ha:.2f} ha")
    print(f"  Mean depth     : {mean_depth:.3f} m")
    print(f"  Max depth      : {max_depth:.3f} m")
    print(f"  Total volume   : {vol_m3:,.0f} m³")
    print(f"                 = {vol_m3/1e6:.6f} km³")
    print(f"                 = {vol_m3*264.172:,.0f} US gallons")
    print(f"                 = {vol_m3/1233.48:.1f} acre-feet")
    if not seasonal_df.empty:
        print(f"  Seasonal swing : {seasonal_df['dem_volume_m3'].max()-seasonal_df['dem_volume_m3'].min():,.0f} m³")


if __name__ == "__main__":
    main()
