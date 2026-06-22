"""
Export simulation frame data → viewer/data/

Reads depth_frames_{scenario}.npz produced by:
    python3 simulation/flood_sim.py --scenario all --save-frames

Outputs (one set per scenario):
  viewer/data/simulation_{scenario}_frames.bin        — multi-frame depth binary
  viewer/data/simulation_{scenario}_infiltration.bin  — multi-frame infiltration binary
  viewer/data/simulation_{scenario}_hydrograph.json   — hydrograph time series
  viewer/data/simulation_index.json                   — list of available scenarios

Also exports:
  viewer/data/s2_ground_truth_20240212.png  — OWM water mask for S2 validation

Binary format for simulation_{scenario}_frames.bin / simulation_{scenario}_infiltration.bin
(identical SIML header — frames.bin values are depth [m], infiltration.bin values are
cumulative infiltration [mm]):
  [0:4]                  b'SIML'           magic
  [4:8]                  uint32 n_frames
  [8:12]                 uint32 rows        (256)
  [12:16]                uint32 cols        (256)
  [16 : 16+n*4]          float32[n]        times_min
  [16+n*4 : end]         float32[n*256*256] values, 0 = dry/none

Usage:
    python3 viewer/preprocess/export_simulation.py
"""
import os
import json
import struct
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from PIL import Image

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # viewer/
FLOOD_DIR   = os.path.dirname(BASE_DIR)
SIM_DIR     = os.path.join(FLOOD_DIR, "simulation")
OUT_DIR     = os.path.join(SIM_DIR, "outputs")
DEM_PATH    = os.path.join(FLOOD_DIR, "dem", "data", "winter_garden_dem.tif")
S2_OWM_PATH = os.path.join(FLOOD_DIR, "sentinel2", "data", "omniwatermask_mask_20240212.tif")
DATA_DIR    = os.path.join(BASE_DIR, "data")
TARGET      = 256

# Scenario metadata: (display_label, duration_hr, data_source)
SCENARIO_META = {
    "flash_1hr_100yr":      ("Flash: 1-hr / 100-yr return",    1,  "NOAA Atlas 14"),
    "flash_1hr_10yr":       ("Flash: 1-hr / 10-yr return",     1,  "NOAA Atlas 14"),
    "sustained_12hr_100yr": ("Sustained: 12-hr / 100-yr return", 12, "NOAA Atlas 14"),
    "sustained_12hr_10yr":  ("Sustained: 12-hr / 10-yr return",  12, "NOAA Atlas 14"),
    "historical_20240212":  ("Historical: Feb 7–14, 2024",     168, "NOAA CDO USC00088788"),
    "historical_gsdr":      ("Historical: GSDR extreme (pre-1985)", None, "GSDR US_086638"),
    "historical_gsdr_extreme": ("Historical: biggest GSDR storm (1945-09-16)", 24, "GSDR US_086638"),
}


def _load_dem_bounds():
    """Return (bounds, transform, crs) from the DEM raster."""
    with rasterio.open(DEM_PATH) as src:
        return src.bounds, src.transform, src.crs


def _downsample_frame(arr_868, bounds, src_transform, crs):
    """Downsample a 868×868 float32 depth array to TARGET×TARGET (bilinear)."""
    dst_transform = from_bounds(
        bounds.left, bounds.bottom, bounds.right, bounds.top, TARGET, TARGET
    )
    dst = np.zeros((TARGET, TARGET), dtype=np.float32)
    reproject(
        arr_868, dst,
        src_transform=src_transform, src_crs=crs,
        dst_transform=dst_transform, dst_crs=crs,
        resampling=Resampling.bilinear,
    )
    dst = np.maximum(dst, 0.0)   # clamp negative numerical artefacts to 0
    return dst


def _write_siml_bin(out_path, frames_raw, times_min, bounds, src_transform, crs):
    """Downsample (n, 868, 868) frames to (n, TARGET, TARGET) and write a SIML binary."""
    n_frames = len(times_min)
    frames_small = np.zeros((n_frames, TARGET, TARGET), dtype=np.float32)
    for i, frame in enumerate(frames_raw):
        frames_small[i] = _downsample_frame(frame, bounds, src_transform, crs)

    with open(out_path, "wb") as f:
        f.write(b"SIML")
        f.write(struct.pack("<I", n_frames))
        f.write(struct.pack("<I", TARGET))
        f.write(struct.pack("<I", TARGET))
        f.write(times_min.astype(np.float32).tobytes())
        f.write(frames_small.astype(np.float32).tobytes())
    size_kb = os.path.getsize(out_path) / 1024
    print(f"    → {os.path.basename(out_path)} ({size_kb:.0f} KB)")


def export_scenario_frames(scenario_key, bounds, src_transform, crs):
    """
    Load depth_frames_{scenario}.npz, downsample each frame, write .bin + .json.
    Returns a dict of scenario metadata for simulation_index.json, or None if no frames.
    """
    npz_path = os.path.join(OUT_DIR, f"depth_frames_{scenario_key}.npz")
    if not os.path.exists(npz_path):
        print(f"  ⚠ No frame file for {scenario_key}: {npz_path}")
        print(f"    Run: python3 simulation/flood_sim.py --scenario {scenario_key} --save-frames")
        return None

    hydro_path = os.path.join(OUT_DIR, f"hydrograph_{scenario_key}.csv")
    if not os.path.exists(hydro_path):
        print(f"  ⚠ No hydrograph CSV for {scenario_key}: {hydro_path}")
        return None

    print(f"  {scenario_key}:")

    data = np.load(npz_path)
    frames_raw = data["frames"]        # (n_frames, 868, 868) float32, depth [m]
    times_min  = data["times_min"]     # (n_frames,) float32
    n_frames   = len(times_min)
    print(f"    {n_frames} frames, t = {times_min[0]:.0f}–{times_min[-1]:.0f} min")

    # Downsample + write depth frames
    bin_path = os.path.join(DATA_DIR, f"simulation_{scenario_key}_frames.bin")
    _write_siml_bin(bin_path, frames_raw, times_min, bounds, src_transform, crs)
    print(f"    Downsampled {n_frames} frames: 868×868 → {TARGET}×{TARGET}")

    # Downsample + write cumulative infiltration frames [mm], if present
    if "infiltration_frames" in data:
        infilt_path = os.path.join(DATA_DIR, f"simulation_{scenario_key}_infiltration.bin")
        _write_siml_bin(infilt_path, data["infiltration_frames"], times_min, bounds, src_transform, crs)

    # Load hydrograph and write JSON
    hydro = pd.read_csv(hydro_path)
    label, duration_hr, data_source = SCENARIO_META.get(
        scenario_key, (scenario_key, None, "unknown")
    )

    hydro_json = {
        "times_min":      hydro["time_min"].tolist(),
        "rain_mm_hr":     hydro["rain_mm_hr"].tolist(),
        "Pe_mm_hr":       hydro["Pe_mm_hr"].tolist(),
        "flooded_ha":     hydro["flooded_ha"].tolist(),
        "lake_rise_m":    hydro["lake_rise_m"].tolist(),
        "mean_depth_m":   hydro["mean_depth_m"].tolist(),
        "data_source":    data_source,
        "total_rain_mm":  round(float(hydro["rain_mm_hr"].sum() * (
            (hydro["time_min"].iloc[1] - hydro["time_min"].iloc[0]) / 60
            if len(hydro) > 1 else 1
        )), 1),
        "peak_flooded_ha": round(float(hydro["flooded_ha"].max()), 2),
        "peak_lake_rise_m": round(float(hydro["lake_rise_m"].max()), 3),
    }
    json_path = os.path.join(DATA_DIR, f"simulation_{scenario_key}_hydrograph.json")
    with open(json_path, "w") as f:
        json.dump(hydro_json, f)
    print(f"    → {os.path.basename(json_path)}")

    return {
        "id":              scenario_key,
        "label":           label,
        "duration_hr":     duration_hr,
        "data_source":     data_source,
        "total_rain_mm":   hydro_json["total_rain_mm"],
        "peak_flooded_ha": hydro_json["peak_flooded_ha"],
        "peak_lake_rise_m": hydro_json["peak_lake_rise_m"],
        "n_frames":        n_frames,
    }


def export_s2_ground_truth():
    """
    Reproject OmniWaterMask binary mask (EPSG:32617) for 2024-02-12 into
    DEM grid (EPSG:5070), output as 512×512 RGBA PNG (red=water, transparent=dry).
    """
    out_path = os.path.join(DATA_DIR, "s2_ground_truth_20240212.png")
    if not os.path.exists(S2_OWM_PATH):
        print(f"  ⚠ S2 OWM mask not found: {S2_OWM_PATH}")
        print("    Run: ~/miniforge3/envs/prithvi/bin/python sentinel2/s2_omniwatermask.py")
        return False

    print("  Exporting S2 OWM ground truth (2024-02-12) …")
    with rasterio.open(DEM_PATH) as dem_src:
        bounds      = dem_src.bounds
        dst_crs     = dem_src.crs

    with rasterio.open(S2_OWM_PATH) as src:
        src_mask = src.read(1).astype(np.float32)
        src_transform = src.transform
        src_crs   = src.crs

    SIZE = 512
    dst_transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, SIZE, SIZE)
    dst_mask = np.zeros((SIZE, SIZE), dtype=np.float32)
    reproject(
        src_mask, dst_mask,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=dst_crs,
        resampling=Resampling.nearest,
    )

    # RGBA: water pixels = semi-transparent red, dry = fully transparent
    rgba = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
    water = dst_mask > 0.5
    rgba[water, 0] = 220   # R
    rgba[water, 1] = 60    # G
    rgba[water, 2] = 60    # B
    rgba[water, 3] = 180   # A (semi-transparent)

    img = Image.fromarray(rgba, mode="RGBA")
    img.save(out_path)
    water_px = int(water.sum())
    print(f"    → s2_ground_truth_20240212.png ({water_px:,} water pixels)")
    return True


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    bounds, src_transform, crs = _load_dem_bounds()
    print(f"DEM bounds: {bounds.left:.0f}, {bounds.bottom:.0f}, "
          f"{bounds.right:.0f}, {bounds.top:.0f}  CRS: {crs}")

    # Export each scenario's frames
    print("\nExporting simulation frames …")
    index = []
    for key in SCENARIO_META:
        meta = export_scenario_frames(key, bounds, src_transform, crs)
        if meta:
            index.append(meta)

    # Write simulation_index.json
    index_path = os.path.join(DATA_DIR, "simulation_index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nsimulation_index.json: {len(index)} scenario(s)")
    for s in index:
        print(f"  {s['id']}: {s['n_frames']} frames, "
              f"{s['total_rain_mm']:.0f} mm rain, {s['peak_flooded_ha']:.1f} ha peak")

    if not index:
        print()
        print("No frame files found. Run simulations first:")
        print("  python3 simulation/flood_sim.py --scenario all --save-frames")

    # Export S2 ground truth overlay
    print()
    export_s2_ground_truth()


if __name__ == "__main__":
    main()
