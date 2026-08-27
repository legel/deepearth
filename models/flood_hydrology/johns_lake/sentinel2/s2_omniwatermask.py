"""
OmniWaterMask Inference — Sentinel-2 Water Segmentation
========================================================
Runs OmniWaterMask (DPIRD-DMA, v0.5.0) on all available Sentinel-2 scenes.
Uses RGBNIR (B04/B03/B02/B08) stacked GeoTIFFs as input.

OmniWaterMask uses NDWI and a CNN ensemble model for water body detection.
It augments with OSM water/building/road vectors as soft guidance.

Outputs (per scene date):
    sentinel2/data/omniwatermask_mask_{date}.tif  — binary water mask (1=water)
    sentinel2/data/water_extent_timeseries.csv    — per-scene area stats (--timeseries)

Usage:
    python3 sentinel2/s2_omniwatermask.py                    # all scenes
    python3 sentinel2/s2_omniwatermask.py --from-ranking     # only keep==True scenes
    python3 sentinel2/s2_omniwatermask.py --timeseries       # rebuild timeseries CSV only
"""

import os
import sys
import glob
import shutil
import tempfile
import argparse
import numpy as np
import pandas as pd
import rasterio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

RANKING_CSV   = os.path.join(DATA_DIR, "scene_ranking.csv")
TIMESERIES_CSV = os.path.join(DATA_DIR, "water_extent_timeseries.csv")

# AOI pixel area in m² (S2 10m grid)
S2_PIXEL_AREA_M2 = 10.0 * 10.0


def get_scene_dates(from_ranking=False):
    """Return sorted list of scene dates to process.

    from_ranking=True: only dates with keep==True in scene_ranking.csv,
    ordered by rank (cloud-cleanest first).
    """
    if from_ranking:
        if not os.path.exists(RANKING_CSV):
            print(f"ERROR: {RANKING_CSV} not found — run s2_rank_scenes.py first.")
            sys.exit(1)
        df = pd.read_csv(RANKING_CSV, dtype={"date": str})
        kept = df[df["keep"] == True].sort_values("rank")
        dates = list(kept["date"].astype(str))
        print(f"  --from-ranking: {len(dates)} kept scenes (of {len(df)} total)")
        return dates

    b02_files = glob.glob(os.path.join(DATA_DIR, "s2_*_B02.tif"))
    dates = sorted({
        os.path.basename(f).replace("s2_", "").replace("_B02.tif", "")
        for f in b02_files
    })
    return dates


def build_water_timeseries():
    """Build water_extent_timeseries.csv from existing omniwatermask_mask_*.tif files.

    Columns: date, n_water_polygons, water_area_ha, water_pct_aoi
    Compares against lake_timeseries.csv (MNDWI-based) and prints a summary.
    """
    from rasterio.features import shapes as rio_shapes

    mask_files = sorted(glob.glob(os.path.join(DATA_DIR, "omniwatermask_mask_*.tif")))
    if not mask_files:
        print("No omniwatermask_mask_*.tif files found.")
        return

    total_aoi_px = None
    records = []

    for path in mask_files:
        date = os.path.basename(path).replace("omniwatermask_mask_", "").replace(".tif", "")
        with rasterio.open(path) as src:
            mask = src.read(1)
            transform = src.transform
            if total_aoi_px is None:
                total_aoi_px = mask.size

        pixel_area_m2 = abs(transform.a * transform.e)
        water_px   = int((mask > 0).sum())
        water_ha   = water_px * pixel_area_m2 / 1e4
        water_pct  = water_px / mask.size * 100.0

        # Count connected water polygons
        n_polys = sum(
            1 for _, val in rio_shapes(mask, mask=(mask > 0), connectivity=4)
            if val > 0
        )

        records.append({
            "date":          date,
            "n_water_polygons": n_polys,
            "water_area_ha": round(water_ha,  2),
            "water_pct_aoi": round(water_pct, 4),
        })

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    df.to_csv(TIMESERIES_CSV, index=False)
    print(f"\nSaved water timeseries: {TIMESERIES_CSV}")
    print(df.to_string(index=False))

    # Plausibility comparison against MNDWI timeseries
    mndwi_path = os.path.join(DATA_DIR, "lake_timeseries.csv")
    if os.path.exists(mndwi_path):
        mndwi = pd.read_csv(mndwi_path, dtype={"date": str})
        merged = df.merge(mndwi[["date", "total_lake_area_ha"]], on="date", how="inner")
        if not merged.empty:
            print("\n--- OmniWaterMask vs MNDWI water area (ha) ---")
            print(merged[["date", "water_area_ha", "total_lake_area_ha"]].rename(
                columns={"water_area_ha": "OWM_ha", "total_lake_area_ha": "MNDWI_ha"}
            ).to_string(index=False))
    return df


def stack_rgbnir(date, tmp_dir):
    """Stack B04(R), B03(G), B02(B), B08(NIR) into a 4-band GeoTIFF."""
    band_files = {
        "B04": os.path.join(DATA_DIR, f"s2_{date}_B04.tif"),
        "B03": os.path.join(DATA_DIR, f"s2_{date}_B03.tif"),
        "B02": os.path.join(DATA_DIR, f"s2_{date}_B02.tif"),
        "B08": os.path.join(DATA_DIR, f"s2_{date}_B08.tif"),
    }
    for name, path in band_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {name}: {path}")

    arrays = {}
    with rasterio.open(band_files["B04"]) as ref:
        meta = ref.meta.copy()
        arrays["B04"] = ref.read(1)

    for band in ["B03", "B02", "B08"]:
        with rasterio.open(band_files[band]) as src:
            arrays[band] = src.read(1)

    meta.update(count=4, dtype="uint16")
    out_path = os.path.join(tmp_dir, f"s2_{date}_RGBNIR.tif")
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(arrays["B04"], 1)  # Red
        dst.write(arrays["B03"], 2)  # Green
        dst.write(arrays["B02"], 3)  # Blue
        dst.write(arrays["B08"], 4)  # NIR
    return out_path


def run_omniwatermask(date, stacked_path, tmp_dir):
    """Run OmniWaterMask on a stacked RGBNIR scene. Returns output path."""
    from omniwatermask import make_water_mask

    owm_out_dir = os.path.join(tmp_dir, "owm_output")
    os.makedirs(owm_out_dir, exist_ok=True)

    import pathlib
    # band_order=[1,2,3,4] = 1-indexed [R, G, B, NIR] positions in the stacked file
    result_paths = make_water_mask(
        scene_paths=[stacked_path],
        band_order=[1, 2, 3, 4],
        output_dir=pathlib.Path(owm_out_dir),
        overwrite=True,
        inference_device="cpu",
        inference_patch_size=512,
        inference_overlap_size=128,
    )
    if not result_paths:
        raise RuntimeError(f"OmniWaterMask returned no output for {date}")
    return result_paths[0]


def main(from_ranking=False, timeseries_only=False):
    if timeseries_only:
        build_water_timeseries()
        return

    dates = get_scene_dates(from_ranking=from_ranking)
    print(f"Found {len(dates)} scene dates: {dates}")

    tmp_root = tempfile.mkdtemp(prefix="owm_")
    try:
        for date in dates:
            out_mask = os.path.join(DATA_DIR, f"omniwatermask_mask_{date}.tif")
            if os.path.exists(out_mask):
                print(f"  {date}: already done, skipping")
                continue

            print(f"  {date}: stacking RGBNIR …", flush=True)
            try:
                stacked = stack_rgbnir(date, tmp_root)
            except FileNotFoundError as e:
                print(f"    SKIP: {e}")
                continue

            print(f"  {date}: running OmniWaterMask …", flush=True)
            try:
                owm_result = run_omniwatermask(date, stacked, tmp_root)
            except Exception as e:
                print(f"    ERROR: {e}")
                continue

            # Read result, ensure binary uint8, save to data dir
            with rasterio.open(owm_result) as src:
                mask = src.read(1)
                meta = src.meta.copy()

            meta.update(dtype="uint8", count=1)
            binary = (mask > 0).astype(np.uint8)
            with rasterio.open(out_mask, "w", **meta) as dst:
                dst.write(binary, 1)

            area_ha = float(binary.sum() * abs(meta.get("transform").a) ** 2 / 1e4)
            print(f"    → {out_mask}  ({area_ha:.1f} ha water)")

            # Clean up large temp stacked file to save disk space
            os.remove(stacked)

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print("\nBuilding water extent timeseries …")
    build_water_timeseries()
    print("\nDone. Run sentinel2/compare_methods.py to update comparison metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmniWaterMask inference for Sentinel-2 scenes")
    parser.add_argument(
        "--from-ranking", action="store_true",
        help="Process only keep==True scenes from scene_ranking.csv (ranked cleanest first)"
    )
    parser.add_argument(
        "--timeseries", action="store_true",
        help="Rebuild water_extent_timeseries.csv from existing masks only (no inference)"
    )
    args = parser.parse_args()
    main(from_ranking=args.from_ranking, timeseries_only=args.timeseries)
