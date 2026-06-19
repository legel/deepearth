"""
Drought drawdown — longest-observed dry spell vs. the lake's VAE curve.

Companion to the wet extremes in `simulation/flood_sim.py` (see scenario
`historical_gsdr_extreme`): "show the 2 extents" — flood high-water vs.
drought low-water. A 2D inundation solver is the wrong tool for a multi-week
drawdown with ~zero rainfall; instead this steps the lake's water-surface
elevation down day by day using the same hypsometric (volume-area-elevation)
relationship as `lake_volume.py`'s Panel 3, driven by an evaporative loss
each day that shrinks the lake's volume, then looks up the new level/area
from the curve. As the lake shrinks, the same evaporative depth removes a
larger share of volume (smaller surface area), so the rate of level decline
accelerates over the dry spell — same physical behavior as undammed
shallow lakes during real droughts.

Duration and rate are both sourced from data, not guessed:

  Duration : 82 days — the longest dry spell (<1 mm/day) in the full GSDR
             hourly record across both nearby stations (US_086638
             1942-1985, US_086628 1974-2011), found 1974-01-13 -> 1974-04-04
             at US_086638. See precipitation/fetch_gsdr_extreme.py for the
             full station survey.
  Rate     : 1.0 -> 5.0 mm/day evaporation, linearly ramped across the
             82-day window. Sourced from USGS energy-budget evaporation at
             Lake Lucerne, central FL (Swancar & Hill, USGS OFR 2015-1075),
             which measured 0.04 in/day (~1.0 mm/day) in early January
             ramping to 0.26 in/day (~6.6 mm/day) by early May — i.e. the
             same Jan-Apr dry-season window as the GSDR drought, low rates
             in the cool early months rising as the dry season warms toward
             the wet-season onset. No rainfall or net inflow is assumed
             during the drought window (the GSDR record confirms <1 mm/day
             throughout), so evaporation is the sole loss term.

Outputs
-------
dem/data/lake_drought.csv
    columns: time_min, day, lake_level_m, area_ha, volume_m3, evap_mm_day
dem/lake_drought_comparison.png
    side-by-side wet-extreme flood extent vs. drought drawdown extent

Usage
-----
    python3 dem/lake_drought.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
import rasterio.features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
S2_DIR   = os.path.join(BASE_DIR, "..", "sentinel2", "data")
SIM_DIR  = os.path.join(BASE_DIR, "..", "simulation", "outputs")
OUT_CSV  = os.path.join(DATA_DIR, "lake_drought.csv")
OUT_PNG  = os.path.join(BASE_DIR, "lake_drought_comparison.png")

DROUGHT_DAYS  = 82            # longest GSDR dry spell, US_086638 1974-01-13 -> 1974-04-04
EVAP_MM_START = 1.0           # Lake Lucerne energy-budget evap, early Jan (USGS OFR 2015-1075)
EVAP_MM_END   = 5.0           # ramped toward early-Apr / pre-wet-season rate
VAE_Z_BELOW   = 2.5           # m below WSE0 to extend the hypsometric curve (drawdown headroom)
VAE_N_STEPS   = 2000          # resolution of the VAE lookup table

EXTREME_WET_SCENARIO = "historical_gsdr_extreme"


def load_tif(fname, data_dir=DATA_DIR):
    path = os.path.join(data_dir, fname)
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nd  = src.nodata
        transform = src.transform
        crs = src.crs
    if nd is not None:
        arr[arr == nd] = np.nan
    return arr, transform, crs, abs(transform.a)


def build_vae_curve(lake_bool, lake_bed, wse0, cell_m, z_below=VAE_Z_BELOW, n_steps=VAE_N_STEPS):
    """Hypsometric (volume-area-elevation) curve, same method as lake_volume.py
    Panel 3: for each candidate water level z, submerged area = lake cells with
    bed below z; volume = sum(max(z - bed, 0)) * cell_area. Monotonic in z
    over the range of interest, used here as a lookup table for drawdown."""
    z_steps = np.linspace(wse0 - z_below, wse0 + 0.1, n_steps)
    areas_ha = np.empty(n_steps)
    vols_m3  = np.empty(n_steps)
    for i, z in enumerate(z_steps):
        submerged = lake_bool & np.isfinite(lake_bed) & (lake_bed < z)
        depth_at_z = np.where(submerged, z - lake_bed, 0.0)
        vols_m3[i]  = float(depth_at_z.sum() * cell_m ** 2)
        areas_ha[i] = float(submerged.sum() * cell_m ** 2 / 1e4)
    return z_steps, areas_ha, vols_m3


def main():
    print("Loading rasters …")
    dem, dem_t, dem_crs, cell_m = load_tif("winter_garden_dem.tif")

    sys.path.insert(0, BASE_DIR)
    from lake_utils import get_lake_mask_and_fwc
    lake_bool, lake_bed, _wse_raw, mask_label = get_lake_mask_and_fwc(
        dem, dem_t, dem_crs, cell_m, DATA_DIR, s2_data_dir=S2_DIR)
    print(f"  Lake mask: {mask_label}")

    vol_csv = pd.read_csv(os.path.join(DATA_DIR, "lake_volume.csv")).iloc[0]
    wse0     = float(vol_csv["water_surface_m"])
    area0_ha = float(vol_csv["area_ha"])
    vol0_m3  = float(vol_csv["volume_m3"])
    print(f"  Authoritative WSE0 = {wse0:.2f} m, area0 = {area0_ha:.2f} ha, "
          f"volume0 = {vol0_m3:,.0f} m³ (from lake_volume.csv)")

    print("\nBuilding VAE (volume-area-elevation) curve …")
    z_steps, areas_ha, vols_m3 = build_vae_curve(lake_bool, lake_bed, wse0, cell_m)
    print(f"  {len(z_steps)} levels from {z_steps[0]:.3f} to {z_steps[-1]:.3f} m")

    # Seed the iteration from the *table's own* volume at wse0, not lake_volume.csv's
    # rounded value — they come from slightly different valid-cell masks (lake_volume.py
    # uses mean-DEM water surface over all lake cells; this table uses the FWC-bed
    # submergence test), so mixing the two would occasionally nudge the inverse VAE
    # lookup the wrong way on the very first (tiny) step and break monotonic decline.
    table_vol0 = float(np.interp(wse0, z_steps, vols_m3))
    print(f"  Table-consistent volume0 = {table_vol0:,.0f} m³ "
          f"({100 * (table_vol0 - vol0_m3) / vol0_m3:+.2f}% vs. lake_volume.csv — "
          f"different valid-cell mask, used only internally for the drawdown integral)")

    print(f"\nStepping {DROUGHT_DAYS}-day drought drawdown "
          f"(evap ramp {EVAP_MM_START:.1f}->{EVAP_MM_END:.1f} mm/day) …")
    evap_schedule = np.linspace(EVAP_MM_START, EVAP_MM_END, DROUGHT_DAYS)

    rows = []
    level = wse0
    volume = table_vol0
    rows.append({"time_min": 0, "day": 0, "lake_level_m": round(level, 4),
                  "area_ha": round(float(np.interp(level, z_steps, areas_ha)), 2),
                  "volume_m3": round(volume, 0), "evap_mm_day": 0.0})
    for day in range(1, DROUGHT_DAYS + 1):
        evap_mm = evap_schedule[day - 1]
        area_ha = float(np.interp(level, z_steps, areas_ha))
        volume_loss_m3 = area_ha * 1e4 * (evap_mm / 1000.0)
        volume = max(volume - volume_loss_m3, 0.0)
        level = float(np.interp(volume, vols_m3, z_steps))
        area_ha = float(np.interp(level, z_steps, areas_ha))
        rows.append({"time_min": day * 1440, "day": day, "lake_level_m": round(level, 4),
                      "area_ha": round(area_ha, 2), "volume_m3": round(volume, 0),
                      "evap_mm_day": round(float(evap_mm), 2)})

    drought_df = pd.DataFrame(rows)
    drought_df.to_csv(OUT_CSV, index=False)
    final = drought_df.iloc[-1]
    print(f"\n  Day 0   : level={wse0:.3f} m, area={area0_ha:.2f} ha, volume={table_vol0:,.0f} m³")
    print(f"  Day {DROUGHT_DAYS:<3} : level={final.lake_level_m:.3f} m, "
          f"area={final.area_ha:.2f} ha, volume={final.volume_m3:,.0f} m³")
    print(f"  Level drop : {wse0 - final.lake_level_m:.3f} m")
    print(f"  Area loss  : {area0_ha - final.area_ha:.2f} ha "
          f"({100 * (area0_ha - final.area_ha) / area0_ha:.1f}%)")
    print(f"  Volume loss: {table_vol0 - final.volume_m3:,.0f} m³ "
          f"({100 * (table_vol0 - final.volume_m3) / table_vol0:.1f}%)")
    print(f"\nSaved drought CSV -> {OUT_CSV}")

    # ── Side-by-side wet-extreme vs. drought comparison ──────────────────
    print("\nBuilding wet-vs-drought comparison figure …")
    drought_level = float(final.lake_level_m)
    drought_extent = lake_bool & np.isfinite(lake_bed) & (lake_bed < drought_level)

    wet_geojson_path = os.path.join(SIM_DIR, f"flood_extent_{EXTREME_WET_SCENARIO}.geojson")
    wet_extent = None
    wet_peak_ha = None
    if os.path.exists(wet_geojson_path):
        with open(wet_geojson_path) as f:
            wet_gj = json.load(f)
        shapes = [(feat["geometry"], 1) for feat in wet_gj["features"]]
        if shapes:
            wet_extent = rasterio.features.rasterize(
                shapes, out_shape=dem.shape, transform=dem_t, fill=0, dtype=np.uint8
            ).astype(bool)
            wet_peak_ha = float(wet_extent.sum() * cell_m ** 2 / 1e4)
    else:
        print(f"  NOTE: {wet_geojson_path} not found yet — run the "
              f"{EXTREME_WET_SCENARIO} scenario first for the wet-extent overlay.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    ax0 = axes[0]
    ax0.imshow(np.where(lake_bool, 0.3, np.nan), cmap="Greys", origin="upper", vmin=0, vmax=1)
    ax0.imshow(np.where(lake_bool, 1.0, np.nan), cmap="Blues", origin="upper",
               alpha=0.6, vmin=0, vmax=1)
    title0 = "Baseline lake extent (WSE = {:.2f} m)\n{:.2f} ha".format(wse0, area0_ha)
    if wet_extent is not None:
        ax0.imshow(np.where(wet_extent, 1.0, np.nan), cmap="Reds", origin="upper",
                   alpha=0.55, vmin=0, vmax=1)
        title0 += (f"\n+ {EXTREME_WET_SCENARIO} peak flood extent: "
                   f"{wet_peak_ha:.1f} ha (red)")
    ax0.set_title(title0, fontsize=10)
    ax0.set_xlabel("col"); ax0.set_ylabel("row")

    ax1 = axes[1]
    ax1.imshow(np.where(lake_bool, 0.3, np.nan), cmap="Greys", origin="upper", vmin=0, vmax=1)
    ax1.imshow(np.where(drought_extent, 1.0, np.nan), cmap="YlOrBr", origin="upper",
               alpha=0.7, vmin=0, vmax=1)
    ax1.set_title(
        f"Drought drawdown after {DROUGHT_DAYS} days\n"
        f"(longest GSDR dry spell, 1974-01-13 -> 1974-04-04)\n"
        f"Level {drought_level:.2f} m ({wse0 - drought_level:.2f} m drop) | "
        f"{final.area_ha:.2f} ha ({100*(area0_ha-final.area_ha)/area0_ha:.1f}% area loss)",
        fontsize=10)
    ax1.set_xlabel("col"); ax1.set_ylabel("row")

    fig.suptitle("Johns Lake extremes — wettest vs. driest observed conditions\n"
                  "Wet: GSDR all-time max storm (extracted from 1942-2011 gauge record) | "
                  "Dry: GSDR longest dry spell + USGS central-FL evaporation",
                  fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison figure -> {OUT_PNG}")


if __name__ == "__main__":
    main()
