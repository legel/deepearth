"""
Fetch NOAA MRMS MultiSensor QPE (radar+gauge multi-sensor QPE, ~1km) hourly rainfall for the
Gee Creek watershed footprint, for the Ian and Milton storm windows this project's own solver
already runs -- direct test of whether the single point gauge this project has always used
(ASOS KSFB, 10.8km from site3, chosen on proximity only -- see the dataset-selection-logic audit
in ../CLAUDE.md) over- or under-represents the real watershed-mean rainfall. This is the "MRMS
gridded rainfall" step logged in ../../NEXT_STEPS.md (added 2026-09-02 after reviewing Inunda,
arXiv:2607.09614) as this project's leading untested lever for its own ~2x magnitude gap --
never attempted anywhere in this project before this script.

Source: MRMS MultiSensor_QPE_01H_Pass2, archived by the Iowa Environmental Mesonet at
https://mtarchive.geol.iastate.edu/{yyyy}/{mm}/{dd}/mrms/ncep/MultiSensor_QPE_01H_Pass2/ .
NOAA's own near-real-time bucket (noaa-mrms-pds on S3) was checked first and confirmed to only
retain ~30 days (a live bucket listing for this exact product prefix returned zero keys) --
IEM's archive is the only source with 2022/2024 history. Requires cfgrib+eccodes to decode
GRIB2 (installed this session via `pip3 install --user eccodes cfgrib`, not previously a
dependency of this project).

Storm windows chosen to reproduce this project's own already-reported storm totals exactly
(see ../../NEXT_STEPS.md's "storm total" column), so the MRMS-vs-gauge ratio below is a
like-for-like comparison over the identical window the solver actually ran:
  - Ian:    2022-09-28 00:00 - 2022-09-30 23:00 UTC (KSFB = 391.7mm over this window, matches
            the already-reported 392mm)
  - Milton: 2024-10-06 00:00 - 2024-10-10 23:00 UTC (KSFB = 288.0mm, matches exactly)

Usage:
    python3 site3_gee_creek/precipitation/fetch_mrms_site3.py --storm ian
    python3 site3_gee_creek/precipitation/fetch_mrms_site3.py --storm milton
    python3 site3_gee_creek/precipitation/fetch_mrms_site3.py --storm both   # default
"""
import os
import gzip
import shutil
import argparse
import json

import numpy as np
import pandas as pd
import requests
import xarray as xr
import geopandas as gpd
import rioxarray  # noqa: F401  (registers the .rio accessor on DataArray)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SITE3_DIR = os.path.dirname(BASE_DIR)
RAW_DIR = os.path.join(BASE_DIR, "data", "mrms_raw")
OUT_DIR = os.path.join(BASE_DIR, "data")
WATERSHED_GEOJSON = os.path.join(SITE3_DIR, "dem", "data", "hydro", "watershed.geojson")

MRMS_PRODUCT = "MultiSensor_QPE_01H_Pass2"
ARCHIVE_BASE = "https://mtarchive.geol.iastate.edu"
BBOX_MARGIN_DEG = 0.03  # generous margin around the watershed bbox before exact polygon clip

STORMS = {
    "ian": {
        "start": "2022-09-28 00:00", "end": "2022-09-30 23:00",
        "gauge_total_mm": 391.74, "gauge_csv": "asos_hourly_SFB.csv",
    },
    "milton": {
        "start": "2024-10-06 00:00", "end": "2024-10-10 23:00",
        "gauge_total_mm": 288.04, "gauge_csv": "asos_hourly_SFB_milton.csv",
    },
}


def mrms_url(dt):
    d = dt.strftime("%Y/%m/%d")
    fname = f"{MRMS_PRODUCT}_00.00_{dt.strftime('%Y%m%d-%H%M%S')}.grib2.gz"
    return f"{ARCHIVE_BASE}/{d}/mrms/ncep/{MRMS_PRODUCT}/{fname}", fname


def fetch_hour(dt, cache_dir):
    """Download+decompress one hourly GRIB2, cached on disk (re-runs skip already-fetched hours)."""
    url, fname = mrms_url(dt)
    gz_path = os.path.join(cache_dir, fname)
    grib_path = gz_path[:-3]
    if os.path.exists(grib_path):
        return grib_path
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        print(f"  MISSING: {fname} (HTTP {r.status_code})")
        return None
    with open(gz_path, "wb") as f:
        f.write(r.content)
    with gzip.open(gz_path, "rb") as fin, open(grib_path, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    os.remove(gz_path)
    return grib_path


def watershed_mean_mm(grib_path, watershed_4326, bounds_360):
    """Exact area-weighted mean of one hourly MRMS grid over the real delineated watershed
    polygon (not just its bounding box) -- crops to a small window first (cheap), then clips
    to the true polygon (also cheap, since the window is already down to ~10x10 cells)."""
    lon_min, lat_min, lon_max, lat_max = bounds_360
    ds = xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    da = ds["unknown"]
    da = da.where(da >= 0)  # MRMS encodes missing/no-coverage as negative sentinels
    # MRMS grid is 0-360 longitude, descending latitude -- slice with matching order first
    sub = da.sel(
        latitude=slice(lat_max + BBOX_MARGIN_DEG, lat_min - BBOX_MARGIN_DEG),
        longitude=slice(lon_min - BBOX_MARGIN_DEG, lon_max + BBOX_MARGIN_DEG),
    )
    if sub.size == 0:
        return np.nan
    sub = sub.assign_coords(longitude=sub.longitude - 360)  # back to -180..180 for this window
    sub = sub.rio.write_crs("EPSG:4326")
    try:
        clipped = sub.rio.clip(watershed_4326.geometry, watershed_4326.crs,
                                drop=True, all_touched=True)
    except Exception:
        # entire watershed footprint is smaller than one MRMS cell somewhere -- fall back
        # to the bbox-window mean rather than dropping the hour
        clipped = sub
    vals = clipped.values
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    return float(np.mean(vals))


def run_storm(storm):
    cfg = STORMS[storm]
    cache_dir = os.path.join(RAW_DIR, storm)
    os.makedirs(cache_dir, exist_ok=True)

    watershed = gpd.read_file(WATERSHED_GEOJSON)
    if watershed.crs is None:
        watershed = watershed.set_crs("EPSG:4326")
    else:
        watershed = watershed.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = watershed.total_bounds
    bounds_360 = (minx + 360, miny, maxx + 360, maxy)

    hours = pd.date_range(cfg["start"], cfg["end"], freq="1h", tz="UTC")
    print(f"\n=== {storm.upper()}: {len(hours)} hourly MRMS grids, "
          f"watershed bbox {minx:.4f},{miny:.4f} - {maxx:.4f},{maxy:.4f} ===")

    rows = []
    for dt in hours:
        grib_path = fetch_hour(dt.tz_localize(None), cache_dir)
        if grib_path is None:
            rows.append({"datetime_utc": dt, "mrms_mm": np.nan})
            continue
        mm = watershed_mean_mm(grib_path, watershed, bounds_360)
        rows.append({"datetime_utc": dt, "mrms_mm": mm})
        print(f"  {dt}  MRMS watershed-mean = {mm:.2f} mm")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, f"mrms_hourly_{storm}.csv")
    df.to_csv(out_csv, index=False)

    total_mrms = float(df["mrms_mm"].sum(skipna=True))
    n_missing = int(df["mrms_mm"].isna().sum())
    gauge_total = cfg["gauge_total_mm"]
    ratio = total_mrms / gauge_total if gauge_total else float("nan")

    summary = {
        "storm": storm,
        "product": MRMS_PRODUCT,
        "window_start_utc": cfg["start"], "window_end_utc": cfg["end"],
        "n_hours": len(df), "n_missing_hours": n_missing,
        "mrms_watershed_total_mm": total_mrms,
        "point_gauge_ksfb_total_mm": gauge_total,
        "ratio_mrms_over_gauge": ratio,
    }
    out_json = os.path.join(OUT_DIR, f"mrms_summary_{storm}.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{storm.upper()}: MRMS watershed-mean total = {total_mrms:.1f} mm "
          f"({n_missing} missing hrs of {len(df)}) vs point-gauge KSFB = {gauge_total:.1f} mm "
          f"-> ratio {ratio:.3f}")
    print(f"  Saved {out_csv}\n  Saved {out_json}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--storm", choices=["ian", "milton", "both"], default="both")
    args = ap.parse_args()
    storms = ["ian", "milton"] if args.storm == "both" else [args.storm]
    results = [run_storm(s) for s in storms]
    if len(results) == 2:
        print("\n=== Summary ===")
        for r in results:
            print(f"  {r['storm']:>8}: MRMS {r['mrms_watershed_total_mm']:.1f}mm vs "
                  f"gauge {r['point_gauge_ksfb_total_mm']:.1f}mm "
                  f"(ratio {r['ratio_mrms_over_gauge']:.3f})")
