"""Static terrain at each tower from the Copernicus DEM GLO-30 (30 m), AWS open data (s3://copernicus-dem-30m).

Per tower: elevation at the tower (bilinear, m above EGM2008), slope and aspect from a 3 x 3 window at 30 m, and
the mean and standard deviation of elevation within 1 km, within the ERA5-Land 0.1 deg cell and within the
ERA5 0.25 deg cell that hold the tower (the latter two from overview levels, about 120 m and 240 m). The
difference between the tower and the cell mean is the height that a lapse-rate adjustment of the reanalysis
temperature and pressure has to bridge.

Licence: Copernicus DEM, (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018 provided under
COPERNICUS by the European Union and ESA; free of charge, worldwide, for any purpose (GLO-30 licence).

    python dem.py [--sites N]
"""

import argparse
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from sites import load_sites

ROOT = Path("/home/photon/4tb/fluxfill/drivers")
URL = "https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif"


def tile_name(lat: float, lon: float) -> str:
    la, lo = math.floor(lat), math.floor(lon)
    ns = f"N{la:02d}" if la >= 0 else f"S{-la:02d}"
    ew = f"E{lo:03d}" if lo >= 0 else f"W{-lo:03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"


def _env():
    import os
    import sys
    # This env's own PROJ and GDAL data (an inherited PROJ_DATA may point at another env's).
    os.environ["PROJ_DATA"] = os.environ["PROJ_LIB"] = str(Path(sys.executable).parents[1] / "share" / "proj")
    os.environ["GDAL_DATA"] = str(Path(sys.executable).parents[1] / "share" / "gdal")
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
    os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")


def _box_stats(lat, lon, half_lat, half_lon, res_deg):
    """Mean and std of elevation in a lat-lon box, read from whichever tiles it covers at about res_deg."""
    import rasterio
    from rasterio.windows import from_bounds
    vals = []
    for la in range(math.floor(lat - half_lat), math.floor(lat + half_lat) + 1):
        for lo in range(math.floor(lon - half_lon), math.floor(lon + half_lon) + 1):
            n = tile_name(la + 0.5, lo + 0.5)
            try:
                with rasterio.open("/vsicurl/" + URL.format(n=n)) as r:
                    b = (max(lon - half_lon, r.bounds.left), max(lat - half_lat, r.bounds.bottom),
                         min(lon + half_lon, r.bounds.right), min(lat + half_lat, r.bounds.top))
                    if b[2] <= b[0] or b[3] <= b[1]:
                        continue
                    w = from_bounds(*b, transform=r.transform)
                    shape = (max(1, int(round((b[3] - b[1]) / res_deg))), max(1, int(round((b[2] - b[0]) / res_deg))))
                    a = r.read(1, window=w, out_shape=shape, resampling=rasterio.enums.Resampling.average,
                               masked=True)
                    vals.append(a.compressed().astype(float))
            except rasterio.errors.RasterioIOError:
                continue  # no tile: open ocean
    if not vals:
        return np.nan, np.nan
    v = np.concatenate(vals)
    return (float(v.mean()), float(v.std())) if v.size else (np.nan, np.nan)


def site_terrain(lat: float, lon: float) -> dict:
    import rasterio
    out = {"dem_tile": tile_name(lat, lon)}
    try:
        with rasterio.open("/vsicurl/" + URL.format(n=out["dem_tile"])) as r:
            row, col = r.index(lon, lat, op=lambda x: x)  # fractional
            r0, c0 = int(math.floor(row - 0.5)), int(math.floor(col - 0.5))
            win = rasterio.windows.Window(c0 - 1, r0 - 1, 4, 4)
            a = r.read(1, window=win, boundless=True, fill_value=np.nan).astype(float)
            fy, fx = row - 0.5 - r0, col - 0.5 - c0
            q = a[1:3, 1:3]
            out["dem_elevation_m"] = float((q[0, 0] * (1 - fy) * (1 - fx) + q[0, 1] * (1 - fy) * fx
                                            + q[1, 0] * fy * (1 - fx) + q[1, 1] * fy * fx))
            dy = abs(r.transform.e) * 111320.0
            dx = abs(r.transform.a) * 111320.0 * math.cos(math.radians(lat))
            z = a[0:3, 0:3] if fy < 0.5 and fx < 0.5 else a[1:4, 1:4]
            gx = ((z[0, 2] + 2 * z[1, 2] + z[2, 2]) - (z[0, 0] + 2 * z[1, 0] + z[2, 0])) / (8 * dx)
            gy = ((z[2, 0] + 2 * z[2, 1] + z[2, 2]) - (z[0, 0] + 2 * z[0, 1] + z[0, 2])) / (8 * dy)
            out["dem_slope_deg"] = float(math.degrees(math.atan(math.hypot(gx, gy))))
            # Aspect: direction the slope faces, degrees clockwise from north (rows increase southward).
            out["dem_aspect_deg"] = float((math.degrees(math.atan2(-gx, gy)) + 360) % 360) if (gx or gy) else np.nan
    except rasterio.errors.RasterioIOError:
        out.update(dem_elevation_m=np.nan, dem_slope_deg=np.nan, dem_aspect_deg=np.nan)
    km_lat = 1.0 / 111.32
    km_lon = km_lat / max(math.cos(math.radians(lat)), 0.05)
    out["dem_mean_1km_m"], out["dem_std_1km_m"] = _box_stats(lat, lon, km_lat, km_lon, 1 / 3600)
    for name, d in (("era5_land", 0.1), ("era5", 0.25)):
        clat, clon = round(lat / d) * d, round(lon / d) * d
        out[f"dem_mean_{name}_cell_m"], out[f"dem_std_{name}_cell_m"] = _box_stats(
            clat, clon, d / 2, d / 2, d / 100)
    return out


def run(n_sites=None, workers=6, root: Path = ROOT):
    _env()
    sites = load_sites()
    if n_sites:
        sites = sites.iloc[:n_sites]
    out = root / "dem"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "terrain.csv"
    done = pd.read_csv(dst) if dst.exists() else pd.DataFrame(columns=["site_id"])
    todo = sites[~sites.site_id.isin(done.site_id)]

    def one(s):
        for k in range(4):
            try:
                return {"site_id": s.site_id, "lat_used": s.lat, "lon_used": s.lon, "elevation_m": s.elevation_m,
                        **site_terrain(s.lat, s.lon)}
            except Exception as e:
                err = e
                time.sleep(2 ** k)
        return {"site_id": s.site_id, "lat_used": s.lat, "lon_used": s.lon, "error": repr(err)}

    rows = []
    t = time.time()
    with ThreadPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(one, list(todo.itertuples()))):
            rows.append(r)
            if (k + 1) % 25 == 0:
                pd.concat([done, pd.DataFrame(rows)]).to_csv(dst, index=False)
                print(f"{k + 1}/{len(todo)} {time.time() - t:.0f} s", flush=True)
    pd.concat([done, pd.DataFrame(rows)]).to_csv(dst, index=False)
    print("done", dst, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    run(a.sites, a.workers)


if __name__ == "__main__":
    main()
