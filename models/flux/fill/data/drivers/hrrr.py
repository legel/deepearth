"""HRRR 3 km hourly analysis at the towers (Dowell et al. 2022), from the dynamical.org Icechunk archive.

Source: dynamical.org "NOAA HRRR analysis" v0.2.0 (s3://dynamical-noaa-hrrr/noaa-hrrr-analysis/v0.2.0.icechunk,
anonymous, CC-BY-4.0), built from the NOAA HRRR archive on AWS (s3://noaa-hrrr-bdp-pds). Each hour is the analysis
(forecast step 0) of that hour's HRRR run; precipitation is the step-1 rate of the previous hour's run, so it covers
(t - 1 h, t]. Radiation is the instantaneous flux at t. Coverage: CONUS, Lambert conformal grid, 3 km, from
2014-10-01; many hours are missing before August 2018 (HRRR v1 and v2) and downward longwave and 2 m relative
humidity start in August 2016.

Data are stored in chunks of 2160 h x 45 x 45 cells (90 days x 135 km), so a tower costs one chunk per variable per
90 days (about 5 MB) whatever the number of hours used; only the 2 x 2 cells around the tower are kept.

The full GRIB2 files are never downloaded. The equivalent GRIB route (Herbie byte ranges from the .idx files)
costs about 1.5 MB per variable per hour for the whole CONUS grid, cheaper only when hundreds of towers are
extracted at once.

    python hrrr.py --sites US-A32 US-Bi1 ... --start 2019-01-01 --end 2021-01-01
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import physics
from grid import combine, haversine_m
from sites import load_sites
from assemble import write_table

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path("/home/photon/4tb/fluxfill/drivers")
STORE = dict(bucket="dynamical-noaa-hrrr", prefix="noaa-hrrr-analysis/v0.2.0.icechunk", region="us-west-2")
DX = 3000.0

VARS = {
    "temperature_2m": "deg C", "dew_point_temperature_2m": "deg C",
    "downward_short_wave_radiation_flux_surface": "W m-2 (instant)",
    "downward_long_wave_radiation_flux_surface": "W m-2 (instant)",
    "precipitation_surface": "kg m-2 s-1 (mean over preceding hour)",
    "wind_u_10m": "m s-1 (grid x)", "wind_v_10m": "m s-1 (grid y)",
    "pressure_surface": "Pa", "snow_thickness_surface": "m",
}

MAPPING = {
    "TA": ("deg C", "temperature_2m", "instant"),
    "VPD": ("hPa", "es(temperature_2m) - es(dew_point_temperature_2m)", "instant"),
    "SW_IN": ("W m-2", "downward_short_wave_radiation_flux_surface", "instant"),
    "LW_IN": ("W m-2", "downward_long_wave_radiation_flux_surface (from 2016-08)", "instant"),
    "P": ("mm", "3600 x precipitation_surface", "sum of the preceding hour"),
    "WS": ("m s-1", "hypot(wind_u_10m, wind_v_10m) (rotation-invariant)", "instant"),
    "PA": ("kPa", "pressure_surface / 1000", "instant"),
    "D_SNOW": ("cm", "100 x snow_thickness_surface", "instant"),
}


def open_store():
    import sys
    # This env's own PROJ database (an inherited PROJ_DATA may point at another env's, of another PROJ version).
    os.environ["PROJ_DATA"] = os.environ["PROJ_LIB"] = str(Path(sys.executable).parents[1] / "share" / "proj")
    import icechunk
    import xarray as xr
    st = icechunk.s3_storage(anonymous=True, **STORE)
    repo = icechunk.Repository.open(st)
    return xr.open_zarr(repo.readonly_session("main").store, chunks=None, consolidated=False)


def transformer(ds):
    import pyproj
    crs = pyproj.CRS.from_wkt(ds.spatial_ref.attrs["crs_wkt"])
    return pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)


def lcc_corners(x, y, xs: np.ndarray, ys: np.ndarray):
    """2 x 2 block around projected (x, y) on axes xs (ascending) and ys (descending).
    Returns (iy0, ix0, wy, wx) with rows iy0, iy0 + 1 and columns ix0, ix0 + 1; None outside the grid."""
    fx = (x - xs[0]) / (xs[1] - xs[0])
    fy = (y - ys[0]) / (ys[1] - ys[0])
    ix0, iy0 = int(np.floor(fx)), int(np.floor(fy))
    if ix0 < 0 or iy0 < 0 or ix0 + 1 >= len(xs) or iy0 + 1 >= len(ys):
        return None
    return iy0, ix0, float(fy - iy0), float(fx - ix0)


def in_domain(sites: pd.DataFrame, ds=None) -> pd.Series:
    ds = ds if ds is not None else open_store()
    tr = transformer(ds)
    xs, ys = ds.x.values, ds.y.values
    x, y = tr.transform(sites.lon.values, sites.lat.values)
    return pd.Series([lcc_corners(a, b, xs, ys) is not None for a, b in zip(x, y)], index=sites.index)


def extract_site(ds, tr, site, start, end):
    xs, ys = ds.x.values, ds.y.values
    x, y = tr.transform(site.lon, site.lat)
    c = lcc_corners(x, y, xs, ys)
    if c is None:
        return None, None
    iy0, ix0, wy, wx = c
    tsel = slice(np.datetime64(start), np.datetime64(end) - np.timedelta64(1, "h"))
    raw = {}
    for v in VARS:
        for k in range(6):
            try:
                a = ds[v].sel(time=tsel).isel(y=slice(iy0, iy0 + 2), x=slice(ix0, ix0 + 2)).load()
                break
            except Exception:
                if k == 5:
                    raise
                time.sleep(2 ** k)
        block = a.values.reshape(a.shape[0], 4).T  # corners (iy0, ix0), (iy0, ix0+1), (iy0+1, ix0), (iy0+1, ix0+1)
        raw[v] = combine(block, wy, wx, valid_corners=np.ones(4, bool))[:2]
        times = a.time.values
    di, dj = int(round(wy)), int(round(wx))
    iy, ix = iy0 + di, ix0 + dj
    clat, clon = float(ds.latitude[iy, ix]), float(ds.longitude[iy, ix])
    meta = {"site_id": site.site_id, "lat_used": site.lat, "lon_used": site.lon, "elevation_m": site.elevation_m,
            "cell_iy": iy, "cell_ix": ix, "cell_lat": clat, "cell_lon": clon, "cell_size_m": DX,
            "distance_to_cell_centre_km": haversine_m(site.lat, site.lon, clat, clon) / 1e3,
            "start": str(start), "end": str(end)}
    cols = {"time_utc": pd.to_datetime(times)}
    for suf, ix_ in (("", 1), ("_NN", 0)):
        r = {v: raw[v][ix_] for v in VARS}
        t, td = r["temperature_2m"], r["dew_point_temperature_2m"]
        cols["TA" + suf] = t
        cols["VPD" + suf] = physics.vpd_hpa(t, td)
        sw = r["downward_short_wave_radiation_flux_surface"]
        cols["SW_IN" + suf] = np.where(sw > 1361, np.nan, sw).astype(np.float32)
        cols["LW_IN" + suf] = r["downward_long_wave_radiation_flux_surface"]
        cols["P" + suf] = (3600.0 * r["precipitation_surface"]).astype(np.float32)
        cols["WS" + suf] = physics.wind_speed(r["wind_u_10m"], r["wind_v_10m"])
        cols["PA" + suf] = (r["pressure_surface"] / 1000.0).astype(np.float32)
        cols["D_SNOW" + suf] = (100.0 * r["snow_thickness_surface"]).astype(np.float32)
    df = pd.DataFrame(cols)
    order = ["time_utc"] + [c for v in MAPPING for c in (v, v + "_NN")]
    return df[order], meta


def run(site_ids, start, end, root: Path = ROOT):
    import pyarrow as pa
    import pyarrow.parquet as pq
    ds = open_store()
    tr = transformer(ds)
    sites = load_sites().set_index("site_id")
    out = root / "hrrr"
    out.mkdir(parents=True, exist_ok=True)
    used_path = out / "sites_used.csv"
    used = pd.read_csv(used_path) if used_path.exists() else pd.DataFrame()
    for sid in site_ids:
        dst = out / f"{sid}.parquet"
        if dst.exists():
            continue
        s = sites.loc[sid]
        t = time.time()
        df, meta = extract_site(ds, tr, pd.Series({**s.to_dict(), "site_id": sid}), start, end)
        if df is None:
            print(f"{sid}: outside the HRRR domain", flush=True)
            continue
        write_table(df, dst, list(df.columns))
        used = pd.concat([used[used.get("site_id", pd.Series(dtype=str)) != sid] if len(used) else used,
                          pd.DataFrame([meta])])
        used.to_csv(used_path, index=False)
        print(f"{sid}: {len(df)} h, {time.time() - t:.0f} s, {dst.stat().st_size / 1e6:.1f} MB", flush=True)
    card = {"source": "hrrr", "description": __doc__.split("\n\n")[0],
            "access": "s3://dynamical-noaa-hrrr/noaa-hrrr-analysis/v0.2.0.icechunk (anonymous, us-west-2)",
            "upstream": "s3://noaa-hrrr-bdp-pds (NOAA Open Data Dissemination)",
            "licence": "CC-BY-4.0 (dynamical.org); NOAA data are public domain",
            "accessed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "grid": "Lambert conformal conic, 3 km, 1799 x 1059, sphere R = 6371229 m",
            "columns": "<VAR> bilinear from the 4 surrounding cells; <VAR>_NN nearest cell",
            "variables": {v: {"unit": u, "from": f, "interval": i} for v, (u, f, i) in MAPPING.items()},
            "source_variables": VARS}
    (out / "source.json").write_text(json.dumps(card, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="+", required=True)
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2021-01-01")
    ap.add_argument("--root", type=Path, default=ROOT)
    a = ap.parse_args(argv)
    run(a.sites, pd.Timestamp(a.start), pd.Timestamp(a.end), a.root)


if __name__ == "__main__":
    main()
