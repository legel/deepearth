"""Gridded precipitation at the towers from the dynamical.org Icechunk archives (anonymous, CC-BY-4.0).

- IMERG V07 Late (Huffman et al. 2023), 0.1 deg, half-hourly, global 60 N-60 S (partial to the poles), from
  1998, about 14 h latency. Merged passive-microwave and infrared, gauge-adjusted where available. Each value is the
  mean rate over the half hour starting at its time stamp. Source: s3://dynamical-nasa-imerg/nasa-imerg-analysis-late.
- MRMS hourly QPE (Zhang et al. 2016), 0.01 deg (about 1 km), CONUS, from 2014-11-01. Gauge-corrected radar
  (GaugeCorr_QPE_01H to October 2020, MultiSensor_QPE_01H_Pass2 after). Each value is the mean rate over the
  preceding hour. Source: s3://dynamical-noaa-mrms/noaa-mrms-conus-analysis-hourly.

Both are stored hourly as P (mm in the hour ending at time_utc), matching the ERA5 convention: IMERG half-hours
starting at T - 60 min and T - 30 min are summed into hour T.

    python precip.py --product imerg --sites US-Bi1 ... --start 2019-01-01 --end 2021-01-01
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from assemble import write_table
from grid import combine, corner_weights, haversine_m
from sites import load_sites

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path("/home/photon/4tb/fluxfill/drivers")
PRODUCTS = {
    "imerg": dict(bucket="dynamical-nasa-imerg", prefix="nasa-imerg-analysis-late/v0.1.0.icechunk", step_min=30,
                  d=0.1, doc="IMERG V07 Late, 0.1 deg, half-hourly"),
    "mrms": dict(bucket="dynamical-noaa-mrms", prefix="noaa-mrms-conus-analysis-hourly/v0.3.0.icechunk",
                 step_min=60, d=0.01, doc="MRMS gauge-corrected hourly QPE, 0.01 deg, CONUS"),
}


def open_product(name):
    import icechunk
    import xarray as xr
    p = PRODUCTS[name]
    st = icechunk.s3_storage(bucket=p["bucket"], prefix=p["prefix"], region="us-west-2", anonymous=True)
    return xr.open_zarr(icechunk.Repository.open(st).readonly_session("main").store, chunks=None, consolidated=False)


def latlon_corners(lat, lon, lats: np.ndarray, lons: np.ndarray):
    """2 x 2 block on axes lats (descending) and lons (ascending): (i0, j0, wy, wx) with rows i0, i0 + 1."""
    dlat, dlon = lats[1] - lats[0], lons[1] - lons[0]
    fi, fj = (lat - lats[0]) / dlat, (lon - lons[0]) / dlon
    i0, j0 = int(np.floor(fi)), int(np.floor(fj))
    if i0 < 0 or j0 < 0 or i0 + 1 >= len(lats) or j0 + 1 >= len(lons):
        return None
    return i0, j0, float(fi - i0), float(fj - j0)


def hourly_from_half_hourly(t: np.ndarray, mm: np.ndarray):
    """Sum half-hours starting at T - 60 and T - 30 min into hour T (mm). Both must exist."""
    s = pd.Series(mm, index=pd.to_datetime(t))
    hours = pd.date_range(s.index[0].ceil("h") + pd.Timedelta("1h"), s.index[-1].floor("h") + pd.Timedelta("1h"),
                          freq="1h")
    a = s.reindex(hours - pd.Timedelta("60min")).values
    b = s.reindex(hours - pd.Timedelta("30min")).values
    return hours, (a + b).astype(np.float32)


def run(product, site_ids, start, end, root=ROOT):
    ds = open_product(product)
    p = PRODUCTS[product]
    v = ds["precipitation_surface"]
    lats, lons = ds.latitude.values, ds.longitude.values
    sites = load_sites().set_index("site_id")
    out = root / product
    out.mkdir(parents=True, exist_ok=True)
    used_p = out / "sites_used.csv"
    used = pd.read_csv(used_p) if used_p.exists() else pd.DataFrame(columns=["site_id"])
    for sid in site_ids:
        dst = out / f"{sid}.parquet"
        if dst.exists():
            continue
        s = sites.loc[sid]
        c = latlon_corners(s.lat, s.lon, lats, lons)
        if c is None:
            print(f"{sid}: outside {product}", flush=True)
            continue
        i0, j0, wy, wx = c
        t = time.time()
        for k in range(6):
            try:
                a = v.sel(time=slice(np.datetime64(start) - np.timedelta64(1, "h"), np.datetime64(end))).isel(
                    latitude=slice(i0, i0 + 2), longitude=slice(j0, j0 + 2)).load()
                break
            except Exception:
                if k == 5:
                    raise
                time.sleep(2 ** k)
        block = a.values.reshape(a.shape[0], 4).T * 60.0 * p["step_min"]  # mm per step
        nn, bl, corner = combine(block, wy, wx)
        if p["step_min"] == 30:
            hours, bl_h = hourly_from_half_hourly(a.time.values, bl)
            _, nn_h = hourly_from_half_hourly(a.time.values, nn)
        else:
            hours, bl_h, nn_h = pd.to_datetime(a.time.values), bl, nn
        df = pd.DataFrame({"time_utc": hours, "P": bl_h, "P_NN": nn_h})
        df = df[(df.time_utc > pd.Timestamp(start)) & (df.time_utc <= pd.Timestamp(end))]
        write_table(df, dst, ["time_utc", "P", "P_NN"])
        di, dj = [(0, 0), (0, 1), (1, 0), (1, 1)][max(corner, 0)]
        clat, clon = float(lats[i0 + di]), float(lons[j0 + dj])
        used = pd.concat([used[used.site_id != sid], pd.DataFrame([{
            "site_id": sid, "lat_used": s.lat, "lon_used": s.lon, "cell_lat": clat, "cell_lon": clon,
            "cell_size_deg": p["d"], "distance_to_cell_centre_km": haversine_m(s.lat, s.lon, clat, clon) / 1e3,
            "start": str(start), "end": str(end)}])])
        used.to_csv(used_p, index=False)
        print(f"{sid}: {len(df)} h, {time.time() - t:.0f} s", flush=True)
    card = {"source": product, "description": p["doc"], "access": f"s3://{p['bucket']}/{p['prefix']} (anonymous)",
            "licence": "CC-BY-4.0 (dynamical.org); NASA / NOAA data are public domain",
            "accessed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "variables": {"P": {"unit": "mm", "interval": "sum of the preceding hour"}},
            "columns": "P bilinear from the 4 surrounding cells; P_NN nearest cell"}
    (out / "source.json").write_text(json.dumps(card, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", choices=list(PRODUCTS), required=True)
    ap.add_argument("--sites", nargs="+", required=True)
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2021-01-01")
    a = ap.parse_args(argv)
    run(a.product, a.sites, pd.Timestamp(a.start), pd.Timestamp(a.end))


if __name__ == "__main__":
    main()
