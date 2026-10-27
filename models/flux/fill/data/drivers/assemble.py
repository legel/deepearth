"""Per-site driver tables from the extraction work files.

Output, per source: `<root>/<source>/<SITE>.parquet` with `time_utc` (hourly, UTC, the source's valid time) and per
FLUXNET variable two float32 columns: `<VAR>` (bilinear value at the tower) and `<VAR>_NN` (nearest grid cell with a
value). `<root>/<source>/sites_used.csv` records the tower coordinates used, the nearest cell centre, the cell size,
the cell orography and the distance from the tower to the cell centre. `<root>/<source>/source.json` records units,
the variable mapping, the time convention of each variable and the provenance of every source file.

    python assemble.py --source era5_land [--sites N]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import physics
from grid import haversine_m
from openmeteo import EPOCH, SOURCES, START, static_field
from sites import load_sites

ROOT = Path("/home/photon/4tb/fluxfill/drivers")

# Storage precision: values are rounded to these steps (finer than the sources' own quantisation: 0.05 K for
# temperature, 0.001 m3 m-3 for soil moisture, 1 W m-2 for radiation in the Open-Meteo files).
PRECISION = {"TA": 0.01, "VPD": 0.01, "SW_IN": 0.1, "LW_IN": 0.1, "P": 0.001, "WS": 0.01, "PA": 0.001, "SWC_1": 0.01,
             "SWC_2": 0.01, "TS_1": 0.01, "TS_2": 0.01, "D_SNOW": 0.1}


def write_table(df, path, order):
    """Parquet, zstd, float columns rounded to PRECISION and byte-stream-split, time delta-encoded."""
    df = df[order].copy()
    for c in order[1:]:
        q = PRECISION.get(c.replace("_NN", ""))
        if q:
            df[c] = (np.round(df[c].values / q) * q).astype(np.float32)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="zstd", compression_level=9,
                   use_dictionary=False, column_encoding={"time_utc": "DELTA_BINARY_PACKED",
                                                          **{c: "BYTE_STREAM_SPLIT" for c in order[1:]}})


# FLUXNET variable -> (unit, how it is formed from the source variables, time convention).
MAPPING = {
    "era5_land": {
        "TA": ("deg C", "temperature_2m", "instant"),
        "VPD": ("hPa", "es(temperature_2m) - es(dew_point_2m), Alduchov and Eskridge (1996)", "instant"),
        "SWC_1": ("%", "100 x soil_moisture_0_to_7cm (volumetric)", "instant"),
        "SWC_2": ("%", "100 x soil_moisture_7_to_28cm (volumetric)", "instant"),
        "TS_1": ("deg C", "soil_temperature_0_to_7cm", "instant"),
        "TS_2": ("deg C", "soil_temperature_7_to_28cm", "instant"),
        "D_SNOW": ("cm", "100 x snow_depth (m)", "instant"),
    },
    "era5": {
        "TA": ("deg C", "temperature_2m", "instant"),
        "VPD": ("hPa", "es(temperature_2m) - es(dew_point_2m)", "instant"),
        "SW_IN": ("W m-2", "shortwave_radiation (ssrd / 3600)", "mean of the preceding hour"),
        "P": ("mm", "precipitation (tp x 1000)", "sum of the preceding hour"),
        "WS": ("m s-1", "hypot(wind_u_component_10m, wind_v_component_10m), 10 m", "instant"),
        "PA": ("kPa", "pressure_msl (Pa) moved to the cell orography with temperature_2m, standard lapse rate", "instant"),
    },
}


def _series(work: Path, var: str, sites: pd.Series, nt_total: int, end_h: int):
    """(nearest, bilinear) arrays (n_sites, nt_total) on the hourly grid from START, year files before chunks."""
    start_h = int((START - EPOCH) / pd.Timedelta("1h"))
    nn = np.full((len(sites), nt_total), np.nan, np.float32)
    bl = np.full((len(sites), nt_total), np.nan, np.float32)
    files = sorted((work / var).glob("*.npz"), key=lambda p: (0 if p.name.startswith("year") else 1, p.name))
    prov = []
    for f in files:
        if ".tmp" in f.name:
            continue
        z = np.load(f, allow_pickle=True)  # work files written by openmeteo.py only
        ids = z["site_id"].astype(str)
        pos = pd.Index(ids).get_indexer(sites)
        ok = pos >= 0
        t0 = int(z["t0_hours"]) - start_h
        nt = int(z["nt"])
        a, b = max(t0, 0), min(t0 + nt, nt_total, end_h - start_h + 1)
        if b <= a:
            continue
        src_n = z["nearest"][pos[ok], a - t0:b - t0]
        src_b = z["bilinear"][pos[ok], a - t0:b - t0]
        tgt_n = nn[ok, a:b]
        fill = np.isnan(tgt_n)
        tgt_n[fill] = src_n[fill]
        tgt_b = bl[ok, a:b]
        fill = np.isnan(tgt_b)
        tgt_b[fill] = src_b[fill]
        nn[ok, a:b], bl[ok, a:b] = tgt_n, tgt_b
        prov.append({"key": str(z["key"]), "modified": str(z["modified"])})
    return nn, bl, prov


def assemble(source: str, root: Path = ROOT, n_sites: int = None, site_ids=None):
    src = SOURCES[source]
    work = root / "_work" / source
    meta = json.loads((work / "meta.json").read_text())
    end_h = int(meta["data_end_time"] // 3600)
    start_h = int((START - EPOCH) / pd.Timedelta("1h"))
    nt_total = end_h - start_h + 1
    sites = load_sites()
    if site_ids:
        sites = sites[sites.site_id.isin(site_ids)].reset_index(drop=True)
    if n_sites:
        sites = sites.iloc[:n_sites]
    ids = sites.site_id
    raw, prov = {}, {}
    for var in src.variables:
        nn, bl, prov[var] = _series(work, var, ids, nt_total, end_h)
        raw[var] = (nn, bl)
    # Grid metadata per site from one work file.
    z = np.load(sorted((work / "temperature_2m").glob("year_*.npz"))[0], allow_pickle=True)
    zid = pd.Index(z["site_id"].astype(str))
    hs = static_field(src.model)
    g = src.grid
    out = root / source
    out.mkdir(parents=True, exist_ok=True)
    time_utc = pd.date_range(START, periods=nt_total, freq="1h")
    used = []
    for k, s in enumerate(sites.itertuples()):
        p = zid.get_loc(s.site_id)
        c = int(z["corner"][p])
        i0, j0, j1 = int(z["i0"][p]), int(z["j0"][p]), int(z["j1"][p])
        di, dj = [(0, 0), (0, 1), (1, 0), (1, 1)][max(c, 0)]
        i, j = i0 + di, (j1 if dj else j0)
        clat, clon = g.centre(i, j)
        zc = float(hs[i, j])
        # Bilinear orography from the same corners (sea cells excluded for land-only products).
        cz = np.array([hs[i0, j0], hs[i0, j1], hs[i0 + 1, j0], hs[i0 + 1, j1]], float)
        from grid import corner_weights
        w = corner_weights(float(z["wy"][p]), float(z["wx"][p]))
        valid = cz > -998
        zb = float((cz * w)[valid].sum() / w[valid].sum()) if valid.any() else np.nan
        used.append({"site_id": s.site_id, "lat_used": s.lat, "lon_used": s.lon, "elevation_m": s.elevation_m,
                     "cell_i": i, "cell_j": j, "cell_lat": clat, "cell_lon": clon, "cell_dlat": g.dlat,
                     "cell_dlon": g.dlon, "cell_orography_m": zc, "bilinear_orography_m": zb,
                     "distance_to_cell_centre_km": haversine_m(s.lat, s.lon, clat, clon) / 1e3,
                     "nearest_corner": c})
        cols = {"time_utc": time_utc}
        for suf, ix in (("", 1), ("_NN", 0)):
            zz = zb if suf == "" else zc
            r = {v: raw[v][ix][k] for v in src.variables}
            t, td = r["temperature_2m"], r["dew_point_2m"]
            cols["TA" + suf] = t
            cols["VPD" + suf] = physics.vpd_hpa(t, td)
            if source == "era5_land":
                cols["SWC_1" + suf] = physics.fraction_to_pct(r["soil_moisture_0_to_7cm"])
                cols["SWC_2" + suf] = physics.fraction_to_pct(r["soil_moisture_7_to_28cm"])
                cols["TS_1" + suf] = r["soil_temperature_0_to_7cm"]
                cols["TS_2" + suf] = r["soil_temperature_7_to_28cm"]
                cols["D_SNOW" + suf] = (100.0 * r["snow_depth"]).astype(np.float32)
            else:
                cols["SW_IN" + suf] = r["shortwave_radiation"]
                cols["P" + suf] = r["precipitation"]
                cols["WS" + suf] = physics.wind_speed(r["wind_u_component_10m"], r["wind_v_component_10m"])
                cols["PA" + suf] = physics.surface_pressure_kpa(r["pressure_msl"] / 100.0, t, zz)  # Pa -> hPa
        df = pd.DataFrame(cols)
        order = ["time_utc"] + [c for v in MAPPING[source] for c in (v, v + "_NN")]
        write_table(df, out / f"{s.site_id}.parquet", order)
    pd.DataFrame(used).to_csv(out / "sites_used.csv", index=False)
    card = {
        "source": source, "description": src.doc, "open_meteo_model": src.model,
        "access": f"s3://openmeteo/data/{src.model}/ (anonymous; https://openmeteo.s3.amazonaws.com)",
        "licence": "CC-BY-4.0 (Open-Meteo); underlying data: Copernicus Climate Change Service (C3S) licence",
        "accessed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "grid": {"lat0": g.lat0, "lon0": g.lon0, "dlat": g.dlat, "dlon": g.dlon, "ny": g.ny, "nx": g.nx,
                 "note": "cell (i, j) centre = (lat0 + i dlat, lon0 + j dlon)"},
        "time": "time_utc is the hourly valid time in UTC; see each variable's 'interval'",
        "precision": PRECISION,
        "derived_not_stored": "RH: RH = 100 (1 - VPD / es(TA)), es = physics.es_hpa",
        "columns": "<VAR> = bilinear value at the tower from the four surrounding cell centres (cells without a "
                   "value dropped, weights renormalised); <VAR>_NN = nearest cell with a value",
        "variables": {v: {"unit": u, "from": f, "interval": i} for v, (u, f, i) in MAPPING[source].items()},
        "data_end_utc": str(pd.Timestamp(meta["data_end_time"], unit="s")),
        "files": prov,
    }
    (out / "source.json").write_text(json.dumps(card, indent=1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=list(SOURCES))
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--sites", type=int, default=None)
    ap.add_argument("--site-ids", nargs="*", default=None)
    a = ap.parse_args(argv)
    print(assemble(a.source, a.root, a.sites, a.site_ids))


if __name__ == "__main__":
    main()
