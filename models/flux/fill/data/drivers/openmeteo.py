"""Point series from the Open-Meteo open-data mirror on AWS (s3://openmeteo, anonymous, CC-BY-4.0).

Open-Meteo stores each model variable as `.om` files: `data/<model>/<variable>/year_<YYYY>.om` for complete years
and `chunk_<n>.om` for recent data, where chunk n starts at hour n * chunk_time_length after 1970-01-01T00 UTC.
Each array is (latitude, longitude, time) on the model's own grid, compressed in small tiles (1 x 6 cells x 504 or
1095 hours), so the 2 x 2 cells around a tower are read with a few HTTP range requests per year.

Provenance (verified 2026-10-26):
- `copernicus_era5_land`: ERA5-Land (Munoz-Sabater et al. 2021), 0.1 deg, hourly, the CDS product regridded only
  by reordering to lat -90..90, lon -180..179.9. Variables held: 2 m temperature and dew point, soil moisture and
  soil temperature in the four ERA5-Land layers, snow depth. Open-Meteo does not hold the ERA5-Land precipitation,
  radiation, wind or pressure: its API serves those from ERA5 for the `era5_land` model.
- `copernicus_era5`: ERA5 (Hersbach et al. 2020), 0.25 deg, hourly, incl. the preliminary ERA5T for recent days.
  ERA5-Land is forced by these same ERA5 fields, interpolated to 0.1 deg (Munoz-Sabater et al. 2021, section 2), so
  ERA5 at the tower is the ERA5-Land forcing for radiation, precipitation and wind, up to the interpolation.

Time convention: values are at the stamped hour t (UTC) for state variables (temperature, dew point, wind,
pressure, soil, snow). Shortwave radiation is the mean and precipitation the sum over the preceding hour (t - 1 h, t].

    python openmeteo.py extract --source era5_land --out /home/photon/4tb/fluxfill/drivers [--sites N]
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from grid import RegularGrid, combine
from sites import load_sites

BUCKET = "openmeteo"
BASE_URL = "https://openmeteo.s3.amazonaws.com"


@dataclass(frozen=True)
class OmSource:
    model: str
    grid: RegularGrid
    variables: Dict[str, str]
    """Open-Meteo variable name -> unit."""
    land_only: bool
    doc: str


SOURCES = {
    "era5_land": OmSource(
        "copernicus_era5_land", RegularGrid(-90.0, -180.0, 0.1, 0.1, 1801, 3600),
        {"temperature_2m": "deg C", "dew_point_2m": "deg C", "soil_moisture_0_to_7cm": "m3 m-3",
         "soil_moisture_7_to_28cm": "m3 m-3", "soil_temperature_0_to_7cm": "deg C",
         "soil_temperature_7_to_28cm": "deg C", "snow_depth": "m"},
        True, "ERA5-Land 0.1 deg hourly (Munoz-Sabater et al. 2021), via the Open-Meteo AWS mirror"),
    "era5": OmSource(
        "copernicus_era5", RegularGrid(-90.0, -180.0, 0.25, 0.25, 721, 1440),
        {"temperature_2m": "deg C", "dew_point_2m": "deg C", "shortwave_radiation": "W m-2 (mean of preceding hour)",
         "precipitation": "mm (sum of preceding hour)", "wind_u_component_10m": "m s-1",
         "wind_v_component_10m": "m s-1", "pressure_msl": "Pa"},
        False, "ERA5 0.25 deg hourly (Hersbach et al. 2020), via the Open-Meteo AWS mirror"),
}

START = pd.Timestamp("2014-01-01T00:00:00")
EPOCH = pd.Timestamp("1970-01-01T00:00:00")


def list_files(model: str, var: str) -> List[dict]:
    """Year and chunk files of a variable with their first hour (hours since 1970) and size, from 2014 on."""
    import requests
    import re
    out, token = [], None
    while True:
        params = {"list-type": "2", "prefix": f"data/{model}/{var}/"}
        if token:
            params["continuation-token"] = token
        r = _get(requests, BASE_URL + "/", params=params)
        xml = r.text
        for key, mod, size in re.findall(r"<Key>([^<]+)</Key><LastModified>([^<]+)</LastModified>.*?<Size>(\d+)</Size>",
                                         xml):
            name = key.rsplit("/", 1)[-1]
            m = re.match(r"(year|chunk)_(\d+)\.om$", name)
            if not m:
                continue
            out.append({"key": key, "kind": m.group(1), "n": int(m.group(2)), "modified": mod, "size": int(size)})
        t = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not t:
            break
        token = t.group(1)
    sel = []
    for f in out:
        if f["kind"] == "year":
            if f["n"] < START.year:
                continue
            f["t0_hours"] = int((pd.Timestamp(f"{f['n']}-01-01") - EPOCH) / pd.Timedelta("1h"))
        else:
            f["t0_hours"] = None  # filled from meta.json chunk_time_length
        sel.append(f)
    return sel


def _get(requests, url, tries=6, **kw):
    for k in range(tries):
        try:
            r = requests.get(url, timeout=60, **kw)
            if r.status_code == 200:
                return r
        except Exception:
            pass
        time.sleep(min(60, 2 ** k))
    raise IOError(f"GET failed after {tries} tries: {url}")


def meta(model: str) -> dict:
    import requests
    return _get(requests, f"{BASE_URL}/data/{model}/static/meta.json").json()


def site_blocks(sites: pd.DataFrame, grid: RegularGrid) -> pd.DataFrame:
    rows = []
    for s in sites.itertuples():
        i0, j0, j1, wy, wx = grid.corners(s.lat, s.lon)
        rows.append({"site_id": s.site_id, "lat": s.lat, "lon": s.lon, "i0": i0, "j0": j0, "j1": j1,
                     "wy": wy, "wx": wx})
    return pd.DataFrame(rows)


async def _read_blocks(reader, blocks: List[tuple], nt: int) -> Dict[tuple, np.ndarray]:
    """Read every (i0, j0, j1) 2 x 2 block as a (4, nt) array, corners in `grid.corner_weights` order.

    The omfiles reader serialises reads within a process (measured 2026-10-26: 16 concurrent readers in one process
    read no faster than one), so blocks are read in sequence here and parallelism comes from worker processes.
    Each block read issues its tile requests concurrently."""
    out = {}
    for b in blocks:
        i0, j0, j1 = b
        for k in range(7):
            try:
                if j1 == j0 + 1:
                    a = await reader.read_array((slice(i0, i0 + 2), slice(j0, j0 + 2), slice(0, nt)))
                    a = np.asarray(a, np.float32).reshape(2, 2, nt)
                else:  # across the date line
                    a0 = await reader.read_array((slice(i0, i0 + 2), slice(j0, j0 + 1), slice(0, nt)))
                    a1 = await reader.read_array((slice(i0, i0 + 2), slice(j1, j1 + 1), slice(0, nt)))
                    a = np.concatenate([np.asarray(a0, np.float32).reshape(2, 1, nt),
                                        np.asarray(a1, np.float32).reshape(2, 1, nt)], axis=1)
                out[b] = a.reshape(4, nt)
                break
            except Exception:  # network errors: back off and retry
                if k == 6:
                    raise
                await asyncio.sleep(2 ** k)
    return out


async def _extract(key: str, tab: pd.DataFrame):
    import fsspec
    import omfiles
    fs = fsspec.filesystem("s3", anon=True, asynchronous=True, skip_instance_cache=True)
    for k in range(7):
        try:
            reader = await omfiles.OmFileReaderAsync.from_fsspec(fs, f"{BUCKET}/{key}")
            break
        except Exception:
            if k == 6:
                raise
            await asyncio.sleep(2 ** k)
    nt = reader.shape[2]
    blocks = sorted({(r.i0, r.j0, r.j1) for r in tab.itertuples()})
    data = await _read_blocks(reader, blocks, nt)
    nn = np.full((len(tab), nt), np.nan, np.float32)
    bl = np.full((len(tab), nt), np.nan, np.float32)
    corner = np.full(len(tab), -1, np.int8)
    for k, r in enumerate(tab.itertuples()):
        a, b, c = combine(data[(r.i0, r.j0, r.j1)], r.wy, r.wx)
        nn[k], bl[k], corner[k] = a, b, c
    return reader.shape, nn, bl, corner


TAB_COLS = ["i0", "j0", "j1", "wy", "wx", "lat", "lon"]


def _task(args):
    """One (variable, file): read the sites not yet in its work file (or whose cell changed) and merge them in."""
    key, modified, t0_hours, dst, tab = args
    dst = Path(dst)
    old = None
    if dst.exists():
        try:
            z = np.load(dst, allow_pickle=True)  # work files written by this module only
            if str(z["modified"]) == modified:
                old = {k: z[k] for k in z.files}
        except Exception:
            old = None
    todo = tab
    if old is not None:
        prev = pd.DataFrame({"site_id": old["site_id"].astype(str), **{c: old[c] for c in TAB_COLS}})
        m = tab.merge(prev, on="site_id", how="left", suffixes=("", "_old"))
        same = np.isclose(m.lat, m.lat_old) & np.isclose(m.lon, m.lon_old)
        todo = tab[~np.asarray(same)]
        if todo.empty:
            return None
    t = time.time()
    shape, nn, bl, corner = asyncio.run(_extract(key, todo))
    rec = {"site_id": np.asarray(todo.site_id.tolist(), dtype="U32"), "nearest": nn, "bilinear": bl, "corner": corner,
           **{c: todo[c].values for c in TAB_COLS}}
    if old is not None:
        keep = ~np.isin(old["site_id"].astype(str), rec["site_id"])
        for k in ["site_id", "nearest", "bilinear", "corner"] + TAB_COLS:
            rec[k] = np.concatenate([np.asarray(old[k][keep]).astype(rec[k].dtype), rec[k]])
    tmp = dst.with_name(dst.stem + ".tmp.npz")
    np.savez_compressed(tmp, **rec, t0_hours=t0_hours, nt=shape[2], modified=modified, key=key)
    tmp.rename(dst)
    return f"{key}: +{len(todo)} sites ({len(rec['site_id'])} in file), {shape[2]} h, {time.time() - t:.0f} s"


def build_tasks(source: str, out: Path, sites: pd.DataFrame, variables=None, years=None, kinds=None):
    src = SOURCES[source]
    tab = site_blocks(sites, src.grid)
    work = out / "_work" / source
    work.mkdir(parents=True, exist_ok=True)
    m = meta(src.model)
    (work / "meta.json").write_text(json.dumps(m))
    ctl = m["chunk_time_length"]
    start_h = (START - EPOCH) / pd.Timedelta("1h")
    tasks = []
    for var in (variables or src.variables):
        (work / var).mkdir(exist_ok=True)
        for f in list_files(src.model, var):
            if f["kind"] == "chunk":
                f["t0_hours"] = f["n"] * ctl
                if (f["n"] + 1) * ctl <= start_h:
                    continue
            if years and not (f["kind"] == "year" and f["n"] in years):
                continue
            if kinds and f["kind"] not in kinds:
                continue
            dst = work / var / (Path(f["key"]).stem + ".npz")
            tasks.append((f["key"], f["modified"], f["t0_hours"], str(dst), tab))
    # Year files first: they are the long tasks.
    tasks.sort(key=lambda a: (0 if "/year_" in a[0] else 1, a[0]))
    return tasks


def run_extract(source: str, out: Path, n_sites: int = None, workers: int = 16, variables=None, site_ids=None,
                years=None, kinds=None, reverse=False):
    """reverse: process the task list from its end. A second job started with reverse=True on the same work
    directory meets the first in the middle: each task skips a file that is already complete."""
    import multiprocessing as mp
    sites = load_sites()
    if site_ids:
        sites = sites[sites.site_id.isin(site_ids)].reset_index(drop=True)
    if n_sites:
        sites = sites.iloc[:n_sites]
    tasks = build_tasks(source, out, sites, variables, years, kinds)
    if reverse:
        tasks = tasks[::-1]
    print(f"{source}: {len(tasks)} file tasks, {len(sites)} sites, {workers} workers", flush=True)
    t = time.time()
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=20) as pool:
        for k, msg in enumerate(pool.imap_unordered(_task, tasks)):
            if msg:
                print(f"[{k + 1}/{len(tasks)} {time.time() - t:.0f} s] {msg}", flush=True)
    print(f"{source}: done in {time.time() - t:.0f} s", flush=True)


def static_field(model: str, name: str = "HSURF") -> np.ndarray:
    """A static field (orography HSURF, m) as a full (lat, lon) array; about 2 MB."""
    import fsspec
    import omfiles
    fs = fsspec.filesystem("s3", anon=True)
    r = omfiles.OmFileReader.from_fsspec(fs, f"{BUCKET}/data/{model}/static/{name}.om")
    return np.asarray(r.read_array(tuple(slice(0, n) for n in r.shape)), np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract"])
    ap.add_argument("--source", required=True, choices=list(SOURCES))
    ap.add_argument("--out", type=Path, default=Path("/home/photon/4tb/fluxfill/drivers"))
    ap.add_argument("--sites", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--vars", nargs="*", default=None)
    ap.add_argument("--site-ids", nargs="*", default=None)
    ap.add_argument("--years", nargs="*", type=int, default=None, help="only these year files (validation runs)")
    ap.add_argument("--kinds", nargs="*", default=None, choices=["year", "chunk"])
    ap.add_argument("--reverse", action="store_true")
    a = ap.parse_args(argv)
    run_extract(a.source, a.out, a.sites, a.workers, a.vars, a.site_ids, a.years, a.kinds, a.reverse)


if __name__ == "__main__":
    main()
