"""Gather every Sentinel-2 L2A acquisition over every tower as 300 m x 300 m chips of all bands.

For each site, in the order of `sites.csv`:
1. Search Earth Search `sentinel-2-c1-l2a` and Planetary Computer `sentinel-2-l2a` for every item containing the
   tower, from mission start to now; reduce to one scene per acquisition (`stac.select`).
2. Skip acquisitions already committed to the site store (resume).
3. Read each band's window with HTTP range reads of the COG (GDAL /vsicurl/), in a thread pool, with retries and
   exponential backoff. No cloud filtering: every scene is kept, and SCL and the cloud probability are stored.
4. Append chips in batches and commit (`store.SiteStore`).

    python gather.py --sites <sentinel2 root>/sites.csv --root <sentinel2 root> [--threads 16] [--only US-Ha1,...]
"""

import os

# GDAL configuration for windowed COG reads over HTTP; set before rasterio loads GDAL.
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.TIF")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
os.environ.setdefault("GDAL_HTTP_VERSION", "2")
os.environ.setdefault("GDAL_INGESTED_BYTES_AT_OPEN", "32768")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
os.environ.setdefault("GDAL_CACHEMAX", "256")
os.environ.setdefault("VSI_CACHE", "FALSE")
os.environ.setdefault("GDAL_HTTP_USERAGENT", "FluxFill-Sentinel2/1.0 (UC Berkeley QED lab)")

import argparse
import json
import random
import shutil
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window as RioWindow

from bands import CLD_UNAVAILABLE, GROUPS, scl_fractions
from geometry import chip_window, lonlat_to_utm, read_window
from stac import PCToken, dump_scenes, scenes_for
from store import SiteStore, to_us

DISK_FLOOR_GB = 45.0   # the shared disk keeps at least 40 GB free; stop 5 GB above that
PROJECTION_CAP_GB = 40.0
PROJECTION_AFTER_SITES = 20


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def _open_read(href: str, x: float, y: float, size_m: int, res: int, dtype):
    """Open one COG, check its grid, and read the chip window. Returns (array, Window)."""
    with rasterio.open(href) as ds:
        t = ds.transform
        if abs(t.a - res) > 1e-6 or abs(t.e + res) > 1e-6 or t.b != 0 or t.d != 0:
            raise ValueError(f"unexpected grid {t} for {href}")
        win = chip_window(x, y, t.c, t.f, res, size_m // res)
        arr = read_window(lambda c, r, w, h: ds.read(1, window=RioWindow(c, r, w, h)), win, ds.width, ds.height,
                          dtype, fill=0)
    return arr, win


def read_band(href: str, x, y, res, dtype, token: PCToken | None, tries: int = 6):
    for k in range(tries):
        try:
            h = token.sign(href) if token is not None else href
            return _open_read(h, x, y, 300, res, dtype)
        except rasterio.errors.RasterioIOError as e:
            if k == tries - 1:
                raise
            if token is not None and ("403" in str(e) or "AuthenticationFailed" in str(e)):
                token._token = None   # force refresh
            time.sleep(min(120.0, 2.0 * 2 ** k) * (0.5 + random.random()))


def read_scene(scene, lon: float, lat: float, token: PCToken | None) -> dict:
    """All band windows of one scene, plus the per-chip metadata fields."""
    x, y = lonlat_to_utm(lon, lat, scene.epsg)
    chip, wins = {}, {}
    for grp in GROUPS:
        planes = []
        for b in grp.bands:
            if b not in scene.hrefs:          # CLD is absent from Planetary Computer items
                planes.append(np.full((grp.size, grp.size), CLD_UNAVAILABLE, dtype=grp.dtype))
                continue
            arr, win = read_band(scene.hrefs[b], x, y, grp.res, grp.dtype,
                                 token if scene.source == "planetary-computer" else None)
            if grp.name in wins and wins[grp.name] != win:
                raise ValueError(f"{scene.item_id}: band grids differ within {grp.name}")
            wins[grp.name] = win
            planes.append(arr)
        chip[grp.name] = np.stack(planes) if grp.name in ("r10", "r20", "r60") else planes[0]
    has_cld = "CLD" in scene.hrefs
    w10 = wins["r10"]
    cx, cy = w10.centre
    m = scene.meta()
    m.update(chip)
    m["time_us"] = to_us(scene.datetime_utc)
    m["has_cld"] = has_cld
    for g in ("r10", "r20", "r60"):
        m[f"x_ul_{g}"], m[f"y_ul_{g}"] = wins[g].ul
    m["tower_dx_m"], m["tower_dy_m"] = x - cx, y - cy
    m["nodata_frac"] = float((chip["r10"] == 0).all(axis=0).mean())
    m.update(scl_fractions(chip["scl"]))
    m["cld_mean"] = float(chip["cld"].mean()) if has_cld else float("nan")
    return m


def gather_site(row, root: Path, threads: int, token: PCToken, batch: int = 32) -> dict:
    sid, lat, lon = row.site_id, float(row.lat), float(row.lon)
    t0 = time.time()
    scenes, counts = scenes_for(lon, lat)
    dump_scenes(scenes, root / "scenes" / f"{sid}.jsonl")
    store = SiteStore(root / "sites" / f"{sid}.zarr", sid, lat, lon)
    done = store.done_keys()
    todo = [s for s in scenes if s.acq_key not in done]
    log(f"{sid}: {sum(counts.values())} items {counts} -> {len(scenes)} acquisitions; {len(done)} done, "
        f"{len(todo)} to read")
    failures, pending, buf, n_new = [], {}, [], 0
    it = iter(todo)
    with ThreadPoolExecutor(threads) as ex:
        for s in it:
            pending[ex.submit(read_scene, s, lon, lat, token)] = s
            if len(pending) >= threads * 2:
                break
        while pending:
            fin, _ = wait(pending, return_when=FIRST_COMPLETED)
            for f in fin:
                s = pending.pop(f)
                try:
                    buf.append(f.result())
                except Exception as e:  # recorded and retried on the next run
                    failures.append({"acq_key": s.acq_key, "item_id": s.item_id, "error": repr(e)[:500]})
                nxt = next(it, None)
                if nxt is not None:
                    pending[ex.submit(read_scene, nxt, lon, lat, token)] = nxt
            if len(buf) >= batch or (not pending and buf):
                store.append(buf)
                n_new += len(buf)
                buf = []
    dt = time.time() - t0
    with open(root / "failures" / f"{sid}.jsonl", "w") as f:
        for r in failures:
            f.write(json.dumps(r) + "\n")
    complete = not failures
    store.set_attrs(complete=complete, searched_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
                    item_counts=counts, n_acquisitions=len(scenes))
    st = {"site_id": sid, "items": counts, "acquisitions": len(scenes), "chips": len(store), "new": n_new,
          "failed": len(failures), "seconds": round(dt, 1), "chips_per_hour": round(n_new / dt * 3600, 1) if dt else 0,
          "complete": complete}
    (root / "status" / f"{sid}.json").write_text(json.dumps(st))
    return st


def du_gb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--redo-complete", action="store_true", help="re-search complete sites for new acquisitions")
    a = ap.parse_args()
    for d in ("sites", "scenes", "failures", "status"):
        (a.root / d).mkdir(parents=True, exist_ok=True)
    sites = pd.read_csv(a.sites)
    if a.only:
        sites = sites[sites.site_id.isin(a.only.split(","))]
    if a.limit:
        sites = sites.head(a.limit)
    token = PCToken()
    tot_new, t_start, n_sites = 0, time.time(), 0
    for row in sites.itertuples():
        free = shutil.disk_usage(a.root).free / 1e9
        if free < DISK_FLOOR_GB:
            log(f"STOP: {free:.1f} GB free on {a.root}, under the {DISK_FLOOR_GB:.0f} GB floor")
            sys.exit(2)
        stp = a.root / "status" / f"{row.site_id}.json"
        if stp.exists() and json.loads(stp.read_text()).get("complete") and not a.redo_complete:
            n_sites += 1
            continue
        try:
            st = gather_site(row, a.root, a.threads, token)
        except Exception:
            log(f"{row.site_id}: site failed\n{traceback.format_exc()}")
            continue
        n_sites += 1
        tot_new += st["new"]
        el = time.time() - t_start
        log(f"{row.site_id}: {st['chips']} chips ({st['new']} new, {st['failed']} failed) in {st['seconds']} s, "
            f"{st['chips_per_hour']} chips/h; run total {tot_new} new chips, {tot_new / el * 3600:.0f} chips/h")
        if n_sites >= PROJECTION_AFTER_SITES and not (a.root / "PROJECTION.txt").exists():
            statuses = [json.loads(p.read_text()) for p in (a.root / "status").glob("*.json")]
            chips = sum(s["chips"] for s in statuses)
            size = du_gb(a.root / "sites")
            proj = size / max(chips, 1) * (chips / len(statuses)) * len(pd.read_csv(a.sites))
            msg = (f"PROJECTION after {len(statuses)} sites: {chips} chips, {size:.2f} GB, "
                   f"{size / max(chips, 1) * 1e6:.0f} bytes/chip; projected {proj:.1f} GB for all sites")
            log(msg)
            (a.root / "PROJECTION.txt").write_text(msg + "\n")
            if proj > PROJECTION_CAP_GB:
                log(f"STOP: projection exceeds {PROJECTION_CAP_GB:.0f} GB")
                sys.exit(3)
    log(f"DONE: {n_sites} sites, {tot_new} new chips")
    if not a.only and not a.limit:
        (a.root / "GATHER_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()))


if __name__ == "__main__":
    main()
