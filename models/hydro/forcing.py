"""Rainfall forcing. The forcing is the site's measured rain; a design storm is only a stated what-if.

- **Tower rain (the forcing):** the nearest flux tower's hourly rain under the P rule of `models/flux`
  (`rain.rain`): the reference gauge where it measured, else the tower gauge vetted hour by hour and month by month
  against the site's independent sources, else AORC. `tower_hyetograph` reads that hourly record over a storm.
- **Gridded or station rain:** AORC's 1 km hourly grid (`aorc_hyetograph`), or a site's ASOS gauge
  (`observed_hyetograph`) where no tower is near.
- **What-if only:** NOAA Atlas 14 depths on an SCS Type II curve (`design_hyetograph`), named as a design storm
  wherever it is shown, never as a measured one.

Every route produces the same thing, a rainfall rate [m/s] on a `dt_s`-spaced axis, so the solver never knows which
it is running.
"""

import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sites import SiteConfig, Storm

SCS_II_T = np.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
SCS_II_P = np.array([0, .04, .1, .2, .35, .63, .78, .87, .93, .97, 1.0])
"""SCS Type II dimensionless cumulative rainfall, the Florida standard. Peak intensity lands at
about 60 % of the duration, which suits convective storms here."""

RETURN_PERIODS_YR = (1, 2, 5, 10, 25, 50, 100, 200, 500)
"""Design-storm ensemble. Atlas 14 is queried per coordinate, so this runs unchanged anywhere,
which is what makes cross-site comparison mean anything."""


def tower_hyetograph(path, storm: Storm, dt_s: float, extend_hours: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Rainfall rate [m/s] from the tower's hourly record under the P rule (`models/flux` `rain.rain`), over the storm.

    Args:
        path: A CSV of the rule's output, one row an hour: `time_utc` (the hour's start) and `p` (mm in the hour).
        storm: Storm defining the UTC window.
        dt_s: Output timestep [s].
        extend_hours: Zero-rain hours appended for the solver's drainage tail.

    Returns:
        (rate [m/s] per dt_s step, hourly depth [mm]).
    """
    df = pd.read_csv(path)
    assert {"time_utc", "p"} <= set(df.columns), f"{path} needs time_utc and p columns (the P rule's hourly output)"
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df = df.set_index("time_utc").sort_index()
    window = df[pd.Timestamp(storm.start, tz="UTC"):pd.Timestamp(storm.end, tz="UTC")]
    assert not window.empty, f"{path} has no hours inside {storm.start}..{storm.end}"
    rain_mm = pd.to_numeric(window["p"], errors="coerce").fillna(0.0).values
    t = np.arange(0.0, (len(rain_mm) + extend_hours) * 3600.0, dt_s)
    h = (t // 3600.0).astype(int)                    # each step takes its own hour's rate: every hour's depth exact
    return np.where(h < len(rain_mm), rain_mm[np.minimum(h, len(rain_mm) - 1)] / 1000 / 3600, 0.0), rain_mm


def observed_hyetograph(site: SiteConfig, storm: Storm, dt_s: float,
                        extend_hours: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Rainfall rate [m/s] from the site's ASOS record over the storm window.

    Args:
        site: Site whose ASOS station to read.
        storm: Storm defining the UTC window.
        dt_s: Output timestep [s].
        extend_hours: Zero-rain hours appended so the solver's own drainage tail is not cut
            off. A magnitude comparison that truncates the model but not the gauge is unfair in
            a way that has been measured, not assumed.

    Returns:
        (rate [m/s] per dt_s step, hourly depth [mm] as recorded).
    """
    path = site.asos(storm.name)
    assert path.exists(), f"{path} missing; run `python3 cli.py fetch --site {site.name}`"

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    time_col = next((c for c in df.columns if "datetime" in c or "valid" in c or "time" in c),
                    df.columns[0])
    precip_col = next((c for c in df.columns if "prcp" in c or "p01" in c or "precip" in c), None)
    assert precip_col is not None, f"{path} has no recognisable precipitation column"

    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()
    window = df[pd.Timestamp(storm.start, tz="UTC"):pd.Timestamp(storm.end, tz="UTC")]
    assert not window.empty, f"{path} has no records inside {storm.start}..{storm.end}"

    rain_mm = pd.to_numeric(window[precip_col], errors="coerce").fillna(0.0).values
    hours = np.arange(len(rain_mm), dtype=float)
    total_s = (len(rain_mm) + extend_hours) * 3600.0
    t = np.arange(0.0, total_s, dt_s)
    return np.interp(t, hours * 3600.0, rain_mm / 1000 / 3600, right=0.0), rain_mm


AORC = "noaa-nws-aorc-v1-1-1km/{year}.zarr"
"""NOAA's Analysis of Record for Calibration, hourly on a 1 km grid (public, AWS)."""


def zstd_compat() -> None:
    """Let an older numcodecs read AORC's zarr: its Zstd codec config carries `checksum`, which numcodecs before 0.13
    does not take. libzstd checks a frame's checksum when it decodes, so the flag is only dropped from the config."""
    import numcodecs
    from numcodecs import registry
    try:
        numcodecs.Zstd(checksum=False)
        return
    except TypeError:
        pass

    class Zstd(numcodecs.Zstd):
        codec_id = "zstd"

        def __init__(self, level=0, checksum=False):
            super().__init__(level=level)

    registry.register_codec(Zstd)


def aorc_hyetograph(storm: Storm, dt_s: float, profile: Dict, valid: np.ndarray,
                    extend_hours: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """The storm's rain from AORC on the solver grid: every cell takes its own 1 km cell's hours.

    The field is carried as the domain's mean hourly depth times each cell's share of the storm total (its total over
    the domain's mean), so the rain the mass balance counts is the grid's own. AORC stamps an hour's accumulation at
    the hour's end.

    Returns:
        (rate [m/s] per dt_s step of the domain mean, its hourly depth [mm], the per-cell weight, a record with the
        domain-mean total and the total at the domain's centroid cell, and each hour's timing as `Surface.rain_hourly`
        takes it: each source cell's share of the hour's domain mean and each solver cell's source).
    """
    import s3fs
    import xarray as xr
    from pyproj import Transformer

    zstd_compat()
    t0, t1 = np.datetime64(storm.start), np.datetime64(storm.end)
    fs = s3fs.S3FileSystem(anon=True)
    ds = xr.open_zarr(fs.get_mapper(AORC.format(year=int(str(t0)[:4]))), consolidated=True)
    rows, cols = valid.shape
    tr = profile["transform"]
    xs = tr.c + (np.arange(cols) + 0.5) * tr.a
    ys = tr.f + (np.arange(rows) + 0.5) * tr.e
    X, Y = np.meshgrid(xs, ys)
    lon, lat = Transformer.from_crs(profile["crs"], 4326, always_xy=True).transform(X, Y)
    la_all, lo_all = ds["latitude"].values, ds["longitude"].values
    ki = np.flatnonzero((la_all >= lat.min() - 0.02) & (la_all <= lat.max() + 0.02))
    kj = np.flatnonzero((lo_all >= lon.min() - 0.02) & (lo_all <= lon.max() + 0.02))
    box = ds["APCP_surface"].isel(latitude=slice(ki.min(), ki.max() + 1), longitude=slice(kj.min(), kj.max() + 1)).sel(
        time=slice(t0 + np.timedelta64(1, "h"), t1 + np.timedelta64(1, "h")))
    p = np.nan_to_num(box.values.astype(np.float64))                         # [hours, lat, lon] mm
    la, lo = box["latitude"].values, box["longitude"].values
    iy = np.abs(la[None, None, :] - lat[..., None]).argmin(axis=-1)
    ix = np.abs(lo[None, None, :] - lon[..., None]).argmin(axis=-1)
    total = p.sum(axis=0)[iy, ix]                                            # each solver cell's storm total
    mean_h = np.array([p[h][iy, ix][valid].mean() for h in range(p.shape[0])]) if valid.any() else p.mean(axis=(1, 2))
    weight = np.where(valid, total / max(float(total[valid].mean()), 1e-9), 0.0)
    r, c = np.nonzero(valid)
    cy, cx = int(round(r.mean())), int(round(c.mean()))
    hours = np.arange(len(mean_h), dtype=float)
    t = np.arange(0.0, (len(mean_h) + extend_hours) * 3600.0, dt_s)
    rec = {"domain_mean_total_mm": round(float(mean_h.sum()), 1),
           "centroid_total_mm": round(float(p[:, iy[cy, cx], ix[cy, cx]].sum()), 1),
           "cell_total_mm_p5_p95": [round(float(np.percentile(total[valid], q)), 1) for q in (5, 95)],
           "aorc_cells": int(len(np.unique(iy[valid] * len(lo) + ix[valid])))}
    code = iy * len(lo) + ix
    src, inv = np.unique(code[valid], return_inverse=True)
    index = np.zeros(valid.shape, dtype=np.int64)
    index[valid] = inv
    by_src = p.reshape(p.shape[0], -1)[:, src]                               # [hours, source cells] mm
    share = np.where(mean_h[:, None] > 0, by_src / np.maximum(mean_h[:, None], 1e-12), 1.0)
    return np.interp(t, hours * 3600.0, mean_h / 1000 / 3600, right=0.0), mean_h, weight, rec, (share, index)


def design_hyetograph(depth_mm: float, duration_hr: float, dt_s: float) -> np.ndarray:
    """SCS Type II rainfall rate [m/s] for a total depth over a duration."""
    assert depth_mm > 0 and duration_hr > 0, "design storm needs a positive depth and duration"
    total_s = duration_hr * 3600.0
    edges = np.arange(0.0, total_s + dt_s, dt_s)
    cumulative = np.interp(edges / total_s, SCS_II_T, SCS_II_P) * (depth_mm / 1000.0)
    return np.diff(cumulative) / dt_s


def atlas14_depth_mm(site: SiteConfig, return_period_yr: int, duration_hr: float) -> float:
    """Design rainfall depth [mm] for one return period and duration.

    Reads the fetched Atlas 14 table. It carries a provenance field because this query silently
    fell back to hardcoded county defaults for months -- a stale URL plus a parser expecting the
    wrong JSON shape -- printing a plausible table the whole time.
    """
    assert site.atlas14.exists(), (
        f"{site.atlas14} missing; run `python3 cli.py fetch --site {site.name}`")
    table = json.loads(site.atlas14.read_text())
    assert table.get("source") == "pfds", (
        f"{site.atlas14} was not fetched from NOAA PFDS (source={table.get('source')!r}); "
        f"re-run the fetch rather than modelling on fallback values")

    key = f"{duration_hr:g}hr"
    assert key in table["depths_mm"], f"duration {key} not in {sorted(table['depths_mm'])}"
    row = table["depths_mm"][key]
    assert str(return_period_yr) in row, f"return period {return_period_yr} not in {sorted(row)}"
    return float(row[str(return_period_yr)])
