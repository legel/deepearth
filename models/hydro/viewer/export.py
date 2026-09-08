"""Build the viewer payload: a downsampled terrain, the flood animation, and the overlays.

Everything here is small and committed, so `python3 cli.py viewer` works on a fresh clone. The
solver's own outputs are not: a full-resolution point cloud is 1,011 MB and the mesh frames are
gigabytes. The payload is a view, not the data.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.transform import from_bounds

import frames as frames_io
from sites import SiteConfig

GRID = 256
"""Terrain and texture resolution served to the browser. 256x256 float32 is 256 kB, which is
the whole point: the twin is inspected at a glance, not streamed."""


def _data_dir(site: SiteConfig) -> Path:
    """The committed payload directory for one site."""
    d = Path(__file__).resolve().parent / "data" / site.name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _storm_label(storm_name: str) -> str:
    """Human-readable storm name, or the key if the registry does not carry one."""
    from sites import STORMS

    storm = STORMS.get(storm_name)
    return storm.label if storm is not None else storm_name


def export_terrain(site: SiteConfig, storm_name: Optional[str] = None,
                   cell_size_m: float = 25.0) -> Dict:
    """Downsample the conditioned DEM to GRID x GRID and write it with its scene metadata.

    `meta.json` is the page's only source of names and numbers. Anything the template would
    otherwise hardcode -- the storm it is animating, the two catchment areas the discharge
    caption compares -- is written here, so a second storm or a second site does not need the
    HTML edited.
    """
    assert site.dem_conditioned.exists(), f"{site.dem_conditioned} missing; run the terrain stage"
    with rasterio.open(site.dem_conditioned) as src:
        left, right = sorted([src.bounds.left, src.bounds.right])
        bottom, top = sorted([src.bounds.bottom, src.bounds.top])
        nodata = src.nodata if src.nodata is not None else -9999.0
        out = np.zeros((GRID, GRID), dtype=np.float32)
        reproject(src.read(1).astype(np.float32), out,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=from_bounds(left, bottom, right, top, GRID, GRID),
                  dst_crs=src.crs, src_nodata=nodata, dst_nodata=nodata,
                  resampling=Resampling.bilinear)

    valid = out != nodata
    out[~valid] = float(np.nanmin(out[valid]))
    d = _data_dir(site)
    (d / "dem.bin").write_bytes(out.astype("<f4").tobytes())

    meta = {
        "site": site.name, "label": site.label,
        "rows": GRID, "cols": GRID,
        "z_min": float(out.min()), "z_max": float(out.max()),
        "width_m": float(right - left), "height_m": float(top - bottom),
        # The area the SOLVER integrated, not the DEM's own extent. The two differ by 0.4 %
        # because the solver grid is a whole number of cells, and the chart caption compares
        # this against the gauge's catchment -- so it has to be the same number `validate`
        # divides by, or the page and the report quote different domains.
        "domain_km2": (int((top - bottom) / cell_size_m) * int((right - left) / cell_size_m)
                       * cell_size_m ** 2 / 1e6),
        "lat": site.lat, "lon": site.lon,
        "gauge": None if site.gauge is None else {
            "site_no": site.gauge.site_no,
            "documented_area_km2": site.gauge.documented_area_km2,
            "delineated_area_km2": site.gauge.delineated_area_km2,
        },
    }
    if storm_name is not None:
        meta["storm"] = {"name": storm_name, "label": _storm_label(storm_name)}
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def export_flood(site: SiteConfig, storm_name: str, cell_size_m: float,
                 stride: int = 1) -> Optional[Dict]:
    """Re-emit a solver SIML frame file at GRID resolution, optionally dropping frames.

    Args:
        site: Site whose outputs to read.
        storm_name: Storm key.
        cell_size_m: Resolution the storm was run at.
        stride: Keep every nth frame. 2 halves the payload and still reads as continuous.
    """
    src_path = site.out_path(f"frames_{storm_name}_{cell_size_m:g}m.bin")
    if not src_path.exists():
        return None

    times, frames = frames_io.read(src_path)
    n, rows, cols = frames.shape

    keep = np.arange(0, n, stride)
    out = np.zeros((len(keep), GRID, GRID), dtype=np.float32)
    ry = np.linspace(0, rows - 1, GRID).astype(int)
    rx = np.linspace(0, cols - 1, GRID).astype(int)
    for i, k in enumerate(keep):
        out[i] = frames[k][np.ix_(ry, rx)]

    frames_io.write(_data_dir(site) / f"flood_{storm_name}.bin", out, times[keep])
    return {"frames": int(out.shape[0]), "hours": float(times[keep][-1] / 60.0)}



def export_hydrograph(site: SiteConfig, storm_name: str, cell_size_m: float) -> Optional[Dict]:
    """Simulated and observed discharge on one time axis, for the viewer's chart."""
    from validate import CFS_TO_CMS, load_observed
    from sites import get_storm

    path = site.out_path(f"hydrograph_{storm_name}_{cell_size_m:g}m.csv")
    if not path.exists():
        return None
    data = np.genfromtxt(path, delimiter=",", names=True)
    step = max(1, len(data["time_h"]) // 1500)  # the chart cannot resolve 12,960 points

    payload = {
        "time_h": data["time_h"][::step].tolist(),
        "rain_mm_hr": data["rain_mm_hr"][::step].tolist(),
        "flooded_ha": data["flooded_ha"][::step].tolist(),
        "sim_cfs": (data["outflow_total_cms"][::step] / CFS_TO_CMS).tolist(),
    }
    if site.gauge is not None and site.discharge(storm_name).exists():
        t_obs, q_obs = load_observed(site, get_storm(storm_name))
        payload["obs_time_h"] = t_obs.tolist()
        payload["obs_cfs"] = q_obs.tolist()

    (_data_dir(site) / f"hydrograph_{storm_name}.json").write_text(json.dumps(payload))
    return {"points": len(payload["time_h"]), "observed": "obs_cfs" in payload}


def export_overlay(site: SiteConfig, name: str, source: Path, cmap: str = "viridis",
                   size: int = 1024) -> Optional[str]:
    """Reproject a raster onto the terrain's bounds and write it as a draped PNG texture."""
    if not source.exists():
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with rasterio.open(site.dem_conditioned) as ref:
        left, right = sorted([ref.bounds.left, ref.bounds.right])
        bottom, top = sorted([ref.bounds.bottom, ref.bounds.top])
        dst_crs, dst_tf = ref.crs, from_bounds(left, bottom, right, top, size, size)

    # NaN, not zero, is the "nothing reprojected here" sentinel. Zero is a legal pixel value in
    # every one of these rasters, so a zero-filled destination cannot be told apart from real
    # data -- which is how the aerial drape came to paint 23 % of the domain opaque black.
    with rasterio.open(source) as src:
        bands = min(src.count, 3)
        out = np.full((bands, size, size), np.nan, dtype=np.float32)
        for b in range(bands):
            reproject(src.read(b + 1).astype(np.float32), out[b],
                      src_transform=src.transform, src_crs=src.crs,
                      src_nodata=src.nodata, dst_nodata=np.nan,
                      dst_transform=dst_tf, dst_crs=dst_crs, resampling=Resampling.bilinear)

    path = _data_dir(site) / f"{name}.png"
    if bands == 3:
        # Alpha, not black. NAIP ships in UTM and the DEM is in Albers, ~8.8 deg apart at this
        # longitude, so the imagery rectangle does not reach the domain's corners however large
        # the mosaic is. Uncovered ground must show the terrain through, not a black wall.
        rgb = np.moveaxis(out, 0, -1)
        covered = np.isfinite(rgb).all(axis=-1)
        rgba = np.zeros((size, size, 4), dtype=np.float32)
        rgba[..., :3] = np.clip(np.nan_to_num(rgb) / max(float(np.nanmax(rgb)), 1e-6), 0.0, 1.0)
        rgba[..., 3] = covered
        plt.imsave(path, rgba)
        return name

    a = out[0]
    finite = np.isfinite(a) & (a > -9000)
    assert finite.any(), f"{source} reprojects to nothing over {site.name}'s domain"
    lo, hi = np.percentile(a[finite], [2, 98])
    scaled = np.clip((a - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)

    # Mask the gaps and pin the range explicitly. imsave autoscales with plain min/max, so a
    # single NaN anywhere makes vmin and vmax NaN and EVERY pixel renders as the colormap's
    # "bad" colour -- a fully transparent PNG that looks like a working export until you open
    # it. Both draped overlays shipped that way once, byte-identical to each other.
    plt.imsave(path, np.ma.masked_where(~finite, scaled), cmap=cmap, vmin=0.0, vmax=1.0)
    return name


def export_all(site: SiteConfig, storm_name: str = "ian", cell_size_m: float = 25.0,
               stride: int = 2) -> Dict:
    """Build the whole payload for one site."""
    summary: Dict[str, object] = {"terrain": export_terrain(site, storm_name, cell_size_m)}
    summary["flood"] = export_flood(site, storm_name, cell_size_m, stride)
    summary["hydrograph"] = export_hydrograph(site, storm_name, cell_size_m)
    summary["overlays"] = [n for n in (
        export_overlay(site, "naip", site.naip_rgb),
        export_overlay(site, "hand", site.hand, cmap="RdYlBu"),
        export_overlay(site, "impervious", site.nlcd_impervious, cmap="inferno"),
    ) if n]
    return summary
