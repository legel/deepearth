"""The region a joint model maps: which species, which training points, which calibration areas.

The published product maps the contiguous United States (CONUS). From the national training data (all US natives),
``restrict`` keeps the species native to CONUS, drops every training point (presence or background) inside the
excluded areas (Alaska with the Aleutians, Hawaii), keeps the points in the species' native ranges abroad (Canada,
Mexico, Eurasia, as Daru 2024 trains on whole native ranges), renumbers the species and subsets the plot sets.

``extend_calibration`` makes sure every species native to the mapped region has a calibration area on its grid. A
species' calibration area is the set of RESOLVE ecoregions holding its cleaned records; when all of them lie off the
grid (records only in Canada or Mexico, or at the edge of a large ecoregion) its map would be empty there. Its
calibration area is then extended with the ecoregions of its WCVP native areas inside the region: those with at
least ``min_share`` of their area inside a native area (the rule used for species without records), else, for small
native areas that hold no such ecoregion, those intersecting it at all.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

CHUNK = 1 << 24                                      # points per pass


def excluded_area(l3_geojson: str | Path, codes: Sequence[str], buffer_deg: float):
    """Union of the WGSRPD level-3 areas ``codes``, buffered by ``buffer_deg`` degrees (coastal records), in a
    longitude frame continuous across 180 degrees: parts east of 180 W are shifted by -360 (the Aleutians)."""
    import geopandas as gpd
    import shapely
    g = gpd.read_file(l3_geojson)
    area = g[g.LEVEL3_COD.isin(list(codes))].geometry.unary_union
    east = shapely.affinity.translate(shapely.intersection(area, shapely.box(0, -90, 180, 90)), -360)
    area = east.union(shapely.intersection(area, shapely.box(-180, -90, 0, 90))).buffer(buffer_deg)
    shapely.prepare(area)
    return area


def in_excluded(lat: np.ndarray, lon: np.ndarray, area, boxes: Sequence[dict] = ()) -> np.ndarray:
    """True for points inside ``area`` (``excluded_area``) or inside any box {"lat": [lo, hi], "lon": [lo, hi]}
    (open intervals)."""
    import shapely
    lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
    out = np.zeros(len(lat), bool)
    for b in boxes:
        out |= (lat > b["lat"][0]) & (lat < b["lat"][1]) & (lon > b["lon"][0]) & (lon < b["lon"][1])
    if area is not None and not area.is_empty:
        x = np.where(lon > 0, lon - 360, lon)
        x0, y0, x1, y1 = area.bounds
        cand = np.flatnonzero((x >= x0) & (x <= x1) & (lat >= y0) & (lat <= y1))
        if len(cand):
            out[cand] |= shapely.contains_xy(area, x[cand], lat[cand])
    return out


def restrict(src: str | Path, dst: str | Path, region: str, l3_geojson: str | Path, exclude_l3: Sequence[str],
             buffer_deg: float, exclude_boxes: Sequence[dict] = (), drop_plot_sets: Sequence[str] = (),
             log=print) -> dict:
    """Write the training data of ``src`` (a national data directory) restricted to ``region`` (a code of the
    species table's ``us_regions`` column, e.g. CONUS) into ``dst``. Per-point arrays are streamed."""
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    area = excluded_area(l3_geojson, exclude_l3, buffer_deg) if exclude_l3 else None
    sp = pd.read_csv(src / "species.csv")
    keep_sp = sp.us_regions.fillna("").str.split(",").apply(lambda r: region in r).values
    new_id = np.full(len(sp), -1, np.int64)
    new_id[keep_sp] = np.arange(keep_sp.sum())
    sp[keep_sp].to_csv(dst / "species.csv", index=False)
    lat, lon = np.load(src / "lat.npy", mmap_mode="r"), np.load(src / "lon.npy", mmap_mode="r")
    sid, pres = np.load(src / "sid.npy", mmap_mode="r"), np.load(src / "pres.npy", mmap_mode="r")
    N = len(sid)
    keep = np.zeros(N, bool)
    n_out = 0
    for i in range(0, N, CHUNK):
        la, lo, s = np.asarray(lat[i:i + CHUNK]), np.asarray(lon[i:i + CHUNK]), np.asarray(sid[i:i + CHUNK])
        out = in_excluded(la, lo, area, exclude_boxes)
        n_out += int(out.sum())
        keep[i:i + CHUNK] = keep_sp[s] & ~out
    M = int(keep.sum())
    log(f"points: {N:,} -> {M:,} (dropped {N - M:,}; {n_out:,} inside the excluded areas)")
    X = np.load(src / "train_points_cache.npy", mmap_mode="r")
    arrays = (("lat", lat), ("lon", lon), ("sid", sid), ("pres", pres), ("train_points_cache", X))
    outs = {k: np.lib.format.open_memmap(dst / f"{k}.npy", mode="w+", dtype=a.dtype, shape=(M,) + a.shape[1:])
            for k, a in arrays}
    o = 0
    for i in range(0, N, CHUNK):
        m = keep[i:i + CHUNK]
        n = int(m.sum())
        for k, a in arrays:
            v = np.asarray(a[i:i + CHUNK])[m]
            outs[k][o:o + n] = new_id[v].astype(a.dtype) if k == "sid" else v
        o += n
    for a in outs.values():
        a.flush()
    if o != M or not (np.asarray(outs["sid"]) >= 0).all():
        raise RuntimeError("restriction lost track of the points")
    for f in ["joint_data.npz", *sorted(p.name for p in src.glob("plots_*.npz"))]:
        if any(k in f for k in drop_plot_sets):
            continue
        z = dict(np.load(src / f))
        if "eval_sid" in z:
            k = keep_sp[z["eval_sid"]]
            for key in ("eval_y", "maxent_occ", "maxent_daru"):
                if key in z:
                    z[key] = z[key][k]
            z["eval_sid"] = new_id[z["eval_sid"][k]].astype(z["eval_sid"].dtype)
            log(f"{f}: {k.sum()} of {len(k)} evaluated species kept")
        for key in ("lon", "lat", "sid", "pres", "X"):
            z.pop(key, None)                                  # per-point arrays live in the .npy files above
        np.savez(dst / f, **z)
    if (src / "names.npy").exists():
        np.save(dst / "names.npy", np.load(src / "names.npy"))
    (dst / "tree").exists() or (dst / "tree").symlink_to((src / "tree").resolve())
    meta = json.loads((src / "meta.json").read_text()) if (src / "meta.json").exists() else {}
    meta.update({"scope": region, "species": int(keep_sp.sum()), "points": M,
                 "presences": int(np.asarray(outs["pres"]).sum()), "source": src.name})
    (dst / "meta.json").write_text(json.dumps(meta, indent=1))
    log(json.dumps(meta))
    return meta


_INTERSECTING: dict = {}


def l3_intersecting(codes: Sequence[str], l3_geojson: str | Path, ecoregions_shp: str | Path) -> set[int]:
    """RESOLVE ecoregion ids intersecting the given WGSRPD level-3 areas at all (equal-area EPSG:6933)."""
    import geopandas as gpd
    key = (str(l3_geojson), str(ecoregions_shp))
    if key not in _INTERSECTING:
        _INTERSECTING[key] = (gpd.read_file(l3_geojson)[["LEVEL3_COD", "geometry"]].to_crs(6933),
                              gpd.read_file(ecoregions_shp)[["ECO_ID", "geometry"]].to_crs(6933))
    l3, eco = _INTERSECTING[key]
    hit = gpd.sjoin(eco, l3[l3.LEVEL3_COD.isin(list(codes))], predicate="intersects")
    return {int(e) for e in hit.ECO_ID}


def extend_calibration(labels: Sequence[str], calibration: Sequence[str], inventory: pd.DataFrame,
                       on_grid: set[int], l3eco: dict[str, list[int]], region_l3: Sequence[str],
                       l3_geojson: str | Path, ecoregions_shp: str | Path, log=print) -> list[str]:
    """Calibration areas (space-separated ecoregion ids, one per label) with every species whose area misses the
    grid (``on_grid``: ecoregion ids present on it) extended by the ecoregions of its WCVP native areas that lie in
    the region (``region_l3``: the region's WGSRPD level-3 codes; ``inventory``: columns ``wcvp_accepted_name``,
    ``native_l3``)."""
    native = dict(zip(inventory.wcvp_accepted_name.str.replace(" ", "_"), inventory.native_l3.fillna("")))
    region_l3 = set(region_l3)
    out, n = [], 0
    for label, c in zip(labels, calibration):
        eco = {int(x) for x in str(c).split()}
        if not (eco & on_grid):
            codes = [x.strip() for x in str(native.get(label, "")).split(",") if x.strip() in region_l3]
            add = {e for code in codes for e in l3eco.get(code, []) if e in on_grid}
            if not add and codes:                                  # small areas cover < min_share of every ecoregion
                add = {e for e in l3_intersecting(codes, l3_geojson, ecoregions_shp) if e in on_grid}
            if add:
                eco |= add
                n += 1
        out.append(" ".join(map(str, sorted(eco))))
    log(f"calibration: {n} species extended with the ecoregions of their native areas in the region")
    return out
