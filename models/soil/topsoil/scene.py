"""Soil layers over a Simulator scene (v0): pH, texture (sand / silt / clay + USDA class), organic matter, drainage.

v0 = the best field-calibrated coarse prior, placed exactly, plus terrain where it is physically justified:
  * SSURGO map-unit polygons under the scene (Soil Data Access, mupolygon intersected with the scene's box), each map
    unit's components weighted by comppct_r and their horizons thickness-weighted over 0-30 cm (the site_build soil
    stage's arithmetic, per map unit instead of per site); rasterised on the scene's ENU grid by point-in-polygon.
    Resolution is the map unit's (≥ ~0.6 ha delineations): stated as such in the provenance. Urban land / rock
    outcrop units without horizon data are NaN (and flagged), never filled.
  * drainage: the map unit's drainage class on the trait ontology's 0 (poor) .. 3 (well drained) scale, modulated
    within the scene by the store's 2 m topographic wetness index: index = class_value - 0.5 * z(TWI) clipped to
    [0, 3] (wetter-than-site-average positions drain worse). Physically: TWI = ln(a / tan b) ranks where subsurface
    flow converges; SSURGO sets the soil's intrinsic drainage.
Not served: NPK. SSURGO / SOLUS / POLARIS / SoilGrids carry no plant-available N, P or K (labile pools, fertiliser
history, seasonal); KSSL pedons have extractable P / K at points only. A site soil test is the defensible source.
Writes SIMF v2 (FRAMES.md) on the scene's ENU grid: rows north-down along basis_north, columns along basis_east.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

USDA = ["sand", "loamy sand", "sandy loam", "loam", "silt loam", "silt", "sandy clay loam", "clay loam",
        "silty clay loam", "sandy clay", "silty clay", "clay"]
DRAIN = {"Excessively drained": 3.0, "Somewhat excessively drained": 2.7, "Well drained": 2.4,
         "Moderately well drained": 1.8, "Somewhat poorly drained": 1.2, "Poorly drained": 0.6, "Very poorly drained": 0.2}


def usda_class(sand, silt, clay):
    """USDA texture triangle -> index into USDA (NaN-safe: -1)."""
    if not all(np.isfinite(v) for v in (sand, silt, clay)):
        return -1
    if silt + 1.5 * clay < 15:
        return 0
    if silt + 1.5 * clay < 30:
        return 1
    if (7 <= clay < 20 and sand > 52 and silt + 2 * clay >= 30) or (clay < 7 and silt < 50 and silt + 2 * clay >= 30):
        return 2
    if 7 <= clay < 27 and 28 <= silt < 50 and sand <= 52:
        return 3
    if (silt >= 50 and 12 <= clay < 27) or (50 <= silt < 80 and clay < 12):
        return 4
    if silt >= 80 and clay < 12:
        return 5
    if 20 <= clay < 35 and silt < 28 and sand > 45:
        return 6
    if 27 <= clay < 40 and 20 < sand <= 45:
        return 7
    if 27 <= clay < 40 and sand <= 20:
        return 8
    if clay >= 35 and sand > 45:
        return 9
    if clay >= 40 and silt >= 40:
        return 10
    return 11


def mapunits(lon_lat_ring):
    """Map-unit polygons (WKT, WGS 84) intersecting the ring, with per-unit 0-30 cm properties and drainage class."""
    from ssurgo_token import query, _f
    wkt = "POLYGON((" + ", ".join(f"{x:.7f} {y:.7f}" for x, y in lon_lat_ring) + "))"
    rows = query(f"""~DeclareGeometry(@aoi)~
select @aoi = geometry::STPolyFromText('{wkt}', 4326)
select mukey, mupolygongeo.STIntersection(@aoi).STAsText() as wkt from mupolygon
where mupolygongeo.STIntersects(@aoi) = 1""")
    if not rows:
        return {}
    keys = ",".join(f"'{k}'" for k in {r["mukey"] for r in rows})
    comps = query(f"select mukey, cokey, compname, comppct_r from component where mukey in ({keys})")
    cok = ",".join(f"'{c['cokey']}'" for c in comps) or "''"
    hz = query(f"select cokey, hzdept_r, hzdepb_r, ph1to1h2o_r, om_r, sandtotal_r, silttotal_r, claytotal_r "
               f"from chorizon where cokey in ({cok})")
    agg = {r["mukey"]: r for r in query(f"select mukey, muname, drclassdcd from muaggatt where mukey in ({keys})")}
    by_co = {}
    for h in hz:
        by_co.setdefault(h["cokey"], []).append(h)
    out = {}
    for r in rows:
        k = r["mukey"]
        if k not in out:
            props = {}
            for p in ("ph1to1h2o_r", "om_r", "sandtotal_r", "silttotal_r", "claytotal_r"):
                num = den = 0.0
                for c in (c for c in comps if c["mukey"] == k):
                    s = t = 0.0
                    for h in by_co.get(c["cokey"], []):
                        a, b, v = _f(h["hzdept_r"]), _f(h["hzdepb_r"]), _f(h[p])
                        ov = min(b, 30.0) - max(a, 0.0)
                        if ov > 0 and np.isfinite(v):
                            s += v * ov; t += ov
                    pct = _f(c["comppct_r"])
                    if t > 0 and pct > 0:
                        num += pct * s / t; den += pct
                props[p] = num / den if den > 0 else float("nan")
            a = agg.get(k, {})
            out[k] = {"wkt": [], "props": props, "muname": a.get("muname"), "drainage_class": a.get("drclassdcd"),
                      "components": sorted({c["compname"] for c in comps if c["mukey"] == k})}
        out[k]["wkt"].append(r["wkt"])
    return out


SOLUS = {"ph": ("ph1to1h2o", 100.0), "clay": ("claytotal", 1.0), "sand": ("sandtotal", 1.0), "oc": ("soc", 1000.0)}


def solus_window(lon, lat, prop, depths=(0, 5, 15, 30)):
    """SOLUS100 (USDA-NRCS 2024, 100 m, EPSG:5070, public COGs gs://solus100pub) at points: ONE window read per depth
    raster around the points (+2 pixels), bilinear in the 100 m grid, the 0-30 cm thickness-weighted mean of the
    point depths (trapezoid between 0, 5, 15, 30 cm, as main's soils/products.py). Main's NEON benchmark (544 plots):
    pH national R² 0.81, within-site +0.18 (SSURGO -0.35)."""
    import rasterio
    from rasterio.windows import from_bounds
    from pyproj import Transformer
    name, scal = SOLUS[prop]
    x, y = Transformer.from_crs(4326, 5070, always_xy=True).transform(lon, lat)
    vals = {}
    env = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif"}
    for dpt in depths:
        with rasterio.Env(**env), rasterio.open(f"/vsicurl/https://storage.googleapis.com/solus100pub/{name}_{dpt}_cm_p.tif") as r:
            w = from_bounds(x.min() - 200, y.min() - 200, x.max() + 200, y.max() + 200, r.transform).round_offsets().round_lengths()
            a = r.read(1, window=w).astype(np.float64)
            if r.nodata is not None:
                a[a == r.nodata] = np.nan
            a /= scal
            T = r.window_transform(w)
        fc = (x - T.c) / T.a - 0.5; fr = (y - T.f) / T.e - 0.5                     # pixel-centre coordinates
        c0 = np.clip(np.floor(fc).astype(int), 0, a.shape[1] - 2); r0 = np.clip(np.floor(fr).astype(int), 0, a.shape[0] - 2)
        tx, ty = np.clip(fc - c0, 0, 1), np.clip(fr - r0, 0, 1)
        v = (a[r0, c0] * (1 - tx) * (1 - ty) + a[r0, c0 + 1] * tx * (1 - ty) + a[r0 + 1, c0] * (1 - tx) * ty +
             a[r0 + 1, c0 + 1] * tx * ty)
        vals[dpt] = v
    lay = [(0, 5), (5, 15), (15, 30)]
    prof = np.stack([(vals[a] + vals[b]) / 2 for a, b in lay], 1)
    wts = np.array([5.0, 10.0, 15.0])
    return (np.nansum(prof * wts, 1) / np.where(np.isnan(prof), 0, wts).sum(1))


def scene_grid(site, half=56.0, cell=1.0):
    """Cell centres of a north-up scene grid (ENU), row 0 north: (xs, ys [ny, nx]), origin (NW corner), shape."""
    n = int(round(2 * half / cell))
    c = -half + (np.arange(n) + 0.5) * cell
    X, Y = np.meshgrid(c, c[::-1])
    return X, Y, (-half, half), (n, n)


def soil_layers(site, half=56.0, cell=1.0, cells=None):
    """`cells` (deepearth_site.cells.Cells) when the scene spans several stores: each grid cell's TWI is its owning
    store's, standardised over that store's grid."""
    from pyproj import Transformer
    from shapely import wkt as W
    from shapely.prepared import prep
    from shapely.geometry import Point
    X, Y, origin, shape = scene_grid(site, half, cell)
    sx, sy = site.scene_to_site(X, Y)                                   # site frame
    E, N = site.frame.to_grid(sx, sy)
    lon, lat = Transformer.from_crs(f"EPSG:{site.frame.epsg}", "EPSG:4326", always_xy=True).transform(E, N)
    ring = [(lon.min() - 1e-4, lat.min() - 1e-4), (lon.max() + 1e-4, lat.min() - 1e-4), (lon.max() + 1e-4, lat.max() + 1e-4),
            (lon.min() - 1e-4, lat.max() + 1e-4), (lon.min() - 1e-4, lat.min() - 1e-4)]
    mu = mapunits(ring)
    key = np.full(shape, "", object)
    # each cell takes the first map unit whose polygon contains it, as a loop over every cell and polygon did; one
    # vectorised test a polygon (shapely 2), the loop was most of SOIL's 63 s on a 200,704-cell block (2026-10)
    import shapely
    left = np.ones(shape, bool)
    for k, w in ((k, w) for k, u in mu.items() for w in u["wkt"] if w and not w.startswith("GEOMETRYCOLLECTION EMPTY")):
        hit = left & shapely.contains_xy(W.loads(w), lon, lat)
        key[hit] = k
        left &= ~hit
    lay = {n_: np.full(shape, np.nan, np.float32) for n_ in ("ph", "sand", "silt", "clay", "om", "texture_class", "drainage", "org_class")}
    for k, u in mu.items():
        m = key == k
        pr = u["props"]
        lay["ph"][m] = pr["ph1to1h2o_r"]; lay["om"][m] = pr["om_r"]
        lay["sand"][m] = pr["sandtotal_r"]; lay["silt"][m] = pr["silttotal_r"]; lay["clay"][m] = pr["claytotal_r"]
        lay["texture_class"][m] = usda_class(pr["sandtotal_r"], pr["silttotal_r"], pr["claytotal_r"])
        dv = DRAIN.get(u["drainage_class"])
        if dv is None and np.isfinite(pr["sandtotal_r"]):
            dv = float(np.clip(1.2 + pr["sandtotal_r"] / 50, 0, 3))
        if dv is not None:
            lay["drainage"][m] = dv
    lay["texture_class"][lay["texture_class"] < 0] = np.nan
    # pH / clay / sand / organic carbon: main's NEON-validated recipe (soils/site_layers.py: SOLUS100 0-30 cm for pH /
    # clay / sand; OC = 10^mean(log SOLUS, log SoilGrids) + a site-centred terrain ridge correction, SOLUS + fitted
    # offset where SoilGrids is urban-masked), on the ~10 m 3DEP lattice, bilinear to the 1 m scene grid. SSURGO keeps
    # the map-unit names and the drainage class. Organic matter = 1.724 x OC; also as site-relative classes.
    try:
        import sys as _s
        from . import config as _cfg
        _s.path.insert(0, str(_cfg.CODE / "soils"))
        import site_layers as SL
        g10 = SL.layers(float(site.lon), float(site.lat), half_m=half * 1.5 + 30)
        lo10, la10 = g10["lon"], g10["lat"]
        x0, y0 = lo10[0, 0], la10[0, 0]; dx = lo10[0, 1] - lo10[0, 0]; dy = la10[1, 0] - la10[0, 0]
        fc = (lon - x0) / dx; fr = (lat - y0) / dy
        c0 = np.clip(np.floor(fc).astype(int), 0, lo10.shape[1] - 2); r0 = np.clip(np.floor(fr).astype(int), 0, lo10.shape[0] - 2)
        tx, ty = np.clip(fc - c0, 0, 1), np.clip(fr - r0, 0, 1)
        bil = lambda a: (a[r0, c0] * (1 - tx) * (1 - ty) + a[r0, c0 + 1] * tx * (1 - ty) + a[r0 + 1, c0] * (1 - tx) * ty  # noqa: E731
                         + a[r0 + 1, c0 + 1] * tx * ty)
        ph_s, sa_s, cl_s, oc_s = (bil(np.asarray(g10[k], np.float64)) for k in ("ph", "sand", "clay", "oc"))
        lay["ph"] = ph_s.astype(np.float32); lay["sand"] = sa_s.astype(np.float32); lay["clay"] = cl_s.astype(np.float32)
        lay["silt"] = np.clip(100 - sa_s - cl_s, 0, 100).astype(np.float32)
        lay["texture_class"] = np.array([usda_class(a_, b_, c_) for a_, b_, c_ in
                                         zip(sa_s.ravel(), lay["silt"].ravel(), cl_s.ravel())], np.float32).reshape(shape)
        lay["texture_class"][lay["texture_class"] < 0] = np.nan
        lay["om"] = (1.724 * oc_s).astype(np.float32)
        om_ = 1.724 * oc_s                       # absolute agronomic classes for mineral soils (main 15:10): within-site
        lay["org_class"] = np.where(np.isnan(om_), np.nan,   # organic skill (~0.13 R²) cannot support site terciles
                                    1 + (om_ >= 2.0) + (om_ > 4.0)).astype(np.float32)
        prior = "main's soils/site_layers.py (NEON-validated): SOLUS100 + SoilGrids OC ensemble with terrain correction"
    except Exception as e:                       # noqa: BLE001  (SSURGO values stay; stated)
        prior = f"SSURGO (site_layers unavailable: {type(e).__name__}: {e})"
    twi_note = "no TWI (store without grid2m)"
    try:
        stores = [site] if cells is None or len(cells) == 1 else cells.sites
        own = np.zeros(X.shape, np.int64) if len(stores) == 1 else cells.owner(X, Y)
        if all("twi" in st_.grid2m["names"] for st_ in stores):
            z = np.full(X.shape, np.nan)
            for k, st_ in enumerate(stores):
                g = st_.grid2m
                twi = g["values"][g["names"].index("twi")]
                half_s, res = g["half"], g["res"]
                kx, ky = (sx, sy) if k == 0 else st_.scene_to_site(X, Y)
                c_ = np.clip(((kx + half_s) / res).astype(int), 0, twi.shape[1] - 1)
                r_ = np.clip(((half_s - ky) / res).astype(int), 0, twi.shape[0] - 1)
                t = twi[r_, c_]
                zk = (t - np.nanmean(twi)) / (np.nanstd(twi) + 1e-6)
                z = zk if len(stores) == 1 else np.where(own == k, zk, z)
            lay["drainage"] = np.clip(lay["drainage"] - 0.5 * z, 0, 3).astype(np.float32)
            twi_note = "x 2 m topographic wetness index of the DeepEarth store (index = class - 0.5 z(TWI))"
    except FileNotFoundError:
        pass
    meta = {"mapunits": {k: {"name": u["muname"], "drainage_class": u["drainage_class"], "components": u["components"],
                             "props_0_30cm": u["props"], "cells": int((key == k).sum())} for k, u in mu.items()},
            "unmapped_or_no_data_cells": int(np.isnan(lay["ph"]).sum()),
            "texture_classes": USDA,
            "units": {"ph": "pH (1:1 H2O)", "sand": "%", "silt": "%", "clay": "%", "om": "% organic matter (1.724 x OC)", "org_class": "organic class, absolute (mineral soils): 1 low < 2 % OM, 2 medium 2-4 %, 3 high > 4 %",
                      "texture_class": "USDA class index", "drainage": "0 poor .. 3 well drained"},
            "provenance": {"source": prior,
                           "ph": "SOLUS100 (USDA-NRCS 2024), 0-30 cm, independent NEON check R² 0.81",
                           "organic": "SOLUS100 + SoilGrids ensemble with terrain correction; shown as agronomic classes (low < 2 %, medium 2-4 %, high > 4 % OM)",
                           "texture": "SOLUS100 sand / clay 0-30 cm (silt = remainder), USDA triangle",
                           "resolution": "~10 m (3DEP 1/3 arc-second lattice) interpolated to the 1 m scene grid",
                           "drainage": f"SSURGO drainage class {twi_note}",
                           "not_served": "NPK: no mapped product carries plant-available N / P / K; use a soil test"},
            "grid": {"cell_m": cell, "origin_nw_scene_m": [origin[0], origin[1]], "shape": list(shape)}}
    return lay, meta


def write_simf(path, fields, units, cell, origin_xy, z0=0.0, sidecar=None, zf=None):
    """SIMF v2, one frame, float32. Surface fields [ny, nx] (nz = 1, NaN faces = follow the sidecar's surface), or
    layered fields [nz, ny, nx] with zf = the nz + 1 level faces (here: metres relative to the local ground surface,
    negative downwards; data z-slowest)."""
    names = list(fields)
    shp = fields[names[0]].shape
    nz, (ny, nx) = (1, shp) if len(shp) == 2 else (shp[0], shp[1:])
    hdr = b"SIMF" + struct.pack("<IIIIIII", 2, 1, nx, ny, nz, len(names), 0)
    hdr += struct.pack("<d", cell) + struct.pack("<ddd", origin_xy[0], origin_xy[1], z0) + struct.pack("<d", 1.0)
    for n_ in names:
        hdr += n_.encode()[:8].ljust(8, b"\0") + units[n_].encode()[:8].ljust(8, b"\0")
    faces = [float("nan")] * (nz + 1) if zf is None else [float(z) for z in zf]
    assert len(faces) == nz + 1
    hdr += struct.pack(f"<{nz + 1}f", *faces)
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(struct.pack("<d", 0.0))
        for n_ in names:
            f.write(np.ascontiguousarray(fields[n_], np.float32).tobytes())
    if sidecar is not None:
        from .jsonsafe import nan_free
        json.dump(nan_free(sidecar), open(str(path) + ".json", "w"), indent=1)


DEPTHS_CM = (0, 5, 15, 30, 60, 100, 150)                 # SOLUS100 predicted depths


def solus_depth_points(lon, lat, prop):
    """SOLUS100 prop at points for every predicted depth: {depth_cm: values}, bilinear in the 100 m grid."""
    import rasterio
    from rasterio.windows import from_bounds
    from pyproj import Transformer
    name, scal = SOLUS[prop]
    x, y = Transformer.from_crs(4326, 5070, always_xy=True).transform(lon, lat)
    env = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif"}
    vals = {}
    for dpt in DEPTHS_CM:
        with rasterio.Env(**env), rasterio.open(f"/vsicurl/https://storage.googleapis.com/solus100pub/{name}_{dpt}_cm_p.tif") as r:
            w = from_bounds(x.min() - 200, y.min() - 200, x.max() + 200, y.max() + 200, r.transform).round_offsets().round_lengths()
            a = r.read(1, window=w).astype(np.float64)
            if r.nodata is not None:
                a[a == r.nodata] = np.nan
            a /= scal
            T = r.window_transform(w)
        fc = (x - T.c) / T.a - 0.5; fr = (y - T.f) / T.e - 0.5
        c0 = np.clip(np.floor(fc).astype(int), 0, a.shape[1] - 2); r0 = np.clip(np.floor(fr).astype(int), 0, a.shape[0] - 2)
        tx, ty = np.clip(fc - c0, 0, 1), np.clip(fr - r0, 0, 1)
        vals[dpt] = (a[r0, c0] * (1 - tx) * (1 - ty) + a[r0, c0 + 1] * tx * (1 - ty) + a[r0 + 1, c0] * (1 - tx) * ty +
                     a[r0 + 1, c0 + 1] * tx * ty)
    return vals


def write_soil_depth(site, out_dir, half=56.0, cell=1.0):
    """soil/soil_depth.simf (+ .json): SOLUS100 pH, clay, sand, organic matter (1.724 x SOC) and USDA texture class by
    depth on the same 1 m scene grid as soil.simf. Layers between the predicted depths 0-5, 5-15, 15-30, 30-60, 60-100,
    100-150 cm carry the mean of their two bounding depths (SOLUS's own trapezoid rule); a 150-200 cm layer carries the
    150 cm prediction. Level faces are metres below the local ground surface."""
    from pyproj import Transformer
    X, Y, origin, shape = scene_grid(site, half, cell)
    sx, sy = site.scene_to_site(X, Y)
    E, N = site.frame.to_grid(sx, sy)
    lon, lat = Transformer.from_crs(f"EPSG:{site.frame.epsg}", "EPSG:4326", always_xy=True).transform(E, N)
    lon, lat = np.asarray(lon).ravel(), np.asarray(lat).ravel()
    D = list(DEPTHS_CM)
    lay = {}
    for prop in ("ph", "clay", "sand", "oc"):
        v = solus_depth_points(lon, lat, prop)
        L = [(v[D[k]] + v[D[k + 1]]) / 2 for k in range(len(D) - 1)] + [v[D[-1]]]
        lay[prop] = np.stack(L).reshape(len(L), *shape).astype(np.float32)
    om = 1.724 * lay.pop("oc")
    silt = np.clip(100 - lay["sand"] - lay["clay"], 0, 100)
    tex = np.vectorize(lambda a_, b_, c_: usda_class(a_, b_, c_) if np.isfinite(a_ + b_ + c_) else -1)(lay["sand"], silt, lay["clay"])
    tex = np.where(tex < 0, 255, tex).astype(np.float32)
    fields = {"ph": lay["ph"], "clay": lay["clay"], "sand": lay["sand"], "om": om, "tex": tex}
    units = {"ph": "pH", "clay": "%", "sand": "%", "om": "%", "tex": "class"}
    zf = [0.0, -0.05, -0.15, -0.30, -0.60, -1.00, -1.50, -2.00]
    side = {"what": "SOLUS100 soil properties by depth on the 1 m scene grid (same origin / cell / nx / ny as soil.simf)",
            "layout": "fields [nz][ny][nx], z slowest; rows north-down; ENU scene metres",
            "zf_m": zf, "zf_reference": "metres relative to the local ground surface (negative = below)",
            "layers_cm": [[0, 5], [5, 15], [15, 30], [30, 60], [60, 100], [100, 150], [150, 200]],
            "layer_value": "mean of the SOLUS100 predictions at the layer's two bounding depths (trapezoid); 150-200 cm "
                           "= the 150 cm prediction",
            "fields": {"ph": "pH (1:1 H2O)", "clay": "clay %", "sand": "sand %", "om": "organic matter % = 1.724 x SOC %",
                       "tex": "USDA texture class index (texture_classes), 255 unknown"},
            "texture_classes": USDA,
            "provenance": "SOLUS100 (USDA-NRCS 2024, 100 m, EPSG:5070; gs://solus100pub *_{0,5,15,30,60,100,150}_cm_p.tif), "
                          "bilinear in the 100 m grid at each 1 m cell centre",
            "site_range": {k: [float(np.nanmin(v)), float(np.nanmax(v))] for k, v in fields.items() if k != "tex"}}
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    write_simf(out / "soil_depth.simf", fields, units, cell, origin, sidecar=side, zf=zf)
    return side
