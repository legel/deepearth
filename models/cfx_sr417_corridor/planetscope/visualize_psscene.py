#!/usr/bin/env python3
"""
visualize_psscene.py
====================

Scan a Planet PSScene "analytic_8b_sr_udm2" delivery folder (the kind exported
from Planet Explorer / the Orders API, containing a ``PSScene/`` sub-folder with
an 8-band SuperDove Surface-Reflectance GeoTIFF, a UDM2 quality mask, and STAC /
Planet metadata) and render a set of scientist-friendly, **lossless PNG**
visualizations:

  * True-colour RGB with robust per-channel exposure / white balance
  * Near-infrared (NIR) rendered with the perceptually-uniform `inferno` map
  * A sensible, justified colormap for every remaining band
  * Vegetation / water spectral indices (NDVI, NDRE, NDWI, red-edge NDWIre)
  * Enhanced water detection for dark tannic / turbid water that defeats NDWI:
    a Green/NIR ratio, then a **sub-pixel** open-water boundary — per-pixel
    water-fraction unmixing, marching-squares isoline, and smooth PyTorch
    Fourier lake-edge curves, with exact polygon areas + GeoJSON/shapefile export
  * Optional Sentinel-2 SWIR fusion -> true MNDWI (``--s2-fusion``)
  * A colour-balanced false-colour (NIR-R-G) composite
  * A UDM2 usable-data / quality visualization
  * A single overview contact-sheet figure (with colorbars)

It also writes a ``README.md`` summarising the dataset, the source files, the
derived products, and every hyper-parameter used to make them.

Usage
-----
    python visualize_psscene.py [DATASET_DIR] [-o OUTPUT_DIR]
                                [--low-pct 2] [--high-pct 98] [--gamma 1.8]

``DATASET_DIR`` defaults to the current directory. All inputs are auto-detected,
so pointing it at a folder "exactly like the current location" is enough.

Dependencies: numpy, rasterio, matplotlib, pillow, scipy. The sub-pixel water
boundary additionally uses scikit-image (marching squares), torch (curve fit)
and geopandas/shapely (vector export); if any are missing the script falls back
to the raw threshold mask. Sentinel-2 fusion needs requests.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import numpy as np
import rasterio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# --------------------------------------------------------------------------- #
# SuperDove (PSB.SD) 8-band layout. Center wavelengths per Planet documentation.
# Each entry: (1-based band index, short name, center wavelength nm, colormap,
#              one-line rationale for the colormap choice).
# --------------------------------------------------------------------------- #
SUPERDOVE_BANDS = [
    (1, "Coastal Blue", 443, "cividis", "colour-blind-safe; classic for water/atmosphere/coastal signal"),
    (2, "Blue",         490, "gray",    "honest single-band luminance (a true-colour primary)"),
    (3, "Green I",      531, "gray",    "honest single-band luminance (chlorophyll/peak-green region)"),
    (4, "Green",        565, "gray",    "honest single-band luminance (a true-colour primary)"),
    (5, "Yellow",       610, "gray",    "honest single-band luminance (carotenoid/senescence region)"),
    (6, "Red",          665, "gray",    "honest single-band luminance (a true-colour primary)"),
    (7, "Red Edge",     705, "magma",   "heat-style emphasis of the vegetation red-edge inflection"),
    (8, "NIR",          865, "inferno", "requested infrared rendering; emphasises vegetation/moisture"),
]
# Band indices used for the natural-colour composite (R, G, B).
RGB_BANDS = (6, 4, 2)
# Band indices used for the standard vegetation false-colour composite (NIR, R, G).
FALSE_COLOR_BANDS = (8, 6, 4)

SR_SCALE = 1.0e-4  # PS Surface-Reflectance DN -> reflectance (0..1)


# --------------------------------------------------------------------------- #
# Discovery & metadata
# --------------------------------------------------------------------------- #
def _first(patterns, root):
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(root, pat)))
        if hits:
            return hits[0]
    return None


def discover(dataset_dir):
    """Locate the SR GeoTIFF, UDM2 mask, band XML and item metadata JSON."""
    search_roots = [dataset_dir, os.path.join(dataset_dir, "PSScene")]
    found = {"sr": None, "udm2": None, "xml": None, "item_json": None, "scene_dir": dataset_dir}
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        found["sr"] = found["sr"] or _first(["*AnalyticMS_SR_8b*clip.tif", "*AnalyticMS_SR*clip.tif",
                                              "*AnalyticMS_SR_8b*.tif", "*_SR_*.tif"], root)
        found["udm2"] = found["udm2"] or _first(["*udm2*clip.tif", "*udm2*.tif", "*_DN_udm*.tif"], root)
        found["xml"] = found["xml"] or _first(["*AnalyticMS*metadata*clip.xml", "*metadata*.xml", "*.xml"], root)
        found["item_json"] = found["item_json"] or _first(["*_metadata.json"], root)
        if found["sr"]:
            found["scene_dir"] = root
    return found


def parse_item_metadata(path):
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            doc = json.load(f)
        props = dict(doc.get("properties", {}))
        # `id`/`geometry` live at the top level of the STAC item, not in properties.
        for k in ("id", "type"):
            if k in doc and k not in props:
                props[k] = doc[k]
        return props or doc
    except Exception:
        return {}


def parse_band_xml(path):
    """Return {band_number: {'reflectance_coefficient':..,'radiometric_scale':..}}."""
    coeffs = {}
    if not path or not os.path.exists(path):
        return coeffs
    try:
        tree = ET.parse(path)
        # Namespace-agnostic search.
        for bsm in tree.iter():
            if not bsm.tag.endswith("bandSpecificMetadata"):
                continue
            num = refl = scale = None
            for child in bsm:
                tag = child.tag.split("}")[-1]
                if tag == "bandNumber":
                    num = int(child.text)
                elif tag == "reflectanceCoefficient":
                    refl = float(child.text)
                elif tag == "radiometricScaleFactor":
                    scale = float(child.text)
            if num is not None:
                coeffs[num] = {"reflectance_coefficient": refl, "radiometric_scale": scale}
    except Exception:
        pass
    return coeffs


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def percentile_stretch(arr, valid, low_pct, high_pct, gamma=1.0):
    """Robust linear stretch to [0,1] using percentiles of the valid pixels,
    followed by an optional gamma (exposure) correction."""
    if valid.sum() == 0:
        return np.zeros_like(arr, dtype=np.float32), (0.0, 1.0)
    lo, hi = np.percentile(arr[valid], [low_pct, high_pct])
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    if gamma and gamma != 1.0:
        out = np.power(out, 1.0 / gamma)
    return out, (float(lo), float(hi))


def rgba_from_gray(norm01, valid, cmap_name):
    """Map a normalized [0,1] single band through a matplotlib colormap,
    returning an 8-bit RGBA array with nodata pixels fully transparent."""
    cmap = matplotlib.colormaps[cmap_name]
    rgba = cmap(norm01, bytes=True)            # H x W x 4 uint8
    rgba[..., 3] = np.where(valid, 255, 0).astype(np.uint8)
    return rgba


def save_png(rgba_or_rgb, path):
    """Write a lossless PNG via Pillow (PNG is inherently lossless)."""
    mode = "RGBA" if rgba_or_rgb.shape[-1] == 4 else "RGB"
    Image.fromarray(rgba_or_rgb, mode).save(path, format="PNG", optimize=True)


def index_norm(a, b, valid):
    """Normalized difference index (a-b)/(a+b), masked, in [-1,1]."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = a + b
    out = np.full(a.shape, np.nan, dtype=np.float32)
    ok = valid & (denom != 0)
    out[ok] = (a[ok] - b[ok]) / denom[ok]
    return out


def band_ratio(a, b, valid):
    """Simple band ratio a/b, masked. Ratios amplify low-reflectance contrast
    (e.g. dark tannic water) far better than normalized differences."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    out = np.full(a.shape, np.nan, dtype=np.float32)
    ok = valid & (b > 0)
    out[ok] = a[ok] / b[ok]
    return out


# --------------------------------------------------------------------------- #
# Sub-pixel water boundaries
# ---------------------------------------------------------------------------
# A hard threshold decides each *whole* pixel as water or land, so its boundary
# is pinned to pixel centres and is systematically 1-2 px too conservative
# (shoreline pixels are mixed land+water and get thrown to land). The pipeline
# below instead (1) estimates a continuous water-area *fraction* per pixel by
# linear spectral unmixing on NIR, (2) extracts the shoreline as the sub-pixel
# marching-squares isoline of that fraction, and (3) fits each lake edge with a
# smooth closed PyTorch curve. The result is a genuine sub-pixel, continuous
# boundary, with exact polygon areas and vector exports.
# --------------------------------------------------------------------------- #
def water_fraction_field(refl, valid, nir_max, gnr_min, land_window, nir_water=0.0):
    """Continuous water-area fraction ``f in [0,1]`` via local linear unmixing.

    A shoreline pixel is a mix of water and the dry land behind it, so its NIR
    reflectance lies between the two endmembers:
    ``NIR = f*W + (1-f)*L  ->  f = (L - NIR) / (L - W)``.

      * ``W`` (water endmember) is the global robust median NIR of confident
        interior water -- water is spectrally consistent across the scene.
      * ``L`` (land endmember) is estimated *locally* (mean NIR of nearby land
        pixels in a ``land_window`` box) because land brightness varies wildly
        (grass / urban / road); a global land value would misplace the edge.

    The ``f = level`` isoline therefore sits *inside* the mixed transition zone
    -- the true sub-pixel shoreline. A Green/NIR gate removes spectrally-flat
    dark pixels (shadow, dark roofs) that are dark in NIR but not water, and the
    confident interior is pinned to 1.0 so enclosed holes are filled.

    Returns ``(f, raw_threshold_mask, endmember_info)``.
    """
    from scipy import ndimage as ndi
    green, nir = refl[3], refl[7]
    gnr = np.full(green.shape, np.nan, np.float32)
    ok = valid & (nir > 0)
    gnr[ok] = green[ok] / nir[ok]

    raw = valid & (nir < nir_max) & (gnr > gnr_min)        # bootstrap threshold
    core = ndi.binary_erosion(raw, iterations=2) & valid   # confident interior

    if not nir_water or nir_water <= 0:
        nir_water = float(np.median(nir[core])) if core.any() else 0.06

    land = valid & ~ndi.binary_dilation(raw, iterations=1)
    landf = land.astype(np.float32)
    num = ndi.uniform_filter(nir * landf, size=land_window, mode="nearest")
    den = ndi.uniform_filter(landf, size=land_window, mode="nearest")
    global_land = float(np.median(nir[land])) if land.any() else 0.25
    nir_land = np.where(den > 1e-3, num / np.maximum(den, 1e-6), global_land)
    nir_land = np.maximum(nir_land, nir_water + 0.03)      # keep endmembers apart

    f = np.clip((nir_land - nir) / np.maximum(nir_land - nir_water, 1e-6), 0.0, 1.0)
    f = f.astype(np.float32)
    f[~valid] = 0.0
    f[gnr < gnr_min] = 0.0                                  # green/NIR gate
    f[core] = 1.0                                           # solid interiors (fill holes)
    f = ndi.gaussian_filter(f, 0.6)                         # mild denoise pre-isoline
    return f, raw, {"nir_water": nir_water, "global_land": global_land}


def _shoelace_rc(rc):
    """Signed polygon area in (row, col) index space (for orientation/size)."""
    r, c = rc[:, 0], rc[:, 1]
    return 0.5 * float(np.sum(c[:-1] * r[1:] - c[1:] * r[:-1]))


def extract_water_contours(f, level, min_area_px):
    """Sub-pixel contours of ``f`` at ``level`` via marching squares. ``f`` is
    zero-padded so ponds touching the frame still close into polygons; the pad
    offset is removed. Tiny specks below ``min_area_px`` are dropped."""
    from skimage.measure import find_contours
    fp = np.pad(f, 1, mode="constant", constant_values=0.0)
    out = []
    for c in find_contours(fp, level):
        c = c - 1.0
        if len(c) >= 8 and abs(_shoelace_rc(c)) >= min_area_px:
            out.append(c)
    return out


def fit_fourier_curve(rc, harmonics=0, smooth=0.04, steps=250, device="cpu"):
    """Fit a smooth closed truncated-Fourier curve to contour points ``rc``.

    ``x(t) = a0 + sum_k a_k cos(2*pi*k*t) + b_k sin(2*pi*k*t)`` (same for ``y``),
    with ``t`` the normalised cumulative arc-length. Coefficients are
    FFT-initialised, then refined by Adam (PyTorch) minimising

        sum_i ||curve(t_i) - P_i||^2  +  smooth * sum_k k^2 (|a_k|^2 + |b_k|^2)

    The ``k^2`` (curvature) penalty is the "don't go wild" guardrail; the
    harmonic count scales with perimeter so small ponds stay smooth. Returns a
    densely-sampled smooth closed polygon ``(row, col)`` and the harmonic count.
    """
    import torch
    P = np.asarray(rc, np.float64)
    if np.allclose(P[0], P[-1]):
        P = P[:-1]
    M = len(P)
    seg = np.sqrt((np.diff(np.vstack([P, P[:1]]), axis=0) ** 2).sum(1))
    perim = float(seg.sum())
    t = np.concatenate([[0.0], np.cumsum(seg)[:-1]]) / max(perim, 1e-9)

    K = harmonics or int(np.clip(round(perim / 6.0), 6, 40))
    K = max(2, min(K, M // 2))

    N = max(256, 4 * K)                                    # FFT init on uniform resample
    tu = np.linspace(0, 1, N, endpoint=False)
    xs = np.interp(tu, np.append(t, 1.0), np.append(P[:, 1], P[0, 1]))
    ys = np.interp(tu, np.append(t, 1.0), np.append(P[:, 0], P[0, 0]))

    def coeffs(sig):
        ak = [2.0 / N * (sig * np.cos(2 * np.pi * k * tu)).sum() for k in range(1, K + 1)]
        bk = [2.0 / N * (sig * np.sin(2 * np.pi * k * tu)).sum() for k in range(1, K + 1)]
        return sig.mean(), np.array(ak), np.array(bk)

    x0, xa, xb = coeffs(xs)
    y0, ya, yb = coeffs(ys)

    dev = torch.device(device)
    tt = torch.tensor(t, dtype=torch.float64, device=dev)
    Px = torch.tensor(P[:, 1], dtype=torch.float64, device=dev)
    Py = torch.tensor(P[:, 0], dtype=torch.float64, device=dev)
    kk = torch.arange(1, K + 1, dtype=torch.float64, device=dev)
    ang = 2 * np.pi * tt[:, None] * kk[None, :]
    cosA, sinA = torch.cos(ang), torch.sin(ang)
    k2 = kk ** 2

    par = {n: torch.tensor(v, dtype=torch.float64, device=dev, requires_grad=True)
           for n, v in dict(x0=x0, xa=xa, xb=xb, y0=y0, ya=ya, yb=yb).items()}
    opt = torch.optim.Adam(par.values(), lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        xfit = par["x0"] + (cosA * par["xa"] + sinA * par["xb"]).sum(1)
        yfit = par["y0"] + (cosA * par["ya"] + sinA * par["yb"]).sum(1)
        data = ((xfit - Px) ** 2 + (yfit - Py) ** 2).mean()
        curv = smooth * (k2 * (par["xa"] ** 2 + par["xb"] ** 2
                               + par["ya"] ** 2 + par["yb"] ** 2)).sum() / max(perim, 1.0)
        (data + curv).backward()
        opt.step()

    S = max(200, 6 * K)
    ts = torch.linspace(0, 1, S, dtype=torch.float64, device=dev)
    ang = 2 * np.pi * ts[:, None] * kk[None, :]
    with torch.no_grad():
        xf = (par["x0"] + (torch.cos(ang) * par["xa"] + torch.sin(ang) * par["xb"]).sum(1)).cpu().numpy()
        yf = (par["y0"] + (torch.cos(ang) * par["ya"] + torch.sin(ang) * par["yb"]).sum(1)).cpu().numpy()
    poly = np.column_stack([yf, xf])
    return np.vstack([poly, poly[:1]]), K


def fit_water_boundaries(f, level, harmonics, smooth, min_area_px):
    """Extract + fit every lake edge. Returns a list of dicts with the fitted
    polygon (row,col, closed), whether it bounds water or an interior hole
    (classified by ``f`` at the polygon centroid), and the index-space area."""
    H, W = f.shape
    bodies = []
    for c in extract_water_contours(f, level, min_area_px):
        poly, K = fit_fourier_curve(c, harmonics=harmonics, smooth=smooth)
        rr = int(np.clip(round(poly[:, 0].mean()), 0, H - 1))
        cc = int(np.clip(round(poly[:, 1].mean()), 0, W - 1))
        bodies.append({"poly": poly, "is_water": bool(f[rr, cc] >= level),
                       "area_px": abs(_shoelace_rc(poly)), "harmonics": K})
    return bodies


def rasterize_water(bodies, H, W, supersample):
    """Fill fitted polygons on an ``supersample``x grid (water first, holes
    subtracted), returning the high-res boolean and the native-resolution
    fractional water coverage (block-mean -> true sub-pixel per-pixel area)."""
    from PIL import Image, ImageDraw
    S = int(supersample)
    img = Image.new("1", (W * S, H * S), 0)
    dr = ImageDraw.Draw(img)
    for b in sorted(bodies, key=lambda d: -d["area_px"]):   # large first
        pts = [(c * S, r * S) for r, c in b["poly"]]
        dr.polygon(pts, fill=1 if b["is_water"] else 0)
    hi = np.array(img, dtype=np.uint8)
    cov = hi.reshape(H, S, W, S).mean(axis=(1, 3)).astype(np.float32)
    return hi, cov


def water_area_m2(bodies, transform):
    """Exact (sub-pixel) water area in m^2 via the shoelace formula on the
    fitted polygons in projected CRS coordinates (water +, holes -)."""
    from rasterio.transform import xy as rio_xy
    total = 0.0
    for b in bodies:
        xy = np.array([rio_xy(transform, r, c, offset="center") for r, c in b["poly"]])
        x, y = xy[:, 0], xy[:, 1]
        a = 0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
        total += a if b["is_water"] else -a
    return total


def export_water_vectors(bodies, transform, crs, out_dir, basename="water_boundaries"):
    """Write the fitted shorelines as polygons: a shapefile + GeoJSON in the
    scene CRS, plus a WGS84 GeoJSON. Outer water rings carry their interior
    holes. Returns the list of written paths, or [] if deps are missing."""
    try:
        import geopandas as gpd
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
    except ImportError:
        return []
    from rasterio.transform import xy as rio_xy

    def to_xy(poly):
        return [tuple(rio_xy(transform, r, c, offset="center")) for r, c in poly]

    water = [Polygon(to_xy(b["poly"])).buffer(0) for b in bodies if b["is_water"]]
    holes = [Polygon(to_xy(b["poly"])).buffer(0) for b in bodies if not b["is_water"]]
    geom = unary_union(water)
    if holes:
        geom = geom.difference(unary_union(holes))
    polys = list(getattr(geom, "geoms", [geom]))
    gdf = gpd.GeoDataFrame(
        {"id": range(1, len(polys) + 1),
         "area_ha": [p.area / 1e4 for p in polys]},
        geometry=polys, crs=crs)

    paths = []
    shp = os.path.join(out_dir, f"{basename}.shp")
    gdf.to_file(shp)
    paths.append(shp)
    gj = os.path.join(out_dir, f"{basename}.geojson")
    gdf.to_file(gj, driver="GeoJSON")
    paths.append(gj)
    gj84 = os.path.join(out_dir, f"{basename}_wgs84.geojson")
    gdf.to_crs("EPSG:4326").to_file(gj84, driver="GeoJSON")
    paths.append(gj84)
    return paths


def _overlay_water(rgb, cov, bodies, supersample):
    """Composite a translucent fractional-coverage fill plus a crisp, anti-aliased
    fitted shoreline onto the true-colour image. Returns an 8-bit RGBA array."""
    from PIL import Image, ImageDraw
    H, W = cov.shape
    base = rgb[..., :3].astype(np.float32)
    fill = np.array([0, 200, 255], np.float32)
    a = (cov * 0.45)[..., None]
    out = base * (1 - a) + fill * a

    if bodies:
        S = int(supersample)
        line = Image.new("L", (W * S, H * S), 0)
        dr = ImageDraw.Draw(line)
        for b in bodies:
            dr.line([(c * S, r * S) for r, c in b["poly"]],
                    fill=255, width=max(2, S // 2))
        edge = (np.array(line, np.float32).reshape(H, S, W, S).mean(axis=(1, 3)) / 255.0)[..., None]
        edge_c = np.array([0, 255, 255], np.float32)
        out = out * (1 - edge) + edge_c * edge

    out = np.clip(out, 0, 255).astype(np.uint8)
    return np.dstack([out, rgb[..., 3]])


def save_index_pair(arr, short, formula, meaning, cmap_name, out_dir, vmin=-1, vmax=1):
    """Save a masked index as both a pixel-exact PNG and a colorbar figure."""
    cmap = matplotlib.colormaps[cmap_name].copy()
    cmap.set_bad((0, 0, 0, 0))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    masked = np.ma.masked_invalid(arr)
    save_png(cmap(norm(masked), bytes=True), os.path.join(out_dir, f"{short.lower()}.png"))
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f"{short} = {formula}\n{meaning}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=short)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{short.lower()}_annotated.png"), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir", nargs="?", default=".", help="PSScene delivery folder (default: cwd)")
    ap.add_argument("-o", "--output", default=None, help="output dir (default: <dataset_dir>/derived)")
    ap.add_argument("--low-pct", type=float, default=2.0, help="lower percentile for stretch (default 2)")
    ap.add_argument("--high-pct", type=float, default=98.0, help="upper percentile for stretch (default 98)")
    ap.add_argument("--gamma", type=float, default=1.8, help="display gamma for RGB/false-colour (default 1.8)")
    ap.add_argument("--s2-fusion", action="store_true",
                    help="fetch a same-window Sentinel-2 scene and compute true MNDWI (SWIR). Needs network.")
    ap.add_argument("--s2-max-cloud", type=float, default=20.0,
                    help="max Sentinel-2 scene cloud cover %% for fusion (default 20)")
    ap.add_argument("--s2-day-window", type=int, default=15,
                    help="+/- days around acquisition to search Sentinel-2 (default 15)")
    ap.add_argument("--water-nir-max", type=float, default=0.16,
                    help="max NIR surface reflectance for a water pixel (default 0.16)")
    ap.add_argument("--water-gnr-min", type=float, default=0.48,
                    help="min Green/NIR ratio for a water pixel (default 0.48)")
    ap.add_argument("--water-frac-level", type=float, default=0.40,
                    help="water-area fraction isoline for the shoreline (default 0.40). "
                         "Lower = less conservative / higher recall (0.35 adds marginal "
                         "canals & shallow ponds); 0.50 = the strict 50%%-water line.")
    ap.add_argument("--water-land-window", type=int, default=25,
                    help="box size (px) for the local land NIR endmember in unmixing (default 25)")
    ap.add_argument("--water-nir-water", type=float, default=0.0,
                    help="water NIR endmember reflectance (default 0 = auto from interior water)")
    ap.add_argument("--water-supersample", type=int, default=8,
                    help="rasterization oversampling for the sub-pixel mask/area (default 8)")
    ap.add_argument("--water-curve-harmonics", type=int, default=0,
                    help="Fourier harmonics per lake-edge curve (default 0 = auto from perimeter)")
    ap.add_argument("--water-curve-smooth", type=float, default=0.04,
                    help="curvature penalty for the PyTorch edge fit; higher = smoother (default 0.04)")
    ap.add_argument("--water-min-area-px", type=float, default=4.0,
                    help="drop water bodies smaller than this many pixels (default 4)")
    args = ap.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    out_dir = os.path.abspath(args.output) if args.output else os.path.join(dataset_dir, "derived")
    bands_dir = os.path.join(out_dir, "bands")
    os.makedirs(bands_dir, exist_ok=True)

    files = discover(dataset_dir)
    if not files["sr"]:
        sys.exit(f"ERROR: no Surface-Reflectance GeoTIFF found under {dataset_dir!r}")

    print(f"[scan]  dataset : {dataset_dir}")
    print(f"[scan]  SR      : {os.path.relpath(files['sr'], dataset_dir)}")
    print(f"[scan]  UDM2    : {os.path.relpath(files['udm2'], dataset_dir) if files['udm2'] else 'none'}")
    print(f"[scan]  XML     : {os.path.relpath(files['xml'], dataset_dir) if files['xml'] else 'none'}")
    print(f"[out]   writing : {out_dir}")

    item_props = parse_item_metadata(files["item_json"])
    band_coeffs = parse_band_xml(files["xml"])

    # ---- Read SR cube as reflectance ----
    with rasterio.open(files["sr"]) as ds:
        raw = ds.read().astype(np.float32)            # (bands, H, W) DN
        profile = {
            "width": ds.width, "height": ds.height, "count": ds.count,
            "dtype": ds.dtypes[0], "crs": str(ds.crs), "nodata": ds.nodata,
            "res": ds.res, "bounds": tuple(round(b, 2) for b in ds.bounds),
        }
        nodata = ds.nodata if ds.nodata is not None else 0
    nbands = raw.shape[0]
    refl = raw * SR_SCALE                              # reflectance 0..~0.65
    # Valid = pixel is non-nodata across all bands (clipped corners are 0).
    valid = np.all(raw != nodata, axis=0)

    band_meta = [b for b in SUPERDOVE_BANDS if b[0] <= nbands]
    by_index = {b[0]: b for b in band_meta}

    outputs = []  # (filename, human description) for the README

    def reflectance(idx):
        return refl[idx - 1]

    # ----------------------------------------------------------------- #
    # 1. True-colour RGB with per-channel exposure / white balance
    # ----------------------------------------------------------------- #
    rgb = np.zeros((profile["height"], profile["width"], 4), dtype=np.uint8)
    rgb_stretch = {}
    for ch, b in enumerate(RGB_BANDS):
        norm, lohi = percentile_stretch(reflectance(b), valid, args.low_pct, args.high_pct, args.gamma)
        rgb[..., ch] = (norm * 255).round().astype(np.uint8)
        rgb_stretch[by_index[b][1]] = lohi
    rgb[..., 3] = np.where(valid, 255, 0).astype(np.uint8)
    save_png(rgb, os.path.join(out_dir, "rgb_truecolor.png"))
    outputs.append(("rgb_truecolor.png",
                    f"Natural-colour composite (R={by_index[6][1]}, G={by_index[4][1]}, B={by_index[2][1]}); "
                    f"independent per-channel {args.low_pct:.0f}–{args.high_pct:.0f}% stretch (gray-world white "
                    f"balance) + gamma {args.gamma}."))
    print("[ok]    rgb_truecolor.png")

    # ----------------------------------------------------------------- #
    # 2. False-colour NIR composite (NIR-R-G) -- vegetation pops red
    # ----------------------------------------------------------------- #
    if nbands >= 8:
        fc = np.zeros_like(rgb)
        for ch, b in enumerate(FALSE_COLOR_BANDS):
            norm, _ = percentile_stretch(reflectance(b), valid, args.low_pct, args.high_pct, args.gamma)
            fc[..., ch] = (norm * 255).round().astype(np.uint8)
        fc[..., 3] = rgb[..., 3]
        save_png(fc, os.path.join(out_dir, "false_color_nir.png"))
        outputs.append(("false_color_nir.png",
                        f"Colour-IR composite (R={by_index[8][1]}, G={by_index[6][1]}, B={by_index[4][1]}); "
                        f"healthy vegetation appears bright red. Same stretch as RGB."))
        print("[ok]    false_color_nir.png")

    # ----------------------------------------------------------------- #
    # 3. NIR -> inferno, plus a colormap for every remaining band
    # ----------------------------------------------------------------- #
    band_stretch = {}
    for idx, name, wl, cmap_name, _why in band_meta:
        norm, lohi = percentile_stretch(reflectance(idx), valid, args.low_pct, args.high_pct, gamma=1.0)
        rgba = rgba_from_gray(norm, valid, cmap_name)
        slug = name.lower().replace(" ", "_")
        fname = f"band{idx}_{slug}_{cmap_name}.png"
        save_png(rgba, os.path.join(bands_dir, fname))
        band_stretch[idx] = lohi
        if idx == 8:  # surface the headline NIR/inferno product at top level too
            save_png(rgba, os.path.join(out_dir, "nir_inferno.png"))
            outputs.append(("nir_inferno.png",
                            f"Band 8 NIR (~{wl} nm) with the perceptually-uniform `inferno` colormap; "
                            f"reflectance stretched {args.low_pct:.0f}–{args.high_pct:.0f}%."))
        print(f"[ok]    bands/{fname}")
    outputs.append(("bands/",
                    "Per-band single-band renderings; colormap chosen per band "
                    "(`inferno` for NIR, `magma` for red-edge, `cividis` for coastal blue, "
                    "grayscale luminance for true-colour primaries)."))

    # ----------------------------------------------------------------- #
    # 4. Spectral indices (NDVI / NDRE / NDWI) with colorbars
    # ----------------------------------------------------------------- #
    index_specs = []
    if nbands >= 8:
        index_specs.append(("NDVI", "(NIR-Red)/(NIR+Red)", 8, 6, "RdYlGn", "vegetation vigour / greenness"))
        index_specs.append(("NDRE", "(NIR-RedEdge)/(NIR+RedEdge)", 8, 7, "RdYlGn", "canopy chlorophyll / N status"))
        index_specs.append(("NDWI", "(Green-NIR)/(Green+NIR)", 4, 8, "RdBu", "open water (McFeeters)"))
        index_specs.append(("NDWIre", "(RedEdge-NIR)/(RedEdge+NIR)", 7, 8, "RdBu",
                            "red-edge water variant; separates dark water from shadow/wet vegetation"))

    for short, formula, a_idx, b_idx, cmap_name, meaning in index_specs:
        arr = index_norm(reflectance(a_idx), reflectance(b_idx), valid)
        save_index_pair(arr, short, formula, meaning, cmap_name, out_dir, vmin=-1, vmax=1)
        med = float(np.nanmedian(arr)) if np.isfinite(arr).any() else float("nan")
        outputs.append((f"{short.lower()}.png / {short.lower()}_annotated.png",
                        f"{short} = {formula} ({meaning}); range [-1,1], `{cmap_name}`. "
                        f"Scene median {med:+.3f}."))
        print(f"[ok]    {short.lower()}.png  (median {med:+.3f})")

    # ----------------------------------------------------------------- #
    # 4b. Enhanced water detection (for dark tannic / turbid water that
    #     defeats McFeeters NDWI): Green/NIR ratio + Otsu multi-band mask.
    # ----------------------------------------------------------------- #
    water_summary = None
    if nbands >= 8:
        green, nir = reflectance(4), reflectance(8)
        # (i) Green/NIR ratio -- ratios amplify dark-water contrast.
        gnr = band_ratio(green, nir, valid)
        finite = gnr[np.isfinite(gnr)]
        hi = float(np.percentile(finite, 98)) if finite.size else 3.0
        gnr_disp = np.clip(gnr / max(hi, 1e-6), 0, 1)
        gnr_disp = np.where(np.isfinite(gnr), gnr_disp, np.nan)
        save_index_pair(gnr_disp, "GreenNIRratio", "Green / NIR",
                        "water-favourable band ratio (high = water; robust to tannic staining)",
                        "YlGnBu", out_dir, vmin=0, vmax=1)
        outputs.append(("greennirratio.png / greennirratio_annotated.png",
                        "Green/NIR band ratio (`YlGnBu`, scaled to its 98th pct); amplifies "
                        "dark tannic-water contrast that normalized NDWI loses."))
        print("[ok]    greennirratio.png")

        # (ii) Sub-pixel open-water boundaries. A hard threshold (low NIR + high
        #      Green/NIR) isolates the turbid/tannic ponds that defeat NDWI>0 and
        #      Otsu, but pins the shoreline to pixel centres and runs 1-2 px tight.
        #      Instead we unmix each pixel's water-area *fraction* (local NIR
        #      endmembers), trace the sub-pixel f=level isoline, and fit each lake
        #      edge with a smooth PyTorch Fourier curve -> a genuine continuous,
        #      sub-pixel shoreline with exact polygon areas and vector exports.
        px_area = abs(profile["res"][0] * profile["res"][1])
        nvalid = max(int(valid.sum()), 1)
        try:
            f_field, water_raw, em = water_fraction_field(
                refl, valid, args.water_nir_max, args.water_gnr_min,
                args.water_land_window, args.water_nir_water)
            bodies = fit_water_boundaries(f_field, args.water_frac_level,
                                          args.water_curve_harmonics,
                                          args.water_curve_smooth, args.water_min_area_px)
            hi, cov = rasterize_water(bodies, profile["height"], profile["width"],
                                      args.water_supersample)
            subpixel = True
        except ImportError as e:
            print(f"[warn]  sub-pixel deps missing ({e}); falling back to raw threshold mask")
            water_raw = valid & (nir < args.water_nir_max) & (gnr > args.water_gnr_min)
            bodies, em = [], {"nir_water": args.water_nir_max, "global_land": float("nan")}
            f_field = water_raw.astype(np.float32)
            cov = water_raw.astype(np.float32)
            subpixel = False

        n_water = sum(b["is_water"] for b in bodies)
        n_holes = sum(not b["is_water"] for b in bodies)

        # Continuous water-fraction field (the sub-pixel evidence layer).
        save_index_pair(np.where(valid, f_field, np.nan), "water_fraction",
                        "linear-unmixed water-area fraction",
                        "per-pixel water fraction (0=land, 1=water); the f=%.2f isoline "
                        "is the shoreline" % args.water_frac_level,
                        "Blues", out_dir, vmin=0, vmax=1)
        outputs.append(("water_fraction.png / water_fraction_annotated.png",
                        "Continuous water-area fraction from NIR linear unmixing (`Blues`); "
                        f"the sub-pixel shoreline is its f={args.water_frac_level:g} isoline."))

        # Categorical mask, antialiased by the native fractional coverage.
        land_c = np.array([210, 210, 205], np.float32)
        water_c = np.array([28, 90, 175], np.float32)
        blend = (cov[..., None] * water_c + (1 - cov[..., None]) * land_c).round().astype(np.uint8)
        wm = np.dstack([blend, np.where(valid, 255, 0).astype(np.uint8)])
        save_png(wm, os.path.join(out_dir, "water_mask.png"))

        # True-colour overlay: translucent fractional fill + crisp fitted edge.
        overlay = _overlay_water(rgb, cov, bodies, args.water_supersample)
        save_png(overlay, os.path.join(out_dir, "water_mask_overlay.png"))

        # High-resolution crisp mask for sub-pixel zoom inspection (4x native).
        if subpixel:
            from PIL import Image as _I
            S = int(args.water_supersample)
            hi_img = _I.fromarray((hi * 255).astype(np.uint8))
            hi_img = hi_img.resize((profile["width"] * 4, profile["height"] * 4), _I.NEAREST)
            hi_img.save(os.path.join(out_dir, "water_mask_hires.png"))
            outputs.append(("water_mask_hires.png",
                            "Crisp 4x-supersampled sub-pixel water mask (for zoom inspection)."))

        # Exact sub-pixel area via polygon shoelace in projected CRS.
        if subpixel and bodies:
            with rasterio.open(files["sr"]) as _ds:
                transform = _ds.transform
            area_m2 = water_area_m2(bodies, transform)
            vec_paths = export_water_vectors(bodies, transform, profile["crs"], out_dir)
        else:
            area_m2 = water_raw.sum() * px_area
            transform, vec_paths = None, []

        water_summary = {
            "water_pct": 100.0 * area_m2 / (nvalid * px_area),
            "water_ha": area_m2 / 1.0e4,
            "raw_pct": 100.0 * water_raw.sum() / nvalid,
            "raw_ha": water_raw.sum() * px_area / 1.0e4,
            "n_bodies": n_water,
            "n_holes": n_holes,
            "subpixel": subpixel,
            "params": {"frac_level": args.water_frac_level,
                       "nir_water": em["nir_water"],
                       "nir_max": args.water_nir_max,
                       "gnr_min": args.water_gnr_min,
                       "land_window": args.water_land_window,
                       "supersample": args.water_supersample,
                       "curve_harmonics": args.water_curve_harmonics,
                       "curve_smooth": args.water_curve_smooth},
        }
        outputs.append(("water_mask.png",
                        "Open-water mask, antialiased by per-pixel water-area fraction "
                        f"(sub-pixel f={args.water_frac_level:g} shoreline)."))
        outputs.append(("water_mask_overlay.png",
                        "Sub-pixel shoreline (cyan) + translucent fractional fill on true colour."))
        for p in vec_paths:
            outputs.append((os.path.basename(p),
                            "Fitted sub-pixel shoreline polygons "
                            + ("(WGS84)" if p.endswith("wgs84.geojson")
                               else f"({profile['crs']})") + "."))
        print(f"[ok]    water_mask.png  ({water_summary['water_pct']:.1f}% of scene, "
              f"{water_summary['water_ha']:.1f} ha; raw threshold {water_summary['raw_ha']:.1f} ha; "
              f"{n_water} bodies, {n_holes} holes; "
              f"{'sub-pixel f=%.2f' % args.water_frac_level if subpixel else 'threshold fallback'})")

    # ----------------------------------------------------------------- #
    # 4c. Optional Sentinel-2 MNDWI fusion (true SWIR-based water index).
    # ----------------------------------------------------------------- #
    if args.s2_fusion:
        try:
            info = sentinel2_mndwi_fusion(files["sr"], out_dir,
                                          item_props.get("acquired", ""),
                                          args.s2_max_cloud, args.s2_day_window)
            if info:
                outputs.append(("mndwi_sentinel2.png / mndwi_sentinel2_annotated.png",
                                f"True MNDWI = (Green-SWIR)/(Green+SWIR) from Sentinel-2 "
                                f"{info['id']} ({info['datetime'][:10]}, cloud "
                                f"{info['cloud']:.0f}%), warped to the PlanetScope grid. "
                                f"SWIR makes this decisive for tannic/turbid water."))
                print(f"[ok]    mndwi_sentinel2.png  (from S2 {info['id']})")
        except Exception as e:
            print(f"[warn]  Sentinel-2 fusion skipped: {e}")

    # ----------------------------------------------------------------- #
    # 5. UDM2 usable-data / quality visualization
    # ----------------------------------------------------------------- #
    udm2_summary = None
    if files["udm2"]:
        # UDM2 band order: 1 clear, 2 snow, 3 shadow, 4 light haze,
        # 5 heavy haze, 6 cloud, 7 confidence %, 8 unusable mask.
        with rasterio.open(files["udm2"]) as ds:
            udm = ds.read()
        clear, snow, shadow, lhaze, hhaze, cloud = (udm[i] for i in range(6))
        conf = udm[6]
        in_scene = valid  # footprint
        # Categorical class map (priority: cloud > haze > shadow > snow > clear).
        classes = {
            "clear":      ((0.20, 0.65, 0.20), clear == 1),
            "shadow":     ((0.30, 0.30, 0.35), shadow == 1),
            "snow/ice":   ((0.60, 0.85, 0.95), snow == 1),
            "light haze": ((0.95, 0.90, 0.55), lhaze == 1),
            "heavy haze": ((0.90, 0.70, 0.30), hhaze == 1),
            "cloud":      ((0.97, 0.97, 0.97), cloud == 1),
        }
        cat = np.zeros((udm.shape[1], udm.shape[2], 4), dtype=np.uint8)
        for (_name, (rgbc, mask)) in classes.items():
            m = mask & in_scene
            for c in range(3):
                cat[..., c][m] = int(rgbc[c] * 255)
            cat[..., 3][m] = 255
        cat[..., 3][~in_scene] = 0
        save_png(cat, os.path.join(out_dir, "udm2_class.png"))

        # Confidence raster (viridis 0-100%).
        conf_norm = np.clip(conf.astype(np.float32) / 100.0, 0, 1)
        save_png(rgba_from_gray(conf_norm, in_scene, "viridis"),
                 os.path.join(out_dir, "udm2_confidence.png"))

        npix = int(in_scene.sum())
        udm2_summary = {
            "clear_pct": 100.0 * (clear[in_scene] == 1).mean(),
            "cloud_pct": 100.0 * (cloud[in_scene] == 1).mean(),
            "haze_pct": 100.0 * ((lhaze[in_scene] == 1) | (hhaze[in_scene] == 1)).mean(),
            "shadow_pct": 100.0 * (shadow[in_scene] == 1).mean(),
            "mean_conf": float(conf[in_scene].mean()),
            "valid_pixels": npix,
        }
        outputs.append(("udm2_class.png",
                        "UDM2 categorical quality map (clear/shadow/snow/haze/cloud)."))
        outputs.append(("udm2_confidence.png",
                        "UDM2 per-pixel usable-data confidence (0–100%), `viridis`."))
        print(f"[ok]    udm2_class.png / udm2_confidence.png  "
              f"(clear {udm2_summary['clear_pct']:.1f}%)")

    # ----------------------------------------------------------------- #
    # 6. Overview contact sheet
    # ----------------------------------------------------------------- #
    panels = [("True colour (R-G-B)", os.path.join(out_dir, "rgb_truecolor.png"))]
    if os.path.exists(os.path.join(out_dir, "false_color_nir.png")):
        panels.append(("False colour (NIR-R-G)", os.path.join(out_dir, "false_color_nir.png")))
    panels.append(("NIR (inferno)", os.path.join(out_dir, "nir_inferno.png")))
    if os.path.exists(os.path.join(out_dir, "ndvi.png")):
        panels.append(("NDVI (RdYlGn)", os.path.join(out_dir, "ndvi.png")))
    if os.path.exists(os.path.join(out_dir, "water_fraction.png")):
        panels.append(("Water fraction (unmixed)", os.path.join(out_dir, "water_fraction.png")))
    if os.path.exists(os.path.join(out_dir, "water_mask_overlay.png")):
        panels.append(("Sub-pixel water mask", os.path.join(out_dir, "water_mask_overlay.png")))
    ncol = 2
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 5.5 * nrow))
    axes = np.atleast_1d(axes).ravel()
    scene_id = item_props.get("id") or os.path.basename(os.path.dirname(files["sr"]))
    acq = item_props.get("acquired", "")
    fig.suptitle(f"PlanetScope SuperDove  |  {scene_id}  |  acquired {acq}", fontsize=13)
    for ax, (title, p) in zip(axes, panels):
        ax.imshow(Image.open(p))
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(out_dir, "overview.png"), dpi=140)
    plt.close(fig)
    outputs.append(("overview.png", "Single-page contact sheet of the headline products."))
    print("[ok]    overview.png")

    # ----------------------------------------------------------------- #
    # 7. README.md
    # ----------------------------------------------------------------- #
    readme_path = os.path.join(dataset_dir, "README.md")
    write_readme(readme_path, out_dir, dataset_dir, files, profile, item_props, band_coeffs,
                 band_meta, band_stretch, rgb_stretch, udm2_summary, water_summary, outputs, args)
    print(f"[ok]    {os.path.relpath(readme_path, dataset_dir)}\n"
          f"[done]  {len(outputs)} product groups -> {out_dir}")


# --------------------------------------------------------------------------- #
# Sentinel-2 MNDWI fusion (true SWIR water index; SuperDove has no SWIR band)
# --------------------------------------------------------------------------- #
PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
PC_SIGN = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"


def sentinel2_mndwi_fusion(ps_path, out_dir, acquired_iso, max_cloud, day_window):
    """Query Microsoft Planetary Computer for a Sentinel-2 L2A scene overlapping
    the PlanetScope footprint and date, compute MNDWI=(Green-SWIR)/(Green+SWIR),
    and warp it onto the PlanetScope grid. Free, no credentials (public SAS
    signing). Returns a small info dict or None.

    MNDWI is the gold-standard open-water index precisely because SWIR (~1.6 um)
    is absorbed near-totally by water regardless of CDOM/tannins or sediment --
    exactly the Central-Florida cases that defeat the green/NIR NDWI.
    """
    import requests
    from datetime import timedelta
    from rasterio.warp import reproject, Resampling, transform_bounds

    with rasterio.open(ps_path) as ps:
        dst_crs, dst_transform = ps.crs, ps.transform
        dst_h, dst_w, bounds = ps.height, ps.width, ps.bounds
    west, south, east, north = transform_bounds(dst_crs, "EPSG:4326", *bounds)

    try:
        t0 = datetime.fromisoformat(acquired_iso.replace("Z", "+00:00"))
    except Exception:
        raise RuntimeError(f"unparseable acquisition date {acquired_iso!r}")
    start = (t0 - timedelta(days=day_window)).strftime("%Y-%m-%d")
    end = (t0 + timedelta(days=day_window)).strftime("%Y-%m-%d")

    body = {
        "collections": ["sentinel-2-l2a"],
        "bbox": [west, south, east, north],
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "limit": 50,
    }
    r = requests.post(PC_STAC, json=body, timeout=60)
    r.raise_for_status()
    feats = r.json().get("features", [])
    if not feats:
        raise RuntimeError(f"no Sentinel-2 L2A scene <{max_cloud:g}% cloud within "
                           f"+/-{day_window}d of {start}..{end}")
    item = min(feats, key=lambda f: f["properties"].get("eo:cloud_cover", 100))

    def signed(href):
        s = requests.get(PC_SIGN, params={"href": href}, timeout=60)
        s.raise_for_status()
        return s.json()["href"]

    def warp_asset(key):
        href = signed(item["assets"][key]["href"])
        dst = np.zeros((dst_h, dst_w), dtype=np.float32)
        with rasterio.open(href) as src:
            reproject(source=rasterio.band(src, 1), destination=dst,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=dst_transform, dst_crs=dst_crs,
                      resampling=Resampling.bilinear)
        return dst

    green = warp_asset("B03")   # 10 m green
    swir = warp_asset("B11")    # 20 m SWIR-1 (~1610 nm), resampled to PS grid
    denom = green + swir
    mndwi = np.full(green.shape, np.nan, dtype=np.float32)
    ok = denom > 0
    mndwi[ok] = (green[ok] - swir[ok]) / denom[ok]

    # Saves mndwi_sentinel2.png and mndwi_sentinel2_annotated.png.
    save_index_pair(mndwi, "mndwi_sentinel2", "(Green - SWIR) / (Green + SWIR)",
                    f"Sentinel-2 {item['id']} -- SWIR-based open water (water > 0)",
                    "RdBu", out_dir, vmin=-1, vmax=1)
    return {"id": item["id"], "datetime": item["properties"].get("datetime", ""),
            "cloud": float(item["properties"].get("eo:cloud_cover", float("nan")))}


def write_readme(readme_path, out_dir, dataset_dir, files, profile, item_props, band_coeffs,
                 band_meta, band_stretch, rgb_stretch, udm2_summary, water_summary, outputs, args):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rel = lambda p: os.path.relpath(p, dataset_dir) if p else "—"

    def prop(k, default="—"):
        v = item_props.get(k, default)
        return v if v not in (None, "") else default

    bounds = profile["bounds"]
    lines = []
    A = lines.append
    A(f"# PlanetScope SuperDove — Derived Visualizations\n")
    A(f"_Scene `{prop('id')}` · generated {now} by `visualize_psscene.py`_\n")
    A("## 1. Dataset summary\n")
    A("Planet **PSScene** delivery, bundle `analytic_8b_sr_udm2`: an 8-band "
      "SuperDove (PSB.SD) image as **bottom-of-atmosphere Surface Reflectance** "
      "with an accompanying UDM2 usable-data mask.\n")
    A("| Property | Value |")
    A("|---|---|")
    A(f"| Item ID | `{prop('id')}` |")
    A(f"| Instrument / satellite | {prop('instrument')} / {prop('satellite_id')} |")
    A(f"| Acquired (UTC) | {prop('acquired')} |")
    A(f"| Ground sample distance | {prop('gsd')} m (pixel grid {profile['res'][0]:g} m) |")
    A(f"| Raster size | {profile['width']} × {profile['height']} px, {profile['count']} bands, {profile['dtype']} |")
    A(f"| CRS | {profile['crs']} |")
    A(f"| Bounds (CRS units) | {bounds} |")
    A(f"| Sun elev / azim | {prop('sun_elevation')}° / {prop('sun_azimuth')}° |")
    A(f"| View angle | {prop('view_angle')}° |")
    A(f"| Cloud / clear | {prop('cloud_percent')}% / {prop('clear_percent')}% |")
    A(f"| Quality category | {prop('quality_category')} |")
    if udm2_summary:
        A(f"| UDM2 clear (in-footprint) | {udm2_summary['clear_pct']:.1f}% |")
        A(f"| UDM2 mean confidence | {udm2_summary['mean_conf']:.1f}% |")
    if water_summary:
        A(f"| Open water (sub-pixel) | {water_summary['water_pct']:.1f}% "
          f"({water_summary['water_ha']:.1f} ha; {water_summary['n_bodies']} bodies) |")
    A("")

    A("## 2. Original data & paths\n")
    A("| Role | File |")
    A("|---|---|")
    A(f"| Surface-reflectance cube (8-band) | `{rel(files['sr'])}` |")
    A(f"| UDM2 quality mask (8-band) | `{rel(files['udm2'])}` |")
    A(f"| Band radiometry XML | `{rel(files['xml'])}` |")
    A(f"| Item metadata (STAC props) | `{rel(files['item_json'])}` |")
    A("")
    A("**Band layout (SuperDove, 1-indexed).** Surface-reflectance DN are stored "
      "as `uint16`; physical reflectance = `DN × 1e-4` (range 0–1). `nodata = 0` "
      "marks the clipped scene corners.\n")
    A("| Band | Name | Center λ (nm) | Render colormap | Stretch lo–hi (reflectance) |")
    A("|---|---|---|---|---|")
    for idx, name, wl, cmap_name, _why in band_meta:
        lo, hi = band_stretch.get(idx, (0, 0))
        A(f"| {idx} | {name} | {wl} | `{cmap_name}` | {lo:.3f} – {hi:.3f} |")
    A("")
    if band_coeffs:
        A("<details><summary>Per-band TOA reflectance coefficients (from XML, "
          "for reference — the SR product is already atmospherically corrected)</summary>\n")
        A("| Band | reflectanceCoefficient | radiometricScaleFactor |")
        A("|---|---|---|")
        for n in sorted(band_coeffs):
            c = band_coeffs[n]
            A(f"| {n} | {c.get('reflectance_coefficient')} | {c.get('radiometric_scale')} |")
        A("\n</details>\n")

    A("## 3. Derived data & products\n")
    outsub = rel(out_dir)
    A("All rasters are **lossless PNG** (8-bit RGBA; `nodata` → transparent), "
      f"written to the `{outsub}/` directory (paths below are relative to this README).\n")
    A("| Product | Description |")
    A("|---|---|")
    for fname, desc in outputs:
        A(f"| `{outsub}/{fname}` | {desc} |")
    A("")

    A("## 4. Processing & hyper-parameters\n")
    rgbtxt = ", ".join(f"{k} {v[0]:.3f}–{v[1]:.3f}" for k, v in rgb_stretch.items())
    A("- **Radiometric scaling:** reflectance = `DN × 1e-4` (Planet SR convention).\n"
      "- **Valid-pixel mask:** pixels non-`nodata` across all bands; nodata is "
      "rendered transparent in every output.\n"
      f"- **RGB / false-colour exposure:** independent per-channel percentile "
      f"stretch at **{args.low_pct:g}–{args.high_pct:g}%** of valid pixels "
      f"(a gray-world white balance), then **gamma {args.gamma}**. "
      f"Per-channel reflectance windows used: {rgbtxt}.\n"
      f"- **Single-band & index renders:** linear {args.low_pct:g}–{args.high_pct:g}% "
      "percentile stretch, gamma 1.0 (radiometrically honest).\n"
      "- **Colormaps:** perceptually-uniform where possible — `inferno` (NIR, as "
      "requested), `magma` (red edge), `cividis` (coastal blue), grayscale for the "
      "true-colour primary bands, diverging `RdYlGn`/`RdBu` for indices fixed to "
      "[-1, 1].\n"
      "- **Spectral indices:** NDVI=(B8−B6)/(B8+B6), NDRE=(B8−B7)/(B8+B7), "
      "NDWI=(B4−B8)/(B4+B8), NDWIre=(B7−B8)/(B7+B8).\n")
    if water_summary:
        p = water_summary["params"]
        A("- **Open-water seed (for dark tannic / turbid water that defeats McFeeters "
          "NDWI):** a pixel is *seed* water if it is **dark in NIR** (NIR reflectance "
          f"< {p['nir_max']:g}) **and** **green-dominant over NIR** (Green/NIR "
          f"> {p['gnr_min']:g}). Low NIR rejects vegetation and bright impervious "
          "surfaces; the ratio rejects spectrally-flat shadow. Thresholds are on "
          "calibrated surface reflectance and are physically meaningful "
          "(`--water-nir-max` / `--water-gnr-min`).\n"
          "- **Sub-pixel shoreline (the headline product).** A hard threshold pins "
          "the boundary to pixel centres and runs ~1–2 px conservative, because "
          "shoreline pixels are *mixed* land+water and get thrown wholesale to land. "
          "Instead each pixel's **water-area fraction** `f∈[0,1]` is recovered by "
          "linear spectral unmixing on NIR, "
          f"`f = (L − NIR)/(L − W)`, with water endmember **W = {p['nir_water']:.3f}** "
          "(robust median NIR of confident interior water) and a *local* land "
          f"endmember **L** (mean NIR of nearby land in a {p['land_window']}-px box, "
          "since land brightness varies). The shoreline is the sub-pixel "
          f"**marching-squares isoline at f = {p['frac_level']:g}** "
          "(`--water-frac-level`; lower = less conservative / higher recall). This "
          "places the edge inside the mixed transition zone — the true sub-pixel "
          "shoreline — and grows gentle shores more than steep ones (physically "
          "correct), rather than a blunt uniform dilation.\n"
          "- **Smooth lake-edge curves (PyTorch).** Each isoline is fit with a "
          f"closed **truncated-Fourier curve** (auto ≤40 harmonics, "
          "`--water-curve-harmonics`) by Adam gradient descent minimising point "
          "distance + a `k²` curvature penalty "
          f"(`--water-curve-smooth {p['curve_smooth']:g}`). The curvature term is the "
          "'don't go wild' guardrail: continuous, smooth shorelines that still hug "
          "the data, with interior holes filled.\n"
          f"- **Areas & exports.** Area is the exact polygon shoelace integral in "
          f"EPSG:32617 (sub-pixel, not a pixel count): **{water_summary['water_ha']:.1f} "
          f"ha across {water_summary['n_bodies']} bodies** "
          f"(raw threshold mask was {water_summary['raw_ha']:.1f} ha). Rasters are "
          f"oversampled {p['supersample']}× for the fractional mask; the fitted "
          "shorelines are exported as `water_boundaries.{shp,geojson}` (EPSG:32617) "
          "and `water_boundaries_wgs84.geojson`.\n"
          "- **Why not classic NDWI>0 / Otsu:** these Central-Florida ponds are "
          "turbid/tannic, so NDWI stays negative everywhere (it never crosses 0); "
          "and a data-driven Otsu threshold splits vegetation/non-vegetation rather "
          "than water/land, flooding bright urban as false water.\n"
          "- **Why no MNDWI:** SuperDove has no SWIR band, so the gold-standard "
          "MNDWI is not directly computable. For a decisive result, `--s2-fusion` "
          "fetches a same-window Sentinel-2 L2A scene and computes true "
          "MNDWI=(Green−SWIR)/(Green+SWIR).\n")

    A("## 5. Reproduce\n")
    A("```bash")
    A(f"python visualize_psscene.py {os.path.basename(dataset_dir) or '.'} \\")
    A(f"    --low-pct {args.low_pct:g} --high-pct {args.high_pct:g} --gamma {args.gamma:g}")
    A("")
    A("# add a true SWIR-based MNDWI by fusing a same-window Sentinel-2 scene")
    A("# (free, no credentials; needs network):")
    A(f"python visualize_psscene.py {os.path.basename(dataset_dir) or '.'} --s2-fusion")
    A("```")
    A("")
    A("_Bands, files and statistics above are read directly from the delivery; "
      "re-running regenerates this README from the data._")

    with open(readme_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
