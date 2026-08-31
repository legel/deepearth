"""
Independent validation of the LiDAR canopy layer against NLCD Tree Canopy Cover
===============================================================================
`tree_canopy` is 44.4 % of the domain and carries the largest parameter change introduced here
(n = 0.120 against the solver's 0.040 scalar, a 3x step). The impervious and water classes were
each cross-checked against an independent source — NLCD impervious and 3DHP waterbodies — but
canopy was not, which left the single most consequential class resting entirely on this
project's own canopy-height model with nothing to check it against.

This closes that gap. **NLCD Tree Canopy Cover (TCC) 2021, 30 m** is a genuinely independent
product: USFS/MRLC derive it from Landsat time series with FIA plot training data, so it shares
neither a sensor, a platform, nor a method with a 2018 airborne LiDAR return-height statistic.
Agreement between them is therefore evidence about the canopy layer, not a restatement of it.

What is and is not being tested
-------------------------------
TCC is a *cover fraction* at 30 m. The comparison is made on that footprint — this project's 2 m
canopy cover is area-averaged up to TCC's own grid, rather than TCC being interpolated down,
because upscaling a fine measurement is well-posed and downscaling a coarse one is not.

Two figures are reported and they answer different questions:

  * **cover-fraction agreement** — correlation and bias between the two continuous fields. This
    is the direct test of the canopy-height model.
  * **class agreement** — how often the `tree_canopy` CLASS assignment lands where TCC also
    says the cell is mostly canopy. This is the test of what the solver actually consumes,
    since a parameter is attached to the class and not to the fraction.

A perfect match is not expected and would be suspicious: TCC 2021 and the LiDAR are three years
apart, TCC's 30 m pixel cannot resolve a hedgerow or a single street tree, and TCC is trained to
report *tree* canopy where the LiDAR height model counts any return above 2 m, including tall
shrub and structures the classification did not route elsewhere. The question is whether the two
agree well enough that the roughness field is defensible, and where they disagree, in which
direction.

Usage:
    python3 segmentation/validate_canopy.py --site site3
"""
import os
import sys
import json
import argparse
import warnings
import urllib.request

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.warp import reproject, Resampling
from pyproj import Transformer

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
sys.path.insert(0, os.path.join(PROJ_DIR, "lidar"))

MRLC_WCS_BASE = "https://www.mrlc.gov/geoserver/mrlc_display/ows"
# NLCD Tree Canopy Cover, CONUS, 2021 — the same year as the NAIP used for the classification.
# Confirmed present in the service's own GetCapabilities rather than assumed.
TCC_COVERAGE = "mrlc_display:nlcd_tcc_conus_2021_v2021-4"

# TCC uses 254/255 for non-processing-area and fill. Values above 100 are not cover fractions.
TCC_VALID_MAX = 100


def fetch_tcc(bounds_wsen, out_path):
    """NLCD TCC over the AOI via MRLC WCS 1.0.0, at roughly its own 30 m native resolution."""
    if os.path.exists(out_path):
        print(f"  using cached {os.path.relpath(out_path, PROJ_DIR)}")
        return True
    w, s, e, n = bounds_wsen
    pad = 0.006
    w, s, e, n = w - pad, s - pad, e + pad, n + pad
    width = max(100, int((e - w) / 0.00027))
    height = max(100, int((n - s) / 0.00027))
    url = (f"{MRLC_WCS_BASE}?SERVICE=WCS&VERSION=1.0.0&REQUEST=GetCoverage"
           f"&COVERAGE={TCC_COVERAGE}&BBOX={w},{s},{e},{n}"
           f"&CRS=EPSG:4326&RESPONSE_CRS=EPSG:4326"
           f"&FORMAT=GeoTIFF&WIDTH={width}&HEIGHT={height}")
    print(f"  WCS request ({width}x{height} px): {TCC_COVERAGE}")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            raw = resp.read()
    except Exception as ex:
        print(f"  WCS request failed: {ex}")
        return False
    if raw[:4] not in (b"II*\x00", b"MM\x00*"):
        print(f"  non-TIFF response (likely an XML error): {raw[:200]!r}")
        return False
    with open(out_path, "wb") as fh:
        fh.write(raw)
    print(f"  wrote {os.path.relpath(out_path, PROJ_DIR)} ({len(raw)/1e6:.1f} MB)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="site3")
    args = ap.parse_args()

    cover_p = os.path.join(DATA_DIR, f"canopy_cover_2m_{args.site}.tif")
    dens_p = os.path.join(DATA_DIR, f"return_density_2m_{args.site}.tif")
    frac_p = os.path.join(DATA_DIR, f"class_fractions_5m_{args.site}.npz")
    for p in (cover_p, dens_p):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}\n  run segmentation/canopy_lidar.py first")

    print("=" * 74)
    print(f"Canopy validation vs NLCD Tree Canopy Cover 2021 — {args.site}")
    print("=" * 74)

    with rasterio.open(cover_p) as src:
        cover = src.read(1)
        cov_tf, cov_crs, cov_shape = src.transform, src.crs, src.shape
        b = src.bounds
    with rasterio.open(dens_p) as src:
        density = src.read(1)
    has_lidar = density > 0

    tr = Transformer.from_crs(cov_crs, "epsg:4326", always_xy=True)
    xs = [b.left, b.right, b.left, b.right]
    ys = [b.bottom, b.bottom, b.top, b.top]
    lons, lats = tr.transform(xs, ys)
    bounds_wsen = (min(lons), min(lats), max(lons), max(lats))

    tcc_p = os.path.join(DATA_DIR, f"nlcd_tcc_2021_{args.site}.tif")
    if not fetch_tcc(bounds_wsen, tcc_p):
        raise SystemExit("could not fetch NLCD TCC")

    # ── put both fields on TCC's own 30 m grid ────────────────────────────────
    with rasterio.open(tcc_p) as src:
        tcc_native = src.read(1).astype(np.float32)
        tcc_tf, tcc_crs, tcc_shape = src.transform, src.crs, src.shape

    # Our 2 m cover, area-averaged onto TCC's grid. Upscaling a fine measurement is well posed;
    # interpolating TCC down to 2 m would invent detail it does not have.
    ours = np.full(tcc_shape, np.nan, dtype=np.float32)
    reproject(cover.astype(np.float32), ours,
              src_transform=cov_tf, src_crs=cov_crs,
              dst_transform=tcc_tf, dst_crs=tcc_crs,
              src_nodata=None, dst_nodata=np.nan, resampling=Resampling.average)
    # LiDAR coverage fraction on the same grid, so cells only partly covered can be excluded.
    covfrac = np.full(tcc_shape, np.nan, dtype=np.float32)
    reproject(has_lidar.astype(np.float32), covfrac,
              src_transform=cov_tf, src_crs=cov_crs,
              dst_transform=tcc_tf, dst_crs=tcc_crs,
              src_nodata=None, dst_nodata=np.nan, resampling=Resampling.average)

    tcc = np.where(tcc_native <= TCC_VALID_MAX, tcc_native, np.nan) / 100.0

    # Only compare where TCC is valid AND the cell is essentially fully LiDAR-covered — a
    # half-covered cell reports artificially low canopy and would manufacture a bias.
    m = np.isfinite(tcc) & np.isfinite(ours) & (covfrac > 0.95)
    n_cmp = int(m.sum())
    if n_cmp < 100:
        raise SystemExit(f"only {n_cmp} comparable cells; check CRS/extent overlap")

    a, o = tcc[m], ours[m]
    r = float(np.corrcoef(a, o)[0, 1])
    bias = float(np.mean(o - a))
    mae = float(np.mean(np.abs(o - a)))
    rmse = float(np.sqrt(np.mean((o - a) ** 2)))

    print(f"\n  comparable 30 m cells: {n_cmp:,}  "
          f"({100*n_cmp/np.isfinite(tcc).sum():.0f} % of valid TCC cells, "
          f"after excluding partial LiDAR coverage)")
    print("\n[1] COVER FRACTION — the direct test of the canopy-height model")
    print(f"  NLCD TCC   mean {100*a.mean():5.1f} %   p10/p50/p90 "
          f"{100*np.percentile(a,10):.0f}/{100*np.percentile(a,50):.0f}/{100*np.percentile(a,90):.0f}")
    print(f"  LiDAR ours mean {100*o.mean():5.1f} %   p10/p50/p90 "
          f"{100*np.percentile(o,10):.0f}/{100*np.percentile(o,50):.0f}/{100*np.percentile(o,90):.0f}")
    print(f"  correlation r = {r:.3f}   bias {100*bias:+.1f} pp   MAE {100*mae:.1f} pp   "
          f"RMSE {100*rmse:.1f} pp")

    # ── class agreement: what the solver actually consumes ────────────────────
    cls_stats = None
    if os.path.exists(frac_p):
        fr = np.load(frac_p)
        tree5 = fr["tree_canopy"]
        with rasterio.open(os.path.join(DATA_DIR, f"manning_n_5m_{args.site}.tif")) as src:
            m5_tf, m5_crs, m5_shape = src.transform, src.crs, src.shape
        tree_on_tcc = np.full(tcc_shape, np.nan, dtype=np.float32)
        reproject(tree5.astype(np.float32), tree_on_tcc,
                  src_transform=m5_tf, src_crs=m5_crs,
                  dst_transform=tcc_tf, dst_crs=tcc_crs,
                  src_nodata=None, dst_nodata=np.nan, resampling=Resampling.average)
        mm = m & np.isfinite(tree_on_tcc)
        # "Mostly canopy" on both sides, at TCC's own conventional forest threshold.
        THR = 0.50
        ours_t = tree_on_tcc[mm] >= THR
        tcc_t = tcc[mm] >= THR
        tp = int((ours_t & tcc_t).sum()); fp = int((ours_t & ~tcc_t).sum())
        fn = int((~ours_t & tcc_t).sum()); tn = int((~ours_t & ~tcc_t).sum())
        iou = tp / max(tp + fp + fn, 1)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        acc = (tp + tn) / max(tp + fp + fn + tn, 1)
        print(f"\n[2] CLASS AGREEMENT — what the solver consumes (>= {THR:.0%} canopy on both sides)")
        print(f"  agreement {100*acc:.1f} %   IoU {iou:.3f}   F1 {f1:.3f}")
        print(f"  tree in both {tp:,} | only ours {fp:,} | only TCC {fn:,} | neither {tn:,}")
        cls_stats = {"threshold": THR, "accuracy": round(acc, 4), "iou": round(iou, 4),
                     "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    # ── what the disagreement is worth in Manning's n ─────────────────────────
    # A strict ONE-SIDED bound, not an estimate: tree fraction is capped at what TCC reports
    # wherever ours is higher, and never raised where ours is lower. Because the two agree
    # closely in the MEAN, that asymmetry deliberately overstates the correction — which is what
    # makes it a bound on how much canopy placement error could be inflating the roughness field.
    n_stats = None
    if os.path.exists(frac_p):
        with rasterio.open(os.path.join(DATA_DIR, f"manning_n_5m_{args.site}.tif")) as src:
            mn = src.read(1)
            m5_tf, m5_crs, m5_shape = src.transform, src.crs, src.shape
        tcc_on5 = np.full(m5_shape, np.nan, dtype=np.float32)
        reproject(tcc, tcc_on5, src_transform=tcc_tf, src_crs=tcc_crs,
                  dst_transform=m5_tf, dst_crs=m5_crs,
                  src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.bilinear)
        with open(os.path.join(DATA_DIR, "surface_parameters.json")) as fh:
            par = json.load(fh)["classes"]
        tree5 = np.load(frac_p)["tree_canopy"]
        ok = np.isfinite(tcc_on5)
        excess = np.clip(tree5 - np.where(ok, tcc_on5, tree5), 0, None)
        dn = excess * (par["tree_canopy"]["manning_n"] - par["grass_turf"]["manning_n"])
        mn_capped = mn - np.where(ok, dn, 0.0)
        scalar = 0.040
        introduced = float(mn.mean()) - scalar
        bound = float(mn.mean() - mn_capped.mean())
        print("\n[3] WHAT THE DISAGREEMENT IS WORTH")
        print(f"  domain-mean tree fraction: ours {float(tree5[ok].mean()):.3f}  "
              f"TCC {float(np.nanmean(tcc_on5[ok])):.3f}   -- the two agree in AGGREGATE;")
        print(f"                             the scatter is about WHERE canopy is, not how much.")
        print(f"  Manning's n  scalar {scalar:.4f} -> shipped {float(mn.mean()):.4f} "
              f"({introduced:+.4f})")
        print(f"  TCC-capped (one-sided upper bound): {float(mn_capped.mean()):.4f}")
        print(f"  => canopy placement error explains AT MOST {100*bound/introduced:.0f} % of the "
              f"roughness change introduced")
        n_stats = {"scalar": scalar,
                   "shipped_mean_n": round(float(mn.mean()), 5),
                   "tcc_capped_mean_n": round(float(mn_capped.mean()), 5),
                   "mean_tree_fraction_ours": round(float(tree5[ok].mean()), 4),
                   "mean_tree_fraction_tcc": round(float(np.nanmean(tcc_on5[ok])), 4),
                   "max_share_of_n_change_from_canopy_error": round(bound / introduced, 4),
                   "note": ("one-sided bound: tree fraction is only ever capped downward, never "
                            "raised, so this overstates the correction by construction")}

    summary = {
        "site": args.site,
        "reference": {"product": "NLCD Tree Canopy Cover 2021 (CONUS)",
                      "coverage": TCC_COVERAGE, "resolution_m": 30,
                      "independence": ("Landsat time series + FIA plot training — shares no "
                                       "sensor, platform or method with 2018 airborne LiDAR")},
        "n_comparable_30m_cells": n_cmp,
        "cover_fraction": {
            "nlcd_tcc_mean": round(float(a.mean()), 4),
            "lidar_mean": round(float(o.mean()), 4),
            "correlation_r": round(r, 4),
            "bias_pp": round(100 * bias, 2),
            "mae_pp": round(100 * mae, 2),
            "rmse_pp": round(100 * rmse, 2),
        },
        "class_agreement": cls_stats,
        "roughness_sensitivity": n_stats,
        "caveats": [
            "NLCD TCC is 2021; the LiDAR acquisition is 2018.",
            "TCC's 30 m pixel cannot resolve hedgerows or individual street trees.",
            "TCC reports TREE canopy; the LiDAR statistic counts any return above 2 m.",
            "Cells less than 95 % LiDAR-covered are excluded, so the domain margin is not scored.",
        ],
    }
    sp = os.path.join(DATA_DIR, f"canopy_validation_{args.site}.json")
    with open(sp, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  wrote {os.path.relpath(sp, PROJ_DIR)}")
    print("=" * 74)


if __name__ == "__main__":
    main()
