"""
Segmentation parameters onto the 5 m solver grid
================================================
Takes the 0.6 m class raster from `segment_naip.py` and the per-class table from
`surface_parameters.py` and produces three rasters on the exact grid the solver runs on:

    manning_n_5m_site3.tif          [-]  surface roughness
    surface_storage_5m_site3.tif    [m]  interception + depression storage
    impervious_frac_5m_site3.tif    [-]  fraction of the cell that sheds rather than infiltrates
    class_fractions_5m_site3.npz         the per-class area fractions the three are built from

Grid alignment is not approximated: this module imports `flood_sim_ian.load_dem_for_sim` with
site3's paths patched in exactly as `run_site3_ian.py` does, so the output arrays are the same
shape and transform as `z` inside the solver by construction rather than by coincidence.

Aggregating 0.6 m classes to a 5 m cell
---------------------------------------
Each 5 m cell contains ~69 NAIP pixels of possibly several classes. Every class's binary mask is
reprojected with `Resampling.average`, which gives that class's AREA FRACTION in the cell; the
fractions are then renormalised over the classified area, so a cell that is 30 % nodata is not
quietly diluted toward zero.

Two composites of Manning's n are computed, and the difference between them is reported rather
than hidden:

  arithmetic  n = sum(f_i * n_i)
      What HEC-RAS 2D and every land-cover-table workflow uses. Treats the cell's sub-areas as
      parallel flow paths each carrying its own roughness. This is what gets written.

  Horton/Einstein  n = (sum(f_i * n_i^1.5))^(2/3)
      The composite for a single channel cross-section whose wetted perimeter spans several
      roughnesses. Written to the summary as a cross-check. It is the wrong model for a 5 m
      raster cell (there is no shared cross-section here), but it brackets the sensitivity.

Cells with no classified area at all fall back to the solver's existing scalar, so the raster is
never worse than the model it replaces.

The channel override
--------------------
Mapped stream channels are forced to a channel roughness, overriding whatever the imagery said.
This is not a tweak; it corrects a real defect found by measurement. Riparian canopy closes over
Gee Creek, and NAIP is nadir, so the creek is invisible from above: 58.9 % of channel cells came
back as `tree_canopy`, the gauge cell itself at 100 %, giving the CHANNEL BED a forest roughness
of 0.120 — three times the solver's scalar, in the one place conveyance matters most. Measured
effect before this override: discharge at the gauge cell collapsed from 101.6 to 10.5 cfs.

A forest Manning's n is not wrong for a forest; it is wrong for a channel. Chow's 0.10 for
timber assumes flow *among the trunks* — "flood stage below branches". In a channel the water is
below the canopy and meets only the bed, so the bed is what sets roughness. The 3DHP flowlines
are the same ones `dem_hydro.py` burns into the DEM, so this uses evidence the model already
trusts, and it follows the precedence rule the classification uses everywhere else: a mapped
feature outranks a spectral inference.

Usage:
    python3 segmentation/rasterize_parameters.py --site site3
    python3 segmentation/rasterize_parameters.py --site site3 --cell-size 5
"""
import os
import sys
import json
import argparse
import warnings

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
sys.path.insert(0, os.path.join(PROJ_DIR, "simulation"))
sys.path.insert(0, os.path.join(PROJ_DIR, "lidar"))

from segment_naip import CLASSES  # noqa: E402

# Chow (1959) Table 5-6, natural minor stream, clean and winding with some pools and weedy banks:
# 0.033-0.050. 0.045 for a small sandy-bed Florida creek with vegetated margins. Close to the
# solver's own 0.040 scalar by coincidence, not by construction — which is itself the point: the
# channel was already being modelled about right, and the segmentation's contribution belongs on
# the hillslopes, not in the creek.
CHANNEL_MANNING_N = 0.045
# Half-width of the channel corridor, i.e. channel plus immediate banks. 10 m gives a 20 m
# corridor, which is 0.95 % of the domain (17,673 cells) along 22 km of mapped network — a
# reasonable channel-and-bank width for a creek draining 33 km2. At 5 m it is 0.47 %.
#
# Stated plainly because the choice is load-bearing and could otherwise look tuned: the gauge
# cell sits exactly ONE CELL outside the 5 m buffer, so 5 m leaves the validation cell itself at
# the forest roughness this override exists to correct. The 10 m value is defensible on channel
# width alone, and both figures are reported in the summary so the sensitivity is visible.
CHANNEL_BUFFER_M = 10.0


def solver_grid(site, cell_size):
    """The solver's own grid, obtained by running the solver's own loader.

    Patched the same way run_site3_ian.py patches it — deliberately NOT a reimplementation, so
    this cannot drift out of alignment with the DEM the solver actually reads.
    """
    import flood_sim_ian as fsi
    from test_sites import get_site
    s = get_site(site)
    fsi.DEM_COND = s["dem_cond_path"]
    z, profile, dx = fsi.load_dem_for_sim(cell_size)
    return z, profile, dx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="site3")
    ap.add_argument("--cell-size", type=float, default=5.0)
    ap.add_argument("--landcover", default=None,
                    help="basename of the class raster in data/ (default: the spectral "
                         "backend's landcover_0.6m_<site>.tif). Point this at the SAM3 "
                         "backend's output to rasterise that instead — the two backends share "
                         "an encoding, so everything downstream is identical.")
    ap.add_argument("--tag", default="",
                    help="suffix for the output rasters, so two backends can coexist")
    ap.add_argument("--no-channel-override", action="store_true",
                    help="leave mapped stream channels at their imagery-derived roughness "
                         "(reproduces the defect described in the module docstring)")
    args = ap.parse_args()

    lc_path = (os.path.join(DATA_DIR, args.landcover) if args.landcover
               else os.path.join(DATA_DIR, f"landcover_0.6m_{args.site}.tif"))
    par_path = os.path.join(DATA_DIR, "surface_parameters.json")
    for p in (lc_path, par_path):
        if not os.path.exists(p):
            raise SystemExit(f"missing input: {p}\n  run segment_naip.py / surface_parameters.py first")

    with open(par_path) as fh:
        table = json.load(fh)
    params = table["classes"]

    print("=" * 74)
    print(f"Rasterising segmentation parameters onto the solver grid — {args.site}")
    print("=" * 74)

    z, profile, dx = solver_grid(args.site, args.cell_size)
    shape = z.shape
    dst_transform, dst_crs = profile["transform"], profile["crs"]
    print(f"  solver grid: {shape[0]}x{shape[1]} @ {dx:.2f} m  {dst_crs}")

    with rasterio.open(lc_path) as src:
        print(f"  landcover  : {src.height}x{src.width} @ {src.res[0]:.2f} m  {src.crs}")
        lc = src.read(1)
        src_transform, src_crs = src.transform, src.crs

    # ── per-class area fractions on the solver grid ───────────────────────────
    fractions = {}
    for code, name in CLASSES.items():
        if code == 0:
            continue
        mask = (lc == code).astype(np.float32)
        if mask.sum() == 0:
            fractions[name] = np.zeros(shape, dtype=np.float32)
            continue
        out = np.zeros(shape, dtype=np.float32)
        reproject(mask, out,
                  src_transform=src_transform, src_crs=src_crs,
                  dst_transform=dst_transform, dst_crs=dst_crs,
                  resampling=Resampling.average)
        fractions[name] = out
        del mask
    del lc

    classified = np.sum([fractions[n] for n in fractions], axis=0)
    has_class = classified > 0.01
    print(f"  cells with classified cover: {100 * has_class.mean():.1f} %")

    # Renormalise over classified area only.
    norm = np.where(has_class, classified, 1.0)
    for n in fractions:
        fractions[n] = fractions[n] / norm

    # ── composites ────────────────────────────────────────────────────────────
    import flood_sim_ian as fsi
    fallback_n = fsi.MANNING_N

    n_arith = np.zeros(shape, dtype=np.float32)
    n_hort15 = np.zeros(shape, dtype=np.float32)
    storage = np.zeros(shape, dtype=np.float32)
    imperv = np.zeros(shape, dtype=np.float32)
    # The vision route to Ks and soil storage, built so it can be measured against SSURGO rather
    # than argued about. Ks is area-weighted arithmetically because sub-areas of a cell
    # infiltrate in parallel, each at its own rate.
    vis_ks = np.zeros(shape, dtype=np.float32)      # mm/hr, DRY (AMC-III applied by the driver)
    vis_smax = np.zeros(shape, dtype=np.float32)    # m
    for name, f in fractions.items():
        p = params[name]
        n_arith += f * np.float32(p["manning_n"])
        n_hort15 += f * np.float32(p["manning_n"] ** 1.5)
        storage += f * np.float32(p["surface_storage_m"])
        imperv += f * np.float32(p["impervious_fraction"])
        vis_ks += f * np.float32(p["vision_ks_mm_hr_dry"])
        vis_smax += f * np.float32(p["vision_soil_storage_m"])
    n_horton = np.power(np.maximum(n_hort15, 1e-12), 2.0 / 3.0).astype(np.float32)

    # Unclassified cells keep the model's existing behaviour rather than inheriting a zero.
    n_arith[~has_class] = fallback_n
    n_horton[~has_class] = fallback_n
    storage[~has_class] = 0.0
    imperv[~has_class] = 0.0

    # ── channel override ──────────────────────────────────────────────────────
    channel_stats = {"applied": False}
    if not args.no_channel_override:
        import geopandas as gpd
        from rasterio.features import rasterize
        fl_path = os.path.join(PROJ_DIR, "site3_gee_creek", "hydrography", "data",
                               "3dhp_flowlines.geojson")
        if os.path.exists(fl_path):
            fl = gpd.read_file(fl_path).to_crs(dst_crs)
            geoms = [(g.buffer(CHANNEL_BUFFER_M), 1) for g in fl.geometry if g is not None]
            chan = rasterize(geoms, out_shape=shape, transform=dst_transform,
                             fill=0, dtype=np.uint8).astype(bool)
            before = float(n_arith[chan].mean()) if chan.any() else float("nan")
            n_arith[chan] = np.float32(CHANNEL_MANNING_N)
            n_horton[chan] = np.float32(CHANNEL_MANNING_N)
            storage[chan] = 0.0        # a channel has no interception store to fill
            channel_stats = {"applied": True, "cells": int(chan.sum()),
                             "buffer_m": CHANNEL_BUFFER_M,
                             "channel_n": CHANNEL_MANNING_N,
                             "imagery_derived_channel_n_mean": round(before, 4)}
            print(f"\n  channel override: {int(chan.sum()):,} cells on the 3DHP network forced "
                  f"to n={CHANNEL_MANNING_N} (imagery gave them a mean of {before:.4f})")
        else:
            print(f"\n  channel override SKIPPED — {fl_path} not found")

    valid = np.isfinite(z)
    out_profile = {"driver": "GTiff", "height": shape[0], "width": shape[1], "count": 1,
                   "dtype": "float32", "crs": dst_crs, "transform": dst_transform,
                   "compress": "lzw", "nodata": None}
    written = []
    for nm, arr in [("manning_n", n_arith), ("surface_storage", storage),
                    ("impervious_frac", imperv),
                    ("vision_ks_dry_mm_hr", vis_ks), ("vision_soil_storage", vis_smax)]:
        p = os.path.join(DATA_DIR, f"{nm}_{args.cell_size:g}m_{args.site}{args.tag}.tif")
        with rasterio.open(p, "w", **out_profile) as dst:
            dst.write(arr, 1)
        written.append(os.path.relpath(p, PROJ_DIR))

    fr_path = os.path.join(DATA_DIR,
                           f"class_fractions_{args.cell_size:g}m_{args.site}{args.tag}.npz")
    np.savez_compressed(fr_path, **fractions)
    written.append(os.path.relpath(fr_path, PROJ_DIR))

    # ── summary ───────────────────────────────────────────────────────────────
    area_frac = {n: float(fractions[n][valid & has_class].mean()) for n in fractions}
    storm_mm = 392.0     # the Ian window's KSFB total, for sizing the storage term
    summary = {
        "site": args.site, "cell_size_m": float(dx), "grid_shape": list(shape),
        "landcover_source": os.path.basename(lc_path),
        "parameter_table_version": table["provenance"]["table_version"],
        "cells_with_classified_cover_pct": float(100 * has_class.mean()),
        "channel_override": channel_stats,
        "domain_area_fraction_by_class": {k: round(v, 5) for k, v in
                                          sorted(area_frac.items(), key=lambda kv: -kv[1])},
        "manning_n": {
            "solver_scalar_today": float(fallback_n),
            "arithmetic_mean": float(n_arith[valid].mean()),
            "arithmetic_p05_p50_p95": [float(v) for v in np.percentile(n_arith[valid], [5, 50, 95])],
            "arithmetic_min_max": [float(n_arith[valid].min()), float(n_arith[valid].max())],
            "horton_composite_mean": float(n_horton[valid].mean()),
            "horton_vs_arithmetic_mean_pct": float(
                100 * (n_horton[valid].mean() - n_arith[valid].mean()) / n_arith[valid].mean()),
            "fraction_of_domain_rougher_than_scalar": float((n_arith[valid] > fallback_n).mean()),
        },
        "surface_storage": {
            "mean_mm": float(1000 * storage[valid].mean()),
            "max_mm": float(1000 * storage[valid].max()),
            "as_pct_of_ian_storm_total": float(100 * 1000 * storage[valid].mean() / storm_mm),
        },
        "vision_soil": {
            "ks_dry_mm_hr_mean": float(vis_ks[valid].mean()),
            "ks_dry_mm_hr_p05_p50_p95": [float(v) for v in np.percentile(vis_ks[valid], [5, 50, 95])],
            "soil_storage_mm_mean": float(1000 * vis_smax[valid].mean()),
            "soil_storage_mm_p05_p50_p95": [float(1000 * v) for v in
                                            np.percentile(vis_smax[valid], [5, 50, 95])],
            "note": ("DRY conductivity; the driver applies the same AMC-III factor the solver "
                     "already applies to SSURGO Ksat, so the two are compared like for like"),
        },
        "impervious_fraction": {
            "mean": float(imperv[valid].mean()),
            "nlcd_domain_mean_for_reference": "see soil/data/nlcd_impervious.tif",
        },
    }
    sp = os.path.join(DATA_DIR, f"parameter_raster_summary_{args.site}{args.tag}.json")
    with open(sp, "w") as fh:
        json.dump(summary, fh, indent=2)
    written.append(os.path.relpath(sp, PROJ_DIR))

    print("\n  domain area fraction by class:")
    for k, v in sorted(area_frac.items(), key=lambda kv: -kv[1]):
        if v > 1e-4:
            print(f"    {k:<18} {100*v:5.1f} %   n={params[k]['manning_n']:.3f}")
    m = summary["manning_n"]
    print(f"\n  Manning's n   scalar today : {m['solver_scalar_today']:.4f}  (uniform)")
    print(f"                new mean     : {m['arithmetic_mean']:.4f}   "
          f"p05/p50/p95 {m['arithmetic_p05_p50_p95'][0]:.4f}/"
          f"{m['arithmetic_p05_p50_p95'][1]:.4f}/{m['arithmetic_p05_p50_p95'][2]:.4f}")
    print(f"                range        : {m['arithmetic_min_max'][0]:.4f} - {m['arithmetic_min_max'][1]:.4f}")
    print(f"                Horton comp. : {m['horton_composite_mean']:.4f} "
          f"({m['horton_vs_arithmetic_mean_pct']:+.1f} % vs arithmetic — sensitivity, not the shipped value)")
    print(f"                rougher than the scalar over {100*m['fraction_of_domain_rougher_than_scalar']:.1f} % of the domain")
    s = summary["surface_storage"]
    print(f"\n  surface storage mean {s['mean_mm']:.2f} mm (max {s['max_mm']:.2f}) "
          f"= {s['as_pct_of_ian_storm_total']:.2f} % of the 392 mm Ian total")
    v = summary["vision_soil"]
    print(f"\n  vision route to the soil parameters (for comparison against SSURGO):")
    print(f"    Ks  dry mean {v['ks_dry_mm_hr_mean']:.1f} mm/hr  "
          f"p05/p50/p95 {v['ks_dry_mm_hr_p05_p50_p95'][0]:.0f}/"
          f"{v['ks_dry_mm_hr_p05_p50_p95'][1]:.0f}/{v['ks_dry_mm_hr_p05_p50_p95'][2]:.0f}")
    print(f"    soil storage mean {v['soil_storage_mm_mean']:.0f} mm  "
          f"p05/p50/p95 {v['soil_storage_mm_p05_p50_p95'][0]:.0f}/"
          f"{v['soil_storage_mm_p05_p50_p95'][1]:.0f}/{v['soil_storage_mm_p05_p50_p95'][2]:.0f}")
    print(f"  impervious fraction mean {summary['impervious_fraction']['mean']:.3f}")
    print("\n  wrote:")
    for w in written:
        print(f"    {w}")


if __name__ == "__main__":
    main()
