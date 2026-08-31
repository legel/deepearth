"""
Surface-class segmentation of NAIP 0.6 m imagery
================================================
Segments the NAIP orthophoto into named surface classes and writes both a class raster and a
per-segment feature table. Downstream, `surface_parameters.py` attaches {material, Smax, Ks,
Manning's n} to each class and `rasterize_parameters.py` puts those on the 5 m solver grid.

Backends
--------
`--backend spectral` (default, works today)
    SLIC superpixels over NAIP RGB+NIR, then a segment-level classifier over features that are
    each a real measurement: NDVI, NIR/visible brightness, the LiDAR canopy-height model from
    `canopy_lidar.py`, HAND from the hydro-conditioning chain, and the OSM road/building
    footprints the solver's impervious mask already uses.

`--backend sam3` (BLOCKED — see below)
    `facebook/sam3` on HuggingFace is `gated: manual`. Without an approved token the config
    fetch returns HTTP 401, and the pipeline interpreter's transformers 4.57.6 has no
    `Sam3Model`/`Sam3Processor` either. Unblocking it needs, in order:
      1. an approved access request at https://huggingface.co/facebook/sam3
      2. `HF_TOKEN` in the environment (or `~/.cache/huggingface/token`)
      3. transformers 5.x — which needs Python >= 3.10, so it CANNOT go in this 3.9.6 pipeline
         interpreter. There is no 4.58; the series goes 4.57.x -> 5.0.0. Run SAM3 in the 3.11
         venv (`cfx_sr417/.venv`, torch 2.13 + MPS) as a standalone stage that writes
         `landcover_0.6m.tif`, and let the 3.9 pipeline consume that GeoTIFF — which is exactly
         what making the class raster the interface buys.
    Hardware is NOT the constraint: the weights are 3.44 GB against 17.2 GB of unified memory
    with MPS available. The MPS OOM recorded elsewhere in this project was the 8.67M-edge mesh
    GNN, an unrelated workload.
    `facebook/sam3.1` is gated the same way. `facebook/sam2.1-hiera-large` IS reachable but
    produces unnamed instance masks — it segments without naming, so it does not replace the
    open-vocabulary labelling that is the whole point of the SAM3 step.

Why the class map, not the instances, is the interface
------------------------------------------------------
What the solver consumes is a per-cell parameter, and a parameter is a property of the SURFACE
CLASS, not of the instance: two adjacent oak crowns get the same Manning's n whether SAM3 calls
them one segment or two. So the contract between this stage and the next is
`landcover_*.tif` + `segments_*.csv`, and a SAM3 backend drops in behind that contract without
anything downstream changing.

The vegetation threshold is calibrated, not assumed
---------------------------------------------------
NAIP for site3 was flown 2021-12-02 — winter, when central Florida turf is partly dormant, so a
textbook NDVI cutoff would be wrong here. Instead the split is derived from two independent
labels already on disk: OSM building footprints (definitely not vegetation) and LiDAR canopy
above 3 m (definitely vegetation). The threshold is the equal-error point between those two
measured NDVI distributions, written to `calibration_*.json` so the run is auditable.

Usage:
    python3 segmentation/segment_naip.py --site site3 --calibrate-only
    python3 segmentation/segment_naip.py --site site3
    python3 segmentation/segment_naip.py --site site3 --max-tiles 2    # quick check
"""
import os
import sys
import json
import time
import argparse
import warnings

import numpy as np
import rasterio
from rasterio.windows import Window, transform as window_transform
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
import geopandas as gpd
from skimage.segmentation import slic

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
sys.path.insert(0, os.path.join(PROJ_DIR, "lidar"))

_PROGRAM_DIR = os.path.dirname(PROJ_DIR)
if _PROGRAM_DIR not in sys.path:
    sys.path.insert(0, _PROGRAM_DIR)
from floodtwin.physics import ROAD_BUFFER_M, ROAD_BUFFER_DEFAULT_M   # noqa: E402

# ── surface classes ───────────────────────────────────────────────────────────
# Ordered by classification precedence, which is also the order the rules below are applied in:
# a directly-mapped footprint (building, road) always wins over a spectral inference.
CLASSES = {
    0: "nodata",
    1: "water",
    2: "building_roof",
    3: "road_paved",
    4: "impervious_other",
    5: "tree_canopy",
    6: "shrub_scrub",
    7: "grass_turf",
    8: "bare_soil",
    9: "wetland_marsh",
}
NAME_TO_CODE = {v: k for k, v in CLASSES.items()}

TILE = 2048                 # NAIP pixels per processing tile
SLIC_SEGMENT_PX = 300       # target superpixel size ~ 17x17 px ~ 10x10 m, i.e. ~2 solver cells.
                            # Was 900 (~18 m). Visual QC showed 18 m segments straddling the
                            # tree/lawn boundary in suburban blocks, and since a segment gets ONE
                            # class, a segment that was half canopy came out entirely canopy.
SLIC_COMPACTNESS = 12.0     # higher = more compact/blobby; 10-15 suits aerial imagery

CANOPY_HEIGHT_M = 2.0       # CHM above this is tree canopy, per canopy_lidar.py's own cutoff
SHRUB_HEIGHT_M = 0.5
WETLAND_HAND_M = 0.5        # HAND below this is at/near the drainage surface
WATER_NIR_MAX = 70          # water absorbs NIR strongly; uint8 NAIP band 4
BRIGHT_IMPERVIOUS = 120     # visible-brightness floor for dry pavement vs. bare soil


def site_paths(site):
    if site != "site3":
        raise SystemExit("only site3 has NAIP + cached LiDAR today")
    s = os.path.join(PROJ_DIR, "site3_gee_creek")
    return {
        "rgb":       os.path.join(s, "imagery", "data", "naip_2021_RGB.tif"),
        "nir":       os.path.join(s, "imagery", "data", "naip_2021_NIR.tif"),
        "ndvi":      os.path.join(s, "imagery", "data", "naip_2021_NDVI.tif"),
        "chm":       os.path.join(DATA_DIR, "chm_2m_site3.tif"),
        "cover":     os.path.join(DATA_DIR, "canopy_cover_2m_site3.tif"),
        "hand":      os.path.join(s, "dem", "data", "hydro", "hand.tif"),
        "roads":     os.path.join(s, "infrastructure", "data", "roads.geojson"),
        "buildings": os.path.join(s, "infrastructure", "data", "buildings.geojson"),
    }


class _Aux:
    """A whole auxiliary raster held in memory, reprojectable into any NAIP tile window.

    Read once, not once per tile: HAND is 7819x7810 float32 (244 MB) and there are 25 tiles, so
    re-reading it inside the loop would cost ~6 GB of I/O for no reason.
    """

    def __init__(self, path):
        with rasterio.open(path) as src:
            self.arr = src.read(1).astype(np.float32)
            self.transform, self.crs, self.nodata = src.transform, src.crs, src.nodata
            self.shape, self.res = (src.height, src.width), src.res

    def window(self, win_transform, shape, dst_crs, resampling=Resampling.bilinear):
        out = np.full(shape, np.nan, dtype=np.float32)
        reproject(self.arr, out,
                  src_transform=self.transform, src_crs=self.crs,
                  dst_transform=win_transform, dst_crs=dst_crs,
                  src_nodata=self.nodata, dst_nodata=np.nan, resampling=resampling)
        return out


def load_footprints(paths, dst_crs):
    """OSM roads (buffered by highway class) and buildings, in the NAIP CRS.

    Buffer widths come from floodtwin.physics, so this classification's 'road' footprint is by
    construction the same one flood_sim_ian.py's impervious mask already uses — the two cannot
    disagree about where a road is.
    """
    roads = gpd.read_file(paths["roads"]).to_crs(dst_crs)
    buildings = gpd.read_file(paths["buildings"]).to_crs(dst_crs)
    road_shapes = [
        (row.geometry.buffer(ROAD_BUFFER_M.get(str(row.get("highway")), ROAD_BUFFER_DEFAULT_M)), 1)
        for _, row in roads.iterrows() if row.geometry is not None
    ]
    bldg_shapes = [(g, 1) for g in buildings.geometry if g is not None]
    print(f"  footprints: {len(road_shapes)} road segments, {len(bldg_shapes)} buildings")
    return road_shapes, bldg_shapes


def calibrate_ndvi(paths, n_samples=40):
    """Equal-error NDVI split between two independently-labelled populations.

    positives: LiDAR CHM > 3 m           — physically tall vegetation, cannot be pavement
    negatives: inside an OSM building    — a roof, cannot be vegetation

    Neither label comes from NAIP, so the threshold this produces is not circular.
    """
    print("\n[calibration] deriving the vegetation NDVI threshold from CHM + OSM buildings …")
    chm_aux = _Aux(paths["chm"])
    with rasterio.open(paths["ndvi"]) as ndvi_src:
        dst_crs = ndvi_src.crs
        _, bldg_shapes = load_footprints(paths, dst_crs)
        H, W = ndvi_src.height, ndvi_src.width

        rng = np.random.default_rng(0)
        pos, neg = [], []
        tries = 0
        while len(pos) < n_samples * 400 and tries < n_samples * 6:
            tries += 1
            r0 = int(rng.integers(0, H - TILE))
            c0 = int(rng.integers(0, W - TILE))
            win = Window(c0, r0, TILE, TILE)
            wt = window_transform(win, ndvi_src.transform)
            nd = ndvi_src.read(1, window=win)
            chm = chm_aux.window(wt, nd.shape, dst_crs)
            bmask = rasterize(bldg_shapes, out_shape=nd.shape, transform=wt,
                              fill=0, dtype=np.uint8).astype(bool) if bldg_shapes else np.zeros(nd.shape, bool)

            p = nd[(chm > 3.0) & np.isfinite(nd)]
            n = nd[bmask & (chm < 1.0) & np.isfinite(nd)]
            if len(p):
                pos.append(rng.choice(p, size=min(len(p), 20000), replace=False))
            if len(n):
                neg.append(rng.choice(n, size=min(len(n), 20000), replace=False))

    pos = np.concatenate(pos) if pos else np.array([])
    neg = np.concatenate(neg) if neg else np.array([])
    if len(pos) < 1000 or len(neg) < 1000:
        raise SystemExit(f"calibration failed: {len(pos)} vegetation / {len(neg)} roof samples")

    # Equal-error point: the threshold where the two misclassification rates match.
    grid = np.linspace(-0.2, 0.8, 501)
    fn = np.array([(pos < t).mean() for t in grid])    # vegetation called non-vegetation
    fp = np.array([(neg >= t).mean() for t in grid])   # roof called vegetation
    thr = float(grid[int(np.argmin(np.abs(fn - fp)))])
    err = float(fn[int(np.argmin(np.abs(fn - fp)))])

    cal = {
        "ndvi_vegetation_threshold": round(thr, 4),
        "equal_error_rate": round(err, 4),
        "n_vegetation_px": int(len(pos)), "n_roof_px": int(len(neg)),
        "vegetation_ndvi_p10_p50_p90": [round(float(v), 4) for v in np.percentile(pos, [10, 50, 90])],
        "roof_ndvi_p10_p50_p90": [round(float(v), 4) for v in np.percentile(neg, [10, 50, 90])],
        "positives": "LiDAR CHM > 3 m", "negatives": "inside an OSM building footprint, CHM < 1 m",
        "naip_date": "2021-12-02 (winter — a textbook NDVI cutoff would not transfer)",
    }
    print(f"  vegetation (CHM>3m) NDVI p10/p50/p90 : {cal['vegetation_ndvi_p10_p50_p90']}  "
          f"n={len(pos):,}")
    print(f"  roof (OSM footprint) NDVI p10/p50/p90: {cal['roof_ndvi_p10_p50_p90']}  n={len(neg):,}")
    print(f"  --> threshold {thr:.3f} at an equal-error rate of {100*err:.1f} %")
    return cal


PERVIOUS_CODES = (7, 8, 9)   # grass_turf, bare_soil, wetland_marsh


def classify_segments(feat, ndvi_thr):
    """Segment-level rules for the SPECTRAL and FOOTPRINT classes.

    Vegetation STRUCTURE (tree vs shrub) is deliberately not decided here — see
    `apply_canopy_structure` below. What is decided here is the cover type of the ground:
    water, pavement, roof, wetland, turf, bare soil.

    Precedence runs from most-directly-measured to most-inferred: a mapped footprint beats a
    spectral inference.
    """
    n = len(feat["ndvi"])
    cls = np.full(n, NAME_TO_CODE["bare_soil"], dtype=np.uint8)

    veg = feat["ndvi"] >= ndvi_thr

    # bare soil is the default, then each rule overwrites in increasing precedence
    cls[veg] = NAME_TO_CODE["grass_turf"]
    cls[~veg & (feat["brightness"] >= BRIGHT_IMPERVIOUS)] = NAME_TO_CODE["impervious_other"]

    # wetland: vegetated AND sitting at the drainage surface. HAND is a terrain measurement,
    # independent of the imagery, which is what makes this more than an NDVI re-slice.
    cls[veg & (feat["hand"] < WETLAND_HAND_M)] = NAME_TO_CODE["wetland_marsh"]

    # water: strong NIR absorption is the reliable signal; NDVI alone is not
    cls[(feat["nir"] <= WATER_NIR_MAX) & (feat["ndvi"] < ndvi_thr) & (feat["chm"] < SHRUB_HEIGHT_M)] = \
        NAME_TO_CODE["water"]

    # directly-mapped footprints win outright
    cls[feat["road_frac"] >= 0.5] = NAME_TO_CODE["road_paved"]
    cls[feat["building_frac"] >= 0.5] = NAME_TO_CODE["building_roof"]
    return cls


def apply_canopy_structure(painted, chm, cover):
    """Assign tree/shrub per PIXEL from the LiDAR, not per segment.

    Why this is not done inside classify_segments: a segment carries one class, so a segment
    straddling a tree line and a lawn has to pick one, and a mean canopy height over that segment
    clears the 2 m threshold even when half of it is grass. Visual QC at 18 m segments showed
    exactly that — whole suburban blocks painted as forest. The canopy-height model does not need
    the denoising a segment provides: each 2 m CHM cell is already an aggregate over ~140 returns.
    So structure is read at the CHM's own resolution.

    Only PERVIOUS ground is eligible. A tree overhanging a road or a roof does not change what
    the water at 5 m grid scale is flowing over, and letting canopy overwrite a mapped footprint
    would undo the most reliable evidence in the whole classification.
    """
    eligible = np.isin(painted, PERVIOUS_CODES)
    chm = np.nan_to_num(chm, nan=0.0)
    cover = np.nan_to_num(cover, nan=0.0)
    out = painted.copy()
    shrub = eligible & (chm >= SHRUB_HEIGHT_M) & (chm < CANOPY_HEIGHT_M)
    tree = eligible & (chm >= CANOPY_HEIGHT_M) & (cover >= 0.25)
    out[shrub] = NAME_TO_CODE["shrub_scrub"]
    out[tree] = NAME_TO_CODE["tree_canopy"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="site3")
    ap.add_argument("--backend", default="spectral", choices=["spectral", "sam3"])
    ap.add_argument("--calibrate-only", action="store_true")
    ap.add_argument("--max-tiles", type=int, default=None, help="stop early (for a quick check)")
    args = ap.parse_args()

    if args.backend == "sam3":
        raise SystemExit(
            "SAM3 backend is blocked, on two counts and neither is hardware:\n"
            "  1. facebook/sam3 is a gated HF repo — config fetch returns HTTP 401 without an "
            "approved token.\n"
            "  2. Sam3Model ships in transformers 5.x, which requires Python >= 3.10. This "
            "interpreter is 3.9.6, and there is no 4.58 to upgrade to.\n"
            "Run SAM3 from the 3.11 venv instead, writing landcover_0.6m.tif; the 3.9 pipeline "
            "consumes that GeoTIFF unchanged. Use --backend spectral meanwhile — same contract."
        )

    os.makedirs(DATA_DIR, exist_ok=True)
    paths = site_paths(args.site)
    for k in ("rgb", "nir", "ndvi", "chm", "hand", "roads", "buildings"):
        if not os.path.exists(paths[k]):
            raise SystemExit(f"missing input '{k}': {paths[k]}"
                             + ("\n  run segmentation/canopy_lidar.py first" if k in ("chm", "cover") else ""))

    print("=" * 74)
    print(f"NAIP surface-class segmentation — {args.site}, backend={args.backend}")
    print("=" * 74)

    cal_path = os.path.join(DATA_DIR, f"calibration_{args.site}.json")
    cal = calibrate_ndvi(paths)
    with open(cal_path, "w") as fh:
        json.dump(cal, fh, indent=2)
    print(f"  wrote {os.path.relpath(cal_path, PROJ_DIR)}")
    if args.calibrate_only:
        return
    ndvi_thr = cal["ndvi_vegetation_threshold"]

    rgb_src = rasterio.open(paths["rgb"])
    nir_src = rasterio.open(paths["nir"])
    ndvi_src = rasterio.open(paths["ndvi"])
    dst_crs = rgb_src.crs
    print("  loading auxiliary rasters (CHM, canopy cover, HAND) …")
    chm_aux, cov_aux, hand_aux = _Aux(paths["chm"]), _Aux(paths["cover"]), _Aux(paths["hand"])

    road_shapes, bldg_shapes = load_footprints(paths, dst_crs)

    H, W = rgb_src.height, rgb_src.width
    out_path = os.path.join(DATA_DIR, f"landcover_0.6m_{args.site}.tif")
    profile = rgb_src.profile.copy()
    profile.update(count=1, dtype="uint8", compress="lzw", nodata=0, tiled=True,
                   blockxsize=512, blockysize=512)

    rows = list(range(0, H, TILE))
    cols = list(range(0, W, TILE))
    tiles = [(r, c) for r in rows for c in cols]
    if args.max_tiles:
        tiles = tiles[:args.max_tiles]
    print(f"\n[segment] {len(tiles)} tiles of {TILE}x{TILE} px "
          f"over {H}x{W} @ 0.6 m  (NDVI threshold {ndvi_thr:.3f})")

    seg_rows = []
    seg_id_base = 0
    t0 = time.time()

    with rasterio.open(out_path, "w", **profile) as dst:
        for i, (r0, c0) in enumerate(tiles, 1):
            h = min(TILE, H - r0)
            w = min(TILE, W - c0)
            win = Window(c0, r0, w, h)
            wt = window_transform(win, rgb_src.transform)

            rgb = rgb_src.read(window=win).astype(np.float32)          # (3, h, w)
            nir = nir_src.read(1, window=win).astype(np.float32)
            ndvi = ndvi_src.read(1, window=win).astype(np.float32)
            chm = chm_aux.window(wt, (h, w), dst_crs)
            cov = cov_aux.window(wt, (h, w), dst_crs)
            hand = hand_aux.window(wt, (h, w), dst_crs)

            valid = np.isfinite(ndvi) & (rgb.sum(axis=0) > 0)
            if not valid.any():
                dst.write(np.zeros((h, w), np.uint8), 1, window=win)
                continue

            road = rasterize(road_shapes, out_shape=(h, w), transform=wt,
                             fill=0, dtype=np.uint8) if road_shapes else np.zeros((h, w), np.uint8)
            bldg = rasterize(bldg_shapes, out_shape=(h, w), transform=wt,
                             fill=0, dtype=np.uint8) if bldg_shapes else np.zeros((h, w), np.uint8)

            # SLIC over RGB + NIR. 4 bands, not 3: NIR is what separates a shaded roof from a
            # shaded tree, and both look like the same dark grey in visible bands alone.
            stack = np.dstack([rgb[0], rgb[1], rgb[2], nir]) / 255.0
            n_seg = max(1, int(h * w / SLIC_SEGMENT_PX))
            labels = slic(stack, n_segments=n_seg, compactness=SLIC_COMPACTNESS,
                          channel_axis=-1, start_label=0, enforce_connectivity=True)
            n_lab = int(labels.max()) + 1
            flat = labels.reshape(-1)

            counts = np.bincount(flat, minlength=n_lab).astype(np.float32)
            counts_safe = np.maximum(counts, 1.0)

            def seg_mean(a):
                a = np.where(np.isfinite(a), a, 0.0).reshape(-1).astype(np.float32)
                return np.bincount(flat, weights=a, minlength=n_lab) / counts_safe

            bright = (rgb[0] + rgb[1] + rgb[2]) / 3.0
            feat = {
                "ndvi":          seg_mean(ndvi),
                "nir":           seg_mean(nir),
                "brightness":    seg_mean(bright),
                "red":           seg_mean(rgb[0]),
                "green":         seg_mean(rgb[1]),
                "blue":          seg_mean(rgb[2]),
                "chm":           seg_mean(chm),
                "canopy_cover":  seg_mean(cov),
                "hand":          seg_mean(hand),
                "road_frac":     seg_mean(road.astype(np.float32)),
                "building_frac": seg_mean(bldg.astype(np.float32)),
                "area_m2":       counts * 0.36,
            }
            codes = classify_segments(feat, ndvi_thr)

            out = apply_canopy_structure(codes[labels].astype(np.uint8), chm, cov)
            out[~valid] = 0
            dst.write(out, 1, window=win)

            # Per-segment class for the table = the modal class of that segment's pixels AFTER
            # the pixel-level canopy override, so segments.csv describes what was actually
            # written to the raster rather than the pre-override segment call.
            ncls = int(max(CLASSES)) + 1
            tab = np.zeros((n_lab, ncls), dtype=np.int32)
            fo = out.reshape(-1)
            for c in range(1, ncls):
                m = fo == c
                if m.any():
                    tab[:, c] = np.bincount(flat[m], minlength=n_lab)
            codes = tab.argmax(axis=1).astype(np.uint8)

            for j in range(n_lab):
                if counts[j] < 20:            # sub-7 m2 slivers carry no usable statistics
                    continue
                seg_rows.append({
                    "segment_id": seg_id_base + j,
                    "tile_row": r0, "tile_col": c0,
                    "class_code": int(codes[j]), "class_name": CLASSES[int(codes[j])],
                    **{k: round(float(v[j]), 4) for k, v in feat.items()},
                })
            seg_id_base += n_lab

            if i % 5 == 0 or i == len(tiles):
                el = time.time() - t0
                print(f"  [{i}/{len(tiles)}] {len(seg_rows):,} segments  "
                      f"[{el:.0f}s, ~{el/i*(len(tiles)-i):.0f}s left]", flush=True)

    for _s in (rgb_src, nir_src, ndvi_src):
        _s.close()

    import pandas as pd
    df = pd.DataFrame(seg_rows)
    seg_path = os.path.join(DATA_DIR, f"segments_{args.site}.csv")
    df.to_csv(seg_path, index=False)

    counts = df["class_name"].value_counts()
    areas = df.groupby("class_name")["area_m2"].sum()
    total_area = float(areas.sum())
    summary = {
        "site": args.site, "backend": args.backend,
        "naip": os.path.relpath(paths["rgb"], PROJ_DIR),
        "n_segments": int(len(df)), "n_tiles": len(tiles),
        "ndvi_vegetation_threshold": ndvi_thr,
        "class_area_fraction": {k: round(float(v) / total_area, 5) for k, v in areas.items()},
        "class_segment_count": {k: int(v) for k, v in counts.items()},
        "total_classified_area_km2": round(total_area / 1e6, 3),
        "wall_s": round(time.time() - t0, 1),
    }
    sp = os.path.join(DATA_DIR, f"segmentation_summary_{args.site}.json")
    with open(sp, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "-" * 74)
    print(f"  {len(df):,} segments over {total_area/1e6:.2f} km2")
    for k in sorted(areas.index, key=lambda x: -areas[x]):
        print(f"    {k:18s} {100*areas[k]/total_area:5.1f} % of area   "
              f"{int(counts[k]):>7,} segments")
    print(f"  wrote {os.path.relpath(out_path, PROJ_DIR)}")
    print(f"  wrote {os.path.relpath(seg_path, PROJ_DIR)}")
    print(f"  wrote {os.path.relpath(sp, PROJ_DIR)}")
    print("-" * 74)


if __name__ == "__main__":
    main()
