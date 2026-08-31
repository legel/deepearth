"""
SAM3 open-vocabulary segmentation of NAIP
=========================================
Runs Meta's SAM3 over the NAIP orthophoto with one text prompt per surface class and writes the
same `landcover_0.6m_<site>.tif` class raster the spectral backend produces, so everything
downstream — `rasterize_parameters.py`, the parameter table, the solver arms — consumes it
unchanged. That is the whole point of having made the class raster the contract rather than the
segment objects.

RUNS UNDER THE 3.11 VENV, NOT THE PIPELINE INTERPRETER
-------------------------------------------------------
`Sam3Model` ships in transformers 5.x, which requires Python >= 3.10. The pipeline interpreter is
3.9.6 (richdem does not build on 3.11, which is why the pipeline is pinned there). So this one
stage runs separately:

    ./.venv/bin/python segmentation/sam3_segment.py --site site3

and the 3.9 pipeline then picks up the GeoTIFF. Verified present in the venv: transformers
5.16.1 with Sam3Model / Sam3Processor / Sam3VideoModel.

ACCESS
------
`facebook/sam3` is a gated repo requiring acceptance of Meta's terms. Access was granted for this
project on 2026-08-28 and the weights are cached locally. On a fresh machine, request access at
huggingface.co/facebook/sam3, then authenticate once:

    ./.venv/bin/python -c "from huggingface_hub import login; login()"

(The venv's console scripts — `hf`, `huggingface-cli` — carry a dead shebang from the repository
reorganisation and fail; `python -c` and `python -m` work.) Without a token `from_pretrained`
raises `OSError: You are trying to access a gated repo`, and this module turns that into the
three unblocking steps rather than a traceback.

Hardware is not a constraint — 3.44 GB of weights against 17.2 GB of unified memory, MPS
available. The MPS out-of-memory result recorded elsewhere in this project was the 8.67M-edge
mesh GNN, an unrelated workload.

HOW THE OPEN VOCABULARY IS MAPPED ONTO CLASSES
-----------------------------------------------
SAM3 names what it sees; the solver needs a parameter, and a parameter is attached to a class.
Each class is queried with several natural phrasings and the results merged, because "roof",
"rooftop" and "building" will not return identical masks and the union is what the class means.

Where two classes claim the same pixel, the higher detection score wins, with one exception that
is not negotiable and is the lesson this project already paid for once: **a mapped road or
building footprint outranks any imagery inference.** OSM footprints are direct evidence and the
spectral backend already defers to them; SAM3 gets the same treatment, applied after its own
masks are merged. Without it, riparian canopy overhanging a road would once again put forest
roughness on pavement.

Note that the channel override lives downstream in `rasterize_parameters.py`, so it protects this
backend automatically — the creek does not need re-fixing here.

Usage:
    ./.venv/bin/python segmentation/sam3_segment.py --site site3 --dry-run   # no weights needed
    ./.venv/bin/python segmentation/sam3_segment.py --site site3 --max-tiles 4
    ./.venv/bin/python segmentation/sam3_segment.py --site site3
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
from rasterio.features import rasterize

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")

# Kept byte-identical to segment_naip.CLASSES on purpose — the two backends must produce the same
# encoding or nothing downstream can be compared. Not imported, because segment_naip pulls in
# geopandas/skimage that the venv does not need for this stage.
CLASSES = {
    0: "nodata", 1: "water", 2: "building_roof", 3: "road_paved", 4: "impervious_other",
    5: "tree_canopy", 6: "shrub_scrub", 7: "grass_turf", 8: "bare_soil", 9: "wetland_marsh",
}
NAME_TO_CODE = {v: k for k, v in CLASSES.items()}

# One entry per class, several phrasings each. SAM3's vocabulary is open, so these are prompts
# rather than labels — the mapping from what it says back to what the solver needs is here.
PROMPTS = {
    "water":            ["water", "pond", "lake", "river"],
    "building_roof":    ["roof", "rooftop", "building"],
    "road_paved":       ["road", "street", "paved road"],
    "impervious_other": ["parking lot", "sidewalk", "driveway", "concrete pavement"],
    "tree_canopy":      ["tree", "tree canopy", "forest"],
    "shrub_scrub":      ["shrub", "bush", "scrub vegetation"],
    "grass_turf":       ["grass", "lawn", "turf"],
    "bare_soil":        ["bare soil", "dirt", "sand"],
    "wetland_marsh":    ["marsh", "wetland", "swamp"],
}

TILE = 1024          # SAM3's native working resolution; 0.6 m NAIP -> ~614 m per tile
OVERLAP = 64         # px, so objects on a tile seam are not cut in half by the tiling itself
SCORE_THRESHOLD = 0.3
MASK_THRESHOLD = 0.5

MODEL_ID = "facebook/sam3"


def require_venv():
    if sys.version_info < (3, 10):
        sys.exit(
            f"This stage needs Python >= 3.10 (running {sys.version_info.major}."
            f"{sys.version_info.minor}). Sam3Model ships in transformers 5.x, which the 3.9.6 "
            f"pipeline interpreter cannot install.\n"
            f"  Run:  ./.venv/bin/python segmentation/sam3_segment.py ...\n"
            f"The 3.9 pipeline then consumes the GeoTIFF this writes, unchanged."
        )


def site_paths(site):
    if site != "site3":
        raise SystemExit("only site3 has NAIP fetched today")
    s = os.path.join(PROJ_DIR, "site3_gee_creek")
    return {
        "rgb":       os.path.join(s, "imagery", "data", "naip_2021_RGB.tif"),
        "roads":     os.path.join(s, "infrastructure", "data", "roads.geojson"),
        "buildings": os.path.join(s, "infrastructure", "data", "buildings.geojson"),
    }


def load_model(device):
    """Import and load SAM3, turning the gated-repo failure into an actionable message."""
    try:
        from transformers import Sam3Processor, Sam3Model
    except ImportError as e:
        sys.exit(f"transformers 5.x with SAM3 not importable in this interpreter: {e}\n"
                 f"  ./.venv/bin/python -m pip install 'transformers>=5.0'")
    import torch
    try:
        processor = Sam3Processor.from_pretrained(MODEL_ID)
        model = Sam3Model.from_pretrained(MODEL_ID, dtype=torch.float32).to(device).eval()
    except OSError as e:
        msg = str(e)
        if "gated repo" in msg or "401" in msg:
            sys.exit(
                "SAM3 weights are gated and this environment has no approved token.\n"
                f"  1. Accept the terms at https://huggingface.co/{MODEL_ID} "
                "(manual review by Meta)\n"
                "  2. Create a read token at https://huggingface.co/settings/tokens\n"
                "  3. export HF_TOKEN=hf_...   (or: huggingface-cli login)\n"
                "Everything else is already in place — transformers 5.16.1 is installed in this "
                "venv and the hardware is adequate (3.44 GB weights, 17.2 GB unified memory)."
            )
        raise
    return processor, model


def load_footprint_masks(paths, crs):
    """OSM roads/buildings rasterised once over the whole scene.

    Uses floodtwin.physics buffer widths so this backend's road footprint is the same one the
    solver's impervious mask and the spectral backend already use.
    """
    import geopandas as gpd
    sys.path.insert(0, os.path.dirname(PROJ_DIR))
    from floodtwin.physics import ROAD_BUFFER_M, ROAD_BUFFER_DEFAULT_M

    roads = gpd.read_file(paths["roads"]).to_crs(crs)
    bldgs = gpd.read_file(paths["buildings"]).to_crs(crs)
    road_shapes = [
        (r.geometry.buffer(ROAD_BUFFER_M.get(str(r.get("highway")), ROAD_BUFFER_DEFAULT_M)), 1)
        for _, r in roads.iterrows() if r.geometry is not None
    ]
    bldg_shapes = [(g, 1) for g in bldgs.geometry if g is not None]
    print(f"  footprints: {len(road_shapes)} road segments, {len(bldg_shapes)} buildings")
    return road_shapes, bldg_shapes


def segment_tile(processor, model, device, rgb_hwc):
    """Run every prompt over one tile; return (class_code_map, best_score_map).

    The vision encoder runs ONCE per tile and its output is reused across all prompts. SAM3's
    forward accepts `vision_embeds` in place of `pixel_values`, and the image does not change
    between prompts, so re-encoding it for each of the ~31 phrases was pure waste. Measured on a
    real NAIP tile: 2.38x faster, with bit-identical masks and scores on every prompt tested.

    Overlapping claims are resolved by taking the higher detection score. That is a heuristic,
    not a calibrated comparison: SAM3's scores are confidences for its own prompt, and there is
    no guarantee that 0.9 for "tree" and 0.85 for "shrub" are on the same scale. It is the
    natural choice and it is what is done here, but the class map should be read as
    "highest-scoring prompt wins", not as a probabilistic assignment.
    """
    import torch
    from PIL import Image
    img = Image.fromarray(rgb_hwc)
    h, w = rgb_hwc.shape[:2]
    best_score = np.zeros((h, w), dtype=np.float32)
    best_code = np.zeros((h, w), dtype=np.uint8)

    vision_embeds = None
    for cls_name, phrases in PROMPTS.items():
        code = NAME_TO_CODE[cls_name]
        for phrase in phrases:
            inputs = processor(images=img, text=phrase, return_tensors="pt").to(device)
            with torch.no_grad():
                if vision_embeds is None:
                    vision_embeds = model.get_vision_features(
                        pixel_values=inputs["pixel_values"])
                text_inputs = {k: v for k, v in inputs.items() if k != "pixel_values"}
                out = model(vision_embeds=vision_embeds, **text_inputs)
            res = processor.post_process_instance_segmentation(
                out, threshold=SCORE_THRESHOLD, mask_threshold=MASK_THRESHOLD,
                target_sizes=[(h, w)],
            )[0]
            masks = res.get("masks")
            scores = res.get("scores")
            if masks is None or len(masks) == 0:
                continue
            masks = masks.cpu().numpy().astype(bool)
            scores = scores.cpu().numpy().astype(np.float32)
            for m_i, s_i in zip(masks, scores):
                take = m_i & (s_i > best_score)
                best_score[take] = s_i
                best_code[take] = code
    return best_code, best_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="site3")
    ap.add_argument("--max-tiles", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    ap.add_argument("--dry-run", action="store_true",
                    help="exercise tiling, footprint precedence and GeoTIFF writing with a "
                         "synthetic class map — needs no weights, so it is runnable today")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore per-tile checkpoints and recompute every tile")
    ap.add_argument("--out-suffix", default="_sam3",
                    help="written as landcover_0.6m_<site><suffix>.tif; the default keeps the "
                         "spectral backend's own output intact so the two can be compared")
    args = ap.parse_args()

    if not args.dry_run:
        require_venv()

    paths = site_paths(args.site)
    if not os.path.exists(paths["rgb"]):
        raise SystemExit(f"missing NAIP: {paths['rgb']}")

    print("=" * 74)
    print(f"SAM3 open-vocabulary segmentation — {args.site}"
          + ("   [DRY RUN, no weights]" if args.dry_run else ""))
    print("=" * 74)

    processor = model = device = None
    if not args.dry_run:
        import torch
        device = ("mps" if (args.device in ("auto", "mps") and torch.backends.mps.is_available())
                  else "cpu")
        print(f"  device: {device}")
        t0 = time.time()
        processor, model = load_model(device)
        print(f"  {MODEL_ID} loaded in {time.time()-t0:.0f}s")

    src = rasterio.open(paths["rgb"])
    H, W = src.height, src.width
    print(f"  NAIP: {H}x{W} @ {src.res[0]:.2f} m  {src.crs}")

    road_shapes, bldg_shapes = load_footprint_masks(paths, src.crs)

    step = TILE - OVERLAP
    tiles = [(r, c) for r in range(0, H, step) for c in range(0, W, step)]
    if args.max_tiles:
        tiles = tiles[:args.max_tiles]
    print(f"  {len(tiles)} tiles of {TILE}x{TILE} px (overlap {OVERLAP})"
          f"  ~{TILE*0.6:.0f} m each\n")

    out_path = os.path.join(DATA_DIR, f"landcover_0.6m_{args.site}{args.out_suffix}.tif")
    profile = src.profile.copy()
    profile.update(count=1, dtype="uint8", compress="lzw", nodata=0,
                   tiled=True, blockxsize=512, blockysize=512)

    # Accumulate into a full-scene score map so overlapping tiles resolve by confidence rather
    # than by whichever tile was written last.
    code_scene = np.zeros((H, W), dtype=np.uint8)
    score_scene = np.zeros((H, W), dtype=np.float32)

    # Per-tile checkpoints. A full scene is ~121 tiles and over an hour of inference, and long
    # background jobs in this environment have been killed mid-run before with no traceback —
    # the same failure that made cache_bbox_points.py and the mesh build checkpoint per tile.
    # Each tile is written atomically as it completes, so a kill costs one tile and a re-run
    # skips everything already done.
    ckpt_dir = os.path.join(DATA_DIR, f"sam3_checkpoints_{args.site}")
    os.makedirs(ckpt_dir, exist_ok=True)

    t0 = time.time()
    n_done = n_cached = 0
    for i, (r0, c0) in enumerate(tiles, 1):
        h = min(TILE, H - r0)
        w = min(TILE, W - c0)
        if h < 32 or w < 32:
            continue

        ck = os.path.join(ckpt_dir, f"tile_{r0:05d}_{c0:05d}.npz")
        if os.path.exists(ck) and not args.no_resume and not args.dry_run:
            try:
                d = np.load(ck)
                code, score = d["code"], d["score"]
                n_cached += 1
            except Exception:
                os.remove(ck)          # truncated by a kill mid-write; recompute
                code = score = None
        else:
            code = score = None

        if code is None:
            win = Window(c0, r0, w, h)
            rgb = src.read(window=win).transpose(1, 2, 0)
            if rgb.shape[2] > 3:
                rgb = rgb[:, :, :3]

            if args.dry_run:
                # Deterministic stand-in so the plumbing is exercised without weights.
                code = np.full((h, w), NAME_TO_CODE["grass_turf"], dtype=np.uint8)
                score = np.full((h, w), 0.5, dtype=np.float32)
            else:
                code, score = segment_tile(processor, model, device, rgb.astype(np.uint8))
                tmp = ck + ".tmp.npz"
                np.savez_compressed(tmp, code=code, score=score)
                os.replace(tmp, ck)    # atomic on the same filesystem
                n_done += 1

        sub_s = score_scene[r0:r0 + h, c0:c0 + w]
        sub_c = code_scene[r0:r0 + h, c0:c0 + w]
        take = score > sub_s
        sub_s[take] = score[take]
        sub_c[take] = code[take]

        if i % 5 == 0 or i == len(tiles):
            el = time.time() - t0
            rate = el / max(n_done, 1)
            left = (len(tiles) - i) * rate
            print(f"  [{i}/{len(tiles)}]  {n_done} computed, {n_cached} from checkpoint  "
                  f"{el/60:.1f} min elapsed, ~{left/60:.0f} min left", flush=True)

    # ── mapped footprints outrank the model ───────────────────────────────────
    # The lesson this project already paid for: direct evidence beats an imagery inference.
    # Applied after the merge so it cannot be overwritten by a later tile.
    if road_shapes:
        road = rasterize(road_shapes, out_shape=(H, W), transform=src.transform,
                         fill=0, dtype=np.uint8).astype(bool)
        code_scene[road] = NAME_TO_CODE["road_paved"]
    if bldg_shapes:
        bldg = rasterize(bldg_shapes, out_shape=(H, W), transform=src.transform,
                         fill=0, dtype=np.uint8).astype(bool)
        code_scene[bldg] = NAME_TO_CODE["building_roof"]

    # All three bands, not just band 1: a pixel that is genuinely dark in red but carries data
    # in green or blue is valid imagery, and testing band 1 alone would discard it. Matches the
    # spectral backend's own `rgb.sum(axis=0) > 0` test so the two class maps have the same
    # footprint and remain comparable.
    valid = src.read().sum(axis=0) > 0
    code_scene[~valid] = 0
    src.close()

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(code_scene, 1)

    counts = {CLASSES[c]: int((code_scene == c).sum()) for c in CLASSES}
    total = sum(v for k, v in counts.items() if k != "nodata")
    summary = {
        "site": args.site, "backend": "sam3" if not args.dry_run else "sam3-dryrun",
        "model": MODEL_ID, "tiles": len(tiles), "tile_px": TILE, "overlap_px": OVERLAP,
        "score_threshold": SCORE_THRESHOLD, "mask_threshold": MASK_THRESHOLD,
        "prompts": PROMPTS,
        "class_area_fraction": {k: round(v / total, 5) for k, v in counts.items()
                                if k != "nodata" and total},
        "wall_s": round(time.time() - t0, 1),
        "footprint_precedence": "OSM roads and buildings overwrite the model's own call",
        "tiles_computed": n_done, "tiles_from_checkpoint": n_cached,
        "vision_embeds_reused_per_tile": True,
        "class_assignment": ("highest-scoring prompt wins; SAM3 scores are per-prompt "
                             "confidences and are not calibrated across prompts"),
    }
    sp = os.path.join(DATA_DIR, f"sam3_summary_{args.site}.json")
    with open(sp, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "-" * 74)
    for k, v in sorted(summary["class_area_fraction"].items(), key=lambda kv: -kv[1]):
        if v > 1e-4:
            print(f"    {k:<18} {100*v:5.1f} %")
    print(f"  wrote {os.path.relpath(out_path, PROJ_DIR)}")
    print(f"  wrote {os.path.relpath(sp, PROJ_DIR)}")
    print("\n  next:  python3 segmentation/rasterize_parameters.py --site "
          f"{args.site} --landcover {os.path.basename(out_path)}")
    print("-" * 74)


if __name__ == "__main__":
    main()
