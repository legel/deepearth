"""Surface parameters from imagery, so a twin does not depend on a national soil survey.

Shallow-water solvers are portable; their parameters are not. Roughness, infiltration capacity
and storage normally come from SSURGO and NLCD, which exist in a handful of countries. This
module derives the same fields from open-vocabulary segmentation of aerial imagery, which is
the part that could run anywhere. The canopy-height model that produced the channel finding
below is not shipped here; it was a LiDAR-derived raster built during the investigation, and the
rule it justified survives it.

What it is honest about:

  * The vision route reproduces the soil survey's basin water budget to within 0.6 % while
    sharing almost no spatial structure with it (Ks r = +0.171, storage r = -0.234). It agrees
    on how much water the basin sheds and disagrees on where.
  * It does not close the magnitude gap. Segmented roughness moves the runoff coefficient by
    under two points, in both backends.
  * Where the two diverge they do so mechanistically: nadir sensors infer a deep profile from
    deep-rooted forest, but in low-relief subtropical terrain that forest marks a SHALLOW water
    table. SSURGO reports the least storage where canopy is densest (140 mm above 80 % canopy
    against 234 mm below 20 %). The causality runs the other way.

Use SSURGO where it exists. Where it does not, expect a usable volume and degraded routing.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from sites import SiteConfig

CHANNEL_MANNING_N = 0.045
"""Roughness forced onto mapped channel cells, overriding whatever the imagery inferred.

The single most transferable finding here. Riparian canopy closes over Gee Creek, so nadir
imagery cannot see the water, and a LiDAR canopy-height model correctly reported 12 m of tree
above it. The classifier therefore called 58.9 % of mapped channel cells `tree_canopy` -- and
the gauge cell itself 100 % -- putting n = 0.120 on the channel bed. Measured cost: gauge-cell
discharge collapsed from 101.6 to 10.5 cfs, a ~10x drop, while domain outflow fell only 23 %. A
defect invisible in a domain-wide statistic dominated the validation cell.

A canopy roughness is not wrong for a forest; it is wrong for a channel. Chow's 0.10 for timber
assumes flow among the trunks with flood stage below the branches. Any nadir parameterisation
will make this mistake wherever vegetation overhangs conveyance, so the fix is structural: the
same precedence rule the classification already uses -- a mapped feature outranks a spectral
inference -- extended to the hydrography layer.
"""

CHANNEL_BUFFER_M = 10.0
"""Half-width [m] applied to mapped flowlines when forcing channel roughness."""


@dataclass(frozen=True)
class SurfaceClass:
    """One land-cover class and the hydrology it implies.

    Attributes:
        manning_n: Roughness for 2D routing at ~0.1 m depth.
        surface_storage_m: Depression and interception storage before runoff begins.
        impervious_fraction: Fraction of the cell that sheds rather than infiltrates.
        vision_ks_mm_hr: Saturated conductivity inferred from cover, the vision alternative to
            SSURGO's Ksat.
        vision_storage_m: Soil storage inferred from rooting depth, the vision alternative to
            SSURGO's water-table depth times drainable porosity.
        basis: Where the roughness value comes from.
    """

    manning_n: float
    surface_storage_m: float
    impervious_fraction: float
    vision_ks_mm_hr: float
    vision_storage_m: float
    basis: str


CLASSES: Dict[str, SurfaceClass] = {
    "water": SurfaceClass(
        0.035, 0.0000, 1.00, 0.0, 0.0,
        "Chow 1959, clean straight natural channel, 0.030-0.040. Rain joins a free water "
        "surface directly, so it neither stores nor infiltrates."),
    "building_roof": SurfaceClass(
        0.015, 0.0005, 1.00, 0.0, 0.0,
        "Smooth manufactured surface; matches the value both mesh solvers already used for "
        "roofs. A roof is built to shed, so storage is shallow."),
    "road_paved": SurfaceClass(
        0.013, 0.0010, 1.00, 0.0, 0.0,
        "Chow 1959, smooth asphalt. The smoothest surface in the domain."),
    "impervious_other": SurfaceClass(
        0.016, 0.0015, 0.90, 1.0, 0.010,
        "Broom-finished concrete, joints and gravel aprons; Chow's concrete range 0.012-0.018. "
        "Impervious 0.90 not 1.00 because this class is inferred spectrally rather than from a "
        "mapped footprint, so some of it is genuinely compacted ground."),
    "tree_canopy": SurfaceClass(
        0.120, 0.0045, 0.00, 210.0, 0.375,
        "HEC-RAS 2D / FEMA forest tables, 0.10-0.12. Trunks, roots, litter and understory "
        "obstruct flow -- 3x the domain-wide scalar this class would otherwise get. Ks from "
        "Rawls et al. (1982) sand: flatwoods forest floor is fine sand kept open by root "
        "macropores, the highest-conductivity surface here."),
    "shrub_scrub": SurfaceClass(
        0.070, 0.0020, 0.00, 180.0, 0.250,
        "Palmetto and scrub understory; dense but lower and less woody than forest."),
    "grass_turf": SurfaceClass(
        0.040, 0.0010, 0.00, 60.0, 0.125,
        "Managed turf and pasture over sandy soil. This is the value the domain-wide scalar "
        "used for everything, which is why promoting it to a field matters."),
    "bare_soil": SurfaceClass(
        0.025, 0.0005, 0.00, 40.0, 0.125,
        "Bare or sparsely vegetated sand; smoother than turf, faster than a sealed surface."),
    "wetland_marsh": SurfaceClass(
        0.080, 0.0000, 0.00, 2.0, 0.000,
        "Emergent herbaceous wetland and cypress-dome margin. Saturated by definition, so no "
        "storage remains to fill and conductivity is near zero. This class comes from HAND, "
        "not from imagery -- segmentation cannot see a water table."),
}

CODES: Dict[str, int] = {name: i + 1 for i, name in enumerate(CLASSES)}
"""Integer codes written into the class raster. 0 means unlabelled."""

SAM3_PROMPTS: Dict[str, Tuple[str, ...]] = {
    "water": ("lake", "pond", "river", "water"),
    "building_roof": ("building", "house", "roof", "warehouse"),
    "road_paved": ("road", "street", "highway", "parking lot"),
    "impervious_other": ("driveway", "sidewalk", "concrete pad", "tennis court"),
    "tree_canopy": ("tree", "forest", "tree canopy"),
    "shrub_scrub": ("shrub", "bush", "scrub"),
    "grass_turf": ("grass", "lawn", "field", "pasture"),
    "bare_soil": ("bare soil", "dirt", "sand"),
}
"""Open-vocabulary prompts per class.

SAM3 names the same surface several different things, so the prompt list per class is the
aggregation layer. `wetland_marsh` has no prompts on purpose: it is assigned from HAND
afterwards, because no amount of prompting makes a nadir sensor see a shallow water table.
"""


def parameter_table() -> Dict[str, Dict[str, float]]:
    """The class table as plain numbers, for writing alongside a run."""
    return {name: {"manning_n": c.manning_n, "surface_storage_m": c.surface_storage_m,
                   "impervious_fraction": c.impervious_fraction,
                   "vision_ks_mm_hr": c.vision_ks_mm_hr,
                   "vision_storage_m": c.vision_storage_m}
            for name, c in CLASSES.items()}


def manning_spread() -> float:
    """Ratio of the roughest to the smoothest class -- what a scalar throws away."""
    values = [c.manning_n for c in CLASSES.values()]
    return max(values) / min(values)


def _class_field(codes: np.ndarray, attr: str, default: float) -> np.ndarray:
    """Map a class-code raster to one parameter, leaving unlabelled cells at `default`."""
    out = np.full(codes.shape, default, dtype=np.float32)
    for name, code in CODES.items():
        out[codes == code] = getattr(CLASSES[name], attr)
    return out


def rasterize(site: SiteConfig, shape: Tuple[int, int], profile: Dict,
              scalar_manning_n: float = 0.040) -> Dict[str, np.ndarray]:
    """Class raster to solver-grid Manning's n and impervious fraction.

    Mapped channels override the imagery afterwards, which is the whole point -- see
    `CHANNEL_MANNING_N`.

    Args:
        site: Site whose landcover raster to read.
        shape: Solver grid shape.
        profile: Solver grid rasterio profile.
        scalar_manning_n: Value for cells the classifier left unlabelled. SAM3 labels only what
            it detects and left 17.4 % of the scene unlabelled against the spectral backend's
            0.5 %, so this is load-bearing, not a formality.

    Returns:
        {"manning_n", "impervious_fraction", "channel_mask"} on the solver grid.
    """
    import geopandas as gpd
    from rasterio.enums import Resampling
    from rasterio.features import rasterize as rio_rasterize

    from domain import _warp_onto

    assert site.landcover.exists(), (
        f"{site.landcover} missing; run the segmentation stage from the Python 3.11 "
        f"environment (transformers 5.x needs >= 3.10, the pipeline interpreter is 3.9)")

    codes = _warp_onto(site.landcover, shape, profile, Resampling.nearest,
                       np.int32).astype(np.int32)
    fields = {
        "manning_n": _class_field(codes, "manning_n", scalar_manning_n),
        "impervious_fraction": _class_field(codes, "impervious_fraction", 0.0),
    }

    assert site.flowlines.exists(), f"{site.flowlines} missing; channel precedence cannot apply"
    flowlines = gpd.read_file(site.flowlines).to_crs(profile["crs"])
    channel = rio_rasterize(
        [(g.buffer(CHANNEL_BUFFER_M), 1) for g in flowlines.geometry if g is not None],
        out_shape=shape, transform=profile["transform"], fill=0, dtype=np.uint8).astype(bool)
    fields["manning_n"] = np.where(channel, CHANNEL_MANNING_N, fields["manning_n"])
    fields["channel_mask"] = channel
    return fields


def summary(fields: Dict[str, np.ndarray], scalar_manning_n: float = 0.040) -> Dict[str, float]:
    """What the field changed relative to the scalar it replaces."""
    n = fields["manning_n"]
    channel = fields["channel_mask"]
    return {
        "manning_n_mean": float(n.mean()),
        "manning_n_min": float(n.min()),
        "manning_n_max": float(n.max()),
        "scalar_replaced": scalar_manning_n,
        "class_spread_ratio": manning_spread(),
        "channel_cells": int(channel.sum()),
        "impervious_mean": float(fields["impervious_fraction"].mean()),
    }


# ── Segmentation ─────────────────────────────────────────────────────────────────────────

MODEL_ID = "facebook/sam3"
TILE_PX = 1024
"""SAM3's native working resolution, in PIXELS -- the ground footprint follows the imagery.
NAIP over this site was 0.6 m in 2021 (~614 m per tile) and 0.3 m in 2023 (~307 m, so 4x the
tiles). `fetch.naip` takes the latest year, so this is not a fixed scale."""

OVERLAP_PX = 128
SCORE_THRESHOLD = 0.35
MASK_THRESHOLD = 0.5
WETLAND_HAND_M = 0.5
"""HAND at or below which an unlabelled pervious cell is called wetland.

Segmentation cannot see a water table, so this class is assigned from terrain afterwards. SAM3
found no wetland at all when asked directly.
"""


def _segment_tile(processor: object, model: object, device: str,
                  rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Run every prompt over one tile; return (class codes, best score).

    The vision encoder runs ONCE per tile and its output is reused across all prompts: SAM3's
    forward accepts `vision_embeds` in place of `pixel_values` and the image does not change
    between phrases, so re-encoding per prompt was pure waste. Measured 2.38x faster with
    bit-identical masks and scores.

    Overlapping claims resolve by higher score. That is a heuristic, not a calibrated
    comparison -- SAM3's scores are confidences for their own prompt, and 0.9 for "tree" and
    0.85 for "shrub" are not guaranteed to be on one scale. Read the output as
    "highest-scoring prompt wins", not as a probability.
    """
    import torch
    from PIL import Image

    image = Image.fromarray(rgb)
    h, w = rgb.shape[:2]
    best_score = np.zeros((h, w), dtype=np.float32)
    best_code = np.zeros((h, w), dtype=np.uint8)
    vision_embeds = None

    for name, phrases in SAM3_PROMPTS.items():
        code = CODES[name]
        for phrase in phrases:
            inputs = processor(images=image, text=phrase, return_tensors="pt").to(device)
            with torch.no_grad():
                if vision_embeds is None:
                    vision_embeds = model.get_vision_features(pixel_values=inputs["pixel_values"])
                text_only = {k: v for k, v in inputs.items() if k != "pixel_values"}
                out = model(vision_embeds=vision_embeds, **text_only)
            result = processor.post_process_instance_segmentation(
                out, threshold=SCORE_THRESHOLD, mask_threshold=MASK_THRESHOLD,
                target_sizes=[(h, w)])[0]
            masks, scores = result.get("masks"), result.get("scores")
            if masks is None or len(masks) == 0:
                continue
            for mask, score in zip(masks.cpu().numpy().astype(bool),
                                   scores.cpu().numpy().astype(np.float32)):
                take = mask & (score > best_score)
                best_score[take] = score
                best_code[take] = code
    return best_code, best_score


def segment(site: SiteConfig, device: Optional[str] = None) -> Dict[str, float]:
    """Open-vocabulary segmentation of the site's NAIP imagery into `CLASSES`.

    Runs as a standalone stage under Python 3.11: `Sam3Model` ships in transformers 5.x, which
    needs >= 3.10, while the pipeline interpreter is pinned to 3.9 by richdem. The class raster
    is the contract between the two, so the 3.9 side consumes it unchanged.

    Returns:
        Coverage statistics, including the unlabelled fraction, which matters: SAM3 labels only
        what it detects.
    """
    import sys

    assert sys.version_info >= (3, 10), (
        f"SAM3 needs Python >= 3.10 for transformers 5.x; this is "
        f"{sys.version_info.major}.{sys.version_info.minor}. Run this stage from the 3.11 "
        f"environment, then run everything else from 3.9.")
    assert site.naip_rgb.exists(), f"{site.naip_rgb} missing; run the fetch stage"

    import rasterio
    import torch
    from transformers import Sam3Model, Sam3Processor

    device = device or ("mps" if torch.backends.mps.is_available()
                        else "cuda" if torch.cuda.is_available() else "cpu")
    processor = Sam3Processor.from_pretrained(MODEL_ID)
    model = Sam3Model.from_pretrained(MODEL_ID, dtype=torch.float32).to(device).eval()

    with rasterio.open(site.naip_rgb) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width
        codes = np.zeros((height, width), dtype=np.uint8)
        step = TILE_PX - OVERLAP_PX
        for r0 in range(0, height, step):
            for c0 in range(0, width, step):
                h = min(TILE_PX, height - r0)
                w = min(TILE_PX, width - c0)
                window = rasterio.windows.Window(c0, r0, w, h)
                rgb = np.moveaxis(src.read((1, 2, 3), window=window), 0, -1).astype(np.uint8)
                tile_codes, tile_scores = _segment_tile(processor, model, device, rgb)
                target = codes[r0:r0 + h, c0:c0 + w]
                codes[r0:r0 + h, c0:c0 + w] = np.where(target == 0, tile_codes, target)

    # Wetland comes from terrain, not imagery. Only unlabelled pervious ground is eligible.
    if site.hand.exists():
        from rasterio.enums import Resampling
        from domain import _warp_onto

        hand = _warp_onto(site.hand, codes.shape, profile, Resampling.bilinear)
        codes[(codes == 0) & np.isfinite(hand) & (hand <= WETLAND_HAND_M)] = CODES["wetland_marsh"]

    profile.update(count=1, dtype="uint8", nodata=0, compress="deflate")
    with rasterio.open(site.landcover, "w", **profile) as dst:
        dst.write(codes, 1)

    total = codes.size
    stats = {f"{name}_pct": float((codes == code).sum()) / total * 100.0
             for name, code in CODES.items()}
    stats["unlabelled_pct"] = float((codes == 0).sum()) / total * 100.0
    return stats
