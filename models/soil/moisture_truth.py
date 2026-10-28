"""Ground truth for SOIL MOISTURE: the Simulator's topographic water balance held inside what the soil survey and the
wetlands inventory say each place's ground can be.

The water model drains freely at its base and has no groundwater, so ground beside and inside a tidal marsh reads dry
(a tidal-marsh site, 2026-10-27: USDA says "Very poorly drained, 95 % hydric, tidally flooded" where it read dry). The
model's pattern is kept: a plantable cell becomes min(ceiling, max(model, floor + w * model / (floor + w))), w = LIFT_W,
on the relative scale the product already uses (0 wilting point, 1 field capacity, 2 saturation;
exposure.water_modes.relative_wetness): ground at or above floor + w keeps its model value, ground below is lifted into
[floor, floor + w] with its pattern muted, and the ceiling only clips. The polygon floors and ceilings are feathered over
FEATHER_M first, since a survey line is only that accurate.

THE BANDS. A floor is the safety-critical half: a wet soil may never read dry. They follow the NRCS natural drainage
classes (Soil Survey Manual, USDA Handbook 18, 2017, ch. 3 "Drainage Class"):
  very poorly drained      water at or near the surface during much of the growing season        floor 1.6
  poorly drained           saturated at or near the surface often during the growing season      floor 1.3
  somewhat poorly drained  wet at a shallow depth for significant periods of the growing season  floor 0.9
  moderately well drained  wet within the rooting depth for a short time in the growing season   floor 0.4
  well, somewhat excessively, excessively drained: water removed readily, no floor; excessively drained alone is
  capped at 1.2, since it holds little water above field capacity and no water table.
A shallow annual minimum water table (muaggatt.wtdepannmin) raises the floor: under 30 cm, 1.5; under 60 cm, 1.1.
A map unit rated 85 % or more hydric (muaggatt.hydclprs) holds floor 1.3: hydric soils are saturated, flooded or
ponded long enough in the growing season to be anaerobic in the upper part, and the NTCHS criteria put that water
table within 30 cm (Federal Register 2012, 77 FR 12234; Field Indicators of Hydric Soils v9.0).
The National Wetlands Inventory (Cowardin et al. 1979, FGDC-STD-004-2013) adds floor 1.8 under estuarine wetlands,
palustrine vegetated or open wetlands and intermittent streambeds, and marks lakes, rivers and the sea as open water.
Ground just above a water body the survey does not resolve (a pond on Urban land) is held by its height above that
water (HAND, Nobre et al. 2011): `riparian_floor`.

Standard library and numpy only: this runs beside the product, never inside the solver.
"""
from __future__ import annotations

import math
import struct
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

CODE_TOP, CODE_SEALED, CODE_BUILDING, CODE_WATER, CODE_NONE = 250, 251, 252, 253, 255
FIELD_CAPACITY, SATURATION = 1.0, 2.0
RELATIVE = "relative_wetness"
OPEN_WATER = "open water"
"""What `wetland_floor` returns for a lake, a river or the sea."""

DRAINAGE_FLOOR = {"very poorly drained": 1.6, "subaqueous": 1.6, "poorly drained": 1.3,
                  "somewhat poorly drained": 0.9, "moderately well drained": 0.4, "well drained": 0.0,
                  "somewhat excessively drained": 0.0, "excessively drained": 0.0}
# "Subaqueous" is the NRCS class for soils under water all year (added 2017): wetter than very poorly drained, never dry
EXCESSIVE_CEILING = 1.2
WATER_TABLE_FLOORS = ((30.0, 1.5), (60.0, 1.1))
HYDRIC_PCT, HYDRIC_FLOOR = 85.0, 1.3
WETLAND_FLOOR = 1.8
DROUGHT_CAPS = ((1.5, 0.0), (1.3, 0.25), (0.9, 0.5))
"""(floor at least, deficit cap as a share of the drought scale's hi): saturated ground has no deficit."""
RIPARIAN_TOP = 1.7
RIPARIAN_HAND_M = (0.5, 2.0)
"""Height above the nearest water [m]: the full floor at or under the first, none at or over the second."""
RIPARIAN_DIST_M = (10.0, 40.0)
"""Distance from it [m]: the full floor within the first, none at the second; no step anywhere, so no hard edge."""
RIPARIAN_REACH_M = RIPARIAN_DIST_M[1]
SHORE_PERCENTILE = 25.0
MIN_BODY_M2 = 500.0
"""A water body smaller than this seeds no riparian floor unless the NWI maps water or wetland within NWI_NEAR_M of it:
a lined pool or fountain basin (UC Berkeley's Mining Circle, about 250 m2) sets no water table (2026-10-27)."""
NWI_NEAR_M = 10.0
FEATHER_M = 10.0
"""Box width the survey's polygon floors and ceilings are averaged over: SSURGO is mapped at 1:12,000 to 1:24,000."""
LIFT_W = 0.1
"""How far above its floor a lifted cell may land [0..2 scale]: ground already that wet keeps the model exactly."""

RULES = {
    "drainage_class_floor": DRAINAGE_FLOOR, "excessively_drained_ceiling": EXCESSIVE_CEILING,
    "water_table_floor_cm": [list(p) for p in WATER_TABLE_FLOORS], "hydric_pct_at_least": HYDRIC_PCT,
    "hydric_floor": HYDRIC_FLOOR, "nwi_wetland_floor": WETLAND_FLOOR,
    "nwi_wetland_codes": "E (estuarine), PEM, PSS, PFO, PUB, PAB, R4SB (intermittent streambed)",
    "nwi_open_water_codes": "L (lacustrine), R (riverine) but R4SB, M (marine), E1UB and E1AB (estuarine subtidal)",
    "drought_cap_share_of_hi": [list(p) for p in DROUGHT_CAPS],
    "riparian": ("height above nearest water (HAND): a water body is the grid's open water (253) and NWI open water, "
                 "4-connected; its level is the 25th percentile of the ground on its plantable shore ring; each cell takes "
                 "its nearest body's level and distance by 8-neighbour dilation to 40 m. floor = 1.7 * s_h * s_d, "
                 "s_h = smoothstep(clip((2.0 - HAND) / 1.5, 0, 1)) (1 at HAND <= 0.5 m, 0 at >= 2.0 m), "
                 "s_d = smoothstep(clip((40 - dist) / 30, 0, 1)) (1 within 10 m, 0 at 40 m), smoothstep(t) = t*t*(3-2t): "
                 "continuous, falling with height and distance, so no hard edge; a bank 2 m or more above the water "
                 f"keeps the model; a body seeds only when it is at least {MIN_BODY_M2:g} m2 or touches or lies within "
                 f"{NWI_NEAR_M:g} m of any NWI polygon (a lined pool or fountain basin sets no water table). Basis: "
                 "ground this little above a water body sits in its capillary fringe and the shallow water table it sets "
                 "(Nobre et al. 2011, J. Hydrol. 404, HAND; Renno et al. 2008)"),
    "riparian_formula": {"top": RIPARIAN_TOP, "hand_full_zero_m": list(RIPARIAN_HAND_M),
                         "dist_full_zero_m": list(RIPARIAN_DIST_M), "ramp": "smoothstep t*t*(3-2t)"},
    "riparian_seed": {"min_area_m2": MIN_BODY_M2, "or_within_m_of_any_nwi_polygon": NWI_NEAR_M},
    "scale": "0 wilting point, 1 field capacity, 2 saturation (each soil's own)",
    "combine": ("the map units' and wetlands' floor field (the higher of the two) and the map units' ceiling field are "
                f"each replaced by their {FEATHER_M:g} m box mean on the modes grid (every cell counted with its own "
                "value, the mean over the cells inside the grid), never below 0; a cell's floor is the higher of that "
                "and its riparian floor (already smooth); a ceiling below the floor yields to it; a plantable cell "
                f"becomes min(ceiling, max(model, floor + w * model / (floor + w))), w = {LIFT_W:g} (model on 0..2): "
                "continuous and monotonic, a cell at or above floor + w keeps the model's value exactly, a cell below "
                "it lands in [floor, floor + w] with its pattern muted (nothing under a 1.9 floor is lifted to 2.0), and "
                "the ceiling only clips"),
    "feather_m": FEATHER_M,
    "lift_w": LIFT_W,
    "kept": "sealed (251), building (252) and no value (255) are kept; NWI open water marks plantable and sealed ground 253"}


def _num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def drainage_band(drainage_class: Optional[str], water_table_cm=None, hydric_pct=None) -> Tuple[float, float]:
    """(floor, ceiling) on the 0..2 relative scale for one map unit from its dominant drainage class (drclassdcd), annual
    minimum water table depth in cm (wtdepannmin) and hydric share in percent (hydclprs). An unknown or None input
    constrains nothing, so all three unknown give (0, 2). Bands in the module docstring (Soil Survey Manual ch. 3; the
    hydric 30 cm water table criterion)."""
    cls = (drainage_class or "").strip().lower()
    floor = DRAINAGE_FLOOR.get(cls, 0.0)
    ceiling = EXCESSIVE_CEILING if cls == "excessively drained" else SATURATION
    wt = _num(water_table_cm)
    if wt is not None:
        for depth, f in WATER_TABLE_FLOORS:
            if wt < depth:
                floor = max(floor, f)
                break
    hy = _num(hydric_pct)
    if hy is not None and hy >= HYDRIC_PCT:
        floor = max(floor, HYDRIC_FLOOR)
    return floor, max(ceiling, floor)         # a record that contradicts itself keeps the floor: wet must not read dry


def wetland_floor(nwi_attribute: Optional[str]):
    """A floor for one NWI polygon from its Cowardin code (ATTRIBUTE, e.g. E2EM1N, PEM1Ch, L2UBH): WETLAND_FLOOR for
    estuarine, palustrine emergent, scrub-shrub, forested, unconsolidated bottom and aquatic bed, and intermittent
    streambeds (R4SB); OPEN_WATER for lacustrine, other riverine and marine; 0 for anything else (uplands, shores)."""
    a = (nwi_attribute or "").strip().upper()
    if not a:
        return 0.0
    if a.startswith("R4SB"):                              # an intermittent streambed is dry most of the year: wet ground
        return WETLAND_FLOOR
    if a[0] in "LRM" or a.startswith(("E1UB", "E1AB")):     # estuarine subtidal is the inlet itself: water
        return OPEN_WATER
    if a[0] == "E" or (a[0] == "P" and a[1:3] in ("EM", "SS", "FO", "UB", "AB")):
        return WETLAND_FLOOR
    return 0.0


# ------------------------------------------------------------------------------------------------ codes

def _scale(scale) -> Tuple[float, float]:
    if isinstance(scale, dict):
        lo, hi = float(scale["lo"]), float(scale["hi"])
    else:
        lo, hi = float(scale[0]), float(scale[1])
    assert hi > lo, f"scale {lo} to {hi}"
    return lo, hi


def encode(values, scale) -> np.ndarray:
    """Values as codes 0..CODE_TOP on the scale (the publisher's own rounding, exposure.water_modes.encode); NaN is
    CODE_NONE."""
    lo, hi = _scale(scale)
    v = np.asarray(values, np.float64)
    q = np.clip(np.round((v - lo) / (hi - lo) * CODE_TOP), 0, CODE_TOP)
    return np.where(np.isfinite(v), q, CODE_NONE).astype(np.uint8)


def decode(codes, scale) -> np.ndarray:
    lo, hi = _scale(scale)
    c = np.asarray(codes)
    return np.where(c <= CODE_TOP, lo + c.astype(np.float64) / CODE_TOP * (hi - lo), np.nan)


def _code_at_least(values, scale) -> np.ndarray:
    """The smallest code that decodes at or above each value, so a floor is never rounded down."""
    lo, hi = _scale(scale)
    # 1e-4 of a code (4e-7 on the scale) absorbs a float32 field's last bit: 1.6 stored as 1.6000000238 is still code 200
    return np.clip(np.ceil((np.asarray(values, np.float64) - lo) / (hi - lo) * CODE_TOP - 1e-4), 0, CODE_TOP).astype(np.int32)


def _code_at_most(values, scale) -> np.ndarray:
    lo, hi = _scale(scale)
    return np.clip(np.floor((np.asarray(values, np.float64) - lo) / (hi - lo) * CODE_TOP + 1e-4), 0, CODE_TOP).astype(np.int32)


def water_cells(codes: np.ndarray, water: Optional[np.ndarray]) -> np.ndarray:
    """Cells NWI open water marks 253: plantable or sealed ground (a still pond reads flat and sealed to a classifier),
    never a surveyed building and never a cell outside the site (255)."""
    c = np.asarray(codes)
    if water is None:
        return np.zeros(c.shape, bool)
    return np.asarray(water, bool) & (c <= CODE_SEALED)


CHUNK = 1 << 22
"""Cells held at once: a strided site's 64 M cells in float64 temporaries would hold several GB."""


def _chunked(fn, codes: np.ndarray, *fields) -> Tuple[np.ndarray, Dict]:
    """`fn` over flat chunks of `codes` and its per-cell fields (None passes through), its counts summed."""
    c = np.asarray(codes, np.uint8)
    flat = c.ravel()
    fl = [None if a is None else np.broadcast_to(np.asarray(a), c.shape).ravel() for a in fields]
    new, n = np.empty_like(flat), {}
    for a in range(0, max(flat.size, 1), CHUNK):
        s = slice(a, a + CHUNK)
        out, k = fn(flat[s], *[None if x is None else x[s] for x in fl])
        new[s] = out
        for key, v in k.items():
            n[key] = n.get(key, 0) + v
    return new.reshape(c.shape), n


def constrain(codes: np.ndarray, scale, floor: np.ndarray, ceiling: np.ndarray,
              water: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
    """SOIL MOISTURE codes held to ground truth.

    Args:
        codes: The model's uint8 codes [ny, nx] (or flat).
        scale: modes.json drainage.scale ({lo, hi}, relative quantity) or (lo, hi).
        floor, ceiling: Per cell bounds on the 0..2 scale, same shape as `codes`.
        water: Per cell NWI open water, or None.
    Returns:
        (new codes, counts): a plantable cell is min(ceiling, max(model, floor + w * model / (floor + w))), w = LIFT_W,
        re-encoded on the same scale and held in code space between its floor rounded up and its ceiling rounded down,
        so it decodes inside its band and a cell the rule leaves alone keeps its code exactly; 251..255 are kept except
        where `water_cells` marks 253. Counts: raised, lowered, water_nwi.
    """
    if isinstance(scale, dict) and scale.get("quantity", RELATIVE) != RELATIVE:
        raise ValueError(f"the bands are on the relative scale; this grid is {scale.get('quantity')}")
    return _chunked(lambda c, f, ce, w: _constrain(c, scale, f, ce, w), codes, floor, ceiling, water)


def _constrain(c: np.ndarray, scale, floor, ceiling, water) -> Tuple[np.ndarray, Dict]:
    f = np.asarray(floor, np.float64)
    ce = np.maximum(np.asarray(ceiling, np.float64), f)
    lo_c = _code_at_least(f, scale)
    hi_c = np.maximum(_code_at_most(ce, scale), lo_c)
    plant = c <= CODE_TOP
    model = np.nan_to_num(decode(c, scale))
    # ground already wet enough keeps the model; ground below is lifted with its pattern muted (2026-10-27)
    # max(model, f + w m / (f + w)) exceeds the model exactly where f > 0 and m < f + w; decided so, since at f = 0 the
    # lift reads m an ulp high and would cap an untouched cell at code 12
    lifted = (f > 0) & (model < f + LIFT_W)
    v = np.minimum(ce, np.where(lifted, f + LIFT_W * model / (f + LIFT_W), model))
    q = encode(v, scale).astype(np.int32)
    # a lifted cell never rounds past floor + w: under a 1.9 floor nothing reaches saturation
    q = np.where(lifted & (v > model), np.minimum(q, _code_at_most(f + LIFT_W, scale)), q)
    new = np.where(plant, np.clip(q, lo_c, hi_c), c).astype(np.uint8)
    wet = water_cells(c, water)
    marked = wet & (c != CODE_WATER)
    new[wet] = CODE_WATER
    keep = plant & ~wet
    return new, {"raised": int((keep & (new > c)).sum()), "lowered": int((keep & (new < c)).sum()),
                 "water_nwi": int(marked.sum())}


def constrain_drought(codes: np.ndarray, scale, floor: np.ndarray,
                      water: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
    """DROUGHT codes (mm of deficit, modes.json drought.scale) held to the same floors: floor 1.5 or more has no deficit,
    1.3 to 1.5 at most 25 % of the scale's hi, 0.9 to 1.3 at most 50 %; `water` (already resolved by `water_cells` on
    the moisture grid) becomes 253; 251 and 255 are kept. Counts: lowered, water_nwi."""
    return _chunked(lambda c, f, w: _constrain_drought(c, scale, f, w), codes, floor, water)


def _constrain_drought(c: np.ndarray, scale, floor, water) -> Tuple[np.ndarray, Dict]:
    lo, hi = _scale(scale)
    f = np.asarray(floor, np.float64)
    cap = np.full(c.shape, hi)
    for at_least, share in reversed(DROUGHT_CAPS):          # the highest floor's cap written last
        cap = np.where(f >= at_least, share * hi, cap)
    cap_c = _code_at_most(cap, scale)
    plant = c <= CODE_TOP
    new = np.where(plant, np.minimum(c.astype(np.int32), cap_c), c).astype(np.uint8)
    wet = np.zeros(c.shape, bool) if water is None else (np.asarray(water, bool) & (c != CODE_NONE))
    marked = wet & (c != CODE_WATER)
    new[wet] = CODE_WATER
    return new, {"lowered": int((plant & ~wet & (new < c)).sum()), "water_nwi": int(marked.sum())}


def wetter_than_fc_share(codes: np.ndarray, scale) -> Optional[float]:
    """The share of plantable cells wetter than field capacity (exposure.water_modes.soil_moisture's statistic, here
    from the codes)."""
    c = np.asarray(codes)
    v = decode(c[c <= CODE_TOP], scale)
    return round(float((v > FIELD_CAPACITY).mean()), 4) if v.size else None


# ------------------------------------------------------------------------------------------------ rasterizing

def _span(a: float, b: float, origin: float, step: float, n: int) -> Tuple[int, int]:
    """Indices [i0, i1) of the cells whose centres origin + (i + 0.5) * step lie within [a, b]."""
    u = sorted(((a - origin) / step - 0.5, (b - origin) / step - 0.5))
    i0 = min(n, max(0, int(math.ceil(u[0]))))
    # an end before the grid would read as a negative slice: a polygon wholly in the query margin took 348 rows (a test site)
    return i0, max(i0, min(n, int(math.floor(u[1])) + 1))


def window(ring, shape: Sequence[int], transform: Dict) -> Tuple[int, int, int, int]:
    """(r0, r1, c0, c1): the cells whose centres lie in the ring's bounding box."""
    r = np.asarray(ring, np.float64)
    t = transform
    r0, r1 = _span(r[:, 1].min(), r[:, 1].max(), t["y0"], t["dy"], shape[0])
    c0, c1 = _span(r[:, 0].min(), r[:, 0].max(), t["x0"], t["dx"], shape[1])
    return r0, r1, c0, c1


ROWS_AT_ONCE = 256
"""Scanlines per batch, so a ring's crossing table stays a few MB however long its edge list."""


def ring_inside(ring, win: Tuple[int, int, int, int], transform: Dict) -> np.ndarray:
    """Even-odd point-in-polygon of every cell centre in `win` against one ring, vectorized over the ring's edges: each
    row's edge crossings toggle the cells to their right, and a running sum's parity is inside."""
    t = transform
    assert t["dx"] > 0, "columns run east"
    r0, r1, c0, c1 = win
    out = np.zeros((max(r1 - r0, 0), max(c1 - c0, 0)), bool)
    p = np.asarray(ring, np.float64)[:, :2]
    if out.size == 0 or len(p) < 3:
        return out
    if not np.array_equal(p[0], p[-1]):
        p = np.vstack([p, p[:1]])
    xi, yi, xj, yj = p[:-1, 0], p[:-1, 1], p[1:, 0], p[1:, 1]
    w = c1 - c0
    for a in range(r0, r1, ROWS_AT_ONCE):
        rows = np.arange(a, min(a + ROWS_AT_ONCE, r1))
        y = (t["y0"] + (rows + 0.5) * t["dy"])[:, None]
        cross = (yi > y) != (yj > y)                         # half-open, so a vertex on the row counts once
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = xi + (y - yi) * (xj - xi) / (yj - yi)
        ri, ei = np.nonzero(cross)
        k = np.floor((xc[ri, ei] - t["x0"]) / t["dx"] - 0.5).astype(np.int64) + 1 - c0   # first column right of it
        tog = np.zeros((len(rows), w + 1), np.int32)
        np.add.at(tog, (ri, np.clip(k, 0, w)), 1)
        out[a - r0:a - r0 + len(rows)] = (np.cumsum(tog[:, :w], axis=1) & 1).astype(bool)
    return out


def polygon_mask(rings: Sequence, shape: Sequence[int], transform: Dict) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
    """One polygon (outer ring, holes...) in local metres: its window and the mask there, every hole subtracted."""
    win = window(rings[0], shape, transform)
    m = ring_inside(rings[0], win, transform)
    for hole in rings[1:]:
        if m.any():
            m &= ~ring_inside(hole, win, transform)
    return win, m


def rasterize(polygons: Iterable[Sequence], shape: Sequence[int], transform: Dict) -> np.ndarray:
    """Cells whose centres any of the polygons covers."""
    out = np.zeros(tuple(shape), bool)
    for rings in polygons:
        (r0, r1, c0, c1), m = polygon_mask(rings, shape, transform)
        out[r0:r1, c0:c1] |= m
    return out


def truth_grids(units: Sequence[Dict], wetlands: Sequence[Dict], shape: Sequence[int], transform: Dict,
                dtype=np.float64) -> Dict:
    """Per cell floor, ceiling, open water and map unit from projected sources.

    Args:
        units: One per SSURGO polygon: {mukey, drclassdcd, wtdepannmin, hydclprs, polygons (local metres)}.
        wetlands: One per NWI polygon: {attribute, polygons (local metres)}.
    Returns:
        {floor, ceiling, ceiling_raw, water, mukey_index (-1 none), mukeys, wetland_floor, soil_floor}, the fields in
        `dtype` (float32 halves a strided site's 64 M-cell grids).
    """
    shape = tuple(shape)
    soil_f = np.zeros(shape, dtype)
    ceil = np.full(shape, SATURATION, dtype)
    held = np.zeros(shape, bool)                             # a surveyed cell whose unit sets a ceiling
    idx = np.full(shape, -1, np.int32)
    keys = []
    for u in units:
        f, c = drainage_band(u.get("drclassdcd"), u.get("wtdepannmin"), u.get("hydclprs"))
        if u["mukey"] not in keys:
            keys.append(u["mukey"])
        k = keys.index(u["mukey"])
        for rings in u["polygons"]:
            (r0, r1, c0, c1), m = polygon_mask(rings, shape, transform)
            sl = (slice(r0, r1), slice(c0, c1))
            soil_f[sl] = np.where(m, np.maximum(soil_f[sl], f), soil_f[sl])
            # where two units share a cell (an edge), the looser ceiling wins: the floor is the safety half
            ceil[sl] = np.where(m, np.where(held[sl], np.maximum(ceil[sl], c), c), ceil[sl])
            held[sl] |= m
            idx[sl] = np.where(m & (idx[sl] < 0), k, idx[sl])
    wet_f = np.zeros(shape, dtype)
    water = np.zeros(shape, bool)
    nwi_any = np.zeros(shape, bool)                          # every NWI polygon, any code: what may seed a small pond
    for w in wetlands:
        f = wetland_floor(w.get("attribute"))
        for rings in w["polygons"]:                        # each polygon in its own window, not a whole-grid pass
            (r0, r1, c0, c1), m = polygon_mask(rings, shape, transform)
            sl = (slice(r0, r1), slice(c0, c1))
            nwi_any[sl] |= m
            if f == OPEN_WATER:
                water[sl] |= m
            elif f > 0:
                wet_f[sl] = np.where(m, np.maximum(wet_f[sl], f), wet_f[sl])
    floor = np.maximum(soil_f, wet_f)
    return {"floor": floor, "ceiling": np.maximum(ceil, floor), "ceiling_raw": ceil, "water": water, "mukey_index": idx,
            "mukeys": keys, "nwi_any": nwi_any,
            "soil_floor": soil_f, "wetland_floor": wet_f}


# ------------------------------------------------------------------------------------------------ feathering

def _running_mean(a: np.ndarray, r: int, axis: int, chunk: int = 256) -> np.ndarray:
    """A [ny, nx] array's mean over i - r .. i + r along one axis, of the cells inside the grid only, so an edge is not
    pulled to 0; summed in float64 a few hundred lines at a time, so a 64 M-cell site never holds a float64 copy."""
    if r <= 0:
        return a
    n = a.shape[axis]
    i = np.arange(n)
    lo, hi = np.clip(i - r, 0, n), np.clip(i + r + 1, 0, n)
    cnt = (hi - lo).astype(np.float64)
    out = np.empty_like(a)
    for s in range(0, a.shape[1 - axis], chunk):
        if axis == 1:
            x = a[s:s + chunk].astype(np.float64)
            c = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(x, 1)], 1)
            out[s:s + chunk] = (c[:, hi] - c[:, lo]) / cnt
        else:
            x = a[:, s:s + chunk].astype(np.float64)
            c = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x, 0)], 0)
            out[:, s:s + chunk] = (c[hi] - c[lo]) / cnt[:, None]
    return out


def feather(field: np.ndarray, transform: Dict, width_m: float = FEATHER_M) -> np.ndarray:
    """A [ny, nx] field's box mean `width_m` wide (separable running means), never below 0, in the field's dtype."""
    a = np.asarray(field)
    a = a if a.dtype.kind == "f" else a.astype(np.float64)
    ry = int(round(width_m / 2 / abs(transform["dy"])))
    rx = int(round(width_m / 2 / abs(transform["dx"])))
    return np.maximum(_running_mean(_running_mean(a, rx, 1), ry, 0), 0)


def bounds(g: Dict, riparian: Optional[np.ndarray], transform: Dict) -> Dict:
    """Each cell's floor and ceiling from `truth_grids` and the riparian floor: the polygon floor (map unit or wetland,
    the higher) and the map units' ceiling feathered over FEATHER_M, the riparian floor as it is; and `by`, the three
    sources' floors [ssurgo, nwi, riparian] as they bind, for counting which one raised a cell."""
    soil, wet = g["soil_floor"], g["wetland_floor"]
    poly = feather(np.maximum(soil, wet), transform)
    fs, fw = feather(soil, transform), feather(wet, transform)
    rip = (np.zeros(poly.shape, poly.dtype) if riparian is None
           else np.asarray(riparian).reshape(poly.shape).astype(poly.dtype, copy=False))
    floor = np.maximum(poly, rip)
    ceiling = np.maximum(feather(g.get("ceiling_raw", g["ceiling"]), transform), floor)
    by = np.stack([np.where(fs >= fw, poly, 0), np.where(fw > fs, poly, 0), rip])
    return {"floor": floor, "ceiling": ceiling, "by": by}


# ------------------------------------------------------------------------------------------------ riparian (HAND)

def ground_grid(meta) -> Dict:
    """The ground array's grid {nx, ny, cell, x0, y0 (the SOUTH edge), from}, its rows from the south, from whatever
    describes it: the solar manifest (grid.cell_m, y0 the south edge, as sims.sun.solar_ground reads it), a collected
    site's ground.json (sims.stride_collect, "rows": "from the south") or a SIMF header (y0 the north edge, as the
    viewer's app.dsm reads it). A .simf.json sidecar names no grid, so it is refused."""
    if isinstance(meta, (bytes, bytearray)):
        assert bytes(meta[:4]) == b"SIMF", "not a SIMF header"
        _, _, nx, ny = struct.unpack_from("<4I", meta, 4)
        cell, x0, y0 = struct.unpack_from("<3d", meta, 32)
        return {"nx": nx, "ny": ny, "cell": cell, "x0": x0, "y0": y0 - ny * cell, "from": "SIMF header (y0 north edge)"}
    if "rows" in meta:
        assert meta["rows"] == "from the south", f"ground rows {meta['rows']}"
        return {"nx": int(meta["nx"]), "ny": int(meta["ny"]), "cell": float(meta["cell"]), "x0": float(meta["x0"]),
                "y0": float(meta["y0"]), "from": "ground.json (rows from the south)"}
    g = meta.get("grid") or {}
    if "cell_m" in g:
        return {"nx": int(g["nx"]), "ny": int(g["ny"]), "cell": float(g["cell_m"]), "x0": float(g["x0"]),
                "y0": float(g["y0"]), "from": "solar manifest grid (y0 south edge)"}
    raise ValueError("no ground grid: give the solar manifest.json, a ground.json or the SIMF file")


def sample_ground(ground, gg: Dict, shape: Sequence[int], transform: Dict) -> np.ndarray:
    """The ground height [m] at every modes cell centre: the ground cell holding it (the viewer's groundAt), NaN off the
    ground grid or where it has no value."""
    ny, nx = shape
    t = transform
    z = np.asarray(ground).ravel()
    z = z if z.dtype.kind == "f" else z.astype(np.float64)                 # the float32 file stays float32
    assert z.size == gg["nx"] * gg["ny"], f"{z.size} ground values for a {gg['nx']} x {gg['ny']} grid"
    Z = z.reshape(gg["ny"], gg["nx"])                       # row 0 the south edge
    ix = np.floor((t["x0"] + (np.arange(nx) + 0.5) * t["dx"] - gg["x0"]) / gg["cell"]).astype(np.int64)
    iy = np.floor((t["y0"] + (np.arange(ny) + 0.5) * t["dy"] - gg["y0"]) / gg["cell"]).astype(np.int64)
    okx, oky = (ix >= 0) & (ix < gg["nx"]), (iy >= 0) & (iy < gg["ny"])
    out = np.full((ny, nx), np.nan, z.dtype)
    out[np.ix_(oky, okx)] = Z[np.ix_(iy[oky], ix[okx])]
    out[~np.isfinite(out) | (np.abs(out) > 1e4)] = np.nan   # a fill sentinel is no height
    return out


def _nb(a: np.ndarray, di: int, dj: int, fill) -> np.ndarray:
    """b[i, j] = a[i + di, j + dj], `fill` past the edge."""
    ny, nx = a.shape
    b = np.full_like(a, fill)
    b[max(0, -di):ny - max(0, di), max(0, -dj):nx - max(0, dj)] = \
        a[max(0, di):ny + min(0, di), max(0, dj):nx + min(0, dj)]
    return b


FOUR = ((1, 0), (-1, 0), (0, 1), (0, -1))


def label_bodies(water: np.ndarray) -> Tuple[np.ndarray, int]:
    """Connected water bodies, 4-connected: (labels 0..n-1, -1 off water; n). Each cell takes its neighbours' least
    index, then pointer jumping collapses chains, so a long creek needs a few passes, not its length."""
    w = np.asarray(water, bool)
    ny, nx = w.shape
    big = ny * nx
    dt = np.int32 if big < 2 ** 31 - 1 else np.int64      # a 64 M-cell site in int32: half the memory
    lab = np.where(w, np.arange(big, dtype=dt).reshape(ny, nx), dt(big))
    idx = np.flatnonzero(w)
    while idx.size:
        m = lab
        for di, dj in FOUR:
            m = np.minimum(m, _nb(lab, di, dj, big))
        f = np.where(w, m, big).ravel()
        while True:
            j = f[f[idx]]
            if np.array_equal(j, f[idx]):
                break
            f[idx] = j
        m = f.reshape(ny, nx)
        if np.array_equal(m, lab):
            break
        lab = m
    out = np.full((ny, nx), -1, dt)
    u, inv = np.unique(lab[w], return_inverse=True)
    out[w] = inv.ravel()
    return out, int(u.size)


def water_levels(lab: np.ndarray, n: int, z: np.ndarray, plant: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Each body's level [m], the SHORE_PERCENTILE of the ground on its shore ring (plantable cells 4-connected to it),
    NaN where its shore has no height; and its shore cell count."""
    L = np.where(lab >= 0, lab, n)
    near = np.full(lab.shape, n, lab.dtype)
    for di, dj in FOUR:
        near = np.minimum(near, _nb(L, di, dj, n))
    shore = plant & (lab < 0) & (near < n)
    s = shore & np.isfinite(z)
    level = np.full(n, np.nan)
    count = np.bincount(near[shore], minlength=n)[:n] if n else np.zeros(0, np.int64)
    if s.any():
        b, h = near[s], z[s]
        o = np.lexsort((h, b))
        b, h = b[o], h[o]
        u, start, cnt = np.unique(b, return_index=True, return_counts=True)
        pos = start + SHORE_PERCENTILE / 100.0 * (cnt - 1)          # np.percentile's linear rule, per body
        lo, hi = np.floor(pos).astype(np.int64), np.ceil(pos).astype(np.int64)
        level[u] = h[lo] + (h[hi] - h[lo]) * (pos - lo)
    return level, count


def nearest_level(lab: np.ndarray, level: np.ndarray, transform: Dict, reach_m: float = RIPARIAN_REACH_M,
                  dtype=np.float64):
    """Every cell's distance [m] to its nearest leveled water body and that body's level, by 8-neighbour dilation (a
    chamfer distance, at most 8 % over the straight line) out to `reach_m`; inf and NaN beyond; both in `dtype`."""
    ax, ay = abs(transform["dx"]), abs(transform["dy"])
    lv = np.where(lab >= 0, level.astype(dtype)[np.maximum(lab, 0)] if level.size else np.nan, np.nan).astype(dtype)
    seed = np.isfinite(lv)
    dist = np.where(seed, 0.0, np.inf).astype(dtype)
    zw = np.where(seed, lv, np.nan)
    steps = [(di, dj, math.hypot(di * ay, dj * ax)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
    for _ in range(int(reach_m / min(ax, ay)) + 3):
        changed = False
        for di, dj, w in steps:
            cand = _nb(dist, di, dj, np.inf) + w
            b = (cand < dist - 1e-9) & (cand <= reach_m + 1e-9)
            if b.any():
                dist[b] = cand[b]
                zw[b] = _nb(zw, di, dj, np.nan)[b]
                changed = True
        if not changed:
            break
    return dist, zw


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def riparian_value(hand, dist):
    """The riparian floor at a height above the nearest water [m] and a distance from it [m]: RIPARIAN_TOP * s_h * s_d,
    each a smoothstep ramp, so it falls continuously with both; 0 where either is unknown or past its far end."""
    h0, h1 = RIPARIAN_HAND_M
    d0, d1 = RIPARIAN_DIST_M
    with np.errstate(invalid="ignore"):
        s_h = smoothstep((h1 - np.asarray(hand)) / (h1 - h0))         # in the inputs' dtype
        s_d = smoothstep((d1 - np.asarray(dist)) / (d1 - d0))
    return np.nan_to_num(RIPARIAN_TOP * s_h * s_d, nan=0.0)


def near(mask: np.ndarray, transform: Dict, reach_m: float) -> np.ndarray:
    """Cells within `reach_m` of any `mask` cell (8-neighbour chamfer distance, as `nearest_level`)."""
    ax, ay = abs(transform["dx"]), abs(transform["dy"])
    dist = np.where(np.asarray(mask, bool), 0.0, np.inf).astype(np.float32)
    steps = [(di, dj, math.hypot(di * ay, dj * ax)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
    for _ in range(int(reach_m / min(ax, ay)) + 3):
        changed = False
        for di, dj, w in steps:
            cand = _nb(dist, di, dj, np.inf) + w
            b = (cand < dist) & (cand <= reach_m + 1e-6)
            if b.any():
                dist[b] = cand[b]
                changed = True
        if not changed:
            break
    return np.isfinite(dist)


def riparian_floor(codes: np.ndarray, water: Optional[np.ndarray], z: np.ndarray, transform: Dict,
                   nwi_any: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
    """Floors from height above the nearest water (HAND, Nobre et al. 2011) on a [ny, nx] modes grid.

    Args:
        codes: The model's moisture codes; 253 is water.
        water: NWI open water already resolved to cells (`water_cells`), or None.
        z: Ground height at every cell centre (`sample_ground`), NaN where unknown.
        nwi_any: Every NWI polygon's cells, any code (`truth_grids`), or None.
    Returns:
        (floor per cell, 0 where none; info): `riparian_value` on plantable cells. A bank 2 m or more above its water
        takes no floor, so a levee top keeps the model. A body seeds only at MIN_BODY_M2 or more, or when it touches or
        lies within NWI_NEAR_M of an NWI polygon: a lined fountain basin sets no water table.
    """
    c = np.asarray(codes)
    w = (c == CODE_WATER) | (np.zeros(c.shape, bool) if water is None else np.asarray(water, bool))
    plant = (c <= CODE_TOP) & ~w
    lab, n = label_bodies(w)
    level, shore = water_levels(lab, n, z, plant)
    area = np.bincount(lab[lab >= 0], minlength=n)[:n] * abs(transform["dx"] * transform["dy"]) if n else np.zeros(0)
    small = area < MIN_BODY_M2
    mapped = np.zeros(n, bool)
    if small.any() and nwi_any is not None and np.asarray(nwi_any, bool).any():
        hit = near(nwi_any, transform, NWI_NEAR_M) & (lab >= 0)
        mapped = np.bincount(lab[hit], minlength=n)[:n] > 0
    unseeded = small & ~mapped
    level = np.where(unseeded, np.nan, level)
    dist, zw = nearest_level(lab, level, transform, dtype=z.dtype if z.dtype.kind == "f" else np.float64)
    f = np.where(plant, riparian_value(z - zw, dist), 0.0)
    big = np.argsort(-area)[:10] if n else []
    return f, {"water_bodies": n, "leveled": int(np.isfinite(level).sum()), "under_min_area": int(small.sum()),
               "small_seeded_by_nwi": int((small & mapped).sum()),
               "cells_within_reach": int((plant & np.isfinite(dist)).sum()),
               "cells_with_floor": {"any": int((f > 0).sum()), "at_least_1.0": int((f >= 1.0).sum()),
                                    "at_least_1.5": int((f >= 1.5).sum()), "full_1.7": int((f >= RIPARIAN_TOP).sum())},
               "largest": [{"area_m2": round(float(area[k]), 1), "shore_cells": int(shore[k]),
                            "level_m": None if not np.isfinite(level[k]) else round(float(level[k]), 3)} for k in big]}
