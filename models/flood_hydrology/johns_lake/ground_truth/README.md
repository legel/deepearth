# Ground-truth validation

Independent checks of this repo's satellite-derived lake metrics against external imagery and
gauge sources — separate from `verify_all.py`'s internal pipeline-consistency checks. See
`CLAUDE.md`'s "Ground-truth validation" section for the one-paragraph summary; this file has
the full method and numbers.

---

## 1. Finer-resolution imagery alternatives to Sentinel-2

The water-segmentation stack (`sentinel2/`) runs on Sentinel-2 L2A at 10m resolution. Two
finer-resolution alternatives were reviewed as candidates for higher-precision shoreline
tracking:

**PlanetScope (Planet Labs), 3m resolution.** Reviewed a set of publicly available sample
images (general marketing/example gallery — none of Johns Lake itself, but several flood/lake
scenes at comparable resolution and water-edge character). At 3m/pixel, individual houses,
roads, and docks are clearly resolved as distinct shapes; shorelines show a visible
color-graded transition buffer (exposed lake-bed / shallow-water mixing) several pixels wide
rather than a hard binary edge. For a well-contrasted water/land edge, sub-pixel
edge-detection realistically achieves ~0.3–1 pixel (~1–3m) precision; for Johns Lake
specifically (turbid water, vegetated fringes, per the existing 10m OWM analysis) the more
realistic estimate is **~2–5m** — still 2–5× better than Sentinel-2's 10m pixel.
- Revisit: PlanetScope offers daily/sub-daily global revisit, technically sufficient to catch a
  specific event date, but Florida wet-season convective storms bring same-day cloud cover —
  the same risk already documented for Sentinel-2 in this repo. A before/after pair around an
  event is realistic; a clean shot *during* the event is not guaranteed.
- Pricing: published direct-order pricing is on the order of $1.80/km² with a 250 km² minimum
  order (~$450/order regardless of how small the target AOI is) — a different structure than
  smaller per-image archive-resale pricing. Get a same-source, in-writing quote for the
  specific AOI before budgeting; the two pricing models found differ by an order of magnitude
  in assumptions.

**Drone RTK/PPK imagery.** A genuine survey-grade capture exists (RTK/PPK GNSS-tagged JPGs,
confirmed via embedded GNSS log files, not just consumer GPS), already processed into two
Pix4D photogrammetry archives. GPS metadata from a sample of frames places the footprint at
**~100–150m across**, centered almost exactly on the project's target address — a
single-residence-scale capture, not a lake-wide or shoreline-length survey. Its useful role is
**cm-precision validation/fine-tuning of the DEM/terrain model at one property**, not a
lake-extent ground-truth source. A validation routine (extract the orthomosaic from the
photogrammetry archive, reproject to EPSG:5070, compare against the same-date OWM mask at the
property edge) is designed but not yet built.

---

## 2. USGS / regional gauge cross-check

No USGS gauge sits directly on Johns Lake. Two candidate USGS station IDs that came up during
the search turned out to be unrelated gauges on different watersheds entirely (confirmed by
fetching their live station pages directly rather than trusting search-result snippets) — a
reminder that USGS station-ID search results can mis-attribute names.

**Real source found: Orange County Water Atlas.** Johns Lake is waterbody ID **7935** on the
Orange County Water Atlas (`orange.wateratlas.usf.edu/waterbodies/lakes/7935/johns-lake`), which
documents a continuous hydrologic sampling location named **"JOHNS"** under "Orange County
Stormwater Water Levels Sampling," and states the lake's documented surface area as 2,580
acres (≈1,044 ha) — independently corroborating the AOI under-coverage finding in §3 below.
The data-download tool is a 3-step interactive workflow (map-select → date range → email
delivery), not a scriptable API.

**Cross-check result: passes.** Filtering the full waterbody-7935 export to
`Characteristic == "Elevation, water surface (NAVD88)"` gives **1,830 readings, all from
station JOHNS, spanning 1959-07-01 to 2026-05-01** (saved as
`ground_truth/data/johns_lake_gauge_wse.csv`).
- All-time range: 22.08–30.22 m NAVD88. The minimum (2000-02-10) lines up with Florida's
  1999–2001 drought; the maximum (2004-09-27) lines up with the 2004 hurricane season — both
  independently dateable external events, a good sign the record is real.
- Restricting to 2016+ (this repo's Sentinel-2 archive window): **mean 29.07 m, stdev 0.35 m,
  n=94**. This repo's authoritative satellite-derived WSE (28.74 m, `dem/data/lake_volume.csv`)
  sits **0.33 m below the gauge mean — within 1 stdev of normal seasonal/interannual
  lake-level variation.**
- Most recent readings (late 2025–2026) show a declining trend through the dry season,
  directionally consistent with this repo's own drought-drawdown model (`dem/lake_drought.py`),
  though not a precise period-matched comparison.

**Note — unreconciled secondary file:** `ground_truth/data/usgs_02237540_johns_lake_dv.csv`
(4,178 daily readings, 1959–2026) was also pulled but its date-level values don't match the
JOHNS-station export above on overlapping dates (checked directly — typically off by
0.1–0.3m). It's kept as raw data but **not yet reconciled or used in the cross-check above**;
worth resolving (is it a genuinely different gauge, a different vertical datum, or a
preprocessing difference?) before relying on it.

---

## 3. AOI coverage check

Script: `seasonal_area_split.py`. Compares the per-scene OWM polygon area
(`sentinel2/data/water_extent_timeseries.csv`, 152 scenes) against the "authoritative"
144.47 ha consensus-mask figure (`dem/data/lake_volume.csv`):

| season | n | mean_ha | median_ha | stdev_ha | min_ha | max_ha |
|---|---|---|---|---|---|---|
| wet (Jun–Sep) | 23 | 120.71 | 121.35 | 7.58 | 95.48 | 132.61 |
| dry (Oct–May) | 130 | 122.39 | 123.30 | 6.45 | 77.05 | 140.30 |

Seasonality is not a meaningful bias source (wet-season mean is actually slightly *lower* than
dry-season, −1.4%, opposite of naive expectation and within scene-to-scene noise). The bigger
finding: the all-scene OWM mean (122.14 ha) is **18% below** the 144.47 ha consensus-mask
figure — explained by methodology (majority-vote consensus mask includes marginal/fringe
pixels wet in most-but-not-all scenes, pulling its area above a typical single-scene polygon),
not by error. See `CLAUDE.md`'s Consistency Check section.

**A separate, larger gap:** Johns Lake's documented full surface area (2,580 acres ≈ 1,044 ha,
Orange County Water Atlas) is more than double the current 2x2km AOI's total grid area
(521.87 ha). Checked directly by reading `dem/data/lake_mask.tif`: water-mask pixels touch the
AOI boundary on the **north (502 cells), west (450 cells), and east (127 cells)** edges — only
the south edge is clear (5 cells). **The lake is being clipped by the download extent on 3 of 4
sides**, not merely smaller than reported. The 144.47 ha figure is the area of one
basin/portion of the lake near the target address, not the whole lake. See `CLAUDE.md`'s
"Future work" item 2; `validate_aoi_expansion.py` in this directory has already validated a
7.37×3.87 km bounding box (buffered around the real NHD shoreline) with **zero edge-clipping**,
ready to use for a full re-run.

---

## 4. Clay Foundation Model — scoping (deprioritized)

Adding Clay as a 5th water-segmentation method alongside MNDWI/WatNet/Prithvi-EO-2.0/OWM would
be mechanically simple (`sentinel2/compare_methods.py` already has a `METHODS` dict pattern to
extend), but Clay (v1.5, 632M-param ViT) is **self-supervised, embeddings-only** — it ships no
water-segmentation head, unlike OmniWaterMask or the existing Prithvi-EO-2.0 path. Getting a
binary water mask out of it would require training a task-specific decoder from scratch — a
multi-day ML task, not a quick integration. Deprioritized: even a successful integration would
still run on the same 10m Sentinel-2 input, so it can't address the resolution question in §1,
only add method diversity to the comparison chart.

---

## Summary

| Question | Verdict |
|---|---|
| Finer-resolution alternative to Sentinel-2? | PlanetScope (~2–5m realistic shoreline precision) is the lake-wide candidate; drone RTK is real but property-scale only |
| Is there a gauge to cross-check the satellite-derived WSE? | Yes — Orange County Water Atlas "JOHNS" station (waterbody 7935); cross-check passes within 1 stdev |
| Is the 144.47 ha vs. ~122 ha gap a bug? | No — two different mask methodologies (majority-vote consensus vs. per-scene polygon), not a measurement disagreement |
| Does the AOI cover the whole lake? | No — clipped on 3 of 4 edges; true lake area is >2× the current AOI |
| Worth adding Clay Foundation Model? | Low priority — doesn't address resolution, and needs a decoder trained from scratch |
