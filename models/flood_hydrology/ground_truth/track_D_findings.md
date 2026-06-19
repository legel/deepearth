# Track D — Ground-truth data sourcing: findings (2026-06-18)

Investigation only. No changes made to the existing OWM/lake_mask pipeline.
See `plans/track_D_ground_truth_data.md` for the original task list.

---

## 1. PlanetScope sample review

`~/Desktop/PlanetScope/` actually contains **29 sample JPGs**, not the 3
mentioned in the plan (more must have been added to the folder after the plan
was written: `Algiers_Algeria`, `Birmingham`, `CarrizoPlain_CA` ×2,
`Derna_Libya_PostFlood`, `FloodingErfstadtGermany`, `Freeland_MI`,
`LakeOroville_Dam`, `LakeTahoe`, `NYC` ×2, `Rutherford_NJ`, etc.). None are
Johns Lake — they're Planet's general marketing/example gallery, but several
are directly relevant as quality references: `LakeOroville_Dam`,
`FloodingErfstadtGermany`, `Freeland_MI` (flood), `LakeTahoe`.

**What the imagery actually looks like at 3 m/pixel:** individual houses,
roads, and boat docks are clearly resolved as distinct shapes (`LakeOroville_Dam`
sample). Shorelines show a visible color-graded buffer (exposed lake-bed /
shallow water transition zone) rather than a hard binary edge — at Oroville
this buffer is several pixels wide, i.e. several meters to a few tens of
meters, consistent with sub-pixel mixed pixels at a sloped shoreline.

**Shoreline-position precision estimate:** for a well-contrasted water/land
edge (calm water, low turbidity, gentle slope), sub-pixel edge-detection
(e.g. fitting NDWI/MNDWI gradient across the boundary) realistically achieves
**~0.3–1 pixel = ~1–3 m** horizontal precision — a commonly cited rule of
thumb for clean spectral edges. For Johns Lake specifically (turbid Florida
lake with vegetated/wetland fringes per the existing 10 m OWM analysis),
expect the worse end of that range, call it **2–5 m** realistic precision,
still ~2–5× better than the 10 m Sentinel-2 pixel OWM currently uses.

**Caveat:** the sample JPGs are 8-bit RGB "ENHANCE" marketing previews, not
analytic-ready products. A real order would come as multi-band (4- or 8-band)
GeoTIFF surface reflectance, which is what any quantitative shoreline
extraction would need — visual impression from these JPGs is a reasonable
proxy for resolution/clarity, not for spectral water-index accuracy.

**$300/image × 3-image minimum ($900) viability:** Planet's own published
direct-order pricing (planet.com/pricing, accessed 2026-06-18) is
**$1.80/km² with a 250 km² minimum order** — at that rate a single order
would cost ~$450 regardless of how small the AOI is, and the 250 km² minimum
absurdly oversizes a 5.2 km² target (Johns Lake AOI is currently 521.87 ha ≈
5.2 km²; see §4). Lance's $300/image, 3-image-minimumi, ~100 km² figure does
not match Planet's direct pricing model and likely came from a reseller
(e.g. Apollo Mapping-style archive-image sales, which sell existing scenes at
a flat per-image rate with a smaller area footprint, not Planet's
subscription/tasking model). **Recommendation: get a same-source, in-writing
quote for the specific AOI before budgeting** — the two pricing structures
found differ by an order of magnitude in assumptions and one isn't
necessarily what Lance was quoted.

**Revisit schedule:** Planet's documentation states PlanetScope provides
daily (or sub-daily) revisit globally between ±81.5° latitude — technically
sufficient to catch a specific flood event date. The real risk is the same
one already documented for Sentinel-2 in this repo: Florida wet-season
convective storms bring heavy same-day cloud cover, so daily revisit doesn't
guarantee a clear shot of the event itself. A before/after pair around a
flood is more achievable than a clean "during" frame.

---

## 2. Drone RTK imagery

**Status as of 2026-06-18:** Lance granted access via Google Drive
(`.../folders/13Q4FpP2xe7U5wcdpTL-fAIJw_5xM1rXJ`, folder name
`Winter_Garden_Residence`). Initially not locally indexed after Qin installed
Drive for Desktop; resolved once Qin added the shortcut from the share link.
Inspected file headers/EXIF only — did not unzip or bulk-download anything
(see sizes below, this matters).

**What's actually there:**
- `Drone_Capture/`: **595 JPGs** (5280×3956, DJI `_V` suffix) + one
  `.MRK`/`.obs`/`.nav`/`.bin` set named
  `DJI_202504280942_001_EcoDash-Lance_PPK*` — these are RTK/PPK
  (Post-Processed Kinematic) GNSS log files, confirming this is genuine
  RTK-precision drone capture, not just a regular consumer flight.
  Captured **2025-04-28**, ~09:56–10:12 local.
- `Ground_GeoFusion/Pix4DCatch/`: one ground-based photogrammetry capture
  (`2025-04-28-12-02-35.zip`, same day, ~2 hrs after the drone flight) —
  likely a ground-level LiDAR/photo scan to fuse with the aerial data.
- `Open_Photogrammetry_Format/`: **two already-processed Pix4D OPF
  archives, 19.0 GB and 20.7 GB** (`boyd_residence_opf.zip`,
  `boyd_residence_opf_with_calibrated_cameras.zip`) — i.e. someone has
  already run this through Pix4D and these are the finished
  camera-calibrated photogrammetry projects (point cloud / mesh / ortho
  generation inputs), not raw. **Did not download these — at ~40 GB
  combined they're exactly the "seems huge" concern Qin raised; if/when
  needed, extract only the orthomosaic GeoTIFF from inside the zip rather
  than the whole archive.**
- `Videos/`: supplementary iPhone photos/video (HEIC/MOV/PNG) from the same
  site visit, not photogrammetry inputs.

**Coverage check (read EXIF GPS from 5 sample JPGs spread across the flight,
no bulk download):** all 5 cluster within **lat 28.52104–28.52227, lon
-81.65750– -81.65663** — a footprint roughly **100–150 m across**, centered
almost exactly on the AOI center given in CLAUDE.md (28.5216, -81.6570).
Flight altitude ~38–47 m AGL. **This confirms the drone survey is a
single-residence-scale capture ("boyd_residence"/"Winter_Garden_Residence"),
not a lake-wide or shoreline-length survey.** It matches the separate
"flood simulation on a single house 17801" effort visible elsewhere in
Qin's Drive, not a Johns Lake shoreline ground-truth campaign. It may
include a strip of lake frontage if the property is lakefront, but it
should be scoped as a **house/property-scale validation source**, not a
lake-extent or whole-shoreline one — don't expect it to validate the
144.47 ha lake-area figure broadly.

**Validation routine design (file format/coverage now known; not yet
built):**
1. Extract the orthomosaic GeoTIFF + camera calibration from the OPF zip
   (Pix4D OPF is a documented open spec — no need to unzip the full
   archive, just the relevant project files) and reproject to EPSG:5070
   the same way S2 bands are reprojected today
   (`rasterio.warp.reproject`, nearest).
2. If the ~100–150 m footprint includes any lake edge, extract a water/land
   boundary from the RGB orthomosaic (no NIR/SWIR on a DJI RGB drone, so
   MNDWI doesn't apply directly — would need an RGB color/texture heuristic
   or manual trace).
3. Where it overlaps an OWM Sentinel-2 mask date, compute shoreline offset
   (m) and any local area discrepancy — but given the small footprint, this
   is a **single-point spot-check at one property edge**, not a substitute
   for the AOI-wide precision question this whole track is about.
4. Realistically, this dataset's best near-term use is **validating the
   DEM/terrain model at house-scale** (matches the existing
   "single house" flood-modeling effort) rather than lake-shoreline OWM
   precision.

**Correction 2026-06-18, re-reading Lance's actual emails closely:** the
house-scale footprint is **not a mismatch to flag** — re-checking his exact
wording, this is most likely exactly what he intended. His first email
(2026-06-16) explicitly says *"I think our high-resolution drone imagery at
John's Lake is not going to help with [precise lake/river extent
tracking]"* — he'd already concluded drone imagery wasn't the tool for
lake-wide tracking before this investigation even started. His second email
(2026-06-17) then reframes drone RTK's role as *"something we can do at a
small scale over a period of time to 'validate'/'fine-tune' model"* — small
scale, explicitly. The GPS-confirmed ~100–150m single-residence footprint
matches that stated scope, and it's centered almost exactly on the AOI/target
address (28.5216, -81.6570) — i.e. this is very likely the intended
validation site, not an accidental wrong dataset. **Lake-wide tracking is
PlanetScope's job in his framing, not drone RTK's** — see the project-status
report for the full breakdown of which dataset answers which of his asks.

---

## 3. USGS gauge cross-check

**`USGS-02234344` is NOT relevant to Johns Lake.** Fetched
`waterdata.usgs.gov/monitoring-location/USGS-02234344/` directly: it is
**"Howell Creek at State Hwy 434 Near Oviedo, FL"** — Seminole County, on the
*east* side of Orlando, part of the Econlockhatchee River basin draining to
the St. Johns River. Johns Lake (Winter Garden, 28.52°N/-81.63°W) is on the
*west* side, in the **Lake Apopka watershed** (confirmed via
orange.wateratlas.usf.edu) — a different basin roughly 25–30 miles away. This
confirms the plan's suspicion: the URL Lance shared was very likely a generic
/ wrong example link, not a deliberate pointer to a Johns-Lake-relevant gauge.
**Conclusion: not usable. Do not use this station for any Johns Lake
cross-check.**

**Searched for a real alternative:**
- `USGS-02237522` is **not** "Johns Lake at Oakland, FL" as an initial search
  snippet suggested — fetching the actual page shows it is **"Dead River NR
  Tavares, Fla."**, also unrelated. (Caught by verifying the live page rather
  than trusting the search-result snippet — worth flagging generally: USGS
  station-number search results can mis-attribute names.)
- No USGS gauge specifically on Johns Lake was found via web search.
- **`johnslakeflorida.com/johns-lake-water-level/`** (Johns Lake Association)
  references a single point reading: lake level **≈99.36 ft NAVD88 on
  2018-08-04**, and a FEMA 100-yr basin flood elevation of **99.7 ft NAVD88**.
  Converting 99.36 ft → **30.28 m NAVD88**. The repo's authoritative WSE is
  **28.74 m NAVD88** (`dem/data/lake_volume.csv`) — a **~1.54 m (5.05 ft)**
  difference. This is a real, citable discrepancy worth investigating, though
  it's a single 2018 reading vs. a multi-year (2016–2026) satellite-derived
  consensus, so it's not a clean apples-to-apples cross-check — could reflect
  real multi-year lake-level change, a datum/measurement-point difference, or
  an error in either source.
- **`orange.wateratlas.usf.edu`** mentions an **"Orange County Stormwater
  Water Levels Sampling" station labeled "JOHNS"** that tracks Johns Lake
  water levels directly. This is the most promising lead for an actual
  continuous, site-specific water-level record — but it's an Orange County
  system, not USGS, so it likely needs a different data portal/API than
  `waterdata.usgs.gov`. **Not yet pulled — next step if this cross-check is
  prioritized.**
- SJRWMD has an active **Minimum Flows and Levels (MFL) determination** for
  Johns Lake (`sjrwmd.com/minimumflowsandlevels/johns-lake/`), draft report in
  peer review as of the page's last update — no numeric values published yet
  on the public page, but this is the eventual authoritative regulatory
  source once finalized.

**Update 2026-06-18, same day — chased the "JOHNS" lead:** confirmed via web
search + direct page fetch. Johns Lake is **waterbody ID 7935** on the Orange
County Water Atlas (`orange.wateratlas.usf.edu/waterbodies/lakes/7935/johns-lake`,
also mirrored on `lake.wateratlas.usf.edu` since the lake straddles Lake/Orange
county lines). That page confirms a continuous hydrologic sampling location
named **"JOHNS"** under "Orange County Stormwater Water Levels Sampling", and
states the lake's documented surface area as **2,580 acres** — independently
corroborating the AOI-coverage finding in §4 below.

**Data access is not a simple API.** The download tool
(`orange.wateratlas.usf.edu/data-download/?wbodyid=7935`) is a 3-step
interactive workflow (map-select station → review/configure date range →
download/graph), with results delivered by **email**, not a direct GET/POST
query string. No public REST API surfaced. To actually pull this data: use
the map-select step to pick the "JOHNS" station, set a date range (blank =
all available history), and request the export — this is a manual/browser
step, not something to script blindly without first seeing what the export
file looks like.

**Bottom line for task 3: a real, named, continuously-monitored Johns Lake
gauge ("JOHNS", waterbody 7935) now confirmed to exist** — upgrade from "no
lead" to "lead found, needs one manual export to get data in hand." The
johnslakeflorida.com single 2018 reading (~1.5 m gap vs. our WSE) remains a
useful independent sanity check until the JOHNS time series is pulled.

---

## 4. OWM seasonality + full-lake coverage check

Script: `ground_truth/seasonal_area_split.py`. Run: `python3
ground_truth/seasonal_area_split.py`.

| season | n   | mean_ha | median_ha | stdev_ha | min_ha | max_ha |
|--------|-----|---------|-----------|----------|--------|--------|
| wet (Jun–Sep) | 23  | 120.71 | 121.35 | 7.58 | 95.48 | 132.61 |
| dry (Oct–May) | 130 | 122.39 | 123.30 | 6.45 | 77.05 | 140.30 |

**Seasonality is not a meaningful bias source.** Wet-season mean is actually
slightly *lower* than dry-season mean (−1.68 ha, −1.4%), opposite of naive
expectation and well within scene-to-scene noise (stdev ~6–8 ha on both
sides). Sample is heavily dry-skewed (23 wet vs. 130 dry scenes over
2016–2026), so the wet-season estimate is less stable, but the magnitude is
small enough that re-balancing wouldn't change the conclusion.

**A much bigger, unrelated gap turned up:** the all-scene OWM mean across all
153 scenes is **122.14 ha**, but the "authoritative" figure in
`dem/data/lake_volume.csv` is **144.47 ha** — a **+22.3 ha (+18%)** gap that
has nothing to do with season. This is most likely explained by methodology,
not error: `lake_volume.csv`'s figure comes from the **majority-vote
consensus mask** (`lake_mask.tif`, pixels wet in ≥50% of 153 scenes) plus
largest-connected-component filtering, which is a different quantity than
**individual per-scene OWM polygon area** — a majority-vote mask can include
marginal/fringe pixels that are water in most-but-not-all scenes, pulling the
consensus area above the typical single-scene area. Worth a one-line note in
CLAUDE.md or the data catalogue clarifying these are two different
methodologies, not two measurements of the same thing that disagree.

**Full-lake coverage — confirmed problem, not just a hypothesis:**
Johns Lake's actual documented surface area (USF Water Atlas /
orange.wateratlas.usf.edu, cross-checked against multiple independent
sources) is **2,580 acres ≈ 1,044 ha**. The current AOI
(`dem/data/lake_mask.tif`, 868×868 cells @ ~2.64×2.62 m) covers only
**521.87 ha total** — already smaller than the full lake by more than 2×,
before even subtracting non-lake land area within the AOI. Checked directly:
water-mask pixels touch the AOI boundary on the **north edge (502 cells),
west edge (450 cells), and east edge (127 cells)** — only the south edge is
essentially clear (5 cells). **This confirms the lake is being clipped by the
current download extent on three sides, not merely smaller than reported.**
This means:
- The 144.47 ha / 203,740-cell "Johns Lake" figure in this repo is the
  area of one (likely the southern/main) basin near the target address, not
  the full lake.
- Per the plan's note, this is a separate problem from precision/resolution —
  it needs a **wider DEM + Sentinel-2 download** (larger AOI), not a
  seasonality or methodology fix, if "extend coverage for the whole lake" is
  pursued.

---

## 5. Clay foundation model — scoping

Checked `sentinel2/compare_methods.py`: adding a method today is a small,
mechanical change — add `"Clay": "clay_mask_{date}.tif"` to the `METHODS`
dict (`compare_methods.py:34`) and a color to `METHOD_COLORS`
(`compare_methods.py:42`), same pattern the 4 existing methods follow, plus
the equivalent entries in `visualize_comparison.py` / other
`method_comparison*.py` scripts.

**The real cost is producing `clay_mask_{date}.tif` in the first place, and
it's not a small add.** Verified via search (2026-06-18): Clay
(Clay Foundation Model, v1.5, 632M-param ViT) is a **self-supervised,
embeddings-only** model — masked-autoencoder pretrained, with **no shipped
water-segmentation head**. Unlike OmniWaterMask (ships a ready
water-specific classifier) or this repo's existing Prithvi-EO-2.0 path
(already has a working segmentation pipeline via terratorch per
`s2_water_segment_v2.py`), getting a binary water mask out of Clay requires
**training a task-specific decoder/head on top of the frozen encoder** —
i.e. assembling a labeled training set (could bootstrap from existing
OWM/NHD masks as pseudo-labels), writing a training loop, and tuning it. This
is genuinely a multi-day ML task, not a "wire up another inference script"
afternoon like the existing 4 methods were.

**Recommendation:** deprioritize, consistent with the plan's framing — even a
successful Clay integration still runs on the same 10 m Sentinel-2 input, so
it cannot address the actual concern motivating this whole track (resolution/
precision), only method diversity on the comparison chart. Worth revisiting
only if there's a specific reason to believe Clay's embeddings generalize
better to Johns Lake's turbid/vegetated-fringe water than Prithvi/OWM do —
no evidence either way without running it.

---

## Summary table

| Task | One-line verdict |
|---|---|
| 1. PlanetScope | ~2–5 m realistic shoreline precision; get a real per-AOI quote before assuming $900/triplet — found pricing models disagree by 10×. |
| 2. Drone RTK | Access confirmed: 595 RTK/PPK photos + 2 processed Pix4D OPF archives (~40 GB, not downloaded). GPS-verified footprint is ~100-150 m, single-residence scale, centered on the AOI/target address — matches Lance's own stated "small scale validate/fine-tune" framing, not a mismatch. Not usable for lake-wide tracking by design, not by accident. |
| 3. USGS-02234344 | **Not relevant** (Howell Creek, Oviedo — wrong watershed). **Update:** found the real lead — "JOHNS" station, waterbody 7935, Orange County Water Atlas. Data export is manual (email-delivered), not yet pulled. |
| 4. Seasonality | Small, not a real bias (−1.4%). Bigger finding: AOI captures less than half of Johns Lake's true extent (521.87 ha AOI vs. 1,044 ha actual lake), confirmed by mask pixels clipped on 3 of 4 AOI edges. |
| 5. Clay | Mechanical to wire into comparison scripts; hard part is training a segmentation head from scratch (Clay ships embeddings only). Low priority — doesn't fix the resolution problem. |
