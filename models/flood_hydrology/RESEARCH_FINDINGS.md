# Research findings — datasets & models/software benchmarking

Consolidated reference, generated 2026-06-18 from a backlog of investigation
tracks (`plans/track_D_ground_truth_data.md`, `plans/track_E_benchmark_literature.md`,
`plans/track_G_aoi_expansion.md`) that were spun up in response to team-lead
meeting notes and two emails from Lance Legel (CEO) raising the same concern
from two angles: **is Sentinel-2 precise enough, and how does our flood
solver compare to the civil-engineering standard?** This file is the
single-page answer; the track files have full depth/sourcing per item.

---

## Part 1 — Datasets

### Sentinel-2 (current primary water-segmentation input)

- **10m resolution.** This repo's whole water-segmentation stack (MNDWI,
  WatNet, Prithvi-EO-2.0, OmniWaterMask) runs on Sentinel-2 L2A bands at this
  resolution. OmniWaterMask is the best of the four (F1=0.882 vs. NHD), but
  no amount of model improvement raises the underlying pixel resolution.
- **The actual concern, in Lance's own words** (email, 2026-06-16): not
  "can we detect a major flood" but *"we can see even slightly higher (+/- a
  few feet of extension) levels of lakes and rivers... It's about having a
  very precise way to do this."* He reframes "flood" as *"probability that
  any location will be inundated"* rather than a binary event. 10m pixels
  can't resolve foot-scale shoreline movement — this is a real, not
  imagined, limitation for that specific goal.
- **Two unrelated data-quality findings surfaced while checking this**
  (full detail: `ground_truth/track_D_findings.md` §4):
  1. The repo's "authoritative" 144.47 ha lake area
     (`dem/data/lake_volume.csv`) and the per-scene OWM average (~122 ha,
     `sentinel2/data/water_extent_timeseries.csv`) are **two different
     methodologies** (majority-vote consensus mask vs. individual-scene
     polygon area), not a real disagreement.
  2. **The current 2×2km AOI clips Johns Lake on 3 of 4 edges** —
     confirmed by reading `lake_mask.tif` directly (502/450/127/5 wet
     pixels on north/west/east/south). Johns Lake's real area is ~1,044 ha
     (Water Atlas) / 989.8 ha (NHD, independently re-confirmed), roughly
     2× the current 521.87 ha AOI. See Track G below — this is now fixed at
     the validation-scoping level, not yet executed.

### PlanetScope (Planet Labs)

- **3m resolution, near-daily revisit since 2020.** Reviewed 29 sample
  images Lance shared (`~/Desktop/PlanetScope/` — general marketing/example
  gallery, none are Johns Lake, but several show flood/lake scenes at
  comparable resolution: `LakeOroville_Dam`, `FloodingErfstadtGermany`).
- **Realistic shoreline precision: ~2–5m** for Johns Lake's turbid,
  vegetated-fringe water (better case ~1–3m for clean spectral edges
  elsewhere) — meaningfully better than Sentinel-2's 10m, in the range
  Lance's "+/- a few feet" ask needs.
- **Pricing doesn't match what Lance was quoted.** Planet's own published
  direct-order pricing is $1.80/km² with a **250 km² minimum** (~$450/order
  regardless of AOI size) — Lance's $300/image, 3-image-minimum, ~100km²
  figure is a different pricing structure entirely (likely a reseller, e.g.
  Apollo Mapping-style archive sales, not Planet's direct tasking product).
  **Get a real, same-source, in-writing quote for the specific AOI before
  budgeting** — the two structures found differ by an order of magnitude in
  assumptions.
- **Revisit doesn't guarantee a clear "during" shot.** Daily/sub-daily
  revisit is real, but Florida wet-season convective storms bring same-day
  cloud cover — the same risk already documented for Sentinel-2 in this
  repo. A before/after pair around an event is realistic; a clean shot
  *during* the event is not guaranteed.
- **Role in the project (per Lance's own framing):** the broad,
  lake/river-wide higher-resolution *replacement* for Sentinel-2's tracking
  job — not a validation-only tool.

### Drone RTK imagery (team's own capture, Winter Garden residence)

- **Access confirmed** via Google Drive (`Winter_Garden_Residence` folder,
  shared 2026-06-17). Contents: 595 RTK/PPK JPGs (5280×3956, captured
  2025-04-28) + RTK/PPK GNSS log files (confirms genuine survey-grade
  capture, not consumer GPS) + 2 already-processed Pix4D OPF photogrammetry
  archives (~40 GB combined, not downloaded — only headers/EXIF inspected).
- **GPS-confirmed footprint: ~100–150m across**, centered almost exactly on
  the project's target address (28.5216, -81.6570) — a single-residence
  scale capture, not a lake-wide or shoreline-length survey.
- **This is not a mismatch — re-reading Lance's emails closely, it's
  exactly what he scoped it for.** His first email (2026-06-16) already
  says *"I think our high-resolution drone imagery at Johns Lake is not
  going to help with [precise lake/river extent tracking]."* His second
  email (2026-06-17) reframes its role as *"something we can do at a
  **small scale** over a period of time to validate/fine-tune model."*
  **PlanetScope is his answer to the lake-wide tracking question; drone RTK
  was never meant to be.**
- **Role in the project:** cm-precision validation/fine-tuning of the
  DEM/terrain model at the one target property — not a lake-extent ground
  truth source. A validation routine (extract orthomosaic from the OPF zip,
  compare against same-date OWM mask at the property edge) is designed but
  not yet built — see `ground_truth/track_D_findings.md` §2 for the exact
  steps.

### USGS / regional gauge data

- **`USGS-02234344` (the link Lance shared) is the wrong watershed** —
  confirmed by fetching the live page directly: it's "Howell Creek at State
  Hwy 434 Near Oviedo, FL," in the Econlockhatchee/St. Johns basin, ~25–30
  miles from Johns Lake's Apopka watershed. A second guess
  (`USGS-02237522`) was also wrong on inspection ("Dead River NR Tavares").
  **No USGS gauge specifically on Johns Lake exists.**
- **Real lead found instead:** Orange County Water Atlas waterbody ID
  **7935** (`orange.wateratlas.usf.edu/waterbodies/lakes/7935/johns-lake`)
  has a continuous monitoring station named **"JOHNS"** under "Orange
  County Stormwater Water Levels Sampling." Confirms the documented lake
  area (2,580 acres) independently. **Not a USGS system — no public API;
  data export is a manual 3-step map-select + email-delivery workflow**, not
  yet pulled.
- **Single independent sanity-check point:** `johnslakeflorida.com` cites a
  2018-08-04 reading of 99.36 ft NAVD88 (= 30.28m) vs. this repo's
  satellite-derived WSE of 28.74m — a ~1.5m gap, not yet resolved (could be
  real multi-year change, a datum difference, or an error in either
  source). SJRWMD also has an active Minimum Flows and Levels determination
  for Johns Lake, draft in peer review, no public numbers yet.

### Clay Foundation Model (mentioned in meeting notes)

- v1.5, 632M-param ViT, **self-supervised embeddings-only** — no shipped
  water-segmentation head, unlike OmniWaterMask or this repo's existing
  Prithvi-EO-2.0 path. Adding it as a 5th comparison method is mechanically
  trivial (`sentinel2/compare_methods.py`) but the real cost is training a
  segmentation decoder from scratch — a multi-day ML task, not an afternoon.
- **Deprioritized**: even a successful integration still runs on the same
  10m Sentinel-2 input, so it can't address the actual resolution concern
  driving this whole investigation — only adds method diversity.

### GSDR / NOAA Atlas 14 / NOAA CDO (precipitation — already integrated)

No open questions here; documented in full in `CLAUDE.md`'s "Precipitation
data sources" table. Noted for completeness: GSDR (US_086638, 1942–1985) is
what the `historical_gsdr` and `historical_gsdr_extreme` simulation
scenarios are built from; NOAA CDO is what `historical_20240212` uses
(works now, but the gauge only captured 1.3mm for that window — a real
data-coverage gap, not a bug).

---

## Part 2 — Models / software benchmarking

### HEC-RAS (USACE) — the civil-engineering gold standard

Re-verified directly against the two specific URLs the team lead gave
(`hec.usace.army.mil/software/hec-ras/2025/` and `.../download.aspx`), not
just general background knowledge:

- **Free and open**, confirmed in the page's own words: *"Use is not
  restricted and individuals outside of USACE may use the program without
  charge"* (no support for non-USACE users, but no access gate).
- **Current stable: HEC-RAS 7.0.1.** HEC-RAS 2025 is a separate **Beta**
  track (entered beta April 2026, running behind its own original Fall-2025
  target) — a UI/performance overhaul (new mesh types, explicit solver,
  cloud/Linux support, 10–50× faster raster I/O), **not a new physics
  model**. Its own page says *"please don't"* use it for production yet.
- **Governing equations** (sourced from the actual 2D User's Manual, not
  just summary pages): HEC-RAS 2D toggles between full **Saint-Venant
  (Shallow Water)** and **Diffusion Wave**, both via implicit finite-volume.
  Diffusion Wave drops local acceleration (∂v/∂t) and convective
  acceleration (v·∂v/∂x) — the manual calls these *"extremely important...
  to model rapidly rising flood waves accurately."* HEC-RAS's own guidance
  names **8 cases recommending full SWE over Diffusion Wave**, and **flash
  floods are explicitly one of them** — directly relevant, since two of our
  four scenario types are literally named "flash."
- **Infiltration: 3 methods offered — Deficit-Constant, SCS Curve Number,
  Green-Ampt. Not Horton**, which is what this repo uses. Not a flaw
  (Horton is standard elsewhere, e.g. EPA SWMM) — a real methodology
  difference worth knowing if ever comparing numbers directly.
- **Where `flood_sim.py` sits:** the local-inertia approximation (Bates et
  al. 2010, LISFLOOD-FP) keeps local acceleration but drops convective
  acceleration — i.e. it sits *between* HEC-RAS's two options. The existing
  CLAUDE.md caveat ("Fr > 1 in channels → advection terms matter") is now
  backed by HEC-RAS's own stated rationale for why that term matters, and
  by the fact our flash scenarios are exactly where HEC-RAS says it matters
  most.
- **Concrete next step, if pursued:** a HEC-RAS 7.0.1 full-SWE reference
  run on just the flash scenarios at the Johns Lake AOI — the single most
  targeted version of this ask, since that's precisely where the gap should
  show up by HEC-RAS's own criteria. Real time investment (DEM import, mesh
  generation, boundary conditions via HEC-RAS's GUI workflow) — an explicit
  decision for Qin/Lance, not started.

### NVIDIA HydroGraphNet (`physicsnemo/examples/weather/flood_modeling/hydrographnet`)

- **Graph Neural Network surrogate** (autoregressive encoder-processor-
  decoder on unstructured mesh graphs) — not a differentiable numerical SWE
  solver. "Shallow water equation learnable by the model" (meeting-notes
  paraphrase) is more precisely: a **physics-informed loss** enforcing mass
  conservation (volume-continuity inequality) + a "pushforward trick" to
  limit autoregressive error growth. Optionally swaps MLPs for
  Kolmogorov-Arnold Networks for interpretability.
- **Trained against HEC-RAS output itself** (White River case study, 4,787
  mesh nodes) — not observational/satellite data.
- **Not a drop-in module.** It needs HEC-RAS-quality ground truth at our
  specific AOI to train meaningfully, which we don't have (our solver *is*
  the simulator — there's no independent high-fidelity reference to train
  against without first running HEC-RAS ourselves). The realistic near-term
  value is conceptual — the mass-conservation loss pattern is reusable *if*
  we ever build a neural surrogate for `_local_inertia_step_torch` — not a
  near-term integration task.

### Our current segmentation models (for context, already in production)

OmniWaterMask (F1=0.882 vs NHD) > Prithvi-EO-2.0 (0.814) ≈ WatNet (0.813) >
MNDWI (0.741), all on the same 10m Sentinel-2 input — the resolution
ceiling discussed in Part 1 applies equally to all four; none of this
ranking changes which dataset problem needs solving.

---

## Cross-cutting recommendation

Two genuinely independent open decisions came out of this investigation,
both deliberately left as team decisions rather than started speculatively:

1. **AOI expansion** (`plans/track_G_aoi_expansion.md`) — the cheap
   validation step is now done and passed (new 7.37×3.87km box, zero edge
   clipping). The expensive part (re-running Sentinel-2 + 4-method
   segmentation at the new extent) is blocked on *when* to spend the
   multi-hour-to-multi-day effort, and *what dataset* to re-run with —
   continue Sentinel-2, or use this as the moment to bring in PlanetScope
   for the expanded ROI, given the resolution findings above.
2. **HEC-RAS reference run** — whether the physics-fidelity gap is worth
   the time investment to quantify directly, and whether that ground truth
   would be worth generating toward a future HydroGraphNet-style surrogate.

Both are flagged in their respective track files (`track_G_aoi_expansion.md`,
`track_E_benchmark_literature.md`) as explicit go/no-go points for
Qin/Lance/team lead, not work items to pick up opportunistically.
