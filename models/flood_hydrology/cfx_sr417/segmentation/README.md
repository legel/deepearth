# segmentation — measured surface parameters for the physics mesh

The Surface Parameterization section of [`../../NEXT_STEPS.md`](../../NEXT_STEPS.md). Replaces
two assumptions in the solver with measurements:

1. **Tree canopy was absent from the flow surface entirely** — the mesh was bare-earth DEM plus
   LiDAR building roofs, with no vegetation in it at all.
2. **Ground Manning's *n* was a single uniform 0.040** everywhere non-roof (roofs 0.015). That
   scalar was the whole spatial variation in the model.

Everything here is scored against the same two numbers the rest of the project uses — rising
limb and runoff coefficient at USGS 02234400 for Hurricane Ian — using
[`../analysis/validate_gauge_site3.py`](../analysis/validate_gauge_site3.py)'s own metric
functions.

---

## Pipeline

```
NAIP 0.6 m RGB/NIR/NDVI ─┐
2018 LiDAR point cloud ──┼─→ canopy_lidar.py      → chm_2m.tif, canopy_cover_2m.tif
OSM roads + buildings ───┤
HAND (hydro chain) ──────┴─→ segment_naip.py      → landcover_0.6m.tif, segments.csv
                                    │
                            surface_parameters.py → surface_parameters.json
                                    │
                            rasterize_parameters.py → manning_n_5m.tif
                                                      surface_storage_5m.tif
                                                      impervious_frac_5m.tif
                                    │
                        run_site3_ian_segmented.py → controlled A/B against the gauge
                                    │
                              qc_preview.py        → visual check
                          validate_canopy.py       → canopy vs NLCD Tree Canopy Cover
```

```bash
python3 segmentation/canopy_lidar.py --site site3          # ~8 min, checkpointed per tile
python3 segmentation/segment_naip.py --site site3          # ~4 min
python3 segmentation/surface_parameters.py
python3 segmentation/rasterize_parameters.py --site site3
python3 segmentation/qc_preview.py --site site3
python3 segmentation/validate_canopy.py --site site3       # independent canopy check
python3 segmentation/run_site3_ian_segmented.py            # ~25 min, three solver runs
```

---

## Three findings that changed the approach, all measured

**1. The LiDAR has no vegetation classes.** `NEXT_STEPS.md` says the 2018 point cloud carries
vegetation returns in ASPRS classes 3/4/5 and points at `load_cached_points`'s
`classification_filter`. Those classes do not exist in this acquisition. Across 48,055,738
points sampled from 7 of site3's 31 cached tiles the histogram is class 1 (unclassified) 61.3 %,
class 2 (ground) 32.6 %, class 6 (building) 5.3 %, with the remainder noise/water/bridge.
`lidar/data/classification_histogram.json` shows the same for the main AOI. Filtering on 3/4/5
returns nothing. Canopy is therefore built from class-1 returns normalised against the bare-earth
DEM — the standard normalised-DSM construction, and the only one this data supports.

**2. SAM3 ran — access approved 2026-08-28, full scene complete.**

`facebook/sam3` is a `gated: manual` repo; access was requested and granted on 2026-08-28. Two
further blockers had to be cleared first, and neither was hardware: `Sam3Model` ships only in
transformers **5.x**, which requires Python >= 3.10, so it cannot go in the 3.9.6 pipeline
interpreter (there is no 4.58 — the series runs 4.57.x -> 5.0.0). It runs from the 3.11 venv
(`cfx_sr417/.venv`, transformers 5.16.1, torch 2.13 + MPS) as a standalone stage writing
`landcover_0.6m_site3_sam3.tif`, which the 3.9 pipeline consumes unchanged. Weights are 3.44 GB
against 17.2 GB of unified memory; the MPS out-of-memory result recorded elsewhere in this
project was the 8.67M-edge mesh GNN, an unrelated workload.

**The run.** 121 tiles at 1024 px with 64 px overlap over the full 36 km² scene, 31 text prompts
per tile (9 classes x 3-4 phrasings), **25.4 minutes** on MPS.

Two pieces of engineering made that time possible and are worth keeping:

* **The vision encoder runs once per tile, not once per prompt.** SAM3's forward accepts
  `vision_embeds` in place of `pixel_values`, and the image does not change between prompts, so
  re-encoding it 31 times was pure waste. Measured **2.38x faster with bit-identical masks and
  scores** — verified against the pre-optimisation output, where every differing pixel fell
  inside a tile-overlap strip the two runs covered with different numbers of tiles, and outside
  that strip the two rasters were identical.
* **Per-tile checkpoints**, atomic, with corrupt-file recovery — the same pattern
  `cache_bbox_points.py` uses, for the same reason: long jobs in this environment have been
  killed mid-run with no traceback.

**Result, against the spectral backend over the same scene:**

| class | SAM3 | spectral + LiDAR |
|---|---|---|
| `tree_canopy` | 39.3 % | 44.2 % |
| `building_roof` | 11.6 % | 8.1 % |
| `road_paved` | 11.6 % | 10.2 % |
| `water` | 5.9 % | 10.3 % |
| `grass_turf` | 7.9 % | 8.9 % |
| `shrub_scrub` | 3.8 % | 5.3 % |
| `wetland_marsh` | **0.0 %** | 7.5 % |
| **unlabelled** | **17.4 %** | 0.5 % |

Agreement where both label a pixel: **69.8 %**. On the 5 m grid the SAM3 field gives mean
*n* = 0.0594 (spectral 0.0655, scalar 0.040) over 69.6 % classified cover, the remainder falling
back to the scalar.

**What SAM3 is better at, and what it is worse at — both visible in
`data/backend_comparison_site3.png`.** Its object delineation is markedly better: individual
rooftops come out as clean countable polygons following real building outlines, roads as
continuous correctly-curved ribbons, lake shorelines sharp, where the spectral backend is blocky
and fragmented. Against that it labels only what it detects above threshold, so 17.4 % of the
scene is left unlabelled, and it finds **no wetland at all** — that class comes from HAND, a
terrain product SAM3 cannot see.

**It over-detects buildings by 2.4x.** SAM3 calls 16.0 % of the domain `building_roof` against
OSM's 6.8 % of mapped footprint area and the spectral backend's 8.1 %. Total impervious is 24.4 %
against NLCD's independent 18.7 %. Some of the excess is real — OSM does miss structures, and
the imagery shows them — but not a factor of two of it. This measurement is independent of the
solver and stands.

**A correction, recorded because the reasoning was wrong and the wrong reasoning was persuasive.**
The first SAM3 solver arm showed heavy early ponding, and this was attributed here to the
impervious over-detection making ~20 % of cells near-frictionless. That diagnosis was **wrong**.
Re-running the *scalar* baseline — n = 0.040 everywhere, no SAM3 involved — reproduced the same
behaviour, so the cause was a concurrent solver change, not the parameter field. The
over-detection is real; its supposed consequence was not.

The pipeline is built so that SAM3 drops in behind an unchanged contract. What the solver
consumes is a per-cell parameter, and a parameter belongs to the surface *class*, not the
instance: two adjacent oak crowns get the same Manning's *n* whether SAM3 calls them one segment
or two. So the interface between step (a) and step (b) is `landcover_0.6m.tif` +
`segments.csv`, and nothing downstream cares which segmenter produced them.

**3. Two of the four requested parameters should not come from segmentation alone.** A
vision-language parameterisation of this kind is asked for `{material, Smax, Ks, Manning's n}`. `Ks` and soil `Smax` are already in the solver from
SSURGO — an actual soil survey with measured conductivity and depth-to-water-table per map unit
(28 units at site3). A vision model looking at a lawn cannot tell Immokalee sand from Basinger
sand, and depth to the seasonal-high water table is not visible from above at all. Overwriting
them with VLM estimates would be a downgrade.

What imagery genuinely knows, and SSURGO does not, is what is *on* the ground. So this stage
supplies **Manning's n** (SSURGO says nothing about roughness), **surface storage** (canopy and
litter interception plus roof/pavement depression storage — a different, additive store from
SSURGO's soil storage, named separately so the two are not conflated), and a **0.6 m impervious
fraction** to feed the existing infiltration machinery in place of NLCD's 30 m one.

---

## What is measured versus inferred

The classification's robustness rests on most of the domain being decided by evidence that is
not the imagery:

| decided by | classes | share of domain |
|---|---|---|
| LiDAR canopy height + cover | `tree_canopy`, `shrub_scrub` | **49.8 %** |
| OSM footprints (the same ones the solver's impervious mask uses) | `building_roof`, `road_paved` | **18.3 %** |
| NIR absorption | `water` | **10.4 %** |
| NDVI threshold (+ HAND for wetland) | `grass_turf`, `bare_soil`, `impervious_other`, `wetland_marsh` | **21.6 %** |

**78 % of the domain is classified by evidence that is not the imagery's spectral response** —
LiDAR return heights and mapped footprints. Only the last row turns on the NDVI split, and
within it the hydrologically consequential distinction is `grass_turf` (n = 0.040) versus
`bare_soil` (n = 0.025) — a small difference between two pervious low-roughness covers. So the
weakest link in the classification governs the least of the physics, which is the property that
makes a winter-NAIP NDVI threshold acceptable here at all.

**The NDVI threshold is calibrated, not assumed.** NAIP for site3 was flown 2021-12-02, in
winter, when central Florida turf is partly dormant; a textbook cutoff would not transfer. The
split is the equal-error point between two independently-labelled populations already on disk —
LiDAR canopy above 3 m (definitely vegetation) and OSM building interiors (definitely not) —
written to `data/calibration_site3.json` with both distributions and the resulting error rate.

---

## Known limitations

- **LiDAR coverage is a rotated inset of the domain.** The cached point bbox was built as a
  true-north box in EPSG:2881 while the solver grid is EPSG:5070 Albers, which is rotated ~8.8°
  here, and the cache radius (2.99 km) is smaller than the DEM box (6.86 km). Result: 72.7 % of
  the domain has returns. Inside the delineated gauge watershed — the part that produces the
  signal being scored — coverage is **97.4 %**. Cells with no returns get canopy height and
  cover of zero, so canopy is under-called on the domain margins.
- **The canopy height statistic is discretised.** It is the highest of a fixed set of height
  thresholds still exceeded by ≥2 % of a cell's returns, not a continuous p98. Precise enough
  for roughness classes, not for forestry.
- **`impervious_other` is spectrally inferred**, unlike roads and roofs, so it carries an
  impervious fraction of 0.90 rather than 1.00.
- **Surface storage is produced but not wired into the solver.** See the results below for its
  size; adding it would require a genuinely new solver term rather than a substitution at an
  existing interface, which is not justified by the magnitude.

---

## Results

### Classification, and two independent cross-checks

343,980 segments over 35.99 km² of NAIP coverage. Domain area fractions on the 5 m solver grid:

| class | area | Manning's *n* |
|---|---|---|
| `tree_canopy` | 44.4 % | 0.120 |
| `water` | 10.4 % | 0.035 |
| `road_paved` | 10.2 % | 0.013 |
| `grass_turf` | 9.0 % | 0.040 |
| `building_roof` | 8.1 % | 0.015 |
| `wetland_marsh` | 7.5 % | 0.080 |
| `shrub_scrub` | 5.4 % | 0.070 |
| `impervious_other` | 3.5 % | 0.016 |
| `bare_soil` | 1.6 % | 0.025 |

Neither of these checks was used to build the classification, so both are genuine tests of it:

| | segmentation | independent source |
|---|---|---|
| total impervious (`road_paved` + `building_roof` + `impervious_other`) | **21.8 %** | **18.7 %** — NLCD 2021 impervious, domain mean |
| open water | **10.4 %** | **13.6 %** — 3DHP mapped waterbodies, clipped to the domain |

Impervious agrees within ~3 pp. Water is *under*-called relative to 3DHP rather than over-called,
which was the failure mode worth worrying about (dark shadows read as water); site3 sits in the
Longwood lake district, where 3DHP maps 167 waterbodies, so a double-digit water fraction is
real.

The NDVI calibration separated its two labelled populations at an **8.4 % equal-error rate**,
with the threshold at −0.020 — far below any textbook value, which is what the December
acquisition demands: canopy NDVI runs p10/p50/p90 = 0.012 / 0.270 / 0.415 while roof NDVI runs
−0.284 / −0.173 / −0.022.

### Canopy validated independently — NLCD Tree Canopy Cover

`tree_canopy` is 44.4 % of the domain and carries the largest parameter change here (n = 0.120
against a 0.040 scalar). Impervious and water each had an independent cross-check; canopy did
not, which left the most consequential class resting on this project's own height model alone.

**NLCD Tree Canopy Cover 2021 (30 m)** closes that. USFS/MRLC derive it from Landsat time series
trained on FIA plots — no shared sensor, platform or method with a 2018 airborne LiDAR
return-height statistic, so agreement is evidence rather than restatement. Compared on TCC's own
30 m grid (upscaling the fine measurement, not interpolating the coarse one down), over 40,957
cells with ≥95 % LiDAR coverage:

| | value |
|---|---|
| domain-mean canopy fraction, ours | **34.1 %** |
| domain-mean canopy fraction, NLCD TCC | **33.9 %** |
| correlation *r* | 0.751 |
| bias | +1.6 pp |
| MAE / RMSE | 17.5 / 22.6 pp |
| class agreement at a 50 % threshold | 78.4 % (IoU 0.554, F1 0.713) |

**The aggregate agreement is near-exact; the disagreement is about *where* canopy is, not how
much of it there is.** 34.1 % against 33.9 % over 46 km² is far closer than two independent
products three years and an order of magnitude in resolution apart have any right to be. The
17.5 pp per-cell MAE is real scatter, and most of it is explainable: TCC's 30 m pixel cannot
resolve a hedgerow or a street tree that 2 m LiDAR sees plainly, TCC's own distribution is
strongly bimodal (p10/p50/p90 = 0/21/95) against our smoother 3/35/74, and the two acquisitions
are three years apart.

That bimodality is also why the class-threshold counts look lopsided — 7,371 cells tree only in
ours against 1,464 only in TCC. With matched means, an asymmetry at a 50 % cut is a threshold
artifact of comparing a saturating product against a continuous one, not evidence of systematic
over-calling.

**What the residual disagreement is worth, as a strict bound.** Capping our tree fraction at
TCC's wherever ours is higher — and never raising it where ours is lower — lowers domain-mean
Manning's *n* from 0.0655 to 0.0543. Against the +0.0255 the segmentation introduced over the
0.040 scalar, that puts an upper bound of **44 %** on how much of the roughness change could be
attributed to canopy placement error. The operation is deliberately one-sided, so it overstates
the correction by construction; it is a bound, not an estimate. Even at that bound the
segmentation still introduces a substantial, independently-corroborated roughness field.

### Parameter rasters on the 5 m grid

| | value |
|---|---|
| Manning's *n*, solver scalar today | 0.0400 uniform |
| Manning's *n*, segmentation-derived mean | **0.0659** (+64.8 %) |
| p05 / p50 / p95 | 0.0142 / 0.0412 / 0.1200 |
| range | 0.0130 – 0.1200 (9.2×) |
| rougher than the scalar over | 50.5 % of the domain |
| Horton/Einstein composite vs arithmetic | **+1.4 %** |
| surface storage, domain mean | 1.84 mm (max 4.50) = **0.47 % of the 392 mm Ian storm** |
| impervious fraction, domain mean | 0.244 (NLCD 30 m: 0.187) |

Two of these settle open questions rather than just reporting a number:

**The aggregation choice does not matter.** The arithmetic and Horton/Einstein composites differ
by 1.4 %. That was a live methodological worry — the two treat a mixed cell as parallel flow
paths versus one shared cross-section — and it turns out to be far smaller than the parameter
change itself, so the shipped arithmetic mean needs no defending.

**Surface storage is too small to be worth wiring in.** At 1.84 mm against a 392 mm storm it is
0.47 % of the water budget. Adding it would need a genuinely new solver term rather than a
substitution at an existing interface, and that is not justified at this magnitude. The raster is
produced so the number is checkable and the term is available if a shorter, lower-intensity event
is ever run — where an initial abstraction of a few millimetres would matter far more than it
does here.

**23.3 % of solver cells have no NAIP classification** and fall back to the scalar 0.040. NAIP's
footprint (6015 × 5984 m in EPSG:26917) is smaller than the DEM domain (6861 × 6817 m in
EPSG:5070), and the two are rotated ~8.8° relative to each other. The uncovered cells are the
domain margin, not the interior.

### Against the gauge — the headline result

> ⚠️ **The gauge numbers below were produced on the solver as it stood at 2026-08-28 12:51 and
> are superseded.** `simulation/flood_sim_ian.py` was rewritten at 14:35 on the
> channel-connectivity track — the single adaptive-dt block was replaced by an inner sub-stepping
> loop (`SUBSTEP_CAP = 20000`). Measured effect, with the ORIGINAL scalar n = 0.040 and every
> input identical, over the first 12 simulated hours:
>
> | solver | h_max | peak flooded |
> |---|---|---|
> | 12:51 | 0.050 m | 0.0 ha |
> | 14:35 (current) | 1.070 m | 173.7 ha |
>
> That is a change in the solver, not in any parameter field, and it is large. Every A/B result
> in this file needs re-running on one fixed solver revision before it can be quoted. The
> internal comparisons remain valid *relative to each other*, because each set of arms ran inside
> a single process against a single revision — but they cannot be mixed across revisions, and the
> SAM3 arm was run after the change while the others were run before it.


Four solver runs at site3 on the real Ian event, all sharing one loaded DEM, Horton field,
soil-storage cap and hyetograph. The baseline arm reproduced **411.6 cfs / 375.1 ha / 1.393 m**
bit-for-bit on two independent invocations, and those are exactly the connectivity
investigation's published post-fix numbers — so the control is the untouched code path on the
corrected terrain, not a re-derivation of it.

| metric | baseline | **segmented *n*** | segmented *n*, no channel fix | observed |
|---|---|---|---|---|
| rising-limb error vs gauge [h] | 0.72 | **0.24** | 0.14 | — |
| runoff coefficient [%] | 8.91 | **8.20** | 6.64 | **28.9 – 31.4** |
| peak boundary outflow [cfs] | 411.6 | 358.5 | 318.3 | — |
| discharge at the gauge cell, peak [cfs] | 101.6 | 37.1 | 10.5 | **1,190** |
| peak flooded [ha] | 375.1 | 383.6 | 368.7 | — |
| peak depth [m] | 1.393 | 1.277 | 1.541 | — |
| outflow + standing water [Mm³] | 2.119 | 1.996 | 1.683 | — |

**Timing improves; magnitude does not.** The rising-limb error falls from 0.72 h to 0.24 h. The
0.72 h figure was resolved — 2.9× the gauge's own 0.25 h sampling interval — so the 0.48 h
*improvement* is a real measurement. **The residual 0.24 h is not claimable**: it sits below one
sample, exactly the resolution ceiling that already disqualified the peak-argmax metric here.
The correct statement is that a resolved timing error became an unresolvable one.

The runoff coefficient moves the wrong way, 8.91 % → 8.20 %, against an observed 28.9–31.4 %.
The mechanism was measured rather than assumed: outflow *plus* standing water falls from 2.119
to 1.996 Mm³, so the missing volume is not in transit at t = 72 h — it infiltrated. A rougher
surface holds water on pervious ground longer, and with soil-storage capacity still available it
soaks in. That is physically coherent, and it is a real cost of the change.

**This corroborates the connectivity investigation's conclusion from an independent
direction.** That investigation ruled out conveyance by measuring channel slope and velocity,
and concluded Manning's *n* is not the bottleneck. Here a physically derived roughness field
spanning 9.2× — mean +64 %, rougher over half the domain — moves runoff volume by 8 % in the
wrong direction and does not touch the ~3.5×
magnitude gap. Two different methods, same answer: **roughness is not the missing physics.** No
arm peaks inside the 72-hour window either; all three rise monotonically to t = 72 h, so
roughness is not why the gauge hydrograph fails to peak.

### The finding worth carrying forward: nadir imagery puts forest roughness in the channel

This answers the standing question for any nadir-imagery parameterisation: what to do about
tree canopies, which the sensor sees from above and the water meets from below.

Riparian canopy closes over Gee Creek. NAIP cannot see through it, and a canopy-height model
says, correctly, that there is 12 m of tree there. The classifier therefore called **58.9 % of
mapped channel cells `tree_canopy`, and the gauge cell itself 100 %** — putting a forest
roughness of 0.120 on the *channel bed*, three times the solver's scalar, in the one place
conveyance matters most.

The measured consequence was severe and localised: discharge at the gauge cell collapsed from
101.6 to **10.5 cfs**, a ~10× drop, while boundary outflow fell only 23 %. A defect that barely
registers in a domain-wide statistic can dominate the validation cell.

The fix follows the precedence rule the classification already uses everywhere else — a mapped
feature outranks a spectral inference — extended to the hydrography layer, which had been left
out. Channel cells are forced to n = 0.045 (Chow, natural minor stream) using the same 3DHP
flowlines `dem_hydro.py` burns into the DEM. Gauge-cell discharge recovers to 37.1 cfs, runoff
coefficient from 6.64 % to 8.20 %, and peak depth drops from 1.541 m to 1.277 m as the
artificial ponding behind the over-rough channel drains.

**The general lesson is not about this creek.** A canopy Manning's *n* is not wrong for a
forest; it is wrong for a channel. Chow's 0.10 for timber assumes flow *among the trunks* —
"flood stage below branches". Where water runs beneath a canopy rather than through it, the bed
sets roughness and the canopy is simply overhead. Any nadir-imagery parameterisation will make
this mistake wherever vegetation overhangs conveyance, and the only defence is to let mapped
hydrography override the imagery.

The buffer half-width is load-bearing and is stated rather than buried: the gauge cell sits
exactly **one cell outside** a 5 m buffer, so 5 m would leave the validation cell at forest
roughness. The shipped 10 m gives a 20 m channel-and-banks corridor over 0.95 % of the domain,
defensible on channel width alone for a creek draining 33 km²; both figures are in
`data/parameter_raster_summary_site3.json`.

### Vision-derived Ks and Smax vs SSURGO

The first pass here declined to let a vision model supply `Ks` and soil `Smax`, on the grounds
that SSURGO measures them. That was a judgement call; it has been replaced by an experiment.
Both routes are built and both were run against the gauge with the segmented Manning field held
constant, so the only variable is where the soil parameters came from.

**This matters past settling an internal disagreement. SSURGO is a United States product.** The
gap between these two arms is what decides whether "a coordinate anywhere becomes a twin" can
leave the US.

Fairness is enforced in one place: the solver already multiplies SSURGO Ksat by an AMC-III factor
of 0.07 for Ian's saturated antecedent conditions, and the same factor is applied to the vision
Ks. Both are dry saturated conductivities; correcting only one would pit a wet soil against a dry
one and let the vision arm infiltrate ~14x too much.

**The two fields disagree profoundly, cell by cell:**

| | SSURGO | vision | |
|---|---|---|---|
| fc_eff mean | 17.16 mm/hr | 5.86 mm/hr | 0.34x |
| soil storage mean | 206 mm | 148 mm | 0.72x |
| zero-storage cells | 26.0 % | 40.9 % | |
| spatial correlation, conductivity | | | *r* = **+0.171** |
| spatial correlation, storage | | | *r* = **−0.234** |

**And yet they produce nearly identical basin runoff:**

| metric | SSURGO | vision | observed |
|---|---|---|---|
| runoff coefficient | 8.202 % | **8.249 %** | 28.9–31.4 % |
| outflow volume | 1.502 Mm³ | 1.511 Mm³ | — |
| infiltrated | 16.314 Mm³ | 16.497 Mm³ | — |
| rising-limb error | **0.24 h** | 0.54 h | — |
| peak flooded | 383.6 ha | 266.5 ha | — |
| **discharge at the gauge cell** | **37.1 cfs** | **3.0 cfs** | 1,190 cfs |

**A tie on the water budget, 12x apart at the validation point.** Runoff coefficient differs by
0.6 % between two parameterisations that share almost no spatial structure — because at 392 mm
the storm overwhelms both, and total runoff is set by what the profile cannot hold either way.
The limiting mechanism is not even the same: SSURGO is storage-limited (it could rate-limit
1,235 mm over 72 h, far past the storm), while the vision route is close to being limited by
both (422 mm of rate capacity against a 392 mm storm).

The 12x gap at the gauge cell is the finding. The vision route's error is **invisible in the
lumped water budget and dominant in routed discharge** — the same shape of failure as the
channel-roughness defect above, in a different parameter.

**Why, mechanistically.** Binning SSURGO storage by canopy fraction shows the two routes
disagree hardest exactly where the canopy is:

| tree canopy in cell | SSURGO storage | vision storage |
|---|---|---|
| 0–20 % | 234 mm | 3 mm |
| 50–80 % | 224 mm | 244 mm |
| **80–100 %** | **140 mm** | **369 mm** |

Where canopy is densest, SSURGO reports the *least* storage. In Florida, deep-rooted forest sits
on hydric ground with a shallow water table — cypress domes and hydric hammocks. **The vision
route infers a deep profile from deep roots; the causality runs the other way.** And because
Florida's forest is concentrated in the riparian corridor, the vision route soaks up water
precisely in the strip that feeds the creek, then sheds it in the outer suburban areas that drain
to the domain boundary instead. Same total volume, completely different routing — which is
exactly what a 12x gauge-cell gap alongside a 0.6 % budget gap looks like.

**The verdict, for use outside the US.** A vision estimate of soil parameters is adequate for a
lumped water balance and materially wrong for routed discharge in low-relief wetland terrain,
for a reason that is specific and understandable rather than random. Where SSURGO exists, use it.
Where it does not, expect the volume to be usable and the timing and routing to degrade —
rising-limb error went from 0.24 h (below the gauge's own 0.25 h resolution) to 0.54 h (2.2x it,
and therefore resolved). Flood extent differs by 30 % between the two and site3 has no extent
ground truth, so that difference cannot currently be adjudicated at all.

### The 0.6 m impervious substitution changes nothing

Replacing NLCD's 30 m impervious fraction with the 0.6 m segmentation one moved the runoff
coefficient by −0.05 pp before the channel fix (6.636 → 6.586 %) and by **−0.004 pp after it**
(8.202 → 8.198 %), with peak outflow unchanged at 358 cfs. Tested in both configurations
precisely because a null result in one is not evidence of a null result in the other. The reason
is visible in
the inputs: over cells not already forced hard by the OSM mask, NLCD reports 17.1 % impervious
against segmentation's 8.8 %. NLCD's 30 m pixels smear road and roof imperviousness into
neighbouring cells that the binary OSM mask has *already* accounted for, so the coarse layer
double-counts. The finer layer is more nearly correct and immaterial to the result — worth
knowing, and worth not claiming as an improvement.

## Outputs

| file | grid | contents |
|---|---|---|
| `data/chm_2m_site3.tif` | 2 m, EPSG:5070 | canopy height above bare earth [m] |
| `data/canopy_cover_2m_site3.tif` | 2 m | fraction of returns above 2 m |
| `data/return_density_2m_site3.tif` | 2 m | returns per m² (also the coverage mask) |
| `data/landcover_0.6m_site3.tif` | 0.6 m, EPSG:26917 | surface class code |
| `data/segments_site3.csv` | — | per-segment features + class |
| `data/surface_parameters.json` | — | the per-class parameter table with per-value provenance |
| `data/manning_n_5m_site3.tif` | 5 m, EPSG:5070 | **the solver input** |
| `data/surface_storage_5m_site3.tif` | 5 m | interception + depression storage [m] |
| `data/impervious_frac_5m_site3.tif` | 5 m | impervious fraction |
| `data/qc_preview_site3.png` | — | NAIP / class / CHM crops, side by side |
| `data/canopy_validation_site3.json` | — | canopy vs NLCD TCC, with the roughness bound |
| `data/nlcd_tcc_2021_site3.tif` | 30 m | the NLCD Tree Canopy Cover reference, as fetched |
| `data/ab_summary_site3.json` | — | the three-arm gauge comparison |

## Solver change

One function signature in `../simulation/flood_sim_ian.py`: `run_sim(..., manning_n=None)`.
The array is averaged onto cell faces internally, because the friction term acts on flux across
a boundary and needs the roughness the water actually crosses. `manning_n=None` takes the
original scalar expression unchanged, so every existing caller and every recorded run is
unaffected. Nothing else in the solver was touched.
