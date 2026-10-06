# Scientific provenance

Every step taken to build US native and adapted range maps, first for a national horticultural priority list and
later for all native species of the contiguous United States, set against what Daru (2024) reports. Daru's text is
taken from the PNAS SI Appendix, which is identical word for word to the Dryad methods. Status: **faithful** (same as
Daru), **changed** (deliberate departure, with reason), **pending**. Analyses of non-public nursery-trade data made
alongside this work are not part of this model and are not recorded here.

Primary reference: Daru, B. H. (2024). Predicting undetected native vascular plant diversity at a global scale.
*PNAS* 121(34): e2319989121. doi:10.1073/pnas.2319989121 (open full text: PMC11348117). Data: Dryad
doi:10.5061/dryad.5x69p8d9w (CC0). Code: Zenodo 10070000 (figure scripts + phyloregion 1.0.9 only; the
range-map pipeline itself was not released).

## Where the code named in this log is published

The log names code as it was called when each entry was written. In this package:

| Named in the log | Published as |
|---|---|
| `deepearth_ranges/...` | `ranges/...` |
| `native_ranges/train_joint.py` (research code) | `ranges/joint/train.py`, `scripts/national_train.py` |
| `native_ranges/national_cards.py` (build, `recode`, `update-zero-shot`, `Store`) | `ranges/joint/store.py`, `scripts/national_store.py` |
| `native_ranges/zero_shot.py` | `ranges/joint/zero_shot.py` |
| `native_ranges/make_conus.py`, `national_cards.conus_calibration` | `ranges/joint/scope.py`, `scripts/national_scope.py` |
| `national_data/prepare_national.py`, `prepare_national_plots.py` | `ranges/joint/prepare.py`, `scripts/national_prepare.py`, `scripts/national_plots.py` |
| `national_data/build_tree_all.py`, `native_ranges/build_tree.py` | `scripts/build_tree.py` |
| `card_fidelity.py`, `eval_national_maps.py`, `store_bench.py` | `scripts/national_store_eval.py fidelity`, `plots`, `bench` |
| `native_ranges/tests_store.py` | `tests/test_joint_store.py` |
| `scripts/cloud/r_env_explicit.txt` | `r_environment.txt` |
| `train_joint.py` `HarmonicField`, `ring_harmonics/` (`--field --field-kind harmonic`) | `ranges/joint/field.py`, `ranges/joint/kernels/ring_harmonics.cu` |
| `train_joint.py` `SinrPlace`, `set_place_head` (`--place-head --place-kind sinr`) | `ranges/joint/place.py`, `ranges/joint/model.py` |
| `train_joint.py` `--calib-soft`, `--tg-bg`, `--global-bg`, `--shore-records`, `--fill-climate` | `TrainConfig` `calibration_penalty`, `target_group`, `continental_background`, `shoreline_records`, `fill_plots` (`ranges/joint/train.py`, `data.py`) |
| `train_joint.py` `--init`, `--shared-from --freeze-shared --frozen-cache`, `cache_shared.py`, `joint_model.py` | `ranges/joint/train.py` (`init`, `shared_from`, `freeze_shared`), `ranges/joint/cache.py`, `JointRangeModel.load`; `scripts/national_train.py --stage`, `scripts/national_cache.py` |
| `train_joint.py` `auc_table_gpu` | `ranges/joint/train.py` `auc_table` |
| `build_field.py` | `ranges/joint/field.py` `build_pyramid`, `scripts/national_field.py` |
| `geodesy.py` | `ranges/joint/geodesy.py` |
| `climate_fill.py`, `build_shore_records.py` | `ranges/joint/climate_fill.py`, `ranges/joint/shoreline.py`, `scripts/build_shoreline.py`, `scripts/build_climate_fill.py` |
| `community_knn.py --snap` | `ranges/joint/prepare.py` `snap_to_records`, `scripts/national_snap.py` |
| `national_cards.py` (place features, penalty), `eval_national_maps.py` (`auc_joint_shipped`) | `ranges/joint/store.py`, `reader.py` (`Store.served`); `national_store_eval.py plots` (`auc_joint_served`) |
| `tests_ring_harmonics.py`, `tests_ring_tokens.py`, `tests_geodesy.py`, `tests_climate_fill.py`, `tests_auc_gpu.py` | `tests/test_field.py`, `tests/test_geodesy.py`, `tests/test_climate_fill.py`, `tests/test_joint_stages.py` |

Research scripts named in the log that are not listed here (ablations, panels, studies) are not part of the published
package; their results are recorded here as measured.

## Inconsistencies inside Daru (2024) that we must resolve explicitly
| # | Issue | Where | Our resolution |
|---|---|---|---|
| D1 | Binary threshold stated two ways: "equal training sensitivity and specificity" vs "95% quantile of suitability at occurrences" | SI methods vs SI methods / Dataset S1 | Compute and store both; the 95%-quantile wording is read as the 5th-percentile training presence (P5) threshold, since a literal 95th percentile would omit 95% of presences. |
| D2 | Prediction resolution ~9 km (text) vs ~18 km (ODMAP) | SI vs Dataset S1 | **Resolved 2026-10-03:** the Dryad per-species rasters are 0.1667° (~18 km, EPSG:4326). Training predictors may still have been ~9 km. |
| D3 | Thinning applies at ≥5 localities (text) vs N > 4000 (Fig. S1); Fig. S1 adds a 30 km buffer for N ≤ 5; phyloregion `sdm()` tops a species up to `size` = 50 points sampled in a buffer around its records | SI vs Fig. S1 vs code | **Applied 2026-10-04** for species with 1–4 cleaned (thinned) records: points drawn uniformly on land within 30 km of the records (Fig. S1 width) until 50 presences exist (code's `size`), seeded, flagged `source = buffer`, used throughout as in Daru's code (`pipeline._buffer_points`; `n_buffer_points` in summary.json). Species with no cleaned record remain unfitted. |
| D4 | Species count 202,284 (Fig. S1) vs 201,681 | Fig. S1 vs text | Not relevant to us. |
| D5 | ODMAP: validation by 5-fold spatial-block CV; text: random 75/25 + 5-fold CV | Dataset S1 vs SI | Text followed (5-fold CV for β, 75/25 replicates); spatial-block CV added as our validation. |
| D6 | "species-specific dispersal rates" from `fit_sbm_const`, which fits ONE diffusivity per tree | SI step 4 | Pending: fit per clade (see step 4). |
| D8 | Text: buffered hull *intersected with* the ecoregions occupied by the species. Published rasters: footprint = whole occupied ecoregions | SI step 4 vs Dryad rasters | **Resolved 2026-10-04 by measurement** (Echinacea purpurea, Daru's non-NaN footprint vs candidates on his grid): geometric intersection IoU 0.473; whole ecoregions intersecting the hull 0.815; **whole ecoregions containing records 0.829** (98% of it inside Daru's footprint); adding the buffered hull (0–50 km) 0.813–0.818. Implemented: whole occupied ecoregions. |
| D9 | Text: background = 10,000 points weighted by the sampling-bias KDE. Code (phyloregion 1.0.9 `sdm` → `predicts::MaxEnt(p, ox)` with no background argument): MaxEnt draws its own uniform background; the bias-weighted points serve only for evaluation (and the GLM) | SI step 5 vs phyloregion code | **Resolved 2026-10-04 by measurement** against Daru's published suitability (Spearman at his ~18 km cells, ~9 km predictors): bias-weighted 0.752 (Larrea) / 0.790 (Echinacea); uniform over the calibration area 0.867 / 0.859; uniform over its bounding box 0.859 / 0.874. Implemented mode `daru` = hull presences + uniform background over the calibration area, with a bias-weighted evaluation background for his metrics. |
| D10 | Which background Daru's reported metrics (median AUC 0.91, TSS 0.42, Boyce 0.87) were scored against. phyloregion `sdm` scores test presences against the bias-weighted points passed as `background` | SI Fig. S2 vs phyloregion code | **Measured 2026-10-04**, mode `daru`, all 114 Dryad overlap species (medians): against the fitting background (uniform) AUC 0.756, TSS-median 0.276, Boyce 0.734; against the bias-weighted set AUC 0.678, TSS-median 0.193, Boyce 0.514. Both are reported (`evaluate.daru_metrics`, suffix `_eval_bg`). Neither reaches his global medians; these are taken over ~200,000 mostly narrow-ranged species, while our set is widespread US natives with large calibration areas, so the numbers are not comparable species for species (Daru published no per-species metrics). An earlier note that the fitting background reproduces his numbers rested on one species (Actaea rubra, 0.908) and is withdrawn. |
| D11 | Order of CoordinateCleaner and the native filter: Fig. S1 cleans all records, then keeps native ones. CoordinateCleaner's outlier test is relative to the species' own records | SI Fig. S1 | **Measured 2026-10-04** (`scripts/cleaning_order_study.py`, 60 random natives + 2 cases): the orders differ by < 4% of native records for 58 of 60 species (median 0%), but a species recorded mostly where it is planted loses its whole native cluster as "outliers" (Abies procera, 79% of records in Europe: 0 vs 1,274 native records; Yucca gloriosa: 0 vs 180). Daru's order is kept; cleaning only the native records is used when Daru's order keeps under half as many (recorded as `cleaning_order` in summary.json). Abies procera then fits with CV AUC 0.890. |
| D7 | MaxEnt "v3.3.4" (ODMAP) — no such release (3.3.3k, 3.4.x exist) | Dataset S1 | maxent.jar 3.4.4 (open source, MIT). |

## Pipeline, step by step
| Step | Daru (2024) | Ours | Status |
|---|---|---|---|
| 0. Taxa | 201,681 global species | A national horticultural priority list of 6,140 taxa (5,654 species + 486 genus-level hybrids); extended on 2026-10-04 to all 18,600 species native to the United States per WCVP, and restricted on 2026-10-05 to the 16,947 species native to the contiguous US (see log) | changed (scope) |
| 1. Occurrences | GBIF full export, TaxonKey = Tracheophyta, 15 May 2023, 402M records (doi:10.15468/dl.vgvc3z) | GBIF download 0009841-260928105237408, **doi:10.15468/dl.wwa829** (created 2026-10-04 07:10 UTC; SIMPLE_PARQUET; predicate TAXON_KEY in 5,598 GBIF species keys matched to the priority-list taxa, HAS_COORDINATE = true, OCCURRENCE_STATUS = PRESENT, worldwide; 173,549,197 records, 11.8 GB; licence CC BY-NC 4.0 as the most restrictive of the records' licences) | done |
| 2a. Names | Match to WCVP; keep WCVP names; 4 family renames to APG | WCVP (Kew, files dated 2026-06-04); status-ranked resolution, Misapplied never followed, nothospecies "×" retried (`names.py`); records selected under the GBIF name, native areas under the WCVP accepted name (they differ for 88 of 2,672 natives) | faithful (done); record selection extended 2026-10-04 (see log) |
| 2b. Cleaning | CoordinateCleaner 3.0.1 `clean_coordinates` (duplicates, outside ±180, sea, country/province centroids, 100 m institution radius) | Same package/version in R 4.3 (conda env `daru`) | faithful (pending run) |
| 2c. Native filter | Keep points intersecting WCVP native TDWG L3 areas | Same, with WGSRPD L3 polygons | pending |
| 2d. Thinning | dismo `gridSample`, resolution unstated | gridSample at the predictor grid (~1 km) | changed (resolution) |
| 3. Range hull | rangeBuilder 2.1 alpha hull, initialAlpha = 2, grown until ≥99% of points enclosed; crop to land (Natural Earth); clip to family range (Heywood 1993 / APG IV) | rangeBuilder 2.2, same settings; land = Natural Earth 10m; family range = WCVP native L3 areas of the family, morphologically closed (±0.02°) to remove border slivers (Heywood polygons are not distributed) | changed (family source) |
| 4. Dispersal + calibration | castor `fit_sbm_const` (SBM) on V.PhyloMaker2 GBOTB.extended.TPL tree, 2 random trees (scenario S2); tip = hull centroid; `expected_SBM_distance` → km/yr; buffer hull; ∩ WWF terrestrial ecoregions (Olson 2001) occupied | SBM per family on 2 Carruthers dated trees (tips located at WCVP native centroids; branch floor 0.1 Myr; castor 1.8.4 `expected_distances_sbm`). **Calibration area = whole ecoregions containing the species' cleaned records** (what Daru's rasters show, D8); RESOLVE Ecoregions 2017 (Olson 2001 download blocked); dispersal rates recorded, map-neutral | changed (tree, ecoregion source); faithful to published rasters |
| 5. Background | spatialEco KDE of sampling → bias grid 0–1; 10,000 points per species sampled ∝ bias inside calibration area (text) — his code fits a uniform background (D9) | mode `daru`: uniform background (as published), bias-weighted evaluation set; `sp.kde` ported exactly (Silverman bandwidth over 524.6M Tracheophyta records from GBIF map tiles, Behrmann 10 km); 10,000 points ∝ bias over ~1 km calibration cells, lakes excluded (`phyloregion::backg` logic) | faithful |
| 6a. Training points | 500 points by regular sampling of the hull rasterized at predictor resolution | Same (`presence_mode="hull"`); improved variant `presence_mode="occurrences"` (≤ 5,000 thinned records) run alongside (ledger L8) | faithful + variant |
| 6b. Predictors | WorldClim 2.1 bio1–19 + elevation, ~9 km; VIF stepwise, threshold 5 (usdm vifstep); lakes masked | WorldClim 2.1 bio1–19 + elevation at **~1 km** (downloaded 2026-10-03, geodata.ucdavis.edu); usdm 2.1.7 vifstep | changed (resolution) |
| 6c. Model | MaxEnt; linear/threshold/hinge; β ∈ {2, 5, 10, 15, 20} under 5-fold CV; 75/25 split; 5 replicate models, median | maxent.jar 3.4.4; same features, β grid, 5-fold CV selection by mean test AUC; 5 replicates at 25% random test; median; fixed seed (reproducible) | faithful |
| 7. Projection | Raster at ~18 km (see D2) | GPU (PyTorch) evaluation of the MaxEnt closed form from each `.lambdas` file. **Verified 2026-10-03:** matches maxent.jar to 4e-15 relative error (float64) / 6e-7 (float32); 263 M cells/s on one RTX 3090 (`tests/test_maxent_parity.py`). Output grid: CONUS Albers EPSG:5070, 240 m, origin aligned to the NLCD/LANDFIRE 30 m grid (x0 = −2,493,045, y1 = 3,310,005; 20,149 × 13,053 cells); WorldClim predictors warped bilinearly with gdalwarp (`scripts/warp_conus_predictors.sh`, archived 2026-10-04) | changed (engine, grid; same maths) |
| 8. Threshold + polygon | D1; terra polygons; smoothr spline smoothing. **Observed in Dryad rasters:** the binary map is not a threshold of the published suitability raster (Echinacea purpurea: presence cells reach down to 0.446 while absence cells reach 0.577), consistent with phyloregion `sdm()`: each replicate thresholded, then majority vote (median of binaries > 0.5); both rasters NaN outside the calibration area | Same construction (per-replicate threshold, majority vote) plus the median-suitability threshold, both stored | faithful (pending run) |
| 9. Evaluation | AUC, TSS, Boyce; IUCN + southern-Africa V-measure; range filling | Same metrics + direct comparison with Daru's Dryad rasters + independent plots (FIA, NEON, AIM) | pending |

## Data sources (all accessed by this project)
| Source | Version / date | Licence | Use |
|---|---|---|---|
| Daru (2024) Dryad DRYAD_DATA.zip (15.18 GB; SDM_set1 = 10,000 species rasters; plant_presab_wag4.csv, 20 km) | Dryad, published 2024-09-23 | CC0 | benchmark |
| phyloregion 1.0.9 (Zenodo 10070000) | 2024 | AGPL-3 | diversity metrics; reference `sdm()` |
| WCVP (sftp.kew.org/pub/data-repositories/WCVP/wcvp.zip) | files dated 2026-06-04, downloaded 2026-10-03 | CC BY | names, native/introduced L3 |
| WorldClim 2.1, ~1 km bio + elev | downloaded 2026-10-03 | CC BY-SA 4.0 (non-commercial) | predictors |
| maxent.jar 3.4.4 (github.com/mrmaxent/Maxent) | 2020-11-23 | MIT | model |
| Carruthers et al. dated trees (OSF 9tbha) | local copy | see OSF | phylogeny |
| GBIF occurrence download doi:10.15468/dl.wwa829 (0009841-260928105237408) | 2026-10-04 | CC BY-NC 4.0 (aggregate; per-record CC0/CC BY/CC BY-NC) | occurrences for all priority-list taxa (173.5M records) |
| GBIF occurrence download doi:10.15468/dl.pkunks (0010385-260928105237408) | 2026-10-04 | CC BY-NC 4.0 (aggregate; per-record CC0/CC BY/CC BY-NC) | supplementary occurrences for US native species outside the first download (16,276 taxon keys; 50.5M records; 4.1 GB zip, 5.0 GB parquet) |
| GBIF occurrence download doi:10.15468/dl.52fhnr (0010622-260928105237408) | 2026-10-05 02:22 UTC | CC BY-NC 4.0 (aggregate) | supplementary occurrences under the GBIF keys of WCVP synonyms of US native species with < 20 records in the first two downloads (1,866 taxon keys, 848 species; 1,241,857 records; 127 MB zip; raw/gbif_us_natives_synonyms) |
| GBIF API (census of counts) | 2026-10-03 | CC0/CC BY/CC BY-NC per record | feasibility |
| Natural Earth 50 m land (naciscdn.org) | downloaded 2026-10-04 | public domain | CoordinateCleaner sea-test reference (`CC_SEAS_REF`) |
| VegBank (ESA plot archive, api.vegbank.org): 142,467 plot observations, 2,202,673 taxon observations | downloaded 2026-10-04 | per project (public plots) | independent presence/absence, incl. eastern US (`scripts/fetch_vegbank.py`) |
| iNaturalist API (captive vs wild counts, 200-species sample) | 2026-10-03 | per record | feasibility |

## Log
- 2026-10-03: Read Daru (2024) main text, SI Appendix, Datasets S1–S2, Dryad methods + README, Zenodo code and all of phyloregion 1.0.9 (2,836 lines). Found D1–D7 and Fig. 5 vs Table S2 disagreement, duplicated PD/PE rows, and that extrapolated hotspots are thresholded at the modelled map's 90th percentile (`hotspots(x, y = ref)`). phyloregion bugs: `PD_ses` z = (obs − mean)/sqrt(sd); `sdm(algorithm = "all")` returns undefined `raw_model`; β chosen by single-fold AUC.
- 2026-10-03: Benchmark — one model set (β grid × 5-fold CV + 5 replicates), *Cercis canadensis*, 499 presences, 10,000 background, 20 predictors: 219 s on one thread; mean test AUC 0.858 (β=2) … 0.825 (β=20). Not a full reproduction (no cleaning, hull, calibration area, bias, VIF, projection).
- 2026-10-03: Daru per-species rasters exist for 215 of 5,654 priority species; the 20 km table covers 4,123.
- 2026-10-03: maxent.jar adds presences to the background by default (`numBackgroundPoints` = 10,374 for 10,000 background + presences) — kept (Daru gives no setting, so defaults apply).
- 2026-10-03: castor 1.8.4 has no `expected_SBM_distance` (Daru's name); the current equivalent is `expected_distances_sbm(diffusivity, radius, deltas)`.
- 2026-10-03: The WWF/Olson (2001) shapefile download is behind a Cloudflare challenge; RESOLVE Ecoregions 2017 (Dinerstein et al. 2017, CC BY 4.0; storage.googleapis.com/teow2016/Ecoregions2017.zip), the successor that keeps Olson's framework, is used instead — **changed** (step 4). TDWG WGSRPD level-3 polygons from github.com/tdwg/wgsrpd (369 areas); Natural Earth 10m land and lakes.
- 2026-10-03: Dispersal step: tip locations for all 128,271 Carruthers-tree tips = area-weighted spherical centroid of WCVP native L3 areas (110,650 tips located, 420 families); SBM fitted per family (>= 10 located tips) and for the whole tree, on dated trees 0001 and 0002 (Daru: 2 random trees). Rationale: `fit_sbm_const` returns one diffusivity per tree, so "species-specific" rates (D6) require clade-level fits.
- 2026-10-03: GBIF pilot download 0009824-260928105237408 (Echinacea purpurea 3150935, Larrea tridentata 7568403; HAS_COORDINATE, OCCURRENCE_STATUS = PRESENT; no other filters, as in Daru's full export). GBIF SQL download 0009825-260928105237408: counts of all Tracheophyta (phylumKey 7707728) records with coordinates and no geospatial issues per ~9 km cell — the sampling-effort input to the KDE bias grid (step 5).
- 2026-10-04: SBM per-family fits (Carruthers dated tree 0001, castor 1.8.4, ~7 s per family): Fagaceae (432 tips) D = 1.63e5 km²/Myr → 0.72 km/yr (22.6 km/kyr); Asteraceae (10,280 tips) D = 4.30e6 km²/Myr → 3.67 km/yr (116 km/kyr). **Consequence:** Daru's one-year dispersal buffer is ~1–4 km, below the 18 km (~18 km) cell of his published rasters, so in his maps the calibration area is effectively hull ∩ occupied ecoregions; on our 240 m grid the same buffer spans 3–15 cells and is retained as specified.
- 2026-10-04: Validation data: BLM AIM Terrestrial Species Indicators (public ArcGIS FeatureServer layer 6; 1,781,532 plot-visit × species records with point coordinates) downloading to raw/validation/blm_aim_species.csv — full species lists per plot provide true absences.
- 2026-10-04: **Sampling-effort input (step 5).** GBIF SQL download 0009825 returned zero rows (GBIF executed the query without error; column semantics of the 2026 SQL interface differ). Replaced by the GBIF v2 map API density tiles (EPSG:4326, zoom 3, square bins of 2 tile units, taxonKey 7707728 Tracheophyta), aggregated to ~9 km cells: 524,624,167 records in 984,367 cells (`scripts/fetch_gbif_effort.py`, 2026-10-04). Check: SF Bay 1°×1° box = 857,439 vs 892,276 from the occurrence search API (ratio 0.961; the map service counts georeferenced records without geospatial issues).
- 2026-10-04: **Bias grid.** `sp.kde` ported exactly (Gaussian product kernel, Silverman bandwidth over all records, min–max standardized) and evaluated as an FFT convolution of the ~9 km count grid on a 10 km Behrmann (ESRI:54017) grid: bandwidth 20.5 km (x) × 13.2 km (y); the standardized surface is extremely skewed (median non-zero 0.000, 99th percentile 0.012).
- 2026-10-04: **GBIF keys.** All 6,140 entries matched with the species-match API (`scripts/match_gbif_keys.py`): 6,089 exact, 40 higher-rank, 11 fuzzy. 33 species entries resolve only to genus/family (e.g. *Myrica pensylvanica* → accepted *Morella pensylvanica*) — to be re-matched through WCVP synonyms.
- 2026-10-04: **Full occurrence download** 0009841-260928105237408 (SIMPLE_PARQUET): TAXON_KEY in 5,598 accepted species keys, HAS_COORDINATE, OCCURRENCE_STATUS = PRESENT, global. Genus-level hybrid entries deliberately excluded (a genus key pulls every species of the genus); their genus-only records are a separate query for the adapted product.
- 2026-10-04: Predictor stacks built: global ~1 km pixel-interleaved float32 memmap (21,600 × 43,200 × 20; 74.6 GB) and the 20 CONUS 240 m layers (EPSG:5070, bilinear).
- 2026-10-04: **Defect found and fixed (step 3, family-range clip).** The union of WGSRPD L3 polygons leaves hairline slivers along shared state borders (neighbouring polygons do not share vertices exactly); intersecting the hull with it cut thin lines through the hull, invisible at Daru's ~18 km but visible at 240 m. Fix: morphological closing of the family range (buffer +0.02°, then −0.02°). Pilot inputs at that stage: Echinacea purpurea 7,899 cleaned native thinned records; VIF kept 8 of 20 predictors (bio2, bio7, bio8, bio9, bio13, bio15, bio18, elev); 500 hull presences; 9,987 background points spread over 1,135 half-degree cells (top 1% of cells hold 16%).
- 2026-10-04: **Independent check vs WCVP** (`benchmark.wcvp_consistency`): share of the predicted range inside the species' WCVP native L3 areas. Daru's Dryad maps: Echinacea purpurea 0.870 (covering 0.699 of the native L3 area), Larrea tridentata 0.950 (covering 0.483).
- 2026-10-04: MaxEnt runs use `randomseed=false` (maxent.jar's fixed seed): cross-validation folds and the 25% random test splits differ between replicates but are identical across reruns, so every published map can be regenerated exactly. The five β values are cross-validated concurrently (one JVM each); results are unchanged by the parallelism.
- 2026-10-04: **Defect found and fixed (step 2a, names).** The first WCVP match kept the first row per binomial; for some names that row is *misapplied* (the name wrongly used for another species), whose accepted id points elsewhere (e.g. *Acer palmatum* → *A. macrophyllum*, *Platanus occidentalis* → *P. racemosa*), corrupting native status. `ranges/names.py` now ranks Accepted > Synonym > Orthographic > Illegitimate/Invalid > Unplaced/Artificial Hybrid and never follows Misapplied rows; nothospecies are retried with "×". Result (5,654 species): 5,311 Accepted, 177 Synonym, 96 Artificial Hybrid, 6 Illegitimate/Invalid, 64 unmatched; native-to-CONUS status changed for 106 species. 2,672 species are native to CONUS with a GBIF species key; 88 of them carry a different name in GBIF than in WCVP, so records are selected by GBIF name and native areas by WCVP name. (Genuine updates such as *Picea glauca* → *Picea laxa* follow WCVP.)
- 2026-10-04: **Pilot results, Larrea tridentata** (whole-ecoregion calibration; 98,876 GBIF → 92,262 CoordinateCleaner-valid → 92,113 native → 24,390 thinned; 8 predictors; β = 2, CV AUC 0.848; fit 172 s, CONUS 240 m render 67 s). vs Daru's raster (his ~18 km grid, CONUS): Jaccard 0.654, κ 0.615, Spearman (suitability) 0.755. Inside WCVP native L3: 0.996 (Daru 0.950). BLM AIM plots (71,422; 5,234 presences): AUC 0.914 (Daru 0.939); binary TSS 0.52 with the replicate-majority rule vs 0.675 with the 5th-percentile rule (Daru 0.72).
- 2026-10-04: **Diagnosis.** Aggregating our suitability to Daru's ~18 km grid leaves the AIM AUC unchanged (0.807 vs 0.807 on 34,835 plots; Daru 0.873), so the gap is in the model, not the resolution. Ablation (single fit, β = 2): predictors averaged to ~9 km (Daru's training resolution) 0.897 vs 0.879 at ~1 km — a small part of the gap; training on thinned occurrences instead of 500 hull points 0.986 (future ledger L8). Leakage check: 3.1% of AIM presence plots lie within 100 m of a training record (18.3% within 1 km); Larrea's GBIF records are 92% iNaturalist (dataset 50c9509d), not AIM.
- 2026-10-04: **Rendering uses a shared 240 m ecoregion-id layer** (RESOLVE ECO_ID rasterized by cell centre onto the CONUS grid; Natural Earth 10m lakes set to 0, as Daru masked lake pixels; 89 ecoregions, 5,876,889 lake cells). The calibration area of a species = cells whose id is among the ecoregions containing its cleaned records. Pilot re-renders: 59 s each for all of CONUS.
- 2026-10-04: **Range cards (storage).** A species' three 240 m maps are stored as its five `.lambdas` files, thresholds and ecoregion ids (~5 kB, zstd), decoded on GPU against the shared layers; byte-identical to the rendered GeoTIFFs for both pilot species (`codec.verify`).
- 2026-10-04: **Pilot results after the calibration correction, Echinacea purpurea:** 16,641 GBIF → 15,490 valid → 11,631 native → 7,899 thinned; 7 predictors; β = 2, CV AUC 0.777; vs Daru κ 0.549 (from 0.085 with the hull-intersection reading), Spearman 0.759; inside WCVP native L3 0.938 (Daru 0.870). Our map excludes New England and peninsular Florida, which Daru's includes.
- 2026-10-04: Validation data. NEON DP1.10058.001 (plant presence, complete subplot lists) — API returns HTTP 403 to anonymous requests; needs a NEON API token (pending). FIA DataMart national CSVs (ENTIRE_PLOT 473 MB, ENTIRE_TREE 13.7 GB) streamed to live-tree species presence per sampled forested plot (`scripts/fetch_fia.py`); public FIA coordinates are perturbed by up to ~1.6 km, so FIA tests range placement, not 240 m detail.
- 2026-10-04: **SBM numerical instability.** With near-zero branch lengths floored at 1e-6 Myr, 4 of 272 family fits diverged (D 1e15–1e16 km²/Myr; e.g. Juncaceae, Dichapetalaceae, and Poaceae on one of the two trees): sister tips with no divergence time but distant native centroids imply unbounded speed. Branch lengths are now floored at 0.1 Myr (dating resolution); Juncaceae 10,008 → 6.2 km/yr, Elatinaceae 9.8 → 7.8, Poaceae 5.7 → 4.0, Asteraceae 3.7 → 2.3. Fits still > 100 km/yr (e.g. Dichapetalaceae, 14 tips) fall back to the median family rate. Because calibration areas are whole occupied ecoregions (D8), dispersal rates do not change any map; they are kept for provenance.
- 2026-10-04: **Full-pipeline test of the occurrence-trained variant** (presence_mode = "occurrences": 5,000 of the 24,390 cleaned, native, ~1 km-thinned Larrea records; everything else identical, incl. 5-fold CV over β and 5 replicates). BLM AIM, 71,422 plots: AUC 0.986, TSS 0.840 (sensitivity 0.868, specificity 0.972) vs Daru's published map 0.939 / 0.723 (0.844 / 0.878) and the Daru-faithful 240 m map 0.914 / 0.521; within both calibration areas AUC 0.967 vs 0.872. Predicted range covers 0.178 of the WCVP native L3 area (Daru 0.483), 0.995 of it inside native L3. Agreement with Daru's map falls (Jaccard 0.300, κ 0.282) because the variant is tighter where the plots indicate Daru over-predicts.
- 2026-10-04: **Rendering speed-up, outputs unchanged.** The 20 CONUS layers are unpacked once into an uncompressed band-sequential float32 memmap (21.0 GB, values bit-identical to the GeoTIFFs); a species is rendered only over the bounding window of its calibration ecoregions, with masking, band stacking and gathering on the GPU in the same row chunks. One code path (`project.compute_maps`) serves rendering and range-card decoding. Echinacea purpurea: 101 s → 7.3 s; Larrea tridentata: 67 s → 4.8 s; all three maps identical to the original renderer cell for cell.
- 2026-10-04: **Scale-out.** CPU stage (cleaning → MaxEnt replicates) for all 2,672 native-to-CONUS priority species in both presence modes (5,344 tasks), in an environment built entirely from public sources; rendering on GPU from the returned coefficients. MaxEnt outputs are slimmed to inputs, coefficients, results tables, sample predictions and omission tables (35 MB → 3.7 MB per species).
- 2026-10-04: **Resolution test (does 240 m add skill?).** Larrea tridentata suitability scored at BLM AIM plots inside the calibration area at native 240 m and block-averaged to 0.48–18 km. Occurrence-trained: AUC 0.967 (240 m) → 0.969 (1.9 km) → 0.970 (18 km); Daru-faithful: 0.789 → 0.793 → 0.811. Aggregation does not reduce skill: with WorldClim ~1 km climate bilinearly interpolated to 240 m (plus ~1 km elevation), the 240 m grid carries finer cells but no finer information. This resolution ablation is the standing test for any fine-scale predictor upgrade (ledger L3, L10).
- 2026-10-04: **Batch-invariant evaluation.** A float32 matrix-vector product on GPU is not bit-identical across batch sizes (381 of 1,000 cells differed when a batch was evaluated alone vs inside a larger one). `MaxentModel.linear_predictor` now forms per-feature contributions element-wise and sums along the feature axis; cell values are then identical for any batch size (`tests/test_maxent_parity.py::test_cell_values_independent_of_batch_size`), and parity with maxent.jar is unchanged (4e-15, float64).
- 2026-10-04: **Tile decoding** (`codec.decode_window`): any window of a species' three maps decodes from its range card alone; 20 random 256 × 256 tiles of Larrea tridentata decoded in 0.2 s total, each byte-identical to the full render.
- 2026-10-04: Calibration footprint check repeated on Larrea tridentata: IoU 0.848 with Daru's published footprint (0.979 of ours inside his), matching Echinacea purpurea (0.829); now computed for every benchmarked species (`benchmark.calibration_iou`).
- 2026-10-04: **Defect found and fixed (GBIF parquet input).** The first full fitting run stopped at once: GBIF's SIMPLE_PARQUET download contains zero-byte part files (`occurrence.parquet/000000`), which pyarrow rejects, and the run script then wrote a completion marker despite the failure. Fixes: datasets are built from non-empty part files only (6,640 parts; a 300-species chunk of 11.0 M records reads in 5.5 s), and the completion marker is RUN_COMPLETE only on a zero exit status (else RUN_FAILED).
- 2026-10-04: Fine stack v1 inputs: 3,900 Copernicus GLO-90 tiles (170°W–50°W, 5°N–75°N; copernicus-dem-90m.s3.amazonaws.com), 3,898 in the first build; the 2 missing tiles (N47 W075, N53 W063, both in Canada, outside CONUS) were re-downloaded afterwards and are no-data in the v1 North America training stack, so training points there are dropped (counted per species).
- 2026-10-04: **R environment (reproducibility).** A fresh build of the R environment failed at the first R step: `r-coordinatecleaner` is no longer installable from conda-forge with R 4.3 (PackagesNotFound), which aborted creation of the whole R environment. First fix: CoordinateCleaner pinned to 3.0.1 via `remotes::install_version` from CRAN, usdm and rangeBuilder from CRAN, and a preflight that loads every R package before fitting. Failed tasks write no summary, so they rerun.
- 2026-10-04: **Fine stack v1 built** (`scripts/build_fine_stacks.py`): GLO-90 mosaic averaged to ~230 m over 170°W–50°W, 5°N–75°N (57,600 × 33,600; elevation, slope, northness, eastness, TPI as int16; 19.4 GB) and to the CONUS 240 m grid (24 float32 layers: 19 WorldClim variables with bio1/5/6/8/9/10/11 lapse-rate corrected at −6.5 °C/km, fine elevation, 4 terrain layers; 25.2 GB). Visual check, Sierra Nevada around Yosemite: lapse correction resolves warm canyon floors (Merced, Tuolumne) that WorldClim ~1 km blurs; mean absolute bio1 adjustment 0.21 °C (range of bio1 −3.0 to 17.3 °C vs −2.0 to 17.2 °C at ~1 km); figure work/fine/sierra_check.png.
- 2026-10-04: The preflight then stopped the next build: CoordinateCleaner's CRAN install failed on a dependency (`xml2` would not compile without libxml2 headers). Root cause of both R failures: the reference environment's CoordinateCleaner 3.0.1 and usdm 2.1-7 come from Anaconda's `r` channel (repo.anaconda.com/pkgs/r), not conda-forge. The environment `daru2` is now built from the reference environment's explicit package list (`r_environment.txt`, 311 exact URLs, no solving or compiling); only rangeBuilder comes from CRAN, as in the reference environment.
- 2026-10-04: **Fine predictors v1 on Larrea tridentata (BLM AIM).** VIF kept all four terrain layers plus lapse-corrected bio8/bio9 (12 predictors). Occurrence-trained: AUC 0.987, TSS 0.830 (WorldClim-only 0.986 / 0.840); Daru-faithful hull: AUC 0.917, TSS 0.500 (0.914 / 0.521). Resolution test still flat (occurrence-trained 0.969 at 240 m vs 0.968 at 18 km). For this lowland desert shrub scored on mostly gentle BLM terrain, v1 terrain/lapse predictors add no measurable skill; to be tested on terrain-dependent species (montane, riparian) and on FIA trees before adoption.
- 2026-10-04: Preflight passed with the replicated R environment (daru2); fitting took ~100–110 s per task (one species in one mode).
- 2026-10-04: **Daru's model metrics reproduced** (`evaluate.daru_metrics`, his Fig. S2): AUC of test presences vs background (phyloregion `pa_evaluate(p = test, a = background)`), TSS as phyloregion computes it — the *median* of TPR + TNR − 1 across all thresholds (explains Daru's low median TSS of 0.42; the conventional maximum is reported alongside) — and phyloregion's continuous Boyce index. Larrea tridentata, Daru-faithful: AUC 0.860, TSS (median rule) 0.388, TSS (max) 0.613, Boyce 0.844 — on the scale of Daru's medians (0.91 / 0.42 / 0.87). maxent.jar's own test AUC (0.839) falls between background-only (0.853) and background + all samples (0.835), as it adds training samples to the background. Computed for every species in the render loop.
- 2026-10-04: **Background used by Daru's published models (D9)** — see table. Independent check (Larrea, BLM AIM, single fit β = 2): hull presences + uniform background AUC 0.906 vs bias-weighted 0.879; occurrence presences 0.987 vs 0.986 — the uniform background is both more faithful and no worse. Also checked: training at Daru's ~9 km resolution changes CV AUC by only +0.002–0.004 (Larrea 0.846 → 0.850, Echinacea 0.772 → 0.774), so resolution does not explain metric differences. The fitting run was restarted with modes `daru` (hull + uniform) and `occurrences` (thinned occurrences + bias-weighted); 142+ tasks of the earlier `hull` (hull + bias-weighted) variant are kept as a third variant. Render chunk reduced to 256 rows (the batch-invariant evaluation builds a cells × features matrix: ~8 GB at 1,024 rows for CONUS-wide calibration areas).
- 2026-10-04: **Hull-lattice failure mode (Festuca idahoensis, BLM AIM, 8,428 presence plots).** Hull-trained model AUC 0.455 (WorldClim) / 0.444 (fine) — below chance — despite CV AUC 0.78. Cause: saturation. The hull spans the interior West (124–104°W, 33–53°N) and the 500 uniform lattice points treat all of it as presence, so suitability is 0.7–0.94 nearly everywhere (Nevada, Wyoming, Utah, Colorado, where the species occurs on 2–7% of plots) while it is concentrated in Oregon, Idaho and Washington (30–41% of plots). Elevation and temperature distributions of lattice points and presence plots are similar (median bio1 7.1 vs 6.7 °C), so the failure is in occupancy density, which only real occurrences carry (future ledger L8). Pinus edulis, same panel: hull 0.851, occurrences 0.925, occurrences + fine predictors 0.936; its occurrence-trained 240 m map also beats its own 18 km aggregate (0.885 vs 0.873) — the first species where 240 m carries measurable information.
- 2026-10-04: **Interim cross-species benchmark** (render loop, species with Daru rasters; `scripts/report_benchmarks.py`). Occurrence-trained variant, 51 species: inside WCVP native areas median 0.992 (Daru 0.905; ours higher for 88%); Daru-style metrics AUC 0.867, TSS-median 0.407, Boyce 0.921 (Daru's medians 0.91 / 0.42 / 0.87); BLM AIM (8 species with ≥ 20 presence plots) AUC 0.847 vs Daru 0.742 (ours better for 75%); FIA (6 tree species) AUC 0.906 vs 0.866 (ours better for 6 of 6). Hull + bias-weighted variant, 69 species: κ vs Daru 0.457, Spearman 0.670, calibration IoU 0.718, inside native 0.978 vs 0.900; AIM (13) 0.719 vs 0.676; FIA (8) 0.913 vs 0.917. The faithful `daru` variant (hull + uniform) is rendering. Panel (AIM): Lupinus argenteus hull 0.436 (below chance, like Festuca 0.455) vs occurrences 0.720, whose 240 m map beats its 18 km aggregate (0.719 vs 0.681).
- 2026-10-04: **Fitting failures (10 of 277 tasks) and fixes.** (1) Records were selected by the GBIF-matched name, but the occurrence `species` column holds GBIF's accepted name, so synonyms came back empty (Heteromeles arbutifolia ↔ Photinia arbutifolia, Myrica cerifera, Myrica californica) → records are now selected by GBIF accepted `specieskey`. (2) rangeBuilder's all-pairs alpha hull needed 624 GB for Rubus idaeus (circumboreal) → hull input capped at 20,000 points by progressively coarser grid thinning (~1.8 km, ~4.6 km, ~9 km …); modelling presences unaffected. (3) GEOS TopologyException for an antimeridian-crossing hull (Carex aquatilis, 169°E) → geometries repaired with `make_valid` before intersection.
- 2026-10-04: **Panel complete (8 western species, BLM AIM).** Median AUC / TSS / resolution gain (240 m minus 18 km): hull + bias-weighted background (Daru's text) 0.623 / 0.181 / −0.003; hull + uniform background (`daru`, Daru's code; 6 of 8 scored) 0.713 / 0.237 / +0.016; occurrences + bias-weighted 0.795 / 0.453 / +0.018; fine predictors change little (occurrences 0.794 / 0.449). **Mechanism of the hull failures:** hull-lattice presences carry no sampling bias, so a bias-weighted background (clustered where botanists sample) teaches the model that unsampled places are suitable — anti-correlated with where species are recorded (Festuca idahoensis 0.455 with bias-weighted vs 0.694 with uniform background). Daru's text pairs exactly these two; his code (uniform background) does not. A bias-weighted background is coherent only with biased presences, i.e. real occurrences.
- 2026-10-04: **GBIF backbone change.** The 2026 occurrence files carry GBIF's new backbone: `specieskey` values are alphanumeric (e.g. Rubus idaeus `4TKGB`) and the `species` field holds the new accepted name, which mostly equals WCVP's (Heteromeles arbutifolia), while the species-match API had returned old numeric keys and sometimes older names (Photinia arbutifolia). Key-based selection therefore matches nothing; records are now selected under the WCVP accepted name or the GBIF-matched name. Earlier name-only selection failed visibly (0 records) rather than silently, so no map was built from wrong records. Retest: Heteromeles arbutifolia 114 s, Rubus idaeus 519 s (hull input capped), Carex aquatilis past the hull (antimeridian repair).
- 2026-10-04: **Daru-style metrics report both backgrounds (D10).** `daru_metrics` returns metrics against the fitting background and, suffixed `_eval_bg`, against the bias-weighted evaluation set; existing render records backfilled. (A first reading from one species, that the fitting background reproduces Daru's reported medians, was withdrawn once all 114 species were scored; see D10.)
- 2026-10-04: **Alpha-hull guard.** GEOS can reject a degenerate ring that the triangulation produces (Juncus effusus, `IllegalArgumentException`); the hull is retried with a seeded ±1e-5° (~1 m) jitter, far inside one ~1 km predictor cell, before failing.
- 2026-10-04: **Map panels** (from `scripts/plot_report_maps.py`: Pinus monophylla, Daru ~18 km vs this work 240 m, BLM AIM plots inside the calibration area, 2,615 presences / 46,318 absences). Panel note corrected: the 240 m-over-18 km gain with occurrence training is positive for 7 of 8 panel species (Artemisia tridentata −0.002), against 3 of 8 with hull training.
- 2026-10-04: **Binary rule (D1) measured on independent plots** (`scripts/threshold_study.py`; occurrence-trained, 28 AIM and 12 FIA species). Daru's higher FIA TSS over all US plots (0.783 vs 0.653 for our ESS vote) comes from easy absences far outside the range; inside the area both models cover our ESS vote leads for 12 of 12 species (median 0.540 vs 0.250). The ESS vote keeps only ~0.80 of presence plots; the P5 binary keeps 0.96 and scores TSS 0.733 (AIM) / 0.795 (FIA) over all plots, above Daru's 0.443 / 0.783. P5 adopted as the binary of the recommended (occurrence-trained) product (ledger L12); mode `daru` keeps Daru's ESS majority vote. `validate.validate` now also reports P5 scores.
- 2026-10-04: **Defect found and repaired (interrupted writes).** An interrupted fitting run lost unflushed writes: five species directories held zero-byte files (including `summary.json`), and the resumed run skipped them because a summary file existed. Every species with files written in the six minutes before the interruption (15 species, both modes) was deleted and refitted (`redo_species.csv`). Fixes: `summary.json` is now written atomically (temp file + rename) as the last step; `run_fit` treats an unparsable summary as unfinished; the render loop re-pulls a species whose summary does not parse or whose replicate models are missing. Also found: the main run had used an older copy of the code, so the alpha-hull jitter guard was not active in it; Juncus effusus is refitted with the current code.
- 2026-10-04: **Records selected through WCVP name resolution of every GBIF name** (Daru step 2a, `names.gbif_names_by_accepted`). A completeness audit of the fitting run (`scripts/audit_run.py`) found species with no records although GBIF holds many: the 2026 GBIF backbone files them under other genera (Mahonia aquifolium 142,293 records for WCVP's Berberis aquifolium; Mahonia repens, Mahonia nervosa, Morella cerifera, Morella californica). All 6,221 distinct GBIF species names in the occurrence files are now resolved through WCVP (status-ranked, misapplied never followed) and a species takes every record whose name resolves to its accepted name; 57 of 2,672 natives gain names, including heterotypic synonyms WCVP sinks (Vaccinium formosum and V. marianum under V. corymbosum, Dodecatheon meadia under Primula meadia). List: work/synonym_gain_species.csv; those already fitted are refitted.
- 2026-10-04: **Faithful replica benchmark complete** (mode `daru`, 114 Dryad overlap species, medians): agreement with Daru's published rasters suitability Spearman 0.823, κ 0.497, Jaccard 0.562 (closest of all variants: hull + bias 0.661, occurrences 0.692); calibration IoU 0.710; share of predicted range inside WCVP native areas 0.983 vs 0.886 for Daru. Independent plots: BLM AIM (29 species) AUC 0.875 vs Daru's maps 0.846; FIA (12) 0.923 vs 0.917. D10 corrected (the one-species reading of Daru's metric regime is withdrawn).
- 2026-10-04: **No fitted species lost its native records to the outlier test** (`scripts/check_cleaning_order.py`: all 55 fitted species keeping < 40% of their records re-cleaned on native records alone; 0 kept under half). The misfire (D11) empties a species entirely, so its victims are exactly the run's "only N cleaned native records" failures, which the final pass refits with the guarded order.
- 2026-10-04: **Resolution claim revised at scale** (`scripts/resolution_study.py`, 29 species × 2 modes, BLM AIM). Median AUC change from block-averaging the 240 m suitability map: to 0.96 km −0.0002 (occurrences) / −0.0004 (faithful); to 18 km −0.0043 (occurrences, i.e. 240 m better by +0.0043, 21/29 species) / +0.0005 (faithful). AUC by cell size (occurrences, median): 0.24 km 0.8906, 0.96 km 0.8909, 3.84 km 0.8943, 18 km 0.8792. The 8-species panel result (L11) does not generalize: with WorldClim ~1 km predictors, the maps hold no information finer than ~1 km. The product is still delivered on the 240 m grid (exact, aligned to NLCD/LANDFIRE for later fine predictors), but its effective resolution is ~1 km until fine predictors are added (L10).
- 2026-10-04: **Non-native pilot** (`scripts/pilot_adapted.py`): Daru's pipeline (occurrence-trained) applied to 6 non-native priority-list species on their global native ranges and projected over CONUS without a calibration mask. Projected suitability separates the species' US records from other non-natives' US records only weakly (AUC 0.56–0.81); Acer palmatum keeps 2% of its 6,262 US records inside its P5 range. Not scaled up: the adapted product needs cultivated and nursery evidence (ledger L4, L9), as decided for the adapted design on 2026-10-03.
- 2026-10-04: **Defect fixed: no-data in the CONUS render stack.** gdalwarp wrote WorldClim's no-data as float32 −3.4e38 (not NaN) into the 240 m layers, and rendering filters invalid cells with `isfinite`. 251,777 cells inside ecoregions (0.136%, mostly coastline where RESOLVE land extends past WorldClim's land mask) carried no-data in all 19 bioclimatic bands and were rendered from values clamped to each model's training minimum. Training was not affected (the global ~1 km stack uses NaN; no fitting background holds the placeholder). Fix: the value set to NaN in `conus240_stack.f32` and `conus240_fine.f32` (1.37 billion values, mostly ocean), `warp_conus_predictors.sh` now passes `-dstnodata nan`; every range card (daru, occurrences, hull) deleted and re-rendered with re-recorded hashes and benchmarks.
- 2026-10-04: **Final pass prepared.** After the main run completes, the 57 species that gain GBIF names through WCVP resolution are deleted (`refit_labels.txt`) and refitted with the current code together with every failed species. Side refits of the species affected by interrupted writes: 28/30 tasks done, no failures (Heteromeles arbutifolia now fitted in both modes).
- 2026-10-04: **VegBank added as a validation source** (BLM AIM covers only the arid West; FIA only trees, with coordinates perturbed up to ~1.6 km). Plant names resolved through WCVP to accepted species (as for GBIF); only plots with exact public coordinates (confidentiality 0) inside CONUS; a plot counts as an absence only if it lists ≥ 10 taxa (projects that record only trees would otherwise give false absences); repeat visits merged per plot. Every rendered species is scored by decoding its range card at the plot cells alone (`codec.decode_cells`, `scripts/validate_vegbank.py`), so validation does not need stored maps.
- 2026-10-04: **VegBank validation, first pass** (preliminary: only species re-rendered after the no-data fix; rerun when re-rendering completes). 53,797 usable plots (13,033 east of 100°W, 40,764 west), 9,310 species. Species with ≥ 20 presence plots, all CONUS plots, medians: occurrence-trained AUC 0.940, TSS (P5) 0.773, TSS (vote) 0.614 (135 species); faithful replica AUC 0.912, TSS (P5) 0.665, TSS (vote) 0.598 (153). By region, occurrence-trained: east AUC 0.939 / TSS (P5) 0.778 (73 species), west 0.942 / 0.760 (62); faithful: east 0.927 / 0.742, west 0.870 / 0.599. First independent check for eastern species (AIM covers only the West). Daru's maps at the same plots, 13 species so far: AUC 0.905, TSS 0.740 vs occurrence-trained 0.932 / 0.745.
- 2026-10-04: **Soil predictors, first test** (`scripts/fetch_soilgrids.sh`, `build_soil_stacks.py`, `soil.py`; SoilGrids 2.0, ISRIC, CC BY 4.0, pH/clay/sand/SOC 5–15 cm, area-averaged to the NA ~230 m training grid and the CONUS 240 m grid; background points jittered uniformly within their ~1 km cell so the ~230 m layers are sampled fairly). 8 western species, occurrence-trained, BLM AIM: AUC 0.821 with soil vs 0.793 without (7/8 better; Bouteloua gracilis 0.926 vs 0.933), TSS (P5) 0.331 vs 0.277; sub-kilometre gain unchanged (−0.0009 vs −0.0002). Second test launched: 40 species with the most VegBank presence plots among those validated so far (20 east, 20 west; drawn from the first, alphabetically early, re-rendered species), scored on VegBank.
- 2026-10-04: **Range-card storage measured at scale** (cards re-rendered after the no-data fix): faithful 812 cards, median 3.4 kB, 2.7 MB in all; occurrence-trained 593 cards, median 6.5 kB, 3.8 MB. Against the three ZSTD GeoTIFFs of the same maps (benchmark species, median 10.1–10.4 MB): median 2,912× (faithful, 107 species, range 80–10,431×) and 1,542× (occurrence-trained, 86 species, range 95–4,222×). The floor (~80×) is set by small-range species whose compressed GeoTIFFs are mostly empty. The pilot figure (2,466–4,226×) is replaced by this distribution.
- 2026-10-04: **Fitting run restructured mid-run.** `run_fit` submitted 300 tasks at a time and waited for the slowest; two range hulls of ~35 min (circumboreal / pantropical species) left nearly all workers idle. Now the pool is kept full: the next chunk is loaded and submitted as soon as fewer tasks than workers remain (verified locally). The main run was stopped (results kept) and restarted as the final pass, which at the same time refits the 57 synonym-gain species and retries every failure with the current code (WCVP name resolution of GBIF names, guarded cleaning order, hull fallbacks: planar engine, ~1 m jitter, antimeridian split, coarser thinning for degenerate rings).
- 2026-10-04: **A second interruption (16:31 UTC).** Atomic summaries protect `summary.json`, but other files written just before an interruption can still be lost from the page cache, so every species with files written 16:20–16:40 UTC (101 species, both modes) is queued for refitting through `refit_labels.txt`, and their earlier copies were removed (to be removed again after the refit, since render loops may pull interim copies).
- 2026-10-04: **Range hulls for circumboreal species.** Deschampsia cespitosa (344,339 cleaned records) fails rangeBuilder's hull in both sf engines (spherical s2: loop error; planar GEOS: side-location conflict at 179.7°E). Fallbacks in `occurrences.alpha_hull`: planar engine on s2 failure; one ~1 m jitter on a degenerate ring; coarser thinning; and, for records spanning the antimeridian, separate hulls for the Americas (169°W–30°W), the Old World (30°W–180°) and Chukotka (180°–169°W). A first version shifted the Old World to 0–191° to keep Chukotka contiguous; that failed with degenerate rings in all six attempts (~25 min each), while the unshifted Old World hull succeeds in 140 s (alpha 17). The Americas hull, the one the CONUS product uses, is unchanged by the split.
- 2026-10-04: **CoordinateCleaner sea test made reproducible.** Its default reference (Natural Earth 50 m land) is downloaded at run time through rnaturalearth; with the locally installed rnaturalearth that call fails ("unused argument" in `ne_file_name`), which failed one local fit (Callicarpa americana). `clean_coordinates.R` now takes the same layer from a local copy when `CC_SEAS_REF` is set (raw/geo/ne_50m_land, naciscdn.org, downloaded 2026-10-04); without it the package default is used (it works wherever the download succeeds). Callicarpa americana: 58,089 records, 53,569 kept.
- 2026-10-04: **VegBank validation at scale** (all species rendered after the no-data fix: ~1,880 of 2,672 per mode; species with ≥ 20 presence plots; medians). Occurrence-trained, 1,101 species: AUC 0.936, TSS (P5) 0.770, TSS (vote) 0.630; east (521) 0.941 / 0.803, west (580) 0.930 / 0.731. Faithful replica, 1,126 species: AUC 0.905, TSS (P5) 0.691; east 0.927 / 0.782, west 0.858 / 0.616. Against Daru's published maps at the same plots (74 species): occurrence-trained AUC 0.940 vs 0.896 (better for 86%; east 0.939 vs 0.917, 69% of 29; west 0.946 vs 0.859, 98% of 45), TSS (P5) 0.764 vs 0.660; faithful replica 0.913 vs 0.896 (47%; east 52%, west 44%): the replica performs like Daru's maps, as it should.
- 2026-10-04: **Refitting after an interruption made automatic.** Before a run resumes, the newest file on disk (the moment the previous run stopped) is found and every species with files written in the 10 minutes before it is deleted, so it is refitted; `pipeline.run_species` calls `os.sync()` before writing the summary, so a species marked finished has all its files on disk. Deschampsia cespitosa end to end with the three-group antimeridian split: hull valid, 1,821 s, Americas alpha 7.
- 2026-10-04: **Soil predictors, second test (40 species, VegBank)**: 37 species paired; median AUC 0.9301 (soil) vs 0.9298 (climate only), soil better for 59% (east 47%, west 72%); TSS (P5) 0.766 vs 0.756; 240 m vs 0.96 km −0.0005 vs 0.0. The +0.028 AUC gain on the 8-species AIM panel is specific to western shrubland/grassland on AIM plots. Soil is not adopted for the product; the predictor set stays available. One fit (Callicarpa americana, climate only) failed on the CoordinateCleaner download issue fixed above, so it is unpaired.
- 2026-10-04: **Record precision tested** (ledger L15): fitting only on records with stated uncertainty ≤ 250 m (54,002 of 118,711 for the 8-species panel) raises TSS (P5) from 0.277 to 0.332 and AUC from 0.793 to 0.799, but the 240 m map still holds no skill beyond its 0.96 km aggregate (−0.0002); with terrain predictors the 240 m map is slightly worse than its aggregate (−0.0029). `panel_ablation.py` gained `--max-uncertainty`.
- 2026-10-04: **Positive control for the resolution test** (`scripts/resolution_positive_control.py`). A classifier trained on BLM AIM plots with fine terrain does worse with 240 m values than with their 0.96 km block averages (8/8 species, median −0.009 AUC), but adding the 240 m values to the 0.96 km averages improves 7/8 (median +0.0012). The plots can register sub-kilometre information, but for these species it is small next to kilometre-scale context; this explains why the 240 m maps show no measurable sub-kilometre skill (L11) and motivates multi-scale predictors (L16).
- 2026-10-04: **Elevation no-data made consistent.** The CONUS stack's elevation band held WorldClim's integer no-data (−32,768) on 71,886,897 cells. Rendering was not affected (all such land cells, 251,777 inside ecoregions, are already NaN in the bioclimatic bands and masked), but neighbourhood means would have mixed it in near coasts. Set to NaN; `tests/test_stack_nodata.py` now also rejects −32,768 and −9,999.
- 2026-10-04: **Multi-scale predictors, setup** (ledger L16): `GlobalStack.focal` (mean over the 9 × 9 block of ~1 km cells around a training point, ~8 km) and `scripts/build_focal_stack.py` (29 × 29-cell, ~7 km, NaN-aware means on the CONUS 240 m grid, GPU). The two agree at 3,000 random CONUS points (bio1 r = 0.99997, median |Δ| 0.007 °C; bio12 r = 0.99999, 0.36 mm).
- 2026-10-04: **National fitting run complete** (RUN_COMPLETE ~20:00 UTC): 2,638 (`daru`) and 2,639 (`occurrences`) of 2,665 unique native species (the table's 2,672 rows include 7 priority-list entries sharing a WCVP name; `audit_run.py` now de-duplicates). Gaps: 14 species with no GBIF records under any name the download requested; 10 with < 5 cleaned native records (Daru buffers such species, D3; not yet applied); 3 CoordinateCleaner download failures (refitted with `CC_SEAS_REF`). `scripts/sync_check.py --apply` removed 200 + 86 rendered species whose final fit differs from the copy they were rendered from; they are re-rendered.
- 2026-10-04: **Decomposed predictors** (neighbourhood mean + local deviation per WorldClim variable; VIF keeps 17–18): first 6 of 8 panel species, AUC 0.777 vs 0.794 climate-only (mixed: Pinus edulis 0.941 vs 0.926, Eriogonum 0.781 vs 0.799), TSS (P5) 0.309 vs 0.279, 240 m vs 0.96 km −0.0012. Not adopted.
- 2026-10-04: **Validation domain fixed: AIM and FIA restricted to CONUS.** BLM AIM includes 366 Alaska plots and FIA 4,934 plots outside CONUS; a CONUS map scores them 0 while Daru's global map covers them. Carex aquatilis had 213 of its 226 AIM presences in Alaska, giving AUC 0.10 (ours) vs 0.98 (Daru) — an artefact; inside CONUS it has 13 AIM presences (below the 20-plot minimum), and on VegBank (already CONUS-only) this work scores 0.873. `validate.PlotTruth`, `plot_truth` and `fia_truth` now keep plots in 24–50°N, 125–66°W; all benchmark records rescored (`scripts/rescore_plots.py`). Carex aquatilis was the only benchmarked species affected; medians barely move (this work vs Daru on AIM: 0.936 vs 0.826, 28 species). **Remaining gaps filled**: the 3 CoordinateCleaner download failures and 9 species with 1–4 cleaned records (Daru's buffer rule, D3) fitted and uploaded; both models now cover 2,649 of 2,665 species (16 gaps: 14 without GBIF records under any requested name, Dudleya gnoma and Robinia × ambigua with none).
- 2026-10-04: **Species with no records under any interpreted GBIF name** (13 of the 14 remaining gaps) are in the occurrence files under a broader species GBIF lumps them into (Mentha canadensis: 10,457 records under Mentha arvensis; Salix lasiandra 4,844 under Salix lucida; Galium porrigens under Trichogalium porrigens). `run_fit` now takes such species' records by the name each record was originally identified under (`verbatimScientificName`, binomial resolved through WCVP: `names.names_of_accepted`, `names.binomial`); used only when the default finds nothing, so fitted species are unchanged. The general choice between the two rules is ledger L17.
- 2026-10-04: **Final VegBank validation** (all 2,649 species per model rendered from the final fits; species with ≥ 20 presence plots; medians): this work 1,510 species, AUC 0.938, TSS (P5) 0.773; Daru method (reproduced) 1,510 species, AUC 0.902, TSS (P5) 0.687. Against Daru's published maps at the same plots (74 species): this work 0.940 vs 0.896 (better for 86%), Daru method (reproduced) 0.913 (47%). All 5,407 range cards (incl. 109 legacy hull-variant cards) open and hold their replicates.
- 2026-10-04: **Species-concept rule for priority-list names (ledger L17), decided with horticultural judgement.** A name on the horticultural priority list means the plant as sold, so records follow GBIF's interpreted species (resolved through WCVP) by default. A record is moved only when the name it was originally identified under — the full name with any variety/subspecies, resolved through WCVP, else its binomial (`names.canonical`, `names.binomial`) — is itself a different listed species (on the priority list). Rationale: (i) the trade sells e.g. Salix lasiandra (Pacific willow) and Salix lucida (shining willow) as distinct plants, so GBIF's lump must not give the eastern willow western records; (ii) records identified as a listed species but filed by GBIF under an unlisted lump belong to it (Hepatica americana: 24,335 records filed under Eurasian Hepatica nobilis); (iii) splits toward taxa the trade does not list are not applied (Pinus ponderosa keeps records identified as P. brachyptera / P. scopulorum: +3 records net), which also neutralises WCVP synonymy oddities (Persea borbonia → Nectandra hihua: +1). Implementation: `scripts/build_name_reassignment.py` (230,161 of 173.5M records move; 639 listed species touched; 69 change by > 5%) and `run_fit.py --reassign`. The 69 species are refitted in both modes; the 14 species previously taken by verbatim fallback are among them and are refitted under the same rule.
- 2026-10-04: **Preparation for all US native species.** `scripts/build_us_natives_table.py`: 18,600 accepted species native (non-extinct) to the United States per WCVP — CONUS 15,922 only, Hawaii 1,196 only, Alaska incl. the Aleutians (WGSRPD ASK + ALU) 457 only, the rest in several; 2,665 already modelled. `scripts/match_gbif_keys_all.py`: GBIF species-match for the 15,935 others, 15,591 at species level (344 match only a genus or higher in GBIF). Supplementary GBIF download **0010385-260928105237408** submitted (`scripts/request_gbif_download.py`: 16,276 new taxon keys, same predicate and format as the first download; DOI recorded on completion). Alaska and Hawaii 240 m grids built (`scripts/build_region_grid.py`): Alaska Albers EPSG:3338, 8,336 × 15,414 cells, 31 ecoregions; Hawaii Albers ESRI:102007, 5,614 × 9,754 cells, 7 ecoregions; WorldClim warped with NaN no-data, RESOLVE ecoregions with Natural Earth lakes masked, same layout as the CONUS grid.
- 2026-10-04: **Pipeline made multi-region (CONUS, Alaska, Hawaii).** `predictors.region_stacks` loads every built 240 m grid (one class for all; each records its CRS); `render_results.py` renders a species on every grid its calibration ecoregions reach; range cards move to format 3 (`codec.grids`: per-region CRS, shape and SHA-256 of each map; format-2 cards read as CONUS-only and still verify byte-exact); Verified: Carex aquatilis (native in Alaska and CONUS) decodes from its existing card.
- 2026-10-04: **Dispersal-rate coverage for all US natives**: 200 of 241 families of the 18,600 US natives have a fitted SBM rate; 246 species in 41 families (chiefly the lycophytes Lycopodiaceae 57, Isoetaceae 54, Selaginellaceae 40, absent from the dated seed-plant trees) take the all-family median (`km_per_year` fallback). Map-neutral, because calibration areas are whole occupied ecoregions (D8), but recorded in each summary (`sbm_clade`).
- 2026-10-04: **Sharded runs prepared** (not launched): each shard downloads each GBIF archive into its own directory and writes per-shard task logs and completion markers; `run_fit.py` gains `--shard i/n` (crc32 of the WCVP name) and accepts several parquet directories; `build_name_reassignment.py` reads every GBIF download present and an optional listed-species table. For the national run the listed set stays the priority list: priority-list species keep the concept they are sold under (Pinus ponderosa broad), while every other native gets its own map from its default records or, where GBIF lumps it, from records identified under its own name (zero-record fallback; e.g. Pinus brachyptera). Each map then answers its own question; records may count toward both a broad priority-list concept and a narrow WCVP species.
- 2026-10-04: **Validation domains per region** (`validate.DOMAINS`): Alaska has BLM AIM 366 plots, VegBank 3,697 public plot visits and FIA 6,816 plots; Hawaii FIA 583 plots (trees only), none from AIM or VegBank. Recorded in docs/validation.md. GPU MaxEnt fitting (ledger L5) started: maxent.jar 3.4 source (github.com/mrmaxent/Maxent, MIT) is being specified for an exact PyTorch reimplementation (docs/maxent_fitting_spec.md), to be validated against maxent.jar before any use.
- 2026-10-04: **Supplementary GBIF download complete**: 0010385-260928105237408, doi:10.15468/dl.pkunks (SUCCEEDED 22:05 UTC; same predicate and SIMPLE_PARQUET format as the first download, 16,276 taxon keys). 50,517,941 records (equal to GBIF's reported total, counted over the 6,442 non-empty part files), 15,874 distinct species keys; 4,108,078,451-byte archive, 5.0 GB as parquet in `raw/gbif_us_natives/parquet`. Not yet used by any fit: it feeds the national all-US-natives run.
- 2026-10-04: **WCVP native-status conflicts.** Six accepted species are coded native to a CONUS state although WCVP's geographic description names only Old-World regions: Caragana arborescens (MAS; "C. Asia to Siberia and Korea"), Rubus phoenicolasius (WVA), Lonicera reticulata (12 states; "SE. China"), Achillea alpina, Aster alpinus, Festuca trachyphylla. Four have range cards. The range-card species table should be audited the same way.
- 2026-10-04: **GPU MaxEnt engine built and validated against maxent.jar** (ledger L5; `ranges/maxent_torch.py`, specification `docs/maxent_fitting_spec.md`, report `docs/maxent_torch.md`).
  - **What it is.** maxent.jar 3.4.4's fitting algorithm (github.com/mrmaxent/Maxent, MIT), reimplemented step for step in float64 PyTorch, with all runs of a species fitted as one GPU batch. It writes maxent.jar's `.lambdas`, sample predictions and results tables.
  - **Two maxent.jar behaviours found in the source and reproduced:**
    - Layers are sorted alphabetically (`ParamsPre.getSelected`). This fixes feature order and the 20-iteration expectation-refresh cycle that steers feature selection.
    - Replicated subsample runs force `randomseed=true`, so production replicate test sets are drawn from the wall clock and are not reproducible run to run. Cross-validation folds use `Random(0)` and are reproduced bit for bit.
  - **Validation.** 34 species (5–5,000 presences, both modes) refitted from their stored SWD inputs:
    - β choice identical 34/34;
    - 850 CV folds: identical feature sets in 99.3%, median max |Δλ| 1.3e-12, test AUC identical in 99.6%;
    - 170 final replicates refitted on maxent.jar's own splits (recovered from its sample predictions): |Δ ESS threshold| ≤ 0.0007, |Δ P5| ≤ 0.0003;
    - 9 benchmark species: 240 m maps identical (suitability ρ ≥ 0.99994, binary agreement ≥ 0.9997), BLM AIM and VegBank AUC within 0.00004.
  - **Remaining differences:**
    - In 0.7% of folds a near-tie in feature selection resolves differently, because GPU sums round differently from Java's. The fitted surfaces still agree (ρ ≥ 0.99996).
    - Threshold rules can differ by one background point.
    - With its own fixed-seed replicate draws the engine's maps differ from maxent.jar's by as much as two maxent.jar runs differ from each other: map ρ 0.960–0.999 vs 0.970–0.998 for jar against jar.
  - **Speed.** Measured on the shared workstation (load average about 60 on 12 cores; GPU shared with four training jobs): 21–31 s per species on one RTX 3090 vs 162–213 s for maxent.jar, about 9× fewer CPU-seconds.
  - **Integration.** Tests: `tests/test_maxent_torch.py` (Java-printed reference values; parity against stored maxent.jar runs in `tools/bench/torch_parity`). Selectable with `run_fit.py --engine torch` (`modelling.fit_species(engine=...)`, `MAXENT_ENGINE`); the default remained maxent.jar until the decision below. Validation records: `work/maxent_torch_validation/`.

### 2026-10-04 — Decision: the GPU MaxEnt engine is the default where a GPU exists
`maxent_torch` reproduced maxent.jar on 34 species in both modes (identical β for 34/34; 850 CV folds, coefficient
differences ~1e-12 in 99.3%; VegBank and AIM AUC within 0.00004 on the same splits; docs/maxent_torch.md) at 7–8×
less wall time and ~9× less CPU. Its replicate splits are seeded, whereas maxent.jar's `replicates=5
replicatetype=subsample` silently forces a clock seed, so maxent.jar's final models are not reproducible even by
maxent.jar. Decision: `run_fit.py
--engine auto` (new default) and `modelling.fit_species` use `torch` when a CUDA GPU is present and `java`
otherwise (CPU-only machines). Fits already completed with maxent.jar are kept; the two engines' maps differ only
as much as two maxent.jar runs differ from each other (map Spearman 0.960–0.999 vs 0.970–0.998).
- 2026-10-04: **Occurrence coverage of all 18,600 US natives** (`scripts/inventory_us_natives.py`, record-selection rules of `run_fit.py`). With the first two downloads, 594 of the 15,935 natives outside the priority list had no record under any name and 512 only through the verbatim-name fallback. Most zero-record species are names GBIF's backbone does not know at species rank (285 matched only a genus or higher: recently moved genera such as Anatherum, Pyrrocoma, Senega; nothospecies), so no key had been requested although GBIF holds their records under a WCVP synonym (Andropogon mohrii for Anatherum mohrii). `scripts/match_gbif_synonym_keys.py` matched every species-rank WCVP synonym of the 2,211 species with < 20 records against the GBIF backbone: 1,866 new species-level keys (848 species; 1.52M georeferenced records by the occurrence API). Third download **0010622-260928105237408, doi:10.15468/dl.52fhnr** (same predicate and format; `scripts/request_gbif_download.py --keys`): 1,241,857 records. Coverage with all three: 454 species without records (331 CONUS, 104 Hawaii, 17 Alaska; chiefly nothospecies and Hawaiian taxa with no georeferenced record), 492 with 1–4, 1,082 with 5–19, 13,907 with ≥ 20 (median 246); `work/national_inventory.csv`.
- 2026-10-04: **Name reassignment (L17) rebuilt over all three downloads** for the national run (`work/national/name_reassignment_national.parquet`, 258,074 records; the priority-list run table `work/name_reassignment.parquet` is unchanged). `build_name_reassignment.py` now decides the rule once per distinct (interpreted species, verbatim name) pair and collects records in batches (3.7 GB peak instead of loading 225M rows); rerun on the first download alone it reproduces the earlier table exactly (230,161 records, same destinations, identical summary).
- 2026-10-04: **Prepare-only path** (`pipeline.run_species(prepare_only=True)`, `run_fit.py --prepare-only`, mode `occurrences`): cleaning (D11 guarded order), WCVP native filter, ~1 km thinning, Daru's buffer rule (D3), calibration ecoregions, presences, bias-weighted background, predictors and VIF, written exactly as a full run writes them (samples.csv, background.csv, summary.json with `stage: prepared`), without the range hull (used by no step of this mode) or MaxEnt. Verified on 8 fitted priority-list species incl. two native-first cleanings and one ≥ 10,000-record species: samples.csv and background.csv byte-identical to the stored full runs (`tests/test_prepare_only.py`). A prepared summary does not count as fitted for a fitting run.
- 2026-10-04: **CoordinateCleaner in one R session for many species** (`occurrences.clean_coordinates_many`, `pipeline.clean_species`, `run_fit.py --group`). Starting R and loading CoordinateCleaner, terra and the sea reference took ~10 s per call, twice per species (both cleaning orders), i.e. most of a prepare-only species. Each record set is still cleaned by its own `clean_coordinates` call, because a joint call is not equivalent: the outlier test switches every species of the call to its raster approximation when any one has ≥ 10,000 records. Flags identical to separate sessions (test with a 12,000-record set beside small ones); the group's R start-up is paid once instead of twice per species.
- 2026-10-04: **Dated tree for all US natives** (`national_data/build_tree_all.py`, rules of the priority-list tree; `work/deepearth/tree_all/`): 18,600 tips, ultrametric at 435 Ma — exact 10,973, species-rank WCVP synonym 673, genus graft 6,509, family graft 289, order graft 5 (Joinvilleaceae, Mayacaceae, Apodanthaceae, Tetrachondraceae, Surianaceae, at the crown of their APG IV order), lycophytes 151. The megatree holds ferns and seed plants (root = euphyllophyte crown, 423 Ma) but no lycophytes; they are attached as a clade sister to it, with the tracheophyte crown at 435 Ma and the lycophyte crown at 413 Ma (midpoints of Morris et al. 2018, PNAS 115:E2274: 450.8–419.3 and 432.5–392.8 Ma), every lycophyte species at the lycophyte crown (no dates within lycophytes are available). Ferns: 574 species (281 exact, 130 synonym, 162 genus, 1 family); gymnosperms: 132 (115 exact). Of the 2,663 priority-list species also in the priority-list tree, 2,661 keep their placement (Rumex salicifolius now grafts at genus because its synonym tip R. verticillatus is itself a US native; Selaginella bigelovii was dropped and is now placed).

### 2026-10-04 21:40 — National run launched (all US native species)
- **Model**: the recommended product only — mode `occurrences` (thinned GBIF occurrences, effort-weighted background
  in whole occupied ecoregions, VIF-screened WorldClim predictors, maxent.jar protocol: β by 5-fold CV over
  {2, 5, 10, 15, 20}, 5 replicates; P5 and ESS binaries; 240 m cards on every US grid the calibration area reaches).
  The Daru-faithful replica (`daru` mode) stays the baseline of the 2,665-species benchmark; national comparison with
  Daru uses his published maps (Dryad SDM_set1: 557 US natives extracted to `daru_ref_all`) plus WCVP regions, BLM AIM
  and FIA plots.
- **Execution**: species A–B (2,100) prepared on CPU (`run_fit.py --prepare-only`; ~6 species/min on a 12-core
  workstation), then fitted in batches with the validated PyTorch engine and rendered with `ranges/render.py`, the
  render code shared with `render_results.py` (`scripts/national_run.py`); species C–Z and × (13,835) fitted in sharded
  runs with maxent.jar and rendered from the returned coefficients.
- First batch: 16 species fitted in 220 s on one GPU while the CPUs prepared; renders 1–27 s per species.
- Experimental joint-model, Earth4D and phylogenomic-encoder work of the research code was stopped at this point and
  kept out of the product.

### 2026-10-04 22:15 — One joint model for the national product; only data preparation is distributed
- Decision (22:10): instead of separate per-species models, the national product is one
  joint model over all US natives (environment network + Brownian-motion phylogenetic prior; no Earth4D, no message-
  passing encoder), validated on the 2,661 benchmark natives against per-species MaxEnt (AIM 0.944 vs 0.927, FIA 0.937
  vs 0.930, VegBank 0.957 vs 0.954; 5 records 0.931 vs 0.895; no records 0.74 vs chance). Only the per-species data
  preparation (cleaning, native filter, thinning, effort-weighted background, calibration ecoregions) is distributed.
- Per-species MaxEnt fitting of national species stopped (205 A–B species had been fitted and rendered; kept).
- Data preparation: run `national_prep_2026-10-05`, 4 shards, prepare-only, mode occurrences, R groups of 8, species
  C–Z and × (13,835; us_natives_new_table_CZ.csv), GBIF downloads 0009841/0010385/0010622-260928105237408, name rule
  L17 (name_reassignment_national.parquet); code from the current tree.
- 2026-10-04 22:40: an interrupted environment build left one shard with a half-written R environment (`Rscript: Exec
  format error`), which failed its preflight. Setup now verifies that the R and Python environments execute and
  rebuilds them otherwise.
- 2026-10-04 22:40: national map storage (`native_ranges/national_cards.py`) tested end to end on
  the Hawaii grid with the 2,661-species joint model: field 5,614 × 9,754 (290,630 land cells) in 152 s; decoded vs
  full-model scores over 300 random species (most outside their range there): Spearman median 0.982; 256² tile decode
  ~0.3 s cold on CPU. National build will sample 32 blocks per grid for the projection.

### 2026-10-04 23:25 — National model choice and the first evaluation of stored joint-model maps
- **Model decision**: joint environment network + Brownian-motion phylogenetic prior + a species-specific term
  (`train_joint.py --species bm --tip-free`, weight decay 1e-2 on clade edges, 1e-4 on the species term). On the 2,661
  benchmark natives (3,000 steps): full data dev/test 0.9388/0.9559, unclipped test 0.9507, AIM 0.9444, FIA 0.9359 (no
  phylogeny: 0.9410/0.9570, 0.9476, 0.9440, 0.9373; plain prior 0.9386/0.9555, 0.9484, 0.9437, 0.9359); 5 records
  (reduced half of the VegBank species) 0.8979/0.9287, unclipped 0.8181/0.8447 (no phylogeny 0.8992/0.9266,
  0.7733/0.7738; MaxEnt on the same 5 records 0.8732/0.8948); no records, unclipped 0.7224/0.7509 (no phylogeny
  ~0.50; nearest relative's map 0.6001/0.6186). Full-data differences between the three are ≤ 0.002.
- **Stored maps evaluated as served** (`national_cards.py`: CONUS 240 m shared field, rank 64, int8, projection energy
  0.996 over 265,256 sampled cells; `eval_national_maps.py`; all plots, species with ≥ 20 presence plots; plots outside
  a species' calibration ecoregions ranked lowest): VegBank 1,514 species median AUC 0.939 — vs Daru's published maps
  0.943 vs 0.896 (74 species, 88% better), vs per-species MaxEnt cards 0.941 vs 0.940 (1,398; 48%); BLM AIM 455 species
  0.928 — vs Daru 0.967 vs 0.826 (28; 96%), vs MaxEnt 0.930 vs 0.909 (414; 74%); FIA 221 species 0.936 — vs Daru 0.940
  vs 0.916 (12; 100%), vs MaxEnt 0.934 vs 0.934 (214; 50%). Model: the 2,661-species plain-prior run (s3_bm).
- 2026-10-04: **Prepare-only cost and memory, measured** (stage profile of 7 species, 20–59,485 records, on the shared 6-core workstation): CoordinateCleaner 24 s per species (one R session per group), rasterizing the occupied ecoregions at ~1 km 12.5 s, their union 11.6 s, bias weights at the calibration cells 4.4 s, background draw 0.9 s, everything else < 1 s; a 133 s scan of the 225M-record files per chunk in the main process. Calibration areas of circumboreal species reach ~2e8 ~1 km cells; per-cell arrays (int64 row/column pairs, float64 coordinates and weights) took a worker to 8–13 GB. Now: cells by row blocks as int32, bias weights in chunks, only the 10,000 drawn cells converted to coordinates, the sampling probabilities formed in place (`background.sample_background_index`); identical draws (byte-identical samples.csv and background.csv for a 6,152-, a 1.4M- and a 188M-cell species). Workers hand freed memory back after every task (`malloc_trim`), exit when their main process ends, and at most `LARGE_AREA_SLOTS` workers hold a calibration area of > 3e7 cells at a time (lock files). The verbatim-name fallback scans the files once per chunk instead of once per fallback species (record sets verified identical).
- 2026-10-04: **National prepare-only run split**: species A–B (2,100) prepared sorted by record count, largest first; C–Z and × in four shards (run national_prep_2026-10-05; archives unpacked to work/national/prepared/). Setup fixes found on a first attempt: a stale R library lock left by an interrupted install is removed before installing, and the preflight of a prepare-only run does not require rangeBuilder.
- 2026-10-05: **Remaining A–B species prepared as a separate shard.** The C–Z shards finished in about an hour each, while the A–B run (largest species first) projected ~8 h for its last 1,378 species. A separate run, national_prep_ab_2026-10-05 (groups of 8, same code, inputs and GBIF downloads as the C–Z run), prepared them: RUN_COMPLETE at 07:34 UTC (1,378 species in 42 min including setup). The 50 species the A–B run had finished are kept; the archive overwrites them with identical files.
- 2026-10-05 00:05: seed check of the national model choice (2,661 natives, 3,000 steps, VegBank/AIM/FIA): no phylogeny
  seeds 0/1 dev 0.9410/0.9403, test 0.9570/0.9559, unclipped test 0.9476/0.9506, AIM 0.9440/0.9439, FIA 0.9373/0.9370;
  Brownian prior + species term seeds 0/1 dev 0.9388/0.9389, test 0.9559/0.9562, unclipped 0.9507/0.9502, AIM
  0.9444/0.9427, FIA 0.9359/0.9350. Seed spread ≈ 0.001–0.003; the prior's full-data cost is within it except dev
  (−0.002), against +0.06 to +0.24 unclipped AUC for species with 5 to 0 records. Choice confirmed.
- 2026-10-05 00:35: national joint-model data (work/deepearth/national; interim: 13,295 species, 145.8M points,
  13.4M presences, 24.2M distinct cells) had plot ecoregion ids of −1 for every CONUS plot (VegBank, AIM, FIA), so no
  plot fell inside any calibration area. Cause: importing rasterio before pyproj makes pyproj's array transforms return
  inf in this environment (rasterio ships its own PROJ). Recomputed from the CONUS ecoregion raster with pyproj
  imported first (VegBank ids equal the benchmark build for 53,796 of 53,797 plots); guards that raise on non-finite
  transforms added to national_cards.py and eval_national_maps.py. Alaska and Hawaii plot sets were unaffected.
  First national training (interim data, 6,000 steps) restarted on the corrected data: 64–180 ms/step, 16 GB GPU.
- 2026-10-05: **Silent projection failures guarded.** A national plot build wrote ecoregion id −1 for every CONUS plot (VegBank, AIM, FIA; those files were corrected, see 00:35): pyproj returned inf for the 4326 → EPSG:5070 transformation in that process, and inf became "outside every ecoregion" (not reproducible in later runs; Alaska and Hawaii grids were unaffected). Every projection in the national data path now fails loudly on non-finite output (`Resources.bias_at`, the national plot builder, `build_hawaii_plots.py`), and pyproj is imported before rasterio there. Audit of the bias-weighted backgrounds of all 13,295 species then prepared or fitted (`national_data/check_background_bias.py`): a weighted draw never picks a zero-bias cell, and none of the 13,215 species with weighted draws (> 10,000 calibration cells) has one, so no background fell back to an unweighted draw.
- 2026-10-05 01:08: prepare run `national_prep_2026-10-05` complete — 4 shards (13,835 species C–Z), archives
  prepared_{0..3}_4.tar.zst.
- 2026-10-05 01:05: interim national joint model (13,295 species prepared at the time; bm + species term; 6,000
  steps): VegBank dev/test 0.9380/0.9556 (MaxEnt cards on the same species 0.9329/0.9535), unclipped test 0.9488,
  > 10 km plots 0.9171, BLM AIM 0.9361 (MaxEnt 0.9268), FIA 0.9326 (MaxEnt 0.9297). The 2,661-species model reached
  AIM 0.944 at equal per-species exposure; the full run uses ~21,000 steps (≈ 290 draws per species).
- 2026-10-05 01:55: **national training data complete** (work/deepearth/national): 17,912 species
  (2,663 priority-list natives from the fitted run + 15,249 prepared nationally), 194,200,260 training points (15,864,145
  presences + effort-weighted background), 27,265,128 distinct 30" cells; dated tree pruned to these species; plot
  sets VegBank, BLM AIM, FIA (CONUS; ecoregion ids recomputed, see 00:35), AIM/FIA Alaska, Hawaii. The remaining ≈ 690
  of the 18,600 US natives have no usable georeferenced record after cleaning.
- 2026-10-05 01:55: **national joint model training started** (train_joint.py --species bm --tip-free, 21,000
  steps ≈ 300 sampling draws per species, evaluation every 3,000 steps; GPU 0).
- Plan, decided: species without usable records get zero-shot maps from the phylogenetic prior (niche vector = sum
  of the trained ancestral-edge vectors on the path to their tip in the full 18,600-species tree; the unobserved
  terminal branch contributes its prior mean, zero), calibrated to the ecoregions overlapping their WCVP native
  regions; flagged as "inferred from relatives" in every product.
- 2026-10-05 02:55: Alaska check — on the 366 BLM AIM plots in Alaska, our per-species MaxEnt range cards score a
  median AUC of 0.582 (15 priority-list species with ≥ 5 presence plots there), the national joint model 0.60 at 3,000
  steps: the Alaska AIM plots cluster in a small, environmentally homogeneous area, so within-area discrimination is
  hard for every model; not a defect of the national model.

### 2026-10-05 03:30 — National joint model trained (all US natives with records)
- `train_joint.py --data-dir work/deepearth/national --species bm --tip-free --steps 21000` on 17,912 species,
  194.2M points, one RTX 3090 (61–141 ms/step; ≈ 50 min of training plus evaluations). Best VegBank-dev checkpoint at
  18,000 steps: VegBank dev/test 0.9378/0.9561 (per-species MaxEnt cards on the same species 0.9329/0.9535),
  unclipped test 0.9485, presence plots > 10 km from training records 0.9141; BLM AIM paired 0.9314 (MaxEnt 0.9268),
  all AIM species median 0.9213; FIA paired 0.9278 (MaxEnt 0.9297), all FIA species 0.9407; FIA Alaska 0.80, AIM
  Alaska 0.59 (MaxEnt cards 0.58 on the same plots), Hawaii NPS plots 0.73 (107 species; first independent Hawaii
  check). Benchmark-species accuracy is kept at national scale (2,661-species model: 0.9388/0.9559).
- Map store build started (national_cards.py, CONUS + Alaska + Hawaii, rank 64, zero-shot): 683 species without
  usable records mapped from relatives (tree position + WCVP native regions) → 18,595 species in the product.

### 2026-10-05 06:00 — Map store: rank 64 costs 0.011 AUC nationally; transform coding instead of rank truncation
- Rank-64 store (cards_national) evaluated on independent plots (`eval_national_maps.py`): VegBank 3,614 species
  median AUC 0.938, AIM 2,013 species 0.898, FIA 259 species 0.926; vs Daru on the shared species VegBank 0.932 vs
  0.910 (better for 75%), AIM 0.889 vs 0.853 (69%), FIA 0.909 vs 0.916 (86% better; medians of 14 species). Against
  per-species MaxEnt range cards the rank-64 maps lose (VegBank 0.928 vs 0.938) although the full joint model wins on
  the same plots (training evaluation above). Cause: with 18,595 species the score matrix spans nearly all 256
  feature dimensions, so 64 channels keep 92.5% of its variance (`card_fidelity.py`: full 0.9516 vs stored 0.9391).
- Rank sweep (`rank_sweep.py`, 1,500 VegBank species, full-model median AUC 0.9512; mean AUC loss / CONUS field):
  r 64: 0.0108 / 12.2 GB; 96: 0.0066 / 18.3; 128: 0.0045 / 24.5; 160: 0.0028 / 30.6; 192: 0.0013 / 36.7;
  256: 0.0001 / 48.9 GB (int8, one scale per channel).
- Per-tile local bases (256² tiles, eigenvectors of each tile's own score covariance, with or without restricting
  to species whose calibration area touches the tile) need a median 120 channels for 99.98% of the tile's variance:
  little gain over a global basis, not adopted.
- Adopted: transform coding of the full feature space (national_cards.py `--codec klt`). y = (h − μ) M^½ V with
  M = WᵀW over all species and V the eigenvectors of the feature covariance in that metric; f_s = y · r_s + o_s,
  r_s = Vᵀ M^−½ w_s. The r_s form an orthonormal frame (Σ r_s r_sᵀ = I), so one uniform quantizer step Δ serves all
  channels and species; low-variance channels round to mostly zeros; 128² tiles as int16 byte planes + zstd.
  Same 1,500 species (`tc_proto`, scratch): Δ 0.8: mean AUC loss 0.000004, 153 B/land cell; Δ 1.6: 0.00001, 119 B;
  **Δ 3.2: 0.00007 (max 0.0036), 83 B → CONUS ≈ 15.8 GB**; Δ 6.4: 0.0003, 47 B (8.9 GB); Δ 12.8: 0.0009, 27 B.
  Spatial prediction (left or planar neighbour) before zstd made tiles larger at every Δ: neighbouring 240 m cells
  correlate (feature median 0.94) but the quantized low-variance channels are close to white.
- Conventional baseline for the same product: per-species uint8 rasters clipped to each species' calibration
  ecoregions = 502 GB (CONUS 458.5, Alaska 42.7, Hawaii 0.3); as zstd GeoTIFFs over the CONUS window ≈ 11 MB per
  species (priority-list cards) ≈ 200 GB. Transform-coded store ≈ 10× smaller than compressed GeoTIFFs, 25× smaller than
  uncompressed clipped rasters, at an AUC change below 1e-4.
- Exact alternative measured: decoding from the 24 predictor grids through the network on CPU (8 threads) costs
  0.65 s per 256² window and 2.2 s per 512² window (5 dense 256 × 256 layers per cell), above the 500 ms target;
  the stored field is the fast path, the predictors + 2 MB network remain the exact reference.
- Builds started: rank 256 int8 (`cards_national_r256`, memmap windows for bulk rendering) and klt
  Δ 3.2 (`cards_national_klt`, distribution format). Codec tests: `native_ranges/tests_store.py` (KLT identity,
  tile writer → reader round trip, packed valid mask).
- Hawaii and Alaska context (national_final evaluation log): Hawaii NPS plots (120 species, median 65 presence plots)
  fall from 0.762 at 3,000 steps to 0.732 at 21,000 while CONUS VegBank-dev holds 0.942–0.944: mild overfitting where
  records are sparse and the climate is unlike the mainland. AIM Alaska (85 species, 366 plots in a few interior
  regions) stays near 0.59, as the per-species MaxEnt cards (0.58); FIA Alaska (11 trees) rises to 0.80. Daru (2024)
  has published maps for only 2 species on each of these plot sets, too few to compare. Checkpoints are not chosen on
  these plots (they are the evaluation). Next lever: the staged Hawaii Climate Data Portal 250 m layers (ledger).

### 2026-10-05 06:45 — Transform-coded national store built and verified
- `national_cards.py build --codec klt --delta 3.2` (national_final, CONUS + Alaska + Hawaii, 18,595 species incl.
  683 zero-shot): fields CONUS 15.18 GB (191.1M land cells, 79 B/cell, 19 min on one RTX 3090), Alaska 3.73 GB
  (59.6M cells), Hawaii 0.04 GB; species table 158 MB; store total 19.2 GB.
- Compression fidelity (`card_fidelity.py`, 3,605 VegBank species): median AUC full model 0.9505, stored 0.9506,
  mean loss −0.0001; Spearman of stored vs full scores inside each calibration area median 0.989 (rank-64 store:
  loss 0.0108, Spearman 0.880). Storing the maps no longer costs accuracy.
- CPU decode (`store_bench.py`, 8 threads, measured while the host ran three other jobs at load 12/12 cores):
  256² window 150 ms, 512² window 780 ms cold / 357 ms cached. Profile per 128² tile: zstd 40 ms, byte interleave
  12 ms, scoring 20 ms. Channels are ordered by variance and within a tile almost all quantized values fit int8
  (median tile: 0 channels need 16 bits, max 1), so the tile layout now stores the first k16 channels as int16 byte
  planes and the rest as int8: 13% smaller tiles, 2.8× faster decode per tile (17 vs 48 ms). The built store is
  rewritten losslessly in that layout (`national_cards.py recode`); readers accept both layouts.
- Engineering notes: a shared zstd compressor across writer threads segfaulted the first build (now one per thread);
  the int8 rank-256 build ran out of GPU memory from allocator fragmentation after the 18,595² eigendecomposition
  (now freed explicitly, features in 1M-cell chunks, expandable segments).

### 2026-10-05 08:45 — Joint model and map store ported into this package
- `ranges/joint/` (tree, model, data, train, zero_shot, store), `scripts/national_{train,store,store_eval}.py`,
  `configs/national_joint.json`, `docs/joint_model.md`, `tests/test_joint_store.py`. Only the production method is
  ported (environment network + Brownian-motion prior with species term; klt store); research options were left out.
- Equivalence with the research harness: production checkpoint scores at 10,000 training points × 50 species agree to
  1.1e-5 (scores up to 55; bit-identical given identical species vectors); the store reader decodes the production
  store identically (max difference 0.0 over 12 windows in three regions, scores, field, cells and uint8 maps); a
  600-step training on the 2,661-species benchmark gives VegBank-dev median AUC 0.93283 / 0.93284 (port, two runs) vs
  0.93276 / 0.93285 (harness), i.e. within the GPU's run-to-run variation; zero-shot species, relatives, calibration
  and offsets identical; plot evaluation reproduces the production eval_summary.json exactly. Tests: 39 pass (34
  existing + 5 new; CPU-only capable).
- Found while porting: the zero-shot code uses the edges common to all closest trained relatives (down to their own
  common ancestor) where the Brownian-motion expectation is the point where the species joins them; different for 75
  of the 683 inferred species (36 with a single sister, whose terminal branch is then included). Being tested.

### 2026-10-05 09:00 — Decision: species without records take the path to the node where they join the tree
- The joint model's path matrix is built on the full 18,600-tip tree (columns = its nodes), so every edge above the
  node x where an unrecorded species joins its closest trained relatives has a learned increment; only its own
  terminal branch is uninformed. Under the Brownian prior its expected vector is the path sum root → x. The first
  implementation used the edges common to all closest relatives, i.e. the path to the relatives' own common ancestor,
  which adds their shared descent below x (a single sister's whole terminal branch when only one relative exists).
- Leave-one-out test with the national model (each of 3,605 trained species with ≥ 20 VegBank presences treated as
  unrecorded; both rules equally favoured by the shared edges having seen its data): median VegBank AUC path-to-x
  0.9062 vs relatives' ancestor 0.9003; where the rules differ (3,086 species) 0.9074 vs 0.9003, better for 72%;
  single-relative cases +0.009 mean. Adopted (zero_shot.py).
- Production: 75 of the 683 species without records change (median 11% of vector length). Both stores updated in
  place without rebuilding fields (`national_cards.py update-zero-shot`: codes and offsets are linear in the species
  vector; the map is recovered from the 17,912 trained species to 7e-8 relative error (codes) and 6e-5 score units
  (offsets); quantiles and P5 recomputed for the 75). Check: stored vs full-model scores of the 683 in a 61 km CONUS
  window, RMS error / score SD median 0.04 (klt), no larger than for trained species (0.11), so the update is exact
  up to the codec's quantization.

### 2026-10-05 11:05 — Ledger L22 tested: global soil at records outside North America
- Global SoilGrids 2.0 1 km aggregates (5–15 cm mean; `scripts/fetch_soilgrids_1km.sh`, 680 MB) sampled at the
  1,662,698 benchmark training points outside the North America soil grid (5.0% of 33.0M; 27% presences); all four
  properties filled at 89% (rest water/no data); medians pH 5.7, clay 24%, sand 40%, SOC 41 g/kg
  (`native_ranges/soil_global_fill.py`, data dir work/deepearth_soilglobal).
- Benchmark (2,661 species), production settings, 3,000 steps, seeds 0 and 1 (`native_ranges/l22_soil_test.sh`):
  VegBank dev 0.9353 → 0.9354, test 0.9478 → 0.9486, > 10 km 0.9085 → 0.9087, AIM 0.9437 → 0.9442,
  FIA 0.9355 → 0.9352 (two-seed means; seed-to-seed spread ~0.001). By share of a species' presences abroad
  (dev/test mean AUC, 1,514 species): < 1%: +0.0001 (1,416 species); ≥ 30%: +0.0036 (60 species, 58% better).
- Decision: adopt in the next national model version (more complete data, gain concentrated in widespread species
  with most records abroad); no retrain of the current national product for it alone (overall change within seed
  noise; a retrain would force a full store rebuild).

### 2026-10-05 12:25 — Capacity is not the limit (benchmark, two seeds each)
- Production settings at 3,000 steps on the 2,661-species benchmark (`native_ranges/v2_capacity_test.sh`), two-seed
  means (dev / test / > 10 km / AIM / FIA): width 256 × depth 3 (production) 0.9353 / 0.9478 / 0.9085 / 0.9437 /
  0.9355; width 512 0.9369 / 0.9481 / 0.9073 / 0.9432 / 0.9339 (25% slower); depth 4 0.9360 / 0.9483 / 0.9082 /
  0.9446 / 0.9353. Differences on the independent plots are within the seed-to-seed spread (~0.001).
- Decision: keep width 256, depth 3. With capacity not limiting, the remaining levers are the predictors (habitat and
  land cover, finer regional climate such as the staged Hawaii 250 m layers; ledger L18 second entry) and the global
  soil fill (L22), which together define the next national model version.

### 2026-10-05 14:15 — Land cover tested: no gain on independent plots; FIA worse
- Predictors: NALCMS 2020 v2 land cover (30 m, Canada/US/Mexico, Lambert azimuthal equal-area; class 0 = unclassified/
  outside, 127 = nodata), fraction of 8 habitat groups in the 240 m × 240 m square centred on each point (8 × 8
  pixels, over classified pixels only): needleleaf (classes 1, 2), broadleaf/mixed forest (3–6), shrubland (7, 8, 11),
  grassland (9, 10, 12), wetland (14), cropland (15), urban/barren (13, 16, 17), water/snow (18, 19). No classified
  pixel → NaN in all 8, turned into one missing flag as for soil (6.1% of the 33.0M training points, all outside
  North America; < 0.1% of plots). Built tile by tile from the raster (`work/deepearth_landcover/build_landcover.py`,
  13.3M distinct squares, 70 min on 10 CPU workers, < 6 GB RAM); 18 points checked by hand against the raster and the
  projection against gdaltransform (< 0.5 m). Data dir work/deepearth_landcover (32 predictors; rest symlinked).
- Trainer: missing flags now one per predictor group found by name prefix (`soil_`, `lc_`) instead of the fixed soil
  columns 20–23. The 24-column inputs are bit-identical before and after; 500-step reruns (seed 0) give dev 0.93179
  before, 0.93183 and 0.93170 after (run-to-run GPU variation of the same code is larger than the before/after gap).
- Benchmark (2,661 species), production settings, 3,000 steps, seeds 0 and 1, two-seed means (baseline → land cover):
  VegBank dev 0.9353 → 0.9348, test 0.9478 → 0.9473, > 10 km 0.9085 → 0.9119, AIM 0.9437 → 0.9440,
  FIA 0.9355 → 0.9304. Per species: > 10 km +0.0015 mean (53% better); FIA −0.0041 mean, 84% of 221 species (214 trees)
  worse (Wilcoxon p = 3e-20); VegBank test −0.0009 mean (44% better). No habitat class or growth form gains
  consistently; wetland species (≥ 20% wetland around their records, 155) +0.0014 on VegBank test.
- Why FIA loses: around FIA plots holding the evaluated trees, 25% of the 240 m squares are < 10% forest and 16% are
  ≥ 50% cropland/urban, consistent with the perturbed public FIA coordinates (up to ~1.6 km): a 240 m predictor is
  read at the wrong place. The FIA loss also grows with the urban share around a species' records (−0.0066 median
  for the most urban third vs −0.0006 for the least): land cover at GBIF records carries where people record and
  plant trees, not only habitat. Records vs effort-weighted background: cropland 0.08 vs 0.14, water 0.04 vs 0.02.
- Decision: do not add land cover to the national model. Overall change within seed noise or negative; the one gain
  (> 10 km, +0.003) does not hold on AIM/FIA. Revisit only with fine-scale recording effort modelled (the bias grid
  is 10 km, so effort inside a cell is attributed to habitat) and scored on plots with exact coordinates.

### 2026-10-05 14:40 — WCVP native status: 8 species flagged as uncertain (kept, with a caution)
- Rubus phoenicolasius (wineberry, East Asian) appears in the product as a US native. WCVP (files dated
  2026-06-04) lists it native in West Virginia (introduced = 0) and introduced in 21 other US areas; our native filter
  follows WCVP exactly, as Daru's does, so the entry is WCVP's.
- Screen over all 18,600 national species (`work/wcvp_suspect_natives.csv`): native in ≤ 2 US level-3 areas,
  introduced in ≥ 3, and otherwise native only outside the Americas → 8 species: Stachys palustris (VER),
  Puccinellia distans (ASK), Rubus phoenicolasius (WVA), Trifolium glomeratum (ALA), Festuca trachyphylla (COL),
  Caragana arborescens (MAS), Nephrolepis cordifolia (HAW), Tribulus cistoides (HAW). The first six look like WCVP
  coding errors or contested status; the two Hawaiian entries are plausibly indigenous. A looser screen (native ≤ 3,
  introduced ≥ 2× native) adds 5, mostly Hawaiian coastal species considered indigenous (e.g. Scaevola taccada).
- Decision: keep WCVP as the authority (faithful to Daru), but mark these 8 in the product as "native status
  uncertain" with the evidence (their WCVP native and introduced US areas), rather than silently removing them.
  Ledger: report the six likely errors to the WCVP team.

### 2026-10-05 14:45 — Decision: Alaska and Hawaii removed from modeling
- Scope is now the contiguous United States: 16,947 species native to CONUS per WCVP (us_regions contains CONUS;
  16,448 with usable records), out of the 18,600 US natives; 1,653 species native only to Alaska and/or Hawaii drop
  out. Training points inside Alaska (WGSRPD ASK + ALU, buffered 0.05°) and Hawaii are removed; points in the species'
  native ranges abroad (e.g. Canada, Europe) stay, as in Daru (`native_ranges/make_conus.py` → work/deepearth/
  national_conus). The national model is retrained with unchanged production settings (`conus_pipeline.sh`: run
  conus_final, store cards_conus_klt evaluated on VegBank/AIM/FIA, cards_conus_r256); Alaska and Hawaii
  plot metrics are no longer reported. The Hawaii 250 m climate test was stopped.

### 2026-10-05 15:00 — Every CONUS native gets a CONUS calibration area
- 98 of the 16,448 CONUS-native species with records had calibration ecoregions (those holding their cleaned
  records) entirely outside the CONUS grid (e.g. records only in Canada or Mexico, or at the edge of a large
  ecoregion), so their maps were empty in the US. For these, the calibration area
  is extended with the RESOLVE ecoregions of their WCVP native areas in the contiguous US (≥ 10% of the ecoregion's
  area inside the area, the rule used for species without records; ecoregions merely intersecting the area when none
  passes, e.g. Saxifraga rivularis on New Hampshire's alpine zone). `national_cards.conus_calibration`; applied in the
  CONUS store build. After it: 0 CONUS natives without a CONUS calibration area.

### 2026-10-05 16:30 — CONUS product: model, store and evaluation
- `conus_final` (production settings unchanged; 16,448 species, 179.1 million points, 15.3 million presences;
  21,000 steps, 43 min on one RTX 3090): the checkpoint chosen on VegBank-dev (median over species with a
  per-species MaxEnt map) is that of 9,000 steps: VegBank dev/test median AUC 0.9422/0.9555 (3,367/3,419 species;
  paired with MaxEnt 0.9371 vs 0.9329 and 0.9547 vs 0.9535), BLM AIM 0.9216 (2,009 species; paired 0.9320 vs
  0.9268), FIA 0.9426 (259; paired 0.9266 vs 0.9297).
- Store `cards_conus_klt` (step 3.2; 494 species without records mapped from relatives; 98 calibration areas
  extended): 9.06 GB (field 8.88 GB, species table 0.14 GB). Fidelity on 3,605 VegBank species: median AUC 0.9509 full
  vs 0.9509 stored, mean loss −0.0001, Spearman inside the calibration area median 0.989 (5th percentile 0.979). CPU
  decode (8 threads, shared workstation): 256² window 111 ms, 512² 633 ms cold, 177 ms cached.
- As stored, on independent plots: VegBank 3,607 species median AUC 0.951 (vs Daru 0.948 vs 0.910 on 141 species,
  joint better for 84%; vs per-species MaxEnt 0.942 vs 0.938 on 1,547, 53%); BLM AIM 2,009 species 0.922 (vs Daru
  0.921 vs 0.853 on 104, 86%; vs MaxEnt 0.915 vs 0.906 on 480, 64%); FIA 259 species 0.942 (vs Daru 0.936 vs 0.916
  on 14, 100%; vs MaxEnt 0.933 vs 0.935 on 224, 46%).

### 2026-10-05 16:45 — Published in the DeepEarth repository (models/habitat/plant)
- The per-species pipeline, the joint model and its map store are published as the package `ranges` with the
  scripts from download to decoding, the configuration of the CONUS product (`configs/conus.json`; every path under
  one data root) and the tests. Web viewing and deployment code are not part of it.
- Checked against the production code and data before publication: the published reader decodes both production
  stores bit for bit (scores, field, cells, suitability, binary range; 28 windows); the zero-shot rule, region
  inventory and calibration extension reproduce the CONUS store's species table (494 species without records,
  codes to 8e-7, all 16,942 calibration areas); the region rule excludes the same training points as the research
  code on 4 million random points; the tree builder and the sampling-effort density reproduce the production files
  byte for byte.
- Reader additions: species by name, the binary range (f ≥ P5 inside the calibration area), and values at points
  given as longitude and latitude. Stores with the earlier all-int16 tile layout (3-column index) are read and
  recoded. Checkpoint selection is configurable (`selection`), with the production rule (median over species with a
  MaxEnt map) as default.


### 2026-10-05 21:55 — Benchmark against published distribution products (SINR, iNaturalist, BIEN, Daru)
- Protocol: every competitor scored as `national_store_eval.py plots` scores the stored maps (VegBank, BLM AIM, FIA;
  CONUS; species with ≥ 20 presence plots; AUC per species × source; 5,875 tests over 4,630 species). Ours: a plot
  outside the calibration ecoregions ranks lowest; competitors: their own score everywhere (missing = 0). Also scored
  both masked to our calibration areas and neither masked. Statistics: median AUC, share of tests where ours is higher,
  two-sided Wilcoxon signed-rank on paired tests. Code: research `benchmark/` (not part of this package).
- Competitors (accessed 2026-10-05): SINR (Cole et al., ICML 2023; github.com/elijahcole/sinr @594040c) released
  coordinates-only and distilled models, and SINR with WorldClim 5′ inputs retrained with the authors' code, data
  (15.1M iNaturalist observations) and the released model's configuration (IUCN range MAP 0.765 vs 0.658 for the
  released coordinates model; 4,282 tests); iNaturalist Open Range Map v2.34 and the public small geomodel; BIEN 4.2.8
  ranges (AIM only: BIEN holds VegBank and FIA plots); Daru (2024).
- Leakage: VegBank, AIM and FIA are not in GBIF; GBIF records within 2 m of a presence plot are 0.28% (VegBank, plot
  vouchers at FLAS, DAV, NCU), 0.01% (AIM), 0.002% (FIA). Excluding them (leak1km), plots near our records (near1km)
  or keeping only presence plots > 10 km from them (far10km) moves no median by more than 0.005.
- Results (environment model, store `cards_conus_klt`; ours vs competitor, share ours higher): Daru 0.939 vs 0.896
  (85%, p = 1e-27); iNaturalist ranges 0.944 vs 0.874 (92%); BIEN (AIM) 0.919 vs 0.743 (96%); SINR coordinates 0.943
  vs 0.943 (58%); SINR distilled 0.943 vs 0.939 (59%); **SINR env 0.943 vs 0.951 (44%, p = 8e-24)**: VegBank 0.947
  vs 0.957 (43%), FIA 0.943 vs 0.950 (33%), AIM 0.925 vs 0.927 (49%, parity), far10km 0.911 vs 0.924 (40%). With
  both maps restricted to our calibration areas: parity (0.943 vs 0.942); with neither: SINR env ahead (0.936 vs
  0.951, 34%). The calibration-area rule is what brought the environment model to parity.

### 2026-10-05 22:45 — Toward beating SINR env: where it leads, background design, learned calibration, exact geodesy
- **Where SINR env leads** (4,282 paired tests): paired median difference −0.0016; 10% of tests (difference < −0.05)
  carry more than the whole mean deficit (−0.0107). In that tail the calibration area excludes the species' habitat:
  median AUC clipped 0.819, unclipped 0.856, SINR 0.946, SINR clipped to our areas 0.850. Example: *Caltha
  leptosepala* on AIM, 19 of 20 presence plots in an ecoregion holding none of its 1,856 training records (clipped
  0.17, unclipped 0.993). Only 0.9–1.4% of presence plots lie outside calibration areas, but 5–9% of species have more
  than 10% outside.
- **Soft calibration bound** (saved VegBank scores of the environment model): penalty δ (score SD) outside the area,
  dev/test: δ = 0 0.9372/0.9498, δ = 1 0.9464/0.9589, δ = ∞ (the hard rule) 0.9443/0.9580. The area is informative
  but should not be absolute.
- **Background design.** (i) Target-group background (Phillips et al. 2009): each effort-weighted background point
  moves to the nearest training record of another species, so presences and background share the records'
  fine-scale sampling. Without it every place pathway tested (absolute Earth4D hash grid of the coordinates; the
  community of nearby records, DeepEarth fusion's neighbour context) learned "a record site is here" and lost on
  held-out plots (e.g. community channel dev 0.906 at 250 steps, training loss −1.97). (ii) Continental background
  (512 points per species and step from all species' background pools, moved the same way): the species learns where
  it is absent, as SINR's "assume negative" loss does. Result (2,661-species benchmark, from the environment model,
  1,000 steps): unclipped above clipped for the first time, dev 0.9414, test 0.9509, AIM 0.9342 (target-group
  control, clipped: 0.9362 / 0.9497 / 0.9291).
- **Learned calibration.** Per species a penalty softplus(Brownian-motion path sum + c0), initially 3, applied
  outside its calibration ecoregions (ecoregion raster at the record's own 240 m cell, as for plots), learned against
  the continental background; relatives share it through the tree. Store: species.npz `penalty`; decoding lowers
  cells outside the area by the penalty instead of zeroing them.
- **Exact coordinates.** Records and plots are placed on the 240 m EPSG:5070 rasters by the same transformation
  gdalwarp applied when warping them: PROJ's per-point choice among EPSG's WGS 84 → NAD83 operations (best accuracy,
  then PROJ's list order), NOAA HARN / NRC Canada grid shifts with PROJ's grid hierarchy (named parents,
  bounding-box fallback, extent tolerance) and the inverse iteration, then Albers on GRS80. Maximum difference from
  pyproj: 7e-8 m over 2M random CONUS points, 4.5e-8 m on the data.

### 2026-10-05 22:50 – 2026-10-06 02:20 — The landscape field (Entropy3D) and the place pathway
- **Absolute place loses.** Earth4D (latitude, longitude, elevation hash grid, 12 levels from 207 km to 249 m, own
  decoder and Brownian-motion place vectors, learned probing) on the corrected base (target-group + continental
  background + learned calibration), 1,000 steps, unclipped dev/test/> 10 km/AIM/FIA: 0.9258/0.9380/0.8917/0.9099/
  0.9269 vs base 0.9417/0.9517/0.9103/0.9342/0.9414, worse with steps: with per-species place vectors a fine absolute
  grid memorizes each species' record cells. Coarse absolute place (finest 10.8 km) and latitude/longitude Fourier
  features (8 octaves) also lost on the earlier base.
- **Entropy3D field** (DeepEarth's Entropy4D perceptive fields in their static, purely spatial case): a centre field
  and 8 rings at learnable radii (initially 0.24–31 km) read at A angles on a 2× pyramid of the 240 m CONUS field
  (12 channels: fine terrain and climate) with continuous level blending, so the extents are differentiable;
  aggregation per ring and channel by masked circular harmonics of (value − centre value), orders 0 to 3 (radial
  profile, north-aligned cos/sin pairs and their rotation-invariant magnitudes) and the missing share; fusion
  self-attention across the tokens; attention pooling; G(C(x)) added to the environment features. Checks: at Mt
  Whitney the 15 km ring's elevation first harmonic points uphill west (264°) and Owens Valley lies east; the radii
  receive gradients. CUDA kernel (one warp per location and ring, lanes = angles, analytic radius gradient) equal to
  the PyTorch path to 3.6e-7 (outputs) and 4 digits (radius gradients).
- First evaluation (250 steps, unclipped): 0.9431/0.9527/0.9145/0.9392/0.9411 vs base 0.9410/0.9514/0.9111/0.9363/
  0.9415. Over 1,000 steps (17-channel field: + SoilGrids and distance to the coast) dev +0.003, test +0.0024,
  > 10 km +0.0048, AIM +0.003, FIA −0.0013; the 17-channel field equals the 12-channel one at 250 steps, so 12
  channels are kept; 8 angles (exact to order 3) replace 32.
- **SINR's place network** as its own pathway with Brownian-motion place vectors (relatives share range geometry):
  untuned 0.9423/0.9517/0.9115/0.9357/0.9419 vs 0.9417/0.9517/0.9103/0.9342/0.9414 (small gains on four columns).
  Uniform pseudo-absences over US land (SINR's random negatives, 1.98M cells) instead of the effort-weighted
  continental background hurt transfer (> 10 km 0.9014): in a niche model uniform negatives penalize suitable,
  unrecorded places.
- **Field against SINR env** (17-channel field run `arm3_field_s0`, identical tests, share ours higher): pooled
  0.9475/0.9485 (57%; the base: 53%), VegBank 53%, VegBank > 10 km 46%, AIM 70%, FIA 54%.
- **Decision (02:20)**: field and place together (run `arm4_both12_s0`, 1,000 steps: 0.9450/0.9537/0.9165/0.9401/
  0.9401; field alone 0.9445/0.9546/0.9156/0.9346/0.9402 at 750). AIM varies by ±0.003 between evaluations of one run,
  so the head-to-head decides: with place the model wins every column beyond 10 km and AIM against SINR env and loses
  FIA over all plots. Place is kept.

### 2026-10-06 02:45 — Shoreline locations take the nearest climate
- Root cause of SINR's remaining lead: per test (field model vs SINR env, plots > 10 km), 225 of 1,545 tests (15%)
  carry 82% of the deficit, AUC 0.004–0.5, almost all shoreline species (mangroves, dune and salt-marsh plants).
  WorldClim 2.1 at 30″ has no value in pixels whose centre is sea or lake: (1) the per-species preparation
  (`pipeline.run_species`, step 6a) dropped drawn presences without climate (*Uniola paniculata* 224 of 683);
  (2) 0.76% of VegBank and 0.09% of FIA plots have no climate (48% of *Uniola paniculata*'s presence plots, 53% of
  *Croton punctatus*'), so a model reading the stack could not score them; (3) land cells without climate are blank
  on the maps. SINR reads a 5′ raster bilinearly and always has a value.
- Rule adopted (`climate_fill.py`): a location without climate takes all 20 WorldClim bands of the nearest pixel that
  has climate — great-circle distance to the 30″ pixel centre for points, exact Euclidean distance on the equal-area
  240 m grid for map cells — if within 5 km; otherwise it stays missing. Soil and terrain keep their own missing
  flags. The batched fill equals its per-point definition on all 693 plots. VegBank: 408 of 408 filled (median 0.6
  km, max 3.1 km); FIA 282 of 285.
- Records (`shoreline.py`): replaying the preparation's seeded draw (`occurrences_clean.sample(min(n, 5000),
  random_state=0)`) restores exactly the dropped presences (*Uniola paniculata* 224, *Croton punctatus* 178, *Abies
  concolor* 0): 40,961 restored presences for the 2,661-species benchmark, 88,042 for the CONUS species (19 more lie
  beyond 5 km and stay out). Proximity masks of the evaluation (near 1 km, far 10 km) count the restored records too.
- Map grid (03:10): the strict US mask follows state boundaries ~3 nautical miles offshore, so most of its 1.07M
  "land cells without climate" were coastal sea. The fill is restricted to cells NALCMS 2020 classifies as under half
  water: 169,719 land cells (~9,800 km²) lacked climate, 169,454 are filled (median 0.24 km).
- Same model, plots filled (`arm4_both12_s0` vs SINR env, share ours higher): pooled 0.9534/0.9485 (60.7%,
  p = 7e-35), > 10 km 54.7%, VegBank 58.1%, VegBank > 10 km 51.1%, AIM 71.5%, FIA 55.0%.

### 2026-10-06 04:05 — Two-stage training: the representation, then every species to convergence
- Profile (20 steps per configuration, benchmark): a step with every pathway costs 8.55 s, without the field 186 ms,
  the environment alone 84 ms: the field is ~97% of the cost (12 µs per row). Ring tokens are now built for all rings
  at once (identical to the per-ring construction, max difference 0).
- With target-group background every training row a step reads is a record, so with the shared networks fixed their
  features are computed once (`cache_shared.py`; benchmark 6.45M record rows, 6.1 GB) and the species parameters are
  fitted on them: `frz_heads_s0`, every species parameter re-learned from scratch on the frozen `arm4_both12_s0`
  representation, 6,000 steps at 16–18 ms per step (150 s). Unclipped 0.9503/0.9621/0.9279/0.9389/0.9446; against
  SINR env pooled 0.9547/0.9485, ours higher 63.2% (p = 6e-46), > 10 km 55.8%, VegBank 61.3%, VegBank > 10 km
  52.8%, AIM 72.4%, FIA 56.4% — above the jointly trained model on every column (60.7/54.7/58.1/51.1/71.5/55.0):
  1,000 joint steps had left the species parameters under-trained.
- Prior strength (04:00–04:45): AdamW's decay is decoupled (a parameter shrinks by lr × decay per step), so decays of
  1e-4 to 1e-3 on the species terms at lr 1e-3 had moved them by ≤ 0.6% over 6,000 steps and 1e-2 on the branch
  vectors by ~6%: the prior had been nearly inert. With all records, species-term decays 0.1/1/10 and branch decay
  0.1 are level with the default (data dominate: benchmark species have ≥ 20 presence plots and thousands of
  records). Data-poor test (half of the evaluated species capped at 5 presences; mean of dev and test unclipped AUC
  of the capped species): branch decay 0.01 0.9133, 0.1 0.9142, 1 0.9153, 10 0.9108; uncapped species unchanged
  (0.9723–0.9724). Adopted: branch decay 1. 12,000 steps (0.9505/0.9618/0.9273/0.9387/0.9438) and 2,048 background
  points did not gain; 6,000 steps and 1,024 are kept.

### 2026-10-06 06:25 — Representation decision
- After the species stage (all plots vs SINR env, identical tests; share ours higher / pooled median): `arm4_both12_s0`
  representation, branch decay 0.01: 63.2% / 0.9547; branch decay 1 with restored shoreline records: 63.2% / 0.9547
  (VegBank 61.2%, AIM 73.3%, FIA 55.9%); representation trained with the restored records (`shore_syn_s0`): 61.3% /
  0.9543; 22-channel field with NALCMS land cover: 62.2% / 0.9554. Draw → the simplest: the `arm4_both12_s0`
  representation (12-channel field + SINR place), species stage with branch decay 1, restored shoreline records and
  filled plots, 6,000 steps. Wider rings (to ~120 km, `shore_wide_s0` 0.9499/0.9608/0.9257/0.9382/0.9461 at 1,000
  steps) were a draw; focusing (each ring's radius adapted per location, `shore_focus2_s0`) was a draw at 250 steps
  (0.9480/0.9604/0.9276/0.9375/0.9447) at 10 s per step. Neither is in the model.

### 2026-10-06 06:45 — National model with the landscape field, place and learned calibration
- `nat_arm4_s2`: the `arm4_both12_s0` representation fixed, species stage on all 16,448 CONUS species (15.37M record
  rows cached), 6,000 steps, 142 s. Unclipped dev/test/> 10 km/AIM/FIA 0.9565/0.9658/0.9354/0.9344/0.9510, against
  0.9434/0.9556/0.9162/0.9209/0.9391 recorded for the environment model `conus_final`.
- Against every competitor, identical tests, unclipped, proximity masks extended to the national restored records:
  SINR env pooled 0.9568 vs 0.9512, ours higher 63.1% (p = 8e-93, n = 4,281; the environment model: 44%), > 10 km
  56.6% (p = 3e-27); SINR coordinates 73.0%; SINR distilled 75.8%; iNaturalist ranges 98.3%; iNaturalist geomodel
  86.4%; BIEN 98.3%; Daru 93.4% (median +0.048). AIM vs SINR env 70.7% (> 10 km 65.2%), FIA 59.4% (> 10 km 54.4%,
  p = 0.1: the one draw).
- Taxon concepts: where GBIF usage of a name clearly spans segregate species (n_s ≥ 4 n_X and n_s ≥ 20; e.g.
  *Pinus ponderosa* ⊃ *P. brachyptera*, *P. scopulorum*), scoring the broad taxon (log-sum-exp of member
  intensities, 74 tests) gives 62.8% (strict names 63.1%) and > 10 km 56.5%: concepts do not drive the result;
  strict-name results are primary.
- Map store `cards_sota_klt` (d = 512: environment and field 256 | place 256; 494 species without records with their
  place vectors and penalties from the joining node; 98 calibration extensions) building at the time of the port
  (07:12: the fused attention launches one GPU block per row and head, so the field splits itself into 8,192-row
  pieces; GPU AUC blocks sized to ~64 MB per float64 temporary).

### 2026-10-06 — The full model ported into this package
- Modules: `ranges/joint/field.py` (field pyramid, ring harmonic sums with the CUDA kernel `kernels/ring_harmonics.cu`
  and its PyTorch reference, the Entropy3D module), `place.py`, `geodesy.py`, `climate_fill.py`, `shoreline.py`,
  `cache.py`; `model.py` (optional place, field and learned penalty, rebuilt from any checkpoint's state dict),
  `data.py` (restored records, field positions, row ecoregions, target-group snapping, filled plots), `train.py`
  (one trainer for every stage: `init`, `shared_from` + `freeze_shared` on the cache; exact batched AUC; clipped and
  unclipped tables), `store.py` / `reader.py` (stored features [F | P], penalty, served scores, grid climate fill),
  `zero_shot.py` (place vectors and penalty from the joining node), `prepare.snap_to_records`. Scripts
  `national_snap.py`, `national_field.py`, `build_shoreline.py`, `build_climate_fill.py`, `national_cache.py`,
  `national_train.py --stage`. Configuration: `configs/conus.json` `joint.field`, `joint.stages`, `store.model`,
  `shoreline`, `geo.proj_grids`. Only the final model is ported: the absolute Earth4D place encodings, the community
  and relative-record channels, the sampled-token field, the effort field, uniform pseudo-absences, focusing and the
  profiling options were left out.
- Kept from the research code where it is a measured rule: weight decay applies to every network parameter (ring
  radii included); the species stage trains c_0 as well as the species' parameters; checkpoints are selected on the
  clipped dev median (`dev_joint_paired`), and both rankings are reported.
- Checks: the trained national checkpoint (`nat_arm4_s2`) loads into the package model by its state dict alone;
  scored by the package with the CUDA kernel at all 53,797 VegBank plots for its 3,605 evaluated species, its scores
  equal the research run's saved scores to their float16 precision (largest difference 0.018 on scores up to 44,
  median 0.004, per-species Spearman median 0.999998) and the unclipped AUCs agree (median 0.96306 vs 0.96303, largest
  per-species difference 4e-4); on the CPU (PyTorch path of the field, bfloat16 on the CPU) 4,000 plots agree to a
  median of 0.015 and the AUC medians to 1e-4. The plots' inputs are kept in float32 as in the research cache (with
  float16 inputs the scores moved by a median 0.009). The CUDA kernel equals the PyTorch reference to
  7e-6 relative (sums) and 3e-5 (radius gradients); the geodesy equals pyproj to 6e-8 m; the batched shoreline fill
  equals the research files exactly (VegBank 408, FIA 282 filled plots); the restored benchmark records are
  byte-identical to the research file (40,961; the research log's 40,962 was a miscount); grid positions of 100,000
  benchmark rows equal the research positions in float32. Tests: `tests/test_field.py`, `test_geodesy.py`,
  `test_climate_fill.py`, `test_joint_stages.py` (synthetic end to end: environment model, representation, cache,
  species stage, store).

### 2026-10-06 08:20 — Map store of the full model evaluated; it is the published product
- Store `cards_sota_klt` (`nat_arm4_s2`; klt, step 3.2, d = 512): 16,942 species (494 inferred from relatives, with
  place vectors and penalties from their joining node; 98 calibration areas extended), 6.95 GB (field 6.75 GB,
  species table 161.7 MB, validity mask 32.9 MB). CPU decode of one species: 256² window median 76 ms (90th
  percentile 98 ms), 512² 163 ms (210 ms).
- Fidelity: per test (species × plot source), AUC of the stored maps vs the full model's own scores, 5,873 tests:
  median absolute difference 0.00056, 99th percentile 0.010; medians 0.9551 vs 0.9559.
- As served (outside a species' calibration ecoregions its score lowered by its learned penalty; every plot with
  climate), `eval_national_maps.py`: VegBank 3,607 species median AUC 0.962, BLM AIM 2,009 species 0.934, FIA 259
  species 0.950 (with the hard calibration rule 0.956 / 0.930 / 0.948). Against Daru's published maps 0.960 vs 0.910
  (141 species, ours better 90%), 0.932 vs 0.853 (104, 96%), 0.973 vs 0.916 (14, 100%); against the per-species MaxEnt
  maps 0.956 vs 0.938 (1,547, 76%), 0.938 vs 0.906 (480, 85%), 0.944 vs 0.935 (224, 73%).
- Against every competitor (benchmark `rescore.py --cards`, the store's penalty applied as rendered; paired tests,
  share ours higher, Wilcoxon p): SINR env (retrained) pooled 0.9563 vs 0.9512, 62.7% of 4,282 tests (p = 8e-85);
  > 10 km from training presences 0.9290 vs 0.9240, 56.1% of 4,132 (p = 8e-24). By source: VegBank 59.4% (> 10 km
  51.9%, p = 0.001), AIM 70.9% (> 10 km 65.1%), FIA 56.6% (p = 0.01; > 10 km 53.2%, p = 0.41: a draw). SINR
  coordinates-only 73.0%; SINR released distilled 75.0%; iNaturalist range maps 98.2% (5,378 tests); iNaturalist
  geomodel 88.0% (125 tests); BIEN 98.3% (1,854 AIM tests; its VegBank and FIA comparisons are not independent, as BIEN
  holds those plots); Daru 93.1% (0.9507 vs 0.8962, 259 tests). The GBIF-leakage exclusion (leak1km) moves no pooled
  share by more than 0.4 points.
- From the model's own scores the same comparison gave 63.1% (2026-10-06 06:45): storing moves it by 0.4 points. The
  environment model's store `cards_conus_klt` had won 44% of the same tests.
- Decision: `cards_sota_klt` replaces `cards_conus_klt` as the published maps (README, docs/joint_model.md section 8).
  `cards_conus_klt` and its numbers remain where the environment-only model is described.
