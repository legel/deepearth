# Validation

Every map is checked against sources that played no part in fitting it. Each source tests something different and
has its own blind spots, so results are reported per source and never pooled.

## Sources

| Source | What it tests | Coverage | Filtering | Weak points | Script |
|---|---|---|---|---|---|
| Daru (2024) Dryad rasters, ~18 km | agreement with the published model (κ, Jaccard, suitability Spearman, calibration footprint IoU) | 114 priority-list natives with a Dryad map | compared on Daru's grid after aggregating our 240 m maps | agreement is not accuracy: a faithful replica should agree, an improved model need not | `render_results.py`, `benchmark.py` |
| WCVP native regions (Kew) | share of the predicted range inside the species' native TDWG level-3 regions | every species | — | coarse (states, provinces); a range can sit inside its regions and still be wrong | `benchmark.wcvp_consistency` |
| BLM AIM plots | presence/absence at precise coordinates | arid and semi-arid West (BLM land), 72,861 plot visits | species named on the plot = presence; every other plot = absence | West only; shrubland and grassland | `fetch_blm_aim.py`, `validate.py` |
| USFS FIA plots | tree presence/absence | national, 797,418 plots, 1,288 tree species | live tree of the species on any inventory of the plot = presence | trees only; public coordinates perturbed up to ~1.6 km, so tests range placement rather than 240 m detail | `fetch_fia.py`, `validate.fia_truth` |
| VegBank (ESA) | presence/absence with full species lists, incl. the East | 53,797 usable plots (13,033 east of 100°W) | exact public coordinates only (confidentiality 0); plot counts as an absence only if it lists ≥ 10 taxa; names resolved through WCVP; repeat visits merged | project-dependent sampling (parks, research sites) | `fetch_vegbank.py`, `validate_vegbank.py` |
| Hawaii plots: NPS Vegetation Mapping Inventory (Hawaiʻi Volcanoes, Haleakalā, Kalaupapa, 1997–2011) and NPS Pacific Island Network plant-community monitoring (2010–2022); OpenNahele (Craven et al. 2018) | presence/absence in Hawaii | 1,332 NPS plots with complete species lists (120 Hawaiian natives with ≥ 20 presence plots); 146 OpenNahele plots (woody stems ≥ 5 cm only) | names resolved through WCVP (synonym and subspecies forms tried in turn; 11.6% of name records, mostly genus-only, unresolved); records outside the plot dropped; OpenNahele studies already in the NPS or FIA data excluded | national-park clusters (Hawaiʻi Island, Maui, Molokaʻi), plus OpenNahele on Oʻahu, Kauaʻi, Lānaʻi; OpenNahele absences hold for woody species only (`hawaii_woody`) | `build_hawaii_plots.py`, `validate.PlotTruth(hawaii_dir=...)` |

## Regions

Scores are computed per region, on plots inside that region's grid (`validate.DOMAINS`, `validate.in_domain`); a plot
outside the product's grid is never scored (a CONUS map cannot be credited or blamed in Alaska).

| Region | BLM AIM plots | VegBank visits (public coordinates) | FIA plots |
|---|---|---|---|
| CONUS | 71,783 | 120,575 | 786,869 |
| Alaska (incl. Aleutians) | 366 | 3,697 | 6,816 |
| Hawaii | — (NPS: 1,332 plots; OpenNahele: 146) | — | 583 (trees only) |

Hawaii: the NPS plots (complete species lists) fill the gap left by AIM and VegBank; all sources are public and need
no credentials (NEON's Pacific Tropical domain would need an API token).

## Metrics

* **AUC** of suitability at presence vs absence plots. Plots outside a model's calibration area score 0
  (unsuitable), as a user of the map would read them.
* **TSS** (sensitivity + specificity − 1) of each binary map: P5 (5th-percentile training presence) and the
  replicate-majority equal-sensitivity-specificity rule Daru's maps use.
* Over **all plots** and inside the **area both models cover** (Daru's calibration area and ours): the second
  removes the easy absences far outside the range.
* **Daru-style metrics** (his Fig. S2: AUC, median TSS over thresholds, Boyce index) from each replicate's test
  points, against the fitting background and against the bias-weighted set (`evaluate.py`; provenance D10).

## Resolution

`resolution_test.py` / `resolution_study.py` score the same 240 m suitability map at the plots after block-averaging
it to 0.48–18 km. A map whose skill holds when coarsened carries no information at the finer scale. With WorldClim
~1 km predictors the maps hold no measurable information below ~1 km (29 species, AIM; ledger L11). Adding SoilGrids
soil improves accuracy but not sub-kilometre skill (ledger L10).

## Exactness checks (not accuracy)

* GPU MaxEnt evaluation equals maxent.jar (`tests/test_maxent_parity.py`).
* Range cards decode byte-identically to rendered maps (`codec.verify`).
* Render stacks hold no placeholder no-data values (`tests/test_stack_nodata.py`).
