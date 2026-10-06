# Native plant habitat

Species distribution models for the 16,942 vascular plant species native to the contiguous United States, at 240 m
resolution. One neural model is fitted jointly to the 16,448 species with usable GBIF occurrence records (179 million
presence and sampling-effort-weighted background points), with the MaxEnt point-process likelihood, WorldClim 2.1
climate and SoilGrids 2.0 soil predictors, and a Brownian-motion prior along a dated phylogeny that shares niche
information among related species; the 494 species without records are inferred from their relatives. The pipeline
reproduces and extends the global native-range method of Daru (2024, *PNAS*,
[doi:10.1073/pnas.2319989121](https://doi.org/10.1073/pnas.2319989121)). On independent vegetation plots the maps
reach a median AUC of 0.951 (VegBank), 0.922 (BLM AIM) and 0.942 (USFS FIA), and score above Daru's published maps for
84 %, 86 % and 100 % of the species both cover. All maps are stored in 9.1 GB and decoded on a CPU.

Method: [docs/joint_model.md](docs/joint_model.md). Every decision, dated, with the measurement behind it:
[docs/scientific_provenance.md](docs/scientific_provenance.md).

## Equations

Data and calibration (per species; [`pipeline.py` `run_species`](ranges/pipeline.py#L218)):

| step | rule | code |
|---|---|---|
| names | every GBIF name resolved through WCVP: Accepted > Synonym > Orthographic > Illegitimate/Invalid > Unplaced, misapplied names never followed; a species takes every record whose name resolves to it | [`names.py` `resolve`](ranges/names.py#L37), [`gbif_names_by_accepted`](ranges/names.py#L61) |
| cleaning | CoordinateCleaner 3.0.1 (capitals, centroids, equal, GBIF, institutions within 100 m, outliers, seas, zeros, duplicates); on the native records alone when the outlier test removes the native cluster (provenance D11) | [`clean_coordinates.R`](ranges/r/clean_coordinates.R), [`clean_species`](ranges/pipeline.py#L205) |
| native filter | records inside the species' WCVP native level-3 areas (WGSRPD) | [`occurrences.py` `native_filter`](ranges/occurrences.py#L65) |
| thinning | one record per 30″ (~1 km) predictor cell at ≥ 5 localities; 1 to 4 localities topped up to 50 points within 30 km (provenance D3) | [`grid_thin`](ranges/occurrences.py#L71) |
| calibration area | $C_s$ = the RESOLVE ecoregions containing the species' cleaned records, lakes excluded (provenance D8) | [`calibration_cells`](ranges/pipeline.py#L178) |
| sampling effort | $\hat\lambda(x) \propto \sum_i n_i\, \phi\!\left(\frac{x_1 - u_{i1}}{h_1}\right) \phi\!\left(\frac{x_2 - u_{i2}}{h_2}\right)$, $h_k = 1.06 \min(\sigma_k, \mathrm{IQR}_k / 1.34)\, n^{-1/5}$, over 524.6 million GBIF vascular-plant records in ~9 km cells, Behrmann 10 km grid, scaled to 0..1 (spatialEco `sp.kde`) | [`background.py` `silverman_bandwidth`](ranges/background.py#L32), [`bias_grid`](ranges/background.py#L41) |
| background | 10,000 ~1 km cells of $C_s$ drawn without replacement with probability $\propto \hat\lambda$ (all cells if fewer) | [`sample_background_index`](ranges/background.py#L72) |
| predictors | WorldClim 2.1 bio1 to bio19 and elevation at 30″, screened per species by stepwise variance inflation, $\mathrm{VIF}_j = [R^{-1}]_{jj} < 5$ (`usdm::vifstep`) | [`modelling.py` `vifstep`](ranges/modelling.py#L40) |

The per-species model (MaxEnt, as fitted by maxent.jar 3.4.4 or its exact PyTorch reimplementation,
[docs/maxent_torch.md](docs/maxent_torch.md)):

| process | equation | code |
|---|---|---|
| MaxEnt | $S(x) = \sum_j \lambda_j f_j(x)$ over linear, threshold and hinge features of the clamped predictors; $\mathrm{raw} = e^{S(x) - S_0} / Z$, $\mathrm{cloglog} = 1 - \exp(-e^{H} \mathrm{raw})$; $\beta \in \{2, 5, 10, 15, 20\}$ by 5-fold CV, then 5 replicates | [`maxent.py` `linear_predictor`](ranges/maxent.py#L122), [`cloglog`](ranges/maxent.py#L158), [`modelling.py` `fit_species`](ranges/modelling.py#L142) |
| maps | suitability $1 + \mathrm{round}(254\, \mathrm{median}_r\, p_r(x))$; range: the replicates' median at least the median of their 5th-percentile training-presence thresholds (P5), or Daru's per-replicate majority vote | [`project.py` `compute_maps`](ranges/project.py#L44) |

The joint model ([`ranges/joint`](ranges/joint)):

| process | equation | code |
|---|---|---|
| inputs | $z = \mathrm{clip}\!\left(\frac{t(X) - \mu}{\sigma}, -6, 6\right)$ over the 20 WorldClim layers and SoilGrids 2.0 topsoil pH, clay, sand and organic carbon (7.5″), $t = \log(1 + \cdot)$ for precipitation and carbon, $\mu, \sigma$ over the background; a missing soil value becomes 0 and sets one flag input | [`data.py` `Standardizer`](ranges/joint/data.py#L69) |
| score | $f_s(x) = \langle h(x), w_s \rangle + b_s$, $h = \mathrm{trunk}(\mathrm{env}(z))$ of width 256, its last LayerNorm without scale fixing $\lVert h \rVert = 16$ | [`model.py` `JointRangeModel`](ranges/joint/model.py#L46), [`forward`](ranges/joint/model.py#L75) |
| phylogenetic prior | $w_s = \sum_{e \in \mathrm{root} \to s} \sqrt{l_e / \bar l}\; z_e + u_s$, weight decay 0.01 on the branch vectors $z_e$ (a Brownian random walk along the dated tree), 0.0001 on the species terms $u_s$ | [`tree.py` `path_matrix`](ranges/joint/tree.py#L121), [`species_vectors`](ranges/joint/model.py#L67) |
| objective | $L_s = -\frac{1}{\lvert P_s \rvert}\sum_{p \in P_s} f_s(p) + \log \frac{1}{\lvert P_s \cup B_s \rvert}\sum_{a \in P_s \cup B_s} e^{f_s(a)}$ (MaxEnt's point-process likelihood, presences in the normalizer), 256 species × (64 presences + 1,024 background points) a step, AdamW, 21,000 steps | [`train.py` `train`](ranges/joint/train.py#L156) |
| species without records | $w_m = \sum_{e \in \mathrm{root} \to x} \sqrt{l_e / \bar l}\; z_e$, $x$ the node where $m$ joins its closest trained relatives $R_m$; $b_m = \overline{b}_{R_m}$; $C_m$ = ecoregions with ≥ 10 % of their area in $m$'s WCVP native areas | [`zero_shot.py` `infer_species`](ranges/joint/zero_shot.py#L68), [`l3_ecoregions`](ranges/joint/zero_shot.py#L36) |
| region | training points in Alaska (with the Aleutians, 0.05° buffer) and Hawaii removed, points in native ranges abroad kept; a CONUS native whose $C_s$ misses the CONUS grid gains the ecoregions of its native areas there | [`scope.py` `restrict`](ranges/joint/scope.py#L57), [`extend_calibration`](ranges/joint/scope.py#L138) |
| transform coding | $y(x) = (h(x) - \mu) M^{1/2} V$, $r_s = V^\top M^{-1/2} w_s$, $o_s = \langle \mu, w_s \rangle + b_s$, $M = W^\top W$, $V$ the eigenvectors of the feature covariance in that metric: $f_s = \langle y, r_s \rangle + o_s$ exactly and $\sum_s r_s r_s^\top = I$ | [`store.py` `klt`](ranges/joint/store.py#L100) |
| stored field | $q(x) = \mathrm{round}(y(x) / \Delta)$, $\Delta = 3.2$; the total squared score error of a cell over all species is $\lVert y - \Delta q \rVert^2 \le 256\, (\Delta/2)^2$; 128 × 128-cell tiles, int16/int8 byte planes, zstd | [`write_field`](ranges/joint/store.py#L179), [`KltWriter`](ranges/joint/store.py#L135) |
| suitability | $1 + \#\{k : Q_{s,k} < f_s(x)\}$ (1..255), $Q_s$ 254 quantiles of $f_s$ over the species' background; 0 outside $C_s$ or without climate | [`reader.py` `suitability_of`](ranges/joint/reader.py#L239) |
| range | $f_s(x) \ge \mathrm{P5}_s$ inside $C_s$, $\mathrm{P5}_s$ the 5th percentile of $f_s$ over its training presences | [`in_range`](ranges/joint/reader.py#L254) |

## Departures from Daru (2024)

| | Daru (2024) | here |
|---|---|---|
| presences | 500 lattice points over an alpha hull | the cleaned, thinned occurrence records |
| predictors | WorldClim 2.1 at ~9 km | WorldClim 2.1 at ~1 km, SoilGrids 2.0 at ~230 m |
| model | one MaxEnt per species | one joint model, MaxEnt's likelihood, phylogenetic prior |
| species without records | not mapped | inferred from relatives on the dated tree |
| maps | ~18 km | 240 m |

Names, cleaning, the native filter, thinning, calibration areas and the sampling-effort background follow Daru step by
step; a faithful replica of his per-species pipeline (`scripts/run_fit.py --modes daru`) is kept as the reference.

## Results

Median AUC of the stored maps on plots withheld from training, for species with ≥ 20 presence plots; Daru's maps and
per-species MaxEnt scored on the same plots ([`national_store_eval.py`](scripts/national_store_eval.py) `plots`).
VegBank: the Ecological Society of America's vegetation plot archive; BLM AIM: Bureau of Land Management monitoring
plots (arid West); USFS FIA: Forest Service inventory plots (trees).

| plots | species | joint model | vs Daru (2024): joint / Daru (species; share better) | vs per-species MaxEnt: joint / MaxEnt (species; share better) |
|---|---|---|---|---|
| VegBank | 3,607 | 0.951 | 0.948 / 0.910 (141; 84 %) | 0.942 / 0.938 (1,547; 53 %) |
| BLM AIM | 2,009 | 0.922 | 0.921 / 0.853 (104; 86 %) | 0.915 / 0.906 (480; 64 %) |
| USFS FIA | 259 | 0.942 | 0.936 / 0.916 (14; 100 %) | 0.933 / 0.935 (224; 46 %) |

The map store holds 16,942 species in 9.06 GB (an 8.88 GB field and a 0.14 GB species table); storing changes the
median VegBank AUC by less than 0.0001 (0.9509 before and after, 3,605 species). Training takes 43 minutes on one
RTX 3090.

## Reading the maps in Python

The store holds every species: `store.json`, one `field_conus.zst` with its tile index and validity mask, and
`species.npz` (codes, offsets, quantiles, P5, calibration ecoregions, an `inferred` flag for species mapped from
relatives). Reading needs numpy and zstandard (rasterio and pyproj for the calibration mask and coordinates), and
the CONUS grid's ecoregion layer `ecoregion_id_conus240.tif` ([`reader.py`](ranges/joint/reader.py)).

```python
import sys; sys.path.insert(0, "models/habitat/plant")
from ranges.joint.reader import Store

st = Store("cards_conus_klt", grids={"conus": "work/conus240"})
s = st.index("Quercus lobata")                                   # valley oak
f = st.scores("conus", 6000, 6512, 1500, 2012, [s])[0]           # float32 scores f_s, a 512 x 512-cell window
u = st.suitability("conus", 6000, 6512, 1500, 2012, s)           # uint8: 1..255 inside the calibration area, else 0
r = st.in_range("conus", 6000, 6512, 1500, 2012, s)              # bool: the binary range
p = st.at("conus", lon=[-122.27, -121.5], lat=[37.87, 38.6], species=s)   # p["score"], p["suitability"], p["range"]
```

Rows and columns index the 240 m CONUS Albers grid (EPSG:5070, 13,053 × 20,149 cells, upper-left corner
x = −2,493,045 m, y = 3,310,005 m); `st.rowcol` converts longitude and latitude. A species row with
`st.T["inferred"][s]` true was mapped from its relatives. A 512 × 512-cell window of one species decodes in 0.2 to
0.6 s on 8 CPU threads.

## Reproducing the maps

Everything lives under one data root (`DEEPEARTH_HABITAT_DATA`, default `models/habitat/plant/data`); paths and
settings of the published run are in [configs/conus.json](configs/conus.json), and every script takes `--config`.
Steps 1 to 5 run on CPUs (R for step 4), steps 6 to 9 on one CUDA GPU (24 GB).

```bash
cd models/habitat/plant
python3 -m pip install --user -r requirements.txt
conda create -n daru --file r_environment.txt && conda run -n daru Rscript -e 'install.packages("rangeBuilder")'

# 1. public sources and the GBIF occurrence downloads
for s in wcvp geo worldclim maxent trees gbif effort soil validation; do scripts/fetch_sources.sh $s; done
python3 scripts/extract_daru_maps.py DRYAD_DATA.zip            # Daru's maps, downloaded by hand from Dryad

# 2. species, names and records
python3 scripts/build_us_natives_table.py
python3 scripts/match_gbif_keys_all.py
python3 scripts/inventory_us_natives.py
python3 scripts/build_name_reassignment.py                       # only with a listed-species table (ledger L17)

# 3. predictor grids, the sampling-effort density and dispersal rates
python3 -m ranges.predictors data/raw/worldclim data/work/global30s/worldclim30s_stack.f32
scripts/warp_conus_predictors.sh && python3 scripts/build_conus_stack.py && python3 scripts/build_conus_layers.py
python3 scripts/build_soil_stacks.py
python3 scripts/build_bias_grid.py
scripts/fit_dispersal.sh

# 4. per-species data: cleaning, native filter, thinning, calibration area, background (CPU, R)
CC_SEAS_REF=data/raw/geo/ne_50m_land/ne_50m_land.shp python3 scripts/run_fit.py --prepare-only --group 8

# 5. the dated tree, the joint model's training points and plots, restricted to CONUS
python3 scripts/build_tree.py
python3 scripts/national_prepare.py && python3 scripts/national_plots.py && python3 scripts/national_scope.py

# 6. train (one RTX 3090: 43 minutes)
python3 scripts/national_train.py

# 7. the map store, every species of CONUS including those without records
python3 scripts/national_store.py build

# 8. evaluate as stored: independent plots, the cost of storing, CPU decode time
python3 scripts/national_store_eval.py plots data/work/deepearth/cards_conus_klt --out data/work/deepearth/eval
python3 scripts/national_store_eval.py fidelity data/work/deepearth/cards_conus_klt --out data/work/deepearth/eval
python3 scripts/national_store_eval.py bench data/work/deepearth/cards_conus_klt

# 9. optional: the per-species MaxEnt reference maps (GPU fit and render) and their evaluation
python3 scripts/run_fit.py --modes daru,occurrences   # or national_run.py on prepared species
python3 scripts/validate_vegbank.py && python3 scripts/national_eval.py
```

Departures of the published run from a fresh run: 2,663 species took their per-species data from an earlier full
fit on the first GBIF download; records of 2,672 horticulturally listed species followed the name rule of ledger
L17, whose list is not public; and the paired MaxEnt baseline used for checkpoint selection came from range cards
decoded on 2026-10-04 (docs/scientific_provenance.md). The dated tree (`build_tree.py`) and the sampling-effort
density (`build_bias_grid.py`) reproduce the published ones byte for byte.

Data layout under the root: `raw/` (wcvp, geo, worldclim, trees, gbif_*, validation, dem_glo90), `tools/maxent.jar`,
`work/` (global30s and conus240 predictor grids, soil, sbm, bias grid, species tables, `natives/` and
`national/prepared/` per-species products, `deepearth/` tree, training data, runs and the map store),
`daru_ref_all/` (Daru's maps).

## Sources

| data | provider | licence |
|---|---|---|
| occurrence records | GBIF.org, doi:10.15468/dl.wwa829, doi:10.15468/dl.pkunks, doi:10.15468/dl.52fhnr | CC BY-NC 4.0 (the most restrictive record licence) |
| names, native areas | World Checklist of Vascular Plants, Royal Botanic Gardens, Kew (2026-06-04) | CC BY 4.0 |
| botanical regions | WGSRPD level 3, TDWG | CC BY 4.0 |
| ecoregions | RESOLVE Ecoregions 2017 (Dinerstein et al. 2017) | CC BY 4.0 |
| climate, elevation | WorldClim 2.1, 30″ (Fick and Hijmans 2017) | CC BY-SA 4.0 |
| soil | SoilGrids 2.0, ISRIC (Poggio et al. 2021) | CC BY 4.0 |
| land and lakes | Natural Earth 10 m, 50 m | public domain |
| dated phylogeny | Carruthers et al., OSF 9tbha | as published (OSF) |
| plots | VegBank (ESA), BLM AIM, USFS FIA DataMart | public |
| benchmark maps | Daru (2024), Dryad doi:10.5061/dryad.5x69p8d9w | CC0 |
| MaxEnt | maxent.jar 3.4.4 (Phillips et al.) | MIT |
| terrain (fine predictors only) | Copernicus GLO-90 DEM | Copernicus licence |

Maps derived from GBIF records carry the records' attribution and non-commercial terms.

## Tests

```bash
cd models/habitat/plant
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest     # tests needing data skip without it
DEEPEARTH_HABITAT_DATA=/path/to/data python3 -m pytest   # also against maxent.jar runs, the stores and the grids
```

## Documentation

- [docs/joint_model.md](docs/joint_model.md): the joint model and its map store, from first principles, with results.
- [docs/scientific_provenance.md](docs/scientific_provenance.md): every step, source, departure from Daru (2024) and
  decision, dated, with its measurement.
- [docs/maxent_torch.md](docs/maxent_torch.md), [docs/maxent_fitting_spec.md](docs/maxent_fitting_spec.md): the GPU
  reimplementation of maxent.jar's fitting and its validation.
- [docs/range_cards.md](docs/range_cards.md): per-species range cards.
- [docs/validation.md](docs/validation.md): the independent plot sources and their blind spots.
- [docs/pipeline_analysis.md](docs/pipeline_analysis.md), [docs/future_ledger.md](docs/future_ledger.md): what each
  step is for, and the opportunities beyond it.

## Credits

Builds on Daru, B. H. (2024), Predicting undetected native vascular plant diversity at a global scale, *PNAS*
121(34): e2319989121, whose published maps are the benchmark throughout. Developed by Ecological Intelligence, Inc.
and the Quantitative Ecosystem Dynamics Lab.

## License

MIT, via the repository root [`LICENSE`](../../../LICENSE).
