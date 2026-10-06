# The joint range model

One model maps the native ranges of the 16,448 plant species native to the contiguous United States (CONUS) that
have usable occurrence records, plus 494 species without usable records whose ranges are inferred from their
relatives: 16,942 maps on the 240 m CONUS grid, stored in one transform-coded field and decoded on an ordinary CPU.

Code: `ranges/joint/` (`prepare`, `scope`, `tree`, `model`, `data`, `train`, `zero_shot`, `store`, `reader`);
command-line entry points `scripts/national_prepare.py`, `national_plots.py`, `national_scope.py`,
`national_train.py`, `national_store.py`, `national_store_eval.py`; configuration `configs/conus.json`. The
measurements and decisions behind every choice are logged in `docs/scientific_provenance.md` (entries of 2026-10-04
and 2026-10-05). The model was developed on all 18,600 US natives (CONUS, Alaska and Hawaii; run `national_final`,
store `cards_national_klt`); on 2026-10-05 the product was restricted to CONUS and retrained with the same settings
(run `conus_final`, store `cards_conus_klt`). Numbers below are for the CONUS product unless they name another run.

## 1. What a range model estimates

The data are presence-only: places where someone recorded the species (herbarium sheets, observations; from
GBIF, cleaned and thinned as described in `docs/scientific_provenance.md`). Nobody recorded where the species is
absent. What such records can tell us is the *relative intensity* of records: how many more records of the species
we expect per square kilometre at a place than on average across the area considered. MaxEnt (Phillips et al.
2006), the method Daru (2024) used for every species, estimates exactly this; it is mathematically the
maximum-likelihood fit of an inhomogeneous Poisson point process, a model in which records fall independently with a
density that varies over space (Renner & Warton 2013).

Writing the log relative intensity of species s at location x as a score f_s(x), the likelihood of the records
P_s compared with the area as a whole is

    L_s = - mean over records p of f_s(p) + log mean over the area a of exp f_s(a).

The first term rewards high scores where the species was recorded; the second term, the normalizer, penalizes high
scores everywhere else. The area integral is approximated by a sample of *background points*.

Two choices define the area and the background:

* **Calibration area.** Each species is modelled over the whole RESOLVE ecoregions (Dinerstein et al. 2017) that
  its records occupy, as Daru's published maps do. Ecoregions are regions of similar climate, landforms and
  vegetation, typically 10,000-100,000 km² in size. Outside its calibration area a species' map is 0.
* **Effort-weighted background.** Records concentrate where people look (roads, towns, universities). The
  background points are therefore drawn inside the calibration area in proportion to the density of *all* plant
  records there, so the comparison becomes "where was this species recorded, relative to where anything was
  recorded", and sampling effort cancels.

The presences also enter the normalizer (as in maxent.jar, option `addsamplestobackground`). Then the normalizer
always contains the points whose scores the first term pushes up, so the objective is bounded below and a species
can never be "perfectly separated" by sending its scores to infinity.

## 2. One model for all species

A per-species MaxEnt sees only that species' records. A rare species with 5 records gets a poorly constrained map,
and a species with none gets no map. Yet species share most of what matters: they respond to the same climate and
soil gradients, and close relatives tend to have similar niches. The joint model fits all species together:

    f_s(x) = <h(x), w_s> + b_s.

* **h(x)**, a list of 256 numbers ("features") describing the environment at x, is computed by one neural network
  shared by every species. Its inputs (25) are the 19 WorldClim 2.1 bioclimatic variables and elevation at 30
  arc-seconds (about 1 km), four SoilGrids 2.0 topsoil (5-15 cm) properties at about 230 m (pH, clay, sand, organic
  carbon), and one flag that is 1 where SoilGrids has no value (water, rock, built land).
* **w_s**, a vector of 256 numbers per species, says how the species weighs each feature: its niche.
* **b_s** is the species' offset (its overall record density).

So a species' map is a weighted sum of the shared features. Learning the features from all 15.3 million presence
records lets a rare species reuse environmental distinctions that common species taught the network.

### Training points and the region

Each species contributes exactly what its own MaxEnt saw (`prepare.training_data`): its thinned presences and its
10,000 effort-weighted background points in its calibration ecoregions, written by the per-species pipeline
(`scripts/run_fit.py --prepare-only`). Every point gets the same 24 predictors. About 194 million points share 27.3
million distinct places (background points are ~1 km cell centres shared by many species), so each distinct
(7.5″, 30″) cell pair is sampled once. For the CONUS product (`scope.restrict`), the species native to CONUS are kept,
their training points inside Alaska (WGSRPD ASK and ALU, buffered by 0.05°) and Hawaii are removed, and their points
in native ranges abroad (Canada, Mexico, Eurasia) are kept, as Daru trains on whole native ranges: 16,448 species,
179.1 million points, 15.3 million presences.

### Inputs

Precipitation variables (bio12-14, bio16-19) and soil organic carbon are heavy-tailed, so they enter as
log(1 + value). Every input is then centred and scaled by its mean and standard deviation over the background
points, clipped to ±6 standard deviations (the network never sees values far outside the training range), and a
missing soil value becomes the mean (0) with the flag set (`data.Standardizer`; `norm.npz` in each run directory).

### Network

`env`: three linear layers of width 256, each of the first two followed by LayerNorm (rescaling the 256 numbers to
mean 0 and variance 1, with a learned scale and shift) and SiLU (the smooth activation x·sigmoid(x)). `trunk`:
LayerNorm, SiLU, two more linear layers with LayerNorm and SiLU between them, and a final LayerNorm *without*
learned scale or shift. That last step fixes the length of h(x) to sqrt(256) = 16 at every location, so a score is
at most |w_s| · 16 in size: no location can dominate a species' normalizer by extrapolating, and the prior on w_s
(next section) keeps |w_s| moderate. The network has 271,872 parameters.

## 3. The phylogenetic prior

Related species descend from common ancestors and inherited their niches from them, diverging since. Brownian
motion is the standard model of this: a trait drifts randomly along each branch of the dated phylogeny, with
variance proportional to the branch's length in time. Two species then share all the drift that happened on
branches above their most recent common ancestor, so their traits are correlated in proportion to how long they
evolved together.

The model builds this into the species vectors. The dated tree (30,859 nodes over the 18,600 US natives,
`scripts/build_tree.py`: the 128,270-tip megatree of ferns and seed plants of Carruthers et al. pruned to our
species, missing species grafted at their genus, family or order, the lycophytes added as a sister clade) has one
branch above every node. Each branch e gets a vector z_e of 256 numbers, and

    w_s = sum over the branches e between the root and species s of sqrt(l_e / l_mean) z_e  +  u_s,

where l_e is the branch length and l_mean the mean branch length (`tree.path_matrix`; the matrix of these loadings
is A, so the first term is A z). With z drawn from a standard normal distribution, A z has exactly the Brownian
covariance. Fitting z with a Gaussian penalty (weight decay 0.01) is the corresponding most-probable estimate: a
species with few records keeps the niche of its clade and moves away from it only as far as its own records
demand; branches shared by many species are estimated from all of their records.

u_s is a species-specific vector with a weaker penalty (weight decay 0.0001). It lets a well-recorded species
depart from its clade's niche without moving the shared branches, i.e. without disturbing its relatives.

Measured on the 2,661-species benchmark (`docs/scientific_provenance.md`, 2026-10-04 23:25 and 2026-10-05 00:05):
with all records the prior changes VegBank AUC by at most 0.002 (within the seed-to-seed spread), while for species
reduced to 5 records it raises AUC of the unclipped map from 0.77 to 0.82-0.84, and for species with no records from
chance (0.50) to 0.72-0.75.

## 4. Training

Each step draws 256 species (uniformly among those with records) and, for each, 64 of its presences and 1,024 of
its background points (with replacement); the loss is L_s of section 1 averaged over the 256 species. The optimizer
is AdamW (stochastic gradient descent with per-parameter step sizes and decoupled weight decay), learning rate
0.001 on a one-cycle schedule (rising over the first 5% of steps, then decaying along a cosine), gradient norm
clipped to 5. Weight decay: 0.0001 on the network and offsets, 0.01 on the branch vectors z, 0.0001 on the species
vectors u. All training points live on the GPU as 16-bit floats; the network runs in bfloat16 (a 16-bit format
with the range of 32-bit floats).

| Setting | Production run `conus_final` |
|---|---|
| Species, training points | 16,448; 179.1 million (15.3 million presences) |
| Steps | 21,000 (about 330 draws per species) |
| Width, network depth | 256; 3 (`env`) + 2 (`trunk`) linear layers |
| Time | 43 min on one RTX 3090, evaluations included |

Model selection uses independent plots, never the training records. VegBank vegetation plots (complete species
lists, so a species missing from a plot is an absence) are split into two halves by 1-degree blocks (about 111 km
north-south): a plot's block, hashed, decides whether it is a "dev" or a "test" plot, so the two halves never share
a neighbourhood. Every 3,000 steps each species with at least 5 presence plots is scored by its AUC (the
probability that a randomly chosen presence plot scores higher than a randomly chosen absence plot; 0.5 is chance,
1 perfect), with plots outside its calibration area ranked lowest, as they show on the map. The checkpoint with the
best median AUC on the dev half is kept (`model_best.pt`); the test half and the BLM AIM and FIA plots are reported.

The comparison is the median over the species that also have a per-species MaxEnt map, paired on the same plots
(`selection: dev_joint_paired`; without MaxEnt maps the median over all scored species). For `conus_final` it chose
the checkpoint of 9,000 steps (`docs/scientific_provenance.md`):

| Independent plots | species | joint model, median AUC | species with a MaxEnt map | joint vs per-species MaxEnt |
|---|---|---|---|---|
| VegBank dev / test | 3,367 / 3,419 | 0.9422 / 0.9555 | 1,086 / 1,085 | 0.9371 vs 0.9329 / 0.9547 vs 0.9535 |
| BLM AIM | 2,009 | 0.9216 | 273 | 0.9320 vs 0.9268 |
| FIA | 259 | 0.9426 | 186 | 0.9266 vs 0.9297 |

## 5. Species without records

About 500 of the 16,947 CONUS natives have no usable georeferenced record after cleaning. The model's tree is the
full tree of all 18,600 US natives, so these species are tips of it, and every branch with a recorded species somewhere
below it has a learned vector z_e. Climbing from a species' tip towards the root, the first node x whose clade
contains trained species is where it joins its closest trained relatives. Under the Brownian prior its expected
niche vector is the value at x, the path sum from the root down to x:

    w_m = sum over the branches e between the root and x of sqrt(l_e / l_mean) z_e;

the branches from x down to the species, which no record informs, contribute their prior mean, zero (as does its
species term). Its offset is the relatives' mean offset. Its calibration area comes from the World Checklist of
Vascular Plants (WCVP) instead of records: the ecoregions that overlap its native WGSRPD level-3 regions (botanical
countries and states) by at least 10% of the ecoregion's area. 494 CONUS species get a map this way; the map store
flags them (`inferred`).

**Calibration areas on the grid.** A species' calibration area comes from the ecoregions holding its records. For 98
CONUS natives all of them lie off the CONUS grid (records only in Canada or Mexico, or at the edge of a large
ecoregion), so the map would be empty in the US. Their calibration area is extended with the ecoregions of their WCVP
native areas in CONUS by the same 10% rule, or, for a small area that holds no such ecoregion, with those
intersecting it (`scope.extend_calibration`). Every CONUS native then has a calibration area on its grid.

An earlier rule took the branches shared by all of the relatives, i.e. the path down to the relatives' own common
ancestor. That ancestor lies below x when the relatives all sit in one sub-clade (75 of the 683 species); for a
single closest relative the path then includes that relative's terminal branch. Leave-one-out test (2026-10-05, on
`national_final`; the 3,605 trained species with at least 20 VegBank presence plots, each treated in turn as
unrecorded): the path to x scores median AUC 0.9062 against 0.9003 for the earlier rule, and is better for 72% of
the species where the two rules differ (single-relative cases: +0.009 on average). The CONUS store is built with the
path-to-x rule.

## 6. The map store

Storing all US natives' maps on 240 m grids directly would take 502 GB as uncompressed rasters clipped to each
species' calibration area, or about 200 GB as compressed GeoTIFFs (measured for the 18,595 maps of `national_final`). The joint model allows much less, because every map is
read off the same 256 features: storing h(x) once per cell and (w_s, b_s) once per species stores every map. What
remains is to store h compactly and to know what rounding it costs.

**Transform coding.** A change e in the features moves the scores of all species by a total squared amount
|W e|², with W the matrix of all species vectors. So errors should be measured in the metric M = WᵀW, which weighs
each feature direction by how much the species use it. The store rotates the features into the principal
directions of their variation in that metric, a Karhunen-Loève transform (KLT):

    y(x) = (h(x) - μ) M^½ V,     r_s = Vᵀ M^-½ w_s,     o_s = <μ, w_s> + b_s,

with μ the mean feature vector over 265,256 sampled land cells of the CONUS grid (up to 12,000 from each of 32
windows of 512 × 512 cells, about 123 km × 123 km) and V the eigenvectors of the feature covariance in the metric,
largest variance first. Then f_s(x) = <y(x), r_s> + o_s exactly, and the r_s form an orthonormal frame
(Σ_s r_s r_sᵀ = I). The
consequence: an error vector e in y changes the scores of all species by a total squared error of exactly |e|².
Every channel and every species is equally sensitive, so one rounding step serves all of them, and no channel needs
to be dropped. The store keeps q(x) = round(y(x) / 3.2) as integers; low-variance channels round mostly to zero.

**Layout** (`reader.py` documents it field by field). Cells are grouped into tiles of 128 × 128 (about 31 km × 31 km). Within a tile the channels are kept up
to the last one that is not zero everywhere. Because channels are ordered by variance, almost every value fits in a
signed byte: only the first k16 channels (in the median tile none, at most 2) need 16 bits and are stored as two
byte planes (all low bytes, then all high bytes), the rest as single bytes. Each tile is compressed with zstd
(Zstandard, a fast general-purpose compressor). Per species the store keeps its code (3.2 · r_s, so
f_s = q · code_s + o_s), offset, 254 quantiles of f_s over its own background points, its P5 threshold (the 5th
percentile of f_s over its training presences) and its calibration ecoregion ids.

Species without records sample up to 30,000 of their relatives' training points for their quantiles and P5,
with a random generator seeded by the species' row in the table (1 + row), so any row can be recomputed on its own.

**Updating species without records.** Codes and offsets are linear in a species' vector
(code_s = 3.2 · w_s M^-½ V, offset_s = <μ, w_s> + b_s), so `store.update_zero_shot` (`national_store.py
update-zero-shot`) recovers that linear map from the store's trained species by least squares (it refuses unless
the codes are reproduced to a relative error below 1e-4 and the offsets to 1e-3 score units), applies it to new
vectors, recomputes quantiles and P5 for the species that changed (with `--all`: every species without records,
e.g. to bring a table drawn with other random generators to the per-row seeds), and replaces species.npz in one
step. The fields are untouched. A store written before the int8 split (all channels int16, a 3-column tile
index) reads as it is and is rewritten in the current layout by `national_store.py recode`.

**Decoding.** A served map value is the species' suitability: 1 + the number of its background quantiles that
f_s(x) exceeds (1-255, the share of its calibration-area background that x outscores), and 0 outside its calibration
ecoregions or where there is no climate. Its binary range is f_s(x) ≥ P5 inside the calibration ecoregions.
`Store.scores` returns a window of scores of one or more species, `Store.suitability` (also `decode`) and
`Store.in_range` the served maps of one species;
`Store.at` the score, suitability and range at points given as longitude and latitude or as grid cells;
`Store.window` and `Store.cells` the stored field itself. Reading needs only numpy and zstandard (rasterio and pyproj
for the calibration mask and coordinates).

| Store `cards_conus_klt` (`conus_final`, step 3.2) | |
|---|---|
| Species | 16,942 (494 inferred from relatives) |
| Size | 9.06 GB: CONUS field 8.88 GB (191.1 million land cells), species table 0.14 GB |
| Median VegBank AUC, as stored vs full model (3,605 species) | 0.9509 vs 0.9509 (mean loss −0.0001; Spearman of stored vs full scores inside the calibration area: median 0.989, 5th percentile 0.979) |
| CPU decode of one species (8 threads, shared workstation) | 256 × 256 cells 111 ms, 512 × 512 cells 633 ms (177 ms with cached tiles) |

The store of all US natives (`cards_national_klt`, `national_final`: 18,595 species over CONUS, Alaska and Hawaii)
is 16.7 GB, with a median VegBank AUC of 0.9506 as stored against 0.9505 for the full model.

## 7. Results of the stored maps

Evaluated exactly as stored and served (`scripts/national_store_eval.py plots`; species with at least 20 presence
plots; plots outside the species' calibration ecoregions ranked lowest), against the maps Daru (2024) published and
the per-species MaxEnt maps of the same species, at the same plots:

| Plot source | Species | Stored joint maps | vs Daru (2024): joint / Daru, joint better for | vs per-species MaxEnt: joint / MaxEnt, joint better for |
|---|---|---|---|---|
| VegBank | 3,607 | 0.951 | 0.948 / 0.910, 84% (141 species) | 0.942 / 0.938, 53% (1,547 species) |
| BLM AIM | 2,009 | 0.922 | 0.921 / 0.853, 86% (104 species) | 0.915 / 0.906, 64% (480 species) |
| FIA | 259 | 0.942 | 0.936 / 0.916, 100% (14 species) | 0.933 / 0.935, 46% (224 species) |

The store of all US natives scored VegBank 0.950 (3,614 species), AIM 0.921 (2,013), FIA 0.941 (259), with the same
comparisons against Daru (0.949 vs 0.910, 0.921 vs 0.853, 0.936 vs 0.916).

## 8. Running

```
python scripts/national_prepare.py      # training points from the per-species products
python scripts/national_plots.py        # independent plot sets
python scripts/national_scope.py        # restrict to CONUS
python scripts/national_train.py        # writes <runs_dir>/conus_final: norm.npz, model_best.pt, model.pt, run.json
python scripts/national_store.py build  # the map store, species without records included
python scripts/national_store_eval.py plots <store> --out <dir>      # independent plots, as stored
python scripts/national_store_eval.py fidelity <store> --out <dir>   # full model vs stored field
python scripts/national_store_eval.py bench <store>                  # CPU decode time, bytes on disk
```

Reading a store in Python:

```python
from ranges.joint.reader import Store
st = Store("cards_conus_klt", grids={"conus": "work/conus240"})
s = st.index("Quercus lobata")
suitability = st.suitability("conus", 6000, 6512, 1500, 2012, s)  # uint8 [512, 512]
in_range = st.in_range("conus", 6000, 6512, 1500, 2012, s)        # bool [512, 512]
```

## 9. Verification

`tests/test_joint_store.py` checks the Newick parser and path matrix, the KLT identity and orthonormal frame, the tile
writer and reader (window, scores, scattered cells, packed validity mask, lossless recoding), and, on synthetic data,
the whole chain: training learns (AUC > 0.75 on held-out plots), the zero-shot vector equals the path sum from the
root to the joining node (and differs from the earlier rule's), every stored score lies within the quantization bound
(total error over all species ≤ sqrt(256) · step / 2 per cell), decoded maps equal the rank of the stored score
exactly and are 0 outside the calibration area, and the species table equals the quantiles of the model's own scores.
A store built with other vectors for the species without records and then updated (`update_zero_shot`) gives them the
same quantiles and P5 as a rebuilt store, and scores within the same quantization bound.

The package code was checked against the research code that produced the stores (2026-10-05; "the research code"
below):

* published package against both production stores (2026-10-05, CPU): over 12 random windows of `cards_national_klt`
  (CONUS, Alaska, Hawaii) and 16 of `cards_conus_klt`, 3 species each, the published reader's scores, integer
  field, scattered cells, suitability maps and binary ranges equal the research code's, bit for bit (largest score
  difference 0.0), and `Store.at` at 2,000 cells per window equals the decoded windows;
* CONUS build rules: from the `conus_final` checkpoint, the published zero-shot rule, region inventory and
  calibration extension reproduce the production store's 494 species without records (names, order, flags), their
  codes (relative error 8e-7) and offsets (3e-6 score units), and every species' calibration area (all 16,942;
  98 extended); the published region rule excludes the same points as the research code's on 4 million random
  points; the published `build_tree.py` and `build_bias_grid.py` reproduce the production tree, placements and
  sampling-effort density byte for byte;

* model: the `national_final` checkpoint gives scores at 10,000 training points for 50 species within
  1.1e-5 of the research code (scores up to 55 in size); with identical species vectors the scores are bit-identical
  (the remaining difference is the run-to-run rounding of the GPU sparse product that forms A z);
* store reader: 12 random windows over the three regions decode bit-identically to the research code's reader (scores,
  integer field, scattered cells, uint8 maps);
* store evaluation: `national_store_eval.py plots` reproduces the research code's evaluation summary of the production
  store exactly (every median, species count and share above), and `fidelity` its 0.9505 / 0.9506;
* store builder: the same zero-shot species, relatives, calibration areas and vectors as the production build (with
  the earlier zero-shot rule; the current rule reproduces the research code's updated rule exactly); a transform
  sample (553,364 land cells) bit-identical to the research code's; species-table quantiles and P5 within 1e-5 (checked
  with the build's sequential random draws for species without records, since replaced by per-species seeds).
  Rebuilding the Hawaii field reproduces the production scores to a root-mean-square difference of 0.145, the level
  expected from two independent roundings at step 3.2 (0.153; the scores of a species vary with a standard deviation
  of about 2.5 within a 61 km window). The transform itself is not reproduced bit for bit: rerunning the research code
  today gives the same 0.009 shift of the sample mean (relative 7e-4) against the production build, i.e. the GPU's
  bfloat16 feature evaluation differed slightly between the two processes, and the low-variance channels, whose
  eigenvalues nearly coincide, then rotate within their shared subspace (which changes the codes but not the
  scores);
* training: 600 steps on the 2,661-species benchmark with the same seed and settings reach VegBank dev median AUC
  0.93283 and 0.93284 in two package runs, 0.93276 and 0.93285 in two runs of the research code (test 0.94646/0.94662 vs
  0.94650/0.94641): the package and the research code differ by no more than the research code differs from itself, which
  comes from the order of floating-point additions in GPU gradient accumulation.
