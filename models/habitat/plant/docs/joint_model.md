# The joint range model

One model maps the native ranges of the 16,448 plant species native to the contiguous United States (CONUS) that
have usable occurrence records, plus 494 species without usable records whose ranges are inferred from their
relatives: 16,942 maps on the 240 m CONUS grid, stored in one transform-coded field and decoded on an ordinary CPU.

The model exists in two versions. The **environment model** (sections 1 to 5; run `conus_final`, store
`cards_conus_klt`, published 2026-10-05) reads climate and soil at a location. The **full model** (section 6; run
`nat_repnp_s2`, store `cards_final_klt`, the published maps) adds what that location's surroundings look like (the
landscape field), where it is (place), a calibration area learned instead of imposed, records and plots on the
shoreline, and a background drawn like the records; it is trained in stages and fits all 16,448 species' parameters
to convergence in 156 s. As stored, its maps beat the strongest published competitor, SINR with environmental inputs,
in 65.9% of 4,282 species-by-plot-source tests, where the environment model's maps won 44% (section 8).

Code: `ranges/joint/` (`prepare`, `scope`, `tree`, `model`, `field` with its CUDA kernel in `kernels/`, `place`,
`geodesy`, `climate_fill`, `shoreline`, `data`, `train`, `cache`, `zero_shot`, `store`, `reader`); command-line entry
points `scripts/national_prepare.py`, `national_plots.py`, `national_scope.py`, `national_snap.py`,
`national_field.py`, `build_shoreline.py`, `build_climate_fill.py`, `national_train.py`, `national_cache.py`,
`national_store.py`, `national_store_eval.py`; configuration `configs/conus.json`. The measurements and decisions
behind every choice are logged in `docs/scientific_provenance.md` (entries of 2026-10-04 to 2026-10-06). The model was
developed on all 18,600 US natives (CONUS, Alaska and Hawaii; run `national_final`, store `cards_national_klt`); on
2026-10-05 the product was restricted to CONUS and retrained with the same settings (run `conus_final`, store
`cards_conus_klt`). Numbers below are for the CONUS product unless they name another run.

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

## 6. The full model: landscape, place, a learned calibration area, the shoreline

### 6.1 Where the environment model fell short

Every published distribution product was scored like the stored maps, on the same plots and species
(`docs/scientific_provenance.md`, 2026-10-05 21:55): the environment model wins clearly against Daru (2024), the
iNaturalist range maps and BIEN, but not against SINR (Cole et al. 2023) with environmental inputs, a network of
coordinates and WorldClim trained on 15 million iNaturalist observations (retrained with its authors' code and data):
pooled median AUC 0.943 against 0.951 over 4,282 tests (species by plot source), ours higher in 44%.

The paired median difference was only -0.002; a tail of 10% of the tests (difference below -0.05) carried more than
the whole deficit, with two causes:

* **The calibration area as a hard rule.** Outside the ecoregions holding a species' records its map is 0. For
  *Caltha leptosepala* on BLM AIM plots, 19 of 20 presence plots lie in an ecoregion holding none of its 1,856
  records: AUC 0.17 with the rule, 0.993 without. The area is informative but must not be absolute: on the
  environment model's saved VegBank scores, lowering scores outside the area by one standard deviation gives dev/test
  median AUC 0.9464/0.9589, against 0.9443/0.9580 with the hard rule and 0.9372/0.9498 with no rule.
* **The shoreline.** WorldClim has no value where a 30″ pixel's centre lies in the sea. Dune, beach, salt-marsh and
  mangrove species (*Uniola paniculata*, *Croton punctatus*, *Iva imbricata*, ...) had their shoreline presences
  dropped from training and their shoreline plots left unscored; they scored 0.6 to 0.7 where SINR, which reads a
  coarser raster everywhere, scored 0.96 to 0.997.

SINR has three things the environment model lacks: a learned function of place shared by all species, negatives
drawn from where other species were recorded ("assume negative"), and one network for all. The full model takes
these over, keeps what SINR lacks (the environmental niche with soil and terrain, the phylogenetic prior, a
calibration area) and adds a representation of the landscape around each location.

### 6.2 The score

For species s at location x,

    f_s(x) = < h(x) + G(C(x)), w_s >  +  < P(x), v_s >  +  b_s  -  pi_s [x outside K_s]

* h(x): the environment features of section 2;
* G(C(x)): the landscape around x (section 6.4), added to the environment features, so every species reads its
  surroundings through its niche vector w_s;
* P(x), v_s: place features and the species' place vector (section 6.5);
* pi_s: the species' learned penalty outside its calibration area K_s (section 6.3).

Every term is linear in a species vector, so the map store keeps working unchanged (section 7). The species vectors
w_s, v_s and the penalty all carry the Brownian-motion prior of section 3 (v = A z_p, pi = softplus(A z_c + c_0)),
so relatives share range geometry and calibration as they share niches, and a species without records gets all
three from its joining node (section 5).

### 6.3 Background drawn like the records, and a learned calibration area

**Target-group background.** The effort-weighted background points are centres of ~1 km cells drawn from a 10 km
effort density, while records sit where people record: along roads and trails, at survey stops, at localities many
species share. Any pathway that can tell nearby places apart then learns "a record is here", which separates every
species' presences from its background and is false at an independent plot; every place encoding tried before this
change (an absolute Earth4D hash grid of the coordinates, the community of nearby records) lost on held-out plots.
So each background point b moves to the nearest record of another species (Phillips et al. 2009):

    snap(b) = the record r of a species other than s that minimizes the great-circle distance to b

(`prepare.snap_to_records`, `community/snap.npy`). Presences and background then share the records' fine-scale
sampling.

**Continental background.** Each species also draws 512 points per step from the background of all species across
the continent, moved the same way. The objective of section 1 becomes

    L_s = - mean_{p in P_s} f_s(p) + log mean_{a in P_s + B_s + G} exp f_s(a),

with B_s the species' own (moved) background and G the continental draw. The species now sees where it is absent
outside its calibration area, as SINR's "assume negative" loss does. Drawing these pseudo-absences uniformly over
land instead, as SINR does, lowered AUC beyond 10 km from records (0.9014): in a niche model uniform negatives
penalize suitable but unrecorded places.

**Learned calibration.** The hard rule becomes a penalty learned from the continental background:

    pi_s = softplus( sum_e A[s, e] z_c,e + c_0 ),     c_0 = softplus^-1(3) at the start,

subtracted from f_s wherever x lies outside K_s. A training row's ecoregion is read from the 240 m ecoregion raster
at its own cell (its position computed as in section 6.4), as for the plots and the maps. The maps then continue
outside the calibration area, lowered by the species' penalty (section 7).

With both changes (2,661-species benchmark, 1,000 steps from the environment model), the scores without any
calibration rule beat the clipped ones for the first time: VegBank dev/test 0.9414/0.9509, BLM AIM 0.9342, against
0.9362/0.9497/0.9291 for the clipped target-group control.

### 6.4 The landscape field (Entropy3D)

The environment network reads a location's own cell. What a plant meets also depends on the surroundings: a ridge or
a valley, the foot of a mountain range or the middle of a basin, a coastline, the edge of a desert. The field reads
them from the raw data, at scales it learns (`field.py`).

**Data.** A pyramid of the 240 m CONUS grid (`field.build_pyramid`, `scripts/national_field.py`) with 12 channels:
fine terrain (elevation, slope, northness, eastness, topographic position index) and climate (lapse-rate-corrected
annual mean temperature, warmest-month maximum and coldest-month minimum; WorldClim temperature seasonality, annual
precipitation, precipitation seasonality, driest-quarter precipitation; precipitation amounts as log(1 + value)).
Each channel is standardized over the grid's valid cells and stored as int8 in steps of 1/16 standard deviation
(-128 = missing); level k (8 levels) is the mean of 2^k × 2^k level-0 cells, missing cells left out. Every training
row and plot gets its fractional (row, column) on the grid by the WGS 84 → NAD83 / Conus Albers transformation that
PROJ applied when the rasters were warped (`geodesy.py`: per point the most accurate of EPSG's operations whose area
contains it, NOAA HARN and NRC Canada grid shifts with PROJ's grid hierarchy, Albers on GRS80; within 1e-7 m of
pyproj on samples of up to 2 million CONUS points), so a record reads exactly its own cell.

**Perceptive fields.** DeepEarth's Entropy4D design builds a representation from perceptive fields, encoders with a
learnable position, extent and shape in space and time; the static, purely spatial case used here is called
Entropy3D. Around x: the centre (the values at x) and R = 8 rings at learnable radii r_j = exp(rho_j), initially
1, 2, 4, ..., 128 cells (0.24 to 31 km; the trained national model reads 0.22, 0.46, 1.04, 1.96, 3.65, 7.61, 15.07,
28.35 km). Ring j is read at A = 8 angles theta_a = 2 pi a / A (a = 0 north, clockwise) at

    (row, column) = (row_x - r_j cos theta_a, column_x + r_j sin theta_a),

by bilinear interpolation on pyramid level l = clamp(log2 r_j - 1, 0, 7) (a cell about half the radius), blending
the two levels around l, so the read moves smoothly with r_j and the radii receive gradients. Missing samples are
left out. Per ring and channel, with d_a = v(theta_a) - v(x) and n valid samples,

    a_0 = (1/n) sum_a d_a,     a_m = (2/n) sum_a d_a cos(m theta_a),     b_m = (2/n) sum_a d_a sin(m theta_a),

m = 1..3, and the magnitudes |(a_m, b_m)|. a_0 is the radial profile (is x higher or lower than its surroundings at
this distance); order 1 is a gradient across the ring (uphill to the west), order 2 an axis (a ridge or valley
through x), order 3 a three-fold pattern. The cos/sin pairs know the direction (a south-facing slope is not a
north-facing one); the magnitudes do not change when the landscape is rotated about the vertical. With 8 angles the
sums are exact for orders up to 3. Each ring also carries its missing share (1 - n / A, averaged over channels:
how much of it is sea or off the grid; the coastline enters here).

**Interaction and pooling.** Each ring becomes a token of width 64 (a two-layer network of its 12 × 10 + 1 numbers),
the centre another (its values and missing flags), each plus a learned embedding of its scale. Two self-attention
blocks (4 heads) let every token read the others (a valley inside a plateau is not a valley inside a plain);
attention with 4 learned queries pools the 9 tokens into C(x); a two-layer network G maps C(x) to the 256 features,
its last layer starting at zero, so the field starts as no change to a trained model.

**Computation.** One fused CUDA kernel (`kernels/ring_harmonics.cu`, compiled from source on first use) evaluates
every ring of every location, one GPU warp per (location, ring) and one lane per angle, and returns the sums above
and, in the backward pass, their analytic derivative with respect to the radii (bilinear quotient rule over the valid
corners plus the level-blend term). It equals the PyTorch reference to 7e-6 relative (sums) and 3e-5 (radius
gradients) (`tests/test_field.py`); on a CPU the reference itself runs. In training only the token network is
recomputed in the backward pass.

Benchmark, from the environment model with the background of section 6.3 (unclipped dev/test/> 10 km/AIM/FIA): at
250 steps 0.9431/0.9527/0.9145/0.9392/0.9411 with the field against 0.9410/0.9514/0.9111/0.9363/0.9415 without; at
1,000 steps (a 17-channel field) dev +0.003, test +0.002, > 10 km +0.005, AIM +0.003, FIA -0.001. The 17-channel
field (adding SoilGrids and distance to the coast), wider rings (to ~120 km) and 22 channels with NALCMS land cover
were draws with the 12-channel field, which is kept.

### 6.5 Place

`place.py`: SINR's location network. The position enters as [sin(pi lon/180), cos(pi lon/180), sin(pi lat/90),
cos(pi lat/90)] (continuous across the antimeridian), then a 256-wide input layer with ReLU, four residual blocks
(Linear-ReLU-Linear-ReLU with a skip connection) and a linear projection to 256 place features P(x), starting at
zero. Each species reads them through its place vector v_s = A z_p under the Brownian prior: range geometry that no
predictor explains (a barrier never crossed, a coastline, a history) is shared by relatives.

A fine absolute place encoding loses: Earth4D's multiresolution hash grid of latitude, longitude and elevation (12
levels to 249 m), decoded with its own network and place vectors, on the same base: unclipped 0.9258/0.9380/0.8917
/0.9099/0.9269 against 0.9417/0.9517/0.9103/0.9342/0.9414, worse with more steps (with per-species place vectors a
fine absolute grid memorizes each species' record cells). SINR's smooth coordinate network gains a little on four of the five columns
(0.9423/0.9517/0.9115/0.9357/0.9419); with the field, in the head-to-head with SINR, it wins every column beyond 10 km
from records and AIM and loses FIA over all plots, so it is kept.

### 6.6 The shoreline

`climate_fill.py`, `shoreline.py`. A location without climate takes all 20 WorldClim bands of the nearest place that
has climate, if it lies within 5 km: for points (records, plots) the nearest 30″ pixel centre by great-circle
distance, for the 240 m map grid the nearest cell by exact distance on the equal-area grid; farther locations stay
missing. Soil and fine terrain keep their own missing flags.

* Plots: 0.76% of VegBank and 0.09% of FIA plots had no climate, concentrated on shorelines (48% of *Uniola
  paniculata*'s presence plots); all 408 VegBank plots and 282 of 285 FIA plots are filled (median 0.6 km). They are
  now scored like every other plot (`plot_fill_<source>.npz`).
* Records: the per-species preparation draws a species' presences with a fixed seed and dropped those without
  climate. Replaying the draw recovers exactly the dropped ones (*Uniola paniculata* 224, *Croton punctatus* 178,
  *Abies concolor* 0), which are restored with the filled climate, SoilGrids at the point and their grid position:
  88,042 presences for the CONUS species (19 more lie beyond 5 km and stay out; `shore_records.npz`).
* Maps: the administrative US mask runs ~3 nautical miles offshore, so a cell is filled only where NALCMS 2020
  classifies it as under half water: 169,719 land cells of the CONUS grid (~9,800 km²) had no climate, 169,454 are
  filled (median 0.24 km; `climate_fill_conus240.npz`, applied by the store's grid inputs).

Scored on the filled plots, the full model wins against SINR env beyond 10 km from records on VegBank (ours higher
in 51.1% of the tests; the field model on the unfilled plots: 46%), pooled 60.7% (benchmark representation, before
the species stage of section 6.7).

### 6.7 Training in stages

The field costs 97% of a training step (benchmark profile: 8.55 s per step with every pathway, 186 ms without the
field, 84 ms for the environment alone). It needs few steps, but the species' parameters need many. With
target-group background every row a step reads is a record (a presence, or the record a background point moves
to), so once the shared networks are fixed the shared features F(x) = h(x) + G(C(x)) and P(x) of exactly those rows
can be computed once (`cache.py`) and a step costs only the species' dot products: 16 to 18 ms. Hence three stages
(`scripts/national_train.py --stage`, settings in `configs/conus.json`, `joint.stages`):

| Stage | What is trained | Data | Settings |
|---|---|---|---|
| base | the environment model (sections 2 to 4) | 2,661-species benchmark | 3,000 steps, lr 0.001 (run `l22_base_s0`) |
| representation | every pathway, started from the base run's final weights (new pathways at zero) | benchmark | 1,000 steps, 256 presences per species a step, lr 0.0003 (place network 0.001), target-group + 512 continental background points, field, place (256), learned penalty (run `rep_np256_s0`) |
| species | every species parameter (z, u, b, z_p, z_c) from scratch, and c_0; shared networks and input standardization fixed | the 16,448 CONUS species, restored shoreline records, filled plots | 6,000 steps, 256 presences per species a step, lr 0.001, weight decay 1 on the branch vectors, on cached features of 15.37 million record rows (156 s; run `nat_repnp_s2`) |

The shared networks are species-independent, so the representation learned on the 2,661-species benchmark serves
the national species stage. Re-learning every species parameter on the fixed representation beats the jointly
trained model on every column: the 1,000-step joint training had left the species' parameters under-trained
(benchmark, against SINR env: ours higher 63.2% after the species stage, 60.7% before).

**Prior strength.** AdamW's weight decay is decoupled: a parameter shrinks by lr × decay per step, so a decay of 0.01
at lr 0.001 moves the branch vectors by 6% over 6,000 steps and the prior had been nearly inert. With all records,
decays of 0.1 and 1 on the branch vectors and of 0.1 to 10 on the species terms change nothing measurable (the
benchmark species have thousands of records). The prior's job is
the data-poor species: with half of the evaluated species reduced to 5 presences, their AUC is 0.9133, 0.9142,
0.9153 and 0.9108 at decays 0.01, 0.1, 1 and 10 on the branch vectors; 1 is used. Longer species training (12,000
steps) and more background points (2,048) did not gain.

A representation trained with the restored shoreline records and one with 22 land-cover channels were draws after
the species stage (ours higher against SINR env in 61.3% and 62.2% of the benchmark tests, against 63.2%), and
focusing (each ring's radius adapted per location by a second pass, Entropy4D's focusing operator) was a draw at
250 steps at 10 s per step, so the simplest representation is kept.

**Presences per step.** A species' term of the objective is computed on the presences drawn for it in that step; 64
gave a noisy view of species with thousands of records, and 256 lowers that variance. Benchmark, species stage on the
fixed representation (share of tests where ours beats SINR env, all plots / > 10 km from records): 64 presences
63.2% / 55.5%, 128 63.5% / 56.0%, 256 65.7% / 58.2%, 512 63.9% / 58.0%, 1,024 63.2% / 58.7%; 512 presences with
2,048 background points 64.6% / 56.5% (more background does not help). With 256 presences in the representation stage
too: 66.3% / 58.5%. Both stages use 256 (the base stage keeps the 64 of section 4).

Also measured and not kept: every thinned record beyond the cap of 5,000 per species restored (12.4 million more
presences) gained in neither stage (62.2% and 62.8% against 63.2%); a representation learned from all 16,448 national
species instead of the benchmark lost (50.9%: in 1,000 steps each species is drawn only about 16 times); an
environment network of width 512 was a draw (64.2% against 63.2%; > 10 km 53.7% against 55.5%) and depth 5 lost
(60.4%); at 256 presences, species-stage prior strengths 0.3 and 3, 3,000 and 12,000 steps and learning rates 3e-4
and 3e-3 were draws or worse.

### 6.8 Results

In-training evaluation (`train.py`; VegBank dev/test halves, presence plots > 10 km from every training presence,
BLM AIM, FIA; median AUC per species, scores served as the maps serve them, i.e. without a hard calibration rule):

| Model | VegBank dev | VegBank test | > 10 km | AIM | FIA |
|---|---|---|---|---|---|
| full model, national species stage (`nat_repnp_s2`, 16,448 species) | 0.9581 | 0.9671 | 0.9382 | 0.9363 | 0.9527 |
| the same with 64 presences per species a step (`nat_arm4_s2`, published 2026-10-06) | 0.9565 | 0.9658 | 0.9354 | 0.9344 | 0.9510 |
| environment model (`conus_final`; the comparison recorded with it, provenance 2026-10-06 06:45) | 0.9434 | 0.9556 | 0.9162 | 0.9209 | 0.9391 |

Against every published product, on identical tests (plots filled, proximity masks extended to the restored
records; 4,281 tests against SINR env; share of tests where ours is higher, Wilcoxon signed-rank p): SINR env pooled
median 0.9580 against 0.9512, ours higher in 66.8% (p = 2e-140; the environment model: 44%), beyond 10 km from records
60.8% (p = 1e-60); BLM AIM 73.0% (beyond 10 km 68.9%), FIA 63.7% (beyond 10 km 62.0%, p = 2e-4); SINR
coordinates-only 75.7%, SINR distilled 78.1%, iNaturalist range maps 98.6%, iNaturalist geomodel 90.4%, BIEN 98.4%,
Daru (2024) 94.6% (median +0.046). For the model with 64 presences per step, scoring the broad taxon where a name's
usage clearly spans segregate species (74 tests) gave 62.8% instead of 63.1%.

These are the model's own scores; the maps as stored (`cards_final_klt`) are evaluated in section 8.

## 7. The map store

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

**The full model** (section 6) is stored the same way, because each of its terms is linear in a species vector: the
stored features of a cell are [F(x) | P(x)] (F = h + G(C), P the place features; 512 numbers) and the species
vectors [w_s | v_s], so f_s = <[F | P], [w_s | v_s]> + b_s and the transform above applies unchanged with d = 512.
A cell's place input is the latitude and longitude of its centre, its field position the centre of its own cell, and
a shoreline cell without climate takes its source cell's WorldClim bands (`climate_fill.GridFill`). The species
table also keeps the learned penalty pi_s (`penalty`).

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

**Decoding.** A served map value is the species' suitability: 1 + the number of its background quantiles that the
served score exceeds (1-255, the share of its calibration-area background that x outscores), 0 where there is no
climate. For the environment model the served score is f_s(x) inside the calibration ecoregions and nothing is served
outside them (0); for a store with a learned penalty it is g_s(x) = f_s(x) - pi_s [x outside K_s], served wherever
there is climate (`Store.served`). Its binary range is served score ≥ P5 where a value is served.
`Store.scores` returns a window of scores of one or more species, `Store.suitability` (also `decode`) and
`Store.in_range` the served maps of one species;
`Store.at` the score, suitability and range at points given as longitude and latitude or as grid cells;
`Store.window` and `Store.cells` the stored field itself. Reading needs only numpy and zstandard (rasterio and pyproj
for the calibration mask and coordinates).

| Store `cards_final_klt` (full model `nat_repnp_s2`, step 3.2, d = 512) | |
|---|---|
| Species | 16,942 (494 inferred from relatives, with place vectors and penalties from their joining node; 98 calibration areas extended) |
| Size | 6.01 GB: CONUS field 5.82 GB, species table 0.16 GB, validity mask 0.03 GB |
| AUC as stored vs full model, per species and plot set (5,873 tests) | median absolute difference 0.0006, 99th percentile 0.010; medians 0.9566 vs 0.9573 |
| CPU decode of one species (benchmark median; 90th percentile) | 256 × 256 cells 48 ms (60 ms), 512 × 512 cells 138 ms (163 ms) |

| Store `cards_conus_klt` (environment model `conus_final`, step 3.2) | |
|---|---|
| Species | 16,942 (494 inferred from relatives) |
| Size | 9.06 GB: CONUS field 8.88 GB (191.1 million land cells), species table 0.14 GB |
| Median VegBank AUC, as stored vs full model (3,605 species) | 0.9509 vs 0.9509 (mean loss −0.0001; Spearman of stored vs full scores inside the calibration area: median 0.989, 5th percentile 0.979) |
| CPU decode of one species (8 threads, shared workstation) | 256 × 256 cells 111 ms, 512 × 512 cells 633 ms (177 ms with cached tiles) |

The store of all US natives (`cards_national_klt`, `national_final`: 18,595 species over CONUS, Alaska and Hawaii)
is 16.7 GB, with a median VegBank AUC of 0.9506 as stored against 0.9505 for the full model.

## 8. Results of the stored maps

Evaluated exactly as stored and served (`scripts/national_store_eval.py plots`; species with at least 20 presence
plots; for `cards_final_klt` every plot with climate, outside the species' calibration ecoregions lowered by its learned
penalty, as the maps serve it), against the maps Daru (2024) published and the per-species MaxEnt maps of the same
species, at the same plots:

| Plot source | Species | Stored joint maps | vs Daru (2024): joint / Daru, joint better for | vs per-species MaxEnt: joint / MaxEnt, joint better for |
|---|---|---|---|---|
| VegBank | 3,607 | 0.963 | 0.961 / 0.910, 93% (141 species) | 0.958 / 0.938, 80% (1,547 species) |
| BLM AIM | 2,009 | 0.935 | 0.932 / 0.853, 96% (104 species) | 0.940 / 0.906, 86% (480 species) |
| FIA | 259 | 0.951 | 0.973 / 0.916, 100% (14 species) | 0.944 / 0.935, 79% (224 species) |

With the hard calibration rule instead (plots outside the area ranked lowest) the same store scores 0.957, 0.931 and
0.948: the learned penalty is worth 0.006, 0.004 and 0.003.

Against every published distribution product, scored the same way on the tests both cover (species × plot source;
share of tests where ours is higher; two-sided Wilcoxon signed-rank p; "> 10 km" keeps the presence plots more than
10 km from every training presence of the species, and all absences):

| Competitor | Tests | Joint / competitor, median AUC | Ours higher | p | > 10 km from training records: ours higher |
|---|---|---|---|---|---|
| SINR, coordinates + environment (retrained with its authors' code and data) | 4,282 | 0.9572 / 0.9512 | 65.9% | 4e-127 | 60.2% (0.9312 / 0.9240; p = 6e-55) |
| SINR, coordinates only (released) | 4,282 | 0.9572 / 0.9429 | 75.3% | 2e-287 | 65.5% |
| SINR, distilled (released) | 4,282 | 0.9572 / 0.9386 | 77.4% | < 1e-300 | 63.4% |
| iNaturalist range maps | 5,378 | 0.9568 / 0.8744 | 98.4% | < 1e-300 | 93.2% |
| iNaturalist geomodel (public small model) | 125 | 0.9222 / 0.8746 | 91.2% | 1e-16 | 89.6% |
| BIEN range maps (AIM only: BIEN holds VegBank and FIA plots, so those are not independent) | 1,854 | 0.9335 / 0.7430 | 98.5% | 1e-297 | 97.3% |
| Daru (2024) | 259 | 0.9536 / 0.8962 | 94.6% | 5e-39 | 82.7% |

Against SINR env by plot source: VegBank 63.2% of 2,750 tests (0.9617 / 0.9572), beyond 10 km 56.1% (p = 2e-16);
BLM AIM 72.5% of 1,281 (0.9416 / 0.9266), beyond 10 km 68.6%; FIA 61.4% of 251 (0.9539 / 0.9501, p = 3e-5), beyond
10 km 60.4% (p = 0.003). Excluding plots within 1 km of GBIF records from the datasets that also hold plot vouchers
changes the pooled share against SINR env by 0.2 points and no pooled share by more than 0.8 points (one of the 125
iNaturalist geomodel tests). The environment model's maps had won 44% of the tests against SINR
env (0.943 vs 0.951; `docs/scientific_provenance.md`, 2026-10-05 21:55).

The environment model's store (`cards_conus_klt`), with its hard calibration rule, scored on the same plot sets:

| Plot source | Species | Stored joint maps | vs Daru (2024): joint / Daru, joint better for | vs per-species MaxEnt: joint / MaxEnt, joint better for |
|---|---|---|---|---|
| VegBank | 3,607 | 0.951 | 0.948 / 0.910, 84% (141 species) | 0.942 / 0.938, 53% (1,547 species) |
| BLM AIM | 2,009 | 0.922 | 0.921 / 0.853, 86% (104 species) | 0.915 / 0.906, 64% (480 species) |
| FIA | 259 | 0.942 | 0.936 / 0.916, 100% (14 species) | 0.933 / 0.935, 46% (224 species) |

The environment model's store of all US natives scored VegBank 0.950 (3,614 species), AIM 0.921 (2,013), FIA 0.941
(259), with the same comparisons against Daru (0.949 vs 0.910, 0.921 vs 0.853, 0.936 vs 0.916).

## 9. Running

```
python scripts/national_prepare.py      # training points from the per-species products
python scripts/national_plots.py        # independent plot sets
python scripts/national_scope.py        # restrict to CONUS
python scripts/national_train.py        # the environment model: <runs_dir>/conus_final (norm.npz, model_best.pt, ...)

# the full model (section 6), for each data directory it trains on (--data-dir; default joint.data_dir)
python scripts/national_snap.py                  # target-group background: community/snap.npy
python scripts/national_field.py                 # field pyramid; grid positions of training rows and plots
python scripts/build_shoreline.py                # shoreline fill of the plots; restored shoreline records
python scripts/build_climate_fill.py             # shoreline fill of the 240 m map grid
python scripts/national_train.py --stage base
python scripts/national_train.py --stage representation
python scripts/national_cache.py                 # shared features of the representation for the species stage
python scripts/national_train.py --stage species

python scripts/national_store.py build  # the map store of store.model, species without records included
python scripts/national_store_eval.py plots <store> --out <dir>      # independent plots, as stored
python scripts/national_store_eval.py fidelity <store> --out <dir>   # full model vs stored field
python scripts/national_store_eval.py bench <store>                  # CPU decode time, bytes on disk
```

Reading a store in Python:

```python
from ranges.joint.reader import Store
st = Store("cards_final_klt", grids={"conus": "work/conus240"})
s = st.index("Quercus lobata")
suitability = st.suitability("conus", 6000, 6512, 1500, 2012, s)  # uint8 [512, 512]
in_range = st.in_range("conus", 6000, 6512, 1500, 2012, s)        # bool [512, 512]
```

## 10. Verification

`tests/test_joint_store.py` checks the Newick parser and path matrix, the KLT identity and orthonormal frame, the tile
writer and reader (window, scores, scattered cells, packed validity mask, lossless recoding), and, on synthetic data,
the whole chain: training learns (AUC > 0.75 on held-out plots), the zero-shot vector equals the path sum from the
root to the joining node (and differs from the earlier rule's), every stored score lies within the quantization bound
(total error over all species ≤ sqrt(256) · step / 2 per cell), decoded maps equal the rank of the stored score
exactly and are 0 outside the calibration area, and the species table equals the quantiles of the model's own scores.
A store built with other vectors for the species without records and then updated (`update_zero_shot`) gives them the
same quantiles and P5 as a rebuilt store, and scores within the same quantization bound.

For the full model (section 6): `tests/test_field.py` checks the field's orientation (on fields rising to the east and
to the north the first harmonic is a pure sine and a pure cosine term of the right size, on every pyramid level), the
reference's radius gradient against finite differences, the CUDA kernel against the reference (sums and radius
gradients, with missing cells, grid edges and radii between levels), the all-rings token construction against the
per-ring one, the module (zero start, chunking, gradients reaching the radii, sizes from a state dict) and the
pyramid builder. `tests/test_geodesy.py` holds the coordinate transformation to pyproj within 1e-6 m (random CONUS
points, every HARN grid edge, Quebec's nested grids, integer coordinates). `tests/test_climate_fill.py` checks that
the batched shoreline fill equals its one-point definition and the grid fill a brute-force nearest cell.
`tests/test_joint_stages.py` checks the exact batched AUC against scikit-learn (CPU and GPU), target-group snapping
against brute force, the penalty and place pathways, and, end to end on synthetic data, the three training stages: the
cache equals the features the representation computes for its rows and plots, the species stage leaves the shared
networks unchanged and its cached plot scores equal the full model's, and a store of the full model decodes within
the quantization bound (d = 512) and serves maps outside the calibration area lowered by each species' penalty.
The trained national checkpoint `nat_repnp_s2` loads into the package model from its state dict alone and, scored by
the package (CUDA kernel) at all 53,797 VegBank plots for its 3,605 evaluated species, reproduces the research run's
saved scores to the precision they were saved in (float16: largest difference 0.018 on scores up to 42, median 0.0036;
per-species Spearman correlation median 0.999997, minimum 0.99999) and its AUCs (median 0.96357 vs 0.96362, largest
per-species difference 4e-4).

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
