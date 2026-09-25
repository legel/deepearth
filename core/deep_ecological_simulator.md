# Deep ecological simulator

A research report and engineering roadmap for adding learned operators to the DeepEarth Simulator's
hydrology, so that it reconstructs and forecasts real, held-out observations at any site in the United
States from national inputs only. It covers what must be proven and how, which parts of NVIDIA's
HydroGraphNet and DeepMind's GraphCast are worth adopting, how the DeepEarth space-time encoder carries
every modality, which inputs are essential, the training and test sites, the cost, and the code: modules,
classes and equations.

Status, 2026-09-25: a plan. The only measurements of our own are HydroGraphNet's size and cost on a GPU
(sections 4.1 and 10); its accuracy was not reproduced. Nothing here is validated yet. The process model it builds on is validated only in the
places section 3 states.

## 1. The answer in brief

| question | decision |
|---|---|
| What is learned | Six named, bounded parameters per ground cell of the existing water balance, and one lateral subsurface flux, predicted by a graph network from national inputs. The network never outputs water. |
| What integrates | The existing hourly per-cell water balance, unchanged in form. Mass closes by construction, to float precision. |
| The graph | Nodes on a 2 m lattice over the site; edges of two types: downslope flow (multiple flow direction) and a multiscale mesh (GraphCast's multimesh, on a square lattice). Encoder and decoder to the 0.5 m water cells. |
| The embedding | From the second milestone, when observations become inputs for inferring a state: Earth4D's relative (translation-equivariant) channel only, each observation a token at its (x, y, z, t) offset averaged over its footprint, and every modality reconstructed through its observation operator. The absolute channel stays off, because it memorizes places. |
| The proof | Three NEON sites never used in training or model selection (Harvard Forest, Konza Prairie, Santa Rita), years 2024 to 2025 never seen at any site: soil water at every NEON probe depth, tower evapotranspiration, throughfall. The learned model must beat the process model as it is today and climatology at every test site, on national inputs only. |
| Essential inputs | Seven, all national: 3DEP LiDAR, SSURGO (POLARIS where blank), NAIP land-cover classes, AORC rain (MRMS for recent hours), NSRDB sun, AORC air, and MODIS LAI (VIIRS for continuity). |
| NISAR, SMAP, PACE, SWOT | NISAR and SMAP are later constraints on the top 5 cm, at their own footprints. PACE and SWOT do not constrain site-scale soil water or flow. |
| First milestone | Ten NEON sites, the six static parameters, soil water and ET at the three test sites. About four weeks. |

## 2. The target and the proof

### 2.1 What must be reconstructed

The model is correct, in the sense the simulator needs, when it ties together real observations it was
not trained on: driven only by national inputs, it reproduces what instruments at a held-out site recorded
in held-out years, and it forecasts them from a state it had to infer. Each target is an observation with
a known footprint, depth and error:

| observable | source | what it measures | footprint | interval | error to carry |
|---|---|---|---|---|---|
| soil water content | NEON DP1.00094.001 | volumetric water at up to 8 depths, 6 cm to 2 m, in 5 soil plots per site | a point per sensor | 30 min | NEON's per-value expanded uncertainty, plus plot-to-cell representativeness |
| soil water content | FLUXNET SWC_F_MDS_n | the tower's own profile | a point | 30 or 60 min | as above |
| evapotranspiration | FLUXNET LE_F_MDS | latent heat flux, eddy covariance | the flux footprint, 10^4 to 10^5 m² upwind | 30 or 60 min | random error, and the energy-balance gap (below) |
| throughfall | NEON DP1.00046.001 | rain below the canopy, tipping buckets with troughs | a trough | 1 and 30 min | undercatch; solid precipitation recorded late |
| precipitation | NEON DP1.00044.001, DP1.00045.001 | weighing gauge (DFIR) and tower-top tipping bucket | a gauge | 1 to 60 min | wind undercatch |
| streamflow | NEON DP4.00130.001, USGS NWIS | discharge from stage | the catchment | 1 to 15 min | rating curve |

Two measured properties of the NEON targets at HARV (2019-01 to 2025-08) shape how they are scored:

- **Soil water levels are not trustworthy; dynamics are.** A sensor's level depends on its calibration,
  which moves the level and not the response, so level enters as a reported bias and the proof is on
  anomalies and storm responses. Depths must come from NEON's `swc_depthsV3` table (the sensor-position
  offsets are wrong for this product), which puts the top sensors at 5 to 7 cm. The final quality flag is
  set on 63 to 92% of the top sensors' half-hours, mostly for missing one-minute samples, and those
  failures cluster in wet spells: the clean hours under-sample wet soil, so their mean is not the soil's
  mean. Where a storm is scored both on clean hours and on hours whose only failures are missing samples,
  arrival, rise and drying rate agree for 97 to 100% of storms at four of the five top sensors.
- **One gauge's rain is uncertain by about 15% a year.** The tower-top tipping bucket against NEON's
  double-fenced weighing gauge: 1.11 of it from May to October, 0.81 from November to April (snow), 0.70 to
  1.16 by year. The two Harvard towers, 140 m apart, agree on total rain to 3.5% but hour for hour only to
  r 0.76.

The energy-balance gap is the largest known error among the targets. At US-xHA (Harvard's NEON tower)
the energy-balance ratio was 0.61 to 0.83 in 2019 to 2024, so raw LE gives 443 to 622 mm a year and LE
closed by the Bowen ratio (Twine et al. 2000) gives 607 to 753 mm. The target for ET is the closed value,
with the half-gap as part of its error (section 5.5).

### 2.2 The split

Space and time are both held out, and the test set is fixed before any model is trained, as in GraphCast's
development protocol (Lam et al. 2023): nobody looks at test data until the architecture and training are
frozen.

- **Test sites, never trained on and never used to choose anything:** HARV (Harvard Forest, deciduous
  broadleaf; towers US-xHA and US-Ha1, NEON stream HOPB), KONZ (Konza Prairie, tallgrass; US-xKZ and
  US-Kon, stream KING), SRER (Santa Rita, desert shrub; US-xSR, US-SRM and US-SRG, stream SYCA). No
  training tower lies within 50 km of a test site.
- **Test years, never seen at any site:** 2024 and 2025. Training 2016 to 2022 (NEON soil water begins
  2015 to 2017; the NEON FLUXNET products cover 2019 to 2024), validation 2023.
- **Model selection** at 20 and 50 sites is by 5-fold cross-validation over groups of NEON domains, so
  every choice is made on sites in unseen ecoregions.

One caveat is stated up front. Harvard has already informed the process model itself (its canopy, its
drainage fix and the diagnosis of its top layer were worked out against HARV and US-Ha1), so for B1 it is
not a clean test. It is held out from everything the learned model sees; the clean test of the process
physics is at KONZ and SRER, where nothing has been tuned.

The UC Berkeley site is an application, not a test: the nearest tower publishing data (US-CGG) is 25.9 km
away, and Berkeley Way West (US-DBK) has no product.

### 2.3 Metrics

For a series of model values $\hat y_t$ against observations $y_t$ over the test years, with means
$\mu$, standard deviations $\sigma$ and correlation $r$:

$$\mathrm{bias} = \mu_{\hat y} - \mu_y, \qquad \mathrm{ubRMSE} = \sqrt{\mathrm{RMSE}^2 - \mathrm{bias}^2}$$

$$\mathrm{KGE} = 1 - \sqrt{(r-1)^2 + (\sigma_{\hat y}/\sigma_y - 1)^2 + (\mu_{\hat y}/\mu_y - 1)^2}$$

$$\mathrm{SS}_{\mathrm{ref}} = 1 - \frac{\mathrm{MSE}_{\mathrm{model}}}{\mathrm{MSE}_{\mathrm{ref}}}$$

- Soil water: daily means per NEON plot and depth band (0 to 10, 10 to 30, 30 to 100 cm), on hours with
  final flag 0: ubRMSE in m³ m⁻³ and r of anomalies from a 31-day running mean (which isolates event
  response from the seasonal cycle), and per storm of 5 mm or more the wetting-front arrival, the rise and
  the drying e-folding time. Bias is reported and not scored (2.1).
- ET: daily and monthly totals, RMSE in mm d⁻¹, bias in percent, r; annual totals against the closed and
  the raw value.
- Throughfall: event totals (events separated by 6 dry hours), bias and RMSE of the ratio throughfall /
  rain.
- Streamflow (from the second milestone): KGE and NSE on daily flow and on log flow.
- Forecast (second milestone): the same scores as a function of lead time, 1 to 30 days, from a state
  inferred from observations up to the forecast date, against persistence.

### 2.4 Baselines and the bar

| baseline | what it is | where it applies |
|---|---|---|
| B0 climatology | the day-of-year mean of that sensor's own record before 2024 | every target; a strong local bar, since a truly ungauged method has no such record |
| B1 process model | the hourly water balance of section 3 with national default parameters, as it runs today | everywhere |
| B2 calibrated process model | B1 with its six parameters fit per site on 2016 to 2022 | training sites only (a held-out site has no data to calibrate on) |
| P persistence | the last observed value held for the lead time | forecasts |

The bar for the first milestone, at each of the three test sites and in 2024 to 2025: skill score above
zero against B1 and against B0 for soil water anomalies in every depth band and for daily ET; no loss of skill
against B2 at the training sites in the test years; and the water budget closed to $10^{-6}$ relative.
What B1 scores today at Harvard (section 3.1) sets the first numbers to beat: deeper-layer soil water
anomaly r 0.65, top-layer r 0.02 in 2024 with ubRMSE 0.054 m³ m⁻³, annual AET 640 to 688 mm against the
tower's closed 607 to 753 mm. Those numbers precede a drainage fix and are being re-measured; the test uses
whatever B1 scores at freeze.

## 3. What exists today

### 3.1 The process model

The water model runs per ground cell of a 0.3 to 0.7 m grid (the LiDAR survey's own pulse spacing; 0.5 m
at Harvard, about 16,200 cells an acre), hourly, for every year of the forcing record, as PyTorch tensors
on a GPU. Each cell carries a canopy store $c$, ponded water $w$, layer 1 water content $\theta_1$ (0 to
$Z_e = 0.10$ m) and layer 2 $\theta_2$ ($Z_e$ to root depth $Z_r$), all in mm or m³ m⁻³. Every hour:

| process | equation | source |
|---|---|---|
| interception | $S_{\max} = 0.935 + 0.498\,\mathrm{LAI} - 0.00575\,\mathrm{LAI}^2$ mm; throughfall is rain beyond it | von Hoyningen-Huene 1981 |
| infiltration | $f = K_s\,(1 + \psi_f(\theta_s - \theta_1)/F)$ with Mein-Larson ponding | Green and Ampt 1911; Mein and Larson 1973 |
| run-on | excess water routed downslope over the filled DEM by multiple flow direction, in topological order, each receiver infiltrating what it can | Quinn et al. 1991; Planchon and Darboux 2002 |
| reference ET | ASCE standardized hourly $ET_0$ from per-cell net radiation and 2 m wind | ASCE-EWRI 2005 |
| actual ET | FAO-56 dual crop coefficient: $T = K_s K_{cb} E'$, $K_s = \mathrm{clip}((TAW - D_r)/((1-p)\,TAW), 0, 1)$; soil evaporation through the exposed fraction | Allen et al. 1998 |
| drainage | unit-gradient Brooks-Corey, $d\theta/dt = -K_s S_e^{3 + 2/\lambda}/(1000\,\Delta z)$, integrated exactly over the hour, never below field capacity | Brooks and Corey 1964 |
| leaves | MODIS MCD15A3H LAI (500 m, 4 days) by day, distributed to canopy columns by their LiDAR plant area | Myneni et al. 2015 |

Each cell takes one of six land-cover classes (built, sealed, transient, pervious ground, tall
vegetation, vegetated roof) that set which terms apply, and its soil from SSURGO horizons, POLARIS where
SSURGO is blank, and the Rawls, Brakensiek and Miller (1983) texture table for missing suction and pore
terms. Net radiation per cell comes from the solar product (section 7) and the 2 m wind from the wind
product. Each cell-year closes its budget to $10^{-3}$ in float32.

What it lacks, stated as physics rather than as error: no lateral subsurface flow, no water table, no
upward capillary flux between the layers, one texture per cell, and leaves only as a site-mean LAI
distributed by plant area. Its measured skill at Harvard (US-xHA cell, 2019 to 2024) is the B1 row of
section 2.4. The top layer's near-zero correlation has a known physical candidate (no capillary rise, a
10 cm layer against a 6 to 7 cm sensor) that should be fixed as physics before anything is learned.

The storm model (DeepEarth `models/hydro`) solves the 2D local-inertial shallow-water equations (Bates,
Horritt and Fewtrell 2010) with Green-Ampt with redistribution (Ogden and Saghafian 1997), and is exact
against analytic solutions to $10^{-9}$. It began as **Qin Huang's** hydro package: the solver, terrain
conditioning (stream burning, depression breaching, D8, HAND), storm forcing, flood probability and gauge
validation are hers, and the deep simulator reuses her terrain conditioning and her gauge scoring
(`models/hydro/terrain.py`, `models/hydro/validate.py`). Against USGS gauge 02234400 through Hurricane Ian
it overpredicts the peak eightfold (NSE -19.7); refining the grid changes the runoff coefficient by 0.4%,
and the README traces the rest to storage (a water table at the surface under a third of the catchment in
SSURGO) and to a delineated catchment of 15.3 of the gauge's 33.2 km². That result is the clearest
argument in this document for learning storage parameters against gauges rather than trusting defaults.

### 3.2 What we hold and can reach

| data | extent | resolution | span | size |
|---|---|---|---|---|
| FLUXNET-1F towers (AmeriFlux, NEON) | 719 US towers indexed, 308 with a product | 30 or 60 min | 2,395 site-years, 1991 to 2025 | 23 GB raw; 1.6 GB hourly; 0.48 GB soil profiles |
| towers with measured LE over 30% of hours and soil water | 200 | as above | | |
| NEON eddy-covariance towers with a FLUXNET product | 45, of which 39 publish soil profiles | 30 min | 2019 to 2024 | |
| NEON soil water (DP1.00094.001) | 46 sites, 5 plots, up to 8 depths | 1 and 30 min | 2015 to 2017 on | HARV held, 2019-01 to 2025-08 |
| NEON throughfall (DP1.00046.001), precipitation (DP1.00044.001, DP1.00045.001) | 35 and 44 sites | 1 and 30 min | 2015 on | HARV held; 0.94 GB for all HARV products |
| NEON discharge (DP4.00130.001) | 28 streams | continuous | | not yet fetched |
| AORC rain and air | CONUS | about 1 km, hourly | 1979 on | fetched per site |
| NSRDB v4 sun | CONUS | 4 km, 30 min | 1998 on | fetched per site |
| MODIS LAI | global | 500 m, 4 days | 2002 on | fetched per site |
| 3DEP LiDAR, SSURGO, POLARIS, NAIP | CONUS | 0.3 to 0.7 m, polygons, 30 m, 0.6 m | one epoch | fetched per site |

Gaps that block the proof: NEON's data API needs a (free) account token for files; product metadata is
open. Only HARV's files are held so far, so the nine other sites must be fetched the same way. The NEON
FLUXNET products end in 2024, so ET in 2025 comes from NEON's own eddy-covariance bundle
(DP4.00200.001). MODIS Terra and Aqua are past their design life, so LAI must move to VIIRS (VNP15A2H,
500 m) for continuity.

## 4. What we studied

### 4.1 HydroGraphNet

Taghizadeh et al. (2025) train a graph network to emulate HEC-RAS 2D flood simulations of the White River
near Muncie, Indiana, and publish the code in NVIDIA PhysicsNeMo
(`examples/weather/flood_modeling/hydrographnet`, model `physicsnemo.models.meshgraphnet.MeshGraphKAN`,
data pipe `physicsnemo.datapipes.gnn.hydrographnet_dataset.HydroGraphDataset`) and the data on Zenodo
(record 14969507, 8.3 GB, CC-BY-4.0). From the code:

| part | what it is |
|---|---|
| graph | 4,787 nodes (HEC-RAS cell centers); edges to the k = 4 nearest neighbors in (x, y) by k-d tree, directed, 19,148 edges; no flow direction |
| node features, 16 | standardized x and y, cell area, elevation, slope, aspect, curvature, Manning n, flow accumulation, infiltration; upstream inflow and rainfall broadcast to every node; water depth and volume at the last 2 time steps |
| edge features, 3 | standardized dx, dy, distance |
| encoder | node: a Fourier Kolmogorov-Arnold layer, $y_o = \sum_i\sum_{k=1}^{5} a_{oik}\cos(k x_i) + b_{oik}\sin(k x_i)$ (the README says splines; the code is Fourier); edge: a 2-layer MLP |
| processor | 15 MeshGraphNet blocks (Pfaff et al. 2021), hidden 128, edge MLP then node MLP, sum aggregation, LayerNorm, residual |
| decoder | 2-layer MLP to $\Delta h$ and $\Delta V$ per node, added to the last state |
| time | autoregressive, $\Delta t$ = 20 min; test rollouts of 30 steps (10 h) |
| loss | MSE on one step, plus a pushforward stability term (Brandstetter et al. 2022), plus a physics term |
| physics term | graph-level only: $\mathrm{ReLU}(V_{pred} - V_{past} - \Delta t(Q_{in} + P A_{inf}))^2 + \mathrm{ReLU}(V_{fut} - V_{pred} - \Delta t(Q_{in}' + P' A_{inf}))^2$, normalized by total area; one-sided, and silent about where water sits |
| parameters | 2,318,722 (measured): processor 2,231,040, edge encoder 33,792, KAN node encoder 20,608, decoder 33,282 |
| reported skill | against a baseline GNN: 67% lower RMSE, near-zero mass-balance error, 58% higher critical success index for major floods |

What we ran, on one NVIDIA L4. The model as published, instantiated from PhysicsNeMo, has 2,318,722
parameters, 96% of them in the processor; its cost against graph size is in section 10. The dataset holds
500 synthetic hydrographs (`train.txt` lists all 500) on a 4,787-cell mesh covering 29.6 km² (mean cell
6,186 m²), each 217 steps of 20 min (72 h) before the loader's trimming. The dataset records HEC-RAS's own
run time: 128.5 s per hydrograph on average (90.8 to 196.7 s). One forward step of the model takes 15 ms on
the L4, so the 217-step hydrograph takes about 3.3 s, about 40 times faster than HEC-RAS on its (unstated)
CPU. Two things in the release matter for anyone reproducing it: the record carries no test split (all
500 hydrographs are in `train.txt`, and the `Test/test.txt` the inference script reads is absent), and
the physics term uses the upstream inflow but not the downstream outflow, although the record has both
(`M80_US_InF_*`, `M80_DS_OuF_*`); the one-sided ReLU is what keeps the missing outflow from breaking it.
**Not measured:** the paper's accuracy. The first GPU was preempted during the 8.3 GB download (26 GB unpacked),
and the second was stopped at its time limit while the loader was still reading the hydrographs (about
3.4 per second), so the 67% figure above is the authors', not ours.

Verdict. HydroGraphNet is a good emulator of one hydraulic model on one mesh, and three of its parts are
worth keeping: the MeshGraphNet processor block, residual (incremental) prediction of the state, and the
pushforward trick for rollout stability. The rest does not fit a model meant to run anywhere and to match
observations:

- **Absolute x and y are node features.** A model that reads where a node is cannot be moved to another
  place without retraining; this is the opposite of transfer.
- **The graph ignores which way water flows.** k-nearest neighbors in the plane connect cells across
  ridges and banks alike; the flow direction is learned from elevation features at best.
- **Mass is a soft, global, one-sided penalty.** It constrains the total volume only, so water can be
  created in one place and destroyed in another at zero cost. DUALFloodGNN, which predicts flows on
  edges and enforces per-node balance, reports depth RMSE 0.21 m against HydroGraphNet's 0.76 m on the
  Wollombi River (Acosta et al. 2025); the two differ in more than the constraint, but it is the same
  point, measured.
- **The target is a solver, not the world.** Its skill is agreement with HEC-RAS; nothing in it touches
  an observation.
- **The KAN layer is not where interpretability comes from.** Only the first layer is a KAN; the 15
  processor blocks are MLPs. Interpretability in our design comes from the network predicting named
  physical parameters (section 5.3).

### 4.2 GraphCast

Lam et al. (2023) forecast the global atmosphere 10 days ahead at 0.25° with an encoder-processor-decoder
graph network of 36.7 million parameters, better than ECMWF's HRES on 90.3% of 1,380 targets; the code is
now in `google-deepmind/weathernext` (`weathernext/weathernext1_graph/graphcast.py`).

| part | what it is | module |
|---|---|---|
| grid nodes | 721 × 1,440 = 1,038,240 latitude-longitude points, 474 input features each: 227 variables at 2 times, 5 forcings at 3 times, 5 constants | `GraphCast._inputs_to_grid_node_features` |
| multimesh | an icosahedron refined 6 times (40,962 nodes); the edges of every refinement level kept together, 327,660 in all, so coarse levels act as long-range shortcuts | `icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere`, `merge_meshes` |
| encoder | one message-passing step on a bipartite grid-to-mesh graph: each grid point to every mesh node within 0.6 of the longest finest-mesh edge (1,618,746 edges) | `grid_mesh_connectivity.radius_query_indices`, `GraphCast._run_grid2mesh_gnn` |
| processor | 16 unshared interaction-network layers on the multimesh, latent 512, swish, LayerNorm | `deep_typed_graph_net.DeepTypedGraphNet` |
| decoder | one step on a mesh-to-grid graph: each grid point from the 3 vertices of its containing triangle | `grid_mesh_connectivity.in_mesh_triangle_indices`, `GraphCast._run_mesh2grid_gnn` |
| edge features | length and 3D displacement in the receiver's local frame | `model_utils.get_graph_spatial_features` |
| prediction | residual: $\hat X_{t+1} = X_t + \hat Y_t$, outputs scaled by the standard deviation of one-step differences | `normalization.InputsAndResiduals` |
| loss | MSE over 12 autoregressive steps, weighted per variable and level by the inverse variance of time differences, by pressure, and by grid-cell area | `losses.weighted_mse_per_level` |
| curriculum | 1,000 warm-up updates, 299,000 one-step updates (half-cosine learning rate), then 11,000 updates growing the rollout from 2 to 12 steps; backpropagation through the full rollout; about 4 weeks on 32 TPU v4 | paper, section 4.5 |
| data split | train 1979 to 2015, validate 2016 to 2017, test 2018 on, fixed before development | paper, section 4.1 |

What makes it "predict anywhere" is that one set of weights is applied at every node: message passing is
the same function everywhere, edge features are relative and expressed in the receiver's frame, and the
multimesh gives every node the same view at every scale. Two things do not transfer to us: its absolute
node features (cosine of latitude, sine and cosine of longitude) are harmless only because there is one
planet and all of it is in training; and its target, a dense reanalysis on every grid point at every step,
has no analog at a site, where observations are sparse points. Removing its multimesh cost skill for every
variable except 50 hPa beyond 5 days (its supplement, section 7.3.1), and its ablation of rollout length showed that training on
longer rollouts trades short-lead for long-lead skill; both lessons carry over.

### 4.3 DeepEarth and Entropy4D

DeepEarth (Legel et al. 2026) is a masked multimodal autoencoder whose position encoder, Earth4D, extends
multiresolution hash encoding (Müller et al. 2022) to four dimensions as one xyz grid and three
space-time grids (xyt, yzt, xzt; after Grid4D), with learned hash probing (Takikawa et al. 2023): 24 levels
per grid, 192 features per coordinate. On Globe-LFMC 2.0 it reached MAE 11.7 percentage points and R² 0.783
from coordinates and a species embedding alone, against 12.6 and 0.72 for Galileo with remote sensing.

In code (`encoders/spacetime/earth4d.py`), `Earth4D` has two channels. The absolute channel hashes where
and when an observation is; it is a memory of the places in training. The relative channel
(`enable_relative=True`, `Earth4D.encode_relative`) hashes the offset $(\Delta N, \Delta E, \Delta z,
\Delta t)$ between two observations, is invariant to absolute position, and is what transfers. `core/fusion.py`
builds on both: `SpaceTimeField` encodes neighbor offsets with the relative channel, `DeepEarth.encode`
refines latent tokens by cross-attention (a Perceiver-style bottleneck), `DeepEarth.decode_field` reads any
variable at any query position, and `DeepEarth.reconstruction_loss` hides random subsets and scores their
reconstruction. For continuous variables `DeepEarth._reconstruction_error` is one minus the cosine of
mean-centered vectors: scale-free, which suits embeddings but cannot hold water to the millimeter.

The Entropy4D note generalizes the encoder to a population of perceptive fields with learned position,
orientation, extent and type (hash lattices and equivariant harmonic stacks) and trains by predicting
future embeddings (a joint-embedding predictive objective). Two ideas carry over now: an observation is
encoded over its own space-time extent, not at a point, and the relative channel, not the absolute, is the
transferable one. The focusing operator and the equivariant harmonic types are not needed for a first
hydrological model and are left out.

### 4.4 Adjacent work that settles design choices

- **Learning parameters, not states, transfers.** Tsai et al. (2021) trained a network to output the
  parameters of a process model from national attributes, end to end through the model; it transferred to
  ungauged places better than per-site calibration, and improved as data grew. Feng et al. (2022) extended
  it to differentiable models whose streamflow skill approaches LSTMs while keeping mass balance and
  interpretable fluxes. Shen et al. (2023) review the approach. This is the design of section 5.
- **Ungauged prediction is solved for streamflow at basin scale.** LSTMs trained across hundreds of basins
  predict unseen ones (Kratzert et al. 2019), and globally (Nearing et al. 2024). They are the bar for
  basin streamflow, and they do not resolve anything inside a basin, which is what a site needs.
- **Local conservation matters more than global.** DUALFloodGNN (Acosta et al. 2025) predicts fluxes on
  edges and enforces per-node balance, with the gain quoted in 4.1.

## 5. The minimum viable deep simulator

### 5.1 Principle

The process model integrates water; the network only sets its parameters and one missing flux. Every
learned quantity has a name, a unit, a physical range and one observation that identifies it. If a learned
parameter reaches the edge of its range at many sites, that is a finding about the physics, reported, not
a reason to widen the range.

### 5.2 The graph

**Cells** $\mathcal V_g$: the process model's ground cells (0.5 m at Harvard), with their states.

**Mesh nodes** $\mathcal V_m$: a square lattice at $\Delta_0 = 2$ m over the site plus a 100 m buffer (16
cells per node at 0.5 m; about 1,000 nodes an acre).

**Multimesh edges** $\mathcal E_{mm}$: GraphCast's multimesh on a square lattice. Level $k = 0, 1, \dots,
K$ keeps the nodes whose lattice indices are both multiples of $2^k$ and joins each to its 8 neighbors at
spacing $2^k \Delta_0$; the union over levels is the edge set, so coarse nodes are a subset of fine ones,
exactly as in the icosahedral hierarchy. $K = 7$ reaches 256 m. Edges are bidirectional.

**Flow edges** $\mathcal E_f$: directed, from each mesh node to its downslope neighbors on the conditioned
2 m DEM, with Quinn multiple-flow-direction weights $w_{ij} = (\tan\beta_{ij})^{1.1} L_{ij} / \sum_k
(\tan\beta_{ik})^{1.1} L_{ik}$. The DEM is conditioned with the storm model's breach and fill
(`models/hydro/terrain.breach_depressions`).

**Encoder and decoder edges**: each cell sends to its containing mesh node and receives from the 4 mesh
nodes around it (the square-lattice analog of GraphCast's containing triangle).

**Features** (all national, all relative; no absolute coordinate anywhere):

| on | features |
|---|---|
| cell | elevation above the lowest cell within 100 m; slope; plan and profile curvature; log contributing area; height above nearest drainage; six-class one-hot; canopy height, first-return gap fraction and plant area index from LiDAR; SSURGO/POLARIS $K_s$, $\theta_s$, $\theta_{fc}$, $\theta_{wp}$, $\lambda$, $\psi_f$, restrictive depth; annual sky-view factor and annual solar irradiation from the solar product; mean 2 m wind speed from the wind product |
| cell, climate | from the national forcing record at the site: mean annual rain, reference ET and their ratio; mean and amplitude of LAI |
| edge | $\Delta N$, $\Delta E$, $\Delta z$ and length; for flow edges also $w_{ij}$ and $\tan\beta_{ij}$ |

Every scalar is standardized by the training sites' statistics, as in both GraphCast and HydroGraphNet.

**Time step**: the first milestone has none; the network runs once per site and returns static
parameters. The dynamic closure of the second milestone runs once a day with daily forcing aggregates and
the day-start state, and the process model runs its 24 hourly steps under it.

### 5.3 The closure

The decoder returns, per cell $i$, six raw outputs $u_{i,1..6}$ mapped into bounded physical ranges by
$\ell + (h - \ell)\,\sigma(u)$ (log-space for the multipliers):

| parameter | range | acts on | identified by |
|---|---|---|---|
| $m_K$, saturated conductivity multiplier | $10^{-1}$ to $10^{1}$ | infiltration and drainage | soil water recession after rain |
| $Z_r$, root depth | 0.3 to 3.0 m | plant-available water | deep soil water, late-summer ET |
| $m_T$, transpiration coefficient multiplier on $K_{cb}$ | 0.5 to 1.5 | transpiration | ET |
| $p$, stress onset (FAO-56 depletion fraction) | 0.2 to 0.8 | water stress | ET in dry spells |
| $m_S$, interception capacity multiplier on $S_{\max}$ | 0.5 to 2.0 | interception | throughfall |
| $a$, lateral anisotropy $K_{lat}/K_s$ | 1 to 1,000 | lateral subsurface flow | downslope soil water; streamflow |

The ranges are the literature spread for each quantity, not tuning knobs: SSURGO $K_s$ is uncertain by
about an order of magnitude, FAO-56 lists $p$ from 0.3 to 0.7 across crops, and hillslope anisotropy runs
from near 1 to several hundred.

The one added flux is hillslope Darcy flow in layer 2 along flow edges (at the cells' own flow network),
the kinematic subsurface flow of TOPMODEL (Beven and Kirkby 1979):

$$Q_{ij} = 1000\,\Delta t\,w_{ij}\,a_i\,m_{K,i}\,K_{s,i}\,\tan\beta_{ij}\,\frac{h_i}{\Delta x}, \qquad
h_i = (Z_{r,i} - Z_e)\,\frac{(\theta_{2,i} - \theta_{fc,i})^+}{\theta_{s,i} - \theta_{fc,i}}$$

in mm of water over cell $i$ per hour. Outflows are scaled so that $\sum_j Q_{ij}$ never exceeds the
drainable water $1000(\theta_{2,i} - \theta_{fc,i})^+(Z_{r,i} - Z_e)$; a receiver takes into layer 2 what
its room allows, and the rest exfiltrates into its ponded store $w_j$, which is return flow and the
saturation-excess runoff the model cannot produce today. On equal-area cells

$$\sum_i \Delta S_i^{lat} = \sum_i \Big(\sum_k Q_{ki} - \sum_j Q_{ij}\Big) = 0$$

exactly, because every $Q_{ij}$ is subtracted once and added once. The other five parameters change rates
inside terms that already debit a store, so the budget identity of section 3.1 holds unchanged:

$$P = \Delta S + ET + R_{out} + D_{deep}$$

per cell and site, to float precision, with no penalty term and no tolerance to tune.

### 5.4 The processor

Each layer is the interaction network used by MeshGraphNet and GraphCast, with typed edges (flow and
multimesh have separate MLPs):

$$e'_{ij} = e_{ij} + \mathrm{MLP}^{E}_{\tau(ij)}\big([e_{ij}, v_i, v_j]\big), \qquad
v'_j = v_j + \mathrm{MLP}^{V}\Big(\Big[v_j, \sum_{i:\,\tau=f} e'_{ij}, \sum_{i:\,\tau=mm} e'_{ij}\Big]\Big)$$

with 2-layer MLPs, SiLU, LayerNorm and sum aggregation. Size: 8 unshared layers, latent 64. The decoder is
a 2-layer MLP per cell from the 4 surrounding mesh nodes and the cell's own features to $u_{1..6}$.
This is 0.33 million parameters measured with one edge type (section 10), about 0.45 million with two.
Eight layers of the multimesh pass information across 256 m at the coarse level, beyond a 10 acre site. HydroGraphNet's 15 layers at latent 128 are sized for a 4,787-node emulation task; the
closure has 6 outputs per cell, and the small model is the better start, grown only if validation asks.

### 5.5 Training

Loss, per observation type $m$ with observation operator $H_m$ (section 6.3) and error $\sigma_{m,o}$:

$$\mathcal L = \sum_m \frac{1}{N_m}\sum_{o} \frac{\big(H_m(\hat x)_o - y_o\big)^2}{\sigma_{m,o}^2}$$

the Gaussian likelihood of the observations. It plays the role of GraphCast's per-variable inverse-variance
weights, with each variance an instrument's stated error rather than a tuning choice. For ET,
$\sigma^2 = \sigma_{rand}^2 + \big((ET_{closed} - ET_{raw})/2\big)^2$, so days with a large closure gap
count for less. For soil water, each sensor's term carries its own additive offset, fit with the
parameters and never transferred, so a sensor's calibration cannot be learned into the soil. There is no
physics penalty (section 5.3) and no bound penalty (the sigmoid enforces it).

Gradients flow from the loss through the hourly water balance to the parameters. The local step is
already differentiable PyTorch; the run-on cascade is sequential and runs without gradient, passing its
arrivals to each cell as an input. Following GraphCast's curriculum: first 30-day windows started from the
process model's own state, then windows grown to a full year, with gradient checkpointing per day. Adam,
learning rate $10^{-3}$ with cosine decay, weight decay $10^{-4}$, early stopping on the validation year.

The ablations that decide what stays: the same closure from a per-cell MLP with no graph (does message
passing earn its place?), without the lateral flux ($Q \equiv 0$), and B2 (does a network from national
inputs match per-site calibration?).

### 5.6 What is adopted and what is not

| component | from | adopt? | why |
|---|---|---|---|
| interaction-network block (edge MLP, node MLP, sum, LayerNorm, residual) | MeshGraphNet via `physicsnemo...MeshGraphNet`; GraphCast `DeepTypedGraphNet` | yes | the proven core of both; re-implemented in about 100 lines of PyTorch with `index_add_`, to avoid the torch-scatter and Transformer Engine dependencies |
| typed edge sets | GraphCast `TypedGraph` | yes | flow and multiscale edges mean different things |
| multimesh | GraphCast | yes, on a square lattice | long-range coupling in few layers; GraphCast's ablation shows it matters |
| grid-to-mesh and mesh-to-grid bipartite graphs | GraphCast | yes | lets the 0.5 m cells and a 2 m processor differ in resolution |
| relative edge features in the receiver's frame | GraphCast | yes | translation invariance |
| residual prediction, outputs scaled by difference variance | GraphCast `InputsAndResiduals`; HydroGraphNet | second milestone only | the static closure predicts parameters, not increments; the dynamic closure will predict daily increments of its parameters this way |
| rollout curriculum with backpropagation through time | GraphCast | yes | cheap here, since the process model carries the state |
| pushforward trick | HydroGraphNet, Brandstetter et al. 2022 | fallback | only if full backpropagation through the year proves too costly |
| inverse-variance loss weights | GraphCast `weighted_mse_per_level` | yes, as instrument error | a likelihood with measured errors |
| causal, frozen train/validation/test split | GraphCast | yes | section 2.2 |
| absolute coordinates as node features | both | no | memorizes place; breaks transfer |
| k-nearest-neighbor graph | HydroGraphNet | no | blind to flow direction |
| global soft mass penalty | HydroGraphNet `compute_physics_loss` | no | replaced by conservation by construction |
| Fourier KAN node encoder | HydroGraphNet `KolmogorovArnoldNetwork` | no | one layer of a deep MLP stack does not make the model interpretable; named parameters do |
| emulating a hydraulic solver as the target | HydroGraphNet | no | the target is observations |
| icosahedral sphere mesh, 474-channel atmosphere state, 36.7 M parameters | GraphCast | no | site scale and six outputs |

## 6. The space-time embedding

The first milestone needs no embedding: its inputs are national layers resampled onto the cells, and its
targets are compared through the observation operators of 6.3. The embedding enters in the second
milestone, when observations themselves become inputs, to infer a site's state before a forecast and to
use coarse remote sensing at its own footprint. It is specified now so that the operators, the
coordinates and the loss are the same in both.

### 6.1 Every modality in its own coordinates

Each observation is a tuple $(y_o, \mathbf c_o, \mathbf s_o, m_o)$: its value, the center $\mathbf c_o =
(N, E, z, t)$ in the site's local frame (meters north and east, height above the bare earth, hours), its
extent $\mathbf s_o$ (the footprint's size in each axis) and its modality:

| modality | center | extent |
|---|---|---|
| NEON soil water sensor | plot location, depth below ground, time | 0.1 m × 0.1 m × sensor length × 30 min |
| tower LE | footprint centroid, measurement height, time | the flux footprint (weights, 6.3) × 30 or 60 min |
| throughfall trough | trough location, 0.5 m, time | trough area × 30 min |
| NISAR soil moisture | pixel center, 0 to 5 cm, overpass | 200 m × 200 m × 5 cm × one overpass |
| SMAP soil moisture | pixel center, 0 to 5 cm, overpass | 9 km × 9 km × 5 cm × one overpass |
| MODIS or VIIRS LAI | pixel center, canopy height, composite | 500 m × 500 m × canopy × 4 or 8 days |
| PlanetScope reflectance (later) | pixel center, canopy top, acquisition | 3 m × 3 m × canopy × one scene |
| stream discharge | gauge, 0, time | the catchment × 15 min |

### 6.2 Encoding

With the absolute channel off, a query at cell $i$ and time $t$ sees each observation by its offset. A
point encoding would treat a 9 km pixel as a point; instead each observation is encoded over its extent,
by averaging Earth4D's relative features over samples of the footprint (the integrated positional
encoding of mip-NeRF, Barron et al. 2021, which Entropy4D's perceptive fields generalize):

$$\bar{\mathbf e}(o \mid i, t) = \frac{1}{K}\sum_{k=1}^{K} \mathrm{E4D}_{rel}\big(\mathbf c_o + \mathbf s_o
\odot \boldsymbol\xi_k - (\mathbf x_i, t)\big), \qquad \boldsymbol\xi_k \in [-\tfrac12, \tfrac12]^4
\text{ stratified}$$

A coarse observation then has a smooth, low-frequency encoding and a point sensor a sharp one, which is
what its information content warrants. The token is $\mathbf W_{m} y_o + \bar{\mathbf e} + \mathbf
t_m$ (value projection, position, modality type), as in `DeepEarth.encode`. Relative window and finest
resolution for the site scale: $(\pm 512, \pm 512, \pm 64$ m, $\pm 30$ days$)$ and $(0.5, 0.5, 0.25$ m, 1 h$)$.

### 6.3 Observation operators and reconstruction

Every modality is reconstructed through an operator $H_m$ that maps the model state to what the
instrument would have recorded, and the loss of 5.5 compares them there, never on the model grid:

- **Point sensor at depth $[z_1, z_2]$:** the layer water contents of the sensor's cell, weighted by
  overlap: $H = \sum_\ell \theta_\ell\,|[z_1, z_2] \cap \ell| / (z_2 - z_1)$, averaged over the 30 min
  window.
- **Tower ET:** the footprint-weighted mean of cell ET, $H = \sum_i \phi_i(t)\,ET_i(t)$, with
  $\phi_i$ from the flux footprint parameterization of Kljun et al. (2015) for the hour's $u_*$, wind
  direction and measurement height above displacement, neutral stability where the Obukhov length is not
  published. Scoring the tower against its own cell, as today, ignores where its flux comes from.
- **Throughfall:** the cell's rain minus its interception, summed over the trough's cells.
- **Area remote sensing (NISAR, SMAP, LAI):** the area-weighted mean over the pixel's cells of the top
  5 cm (or LAI), over the overpass or composite window. A SMAP pixel is 9 km across and a site is a few
  hundred meters, so SMAP enters as the site-mean anomaly against the pixel anomaly, with a
  representativeness error added to its 0.04 m³ m⁻³.
- **Streamflow:** the sum of exfiltration and surface outflow over the gauge's catchment, routed by the
  storm model (second milestone).

Masked reconstruction is the regularizer: in the second milestone, when observations also enter as tokens
to infer the state (for forecasting), a random share of tokens is hidden and must be reconstructed through
$H_m$, which is `DeepEarth.reconstruction_loss` with the cosine replaced by the Gaussian likelihood
above.

## 7. Forcing

Solar and wind enter as spatially distributed forcing, exactly as the validated products deliver them:

- **Sun.** Each cell's hourly shortwave $R_s$ is the solar product's value on its ground normal: beam from
  the hour's DNI through the 3D scene (terrain, buildings, canopy transmittance from LiDAR), isotropic sky
  diffuse times the sky-view factor, and ground reflection, driven by NSRDB or the tower. Net radiation
  $R_n = (1-\alpha)R_s + \varepsilon(L_{in} - \sigma T^4)$ feeds $ET_0$. Canopy transmittance is validated
  for PAR against Harvard's HF004 sensor at 12.7 m (summer 0.163 against 0.16); the known open error is that
  shortwave is applied with PAR's extinction, which reads understory shortwave low by a factor of 1.7 to 2.1
  in summer until the two-band fix lands.
- **Wind.** Each cell's 2 m wind is the unit-speed basis of the mass-consistent solver at the hour's
  heading, scaled by the hour's reference speed. Point wind is not yet validated independently; it enters
  $ET_0$'s aerodynamic term only.

Both enter the closure as static features (annual irradiation, sky-view factor, mean wind) and, in the
dynamic closure, as daily sums. The network never recomputes them.

## 8. Inputs: national only

The rule: an input must exist for any site in the United States (then the world) at inference time.
Observations that exist only at research sites are targets, never inputs. Tower forcing is used where a
tower is near, as production does today, but every proof runs on national forcing.

| candidate | resolution | revisit | latency | coverage | license | verdict |
|---|---|---|---|---|---|---|
| 3DEP LiDAR and 1 m DEM | ≥ 2 pulses m⁻², 1 m | one epoch, years | none | most of CONUS | public domain | **ESSENTIAL NOW**: terrain, flow graph, canopy structure |
| SSURGO, POLARIS where blank | 1:12k to 1:24k polygons; 30 m, 6 depths | static | none | CONUS; SSURGO blank under most metro cores | public domain; POLARIS open | **ESSENTIAL NOW**: every soil parameter |
| NAIP (six land-cover classes) | 0.6 m, 4 bands | 2 to 3 years | months | CONUS | public domain | **ESSENTIAL NOW**: which terms apply per cell |
| AORC rain and air; MRMS for recent hours | about 1 km, hourly; 1 km, 2 min | continuous | AORC retrospective; MRMS minutes | CONUS | public domain | **ESSENTIAL NOW**: rain, temperature, humidity, pressure, longwave, wind |
| NSRDB | 4 km, 30 min (2 km, 5 min from 2018) | continuous | about a year | Americas | open | **ESSENTIAL NOW**: the sun, through the solar product |
| MODIS LAI, then VIIRS | 500 m | 4 or 8 days | days | global | public domain | **ESSENTIAL NOW**: leaves through the year |
| PlanetScope | 3 m, 4 or 8 bands | near daily | about a day | global; 5 million km² awarded to UC Berkeley investigators through NASA CSDA | commercial; CSDA terms limit use to the funded research | **LATER, first addition**: per-crown greenness in place of site-mean LAI, kept only if held-out ET and soil water improve; HLS (30 m, open) is the fallback |
| NISAR soil moisture (SME2) | 200 m posting (400 m in 5+5 MHz modes), 0 to 5 cm | twice per 12 days | 72 h | global land, flagged where vegetation water > 5 kg m⁻², urban, frozen, snow | open | **LATER, second milestone**: a footprint constraint where unflagged (grasslands, shrublands, crops); provisional in 2026, accuracy goal 0.06 m³ m⁻³ |
| SMAP | 36 km, 9 km enhanced, 0 to 5 cm | 2 to 3 days | about 2 days | global since 2015 | open | **LATER, second milestone**: regional anomaly constraint, 0.04 m³ m⁻³ requirement; the 1 to 3 km SMAP/Sentinel-1 product is not retrieved where vegetation water exceeds 3 kg m⁻² |
| SWOT | 100 m rivers (50 m goal), lakes > 250 m × 250 m | 21 days | days | global | open | **NO**: sees neither soil water nor small streams |
| PACE OCI | about 1.2 km, 340 to 890 nm hyperspectral plus 7 SWIR bands | 1 to 2 days | days | global | open | **NO for now**: one pixel covers a whole site many times over; its land indices (PRI, CCI, NDII) are a later regional canopy-stress signal |
| NEON ground sensors | points at 81 sites | 1 to 30 min | a month | research sites | open, token needed | **TARGET, ESSENTIAL NOW** |
| NEON AOP | 1 m hyperspectral, LiDAR | yearly at a few sites | months | NEON sites only | open | **NO as input** (not national) |
| FLUXNET / AmeriFlux towers | points | 30 min | months to years | about 300 US towers | CC-BY-4.0 | **TARGET, ESSENTIAL NOW**; forcing only where near |
| USGS NWIS gauges | points | 15 min | real time | national | public domain | **TARGET, second milestone** (streamflow) |
| 3D Hydrography Program | derived from 3DEP | as 3DEP | | published quarterly, complete coverage planned for 2032; NHDPlus HR everywhere today | public domain | **LATER**: stream burning at catchment scale |
| iNaturalist | points, meters to kilometers | opportunistic | real time | global, biased to people | mixed, often non-commercial | **NO** now |
| Forest Service FIA | plots, coordinates perturbed by up to about 1.6 km | 5 to 10 years | years | national | public, fuzzed locations | **NO** at site scale; later as a regional species prior |
| phylogenomic species priors | per species | static | | where species are known | open | **LATER**, with species-specific ET (section 11) |

The smallest essential set is seven: terrain and structure (3DEP), soil (SSURGO, POLARIS), surface
class (NAIP), rain (AORC, MRMS), air (AORC), sun (NSRDB), and leaves (MODIS, then VIIRS). Each later input
enters by ablation and stays only if held-out skill improves.

**NISAR, PACE and SWOT for soil moisture and the flow operator.** Of the three, only NISAR measures soil
water, and only in the top 5 cm, at 200 m, about twice in 12 days, where vegetation holds less than
5 kg m⁻² of water: not under closed forest (Harvard, where a flagged NISAR cell or SMAP's 9 km pixel can
check only the timing of wetting and drying, not the level), yes over grassland and shrubland (Konza,
Santa Rita). It can calibrate the model only through the observation operator of 6.3, never as a per-cell truth,
and its 0.06 m³ m⁻³ goal is comparable to the model's own error today. PACE sees land at about 1.2 km, so a
10 acre site is a few percent of one pixel. SWOT measures water surface elevation on rivers wider than about
100 m and lakes larger than 250 m on a side, neither of which a site-scale soil model contains. For a NASA
proposal the defensible experiment is NISAR: at NEON grassland and shrubland sites where retrievals are
unflagged, does adding NISAR as a footprint constraint improve held-out skill at NEON probes from 6 to 30
cm, against the same model without it; and does the model's 0.5 m field, averaged to NISAR's footprint,
serve as an independent check on NISAR in the 0.5 to 200 m range no probe network samples. SMAP is the
long-record partner in both. PACE and SWOT would be passengers in such a proposal, not instruments.

## 9. Training and test sites

All sites are NEON towers with FLUXNET products, NEON soil water and precipitation, unless marked; the
tower code follows the NEON code.

**10 sites (first milestone).** Test: HARV (US-xHA; plus US-Ha1, 1991 on), KONZ (US-xKZ; plus US-Kon),
SRER (US-xSR; plus US-SRM, US-SRG). Train: BART (US-xBR), SCBI (US-xSC), ORNL (US-xRN), WREF (US-xWR),
OSBS (US-xSB), CPER (US-xCP), JORN (US-xJR). Deciduous, evergreen, grassland and shrubland each appear
in training and in test; throughfall is measured at all ten.

**20 sites.** Add UNDE (US-xUN), TREE (US-xTR), MLBS (US-xML), GRSM (US-xGR), TALL (US-xTA), JERC
(US-xJE), CLBJ (US-xCL), ONAQ (US-xNQ), SOAP (US-xSP), UKFS (US-xUK). Streams for the second milestone:
HOPB (HARV), KING (KONZ), SYCA (SRER), WALK (ORNL), POSE (SCBI), LECO (GRSM), MAYF (TALL), PRIN (CLBJ),
REDB (ONAQ), MART (WREF).

**50 sites.** Add the remaining NEON towers: ABBY, BLAN, DCFS, DELA, DSNY, MOAB, NOGP, OAES, RMNP, SERC,
SJER, STEI, STER, WOOD, YELL, and in Alaska BONA, DEJU and HEAL; and twelve long AmeriFlux records with
soil water profiles: US-MMS, US-WCr, US-Var, US-ARM, US-Wkg, US-Whs, US-GLE, US-ChR, US-MOz, US-NR1,
US-Me6, US-Ro4.

HARV, KONZ and SRER stay the fixed test set at every size, and every tower within 50 km of them is a test
target too, never training: US-Ha1, US-Kon, KONA (US-xKA, 5 km from KONZ), US-SRM, US-SRG and US-Aud. At
20 and 50 sites, model selection is 5-fold cross-validation over groups of NEON domains.

## 10. Compute and cost

Measured on one NVIDIA L4 (24 GB) with PhysicsNeMo's MeshGraphKAN:

| model | parameters | nodes (edges, k = 4) | one pass, fp32 | its peak memory | one training step | its peak memory |
|---|---|---|---|---|---|---|
| HydroGraphNet, 15 layers × 128 | 2,318,722 | 4,787 (19,148) | 15 ms | 0.09 GB | 46 ms | 1.2 GB |
| | | 50,000 (200,000) | 0.22 s | 0.8 GB | 0.59 s | 11.9 GB |
| | | 200,000 (800,000) | 0.91 s | 3.2 GB | out of memory | > 24 GB |
| | | 800,000 (3,200,000) | 3.7 s | 12.5 GB | out of memory | |
| lean, 8 layers × 64 | 327,490 | 4,787 | 5.7 ms | 0.06 GB | 26 ms | 0.35 GB |
| | | 50,000 | 67 ms | 0.4 GB | 0.16 s | 3.4 GB |
| | | 200,000 | 0.28 s | 1.6 GB | 0.68 s | 13.4 GB |
| | | 800,000 | 1.16 s | 6.3 GB | out of memory | |

A training step is forward, backward and Adam on the whole graph; bfloat16 inference saves 0 to 20%. Training
memory is about 240 kB per node for HydroGraphNet and 67 kB for the lean model, so full-graph training
fits about 90,000 and 350,000 nodes on a 24 GB GPU. This is why the processor runs on a 2 m mesh and not on
the 0.5 m cells, which are 16 times as many.

The processor is cheap next to the process model it drives. The water balance runs a year of a 10 acre
site with its 100 m run-on buffer (about 0.6 million 0.5 m cells) in 38 to 45 s on one L4, all record
years batched. Estimates for the closure, from these rates:

- **Model:** about 0.45 million parameters, 5 times smaller than HydroGraphNet. At a 10 acre site with
  its buffer the 2 m mesh has about 40,000 nodes, so one pass is under 0.1 s and one full-graph training
  step under 0.2 s.
- **Training, first milestone:** 7 training sites × 5 years, each site a 400 m square around the tower
  (its footprint and the five soil plots) at 1 m cells, about 160,000 cells. A site-year is then about
  10 s forward and 30 s with the backward pass, so 35 site-years take about 20 min an epoch; 20 epochs,
  the three ablations and B2 make 10 to 40 L4-hours. Training at 1 m and running at 0.5 m is checked at
  two training sites before the test.
- **Training at 50 sites:** 40 to 200 L4-hours.
- **Inference per site:** one network pass per site for the static closure (seconds), and the lateral
  flux adds one sparse matrix-vector product per hour to the water balance, under 10% of its time.

## 11. Deep operators: the blueprint

Every later operator follows the same pattern: a physical equation, named and bounded parameters from the
network, conservation by construction, and one observation that identifies each parameter.

- **Evapotranspiration.** Replace $K_{cb}$ with Penman-Monteith and a canopy conductance from the Medlyn
  stomatal model, $g_s = g_0 + 1.6(1 + g_1/\sqrt{D})A/C_a$, with $g_1$ per cell from the network; the
  identifying observations are LE and, where towers publish it, GPP.
- **Species-specific ET.** $g_1$ and root depth from a species embedding (DeepEarth's `SpeciesGraph`,
  refined over the phylogeny), with plant-functional-type values (Lin et al. 2015) as the prior, once
  per-crown species maps exist.
- **Carbon and phenology.** The Medlyn model couples $g_s$ to assimilation $A$, so the same $g_1$ gives
  GPP, identified by tower GPP; leaf-out and leaf-fall per crown become learned dates driven by
  temperature and day length, identified by PlanetScope greenness once it is an input.
- **Heat.** The surface energy balance $R_n - G = H + LE$ with $H = \rho c_p (T_s - T_a)/r_a$, solved
  for $T_s$ per cell, with learned corrections to $r_a$ and $G$; identified by tower LW_OUT and ECOSTRESS
  land surface temperature.
- **Advection.** Heat and vapor carried between cells by the wind product's velocities as upwind
  finite-volume fluxes on the same graph, with a learned eddy diffusivity; conserved by the same
  antisymmetry as 5.3.

None of these starts before the first milestone's proof holds.

## 12. Integration with DeepEarth

The code lives in this repository, beside the storm model it extends. New files:

| file | contents |
|---|---|
| `models/hydro/balance.py` | The hourly per-cell water balance as a differentiable module: `Cells` and `State` dataclasses (the per-cell constants and states of 3.1), `local_step(state, cells, rs, u2, air)`, `rain_step(state, cells, rain, network)`, `class WaterBalance(torch.nn.Module)` with `forward(cells, forcing, hours, state0) -> (states, fluxes)` and a `budget()` receipt. Equations exactly as 3.1, published with the simulator's method notes. |
| `models/hydro/graph.py` | `class CellGraph` (cells, mesh nodes, typed edge index tensors, edge features); `multimesh(shape, levels)`; `flow_edges(dem, dx)` returning MFD weights and slopes; `grid2mesh(cells, mesh)`, `mesh2grid(cells, mesh)`; `features(site) -> (cell_x, mesh_x, edge_x)` from national layers only. Uses `terrain.breach_depressions`. |
| `models/hydro/closure.py` | `RANGES` (the table of 5.3); `class Closure(torch.nn.Module)`: encoder, `GraphProcessor`, decoder, `forward(graph) -> Parameters`; `Parameters.apply(cells) -> Cells`; `lateral_flux(state, cells, graph, a, m_K) -> Q` with the exact bounds of 5.3. |
| `models/hydro/observe.py` | Observation operators: `PointSensor(cell, z1, z2)`, `Footprint(weights)`, `ffp_weights(ustar, wd, z_m, d, z0, grid)` after Kljun et al. 2015, `Throughfall(cells)`, `Pixel(cells, weights)`; each `__call__(states, fluxes) -> values`. |
| `models/hydro/train_closure.py` | Loss of 5.5, the curriculum, the split of 2.2 as a frozen table, the metrics of 2.3, and a JSON receipt per run. |
| `core/graph.py` | `class InteractionBlock` and `class GraphProcessor(n_layers, latent, edge_types)`: the block of 5.4, pure PyTorch with `index_add_`. |

Changes to existing code:

- `encoders/spacetime/earth4d.py`: add `Earth4D.encode_footprint(center_offset, extent, samples=16)`, the
  integrated encoding of 6.2, on top of `encode_relative`.
- `core/fusion.py`: `Variable` gains `loss="cosine"|"gaussian"` and `sigma`; `DeepEarth._reconstruction_error`
  uses the Gaussian likelihood for physical quantities. Nothing else changes; the second milestone
  instantiates `DeepEarth` with the relative channel only (`SpaceTimeField`) and the absolute channel off.
- `models/hydro/validate.py`: `nse` and `kge` are reused for streamflow as they are; soil water and ET
  scores (ubRMSE, anomaly r) are added beside them.
- `models/README.md`: a row for the closure once it is validated, and not before.

## 13. What is essential next

1. NEON soil water, throughfall and precipitation for the nine sites beyond HARV, 2016 to 2025, and
   NEON's eddy-covariance bundle for 2025 ET at all ten.
2. The known physics first: upward capillary flux between the two layers and a sensor-depth observation
   operator, then B1 re-scored at all ten sites. The deep model is measured against that B1.
3. `balance.py` published and differentiable, with a test that its gradients match finite differences.
4. `graph.py`, `core/graph.py` and `closure.py` with the six static parameters and the lateral flux; the
   budget test at $10^{-6}$.
5. **First milestone, about four weeks:** trained on the seven training sites, scored once on HARV, KONZ
   and SRER in 2024 to 2025 against B0, B1 and B2, with the three ablations of 5.5. Published whether it
   passes or fails.
6. Then, in order and each by ablation: 20 sites and the streams with NISAR and SMAP as footprint
   constraints and forecasting from an inferred state; PlanetScope; the dynamic daily closure.

## References

Allen, R. G., Pereira, L. S., Raes, D., Smith, M. (1998). Crop evapotranspiration. FAO Irrigation and
Drainage Paper 56. [link][fao56]

Acosta, C. M., Herath, H. M. V. V., Lim, J. Y., Saha, A., Rasnayaka, S., Marshall, L. (2025).
DUALFloodGNN: physics-informed graph neural network for operational flood modeling. arXiv:2512.23964.
[link][dual]

ASCE-EWRI (2005). The ASCE standardized reference evapotranspiration equation.

Barron, J. T., Mildenhall, B., Tancik, M., Hedman, P., Martin-Brualla, R., Srinivasan, P. P. (2021).
Mip-NeRF: a multiscale representation for anti-aliasing neural radiance fields. ICCV. [link][mipnerf]

Bates, P. D., Horritt, M. S., Fewtrell, T. J. (2010). A simple inertial formulation of the shallow water
equations for efficient two-dimensional flood inundation modelling. J. Hydrol. 387, 33-45.

Beven, K. J., Kirkby, M. J. (1979). A physically based, variable contributing area model of basin
hydrology. Hydrol. Sci. Bull. 24, 43-69.

Brandstetter, J., Worrall, D., Welling, M. (2022). Message passing neural PDE solvers. ICLR.
[link][mppde]

Brooks, R. H., Corey, A. T. (1964). Hydraulic properties of porous media. Hydrology Papers 3, Colorado
State University.

Chaney, N. W., et al. (2019). POLARIS soil properties: 30-m probabilistic maps of soil properties over the
contiguous United States. Water Resour. Res. 55, 2916-2938.

Feng, D., Liu, J., Lawson, K., Shen, C. (2022). Differentiable, learnable, regionalized process-based
models with multiphysical outputs can approach state-of-the-art hydrologic prediction accuracy. Water
Resour. Res. 58, e2022WR032404.

Kljun, N., Calanca, P., Rotach, M. W., Schmid, H. P. (2015). A simple two-dimensional parameterisation for
Flux Footprint Prediction (FFP). Geosci. Model Dev. 8, 3695-3713. [link][ffp]

Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., Nearing, G. S. (2019). Toward
improved predictions in ungauged basins: exploiting the power of machine learning. Water Resour. Res.
55, 11344-11354.

Lam, R., et al. (2023). Learning skillful medium-range global weather forecasting. Science 382,
1416-1421. [paper][graphcast], [code][wn]

Legel, L., Huang, Q., Voelker, B., Neamati, D., Johnson, P. A., Bastani, F., Rose, J., Hennessy, J. R.,
Guralnick, R., Soltis, D., Soltis, P., Wang, S. (2026). Self-supervised multi-modal world model with 4D
space-time embedding. World Modeling Workshop, Mila. arXiv:2603.07039. [link][deepearth]

Lin, Y.-S., et al. (2015). Optimal stomatal behaviour around the world. Nat. Clim. Change 5, 459-464.

Medlyn, B. E., et al. (2011). Reconciling the optimal and empirical approaches to modelling stomatal
conductance. Glob. Change Biol. 17, 2134-2144.

Mein, R. G., Larson, C. L. (1973). Modeling infiltration during a steady rain. Water Resour. Res. 9,
384-394.

Müller, T., Evans, A., Schied, C., Keller, A. (2022). Instant neural graphics primitives with a
multiresolution hash encoding. ACM Trans. Graph. 41, 102.

Nearing, G., et al. (2024). Global prediction of extreme floods in ungauged watersheds. Nature 627,
559-563.

Ogden, F. L., Saghafian, B. (1997). Green and Ampt infiltration with redistribution. J. Irrig. Drain.
Eng. 123, 386-393.

Pfaff, T., Fortunato, M., Sanchez-Gonzalez, A., Battaglia, P. W. (2021). Learning mesh-based simulation
with graph networks. ICLR. [link][mgn]

Quinn, P., Beven, K., Chevallier, P., Planchon, O. (1991). The prediction of hillslope flow paths for
distributed hydrological modelling using digital terrain models. Hydrol. Process. 5, 59-79.

Rawls, W. J., Brakensiek, D. L., Miller, N. (1983). Green-Ampt infiltration parameters from soils data.
J. Hydraul. Eng. 109, 62-70.

Shen, C., et al. (2023). Differentiable modelling to unify machine learning and physical models for
geosciences. Nat. Rev. Earth Environ. 4, 552-567.

Taghizadeh, M., Zandsalimi, Z., Nabian, M. A., Shafiee-Jood, M., Alemazkoor, N. (2025). Interpretable
physics-informed graph neural networks for flood forecasting. Comput.-Aided Civ. Infrastruct. Eng.
doi:10.1111/mice.13484. [paper][hgn], [code][hgncode], [data][hgndata]

Takikawa, T., Müller, T., Nimier-David, M., Evans, A., Fidler, S., Jacobson, A., Keller, A. (2023).
Compact neural graphics primitives with learned hash probing. arXiv:2312.17241.

Tsai, W.-P., Feng, D., Pan, M., Beck, H., Lawson, K., Yang, Y., Liu, J., Shen, C. (2021). From
calibration to parameter learning: harnessing the scaling effects of big data in geoscientific
modeling. Nat. Commun. 12, 5988.

Twine, T. E., et al. (2000). Correcting eddy-covariance flux underestimates over a grassland. Agric. For.
Meteorol. 103, 279-300.

von Hoyningen-Huene, J. (1981). Die Interzeption des Niederschlags in landwirtschaftlichen
Pflanzenbeständen. DVWK Schriften 57.

Mission and product sources: NISAR soil moisture [SME2 guide][nisar]; SWOT [science requirements][swot];
PACE OCI land products [LANDVI][pace]; the USGS [3D Hydrography Program][3dhp]; NEON products
[data portal][neon]; Entropy4D developer note [note][e4d].

[fao56]: https://www.fao.org/4/x0490e/x0490e00.htm
[dual]: https://arxiv.org/abs/2512.23964
[mipnerf]: https://arxiv.org/abs/2103.13415
[mppde]: https://arxiv.org/abs/2202.03376
[ffp]: https://doi.org/10.5194/gmd-8-3695-2015
[graphcast]: https://www.science.org/doi/10.1126/science.adi2336
[wn]: https://github.com/google-deepmind/graphcast
[deepearth]: https://arxiv.org/abs/2603.07039
[mgn]: https://arxiv.org/abs/2010.03409
[hgn]: https://doi.org/10.1111/mice.13484
[hgncode]: https://github.com/NVIDIA/physicsnemo/tree/main/examples/weather/flood_modeling/hydrographnet
[hgndata]: https://zenodo.org/records/14969507
[nisar]: https://nisar-docs.asf.alaska.edu/sme2/
[swot]: https://swot.jpl.nasa.gov/system/documents/files/2176_2176_D-61923_SRD_Rev_B_20181113.pdf
[pace]: https://www.earthdata.nasa.gov/data/catalog/ob-cloud-pace-oci-l3m-landvi-3.2
[neon]: https://data.neonscience.org
[e4d]: https://github.com/user-attachments/files/30716504/entropy4d.md
[3dhp]: https://www.usgs.gov/3d-hydrography-program
