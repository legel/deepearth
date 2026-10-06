# maxent_torch: maxent.jar's model fitting on a GPU

`ranges/maxent_torch.py` reimplements the fitting algorithm of maxent.jar 3.4.4 in PyTorch, step for
step, and runs many fits as one GPU batch. It writes the same files as maxent.jar (`.lambdas`,
`*_samplePredictions.csv`, `maxentResults.csv`), so rendering, range cards, evaluation and validation read its
output unchanged. The algorithm is specified, with Java source references, in
[maxent_fitting_spec.md](maxent_fitting_spec.md).

**Status (2026-10-04):** validated against maxent.jar (below). It is selectable with
`--engine torch` in `scripts/run_fit.py`, or `MAXENT_ENGINE=torch` for `modelling.fit_species`. **At the time of this report the default
was `java`; it was changed the same day to `auto` (`torch` where a CUDA GPU exists; see scientific_provenance.md).**

## Design

A species needs 30 maxent.jar runs: 5 β × 5 cross-validation folds, then 5 subsampled replicates at the chosen β.
Each run is maxent.jar's sequential optimizer:

1. pick the feature, or the candidate threshold/hinge feature, with the best bound on loss decrease;
2. update its coefficient;
3. repeat for up to 500 iterations.

This has little parallelism within one run, but every run of a species advances in lock-step. maxent_torch keeps
all runs as one padded batch and executes each iteration as a few dozen float64 GPU kernels. All 50 runs (25 CV
plus the replicates of every β) go in one batch, and only the chosen β's replicates are written.

Parts of the implementation:

* **Data preparation (CPU, numpy).** Its main steps:
  * Runs are prepared the way maxent.jar prepares them: duplicate coordinates are removed; presences not already
    present as background rows are added to the background; layers are sorted alphabetically (ParamsPre sorts
    them, which fixes feature order and tie-breaking).
  * The knot grid uses maxent.jar's significant-digit grouping (`precision`).
  * Sample statistics and regularization come from maxent.jar's tables.
  * Cross-validation folds come from maxent.jar's `java.util.Random(0)` draws (bit-identical).
  * Replicate test sets use maxent.jar's procedure with a fixed seed.
* **Candidate features.** These are the threshold, forward-hinge and reverse-hinge candidates of every variable.
  * Each run holds them as one flat array in maxent.jar's selection order, so `argmin` gives Java's
    first-best tie rule.
  * Their model expectations come from per-variable prefix sums of the density over value-sorted points, summed
    from the largest value down as Java does.
  * A fused, torch-compiled kernel computes every candidate's expectation and loss bound (`goodAlpha` and
    `deltaLossBound`).
* **Optimizer.** It reproduces maxent.jar's `Sequential` exactly, including the stale-expectation bookkeeping
  that decides which feature is selected:
  * sign-preserving Newton steps for linear and hinge features, and bound-optimal steps for threshold features;
  * the 1/50, 1/10, 1/3 damping of early iterations;
  * an undo plus ×4 line search when Newton disappoints;
  * a joint Newton step along the recent change every 30 iterations, with zero snapping;
  * the `featuresToUpdate` refresh schedule;
  * the 20-iteration, 1e-5 convergence test.
* **Outputs.** These are maxent.jar's formats:
  * Java `Double.toString` for numbers.
  * The `.lambdas` trailer.
  * Training and test AUC with ties counted 1/2, and the DeLong-style standard deviation.
  * The threshold rules (ESS, 10th percentile, …).
  * The four-decimal `maxentResults.csv`, including the `(average)` row used for β selection.

Engineering knobs (environment):

| variable | meaning |
|---|---|
| `MAXENT_ENGINE` | `java` (default) or `torch` |
| `MAXENT_TORCH_DEVICE` | default `cuda` |
| `MAXENT_TORCH_BATCH_POINTS` | default 8e6 runs×points×variables per batch, about 0.7 GB per million |
| `MAXENT_TORCH_GPU_SLOTS` | default 2 via `run_fit --gpu-slots`; at most this many worker processes fit on the GPU at once, so many CPU workers can share one GPU |

`maxent_torch.fit_species_dirs` fits several species in shared batches.

## Validation against maxent.jar

The inputs were the stored SWD files of 34 species fitted by maxent.jar in the national run, covering both modes
and 5 to 5,000 presences with 5 to 9 variables. The set has three parts:

* 24 species drawn across the size range;
* *A. auriculata*;
* 9 benchmark species with BLM AIM and VegBank plots.

Scripts: `scripts/validate_maxent_torch.py`, `validate_maxent_torch_downstream.py`, `report_maxent_torch.py`.
Raw records are in `work/maxent_torch_validation/`.

**Pass criteria, fixed before scoring:**

* **C1:** the chosen β is identical.
* **C2:** every CV fold has prediction Spearman > 0.999, |Δ test AUC| < 0.002 and |Δ regularized gain| < 0.001.
* **C3:** every final replicate refitted on maxent.jar's own split (recovered from its sample predictions) has
  prediction Spearman > 0.999, |Δ test AUC| < 0.002, |Δ ESS threshold| < 0.01 and |Δ P5 threshold| < 0.01.
* **C4:** 240 m maps from the same-split refit have suitability Spearman > 0.999 and binary agreement > 0.99,
  and their AIM and VegBank plot AUC differs by less than 0.002.

| criterion | result |
|---|---|
| C1 β choice | **34/34** identical |
| C2 CV folds (850) | **34/34** species. Feature set identical in 99.3% of folds; median max \|Δλ\| 1.3e-12; test AUC identical (4 decimals) in 99.6%; worst fold Spearman 0.99996 |
| C3 final replicates, same split (170) | **34/34** species. Feature set identical in 99.4%; \|Δ ESS\| ≤ 0.0007; \|Δ P5\| ≤ 0.0003; test AUC identical |
| C4 maps and plots, same split (9 species) | **9/9**. Suitability ρ ≥ 0.99994; vote agreement ≥ 0.99996; P5 agreement ≥ 0.9997; AIM and VegBank \|Δ AUC\| ≤ 0.00004 |

Per-species values (CV: minimum feature Jaccard, max |Δλ| over common features, minimum fold Spearman; same
split: max |Δ ESS|):

| mode | species | n | β jar/torch | CV Jaccard min | CV max \|Δλ\| | CV min ρ | rep max \|ΔESS\| |
|---|---|---:|---|---:|---:|---:|---:|
| occ | Arctostaphylos morroensis | 5 | 2/2 | 1 | 1.1e-3 | 1.000000 | 0 |
| occ | Arctostaphylos auriculata | 101 | 2/2 | 1 | 5.0e-11 | 1.000000 | 0.0002 |
| occ | Nolina interrata | 219 | 2/2 | 1 | 3.4e-7 | 1.000000 | 0.0003 |
| occ | Clinopodium carolinianum | 427 | 2/2 | 1 | 5.5e-12 | 1.000000 | 0.0001 |
| occ | Festuca californica | 824 | 2/2 | 1 | 1.1e-11 | 1.000000 | 0 |
| occ | Sorbus sitchensis | 1,298 | 2/2 | 0.897 | 0.20 | 0.999994 | 0.0001 |
| occ | Physocarpus malvaceus | 1,653 | 2/2 | 1 | 1.2e-10 | 1.000000 | 0.0004 |
| occ | Olsynium douglasii | 1,764 | 2/2 | 1 | 2.5e-11 | 1.000000 | 0 |
| occ | Salix laevigata | 2,467 | 2/2 | 1 | 6.6e-11 | 1.000000 | 0 |
| occ | Rosa nutkana | 2,629 | 2/2 | 1 | 1.2e-11 | 1.000000 | 0 |
| occ | Vaccinium ovalifolium | 3,119 | 2/2 | 1 | 1.2e-11 | 1.000000 | 0 |
| occ | Penstemon barbatus | 3,667 | 2/2 | 1 | 8.1e-12 | 1.000000 | 0.0002 |
| occ | Pinus monophylla | 4,022 | 2/2 | 1 | 4.5e-12 | 1.000000 | 0.0001 |
| occ | Fritillaria pudica | 4,333 | 2/2 | 1 | 1.1e-11 | 1.000000 | 0 |
| occ | Helianthus annuus | 4,470 | 2/2 | 0.600 | 0.16 | 0.999977 | 0 |
| occ | Cystopteris fragilis | 4,974 | 2/2 | 0.955 | 0.010 | 0.999999 | 0.0001 |
| occ | Rubus odoratus | 4,987 | 2/2 | 1 | 9.6e-12 | 1.000000 | 0 |
| occ | Larrea tridentata | 4,993 | 2/2 | 1 | 1.6e-10 | 1.000000 | 0.0001 |
| occ | Opuntia engelmannii | 4,995 | 2/2 | 1 | 6.8e-12 | 1.000000 | 0 |
| occ | Salix exigua | 4,995 | 2/2 | 1 | 4.0e-11 | 1.000000 | 0.0001 |
| occ | Silphium perfoliatum | 5,000 | 2/2 | 1 | 3.9e-11 | 1.000000 | 0 |
| occ | Quercus chrysolepis | 5,000 | 2/2 | 0.746 | 0.24 | 0.999961 | 0.0002 |
| daru | Vaccinium vitis-idaea | 477 | 5/5 | 1 | 2.8e-7 | 1.000000 | 0 |
| daru | Guaiacum sanctum | 498 | 2/2 | 1 | 5.0e-12 | 1.000000 | 0 |
| daru | Duranta erecta | 499 | 2/2 | 1 | 4.0e-5 | 1.000000 | 0 |
| daru | Trillium chloropetalum | 499 | 2/2 | 1 | 2.1e-8 | 1.000000 | 0 |
| daru | Asarum lemmonii | 500 | 10/10 | 1 | 7.0e-12 | 1.000000 | 0 |
| daru | Cinna arundinacea | 500 | 2/2 | 1 | 5.1e-12 | 1.000000 | 0 |
| daru | Eriogonum parvifolium | 500 | 2/2 | 1 | 3.8e-7 | 1.000000 | 0.0001 |
| daru | Impatiens capensis | 500 | 2/2 | 1 | 1.1e-8 | 1.000000 | 0 |
| daru | Osmorhiza claytonii | 500 | 2/2 | 1 | 4.2e-12 | 1.000000 | 0 |
| daru | Rhododendron maximum | 500 | 5/5 | 1 | 2.6e-11 | 1.000000 | 0 |
| daru | Staphylea trifolia | 500 | 2/2 | 1 | 7.0e-12 | 1.000000 | 0 |
| daru | Yucca schidigera | 500 | 2/2 | 1 | 4.3e-11 | 1.000000 | 0.0007 |

**Downstream, same split.** These are 240 m maps over the whole calibration area and production plot scoring.
Here the two engines give the same maps. With PyTorch's own replicate draws, the maps differ from maxent.jar's
exactly as much as two maxent.jar runs differ from each other, because maxent.jar seeds its replicate draws from
the wall clock:

| comparison (map over calibration area) | suitability ρ | vote agreement | P5 agreement |
|---|---|---|---|
| torch, maxent.jar's splits vs maxent.jar (9 species) | 0.9999–1.0000 | 1.0000 | 0.9997–1.0000 |
| torch, own splits vs maxent.jar (9 + 3 species) | 0.960–0.999 | 0.990–0.998 | 0.983–0.996 |
| maxent.jar rerun vs maxent.jar (3 species) | 0.970–0.998 | 0.990–0.997 | 0.987–0.997 |

Plot AUC differences (own splits minus maxent.jar) are within that same replicate noise: AIM median +0.0002
(max |Δ| 0.0063) and VegBank median +0.0001 (max |Δ| 0.0050), over 9 species.

Unit tests (`tests/test_maxent_torch.py`) cover the Java helpers using values printed by Java itself
(`java.util.Random`, `Double.toString`, `precision`). Parity tests refit deterministic maxent.jar reference runs
(a single run and 5-fold CV, stored in `tools/bench/torch_parity`) and require:

* the same features in the same order;
* |Δλ| < 1e-8;
* identical gains, AUCs and iterations;
* the files must load in `project.load_replicates` and `evaluate.daru_metrics`.

## Speed

The full protocol per species (`modelling.fit_species`) was measured on a shared 12-core workstation in a heavily
loaded state: load average about 60 on 12 cores, and GPU 0 time-sliced with four other training jobs. Both engines
therefore ran slower than they would on an idle machine.

| species (presences) | maxent.jar wall | maxent.jar CPU-s | torch wall (1 RTX 3090) | torch CPU-s | torch peak GPU memory |
|---|---:|---:|---:|---:|---:|
| A. auriculata (101) | 162 s | 191 | 20.9 s | 21 | 1.2 GB |
| F. californica (824) | 213 s | 260 | 30.1 s | 30 | 1.3 GB |
| V. ovalifolium (3,119) | 211 s | 262 | 31.1 s | 31 | 1.5 GB |

That is a **7–8× speed-up in wall clock and about 9× less CPU time per species**. maxent.jar's
production cost is about 100 s per species and mode on about 2.2 vCPU (about 220 vCPU-s). The torch engine needs
about 20–30 CPU-s plus about 10–25 s of one GPU.

Where the time goes, for a large species (*Larrea*, 4,993 presences, 50 runs; shared machine):

| stage | time |
|---|---:|
| CPU preparation | 4–5 s |
| GPU optimization (500 iterations) | about 25 s, of which about 6–20 ms per iteration is kernel time |
| scoring and file writing | 3 s |

On an idle GPU, the 50-run batch of a 10,000-point species took 6.5 s.

Fitting 9 large species in shared batches gave 34 s per species on the shared machine, so with contention there
is no gain over fitting one species at a time. Per-iteration cost is set by kernel-launch overhead and shared GPU
time, not by arithmetic.

## Caveats and remaining differences

* **Replicate splits.** maxent.jar draws replicate test sets from a wall-clock seed (it forces
  `randomseed=true` for replicated runs). Its final models are therefore not reproducible run to run. maxent_torch
  draws them with maxent.jar's procedure from a fixed seed (reproducible). Cross-validation folds are identical
  to maxent.jar's.
* **Near-ties.** In 0.7% of CV folds, a near-tie in feature selection resolved differently. GPU sums differ from
  Java's sequential sums at about 1e-16 relative. The selected features then partly differ (Jaccard ≥ 0.6), but
  the fitted surface does not: Spearman ≥ 0.99996 and test AUC within 0.0001.
* **Threshold rules.** maxent.jar compares each training presence with its own copy among the background points.
  That copy's density comes from its running linear predictor, so these exact ties break by rounding. The rules
  can then differ by one background point (|Δ ESS| ≤ 0.0007 observed).
* **Not written:**
  * `_omission.csv`, the HTML pages, and the variable contribution and permutation-importance columns of
    `maxentResults.csv` (nothing in the pipeline reads them);
  * `samplePredictions` "Cumulative prediction", which is interpolated on the full background rather than
    maxent.jar's thinned table.
* **Scope.** Only the code paths of our call are implemented: SWD background, continuous variables,
  linear/threshold/hinge features, no bias file, cloglog output, and `autofeature=false`. Quadratic, product and
  categorical features are not implemented.
* **Requirements.** A CUDA GPU with float64 is required for speed. CPU works, but it is slower than maxent.jar.
