# FluxFill-Bench: specification

Version 0.2, 2026-10-28 (0.1: 2026-10-26). This file specifies the benchmark. The masks, splits and metrics are frozen before any
model is fitted, and every change after that gets a new version number.

## 1. Purpose

The benchmark measures how well a method fills the missing values of eddy-covariance (EC) tower records. It
scores both the point values and the full probability distribution. It covers gap lengths from one half-hour to
a full 180-day window, sums over days to whole windows, and three regimes of data availability. It is built to
address the evaluation gaps found in the 2021 to 2026 literature (docs/LITERATURE.md):

| Gap in the literature | Benchmark rule |
|---|---|
| random half-hour splits leak autocorrelation | gaps are contiguous blocks; sites are split 80/20 with spatial blocking |
| tuning on test data, calibration on the test set | a validation-site subset inside the training sites; test sites and masks are frozen and never used for any fitting |
| no proper scores | CRPS is the primary score for every variable and sum |
| sums scored, if at all, without uncertainty | CRPS and bias of daily, monthly and window sums from joint samples |
| skill depends on regime | three regimes scored separately |
| single target variables | every variable is a target; the joint outage of all channels is a mask family |
| gap-filled values used as truth | only measured values (QC = 0, raw sensor columns) are ever scored |

## 2. Data

- **Towers.** All towers with measured half-hourly data from 2014-01-01 onward. Sources:
  - FLUXNET ONEFlux v1.3 via the FLUXNET Shuttle;
  - AmeriFlux BASE;
  - FLUXNET-CH4 V1.0;
  - ICOS.

  Quality rules, deduplication and the dataset card are in `data/towers/README.md`.
- **Reanalysis drivers.** ERA5, downscaled to each site by ONEFlux, and ERA5-Land hourly. Sharper regional
  drivers will be added as they are evaluated (see docs/MODEL.md, section Drivers).
- **Imagery.** Every Sentinel-2 L2A acquisition over a 300 m x 300 m area centred on each tower, at native
  resolution for all bands, embedded by Clay (`data/sentinel2/`, `embeddings/clay/`).

## 3. Splits

- **Sites.** 80% train, 20% test, split by site. The split is stratified by IGBP class and Koppen climate and
  spatially blocked: no test site lies within 50 km of a training site. A validation subset of the training
  sites is used for all model selection. The split file is versioned and frozen.
- **Held-out year.** The last full calendar year of each training site is withheld from training. It is
  scored as regime R2.

## 4. Windows and regimes

The unit of evaluation is a 180-day window at half-hourly resolution (8640 steps) at one site. Every input
modality in the window is provided except what the mask removes.

- **R1, gap inside a record.** Test sites. The window's unmasked observations are visible. This measures
  filling at a site the method has never seen, given that site's own surrounding data.
- **R2, held-out year at a known site.** Training sites, held-out year. The method has seen the site's other
  years.
- **R3, no tower data.** Test sites, with every tower channel masked for the whole window. Only reanalysis and
  imagery remain. This measures upscaling skill and sets the floor for very long gaps.

## 5. Mask families

Masks apply only where the truth is measured. Each family is drawn with a fixed seed per site and window.

| family | definition |
|---|---|
| M1 distribution-matched | artificial gaps whose lengths, merged with real gaps, reproduce the site's observed gap-length distribution (Irvin et al. 2021); for continuity with published numbers |
| M2 contiguous blocks | one block per window of length 1 d, 7 d, 30 d, 90 d or 180 d, placed with uniform start over the window; every season covered |
| M2 sub-daily blocks | one block of 2 h or 6 h on all tower channels, uniform start (release 0.2) |
| M3 transplanted outages | the real joint missingness pattern of all tower channels from another site-year, applied to this window |
| M4 modality dropout | one modality fully removed for the whole window: tower meteorology, reanalysis, or Sentinel-2 |
| M5 mixed | an independent block per tower and reanalysis variable, each with its own placement and a log-uniform length from one half-hour to 90 days, plus one whole modality hidden for the window (tower meteorology, reanalysis or all fluxes, chosen by the seed) (release 0.2) |

M2 and M3 apply to all tower channels together by default: a station outage removes fluxes and meteorology at
once. A per-variable variant of M2 is scored separately.

## 6. Targets

Fluxes: NEE (or FC), LE, H, FCH4 where measured. Meteorology: TA, SW_IN, VPD, PA, WS, P, TS, SWC, USTAR, LW_IN,
NETRAD, PPFD_IN where measured. Imagery: the Clay embedding at each acquisition date (imputation of the
embedding when the image is masked).

## 7. Metrics

All metrics are computed on masked, measured values only, per variable, per regime, per mask family and per gap
length class (1 half-hour; 2 to 8; 9 to 64; 65 to 576; more than 576 half-hours).

- **Primary: CRPS.** Computed from at least 64 joint samples per window, or in closed form for parametric
  predictive distributions. Normalised CRPS divides by the variable's standard deviation at the site.
- **Sums.** Daily, monthly (30-day) and whole-window sums of each flux over the masked span, computed per
  joint sample. Scores: CRPS of the sum and bias of the sum's mean. Methods without joint samples have their
  per-step marginals summed under independence and are labelled as such.
- **Calibration.** Coverage of the 50%, 80% and 95% central intervals, and PIT histograms by gap-length class.
- **Point accuracy.** Of the predictive mean (the mean of a method's samples): MAE, MSE, RMSE, nMAE (MAE /
  standard deviation of the truth), R2 (sklearn) and squared Pearson r, bias, and the weighted absolute percentage
  error WAPE = 100 x sum |pred - truth| / sum |truth|. WAPE is not capped at 100%. Per-value MAPE is not used: NEE, H
  and FCH4 cross zero, so single terms with |truth| near 0 are unbounded. Reported for comparison with the
  literature. Pooled over cells: MSE = sum n rmse^2 / sum n; R2 = 1 - sum n rmse^2 / (sum y^2 - (sum y)^2 / N), with
  the truth sums that `bench/run.py` writes per cell (`bench/truth_stats.py` recomputes them for older parts). R2
  and WAPE are reported only on cells a method fills completely.
- **Variability.** Correlation of predicted and true daily anomalies, and of window means across years at one
  site (interannual variability).
- **Tails.** The 90th percentile of absolute error, and CRPS on the top 5% of |flux| values (CH4 pulses,
  peak uptake).

## 8. Baselines

| method | reference | uncertainty | notes |
|---|---|---|---|
| MDS | Reichstein et al. 2005; REddyProc (Wutzler et al. 2018) | spread of similar conditions | the operational standard |
| RF, XGBoost, ANN, Lasso ensembles | Irvin et al. 2021 | ensemble spread, Platt scaling fitted on validation masks | released harness, `delta` feature corrected |
| XGBoost with vegetation state | Liu et al. 2025 | none | with Clay embeddings in place of PhenoCam |
| Deep ensemble, Laplace likelihood | Vekuri et al. 2025 | aleatoric plus epistemic | strongest published uncertainty baseline |
| SAITS, BRITS, ImputeFormer, CSDI | PyPOTS (Du et al. 2023) | CSDI samples; others point | generic imputers |
| Chronos-2 | Ansari et al. 2025 | quantiles | zero-shot with covariates |
| climatology | site mean diurnal cycle per day of year | empirical spread | floor reference |

Every baseline gets the same inputs that its method admits, the same masks, and model selection on the
validation sites only.

## 9. Rules

1. Test sites, held-out years and masks are fixed before fitting, and the files are hashed in
   `bench/release/MANIFEST.json`. Release 0.1 (2026-10-27) holds 514 test windows at 131 of the 136 test sites (5 have no
   usable window starting in 2015 or later), mask families M2, M3 and M4, built by `bench/freeze.py`. Release 0.2
   (2026-10-28, `bench/release/0.2/`) keeps the same windows and adds M1, the sub-daily M2 blocks and M5 (M1 excludes CN-You, where the released Irvin et al. sampler cannot reach its 10% evaluation fraction on a record with 17% measured NEE; the manifest lists exclusions); it was frozen
   before any FluxFill model was scored on test sites.
2. No method may be tuned, calibrated or checkpoint-selected on test sites or on the held-out years.
3. Every reported number names its regime, mask family, gap class, variable and the number of values it rests on.
4. Every data transformation is recorded with its source, version and checksum.
