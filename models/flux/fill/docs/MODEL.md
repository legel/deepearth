# FluxFill model: design

Version 0.1, 2026-10-26. Design before measurement. Each choice below is tested by an ablation (section 7).

## 1. The field

A site's record is a field over time t (half-hours), variables v and modalities. Each value carries:
- an observation flag: measured, masked by the benchmark, or missing in the source;
- a provenance tag: tower sensor, reanalysis, or imagery.

The model learns p(missing values | observed values) for any pattern of missingness, over a 180-day window:
8640 half-hours.

## 2. Tokens

- **Tower and reanalysis variables.** One token per variable per day. It holds the 48 half-hourly values,
  standardised per variable with training-set statistics, and their 48 observation flags. A shared linear
  projection embeds it. Each token also gets:
  - a learned variable embedding and a provenance embedding (sensor or reanalysis);
  - the day index within the window and the day of year (sine and cosine);
  - the site location encoding.

  With about 20 variables this is about 3,600 tokens per window.
- **Imagery.** One token per Sentinel-2 acquisition inside the window: the Clay CLS embedding, projected from
  1024 dimensions to the model width. It also carries the acquisition time, the SCL cloud and shadow fraction,
  and the view and sun angles. Acquisitions with no usable pixels are kept and flagged, so the model learns
  reliability.
- **Masks.** A masked token keeps its variable, time, provenance and location embeddings. Its values and flags
  are replaced by a learned mask vector. A modality removed by M4 contributes no tokens at all.

## 3. Encoder

A transformer with factorised attention, alternating two layer types:
- **time attention:** each variable attends over its 180 days;
- **variable attention:** each day attends across its variables and the imagery tokens near that date.

Full attention over 3,600 tokens is affordable at this size. The factorised form is the default because it
scales to longer windows and more variables. Width, depth and the factorisation are set by validation-site CRPS.

## 4. Generative head

The model must produce joint samples, so that sums over a gap carry correct uncertainty.

- **v0, marginal.** A per-half-hour heteroscedastic Student-t head, trained on negative log-likelihood. It is
  scored with CRPS and labelled as marginal.
- **v1, joint.** Masked generation with a diffusion head per token, following masked autoregressive generation
  with continuous tokens (Li et al. 2024, "Autoregressive Image Generation without Vector Quantization"):
  1. The encoder conditions a small denoising network that generates the 48 values of a masked day-token.
  2. Masked tokens are generated in a random order, a few at a time.
  3. Each generated token is fed back as observed before the next ones.

  Repeating with different noise and order seeds gives joint samples over the whole window. A
  conditional-flow-matching variant of the head is the alternative.

## 5. Site memory and cross-site retrieval

- **Site memory.** Summary tokens from the same site's other years. Each is one pooled embedding per 30-day
  block from a frozen pass of the encoder over the block's observed data.
- **Cross-site retrieval.** Summary tokens from the k most similar training sites, chosen by Clay embedding,
  climate and IGBP class. Spatial distance is used only within the training sites, so the 50 km blocking
  of test sites still holds.

Both enter as extra tokens in variable attention.

## 6. Training

- **Masking.** Training masks sample from the benchmark mask families M1 to M4, plus random blocks per
  variable. A fraction of windows has every tower channel masked (regime R3), and a fraction has Sentinel-2
  fully masked. Every modality, Clay embeddings included, is a reconstruction target when masked.
- **Targets.** Loss is computed only on measured values. Gap-filled source values never enter, as input or as
  target.
- **Sampling.** Windows are balanced by IGBP class and climate. Site weights stop long records from dominating.
- **Selection.** All selection uses validation-site CRPS. Test sites are untouched.

## 6b. Site adaptation at inference (FluxFill v4)

Before the first mask of a window is filled, the trained network is fine-tuned on that site's own data and reset
to the trained weights after the window (`model/method.py`, `FluxFill.adapt`; `FLUXFILL_ADAPT_STEPS`,
`FLUXFILL_ADAPT_LR`, `FLUXFILL_ADAPT_ROW_W` in `bench/run.py`).

- **Data.** Each step's batch mixes (a) the window itself, with a fresh training mask over its visible values as
  targets, and (b) 180-day windows of the site's other years from the runner's context table, when the record has
  any (a context shorter than one window is padded with unmeasured days). The values the benchmark hides are NaN
  when the method receives the window, so they never enter the loss.
- **Optimiser.** AdamW, weight decay 0, betas (0.9, 0.98), gradient clip 1.0, batch 2, the training losses (NLL
  and flow matching). 300 steps at lr 1e-4 is FluxFill v4.
- **Why.** Per-site regressors such as XGBoost (Liu et al. 2025) refit on each site's record; adaptation gives the
  cross-site network the same access to the site's own driver-to-flux relation while keeping its prior.
- **Cost.** About 40 s per window on one RTX 3090 at width 320 (5.7 s per mask on average, against 0.3 s without
  adaptation).

## 7. Ablations (for publication)

Each is a retrained model, scored on the full benchmark:
1. no Sentinel-2;
2. no reanalysis;
3. no tower meteorology (fluxes only);
4. no site memory;
5. no cross-site retrieval;
6. marginal head instead of joint;
7. per-variable masks only (no joint outages in training);
8. Clay CLS vs Clay patch tokens;
9. drivers: ONEFlux ERA5 vs ERA5-Land vs sharper regional drivers;
10. window length 90 vs 180 vs 365 days.

## 8. Drivers

ERA5-Land (0.1 degree, hourly) is the global baseline driver. Candidates for sharper drivers, to be measured
against tower meteorology at the training sites:
- HRRR, 3 km hourly, CONUS, from 2014;
- CERRA, 5.5 km, Europe;
- geostationary-satellite surface radiation, 3 to 5 km;
- MODIS and VIIRS land surface temperature, 1 km;
- Copernicus DEM, 30 m (static terrain).

A driver is kept if it lowers validation-site CRPS for the meteorological variables in regime R3.

## 9. Scaling

Validation CRPS is measured against model size (width and depth at fixed data) and against the number of training
sites (at fixed model size). Each finished run adds one record to `model/scaling/scaling.json`: run, n_params,
n_train_sites, n_train_windows, tokens_seen, steps, val_crps, val_crps_flux, and axis ("model" or "data"). The
dashboard fits L = a x^(-alpha) + c to each axis once it has 3 or more points.

