# FluxFill

FluxFill is the first application of a flux foundation model in DeepEarth. The model learns the joint field of
everything measured at and around eddy-covariance towers: site x time x variable, at half-hourly resolution. It
learns this field by masking and reconstructing it. Its first scientific task is filling gaps in tower records:
CO2 (NEE), latent and sensible heat (LE, H), methane (FCH4) and the meteorological drivers. Every value it fills
comes with a calibrated probability distribution.

Work of the Quantitative Ecosystem Dynamics lab, University of California, Berkeley. Sponsored by Ecological
Intelligence, Inc.

Authors: Mira Shivani Sankar (mirashivanisankar@berkeley.edu), Lance Legel (legel@berkeley.edu), Trevor Keenan
(trevorkeenan@berkeley.edu).

## What it addresses

Published gap-filling methods (see [docs/LITERATURE.md](docs/LITERATURE.md)) share these limits:

1. **Single-target models.** Methods fill one variable at a time, from drivers that go missing during the same
   outages.
2. **No proper scores.** None scores predictions with a proper scoring rule. Uncertainty, where given, is
   calibrated on the test set itself or on short gaps only.
3. **No long-gap benchmark.** No benchmark has contiguous gaps of weeks to months across many sites and
   variables, scored on sums.
4. **No cross-site learning.** Methods are site-specific and do not learn from other towers.
5. **Weak state covariates.** Ecosystem state is observed at best through coarse vegetation indices.

FluxFill and its benchmark address each limit:

1. **One model for all variables.** It fills every variable jointly and treats every modality as maskable:
   tower fluxes, tower meteorology, reanalysis, and Sentinel-2.
2. **Joint samples.** It outputs joint samples over each 180-day window, so sums over a day, month or window
   carry correct uncertainty.
3. **FluxFill-Bench.** The benchmark ([docs/BENCHMARK.md](docs/BENCHMARK.md)) scores CRPS, calibration by gap
   length, and the bias and CRPS of sums. It covers three regimes: a gap inside a record, a held-out year at a
   known site, and a site never seen.
4. **Trained on all towers.** The model learns from every tower with real measurements since 2014, with 20% of
   sites held out for test.
5. **State from imagery.** It observes ecosystem state through every Sentinel-2 acquisition over a 300 m x
   300 m area around each tower, embedded by the Clay foundation model.

## Layout

| path | content |
|---|---|
| `data/towers/` | tower records: sources, quality control, measured-only rule, site split, dataset card |
| `data/sentinel2/` | Sentinel-2 L2A chips around every tower, every acquisition since 2015 |
| `embeddings/clay/` | Clay embeddings of every chip |
| `bench/` | FluxFill-Bench: masks, metrics, regimes, baseline runners |
| `baselines/` | established methods run on the benchmark (MDS, random forest, XGBoost, deep ensembles, generic imputers) |
| `model/` | the FluxFill model |
| `dashboard/` | the public dashboard at deepearth.ai/fluxfill |
| `docs/` | benchmark specification, model design, literature, provenance |

## Status

In construction since 2026-10-26. Results appear here and on the dashboard once they are measured.
