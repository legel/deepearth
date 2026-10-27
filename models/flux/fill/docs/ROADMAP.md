# FluxFill roadmap

## Scaling

FluxFill is a model of the site x time x variable field of eddy-covariance towers. We measure how its accuracy grows
with the number of towers it is trained on and with its size, so that the flux community can see what a foundation
model of tower data gains from more data and from more parameters.

**Data axis.** One fixed model (192-wide, 6 layers, 7.3 M parameters) is trained on nested subsets of the training
towers: 30, 60, 120, 240 and all 488. Each subset contains the smaller ones and holds the land-cover, climate and
methane strata of the training set in proportion. Every run has the same number of optimisation steps.

**Model axis.** All 488 training towers; the transformer width and depth grow from 0.9 M to about 35 M block
parameters (2.5 M to 38 M in total). Every run sees the same number of training windows.

**Evaluation.** Every run is scored on the same 60 windows at the 54 validation towers of the frozen split, with the
benchmark's contiguous gaps of 1 to 180 days: CRPS over all tower variables and over the fluxes. Validation towers are
never trained on; test towers are never used.

**Fit.** We fit L(x) = a x^(-alpha) + c, where x is the number of training towers or of block parameters, and report the
exponent alpha and the floor c with bootstrap intervals over the validation windows. Results are published as a
table and as log-log charts on the dashboard, and the run records are in `scaling.json`.

**More data.** The present dataset has 678 towers and 4,149 site-years from 2014. Further towers and years are
available: years before 2014 at the same towers (2,258 site-years, 112 more towers; no Sentinel-2 imagery, which tests
how the model works without it), AmeriFlux BASE records outside the FLUXNET products (129 towers from 2014), and
JapanFlux2024 sites processed outside ONEFlux (29 towers). Together they would bring the dataset to about 950 towers and
6,900 site-years, extending the data axis beyond the present training set. New towers are assigned to the frozen
split by its 50 km rule, so the test set does not change.
