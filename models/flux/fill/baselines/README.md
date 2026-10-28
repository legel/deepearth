# FluxFill-Bench baselines

Established gap-filling methods as benchmark methods of `bench/run.py`. Every method implements
`predict(window, mask, context) -> (S, V, T)` samples (S >= 64). The window it receives has every hidden value set to
NaN, and `context` holds the site's other calendar years. All methods are scored on the same windows and masks as
FluxFill. Methods are registered in `bench/run.py:method_registry` and in `bench/export_dashboard.py --methods`.

| name | method | reference and code | trained on | uncertainty | joint over time |
|---|---|---|---|---|---|
| `climatology` | site mean diurnal and seasonal cycle | `climatology.py` | site, other years | empirical pool | no |
| `mds` | marginal distribution sampling | Reichstein et al. 2005; REddyProc 1.3.2 (Wutzler et al. 2018); `mds.py`, `mds.R` | visible window | Gaussian with MDS `_fsd` | no |
| `irvin_rf` | random forest ensemble | Irvin et al. 2021, Agric. For. Meteorol. 308-309:108528; harness github.com/stanfordmlgroup/methane-gapfill-ml @473db3e; `irvin_rf.py` | site: other years plus visible window | Laplace, member std / 2, Platt factor fitted on held-out context blocks | no |
| `vekuri_de` | deep ensemble, Laplace likelihood | Vekuri et al. 2025; github.com/hvekuri/nn_gapfilling_unc @7edd07a; `vekuri_de.py` | site: other years plus visible window | 5-member mixture of Laplace, aleatoric plus epistemic | partly: member mean trajectories coherent, aleatoric noise independent |
| `liu_xgb` | XGBoost, base drivers plus vegetation state | Liu et al. 2025, Agric. For. Meteorol. 364:110438; github.com/YujieLiu666/fluxgapfill_XGB_vs_MDS @a9a6eeb; `liu_xgb.py` | site: other years plus visible window | multi-quantile regression (ours) | no |
| `saits` | SAITS | Du et al. 2023, Expert Syst. Appl. 219:119619; PyPOTS 1.5 `_SAITS`; `pypots_imputers.py` | training sites | residual-based Gaussian fitted on calibration validation sites (ours) | no |
| `csdi` | CSDI | Tashiro et al. 2021, NeurIPS; PyPOTS 1.5 `_CSDI`; `pypots_imputers.py` | training sites | 64 diffusion samples | within 6-day chunks |
| `chronos2` | Chronos-2, zero-shot with covariates | Ansari et al. 2025, arXiv:2510.15821; chronos-forecasting 2.3.2, amazon/chronos-2 @29ec376; `chronos2.py` | none (pretrained) | 21 quantiles | no |

Each module's docstring lists the ported lines of the released code and every change, with its reason. The changes
in brief:

- **All per-site methods** (`irvin_rf`, `vekuri_de`, `liu_xgb`) fit on the site's other calendar years plus the
  visible part of the window. Drivers hidden by the mask, or missing, are replaced by `ERA5_<VAR>`, as FLUXNET
  does. They fill the fluxes NEE, LE, H and FCH4 only; tower meteorology is left unfilled (fill rate 0), as in the
  released methods.
- **No hyperparameter search.** Irvin (RandomizedSearchCV, 20 configurations) and Liu (BayesSearchCV, up to 5,100
  trials) are run with one fixed configuration from the released search space. Irvin's trees each draw a bootstrap
  sample of 4,000 rows (`max_samples=4000`).
- **Irvin `delta` fix.** The harness measures `delta` from the first row of whatever table it receives
  (`predictors/process.py:40`), so fit and prediction use different origins. Here `delta` is days since
  2000-01-01 local standard time for every table (`irvin_rf.temporal`; test `test_irvin_delta_fix`).
- **Irvin members and calibration.** Ten members, each fit on a bootstrap resample of whole days. The Platt factor
  (`ensemble.py:125-132`) is fitted on 7-day blocks holding 10 % of the context's measured values, predicted with
  ERA5 drivers by an ensemble fit on the rest of the context. It is never fitted on the window. The harness fits
  it on its test set.
- **Vekuri.** PyTorch port of the Keras network, loss, Adam rule and early stopping. All members of a window's
  masks train as one batched network with independent weights, data and stopping (`predict_many`). Month_sin and
  Month_cos are computed from the timestamp, not the row number.
- **Liu.** SW_IN stands in for the three PPFD columns. The Clay v1.5 CLS embedding (first 8 principal components
  of clear chips) stands in for PhenoCam GCC where the site has a Clay store. Among the validation sites only
  US-Ha1 has one (713 chips, 266 clear). Uncertainty comes from XGBoost multi-quantile regression, since the
  release has none.
- **SAITS and CSDI** train on training sites only. Until `towers/split.csv` exists, this is the provisional
  hold-out by SHA-256 of the site id, which the model also uses. Training masks are benchmark-like blocks plus the
  PyPOTS default MCAR. SAITS uses day tokens (180 steps x 1,058 features). CSDI uses 6-day half-hourly chunks with
  float16 sampling.
- **Chronos-2** forecasts each hidden block from the 2,048 half-hours before it, with ERA5 and time as covariates.
  When there is no history before the block, it forecasts the time-reversed series from the data after the block.

Joint samples and sums. The runner sums each sample over daily, 30-day and window blocks. For methods whose
samples are independent per half-hour, the spread of these sums is the independence spread, too narrow for
correlated errors. Vekuri samples are joint through the member: each member's mean is one trajectory, so the
spread of the sums is the between-member spread of summed means plus the summed aleatoric variance. This is the
structure of the released sum uncertainty (`gapfilling_demo.ipynb` cell 6). CSDI samples are joint within 6 days.

Validation windows: `val_windows.py` (rule in its docstring). These are validation sites only, never test sites.

Run: `bench/run.py --windows /home/photon/4tb/fluxfill/bench/val/windows_val_list.json --families M2 --methods
<name> --out ...`; `pypots_imputers.py train saits|csdi`, then `pypots_imputers.py spread`. Tests:
`pytest baselines/tests` (shape, no use of hidden values, determinism).
