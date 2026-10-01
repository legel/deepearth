# DeepEarth sparse ecological Earth4D scorecard

Protocol: `public-main-bbbe6be6-fixed-8000-steps`. Data: 621,558 observations and a 29,668-row spatial holdout.
Values are unrounded means of two independent checkpoints through Lance's unchanged public evaluator.

| Model | Seed | Steps | Harmonic | Arithmetic | Parameters |
|---|---:|---:|---:|---:|---:|
| Prior family-preserving record | two-seed mean | 8,300 | 0.385644 | 0.575712 | 27.6M |
| Sparse ecological Earth4D | 1337 | 8,000 | 0.388024 | 0.577688 | 52.7M |
| Sparse ecological Earth4D | 1338 | 8,000 | 0.388125 | 0.577697 | 52.7M |
| **Sparse ecological mean** | **two seeds** | **8,000** | **0.388075** | **0.577692** | **52.7M** |
| **Record delta** |  |  | **+0.002430** | **+0.001980** | **+25.1M** |

## Capability scorecard

Rows are ordered weakest-first by the candidate mean. Delta is relative to the prior family-preserving record.

| Benchmark | Mean | Seed 1337 | Seed 1338 | Delta |
|---|---:|---:|---:|---:|
| `B55_pollinator_phylo_transfer_recall` | 0.038644 | 0.038601 | 0.038686 | +0.000259 |
| `B50_pollinator_from_spacetime_recall` | 0.174236 | 0.176513 | 0.171958 | -0.001130 |
| `B51_pollinator_from_env_recall` | 0.180842 | 0.180559 | 0.181124 | +0.003120 |
| `B8_family_from_spacetime` | 0.180919 | 0.180632 | 0.181205 | -0.000860 |
| `B6_family_from_env` | 0.197924 | 0.197890 | 0.197957 | +0.007584 |
| `B23_species_calibration_mrr` | 0.202651 | 0.202628 | 0.202674 | +0.022162 |
| `B15_vision_from_aerial_cos` | 0.259334 | 0.259347 | 0.259321 | -0.002729 |
| `B22_companions_recall` | 0.277953 | 0.276007 | 0.279899 | -0.003498 |
| `B21_community_from_species_recall` | 0.286956 | 0.285073 | 0.288838 | +0.001541 |
| `B20_community_from_env_recall` | 0.298385 | 0.298351 | 0.298419 | -0.003812 |
| `B28_flowering_peak_month_mrr` | 0.298976 | 0.299014 | 0.298938 | -0.041627 |
| `B42_mycorrhiza_from_env` | 0.301170 | 0.301409 | 0.300931 | +0.010770 |
| `B18_infer_climate_cos` | 0.330662 | 0.330461 | 0.330862 | +0.004716 |
| `B5_species_from_spacetime_top10` | 0.348220 | 0.348355 | 0.348085 | +0.019229 |
| `B34_lfmc_from_env` | 0.412978 | 0.413005 | 0.412952 | +0.027153 |
| `B47_infer_naip_ir_cos` | 0.419256 | 0.419028 | 0.419483 | -0.009889 |
| `B44_infer_topo_cos` | 0.433676 | 0.432919 | 0.434433 | -0.023457 |
| `B1_species_from_env_top10` | 0.441402 | 0.441317 | 0.441486 | +0.049633 |
| `B43_infer_hydro_cos` | 0.456338 | 0.455486 | 0.457191 | +0.017958 |
| `B48_pollinator_from_photo_only_recall` | 0.496441 | 0.498159 | 0.494722 | +0.017505 |
| `B52_pollinator_from_photo_recall` | 0.498378 | 0.500542 | 0.496214 | +0.018484 |
| `B19_infer_aerial_cos` | 0.525534 | 0.525538 | 0.525530 | -0.005658 |
| `B17_infer_soil_cos` | 0.528748 | 0.527345 | 0.530150 | +0.006101 |
| `B37_imagine_vision_bio_cos` | 0.536100 | 0.536105 | 0.536096 | -0.002286 |
| `B41_pollinator_from_species_recall` | 0.546267 | 0.547322 | 0.545211 | +0.018157 |
| `B45_vision_bio_leave_one_out_cos` | 0.546671 | 0.546675 | 0.546666 | -0.003835 |
| `B13_imagine_vision_cos` | 0.628427 | 0.628430 | 0.628423 | +0.004900 |
| `B14_vision_leave_one_out_cos` | 0.654979 | 0.654983 | 0.654976 | +0.002928 |
| `B27_flowering_fidelity` | 0.667776 | 0.667696 | 0.667856 | -0.011478 |
| `B16_infer_clay_cos` | 0.672570 | 0.671627 | 0.673514 | +0.019532 |
| `B26_flowering_auc` | 0.737997 | 0.737985 | 0.738010 | -0.003213 |
| `B54_pollinator_dist_kl` | 0.750413 | 0.747618 | 0.753209 | +0.031930 |
| `B46_infer_chm_cos` | 0.753725 | 0.753400 | 0.754049 | -0.016043 |
| `B49_form_trait_f1` | 0.764243 | 0.764260 | 0.764227 | -0.026432 |
| `B4_species_from_photo_only_top1` | 0.778364 | 0.778381 | 0.778347 | +0.030251 |
| `B2_species_from_photo_top1` | 0.781532 | 0.781414 | 0.781650 | +0.024572 |
| `B9_phylo_from_photo_cos` | 0.781604 | 0.781604 | 0.781604 | -0.011584 |
| `B38_water_soil_regime_f1` | 0.793535 | 0.793550 | 0.793520 | -0.038979 |
| `B53_pollinator_calibration_mrr` | 0.803315 | 0.807638 | 0.798992 | +0.006078 |
| `B11_traits_from_photo_f1` | 0.855996 | 0.856015 | 0.855977 | -0.011588 |
| `B10_traits_from_photo_env_f1` | 0.856209 | 0.856216 | 0.856202 | -0.015417 |
| `B35_sun_trait_f1` | 0.865155 | 0.865155 | 0.865155 | -0.009533 |
| `B33_growth_rate_trait_f1` | 0.870096 | 0.870096 | 0.870096 | -0.003423 |
| `B30_seasonality_trait_f1` | 0.906597 | 0.906597 | 0.906597 | -0.009075 |
| `B32_plant_type_trait_f1` | 0.917629 | 0.917642 | 0.917616 | +0.006893 |
| `B3_species_from_photo_top5` | 0.937963 | 0.937913 | 0.938014 | +0.020392 |
| `B36_ease_of_care_trait_f1` | 0.938881 | 0.938881 | 0.938881 | -0.003806 |
| `B7_family_from_phylo` | 0.951126 | 0.951159 | 0.951092 | -0.014763 |
| `B12_traits_leave_one_out_f1` | 0.997831 | 0.997831 | 0.997831 | +0.001266 |
| `B63_myco_from_species_f1` | 1.000000 | 1.000000 | 1.000000 | +0.000000 |

## Mechanism diagnostics

These remain visible but are not human-capability claims.

| Benchmark | Mean | Seed 1337 | Seed 1338 | Delta |
|---|---:|---:|---:|---:|
| `B57_flowering_phylo_graph_gain` | 0.000000 | 0.000000 | 0.000000 | -0.001880 |
| `B61_trait_phylo_graph_gain` | 0.000000 | 0.000000 | 0.000000 | -0.001876 |
| `B58_lfmc_phylo_graph_gain` | 0.001441 | 0.001436 | 0.001445 | +0.000135 |
| `B56_family_phylo_graph_gain` | 0.009202 | 0.009202 | 0.009202 | -0.063553 |
| `B59_pollinator_phylo_graph_gain` | 0.011603 | 0.009684 | 0.013523 | -0.031634 |
| `B62_mycorrhiza_phylo_graph_gain` | 0.025469 | 0.024490 | 0.026448 | -0.023550 |
| `B60_community_phylo_graph_gain` | 0.086133 | 0.086683 | 0.085582 | -0.206296 |
| `B24_geo_information_gain` | 0.340131 | 0.340097 | 0.340164 | -0.025061 |

## Inactive

| Benchmark | Reason |
|---|---|
| `B25_forecast_climate_cos` | needs temporal holdout |
| `B31_forecast_vision_cos` | needs temporal holdout |
| `B29_species_dist_30m_skill` | required inputs or labels absent |
| `B39_species_dist_3km_skill` | required inputs or labels absent |
| `B40_species_dist_300m_skill` | required inputs or labels absent |
