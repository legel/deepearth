# DeepCal detail-evidence record scorecard

Protocol: `public-main-bbbe6be-fixed-2291-steps`. Config: `autoresearch/champion.yaml`. Data: 621,558 observations,
29,668-row spatial holdout, 58/63 active benchmarks. Values are the unrounded mean of two checkpoint replays through
the unchanged public evaluator.

| Model | Seed | Steps | Harmonic | Arithmetic |
|---|---:|---:|---:|---:|
| Registered champion | 2-seed mean | 2,291 | 0.378407 | 0.587374 |
| Detail evidence | 1337 | 2,291 | 0.381355 | 0.588091 |
| Detail evidence | 1338 | 2,291 | 0.383536 | 0.594141 |
| **Detail-evidence mean** | **2 seeds** | **2,291** | **0.382446** | **0.591116** |
| **Delta** |  |  | **+0.004039 (+1.07%)** | **+0.003742** |

## Complete scorecard

| Benchmark | Prior | Candidate | Delta |
|---|---:|---:|---:|
| `B57_flowering_phylo_graph_gain` | 0.001267 | 0.000223 | -0.001044 |
| `B62_mycorrhiza_phylo_graph_gain` | 0.002913 | 0.001250 | -0.001663 |
| `B58_lfmc_phylo_graph_gain` | 0.000000 | 0.001319 | +0.001319 |
| `B59_pollinator_phylo_graph_gain` | 0.002130 | 0.002063 | -0.000067 |
| `B61_trait_phylo_graph_gain` | 0.035370 | 0.035785 | +0.000415 |
| `B55_pollinator_phylo_transfer_recall` | 0.038147 | 0.039567 | +0.001420 |
| `B60_community_phylo_graph_gain` | 0.058334 | 0.063768 | +0.005435 |
| `B8_family_from_spacetime` | 0.170268 | 0.173234 | +0.002966 |
| `B6_family_from_env` | 0.172357 | 0.176756 | +0.004399 |
| `B23_species_calibration_mrr` | 0.186731 | 0.189058 | +0.002328 |
| `B50_pollinator_from_spacetime_recall` | 0.188731 | 0.189666 | +0.000935 |
| `B51_pollinator_from_env_recall` | 0.190693 | 0.191922 | +0.001230 |
| `B22_companions_recall` | 0.202491 | 0.203090 | +0.000599 |
| `B21_community_from_species_recall` | 0.204930 | 0.203961 | -0.000969 |
| `B42_mycorrhiza_from_env` | 0.209131 | 0.212434 | +0.003304 |
| `B20_community_from_env_recall` | 0.222702 | 0.218864 | -0.003838 |
| `B56_family_phylo_graph_gain` | 0.229507 | 0.250961 | +0.021454 |
| `B15_vision_from_aerial_cos` | 0.254809 | 0.254164 | -0.000645 |
| `B28_flowering_peak_month_mrr` | 0.366676 | 0.373241 | +0.006565 |
| `B34_lfmc_from_env` | 0.375803 | 0.389696 | +0.013894 |
| `B1_species_from_env_top10` | 0.402252 | 0.404004 | +0.001753 |
| `B5_species_from_spacetime_top10` | 0.400988 | 0.404409 | +0.003421 |
| `B47_infer_naip_ir_cos` | 0.392021 | 0.404794 | +0.012774 |
| `B24_geo_information_gain` | 0.426335 | 0.438739 | +0.012404 |
| `B41_pollinator_from_species_recall` | 0.460037 | 0.439075 | -0.020962 |
| `B18_infer_climate_cos` | 0.457286 | 0.462169 | +0.004883 |
| `B54_pollinator_dist_kl` | 0.531038 | 0.474724 | -0.056314 |
| `B19_infer_aerial_cos` | 0.492016 | 0.495354 | +0.003338 |
| `B17_infer_soil_cos` | 0.503309 | 0.500063 | -0.003246 |
| `B37_imagine_vision_bio_cos` | 0.505227 | 0.511896 | +0.006669 |
| `B52_pollinator_from_photo_recall` | 0.508385 | 0.512034 | +0.003649 |
| `B48_pollinator_from_photo_only_recall` | 0.508384 | 0.512110 | +0.003726 |
| `B45_vision_bio_leave_one_out_cos` | 0.524388 | 0.533700 | +0.009312 |
| `B13_imagine_vision_cos` | 0.604508 | 0.610956 | +0.006448 |
| `B44_infer_topo_cos` | 0.623789 | 0.662727 | +0.038938 |
| `B53_pollinator_calibration_mrr` | 0.684013 | 0.663654 | -0.020359 |
| `B16_infer_clay_cos` | 0.668894 | 0.670332 | +0.001438 |
| `B14_vision_leave_one_out_cos` | 0.666100 | 0.675849 | +0.009749 |
| `B43_infer_hydro_cos` | 0.657898 | 0.698876 | +0.040978 |
| `B27_flowering_fidelity` | 0.734281 | 0.727470 | -0.006811 |
| `B26_flowering_auc` | 0.750456 | 0.747735 | -0.002722 |
| `B9_phylo_from_photo_cos` | 0.826584 | 0.840606 | +0.014022 |
| `B2_species_from_photo_top1` | 0.828586 | 0.842743 | +0.014157 |
| `B4_species_from_photo_only_top1` | 0.829429 | 0.844260 | +0.014831 |
| `B46_infer_chm_cos` | 0.865668 | 0.875391 | +0.009723 |
| `B49_form_trait_f1` | 0.872081 | 0.888716 | +0.016635 |
| `B38_water_soil_regime_f1` | 0.908194 | 0.909761 | +0.001567 |
| `B33_growth_rate_trait_f1` | 0.909192 | 0.911617 | +0.002424 |
| `B35_sun_trait_f1` | 0.912032 | 0.919792 | +0.007760 |
| `B11_traits_from_photo_f1` | 0.915436 | 0.920579 | +0.005143 |
| `B10_traits_from_photo_env_f1` | 0.918399 | 0.923388 | +0.004989 |
| `B32_plant_type_trait_f1` | 0.941102 | 0.939120 | -0.001981 |
| `B30_seasonality_trait_f1` | 0.940832 | 0.948341 | +0.007510 |
| `B3_species_from_photo_top5` | 0.953485 | 0.958811 | +0.005326 |
| `B36_ease_of_care_trait_f1` | 0.955567 | 0.959994 | +0.004427 |
| `B7_family_from_phylo` | 0.952676 | 0.961962 | +0.009286 |
| `B63_myco_from_species_f1` | 0.987818 | 0.989523 | +0.001705 |
| `B12_traits_leave_one_out_f1` | 0.992887 | 0.993623 | +0.000736 |

## Inactive

`B25_forecast_climate_cos` and `B31_forecast_vision_cos` require a temporal holdout.
`B29_species_dist_30m_skill`, `B39_species_dist_3km_skill`, and `B40_species_dist_300m_skill` lack the required
distribution labels under this registered public protocol.
