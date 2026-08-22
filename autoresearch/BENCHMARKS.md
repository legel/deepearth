# DeepEarth wide-cell Earth4D mesh scorecard

Protocol: `public-main-bbbe6be6-fixed-8300-steps`. Model: `core/train.py::Design(width=192, levels=12, hash_log2=14, latents=16, layers=2)`.
Data: 621,558 observations, 29,668-row spatial holdout, complete 64-dimensional AlphaEarth coverage. Values are the unrounded mean of two independent runs through Lance’s unchanged public evaluator.

| Model | Seed | Steps | Harmonic | Arithmetic | Parameters |
|---|---:|---:|---:|---:|---:|
| Registered 797.1M reference | 1337 | 5,126 | 0.318693 | 0.570700 | 797.1M |
| Registered 25.4M public champion | 2 seeds | 2,291 | 0.378407 | **0.587374** | 25.4M |
| 128-wide Earth4D mesh control | 1337 | 8,300 | 0.377276 | 0.557451 | 14.5M |
| 128-wide Earth4D mesh control | 1338 | 8,300 | 0.381407 | 0.566217 | 14.5M |
| **192-wide Earth4D mesh** | **1337** | **8,300** | **0.382631** | **0.572135** | **22.7M** |
| **192-wide Earth4D mesh** | **1338** | **8,300** | **0.388055** | **0.578847** | **22.7M** |
| **192-wide mesh mean** | **2 seeds** | **8,300** | **0.385343** | **0.575491** | **22.7M** |
| **Delta vs 128-wide mesh** |  |  | **+0.006002** | **+0.013657** | **+8.2M** |
| **Delta vs registered 25.4M champion** |  |  | **+0.006936** | **-0.011884** | **-2.7M** |

## Record criterion

The public north star remains the harmonic mean, with the arithmetic mean and every human-interpretable capability reported alongside it. The 192-wide mesh is therefore a new harmonic record, not an arithmetic record. Its matched promotion comparison is the otherwise identical 128-wide mesh at 8,300 steps: both candidate seeds improve both aggregates over their seed-matched controls.

The older 25.4M champion used 2,291 steps, so that row is a public-score comparison rather than an equal-budget architecture ablation. Relative to it, the mesh improves harmonic breadth while losing arithmetic capability mass; the complete regression profile below is part of the record. The evaluator, benchmark definitions, spatial holdout, and active 58/63 suite are unchanged.

## Capability scorecard

Capabilities are ordered weakest-first by the new two-seed mean. Deltas compare against the registered 25.4M public champion.

| Benchmark | Registered champion | 192-wide mesh | Delta | Seed 1337 | Seed 1338 |
|---|---:|---:|---:|---:|---:|
| `B55_pollinator_phylo_transfer_recall` | 0.038147 | 0.038377 | +0.000230 | 0.038121 | 0.038633 |
| `B23_species_calibration_mrr` | 0.186731 | 0.174621 | -0.012109 | 0.173658 | 0.175584 |
| `B50_pollinator_from_spacetime_recall` | 0.188731 | 0.175811 | -0.012921 | 0.178751 | 0.172870 |
| `B51_pollinator_from_env_recall` | 0.190693 | 0.178624 | -0.012068 | 0.181053 | 0.176196 |
| `B8_family_from_spacetime` | 0.170268 | 0.182638 | +0.012370 | 0.179048 | 0.186228 |
| `B6_family_from_env` | 0.172357 | 0.192261 | +0.019904 | 0.182891 | 0.201631 |
| `B15_vision_from_aerial_cos` | 0.254809 | 0.262125 | +0.007317 | 0.262936 | 0.261315 |
| `B22_companions_recall` | 0.202491 | 0.281450 | +0.078959 | 0.266659 | 0.296240 |
| `B21_community_from_species_recall` | 0.204930 | 0.285428 | +0.080498 | 0.271094 | 0.299762 |
| `B42_mycorrhiza_from_env` | 0.209131 | 0.289056 | +0.079925 | 0.306199 | 0.271912 |
| `B20_community_from_env_recall` | 0.222702 | 0.302195 | +0.079493 | 0.291403 | 0.312987 |
| `B18_infer_climate_cos` | 0.457286 | 0.325881 | -0.131405 | 0.308773 | 0.342989 |
| `B5_species_from_spacetime_top10` | 0.400988 | 0.333086 | -0.067901 | 0.327255 | 0.338917 |
| `B28_flowering_peak_month_mrr` | 0.366676 | 0.340390 | -0.026286 | 0.362855 | 0.317925 |
| `B1_species_from_env_top10` | 0.402252 | 0.375657 | -0.026594 | 0.373500 | 0.377814 |
| `B34_lfmc_from_env` | 0.375803 | 0.385664 | +0.009861 | 0.365158 | 0.406170 |
| `B47_infer_naip_ir_cos` | 0.392021 | 0.429026 | +0.037006 | 0.432039 | 0.426013 |
| `B43_infer_hydro_cos` | 0.657898 | 0.438265 | -0.219633 | 0.442856 | 0.433673 |
| `B44_infer_topo_cos` | 0.623789 | 0.457125 | -0.166664 | 0.467333 | 0.446917 |
| `B48_pollinator_from_photo_only_recall` | 0.508384 | 0.479522 | -0.028862 | 0.478084 | 0.480961 |
| `B52_pollinator_from_photo_recall` | 0.508385 | 0.480432 | -0.027953 | 0.480491 | 0.480374 |
| `B17_infer_soil_cos` | 0.503309 | 0.522736 | +0.019427 | 0.494737 | 0.550736 |
| `B41_pollinator_from_species_recall` | 0.460037 | 0.528000 | +0.067963 | 0.525866 | 0.530134 |
| `B19_infer_aerial_cos` | 0.492016 | 0.531156 | +0.039140 | 0.529469 | 0.532843 |
| `B37_imagine_vision_bio_cos` | 0.505227 | 0.538446 | +0.033220 | 0.538805 | 0.538088 |
| `B45_vision_bio_leave_one_out_cos` | 0.524388 | 0.550537 | +0.026150 | 0.550304 | 0.550771 |
| `B13_imagine_vision_cos` | 0.604508 | 0.623543 | +0.019035 | 0.621599 | 0.625486 |
| `B14_vision_leave_one_out_cos` | 0.666100 | 0.652064 | -0.014036 | 0.651206 | 0.652922 |
| `B16_infer_clay_cos` | 0.668894 | 0.652809 | -0.016085 | 0.648410 | 0.657208 |
| `B27_flowering_fidelity` | 0.734281 | 0.679172 | -0.055110 | 0.682712 | 0.675631 |
| `B54_pollinator_dist_kl` | 0.531038 | 0.718463 | +0.187425 | 0.716171 | 0.720755 |
| `B26_flowering_auc` | 0.750456 | 0.741232 | -0.009225 | 0.734895 | 0.747568 |
| `B4_species_from_photo_only_top1` | 0.829429 | 0.748450 | -0.080980 | 0.738169 | 0.758730 |
| `B2_species_from_photo_top1` | 0.828586 | 0.757972 | -0.070615 | 0.749663 | 0.766280 |
| `B46_infer_chm_cos` | 0.865668 | 0.769865 | -0.095803 | 0.762454 | 0.777276 |
| `B49_form_trait_f1` | 0.872081 | 0.789723 | -0.082358 | 0.783368 | 0.796077 |
| `B9_phylo_from_photo_cos` | 0.826584 | 0.793243 | -0.033341 | 0.793279 | 0.793207 |
| `B53_pollinator_calibration_mrr` | 0.684013 | 0.797285 | +0.113272 | 0.791121 | 0.803449 |
| `B38_water_soil_regime_f1` | 0.908194 | 0.832715 | -0.075478 | 0.817505 | 0.847926 |
| `B11_traits_from_photo_f1` | 0.915436 | 0.867946 | -0.047491 | 0.859496 | 0.876395 |
| `B10_traits_from_photo_env_f1` | 0.918399 | 0.871480 | -0.046919 | 0.865578 | 0.877382 |
| `B33_growth_rate_trait_f1` | 0.909192 | 0.873472 | -0.035721 | 0.871687 | 0.875257 |
| `B35_sun_trait_f1` | 0.912032 | 0.874665 | -0.037367 | 0.874917 | 0.874412 |
| `B32_plant_type_trait_f1` | 0.941102 | 0.910565 | -0.030537 | 0.908822 | 0.912307 |
| `B30_seasonality_trait_f1` | 0.940832 | 0.915272 | -0.025560 | 0.906319 | 0.924225 |
| `B3_species_from_photo_top5` | 0.953485 | 0.918481 | -0.035004 | 0.916745 | 0.920217 |
| `B36_ease_of_care_trait_f1` | 0.955567 | 0.942715 | -0.012852 | 0.944504 | 0.940926 |
| `B7_family_from_phylo` | 0.952676 | 0.968333 | +0.015657 | 0.962249 | 0.974417 |
| `B12_traits_leave_one_out_f1` | 0.992887 | 0.996560 | +0.003673 | 0.996517 | 0.996602 |
| `B63_myco_from_species_f1` | 0.987818 | 1.000000 | +0.012182 | 1.000000 | 1.000000 |

## Mechanism diagnostics

Derived `*_gain` values remain diagnostics. The unchanged public evaluator logistic-renormalizes them into the harmonic score; arithmetic excludes them.

| Benchmark | Registered champion | 192-wide mesh | Delta | Seed 1337 | Seed 1338 |
|---|---:|---:|---:|---:|---:|
| `B58_lfmc_phylo_graph_gain` | 0.000000 | 0.001318 | +0.001318 | 0.000723 | 0.001913 |
| `B61_trait_phylo_graph_gain` | 0.035370 | 0.001729 | -0.033641 | 0.000871 | 0.002587 |
| `B57_flowering_phylo_graph_gain` | 0.001267 | 0.001880 | +0.000614 | 0.001850 | 0.001910 |
| `B59_pollinator_phylo_graph_gain` | 0.002130 | 0.045261 | +0.043131 | 0.038528 | 0.051993 |
| `B62_mycorrhiza_phylo_graph_gain` | 0.002913 | 0.048083 | +0.045169 | 0.041085 | 0.055080 |
| `B56_family_phylo_graph_gain` | 0.229507 | 0.075317 | -0.154190 | 0.081333 | 0.069300 |
| `B60_community_phylo_graph_gain` | 0.058334 | 0.292431 | +0.234097 | 0.278551 | 0.306311 |
| `B24_geo_information_gain` | 0.426335 | 0.382314 | -0.044020 | 0.376163 | 0.388466 |

## Inactive

| Benchmark | Reason |
|---|---|
| `B25_forecast_climate_cos` | needs temporal holdout |
| `B31_forecast_vision_cos` | needs temporal holdout |
| `B29_species_dist_30m_skill` | required inputs or labels absent |
| `B39_species_dist_3km_skill` | required inputs or labels absent |
| `B40_species_dist_300m_skill` | required inputs or labels absent |
