# DeepCal compact record scorecard

Protocol: `public-main-4d6cb44-fixed-600s`. Config: `autoresearch/champion.yaml`. Data: 621,558 observations,
29,668-row spatial holdout, complete 64-dimensional AlphaEarth coverage. Candidate values are the mean of two
600-second receipts from the unchanged public evaluator.

| Model | Seed | Steps completed | Harmonic | Arithmetic | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| Registered 797.1M reference | 1337 | 5,126 | 0.318693 | 0.570700 | 37,003.6 MB |
| 24.9M compact | 1337 | 2,291 | 0.363696 | 0.576871 | 17,971.6 MB |
| 24.9M compact | 1338 | 2,274 | 0.364740 | 0.574168 | 17,971.6 MB |
| **24.9M mean** | **2 seeds** | **2,282.5** | **0.364218** | **0.575520** | **17,971.6 MB** |
| **Delta vs reference** |  |  | **+0.045525** | **+0.004820** | **-51.4%** |

## Record criterion

A replacement record uses the unchanged public evaluator, data, holdout, and 600-second training budget; reports
every active benchmark; runs seeds 1337 and 1338; and requires each seed to exceed the registered reference on both
harmonic and arithmetic mean. Both candidate seeds pass.

## Capability scorecard

Capabilities are ordered weakest first. Reference values are from the registered public-main baseline receipt.

| Benchmark | Reference | Two-seed mean | Delta | Seed 1337 | Seed 1338 |
|---|---:|---:|---:|---:|---:|
| `B55_pollinator_phylo_transfer_recall` | 0.038000 | 0.039269 | +0.001269 | 0.038182 | 0.040356 |
| `B8_family_from_spacetime` | 0.085000 | 0.144010 | +0.059010 | 0.144567 | 0.143454 |
| `B6_family_from_env` | 0.084000 | 0.147954 | +0.063954 | 0.147330 | 0.148578 |
| `B51_pollinator_from_env_recall` | 0.146000 | 0.167639 | +0.021639 | 0.165500 | 0.169779 |
| `B23_species_calibration_mrr` | 0.074000 | 0.168410 | +0.094410 | 0.169932 | 0.166889 |
| `B50_pollinator_from_spacetime_recall` | 0.146000 | 0.173606 | +0.027606 | 0.171246 | 0.175966 |
| `B42_mycorrhiza_from_env` | 0.185000 | 0.193595 | +0.008595 | 0.190994 | 0.196196 |
| `B20_community_from_env_recall` | 0.247000 | 0.194574 | -0.052426 | 0.197985 | 0.191162 |
| `B22_companions_recall` | 0.265000 | 0.195569 | -0.069431 | 0.200654 | 0.190483 |
| `B21_community_from_species_recall` | 0.277000 | 0.202271 | -0.074729 | 0.208968 | 0.195573 |
| `B15_vision_from_aerial_cos` | 0.220000 | 0.253210 | +0.033210 | 0.252612 | 0.253808 |
| `B1_species_from_env_top10` | 0.157000 | 0.343704 | +0.186704 | 0.346434 | 0.340973 |
| `B5_species_from_spacetime_top10` | 0.249000 | 0.357287 | +0.108287 | 0.361467 | 0.353108 |
| `B34_lfmc_from_env` | 0.387000 | 0.374173 | -0.012827 | 0.398606 | 0.349741 |
| `B28_flowering_peak_month_mrr` | 0.441000 | 0.375121 | -0.065879 | 0.376787 | 0.373454 |
| `B47_infer_naip_ir_cos` | 0.320000 | 0.394719 | +0.074719 | 0.401227 | 0.388211 |
| `B48_pollinator_from_photo_only_recall` | 0.456000 | 0.424396 | -0.031604 | 0.421565 | 0.427226 |
| `B52_pollinator_from_photo_recall` | 0.457000 | 0.427904 | -0.029096 | 0.425817 | 0.429992 |
| `B18_infer_climate_cos` | 0.632000 | 0.458826 | -0.173174 | 0.461493 | 0.456159 |
| `B41_pollinator_from_species_recall` | 0.507000 | 0.469630 | -0.037370 | 0.475574 | 0.463687 |
| `B19_infer_aerial_cos` | 0.415000 | 0.491991 | +0.076991 | 0.496274 | 0.487709 |
| `B37_imagine_vision_bio_cos` | 0.536000 | 0.508539 | -0.027461 | 0.508169 | 0.508910 |
| `B45_vision_bio_leave_one_out_cos` | 0.566000 | 0.533171 | -0.032829 | 0.533833 | 0.532509 |
| `B17_infer_soil_cos` | 0.437000 | 0.541737 | +0.104737 | 0.524310 | 0.559164 |
| `B54_pollinator_dist_kl` | 0.590000 | 0.552245 | -0.037755 | 0.554915 | 0.549575 |
| `B13_imagine_vision_cos` | 0.660000 | 0.610033 | -0.049967 | 0.609347 | 0.610719 |
| `B44_infer_topo_cos` | 0.433000 | 0.643695 | +0.210695 | 0.635972 | 0.651418 |
| `B16_infer_clay_cos` | 0.562000 | 0.652900 | +0.090900 | 0.667275 | 0.638526 |
| `B53_pollinator_calibration_mrr` | 0.720000 | 0.667981 | -0.052019 | 0.665568 | 0.670393 |
| `B43_infer_hydro_cos` | 0.525000 | 0.690921 | +0.165921 | 0.692738 | 0.689105 |
| `B14_vision_leave_one_out_cos` | 0.756000 | 0.692874 | -0.063126 | 0.688419 | 0.697328 |
| `B27_flowering_fidelity` | 0.718000 | 0.729069 | +0.011069 | 0.714649 | 0.743489 |
| `B26_flowering_auc` | 0.705000 | 0.747963 | +0.042963 | 0.748612 | 0.747314 |
| `B2_species_from_photo_top1` | 0.848000 | 0.802059 | -0.045941 | 0.802784 | 0.801335 |
| `B4_species_from_photo_only_top1` | 0.852000 | 0.804048 | -0.047952 | 0.805683 | 0.802413 |
| `B49_form_trait_f1` | 0.858000 | 0.811318 | -0.046682 | 0.797713 | 0.824922 |
| `B9_phylo_from_photo_cos` | 0.902000 | 0.843236 | -0.058764 | 0.841943 | 0.844529 |
| `B46_infer_chm_cos` | 0.671000 | 0.879135 | +0.208135 | 0.883829 | 0.874441 |
| `B33_growth_rate_trait_f1` | 0.952000 | 0.884570 | -0.067430 | 0.873065 | 0.896076 |
| `B38_water_soil_regime_f1` | 0.908000 | 0.888319 | -0.019681 | 0.894636 | 0.882002 |
| `B10_traits_from_photo_env_f1` | 0.924000 | 0.890170 | -0.033830 | 0.893086 | 0.887254 |
| `B11_traits_from_photo_f1` | 0.923000 | 0.891744 | -0.031256 | 0.895396 | 0.888092 |
| `B30_seasonality_trait_f1` | 0.948000 | 0.894353 | -0.053647 | 0.937416 | 0.851291 |
| `B35_sun_trait_f1` | 0.921000 | 0.900463 | -0.020537 | 0.893148 | 0.907777 |
| `B32_plant_type_trait_f1` | 0.943000 | 0.910052 | -0.032948 | 0.915007 | 0.905097 |
| `B7_family_from_phylo` | 0.972000 | 0.934610 | -0.037390 | 0.942194 | 0.927026 |
| `B3_species_from_photo_top5` | 0.969000 | 0.943660 | -0.025340 | 0.947486 | 0.939834 |
| `B36_ease_of_care_trait_f1` | 0.952000 | 0.943964 | -0.008036 | 0.939067 | 0.948862 |
| `B63_myco_from_species_f1` | 0.998000 | 0.989686 | -0.008314 | 0.988516 | 0.990856 |
| `B12_traits_leave_one_out_f1` | 0.998000 | 0.995607 | -0.002393 | 0.995582 | 0.995631 |

## Mechanism diagnostics

Derived `*_gain` values are reported raw. The unchanged evaluator maps them affinely to the harmonic contribution
(`0.5 + 0.5 × delta`); arithmetic excludes them.

| Benchmark | Reference raw | Candidate raw | Delta | Net contribution | Seed 1337 | Seed 1338 |
|---|---:|---:|---:|---:|---:|---:|
| `B58_lfmc_phylo_graph_gain` | 0.065000 | 0.000000 | -0.065000 | 0.500000 | 0.000000 | 0.000000 |
| `B59_pollinator_phylo_graph_gain` | 0.000000 | 0.000000 | +0.000000 | 0.500000 | 0.000000 | 0.000000 |
| `B57_flowering_phylo_graph_gain` | 0.000000 | 0.000484 | +0.000484 | 0.500242 | 0.000967 | 0.000000 |
| `B62_mycorrhiza_phylo_graph_gain` | 0.000000 | 0.002470 | +0.002470 | 0.501235 | 0.000000 | 0.004940 |
| `B61_trait_phylo_graph_gain` | 0.007000 | 0.027617 | +0.020617 | 0.513808 | 0.032449 | 0.022784 |
| `B60_community_phylo_graph_gain` | 0.000000 | 0.034196 | +0.034196 | 0.517098 | 0.035616 | 0.032777 |
| `B56_family_phylo_graph_gain` | 0.009000 | 0.117281 | +0.108281 | 0.558641 | 0.103445 | 0.131118 |
| `B24_geo_information_gain` | 0.691000 | 0.458356 | -0.232644 | 0.729178 | 0.456350 | 0.460361 |

## Inactive

| Benchmark | Reason |
|---|---|
| `B25_forecast_climate_cos` | needs temporal holdout |
| `B31_forecast_vision_cos` | needs temporal holdout |
| `B29_species_dist_30m_skill` | required inputs or labels absent |
| `B39_species_dist_3km_skill` | required inputs or labels absent |
| `B40_species_dist_300m_skill` | required inputs or labels absent |
