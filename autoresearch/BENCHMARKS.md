# DeepCal compact niche-fusion record scorecard

Protocol: `public-main-4d6cb44-fixed-2291-steps`. Config: `autoresearch/champion.yaml`. Data: 621,558 observations,
29,668-row spatial holdout, complete 64-dimensional AlphaEarth coverage. Values are the unrounded mean of two
checkpoint replays through the unchanged public evaluator.

| Model | Seed | Steps | Harmonic | Arithmetic | Training VRAM |
|---|---:|---:|---:|---:|---:|
| Registered 797.1M reference | 1337 | 5,126 | 0.318693 | 0.570700 | 37,003.6 MB |
| Prior PR compact mean (time-budget receipt) | 2 seeds | 2,282.5 mean | 0.364218 | 0.575520 | 17,971.6 MB |
| Fixed-step 24.9M control | 1337 | 2,291 | 0.367661 | 0.578883 | 17,971.6 MB |
| Fixed-step 24.9M control | 1338 | 2,291 | 0.365992 | 0.581475 | 17,971.6 MB |
| **25.4M niche fusion** | **1337** | **2,291** | **0.373074** | **0.581691** | **19,100 MB observed** |
| **25.4M niche fusion** | **1338** | **2,291** | **0.374775** | **0.584717** | **19,100 MB observed** |
| **Niche-fusion mean** | **2 seeds** | **2,291** | **0.373924** | **0.583204** | **19,100 MB observed** |
| **Delta vs fixed-step control** |  |  | **+0.007098 (+1.93%)** | **+0.003025** | **+1,128 MB** |
| **Delta vs 797.1M reference** |  |  | **+0.055231 (+17.33%)** | **+0.012504** | **-48.4%** |

## Record criterion

A replacement record uses the unchanged public evaluator, data, and holdout; completes exactly 2,291 optimizer steps;
reports every active benchmark; runs seeds 1337 and 1338; and requires each candidate seed to exceed its seed-matched
control on both harmonic and arithmetic mean. Wall time is reported as a resource diagnostic, never converted into
extra training steps. Both candidate seeds pass.

## Capability scorecard

Capabilities are ordered weakest-first by the candidate mean. Deltas use the paired fixed-step control.

| Benchmark | Control mean | Candidate mean | Delta | Seed 1337 | Seed 1338 |
|---|---:|---:|---:|---:|---:|
| `B55_pollinator_phylo_transfer_recall` | 0.038175 | 0.038138 | -0.000038 | 0.038099 | 0.038177 |
| `B6_family_from_env` | 0.148915 | 0.159515 | +0.010601 | 0.158218 | 0.160813 |
| `B8_family_from_spacetime` | 0.145881 | 0.160965 | +0.015084 | 0.161689 | 0.160240 |
| `B50_pollinator_from_spacetime_recall` | 0.173768 | 0.184510 | +0.010742 | 0.184564 | 0.184456 |
| `B51_pollinator_from_env_recall` | 0.171481 | 0.184538 | +0.013057 | 0.184163 | 0.184912 |
| `B23_species_calibration_mrr` | 0.172755 | 0.191710 | +0.018955 | 0.192102 | 0.191318 |
| `B22_companions_recall` | 0.201295 | 0.202349 | +0.001054 | 0.201418 | 0.203280 |
| `B21_community_from_species_recall` | 0.204660 | 0.205023 | +0.000362 | 0.204151 | 0.205894 |
| `B42_mycorrhiza_from_env` | 0.207777 | 0.208953 | +0.001176 | 0.205995 | 0.211911 |
| `B20_community_from_env_recall` | 0.217219 | 0.222499 | +0.005280 | 0.219353 | 0.225646 |
| `B15_vision_from_aerial_cos` | 0.254668 | 0.254783 | +0.000115 | 0.255252 | 0.254313 |
| `B28_flowering_peak_month_mrr` | 0.366293 | 0.366489 | +0.000196 | 0.365970 | 0.367008 |
| `B34_lfmc_from_env` | 0.368882 | 0.370897 | +0.002015 | 0.373093 | 0.368700 |
| `B47_infer_naip_ir_cos` | 0.400820 | 0.392079 | -0.008740 | 0.393762 | 0.390397 |
| `B5_species_from_spacetime_top10` | 0.371663 | 0.401392 | +0.029729 | 0.401375 | 0.401409 |
| `B1_species_from_env_top10` | 0.354641 | 0.403229 | +0.048588 | 0.404038 | 0.402420 |
| `B48_pollinator_from_photo_only_recall` | 0.419422 | 0.419360 | -0.000062 | 0.414556 | 0.424165 |
| `B52_pollinator_from_photo_recall` | 0.420977 | 0.421213 | +0.000236 | 0.417016 | 0.425410 |
| `B18_infer_climate_cos` | 0.459803 | 0.457355 | -0.002448 | 0.463626 | 0.451085 |
| `B41_pollinator_from_species_recall` | 0.460338 | 0.460345 | +0.000007 | 0.461703 | 0.458987 |
| `B19_infer_aerial_cos` | 0.497654 | 0.492074 | -0.005580 | 0.493021 | 0.491127 |
| `B17_infer_soil_cos` | 0.510646 | 0.503335 | -0.007311 | 0.496606 | 0.510064 |
| `B37_imagine_vision_bio_cos` | 0.505623 | 0.505222 | -0.000401 | 0.506464 | 0.503981 |
| `B45_vision_bio_leave_one_out_cos` | 0.524562 | 0.524388 | -0.000174 | 0.524144 | 0.524631 |
| `B54_pollinator_dist_kl` | 0.528459 | 0.531040 | +0.002581 | 0.529945 | 0.532134 |
| `B13_imagine_vision_cos` | 0.604462 | 0.604507 | +0.000046 | 0.604195 | 0.604820 |
| `B44_infer_topo_cos` | 0.625011 | 0.623805 | -0.001206 | 0.624894 | 0.622716 |
| `B43_infer_hydro_cos` | 0.661057 | 0.657912 | -0.003145 | 0.666631 | 0.649193 |
| `B14_vision_leave_one_out_cos` | 0.666299 | 0.666098 | -0.000201 | 0.664240 | 0.667956 |
| `B16_infer_clay_cos` | 0.663713 | 0.668923 | +0.005210 | 0.684517 | 0.653329 |
| `B53_pollinator_calibration_mrr` | 0.678240 | 0.683551 | +0.005311 | 0.671525 | 0.695578 |
| `B27_flowering_fidelity` | 0.733962 | 0.734310 | +0.000349 | 0.729747 | 0.738874 |
| `B26_flowering_auc` | 0.751037 | 0.750813 | -0.000224 | 0.757636 | 0.743990 |
| `B9_phylo_from_photo_cos` | 0.826648 | 0.826586 | -0.000062 | 0.819868 | 0.833303 |
| `B2_species_from_photo_top1` | 0.830356 | 0.827963 | -0.002393 | 0.822806 | 0.833120 |
| `B4_species_from_photo_only_top1` | 0.831772 | 0.830322 | -0.001449 | 0.824660 | 0.835985 |
| `B46_infer_chm_cos` | 0.863820 | 0.865741 | +0.001920 | 0.869505 | 0.861976 |
| `B49_form_trait_f1` | 0.845490 | 0.873804 | +0.028314 | 0.847340 | 0.900267 |
| `B38_water_soil_regime_f1` | 0.909568 | 0.906860 | -0.002708 | 0.907398 | 0.906322 |
| `B33_growth_rate_trait_f1` | 0.921155 | 0.909036 | -0.012118 | 0.905859 | 0.912214 |
| `B35_sun_trait_f1` | 0.911435 | 0.911318 | -0.000116 | 0.900351 | 0.922286 |
| `B11_traits_from_photo_f1` | 0.916162 | 0.914994 | -0.001168 | 0.906931 | 0.923057 |
| `B10_traits_from_photo_env_f1` | 0.916988 | 0.918148 | +0.001160 | 0.912009 | 0.924287 |
| `B30_seasonality_trait_f1` | 0.942537 | 0.940518 | -0.002018 | 0.938345 | 0.942691 |
| `B32_plant_type_trait_f1` | 0.939848 | 0.941078 | +0.001230 | 0.937167 | 0.944989 |
| `B7_family_from_phylo` | 0.952609 | 0.952238 | -0.000371 | 0.954766 | 0.949710 |
| `B3_species_from_photo_top5` | 0.953452 | 0.953890 | +0.000438 | 0.950283 | 0.957496 |
| `B36_ease_of_care_trait_f1` | 0.956304 | 0.955709 | -0.000595 | 0.952217 | 0.959200 |
| `B63_myco_from_species_f1` | 0.987793 | 0.987767 | -0.000027 | 0.988073 | 0.987460 |
| `B12_traits_leave_one_out_f1` | 0.992842 | 0.992915 | +0.000073 | 0.993254 | 0.992576 |

## Mechanism diagnostics

Derived `*_gain` values remain diagnostics. The unchanged evaluator applies its existing affine map only when
computing the harmonic score; arithmetic excludes them.

| Benchmark | Control mean | Candidate raw | Delta | Seed 1337 | Seed 1338 |
|---|---:|---:|---:|---:|---:|
| `B58_lfmc_phylo_graph_gain` | 0.000093 | 0.000000 | -0.000093 | 0.000000 | 0.000000 |
| `B59_pollinator_phylo_graph_gain` | 0.000000 | 0.000051 | +0.000051 | 0.000000 | 0.000102 |
| `B57_flowering_phylo_graph_gain` | 0.000597 | 0.000540 | -0.000057 | 0.000000 | 0.001080 |
| `B62_mycorrhiza_phylo_graph_gain` | 0.001019 | 0.003362 | +0.002343 | 0.004591 | 0.002133 |
| `B61_trait_phylo_graph_gain` | 0.034643 | 0.035080 | +0.000437 | 0.028232 | 0.041928 |
| `B60_community_phylo_graph_gain` | 0.063861 | 0.058153 | -0.005708 | 0.059955 | 0.056352 |
| `B56_family_phylo_graph_gain` | 0.216749 | 0.229102 | +0.012353 | 0.195564 | 0.262640 |
| `B24_geo_information_gain` | 0.475715 | 0.424734 | -0.050981 | 0.418768 | 0.430700 |

## Inactive

| Benchmark | Reason |
|---|---|
| `B25_forecast_climate_cos` | needs temporal holdout |
| `B31_forecast_vision_cos` | needs temporal holdout |
| `B29_species_dist_30m_skill` | required inputs or labels absent |
| `B39_species_dist_3km_skill` | required inputs or labels absent |
| `B40_species_dist_300m_skill` | required inputs or labels absent |
