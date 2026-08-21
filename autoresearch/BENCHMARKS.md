# DeepCal v2 baseline scorecard

Protocol: `v2-held-species-pollinator-transfer`. Config: `autoresearch/champion.yaml`. Data: 621,558 observations,
29,668-row spatial holdout, complete 64-dimensional AlphaEarth coverage. Both seeds completed 2,291 optimizer steps;
the model has 25,696,755 parameters and used at most 19,558.2 MB training VRAM.

| Seed | Harmonic | Arithmetic | Active capabilities | B64 held-species NDCG@10 |
|---:|---:|---:|---:|---:|
| 1337 | 0.418762 | 0.581057 | 50 | 0.172044 |
| 1338 | 0.419921 | 0.584020 | 50 | 0.174569 |
| **Mean** | **0.419341** | **0.582539** | **50** | **0.173307** |

The headline is the mean of the two independently computed seed scores. The harmonic of the mean benchmark row is
`0.419584`; it is retained separately in the machine-readable receipt and is not used as the paired-run headline.

## Human capability scorecard

Capabilities are ordered weakest-first by their two-seed mean. Full precision is retained in
`champion_scores.json`.

| Benchmark | Seed 1337 | Seed 1338 | Mean |
|---|---:|---:|---:|
| `B64_pollinator_phylo_transfer_ndcg` | 0.172044 | 0.174569 | 0.173307 |
| `B8_family_from_spacetime` | 0.175677 | 0.171329 | 0.173503 |
| `B6_family_from_env` | 0.180666 | 0.174599 | 0.177632 |
| `B50_pollinator_from_spacetime_recall` | 0.182364 | 0.183174 | 0.182769 |
| `B51_pollinator_from_env_recall` | 0.185400 | 0.185869 | 0.185634 |
| `B23_species_calibration_mrr` | 0.194940 | 0.187359 | 0.191150 |
| `B22_companions_recall` | 0.200481 | 0.197134 | 0.198807 |
| `B21_community_from_species_recall` | 0.203909 | 0.201108 | 0.202509 |
| `B20_community_from_env_recall` | 0.222799 | 0.210828 | 0.216813 |
| `B42_mycorrhiza_from_env` | 0.203165 | 0.253248 | 0.228206 |
| `B15_vision_from_aerial_cos` | 0.256000 | 0.258026 | 0.257013 |
| `B28_flowering_peak_month_mrr` | 0.376179 | 0.366890 | 0.371534 |
| `B41_pollinator_from_species_recall` | 0.395994 | 0.395091 | 0.395542 |
| `B34_lfmc_from_env` | 0.387415 | 0.404749 | 0.396082 |
| `B47_infer_naip_ir_cos` | 0.389209 | 0.404174 | 0.396692 |
| `B1_species_from_env_top10` | 0.413476 | 0.402353 | 0.407914 |
| `B5_species_from_spacetime_top10` | 0.408049 | 0.408150 | 0.408100 |
| `B54_pollinator_dist_kl` | 0.430411 | 0.433118 | 0.431764 |
| `B48_pollinator_from_photo_only_recall` | 0.439985 | 0.436505 | 0.438245 |
| `B52_pollinator_from_photo_recall` | 0.440500 | 0.437057 | 0.438778 |
| `B18_infer_climate_cos` | 0.460729 | 0.452677 | 0.456703 |
| `B19_infer_aerial_cos` | 0.487333 | 0.493831 | 0.490582 |
| `B17_infer_soil_cos` | 0.498696 | 0.497983 | 0.498339 |
| `B37_imagine_vision_bio_cos` | 0.506594 | 0.503399 | 0.504997 |
| `B45_vision_bio_leave_one_out_cos` | 0.526449 | 0.527717 | 0.527083 |
| `B53_pollinator_calibration_mrr` | 0.595462 | 0.573202 | 0.584332 |
| `B13_imagine_vision_cos` | 0.603793 | 0.605345 | 0.604569 |
| `B44_infer_topo_cos` | 0.612707 | 0.627816 | 0.620262 |
| `B43_infer_hydro_cos` | 0.664585 | 0.650429 | 0.657507 |
| `B14_vision_leave_one_out_cos` | 0.662381 | 0.671536 | 0.666958 |
| `B16_infer_clay_cos` | 0.672740 | 0.661480 | 0.667110 |
| `B27_flowering_fidelity` | 0.727995 | 0.742783 | 0.735389 |
| `B26_flowering_auc` | 0.750321 | 0.746571 | 0.748446 |
| `B2_species_from_photo_top1` | 0.824087 | 0.833255 | 0.828671 |
| `B4_species_from_photo_only_top1` | 0.824255 | 0.835884 | 0.830069 |
| `B9_phylo_from_photo_cos` | 0.825107 | 0.837364 | 0.831236 |
| `B46_infer_chm_cos` | 0.871997 | 0.862192 | 0.867094 |
| `B33_growth_rate_trait_f1` | 0.887442 | 0.888849 | 0.888145 |
| `B49_form_trait_f1` | 0.860839 | 0.925037 | 0.892938 |
| `B38_water_soil_regime_f1` | 0.900942 | 0.896245 | 0.898593 |
| `B35_sun_trait_f1` | 0.905469 | 0.914598 | 0.910034 |
| `B10_traits_from_photo_env_f1` | 0.909912 | 0.919837 | 0.914874 |
| `B11_traits_from_photo_f1` | 0.910011 | 0.920853 | 0.915432 |
| `B32_plant_type_trait_f1` | 0.934321 | 0.936009 | 0.935165 |
| `B30_seasonality_trait_f1` | 0.939254 | 0.944998 | 0.942126 |
| `B7_family_from_phylo` | 0.951834 | 0.951665 | 0.951749 |
| `B36_ease_of_care_trait_f1` | 0.950090 | 0.956711 | 0.953401 |
| `B3_species_from_photo_top5` | 0.952710 | 0.954294 | 0.953502 |
| `B63_myco_from_species_f1` | 0.987533 | 0.989006 | 0.988270 |
| `B12_traits_leave_one_out_f1` | 0.988608 | 0.994104 | 0.991356 |

## Record criterion

A candidate completes exactly 2,291 steps for seeds 1337 and 1338 on the same holdout and active suite. Promotion is
judged against this fresh baseline, not the incomparable legacy migration receipt. Report both means, every capability,
quarantined B55, and all mechanism diagnostics.

B25 and B31 require the temporal holdout. B29, B39, and B40 lack the required labels in this run. They are inactive,
never silently treated as zero.
