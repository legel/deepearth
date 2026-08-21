# DeepCal v2 baseline scorecard

Protocol: `v2-held-species-pollinator-transfer`. Config: `autoresearch/champion.yaml`. Data: 621,558 observations,
29,668-row spatial holdout, complete 64-dimensional AlphaEarth coverage. Both seeds completed 2,291 optimizer steps;
the model has 25.7M parameters and used at most 19,558.2 MB training VRAM.

| Seed | Harmonic | Arithmetic | Active capabilities | B64 held-species NDCG@10 |
|---:|---:|---:|---:|---:|
| 1337 | 0.420846 | 0.581585 | 50 | 0.172044 |
| 1338 | 0.419688 | 0.584098 | 50 | 0.174569 |
| **Mean** | **0.420267** | **0.582842** | **50** | **0.173307** |

The headline is the mean of the two independently computed seed scores. The harmonic of the mean benchmark row is
`0.420357`; it is retained separately in the machine-readable receipt and is not used as the paired-run headline.

## Human capability scorecard

Capabilities are ordered weakest-first by their two-seed mean. Full precision is retained in
`champion_scores.json`.

| Benchmark | Seed 1337 | Seed 1338 | Mean |
|---|---:|---:|---:|
| `B64_pollinator_phylo_transfer_ndcg` | 0.172044 | 0.174569 | 0.173307 |
| `B8_family_from_spacetime` | 0.175610 | 0.171734 | 0.173672 |
| `B6_family_from_env` | 0.180666 | 0.174599 | 0.177632 |
| `B50_pollinator_from_spacetime_recall` | 0.181391 | 0.182426 | 0.181908 |
| `B51_pollinator_from_env_recall` | 0.185567 | 0.185198 | 0.185383 |
| `B23_species_calibration_mrr` | 0.194182 | 0.188283 | 0.191232 |
| `B22_companions_recall` | 0.201116 | 0.196966 | 0.199041 |
| `B21_community_from_species_recall` | 0.203941 | 0.200865 | 0.202403 |
| `B20_community_from_env_recall` | 0.222886 | 0.212345 | 0.217616 |
| `B42_mycorrhiza_from_env` | 0.228203 | 0.253116 | 0.240659 |
| `B15_vision_from_aerial_cos` | 0.261292 | 0.255524 | 0.258408 |
| `B28_flowering_peak_month_mrr` | 0.375857 | 0.366828 | 0.371342 |
| `B34_lfmc_from_env` | 0.388532 | 0.401351 | 0.394941 |
| `B47_infer_naip_ir_cos` | 0.388363 | 0.401706 | 0.395034 |
| `B41_pollinator_from_species_recall` | 0.396131 | 0.395120 | 0.395625 |
| `B1_species_from_env_top10` | 0.410982 | 0.401510 | 0.406246 |
| `B5_species_from_spacetime_top10` | 0.406802 | 0.408993 | 0.407897 |
| `B54_pollinator_dist_kl` | 0.430235 | 0.433122 | 0.431679 |
| `B48_pollinator_from_photo_only_recall` | 0.438939 | 0.436545 | 0.437742 |
| `B52_pollinator_from_photo_recall` | 0.440254 | 0.437695 | 0.438974 |
| `B18_infer_climate_cos` | 0.464394 | 0.449354 | 0.456874 |
| `B19_infer_aerial_cos` | 0.487467 | 0.491572 | 0.489519 |
| `B17_infer_soil_cos` | 0.500528 | 0.494983 | 0.497756 |
| `B37_imagine_vision_bio_cos` | 0.506601 | 0.503405 | 0.505003 |
| `B45_vision_bio_leave_one_out_cos` | 0.526448 | 0.527721 | 0.527084 |
| `B53_pollinator_calibration_mrr` | 0.595848 | 0.572737 | 0.584293 |
| `B13_imagine_vision_cos` | 0.603835 | 0.605334 | 0.604585 |
| `B44_infer_topo_cos` | 0.616941 | 0.624680 | 0.620810 |
| `B43_infer_hydro_cos` | 0.665814 | 0.652413 | 0.659114 |
| `B14_vision_leave_one_out_cos` | 0.662393 | 0.671525 | 0.666959 |
| `B16_infer_clay_cos` | 0.675062 | 0.660669 | 0.667866 |
| `B27_flowering_fidelity` | 0.727692 | 0.747851 | 0.737772 |
| `B26_flowering_auc` | 0.750768 | 0.748038 | 0.749403 |
| `B2_species_from_photo_top1` | 0.821727 | 0.833558 | 0.827643 |
| `B4_species_from_photo_only_top1` | 0.821491 | 0.836760 | 0.829126 |
| `B9_phylo_from_photo_cos` | 0.824772 | 0.838059 | 0.831416 |
| `B46_infer_chm_cos` | 0.874132 | 0.862734 | 0.868433 |
| `B33_growth_rate_trait_f1` | 0.885524 | 0.889608 | 0.887566 |
| `B49_form_trait_f1` | 0.860617 | 0.927921 | 0.894269 |
| `B38_water_soil_regime_f1` | 0.900155 | 0.897000 | 0.898578 |
| `B35_sun_trait_f1` | 0.904015 | 0.914619 | 0.909317 |
| `B10_traits_from_photo_env_f1` | 0.909192 | 0.921057 | 0.915125 |
| `B11_traits_from_photo_f1` | 0.908539 | 0.922329 | 0.915434 |
| `B32_plant_type_trait_f1` | 0.934897 | 0.936317 | 0.935607 |
| `B30_seasonality_trait_f1` | 0.939670 | 0.947351 | 0.943511 |
| `B7_family_from_phylo` | 0.951362 | 0.952272 | 0.951817 |
| `B3_species_from_photo_top5` | 0.951800 | 0.954867 | 0.953334 |
| `B36_ease_of_care_trait_f1` | 0.948505 | 0.958642 | 0.953574 |
| `B63_myco_from_species_f1` | 0.987482 | 0.988915 | 0.988199 |
| `B12_traits_leave_one_out_f1` | 0.988607 | 0.994104 | 0.991355 |

## Record criterion

A candidate completes exactly 2,291 steps for seeds 1337 and 1338 on the same holdout and active suite. Promotion is
judged against this fresh baseline, not the incomparable legacy migration receipt. Report both means, every capability,
quarantined B55, and all mechanism diagnostics.

B25 and B31 require the temporal holdout. B29, B39, and B40 lack the required labels in this run. They are inactive,
never silently treated as zero.
