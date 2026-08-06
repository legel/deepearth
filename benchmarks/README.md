# DeepEarth benchmarks

The full evaluation suite, measured on the current `main`. Every number here comes from one reproducible
run — nothing is copied forward from an earlier commit.

## Current baseline

| | |
|---|---|
| **Commit** | `4d6cb44` |
| **Harmonic mean** | **0.318693** |
| **Arithmetic mean** | **0.5707** |
| Active benchmarks | 58 of 63 |
| Held-out rows | 29,668 (spatial holdout) |
| Observations | 621,558 (train 591,890) |
| Parameters | 797.1M |
| Training | 600 s budget, 5,126 steps, seed 1337, bf16 |
| Hardware | 1x NVIDIA RTX PRO 6000 Blackwell, peak 37,003 MB |
| Eval | 22.4 s |

```
python -m deepearth.autoresearch.train autoresearch/deepcal.yaml \
  --cache_dir <prepared-cache> --device cuda:0 --time_budget 600
```

The harmonic mean is the headline because it cannot be raised by trading one capability for another —
lifting the weakest helps most. The arithmetic mean is reported alongside because the harmonic is pinned
by the current weakest benchmarks and moves slowly.

## Capabilities (50 scored, weakest first)

Worst-first is the priority order: these are where the model has the most to gain.

| Benchmark | Score |
|---|---:|
| `B55_pollinator_phylo_transfer_recall` | 0.038 |
| `B23_species_calibration_mrr` | 0.074 |
| `B6_family_from_env` | 0.084 |
| `B8_family_from_spacetime` | 0.085 |
| `B50_pollinator_from_spacetime_recall` | 0.146 |
| `B51_pollinator_from_env_recall` | 0.146 |
| `B1_species_from_env_top10` | 0.157 |
| `B42_mycorrhiza_from_env` | 0.185 |
| `B15_vision_from_aerial_cos` | 0.220 |
| `B20_community_from_env_recall` | 0.247 |
| `B5_species_from_spacetime_top10` | 0.249 |
| `B22_companions_recall` | 0.265 |
| `B21_community_from_species_recall` | 0.277 |
| `B47_infer_naip_ir_cos` | 0.320 |
| `B34_lfmc_from_env` | 0.387 |
| `B19_infer_aerial_cos` | 0.415 |
| `B44_infer_topo_cos` | 0.433 |
| `B17_infer_soil_cos` | 0.437 |
| `B28_flowering_peak_month_mrr` | 0.441 |
| `B48_pollinator_from_photo_only_recall` | 0.456 |
| `B52_pollinator_from_photo_recall` | 0.457 |
| `B41_pollinator_from_species_recall` | 0.507 |
| `B43_infer_hydro_cos` | 0.525 |
| `B37_imagine_vision_bio_cos` | 0.536 |
| `B16_infer_clay_cos` | 0.562 |
| `B45_vision_bio_leave_one_out_cos` | 0.566 |
| `B54_pollinator_dist_kl` | 0.590 |
| `B18_infer_climate_cos` | 0.632 |
| `B13_imagine_vision_cos` | 0.660 |
| `B46_infer_chm_cos` | 0.671 |
| `B26_flowering_auc` | 0.705 |
| `B27_flowering_fidelity` | 0.718 |
| `B53_pollinator_calibration_mrr` | 0.720 |
| `B14_vision_leave_one_out_cos` | 0.756 |
| `B2_species_from_photo_top1` | 0.848 |
| `B4_species_from_photo_only_top1` | 0.852 |
| `B49_form_trait_f1` | 0.858 |
| `B9_phylo_from_photo_cos` | 0.902 |
| `B38_water_soil_regime_f1` | 0.908 |
| `B35_sun_trait_f1` | 0.921 |
| `B11_traits_from_photo_f1` | 0.923 |
| `B10_traits_from_photo_env_f1` | 0.924 |
| `B32_plant_type_trait_f1` | 0.943 |
| `B30_seasonality_trait_f1` | 0.948 |
| `B33_growth_rate_trait_f1` | 0.952 |
| `B36_ease_of_care_trait_f1` | 0.952 |
| `B3_species_from_photo_top5` | 0.969 |
| `B7_family_from_phylo` | 0.972 |
| `B12_traits_leave_one_out_f1` | 0.998 |
| `B63_myco_from_species_f1` | 0.998 |

## Ablation deltas (8)

Derived differences that isolate one mechanism's contribution — a capability with a subsystem minus the
same capability without it. Reported on a compressed scale where 0.5 is neutral, so they can never
dominate the mean. A delta at 0.000 means that subsystem contributes nothing measurable to that
capability, which is a finding worth acting on.

| Benchmark | Raw delta | Net contribution |
|---|---:|---:|
| `B24_geo_information_gain` | 0.691 | 0.999 |
| `B58_lfmc_phylo_graph_gain` | 0.065 | 0.656 |
| `B56_family_phylo_graph_gain` | 0.009 | 0.524 |
| `B61_trait_phylo_graph_gain` | 0.007 | 0.517 |
| `B62_mycorrhiza_phylo_graph_gain` | 0.000 | 0.500 |
| `B57_flowering_phylo_graph_gain` | 0.000 | 0.500 |
| `B59_pollinator_phylo_graph_gain` | 0.000 | 0.500 |
| `B60_community_phylo_graph_gain` | 0.000 | 0.500 |

## Inactive (5)

Declared capabilities this run could not produce, with the reason.

| Benchmark | Reason |
|---|---|
| `B25_forecast_climate_cos` | needs holdout: temporal (strictly-future forecast split) |
| `B31_forecast_vision_cos` | needs holdout: temporal (strictly-future forecast split) |
| `B29_species_dist_30m_skill` | required inputs/labels absent for this run |
| `B39_species_dist_3km_skill` | required inputs/labels absent for this run |
| `B40_species_dist_300m_skill` | required inputs/labels absent for this run |

## Keeping this current

This file is regenerated from a real run whenever the champion changes — see `science.md` rule 33. A
benchmark table that lags the model is worse than none, because it is quoted as if it were current.
