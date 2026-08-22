# Registered Lance-public control scorecard

Protocol: `public-main-4d6cb44-fixed-2291-steps`. Data: 621,558 observations, 29,668-row spatial holdout, complete
64-dimensional AlphaEarth coverage. Values are the unrounded mean of two checkpoint replays through the public
evaluator. This is the frozen public acceptance bar, not a score for the new mesh model. Scores produced by any other
evaluator or protocol are not comparable.

| Model | Seed | Steps | Harmonic | Arithmetic | Training VRAM |
|---|---:|---:|---:|---:|---:|
| Registered 797.1M reference | 1337 | 5,126 | 0.318693 | 0.570700 | 37,003.6 MB |
| Fixed-step 24.9M control | 1337 | 2,291 | 0.367661 | 0.578883 | 17,971.6 MB |
| Fixed-step 24.9M control | 1338 | 2,291 | 0.365992 | 0.581475 | 17,971.6 MB |
| Prior 25.4M niche fusion | 2 seeds | 2,291 | 0.373924 | 0.583204 | 19,100 MB observed |
| Prior masked-pollinator record | 2 seeds | 2,291 | 0.376617 | 0.586926 | 19,100 MB observed |
| **Hierarchical family MAP** | **1337** | **2,291** | **0.377589** | **0.586005** | **17,436 MB replay** |
| **Hierarchical family MAP** | **1338** | **2,291** | **0.379225** | **0.588743** | **17,436 MB replay** |
| **Candidate mean** | **2 seeds** | **2,291** | **0.378407** | **0.587374** | **17,436 MB replay** |
| **Delta vs prior PR record** |  |  | **+0.001790 (+0.48%)** | **+0.000448** | **no training change** |
| **Delta vs fixed-step control** |  |  | **+0.011581 (+3.16%)** | **+0.007195** | **+1,128 MB** |
| **Difference vs historical 797.1M reference** |  |  | **+0.059714 (+18.74%)** | **+0.016674** | **-48.4% training VRAM** |

## Record criterion

A replacement record uses the unchanged public evaluator, data, and holdout; completes exactly 2,291 optimizer steps;
reports every active benchmark; runs seeds 1337 and 1338; and requires each candidate seed to exceed its seed-matched
control on both harmonic and arithmetic mean. Wall time is reported as a resource diagnostic, never converted into
extra training steps. Both candidate seeds pass. The replay cache also contains distribution labels for B29/B39/B40;
those remain inactive here so the record uses the same 58/63 suite as the registered PR scorecard.

The final decoder is isolated by replaying the same checkpoints with the option disabled: harmonic
0.376196977 -> 0.377588822 and arithmetic 0.585728514 -> 0.586005331 on seed 1337; harmonic
0.377434174 -> 0.379225169 and arithmetic 0.588344374 -> 0.588742965 on seed 1338.

Only the 2,291-step control-to-candidate comparison is a matched promotion claim. The registered 797.1M row is an
older 5,126-step receipt: it is retained as historical replacement context, not treated as an equal-budget ablation.

## Mesh replay

The 13.7M-parameter mesh record replays at harmonic `0.377728` and arithmetic `0.559667` after 8,000 steps. It does
not pass the registered public control (`0.378407` / `0.587374`) and is not a public promotion claim. Its private v6
receipt is preserved in `core/champion.json` as architecture provenance only.

## Capability scorecard

Capabilities are ordered weakest-first. Deltas isolate hierarchical family MAP by replaying the same PR checkpoints with the option disabled.

| Benchmark | Prior PR checkpoint | Candidate | Delta | Seed 1337 | Seed 1338 |
|---|---:|---:|---:|---:|---:|
| `B55_pollinator_phylo_transfer_recall` | 0.038160 | 0.038147 | -0.000013 | 0.038115 | 0.038178 |
| `B8_family_from_spacetime` | 0.161352 | 0.170268 | +0.008915 | 0.171464 | 0.169071 |
| `B6_family_from_env` | 0.159364 | 0.172357 | +0.012994 | 0.169037 | 0.175677 |
| `B23_species_calibration_mrr` | 0.191363 | 0.186731 | -0.004633 | 0.186955 | 0.186506 |
| `B50_pollinator_from_spacetime_recall` | 0.188892 | 0.188731 | -0.000161 | 0.189055 | 0.188408 |
| `B51_pollinator_from_env_recall` | 0.190865 | 0.190693 | -0.000173 | 0.190476 | 0.190909 |
| `B22_companions_recall` | 0.202491 | 0.202491 | +0.000000 | 0.201561 | 0.203421 |
| `B21_community_from_species_recall` | 0.204930 | 0.204930 | +0.000000 | 0.204440 | 0.205421 |
| `B42_mycorrhiza_from_env` | 0.209131 | 0.209131 | +0.000000 | 0.206051 | 0.212210 |
| `B20_community_from_env_recall` | 0.222702 | 0.222702 | +0.000000 | 0.219552 | 0.225852 |
| `B15_vision_from_aerial_cos` | 0.254809 | 0.254809 | +0.000000 | 0.255129 | 0.254488 |
| `B28_flowering_peak_month_mrr` | 0.366676 | 0.366676 | +0.000000 | 0.366593 | 0.366759 |
| `B34_lfmc_from_env` | 0.375803 | 0.375803 | +0.000000 | 0.378235 | 0.373370 |
| `B47_infer_naip_ir_cos` | 0.392021 | 0.392021 | +0.000000 | 0.393639 | 0.390402 |
| `B5_species_from_spacetime_top10` | 0.400971 | 0.400988 | +0.000017 | 0.401342 | 0.400634 |
| `B1_species_from_env_top10` | 0.402218 | 0.402252 | +0.000034 | 0.402589 | 0.401915 |
| `B18_infer_climate_cos` | 0.457286 | 0.457286 | +0.000000 | 0.463462 | 0.451111 |
| `B41_pollinator_from_species_recall` | 0.460153 | 0.460037 | -0.000116 | 0.461319 | 0.458756 |
| `B19_infer_aerial_cos` | 0.492016 | 0.492016 | +0.000000 | 0.492993 | 0.491039 |
| `B17_infer_soil_cos` | 0.503309 | 0.503309 | +0.000000 | 0.496461 | 0.510157 |
| `B37_imagine_vision_bio_cos` | 0.505227 | 0.505227 | +0.000000 | 0.506467 | 0.503987 |
| `B48_pollinator_from_photo_only_recall` | 0.508364 | 0.508384 | +0.000020 | 0.507540 | 0.509228 |
| `B52_pollinator_from_photo_recall` | 0.508384 | 0.508385 | +0.000001 | 0.507567 | 0.509204 |
| `B45_vision_bio_leave_one_out_cos` | 0.524388 | 0.524388 | +0.000000 | 0.524146 | 0.524630 |
| `B54_pollinator_dist_kl` | 0.531038 | 0.531038 | +0.000000 | 0.529911 | 0.532164 |
| `B13_imagine_vision_cos` | 0.604508 | 0.604508 | +0.000000 | 0.604189 | 0.604827 |
| `B44_infer_topo_cos` | 0.623789 | 0.623789 | +0.000000 | 0.624860 | 0.622719 |
| `B43_infer_hydro_cos` | 0.657898 | 0.657898 | +0.000000 | 0.666535 | 0.649262 |
| `B14_vision_leave_one_out_cos` | 0.666100 | 0.666100 | +0.000000 | 0.664242 | 0.667959 |
| `B16_infer_clay_cos` | 0.668894 | 0.668894 | +0.000000 | 0.684383 | 0.653405 |
| `B53_pollinator_calibration_mrr` | 0.684013 | 0.684013 | +0.000000 | 0.671509 | 0.696516 |
| `B27_flowering_fidelity` | 0.734281 | 0.734281 | +0.000000 | 0.729921 | 0.738641 |
| `B26_flowering_auc` | 0.750456 | 0.750456 | +0.000000 | 0.757572 | 0.743341 |
| `B9_phylo_from_photo_cos` | 0.826584 | 0.826584 | +0.000000 | 0.819853 | 0.833316 |
| `B2_species_from_photo_top1` | 0.828586 | 0.828586 | +0.000000 | 0.823244 | 0.833929 |
| `B4_species_from_photo_only_top1` | 0.829429 | 0.829429 | +0.000000 | 0.823615 | 0.835243 |
| `B46_infer_chm_cos` | 0.865668 | 0.865668 | +0.000000 | 0.869409 | 0.861927 |
| `B49_form_trait_f1` | 0.872081 | 0.872081 | +0.000000 | 0.847727 | 0.896435 |
| `B38_water_soil_regime_f1` | 0.908194 | 0.908194 | +0.000000 | 0.907955 | 0.908433 |
| `B33_growth_rate_trait_f1` | 0.909192 | 0.909192 | +0.000000 | 0.906403 | 0.911981 |
| `B35_sun_trait_f1` | 0.912032 | 0.912032 | +0.000000 | 0.902254 | 0.921810 |
| `B11_traits_from_photo_f1` | 0.915436 | 0.915436 | +0.000000 | 0.906218 | 0.924654 |
| `B10_traits_from_photo_env_f1` | 0.918399 | 0.918399 | +0.000000 | 0.912487 | 0.924312 |
| `B30_seasonality_trait_f1` | 0.940832 | 0.940832 | +0.000000 | 0.938562 | 0.943102 |
| `B32_plant_type_trait_f1` | 0.941102 | 0.941102 | +0.000000 | 0.936822 | 0.945381 |
| `B7_family_from_phylo` | 0.952676 | 0.952676 | +0.000000 | 0.955137 | 0.950216 |
| `B3_species_from_photo_top5` | 0.953485 | 0.953485 | +0.000000 | 0.949676 | 0.957294 |
| `B36_ease_of_care_trait_f1` | 0.955567 | 0.955567 | +0.000000 | 0.952217 | 0.958917 |
| `B63_myco_from_species_f1` | 0.987818 | 0.987818 | +0.000000 | 0.988121 | 0.987516 |
| `B12_traits_leave_one_out_f1` | 0.992887 | 0.992887 | +0.000000 | 0.993258 | 0.992516 |

## Mechanism diagnostics

Derived `*_gain` values remain diagnostics. The unchanged evaluator applies its existing affine map only when computing the harmonic score; arithmetic excludes them.

| Benchmark | Prior PR checkpoint | Candidate | Delta | Seed 1337 | Seed 1338 |
|---|---:|---:|---:|---:|---:|
| `B58_lfmc_phylo_graph_gain` | 0.000000 | 0.000000 | +0.000000 | 0.000000 | 0.000000 |
| `B57_flowering_phylo_graph_gain` | 0.001267 | 0.001267 | +0.000000 | 0.000000 | 0.002533 |
| `B59_pollinator_phylo_graph_gain` | 0.002132 | 0.002130 | -0.000003 | 0.001908 | 0.002351 |
| `B62_mycorrhiza_phylo_graph_gain` | 0.002913 | 0.002913 | +0.000000 | 0.005827 | 0.000000 |
| `B61_trait_phylo_graph_gain` | 0.035370 | 0.035370 | +0.000000 | 0.029383 | 0.041357 |
| `B60_community_phylo_graph_gain` | 0.058334 | 0.058334 | +0.000000 | 0.060157 | 0.056510 |
| `B56_family_phylo_graph_gain` | 0.229507 | 0.229507 | +0.000000 | 0.196339 | 0.262674 |
| `B24_geo_information_gain` | 0.426368 | 0.426335 | -0.000034 | 0.420655 | 0.432014 |

## Inactive

| Benchmark | Reason |
|---|---|
| `B25_forecast_climate_cos` | needs temporal holdout |
| `B31_forecast_vision_cos` | needs temporal holdout |
| `B29_species_dist_30m_skill` | required inputs or labels absent |
| `B39_species_dist_3km_skill` | required inputs or labels absent |
| `B40_species_dist_300m_skill` | required inputs or labels absent |
