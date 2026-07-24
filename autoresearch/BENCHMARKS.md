# DeepCal — committed baseline benchmark

Reference scores for the /autoresearch pipeline. Reproduce, then push against these.

**Config:** `autoresearch/deepcal.yaml` (default, pollinator subsystem ON), `operator: latent-clade`, `bioclip_init: true`, spatial holdout, bf16, **batch 256, 8000 steps** on a single RTX 3090 (24GB). 
On a ≥40GB GPU use the default `batch: 512` for ~2–3% higher scores. Command:
```
python -m deepearth.autoresearch.train autoresearch/deepcal.yaml --steps 8000 --device cuda:0
```

**NET SCORE — harmonic mean: 0.020421  |  arithmetic mean: 0.4461** (over the active B1–B55 suite; the harmonic is dragged by the hard low benchmarks — that is the honest full-suite number and the target to raise).

| Benchmark | Score |
|---|---|
| B1_species_from_env_top10 | 0.234 |
| B2_species_from_photo_top1 | 0.900 |
| B3_species_from_photo_top5 | 0.972 |
| B4_species_from_photo_only_top1 | 0.898 |
| B5_species_from_spacetime_top10 | 0.254 |
| B6_family_from_env | 0.093 |
| B7_family_from_phylo | 0.969 |
| B8_family_from_spacetime | 0.083 |
| B9_phylo_from_photo_cos | 0.884 |
| B10_traits_from_photo_env_f1 | 0.952 |
| B11_traits_from_photo_f1 | 0.952 |
| B12_traits_leave_one_out_f1 | 0.994 |
| B13_imagine_vision_cos | 0.553 |
| B14_vision_leave_one_out_cos | 0.671 |
| B15_vision_from_aerial_cos | 0.133 |
| B16_infer_clay_cos | 0.286 |
| B17_infer_soil_cos | 0.625 |
| B18_infer_climate_cos | 0.710 |
| B19_infer_aerial_cos | 0.000 |
| B20_community_from_env_recall | 0.004 |
| B21_community_from_species_recall | 0.006 |
| B22_companions_recall | 0.004 |
| B23_species_calibration_mrr | 0.109 |
| B24_geo_information_gain | 0.666 |
| B26_flowering_auc | 0.504 |
| B27_flowering_fidelity | 0.873 |
| B28_flowering_peak_month_mrr | 0.300 |
| B29_species_dist_30m_skill | 0.122 |
| B30_seasonality_trait_f1 | 0.968 |
| B34_lfmc_from_env | 0.214 |
| B38_water_soil_regime_f1 | 0.937 |
| B39_species_dist_3km_skill | 0.011 |
| B40_species_dist_300m_skill | 0.033 |
| B41_pollinator_from_species_recall | 0.198 |
| B42_mycorrhiza_from_env | 0.071 |
| B43_infer_hydro_cos | 0.736 |
| B49_form_trait_f1 | 0.939 |
| B51_pollinator_from_env_recall | 0.174 |
| B52_pollinator_from_photo_recall | 0.497 |
| B53_pollinator_calibration_mrr | 0.280 |
| B54_pollinator_dist_kl | 0.095 |
| B55_pollinator_phylo_transfer_recall | 0.054 |
| B56_family_phylo_graph_gain | 0.500 |
| B57_flowering_phylo_graph_gain | 0.009 |
| B58_lfmc_phylo_graph_gain | 0.001 |
| B59_pollinator_phylo_graph_gain | 0.000 |
| B60_community_phylo_graph_gain | 0.001 |
| B62_mycorrhiza_phylo_graph_gain | 0.000 |

## Phylogenetic graph-gain (ablation delta: graph − graph-ablated, spatial holdout)
- **B56 family_phylo_graph_gain = 0.500** — the LCA species graph substantially improves family prediction (the core rule-29 signal).
- B57–B62 (flowering/lfmc/community/mycorrhiza/pollinator) ≈ 0 under spatial holdout — these traits are predicted from environment, so the graph is redundant there; the phylo-generalization test needs an env-masked benchmark (roadmap).

## Known low benchmarks (improvement targets)
B21 community, B39/B40 species-distribution skill, B15 vision-from-aerial, B6/B8 family-from-env/spacetime, B54/B55 pollinator dist/phylo-transfer, B1/B5 species-from-env/spacetime. The rule-25 mask-and-reconstruct loss (`phylo_mask_weight`, currently 0 in the default) is the leading lever under test.
