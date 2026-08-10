# DeepCal champion report

## 24.9M fixed-budget compact record

The compact model clears both registered public aggregates at the same 600-second budget while reducing parameters
by 96.9% and peak VRAM by 51.4%. Both required seeds pass independently.

| Model | Seed | Steps | Harmonic | Arithmetic |
|---|---:|---:|---:|---:|
| Registered 797.1M reference | 1337 | 5,126 | 0.318693 | 0.570700 |
| 24.9M compact | 1337 | 2,291 | 0.363696 | 0.576871 |
| 24.9M compact | 1338 | 2,274 | 0.364740 | 0.574168 |
| **Compact mean** | **2 seeds** | **2,282.5** | **0.364218** | **0.575520** |
| **Delta** |  |  | **+0.045525 (+14.3%)** | **+0.004820 (+0.84%)** |

The gain is concentrated in weak spatial and environmental capabilities. Largest lifts are `B44_infer_topo_cos` +0.211, `B46_infer_chm_cos` +0.208, `B1_species_from_env_top10` +0.187, `B43_infer_hydro_cos` +0.166, `B5_species_from_spacetime_top10` +0.108, `B17_infer_soil_cos` +0.105.
Largest tradeoffs are `B18_infer_climate_cos` -0.173, `B21_community_from_species_recall` -0.075, `B22_companions_recall` -0.069, `B33_growth_rate_trait_f1` -0.067, `B28_flowering_peak_month_mrr` -0.066, `B14_vision_leave_one_out_cos` -0.063. Overall, 19 of 50 capabilities improve and 31 regress; the full receipt is in `BENCHMARKS.md`.

The canonical files use batch 512, dense hash optimization, learning rate `1e-3`, an 8,000-step scheduler horizon,
and the fixed 600-second wall-clock stop. The evaluator, aggregate definitions, spatial holdout, and extraction recipe
are unchanged.
