# DeepEarth champion report

## Sparse ecological Earth4D record

The model builds typed, multiresolution Earth4D state, processes each scientific lens with paired graph streams, and
denoises only query-selected mesh segments. Relation-specific meshes preserve identity, pollinator, and mycorrhizal
structure; the ecological reader combines AlphaEarth, WorldClim, habitat, distribution, community-scale, and seasonal
evidence before ranking species.

| Model | Seed | Steps | Harmonic | Arithmetic |
|---|---:|---:|---:|---:|
| Prior family-preserving record | two-seed mean | 8,300 | 0.385644 | 0.575712 |
| Sparse ecological Earth4D | 1337 | 8,000 | 0.388024 | 0.577688 |
| Sparse ecological Earth4D | 1338 | 8,000 | 0.388125 | 0.577697 |
| **Sparse ecological mean** | **two seeds** | **8,000** | **0.388075** | **0.577692** |
| **Record delta** |  |  | **+0.002430** | **+0.001980** |

The same architecture wins on both independent checkpoints. B1 environment-to-species top-10 rises
`0.391769 -> 0.441402`; B23 species-calibration MRR rises
`0.180489 -> 0.202651`. The evaluator, benchmark
definitions, data, and spatial holdout are unchanged.

The production model has 52,664,449 parameters. Both checkpoint conversions cover every production parameter; the
seed-1337 conversion drops only 19 research-only tensors. The complete two-seed receipt is in `BENCHMARKS.md` and
`champion_scores.json`.

Evaluation base: `bbbe6be6bf30a8d169605d67b0b6b9eec4d29b74`. Evaluator SHA-256:
`59d37fbe2d5645c8169475ef61d298637638cc996403ba77da9b5ed20cdba99c`.
