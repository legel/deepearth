# DeepEarth champion report

## 22.7M wide-cell Earth4D mesh harmonic record

The production model replaces fusion-only modality mixing with a fibered Earth4D world state. Environmental, visual,
biological, and ecological evidence is written into typed multiresolution cells; task-conditioned attention reads the
shared mesh without a raw-modality bypass.

The 192-dimensional mesh improves both public aggregates over the otherwise identical 128-dimensional mesh on both
seeds at exactly 8,300 optimizer steps. It also sets a new public harmonic score relative to the registered 25.4M
champion. Arithmetic remains below that champion, so this is explicitly a harmonic record rather than an
across-the-board aggregate record.

| Model | Seed | Steps | Harmonic | Arithmetic |
|---|---:|---:|---:|---:|
| Registered 25.4M public champion | 2-seed mean | 2,291 | 0.378407 | **0.587374** |
| 128-wide Earth4D mesh | 1337 | 8,300 | 0.377276 | 0.557451 |
| 128-wide Earth4D mesh | 1338 | 8,300 | 0.381407 | 0.566217 |
| **192-wide Earth4D mesh** | **1337** | **8,300** | **0.382631** | **0.572135** |
| **192-wide Earth4D mesh** | **1338** | **8,300** | **0.388055** | **0.578847** |
| **192-wide mesh mean** | **2 seeds** | **8,300** | **0.385343** | **0.575491** |
| **Delta vs 128-wide mesh** |  |  | **+0.006002** | **+0.013657** |
| **Delta vs registered champion** |  |  | **+0.006936** | **-0.011884** |

The matched width experiment attributes the gain to representation capacity: mesh cell width increases from 128 to
192 while Earth4D levels, hash capacity, latent count, reader depth, data, evaluator, and optimizer-step budget remain
fixed. The resulting production model has 22,744,486 parameters, 2.7M fewer than the registered public champion.

The strongest human-capability gains over the registered champion are pollinator distribution quality (+0.187425),
pollinator calibration MRR (+0.113272), community-from-species recall (+0.080498), mycorrhiza-from-environment
(+0.079925), community-from-environment recall (+0.079493), and companion recall (+0.078959). The largest regressions
are hydro reconstruction (-0.219633), topography reconstruction (-0.166664), climate reconstruction (-0.131405), CHM
reconstruction (-0.095803), and form-trait F1 (-0.082358). Every active score and both seed receipts are recorded in
`BENCHMARKS.md` and `champion_scores.json`.

The unchanged evaluator includes logistic-renormalized mechanism deltas in harmonic and excludes them from
arithmetic. In particular, community phylogenetic gain rises by +0.234097. This helps explain why harmonic improves
while the arithmetic mean of direct capabilities falls; the PR does not hide or reinterpret that tradeoff.

The canonical design is
`Design(width=192, levels=12, hash_log2=14, latents=16, layers=2)`. Training uses 8,000 mesh steps followed by a
300-step frozen-graph reader fit. Both checkpoints strict-load into the production model. Peak training VRAM was
41,970 MiB.

Evaluation base: `bbbe6be6bf30a8d169605d67b0b6b9eec4d29b74`. Protocol:
`public-main-bbbe6be6-fixed-8300-steps`. Public evaluator SHA-256:
`59d37fbe2d5645c8169475ef61d298637638cc996403ba77da9b5ed20cdba99c`.
