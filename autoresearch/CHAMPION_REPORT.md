# DeepCal champion report

## 25.4M fixed-step hierarchical-family record

The compact model improves both public aggregates over a seed-matched 24.9M control at exactly 2,291 optimizer
steps. It retains the PR's central result: the 797.1M default is unnecessary. The candidate is 96.8% smaller and
uses 48.4% less training VRAM. Its 2,291-step score is also 18.2% higher than the historical 797M receipt, which
used 5,126 steps; that row is replacement context, not the matched promotion comparison.

| Model | Seed | Steps | Harmonic | Arithmetic |
|---|---:|---:|---:|---:|
| Registered 797.1M reference | 1337 | 5,126 | 0.318693 | 0.570700 |
| Fixed-step 24.9M control | 1337 | 2,291 | 0.367661 | 0.578883 |
| Fixed-step 24.9M control | 1338 | 2,291 | 0.365992 | 0.581475 |
| Prior 25.4M niche fusion | 2-seed mean | 2,291 | 0.373924 | 0.583204 |
| Prior masked-pollinator record | 2-seed mean | 2,291 | 0.376617 | 0.586926 |
| **Hierarchical family MAP** | **1337** | **2,291** | **0.377589** | **0.586005** |
| **Hierarchical family MAP** | **1338** | **2,291** | **0.379225** | **0.588743** |
| **Candidate mean** | **2 seeds** | **2,291** | **0.378407** | **0.587374** |
| **Delta vs prior PR record** |  |  | **+0.001790 (+0.48%)** | **+0.000448** |
| **Delta vs fixed-step control** |  |  | **+0.011581 (+3.16%)** | **+0.007195** |
| **Difference vs historical reference** |  |  | **+0.059714 (+18.74%)** | **+0.016674** |

When plant identity is hidden, the model now composes its species posterior through the empirical
species-to-pollinator table and blends that distribution with the learned pollinator decoder. This makes the
interaction prediction consistent with the model's own uncertain species belief instead of requiring a single
guessed species. It implements science rule 27: plant-pollinator interactions carry biological signal.

The mechanism improves photo-only pollinator recall by 0.0892, photo-plus-environment recall by 0.0872,
environment recall by 0.0054, and spacetime recall by 0.0043. Scores outside the pollinator pathway are identical
because the validation replays the exact same trained checkpoints; no additional optimization steps are involved.

For masked species queries, the new decoder marginalizes the species posterior by botanical family and promotes
the most likely species inside the family with the greatest total probability. It preserves every other species
logit, so fine-grained evidence is retained while family-level decisions become coherent. On the exact same
checkpoints it raises B6 family-from-environment from 0.159364 to 0.172357 and B8 family-from-spacetime from
0.161352 to 0.170268, while increasing both public aggregates on both seeds. This implements science rules 17 and
23: posterior evidence is composed probabilistically without collapsing the species distribution.

The canonical configuration uses batch 512, dense hash optimization, learning rate `1e-3`, and exactly 2,291
optimizer steps. The evaluator, aggregate definitions, spatial holdout, and extraction recipe are unchanged.
