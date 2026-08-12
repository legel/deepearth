# DeepCal champion report

## 25.4M fixed-step masked-pollinator record

The compact model improves both public aggregates over a seed-matched 24.9M control at exactly 2,291 optimizer
steps. It retains the PR's central result: the 797.1M default is unnecessary. The candidate is 96.8% smaller and
uses 48.4% less training VRAM while improving harmonic by 18.2% over the registered reference.

| Model | Seed | Steps | Harmonic | Arithmetic |
|---|---:|---:|---:|---:|
| Registered 797.1M reference | 1337 | 5,126 | 0.318693 | 0.570700 |
| Fixed-step 24.9M control | 1337 | 2,291 | 0.367661 | 0.578883 |
| Fixed-step 24.9M control | 1338 | 2,291 | 0.365992 | 0.581475 |
| Prior 25.4M niche fusion | 2-seed mean | 2,291 | 0.373924 | 0.583204 |
| **Masked pollinator composition** | **1337** | **2,291** | **0.375845** | **0.585553** |
| **Masked pollinator composition** | **1338** | **2,291** | **0.377390** | **0.588299** |
| **Candidate mean** | **2 seeds** | **2,291** | **0.376617** | **0.586926** |
| **Delta vs prior PR record** |  |  | **+0.002693 (+0.72%)** | **+0.003722** |
| **Delta vs fixed-step control** |  |  | **+0.009791 (+2.67%)** | **+0.006747** |
| **Delta vs registered reference** |  |  | **+0.057924 (+18.18%)** | **+0.016226** |

When plant identity is hidden, the model now composes its species posterior through the empirical
species-to-pollinator table and blends that distribution with the learned pollinator decoder. This makes the
interaction prediction consistent with the model's own uncertain species belief instead of requiring a single
guessed species. It implements science rule 27: plant-pollinator interactions carry biological signal.

The mechanism improves photo-only pollinator recall by 0.0892, photo-plus-environment recall by 0.0872,
environment recall by 0.0054, and spacetime recall by 0.0043. Scores outside the pollinator pathway are identical
because the validation replays the exact same trained checkpoints; no additional optimization steps are involved.

The canonical configuration uses batch 512, dense hash optimization, learning rate `1e-3`, and exactly 2,291
optimizer steps. The evaluator, aggregate definitions, spatial holdout, and extraction recipe are unchanged.
