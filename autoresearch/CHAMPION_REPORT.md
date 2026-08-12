# DeepCal champion report

## 25.4M fixed-step habitat-niche fusion record

The compact habitat-niche model improves both public aggregates over a seed-matched 24.9M control at exactly 2,291
optimizer steps. It retains the original PR's central result: the 797.1M default is unnecessary. The candidate is
96.8% smaller and uses 48.4% less VRAM while improving harmonic by 17.3% over the registered reference.

| Model | Seed | Steps | Harmonic | Arithmetic |
|---|---:|---:|---:|---:|
| Registered 797.1M reference | 1337 | 5,126 | 0.318693 | 0.570700 |
| Fixed-step 24.9M control | 1337 | 2,291 | 0.367661 | 0.578883 |
| Fixed-step 24.9M control | 1338 | 2,291 | 0.365992 | 0.581475 |
| **25.4M niche fusion** | **1337** | **2,291** | **0.373074** | **0.581691** |
| **25.4M niche fusion** | **1338** | **2,291** | **0.374775** | **0.584717** |
| **Candidate mean** | **2 seeds** | **2,291** | **0.373924** | **0.583204** |
| **Delta vs fixed-step control** |  |  | **+0.007098 (+1.93%)** | **+0.003025** |
| **Delta vs registered reference** |  |  | **+0.055231 (+17.33%)** | **+0.012504** |

The mechanism isolates habitat occupancy from the shared backbone. Training-split normalized AlphaEarth and
multiscale space-time features feed task-specific family, species, community, and pollinator decoders; detached
features and a separate optimizer prevent niche supervision from commandeering universal fusion. This implements
science rules 18, 23, and 31 while preserving rule 5's capacity criterion.

Across 50 capabilities, 27 improve and 23 regress. The largest lifts are species from environment +0.0486, species
from spacetime +0.0297, form traits +0.0283, species calibration +0.0190, family from spacetime +0.0151, pollinators
from environment +0.0131, and family from environment +0.0106. The largest tradeoffs are growth-rate traits -0.0121,
NAIP-IR -0.0087, soil -0.0073, aerial reconstruction -0.0056, and hydrology -0.0031. Both aggregate gates improve on
both seeds; the complete unrounded receipt is in `BENCHMARKS.md`.

The canonical configuration uses batch 512, dense hash optimization, learning rate `1e-3`, and exactly 2,291
optimizer steps. The evaluator, aggregate definitions, spatial holdout, and extraction recipe are unchanged.
