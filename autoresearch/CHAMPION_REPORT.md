# DeepCal champion report

## v2 pollinator-transfer protocol migration

This migration changes score membership and interaction supervision, not the incumbent model. Reaggregating the
stored two-seed rows moves harmonic `0.378407 -> 0.436640` and arithmetic `0.587374 -> 0.598583`; scores across that
boundary are incomparable and the change is not a scientific improvement.

| Stored receipt | Seed | Harmonic | Arithmetic |
|---|---:|---:|---:|
| Hierarchical family MAP, v2 | 1337 | 0.435648 | 0.597187 |
| Hierarchical family MAP, v2 | 1338 | 0.437633 | 0.599979 |
| **Incumbent v2 mean** | **2 seeds** | **0.436640** | **0.598583** |
| Rejected mesh candidate, v2 mean | 2 seeds | 0.433762 | 0.570295 |

The mesh candidate remains below the incumbent on both means, so the membership correction does not reverse its
rejection.

### Membership correction

- Derived `*_gain` values measure dependence on a mechanism. They remain fully reported but enter neither mean.
- Legacy B55 remains fully reported but is quarantined. It predicts from focal identity plus environment and scores
  against spatial neighbors' pollinator union; it does not test transfer from phylogenetic relatives.
- Raw cosine capabilities are unchanged by this PR.

### Valid transfer benchmark

B64 withholds a deterministic species-level subset of plant-pollinator interactions before training. Held species
cannot contribute pollinator loss or direct species-to-pollinator lookup rows. Evaluation retains their legitimate
species identity and phylogenetic placement, predicts the focal plant's own interaction distribution, and reports
NDCG@10 normalized against the exact uniform-ranking null. B65 reports the paired species-graph ablation delta as a
mechanism diagnostic.

The holdout mask is checkpointed. Legacy checkpoints cannot report B64, and the first fresh two-seed B64-active run
must be registered explicitly as a new baseline rather than promoted against this migrated scorecard. This implements
science rules 27, 30, and 32 without fabricating a score for the incumbent.
