# Scorecard — how science is measured

Protocol `v6-canonical-family-identity` makes the human-interpretable benchmark suite the promotion
instrument. `val_bpb` remains a reconstruction diagnostic and never decides promotion.

## Promotion gate

A candidate and incumbent must carry the same protocol and the same active capability suite. The
headline is the mean of the two per-seed harmonics and arithmetic means, not the aggregate of averaged
benchmark rows. The incumbent control's two-seed full spreads are the floors. A candidate is promoted
only when:

1. its capability harmonic improves by more than the harmonic floor; and
2. its capability arithmetic does not regress by more than the arithmetic floor.

The harmonic is primary because it refuses to hide a weak capability behind strength elsewhere. The
arithmetic is the breadth guard. There is no per-benchmark hard gate across the tail: dozens of
simultaneous two-seed comparisons would reject real improvements through multiplicity. Every individual
score is still reported before→after.

This preserves Lance's intended harmonic. The AlphaEarth ablation is the control: harmonic fell
`0.3614 -> 0.3422` and arithmetic also failed, while the dimension-weighted reconstruction aggregate
rose inside its floor. The defect was not the harmonic; promotion ignored it.

## Membership

- Ordinary human-interpretable task scores are capabilities and enter both means.
- `*_gain` values are mechanism diagnostics. They show whether a subsystem matters; dependence is not
  capability, so they enter neither mean.
- Raw `*_cos` representation scores remain reported but enter neither mean until they carry an
  empirical null. A cosine such as `0.556` has no human meaning by itself.
- `B55_pollinator_phylo_transfer_recall` remains measured and displayed but is quarantined from both
  means. It predicts from focal identity plus environment, scores against the neighbors' pollinator
  union, and never supplies relatives' pollinators as inputs; that does not test the capability named.
- `B66_community_phylo_conditional_auc` measures the incremental community-ranking signal contributed
  by a relative-reconstructed, seed-masked species identity over the exact same position and neighbor
  context. It is a per-query, tie-aware ROC-AUC: `0.5` means the phylogenetic contribution is no better
  than chance. The former contextual recall remains visible as
  `B66_contextual_masked_community_recall`, a diagnostic excluded from both means.
- A changed active capability set is a protocol break, not a comparable result.

Quarantine requires a structural defect, not an inconvenient score. Repairing B55 requires a new
protocol baseline before it can re-enter the gate.

## Likelihood diagnostics

`val_bpb` is held-out masked-reconstruction likelihood in bits per revealed dimension. Its aggregate,
macro view, per-variable decomposition, retrieval floors and headroom explain where a change landed.
They are reported on every scorecard but do not affect the promotion decision.

Lower `val_bpb` is better. Absolute per-variable values are not comparable because differential entropy
depends on target scale. For retrieval-scored variables, headroom is bits minus the measured retrieval
floor, never raw bits.

## Migration

Earlier protocol scores and v6 scores are incomparable. Promotion is frozen until one fresh two-seed
v6 baseline is published explicitly as the new baseline. Historical runs are not replayed. Thereafter
the coordinator mechanically rejects protocol or capability-suite mismatches.

## What a run reports

```text
HUMAN CAPABILITIES (weakest first)
CAPABILITY HARMONIC                              <- primary gate
CAPABILITY ARITHMETIC                            <- breadth guard
QUARANTINED                                      <- raw score + reason
MECHANISM DIAGNOSTICS                            <- raw ablation/information gains
CONTEXT DIAGNOSTICS                              <- valid raw checks that do not isolate a capability
UNCALIBRATED REPRESENTATION METRICS              <- raw cosine, reported but not gated
val_bpb + macro + per-variable decomposition     <- likelihood diagnostics
```
