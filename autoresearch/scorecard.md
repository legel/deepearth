# Scorecard — how science is measured

One objective, reported at every scale. See [`main/program/unified-objective.md`](main/program/unified-objective.md)
for why the previous three-loop scheme could not work.

## The number

`val_bpb` — held-out masked-reconstruction loss in **bits per revealed dimension**. The same objective
the model trains on, evaluated on held-out rows with a seeded reveal mask, so it is deterministic and
comparable across runs and across model sizes.

Lower is better. It is additive over variables, which is what makes one number and granular targets the
same measurement:

| level | what it is | who reads it |
|---|---|---|
| **aggregate** | total bits / total revealed dims | the promotion gate |
| **per-variable** | bits/dim for one variable | the lens a given piece of science steers by |
| **ablation delta** | per-variable bits with a subsystem nulled, minus without | that subsystem's in-situ contribution |

There is no separate probe metric. A space-time result is the change in the bits of the space-time-dependent
variables under an Earth4D ablation; a biological result is the change in species/phylo bits under a graph
ablation. Both are terms in the fusion number, so a probe win *is* a fusion win by construction.

## The gate

A result is promoted when `val_bpb` drops, at fixed steps, by more than **the noise floor measured at that
scale**, on at least two scales.

Fixed thresholds are gone. `MIN_REL_IMPROVEMENT` (1.5%), `MIN_ABS_IMPROVEMENT` (0.002) and
`SEED_SIGMA_MULTIPLE` let the campaign promote inside its own noise — champion steps of +0.0013 to +0.0034
against measured two-seed spreads of:

| scale | budget | two-seed spread |
|---|---|---:|
| 796M | 600s | 0.027 |
| 172.6M | 120s | 0.0033 |
| 21.8M | 120s | 0.0167 |

Measure the floor for your configuration before believing any delta. `objective.noise_floor()` takes matched
seeds and has no default.

## Benchmarks

The 63 benchmarks remain, as **diagnostics**. They are not a gate.

The harmonic mean cannot resolve model size — 24.0M and 796M tie (0.332 vs 0.319–0.325) because it is
dominated by the near-zero benchmarks neither model solves. Use them to see *where* a change landed, after
`val_bpb` has already said *whether* it landed.

## Relation to science.md

`science.md` rule 32 asks that 100% of the suite be scored and optimized; that still holds — every benchmark
is measured and reported. What changed is that the suite no longer decides promotion, because it cannot
distinguish a 4.6x model-size difference.

Rule 30's before→after discipline is unchanged and now applies to `val_bpb` and its decomposition rather
than to a harmonic mean.

Rule 20's fixed budget is superseded by **fixed steps**: wall-clock equal-time made model sizes
incomparable, since a small model takes more steps in the same seconds. Step counts measured flat
(~1,030) across 21.7M–172.6M at 120s, so this is a change of contract, not of results.

## What a run reports

```
val_bpb:          <aggregate, bits/dim>          <- the gate
  per-variable:   <variable> <bits/dim>          <- the lens
net_score / arithmetic mean                      <- diagnostics only
```
