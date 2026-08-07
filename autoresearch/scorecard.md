# Scorecard — how science is measured

One objective, reported at every scale. The `/research` command carries the program and the loop.

## The number

`val_bpb` — held-out masked reconstruction scored as a proper likelihood, in **bits per revealed
dimension**. It shares the model's data, split, masking and decoder path, but **not its loss functions**:
training uses centered cosine for continuous targets and cross-entropy divided by `log(num_classes)` for
categorical, neither of which is a log-likelihood. `val_bpb` computes its own — Gaussian density, cosine
retrieval against a frozen bank, raw cross-entropy — so a change can improve one and worsen the other,
most plausibly on the z-scored continuous variables (soil, topo, climate, hydro).

The reveal mask is seeded and the reference statistics are frozen, so it is deterministic and comparable
across runs and model sizes.

**Diffusion-scored variables have no likelihood.** `val_bpb` raises rather than omitting them, so a
diffusion-enabled run cannot produce a valid score until that head exposes a log-density.

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

`objective.judge()` decides keep or discard. Three conditions, all required:

1. **Reconstruction** — the aggregate `val_bpb` improves by more than its measured floor.
2. **No regression** — no owned variable worsens by more than its own floor. An aggregate win paid for
   elsewhere is a trade, and rule 32 forbids trades.
3. **Coverage** — at least one *weak* capability improves, where weak comes from the benchmark scores.

Condition 3 exists because the aggregate is dimension-weighted and therefore badly unbalanced. Measured
directly by replaying the masking loop: `climate` carries **95.3%** of it, then phenology 0.92%, topo
0.91%, chm 0.84%, soil 0.64%, hydro 0.45%, and every remaining capability — `identity`, `clay`, `phylo`,
the vision embeddings — about **0.076%** each. Directional variables are scored by retrieval against a
frozen bank, so each contributes ONE revealed dimension regardless of native width; an earlier analysis
using native widths reported "six embeddings 97.8%, clay 30.1%" and was wrong by ~400×. Without a
coverage rule, improving one variable satisfies the gate on its own and the model narrows while the
number rises.

Weakness cannot be read off `val_bpb`. Bits/dim is a differential entropy whose scale reflects a
variable's target variance, so it is not comparable across variables — the benchmark scores are, and
they are what ranks capabilities.

Report `macro()` alongside the aggregate: the unweighted mean over variables, where every scientific
capability counts equally. Aggregate measures reconstruction efficiency; macro measures coverage.



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

## The benchmark suite (science.md rule 32)

The harmonic and arithmetic means over the **whole** declared suite are the standing report. A champion
carries the whole suite, not a subset, and **no individual metric may regress**. This is the language the
public repository is reviewed in and the number that standardizes performance across runs — it is
reported on every run and required for every champion commit via
`main/harness/champion_report.py` (rule 30).

`val_bpb` sits alongside it, not above it. It exists because the harmonic mean cannot resolve model size
— 24.0M and 796M tie at 0.332 vs 0.319–0.325 — so it is what a *screen-scale* experiment steers by while
the suite remains what a champion is judged on.

Use the decomposition to see *where* a change landed and the suite to confirm nothing regressed.

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
