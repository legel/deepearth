# One objective, measured at every scale

## The problem

The probe and fusion optimize different objectives. The probe trains a
discriminative classifier (species/family accuracy from encoder features);
fusion trains masked reconstruction across all modalities. An encoder that helps
a classifier has no principled reason to help a reconstruction model.

Measured 2026-08-06: `f0008d1`'s orthogonal temporal transport is +5.69% on the
probe (reproduced at 0.226096, fair gain +0.073, EARNING) and **-0.006 on B8**
in fusion. Two matched seeds of the fusion port gave +0.0112 and -0.0116 --
mean zero, sign flipped by the seed.

Four further defects make cross-loop comparison impossible:

- `fair_gain` / `share` (EARNING vs LIMITED) is confounded with encoder output
  width; zero-padding moved share 20.7% -> 27.2% -> 15.1%.
- Five protocol versions (v1-prefix .. v5-encoder-only) each void the previous
  board, so no record is comparable to any other.
- The harmonic net score cannot resolve model size: 24.0M and 796M tie
  (0.332 vs 0.319-0.325) because it is dominated by the near-zero benchmarks
  neither model solves. Arithmetic shows the real gap (0.5229 vs 0.5707).
- Noise exceeds effect size everywhere: two-seed spread is 0.027 at 796M and
  0.0167 at 21.8M, against champion-ladder steps of +0.0013 to +0.0034.

## The objective

`val_bpb`: held-out masked-reconstruction loss in bits per revealed dimension,
same objective the model trains on, seeded reveal mask so it is deterministic
and comparable across runs and model sizes. Added in `18a807b`.

It is **additive over variables**, so alignment and granularity are the same
quantity at two aggregation levels:

- the **aggregate** is what every loop is judged on;
- the **per-variable decomposition** is the granular target each loop steers by.

A space-time experiment reads the bits of the variables that depend on
space-time; a biological experiment reads the phylo and species terms. Neither
can win by trading another variable's budget, which separate scorecards allow
today. Attribution falls out for free: when an encoder improves, the
decomposition shows which variables' bits dropped.

Bits are comparable across scale. 0.01 bits on `identity` means the same thing
in the probe, the 24M screen and the 796M model -- accuracy does not.

## The gate

Promote when `val_bpb` drops at fixed steps across at least two scales, by more
than the noise floor measured at each scale. Falsifiable, continuous, and immune
to the harmonic-mean insensitivity.

## What this removes

`fair_gain`, `share`, the EARNING/LIMITED read, the five protocol versions,
`MIN_REL_IMPROVEMENT` / `MIN_ABS_IMPROVEMENT` / `SEED_SIGMA_MULTIPLE` as fixed
constants (replaced by a measured per-scale floor), the harmonic net score as a
promotion gate, the graduation ledger (zero rows ever written), and the separate
probe boards.

The 63 benchmarks stay as **diagnostics**, not as the gate.

## Measured constants

| quantity | value | source |
|---|---|---|
| two-seed spread, 796M, 600s | 0.027 harmonic | base runs 0.325408 / 0.318693 |
| two-seed spread, 172.6M, 120s | 0.0033 harmonic | screen 0.322195 / 0.325487 |
| two-seed spread, 21.8M, 120s | 0.0167 harmonic | h10 0.328936 / 0.312220 |
| steps at 120s | ~1,030, flat across 21.7M-172.6M | sweep step counts |
| warm-cache startup | ~1s | inductor cache retained |
