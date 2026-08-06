# The research program

One model, one objective, one loop. Improve DeepEarth, train at fixed steps, measure `val_bpb` on
held-out data, keep what clears the noise floor, repeat.

## Why there is one loop

There used to be three — a space-time probe, a biological probe, and fusion — each with its own
objective, board, protocol and gate. They could not be reconciled, because the probes trained
discriminative classifiers over encoder features while fusion trains masked reconstruction over all
modalities. An encoder that helps a classifier has no reason to help a reconstruction model, and
measurably did not: `f0008d1` was +5.69% on the probe and −0.006 on the benchmark it targeted, with the
fusion port giving +0.0112 and −0.0116 on two seeds — mean zero, sign set by the seed.

The granularity the probes bought was isolation. That is now an ablation delta on the per-variable
decomposition: null a subsystem, read the change in the bits of the variables it serves. Isolated, in
situ, and a term in the fusion number rather than a separate measurement.

## The objective

`val_bpb` — held-out masked reconstruction, scored as a proper likelihood in bits per revealed dimension. It shares the
model's data, split, masking and decoder path, but NOT its loss functions: training uses centered cosine for
continuous targets and cross-entropy divided by log(num_classes) for categorical, while val_bpb uses a
Gaussian density, cosine retrieval against a frozen bank, and raw cross-entropy. That is deliberate -- the
training rescalings keep the shared gradient balanced and are not log-likelihoods -- but it means a change
can improve one and worsen the other, most plausibly on the z-scored continuous variables.

The reveal mask is seeded, and the reference statistics are frozen, so it is deterministic
across runs and model sizes.

It is additive over variables, so one number and granular targets are the same measurement:

| level | what it is |
|---|---|
| aggregate | the gate |
| per-variable | the lens the work steers by |
| ablation delta | that subsystem's in-situ contribution |

Lower is better. It is a differential entropy, so it is not zero-based and absolute per-variable values
say more about target variance than about model quality. **Only differences are meaningful.**

The harmonic and arithmetic benchmark means are still computed and still reported — they are the
language the public repository is reviewed in. They are diagnostics, not the gate: the harmonic mean
cannot resolve a 4.6x model-size difference (24.0M and 796M tie).

## The loop

1. **State one causal hypothesis.** Which variables should lose bits, and through which subsystem.
2. **Change only the surface that tests it.** Commit before running — a dirty tree makes the run
   unreproducible and it cannot set a record.
3. **Run the pair at fixed steps**, candidate and control, same seed, same cache, same prepared data.
4. **Measure the noise floor at that scale** if you do not already have it: two matched seeds of the
   control, full spread. There is no default and no fixed threshold.
5. **Read the decomposition.** Did the bits drop in the variables the hypothesis named? Does an ablation
   of the subsystem account for it? If the gain landed elsewhere, it is an initialization re-roll —
   adding parameters shifts the RNG stream and re-initializes the whole model.
7. **Record it**, in `val_bpb` and its decomposition, with both benchmark means alongside.

## Rules that survive from the old program

- **No individual capability may regress** to raise an aggregate.
- **Never tune a metric to make a candidate pass.** Improve the model.
- **Commit the candidate before running.** The diff is the experiment.
- **Every change is a config toggle defaulting to current behaviour**, so the default path stays
  byte-identical and a flag can be flipped off without a rebuild.
- **Publish dead ends with their reason.** A negative result that is not written down gets re-run.

**Stay at the screen scale for the whole loop.** Confirmation happens once, at the merge decision,
against the full model -- the state a result actually has to hold in. An intermediate scale answers
neither question: it does not iterate fast and it is not what you ship.

## Rules that changed

- **Fixed steps, not fixed wall clock.** Equal-time made sizes incomparable: a smaller model takes more
  steps in the same seconds. Step counts measured flat (~1,030) across 21.7M–172.6M at 120s.
- **Confirmation is a merge gate against the full model, not a per-experiment step.**
- **The gate is a measured floor, not a constant.** `MIN_REL_IMPROVEMENT` (1.5%) and
  `MIN_ABS_IMPROVEMENT` (0.002) admitted champion steps of +0.0013 to +0.0034 against two-seed spreads
  of 0.0033 / 0.0167 / 0.027 depending on scale.
- **Benchmarks diagnose, they do not gate.**

## Scales

| scale | params | use |
|---|---:|---|
| screen | ~24M | the loop — every hypothesis, ~2 min warm |
| full | ~796M | the merge gate and the product |

Never compare a mirror run to a public-main run: the evaluators differ by ~158 lines, and the same
config scores ~0.279 on the mirror against 0.332464 on public main.

## Where things live

```
main/editable_files/     the science — fusion/, encoders/, train.py, lib/, configs
main/harness/            the instruments — evaluate.py, hooks.py (ablations)
scoring/objective.py     val_bpb, the decomposition, the measured noise floor
main/program/            this file, the scorecard, recorded results
datatools/               one-time data ETL; not part of the loop
```
