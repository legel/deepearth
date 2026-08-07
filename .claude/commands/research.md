---
description: Run the autonomous DeepEarth research loop.
---

# DeepEarth autoresearch

## The goal

DeepEarth is a self-supervised multi-modal architecture for **ecological simulation and optimization**,
and specifically a **causal forecaster** — a predictor of plant growth and flowering. It rests on two
innovations: the **Earth4D space-time encoder** and the **phylogenomic species GNN**. It learns by
masked autoencoding, including of embeddings, so any variable can be queried at any point in space-time
and return a posterior that sharpens as evidence is added.

Your job is to find breakthroughs that make that true, in this phase, before scaling. `val_bpb` is the
instrument, not the goal: a number that improves while the model gets no better at ecological
simulation is a tuning result — report it, do not build on it. `autoresearch/science.md` says what the
model must be. What to try next is yours to find; nothing here tells you where to look.

## What you CAN do

Everything in `autoresearch/main/editable_files/**`: architecture, the two encoders, fusion, optimizer,
schedule, losses, masking, hyperparameters, model size, the configs, and which data channels are fed.
Add a modality, delete a mechanism, change the shape of the objective. Simpler is better when results
are equal.

## What you CANNOT do

- Edit `main/harness/**`, `scoring/objective.py`, `tests/**`, or the prepared data. Those are ground
  truth; changing them changes what a number means.
- Tune a metric, split, floor or baseline to make a candidate pass. Improve the model instead.
- Repair fixed infrastructure inside the loop. Publish the blocker as an insight and pick another legal
  hypothesis.
- Spend the loop on confirmation — see **Do not grind**.
- Hand-fix a broken record yourself — see **Broken records**.

## Scope

$ARGUMENTS

With no argument, target the weakest variable in the current `val_bpb` decomposition.

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

The harmonic and arithmetic means over the whole suite are the standing report (rule 32) — the language
the public repository is reviewed in, and what standardizes performance across runs. A champion carries
the whole suite and no individual metric may regress; every champion commit goes through
`champion_report.py` (rule 30).

`val_bpb` sits alongside, not above. It is what a screen-scale experiment steers by, because the harmonic
mean cannot resolve a 4.6x size difference (24.0M and 796M tie).

## The loop

0. **THINK.** Three inputs, in this order.
   - **The decomposition.** Target the weakest variable — the one costing the most bits. That is where
     the headroom is, and the aggregate cannot improve much while it dominates.
   - **`customer_feedback/`.** Read the original files, not a summary of them. They say which
     capabilities the customer actually wants and where the science is heading; a technically valid
     result on a capability nobody asked for is a wasted iteration. Use them with the weakest variable
     to choose the **surface area** — which subsystem to change.
   - **The swarm.** `coord.assert_connected()` first — it raises rather than degrading to silence,
     because every way this client breaks returns "no results", which looks exactly like "nothing to
     learn". Then `state()` for the board, live claims, insights and open
     hypotheses. Your baseline is the swarm's best, not your local one: if someone has beaten it, adopt
     theirs and push from there. Run `already_tried(description)` before claiming: it searches every
     campaign semantically, including the 568 `LOOP-` records from prior work, so you never pay for a
     negative someone already published.

   The target is `science.md` realized **while staying well-rounded**: no capability may be traded away
   to lift another. The weakest variable says where to push; the decomposition says whether you pushed
   it without paying for it elsewhere.
1. **CLAIM.** `coord.claim("description")`. If it returns `None` someone holds it — pick another. Claims
   expire, so a dead agent never blocks the swarm.
2. **State one causal hypothesis.** Which variables should lose bits, and through which subsystem.
3. **Change only the surface that tests it.** Commit before running — a dirty tree makes the run
   unreproducible and it cannot set a record.
3. **Run the pair at fixed steps**, candidate and control, same seed, same cache, same prepared data.
4. **Measure the noise floor at that scale** if you do not already have it: two matched seeds of the
   control, full spread. There is no default and no fixed threshold.
5. **Read the decomposition.** Did the bits drop in the variables the hypothesis named? Does an ablation
   of the subsystem account for it? If the gain landed elsewhere, it is an initialization re-roll —
   adding parameters shifts the RNG stream and re-initializes the whole model.
7. **Record it**, in `val_bpb` and its decomposition, with both benchmark means alongside.

## Standing rules

- **No individual capability may regress** to raise an aggregate. An aggregate win paid for elsewhere is
  a trade, not a result.
- **Every change is a config toggle defaulting to current behaviour**, so the default path stays
  byte-identical and a flag can be flipped off without a rebuild.
- **Commit the candidate before running it.** The diff IS the experiment; a number measured against
  uncommitted code is unrecoverable by anyone else, including you tomorrow.
- **Publish dead ends with their reason.** A negative that is not written down gets re-run by someone.
- **Fixed steps, never fixed wall clock.** Equal-time makes sizes incomparable — a smaller model takes
  more steps in the same seconds.
- **The gate is a measured floor, not a constant.** Two-seed spreads run 0.0033 / 0.0167 / 0.027
  depending on scale; fixed thresholds admitted champion steps of +0.0013, which is inside the noise.
- **Benchmarks diagnose; `val_bpb` gates.** Confirmation against the full model happens once, at the
  merge decision — not per experiment.
- **Stay at the screen scale for the whole loop.** An intermediate scale answers neither question: it
  does not iterate fast and it is not what you ship.

## Rule breaks

Six failure modes, how to recognize each, and the required response. This is the complete list of
rituals — nothing elsewhere in this document adds another.

| break | you are doing it when | do this |
|---|---|---|
| **Grinding** | measuring variance, re-tuning a weight you already tuned, re-running a control that exists, or three experiments deep with no new mechanism | Stop and change subsystem. Publish what you have with its seed count and spread. |
| **Broken record** | a run fails for a reason that is not your hypothesis — crash, missing method, unbuildable config, stale or non-reproducing baseline, an instrument disagreeing with itself | Do not fix it in the loop. Publish an insight with the evidence, hand it to `ship-deepearth-improvement`, pick another hypothesis. |
| **Metric tampering** | changing a metric, split, floor or baseline so a candidate passes | Revert it. Improve the model instead. The instruments are read-only for a reason. |
| **Trading** | an aggregate improves while an owned variable regresses past its floor | Not a result. Report it as a trade and keep looking. |
| **Unreproducible run** | the tree was dirty, the cache differed, or the arms did not share seed and data | The number does not exist. Discard it — do not report it with a caveat. |
| **Incomparable numbers** | comparing a mirror run to a public-main run | Void. The evaluators differ by ~158 lines; the same config scores ~0.279 on the mirror and 0.332464 on public main. |

**Two matched seeds per arm is the standard.** Take a third only when the verdict actually turns on it —
the arms overlap, or a regression is the only thing blocking a keep. Never open an experiment whose
purpose is raising confidence in a result you already have: uncertainty is recorded, not resolved. A
cycle spent re-confirming is a cycle not spent finding the next thing.

**Discard by abandoning the branch, not by reverting in place.** If the screen misses the floor, drop the
branch and pick a materially different hypothesis. Do not iterate a losing idea by patching it.

## Scales

| scale | params | use |
|---|---:|---|
| screen | ~24M | the loop — every hypothesis, ~2 min warm |
| full | ~796M | the merge gate and the product |

## Where things live

```
main/editable_files/     the science — fusion/, encoders/, train.py, lib/, configs
main/harness/            the instruments — evaluate.py, hooks.py, coordinator.py, champion_report.py
scoring/objective.py     val_bpb, the decomposition, the measured noise floor
scorecard.md             how science is measured; science.md, what the model must be
datatools/               one-time data ETL; not part of the loop
```

## Ensue

Shared memory across every agent in the swarm. `coord.assert_connected()` first — it raises rather than
degrading to silence, because every way this client breaks returns "no results", which reads exactly
like "nothing to learn". Needs `ENSUE_API_KEY` or `ENSUE_API_TOKEN` (env, `.autoresearch-key`, or
`/workspace/.env`). Pick a short memorable codename as `agent_id`.

| namespace | holds | written by |
|---|---|---|
| `LOOP-deepearth-best` | **the current scorecard** — the full state of what is best and how it was measured | `coord.publish_best()`, at promotion and at delivery |
| `LOOP-deepearth-<variable>` | per-variable board: best `val_bpb` for that variable | `publish_result()`, when a run sets a record |
| `LOOP-deepearth-runs/<variable>/…` | one record per experiment, win or loss | `publish_result()` |
| `LOOP-deepearth-claims/…` | live claims, expiring | `claim()` |
| `LOOP-deepearth-insights/…` | findings, corrections, blockers | `post_insight()` |
| `LOOP-deepearth-hypotheses/…` | open leads for other agents | `publish_hypothesis()` |

**`-best` is the single source of truth for "where the science stands"** and the record the front end
renders. Its shape is defined by `coordinator.scorecard()`, not by this document — build it there and
publish with `publish_best()`:

```python
card = scorecard(val_bpb=..., macro=..., decomposition=..., revealed_dims=...,
                 benchmarks=..., harmonic=..., arithmetic=...,     # 100% of the suite, rule 32
                 seeds=..., noise_floor=..., params=..., steps=..., config=..., agent=coord.agent_id)
coord.publish_best(card)     # refuses a card that does not beat the standing val_bpb
```

Every field is required and validated: a card missing its benchmarks, its decomposition or its seed
count **raises** rather than publishing a partial record the front end would render as fact. Single-seed
numbers, NaN and non-finite metrics are rejected outright. Commit, branch and **hardware — GPU model,
count, CUDA and torch versions — are detected, not declared**, because a hand-typed GPU name is the
field most likely to be carried forward from the previous card and be quietly wrong. Floats are rounded
and collections sorted, so the same run published twice is byte-identical.

Read it in THINK: your baseline is the swarm's best, not your local one, and anything whose `delivery.pr`
is set is already public and not a breakthrough left to find. `ship-deepearth-improvement` owns the
delivery fields via `stamp_delivery()`.

Run `already_tried(description)` before claiming. It searches every campaign semantically, including the
568 `LOOP-` records from prior work, so you never pay for a negative someone already published.

## Workspace and git

- **Work in an isolated worktree/branch per experiment.** `git worktree add -b exp/<slug> <path> HEAD`.
  The branch is the experiment: one hypothesis, one branch, one diff to read.
- **One worktree per config, reused.** The inductor cache keys on source, so a fresh worktree per run
  pays ~200s of recompilation; a reused one starts in ~1s.
- **Never delete `/tmp/torchinductor_root`.** That cache is the difference between a 2-minute loop and a
  6-minute one.
- **Keep the prepared cache shared.** Point `--cache_dir` at the shared prepared cache; a worktree that
  builds its own writes ~15.7 GB and will fill the disk. Symlink, never copy.
- **Run over SSH in the foreground.** Backgrounding detaches the process and it dies with the channel.
- **Delivery to the public repository is separate and explicitly authorized:**
  `ship-deepearth-improvement`.
