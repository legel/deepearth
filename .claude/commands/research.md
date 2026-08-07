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

## Do not grind

- **Two matched seeds per arm is the standard.** Take a third only when the verdict turns on it — the
  arms overlap, or a regression is the only thing blocking a keep.
- **Never open an experiment whose purpose is to raise confidence in a result you already have.**
  Uncertainty is recorded, not resolved: publish the number with its seed count and spread.
- **Stop if you are measuring variance, re-tuning a weight you already tuned, or re-running a control
  that exists.** Pick a different mechanism instead.
- **Three consecutive experiments without a new mechanism means you are grinding.** Change subsystem.
- A cycle spent re-confirming is a cycle not spent finding the next thing.

## Broken records

A run that fails for a reason that is **not your hypothesis** — a crash, a missing method, a config that
cannot construct, a stale or non-reproducing baseline, an instrument that disagrees with itself — is a
delivery problem, not a research problem.

- **Do not fix it in the loop.** The repair lands untested because the loop is mid-flight.
- **Publish an insight** naming the break with its evidence.
- **Hand it to `ship-deepearth-improvement`**, which owns diagnosing and repairing what ships.
- **Pick another legal hypothesis and keep going.** Loop throughput is the point.

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


## Workspace and git

- **Work in an isolated worktree/branch per experiment.** `git worktree add -b exp/<slug> <path> HEAD`.
  The branch is the experiment: one hypothesis, one branch, one diff to read.
- **Commit the candidate before running it.** A dirty tree makes a run unreproducible and it cannot set
  a record — the diff IS the experiment, and a number measured against uncommitted code is unrecoverable
  by anyone else, including you tomorrow.
- **Discard by abandoning the branch, not by reverting in place.** If the screen misses the floor, drop
  the branch and pick a materially different hypothesis. Do not iterate a losing idea by patching it.
- **One worktree per config, reused.** The inductor cache keys on source, so a fresh worktree per run
  pays ~200s of recompilation; a reused one starts in ~1s.
- **Never delete `/tmp/torchinductor_root`.** That cache is the difference between a 2-minute loop and a
  6-minute one.
- **Keep the prepared cache shared.** Point `--cache_dir` at the shared prepared cache; a worktree that
  builds its own writes ~15.7 GB and will fill the disk. Symlink, never copy.
- **Run over SSH in the foreground.** Backgrounding detaches the process and it dies with the channel.

## Operational

- Ensue needs `ENSUE_API_KEY` or `ENSUE_API_TOKEN` (env, `.autoresearch-key`, or `/workspace/.env`).
  `coord.assert_connected()` raises rather than degrading to silence. Pick a short memorable codename as
  `agent_id`.
- Never compare a mirror run to a public-main run; the evaluators differ.
- Delivery to the public repository is separate and explicitly authorized: `ship-deepearth-improvement`.
