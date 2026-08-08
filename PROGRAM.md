# PROGRAM — Operating Doctrine

Authority is divided by subject:

- `autoresearch/science.md` defines the scientific goal.
- `autoresearch/scorecard.md` defines suite membership, scoring, and promotion.
- `.claude/commands/research.md` defines the executable research procedure.
- This file is the concise operating contract. It summarizes those definitions without overriding
  them.

The implementation named by each document is the mechanical authority: capability membership and
aggregation come from `autoresearch/scoring/objective.py`, evaluation from
`autoresearch/main/harness/evaluate.py`, and publication from
`autoresearch/main/harness/coordinator.py`. If prose and implementation disagree, stop the loop and
repair the prose or hand the implementation defect to the software-engineering program before running
research. Never choose whichever interpretation would make a candidate pass.

## Objective

Improve DeepEarth's weakest active human capability while making the whole capability suite more
balanced. The mean of the two per-seed capability harmonics is the primary score; the mean of the two
per-seed capability arithmetic means is the breadth guard.
`val_bpb`, its decomposition, raw cosine metrics, quarantined benchmarks, and mechanism deltas are
diagnostics only and never affect target ranking or promotion.

## Select the target

1. Read `LOOP-deepearth-best`, the live Ensue scorecard. It must use the evaluator's current protocol,
   declare its active capability suite, and contain exactly two benchmark seeds. A stale or partial
   scorecard is a baseline-migration blocker, not research evidence.
2. Read only benchmark rows whose role is `capability` and whose names belong to that declared suite.
   Rank their published two-seed mean scores from lowest to highest, breaking an exact tie by benchmark
   name. The lowest row is the target. Quarantined, inactive, uncalibrated, and
   mechanism-diagnostic rows never enter the ranking.
3. Never skip an active capability. Difficulty, a low score, or failed hypotheses are reasons to change
   mechanism, not target. If the target's evaluator or required data is structurally invalid, publish
   the evidence and stop research for a software-engineering repair. It must either be repaired in
   place without changing meaning or quarantined through a new protocol and two-seed baseline before
   the loop may select the next row.
4. Read the original files in `customer_feedback/`, then the relevant `val_bpb` decomposition and
   mechanism diagnostics. Use them to choose the subsystem and hypothesis; do not replace the selected
   human capability with a proxy target.

## Run the experiment

Form one distinct causal hypothesis, check Ensue for prior evidence, claim it, and build the smallest
clean test. Screening and promotion are separate comparisons:

- **1k screen:** compare the candidate with the approved 1k control for the current research base.
- **8k confirmation:** compare the surviving candidate with the current live champion at 8k.

Never compare a 1k result with an 8k result. Within either comparison, candidate and control must use
the same research base, protocol, active capability suite, cache, prepared data, hardware class, step
budget, and the fixed seed pair `1337` and `1338`. The committed hypothesis must be the only
experimental difference: its editable code and declared config values may differ; every other
condition must not. Reuse a valid matched control at that base and scale; rerunning one is grinding.
Commit the candidate before measuring it.

Two separate decisions are required:

1. **Research advancement:** the selected benchmark's two-seed mean must rise. This is the causal check
   on the stated hypothesis. If it does not rise, the hypothesis failed even when unrelated benchmarks
   raise the harmonic.
2. **Score promotion:** `judge()` must pass. This is the only scoring gate. It deliberately does not
   contain the selected-target check, because target selection is research steering rather than a score
   definition.

Both decisions must pass for a 1k candidate to advance to full confirmation. For a 1k screen,
`before` is the approved 1k control; for an 8k confirmation, `before` is the live 8k champion. The v4
judge requires:

1. candidate and incumbent have the same protocol and identical active capability suite;
2. mean candidate harmonic minus mean incumbent harmonic is greater than the incumbent's two-seed
   harmonic spread; and
3. mean arithmetic regression is no greater than the incumbent's two-seed arithmetic spread.

The judge averages the two per-seed capability harmonics and the two per-seed capability arithmetic
means. It does not compute either headline from already-averaged benchmark rows, consult the chosen
target, use a third seed, impose dozens of per-benchmark gates, or consult `val_bpb`.
Every other capability movement remains visible in the scorecard and can motivate the next experiment,
but it is not an undeclared promotion veto. This keeps rule 18's regressions observable without
turning ordinary two-seed tail noise into a second judge.

The fixed 1k screen decides whether a hypothesis advances, not whether it becomes champion. It never
updates `LOOP-deepearth-best` and is never called a record break. Only a scientifically eligible screen
winner proceeds through the software-engineering handoff and the fixed 8k full-scale confirmation.
At 8k, re-check both decisions against the live champion: the target must still rise and the v4 judge
must pass. The loop enforces the target check before calling `publish_best()`; the coordinator
mechanically enforces the judge. Only then may the result replace the live scorecard or be called a
record break. The only non-breakthrough writes to that key are explicit protocol/schema baseline
migrations and delivery stamps; neither may be described as a scientific improvement.

## Research surface

Pursue architectural and data-channel swings when they directly test the hypothesis. Data is not
categorically closed; aimless collection and harness work are. The editable scientific surface includes
encoding, routing, fusion, learning, optimization, masking, and the modalities supplied to the model.
The evaluator, judge, score definitions, floors, splits, and baseline are immutable during research.

## Record every result

- After every completed two-seed experiment, show every active human capability as
  `control -> candidate (delta)`, ordered by the control's two-seed mean from weakest to strongest.
  Then show harmonic, arithmetic, the named-target decision, the official judge decision, and only
  then the likelihood and mechanism diagnostics. Also show quarantined, inactive, and uncalibrated
  rows in their labeled sections; never silently omit them.
- Publish every valid win, loss, and dead end as an experiment record in Ensue so the swarm does not
  repeat it. Experiment records do not update the live scorecard; only `publish_best()` does that after
  an eligible 8k confirmation or an explicitly labeled migration/delivery operation.
- Prefer a new mechanism after a losing result; after three experiments in one subsystem, change the
  subsystem rather than tuning the same idea.
- Keep experiments isolated, committed, reproducible, and fixed-step. Abandon rejected branches; do not
  repair them by stacking fixes.
- Push a confirmed full-scale record promptly to the private research mirror as one meaningful
  champion commit whose message includes the record diff. Never push candidate branches. Delivery to
  Lance's public repository is a separate, curated software-engineering PR, and a protocol migration is
  never a scientific breakthrough.
