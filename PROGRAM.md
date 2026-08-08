# PROGRAM — Operating Doctrine

The executable contract is `.claude/commands/research.md`; `autoresearch/science.md` defines the
scientific goal and `autoresearch/scorecard.md` defines the scoring and promotion rules. If this summary
ever disagrees with either source, stop the loop and repair the inconsistency before running research.

## Objective

Improve DeepEarth's weakest actionable human capability while making the whole capability suite more
balanced. Capability harmonic is the primary score; capability arithmetic is the breadth guard.
`val_bpb`, its decomposition, raw cosine metrics, quarantined benchmarks, and mechanism deltas are
diagnostics only and never affect target ranking or promotion.

## Select the target

1. Read the live Ensue scorecard. It must use the evaluator's current protocol and contain exactly two
   seeds. A stale or partial scorecard is a baseline-migration blocker, not research evidence.
2. Read only rows whose role is `capability` and which belong to the scorecard's declared active suite.
   Rank their published two-seed mean scores from lowest to highest. The lowest is the first target.
   Quarantined, inactive, uncalibrated, and mechanism-diagnostic rows never enter this ranking.
3. A target is actionable when its required data and labels are present and an editable model surface
   can causally affect it. Difficulty, a low score, or prior failed hypotheses do not make it
   unactionable. If an instrument is structurally invalid, record the evidence, hand the repair to the
   software-engineering program, and select the next-lowest active capability.
4. Read the original files in `customer_feedback/`, then the relevant `val_bpb` decomposition and
   mechanism diagnostics. Use them to choose the subsystem and hypothesis; do not replace the selected
   human capability with a proxy target.

## Run the experiment

Form one distinct causal hypothesis, check Ensue for prior evidence, claim it, and build the smallest
clean test. Compare the candidate with the live incumbent commit under the same config, cache, prepared
data, hardware class, step budget, and exactly two matched seeds per arm. Reuse a valid matched control;
rerunning one is grinding. Commit the candidate before measuring it.

The selected benchmark's two-seed mean must rise. If it does not, the hypothesis failed even when an
unrelated benchmark raises the harmonic. A successful screen must also satisfy the official v4 judge:

1. candidate and incumbent have the same protocol and identical active capability suite;
2. mean candidate harmonic minus mean incumbent harmonic is greater than the incumbent's two-seed
   harmonic spread; and
3. mean arithmetic regression is no greater than the incumbent's two-seed arithmetic spread.

The judge computes each mean from the two per-seed aggregate scores. It does not aggregate averaged
benchmark rows, use a third seed, impose dozens of per-benchmark gates, or consult `val_bpb`.

The fixed 1k screen decides whether a hypothesis advances, not whether it becomes champion. Only a
screen winner proceeds through the software-engineering handoff and the fixed 8k full-scale confirmation.
Only the confirmed full-scale result may replace the live champion or be pushed as a record.

## Research surface

Pursue architectural and data-channel swings when they directly test the hypothesis. Data is not
categorically closed; aimless collection and harness work are. The editable scientific surface includes
encoding, routing, fusion, learning, optimization, masking, and the modalities supplied to the model.
The evaluator, judge, score definitions, floors, splits, and baseline are immutable during research.

## Record every result

- After every completed two-seed experiment, show the full human-capability scorecard weakest-first,
  then harmonic, arithmetic, named-target movement, and only then the likelihood and mechanism
  diagnostics.
- Publish every valid win, loss, and dead end to Ensue so the swarm does not repeat it.
- Prefer a new mechanism after a losing result; after three experiments in one subsystem, change the
  subsystem rather than tuning the same idea.
- Keep experiments isolated, committed, reproducible, and fixed-step. Abandon rejected branches; do not
  repair them by stacking fixes.
- Push confirmed full-scale records promptly. Never push candidate branches or describe a protocol
  migration as a scientific breakthrough.
