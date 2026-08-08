# PROGRAM — Operating Doctrine

The executable contract is `.claude/commands/research.md`; `autoresearch/science.md` defines the
scientific goal and `autoresearch/scorecard.md` defines promotion. This file summarizes them and must
not redefine them.

## Objective

Improve DeepEarth's weakest actionable human capability while making the system more balanced.
Capability harmonic is the primary promotion score; arithmetic is the breadth guard. `val_bpb`, its
decomposition, raw cosine metrics, quarantined benchmarks, and mechanism deltas are diagnostics only.

## Selection

1. Read the live Ensue scorecard and require the evaluator's current protocol. A stale protocol is a
   migration blocker, not a research baseline.
2. Consider only active human-capability benchmarks. Exclude quarantined, uncalibrated, and mechanism
   diagnostics from target selection.
3. Start with the lowest score. If it is not actionable, record the concrete reason and choose the next
   weakest; never skip a weak capability merely because it is difficult.
4. Use original customer feedback and the likelihood decomposition to choose which subsystem can
   causally improve that capability.

## Discovery loop

Form a distinct scientific hypothesis, check Ensue for prior evidence, build the smallest clean test,
and run candidate versus matched control. Use exactly two matched seeds per arm. Promote only when the
mean harmonic gain beats the incumbent spread and arithmetic holds within its spread. Publish wins and
dead ends so the swarm does not repeat them.

Pursue architectural and data-channel swings when they directly test the hypothesis. Data is not
categorically closed; aimless collection and harness work are. The editable scientific surface includes
encoding, routing, fusion, learning, optimization, masking, and the modalities supplied to the model.

## Discipline

- Prefer new mechanisms over repeated tuning of a losing idea.
- Keep experiments isolated, committed, reproducible, and fixed-step.
- Report every human benchmark before diagnostics; never promote a proxy-only win.
- Do not edit the evaluator, judge, floors, splits, or baseline inside the research loop.
- Push confirmed records promptly; never describe a protocol migration as a model breakthrough.
