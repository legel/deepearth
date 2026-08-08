# records

`champion_scores.json` is the git-visible history used by `harness/champion_report.py` to produce a
before→after report for a confirmed champion. Do not edit scores by hand.

The live scorecard and all win/loss experiment records live in Ensue. `LOOP-deepearth-best` changes only
through `coordinator.publish_best()` after a confirmed full-scale result or an explicitly labeled
protocol migration/delivery stamp. See [`../../scorecard.md`](../../scorecard.md) for the gate.
