---
description: Run the autonomous Earth4D space-time research loop on the declared capability.
---

# Earth4D space-time autoresearch

`autoresearch/probes/spacetime/program/program.md` is the binding contract. Read it first and follow it
literally. This command is only a short operational entry point; it never overrides the program.

## Fixed boundary

- Edit only `autoresearch/probes/spacetime/editable_files/**`.
- Treat the harness, scoring, definitions, tests, protocol, records, prepared data, fusion model, and
  steering documents as read-only.
- Let the harness write traces, Ensue, `records.json`, and `scorecard.txt`; never hand-edit them.
- If fixed infrastructure appears wrong, report the blocker and choose another legal hypothesis. Do not
  repair or audit it inside the research loop.

## Capability

$ARGUMENTS

Use one declared capability at a time. Before choosing an experiment, read its current `--insights`, the
full generated scorecard, and its archived dead-end reasons.

## Loop forever

1. Predeclare one coherent, substantive DATA or ARCHITECTURE hypothesis.
2. Create an isolated experiment worktree/branch and edit only `editable_files/**`.
3. Commit the experimental diff before running.
4. Run one deterministic screen through the fixed harness with `--ensue`.
5. Show the human the full scorecard immediately after the completed run.
6. If the screen misses the barrier, abandon the branch and choose a materially different hypothesis.
7. If it clears the barrier, run exactly two matched seeds. Promotion requires the two-seed mean to clear
   the declared margin, each seed to beat the strongest paired fair baseline, and no regression.
8. Push to `deepcal-ensue-autoresearch` only for that confirmed genuine breakthrough. Never push a screen,
   baseline, repair, provisional result, diagnostic, or dead end.
9. Continue until the human interrupts the loop.

## Commands

```bash
# Read the archive before choosing the hypothesis.
python -m deepearth.autoresearch.probes.spacetime.harness \
  --insights --metric <capability>

# One-seed screen.
EARTH4D_DETERMINISTIC=1 \
python -m deepearth.autoresearch.probes.spacetime.harness \
  --metric <capability> --tag <tag> --device cuda:N --ensue

# Only after the screen clears: two matched seeds.
EARTH4D_DETERMINISTIC=1 \
python -m deepearth.autoresearch.probes.spacetime.harness \
  --metric <capability> --tag <tag>_2s --seeds 2 --device cuda:N --ensue
```

After every completed command that runs an experiment, report the hypothesis, score, strongest fair
control, fair gain, barrier verdict, protocol/status, and the full current scorecard.
