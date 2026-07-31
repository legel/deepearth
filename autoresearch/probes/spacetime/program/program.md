# Earth4D autoresearch

This loop autonomously improves Earth4D on one declared space-time capability at a time. Its job is to
discover real encoder signal through meaningful DATA or ARCHITECTURE experiments, not to maintain the
research infrastructure or chase an aggregate score.

`program.md` is human-owned steering. Agents read it and never edit it; only the human/operator changes
the research doctrine in a separate maintenance task.

## Authority and scope

### The only editable source

Edit only `autoresearch/probes/spacetime/editable_files/**`.

Everything inside that surface is fair game when required by the hypothesis: representation, objective,
data loader, channel, optimizer behavior, or causal mechanism. One experiment is one coherent hypothesis
implemented as one direct diff; do not create a copied probe, config variant, or flag for an unproven
idea.

### The fixed judge

The following are categorically read-only in this loop, for every reason:

- `autoresearch/probes/spacetime/harness.py`
- `autoresearch/scoring/**`, including `definitions.py`
- tests outside `editable_files/**`
- protocol, graduation, scorecard, and record-management code
- prepared data and the fusion model

Do not repair, refactor, extend, rebaseline, retire records, add tests, or reinterpret metrics. If the
fixed judge appears wrong, report one concise blocker and move to another legal hypothesis or capability.
Only the human/operator may start a separate maintenance task.

Generated research state is the sole exception: the existing harness writes traces, `records.json`,
`scorecard.txt`, and Ensue. Never hand-edit any of them.

## Setup — once per campaign

1. Work on a dedicated experiment branch/worktree made from the current promoted base.
2. Read this file, then run `harness.py --insights --metric <capability>`.
3. Read the capability using the read-only routing query:
   `python -m deepearth.autoresearch.scoring.definitions --capability <capability>`.
4. Verify the required data and one GPU are available. Box details live in `program/box-operations.md`.
5. Use the standing current-protocol record as the baseline. When the operator starts a new protocol or
   a capability has no baseline, first run the unchanged editable tree once and then freeze that baseline.
   Never begin a rebaseline campaign from inside the research loop.
6. Begin experimentation and continue until interrupted.

## Fixed experiment

The judge measures:

```text
encoder   Earth4D as the production fusion model instantiates it
          spatial_levels=18, temporal_levels=18, log2_hashmap=20
          36 spatial + 108 tri-plane = 144 dimensions
control   train-extent, bandwidth-selected RFF = 144 dimensions
training  train_encoder=True, deterministic, 800 steps per arm
gain      Earth4D - strongest matched RFF control
```

Run one seed with:

```bash
EARTH4D_DETERMINISTIC=1 \
python -m deepearth.autoresearch.probes.spacetime.harness \
  --metric <capability> --tag <tag> --device cuda:N --ensue
```

The command, metric, split, controls, protocol, and budget are fixed. Do not add harness flags or modify
the judge to accommodate an idea. Make the idea fit through the editable source or choose another idea.
Determinism is a fixed protocol assumption, not a research hypothesis; do not re-audit it in this loop.

## Choose a real hypothesis

Every experiment must state before editing:

- the capability;
- one causal hypothesis;
- DATA or ARCHITECTURE;
- one coherent intervention;
- the expected score and fair-gain direction;
- the reason the archive does not already settle it.

Use the native diagnosis:

```text
Earth4D - RFF <= 0                 Earth4D - RFF > 0
input lacks usable encoder signal  mechanism carries distinct signal
make a DATA swing                  make an ARCHITECTURE swing
```

Use signal capture as a second read: exhausted coordinates call for DATA; high positional headroom with
low capture can justify ARCHITECTURE. Do not substitute parameter sweeps for a hypothesis.

### DATA swings

- One attributable source per run.
- Fit normalizers, imputers, ranges, and aggregates on train rows only.
- Give Earth4D and every fair control identical access to side information.
- Respect the capability definition. If adding a source changes the declared measurement (for example,
  fused context versus bare space-time), use a capability that owns that source. Never promote the fused
  number into the bare row.
- Label frozen pretrained signal as borrowed; an AlphaEarth/DINO/BioCLIP lift is not an encoder lift.

### ARCHITECTURE swings

- Change the representation, interaction, objective, memory, or causal mechanism—not just width, level
  count, head size, or another capacity sweep.
- Previously rejected architecture is fresh only when the archived failure reason no longer applies.
- A bolt-on basis is an encoder-plus-basis result and must be described that way.

After two flat experiments in one lever family, switch families or try a materially more radical idea.
Do not spend the loop on audits, cleanup, micro-tuning, or infrastructure work.

## The experiment loop

LOOP FOREVER:

1. Inspect the clean branch and current promoted base.
2. Read current insights for the declared capability.
3. Pick and predeclare one substantive hypothesis.
4. Create an isolated experiment worktree/branch.
5. Edit only `editable_files/**`.
6. Commit the experimental diff before running; dirty runs cannot qualify.
7. Run one deterministic screen with `--ensue`, redirecting the full log to scratch.
8. Read only the native result, fair control, barrier decision, and relevant diagnostics.
9. Show the human the full current scorecard after every completed run, including protocol/status and
   marking all legacy pre-v5 rows VOID.
10. If the result clears the barrier with no regression and supports the claim against the strongest fair
    control, keep the candidate branch. Otherwise abandon it; the Ensue dead-end is the durable result.
11. Immediately choose the next hypothesis.

Do not consolidate failed experiment code into the campaign branch. Do not push screens, baselines,
diagnostics, repairs, provisional records, or dead ends.

## Crashes

If a run crashes, inspect the tail of its log. Fix only an obvious mistake inside the editable diff, with
at most two retries. If the hypothesis itself is broken, record the crash, abandon the branch, and move
on. A failure in fixed infrastructure is a blocker, not permission to edit it.

## Keep, confirm, promote

### Screen

- One seed.
- Must beat the standing comparable record by `max(2%, 0.002)`.
- Must have no registered regression.
- A higher absolute score with a losing fair gain is a DATA/control finding, not an encoder breakthrough.

### Confirm

Only a barrier-clearing screen earns five matched seeds. A confirmable claim requires:

- the point estimate to clear the declared margin;
- the lower 95% bound over the strongest paired fair baseline to be greater than zero;
- identical data, split, seeds, budget, and head/control treatment;
- clean committed code and no regression;
- no target proxy, future leakage, or borrowed-signal laundering.

### Promote

Push to `deepcal-ensue-autoresearch` only when the five-seed gate passes and the result is a genuine
breakthrough for the declared capability. The user has authorized pushes only under that condition.

## Evidence rules

- A target must represent measured state, not observer/sampling behavior.
- Fit every data-derived transform on train only.
- A causal claim must consume observed past state and roll predictions forward; a delayed positional basis
  is not memory.
- Paired controls receive the same data, seeds, budget, and capacity treatment.
- The probe ranks hypotheses. It is not a headline claim without five seeds and independent replication.
- Any neighbor/target window that can cross the forecast origin is quarantined until future-sentinel,
  horizon-purge, and right-censoring checks pass.

## Shared memory and reporting

- `--insights` is mandatory before choosing each hypothesis.
- `--ensue` is mandatory on every completed run, win or dead end.
- Never publish a max-of-reruns; one seed screens, five matched seeds confirm.
- Never manually change the board or ledger.
- After every run, show the full scorecard—not the compact protocol-free terminal summary.

## Never stop

Once setup is complete, do not pause to ask whether to continue. If ideas run thin, re-read archived
dead-end reasons, choose a different capability or lever family, and make a larger experiment inside the
editable surface. The loop continues until the human interrupts it.
