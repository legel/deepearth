# DeepEarth autoresearch — granular probe loops, then fusion

We are building **backwards**. Each research loop owns **one probe and its data**, and recovers signal
for one part of the science. Only when the science in `science.md` is actually filled out do those
recovered signals get plugged into the fusion layer — the full model comes last, not first.

```
   probe loop            probe loop            probe loop
   one probe             one probe             one probe
   own data              own data              own data
   own validation        own validation        own validation
        │                     │                     │
        └──────── recovered signal ─────────────────┘
                          ▼
                    FUSION  (main/)   ← integrates, and only after the science is filled out
```

Why this way round: a fusion model trained before its constituent signals are established cannot tell
you which part works. It is confounded and slow, and every result it produces is a joint claim about
everything at once. A probe loop makes one narrow claim, in minutes, against fair controls — and a claim
that survives its own validation is what earns a place in the fusion layer.

So a probe loop's job is not "raise the aggregate". It is: **recover a real signal on one capability,
with its own evals, and prove it is the encoder's and not borrowed.** `main/` is the destination, not a
peer competing for the same score.

Each loop is a directory, owns its program, and has **the same three subdirectories with the same
meaning**, so an agent never has to guess what it may touch.

```
autoresearch/
  .env                credentials (gitignored) — ENSUE_API_TOKEN; template in .env.example
  scorecard.md        INDEX of every loop's scorecard — start here to read progress
  science.md          binding research rules — all three loops obey them
  bibliography.md     references
  main/               full-model DeepCal, the whole B1..B60 suite
  biological/         biological encoder probe
  spacetime/          Earth4D spacetime encoder probe
        │
        ├── program/           the contract. Read first; change only when doctrine changes.
        ├── editable_files/    ← THE ONLY CODE AN EXPERIMENT MAY EDIT
        │     ├── harness/       the loop itself
        │     └── lib/           auxiliary code the loop calls
        │     └── data/          the DATA lever: sources added, moved and removed by the signal
        │                        they provide. One source per directory.
        └── records/           harness-written: board, traces, ledgers. Never hand-edited.
```

The shared dataset lives at the repo root `data/deepcal`: all three loops read it, so it belongs to no
single loop. Loop-specific data lives with its loop (`spacetime/data/lfmc/`).

## The three roles

| directory | what it is | policy |
|---|---|---|
| `program/` | objective, how to pick a target, what counts as evidence, ops notes | Read before every cycle. Change when the *doctrine* changes — never to accommodate a result. |
| `editable_files/harness/` | the loop: the driver, the modes, the scoring and recording path | Editable. But a change to what a number *means* — recording, comparability, gates — goes in its own commit with a test that fails before and passes after. Never inside an experiment. |
| `editable_files/lib/` | auxiliary code the loop calls: mechanisms, target builders, data loaders | Editable. This is where most experiments live. |
| `data/` | the channels and observations that feed the model | **Editable — this is the DATA lever.** When the diagnosis reads INPUT-LIMITED, changing what feeds the encoder is the correct move. Rebuild the prepared cache after any change; attribute borrowed signal honestly. |
| `data/` | the board, traces, ledgers — everything the harness writes | Never hand-edit; that forges a result. Corrections go through the ledger, with a reason. Each loop's records are disentangled from every other loop's. |

## Scope — what an experiment may edit

```
   IN SCOPE      <loop>/editable_files/**         the loop and its libraries
                 <loop>/data/**            the DATA lever — a first-class experiment
                 encoders/**                      when the experiment IS an encoder change
   OUT OF SCOPE  <loop>/data/**           the board and its ledgers
                 <loop>/program/**                the contract
                 any OTHER loop's directory       not your surface
                 core/**, autoresearch/science.md unless that is the declared experiment
```

Anything outside `editable_files/` and `data/` is not an experiment — it is infrastructure work, and it goes in its
own commit with its own tests, separate from any result.

## Per-loop contents

| loop | program/ | editable_files/harness/ | editable_files/lib/ | records/ |
|---|---|---|---|---|
| `main/` | `autoresearch.md`, `BENCHMARKS.md`, `CHAMPION_REPORT.md`, `audit.md`, `GRADUATION_BLUEPRINT.md` | `train.py`, `run_experiment.py`, `evaluate.py`, `score.py`, `score_encoders.py`, `champion_report.py`, `hooks.py`, `perception_diag.py`, `deepcal.yaml`, `champion.yaml` | `data.py`, `prepare.py`, `recipes/` | `champion_scores.json` |
| `biological/` | `program.md` | `probe.py`, `stage1…stage4`, `ensue_log.py` | `traitprobe.py` | — |
| `spacetime/` | `program.md`, `scorecard.md`, `lfmc_gate.md`, `box-operations.md` | `probe.py`, `probe_modes_tables.py`, `trace.py`, `probe_contract.py`, `probe_emit.py`, `probe_registry.py` | `recurrence.py`, `gnn.py`, `phenology.py`, `dyntargets.py`, `env_field.py`, `calib_probe.py`, `lfmc_recurrent.py`, `science_gate.py` | `records.json`, `traces/` |

## Boundaries between the loops

| loop | objective | instrument | state it owns |
|---|---|---|---|
| `main/` | integrate established signals; B1..B60 means, no metric regressing | full 799M fusion model | `main/records/champion_scores.json`, `champion.yaml` |
| `biological/` | recover signal on the biological capabilities | biological probe pipeline | its own logs |
| `spacetime/` | recover signal on one capability at a time (`spacetime/program/scorecard.md`) | Earth4D + light head, minutes per run | `spacetime/records/records.json`, Ensue `LOOP-earth4d-<capability>` |

- **Only `main/` trains the full fusion model, and it runs LAST.** A probe loop trains a light head on
  encoder features in minutes. A probe record is not a champion result; it is a candidate signal that
  has to clear its own loop's validation before fusion is the right place for it.
- **One probe per loop.** If a loop grows a second probe with its own targets and its own scoring, it is
  two loops wearing one directory — split it. That accretion is what produced 113 flags and 19 modes in
  a single file here.
- **Each loop is independent CODE.** No loop imports another loop. If two loops need the same loader,
  each keeps its own copy — a cross-loop import means a change in one loop silently moves another loop's
  numbers with no record saying so. Sharing is allowed only *downward*, into code no loop owns
  (`encoders/`, and `autoresearch/main/editable_files/fusion/` for the fusion loop alone). `tests/test_loop_independence.py` enforces this,
  plus the identical four directories and the presence of each loop's program.
- **Each loop optimizes its own metric under its own evals.** A loop's program declares what it is
  raising and what would falsify it. No loop is scored on another's number, and no loop's result is
  promoted by another loop's evidence.
- **No loop writes another loop's `records/`.**
- **One program per surface.** Two definitions of the same surface means one is stale — reconcile before
  running anything. (`spacetime/program/lfmc_gate.md` is a gate record, not a second program.)

## Setup

1. Clone `github.com/legel/deepearth` (branch `deepcal`).
2. `pip install -r requirements.txt`, then build the Earth4D CUDA hash encoder against your torch:
   `cd encoders/spacetime && bash install.sh` (the shipped .so is ABI-specific — you MUST rebuild it).
3. Read `science.md` (binding), then the `program/` of the loop you are running.
4. `python -m deepearth.autoresearch.main.editable_files.lib.prepare` — downloads and extracts the audited
   dataset (deepcal_data.zip) from NERSC into `data/deepcal/`.
5. `python -m deepearth.autoresearch.main.editable_files.harness.train autoresearch/main/editable_files/harness/deepcal.yaml --steps 8000 --device cuda:0`
   (batch 512 needs ~27GB; on a 24GB card set `batch: 256` + `pollinator_top_k: 32`). Score against
   `main/program/BENCHMARKS.md`, edit, repeat.

## Experiment budget: 10 minutes (hard cap)

Every full-model run trains for at most **10 minutes** of wall-clock (`time_budget_s: 600`, measured from
step 10 so startup and compilation are excluded), then is scored by `main/editable_files/harness/evaluate.py`
(science.md rule 20). A hard cap, not a target: never raise it, never report benchmarks from a longer
run. Comparing experiments only at the equal budget is what makes a gain reflect real efficiency rather
than more steps. Kill any run that exceeds it and rerun at 600s.
