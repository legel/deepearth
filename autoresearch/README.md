# DeepEarth autoresearch — three loops, one ontology

Three autonomous research loops. Each is a directory, each owns its program, and **each has the same
three subdirectories with the same meaning**, so an agent never has to guess what it may touch.

```
autoresearch/
  .env                credentials (gitignored) — ENSUE_API_TOKEN; template in .env.example
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
        └── data/
              ├── inputs/        ← EDITABLE. The DATA lever: channels, densification, new sources.
              └── records/       harness-written: the board, traces, ledgers. Never hand-edited.
```

## The three roles

| directory | what it is | policy |
|---|---|---|
| `program/` | objective, how to pick a target, what counts as evidence, ops notes | Read before every cycle. Change when the *doctrine* changes — never to accommodate a result. |
| `editable_files/harness/` | the loop: the driver, the modes, the scoring and recording path | Editable. But a change to what a number *means* — recording, comparability, gates — goes in its own commit with a test that fails before and passes after. Never inside an experiment. |
| `editable_files/lib/` | auxiliary code the loop calls: mechanisms, target builders, data loaders | Editable. This is where most experiments live. |
| `data/inputs/` | the channels and observations that feed the model | **Editable — this is the DATA lever.** When the diagnosis reads INPUT-LIMITED, changing what feeds the encoder is the correct move. Rebuild the prepared cache after any change; attribute borrowed signal honestly. |
| `data/records/` | the board, traces, ledgers — everything the harness writes | Never hand-edit; that forges a result. Corrections go through the ledger, with a reason. Each loop's records are disentangled from every other loop's. |

## Scope — what an experiment may edit

```
   IN SCOPE      <loop>/editable_files/**         the loop and its libraries
                 <loop>/data/inputs/**            the DATA lever — a first-class experiment
                 encoders/**                      when the experiment IS an encoder change
   OUT OF SCOPE  <loop>/data/records/**           the board and its ledgers
                 <loop>/program/**                the contract
                 any OTHER loop's directory       not your surface
                 core/**, autoresearch/science.md unless that is the declared experiment
```

Anything outside `editable_files/` and `data/inputs/` is not an experiment — it is infrastructure work, and it goes in its
own commit with its own tests, separate from any result.

## Per-loop contents

| loop | program/ | editable_files/harness/ | editable_files/lib/ | data/records/ |
|---|---|---|---|---|
| `main/` | `autoresearch.md`, `BENCHMARKS.md`, `CHAMPION_REPORT.md`, `audit.md`, `GRADUATION_BLUEPRINT.md` | `train.py`, `run_experiment.py`, `evaluate.py`, `score.py`, `score_encoders.py`, `champion_report.py`, `hooks.py`, `perception_diag.py`, `deepcal.yaml`, `champion.yaml` | `data.py`, `prepare.py`, `recipes/` | `records/champion_scores.json` |
| `biological/` | `program.md` | `probe.py`, `stage1…stage4`, `ensue_log.py` | `traitprobe.py` | — |
| `spacetime/` | `program.md`, `scorecard.md`, `lfmc_gate.md`, `box-operations.md` | `probe.py`, `probe_modes_tables.py`, `trace.py`, `probe_contract.py`, `probe_emit.py`, `probe_registry.py` | `recurrence.py`, `gnn.py`, `phenology.py`, `dyntargets.py`, `env_field.py`, `calib_probe.py`, `lfmc_recurrent.py`, `science_gate.py` | `records/records.json`, `records/traces/`, ledgers |

## Boundaries between the loops

| loop | objective | instrument | state it owns |
|---|---|---|---|
| `main/` | arithmetic / harmonic mean over B1..B60, no metric regressing | full 799M fusion model | `main/data/records/champion_scores.json`, `champion.yaml` |
| `biological/` | the biological capability set | biological probe pipeline | its own logs |
| `spacetime/` | one capability record at a time from `spacetime/program/scorecard.md` | Earth4D + light head, minutes per run | `spacetime/data/records/records.json`, Ensue `LOOP-earth4d-<capability>` |

- **Only `main/` trains the full fusion model.** A probe loop trains a light head on encoder features in
  minutes; the full model is confounded and slow, and a probe record is not a champion result.
- **No loop writes another loop's `data/records/`.**
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
