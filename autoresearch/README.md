# DeepEarth autoresearch — three loops, one ontology

Three autonomous research loops. Each is a directory, each owns its program, and **each has the same
four subdirectories with the same meaning**, so an agent never has to guess what it may touch.

```
autoresearch/
  science.md          binding research rules — all three loops obey them
  bibliography.md     references
  main/               full-model DeepCal, the whole B1..B60 suite
  biological/         biological encoder probe
  spacetime/          Earth4D spacetime encoder probe
        │
        ├── program/      READ FIRST, edit deliberately   the contract: objective, board, evidence bar
        ├── instrument/   ← THE ONLY CODE YOU EDIT        the thing being researched
        ├── harness/      DO NOT EDIT to win a run        the judge: measures and records
        └── state/        NEVER hand-edit                 generated: records, scores, traces, ledgers
```

## The four roles

| directory | what it is | policy |
|---|---|---|
| `program/` | the loop's own definition: objective, how to pick a target, what counts as evidence, ops notes | Read before every cycle. Change when the *doctrine* changes — never to accommodate a result. |
| `instrument/` | the code under study: models, probes, propagators, data channels, configs | **This is the only place an experiment edits.** Edit in place, on a branch. Do not add a file per idea. |
| `harness/` | measurement, scoring, recording, gates | Decides what a number *means*. Editing it to make your run look better invalidates everything. Fix it only as its own change, with a test. |
| `state/` | outputs the harness writes | Hand-editing it forges a result. Corrections go through the ledger, with a reason. |

## Scope — what an experiment may edit

```
   IN SCOPE      <loop>/instrument/**            the instrument you are researching
                 encoders/**                     when the experiment IS an encoder change
   OUT OF SCOPE  <loop>/harness/**               the judge
                 <loop>/state/**                 the record
                 <loop>/program/**               the contract
                 any OTHER loop's directory      not your surface
                 core/**, autoresearch/science.md unless that is the declared experiment
```

An experiment that needs a change outside `instrument/` is not an experiment — it is infrastructure
work, and it goes in its own commit with its own tests, separate from any result.

## Per-loop contents

| loop | program/ | instrument/ | harness/ | state/ |
|---|---|---|---|---|
| `main/` | `autoresearch.md`, `BENCHMARKS.md`, `CHAMPION_REPORT.md`, `audit.md`, `GRADUATION_BLUEPRINT.md` | `train.py`, `data.py`, `prepare.py`, `deepcal.yaml`, `champion.yaml`, `recipes/` | `evaluate.py`, `score.py`, `score_encoders.py`, `champion_report.py`, `hooks.py`, `run_experiment.py`, `perception_diag.py` | `champion_scores.json` |
| `biological/` | `program.md` | `probe.py`, `traitprobe.py`, `stage1…stage4` | `ensue_log.py` | — |
| `spacetime/` | `program.md`, `scorecard.md`, `lfmc_gate.md`, `box-operations.md` | `probe.py`, `probe_modes_*.py`, `recurrence.py`, `gnn.py`, `phenology.py`, `dyntargets.py`, `env_field.py`, `calib_probe.py`, `lfmc_recurrent.py` | `trace.py`, `probe_contract.py`, `probe_emit.py`, `probe_registry.py`, `science_gate.py` | `records.json`, `traces/`, ledgers |

## Boundaries between the loops

| loop | objective | instrument | state it owns |
|---|---|---|---|
| `main/` | arithmetic / harmonic mean over B1..B60, no metric regressing | full 799M fusion model | `main/state/champion_scores.json`, `champion.yaml` |
| `biological/` | the biological capability set | biological probe pipeline | its own logs |
| `spacetime/` | one capability record at a time from `spacetime/program/scorecard.md` | Earth4D + light head, minutes per run | `spacetime/state/records.json`, Ensue `LOOP-earth4d-<capability>` |

- **Only `main/` trains the full fusion model.** A probe loop trains a light head on encoder features in
  minutes; the full model is confounded and slow, and a probe record is not a champion result.
- **No loop writes another loop's state.**
- **One program per surface.** Two definitions of the same surface means one is stale — reconcile before
  running anything. (`spacetime/program/lfmc_gate.md` is a gate record, not a second program.)

## Setup

1. Clone `github.com/legel/deepearth` (branch `deepcal`).
2. `pip install -r requirements.txt`, then build the Earth4D CUDA hash encoder against your torch:
   `cd encoders/spacetime && bash install.sh` (the shipped .so is ABI-specific — you MUST rebuild it).
3. Read `science.md` (binding), then the `program/` of the loop you are running.
4. `python -m deepearth.autoresearch.main.instrument.prepare` — downloads and extracts the audited
   dataset (deepcal_data.zip) from NERSC into `data/deepcal/`.
5. `python -m deepearth.autoresearch.main.instrument.train autoresearch/main/instrument/deepcal.yaml --steps 8000 --device cuda:0`
   (batch 512 needs ~27GB; on a 24GB card set `batch: 256` + `pollinator_top_k: 32`). Score against
   `main/program/BENCHMARKS.md`, edit, repeat.

## Experiment budget: 10 minutes (hard cap)

Every full-model run trains for at most **10 minutes** of wall-clock (`time_budget_s: 600`, measured from
step 10 so startup and compilation are excluded), then is scored by `main/harness/evaluate.py`
(science.md rule 20). A hard cap, not a target: never raise it, never report benchmarks from a longer
run. Comparing experiments only at the equal budget is what makes a gain reflect real efficiency rather
than more steps. Kill any run that exceeds it and rerun at 600s.
