# DeepEarth autoresearch — three loops, three directories

A self-contained environment for autonomously researching and improving **DeepEarth**.

Each directory below is one autonomous research loop with **its own program definition inside it**. They
have different objectives, different instruments and different state. Do not mix them: an agent reading
two program definitions for one surface ends up optimizing a scalar nobody is scoring.

| directory | loop | program | objective | instrument |
|---|---|---|---|---|
| `main/` | full-model DeepCal | `main/autoresearch.md` | arithmetic / harmonic mean over the whole B1..B60 suite, no metric regressing | `main/train.py` + `main/evaluate.py`, via `main/run_experiment.py` and `main/score.py` |
| `biological/` | biological encoder probe | `biological/program.md` | the biological capability set | `biological/probe.py` + its staged pipeline |
| `spacetime/` | Earth4D spacetime encoder | `spacetime/program.md` | one capability record at a time from `spacetime/scorecard.md` | `spacetime/probe.py` through `spacetime/trace.py` |

Shared, belonging to no single loop: `science.md` (the binding research rules all three obey) and
`bibliography.md`.

## Boundaries that matter

- **`main/` trains the full 799M fusion model. `spacetime/` and `biological/` must not.** A probe loop
  trains a light head on frozen (or its own) encoder features in minutes; the full model is confounded
  and slow, and a probe record is not a champion result.
- **Each loop owns its own state.** `spacetime/records.json` (gitignored, single owner) plus its Ensue
  keys `LOOP-earth4d-<capability>`; `main/champion_scores.json` and `main/champion.yaml`. Nothing writes
  another loop's board.
- **The rules are shared, graduation is per-loop.** `science.md` binds everywhere; each program states
  how its own results graduate. `spacetime/lfmc_gate.md` holds the preregistered LFMC gate and its
  pinned split provenance — a gate record, not a program.
- **One agent per loop.** Two program definitions for one surface means one is stale. Reconcile before
  running anything.

## Setup

1. Clone `github.com/legel/deepearth` (branch `deepcal`).
2. `pip install -r requirements.txt`, then build the Earth4D CUDA hash encoder against your torch:
   `cd encoders/spacetime && bash install.sh` (the shipped .so is ABI-specific — you MUST rebuild it).
3. `cd deepearth/autoresearch`; read `science.md` (binding), then the program of the loop you are running.
4. `python -m deepearth.autoresearch.main.prepare` — auto-downloads + extracts the audited dataset
   (deepcal_data.zip) from NERSC into `data/deepcal/`.
5. `python -m deepearth.autoresearch.main.train autoresearch/main/deepcal.yaml --steps 8000 --device cuda:0`
   (batch 512 needs ~27GB; on a 24GB card set `batch: 256` + `pollinator_top_k: 32`). Score vs the
   committed baseline in `main/BENCHMARKS.md`, edit, repeat.

## Experiment budget: 10 minutes (hard cap)

Every full-model run trains for at most **10 minutes** of wall-clock (`time_budget_s: 600`, measured from
step 10 so startup and compilation are excluded), then is scored by `main/evaluate.py` (science.md rule
20). This is a hard cap, not a target: never raise it, never report benchmarks from a longer run.
Comparing experiments only at the equal 10-minute budget is what makes a gain reflect real efficiency
(throughput, architecture) rather than just more steps. Kill any run that exceeds the budget and rerun
at 600s.
