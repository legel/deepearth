# DeepEarth autoresearch — granular probe loops, then fusion

We are building **backwards**. Each research loop owns **one probe and its data**, and recovers signal
for one part of the science. Only when the science in `science.md` is actually filled out do those
recovered signals get plugged into the fusion layer — the full model comes last, not first.

```
                          ┌──────────────────────────────────────────┐
   APEX                   │  autoresearch/main/       FUSION         │  runs LAST
   consumes probe results │  integrates finished encoders            │
                          └───────────────▲──────────────────────────┘
                                          │ graduation.py — a tested prediction, not a copied score
            ┌─────────────────────────────┴─────────────────────────────┐
   PROBE    │ autoresearch/probes/spacetime/   autoresearch/probes/biological/ │
   LOOPS    │ one probe · own data · own metric · own evals · independent code │
            └───────────────────────────────────────────────────────────┘
```

Why this way round: a fusion model trained before its constituent signals are established cannot tell
you which part works. It is confounded and slow, and every result it produces is a joint claim about
everything at once. A probe loop makes one narrow claim, in minutes, against fair controls — and a claim
that survives its own validation is what earns a place in the fusion layer.

So a probe loop's job is not "raise the aggregate". It is: **recover a real signal on one capability,
with its own evals, and prove it is the encoder's and not borrowed.**

## The one rule the layout enforces: the judge is not editable

```
   harness/            THE JUDGE      measurement, scoring, gating, recording   never edited to win a run
   editable_files/     THE SCIENCE    model, objective, data, config            the ONLY surface
```

Nothing in `editable_files/` should be something you must read to understand *how you are scored* — only
things you change *in order to score better*. Every scorer used to live inside a directory named for the
fact that agents edit it, and `score.py` kept a hand-copy of `is_diagnostic`/`_net_value` under a comment
reading "keep byte-identical". One definition now, and it is out of reach.

```
autoresearch/
  science.md          binding research rules — all loops obey them
  scorecard.md        INDEX of every loop's scorecard — start here to read progress
  bibliography.md     references
  .env                credentials (gitignored) — ENSUE_API_TOKEN; template in .env.example

  harness/            SHARED JUDGE — only what is genuinely cross-loop
    definitions.py          what every number MEANS + the METRIC REGISTRY + routing
    graduation.py         probe record -> champion crossing ledger

  probes/             the probe loops — independent siblings, one probe each
    spacetime/          Earth4D space-time encoder
      harness.py          this probe's own judge: contract, registry, gate, ledger, publish
      determinism.py      attributes the trained path's nondeterminism
      probe.py             fixed evaluation implementation
      editable_files/     SCIENCE: earth4d.py, hashencoder/, lib/
      program/  records/
    biological/         biological encoder — same shape

  main/               the apex — fusion over the whole B1..B60 suite, runs last
    harness/            evaluate.py, champion_report.py, hooks.py, score.py, run_experiment.py
    editable_files/     SCIENCE: fusion/fusion.py, train.py, *.yaml, lib/{data,prepare,recipes}
    program/  records/

  data/               the shared corpus. All loops read it, so it belongs to no single loop.
```

## Routing: pick a metric, get the file

The mapping from *what you want to improve* to *what you edit* is one table — `scoring.METRICS` — and it
lives in the harness so an experiment cannot widen its own scope. It used to live in four places that
disagreed (`scorecard.md` prose, `program.md`'s LEVER_SITES, `score.py`'s partitions, `graduation.py`'s own
dict), and LEVER_SITES was still pointing at a `lib/gnn.py` that had been deleted weeks earlier.

```bash
python -m deepearth.autoresearch.scoring.definitions --capability species_from_spacetime
python -m deepearth.autoresearch.scoring.definitions --metric B8_family_from_spacetime
python -m deepearth.autoresearch.scoring.definitions --file earth4d.py
python -m deepearth.autoresearch.scoring.definitions --audit      # orphan metrics, orphan files, leaks
python -m deepearth.autoresearch.scoring.definitions --coverage   # which science.md rules have an instrument
```

Every metric row names: what it measures · the science.md rule that demands it · the editable file(s)
that move it · the probe capability that estimates it cheaply.

## Scope — what an experiment may edit

```
   IN SCOPE      <loop>/editable_files/**    public model entrypoint + modular scientific lib
   OUT OF SCOPE  harness/**                  the shared judge
                 <loop>/harness*             the loop's own judge
                 <loop>/records/**           the board and its ledgers
                 <loop>/program/**           the contract
                 any OTHER loop's directory  not your surface
                 science.md                  unless that is the declared experiment
```

Anything outside `editable_files/` is not an experiment — it is infrastructure, and it goes in its own
commit with its own tests, separate from any result. A change to what a number *means* re-baselines every
board that number appears on: bump the protocol, and say so.

## Per-loop contents

| loop | program/ | harness (the judge) | editable_files/ (the science) | records/ |
|---|---|---|---|---|
| `main/` | `autoresearch.md`, `BENCHMARKS.md`, `CHAMPION_REPORT.md`, `audit.md`, `GRADUATION_BLUEPRINT.md` | `evaluate.py`, `champion_report.py`, `hooks.py`, `score.py`, `run_experiment.py` | `fusion/fusion.py`, `train.py`, `deepcal.yaml`, `champion.yaml`, `lib/{data,prepare,recipes}` | `champion_scores.json`, `graduation.jsonl` |
| `probes/spacetime/` | `program.md`, `scorecard.md`, `scorecard.txt`, `box-operations.md` | `harness.py`, `probe.py`, `determinism.py` | package API, `earth4d.py`, `hashencoder/`, and modular `lib/` science | `records.json`, `traces/` |
| `probes/biological/` | `program.md` | `harness/` | `phylogenomic.py`, `lib/{seeds,training}.py` | — |

## Boundaries between the loops

- **Only `main/` trains the full fusion model, and it runs LAST.** A probe trains a light head on encoder
  features in minutes. A probe record is not a champion result; it is a candidate signal.
- **Probe results reach `main` through `graduation.py`, never by copying a score.** A probe score and a
  benchmark score are different instruments on different models — neither bounds the other. What crosses
  is a prediction that gets tested: *probe says capability X improved → champion re-measures B(X) → did it
  move?* Each crossing appends a row to `main/records/graduation.jsonl`, and that ledger is the only thing
  that will ever tell you whether a probe gain transfers.
- **Each probe is a fixed consumer of its loop's public science entrypoint.** Fusion imports those same
  entrypoints (`earth4d.py` and `phylogenomic.py`), never probe code or their private `lib/` modules.
- **Each loop optimizes its own metric under its own evals.** No loop is scored on another's number.
- **No loop writes another loop's `records/`.**
- **One program per surface.** Two definitions of the same surface means one is stale — reconcile before
  running anything.

## Setup

1. Clone `github.com/legel/deepearth` (branch `deepcal`).
2. `pip install -r requirements.txt` and `pip install ninja` — the Earth4D CUDA kernel JIT-compiles on
   first import and caches under `hashencoder/build/`.
3. Read `science.md` (binding), then the `program/` of the loop you are running.
4. `python -m deepearth.autoresearch.main.editable_files.lib.prepare` — downloads and extracts the audited
   dataset (deepcal_data.zip) from NERSC into `autoresearch/data/deepcal/`.
5. `python -m deepearth.autoresearch.main.editable_files.train autoresearch/main/editable_files/deepcal.yaml --steps 8000 --device cuda:0`
   (batch 512 needs ~27GB; on a 24GB card set `batch: 256` + `pollinator_top_k: 32`). Score against
   `main/program/BENCHMARKS.md`, edit, repeat.

## Experiment budget: 10 minutes (hard cap)

Every full-model run trains for at most **10 minutes** of wall-clock (`time_budget_s: 600`, measured from
step 10 so startup and compilation are excluded), then is scored by `main/harness/evaluate.py`
(science.md rule 20). A hard cap, not a target: never raise it, never report benchmarks from a longer run.
Comparing experiments only at equal budget is what makes a gain reflect real efficiency rather than more
steps. Kill any run that exceeds it and rerun at 600s.

**The probe loops use a different currency** — `CONFIG["steps"]`, not wall-clock. That is a known gap:
science.md rule 21 makes throughput a first-class score lever, and under a step budget a kernel speedup
cannot move any probe number. `scoring --coverage` lists it alongside the other unmeasured axes.
