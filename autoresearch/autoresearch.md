# autoresearch

Autonomous, hill-climbing research on the DeepEarth model. You edit the model, train for a fixed budget, score, keep
the change if the score improved, and repeat — indefinitely.

**Objective:** maximize the single **`net_score`** printed by `evaluate.py` — the unweighted mean of the frozen
benchmark suite (see `README.md` for the numbers). Higher is better.

**What you edit (fair game):** the model and its config — the enumerated critical-path files in `README.md`
(`core/fusion.py`, `encoders/spacetime/earth4d.py`, `encoders/biological/phylogenomic.py`, `deepcal.yaml`). The
architecture, encoders, fusion, operators, optimizer/schedule, and hyperparameters are all fair game.

**What you must NOT touch (fixed harness / ground truth):** `prepare.py` (data + kernel setup) and `evaluate.py`
(the benchmark suite and scoring). Never edit scoring to inflate a result; improve the model.

**Rules of the science:** every change must respect `science.md`.

## The loop

Run on a dedicated branch (`autoresearch/<tag>`). The first run is the unmodified **baseline**.

1. Read the current `results.tsv` and the benchmark profile; form a hypothesis grounded in `science.md`.
2. Edit a model file (or `deepcal.yaml`); `git commit`.
3. Train: `python -m deepearth.autoresearch.train --time_budget 600 > run.log 2>&1` (10 min; redirect —
   never flood context).
4. Read `grep "^net_score:\|^peak_vram_mb:" run.log`. Empty ⇒ crash: `tail -50 run.log`, fix if simple else revert.
5. Append to `results.tsv` (untracked). If `net_score` rose, keep the commit; else `git reset --hard` to the base.

**Budget:** 10 minutes of training per experiment (startup/compile excluded), enforced by `--time_budget 600`.
**Simplicity wins:** a gain (or a wash) from *deleting* code is a great outcome; ugly complexity for a tiny gain is not.
**Never stop:** once looping, do not pause to ask whether to continue — think harder, read the papers in `science.md`,
combine near-misses, try more radical architecture. Fix crashes at the root; don't let experiments fail repeatedly.

## First actions

1. `python -m deepearth.autoresearch.prepare` — downloads + caches data, builds the CUDA kernel, precaches test I/O.
2. Baseline: `python -m deepearth.autoresearch.train --time_budget 600`; record row 1 of `results.tsv`.
3. Begin the loop.
