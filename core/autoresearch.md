# autoresearch — DeepCal

Autonomous, hill-climbing research on DeepCal, in the spirit of Karpathy's
[autoresearch](https://github.com/karpathy/autoresearch). An agent edits the model, trains for a fixed time budget,
scores it on a frozen benchmark suite, keeps the change if the score improved, and repeats — indefinitely.

**Read [`science.md`](science.md) first.** It defines what DeepEarth is, the two innovations (Earth4D SpaceTime GNN,
Phylogenomic Species GNN), and the 17 rules every change must respect. This file is the *how*; `science.md` is the
*what* and *why*.

## The three-part split (do not blur it)

- **`prepare.py`** — one-time setup: downloads the DeepCal data, builds the Earth4D CUDA kernel, precaches the
  test-inference I/O. Fixed. Do not modify.
- **`train.py`** — the harness: the training loop, the 26 benchmarks, and the harmonic-mean **north star**. This is
  the ground-truth metric. **Do not modify it to change how scoring works** (that is cheating the benchmark).
- **The model — the ONLY thing you edit:**
  - `core/fusion.py` — the multi-modal masked autoencoder (tokens, latent attention, decoders).
  - `encoders/spacetime/earth4d.py` — the Earth4D space-time encoder.
  - `encoders/biological/phylogenomic.py` — the phylogenomic species GNN.
  - `core/deepcal.yaml` — the model/training config (sizes, LRs, windows, operator choice).

Everything about the model is fair game: architecture, attention, operators, fusion, hyperparameters, the species
operator (`ou-attention` ↔ `tree`), Earth4D levels/windows/hashmaps — **as long as the 17 rules in `science.md`
hold** (Earth4D stays ≥100M params and CUDA-fast; the phylo network stays tree-derived, learnable, and updates
neighbors every batch; every token carries space-time + phylo + modality + type; masked-autoencoding /
distribution-estimation objective).

## The metric — the north star

The frozen suite is **26 benchmarks** (10 scientific `Q1..Q10` + 15 practical `A1..A15` + the 12-month flowering
temporal benchmark), each normalized `(value − baseline) / (target − baseline)` clipped to `[0, 1]`. The single
**net score** is their **harmonic mean** (power mean, p = −1): it rewards lifting the *weakest* benchmarks, so you
cannot win by sacrificing one modality for another. Higher is better. `train.py` prints it as `net_score`, plus the
per-benchmark raw values. Extract with `grep "^net_score:" run.log`.

**Simplicity criterion** (from Karpathy): all else equal, simpler wins. A small gain that adds ugly complexity is
not worth it; a gain (or a wash) from *deleting* code is a great outcome.

## The experiment loop

Each experiment runs for a **fixed budget of up to 10 minutes** of training (wall clock, excluding
startup/compile). Launch: `python -m deepearth.core.train` (defaults to `core/deepcal.yaml`). The first run is
always the **baseline** — run it as-is, unmodified.

The experiment runs on a dedicated branch (`autoresearch/<tag>`).

**LOOP FOREVER:**
1. Look at the git state (current branch/commit); read the latest `results.tsv` and think about what to try next.
2. Form a hypothesis grounded in `science.md` and the current benchmark profile (which benchmarks are lowest?).
3. Edit the model (`fusion.py` / `earth4d.py` / `phylogenomic.py` / `deepcal.yaml`).
4. `git commit` the change.
5. Run: `python -m deepearth.core.train > run.log 2>&1` (redirect everything; never flood context with `tee`).
6. Read the result: `grep "^net_score:\|^peak_vram_mb:" run.log`. Empty ⇒ it crashed: `tail -n 50 run.log`, read
   the traceback, fix if it's a simple bug, else discard.
7. Record in `results.tsv` (leave it untracked by git).
8. If `net_score` improved, **keep** the commit (advance the branch). If equal or worse, `git reset --hard` back.

`results.tsv` (tab-separated, header + rows):

```
commit	net_score	species@1	trait_f1	phylo_fam	vram_gb	status	description
```

`status` ∈ {`keep`, `discard`, `crash`}. Use `net_score = 0.000000` for crashes.

## Rules of engagement

- **NEVER STOP.** Once the loop begins, do not pause to ask whether to continue. If out of ideas, think harder: read
  the papers referenced in `science.md`, re-read the in-scope files for new angles, combine near-misses, try more
  radical architecture. The loop runs until manually stopped.
- **Fix, don't sleep on, failures.** A crash is a bug to solve, not a reason to give up on the idea. Get to the root
  cause. Do not let experiments crash repeatedly without resolution.
- **True hill-climbing.** Every experiment targets the same north star. No divergent goals. Advance the branch on
  real gains only; revert otherwise.
- **Respect the harness.** Never edit `train.py`'s scoring or `prepare.py`. The benchmark is the ground truth.
- **Timeout.** An experiment should finish in ≈10 min. If a run exceeds ~15 min, kill it, treat as failure, revert.

## First actions

1. `python -m deepearth.core.prepare` (idempotent; downloads data, builds the kernel, precaches test I/O).
2. Establish the baseline: `python -m deepearth.core.train`, record it as row 1 of `results.tsv`.
3. Begin the loop.
