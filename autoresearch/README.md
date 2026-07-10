# DeepEarth autoresearch

A self-contained environment for autonomously researching and improving the **DeepCal** model — DeepEarth trained
over California plant observations. Land here with zero context, read this file, and you can go end-to-end: prepare
the data, train, score, and iterate.

## Quick start

```bash
export PYTHONPATH=/path/to/parent-of-deepearth      # so `import deepearth` works

python -m deepearth.autoresearch.prepare            # downloads + caches data, builds the Earth4D CUDA kernel,
                                                    # precaches the assembled dataset + test I/O (idempotent)
python -m deepearth.autoresearch.train              # trains DeepCal and prints the benchmark suite + net_score
```

`prepare.py` fetches the pre-processed data (≈3.6 GB) from the NERSC portal, compiles the kernel, and builds a
prepared cache so every later run starts in ~1 s. Then an experiment is one command:

```bash
python -m deepearth.autoresearch.train --time_budget 600    # a 10-minute experiment
```

## The objective

`evaluate.py` scores a trained model on a **frozen benchmark suite** and prints a single **`net_score`** — the
unweighted mean of the normalized benchmarks. Higher is better. That number is the north star; maximize it by
improving the model. Never edit the scoring.

## The loop

Read [`autoresearch.md`](autoresearch.md) — it defines the experiment loop (edit → train 10 min → score →
keep/revert), the budget, and the rules of engagement. It is short; read it in full.

## The science

Read [`science.md`](science.md) — what DeepEarth is, its two learnable encoders, and the numbered rules every change
must respect (all data always included; minimal files and tokens; the 10-minute budget; the architecture
constraints). Reference it for *why*; do not re-derive it.

## The files — read exactly these

The **critical-path surface that is subject to change** (rule 19). Read all of them; the four marked ✎ are your
primary edit targets for improving the model.

| File | Role | |
|---|---|---|
| `core/fusion.py` | the multimodal masked-autoencoder model | ✎ |
| `encoders/spacetime/earth4d.py` | Earth4D space-time hash-grid encoder | ✎ |
| `encoders/biological/phylogenomic.py` | phylogenomic species GNN | ✎ |
| `autoresearch/deepcal.yaml` | model + training config | ✎ |
| `autoresearch/train.py` | training loop (optimizer, schedule, masking, budget) | |
| `encoders/spacetime/hashencoder/` | the CUDA hash kernel (`hashgrid.py`, `backend.py`, `setup.py`, `src/{hashencoder.cu,hashencoder.h,bindings.cpp,utils.cuh}`) — keep it fast (science rule 4) | |
| `core/README.md`, `autoresearch/README.md`, `science.md`, `autoresearch.md` | the docs — kept current with the code | |

**Fixed harness — never edit** (not part of the surface above): `prepare.py` (data + kernel setup) and `evaluate.py`
(the benchmark suite and scoring, the immutable ground truth). `data.py` is the data adapter, fixed alongside them.

### Size of the surface (rule 19)

Currently **22 files, ~71k tokens** (tiktoken `cl100k`). Every condensing pass re-measures this and drives it down.

| group | files | tokens |
|---|---|---|
| model + config (edit targets) | 4 | 25.0k |
| training regime (`train.py`) | 1 | 3.8k |
| CUDA kernel (`hashencoder/`) | 7 | 37.2k |
| docs | 4 | 5.0k |
| package `__init__.py` | 6 | 0.4k |

The CUDA group grew (18.6k → 37.2k) when the sparse-Adam + precompute subsystem — the champion default (`sparse_hash: true`, +12% throughput / +0.011 net) — was synthesized into the single `hashencoder.cu` and made non-compromising (bit-identical training, learnable resolution). A dead-code + parsimony pass trimmed it 43k → 37.2k; the remainder is functional kernel/dispatch code earning its tokens, not prose.

The fixed harness (`prepare.py`, `evaluate.py`, `data.py` — ~10k tokens) is **excluded** by rule 19.

## End-to-end, from nothing

1. Clone `github.com/legel/deepearth`, `cd deepearth/autoresearch`, read this `README.md`.
2. Read `autoresearch.md` (the loop) and `science.md` (the rules).
3. `python -m deepearth.autoresearch.prepare` — data + kernel + caches are ready.
4. Establish the baseline (`train.py`), then run the loop: edit a ✎ file, train 10 min, score, keep if `net_score`
   rose, else revert. Repeat, indefinitely.
