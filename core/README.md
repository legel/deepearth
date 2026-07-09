# DeepEarth Core — DeepCal

The core of DeepEarth: a self-supervised multi-modal masked autoencoder that fuses a learnable **Earth4D
space-time GNN** and a learnable **phylogenomic species GNN** to infer any ecological variable from the others.
**DeepCal** is the California instantiation. Read [`science.md`](science.md) for what the model is and the rules it
must respect; this README is how to run it.

```python
from deepearth.core.fusion import DeepEarth
```

## Layout

| Path | Role | Edited by autoresearch? |
|---|---|---|
| `core/fusion.py` | the multi-modal masked autoencoder (tokens, latent attention, decoders) | **yes** |
| `encoders/spacetime/earth4d.py` | Earth4D space-time encoder (absolute + relative, CUDA hash grid) | **yes** |
| `encoders/biological/phylogenomic.py` | phylogenomic species GNN (tree loading + OU-attention / tree operators) | **yes** |
| `core/deepcal.yaml` | the model + training config | **yes** |
| `core/train.py` | training loop + the 26-benchmark **north star** (fixed harness) | no |
| `core/benchmarks.py` | benchmark definitions + harmonic-mean net score (the ground-truth metric) | no |
| `core/data.py` | the California data adapter (+ prepared-cache fast load) | no |
| `core/prepare.py` | one-time setup: data, CUDA kernel, prepared cache, test I/O | no |
| `core/science.md` | the scientific framing and 17 rules | — |
| `core/autoresearch.md` | the autonomous experiment loop | — |

## Quick start

```bash
# from the repo's parent directory so `deepearth` is importable
export PYTHONPATH=/home/photon/ecological

# 1. One-time setup: downloads the pre-processed DeepCal data, compiles the Earth4D CUDA kernel
#    (encoders/spacetime/install.sh), builds the prepared dataset cache, and materializes the test I/O.
python -m deepearth.core.prepare                      # uses core/deepcal.yaml

# 2. Train the champion and score the full benchmark suite + net score.
python -m deepearth.core.train                        # defaults to core/deepcal.yaml
#    add --steps N for a shorter run, --device cuda|cpu

# 3. Autonomous research (see autoresearch.md): edit the model, train under a time budget, keep if net_score rose.
```

`prepare.py` is idempotent and self-contained: on a fresh machine it pulls the data, compiles the kernel (full
`install.sh` + CUDA build), and precaches everything so training and every subsequent experiment spin up in ~1s.

## The metric

`train.py` prints the frozen benchmark suite and a single **`net_score`** — the harmonic mean of the 26 normalized
benchmarks (see `benchmarks.py`). It is the objective for autoresearch: higher is better, and because it is a
harmonic mean, the fastest way up is to lift the *weakest* benchmark. Do not modify the scoring; improve the model.

## Reproducing SoTA

The champion config (`deepcal.yaml`: `d_model 256`, ou-attention species operator, community-dropout, 8000 steps)
reaches held-out **species top-1 ≈ 0.848**, trait macro-F1 ≈ 0.895, phylo→family ≈ 0.748 on spatial-block holdout.
