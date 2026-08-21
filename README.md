# DeepEarth

DeepEarth is a self-supervised multimodal world model for ecological inference. It represents planetary state as a
fibered, hash-addressed Earth4D mesh and learns by reconstructing masked observations from the remaining evidence.

## Architecture

```text
latitude, longitude, elevation, time
                  |
      multiresolution Earth4D hashes
                  |
    addressed cells x resolution levels
                  |
      +-----------+-----------+--------------+
      |           |           |              |
   abiotic      visual     biological     ecological
      |           |           |              |
      +------ typed residual mesh writes -----+
                  |
       cross-scale and cross-lens state
                  |
        query-conditioned reader/fusion
                  |
             target prediction
```

Earth4D supplies persistent spatial and temporal addressing. Environmental, visual, phylogenomic, and ecological
measurements write into distinct lenses at those addresses. Fusion reads only the shared mesh; raw modalities do not
bypass the world state.

## Repository

- `core/fusion.py` — the complete production mesh and query-conditioned reader.
- `core/train.py` — fixed-step optimization and public scoring entrypoint.
- `core/data.py` — runtime California data adapter.
- `encoders/` — Earth4D hash and phylogenomic primitives.
- `data/deepcal/` — reproducible source-data preparation.
- `autoresearch/` — immutable public evaluator, aggregate definitions, and score receipts only.
- `SCIENCE.md` — the scientific contract.

## Run

Install the dependencies, then build the Earth4D CUDA extension against the installed PyTorch:

```bash
pip install -r requirements.txt
cd encoders/spacetime/hashencoder && pip install -e . && cd ../../..
```

Train and score one fixed-budget seed:

```bash
python -m deepearth.core.train \
  --cache /path/to/deepcal-cache \
  --device cuda \
  --steps 2291 \
  --seed 1337
```

Repeat with seed 1338 before promoting a public record. Scores are comparable only when they use the same evaluator,
protocol tag, data, step budget, and seeds.

## Citation

DeepEarth was introduced in *Self-Supervised Multi-Modal World Model with 4D Space-Time Embedding* (2026),
[arXiv:2603.07039](https://arxiv.org/abs/2603.07039).
