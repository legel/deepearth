# DeepEarth core

The production model has five explicit layers:

- `world_mesh.py` builds the typed, multiresolution Earth4D state.
- `fusion.py` composes the model and writes observations into that state.
- `reader.py` routes scientific queries through the shared mesh.
- `ecology.py` adds environmental evidence without changing the backbone state.
- `objective.py` defines the self-supervised training objective.

```text
coordinates + time -> multiresolution Earth4D cells
modalities ---------> typed residual writes
neighbors ----------> relative mesh cells
                       |
               fibered world state
                       |
task query ----------> mesh reader -----> prediction
                          |
AlphaEarth + WorldClim + coordinates
                          |
                 ecological reranker
                          |
                 family-preserved result
```

The mesh separates abiotic, visual, biological, and ecological fibers within each cell. Scientific queries read that
shared state. For environment-to-species inference, a detached ecological stage combines AlphaEarth, WorldClim,
coordinates, and frozen mesh features. It reranks candidates only inside the family selected by the backbone, so
finer evidence cannot erase the model's coarse ecological decision.

The evaluated architecture uses 192-dimensional cells and 27,613,447 parameters. Across seeds 1337 and 1338,
Lance's unchanged public evaluator scores it at `0.385644` harmonic and `0.575712` arithmetic, improving the same
backbones from `0.385343` and `0.575491`. B23 improves by `0.005868`; B1 improves by `0.016112`.

- `world_mesh.py`: Earth4D cells, relative fields, and fiber adapters.
- `fusion.py`: graph integration, mesh updates, and model interface.
- `reader.py`: query-conditioned fusion and scientific read paths.
- `ecology.py`: family-preserving environmental reader.
- `ecology_training.py`: detached ecological fitting stages.
- `objective.py`: reconstruction and structured learning terms.
- `data.py`: runtime California data contract.
- `train.py`: fixed-step training and scoring entrypoint.

The backbone uses 7,900 joint steps, 100 reader steps, then a 300-step frozen-graph reader expansion. The ecological
stages fit after the backbone and keep it frozen. Repeat both commands for seeds 1337 and 1338:

```text
python -m deepearth.core.train --cache DATA --seed SEED --steps 8000 --reader-steps 100
python -m deepearth.core.train --cache DATA --seed SEED --steps 300 \
  --checkpoint deepearth/core/checkpoint.pt --reader-only
```

The public evaluator and score receipts live in `autoresearch/`; the production architecture is defined only in
`core/`.
