# DeepEarth core

The production model has four explicit layers:

- `world_mesh.py` builds the typed, multiresolution Earth4D state.
- `fusion.py` composes the model and writes observations into that state.
- `reader.py` routes scientific queries through the shared mesh.
- `objective.py` defines the self-supervised training objective.

```text
coordinates + time -> multiresolution Earth4D cells
modalities ---------> typed residual writes
neighbors ----------> relative mesh cells
                       |
               fibered world state
                       |
task query ----------> reader/fusion -> prediction
```

The mesh separates abiotic, visual, biological, and ecological fibers within each cell. Earth4D supplies spatial and
temporal addressing, while `SpeciesGraph` supplies phylogenomic state. Cross-scale and cross-fiber operations compose
the world state before task-conditioned attention reduces it to a prediction. Raw modalities do not bypass the mesh.

The recorded model uses 192-dimensional cells and 22,744,486 parameters. Across two public-evaluator seeds it scores
`0.385343` harmonic and `0.575491` arithmetic. The matched 128-dimensional control scores `0.379341` and
`0.561834`.

- `world_mesh.py`: Earth4D cells, relative fields, and fiber adapters.
- `fusion.py`: graph integration, mesh updates, and model interface.
- `reader.py`: query-conditioned fusion and scientific read paths.
- `objective.py`: reconstruction and structured learning terms.
- `data.py`: runtime California data contract.
- `train.py`: fixed-step training and scoring entrypoint.

The record uses 8,000 mesh steps followed by a 300-step frozen-graph reader fit:

```text
python -m deepearth.core.train --cache DATA --seed 1338 --steps 8000 --reader-steps 0
python -m deepearth.core.train --cache DATA --seed 1338 --steps 300 \
  --checkpoint deepearth/core/checkpoint.pt --reader-only
```

The public evaluator and score receipts live in `autoresearch/`; the production architecture is defined only in
`core/`.
