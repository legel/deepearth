# DeepEarth core

`fusion.py` is the single production model. It writes every observation into a fibered, hash-addressed Earth4D mesh
and lets target queries read only that shared world state.

```text
coordinates + time -> multiresolution Earth4D cells
modalities ---------> typed residual writes
neighbors ----------> relative mesh cells
                       |
               fibered world state
                       |
task query ----------> sparse reader/fusion -> prediction
```

Each cell separates abiotic, visual, biological, and ecological lenses. Earth4D provides persistent spatial and
temporal addressing; the phylogenomic `SpeciesGraph` GNN supplies biological state; cross-scale and cross-lens
operations compose the mesh before query-conditioned attention reduces it to a prediction. Raw modalities do not
bypass the mesh.

The recorded 22.7M-parameter model uses 192-dimensional mesh cells. It scores
`0.385343` harmonic and `0.575491` arithmetic across two seeds with the public
evaluator. The 14.5M, 128-dimensional mesh scored `0.379341` and `0.561834`.

- `fusion.py`: complete model and reconstruction objective.
- `data.py`: runtime California data adapter.
- `train.py`: fixed-step optimization and scoring entrypoint.

The recorded model uses an 8,000-step mesh checkpoint followed by a 300-step
frozen-graph reader fit:

```text
python -m deepearth.core.train --cache DATA --seed 1338 --steps 8000
python -m deepearth.core.train --cache DATA --seed 1338 --steps 300 \
  --checkpoint deepearth/core/checkpoint.pt --reader-only
```

The unchanged public evaluator lives in `autoresearch/`; no architecture is duplicated there.
