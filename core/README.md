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

The recorded 14.5M-parameter model scores `0.379341` harmonic and `0.561834`
arithmetic across two seeds with the public evaluator. The previous public
control scores `0.378407` and `0.587374`, respectively.

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

The public evaluator and score receipts live in `autoresearch/`; no architecture is duplicated there.
