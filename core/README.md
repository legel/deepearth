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

- `fusion.py`: complete model and reconstruction objective.
- `data.py`: runtime California data adapter.
- `train.py`: fixed-step optimization and scoring entrypoint.

The public evaluator and score receipts live in `autoresearch/`; no architecture is duplicated there.
