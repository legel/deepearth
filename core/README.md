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
               paired lens GNNs
                       |
               relation meshes
                       |
task query ----------> sparse denoiser -> prediction
```

The mesh separates abiotic, visual, biological, and ecological fibers within each cell. Paired graph streams refine
each lens, relation meshes join the fibers needed for identity, pollination, and mycorrhiza, and a task query denoises
only the relevant local segments. Raw modalities do not bypass the mesh.

The model uses 192-dimensional cells and 45,977,005 active parameters. At seed 1337 and 8,000 steps, Lance's public
evaluator scores it at `0.384141` harmonic and `0.577218` arithmetic, improving the prior seed-matched mesh from
`0.382631` and `0.572135`. The published two-seed harmonic record remains `0.385343` pending a second run.

- `world_mesh.py`: Earth4D cells, relative fields, and fiber adapters.
- `fusion.py`: graph integration, mesh updates, and model interface.
- `reader.py`: query-conditioned fusion and scientific read paths.
- `objective.py`: reconstruction and structured learning terms.
- `data.py`: runtime California data contract.
- `train.py`: fixed-step training and scoring entrypoint.

The record uses 7,900 joint steps, 100 reader steps, then a 300-step frozen-graph reader expansion. Expansion joins
the joint phase, pauses for general reader specialization, then trains alone. Repeat both commands for seeds 1337
and 1338:

```text
python -m deepearth.core.train --cache DATA --seed SEED --steps 8000 --reader-steps 100
python -m deepearth.core.train --cache DATA --seed SEED --steps 300 \
  --checkpoint deepearth/core/checkpoint.pt --reader-only
```

The public evaluator and score receipts live in `autoresearch/`; the production architecture is defined only in
`core/`.
