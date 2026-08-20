# DeepEarth core

DeepEarth has two production model surfaces:

```python
from deepearth.core.fusion import DeepEarth
from deepearth.core.mesh import MeshModel
```

`fusion.py` is the established masked multimodal autoencoder. `mesh.py` tests a stricter world-model boundary:
all observations write into one hash-addressed state and task-conditioned fusion reads only that state.

Both compose the same learnable foundations:

- **Space-time** — `deepearth.encoders.spacetime.earth4d.Earth4D`: a CUDA hash-grid over (lat, lon, elev, time) with
  an *absolute* channel (coarse regional memory) and a *relative* channel (neighbor offsets, transferable across
  place and time).
- **Phylogenomic** — `deepearth.encoders.biological.phylogenomic.SpeciesGraph`: a learnable per-species
  representation refined over the evolutionary tree, so an observation of one species informs its relatives.

## Mesh path

The mesh path separates representation from reading:

```text
Earth4D address + typed observations -> multiresolution fibered mesh
query + mesh state                  -> sparse reader/fusion -> prediction
```

- Earth4D addresses planetary cells across space, time, and resolution.
- Abiotic, visual, biological, and ecological lenses keep unlike evidence distinct within each cell.
- Residual writes update one shared state; raw modalities do not bypass it.
- Query-conditioned attention reads the relevant cells, levels, and lenses into one latent.

`MeshModel` accepts the prepared source and variable contract from its caller. Data assembly, optimization, and
canonical scoring remain in the research harness; they are not part of the production model.

## Established fusion path

`DeepEarth(variables, ...)` is config-driven—variables (name, continuous/categorical, width, whether a
reconstruction target, whether carried from neighbors) are passed in, not hard-coded.

- **Tokens** — each observed variable becomes a token: its value-embedding + a learnable type-embedding, fused with
  the query's Earth4D position (`tok_norm`/`pos_norm` keep content and position at matched scale). A dedicated
  always-present position token survives full masking. Neighbors add one token per (neighbor, subspace) via
  `NeighborContext`; the species variable is read from the refined `SpeciesGraph`.
- **Processor** — a small set of learnable latents read the token set (cross-attention) then attend among themselves
  (`n_layers` transformer blocks). Pure PyTorch, so it compiles cleanly while the Earth4D CUDA kernel stays eager.
- **Reconstruction** — for a random reveal mask, every hidden-but-observed variable is decoded from the latents and
  scored (cosine for continuous, class-normalized cross-entropy for categorical). `reconstruction_loss` and
  `infer(given, targets)` are the training and inference entry points.

The architecture is deliberately minimal and general — the specifics of any instantiation live in a config and a
data adapter, not here. See `deepearth/autoresearch/` for a complete training environment and the scientific rules.
