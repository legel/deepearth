# DeepEarth core

The model. A masked multimodal autoencoder over spatio-temporally covarying variables: given whichever variables
are observed at a location and its neighbors, it infers the rest — trained by hiding random subsets and
reconstructing them, so at inference any variable predicts any other.

```python
from deepearth.core.fusion import DeepEarth
```

`fusion.py` is the whole core. It composes two learnable encoders and fuses everything through latent attention:

- **Space-time** — `deepearth.encoders.spacetime.earth4d.Earth4D`: a CUDA hash-grid over (lat, lon, elev, time) with
  an *absolute* channel (coarse regional memory) and a *relative* channel (neighbor offsets, transferable across
  place and time).
- **Phylogenomic** — `deepearth.encoders.biological.phylogenomic.SpeciesGraph`: a learnable per-species
  representation refined over the evolutionary tree, so an observation of one species informs its relatives.

## How it works

`DeepEarth(variables, ...)` is config-driven — variables (name, continuous/categorical, width, whether a
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
