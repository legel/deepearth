# data/inputs — EDITABLE. This is the DATA lever.

When the diagnosis reads INPUT-LIMITED — the encoder does not beat a generic PE, so the current channel
lacks the signal — changing what feeds the model is the correct move, not a bigger architecture swing.
That work happens here and in `editable_files/lib/` (the loaders).

Fair game: new or richer channels, densified observations, channel fusion, per-entity aggregation,
different normalization, an additional modality.

Two rules:

- **Rebuild the prepared cache whenever inputs change** (`rm -f data/deepcal/prepared_*.pt`, or
  `--fresh-data`). The cache is lossy across data changes and will silently serve the old signal.
- **Attribute borrowed signal.** A win that comes from a frozen pretrained embedding (DINO, BioCLIP)
  is that model's signal, not the encoder's. Label it: env = *where*, vision = *which*.

`../records/` is the opposite of this directory: written by the harness, never hand-edited.
