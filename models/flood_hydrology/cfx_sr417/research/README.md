# research/ — learned flood surrogates (exploratory)

**This directory is not part of the reproducible digital-twin pipeline.** Nothing in
`../simulation/`, `../viewer/`, or the data-fetch scripts depends on it. It is kept because the
results are informative — including where they are negative — not because anything here is
production-ready.

The question these experiments ask: **can a learned surrogate replace the physics solver and
still be trustworthy?** Two architectures were tried on the same site. The honest summary is
that neither is usable as a solver replacement today, and the *reason* is more interesting than
the failure itself.

## What was tried

| | Mesh GNN | Grid Transformer |
|---|---|---|
| Files | `train_mesh_gnn*.py`, `validate_gnn_rollout.py`, `run_gnn_training_sweep.py` | `grid_transformer_surrogate.py`, `train_grid_transformer_surrogate.py`, `evaluate_grid_transformer_*.py` |
| Representation | Per-edge message passing on an unstructured triangle mesh | Conv autoencoder (32x spatial compression) + cross-attention Transformer |
| Reference | HydroGraphNet / MeshGraphKAN | FloodSformer (Pianforini et al. 2025) |

Training corpora are built by `build_grid_surrogate_dataset_site3*.py` (design-storm sweep) and
`build_ian_rollout_dataset_site3.py` (a real Hurricane Ian reconstruction — the only
non-synthetic test set).

## Results, stated plainly

**1. Speed is real, but only for the compressed representation.**
The grid transformer runs a full-resolution forward pass (1,363x1,372 = 1.87M cells) in 0.231 s
on CPU / 0.120 s on MPS — roughly 7.9x / 15.3x faster than the physics solver's equivalent wall
time. The mesh GNN is the opposite: fine at the coarse crop it was trained on (~0.022 s/pass at
6,701 nodes), but at full site3 scale (8.67M edges) it OOMs on GPU and takes 304.8 s/pass on
CPU — about **56x slower than the solver it was meant to accelerate**. Spatial compression
decouples cost from cell count; per-edge message passing does not.

**2. Accuracy has at least three independent axes, and fixing one says nothing about the others.**
This is the most transferable finding here.

- *Pointwise RMSE* stayed ~0.032 m across variants whose volume error ranged from −89% to −99.9%.
  RMSE is averaged over a domain that is mostly correctly-dry, so a model that has stopped
  tracking water accumulation still scores well. **Single-frame RMSE alone is not evidence of a
  physically trustworthy rollout.**
- *Aggregate volume* collapsed under plain MSE training. Adding a mass-conservation loss term
  (`--vol-loss-weight`) improved 1-yr volume drift from −94.5% ± 4.0pp to −48.2% ± 29.4pp
  (3 seeds). The improvement generalises to the real Ian event (−97.7% → −84.9%).
- *Spatial wet-cell pattern* did not improve at all. Wet-cell IoU is near-zero for **every**
  variant tried, including the best mass-conservation run (~0.013–0.021 vs FloodSformer's
  reported 0.818). The working diagnosis is that 32x compression reduces the training domain to
  a 9x9 latent grid, too coarse to resolve flood-extent boundaries regardless of loss function.

**3. A naive "predict no change" baseline is competitive with the trained model.**
Persistence gives −93.3% / −93.8% volume drift (1yr / 500yr) versus the trained plain-MSE
baseline's −94.5% ± 4.0pp / −98.1% ± 1.6pp. At 10% into the rollout the trained model is
actually *worse* than doing nothing. Any surrogate claim here has to clear this bar first.

**4. The architecture gap is not about domain size.**
Running the grid transformer on the mesh GNN's exact crop and resolution did not rescue it
(−87.1% ± 1.8pp with the mass-conservation loss), while the mesh GNN on that same region
reached +25.1% / +4.3% volume drift. So the difference is a genuine architecture effect, not a
consequence of how much domain each model sees.

## Known limitations

- The mass-conservation loss is genuinely hyperparameter-sensitive: a single-seed sweep shows a
  mid-range sweet spot (epochs 10→−11.3%, 15→−0.9%, 20→−7.3%, 30→−42.9%, 80→−94.7%), and it is
  **not** characterised across seeds. Treat the headline number as fragile.
- A less-aggressively-compressed variant (fewer downsampling layers) is the obvious next test
  for the spatial-IoU problem. It has not been attempted.
- All results are site3 (Gee Creek) only.

## Running these

Scripts resolve the project root from their own location and add `../simulation/` to
`sys.path` for the solver modules, so they can be run from anywhere:

```bash
python3 research/build_grid_surrogate_dataset_site3.py     # build the training corpus
python3 research/train_grid_transformer_surrogate.py --epochs 20 --vol-loss-weight 5.0 --seed 0
python3 research/evaluate_grid_transformer_checkpoints.py  # rollout + naive baseline
bash    research/run_reliability_batch.sh                  # multi-seed sweep
```

Trained weights land in `checkpoints/` and are gitignored (`*.pt`) — regenerate rather than
expect them in a clone. `LITERATURE_REVIEW.md` in this directory is the citation source of
truth for the papers these experiments are measured against.
