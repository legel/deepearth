# autoresearch

Autonomous research on DeepEarth. One model, one objective, one loop.

**Start here:** [`PROGRAM.md`](../PROGRAM.md) is the operating contract; the `/research` command executes it.
**How science is measured:** [`scorecard.md`](../autoresearch/scorecard.md).
**What the model must be:** [`science.md`](science.md).

## Structure

```
main/
  editable_files/      the science — change these to test a hypothesis
    fusion/fusion.py     the model
    encoders/            earth4d.py (space-time), phylogenomic.py (biological), hashencoder/ (CUDA)
    train.py             the training loop and val_bpb
    lib/                 data assembly and preparation
    *.yaml               configs: deepcal (full), screen (proxy), champion
  harness/             the instruments — do not change to make a result pass
    evaluate.py          the 63 benchmarks
    hooks.py             ablation primitives (earth4d, species graph)
  records/             champion scores

scoring/objective.py   val_bpb, its per-variable decomposition, the measured noise floor
BENCHMARKS.md          the suite
tests/                 what must not break
```

Data ETL lives in top-level [`datatools/`](../datatools) — one-time downloaders for GBIF, NAIP, soil,
topography, Daymet and the phylogenies. It builds the prepared cache and is not part of the loop.

## The objective in one paragraph

The human-capability suite decides promotion. Its two-seed harmonic is the primary score and its
two-seed arithmetic mean is the breadth guard. The weakest active capability selects the research
target. `val_bpb` is held-out masked-reconstruction likelihood in bits per revealed dimension; its
aggregate, macro view, decomposition and ablations diagnose where a change landed but never decide
promotion. See [`scorecard.md`](scorecard.md) for exact membership and gate semantics.

## What changed, and why

This was three loops: a space-time probe, a biological probe, and fusion, each with its own objective,
board, protocol and gate. Probe results did not transfer, because the probes trained discriminative
classifiers over encoder features while fusion trains masked reconstruction. That is not a bug to fix by
tuning — the objectives were different, so a probe win had no reason to be a fusion win, and measurably
was not.

Collapsing them removed ~11,000 lines with no measurable score change (0.278777 → 0.279769 harmonic at
24M, inside the 0.0167 two-seed spread at that scale). The isolation the probes provided is now an
ablation on the decomposition.
