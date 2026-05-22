# Flowering

Dynamic forecasting of plant flowering across California (and ultimately Earth) — when, where, and for which species.

## Dataset

Per-image DINOv3 ViT-L/16 spatial embeddings of every Research-grade iNaturalist observation of every California-native plant taxon, plus PhenoVision (Dinnage 2025) flower-presence labels:

**[`deepearth/california-flourishing-pollination`](https://huggingface.co/datasets/deepearth/california-flourishing-pollination)** on Hugging Face

The companion pipeline that produces this dataset (data collection → DINOv3 inference → HF publication, with full scientific provenance) lives in [`legel/california_flourishing_pollination`](https://github.com/legel/california_flourishing_pollination). The `flourishing` half of that repository (flower-presence classification, phenological state, plant condition) feeds this model.

## Model (in development)

A DeepEarth flowering model that ingests:

- **Earth4D positional encoding** (latitude, longitude, elevation, time) — proven state-of-the-art on the LFMC benchmark with no other inputs (see [`fire_ecology`](../fire_ecology/) and [`encoders/xyzt/benchmarks/lfmc`](../../encoders/xyzt/benchmarks/lfmc)).
- **DINOv3 spatial features** of the CFP dataset — frozen, reusable.
- **Species embeddings** — learned per CA-native taxon.
- **PhenoVision flower-presence label** — supervision signal.

The goal: continuous, per-species, spatially-resolved flowering probability surfaces, refreshed daily from in-situ observations + Earth observation, supporting environmental planning that maximizes biodiversity and pollination outcomes.

_Companion model:_ [`pollination`](../pollination/) — predicts the resulting plant-pollinator interaction network conditioned on flowering state.
