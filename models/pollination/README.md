# Pollination

Dynamic forecasting of plant-pollinator interaction networks across California (and ultimately Earth).

## Dataset

Per-image DINOv3 ViT-L/16 spatial embeddings of every Research-grade iNaturalist observation of every California-observed flying pollinator (1,275 taxa), joined with 45,805 GloBi pollination interaction records linking those animals to California-native plants:

**[`deepearth/california-flourishing-pollination`](https://huggingface.co/datasets/deepearth/california-flourishing-pollination)** on Hugging Face

The companion pipeline ([`legel/california_flourishing_pollination`](https://github.com/legel/california_flourishing_pollination)) ingests GloBi (Poelen 2014) with the controlled vocabulary of pollination interaction types (`pollinates`, `pollinatedBy`, `visitsFlowersOf`, `flowersVisitedBy` — RO ontology IRIs), cross-checks each candidate pollinator against iNaturalist CA observations and a curated flight-ability rule table, and emits an analysis-ready interaction graph. The `pollination` half of that repository (network construction, taxonomic cross-checks, animal-side embeddings) feeds this model.

## Model (in development)

A DeepEarth pollination model that ingests:

- **Earth4D positional encoding** for both plant and pollinator localities.
- **DINOv3 species embeddings** for plants and pollinators.
- **Flowering state** from the companion [`flowering`](../flowering/) model.
- **Climate + weather forcing** (NSF NERSC compute via DOE BER).

The goal: probabilistic per-day, per-location, per-(plant×pollinator)-pair visitation surfaces, with Bayesian uncertainty — supporting environmental planners in maximizing pollination services through native-plant placement, and supporting ecological scientists in studying pollinator decline.

_Companion model:_ [`flowering`](../flowering/) — predicts the floral-resource state that drives pollinator visitation.
