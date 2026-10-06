"""Native-plant habitat: where each native vascular plant of the contiguous United States can live, mapped at 240 m.

Two models share one data pipeline. The per-species model follows Daru (2024, PNAS) step by step (occurrences, names,
cleaning, native filter, thinning, calibration area, effort-weighted background, MaxEnt) and stores each species as a
range card (``codec``). The joint model (``joint``) fits every species at once: one environment network shared by all
species, a Brownian-motion phylogenetic prior on their niche vectors, MaxEnt's point-process objective, and a
transform-coded map store decoded on any CPU (``joint.Store``). Method and provenance: README.md, docs/.
"""
