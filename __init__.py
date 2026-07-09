"""DeepEarth: a self-supervised multimodal model of spatio-temporally covarying ecological variables.

Two learnable encoders -- an Earth4D space-time GNN (:mod:`deepearth.encoders.spacetime`) and a phylogenomic
species GNN (:mod:`deepearth.encoders.biological`) -- are fused by a masked multimodal autoencoder
(:mod:`deepearth.core.fusion`), which learns to reconstruct any hidden variable from the others.

    from deepearth.core.fusion import DeepEarth

See ``core/science.md`` for the scientific framing, ``core/README.md`` to prepare data and run, and
``core/autoresearch.md`` for the autonomous-experimentation loop.
"""
__version__ = "1.0.0"
