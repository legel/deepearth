"""DeepEarth: a self-supervised multimodal model of spatio-temporally covarying ecological variables.

Two learnable encoders -- Earth4D space-time and phylogenomic species -- are fused by a masked
multimodal autoencoder (:mod:`deepearth.autoresearch.main.editable_files.fusion.fusion`), which learns
to reconstruct any hidden variable from the others.

    from deepearth.autoresearch.main.editable_files.fusion.fusion import DeepEarth

See ``autoresearch/science.md`` for the scientific framing, ``autoresearch/README.md`` to prepare data and run,
and the `/research` command for the autonomous-experimentation loop.
"""
__version__ = "1.0.0"
