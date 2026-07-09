"""DeepEarth core: the multimodal fusion model, its data adapter, and its training + evaluation harness.

    from deepearth.core.fusion import DeepEarth

Kept import-light on purpose: importing this package does not eagerly load the CUDA space-time kernel. Import
:mod:`deepearth.core.fusion` (or run :mod:`deepearth.core.train`) when you actually need the model.
"""
