"""DeepEarth core: the multimodal fusion model, its data adapter, and its training + evaluation harness.

    from deepearth.autoresearch.main.editable_files.fusion.fusion import DeepEarth

Kept import-light on purpose: importing this package does not eagerly load the CUDA space-time kernel. Import
:mod:`deepearth.autoresearch.main.editable_files.fusion.fusion` (or run :mod:`deepearth.autoresearch.main.editable_files.fusion.train`) when you actually need the model.
"""
