"""DeepEarth: Self-Supervised Spatiotemporal Multimodality Simulator.

DeepEarth is an AI foundation model for deep multimodal spatiotemporal 
simulation of physical systems. It can learn to reconstruct masked 
spatiotemporal distributions of datasets from physics, chemistry, biology, 
geology, and ecology.
"""

__version__ = "0.1.0"

from .models import (
    DeepEarthConfig,
    TransformerConfig,
    ModalityConfig,
    Grid4DEncoder,
    ModalityEncoder,
    ModalityDecoder,
    SpatiotemporalDecoder,
    Transformer
)

# Import main model if it exists
try:
    from .deepearth import DeepEarthModel
    __all__ = [
        'DeepEarthModel',
        'DeepEarthConfig',
        'TransformerConfig',
        'ModalityConfig',
        'Grid4DEncoder',
        'ModalityEncoder',
        'ModalityDecoder',
        'SpatiotemporalDecoder',
        'Transformer'
    ]
except ImportError:
    # Main model not yet implemented
    __all__ = [
        'DeepEarthConfig',
        'TransformerConfig',
        'ModalityConfig',
        'Grid4DEncoder',
        'ModalityEncoder',
        'ModalityDecoder',
        'SpatiotemporalDecoder',
        'Transformer'
    ]
