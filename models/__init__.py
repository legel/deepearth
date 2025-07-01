"""DeepEarth model components.

This module provides the core building blocks for the DeepEarth
spatiotemporal multimodality simulator.
"""

from .configs import (
    DeepEarthConfig,
    TransformerConfig,
    ModalityConfig
)

from .hash_encoding import MultiResolutionHashEncoding

from .encoders import (
    Grid4DEncoder,
    ModalityEncoder
)

from .decoders import (
    ModalityDecoder,
    SpatiotemporalDecoder
)

from .transformers import (
    Transformer,
    TransformerBlock,
    MultiHeadAttention
)

__all__ = [
    # Configurations
    'DeepEarthConfig',
    'TransformerConfig',
    'ModalityConfig',
    
    # Encoders
    'Grid4DEncoder',
    'ModalityEncoder',
    'MultiResolutionHashEncoding',
    
    # Decoders
    'ModalityDecoder',
    'SpatiotemporalDecoder',
    
    # Transformers
    'Transformer',
    'TransformerBlock',
    'MultiHeadAttention',
]
