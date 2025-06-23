"""
V-JEPA 2 Vision Encoder for DeepEarth

This module provides tools for extracting spatiotemporal visual features
using the V-JEPA 2 model.
"""

from .vjepa2_extractor import (
    VJEPA2Extractor,
    BatchVJEPA2Extractor,
    setup_logger
)

__all__ = [
    'VJEPA2Extractor',
    'BatchVJEPA2Extractor',
    'setup_logger'
]