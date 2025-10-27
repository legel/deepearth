"""
DeepEarth XYZT Encoders
======================

Spatiotemporal encoders for planetary-scale (X,Y,Z,T) deep learning.

Available encoders:
- Earth4D: Grid4D-based encoder for latitude, longitude, elevation, timestamp
"""

from .earth4d import (
    Earth4D,
    Grid4DSpatiotemporalEncoder,
    CoordinateConverter
)

__all__ = [
    'Earth4D',
    'Grid4DSpatiotemporalEncoder',
    'CoordinateConverter'
]

__version__ = '1.0.0'