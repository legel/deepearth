"""
DeepEarth XYZT Encoders
======================

Spatiotemporal encoders for planetary-scale (X,Y,Z,T) deep learning.

Available encoders:
- Earth4D: Grid4D-based encoder for latitude, longitude, elevation, timestamp

Tools:
- earth4d_collision_profiler: Statistical profiling of hash collisions in Earth4D
"""

from .earth4d import (
    Earth4D,
    Grid4DSpatiotemporalEncoder
)

__all__ = [
    'Earth4D',
    'Grid4DSpatiotemporalEncoder'
]

__version__ = '1.0.0'