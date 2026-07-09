"""
DeepEarth SpaceTime Encoders
============================

Spatiotemporal encoders for planetary-scale (X, Y, Z, T) deep learning.

Available encoders:
- Earth4D: hash-grid encoder over latitude, longitude, elevation, timestamp (absolute + relative channels).
"""

from .earth4d import Earth4D, parse_timestamp

__all__ = ['Earth4D', 'parse_timestamp']

__version__ = '1.0.0'