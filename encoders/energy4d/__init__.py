"""
Energy4D: Relative Spatiotemporal Encoder

Energy4D uses relative coordinate offsets instead of absolute coordinates,
enabling better generalization across different locations and times.

Key difference from Earth4D:
- Earth4D: Encodes absolute (lat, lon, elev, time)
- Energy4D: Encodes relative (Δx, Δy, Δz, Δt) from each reference point
"""

from .energy4d import Energy4D

__all__ = ["Energy4D"]
