#!/usr/bin/env python3
# =============================================================================
#  data_structures.py
# -----------------------------------------------------------------------------
#  Geospatial Data Structures – Core Types
# -----------------------------------------------------------------------------
#  Essential data structures for geospatial coordinate handling:
#
#  Core Types
#  ---------
#  • BoundingBox – Geographic area definition
#  • GeoOrientation – Camera orientation angles
#  • GeoPoint – Single point coordinates
#  • CoordinateSet – Complete coordinate data
#
#  Features
#  --------
#  • Type-safe coordinate handling
#  • Validation and normalization
#  • CSV import/export support
#  • Tensor-based calculations
#  • Orientation data management
#
#  Quick‑start
#  -----------
#  >>> from geospatial.data_structures import CoordinateSet
#  >>> coords = CoordinateSet(
#  ...     lat=37.7749,
#  ...     lon=-122.4194,
#  ...     alt=0.0
#  ... )
#  >>> coords.to_csv_row()  # Export to CSV format
#
#  These structures are used throughout the GeospatialConverter.
#
#  MIT License – © 2025 DeepEarth Contributors
# =============================================================================

import torch
from dataclasses import dataclass
from typing import Tuple, List, Optional

# --------------------------------------------------------------------------- #
#  Core data structures                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class BoundingBox:
    """Represents a 3D bounding box for coordinate normalization."""
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float
    device: str = "cpu"  # Device for tensor operations

    def to(self, device: str) -> 'BoundingBox':
        """Return a copy of this bounding box on the specified device."""
        return BoundingBox(
            min_x=self.min_x, min_y=self.min_y, min_z=self.min_z,
            max_x=self.max_x, max_y=self.max_y, max_z=self.max_z,
            device=device
        )

    @property
    def min_point(self) -> torch.Tensor:
        """Return minimum point as tensor on the correct device."""
        return torch.tensor([self.min_x, self.min_y, self.min_z],
                          dtype=torch.float64, device=self.device)

    @property
    def max_point(self) -> torch.Tensor:
        """Return maximum point as tensor on the correct device."""
        return torch.tensor([self.max_x, self.max_y, self.max_z],
                          dtype=torch.float64, device=self.device)

    @property
    def span(self) -> torch.Tensor:
        """Return the span (max - min) as tensor on the correct device."""
        return self.max_point - self.min_point

    @classmethod
    def from_tensors(cls, min_point: torch.Tensor, max_point: torch.Tensor) -> 'BoundingBox':
        """Create BoundingBox from tensor min/max points."""
        return cls(
            min_x=min_point[0].item(),
            min_y=min_point[1].item(),
            min_z=min_point[2].item(),
            max_x=max_point[0].item(),
            max_y=max_point[1].item(),
            max_z=max_point[2].item(),
            device=min_point.device.type
        )

    @classmethod
    def from_points(cls, points: torch.Tensor) -> 'BoundingBox':
        """Create BoundingBox from a set of points."""
        min_point = torch.amin(points, dim=tuple(range(points.ndim - 1)))
        max_point = torch.amax(points, dim=tuple(range(points.ndim - 1)))
        return cls.from_tensors(min_point, max_point)

@dataclass
class GeoOrientation:
    """Represents orientation angles in geodetic space.
    
    Attributes:
        yaw: Rotation around Z axis (heading) in degrees
        pitch: Rotation around Y axis (elevation) in degrees
        roll: Rotation around X axis (bank) in degrees
    """
    yaw: float
    pitch: float
    roll: float
    
    def to_radians(self) -> Tuple[float, float, float]:
        """Convert angles to radians."""
        return (
            torch.deg2rad(self.yaw).item(),
            torch.deg2rad(self.pitch).item(),
            torch.deg2rad(self.roll).item()
        )
    
    def to_rotation_matrix(self) -> torch.Tensor:
        """Convert YPR angles to 3x3 rotation matrix.
        
        Returns:
            3x3 rotation matrix representing the orientation
            
        Note:
            Uses the aerospace sequence: yaw -> pitch -> roll
            This matches the common convention for drone navigation
        """
        y, p, r = self.to_radians()
        
        # Yaw matrix (around Z)
        cy, sy = torch.cos(y), torch.sin(y)
        Ry = torch.tensor([
            [cy, -sy, 0],
            [sy,  cy, 0],
            [0,   0,  1]
        ], dtype=torch.float64)
        
        # Pitch matrix (around Y)
        cp, sp = torch.cos(p), torch.sin(p)
        Rp = torch.tensor([
            [cp,  0, sp],
            [0,   1,  0],
            [-sp, 0, cp]
        ], dtype=torch.float64)
        
        # Roll matrix (around X)
        cr, sr = torch.cos(r), torch.sin(r)
        Rr = torch.tensor([
            [1,  0,   0],
            [0, cr, -sr],
            [0, sr,  cr]
        ], dtype=torch.float64)
        
        # Combined rotation matrix (applied in order: yaw -> pitch -> roll)
        return Rr @ Rp @ Ry

@dataclass
class GeoPoint:
    """Represents a point in geodetic space with optional orientation.
    
    Attributes:
        lat: Latitude in degrees
        lon: Longitude in degrees
        alt: Altitude in meters
        orientation: Optional YPR orientation angles
    """
    lat: float
    lon: float
    alt: float
    orientation: GeoOrientation | None = None

@dataclass
class CoordinateSet:
    """Represents a complete set of coordinates in all spaces."""
    # Original geodetic coordinates
    lat: float
    lon: float
    alt: float
    # Global XYZ coordinates
    x: float
    y: float
    z: float
    # Normalized coordinates
    rel_x: float
    rel_y: float
    rel_z: float
    # Bounding box
    bbox: BoundingBox
    # Orientation data
    orientation: GeoOrientation | None = None  # YPR angles
    rotation_matrix: torch.Tensor | None = None  # 3x3 rotation matrix
    # Metadata and accuracy metrics
    timestamp: float | None = None  # Unix timestamp
    image_path: str | None = None  # Path to associated image
    latitudinal_accuracy: float | None = None  # Latitude accuracy in meters
    longitudinal_accuracy: float | None = None  # Longitude accuracy in meters
    altitudinal_accuracy: float | None = None  # Altitude accuracy in meters

    def to_csv_row(self) -> List[str]:
        """Convert to CSV row format."""
        row = [
            f"{self.lat:.14f}", f"{self.lon:.14f}", f"{self.alt:.11f}",
            f"{self.x:.14f}", f"{self.y:.14f}", f"{self.z:.14f}",
            f"{self.rel_x:.14f}", f"{self.rel_y:.14f}", f"{self.rel_z:.14f}",
            f"{self.bbox.min_x:.14f}", f"{self.bbox.min_y:.14f}", f"{self.bbox.min_z:.14f}",
            f"{self.bbox.max_x:.14f}", f"{self.bbox.max_y:.14f}", f"{self.bbox.max_z:.14f}"
        ]
        
        # Add metadata if present
        if self.timestamp is not None:
            row.append(f"{self.timestamp:.6f}")
        if self.image_path is not None:
            row.append(self.image_path)
        if self.latitudinal_accuracy is not None:
            row.append(f"{self.latitudinal_accuracy:.6f}")
        if self.longitudinal_accuracy is not None:
            row.append(f"{self.longitudinal_accuracy:.6f}")
        if self.altitudinal_accuracy is not None:
            row.append(f"{self.altitudinal_accuracy:.6f}")
            
        # Add orientation data if present
        if self.orientation:
            row.extend([
                f"{self.orientation.yaw:.14f}",
                f"{self.orientation.pitch:.14f}",
                f"{self.orientation.roll:.14f}"
            ])
            if self.rotation_matrix is not None:
                row.extend([f"{val:.14f}" for val in self.rotation_matrix.flatten().cpu()])
        return row

    @classmethod
    def from_csv_row(cls, row: List[str], device: str = "cpu") -> 'CoordinateSet':
        """Create CoordinateSet from CSV row.
        
        Args:
            row: CSV row values
            device: Device to create tensors on
        """
        values = [float(x) if not x.endswith('.jpg') and not x.endswith('.png') else x for x in row]
        orientation = None
        rotation_matrix = None
        timestamp = None
        image_path = None
        lat_accuracy = None
        lon_accuracy = None
        alt_accuracy = None
        
        # Find where the basic coordinates end
        base_end = 15  # Basic coordinates end at index 14
        current_idx = base_end
        
        # Parse metadata if present
        if len(values) > current_idx and isinstance(values[current_idx], float):
            timestamp = values[current_idx]
            current_idx += 1
        if len(values) > current_idx and isinstance(values[current_idx], str):
            image_path = values[current_idx]
            current_idx += 1
        if len(values) > current_idx + 2:
            lat_accuracy = float(values[current_idx])
            lon_accuracy = float(values[current_idx + 1])
            alt_accuracy = float(values[current_idx + 2])
            current_idx += 3
            
        # Parse orientation data if present
        if len(values) > current_idx + 2:
            orientation = GeoOrientation(
                yaw=float(values[current_idx]),
                pitch=float(values[current_idx + 1]),
                roll=float(values[current_idx + 2])
            )
            current_idx += 3
            
            # Parse rotation matrix if present
            if len(values) > current_idx + 8:
                rotation_matrix = torch.tensor(
                    [float(x) for x in values[current_idx:current_idx + 9]], 
                    dtype=torch.float64, 
                    device=device
                ).reshape(3, 3)
        
        return cls(
            lat=values[0], lon=values[1], alt=values[2],
            x=values[3], y=values[4], z=values[5],
            rel_x=values[6], rel_y=values[7], rel_z=values[8],
            bbox=BoundingBox(
                min_x=values[9], min_y=values[10], min_z=values[11],
                max_x=values[12], max_y=values[13], max_z=values[14],
                device=device
            ),
            orientation=orientation,
            rotation_matrix=rotation_matrix,
            timestamp=timestamp,
            image_path=image_path,
            latitudinal_accuracy=lat_accuracy,
            longitudinal_accuracy=lon_accuracy,
            altitudinal_accuracy=alt_accuracy
        )
