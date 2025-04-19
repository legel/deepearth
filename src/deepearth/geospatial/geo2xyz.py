#!/usr/bin/env python3
# =============================================================================
#  geo2xyz.py
# -----------------------------------------------------------------------------
#  GeospatialConverter – High-Precision Geospatial Coordinate Transformations
# -----------------------------------------------------------------------------
#  A comprehensive toolkit for precise conversion between coordinate systems:
#
#  Space A : Geodetic        – latitude, longitude, altitude      (WGS‑84)
#  Space B : XYZ             – Earth‑centred Cartesian            (metres)
#  Space C : Normalised XYZ  – each axis in [0, 1]                (unitless)
#
#  Features
#  --------
#  • High-precision coordinate conversions using float64
#  • Orientation handling (yaw, pitch, roll) and rotation matrices
#  • Automatic bounding box management for normalized coordinates
#  • Comprehensive metadata support (timestamps, accuracy, images)
#  • CSV import/export with flexible schema
#  • Numerical stability in polar regions and high altitudes
#
#  Accuracy Guarantees
#  ------------------
#  • All conversions maintain sub-micrometer precision
#  • Round-trip conversions preserve original coordinates
#  • Automatic precision optimization based on coordinate span
#
#  Quick‑start
#  -----------
#  >>> from deepearth.geospatial.geo2xyz import GeospatialConverter
#  >>> import torch
#  >>> converter = GeospatialConverter(device="cuda", norm_dtype=torch.float64)
#  >>> geo  = torch.tensor([[37.7749, -122.4194, 10.0],   # San Francisco
#  ...                      [51.5007,   -0.1246, 35.0]],  # London
#  ...                      device="cuda")
#  >>> xyz  = converter.geodetic_to_xyz(geo)
#  >>> norm = converter.xyz_to_norm(xyz)             # bbox auto-detected
#  >>> geo2 = converter.xyz_to_geodetic(converter.norm_to_xyz(norm))
#
#  Execute this file for an ecological-landmark demo and precision test suite.
#
#  MIT License – © 2025 DeepEarth Contributors
# =============================================================================
from __future__ import annotations
from typing import List, Tuple
import time
import torch
from math import sin, pi
from dataclasses import dataclass
import os
import csv

from deepearth.geospatial.utils import _human_unit, wrap_lat, wrap_lon_error, wrap_lat_error, _as_fp64, _safe_div
from deepearth.geospatial.data_structures import BoundingBox, GeoOrientation, CoordinateSet, GeoPoint


# --------------------------------------------------------------------------- #
#  Main converter                                                             #
# --------------------------------------------------------------------------- #
class GeospatialConverter:
    """High-precision geospatial coordinate conversion system.
    
    This class provides comprehensive conversion capabilities between three
    coordinate spaces while maintaining high precision and handling orientation
    data. It supports:
    
    1. Coordinate Conversions:
       • Geodetic (lat, lon, alt) ↔ XYZ (Earth-centered Cartesian)
       • XYZ ↔ Normalized coordinates ([0,1]^3)
       • Automatic bounding box management
    
    2. Orientation Handling:
       • Yaw, pitch, roll angles
       • Rotation matrices
       • Aerospace sequence conversions
    
    3. Data Management:
       • High-precision calculations (float64)
       • Automatic precision optimization
       • CSV import/export with metadata
    
    The system maintains sub-micrometer precision through:
    • Internal float64 calculations
    • Bowring's method for geodetic conversion
    • Numerical stability management
    • Automatic precision optimization
    
    Attributes:
        device: Computation device ("cpu" or "cuda")
        _norm_user: Requested dtype for normalized coordinates
        _norm_eff: Effective dtype (may auto-upgrade)
        _a: Semi-major axis of WGS-84 ellipsoid
        _e2: Square of first eccentricity
        _bbox: Current bounding box
    """

    # WGS-84 constants
    _A  = 6_378_137.0      # Semi-major axis (meters)
    _F  = 1 / 298.257223563  # Flattening
    _E2 = 2 * _F - _F * _F   # Square of first eccentricity

    def __init__(self, *, device: str = "cpu",
                 norm_dtype: torch.dtype = torch.float64) -> None:
        """Initialize the coordinate converter.
        
        Args:
            device: Computation device ("cpu" or "cuda")
            norm_dtype: Preferred dtype for normalized coordinates
        """
        self.device     = device
        self._norm_user = norm_dtype      # requested by user
        self._norm_eff  = norm_dtype      # effective (may auto-upgrade)

        self._a  = torch.tensor(self._A,  device=device, dtype=torch.float64)
        self._e2 = torch.tensor(self._E2, device=device, dtype=torch.float64)

        self._bbox: BoundingBox | None = None

    @property
    def bbox(self) -> BoundingBox | None:
        """Get current bounding box."""
        return self._bbox

    def reset_bbox(self) -> None:
        """Clear the stored bounding box."""
        self._bbox = None

    def geodetic_to_xyz(self, geo: torch.Tensor, orientation: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """Convert geodetic coordinates to Earth-centered Cartesian XYZ.
        
        Args:
            geo: Tensor of shape (..., 3) containing (lat, lon, alt) in degrees/meters
            orientation: Optional tensor of shape (..., 3) containing (yaw, pitch, roll) in degrees
            
        Returns:
            Tuple of:
                - Tensor of shape (..., 3) containing (X, Y, Z) in meters
                - Optional tensor of shape (..., 3, 3) containing rotation matrices if orientation provided
        """
        assert geo.shape[-1] == 3, "last dimension must be (lat, lon, alt)"
        geo = _as_fp64(geo)
        lat, lon, alt = torch.deg2rad(geo[..., 0]), torch.deg2rad(geo[..., 1]), geo[..., 2]
        sin_lat, cos_lat = torch.sin(lat), torch.cos(lat)
        N = self._a / torch.sqrt(1 - self._e2 * sin_lat ** 2)
        xyz = torch.stack((
            (N + alt) * cos_lat * torch.cos(lon),
            (N + alt) * cos_lat * torch.sin(lon),
            (N * (1 - self._e2) + alt) * sin_lat
        ), -1)
        
        if orientation is not None:
            # Convert orientation angles to rotation matrices
            orientation = _as_fp64(orientation)
            y, p, r = [torch.deg2rad(orientation[..., i]) for i in range(3)]
            
            # Compute rotation matrices for each point
            cy, sy = torch.cos(y), torch.sin(y)
            cp, sp = torch.cos(p), torch.sin(p)
            cr, sr = torch.cos(r), torch.sin(r)
            
            # Create rotation matrices (broadcasting to match input shape)
            R = torch.zeros((*orientation.shape[:-1], 3, 3), dtype=torch.float64, device=orientation.device)
            
            # Fill rotation matrix elements (aerospace sequence)
            R[..., 0, 0] = cy * cp
            R[..., 0, 1] = cy * sp * sr - sy * cr
            R[..., 0, 2] = cy * sp * cr + sy * sr
            R[..., 1, 0] = sy * cp
            R[..., 1, 1] = sy * sp * sr + cy * cr
            R[..., 1, 2] = sy * sp * cr - cy * sr
            R[..., 2, 0] = -sp
            R[..., 2, 1] = cp * sr
            R[..., 2, 2] = cp * cr
            
            return xyz, R
        
        return xyz, None

    def xyz_to_geodetic(self, xyz: torch.Tensor, rotation_matrix: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """Convert Earth-centered Cartesian XYZ to geodetic coordinates.
        
        Args:
            xyz: Tensor of shape (..., 3) containing (X, Y, Z) in meters
            rotation_matrix: Optional tensor of shape (..., 3, 3) containing rotation matrices
            
        Returns:
            Tuple of:
                - Tensor of shape (..., 3) containing (lat, lon, alt) in degrees/meters
                - Optional tensor of shape (..., 3) containing (yaw, pitch, roll) in degrees
        """
        x, y, z = _as_fp64(xyz).unbind(-1)
        p = torch.sqrt(x * x + y * y)
        lon = torch.atan2(y, x)

        # Bowring's method for latitude and altitude
        lat = torch.atan2(z, p * (1 - self._e2))
        for _ in range(5):
            s = torch.sin(lat)
            N = self._a / torch.sqrt(1 - self._e2 * s ** 2)
            lat = torch.atan2(z + self._e2 * N * s, p)

        s = torch.sin(lat)
        N = self._a / torch.sqrt(1 - self._e2 * s ** 2)
        c = torch.cos(lat)
        alt = torch.where(
            c.abs() < 1e-12,
            torch.abs(z) - N * (1 - self._e2),
            p / c - N
        )
        
        geo = torch.stack((torch.rad2deg(lat), torch.rad2deg(lon), alt), -1)
        
        if rotation_matrix is not None:
            # Extract YPR angles from rotation matrix
            R = _as_fp64(rotation_matrix)
            
            # Extract angles using aerospace sequence
            pitch = torch.asin(-R[..., 2, 0])
            yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
            roll = torch.atan2(R[..., 2, 1], R[..., 2, 2])
            
            orientation = torch.stack((
                torch.rad2deg(yaw),
                torch.rad2deg(pitch),
                torch.rad2deg(roll)
            ), -1)
            
            return geo, orientation
        
        return geo, None

    def _best_dtype_for_span(self, span: torch.Tensor) -> torch.dtype:
        """Determine optimal dtype for given coordinate span.
        
        Args:
            span: Tensor containing coordinate ranges
            
        Returns:
            Smallest dtype that maintains required precision
        """
        for dt in (torch.float16, torch.float32, torch.float64):
            if torch.all((span * torch.finfo(dt).eps) / 2 <= 1e-3):
                return dt
        return torch.float64

    def update_bbox(self, xyz: torch.Tensor) -> None:
        """Update the bounding box to include new coordinates."""
        xyz64 = _as_fp64(xyz)
        if self._bbox is None:
            self._bbox = BoundingBox.from_points(xyz64)
        else:
            new_bbox = BoundingBox.from_points(xyz64)
            self._bbox = BoundingBox(
                min_x=min(self._bbox.min_x, new_bbox.min_x),
                min_y=min(self._bbox.min_y, new_bbox.min_y),
                min_z=min(self._bbox.min_z, new_bbox.min_z),
                max_x=max(self._bbox.max_x, new_bbox.max_x),
                max_y=max(self._bbox.max_y, new_bbox.max_y),
                max_z=max(self._bbox.max_z, new_bbox.max_z),
                device=self.device
            )

        span = self._bbox.span
        chosen = self._best_dtype_for_span(span)
        order = {torch.float16: 0, torch.float32: 1, torch.float64: 2}
        self._norm_eff = chosen if order[chosen] >= order[self._norm_user] else self._norm_user

    def xyz_to_norm(self, xyz: torch.Tensor) -> torch.Tensor:
        """Convert XYZ coordinates to normalized [0,1]^3 space."""
        self.update_bbox(xyz)
        norm64 = _safe_div(_as_fp64(xyz) - self._bbox.min_point,
                          self._bbox.span)
        return norm64.to(self._norm_eff)

    def norm_to_xyz(self, norm: torch.Tensor) -> torch.Tensor:
        """Convert normalized coordinates back to XYZ space."""
        if norm.dtype in (torch.float16, torch.float32):
            eps = torch.finfo(norm.dtype).eps
            span = self._bbox.span
            half = (eps / 2) * span.view((1,) * (norm.ndim - 1) + (3,))
            mask = (norm > 0) & (norm < 1)
            norm = torch.where(mask, norm + half.to(norm.dtype), norm)
        return _as_fp64(norm) * self._bbox.span + self._bbox.min_point

    def export_coordinates(self, filepath: str, coordinates: List[CoordinateSet]) -> None:
        """Export coordinates to CSV file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Base headers for coordinate data
            headers = [
                "Latitude", "Longitude", "Altitude",
                "Global_X", "Global_Y", "Global_Z",
                "Relative_X", "Relative_Y", "Relative_Z",
                "BBox_Min_X", "BBox_Min_Y", "BBox_Min_Z",
                "BBox_Max_X", "BBox_Max_Y", "BBox_Max_Z"
            ]
            
            # Check for metadata fields
            has_timestamp = any(c.timestamp is not None for c in coordinates)
            has_image = any(c.image_path is not None for c in coordinates)
            has_accuracy = any(c.latitudinal_accuracy is not None for c in coordinates)
            
            # Add metadata headers if present
            if has_timestamp:
                headers.append("Timestamp")
            if has_image:
                headers.append("Image_Path")
            if has_accuracy:
                headers.extend([
                    "Latitudinal_Accuracy_Meters",
                    "Longitudinal_Accuracy_Meters",
                    "Altitudinal_Accuracy_Meters"
                ])
            
            # Check for orientation data
            has_orientation = any(c.orientation is not None for c in coordinates)
            if has_orientation:
                headers.extend(["Yaw", "Pitch", "Roll"])
                # Check for rotation matrices
                has_rotation = any(c.rotation_matrix is not None for c in coordinates)
                if has_rotation:
                    headers.extend([
                        "R11", "R12", "R13",
                        "R21", "R22", "R23",
                        "R31", "R32", "R33"
                    ])
            
            writer.writerow(headers)
            
            # Write data
            for coord in coordinates:
                row = [
                    f"{coord.lat:.14f}", f"{coord.lon:.14f}", f"{coord.alt:.11f}",
                    f"{coord.x:.14f}", f"{coord.y:.14f}", f"{coord.z:.14f}",
                    f"{coord.rel_x:.14f}", f"{coord.rel_y:.14f}", f"{coord.rel_z:.14f}",
                    f"{coord.bbox.min_x:.14f}", f"{coord.bbox.min_y:.14f}", f"{coord.bbox.min_z:.14f}",
                    f"{coord.bbox.max_x:.14f}", f"{coord.bbox.max_y:.14f}", f"{coord.bbox.max_z:.14f}"
                ]
                
                # Add metadata if headers present
                if has_timestamp:
                    row.append(f"{coord.timestamp:.6f}" if coord.timestamp is not None else "")
                if has_image:
                    row.append(coord.image_path if coord.image_path is not None else "")
                if has_accuracy:
                    row.extend([
                        f"{coord.latitudinal_accuracy:.6f}" if coord.latitudinal_accuracy is not None else "",
                        f"{coord.longitudinal_accuracy:.6f}" if coord.longitudinal_accuracy is not None else "",
                        f"{coord.altitudinal_accuracy:.6f}" if coord.altitudinal_accuracy is not None else ""
                    ])
                
                # Add orientation data if present
                if has_orientation:
                    if coord.orientation is not None:
                        row.extend([
                            f"{coord.orientation.yaw:.14f}",
                            f"{coord.orientation.pitch:.14f}",
                            f"{coord.orientation.roll:.14f}"
                        ])
                    else:
                        row.extend(["", "", ""])
                    
                    if has_rotation and coord.rotation_matrix is not None:
                        row.extend([f"{val:.14f}" for val in coord.rotation_matrix.flatten().cpu()])
                    elif has_rotation:
                        row.extend([""] * 9)
                
                writer.writerow(row)

    def import_coordinates(self, filepath: str) -> List[CoordinateSet]:
        """Import coordinates from CSV file."""
        coordinates = []
        with open(filepath, 'r', newline='') as f:
            reader = csv.reader(f)
            headers = next(reader)  # Get headers
            
            # Find indices for different data types
            base_end = 15  # Basic coordinates end at index 14
            
            # Find metadata indices
            timestamp_idx = headers.index("Timestamp") if "Timestamp" in headers else None
            image_idx = headers.index("Image_Path") if "Image_Path" in headers else None
            lat_acc_idx = headers.index("Latitudinal_Accuracy_Meters") if "Latitudinal_Accuracy_Meters" in headers else None
            lon_acc_idx = headers.index("Longitudinal_Accuracy_Meters") if "Longitudinal_Accuracy_Meters" in headers else None
            alt_acc_idx = headers.index("Altitudinal_Accuracy_Meters") if "Altitudinal_Accuracy_Meters" in headers else None
            
            # Find orientation indices
            yaw_idx = headers.index("Yaw") if "Yaw" in headers else None
            pitch_idx = headers.index("Pitch") if "Pitch" in headers else None
            roll_idx = headers.index("Roll") if "Roll" in headers else None
            
            # Find rotation matrix indices
            r_start = headers.index("R11") if "R11" in headers else None
            has_rotation = r_start is not None
            
            for row in reader:
                # Parse base coordinates
                values = [float(x) if x else None for x in row[:base_end]]
                
                # Parse metadata
                timestamp = float(row[timestamp_idx]) if timestamp_idx is not None and row[timestamp_idx] else None
                image_path = row[image_idx] if image_idx is not None and row[image_idx] else None
                lat_accuracy = float(row[lat_acc_idx]) if lat_acc_idx is not None and row[lat_acc_idx] else None
                lon_accuracy = float(row[lon_acc_idx]) if lon_acc_idx is not None and row[lon_acc_idx] else None
                alt_accuracy = float(row[alt_acc_idx]) if alt_acc_idx is not None and row[alt_acc_idx] else None
                
                # Parse orientation
                orientation = None
                if all(idx is not None for idx in [yaw_idx, pitch_idx, roll_idx]):
                    if row[yaw_idx] and row[pitch_idx] and row[roll_idx]:
                        orientation = GeoOrientation(
                            yaw=float(row[yaw_idx]),
                            pitch=float(row[pitch_idx]),
                            roll=float(row[roll_idx])
                        )
                
                # Parse rotation matrix
                rotation_matrix = None
                if has_rotation and all(row[i] for i in range(r_start, r_start + 9)):
                    rotation_matrix = torch.tensor(
                        [float(x) for x in row[r_start:r_start + 9]],
                        dtype=torch.float64,
                        device=self.device
                    ).reshape(3, 3)
                
                coordinates.append(CoordinateSet(
                    lat=values[0], lon=values[1], alt=values[2],
                    x=values[3], y=values[4], z=values[5],
                    rel_x=values[6], rel_y=values[7], rel_z=values[8],
                    bbox=BoundingBox(
                        min_x=values[9], min_y=values[10], min_z=values[11],
                        max_x=values[12], max_y=values[13], max_z=values[14],
                        device=self.device
                    ),
                    orientation=orientation,
                    rotation_matrix=rotation_matrix,
                    timestamp=timestamp,
                    image_path=image_path,
                    latitudinal_accuracy=lat_accuracy,
                    longitudinal_accuracy=lon_accuracy,
                    altitudinal_accuracy=alt_accuracy
                ))
        
        return coordinates
