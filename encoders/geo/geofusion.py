#!/usr/bin/env python3
# =============================================================================
#  geofusion.py
# -----------------------------------------------------------------------------
#  GeoFusion Data Loader – RTK Data Processing and Management
# -----------------------------------------------------------------------------
#  A specialized data loader for processing RTK (Real-Time Kinematic) data:
#
#  Data Types
#  ---------
#  • Position Data – latitude, longitude, altitude (WGS-84)
#  • Orientation Data – yaw, pitch, roll angles
#  • Accuracy Metrics – position uncertainty estimates
#  • Metadata – timestamps, image references, etc.
#
#  Features
#  --------
#  • CSV data loading and validation
#  • Coordinate conversion integration
#  • Batch processing capabilities
#  • Accuracy-aware data handling
#  • Image name management
#
#  Quick‑start
#  -----------
#  >>> from encoders.geo.geofusion import GeoFusionDataLoader
#  >>> from encoders.geo.geo2xyz import GeospatialConverter
#  >>> converter = GeospatialConverter()
#  >>> loader = GeoFusionDataLoader(converter)
#  >>> loader.load_csv("geofusion_data.csv")
#  >>> positions, orientations = loader.convert_all()
#
#  Execute this file for data loading and conversion examples.
#
#  MIT License – © 2025 DeepEarth Contributors
# =============================================================================
from __future__ import annotations
import os
import pandas as pd
import torch
from typing import Optional, Tuple
from dataclasses import dataclass

from encoders.geo.data_structures import GeoOrientation

# --------------------------------------------------------------------------- #
#  GeoFusion data loader                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class GeoFusionEntry:
    """Single entry from GeoFusion data.
    
    Attributes:
        timestamp: Unix timestamp in seconds
        image_name: Name of the associated image file
        lat: Latitude in degrees
        lon: Longitude in degrees
        alt: Altitude in meters
        yaw: Yaw angle in degrees (heading)
        pitch: Pitch angle in degrees (elevation)
        roll: Roll angle in degrees (bank)
        latitudinal_accuracy: Latitude accuracy in meters
        longitudinal_accuracy: Longitude accuracy in meters
        altitudinal_accuracy: Altitude accuracy in meters
    """
    timestamp: float
    image_name: str
    lat: float
    lon: float
    alt: float
    yaw: float
    pitch: float
    roll: float
    latitudinal_accuracy: float
    longitudinal_accuracy: float
    altitudinal_accuracy: float
    
    @property
    def orientation(self) -> GeoOrientation:
        """Get orientation angles as GeoOrientation object."""
        return GeoOrientation(yaw=self.yaw, pitch=self.pitch, roll=self.roll)
    
    @property
    def position(self) -> List[float]:
        """Get position as [lat, lon, alt] list."""
        return [self.lat, self.lon, self.alt]


class GeoFusionDataLoader:
    """Loads and processes GeoFusion RTK data from CSV files."""
    
    def __init__(self, converter):
        """Initialize the data loader with a GeospatialConverter converter.
        
        Args:
            converter: GeospatialConverter instance for coordinate conversions
        """
        self.converter = converter
        self.data_dir = os.path.join("data", "testing")
        self.entries: list[GeoFusionEntry] = []
        
    def load_csv(self, filename: str = "geofusion.csv") -> None:
        """Load GeoFusion data from a CSV file.
        
        Args:
            filename: Name of CSV file in data/testing directory
        """
        filepath = os.path.join(self.data_dir, filename)
        data = pd.read_csv(filepath)

        # Convert to list of GeoFusionEntry objects
        self.entries = [
            GeoFusionEntry(
                timestamp=float(row['time']),
                image_name=f"{row['image']}.jpg",
                lat=float(row['latitude']),
                lon=float(row['longitude']),
                alt=float(row['altitude']),
                yaw=float(row['yaw']),
                pitch=float(row['pitch']),
                roll=float(row['roll']),
                latitudinal_accuracy=row['xyAccuracy'],
                longitudinal_accuracy=row['xyAccuracy'],
                altitudinal_accuracy=float(row['zAccuracy'])
                )
            for _, row in data.iterrows()
        ]
        
    def get_locations(self) -> torch.Tensor:
        """Get loaded location data.
        
        Returns:
            Tensor of shape (N, 3) containing [lat, lon, alt] coordinates
        """
        if not self.entries:
            raise RuntimeError("No data loaded. Call load_csv() first.")
        return torch.tensor([[e.lat, e.lon, e.alt] for e in self.entries], 
                          device=self.converter.device)
        
    def get_orientations(self) -> Optional[torch.Tensor]:
        """Get loaded orientation data.
        
        Returns:
            Tensor of shape (N, 3) containing [yaw, pitch, roll] angles
        """
        if not self.entries:
            raise RuntimeError("No data loaded. Call load_csv() first.")
        return torch.tensor([[e.yaw, e.pitch, e.roll] for e in self.entries],
                          device=self.converter.device)
        
    def get_accuracy(self) -> torch.Tensor:
        """Get position accuracy data.
        
        Returns:
            Tensor of shape (N, 2) containing [xy_accuracy, z_accuracy]
        """
        if not self.entries:
            raise RuntimeError("No data loaded. Call load_csv() first.")
        return torch.tensor([[e.xy_accuracy, e.z_accuracy] for e in self.entries],
                          device=self.converter.device)
    
    def convert_all(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert all entries to tensors.
        
        Returns:
            Tuple of:
                - positions: Tensor of shape (N, 3) with [lat, lon, alt]
                - orientations: Tensor of shape (N, 3) with [yaw, pitch, roll]
        """
        positions = torch.tensor([e.position for e in self.entries],
                               dtype=torch.float64, device=self.converter.device)
        orientations = torch.tensor([[e.yaw, e.pitch, e.roll] for e in self.entries],
                                  dtype=torch.float64, device=self.converter.device)
        return positions, orientations

