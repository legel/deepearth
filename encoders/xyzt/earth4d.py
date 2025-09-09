"""
Earth4D: Grid4D Encoder for Planetary (X,Y,Z,T) Deep Learning
============================================================

Earth4D provides a Grid4D-based spatiotemporal encoder for planetary-scale 
deep learning tasks involving latitude, longitude, elevation, and time coordinates.

The encoder uses decomposed hash encoding with separate spatial (xyz) and 
temporal (xyt, yzt, xzt) projections for efficient 4D representation learning.

Key Features:
- Multi-resolution hash encoding for scalable feature extraction
- Configurable spatial and temporal resolution hierarchies
- Optional automatic ECEF coordinate conversion and normalization
- Configurable multi-resolution scales in meters and seconds
- Designed for planetary-scale spatiotemporal modeling

Usage:
    from deepearth.encoders.xyzt import Earth4D
    
    # Basic usage with normalized coordinates
    encoder = Earth4D()
    spatial_features, temporal_features = encoder(coordinates_xyzt)
    
    # Advanced usage with raw coordinates and custom scales
    encoder = Earth4D(
        auto_ecef_convert=True,
        spatial_scales_meters=[16, 32, 64, 128, 256, 512],
        temporal_scales_seconds=[3600, 86400, 604800]  # hour, day, week
    )
    features = encoder(raw_coordinates)

Author: Grid4D LFMC Team
License: MIT
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Union, Tuple, Dict
import warnings

# Import hash encoder - adjust path as needed for your environment
try:
    from .hashencoder.hashgrid import HashEncoder
except ImportError:
    warnings.warn(
        "HashEncoder not found. Please install the hash encoding library. "
        "This encoder is required for Grid4D functionality."
    )
    HashEncoder = None


class CoordinateConverter:
    """
    Coordinate conversion utilities for Earth4D.
    
    Handles conversion between geographic coordinates (lat, lon, elevation, time)
    and normalized ECEF coordinates suitable for Grid4D encoding.
    """
    
    def __init__(self, 
                 earth_radius: float = 6371000.0,  # meters
                 time_origin: float = 0.0,          # reference time
                 time_scale: float = 1.0):          # time scaling factor
        """
        Initialize coordinate converter.
        
        Args:
            earth_radius: Earth radius in meters for ECEF conversion
            time_origin: Reference time point (seconds since epoch)  
            time_scale: Scaling factor for time normalization
        """
        self.earth_radius = earth_radius
        self.time_origin = time_origin
        self.time_scale = time_scale
        
    def geographic_to_ecef(self, 
                          lat: torch.Tensor, 
                          lon: torch.Tensor, 
                          elevation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert geographic coordinates (lat, lon, elevation) to ECEF.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees  
            elevation: Elevation in meters above sea level
            
        Returns:
            Tuple of (x, y, z) ECEF coordinates in meters
        """
        # Convert to radians
        lat_rad = torch.deg2rad(lat)
        lon_rad = torch.deg2rad(lon)
        
        # ECEF conversion
        radius = self.earth_radius + elevation
        x = radius * torch.cos(lat_rad) * torch.cos(lon_rad)
        y = radius * torch.cos(lat_rad) * torch.sin(lon_rad)  
        z = radius * torch.sin(lat_rad)
        
        return x, y, z
    
    def normalize_coordinates(self, 
                            x: torch.Tensor, 
                            y: torch.Tensor, 
                            z: torch.Tensor, 
                            t: torch.Tensor,
                            spatial_bound: float = 1.0,
                            temporal_bound: float = 1.0) -> torch.Tensor:
        """
        Normalize ECEF coordinates and time to [0, 1] range.
        
        Args:
            x, y, z: ECEF coordinates
            t: Time coordinates
            spatial_bound: Spatial normalization bound
            temporal_bound: Temporal normalization bound
            
        Returns:
            Normalized coordinates tensor of shape (..., 4)
        """
        # Find bounds for normalization
        spatial_coords = torch.stack([x, y, z], dim=-1)
        spatial_min = torch.min(spatial_coords.view(-1, 3), dim=0)[0]
        spatial_max = torch.max(spatial_coords.view(-1, 3), dim=0)[0]
        spatial_range = spatial_max - spatial_min
        
        # Normalize spatial coordinates to [0, spatial_bound]
        x_norm = (x - spatial_min[0]) / spatial_range[0] * spatial_bound
        y_norm = (y - spatial_min[1]) / spatial_range[1] * spatial_bound  
        z_norm = (z - spatial_min[2]) / spatial_range[2] * spatial_bound
        
        # Normalize temporal coordinates
        t_normalized = (t - self.time_origin) / self.time_scale
        t_min = torch.min(t_normalized)
        t_max = torch.max(t_normalized)
        t_range = t_max - t_min
        t_norm = (t_normalized - t_min) / t_range * temporal_bound
        
        return torch.stack([x_norm, y_norm, z_norm, t_norm], dim=-1)
    
    def process_raw_coordinates(self,
                               lat: torch.Tensor,
                               lon: torch.Tensor, 
                               elevation: torch.Tensor,
                               time: torch.Tensor,
                               spatial_bound: float = 1.0,
                               temporal_bound: float = 1.0) -> torch.Tensor:
        """
        Complete pipeline: geographic -> ECEF -> normalized.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            elevation: Elevation in meters
            time: Time coordinates
            spatial_bound: Spatial normalization bound
            temporal_bound: Temporal normalization bound
            
        Returns:
            Normalized XYZT coordinates tensor
        """
        # Convert to ECEF
        x, y, z = self.geographic_to_ecef(lat, lon, elevation)
        
        # Normalize
        return self.normalize_coordinates(x, y, z, time, spatial_bound, temporal_bound)


class Grid4DSpatiotemporalEncoder(nn.Module):
    """
    Core Grid4D spatiotemporal encoder.
    
    Implements decomposed hash encoding for 4D coordinates with separate
    spatial (xyz) and temporal (xyt, yzt, xzt) encoding pathways.
    """
    
    def __init__(self,
                 # Spatial encoding configuration
                 spatial_levels: int = 16,
                 spatial_features: int = 2,
                 spatial_base_res: int = 16,
                 spatial_max_res: int = 512,
                 spatial_hashmap: int = 19,
                 # Temporal encoding configuration  
                 temporal_levels: int = 16,
                 temporal_features: int = 2,
                 temporal_base_res: List[int] = None,
                 temporal_max_res: List[int] = None,
                 temporal_hashmap: int = 19,
                 # Coordinate bounds
                 spatial_bound: float = 1.0,
                 temporal_bound: float = 1.0):
        """
        Initialize Grid4D encoder.
        
        Args:
            spatial_*: Configuration for xyz spatial encoding
            temporal_*: Configuration for 3D temporal projection encodings
            spatial_bound: Normalization bound for spatial coordinates
            temporal_bound: Normalization bound for temporal coordinates
        """
        super().__init__()
        
        if HashEncoder is None:
            raise ImportError(
                "HashEncoder is required for Grid4D functionality. "
                "Please install the hash encoding library."
            )
        
        # Default temporal resolutions if not provided
        if temporal_base_res is None:
            temporal_base_res = [8, 8, 8]
        if temporal_max_res is None:
            temporal_max_res = [32, 32, 16]
            
        self.spatial_bound = spatial_bound
        self.temporal_bound = temporal_bound
        
        # Store dimensions for output
        self.spatial_dim = spatial_levels * spatial_features
        self.temporal_dim = temporal_levels * temporal_features * 3  # 3 projections
        self.output_dim = self.spatial_dim + self.temporal_dim
        
        # Spatial encoder for xyz coordinates
        self.xyz_encoder = HashEncoder(
            input_dim=3,
            num_levels=spatial_levels,
            level_dim=spatial_features,
            per_level_scale=2,
            base_resolution=spatial_base_res,
            log2_hashmap_size=spatial_hashmap,
            desired_resolution=spatial_max_res
        )
        
        # Temporal projection encoders (xyt, yzt, xzt)
        self.xyt_encoder = HashEncoder(
            input_dim=3,
            num_levels=temporal_levels,
            level_dim=temporal_features,
            per_level_scale=2,
            base_resolution=temporal_base_res,
            log2_hashmap_size=temporal_hashmap,
            desired_resolution=temporal_max_res
        )
        
        self.yzt_encoder = HashEncoder(
            input_dim=3,
            num_levels=temporal_levels,
            level_dim=temporal_features,
            per_level_scale=2,
            base_resolution=temporal_base_res,
            log2_hashmap_size=temporal_hashmap,
            desired_resolution=temporal_max_res
        )
        
        self.xzt_encoder = HashEncoder(
            input_dim=3,
            num_levels=temporal_levels,
            level_dim=temporal_features,
            per_level_scale=2,
            base_resolution=temporal_base_res,
            log2_hashmap_size=temporal_hashmap,
            desired_resolution=temporal_max_res
        )
        
    def encode_spatial(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Encode spatial xyz coordinates.
        
        Args:
            xyz: Tensor of shape (..., 3) with normalized coordinates
            
        Returns:
            Spatial features of shape (..., spatial_dim)
        """
        return self.xyz_encoder(xyz, size=self.spatial_bound)
    
    def encode_temporal(self, xyzt: torch.Tensor) -> torch.Tensor:
        """
        Encode temporal projections (xyt, yzt, xzt).
        
        Args:
            xyzt: Tensor of shape (..., 4) with normalized coordinates
            
        Returns:
            Temporal features of shape (..., temporal_dim)
        """
        # Scale time dimension appropriately
        xyz_scaled = xyzt[..., :3]
        t_scaled = (xyzt[..., 3:] * 2 * self.temporal_bound - self.temporal_bound) * 0.9
        xyzt_scaled = torch.cat([xyz_scaled, t_scaled], dim=-1)
        
        # Create 3D projections with time
        xyt = torch.cat([xyzt_scaled[..., :2], xyzt_scaled[..., 3:]], dim=-1)
        yzt = xyzt_scaled[..., 1:]
        xzt = torch.cat([xyzt_scaled[..., :1], xyzt_scaled[..., 2:]], dim=-1)
        
        # Encode each projection
        xyt_features = self.xyt_encoder(xyt, size=self.temporal_bound)
        yzt_features = self.yzt_encoder(yzt, size=self.temporal_bound)
        xzt_features = self.xzt_encoder(xzt, size=self.temporal_bound)
        
        # Concatenate temporal features
        return torch.cat([xyt_features, yzt_features, xzt_features], dim=-1)
    
    def forward(self, xyzt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode 4D spatiotemporal coordinates.
        
        Args:
            xyzt: Tensor of shape (..., 4) with normalized coordinates
            
        Returns:
            Tuple of (spatial_features, temporal_features)
        """
        xyz = xyzt[..., :3]
        spatial_features = self.encode_spatial(xyz)
        temporal_features = self.encode_temporal(xyzt)
        return spatial_features, temporal_features


class Earth4D(nn.Module):
    """
    Earth4D: Complete Grid4D encoder for planetary spatiotemporal modeling.
    
    Main interface for Earth4D functionality with optional coordinate conversion,
    configurable multi-resolution scales, and flexible input handling.
    """
    
    def __init__(self,
                 # Core encoder configuration
                 spatial_levels: int = 16,
                 spatial_features: int = 2, 
                 spatial_base_res: int = 16,
                 spatial_max_res: int = 512,
                 temporal_levels: int = 16,
                 temporal_features: int = 2,
                 temporal_base_res: List[int] = None,
                 temporal_max_res: List[int] = None,
                 # Advanced features
                 auto_ecef_convert: bool = False,
                 spatial_scales_meters: Optional[List[float]] = None,
                 temporal_scales_seconds: Optional[List[float]] = None,
                 # Coordinate processing
                 earth_radius: float = 6371000.0,
                 spatial_bound: float = 1.0,
                 temporal_bound: float = 1.0,
                 return_separate_features: bool = True):
        """
        Initialize Earth4D encoder.
        
        Args:
            spatial_levels: Number of spatial hash levels
            spatial_features: Features per spatial level
            spatial_base_res: Base spatial resolution
            spatial_max_res: Maximum spatial resolution
            temporal_levels: Number of temporal hash levels
            temporal_features: Features per temporal level
            temporal_base_res: Base temporal resolution (list of 3)
            temporal_max_res: Max temporal resolution (list of 3)
            auto_ecef_convert: Automatically convert lat/lon/elev to ECEF
            spatial_scales_meters: Custom spatial scales in meters
            temporal_scales_seconds: Custom temporal scales in seconds
            earth_radius: Earth radius for ECEF conversion
            spatial_bound: Spatial coordinate normalization bound
            temporal_bound: Temporal coordinate normalization bound
            return_separate_features: Return separate spatial/temporal features
        """
        super().__init__()
        
        self.auto_ecef_convert = auto_ecef_convert
        self.return_separate_features = return_separate_features
        
        # Initialize coordinate converter if needed
        if auto_ecef_convert:
            self.converter = CoordinateConverter(
                earth_radius=earth_radius,
                time_origin=0.0,
                time_scale=1.0
            )
        else:
            self.converter = None
            
        # Handle custom scales (stretch goals)
        if spatial_scales_meters is not None or temporal_scales_seconds is not None:
            # Convert physical scales to hash encoder parameters
            if spatial_scales_meters is not None:
                spatial_levels = len(spatial_scales_meters)
                # Approximate resolution mapping (this would need refinement)
                spatial_base_res = int(min(spatial_scales_meters))
                spatial_max_res = int(max(spatial_scales_meters))
                
            if temporal_scales_seconds is not None:
                temporal_levels = len(temporal_scales_seconds)
                # Approximate temporal resolution mapping
                min_scale = min(temporal_scales_seconds)
                max_scale = max(temporal_scales_seconds)
                if temporal_base_res is None:
                    temporal_base_res = [int(min_scale // 3600) or 1] * 3  # hours
                if temporal_max_res is None:
                    temporal_max_res = [int(max_scale // 3600) or 1] * 3   # hours
        
        # Initialize core Grid4D encoder
        self.encoder = Grid4DSpatiotemporalEncoder(
            spatial_levels=spatial_levels,
            spatial_features=spatial_features,
            spatial_base_res=spatial_base_res,
            spatial_max_res=spatial_max_res,
            temporal_levels=temporal_levels,
            temporal_features=temporal_features,
            temporal_base_res=temporal_base_res,
            temporal_max_res=temporal_max_res,
            spatial_bound=spatial_bound,
            temporal_bound=temporal_bound
        )
        
        # Store for information
        self.spatial_scales_meters = spatial_scales_meters
        self.temporal_scales_seconds = temporal_scales_seconds
        
    def forward(self, coordinates: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through Earth4D encoder.
        
        Args:
            coordinates: Input coordinates. Shape (..., 4)
                        If auto_ecef_convert=True: (lat, lon, elevation, time)
                        If auto_ecef_convert=False: (x, y, z, t) normalized
        
        Returns:
            If return_separate_features=True: (spatial_features, temporal_features)
            If return_separate_features=False: concatenated_features
        """
        # Handle coordinate conversion if needed
        if self.auto_ecef_convert:
            if coordinates.shape[-1] != 4:
                raise ValueError("Expected 4 coordinates (lat, lon, elevation, time) for auto conversion")
            
            lat = coordinates[..., 0]
            lon = coordinates[..., 1] 
            elevation = coordinates[..., 2]
            time = coordinates[..., 3]
            
            # Convert to normalized ECEF
            coordinates = self.converter.process_raw_coordinates(
                lat, lon, elevation, time,
                self.encoder.spatial_bound,
                self.encoder.temporal_bound
            )
        
        # Encode with Grid4D
        spatial_features, temporal_features = self.encoder(coordinates)
        
        if self.return_separate_features:
            return spatial_features, temporal_features
        else:
            return torch.cat([spatial_features, temporal_features], dim=-1)
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """
        Get feature dimensions for the encoder.
        
        Returns:
            Dictionary with spatial, temporal, and total dimensions
        """
        return {
            'spatial': self.encoder.spatial_dim,
            'temporal': self.encoder.temporal_dim, 
            'total': self.encoder.output_dim
        }
    
    def get_configuration(self) -> Dict:
        """
        Get encoder configuration for reproducibility.
        
        Returns:
            Configuration dictionary
        """
        return {
            'auto_ecef_convert': self.auto_ecef_convert,
            'return_separate_features': self.return_separate_features,
            'spatial_scales_meters': self.spatial_scales_meters,
            'temporal_scales_seconds': self.temporal_scales_seconds,
            'encoder_config': {
                'spatial_levels': self.encoder.xyz_encoder.num_levels,
                'spatial_features': self.encoder.xyz_encoder.level_dim,
                'temporal_levels': self.encoder.xyt_encoder.num_levels,
                'temporal_features': self.encoder.xyt_encoder.level_dim,
                'spatial_bound': self.encoder.spatial_bound,
                'temporal_bound': self.encoder.temporal_bound
            }
        }


# Convenience functions for common use cases
def create_basic_earth4d(**kwargs) -> Earth4D:
    """
    Create a basic Earth4D encoder with default settings.
    
    Args:
        **kwargs: Additional arguments passed to Earth4D
        
    Returns:
        Earth4D encoder instance
    """
    return Earth4D(**kwargs)


def create_earth4d_with_physical_scales(
    spatial_scales_meters: List[float],
    temporal_scales_seconds: List[float],
    **kwargs
) -> Earth4D:
    """
    Create Earth4D encoder with physical scale specifications.
    
    Args:
        spatial_scales_meters: Spatial scales in meters
        temporal_scales_seconds: Temporal scales in seconds
        **kwargs: Additional arguments
        
    Returns:
        Earth4D encoder with custom scales
    """
    return Earth4D(
        spatial_scales_meters=spatial_scales_meters,
        temporal_scales_seconds=temporal_scales_seconds,
        **kwargs
    )


def create_earth4d_with_auto_conversion(**kwargs) -> Earth4D:
    """
    Create Earth4D encoder with automatic coordinate conversion.
    
    Args:
        **kwargs: Additional arguments
        
    Returns:
        Earth4D encoder with auto conversion enabled
    """
    return Earth4D(auto_ecef_convert=True, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    print("Earth4D: Grid4D Encoder for Planetary (X,Y,Z,T) Deep Learning")
    print("=" * 60)
    
    # Example 1: Basic usage with normalized coordinates
    print("\n1. Basic Earth4D Encoder:")
    encoder_basic = create_basic_earth4d()
    
    # Dummy normalized coordinates (batch_size=10, 4D coordinates)
    normalized_coords = torch.rand(10, 4)
    spatial_feat, temporal_feat = encoder_basic(normalized_coords)
    
    print(f"   Input shape: {normalized_coords.shape}")
    print(f"   Spatial features: {spatial_feat.shape}")
    print(f"   Temporal features: {temporal_feat.shape}")
    print(f"   Feature dims: {encoder_basic.get_feature_dimensions()}")
    
    # Example 2: With auto conversion (stretch goal)
    print("\n2. Earth4D with Auto ECEF Conversion:")
    try:
        encoder_auto = create_earth4d_with_auto_conversion()
        
        # Dummy geographic coordinates (lat, lon, elevation, time)
        # lat/lon in degrees, elevation in meters, time in seconds
        geo_coords = torch.tensor([
            [37.7749, -122.4194, 50.0, 1640995200.0],  # San Francisco
            [40.7128, -74.0060, 100.0, 1640995260.0],  # New York
            [51.5074, -0.1278, 25.0, 1640995320.0],    # London
        ])
        
        spatial_feat, temporal_feat = encoder_auto(geo_coords)
        print(f"   Geographic input: {geo_coords.shape}")
        print(f"   Spatial features: {spatial_feat.shape}")  
        print(f"   Temporal features: {temporal_feat.shape}")
        
    except Exception as e:
        print(f"   Auto conversion example failed: {e}")
    
    # Example 3: Custom physical scales (stretch goal)
    print("\n3. Earth4D with Custom Physical Scales:")
    try:
        encoder_scales = create_earth4d_with_physical_scales(
            spatial_scales_meters=[16, 32, 64, 128, 256, 512],  # meters
            temporal_scales_seconds=[3600, 86400, 604800, 2592000]  # hour, day, week, month
        )
        
        coords = torch.rand(5, 4)  
        features = encoder_scales(coords)
        print(f"   Custom scales input: {coords.shape}")
        if isinstance(features, tuple):
            print(f"   Spatial features: {features[0].shape}")
            print(f"   Temporal features: {features[1].shape}")
        else:
            print(f"   Combined features: {features.shape}")
            
    except Exception as e:
        print(f"   Custom scales example failed: {e}")
    
    print("\n" + "=" * 60)
    print("Earth4D encoder examples completed!")
    print("Ready for integration into DeepEarth framework.")