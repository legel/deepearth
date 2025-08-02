"""
Data Source Registry for DeepEarth
Seamless API for adding new data sources with automatic encoder/decoder creation
"""

import torch
import torch.nn as nn
from typing import Dict, List, Union, Optional, Callable, Any
from dataclasses import dataclass
import numpy as np
from pathlib import Path

from encoders.universal_encoder import UniversalProjector, EncoderConfig


@dataclass
class DataSourceConfig:
    """Configuration for a data source"""
    name: str
    input_type: str  # 'image', 'vector', 'raster', 'point_cloud', 'time_series'
    input_shape: Union[tuple, dict]  # Expected input shape or shape dict
    native_encoder: str = "vjepa2"  # Which pretrained encoder to use
    num_tokens: int = 1  # Number of universal tokens to generate
    preprocessing: Optional[Callable] = None  # Custom preprocessing function
    metadata: Dict[str, Any] = None  # Additional metadata
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DatasetSpecificEncoder(nn.Module):
    """Lightweight encoder that adapts data to pretrained backbone format"""
    
    def __init__(self, config: DataSourceConfig, universal_dim: int = 2048):
        super().__init__()
        self.config = config
        self.universal_dim = universal_dim
        
        # Create input adapter based on data type
        if config.input_type == 'image':
            self.adapter = self._create_image_adapter()
        elif config.input_type == 'vector':
            self.adapter = self._create_vector_adapter()
        elif config.input_type == 'raster':
            self.adapter = self._create_raster_adapter()
        elif config.input_type == 'time_series':
            self.adapter = self._create_timeseries_adapter()
        else:
            self.adapter = nn.Identity()
    
    def _create_image_adapter(self):
        """Adapt arbitrary image data to V-JEPA expected format"""
        # V-JEPA expects 3x224x224 RGB images
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((224, 224)) if self.config.input_shape[-2:] != (224, 224) else nn.Identity(),
            nn.Conv2d(
                self.config.input_shape[0],  # Input channels
                3,  # RGB output
                kernel_size=1
            ) if self.config.input_shape[0] != 3 else nn.Identity()
        )
    
    def _create_vector_adapter(self):
        """Adapt vector data (e.g., tabular) to image-like format"""
        # Convert vector to 2D feature map
        vector_dim = self.config.input_shape[0]
        spatial_size = int(np.sqrt(vector_dim))
        if spatial_size * spatial_size < vector_dim:
            spatial_size += 1
        
        return VectorToImageAdapter(vector_dim, spatial_size)
    
    def _create_raster_adapter(self):
        """Adapt GIS raster layers to image format"""
        # Handle multi-band rasters (e.g., hyperspectral, SAR)
        return nn.Sequential(
            nn.Conv2d(
                self.config.input_shape[0],  # Number of bands
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 3, kernel_size=1),  # Project to RGB
            nn.AdaptiveAvgPool2d((224, 224))
        )
    
    def _create_timeseries_adapter(self):
        """Adapt time series to image-like format"""
        return TimeSeriesImageAdapter(
            seq_length=self.config.input_shape[0],
            num_features=self.config.input_shape[1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adapt input data to format expected by pretrained encoder"""
        if self.config.preprocessing is not None:
            x = self.config.preprocessing(x)
        
        return self.adapter(x)


class DatasetSpecificDecoder(nn.Module):
    """Lightweight decoder for dataset-specific reconstruction"""
    
    def __init__(self, config: DataSourceConfig, native_dim: int, universal_dim: int = 2048):
        super().__init__()
        self.config = config
        
        # Decode from native embeddings back to original space
        if config.input_type == 'vector':
            output_dim = config.input_shape[0]
        elif config.input_type == 'image' or config.input_type == 'raster':
            output_dim = np.prod(config.input_shape)
        else:
            output_dim = np.prod(config.input_shape)
        
        self.decoder = nn.Sequential(
            nn.Linear(native_dim, native_dim // 2),
            nn.LayerNorm(native_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(native_dim // 2, output_dim)
        )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Decode embeddings back to original data space"""
        decoded = self.decoder(embeddings)
        
        # Reshape if needed
        if self.config.input_type in ['image', 'raster']:
            B = decoded.shape[0]
            decoded = decoded.reshape(B, *self.config.input_shape)
        
        return decoded


class VectorToImageAdapter(nn.Module):
    """Convert vector data to pseudo-image format"""
    
    def __init__(self, vector_dim: int, spatial_size: int):
        super().__init__()
        self.vector_dim = vector_dim
        self.spatial_size = spatial_size
        self.padding = spatial_size * spatial_size - vector_dim
        
        # Learn spatial arrangement
        self.spatial_embed = nn.Sequential(
            nn.Linear(vector_dim, spatial_size * spatial_size * 16),
            nn.GELU(),
            nn.Unflatten(1, (16, spatial_size, spatial_size)),
            nn.Conv2d(16, 3, kernel_size=1),
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert (B, D) vector to (B, 3, 224, 224) image"""
        return self.spatial_embed(x)


class TimeSeriesImageAdapter(nn.Module):
    """Convert time series to image representation"""
    
    def __init__(self, seq_length: int, num_features: int):
        super().__init__()
        self.seq_length = seq_length
        self.num_features = num_features
        
        # Create 2D representation of time series
        self.adapter = nn.Sequential(
            nn.Unflatten(1, (1, seq_length, num_features)),
            nn.Conv2d(1, 32, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.AdaptiveAvgPool2d((224, 224))
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert (B, L, F) time series to (B, 3, 224, 224) image"""
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # Add feature dimension if needed
        x = x.transpose(1, 2)  # (B, F, L)
        x = x.unsqueeze(1)  # (B, 1, F, L)
        return self.adapter(x)


class DataSourceRegistry:
    """Central registry for all data sources in DeepEarth"""
    
    def __init__(self, model):
        self.model = model
        self.data_sources = {}
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        self.projectors = nn.ModuleDict()
        
        # Get native encoder dimensions
        self.native_dims = {
            'vjepa2': 768,  # V-JEPA base
            'deepseek': 4096,  # DeepSeek
            'direct': 2048  # Direct projection without backbone
        }
    
    def register_data_source(
        self,
        name: str,
        input_type: str,
        input_shape: Union[tuple, dict],
        native_encoder: str = "vjepa2",
        num_tokens: int = 1,
        preprocessing: Optional[Callable] = None,
        **kwargs
    ) -> None:
        """
        Register a new data source with automatic encoder/decoder creation
        
        Args:
            name: Unique name for this data source
            input_type: Type of data ('image', 'vector', 'raster', 'time_series')
            input_shape: Expected shape of input data
            native_encoder: Which pretrained encoder to use ('vjepa2', 'deepseek', 'direct')
            num_tokens: Number of universal tokens to generate
            preprocessing: Optional preprocessing function
            **kwargs: Additional metadata
        
        Example:
            >>> registry.register_data_source(
            ...     name="soil_chemistry",
            ...     input_type="vector",
            ...     input_shape=(15,),  # 15 chemical properties
            ...     native_encoder="vjepa2"
            ... )
        """
        # Create configuration
        config = DataSourceConfig(
            name=name,
            input_type=input_type,
            input_shape=input_shape,
            native_encoder=native_encoder,
            num_tokens=num_tokens,
            preprocessing=preprocessing,
            metadata=kwargs
        )
        
        # Store configuration
        self.data_sources[name] = config
        
        # Create dataset-specific encoder (adapter)
        self.encoders[name] = DatasetSpecificEncoder(config)
        
        # Create dataset-specific decoder
        native_dim = self.native_dims[native_encoder]
        self.decoders[name] = DatasetSpecificDecoder(config, native_dim)
        
        # Create projector to universal space
        projector_config = EncoderConfig(
            name=name,
            native_dim=native_dim,
            universal_dim=self.model.config.universal_dim,
            num_tokens_per_sample=num_tokens,
            projection_type="attention" if num_tokens > 1 else "mlp"
        )
        self.projectors[name] = UniversalProjector(projector_config)
        
        # Register with model
        self._integrate_with_model(name, config)
        
        print(f"✓ Registered data source '{name}' ({input_type}, shape={input_shape})")
    
    def _integrate_with_model(self, name: str, config: DataSourceConfig):
        """Integrate the new data source with the model"""
        # Add to model's fusion module
        if hasattr(self.model.fusion, 'st_embedding'):
            self.model.fusion.st_embedding.add_modality(name)
        
        # Add to model's additional encoders
        self.model.additional_encoders[name] = self.encoders[name]
        self.model.additional_projectors[name] = self.projectors[name]
        
        # Add decoder to universal decoder if it exists
        if hasattr(self.model, 'universal_decoder'):
            self.model.universal_decoder.decoders[name] = self.decoders[name]
    
    def process_data(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Process data from multiple sources through their adapters
        
        Args:
            data: Dict mapping data source names to tensors
            
        Returns:
            Dict of adapted data ready for the model
        """
        adapted_data = {}
        
        for name, tensor in data.items():
            if name in self.data_sources:
                config = self.data_sources[name]
                
                if config.native_encoder == 'vjepa2':
                    # Adapt to image format for V-JEPA
                    adapted = self.encoders[name](tensor)
                    adapted_data[name] = adapted
                elif config.native_encoder == 'direct':
                    # Use directly without backbone
                    adapted_data[name] = tensor
                else:
                    adapted_data[name] = tensor
            else:
                # Pass through unchanged if not registered
                adapted_data[name] = tensor
        
        return adapted_data
    
    def get_native_embeddings(self, name: str, data: torch.Tensor) -> torch.Tensor:
        """Get native embeddings for a specific data source"""
        config = self.data_sources[name]
        
        if config.native_encoder == 'vjepa2':
            # Adapt data
            adapted = self.encoders[name](data)
            
            # Pass through V-JEPA
            with torch.no_grad():
                vision_encoder = self.model.universal_encoder.encoders['vision']
                embeddings = vision_encoder.extract_native_embeddings(adapted)
                return embeddings['global_embedding']
        
        elif config.native_encoder == 'direct':
            # Project directly without backbone
            return self.encoders[name](data)
        
        else:
            raise ValueError(f"Unknown encoder: {config.native_encoder}")
    
    def list_sources(self) -> List[str]:
        """List all registered data sources"""
        return list(self.data_sources.keys())
    
    def get_info(self, name: str) -> Dict[str, Any]:
        """Get information about a data source"""
        if name not in self.data_sources:
            raise ValueError(f"Data source '{name}' not found")
        
        config = self.data_sources[name]
        return {
            'name': config.name,
            'input_type': config.input_type,
            'input_shape': config.input_shape,
            'native_encoder': config.native_encoder,
            'num_tokens': config.num_tokens,
            'metadata': config.metadata
        }


# Convenience function for model integration
def create_deepearth_with_registry():
    """Create DeepEarth model with data source registry"""
    from models.deepearth_integrated import create_integrated_deepearth
    
    # Create base model
    model = create_integrated_deepearth(
        freeze_backbones=True  # Always freeze pretrained weights
    )
    
    # Create registry
    registry = DataSourceRegistry(model)
    
    # Attach registry to model for easy access
    model.data_registry = registry
    
    # Override forward to handle registered data sources
    original_forward = model.forward
    
    def forward_with_registry(
        xyzt: torch.Tensor,
        vision_input: Optional[torch.Tensor] = None,
        language_input: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        additional_modalities: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ):
        # Process registered data sources
        if additional_modalities is not None:
            additional_modalities = registry.process_data(additional_modalities)
        
        return original_forward(
            xyzt=xyzt,
            vision_input=vision_input,
            language_input=language_input,
            additional_modalities=additional_modalities,
            **kwargs
        )
    
    model.forward = forward_with_registry
    
    return model


# Example usage
if __name__ == "__main__":
    # Create model with registry
    model = create_deepearth_with_registry()
    
    # Register various data sources with ONE LINE each!
    
    # Soil chemistry data (tabular)
    model.data_registry.register_data_source(
        name="soil_chemistry",
        input_type="vector",
        input_shape=(15,),  # pH, N, P, K, organic matter, etc.
    )
    
    # Hyperspectral imagery
    model.data_registry.register_data_source(
        name="hyperspectral",
        input_type="raster",
        input_shape=(224, 10, 10),  # 224 spectral bands, 10x10 spatial
        num_tokens=4  # Generate multiple tokens for rich spectral data
    )
    
    # SAR (Synthetic Aperture Radar)
    model.data_registry.register_data_source(
        name="sar_vv_vh",
        input_type="raster",
        input_shape=(2, 256, 256),  # VV and VH polarizations
    )
    
    # Weather time series
    model.data_registry.register_data_source(
        name="weather_forecast",
        input_type="time_series",
        input_shape=(168, 5),  # 7 days hourly, 5 variables
    )
    
    # Elevation data
    model.data_registry.register_data_source(
        name="elevation",
        input_type="raster",
        input_shape=(1, 100, 100),  # Single band DEM
    )
    
    # Land use classification map
    model.data_registry.register_data_source(
        name="land_use",
        input_type="raster",
        input_shape=(10, 50, 50),  # 10 land use classes as channels
    )
    
    # Species occurrence data (sparse vector)
    model.data_registry.register_data_source(
        name="species_observations",
        input_type="vector",
        input_shape=(500,),  # 500 species presence/absence
        preprocessing=lambda x: torch.sigmoid(x)  # Convert to probabilities
    )
    
    # Test forward pass with multiple data sources
    batch_size = 2
    test_data = {
        'xyzt': torch.randn(batch_size, 4),
        'vision_input': torch.randn(batch_size, 3, 224, 224),  # Regular RGB
        'additional_modalities': {
            'soil_chemistry': torch.randn(batch_size, 15),
            'hyperspectral': torch.randn(batch_size, 224, 10, 10),
            'weather_forecast': torch.randn(batch_size, 168, 5),
            'elevation': torch.randn(batch_size, 1, 100, 100)
        }
    }
    
    # Forward pass - all data sources automatically handled!
    with torch.no_grad():
        outputs = model(**test_data)
    
    print(f"\nModel output shape: {outputs['fused_representation'].shape}")
    print(f"Registered data sources: {model.data_registry.list_sources()}")
