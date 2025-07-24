"""Encoder modules for DeepEarth model."""

import torch
import torch.nn as nn
from typing import Optional

from .configs import DeepEarthConfig, TransformerConfig
from .hash_encoding import MultiResolutionHashEncoding
from .transformers import Transformer


class Grid4DEncoder(nn.Module):
    """Grid4D encoder for spatiotemporal coordinates.
    
    Encodes (x, y, z, t) coordinates using multi-resolution hash encoding
    with separate treatment for spatial and temporal dimensions.
    """
    
    def __init__(self, config: DeepEarthConfig):
        super().__init__()
        self.config = config
        
        # Spatial encoding (x, y, z)
        self.spatial_encoder = MultiResolutionHashEncoding(
            n_levels=config.n_spatial_levels,
            n_features_per_level=config.n_features_per_level,
            resolutions=config.spatial_resolutions,
            hash_table_size=config.hash_table_size,
            coords_dim=3
        )
        
        # Temporal encoding (t)
        self.temporal_encoder = MultiResolutionHashEncoding(
            n_levels=config.n_temporal_levels,
            n_features_per_level=config.n_features_per_level,
            resolutions=config.temporal_resolutions,
            hash_table_size=config.hash_table_size // 4,  # Smaller table for 1D
            coords_dim=1
        )
        
        # Calculate total encoding dimension
        spatial_dim = config.n_spatial_levels * config.n_features_per_level
        temporal_dim = config.n_temporal_levels * config.n_features_per_level
        total_dim = spatial_dim + temporal_dim
        
        # Project to model dimension
        self.projection = nn.Sequential(
            nn.Linear(total_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
    
    def forward(self, xyzt: torch.Tensor, 
                spatial_mask: Optional[torch.Tensor] = None,
                temporal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode spatiotemporal coordinates.
        
        Args:
            xyzt: (B, 4) tensor of normalized coordinates
            spatial_mask: (B,) boolean mask for spatial coordinates
            temporal_mask: (B,) boolean mask for temporal coordinates
            
        Returns:
            embeddings: (B, D) coordinate embeddings
        """
        # Split spatial and temporal coordinates
        xyz = xyzt[:, :3]  # (B, 3)
        t = xyzt[:, 3:4]   # (B, 1)
        
        # Encode spatial coordinates
        spatial_features = self.spatial_encoder(xyz)
        if spatial_mask is not None:
            spatial_features = spatial_features * spatial_mask.unsqueeze(-1).float()
        
        # Encode temporal coordinates
        temporal_features = self.temporal_encoder(t)
        if temporal_mask is not None:
            temporal_features = temporal_features * temporal_mask.unsqueeze(-1).float()
        
        # Concatenate and project
        combined_features = torch.cat([spatial_features, temporal_features], dim=-1)
        embeddings = self.projection(combined_features)
        
        return embeddings


class ModalityEncoder(nn.Module):
    """Encoder for a specific data modality.
    
    Uses a small Transformer to process modality-specific features
    and produce a fixed-size embedding.
    """
    
    def __init__(self, modality_name: str, input_dim: int, 
                 config: DeepEarthConfig, encoder_config: TransformerConfig):
        super().__init__()
        self.modality_name = modality_name
        self.input_dim = input_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, encoder_config.hidden_dim)
        
        # Modality embedding - learnable token that identifies this modality
        self.modality_embedding = nn.Parameter(
            torch.randn(1, 1, encoder_config.hidden_dim) * 0.02
        )
        
        # Transformer encoder
        self.transformer = Transformer(encoder_config)
        
        # Output projection to main model dimension
        self.output_projection = nn.Linear(encoder_config.hidden_dim, config.hidden_dim)
        
        # Layer norm
        self.norm = nn.LayerNorm(config.hidden_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode modality data.
        
        Args:
            x: (B, D) modality features
            mask: (B,) boolean mask
            
        Returns:
            embeddings: (B, D) modality embeddings
        """
        B = x.shape[0]
        
        # Project input
        x = self.input_projection(x).unsqueeze(1)  # (B, 1, D)
        
        # Add modality embedding
        x = x + self.modality_embedding.expand(B, -1, -1)
        
        # Apply transformer
        x = self.transformer(x, mask=mask)
        
        # Take the single token output
        x = x[:, 0]
        
        # Project to main model dimension
        x = self.output_projection(x)
        x = self.norm(x)
        
        return x
