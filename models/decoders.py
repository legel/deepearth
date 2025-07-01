"""Decoder modules for DeepEarth model."""

import torch
import torch.nn as nn

from .configs import DeepEarthConfig


class ModalityDecoder(nn.Module):
    """Decoder for reconstructing modality data from embeddings.
    
    Uses a multi-layer MLP to transform embeddings back to the original
    modality space for self-supervised reconstruction.
    """
    
    def __init__(self, modality_name: str, output_dim: int, config: DeepEarthConfig):
        super().__init__()
        self.modality_name = modality_name
        self.output_dim = output_dim
        
        # Decoding MLP with residual connections
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim // 2, output_dim)
        )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Decode embeddings to modality data.
        
        Args:
            embeddings: (B, D) embeddings from fusion network
            
        Returns:
            reconstructions: (B, output_dim) reconstructed values
        """
        return self.decoder(embeddings)


class SpatiotemporalDecoder(nn.Module):
    """Decoder for spatiotemporal coordinates.
    
    Specialized decoder for reconstructing spatial (x,y,z) or temporal (t)
    coordinates from embeddings. Uses sigmoid activation to ensure
    outputs are in normalized range [0, 1].
    """
    
    def __init__(self, coord_type: str, output_dim: int, config: DeepEarthConfig):
        super().__init__()
        self.coord_type = coord_type
        self.output_dim = output_dim
        
        # Validate coord_type
        if coord_type not in ['spatial', 'temporal']:
            raise ValueError(f"coord_type must be 'spatial' or 'temporal', got {coord_type}")
        
        # Coordinate-specific decoder
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.LayerNorm(config.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, output_dim)
        )
        
        # Output activation for normalized coordinates
        self.output_activation = nn.Sigmoid()
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Decode embeddings to coordinates.
        
        Args:
            embeddings: (B, D) embeddings
            
        Returns:
            coords: (B, output_dim) normalized coordinates in [0, 1]
        """
        coords = self.decoder(embeddings)
        coords = self.output_activation(coords)
        return coords
