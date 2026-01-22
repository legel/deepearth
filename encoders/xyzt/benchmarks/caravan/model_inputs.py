"""
Earth4D with meteorological input features.

Architecture (based on Lance's guidance):
    raw (x,y,z,t) → Earth4D → 192D embedding_xyzt
    raw temperature → MLP1 (2 layers × 64) → 32D embedding_temp
    raw precipitation → MLP2 (2 layers × 64) → 32D embedding_precip
    raw snow → MLP3 (2 layers × 64) → 32D embedding_snow

    Then: MLP5(192D + 32D + 32D + 32D + basin_embedding) → streamflow

Key insight: "combining raw data with deep embeddings is often not a good idea"
Solution: Pass each raw feature through a small MLP before concatenation
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import sys
sys.path.append('../..')
from earth4d import Earth4D


class FeatureEmbeddingMLP(nn.Module):
    """
    Small MLP to embed a single scalar feature before combining with Earth4D.

    Args:
        output_dim: Dimension of output embedding (default 32)
    """
    def __init__(self, output_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size,) or (batch_size, 1) - single feature values

        Returns:
            (batch_size, output_dim) embeddings
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # (batch_size,) -> (batch_size, 1)
        return self.mlp(x)


class Earth4DWithInputs(nn.Module):
    """
    Earth4D with meteorological input features (precipitation, temperature, snow).

    Implements multi-modal fusion:
    - Spatiotemporal embeddings from Earth4D (x,y,z,t)
    - Feature-specific embeddings from small MLPs (P, T, Snow)
    - Basin embeddings (learnable per-basin features)
    - All concatenated and passed through prediction head

    Args:
        basin_dim: Dimension of learnable basin embeddings
        num_basins: Number of unique basins
        feature_embedding_dim: Dimension of feature embeddings (P, T, Snow)
        coordinate_system: 'ecef' or 'latlon' coordinate system
    """

    def __init__(
        self,
        basin_dim: int = 256,
        num_basins: int = 144,
        feature_embedding_dim: int = 32,
        coordinate_system: str = 'ecef',
    ):
        super().__init__()

        # Earth4D encoder for (x,y,z,t) coordinates
        # Use default Earth4D settings (same as baseline model)
        self.earth4d = Earth4D(
            coordinate_system=coordinate_system,
        )
        earth4d_dim = self.earth4d.output_dim  # 192 by default

        # Learnable basin embeddings
        self.basin_embeddings = nn.Embedding(num_basins, basin_dim)

        # Feature-specific MLPs (Lance's architecture)
        self.precipitation_mlp = FeatureEmbeddingMLP(feature_embedding_dim)
        self.temperature_mlp = FeatureEmbeddingMLP(feature_embedding_dim)
        self.snow_mlp = FeatureEmbeddingMLP(feature_embedding_dim)

        # Prediction head
        # Input: 192 (Earth4D) + 32 (precip) + 32 (temp) + 32 (snow) + 256 (basin) = 544
        combined_dim = earth4d_dim + 3 * feature_embedding_dim + basin_dim
        self.prediction_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        print(f"\nEarth4D with Input Features:")
        print(f"  Earth4D embeddings: {earth4d_dim}D")
        print(f"  Precipitation embedding: {feature_embedding_dim}D")
        print(f"  Temperature embedding: {feature_embedding_dim}D")
        print(f"  Snow embedding: {feature_embedding_dim}D")
        print(f"  Basin embeddings: {basin_dim}D")
        print(f"  Combined dimension: {combined_dim}D")
        print(f"  Output: streamflow (1D)")

    def forward(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch_data: Dictionary containing:
                - coords: (batch_size, 4) - (x, y, z, t) in ECEF or lat/lon/elev/time
                - precipitation: (batch_size,) - precipitation in mm/day
                - temperature: (batch_size,) - temperature in °C
                - snow: (batch_size,) - snow depth water equivalent in mm
                - basin_idx: (batch_size,) - basin indices

        Returns:
            (batch_size,) streamflow predictions
        """
        # Extract inputs
        coords = batch_data['coords']  # (batch_size, 4)
        precip = batch_data['precipitation']  # (batch_size,)
        temp = batch_data['temperature']  # (batch_size,)
        snow = batch_data['snow']  # (batch_size,)
        basin_idx = batch_data['basin_idx']  # (batch_size,)

        # Encode coordinates with Earth4D
        earth4d_embedding = self.earth4d(coords)  # (batch_size, 192)

        # Encode each feature with its own MLP
        precip_embedding = self.precipitation_mlp(precip)  # (batch_size, 32)
        temp_embedding = self.temperature_mlp(temp)  # (batch_size, 32)
        snow_embedding = self.snow_mlp(snow)  # (batch_size, 32)

        # Get basin embeddings
        basin_embedding = self.basin_embeddings(basin_idx)  # (batch_size, 256)

        # Concatenate all embeddings
        combined = torch.cat([
            earth4d_embedding,
            precip_embedding,
            temp_embedding,
            snow_embedding,
            basin_embedding,
        ], dim=-1)  # (batch_size, 544)

        # Predict streamflow
        streamflow = self.prediction_head(combined).squeeze(-1)  # (batch_size,)

        return streamflow

    def forward_precomputed(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass with precomputed Earth4D embeddings (for efficiency).

        Args:
            batch_data: Dictionary containing:
                - earth4d_embeddings: (batch_size, 192) - precomputed embeddings
                - precipitation: (batch_size,) - precipitation in mm/day
                - temperature: (batch_size,) - temperature in °C
                - snow: (batch_size,) - snow depth water equivalent in mm
                - basin_idx: (batch_size,) - basin indices

        Returns:
            (batch_size,) streamflow predictions
        """
        # Extract inputs
        earth4d_embedding = batch_data['earth4d_embeddings']  # (batch_size, 192)
        precip = batch_data['precipitation']  # (batch_size,)
        temp = batch_data['temperature']  # (batch_size,)
        snow = batch_data['snow']  # (batch_size,)
        basin_idx = batch_data['basin_idx']  # (batch_size,)

        # Encode each feature with its own MLP
        precip_embedding = self.precipitation_mlp(precip)  # (batch_size, 32)
        temp_embedding = self.temperature_mlp(temp)  # (batch_size, 32)
        snow_embedding = self.snow_mlp(snow)  # (batch_size, 32)

        # Get basin embeddings
        basin_embedding = self.basin_embeddings(basin_idx)  # (batch_size, 256)

        # Concatenate all embeddings
        combined = torch.cat([
            earth4d_embedding,
            precip_embedding,
            temp_embedding,
            snow_embedding,
            basin_embedding,
        ], dim=-1)  # (batch_size, 544)

        # Predict streamflow
        streamflow = self.prediction_head(combined).squeeze(-1)  # (batch_size,)

        return streamflow
