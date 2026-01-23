"""
Multi-task streamflow prediction model using Earth4D positional encoding.

Forces Earth4D to reconstruct multiple hydrological modalities simultaneously:
- Primary task: streamflow prediction
- Auxiliary task 1: precipitation prediction
- Auxiliary task 2: temperature prediction

This multi-task learning approach provides regularization and forces the model
to learn physical processes rather than memorizing patterns.
"""

import torch
import torch.nn as nn
from typing import Dict

# Import Earth4D from parent package
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from earth4d import Earth4D


class StreamflowMultitaskModel(nn.Module):
    """
    Multi-task streamflow prediction model with learnable basin embeddings.

    Implements multi-task learning by predicting streamflow, precipitation, and temperature
    from coordinates only. This forces Earth4D to learn spatiotemporal patterns
    that generalize better than single-task models.

    Architecture:
    - Earth4D encodes (x,y,z,t) → spatiotemporal features
    - Basin embedding encodes basin identity → learned characteristics
    - Shared trunk combines both → rich representation
    - Multiple prediction heads → streamflow, precipitation, temperature
    """

    def __init__(
        self,
        n_basins: int,
        basin_dim: int = 256,
        use_adaptive_range: bool = False,
        verbose: bool = True,
        coordinate_system: str = 'ecef',
        resolution_mode: str = 'balanced',
        base_temporal_resolution: float = 32.0,
        temporal_growth_factor: float = 2.0,
        latlon_growth_factor: float = None,
        elev_growth_factor: float = None,
        shared_trunk_dim: int = 256,
    ):
        """
        Initialize multi-task streamflow model.

        Args:
            n_basins: Number of unique basins in the dataset
            basin_dim: Dimension of basin embedding vectors
            use_adaptive_range: Whether to use adaptive range for Earth4D
            verbose: Print model architecture details
            coordinate_system: 'geographic' or 'ecef'
            resolution_mode: Resolution scaling mode
            base_temporal_resolution: Base temporal resolution
            temporal_growth_factor: Growth factor for temporal levels (1.2 optimal)
            latlon_growth_factor: Growth factor for lat/lon dimensions
            elev_growth_factor: Growth factor for elevation dimension
            shared_trunk_dim: Dimension of shared trunk before task heads
        """
        super().__init__()

        self.earth4d = Earth4D(
            verbose=verbose,
            use_adaptive_range=use_adaptive_range,
            coordinate_system=coordinate_system,
            resolution_mode=resolution_mode,
            base_temporal_resolution=base_temporal_resolution,
            temporal_growth_factor=temporal_growth_factor,
            latlon_growth_factor=latlon_growth_factor,
            elev_growth_factor=elev_growth_factor,
        )

        earth4d_dim = self.earth4d.get_output_dim()

        # Learnable basin embeddings
        self.basin_embeddings = nn.Embedding(n_basins, basin_dim)
        nn.init.normal_(self.basin_embeddings.weight, mean=0.0, std=0.1)

        if verbose:
            print(f"  Using learnable basin embeddings: ({n_basins}, {basin_dim})", flush=True)

        # Shared trunk that combines Earth4D spatiotemporal features with basin characteristics
        input_dim = earth4d_dim + basin_dim

        if verbose:
            print(f"  Shared trunk input dimension: {input_dim} (Earth4D: {earth4d_dim} + Basin: {basin_dim})", flush=True)
            print(f"  Multi-task architecture with 3 prediction heads", flush=True)

        # Shared representation learning
        self.shared_trunk = nn.Sequential(
            nn.Linear(input_dim, shared_trunk_dim),
            nn.ReLU(inplace=True),
            nn.Linear(shared_trunk_dim, shared_trunk_dim),
            nn.ReLU(inplace=True),
        )

        # Task-specific prediction heads
        # Head 1: Streamflow prediction (primary task)
        self.streamflow_head = nn.Sequential(
            nn.Linear(shared_trunk_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output in [0, 1] to match normalized targets
        )

        # Head 2: Precipitation prediction (auxiliary task for regularization)
        self.precipitation_head = nn.Sequential(
            nn.Linear(shared_trunk_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output in [0, 1] to match normalized targets
        )

        # Head 3: Temperature prediction (auxiliary task for regularization)
        self.temperature_head = nn.Sequential(
            nn.Linear(shared_trunk_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            # No sigmoid - temperature can be z-scored or normalized differently
        )

        # Initialize Earth4D parameters
        with torch.no_grad():
            for p in self.earth4d.parameters():
                if p.dim() > 1:
                    nn.init.uniform_(p, -0.1, 0.1)

        self.n_basins = n_basins
        self.basin_dim = basin_dim
        self.shared_trunk_dim = shared_trunk_dim

    def forward(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass using coordinate data.

        Multi-task forward returns predictions for all tasks.

        Args:
            batch_data: Dictionary containing:
                - 'coords': (B, 4) tensor of [lat, lon, elev, time]
                - 'basin_idx': (B,) tensor of basin indices

        Returns:
            Dictionary with predictions:
                - 'streamflow': (B,) tensor of predicted streamflow (normalized [0, 1])
                - 'precipitation': (B,) tensor of predicted precipitation (normalized [0, 1])
        """
        coords = batch_data['coords']
        basin_idx = batch_data['basin_idx']

        # Get Earth4D spatiotemporal features
        earth4d_features = self.earth4d(coords)

        # Get basin embeddings
        basin_features = self.basin_embeddings(basin_idx)

        # Concatenate features
        combined_features = torch.cat([earth4d_features, basin_features], dim=-1)

        # Shared representation
        shared_repr = self.shared_trunk(combined_features)

        # Task-specific predictions
        streamflow_pred = self.streamflow_head(shared_repr).squeeze(-1)
        precipitation_pred = self.precipitation_head(shared_repr).squeeze(-1)
        temperature_pred = self.temperature_head(shared_repr).squeeze(-1)

        return {
            'streamflow': streamflow_pred,
            'precipitation': precipitation_pred,
            'evapotranspiration': temperature_pred,
        }

    def forward_precomputed(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass using precomputed hash indices.

        Args:
            batch_data: Dictionary containing:
                - 'indices': (B,) sample indices into precomputed buffers
                - 'basin_idx': (B,) tensor of basin indices

        Returns:
            Dictionary with predictions:
                - 'streamflow': (B,) tensor of predicted streamflow (normalized [0, 1])
                - 'precipitation': (B,) tensor of predicted precipitation (normalized [0, 1])
        """
        batch_indices = batch_data['indices']
        basin_idx = batch_data['basin_idx']

        # Get Earth4D features using precomputed indices
        earth4d_features = self.earth4d.forward_precomputed(batch_indices)

        # Get basin embeddings
        basin_features = self.basin_embeddings(basin_idx)

        # Concatenate features
        combined_features = torch.cat([earth4d_features, basin_features], dim=-1)

        # Shared representation
        shared_repr = self.shared_trunk(combined_features)

        # Task-specific predictions
        streamflow_pred = self.streamflow_head(shared_repr).squeeze(-1)
        precipitation_pred = self.precipitation_head(shared_repr).squeeze(-1)
        temperature_pred = self.temperature_head(shared_repr).squeeze(-1)

        return {
            'streamflow': streamflow_pred,
            'precipitation': precipitation_pred,
            'evapotranspiration': temperature_pred,
        }
