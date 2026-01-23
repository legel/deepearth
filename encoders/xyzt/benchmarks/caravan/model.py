"""
Streamflow prediction model using Earth4D positional encoding.

Analogous to the LFMC model but for hydrology.
"""

import torch
import torch.nn as nn
from typing import Dict

# Import Earth4D from parent package
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from earth4d import Earth4D


class StreamflowModel(nn.Module):
    """
    Streamflow prediction model with learnable basin embeddings.

    Implements TrainableModel protocol with dict-based forward methods.
    Combines Earth4D spatiotemporal features with learned basin embeddings.

    Architecture is analogous to SpeciesAwareLFMCModel:
    - Earth4D encodes (x,y,z,t) → spatial-temporal features
    - Basin embedding encodes basin identity → learned characteristics
    - MLP combines both → streamflow prediction
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
    ):
        """
        Initialize streamflow model.

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

        # Learnable basin embeddings (analogous to species embeddings in LFMC)
        self.basin_embeddings = nn.Embedding(n_basins, basin_dim)
        nn.init.normal_(self.basin_embeddings.weight, mean=0.0, std=0.1)

        if verbose:
            print(f"  Using learnable basin embeddings: ({n_basins}, {basin_dim})", flush=True)

        # MLP that combines Earth4D spatiotemporal features with basin characteristics
        input_dim = earth4d_dim + basin_dim

        if verbose:
            print(f"  MLP input dimension: {input_dim} (Earth4D: {earth4d_dim} + Basin: {basin_dim})", flush=True)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output in [0, 1] to match normalized targets
        )

        # Initialize Earth4D parameters
        with torch.no_grad():
            for p in self.earth4d.parameters():
                if p.dim() > 1:
                    nn.init.uniform_(p, -0.1, 0.1)

        self.n_basins = n_basins
        self.basin_dim = basin_dim

    def forward(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass using coordinate data.

        Implements TrainableModel protocol.

        Args:
            batch_data: Dictionary containing:
                - 'coords': (B, 4) tensor of [lat, lon, elev, time]
                - 'basin_idx': (B,) tensor of basin indices

        Returns:
            (B,) tensor of predicted streamflow values (normalized [0, 1])
        """
        coords = batch_data['coords']
        basin_idx = batch_data['basin_idx']

        # Get Earth4D spatiotemporal features
        earth4d_features = self.earth4d(coords)

        # Get basin embeddings
        basin_features = self.basin_embeddings(basin_idx)

        # Concatenate features
        combined_features = torch.cat([earth4d_features, basin_features], dim=-1)

        # Predict streamflow
        return self.mlp(combined_features).squeeze(-1)

    def forward_precomputed(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass using precomputed hash indices.

        Args:
            batch_data: Dictionary containing:
                - 'indices': (B,) sample indices into precomputed buffers
                - 'basin_idx': (B,) tensor of basin indices

        Returns:
            (B,) tensor of predicted streamflow values (normalized [0, 1])
        """
        batch_indices = batch_data['indices']
        basin_idx = batch_data['basin_idx']

        # Get Earth4D features using precomputed indices
        earth4d_features = self.earth4d.forward_precomputed(batch_indices)

        # Get basin embeddings
        basin_features = self.basin_embeddings(basin_idx)

        # Concatenate features
        combined_features = torch.cat([earth4d_features, basin_features], dim=-1)

        # Predict streamflow
        return self.mlp(combined_features).squeeze(-1)
