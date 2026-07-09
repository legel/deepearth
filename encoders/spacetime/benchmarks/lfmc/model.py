"""
LFMC Model with species-aware Earth4D positional encoding.
"""

import torch
import torch.nn as nn
from typing import Dict

# Import Earth4D from parent package
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from earth4d import Earth4D


class SpeciesAwareLFMCModel(nn.Module):
    """
    LFMC model with learnable species embeddings.

    Implements TrainableModel protocol with dict-based forward methods.
    Combines Earth4D spatiotemporal features with learned species embeddings.
    """

    def __init__(self, n_species: int, species_dim: int = 32,
                 use_adaptive_range: bool = False, verbose: bool = True,
                 coordinate_system: str = 'ecef',
                 resolution_mode: str = 'balanced',
                 base_temporal_resolution: float = 32.0,
                 temporal_growth_factor: float = None,
                 latlon_growth_factor: float = None,
                 elev_growth_factor: float = None):
        """
        Initialize species-aware LFMC model.

        Args:
            n_species: Number of unique species in the dataset
            species_dim: Dimension of species embedding vectors
            use_adaptive_range: Whether to use adaptive range for Earth4D
            verbose: Print model architecture details
            coordinate_system: 'geographic' or 'ecef'
            resolution_mode: Resolution mode for geographic coordinates
            base_temporal_resolution: Base resolution for temporal encoder
            temporal_growth_factor: Growth factor for temporal resolution levels
            latlon_growth_factor: Growth factor for lat/lon dimensions (decoupled from elevation)
            elev_growth_factor: Growth factor for elevation dimension (decoupled from lat/lon)
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

        # Learnable species embeddings
        self.species_embeddings = nn.Embedding(n_species, species_dim)
        nn.init.normal_(self.species_embeddings.weight, mean=0.0, std=0.1)
        if verbose:
            print(f"  Using learnable species embeddings: ({n_species}, {species_dim})", flush=True)

        # MLP that takes concatenated Earth4D features and species embedding
        input_dim = earth4d_dim + species_dim
        if verbose:
            print(f"  MLP input dimension: {input_dim} (Earth4D: {earth4d_dim} + Species: {species_dim})", flush=True)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Constrain output to [0, 1] to match normalized targets
        )

        # Initialize Earth4D parameters
        with torch.no_grad():
            for p in self.earth4d.parameters():
                if p.dim() > 1:
                    nn.init.uniform_(p, -0.1, 0.1)

        self.n_species = n_species
        self.species_dim = species_dim

    def forward(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass using coordinate data.

        Implements TrainableModel protocol.

        Args:
            batch_data: Dictionary containing:
                - 'coords': (B, 4) tensor of [lat, lon, elev, time]
                - 'species_idx': (B,) tensor of species indices

        Returns:
            (B,) tensor of predicted LFMC values (normalized [0, 1])
        """
        coords = batch_data['coords']
        species_idx = batch_data['species_idx']

        # Get Earth4D spatiotemporal features
        earth4d_features = self.earth4d(coords)

        # Get species embeddings
        species_features = self.species_embeddings(species_idx)

        # Concatenate features
        combined_features = torch.cat([earth4d_features, species_features], dim=-1)

        # Predict LFMC
        return self.mlp(combined_features).squeeze(-1)

    def forward_precomputed(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass using precomputed hash indices.

        Use this when hash encoding indices have been precomputed for all coordinates
        via model.earth4d.precompute(coords).

        Args:
            batch_data: Dictionary containing:
                - 'indices': (B,) sample indices into precomputed buffers
                - 'species_idx': (B,) tensor of species indices

        Returns:
            (B,) tensor of predicted LFMC values (normalized [0, 1])
        """
        batch_indices = batch_data['indices']
        species_idx = batch_data['species_idx']

        # Get Earth4D features using precomputed indices (sample indices into buffer)
        earth4d_features = self.earth4d.forward_precomputed(batch_indices)

        # Get species embeddings
        species_features = self.species_embeddings(species_idx)

        # Concatenate features
        combined_features = torch.cat([earth4d_features, species_features], dim=-1)

        # Predict LFMC
        return self.mlp(combined_features).squeeze(-1)

    def precompute_hash_indices(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Precompute hash indices for all coordinates.

        Call this once before training with fixed coordinates to enable
        faster forward passes via forward_precomputed().

        Args:
            coords: (N, 4) tensor of [lat, lon, elev, time]

        Returns:
            Precomputed hash indices tensor
        """
        return self.earth4d.precompute_indices(coords)
