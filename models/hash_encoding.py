"""Multi-resolution hash encoding for continuous coordinates.

Based on InstantNGP: https://nvlabs.github.io/instant-ngp/
"""

import torch
import torch.nn as nn
from typing import List


class MultiResolutionHashEncoding(nn.Module):
    """Multi-resolution hash encoding for continuous coordinates.
    
    Encodes continuous coordinates into learned features at multiple resolutions.
    This allows the model to capture both fine-grained local patterns and 
    coarse-grained global structure efficiently.
    """
    
    def __init__(self, n_levels: int, n_features_per_level: int, 
                 resolutions: List[int], hash_table_size: int, coords_dim: int):
        """Initialize multi-resolution hash encoding.
        
        Args:
            n_levels: Number of resolution levels
            n_features_per_level: Number of features per resolution level
            resolutions: List of resolution values for each level
            hash_table_size: Size of the hash table
            coords_dim: Dimensionality of input coordinates (3 for spatial, 1 for temporal)
        """
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.resolutions = resolutions
        self.hash_table_size = hash_table_size
        self.coords_dim = coords_dim
        
        # Create hash tables for each resolution level
        self.hash_tables = nn.ModuleList([
            nn.Embedding(hash_table_size, n_features_per_level)
            for _ in range(n_levels)
        ])
        
        # Initialize hash tables with small random values
        for table in self.hash_tables:
            nn.init.uniform_(table.weight, -1e-4, 1e-4)
    
    def hash_coords(self, coords: torch.Tensor, resolution: int) -> torch.Tensor:
        """Hash continuous coordinates to hash table indices.
        
        Args:
            coords: (B, D) continuous coordinates
            resolution: Resolution level for discretization
            
        Returns:
            indices: (B,) hash table indices
        """
        # Scale coordinates by resolution
        scaled_coords = coords * resolution
        
        # Get integer grid coordinates
        grid_coords = torch.floor(scaled_coords).long()
        
        # Simple spatial hash function using large prime numbers
        primes = torch.tensor([1, 2654435761, 805459861, 3674653429], 
                            device=coords.device)[:self.coords_dim]
        hash_indices = torch.zeros(coords.shape[0], device=coords.device, dtype=torch.long)
        
        for i in range(self.coords_dim):
            hash_indices ^= grid_coords[:, i] * primes[i]
        
        # Wrap to hash table size
        hash_indices = hash_indices % self.hash_table_size
        
        return hash_indices
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode coordinates at multiple resolutions.
        
        Args:
            coords: (B, D) continuous coordinates
            
        Returns:
            features: (B, n_levels * n_features_per_level) encoded features
        """
        features = []
        
        for level, resolution in enumerate(self.resolutions[:self.n_levels]):
            # Get hash indices for this resolution
            indices = self.hash_coords(coords, resolution)
            
            # Look up features from hash table
            level_features = self.hash_tables[level](indices)
            features.append(level_features)
        
        # Concatenate features from all levels
        return torch.cat(features, dim=-1)
