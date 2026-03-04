"""
Multi-Scale Spatiotemporal Node Embedder for GNN Weather Forecasting.

This module provides spatiotemporal positional encoding for graph neural network
nodes using multi-scale hash-based encoding. It integrates with GNN architectures
following the encode-process-decode pattern (e.g., GraphCast, Neural-LAM).

Supports two encoder types:
- Earth4D: Absolute coordinates (lat, lon, elev, time)
- Energy4D: Relative coordinates (Δx, Δy, Δz, Δt) - Better generalization
"""

from typing import Optional, Literal

import torch
import torch.nn as nn

# Import Earth4D encoder
try:
    from encoders.xyzt.earth4d import Earth4D
    EARTH4D_AVAILABLE = True
except ImportError as e:
    EARTH4D_AVAILABLE = False
    EARTH4D_IMPORT_ERROR = str(e)

# Import Energy4D encoder
try:
    from encoders.energy4d.energy4d import Energy4D
    ENERGY4D_AVAILABLE = True
except ImportError as e:
    ENERGY4D_AVAILABLE = False
    ENERGY4D_IMPORT_ERROR = str(e)


class MultiScaleSpatioTemporalNode(nn.Module):
    """
    Multi-scale spatiotemporal node embedder for GNN-based weather models.

    This embedder combines Earth4D's multi-scale hash encoding with input features
    and projects to the GNN hidden dimension. It provides spatiotemporally-aware
    node representations that capture patterns from global to local scales.

    Architecture:
        Input: Features (B, N, feature_dim) + Coordinates (N, 4)
            ↓
        Earth4D Multi-Scale Hash Encoding → (B, N, 192)
            ↓
        Concatenate with Input Features
            ↓
        Projection MLP → (B, N, hidden_dim)
            ↓
        Output: Node embeddings for GNN

    Parameters
    ----------
    input_dim : int
        Dimension of input features per node.
    hidden_dim : int
        Output dimension for GNN processing.
    spatial_levels : int
        Number of spatial resolution levels (default: 24).
    temporal_levels : int
        Number of temporal resolution levels (default: 24).
    features_per_level : int
        Number of features per hash table level (default: 2).
    coordinate_system : str
        Coordinate system: 'geographic' (lat/lon/elev) or 'ecef' (X/Y/Z).
    use_adaptive_range : bool
        Whether to fit coordinate range to training data (default: True).
    resolution_mode : str
        Resolution distribution mode (default: 'balanced').
    verbose : bool
        Print configuration details (default: False).

    Examples
    --------
    >>> embedder = MultiScaleSpatioTemporalNode(
    ...     input_dim=13,
    ...     hidden_dim=64,
    ...     spatial_levels=24,
    ...     temporal_levels=24
    ... ).cuda()
    >>> features = torch.randn(4, 7680, 13).cuda()  # (batch, nodes, features)
    >>> coords = torch.randn(7680, 4).cuda()  # (nodes, 4) - [lat, lon, elev, time]
    >>> embeddings = embedder(features, coords)  # (4, 7680, 64)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        encoder_type: Literal["earth4d", "energy4d"] = "earth4d",
        spatial_levels: int = 24,
        temporal_levels: int = 24,
        features_per_level: int = 2,
        coordinate_system: str = "geographic",
        use_adaptive_range: bool = True,
        resolution_mode: str = "balanced",
        verbose: bool = False,
    ):
        super().__init__()

        self.encoder_type = encoder_type

        # Initialize encoder based on type
        if encoder_type == "earth4d":
            if not EARTH4D_AVAILABLE:
                raise ImportError(
                    f"Earth4D encoder could not be imported. Error: {EARTH4D_IMPORT_ERROR}\n"
                    f"Please ensure the encoders.xyzt module is in your Python path."
                )

            self.encoder = Earth4D(
                spatial_levels=spatial_levels,
                temporal_levels=temporal_levels,
                features_per_level=features_per_level,
                coordinate_system=coordinate_system,
                use_adaptive_range=use_adaptive_range,
                resolution_mode=resolution_mode,
                verbose=verbose,
                enable_learned_probing=False,  # Disabled: in-place operation causes gradient errors
            )

        elif encoder_type == "energy4d":
            if not ENERGY4D_AVAILABLE:
                raise ImportError(
                    f"Energy4D encoder could not be imported. Error: {ENERGY4D_IMPORT_ERROR}\n"
                    f"Please ensure the encoders.energy4d module is in your Python path."
                )

            self.encoder = Energy4D(
                spatial_levels=spatial_levels,
                temporal_levels=temporal_levels,
                features_per_level=features_per_level,
                coordinate_system=coordinate_system,
                use_adaptive_range=use_adaptive_range,
                resolution_mode=resolution_mode,
                verbose=verbose,
                enable_learned_probing=False,  # Disabled: in-place operation causes gradient errors
            )

        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. Must be 'earth4d' or 'energy4d'")

        # Calculate encoder output dimension
        # Encoder has 1 spatial encoder (XYZ) + 3 spatiotemporal encoders (XYT, YZT, XZT)
        spatial_dim = spatial_levels * features_per_level
        spatiotemporal_dim = temporal_levels * features_per_level * 3
        encoder_dim = spatial_dim + spatiotemporal_dim

        # Combined dimension
        combined_dim = encoder_dim + input_dim

        # Projection MLP: combined features → hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Store configuration
        self.encoder_dim = encoder_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.spatial_levels = spatial_levels
        self.temporal_levels = temporal_levels
        self.features_per_level = features_per_level
        self.coordinate_system = coordinate_system
        self.use_adaptive_range = use_adaptive_range

        if verbose:
            print(f"\n{'='*50}")
            print("Multi-Scale Spatiotemporal Node Embedder")
            print(f"{'='*50}")
            print(f"  Encoder type: {encoder_type.upper()}")
            print(f"  Input features: {input_dim}")
            print(f"  Encoder output: {encoder_dim}")
            print(f"  Spatial levels: {spatial_levels}")
            print(f"  Temporal levels: {temporal_levels}")
            print(f"  Features per level: {features_per_level}")
            print(f"  Combined dimension: {combined_dim}")
            print(f"  Hidden dimension: {hidden_dim}")
            print(f"  Coordinate system: {coordinate_system}")
            print(f"  Adaptive range: {use_adaptive_range}")
            print(f"  Resolution mode: {resolution_mode}")
            print(f"  Learned probing: False (disabled for stability)")
            print(f"{'='*50}\n")

    def forward(
        self,
        node_features: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode graph nodes with spatiotemporal embeddings.

        Parameters
        ----------
        node_features : torch.Tensor
            Input features at each node, shape (B, N, input_dim).
        coordinates : torch.Tensor
            Coordinates [lat, lon, elev, time] for each node.
            Shape: (N, 4) or (B, N, 4).

        Returns
        -------
        embeddings : torch.Tensor
            Spatiotemporally-encoded node embeddings, shape (B, N, hidden_dim).
        """
        batch_size = node_features.shape[0]
        num_nodes = node_features.shape[1]

        # Handle coordinate broadcasting
        if coordinates.dim() == 2:  # (N, 4)
            coords = coordinates.unsqueeze(0).expand(batch_size, -1, -1)
        else:  # (B, N, 4)
            coords = coordinates

        # Flatten for encoder
        coords_flat = coords.reshape(-1, 4)

        # Apply multi-scale encoding
        spatial_temporal_features = self.encoder(coords_flat)

        # Reshape back to batch
        spatial_temporal_features = spatial_temporal_features.reshape(
            batch_size, num_nodes, self.encoder_dim
        )

        # Concatenate spatiotemporal encoding with input features
        combined = torch.cat([spatial_temporal_features, node_features], dim=-1)

        # Project to hidden dimension
        embeddings = self.projection(combined)

        return embeddings

    def fit_to_data(self, coords: torch.Tensor):
        """
        Fit Earth4D's adaptive range to training coordinates.

        Call this once before training if use_adaptive_range=True.

        Parameters
        ----------
        coords : torch.Tensor
            All training coordinates, shape (N, 4).
        """
        if not self.use_adaptive_range:
            print(
                "Warning: fit_to_data called but use_adaptive_range=False. "
                "Skipping range fitting."
            )
            return

        print(f"Fitting adaptive range to {len(coords)} coordinates...")

        if self.coordinate_system == "geographic":
            self.encoder.fit_geo_range(coords)
        else:
            self.encoder.fit_range(coords)

        print("Adaptive range fitting complete.")

    def get_config(self) -> dict:
        """
        Get embedder configuration.

        Returns
        -------
        config : dict
            Configuration dictionary.
        """
        config = {
            "encoder_type": self.encoder_type,
            "encoder_dim": self.encoder_dim,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "spatial_levels": self.spatial_levels,
            "temporal_levels": self.temporal_levels,
            "features_per_level": self.features_per_level,
            "coordinate_system": self.coordinate_system,
            "use_adaptive_range": self.use_adaptive_range,
            "total_parameters": sum(p.numel() for p in self.parameters()),
        }

        if hasattr(self.encoder, "geo_range"):
            geo_range = self.encoder.geo_range
            config["coordinate_range"] = {
                "lat_min": geo_range.lat_min,
                "lat_max": geo_range.lat_max,
                "lon_min": geo_range.lon_min,
                "lon_max": geo_range.lon_max,
                "elev_min": geo_range.elev_min,
                "elev_max": geo_range.elev_max,
            }

        return config


def check_availability(encoder_type: str = "earth4d") -> tuple[bool, Optional[str]]:
    """
    Check if specified encoder is available.

    Parameters
    ----------
    encoder_type : str
        Encoder type: "earth4d" or "energy4d"

    Returns
    -------
    available : bool
        True if encoder can be imported.
    error_message : str or None
        Error message if import failed, None otherwise.
    """
    if encoder_type == "earth4d":
        return EARTH4D_AVAILABLE, EARTH4D_IMPORT_ERROR if not EARTH4D_AVAILABLE else None
    elif encoder_type == "energy4d":
        return ENERGY4D_AVAILABLE, ENERGY4D_IMPORT_ERROR if not ENERGY4D_AVAILABLE else None
    else:
        return False, f"Unknown encoder type: {encoder_type}"
