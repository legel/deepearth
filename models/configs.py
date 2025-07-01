"""Configuration classes for DeepEarth models."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class TransformerConfig:
    """Configuration for Transformer blocks."""
    hidden_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-6


@dataclass
class ModalityConfig:
    """Configuration for a specific data modality."""
    name: str
    encoding_type: str  # 'learned_embedding' or 'continuous_values'
    input_type: str  # 'categorical' or 'numerical'
    column_name: Optional[str] = None
    column_names: Optional[List[str]] = None
    embed_dim: Optional[int] = None
    custom_encoder_config: Optional[Dict[str, Any]] = None


@dataclass
class DeepEarthConfig:
    """Main configuration for DeepEarth model."""
    # Spatiotemporal encoding
    spatial_coordinate_system: str = "cartesian"  # or "spherical"
    spatial_resolutions: List[int] = None
    temporal_resolutions: List[int] = None
    n_spatial_levels: int = 16
    n_temporal_levels: int = 8
    n_features_per_level: int = 2
    hash_table_size: int = 2**19
    
    # Model dimensions
    hidden_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    
    # Modality configurations
    modality_configs: Dict[str, ModalityConfig] = None
    
    # Sub-model configurations
    modality_encoder_config: TransformerConfig = None
    cross_modal_fusion_config: TransformerConfig = None
    
    def __post_init__(self):
        if self.spatial_resolutions is None:
            # Default multi-resolution levels from coarse to fine
            self.spatial_resolutions = [2**i for i in range(4, 4 + self.n_spatial_levels)]
        if self.temporal_resolutions is None:
            self.temporal_resolutions = [2**i for i in range(2, 2 + self.n_temporal_levels)]
        if self.modality_encoder_config is None:
            self.modality_encoder_config = TransformerConfig(
                hidden_dim=self.hidden_dim // 2,
                n_heads=6,
                n_layers=4
            )
        if self.cross_modal_fusion_config is None:
            self.cross_modal_fusion_config = TransformerConfig(
                hidden_dim=self.hidden_dim,
                n_heads=self.n_heads,
                n_layers=self.n_layers
            )
        if self.modality_configs is None:
            self.modality_configs = {}
