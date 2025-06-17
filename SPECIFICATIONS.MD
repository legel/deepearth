# DeepEarth: A Universal Multimodal Foundation Model for Earth System Science

## Introduction

DeepEarth is a self-supervised foundation model that learns unified representations of Earth's ecosystems by reconstructing masked observations across space, time, and arbitrary data modalities. Built on DeepSeek-V3's state-of-the-art transformer architecture and Grid4D's efficient spatiotemporal encoding, DeepEarth can seamlessly integrate any Earth observation data—from species occurrences to satellite imagery—into a coherent understanding of our planet's dynamics.

## Core Principles

1. **Modality Agnostic**: Any Earth observation data can be integrated without architectural changes
2. **Self-Supervised**: Learns by reconstructing masked inputs, requiring no labels
3. **Spatiotemporally Aware**: Native understanding of space and time through Grid4D encoding
4. **Hierarchical Processing**: Small transformers for modality-specific processing, large transformer for cross-modal fusion
5. **Context-Rich Learning**: Samples related observations to learn ecological relationships

## Architecture Overview

```
Input Data → Grid4D Encoding → Modality Encoders → Cross-Modal Fusion → Modality Decoders → Reconstructions
    ↓              ↓                    ↓                   ↓                    ↓              ↓
(lat,lon,alt,t) (Multi-res)    (Small, specific)    (Large, fusion)     (Small, specific)  (Original space)
```

## Implementation

### 1. Core Dependencies and Setup

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, replace as dataclasses_replace
import os
import sys
import logging
from datetime import datetime, timedelta
import json
from collections import defaultdict
import math
import faiss  # For efficient nearest neighbor search

# Add external dependencies
sys.path.append('/home/photon/deepearth/external/DeepSeek-V3')
sys.path.append('/home/photon/deepearth/external/Grid4D')

# DeepSeek-V3 imports
from inference.model import (
    Transformer,
    ModelArgs,
    Block,
    MLA,
    MoE,
    MLP,
    RMSNorm,
    precompute_freqs_cis,
    apply_rotary_emb
)

# Grid4D imports
from hashencoder import HashEncoder
from scene.network import Grid4D as Grid4DBase

# DeepEarth utilities
from deepearth.geospatial.geo2xyz import GeospatialConverter

# Configure logging
def setup_logging(level: str = "INFO"):
    """Initialize comprehensive logging system"""
    log_level = getattr(logging, level.upper())
    formatter = logging.Formatter(
        '%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger('DeepEarth')
    logger.setLevel(log_level)
    logger.addHandler(console_handler)
    
    return logger
```

### 2. Configuration System

```python
@dataclass
class ModalityEncoderConfig(ModelArgs):
    """Configuration for small modality-specific transformers"""
    
    # Compact architecture
    n_layers: int = 1
    dim: int = 128
    n_heads: int = 4
    head_dim: int = 32
    
    # Minimal MoE
    moe_layer_start: int = 0
    n_routed_experts: int = 4
    n_activated_experts: int = 2
    n_group: int = 1
    n_limited_groups: int = 1
    n_shared_experts: int = 1
    moe_intermediate_size: int = 256
    
    # Light compression
    q_lora_rank: int = 32
    kv_lora_rank: int = 16
    qk_nope_head_dim: int = 16
    qk_rope_head_dim: int = 16
    
    # Modality-specific settings
    max_position_embeddings: int = 512
    use_causal_mask: bool = False


@dataclass
class SpatiotemporalTransformerConfig(ModelArgs):
    """Configuration for spatiotemporal coordinate encoding transformers"""
    
    # Architecture optimized for coordinate processing
    n_layers: int = 2
    dim: int = 256
    n_heads: int = 8
    head_dim: int = 32
    
    # MoE for multi-scale reasoning
    moe_layer_start: int = 1
    n_routed_experts: int = 8
    n_activated_experts: int = 2
    n_group: int = 1
    n_limited_groups: int = 1
    n_shared_experts: int = 1
    moe_intermediate_size: int = 512
    
    # Compression settings
    q_lora_rank: int = 64
    kv_lora_rank: int = 32
    qk_nope_head_dim: int = 24
    qk_rope_head_dim: int = 24
    
    # Spatiotemporal-specific settings
    max_position_embeddings: int = 1024
    use_causal_mask: bool = False


@dataclass
class CrossModalFusionConfig(ModelArgs):
    """Configuration for large cross-modal fusion network"""
    
    # Full architecture
    n_layers: int = 12
    dim: int = 768
    n_heads: int = 12
    head_dim: int = 64
    
    # Rich MoE
    moe_layer_start: int = 0
    n_routed_experts: int = 32
    n_activated_experts: int = 4
    n_group: int = 1
    n_limited_groups: int = 1
    n_shared_experts: int = 2
    moe_intermediate_size: int = 1536
    
    # Full compression
    q_lora_rank: int = 256
    kv_lora_rank: int = 128
    qk_nope_head_dim: int = 32
    qk_rope_head_dim: int = 32
    
    # Cross-modal settings
    max_position_embeddings: int = 2048
    use_causal_mask: bool = False


@dataclass
class ModalityConfig:
    """Configuration for a single data modality
    
    This defines how a particular type of Earth observation data is processed.
    Supports both categorical data (e.g., species names) and numerical data
    (e.g., climate measurements), with flexible encoding strategies.
    """
    
    name: str                                     # Unique identifier (e.g., 'species', 'climate')
    encoding_type: str                           # 'learned_embedding' or 'continuous_values'
    input_type: str                              # 'categorical' or 'numerical'
    column_name: Optional[str] = None            # For categorical data
    column_names: Optional[List[str]] = None     # For numerical data
    embed_dim: Optional[int] = None              # Embedding dimension for learned embeddings
    vocab_size: Optional[int] = None             # Set automatically from data
    
    # Optional custom transformer config
    custom_encoder_config: Optional[Dict[str, Any]] = None  # Override default encoder settings
    
    def validate(self):
        """Validate configuration consistency"""
        if self.input_type == 'categorical':
            assert self.encoding_type == 'learned_embedding', "Categorical data requires learned embeddings"
            assert self.column_name is not None, "Categorical data requires column_name"
            assert self.embed_dim is not None, "Learned embeddings require embed_dim"
        elif self.input_type == 'numerical':
            assert self.column_names is not None, "Numerical data requires column_names"
            if self.encoding_type == 'learned_embedding':
                assert self.embed_dim is not None, "Learned embeddings require embed_dim"
        else:
            raise ValueError(f"Unknown input type: {self.input_type}")


@dataclass
class DeepEarthConfig:
    """Master configuration for DeepEarth system"""
    
    # Network configurations
    modality_encoder_config: ModalityEncoderConfig = field(default_factory=ModalityEncoderConfig)
    spatiotemporal_encoder_config: SpatiotemporalTransformerConfig = field(default_factory=SpatiotemporalTransformerConfig)
    cross_modal_fusion_config: CrossModalFusionConfig = field(default_factory=CrossModalFusionConfig)
    
    # Grid4D multi-resolution settings
    spatial_resolutions: List[float] = field(default_factory=lambda: [
        1.0,      # 1 meter
        10.0,     # 10 meters
        100.0,    # 100 meters
        1000.0    # 1 kilometer
    ])
    
    temporal_resolutions: List[float] = field(default_factory=lambda: [
        3600.0,           # 1 hour
        86400.0,          # 1 day
        604800.0,         # 1 week
        31536000.0        # 1 year
    ])
    
    # Coordinate system
    spatial_coordinate_system: str = 'geographic'  # 'geographic' or 'ecef' or 'local_xyz'
    altitude_column: str = 'altitude'             # Supports both elevation and depth
    
    # Geographic bounds
    geographic_bounds: Optional[Dict] = None
    temporal_bounds: Dict = field(default_factory=lambda: {
        'start': datetime(2010, 1, 1, 0, 0, 0),
        'end': datetime(2025, 12, 31, 23, 59, 59)
    })
    
    # Modality definitions (extensible)
    modality_configs: Dict[str, ModalityConfig] = field(default_factory=dict)
    
    # Masking strategy
    masking_config: Dict = field(default_factory=lambda: {
        'modality_mask_prob': 0.2,        # Probability to mask entire modality
        'partial_mask_prob': 0.3,         # Probability to partially mask
        'partial_mask_ratio': 0.15,       # Ratio when partially masking
        'mask_deep_embeddings': False     # Whether to mask at embedding level
    })
    
    # Context sampling
    context_size: int = 32                # Total tokens per batch
    batch_size: int = 8                   # Number of independent contexts
    neighbor_sampling_ratios: Dict = field(default_factory=lambda: {
        'spatial': 0.33,
        'temporal': 0.33,
        'ecological': 0.34
    })
    
    # Training settings
    learning_rate: float = 1e-4
    n_epochs: int = 100
    
    # Validation settings
    validation_patch_size: float = 0.02   # 2% of area per patch
    n_spatial_holdout_patches: int = 10
    temporal_holdout_years: int = 1
    
    # System settings
    log_level: str = "INFO"
    log_every_n_steps: int = 50
    validate_every_n_epochs: int = 5
    visualize_every_n_epochs: int = 10
    
    # Data statistics (computed during preprocessing)
    data_statistics: Optional[Dict] = None
    altitude_outlier_std: float = 3.0
    
    # Grid4D hash encoding parameters
    grid4d_level_dim: int = 2
    grid4d_log2_hashmap_size: int = 19
    grid4d_bound: float = 1.6  # Must be positive
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.grid4d_bound <= 0:
            raise ValueError("grid4d_bound must be positive")
        if not isinstance(self.temporal_bounds['start'], datetime):
            raise ValueError("temporal_bounds['start'] must be a datetime object")
        if not isinstance(self.temporal_bounds['end'], datetime):
            raise ValueError("temporal_bounds['end'] must be a datetime object")
    
    # Default temporal values for missing data
    default_month: int = 6      # June (middle of year)
    default_day: int = 15       # Middle of month
    default_hour: int = 12      # Noon
    
    # Visualization parameters
    viz_grid_size: int = 10
    viz_lat_range: float = 0.01
    viz_lon_range: float = 0.01
```

### 3. Data Loading Standards and Infrastructure

#### Data Format Specification

DeepEarth supports flexible spatial coordinate systems and temporal resolutions.

**Spatial Coordinate Options:**

1. **Geographic Coordinates** (default):
   - `latitude`: Decimal degrees (-90 to 90)
   - `longitude`: Decimal degrees (-180 to 180)
   - `altitude`: Meters relative to sea level (negative for below surface)

2. **ECEF Coordinates**:
   - `x`: Meters from Earth center
   - `y`: Meters from Earth center
   - `z`: Meters from Earth center

3. **Local XYZ Coordinates**:
   - `x`: Meters in local coordinate system
   - `y`: Meters in local coordinate system
   - `z`: Meters in local coordinate system

**Temporal Format Options:**

1. **Component Format** (default):
   - `year`: Integer year
   - `month`: Integer month (optional)
   - `day`: Integer day (optional)
   - `hour`: Integer hour (optional)
   - `minute`: Integer minute (optional)
   - `second`: Float second with microsecond precision (optional)

2. **Timestamp Format**:
   - `timestamp`: Float seconds since epoch with arbitrary precision

**Modality Columns:**
- Categorical data: Single column with string/integer values
- Numerical data: Multiple columns with float values

#### Data Loading API

```python
class DatasetLoader:
    """Standardized data loading for DeepEarth"""
    
    def __init__(self, csv_path: str, modality_configs: List[ModalityConfig], 
                 coordinate_system: str = 'geographic'):
        self.csv_path = csv_path
        self.modality_configs = {config.name: config for config in modality_configs}
        self.coordinate_system = coordinate_system
        self.df = None
        
    def load_and_validate(self) -> pd.DataFrame:
        """Load CSV and validate against expected format"""
        
        # Load data
        self.df = pd.read_csv(self.csv_path)
        
        # Validate spatial coordinates
        if self.coordinate_system == 'geographic':
            required_spatial = ['latitude', 'longitude']
            missing = [col for col in required_spatial if col not in self.df.columns]
            if missing:
                raise ValueError(f"Missing required geographic columns: {missing}")
            # Add altitude if missing
            if 'altitude' not in self.df.columns:
                self.df['altitude'] = 0.0
        elif self.coordinate_system in ['ecef', 'local_xyz']:
            required_spatial = ['x', 'y', 'z']
            missing = [col for col in required_spatial if col not in self.df.columns]
            if missing:
                raise ValueError(f"Missing required {self.coordinate_system} columns: {missing}")
        else:
            raise ValueError(f"Unknown coordinate system: {self.coordinate_system}")
        
        # Validate temporal coordinates
        if 'timestamp' in self.df.columns:
            # Using timestamp format
            pass
        elif 'year' in self.df.columns:
            # Using component format
            # Add defaults for missing components
            if 'month' not in self.df.columns:
                self.df['month'] = self.config.default_month if hasattr(self, 'config') else 6
            if 'day' not in self.df.columns:
                self.df['day'] = self.config.default_day if hasattr(self, 'config') else 15
            if 'hour' not in self.df.columns:
                self.df['hour'] = self.config.default_hour if hasattr(self, 'config') else 12
            if 'minute' not in self.df.columns:
                self.df['minute'] = 0
            if 'second' not in self.df.columns:
                self.df['second'] = 0.0
        else:
            raise ValueError("Missing temporal data: need either 'timestamp' or 'year' column")
        
        # Validate modality columns
        for config in self.modality_configs.values():
            config.validate()
            
            if config.input_type == 'categorical':
                if config.column_name not in self.df.columns:
                    raise ValueError(f"Missing column for {config.name}: {config.column_name}")
            else:  # numerical
                missing = [col for col in config.column_names if col not in self.df.columns]
                if missing:
                    raise ValueError(f"Missing columns for {config.name}: {missing}")
        
        return self.df
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics about the dataset"""
        
        if self.df is None:
            self.load_and_validate()
        
        summary = {
            'n_observations': len(self.df),
            'coordinate_system': self.coordinate_system,
            'modalities': {}
        }
        
        # Spatial extent
        if self.coordinate_system == 'geographic':
            summary['spatial_extent'] = {
                'lat_range': (self.df['latitude'].min(), self.df['latitude'].max()),
                'lon_range': (self.df['longitude'].min(), self.df['longitude'].max()),
                'alt_range': (self.df['altitude'].min(), self.df['altitude'].max())
            }
        else:
            summary['spatial_extent'] = {
                'x_range': (self.df['x'].min(), self.df['x'].max()),
                'y_range': (self.df['y'].min(), self.df['y'].max()),
                'z_range': (self.df['z'].min(), self.df['z'].max())
            }
        
        # Temporal extent
        if 'timestamp' in self.df.columns:
            summary['temporal_extent'] = {
                'timestamp_range': (self.df['timestamp'].min(), self.df['timestamp'].max())
            }
        else:
            summary['temporal_extent'] = {
                'year_range': (self.df['year'].min(), self.df['year'].max())
            }
        
        # Modality summaries
        for name, config in self.modality_configs.items():
            if config.input_type == 'categorical':
                n_unique = self.df[config.column_name].nunique()
                summary['modalities'][name] = {
                    'type': 'categorical',
                    'encoding': config.encoding_type,
                    'n_unique_values': n_unique,
                    'sample_values': self.df[config.column_name].value_counts().head(5).to_dict()
                }
            else:
                stats = self.df[config.column_names].describe().to_dict()
                summary['modalities'][name] = {
                    'type': 'numerical',
                    'encoding': config.encoding_type,
                    'n_dimensions': len(config.column_names),
                    'statistics': stats
                }
        
        return summary


class DataPreprocessor:
    """Universal preprocessor for arbitrary modalities"""
    
    def __init__(self, config: DeepEarthConfig):
        self.config = config
        self.geo_converter = GeospatialConverter()
        self.logger = logging.getLogger('DeepEarth.Preprocessor')
        
        self._load_geographic_bounds()
        
        # Statistics
        self.spatial_stats = {}
        self.temporal_stats = {}
        self.modality_stats = {}
        
    def _load_geographic_bounds(self):
        """Load geographic bounds from configuration"""
        geojson_path = self.config.geographic_bounds.get('geojson_path') if self.config.geographic_bounds else None
        
        if geojson_path and os.path.exists(geojson_path):
            gdf = gpd.read_file(geojson_path)
            bounds = gdf.total_bounds
            
            self.config.geographic_bounds = {
                'min_lon': bounds[0],
                'min_lat': bounds[1],
                'max_lon': bounds[2],
                'max_lat': bounds[3]
            }
    
    def compute_dataset_statistics(self, df: pd.DataFrame):
        """Compute universal statistics for any dataset"""
        
        # Clean outliers
        df = self._remove_outliers(df)
        
        # Spatial statistics
        self._compute_spatial_stats(df)
        
        # Temporal statistics
        self._compute_temporal_stats(df)
        
        # Modality statistics (dynamically for any modality)
        self._compute_modality_stats(df)
        
        # Store in config
        self.config.data_statistics = {
            'n_observations': len(df),
            'spatial_stats': self.spatial_stats,
            'temporal_stats': self.temporal_stats,
            'modality_stats': self.modality_stats
        }
        
        return df
    
    def _compute_modality_stats(self, df: pd.DataFrame):
        """Dynamically compute statistics for each modality"""
        
        for modality_name, modality_config in self.config.modality_configs.items():
            if modality_config.input_type == 'categorical':
                # Handle categorical modalities
                column_name = modality_config.column_name
                unique_values = sorted(df[column_name].unique())
                
                self.modality_stats[modality_name] = {
                    'vocab': {val: i for i, val in enumerate(unique_values)},
                    'inverse_vocab': {i: val for i, val in enumerate(unique_values)},
                    'n_unique': len(unique_values),
                    'embeddings': nn.Embedding(len(unique_values), modality_config.embed_dim)
                }
                
                # Update vocab size
                self.config.modality_configs[modality_name].vocab_size = len(unique_values)
            
            elif modality_config.input_type == 'numerical':
                # Handle numerical modalities
                column_names = modality_config.column_names
                data = df[column_names].values
                
                self.modality_stats[modality_name] = {
                    'mean': data.mean(axis=0),
                    'std': data.std(axis=0),
                    'min': data.min(axis=0),
                    'max': data.max(axis=0)
                }
                
                if modality_config.encoding_type == 'learned_embedding':
                    # Create projection layer for numerical to embedding
                    input_dim = len(column_names)
                    self.modality_stats[modality_name]['projection'] = nn.Linear(
                        input_dim, modality_config.embed_dim
                    )
    
    def preprocess_batch(self, batch_df: pd.DataFrame, apply_masking: bool = True):
        """Universal preprocessing for any modality configuration"""
        
        batch_size = len(batch_df)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Core coordinates
        xyz_normalized = self._process_spatial(batch_df, device)
        t_normalized = self._process_temporal(batch_df, device)
        
        # Initialize masks
        modality_masks = {
            'spatial': torch.ones(batch_size, dtype=torch.bool, device=device),
            'temporal': torch.ones(batch_size, dtype=torch.bool, device=device)
        }
        
        # Process each configured modality
        modalities = {}
        internal_masks = {}
        ground_truth = {
            'xyz': xyz_normalized.clone(),
            't': t_normalized.clone()
        }
        
        for modality_name, modality_config in self.config.modality_configs.items():
            # Process modality data
            modality_data, modality_ground_truth = self._process_modality(
                batch_df, modality_name, modality_config, device
            )
            modalities[modality_name] = modality_data
            ground_truth[modality_name] = modality_ground_truth
            
            # Initialize masks
            modality_masks[modality_name] = torch.ones(batch_size, dtype=torch.bool, device=device)
            
            if modality_config.embed_dim:
                internal_masks[modality_name] = torch.ones(
                    (batch_size, modality_config.embed_dim), 
                    dtype=torch.bool, device=device
                )
        
        # Apply masking strategy
        if apply_masking:
            self._apply_masking(modality_masks, internal_masks, modalities, xyz_normalized, t_normalized)
        
        return {
            'xyz': xyz_normalized,
            't': t_normalized,
            'xyzt': torch.cat([xyz_normalized, t_normalized.unsqueeze(-1)], dim=-1),
            'modality_masks': modality_masks,
            'internal_masks': internal_masks,
            'modalities': modalities,
            'ground_truth': ground_truth,
            'human_readable': self._extract_human_readable(batch_df)
        }
    
    def _apply_masking(self, modality_masks, internal_masks, modalities, xyz, t):
        """Universal masking strategy for all modalities"""
        
        batch_size = xyz.shape[0]
        
        # Spatial and temporal masking
        for i in range(batch_size):
            # Spatial
            if torch.rand(1).item() < self.config.masking_config['modality_mask_prob']:
                modality_masks['spatial'][i] = False
                xyz[i] = 0
            elif torch.rand(1).item() < self.config.masking_config['partial_mask_prob']:
                # Partial masking of coordinates
                mask_dims = torch.randperm(3)[:int(3 * self.config.masking_config['partial_mask_ratio'])]
                xyz[i, mask_dims] = 0
            
            # Temporal
            if torch.rand(1).item() < self.config.masking_config['modality_mask_prob']:
                modality_masks['temporal'][i] = False
                t[i] = 0
        
        # Modality-specific masking
        for modality_name, modality_data in modalities.items():
            for i in range(batch_size):
                if torch.rand(1).item() < self.config.masking_config['modality_mask_prob']:
                    # Mask entire modality
                    modality_masks[modality_name][i] = False
                    if modality_name in internal_masks:
                        internal_masks[modality_name][i] = False
                elif torch.rand(1).item() < self.config.masking_config['partial_mask_prob']:
                    # Partial masking
                    if modality_name in internal_masks:
                        n_mask = int(internal_masks[modality_name].shape[1] * 
                                    self.config.masking_config['partial_mask_ratio'])
                        mask_indices = torch.randperm(internal_masks[modality_name].shape[1])[:n_mask]
                        internal_masks[modality_name][i, mask_indices] = False
    
    def _remove_outliers(self, df):
        """Remove spatial, temporal, and altitude outliers"""
        
        initial_count = len(df)
        
        # Geographic bounds (if using geographic coordinates)
        if self.config.spatial_coordinate_system == 'geographic' and self.config.geographic_bounds:
            df = df[
                (df['latitude'] >= self.config.geographic_bounds['min_lat']) &
                (df['latitude'] <= self.config.geographic_bounds['max_lat']) &
                (df['longitude'] >= self.config.geographic_bounds['min_lon']) &
                (df['longitude'] <= self.config.geographic_bounds['max_lon'])
            ]
        
        # Temporal bounds
        if 'year' in df.columns:
            df = df[
                (df['year'] >= self.config.temporal_bounds['start'].year) &
                (df['year'] <= self.config.temporal_bounds['end'].year)
            ]
        
        # Altitude outliers
        alt_col = self.config.altitude_column
        if alt_col in df.columns:
            alt = df[alt_col].fillna(0.0)
            alt_mean, alt_std = alt.mean(), alt.std()
            df = df[
                (alt >= alt_mean - self.config.altitude_outlier_std * alt_std) &
                (alt <= alt_mean + self.config.altitude_outlier_std * alt_std)
            ]
        
        removed = initial_count - len(df)
        if removed > 0:
            self.logger.info(f"Removed {removed} outliers ({removed/initial_count*100:.1f}%)")
        
        return df
    
    def _process_spatial(self, batch_df, device):
        """Convert spatial coordinates to normalized representation"""
        
        batch_size = len(batch_df)
        
        if self.config.spatial_coordinate_system == 'geographic':
            # Convert to ECEF then normalize
            xyz_normalized = torch.zeros((batch_size, 3), device=device)
            
            for i, (_, row) in enumerate(batch_df.iterrows()):
                ecef = self.geo_converter.geodetic_to_ecef(
                    torch.tensor([row['latitude']]),
                    torch.tensor([row['longitude']]),
                    torch.tensor([row.get(self.config.altitude_column, 0.0)])
                ).to(device)
                
                xyz_norm = (ecef - self.spatial_stats['center'].to(device)) / self.spatial_stats['scale'].to(device)
                xyz_normalized[i] = xyz_norm.squeeze()
        
        elif self.config.spatial_coordinate_system == 'ecef':
            # Already in ECEF, just normalize
            xyz = torch.tensor(batch_df[['x', 'y', 'z']].values, device=device)
            xyz_normalized = (xyz - self.spatial_stats['center'].to(device)) / self.spatial_stats['scale'].to(device)
        
        else:  # local_xyz
            # Direct normalization of local coordinates
            xyz = torch.tensor(batch_df[['x', 'y', 'z']].values, device=device)
            xyz_normalized = (xyz - self.spatial_stats['center'].to(device)) / self.spatial_stats['scale'].to(device)
        
        return xyz_normalized
    
    def _process_temporal(self, batch_df, device):
        """Convert temporal data to normalized continuous time"""
        
        batch_size = len(batch_df)
        t_normalized = torch.zeros(batch_size, device=device)
        
        for i, (_, row) in enumerate(batch_df.iterrows()):
            if 'timestamp' in row:
                # Direct timestamp
                timestamp = row['timestamp']
            else:
                # Build from components
                dt = datetime(
                    int(row['year']),
                    int(row.get('month', self.config.default_month)),
                    int(row.get('day', self.config.default_day)),
                    int(row.get('hour', self.config.default_hour)),
                    int(row.get('minute', 0)),
                    int(row.get('second', 0))
                )
                # Add microseconds if available
                if 'second' in row and isinstance(row['second'], float):
                    microseconds = int((row['second'] % 1) * 1e6)
                    dt = dt.replace(microsecond=microseconds)
                
                timestamp = dt.timestamp()
            
            t_norm = (timestamp - self.temporal_stats['min']) / (self.temporal_stats['max'] - self.temporal_stats['min'])
            t_normalized[i] = t_norm
        
        return t_normalized
    
    def _process_modality(self, batch_df, modality_name, modality_config, device):
        """Process a single modality based on its configuration"""
        
        batch_size = len(batch_df)
        
        if modality_config.input_type == 'categorical':
            # Categorical modality
            column_name = modality_config.column_name
            vocab = self.modality_stats[modality_name]['vocab']
            embeddings = self.modality_stats[modality_name]['embeddings']
            
            ids = torch.tensor([vocab[val] for val in batch_df[column_name]], device=device)
            data = embeddings(ids)
            
            return data, ids
        
        elif modality_config.input_type == 'numerical':
            # Numerical modality
            column_names = modality_config.column_names
            data = torch.tensor(batch_df[column_names].values, dtype=torch.float32, device=device)
            
            # Normalize
            mean = torch.tensor(self.modality_stats[modality_name]['mean'], device=device)
            std = torch.tensor(self.modality_stats[modality_name]['std'], device=device)
            data_normalized = (data - mean) / (std + 1e-6)
            
            if modality_config.encoding_type == 'learned_embedding':
                # Project to embedding space
                projection = self.modality_stats[modality_name]['projection']
                data_embedded = projection(data_normalized)
                return data_embedded, data
            else:
                # Keep as continuous values
                return data_normalized, data
        
        else:
            raise ValueError(f"Unknown modality input type: {modality_config.input_type}")
    
    def _extract_human_readable(self, batch_df):
        """Extract human-readable information for logging"""
        
        human_readable = []
        for _, row in batch_df.iterrows():
            hr = {}
            
            # Spatial info
            if self.config.spatial_coordinate_system == 'geographic':
                hr['latitude'] = row['latitude']
                hr['longitude'] = row['longitude']
                hr['altitude'] = row.get(self.config.altitude_column, 0.0)
            else:
                hr['x'] = row['x']
                hr['y'] = row['y']
                hr['z'] = row['z']
            
            # Temporal info
            if 'timestamp' in row.index and pd.notna(row.get('timestamp')):
                hr['timestamp'] = row['timestamp']
            else:
                hr['year'] = row['year']
                hr['month'] = row.get('month', self.config.default_month)
                hr['day'] = row.get('day', self.config.default_day)
                hr['hour'] = row.get('hour', self.config.default_hour)
            
            # Add modality-specific information
            for modality_name, modality_config in self.config.modality_configs.items():
                if modality_config.input_type == 'categorical':
                    hr[modality_name] = row[modality_config.column_name]
            
            human_readable.append(hr)
        
        return human_readable
    
    def _compute_spatial_stats(self, df):
        """Compute spatial statistics based on coordinate system"""
        
        if self.config.spatial_coordinate_system == 'geographic':
            # Convert all to ECEF for unified processing
            all_ecef = []
            for _, row in df.iterrows():
                ecef = self.geo_converter.geodetic_to_ecef(
                    torch.tensor([row['latitude']]),
                    torch.tensor([row['longitude']]),
                    torch.tensor([row.get(self.config.altitude_column, 0.0)])
                )
                all_ecef.append(ecef)
            
            all_ecef = torch.cat(all_ecef, dim=0)
            self.spatial_stats = {
                'center': all_ecef.mean(dim=0),
                'scale': all_ecef.std(dim=0).max() * 2.0,
                'min': all_ecef.min(dim=0).values,
                'max': all_ecef.max(dim=0).values,
                'span_km': (all_ecef.max(dim=0).values - all_ecef.min(dim=0).values).max().item() / 1000,
                'coordinate_system': 'geographic_to_ecef'
            }
        else:
            # Direct XYZ statistics
            xyz = torch.tensor(df[['x', 'y', 'z']].values)
            self.spatial_stats = {
                'center': xyz.mean(dim=0),
                'scale': xyz.std(dim=0).max() * 2.0,
                'min': xyz.min(dim=0).values,
                'max': xyz.max(dim=0).values,
                'span_m': (xyz.max(dim=0).values - xyz.min(dim=0).values).max().item(),
                'coordinate_system': self.config.spatial_coordinate_system
            }
    
    def _compute_temporal_stats(self, df):
        """Compute temporal statistics"""
        
        timestamps = []
        for _, row in df.iterrows():
            if 'timestamp' in row.index and pd.notna(row.get('timestamp')):
                timestamps.append(row['timestamp'])
            else:
                dt = datetime(
                    int(row['year']),
                    int(row.get('month', self.config.default_month)),
                    int(row.get('day', self.config.default_day)),
                    int(row.get('hour', self.config.default_hour)),
                    int(row.get('minute', 0)),
                    int(row.get('second', 0))
                )
                timestamps.append(dt.timestamp())
        
        timestamps = torch.tensor(timestamps)
        self.temporal_stats = {
            'min': timestamps.min().item(),
            'max': timestamps.max().item(),
            'span_seconds': (timestamps.max() - timestamps.min()).item(),
            'span_hours': (timestamps.max() - timestamps.min()).item() / 3600
        }


class ContextSamplingEngine:
    """Intelligent sampling of related observations for context-rich learning"""
    
    def __init__(self, df: pd.DataFrame, preprocessor: DataPreprocessor, config: DeepEarthConfig):
        self.df = df
        self.preprocessor = preprocessor
        self.config = config
        self.logger = logging.getLogger('DeepEarth.Sampling')
        
        # Build indices for fast nearest neighbor search
        self._build_indices()
    
    def _build_indices(self):
        """Build FAISS indices for efficient neighbor search"""
        
        # Spatial index
        if self.config.spatial_coordinate_system == 'geographic':
            # Convert to ECEF for uniform distance metrics
            spatial_data = []
            for _, row in self.df.iterrows():
                ecef = self.preprocessor.geo_converter.geodetic_to_ecef(
                    torch.tensor([row['latitude']]),
                    torch.tensor([row['longitude']]),
                    torch.tensor([row.get(self.config.altitude_column, 0.0)])
                )
                spatial_data.append(ecef.numpy())
            spatial_data = np.vstack(spatial_data).astype('float32')
        else:
            # Direct XYZ
            spatial_data = self.df[['x', 'y', 'z']].values.astype('float32')
        
        self.spatial_index = faiss.IndexFlatL2(3)
        self.spatial_index.add(spatial_data)
        
        # Temporal index
        temporal_data = []
        for _, row in self.df.iterrows():
            if 'timestamp' in row.index and pd.notna(row.get('timestamp')):
                temporal_data.append([row['timestamp']])
            else:
                dt = datetime(
                    int(row['year']),
                    int(row.get('month', self.config.default_month)),
                    int(row.get('day', self.config.default_day)),
                    int(row.get('hour', self.config.default_hour))
                )
                temporal_data.append([dt.timestamp()])
        
        temporal_data = np.array(temporal_data).astype('float32')
        self.temporal_index = faiss.IndexFlatL2(1)
        self.temporal_index.add(temporal_data)
        
        # Ecological indices (per modality)
        self.ecological_indices = {}
        for modality_name, modality_stats in self.preprocessor.modality_stats.items():
            if 'embeddings' in modality_stats:
                # Get embeddings for all data
                embeddings = []
                for _, row in self.df.iterrows():
                    vocab = modality_stats['vocab']
                    column_name = self.config.modality_configs[modality_name].column_name
                    idx = vocab[row[column_name]]
                    emb = modality_stats['embeddings'](torch.tensor(idx))
                    embeddings.append(emb.detach().numpy())
                
                embeddings = np.vstack(embeddings).astype('float32')
                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(embeddings)
                self.ecological_indices[modality_name] = index
    
    def sample_context(self, seed_idx: int) -> List[int]:
        """Sample a context of related observations"""
        
        context_size = self.config.context_size
        ratios = self.config.neighbor_sampling_ratios
        
        # Calculate samples per category
        n_spatial = int(context_size * ratios.get('spatial', 0.33))
        n_temporal = int(context_size * ratios.get('temporal', 0.33))
        n_ecological = context_size - n_spatial - n_temporal - 1  # -1 for seed
        
        # Get seed point features
        seed_row = self.df.iloc[seed_idx]
        
        if self.config.spatial_coordinate_system == 'geographic':
            seed_ecef = self.preprocessor.geo_converter.geodetic_to_ecef(
                torch.tensor([seed_row['latitude']]),
                torch.tensor([seed_row['longitude']]),
                torch.tensor([seed_row.get(self.config.altitude_column, 0.0)])
            ).numpy().astype('float32')
        else:
            seed_ecef = np.array([seed_row['x'], seed_row['y'], seed_row['z']]).astype('float32')
        
        if 'timestamp' in seed_row.index and pd.notna(seed_row.get('timestamp')):
            seed_time = seed_row['timestamp']
        else:
            seed_time = datetime(
                int(seed_row['year']),
                int(seed_row.get('month', self.config.default_month)),
                int(seed_row.get('day', self.config.default_day)),
                int(seed_row.get('hour', self.config.default_hour))
            ).timestamp()
        
        # Find neighbors
        indices = [seed_idx]
        
        # Spatial neighbors
        _, spatial_neighbors = self.spatial_index.search(seed_ecef.reshape(1, -1), n_spatial + 1)
        indices.extend(spatial_neighbors[0][1:n_spatial+1].tolist())
        
        # Temporal neighbors
        _, temporal_neighbors = self.temporal_index.search(np.array([[seed_time]]).astype('float32'), n_temporal + 1)
        indices.extend(temporal_neighbors[0][1:n_temporal+1].tolist())
        
        # Ecological neighbors (distribute across modalities)
        if self.ecological_indices:
            n_per_modality = n_ecological // len(self.ecological_indices)
            for modality_name, index in self.ecological_indices.items():
                # Get seed embedding
                vocab = self.preprocessor.modality_stats[modality_name]['vocab']
                column_name = self.config.modality_configs[modality_name].column_name
                seed_id = vocab[seed_row[column_name]]
                seed_emb = self.preprocessor.modality_stats[modality_name]['embeddings'](
                    torch.tensor(seed_id)
                ).detach().numpy().reshape(1, -1).astype('float32')
                
                _, eco_neighbors = index.search(seed_emb, n_per_modality + 1)
                indices.extend(eco_neighbors[0][1:n_per_modality+1].tolist())
        
        # Remove duplicates and limit to context size
        indices = list(dict.fromkeys(indices))[:context_size]
        
        return indices
    
    def create_dataloader(self, df: pd.DataFrame, shuffle: bool = True):
        """Create context-aware dataloader"""
        
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        
        n_batches = len(df) // (self.config.batch_size * self.config.context_size)
        
        for batch_idx in range(n_batches):
            batch_dfs = []
            
            for i in range(self.config.batch_size):
                # Sample seed
                seed_idx = (batch_idx * self.config.batch_size + i) * self.config.context_size
                seed_idx = seed_idx % len(df)  # Wrap around
                
                # Get context
                context_indices = self.sample_context(seed_idx)
                context_df = df.iloc[context_indices]
                batch_dfs.append(context_df)
            
            # Concatenate all contexts in batch
            batch_df = pd.concat(batch_dfs, ignore_index=True)
            yield batch_df
```

### 4. Model Architecture

```python
class Grid4DEncoder(nn.Module):
    """Multi-resolution spatiotemporal hash encoder.
    
    Encodes 4D coordinates (x,y,z,t) using Grid4D's multi-resolution
    hash encoding, capturing patterns at multiple spatial and temporal
    scales. The encoded features are projected to the fusion dimension
    using a mixture of experts.
    
    Args:
        config: DeepEarthConfig with Grid4D parameters
    """
    
    def __init__(self, config: DeepEarthConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger('DeepEarth.Grid4D')
        
        # Calculate resolution levels
        self.spatial_levels = len(config.spatial_resolutions)
        self.temporal_levels = len(config.temporal_resolutions)
        
        # Initialize Grid4D with multi-resolution encoding
        self.grid4d = Grid4DBase(
            canonical_num_levels=self.spatial_levels,
            canonical_level_dim=config.grid4d_level_dim,
            canonical_log2_hashmap_size=config.grid4d_log2_hashmap_size,
            deform_num_levels=self.temporal_levels,
            deform_level_dim=config.grid4d_level_dim,
            deform_log2_hashmap_size=config.grid4d_log2_hashmap_size,
            bound=config.grid4d_bound
        )
        
        # Calculate total feature dimension
        # Spatial: xyz encoding
        # Temporal: xyt, yzt, xzt decomposed encodings
        self.spatial_dim = self.spatial_levels * config.grid4d_level_dim
        self.temporal_dim = self.temporal_levels * config.grid4d_level_dim
        self.total_dim = self.spatial_dim + 3 * self.temporal_dim
        
        self.logger.info(f"Grid4D initialized:")
        self.logger.info(f"  Spatial: {self.spatial_levels} levels, {self.spatial_dim} dims")
        self.logger.info(f"  Temporal: {self.temporal_levels} levels, {self.temporal_dim} dims per decomposition")
        self.logger.info(f"  Total dimensions: {self.total_dim}")
        
        # Spatiotemporal transformer for processing hash features
        self.spatiotemporal_transformer = Transformer(config.spatiotemporal_encoder_config)
        
        # Input projection: hash features -> transformer space
        self.input_projection = MoE(
            dim=self.total_dim,
            n_heads=config.spatiotemporal_encoder_config.n_heads,
            head_dim=config.spatiotemporal_encoder_config.head_dim,
            n_routed_experts=config.spatiotemporal_encoder_config.n_routed_experts,
            n_activated_experts=config.spatiotemporal_encoder_config.n_activated_experts,
            n_shared_experts=config.spatiotemporal_encoder_config.n_shared_experts,
            n_group=1,
            n_limited_groups=1,
            moe_intermediate_size=config.spatiotemporal_encoder_config.dim,
            score_func='sigmoid',
            route_scale=2.5
        )
        
        # Output projection: transformer space -> fusion space
        self.output_projection = MoE(
            dim=config.spatiotemporal_encoder_config.dim,
            n_heads=config.spatiotemporal_encoder_config.n_heads,
            head_dim=config.spatiotemporal_encoder_config.head_dim,
            n_routed_experts=config.spatiotemporal_encoder_config.n_routed_experts,
            n_activated_experts=config.spatiotemporal_encoder_config.n_activated_experts,
            n_shared_experts=config.spatiotemporal_encoder_config.n_shared_experts,
            n_group=1,
            n_limited_groups=1,
            moe_intermediate_size=config.cross_modal_fusion_config.dim,
            score_func='sigmoid',
            route_scale=2.5
        )
    
    def forward(self, xyzt: torch.Tensor, 
                spatial_mask: Optional[torch.Tensor] = None,
                temporal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode spatiotemporal coordinates.
        
        Args:
            xyzt: (B, 4) normalized coordinates
            spatial_mask: (B,) bool mask for spatial coordinates
            temporal_mask: (B,) bool mask for temporal coordinate
        
        Returns:
            encoded: (B, D_fusion) encoded spatiotemporal features
        """
        # Apply coordinate masking
        if spatial_mask is not None or temporal_mask is not None:
            xyzt = xyzt.clone()
            if spatial_mask is not None:
                xyzt[~spatial_mask, :3] = 0
            if temporal_mask is not None:
                xyzt[~temporal_mask, 3] = 0
        
        # Extract coordinate components
        xyz = xyzt[..., :3]  # Spatial
        t = xyzt[..., 3:4]   # Temporal
        
        # Multi-resolution spatial encoding
        xyz_enc = self.grid4d.xyz_encoding(xyz)  # (B, spatial_dim)
        
        # Decomposed temporal encodings for better gradients
        xyt = torch.cat([xyzt[..., :2], t], dim=-1)  # x,y,t
        yzt = torch.cat([xyzt[..., 1:3], t], dim=-1)  # y,z,t
        xzt = torch.cat([xyzt[..., :1], xyzt[..., 2:3], t], dim=-1)  # x,z,t
        
        xyt_enc = self.grid4d.xyt_encoding(xyt)  # (B, temporal_dim)
        yzt_enc = self.grid4d.yzt_encoding(yzt)  # (B, temporal_dim)
        xzt_enc = self.grid4d.xzt_encoding(xzt)  # (B, temporal_dim)
        
        # Concatenate all encodings
        hash_features = torch.cat([xyz_enc, xyt_enc, yzt_enc, xzt_enc], dim=-1)
        # (B, spatial_dim + 3*temporal_dim)
        
        # Process through spatiotemporal transformer
        # Add sequence dimension
        features = hash_features.unsqueeze(1)  # (B, 1, D)
        
        # Project to transformer space
        features = self.input_projection(features)  # (B, 1, D_st)
        
        # Apply transformer
        features = self.spatiotemporal_transformer(features)  # (B, 1, D_st)
        
        # Project to fusion space
        features = self.output_projection(features)  # (B, 1, D_fusion)
        
        # Remove sequence dimension
        return features.squeeze(1)  # (B, D_fusion)


class ModalityEncoder(nn.Module):
    """Modality-specific encoder with configurable transformer.
    
    Processes individual modalities through a small transformer before
    cross-modal fusion. Supports custom configurations per modality.
    
    Args:
        modality_name: Name of the modality
        input_dim: Input dimension for this modality
        config: Main DeepEarthConfig
        encoder_config: Optional custom encoder configuration
    """
    
    def __init__(self, modality_name: str, input_dim: int, config: DeepEarthConfig,
                 encoder_config: Optional[ModalityEncoderConfig] = None):
        super().__init__()
        self.modality_name = modality_name
        self.input_dim = input_dim
        self.config = config
        
        # Use custom config if provided, otherwise default
        self.encoder_config = encoder_config or config.modality_encoder_config
        
        # Small transformer for this modality
        self.transformer = Transformer(self.encoder_config)
        
        # Input projection: modality space -> encoder space
        self.input_projection = self._build_projection(
            input_dim=input_dim,
            output_dim=self.encoder_config.dim,
            name='input'
        )
        
        # Output projection: encoder space -> fusion space
        self.output_projection = self._build_projection(
            input_dim=self.encoder_config.dim,
            output_dim=config.cross_modal_fusion_config.dim,
            name='output'
        )
    
    def _build_projection(self, input_dim: int, output_dim: int, name: str) -> nn.Module:
        """Build MoE projection head."""
        return MoE(
            dim=input_dim,
            n_heads=self.encoder_config.n_heads,
            head_dim=self.encoder_config.head_dim,
            n_routed_experts=self.encoder_config.n_routed_experts,
            n_activated_experts=self.encoder_config.n_activated_experts,
            n_shared_experts=self.encoder_config.n_shared_experts,
            n_group=1,
            n_limited_groups=1,
            moe_intermediate_size=output_dim,
            score_func='sigmoid',
            route_scale=1.0
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process modality data through encoder.
        
        Args:
            x: (B, D) or (B, 1, D) modality features
            mask: (B,) boolean mask, True = keep, False = mask
        
        Returns:
            encoded: (B, D_fusion) encoded features
        """
        # Ensure sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, D)
        
        # Project to encoder space
        x = self.input_projection(x)  # (B, 1, D_enc)
        
        # Apply transformer
        x = self.transformer(x)  # (B, 1, D_enc)
        
        # Project to fusion space
        x = self.output_projection(x)  # (B, 1, D_fusion)
        
        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).float()
            x = x * mask_expanded
        
        # Remove sequence dimension if single token
        return x.squeeze(1) if x.shape[1] == 1 else x


class ModalityDecoder(nn.Module):
    """Modality-specific decoder for reconstruction.
    
    Reconstructs original modality values from fused embeddings.
    Uses a small transformer to specialize the general representation
    back to modality-specific space.
    
    Args:
        modality_name: Name of the modality
        output_dim: Output dimension for reconstruction
        config: Main DeepEarthConfig
    """
    
    def __init__(self, modality_name: str, output_dim: int, config: DeepEarthConfig):
        super().__init__()
        self.modality_name = modality_name
        self.output_dim = output_dim
        self.config = config
        
        # Use same architecture as encoder for symmetry
        self.decoder_config = config.modality_encoder_config
        
        # Transformer for decoding
        self.transformer = Transformer(self.decoder_config)
        
        # Input projection: fusion space -> decoder space
        self.input_projection = self._build_projection(
            input_dim=config.cross_modal_fusion_config.dim,
            output_dim=self.decoder_config.dim,
            name='input'
        )
        
        # Output projection: decoder space -> modality space
        self.output_projection = self._build_projection(
            input_dim=self.decoder_config.dim,
            output_dim=output_dim,
            name='output'
        )
    
    def _build_projection(self, input_dim: int, output_dim: int, name: str) -> nn.Module:
        """Build MoE projection head."""
        return MoE(
            dim=input_dim,
            n_heads=self.decoder_config.n_heads,
            head_dim=self.decoder_config.head_dim,
            n_routed_experts=self.decoder_config.n_routed_experts,
            n_activated_experts=self.decoder_config.n_activated_experts,
            n_shared_experts=self.decoder_config.n_shared_experts,
            n_group=1,
            n_limited_groups=1,
            moe_intermediate_size=output_dim,
            score_func='sigmoid',
            route_scale=1.0
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode from fused representation.
        
        Args:
            x: (B, D_fusion) or (B, 1, D_fusion) fused features
        
        Returns:
            decoded: (B, D_out) reconstructed modality values
        """
        # Ensure sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, D)
        
        # Project from fusion space
        x = self.input_projection(x)  # (B, 1, D_dec)
        
        # Apply transformer
        x = self.transformer(x)  # (B, 1, D_dec)
        
        # Project to output space
        x = self.output_projection(x)  # (B, 1, D_out)
        
        # Remove sequence dimension
        return x.squeeze(1) if x.shape[1] == 1 else x


class SpatiotemporalDecoder(nn.Module):
    """Specialized decoder for spatiotemporal coordinates.
    
    Uses a dedicated transformer configuration optimized for
    coordinate reconstruction with higher capacity than standard
    modality decoders.
    
    Args:
        name: 'spatial' or 'temporal'
        output_dim: 3 for spatial (xyz), 1 for temporal (t)
        config: Main DeepEarthConfig
    """
    
    def __init__(self, name: str, output_dim: int, config: DeepEarthConfig):
        super().__init__()
        self.name = name
        self.output_dim = output_dim
        self.config = config
        
        # Use spatiotemporal-specific config
        self.decoder_config = config.spatiotemporal_encoder_config
        
        # Transformer for coordinate decoding
        self.transformer = Transformer(self.decoder_config)
        
        # Input projection: fusion space -> decoder space
        self.input_projection = self._build_projection(
            input_dim=config.cross_modal_fusion_config.dim,
            output_dim=self.decoder_config.dim,
            name='input'
        )
        
        # Output projection: decoder space -> coordinate space
        self.output_projection = self._build_projection(
            input_dim=self.decoder_config.dim,
            output_dim=output_dim,
            name='output'
        )
    
    def _build_projection(self, input_dim: int, output_dim: int, name: str) -> nn.Module:
        """Build MoE projection head."""
        return MoE(
            dim=input_dim,
            n_heads=self.decoder_config.n_heads,
            head_dim=self.decoder_config.head_dim,
            n_routed_experts=self.decoder_config.n_routed_experts,
            n_activated_experts=self.decoder_config.n_activated_experts,
            n_shared_experts=self.decoder_config.n_shared_experts,
            n_group=1,
            n_limited_groups=1,
            moe_intermediate_size=output_dim,
            score_func='sigmoid',
            route_scale=2.5  # Higher for coordinate precision
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode coordinates from fused representation.
        
        Args:
            x: (B, D_fusion) fused features
        
        Returns:
            coords: (B, 3) for spatial or (B, 1) for temporal
        """
        # Ensure sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Project from fusion space
        x = self.input_projection(x)
        
        # Apply transformer with dedicated capacity
        x = self.transformer(x)
        
        # Project to coordinate space
        x = self.output_projection(x)
        
        # Remove sequence dimension
        return x.squeeze(1)


class DeepEarthModel(nn.Module):
    """DeepEarth: Hierarchical Multimodal Transformer for Earth System Science
    
    This model implements a hierarchical architecture that processes arbitrary Earth
    observation data through three stages:
    1. Spatiotemporal encoding: Multi-resolution hash encoding of coordinates
    2. Modality-specific encoding: Small transformers process each data type
    3. Cross-modal fusion: Large transformer integrates all information
    
    The model learns through self-supervised reconstruction of masked inputs,
    enabling it to understand complex Earth system dynamics without labels.
    
    Args:
        config: DeepEarthConfig with all model specifications
    
    Example:
        >>> config = DeepEarthConfig()
        >>> config.modality_configs = {
        ...     'species': ModalityConfig(
        ...         name='species',
        ...         encoding_type='learned_embedding',
        ...         input_type='categorical',
        ...         column_name='species_name',
        ...         embed_dim=64
        ...     ),
        ...     'climate': ModalityConfig(
        ...         name='climate',
        ...         encoding_type='continuous_values',
        ...         input_type='numerical',
        ...         column_names=['temperature', 'precipitation']
        ...     )
        ... }
        >>> model = DeepEarthModel(config)
        >>> outputs, embeddings = model(batch)
    """
    
    def __init__(self, config: DeepEarthConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger('DeepEarth.Model')
        
        # Initialize components
        self._build_spatiotemporal_encoder()
        self._build_modality_processors()
        self._build_cross_modal_fusion()
        
        # Log model statistics
        self._log_model_info()
    
    def _build_spatiotemporal_encoder(self):
        """Build the spatiotemporal coordinate encoding system"""
        self.coord_encoder = Grid4DEncoder(self.config)
        
        # Dedicated decoders for coordinate reconstruction
        self.spatial_decoder = SpatiotemporalDecoder(
            'spatial', output_dim=3, config=self.config
        )
        self.temporal_decoder = SpatiotemporalDecoder(
            'temporal', output_dim=1, config=self.config
        )
    
    def _build_modality_processors(self):
        """Build encoders and decoders for each configured modality"""
        self.modality_encoders = nn.ModuleDict()
        self.modality_decoders = nn.ModuleDict()
        
        # Validate modality names
        reserved_names = {'spatial', 'temporal', 'xyz', 't'}
        for name in self.config.modality_configs:
            if name in reserved_names:
                raise ValueError(f"Modality name '{name}' is reserved. Please use a different name.")
        
        for modality_name, modality_config in self.config.modality_configs.items():
            # Determine input dimension
            if modality_config.encoding_type == 'learned_embedding':
                input_dim = modality_config.embed_dim
            elif modality_config.encoding_type == 'continuous_values':
                input_dim = len(modality_config.column_names)
            else:
                self.logger.warning(f"Unknown encoding type for {modality_name}, skipping")
                continue
            
            # Create encoder with optional custom config
            encoder_config = self.config.modality_encoder_config
            if modality_config.custom_encoder_config:
                # Override default settings with custom ones
                encoder_config = dataclasses_replace(
                    encoder_config,
                    **modality_config.custom_encoder_config
                )
            
            self.modality_encoders[modality_name] = ModalityEncoder(
                modality_name=modality_name,
                input_dim=input_dim,
                config=self.config,
                encoder_config=encoder_config
            )
            
            self.modality_decoders[modality_name] = ModalityDecoder(
                modality_name=modality_name,
                output_dim=input_dim,
                config=self.config
            )
    
    def _build_cross_modal_fusion(self):
        """Build the main cross-modal fusion network"""
        self.cross_modal_fusion = Transformer(self.config.cross_modal_fusion_config)
    
    def _log_model_info(self):
        """Log model configuration and statistics"""
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        self.logger.info(f"DeepEarth Model initialized:")
        self.logger.info(f"  Total parameters: {n_params:,}")
        self.logger.info(f"  Trainable parameters: {n_trainable:,}")
        self.logger.info(f"  Coordinate system: {self.config.spatial_coordinate_system}")
        self.logger.info(f"  Modalities: {list(self.config.modality_configs.keys())}")
        self.logger.info(f"  Grid4D resolutions: {len(self.config.spatial_resolutions)} spatial, "
                        f"{len(self.config.temporal_resolutions)} temporal")
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass through the hierarchical architecture.
        
        Args:
            batch: Dictionary containing:
                - xyzt: (B, 4) normalized spatiotemporal coordinates
                - modalities: Dict of modality_name -> (B, D) tensors
                - modality_masks: Dict of modality_name -> (B,) bool masks
                - internal_masks: Dict of modality_name -> (B, D) bool masks
                - ground_truth: Dict of ground truth values for reconstruction loss
        
        Returns:
            reconstructions: Dict of modality_name -> reconstructed values
            embeddings: (B, N, D) full sequence of embeddings from fusion network
        """
        if 'xyzt' not in batch:
            raise ValueError("Batch must contain 'xyzt' key with spatiotemporal coordinates")
        
        device = batch['xyzt'].device
        B = batch['xyzt'].shape[0]
        
        # Stage 1: Encode spatiotemporal coordinates
        coord_embeddings = self._encode_coordinates(batch)
        
        # Stage 2: Encode modalities with masking
        modality_embeddings = self._encode_modalities(batch)
        
        # Stage 3: Cross-modal fusion
        fused_embeddings = self._fuse_modalities(coord_embeddings, modality_embeddings)
        
        # Stage 4: Decode all modalities
        reconstructions = self._decode_all(fused_embeddings, batch)
        
        return reconstructions, fused_embeddings
    
    def _encode_coordinates(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode spatiotemporal coordinates with Grid4D.
        
        Args:
            batch: Input batch with 'xyzt' and optional masks
        
        Returns:
            coord_embeddings: (B, D) coordinate embeddings
        """
        coord_embeddings = self.coord_encoder(
            batch['xyzt'],
            spatial_mask=batch['modality_masks'].get('spatial'),
            temporal_mask=batch['modality_masks'].get('temporal')
        )
        return coord_embeddings
    
    def _encode_modalities(self, batch: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Encode each modality through its specific encoder.
        
        Args:
            batch: Input batch with modality data and masks
        
        Returns:
            modality_embeddings: List of (B, D) embeddings for each modality
        """
        modality_embeddings = []
        
        for modality_name, modality_data in batch['modalities'].items():
            if modality_name not in self.modality_encoders:
                self.logger.warning(f"No encoder for modality {modality_name}")
                continue
            
            # Apply internal feature-level masking if provided
            if modality_name in batch.get('internal_masks', {}):
                modality_data = modality_data * batch['internal_masks'][modality_name].float()
            
            # Encode with modality-level masking
            encoded = self.modality_encoders[modality_name](
                modality_data,
                mask=batch['modality_masks'].get(modality_name)
            )
            
            modality_embeddings.append(encoded)
        
        return modality_embeddings
    
    def _fuse_modalities(self, coord_embeddings: torch.Tensor, 
                        modality_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Fuse coordinate and modality embeddings through cross-modal transformer.
        
        Args:
            coord_embeddings: (B, D) spatiotemporal embeddings
            modality_embeddings: List of (B, D) modality embeddings
        
        Returns:
            fused_embeddings: (B, N, D) fused representation
        """
        # Build token sequence: [coordinates, modality_1, modality_2, ...]
        tokens = [coord_embeddings.unsqueeze(1)]
        
        for modality_embedding in modality_embeddings:
            if modality_embedding.dim() == 2:
                modality_embedding = modality_embedding.unsqueeze(1)
            tokens.append(modality_embedding)
        
        # Concatenate all tokens
        token_sequence = torch.cat(tokens, dim=1)  # (B, N, D)
        
        # Apply cross-modal fusion
        fused_embeddings = self.cross_modal_fusion(token_sequence)
        
        return fused_embeddings
    
    def _decode_all(self, embeddings: torch.Tensor, 
                    batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Decode all modalities from fused embeddings.
        
        Args:
            embeddings: (B, N, D) fused embeddings
            batch: Original batch for modality ordering
        
        Returns:
            reconstructions: Dict mapping modality names to reconstructed values
        """
        reconstructions = {}
        
        # Decode spatiotemporal coordinates from first token
        coord_token = embeddings[:, 0]
        reconstructions['xyz'] = self.spatial_decoder(coord_token)
        reconstructions['t'] = self.temporal_decoder(coord_token)
        
        # Decode each modality from its corresponding token
        token_idx = 1
        for modality_name in batch['modalities'].keys():
            if modality_name in self.modality_decoders and token_idx < embeddings.shape[1]:
                modality_token = embeddings[:, token_idx]
                reconstructions[modality_name] = self.modality_decoders[modality_name](
                    modality_token
                )
                token_idx += 1
        
        return reconstructions
    
    def get_modality_embedding(self, batch: Dict[str, torch.Tensor], 
                              modality_name: str) -> torch.Tensor:
        """Extract embedding for a specific modality.
        
        Useful for downstream tasks or analysis.
        
        Args:
            batch: Input batch
            modality_name: Name of modality to extract
        
        Returns:
            embedding: (B, D) embedding for the specified modality
        """
        with torch.no_grad():
            _, embeddings = self.forward(batch)
            
            # Find the token index for this modality
            token_idx = 1  # 0 is coordinates
            for name in batch['modalities'].keys():
                if name == modality_name:
                    return embeddings[:, token_idx]
                token_idx += 1
        
        raise ValueError(f"Modality {modality_name} not found in batch")
```

### 5. Training System

```python
class DeepEarthTrainer:
    """Training system with comprehensive metrics"""
    
    def __init__(self, model: DeepEarthModel, config: DeepEarthConfig, preprocessor: DataPreprocessor):
        self.model = model
        self.config = config
        self.preprocessor = preprocessor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger('DeepEarth.Trainer')
        
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        
        # Metrics
        self.metrics_history = defaultdict(list)
    
    def compute_losses(self, reconstructions, batch):
        """Compute dynamic losses for all modalities"""
        
        losses = {}
        
        # Spatial loss
        spatial_mask = ~batch['modality_masks']['spatial']
        if spatial_mask.any():
            losses['S'] = F.mse_loss(
                reconstructions['xyz'][spatial_mask],
                batch['ground_truth']['xyz'][spatial_mask]
            )
        else:
            losses['S'] = torch.tensor(0.0, device=self.device)
        
        # Temporal loss
        temporal_mask = ~batch['modality_masks']['temporal']
        if temporal_mask.any():
            losses['T'] = F.mse_loss(
                reconstructions['t'][temporal_mask],
                batch['ground_truth']['t'][temporal_mask]
            )
        else:
            losses['T'] = torch.tensor(0.0, device=self.device)
        
        # Modality losses
        for modality_name in batch['modalities']:
            mask = ~batch['modality_masks'][modality_name]
            if mask.any():
                pred = reconstructions[modality_name][mask]
                true = batch['ground_truth'][modality_name]
                
                if true.dtype == torch.long:
                    # Embedding reconstruction
                    true_emb = self.preprocessor.modality_stats[modality_name]['embeddings'](true[mask])
                    losses[modality_name[:2]] = F.mse_loss(pred, true_emb)
                else:
                    # Continuous reconstruction
                    losses[modality_name[:2]] = F.mse_loss(pred, true[mask])
            else:
                losses[modality_name[:2]] = torch.tensor(0.0, device=self.device)
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses
    
    def compute_metrics(self, reconstructions, batch):
        """Compute human-interpretable metrics"""
        
        metrics = {}
        
        # Spatial metrics
        if self.config.spatial_coordinate_system in ['geographic', 'ecef']:
            # Convert to meters for error reporting
            spatial_mask = ~batch['modality_masks']['spatial']
            if spatial_mask.any():
                pred = reconstructions['xyz'][spatial_mask]
                true = batch['ground_truth']['xyz'][spatial_mask]
                # Denormalize to get meter-scale errors
                scale = self.preprocessor.spatial_stats['scale']
                if isinstance(scale, torch.Tensor) and scale.numel() > 1:
                    error_m = (pred - true) * scale.unsqueeze(0)
                else:
                    error_m = (pred - true) * scale
                metrics['xyz_m'] = error_m.norm(dim=1).mean().item()
        else:
            # Already in meters
            spatial_mask = ~batch['modality_masks']['spatial']
            if spatial_mask.any():
                pred = reconstructions['xyz'][spatial_mask]
                true = batch['ground_truth']['xyz'][spatial_mask]
                metrics['xyz_m'] = (pred - true).norm(dim=1).mean().item()
        
        # Temporal metrics (convert to hours)
        temporal_mask = ~batch['modality_masks']['temporal']
        if temporal_mask.any():
            pred = reconstructions['t'][temporal_mask]
            true = batch['ground_truth']['t'][temporal_mask]
            error_norm = torch.abs(pred - true)
            error_hours = error_norm * self.preprocessor.temporal_stats['span_hours']
            metrics['t_h'] = error_hours.mean().item()
        
        # Modality-specific metrics
        for modality_name in batch['modalities']:
            mask = ~batch['modality_masks'][modality_name]
            if mask.any():
                pred = reconstructions[modality_name][mask]
                true = batch['ground_truth'][modality_name]
                
                if true.dtype == torch.long and modality_name in self.preprocessor.modality_stats:
                    # Convert to embeddings for comparison
                    true = self.preprocessor.modality_stats[modality_name]['embeddings'](true[mask])
                else:
                    true = true[mask]
                
                # Relative error
                diff = torch.abs(pred - true) / (torch.abs(true) + 1e-6)
                metrics[f'{modality_name[:2]}_m%'] = diff.mean().item() * 100
            
            # Unmasked accuracy (for monitoring)
            unmask = batch['modality_masks'][modality_name]
            if unmask.any():
                pred = reconstructions[modality_name][unmask]
                true = batch['ground_truth'][modality_name]
                
                if true.dtype == torch.long and modality_name in self.preprocessor.modality_stats:
                    true = self.preprocessor.modality_stats[modality_name]['embeddings'](true[unmask])
                else:
                    true = true[unmask]
                
                diff = torch.abs(pred - true) / (torch.abs(true) + 1e-6)
                metrics[f'{modality_name[:2]}_u%'] = diff.mean().item() * 100
        
        return metrics
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        
        self.model.train()
        epoch_losses = defaultdict(list)
        epoch_metrics = defaultdict(list)
        
        for step, batch_df in enumerate(train_loader):
            # Preprocess
            batch = self.preprocessor.preprocess_batch(batch_df, apply_masking=True)
            
            # Forward
            reconstructions, embeddings = self.model(batch)
            
            # Compute losses
            losses = self.compute_losses(reconstructions, batch)
            
            # Backward
            self.optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Track losses
            for k, v in losses.items():
                epoch_losses[k].append(v.item())
            
            # Compute metrics periodically
            if step % self.config.log_every_n_steps == 0:
                metrics = self.compute_metrics(reconstructions, batch)
                for k, v in metrics.items():
                    epoch_metrics[k].append(v)
        
        # Average metrics
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items() if v}
        
        # Concise logging
        loss_str = f"L:{avg_losses['total']:.3f}"
        for k, v in avg_losses.items():
            if k != 'total':
                loss_str += f" {k}:{v:.2f}"
        
        metric_str = ""
        for k, v in sorted(avg_metrics.items()):
            metric_str += f" {k}:{v:.1f}"
        
        self.logger.info(f"E{epoch:03d} | {loss_str} |{metric_str}")
        
        return avg_losses, avg_metrics
    
    def validate(self, val_loader):
        """Validate with targeted masking"""
        
        self.model.eval()
        results = defaultdict(list)
        
        with torch.no_grad():
            # Test each modality masked separately
            for mask_target in ['spatial', 'temporal'] + list(self.config.modality_configs.keys()):
                
                for batch_df in val_loader:
                    # Preprocess without random masking
                    batch = self.preprocessor.preprocess_batch(batch_df, apply_masking=False)
                    
                    # Mask only target
                    batch['modality_masks'][mask_target] = torch.zeros_like(
                        batch['modality_masks'][mask_target]
                    )
                    
                    # Forward
                    reconstructions, _ = self.model(batch)
                    
                    # Compute metrics
                    metrics = self.compute_metrics(reconstructions, batch)
                    for k, v in metrics.items():
                        results[f"{mask_target}_{k}"].append(v)
        
        # Average
        avg_results = {k: np.mean(v) for k, v in results.items() if v}
        
        return avg_results
    
    def visualize_embedding_field(self, test_df, modality_name, value_name, save_path):
        """Visualize model's understanding of ecological distributions"""
        
        self.model.eval()
        
        # Find observations
        column_name = self.config.modality_configs[modality_name].column_name
        obs_df = test_df[test_df[column_name] == value_name]
        
        if len(obs_df) < 3:
            return
        
        # Create query grid
        if self.config.spatial_coordinate_system == 'geographic':
            center_lat = obs_df['latitude'].mean()
            center_lon = obs_df['longitude'].mean()
            
            grid_size = self.config.viz_grid_size
            lat_range = self.config.viz_lat_range
            lon_range = self.config.viz_lon_range
            
            query_lats = np.linspace(center_lat - lat_range/2, center_lat + lat_range/2, grid_size)
            query_lons = np.linspace(center_lon - lon_range/2, center_lon + lon_range/2, grid_size)
            
            # Fixed time
            query_time = obs_df.iloc[0][['year', 'month', 'day', 'hour']]
            
            # Create queries
            query_data = []
            for lat in query_lats:
                for lon in query_lons:
                    row = {
                        'latitude': lat,
                        'longitude': lon,
                        'altitude': obs_df['altitude'].mean() if 'altitude' in obs_df else 0,
                        **query_time.to_dict()
                    }
                    # Add dummy modality value
                    row[column_name] = value_name
                    query_data.append(row)
        else:
            # Local XYZ visualization
            center_x = obs_df['x'].mean()
            center_y = obs_df['y'].mean()
            
            grid_size = self.config.viz_grid_size
            x_range = self.config.viz_lon_range * 111000  # Convert degrees to meters approx
            y_range = self.config.viz_lat_range * 111000
            
            query_xs = np.linspace(center_x - x_range/2, center_x + x_range/2, grid_size)
            query_ys = np.linspace(center_y - y_range/2, center_y + y_range/2, grid_size)
            
            # Create queries
            query_data = []
            for x in query_xs:
                for y in query_ys:
                    row = {
                        'x': x,
                        'y': y,
                        'z': obs_df['z'].mean()
                    }
                    if 'timestamp' in obs_df:
                        row['timestamp'] = obs_df['timestamp'].iloc[0]
                    else:
                        row.update(obs_df.iloc[0][['year', 'month', 'day', 'hour']].to_dict())
                    row[column_name] = value_name
                    query_data.append(row)
        
        query_df = pd.DataFrame(query_data)
        
        # Get predictions
        with torch.no_grad():
            batch = self.preprocessor.preprocess_batch(query_df, apply_masking=False)
            
            # Mask modality
            batch['modality_masks'][modality_name] = torch.zeros_like(
                batch['modality_masks'][modality_name]
            )
            
            reconstructions, _ = self.model(batch)
            
            # Compute distances
            vocab = self.preprocessor.modality_stats[modality_name]['vocab']
            target_id = vocab[value_name]
            target_emb = self.preprocessor.modality_stats[modality_name]['embeddings'](
                torch.tensor(target_id)
            )
            
            distances = torch.norm(reconstructions[modality_name] - target_emb.unsqueeze(0), dim=1)
            distances = distances.cpu().numpy().reshape(grid_size, grid_size)
        
        # Visualize
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if self.config.spatial_coordinate_system == 'geographic':
            im = ax.imshow(distances, extent=[query_lons[0], query_lons[-1], query_lats[0], query_lats[-1]],
                          origin='lower', cmap='viridis_r', alpha=0.8)
            
            ax.scatter(obs_df['longitude'], obs_df['latitude'], 
                      c='red', s=100, marker='o', edgecolors='white', linewidth=2,
                      label=f'{value_name} observations')
            
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
        else:
            im = ax.imshow(distances, extent=[query_xs[0], query_xs[-1], query_ys[0], query_ys[-1]],
                          origin='lower', cmap='viridis_r', alpha=0.8)
            
            ax.scatter(obs_df['x'], obs_df['y'], 
                      c='red', s=100, marker='o', edgecolors='white', linewidth=2,
                      label=f'{value_name} observations')
            
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')
        
        ax.set_title(f'Embedding Distance Field: {value_name}')
        ax.legend()
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Embedding Distance')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def create_train_test_splits(df: pd.DataFrame, config: DeepEarthConfig):
    """Create spatial and temporal validation splits"""
    
    # Temporal split
    if 'year' in df.columns:
        temporal_cutoff = config.temporal_bounds['end'].year - config.temporal_holdout_years
        temporal_test = df[df['year'] > temporal_cutoff]
        remaining = df[df['year'] <= temporal_cutoff]
    else:
        # Use timestamp
        cutoff_timestamp = config.temporal_bounds['end'].timestamp()
        temporal_test = df[df['timestamp'] >= cutoff_timestamp - 365*24*3600]
        remaining = df[df['timestamp'] < cutoff_timestamp - 365*24*3600]
    
    # Spatial split
    if config.spatial_coordinate_system == 'geographic':
        min_lat = remaining['latitude'].min()
        max_lat = remaining['latitude'].max()
        min_lon = remaining['longitude'].min()
        max_lon = remaining['longitude'].max()
        
        # Patch size
        area = (max_lat - min_lat) * (max_lon - min_lon)
        patch_area = area * config.validation_patch_size
        patch_size = math.sqrt(patch_area)
        
        # Generate patches
        np.random.seed(42)
        spatial_test_indices = []
        
        for i in range(config.n_spatial_holdout_patches):
            center_lat = np.random.uniform(min_lat + patch_size/2, max_lat - patch_size/2)
            center_lon = np.random.uniform(min_lon + patch_size/2, max_lon - patch_size/2)
            
            patch_mask = (
                (remaining['latitude'] >= center_lat - patch_size/2) &
                (remaining['latitude'] <= center_lat + patch_size/2) &
                (remaining['longitude'] >= center_lon - patch_size/2) &
                (remaining['longitude'] <= center_lon + patch_size/2)
            )
            
            spatial_test_indices.extend(remaining[patch_mask].index.tolist())
    else:
        # XYZ patches
        min_x = remaining['x'].min()
        max_x = remaining['x'].max()
        min_y = remaining['y'].min()
        max_y = remaining['y'].max()
        
        # Patch size
        area = (max_x - min_x) * (max_y - min_y)
        patch_area = area * config.validation_patch_size
        patch_size = math.sqrt(patch_area)
        
        # Generate patches
        np.random.seed(42)
        spatial_test_indices = []
        
        for i in range(config.n_spatial_holdout_patches):
            center_x = np.random.uniform(min_x + patch_size/2, max_x - patch_size/2)
            center_y = np.random.uniform(min_y + patch_size/2, max_y - patch_size/2)
            
            patch_mask = (
                (remaining['x'] >= center_x - patch_size/2) &
                (remaining['x'] <= center_x + patch_size/2) &
                (remaining['y'] >= center_y - patch_size/2) &
                (remaining['y'] <= center_y + patch_size/2)
            )
            
            spatial_test_indices.extend(remaining[patch_mask].index.tolist())
    
    spatial_test = remaining.loc[spatial_test_indices]
    train = remaining.drop(spatial_test_indices)
    
    return train, spatial_test, temporal_test
```


### 6. Main Execution

```python
def main():
    """DeepEarth Training Pipeline
    
    Complete example using GBIF biodiversity observations to demonstrate
    self-supervised learning of ecological patterns.
    """
    
    # Setup
    logger = setup_logging("INFO")
    logger.info("="*80)
    logger.info("DeepEarth: Multimodal Foundation Model for Earth System Science")
    logger.info("MVP Example: Central Florida Native Plants")
    logger.info("="*80)
    
    # 1. Configuration
    logger.info("\n1. Setting up configuration...")
    config = DeepEarthConfig()
    
    # Configure modalities
    config.modality_configs = {
        'species': ModalityConfig(
            name='species',
            encoding_type='learned_embedding',
            input_type='categorical',
            column_name='species_name',
            embed_dim=64
        )
    }
    
    # Optional: Set geographic bounds for outlier removal
    config.geographic_bounds = {
        'geojson_path': '/home/photon/deepearth/data/deepearth_v0.01/deepearth_v0.01_geographic_range.geojson'
    }
    
    # 2. Load and validate data
    logger.info("\n2. Loading biodiversity data...")
    csv_path = '/home/photon/deepearth/data/deepearth_v0.01/processed/gbif_central_florida_native_plants_v0.01_ml_ready.csv'
    
    data_loader = DatasetLoader(csv_path, list(config.modality_configs.values()), 
                               coordinate_system='geographic')
    df = data_loader.load_and_validate()
    
    summary = data_loader.get_data_summary()
    logger.info(f"  Observations: {summary['n_observations']:,}")
    logger.info(f"  Species: {summary['modalities']['species']['n_unique_values']} unique")
    logger.info(f"  Spatial: {summary['spatial_extent']['lat_range'][0]:.2f}° to {summary['spatial_extent']['lat_range'][1]:.2f}°")
    logger.info(f"  Temporal: {summary['temporal_extent']['year_range'][0]}-{summary['temporal_extent']['year_range'][1]}")
    
    # 3. Preprocess and compute statistics
    logger.info("\n3. Computing dataset statistics...")
    preprocessor = DataPreprocessor(config)
    df_clean = preprocessor.compute_dataset_statistics(df)
    
    logger.info(f"  Spatial span: {preprocessor.spatial_stats.get('span_km', preprocessor.spatial_stats.get('span_m', 0)):.1f} {'km' if 'span_km' in preprocessor.spatial_stats else 'm'}")
    logger.info(f"  Temporal span: {preprocessor.temporal_stats['span_hours']/24:.1f} days")
    
    # 4. Create validation splits
    logger.info("\n4. Creating validation splits...")
    train_df, spatial_test_df, temporal_test_df = create_train_test_splits(df_clean, config)
    logger.info(f"  Train: {len(train_df):,} | Spatial test: {len(spatial_test_df):,} | Temporal test: {len(temporal_test_df):,}")
    
    # 5. Create context-aware dataloaders
    logger.info("\n5. Setting up context-aware data loading...")
    sampling_engine = ContextSamplingEngine(train_df, preprocessor, config)
    train_loader = sampling_engine.create_dataloader(train_df, shuffle=True)
    
    val_sampling = ContextSamplingEngine(spatial_test_df, preprocessor, config)
    spatial_val_loader = val_sampling.create_dataloader(spatial_test_df, shuffle=False)
    
    # 6. Initialize model
    logger.info("\n6. Initializing DeepEarth model...")
    model = DeepEarthModel(config)
    
    # 7. Create trainer
    logger.info("\n7. Setting up training...")
    trainer = DeepEarthTrainer(model, config, preprocessor)
    
    # 8. Training loop
    logger.info("\n8. Starting self-supervised training...")
    
    best_val_loss = float('inf')
    
    for epoch in range(config.n_epochs):
        # Train
        train_losses, train_metrics = trainer.train_epoch(train_loader, epoch)
        
        # Validate
        if (epoch + 1) % config.validate_every_n_epochs == 0:
            val_metrics = trainer.validate(spatial_val_loader)
            
            # Log validation
            val_str = "Val |"
            for k, v in sorted(val_metrics.items()):
                val_str += f" {k}:{v:.1f}"
            logger.info(val_str)
            
            # Save best
            val_loss = sum(v for k, v in val_metrics.items() if '_m%' in k)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'config': config
                }, f'deepearth_best.pt')
        
        # Visualize
        if (epoch + 1) % config.visualize_every_n_epochs == 0:
            # Visualize embedding fields
            if 'species' in preprocessor.modality_stats:
                vocab = preprocessor.modality_stats['species']['inverse_vocab']
                value_name = vocab[min(5, len(vocab)-1)]
                
                os.makedirs('outputs', exist_ok=True)
                trainer.visualize_embedding_field(
                    spatial_test_df,
                    'species',
                    value_name,
                    f'outputs/embedding_field_e{epoch+1}.png'
                )
    
    # 9. Final evaluation
    logger.info("\n9. Final evaluation...")
    temporal_sampling = ContextSamplingEngine(temporal_test_df, preprocessor, config)
    temporal_test_loader = temporal_sampling.create_dataloader(temporal_test_df, shuffle=False)
    temporal_metrics = trainer.validate(temporal_test_loader)
    
    logger.info("\nTemporal generalization:")
    for k, v in sorted(temporal_metrics.items()):
        if '_m%' in k or 'xyz_m' in k or 't_h' in k:
            logger.info(f"  {k}: {v:.1f}")
    
    logger.info("\nTraining complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
```

## Key Features

1. **Universal Architecture**: Handles any modality without code changes
2. **Context-Rich Learning**: Samples spatially, temporally, and ecologically related observations
3. **Hierarchical Processing**: Small transformers for modalities, large for fusion
4. **Self-Supervised**: Learns from reconstruction without labels
5. **Production Ready**: Comprehensive logging, validation, and visualization

## Getting Started

### 1. Install Dependencies

```bash
pip install torch pandas numpy geopandas matplotlib seaborn faiss-cpu
```

### 2. Prepare Your Data

DeepEarth supports flexible data formats with multiple coordinate systems.

#### Geographic Coordinates Example:
```csv
latitude,longitude,altitude,year,month,day,hour,species_name
28.123,-81.456,-2.5,2023,3,15,14,Quercus virginiana
28.234,-81.567,12.0,2023,3,16,10,Pinus elliottii
...
```

#### ECEF Coordinates Example:
```csv
x,y,z,year,month,day,hour,species_name
924842.3,-5532482.1,2998234.2,2023,3,15,14,Quercus virginiana
...
```

#### Local XYZ Coordinates Example:
```csv
x,y,z,timestamp,species_name
1000.0,2000.0,-5.0,1678896000.123456,Quercus virginiana
...
```

### 3. Configure Modalities

```python
# Example configurations for different data types:

# Biodiversity observations (categorical → learned embeddings)
species_config = ModalityConfig(
    name='species',
    encoding_type='learned_embedding',
    input_type='categorical',
    column_name='species_name',
    embed_dim=64
)

# Climate data (numerical → continuous values)
climate_config = ModalityConfig(
    name='climate',
    encoding_type='continuous_values',
    input_type='numerical',
    column_names=['temperature', 'precipitation', 'humidity']
)

# Spectral data (numerical → learned embeddings)
spectral_config = ModalityConfig(
    name='spectral',
    encoding_type='learned_embedding',
    input_type='numerical',
    column_names=[f'band_{i}' for i in range(1, 13)],
    embed_dim=128
)

# Land cover (categorical → learned embeddings)
landcover_config = ModalityConfig(
    name='landcover',
    encoding_type='learned_embedding',
    input_type='categorical',
    column_name='landcover_class',
    embed_dim=32
)
```

### 4. Run Training

```python
# Initialize configuration
config = DeepEarthConfig()
config.spatial_coordinate_system = 'geographic'  # or 'ecef' or 'local_xyz'
config.modality_configs = {
    'species': species_config,
    'climate': climate_config
}

# Load data
loader = DatasetLoader('your_data.csv', [species_config, climate_config], 
                      coordinate_system=config.spatial_coordinate_system)
df = loader.load_and_validate()

# Run the full pipeline
python deepearth_architecture.py
```

### 5. Add New Modalities

DeepEarth's architecture makes it easy to integrate new data sources:

1. **Satellite Imagery**: Process through external encoder (e.g., SatMAE), add as continuous modality
2. **Audio Data**: Extract spectrograms, add as continuous modality  
3. **Soil Properties**: Direct continuous measurements
4. **Human Activity**: Categorical or continuous depending on representation
5. **Underwater/Underground Data**: Use negative altitude values

The model will automatically learn cross-modal relationships through the self-supervised objective.

## Example Datasets

### Central Florida Native Plants (Included)
- **Source**: GBIF (Global Biodiversity Information Facility)
- **Observations**: ~50,000 native plant observations
- **Species**: ~500 unique species
- **Timespan**: 2010-2025
- **Area**: Central Florida watersheds
- **Altitude range**: -10m to 100m (includes wetlands)

## Advanced Configuration Examples

### Custom Modality Configurations

```python
# Example 1: High-resolution species embeddings with larger encoder
species_config = ModalityConfig(
    name='species',
    encoding_type='learned_embedding',
    input_type='categorical',
    column_name='species_name',
    embed_dim=128,
    custom_encoder_config={
        'n_layers': 2,  # Deeper encoder for complex taxonomy
        'dim': 256,     # Larger hidden dimension
        'n_heads': 8
    }
)

# Example 2: Multi-scale climate data
climate_config = ModalityConfig(
    name='climate',
    encoding_type='continuous_values',
    input_type='numerical',
    column_names=['temp_c', 'precip_mm', 'humidity', 'wind_speed'],
    custom_encoder_config={
        'n_routed_experts': 8,  # More experts for diverse climate patterns
        'n_activated_experts': 3
    }
)

# Example 3: Hyperspectral imagery with dimensionality reduction
hyperspectral_config = ModalityConfig(
    name='hyperspectral',
    encoding_type='learned_embedding',
    input_type='numerical',
    column_names=[f'band_{i}' for i in range(224)],  # Hyperspectral bands
    embed_dim=256,  # Compress 224 bands to 256D
    custom_encoder_config={
        'dim': 512,  # Larger capacity for spectral processing
        'moe_intermediate_size': 1024
    }
)
```

### Spatiotemporal Resolution Configuration

```python
# Fine-grained urban monitoring
urban_config = DeepEarthConfig(
    spatial_resolutions=[0.1, 1.0, 10.0, 100.0],  # 10cm to 100m
    temporal_resolutions=[
        3600.0,      # 1 hour - diurnal patterns
        86400.0,     # 1 day - daily cycles
        604800.0,    # 1 week - weekly patterns
        2592000.0    # 30 days - monthly changes
    ]
)

# Continental-scale ecosystem monitoring
ecosystem_config = DeepEarthConfig(
    spatial_resolutions=[10.0, 100.0, 1000.0, 10000.0],  # 10m to 10km
    temporal_resolutions=[
        86400.0,       # 1 day - weather
        604800.0,      # 1 week - phenology
        2592000.0,     # 30 days - seasonal
        31536000.0     # 1 year - annual cycles
    ]
)
```

### Multi-Scale Deployment

```python
# Initialize model with mixed modalities
config = DeepEarthConfig()
config.modality_configs = {
    'species': species_config,
    'climate': climate_config,
    'hyperspectral': hyperspectral_config,
    'landcover': ModalityConfig(
        name='landcover',
        encoding_type='learned_embedding',
        input_type='categorical',
        column_name='lc_class',
        embed_dim=32
    ),
    'soil': ModalityConfig(
        name='soil',
        encoding_type='continuous_values',
        input_type='numerical',
        column_names=['ph', 'nitrogen', 'phosphorus', 'organic_matter']
    )
}

# Create model
model = DeepEarthModel(config)

# The model automatically:
# - Creates specialized encoders for each modality
# - Configures Grid4D for the specified resolutions  
# - Builds cross-modal fusion network
# - Sets up reconstruction decoders
```

## Conclusion

DeepEarth represents a new paradigm in Earth system modeling—one that learns directly from the planet's data to understand its complex dynamics across all scales and modalities. Its hierarchical architecture, flexible data integration, and self-supervised learning approach enable it to discover patterns and relationships that traditional models might miss.

The key innovations include:
- **Universal modality integration** without architectural changes
- **Multi-resolution spatiotemporal encoding** via Grid4D
- **Hierarchical processing** with specialized and fusion transformers
- **Context-aware learning** through intelligent sampling
- **Production-ready implementation** with comprehensive tooling

As we continue to gather more diverse Earth observation data, DeepEarth provides the foundation for building truly comprehensive models of our planet's interconnected systems.

