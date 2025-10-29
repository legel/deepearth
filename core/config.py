# deepearth/core/config.py
"""
DeepEarth Configuration System
═════════════════════════════

The configuration architecture orchestrates all hyperparameters and structural
decisions for the planetary-scale learning system. Each parameter has been 
carefully selected to balance computational efficiency with the expressive
power needed to model Earth's complex spatiotemporal dynamics.

Configuration Flow:
    YAML/Dict → DeepEarthConfig → Model Architecture
                              ↓
                        Data Pipeline
                              ↓
                        Training Loop
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Union, Any
import yaml
import json


@dataclass
class ModalityConfig:
    """
    Configuration for a specific observation modality.
    
    Each modality represents a distinct type of Earth observation data
    (e.g., visual imagery, spectral signatures, weather measurements).
    The configuration defines how positional information is encoded
    within that modality's context.
    
    Examples:
        Visual imagery: position_shape=[16, 16] for 256 patches
        Time series: position_shape=[128] for sequence positions
        Point clouds: position_shape=[1024] for spatial points
    """
    name: str                           # Human-readable modality identifier
    encoder_name: str                    # Associated encoder model
    position_dim: int = 0               # Dimensions for position encoding
    position_shape: List[int] = field(default_factory=list)  # Shape of position grid
    max_tokens: int = 128               # Maximum tokens per observation
    
    def get_position_cardinality(self) -> int:
        """Calculate total number of unique positions in this modality."""
        if not self.position_shape:
            return 0
        cardinality = 1
        for dim in self.position_shape:
            cardinality *= dim
        return cardinality


@dataclass
class SamplingStrategy:
    """
    Configuration for context window sampling during training.
    
    The sampling strategy determines how training examples are grouped
    into context windows, balancing different aspects of similarity:
    temporal (time of day, season, historical period), spatial (geographic
    proximity), and semantic (modality-specific features).
    
    Weights should sum to 1.0 for interpretability as probability distribution.
    """
    clusters_per_context: int = 4      # Number of focal points per context
    samples_per_cluster: int = 8       # Samples around each focal point
    
    # Temporal sampling weights
    time_of_day_weight: float = 0.1    # Diurnal patterns
    time_of_year_weight: float = 0.1   # Seasonal patterns
    time_of_history_weight: float = 0.1 # Historical trends
    
    # Spatial sampling weight
    spatial_weight: float = 0.2         # Geographic proximity
    
    # Semantic sampling weights
    modality_weight: float = 0.3        # Within-modality similarity
    universal_weight: float = 0.2       # Cross-modal similarity
    
    # Sampling behavior
    sampling_type: str = "contiguous"   # "contiguous" or "probabilistic"
    
    def validate(self):
        """Ensure weights sum to 1.0 for probabilistic interpretation."""
        total = (self.time_of_day_weight + self.time_of_year_weight + 
                self.time_of_history_weight + self.spatial_weight + 
                self.modality_weight + self.universal_weight)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Sampling weights sum to {total}, expected 1.0")


@dataclass
class DeepEarthConfig:
    """
    Master Configuration for DeepEarth World Model
    ═══════════════════════════════════════════════
    
    Architecture Overview:
    
    ┌─────────────────────────────────────────────────────────────┐
    │                      Universal Token (1024D)                 │
    ├─────────────────┬────────────┬──────────┬────────┬─────────┤
    │  Spacetime      │    Data    │ Metadata │  Mask  │Position │
    │    (512D)       │   (507D)   │   (7D)   │  (4D)  │  (4D)   │
    └─────────────────┴────────────┴──────────┴────────┴─────────┘
    
    The universal token unifies heterogeneous Earth observations into
    a common representation space where patterns can be discovered
    across modalities, locations, and time.
    """
    
    # ══════════════════════════════════════════════════════════════
    # Data Pipeline Configuration
    # ══════════════════════════════════════════════════════════════
    
    input_csv: str = None               # Path to input observations
    output_dir: str = "./deepearth_output"
    cache_dir: str = "./deepearth_cache"
    regenerate_cache: bool = False      # Force cache regeneration
    
    # ══════════════════════════════════════════════════════════════
    # Neural Architecture Dimensions
    # ══════════════════════════════════════════════════════════════
    
    # Total token dimension (power of 2 for hardware efficiency)
    universal_dim: int = 1024
    
    # Component dimensions that sum to universal_dim
    spacetime_dim: int = 512           # Earth4D spatiotemporal encoding
    # data_dim computed as remainder
    dataset_embed_dim: int = 2          # Dataset identifier embedding
    modality_embed_dim: int = 2         # Modality type embedding  
    encoder_embed_dim: int = 2          # Encoder model embedding
    mask_embed_dim: int = 4             # Mask pattern embedding (2^5 patterns)
    
    # Positional encoding dimensions (optional)
    context_position_dim: int = 2       # Position in context window [0, context_window)
    modality_position_dim: int = 2      # Position within modality observation
    
    # ══════════════════════════════════════════════════════════════
    # Modality Definitions
    # ══════════════════════════════════════════════════════════════
    
    modalities: Dict[str, ModalityConfig] = field(default_factory=lambda: {
        "visual": ModalityConfig(
            name="visual",
            encoder_name="BioCLIP",
            position_dim=2,
            position_shape=[16, 16],  # 256 patches for vision transformer
            max_tokens=256
        ),
        "spectral": ModalityConfig(
            name="spectral", 
            encoder_name="SpectralNet",
            position_dim=1,
            position_shape=[64],      # 64 spectral bands
            max_tokens=64
        ),
        "weather": ModalityConfig(
            name="weather",
            encoder_name="WeatherNet",
            position_dim=0,           # No positional encoding needed
            position_shape=[],
            max_tokens=1
        )
    })
    
    # ══════════════════════════════════════════════════════════════
    # Attention Architecture (Perceiver)
    # ══════════════════════════════════════════════════════════════

    context_window: int = 128           # Maximum sequence length

    # Main Perceiver configuration (heavy compute for multimodal fusion)
    num_latents: int = 256              # Latent bottleneck size
    latent_dim: int = 512               # Latent vector dimension
    num_blocks: int = 8                 # Depth of iterative processing
    num_cross_attention_heads: int = 8  # Parallel attention paths (input → latent)
    num_self_attention_heads: int = 8   # Parallel attention paths (latent → latent)
    dropout: float = 0.1                # Regularization strength

    # Lightweight PerceiverProjector configuration (for modality translation)
    # These are minimal "translators" that convert each expert's language to universal tokens
    projector_config: Dict = field(default_factory=lambda: {
        'num_latents': 64,              # Much smaller than main perceiver
        'latent_dim': 64,               # Lightweight intermediate representation
        'num_blocks': 2,                # Just 2 blocks for simple projection
        'num_cross_attention_heads': 2,  # Minimal attention heads
        'num_self_attention_heads': 2,   # Minimal self-attention
        'dropout': 0.1
    })
    
    # Cardinality limits for categorical variables
    max_datasets: int = 100
    max_modalities: int = 100  
    max_encoders: int = 100
    
    # ══════════════════════════════════════════════════════════════
    # Training Configuration
    # ══════════════════════════════════════════════════════════════
    
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 100
    gradient_clip: float = 1.0
    
    # Performance optimizations
    mixed_precision: bool = True        # FP16 computation
    compile_model: bool = True          # torch.compile() optimization
    
    # ══════════════════════════════════════════════════════════════
    # Self-Supervised Learning (Masking)
    # ══════════════════════════════════════════════════════════════
    
    # Component masking probabilities for self-supervised learning
    mask_spacetime_prob: float = 0.15   # Predict location/time from data
    mask_data_prob: float = 0.15        # Predict data from context
    mask_dataset_prob: float = 0.05     # Predict data source
    mask_modality_prob: float = 0.05    # Predict observation type
    mask_encoder_prob: float = 0.05     # Predict encoding method
    
    # Modality-specific masking (overrides global probabilities)
    modality_mask_probs: Dict[str, float] = None
    
    # ══════════════════════════════════════════════════════════════
    # Evaluation Strategy
    # ══════════════════════════════════════════════════════════════
    
    test_ratio: float = 0.2             # Fraction for random holdout
    
    # Spatial holdout regions for testing geographic generalization
    spatial_holdouts: List[Dict] = field(default_factory=lambda: [
        {
            "type": "percentage",        # Region as % of total area
            "lat_pct": 0.01,            # 1% of latitude range
            "lon_pct": 0.01,            # 1% of longitude range  
            "elev_pct": 1.0,            # Full elevation range
            "min_separation_pct": 0.1    # Minimum 10% separation between regions
        }
    ])
    
    # Temporal holdout periods for testing future prediction
    temporal_holdouts: List[Dict] = field(default_factory=lambda: [
        {
            "type": "percentage",        # Period as % of time range
            "value": 0.1,               # Last 10% of data
            "position": "end",          # "end" or "start"
            "min_separation_pct": 0.1    # Minimum 10% separation
        }
    ])
    
    # ══════════════════════════════════════════════════════════════
    # Coordinate Systems
    # ══════════════════════════════════════════════════════════════
    
    coordinate_system: str = "ecef"     # "ecef", "latlon", or "data_normalized"
    
    # Temporal normalization
    time_range_mode: str = "fixed"      # "fixed" or "data_normalized"
    time_range: Tuple[Union[int, str], Union[int, str]] = (1900, 2100)
    
    # Spatial normalization  
    spatial_range_mode: str = "fixed"   # "fixed" or "data_normalized"
    spatial_range: Dict = field(default_factory=lambda: {
        "lat": [-90, 90],
        "lon": [-180, 180], 
        "elev": [-11000, 9000]          # Mariana Trench to Everest
    })
    
    # ══════════════════════════════════════════════════════════════
    # Context Sampling Configuration
    # ══════════════════════════════════════════════════════════════
    
    sampling_strategy: SamplingStrategy = field(default_factory=SamplingStrategy)
    
    # UMAP dimensionality reduction for efficient similarity search
    umap_dim: int = 1                   # Target dimension for UMAP
    umap_max_samples: int = 1000000     # Max samples for UMAP training
    umap_n_neighbors: int = 15          # UMAP neighborhood size
    umap_min_dist: float = 0.1          # UMAP minimum distance
    
    # ══════════════════════════════════════════════════════════════
    # Hardware Configuration  
    # ══════════════════════════════════════════════════════════════
    
    device: str = "cuda"
    num_workers: int = 4                # DataLoader workers
    pin_memory: bool = True             # Fast CPU→GPU transfer
    
    # ══════════════════════════════════════════════════════════════
    # Reproducibility
    # ══════════════════════════════════════════════════════════════
    
    seed: int = 42
    
    # Additional configuration
    spatial_dims: int = 3               # Detected from data
    earth4d_config: Dict = field(default_factory=dict)
    
    @property
    def data_dim(self) -> int:
        """
        Calculate remaining dimensions for data representation.
        
        After allocating dimensions for spacetime, metadata, masking,
        and positional encodings, the remainder is used for the actual
        observation data projected into universal space.
        """
        allocated = (self.spacetime_dim + 
                    self.dataset_embed_dim + 
                    self.modality_embed_dim + 
                    self.encoder_embed_dim +
                    self.mask_embed_dim +
                    self.context_position_dim +
                    self.modality_position_dim)
        remaining = self.universal_dim - allocated
        
        if remaining <= 0:
            raise ValueError(
                f"No dimensions left for data! "
                f"Universal: {self.universal_dim}, Allocated: {allocated}"
            )
        
        return remaining
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate sampling strategy
        self.sampling_strategy.validate()
        
        # Validate dimension allocation
        _ = self.data_dim  # Triggers validation
        
        # Validate modality configurations
        for name, modality in self.modalities.items():
            if modality.position_dim > 0 and not modality.position_shape:
                raise ValueError(
                    f"Modality '{name}' has position_dim={modality.position_dim} "
                    f"but no position_shape defined"
                )
    
    @classmethod
    def from_yaml(cls, path: str):
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Handle modality configurations
        if 'modalities' in config_dict:
            modalities = {}
            for name, mod_dict in config_dict['modalities'].items():
                modalities[name] = ModalityConfig(**mod_dict)
            config_dict['modalities'] = modalities
        
        # Handle sampling strategy
        if 'sampling_strategy' in config_dict:
            config_dict['sampling_strategy'] = SamplingStrategy(**config_dict['sampling_strategy'])
        
        return cls(**config_dict)
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        # Convert dataclasses to dicts for serialization
        config_dict = self.__dict__.copy()
        
        # Convert modalities
        if 'modalities' in config_dict:
            modalities = {}
            for name, modality in config_dict['modalities'].items():
                modalities[name] = modality.__dict__
            config_dict['modalities'] = modalities
        
        # Convert sampling strategy
        if 'sampling_strategy' in config_dict:
            config_dict['sampling_strategy'] = config_dict['sampling_strategy'].__dict__
        
        # Convert non-serializable objects to strings
        def convert_to_serializable(obj):
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            else:
                return str(obj)

        config_dict = convert_to_serializable(config_dict)

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
