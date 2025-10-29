# DeepEarth: Multimodal Probabilistic World Model with 4D Spacetime Embedding

A planetary-scale neural architecture for understanding Earth through unified multimodal observation learning.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Core Components](#core-components)
- [Data Pipeline](#data-pipeline)
- [Training System](#training-system)
- [Inference Engine](#inference-engine)
- [Advanced Features](#advanced-features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Performance Optimization](#performance-optimization)
- [API Reference](#api-reference)

## Overview

DeepEarth is a revolutionary approach to planetary intelligence that learns to understand Earth observations across all scales of space and time. Unlike traditional models that process different data types separately, DeepEarth creates a unified representation space where patterns can be discovered across modalities, locations, and temporal scales.

### Recent Updates (September 29, 2024)

**First End-to-End Training Results**: The full DeepEarth architecture is now successfully training with the complete "Symphony of Experts" design, featuring:
- Dual-loss architecture: Universal space (256D) and modality-specific reconstruction
- Lightweight PerceiverProjectors (16 latents, 32D) for each encoder with bidirectional mapping
- Earth4D direct integration (162D native output, skip_projection mode)
- Real-time reconstruction visualization showing original vs predicted with Δ and MAPE
- Comprehensive gradient flow monitoring confirming all components are learning
- See [hello_deepearth_training.txt](../hello_deepearth_training.txt) for training logs

### Key Innovations

1. **Universal Token Representation**: A 1024-dimensional token that unifies heterogeneous Earth observations
2. **4D Spacetime Embedding**: Trainable Earth4D encoder (70% of parameters) that learns spatiotemporal patterns
3. **Perceiver-Based Multimodal Fusion**: Bidirectional projections between modality-specific and universal spaces
4. **UMAP-Indexed Context Sampling**: Intelligent selection of related observations for training
5. **Component-Wise Self-Supervision**: Learning through masked reconstruction of observation components

### Supported Data Types

- **Spatial Coordinates**: Latitude, longitude, elevation (WGS84 or ECEF)
- **Temporal Information**: Timestamps with cyclic decomposition
- **Multimodal Embeddings**: Vision (BioCLIP), environmental (AlphaEarth), text (species descriptions)
- **Categorical Metadata**: Dataset sources, modality types, encoder identities

## Architecture

### Universal Token Structure

The heart of DeepEarth is the universal token, a carefully designed 1024-dimensional representation:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Universal Token (1024D)                            │
├────────────────┬───────────────┬──────────────┬──────────┬─────────────────┤
│   Spacetime    │     Data      │   Metadata   │   Mask   │    Position     │
│     (512D)     │    (500D)     │     (6D)      │   (6D)   │      (0D)       │
├────────────────┼───────────────┼──────────────┼──────────┼─────────────────┤
│                │               │ ┌──────────┐ │          │ ┌─────────────┐ │
│   Earth4D      │  Multimodal   │ │ Dataset  │ │  Mask    │ │   Context   │ │
│   Encoding     │   Fusion      │ │ Modality │ │  Pattern │ │   Position  │ │
│                │   Output      │ │ Encoder  │ │          │ │             │ │
│                │               │ └──────────┘ │          │ └─────────────┘ │
└────────────────┴───────────────┴──────────────┴──────────┴─────────────────┘
```

### Model Architecture

```
Input Observations
       ↓
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Earth4D    │     │  Multimodal  │     │   Metadata   │
│   Encoder    │     │    Fusion    │     │  Embeddings  │
└──────────────┘     └──────────────┘     └──────────────┘
       ↓                     ↓                     ↓
       └─────────────────────┴─────────────────────┘
                             ↓
                    ┌──────────────┐
                    │  Universal   │
                    │    Tokens    │
                    └──────────────┘
                             ↓
                    ┌──────────────┐
                    │   Perceiver  │
                    │    Encoder   │ ← Iterative Refinement
                    └──────────────┘
                             ↓
                    ┌──────────────┐
                    │    Latent    │
                    │    States    │
                    └──────────────┘
                             ↓
                    ┌──────────────┐
                    │   Perceiver  │
                    │    Decoder   │
                    └──────────────┘
                             ↓
                    ┌──────────────┐
                    │ Predictions  │
                    └──────────────┘
```

## Core Components

### 1. Configuration System (`config.py`)

The configuration system uses dataclasses for type-safe, hierarchical configuration management:

- **DeepEarthConfig**: Master configuration with all hyperparameters
- **ModalityConfig**: Modality-specific settings including position encoding shapes
- **SamplingStrategy**: Weights for different similarity dimensions in context sampling

Key configuration parameters:
- **Dimension Allocation**: Precise control over how the 1024D token is divided
- **Architecture Settings**: Perceiver blocks, attention heads, latent dimensions
- **Training Strategy**: Learning rates, masking probabilities, batch sizes
- **Sampling Weights**: Balance between temporal, spatial, and semantic similarity

### 2. Data Preprocessor (`preprocessor.py`)

Sophisticated data transformation pipeline with multiple stages:

#### Coordinate Transformations
- **Geographic to ECEF**: Converts lat/lon/elevation to Earth-Centered, Earth-Fixed coordinates
- **ECEF Normalization**: Projects to unit sphere for bounded neural inputs
- **Temporal Decomposition**: Splits time into cyclic components (day, year, historical)

#### Column Detection
Intelligent pattern matching for automatic column mapping:
- Spatial patterns: x/lat/latitude, y/lon/longitude, z/elev/elevation
- Temporal patterns: t/time/timestamp/datetime
- Metadata detection: dataset, modality, encoder identifiers

#### Caching System
MD5-based cache invalidation considering:
- File contents
- Configuration parameters
- Modality definitions

### 3. Sampling Engine (`sampling.py`)

UMAP-based intelligent context window construction:

#### Index Types
- **Temporal Indices**: Time of day, season, historical period
- **Spatial Index**: 3D Euclidean distance in normalized space
- **Modality Indices**: UMAP projections per encoder
- **Universal Index**: Combined UMAP across all dimensions

#### Sampling Strategies
- **Contiguous**: Adjacent samples in similarity space
- **Probabilistic**: Random sampling within similarity bins
- **Hierarchical**: Cluster centers with local neighborhoods

### 4. Multimodal Fusion (`multimodal_fusion.py`)

Perceiver-based bidirectional projection networks:

#### Architecture Adaptation
- Small encoders (<128D): 16 latents, 1 block
- Medium encoders (<512D): 32 latents, 1 block  
- Large encoders (≥512D): 64 latents, 2 blocks

#### Projection Process
```
Modality Space → Input Projection → Cross-Attention with Latents
              → Self-Attention (Refinement) → Output Projection → Universal Space
```

### 5. DeepEarth Perceiver (`perceiver.py`)

The core model implementing iterative attention-based processing:

#### Component Masking
Binary mask patterns (5 bits = 32 patterns):
```
┌────────┬──────────┬─────────┬────────┬───────────┐
│ Bit 4  │  Bit 3   │  Bit 2  │ Bit 1  │   Bit 0   │
│Encoder │ Modality │ Dataset │  Data  │ Spacetime │
└────────┴──────────┴─────────┴────────┴───────────┘
```

#### Loss Components
Weighted reconstruction losses:
- Spacetime: 1.0 (high priority)
- Data: 1.0 (high priority)
- Dataset: 0.1 (auxiliary)
- Modality: 0.1 (auxiliary)
- Encoder: 0.1 (auxiliary)

### 6. Training Infrastructure (`trainer.py`)

State-of-the-art training pipeline with:

#### Mixed Precision Training
- Automatic mixed precision (AMP) with GradScaler
- FP16 computation with FP32 master weights
- Dynamic loss scaling for numerical stability

#### Learning Rate Scheduling
- OneCycleLR with cosine annealing
- Configurable warmup period
- Automatic adjustment based on dataset size

#### Checkpointing System
- Best model selection based on validation loss
- Regular checkpoints every N epochs
- Metric history tracking in JSON format

#### Progress Reporting
- Real-time loss tracking with TQDM
- Component-wise loss breakdown
- Gradient norm monitoring
- Learning rate visualization

### 7. Inference Engine (`inference.py`)

Production-ready inference system:

#### Query Interfaces
- **Single Location**: Predict at specific (x, y, z, t)
- **Batch Processing**: Efficient multi-sample inference
- **Cross-Modal**: Translate between modalities

#### Masking Specifications
Flexible control over what to predict:
- CSV-based masks for fine-grained control
- Dictionary specifications for programmatic use
- Default patterns for common tasks

#### Output Formats
- JSON for web APIs
- PyTorch tensors for further processing
- NumPy arrays for scientific computing

## Data Pipeline

### Input Data Structure

DeepEarth accepts data in multiple formats:

1. **Direct CSV**: Columns for coordinates, metadata, and data values
2. **CSV with File References**: External embeddings in .pt or .npy files
3. **Preprocessed Dictionaries**: Direct tensor inputs

### Processing Flow

```
Raw Data → Column Detection → Coordinate Transformation → Temporal Decomposition
        ↓                                                              ↓
Metadata Encoding ← Tensor Packing ← Embedding Organization ← Feature Extraction
        ↓
Cache Storage → UMAP Indexing → Context Sampling → DataLoader
```

### Angiosperm Dataset Bridge

Specialized loader for the angiosperm flowering dataset (`angiosperm_bridge.py`):

#### Supported Combinations
- (D1, M1, E1): AlphaEarth environmental embeddings
- (D1, M1, E2): Earth4D spatiotemporal embeddings
- (D2, M2, E3): BioCLIP vision embeddings
- (D2, M3, E3): BioCLIP text embeddings
- (D2, M2, E4): PhenoVision flowering predictions

#### Error Handling
- Validates all embedding indices
- Checks datetime parsing
- Ensures valid value ranges
- Verifies encoder combinations

## Training System

### Self-Supervised Learning

The model learns through masked component prediction:

1. **Spacetime → Data**: Predict observations from location/time
2. **Data → Spacetime**: Predict when/where from observations
3. **Cross-Modal**: Predict one modality from another
4. **Temporal Forecasting**: Predict future from past

### Context Window Construction

Intelligent sampling based on multiple similarity dimensions:

```python
sampling_strategy:
  clusters_per_context: 8      # Number of focal points
  samples_per_cluster: 32      # Samples around each center
  time_of_day_weight: 0.1      # Diurnal patterns
  time_of_year_weight: 0.3     # Seasonal patterns
  time_of_history_weight: 0.1  # Long-term trends
  spatial_weight: 0.2          # Geographic proximity
  modality_weight: 0.2         # Within-modality similarity
  universal_weight: 0.1        # Cross-modal patterns
```

### Evaluation Strategy

#### Spatial Holdouts
Test geographic generalization:
```yaml
spatial_holdouts:
  - type: percentage
    lat_pct: 0.05        # 5% of latitude range
    lon_pct: 0.05        # 5% of longitude range
    min_separation_pct: 0.2  # Minimum 20% separation
```

#### Temporal Holdouts
Test future prediction:
```yaml
temporal_holdouts:
  - type: percentage
    value: 0.15          # Last 15% of time range
    position: end
    min_separation_pct: 0.1
```

## Advanced Features

### GPU Acceleration

- **CUDA Operations**: All tensor operations on GPU
- **TorchDR UMAP**: GPU-accelerated dimensionality reduction
- **Mixed Precision**: FP16 computation for 2x memory efficiency
- **torch.compile()**: Graph compilation for optimized execution

### Memory Management

- **Gradient Accumulation**: Support for larger effective batches
- **Checkpoint Gradients**: Trade computation for memory
- **Efficient Caching**: Preprocessed data stored on disk
- **Lazy Loading**: Embeddings loaded on-demand

### Scalability

- **Multi-GPU Support**: DataParallel and DistributedDataParallel ready
- **Batch Processing**: Efficient handling of large datasets
- **Streaming Data**: Support for datasets larger than memory
- **Incremental Learning**: Resume training from checkpoints

## Installation

### Requirements

```bash
# Core dependencies
torch >= 2.0.0
pandas >= 1.5.0
numpy >= 1.24.0
tqdm >= 4.65.0
pyproj >= 3.4.0
pyyaml >= 6.0
art >= 5.9

# For coordinate transformations
pip install pyproj

# For ASCII art headers
pip install art

# Optional: GPU-accelerated UMAP
pip install torchdr
conda install -c pytorch -c nvidia faiss-gpu

# Optional: Hugging Face transformers for Perceiver
pip install transformers
```

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/deepearth.git
cd deepearth

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Verify installation
python -c "from deepearth.core import DeepEarthConfig; print('DeepEarth ready!')"
```

## Usage

### Quick Start

```python
from deepearth.core.config import DeepEarthConfig
from deepearth.core.preprocessor import DatasetPreprocessor
from deepearth.core.perceiver import DeepEarthPerceiver
from deepearth.core.trainer import DeepEarthTrainer

# Load configuration
config = DeepEarthConfig.from_yaml('configs/default.yaml')

# Process data
preprocessor = DatasetPreprocessor(config)
data = preprocessor.process_csv('observations.csv')

# Initialize model
model = DeepEarthPerceiver(config, encoder_configs)

# Train
trainer = DeepEarthTrainer(model, config, train_loader, val_loader)
trainer.train()
```

### Training on Angiosperm Dataset

```bash
# Using the provided script
./train_angiosperm.sh

# Or manually
python main_angiosperm.py \
    --config configs/angiosperm.yaml \
    --data_dir /path/to/angiosperm/data \
    --output_dir experiments/flowering \
    --epochs 100 \
    --batch_size 256 \
    --compile \
    --context_sampling
```

### Inference

```python
from deepearth.core.inference import DeepEarthInference

# Load trained model
engine = DeepEarthInference('checkpoints/best.pt')

# Predict at location
result = engine.predict_at_location(
    coordinates={'lat': 40.7, 'lon': -74.0, 'elev': 10, 'time': '2024-06-15'},
    modality='visual',
    encoder='bioclip'
)

# Batch inference
results = engine.query(
    query_data='queries.csv',
    mask_spec={'spacetime': False, 'data': True},
    batch_size=64
)

# Cross-modal inference
cross_modal = engine.cross_modal_inference(
    source_data='visual_observations.csv',
    source_modality='visual',
    target_modality='environmental'
)
```

## Configuration

### Key Parameters

#### Architecture
- `universal_dim`: Total dimension of universal token (default: 1024)
- `spacetime_dim`: Dimension for spatiotemporal encoding (default: 512)
- `num_latents`: Number of latent vectors in Perceiver (default: 256)
- `latent_dim`: Dimension of each latent vector (default: 512)
- `num_blocks`: Number of Perceiver blocks (default: 8)

#### Training
- `batch_size`: Training batch size (default: 32)
- `learning_rate`: Initial learning rate (default: 1e-4)
- `num_epochs`: Total training epochs (default: 100)
- `gradient_clip`: Gradient clipping threshold (default: 1.0)
- `mixed_precision`: Enable FP16 training (default: true)

#### Masking
- `mask_spacetime_prob`: Probability of masking spacetime (default: 0.15)
- `mask_data_prob`: Probability of masking data (default: 0.15)
- `mask_dataset_prob`: Probability of masking dataset (default: 0.05)
- `mask_modality_prob`: Probability of masking modality (default: 0.05)
- `mask_encoder_prob`: Probability of masking encoder (default: 0.05)

### Creating Custom Configurations

```yaml
# my_config.yaml
universal_dim: 2048  # Larger model
spacetime_dim: 768   # More spacetime capacity

modalities:
  satellite:
    name: satellite
    encoder_name: SatelliteEncoder
    position_dim: 2
    position_shape: [32, 32]  # 1024 patches
    max_tokens: 1024

  weather:
    name: weather
    encoder_name: WeatherNet
    position_dim: 0
    position_shape: []
    max_tokens: 1

# Custom sampling strategy
sampling_strategy:
  clusters_per_context: 16
  samples_per_cluster: 16
  spatial_weight: 0.4  # Emphasize geography
  temporal_weight: 0.3  # And time
  modality_weight: 0.3
```

## Performance Optimization

### Memory Optimization

1. **Reduce batch size**: Start with smaller batches
2. **Enable gradient accumulation**: Simulate larger batches
3. **Use mixed precision**: Automatic FP16 conversion
4. **Optimize context window**: Balance context vs memory

### Speed Optimization

1. **Enable model compilation**: `--compile` flag
2. **Increase number of workers**: Set `num_workers` appropriately
3. **Use pinned memory**: `pin_memory: true` for faster transfers
4. **Optimize UMAP**: Reduce `umap_max_samples` for faster indexing

### Quality Optimization

1. **Increase model capacity**: More latents, deeper blocks
2. **Tune masking probabilities**: Balance different objectives
3. **Adjust sampling strategy**: Weight important similarities
4. **Use longer training**: More epochs, lower learning rate

## API Reference

### Core Classes

#### DeepEarthConfig
```python
config = DeepEarthConfig(
    universal_dim=1024,
    spacetime_dim=512,
    num_latents=256,
    batch_size=32,
    learning_rate=1e-4
)
```

#### DatasetPreprocessor
```python
preprocessor = DatasetPreprocessor(config)
data = preprocessor.process_csv('data.csv')
data = preprocessor.process_dataframe(df, columns)
```

#### DeepEarthPerceiver
```python
model = DeepEarthPerceiver(config, encoder_configs)
outputs = model(batch, mask=masks, inference_mode=True)
```

#### DeepEarthTrainer
```python
trainer = DeepEarthTrainer(
    model, config, 
    train_loader, val_loader, test_loader
)
trainer.train()
trainer.save_checkpoint('checkpoint.pt')
trainer.load_checkpoint('checkpoint.pt')
```

#### DeepEarthInference
```python
engine = DeepEarthInference(checkpoint_path, device='cuda')
results = engine.query(query_data, mask_spec, batch_size=32)
predictions = engine.predict_at_location(coordinates, modality)
```

### Data Structures

#### Universal Token
```python
token = {
    'spacetime': tensor[512],      # Earth4D encoding
    'data': tensor[503],           # Multimodal fusion output
    'dataset': tensor[2],          # Dataset embedding
    'modality': tensor[2],         # Modality embedding
    'encoder': tensor[3],          # Encoder embedding
    'mask': tensor[4],             # Mask pattern embedding
    'context_position': tensor[2]  # Position in context
}
```

#### Batch Format
```python
batch = {
    'xyzt': tensor[B, S, 4],                    # Coordinates
    'dataset_modality_encoder': tensor[B, S, 3], # Metadata
    'encoded_data': tensor[B, S, D],            # Embeddings
    'modality_positions': dict,                 # Optional positions
}
```

