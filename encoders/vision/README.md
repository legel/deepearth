# V-JEPA 2 Vision Encoder

This module provides infrastructure for working with V-JEPA 2 (Video Joint Embedding Predictive Architecture) visual embeddings for biodiversity observations.

## Overview

V-JEPA 2 is a self-supervised video representation learning model from Meta AI that learns spatiotemporal features without phenological bias. It processes images/videos to extract rich visual features that capture both spatial patterns and temporal dynamics.

## Model Details

- **Model**: facebook/vjepa2-vitg-fpc64-384 (Vision Transformer Giant)
- **Architecture**: Self-supervised video transformer
- **Input**: Images/video frames
- **Output**: 4,608 patches × 1,408 dimensions per image
- **Patch Structure**: 576 spatial patches (24×24 grid) × 8 temporal frames
- **Memory per image**: ~12.4 MB (fp16)

## Feature Structure

Each `.pt` file contains a dictionary with:
```python
{
    'features': torch.Tensor,  # Shape: [4608, 1408] - spatiotemporal patches
    'shape': torch.Size,       # Original feature dimensions
    'dtype': str,             # Data type (usually 'float16')
    'source_chunk': str,      # Source chunk file
    'image_id': str          # Image identifier
}
```

### Patch Organization
- Total patches: 4,608 = 576 spatial × 8 temporal
- Spatial grid: 24×24 patches
- Temporal frames: 8 frames
- Each patch: 1,408-dimensional feature vector

## Usage

### Loading Features
```python
import torch
from pathlib import Path

# Load features for a single image
features = torch.load('path/to/features.pt')
spatial_temporal_features = features['features']  # [4608, 1408]

# Aggregate to image-level representation
mean_features = spatial_temporal_features.mean(dim=0)  # [1408]
```

### Spatial-Temporal Decomposition
```python
# Reshape to separate spatial and temporal dimensions
# [4608, 1408] -> [8, 576, 1408] -> [8, 24, 24, 1408]
features_reshaped = spatial_temporal_features.view(8, 576, 1408)
features_grid = features_reshaped.view(8, 24, 24, 1408)

# Get features for specific temporal frame
frame_0_features = features_grid[0]  # [24, 24, 1408]

# Get features for specific spatial location across time
location_features = features_grid[:, 12, 12]  # [8, 1408] - center patch
```

## Data Organization

### Directory Structure
```
/home/photon/4tb/deepearth_v0.01_visual_features_individual/
├── taxon_2650927/
│   ├── gbif_1291162453_taxon_2650927_nephrolepis_exaltata_img_1_features.pt
│   ├── gbif_1291162453_taxon_2650927_nephrolepis_exaltata_img_2_features.pt
│   └── ...
├── taxon_2651707/
│   └── ...
```

### File Naming Convention
`gbif_{gbif_id}_taxon_{taxon_id}_{species_name}_img_{img_num}_features.pt`

## Integration with DeepEarth

The V-JEPA 2 features serve as the visual modality input for the DeepEarth multimodal system:

1. **Raw Features**: 4,608 × 1,408 patches per image
2. **Aggregation**: Various pooling strategies (mean, attention, hierarchical)
3. **Projection**: MLP to project to common embedding space
4. **Fusion**: Cross-modal attention with language and geospatial features

## Key Advantages

1. **No Phenological Bias**: Self-supervised training on general video data
2. **Spatiotemporal Structure**: Preserves spatial layout and temporal dynamics
3. **Rich Representations**: 1,408-dimensional features per patch
4. **Interpretability**: Can visualize attention on specific image regions
5. **Flexibility**: Multiple aggregation strategies possible

## Installation

```bash
# Install required dependencies
pip install torch torchvision transformers pillow tqdm

# Optional: Install for GPU support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage Examples

### Basic Feature Extraction

```python
from encoders.vision import VJEPA2Extractor

# Initialize extractor
extractor = VJEPA2Extractor(
    model_name="facebook/vjepa2-vitg-fpc64-384",
    device="cuda:0",
    use_fp16=True
)

# Extract features from a single image
features = extractor.extract_features("path/to/image.jpg")
print(f"Features shape: {features.shape}")  # [4608, 1408]

# Aggregate to image-level representation
image_embedding = extractor.aggregate_features(features, method="mean")
print(f"Image embedding shape: {image_embedding.shape}")  # [1408]

# Save features
extractor.save_features(features, "output/features.pt", metadata={"image_id": "12345"})
```

### Batch Processing

```python
from encoders.vision import BatchVJEPA2Extractor

# Initialize batch extractor
batch_extractor = BatchVJEPA2Extractor(
    output_dir="output/features",
    chunk_size=1000,
    device="cuda:0",
    use_fp16=True
)

# Process entire directory
batch_extractor.process_directory(
    image_dir="path/to/images",
    pattern="*.jpg"
)
```

### Command Line Usage

```bash
# Extract features from a directory
python -m encoders.vision.vjepa2_extractor \
    --image_dir /path/to/images \
    --output_dir /path/to/output \
    --device cuda:0 \
    --chunk_size 1000

# Run parallel extraction on multiple GPUs
./encoders/vision/run_parallel_extraction.sh \
    /path/to/images \
    /path/to/output \
    4  # number of GPUs
```

### Loading and Using Saved Features

```python
from encoders.vision import VJEPA2Extractor

# Load saved features
features_data = VJEPA2Extractor.load_features("path/to/features.pt")
features = features_data['features']  # [4608, 1408]
metadata = features_data.get('metadata', {})

# Access spatial features for visualization
extractor = VJEPA2Extractor()
spatial_features = extractor.get_spatial_features(features, frame=0)  # [24, 24, 1408]
```

## Feature Post-Processing

### Dimensionality Reduction

```python
# Project to lower dimension using PCA
from sklearn.decomposition import PCA

# Aggregate features
mean_features = features.mean(dim=0).numpy()  # [1408]

# Reduce dimension
pca = PCA(n_components=768)
reduced_features = pca.fit_transform(mean_features.reshape(1, -1))
```

### Multi-Level Aggregation

```python
# Hierarchical pooling
def hierarchical_pool(features, extractor):
    # Get spatial-temporal structure
    features_4d = features.view(
        extractor.temporal_frames,
        extractor.spatial_grid_size,
        extractor.spatial_grid_size,
        extractor.patch_dim
    )
    
    # Pool at different levels
    frame_means = features_4d.mean(dim=(1, 2))  # [8, 1408] - temporal
    spatial_means = features_4d.mean(dim=0)  # [24, 24, 1408] - spatial
    global_mean = features_4d.mean(dim=(0, 1, 2))  # [1408] - global
    
    return {
        'temporal': frame_means,
        'spatial': spatial_means,
        'global': global_mean
    }
```

## Performance Considerations

1. **Memory Usage**: Each image requires ~25MB GPU memory during extraction
2. **Speed**: ~5-10 images/second on V100 GPU
3. **Storage**: ~12.4MB per image (fp16)
4. **Batch Size**: Use batch_size=1 for V-JEPA 2 (model processes single images)

## Citation

```bibtex
@article{bardes2024revisiting,
  title={Revisiting Feature Prediction for Learning Visual Representations from Video},
  author={Bardes, Adrien and Garrido, Quentin and Ponce, Jean and Chen, Xinlei and Rabbat, Michael and LeCun, Yann and Assran, Mido and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2404.08471},
  year={2024}
}
```