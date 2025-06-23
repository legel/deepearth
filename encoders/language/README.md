# DeepSeek-V3 Language Encoder

This module provides infrastructure for extracting high-quality language embeddings using DeepSeek-V3, a state-of-the-art language model. It includes server deployment, client libraries, and tools for converting embeddings to 3D visualizations.

## Overview

The DeepSeek-V3 language encoder system consists of:

1. **Model Server** (`server.py`) - HTTP API server that loads the model once and serves embedding/completion requests
2. **Client Library** (`client.py`) - Python client for interacting with the server
3. **UMAP Processor** (`umap_processor.py`) - Converts high-dimensional embeddings to 3D coordinates for visualization
4. **GCloud Deployment** (`deploy_gcloud.py`) - Automated deployment to Google Cloud Platform

## Requirements

### Hardware Requirements

**IMPORTANT**: The DeepSeek-V3 model requires substantial memory. Our testing shows the model uses **300-400GB of RAM** during inference.

#### Minimum Production Requirements:
- **RAM**: 400GB minimum (model uses ~300GB during active inference)
- **Storage**: 200GB SSD (model files ~50GB + workspace)
- **CPU**: 40+ cores recommended for reasonable performance

#### Tested Configuration (Google Cloud):
- **Instance Type**: `n2-highmem-80` (640GB RAM, 80 vCPUs)
- **Actual RAM Usage**: ~300GB during active inference
- **Disk**: 500GB SSD
- **Cost**: ~$5-6/hour (on-demand), ~$1.5-2/hour (preemptible)

**Note**: We have not tested with instances smaller than n2-highmem-80. The 640GB RAM provides comfortable headroom for the ~300GB active usage.

#### Alternative Cloud Options:
- **AWS**: `x2gd.16xlarge` (1024GB RAM) or `r6i.16xlarge` (512GB RAM)
- **Azure**: `Standard_M128s` (2048GB RAM) or `Standard_E104i_v5` (672GB RAM)

#### NOT Suitable:
- Standard desktop/laptop configurations
- Instances with <300GB RAM
- The previously mentioned `n1-highmem-16` (104GB) is **insufficient**

### Software Requirements

```bash
# System packages
apt-get install python3-pip python3-venv git curl wget build-essential

# Python packages
pip install flask requests numpy pandas torch llama-cpp-python umap-learn hdbscan scikit-learn
```

## Quick Start

### 1. Local Setup

```bash
# Clone the repository
git clone https://github.com/legel/deepearth.git
cd deepearth/src/deepearth/encoders/language

# Download the model (example URL - replace with actual)
mkdir -p models
wget -O models/DeepSeek-V3-0324-UD-Q4_K_XL-00001-of-00009.gguf "MODEL_URL"

# Start the server
python server.py --model-path models/DeepSeek-V3-0324-UD-Q4_K_XL-00001-of-00009.gguf
```

### 2. Extract Embeddings

```python
from client import DeepSeekClient

# Connect to server
client = DeepSeekClient('http://localhost:8888')

# Get embedding for text
result = client.embed("The ecology of coastal wetlands")
print(f"Embedding shape: {result['embedding_shape']}")
print(f"Mean embedding: {result['mean_embedding'][:5]}...")  # First 5 dimensions

# Extract embeddings with token information
texts = [
    "Climate adaptation in tropical forests",
    "Urban biodiversity patterns",
    "Coral reef ecosystem dynamics"
]

results = client.extract_embeddings_with_tokens(
    texts, 
    labels=["forests", "urban", "coral"],
    output_dir="embeddings"
)
```

### 3. Convert to 3D Visualization

```python
from umap_processor import compute_3d_umap_and_clusters

# Process embeddings directory
umap_data = compute_3d_umap_and_clusters(
    "embeddings",  # Directory with .pt files
    output_path="umap_3d_data.json"
)

# View with the 3D visualization tool
cd ../../visualization/embeddings
python server.py --data umap_3d_data.json
```

## Cloud Deployment

### Google Cloud Platform

1. **Set up GCloud CLI**:
```bash
# Install gcloud SDK
curl https://sdk.cloud.google.com | bash

# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

2. **Deploy DeepSeek Server**:
```bash
python deploy_gcloud.py \
    --project YOUR_PROJECT_ID \
    --machine-type n2-highmem-80 \
    --disk-size 500 \
    --preemptible
```

3. **Process Data on Cloud**:
```bash
python deploy_gcloud.py \
    --project YOUR_PROJECT_ID \
    --data-file species_data.csv \
    --output-dir embeddings \
    --gpu-type nvidia-tesla-v100 \
    --gpu-count 1
```

## API Reference

### Server Endpoints

#### `POST /embed`
Get embeddings for text with optional token information.

**Request:**
```json
{
    "text": "Your text here",
    "return_tokens": true,
    "return_token_embeddings": false
}
```

**Response:**
```json
{
    "text": "Your text here",
    "embedding_shape": [10, 7168],
    "num_tokens": 10,
    "mean_embedding": [...],
    "mean_embedding_stats": {
        "mean": -0.0023,
        "std": 0.0421,
        "min": -0.234,
        "max": 0.187
    },
    "tokens": [
        {
            "position": 0,
            "token_id": 12345,
            "token_str": "Your"
        }
    ]
}
```

#### `POST /tokenize`
Tokenize text and get token information.

#### `POST /embed_batch`
Process multiple texts efficiently.

#### `POST /complete`
Generate text completions.

## Data Structures

### Embedding Output Format (.pt files)

```python
{
    'text': str,                    # Original text
    'label': str,                   # Label for the text
    'mean_embedding': torch.Tensor, # Mean pooled embedding [embedding_dim]
    'num_tokens': int,              # Number of tokens
    'tokens': [                     # Token information
        {
            'position': int,
            'token_id': int,
            'token_str': str
        }
    ],
    'token_embeddings': torch.Tensor  # Optional: per-token embeddings [n_tokens, embedding_dim]
}
```

### UMAP 3D Output Format

```json
{
    "points": [
        {
            "id": 0,
            "name": "Label for point",
            "x": 1.234,
            "y": 2.345,
            "z": 3.456,
            "cluster": 0,
            "color": "#ff6b6b"
        }
    ],
    "stats": {
        "total_points": 100,
        "n_clusters": 5,
        "noise_points": 3,
        "dimensions": 7168,
        "cluster_sizes": {
            "0": 20,
            "1": 30
        }
    },
    "colors": {
        "-1": "#808080",
        "0": "#ff6b6b",
        "1": "#4ecdc4"
    }
}
```

## Example: Species Embeddings Pipeline

```python
# 1. Extract embeddings for species
from client import DeepSeekClient, extract_species_embeddings

client = DeepSeekClient('http://localhost:8888')
extract_species_embeddings(
    client,
    'species_data.csv',
    'species_embeddings',
    species_column='taxon_name',
    id_column='taxon_id'
)

# 2. Compute 3D projection
from umap_processor import compute_3d_umap_and_clusters

umap_data = compute_3d_umap_and_clusters(
    'species_embeddings',
    output_path='species_umap_3d.json',
    n_neighbors=15,
    min_dist=0.1,
    min_cluster_size=5
)

# 3. Create visualization config
from umap_processor import create_visualization_config

create_visualization_config(
    umap_data,
    title="Florida Plant Species Embeddings",
    subtitle="232 species visualized using DeepSeek-V3",
    output_path='species_viz_config.json'
)
```

## Performance Optimization

### Memory Management
- Use `low_vram=True` in model configuration
- Enable memory mapping with `use_mmap=True`
- Process in batches for large datasets

### GPU Acceleration
```python
# Enable GPU layers (adjust based on VRAM)
python server.py --gpu-layers 20
```

### Scaling
- Use cloud deployment for large batches
- Enable preemptible instances for cost savings
- Consider multi-GPU setups for production

## Troubleshooting

### Out of Memory
- Reduce batch size
- Use CPU-only mode
- Increase system swap space
- Use cloud instance with more RAM

### Slow Performance
- Enable GPU acceleration
- Increase number of threads
- Use SSD for model storage
- Consider quantized model versions

### Connection Issues
- Check firewall rules
- Verify server is running
- Test with health endpoint: `curl http://localhost:8888/health`

## Model Information

DeepSeek-V3 is a large language model optimized for:
- High-quality text embeddings
- Contextual understanding
- Multi-language support
- Scientific text comprehension

**Model Details:**
- Architecture: Transformer-based
- Parameters: 671B (4.5-bit quantized)
- Embedding Dimension: 7168
- Context Length: 2048 tokens
- Quantization: 4.5-bit (Q4_K_XL)

## Citation

If you use this system in your research, please cite:

```bibtex
@software{deepearth2024,
  title = {DeepEarth: Language Encoders},
  author = {DeepEarth Contributors},
  year = {2024},
  url = {https://github.com/legel/deepearth}
}
```

## License

This module is part of the DeepEarth project. See the main repository for license information.