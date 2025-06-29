# DeepEarth Dashboard

Interactive visualization and ML-ready data indexing for the DeepEarth Self-Supervised Spatiotemporal Multimodality Simulator.

![DeepEarth Dashboard](https://img.shields.io/badge/DeepEarth-Dashboard-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview

The DeepEarth Dashboard is a visualization and data preparation system designed to support multimodal machine learning research. This implementation demonstrates the system using Central Florida Native Plants as an exemplar dataset, showcasing how biodiversity observations can be indexed and explored through advanced embedding spaces.

### Core Capabilities

- **🧠 ML-Ready Data Indexing**: Memory-mapped embeddings enable direct tensor operations for model training
- **🔍 Multimodal Analysis**: Visualize V-JEPA-2 vision features and DeepSeek-V3 language embeddings
- **📊 Embedding Space Navigation**: Interactive 3D projections reveal learned representations
- **⚡ High Performance**: Sub-100ms retrieval enables real-time ML experimentation
- **🎯 Spatiotemporal Filtering**: Query by location, time, and semantic similarity

### Vision: ML Control System Integration

The dashboard architecture is designed as a foundation for integrated machine learning workflows, similar to NeRFStudio and other modern ML visualization systems. The memory-mapped embedding format isn't just for fast viewing—it's the first step toward:

- **Real-time model training visualization**: Watch embeddings update during training
- **Interactive hyperparameter tuning**: Adjust model parameters and see immediate effects
- **Active learning interfaces**: Identify and label high-value training samples
- **Model comparison tools**: Compare different architectures side-by-side

## Quick Start

This example demonstrates the dashboard using the Central Florida Native Plants dataset:

### 1. Download and Index Dataset

```bash
# Download dataset and prepare ML-ready embeddings
python prepare_embeddings.py --download deepearth/central-florida-native-plants
```

This creates:
- `embeddings.mmap` (206GB) - Direct-access tensor storage for ML pipelines
- `embeddings_index.db` - Spatiotemporal index for efficient queries

**Processing time**: ~50 minutes (indexes 7,949 vision embeddings)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:
- Flask (web framework)
- PyTorch (tensor operations)
- UMAP (dimensionality reduction)
- NumPy, Pandas (data manipulation)

### 3. Configure Dataset Path

The dataset configuration will be automatically set to use the downloaded data in `huggingface_dataset/` directory.

### 4. Run the Dashboard

```bash
# Development mode
python deepearth_dashboard.py

# Production mode with Gunicorn
./run_production.sh
```

Access the dashboard at http://localhost:5000

## Features

### 🧠 ML-Ready Data Infrastructure

- **Memory-Mapped Tensors**: Direct PyTorch/TensorFlow access without loading into RAM
- **Spatiotemporal Indexing**: SQLite-powered queries for training data selection
- **Batch Operations**: Optimized retrieval for mini-batch training
- **Thread-Safe Access**: Concurrent data loading for distributed training

### 🗺️ Spatiotemporal Exploration

- **Interactive Mapping**: Visualize 33,665 observations with multimodal features
- **Temporal Dynamics**: Filter by year, month, day, hour for time-series analysis
- **Spatial Statistics**: Grid-based aggregation for geographic patterns
- **Species Relationships**: UMAP-derived color mapping reveals semantic similarities

### 🔍 Embedding Space Analysis

- **Vision Feature Visualization**: 
  - Spatial attention maps from V-JEPA-2 (8×24×24×1408 features)
  - Multiple analysis methods (L2 norm, PCA, variance, entropy)
  - Temporal frame navigation for video understanding
- **Language Embedding Navigation**:
  - 3D UMAP projections of DeepSeek-V3 embeddings (7,168 dims)
  - HDBSCAN clustering for community detection
  - Interactive exploration of semantic relationships

### 📊 Model-Ready Visualizations

- **Attention Mechanism Insights**: Understand what vision models learn
- **Embedding Space Topology**: Explore learned representation structure
- **Cross-Modal Alignment**: Compare vision and language embedding spaces
- **Performance Metrics**: Real-time retrieval statistics for optimization

### ⚡ Performance

- **Memory-Mapped Files**: Direct disk access without loading into RAM
- **SQLite Indexing**: O(1) lookup by GBIF ID
- **Thread-Safe**: Handles concurrent requests
- **LRU Caching**: Frequently accessed embeddings stay in memory
- **Batch Operations**: Optimized retrieval for multiple embeddings

Typical performance:
- Single embedding: ~71ms
- Batch of 100: ~25ms per embedding
- 21x faster than vector databases
- 140x faster than Parquet files

## Architecture

The dashboard follows a modular design optimized for ML workflows:

```
DeepEarth Dashboard
├── Visualization Layer
│   ├── Interactive maps (Leaflet.js)
│   ├── 3D embedding spaces (Three.js)
│   └── Real-time analytics
├── ML Integration Layer
│   ├── PyTorch tensor operations
│   ├── Memory-mapped data access
│   ├── Batch loading utilities
│   └── RESTful API for model integration
└── Data Index Layer
    ├── Embeddings (mmap format for direct tensor access)
    ├── Metadata (Parquet for flexibility)
    └── Spatiotemporal index (SQLite for queries)
```

This architecture enables future integration with:
- Training loops (real-time loss visualization)
- Model servers (inference endpoints)
- Experiment tracking (MLflow, Weights & Biases)
- Active learning pipelines

## API Endpoints

### Core Data
- `GET /api/observations` - All observations with metadata
- `GET /api/observation/<gbif_id>` - Detailed observation info
- `GET /api/config` - Dataset configuration

### Embeddings
- `GET /api/language_embeddings/umap` - 3D UMAP projection of species
- `GET /api/vision_embeddings/umap` - Regional vision UMAP
- `GET /api/species_umap_colors` - RGB colors for species

### Vision Features
- `GET /api/features/<image_id>/attention` - Spatial attention maps
- `GET /api/features/<image_id>/umap-rgb` - UMAP RGB visualization
- `GET /api/features/<image_id>/statistics` - Feature statistics

### Analysis
- `GET /api/grid_statistics` - Species composition for map regions
- `GET /api/ecosystem_analysis` - Community analysis
- `GET /api/health` - System health check

## Deployment

### Production with Gunicorn

```bash
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 deepearth_dashboard:app
```

### Systemd Service

```ini
[Unit]
Description=DeepEarth Dashboard
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/dashboard
ExecStart=/path/to/venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 deepearth_dashboard:app
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

### Nginx Configuration

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_read_timeout 120s;
    }
    
    location /static {
        alias /path/to/dashboard/static;
        expires 1d;
    }
}
```

## Dataset Information

**Central Florida Native Plants v0.2.0**
- 33,665 biodiversity observations
- 232 native plant species
- 7,949 observations with V-JEPA-2 vision embeddings
- 6,488,064 dimensions per vision embedding
- 7,168 dimensions per DeepSeek-V3 language embedding
- Temporal range: 2010-2025
- Geographic bounds: Central Florida (28.03°N to 28.98°N)

## Development

### Adding New Features

1. **Backend**: Add endpoint to `deepearth_dashboard.py`
2. **Frontend**: Update JavaScript in `static/js/`
3. **Styling**: Modify `static/css/dashboard.css`

### Data Flow

1. User interacts with map/UI
2. JavaScript sends API request
3. Flask endpoint processes request
4. Data loaded via HuggingFace/MMap loader
5. PyTorch operations (if needed)
6. JSON response to frontend
7. JavaScript updates visualization

## Troubleshooting

### "Too many open files" error
Increase system limits:
```bash
ulimit -n 65536
```

### Slow initial load
- First access loads OS page cache
- Subsequent accesses are faster
- Consider warming cache on startup

### Memory-mapped loader fails
- Check file permissions
- Verify `embeddings.mmap` exists
- Falls back to Parquet automatically

## Citation

If you use DeepEarth in your research, please cite:

```bibtex
@article{deepearth2025,
  title = {DeepEarth: Self-Supervised Spatiotemporal Multimodality Simulator},
  author = {DeepEarth Contributors},
  year = {2025},
  url = {https://github.com/deepearth}
}
```

## License

MIT License - see LICENSE file for details