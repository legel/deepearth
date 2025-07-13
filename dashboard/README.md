# DeepEarth Dashboard

![DeepEarth Dashboard: Geospatial Vision Features](https://github.com/legel/deepearth/blob/main/docs/deepearth_geospatial_vision_features.png)

Interactive visualization and ML-ready data orchestration for the DeepEarth Self-Supervised Spatiotemporal Multimodality Simulator.

![DeepEarth Dashboard](https://img.shields.io/badge/DeepEarth-Dashboard-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview

The DeepEarth Dashboard is a production-ready ML orchestration system designed for multimodal biodiversity research. Built with a service-oriented architecture, it transitions seamlessly from interactive data exploration to automated machine learning pipeline control.

### 🧠 ML-Native Architecture

**Service Layer Design**: Each capability encapsulated in focused, testable modules
- **12 Specialized Services**: Vision processing, UMAP computation, ecosystem analysis
- **4 Error Handling Decorators**: Consistent API responses across all endpoints  
- **9 Request Parsing Utilities**: Consolidated parameter validation and extraction
- **Memory-Mapped Tensors**: Direct PyTorch access for real-time model integration

### Core Capabilities

- **🔍 Interactive Data Exploration**: Spatiotemporal filtering with sub-100ms response
- **🧠 ML Pipeline Integration**: Direct tensor operations, batch sampling, streaming data
- **🎨 Multimodal Visualization**: V-JEPA-2 attention maps, DeepSeek-V3 semantic spaces
- **🚀 Automated System Control**: Training loop integration, model deployment, monitoring
- **📊 Performance Analytics**: 21x faster than vector DBs, intelligent caching

### Vision: Beyond Visualization to ML Control

This dashboard architecture enables the transition from exploratory research to production ML systems:

```
🔬 Research Phase          🤖 Production Phase
├── Data Exploration  ──►  ├── Automated Sampling
├── Pattern Discovery ──►  ├── Model Training Control  
├── Hypothesis Testing ──► ├── Real-time Inference
└── Manual Analysis   ──►  └── Autonomous Discovery
```

## Quick Start

### 1. Download and Index Dataset

```bash
# Download dataset and prepare ML-ready embeddings
python3 prepare_embeddings.py --download deepearth/central-florida-native-plants
```

This creates:
- `embeddings.mmap` (206GB) - Memory-mapped tensors for direct ML access
- `embeddings_index.db` - Spatiotemporal index for efficient querying
- **Processing time**: ~50 minutes (indexes 7,949 vision embeddings)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard

```bash
# Development mode
python3 deepearth_dashboard.py

# Production mode with Gunicorn
./run_production.sh
```

Access at http://localhost:5000

## 🏗️ Modular Service Architecture

The dashboard follows a **service-oriented design** optimized for ML workflows:

### Service Layer Organization

```
dashboard/
├── deepearth_dashboard.py         # 🎯 API Orchestration (314 lines)
├── services/                      # 🔧 Business Logic Layer
│   ├── app_initialization.py      #   Application bootstrap & config
│   ├── vision_features.py         #   Vision processing & attention maps
│   ├── observation_processing.py  #   Observation data formatting
│   ├── feature_analysis.py        #   PCA computation & statistics
│   ├── health_monitoring.py       #   System health & diagnostics
│   ├── umap_visualization.py      #   UMAP RGB visualization
│   ├── attention_processing.py    #   Attention map generation
│   ├── vision_processing.py       #   Vision embedding filtering
│   ├── color_processing.py        #   Species cluster colors
│   ├── image_processing.py        #   Image proxy functionality
│   ├── ecosystem_processing.py    #   Ecosystem analysis
│   ├── umap_processing.py         #   Language UMAP processing
│   └── training_data.py           #   ML training batch operations
├── api/                           # 🛡️ API Infrastructure
│   └── error_handling.py          #   Unified error handling decorators
├── utils/                         # 🔧 Common Utilities
│   └── request_parsing.py         #   Parameter parsing & validation
└── vision/                        # 👁️ Vision Processing
    └── attention_utils.py          #   Attention overlay generation
```

### Service Capabilities

#### **Vision & Multimodal Processing**
- **`vision_features.py`**: V-JEPA-2 embedding analysis, temporal frame navigation
- **`attention_processing.py`**: Spatial attention map generation with custom colormaps
- **`umap_visualization.py`**: False-color imagery from embedding UMAP coordinates
- **`vision_processing.py`**: Geographic filtering of 6.4M-dimensional embeddings

#### **Data & Analytics Services**
- **`observation_processing.py`**: 33K+ observation formatting, species summaries
- **`feature_analysis.py`**: Real-time PCA computation, statistical analysis
- **`ecosystem_processing.py`**: Community structure analysis, biodiversity metrics
- **`umap_processing.py`**: 7.2K-dimensional language embedding clustering

#### **System Infrastructure**
- **`app_initialization.py`**: UMAP compilation, config loading, cache warming
- **`health_monitoring.py`**: Memory usage, data availability, performance metrics
- **`color_processing.py`**: HDBSCAN-derived color harmonization
- **`image_processing.py`**: Optimized image proxy with size transformation

### API Infrastructure

#### **Unified Error Handling** (`api/error_handling.py`)
```python
@handle_api_error        # Standard API errors with JSON responses
@handle_vision_error     # Vision processing specific errors
@handle_image_proxy_error # Image loading and proxy errors  
@handle_health_check_error # System diagnostics errors
```

#### **Request Parsing Utilities** (`utils/request_parsing.py`)
```python
parse_geographic_bounds()           # Lat/lng bounds with validation
parse_temporal_filters()            # Year/month/day/hour filtering
parse_vision_features_parameters()  # Vision analysis parameters
parse_ecosystem_analysis_parameters() # Community analysis setup
```

## 🚀 ML Pipeline Integration

### Direct Tensor Access
```python
# Memory-mapped embeddings enable direct PyTorch operations
embeddings = cache.get_vision_embedding(gbif_id)  # 6.4M dims
tensor = torch.from_numpy(embeddings)             # Zero-copy conversion
model_input = tensor.reshape(8, 24, 24, 1408)     # Ready for training
```

### Batch Operations
```python
# Optimized for mini-batch training workflows
batch_ids = filter_available_vision_embeddings(bounds, max_images=64)
batch_data = [cache.get_vision_embedding(id) for id in batch_ids]
training_batch = torch.stack([torch.from_numpy(emb) for emb in batch_data])
```

### Real-time Analytics
```python
# Performance monitoring for ML system integration
health_status = generate_health_status(cache, CONFIG)
# Returns: cache_size, memory_usage, data_availability, retrieval_times
```

## 📊 Performance & Scalability

### Intelligent Caching System
- **LRU Cache Management**: Prevents memory bloat during extended sessions
- **Memory-Mapped Access**: Sub-100ms retrieval without loading into RAM
- **Thread-Safe Operations**: Concurrent access for distributed training
- **Smart Cache Warming**: Predictive loading based on access patterns

### Benchmark Results
```
Single embedding retrieval:    ~71ms
Batch of 100 embeddings:      ~25ms per item
vs Vector databases:          21x faster
vs Parquet files:            140x faster
Memory footprint:            <2GB for 206GB dataset
```

### Production Optimizations
- **Streamlined Logging**: Reduced overhead in hot execution paths
- **Dependency Warning Suppression**: Clean startup without library noise
- **Modular Import Strategy**: Minimal coupling, faster cold starts
- **Error Handler Decorators**: Consistent responses with minimal overhead

## 🔮 Future ML Control Integration

The service architecture enables seamless expansion into automated ML workflows:

### **Training Loop Integration**
```python
# ML Training Data Pipeline (existing endpoint)
@app.route('/api/training/batch', methods=['POST'])
def get_training_data_batch():
    from services.training_data import get_training_batch
    # Returns PyTorch-ready tensors for direct model consumption
    return get_training_batch(cache, observation_ids, include_vision=True, include_language=True)

# Training script integration (../training/train_classifier.py)
from services.training_data import get_training_batch, get_available_observation_ids
batch_data = get_training_batch(cache, filtered_train_ids, include_vision=True)
```

### **Model Deployment Services**
```python
# Model serving integration
@app.route('/api/inference/<model_id>')
def run_inference():
    return process_model_inference(cache, model_registry)

# A/B testing framework
@app.route('/api/experiments/compare')
def compare_models():
    return compare_model_performance(cache, experiment_config)
```

### **Automated Discovery Pipelines**
```python
# Hypothesis generation from embedding spaces
@app.route('/api/discovery/patterns')
def discover_patterns():
    return analyze_embedding_patterns(cache, discovery_params)
```

## API Endpoints

### **Core Data Services**
- `GET /api/observations` - Spatiotemporal observation stream (33K+ records)
- `GET /api/observation/<gbif_id>` - Detailed observation with full metadata
- `GET /api/config` - System configuration for ML pipeline integration

### **Embedding & ML Services**
- `GET /api/language_embeddings/umap` - 3D semantic landscape (7.2K dims → 3D)
- `GET /api/vision_embeddings/umap` - Regional vision clustering (6.4M dims → 3D) 
- `GET /api/species_umap_colors` - HDBSCAN-derived color harmonization

### **Vision Intelligence Services**
- `GET /api/vision_features/<gbif_id>` - Spatial attention analysis (8×24×24×1408)
- `GET /api/features/<image_id>/attention` - Attention overlay generation
- `GET /api/features/<image_id>/umap-rgb` - False-color embedding visualization
- `GET /api/features/<image_id>/statistics` - Real-time feature statistics

### **Analytics & System Services**
- `GET /api/ecosystem_analysis` - Community structure analysis
- `GET /api/grid_statistics` - Geographic aggregation with temporal filtering
- `GET /api/health` - System diagnostics, performance metrics, cache status

## Dataset Information

**Central Florida Native Plants v0.2.0**
- **33,665 biodiversity observations** across 232 native species
- **7,949 vision embeddings** (V-JEPA-2, 6,488,064 dimensions each)
- **Language embeddings** (DeepSeek-V3, 7,168 dimensions per species)
- **Temporal coverage**: 2010-2025
- **Geographic scope**: Central Florida (28.03°N to 28.98°N)

## Development Guide

### Adding New Service Modules

```python
# services/new_service.py
def process_new_functionality(cache, parameters):
    """
    Clear documentation of inputs, outputs, and ML integration points.
    
    Args:
        cache: UnifiedDataCache for data access
        parameters: Parsed request parameters
        
    Returns:
        dict: JSON-serializable result for API response
    """
    # Service implementation
    return result

# Import in main dashboard
from services.new_service import process_new_functionality

@app.route('/api/new_endpoint')
@handle_api_error
def new_endpoint():
    """ML-focused endpoint documentation."""
    params = parse_new_parameters()
    result = process_new_functionality(cache, params)
    return jsonify(result)
```

### Testing Service Modules

```bash
# Test individual services
python -c "from services.vision_features import process_vision_features_request; print('Vision OK')"
python -c "from api.error_handling import handle_api_error; print('Error handling OK')"
python -c "from utils.request_parsing import parse_geographic_bounds; print('Parsing OK')"

# Integration testing
python -m py_compile deepearth_dashboard.py
python deepearth_dashboard.py
```

## Deployment

### Production with Gunicorn

```bash
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 --worker-class sync \
         --max-requests 1000 --preload deepearth_dashboard:app
```

### Systemd Service

```ini
[Unit]
Description=DeepEarth Dashboard ML System
After=network.target

[Service]
Type=simple
User=deepearth
WorkingDirectory=/opt/deepearth/dashboard
ExecStart=/opt/deepearth/venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 deepearth_dashboard:app
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

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