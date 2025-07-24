#!/usr/bin/env python3
"""
DeepEarth Dashboard - ML-Ready Visualization & Control System

    ╭─────────────────────────────────────────────────╮
    │  🌍 DEEPEARTH: Planetary Simulator Interface   │
    │                                                 │
    │  📊 Data Streams ──► 🧠 ML Pipelines ──► 🎯 Insights │
    │  🔬 Research ──────► 🤖 Automation ───► 🌱 Discovery │
    ╰─────────────────────────────────────────────────╯

A modular, production-ready dashboard bridging biodiversity visualization 
with machine learning orchestration. Built for researchers to transition 
seamlessly from data exploration to automated ML system control.

Architecture Philosophy:
    Service-Oriented Design │ Each capability encapsulated in focused modules
    ML-Native Integration   │ Direct tensor access, batch operations, streaming
    Zero-Latency Access     │ Memory-mapped storage, intelligent caching
    Research-to-Production  │ From interactive exploration to automated systems

Core Capabilities:
    🔍 Interactive Data Exploration  │ Spatiotemporal filtering, real-time stats
    🎨 Multimodal Visualization      │ Vision attention maps, embedding spaces  
    🚀 ML Pipeline Integration       │ Direct tensor access, batch sampling
    🤖 Automated System Control      │ Training loops, model deployment, monitoring
    📈 Performance Analytics         │ Sub-100ms retrieval, memory efficiency

Technical Foundation:
    Vision Embeddings    │ V-JEPA-2 (6.4M dims) via memory-mapped tensors
    Language Embeddings  │ DeepSeek-V3 (7.2K dims) with semantic clustering
    Data Architecture    │ HuggingFace + SQLite + PyTorch integration
    Performance Profile  │ 21x faster than vector DBs, 140x faster than Parquet

Author: DeepEarth Project  │  License: MIT  │  Version: 2.0.0-modular
"""

# Core Flask and web framework imports
from flask import Flask, render_template, jsonify, request, send_from_directory, redirect

# Core Python imports
from datetime import datetime
import json
import logging
import warnings
from pathlib import Path

# DeepEarth modules
from services.app_initialization import initialize_app
from utils.request_parsing import (
    extract_gbif_id_from_image_id, parse_geographic_bounds, parse_temporal_filters, 
    parse_required_geographic_bounds, parse_grid_statistics_parameters, 
    parse_ecosystem_analysis_parameters, parse_attention_map_parameters, 
    parse_vision_features_parameters
)
from api.error_handling import handle_api_error, handle_image_proxy_error, handle_vision_error, handle_health_check_error
from services.umap_visualization import compute_umap_rgb_visualization
from services.attention_processing import compute_attention_map
from services.umap_processing import process_language_umap_request, parse_language_umap_parameters
from services.feature_analysis import compute_pca_raw, compute_feature_statistics
from services.vision_processing import filter_available_vision_embeddings, parse_vision_embedding_parameters
from services.color_processing import process_species_cluster_colors
from services.observation_processing import build_observation_details, get_species_observation_summary, prepare_observations_for_frontend
from services.image_processing import proxy_image_request
from services.ecosystem_processing import perform_ecosystem_analysis
from services.health_monitoring import generate_health_status
from services.vision_features import process_vision_features_request
from services.training_data import get_training_batch

# Suppress known warnings from dependencies
warnings.filterwarnings("ignore", message="'force_all_finite' was renamed to 'ensure_all_finite'")
warnings.filterwarnings("ignore", message="n_jobs value .* overridden to 1 by setting random_state")

# Initialize Flask application
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# APPLICATION BOOTSTRAP: Initialize the ML-Ready Data Engine
# ════════════════════════════════════════════════════════════════════════════

# 🚀 Bootstrap sequence: UMAP compilation → Config loading → Cache warming → Service readiness
CONFIG, cache = initialize_app(__file__)


# ════════════════════════════════════════════════════════════════════════════
# API ORCHESTRATION: Research Interface ──► ML Control Layer
# ════════════════════════════════════════════════════════════════════════════
@app.route('/')
def index():
    """
    🏠 Primary Research Interface
    
    Serves the interactive dashboard where researchers transition from
    exploratory data analysis to ML system orchestration. The gateway
    between human insight and automated discovery.
    """
    return render_template('dashboard.html', config=CONFIG)


@app.route('/api/config')
@handle_api_error
def get_config():
    """
    ⚙️ System Configuration Oracle
    
    Exposes the complete system configuration to downstream ML pipelines.
    Essential for programmatic integration and automated system coordination.
    """
    return jsonify(CONFIG)


@app.route('/api/progress')
def get_progress():
    """
    📊 Real-Time Operation Monitor
    
    Tracks the heartbeat of expensive computations (UMAP clustering,
    batch processing). Critical for coordinating distributed ML workflows
    and providing user feedback during intensive operations.
    """
    if cache.current_progress:
        return jsonify(cache.current_progress)
    return jsonify({'status': 'idle', 'message': 'No operations in progress'})


@app.route('/api/species_umap_colors')
@handle_api_error
def get_species_umap_colors():
    """
    🎨 Semantic Color Harmonization
    
    Maps species to perceptually-uniform colors derived from HDBSCAN
    clustering in embedding space. Ensures visual consistency across
    all interfaces while preserving semantic relationships.
    
    Flow: Embedding Space → HDBSCAN → HSV Color Space → RGB Mapping
    """
    result = process_species_cluster_colors(cache)
    return jsonify(result)


@app.route('/api/observations')
@handle_api_error
def get_observations():
    """
    🗺️ Spatiotemporal Data Stream
    
    Primary data pipeline serving georeferenced biodiversity observations.
    Optimized for real-time map rendering while maintaining ML pipeline
    compatibility for batch sampling and training set construction.
    """
    result = prepare_observations_for_frontend(cache)
    return jsonify(result)


@app.route('/api/observation/<int:gbif_id>')
@handle_api_error
def get_observation_details(gbif_id):
    """Get detailed information for a specific observation"""
    result = build_observation_details(cache, gbif_id)
    return jsonify(result)


@app.route('/api/image_proxy/<int:gbif_id>/<int:image_num>')
@handle_image_proxy_error
def get_image_proxy(gbif_id, image_num):
    """
    Proxy for serving images from remote URLs.
    
    Automatically transforms iNaturalist images to use 'large' size (1024px)
    instead of 'original' for faster loading.
    
    Optional query parameter:
    - size: Image size preference (original, large, medium, small)
    """
    # Get optional size parameter
    requested_size = request.args.get('size', 'large')
    
    # Delegate to service
    url = proxy_image_request(cache, gbif_id, image_num, requested_size)
    return redirect(url)


@app.route('/api/language_embeddings/umap')
@handle_api_error
def get_language_umap():
    """
    🧠 Semantic Landscape Navigator
    
    Projects DeepSeek-V3 language embeddings into 3D UMAP space for
    ecological community analysis. Supports dynamic filtering for
    hypothesis testing and automated sample selection.
    
        7,168D Semantic Space → UMAP → 3D Coordinates + HDBSCAN Clusters
    
    Essential for ML systems conducting semantic species selection.
    """
    params = parse_language_umap_parameters(request.args)
    
    result = process_language_umap_request(
        cache,
        use_precomputed=params['use_precomputed'],
        force_recompute=params['force_recompute'],
        taxon_ids=params['taxon_ids'],
        geographic_bounds=params['geographic_bounds'],
        temporal_filters=params['temporal_filters']
    )
    
    return jsonify(result)


@app.route('/api/vision_features/<int:gbif_id>')
@handle_api_error
def get_vision_features(gbif_id):
    """
    👁️ Visual Attention Decoder
    
    Extracts spatial attention patterns from V-JEPA-2 vision embeddings,
    revealing what the model "sees" as salient features. Critical for
    interpretable ML and automated quality assessment.
    
        6.4M Embedding → 8×24×24×1408 → Attention Maps → Visualizations
    """
    temporal_mode, visualization = parse_vision_features_parameters()
    
    result = process_vision_features_request(cache, gbif_id, temporal_mode, visualization)
    return jsonify(result)


@app.route('/api/grid_statistics')
@handle_api_error
def get_grid_statistics():
    """Get statistics for a geographic grid cell with optional temporal filtering"""
    # Parse parameters using utility function
    bounds, grid_size, temporal_filters = parse_grid_statistics_parameters()
    
    stats = cache.get_grid_statistics(bounds, grid_size, temporal_filters)
    if stats is None:
        raise FileNotFoundError('No data in selected region')
    
    return jsonify(stats)


@app.route('/api/vision_embeddings/available')
@handle_vision_error
def get_available_vision_embeddings():
    """Get list of observations with vision embeddings that match geographic and temporal filters"""
    # Parse parameters using service function
    params = parse_vision_embedding_parameters(request.args)
    
    # Delegate to service
    result = filter_available_vision_embeddings(
        cache, 
        params['bounds'], 
        params['max_images'], 
        params['temporal_params']
    )
    
    return jsonify(result)


@app.route('/api/vision_embeddings/umap')
@handle_api_error
def get_vision_umap():
    """Get UMAP projection of vision embeddings with geographic filtering"""
    bounds = parse_geographic_bounds()
    
    result = cache.compute_vision_umap_for_region(bounds)
    
    return jsonify({
        'embeddings': result,
        'total': len(result),
        'bounds': bounds,
        'species_colors': {}
    })


@app.route('/api/ecosystem_analysis')
@handle_api_error
def get_ecosystem_analysis():
    """
    🌳 Ecological Community Intelligence
    
    Analyzes biodiversity patterns within geographic regions using either
    language embeddings (species relationships) or vision embeddings
    (phenotypic similarity). Foundation for automated ecosystem monitoring.
    
        Geographic Bounds → Species Filter → Embedding Analysis → Community Structure
    """
    bounds, analysis_type = parse_ecosystem_analysis_parameters()
    
    result = perform_ecosystem_analysis(cache, bounds, analysis_type)
    return jsonify(result)


@app.route('/api/features/<image_id>/attention')
@handle_api_error
def get_attention_map(image_id):
    """Generate attention map for an image (compatibility endpoint)"""
    # Parse parameters using utility function
    temporal_mode, visualization, colormap, alpha = parse_attention_map_parameters()
    
    # Delegate to service
    result = compute_attention_map(image_id, cache, temporal_mode, visualization, colormap, alpha)
    return jsonify(result)


@app.route('/api/features/<image_id>/umap-rgb')
@handle_api_error
def get_umap_rgb(image_id):
    """
    🌈 Spatial-Semantic Color Synthesis
    
    Transforms vision embedding patches into false-color imagery where
    RGB values encode UMAP coordinates. Reveals the model's internal
    semantic organization as visual art.
    
        24×24 Patches → UMAP 3D → RGB Color Space → False-Color Image
    
    Used for interpretability research and artistic visualization.
    """
    result = compute_umap_rgb_visualization(image_id, cache)
    return jsonify(result)


@app.route('/api/features/<image_id>/statistics')
@handle_api_error
def get_feature_statistics(image_id):
    """Get detailed statistics for image features"""
    result = compute_feature_statistics(image_id, cache)
    return jsonify(result)


@app.route('/api/features/<image_id>/pca-raw')
@handle_api_error
def get_pca_raw(image_id):
    """Fast PCA computation endpoint optimized for instant response times"""
    result = compute_pca_raw(image_id, cache)
    return jsonify(result)


@app.route('/api/health')
@handle_health_check_error
def health_check():
    """
    ⚕️ System Vitals Monitor
    
    Comprehensive health diagnostics for production ML systems.
    Reports cache performance, data availability, memory usage,
    and service readiness. Essential for automated monitoring.
    """
    health_status = generate_health_status(cache, CONFIG)
    return jsonify(health_status)


@app.route('/api/species/<taxon_id>/observations')
@handle_api_error
def get_species_observations(taxon_id):
    """
    Get all observations for a specific species with vision embeddings.
    """
    result = get_species_observation_summary(cache, taxon_id)
    return jsonify(result)


@app.route('/api/training/batch', methods=['POST'])
@handle_api_error
def get_training_data_batch():
    """
    🚀 ML Training Data Pipeline
    
    High-performance batch loading endpoint optimized for PyTorch training.
    Returns multimodal data tensors for direct model consumption.
    
    Expected JSON payload:
        {
            "observation_ids": ["4024234567_1", "4024234568_1", ...],
            "include_vision": true,
            "include_language": true
        }
    
    Returns PyTorch-ready tensors with species, images, coordinates, 
    timestamps, and embeddings for seamless ML integration.
    """
    if not request.is_json:
        raise ValueError("Request must be JSON")
    
    data = request.get_json()
    observation_ids = data.get('observation_ids', [])
    include_vision = data.get('include_vision', True)
    include_language = data.get('include_language', True)
    
    if not observation_ids:
        raise ValueError("observation_ids cannot be empty")
    
    if len(observation_ids) > 1000:
        raise ValueError("Batch size cannot exceed 1000 observations")
    
    # Get training batch (returns dict with numpy arrays)
    batch_data = get_training_batch(
        cache, 
        observation_ids,
        include_vision=include_vision,
        include_language=include_language,
        device='cpu'  # API returns CPU tensors for JSON serialization
    )
    
    # Convert tensors to lists for JSON serialization
    json_data = {
        'species': batch_data['species'],
        'image_urls': batch_data['image_urls'],
        'locations': batch_data['locations'].tolist(),
        'timestamps': batch_data['timestamps'].tolist(),
        'metadata': batch_data['metadata']
    }
    
    if include_language:
        json_data['language_embeddings'] = batch_data['language_embeddings'].tolist()
    
    if include_vision:
        json_data['vision_embeddings'] = batch_data['vision_embeddings'].tolist()
    
    return jsonify(json_data)


@app.route('/test_frontend.html')
def test_frontend():
    """Serve test frontend page"""
    return send_from_directory('.', 'test_frontend.html')


@app.route('/deepearth-static/<path:path>')
def serve_static(path):
    """Serve static files from a unique path to avoid conflicts with main site"""
    return send_from_directory('static', path)


# ════════════════════════════════════════════════════════════════════════════
# DEVELOPMENT SERVER: For production deployment, use run_server.py
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # 🛠️ Development mode: Use run_server.py for production startup
    app.run(debug=False, host='0.0.0.0', port=5000)