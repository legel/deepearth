#!/usr/bin/env python3
"""
DeepEarth Dashboard - ML-Ready Visualization System

Interactive visualization and data indexing for the DeepEarth Self-Supervised 
Spatiotemporal Multimodality Simulator. This dashboard demonstrates multimodal
machine learning capabilities using biodiversity data.

Key Features:
- Memory-mapped tensor storage for direct ML pipeline integration
- Real-time embedding space visualization (vision and language)
- Spatiotemporal data indexing for training set construction
- Foundation for integrated ML control systems (planned)

Technical Stack:
- Vision: V-JEPA-2 embeddings (6.4M dimensions)
- Language: DeepSeek-V3 embeddings (7.2K dimensions)
- Performance: Sub-100ms retrieval via mmap indexing

Author: DeepEarth Project
License: MIT
Version: 1.0.0
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

# Suppress known warnings from dependencies
warnings.filterwarnings("ignore", message="'force_all_finite' was renamed to 'ensure_all_finite'")
warnings.filterwarnings("ignore", message="n_jobs value .* overridden to 1 by setting random_state")

# Initialize Flask application
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# APPLICATION INITIALIZATION
# ============================================================================

# Initialize application with all startup operations
CONFIG, cache = initialize_app(__file__)


# ============================================================================
# FLASK API ROUTES
# ============================================================================
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html', config=CONFIG)


@app.route('/api/config')
@handle_api_error
def get_config():
    """Get dataset configuration"""
    return jsonify(CONFIG)


@app.route('/api/progress')
def get_progress():
    """Get current progress of long-running operations"""
    if cache.current_progress:
        return jsonify(cache.current_progress)
    return jsonify({'status': 'idle', 'message': 'No operations in progress'})


@app.route('/api/species_umap_colors')
@handle_api_error
def get_species_umap_colors():
    """
    Get colors for all species based on their HDBSCAN cluster assignments.
    
    This endpoint provides consistent colors across the map and 3D visualization
    by using the cluster colors from the precomputed HDBSCAN results.
    """
    result = process_species_cluster_colors(cache)
    return jsonify(result)


@app.route('/api/observations')
@handle_api_error
def get_observations():
    """Get all observations for map display"""
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
    """Get UMAP projection of language embeddings with optional filtering"""
    # Parse request parameters
    params = parse_language_umap_parameters(request.args)
    
    # Delegate to service
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
    """Get vision features and attention maps for an observation"""
    # Parse parameters using utility function
    temporal_mode, visualization = parse_vision_features_parameters()
    
    # Delegate to service
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
    """Perform ecosystem community analysis (language or vision) for a geographic region"""
    # Parse parameters using utility function
    bounds, analysis_type = parse_ecosystem_analysis_parameters()
    
    # Delegate to service
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
    Compute UMAP RGB visualization for spatial features.
    
    This endpoint creates a false-color image where each spatial patch's
    color represents its position in UMAP space, revealing semantic structure
    in the vision features.
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
    """Health check endpoint for monitoring"""
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


@app.route('/test_frontend.html')
def test_frontend():
    """Serve test frontend page"""
    return send_from_directory('.', 'test_frontend.html')


@app.route('/deepearth-static/<path:path>')
def serve_static(path):
    """Serve static files from a unique path to avoid conflicts with main site"""
    return send_from_directory('static', path)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)