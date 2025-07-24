"""
Application initialization services for DeepEarth Dashboard.

Handles configuration loading, cache setup, background task management,
and other startup operations.
"""

import json
import logging
import threading
from pathlib import Path

from data_cache import UnifiedDataCache
from umap_optimized import warm_up_umap

logger = logging.getLogger(__name__)


def initialize_app(app_file_path=None):
    """
    Initialize the DeepEarth Dashboard application.
    
    Performs all startup operations including:
    - UMAP warm-up for optimal performance
    - Configuration loading
    - Data cache initialization  
    - Background precomputation setup
    
    Args:
        app_file_path: Path to the main application file (for relative path resolution)
        
    Returns:
        tuple: (CONFIG, cache) - Configuration dict and initialized cache
    """
    logger.info("Starting DeepEarth Dashboard initialization...")
    
    # Pre-warm UMAP for optimal performance
    logger.info("Warming up UMAP...")
    warm_up_umap()
    
    # Load configuration and initialize data paths
    config, data_dir = _load_configuration(app_file_path)
    logger.info(f"Loaded configuration: {config['dataset_name']} v{config['dataset_version']}")
    
    # Initialize global data cache
    logger.info("Initializing data cache...")
    cache = UnifiedDataCache(Path(app_file_path).parent / "dataset_config.json" if app_file_path else "dataset_config.json")
    
    # Start background precomputation if enabled
    if config['caching']['precompute_language_umap']:
        _start_background_precomputation(cache)
    
    logger.info("DeepEarth Dashboard initialization complete")
    return config, cache


def _load_configuration(app_file_path=None):
    """Load application configuration from JSON file."""
    base_dir = Path(app_file_path).parent if app_file_path else Path(__file__).parent.parent
    config_path = base_dir / "dataset_config.json"
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    data_dir = base_dir / config['data_paths']['base_dir']
    return config, data_dir


def _start_background_precomputation(cache):
    """Start background thread for expensive UMAP precomputation."""
    logger.info("Starting background precomputation of language UMAP clusters...")
    
    def precompute_background():
        """Background task to precompute expensive UMAP operations"""
        try:
            cache.compute_and_cache_language_umap_clusters()
            logger.info("Background precomputation completed successfully")
        except Exception as e:
            logger.error(f"Background precomputation failed: {e}")
    
    thread = threading.Thread(target=precompute_background, daemon=True)
    thread.start()
    logger.info("Background precomputation thread started")