"""
Health monitoring services for DeepEarth Dashboard.

Provides comprehensive system health status including dataset,
cache statistics, and precomputed data availability.
"""

from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def generate_health_status(cache, config):
    """
    Generate comprehensive health status for the DeepEarth Dashboard.
    
    Args:
        cache: UnifiedDataCache instance for data access
        config: Application configuration dictionary
        
    Returns:
        dict: Comprehensive health status with dataset, cache, and system info
    """
    logger.info("Generating health status report")
    
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'dataset': _get_dataset_info(config),
        'data_loaded': _get_data_loaded_stats(cache),
        'cache_stats': _get_cache_statistics(cache),
        'mmap_loader': _get_mmap_loader_stats(cache),
        'precomputed_data': _get_precomputed_data_status(cache)
    }
    
    logger.info(f"Health status generated successfully: {health_status['status']}")
    return health_status


def _get_dataset_info(config):
    """Get dataset configuration information."""
    return {
        'name': config['dataset_name'],
        'version': config['dataset_version']
    }


def _get_data_loaded_stats(cache):
    """Get statistics about loaded data."""
    return {
        'observations': len(cache.loader.observations) if cache.loader.observations is not None else 0,
        'vision_metadata': len(cache.loader.vision_index) if cache.loader.vision_index is not None else 0,
        'species': len(cache.loader.taxon_to_gbifs) if hasattr(cache.loader, 'taxon_to_gbifs') else 0
    }


def _get_cache_statistics(cache):
    """Get cache performance statistics."""
    return cache.loader.get_cache_stats()


def _get_mmap_loader_stats(cache):
    """Get memory-mapped loader statistics."""
    return {
        'enabled': cache.mmap_loader is not None,
        'cache_stats': cache.mmap_loader.get_cache_stats() if cache.mmap_loader else None
    }


def _get_precomputed_data_status(cache):
    """Get status of precomputed data availability."""
    return {
        'language_umap': cache.precomputed_language_umap is not None,
        'language_clusters': cache.language_clusters is not None,
        'vision_umap': cache.precomputed_vision_umap is not None
    }