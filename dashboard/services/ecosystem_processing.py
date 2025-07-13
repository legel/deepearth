"""
Ecosystem analysis services for DeepEarth Dashboard.

Provides ecosystem community analysis for geographic regions using
both language and vision embeddings for biodiversity insights.
"""

import logging

logger = logging.getLogger(__name__)


def perform_ecosystem_analysis(cache, bounds, analysis_type='language'):
    """
    Perform ecosystem community analysis for a geographic region.
    
    Args:
        cache: UnifiedDataCache instance for data access
        bounds: Dict with geographic bounds (north, south, east, west)
        analysis_type: Type of analysis ('language' or 'vision')
        
    Returns:
        dict: Analysis results with embeddings and metadata
        
    Raises:
        ValueError: If insufficient species in region for analysis
    """
    logger.info(f"Performing {analysis_type} ecosystem analysis for bounds: {bounds}")
    
    if analysis_type == 'language':
        return _analyze_language_ecosystem(cache, bounds)
    else:
        return _analyze_vision_ecosystem(cache, bounds)


def _analyze_language_ecosystem(cache, bounds):
    """
    Analyze ecosystem using language embeddings for species in region.
    
    Args:
        cache: UnifiedDataCache instance
        bounds: Geographic bounds dict
        
    Returns:
        dict: Language ecosystem analysis results
        
    Raises:
        ValueError: If insufficient species for analysis
    """
    # Get taxa in region
    obs = cache.load_observations()
    mask = (
        (obs['latitude'] >= bounds['south']) &
        (obs['latitude'] <= bounds['north']) &
        (obs['longitude'] >= bounds['west']) &
        (obs['longitude'] <= bounds['east'])
    )
    region_taxa = obs[mask]['taxon_id'].unique()
    
    if len(region_taxa) < 3:
        raise ValueError('Not enough species in region')
    
    # Compute UMAP for these taxa
    result = cache.compute_language_umap(region_taxa)
    
    logger.info(f"Language ecosystem analysis complete: {len(result)} species")
    return {
        'type': 'language',
        'embeddings': result,
        'total_species': len(result)
    }


def _analyze_vision_ecosystem(cache, bounds):
    """
    Analyze ecosystem using vision embeddings for observations in region.
    
    Args:
        cache: UnifiedDataCache instance
        bounds: Geographic bounds dict
        
    Returns:
        dict: Vision ecosystem analysis results
    """
    # Vision embeddings analysis
    result = cache.compute_vision_umap_for_region(bounds)
    
    logger.info(f"Vision ecosystem analysis complete: {len(result)} observations")
    return {
        'type': 'vision',
        'embeddings': result,
        'total_observations': len(result)
    }