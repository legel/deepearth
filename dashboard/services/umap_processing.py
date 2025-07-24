"""
UMAP processing services for DeepEarth Dashboard.

Provides language embedding UMAP computation with filtering, precomputed data handling,
and geographic/temporal filtering capabilities.
"""

import logging

logger = logging.getLogger(__name__)


def process_language_umap_request(cache, use_precomputed=True, force_recompute=False, 
                                taxon_ids=None, geographic_bounds=None, temporal_filters=None):
    """
    Process language UMAP request with comprehensive filtering and caching logic.
    
    Args:
        cache: UnifiedDataCache instance for data access
        use_precomputed: Whether to use precomputed full dataset
        force_recompute: Whether to force recomputation of precomputed data
        taxon_ids: List of specific taxon IDs to include
        geographic_bounds: Dict with 'north', 'south', 'east', 'west' keys
        temporal_filters: Dict with temporal filtering parameters
        
    Returns:
        dict: Response data containing embeddings, clusters, and metadata
        
    Raises:
        ValueError: If not enough data for UMAP computation
    """
    logger.info(f"Processing language UMAP request: precomputed={use_precomputed}, force={force_recompute}")
    
    # Handle force recomputation
    if force_recompute:
        _force_recompute_data(cache)
    
    # Apply geographic and temporal filtering if provided
    filtered_taxon_ids = _apply_geographic_temporal_filtering(
        cache, geographic_bounds, temporal_filters, taxon_ids
    )
    
    # Use filtered taxon IDs if any filtering was applied
    if filtered_taxon_ids is not None:
        taxon_ids = filtered_taxon_ids
        use_precomputed = False  # Don't use precomputed when filtering
    
    # Return precomputed data if available and no filtering
    if use_precomputed and not taxon_ids and cache.precomputed_language_umap:
        logger.info("Using precomputed language UMAP data")
        return {
            'embeddings': cache.precomputed_language_umap,
            'clusters': cache.language_clusters,
            'total': len(cache.precomputed_language_umap),
            'precomputed': True
        }
    
    # Fall back to dynamic computation
    logger.info("Computing dynamic language UMAP")
    result = _compute_dynamic_umap(cache, taxon_ids)
    
    return {
        'embeddings': result,
        'total': len(result),
        'precomputed': False
    }


def _force_recompute_data(cache):
    """Force recomputation of precomputed language UMAP and clusters."""
    logger.info("Forcing recomputation of language UMAP clusters")
    cache.precomputed_language_umap = None
    cache.language_clusters = None
    cache.compute_and_cache_language_umap_clusters()


def _apply_geographic_temporal_filtering(cache, geographic_bounds, temporal_filters, existing_taxon_ids):
    """
    Apply geographic and temporal filtering to determine taxon IDs.
    
    Returns:
        list or None: Filtered taxon IDs if filtering was applied, None otherwise
    """
    if geographic_bounds is None:
        return None
    
    logger.info(f"Applying geographic filtering: {geographic_bounds}")
    
    obs = cache.load_observations()
    
    # Create geographic mask
    mask = (
        (obs['latitude'] >= geographic_bounds['south']) & 
        (obs['latitude'] <= geographic_bounds['north']) &
        (obs['longitude'] >= geographic_bounds['west']) & 
        (obs['longitude'] <= geographic_bounds['east'])
    )
    
    # Apply temporal filters if provided
    if temporal_filters:
        mask = _apply_temporal_filters(obs, mask, temporal_filters)
    
    # Get unique taxon IDs from filtered observations
    filtered_obs = obs[mask]
    taxon_ids = filtered_obs['taxon_id'].unique().tolist()
    taxon_ids = [str(tid) for tid in taxon_ids]
    
    logger.info(f"Geographic/temporal filtering resulted in {len(taxon_ids)} unique taxa")
    return taxon_ids


def _apply_temporal_filters(obs, mask, temporal_filters):
    """Apply temporal filtering to the observation mask."""
    if not temporal_filters:
        return mask
    
    logger.info(f"Applying temporal filters: {temporal_filters}")
    
    # Apply year filters
    year_min = temporal_filters.get('year_min')
    year_max = temporal_filters.get('year_max')
    if year_min is not None and year_max is not None:
        mask = mask & (obs['year'] >= year_min) & (obs['year'] <= year_max)
    
    # Apply month filters
    month_min = temporal_filters.get('month_min')
    month_max = temporal_filters.get('month_max')
    if month_min is not None and month_max is not None:
        mask = mask & (obs['month'] >= month_min) & (obs['month'] <= month_max)
    
    # Apply hour filters
    hour_min = temporal_filters.get('hour_min')
    hour_max = temporal_filters.get('hour_max')
    if hour_min is not None and hour_max is not None:
        mask = mask & (obs['hour'] >= hour_min) & (obs['hour'] <= hour_max)
    
    return mask


def _compute_dynamic_umap(cache, taxon_ids):
    """Compute UMAP for specified taxon IDs or raise error if insufficient data."""
    # Prepare taxon IDs for computation
    if taxon_ids:
        taxon_ids = [str(tid) for tid in taxon_ids]
        logger.info(f"Computing UMAP for {len(taxon_ids)} specified taxa")
    else:
        taxon_ids = None
        logger.info("Computing UMAP for all available taxa")
    
    # Compute UMAP using cache
    result = cache.compute_language_umap(taxon_ids)
    if result is None:
        logger.error("Not enough data for UMAP computation")
        raise ValueError('Not enough data for UMAP')
    
    logger.info(f"Successfully computed UMAP with {len(result)} embeddings")
    return result


def parse_language_umap_parameters(request_args):
    """
    Parse and validate parameters for language UMAP requests.
    
    Args:
        request_args: Flask request.args object
        
    Returns:
        dict: Parsed parameters including geographic bounds and temporal filters
    """
    # Parse basic parameters
    use_precomputed = request_args.get('precomputed', 'true').lower() == 'true'
    force_recompute = request_args.get('force_recompute', 'false').lower() == 'true'
    taxon_ids = request_args.getlist('taxon_ids')
    
    # Parse geographic bounds
    geographic_bounds = None
    north = request_args.get('north', type=float)
    south = request_args.get('south', type=float)
    east = request_args.get('east', type=float)
    west = request_args.get('west', type=float)
    
    if all(coord is not None for coord in [north, south, east, west]):
        geographic_bounds = {
            'north': north,
            'south': south,
            'east': east,
            'west': west
        }
    
    # Parse temporal filters
    temporal_filters = {}
    temporal_params = ['year_min', 'year_max', 'month_min', 'month_max', 'hour_min', 'hour_max']
    for param in temporal_params:
        value = request_args.get(param, type=int)
        if value is not None:
            temporal_filters[param] = value
    
    temporal_filters = temporal_filters if temporal_filters else None
    
    return {
        'use_precomputed': use_precomputed,
        'force_recompute': force_recompute,
        'taxon_ids': taxon_ids,
        'geographic_bounds': geographic_bounds,
        'temporal_filters': temporal_filters
    }