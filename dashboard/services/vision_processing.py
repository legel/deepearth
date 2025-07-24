"""
Vision processing services for DeepEarth Dashboard.

Provides vision embedding filtering, availability checking, and processing
for geographic and temporal constraints.
"""

import logging

logger = logging.getLogger(__name__)


def filter_available_vision_embeddings(cache, bounds, max_images=250, temporal_params=None):
    """
    Get list of observations with vision embeddings that match geographic and temporal filters.
    
    Args:
        cache: UnifiedDataCache instance for data access
        bounds: Dict with geographic bounds (north, south, east, west)
        max_images: Maximum number of images to return
        temporal_params: Dict with temporal filtering parameters
        
    Returns:
        dict: Filtered observations with vision embeddings
        
    Raises:
        ValueError: If no vision embeddings are available
    """
    logger.info(f"Filtering vision embeddings: bounds={bounds}, max_images={max_images}")
    
    # Load data
    obs = cache.load_observations()
    vision_meta = cache.load_vision_metadata()
    
    # Check if vision metadata exists
    if vision_meta is None or len(vision_meta) == 0:
        raise ValueError('No vision embeddings available')
    
    # Apply filtering
    filtered_obs = _apply_vision_filters(obs, bounds, temporal_params)
    
    # Find observations with vision embeddings
    obs_with_vision = filtered_obs[filtered_obs['has_vision'] == True]
    
    # Limit to max_images
    if len(obs_with_vision) > max_images:
        obs_with_vision = obs_with_vision.sample(n=max_images, random_state=42)
    
    # Return minimal metadata for preloading
    result = {
        'count': len(obs_with_vision),
        'observations': obs_with_vision[['gbif_id', 'taxon_id']].to_dict('records')
    }
    
    logger.info(f"Found {len(obs_with_vision)} observations with vision embeddings")
    return result


def _apply_vision_filters(obs, bounds, temporal_params):
    """
    Apply geographic and temporal filters to observations.
    
    Args:
        obs: Observations DataFrame
        bounds: Geographic bounds dict
        temporal_params: Temporal parameters dict
        
    Returns:
        DataFrame: Filtered observations
    """
    # Apply geographic bounds
    mask = (
        (obs['latitude'] >= bounds['south']) &
        (obs['latitude'] <= bounds['north']) &
        (obs['longitude'] >= bounds['west']) &
        (obs['longitude'] <= bounds['east'])
    )
    
    # Apply temporal filters if provided
    if temporal_params:
        mask = _apply_temporal_vision_filters(obs, mask, temporal_params)
    
    return obs[mask]


def _apply_temporal_vision_filters(obs, mask, temporal_params):
    """Apply temporal filters to vision observations."""
    # Year filters
    year_min = temporal_params.get('year_min', 2010)
    year_max = temporal_params.get('year_max', 2025)
    mask &= (obs['year'] >= year_min) & (obs['year'] <= year_max)
    
    # Month filters
    month_min = temporal_params.get('month_min', 1)
    month_max = temporal_params.get('month_max', 12)
    if 'month' in obs.columns:
        mask &= (obs['month'].isna() | ((obs['month'] >= month_min) & (obs['month'] <= month_max)))
    
    # Hour filters
    hour_min = temporal_params.get('hour_min', 0)
    hour_max = temporal_params.get('hour_max', 23)
    if 'hour' in obs.columns:
        mask &= (obs['hour'].isna() | ((obs['hour'] >= hour_min) & (obs['hour'] <= hour_max)))
    
    return mask


def parse_vision_embedding_parameters(request_args):
    """
    Parse parameters for vision embedding requests.
    
    Args:
        request_args: Flask request.args object
        
    Returns:
        dict: Parsed parameters including bounds and temporal filters
    """
    # Parse geographic bounds using consistent defaults
    bounds = _parse_bounds_from_args(request_args)
    
    # Parse other parameters
    max_images = int(request_args.get('max_images', 250))
    
    # Parse temporal parameters
    temporal_params = {
        'year_min': request_args.get('year_min', type=int, default=2010),
        'year_max': request_args.get('year_max', type=int, default=2025),
        'month_min': request_args.get('month_min', type=int, default=1),
        'month_max': request_args.get('month_max', type=int, default=12),
        'hour_min': request_args.get('hour_min', type=int, default=0),
        'hour_max': request_args.get('hour_max', type=int, default=23)
    }
    
    return {
        'bounds': bounds,
        'max_images': max_images,
        'temporal_params': temporal_params
    }


def _parse_bounds_from_args(request_args):
    """Parse geographic bounds from request args with consistent defaults."""
    return {
        'north': float(request_args.get('north', 90)),
        'south': float(request_args.get('south', -90)),
        'east': float(request_args.get('east', 180)),
        'west': float(request_args.get('west', -180))
    }