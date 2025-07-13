"""
Observation processing services for DeepEarth Dashboard.

Provides observation data formatting, species filtering, and metadata 
preparation for frontend display.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def build_observation_details(cache, gbif_id):
    """
    Get detailed information for a specific observation.
    
    Args:
        cache: UnifiedDataCache instance for data access
        gbif_id: GBIF identifier for the observation
        
    Returns:
        dict: Detailed observation information with images and embeddings
        
    Raises:
        FileNotFoundError: If observation not found
    """
    # Get observation using HuggingFaceDataLoader
    obs_data = cache.loader.get_observation(gbif_id)
    if obs_data is None:
        raise FileNotFoundError('Observation not found')
    
    taxon_id = obs_data['taxon_id']
    
    # Process images
    images = _process_image_urls(obs_data, gbif_id, taxon_id)
    
    # Check embedding availability
    has_vision = bool(obs_data.get('has_vision', False))
    has_language = bool(obs_data.get('language_embedding') is not None)
    
    # Build comprehensive result
    result = {
        'gbif_id': int(gbif_id),
        'taxon_id': taxon_id,
        'taxon_name': obs_data['taxon_name'],
        'location': {
            'latitude': float(obs_data['latitude']),
            'longitude': float(obs_data['longitude'])
        },
        'temporal': _build_temporal_info(obs_data),
        'images': images,
        'has_vision_embedding': has_vision,
        'has_language_embedding': has_language,
        'split': obs_data.get('split', 'unknown')
    }
    
    logger.info(f"Built observation details for GBIF {gbif_id}: {obs_data['taxon_name']}")
    return result


def get_species_observation_summary(cache, taxon_id, max_observations=1000):
    """
    Get all observations for a specific species with vision embeddings.
    
    This is much more efficient than loading all observations and filtering client-side.
    
    Args:
        cache: UnifiedDataCache instance for data access
        taxon_id: Species taxon identifier
        max_observations: Maximum number of observations to return
        
    Returns:
        dict: Species observation summary with vision-enabled observations
    """
    obs = cache.load_observations()
    vision_meta = cache.load_vision_metadata()
    
    # Filter observations for this taxon
    species_obs = obs[obs['taxon_id'] == taxon_id]
    
    # Get vision-enabled observations
    vision_gbif_ids = _get_vision_gbif_ids(vision_meta)
    
    # Prepare observation data
    observations = _prepare_species_observations(species_obs, vision_gbif_ids)
    
    # Apply truncation if needed
    truncated = len(observations) > max_observations
    if truncated:
        observations = observations[:max_observations]
    
    result = {
        'taxon_id': taxon_id,
        'taxon_name': species_obs.iloc[0]['taxon_name'] if len(species_obs) > 0 else 'Unknown',
        'total_observations': len(species_obs),
        'observations_with_vision': len(observations) if not truncated else max_observations,
        'observations': observations,
        'truncated': truncated,
        'max_returned': max_observations if truncated else len(observations)
    }
    
    logger.info(f"Generated species summary for {taxon_id}: {len(observations)} vision observations")
    return result


def prepare_observations_for_frontend(cache):
    """
    Get all observations formatted for map display.
    
    Args:
        cache: UnifiedDataCache instance for data access
        
    Returns:
        dict: Observations data with bounds and metadata
    """
    obs = cache.load_observations()
    vision_meta = cache.load_vision_metadata()
    
    # Get unique gbif_ids that have vision embeddings
    vision_gbif_ids = _get_vision_gbif_ids(vision_meta)
    
    # Prepare data for frontend efficiently
    data = _format_observations_for_frontend(obs, vision_gbif_ids)
    
    # Calculate geographic bounds
    bounds = {
        'north': float(obs['latitude'].max()),
        'south': float(obs['latitude'].min()),
        'east': float(obs['longitude'].max()),
        'west': float(obs['longitude'].min())
    }
    
    result = {
        'observations': data,
        'total': len(data),
        'bounds': bounds
    }
    
    logger.info(f"Prepared {len(data)} observations for frontend display")
    return result


def _process_image_urls(obs_data, gbif_id, taxon_id):
    """Process image URLs into structured format."""
    images = []
    image_urls = obs_data.get('image_urls', [])
    if isinstance(image_urls, list):
        for i, url in enumerate(image_urls):
            images.append({
                'filename': f"image_{i+1}.jpg",
                'image_id': f"gbif_{gbif_id}_taxon_{taxon_id}_img_{i+1}",
                'url': url,
                'local_url': f"/api/image_proxy/{gbif_id}/{i+1}"
            })
    return images


def _build_temporal_info(obs_data):
    """Build temporal information structure."""
    return {
        'year': int(obs_data['year']),
        'month': int(obs_data['month']) if pd.notna(obs_data['month']) else None,
        'day': int(obs_data['day']) if pd.notna(obs_data['day']) else None,
        'hour': int(obs_data['hour']) if pd.notna(obs_data['hour']) else None,
        'minute': int(obs_data['minute']) if pd.notna(obs_data['minute']) else None,
        'second': int(obs_data['second']) if pd.notna(obs_data['second']) else None
    }


def _get_vision_gbif_ids(vision_meta):
    """Get set of GBIF IDs that have vision embeddings."""
    vision_gbif_ids = set()
    if vision_meta is not None:
        vision_gbif_ids = set(vision_meta['gbif_id'].unique())
    return vision_gbif_ids


def _prepare_species_observations(species_obs, vision_gbif_ids):
    """Prepare observations for a specific species."""
    observations = []
    for _, row in species_obs.iterrows():
        has_vision = row.get('has_vision', False) or int(row['gbif_id']) in vision_gbif_ids
        if has_vision:
            observations.append({
                'gbif_id': int(row['gbif_id']),
                'lat': float(row['latitude']),
                'lon': float(row['longitude']),
                'year': int(row['year']),
                'month': int(row.get('month', 0)) if pd.notna(row.get('month')) else None,
                'has_vision': True
            })
    return observations


def _format_observations_for_frontend(obs, vision_gbif_ids):
    """Format all observations for frontend display."""
    data = []
    for _, row in obs.iterrows():
        has_vision = row.get('has_vision', False) or int(row['gbif_id']) in vision_gbif_ids
        
        data.append({
            'gbif_id': int(row['gbif_id']),
            'taxon_id': row['taxon_id'],
            'taxon_name': row['taxon_name'],
            'lat': row['latitude'],
            'lon': row['longitude'],
            'year': int(row['year']),
            'month': int(row['month']) if pd.notna(row['month']) else None,
            'day': int(row['day']) if pd.notna(row['day']) else None,
            'hour': int(row['hour']) if pd.notna(row['hour']) else None,
            'minute': int(row['minute']) if pd.notna(row['minute']) else None,
            'second': int(row['second']) if pd.notna(row['second']) else None,
            'has_vision': has_vision,
            'split': row.get('split', 'unknown')
        })
    return data