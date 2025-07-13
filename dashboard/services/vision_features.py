"""
Vision features processing services for DeepEarth Dashboard.

Provides vision feature extraction, attention map computation, and
visualization generation for observations.
"""

import logging
from vision.attention_utils import generate_attention_overlay

logger = logging.getLogger(__name__)


def process_vision_features_request(cache, gbif_id, temporal_mode='mean', visualization='l2norm'):
    """
    Process vision features request for an observation.
    
    Loads vision embeddings, computes attention maps, and generates
    appropriate visualizations based on temporal mode.
    
    Args:
        cache: UnifiedDataCache instance for data access
        gbif_id: GBIF identifier for the observation
        temporal_mode: Temporal processing mode ('mean' or other)
        visualization: Visualization type for attention computation
        
    Returns:
        dict: Vision features response with attention maps and statistics
        
    Raises:
        FileNotFoundError: If observation or vision features not found
    """
    logger.info(f"Processing vision features for GBIF {gbif_id}")
    
    # Load observation and validate
    obs_data, taxon_id = _load_and_validate_observation(cache, gbif_id)
    
    # Load vision embedding
    features = cache.get_vision_embedding(gbif_id, taxon_id, 1)
    if features is None:
        raise FileNotFoundError('Vision features not found')
    
    # Compute attention using the cache method
    attention = cache.compute_spatial_attention(features, temporal_mode, 'mean', visualization)
    
    # Generate appropriate response based on temporal mode
    if temporal_mode == 'mean':
        return _generate_spatial_response(attention)
    else:
        return _generate_temporal_response(attention)


def _load_and_validate_observation(cache, gbif_id):
    """Load observation data and validate existence."""
    obs = cache.load_observations()
    obs_data = obs[obs['gbif_id'] == gbif_id]
    
    if len(obs_data) == 0:
        raise FileNotFoundError('Observation not found')
    
    taxon_id = obs_data.iloc[0]['taxon_id']
    logger.info(f"Found observation for GBIF {gbif_id}, taxon {taxon_id}")
    
    return obs_data, taxon_id


def _generate_spatial_response(attention):
    """Generate response for spatial attention visualization."""
    # Generate visualization
    attention_img = generate_attention_overlay(attention)
    
    return {
        'mode': 'spatial',
        'attention_map': attention_img,
        'stats': {
            'max': float(attention.max()),
            'mean': float(attention.mean()),
            'std': float(attention.std())
        }
    }


def _generate_temporal_response(attention):
    """Generate response for temporal attention sequence."""
    # Return temporal sequence
    attention_frames = []
    for t in range(8):
        attention_img = generate_attention_overlay(attention[t])
        attention_frames.append(attention_img)
    
    return {
        'mode': 'temporal',
        'attention_frames': attention_frames,
        'num_frames': 8
    }