"""
Attention processing services for DeepEarth Dashboard.

Provides attention map computation for vision features with detailed
performance monitoring and feature validation.
"""

import torch
from datetime import datetime
import logging

from utils.request_parsing import extract_gbif_id_from_image_id
from vision.attention_utils import generate_attention_overlay

logger = logging.getLogger(__name__)


def compute_attention_map(image_id, cache, temporal_mode='mean', visualization='l2norm', 
                         colormap='plasma', alpha=0.7):
    """
    Generate attention map for an image with comprehensive logging and timing.
    
    Args:
        image_id: Image identifier in format "gbif_XXXXXXX_taxon_XXXXXXX_img_N"
        cache: UnifiedDataCache instance for data access
        temporal_mode: 'mean' for static or 'temporal' for animation
        visualization: Method to convert features to attention ('l2norm', 'pca1', etc.)
        colormap: Matplotlib colormap name for visualization
        alpha: Transparency for overlay
        
    Returns:
        dict: Response data containing attention map(s) and statistics
        
    Raises:
        ValueError: If image_id format is invalid
        FileNotFoundError: If observation or vision features not found
    """
    start_time = datetime.now()
    logger.info(f"🔄 Loading attention map for image: {image_id}")
    
    # Extract GBIF ID from image_id
    try:
        gbif_id, img_num = extract_gbif_id_from_image_id(image_id)
    except ValueError as e:
        logger.error(f"❌ {str(e)}")
        raise ValueError(str(e))
    
    logger.info(f"📍 Extracted GBIF ID: {gbif_id}")
    
    # Load and validate observation data
    obs_data = _load_and_validate_observation(cache, gbif_id)
    
    # Load and validate vision features
    features = _load_and_validate_features(cache, gbif_id, obs_data, start_time)
    
    # Parse processing parameters
    logger.info(f"🎛️ Parameters: temporal={temporal_mode}, viz={visualization}, colormap={colormap}, alpha={alpha}")
    
    # Compute attention with timing
    attention = _compute_spatial_attention(cache, features, temporal_mode, visualization, start_time)
    
    # Generate response based on temporal mode
    if temporal_mode == 'mean':
        return _generate_spatial_response(attention, colormap, alpha, start_time)
    else:
        return _generate_temporal_response(attention, colormap, alpha)


def _load_and_validate_observation(cache, gbif_id):
    """Load observation data and validate it exists."""
    logger.info(f"🧠 Loading vision embedding for GBIF {gbif_id}")
    
    obs_data = cache.loader.get_observation(gbif_id)
    if obs_data is None:
        logger.error(f"❌ Observation {gbif_id} not found in dataset")
        raise FileNotFoundError(f"Observation {gbif_id} not found")
    
    logger.info(f"📋 Observation details: species='{obs_data['taxon_name']}', taxon_id={obs_data['taxon_id']}, has_vision={obs_data.get('has_vision', False)}")
    return obs_data


def _load_and_validate_features(cache, gbif_id, obs_data, start_time):
    """Load vision features and validate their quality."""
    checkpoint_1 = datetime.now()
    logger.info(f"⏱️ Time to check observation: {(checkpoint_1 - start_time).total_seconds():.2f}s")
    
    features = cache.get_vision_embedding(gbif_id, obs_data['taxon_id'], 1)
    if features is None:
        logger.warning(f"❌ No vision features found for GBIF {gbif_id} (species: {obs_data['taxon_name']})")
        raise FileNotFoundError(f"Vision features not found for GBIF {gbif_id}")
    
    checkpoint_2 = datetime.now()
    logger.info(f"⏱️ Time to load embedding: {(checkpoint_2 - checkpoint_1).total_seconds():.2f}s")
    logger.info(f"✅ Loaded vision features for {obs_data['taxon_name']}, shape: {features.shape}, dtype: {features.dtype}")
    
    # Validate feature quality
    _validate_feature_quality(features)
    
    return features


def _validate_feature_quality(features):
    """Validate the quality and characteristics of vision features."""
    feature_stats = {
        'min': float(features.min().item()),
        'max': float(features.max().item()), 
        'mean': float(features.mean().item()),
        'std': float(features.std().item())
    }
    logger.info(f"📊 Feature stats: {feature_stats}")
    
    # Check for potential issues
    if feature_stats['max'] == feature_stats['min']:
        logger.warning(f"⚠️ Features appear to be constant (all same value: {feature_stats['max']})")
    if abs(feature_stats['mean']) > 100:
        logger.warning(f"⚠️ Features have unusually large magnitude (mean: {feature_stats['mean']})")


def _compute_spatial_attention(cache, features, temporal_mode, visualization, start_time):
    """Compute spatial attention with performance timing."""
    checkpoint_3 = datetime.now()
    logger.info("🔥 Computing spatial attention...")
    
    attention = cache.compute_spatial_attention(features, temporal_mode, 'mean', visualization)
    
    checkpoint_4 = datetime.now()
    logger.info(f"⏱️ Time to compute attention: {(checkpoint_4 - checkpoint_3).total_seconds():.2f}s")
    logger.info(f"✅ Attention computed, shape: {attention.shape if hasattr(attention, 'shape') else type(attention)}")
    
    return attention


def _generate_spatial_response(attention, colormap, alpha, start_time):
    """Generate response for spatial (static) attention mode."""
    logger.info(f"🎨 Generating attention overlay with {colormap} colormap")
    checkpoint_5 = datetime.now()
    
    attention_img = generate_attention_overlay(attention, colormap, alpha)
    
    checkpoint_6 = datetime.now()
    logger.info(f"⏱️ Time to generate overlay: {(checkpoint_6 - checkpoint_5).total_seconds():.2f}s")
    
    # Calculate total processing time
    processing_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"⚡ Attention map completed in {processing_time:.2f}s")
    
    response_data = {
        'mode': 'spatial',
        'attention_map': attention_img,
        'stats': {
            'max': float(attention.max().item() if isinstance(attention, torch.Tensor) else attention.max()),
            'mean': float(attention.mean().item() if isinstance(attention, torch.Tensor) else attention.mean()),
            'std': float(attention.std().item() if isinstance(attention, torch.Tensor) else attention.std())
        }
    }
    
    checkpoint_7 = datetime.now()
    logger.info(f"⏱️ Time to prepare response: {(checkpoint_7 - checkpoint_6).total_seconds():.2f}s")
    
    return response_data


def _generate_temporal_response(attention, colormap, alpha):
    """Generate response for temporal (animated) attention mode."""
    attention_frames = []
    for t in range(8):
        attention_img = generate_attention_overlay(attention[t], colormap, alpha)
        attention_frames.append(attention_img)
    
    return {
        'mode': 'temporal',
        'attention_frames': attention_frames,
        'num_frames': 8
    }