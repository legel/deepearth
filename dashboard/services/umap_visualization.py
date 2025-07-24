"""
UMAP visualization services for DeepEarth Dashboard.

Provides UMAP RGB visualization computation for spatial vision features,
creating false-color images where each spatial patch's color represents
its position in UMAP space to reveal semantic structure.
"""

import numpy as np
import torch
from datetime import datetime
from PIL import Image
import io
import base64
import logging
import gc

from umap_optimized import OptimizedUMAP
from utils.request_parsing import extract_gbif_id_from_image_id

logger = logging.getLogger(__name__)


def compute_umap_rgb_visualization(image_id, cache):
    """
    Compute UMAP RGB visualization for spatial vision features.
    
    Creates a false-color image where each spatial patch's color represents
    its position in UMAP space, revealing semantic structure in the vision features.
    
    Args:
        image_id: Image identifier in format "gbif_XXXXXXX_taxon_XXXXXXX_img_N"
        cache: UnifiedDataCache instance for data access
        
    Returns:
        dict: Response data containing UMAP RGB image and metadata
        
    Raises:
        ValueError: If image_id format is invalid
        FileNotFoundError: If observation or vision features not found
    """
    start_time = datetime.now()
    logger.info(f"🌈 Computing UMAP RGB for image: {image_id}")
    
    # Check cache first
    cache_key = f"umap_rgb_{image_id}"
    if hasattr(cache, '_umap_rgb_cache') and cache_key in cache._umap_rgb_cache:
        logger.info(f"✅ Using cached UMAP RGB for {image_id}")
        return cache._umap_rgb_cache[cache_key]
    
    # Extract GBIF ID from image_id
    try:
        gbif_id, img_num = extract_gbif_id_from_image_id(image_id)
    except ValueError as e:
        logger.error(f"❌ {str(e)}")
        raise ValueError(str(e))
    
    logger.info(f"📍 Computing UMAP for GBIF ID: {gbif_id}")
    
    # Get observation data first
    obs_data = cache.loader.get_observation(gbif_id)
    if obs_data is None:
        logger.error(f"❌ Observation {gbif_id} not found")
        raise FileNotFoundError(f"Observation {gbif_id} not found")
    
    # Load vision embedding
    features = cache.get_vision_embedding(gbif_id, obs_data['taxon_id'], 1)
    if features is None:
        logger.warning(f"❌ No vision features found for GBIF {gbif_id}")
        raise FileNotFoundError(f"Vision features not found for GBIF {gbif_id}")
    
    logger.info(f"✅ Loaded vision features, shape: {features.shape}")
    
    # Use fast PyTorch view operations
    features = features.view(8, 576, 1408).mean(dim=0)  # [576, 1408]
    logger.info(f"🔄 Reshaped to spatial features: {features.shape}")
    
    # Convert to numpy for sklearn operations
    features_flat = features.detach().cpu().numpy()  # [576, 1408]
    logger.info(f"📊 Feature range: min={features_flat.min():.3f}, max={features_flat.max():.3f}, mean={features_flat.mean():.3f}")
    
    # Free the torch tensor to save memory
    del features
    gc.collect()
    
    # Apply UMAP to high-dimensional features
    logger.info("🗺️ Applying UMAP directly to high-dimensional features...")
    coords_3d = _compute_umap_coordinates(features_flat)
    logger.info(f"✅ UMAP coords shape: {coords_3d.shape}")
    logger.info(f"📊 UMAP range: min={coords_3d.min(axis=0)}, max={coords_3d.max(axis=0)}")
    
    # Normalize coordinates to RGB values
    coords_normalized = _normalize_coordinates_to_rgb(coords_3d)
    
    # Create spatial RGB image
    rgb_spatial = coords_normalized.reshape(24, 24, 3)
    logger.info(f"🖼️ RGB spatial shape: {rgb_spatial.shape}")
    
    # Sample some RGB values for verification
    sample_pixels = rgb_spatial[:3, :3, :].reshape(-1, 3)
    logger.info(f"🔍 Sample RGB pixels: {sample_pixels[:3].tolist()}")
    
    # Generate final image and encode
    img_str = _create_rgb_image(rgb_spatial)
    
    # Calculate processing time
    processing_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"🌈 UMAP RGB completed in {processing_time:.2f}s")
    
    # Prepare result data
    rgb_values_list = coords_normalized.flatten().tolist()  # Flatten to 1D array for JS
    logger.info(f"📊 Returning {len(rgb_values_list)} RGB values (flattened {rgb_spatial.shape} -> 1D)")
    
    result = {
        'umap_rgb': f"data:image/png;base64,{img_str}",
        'rgb_values': rgb_values_list,  # Raw RGB values for client-side alpha blending
        'coords_3d': coords_3d.tolist(),
        'shape': [24, 24, 3]
    }
    
    # Cache the result
    if not hasattr(cache, '_umap_rgb_cache'):
        cache._umap_rgb_cache = {}
    cache._umap_rgb_cache[cache_key] = result
    logger.info(f"💾 Cached UMAP RGB result for {image_id}")
    
    return result


def _compute_umap_coordinates(features_flat):
    """
    Apply UMAP to high-dimensional features.
    
    Args:
        features_flat: numpy array [576, 1408] of flattened spatial features
        
    Returns:
        numpy array: UMAP coordinates [576, 3]
    """
    # Use OptimizedUMAP for faster performance with caching
    reducer = OptimizedUMAP(
        cache_dir="/tmp/deepearth_umap_cache",
        n_components=3, 
        n_neighbors=15,  # Good for 576 points
        min_dist=0.1, 
        random_state=42,
        n_epochs=30,  # Fewer epochs for faster computation on small dataset
        init='random',  # Faster than spectral
        low_memory=False,  # Faster for small datasets
        metric='cosine',  # Better for high-dimensional data
        n_jobs=1,  # Single thread is often faster for small data
        transform_seed=42  # Consistent results
    )
    return reducer.fit_transform(features_flat)


def _normalize_coordinates_to_rgb(coords_3d):
    """
    Normalize UMAP coordinates to [0,1] RGB values with proper handling.
    
    Args:
        coords_3d: numpy array [576, 3] of UMAP coordinates
        
    Returns:
        numpy array: Normalized RGB values [576, 3] in range [0,1]
    """
    coords_min = coords_3d.min(axis=0)
    coords_max = coords_3d.max(axis=0)
    coords_range = coords_max - coords_min
    
    # Check for degenerate cases
    for i, (min_val, max_val, range_val) in enumerate(zip(coords_min, coords_max, coords_range)):
        logger.info(f"🎨 Dimension {i}: min={min_val:.3f}, max={max_val:.3f}, range={range_val:.3f}")
        if range_val < 1e-6:
            logger.warning(f"⚠️ Dimension {i} has very small range, may cause normalization issues")
    
    # Avoid division by zero
    coords_range = np.maximum(coords_range, 1e-6)
    coords_normalized = (coords_3d - coords_min) / coords_range
    
    # Ensure values are in [0,1]
    coords_normalized = np.clip(coords_normalized, 0, 1)
    logger.info(f"🎨 Normalized RGB range: min={coords_normalized.min(axis=0)}, max={coords_normalized.max(axis=0)}")
    
    # Check for all-black or all-white issues
    if coords_normalized.max() < 0.1:
        logger.warning("⚠️ UMAP RGB values are very dark, may appear black")
    elif coords_normalized.min() > 0.9:
        logger.warning("⚠️ UMAP RGB values are very bright, may appear white")
    
    return coords_normalized


def _create_rgb_image(rgb_spatial):
    """
    Create and encode RGB image from spatial RGB values.
    
    Args:
        rgb_spatial: numpy array [24, 24, 3] of RGB values in range [0,1]
        
    Returns:
        str: Base64 encoded image string
    """
    # Convert to uint8 and create PIL image
    rgb_uint8 = (rgb_spatial * 255).astype(np.uint8)
    logger.info(f"🎨 RGB uint8 range: min={rgb_uint8.min()}, max={rgb_uint8.max()}")
    
    img = Image.fromarray(rgb_uint8)
    img = img.resize((384, 384), Image.NEAREST)  # Match overlay size
    logger.info(f"🖼️ Resized image to: {img.size}")
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()