"""
Feature analysis services for DeepEarth Dashboard.

Provides statistical analysis and dimensionality reduction for vision features,
including PCA computation and feature statistics calculation.
"""

import torch
import numpy as np
from datetime import datetime
import logging
import time

from utils.request_parsing import extract_gbif_id_from_image_id

logger = logging.getLogger(__name__)


def compute_pca_raw(image_id, cache):
    """
    Fast PCA computation optimized for instant response times.
    
    Args:
        image_id: Image identifier in format "gbif_XXXXXXX_taxon_XXXXXXX_img_N"
        cache: UnifiedDataCache instance for data access
        
    Returns:
        dict: PCA values, statistics, and timing information
        
    Raises:
        ValueError: If image_id format is invalid
        FileNotFoundError: If observation or vision features not found
    """
    start_time = datetime.now()
    
    # Extract GBIF ID from image_id
    try:
        gbif_id, img_num = extract_gbif_id_from_image_id(image_id)
    except ValueError as e:
        raise ValueError(str(e))
    
    # Get observation data
    obs_data = cache.loader.get_observation(gbif_id)
    if obs_data is None:
        raise FileNotFoundError('Observation not found')
    
    # Load vision embedding
    features = cache.get_vision_embedding(gbif_id, obs_data['taxon_id'], 1)
    if features is None:
        raise FileNotFoundError('Vision features not found')
    
    # Reshape and compute mean across temporal dimension
    features = features.view(8, 576, 1408).mean(dim=0)  # [576, 1408]
    
    # Convert to numpy for sklearn
    features_numpy = features.detach().cpu().numpy()
    
    # Compute PCA with timing
    pca_result, pca_stats, pca_time = _compute_pca_with_timing(features_numpy)
    
    # Reshape to 24x24 grid
    pca_grid = pca_result.reshape(24, 24)
    
    total_time = (datetime.now() - start_time).total_seconds()
    
    return {
        'pca_values': pca_grid.tolist(),
        'stats': pca_stats,
        'timing': {
            'pca_computation': pca_time,
            'total': total_time
        },
        'shape': [24, 24]
    }


def compute_feature_statistics(image_id, cache):
    """
    Get detailed statistics for image features.
    
    Args:
        image_id: Image identifier in format "gbif_XXXXXXX_taxon_XXXXXXX_img_N"
        cache: UnifiedDataCache instance for data access
        
    Returns:
        dict: Feature statistics including diversity, stability, magnitude, and entropy
        
    Raises:
        ValueError: If image_id format is invalid
        FileNotFoundError: If observation or vision features not found
    """
    # Extract GBIF ID from image_id
    try:
        gbif_id, img_num = extract_gbif_id_from_image_id(image_id)
    except ValueError as e:
        raise ValueError(str(e))
    
    # Get observation data first
    obs_data = cache.loader.get_observation(gbif_id)
    if obs_data is None:
        raise FileNotFoundError('Observation not found')
    
    # Load vision embedding
    features = cache.get_vision_embedding(gbif_id, obs_data['taxon_id'], 1)
    if features is None:
        raise FileNotFoundError('Vision features not found')
    
    # Reshape to temporal and spatial using PyTorch operations
    features = features.view(8, 576, 1408)
    
    # Compute various feature statistics
    stats = _compute_comprehensive_statistics(features)
    
    return {
        'spatial_diversity': stats['spatial_diversity'],
        'temporal_stability': stats['temporal_stability'],
        'feature_magnitude': stats['feature_magnitude'],
        'information_density': stats['information_density'],
        'total_features': int(features.numel()),
        'shape': list(features.shape)
    }


def _compute_pca_with_timing(features_numpy):
    """
    Compute PCA with timing information.
    
    Args:
        features_numpy: numpy array [576, 1408] of features
        
    Returns:
        tuple: (pca_values, stats_dict, computation_time)
    """
    from sklearn.decomposition import PCA
    
    pca_start = time.time()
    pca = PCA(n_components=1, svd_solver='randomized')
    pca_result = pca.fit_transform(features_numpy)  # [576, 1]
    pca_time = time.time() - pca_start
    
    # Get the first component values
    pca_values = pca_result[:, 0]
    
    # Normalize to [0, 1]
    pca_min = pca_values.min()
    pca_max = pca_values.max()
    pca_normalized = (pca_values - pca_min) / (pca_max - pca_min + 1e-8)
    
    stats = {
        'min': float(pca_min),
        'max': float(pca_max),
        'mean': float(pca_values.mean()),
        'std': float(pca_values.std()),
        'explained_variance': float(pca.explained_variance_ratio_[0])
    }
    
    return pca_normalized, stats, pca_time


def _compute_comprehensive_statistics(features):
    """
    Compute comprehensive statistics for vision features.
    
    Args:
        features: torch.Tensor [8, 576, 1408] of vision features
        
    Returns:
        dict: Comprehensive feature statistics
    """
    # Compute spatial diversity (variance across spatial locations)
    spatial_diversity = float(features.var(dim=(0, 2)).mean().item())
    
    # Compute temporal stability (1 - variance across time)
    temporal_variance = float(features.var(dim=0).mean().item())
    temporal_stability = max(0, 1 - temporal_variance)
    
    # Overall feature magnitude
    feature_magnitude = float(features.norm(dim=-1).mean().item())
    
    # Information density (entropy approximation)
    features_abs = torch.abs(features) + 1e-10
    features_normalized = features_abs / features_abs.sum(dim=-1, keepdim=True)
    entropy = -(features_normalized * torch.log(features_normalized)).sum(dim=-1).mean()
    information_density = float(entropy.item())
    
    return {
        'spatial_diversity': spatial_diversity,
        'temporal_stability': temporal_stability,
        'feature_magnitude': feature_magnitude,
        'information_density': information_density
    }