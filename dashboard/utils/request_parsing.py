"""
Request parsing utilities for DeepEarth Dashboard API endpoints.

Provides common functions for parsing and validating request parameters
across multiple Flask routes.
"""

from flask import request
import re


def extract_gbif_id_from_image_id(image_id):
    """
    Extract GBIF ID and image number from standardized image_id format.
    
    Args:
        image_id: String in format "gbif_XXXXXXX_taxon_XXXXXXX_img_N"
        
    Returns:
        tuple: (gbif_id: int, image_num: int)
        
    Raises:
        ValueError: If image_id format is invalid
    """
    match = re.match(r'gbif_(\d+)_taxon_\d+_img_(\d+)', image_id)
    if not match:
        raise ValueError(f"Invalid image_id format: {image_id}")
    return int(match.group(1)), int(match.group(2))


def parse_geographic_bounds():
    """
    Parse geographic bounds from Flask request arguments with sensible defaults.
    
    Returns:
        dict: Geographic bounds with keys 'north', 'south', 'east', 'west'
    """
    return {
        'north': float(request.args.get('north', 90)),
        'south': float(request.args.get('south', -90)),
        'east': float(request.args.get('east', 180)),
        'west': float(request.args.get('west', -180))
    }


def parse_required_geographic_bounds():
    """
    Parse required geographic bounds from Flask request arguments.
    
    Returns:
        dict: Geographic bounds with keys 'north', 'south', 'east', 'west'
        
    Raises:
        ValueError: If any required bound is missing
    """
    try:
        return {
            'north': float(request.args.get('north')),
            'south': float(request.args.get('south')),
            'east': float(request.args.get('east')),
            'west': float(request.args.get('west'))
        }
    except (TypeError, ValueError) as e:
        raise ValueError("Missing required geographic bounds: north, south, east, west")


def parse_temporal_filters():
    """
    Parse temporal filters from Flask request arguments.
    
    Returns:
        dict or None: Temporal filter parameters if year_min is provided, None otherwise
    """
    year_min = request.args.get('year_min', type=int)
    if year_min is None:
        return None
    return {
        'year_min': year_min,
        'year_max': request.args.get('year_max', type=int, default=2025),
        'month_min': request.args.get('month_min', type=int, default=1),
        'month_max': request.args.get('month_max', type=int, default=12),
        'hour_min': request.args.get('hour_min', type=int, default=0),
        'hour_max': request.args.get('hour_max', type=int, default=23)
    }


def parse_grid_statistics_parameters():
    """
    Parse parameters for grid statistics requests.
    
    Returns:
        tuple: (bounds, grid_size, temporal_filters)
    """
    bounds = parse_required_geographic_bounds()
    grid_size = float(request.args.get('grid_size', 0.01))
    temporal_filters = parse_temporal_filters()
    return bounds, grid_size, temporal_filters


def parse_ecosystem_analysis_parameters():
    """
    Parse parameters for ecosystem analysis requests.
    
    Returns:
        tuple: (bounds, analysis_type)
    """
    bounds = parse_required_geographic_bounds()
    analysis_type = request.args.get('type', 'language')
    return bounds, analysis_type


def parse_attention_map_parameters():
    """
    Parse parameters for attention map requests.
    
    Returns:
        tuple: (temporal_mode, visualization, colormap, alpha)
    """
    temporal_mode = request.args.get('temporal', 'mean')
    visualization = request.args.get('visualization', 'l2norm')
    colormap = request.args.get('colormap', 'plasma')
    alpha = float(request.args.get('alpha', 0.7))
    return temporal_mode, visualization, colormap, alpha


def parse_vision_features_parameters():
    """
    Parse parameters for vision features requests.
    
    Returns:
        tuple: (temporal_mode, visualization)
    """
    temporal_mode = request.args.get('temporal', 'mean')
    visualization = request.args.get('visualization', 'l2norm')
    return temporal_mode, visualization