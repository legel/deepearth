"""
Color processing services for DeepEarth Dashboard.

Provides color conversion and species cluster color management for
consistent visualization across map and 3D displays.
"""

import logging

logger = logging.getLogger(__name__)


def process_species_cluster_colors(cache):
    """
    Get colors for all species based on their HDBSCAN cluster assignments.
    
    This function provides consistent colors across the map and 3D visualization
    by using the cluster colors from the precomputed HDBSCAN results.
    
    Args:
        cache: UnifiedDataCache instance for data access
        
    Returns:
        dict: Taxon colors and metadata
        
    Raises:
        RuntimeError: If UMAP/cluster data is not available
    """
    logger.info("Processing species cluster colors")
    
    # Get or compute UMAP and cluster data
    umap_data, clusters = _get_or_compute_umap_clusters(cache)
    
    if not umap_data or not clusters:
        raise RuntimeError('UMAP/cluster data not available')
    
    # Process colors for all species
    taxon_colors = _convert_cluster_colors_to_formats(umap_data)
    
    result = {
        'taxon_colors': taxon_colors,
        'total_species': len(taxon_colors)
    }
    
    logger.info(f"Processed colors for {len(taxon_colors)} species")
    return result


def _get_or_compute_umap_clusters(cache):
    """Get precomputed UMAP and clusters, computing if necessary."""
    # Use precomputed language UMAP and clusters if available
    if cache.precomputed_language_umap and cache.language_clusters:
        logger.info("Using existing precomputed UMAP and clusters")
        return cache.precomputed_language_umap, cache.language_clusters
    else:
        # Compute if not available
        logger.info("Computing UMAP and clusters as they're not available")
        cache.compute_and_cache_language_umap_clusters()
        return cache.precomputed_language_umap, cache.language_clusters


def _convert_cluster_colors_to_formats(umap_data):
    """
    Convert cluster colors from various formats to standardized RGB and hex formats.
    
    Args:
        umap_data: List of UMAP data points with color information
        
    Returns:
        dict: Taxon colors in multiple formats
    """
    taxon_colors = {}
    
    for pt in umap_data:
        taxon_id = pt['taxon_id']
        cluster = pt.get('cluster', -1)
        color = pt.get('color', '#666666')  # Default gray for unclustered
        
        # Convert color to hex and RGB formats
        if color.startswith('rgb('):
            # Parse rgb(r, g, b) format
            r, g, b = _parse_rgb_string(color)
            hex_color = '#{:02x}{:02x}{:02x}'.format(r, g, b)
        else:
            # Assume it's already hex
            hex_color = color
            # Convert hex to RGB
            r, g, b = _parse_hex_color(hex_color)
        
        taxon_colors[taxon_id] = {
            'r': r,
            'g': g,
            'b': b,
            'hex': hex_color,
            'cluster': cluster
        }
    
    return taxon_colors


def _parse_rgb_string(rgb_string):
    """
    Parse RGB string in format 'rgb(r, g, b)' to individual components.
    
    Args:
        rgb_string: String in format 'rgb(255, 128, 64)'
        
    Returns:
        tuple: (r, g, b) as integers
    """
    rgb_vals = rgb_string[4:-1].split(', ')
    return int(rgb_vals[0]), int(rgb_vals[1]), int(rgb_vals[2])


def _parse_hex_color(hex_color):
    """
    Parse hex color string to RGB components.
    
    Args:
        hex_color: Hex color string like '#ff8040'
        
    Returns:
        tuple: (r, g, b) as integers
    """
    hex_clean = hex_color.lstrip('#')
    return (
        int(hex_clean[0:2], 16),
        int(hex_clean[2:4], 16), 
        int(hex_clean[4:6], 16)
    )