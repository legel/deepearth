"""
Image processing services for DeepEarth Dashboard.

Provides image proxy functionality with automatic size transformation
for iNaturalist images and remote URL handling.
"""

import logging
import re

logger = logging.getLogger(__name__)


def proxy_image_request(cache, gbif_id, image_num, requested_size='large'):
    """
    Proxy for serving images from remote URLs with automatic size transformation.
    
    Automatically transforms iNaturalist images to use specified size
    instead of 'original' for faster loading.
    
    Args:
        cache: UnifiedDataCache instance for data access
        gbif_id: GBIF identifier for the observation
        image_num: Image number (1-based)
        requested_size: Image size preference (original, large, medium, small)
        
    Returns:
        str: Transformed image URL for redirect
        
    Raises:
        FileNotFoundError: If observation or image not found
    """
    logger.info(f"Proxying image request: GBIF {gbif_id}, image {image_num}, size {requested_size}")
    
    # Get observation
    obs_data = cache.loader.get_observation(gbif_id)
    if obs_data is None:
        raise FileNotFoundError('Observation not found')
    
    # Get image URL from dataset
    image_urls = obs_data.get('image_urls', [])
    if not isinstance(image_urls, list) or len(image_urls) < image_num:
        raise FileNotFoundError('Image not found')
    
    url = image_urls[image_num - 1]
    
    # Transform iNaturalist URLs to requested size
    if 'inaturalist' in url and '/photos/' in url:
        url = _transform_inaturalist_url(url, requested_size)
    
    logger.info(f"Serving image URL: {url}")
    return url


def _transform_inaturalist_url(url, requested_size):
    """
    Transform iNaturalist URLs to use the requested image size.
    
    Args:
        url: Original iNaturalist image URL
        requested_size: Desired image size (original, large, medium, small)
        
    Returns:
        str: Transformed URL with the requested size
    """
    # Extract current size from URL
    size_pattern = r'/([^/]+)\.(jpg|jpeg|png)$'
    match = re.search(size_pattern, url)
    
    if match:
        current_size = match.group(1)
        extension = match.group(2)
        
        # Only transform if not already the requested size
        if current_size != requested_size:
            url = url.replace(f'/{current_size}.{extension}', f'/{requested_size}.{extension}')
            logger.info(f"Transformed iNaturalist image from {current_size} to {requested_size}")
    
    return url