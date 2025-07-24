"""
Vision attention utilities for DeepEarth Dashboard.

Provides functions for generating attention overlays and visualizations from vision features.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
import logging

logger = logging.getLogger(__name__)


def generate_attention_overlay(attention_map, colormap='plasma', alpha=0.7):
    """
    Generate base64 encoded attention overlay image.
    
    Args:
        attention_map: PyTorch tensor or numpy array [576] or [24, 24]
        colormap: Matplotlib colormap name
        alpha: Transparency for overlay
        
    Returns:
        Base64 encoded PNG image string
    """
    # Convert PyTorch tensor to numpy if needed
    if hasattr(attention_map, 'detach'):  # Check if it's a PyTorch tensor
        attention_map = attention_map.detach().cpu().numpy()
    
    # Reshape from flat [576] to spatial [24, 24] if needed
    if attention_map.ndim == 1 and len(attention_map) == 576:
        attention_map = attention_map.reshape(24, 24)
    
    # Check for degenerate attention maps
    if attention_map.max() == attention_map.min():
        logger.warning(f"⚠️ Attention map is constant (value: {attention_map.max():.3f})")
    
    # Convert to float32 if needed (scipy doesn't support float16)
    if attention_map.dtype == np.float16:
        attention_map = attention_map.astype(np.float32)
    
    # Upsample with PIL for better performance
    # First normalize to [0, 255] for uint8
    attention_norm = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    attention_uint8 = (attention_norm * 255).astype(np.uint8)
    
    # Create PIL image and resize to 384x384 to match image display size
    img_small = Image.fromarray(attention_uint8, mode='L')
    img_large = img_small.resize((384, 384), Image.BILINEAR)
    
    # Convert back to normalized float array
    attention_hr = np.array(img_large).astype(np.float32) / 255.0
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    attention_colored = cmap(attention_hr)
    
    # Set alpha channel
    attention_colored[:, :, 3] = attention_hr * alpha
    
    # Convert to image and encode as base64
    img = Image.fromarray((attention_colored * 255).astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"