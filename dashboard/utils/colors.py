"""
Color generation utilities for DeepEarth Dashboard.

Provides functions for generating visually distinct colors for data visualization.
"""

import colorsys


def generate_hsv_colors(n):
    """
    Generate n visually distinct colors using HSV color space.
    
    Distributes hues evenly around the color wheel with high saturation
    and medium lightness for vivid, distinguishable colors.
    
    Args:
        n: Number of colors needed
        
    Returns:
        List of RGB color strings in format "rgb(r, g, b)"
    """
    colors = []
    for i in range(n):
        hue = i * 360.0 / n
        # Use high saturation and medium lightness for vivid, distinct colors
        rgb = colorsys.hsv_to_rgb(hue/360.0, 0.9, 0.8)
        colors.append(f"rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})")
    return colors