#!/usr/bin/env python3
# =============================================================================
#  utils.py
# -----------------------------------------------------------------------------
#  Geospatial Utilities – Core Helper Functions
# -----------------------------------------------------------------------------
#  Essential utility functions for geospatial coordinate handling:
#
#  Core Functions
#  ------------
#  • _as_fp64 – High-precision tensor conversion
#  • _safe_div – Numerically stable division
#  • wrap_lat – Latitude normalization
#  • wrap_lon_error – Longitude error calculation
#  • wrap_lat_error – Latitude error calculation
#  • _human_unit – Human-readable unit formatting
#
#  Features
#  --------
#  • Numerical stability in edge cases
#  • Polar region handling
#  • High-precision calculations
#  • Error metric computation
#  • Human-readable output formatting
#
#  Quick‑start
#  -----------
#  >>> from encoders.geo.utils import wrap_lat, wrap_lon_error
#  >>> lat = wrap_lat(91.0)  # Normalizes to 89.0
#  >>> error = wrap_lon_error(lon1, lon2, lat)  # Computes scaled error
#
#  These utilities are used internally by the GeospatialConverter.
#
#  MIT License – © 2025 DeepEarth Contributors
# =============================================================================

import torch

# --------------------------------------------------------------------------- #
#  Helper functions                                                           #
# --------------------------------------------------------------------------- #
def _as_fp64(t: torch.Tensor) -> torch.Tensor:
    """Convert tensor to float64 without unnecessary copy.
    
    Args:
        t: Input tensor of any dtype
        
    Returns:
        Tensor converted to float64, reusing memory if possible
    """
    return t.to(torch.float64)


def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    """Element-wise division safe for normalization.
    Maps division by zero (0/0 or x/0 where span=0) to 0.5.
    
    Args:
        num: Numerator tensor (e.g., value - min)
        den: Denominator tensor (e.g., max - min, i.e., span)
        
    Returns:
        Division result, with division by zero mapped to 0.5.
    """
    # Create a mask for denominators that are close to zero
    zero_den_mask = torch.abs(den) < 1e-9 # Use a small threshold for safety
    
    # Replace near-zero denominators with 1 to avoid NaN/Inf during division
    safe_den = torch.where(zero_den_mask, torch.ones_like(den), den)
    
    # Perform the division
    result = num / safe_den
    
    # Where the original denominator was near-zero, set the result to 0.5
    result = torch.where(zero_den_mask, torch.full_like(result, 0.5), result)
    
    return result


def wrap_lat(lat: float) -> float:
    """Normalize latitude to [-90, 90] range."""
    # First wrap to [-180, 180]
    lat = (lat + 180) % 360 - 180
    # Then reflect over poles
    if lat > 90:
        lat = 180 - lat
    elif lat < -90:
        lat = -180 - lat
    return lat


def wrap_lon_error(lon1: torch.Tensor, lon2: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
    """Calculate longitude error accounting for wrapping and latitude scaling."""
    # Near poles or when points are antipodal, longitude differences are less meaningful
    cos_lat = torch.cos(torch.deg2rad(lat))
    # Ignore longitude differences when too close to poles
    near_pole = cos_lat.abs() < 1e-7
    # Handle -180° ≡ 180° wrapping
    basic_err = (lon2 - lon1).abs()
    wrap_err = 360.0 - basic_err
    min_err = torch.minimum(basic_err, wrap_err)
    # Zero out errors near poles
    return torch.where(near_pole, torch.zeros_like(min_err), min_err * cos_lat)


def wrap_lat_error(lat1: torch.Tensor, lat2: torch.Tensor) -> torch.Tensor:
    """Calculate latitude error accounting for polar equivalence.
    
    Args:
        lat1: First set of latitudes in degrees
        lat2: Second set of latitudes in degrees
        
    Returns:
        Tensor of latitude errors in degrees, accounting for polar equivalence
    """
    # First normalize both latitudes to [-90, 90]
    lat1_norm = torch.tensor([wrap_lat(l.item()) for l in lat1], device=lat1.device)
    lat2_norm = torch.tensor([wrap_lat(l.item()) for l in lat2], device=lat2.device)
    
    # At poles, longitude differences don't matter
    is_pole1 = (lat1_norm.abs() - 90.0).abs() < 1e-7
    is_pole2 = (lat2_norm.abs() - 90.0).abs() < 1e-7
    
    # Calculate basic error
    basic_err = (lat2_norm - lat1_norm).abs()
    
    # Zero out errors when both points are at poles
    return torch.where(is_pole1 & is_pole2, torch.zeros_like(basic_err), basic_err)


def _human_unit(val: float, unit: str) -> str:
    """Format value with appropriate SI prefix for human readability.
    
    Args:
        val: Value to format
        unit: Base unit (e.g., 'm', 'deg')
        
    Returns:
        Formatted string with appropriate prefix (pico, nano, micro, milli)
    """
    a = abs(val)
    suffix = " " + unit
    if a < 1e-12:
        return f"{val*1e12:10.3f} p{suffix}"
    if a < 1e-9:
        return f"{val*1e9:10.3f} n{suffix}"
    if a < 1e-6:
        return f"{val*1e6:10.3f} µ{suffix}"
    if a < 1e-3:
        return f"{val*1e3:10.3f} m{suffix}"
    return f"{val:13.3f}{suffix}"