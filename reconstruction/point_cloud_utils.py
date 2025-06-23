#!/usr/bin/env python3
# =============================================================================
#  point_cloud_utils.py
# -----------------------------------------------------------------------------
#  Utilities for generating point clouds from depth maps
# -----------------------------------------------------------------------------
#  This module provides functions to load ARKit depth/confidence maps (TIFF)
#  and unproject them into 3D points using camera intrinsics and poses.
#
#  MIT License – © 2025 DeepEarth Contributors
# =============================================================================

import torch
import pathlib
import numpy as np
from typing import Optional, Tuple
import torch.nn.functional as F

try:
    import tifffile
except ImportError:
    print("Error: 'tifffile' package not found. Please install it: pip install tifffile")
    tifffile = None

def load_tiff(path: pathlib.Path) -> Optional[np.ndarray]:
    """Loads a TIFF file, typically a single-channel float32 depth/confidence map."""
    if tifffile is None:
        raise ImportError("'tifffile' package is required to load TIFF files.")
    if not path.exists():
        print(f"Warning: TIFF file not found at {path}")
        return None
    try:
        data = tifffile.imread(path)
        return data.astype(np.float32) # Ensure float32
    except Exception as e:
        print(f"Error loading TIFF file {path}: {e}")
        return None

def project_points_to_image(
    points_world: torch.Tensor,       # (N, 3)
    cam_to_world_rot: torch.Tensor, # (3, 3) R_cw
    cam_center_world: torch.Tensor, # (3,) C_world
    intrinsics: torch.Tensor        # (3, 3)
) -> torch.Tensor:
    """Projects 3D world points into 2D pixel coordinates (u, v).

    Args:
        points_world: (N, 3) tensor of world coordinates.
        cam_to_world_rot: Camera-to-world rotation R_cw (3, 3).
        cam_center_world: Camera center position C_world (3,).
        intrinsics: Camera intrinsics K (3, 3).

    Returns:
        (N, 2) tensor of pixel coordinates (u, v).
    """
    # Derive W2C from C2W
    world_to_camera_rot = cam_to_world_rot.T # R_wc = R_cw^T
    world_to_camera_trans = -world_to_camera_rot @ cam_center_world # t_wc = -R_wc * C_world

    # Transform to Camera Coordinates: P_cam = R_wc @ P_world + t_wc
    points_cam = world_to_camera_rot @ points_world.T + world_to_camera_trans.unsqueeze(1)

    # Project to Image Plane (Homogeneous): p_img_hom = K @ P_cam
    pixels_hom = intrinsics @ points_cam # (3, 3) @ (3, N) -> (3, N)

    # Convert to Pixel Coordinates (u, v)
    depth = pixels_hom[2, :] + 1e-8
    u = pixels_hom[0, :] / depth
    v = pixels_hom[1, :] / depth

    # Stack coordinates
    pixel_coords = torch.stack([u, v], dim=1) # (N, 2)
    return pixel_coords

def sample_colors_from_image(
    pixel_coords: torch.Tensor, # (N, 2)
    image: torch.Tensor        # (H, W, 3), range 0-1
) -> torch.Tensor:
    """Samples colors from an image at given pixel coordinates using bilinear interpolation.

    Args:
        pixel_coords: (N, 2) tensor of (u, v) coordinates.
        image: (H, W, 3) tensor representing the image.

    Returns:
        (N, 3) tensor of sampled RGB colors.
    """
    H, W, _ = image.shape
    N = pixel_coords.shape[0]
    device = image.device
    dtype = image.dtype

    # Normalize pixel coordinates for grid_sample: range [-1, 1]
    u = pixel_coords[:, 0]
    v = pixel_coords[:, 1]
    x_norm = (u / (W - 1)) * 2 - 1 # Normalize by W-1 (pixel centers)
    y_norm = (v / (H - 1)) * 2 - 1 # Normalize by H-1 (pixel centers)

    # grid_sample expects (N, H_out, W_out, 2) or (N, 1, 1, 2) for point sampling
    # grid_sample also expects image in (N, C, H_in, W_in)
    grid = torch.stack([x_norm, y_norm], dim=1).reshape(1, N, 1, 2).to(dtype) # (1, N, 1, 2)
    image_nchw = image.permute(2, 0, 1).unsqueeze(0) # (1, 3, H, W)

    # Sample using bilinear interpolation, padding mode border repeats edge values
    sampled_colors_nchw = F.grid_sample(
        image_nchw,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=False # Typically False for pixel center alignment
    )

    # Reshape to (N, 3)
    sampled_colors = sampled_colors_nchw.squeeze().permute(1, 0) # (N, 3)
    return sampled_colors

def unproject_depth(
    depth_map: torch.Tensor,        # (H, W)
    intrinsics: torch.Tensor,       # (3, 3)
    cam_to_world_rot: torch.Tensor, # (3, 3) R_cw
    cam_center_world: torch.Tensor, # (3,) C_world
    confidence_map: Optional[torch.Tensor] = None, # (H, W)
    confidence_threshold: float = 0.5, # Threshold for filtering points
    return_cam_coords: bool = False # Add flag
) -> Optional[torch.Tensor] | Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Unprojects a depth map to a 3D point cloud in world coordinates.

    Assumes depth map and intrinsics correspond to the same image resolution.
    Uses the pinhole camera model inversion: P_cam = depth * K_inv @ [u, v, 1]^T
    Then transforms to world: P_world = R_cw @ P_cam + C_world

    Args:
        depth_map: Depth map tensor (H, W).
        intrinsics: Camera intrinsics matrix K (3, 3) scaled for depth map resolution.
        cam_to_world_rot: Camera-to-world rotation matrix R_cw (3, 3).
        cam_center_world: Camera center position in world coordinates C_world (3,).
        confidence_map: Optional confidence map (H, W), values typically 0-2.
        confidence_threshold: Minimum confidence value (e.g., 0.5 maps to Medium/High).
        return_cam_coords: If True, also return points in camera coordinates.

    Returns:
        If return_cam_coords is False: Tensor (N, 3) of world points or None.
        If return_cam_coords is True: Tuple (world_points, camera_points) or (None, None).
    """
    device = depth_map.device
    dtype = intrinsics.dtype
    H, W = depth_map.shape

    # Create pixel coordinates
    v_coords, u_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )

    # Create homogenous pixel coordinates (u, v, 1)
    pixels_hom = torch.stack([u_coords, v_coords, torch.ones_like(u_coords)], dim=-1)

    # Flatten tensors for batch processing
    pixels_hom_flat = pixels_hom.reshape(-1, 3)
    depth_flat = depth_map.reshape(-1)

    # Filter based on confidence map if provided
    valid_mask = torch.ones_like(depth_flat, dtype=torch.bool)
    if confidence_map is not None:
        confidence_flat = confidence_map.reshape(-1)
        valid_mask &= (confidence_flat >= confidence_threshold)

    # Filter based on valid depth (e.g., depth > 0)
    valid_mask &= (depth_flat > 1e-6)

    if not valid_mask.any():
        # Return Nones with correct tuple structure if requested
        return (None, None) if return_cam_coords else None 

    pixels_hom_valid = pixels_hom_flat[valid_mask]
    depth_valid = depth_flat[valid_mask]

    # Calculate inverse intrinsics
    try:
        K_inv = torch.inverse(intrinsics)
    except Exception as e:
        print(f"Error inverting intrinsics matrix: {e}")
        return (None, None) if return_cam_coords else None

    # Unproject to Camera Coordinates: P_cam = depth * K_inv @ pixels_hom^T
    unprojected_cam_dir = K_inv @ pixels_hom_valid.T
    points_cam = depth_valid.unsqueeze(0) * unprojected_cam_dir
    points_cam = points_cam.T

    # Transform to World Coordinates: P_world = R_cw @ P_cam + C_world
    points_world = torch.matmul(points_cam, cam_to_world_rot.T) + cam_center_world.unsqueeze(0)

    if return_cam_coords:
        return points_world, points_cam
    else:
        return points_world 