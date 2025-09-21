#!/usr/bin/env python3
# =============================================================================
#  gaussian_splat.py
# -----------------------------------------------------------------------------
#  Gaussian Splatting Reconstruction using GeoFusion Data
# -----------------------------------------------------------------------------
#  This module handles the reconstruction of 3D scenes using Gaussian Splatting,
#  leveraging posed images derived from GeoFusion data loaded via GeoFusionDataset.
#
#  Workflow:
#  1. Initialize GeoFusionDataset to load images, poses, intrinsics.
#  2. Generate initial point cloud from dataset depth maps.
#  3. Initialize Gaussian model parameters (means, colors, opacities, etc.)
#     using the initial point cloud.
#  4. Set up the gsplat renderer and optimizer.
#  5. Run the optimization/training loop:
#     - Render images using gsplat.
#     - Calculate loss against ground truth images.
#     - Backpropagate and update Gaussian parameters.
#     - Perform densification and pruning.
#
#  MIT License – © 2025 DeepEarth Contributors
# =============================================================================

import os
import torch
import torch.nn.functional as F
import pathlib
import numpy as np
import tqdm
from typing import Dict, Optional

# Assuming gsplat library is installed
import gsplat

from encoders.geo.reconstruction.geofusion_dataset import GeoFusionDataset
from encoders.geo.reconstruction.point_cloud_utils import load_tiff, unproject_depth


# Define project and data paths
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent.parent
DATASET_DIR = BASE_DIR / "src" / "datasets" / "geofusion" / "landscape_architecture_studio"

# --- Configuration ---
INITIAL_POINT_CLOUD_VIEWS = 20 # Use more views for better initial cloud
INITIAL_CONFIDENCE_THRESHOLD = 1.0 # Slightly lower threshold for more points
TRAINING_ITERATIONS = 30000
LOG_INTERVAL = 100
CHECKPOINT_INTERVAL = 5000
OUTPUT_DIR = BASE_DIR / "output" / "gsplat_geofusion"

# --- Point Cloud Generation Helper ---
def create_initial_point_cloud(
    dataset: GeoFusionDataset,
    num_views: int,
    confidence_threshold: float,
    device: torch.device,
    dtype: torch.dtype
) -> Optional[torch.Tensor]:
    """Generates an initial point cloud by unprojecting depth from multiple views."""
    all_points_world = []
    num_views_to_process = min(num_views, len(dataset))
    if num_views_to_process == 0:
        print("Error: No views available in the dataset.")
        return None
        
    print(f"Generating initial point cloud from first {num_views_to_process} views...")

    for i in tqdm.tqdm(range(num_views_to_process), desc="Unprojecting Depth"):
        try:
            data_item = dataset[i] 
        except Exception as e:
            print(f"Warning: Error loading data for view {i}: {e}")
            continue

        depth_map = data_item.get("depth_map")
        conf_map = data_item.get("confidence_map")
        intrinsics_depth = data_item.get("intrinsics_depth")
        # Need Camera-to-World pose for unprojection
        c2w_rot = data_item.get("c2w_body_rotation") # This is now C2W_camera
        relative_center = data_item.get("camera_center_relative")

        if (depth_map is None or conf_map is None or intrinsics_depth is None or
            c2w_rot is None or relative_center is None):
            print(f"Warning: Missing required data for view {i} to unproject depth.")
            continue

        points_world = unproject_depth(
            depth_map=depth_map,
            intrinsics=intrinsics_depth,
            cam_to_world_rot=c2w_rot,
            cam_center_world=relative_center,
            confidence_map=conf_map,
            confidence_threshold=confidence_threshold,
            return_cam_coords=False # Only need world points
        )

        if points_world is not None and points_world.shape[0] > 0:
            all_points_world.append(points_world)

    if not all_points_world:
        print("Error: Failed to generate any points for initialization.")
        return None

    combined_points = torch.cat(all_points_world, dim=0)
    print(f"Generated initial point cloud with {combined_points.shape[0]} points.")
    return combined_points.to(device=device, dtype=dtype)


# --- Gaussian Model Initialization ---
def initialize_gaussians(
    initial_points: torch.Tensor, 
    device: torch.device,
    dtype: torch.dtype
) -> Dict[str, torch.Tensor]:
    """Initializes Gaussian parameters based on an initial point cloud."""
    print("Initializing Gaussian model from point cloud...")
    means = initial_points.clone().detach()
    num_points = means.shape[0]
    if num_points == 0:
        raise ValueError("Initial point cloud is empty.")
    print(f"  Initializing {num_points} Gaussians.")
    dist2 = torch.clamp_min(gsplat.distCUDA2(means.float()), 1e-7)
    scales = torch.sqrt(dist2).unsqueeze(-1).repeat(1, 3)
    log_scales = torch.log(scales).detach()
    quats = torch.zeros((num_points, 4), device=device, dtype=dtype)
    quats[:, 0] = 1.0
    colors = torch.full((num_points, 3), 0.5, device=device, dtype=dtype)
    opacities = torch.logit(torch.full((num_points, 1), 0.1, device=device)).to(dtype)
    means.requires_grad_(True)
    log_scales.requires_grad_(True)
    quats.requires_grad_(True)
    colors.requires_grad_(True)
    opacities.requires_grad_(True)
    print("  Gaussian initialization complete.")
    return {
        "means": means,
        "log_scales": log_scales,
        "quats": quats,
        "colors": colors,
        "opacities": opacities,
    }

# --- Main Training Script ---
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.float32

    print(f"Using device: {DEVICE}")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Dataset Directory: {DATASET_DIR}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # --- 1. Load Dataset ---
    try:
        # Load depth maps for initialization
        dataset = GeoFusionDataset(DATASET_DIR, device=DEVICE, dtype=DTYPE, verbose=False, load_depth=True)
    except Exception as e:
        print(f"Failed to initialize dataset: {e}")
        import traceback
        traceback.print_exc()
        exit()

    # --- 2. Create Initial Point Cloud ---
    initial_points = create_initial_point_cloud(
        dataset=dataset,
        num_views=INITIAL_POINT_CLOUD_VIEWS,
        confidence_threshold=INITIAL_CONFIDENCE_THRESHOLD,
        device=DEVICE,
        dtype=DTYPE
    )
    if initial_points is None:
        exit("Could not create initial point cloud.")
        
    # Free up dataset depth map memory if not needed further
    if not dataset.load_depth:
         del dataset.depth_map
         del dataset.confidence_map
         torch.cuda.empty_cache()

    # --- 3. Initialize Gaussian Model ---
    try:
        gaussians = initialize_gaussians(initial_points, DEVICE, DTYPE)
    except ValueError as e:
        print(f"Error initializing Gaussians: {e}")
        exit()

    # --- 4. Setup Optimizer ---
    params = [
        {'params': [gaussians['means']], 'lr': 1.6e-4, "name": "means"},
        {'params': [gaussians['colors']], 'lr': 2.5e-3, "name": "colors"},
        {'params': [gaussians['opacities']], 'lr': 5e-2, "name": "opacities"},
        {'params': [gaussians['log_scales']], 'lr': 5e-3, "name": "scales"},
        {'params': [gaussians['quats']], 'lr': 1e-3, "name": "quats"}
    ]
    optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
    # TODO: Add LR Scheduler

    # --- 5. Training Loop ---
    print(f"\nStarting training for {TRAINING_ITERATIONS} iterations...")

    # Basic Training Loop Structure (Placeholders)
    for step in tqdm.tqdm(range(TRAINING_ITERATIONS), desc="Training"):
        optimizer.zero_grad()

        # --- Select View(s) ---
        view_idx = step % len(dataset)
        try:
             # Load data for current view (without depth this time)
             dataset.load_depth = False # Ensure depth isn't loaded repeatedly
             data_item = dataset[view_idx]
        except Exception as e:
             print(f"Warning: Error loading data for view {view_idx} at step {step}: {e}")
             continue # Skip iteration

        # --- Prepare Inputs for gsplat ---
        gt_image = data_item.get("image")
        if gt_image is None:
            print(f"Warning: Missing ground truth image for view {view_idx}. Skipping step {step}.")
            continue

        W, H = data_item["image_width"].item(), data_item["image_height"].item()
        # Get W2C pose components from dataset
        w2c_rot = data_item["w2c_rotation"] 
        w2c_trans = data_item["w2c_translation"]
        
        # Construct 4x4 view matrix (World-to-Camera)
        viewmat = torch.eye(4, device=DEVICE, dtype=DTYPE)
        viewmat[:3, :3] = w2c_rot
        viewmat[:3, 3] = w2c_trans
        
        K = data_item["intrinsics"] # Use RGB intrinsics for rendering

        # Get current Gaussian parameters
        means = gaussians["means"]
        log_scales = gaussians["log_scales"]
        scales = torch.exp(log_scales)
        quats = F.normalize(gaussians["quats"], dim=-1)
        colors = torch.sigmoid(gaussians["colors"])
        opacities = torch.sigmoid(gaussians["opacities"])

        # --- Render Image ---
        try:
            render_rgb, render_alpha = gsplat.rasterize_gaussians(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=viewmat.unsqueeze(0),
                Ks=K.unsqueeze(0),
                width=W,
                height=H,
            )
            rendered_image = render_rgb[0]
        except Exception as e:
            print(f"\nError during gsplat rasterization at step {step}: {e}")
            continue

        # --- Calculate Loss ---
        loss_l1 = F.l1_loss(rendered_image, gt_image)
        # TODO: Add D-SSIM loss
        loss = loss_l1

        # --- Backpropagate and Optimize ---
        loss.backward()
        optimizer.step()

        # --- Densification & Pruning ---
        # TODO: Implement properly (critical step!)

        # --- Logging / Visualization ---
        if step % LOG_INTERVAL == 0:
            print(f"\nStep {step}/{TRAINING_ITERATIONS}, Loss: {loss.item():.4f}")
            # TODO: Log metrics (PSNR, SSIM)
            # TODO: Visualize rendered image vs GT
            
        # --- Checkpointing ---
        # if step > 0 and step % CHECKPOINT_INTERVAL == 0:
            # print(f"\nSaving checkpoint at step {step}...")
            # checkpoint_path = OUTPUT_DIR / f"checkpoint_{step:06d}.pth"
            # TODO: Implement save_checkpoint function
            # save_checkpoint(gaussians, optimizer, step, checkpoint_path)

    print("\nTraining finished.")
    # TODO: Save final Gaussian model


# --- Helper functions for Densification/Pruning (Placeholders) ---
# TODO: Implement based on gsplat examples (e.g., simple_trainer.py)
# Requires access to gradients and careful handling of optimizer state

