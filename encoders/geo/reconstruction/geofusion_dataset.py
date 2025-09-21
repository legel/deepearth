#!/usr/bin/env python3
# =============================================================================
#  geofusion_dataset.py
# -----------------------------------------------------------------------------
#  Dataset class for loading GeoFusion data for Gaussian Splatting
# -----------------------------------------------------------------------------
#  This module provides a PyTorch Dataset for loading images, depth maps,
#  confidence maps, poses, and intrinsics derived from GeoFusion CSV and
#  NDJSON log files.
#
#  MIT License – © 2025 DeepEarth Contributors
# =============================================================================

import os
import json
import torch
import pathlib
from PIL import Image
import tqdm
import torchvision.transforms.functional as TF
from typing import Tuple, Optional, List, Dict, Any
import numpy as np # For checking depth map shape

from torch.utils.data import Dataset

from encoders.geo.geo2xyz import GeospatialConverter
from encoders.geo.geofusion import GeoFusionDataLoader
# Import utility functions
from encoders.geo.utils import _safe_div # Import specifically for debug print
# Import TIFF loading utility
from .point_cloud_utils import load_tiff

class GeoFusionDataset(Dataset):
    """PyTorch Dataset for loading GeoFusion images, depth, confidence, poses, and intrinsics."""

    def __init__(
        self,
        dataset_dir: pathlib.Path,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        verbose: bool = False,
        load_depth: bool = False # Option to load depth maps in __getitem__
    ):
        """Initialize the dataset.

        Args:
            dataset_dir: Path to the root dataset directory
                         (e.g., .../landscape_architecture_studio).
            device: Torch device for tensors.
            dtype: Torch dtype for tensors.
            verbose: Whether to print detailed loading information.
            load_depth: If True, load depth/confidence maps in __getitem__.
                        Otherwise, only paths are returned.
        """
        self.dataset_dir = dataset_dir
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self.load_depth = load_depth

        self.image_dir = self.dataset_dir / "images" / "ground" / "rgb"
        self.depth_dir = self.dataset_dir / "images" / "ground" / "lidar"
        self.geofusion_csv = self.dataset_dir / "logs" / "geofusion.csv"
        self.log_ndjson = self.dataset_dir / "logs" / "log.ndjson"

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.depth_dir.exists():
            print(f"Warning: Depth directory not found: {self.depth_dir}. Depth maps cannot be loaded.")
            self.depth_dir = None
        if not self.geofusion_csv.exists():
            raise FileNotFoundError(f"GeoFusion CSV not found: {self.geofusion_csv}")
        if not self.log_ndjson.exists():
            raise FileNotFoundError(f"Log NDJSON not found: {self.log_ndjson}")

        self.converter = GeospatialConverter(device=self.device, norm_dtype=torch.float64)
        self.depth_dims = None
        self.scaled_intrinsics = None
        self.rgb_intrinsics = None
        self.actual_rgb_dims = None
        self.camera_centers_ecef = None
        self.camera_centers_relative = None
        self.scene_origin_ecef = None
        # Store the Body-to-World rotation from geo2xyz
        self.c2w_body_rotation = None 
        # W2C transforms are no longer precomputed here
        # self.w2c_rotations = None
        # self.w2c_translations = None

        self._load_data()

    def _load_data(self):
        """Loads poses from CSV, intrinsics from NDJSON, matches them, and scales intrinsics."""
        if self.verbose:
            print(f"Loading GeoFusion poses from: {self.geofusion_csv}")

        # --- Load Poses from CSV ---
        loader = GeoFusionDataLoader(self.converter)
        try:
            loader.load_csv(str(self.geofusion_csv))
        except Exception as e:
            raise RuntimeError(f"Error loading GeoFusion CSV: {e}") from e
        geo_coords, orientations = loader.convert_all()
        csv_image_names = [entry.image_name for entry in loader.entries]

        # Convert to ECEF XYZ and camera-to-world rotations (float64)
        # Also get intermediate matrices if verbose
        geo_to_xyz_result = self.converter.geodetic_to_xyz(
            geo_coords, 
            orientations, 
            return_intermediates=self.verbose
        )
        if self.verbose:
            xyz_positions_f64, cam_to_world_rotations_f64, R_ned_body, R_ecef_ned = geo_to_xyz_result
            if R_ned_body is not None:
                 print(f"  Intermediate R_ned_body[0]:\n{R_ned_body[0]}")
            if R_ecef_ned is not None:
                 print(f"  Intermediate R_ecef_ned[0]:\n{R_ecef_ned[0]}")
        else:
            xyz_positions_f64, cam_to_world_rotations_f64 = geo_to_xyz_result

        # --- DEBUG: Print initial ECEF/C2W for first few views --- 
        if self.verbose and xyz_positions_f64 is not None and cam_to_world_rotations_f64 is not None:
            num_debug_views = min(3, xyz_positions_f64.shape[0])
            print(f"--- Debug Initial ECEF Positions (First {num_debug_views}) ---")
            for i in range(num_debug_views):
                print(f"  View {i}: {xyz_positions_f64[i].cpu().numpy()}")
            print(f"--- Debug Initial C2W Rotations (First {num_debug_views}) ---")
            for i in range(num_debug_views):
                print(f"  View {i}:\n{cam_to_world_rotations_f64[i].cpu().numpy()}")
            print("--- End Initial Pose Debug ---")
        # --- End DEBUG ---

        if self.verbose:
             print(f"  Loaded {len(csv_image_names)} poses from CSV.")

        # --- Load Intrinsics from NDJSON & Determine Log Resolution ---
        if self.verbose: print(f"Loading intrinsics and log resolution from: {self.log_ndjson}")
        intrinsics_map_log: Dict[str, torch.Tensor] = {}
        log_image_dims = None # (W_log, H_log)
        try:
            with open(self.log_ndjson, 'r') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line)
                        # Get resolution reported in logs (likely from triggerSettings)
                        if log_entry.get("triggerSettings") and not log_image_dims:
                            res_str = log_entry["triggerSettings"].get("imageResolution", "0.0x0.0")
                            W_str, H_str = res_str.split('x')
                            log_image_dims = (int(float(W_str)), int(float(H_str)))
                            if self.verbose: print(f"  Found Logged image resolution: {log_image_dims}")

                        # Extract intrinsics corresponding to the *logged* resolution
                        if "photoTaken" in log_entry:
                            photo_data = log_entry["photoTaken"]
                            img_name_base = photo_data.get("imageName") # Name without extension
                            intrinsic_list = photo_data.get("intrinsicMatrix")
                            if img_name_base and intrinsic_list and len(intrinsic_list) == 9:
                                fx, _, cx, _, fy, cy, _, _, _ = intrinsic_list
                                # Store K relative to logged resolution initially
                                K_log = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=self.device, dtype=self.dtype)
                                intrinsics_map_log[img_name_base] = K_log
                    except json.JSONDecodeError:
                        if self.verbose: print(f"Warning: Skipping invalid JSON line in {self.log_ndjson}")
                        continue
        except Exception as e:
             raise RuntimeError(f"Error reading or parsing {self.log_ndjson}: {e}") from e

        if not intrinsics_map_log:
            raise RuntimeError("No intrinsic matrices found in log NDJSON file.")
        if not log_image_dims:
             print("Warning: Could not determine logged image dimensions from NDJSON. Assuming intrinsics match actual image size.")
             # Fallback: proceed without scaling from log->actual later

        if self.verbose: print(f"  Loaded intrinsics for {len(intrinsics_map_log)} images from NDJSON.")

        # --- Match Poses and Intrinsics --- Filter data ---
        matched_indices = []
        final_image_names = []
        final_intrinsics_log = [] # Store the K corresponding to log resolution

        # Determine actual RGB / Depth dimensions from files
        determined_actual_rgb_dims = False
        determined_depth_dims = False

        for i, img_name_with_ext in enumerate(csv_image_names):
            img_name_base = img_name_with_ext.split('.')[0]

            if img_name_base in intrinsics_map_log:
                # Check actual RGB dimension only once if needed
                if not determined_actual_rgb_dims:
                    try:
                        first_img_path = self.image_dir / img_name_with_ext
                        with Image.open(first_img_path) as img:
                            self.actual_rgb_dims = img.size # (W_rgb, H_rgb)
                        if self.verbose: print(f"  Determined Actual RGB dimensions: {self.actual_rgb_dims} from {img_name_with_ext}")
                        determined_actual_rgb_dims = True
                    except Exception as e:
                        raise RuntimeError(f"Could not load first image {img_name_with_ext} to determine actual RGB dimensions: {e}")

                # Check depth dimensions only once if needed
                if self.depth_dir and not determined_depth_dims:
                    img_idx_str = img_name_base.split('_')[-1]
                    depth_fname = f"DepthMap_{img_idx_str}.tiff"
                    first_depth_path = self.depth_dir / depth_fname
                    if first_depth_path.exists():
                        depth_map_np = load_tiff(first_depth_path)
                        if depth_map_np is not None:
                            self.depth_dims = (depth_map_np.shape[0], depth_map_np.shape[1]) # (H_d, W_d)
                            if self.verbose: print(f"  Determined Depth map dimensions: {self.depth_dims} from {depth_fname}")
                            determined_depth_dims = True
                    if not determined_depth_dims and self.verbose:
                         print(f"  Warning: Could not load {depth_fname} to determine depth dimensions.")

                # Always add if intrinsics are found
                matched_indices.append(i)
                final_image_names.append(img_name_with_ext)
                final_intrinsics_log.append(intrinsics_map_log[img_name_base])

            elif self.verbose:
                 print(f"Warning: Pose for image '{img_name_with_ext}' found in CSV but no matching intrinsics in NDJSON. Skipping.")

        if not matched_indices:
            raise RuntimeError("No matching images found between GeoFusion CSV and Log NDJSON.")
        if not self.actual_rgb_dims:
             raise RuntimeError("Could not determine actual RGB image dimensions.")

        # Filter poses based on matched indices
        xyz_positions_f64 = xyz_positions_f64[matched_indices]
        cam_to_world_rotations_f64 = cam_to_world_rotations_f64[matched_indices]

        self.image_names = final_image_names
        intrinsics_log_tensor = torch.stack(final_intrinsics_log) # (N, 3, 3) K for log resolution

        # --- Scale Log Intrinsics to Actual RGB Intrinsics ---
        W_rgb, H_rgb = self.actual_rgb_dims
        if log_image_dims and log_image_dims != self.actual_rgb_dims:
             W_log, H_log = log_image_dims
             if W_log == 0 or H_log == 0: # Avoid division by zero if log dims invalid
                 print("Warning: Logged image dimensions are zero. Cannot scale intrinsics from log to actual RGB.")
                 self.rgb_intrinsics = intrinsics_log_tensor
             else:
                 if self.verbose: print(f"  Scaling intrinsics from Log resolution {log_image_dims} to Actual RGB resolution {self.actual_rgb_dims}")
                 scale_w_rgb = W_rgb / W_log
                 scale_h_rgb = H_rgb / H_log
                 scale_mat_rgb = torch.diag(torch.tensor([scale_w_rgb, scale_h_rgb, 1.0], device=self.device, dtype=self.dtype))
                 # K_rgb = scale_mat_rgb @ K_log (apply to each K in the batch)
                 self.rgb_intrinsics = scale_mat_rgb.unsqueeze(0) @ intrinsics_log_tensor # (1,3,3) @ (N,3,3) -> (N,3,3)
        else:
            if self.verbose and log_image_dims:
                 print(f"  Actual RGB dims {self.actual_rgb_dims} match Log dims {log_image_dims}. No RGB intrinsic scaling needed.")
            self.rgb_intrinsics = intrinsics_log_tensor # Use log intrinsics directly if resolutions match or log dims unknown

        # --- Scale Actual RGB Intrinsics to Depth Intrinsics ---
        if self.depth_dims and self.depth_dims != (H_rgb, W_rgb):
            if self.verbose: print(f"  Actual RGB dims {self.actual_rgb_dims} != Depth dims {self.depth_dims}. Scaling intrinsics for depth.")
            H_d, W_d = self.depth_dims
            if W_rgb == 0 or H_rgb == 0: # Avoid division by zero
                print("Warning: Actual RGB dimensions are zero. Cannot scale intrinsics to depth resolution.")
                self.scaled_intrinsics = self.rgb_intrinsics
            else:
                scale_w_d = W_d / W_rgb
                scale_h_d = H_d / H_rgb
                scale_mat_d = torch.diag(torch.tensor([scale_w_d, scale_h_d, 1.0], device=self.device, dtype=self.dtype))
                # K_depth = scale_mat_d @ K_rgb_actual
                self.scaled_intrinsics = scale_mat_d.unsqueeze(0) @ self.rgb_intrinsics
        else:
             if self.verbose and self.depth_dims:
                 print(f"  Depth dims {self.depth_dims} match Actual RGB dims {self.actual_rgb_dims}. No depth intrinsic scaling needed.")
             self.scaled_intrinsics = self.rgb_intrinsics # Use actual RGB intrinsics if resolutions match or depth missing

        # --- Define Scene Origin and Calculate Relative Poses ---
        if xyz_positions_f64.shape[0] == 0:
             raise RuntimeError("No valid poses available after filtering.")
             
        # Use the ECEF position of the *first valid matched view* as the origin
        self.scene_origin_ecef = xyz_positions_f64[0].clone().detach() # Store origin (float64)
        if self.verbose:
             print(f"  Scene Origin (ECEF of first view): {self.scene_origin_ecef.cpu().tolist()}")
             
        # Calculate relative ECEF positions
        self.camera_centers_relative = (xyz_positions_f64 - self.scene_origin_ecef).to(self.dtype)
        # Keep original ECEF as metadata if needed
        self.camera_centers_ecef = xyz_positions_f64.to(self.dtype)
        
        # --- DEBUG: Print BBox of Relative Centers --- 
        if self.verbose:
            # Calculate bbox based on relative coordinates
            rel_min = self.camera_centers_relative.min(0).values.cpu().numpy()
            rel_max = self.camera_centers_relative.max(0).values.cpu().numpy()
            rel_span = rel_max - rel_min
            print(f"  Relative Centers Min:   [{rel_min[0]:.4f} {rel_min[1]:.4f} {rel_min[2]:.4f}]")
            print(f"  Relative Centers Max:   [{rel_max[0]:.4f} {rel_max[1]:.4f} {rel_max[2]:.4f}]")
            print(f"  Relative Centers Span:  [{rel_span[0]:.4f} {rel_span[1]:.4f} {rel_span[2]:.4f}]")
        # --- End DEBUG ---

        # Calculate W2C transform using RELATIVE centers -> NO, store C2W body rotation
        cam_to_world_rotations_body = cam_to_world_rotations_f64.to(self.dtype)
        self.c2w_body_rotation = cam_to_world_rotations_body
        # self.w2c_rotations = cam_to_world_rotations.transpose(1, 2) # R_wc = R_cw^T
        # # t_wc = -R_wc @ C_relative 
        # self.w2c_translations = -self.w2c_rotations @ self.camera_centers_relative.unsqueeze(-1) 
        # self.w2c_translations = self.w2c_translations.squeeze(-1)

        # --- DEBUG: Print final W2C components for first view ---
        # Adjust debug prints
        if self.verbose and len(self.image_names) > 0:
            print(f"--- Debug Final Pose Info (View 0: {self.image_names[0]}) ---")
            print(f"  Final Body-to-World Rotation (R_cw_body):\n{self.c2w_body_rotation[0].cpu().numpy()}")
            # print(f"  Final World-to-Cam Rotation (R_wc):\n{self.w2c_rotations[0].cpu().numpy()}")
            print(f"  Final Camera Center (Relative): {self.camera_centers_relative[0].cpu().numpy()}")
            # print(f"  Final World-to-Cam Translation (t_wc): {self.w2c_translations[0].cpu().numpy()}")
            # print(f"  Manual t_wc check: ...")
            print("--- End Debug Final Pose Info ---")
        # --- End DEBUG ---

        # Store image and depth paths
        self.image_paths = [self.image_dir / name for name in self.image_names]
        self.depth_paths = []
        self.confidence_paths = []
        if self.depth_dir:
            for name_with_ext in self.image_names:
                img_name_base = name_with_ext.split('.')[0]
                img_idx_str = img_name_base.split('_')[-1]
                depth_fname = f"DepthMap_{img_idx_str}.tiff"
                conf_fname = f"Confidence_{img_idx_str}.tiff"
                dp = self.depth_dir / depth_fname
                cp = self.depth_dir / conf_fname
                self.depth_paths.append(dp)
                self.confidence_paths.append(cp)
        else:
            self.depth_paths = [None] * len(self.image_names)
            self.confidence_paths = [None] * len(self.image_names)

        if self.verbose:
            print(f"Dataset initialized with {len(self.image_names)} matched entries.")
            print(f"  Actual RGB Image dimensions: {self.actual_rgb_dims}")
            if log_image_dims: print(f"  Logged Image dimensions: {log_image_dims}")
            if self.depth_dims: print(f"  Depth map dimensions: {self.depth_dims}")
            print(f"  Final RGB Intrinsics shape: {self.rgb_intrinsics.shape}")
            if self.scaled_intrinsics is not self.rgb_intrinsics: print(f"  Scaled Depth Intrinsics shape: {self.scaled_intrinsics.shape}")
            print(f"  C2W Body Rotation shape: {self.c2w_body_rotation.shape}") # Add this

    def relative_world_to_geodetic(self, points_relative: torch.Tensor) -> torch.Tensor:
        """Converts points from the internal relative world frame back to Geodetic.

        Args:
            points_relative: Tensor of shape (N, 3) with coordinates in the 
                             scene-relative frame (meters).

        Returns:
            Tensor of shape (N, 3) containing (latitude, longitude, altitude) 
            in degrees/meters.
            
        Raises:
            RuntimeError: If the scene origin was not properly initialized.
        """
        if self.scene_origin_ecef is None:
            raise RuntimeError("Scene origin (ECEF) is not set. Dataset may not be initialized correctly.")
            
        # Ensure input is float64 for precision with ECEF
        points_relative_f64 = points_relative.to(dtype=torch.float64, device=self.device)
        
        # Add scene origin (also float64) to get ECEF coordinates
        # scene_origin_ecef needs to be broadcast: (3,) -> (1, 3)
        points_ecef = points_relative_f64 + self.scene_origin_ecef.unsqueeze(0)
        
        # Convert ECEF to Geodetic using the converter
        # xyz_to_geodetic returns (geo, orientation), we only need geo
        geo_coords, _ = self.converter.xyz_to_geodetic(points_ecef)
        
        return geo_coords

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a dataset item by index.

        Args:
            idx: Index of the item.

        Returns:
            A dictionary containing:
                - image: (H_rgb, W_rgb, 3) tensor, channels last, float32, range [0, 1]
                - c2w_body_rotation: (3, 3) body-to-world rotation matrix
                - intrinsics: (3, 3) camera intrinsics matrix K for ACTUAL RGB image
                - image_path: Path to the image file
                - image_height: Height of the ACTUAL RGB image
                - image_width: Width of the ACTUAL RGB image
                - view_idx: The index of the view
                - depth_path: Path to the depth map file (or None)
                - confidence_path: Path to the confidence map file (or None)
                - intrinsics_depth: (3, 3) intrinsics matrix K scaled for depth map
                - depth_map: (H_d, W_d) tensor if load_depth=True, else None
                - confidence_map: (H_d, W_d) tensor if load_depth=True, else None
                - camera_center_ecef: (3,) ECEF coordinates of the camera center
                - camera_center_relative: (3,) relative coordinates of the camera center
        """
        image_path = self.image_paths[idx]
        W_rgb, H_rgb = self.actual_rgb_dims # Use actual loaded dimensions
        depth_path = self.depth_paths[idx]
        conf_path = self.confidence_paths[idx]

        # Load RGB Image
        try:
            img = Image.open(image_path).convert('RGB')
            if img.size != (W_rgb, H_rgb):
                 # This shouldn't happen if actual_rgb_dims is determined correctly
                 print(f"Critical Warning: Image {image_path.name} size {img.size} differs from determined actual size {(W_rgb, H_rgb)}. Resizing.")
                 img = img.resize((W_rgb, H_rgb), Image.Resampling.LANCZOS)
            img_tensor = TF.to_tensor(img).permute(1, 2, 0).to(device=self.device, dtype=self.dtype)
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {e}") from e

        # Prepare output dict
        data = {
            "image": img_tensor,
            # Return C2W Body rotation instead of W2C
            # "w2c_rotation": self.w2c_rotations[idx],
            # "w2c_translation": self.w2c_translations[idx],
            "c2w_body_rotation": self.c2w_body_rotation[idx],
            "intrinsics": self.rgb_intrinsics[idx],
            "image_path": str(image_path),
            "image_height": torch.tensor(H_rgb, device=self.device),
            "image_width": torch.tensor(W_rgb, device=self.device),
            "view_idx": torch.tensor(idx, device=self.device),
            "depth_path": str(depth_path) if depth_path else None,
            "confidence_path": str(conf_path) if conf_path else None,
            "intrinsics_depth": self.scaled_intrinsics[idx],
            "depth_map": None,
            "confidence_map": None,
            "camera_center_ecef": self.camera_centers_ecef[idx],
            "camera_center_relative": self.camera_centers_relative[idx]
        }

        # Optionally load depth/confidence maps
        if self.load_depth:
            # Use try-except block for safer loading
            try:
                 if depth_path and conf_path and depth_path.exists() and conf_path.exists():
                     depth_np = load_tiff(depth_path)
                     conf_np = load_tiff(conf_path)
                     if depth_np is not None and conf_np is not None:
                         data["depth_map"] = torch.from_numpy(depth_np).to(device=self.device, dtype=self.dtype)
                         data["confidence_map"] = torch.from_numpy(conf_np).to(device=self.device, dtype=self.dtype)
                         # Verify dimensions if needed
                         if self.depth_dims and data["depth_map"].shape != self.depth_dims:
                             print(f"Warning: Loaded depth map {depth_path.name} has unexpected shape {data['depth_map'].shape}. Expected {self.depth_dims}.")
                 elif self.verbose:
                     print(f"Notice: Could not load depth/conf for index {idx}, paths: {depth_path}, {conf_path}")
            except Exception as e:
                 print(f"Error loading depth/confidence for index {idx}: {e}")
                 # Keep depth_map and confidence_map as None

        return data 