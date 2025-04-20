#!/usr/bin/env python3
# =============================================================================
#  interactive_visualizer.py
# -----------------------------------------------------------------------------
#  Interactive Debugging Tool for GeoFusion Camera Poses and Point Clouds
# -----------------------------------------------------------------------------
#  Loads GeoFusion data (camera poses, images, depth maps, confidence maps)
#  for a user-specified range of views. It allows interactively cycling 
#  through different potential body-to-camera frame transformations and applying
#  fine-grained rotational adjustments to determine the correct transformation 
#  that aligns the point clouds from different views in a common world frame.
#
#  Usage:
#      python interactive_visualizer.py [options]
#      Run with --help for detailed options.
#
#  Controls:
#    T : Cycle through base body-to-camera transformations (defined in CANDIDATE_TRANSFORMS).
#    C : Toggle point colors (Image Colors vs. View Index Colors).
#    X/x : Apply positive/negative rotation adjustment around the X-axis.
#    Y/y : Apply positive/negative rotation adjustment around the Y-axis.
#    Z/z : Apply positive/negative rotation adjustment around the Z-axis.
#    R : Reset rotation adjustments for the current base transform to zero.
#    Q : Quit the visualizer.
#
#  MIT License – © 2025 DeepEarth Contributors
# =============================================================================

import torch
import pathlib
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import copy
import math
from scipy.spatial.transform import Rotation as R_scipy
from torch.utils.data import DataLoader
import argparse
from typing import Dict, List, Tuple, Optional, Any

try:
    import open3d as o3d
except ImportError:
    print("Error: 'open3d' package not found. Please install it: pip install open3d")
    o3d = None

from deepearth.reconstruction.geofusion_dataset import GeoFusionDataset
from deepearth.reconstruction.point_cloud_utils import load_tiff, unproject_depth, project_points_to_image, sample_colors_from_image

# --- Script Configuration --- 
# Default values, can be overridden by command-line arguments
DEFAULT_MIN_IMAGE_IDX = 0
DEFAULT_MAX_IMAGE_IDX = 10 # -1 means use all available images
DEFAULT_IMAGE_STEP = 2
DEFAULT_CONFIDENCE_THRESHOLD = 1.5
DEFAULT_BATCH_SIZE = 8
DEFAULT_CAMERA_FRAME_SIZE = 0.2
DEFAULT_POINT_SIZE = 2.0
DEFAULT_SHOW_IMAGE_COLORS = True
DEFAULT_ADJUSTMENT_EULER = [0.0, 0.0, 0.0] # Start with no adjustment
ADJUST_ANGLE_DEG = 5.0 # Rotation increment per key press

# --- Candidate Body-to-Camera Transformations --- 
# Define candidate R_body_cam matrices (transform points FROM body TO camera)
# Add more candidates based on potential hardware/software conventions.
CANDIDATE_TRANSFORMS: Dict[str, torch.Tensor] = {
    "Identity (Body=Cam)": torch.eye(3, dtype=torch.float64),
    "PIX4D CbB.T": torch.tensor([
        [0.0, 1.0,  0.0],
        [1.0, 0.0,  0.0],
        [0.0, 0.0, -1.0]
        ], dtype=torch.float64),
    "Z-Flip (Attempt)": torch.diag(torch.tensor([1.0, 1.0, -1.0], dtype=torch.float64)),
    # OpenCV (+X Right, +Y Down, +Z Fwd) to ARKit (+X Right, +Y Up, -Z Fwd)
    "OpenCV_to_ARKit": torch.tensor([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0]
        ], dtype=torch.float64),
    # Phone Cam (Assumed: Y up screen, X right screen, Z out screen) to ARKit
    "Phone_to_ARKit": torch.diag(torch.tensor([1.0, 1.0, -1.0], dtype=torch.float64))
}
TRANSFORM_NAMES: List[str] = list(CANDIDATE_TRANSFORMS.keys())

# --- Rotation Helper Functions ---
def rotation_matrix(axis: str, angle_deg: float, device: torch.device, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Creates a 3x3 rotation matrix around a given axis."""
    angle_rad = math.radians(angle_deg)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    if axis == 'x':
        return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=dtype, device=device)
    elif axis == 'y':
        return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=dtype, device=device)
    elif axis == 'z':
        return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=dtype, device=device)
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

# --- Global State for Visualization ---
class VisState:
    """Manages the interactive state of the visualizer."""
    def __init__(self, args: argparse.Namespace):
        self.current_transform_idx: int = 0
        self.show_image_colors: bool = args.show_image_colors
        self.needs_update: bool = True
        # Initialize adjustment rotation based on default or args
        self.adjustment_rotation: torch.Tensor = torch.eye(3, dtype=torch.float64)
        # Apply initial adjustment rotations in Z, Y, X order
        if abs(args.adjust_z) > 1e-6:
             rot_z = rotation_matrix('z', args.adjust_z, device="cpu", dtype=torch.float64)
             self.adjustment_rotation = rot_z @ self.adjustment_rotation
        if abs(args.adjust_y) > 1e-6:
             rot_y = rotation_matrix('y', args.adjust_y, device="cpu", dtype=torch.float64)
             self.adjustment_rotation = rot_y @ self.adjustment_rotation
        if abs(args.adjust_x) > 1e-6:
             rot_x = rotation_matrix('x', args.adjust_x, device="cpu", dtype=torch.float64)
             self.adjustment_rotation = rot_x @ self.adjustment_rotation

vis_state: Optional[VisState] = None # Will be initialized in main

# --- Data Storage --- 
class ViewData:
    """Stores pre-processed data for each visualized view."""
    def __init__(self):
        self.points_cam: List[torch.Tensor] = [] # Points in camera coordinates
        self.colors_img: List[Optional[torch.Tensor]] = [] # Colors sampled from RGB image
        self.colors_view: List[torch.Tensor] = [] # Colors assigned based on view index
        self.c2w_body_rotations: List[torch.Tensor] = [] # Original Body-to-World rotations
        self.relative_centers: List[torch.Tensor] = [] # Camera centers relative to first view
        self.num_valid_views: int = 0

# --- Argument Parser --- 
def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Interactive visualizer for GeoFusion camera poses and point clouds.")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to the dataset directory (e.g., landscape_architecture_studio). Required if default path is incorrect.")
    parser.add_argument("--min_idx", type=int, default=DEFAULT_MIN_IMAGE_IDX, help="Index of the first image to process.")
    parser.add_argument("--max_idx", type=int, default=DEFAULT_MAX_IMAGE_IDX, help="Index of the last image to process (-1 for all).")
    parser.add_argument("--step", type=int, default=DEFAULT_IMAGE_STEP, help="Process every Nth image.")
    parser.add_argument("--conf_thresh", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD, help="Confidence threshold for depth points.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for data loading.")
    parser.add_argument("--cam_size", type=float, default=DEFAULT_CAMERA_FRAME_SIZE, help="Size of visualized camera frames.")
    parser.add_argument("--point_size", type=float, default=DEFAULT_POINT_SIZE, help="Visual size of points.")
    parser.add_argument("--start_color_mode", choices=["image", "view"], default="image" if DEFAULT_SHOW_IMAGE_COLORS else "view", help="Initial color mode.")
    parser.add_argument("--adjust_x", type=float, default=DEFAULT_ADJUSTMENT_EULER[0], help="Initial X rotation adjustment (degrees).")
    parser.add_argument("--adjust_y", type=float, default=DEFAULT_ADJUSTMENT_EULER[1], help="Initial Y rotation adjustment (degrees).")
    parser.add_argument("--adjust_z", type=float, default=DEFAULT_ADJUSTMENT_EULER[2], help="Initial Z rotation adjustment (degrees).")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose dataset loading output.")
    
    args = parser.parse_args()

    if args.dataset_dir is None:
        base_dir = pathlib.Path(__file__).resolve().parent.parent.parent.parent
        args.dataset_dir = base_dir / "src" / "datasets" / "geofusion" / "landscape_architecture_studio"
        print(f"Warning: --dataset_dir not specified, using default: {args.dataset_dir}")
    else:
        args.dataset_dir = pathlib.Path(args.dataset_dir)
        
    args.show_image_colors = (args.start_color_mode == "image")

    return args

# --- Main Visualization Logic --- 
def main(args: argparse.Namespace):
    """Main function to load data, set up, and run the visualizer."""
    global vis_state
    if o3d is None:
        exit("Open3D is required for visualization.")
        
    vis_state = VisState(args)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.float32 

    print(f"Using device: {DEVICE}")
    print(f"Loading dataset from: {args.dataset_dir}")

    # --- 1. Load Dataset & Create DataLoader ---
    try:
        dataset = GeoFusionDataset(args.dataset_dir, device=DEVICE, dtype=DTYPE, verbose=args.verbose, load_depth=True)
        
        start_idx = max(0, args.min_idx)
        end_idx = len(dataset) if args.max_idx < 0 else min(len(dataset), args.max_idx + 1)
        step = max(1, args.step)
        subset_indices = list(range(start_idx, end_idx, step))
        
        num_views = len(subset_indices)
        if num_views == 0:
             print("Error: No views selected with current min/max/step settings. Exiting.")
             exit()
             
        subset_dataset = torch.utils.data.Subset(dataset, subset_indices)
        # num_workers > 0 can sometimes cause issues with CUDA in setup phase
        dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0) 
        print(f"Created DataLoader for {num_views} views (Indices {start_idx} to {end_idx-1}, step {step}) with batch size {args.batch_size}.")

    except FileNotFoundError as e:
         print(f"Error: Dataset directory or required files not found: {e}")
         exit()
    except Exception as e:
        print(f"Failed to initialize dataset/dataloader: {e}")
        import traceback
        traceback.print_exc()
        exit()

    # --- 2. Pre-process Data for Visualization (Batched) ---
    print(f"\nPre-processing data using DataLoader...")
    view_data = ViewData()
    cmap = plt.get_cmap("viridis")
    view_color_map = [cmap(k / max(1, num_views - 1))[:3] for k in range(num_views)]

    for batch_idx, data_batch in enumerate(tqdm.tqdm(dataloader, desc="Processing Batches")):
        batch_size_actual = data_batch["view_idx"].shape[0]
        # Extract required tensors from the batch
        # ... (extraction logic remains the same) ...
        depth_maps = data_batch.get("depth_map")
        conf_maps = data_batch.get("confidence_map")
        intrinsics_depth = data_batch.get("intrinsics_depth")
        c2w_body_rots = data_batch.get("c2w_body_rotation")
        relative_centers = data_batch.get("camera_center_relative")
        view_indices_batch = data_batch.get("view_idx")
        rgb_images = data_batch.get("image")
        intrinsics_rgb_batch = data_batch.get("intrinsics")

        if not all(t is not None for t in [depth_maps, conf_maps, intrinsics_depth, c2w_body_rots, relative_centers, view_indices_batch]):
            print(f"Warning: Skipping batch {batch_idx} due to missing essential data.")
            continue
            
        points_cam_list_batch = []
        colors_img_list_batch = []
        colors_view_list_batch = []
        c2w_body_rot_list_batch = []
        relative_center_list_batch = []
        valid_view_processed_count = 0

        for j in range(batch_size_actual):
            # Unproject one view at a time to get camera coordinates
            _, points_cam_single = unproject_depth(
                depth_map=depth_maps[j],
                intrinsics=intrinsics_depth[j],
                # Pass dummy C2W pose as only P_cam is needed here
                cam_to_world_rot=torch.eye(3, device=DEVICE, dtype=DTYPE),
                cam_center_world=torch.zeros(3, device=DEVICE, dtype=DTYPE),
                confidence_map=conf_maps[j],
                confidence_threshold=args.conf_thresh,
                return_cam_coords=True
            )
            
            if points_cam_single is not None and points_cam_single.shape[0] > 0:
                points_cam_list_batch.append(points_cam_single)
                # Store the corresponding pose components for this valid view
                c2w_body_rot_list_batch.append(c2w_body_rots[j])
                relative_center_list_batch.append(relative_centers[j])
                valid_view_processed_count += 1
                
                # Assign view color
                try:
                    original_subset_index = subset_indices.index(view_indices_batch[j].item())
                except ValueError:
                    print(f"Error: Could not find view index {view_indices_batch[j].item()} in subset list.")
                    original_subset_index = 0 
                
                view_color_np = np.array(view_color_map[original_subset_index]).reshape(1, 3)
                view_point_colors = np.tile(view_color_np, (points_cam_single.shape[0], 1))
                colors_view_list_batch.append(torch.from_numpy(view_point_colors).to(DEVICE, DTYPE))
                
                # Sample image colors if available
                current_intrinsics_rgb = intrinsics_rgb_batch[j] if intrinsics_rgb_batch is not None else None
                if rgb_images is not None and current_intrinsics_rgb is not None:
                    # Project P_cam (relative to origin) back to image plane
                    pixel_coords_rgb = project_points_to_image(
                        points_world=points_cam_single,
                        cam_to_world_rot=torch.eye(3, device=DEVICE, dtype=DTYPE),
                        cam_center_world=torch.zeros(3, device=DEVICE, dtype=DTYPE),
                        intrinsics=current_intrinsics_rgb 
                    )
                    colors_img_single = sample_colors_from_image(pixel_coords=pixel_coords_rgb, image=rgb_images[j])
                    colors_img_list_batch.append(colors_img_single)
                else:
                    colors_img_list_batch.append(None) 
            else:
                # print(f"Warning: No valid points generated for view index {view_indices_batch[j].item()} in camera coords.")
                # Add placeholders if a view fails inside the batch
                colors_view_list_batch.append(None)
                colors_img_list_batch.append(None) 
        
        # Append processed data for valid views in the batch to the main storage
        if points_cam_list_batch:
            view_data.points_cam.extend(points_cam_list_batch)
            view_data.colors_img.extend(colors_img_list_batch)
            view_data.colors_view.extend(colors_view_list_batch)
            view_data.c2w_body_rotations.extend(c2w_body_rot_list_batch)
            view_data.relative_centers.extend(relative_center_list_batch)
        
        view_data.num_valid_views += valid_view_processed_count

    if view_data.num_valid_views == 0:
        print("No valid points generated from any selected views. Exiting.")
        exit()
        
    print(f"Pre-processed data for {view_data.num_valid_views} views.")

    # --- 3. Setup Open3D Visualization --- 
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Interactive Pose Debugger")
    opt = vis.get_render_option()
    opt.point_size = args.point_size
    
    pcd = o3d.geometry.PointCloud()
    # Keep track of all geometries added to the scene
    geometries: Dict[str, o3d.geometry.Geometry] = {"pcd": pcd}
    first_update = True

    # --- 4. Define Update Function --- 
    def update_geometry(vis: o3d.visualization.VisualizerWithKeyCallback, init: bool = False):
        """Updates the point cloud and camera poses in the visualizer based on current state."""
        nonlocal first_update, geometries # Allow modification of these variables
        state_transform_idx = vis_state.current_transform_idx
        state_show_image_colors = vis_state.show_image_colors
        R_adjust = vis_state.adjustment_rotation.to(DEVICE, DTYPE)
        base_transform_name = TRANSFORM_NAMES[state_transform_idx]
        R_body_cam_base = CANDIDATE_TRANSFORMS[base_transform_name].to(DEVICE, DTYPE)
        R_body_cam_effective = R_body_cam_base @ R_adjust

        try:
            adj_euler_deg = R_scipy.from_matrix(R_adjust.cpu().numpy()).as_euler('xyz', degrees=True)
        except Exception:
            adj_euler_deg = np.array([np.nan, np.nan, np.nan])

        print(f"\nUpdating geometry with:")
        print(f"  Base Transform : {base_transform_name}")
        print(f"  Adjustment (Euler XYZ deg): [{adj_euler_deg[0]:>6.1f} {adj_euler_deg[1]:>6.1f} {adj_euler_deg[2]:>6.1f}] ")
        print(f"  Color Mode     : {'Image Colors' if state_show_image_colors else 'View Colors'}")

        all_points_world_gpu_list = []
        all_colors_gpu_list = []
        new_cam_geometries = []

        for i in range(view_data.num_valid_views):
            # Retrieve pre-processed data for this view
            points_cam = view_data.points_cam[i]
            R_cw_body = view_data.c2w_body_rotations[i]
            C_relative = view_data.relative_centers[i]
            
            # Apply effective body-to-camera transform
            R_cw_cam = R_cw_body @ R_body_cam_effective 
            
            # Transform points_cam to world (relative)
            points_world = torch.matmul(points_cam, R_cw_cam.T) + C_relative.unsqueeze(0)
            all_points_world_gpu_list.append(points_world)
            
            # Select colors
            if state_show_image_colors and view_data.colors_img[i] is not None:
                all_colors_gpu_list.append(view_data.colors_img[i])
            else:
                all_colors_gpu_list.append(view_data.colors_view[i])
                
            # Create camera frame geometry
            cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.cam_size, origin=[0, 0, 0])
            pose_matrix = np.eye(4)
            pose_matrix[:3, :3] = R_cw_cam.cpu().numpy()
            pose_matrix[:3, 3] = C_relative.cpu().numpy()
            cam_frame.transform(pose_matrix)
            new_cam_geometries.append(cam_frame)

        # --- Update Point Cloud Geometry --- 
        if not all_points_world_gpu_list:
             print("Warning: No points to visualize after processing views.")
             combined_points_np = np.empty((0,3), dtype=np.float32)
             combined_colors_np = np.empty((0,3), dtype=np.float32)
        else:     
            combined_points_gpu = torch.cat(all_points_world_gpu_list, dim=0)
            combined_colors_gpu = torch.cat(all_colors_gpu_list, dim=0)
            combined_points_np = combined_points_gpu.cpu().numpy()
            combined_colors_np = combined_colors_gpu.cpu().numpy()
        
        # Update points and colors of the existing pcd object
        geometries["pcd"].points = o3d.utility.Vector3dVector(combined_points_np)
        if combined_colors_np.shape[0] == combined_points_np.shape[0]:
            geometries["pcd"].colors = o3d.utility.Vector3dVector(combined_colors_np)
        else:
            print(f"Warning: Color/Point mismatch ({combined_colors_np.shape[0]} vs {combined_points_np.shape[0]}). Clearing colors.")
            geometries["pcd"].colors = o3d.utility.Vector3dVector([])

        # --- Update Scene Geometries (Remove old, Add new) --- 
        # Remove old camera frames
        geometries_to_remove = [name for name in geometries if name.startswith("cam_")]
        for name in geometries_to_remove:
            vis.remove_geometry(geometries[name], reset_bounding_box=False)
            del geometries[name]
        
        # Add new camera frames
        for i, frame in enumerate(new_cam_geometries):
            name = f"cam_{i}"
            vis.add_geometry(frame, reset_bounding_box=False)
            geometries[name] = frame
            
        # Add world frame only on initialization
        if init:
            if "world_frame" not in geometries:
                 world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
                 vis.add_geometry(world_frame, reset_bounding_box=False)
                 geometries["world_frame"] = world_frame
            # Add the point cloud geometry on init
            vis.add_geometry(geometries["pcd"], reset_bounding_box=True) 
        else:
            # Update existing point cloud geometry
            vis.update_geometry(geometries["pcd"])

        # Reset bounding box only on initial load
        # vis.reset_view_point(True) # Optionally reset camera view point
        vis.update_renderer()
        vis_state.needs_update = False
        print("Update complete.")

    # --- 5. Define Key Callbacks --- 
    def cycle_transform(vis):
        """Cycles through the base body-to-camera transformations."""
        vis_state.current_transform_idx = (vis_state.current_transform_idx + 1) % len(TRANSFORM_NAMES)
        vis_state.adjustment_rotation = torch.eye(3, dtype=torch.float64) # Reset adjustment
        vis_state.needs_update = True
        update_geometry(vis)
        return False

    def toggle_colors(vis):
        """Toggles point cloud colors between image colors and view index colors."""
        vis_state.show_image_colors = not vis_state.show_image_colors
        vis_state.needs_update = True
        update_geometry(vis)
        return False
        
    def adjust_rotation(vis, axis: str, angle_deg: float):
        """Applies an incremental rotation adjustment."""
        rot_mat = rotation_matrix(axis, angle_deg, device=vis_state.adjustment_rotation.device, dtype=torch.float64)
        vis_state.adjustment_rotation = rot_mat @ vis_state.adjustment_rotation
        vis_state.needs_update = True
        update_geometry(vis)
        return False
        
    def reset_adjustment(vis):
        """Resets the rotation adjustment to identity."""
        vis_state.adjustment_rotation = torch.eye(3, dtype=torch.float64)
        vis_state.needs_update = True
        update_geometry(vis)
        print("Adjustment rotation reset to Identity.")
        return False

    def request_quit(vis):
        """Closes the visualization window."""
        vis.destroy_window()
        print("Visualizer closing.")
        return False

    # --- 6. Register Callbacks and Run --- 
    vis.register_key_callback(ord("T"), cycle_transform)
    vis.register_key_callback(ord("C"), toggle_colors)
    vis.register_key_callback(ord("X"), lambda v: adjust_rotation(v, 'x', ADJUST_ANGLE_DEG))
    vis.register_key_callback(ord("x"), lambda v: adjust_rotation(v, 'x', -ADJUST_ANGLE_DEG))
    vis.register_key_callback(ord("Y"), lambda v: adjust_rotation(v, 'y', ADJUST_ANGLE_DEG))
    vis.register_key_callback(ord("y"), lambda v: adjust_rotation(v, 'y', -ADJUST_ANGLE_DEG))
    vis.register_key_callback(ord("Z"), lambda v: adjust_rotation(v, 'z', ADJUST_ANGLE_DEG))
    vis.register_key_callback(ord("z"), lambda v: adjust_rotation(v, 'z', -ADJUST_ANGLE_DEG))
    vis.register_key_callback(ord("R"), reset_adjustment)
    vis.register_key_callback(ord("Q"), request_quit)
    
    print("\nStarting interactive visualizer...")
    print("  Controls:")
    print("    [T] Cycle Base Body->Camera Transform")
    print("    [C] Toggle Point Colors (Image vs. View Index)")
    print("    [X/x] Adjust X Rotation (+/- {} deg)".format(ADJUST_ANGLE_DEG))
    print("    [Y/y] Adjust Y Rotation (+/- {} deg)".format(ADJUST_ANGLE_DEG))
    print("    [Z/z] Adjust Z Rotation (+/- {} deg)".format(ADJUST_ANGLE_DEG))
    print("    [R] Reset Rotation Adjustment to Identity")
    print("    [Q] Quit")

    # Initial update to populate the scene
    update_geometry(vis, init=True)
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    args = parse_args()
    main(args) 