#!/usr/bin/env python3
"""
End-to-end V-JEPA 2 processing pipeline.
Handles feature extraction, visualization, and packaging.
"""

import os
import sys
import argparse
import subprocess
import shutil
import json
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
from PIL import Image
import logging
from typing import List, Dict, Optional, Tuple
import re
import umap
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import cv2
import imageio


def setup_logger(name="vjepa2_pipeline", log_dir="logs"):
    """Setup logger for pipeline"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"{name}_{timestamp}.log"
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler(sys.stdout)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


class VJEPA2Pipeline:
    """Complete V-JEPA 2 processing pipeline."""
    
    def __init__(self, 
                 frame_dir: str,
                 output_dir: str,
                 first_frame: int = -1,
                 last_frame: int = -1,
                 device: str = "cuda:0",
                 batch_size: int = 2,
                 stride: Optional[int] = None):
        """
        Initialize the pipeline.
        
        Args:
            frame_dir: Directory containing input frames
            output_dir: Directory for all outputs
            first_frame: First frame to process (-1 for all)
            last_frame: Last frame to process (-1 for all)
            device: GPU device to use
            batch_size: Batch size for processing
            stride: Stride for sliding windows (default=16)
        """
        self.frame_dir = Path(frame_dir)
        self.output_dir = Path(output_dir)
        self.first_frame = first_frame
        self.last_frame = last_frame
        self.device = device
        self.batch_size = batch_size
        self.stride = stride if stride else 16
        
        self.logger = setup_logger("vjepa2_pipeline")
        
        # Create output directories
        self.features_dir = self.output_dir / "features"
        self.viz_dir = self.output_dir / "visualizations"
        self.video_dir = self.output_dir / "videos"
        self.package_dir = self.output_dir / "package"
        
        for dir_path in [self.features_dir, self.viz_dir, self.video_dir, self.package_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
    
    def extract_features(self):
        """Run feature extraction using vjepa2_sequential_extractor.py"""
        self.logger.info("Starting feature extraction...")
        
        cmd = [
            "python", "vjepa2_sequential_extractor.py",
            "--frame_dir", str(self.frame_dir),
            "--output_dir", str(self.features_dir),
            "--device", self.device,
            "--batch_size", str(self.batch_size),
            "--stride", str(self.stride)
        ]
        
        if self.first_frame != -1:
            cmd.extend(["--first_frame", str(self.first_frame)])
        if self.last_frame != -1:
            cmd.extend(["--last_frame", str(self.last_frame)])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            self.logger.error(f"Feature extraction failed: {result.stderr}")
            raise RuntimeError("Feature extraction failed")
        
        self.logger.info("Feature extraction complete")
        
        # Get list of generated feature files
        self.feature_files = sorted(self.features_dir.glob("features_frames_*.pt"))
        return self.feature_files
    
    def load_features(self, feature_path: Path) -> Tuple[torch.Tensor, Dict]:
        """Load features and metadata from file."""
        data = torch.load(feature_path, map_location='cpu', weights_only=False)
        features = data['features']
        metadata = data.get('metadata', {})
        return features, metadata
    
    def get_original_frame_paths(self, metadata: Dict) -> List[Path]:
        """Get paths to original frames from metadata."""
        frame_files = metadata.get('frame_files', [])
        frame_paths = []
        
        for frame_file in frame_files:
            frame_path = self.frame_dir / frame_file
            if frame_path.exists():
                frame_paths.append(frame_path)
        
        return frame_paths
    
    def compute_pca_visualization(self, features: torch.Tensor) -> np.ndarray:
        """
        Compute PCA component 1 across spatial features per temporal frame.
        
        Args:
            features: Tensor of shape [4608, 1408]
            
        Returns:
            Array of shape [8, 24, 24] with PCA component 1 values
        """
        # Reshape to [8, 576, 1408]
        features_reshaped = features.view(8, 576, 1408).numpy()
        
        pca_results = np.zeros((8, 576))
        
        for t in range(8):
            # Get spatial features for this temporal frame
            spatial_features = features_reshaped[t]  # [576, 1408]
            
            # Compute PCA
            pca = PCA(n_components=1)
            pca_component = pca.fit_transform(spatial_features)  # [576, 1]
            pca_results[t] = pca_component.squeeze()
        
        # Reshape to [8, 24, 24]
        pca_results = pca_results.reshape(8, 24, 24)
        
        return pca_results
    
    def apply_turbo_colormap(self, values: np.ndarray) -> np.ndarray:
        """
        Apply Turbo colormap to values.
        
        Args:
            values: Array of values
            
        Returns:
            RGB array with Turbo colormap applied
        """
        # Normalize to [0, 1]
        vmin, vmax = values.min(), values.max()
        if vmax > vmin:
            normalized = (values - vmin) / (vmax - vmin)
        else:
            normalized = np.zeros_like(values)
        
        # Apply Turbo colormap
        colormap = cm.get_cmap('turbo')
        colored = colormap(normalized)
        
        # Convert to RGB (remove alpha channel)
        rgb = (colored[..., :3] * 255).astype(np.uint8)
        
        return rgb
    
    def compute_umap_visualization(self, features: torch.Tensor, shared_reducer=None) -> Tuple[np.ndarray, Optional[object]]:
        """
        Compute UMAP reduction to RGB.
        
        Args:
            features: Tensor of shape [N*4608, 1408] where N is number of batches
            shared_reducer: Pre-fitted UMAP reducer for consistent mapping across batches
            
        Returns:
            Tuple of (RGB array, fitted reducer)
        """
        features_np = features.numpy()
        
        if shared_reducer is None:
            # Create and fit new reducer
            reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, 
                               random_state=42, metric='cosine')
            embeddings_3d = reducer.fit_transform(features_np)
        else:
            # Use existing reducer for consistent mapping
            reducer = shared_reducer
            embeddings_3d = reducer.transform(features_np)
        
        # Normalize to [0, 1]
        embeddings_3d -= embeddings_3d.min(axis=0)
        embeddings_3d /= embeddings_3d.max(axis=0)
        
        # Convert to RGB
        rgb_values = (embeddings_3d * 255).astype(np.uint8)
        
        return rgb_values, reducer
    
    def save_visualization_frames(self, 
                                 feature_file: Path,
                                 umap_viz: bool = False,
                                 pca_viz: bool = False,
                                 original_images: bool = False):
        """Save visualization frames for a feature file."""
        features, metadata = self.load_features(feature_file)
        
        # Get frame info from filename
        match = re.search(r'features_frames_(\d+)_(\d+)', feature_file.name)
        if match:
            start_frame = int(match.group(1))
            end_frame = int(match.group(2))
            frame_prefix = f"frames_{start_frame:04d}_{end_frame:04d}"
        else:
            frame_prefix = feature_file.stem
        
        frame_viz_dir = self.viz_dir / frame_prefix
        frame_viz_dir.mkdir(exist_ok=True)
        
        saved_files = []
        
        # Save original images if requested
        if original_images:
            frame_paths = self.get_original_frame_paths(metadata)
            for i, frame_path in enumerate(frame_paths):
                if frame_path.exists():
                    img = Image.open(frame_path)
                    # Get temporal token this frame belongs to
                    t = i // 2  # Each temporal token represents 2 frames
                    output_path = frame_viz_dir / f"original_t{t}_frame_{i:02d}.png"
                    img.save(output_path)
                    saved_files.append(output_path)
        
        # UMAP visualization
        if umap_viz:
            self.logger.info(f"Computing UMAP visualization for {frame_prefix}")
            rgb_values, _ = self.compute_umap_visualization(features)
            rgb_patches = rgb_values.reshape(8, 24, 24, 3)
            
            for t in range(8):
                frame = rgb_patches[t]
                img = Image.fromarray(frame, mode='RGB')
                img_large = img.resize((240, 240), Image.NEAREST)
                output_path = frame_viz_dir / f"umap_t{t}.png"
                img_large.save(output_path)
                saved_files.append(output_path)
        
        # PCA visualization
        if pca_viz:
            self.logger.info(f"Computing PCA visualization for {frame_prefix}")
            pca_values = self.compute_pca_visualization(features)
            
            for t in range(8):
                frame = pca_values[t]
                rgb_frame = self.apply_turbo_colormap(frame)
                img = Image.fromarray(rgb_frame, mode='RGB')
                img_large = img.resize((240, 240), Image.NEAREST)
                output_path = frame_viz_dir / f"pca_t{t}.png"
                img_large.save(output_path)
                saved_files.append(output_path)
        
        return saved_files
    
    def create_video(self,
                    feature_files: List[Path],
                    video_fps: int = 6,
                    mp4: bool = False,
                    gif: bool = False,
                    video_original_images: bool = True,
                    video_umap_visualization: bool = False,
                    video_pca_visualization: bool = False,
                    independent_videos: bool = True,
                    video_stacked_visualization: bool = False):
        """Create video animations from processed frames."""
        
        if not (mp4 or gif):
            self.logger.info("No video output requested")
            return []
        
        video_files = []
        
        # Determine which visualizations to include
        viz_types = []
        if video_original_images:
            viz_types.append('original')
        if video_umap_visualization:
            viz_types.append('umap')
        if video_pca_visualization:
            viz_types.append('pca')
        
        if not viz_types and not video_stacked_visualization:
            self.logger.warning("No visualization types selected for video")
            return []
        
        # Check if we have multiple batches for a continuous sequence
        if len(feature_files) > 1:
            self.logger.info(f"Processing {len(feature_files)} batches as continuous sequence")
            return self.create_combined_video(
                feature_files, video_fps, mp4, gif,
                video_original_images, video_umap_visualization, 
                video_pca_visualization, independent_videos,
                video_stacked_visualization
            )
        
        # Process each feature file
        for feature_file in feature_files:
            features, metadata = self.load_features(feature_file)
            
            # Get frame info
            match = re.search(r'features_frames_(\d+)_(\d+)', feature_file.name)
            if match:
                start_frame = int(match.group(1))
                end_frame = int(match.group(2))
                frame_prefix = f"frames_{start_frame:04d}_{end_frame:04d}"
            else:
                frame_prefix = feature_file.stem
            
            # Prepare frames for each visualization type
            all_frames = {viz_type: [] for viz_type in viz_types}
            
            # Get original frames
            if 'original' in viz_types:
                frame_paths = self.get_original_frame_paths(metadata)
                for frame_path in frame_paths:
                    if frame_path.exists():
                        img = cv2.imread(str(frame_path))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        all_frames['original'].append(img)
            
            # Generate UMAP frames
            if 'umap' in viz_types:
                rgb_values, _ = self.compute_umap_visualization(features)
                rgb_patches = rgb_values.reshape(8, 24, 24, 3)
                for t in range(8):
                    frame = rgb_patches[t]
                    # Resize for better visibility
                    frame_large = cv2.resize(frame, (480, 480), interpolation=cv2.INTER_NEAREST)
                    # Duplicate frame for both input frames this token represents
                    all_frames['umap'].extend([frame_large, frame_large])
            
            # Generate PCA frames
            if 'pca' in viz_types:
                pca_values = self.compute_pca_visualization(features)
                for t in range(8):
                    frame = pca_values[t]
                    rgb_frame = self.apply_turbo_colormap(frame)
                    # Resize for better visibility
                    frame_large = cv2.resize(rgb_frame, (480, 480), interpolation=cv2.INTER_NEAREST)
                    # Duplicate frame for both input frames this token represents
                    all_frames['pca'].extend([frame_large, frame_large])
            
            # Create videos
            if independent_videos:
                # Create separate video for each visualization type
                for viz_type in viz_types:
                    frames = all_frames[viz_type]
                    if not frames:
                        continue
                    
                    base_name = f"{frame_prefix}_{viz_type}"
                    
                    if mp4:
                        mp4_path = self.video_dir / f"{base_name}.mp4"
                        self.save_mp4(frames, mp4_path, video_fps)
                        video_files.append(mp4_path)
                    
                    if gif:
                        gif_path = self.video_dir / f"{base_name}.gif"
                        self.save_gif(frames, gif_path, video_fps)
                        video_files.append(gif_path)
            else:
                # Create combined video with all visualizations
                # Stack frames horizontally
                combined_frames = []
                max_frames = max(len(all_frames[vt]) for vt in viz_types)
                
                for i in range(max_frames):
                    row = []
                    for viz_type in viz_types:
                        if i < len(all_frames[viz_type]):
                            frame = all_frames[viz_type][i]
                        else:
                            # Use last frame if this viz has fewer frames
                            frame = all_frames[viz_type][-1]
                        row.append(frame)
                    
                    # Resize all frames to same height
                    min_height = min(f.shape[0] for f in row)
                    row_resized = []
                    for frame in row:
                        if frame.shape[0] != min_height:
                            aspect = frame.shape[1] / frame.shape[0]
                            new_width = int(min_height * aspect)
                            frame = cv2.resize(frame, (new_width, min_height))
                        row_resized.append(frame)
                    
                    combined = np.hstack(row_resized)
                    combined_frames.append(combined)
                
                base_name = f"{frame_prefix}_combined"
                
                if mp4:
                    mp4_path = self.video_dir / f"{base_name}.mp4"
                    self.save_mp4(combined_frames, mp4_path, video_fps)
                    video_files.append(mp4_path)
                
                if gif:
                    gif_path = self.video_dir / f"{base_name}.gif"
                    self.save_gif(combined_frames, gif_path, video_fps)
                    video_files.append(gif_path)
            
            # Create stacked visualization if requested
            if video_stacked_visualization:
                stacked_frames = self.create_stacked_frames(feature_file, features, metadata)
                if stacked_frames:
                    base_name = f"{frame_prefix}_stacked"
                    
                    if mp4:
                        mp4_path = self.video_dir / f"{base_name}.mp4"
                        self.save_mp4(stacked_frames, mp4_path, video_fps)
                        video_files.append(mp4_path)
                    
                    if gif:
                        gif_path = self.video_dir / f"{base_name}.gif"
                        self.save_gif(stacked_frames, gif_path, video_fps)
                        video_files.append(gif_path)
        
        return video_files
    
    def create_stacked_frames(self, feature_file: Path, features: torch.Tensor, metadata: Dict) -> List[np.ndarray]:
        """Create stacked frames with original on top and overlay on bottom.
        
        Returns frames with:
        - Top row: original images at 1024x1024
        - Bottom row: original images with UMAP overlay at 50% alpha
        """
        # Get original frame paths
        frame_paths = self.get_original_frame_paths(metadata)
        if not frame_paths:
            self.logger.warning("No original frames found for stacked visualization")
            return []
        
        # Generate UMAP visualization
        rgb_values, _ = self.compute_umap_visualization(features)
        rgb_patches = rgb_values.reshape(8, 24, 24, 3)
        
        stacked_frames = []
        
        # Process each original frame
        for i, frame_path in enumerate(frame_paths):
            if not frame_path.exists():
                continue
            
            # Load and resize original image to 1024x1024
            original = cv2.imread(str(frame_path))
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            original_1024 = cv2.resize(original, (1024, 1024))
            
            # Determine which temporal token this frame belongs to
            # Each temporal token represents 2 consecutive frames
            t = i // 2
            
            # Get UMAP patch for this temporal token
            umap_patch = rgb_patches[t]  # [24, 24, 3]
            
            # Resize UMAP patch to 1024x1024
            umap_1024 = cv2.resize(umap_patch, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            
            # Create overlay with 50% alpha
            overlay = original_1024.copy()
            overlay = cv2.addWeighted(overlay, 0.5, umap_1024, 0.5, 0)
            
            # Stack vertically: original on top, overlay on bottom
            stacked = np.vstack([original_1024, overlay])
            stacked_frames.append(stacked)
        
        return stacked_frames
    
    def create_combined_video(self,
                            feature_files: List[Path],
                            video_fps: int,
                            mp4: bool,
                            gif: bool,
                            video_original_images: bool,
                            video_umap_visualization: bool,
                            video_pca_visualization: bool,
                            independent_videos: bool,
                            video_stacked_visualization: bool) -> List[Path]:
        """Create combined video from multiple batches as continuous sequence."""
        
        video_files = []
        
        # First, load all features and combine them
        all_features = []
        all_metadata = []
        all_original_frames = []
        
        for feature_file in sorted(feature_files):
            features, metadata = self.load_features(feature_file)
            all_features.append(features)
            all_metadata.append(metadata)
            
            # Collect original frame paths
            if video_original_images or video_stacked_visualization:
                frame_paths = self.get_original_frame_paths(metadata)
                all_original_frames.extend(frame_paths)
        
        # Stack all features for unified UMAP
        combined_features = torch.cat(all_features, dim=0)  # [N*4608, 1408]
        
        # Get overall frame range
        first_meta = all_metadata[0]
        last_meta = all_metadata[-1]
        start_frame = first_meta.get('start_frame', 0)
        end_frame = last_meta.get('end_frame', len(all_original_frames)-1)
        
        self.logger.info(f"Creating combined video for frames {start_frame:04d}-{end_frame:04d}")
        
        # Prepare frames for each visualization type
        viz_frames = {}
        
        # Original frames
        if video_original_images:
            viz_frames['original'] = []
            for frame_path in all_original_frames:
                if frame_path.exists():
                    img = cv2.imread(str(frame_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    viz_frames['original'].append(img)
        
        # UMAP visualization with unified reduction across all batches
        if video_umap_visualization or video_stacked_visualization:
            self.logger.info("Computing unified UMAP across all batches...")
            rgb_values, reducer = self.compute_umap_visualization(combined_features)
            
            # Reshape back to patches per batch
            rgb_patches_all = rgb_values.reshape(-1, 24, 24, 3)  # [N*8, 24, 24, 3]
            
            if video_umap_visualization:
                viz_frames['umap'] = []
                for t in range(rgb_patches_all.shape[0]):
                    frame = rgb_patches_all[t]
                    frame_large = cv2.resize(frame, (480, 480), interpolation=cv2.INTER_NEAREST)
                    # Each temporal token represents 2 frames
                    viz_frames['umap'].extend([frame_large, frame_large])
        
        # PCA visualization
        if video_pca_visualization:
            viz_frames['pca'] = []
            for features in all_features:
                pca_values = self.compute_pca_visualization(features)
                for t in range(8):
                    frame = pca_values[t]
                    rgb_frame = self.apply_turbo_colormap(frame)
                    frame_large = cv2.resize(rgb_frame, (480, 480), interpolation=cv2.INTER_NEAREST)
                    viz_frames['pca'].extend([frame_large, frame_large])
        
        # Create output filename base
        base_name = f"frames_{start_frame:04d}_{end_frame:04d}"
        
        # Save independent videos
        if independent_videos and viz_frames:
            for viz_type, frames in viz_frames.items():
                if not frames:
                    continue
                
                video_base = f"{base_name}_{viz_type}"
                
                if mp4:
                    mp4_path = self.video_dir / f"{video_base}.mp4"
                    self.save_mp4(frames, mp4_path, video_fps)
                    video_files.append(mp4_path)
                
                if gif:
                    gif_path = self.video_dir / f"{video_base}.gif"
                    self.save_gif(frames, gif_path, video_fps)
                    video_files.append(gif_path)
        
        # Create stacked visualization
        if video_stacked_visualization:
            stacked_frames = []
            
            for i, frame_path in enumerate(all_original_frames):
                if not frame_path.exists():
                    continue
                
                # Load and resize original
                original = cv2.imread(str(frame_path))
                original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                original_1024 = cv2.resize(original, (1024, 1024))
                
                # Get corresponding UMAP patch
                batch_idx = i // 16  # Which batch this frame belongs to
                local_idx = i % 16   # Position within the batch
                t = batch_idx * 8 + local_idx // 2  # Overall temporal token index
                
                if t < rgb_patches_all.shape[0]:
                    umap_patch = rgb_patches_all[t]
                    umap_1024 = cv2.resize(umap_patch, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                    
                    # Create overlay
                    overlay = cv2.addWeighted(original_1024, 0.5, umap_1024, 0.5, 0)
                    
                    # Stack vertically
                    stacked = np.vstack([original_1024, overlay])
                    stacked_frames.append(stacked)
            
            if stacked_frames:
                stacked_base = f"{base_name}_stacked"
                
                if mp4:
                    mp4_path = self.video_dir / f"{stacked_base}.mp4"
                    self.save_mp4(stacked_frames, mp4_path, video_fps)
                    video_files.append(mp4_path)
                
                if gif:
                    gif_path = self.video_dir / f"{stacked_base}.gif"
                    self.save_gif(stacked_frames, gif_path, video_fps)
                    video_files.append(gif_path)
        
        return video_files
    
    def save_mp4(self, frames: List[np.ndarray], output_path: Path, fps: int):
        """Save frames as MP4 video."""
        if not frames:
            return
        
        height, width = frames[0].shape[:2]
        
        # Use OpenCV VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        self.logger.info(f"Saved MP4: {output_path}")
    
    def save_gif(self, frames: List[np.ndarray], output_path: Path, fps: int):
        """Save frames as GIF."""
        if not frames:
            return
        
        # Convert to PIL Images
        pil_frames = [Image.fromarray(frame) for frame in frames]
        
        # Save as GIF
        duration = int(1000 / fps)  # Duration in milliseconds
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=0
        )
        
        self.logger.info(f"Saved GIF: {output_path}")
    
    def package_outputs(self,
                       umap_visualization: bool = False,
                       pca_visualization: bool = False,
                       original_images: bool = False,
                       embeddings: bool = True,
                       include_videos: bool = False):
        """Package outputs into final structure."""
        
        self.logger.info("Packaging outputs...")
        
        # Clear package directory
        if self.package_dir.exists():
            shutil.rmtree(self.package_dir)
        self.package_dir.mkdir(parents=True)
        
        # Copy embeddings if requested
        if embeddings:
            embeddings_dir = self.package_dir / "embeddings"
            embeddings_dir.mkdir()
            for feature_file in self.feature_files:
                shutil.copy2(feature_file, embeddings_dir)
        
        # Copy visualizations
        if any([umap_visualization, pca_visualization, original_images]):
            viz_package_dir = self.package_dir / "visualizations"
            if self.viz_dir.exists():
                shutil.copytree(self.viz_dir, viz_package_dir)
        
        # Copy videos
        if include_videos and self.video_dir.exists() and list(self.video_dir.glob("*")):
            videos_package_dir = self.package_dir / "videos"
            shutil.copytree(self.video_dir, videos_package_dir)
        
        # Create metadata file
        metadata = {
            'processing_date': datetime.now().isoformat(),
            'frame_range': f"{self.first_frame} to {self.last_frame}",
            'stride': self.stride,
            'batch_size': self.batch_size,
            'options': {
                'umap_visualization': umap_visualization,
                'pca_visualization': pca_visualization,
                'original_images': original_images,
                'embeddings': embeddings,
                'videos_included': include_videos
            }
        }
        
        with open(self.package_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create zip file
        zip_name = f"vjepa2_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        zip_path = self.output_dir / f"{zip_name}.zip"
        
        shutil.make_archive(str(self.output_dir / zip_name), 'zip', self.package_dir)
        
        self.logger.info(f"Created package: {zip_path}")
        return zip_path
    
    def run(self,
            umap_visualization: bool = False,
            pca_visualization: bool = False,
            original_images: bool = False,
            embeddings: bool = True,
            video_animation: bool = False,
            video_fps: int = 6,
            mp4: bool = False,
            gif: bool = False,
            video_original_images: bool = True,
            video_umap_visualization: bool = False,
            video_pca_visualization: bool = False,
            independent_videos: bool = True,
            video_stacked_visualization: bool = False):
        """Run complete pipeline."""
        
        self.logger.info("Starting V-JEPA 2 processing pipeline")
        
        # Step 1: Extract features
        self.extract_features()
        
        # Step 2: Generate visualizations
        if any([umap_visualization, pca_visualization, original_images]):
            self.logger.info("Generating visualizations...")
            for feature_file in self.feature_files:
                self.save_visualization_frames(
                    feature_file,
                    umap_viz=umap_visualization,
                    pca_viz=pca_visualization,
                    original_images=original_images
                )
        
        # Step 3: Create videos
        if video_animation:
            self.logger.info("Creating video animations...")
            self.create_video(
                self.feature_files,
                video_fps=video_fps,
                mp4=mp4,
                gif=gif,
                video_original_images=video_original_images,
                video_umap_visualization=video_umap_visualization,
                video_pca_visualization=video_pca_visualization,
                independent_videos=independent_videos,
                video_stacked_visualization=video_stacked_visualization
            )
        
        # Step 4: Package outputs
        include_videos = video_animation and (mp4 or gif)
        zip_path = self.package_outputs(
            umap_visualization=umap_visualization,
            pca_visualization=pca_visualization,
            original_images=original_images,
            embeddings=embeddings,
            include_videos=include_videos
        )
        
        self.logger.info(f"Pipeline complete! Output: {zip_path}")
        return zip_path


def main():
    parser = argparse.ArgumentParser(description='V-JEPA 2 end-to-end processing pipeline')
    
    # Required arguments
    parser.add_argument('--frame_dir', type=str, required=True,
                       help='Directory containing input frames')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory for all outputs')
    
    # Frame range
    parser.add_argument('--first_frame', type=int, default=-1,
                       help='First frame to process (-1 for all)')
    parser.add_argument('--last_frame', type=int, default=-1,
                       help='Last frame to process (-1 for all)')
    
    # Processing options
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='GPU device to use')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size for processing')
    parser.add_argument('--stride', type=int, default=16,
                       help='Stride for sliding windows')
    
    # Visualization flags
    parser.add_argument('--umap_visualization', action='store_true',
                       help='Generate UMAP visualizations (default: False)')
    parser.add_argument('--pca_visualization', action='store_true',
                       help='Generate PCA visualizations (default: False)')
    parser.add_argument('--original_images', action='store_true',
                       help='Include original images (default: False)')
    parser.add_argument('--no_embeddings', action='store_true',
                       help='Exclude embeddings from output (default: include)')
    
    # Video options
    parser.add_argument('--video_animation', action='store_true',
                       help='Create video animations (default: False)')
    parser.add_argument('--video_fps', type=int, default=6,
                       help='Video frames per second (default: 6)')
    parser.add_argument('--mp4', action='store_true',
                       help='Output MP4 video (default: False)')
    parser.add_argument('--gif', action='store_true',
                       help='Output GIF animation (default: False)')
    parser.add_argument('--no_video_original_images', action='store_true',
                       help='Exclude original images from video (default: include)')
    parser.add_argument('--video_umap_visualization', action='store_true',
                       help='Include UMAP in video (default: False)')
    parser.add_argument('--video_pca_visualization', action='store_true',
                       help='Include PCA in video (default: False)')
    parser.add_argument('--no_independent_videos', action='store_true',
                       help='Create combined video instead of independent (default: independent)')
    parser.add_argument('--video_stacked_visualization', action='store_true',
                       help='Create stacked video with original and UMAP overlay (default: False)')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = VJEPA2Pipeline(
        frame_dir=args.frame_dir,
        output_dir=args.output_dir,
        first_frame=args.first_frame,
        last_frame=args.last_frame,
        device=args.device,
        batch_size=args.batch_size,
        stride=args.stride
    )
    
    # Run pipeline
    pipeline.run(
        umap_visualization=args.umap_visualization,
        pca_visualization=args.pca_visualization,
        original_images=args.original_images,
        embeddings=not args.no_embeddings,
        video_animation=args.video_animation,
        video_fps=args.video_fps,
        mp4=args.mp4,
        gif=args.gif,
        video_original_images=not args.no_video_original_images,
        video_umap_visualization=args.video_umap_visualization,
        video_pca_visualization=args.video_pca_visualization,
        independent_videos=not args.no_independent_videos,
        video_stacked_visualization=args.video_stacked_visualization
    )


if __name__ == "__main__":
    main()