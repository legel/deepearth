#!/usr/bin/env python3
"""
UMAP visualization for V-JEPA 2 embeddings.
Reduces patch embeddings to RGB colors and creates temporal frame visualizations.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import umap
from typing import Dict, Tuple, Optional, List
import argparse
from tqdm import tqdm


class EmbeddingVisualizer:
    """Visualize V-JEPA 2 embeddings using UMAP dimensionality reduction."""
    
    def __init__(self, n_neighbors: int = 15, min_dist: float = 0.1, random_state: int = 42):
        """
        Initialize the visualizer with UMAP parameters.
        
        Args:
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter
            random_state: Random seed for reproducibility
        """
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        self.reducer = None
        
    def load_embeddings(self, pt_path: str) -> Tuple[torch.Tensor, Dict, Optional[List[str]]]:
        """
        Load embeddings from .pt file.
        
        Args:
            pt_path: Path to .pt file
            
        Returns:
            features: Tensor of shape [4608, 1408]
            metadata: Metadata dictionary
            image_paths: List of paths to original images if available
        """
        data = torch.load(pt_path, map_location='cpu', weights_only=False)
        
        # Handle batch format (dictionary of images)
        if isinstance(data, dict) and not 'features' in data:
            # Get first image entry
            first_key = list(data.keys())[0]
            entry = data[first_key]
            features = entry['features']
            metadata = entry.get('metadata', {})
            image_paths = [first_key] if Path(first_key).exists() else None
        else:
            # Handle single image or sequential format
            features = data['features']
            metadata = data.get('metadata', {})
            
            # Check for sequential format with frame files
            if metadata and 'frame_files' in metadata:
                # Sequential format - try multiple possible locations
                possible_dirs = [
                    Path(pt_path).parent.parent / "images" / "NCAR_frames_superlong",
                    Path("/home/lance/deepearth/encoders/vision/images/NCAR_frames_superlong"),
                    Path("images/NCAR_frames_superlong"),
                    Path(pt_path).parent.parent / "images" / "NCAR",
                    Path("/home/lance/deepearth/encoders/vision/images/NCAR"),
                    Path("images/NCAR")
                ]
                
                image_paths = []
                for frame_file in metadata['frame_files']:
                    for base_dir in possible_dirs:
                        frame_path = base_dir / frame_file
                        if frame_path.exists():
                            image_paths.append(str(frame_path))
                            break
                
                if not image_paths:
                    print(f"Warning: Could not find frame files. Tried: {possible_dirs}")
                    image_paths = None
            else:
                # Single image format
                image_path = metadata.get('image_path')
                if not image_path:
                    # Try to find from the keys
                    first_key = list(data.keys())[0] if isinstance(data, dict) else None
                    if first_key and Path(first_key).exists():
                        image_path = first_key
                image_paths = [image_path] if image_path else None
            
        return features, metadata, image_paths
    
    def reshape_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Reshape features to temporal-spatial structure.
        
        Args:
            features: Tensor of shape [4608, 1408]
            
        Returns:
            Tensor of shape [8, 24, 24, 1408]
        """
        # Constants from V-JEPA 2 architecture
        temporal_frames = 8
        spatial_size = 24
        
        # Reshape: [4608, 1408] -> [8, 576, 1408] -> [8, 24, 24, 1408]
        features_3d = features.view(temporal_frames, spatial_size * spatial_size, -1)
        features_4d = features_3d.view(temporal_frames, spatial_size, spatial_size, -1)
        
        return features_4d
    
    def compute_umap_reduction(self, features: torch.Tensor) -> np.ndarray:
        """
        Reduce patch embeddings from 1408 to 3 dimensions using UMAP.
        
        Args:
            features: Tensor of shape [4608, 1408]
            
        Returns:
            Array of shape [4608, 3] with values in [0, 1]
        """
        print("Computing UMAP reduction (1408 -> 3 dimensions)...")
        
        # Convert to numpy
        features_np = features.numpy() if isinstance(features, torch.Tensor) else features
        
        # Initialize UMAP reducer
        self.reducer = umap.UMAP(
            n_components=3,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=self.random_state,
            metric='cosine'
        )
        
        # Fit and transform
        embeddings_3d = self.reducer.fit_transform(features_np)
        
        # Normalize to [0, 1]
        embeddings_3d -= embeddings_3d.min(axis=0)
        embeddings_3d /= embeddings_3d.max(axis=0)
        
        return embeddings_3d
    
    def create_rgb_patches(self, embeddings_3d: np.ndarray) -> np.ndarray:
        """
        Convert 3D embeddings to RGB colors.
        
        Args:
            embeddings_3d: Array of shape [4608, 3] with values in [0, 1]
            
        Returns:
            Array of shape [8, 24, 24, 3] with values in [0, 255]
        """
        # Reshape to temporal-spatial structure
        rgb_patches = embeddings_3d.reshape(8, 24, 24, 3)
        
        # Convert to 0-255 range
        rgb_patches = (rgb_patches * 255).astype(np.uint8)
        
        return rgb_patches
    
    def save_temporal_frames(self, rgb_patches: np.ndarray, output_prefix: str, metadata: Optional[Dict] = None):
        """
        Save each temporal frame as a separate image.
        
        Args:
            rgb_patches: Array of shape [8, 24, 24, 3]
            output_prefix: Prefix for output files
            metadata: Optional metadata with temporal mapping info
        """
        output_path = Path(output_prefix).parent
        output_stem = Path(output_prefix).stem
        
        for t in range(8):
            frame = rgb_patches[t]  # [24, 24, 3]
            img = Image.fromarray(frame, mode='RGB')
            
            # Optionally resize for better visibility
            img_large = img.resize((240, 240), Image.NEAREST)
            
            # Add frame range info if available
            if metadata and 'temporal_mapping' in metadata:
                frame_info = metadata['temporal_mapping'].get(f't{t}', '')
                frame_suffix = f"_t{t}_{frame_info}" if frame_info else f"_t{t}"
            else:
                frame_suffix = f"_t{t}"
            
            output_file = output_path / f"{output_stem}{frame_suffix}.png"
            img_large.save(output_file)
            print(f"Saved: {output_file}")
    
    def create_overlay_sequential(self, rgb_patches: np.ndarray, image_paths: List[str], 
                                 output_prefix: str, metadata: Optional[Dict] = None, alpha: float = 0.5):
        """
        Overlay UMAP patches on corresponding original frames.
        
        Args:
            rgb_patches: Array of shape [8, 24, 24, 3]
            image_paths: List of paths to original frames
            output_prefix: Prefix for output files
            metadata: Optional metadata with temporal mapping
            alpha: Transparency of overlay (0-1)
        """
        if not image_paths:
            print("No original images found for overlay")
            return
            
        output_path = Path(output_prefix).parent
        output_stem = Path(output_prefix).stem
        
        for t in range(8):
            # Get frame patches
            frame = rgb_patches[t]  # [24, 24, 3]
            
            # Determine which original frames correspond to this temporal token
            # Each temporal token represents 2 frames (FPC compression)
            frame_idx_start = t * 2
            frame_idx_end = t * 2 + 1
            
            # Use the first frame of the pair for overlay
            if frame_idx_start < len(image_paths):
                original_path = image_paths[frame_idx_start]
                
                if Path(original_path).exists():
                    # Load original image
                    original = Image.open(original_path).convert('RGBA')
                    orig_width, orig_height = original.size
                    
                    # Create patch image
                    patch_img = Image.fromarray(frame, mode='RGB').convert('RGBA')
                    
                    # Resize patches to match original image
                    patch_resized = patch_img.resize((orig_width, orig_height), Image.NEAREST)
                    
                    # Apply alpha
                    patch_array = np.array(patch_resized)
                    patch_array[:, :, 3] = int(255 * alpha)
                    patch_transparent = Image.fromarray(patch_array, mode='RGBA')
                    
                    # Create composite
                    composite = Image.alpha_composite(original, patch_transparent)
                    
                    # Add frame info if available
                    if metadata and 'temporal_mapping' in metadata:
                        frame_info = metadata['temporal_mapping'].get(f't{t}', '')
                        frame_suffix = f"_t{t}_{frame_info}_overlay" if frame_info else f"_t{t}_overlay"
                    else:
                        frame_suffix = f"_t{t}_overlay"
                    
                    output_file = output_path / f"{output_stem}{frame_suffix}.png"
                    composite.save(output_file)
                    print(f"Saved overlay: {output_file}")
    
    def visualize(self, pt_path: str, output_dir: Optional[str] = None, 
                  create_overlays: bool = True):
        """
        Complete visualization pipeline.
        
        Args:
            pt_path: Path to .pt file
            output_dir: Output directory (defaults to same as input)
            create_overlays: Whether to create overlay images
        """
        # Setup paths
        pt_path = Path(pt_path)
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)
        else:
            output_path = pt_path.parent
        
        output_prefix = output_path / pt_path.stem
        
        # Load embeddings
        print(f"Loading embeddings from: {pt_path}")
        features, metadata, image_paths = self.load_embeddings(str(pt_path))
        print(f"Features shape: {features.shape}")
        
        # Print metadata info if available
        if metadata:
            if 'frame_numbers' in metadata:
                print(f"Frame sequence: {metadata['frame_numbers'][0]} to {metadata['frame_numbers'][-1]}")
            if 'temporal_mapping' in metadata:
                print("Temporal mapping detected (sequential processing)")
        
        # Reshape to temporal-spatial structure
        features_4d = self.reshape_features(features)
        print(f"Reshaped to: {features_4d.shape}")
        
        # Compute UMAP reduction
        embeddings_3d = self.compute_umap_reduction(features)
        
        # Convert to RGB patches
        rgb_patches = self.create_rgb_patches(embeddings_3d)
        print(f"RGB patches shape: {rgb_patches.shape}")
        
        # Save temporal frames
        print("\nSaving temporal frames...")
        self.save_temporal_frames(rgb_patches, str(output_prefix), metadata)
        
        # Create overlays if requested and images available
        if create_overlays and image_paths:
            print("\nCreating overlay visualizations...")
            self.create_overlay_sequential(rgb_patches, image_paths, str(output_prefix), metadata)
        
        print("\nVisualization complete!")
        return rgb_patches


def main():
    parser = argparse.ArgumentParser(description='Visualize V-JEPA 2 embeddings using UMAP')
    parser.add_argument('pt_file', type=str, help='Path to .pt embeddings file')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--n_neighbors', type=int, default=15, help='UMAP n_neighbors')
    parser.add_argument('--min_dist', type=float, default=0.1, help='UMAP min_dist')
    parser.add_argument('--alpha', type=float, default=0.5, help='Overlay transparency')
    parser.add_argument('--no_overlay', action='store_true', help='Skip overlay creation')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = EmbeddingVisualizer(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist
    )
    
    # Run visualization
    visualizer.visualize(
        pt_path=args.pt_file,
        output_dir=args.output_dir,
        create_overlays=not args.no_overlay
    )


if __name__ == "__main__":
    main()