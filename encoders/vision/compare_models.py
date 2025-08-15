#!/usr/bin/env python3
"""
Compare V-JEPA 2 and DINOv3 features on the same NAIP imagery.
Creates side-by-side visualizations using UMAP reduction.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import umap
from sklearn.decomposition import PCA
import imageio
from tqdm import tqdm


def load_vjepa2_features(path: Path):
    """Load V-JEPA 2 features"""
    data = torch.load(path)
    if 'features' in data:
        features = data['features']
    else:
        # Try to find the right file
        pattern = "vjepa2_window_*_frames_0000_0015.pt"
        files = list(path.parent.glob(pattern))
        if files:
            data = torch.load(files[0])
            features = data['features']
        else:
            raise FileNotFoundError(f"No V-JEPA 2 features found in {path.parent}")
    
    print(f"V-JEPA 2 features shape: {features.shape}")
    return features


def load_dinov3_features(path: Path):
    """Load DINOv3 features"""
    data = torch.load(path)
    features = data['features']
    print(f"DINOv3 features shape: {features.shape}")
    return features


def reduce_features_umap(features, n_components=3):
    """Reduce features to RGB using UMAP"""
    # Flatten if needed
    if features.dim() == 3:
        # [frames, patches, dims] -> [frames*patches, dims]
        features_flat = features.reshape(-1, features.shape[-1])
    else:
        features_flat = features
    
    # UMAP reduction
    print(f"Running UMAP on {features_flat.shape[0]} samples...")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        metric='cosine'
    )
    
    reduced = reducer.fit_transform(features_flat.cpu().numpy())
    
    # Normalize to [0, 1]
    reduced = (reduced - reduced.min()) / (reduced.max() - reduced.min())
    
    return reduced


def create_comparison_visualization(vjepa2_features, dinov3_features, frame_dir, output_dir):
    """Create side-by-side comparison of V-JEPA 2 and DINOv3"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load original frames
    frame_files = sorted(Path(frame_dir).glob("*_frame_*.tif"))[:16]
    original_frames = []
    for f in frame_files:
        img = Image.open(f).convert("RGB")
        img = img.resize((224, 224), Image.LANCZOS)
        original_frames.append(np.array(img))
    
    # Reduce features for both models
    print("\nReducing V-JEPA 2 features...")
    # V-JEPA 2: [4608, 1408] -> 576 spatial * 8 temporal
    if vjepa2_features.dim() == 2 and vjepa2_features.shape[0] == 4608:
        # Single window with 4608 patches (576 spatial * 8 temporal)
        # Reshape to [8, 576, 1408] for temporal frames
        vjepa2_reshaped = vjepa2_features.view(8, 576, -1)
        # Take mean across temporal dimension for visualization
        vjepa2_spatial = vjepa2_reshaped.mean(dim=0)  # [576, 1408]
        # Reduce
        vjepa2_reduced = reduce_features_umap(vjepa2_spatial)
        # Reshape to spatial grid (24x24)
        vjepa2_rgb = vjepa2_reduced.reshape(24, 24, 3)
        # Resize to match image size
        vjepa2_rgb_resized = np.array(Image.fromarray((vjepa2_rgb * 255).astype(np.uint8)).resize((224, 224), Image.NEAREST))
    else:
        # Multiple frames
        vjepa2_reduced = reduce_features_umap(vjepa2_features)
        vjepa2_rgb = vjepa2_reduced.reshape(-1, 14, 14, 3)
    
    print("\nReducing DINOv3 features...")
    # DINOv3: [16, 196, 1024]
    dinov3_reduced = reduce_features_umap(dinov3_features)
    dinov3_rgb = dinov3_reduced.reshape(16, 14, 14, 3)
    
    # Create comparison frames
    comparison_frames = []
    
    for i in range(min(len(original_frames), dinov3_rgb.shape[0])):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original
        axes[0].imshow(original_frames[i])
        axes[0].set_title(f"Original Frame {i}", fontsize=12)
        axes[0].axis('off')
        
        # V-JEPA 2
        if vjepa2_features.dim() == 2 and vjepa2_features.shape[0] == 4608:
            # Use the same spatial features for all frames
            axes[1].imshow(vjepa2_rgb_resized)
        else:
            vjepa2_frame = np.array(Image.fromarray((vjepa2_rgb[i] * 255).astype(np.uint8)).resize((224, 224), Image.NEAREST))
            axes[1].imshow(vjepa2_frame)
        axes[1].set_title("V-JEPA 2 Features", fontsize=12)
        axes[1].axis('off')
        
        # DINOv3
        dinov3_frame = np.array(Image.fromarray((dinov3_rgb[i] * 255).astype(np.uint8)).resize((224, 224), Image.NEAREST))
        axes[2].imshow(dinov3_frame)
        axes[2].set_title("DINOv3 Features (SAT-493M)", fontsize=12)
        axes[2].axis('off')
        
        plt.suptitle(f"NAIP Aerial Imagery - Frame {i}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save frame
        frame_path = output_dir / f"comparison_frame_{i:03d}.png"
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        comparison_frames.append(imageio.imread(frame_path))
    
    # Create GIF
    gif_path = output_dir / "vjepa2_vs_dinov3_comparison.gif"
    imageio.mimsave(gif_path, comparison_frames, fps=2, loop=0)
    print(f"\nComparison GIF saved to {gif_path}")
    
    # Create a summary figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Top row - Frame 0
    axes[0, 0].imshow(original_frames[0])
    axes[0, 0].set_title("Original NAIP Frame 0", fontsize=12)
    axes[0, 0].axis('off')
    
    if vjepa2_features.dim() == 2 and vjepa2_features.shape[0] == 4608:
        axes[0, 1].imshow(vjepa2_rgb_resized)
    else:
        axes[0, 1].imshow(np.array(Image.fromarray((vjepa2_rgb[0] * 255).astype(np.uint8)).resize((224, 224), Image.NEAREST)))
    axes[0, 1].set_title("V-JEPA 2 Features", fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(np.array(Image.fromarray((dinov3_rgb[0] * 255).astype(np.uint8)).resize((224, 224), Image.NEAREST)))
    axes[0, 2].set_title("DINOv3 Features (SAT-493M)", fontsize=12)
    axes[0, 2].axis('off')
    
    # Bottom row - Frame 8
    axes[1, 0].imshow(original_frames[8])
    axes[1, 0].set_title("Original NAIP Frame 8", fontsize=12)
    axes[1, 0].axis('off')
    
    if vjepa2_features.dim() == 2 and vjepa2_features.shape[0] == 4608:
        axes[1, 1].imshow(vjepa2_rgb_resized)
    else:
        axes[1, 1].imshow(np.array(Image.fromarray((vjepa2_rgb[8] * 255).astype(np.uint8)).resize((224, 224), Image.NEAREST)))
    axes[1, 1].set_title("V-JEPA 2 Features", fontsize=12)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(np.array(Image.fromarray((dinov3_rgb[8] * 255).astype(np.uint8)).resize((224, 224), Image.NEAREST)))
    axes[1, 2].set_title("DINOv3 Features (SAT-493M)", fontsize=12)
    axes[1, 2].axis('off')
    
    plt.suptitle("V-JEPA 2 vs DINOv3: NAIP Aerial Imagery Feature Comparison", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    summary_path = output_dir / "model_comparison_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Summary figure saved to {summary_path}")
    
    return gif_path, summary_path


def main():
    # Paths
    vjepa2_path = Path("outputs/vjepa2_test/features_frames_0000_0015.pt")
    dinov3_path = Path("outputs/dinov3_test/dinov3_vitl16_frames_0000_0015.pt")
    frame_dir = Path("images/NCAR_frames_superlong")
    output_dir = Path("outputs/model_comparison")
    
    # Load features
    print("Loading features...")
    vjepa2_features = load_vjepa2_features(vjepa2_path)
    dinov3_features = load_dinov3_features(dinov3_path)
    
    # Create visualization
    gif_path, summary_path = create_comparison_visualization(
        vjepa2_features, 
        dinov3_features,
        frame_dir,
        output_dir
    )
    
    print("\n" + "="*50)
    print("Model Comparison Complete!")
    print("="*50)
    print(f"\nV-JEPA 2:")
    print(f"  - Model: facebook/vjepa2-vitg-fpc64-384")
    print(f"  - Features: {vjepa2_features.shape}")
    print(f"  - Pretrained on: Video data (self-supervised)")
    print(f"\nDINOv3:")
    print(f"  - Model: dinov3_vitl16")  
    print(f"  - Features: {dinov3_features.shape}")
    print(f"  - Pretrained on: SAT-493M (satellite imagery)")
    print(f"\nOutputs:")
    print(f"  - Comparison GIF: {gif_path}")
    print(f"  - Summary figure: {summary_path}")


if __name__ == "__main__":
    main()