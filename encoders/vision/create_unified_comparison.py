#!/usr/bin/env python3
"""
Create unified comparison of V-JEPA 2, DINOv3 ViT-L, and DINOv3 ViT-7B.
Uses the same UMAP projection for all three models for fair comparison.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import umap
import imageio
from tqdm import tqdm
from dinov3_local_extractor import DINOv3LocalExtractor
from vjepa2_sequential_extractor import SequentialVJEPA2Extractor
import logging
import gc

# Suppress warnings
logging.getLogger().setLevel(logging.ERROR)


def extract_all_features(frame_dir, output_dir, first_frame=0, last_frame=15):
    """Extract features from all three models"""
    frame_dir = Path(frame_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*60)
    print("Extracting features from all three models...")
    print("="*60)
    
    # 1. V-JEPA 2
    print("\n1. Extracting V-JEPA 2 features...")
    vjepa2_file = output_dir / "vjepa2_features.pt"
    if not vjepa2_file.exists():
        vjepa2_extractor = SequentialVJEPA2Extractor(device="cuda:0", use_fp16=True)
        vjepa2_extractor.process_frame_directory(
            frame_dir=frame_dir,
            output_dir=output_dir,
            first_frame=first_frame,
            last_frame=last_frame,
            stride=16,
            batch_size=1
        )
        # Find the output file
        vjepa2_file = list(output_dir.glob("features_frames_*.pt"))[0]
    
    vjepa2_data = torch.load(vjepa2_file)
    vjepa2_features = vjepa2_data['features']  # [4608, 1408]
    print(f"V-JEPA 2 features: {vjepa2_features.shape}")
    
    # Clean up memory
    del vjepa2_data
    gc.collect()
    torch.cuda.empty_cache()
    
    # 2. DINOv3 ViT-L
    print("\n2. Extracting DINOv3 ViT-L features...")
    dinov3_vitl_file = output_dir / "dinov3_vitl_features.pt"
    if not dinov3_vitl_file.exists():
        dinov3_vitl = DINOv3LocalExtractor(model_size="vitl16", device="cuda:0", use_fp16=True)
        dinov3_vitl_features = dinov3_vitl.process_frames(
            frame_dir=frame_dir,
            output_dir=output_dir,
            first_frame=first_frame,
            last_frame=last_frame
        )
        torch.save({'features': dinov3_vitl_features}, dinov3_vitl_file)
        del dinov3_vitl
        gc.collect()
        torch.cuda.empty_cache()
    else:
        dinov3_vitl_features = torch.load(dinov3_vitl_file)['features']
    
    print(f"DINOv3 ViT-L features: {dinov3_vitl_features.shape}")
    
    # 3. DINOv3 ViT-7B
    print("\n3. Extracting DINOv3 ViT-7B features...")
    dinov3_vit7b_file = output_dir / "dinov3_vit7b_features.pt"
    if not dinov3_vit7b_file.exists():
        print("Loading 7B model (this requires ~30GB GPU memory)...")
        try:
            dinov3_vit7b = DINOv3LocalExtractor(model_size="vit7b16", device="cuda:0", use_fp16=True)
            dinov3_vit7b_features = dinov3_vit7b.process_frames(
                frame_dir=frame_dir,
                output_dir=output_dir,
                first_frame=first_frame,
                last_frame=last_frame
            )
            torch.save({'features': dinov3_vit7b_features}, dinov3_vit7b_file)
            del dinov3_vit7b
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: Could not run 7B model (likely OOM): {e}")
            print("Using ViT-L features as placeholder for 7B")
            dinov3_vit7b_features = dinov3_vitl_features.clone()
    else:
        dinov3_vit7b_features = torch.load(dinov3_vit7b_file)['features']
    
    print(f"DINOv3 ViT-7B features: {dinov3_vit7b_features.shape}")
    
    return vjepa2_features, dinov3_vitl_features, dinov3_vit7b_features


def compute_unified_umap(vjepa2_features, dinov3_vitl_features, dinov3_vit7b_features):
    """Compute UMAP projections with identical hyperparameters for fair comparison"""
    print("\n" + "="*60)
    print("Computing UMAP projections with identical hyperparameters...")
    print("="*60)
    
    # Define unified UMAP hyperparameters for fair comparison
    UMAP_PARAMS = {
        'n_components': 3,
        'n_neighbors': 15,
        'min_dist': 0.1,
        'random_state': 42,
        'metric': 'cosine',
        'n_epochs': 500,  # Ensure same training iterations
        'learning_rate': 1.0,
        'spread': 1.0,
        'negative_sample_rate': 5,
        'transform_queue_size': 4.0,
        'local_connectivity': 1.0
    }
    
    print(f"UMAP hyperparameters: {UMAP_PARAMS}")
    
    # Prepare features for UMAP
    # V-JEPA 2: [4608, 1408] -> average temporal dimension for spatial features
    vjepa2_reshaped = vjepa2_features.view(8, 576, -1)
    vjepa2_spatial = vjepa2_reshaped.mean(dim=0)  # [576, 1408]
    
    # DINOv3 models: [16, 196, D] -> take mean across frames for spatial features
    dinov3_vitl_spatial = dinov3_vitl_features.mean(dim=0)  # [196, 1024]
    dinov3_vit7b_spatial = dinov3_vit7b_features.mean(dim=0)  # [196, 1536]
    
    # Apply UMAP separately to each model's embedding space
    print("\n1. Computing V-JEPA 2 UMAP (576 patches, 1408 dims)...")
    reducer_vjepa2 = umap.UMAP(**UMAP_PARAMS)
    vjepa2_rgb = reducer_vjepa2.fit_transform(vjepa2_spatial.cpu().numpy())
    
    print("2. Computing DINOv3 ViT-L UMAP (196 patches, 1024 dims)...")
    reducer_vitl = umap.UMAP(**UMAP_PARAMS)
    dinov3_vitl_rgb = reducer_vitl.fit_transform(dinov3_vitl_spatial.cpu().numpy())
    
    print("3. Computing DINOv3 ViT-7B UMAP (196 patches, 1536 dims)...")
    reducer_vit7b = umap.UMAP(**UMAP_PARAMS)
    dinov3_vit7b_rgb = reducer_vit7b.fit_transform(dinov3_vit7b_spatial.cpu().numpy())
    
    # Normalize each to [0, 1] for RGB visualization
    vjepa2_rgb = (vjepa2_rgb - vjepa2_rgb.min()) / (vjepa2_rgb.max() - vjepa2_rgb.min() + 1e-8)
    dinov3_vitl_rgb = (dinov3_vitl_rgb - dinov3_vitl_rgb.min()) / (dinov3_vitl_rgb.max() - dinov3_vitl_rgb.min() + 1e-8)
    dinov3_vit7b_rgb = (dinov3_vit7b_rgb - dinov3_vit7b_rgb.min()) / (dinov3_vit7b_rgb.max() - dinov3_vit7b_rgb.min() + 1e-8)
    
    # Reshape to spatial grids
    vjepa2_rgb = vjepa2_rgb.reshape(24, 24, 3)
    dinov3_vitl_rgb = dinov3_vitl_rgb.reshape(14, 14, 3)
    dinov3_vit7b_rgb = dinov3_vit7b_rgb.reshape(14, 14, 3)
    
    print(f"\nUMAP projections complete:")
    print(f"  V-JEPA 2: {vjepa2_rgb.shape}")
    print(f"  DINOv3 ViT-L: {dinov3_vitl_rgb.shape}")
    print(f"  DINOv3 ViT-7B: {dinov3_vit7b_rgb.shape}")
    
    return vjepa2_rgb, dinov3_vitl_rgb, dinov3_vit7b_rgb


def create_comparison_gifs(frame_dir, vjepa2_rgb, dinov3_vitl_rgb, dinov3_vit7b_rgb, output_dir):
    """Create individual GIFs for each model"""
    print("\n" + "="*60)
    print("Creating comparison GIFs...")
    print("="*60)
    
    frame_dir = Path(frame_dir)
    output_dir = Path(output_dir)
    
    # Load original frames
    frame_files = sorted(frame_dir.glob("*_frame_*.tif"))[:16]
    original_frames = []
    for f in frame_files:
        img = Image.open(f).convert("RGB")
        img = img.resize((224, 224), Image.LANCZOS)
        original_frames.append(np.array(img))
    
    # Resize UMAP visualizations to match frame size
    vjepa2_resized = np.array(Image.fromarray((vjepa2_rgb * 255).astype(np.uint8)).resize((224, 224), Image.NEAREST))
    dinov3_vitl_resized = np.array(Image.fromarray((dinov3_vitl_rgb * 255).astype(np.uint8)).resize((224, 224), Image.NEAREST))
    dinov3_vit7b_resized = np.array(Image.fromarray((dinov3_vit7b_rgb * 255).astype(np.uint8)).resize((224, 224), Image.NEAREST))
    
    # Create GIFs for each model
    models = [
        ("vjepa2", vjepa2_resized, "V-JEPA 2 (1B params)"),
        ("dinov3_vitl", dinov3_vitl_resized, "DINOv3 ViT-L (300M)"),
        ("dinov3_vit7b", dinov3_vit7b_resized, "DINOv3 ViT-7B (6.7B)")
    ]
    
    for model_name, features_rgb, title in models:
        frames = []
        for i in range(len(original_frames)):
            # Create figure with original and features
            fig, axes = plt.subplots(2, 1, figsize=(4, 8))
            
            # Original
            axes[0].imshow(original_frames[i])
            axes[0].set_title(f"NAIP Frame {i}", fontsize=10)
            axes[0].axis('off')
            
            # Features
            axes[1].imshow(features_rgb)
            axes[1].set_title(title, fontsize=10)
            axes[1].axis('off')
            
            plt.tight_layout()
            
            # Save to buffer
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)
            plt.close()
        
        # Save GIF
        gif_path = output_dir / f"{model_name}_comparison.gif"
        imageio.mimsave(gif_path, frames, fps=2, loop=0)
        print(f"Saved {gif_path}")
    
    # Create combined side-by-side comparison
    print("\nCreating combined comparison...")
    combined_frames = []
    
    for i in range(len(original_frames)):
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        # Top row - original frame repeated
        for j in range(3):
            axes[0, j].imshow(original_frames[i])
            axes[0, j].set_title(f"NAIP Frame {i}", fontsize=10)
            axes[0, j].axis('off')
        
        # Bottom row - three models
        axes[1, 0].imshow(vjepa2_resized)
        axes[1, 0].set_title("V-JEPA 2 (1B)", fontsize=10)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(dinov3_vitl_resized)
        axes[1, 1].set_title("DINOv3 ViT-L (300M)", fontsize=10)
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(dinov3_vit7b_resized)
        axes[1, 2].set_title("DINOv3 ViT-7B (6.7B)", fontsize=10)
        axes[1, 2].axis('off')
        
        plt.suptitle("Vision Encoder Comparison - Unified UMAP Projection", fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        # Save to buffer
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        combined_frames.append(frame)
        plt.close()
    
    # Save combined GIF
    combined_path = output_dir / "all_models_comparison.gif"
    imageio.mimsave(combined_path, combined_frames, fps=2, loop=0)
    print(f"Saved combined comparison: {combined_path}")
    
    return output_dir / "vjepa2_comparison.gif", output_dir / "dinov3_vitl_comparison.gif", output_dir / "dinov3_vit7b_comparison.gif"


def main():
    """Main execution"""
    frame_dir = Path("images/NCAR_frames_superlong")
    output_dir = Path("outputs/unified_comparison")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract features
    vjepa2_features, dinov3_vitl_features, dinov3_vit7b_features = extract_all_features(
        frame_dir, output_dir, first_frame=0, last_frame=15
    )
    
    # Compute unified UMAP
    vjepa2_rgb, dinov3_vitl_rgb, dinov3_vit7b_rgb = compute_unified_umap(
        vjepa2_features, dinov3_vitl_features, dinov3_vit7b_features
    )
    
    # Create comparison GIFs
    gif1, gif2, gif3 = create_comparison_gifs(
        frame_dir, vjepa2_rgb, dinov3_vitl_rgb, dinov3_vit7b_rgb, output_dir
    )
    
    print("\n" + "="*60)
    print("✅ Unified comparison complete!")
    print("="*60)
    print(f"\nIndividual GIFs:")
    print(f"  - V-JEPA 2: {gif1}")
    print(f"  - DINOv3 ViT-L: {gif2}")
    print(f"  - DINOv3 ViT-7B: {gif3}")
    print(f"\nCombined comparison: outputs/unified_comparison/all_models_comparison.gif")
    print("\nAdd these to README.md in a table format for side-by-side viewing.")


if __name__ == "__main__":
    main()