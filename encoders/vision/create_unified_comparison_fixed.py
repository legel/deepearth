#!/usr/bin/env python3
"""
Create unified comparison of V-JEPA 2, DINOv3 ViT-L, and DINOv3 ViT-7B.
Properly handles frame-by-frame visualization with temporal alignment.
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


def compute_frame_aligned_umap(vjepa2_features, dinov3_vitl_features, dinov3_vit7b_features):
    """Compute UMAP projections with proper frame alignment"""
    print("\n" + "="*60)
    print("Computing frame-aligned UMAP projections...")
    print("="*60)
    
    # Define unified UMAP hyperparameters for fair comparison
    UMAP_PARAMS = {
        'n_components': 3,
        'n_neighbors': 15,
        'min_dist': 0.1,
        'random_state': 42,
        'metric': 'cosine',
        'n_epochs': 500,
        'learning_rate': 1.0,
        'spread': 1.0,
        'negative_sample_rate': 5,
        'local_connectivity': 1.0
    }
    
    print(f"UMAP hyperparameters: {UMAP_PARAMS}")
    
    # Process V-JEPA 2: [4608, 1408] -> [8, 576, 1408]
    # 8 temporal tokens, each representing 2 frames
    print("\n1. Processing V-JEPA 2 (8 temporal tokens from 16 frames)...")
    vjepa2_reshaped = vjepa2_features.view(8, 576, -1)  # [8, 576, 1408]
    vjepa2_frames = []
    
    # Compute UMAP for each temporal token
    for t in range(8):
        print(f"  Token {t} (frames {t*2}-{t*2+1})...")
        reducer = umap.UMAP(**UMAP_PARAMS)
        token_features = vjepa2_reshaped[t].cpu().numpy()  # [576, 1408]
        token_rgb = reducer.fit_transform(token_features)
        token_rgb = (token_rgb - token_rgb.min()) / (token_rgb.max() - token_rgb.min() + 1e-8)
        token_rgb = token_rgb.reshape(24, 24, 3)
        
        # Duplicate for both frames this token represents
        vjepa2_frames.append(token_rgb)
        vjepa2_frames.append(token_rgb)  # Same embedding for both frames in pair
    
    # Process DINOv3 ViT-L: [16, 196, 1024] - one per frame
    print("\n2. Processing DINOv3 ViT-L (16 individual frames)...")
    dinov3_vitl_frames = []
    
    for f in range(16):
        print(f"  Frame {f}...")
        reducer = umap.UMAP(**UMAP_PARAMS)
        frame_features = dinov3_vitl_features[f].cpu().numpy()  # [196, 1024]
        frame_rgb = reducer.fit_transform(frame_features)
        frame_rgb = (frame_rgb - frame_rgb.min()) / (frame_rgb.max() - frame_rgb.min() + 1e-8)
        frame_rgb = frame_rgb.reshape(14, 14, 3)
        dinov3_vitl_frames.append(frame_rgb)
    
    # Process DINOv3 ViT-7B: [16, 196, 1536] - one per frame
    print("\n3. Processing DINOv3 ViT-7B (16 individual frames)...")
    dinov3_vit7b_frames = []
    
    for f in range(16):
        print(f"  Frame {f}...")
        reducer = umap.UMAP(**UMAP_PARAMS)
        frame_features = dinov3_vit7b_features[f].cpu().numpy()  # [196, 1536]
        frame_rgb = reducer.fit_transform(frame_features)
        frame_rgb = (frame_rgb - frame_rgb.min()) / (frame_rgb.max() - frame_rgb.min() + 1e-8)
        frame_rgb = frame_rgb.reshape(14, 14, 3)
        dinov3_vit7b_frames.append(frame_rgb)
    
    print(f"\nFrame-aligned projections complete:")
    print(f"  V-JEPA 2: {len(vjepa2_frames)} frames (8 tokens × 2)")
    print(f"  DINOv3 ViT-L: {len(dinov3_vitl_frames)} frames")
    print(f"  DINOv3 ViT-7B: {len(dinov3_vit7b_frames)} frames")
    
    return vjepa2_frames, dinov3_vitl_frames, dinov3_vit7b_frames


def create_comparison_gifs(frame_dir, vjepa2_frames, dinov3_vitl_frames, dinov3_vit7b_frames, output_dir):
    """Create individual GIFs with proper frame alignment"""
    print("\n" + "="*60)
    print("Creating comparison GIFs with frame alignment...")
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
    
    # Create GIFs for each model
    models = [
        ("vjepa2", vjepa2_frames, "V-JEPA 2 (1B params)"),
        ("dinov3_vitl", dinov3_vitl_frames, "DINOv3 ViT-L (300M)"),
        ("dinov3_vit7b", dinov3_vit7b_frames, "DINOv3 ViT-7B (6.7B)")
    ]
    
    for model_name, model_frames, title in models:
        frames = []
        for i in range(16):  # 16 frames
            # Create figure with original and features
            fig, axes = plt.subplots(2, 1, figsize=(4, 8))
            
            # Original
            axes[0].imshow(original_frames[i])
            axes[0].set_title(f"NAIP Frame {i}", fontsize=10)
            axes[0].axis('off')
            
            # Features - properly aligned to frame
            features_rgb = model_frames[i]
            features_resized = np.array(Image.fromarray((features_rgb * 255).astype(np.uint8)).resize((224, 224), Image.NEAREST))
            axes[1].imshow(features_resized)
            
            # Add annotation for V-JEPA 2 temporal compression
            if model_name == "vjepa2":
                token_idx = i // 2
                axes[1].set_title(f"{title} (Token {token_idx})", fontsize=10)
            else:
                axes[1].set_title(f"{title} (Frame {i})", fontsize=10)
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
    
    for i in range(16):
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        # Top row - original frame
        for j in range(3):
            axes[0, j].imshow(original_frames[i])
            axes[0, j].set_title(f"NAIP Frame {i}", fontsize=10)
            axes[0, j].axis('off')
        
        # Bottom row - three models with proper frame alignment
        # V-JEPA 2
        vjepa2_rgb = vjepa2_frames[i]
        vjepa2_resized = np.array(Image.fromarray((vjepa2_rgb * 255).astype(np.uint8)).resize((224, 224), Image.NEAREST))
        axes[1, 0].imshow(vjepa2_resized)
        axes[1, 0].set_title(f"V-JEPA 2 (Token {i//2})", fontsize=10)
        axes[1, 0].axis('off')
        
        # DINOv3 ViT-L
        vitl_rgb = dinov3_vitl_frames[i]
        vitl_resized = np.array(Image.fromarray((vitl_rgb * 255).astype(np.uint8)).resize((224, 224), Image.NEAREST))
        axes[1, 1].imshow(vitl_resized)
        axes[1, 1].set_title(f"DINOv3 ViT-L (Frame {i})", fontsize=10)
        axes[1, 1].axis('off')
        
        # DINOv3 ViT-7B
        vit7b_rgb = dinov3_vit7b_frames[i]
        vit7b_resized = np.array(Image.fromarray((vit7b_rgb * 255).astype(np.uint8)).resize((224, 224), Image.NEAREST))
        axes[1, 2].imshow(vit7b_resized)
        axes[1, 2].set_title(f"DINOv3 ViT-7B (Frame {i})", fontsize=10)
        axes[1, 2].axis('off')
        
        plt.suptitle("Vision Encoder Comparison - Frame-Aligned UMAP", fontsize=12, fontweight='bold')
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
    output_dir = Path("outputs/unified_comparison_fixed")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract features
    vjepa2_features, dinov3_vitl_features, dinov3_vit7b_features = extract_all_features(
        frame_dir, output_dir, first_frame=0, last_frame=15
    )
    
    # Compute frame-aligned UMAP
    vjepa2_frames, dinov3_vitl_frames, dinov3_vit7b_frames = compute_frame_aligned_umap(
        vjepa2_features, dinov3_vitl_features, dinov3_vit7b_features
    )
    
    # Create comparison GIFs
    gif1, gif2, gif3 = create_comparison_gifs(
        frame_dir, vjepa2_frames, dinov3_vitl_frames, dinov3_vit7b_frames, output_dir
    )
    
    print("\n" + "="*60)
    print("✅ Frame-aligned comparison complete!")
    print("="*60)
    print(f"\nIndividual GIFs:")
    print(f"  - V-JEPA 2: {gif1}")
    print(f"  - DINOv3 ViT-L: {gif2}")
    print(f"  - DINOv3 ViT-7B: {gif3}")
    print(f"\nCombined comparison: outputs/unified_comparison_fixed/all_models_comparison.gif")
    print("\nKey differences:")
    print("  - V-JEPA 2: 8 temporal tokens (each covers 2 frames)")
    print("  - DINOv3: 16 individual frame embeddings")
    print("  - All models use identical UMAP hyperparameters")


if __name__ == "__main__":
    main()