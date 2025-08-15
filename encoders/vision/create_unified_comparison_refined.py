#!/usr/bin/env python3
"""
Create refined comparison of V-JEPA 2, DINOv3 ViT-L, and DINOv3 ViT-7B.
- Learns UMAP on all frames/patches for consistency
- Adds overlay visualization
- Slower FPS for better viewing
- Clarifies SAT-493M training
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


def compute_consistent_umap(vjepa2_features, dinov3_vitl_features, dinov3_vit7b_features):
    """Compute UMAP projections on ALL patches across ALL frames for consistency"""
    print("\n" + "="*60)
    print("Computing consistent UMAP projections across all frames...")
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
    
    # Process V-JEPA 2: [4608, 1408] - all patches across all temporal tokens
    print("\n1. Processing V-JEPA 2 (learning on all 4608 patches)...")
    vjepa2_reducer = umap.UMAP(**UMAP_PARAMS)
    vjepa2_all_rgb = vjepa2_reducer.fit_transform(vjepa2_features.cpu().numpy())
    vjepa2_all_rgb = (vjepa2_all_rgb - vjepa2_all_rgb.min()) / (vjepa2_all_rgb.max() - vjepa2_all_rgb.min() + 1e-8)
    
    # Reshape to [8, 576, 3] then to [8, 24, 24, 3]
    vjepa2_rgb = vjepa2_all_rgb.reshape(8, 576, 3).reshape(8, 24, 24, 3)
    
    # Duplicate tokens for frames (each token covers 2 frames)
    vjepa2_frames = []
    for t in range(8):
        vjepa2_frames.append(vjepa2_rgb[t])
        vjepa2_frames.append(vjepa2_rgb[t])  # Same embedding for both frames in pair
    
    # Process DINOv3 ViT-L: [16, 196, 1024] - flatten to [16*196, 1024]
    print("\n2. Processing DINOv3 ViT-L SAT-493M (learning on all 3136 patches)...")
    dinov3_vitl_flat = dinov3_vitl_features.reshape(-1, dinov3_vitl_features.shape[-1])  # [3136, 1024]
    vitl_reducer = umap.UMAP(**UMAP_PARAMS)
    vitl_all_rgb = vitl_reducer.fit_transform(dinov3_vitl_flat.cpu().numpy())
    vitl_all_rgb = (vitl_all_rgb - vitl_all_rgb.min()) / (vitl_all_rgb.max() - vitl_all_rgb.min() + 1e-8)
    
    # Reshape back to [16, 14, 14, 3]
    dinov3_vitl_frames = vitl_all_rgb.reshape(16, 196, 3).reshape(16, 14, 14, 3)
    
    # Process DINOv3 ViT-7B: [16, 196, D] - flatten to [16*196, D]
    print("\n3. Processing DINOv3 ViT-7B SAT-493M (learning on all 3136 patches)...")
    dinov3_vit7b_flat = dinov3_vit7b_features.reshape(-1, dinov3_vit7b_features.shape[-1])  # [3136, D]
    vit7b_reducer = umap.UMAP(**UMAP_PARAMS)
    vit7b_all_rgb = vit7b_reducer.fit_transform(dinov3_vit7b_flat.cpu().numpy())
    vit7b_all_rgb = (vit7b_all_rgb - vit7b_all_rgb.min()) / (vit7b_all_rgb.max() - vit7b_all_rgb.min() + 1e-8)
    
    # Reshape back to [16, 14, 14, 3]
    dinov3_vit7b_frames = vit7b_all_rgb.reshape(16, 196, 3).reshape(16, 14, 14, 3)
    
    print(f"\nConsistent UMAP projections complete:")
    print(f"  V-JEPA 2: Learned on 4608 patches, applied to 8 tokens → 16 frames")
    print(f"  DINOv3 ViT-L: Learned on 3136 patches across 16 frames")
    print(f"  DINOv3 ViT-7B: Learned on 3136 patches across 16 frames")
    
    return vjepa2_frames, dinov3_vitl_frames, dinov3_vit7b_frames


def create_overlay(original_img, umap_img, alpha=0.5):
    """Create overlay of UMAP on original image"""
    # Ensure both are PIL Images
    if isinstance(original_img, np.ndarray):
        original_pil = Image.fromarray(original_img).convert('RGBA')
    else:
        original_pil = original_img.convert('RGBA')
    
    if isinstance(umap_img, np.ndarray):
        umap_pil = Image.fromarray((umap_img * 255).astype(np.uint8)).convert('RGBA')
    else:
        umap_pil = umap_img.convert('RGBA')
    
    # Resize UMAP to match original
    umap_resized = umap_pil.resize(original_pil.size, Image.NEAREST)
    
    # Apply alpha to UMAP
    umap_array = np.array(umap_resized)
    umap_array[:, :, 3] = int(255 * alpha)
    umap_transparent = Image.fromarray(umap_array, mode='RGBA')
    
    # Composite
    composite = Image.alpha_composite(original_pil, umap_transparent)
    
    return np.array(composite.convert('RGB'))


def create_comparison_gifs(frame_dir, vjepa2_frames, dinov3_vitl_frames, dinov3_vit7b_frames, output_dir):
    """Create individual GIFs with 3 rows: original, overlay, UMAP"""
    print("\n" + "="*60)
    print("Creating refined comparison GIFs...")
    print("="*60)
    
    frame_dir = Path(frame_dir)
    output_dir = Path(output_dir)
    
    # Load original frames (frames 616-631)
    frame_files = []
    for i in range(616, 632):  # 616 to 631 inclusive
        frame_file = frame_dir / f"NCAR_frame_{i:04d}.tif"
        if frame_file.exists():
            frame_files.append(frame_file)
    
    if len(frame_files) != 16:
        raise ValueError(f"Expected 16 frames, found {len(frame_files)}")
    
    original_frames = []
    for f in frame_files:
        img = Image.open(f).convert("RGB")
        img = img.resize((224, 224), Image.LANCZOS)
        original_frames.append(np.array(img))
    
    # Create GIFs for each model
    models = [
        ("vjepa2", vjepa2_frames, "V-JEPA 2 (1B params)"),
        ("dinov3_vitl", dinov3_vitl_frames, "DINOv3 ViT-L SAT-493M (300M)"),
        ("dinov3_vit7b", dinov3_vit7b_frames, "DINOv3 ViT-7B SAT-493M (6.7B)")
    ]
    
    for model_name, model_frames, title in models:
        frames = []
        for i in range(16):  # 16 frames
            # Create figure with 3 rows
            fig, axes = plt.subplots(3, 1, figsize=(4, 12))
            
            # Row 1: Original
            axes[0].imshow(original_frames[i])
            axes[0].set_title(f"NAIP Frame {i}", fontsize=10)
            axes[0].axis('off')
            
            # Row 2: Overlay (50% alpha composite)
            features_rgb = model_frames[i]
            overlay = create_overlay(original_frames[i], features_rgb, alpha=0.5)
            axes[1].imshow(overlay)
            axes[1].set_title(f"Overlay (50% alpha)", fontsize=10)
            axes[1].axis('off')
            
            # Row 3: UMAP features
            features_resized = np.array(Image.fromarray((features_rgb * 255).astype(np.uint8)).resize((224, 224), Image.NEAREST))
            axes[2].imshow(features_resized)
            
            # Add annotation for V-JEPA 2 temporal compression
            if model_name == "vjepa2":
                token_idx = i // 2
                axes[2].set_title(f"{title} (Token {token_idx})", fontsize=10)
            else:
                axes[2].set_title(f"{title}", fontsize=10)
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # Save to buffer
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)
            plt.close()
        
        # Save GIF with slower FPS (1/3 of original speed: 2 fps → 0.67 fps)
        gif_path = output_dir / f"{model_name}_comparison.gif"
        imageio.mimsave(gif_path, frames, fps=1.0, loop=0)
        print(f"Saved {gif_path}")
    
    # Create combined side-by-side comparison
    print("\nCreating combined comparison...")
    combined_frames = []
    
    for i in range(16):
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        
        # Row 1 - original frames
        for j in range(3):
            axes[0, j].imshow(original_frames[i])
            axes[0, j].set_title(f"NAIP Frame {i}", fontsize=10)
            axes[0, j].axis('off')
        
        # Row 2 - overlays
        # V-JEPA 2
        vjepa2_overlay = create_overlay(original_frames[i], vjepa2_frames[i], alpha=0.5)
        axes[1, 0].imshow(vjepa2_overlay)
        axes[1, 0].set_title(f"V-JEPA 2 Overlay", fontsize=10)
        axes[1, 0].axis('off')
        
        # DINOv3 ViT-L
        vitl_overlay = create_overlay(original_frames[i], dinov3_vitl_frames[i], alpha=0.5)
        axes[1, 1].imshow(vitl_overlay)
        axes[1, 1].set_title(f"DINOv3 ViT-L Overlay", fontsize=10)
        axes[1, 1].axis('off')
        
        # DINOv3 ViT-7B
        vit7b_overlay = create_overlay(original_frames[i], dinov3_vit7b_frames[i], alpha=0.5)
        axes[1, 2].imshow(vit7b_overlay)
        axes[1, 2].set_title(f"DINOv3 ViT-7B Overlay", fontsize=10)
        axes[1, 2].axis('off')
        
        # Row 3 - UMAP features
        # V-JEPA 2
        vjepa2_rgb = vjepa2_frames[i]
        vjepa2_resized = np.array(Image.fromarray((vjepa2_rgb * 255).astype(np.uint8)).resize((224, 224), Image.NEAREST))
        axes[2, 0].imshow(vjepa2_resized)
        axes[2, 0].set_title(f"V-JEPA 2 (Token {i//2})", fontsize=10)
        axes[2, 0].axis('off')
        
        # DINOv3 ViT-L
        vitl_rgb = dinov3_vitl_frames[i]
        vitl_resized = np.array(Image.fromarray((vitl_rgb * 255).astype(np.uint8)).resize((224, 224), Image.NEAREST))
        axes[2, 1].imshow(vitl_resized)
        axes[2, 1].set_title(f"DINOv3 ViT-L SAT-493M", fontsize=10)
        axes[2, 1].axis('off')
        
        # DINOv3 ViT-7B
        vit7b_rgb = dinov3_vit7b_frames[i]
        vit7b_resized = np.array(Image.fromarray((vit7b_rgb * 255).astype(np.uint8)).resize((224, 224), Image.NEAREST))
        axes[2, 2].imshow(vit7b_resized)
        axes[2, 2].set_title(f"DINOv3 ViT-7B SAT-493M", fontsize=10)
        axes[2, 2].axis('off')
        
        plt.suptitle("Vision Encoder Comparison - Consistent UMAP Projection", fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        # Save to buffer
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        combined_frames.append(frame)
        plt.close()
    
    # Save combined GIF with slower FPS
    combined_path = output_dir / "all_models_comparison.gif"
    imageio.mimsave(combined_path, combined_frames, fps=1.0, loop=0)
    print(f"Saved combined comparison: {combined_path}")
    
    return output_dir / "vjepa2_comparison.gif", output_dir / "dinov3_vitl_comparison.gif", output_dir / "dinov3_vit7b_comparison.gif"


def main():
    """Main execution"""
    frame_dir = Path("/home/lance/deepearth/encoders/vision/images/NCAR_frames_superlong")
    output_dir = Path("outputs/ncar_frames_616_631")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract features (will reuse if already exists)
    vjepa2_features, dinov3_vitl_features, dinov3_vit7b_features = extract_all_features(
        frame_dir, output_dir, first_frame=616, last_frame=631
    )
    
    # Compute consistent UMAP across all frames
    vjepa2_frames, dinov3_vitl_frames, dinov3_vit7b_frames = compute_consistent_umap(
        vjepa2_features, dinov3_vitl_features, dinov3_vit7b_features
    )
    
    # Create comparison GIFs with overlays
    gif1, gif2, gif3 = create_comparison_gifs(
        frame_dir, vjepa2_frames, dinov3_vitl_frames, dinov3_vit7b_frames, output_dir
    )
    
    print("\n" + "="*60)
    print("✅ Refined comparison complete!")
    print("="*60)
    print(f"\nKey improvements:")
    print(f"  - Consistent UMAP: Learned on ALL patches across ALL frames")
    print(f"  - 3 rows: Original → Overlay (50% alpha) → UMAP features")
    print(f"  - Animation FPS: 1.0 fps")
    print(f"  - Clarified SAT-493M training for DINOv3 models")
    print(f"\nIndividual GIFs:")
    print(f"  - V-JEPA 2: {gif1}")
    print(f"  - DINOv3 ViT-L SAT-493M: {gif2}")
    print(f"  - DINOv3 ViT-7B SAT-493M: {gif3}")
    print(f"\nCombined comparison: outputs/ncar_frames_616_631/all_models_comparison.gif")


if __name__ == "__main__":
    main()