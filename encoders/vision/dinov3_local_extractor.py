#!/usr/bin/env python3
"""
DINOv3 feature extraction using local weights for satellite imagery.
Uses the satellite-pretrained models for Earth observation.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Optional, Union, Tuple
import gc

# Add DINOv3 to path
sys.path.append('/home/lance/deepearth/encoders/vision/dinov3')


class DINOv3LocalExtractor:
    """
    DINOv3 feature extractor using locally downloaded weights.
    Optimized for satellite imagery with SAT-493M pretrained models.
    """
    
    def __init__(self, 
                 model_size: str = "vitl16",  # or "vit7b16"
                 weights_path: Optional[str] = None,
                 device: Union[str, torch.device] = "cuda:0",
                 use_fp16: bool = True):
        """
        Initialize DINOv3 with local weights.
        
        Args:
            model_size: Model size - "vitl16" (300M) or "vit7b16" (6.7B)
            weights_path: Path to weights file (auto-detected if None)
            device: Device for computation
            use_fp16: Use half precision
        """
        self.model_size = model_size
        self.device = torch.device(device) if isinstance(device, str) else device
        self.use_fp16 = use_fp16
        
        # Setup weights path
        if weights_path is None:
            # Check multiple possible weight directories
            possible_dirs = [
                Path("/opt/ecodash/deepearth/encoders/vision/dinov3_weights"),
                Path("/home/lance/deepearth/encoders/vision/dinov3/dinov3/weights"),
                Path(__file__).parent / "dinov3_weights"
            ]
            
            weights_dir = None
            for d in possible_dirs:
                if d.exists():
                    weights_dir = d
                    break
            
            if weights_dir is None:
                raise ValueError(f"No weights directory found. Tried: {possible_dirs}")
            
            # Map model sizes to available weight files
            if model_size == "vitl16" or model_size == "vitl":
                # Check for satellite or general model
                sat_path = weights_dir / "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
                lvd_path = weights_dir / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
                
                if sat_path.exists():
                    weights_path = sat_path
                    print(f"Using SAT-493M satellite-pretrained model")
                elif lvd_path.exists():
                    weights_path = lvd_path
                    print(f"Using LVD-1689M general-pretrained model")
                else:
                    raise FileNotFoundError(f"No ViT-L weights found in {weights_dir}")
                    
            elif model_size == "vit7b16" or model_size == "vit7b":
                weights_path = weights_dir / "dinov3_vit7b16_pretrain_sat493m.pth"
            else:
                # Try to find any matching weight file
                available_weights = list(weights_dir.glob("dinov3_*.pth"))
                if available_weights:
                    weights_path = available_weights[0]
                    print(f"Using available weights: {weights_path.name}")
                else:
                    raise ValueError(f"No weights found for model size: {model_size} in {weights_dir}")
        
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")
        
        print(f"Loading DINOv3 {model_size} from {self.weights_path}")
        self._load_model()
        
    def _load_model(self):
        """Load DINOv3 model using torch hub with local weights"""
        # Try multiple possible dinov3 locations
        dinov3_paths = [
            Path("/opt/ecodash/dinov3"),
            Path("/opt/ecodash/deepearth/encoders/vision/dinov3"),
            Path("/home/lance/deepearth/encoders/vision/dinov3"),
            Path(__file__).parent / "dinov3"
        ]
        
        dinov3_dir = None
        for p in dinov3_paths:
            if p.exists() and (p / "hubconf.py").exists():
                dinov3_dir = str(p)
                break
        
        if dinov3_dir is None:
            raise ValueError(f"DINOv3 repository not found. Tried: {dinov3_paths}")
        
        # Load model architecture from hub
        # Map model sizes to hub function names - use correct names!
        if self.model_size in ["vitl16", "vitl"]:
            model_fn = "dinov3_vitl16"  # Correct function name
        elif self.model_size in ["vit7b16", "vit7b"]:
            model_fn = "dinov3_vit7b16"  # Correct function name
        else:
            model_fn = f"dinov3_{self.model_size}"
        
        try:
            self.model = torch.hub.load(
                repo_or_dir=dinov3_dir,
                model=model_fn,
                source="local",
                pretrained=False  # Don't download weights
            )
        except Exception as e:
            print(f"Failed to load {model_fn}, trying alternative loading method: {e}")
            # Try direct model loading
            sys.path.insert(0, dinov3_dir)
            import hubconf
            # Use the correct model function directly
            if self.model_size in ["vitl16", "vitl"]:
                self.model = hubconf.dinov3_vitl16(pretrained=False)
            elif self.model_size in ["vit7b16", "vit7b"]:
                self.model = hubconf.dinov3_vit7b16(pretrained=False)
            else:
                # List available models as fallback
                available = [attr for attr in dir(hubconf) if attr.startswith('dinov3_')]
                print(f"Available models in hubconf: {available}")
                raise ValueError(f"Cannot find model for {self.model_size}")
        
        # Load local weights
        print(f"Loading weights from {self.weights_path}")
        state_dict = torch.load(self.weights_path, map_location='cpu')
        
        # Handle different weight formats
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        # Load weights
        self.model.load_state_dict(new_state_dict, strict=False)
        
        # Move to device and set precision
        self.model = self.model.to(self.device)
        if self.use_fp16:
            self.model = self.model.half()
        self.model.eval()
        
        # Model configuration
        if "vitl16" in self.model_size:
            self.hidden_dim = 1024
            self.num_heads = 16
            self.num_layers = 24
        elif "vit7b16" in self.model_size:
            self.hidden_dim = 1536
            self.num_heads = 24
            self.num_layers = 40
        
        self.patch_size = 16
        print(f"Model loaded: {self.model_size} (dim={self.hidden_dim})")
        
    def preprocess_image(self, image_path: Union[str, Path, Image.Image]) -> torch.Tensor:
        """Preprocess image for DINOv3"""
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path
        
        # Resize to 224x224 (standard DINOv3 input)
        image = image.resize((224, 224), Image.LANCZOS)
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        return image_tensor.unsqueeze(0).to(self.device)
    
    def extract_features(self, 
                        image_path: Union[str, Path, Image.Image],
                        layer: int = -1) -> Dict[str, torch.Tensor]:
        """
        Extract DINOv3 features from an image.
        
        Args:
            image_path: Path to image or PIL Image
            layer: Which layer to extract from (-1 for last)
            
        Returns:
            Dictionary with patch features and metadata
        """
        # Preprocess
        image_tensor = self.preprocess_image(image_path)
        if self.use_fp16:
            image_tensor = image_tensor.half()
        
        # Extract features
        with torch.no_grad():
            if self.use_fp16:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    # Get intermediate layers
                    features = self.model.get_intermediate_layers(
                        image_tensor, 
                        n=[self.num_layers + layer] if layer < 0 else [layer],
                        reshape=True,
                        norm=True
                    )[0]
            else:
                features = self.model.get_intermediate_layers(
                    image_tensor,
                    n=[self.num_layers + layer] if layer < 0 else [layer],
                    reshape=True,
                    norm=True
                )[0]
        
        # Features shape from DINOv3: [B, D, H, W]
        # We need to transpose to [B, H, W, D] then flatten
        B, D, H, W = features.shape
        features = features.permute(0, 2, 3, 1)  # [B, H, W, D]
        features_flat = features.view(B, H*W, D).squeeze(0)  # [H*W, D]
        
        return {
            'patch_features': features_flat.cpu().float(),  # [H*W, D]
            'spatial_shape': (H, W),
            'feature_dim': D,
            'model': self.model_size
        }
    
    def process_frames(self,
                      frame_dir: Union[str, Path],
                      output_dir: Union[str, Path],
                      first_frame: int = 0,
                      last_frame: int = 15):
        """
        Process sequential frames like we did with V-JEPA 2.
        
        Args:
            frame_dir: Directory with frames
            output_dir: Output directory
            first_frame: First frame index
            last_frame: Last frame index
        """
        frame_dir = Path(frame_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Get frame files
        frame_files = sorted(frame_dir.glob("*_frame_*.tif"))
        
        # Filter by frame range
        selected_frames = []
        for f in frame_files:
            try:
                frame_num = int(f.stem.split('_')[-1])
                if first_frame <= frame_num <= last_frame:
                    selected_frames.append((frame_num, f))
            except:
                continue
        
        selected_frames.sort(key=lambda x: x[0])
        print(f"Processing {len(selected_frames)} frames from {first_frame} to {last_frame}")
        
        all_features = []
        frame_numbers = []
        
        for frame_num, frame_path in tqdm(selected_frames, desc="Extracting features"):
            result = self.extract_features(frame_path)
            all_features.append(result['patch_features'])
            frame_numbers.append(frame_num)
        
        # Stack features
        features_tensor = torch.stack(all_features)  # [N, 196, D]
        
        # Save features
        output_file = output_dir / f"dinov3_{self.model_size}_frames_{first_frame:04d}_{last_frame:04d}.pt"
        
        torch.save({
            'features': features_tensor,
            'shape': features_tensor.shape,
            'frame_numbers': frame_numbers,
            'model_info': {
                'model': self.model_size,
                'weights': str(self.weights_path),
                'feature_dim': self.hidden_dim,
                'patch_size': self.patch_size,
                'spatial_patches': 196,  # 14x14 for 224x224 input
                'pretrained_on': 'SAT-493M'
            },
            'extraction_timestamp': datetime.now().isoformat()
        }, output_file)
        
        print(f"Saved features to {output_file}")
        print(f"Features shape: {features_tensor.shape}")
        
        return features_tensor


def main():
    """Test DINOv3 on NAIP frames 0-15 like we did with V-JEPA 2"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DINOv3 feature extraction with local weights')
    parser.add_argument('--frame_dir', type=str, 
                       default='/home/lance/deepearth/encoders/vision/images/NCAR_frames_superlong',
                       help='Directory containing frames')
    parser.add_argument('--output_dir', type=str,
                       default='/home/lance/deepearth/encoders/vision/outputs/dinov3_test',
                       help='Output directory')
    parser.add_argument('--model', type=str, default='vitl16',
                       choices=['vitl16', 'vit7b16'],
                       help='Model size')
    parser.add_argument('--first_frame', type=int, default=0,
                       help='First frame index')
    parser.add_argument('--last_frame', type=int, default=15,
                       help='Last frame index')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    parser.add_argument('--no_fp16', action='store_true',
                       help='Disable fp16')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = DINOv3LocalExtractor(
        model_size=args.model,
        device=args.device,
        use_fp16=not args.no_fp16
    )
    
    # Process frames
    features = extractor.process_frames(
        frame_dir=args.frame_dir,
        output_dir=args.output_dir,
        first_frame=args.first_frame,
        last_frame=args.last_frame
    )
    
    print(f"\nExtraction complete!")
    print(f"Processed frames {args.first_frame} to {args.last_frame}")
    print(f"Output shape: {features.shape}")


if __name__ == "__main__":
    main()