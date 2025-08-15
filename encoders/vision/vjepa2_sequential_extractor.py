#!/usr/bin/env python3
"""
V-JEPA 2 sequential frame feature extraction.
Processes sequences of 16 frames through V-JEPA 2 to extract true spatiotemporal features.
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
import json
from transformers import AutoVideoProcessor, AutoModel
import gc
import argparse
from typing import List, Dict, Optional, Union, Tuple
import re

# Setup logging
def setup_logger(name="vjepa2_seq", log_dir="logs"):
    """Setup logger for tracking extraction progress"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"{name}_{timestamp}.log"
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Add handlers
    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler(sys.stdout)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


class SequentialVJEPA2Extractor:
    """
    Sequential V-JEPA 2 feature extractor for frame sequences.
    Processes actual frame sequences (16 frames) to extract true spatiotemporal features.
    """
    
    def __init__(self, 
                 model_name: str = "facebook/vjepa2-vitg-fpc64-384",
                 device: Union[str, torch.device] = "cuda:0",
                 use_fp16: bool = True,
                 log_dir: str = "logs"):
        """
        Initialize the sequential V-JEPA 2 extractor.
        
        Args:
            model_name: Hugging Face model name/path
            device: Device to run the model on
            use_fp16: Whether to use half precision
            log_dir: Directory for logs
        """
        self.model_name = model_name
        self.device = torch.device(device) if isinstance(device, str) else device
        self.use_fp16 = use_fp16
        
        # Setup logger
        self.logger = setup_logger("vjepa2_seq", log_dir)
        self.logger.info(f"Initializing Sequential V-JEPA 2 extractor on {self.device}")
        
        # Load model
        self._load_model()
        
        # Model specifications
        self.num_input_frames = 16  # Model expects 16 frames
        self.num_output_temporal = 8  # FPC compresses to 8 temporal tokens
        self.spatial_patches = 576  # 24×24 grid
        self.spatial_grid_size = 24
        self.patch_dim = 1408
        self.total_patches = self.num_output_temporal * self.spatial_patches  # 4608
    
    def _load_model(self):
        """Load the V-JEPA 2 model and processor"""
        self.logger.info(f"Loading model: {self.model_name}")
        
        if self.use_fp16:
            self.model = AutoModel.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16
            ).to(self.device)
        else:
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            
        self.processor = AutoVideoProcessor.from_pretrained(self.model_name)
        self.model.eval()
        
        # Log model info
        param_count = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model loaded successfully")
        self.logger.info(f"Parameters: {param_count:,}")
        self.logger.info(f"Precision: {'fp16' if self.use_fp16 else 'fp32'}")
    
    def load_frame_sequence(self, frame_paths: List[Path]) -> List[Image.Image]:
        """
        Load a sequence of frames.
        
        Args:
            frame_paths: List of paths to frame files
            
        Returns:
            List of PIL Images
        """
        frames = []
        for path in frame_paths:
            try:
                img = Image.open(path).convert("RGB")
                frames.append(img)
            except Exception as e:
                self.logger.error(f"Error loading frame {path}: {e}")
                return None
        return frames
    
    def extract_features_from_sequence(self, frames: List[Image.Image]) -> Optional[torch.Tensor]:
        """
        Extract V-JEPA 2 features from a sequence of 16 frames.
        
        Args:
            frames: List of 16 PIL Images
            
        Returns:
            torch.Tensor: Features of shape [4608, 1408] or None if error
        """
        if len(frames) != self.num_input_frames:
            self.logger.error(f"Expected {self.num_input_frames} frames, got {len(frames)}")
            return None
        
        try:
            # Process frames - V-JEPA 2 expects video format
            # The processor handles the normalization and resizing
            inputs = self.processor(frames, return_tensors="pt")
            pixel_values = inputs["pixel_values_videos"].to(self.device)
            
            # Extract features
            with torch.no_grad():
                # Get patch embeddings from the model
                patch_embeddings = self.model.get_vision_features(pixel_values)
                
                # Remove batch dimension
                features = patch_embeddings[0]  # [4608, 1408]
                
                if self.use_fp16 and features.dtype != torch.float16:
                    features = features.half()
                
                # Move to CPU for storage
                features = features.cpu()
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error processing frame sequence: {e}")
            return None
    
    def process_frame_directory(self, 
                              frame_dir: Union[str, Path],
                              output_dir: Union[str, Path],
                              frame_pattern: str = r".*_frame_(\d+)\.tif",
                              window_size: int = 16,
                              stride: Optional[int] = None,
                              first_frame: int = -1,
                              last_frame: int = -1,
                              batch_size: int = 1):
        """
        Process all frames in a directory using sliding windows.
        
        Args:
            frame_dir: Directory containing frames
            output_dir: Directory to save features
            frame_pattern: Regex pattern to extract frame number
            window_size: Number of frames per window (must be 16 for V-JEPA 2)
            stride: Stride for sliding window (default = window_size for non-overlapping)
            first_frame: First frame number to process (-1 for all)
            last_frame: Last frame number to process (-1 for all)
            batch_size: Number of windows to process in parallel
        """
        if window_size != 16:
            self.logger.warning(f"V-JEPA 2 requires 16 frames, but window_size={window_size}")
            window_size = 16
        
        if stride is None:
            stride = window_size
        
        frame_dir = Path(frame_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Get all frame files and sort by frame number
        frame_files = []
        pattern = re.compile(frame_pattern)
        
        for file_path in sorted(frame_dir.glob("*")):
            match = pattern.match(file_path.name)
            if match:
                frame_num = int(match.group(1))
                # Apply frame range filtering
                if first_frame != -1 and frame_num < first_frame:
                    continue
                if last_frame != -1 and frame_num > last_frame:
                    continue
                frame_files.append((frame_num, file_path))
        
        # Sort by frame number
        frame_files.sort(key=lambda x: x[0])
        self.logger.info(f"Found {len(frame_files)} frames in range {first_frame} to {last_frame}")
        
        if len(frame_files) == 0:
            self.logger.error("No frames found matching pattern and range")
            return
        
        # Process in windows
        num_windows = (len(frame_files) - window_size) // stride + 1
        self.logger.info(f"Processing {num_windows} windows with stride {stride}, batch size {batch_size}")
        
        # Process windows in batches for better GPU utilization
        for batch_start_idx in range(0, num_windows, batch_size):
            batch_end_idx = min(batch_start_idx + batch_size, num_windows)
            batch_windows = []
            
            # Collect batch of windows
            for window_idx in range(batch_start_idx, batch_end_idx):
                start_idx = window_idx * stride
                end_idx = start_idx + window_size
                
                if end_idx > len(frame_files):
                    break
                
                # Get frame paths for this window
                window_frames = frame_files[start_idx:end_idx]
                frame_paths = [f[1] for f in window_frames]
                frame_numbers = [f[0] for f in window_frames]
                
                batch_windows.append({
                    'window_idx': window_idx,
                    'frame_paths': frame_paths,
                    'frame_numbers': frame_numbers
                })
            
            # Process batch
            for window_data in batch_windows:
                window_idx = window_data['window_idx']
                frame_paths = window_data['frame_paths']
                frame_numbers = window_data['frame_numbers']
                
                self.logger.info(f"Processing window {window_idx}: frames {frame_numbers[0]:04d}-{frame_numbers[-1]:04d}")
                
                # Load frames
                frames = self.load_frame_sequence(frame_paths)
                if frames is None:
                    continue
                
                # Extract features
                features = self.extract_features_from_sequence(frames)
                if features is None:
                    continue
                
                # Prepare metadata
                metadata = {
                    'window_idx': window_idx,
                    'start_frame': frame_numbers[0],
                    'end_frame': frame_numbers[-1],
                    'frame_numbers': frame_numbers,
                    'frame_files': [str(p.name) for p in frame_paths],
                    'num_input_frames': window_size,
                    'num_output_temporal': self.num_output_temporal,
                    'temporal_mapping': {
                        f't{i}': f'frames_{frame_numbers[i*2]:04d}_{frame_numbers[i*2+1]:04d}' 
                        for i in range(self.num_output_temporal)
                    },
                    'model': self.model_name,
                    'extraction_timestamp': datetime.now().isoformat()
                }
                
                # Save features
                output_file = output_dir / f"features_frames_{frame_numbers[0]:04d}_{frame_numbers[-1]:04d}.pt"
                self.save_features(features, output_file, metadata)
                
                self.logger.info(f"Saved features to {output_file}")
            
            # Clear GPU cache after batch
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
        
        self.logger.info("Processing complete!")
    
    def save_features(self, 
                     features: torch.Tensor,
                     output_path: Union[str, Path],
                     metadata: Dict):
        """
        Save extracted features with metadata.
        
        Args:
            features: Feature tensor [4608, 1408]
            output_path: Path to save the features
            metadata: Metadata dictionary
        """
        data = {
            'features': features,
            'shape': features.shape,
            'dtype': str(features.dtype),
            'model_info': {
                'model_name': self.model_name,
                'num_input_frames': self.num_input_frames,
                'num_output_temporal': self.num_output_temporal,
                'spatial_patches': self.spatial_patches,
                'spatial_grid_size': self.spatial_grid_size,
                'patch_dim': self.patch_dim,
                'total_patches': self.total_patches
            },
            'metadata': metadata
        }
        
        torch.save(data, output_path)
    
    @staticmethod
    def load_features(features_path: Union[str, Path]) -> Dict:
        """
        Load saved features.
        
        Args:
            features_path: Path to saved features
            
        Returns:
            Dictionary containing features and metadata
        """
        return torch.load(features_path, map_location='cpu')


def main():
    parser = argparse.ArgumentParser(description='V-JEPA 2 sequential frame extraction')
    parser.add_argument('--frame_dir', type=str, required=True,
                       help='Directory containing frame sequences')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save features')
    parser.add_argument('--model', type=str, default="facebook/vjepa2-vitg-fpc64-384",
                       help='Model name or path')
    parser.add_argument('--device', type=str, default="cuda:0",
                       help='Device to use')
    parser.add_argument('--frame_pattern', type=str, default=r".*_frame_(\d+)\.tif",
                       help='Regex pattern to extract frame number')
    parser.add_argument('--stride', type=int, default=None,
                       help='Stride for sliding window (default = 16 for non-overlapping)')
    parser.add_argument('--first_frame', type=int, default=-1,
                       help='First frame number to process (-1 for all)')
    parser.add_argument('--last_frame', type=int, default=-1,
                       help='Last frame number to process (-1 for all)')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Number of windows to process in batch (for GPU memory optimization)')
    parser.add_argument('--no_fp16', action='store_true',
                       help='Disable fp16 precision')
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = SequentialVJEPA2Extractor(
        model_name=args.model,
        device=args.device,
        use_fp16=not args.no_fp16
    )
    
    # Process directory
    extractor.process_frame_directory(
        frame_dir=args.frame_dir,
        output_dir=args.output_dir,
        frame_pattern=args.frame_pattern,
        stride=args.stride,
        first_frame=args.first_frame,
        last_frame=args.last_frame,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()