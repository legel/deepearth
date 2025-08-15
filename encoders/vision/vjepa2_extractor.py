#!/usr/bin/env python3
"""
V-JEPA 2 feature extraction for images.
Processes images through the V-JEPA 2 model to extract spatiotemporal features.
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

# Setup logging
def setup_logger(name="vjepa2", log_dir="logs", gpu_id=None):
    """Setup logger for tracking extraction progress"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    gpu_suffix = f"_gpu{gpu_id}" if gpu_id is not None else ""
    log_file = log_dir / f"{name}{gpu_suffix}_{timestamp}.log"
    
    logger = logging.getLogger(f'{name}{gpu_suffix}')
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

class VJEPA2Extractor:
    """
    V-JEPA 2 feature extractor for images.
    
    This extractor processes images through the V-JEPA 2 model to extract
    rich spatiotemporal features. The model is self-supervised and trained
    on general video data, making it suitable for any visual domain.
    """
    
    def __init__(self, 
                 model_name: str = "facebook/vjepa2-vitg-fpc64-384",
                 device: Union[str, torch.device] = "cuda:0",
                 use_fp16: bool = True,
                 log_dir: str = "logs"):
        """
        Initialize the V-JEPA 2 extractor.
        
        Args:
            model_name: Hugging Face model name/path
            device: Device to run the model on
            use_fp16: Whether to use half precision
            log_dir: Directory for logs
        """
        self.model_name = model_name
        self.device = torch.device(device) if isinstance(device, str) else device
        self.use_fp16 = use_fp16
        
        # Extract GPU ID from device for logging
        gpu_id = None
        if self.device.type == 'cuda':
            gpu_id = self.device.index if self.device.index is not None else 0
        
        # Setup logger
        self.logger = setup_logger("vjepa2", log_dir, gpu_id)
        self.logger.info(f"Initializing V-JEPA 2 extractor on {self.device}")
        
        # Load model
        self._load_model()
        
        # Model specifications
        self.num_patches = 4608  # 576 spatial × 8 temporal
        self.patch_dim = 1408
        self.num_frames = 16
        self.spatial_patches = 576  # 24×24 grid
        self.temporal_frames = 8
        self.spatial_grid_size = 24
    
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
    
    def extract_features(self, image: Union[str, Path, Image.Image]) -> Optional[torch.Tensor]:
        """
        Extract V-JEPA 2 features from a single image.
        
        Args:
            image: Image path or PIL Image
            
        Returns:
            torch.Tensor: Features of shape [4608, 1408] or None if error
        """
        try:
            # Load image if path provided
            if isinstance(image, (str, Path)):
                image = Image.open(image).convert("RGB")
            elif not isinstance(image, Image.Image):
                raise ValueError("Input must be a path or PIL Image")
            
            # Process image - V-JEPA 2 expects video format
            pixel_values = self.processor(image, return_tensors="pt").to(self.device)["pixel_values_videos"]
           
            # Repeat for temporal dimension (simulate video from single image)
            pixel_values = pixel_values.repeat(1, self.num_frames, 1, 1, 1)
            
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
            self.logger.error(f"Error processing image: {e}")
            return None
    
    def extract_features_batch(self, 
                              images: List[Union[str, Path, Image.Image]], 
                              batch_size: int = 1) -> List[Optional[torch.Tensor]]:
        """
        Extract features from multiple images.
        
        Args:
            images: List of image paths or PIL Images
            batch_size: Batch size for processing
            
        Returns:
            List of features tensors
        """
        features_list = []
        
        for i in tqdm(range(0, len(images), batch_size), desc="Extracting features"):
            batch = images[i:i + batch_size]
            
            for image in batch:
                features = self.extract_features(image)
                features_list.append(features)
                
            # Clear cache periodically
            if i % (batch_size * 10) == 0:
                gc.collect()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        return features_list
    
    def aggregate_features(self, 
                          features: torch.Tensor, 
                          method: str = "mean") -> torch.Tensor:
        """
        Aggregate patch features to image-level representation.
        
        Args:
            features: Tensor of shape [4608, 1408]
            method: Aggregation method ('mean', 'max', 'cls', 'spatial_mean')
            
        Returns:
            Aggregated features tensor
        """
        if method == "mean":
            return features.mean(dim=0)  # [1408]
        elif method == "max":
            return features.max(dim=0)[0]  # [1408]
        elif method == "cls":
            # Use first patch as CLS token
            return features[0]  # [1408]
        elif method == "spatial_mean":
            # Average across spatial dimension, keep temporal
            features_reshaped = features.view(self.temporal_frames, self.spatial_patches, self.patch_dim)
            return features_reshaped.mean(dim=1)  # [8, 1408]
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def get_spatial_features(self, features: torch.Tensor, frame: int = 0) -> torch.Tensor:
        """
        Get spatial features for a specific temporal frame.
        
        Args:
            features: Tensor of shape [4608, 1408]
            frame: Temporal frame index (0-7)
            
        Returns:
            Spatial features of shape [24, 24, 1408]
        """
        # Reshape to separate temporal and spatial
        features_reshaped = features.view(self.temporal_frames, self.spatial_patches, self.patch_dim)
        
        # Get specific frame and reshape to grid
        frame_features = features_reshaped[frame]  # [576, 1408]
        spatial_grid = frame_features.view(self.spatial_grid_size, self.spatial_grid_size, self.patch_dim)
        
        return spatial_grid
    
    def save_features(self, 
                     features: torch.Tensor,
                     output_path: Union[str, Path],
                     metadata: Optional[Dict] = None):
        """
        Save extracted features with metadata.
        
        Args:
            features: Feature tensor
            output_path: Path to save the features
            metadata: Optional metadata dictionary
        """
        data = {
            'features': features,
            'shape': features.shape,
            'dtype': str(features.dtype),
            'model': self.model_name,
            'extraction_timestamp': datetime.now().isoformat(),
            'model_info': {
                'num_patches': self.num_patches,
                'patch_dim': self.patch_dim,
                'spatial_patches': self.spatial_patches,
                'temporal_frames': self.temporal_frames,
                'spatial_grid_size': self.spatial_grid_size
            }
        }
        
        if metadata:
            data.update(metadata)
        
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


class BatchVJEPA2Extractor(VJEPA2Extractor):
    """
    Batch processor for extracting V-JEPA 2 features from directories of images.
    """
    
    def __init__(self, 
                 output_dir: Union[str, Path],
                 chunk_size: int = 1000,
                 resume: bool = True,
                 **kwargs):
        """
        Initialize batch extractor.
        
        Args:
            output_dir: Directory to save features
            chunk_size: Number of images per chunk file
            resume: Whether to resume from previous progress
            **kwargs: Arguments passed to VJEPA2Extractor
        """
        super().__init__(**kwargs)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.chunk_size = chunk_size
        self.resume = resume
        
        # Progress tracking
        self.progress_file = self.output_dir / "extraction_progress.json"
        self.progress = self._load_progress() if resume else self._init_progress()
    
    def _init_progress(self) -> Dict:
        """Initialize progress tracking"""
        return {
            'processed_images': [],
            'last_chunk': -1,
            'total_processed': 0,
            'started_at': datetime.now().isoformat()
        }
    
    def _load_progress(self) -> Dict:
        """Load progress from file"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return self._init_progress()
    
    def _save_progress(self):
        """Save progress to file"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def process_directory(self, 
                         image_dir: Union[str, Path],
                         pattern: str = "*.jpg",
                         metadata_func: Optional[callable] = None):
        """
        Process all images in a directory.
        
        Args:
            image_dir: Directory containing images
            pattern: Glob pattern for image files
            metadata_func: Function to extract metadata from image path
        """
        image_dir = Path(image_dir)
        image_paths = sorted(image_dir.rglob(pattern))
        
        self.logger.info(f"Found {len(image_paths)} images in {image_dir}")
        
        # Filter already processed if resuming
        if self.resume:
            processed_set = set(self.progress['processed_images'])
            image_paths = [p for p in image_paths if str(p) not in processed_set]
            self.logger.info(f"Resuming: {len(image_paths)} images remaining")
        
        # Process images in chunks
        chunk_features = {}
        chunk_id = self.progress['last_chunk'] + 1
        
        for img_path in tqdm(image_paths, desc="Processing images"):
            # Extract features
            features = self.extract_features(img_path)
            
            if features is not None:
                # Get metadata if function provided
                metadata = metadata_func(img_path) if metadata_func else {}
                
                # Store features
                img_id = str(img_path)
                chunk_features[img_id] = {
                    'features': features,
                    'metadata': metadata
                }
                
                # Update progress
                self.progress['processed_images'].append(img_id)
                self.progress['total_processed'] += 1
                
                # Save chunk when full
                if len(chunk_features) >= self.chunk_size:
                    self._save_chunk(chunk_features, chunk_id)
                    chunk_features = {}
                    chunk_id += 1
                    self.progress['last_chunk'] = chunk_id - 1
                    self._save_progress()
        
        # Save final chunk
        if chunk_features:
            self._save_chunk(chunk_features, chunk_id)
            self.progress['last_chunk'] = chunk_id
        
        self._save_progress()
        self.logger.info(f"Processing complete. Total: {self.progress['total_processed']} images")
    
    def _save_chunk(self, features_dict: Dict, chunk_id: int):
        """Save a chunk of features"""
        chunk_file = self.output_dir / f"features_chunk_{chunk_id:04d}.pt"
        torch.save(features_dict, chunk_file)
        self.logger.info(f"Saved chunk {chunk_id} with {len(features_dict)} images")


def main():
    parser = argparse.ArgumentParser(description='V-JEPA 2 feature extraction')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save features')
    parser.add_argument('--model', type=str, default="facebook/vjepa2-vitg-fpc64-384",
                       help='Model name or path')
    parser.add_argument('--device', type=str, default="cuda:0",
                       help='Device to use')
    parser.add_argument('--chunk_size', type=int, default=1000,
                       help='Images per chunk file')
    parser.add_argument('--pattern', type=str, default="*.jpg",
                       help='Image file pattern')
    parser.add_argument('--no_fp16', action='store_true',
                       help='Disable fp16 precision')
    parser.add_argument('--no_resume', action='store_true',
                       help='Start fresh, ignore previous progress')
    
    args = parser.parse_args()
    
    # Create batch extractor
    extractor = BatchVJEPA2Extractor(
        output_dir=args.output_dir,
        model_name=args.model,
        device=args.device,
        use_fp16=not args.no_fp16,
        chunk_size=args.chunk_size,
        resume=not args.no_resume
    )
    
    # Process directory
    extractor.process_directory(
        image_dir=args.image_dir,
        pattern=args.pattern
    )


if __name__ == "__main__":
    main()
