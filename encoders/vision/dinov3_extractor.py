#!/usr/bin/env python3
"""
DINOv3 feature extraction for Earth observation imagery.
Provides efficient extraction of dense visual features using Meta's DINOv3 foundation model.
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
from typing import List, Dict, Optional, Union, Tuple
from transformers import AutoImageProcessor, AutoModel
import gc
import argparse


# Setup logging
def setup_logger(name="dinov3", log_dir="logs"):
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


class DINOv3Extractor:
    """
    DINOv3 feature extractor for dense visual representations.
    
    DINOv3 produces patch-level features that capture semantic information
    without supervised training bias, ideal for Earth observation tasks.
    """
    
    def __init__(self, 
                 model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
                 device: Union[str, torch.device] = "cuda:0",
                 use_fp16: bool = True,
                 log_dir: str = "logs"):
        """
        Initialize the DINOv3 extractor.
        
        Args:
            model_name: Hugging Face model name
            device: Device to run the model on
            use_fp16: Whether to use half precision
            log_dir: Directory for logs
        
        Available models:
        - facebook/dinov3-vits16-pretrain-lvd1689m (21M params)
        - facebook/dinov3-vitb16-pretrain-lvd1689m (86M params)
        - facebook/dinov3-vitl16-pretrain-lvd1689m (300M params)
        - facebook/dinov3-vith16plus-pretrain-lvd1689m (840M params)
        - facebook/dinov3-vit7b16-pretrain-lvd1689m (6.7B params)
        - facebook/dinov3-vitl16-pretrain-sat493m (satellite-specific)
        """
        self.model_name = model_name
        self.device = torch.device(device) if isinstance(device, str) else device
        self.use_fp16 = use_fp16
        
        # Setup logger
        self.logger = setup_logger("dinov3", log_dir)
        self.logger.info(f"Initializing DINOv3 extractor with {model_name}")
        
        # Load model and processor
        self._load_model()
        
        # Get model configuration
        self._setup_model_config()
        
    def _load_model(self):
        """Load the DINOv3 model and processor from Hugging Face"""
        self.logger.info(f"Loading model: {self.model_name}")
        
        try:
            # Load processor
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            
            # Load model with appropriate dtype
            if self.use_fp16:
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map=self.device
                )
            else:
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    device_map=self.device
                )
            
            self.model.eval()
            
            # Log model info
            param_count = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"Model loaded successfully")
            self.logger.info(f"Parameters: {param_count:,}")
            self.logger.info(f"Precision: {'fp16' if self.use_fp16 else 'fp32'}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _setup_model_config(self):
        """Setup model configuration based on architecture"""
        # Determine patch size and grid from model name
        if "vits" in self.model_name.lower():
            self.patch_size = 16
            self.hidden_dim = 384
            self.num_heads = 6
        elif "vitb" in self.model_name.lower():
            self.patch_size = 16
            self.hidden_dim = 768
            self.num_heads = 12
        elif "vitl" in self.model_name.lower():
            self.patch_size = 16
            self.hidden_dim = 1024
            self.num_heads = 16
        elif "vith" in self.model_name.lower():
            self.patch_size = 16
            self.hidden_dim = 1280
            self.num_heads = 16
        elif "vit7b" in self.model_name.lower():
            self.patch_size = 16
            self.hidden_dim = 1536
            self.num_heads = 24
        elif "convnext" in self.model_name.lower():
            # ConvNeXt models have different architecture
            self.patch_size = None  # Not applicable for ConvNeXt
            self.hidden_dim = None  # Varies by layer
            self.num_heads = None   # Not applicable
        else:
            # Default values
            self.patch_size = 16
            self.hidden_dim = 768
            self.num_heads = 12
        
        # Default image size
        self.image_size = 224
        self.num_patches = (self.image_size // self.patch_size) ** 2 if self.patch_size else None
    
    def extract_features(self, 
                        image_path: Union[str, Path, Image.Image],
                        return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Extract DINOv3 features from an image.
        
        Args:
            image_path: Path to image or PIL Image
            return_attention: Whether to return attention maps
            
        Returns:
            Dictionary containing:
                - 'patch_features': Patch-level features [N, D]
                - 'cls_token': Global CLS token feature [D]
                - 'pooled_output': Pooled representation [D]
                - 'attention': Attention maps (if requested)
        """
        # Load image
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # Extract features
        with torch.no_grad():
            if self.use_fp16:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.model(**inputs, output_attentions=return_attention)
            else:
                outputs = self.model(**inputs, output_attentions=return_attention)
        
        # Prepare output dictionary
        result = {}
        
        # Get patch features (excluding CLS token)
        last_hidden_state = outputs.last_hidden_state[0]  # Remove batch dimension
        result['patch_features'] = last_hidden_state[1:].cpu()  # Skip CLS token
        result['cls_token'] = last_hidden_state[0].cpu()  # CLS token
        
        # Get pooled output if available
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            result['pooled_output'] = outputs.pooler_output[0].cpu()
        else:
            # Use CLS token as pooled output
            result['pooled_output'] = result['cls_token']
        
        # Add attention if requested
        if return_attention and hasattr(outputs, 'attentions'):
            result['attention'] = [att[0].cpu() for att in outputs.attentions]  # Remove batch dim
        
        return result
    
    def extract_intermediate_features(self, 
                                     image_path: Union[str, Path, Image.Image],
                                     layers: List[int] = None) -> Dict[str, torch.Tensor]:
        """
        Extract features from intermediate layers.
        
        Args:
            image_path: Path to image or PIL Image
            layers: List of layer indices to extract (None = all layers)
            
        Returns:
            Dictionary with layer indices as keys and features as values
        """
        # Load image
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # Extract features with hidden states
        with torch.no_grad():
            if self.use_fp16:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.model(**inputs, output_hidden_states=True)
            else:
                outputs = self.model(**inputs, output_hidden_states=True)
        
        # Get hidden states from all layers
        hidden_states = outputs.hidden_states
        
        # Select requested layers
        if layers is None:
            layers = list(range(len(hidden_states)))
        
        result = {}
        for layer_idx in layers:
            if layer_idx < len(hidden_states):
                # Remove batch dimension and skip CLS token
                result[layer_idx] = hidden_states[layer_idx][0][1:].cpu()
        
        return result
    
    def process_batch(self, 
                     image_paths: List[Union[str, Path]],
                     batch_size: int = 8) -> List[Dict[str, torch.Tensor]]:
        """
        Process multiple images in batches.
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing
            
        Returns:
            List of feature dictionaries
        """
        results = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    batch_images.append(img)
                except Exception as e:
                    self.logger.error(f"Error loading {path}: {e}")
                    continue
            
            if not batch_images:
                continue
            
            # Process batch
            inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                if self.use_fp16:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)
            
            # Extract features for each image
            for j in range(len(batch_images)):
                result = {
                    'patch_features': outputs.last_hidden_state[j][1:].cpu(),
                    'cls_token': outputs.last_hidden_state[j][0].cpu(),
                    'pooled_output': outputs.pooler_output[j].cpu() if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None else outputs.last_hidden_state[j][0].cpu()
                }
                results.append(result)
        
        return results
    
    def save_features(self, 
                     features: Dict[str, torch.Tensor],
                     output_path: Union[str, Path],
                     metadata: Optional[Dict] = None):
        """
        Save extracted features with metadata.
        
        Args:
            features: Feature dictionary
            output_path: Path to save the features
            metadata: Optional metadata dictionary
        """
        data = {
            'features': features,
            'model_info': {
                'model_name': self.model_name,
                'patch_size': self.patch_size,
                'hidden_dim': self.hidden_dim,
                'num_patches': self.num_patches,
                'precision': 'fp16' if self.use_fp16 else 'fp32'
            },
            'extraction_timestamp': datetime.now().isoformat()
        }
        
        if metadata:
            data['metadata'] = metadata
        
        torch.save(data, output_path)
        self.logger.info(f"Saved features to {output_path}")
    
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
    
    def compute_similarity_map(self, 
                               features: torch.Tensor,
                               query_idx: int) -> torch.Tensor:
        """
        Compute cosine similarity between a query patch and all other patches.
        
        Args:
            features: Patch features [N, D]
            query_idx: Index of query patch
            
        Returns:
            Similarity map [N]
        """
        # Normalize features
        features_norm = torch.nn.functional.normalize(features, dim=-1)
        
        # Get query feature
        query = features_norm[query_idx:query_idx+1]
        
        # Compute similarities
        similarities = torch.matmul(features_norm, query.T).squeeze()
        
        return similarities


class BatchDINOv3Extractor(DINOv3Extractor):
    """
    Batch processor for DINOv3 feature extraction on large datasets.
    """
    
    def __init__(self,
                 output_dir: Union[str, Path],
                 model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
                 device: Union[str, torch.device] = "cuda:0",
                 use_fp16: bool = True,
                 chunk_size: int = 1000):
        """
        Initialize batch extractor.
        
        Args:
            output_dir: Directory to save features
            model_name: DINOv3 model to use
            device: Device for computation
            use_fp16: Use half precision
            chunk_size: Number of images per chunk file
        """
        super().__init__(model_name, device, use_fp16)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.chunk_size = chunk_size
        
        # Progress tracking
        self.progress_file = self.output_dir / "extraction_progress.json"
    
    def process_directory(self, 
                         image_dir: Union[str, Path],
                         pattern: str = "*.jpg",
                         resume: bool = True):
        """
        Process all images in a directory.
        
        Args:
            image_dir: Directory containing images
            pattern: Glob pattern for image files
            resume: Whether to resume from previous progress
        """
        image_dir = Path(image_dir)
        
        # Get all image files
        image_files = sorted(image_dir.glob(pattern))
        self.logger.info(f"Found {len(image_files)} images to process")
        
        # Load progress if resuming
        processed_files = set()
        if resume and self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                progress = json.load(f)
                processed_files = set(progress.get('processed_files', []))
                self.logger.info(f"Resuming from {len(processed_files)} processed files")
        
        # Process in chunks
        chunk_buffer = []
        chunk_idx = len(processed_files) // self.chunk_size
        
        for img_path in tqdm(image_files, desc="Extracting features"):
            # Skip if already processed
            if str(img_path) in processed_files:
                continue
            
            try:
                # Extract features
                features = self.extract_features(img_path)
                
                # Add to buffer
                chunk_buffer.append({
                    'path': str(img_path),
                    'features': features
                })
                
                # Save chunk when buffer is full
                if len(chunk_buffer) >= self.chunk_size:
                    self._save_chunk(chunk_buffer, chunk_idx)
                    processed_files.update([item['path'] for item in chunk_buffer])
                    self._save_progress(list(processed_files))
                    chunk_buffer = []
                    chunk_idx += 1
                    
                    # Clear GPU cache
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                        gc.collect()
                    
            except Exception as e:
                self.logger.error(f"Error processing {img_path}: {e}")
                continue
        
        # Save remaining features
        if chunk_buffer:
            self._save_chunk(chunk_buffer, chunk_idx)
            processed_files.update([item['path'] for item in chunk_buffer])
            self._save_progress(list(processed_files))
        
        self.logger.info(f"Feature extraction complete! Processed {len(processed_files)} images")
    
    def _save_chunk(self, chunk_data: List[Dict], chunk_idx: int):
        """Save a chunk of features"""
        chunk_file = self.output_dir / f"features_chunk_{chunk_idx:04d}.pt"
        
        # Prepare chunk dictionary
        chunk_dict = {}
        for item in chunk_data:
            chunk_dict[item['path']] = item['features']
        
        # Add metadata
        chunk_dict['_metadata'] = {
            'chunk_idx': chunk_idx,
            'num_images': len(chunk_data),
            'model_name': self.model_name,
            'extraction_timestamp': datetime.now().isoformat()
        }
        
        torch.save(chunk_dict, chunk_file)
        self.logger.info(f"Saved chunk {chunk_idx} with {len(chunk_data)} images")
    
    def _save_progress(self, processed_files: List[str]):
        """Save processing progress"""
        progress = {
            'processed_files': processed_files,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f)


def main():
    parser = argparse.ArgumentParser(description='DINOv3 feature extraction')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save features')
    parser.add_argument('--model', type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m",
                       help='DINOv3 model name')
    parser.add_argument('--device', type=str, default="cuda:0",
                       help='Device to use')
    parser.add_argument('--batch_mode', action='store_true',
                       help='Use batch processing mode')
    parser.add_argument('--chunk_size', type=int, default=1000,
                       help='Images per chunk (batch mode)')
    parser.add_argument('--no_fp16', action='store_true',
                       help='Disable fp16 precision')
    parser.add_argument('--pattern', type=str, default="*.jpg",
                       help='Image file pattern')
    
    args = parser.parse_args()
    
    if args.batch_mode:
        # Batch processing
        extractor = BatchDINOv3Extractor(
            output_dir=args.output_dir,
            model_name=args.model,
            device=args.device,
            use_fp16=not args.no_fp16,
            chunk_size=args.chunk_size
        )
        extractor.process_directory(
            image_dir=args.image_dir,
            pattern=args.pattern
        )
    else:
        # Single directory processing
        extractor = DINOv3Extractor(
            model_name=args.model,
            device=args.device,
            use_fp16=not args.no_fp16
        )
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Process each image
        image_dir = Path(args.image_dir)
        for img_path in tqdm(list(image_dir.glob(args.pattern))):
            try:
                features = extractor.extract_features(img_path)
                output_path = output_dir / f"{img_path.stem}_features.pt"
                extractor.save_features(features, output_path, {'source_image': str(img_path)})
            except Exception as e:
                print(f"Error processing {img_path}: {e}")


if __name__ == "__main__":
    main()