#!/usr/bin/env python3
"""
Multimodal Autoencoder with Input Masking and MLP U-Net Architecture
- Uses existing DeepEarth data loading infrastructure
- MLP U-Net architecture with skip connections
- Masks at input level (30% language, 0% vision)
- Reconstructs in original dimensions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.optim.lr_scheduler import SequentialLR
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import json
import sys
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import os
import random
from tqdm import tqdm

# Add dashboard to path
dashboard_path = Path(__file__).parent.parent / "dashboard"
sys.path.insert(0, str(dashboard_path))

from services.training_data import get_training_batch, get_available_observation_ids
from data_cache import UnifiedDataCache

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class MLPBlock(nn.Module):
    """Basic MLP block with residual connection"""
    
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Residual connection if dimensions match
        self.residual = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        
    def forward(self, x):
        return self.net(x) + self.residual(x)


class MLPUNetEncoder(nn.Module):
    """MLP-based U-Net encoder with skip connections"""
    
    def __init__(self, input_dim, hidden_dims=[2048, 1024, 512], dropout=0.1):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        dims = [input_dim] + hidden_dims
        
        for i in range(len(hidden_dims)):
            self.blocks.append(MLPBlock(dims[i], dims[i+1], dropout))
        
    def forward(self, x):
        skip_connections = []
        
        for block in self.blocks:
            x = block(x)
            skip_connections.append(x)
        
        return x, skip_connections


class MLPUNetDecoder(nn.Module):
    """MLP-based U-Net decoder with skip connections"""
    
    def __init__(self, latent_dim, hidden_dims=[512, 1024, 2048], output_dim=None, dropout=0.1, skip_mode='concat', encoder_dims=None):
        super().__init__()
        
        self.skip_mode = skip_mode
        self.blocks = nn.ModuleList()
        dims = [latent_dim] + hidden_dims
        
        # Store encoder dimensions for proper skip connections
        if encoder_dims is None:
            # If not provided, assume symmetric architecture
            encoder_dims = hidden_dims[::-1]
        
        # Create decoder blocks that handle skip connections
        for i in range(len(hidden_dims)):
            if i == 0:
                # First block: only latent input
                in_dim = dims[i]
            else:
                # Later blocks: depends on skip mode
                if skip_mode == 'concat':
                    # Get the corresponding encoder dimension
                    skip_dim = encoder_dims[-i]  # -1, -2, -3, ... for i=1, 2, 3, ...
                    in_dim = dims[i] + skip_dim
                else:  # additive
                    in_dim = dims[i]
            out_dim = dims[i+1]
            self.blocks.append(MLPBlock(in_dim, out_dim, dropout))
        
        # Final projection
        self.output_proj = nn.Linear(hidden_dims[-1], output_dim) if output_dim else None
        
    def forward(self, x, skip_connections, mask=None):
        # Keep ALL skips, just reverse order
        skip_connections = skip_connections[::-1]
        
        for i, block in enumerate(self.blocks):
            if i > 0 and i - 1 < len(skip_connections):
                skip = skip_connections[i - 1]
                # Gate skip connections for masked samples during training
                # Keep the deepest encoder skip (last one) even when masked
                if self.training and mask is not None and skip.shape[0] == mask.shape[0]:
                    if i - 1 < len(skip_connections) - 1:  # Don't gate the deepest skip
                        skip = skip * (~mask).float().unsqueeze(-1)
                    
                if self.skip_mode == 'concat':
                    x = torch.cat([x, skip], dim=-1)
                else:  # additive
                    x = x + skip
            x = block(x)
        
        if self.output_proj:
            x = self.output_proj(x)
        
        return x


class CrossModalAttention(nn.Module):
    """Multi-head cross-modal attention for better fusion"""
    
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, dim * 2)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(0.1)
        )
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        
    def forward(self, queries, keys_values):
        B, D = queries.shape
        
        # Normalize inputs
        queries = self.norm_q(queries)
        keys_values = self.norm_kv(keys_values)
        
        # Create queries, keys, values
        q = self.to_q(queries).view(B, self.num_heads, self.head_dim)
        k, v = self.to_kv(keys_values).chunk(2, dim=-1)
        k = k.view(B, self.num_heads, self.head_dim)
        v = v.view(B, self.num_heads, self.head_dim)
        
        # Compute attention scores using einsum for clarity
        # (B, H, D) @ (B, H, D) -> (B, H)
        scores = torch.einsum('bhd,bhd->bh', q, k) * self.scale
        
        # Use sigmoid for gating instead of softmax over heads
        attn = scores.sigmoid().unsqueeze(-1)  # (B, H, 1)
        
        # Apply attention to values
        # (B, H, 1) * (B, H, D) -> (B, H, D)
        out = (attn * v).view(B, D)
        
        return self.to_out(out) + queries  # Residual


class SimpleFusion(nn.Module):
    """Simple MLP fusion for single-vector cross-modal interaction"""
    
    def __init__(self, dim):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),  # [v, l, |v-l|, v*l]
            nn.LayerNorm(dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )
    
    def forward(self, vision, language):
        # Compute interaction features
        diff = torch.abs(vision - language)
        prod = vision * language
        
        # Concatenate all features
        combined = torch.cat([vision, language, diff, prod], dim=-1)
        
        # Fuse and add residual
        fused = self.fusion(combined)
        return fused + vision, fused + language  # Residual connections


class MultimodalMLPUNet(nn.Module):
    """
    Multimodal autoencoder with MLP U-Net architecture
    - Input masking (not universal token masking)
    - Skip connections between encoder and decoder
    - Reconstruction in original dimensions
    """
    
    def __init__(self, vision_dim=1408, language_dim=7168, universal_dim=2048, use_simple_fusion=False):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.universal_dim = universal_dim
        
        # Vision U-Net - switch back to concat for proper skip connections
        self.vision_encoder = MLPUNetEncoder(
            input_dim=vision_dim,
            hidden_dims=[1408, 1792, 2048]  # Adjusted for 2048D universal
        )
        self.vision_decoder = MLPUNetDecoder(
            latent_dim=2048,  # Updated to 2048D
            hidden_dims=[1792, 1408, 1408],
            output_dim=self.vision_dim,  # Explicitly set output dimension
            skip_mode='concat',  # Concat mode to avoid dimension mismatches
            encoder_dims=[1408, 1792, 2048]  # Pass encoder dims for proper skip connections
        )
        
        # Language U-Net - switch back to concat
        self.language_encoder = MLPUNetEncoder(
            input_dim=language_dim,
            hidden_dims=[5120, 3584, 2048]  # Adjusted for 2048D universal
        )
        self.language_decoder = MLPUNetDecoder(
            latent_dim=2048,  # Updated to 2048D
            hidden_dims=[3584, 5120, 7168],
            output_dim=self.language_dim,  # Explicitly set output dimension
            skip_mode='concat',  # Concat mode
            encoder_dims=[5120, 3584, 2048]  # Pass encoder dims for proper skip connections
        )
        
        # Cross-modal fusion
        self.use_simple_fusion = use_simple_fusion
        if use_simple_fusion:
            self.cross_modal_fusion = SimpleFusion(universal_dim)
        else:
            self.cross_modal_attn = CrossModalAttention(universal_dim, num_heads=8)
        
        # Post-encoder normalization for stability
        self.post_enc_norm_v = nn.LayerNorm(universal_dim)
        self.post_enc_norm_l = nn.LayerNorm(universal_dim)
        
        # Learnable type embeddings for modality awareness
        self.vision_type_embedding = nn.Parameter(torch.randn(universal_dim) * 0.02)
        self.language_type_embedding = nn.Parameter(torch.randn(universal_dim) * 0.02)
        
        # Learnable mask tokens - initialize closer to typical embedding values
        self.vision_mask_token = nn.Parameter(torch.randn(vision_dim) * 0.02)
        self.language_mask_token = nn.Parameter(torch.randn(language_dim) * 0.02)
        
        # Add normalization layers for input embeddings
        self.vision_input_norm = nn.LayerNorm(vision_dim)
        self.language_input_norm = nn.LayerNorm(language_dim)
        
        # Initialize weights properly
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier/He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
    def forward(self, vision_emb, language_emb, vision_mask_ratio=0.0, language_mask_ratio=1.0, language_loss_weight=1.0):
        """
        Forward pass with U-Net skip connections
        
        Args:
            vision_emb: (B, 1408) for mean pooled, or (B, 8, 24, 24, 1408) for full
            language_emb: (B, 7168) mean pooled embeddings
            vision_mask_ratio: probability of masking vision (default 0)
            language_mask_ratio: probability of masking language (default 0.3)
        """
        B = vision_emb.shape[0]
        device = vision_emb.device
        
        # Handle vision embedding shape
        if vision_emb.dim() == 5:
            # Full vision embeddings - mean pool
            vision_emb = vision_emb.mean(dim=(1, 2, 3))  # (B, 1408)
        
        # Normalize inputs
        vision_emb = self.vision_input_norm(vision_emb)
        language_emb = self.language_input_norm(language_emb)
        
        # Log variance for debugging (only in training)
        if self.training and torch.rand(1).item() < 0.01:  # Log 1% of the time
            vision_var = vision_emb.var(dim=1).mean()
            language_var = language_emb.var(dim=1).mean()
            print(f"Input variances - Vision: {vision_var:.4f}, Language: {language_var:.4f}")
        
        # Store originals for reconstruction loss
        vision_original = vision_emb.clone()
        language_original = language_emb.clone()
        
        # Apply input masking
        if self.training:
            # Vision masking (not used in default training)
            vision_mask = torch.rand(B, device=device) < vision_mask_ratio
            vision_emb = torch.where(
                vision_mask.unsqueeze(1),
                self.vision_mask_token.unsqueeze(0).expand(B, -1),
                vision_emb
            )
            
            # Language masking
            language_mask = torch.rand(B, device=device) < language_mask_ratio
            language_emb = torch.where(
                language_mask.unsqueeze(1),
                self.language_mask_token.unsqueeze(0).expand(B, -1),
                language_emb
            )
        else:
            vision_mask = torch.zeros(B, dtype=torch.bool, device=device)
            language_mask = torch.zeros(B, dtype=torch.bool, device=device)
        
        # Encode with skip connections
        vision_latent, vision_skips = self.vision_encoder(vision_emb)  # (B, universal_dim)
        language_latent, language_skips = self.language_encoder(language_emb)  # (B, universal_dim)
        
        # Normalize latents for stability
        vision_latent = self.post_enc_norm_v(vision_latent)
        language_latent = self.post_enc_norm_l(language_latent)
        
        # Add type embeddings to distinguish modalities in shared space
        vision_latent = vision_latent + self.vision_type_embedding.unsqueeze(0)
        language_latent = language_latent + self.language_type_embedding.unsqueeze(0)
        
        # Cross-modal fusion
        if self.use_simple_fusion:
            vision_fused, language_fused = self.cross_modal_fusion(vision_latent, language_latent)
        else:
            # Cross-modal attention fusion (shared weights for symmetry)
            vision_fused = self.cross_modal_attn(vision_latent, language_latent)
            language_fused = self.cross_modal_attn(language_latent, vision_latent)
        
        # Decode with skip connections (pass masks to gate skips)
        vision_recon = self.vision_decoder(vision_fused, vision_skips, mask=vision_mask)
        language_recon = self.language_decoder(language_fused, language_skips, mask=language_mask)
        
        # Compute reconstruction losses in original spaces
        vision_loss = F.mse_loss(vision_recon, vision_original, reduction='none').mean(dim=1)
        language_loss = F.mse_loss(language_recon, language_original, reduction='none').mean(dim=1)
        
        # Simple batch mean for loss (100% masking makes per-sample normalization redundant)
        vision_loss = vision_loss.mean()
        language_loss = language_loss.mean()
        
        # Apply loss weighting to balance vision vs language
        total_loss = vision_loss + language_loss_weight * language_loss
        
        return {
            'loss': total_loss,
            'vision_loss': vision_loss,
            'language_loss': language_loss,
            'vision_recon': vision_recon,
            'language_recon': language_recon,
            'vision_latent': vision_latent,
            'language_latent': language_latent
        }


class DeepEarthStreamingDataset(IterableDataset):
    """Streaming dataset that loads on-demand from cache"""
    
    # Class-level cache storage to persist across epochs
    _worker_caches = {}
    _preloaded_data = {}  # Store preloaded embeddings
    
    def __init__(self, observation_ids: List[str], cache_config_path: str = "dataset_config.json", 
                 batch_size: int = 8, shared_cache=None, preload=True):
        self.observation_ids = observation_ids
        self.cache_config_path = cache_config_path
        self.batch_size = batch_size
        self.epoch = 0
        self.shared_cache = shared_cache  # For single-process mode
        self.preload = preload
        
        # Preload data if requested
        if preload and not self._is_data_preloaded():
            self._preload_all_data()
        
    def _is_data_preloaded(self):
        """Check if data is already preloaded"""
        return len(self._preloaded_data) > 0
    
    def _preload_all_data(self):
        """Preload all embeddings into memory for fast access"""
        logger.info(f"Preloading {len(self.observation_ids)} observations into memory...")
        
        # Initialize cache for preloading
        cache = UnifiedDataCache(self.cache_config_path)
        
        # Load in batches to manage memory
        batch_size = 64
        for i in tqdm(range(0, len(self.observation_ids), batch_size), desc="Preloading"):
            batch_ids = self.observation_ids[i:i + batch_size]
            
            try:
                batch_data = get_training_batch(
                    cache,
                    batch_ids,
                    include_vision=True,
                    include_language=True,
                    device='cpu'
                )
                
                # Store each observation's data
                for j, obs_id in enumerate(batch_ids):
                    self._preloaded_data[obs_id] = {
                        'vision_embedding': batch_data['vision_embeddings'][j],
                        'language_embedding': batch_data['language_embeddings'][j]
                    }
                    
            except Exception as e:
                logger.warning(f"Error preloading batch starting at {batch_ids[0]}: {e}")
        
        logger.info(f"Preloaded {len(self._preloaded_data)} observations")
        
    def set_epoch(self, epoch):
        """Set epoch for proper shuffling across epochs"""
        self.epoch = epoch
    
    def get_estimated_length(self):
        """Get estimated number of samples this dataset will yield"""
        return len(self.observation_ids)
    
    def _get_cache(self):
        """Get or create cache for current worker, reusing across epochs"""
        # If using preloaded data, no need for cache
        if self.preload and self._is_data_preloaded():
            return None
            
        worker_info = torch.utils.data.get_worker_info()
        
        # Single process mode - use shared cache if provided
        if worker_info is None:
            if self.shared_cache is not None:
                return self.shared_cache
            if -1 not in DeepEarthStreamingDataset._worker_caches:
                logger.info(f"Initializing cache for main process")
                DeepEarthStreamingDataset._worker_caches[-1] = UnifiedDataCache(self.cache_config_path)
            return DeepEarthStreamingDataset._worker_caches[-1]
        
        # Multi-process mode
        worker_id = worker_info.id
        if worker_id not in DeepEarthStreamingDataset._worker_caches:
            logger.info(f"Initializing cache for worker {worker_id}")
            DeepEarthStreamingDataset._worker_caches[worker_id] = UnifiedDataCache(self.cache_config_path)
        
        return DeepEarthStreamingDataset._worker_caches[worker_id]
        
    def __iter__(self):
        # Get worker info for distributed loading
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single process
            start_idx = 0
            end_idx = len(self.observation_ids)
            worker_id = 0
        else:
            # Multiple workers - split the data
            per_worker = int(np.ceil(len(self.observation_ids) / worker_info.num_workers))
            worker_id = worker_info.id
            start_idx = worker_id * per_worker
            end_idx = min(start_idx + per_worker, len(self.observation_ids))
        
        # Work with this worker's slice
        worker_ids = self.observation_ids[start_idx:end_idx]
        
        # Shuffle with epoch-specific seed for reproducibility
        shuffled_ids = worker_ids.copy()
        rng = random.Random(self.epoch + worker_id)
        rng.shuffle(shuffled_ids)
        
        # Use preloaded data if available
        if self.preload and self._is_data_preloaded():
            # Yield from preloaded data
            for obs_id in shuffled_ids:
                if obs_id in self._preloaded_data:
                    data = self._preloaded_data[obs_id]
                    yield {
                        'vision_embedding': data['vision_embedding'],
                        'language_embedding': data['language_embedding'],
                        'obs_id': obs_id
                    }
        else:
            # Original streaming approach
            cache = self._get_cache()
            
            # Stream in small batches
            for i in range(0, len(shuffled_ids), self.batch_size):
                batch_ids = shuffled_ids[i:i + self.batch_size]
                
                try:
                    # Load batch from cache
                    batch_data = get_training_batch(
                        cache,
                        batch_ids,
                        include_vision=True,
                        include_language=True,
                        device='cpu'  # Always load to CPU for DataLoader
                    )
                    
                    # Yield individual samples
                    for j in range(len(batch_ids)):
                        yield {
                            'vision_embedding': batch_data['vision_embeddings'][j],
                            'language_embedding': batch_data['language_embeddings'][j],
                            'obs_id': batch_ids[j]
                        }
                except Exception as e:
                    logger.warning(f"Error loading batch starting at {batch_ids[0]}: {e}")
                    continue


def visualize_embeddings(original, reconstructed, name, epoch, save_dir):
    """Visualize embeddings as 2D images (100% language mask run)"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Determine shape based on embedding size
    if original.shape[0] == 1408:
        # Vision: 44x32 = 1408 exactly
        h, w = 44, 32
        orig_2d = original.view(h, w).cpu().numpy()
        recon_2d = reconstructed.view(h, w).cpu().numpy()
    elif original.shape[0] == 7168:
        # Language: 64x112 = 7168 exactly
        h, w = 64, 112
        orig_2d = original.view(h, w).cpu().numpy()
        recon_2d = reconstructed.view(h, w).cpu().numpy()
    else:
        # Generic case
        size = original.shape[0]
        h = int(np.sqrt(size))
        w = size // h
        if h * w < size:
            h += 1
        orig_2d = original[:h*w].view(h, w).cpu().numpy()
        recon_2d = reconstructed[:h*w].view(h, w).cpu().numpy()
    
    # Normalize to [0, 1]
    orig_2d = (orig_2d - orig_2d.min()) / (orig_2d.max() - orig_2d.min() + 1e-8)
    recon_2d = (recon_2d - recon_2d.min()) / (recon_2d.max() - recon_2d.min() + 1e-8)
    
    # Plot
    im1 = axes[0].imshow(orig_2d, cmap='turbo', aspect='auto')
    axes[0].set_title(f'{name} Original')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    
    im2 = axes[1].imshow(recon_2d, cmap='turbo', aspect='auto')
    axes[1].set_title(f'{name} Reconstructed')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    plt.suptitle(f'Epoch {epoch}')
    plt.tight_layout()
    
    save_path = save_dir / f'{name.lower()}_epoch_{epoch}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_latent_space(model, loader, device, epoch, save_dir, max_batches=10):
    """Visualize latent space representations"""
    model.eval()
    
    vision_latents = []
    language_latents = []
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:  # Only use first N batches for visualization
                break
                
            vision = batch['vision_embedding'].to(device)
            language = batch['language_embedding'].to(device)
            
            # Get latent representations
            outputs = model(vision, language, vision_mask_ratio=0.0, language_mask_ratio=0.0)
            
            vision_latents.append(outputs['vision_latent'].cpu())
            language_latents.append(outputs['language_latent'].cpu())
    
    # Early return if no data
    if not vision_latents:
        return
    
    # Concatenate all latents
    vision_latents = torch.cat(vision_latents, dim=0).numpy()
    language_latents = torch.cat(language_latents, dim=0).numpy()
    
    # Plot latent distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Vision latent
    axes[0].hist(vision_latents.flatten(), bins=50, alpha=0.7, color='blue')
    axes[0].set_title('Vision Universal Embedding Distribution')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Count')
    
    # Language latent
    axes[1].hist(language_latents.flatten(), bins=50, alpha=0.7, color='green')
    axes[1].set_title('Language Universal Embedding Distribution')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Count')
    
    plt.suptitle(f'Universal Embedding Space Analysis - Epoch {epoch}')
    plt.tight_layout()
    
    save_path = save_dir / f'universal_space_epoch_{epoch}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def train_epoch(model, loader, optimizer, device, scaler=None, dataset=None, 
                vision_mask_ratio=0.0, language_mask_ratio_train=1.0, language_loss_weight=1.0):
    """Train for one epoch with optional mixed precision"""
    model.train()
    total_loss = 0
    total_vision_loss = 0
    total_language_loss = 0
    total_contrastive_loss = 0
    
    use_amp = scaler is not None
    
    # Estimate total batches if dataset is provided
    estimated_batches = None
    if dataset is not None and hasattr(dataset, 'get_estimated_length'):
        estimated_samples = dataset.get_estimated_length()
        batch_size = loader.batch_size
        estimated_batches = (estimated_samples + batch_size - 1) // batch_size
    
    for i, batch in enumerate(loader):
        try:
            # Move batch to device with pin_memory benefits
            vision = batch['vision_embedding'].to(device, non_blocking=True)
            language = batch['language_embedding'].to(device, non_blocking=True)
            
            # Forward pass with optional mixed precision
            if use_amp:
                from torch.cuda.amp import autocast
                # Use bfloat16 for H100 if supported
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                with autocast(dtype=dtype):
                    outputs = model(vision, language, 
                                  vision_mask_ratio=vision_mask_ratio, 
                                  language_mask_ratio=language_mask_ratio_train, 
                                  language_loss_weight=language_loss_weight)
                    loss = outputs['loss']
            else:
                outputs = model(vision, language, 
                              vision_mask_ratio=vision_mask_ratio, 
                              language_mask_ratio=language_mask_ratio_train,
                              language_loss_weight=language_loss_weight)
                loss = outputs['loss']
            
            # Check for NaN or extremely high loss
            if torch.isnan(loss) or loss.item() > 1e6:
                logger.warning(f"Abnormal loss detected at batch {i}: {loss.item()}")
                continue
            
            # Backward with optional mixed precision
            optimizer.zero_grad()
            
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            # Accumulate
            total_loss += loss.item()
            total_vision_loss += outputs['vision_loss'].item()
            total_language_loss += outputs['language_loss'].item()
            # Check if contrastive_loss exists in outputs
            if 'contrastive_loss' in outputs:
                total_contrastive_loss += outputs['contrastive_loss'].item()
                contrastive_str = f", C={outputs['contrastive_loss'].item():.4f}"
            else:
                contrastive_str = ""
            
            if i % 10 == 0:
                # Show progress with estimated total if available
                if estimated_batches:
                    progress_str = f"\r  Batch {i}/{estimated_batches} (est.)"
                else:
                    progress_str = f"\r  Batch {i}"
                    
                print(f"{progress_str}: Loss={loss.item():.4f}, "
                      f"V={outputs['vision_loss'].item():.4f}, "
                      f"L={outputs['language_loss'].item():.4f}{contrastive_str}", end='', flush=True)
                
        except RuntimeError as e:
            logger.error(f"Error at batch {i}: {e}")
            if "CUDA" in str(e):
                logger.error("CUDA error detected, trying to recover...")
                torch.cuda.empty_cache()
            raise
    
    print()
    
    # Count actual batches processed
    n_batches = i + 1
    return {
        'loss': total_loss / n_batches,
        'vision_loss': total_vision_loss / n_batches,
        'language_loss': total_language_loss / n_batches,
        'contrastive_loss': total_contrastive_loss / n_batches if total_contrastive_loss > 0 else 0.0
    }


def evaluate_comprehensive(model, train_loader, test_loader, device, epoch, save_dir, config_path, train_obs_ids, test_obs_ids):
    """
    Comprehensive evaluation including:
    1. Vision→Language retrieval @1, @5, @10
    2. Language→Vision retrieval @1, @5, @10
    3. Nearest neighbor analysis
    4. Classification using universal embeddings with actual species labels
    """
    model.eval()
    
    # Extract all embeddings and observation IDs
    train_vision_embeds = []
    train_language_embeds = []
    train_obs_ids_batch = []
    test_vision_embeds = []
    test_language_embeds = []
    test_obs_ids_batch = []
    
    with torch.no_grad():
        # Get train embeddings
        for batch in train_loader:
            vision = batch['vision_embedding'].to(device)
            language = batch['language_embedding'].to(device)
            outputs = model(vision, language, vision_mask_ratio=0.0, language_mask_ratio=0.0)
            train_vision_embeds.append(outputs['vision_latent'])
            train_language_embeds.append(outputs['language_latent'])
            train_obs_ids_batch.extend(batch['obs_id'])
        
        # Get test embeddings
        for batch in test_loader:
            vision = batch['vision_embedding'].to(device)
            language = batch['language_embedding'].to(device)
            outputs = model(vision, language, vision_mask_ratio=0.0, language_mask_ratio=0.0)
            test_vision_embeds.append(outputs['vision_latent'])
            test_language_embeds.append(outputs['language_latent'])
            test_obs_ids_batch.extend(batch['obs_id'])
    
    # Concatenate
    train_vision = F.normalize(torch.cat(train_vision_embeds, dim=0), p=2, dim=1)
    train_language = F.normalize(torch.cat(train_language_embeds, dim=0), p=2, dim=1)
    test_vision = F.normalize(torch.cat(test_vision_embeds, dim=0), p=2, dim=1)
    test_language = F.normalize(torch.cat(test_language_embeds, dim=0), p=2, dim=1)
    
    results = {}
    
    # Create ID to index mappings
    train_id_to_idx = {obs_id: idx for idx, obs_id in enumerate(train_obs_ids_batch)}
    test_id_to_idx = {obs_id: idx for idx, obs_id in enumerate(test_obs_ids_batch)}
    
    # 1. Vision→Language Retrieval
    print("\n  📊 Vision→Language Retrieval:")
    for split_name, vision_emb, language_emb, obs_ids, id_to_idx in [
        ("Train", train_vision, train_language, train_obs_ids_batch, train_id_to_idx), 
        ("Test", test_vision, test_language, test_obs_ids_batch, test_id_to_idx)
    ]:
        similarity = torch.mm(vision_emb, language_emb.t())
        
        # Get top-k accuracies
        for k in [1, 5, 10]:
            if k > len(vision_emb):
                continue
            _, indices = similarity.topk(k, dim=1)
            
            # Check if correct match is in top-k
            correct = 0
            for i, obs_id in enumerate(obs_ids):
                target_idx = id_to_idx[obs_id]  # The correct language index for this vision
                if target_idx in indices[i]:
                    correct += 1
                    
            acc = 100.0 * correct / len(vision_emb)
            results[f'{split_name}_V2L_R@{k}'] = acc
            print(f"    {split_name} R@{k}: {acc:.1f}%")
    
    # 2. Language→Vision Retrieval
    print("\n  📊 Language→Vision Retrieval:")
    for split_name, vision_emb, language_emb, obs_ids, id_to_idx in [
        ("Train", train_vision, train_language, train_obs_ids_batch, train_id_to_idx), 
        ("Test", test_vision, test_language, test_obs_ids_batch, test_id_to_idx)
    ]:
        similarity = torch.mm(language_emb, vision_emb.t())
        
        for k in [1, 5, 10]:
            if k > len(language_emb):
                continue
            _, indices = similarity.topk(k, dim=1)
            
            # Check if correct match is in top-k
            correct = 0
            for i, obs_id in enumerate(obs_ids):
                target_idx = id_to_idx[obs_id]  # The correct vision index for this language
                if target_idx in indices[i]:
                    correct += 1
                    
            acc = 100.0 * correct / len(language_emb)
            results[f'{split_name}_L2V_R@{k}'] = acc
            print(f"    {split_name} R@{k}: {acc:.1f}%")
    
    # 3. Species Classification Probe
    print("\n  🎯 Species Classification Probe:")
    
    # Load species labels
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    observation_mappings = config['observation_mappings']
    
    # Create species to index mapping
    species_to_idx = {}
    train_species_labels = []
    test_species_labels = []
    
    for obs_id in train_obs_ids_batch:
        species = observation_mappings[obs_id]['taxon_name']
        if species not in species_to_idx:
            species_to_idx[species] = len(species_to_idx)
        train_species_labels.append(species_to_idx[species])
    
    for obs_id in test_obs_ids_batch:
        species = observation_mappings[obs_id]['taxon_name']
        if species not in species_to_idx:
            species_to_idx[species] = len(species_to_idx)
        test_species_labels.append(species_to_idx[species])
    
    train_species_labels = torch.tensor(train_species_labels).to(device)
    test_species_labels = torch.tensor(test_species_labels).to(device)
    
    # Train a simple linear classifier on universal embeddings
    num_species = len(species_to_idx)
    classifier = nn.Linear(train_vision.shape[1], num_species).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
    
    # Quick training with frozen backbone
    model.eval()
    for _ in range(100):
        logits = classifier(train_vision.detach())
        loss = F.cross_entropy(logits, train_species_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluate
    with torch.no_grad():
        train_logits = classifier(train_vision)
        train_pred = train_logits.argmax(dim=1)
        train_acc = (train_pred == train_species_labels).float().mean() * 100
        
        if len(test_vision) > 0:
            test_logits = classifier(test_vision)
            test_pred = test_logits.argmax(dim=1)
            test_acc = (test_pred == test_species_labels).float().mean() * 100
            print(f"    Train accuracy: {train_acc:.1f}% ({num_species} species)")
            print(f"    Test accuracy: {test_acc:.1f}%")
            results['train_species_acc'] = train_acc.item()
            results['test_species_acc'] = test_acc.item()
    
    # 4. Nearest Neighbor Analysis
    print("\n  🔍 Nearest Neighbor Analysis (Test set):")
    if len(test_vision) > 5:
        # For first 5 test samples, find nearest neighbors in train set
        test_to_train_sim = torch.mm(test_vision[:5], train_vision.t())
        
        for i in range(min(5, len(test_vision))):
            _, nn_indices = test_to_train_sim[i].topk(3)
            test_species = observation_mappings[test_obs_ids_batch[i]]['taxon_name']
            nn_species = [observation_mappings[train_obs_ids_batch[idx]]['taxon_name'] for idx in nn_indices.cpu().tolist()]
            print(f"    Test {test_species} → NN: {nn_species}")
    
    # 5. Embedding Statistics
    print("\n  📈 Embedding Statistics:")
    print(f"    Train vision embedding mean norm: {train_vision.norm(dim=1).mean():.3f}")
    print(f"    Train language embedding mean norm: {train_language.norm(dim=1).mean():.3f}")
    print(f"    Test vision embedding mean norm: {test_vision.norm(dim=1).mean():.3f}")
    print(f"    Test language embedding mean norm: {test_language.norm(dim=1).mean():.3f}")
    
    # Save detailed results
    results_path = save_dir / f'evaluation_epoch_{epoch}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


@torch.no_grad()
def evaluate(model, loader, device, epoch=0, visualize=False, save_dir=None,
             vision_mask_ratio=0.0, language_mask_ratio_eval=1.0, language_loss_weight=1.0):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    total_vision_loss = 0
    total_language_loss = 0
    total_contrastive_loss = 0
    has_contrastive = False
    n_batches = 0
    
    # Store first batch for visualization
    first_batch_outputs = None
    
    for batch in loader:
        # Move batch to device
        vision = batch['vision_embedding'].to(device)
        language = batch['language_embedding'].to(device)
        
        # Forward pass (evaluate with same masking as training)
        outputs = model(vision, language, 
                       vision_mask_ratio=vision_mask_ratio, 
                       language_mask_ratio=language_mask_ratio_eval, 
                       language_loss_weight=language_loss_weight)
        
        total_loss += outputs['loss'].item()
        total_vision_loss += outputs['vision_loss'].item()
        total_language_loss += outputs['language_loss'].item()
        
        # Check if contrastive loss exists
        if 'contrastive_loss' in outputs:
            total_contrastive_loss += outputs['contrastive_loss'].item()
            has_contrastive = True
        
        if first_batch_outputs is None:
            first_batch_outputs = {
                'vision_original': vision[0].mean(dim=(0, 1, 2)) if vision.dim() == 5 else vision[0],
                'language_original': language[0],
                'vision_recon': outputs['vision_recon'][0],
                'language_recon': outputs['language_recon'][0]
            }
        
        n_batches += 1
    
    # Visualize if requested
    if visualize and first_batch_outputs is not None and save_dir is not None:
        visualize_embeddings(
            first_batch_outputs['vision_original'],
            first_batch_outputs['vision_recon'],
            'Vision',
            epoch,
            save_dir
        )
        visualize_embeddings(
            first_batch_outputs['language_original'],
            first_batch_outputs['language_recon'],
            'Language',
            epoch,
            save_dir
        )
        
        # Also visualize latent space
        visualize_latent_space(model, loader, device, epoch, save_dir)
    
    # Avoid division by zero
    if n_batches == 0:
        return {
            'loss': 0.0,
            'vision_loss': 0.0,
            'language_loss': 0.0,
            'contrastive_loss': 0.0
        }
    
    result = {
        'loss': total_loss / n_batches,
        'vision_loss': total_vision_loss / n_batches,
        'language_loss': total_language_loss / n_batches,
    }
    
    if has_contrastive:
        result['contrastive_loss'] = total_contrastive_loss / n_batches
    
    return result


def load_splits(config_path):
    """Load train/test observation IDs from config file"""
    logger.info(f"Loading splits from {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    observation_mappings = config['observation_mappings']
    
    # Extract train and test observation IDs
    train_obs_ids = [obs_id for obs_id, metadata in observation_mappings.items() 
                     if metadata['split'] == 'train']
    test_obs_ids = [obs_id for obs_id, metadata in observation_mappings.items() 
                    if metadata['split'] == 'test']
    
    logger.info(f"Loaded {len(train_obs_ids)} train and {len(test_obs_ids)} test observations")
    
    # Log some statistics
    test_species = set([metadata['taxon_name'] for metadata in observation_mappings.values() 
                       if metadata['split'] == 'test'])
    train_species = set([metadata['taxon_name'] for metadata in observation_mappings.values() 
                        if metadata['split'] == 'train'])
    
    logger.info(f"Species in test set: {len(test_species)}")
    logger.info(f"Species in train set: {len(train_species)}")
    logger.info(f"Common species: {len(train_species & test_species)}")
    
    return train_obs_ids, test_obs_ids


def custom_collate_fn(batch):
    """Custom collate function to properly stack tensors from streaming dataset"""
    vision_embeddings = torch.stack([item['vision_embedding'] for item in batch])
    language_embeddings = torch.stack([item['language_embedding'] for item in batch])
    obs_ids = [item['obs_id'] for item in batch]
    
    return {
        'vision_embedding': vision_embeddings,
        'language_embedding': language_embeddings,
        'obs_id': obs_ids
    }


def main():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 64  # Increased for H100
    epochs = 100  # Increased epochs for harder task
    lr = 1e-4  # Lower learning rate for harder optimization
    num_workers = 4
    universal_dim = 2048  # Updated to match boss's specification
    
    # Masking and loss configuration
    vision_mask_ratio = 0.0
    language_mask_ratio_train = 1.0
    language_mask_ratio_eval = 1.0
    language_loss_weight = 1.0  # Equal weight for baseline
    
    # H100 optimizations
    if device == 'cuda':
        # Enable Flash Attention if available (PyTorch 2.0+)
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
            logger.info("Flash Attention enabled for H100")
        
        # Log GPU info
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {gpu_name}")
    
    logger.info(f"Using device: {device}")
    
    # Set CUDA environment variables to help with initialization
    if device == 'cuda':
        # Set deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Change to dashboard directory for cache
    original_dir = os.getcwd()
    os.chdir(dashboard_path)
    
    try:
        # Load splits from config file
        config_path = Path(__file__).parent / "config" / "central_florida_split.json"
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return
            
        train_obs_ids, test_obs_ids = load_splits(config_path)
        logger.info(f"Loaded splits: {len(train_obs_ids)} train, {len(test_obs_ids)} test")
        
        # Create a single shared cache for single-process loaders
        shared_cache = UnifiedDataCache("dataset_config.json")
        logger.info("Created shared cache for single-process data loading")
        
        # Create streaming datasets
        logger.info("Creating streaming datasets...")
        
        # Determine whether to preload based on dataset size
        preload_train = len(train_obs_ids) < 10000  # Preload if < 10k samples
        preload_test = True  # Always preload test set (smaller)
        
        train_dataset = DeepEarthStreamingDataset(
            observation_ids=train_obs_ids, 
            cache_config_path="dataset_config.json",
            batch_size=8,
            shared_cache=None,  # Let workers create their own caches
            preload=preload_train
        )
        test_dataset = DeepEarthStreamingDataset(
            observation_ids=test_obs_ids, 
            cache_config_path="dataset_config.json",
            batch_size=8,
            shared_cache=shared_cache,  # Use shared cache for test set
            preload=preload_test
        )
        
        if preload_train:
            logger.info(f"Train data preloaded into memory ({len(train_obs_ids)} samples)")
        else:
            logger.info(f"Train data will stream from disk ({len(train_obs_ids)} samples)")
            
        logger.info(f"Test data preloaded into memory ({len(test_obs_ids)} samples)")
        
        # Create loaders with proper settings for streaming
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=(device == 'cuda'),
            persistent_workers=(num_workers > 0),  # Only use persistent workers if num_workers > 0
            collate_fn=custom_collate_fn,  # Custom collate for proper tensor stacking
            prefetch_factor=2 if num_workers > 0 else None,  # Reduce prefetch to avoid memory issues
            multiprocessing_context='spawn' if num_workers > 0 else None  # Use spawn to avoid fork issues
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=min(batch_size, len(test_obs_ids)),
            num_workers=min(2, num_workers),  # Fewer workers for test set
            pin_memory=(device == 'cuda'),
            persistent_workers=(num_workers > 0),
            collate_fn=custom_collate_fn,
            prefetch_factor=2 if num_workers > 0 else None,
            multiprocessing_context='spawn' if num_workers > 0 else None
        )
        
        # Create model
        model = MultimodalMLPUNet(universal_dim=universal_dim).to(device)
        
        # Check and scale type embeddings if needed
        with torch.no_grad():
            vision_type_norm = model.vision_type_embedding.norm().item()
            language_type_norm = model.language_type_embedding.norm().item()
            print(f"Initial type embedding norms - Vision: {vision_type_norm:.3f}, Language: {language_type_norm:.3f}")
            
            # Scale down if too large
            if vision_type_norm > 10 or language_type_norm > 10:
                scale_factor = 0.1
                model.vision_type_embedding.data *= scale_factor
                model.language_type_embedding.data *= scale_factor
                print(f"Scaled type embeddings by {scale_factor}")
        
        # Optimizer with proper learning rate
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)  # Reduced weight decay
        
        # Sequential scheduler for smooth transitions
        warmup_epochs = 3  # Faster warmup
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs-warmup_epochs, eta_min=1e-6
        )
        scheduler = SequentialLR(
            optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
        )
        
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Setup visualization directory
        viz_dir = Path('visualizations')
        viz_dir.mkdir(exist_ok=True)
        
        # Create timestamped checkpoint name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f'multimodal_mlp_unet_best_{timestamp}.pth'
        
        print("\n" + "="*80)
        print("🚀 MULTIMODAL AUTOENCODER WITH MLP U-NET TRAINING")
        print("="*80)
        print(f"Architecture: MLP U-Net with concatenative skip connections")
        print(f"Universal embedding dimension: {universal_dim}")
        print(f"Cross-modal fusion: Multi-head attention (limited to mean-pooled)")
        print(f"Masking: {int(language_mask_ratio_train*100)}% Language, {int(vision_mask_ratio*100)}% Vision")
        print(f"Language loss weight: {language_loss_weight} (baseline: equal weights)")
        print(f"Learning rate: {lr} with {warmup_epochs} warmup epochs (start: {lr * 0.1})")
        print(f"Batch size: {batch_size}")
        print(f"Dataset: {len(train_obs_ids)} train, {len(test_obs_ids)} test samples")
        print(f"Estimated batches per epoch: {(len(train_obs_ids) + batch_size - 1) // batch_size}")
        if device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Mixed Precision: Enabled ({'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'})")
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                print(f"Flash Attention: Enabled")
        else:
            print(f"Device: CPU (Mixed Precision disabled)")
        print(f"Checkpoint: {checkpoint_name}")
        print("="*80 + "\n")
        
        best_loss = float('inf')
        
        # Initialize mixed precision scaler if using CUDA
        scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda')) if device == 'cuda' else None
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 40)
            
            # Set epoch for proper shuffling
            train_dataset.set_epoch(epoch)
            test_dataset.set_epoch(epoch)
            
            # Train - pass the dataset to train_epoch
            train_metrics = train_epoch(model, train_loader, optimizer, device, scaler, train_dataset, 
                                      vision_mask_ratio=vision_mask_ratio,
                                      language_mask_ratio_train=language_mask_ratio_train,
                                      language_loss_weight=language_loss_weight)
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Vision: {train_metrics['vision_loss']:.4f}, "
                  f"Language: {train_metrics['language_loss']:.4f}")
            
            # Evaluate with both masked and unmasked settings
            visualize = (epoch % 5 == 0)
            
            # Masked evaluation (matches training)
            test_metrics_masked = evaluate(model, test_loader, device, epoch, visualize, viz_dir,
                                         vision_mask_ratio=vision_mask_ratio,
                                         language_mask_ratio_eval=language_mask_ratio_eval,
                                         language_loss_weight=language_loss_weight)
            
            # Unmasked evaluation (pure reconstruction capacity)
            with torch.no_grad():
                test_metrics_unmasked = evaluate(model, test_loader, device, epoch, False, None,
                                               vision_mask_ratio=0.0,
                                               language_mask_ratio_eval=0.0,
                                               language_loss_weight=language_loss_weight)
            
            print(f"Test (Masked)   - Loss: {test_metrics_masked['loss']:.4f}, "
                  f"Vision: {test_metrics_masked['vision_loss']:.4f}, "
                  f"Language: {test_metrics_masked['language_loss']:.4f}")
            print(f"Test (Unmasked) - Loss: {test_metrics_unmasked['loss']:.4f}, "
                  f"Vision: {test_metrics_unmasked['vision_loss']:.4f}, "
                  f"Language: {test_metrics_unmasked['language_loss']:.4f}")
            
            # Comprehensive evaluation every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                print("\n🔬 Running comprehensive evaluation...")
                eval_results = evaluate_comprehensive(
                    model, train_loader, test_loader, device, epoch, viz_dir, config_path, train_obs_ids, test_obs_ids
                )
                print()
            
            # Update scheduler
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Learning rate: {current_lr:.6f}")
            
            # Save best model based on masked evaluation
            if test_metrics_masked['loss'] < best_loss:
                best_loss = test_metrics_masked['loss']
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'test_loss_masked': test_metrics_masked['loss'],
                    'test_loss_unmasked': test_metrics_unmasked['loss'],
                }, checkpoint_name)
                print(f"  ✓ Saved best model to {checkpoint_name}!")
        
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE!")
        print(f"Best test loss: {best_loss:.4f}")
        print(f"Model saved as: {checkpoint_name}")
        print("="*80)
        
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    main()
