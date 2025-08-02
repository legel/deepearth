"""
Simple multimodal autoencoder with dataset-specific decoders.

This model:
1. Takes mean vision (1408D) and language (7168D) embeddings
2. Masks language embeddings (100%) during training
3. Encodes both to universal space (2048D)
4. Processes through shared backbone
5. Decodes back to original dimensions
6. Computes MSE loss in original spaces
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import os
from tqdm import tqdm
import sys
from PIL import Image
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultimodalAutoencoder(nn.Module):
    """
    Multimodal autoencoder that masks raw embeddings and reconstructs in original spaces
    """
    def __init__(self, vision_dim=1408, language_dim=7168, universal_dim=2048):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.universal_dim = universal_dim
        
        # Encoders: dataset-specific to universal
        self.vision_encoder = nn.Sequential(
            nn.Linear(vision_dim, universal_dim),
            nn.LayerNorm(universal_dim),
            nn.ReLU()
        )
        
        self.language_encoder = nn.Sequential(
            nn.Linear(language_dim, universal_dim),
            nn.LayerNorm(universal_dim),
            nn.ReLU()
        )
        
        # Shared multimodal backbone
        self.backbone = MultimodalBackbone(universal_dim)
        
        # Decoders: universal to dataset-specific
        self.vision_decoder = nn.Sequential(
            nn.Linear(universal_dim, vision_dim // 2),
            nn.LayerNorm(vision_dim // 2),
            nn.ReLU(),
            nn.Linear(vision_dim // 2, vision_dim)
        )
        
        self.language_decoder = nn.Sequential(
            nn.Linear(universal_dim, language_dim // 2),
            nn.LayerNorm(language_dim // 2),
            nn.ReLU(),
            nn.Linear(language_dim // 2, language_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, vision_emb, language_emb, vision_mask_ratio=0.0, language_mask_ratio=1.0):
        """
        Args:
            vision_emb: (B, 1408) - Mean-pooled vision embeddings
            language_emb: (B, 7168) - Mean-pooled language embeddings
            vision_mask_ratio: Fraction of vision to mask (default 0.0)
            language_mask_ratio: Fraction of language to mask (default 1.0 = 100%)
        """
        B = vision_emb.shape[0]
        device = vision_emb.device
        
        # Store originals for loss computation
        vision_original = vision_emb.clone()
        language_original = language_emb.clone()
        
        # Apply masking to RAW EMBEDDINGS (not universal space!)
        vision_masked = vision_emb.clone()
        language_masked = language_emb.clone()
        
        # Language masking - replace with zeros (100% masking)
        if self.training and language_mask_ratio > 0:
            language_mask = torch.rand(B, device=device) < language_mask_ratio
            language_masked[language_mask] = 0
            masked_count = language_mask.sum().item()
            logger.debug(f"Masked {masked_count}/{B} language samples ({masked_count/B*100:.1f}%)")
        
        # Vision masking (usually 0)
        if self.training and vision_mask_ratio > 0:
            vision_mask = torch.rand(B, device=device) < vision_mask_ratio
            vision_masked[vision_mask] = 0
            logger.debug(f"Masked {vision_mask.sum().item()}/{B} vision samples")
        
        # Encode to universal space
        vision_universal = self.vision_encoder(vision_masked)      # (B, 2048)
        language_universal = self.language_encoder(language_masked) # (B, 2048)
        
        # Process through multimodal backbone
        vision_processed, language_processed = self.backbone(vision_universal, language_universal)
        
        # Decode back to original spaces
        vision_reconstructed = self.vision_decoder(vision_processed)      # (B, 1408)
        language_reconstructed = self.language_decoder(language_processed) # (B, 7168)
        
        # Compute losses in ORIGINAL embedding spaces
        vision_loss = F.mse_loss(vision_reconstructed, vision_original)
        language_loss = F.mse_loss(language_reconstructed, language_original)
        
        # Only use vision loss for training (language is 100% masked)
        total_loss = vision_loss
        
        return {
            'loss': total_loss,
            'vision_loss': vision_loss,
            'language_loss': language_loss,
            'vision_recon': vision_reconstructed,
            'language_recon': language_reconstructed,
            'vision_original': vision_original,
            'language_original': language_original,
            'vision_universal': vision_processed,  # Universal representation after backbone
            'language_universal': language_processed  # This will be used for species classification
        }


class MultimodalBackbone(nn.Module):
    """
    U-Net backbone that processes vision and language as channels in universal space
    """
    def __init__(self, dim=2048):
        super().__init__()
        self.dim = dim
        
        # U-Net encoder path
        self.enc1 = self._make_encoder_block(2, 8, dim)      # (B, 2, 2048) -> (B, 8, 2048)
        self.enc2 = self._make_encoder_block(8, 16, dim)     # (B, 8, 2048) -> (B, 16, 2048)
        self.enc3 = self._make_encoder_block(16, 32, dim)    # (B, 16, 2048) -> (B, 32, 2048)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # U-Net decoder path (with skip connections)
        self.dec3 = self._make_decoder_block(64 + 32, 32, dim)  # Concat with enc3
        self.dec2 = self._make_decoder_block(32 + 16, 16, dim)  # Concat with enc2
        self.dec1 = self._make_decoder_block(16 + 8, 8, dim)    # Concat with enc1
        
        # Final projection back to 2 channels
        self.final = nn.Sequential(
            nn.Conv1d(8, 2, kernel_size=1),
            nn.GroupNorm(1, 2),
            nn.GELU()
        )
        
        # Output projections to separate vision and language
        self.vision_out = nn.Linear(dim, dim)
        self.language_out = nn.Linear(dim, dim)
        
    def _make_encoder_block(self, in_channels, out_channels, dim):
        """Create an encoder block with conv1d operations"""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU(),
            nn.Dropout(0.1)
        )
    
    def _make_decoder_block(self, in_channels, out_channels, dim):
        """Create a decoder block with conv1d operations"""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, vision_universal, language_universal):
        """
        Process vision and language embeddings using U-Net architecture
        
        Args:
            vision_universal: (B, 2048)
            language_universal: (B, 2048)
            
        Returns:
            vision_out: (B, 2048)
            language_out: (B, 2048)
        """
        B = vision_universal.shape[0]
        
        # Stack vision and language as channels: (B, 2, 2048)
        x = torch.stack([vision_universal, language_universal], dim=1)
        
        # Encoder path with skip connections
        enc1_out = self.enc1(x)           # (B, 8, 2048)
        enc2_out = self.enc2(enc1_out)    # (B, 16, 2048)
        enc3_out = self.enc3(enc2_out)    # (B, 32, 2048)
        
        # Bottleneck
        x = self.bottleneck(enc3_out)     # (B, 64, 2048)
        
        # Decoder path with skip connections
        x = torch.cat([x, enc3_out], dim=1)  # (B, 96, 2048)
        x = self.dec3(x)                      # (B, 32, 2048)
        
        x = torch.cat([x, enc2_out], dim=1)  # (B, 48, 2048)
        x = self.dec2(x)                      # (B, 16, 2048)
        
        x = torch.cat([x, enc1_out], dim=1)  # (B, 24, 2048)
        x = self.dec1(x)                      # (B, 8, 2048)
        
        # Final projection back to 2 channels
        x = self.final(x)  # (B, 2, 2048)
        
        # Split channels and apply output projections
        vision_out = self.vision_out(x[:, 0, :])      # (B, 2048)
        language_out = self.language_out(x[:, 1, :])  # (B, 2048)
        
        return vision_out, language_out


def visualize_embeddings(original, reconstructed, name, epoch, save_dir):
    """Visualize embeddings as 2D images"""
    # Determine shape based on embedding size
    embedding_size = original.shape[0]
    if embedding_size == 1408:
        # Vision embedding - roughly square
        h, w = 32, 44  # 32 * 44 = 1408
    elif embedding_size == 7168:
        # Language embedding - roughly square
        h, w = 64, 112  # 64 * 112 = 7168
    else:
        # Generic case
        h = int(np.sqrt(embedding_size))
        w = embedding_size // h
        if h * w < embedding_size:
            w += 1
    
    # Calculate MSE
    mse = F.mse_loss(original, reconstructed).item()
    
    # Reshape tensors
    def reshape_to_2d(tensor, h, w):
        flat = tensor.cpu().numpy()
        # Pad if necessary
        if len(flat) < h * w:
            flat = np.pad(flat, (0, h * w - len(flat)), mode='constant')
        return flat[:h * w].reshape(h, w)
    
    orig_2d = reshape_to_2d(original, h, w)
    recon_2d = reshape_to_2d(reconstructed, h, w)
    
    # Enhanced normalization with percentile clipping
    def normalize(arr, percentile_clip=2):
        # Clip extreme values to enhance contrast in the middle range
        low = np.percentile(arr, percentile_clip)
        high = np.percentile(arr, 100 - percentile_clip)
        arr_clipped = np.clip(arr, low, high)
        
        min_val = arr_clipped.min()
        max_val = arr_clipped.max()
        if max_val > min_val:
            return (arr_clipped - min_val) / (max_val - min_val)
        return arr * 0 + 0.5
    
    orig_norm = normalize(orig_2d)
    recon_norm = normalize(recon_2d)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original - using plasma colormap for better contrast
    im1 = axes[0].imshow(orig_norm, cmap='plasma', aspect='auto')
    axes[0].set_title(f'{name} Original\nShape: {h}×{w}')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    
    # Reconstructed - using plasma colormap
    im2 = axes[1].imshow(recon_norm, cmap='plasma', aspect='auto')
    axes[1].set_title(f'{name} Reconstructed\nMSE: {mse:.6f}')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    # Enhanced difference visualization - showing signed differences
    diff_signed = recon_norm - orig_norm
    im3 = axes[2].imshow(diff_signed, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
    axes[2].set_title('Signed Difference\n(Blue=Under, Red=Over)')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046)
    
    plt.suptitle(f'{name} Embeddings - Epoch {epoch}', fontsize=16)
    plt.tight_layout()
    
    # Save
    save_path = save_dir / f'{name.lower()}_epoch_{epoch:03d}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def compare_language_reconstructions(vis_samples, epoch, save_dir):
    """Compare multiple language reconstructions to check if they're all the same"""
    n_samples = min(len(vis_samples), 6)
    
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 3, 6))
    
    # Reshape language embeddings
    h, w = 64, 112  # for 7168D
    
    def reshape_and_normalize(tensor):
        flat = tensor.cpu().numpy()
        if len(flat) < h * w:
            flat = np.pad(flat, (0, h * w - len(flat)), mode='constant')
        reshaped = flat[:h * w].reshape(h, w)
        min_val, max_val = reshaped.min(), reshaped.max()
        if max_val > min_val:
            return (reshaped - min_val) / (max_val - min_val)
        return reshaped * 0 + 0.5
    
    # Plot originals and reconstructions
    for i in range(n_samples):
        # Original
        orig = reshape_and_normalize(vis_samples[i]['language_original'])
        axes[0, i].imshow(orig, cmap='turbo', aspect='auto')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Reconstruction
        recon = reshape_and_normalize(vis_samples[i]['language_recon'])
        axes[1, i].imshow(recon, cmap='turbo', aspect='auto')
        mse = F.mse_loss(vis_samples[i]['language_original'], 
                        vis_samples[i]['language_recon']).item()
        axes[1, i].set_title(f'Recon {i+1}\nMSE: {mse:.4f}')
        axes[1, i].axis('off')
    
    plt.suptitle(f'Language Reconstruction Comparison - Epoch {epoch}\n' + 
                 'check if model is outputting mean', 
                 fontsize=14)
    plt.tight_layout()
    
    save_path = save_dir / f'language_comparison_epoch_{epoch:03d}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also compute and print statistics
    recon_tensors = torch.stack([s['language_recon'] for s in vis_samples[:n_samples]])
    orig_tensors = torch.stack([s['language_original'] for s in vis_samples[:n_samples]])
    
    print(f"\nLanguage Reconstruction Statistics (Epoch {epoch}):")
    print(f"Std between originals: {orig_tensors.std(dim=0).mean():.6f}")
    print(f"Std between reconstructions: {recon_tensors.std(dim=0).mean():.6f}")
    print(f"Mean correlation between reconstructions: {torch.corrcoef(recon_tensors.view(n_samples, -1)).mean():.6f}")
    
    return recon_tensors.std(dim=0).mean().item()


def compare_vision_reconstructions(vis_samples, epoch, save_dir):
    """Compare multiple vision reconstructions to check diversity"""
    n_samples = min(len(vis_samples), 6)
    
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 3, 6))
    
    # Reshape vision embeddings
    h, w = 32, 44  # for 1408D
    
    def reshape_and_normalize(tensor, percentile_clip=2):
        flat = tensor.cpu().numpy()
        if len(flat) < h * w:
            flat = np.pad(flat, (0, h * w - len(flat)), mode='constant')
        reshaped = flat[:h * w].reshape(h, w)
        
        # Enhanced normalization with percentile clipping
        low = np.percentile(reshaped, percentile_clip)
        high = np.percentile(reshaped, 100 - percentile_clip)
        reshaped_clipped = np.clip(reshaped, low, high)
        
        min_val, max_val = reshaped_clipped.min(), reshaped_clipped.max()
        if max_val > min_val:
            return (reshaped_clipped - min_val) / (max_val - min_val)
        return reshaped * 0 + 0.5
    
    # Plot originals and reconstructions with plasma colormap
    for i in range(n_samples):
        # Original
        orig = reshape_and_normalize(vis_samples[i]['vision_original'])
        axes[0, i].imshow(orig, cmap='plasma', aspect='auto')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Reconstruction
        recon = reshape_and_normalize(vis_samples[i]['vision_recon'])
        axes[1, i].imshow(recon, cmap='plasma', aspect='auto')
        mse = F.mse_loss(vis_samples[i]['vision_original'], 
                        vis_samples[i]['vision_recon']).item()
        axes[1, i].set_title(f'Recon {i+1}\nMSE: {mse:.4f}')
        axes[1, i].axis('off')
    
    plt.suptitle(f'Vision Reconstruction Comparison - Epoch {epoch}\n' + 
                 'Vision samples', 
                 fontsize=14)
    plt.tight_layout()
    
    save_path = save_dir / f'vision_comparison_epoch_{epoch:03d}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create enhanced difference visualization
    fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 3, 3))
    
    for i in range(n_samples):
        orig_flat = vis_samples[i]['vision_original'].cpu().numpy()
        recon_flat = vis_samples[i]['vision_recon'].cpu().numpy()
        
        # Reshape for visualization
        orig_2d = orig_flat[:h*w].reshape(h, w)
        recon_2d = recon_flat[:h*w].reshape(h, w)
        
        # Normalize with clipping
        orig_norm = reshape_and_normalize(vis_samples[i]['vision_original'])
        recon_norm = reshape_and_normalize(vis_samples[i]['vision_recon'])
        
        # Compute signed difference
        diff_signed = recon_norm - orig_norm
        
        # Plot with enhanced colormap
        im = axes[i].imshow(diff_signed, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
        mse = F.mse_loss(vis_samples[i]['vision_original'], 
                        vis_samples[i]['vision_recon']).item()
        axes[i].set_title(f'Sample {i+1}\nMSE: {mse:.4f}')
        axes[i].axis('off')
        
        # Add colorbar for the last subplot
        if i == n_samples - 1:
            plt.colorbar(im, ax=axes[i], fraction=0.046)
    
    plt.suptitle(f'Vision Reconstruction Errors - Epoch {epoch}\n' + 
                 'Blue = Underestimated, Red = Overestimated', 
                 fontsize=14)
    plt.tight_layout()
    
    save_path = save_dir / f'vision_errors_epoch_{epoch:03d}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also compute and print statistics
    recon_tensors = torch.stack([s['vision_recon'] for s in vis_samples[:n_samples]])
    orig_tensors = torch.stack([s['vision_original'] for s in vis_samples[:n_samples]])
    
    print(f"\nVision Reconstruction Statistics (Epoch {epoch}):")
    print(f"Std between originals: {orig_tensors.std(dim=0).mean():.6f}")
    print(f"Std between reconstructions: {recon_tensors.std(dim=0).mean():.6f}")
    print(f"Mean correlation between reconstructions: {torch.corrcoef(recon_tensors.view(n_samples, -1)).mean():.6f}")
    
    return recon_tensors.std(dim=0).mean().item()


def train_epoch(model, loader, optimizer, device, language_mask_ratio=0.9, scaler=None):
    model.train()
    total_loss = 0
    total_vision_loss = 0
    total_language_loss = 0
    
    progress_bar = tqdm(loader, desc="Training")
    for batch in progress_bar:
        vision = batch['vision_embedding'].to(device)
        language = batch['language_embedding'].to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(vision, language, 
                              vision_mask_ratio=0.0, 
                              language_mask_ratio=language_mask_ratio)
                loss = outputs['loss']
        else:
            outputs = model(vision, language,
                          vision_mask_ratio=0.0,
                          language_mask_ratio=language_mask_ratio)
            loss = outputs['loss']
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        total_loss += loss.item()
        total_vision_loss += outputs['vision_loss'].item()
        total_language_loss += outputs['language_loss'].item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'v_loss': f"{outputs['vision_loss'].item():.4f}",
            'l_loss': f"{outputs['language_loss'].item():.4f}",
            'mask': f"{language_mask_ratio:.1%}"
        })
    
    n_batches = len(loader)
    return {
        'loss': total_loss / n_batches,
        'vision_loss': total_vision_loss / n_batches,
        'language_loss': total_language_loss / n_batches
    }


@torch.no_grad()
def evaluate(model, loader, device, epoch=0, visualize=False, save_dir=None, num_vis_samples=5):
    model.eval()
    total_loss = 0
    total_vision_loss = 0
    total_language_loss = 0
    
    vis_samples = []
    
    for batch_idx, batch in enumerate(loader):
        vision = batch['vision_embedding'].to(device)
        language = batch['language_embedding'].to(device)
        
        # Evaluate without masking to see reconstruction quality
        outputs = model(vision, language, vision_mask_ratio=0.0, language_mask_ratio=0.0)
        
        total_loss += outputs['loss'].item()
        total_vision_loss += outputs['vision_loss'].item()
        total_language_loss += outputs['language_loss'].item()
        
        # Collect samples for visualization
        if visualize and len(vis_samples) < num_vis_samples:
            # Take different samples from the batch
            for i in range(min(len(vision), num_vis_samples - len(vis_samples))):
                vis_samples.append({
                    'vision_original': outputs['vision_original'][i],
                    'language_original': outputs['language_original'][i],
                    'vision_recon': outputs['vision_recon'][i],
                    'language_recon': outputs['language_recon'][i]
                })
    
    # Visualize multiple samples
    if visualize and vis_samples and save_dir is not None:
        for idx, sample in enumerate(vis_samples):
            visualize_embeddings(
                sample['vision_original'],
                sample['vision_recon'],
                f'Vision_Sample{idx+1}',
                epoch,
                save_dir
            )
            visualize_embeddings(
                sample['language_original'],
                sample['language_recon'],
                f'Language_Sample{idx+1}',
                epoch,
                save_dir
            )
        
        # Also create a comparison plot showing all language reconstructions
        if len(vis_samples) >= 3:
            compare_language_reconstructions(vis_samples, epoch, save_dir)
            compare_vision_reconstructions(vis_samples, epoch, save_dir)
    
    n_batches = len(loader)
    return {
        'loss': total_loss / n_batches,
        'vision_loss': total_vision_loss / n_batches,
        'language_loss': total_language_loss / n_batches
    }


def load_splits(config_path, min_observations_per_species=5):
    """Load train/test observation IDs from config file"""
    logger.info(f"Loading splits from {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    observation_mappings = config['observation_mappings']
    
    # Count observations per species
    species_counts = {}
    for obs_id, metadata in observation_mappings.items():
        species = metadata['taxon_name']
        if species not in species_counts:
            species_counts[species] = []
        species_counts[species].append((obs_id, metadata))
    
    # Filter species with enough observations
    train_obs_ids = []
    test_obs_ids = []
    
    for species, obs_list in species_counts.items():
        if len(obs_list) >= min_observations_per_species:
            for obs_id, metadata in obs_list:
                if metadata['split'] == 'train':
                    train_obs_ids.append(obs_id)
                elif metadata['split'] == 'test':
                    test_obs_ids.append(obs_id)
    
    logger.info(f"Loaded {len(train_obs_ids)} train and {len(test_obs_ids)} test observations")
    return train_obs_ids, test_obs_ids


class DeepEarthDataset(Dataset):
    """Dataset that loads mean-pooled embeddings"""
    def __init__(self, observation_ids, cache, device='cpu'):
        self.observation_ids = observation_ids
        self.device = device
        
        # Load all data into memory
        self._load_all_data(cache)
        
    def _load_all_data(self, cache, batch_size=64):
        """Load all data into memory at once"""
        logger.info(f"Loading {len(self.observation_ids)} observations...")
        
        all_vision_embs = []
        all_language_embs = []
        all_species = []
        
        # Load data in batches
        for i in tqdm(range(0, len(self.observation_ids), batch_size), desc="Loading"):
            batch_ids = self.observation_ids[i:i + batch_size]
            
            try:
                from services.training_data import get_training_batch
                batch_data = get_training_batch(
                    cache,
                    batch_ids,
                    include_vision=True,
                    include_language=True,
                    device='cpu'
                )
                
                # Mean pool vision embeddings: (B, 8, 24, 24, 1408) -> (B, 1408)
                vision_batch = batch_data['vision_embeddings']
                if vision_batch.dim() == 5:
                    vision_batch = vision_batch.mean(dim=(1, 2, 3))
                
                # Language embeddings should already be (B, 7168)
                language_batch = batch_data['language_embeddings']
                
                all_vision_embs.append(vision_batch)
                all_language_embs.append(language_batch)
                all_species.extend(batch_data['species'])
                
            except Exception as e:
                logger.warning(f"Error loading batch: {e}")
                # Skip failed batches
                continue
        
        # Concatenate all embeddings
        self.vision_embeddings = torch.cat(all_vision_embs, dim=0)
        self.language_embeddings = torch.cat(all_language_embs, dim=0)
        self.species = all_species
        
        logger.info(f"Loaded {len(self.vision_embeddings)} samples")
        logger.info(f"Vision shape: {self.vision_embeddings.shape}")
        logger.info(f"Language shape: {self.language_embeddings.shape}")
        
    def __len__(self):
        return len(self.vision_embeddings)
    
    def __getitem__(self, idx):
        return {
            'vision_embedding': self.vision_embeddings[idx],
            'language_embedding': self.language_embeddings[idx],
            'species': self.species[idx]
        }


def analyze_dataset_statistics(dataset):
    """Analyze the statistics of embeddings in the dataset"""
    print("\nDataset Statistics:")
    print("-" * 50)
    
    # Stack all embeddings
    all_vision = torch.stack([dataset[i]['vision_embedding'] for i in range(len(dataset))])
    all_language = torch.stack([dataset[i]['language_embedding'] for i in range(len(dataset))])
    
    # Vision statistics
    vision_mean = all_vision.mean(dim=0)
    vision_std = all_vision.std(dim=0)
    vision_var = all_vision.var(dim=0)
    
    print(f"Vision Embeddings (1408D):")
    print(f"  Mean of means: {vision_mean.mean():.6f}")
    print(f"  Mean of stds: {vision_std.mean():.6f}")
    print(f"  Mean of vars: {vision_var.mean():.6f}")
    print(f"  Min value: {all_vision.min():.6f}")
    print(f"  Max value: {all_vision.max():.6f}")
    print(f"  Std across samples: {all_vision.std(dim=0).mean():.6f}")
    
    # Language statistics
    language_mean = all_language.mean(dim=0)
    language_std = all_language.std(dim=0)
    language_var = all_language.var(dim=0)
    
    print(f"\nLanguage Embeddings (7168D):")
    print(f"  Mean of means: {language_mean.mean():.6f}")
    print(f"  Mean of stds: {language_std.mean():.6f}")
    print(f"  Mean of vars: {language_var.mean():.6f}")
    print(f"  Min value: {all_language.min():.6f}")
    print(f"  Max value: {all_language.max():.6f}")
    print(f"  Std across samples: {all_language.std(dim=0).mean():.6f}")
    
    # Compare variance between samples
    print(f"\nVariance Analysis:")
    print(f"  Language/Vision std ratio: {all_language.std(dim=0).mean() / all_vision.std(dim=0).mean():.3f}")
    
    # Check how similar language embeddings are to each other
    # Compute pairwise correlations for a few samples
    n_samples = min(10, len(dataset))
    lang_subset = all_language[:n_samples]
    corr_matrix = torch.corrcoef(lang_subset.view(n_samples, -1))
    
    # Exclude diagonal (self-correlation)
    mask = ~torch.eye(n_samples, dtype=bool)
    off_diagonal_corr = corr_matrix[mask].mean()
    
    print(f"  Mean pairwise correlation (first {n_samples} language samples): {off_diagonal_corr:.4f}")
    
    # Visualize distribution of a few dimensions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Vision distribution
    axes[0, 0].hist(all_vision[:, 0].numpy(), bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_title('Vision Embedding - Dimension 0 Distribution')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    
    axes[0, 1].hist(vision_std.numpy(), bins=50, alpha=0.7, color='green')
    axes[0, 1].set_title('Vision - Std Dev per Dimension')
    axes[0, 1].set_xlabel('Std Dev')
    axes[0, 1].set_ylabel('Frequency')
    
    # Language distribution
    axes[1, 0].hist(all_language[:, 0].numpy(), bins=50, alpha=0.7, color='red')
    axes[1, 0].set_title('Language Embedding - Dimension 0 Distribution')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Frequency')
    
    axes[1, 1].hist(language_std.numpy(), bins=50, alpha=0.7, color='orange')
    axes[1, 1].set_title('Language - Std Dev per Dimension')
    axes[1, 1].set_xlabel('Std Dev')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.suptitle('Embedding Statistics Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig('embedding_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'vision_mean': vision_mean,
        'vision_std': vision_std,
        'language_mean': language_mean,
        'language_std': language_std,
        'language_corr': off_diagonal_corr
    }


def create_training_gifs(viz_dir):
    """Create GIFs from saved visualization images"""
    logger.info("Creating training GIFs...")
    
    # Define the types of visualizations to create GIFs for
    viz_types = [
        ('vision_sample*', 'vision_training.gif'),
        ('language_sample*', 'language_training.gif'),
        ('vision_comparison*', 'vision_comparison_training.gif'),
        ('language_comparison*', 'language_comparison_training.gif')
    ]
    
    for pattern, output_name in viz_types:
        # Find all matching images
        image_files = sorted(glob.glob(str(viz_dir / f'{pattern}_epoch_*.png')))
        
        if not image_files:
            logger.warning(f"No images found for pattern {pattern}")
            continue
            
        # Load images
        images = []
        for img_path in image_files:
            img = Image.open(img_path)
            images.append(img)
        
        if images:
            # Save as GIF
            output_path = viz_dir / output_name
            images[0].save(
                output_path,
                save_all=True,
                append_images=images[1:],
                duration=500,  # 500ms per frame
                loop=0  # Infinite loop
            )
            logger.info(f"Created GIF: {output_path}")
    
    # Also create a combined visualization showing both modalities
    create_combined_gif(viz_dir)


def create_combined_gif(viz_dir):
    """Create a combined GIF showing vision and language side by side"""
    logger.info("Creating combined modality GIF...")
    
    # Find matching vision and language comparison images
    vision_files = sorted(glob.glob(str(viz_dir / 'vision_comparison_epoch_*.png')))
    language_files = sorted(glob.glob(str(viz_dir / 'language_comparison_epoch_*.png')))
    
    if not vision_files or not language_files:
        logger.warning("Not enough comparison images for combined GIF")
        return
    
    # Take the minimum number of frames
    n_frames = min(len(vision_files), len(language_files))
    combined_images = []
    
    for i in range(n_frames):
        # Load vision and language images
        vis_img = Image.open(vision_files[i])
        lang_img = Image.open(language_files[i])
        
        # Get dimensions
        vis_w, vis_h = vis_img.size
        lang_w, lang_h = lang_img.size
        
        # Create combined image (side by side)
        combined_width = vis_w + lang_w + 20  # 20 pixel gap
        combined_height = max(vis_h, lang_h)
        
        combined = Image.new('RGB', (combined_width, combined_height), 'white')
        combined.paste(vis_img, (0, 0))
        combined.paste(lang_img, (vis_w + 20, 0))
        
        combined_images.append(combined)
    
    if combined_images:
        output_path = viz_dir / 'combined_modalities_training.gif'
        combined_images[0].save(
            output_path,
            save_all=True,
            append_images=combined_images[1:],
            duration=500,
            loop=0
        )
        logger.info(f"Created combined GIF: {output_path}")


def main():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    epochs = 50
    lr = 1e-3
    
    # Model dimensions
    vision_dim = 1408
    language_dim = 7168
    universal_dim = 2048
    
    # Masking schedule
    language_mask_ratio = 1.0  # 100% masking throughout
    
    logger.info(f"Using device: {device}")
    logger.info(f"Language masking: {language_mask_ratio:.0%} throughout training")
    
    # Setup paths
    dashboard_path = Path(__file__).parent.parent / "dashboard"
    sys.path.insert(0, str(dashboard_path))
    
    from data_cache import UnifiedDataCache
    
    # Change to dashboard directory for cache
    original_dir = os.getcwd()
    os.chdir(dashboard_path)
    
    try:
        # Load data
        config_path = Path(__file__).parent / "config" / "central_florida_split.json"
        train_obs_ids, test_obs_ids = load_splits(config_path)
        
        # Create cache and datasets
        cache = UnifiedDataCache("dataset_config.json")
        train_dataset = DeepEarthDataset(train_obs_ids, cache, device='cpu')
        test_dataset = DeepEarthDataset(test_obs_ids, cache, device='cpu')
        
    finally:
        os.chdir(original_dir)
    
    # Analyze dataset statistics before training
    stats = analyze_dataset_statistics(train_dataset)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == 'cuda')
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == 'cuda')
    )
    
    # Create model
    model = MultimodalAutoencoder(
        vision_dim=vision_dim,
        language_dim=language_dim,
        universal_dim=universal_dim
    ).to(device)
    
    # Freeze language encoder/decoder since we're masking 100% of language
    for p in model.language_encoder.parameters():
        p.requires_grad = False
    for p in model.language_decoder.parameters():
        p.requires_grad = False
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Optimizer and scheduler - only optimize parameters that require gradients
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr, 
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    
    # Visualization directory
    viz_dir = Path('visualizations')
    viz_dir.mkdir(exist_ok=True)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs} - Language masking: {language_mask_ratio:.1%}")
        print("-" * 50)
        
        # Train with constant 100% mask ratio
        train_metrics = train_epoch(model, train_loader, optimizer, device, 
                                  language_mask_ratio=language_mask_ratio, scaler=scaler)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Vision: {train_metrics['vision_loss']:.4f}, "
              f"Language: {train_metrics['language_loss']:.4f}")
        
        # Evaluate
        visualize = (epoch % 5 == 0) or (epoch == epochs - 1)
        test_metrics = evaluate(model, test_loader, device, epoch, visualize, viz_dir)
        print(f"Test  - Loss: {test_metrics['loss']:.4f}, "
              f"Vision: {test_metrics['vision_loss']:.4f}, "
              f"Language: {test_metrics['language_loss']:.4f}")
        
        # Save best model
        if test_metrics['loss'] < best_loss:
            best_loss = test_metrics['loss']
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'test_loss': test_metrics['loss'],
                'vision_dim': vision_dim,
                'language_dim': language_dim,
                'universal_dim': universal_dim
            }, 'best_model.pth')
            print(f"✓ Saved best model!")
        
        scheduler.step()
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
    
    print(f"\nTraining complete! Best test loss: {best_loss:.4f}")
    
    # Create training GIFs
    create_training_gifs(viz_dir)
    
    # Save final universal representations for downstream tasks
    logger.info("Extracting universal representations for species classification...")
    model.eval()
    
    with torch.no_grad():
        all_vision_universal = []
        all_language_universal = []
        all_species = []
        
        for batch in test_loader:
            vision = batch['vision_embedding'].to(device)
            language = batch['language_embedding'].to(device)
            
            outputs = model(vision, language, vision_mask_ratio=0.0, language_mask_ratio=0.0)
            
            all_vision_universal.append(outputs['vision_universal'].cpu())
            all_language_universal.append(outputs['language_universal'].cpu())
            all_species.extend(batch['species'])
        
        universal_representations = {
            'vision_universal': torch.cat(all_vision_universal, dim=0),
            'language_universal': torch.cat(all_language_universal, dim=0),
            'species': all_species
        }
        
        torch.save(universal_representations, 'universal_representations.pt')
        logger.info(f"Saved universal representations for {len(all_species)} samples")
        logger.info("These 2048D vectors can be used for species classification downstream")


if __name__ == "__main__":
    main()
