#!/usr/bin/env python3
"""
Multimodal Token-Based Autoencoder with Masked Language Modeling
- Vision patches and language tokens in unified 512-D space
- True masked language modeling (MLM) with vocabulary prediction
- 1D U-Net processes concatenated token sequences
- Proper token-level masking instead of embedding-level
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import argparse
import logging
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from einops import rearrange
from einops.layers.torch import Rearrange
from transformers import AutoTokenizer, AutoModel
from collections import Counter

# Add dashboard to path
dashboard_path = Path(__file__).parent.parent / "dashboard"
sys.path.insert(0, str(dashboard_path))

# Suppress verbose logging
logging.getLogger('services.training_data').setLevel(logging.WARNING)
logging.getLogger('mmap_embedding_loader').setLevel(logging.WARNING)
logging.getLogger('huggingface_data_loader').setLevel(logging.WARNING)
logging.getLogger('data_cache').setLevel(logging.WARNING)

from services.training_data import get_training_batch
from data_cache import UnifiedDataCache

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ============== TOKENIZER UTILITIES ==============

class DeepSeekTokenizerWrapper:
    """Wrapper for DeepSeek tokenizer with consistent interface"""
    
    def __init__(self, model_name='deepseek-ai/deepseek-moe-16b-base', max_length=64):
        from transformers import AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.vocab_size = len(self.tokenizer)
        
    def text_to_tensor(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Convert texts to padded token tensors"""
        # Tokenize with padding
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'],  # (batch, seq_len)
            'attention_mask': encoding['attention_mask']  # (batch, seq_len)
        }


# ============== VISION PATCH PROJECTOR ==============

class VisionPatchProjector(nn.Module):
    """Project VJEPA2 vision patches to token dimension"""
    
    def __init__(self, in_dim=1408, out_dim=512, num_patches=4608):
        super().__init__()
        # VJEPA2 outputs are already high-quality frozen features
        self.proj = nn.Linear(in_dim, out_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches, out_dim) * 0.02)
        
    def forward(self, x):
        #once we have good test step for this thing 
        # want to perserve spatial features 
        # x: (B, 8, 24, 24, 1408) - VJEPA2 patch embeddings
        B, T, H, W, C = x.shape
        x = x.view(B, T*H*W, C)  # (B, 4608, 1408)
        x = self.proj(x)  # (B, 4608, 512)
        x = x + self.pos_emb[:, :T*H*W]  # Add position embeddings
        return x  # (B, 4608, 512)


# ============== LANGUAGE TOKEN EMBEDDER ==============

class LanguageTokenEmbedder(nn.Module):
    """Embed language tokens using frozen DeepSeek + learned projection"""
    
    def __init__(self, model_name='deepseek-ai/deepseek-moe-16b-base', out_dim=512, max_length=64):
        super().__init__()
        # Note: In practice, you'd load the actual DeepSeek model
        # For now, using BERT as placeholder - replace with DeepSeek
        from transformers import AutoModel, AutoTokenizer
        
        # Load DeepSeek (or use BERT for testing)
        try:
            self.model = AutoModel.from_pretrained(model_name)
            self.hidden_size = self.model.config.hidden_size
        except:
            # Fallback to BERT if DeepSeek not available
            logger.warning(f"Could not load {model_name}, falling back to BERT")
            self.model = AutoModel.from_pretrained('bert-base-uncased')
            self.hidden_size = self.model.config.hidden_size
        
        self.proj = nn.Linear(self.hidden_size, out_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_length, out_dim) * 0.02)
        
        # Freeze DeepSeek/model parameters
        for p in self.model.parameters():
            p.requires_grad = False
            
    def forward(self, input_ids, attention_mask):
        # Get frozen embeddings from DeepSeek
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            h = outputs.last_hidden_state  # (B, L, hidden_size)
        
        # Project and add position embeddings
        h = self.proj(h)  # (B, L, 512)
        seq_len = h.size(1)
        h = h + self.pos_emb[:, :seq_len]
        
        return h  # (B, L, 512)


# ============== 1D U-NET FOR TOKEN SEQUENCES ==============

class UNet1D(nn.Module):
    """1D U-Net for processing token sequences"""
    
    def __init__(self, in_channels=512, base_channels=128, depth=4, channel_cap=512):
        super().__init__()
        
        # Calculate channels at each level
        channels = []
        for i in range(depth):
            ch = min(base_channels * (2**i), channel_cap)
            channels.append(ch)
        
        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        in_ch = in_channels
        for ch in channels:
            self.encoders.append(self._make_encoder(in_ch, ch))
            self.pools.append(nn.MaxPool1d(2))
            in_ch = ch
        
        # Bottleneck
        self.bottleneck = self._make_encoder(channels[-1], channels[-1])
        
        # Decoder
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        for i in range(depth-1, -1, -1):
            self.upsamples.append(nn.ConvTranspose1d(
                channels[i] if i == depth-1 else channels[i+1],
                channels[i],
                kernel_size=2,
                stride=2
            ))
            self.decoders.append(self._make_encoder(channels[i] * 2, channels[i]))
        
        # Final projection back to input channels
        self.final_conv = nn.Conv1d(channels[0], in_channels, kernel_size=1)
        
    def _make_encoder(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # x: (B, C, N) where C=512, N=num_tokens
        
        # Encoder path
        encoder_features = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            encoder_features.append(x)
            x = pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        for i, (upsample, decoder) in enumerate(zip(self.upsamples, self.decoders)):
            x = upsample(x)
            
            # Handle size mismatch
            enc_feat = encoder_features[-(i+1)]
            if x.size(2) != enc_feat.size(2):
                x = F.interpolate(x, size=enc_feat.size(2), mode='linear', align_corners=False)
            
            x = torch.cat([x, enc_feat], dim=1)
            x = decoder(x)
        
        # Final projection
        x = self.final_conv(x)
        
        return x  # (B, 512, N)


# ============== MASKING UTILITIES ==============

def apply_language_mask(tokens, attention_mask, mask_token, mask_ratio=0.5):
    """
    Apply token-level masking to language tokens
    
    Args:
        tokens: (B, L, D) language token features
        attention_mask: (B, L) binary mask (1 for real tokens, 0 for padding)
        mask_token: (1, 1, D) learnable mask token
        mask_ratio: probability of masking each token
    
    Returns:
        masked_tokens: (B, L, D) tokens with some replaced by mask_token
        mask: (B, L) binary mask indicating which tokens were masked
    """
    B, L, D = tokens.shape
    device = tokens.device
    
    # Create random mask
    rand_mask = torch.rand(B, L, device=device) < mask_ratio
    
    # Don't mask padding tokens
    mask = rand_mask & attention_mask.bool()
    
    # Apply mask
    mask_expanded = mask.unsqueeze(-1).expand(-1, -1, D)
    masked_tokens = torch.where(mask_expanded, mask_token.expand(B, L, D), tokens)
    
    return masked_tokens, mask


# ============== MODIFIED DATASET FOR TOKEN-BASED APPROACH ==============

class TokenBasedMultimodalDataset(Dataset):
    """Dataset that returns raw text and vision patches"""
    
    def __init__(self, observation_ids: List[str], cache, config, 
                 species_mapping=None, tokenizer=None):
        self.observation_ids = observation_ids
        self.cache = cache
        self.config = config
        self.tokenizer = tokenizer or BERTTokenizerWrapper()
        
        # Build species mapping
        if species_mapping is not None:
            self.species_to_idx = species_mapping
            self.num_classes = len(species_mapping)
        else:
            logger.info("Building species mapping...")
            all_species = set()
            
            for obs_id in observation_ids:
                if obs_id in config['observation_mappings']:
                    species = config['observation_mappings'][obs_id]['taxon_name']
                    all_species.add(species)
            
            self.species_to_idx = {s: i for i, s in enumerate(sorted(all_species))}
            self.num_classes = len(self.species_to_idx)
            
        logger.info(f"Dataset initialized with {self.num_classes} species")
        
        # Pre-compute valid indices and labels
        self.valid_data = []
        for obs_id in observation_ids:
            if obs_id in config['observation_mappings']:
                meta = config['observation_mappings'][obs_id]
                species = meta['taxon_name']
                if species in self.species_to_idx:
                    self.valid_data.append({
                        'obs_id': obs_id,
                        'species': species,
                        'species_idx': self.species_to_idx[species],
                        'caption': meta.get('caption', f"An observation of {species}")
                    })
        
        logger.info(f"Found {len(self.valid_data)} valid observations")
        
        # Compute species frequencies for weighted sampling
        species_counts = Counter([item['species_idx'] for item in self.valid_data])
        total_count = sum(species_counts.values())
        self.species_weights = {
            idx: total_count / (count * self.num_classes)  # Inverse frequency
            for idx, count in species_counts.items()
        }
    
    def __len__(self):
        return len(self.valid_data)
    
    def get_species_label(self, idx):
        return self.valid_data[idx]['species_idx']
    
    def get_sample_weight(self, idx):
        species_idx = self.get_species_label(idx)
        return self.species_weights[species_idx]
    
    def __getitem__(self, idx):
        data = self.valid_data[idx]
        obs_id = data['obs_id']
        
        try:
            # Get vision embeddings
            batch_data = get_training_batch(
                self.cache,
                [obs_id],
                include_vision=True,
                include_language=False,  # We'll use raw text instead
                device='cpu'
            )
            
            vision_emb = batch_data['vision_embeddings'][0]  # (8, 24, 24, 1408)
            
            return {
                'vision_embedding': vision_emb,
                'caption': data['caption'],
                'species_label': torch.tensor(data['species_idx'], dtype=torch.long)
            }
            
        except Exception as e:
            logger.error(f"Error loading observation {obs_id}: {e}")
            # Return a different sample on error
            return self.__getitem__(np.random.randint(0, len(self)))


def collate_fn_token_based(batch):
    """Collate function that handles tokenization"""
    vision = torch.stack([item['vision_embedding'] for item in batch])
    captions = [item['caption'] for item in batch]
    labels = torch.stack([item['species_label'] for item in batch])
    
    return {
        'vision_embedding': vision,
        'captions': captions,  # Raw text, will be tokenized in model
        'species_label': labels
    }


# ============== MAIN MODEL WITH TOKEN-BASED ARCHITECTURE ==============

class TokenBasedMultimodalModel(nn.Module):
    """
    Token-based multimodal model with MLM
    - Vision patches → tokens
    - Language text → tokens  
    - Unified 512-D token space
    - 1D U-Net processes concatenated sequences
    - MLM head for language reconstruction
    """
    
    def __init__(self, num_classes, tokenizer, vision_dim=1408, token_dim=512,
                 unet_depth=4, mlm_weight=1.0, contrast_weight=0.0):
        super().__init__()
        
        self.num_classes = num_classes
        self.tokenizer = tokenizer
        self.token_dim = token_dim
        self.mlm_weight = mlm_weight
        self.contrast_weight = contrast_weight
        
        # Vision patch projector (for VJEPA2 embeddings)
        self.vision_projector = VisionPatchProjector(
            in_dim=vision_dim,  # 1408 from VJEPA2
            out_dim=token_dim,
            num_patches=4608  # 8*24*24
        )
        
        # Language token embedder (for DeepSeek embeddings)
        self.language_embedder = LanguageTokenEmbedder(
            model_name='deepseek-ai/deepseek-moe-16b-base',  # Or your DeepSeek variant
            out_dim=token_dim,
            max_length=64
        )
        
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)
        
        # 1D U-Net backbone
        self.unet = UNet1D(
            in_channels=token_dim,
            base_channels=128,
            depth=unet_depth,
            channel_cap=512
        )
        
        # MLM head
        self.mlm_head = nn.Linear(token_dim, tokenizer.vocab_size)
        
        # Universal projection: 512 -> 2048
        self.universal_proj = nn.Sequential(
            nn.Linear(token_dim, 2048),
            nn.LayerNorm(2048),
            nn.Dropout(0.2)
        )
        
        # Global classification head (from universal embedding)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )
        
        # Contrastive projection heads (from universal embedding)
        self.vision_contrast_proj = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        self.language_contrast_proj = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
    def forward(self, vision, captions, labels=None, mask_ratio=0.5):
        """
        Forward pass
        
        Args:
            vision: (B, 8, 24, 24, 1408) vision patches
            captions: List[str] of length B
            labels: (B,) species labels
            mask_ratio: probability of masking language tokens
        """
        B = vision.shape[0]
        device = vision.device
        
        # Tokenize captions
        tokenized = self.tokenizer.text_to_tensor(captions)
        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)
        
        # 1. Produce token sequences
        v_tokens = self.vision_projector(vision)  # (B, 4608, 512)
        l_tokens = self.language_embedder(input_ids, attention_mask)  # (B, L, 512)
        
        # Store original language tokens for MLM loss
        l_tokens_original = l_tokens.clone()
        
        # 2. Mask language tokens (only during training)
        if self.training:
            l_tokens, mask = apply_language_mask(
                l_tokens, attention_mask, self.mask_token, mask_ratio
            )
        else:
            mask = torch.zeros_like(attention_mask).bool()
        
        # 3. Concatenate and run U-Net
        tokens = torch.cat([v_tokens, l_tokens], dim=1)  # (B, N, 512)
        x = tokens.transpose(1, 2)  # (B, 512, N)
        x = self.unet(x)
        x = x.transpose(1, 2)  # (B, N, 512)
        
        # 4. Split back
        v_out = x[:, :v_tokens.size(1)]  # (B, 4608, 512)
        l_out = x[:, v_tokens.size(1):]  # (B, L, 512)
        
        # 5. MLM predictions
        mlm_logits = self.mlm_head(l_out)  # (B, L, vocab_size)
        
        # 6. Global representations (mean pooling)
        # Only pool over valid tokens for language
        v_global = v_out.mean(dim=1)  # (B, 512)
        
        # Masked mean for language (only real tokens)
        l_valid_sum = (l_out * attention_mask.unsqueeze(-1)).sum(dim=1)
        l_valid_count = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        l_global = l_valid_sum / l_valid_count  # (B, 512)
        
        # Project to universal embedding space (2048D)
        v_universal = self.universal_proj(v_global)  # (B, 2048)
        l_universal = self.universal_proj(l_global)  # (B, 2048)
        
        # Average vision and language universal embeddings
        universal_embedding = (v_universal + l_universal) / 2  # (B, 2048)
        
        # 7. Classification from universal embedding
        logits = self.classifier(universal_embedding)
        
        # 8. Contrastive projections from universal embeddings
        v_contrast = self.vision_contrast_proj(v_universal)
        l_contrast = self.language_contrast_proj(l_universal)
        
        # Compute losses
        losses = {}
        
        # MLM loss (only on masked positions)
        if self.training and mask.any():
            mlm_loss = F.cross_entropy(
                mlm_logits[mask],
                input_ids[mask],
                reduction='mean'
            )
            losses['mlm'] = mlm_loss
        else:
            losses['mlm'] = torch.tensor(0.0, device=device)
        
        # Classification loss
        if labels is not None:
            cls_loss = F.cross_entropy(logits, labels)
            losses['cls'] = cls_loss
        
        # Contrastive loss (if weight > 0)
        if self.contrast_weight > 0 and labels is not None:
            contrast_loss = self.compute_contrastive_loss(
                v_contrast, l_contrast, labels
            )
            losses['contrast'] = contrast_loss
        else:
            losses['contrast'] = torch.tensor(0.0, device=device)
        
        # Total loss
        total_loss = (
            losses.get('cls', 0) + 
            self.mlm_weight * losses['mlm'] + 
            self.contrast_weight * losses['contrast']
        )
        
        return {
            'loss': total_loss,
            'losses': losses,
            'logits': logits,
            'mlm_logits': mlm_logits,
            'mask': mask,
            'v_global': v_global,
            'l_global': l_global,
            'v_universal': v_universal,
            'l_universal': l_universal,
            'universal_embedding': universal_embedding,
            'v_contrast': v_contrast,
            'l_contrast': l_contrast
        }
    
    def compute_contrastive_loss(self, v_features, l_features, labels, temperature=0.07):
        """Compute species-aware contrastive loss"""
        # Normalize features
        v_norm = F.normalize(v_features, p=2, dim=1)
        l_norm = F.normalize(l_features, p=2, dim=1)
        
        # Compute similarities
        similarities = torch.matmul(v_norm, l_norm.T) / temperature
        
        # Create label matrix
        labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        # Compute loss both directions
        exp_sim = torch.exp(similarities)
        
        # V->L
        pos_sum_v2l = (exp_sim * labels_matrix.float()).sum(dim=1)
        total_sum_v2l = exp_sim.sum(dim=1)
        loss_v2l = -torch.log(pos_sum_v2l / total_sum_v2l).mean()
        
        # L->V
        exp_sim_t = torch.exp(similarities.T)
        pos_sum_l2v = (exp_sim_t * labels_matrix.float()).sum(dim=1)
        total_sum_l2v = exp_sim_t.sum(dim=1)
        loss_l2v = -torch.log(pos_sum_l2v / total_sum_l2v).mean()
        
        return (loss_v2l + loss_l2v) / 2


# ============== TRAINING FUNCTIONS ==============

def train_epoch_token_based(model, loader, optimizer, device, epoch, 
                          total_epochs, mask_ratio=0.5):
    """Training epoch for token-based model"""
    model.train()
    
    total_loss = 0
    total_cls_loss = 0
    total_mlm_loss = 0
    total_contrast_loss = 0
    correct = 0
    total = 0
    
    # Dynamic weight scheduling
    if epoch < 3:
        # First 3 epochs: focus on MLM
        model.mlm_weight = 1.0
        model.contrast_weight = 0.0
    else:
        # Ramp up contrastive loss
        progress = (epoch - 3) / (total_epochs - 3)
        model.mlm_weight = 1.0
        model.contrast_weight = min(0.5, progress * 0.5)
    
    logger.info(f"Epoch {epoch}: MLM weight={model.mlm_weight:.2f}, "
                f"Contrast weight={model.contrast_weight:.2f}")
    
    for i, batch in enumerate(loader):
        vision = batch['vision_embedding'].to(device)
        captions = batch['captions']
        labels = batch['species_label'].to(device)
        
        # Forward pass
        outputs = model(vision, captions, labels, mask_ratio=mask_ratio)
        
        loss = outputs['loss']
        losses = outputs['losses']
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item() * len(labels)
        total_cls_loss += losses.get('cls', 0).item() * len(labels)
        total_mlm_loss += losses.get('mlm', 0).item() * len(labels)
        total_contrast_loss += losses.get('contrast', 0).item() * len(labels)
        
        pred = outputs['logits'].argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += len(labels)
        
        if i % 10 == 0:
            print(f"\r  Batch {i}/{len(loader)}: Loss={loss.item():.4f}, "
                  f"MLM={losses['mlm'].item():.4f}, "
                  f"Cls={losses['cls'].item():.4f}, "
                  f"Acc={correct/total:.2%}", end='', flush=True)
    
    print()
    
    return {
        'loss': total_loss / total,
        'cls_loss': total_cls_loss / total,
        'mlm_loss': total_mlm_loss / total,
        'contrast_loss': total_contrast_loss / total,
        'accuracy': correct / total
    }


@torch.no_grad()
def evaluate_token_based(model, loader, device):
    """Evaluation for token-based model"""
    model.eval()
    
    total_loss = 0
    total_cls_loss = 0
    total_mlm_loss = 0
    correct = 0
    total = 0
    
    all_v_universal = []
    all_l_universal = []
    all_universal_embeddings = []
    all_labels = []
    
    for batch in loader:
        vision = batch['vision_embedding'].to(device)
        captions = batch['captions']
        labels = batch['species_label'].to(device)
        
        # Forward pass (no masking during eval)
        outputs = model(vision, captions, labels, mask_ratio=0.0)
        
        loss = outputs['loss']
        losses = outputs['losses']
        
        # Metrics
        total_loss += loss.item() * len(labels)
        total_cls_loss += losses.get('cls', 0).item() * len(labels)
        total_mlm_loss += losses.get('mlm', 0).item() * len(labels)
        
        pred = outputs['logits'].argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += len(labels)
        
        # Store universal embeddings for retrieval metrics
        all_v_universal.append(outputs['v_universal'])
        all_l_universal.append(outputs['l_universal'])
        all_universal_embeddings.append(outputs['universal_embedding'])
        all_labels.append(labels)
    
    # Compute retrieval metrics on universal embeddings
    all_v_universal = torch.cat(all_v_universal, dim=0)
    all_l_universal = torch.cat(all_l_universal, dim=0)
    all_universal_embeddings = torch.cat(all_universal_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Normalize universal embeddings for similarity
    v_norm = F.normalize(all_v_universal, p=2, dim=1)
    l_norm = F.normalize(all_l_universal, p=2, dim=1)
    
    # V->L retrieval
    similarities = torch.matmul(v_norm, l_norm.T)
    v2l_correct = 0
    
    for i in range(len(all_labels)):
        retrieved_idx = similarities[i].argmax()
        if all_labels[retrieved_idx] == all_labels[i]:
            v2l_correct += 1
    
    v2l_acc = v2l_correct / len(all_labels)
    
    # k-NN accuracy on universal embeddings
    knn = NearestNeighbors(n_neighbors=6, metric='cosine')
    knn.fit(all_universal_embeddings.cpu().numpy())
    
    _, indices = knn.kneighbors(all_universal_embeddings.cpu().numpy())
    knn_correct = 0
    for i, neighbors in enumerate(indices):
        neighbor_labels = all_labels[neighbors[1:]].cpu().numpy()  # Skip self
        if np.any(neighbor_labels == all_labels[i].cpu().numpy()):
            knn_correct += 1
    knn_acc = knn_correct / len(all_labels)
    
    return {
        'loss': total_loss / total,
        'cls_loss': total_cls_loss / total,
        'mlm_loss': total_mlm_loss / total,
        'accuracy': correct / total,
        'v2l_retrieval': v2l_acc,
        'knn_accuracy': knn_acc
    }


# ============== WEIGHTED SAMPLER ==============

class InverseFrequencySampler(torch.utils.data.Sampler):
    """Sample with probability inversely proportional to species frequency"""
    
    def __init__(self, dataset, num_samples=None):
        self.dataset = dataset
        self.num_samples = num_samples or len(dataset)
        
        # Get weights for each sample
        self.weights = torch.tensor([
            dataset.get_sample_weight(i) for i in range(len(dataset))
        ])
        
    def __iter__(self):
        # Sample with replacement according to weights
        indices = torch.multinomial(
            self.weights, 
            self.num_samples, 
            replacement=True
        )
        return iter(indices.tolist())
    
    def __len__(self):
        return self.num_samples


# ============== MAIN TRAINING SCRIPT ==============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--effective-batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--config', type=str, default='config/central_florida_split.json')
    parser.add_argument('--mask-ratio', type=float, default=0.5)
    parser.add_argument('--max-species', type=int, default=None)
    parser.add_argument('--subset', type=int, default=None)
    parser.add_argument('--save-dir', type=str, default='checkpoints_token_mlm')
    parser.add_argument('--unet-depth', type=int, default=4)
    parser.add_argument('--token-dim', type=int, default=512)
    parser.add_argument('--use-weighted-sampling', action='store_true',
                       help='Use inverse frequency weighted sampling')
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Calculate gradient accumulation steps
    grad_accum_steps = args.effective_batch_size // args.batch_size
    logger.info(f"Gradient accumulation steps: {grad_accum_steps}")
    
    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get observation IDs
    train_ids = []
    test_ids = []
    
    for obs_id, meta in config['observation_mappings'].items():
        if meta['split'] == 'train':
            train_ids.append(obs_id)
        else:
            test_ids.append(obs_id)
    
    if args.subset:
        train_ids = train_ids[:args.subset]
        test_ids = test_ids[:min(args.subset//5, len(test_ids))]
    
    logger.info(f"Data split: {len(train_ids)} train, {len(test_ids)} test")
    
    # Change to dashboard directory
    original_dir = os.getcwd()
    os.chdir(dashboard_path)
    
    try:
        # Initialize cache and tokenizer
        cache = UnifiedDataCache("dataset_config.json")
        tokenizer = DeepSeekTokenizerWrapper()  # Use DeepSeek tokenizer
        
        # Create datasets
        logger.info("Creating datasets...")
        train_dataset = TokenBasedMultimodalDataset(
            train_ids, cache, config, tokenizer=tokenizer
        )
        
        test_dataset = TokenBasedMultimodalDataset(
            test_ids, cache, config, 
            species_mapping=train_dataset.species_to_idx,
            tokenizer=tokenizer
        )
        
        # Create data loaders
        if args.use_weighted_sampling:
            logger.info("Using inverse frequency weighted sampling")
            train_sampler = InverseFrequencySampler(
                train_dataset,
                num_samples=len(train_dataset)
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                sampler=train_sampler,
                num_workers=args.num_workers,
                collate_fn=collate_fn_token_based,
                pin_memory=(device == 'cuda')
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                collate_fn=collate_fn_token_based,
                pin_memory=(device == 'cuda')
            )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn_token_based,
            pin_memory=(device == 'cuda')
        )
        
        # Create model
        logger.info("Initializing token-based model...")
        model = TokenBasedMultimodalModel(
            num_classes=train_dataset.num_classes,
            tokenizer=tokenizer,
            token_dim=args.token_dim,
            unet_depth=args.unet_depth
        )
        model = model.to(device)
        
        logger.info(f"Model initialized:")
        logger.info(f"  Vision patches: 4608 × {args.token_dim}")
        logger.info(f"  Language tokens: ≤64 × {args.token_dim}")
        logger.info(f"  Universal embedding: 1 × 2048")
        logger.info(f"  U-Net depth: {args.unet_depth}")
        logger.info(f"  Vocabulary size: {tokenizer.vocab_size}")
        logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Training setup
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        print("\n" + "="*80)
        print("🚀 STARTING TOKEN-BASED MULTIMODAL TRAINING WITH MLM")
        print("="*80)
        print(f"Vision: VJEPA2 (frozen) → patches → tokens")
        print(f"Language: DeepSeek (frozen) → tokens")
        print(f"Architecture: Vision Patches + Language Tokens → 1D U-Net → Universal 2048D")
        print(f"Token dimension: {args.token_dim}")
        print(f"Mask ratio: {args.mask_ratio}")
        print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
        print("="*80 + "\n")
        
        best_metrics = {
            'epoch': 0,
            'test_acc': 0,
            'v2l_retrieval': 0,
            'combined_score': 0
        }
        
        for epoch in range(args.epochs):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1}/{args.epochs}")
            print(f"{'='*60}")
            
            # Train
            print("Training...")
            train_metrics = train_epoch_token_based(
                model, train_loader, optimizer, device, 
                epoch, args.epochs, mask_ratio=args.mask_ratio
            )
            
            print(f"\n📊 Train Results:")
            print(f"   Total Loss: {train_metrics['loss']:.4f}")
            print(f"   MLM Loss: {train_metrics['mlm_loss']:.4f}")
            print(f"   Cls Loss: {train_metrics['cls_loss']:.4f}")
            print(f"   Contrast Loss: {train_metrics['contrast_loss']:.4f}")
            print(f"   Accuracy: {train_metrics['accuracy']:.2%}")
            
            # Evaluate
            print("\nEvaluating...")
            test_metrics = evaluate_token_based(model, test_loader, device)
            
            print(f"\n📊 Test Results:")
            print(f"   Total Loss: {test_metrics['loss']:.4f}")
            print(f"   MLM Loss: {test_metrics['mlm_loss']:.4f}")
            print(f"   Accuracy: {test_metrics['accuracy']:.2%}")
            print(f"   V→L Retrieval: {test_metrics['v2l_retrieval']:.2%}")
            print(f"   k-NN Accuracy: {test_metrics['knn_accuracy']:.2%}")
            
            # Update learning rate
            scheduler.step()
            
            # Save best model (now includes k-NN in score)
            combined_score = (test_metrics['accuracy'] + test_metrics['v2l_retrieval'] + test_metrics['knn_accuracy']) / 3
            
            if combined_score > best_metrics['combined_score']:
                best_metrics.update({
                    'epoch': epoch + 1,
                    'test_acc': test_metrics['accuracy'],
                    'v2l_retrieval': test_metrics['v2l_retrieval'],
                    'combined_score': combined_score
                })
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'test_metrics': test_metrics,
                    'args': vars(args),
                    'species_mapping': train_dataset.species_to_idx,
                    'tokenizer_config': {
                        'model_name': 'bert-base-uncased',
                        'max_length': 64
                    }
                }, Path(args.save_dir) / 'token_mlm_best.pth')
                
                print(f"\n🎯 NEW BEST MODEL! Score: {combined_score:.3f}")
            
            print(f"\n   Best so far: Acc={best_metrics['test_acc']:.2%}, "
                  f"V→L={best_metrics['v2l_retrieval']:.2%} (Epoch {best_metrics['epoch']})")
        
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE!")
        print(f"🏆 Best Model (Epoch {best_metrics['epoch']}):")
        print(f"   Test Accuracy: {best_metrics['test_acc']:.2%}")
        print(f"   V→L Retrieval: {best_metrics['v2l_retrieval']:.2%}")
        print(f"   Combined Score: {best_metrics['combined_score']:.3f}")
        print("="*80)
        
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    main()
