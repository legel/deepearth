#!/usr/bin/env python3
"""
Multimodal Autoencoder for DeepEarth
Following meeting notes specifications:
- Separate vision and language decoders (2-layer MLPs)
- Variable masking strategies
- Universal 1x2048 embedding space
- Foundation for Grid4D integration
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
from sklearn.manifold import TSNE
import umap

def create_alignment_visualization(model, loader, device, epoch, save_dir):
    """Create UMAP visualization of vision-language alignment"""
    model.eval()
    
    all_vision_universal = []
    all_language_universal = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            vision = batch['vision_embedding'].to(device)
            language = batch['language_embedding'].to(device)
            labels = batch['species_label'].to(device)
            
            outputs = model(vision, language)
            all_vision_universal.append(outputs['vision_universal'].cpu())
            all_language_universal.append(outputs['language_universal'].cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate
    vision_universal = torch.cat(all_vision_universal, dim=0).numpy()
    language_universal = torch.cat(all_language_universal, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    
    # Normalize for cosine similarity
    vision_norm = F.normalize(torch.from_numpy(vision_universal), p=2, dim=1).numpy()
    language_norm = F.normalize(torch.from_numpy(language_universal), p=2, dim=1).numpy()
    
    # Compute alignment metrics
    n_samples = len(labels)
    diagonal_sim = np.sum(vision_norm * language_norm, axis=1)  # Paired similarity
    avg_alignment = diagonal_sim.mean()
    
    # Cross-modal retrieval
    v2l_correct = 0
    for i in range(n_samples):
        sims = np.dot(vision_norm[i], language_norm.T)
        if labels[sims.argmax()] == labels[i]:
            v2l_correct += 1
    v2l_acc = v2l_correct / n_samples
    
    # UMAP visualization
    combined = np.vstack([vision_universal, language_universal])
    combined_labels = np.hstack([labels, labels])
    modality = ['Vision'] * n_samples + ['Language'] * n_samples
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    embeddings_2d = reducer.fit_transform(combined)
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    # By modality
    for mod in ['Vision', 'Language']:
        mask = np.array(modality) == mod
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   label=mod, alpha=0.6, s=30)
    plt.title(f'Universal Space by Modality - Epoch {epoch}')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    # Connected pairs
    plt.scatter(embeddings_2d[:n_samples, 0], embeddings_2d[:n_samples, 1], 
               c=labels, cmap='tab20', alpha=0.8, s=50, marker='o')
    plt.scatter(embeddings_2d[n_samples:, 0], embeddings_2d[n_samples:, 1], 
               c=labels, cmap='tab20', alpha=0.8, s=50, marker='^')
    
    # Draw connections
    for i in range(min(n_samples, 50)):  # Limit lines for clarity
        plt.plot([embeddings_2d[i, 0], embeddings_2d[n_samples + i, 0]], 
                [embeddings_2d[i, 1], embeddings_2d[n_samples + i, 1]], 
                'gray', alpha=0.3, linewidth=0.5)
    
    plt.title(f'V-L Alignment - Avg Sim: {avg_alignment:.3f}, V→L R@1: {v2l_acc:.2%}')
    
    plt.tight_layout()
    plt.savefig(save_dir / f'alignment_epoch_{epoch}.png', dpi=150)
    plt.close()
    
    logger.info(f"Alignment metrics - Avg similarity: {avg_alignment:.3f}, V→L retrieval: {v2l_acc:.2%}")
    
    return avg_alignment, v2l_acc

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


class EfficientMultimodalDataset(Dataset):
    """Memory-efficient dataset that loads data on-demand"""
    
    def __init__(self, observation_ids: List[str], cache, device='cpu', species_mapping=None):
        self.observation_ids = observation_ids
        self.cache = cache
        self.device = device
        
        if species_mapping is not None:
            self.species_to_idx = species_mapping
            self.num_classes = len(species_mapping)
            logger.info(f"Using provided species mapping with {self.num_classes} classes")
        else:
            # Build species mapping from ALL observations
            logger.info("Building species mapping from all observations...")
            all_species = set()
            
            # Sample in batches to find all species
            batch_size = 100
            for i in range(0, len(self.observation_ids), batch_size):
                batch_ids = self.observation_ids[i:i+batch_size]
                batch_data = get_training_batch(
                    self.cache,
                    batch_ids,
                    include_vision=False,
                    include_language=True,
                    device='cpu'
                )
                all_species.update(batch_data['species'])
                
                if i % 1000 == 0:
                    logger.info(f"  Processed {i}/{len(self.observation_ids)} observations, "
                               f"found {len(all_species)} species so far")
            
            self.species_to_idx = {s: i for i, s in enumerate(sorted(all_species))}
            self.num_classes = len(self.species_to_idx)
            logger.info(f"Found {self.num_classes} unique species in dataset")
        
        self.valid_indices = list(range(len(observation_ids)))
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        obs_id = self.observation_ids[actual_idx]
        
        try:
            batch_data = get_training_batch(
                self.cache,
                [obs_id],
                include_vision=True,
                include_language=True,
                device='cpu'
            )
            
            species = batch_data['species'][0]
            
            if species not in self.species_to_idx:
                logger.warning(f"Species '{species}' not in mapping, skipping")
                return self.__getitem__(np.random.randint(0, len(self)))
            
            vision_emb = batch_data['vision_embeddings'][0]  # (8, 24, 24, 1408)
            language_emb = batch_data['language_embeddings'][0]  # (7168,)
            
            # Pool vision embedding
            vision_pooled = vision_emb.mean(dim=(0, 1, 2))  # (1408,)
            
            return {
                'vision_embedding': vision_pooled,
                'language_embedding': language_emb,
                'species_label': torch.tensor(self.species_to_idx[species], dtype=torch.long)
            }
            
        except Exception as e:
            logger.error(f"Error loading observation {obs_id}: {e}")
            return self.__getitem__(np.random.randint(0, len(self)))


class MultimodalAutoencoder(nn.Module):
    """
    Multimodal Autoencoder with separate MLP decoders
    Following meeting notes:
    - Vision encoder: 1408 → 2048
    - Language encoder: 7168 → 2048  
    - Separate 2-layer MLP decoders for each modality
    - Configurable masking per modality
    """
    
    def __init__(self, num_classes, vision_dim=1408, language_dim=7168, 
                 universal_dim=2048, hidden_dim=512, dropout_rate=0.2):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.universal_dim = universal_dim
        
        # Modality encoders to universal dimension
        self.vision_encoder = nn.Sequential(
            nn.Linear(vision_dim, universal_dim),
            nn.LayerNorm(universal_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.language_encoder = nn.Sequential(
            nn.Linear(language_dim, universal_dim),
            nn.LayerNorm(universal_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Fusion at universal dimension
        self.fusion = nn.Sequential(
            nn.Linear(universal_dim * 2, universal_dim),
            nn.LayerNorm(universal_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Classifier from universal embedding
        self.classifier = nn.Sequential(
            nn.Linear(universal_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 1.5),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Vision decoder - 2 layer MLP as specified
        self.vision_decoder = nn.Sequential(
            nn.Linear(universal_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, vision_dim)
        )
        
        # Language decoder - 2 layer MLP as specified  
        self.language_decoder = nn.Sequential(
            nn.Linear(universal_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, language_dim)
        )
    
    def forward(self, vision, language, mask_vision=False, mask_language=False, 
                vision_mask_ratio=0.5, language_mask_ratio=0.5, mask_type='feature'):
        """
        Forward pass with configurable masking per modality
        
        Args:
            vision: Vision embeddings (batch, 1408)
            language: Language embeddings (batch, 7168)
            mask_vision: Whether to mask vision modality
            mask_language: Whether to mask language modality
            vision_mask_ratio: Ratio of vision features to mask
            language_mask_ratio: Ratio of language features to mask
            mask_type: 'feature' (mask dimensions) or 'sample' (mask entire embeddings)
        """
        batch_size = vision.shape[0]
        device = vision.device
        
        # Store originals for reconstruction loss
        original_vision = vision.clone()
        original_language = language.clone()
        
        # Apply masking if requested
        vision_mask = None
        language_mask = None
        
        if mask_vision and self.training:
            if mask_type == 'sample':
                # Mask entire samples
                vision_mask = torch.rand(batch_size, device=device) < vision_mask_ratio
                vision = vision * (~vision_mask).float().unsqueeze(1)
            else:
                # Feature-level masking
                vision_mask = torch.rand(batch_size, self.vision_dim, device=device) < vision_mask_ratio
                vision = vision * (~vision_mask).float()
            
        if mask_language and self.training:
            if mask_type == 'sample':
                # Mask entire samples (50% of language embeddings completely zeroed)
                language_mask = torch.rand(batch_size, device=device) < language_mask_ratio
                language = language * (~language_mask).float().unsqueeze(1)
            else:
                # Feature-level masking (50% of dimensions zeroed for all samples)
                language_mask = torch.rand(batch_size, self.language_dim, device=device) < language_mask_ratio
                language = language * (~language_mask).float()
        
        # Encode to universal space
        vision_universal = self.vision_encoder(vision)      # 1408 → 2048
        language_universal = self.language_encoder(language)  # 7168 → 2048
        
        # Fusion
        fused = torch.cat([vision_universal, language_universal], dim=1)  # 4096
        bottleneck = self.fusion(fused)  # 4096 → 2048 (universal embedding)
        
        # Classification
        logits = self.classifier(bottleneck)
        
        # Decode from universal embedding
        vision_recon = self.vision_decoder(bottleneck)      # 2048 → 1408
        language_recon = self.language_decoder(bottleneck)  # 2048 → 7168
        
        return {
            'logits': logits,
            'vision_recon': vision_recon,
            'language_recon': language_recon,
            'bottleneck': bottleneck,  # The 1x2048 universal embedding
            'vision_universal': vision_universal,
            'language_universal': language_universal,
            'original_vision': original_vision,
            'original_language': original_language,
            'vision_mask': vision_mask,
            'language_mask': language_mask
        }


def collate_fn(batch):
    """Custom collate function to handle batching"""
    vision = torch.stack([item['vision_embedding'] for item in batch])
    language = torch.stack([item['language_embedding'] for item in batch])
    labels = torch.stack([item['species_label'] for item in batch])
    
    return {
        'vision_embedding': vision,
        'language_embedding': language,
        'species_label': labels
    }


def compute_species_aware_contrastive_loss(vision_universal, language_universal, labels, temperature=0.07):
    """
    Symmetric contrastive loss that treats all same-species pairs as positives.
    This handles the many-to-one nature of the data.
    """
    batch_size = vision_universal.shape[0]
    device = vision_universal.device
    
    # Normalize embeddings
    vision_norm = F.normalize(vision_universal, p=2, dim=1)
    language_norm = F.normalize(language_universal, p=2, dim=1)
    
    # Compute all pairwise similarities
    similarities = torch.matmul(vision_norm, language_norm.T) / temperature
    
    # Create label matrix - True where labels match
    labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
    
    # Vision → Language loss
    exp_sim = torch.exp(similarities)
    positive_sum = (exp_sim * labels_matrix.float()).sum(dim=1)
    total_sum = exp_sim.sum(dim=1)
    loss_v2l = -torch.log(positive_sum / total_sum).mean()
    
    # Language → Vision loss (symmetric)
    exp_sim_T = torch.exp(similarities.T)
    positive_sum_T = (exp_sim_T * labels_matrix.float()).sum(dim=1)
    total_sum_T = exp_sim_T.sum(dim=1)
    loss_l2v = -torch.log(positive_sum_T / total_sum_T).mean()
    
    # Symmetric loss
    loss = 0.5 * (loss_v2l + loss_l2v)
    
    return loss


def train_epoch_fixed(model, loader, optimizer, device, lambda_rec=0.1, lambda_contrast=0.1,
                mask_config=None, temperature=0.07, epoch=0):
    """Fixed training with species-aware contrastive loss"""
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_rec_loss = 0
    total_contrast_loss = 0
    correct = 0
    total = 0
    
    # For k-NN evaluation
    all_bottlenecks = []
    all_labels = []
    
    for i, batch in enumerate(loader):
        # Move to device
        vision = batch['vision_embedding'].to(device)
        language = batch['language_embedding'].to(device)
        labels = batch['species_label'].to(device)
        
        # Configure masking based on strategy
        mask_vision = False
        mask_language = False
        
        if mask_config:
            strategy = mask_config.get('strategy', 'none')
            if strategy == 'vision_only':
                mask_vision = True
            elif strategy == 'language_only':
                mask_language = True
            elif strategy == 'both':
                mask_vision = True
                mask_language = True
            elif strategy == 'alternate':
                if i % 2 == 0:
                    mask_language = True
                else:
                    mask_vision = True
            elif strategy == 'sample_level':
                # Mask entire samples instead of features
                mask_language = True
        
        # Forward pass
        outputs = model(
            vision, language, 
            mask_vision=mask_vision,
            mask_language=mask_language,
            vision_mask_ratio=mask_config.get('vision_ratio', 0.5),
            language_mask_ratio=mask_config.get('language_ratio', 0.5),
            mask_type=mask_config.get('mask_type', 'feature')
        )
        
        # Classification loss
        cls_loss = F.cross_entropy(outputs['logits'], labels)
        
        # Reconstruction losses
        vision_rec_loss = 0
        language_rec_loss = 0
        
        # Vision reconstruction loss
        if mask_vision and outputs['vision_mask'] is not None:
            mask = outputs['vision_mask']
            masked_errors = ((outputs['vision_recon'] - outputs['original_vision']) ** 2) * mask.float()
            vision_rec_loss = masked_errors.sum() / mask.float().sum()
        else:
            vision_rec_loss = F.mse_loss(outputs['vision_recon'], outputs['original_vision'])
        
        # Language reconstruction loss
        if mask_language and outputs['language_mask'] is not None:
            mask = outputs['language_mask']
            masked_errors = ((outputs['language_recon'] - outputs['original_language']) ** 2) * mask.float()
            language_rec_loss = masked_errors.sum() / mask.float().sum()
        else:
            language_rec_loss = F.mse_loss(outputs['language_recon'], outputs['original_language'])
        
        # Normalize by dimension
        vision_rec_loss = vision_rec_loss / vision.shape[1]
        language_rec_loss = language_rec_loss / language.shape[1]
        
        rec_loss = (vision_rec_loss + language_rec_loss) / 2
        
        # Species-aware contrastive loss with temperature
        contrast_loss = compute_species_aware_contrastive_loss(
            outputs['vision_universal'], 
            outputs['language_universal'], 
            labels,
            temperature=temperature
        )
        
        # Center regularization to prevent drift
        center_loss = F.mse_loss(
            outputs['vision_universal'].mean(0), 
            outputs['language_universal'].mean(0)
        )
        
        # Total loss with dynamic reconstruction weight
        if epoch > 5 and total > 0 and correct / total > 0.99:
            # Drop reconstruction after warm-up if classification is perfect
            effective_lambda_rec = 0.0
        else:
            effective_lambda_rec = lambda_rec
            
        loss = cls_loss + effective_lambda_rec * rec_loss + lambda_contrast * contrast_loss + 0.01 * center_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item() * len(labels)
        total_cls_loss += cls_loss.item() * len(labels)
        total_rec_loss += rec_loss.item() * len(labels)
        total_contrast_loss += contrast_loss.item() * len(labels)
        pred = outputs['logits'].argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += len(labels)
        
        # Store embeddings for k-NN
        all_bottlenecks.append(outputs['bottleneck'].detach().cpu())
        all_labels.append(labels.cpu())
        
        # Progress
        if i % 10 == 0:
            print(f"\r  Batch {i}/{len(loader)}: Loss={loss.item():.4f}, "
                  f"Acc={correct/total:.2%}, Contrast={contrast_loss.item():.3f}", end='', flush=True)
    
    # Compute k-NN accuracy
    all_bottlenecks = torch.cat(all_bottlenecks, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    knn = NearestNeighbors(n_neighbors=6, metric='cosine')
    knn.fit(all_bottlenecks)
    
    _, indices = knn.kneighbors(all_bottlenecks)
    knn_correct = 0
    for i, neighbors in enumerate(indices):
        neighbor_labels = all_labels[neighbors[1:]]  # Skip self
        if np.any(neighbor_labels == all_labels[i]):
            knn_correct += 1
    knn_acc = knn_correct / len(all_labels)
    
    print()  # New line
    return (total_loss / total, correct / total, total_cls_loss / total, 
            total_rec_loss / total, knn_acc, total_contrast_loss / total)


@torch.no_grad()
def evaluate_with_species_aware_retrieval(model, loader, device, lambda_rec=0.1):
    """
    Evaluation that accounts for many-to-one nature of vision-to-language mapping.
    Multiple images can have the same species (same language description).
    """
    model.eval()
    total_loss = 0
    total_cls_loss = 0
    total_rec_loss = 0
    correct = 0
    total = 0
    
    # For retrieval metrics
    all_vision_universal = []
    all_language_universal = []
    all_labels = []
    all_bottlenecks = []
    
    for batch in loader:
        vision = batch['vision_embedding'].to(device)
        language = batch['language_embedding'].to(device)
        labels = batch['species_label'].to(device)
        
        outputs = model(vision, language)
        
        # Losses
        cls_loss = F.cross_entropy(outputs['logits'], labels)
        vision_rec_loss = F.mse_loss(outputs['vision_recon'], vision) / vision.shape[1]
        language_rec_loss = F.mse_loss(outputs['language_recon'], language) / language.shape[1]
        rec_loss = (vision_rec_loss + language_rec_loss) / 2
        loss = cls_loss + lambda_rec * rec_loss
        
        total_loss += loss.item() * len(labels)
        total_cls_loss += cls_loss.item() * len(labels)
        total_rec_loss += rec_loss.item() * len(labels)
        pred = outputs['logits'].argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += len(labels)
        
        # Store embeddings
        all_vision_universal.append(outputs['vision_universal'])
        all_language_universal.append(outputs['language_universal'])
        all_labels.append(labels)
        all_bottlenecks.append(outputs['bottleneck'])
    
    # Compute retrieval metrics
    if len(all_vision_universal) > 0:
        all_vision = torch.cat(all_vision_universal, dim=0)
        all_language = torch.cat(all_language_universal, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_bottlenecks = torch.cat(all_bottlenecks, dim=0)
        
        # Normalize for cosine similarity
        vision_norm = F.normalize(all_vision, p=2, dim=1)
        language_norm = F.normalize(all_language, p=2, dim=1)
        bottleneck_norm = F.normalize(all_bottlenecks, p=2, dim=1)
        
        # Compute similarity matrix
        v2l_similarities = torch.matmul(vision_norm, language_norm.T)
        
        # 1. Standard R@1 retrieval (Vision → Language)
        v2l_r1_correct = 0
        for i in range(len(all_labels)):
            similarities = v2l_similarities[i]
            retrieved_idx = similarities.argmax().item()
            if all_labels[retrieved_idx] == all_labels[i]:
                v2l_r1_correct += 1
        v2l_r1 = v2l_r1_correct / len(all_labels)
        
        # 2. Species-aware retrieval R@5 (Vision → Language)
        v2l_species_correct = 0
        for i in range(len(all_labels)):
            similarities = v2l_similarities[i]
            
            # Get top-5 retrievals
            top_k = min(5, len(similarities))
            _, top_indices = similarities.topk(top_k)
            
            # Check if any of the top-k have the same species
            retrieved_labels = all_labels[top_indices]
            if (retrieved_labels == all_labels[i]).any():
                v2l_species_correct += 1
        
        v2l_species_acc = v2l_species_correct / len(all_labels)
        
        # 3. Language → Vision R@1 (species-aware)
        l2v_r1_correct = 0
        l2v_species_correct = 0
        l2v_similarities = v2l_similarities.T  # Transpose for L→V
        
        for i in range(len(all_labels)):
            similarities = l2v_similarities[i]
            
            # Standard R@1: exact match
            retrieved_idx = similarities.argmax().item()
            if all_labels[retrieved_idx] == all_labels[i]:
                l2v_r1_correct += 1
            
            # Species R@5: any of top-5 have same species
            top_k = min(5, len(similarities))
            _, top_indices = similarities.topk(top_k)
            retrieved_labels = all_labels[top_indices]
            if (retrieved_labels == all_labels[i]).any():
                l2v_species_correct += 1
        
        l2v_r1 = l2v_r1_correct / len(all_labels)
        l2v_r5 = l2v_species_correct / len(all_labels)
        
        # 4. Instance-level alignment (diagonal similarity)
        instance_similarities = torch.diag(v2l_similarities)
        instance_retrieval_score = instance_similarities.mean().item()
        
        # 5. Embedding space clustering quality
        similarities = torch.matmul(bottleneck_norm, bottleneck_norm.T)
        
        # Mask out self-similarities
        mask = torch.eye(len(all_labels), device=device).bool()
        similarities.masked_fill_(mask, -float('inf'))
        
        # Find nearest neighbor
        _, indices = similarities.max(dim=1)
        
        # Check if nearest neighbor has same species
        nn_same_species = (all_labels[indices] == all_labels).float().mean().item()
        
    else:
        v2l_r1 = 0.0
        v2l_species_acc = 0.0
        l2v_r1 = 0.0
        instance_retrieval_score = 0.0
        nn_same_species = 0.0
    
    return (total_loss / total, correct / total, total_cls_loss / total, 
            total_rec_loss / total, v2l_r1, v2l_species_acc, l2v_r1, l2v_r5,
            instance_retrieval_score, nn_same_species)


def train_epoch(model, loader, optimizer, device, lambda_rec=0.1, lambda_contrast=0.1,
                mask_config=None):
    """
    Train with configurable masking strategies
    
    mask_config: dict with keys:
        - strategy: 'none', 'vision_only', 'language_only', 'both', 'alternate'
        - vision_ratio: float (0-1)
        - language_ratio: float (0-1)
    """
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_rec_loss = 0
    correct = 0
    total = 0
    
    # For k-NN evaluation
    all_bottlenecks = []
    all_labels = []
    
    for i, batch in enumerate(loader):
        # Move to device
        vision = batch['vision_embedding'].to(device)
        language = batch['language_embedding'].to(device)
        labels = batch['species_label'].to(device)
        
        # Configure masking based on strategy
        mask_vision = False
        mask_language = False
        
        if mask_config:
            strategy = mask_config.get('strategy', 'none')
            if strategy == 'vision_only':
                mask_vision = True
            elif strategy == 'language_only':
                mask_language = True
            elif strategy == 'both':
                mask_vision = True
                mask_language = True
            elif strategy == 'alternate':
                # Alternate between masking vision and language each batch
                if i % 2 == 0:
                    mask_language = True
                else:
                    mask_vision = True
        
        # Forward pass
        outputs = model(
            vision, language, 
            mask_vision=mask_vision,
            mask_language=mask_language,
            vision_mask_ratio=mask_config.get('vision_ratio', 0.5),
            language_mask_ratio=mask_config.get('language_ratio', 0.5)
        )
        
        # Classification loss
        cls_loss = F.cross_entropy(outputs['logits'], labels)
        
        # Reconstruction losses
        vision_rec_loss = 0
        language_rec_loss = 0
        
        # Vision reconstruction loss (only on masked parts if masking)
        if mask_vision and outputs['vision_mask'] is not None:
            mask = outputs['vision_mask']
            masked_errors = ((outputs['vision_recon'] - outputs['original_vision']) ** 2) * mask.float()
            vision_rec_loss = masked_errors.sum() / mask.float().sum()
        else:
            vision_rec_loss = F.mse_loss(outputs['vision_recon'], outputs['original_vision'])
        
        # Language reconstruction loss (only on masked parts if masking)
        if mask_language and outputs['language_mask'] is not None:
            mask = outputs['language_mask']
            masked_errors = ((outputs['language_recon'] - outputs['original_language']) ** 2) * mask.float()
            language_rec_loss = masked_errors.sum() / mask.float().sum()
        else:
            language_rec_loss = F.mse_loss(outputs['language_recon'], outputs['original_language'])
        
        # Normalize by dimension
        vision_rec_loss = vision_rec_loss / vision.shape[1]
        language_rec_loss = language_rec_loss / language.shape[1]
        
        rec_loss = (vision_rec_loss + language_rec_loss) / 2
        
        # Add contrastive loss to align vision and language
        vision_norm = F.normalize(outputs['vision_universal'], p=2, dim=1)
        language_norm = F.normalize(outputs['language_universal'], p=2, dim=1)
        
        # Positive pairs: diagonal should be high
        positive_sim = (vision_norm * language_norm).sum(dim=1)
        
        # Negative pairs: off-diagonal should be low
        similarity_matrix = torch.matmul(vision_norm, language_norm.T)
        
        # Contrastive loss (InfoNCE style)
        temperature = 0.07
        labels_contrast = torch.arange(len(vision), device=device)
        contrast_loss = F.cross_entropy(similarity_matrix / temperature, labels_contrast)
        
        # Total loss
        loss = cls_loss + lambda_rec * rec_loss + lambda_contrast * contrast_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item() * len(labels)
        total_cls_loss += cls_loss.item() * len(labels)
        total_rec_loss += rec_loss.item() * len(labels)
        pred = outputs['logits'].argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += len(labels)
        
        # Store embeddings for k-NN
        all_bottlenecks.append(outputs['bottleneck'].detach().cpu())
        all_labels.append(labels.cpu())
        
        # Progress
        if i % 10 == 0:
            print(f"\r  Batch {i}/{len(loader)}: Loss={loss.item():.4f}, "
                  f"Acc={correct/total:.2%}", end='', flush=True)
    
    # Compute k-NN accuracy
    all_bottlenecks = torch.cat(all_bottlenecks, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    knn = NearestNeighbors(n_neighbors=6, metric='cosine')
    knn.fit(all_bottlenecks)
    
    _, indices = knn.kneighbors(all_bottlenecks)
    knn_correct = 0
    for i, neighbors in enumerate(indices):
        neighbor_labels = all_labels[neighbors[1:]]  # Skip self
        if np.any(neighbor_labels == all_labels[i]):
            knn_correct += 1
    knn_acc = knn_correct / len(all_labels)
    
    print()  # New line
    return (total_loss / total, correct / total, total_cls_loss / total, 
            total_rec_loss / total, knn_acc)


@torch.no_grad()
def evaluate(model, loader, device, lambda_rec=0.1):
    model.eval()
    total_loss = 0
    total_cls_loss = 0
    total_rec_loss = 0
    correct = 0
    total = 0
    
    # For cross-modal retrieval
    all_bottlenecks = []
    all_labels = []
    
    for batch in loader:
        vision = batch['vision_embedding'].to(device)
        language = batch['language_embedding'].to(device)
        labels = batch['species_label'].to(device)
        
        outputs = model(vision, language)
        
        # Losses
        cls_loss = F.cross_entropy(outputs['logits'], labels)
        vision_rec_loss = F.mse_loss(outputs['vision_recon'], vision) / vision.shape[1]
        language_rec_loss = F.mse_loss(outputs['language_recon'], language) / language.shape[1]
        rec_loss = (vision_rec_loss + language_rec_loss) / 2
        loss = cls_loss + lambda_rec * rec_loss
        
        total_loss += loss.item() * len(labels)
        total_cls_loss += cls_loss.item() * len(labels)
        total_rec_loss += rec_loss.item() * len(labels)
        pred = outputs['logits'].argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += len(labels)
        
        all_bottlenecks.append(outputs['bottleneck'])
        all_labels.append(labels)
    
    # Compute retrieval metrics
    if len(all_bottlenecks) > 1:
        all_bottlenecks = torch.cat(all_bottlenecks, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Normalize
        all_bottlenecks = F.normalize(all_bottlenecks, p=2, dim=1)
        
        # Compute similarities
        similarities = torch.matmul(all_bottlenecks, all_bottlenecks.T)
        
        # Mask self
        mask = torch.eye(len(all_labels), device=device).bool()
        similarities.masked_fill_(mask, -float('inf'))
        
        # Find top-1
        _, indices = similarities.max(dim=1)
        
        # Check if retrieved has same species
        retrieval_correct = (all_labels[indices] == all_labels).float().mean().item()
    else:
        retrieval_correct = 0.0
    
    return (total_loss / total, correct / total, total_cls_loss / total, 
            total_rec_loss / total, retrieval_correct)


def load_split(config_path, max_species=None, train_ids_subset=None, test_ids_subset=None):
    """Load train/test split and species mapping, optionally filtered to most common species"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Use provided subsets or load all
    if train_ids_subset is None or test_ids_subset is None:
        train_ids = []
        test_ids = []
        for obs_id, meta in config['observation_mappings'].items():
            if meta['split'] == 'train':
                train_ids.append(obs_id)
            else:
                test_ids.append(obs_id)
        
        if train_ids_subset is not None:
            train_ids = train_ids_subset
        if test_ids_subset is not None:
            test_ids = test_ids_subset
    else:
        train_ids = train_ids_subset
        test_ids = test_ids_subset
    
    # Count species frequencies in the subset
    from collections import Counter
    species_counts = Counter()
    
    for obs_id in train_ids:
        if obs_id in config['observation_mappings']:
            species_counts[config['observation_mappings'][obs_id]['taxon_name']] += 1
    
    # Filter to top N species if requested
    if max_species and max_species < len(species_counts):
        top_species = set(species for species, _ in species_counts.most_common(max_species))
        logger.info(f"Filtering to top {max_species} most common species from {len(train_ids)} samples")
        
        # Show the selected species
        logger.info("Selected species (by frequency):")
        for i, (species, count) in enumerate(species_counts.most_common(max_species)):
            logger.info(f"  {i+1:2d}. {count:4d} samples: {species}")
    else:
        top_species = set(species_counts.keys())
    
    # Now filter observations to only include selected species
    filtered_train_ids = []
    filtered_test_ids = []
    
    for obs_id in train_ids:
        if obs_id in config['observation_mappings']:
            if config['observation_mappings'][obs_id]['taxon_name'] in top_species:
                filtered_train_ids.append(obs_id)
    
    for obs_id in test_ids:
        if obs_id in config['observation_mappings']:
            if config['observation_mappings'][obs_id]['taxon_name'] in top_species:
                filtered_test_ids.append(obs_id)
    
    # Create species mapping only for selected species
    species_to_idx = {species: idx for idx, species in enumerate(sorted(top_species))}
    logger.info(f"Filtered dataset: {len(filtered_train_ids)} train, {len(filtered_test_ids)} test, {len(species_to_idx)} species")
    
    return filtered_train_ids, filtered_test_ids, species_to_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--config', type=str, default='config/central_florida_split.json')
    parser.add_argument('--lambda-rec', type=float, default=0.1, help='Reconstruction loss weight')
    parser.add_argument('--lambda-contrast', type=float, default=0.5, help='Contrastive loss weight')
    parser.add_argument('--hidden-dim', type=int, default=512, help='Hidden dimension for decoders')
    parser.add_argument('--universal-dim', type=int, default=2048, help='Universal embedding dimension')
    parser.add_argument('--subset', type=int, default=None, help='Use subset of data for testing')
    parser.add_argument('--mask-strategy', type=str, default='language_only',
                       choices=['none', 'vision_only', 'language_only', 'both', 'alternate'],
                       help='Masking strategy for training')
    parser.add_argument('--vision-mask-ratio', type=float, default=0.5)
    parser.add_argument('--language-mask-ratio', type=float, default=0.5)
    parser.add_argument('--max-species', type=int, default=None,
                       help='Maximum number of species to use (filters to most common)')
    parser.add_argument('--auto-species', action='store_true',
                       help='Automatically choose number of species to reach subset target')
    parser.add_argument('--use-all-data', action='store_true',
                       help='Use all data for training (no train/test split)')
    parser.add_argument('--balanced', action='store_true',
                       help='Use balanced sampling (equal samples per species)')
    parser.add_argument('--mask-type', type=str, default='feature', choices=['feature', 'sample'],
                       help='Masking type: feature (mask dimensions) or sample (mask entire embeddings)')
    parser.add_argument('--temperature', type=float, default=0.07, 
                       help='Temperature for contrastive loss')
    parser.add_argument('--save-dir', type=str, default='checkpoints_autoencoder',
                       help='Directory to save checkpoints')
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load config and handle balanced sampling
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if args.use_all_data:
        # Use all available data for training
        logger.info("Using all data for training (no train/test split)")
        
        from collections import Counter
        
        # Get ALL observation IDs
        all_obs_ids = list(config['observation_mappings'].keys())
        
        # Count species
        species_counts = Counter()
        obs_ids_by_species = {}
        
        for obs_id, meta in config['observation_mappings'].items():
            species = meta['taxon_name']
            species_counts[species] += 1
            if species not in obs_ids_by_species:
                obs_ids_by_species[species] = []
            obs_ids_by_species[species].append(obs_id)
        
        # Select top species
        if args.max_species:
            top_species = [s for s, _ in species_counts.most_common(args.max_species)]
            selected_obs_ids = []
            for species in top_species:
                selected_obs_ids.extend(obs_ids_by_species[species])
            
            species_mapping = {species: idx for idx, species in enumerate(sorted(top_species))}
            logger.info(f"Selected {len(top_species)} species with {len(selected_obs_ids)} total observations")
        else:
            selected_obs_ids = all_obs_ids
            all_species = sorted(species_counts.keys())
            species_mapping = {species: idx for idx, species in enumerate(all_species)}
        
        # Apply subset if requested
        if args.subset and args.subset < len(selected_obs_ids):
            selected_obs_ids = selected_obs_ids[:args.subset]
        
        # Use same data for both "train" and "test" 
        # (test is just for evaluation metrics, not real testing)
        train_ids = selected_obs_ids
        
        # Use more samples for stable evaluation metrics
        eval_size = min(1000, len(selected_obs_ids) // 5)  # 20% of data or 1000, whichever is smaller
        test_ids = selected_obs_ids[:eval_size]
        
        logger.info(f"Total samples: {len(train_ids)}")
        logger.info(f"Using {len(test_ids)} samples for evaluation metrics (contrastive signals need more data)")
        
    elif args.balanced and args.subset and args.max_species:
        # Smart balanced sampling: maximize samples while respecting data limits
        logger.info(f"Using smart balanced sampling: up to {args.subset} samples across {args.max_species} species")
        
        from collections import Counter
        
        # Count species in training
        species_counts = Counter()
        train_ids_by_species = {}
        
        for obs_id, meta in config['observation_mappings'].items():
            if meta['split'] == 'train':
                species = meta['taxon_name']
                species_counts[species] += 1
                if species not in train_ids_by_species:
                    train_ids_by_species[species] = []
                train_ids_by_species[species].append(obs_id)
        
        # Get top species by count
        top_species = species_counts.most_common(args.max_species)
        
        # Strategy: Take all available samples from top species, up to subset limit
        train_ids_subset = []
        test_ids_subset = []
        selected_species = []
        samples_per_species_actual = {}
        
        # First pass: see how many samples we can get from top species
        total_available = sum(min(count, args.subset // args.max_species * 2) for species, count in top_species)
        
        if total_available < args.subset:
            # Take all samples from top species
            for species, count in top_species:
                species_train_ids = train_ids_by_species[species]
                train_ids_subset.extend(species_train_ids)
                selected_species.append(species)
                samples_per_species_actual[species] = len(species_train_ids)
        else:
            # Distribute samples proportionally
            # Start with minimum samples per species
            min_samples = 20  # Minimum to ensure decent learning
            remaining_samples = args.subset
            
            # First, allocate minimum to each species
            for species, count in top_species:
                if count >= min_samples and remaining_samples >= min_samples:
                    take = min(min_samples, count)
                    species_train_ids = train_ids_by_species[species][:take]
                    train_ids_subset.extend(species_train_ids)
                    selected_species.append(species)
                    samples_per_species_actual[species] = take
                    remaining_samples -= take
            
            # Then distribute remaining samples proportionally
            if remaining_samples > 0 and selected_species:
                total_count = sum(count for species, count in top_species if species in selected_species)
                
                for species in selected_species:
                    if remaining_samples <= 0:
                        break
                    
                    species_count = species_counts[species]
                    current_samples = samples_per_species_actual[species]
                    
                    # Proportional allocation of remaining samples
                    proportion = species_count / total_count
                    additional = int(remaining_samples * proportion)
                    
                    # Don't exceed available samples for this species
                    can_add = min(additional, species_count - current_samples)
                    
                    if can_add > 0:
                        species_train_ids = train_ids_by_species[species][current_samples:current_samples + can_add]
                        train_ids_subset.extend(species_train_ids)
                        samples_per_species_actual[species] += len(species_train_ids)
                        remaining_samples -= len(species_train_ids)
        
        # Add test samples for selected species
        min_test_per_species = 5  # Ensure at least 5 test samples per species
        for species in selected_species:
            test_ids_for_species = [obs_id for obs_id, meta in config['observation_mappings'].items() 
                                  if meta['split'] == 'test' and meta['taxon_name'] == species]
            
            # If not enough test samples, take some from training
            if len(test_ids_for_species) < min_test_per_species:
                # Get species training IDs we haven't used yet
                species_all_train = train_ids_by_species[species]
                used_count = samples_per_species_actual[species]
                available_for_test = species_all_train[used_count:]
                
                # Move some training samples to test to ensure minimum
                needed = min_test_per_species - len(test_ids_for_species)
                if len(available_for_test) >= needed:
                    # Remove from train and add to test
                    move_to_test = available_for_test[:needed]
                    test_ids_for_species.extend(move_to_test)
                    
                    # Remove these from train_ids_subset
                    train_ids_subset = [id for id in train_ids_subset if id not in move_to_test]
                    
                    logger.info(f"  Moved {needed} samples from train to test for species {species}")
            
            test_ids_subset.extend(test_ids_for_species[:min_test_per_species])
        
        # Create species mapping
        species_mapping = {species: idx for idx, species in enumerate(sorted(selected_species))}
        
        # Log distribution
        logger.info(f"Smart balanced dataset created:")
        logger.info(f"  Species: {len(selected_species)}")
        logger.info(f"  Train samples: {len(train_ids_subset)}")
        logger.info(f"  Test samples: {len(test_ids_subset)}")
        logger.info(f"\nSamples per species:")
        for species in sorted(selected_species):
            logger.info(f"    {species}: {samples_per_species_actual[species]} samples")
        
        train_ids = train_ids_subset
        test_ids = test_ids_subset
        
    else:
        # Original loading logic
        all_train_ids = []
        all_test_ids = []
        for obs_id, meta in config['observation_mappings'].items():
            if meta['split'] == 'train':
                all_train_ids.append(obs_id)
            else:
                all_test_ids.append(obs_id)
        
        logger.info(f"Total available: {len(all_train_ids)} train, {len(all_test_ids)} test")
        
        # Apply subset FIRST
        if args.subset:
            train_ids_subset = all_train_ids[:args.subset]
            test_ids_subset = all_test_ids[:min(args.subset//5, len(all_test_ids))]
        else:
            train_ids_subset = all_train_ids
            test_ids_subset = all_test_ids
        
        # Then filter by species
        train_ids, test_ids, species_mapping = load_split(
            str(config_path), 
            max_species=args.max_species,
            train_ids_subset=train_ids_subset,
            test_ids_subset=test_ids_subset
        )
    logger.info(f"Split: {len(train_ids)} train, {len(test_ids)} test")
    
    # Change to dashboard directory
    original_dir = os.getcwd()
    os.chdir(dashboard_path)
    
    try:
        # Initialize cache
        cache = UnifiedDataCache("dataset_config.json")
        
        # Create datasets
        if args.subset:
            logger.info(f"Using subset of {args.subset} samples")
            # IMPORTANT: Take subset BEFORE species filtering
            train_ids = train_ids[:args.subset]
            test_ids = test_ids[:min(args.subset//5, len(test_ids))]
        
        # NOW filter to top species from the subset
        train_ids, test_ids, species_mapping = load_split(
            str(config_path), 
            max_species=args.max_species,
            train_ids_subset=train_ids,
            test_ids_subset=test_ids
        )
        
        logger.info("Creating training dataset...")
        train_dataset = EfficientMultimodalDataset(
            train_ids, cache, species_mapping=species_mapping
        )
        
        logger.info("Creating test dataset...")
        test_dataset = EfficientMultimodalDataset(
            test_ids, cache, species_mapping=species_mapping
        )
        
        # Debug: Check test set distribution
        test_labels = []
        for i in range(len(test_dataset)):
            test_labels.append(test_dataset[i]['species_label'].item())
        
        from collections import Counter
        test_distribution = Counter(test_labels)
        logger.info(f"Test set species distribution: {dict(test_distribution)}")
        logger.info(f"Species in test set: {len(test_distribution)} out of {len(species_mapping)}")
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=(device == 'cuda')
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=(device == 'cuda')
        )
        
        # Create model
        model = MultimodalAutoencoder(
            num_classes=train_dataset.num_classes,
            hidden_dim=args.hidden_dim,
            universal_dim=args.universal_dim
        ).to(device)
        
        logger.info(f"Multimodal Autoencoder initialized:")
        logger.info(f"  Vision: 1408 → {args.universal_dim}")
        logger.info(f"  Language: 7168 → {args.universal_dim}")
        logger.info(f"  Decoders: 2-layer MLPs with hidden_dim={args.hidden_dim}")
        logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Training setup
        Path(args.save_dir).mkdir(exist_ok=True)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        
        # Use ReduceLROnPlateau instead of CosineAnnealingLR
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',  # Maximize score
            factor=0.5,  # Reduce LR by half
            patience=5   # Wait 5 epochs before reducing
        )
        
        # Masking configuration
        mask_config = {
            'strategy': args.mask_strategy,
            'vision_ratio': args.vision_mask_ratio,
            'language_ratio': args.language_mask_ratio,
            'mask_type': args.mask_type
        }
        
        print("\n" + "="*80)
        print("🚀 STARTING MULTIMODAL AUTOENCODER TRAINING")
        print("="*80)
        print(f"Architecture: Vision(1408) + Language(7168) → Universal({args.universal_dim})")
        print(f"Decoders: 2-layer MLPs (hidden_dim={args.hidden_dim})")
        print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
        print(f"Masking strategy: {args.mask_strategy}")
        if args.mask_strategy != 'none':
            print(f"  Vision mask ratio: {args.vision_mask_ratio}")
            print(f"  Language mask ratio: {args.language_mask_ratio}")
        print("="*80 + "\n")
        
        # Track multiple metrics for best model
        best_metrics = {
            'epoch': 0,
            'test_acc': 0,
            'v2l_r1': 0,
            'v2l_r5': 0,
            'l2v_r1': 0,
            'l2v_r5': 0,
            'instance_alignment': 0,
            'combined_score': 0
        }
        
        # Track the best value for EACH metric individually
        best_individual_metrics = {
            'test_acc': {'value': 0, 'epoch': 0},
            'v2l_r1': {'value': 0, 'epoch': 0},
            'v2l_r5': {'value': 0, 'epoch': 0},
            'l2v_r1': {'value': 0, 'epoch': 0},
            'l2v_r5': {'value': 0, 'epoch': 0},
            'instance_alignment': {'value': 0, 'epoch': 0},
            'nn_same_species': {'value': 0, 'epoch': 0},
            'train_knn': {'value': 0, 'epoch': 0},
            'avg_contrast_loss': {'value': float('inf'), 'epoch': 0}  # Lower is better
        }
        
        # For tracking history
        history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'v2l_r1': [],
            'v2l_r5': [],
            'l2v_r1': [], 
            'l2v_r5': [],
            'alignment': [],
            'nn_same_species': [],
            'train_knn': [],
            'contrast_loss': []
        }
        
        for epoch in range(args.epochs):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1}/{args.epochs}")
            print(f"{'='*60}")
            
            # Train with species-aware contrastive loss
            print("Training...")
            train_loss, train_acc, train_cls, train_rec, train_knn, train_contrast = train_epoch_fixed(
                model, train_loader, optimizer, device, 
                lambda_rec=args.lambda_rec,
                lambda_contrast=args.lambda_contrast,
                mask_config=mask_config,
                temperature=args.temperature,
                epoch=epoch
            )
            
            print(f"\n📊 Train Results:")
            print(f"   Loss: {train_loss:.4f} (Cls: {train_cls:.4f}, Rec: {train_rec:.4f}, Contrast: {train_contrast:.4f})")
            print(f"   Classification Accuracy: {train_acc:.2%}")
            print(f"   k-NN Accuracy (k=5): {train_knn:.2%}")
            
            # Evaluate with species-aware metrics
            print("\nEvaluating...")
            test_loss, test_acc, test_cls, test_rec, v2l_r1, v2l_r5, l2v_r1, l2v_r5, instance_sim, nn_species = evaluate_with_species_aware_retrieval(
                model, test_loader, device, args.lambda_rec
            )
            
            print(f"\n📊 Test Results:")
            print(f"   Loss: {test_loss:.4f} (Cls: {test_cls:.4f}, Rec: {test_rec:.4f})")
            print(f"   Classification Accuracy: {test_acc:.2%}")
            print(f"   V→L Retrieval R@1: {v2l_r1:.2%}")
            print(f"   V→L Species R@5: {v2l_r5:.2%}")
            print(f"   L→V Species R@1: {l2v_r1:.2%}")
            print(f"   L→V Species R@5: {l2v_r5:.2%}")
            print(f"   Instance Alignment: {instance_sim:.3f}")
            print(f"   NN Same Species: {nn_species:.2%}")
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            history['v2l_r1'].append(v2l_r1)
            history['v2l_r5'].append(v2l_r5)
            history['l2v_r1'].append(l2v_r1)
            history['l2v_r5'].append(l2v_r5)
            history['alignment'].append(instance_sim)
            history['nn_same_species'].append(nn_species)
            history['train_knn'].append(train_knn)
            history['contrast_loss'].append(train_contrast)
            
            # Update best individual metrics
            current_metrics = {
                'test_acc': test_acc,
                'v2l_r1': v2l_r1,
                'v2l_r5': v2l_r5,
                'l2v_r1': l2v_r1,
                'l2v_r5': l2v_r5,
                'instance_alignment': instance_sim,
                'nn_same_species': nn_species,
                'train_knn': train_knn,
                'avg_contrast_loss': train_contrast
            }
            
            # Check if any individual metric is best
            new_bests = []
            for metric_name, metric_value in current_metrics.items():
                if metric_name == 'avg_contrast_loss':
                    # Lower is better for loss
                    if metric_value < best_individual_metrics[metric_name]['value']:
                        best_individual_metrics[metric_name]['value'] = metric_value
                        best_individual_metrics[metric_name]['epoch'] = epoch + 1
                        new_bests.append(f"{metric_name}: {metric_value:.4f}")
                else:
                    # Higher is better for accuracy/retrieval metrics
                    if metric_value > best_individual_metrics[metric_name]['value']:
                        best_individual_metrics[metric_name]['value'] = metric_value
                        best_individual_metrics[metric_name]['epoch'] = epoch + 1
                        new_bests.append(f"{metric_name}: {metric_value:.2%}" if metric_value <= 1 else f"{metric_name}: {metric_value:.3f}")
            
            if new_bests:
                print(f"\n🌟 New best(s): {', '.join(new_bests)}")
            
            # Create alignment visualization every 3 epochs
            if (epoch + 1) % 3 == 0 or epoch == 0:
                print("\n📊 Creating alignment visualization...")
                align_dir = Path(args.save_dir) / 'alignment_visualizations'
                align_dir.mkdir(exist_ok=True)
                
                avg_sim, v2l_acc = create_alignment_visualization(
                    model, test_loader, device, epoch + 1, align_dir
                )
            
            # Calculate combined score for determining best model
            combined_score = (test_acc + v2l_r1 + v2l_r5 + l2v_r1 + l2v_r5 + instance_sim) / 6
            
            # Update best metrics if this is better
            if combined_score > best_metrics['combined_score']:
                best_metrics.update({
                    'epoch': epoch + 1,
                    'test_acc': test_acc,
                    'v2l_r1': v2l_r1,
                    'v2l_r5': v2l_r5,
                    'l2v_r1': l2v_r1,
                    'l2v_r5': l2v_r5,
                    'instance_alignment': instance_sim,
                    'combined_score': combined_score
                })
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'test_acc': test_acc,
                    'v2l_r1': v2l_r1,
                    'v2l_species_r5': v2l_r5,
                    'l2v_species_r1': l2v_r1,
                    'l2v_species_r5': l2v_r5,
                    'instance_alignment': instance_sim,
                    'nn_same_species': nn_species,
                    'train_knn': train_knn,
                    'args': vars(args),
                    'species_mapping': species_mapping,
                    'history': history,
                    'best_individual_metrics': best_individual_metrics
                }, Path(args.save_dir) / 'autoencoder_best.pth')
                print(f"\n🎯 NEW BEST MODEL! Combined Score: {combined_score:.3f}")
            else:
                print(f"\n   Best model: Score={best_metrics['combined_score']:.3f}, Acc={best_metrics['test_acc']:.2%} (Epoch {best_metrics['epoch']})")
            
            # Update scheduler with combined score
            scheduler.step(combined_score)
        
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE!")
        print(f"🏆 Best Overall Model (Epoch {best_metrics['epoch']}):")
        print(f"   Combined Score: {best_metrics['combined_score']:.3f}")
        print(f"   Test Accuracy: {best_metrics['test_acc']:.2%}")
        print(f"   V→L Retrieval: R@1={best_metrics['v2l_r1']:.2%}, R@5={best_metrics['v2l_r5']:.2%}")
        print(f"   L→V Retrieval: R@1={best_metrics['l2v_r1']:.2%}, R@5={best_metrics['l2v_r5']:.2%}")
        print(f"   Instance Alignment: {best_metrics['instance_alignment']:.3f}")
        
        print(f"\n📊 Best Individual Metrics Achieved:")
        for metric_name, metric_info in best_individual_metrics.items():
            if metric_name == 'avg_contrast_loss':
                print(f"   {metric_name}: {metric_info['value']:.4f} (Epoch {metric_info['epoch']})")
            elif metric_info['value'] <= 1:
                print(f"   {metric_name}: {metric_info['value']:.2%} (Epoch {metric_info['epoch']})")
            else:
                print(f"   {metric_name}: {metric_info['value']:.3f} (Epoch {metric_info['epoch']})")
        
        # Save final metrics summary
        metrics_summary = {
            'best_combined_model': best_metrics,
            'best_individual_metrics': best_individual_metrics,
            'final_history': {k: v[-1] if v else None for k, v in history.items()},
            'training_config': vars(args)
        }
        
        with open(Path(args.save_dir) / 'training_metrics_summary.json', 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        
        print(f"\n📁 Saved metrics summary to {Path(args.save_dir) / 'training_metrics_summary.json'}")
        print("="*80)
        
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    main()

