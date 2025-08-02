"""
Training script for DeepEarth multimodal model
Implements self-supervised learning with spatiotemporal masking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import wandb

from models.deepearth_multimodal import create_deepearth_model


class DeepEarthLoss(nn.Module):
    """
    Combined loss for DeepEarth training:
    - Spatiotemporal reconstruction loss
    - Modality reconstruction loss (MAE-style)
    - Cross-modal contrastive loss (CLIP-style)
    - Optional: Next-frame prediction loss
    """
    
    def __init__(
        self,
        spatial_weight: float = 1.0,
        temporal_weight: float = 1.0,
        vision_weight: float = 1.0,
        language_weight: float = 1.0,
        contrastive_weight: float = 1.0,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight
        self.vision_weight = vision_weight
        self.language_weight = language_weight
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        
    def spatial_reconstruction_loss(self, pred_xyz: torch.Tensor, true_xyz: torch.Tensor) -> torch.Tensor:
        """MSE loss for spatial coordinate reconstruction"""
        return F.mse_loss(pred_xyz, true_xyz)
    
    def temporal_reconstruction_loss(self, pred_t: torch.Tensor, true_t: torch.Tensor) -> torch.Tensor:
        """MSE loss for temporal coordinate reconstruction"""
        return F.mse_loss(pred_t, true_t)
    
    def vision_reconstruction_loss(
        self, 
        pred_vision: torch.Tensor, 
        true_vision: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """MAE-style loss for vision reconstruction"""
        if mask is not None:
            pred_vision = pred_vision[mask]
            true_vision = true_vision[mask]
        return F.mse_loss(pred_vision, true_vision)
    
    def language_reconstruction_loss(
        self,
        pred_tokens: torch.Tensor,
        true_tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Cross-entropy loss for masked language modeling"""
        if mask is not None:
            pred_tokens = pred_tokens[mask]
            true_tokens = true_tokens[mask]
        return F.cross_entropy(
            pred_tokens.reshape(-1, pred_tokens.size(-1)),
            true_tokens.reshape(-1),
            ignore_index=-100
        )
    
    def contrastive_loss(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor
    ) -> torch.Tensor:
        """CLIP-style contrastive loss between modalities"""
        # Normalize embeddings
        a_norm = F.normalize(embeddings_a, dim=-1)
        b_norm = F.normalize(embeddings_b, dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(a_norm, b_norm.T) / self.temperature
        
        # Create targets (diagonal elements are positive pairs)
        targets = torch.arange(logits.size(0), device=logits.device)
        
        # Compute loss both ways
        loss_a = F.cross_entropy(logits, targets)
        loss_b = F.cross_entropy(logits.T, targets)
        
        return (loss_a + loss_b) / 2
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss and individual components
        """
        losses = {}
        total_loss = 0.0
        
        # Spatial reconstruction
        if 'xyz' in outputs['reconstructions'] and 'xyz' in targets:
            loss_xyz = self.spatial_reconstruction_loss(
                outputs['reconstructions']['xyz'],
                targets['xyz']
            )
            losses['spatial'] = loss_xyz
            total_loss += self.spatial_weight * loss_xyz
        
        # Temporal reconstruction
        if 't' in outputs['reconstructions'] and 't' in targets:
            loss_t = self.temporal_reconstruction_loss(
                outputs['reconstructions']['t'],
                targets['t']
            )
            losses['temporal'] = loss_t
            total_loss += self.temporal_weight * loss_t
        
        # Vision reconstruction
        if 'vision' in outputs['reconstructions'] and 'vision' in targets:
            loss_vision = self.vision_reconstruction_loss(
                outputs['reconstructions']['vision'],
                targets['vision'],
                masks.get('vision')
            )
            losses['vision'] = loss_vision
            total_loss += self.vision_weight * loss_vision
        
        # Language reconstruction
        if 'language' in outputs['reconstructions'] and 'language' in targets:
            loss_language = self.language_reconstruction_loss(
                outputs['reconstructions']['language'],
                targets['language'],
                masks.get('language')
            )
            losses['language'] = loss_language
            total_loss += self.language_weight * loss_language
        
        # Cross-modal contrastive losses
        if 'vision_pool' in outputs and 'language_pool' in outputs:
            loss_vl = self.contrastive_loss(
                outputs['vision_pool'],
                outputs['language_pool']
            )
            losses['vision_language_contrastive'] = loss_vl
            total_loss += self.contrastive_weight * loss_vl
        
        if 'vision_pool' in outputs and 'spatial_pool' in outputs:
            loss_vs = self.contrastive_loss(
                outputs['vision_pool'],
                outputs['spatial_pool']
            )
            losses['vision_spatial_contrastive'] = loss_vs
            total_loss += self.contrastive_weight * loss_vs
        
        losses['total'] = total_loss
        return total_loss, losses


class SpatiotemporalMasking:
    """
    Implements various masking strategies for self-supervised learning
    """
    
    def __init__(
        self,
        spatial_mask_ratio: float = 0.15,
        temporal_mask_ratio: float = 0.15,
        vision_mask_ratio: float = 0.75,  # High masking ratio like MAE
        language_mask_ratio: float = 0.15,  # Standard MLM ratio
    ):
        self.spatial_mask_ratio = spatial_mask_ratio
        self.temporal_mask_ratio = temporal_mask_ratio
        self.vision_mask_ratio = vision_mask_ratio
        self.language_mask_ratio = language_mask_ratio
    
    def mask_spatial_coordinates(
        self, 
        xyz: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mask spatial coordinates by setting some to zero"""
        B = xyz.shape[0]
        mask = torch.rand(B) < self.spatial_mask_ratio
        masked_xyz = xyz.clone()
        masked_xyz[mask] = 0
        return masked_xyz, mask
    
    def mask_temporal_coordinates(
        self,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mask temporal coordinates"""
        B = t.shape[0]
        mask = torch.rand(B) < self.temporal_mask_ratio
        masked_t = t.clone()
        masked_t[mask] = 0
        return masked_t, mask
    
    def mask_vision_patches(
        self,
        num_patches: int,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """Random masking of vision patches (MAE-style)"""
        num_masked = int(num_patches * self.vision_mask_ratio)
        mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
        
        for i in range(batch_size):
            masked_indices = torch.randperm(num_patches)[:num_masked]
            mask[i, masked_indices] = True
            
        return mask
    
    def mask_language_tokens(
        self,
        input_ids: torch.Tensor,
        special_token_ids: set
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mask language tokens (MLM-style)"""
        mask = torch.rand_like(input_ids, dtype=torch.float) < self.language_mask_ratio
        
        # Don't mask special tokens
        for token_id in special_token_ids:
            mask &= (input_ids != token_id)
        
        masked_input_ids = input_ids.clone()
        masked_input_ids[mask] = 103  # [MASK] token ID (adjust based on tokenizer)
        
        return masked_input_ids, mask


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: DeepEarthLoss,
    masking: SpatiotemporalMasking,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    total_losses = {
        'total': 0.0,
        'spatial': 0.0,
        'temporal': 0.0,
        'vision': 0.0,
        'language': 0.0,
        'vision_language_contrastive': 0.0,
        'vision_spatial_contrastive': 0.0,
    }
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        xyzt = batch['xyzt'].to(device)
        images = batch.get('images', None)
        input_ids = batch.get('input_ids', None)
        attention_mask = batch.get('attention_mask', None)
        modalities = batch.get('modalities', None)
        
        # Apply masking
        masks = {}
        targets = {}
        
        # Mask spatial coordinates
        xyz = xyzt[:, :3]
        t = xyzt[:, 3:4]
        masked_xyz, spatial_mask = masking.mask_spatial_coordinates(xyz)
        masked_t, temporal_mask = masking.mask_temporal_coordinates(t)
        masked_xyzt = torch.cat([masked_xyz, masked_t], dim=1)
        
        targets['xyz'] = xyz
        targets['t'] = t
        masks['spatial'] = spatial_mask
        masks['temporal'] = temporal_mask
        
        # Mask vision if provided
        if images is not None:
            images = images.to(device)
            # Get vision features for targets before masking
            with torch.no_grad():
                vision_features = model.encode_vision(images)
            targets['vision'] = vision_features
            
            # Create vision mask
            num_patches = vision_features.shape[1]
            vision_mask = masking.mask_vision_patches(num_patches, images.shape[0], device)
            masks['vision'] = vision_mask
        
        # Mask language if provided
        if input_ids is not None:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device) if attention_mask is not None else None
            
            # Mask tokens
            special_tokens = {0, 1, 2, 3}  # Adjust based on tokenizer
            masked_input_ids, language_mask = masking.mask_language_tokens(input_ids, special_tokens)
            targets['language'] = input_ids
            masks['language'] = language_mask
            input_ids = masked_input_ids
        
        # Move additional modalities to device
        if modalities is not None:
            modalities = {k: v.to(device) for k, v in modalities.items()}
            # Store original values as targets
            for k, v in modalities.items():
                targets[k] = v
        
        # Forward pass
        outputs = model(
            xyzt=masked_xyzt,
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            modalities=modalities,
            return_reconstructions=True
        )
        
        # Compute loss
        loss, loss_components = loss_fn(outputs, targets, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update running losses
        for key, value in loss_components.items():
            if key in total_losses:
                total_losses[key] += value.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item(),
            'spatial': loss_components.get('spatial', 0).item(),
            'vision': loss_components.get('vision', 0).item(),
        })
        
        # Log to wandb
        if batch_idx % 100 == 0 and wandb.run is not None:
            wandb.log({
                f'train/{k}': v.item() 
                for k, v in loss_components.items()
            })
    
    # Average losses
    num_batches = len(dataloader)
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    
    return avg_losses


def train_deepearth(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.05,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    checkpoint_dir: str = './checkpoints',
    use_wandb: bool = True,
):
    """Main training loop for DeepEarth"""
    
    # Initialize wandb
    if use_wandb:
        wandb.init(
            project="deepearth",
            config={
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "num_epochs": num_epochs,
                "batch_size": train_dataloader.batch_size,
            }
        )
    
    # Move model to device
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Initialize scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=learning_rate * 0.1
    )
    
    # Initialize loss function
    loss_fn = DeepEarthLoss()
    
    # Initialize masking
    masking = SpatiotemporalMasking()
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Train
        train_losses = train_epoch(
            model, train_dataloader, optimizer, 
            loss_fn, masking, device, epoch
        )
        
        # Step scheduler
        scheduler.step()
        
        # Validation
        if val_dataloader is not None:
            val_losses = validate(
                model, val_dataloader, loss_fn, 
                masking, device
            )
            
            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                }, f'{checkpoint_dir}/best_model.pt')
        
        # Regular checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, f'{checkpoint_dir}/checkpoint_epoch_{epoch}.pt')
        
        # Log metrics
        print(f"\nEpoch {epoch} Summary:")
        print(f"Train Loss: {train_losses['total']:.4f}")
        if val_dataloader is not None:
            print(f"Val Loss: {val_losses['total']:.4f}")
        
        if use_wandb:
            log_dict = {f'train/{k}': v for k, v in train_losses.items()}
            if val_dataloader is not None:
                log_dict.update({f'val/{k}': v for k, v in val_losses.items()})
            log_dict['lr'] = scheduler.get_last_lr()[0]
            wandb.log(log_dict)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: DeepEarthLoss,
    masking: SpatiotemporalMasking,
    device: torch.device
) -> Dict[str, float]:
    """Validation loop"""
    model.eval()
    total_losses = {
        'total': 0.0,
        'spatial': 0.0,
        'temporal': 0.0,
        'vision': 0.0,
        'language': 0.0,
        'vision_language_contrastive': 0.0,
        'vision_spatial_contrastive': 0.0,
    }
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            # Similar to training loop but without backward pass
            # ... (implement similar to train_epoch but without optimization)
            pass
    
    # Average losses
    num_batches = len(dataloader)
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    
    return avg_losses


if __name__ == "__main__":
    # Example usage
    from torch.utils.data import TensorDataset
    
    # Create dummy data
    batch_size = 8
    num_samples = 100
    
    # Dummy dataset
    xyzt = torch.rand(num_samples, 4)
    images = torch.randn(num_samples, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (num_samples, 32))
    attention_mask = torch.ones(num_samples, 32)
    
    dataset = TensorDataset(xyzt, images, input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = create_deepearth_model()
    
    # Train
    train_deepearth(
        model=model,
        train_dataloader=dataloader,
        num_epochs=10,
        use_wandb=False  # Set to True if you have wandb configured
    )
