if __name__ == "__main__":
    # Example usage
    from torch.utils.data import TensorDataset
    from models.deepearth_integrated import create_integrated_deepearth
    
    # Create model with new architecture
    model = create_integrated_deepearth(
        universal_dim=2048,
        num_fusion_layers=12,
        freeze_backbones=True
    )
    
    # Create dummy data
    batch_size = 8
    num_samples = 100
    
    # Dummy dataset with all modalities
    xyzt = torch.rand(num_samples, 4)
    images = torch.randn(num_samples, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (num_samples, 32))
    attention_mask = torch.ones(num_samples, 32)
    
    # Create dataset that returns proper format
    class DeepEarthDataset(torch.utils.data.Dataset):
        def __init__(self, xyzt, images, input_ids, attention_mask):
            self.xyzt = xyzt
            self.images = images
            self.input_ids = input_ids
            self.attention_mask = attention_mask
        
        def __len__(self):
            return len(self.xyzt)
        
        def __getitem__(self, idx):
            return {
                'xyzt': self.xyzt[idx],
                'images': self.images[idx],
                'input_ids': self.input_ids[idx],
                'attention_mask': self.attention_mask[idx],
                'modalities': {
                    'weather': torch.randn(5),
                    'species': torch.randn(64)
                }
            }
    
    dataset = DeepEarthDataset(xyzt, images, input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Train
    train_deepearth(
        model=model,
        train_dataloader=dataloader,
        num_epochs=10,
        use_wandb=False  # Set to True if you have wandb configured
    )"""
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

from models.deepearth_integrated import create_integrated_deepearth


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
        universal_reconstruction_weight: float = 1.0,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight
        self.vision_weight = vision_weight
        self.language_weight = language_weight
        self.contrastive_weight = contrastive_weight
        self.universal_reconstruction_weight = universal_reconstruction_weight
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
        if 'spatial' in outputs.get('reconstructions', {}) and 'xyz' in targets:
            loss_xyz = self.spatial_reconstruction_loss(
                outputs['reconstructions']['spatial'],
                targets['xyz']
            )
            losses['spatial'] = loss_xyz
            total_loss += self.spatial_weight * loss_xyz
        
        # Temporal reconstruction
        if 'temporal' in outputs.get('reconstructions', {}) and 't' in targets:
            loss_t = self.temporal_reconstruction_loss(
                outputs['reconstructions']['temporal'],
                targets['t']
            )
            losses['temporal'] = loss_t
            total_loss += self.temporal_weight * loss_t
        
        # Universal token reconstruction (new)
        if 'vision' in outputs.get('reconstructions', {}) and 'vision_embeddings' in targets:
            loss_vision_universal = F.mse_loss(
                outputs['reconstructions']['vision'],
                targets['vision_embeddings']
            )
            losses['vision_universal'] = loss_vision_universal
            total_loss += self.universal_reconstruction_weight * loss_vision_universal
        
        # Cross-modal contrastive losses
        if 'modality_tokens' in outputs:
            tokens = outputs['modality_tokens']
            
            # Vision-Language contrastive
            if 'vision' in tokens and 'language' in tokens:
                # Pool tokens if needed
                vision_pooled = tokens['vision'].mean(dim=1)  # (B, D)
                language_pooled = tokens['language'].mean(dim=1)  # (B, D)
                
                loss_vl = self.contrastive_loss(vision_pooled, language_pooled)
                losses['vision_language_contrastive'] = loss_vl
                total_loss += self.contrastive_weight * loss_vl
            
            # Vision-Spatial contrastive
            if 'vision' in tokens and 'spatial' in tokens:
                vision_pooled = tokens['vision'].mean(dim=1)
                spatial_pooled = tokens['spatial'].squeeze(1) if tokens['spatial'].dim() == 3 else tokens['spatial']
                
                loss_vs = self.contrastive_loss(vision_pooled, spatial_pooled)
                losses['vision_spatial_contrastive'] = loss_vs
                total_loss += self.contrastive_weight * loss_vs
        
        # Task-specific losses
        if 'task_output' in outputs and 'task_target' in targets:
            task_name = targets.get('task_name', 'unknown')
            if 'classification' in task_name:
                loss_task = F.cross_entropy(outputs['task_output'], targets['task_target'])
            else:
                loss_task = F.mse_loss(outputs['task_output'], targets['task_target'])
            
            losses[f'{task_name}_loss'] = loss_task
            total_loss += loss_task
        
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
        'vision_universal': 0.0,
        'vision_language_contrastive': 0.0,
        'vision_spatial_contrastive': 0.0,
    }
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        xyzt = batch['xyzt'].to(device)
        images = batch.get('images', None)
        language_input = None
        modalities = batch.get('modalities', None)
        
        # Prepare language input
        if 'input_ids' in batch:
            language_input = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch.get('attention_mask', torch.ones_like(batch['input_ids'])).to(device)
            }
        
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
        
        # Store native embeddings for reconstruction
        if images is not None:
            images = images.to(device)
            # Extract native vision embeddings
            with torch.no_grad():
                vision_encoder = model.universal_encoder.encoders.get('vision')
                if vision_encoder is not None:
                    native_embeds = vision_encoder.extract_native_embeddings(images)
                    targets['vision_embeddings'] = native_embeds['global_embedding']
        
        # Move additional modalities to device
        if modalities is not None:
            modalities = {k: v.to(device) for k, v in modalities.items()}
        
        # Forward pass
        outputs = model(
            xyzt=masked_xyzt,
            vision_input=images,
            language_input=language_input,
            additional_modalities=modalities,
            return_intermediates=True
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
            'contrast': loss_components.get('vision_spatial_contrastive', 0).item(),
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
                "model_type": "integrated_deepearth"
            }
        )
    
    # Move model to device
    model = model.to(device)
    
    # Initialize optimizer with different learning rates for different components
    backbone_params = []
    projection_params = []
    fusion_params = []
    
    for name, param in model.named_parameters():
        if 'universal_encoder.encoders' in name:
            backbone_params.append(param)
        elif 'projector' in name or 'projection' in name:
            projection_params.append(param)
        else:
            fusion_params.append(param)
    
    optimizer = AdamW([
        {'params': backbone_params, 'lr': learning_rate * 0.1},  # Lower LR for pretrained
        {'params': projection_params, 'lr': learning_rate},
        {'params': fusion_params, 'lr': learning_rate}
    ], weight_decay=weight_decay, betas=(0.9, 0.95))
    
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
        'vision_universal': 0.0,
        'vision_language_contrastive': 0.0,
        'vision_spatial_contrastive': 0.0,
    }
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            # Move batch to device
            xyzt = batch['xyzt'].to(device)
            images = batch.get('images', None)
            language_input = None
            modalities = batch.get('modalities', None)
            
            # Prepare language input
            if 'input_ids' in batch:
                language_input = {
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch.get('attention_mask', torch.ones_like(batch['input_ids'])).to(device)
                }
            
            if images is not None:
                images = images.to(device)
            
            if modalities is not None:
                modalities = {k: v.to(device) for k, v in modalities.items()}
            
            # Forward pass (no masking for validation)
            outputs = model(
                xyzt=xyzt,
                vision_input=images,
                language_input=language_input,
                additional_modalities=modalities,
                return_intermediates=True
            )
            
            # Prepare targets
            targets = {
                'xyz': xyzt[:, :3],
                't': xyzt[:, 3:4]
            }
            
            # Compute loss
            loss, loss_components = loss_fn(outputs, targets, {})
            
            # Update totals
            for key, value in loss_components.items():
                if key in total_losses:
                    total_losses[key] += value.item()
    
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
