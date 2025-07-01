"""
Distributed training script for DeepEarth on HPC clusters
Supports multi-GPU and multi-node training with PyTorch DDP
"""

import os
import sys
import argparse
import yaml
import time
from datetime import datetime
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

import numpy as np
from tqdm import tqdm
import wandb

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.deepearth_integrated import create_integrated_deepearth
from training.train_deepearth2 import DeepEarthLoss, SpatiotemporalMasking


def setup_distributed():
    """Initialize distributed training environment"""
    # Get distributed training parameters
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    return rank, local_rank, world_size, device


def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()


class DeepEarthDataset(torch.utils.data.Dataset):
    """
    Dataset for DeepEarth training
    Loads data from HPC storage efficiently
    """
    
    def __init__(
        self, 
        data_path: str,
        split: str = 'train',
        max_samples: int = None,
        cache_size: int = 1000
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.cache_size = cache_size
        self.cache = {}
        
        # Load metadata
        metadata_path = self.data_path / f'{split}_metadata.json'
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.samples = self.metadata['samples']
        if max_samples:
            self.samples = self.samples[:max_samples]
            
        # Pre-load frequently accessed data into cache
        if dist.get_rank() == 0:
            print(f"Loading {split} dataset with {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Check cache first
        if idx in self.cache:
            return self.cache[idx]
        
        sample_info = self.samples[idx]
        
        # Load data efficiently with proper format for integrated model
        data = {
            'xyzt': self._load_coordinates(sample_info['coord_file']),
            'sample_id': sample_info['id']
        }
        
        # Load optional modalities
        if 'image_file' in sample_info:
            data['images'] = self._load_image(sample_info['image_file'])
        
        if 'text_file' in sample_info:
            text_data = self._load_text(sample_info['text_file'])
            data['input_ids'] = text_data['input_ids']
            data['attention_mask'] = text_data['attention_mask']
        
        # Load additional modalities
        if 'modalities' in sample_info:
            data['modalities'] = {}
            for mod_name, mod_file in sample_info['modalities'].items():
                data['modalities'][mod_name] = self._load_modality(mod_file)
        
        # Update cache (FIFO)
        if len(self.cache) >= self.cache_size:
            # Remove oldest item
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[idx] = data
        
        return data
    
    def _load_coordinates(self, filename):
        """Load spatiotemporal coordinates"""
        filepath = self.data_path / 'coordinates' / filename
        return torch.from_numpy(np.load(filepath)).float()
    
    def _load_image(self, filename):
        """Load and preprocess image"""
        filepath = self.data_path / 'images' / filename
        # Implement efficient image loading (e.g., using turbojpeg)
        image = torch.from_numpy(np.load(filepath)).float()
        return image
    
    def _load_text(self, filename):
        """Load tokenized text"""
        filepath = self.data_path / 'text' / filename
        data = np.load(filepath)
        return {
            'input_ids': torch.from_numpy(data['input_ids']).long(),
            'attention_mask': torch.from_numpy(data['attention_mask']).bool()
        }
    
    def _load_modality(self, filename):
        """Load additional modality data"""
        filepath = self.data_path / 'modalities' / filename
        return torch.from_numpy(np.load(filepath)).float()


def create_dataloaders(args, world_size, rank):
    """Create distributed data loaders"""
    # Create datasets
    train_dataset = DeepEarthDataset(
        args.data_path,
        split='train',
        max_samples=args.max_train_samples
    )
    
    val_dataset = DeepEarthDataset(
        args.data_path,
        split='val',
        max_samples=args.max_val_samples
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    )
    
    # Create data loaders with optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,  # Larger batch for validation
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    return train_loader, val_loader, train_sampler, val_sampler


def train_epoch_distributed(
    model: DDP,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    loss_fn: DeepEarthLoss,
    masking: SpatiotemporalMasking,
    epoch: int,
    args,
    device: torch.device,
    rank: int
):
    """Train for one epoch with distributed training"""
    model.train()
    
    # Metrics tracking
    metric_tracker = {
        'loss': 0.0,
        'spatial_loss': 0.0,
        'vision_universal_loss': 0.0,
        'contrastive_loss': 0.0,
        'samples_processed': 0
    }
    
    # Progress bar only on rank 0
    if rank == 0:
        pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(train_loader):
        # Move batch to device
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Handle nested modalities dict
        if 'modalities' in batch:
            batch['modalities'] = {k: v.to(device, non_blocking=True) 
                                 for k, v in batch['modalities'].items()}
        
        # Prepare inputs with masking
        xyzt = batch['xyzt']
        targets = {'xyz': xyzt[:, :3], 't': xyzt[:, 3:4]}
        masks = {}
        
        # Apply coordinate masking
        masked_xyz, spatial_mask = masking.mask_spatial_coordinates(targets['xyz'])
        masked_t, temporal_mask = masking.mask_temporal_coordinates(targets['t'])
        masked_xyzt = torch.cat([masked_xyz, masked_t], dim=1)
        masks['spatial'] = spatial_mask
        masks['temporal'] = temporal_mask
        
        # Prepare language input
        language_input = None
        if 'input_ids' in batch:
            language_input = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch.get('attention_mask', torch.ones_like(batch['input_ids']))
            }
        
        # Store native embeddings for reconstruction loss
        if 'images' in batch and hasattr(model.module, 'universal_encoder'):
            with torch.no_grad():
                vision_encoder = model.module.universal_encoder.encoders.get('vision')
                if vision_encoder is not None:
                    native_embeds = vision_encoder.extract_native_embeddings(batch['images'])
                    targets['vision_embeddings'] = native_embeds['global_embedding']
        
        # Mixed precision forward pass
        with autocast(dtype=torch.float16 if args.mixed_precision == 'fp16' else torch.bfloat16):
            outputs = model(
                xyzt=masked_xyzt,
                vision_input=batch.get('images'),
                language_input=language_input,
                additional_modalities=batch.get('modalities'),
                return_intermediates=True
            )
            
            # Compute loss
            loss, loss_components = loss_fn(outputs, targets, masks)
            
            # Scale loss for gradient accumulation
            loss = loss / args.gradient_accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                args.max_grad_norm
            )
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Scheduler step
            if args.scheduler == 'onecycle':
                scheduler.step()
        
        # Update metrics
        metric_tracker['loss'] += loss.item() * args.gradient_accumulation_steps
        metric_tracker['samples_processed'] += xyzt.size(0)
        
        for key in ['spatial', 'vision_universal', 'vision_spatial_contrastive']:
            if key in loss_components:
                if 'contrastive' in key:
                    metric_tracker['contrastive_loss'] += loss_components[key].item()
                else:
                    metric_tracker[f'{key}_loss'] += loss_components[key].item()
        
        # Update progress bar
        if rank == 0:
            pbar.update(1)
            if batch_idx % args.log_interval == 0:
                avg_loss = metric_tracker['loss'] / (batch_idx + 1)
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}',
                    'grad_norm': f'{grad_norm:.2f}' if 'grad_norm' in locals() else 'N/A'
                })
        
        # Synchronize metrics across GPUs periodically
        if batch_idx % args.sync_interval == 0:
            for key in metric_tracker:
                if key != 'samples_processed':
                    tensor = torch.tensor(metric_tracker[key], device=device)
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                    metric_tracker[key] = tensor.item() / dist.get_world_size()
    
    if rank == 0:
        pbar.close()
    
    # Final synchronization
    for key in metric_tracker:
        if key != 'samples_processed':
            tensor = torch.tensor(metric_tracker[key], device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            metric_tracker[key] = tensor.item() / dist.get_world_size()
    
    # Average metrics
    num_batches = len(train_loader)
    avg_metrics = {k: v / num_batches for k, v in metric_tracker.items() 
                   if k != 'samples_processed'}
    
    return avg_metrics


def validate_distributed(
    model: DDP,
    val_loader: DataLoader,
    loss_fn: DeepEarthLoss,
    epoch: int,
    device: torch.device,
    rank: int
):
    """Validation with distributed training"""
    model.eval()
    
    metric_tracker = {
        'loss': 0.0,
        'spatial_loss': 0.0,
        'vision_universal_loss': 0.0,
        'contrastive_loss': 0.0,
    }
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation', disable=(rank != 0)):
            # Move batch to device
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            if 'modalities' in batch:
                batch['modalities'] = {k: v.to(device, non_blocking=True) 
                                     for k, v in batch['modalities'].items()}
            
            # Prepare language input
            language_input = None
            if 'input_ids' in batch:
                language_input = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch.get('attention_mask', torch.ones_like(batch['input_ids']))
                }
            
            # Forward pass
            outputs = model(
                xyzt=batch['xyzt'],
                vision_input=batch.get('images'),
                language_input=language_input,
                additional_modalities=batch.get('modalities'),
                return_intermediates=True
            )
            
            # Prepare targets
            targets = {
                'xyz': batch['xyzt'][:, :3],
                't': batch['xyzt'][:, 3:4]
            }
            
            # Compute loss
            loss, loss_components = loss_fn(outputs, targets, {})
            
            # Update metrics
            metric_tracker['loss'] += loss.item()
            for key in ['spatial', 'vision_universal', 'vision_spatial_contrastive']:
                if key in loss_components:
                    if 'contrastive' in key:
                        metric_tracker['contrastive_loss'] += loss_components[key].item()
                    else:
                        metric_tracker[f'{key}_loss'] += loss_components[key].item()
    
    # Synchronize metrics across GPUs
    for key in metric_tracker:
        tensor = torch.tensor(metric_tracker[key], device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        metric_tracker[key] = tensor.item() / dist.get_world_size()
    
    # Average metrics
    num_batches = len(val_loader)
    avg_metrics = {k: v / num_batches for k, v in metric_tracker.items()}
    
    return avg_metrics


def main(args):
    """Main distributed training function"""
    # Setup distributed training
    rank, local_rank, world_size, device = setup_distributed()
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    
    # Initialize wandb on rank 0
    if rank == 0 and args.use_wandb:
        wandb.init(
            project=os.environ.get('WANDB_PROJECT', 'deepearth'),
            name=os.environ.get('WANDB_RUN_NAME', f'deepearth_{datetime.now():%Y%m%d_%H%M%S}'),
            config=vars(args)
        )
    
    # Create model
    if rank == 0:
        print("Creating DeepEarth model...")
    
    model = create_integrated_deepearth(
        universal_dim=args.hidden_dim,
        num_fusion_layers=args.num_fusion_layers if hasattr(args, 'num_fusion_layers') else 24,
        freeze_backbones=args.freeze_backbones
    )
    
    # Add custom modalities if specified
    if args.additional_modalities:
        for mod_config in args.additional_modalities:
            # Create simple encoder for additional modalities
            from models.encoders import ModalityEncoder as SimpleModalityEncoder
            encoder = SimpleModalityEncoder(
                modality_name=mod_config['name'],
                input_dim=mod_config['input_dim'],
                config=model.config.grid4d_config,
                encoder_config=model.config.grid4d_config.modality_encoder_config
            )
            model.add_modality(
                name=mod_config['name'],
                encoder=encoder,
                native_dim=model.config.universal_dim // 2,
                num_tokens=mod_config.get('num_tokens', 1)
            )
    
    # Move model to device
    model = model.to(device)
    
    # Compile model if requested (PyTorch 2.0+)
    if args.compile_model and hasattr(torch, 'compile'):
        if rank == 0:
            print("Compiling model with torch.compile()...")
        model = torch.compile(model, mode='max-autotune')
    
    # Wrap model with DDP
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=args.find_unused_parameters
    )
    
    # Create data loaders
    if rank == 0:
        print("Creating data loaders...")
    
    train_loader, val_loader, train_sampler, val_sampler = create_dataloaders(
        args, world_size, rank
    )
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate * world_size,  # Scale LR by world size
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    # Create scheduler
    if args.scheduler == 'onecycle':
        total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.learning_rate * world_size,
            total_steps=total_steps,
            pct_start=0.05,
            anneal_strategy='cos'
        )
    else:
        scheduler = None
    
    # Create loss function and masking
    loss_fn = DeepEarthLoss(
        spatial_weight=args.spatial_weight,
        temporal_weight=args.temporal_weight,
        vision_weight=args.vision_weight,
        language_weight=args.language_weight,
        contrastive_weight=args.contrastive_weight,
        temperature=args.temperature
    )
    
    masking = SpatiotemporalMasking(
        spatial_mask_ratio=args.spatial_mask_ratio,
        temporal_mask_ratio=args.temporal_mask_ratio,
        vision_mask_ratio=args.vision_mask_ratio,
        language_mask_ratio=args.language_mask_ratio
    )
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=(args.mixed_precision != 'fp32'))
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume_from_checkpoint:
        checkpoint_path = Path(args.checkpoint_dir) / 'latest_checkpoint.pt'
        if checkpoint_path.exists():
            if rank == 0:
                print(f"Resuming from checkpoint: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
    
    # Training loop
    if rank == 0:
        print(f"\nStarting training on {world_size} GPUs...")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * world_size * args.gradient_accumulation_steps}")
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.num_epochs):
        # Set epoch for distributed sampler
        train_sampler.set_epoch(epoch)
        
        # Train
        train_metrics = train_epoch_distributed(
            model, train_loader, optimizer, scheduler, scaler,
            loss_fn, masking, epoch, args, device, rank
        )
        
        # Validate
        val_metrics = validate_distributed(
            model, val_loader, loss_fn, epoch, device, rank
        )
        
        # Logging and checkpointing on rank 0
        if rank == 0:
            # Print metrics
            print(f"\nEpoch {epoch} Summary:")
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            
            # Log to wandb
            if args.use_wandb:
                log_dict = {
                    'epoch': epoch,
                    'lr': optimizer.param_groups[0]['lr'],
                }
                log_dict.update({f'train/{k}': v for k, v in train_metrics.items()})
                log_dict.update({f'val/{k}': v for k, v in val_metrics.items()})
                wandb.log(log_dict)
            
            # Save checkpoints
            checkpoint_dir = Path(args.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Save latest checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'args': vars(args)
            }
            
            if scheduler:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            torch.save(checkpoint, checkpoint_dir / 'latest_checkpoint.pt')
            
            # Save best checkpoint
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save(checkpoint, checkpoint_dir / 'best_checkpoint.pt')
                print(f"New best model saved with val loss: {best_val_loss:.4f}")
            
            # Save periodic checkpoints
            if epoch % args.save_every == 0:
                torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')
    
    # Cleanup
    cleanup_distributed()
    
    if rank == 0:
        print("\nTraining completed!")
        if args.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distributed DeepEarth Training')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--spatial_resolutions', type=int, nargs='+', 
                        default=[16, 32, 64, 128, 256])
    parser.add_argument('--temporal_resolutions', type=int, nargs='+', 
                        default=[1, 7, 30, 365])
    parser.add_argument('--freeze_backbones', action='store_true')
    parser.add_argument('--compile_model', action='store_true',
                        help='Use torch.compile for optimization')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--max_val_samples', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--mixed_precision', choices=['fp16', 'bf16', 'fp32'], 
                        default='fp16')
    parser.add_argument('--scheduler', choices=['onecycle', 'cosine', 'none'], 
                        default='onecycle')
    
    # Loss weights
    parser.add_argument('--spatial_weight', type=float, default=1.0)
    parser.add_argument('--temporal_weight', type=float, default=1.0)
    parser.add_argument('--vision_weight', type=float, default=1.0)
    parser.add_argument('--language_weight', type=float, default=1.0)
    parser.add_argument('--contrastive_weight', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=0.07)
    
    # Masking ratios
    parser.add_argument('--spatial_mask_ratio', type=float, default=0.15)
    parser.add_argument('--temporal_mask_ratio', type=float, default=0.15)
    parser.add_argument('--vision_mask_ratio', type=float, default=0.75)
    parser.add_argument('--language_mask_ratio', type=float, default=0.15)
    
    # Distributed training
    parser.add_argument('--find_unused_parameters', action='store_true')
    
    # Logging and checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--resume_from_checkpoint', action='store_true')
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--sync_interval', type=int, default=100)
    parser.add_argument('--use_wandb', action='store_true')
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--additional_modalities', type=json.loads, default=None,
                        help='JSON string of additional modalities')
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        # Update args with config values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    main(args)
