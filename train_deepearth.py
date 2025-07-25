"""
DeepEarth Training Script for Lambda Labs
Optimized for A100 40GB
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import GPUtil
import psutil
import wandb
from pathlib import Path

# Import DeepEarth components
from core.inductive_simulator import create_inductive_simulator
from deepearth_integration import DeepEarthSystem, create_deepearth_system
from deepearth_seamless_api import DeepEarthDataIntegrator

# Initialize Weights & Biases for monitoring
wandb.init(project="deepearth", name=f"lambda-training-{time.strftime('%Y%m%d-%H%M%S')}")

class LambdaLabsConfig:
    """Optimized configuration for Lambda Labs A100"""
    
    # Model settings
    universal_dim = 2048
    vision_model = "vjepa2-base"  # Start with base, can upgrade to large
    language_model = "deepseek-7b"
    language_precision = "int8"  # Save memory
    
    # Simulator settings - full power on A100
    simulator_preset = "standard"  # 24 layers, 32 experts
    use_moe = True
    num_experts = 32
    
    # Training settings
    batch_size = 8  # Good for A100 40GB
    gradient_accumulation_steps = 4  # Effective batch size of 32
    learning_rate = 1e-4
    warmup_steps = 1000
    max_steps = 50000
    
    # Memory optimization
    gradient_checkpointing = True
    mixed_precision = "fp16"  # Use automatic mixed precision
    
    # Data settings
    num_workers = 4
    prefetch_factor = 2
    
    # Monitoring
    log_every = 10
    save_every = 1000
    eval_every = 500


def setup_training():
    """Setup DeepEarth for training on Lambda Labs"""
    
    config = LambdaLabsConfig()
    
    # 1. Initialize DeepEarth system
    print("Initializing DeepEarth system...")
    deepearth = create_deepearth_system(
        universal_dim=config.universal_dim,
        vision_model=config.vision_model,
        language_model_size="7b",
        enable_all_tasks=True,
        device="cuda"
    )
    
    # 2. Create data integrator for seamless data loading
    print("Setting up data integrator...")
    integrator = DeepEarthDataIntegrator(
        universal_dim=config.universal_dim,
        device="cuda"
    )
    
    # 3. Add datasets (example with Florida plants)
    integrator.add_dataset(
        "florida_plants_rgb",
        data_sample=torch.randn(3, 224, 224),
        modality_type="vision",
        num_tokens=16  # 4x4 grid
    ).add_dataset(
        "climate_data",
        data_sample=torch.randn(12, 5),  # 12 months, 5 variables
        modality_type="timeseries",
        num_tokens=4
    ).add_dataset(
        "soil_chemistry",
        data_sample=torch.randn(10),
        modality_type="map_layer",
        num_tokens=1
    )
    
    # 4. Setup mixed precision training
    scaler = torch.cuda.amp.GradScaler() if config.mixed_precision == "fp16" else None
    
    # 5. Create optimizer with different learning rates
    optimizer = torch.optim.AdamW([
        {'params': deepearth.fusion.parameters(), 'lr': config.learning_rate},
        {'params': deepearth.task_heads.parameters(), 'lr': config.learning_rate * 2},
        {'params': integrator.decoders.parameters(), 'lr': config.learning_rate},
    ], weight_decay=0.01)
    
    # 6. Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=config.max_steps,
        pct_start=config.warmup_steps / config.max_steps
    )
    
    return deepearth, integrator, optimizer, scheduler, scaler, config


def monitor_resources():
    """Monitor GPU and system resources"""
    # GPU monitoring
    gpus = GPUtil.getGPUs()
    gpu = gpus[0]
    
    gpu_memory_used = gpu.memoryUsed
    gpu_memory_total = gpu.memoryTotal
    gpu_utilization = gpu.load * 100
    gpu_temp = gpu.temperature
    
    # CPU monitoring
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    return {
        'gpu_memory_used_gb': gpu_memory_used / 1024,
        'gpu_memory_total_gb': gpu_memory_total / 1024,
        'gpu_utilization': gpu_utilization,
        'gpu_temperature': gpu_temp,
        'cpu_percent': cpu_percent,
        'ram_used_gb': memory.used / (1024**3),
        'ram_total_gb': memory.total / (1024**3)
    }


def train_step(deepearth, integrator, batch, optimizer, scheduler, scaler, config):
    """Single training step with mixed precision"""
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Process with automatic mixed precision
    with torch.cuda.amp.autocast(enabled=config.mixed_precision == "fp16"):
        # Forward pass through integrator
        tokens = integrator.process(batch)
        
        # Forward through DeepEarth
        outputs = deepearth(
            xyzt=batch['xyzt'],
            inputs=tokens,
            task='temperature_prediction'  # Example task
        )
        
        # Compute loss
        loss, loss_dict = deepearth.compute_loss(outputs, batch.get('targets', {}))
        
    # Backward pass with gradient scaling
    if scaler:
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(deepearth.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(deepearth.parameters(), max_norm=1.0)
        optimizer.step()
    
    scheduler.step()
    
    return loss.item(), loss_dict


def main_training_loop():
    """Main training loop optimized for Lambda Labs"""
    
    # Setup
    deepearth, integrator, optimizer, scheduler, scaler, config = setup_training()
    
    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    print(f"Starting training on {torch.cuda.get_device_name(0)}")
    print(f"Model parameters: {sum(p.numel() for p in deepearth.parameters()):,}")
    
    # Training loop
    step = 0
    accumulated_loss = 0.0
    
    while step < config.max_steps:
        # Generate batch (replace with your actual data loader)
        batch = {
            'xyzt': torch.randn(config.batch_size, 4).cuda(),
            'florida_plants_rgb': torch.randn(config.batch_size, 3, 224, 224).cuda(),
            'climate_data': torch.randn(config.batch_size, 12, 5).cuda(),
            'soil_chemistry': torch.randn(config.batch_size, 10).cuda(),
        }
        
        # Training step
        loss, loss_dict = train_step(
            deepearth, integrator, batch, optimizer, scheduler, scaler, config
        )
        
        accumulated_loss += loss
        
        # Logging
        if step % config.log_every == 0:
            resources = monitor_resources()
            
            log_dict = {
                'step': step,
                'loss': loss,
                'learning_rate': scheduler.get_last_lr()[0],
                **loss_dict,
                **resources
            }
            
            wandb.log(log_dict)
            
            print(f"Step {step}: Loss = {loss:.4f}, "
                  f"GPU Mem = {resources['gpu_memory_used_gb']:.1f}/{resources['gpu_memory_total_gb']:.1f} GB, "
                  f"GPU Util = {resources['gpu_utilization']:.1f}%")
        
        # Checkpointing
        if step % config.save_every == 0 and step > 0:
            checkpoint = {
                'step': step,
                'model_state_dict': deepearth.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'config': config,
            }
            
            checkpoint_path = checkpoint_dir / f"deepearth_step_{step}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            
            # Keep only last 3 checkpoints to save space
            checkpoints = sorted(checkpoint_dir.glob("deepearth_step_*.pt"))
            for old_checkpoint in checkpoints[:-3]:
                old_checkpoint.unlink()
        
        # Evaluation
        if step % config.eval_every == 0 and step > 0:
            # Run evaluation (implement your eval logic)
            print(f"Running evaluation at step {step}...")
            # eval_metrics = evaluate(deepearth, eval_data)
            # wandb.log({'eval': eval_metrics})
        
        step += 1
    
    print("Training completed!")
    
    # Final save
    final_checkpoint = {
        'model_state_dict': deepearth.state_dict(),
        'config': config,
    }
    torch.save(final_checkpoint, checkpoint_dir / "deepearth_final.pt")


if __name__ == "__main__":
    # Set CUDA settings for optimal performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Run training
    main_training_loop()
