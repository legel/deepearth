# deepearth/models/flowering/flowering_model.py
"""
DeepEarth Training Script for Angiosperm Dataset
════════════════════════════════════════════════

Specialized training script for the angiosperm flowering dataset.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime
# from art import text2art  # Optional ASCII art

import sys
sys.path.append('/opt/ecodash/deepearth')

from core.config import DeepEarthConfig
from core.perceiver import DeepEarthPerceiver
from core.trainer import DeepEarthTrainer
from models.flowering.preprocess_flowering_data import FloweringDatasetPreprocessor

def set_reproducibility(seed: int):
    """Configure deterministic behavior for reproducible experiments."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"[Reproducibility] Random seed set to {seed}")


def print_header():
    """Display DeepEarth header."""
    print("\n" + "="*70)
    print(" " * 25 + "DEEPEARTH 🌻")
    print(" " * 10 + "Multimodal World Model for Flowering Prediction")
    print("="*70 + "\n")


def main():
    """Main training pipeline for angiosperm dataset."""
    parser = argparse.ArgumentParser(
        description='Train DeepEarth on Angiosperm Flowering Dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core arguments
    parser.add_argument('--config', type=str, default='/opt/ecodash/deepearth/models/flowering/flowering.yaml',
                       help='Configuration file path')
    parser.add_argument('--data_dir', type=str,
                       default='/opt/ecodash/deepearth/models/flowering/data',
                       help='Directory containing angiosperm dataset files')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Resume from checkpoint')
    
    # Optional overrides
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Override output directory')
    parser.add_argument('--compile', action='store_true',
                       help='Enable torch.compile()')
    parser.add_argument('--context_sampling', action='store_true',
                       help='Use context-based sampling')
    parser.add_argument('--print_architecture', action='store_true',
                       help='Print detailed model architecture')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size')
    
    args = parser.parse_args()
    
    # Display header
    print_header()
    
    # ═══════════════════════════════════════════════════════════
    # Load configuration
    # ═══════════════════════════════════════════════════════════
    
    if Path(args.config).exists():
        print(f"Loading configuration: {args.config}")
        config = DeepEarthConfig.from_yaml(args.config)
    else:
        print("Using default configuration")
        config = DeepEarthConfig()
    
    # Apply overrides
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.compile:
        config.compile_model = True
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    
    # Set reproducibility
    set_reproducibility(config.seed)
    
    # ═══════════════════════════════════════════════════════════
    # Create output directory
    # ═══════════════════════════════════════════════════════════
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config.to_yaml(output_dir / 'config.yaml')
    
    # ═══════════════════════════════════════════════════════════
    # Load angiosperm dataset
    # ═══════════════════════════════════════════════════════════
    
    print("\n" + "="*70)
    print("Loading Angiosperm Dataset")
    print("="*70)
    
    preprocessor = FloweringDatasetPreprocessor(config, args.data_dir)
    data = preprocessor.load_dataset()
    
    # ═══════════════════════════════════════════════════════════
    # Extract encoder configurations
    # ═══════════════════════════════════════════════════════════
    
    # Create encoder configs from the dataset
    encoder_configs = {}
    for encoder_id, tensor in data['encoded_data'].items():
        encoder_name = list(data['encoder_map'].keys())[
            list(data['encoder_map'].values()).index(encoder_id)
        ]
        encoder_configs[encoder_id] = {
            'name': encoder_name,
            'input_dim': tensor.shape[-1]
        }
    
    print(f"\nEncoder configurations:")
    for enc_id, enc_cfg in encoder_configs.items():
        print(f"  [{enc_id}] {enc_cfg['name']}: {enc_cfg['input_dim']}D")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'data_dir': args.data_dir,
        'config_path': args.config,
        'device': config.device,
        'seed': config.seed,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'encoder_configs': encoder_configs,
        'dataset_map': data['dataset_map'],
        'modality_map': data['modality_map'],
        'encoder_map': data['encoder_map'],
        'n_samples': data['n_samples'],
        'n_observations': data['n_observations']
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # ═══════════════════════════════════════════════════════════
    # Create data loaders
    # ═══════════════════════════════════════════════════════════
    
    print("\n" + "="*70)
    print("Creating Data Loaders")
    print("="*70)
    
    # Create data loaders using sampling engine
    from core.sampling import DeepEarthDataSamplingEngine
    from torch.utils.data import DataLoader, TensorDataset, random_split

    # Create sampling engine
    sampling_engine = DeepEarthDataSamplingEngine(config, data)

    # Create dataset from tensors
    n_observations = data['n_observations']
    dataset = TensorDataset(
        data['xyzt'],
        data['dataset_modality_encoder'],
        data['encoded_file_indices'],
        data['encoded_row_indices'],
        data['target']
    )

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.seed)
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    # ═══════════════════════════════════════════════════════════
    # Initialize model
    # ═══════════════════════════════════════════════════════════
    
    print("\n" + "="*70)
    print("Model Initialization")
    print("="*70)
    
    model = DeepEarthPerceiver(config, encoder_configs)
    
    if args.print_architecture:
        print("\nModel Architecture:")
        print(model)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nParameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Size (FP32): {total_params * 4 / 1024**2:.1f} MB")
    
    # ═══════════════════════════════════════════════════════════
    # Initialize trainer
    # ═══════════════════════════════════════════════════════════
    
    trainer = DeepEarthTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )
    
    # Resume from checkpoint if provided
    if args.checkpoint:
        print(f"\nResuming from checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
    
    # ═══════════════════════════════════════════════════════════
    # Start training
    # ═══════════════════════════════════════════════════════════
    
    trainer.train()
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print(f"Best checkpoint: {output_dir / 'checkpoints/best.pt'}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
