# deepearth/core/main.py
"""
DeepEarth: Multimodal Probabilistic World Model with 4D Spacetime Embedding

Main training script for the DeepEarth planetary intelligence system.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from art import text2art

from deepearth.core.config import DeepEarthConfig
from deepearth.core.preprocessor import DatasetPreprocessor
from deepearth.core.dataloader import DeepEarthDataLoader
from deepearth.core.perceiver import DeepEarthPerceiver
from deepearth.core.trainer import DeepEarthTrainer


def set_reproducibility(seed: int):
    """Configure deterministic behavior for reproducible experiments."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Trade speed for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"[Reproducibility] Random seed set to {seed}")


def print_header():
    """Display DeepEarth ASCII art header."""
    ascii_art = text2art("DEEPEARTH", font='standard')
    
    print("\n" + "="*70)
    print(ascii_art)
    print("Multimodal Probabilistic World Model with 4D Spacetime Embedding")
    print("="*70 + "\n")


def print_model_architecture(model: DeepEarthPerceiver):
    """Print detailed model architecture."""
    print("\n" + "="*70)
    print("Model Architecture")
    print("="*70)
    
    # Component sizes
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Non-trainable: {total_params - trainable_params:,}")
    print(f"  Size (FP32): {total_params * 4 / 1024**2:.1f} MB")
    print(f"  Size (FP16): {total_params * 2 / 1024**2:.1f} MB")
    
    # Layer breakdown
    print(f"\nComponents:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {params:,} parameters")
    
    print("="*70 + "\n")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description='Train DeepEarth world model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core arguments
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Configuration file path')
    parser.add_argument('--input_csv', type=str, required=True,
                       help='Input CSV with Earth observations')
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
    config.input_csv = args.input_csv
    
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
    
    # Log metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'input_csv': args.input_csv,
        'config_path': args.config,
        'device': config.device,
        'seed': config.seed,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
    }
    
    print("\nExperiment Metadata:")
    print(json.dumps(metadata, indent=2))
    
    # ═══════════════════════════════════════════════════════════
    # Data preprocessing
    # ═══════════════════════════════════════════════════════════
    
    print("\n" + "="*70)
    print("Data Preprocessing")
    print("="*70)
    
    preprocessor = DatasetPreprocessor(config)
    data = preprocessor.process_csv(config.input_csv)
    
    # ═══════════════════════════════════════════════════════════
    # Extract encoder configurations
    # ═══════════════════════════════════════════════════════════
    
    encoder_configs = {}
    for encoder_id, encoder_name in data['encoder_map'].items():
        if encoder_id in data['encoded_data']:
            input_dim = data['encoded_data'][encoder_id].shape[-1]
            encoder_configs[encoder_id] = {
                'name': encoder_name,
                'input_dim': input_dim
            }
    
    # Save complete metadata
    metadata['encoder_configs'] = encoder_configs
    metadata['dataset_map'] = data['dataset_map']
    metadata['modality_map'] = data['modality_map']
    metadata['encoder_map'] = data['encoder_map']
    metadata['n_samples'] = data['n_samples']
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # ═══════════════════════════════════════════════════════════
    # Create data loaders
    # ═══════════════════════════════════════════════════════════
    
    print("\n" + "="*70)
    print("Data Loading")
    print("="*70)
    
    dataloader_factory = DeepEarthDataLoader(config, data)
    train_loader, val_loader, test_loader = dataloader_factory.get_data_loaders(
        use_context_sampling=args.context_sampling
    )
    
    # ═══════════════════════════════════════════════════════════
    # Initialize model
    # ═══════════════════════════════════════════════════════════
    
    print("\n" + "="*70)
    print("Model Initialization")
    print("="*70)
    
    model = DeepEarthPerceiver(config, encoder_configs)
    
    if args.print_architecture:
        print_model_architecture(model)
    
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
    
    # ═══════════════════════════════════════════════════════════
    # Training complete
    # ═══════════════════════════════════════════════════════════
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print(f"Best checkpoint: {output_dir / 'checkpoints/best.pt'}")
    print(f"Configuration: {output_dir / 'config.yaml'}")
    print(f"Metadata: {output_dir / 'metadata.json'}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
