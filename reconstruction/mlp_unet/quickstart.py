#!/usr/bin/env python3
"""
Quick start script for training the MLP U-Net on Central Florida Plants
Handles everything from dataset download to training
"""

import os
import torch
from datetime import datetime
import argparse

from model import BimodalMLPUNet
from dataset import CentralFloridaPlantsDataModule
from train import BimodalTrainer


def main():
    parser = argparse.ArgumentParser(description='Train MLP U-Net on Central Florida Plants')
    parser.add_argument('--explore', action='store_true', help='Explore dataset first')
    parser.add_argument('--species', type=int, default=10, help='Number of species to use')
    parser.add_argument('--images-per-species', type=int, default=50, help='Images per species')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--mask-ratio', type=float, default=0.5, help='Masking ratio')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    args = parser.parse_args()
    
    print("=" * 60)
    print("MLP U-Net Training on Central Florida Native Plants")
    print("=" * 60)
    
    # Explore dataset if requested
    if args.explore:
        from dataset import explore_hf_dataset
        explore_hf_dataset()
        return
    
    # Configuration
    config = {
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'num_epochs': args.epochs,
        'mask_ratio': args.mask_ratio,
        'embedding_dim': 2048,
        'hidden_dim': 512,
        'bottleneck_dim': 128,
        'images_per_species': args.images_per_species,
        'test_split': 0.2
    }
    
    # Check if images are downloaded
    data_dir = "./data/plants"
    species_file = os.path.join(data_dir, 'species_list.txt')
    
    if not os.path.exists(data_dir) or not os.path.exists(species_file):
        print("⚠️  Images not found! Downloading from HuggingFace dataset...")
        print("This will automatically select species with sufficient images...")
        
        # Import and run download script
        from download_images import download_dataset_images_smart
        
        # Try preferred species first
        preferred_species = [
            'Helianthus debilis',          # Beach Sunflower
            'Gaillardia pulchella',        # Blanket Flower  
            'Iris virginica',              # Blue Flag Iris
            'Lonicera sempervirens',       # Coral Honeysuckle
            'Hamelia patens',              # Firebush
            'Echinacea purpurea',          # Purple Coneflower
            'Serenoa repens',              # Saw Palmetto
            'Hymenocallis latifolia',      # Spider Lily
            'Psychotria nervosa',          # Wild Coffee
            'Canna flaccida'               # Yellow Canna
        ]
        
        final_species = download_dataset_images_smart(
            preferred_species=preferred_species,
            output_dir=data_dir,
            images_per_species=config['images_per_species'],
            max_workers=10,
            target_species=args.species
        )
        selected_species = final_species
    else:
        # Load species list from file
        with open(species_file, 'r') as f:
            selected_species = [line.strip() for line in f if line.strip()]
        print(f"Using {len(selected_species)} species from previous download")
    
    print(f"\nTraining Configuration:")
    print(f"  Species: {len(selected_species)}")
    print(f"  Images per species: {config['images_per_species']}")
    print(f"  Total images: ~{len(selected_species) * config['images_per_species']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Mask ratio: {config['mask_ratio']}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    print(f"\nSelected species:")
    for i, species in enumerate(selected_species, 1):
        print(f"  {i}. {species}")
    
    # Create data module
    print("\n📊 Setting up data module...")
    data_module = CentralFloridaPlantsDataModule(
        root_dir=data_dir,
        species_list=selected_species,
        batch_size=config['batch_size'],
        images_per_species=config['images_per_species'],
        test_split=config['test_split']
    )
    
    # Show data distribution
    train_dist, test_dist = data_module.get_species_distribution()
    print(f"\n📈 Dataset statistics:")
    print(f"  Training samples: {sum(train_dist.values())}")
    print(f"  Test samples: {sum(test_dist.values())}")
    
    # Create model
    print("\n🏗️  Building model...")
    model = BimodalMLPUNet(
        species_list=selected_species,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        bottleneck_dim=config['bottleneck_dim'],
        pretrained_encoder=True
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    print("\n🎯 Initializing trainer...")
    trainer = BimodalTrainer(
        model=model,
        data_module=data_module,
        learning_rate=config['learning_rate'],
        mask_ratio=config['mask_ratio']
    )
    
    # Resume if checkpoint provided
    if args.resume:
        print(f"\n📂 Resuming from checkpoint: {args.resume}")
        start_epoch, accuracy = trainer.load_checkpoint(args.resume)
        print(f"  Starting from epoch {start_epoch} with accuracy {accuracy:.4f}")
    
    # Start training
    print("\n🚀 Starting training...")
    print("  Progress will be logged to TensorBoard")
    print("  Run 'tensorboard --logdir=./logs' to monitor training")
    
    try:
        trainer.train(num_epochs=config['num_epochs'], eval_every=5)
        
        print("\n✅ Training completed successfully!")
        print(f"  Best model saved to: ./checkpoints/best.pth")
        print(f"  Final model saved to: ./checkpoints/latest.pth")
        
        # Final evaluation
        print("\n📊 Running final evaluation...")
        accuracy_top1, accuracy_top3, avg_distance = trainer.evaluate(config['num_epochs'])
        
        print("\n🎉 Final Results:")
        print(f"  Top-1 Accuracy: {accuracy_top1:.2%}")
        print(f"  Top-3 Accuracy: {accuracy_top3:.2%}")
        print(f"  Average Embedding Distance: {avg_distance:.4f}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("  Progress has been saved to checkpoints")
        print("  Use --resume flag to continue training")
    
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
