#!/usr/bin/env python3
"""
Minimal test to isolate quickstart issue
"""

import os
import torch
from dataset import CentralFloridaPlantsDataModule
from model import BimodalMLPUNet
from train import BimodalTrainer

def main():
    # Configuration
    config = {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 5,
        'mask_ratio': 0.5,
        'embedding_dim': 2048,
        'hidden_dim': 512,
        'bottleneck_dim': 128,
        'images_per_species': 50,
        'test_split': 0.2
    }
    
    print("Loading species list...")
    species_file = "./data/plants/species_list.txt"
    with open(species_file, 'r') as f:
        species_list = [line.strip() for line in f if line.strip()]
    print(f"Found {len(species_list)} species: {species_list}")
    
    print("\nCreating data module...")
    try:
        data_module = CentralFloridaPlantsDataModule(
            root_dir='./data/plants',
            species_list=species_list,
            batch_size=config['batch_size'],
            images_per_species=config['images_per_species'],
            test_split=config['test_split']
        )
        print("✓ Data module created successfully!")
    except Exception as e:
        print(f"✗ Error creating data module: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nCreating model...")
    try:
        model = BimodalMLPUNet(
            species_list=species_list,
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            bottleneck_dim=config['bottleneck_dim']
        )
        print("✓ Model created successfully!")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nCreating trainer...")
    try:
        trainer = BimodalTrainer(
            model=model,
            data_module=data_module,
            learning_rate=config['learning_rate'],
            mask_ratio=config['mask_ratio']
        )
        print("✓ Trainer created successfully!")
    except Exception as e:
        print(f"✗ Error creating trainer: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nEverything looks good! Ready to train.")
    print("Run 'python quickstart.py' to start full training.")

if __name__ == "__main__":
    main()
