#!/usr/bin/env python3
"""
Test script to verify data loading and display dataset statistics
"""

import os
import glob
from collections import defaultdict
import json
from PIL import Image

def analyze_plant_dataset(data_root="/home/ubuntu/a/deepearth/reconstruction/mlp_unet/data/plants"):
    """Analyze the plant dataset structure and contents"""
    
    print(f"Analyzing dataset at: {data_root}")
    print("=" * 60)
    
    # Get all species directories
    species_dirs = [d for d in os.listdir(data_root) 
                   if os.path.isdir(os.path.join(data_root, d)) 
                   and not d.startswith('.')]
    
    # Statistics
    stats = defaultdict(dict)
    total_images = 0
    image_formats = defaultdict(int)
    
    # Species name mapping (common name -> scientific name)
    species_mapping = {
        "american_beautyberry": "Callicarpa americana",
        "beach_sunflower": "Helianthus debilis",
        "black_eyed_susan": "Rudbeckia hirta",
        "blanket_flower": "Gaillardia pulchella",
        "coontie": "Zamia integrifolia",
        "gaillardia_pulchella": "Gaillardia pulchella",
        "helianthus_debilis": "Helianthus debilis",
        "leavenworth_tickseed": "Coreopsis leavenworthii",
        "saw_palmetto": "Serenoa repens",
        "spiderwort": "Tradescantia ohiensis",
        "spotted_beebalm": "Monarda punctata",
        "tropical_sage": "Salvia coccinea"
    }
    
    print(f"Found {len(species_dirs)} species directories:\n")
    
    for species_dir in sorted(species_dirs):
        species_path = os.path.join(data_root, species_dir)
        
        # Get all images
        image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        for pattern in image_patterns:
            image_files.extend(glob.glob(os.path.join(species_path, pattern)))
        
        # Get scientific name
        scientific_name = species_mapping.get(species_dir, species_dir)
        
        # Store stats
        stats[species_dir] = {
            'scientific_name': scientific_name,
            'num_images': len(image_files),
            'image_files': sorted([os.path.basename(f) for f in image_files])[:5]  # First 5 files
        }
        
        total_images += len(image_files)
        
        # Check image formats
        for img_file in image_files:
            ext = os.path.splitext(img_file)[1].lower()
            image_formats[ext] += 1
        
        # Print summary
        print(f"  {species_dir:25} ({scientific_name:30}) - {len(image_files):3} images")
        
        # Check first image dimensions
        if image_files:
            try:
                with Image.open(image_files[0]) as img:
                    stats[species_dir]['sample_size'] = img.size
            except Exception as e:
                print(f"    Warning: Could not open {image_files[0]}: {e}")
    
    print("\n" + "=" * 60)
    print(f"Total images: {total_images}")
    print(f"\nImage formats found:")
    for fmt, count in image_formats.items():
        print(f"  {fmt}: {count} images")
    
    # Save species mapping
    mapping_file = os.path.join(data_root, "species_mapping.json")
    with open(mapping_file, 'w') as f:
        json.dump(species_mapping, f, indent=2)
    print(f"\nSaved species mapping to: {mapping_file}")
    
    # Test loading with the dataset class
    print("\n" + "=" * 60)
    print("Testing dataset loading...")
    
    try:
        from multimodal_plant_training import LocalPlantDataset, MultimodalConfig
        
        config = MultimodalConfig()
        config.data_root = data_root
        
        # Test train dataset
        train_dataset = LocalPlantDataset(config, split='train')
        print(f"\nTrain dataset created successfully!")
        print(f"  Number of samples: {len(train_dataset)}")
        
        # Test val dataset
        val_dataset = LocalPlantDataset(config, split='val')
        print(f"\nValidation dataset created successfully!")
        print(f"  Number of samples: {len(val_dataset)}")
        
        # Test loading a sample
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"\nSample data keys: {list(sample.keys())}")
            print(f"  Image shape: {sample['pixel_values'].shape}")
            print(f"  Input IDs shape: {sample['input_ids'].shape}")
            print(f"  Species: {sample['species']}")
            
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        print("Make sure multimodal_plant_training.py is in the same directory")
    
    return stats

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        print("\n" + "=" * 60)
        print("GPU Check:")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU count: {torch.cuda.device_count()}")
            print(f"  GPU name: {torch.cuda.get_device_name(0)}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    except ImportError:
        print("\nPyTorch not installed yet")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test plant dataset loading")
    parser.add_argument('--data-root', type=str, 
                      default="/home/ubuntu/a/deepearth/reconstruction/mlp_unet/data/plants",
                      help='Path to plant data directory')
    
    args = parser.parse_args()
    
    # Analyze dataset
    stats = analyze_plant_dataset(args.data_root)
    
    # Check GPU
    check_gpu()
    
    print("\n" + "=" * 60)
    print("Dataset analysis complete!")
