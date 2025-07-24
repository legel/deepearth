#!/usr/bin/env python3
"""
Test script to verify data module creation
"""

import os
from dataset import CentralFloridaPlantsDataModule

# Load species list
species_file = "./data/plants/species_list.txt"
if os.path.exists(species_file):
    with open(species_file, 'r') as f:
        species_list = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(species_list)} species:")
    for sp in species_list:
        print(f"  - {sp}")
else:
    print("No species list found, using defaults")
    species_list = ['Helianthus debilis', 'Gaillardia pulchella']

print("\nCreating data module with parameters:")
print(f"  root_dir: './data/plants'")
print(f"  species_list: {len(species_list)} species")
print(f"  batch_size: 32")
print(f"  images_per_species: 50")
print(f"  test_split: 0.2")

try:
    data_module = CentralFloridaPlantsDataModule(
        root_dir='./data/plants',
        species_list=species_list,
        batch_size=32,
        images_per_species=50,
        test_split=0.2
    )
    print("\n✓ Data module created successfully!")
    
    # Check distribution
    train_dist, test_dist = data_module.get_species_distribution()
    print(f"\nTrain samples: {sum(train_dist.values())}")
    print(f"Test samples: {sum(test_dist.values())}")
    
except Exception as e:
    print(f"\n✗ Error creating data module: {e}")
    import traceback
    traceback.print_exc()
