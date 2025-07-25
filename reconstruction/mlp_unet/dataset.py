import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
from typing import List, Tuple, Optional, Dict
import numpy as np
from collections import defaultdict

# HuggingFace datasets
from datasets import load_dataset

# Species folder name mapping for the downloaded folders
SPECIES_FOLDER_MAPPING = {
    # Your 10 downloaded folders
    'Callicarpa americana': 'american_beautyberry',
    'Helianthus debilis': 'beach_sunflower',
    'Rudbeckia hirta': 'black_eyed_susan',
    'Gaillardia pulchella': 'blanket_flower',
    'Zamia integrifolia': 'coontie',
    'Coreopsis leavenworthii': 'leavenworth_tickseed',
    'Serenoa repens': 'saw_palmetto',
    'Tradescantia ohiensis': 'spiderwort',
    'Monarda punctata': 'spotted_beebalm',
    'Salvia coccinea': 'tropical_sage',
    
    # Also support folder names directly
    'american_beautyberry': 'american_beautyberry',
    'beach_sunflower': 'beach_sunflower',
    'black_eyed_susan': 'black_eyed_susan',
    'blanket_flower': 'blanket_flower',
    'coontie': 'coontie',
    'leavenworth_tickseed': 'leavenworth_tickseed',
    'saw_palmetto': 'saw_palmetto',
    'spiderwort': 'spiderwort',
    'spotted_beebalm': 'spotted_beebalm',
    'tropical_sage': 'tropical_sage',
}


class CentralFloridaPlantsDataset(Dataset):
    """Dataset for Central Florida Native Plants"""
    
    def __init__(self, 
                 root_dir: str,
                 species_list: List[str],
                 images_per_species: int = 50,
                 test_split: float = 0.2,
                 mode: str = 'train',
                 transform: Optional[transforms.Compose] = None,
                 random_seed: int = 42):
        """
        Args:
            root_dir: Root directory containing species folders with downloaded images
            species_list: List of species names to use
            images_per_species: Maximum images per species to use
            test_split: Fraction of images to use for testing
            mode: 'train' or 'test'
            transform: Image transformations
            random_seed: Random seed for reproducibility
        """
        self.root_dir = root_dir
        self.species_list = species_list
        self.mode = mode
        self.transform = transform or self.get_default_transform()
        
        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Build dataset from local files
        self.samples = []
        self._build_dataset(images_per_species, test_split)
        
    def get_default_transform(self):
        """Default image transformations"""
        if self.mode == 'train':
            return transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def _get_species_folder(self, species: str) -> str:
        """Get the folder name for a species"""
        # Check mapping first
        if species in SPECIES_FOLDER_MAPPING:
            return SPECIES_FOLDER_MAPPING[species]
        
        # Try direct folder name
        folder_path = os.path.join(self.root_dir, species)
        if os.path.exists(folder_path):
            return species
        
        # Try lowercase with underscores
        folder_name = species.lower().replace(' ', '_').replace("'", '')
        folder_path = os.path.join(self.root_dir, folder_name)
        if os.path.exists(folder_path):
            return folder_name
        
        # Default
        return species.replace(' ', '_').lower()
    
    def _build_dataset(self, images_per_species: int, test_split: float):
        """Build the dataset by scanning local directories"""
        for species in self.species_list:
            # Get the correct folder name
            folder_name = self._get_species_folder(species)
            species_dir = os.path.join(self.root_dir, folder_name)
            
            if not os.path.exists(species_dir):
                print(f"Warning: Directory {species_dir} not found for species {species}")
                continue
                
            # Get all image files
            image_files = sorted([f for f in os.listdir(species_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            if not image_files:
                print(f"Warning: No images found in {species_dir}")
                continue
            
            # Limit to images_per_species
            if len(image_files) > images_per_species:
                random.shuffle(image_files)
                image_files = image_files[:images_per_species]
            
            # Split train/test
            split_idx = int(len(image_files) * (1 - test_split))
            
            if self.mode == 'train':
                selected_files = image_files[:split_idx]
            else:  # test
                selected_files = image_files[split_idx:]
                
            # Add to samples
            for img_file in selected_files:
                img_path = os.path.join(species_dir, img_file)
                self.samples.append((img_path, species))
                
        print(f"Loaded {len(self.samples)} samples for {self.mode} set")
        
        # Shuffle training samples
        if self.mode == 'train':
            random.shuffle(self.samples)
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, species = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        return image, species


class CentralFloridaPlantsDataModule:
    """Data module to handle train/test splits and dataloaders"""
    
    def __init__(self,
                 root_dir: str,
                 species_list: List[str],
                 batch_size: int = 32,
                 num_workers: int = 4,
                 images_per_species: int = 50,
                 test_split: float = 0.2):
        
        self.root_dir = root_dir
        self.species_list = species_list
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Create datasets
        self.train_dataset = CentralFloridaPlantsDataset(
            root_dir=root_dir,
            species_list=species_list,
            images_per_species=images_per_species,
            test_split=test_split,
            mode='train'
        )
        
        self.test_dataset = CentralFloridaPlantsDataset(
            root_dir=root_dir,
            species_list=species_list,
            images_per_species=images_per_species,
            test_split=test_split,
            mode='test'
        )
        
    def train_dataloader(self):
        """Stochastic training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Important for stochastic training
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True  # Drop last incomplete batch for stable training
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
    def get_sample_batch(self, n_samples: int = 8, mode: str = 'train'):
        """Get a small batch of samples for visualization"""
        dataset = self.train_dataset if mode == 'train' else self.test_dataset
        indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
        
        images = []
        species = []
        
        for idx in indices:
            img, sp = dataset[idx]
            images.append(img)
            species.append(sp)
            
        return torch.stack(images), species
    
    def get_species_distribution(self):
        """Get distribution of species in the datasets"""
        train_dist = defaultdict(int)
        test_dist = defaultdict(int)
        
        for _, species in self.train_dataset.samples:
            train_dist[species] += 1
            
        for _, species in self.test_dataset.samples:
            test_dist[species] += 1
            
        return dict(train_dist), dict(test_dist)


def explore_hf_dataset():
    """Explore the HuggingFace dataset to understand its structure"""
    print("Exploring Central Florida Native Plants dataset...")
    
    # Load dataset
    dataset = load_dataset("deepearth/central-florida-native-plants", split="train")
    
    # Look at first example
    print("\nFirst example:")
    example = dataset[0]
    for key, value in example.items():
        if key not in ['vision_file_indices', 'language_embedding']:  # Skip large arrays
            print(f"  {key}: {value}")
    
    # Count species
    print("\nCounting species...")
    species_counts = defaultdict(int)
    
    for example in dataset:
        species_counts[example['taxon_name']] += 1
        
    # Sort by count
    sorted_species = sorted(species_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 20 species by image count:")
    for species, count in sorted_species[:20]:
        print(f"  {species}: {count}")
        
    return dataset, species_counts


if __name__ == "__main__":
    # Test with the actual folders
    print("="*60)
    print("Testing Dataset with Your Folder Structure")
    print("="*60)
    
    data_dir = "./data/plants"
    
    # The species names that correspond to your folders
    species_list = [
        'Callicarpa americana',      # → american_beautyberry
        'Helianthus debilis',        # → beach_sunflower
        'Rudbeckia hirta',           # → black_eyed_susan
        'Gaillardia pulchella',      # → blanket_flower
        'Zamia integrifolia',        # → coontie
        'Coreopsis leavenworthii',   # → leavenworth_tickseed
        'Serenoa repens',            # → saw_palmetto
        'Tradescantia ohiensis',     # → spiderwort
        'Monarda punctata',          # → spotted_beebalm
        'Salvia coccinea'            # → tropical_sage
    ]
    
    # Create data module
    data_module = CentralFloridaPlantsDataModule(
        root_dir=data_dir,
        species_list=species_list,
        batch_size=32,
        images_per_species=200,
        test_split=0.2
    )
    
    # Check distribution
    train_dist, test_dist = data_module.get_species_distribution()
    print("\nTraining set distribution:")
    for species, count in sorted(train_dist.items()):
        folder = SPECIES_FOLDER_MAPPING.get(species, 'unknown')
        print(f"  {species:<30} (folder: {folder:<25}) : {count} images")
        
    print(f"\nTotal training images: {sum(train_dist.values())}")
    print(f"Total test images: {sum(test_dist.values())}")
    
    # Test loading a batch
    print("\nTesting dataloader...")
    train_loader = data_module.train_dataloader()
    images, species = next(iter(train_loader))
    print(f"✅ Successfully loaded batch: {images.shape}")
    print(f"Species in batch: {species}")
