#!/usr/bin/env python3
"""
DeepEarth Species Classification Training Script

Skeleton training script demonstrating end-to-end ML pipeline with DeepEarth data.
Supports both language embedding → species classification and vision embedding → species classification.

This script serves as a foundation for more advanced ML research and can be easily
extended with sophisticated architectures and training techniques.

Usage:
    # Language embedding classification
    python train_classifier.py --mode language --epochs 10
    
    # Vision embedding classification  
    python train_classifier.py --mode vision --epochs 10
    
    # Both modalities
    python train_classifier.py --mode both --epochs 10
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import argparse
import logging
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt

# Add dashboard to path for data access
dashboard_path = Path(__file__).parent.parent / "dashboard"
sys.path.insert(0, str(dashboard_path))

from services.training_data import get_training_batch, get_available_observation_ids
from data_cache import UnifiedDataCache

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DeepEarthDataset(Dataset):
    """
    🌍 DeepEarth PyTorch Dataset
    
    Efficient dataset class for loading batched multimodal biodiversity data.
    Supports both language and vision embeddings with species classification targets.
    """
    
    def __init__(self, observation_ids: List[str], cache, mode: str = 'both', device: str = 'cpu', species_mapping: Optional[Dict] = None):
        """
        Initialize dataset with observation IDs and data cache.
        
        Args:
            observation_ids: List of OBSERVATION_IDs to include
            cache: UnifiedDataCache instance for data access
            mode: 'language', 'vision', or 'both'
            device: PyTorch device for tensor placement
            species_mapping: Optional pre-defined species to index mapping
        """
        self.observation_ids = observation_ids
        self.cache = cache
        self.mode = mode
        self.device = device
        self.species_mapping = species_mapping
        
        # Load all data at initialization for consistency
        self._load_dataset()
        
    def _load_dataset(self):
        """Load and prepare all data at initialization."""
        logger.info(f"Loading dataset with {len(self.observation_ids)} observations...")
        
        # Load data in batches to manage memory
        batch_size = 64
        all_species = []
        all_language_embs = []
        all_vision_embs = []
        
        total_batches = (len(self.observation_ids) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(self.observation_ids), batch_size):
            batch_ids = self.observation_ids[batch_idx:batch_idx + batch_size]
            current_batch = (batch_idx // batch_size) + 1
            
            print(f"Loading batch {current_batch}/{total_batches} ({len(batch_ids)} observations)...")
            
            batch_data = get_training_batch(
                self.cache,
                batch_ids,
                include_vision=(self.mode in ['vision', 'both']),
                include_language=(self.mode in ['language', 'both']),
                device='cpu'  # Load to CPU first, move to device later
            )
            
            all_species.extend(batch_data['species'])
            
            if self.mode in ['language', 'both']:
                all_language_embs.append(batch_data['language_embeddings'])
            if self.mode in ['vision', 'both']:
                all_vision_embs.append(batch_data['vision_embeddings'])
        
        # Create species label mapping
        if self.species_mapping is not None:
            self.species_to_idx = self.species_mapping
            self.idx_to_species = {idx: species for species, idx in self.species_mapping.items()}
            self.num_classes = len(self.species_mapping)
        else:
            unique_species = sorted(list(set(all_species)))
            self.species_to_idx = {species: idx for idx, species in enumerate(unique_species)}
            self.idx_to_species = {idx: species for species, idx in self.species_to_idx.items()}
            self.num_classes = len(unique_species)
        
        # Convert to tensors
        self.species_labels = torch.tensor([self.species_to_idx[species] for species in all_species], 
                                         dtype=torch.long, device=self.device)
        
        if self.mode in ['language', 'both']:
            self.language_embeddings = torch.cat(all_language_embs, dim=0).to(self.device)
            
        if self.mode in ['vision', 'both']:
            self.vision_embeddings = torch.cat(all_vision_embs, dim=0).to(self.device)
        
        logger.info(f"Dataset loaded: {len(self.observation_ids)} observations, {self.num_classes} species")
        
    def __len__(self):
        return len(self.observation_ids)
    
    def __getitem__(self, idx):
        """Get single sample from dataset."""
        sample = {'species_label': self.species_labels[idx]}
        
        if self.mode in ['language', 'both']:
            sample['language_embedding'] = self.language_embeddings[idx]
            
        if self.mode in ['vision', 'both']:
            sample['vision_embedding'] = self.vision_embeddings[idx]
            
        return sample


class LanguageClassifier(nn.Module):
    """
    🔤 Language Embedding → Species Classifier
    
    Simple MLP for classifying species from DeepSeek-V3 language embeddings.
    Architecture: 7168 → Linear → 128 → 128 → num_species
    """
    
    def __init__(self, num_classes: int, embedding_dim: int = 7168, hidden_dim: int = 128):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, language_embeddings):
        """Forward pass through language classifier."""
        return self.classifier(language_embeddings)


class VisionClassifier(nn.Module):
    """
    👁️ Vision Embedding → Species Classifier
    
    Simple MLP for classifying species from V-JEPA-2 vision embeddings.
    Since vision features are already transformer-processed, we use global pooling + MLP.
    Architecture: (8, 24, 24, 1408) → Global Mean Pool → MLP → num_species
    """
    
    def __init__(self, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        
        # Vision embeddings are (8, 24, 24, 1408) - temporal, height, width, features
        # Global average pooling across spatial and temporal dimensions
        # Results in 1408-dimensional feature vector per image
        
        self.classifier = nn.Sequential(
            nn.Linear(1408, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, vision_embeddings):
        """Forward pass through vision classifier."""
        # Global average pooling across spatial (H, W) and temporal (T) dimensions
        # Input: (batch, 8, 24, 24, 1408) → Output: (batch, 1408)
        x = vision_embeddings.mean(dim=(1, 2, 3))  # Average over T, H, W dimensions
        
        # Pass through classifier
        x = self.classifier(x)
        return x


def create_training_visualization(train_losses, train_accuracies, test_accuracies, mode):
    """
    Create elegant training visualization with loss and accuracy curves.
    
    Args:
        train_losses: List of training losses per epoch
        train_accuracies: List of training accuracies per epoch  
        test_accuracies: List of test accuracies per epoch
        mode: Training mode (language, vision, or both)
    """
    # Set elegant style
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.alpha': 0.3
    })
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Color scheme
    train_color = '#2E86AB'  # Deep blue
    test_color = '#A23B72'   # Deep red/magenta
    loss_color = '#F18F01'   # Orange
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curve
    ax1.plot(epochs, train_losses, color=loss_color, linewidth=2.5, marker='o', 
             markersize=6, label='Training Loss', alpha=0.9)
    ax1.set_xlabel('Epoch', fontweight='medium')
    ax1.set_ylabel('Cross-Entropy Loss', fontweight='medium')
    ax1.set_title(f'Training Loss Curve\n{mode.title()} Embedding Classifier', 
                  fontweight='bold', fontsize=13)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    
    # Accuracy curves
    ax2.plot(epochs, train_accuracies, color=train_color, linewidth=2.5, marker='o',
             markersize=6, label='Training Accuracy', alpha=0.9)
    ax2.plot(epochs, test_accuracies, color=test_color, linewidth=2.5, marker='s',
             markersize=6, label='Test Accuracy', alpha=0.9)
    ax2.set_xlabel('Epoch', fontweight='medium')
    ax2.set_ylabel('Accuracy (%)', fontweight='medium')
    ax2.set_title(f'Classification Accuracy\n{mode.title()} Embedding Classifier', 
                  fontweight='bold', fontsize=13)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    ax2.set_ylim(0, 100)
    
    # Main title
    fig.suptitle('DeepEarth Species Classification Training Results', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Add summary statistics
    final_train_acc = train_accuracies[-1]
    final_test_acc = test_accuracies[-1]
    best_test_acc = max(test_accuracies)
    final_loss = train_losses[-1]
    
    stats_text = f"""Training Summary:
    • Final Train Accuracy: {final_train_acc:.1f}%
    • Final Test Accuracy: {final_test_acc:.1f}%
    • Best Test Accuracy: {best_test_acc:.1f}%
    • Final Loss: {final_loss:.4f}
    • Training Mode: {mode.title()} embeddings"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', 
                      edgecolor='#dee2e6', alpha=0.9))
    
    plt.tight_layout()
    
    # Save the visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_path = Path(__file__).parent / f"docs/training_curves_{mode}_{timestamp}.png"
    viz_path.parent.mkdir(exist_ok=True)
    
    plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Training visualization saved to: {viz_path}")
    
    plt.close(fig)  # Close figure instead of showing
    return viz_path


def load_train_test_split(config_path: str) -> Tuple[List[str], List[str]]:
    """
    Load train/test split from configuration file.
    
    Args:
        config_path: Path to split configuration JSON
        
    Returns:
        Tuple of (train_observation_ids, test_observation_ids)
    """
    logger.info(f"Loading train/test split from {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    observation_mappings = config['observation_mappings']
    
    train_ids = [obs_id for obs_id, metadata in observation_mappings.items() 
                 if metadata['split'] == 'train']
    test_ids = [obs_id for obs_id, metadata in observation_mappings.items() 
                if metadata['split'] == 'test']
    
    logger.info(f"Loaded split: {len(train_ids)} train, {len(test_ids)} test observations")
    return train_ids, test_ids


def train_epoch(model, dataloader, criterion, optimizer, device: str) -> Tuple[float, float]:
    """
    Train model for one epoch.
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        # Get inputs based on model type
        if isinstance(model, LanguageClassifier):
            inputs = batch['language_embedding']
        elif isinstance(model, VisionClassifier):
            inputs = batch['vision_embedding']
        else:
            raise ValueError(f"Unknown model type: {type(model)}")
        
        targets = batch['species_label']
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def evaluate_model(model, dataloader, criterion, device: str) -> Tuple[float, float]:
    """
    Evaluate model on validation/test set.
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Get inputs based on model type
            if isinstance(model, LanguageClassifier):
                inputs = batch['language_embedding']
            elif isinstance(model, VisionClassifier):
                inputs = batch['vision_embedding']
            else:
                raise ValueError(f"Unknown model type: {type(model)}")
            
            targets = batch['species_label']
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def main():
    """Main training loop."""
    parser = argparse.ArgumentParser(description='DeepEarth Species Classification Training')
    parser.add_argument('--mode', choices=['language', 'vision', 'both'], default='language',
                       help='Training mode: language, vision, or both modalities')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device: cuda, cpu, or auto')
    parser.add_argument('--config', type=str, help='Path to train/test split config')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    logger.info(f"Using device: {device}")
    
    # Auto-detect config file if not provided
    if not args.config:
        config_path = Path(__file__).parent / "config" / "central_florida_split.json"
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            logger.error("Please run create_train_test_split.py first or specify --config")
            return 1
        args.config = str(config_path)
    
    print(f"🎯 DeepEarth Species Classification Training")
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    
    # Initialize cache and load data split
    print("\n📊 Loading data...")
    try:
        # Change to dashboard directory for cache initialization
        original_cwd = Path.cwd()
        dashboard_dir = Path(__file__).parent.parent / "dashboard"
        import os
        os.chdir(dashboard_dir)
        
        cache = UnifiedDataCache("dataset_config.json")
        train_ids, test_ids = load_train_test_split(args.config)
        
        # Return to original directory
        os.chdir(original_cwd)
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1
    
    # Create datasets with consistent species mapping
    print("📦 Creating datasets...")
    
    # First, determine the species that appear in both splits for consistent mapping
    print("   Analyzing species distribution...")
    sample_train_data = get_training_batch(cache, train_ids[:500], include_vision=False, include_language=True, device='cpu')
    sample_test_data = get_training_batch(cache, test_ids[:100], include_vision=False, include_language=True, device='cpu')
    
    train_species = set(sample_train_data['species'])
    test_species = set(sample_test_data['species'])
    common_species = train_species & test_species
    
    print(f"   Train species: {len(train_species)}, Test species: {len(test_species)}")
    print(f"   Common species: {len(common_species)}")
    
    # Filter observation IDs to only include common species
    def filter_ids_by_species(obs_ids, target_species, max_samples=500):
        """Filter observation IDs to only include observations from target species."""
        filtered_ids = []
        batch_size = 64
        
        for i in range(0, min(len(obs_ids), max_samples * 2), batch_size):
            batch_ids = obs_ids[i:i + batch_size]
            if not batch_ids:
                break
                
            batch_data = get_training_batch(cache, batch_ids, include_vision=False, include_language=True, device='cpu')
            
            for j, species in enumerate(batch_data['species']):
                if species in target_species and len(filtered_ids) < max_samples:
                    filtered_ids.append(batch_ids[j])
                    
            if len(filtered_ids) >= max_samples:
                break
                
        return filtered_ids
    
    print("   Filtering to common species...")
    filtered_train_ids = filter_ids_by_species(train_ids, common_species, max_samples=800)
    filtered_test_ids = filter_ids_by_species(test_ids, common_species, max_samples=200)
    
    # Create consistent species mapping
    common_species_sorted = sorted(list(common_species))
    species_mapping = {species: idx for idx, species in enumerate(common_species_sorted)}
    
    train_dataset = DeepEarthDataset(filtered_train_ids, cache, args.mode, device, species_mapping)
    test_dataset = DeepEarthDataset(filtered_test_ids, cache, args.mode, device, species_mapping)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of species: {train_dataset.num_classes}")
    
    # Initialize model based on mode
    print(f"\n🧠 Initializing {args.mode} classifier...")
    if args.mode == 'language':
        model = LanguageClassifier(train_dataset.num_classes).to(device)
    elif args.mode == 'vision':
        model = VisionClassifier(train_dataset.num_classes).to(device)
    else:
        logger.error("Multi-modal training not implemented in this skeleton")
        return 1
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        
        # Record metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        print(f"Epoch {epoch+1:2d}/{args.epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.1f}%, "
              f"Test Acc: {test_acc:.1f}%")
        
        # Log embedding loader performance every few epochs
        if (epoch + 1) % 5 == 0 and hasattr(cache, 'mmap_loader') and cache.mmap_loader:
            cache.mmap_loader.log_performance_summary()
    
    # Final results
    final_test_loss, final_test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f"\n✅ Training Complete!")
    print(f"Final Test Accuracy: {final_test_acc:.1f}%")
    print(f"Best Test Accuracy: {max(test_accuracies):.1f}%")
    
    # Create elegant loss and accuracy curves
    print("📊 Creating training visualization...")
    create_training_visualization(train_losses, train_accuracies, test_accuracies, args.mode)
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = Path(__file__).parent / f"models/deepearth_{args.mode}_classifier_{timestamp}.pth"
    model_path.parent.mkdir(exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': train_dataset.num_classes,
        'species_to_idx': train_dataset.species_to_idx,
        'mode': args.mode,
        'final_accuracy': final_test_acc
    }, model_path)
    
    print(f"Model saved to: {model_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())