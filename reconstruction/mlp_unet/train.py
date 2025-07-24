#!/usr/bin/env python3
"""
Train with ALL available images instead of limiting to 50 per species
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from datetime import datetime

from model import BimodalMLPUNet
from dataset import CentralFloridaPlantsDataModule

# Species list
SPECIES_LIST = [
    'Callicarpa americana',      # american_beautyberry
    'Helianthus debilis',        # beach_sunflower
    'Rudbeckia hirta',           # black_eyed_susan
    'Gaillardia pulchella',      # blanket_flower
    'Zamia integrifolia',        # coontie
    'Coreopsis leavenworthii',   # leavenworth_tickseed
    'Serenoa repens',            # saw_palmetto
    'Tradescantia ohiensis',     # spiderwort
    'Monarda punctata',          # spotted_beebalm
    'Salvia coccinea'            # tropical_sage
]


def count_available_images():
    """Count actual available images"""
    folder_mapping = {
        'Callicarpa americana': 'american_beautyberry',
        'Helianthus debilis': 'beach_sunflower',
        'Rudbeckia hirta': 'black_eyed_susan',
        'Gaillardia pulchella': 'blanket_flower',
        'Zamia integrifolia': 'coontie',
        'Coreopsis leavenworthii': 'leavenworth_tickseed',
        'Serenoa repens': 'saw_palmetto',
        'Tradescantia ohiensis': 'spiderwort',
        'Monarda punctata': 'spotted_beebalm',
        'Salvia coccinea': 'tropical_sage'
    }
    
    total_images = 0
    print("\n📊 Available images per species:")
    for species in SPECIES_LIST:
        folder = folder_mapping[species]
        folder_path = f'./data/plants/{folder}'
        if os.path.exists(folder_path):
            count = len([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
            print(f"  {species:<30} ({folder:<25}): {count} images")
            total_images += count
    
    print(f"\n  Total available images: {total_images}")
    return total_images


class SimpleTrainer:
    def __init__(self, model, data_module, lr=1e-4, device='cuda'):
        self.model = model.to(device)
        self.data_module = data_module
        self.device = device
        
        # Adam optimizer with lower learning rate
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=5, factor=0.5
        )
        self.criterion = nn.MSELoss()
        
        os.makedirs('./checkpoints_alldata', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(f"./logs/alldata_{timestamp}")
        
        self.best_accuracy = 0.0
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        train_loader = self.data_module.train_dataloader()
        
        total_loss = 0
        correct = 0
        total = 0
        
        # Use lower mask ratio for more stable training
        mask_ratio = min(0.5, 0.1 + 0.4 * (epoch / 50))
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for images, species_names in pbar:
            images = images.to(self.device)
            batch_size = images.size(0)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            reconstructed, _, _ = self.model.forward_image_to_species(images, mask_ratio)
            target = self.model.species_embeddings.get_embedding(species_names)
            
            # Loss
            loss = self.criterion(reconstructed, target)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            with torch.no_grad():
                predictions = self.model.predict_species(images, mask_ratio=0.0, top_k=1)
                for i in range(batch_size):
                    if predictions[i][0] == species_names[i]:
                        correct += 1
                    total += 1
            
            acc = correct / total if total > 0 else 0
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{acc:.2%}"})
        
        avg_loss = total_loss / len(train_loader)
        train_acc = correct / total if total > 0 else 0
        
        return avg_loss, train_acc
    
    def evaluate(self):
        """Evaluate on test set"""
        self.model.eval()
        test_loader = self.data_module.test_dataloader()
        
        correct = {1: 0, 3: 0, 5: 0}
        total = 0
        
        # Confusion matrix
        confusion = torch.zeros(len(SPECIES_LIST), len(SPECIES_LIST))
        species_to_idx = {sp: i for i, sp in enumerate(SPECIES_LIST)}
        
        with torch.no_grad():
            for images, species_names in tqdm(test_loader, desc="Evaluating"):
                images = images.to(self.device)
                predictions = self.model.predict_species(images, mask_ratio=0.0, top_k=5)
                
                for i, true_species in enumerate(species_names):
                    true_idx = species_to_idx.get(true_species, -1)
                    
                    if len(predictions[i]) > 0:
                        pred_idx = species_to_idx.get(predictions[i][0], -1)
                        if true_idx >= 0 and pred_idx >= 0:
                            confusion[true_idx, pred_idx] += 1
                        
                        # Check top-k accuracy
                        for k in [1, 3, 5]:
                            if true_species in predictions[i][:k]:
                                correct[k] += 1
                    
                    total += 1
        
        accuracies = {k: correct[k] / total if total > 0 else 0 for k in [1, 3, 5]}
        
        # Print confusion matrix
        print("\nConfusion Matrix (top-1 predictions):")
        print("True\\Pred", end="")
        for sp in SPECIES_LIST:
            print(f"\t{sp[:8]}", end="")
        print()
        
        for i, true_sp in enumerate(SPECIES_LIST):
            print(f"{true_sp[:8]}", end="")
            for j in range(len(SPECIES_LIST)):
                print(f"\t{int(confusion[i, j])}", end="")
            print()
        
        return accuracies
    
    def train(self, num_epochs=50):
        """Main training loop"""
        for epoch in range(1, num_epochs + 1):
            # Train
            avg_loss, train_acc = self.train_epoch(epoch)
            
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Train Acc: {train_acc:.2%}")
            
            # Log
            self.writer.add_scalar('Loss/train', avg_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            
            # Evaluate every 5 epochs
            if epoch % 5 == 0:
                accuracies = self.evaluate()
                
                print(f"  Test Accuracy:")
                print(f"    Top-1: {accuracies[1]:.2%}")
                print(f"    Top-3: {accuracies[3]:.2%}")
                print(f"    Top-5: {accuracies[5]:.2%}")
                
                for k, acc in accuracies.items():
                    self.writer.add_scalar(f'Accuracy/test_top{k}', acc, epoch)
                
                # Update learning rate
                self.scheduler.step(accuracies[1])
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  Learning rate: {current_lr:.2e}")
                
                # Save if best
                if accuracies[1] > self.best_accuracy:
                    self.best_accuracy = accuracies[1]
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'accuracy': accuracies[1],
                        'species_list': SPECIES_LIST
                    }, './checkpoints_alldata/best.pth')
                    print(f"  ✅ New best: {accuracies[1]:.2%}")
        
        print(f"\n✅ Training complete! Best: {self.best_accuracy:.2%}")


def main():
    # Count available images
    total_available = count_available_images()
    
    if total_available < 500:
        print("\n⚠️  Warning: Very few images available!")
        print("  Consider downloading more images for better results.")
    
    # Create data module with ALL available images
    print("\n📊 Loading ALL available images...")
    data_module = CentralFloridaPlantsDataModule(
        root_dir='./data/plants',
        species_list=SPECIES_LIST,
        batch_size=32,
        images_per_species=999999,  # Use all available images
        test_split=0.2,
        num_workers=4
    )
    
    train_dist, test_dist = data_module.get_species_distribution()
    print(f"\nActual dataset loaded:")
    print(f"  Train: {sum(train_dist.values())} images")
    print(f"  Test: {sum(test_dist.values())} images")
    
    print("\nPer species:")
    for species in SPECIES_LIST:
        train_count = train_dist.get(species, 0)
        test_count = test_dist.get(species, 0)
        print(f"  {species:<30}: {train_count:>3} train, {test_count:>3} test")
    
    # Create model
    print("\n🧠 Creating model...")
    model = BimodalMLPUNet(
        species_list=SPECIES_LIST,
        embedding_dim=2048,
        hidden_dim=512,
        bottleneck_dim=128
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train
    trainer = SimpleTrainer(
        model=model,
        data_module=data_module,
        lr=5e-5,  # Lower learning rate
        device=device
    )
    
    print("\n🚀 Starting training with ALL available data...")
    trainer.train(num_epochs=50)


if __name__ == "__main__":
    main()
