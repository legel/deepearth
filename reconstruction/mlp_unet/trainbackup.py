#!/usr/bin/env python3
"""
Training script for 10 species with the downloaded folder structure
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from datetime import datetime
import argparse

from model import BimodalMLPUNet
from dataset import CentralFloridaPlantsDataModule


# Map folder names back to species names
FOLDER_TO_SPECIES = {
    'american_beautyberry': 'Callicarpa americana',
    'beach_sunflower': 'Helianthus debilis',
    'black_eyed_susan': 'Rudbeckia hirta',
    'blanket_flower': 'Gaillardia pulchella',
    'coontie': 'Zamia integrifolia',
    'leavenworth_tickseed': 'Coreopsis leavenworthii',
    'saw_palmetto': 'Serenoa repens',
    'spiderwort': 'Tradescantia ohiensis',
    'spotted_beebalm': 'Monarda punctata',
    'tropical_sage': 'Salvia coccinea'
}


def get_species_list_from_folders(data_dir='./data/plants'):
    """Get species list based on actual folders present"""
    species_list = []
    
    # Get all directories in data_dir
    folders = [f for f in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, f)) and not f.startswith('.')]
    
    # Sort for consistency
    folders.sort()
    
    print(f"\n📁 Found {len(folders)} species folders:")
    for folder in folders:
        species_name = FOLDER_TO_SPECIES.get(folder, folder.replace('_', ' ').title())
        species_list.append(species_name)
        
        # Count images
        folder_path = os.path.join(data_dir, folder)
        image_count = len([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
        print(f"   {folder:<25} → {species_name:<35} ({image_count} images)")
    
    return species_list


class ContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss for better species separation"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, image_features, text_features):
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity matrix
        logits = image_features @ text_features.T / self.temperature
        
        # Create labels (diagonal should be 1)
        labels = torch.arange(len(logits), device=logits.device)
        
        # Compute cross entropy loss in both directions
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        return (loss_i2t + loss_t2i) / 2


class ImprovedTrainer:
    """Trainer with curriculum learning and contrastive loss"""
    
    def __init__(self,
                 model: BimodalMLPUNet,
                 data_module: CentralFloridaPlantsDataModule,
                 learning_rate: float = 5e-5,
                 weight_decay: float = 0.01,
                 warmup_epochs: int = 5,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 checkpoint_dir: str = './checkpoints',
                 log_dir: str = './logs'):
        
        self.model = model.to(device)
        self.data_module = data_module
        self.device = device
        self.warmup_epochs = warmup_epochs
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Losses
        self.mse_loss = nn.MSELoss()
        self.contrastive_loss = ContrastiveLoss(temperature=0.07)
        
        # Logging
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(f"{log_dir}/run_10species_{timestamp}")
        
        self.global_step = 0
        self.best_accuracy = 0.0
        
    def get_adaptive_mask_ratio(self, epoch, max_epochs):
        """Curriculum learning: gradually increase masking"""
        if epoch < self.warmup_epochs:
            return 0.1  # Easy start
        else:
            progress = (epoch - self.warmup_epochs) / (max_epochs - self.warmup_epochs)
            return min(0.7, 0.1 + 0.6 * progress)
    
    def train_epoch(self, epoch: int, max_epochs: int):
        """Train for one epoch"""
        self.model.train()
        train_loader = self.data_module.train_dataloader()
        
        # Adaptive mask ratio
        mask_ratio = self.get_adaptive_mask_ratio(epoch, max_epochs)
        
        total_loss = 0
        total_mse_loss = 0
        total_contrastive_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} (mask={mask_ratio:.2f})")
        for batch_idx, (images, species_names) in enumerate(pbar):
            images = images.to(self.device)
            
            # Clear gradients
            self.optimizer.zero_grad()
            
            # Forward direction: Image → Species
            reconstructed_species, image_embeddings, _ = self.model.forward_image_to_species(
                images, mask_ratio
            )
            target_species_embeddings = self.model.species_embeddings.get_embedding(species_names)
            
            # Reverse direction: Species → Image  
            reconstructed_images, species_embeddings, _ = self.model.forward_species_to_image(
                species_names, mask_ratio
            )
            target_image_embeddings = self.model.image_encoder(images)
            
            # MSE losses
            loss_mse_forward = self.mse_loss(reconstructed_species, target_species_embeddings)
            loss_mse_reverse = self.mse_loss(reconstructed_images, target_image_embeddings)
            loss_mse = loss_mse_forward + loss_mse_reverse
            
            # Contrastive loss on original embeddings
            with torch.no_grad():
                clean_image_emb = self.model.image_encoder(images)
                clean_species_emb = self.model.species_embeddings.get_embedding(species_names)
            
            loss_contrastive = self.contrastive_loss(clean_image_emb, clean_species_emb)
            
            # Total loss with dynamic weighting
            contrastive_weight = min(1.0, epoch / 20)  # Gradually increase
            loss = loss_mse + contrastive_weight * loss_contrastive
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            total_mse_loss += loss_mse.item()
            total_contrastive_loss += loss_contrastive.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'mse': f"{loss_mse.item():.4f}",
                'cont': f"{loss_contrastive.item():.4f}"
            })
            
            # Log to tensorboard
            if batch_idx % 10 == 0:
                self.writer.add_scalar('Loss/train_total', loss.item(), self.global_step)
                self.writer.add_scalar('Loss/train_mse', loss_mse.item(), self.global_step)
                self.writer.add_scalar('Loss/train_contrastive', loss_contrastive.item(), self.global_step)
                self.writer.add_scalar('Training/mask_ratio', mask_ratio, self.global_step)
                self.global_step += 1
                
        # Average losses
        num_batches = len(train_loader)
        avg_loss = total_loss / num_batches
        avg_mse_loss = total_mse_loss / num_batches
        avg_contrastive_loss = total_contrastive_loss / num_batches
        
        return avg_loss, avg_mse_loss, avg_contrastive_loss
    
    def evaluate(self, epoch: int):
        """Evaluate model performance"""
        self.model.eval()
        test_loader = self.data_module.test_dataloader()
        
        correct_top1 = 0
        correct_top3 = 0
        correct_top5 = 0
        total = 0
        
        all_distances = []
        
        with torch.no_grad():
            for images, species_names in tqdm(test_loader, desc="Evaluating"):
                images = images.to(self.device)
                batch_size = images.size(0)
                
                # Predict species (no masking during evaluation)
                predictions = self.model.predict_species(images, mask_ratio=0.0, top_k=5)
                
                # Check accuracy
                for i in range(batch_size):
                    true_species = species_names[i]
                    pred_topk = predictions[i]
                    
                    if len(pred_topk) > 0 and true_species == pred_topk[0]:
                        correct_top1 += 1
                    if len(pred_topk) >= 3 and true_species in pred_topk[:3]:
                        correct_top3 += 1
                    if true_species in pred_topk:
                        correct_top5 += 1
                    total += 1
                    
                # Calculate embedding distances
                reconstructed, _, _ = self.model.forward_image_to_species(images, mask_ratio=0.0)
                target_embeddings = self.model.species_embeddings.get_embedding(species_names)
                distances = torch.norm(reconstructed - target_embeddings, dim=1)
                all_distances.extend(distances.cpu().numpy())
                
        # Calculate metrics
        accuracy_top1 = correct_top1 / total if total > 0 else 0
        accuracy_top3 = correct_top3 / total if total > 0 else 0
        accuracy_top5 = correct_top5 / total if total > 0 else 0
        avg_distance = np.mean(all_distances) if all_distances else 0
        
        # Log metrics
        self.writer.add_scalar('Accuracy/top1', accuracy_top1, epoch)
        self.writer.add_scalar('Accuracy/top3', accuracy_top3, epoch)
        self.writer.add_scalar('Accuracy/top5', accuracy_top5, epoch)
        self.writer.add_scalar('Distance/avg_embedding', avg_distance, epoch)
        
        print(f"\nEvaluation Results:")
        print(f"Top-1 Accuracy: {accuracy_top1:.4f}")
        print(f"Top-3 Accuracy: {accuracy_top3:.4f}")
        print(f"Top-5 Accuracy: {accuracy_top5:.4f}")
        print(f"Avg Embedding Distance: {avg_distance:.4f}")
        
        return accuracy_top1, accuracy_top3, avg_distance
    
    def save_checkpoint(self, epoch: int, accuracy: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'species_list': self.model.species_embeddings.species_list
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'latest.pth'))
        
        # Save best checkpoint
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'best.pth'))
            print(f"New best model saved with accuracy: {accuracy:.4f}")
            
    def train(self, num_epochs: int, eval_every: int = 5):
        """Main training loop"""
        print(f"\n🚀 Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{num_epochs}")
            
            # Train
            avg_loss, avg_mse, avg_cont = self.train_epoch(epoch, num_epochs)
            print(f"Avg Loss: {avg_loss:.4f} (MSE: {avg_mse:.4f}, Contrastive: {avg_cont:.4f})")
            
            # Update learning rate
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            self.writer.add_scalar('Learning_rate', current_lr, epoch)
            
            # Evaluate
            if epoch % eval_every == 0:
                accuracy_top1, accuracy_top3, avg_distance = self.evaluate(epoch)
                self.save_checkpoint(epoch, accuracy_top1)
                
        print("\n✅ Training completed!")
        print(f"Best accuracy: {self.best_accuracy:.4f}")
        self.writer.close()


def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description='Train MLP U-Net with 10 plant species')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--eval-every', type=int, default=5,
                        help='Evaluate every N epochs')
    parser.add_argument('--images-per-species', type=int, default=200,
                        help='Maximum images per species')
    parser.add_argument('--data-dir', type=str, default='./data/plants',
                        help='Directory containing species folders')
    
    args = parser.parse_args()
    
    print("🌿 MLP U-Net Training with 10 Plant Species")
    print("=" * 50)
    
    # Get species list from folders
    species_list = get_species_list_from_folders(args.data_dir)
    
    if not species_list:
        print("❌ No species folders found!")
        return
    
    print(f"\n✅ Found {len(species_list)} species to train with")
    
    # Configuration
    config = {
        'species_list': species_list,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'embedding_dim': 2048,
        'hidden_dim': 512,
        'bottleneck_dim': 128,
        'images_per_species': args.images_per_species,
        'test_split': 0.2
    }
    
    # Create data module
    print("\n📊 Creating data module...")
    data_module = CentralFloridaPlantsDataModule(
        root_dir=args.data_dir,
        species_list=config['species_list'],
        batch_size=config['batch_size'],
        images_per_species=config['images_per_species'],
        test_split=config['test_split']
    )
    
    # Check dataset
    train_dist, test_dist = data_module.get_species_distribution()
    print(f"\nDataset statistics:")
    print(f"  Training samples: {sum(train_dist.values())}")
    print(f"  Test samples: {sum(test_dist.values())}")
    
    # Create model
    print("\n🧠 Creating model...")
    model = BimodalMLPUNet(
        species_list=config['species_list'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        bottleneck_dim=config['bottleneck_dim']
    )
    
    # Create trainer
    trainer = ImprovedTrainer(
        model=model,
        data_module=data_module,
        learning_rate=config['learning_rate'],
        warmup_epochs=args.warmup_epochs
    )
    
    # Train
    print("\n🚀 Starting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Warmup epochs: {args.warmup_epochs}")
    print(f"  Features: Contrastive learning + Curriculum masking")
    
    trainer.train(num_epochs=args.epochs, eval_every=args.eval_every)
    
    print("\n✅ Training complete!")
    print(f"📁 Checkpoints saved to: ./checkpoints/")
    print(f"📊 To view training logs: tensorboard --logdir ./logs")


if __name__ == "__main__":
    main()
