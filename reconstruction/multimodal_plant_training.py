import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import ViTModel, ViTImageProcessor, AutoModel, AutoTokenizer
from datasets import load_dataset
from PIL import Image
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from sklearn.neighbors import NearestNeighbors
import wandb
from typing import Dict, List, Tuple, Optional
import os

class MultimodalConfig:
    """Configuration for multimodal training"""
    def __init__(self):
        # Model configs
        self.vit_model_name = "google/vit-base-patch16-224"
        self.deepseek_model_name = "deepseek-ai/deepseek-coder-1.3b-base"  # Adjust based on available model
        
        # Embedding dimensions
        self.vision_embed_dim = 768  # ViT base output
        self.language_embed_dim = 2048  # DeepSeek output dimension
        self.shared_embed_dim = 2048  # Shared multimodal space
        
        # Training configs
        self.batch_size = 8
        self.learning_rate = 1e-4
        self.num_epochs = 100
        self.mask_ratio = 0.5
        self.num_species = 10
        self.images_per_species = 25
        self.test_images_per_species = 10
        
        # Architecture configs
        self.mlp_hidden_dim = 256
        self.dropout = 0.1
        
        # Logging
        self.log_interval = 10
        self.eval_interval = 5

class VisionEncoder(nn.Module):
    """ViT-based vision encoder with projection to shared space"""
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.vit = ViTModel.from_pretrained(config.vit_model_name)
        self.processor = ViTImageProcessor.from_pretrained(config.vit_model_name)
        
        # Freeze ViT weights as per requirements
        for param in self.vit.parameters():
            param.requires_grad = False
            
        # Projection to shared embedding space
        self.vision_proj = nn.Sequential(
            nn.Linear(config.vision_embed_dim, config.mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden_dim, config.shared_embed_dim)
        )
        
    def forward(self, images):
        # Get ViT embeddings
        with torch.no_grad():
            vit_outputs = self.vit(pixel_values=images)
            vision_embeds = vit_outputs.last_hidden_state.mean(dim=1)  # Average pooling
        
        # Project to shared space
        shared_embeds = self.vision_proj(vision_embeds)
        return shared_embeds

class LanguageEncoder(nn.Module):
    """DeepSeek-based language encoder with projection to shared space"""
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.deepseek = AutoModel.from_pretrained(config.deepseek_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.deepseek_model_name)
        
        # Freeze DeepSeek weights
        for param in self.deepseek.parameters():
            param.requires_grad = False
            
        # Get actual embedding dimension from model
        actual_dim = self.deepseek.config.hidden_size
        
        # Projection to shared embedding space
        self.language_proj = nn.Sequential(
            nn.Linear(actual_dim, config.mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden_dim, config.shared_embed_dim)
        )
        
    def forward(self, input_ids, attention_mask):
        # Get DeepSeek embeddings
        with torch.no_grad():
            outputs = self.deepseek(input_ids=input_ids, attention_mask=attention_mask)
            language_embeds = outputs.last_hidden_state.mean(dim=1)  # Average pooling
        
        # Project to shared space
        shared_embeds = self.language_proj(language_embeds)
        return shared_embeds

class CrossModalTransformer(nn.Module):
    """Transformer for cross-modal interaction and reconstruction"""
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config
        
        # Transformer for multimodal integration
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.shared_embed_dim,
                nhead=8,
                dim_feedforward=2048,
                dropout=config.dropout,
                batch_first=True
            ),
            num_layers=4
        )
        
        # Reconstruction heads
        self.vision_decoder = nn.Sequential(
            nn.Linear(config.shared_embed_dim, config.mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden_dim, config.shared_embed_dim)
        )
        
        self.language_decoder = nn.Sequential(
            nn.Linear(config.shared_embed_dim, config.mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden_dim, config.shared_embed_dim)
        )
        
    def forward(self, vision_embeds, language_embeds, mask_vision=True, mask_language=True):
        batch_size = vision_embeds.shape[0]
        device = vision_embeds.device
        
        # Create masks
        vision_mask = torch.zeros(batch_size, 1, device=device)
        language_mask = torch.zeros(batch_size, 1, device=device)
        
        if mask_vision:
            # Randomly mask vision embeddings
            vision_mask = (torch.rand(batch_size, 1, device=device) < self.config.mask_ratio).float()
            vision_embeds = vision_embeds * (1 - vision_mask)
            
        if mask_language:
            # Randomly mask language embeddings
            language_mask = (torch.rand(batch_size, 1, device=device) < self.config.mask_ratio).float()
            language_embeds = language_embeds * (1 - language_mask)
        
        # Stack embeddings for transformer input [batch, 2, embed_dim]
        multimodal_input = torch.stack([vision_embeds, language_embeds], dim=1)
        
        # Apply transformer
        transformed = self.transformer(multimodal_input)
        
        # Reconstruct
        vision_recon = self.vision_decoder(transformed[:, 0])
        language_recon = self.language_decoder(transformed[:, 1])
        
        return vision_recon, language_recon, vision_mask, language_mask

class MultimodalModel(nn.Module):
    """Complete multimodal model combining all components"""
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config
        self.vision_encoder = VisionEncoder(config)
        self.language_encoder = LanguageEncoder(config)
        self.cross_modal = CrossModalTransformer(config)
        
    def forward(self, images, input_ids, attention_mask, mask_vision=True, mask_language=True):
        # Encode modalities
        vision_embeds = self.vision_encoder(images)
        language_embeds = self.language_encoder(input_ids, attention_mask)
        
        # Cross-modal interaction and reconstruction
        vision_recon, language_recon, vision_mask, language_mask = self.cross_modal(
            vision_embeds, language_embeds, mask_vision, mask_language
        )
        
        return {
            'vision_embeds': vision_embeds,
            'language_embeds': language_embeds,
            'vision_recon': vision_recon,
            'language_recon': language_recon,
            'vision_mask': vision_mask,
            'language_mask': language_mask
        }

class PlantSpeciesDataset(Dataset):
    """Dataset for Central Florida Native Plants"""
    def __init__(self, config: MultimodalConfig, split='train'):
        self.config = config
        self.split = split
        
        # Load dataset
        dataset = load_dataset("deepearth/central-florida-native-plants", split="train")
        
        # Select subset of species
        unique_species = list(set(dataset['taxon_name']))[:config.num_species]
        self.species_to_idx = {species: idx for idx, species in enumerate(unique_species)}
        
        # Filter and split data
        self.data = []
        for species in unique_species:
            species_data = [item for item in dataset if item['taxon_name'] == species]
            
            if split == 'train':
                self.data.extend(species_data[:config.images_per_species - config.test_images_per_species])
            else:
                self.data.extend(species_data[-config.test_images_per_species:])
        
        # Initialize processors
        self.image_processor = ViTImageProcessor.from_pretrained(config.vit_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.deepseek_model_name)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Process image
        image = item['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        pixel_values = self.image_processor(image, return_tensors="pt")['pixel_values'].squeeze()
        
        # Process text - using "ecophysiology of [species]" as mentioned in notes
        text = f"ecophysiology of {item['taxon_name']}"
        text_inputs = self.tokenizer(text, padding='max_length', truncation=True, 
                                     max_length=128, return_tensors="pt")
        
        return {
            'pixel_values': pixel_values,
            'input_ids': text_inputs['input_ids'].squeeze(),
            'attention_mask': text_inputs['attention_mask'].squeeze(),
            'species': item['taxon_name'],
            'species_idx': self.species_to_idx[item['taxon_name']]
        }

class MultimodalTrainer:
    """Trainer for the multimodal model"""
    def __init__(self, model: MultimodalModel, config: MultimodalConfig):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        
        # Loss tracking
        self.train_losses = []
        self.val_losses = []
        self.embeddings_history = []
        
    def compute_loss(self, outputs, original_vision, original_language):
        """Compute reconstruction loss (MSE)"""
        vision_loss = F.mse_loss(outputs['vision_recon'], original_vision)
        language_loss = F.mse_loss(outputs['language_recon'], original_language)
        
        # Weight losses based on what was masked
        total_loss = outputs['vision_mask'].mean() * vision_loss + \
                    outputs['language_mask'].mean() * language_loss
        
        return total_loss, vision_loss, language_loss
    
    def evaluate_nearest_neighbors(self, val_loader):
        """Evaluate using k-nearest neighbors in embedding space"""
        self.model.eval()
        
        all_vision_embeds = []
        all_language_embeds = []
        all_species = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(
                    batch['pixel_values'], 
                    batch['input_ids'], 
                    batch['attention_mask'],
                    mask_vision=False,
                    mask_language=False
                )
                
                all_vision_embeds.append(outputs['vision_embeds'].cpu())
                all_language_embeds.append(outputs['language_embeds'].cpu())
                all_species.extend(batch['species'])
        
        # Concatenate all embeddings
        vision_embeds = torch.cat(all_vision_embeds)
        language_embeds = torch.cat(all_language_embeds)
        
        # Compute nearest neighbors
        nn_model = NearestNeighbors(n_neighbors=5)
        nn_model.fit(language_embeds.numpy())
        
        # Find nearest language embeddings for each vision embedding
        distances, indices = nn_model.kneighbors(vision_embeds.numpy())
        
        # Calculate accuracy
        correct = 0
        for i, neighbors in enumerate(indices):
            if all_species[i] == all_species[neighbors[0]]:
                correct += 1
        
        accuracy = correct / len(indices)
        return accuracy, vision_embeds, language_embeds, all_species
    
    def visualize_embeddings(self, vision_embeds, language_embeds, species, epoch):
        """Visualize embeddings using UMAP"""
        # Combine embeddings for visualization
        all_embeds = torch.cat([vision_embeds, language_embeds]).numpy()
        labels = species + species  # Duplicate for both modalities
        modality = ['vision'] * len(vision_embeds) + ['language'] * len(language_embeds)
        
        # Apply UMAP
        reducer = umap.UMAP(n_components=3, random_state=42)
        reduced_embeds = reducer.fit_transform(all_embeds)
        
        # Plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color by species
        unique_species = list(set(species))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_species)))
        
        for i, spec in enumerate(unique_species):
            mask = [l == spec for l in labels]
            points = reduced_embeds[mask]
            mod = [m for m, ma in zip(modality, mask) if ma]
            
            # Different markers for vision/language
            vision_mask = [m == 'vision' for m in mod]
            language_mask = [m == 'language' for m in mod]
            
            ax.scatter(points[vision_mask, 0], points[vision_mask, 1], points[vision_mask, 2],
                      c=[colors[i]], marker='o', s=50, label=f'{spec} (vision)')
            ax.scatter(points[language_mask, 0], points[language_mask, 1], points[language_mask, 2],
                      c=[colors[i]], marker='^', s=50, label=f'{spec} (language)')
        
        ax.set_title(f'Embedding Space at Epoch {epoch}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'embeddings_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return reduced_embeds
    
    def train(self, train_loader, val_loader, num_epochs):
        """Main training loop"""
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0
            epoch_vision_loss = 0
            epoch_language_loss = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            for batch_idx, batch in enumerate(pbar):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    batch['pixel_values'], 
                    batch['input_ids'], 
                    batch['attention_mask']
                )
                
                # Compute loss
                loss, vision_loss, language_loss = self.compute_loss(
                    outputs, 
                    outputs['vision_embeds'].detach(), 
                    outputs['language_embeds'].detach()
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                epoch_vision_loss += vision_loss.item()
                epoch_language_loss += language_loss.item()
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'v_loss': vision_loss.item(),
                    'l_loss': language_loss.item()
                })
            
            # Average losses
            avg_loss = epoch_loss / len(train_loader)
            avg_vision_loss = epoch_vision_loss / len(train_loader)
            avg_language_loss = epoch_language_loss / len(train_loader)
            
            self.train_losses.append(avg_loss)
            
            # Validation
            if (epoch + 1) % self.config.eval_interval == 0:
                accuracy, vision_embeds, language_embeds, species = self.evaluate_nearest_neighbors(val_loader)
                print(f"\nEpoch {epoch+1} - Train Loss: {avg_loss:.4f}, "
                      f"Vision Loss: {avg_vision_loss:.4f}, Language Loss: {avg_language_loss:.4f}, "
                      f"NN Accuracy: {accuracy:.4f}")
                
                # Visualize embeddings
                self.visualize_embeddings(vision_embeds, language_embeds, species, epoch+1)
                self.embeddings_history.append((vision_embeds, language_embeds, species))
        
        return self.train_losses, self.embeddings_history

def main():
    # Initialize configuration
    config = MultimodalConfig()
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = PlantSpeciesDataset(config, split='train')
    val_dataset = PlantSpeciesDataset(config, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Create model
    print("Initializing model...")
    model = MultimodalModel(config)
    
    # Create trainer
    trainer = MultimodalTrainer(model, config)
    
    # Train
    print("Starting training...")
    losses, embeddings_history = trainer.train(train_loader, val_loader, config.num_epochs)
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('training_loss.png')
    plt.close()
    
    # Save model
    torch.save(model.state_dict(), 'multimodal_model.pth')
    print("Training complete!")

if __name__ == "__main__":
    main()
