#!/usr/bin/env python3
"""
train_local.py - Training script using only local plant data
This avoids the HuggingFace dataset issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import ViTModel, ViTImageProcessor, AutoModel, AutoTokenizer
from PIL import Image
import glob
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import json

# Configuration
class Config:
    # Data
    data_root = "/home/ubuntu/a/deepearth/reconstruction/mlp_unet/data/plants"
    
    # Model
    vit_model_name = "google/vit-base-patch16-224"
    deepseek_model_name = "deepseek-ai/deepseek-coder-1.3b-base"
    
    # Dimensions
    vision_embed_dim = 768
    shared_embed_dim = 2048
    mlp_hidden_dim = 256
    dropout = 0.1
    
    # Training
    batch_size = 4  # Small batch size for A100
    learning_rate = 1e-4
    num_epochs = 50
    mask_ratio = 0.5
    
    # Logging
    eval_interval = 5
    checkpoint_dir = "checkpoints"

# Local Dataset
class LocalPlantDataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        self.data = []
        
        # Initialize processors
        print(f"Loading processors for {split}...")
        self.image_processor = ViTImageProcessor.from_pretrained(config.vit_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.deepseek_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Species mapping
        self.species_mapping = {
            "american_beautyberry": "Callicarpa americana",
            "beach_sunflower": "Helianthus debilis",
            "black_eyed_susan": "Rudbeckia hirta",
            "blanket_flower": "Gaillardia pulchella",
            "coontie": "Zamia integrifolia",
            "leavenworth_tickseed": "Coreopsis leavenworthii",
            "saw_palmetto": "Serenoa repens",
            "spiderwort": "Tradescantia ohiensis",
            "spotted_beebalm": "Monarda punctata",
            "tropical_sage": "Salvia coccinea"
        }
        
        self._load_data()
        
    def _load_data(self):
        # Use only common name folders
        species_dirs = list(self.species_mapping.keys())
        
        for species_dir in species_dirs:
            species_path = os.path.join(self.config.data_root, species_dir)
            if not os.path.exists(species_path):
                print(f"Warning: {species_path} not found")
                continue
                
            scientific_name = self.species_mapping[species_dir]
            
            # Get images
            image_files = glob.glob(os.path.join(species_path, "*.jpg"))
            image_files.extend(glob.glob(os.path.join(species_path, "*.JPG")))
            image_files.sort()
            
            # Split 80/20
            n_test = int(len(image_files) * 0.2)
            if self.split == 'train':
                selected = image_files[:-n_test] if n_test > 0 else image_files
            else:
                selected = image_files[-n_test:] if n_test > 0 else []
                
            for img_path in selected:
                self.data.append({
                    'image_path': img_path,
                    'species': scientific_name
                })
        
        # Create species index
        unique_species = sorted(set(item['species'] for item in self.data))
        self.species_to_idx = {sp: i for i, sp in enumerate(unique_species)}
        
        print(f"{self.split}: {len(self.data)} images, {len(unique_species)} species")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image = Image.open(item['image_path']).convert('RGB')
        pixel_values = self.image_processor(image, return_tensors="pt")['pixel_values'].squeeze()
        
        # Create text
        text = f"ecophysiology of {item['species']}"
        text_inputs = self.tokenizer(text, padding='max_length', truncation=True,
                                     max_length=128, return_tensors="pt")
        
        return {
            'pixel_values': pixel_values,
            'input_ids': text_inputs['input_ids'].squeeze(),
            'attention_mask': text_inputs['attention_mask'].squeeze(),
            'species': item['species'],
            'species_idx': self.species_to_idx[item['species']]
        }

# Model components (simplified)
class VisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vit = ViTModel.from_pretrained(config.vit_model_name)
        for param in self.vit.parameters():
            param.requires_grad = False
            
        self.proj = nn.Sequential(
            nn.Linear(config.vision_embed_dim, config.mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden_dim, config.shared_embed_dim)
        )
        
    def forward(self, pixel_values):
        with torch.no_grad():
            outputs = self.vit(pixel_values=pixel_values)
            vision_embeds = outputs.last_hidden_state.mean(dim=1)
        return self.proj(vision_embeds)

class LanguageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = AutoModel.from_pretrained(config.deepseek_model_name)
        for param in self.model.parameters():
            param.requires_grad = False
            
        hidden_size = self.model.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, config.mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden_dim, config.shared_embed_dim)
        )
        
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            lang_embeds = outputs.last_hidden_state.mean(dim=1)
        return self.proj(lang_embeds)

class CrossModalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_encoder = VisionEncoder(config)
        self.language_encoder = LanguageEncoder(config)
        
        # Transformer
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
        
        # Decoders
        self.vision_decoder = nn.Linear(config.shared_embed_dim, config.shared_embed_dim)
        self.language_decoder = nn.Linear(config.shared_embed_dim, config.shared_embed_dim)
        
    def forward(self, pixel_values, input_ids, attention_mask):
        # Encode
        vision_embeds = self.vision_encoder(pixel_values)
        language_embeds = self.language_encoder(input_ids, attention_mask)
        
        # Mask
        batch_size = vision_embeds.shape[0]
        device = vision_embeds.device
        
        vision_mask = torch.rand(batch_size, 1, device=device) < self.config.mask_ratio
        language_mask = torch.rand(batch_size, 1, device=device) < self.config.mask_ratio
        
        masked_vision = vision_embeds * (~vision_mask).float()
        masked_language = language_embeds * (~language_mask).float()
        
        # Transform
        combined = torch.stack([masked_vision, masked_language], dim=1)
        transformed = self.transformer(combined)
        
        # Decode
        vision_recon = self.vision_decoder(transformed[:, 0])
        language_recon = self.language_decoder(transformed[:, 1])
        
        return {
            'vision_embeds': vision_embeds,
            'language_embeds': language_embeds,
            'vision_recon': vision_recon,
            'language_recon': language_recon,
            'vision_mask': vision_mask,
            'language_mask': language_mask
        }

def train():
    config = Config()
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = LocalPlantDataset(config, 'train')
    val_dataset = LocalPlantDataset(config, 'val')
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Create model
    print("Creating model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CrossModalModel(config).to(device)
    
    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}, Trainable: {trainable:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    print("Starting training...")
    train_losses = []
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}')
        for batch in pbar:
            # Move to device
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward
            outputs = model(pixel_values, input_ids, attention_mask)
            
            # Loss (only on masked parts)
            vision_loss = F.mse_loss(
                outputs['vision_recon'][outputs['vision_mask'].squeeze()],
                outputs['vision_embeds'][outputs['vision_mask'].squeeze()].detach()
            ) if outputs['vision_mask'].any() else 0
            
            language_loss = F.mse_loss(
                outputs['language_recon'][outputs['language_mask'].squeeze()],
                outputs['language_embeds'][outputs['language_mask'].squeeze()].detach()
            ) if outputs['language_mask'].any() else 0
            
            loss = vision_loss + language_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
        # Evaluate
        if (epoch + 1) % config.eval_interval == 0:
            model.eval()
            print("Evaluating...")
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(config.checkpoint_dir, f'checkpoint_{epoch+1}.pth'))
    
    # Save final model
    torch.save(model.state_dict(), 'final_model.pth')
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('training_loss.png')
    
    print("Training complete!")

if __name__ == "__main__":
    train()
