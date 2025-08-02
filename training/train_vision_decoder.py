#!/usr/bin/env python3
"""
Train the vision decoder for language → vision reconstruction

This script loads the pretrained model and trains only the vision decoder
to reconstruct vision embeddings from language embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from deepearth_multimodal_training import DeepEarthDataset
from bidirectional_reconstruction import BidirectionalMultimodalModel, load_pretrained_and_extend


def train_vision_decoder(
    model_path='models/multimodal_model_best.pth',
    data_dir='../dashboard/huggingface_dataset/hf_download/',
    epochs=20,
    batch_size=16,
    learning_rate=1e-3,
    device='cuda'
):
    """Train only the vision decoder while keeping encoders frozen."""
    
    # Load model with pretrained encoders
    print("Loading pretrained model...")
    model = load_pretrained_and_extend()
    model.to(device)
    
    # Freeze encoders and language decoder
    for param in model.vision_mlp.parameters():
        param.requires_grad = False
    for param in model.language_mlp.parameters():
        param.requires_grad = False
    for param in model.language_decoder.parameters():
        param.requires_grad = False
    
    print("Frozen encoders and language decoder")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create dataset
    print("\nLoading dataset...")
    dataset = DeepEarthDataset(
        data_dir=data_dir,
        max_samples=1000,  # Use more data for training
        cache_embeddings=True
    )
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Optimizer - only for vision decoder
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    
    print("\nStarting training...")
    for epoch in range(epochs):
        # Training
        model.train()
        # Keep encoders in eval mode
        model.vision_mlp.eval()
        model.language_mlp.eval()
        
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            vision_emb = batch['vision_embedding'].to(device)
            language_emb = batch['language_embedding'].to(device)
            
            # Forward pass - reconstruct vision from language
            outputs = model(vision_emb, language_emb, mask_modality='vision', mask_prob=1.0)
            vision_recon = outputs['vision_reconstructed']
            
            # Loss - MSE between original and reconstructed vision
            loss = F.mse_loss(vision_recon, vision_emb)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_similarities = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                vision_emb = batch['vision_embedding'].to(device)
                language_emb = batch['language_embedding'].to(device)
                
                outputs = model(vision_emb, language_emb, mask_modality='vision', mask_prob=1.0)
                vision_recon = outputs['vision_reconstructed']
                
                loss = F.mse_loss(vision_recon, vision_emb)
                val_loss += loss.item()
                
                # Calculate similarity after pooling
                vision_recon_pooled = vision_recon.mean(dim=(1,2,3))
                vision_orig_pooled = vision_emb.mean(dim=(1,2,3))
                similarities = F.cosine_similarity(vision_recon_pooled, vision_orig_pooled, dim=1)
                val_similarities.extend(similarities.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        avg_similarity = np.mean(val_similarities)
        
        print(f"\nEpoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, "
              f"Val Loss = {avg_val_loss:.4f}, Val Similarity = {avg_similarity:.3f}")
    
    # Save the complete model
    save_path = 'models/bidirectional_model.pth'
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved model to {save_path}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Vision Reconstruction Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(val_similarities, bins=50)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    plt.title(f'Final Val Similarities (mean={avg_similarity:.3f})')
    
    plt.tight_layout()
    plt.savefig('vision_decoder_training.png')
    print("Saved training plot to vision_decoder_training.png")
    
    return model


def test_trained_model(model_path='models/bidirectional_model.pth'):
    """Test the trained bidirectional model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model = BidirectionalMultimodalModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load test data
    dataset = DeepEarthDataset(
        data_dir='../dashboard/huggingface_dataset/hf_download/',
        max_samples=10,
        cache_embeddings=False
    )
    
    print("\n" + "="*60)
    print("Testing Bidirectional Reconstruction")
    print("="*60)
    
    with torch.no_grad():
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            vision_emb = sample['vision_embedding'].unsqueeze(0).to(device)
            language_emb = sample['language_embedding'].unsqueeze(0).to(device)
            
            print(f"\nSpecies: {sample['taxon_name']}")
            
            # Test both directions
            outputs = model(vision_emb, language_emb, mask_modality='both')
            
            # Language reconstruction (vision → language)
            lang_recon = outputs['language_reconstructed']
            lang_sim = F.cosine_similarity(lang_recon, language_emb)
            print(f"\nVision → Language:")
            print(f"  Similarity: {lang_sim.item():.3f}")
            
            # Vision reconstruction (language → vision)
            vision_recon = outputs['vision_reconstructed']
            vision_recon_pooled = vision_recon.mean(dim=(1,2,3))
            vision_orig_pooled = vision_emb.mean(dim=(1,2,3))
            vision_sim = F.cosine_similarity(vision_recon_pooled, vision_orig_pooled)
            print(f"\nLanguage → Vision:")
            print(f"  Similarity: {vision_sim.item():.3f}")
            
            # Cross-modal alignment
            vision_universal = outputs['vision_universal']
            language_universal = outputs['language_universal']
            universal_sim = F.cosine_similarity(vision_universal, language_universal)
            print(f"\nUniversal space alignment: {universal_sim.item():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train vision decoder for language→vision reconstruction')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--test-only', action='store_true', help='Only test the model')
    
    args = parser.parse_args()
    
    if args.test_only:
        test_trained_model()
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Training on {device}")
        
        model = train_vision_decoder(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device
        )
        
        # Test after training
        print("\n" + "="*60)
        print("Testing after training...")
        test_trained_model()
