#!/usr/bin/env python3
"""
Bidirectional Cross-Modal Reconstruction Model

This extends the original model to support both:
1. Vision → Language reconstruction (original)
2. Language → Vision reconstruction (new)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Import the original components
from deepearth_multimodal_training import (
    VisionMLP, LanguageMLP, LanguageDecoder,
    DeepEarthDataset
)


class VisionDecoder(nn.Module):
    """MLP decoder to reconstruct vision embeddings from universal space."""
    def __init__(self, universal_dim=2048, hidden_dim=512, vision_dim=1408, dropout=0.1):
        super().__init__()
        
        # Decode to spatial features
        self.decoder = nn.Sequential(
            nn.Linear(universal_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 8 * 24 * 24 * vision_dim)  # Flattened vision embedding
        )
        
        self.vision_dim = vision_dim
        
    def forward(self, x):
        # Decode to flattened features
        decoded = self.decoder(x)
        # Reshape to vision embedding format
        batch_size = decoded.shape[0]
        decoded = decoded.view(batch_size, 8, 24, 24, self.vision_dim)
        return decoded


class BidirectionalMultimodalModel(nn.Module):
    """Complete bidirectional multimodal system for masked reconstruction."""
    def __init__(self, vision_dim=1408, language_dim=7168, 
                 hidden_dim=256, universal_dim=2048, dropout=0.1):
        super().__init__()
        
        # Encoders (shared with original model)
        self.vision_mlp = VisionMLP(vision_dim, hidden_dim, universal_dim, dropout)
        self.language_mlp = LanguageMLP(language_dim, hidden_dim, universal_dim, dropout)
        
        # Decoders
        self.language_decoder = LanguageDecoder(universal_dim, hidden_dim, language_dim, dropout)
        self.vision_decoder = VisionDecoder(universal_dim, hidden_dim, vision_dim, dropout)
        
    def forward(self, vision_emb, language_emb, mask_modality='language', mask_prob=0.5):
        batch_size = vision_emb.shape[0]
        device = vision_emb.device
        
        # Encode both modalities to universal space
        vision_universal = self.vision_mlp(vision_emb)
        language_universal = self.language_mlp(language_emb)
        
        # Create mask
        mask = torch.rand(batch_size, device=device) < mask_prob
        
        if mask_modality == 'language':
            # Reconstruct language from vision (original)
            language_reconstructed = self.language_decoder(vision_universal)
            vision_reconstructed = None
        elif mask_modality == 'vision':
            # Reconstruct vision from language (new!)
            vision_reconstructed = self.vision_decoder(language_universal)
            language_reconstructed = None
        else:  # 'both'
            # Reconstruct both directions
            language_reconstructed = self.language_decoder(vision_universal)
            vision_reconstructed = self.vision_decoder(language_universal)
        
        return {
            'vision_universal': vision_universal,
            'language_universal': language_universal,
            'language_reconstructed': language_reconstructed,
            'vision_reconstructed': vision_reconstructed,
            'mask': mask
        }


def test_bidirectional_reconstruction():
    """Test both reconstruction directions."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing bidirectional reconstruction on {device}")
    
    # Create model
    model = BidirectionalMultimodalModel().to(device)
    model.eval()
    
    # Load a sample
    dataset = DeepEarthDataset(
        data_dir='../dashboard/huggingface_dataset/hf_download/',
        max_samples=10,
        cache_embeddings=False
    )
    
    if len(dataset) == 0:
        print("No data available!")
        return
    
    # Get first sample
    sample = dataset[0]
    vision_emb = sample['vision_embedding'].unsqueeze(0).to(device)
    language_emb = sample['language_embedding'].unsqueeze(0).to(device)
    
    print(f"\nTesting with species: {sample['taxon_name']}")
    print(f"Original shapes:")
    print(f"  Vision: {vision_emb.shape}")
    print(f"  Language: {language_emb.shape}")
    
    with torch.no_grad():
        # Test language reconstruction (vision → language)
        print("\n1. Vision → Language Reconstruction:")
        outputs = model(vision_emb, language_emb, mask_modality='language', mask_prob=1.0)
        
        if outputs['language_reconstructed'] is not None:
            lang_recon = outputs['language_reconstructed']
            lang_sim = F.cosine_similarity(lang_recon, language_emb)
            lang_mse = F.mse_loss(lang_recon, language_emb)
            print(f"  Reconstructed shape: {lang_recon.shape}")
            print(f"  Cosine similarity: {lang_sim.item():.3f}")
            print(f"  MSE loss: {lang_mse.item():.4f}")
        
        # Test vision reconstruction (language → vision)
        print("\n2. Language → Vision Reconstruction:")
        outputs = model(vision_emb, language_emb, mask_modality='vision', mask_prob=1.0)
        
        if outputs['vision_reconstructed'] is not None:
            vision_recon = outputs['vision_reconstructed']
            # For vision, compare after pooling since it's high-dimensional
            vision_recon_pooled = vision_recon.mean(dim=(1,2,3))
            vision_orig_pooled = vision_emb.mean(dim=(1,2,3))
            vision_sim = F.cosine_similarity(vision_recon_pooled, vision_orig_pooled)
            vision_mse = F.mse_loss(vision_recon, vision_emb)
            print(f"  Reconstructed shape: {vision_recon.shape}")
            print(f"  Cosine similarity (pooled): {vision_sim.item():.3f}")
            print(f"  MSE loss: {vision_mse.item():.4f}")
        
        # Test bidirectional
        print("\n3. Bidirectional Reconstruction:")
        outputs = model(vision_emb, language_emb, mask_modality='both', mask_prob=1.0)
        print(f"  Both reconstructions generated successfully!")


def load_pretrained_and_extend():
    """Load the pretrained model and extend it with vision decoder."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create the bidirectional model
    model = BidirectionalMultimodalModel().to(device)
    
    # Load pretrained weights for the parts that exist
    pretrained_path = 'models/multimodal_model_best.pth'
    if Path(pretrained_path).exists():
        print(f"Loading pretrained weights from {pretrained_path}")
        pretrained_state = torch.load(pretrained_path, map_location=device)
        
        # Load only the matching keys
        model_state = model.state_dict()
        matched_state = {}
        
        for key, value in pretrained_state.items():
            if key in model_state and model_state[key].shape == value.shape:
                matched_state[key] = value
                print(f"  Loaded: {key}")
            else:
                print(f"  Skipped: {key}")
        
        model.load_state_dict(matched_state, strict=False)
        print(f"\nLoaded {len(matched_state)}/{len(model_state)} parameters")
        
        # The vision decoder will be randomly initialized
        print("\nVision decoder initialized randomly (needs training)")
    
    return model


def demonstrate_language_to_vision():
    """Demonstrate language → vision reconstruction with the pretrained model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load the extended model
    model = load_pretrained_and_extend()
    model.eval()
    
    # Load dataset
    dataset = DeepEarthDataset(
        data_dir='../dashboard/huggingface_dataset/hf_download/',
        max_samples=5,
        cache_embeddings=False
    )
    
    print(f"\n{'='*60}")
    print("Language → Vision Reconstruction Demo")
    print(f"{'='*60}")
    
    with torch.no_grad():
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            vision_emb = sample['vision_embedding'].unsqueeze(0).to(device)
            language_emb = sample['language_embedding'].unsqueeze(0).to(device)
            
            print(f"\nSample {i+1}: {sample['taxon_name']}")
            
            # Get universal embeddings
            vision_universal = model.vision_mlp(vision_emb)
            language_universal = model.language_mlp(language_emb)
            
            # Check if vision and language map to similar universal space
            universal_sim = F.cosine_similarity(vision_universal, language_universal)
            print(f"  Universal space similarity: {universal_sim.item():.3f}")
            
            # Reconstruct vision from language
            vision_recon = model.vision_decoder(language_universal)
            
            # Compare shapes
            print(f"  Original vision shape: {vision_emb.shape}")
            print(f"  Reconstructed vision shape: {vision_recon.shape}")
            
            # Since vision decoder is untrained, similarity will be random
            vision_recon_pooled = vision_recon.mean(dim=(1,2,3))
            vision_orig_pooled = vision_emb.mean(dim=(1,2,3))
            vision_sim = F.cosine_similarity(vision_recon_pooled, vision_orig_pooled)
            print(f"  Reconstruction similarity: {vision_sim.item():.3f} (random - needs training)")


if __name__ == "__main__":
    print("🔄 Bidirectional Cross-Modal Reconstruction\n")
    
    # Test the bidirectional model architecture
    print("Step 1: Testing model architecture...")
    test_bidirectional_reconstruction()
    
    # Demonstrate with pretrained weights
    print("\n\nStep 2: Loading pretrained model and extending...")
    demonstrate_language_to_vision()
    
    print("\n\n💡 To train the vision decoder:")
    print("1. Create a training script with vision reconstruction loss")
    print("2. Fine-tune from the pretrained model")
    print("3. Use the same dataset but with vision masking")
