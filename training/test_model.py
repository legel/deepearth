#!/usr/bin/env python3
"""
Simple test script to verify the model works
"""

import torch
import numpy as np
from pathlib import Path
from deepearth_multimodal_training import MultimodalMaskingModel, DeepEarthDataset

def test_model():
    # Load the model
    print("Loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MultimodalMaskingModel()
    model.load_state_dict(torch.load('models/multimodal_model_best.pth', map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    
    # Create a small test dataset
    print("\nLoading test data...")
    dataset = DeepEarthDataset(
        data_dir='../dashboard/huggingface_dataset/hf_download/',
        max_samples=10,
        split='train',
        cache_embeddings=False  # Don't cache for quick test
    )
    
    print(f"Test dataset size: {len(dataset)}")
    
    # Test on a few samples
    print("\nTesting model on samples...")
    
    with torch.no_grad():
        for i in range(min(3, len(dataset))):
            # Get a sample
            sample = dataset[i]
            
            # Prepare inputs
            vision_emb = sample['vision_embedding'].unsqueeze(0).to(device)
            language_emb = sample['language_embedding'].unsqueeze(0).to(device)
            
            print(f"\nSample {i+1}:")
            print(f"  Species: {sample['taxon_name']}")
            print(f"  GBIF ID: {sample['gbif_id']}")
            print(f"  Location: ({sample['latitude']:.2f}, {sample['longitude']:.2f})")
            
            # Forward pass
            outputs = model(vision_emb, language_emb, mask_language=True, mask_prob=1.0)
            
            # Extract embeddings
            vision_universal = outputs['vision_universal']
            language_universal = outputs['language_universal']
            language_reconstructed = outputs['language_reconstructed']
            
            # Calculate similarities
            vision_lang_sim = torch.cosine_similarity(vision_universal, language_universal)
            recon_sim = torch.cosine_similarity(language_reconstructed, language_emb)
            recon_loss = torch.nn.functional.mse_loss(language_reconstructed, language_emb)
            
            print(f"  Vision-Language Universal Similarity: {vision_lang_sim.item():.3f}")
            print(f"  Reconstruction Similarity: {recon_sim.item():.3f}")
            print(f"  Reconstruction MSE Loss: {recon_loss.item():.4f}")
            
            # Show embedding dimensions
            print(f"  Embedding shapes:")
            print(f"    Vision Universal: {vision_universal.shape}")
            print(f"    Language Universal: {language_universal.shape}")
            print(f"    Language Reconstructed: {language_reconstructed.shape}")
    
    print("\n✅ Model test complete!")
    
    # Quick stats on the full dataset
    print("\n📊 Dataset Statistics:")
    taxon_counts = {}
    for i in range(len(dataset)):
        taxon = dataset.observations_df.iloc[i]['taxon_name']
        taxon_counts[taxon] = taxon_counts.get(taxon, 0) + 1
    
    print(f"Total samples: {len(dataset)}")
    print(f"Unique species: {len(taxon_counts)}")
    print("\nSpecies in this sample:")
    for taxon, count in sorted(taxon_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {taxon}: {count} samples")

if __name__ == "__main__":
    test_model()
