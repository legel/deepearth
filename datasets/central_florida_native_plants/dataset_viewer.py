#!/usr/bin/env python3
"""
Dataset viewer for Central Florida Native Plants embeddings
"""

import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path

def load_species_list():
    """Load list of species from embeddings directory"""
    embeddings_dir = Path(__file__).parent / "embeddings"
    if not embeddings_dir.exists():
        print("Error: embeddings directory not found. Please run download_dataset.sh first.")
        return []
    
    species_ids = []
    for file in sorted(embeddings_dir.glob("*.pt")):
        species_id = file.stem
        species_ids.append(species_id)
    
    return species_ids

def load_embedding(species_id):
    """Load embedding for a specific species"""
    embedding_path = Path(__file__).parent / "embeddings" / f"{species_id}.pt"
    if not embedding_path.exists():
        return None
    return torch.load(embedding_path)

def load_tokens(species_id):
    """Load token mapping for a specific species"""
    token_path = Path(__file__).parent / "tokens" / f"{species_id}.csv"
    if not token_path.exists():
        return None
    return pd.read_csv(token_path)

def analyze_dataset():
    """Analyze the dataset and print summary statistics"""
    species_ids = load_species_list()
    
    print(f"Total species: {len(species_ids)}")
    print("\nFirst 10 species IDs:")
    for i, species_id in enumerate(species_ids[:10]):
        print(f"  {i+1}. {species_id}")
    
    if species_ids:
        # Analyze first species as example
        example_id = species_ids[0]
        data = load_embedding(example_id)
        tokens = load_tokens(example_id)
        
        print(f"\nExample species: {example_id}")
        print(f"Species name: {data['species_name']}")
        print(f"Taxon ID: {data['taxon_id']}")
        print(f"Number of tokens: {data['num_tokens']}")
        
        # Mean embedding info
        mean_emb = data['mean_embedding']
        print(f"\nMean embedding:")
        print(f"  Shape: {mean_emb.shape}")
        print(f"  Dtype: {mean_emb.dtype}")
        print(f"  Min/Max: {mean_emb.min():.4f} / {mean_emb.max():.4f}")
        print(f"  Mean/Std: {mean_emb.mean():.4f} / {mean_emb.std():.4f}")
        
        # Show first 10 and last 10 values of mean embedding
        print(f"\n  First 10 values: {mean_emb[:10].numpy()}")
        print(f"  Last 10 values: {mean_emb[-10:].numpy()}")
        
        # Token embeddings info
        token_embs = data['token_embeddings']
        print(f"\nToken embeddings:")
        print(f"  Shape: {token_embs.shape}")
        print(f"  Per-token dimension: {token_embs.shape[1]}")
        
        # Show embedding values for first token
        print(f"\n  First token embedding (first 10 dims): {token_embs[0, :10].numpy()}")
        print(f"  First token embedding (last 10 dims): {token_embs[0, -10:].numpy()}")
        
        # Show embedding statistics
        print(f"\n  Token embeddings statistics:")
        print(f"    Min/Max across all: {token_embs.min():.4f} / {token_embs.max():.4f}")
        print(f"    Mean/Std across all: {token_embs.mean():.4f} / {token_embs.std():.4f}")
        
        if tokens is not None:
            print(f"\nToken information:")
            print(f"Number of tokens in CSV: {len(tokens)}")
            print("\nFirst 5 tokens:")
            print(tokens.head())
            
            # Reconstruct text
            text = ''.join(tokens['token'].tolist())
            print(f"\nReconstructed text: {text}")

def compute_similarity_matrix(n_samples=10):
    """Compute pairwise cosine similarities between species using mean embeddings"""
    species_ids = load_species_list()[:n_samples]
    
    embeddings = []
    species_names = []
    for species_id in species_ids:
        data = load_embedding(species_id)
        if data is not None:
            embeddings.append(data['mean_embedding'].numpy())
            species_names.append(data['species_name'])
    
    if len(embeddings) < 2:
        print("Not enough embeddings to compute similarities")
        return
    
    # Stack embeddings
    embeddings = np.stack(embeddings)
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    
    # Compute cosine similarity matrix
    similarity_matrix = normalized @ normalized.T
    
    print(f"\nCosine similarity matrix ({n_samples}x{n_samples}):")
    print("Species:", species_names)
    print("\nSimilarity matrix (first 5x5):")
    print(similarity_matrix[:5, :5])
    
    # Find most similar pairs
    mask = np.triu(np.ones_like(similarity_matrix), k=1).astype(bool)
    similarities = similarity_matrix[mask]
    indices = np.argwhere(mask)
    
    sorted_idx = np.argsort(similarities)[::-1]
    print(f"\nMost similar pairs:")
    for i in range(min(5, len(sorted_idx))):
        idx = sorted_idx[i]
        i1, i2 = indices[idx]
        sim = similarities[idx]
        print(f"  {species_names[i1]} - {species_names[i2]}: {sim:.4f}")

def explore_species(species_id=None):
    """Explore a specific species' embeddings in detail"""
    species_ids = load_species_list()
    
    if species_id is None:
        # Pick a random species
        import random
        species_id = random.choice(species_ids)
    
    if species_id not in species_ids:
        print(f"Species ID {species_id} not found in dataset")
        return
    
    data = load_embedding(species_id)
    tokens = load_tokens(species_id)
    
    print(f"\nDetailed exploration of species: {species_id}")
    print("=" * 60)
    print(f"Species name: {data['species_name']}")
    print(f"Taxon ID: {data['taxon_id']}")
    print(f"Timestamp: {data.get('timestamp', 'N/A')}")
    
    # Mean embedding analysis
    mean_emb = data['mean_embedding']
    print(f"\nMean Embedding Analysis:")
    print(f"  Dimension: {mean_emb.shape[0]}")
    print(f"  Norm (L2): {torch.norm(mean_emb).item():.4f}")
    print(f"  Top 5 positive values: {torch.topk(mean_emb, 5).values.numpy()}")
    print(f"  Top 5 negative values: {torch.topk(-mean_emb, 5).values.numpy() * -1}")
    
    # Embedding statistics from stored data
    if 'embedding_stats' in data:
        stats = data['embedding_stats']
        print(f"\nStored embedding statistics:")
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Token-level analysis
    token_embs = data['token_embeddings']
    print(f"\nToken-level Analysis:")
    print(f"  Number of tokens: {token_embs.shape[0]}")
    print(f"  Embedding dimension per token: {token_embs.shape[1]}")
    
    if tokens is not None and len(tokens) > 0:
        print(f"\nToken Details:")
        for idx, row in tokens.iterrows():
            if idx < 5:  # Show first 5 tokens
                token_emb = token_embs[idx]
                print(f"  Token {idx}: '{row['token']}' (ID: {row['token_id']})")
                print(f"    Norm: {torch.norm(token_emb).item():.4f}")
                print(f"    Mean: {token_emb.mean().item():.4f}, Std: {token_emb.std().item():.4f}")
                print(f"    First 5 dims: {token_emb[:5].numpy()}")

    # Variance analysis across dimensions
    print(f"\nDimensional Variance Analysis:")
    dim_vars = mean_emb.var()
    print(f"  Overall variance: {dim_vars:.6f}")
    
    # Find most variable dimensions
    token_vars = token_embs.var(dim=0)  # Variance across tokens for each dimension
    top_var_dims = torch.topk(token_vars, 10).indices
    print(f"  Top 10 most variable dimensions across tokens: {top_var_dims.numpy()}")
    
    return data, tokens

if __name__ == "__main__":
    print("Central Florida Native Plants Dataset Viewer")
    print("=" * 50)
    
    analyze_dataset()
    print("\n" + "=" * 50)
    compute_similarity_matrix(n_samples=10)
    print("\n" + "=" * 50)
    explore_species()  # Explore a random species in detail