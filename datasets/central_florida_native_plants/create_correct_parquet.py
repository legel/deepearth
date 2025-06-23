#!/usr/bin/env python3
"""
Convert embeddings to Parquet format with correct species token identification
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
import re

def identify_species_tokens(tokens_df, species_name):
    """
    Identify which tokens correspond to the species name by matching token strings
    """
    # Clean the species name (remove extra spaces)
    species_name = species_name.strip()
    
    # Initialize all tokens as not species tokens
    is_species_token = [False] * len(tokens_df)
    
    # Try to find the species name in the concatenated tokens
    # We'll look for the sequence of tokens that forms the species name
    
    # Build a string from all tokens to find the species name
    all_tokens_str = ''.join(tokens_df['token_str'].fillna('').astype(str))
    
    # Find where the species name appears
    # The prompt format is "Ecophysiology of {species_name}:"
    # So we look for the species name after "of" and before ":"
    
    # Look for patterns like "of Species name:" or "of Species name :"
    # First, let's find tokens between "of" and ":"
    
    found_species = False
    for i in range(len(tokens_df)):
        token_str = str(tokens_df.iloc[i]['token_str'])
        
        # Look for " of" token (usually position 8)
        if token_str.strip() == 'of' or token_str == ' of':
            # Start looking for species tokens after this
            # Build the string from subsequent tokens until we find ":"
            remaining_tokens = []
            for j in range(i+1, len(tokens_df)):
                next_token = str(tokens_df.iloc[j]['token_str'])
                if ':' in next_token:
                    # Found the end marker
                    # Check if we can match the species name
                    candidate = ''.join(remaining_tokens).strip()
                    
                    # Try to match with species name (case insensitive)
                    if candidate.lower().replace(' ', '') == species_name.lower().replace(' ', ''):
                        # Mark these tokens as species tokens
                        for k in range(i+1, j):
                            is_species_token[k] = True
                        found_species = True
                    break
                else:
                    remaining_tokens.append(next_token)
            
            if found_species:
                break
    
    # If we didn't find it with the above method, try a more flexible approach
    if not found_species:
        # Split species name into words
        species_words = species_name.split()
        
        # Try to find consecutive tokens that match the species words
        for start_idx in range(len(tokens_df)):
            matched_words = 0
            current_idx = start_idx
            
            for word in species_words:
                if current_idx >= len(tokens_df):
                    break
                    
                # Build word from tokens
                accumulated = ""
                word_start_idx = current_idx
                
                while current_idx < len(tokens_df) and len(accumulated) < len(word):
                    token_str = str(tokens_df.iloc[current_idx]['token_str'])
                    accumulated += token_str
                    current_idx += 1
                    
                    if accumulated.strip().lower() == word.lower():
                        # Found a matching word
                        matched_words += 1
                        # Mark these tokens as species tokens
                        for k in range(word_start_idx, current_idx):
                            is_species_token[k] = True
                        break
                    elif not word.lower().startswith(accumulated.lower().strip()):
                        # This path won't lead to a match
                        break
            
            if matched_words == len(species_words):
                found_species = True
                break
    
    return is_species_token

def create_correct_parquet_dataset():
    """Create a Parquet file with correct species token identification"""
    embeddings_dir = Path("embeddings")
    tokens_dir = Path("tokens")
    
    if not embeddings_dir.exists():
        print("Error: embeddings directory not found. Please run download_dataset.sh first.")
        return
    
    # First, let's analyze a few examples to understand the pattern
    print("Analyzing token patterns...")
    
    for i, pt_file in enumerate(sorted(embeddings_dir.glob("*.pt"))[:3]):
        taxon_id = pt_file.stem
        data = torch.load(pt_file)
        token_file = tokens_dir / f"{taxon_id}.csv"
        
        if token_file.exists():
            tokens_df = pd.read_csv(token_file)
            species_name = data.get('species_name', '')
            
            print(f"\nExample {i+1}: {species_name} (taxon_id: {taxon_id})")
            print("Tokens:")
            for idx, row in tokens_df.iterrows():
                if idx < 20:  # Show first 20 tokens
                    print(f"  {idx}: '{row['token_str']}' (id: {row['token_id']})")
            
            # Identify species tokens
            is_species = identify_species_tokens(tokens_df, species_name)
            species_token_indices = [i for i, v in enumerate(is_species) if v]
            print(f"Species tokens at positions: {species_token_indices}")
            if species_token_indices:
                species_tokens = [tokens_df.iloc[i]['token_str'] for i in species_token_indices]
                print(f"Species tokens: {species_tokens}")
                print(f"Reconstructed: '{''.join(species_tokens)}'")
    
    # Now process all data
    print("\n" + "="*60)
    print("Processing all species...")
    
    records = []
    species_mean_embeddings = {}  # Store mean embeddings calculated from species tokens only
    
    for pt_file in sorted(embeddings_dir.glob("*.pt")):
        try:
            taxon_id = pt_file.stem
            
            # Load embedding data
            data = torch.load(pt_file)
            species_name = data.get('species_name', '')
            
            # Load corresponding token data
            token_file = tokens_dir / f"{taxon_id}.csv"
            if token_file.exists():
                tokens_df = pd.read_csv(token_file)
            else:
                print(f"Warning: No token file found for {taxon_id}")
                continue
            
            # Get token embeddings
            token_embeddings = data['token_embeddings']  # Shape: [num_tokens, 7168]
            
            # Identify species tokens
            is_species_token_list = identify_species_tokens(tokens_df, species_name)
            
            # Calculate mean embedding from species tokens only
            species_token_indices = [i for i, v in enumerate(is_species_token_list) if v]
            if species_token_indices:
                species_token_embeddings = token_embeddings[species_token_indices]
                species_mean_embedding = species_token_embeddings.mean(dim=0).numpy().tolist()
            else:
                # Fallback to overall mean if no species tokens identified
                print(f"Warning: No species tokens identified for {species_name}")
                species_mean_embedding = data['mean_embedding'].numpy().tolist()
            
            species_mean_embeddings[taxon_id] = species_mean_embedding
            
            # Create a record for each token
            for idx, token_row in tokens_df.iterrows():
                if idx < len(token_embeddings):
                    token_embedding = token_embeddings[idx].numpy().tolist()
                else:
                    print(f"Warning: Token index {idx} out of bounds for {taxon_id}")
                    continue
                
                record = {
                    'taxon_id': taxon_id,
                    'species_name': species_name,
                    'timestamp': data.get('timestamp', ''),
                    'token_position': int(token_row['position']),
                    'token_id': int(token_row['token_id']),
                    'token_str': str(token_row['token_str']),
                    'is_species_token': is_species_token_list[idx],
                    'token_embedding': token_embedding,  # Full 7168 dimensions
                    'species_mean_embedding': species_mean_embedding,  # Mean of species tokens only
                    'all_tokens_mean_embedding': data['mean_embedding'].numpy().tolist(),  # Original mean of all tokens
                    'num_tokens': int(data.get('num_tokens', len(tokens_df))),
                    'num_species_tokens': len(species_token_indices),
                }
                
                records.append(record)
            
        except Exception as e:
            print(f"Error loading {pt_file}: {e}")
            continue
    
    print(f"\nLoaded {len(records)} token records from {len(set(r['taxon_id'] for r in records))} species")
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Create directory for data files
    Path('data').mkdir(exist_ok=True)
    
    # Save the full dataset with all embeddings
    chunk_size = 1000
    num_chunks = (len(df) + chunk_size - 1) // chunk_size
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))
        chunk = df.iloc[start_idx:end_idx]
        
        filename = f'data/train-{i:05d}-of-{num_chunks:05d}.parquet'
        chunk.to_parquet(filename, index=False)
        print(f"Saved {filename} with {len(chunk)} records")
    
    # Create species summary with species-token-based mean embeddings
    species_records = []
    for taxon_id in df['taxon_id'].unique():
        species_data = df[df['taxon_id'] == taxon_id].iloc[0]
        species_records.append({
            'taxon_id': taxon_id,
            'species_name': species_data['species_name'],
            'num_tokens': species_data['num_tokens'],
            'num_species_tokens': species_data['num_species_tokens'],
            'timestamp': species_data['timestamp'],
            'species_mean_embedding': species_data['species_mean_embedding'],
            'all_tokens_mean_embedding': species_data['all_tokens_mean_embedding'],
        })
    
    species_df = pd.DataFrame(species_records)
    species_df.to_parquet('data/species_summary.parquet', index=False)
    print(f"Saved species summary with {len(species_df)} species")
    
    # Create metadata
    metadata = {
        "dataset_name": "Central Florida Native Plants Embeddings (Corrected)",
        "num_examples": len(df),
        "num_species": len(df['taxon_id'].unique()),
        "embedding_dimension": 7168,
        "model": "DeepSeek-V3-0324-UD-Q4_K_XL",
        "features": {
            "taxon_id": "GBIF taxonomic identifier",
            "species_name": "Scientific name of the plant species",
            "timestamp": "When the embedding was created",
            "token_position": "Position of token in sequence",
            "token_id": "Token ID in model vocabulary",
            "token_str": "String representation of token",
            "is_species_token": "Whether this token is part of the species name (correctly identified)",
            "token_embedding": "7168-dimensional embedding vector for this specific token",
            "species_mean_embedding": "7168-dimensional mean embedding of species name tokens only",
            "all_tokens_mean_embedding": "7168-dimensional mean embedding across all tokens (including prompt)",
            "num_tokens": "Total number of tokens for this species",
            "num_species_tokens": "Number of tokens that are part of the species name"
        }
    }
    
    with open('data/dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\nDataset conversion complete!")
    print(f"Total records: {len(df)}")
    print(f"Files created: {num_chunks} parquet files + species summary")

if __name__ == "__main__":
    create_correct_parquet_dataset()