#!/usr/bin/env python3
"""
Convert embeddings to comprehensive Parquet format with full dimensionality and token-level data
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json

def create_full_parquet_dataset():
    """Create a Parquet file with complete embeddings data including all tokens"""
    embeddings_dir = Path("embeddings")
    tokens_dir = Path("tokens")
    
    if not embeddings_dir.exists():
        print("Error: embeddings directory not found. Please run download_dataset.sh first.")
        return
    
    # Collect all data
    records = []
    
    for pt_file in sorted(embeddings_dir.glob("*.pt")):
        try:
            taxon_id = pt_file.stem
            
            # Load embedding data
            data = torch.load(pt_file)
            
            # Load corresponding token data
            token_file = tokens_dir / f"{taxon_id}.csv"
            if token_file.exists():
                tokens_df = pd.read_csv(token_file)
            else:
                print(f"Warning: No token file found for {taxon_id}")
                continue
            
            # Get token embeddings
            token_embeddings = data['token_embeddings']  # Shape: [num_tokens, 7168]
            
            # Create a record for each token
            for idx, token_row in tokens_df.iterrows():
                # Get the embedding for this specific token
                if idx < len(token_embeddings):
                    token_embedding = token_embeddings[idx].numpy().tolist()
                else:
                    print(f"Warning: Token index {idx} out of bounds for {taxon_id}")
                    continue
                
                record = {
                    'taxon_id': taxon_id,
                    'species_name': data.get('species_name', token_row.get('species_name', 'Unknown')),
                    'timestamp': data.get('timestamp', ''),
                    'token_position': int(token_row['position']),
                    'token_id': int(token_row['token_id']),
                    'token_str': str(token_row['token_str']),
                    'is_species_token': bool(token_row.get('is_species_token', False)),
                    'token_embedding': token_embedding,  # Full 7168 dimensions
                    'mean_embedding': data['mean_embedding'].numpy().tolist(),  # Full 7168 dimensions
                    'num_tokens': int(data.get('num_tokens', len(tokens_df))),
                }
                
                # Add embedding statistics if available
                if 'embedding_stats' in data:
                    stats = data['embedding_stats']
                    record['embedding_mean'] = float(stats.get('mean', 0))
                    record['embedding_std'] = float(stats.get('std', 0))
                    record['embedding_min'] = float(stats.get('min', 0))
                    record['embedding_max'] = float(stats.get('max', 0))
                
                records.append(record)
            
        except Exception as e:
            print(f"Error loading {pt_file}: {e}")
            continue
    
    print(f"Loaded {len(records)} token records from {len(set(r['taxon_id'] for r in records))} species")
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Create directory for data files
    Path('data').mkdir(exist_ok=True)
    
    # Save the full dataset with all embeddings
    # Split into multiple files if needed for size
    chunk_size = 1000  # Adjust based on file size limits
    num_chunks = (len(df) + chunk_size - 1) // chunk_size
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))
        chunk = df.iloc[start_idx:end_idx]
        
        # Save with proper naming convention for Hugging Face
        filename = f'data/train-{i:05d}-of-{num_chunks:05d}.parquet'
        chunk.to_parquet(filename, index=False)
        print(f"Saved {filename} with {len(chunk)} records")
    
    # Also create a species-level summary file with just mean embeddings
    species_records = []
    for taxon_id in df['taxon_id'].unique():
        species_data = df[df['taxon_id'] == taxon_id].iloc[0]
        species_records.append({
            'taxon_id': taxon_id,
            'species_name': species_data['species_name'],
            'num_tokens': species_data['num_tokens'],
            'timestamp': species_data['timestamp'],
            'mean_embedding': species_data['mean_embedding'],
            'embedding_mean': species_data.get('embedding_mean', None),
            'embedding_std': species_data.get('embedding_std', None),
            'embedding_min': species_data.get('embedding_min', None),
            'embedding_max': species_data.get('embedding_max', None),
        })
    
    species_df = pd.DataFrame(species_records)
    species_df.to_parquet('data/species_summary.parquet', index=False)
    print(f"Saved species summary with {len(species_df)} species")
    
    # Create metadata
    metadata = {
        "dataset_name": "Central Florida Native Plants Embeddings (Full)",
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
            "is_species_token": "Whether this token is part of the species name",
            "token_embedding": "7168-dimensional embedding vector for this specific token",
            "mean_embedding": "7168-dimensional mean embedding across all tokens",
            "num_tokens": "Total number of tokens for this species",
            "embedding_mean": "Mean value of the embedding",
            "embedding_std": "Standard deviation of the embedding",
            "embedding_min": "Minimum value in the embedding",
            "embedding_max": "Maximum value in the embedding"
        }
    }
    
    with open('data/dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\nDataset conversion complete!")
    print(f"Total records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Files created: {num_chunks} parquet files + species summary")

if __name__ == "__main__":
    create_full_parquet_dataset()