#!/usr/bin/env python3
"""
Convert embeddings to Parquet format for Hugging Face Dataset Viewer
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json

def create_parquet_dataset():
    """Create a Parquet file with the embeddings data"""
    embeddings_dir = Path("embeddings")
    
    if not embeddings_dir.exists():
        print("Error: embeddings directory not found. Please run download_dataset.sh first.")
        return
    
    # Collect all data
    records = []
    
    for pt_file in sorted(embeddings_dir.glob("*.pt")):
        try:
            # Load embedding data
            data = torch.load(pt_file)
            
            # Convert embedding stats to JSON string for storage
            embedding_stats = json.dumps(data.get('embedding_stats', {}))
            
            # Create record with flattened structure
            record = {
                'taxon_id': data.get('taxon_id', pt_file.stem),
                'species_name': data.get('species_name', 'Unknown'),
                'num_tokens': int(data.get('num_tokens', 0)),
                'timestamp': data.get('timestamp', ''),
                'embedding_stats': embedding_stats,
                # Store mean embedding as list (Dataset Viewer can handle this)
                'mean_embedding': data['mean_embedding'].numpy().tolist()
            }
            
            records.append(record)
            
        except Exception as e:
            print(f"Error loading {pt_file}: {e}")
            continue
    
    print(f"Loaded {len(records)} species embeddings")
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Save as Parquet (split embeddings into chunks if needed for viewer)
    # We'll create a main file with metadata and a sample of embeddings
    df_viewer = df.copy()
    
    # For the viewer, we'll show just the first 10 dimensions of embeddings
    # to make it more readable
    df_viewer['embedding_sample'] = df_viewer['mean_embedding'].apply(lambda x: x[:10])
    df_viewer['embedding_dimension'] = df_viewer['mean_embedding'].apply(lambda x: len(x))
    
    # Drop the full embedding for the viewer version
    df_viewer_display = df_viewer.drop('mean_embedding', axis=1)
    
    # Save viewer-friendly version
    df_viewer_display.to_parquet('data/train-00000-of-00001.parquet', index=False)
    
    # Also save the full dataset
    df.to_parquet('data/embeddings_full.parquet', index=False)
    
    print("Created Parquet files in data/ directory")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Create a JSON metadata file for the dataset
    metadata = {
        "dataset_name": "Central Florida Native Plants Embeddings",
        "num_examples": len(df),
        "embedding_dimension": 7168,
        "model": "DeepSeek-V3-0324-UD-Q4_K_XL",
        "features": {
            "taxon_id": "GBIF taxonomic identifier",
            "species_name": "Scientific name of the plant species",
            "num_tokens": "Number of tokens in the embedding",
            "timestamp": "When the embedding was created",
            "embedding_stats": "JSON string with embedding statistics",
            "mean_embedding": "7168-dimensional embedding vector"
        }
    }
    
    Path('data').mkdir(exist_ok=True)
    with open('data/dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Dataset conversion complete!")

if __name__ == "__main__":
    create_parquet_dataset()