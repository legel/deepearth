#!/usr/bin/env python3
"""
Client for DeepSeek Model Server
"""

import requests
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch

class DeepSeekClient:
    def __init__(self, base_url='http://localhost:8888'):
        self.base_url = base_url
        
    def health_check(self) -> Dict[str, Any]:
        """Check if server is healthy"""
        response = requests.get(f'{self.base_url}/health')
        return response.json()
    
    def tokenize(self, text: str) -> Dict[str, Any]:
        """Tokenize text"""
        response = requests.post(f'{self.base_url}/tokenize', 
                               json={'text': text})
        return response.json()
    
    def embed(self, text: str, return_tokens=True, return_token_embeddings=False) -> Dict[str, Any]:
        """Get embeddings for text"""
        response = requests.post(f'{self.base_url}/embed', 
                               json={
                                   'text': text,
                                   'return_tokens': return_tokens,
                                   'return_token_embeddings': return_token_embeddings
                               })
        return response.json()
    
    def embed_batch(self, texts: List[str], save_to_csv=False, csv_path=None) -> Dict[str, Any]:
        """Get embeddings for multiple texts"""
        payload = {
            'texts': texts,
            'save_to_csv': save_to_csv
        }
        if csv_path:
            payload['csv_path'] = csv_path
            
        response = requests.post(f'{self.base_url}/embed_batch', json=payload)
        return response.json()
    
    def complete(self, prompt: str, max_tokens=100, temperature=0.7, top_p=0.9) -> Dict[str, Any]:
        """Generate completion"""
        response = requests.post(f'{self.base_url}/complete', 
                               json={
                                   'prompt': prompt,
                                   'max_tokens': max_tokens,
                                   'temperature': temperature,
                                   'top_p': top_p
                               })
        return response.json()
    
    def extract_embeddings_with_tokens(self, 
                                     texts: List[str], 
                                     labels: Optional[List[str]] = None,
                                     output_dir: Optional[Path] = None,
                                     save_format: str = 'pt') -> Dict[str, Any]:
        """
        Extract embeddings for multiple texts with full token information
        
        Args:
            texts: List of text strings to embed
            labels: Optional labels for each text
            output_dir: Directory to save embeddings
            save_format: 'pt' for PyTorch, 'json' for JSON
            
        Returns:
            Dictionary with embedding results and file paths
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        results = []
        
        for idx, text in enumerate(texts):
            label = labels[idx] if labels else f"text_{idx}"
            
            # Get embeddings with full token info
            result = self.embed(text, return_tokens=True, return_token_embeddings=True)
            
            if output_dir:
                if save_format == 'pt':
                    # Save as PyTorch format
                    save_data = {
                        'text': text,
                        'label': label,
                        'mean_embedding': torch.tensor(result['mean_embedding'], dtype=torch.float32),
                        'num_tokens': result['num_tokens'],
                        'tokens': result['tokens']
                    }
                    
                    if 'token_embeddings' in result:
                        save_data['token_embeddings'] = torch.tensor(result['token_embeddings'], dtype=torch.float32)
                    
                    filename = f"{label.replace(' ', '_').lower()}.pt"
                    filepath = output_dir / filename
                    torch.save(save_data, filepath)
                    
                elif save_format == 'json':
                    # Save as JSON
                    filename = f"{label.replace(' ', '_').lower()}.json"
                    filepath = output_dir / filename
                    with open(filepath, 'w') as f:
                        json.dump(result, f)
                
                results.append({
                    'label': label,
                    'text': text,
                    'filepath': str(filepath),
                    'num_tokens': result['num_tokens']
                })
            else:
                results.append({
                    'label': label,
                    'text': text,
                    'embedding': result['mean_embedding'],
                    'num_tokens': result['num_tokens']
                })
        
        return {
            'num_processed': len(texts),
            'results': results,
            'output_dir': str(output_dir) if output_dir else None
        }

def extract_species_embeddings(client: DeepSeekClient, 
                             csv_path: str, 
                             output_dir: str,
                             species_column: str = 'taxon_name',
                             id_column: str = 'taxon_id'):
    """
    Extract embeddings for species from a CSV file
    
    Args:
        client: DeepSeekClient instance
        csv_path: Path to CSV with species data
        output_dir: Directory to save embeddings
        species_column: Column name for species names
        id_column: Column name for species IDs
    """
    # Load species data
    df = pd.read_csv(csv_path)
    unique_taxa = df[[species_column, id_column]].drop_duplicates().reset_index(drop=True)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Process each species
    for idx, row in unique_taxa.iterrows():
        species = row[species_column]
        taxon_id = row[id_column]
        
        print(f"[{idx+1}/{len(unique_taxa)}] Processing: {species}")
        
        # Format prompt - using DeepSeek special tokens
        prompt = f"<｜User｜>Ecophysiology of {species}:<｜Assistant｜>"
        
        # Get embeddings with full token information
        result = client.embed(prompt, return_tokens=True, return_token_embeddings=True)
        
        # Save embeddings
        taxon_id_str = str(int(taxon_id)) if not pd.isna(taxon_id) else f"unknown_{idx}"
        
        # Save .pt file with embeddings
        save_data = {
            'mean_embedding': torch.tensor(result['mean_embedding'], dtype=torch.float32),
            'species_name': species,
            'taxon_id': taxon_id,
            'prompt': prompt,
            'num_tokens': result['num_tokens'],
            'tokens': result['tokens']
        }
        
        if 'token_embeddings' in result:
            save_data['token_embeddings'] = torch.tensor(result['token_embeddings'], dtype=torch.float32)
        
        pt_path = output_dir / f'{taxon_id_str}.pt'
        torch.save(save_data, pt_path)
        
        # Save token mapping CSV
        token_df = pd.DataFrame(result['tokens'])
        token_df['species_name'] = species
        token_df['taxon_id'] = taxon_id
        
        csv_path = output_dir / f'{taxon_id_str}_tokens.csv'
        token_df.to_csv(csv_path, index=False)
        
        print(f"  Saved: {pt_path.name} and {csv_path.name}")

def main():
    """Example usage"""
    # Create client
    client = DeepSeekClient()
    
    # Check health
    print("Health check:", client.health_check())
    
    # Example: Tokenize text
    tokens = client.tokenize("Ecophysiology of Magnolia grandiflora")
    print("\nTokenization result:")
    print(f"  Num tokens: {tokens['num_tokens']}")
    for token in tokens['tokens'][:5]:
        print(f"  Position {token['position']}: '{token['token_str']}' (ID: {token['token_id']})")
    
    # Example: Get embeddings
    result = client.embed("Ecophysiology of Quercus virginiana")
    print("\nEmbedding result:")
    print(f"  Shape: {result['embedding_shape']}")
    print(f"  Mean embedding stats: {result['mean_embedding_stats']}")
    
    # Example: Extract embeddings for multiple texts
    texts = [
        "The ecology of coastal wetlands",
        "Climate adaptation in tropical forests",
        "Urban biodiversity patterns"
    ]
    
    results = client.extract_embeddings_with_tokens(texts, output_dir="embeddings")
    print(f"\nProcessed {results['num_processed']} texts")

if __name__ == '__main__':
    main()