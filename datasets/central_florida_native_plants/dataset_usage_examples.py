#!/usr/bin/env python3
"""
Dataset Usage Examples for Central Florida Native Plants v0.2.0

This module provides convenient classes and examples for working with the
DeepEarth Central Florida Native Plants dataset.
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import json


class CentralFloridaPlantsDataset:
    """
    Convenient interface for the Central Florida Native Plants dataset.
    
    Example:
        dataset = CentralFloridaPlantsDataset("path/to/downloaded/dataset")
        observations = dataset.get_observations(year_range=(2020, 2025))
        embedding = dataset.load_vision_embedding(gbif_id)
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize dataset loader.
        
        Args:
            dataset_path: Path to the downloaded HuggingFace dataset
        """
        self.base_path = Path(dataset_path)
        
        # Load main dataset
        self.observations = pd.read_parquet(self.base_path / "observations.parquet")
        
        # Load vision index if available
        vision_index_path = self.base_path / "vision_index.parquet"
        if vision_index_path.exists():
            self.vision_index = pd.read_parquet(vision_index_path)
        else:
            self.vision_index = None
            
        # Load metadata
        metadata_path = self.base_path / "dataset_info.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def get_observations(self, 
                        year_range: Optional[Tuple[int, int]] = None,
                        bbox: Optional[Tuple[float, float, float, float]] = None,
                        species: Optional[List[str]] = None,
                        has_vision: Optional[bool] = None) -> pd.DataFrame:
        """
        Get filtered observations.
        
        Args:
            year_range: (min_year, max_year) tuple
            bbox: (min_lat, min_lon, max_lat, max_lon) bounding box
            species: List of species names to filter
            has_vision: Filter for observations with/without vision embeddings
            
        Returns:
            Filtered DataFrame of observations
        """
        df = self.observations.copy()
        
        if year_range:
            df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
            
        if bbox:
            min_lat, min_lon, max_lat, max_lon = bbox
            df = df[(df['latitude'] >= min_lat) & (df['latitude'] <= max_lat) &
                   (df['longitude'] >= min_lon) & (df['longitude'] <= max_lon)]
            
        if species:
            df = df[df['taxon_name'].isin(species)]
            
        if has_vision is not None:
            df = df[df['has_vision'] == has_vision]
            
        return df
    
    def load_vision_embedding(self, gbif_id: int) -> Optional[np.ndarray]:
        """
        Load vision embedding for a specific observation.
        
        Args:
            gbif_id: GBIF observation ID
            
        Returns:
            Flattened vision embedding array or None if not found
        """
        if self.vision_index is None:
            return None
            
        # Find the file containing this embedding
        row = self.vision_index[self.vision_index['gbif_id'] == gbif_id]
        if row.empty:
            return None
            
        file_idx = row.iloc[0]['file_idx']
        file_path = self.base_path / f"vision_embeddings/embeddings_{file_idx:06d}.parquet"
        
        if not file_path.exists():
            return None
            
        # Load the specific embedding
        embeddings_df = pd.read_parquet(file_path)
        embedding_row = embeddings_df[embeddings_df['gbif_id'] == gbif_id]
        
        if embedding_row.empty:
            return None
            
        return embedding_row.iloc[0]['embedding']
    
    def reshape_vision_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Reshape flattened vision embedding to 4D tensor.
        
        Args:
            embedding: Flattened embedding of shape (6488064,)
            
        Returns:
            Reshaped embedding of shape (8, 24, 24, 1408)
        """
        return embedding.reshape(8, 24, 24, 1408)
    
    def get_language_embedding(self, species_name: str) -> Optional[np.ndarray]:
        """
        Get language embedding for a species.
        
        Args:
            species_name: Scientific name of the species
            
        Returns:
            Language embedding array or None if not found
        """
        species_obs = self.observations[self.observations['taxon_name'] == species_name]
        if species_obs.empty:
            return None
            
        # All observations of the same species have the same language embedding
        return np.array(species_obs.iloc[0]['language_embedding'])
    
    def get_species_list(self) -> List[str]:
        """Get list of all unique species in the dataset."""
        return sorted(self.observations['taxon_name'].unique())
    
    def get_dataset_stats(self) -> Dict[str, any]:
        """Get basic dataset statistics."""
        return {
            'total_observations': len(self.observations),
            'unique_species': self.observations['taxon_name'].nunique(),
            'observations_with_images': (self.observations['num_images'] > 0).sum(),
            'observations_with_vision': self.observations['has_vision'].sum(),
            'year_range': (self.observations['year'].min(), self.observations['year'].max()),
            'splits': self.observations['split'].value_counts().to_dict()
        }


# Example usage functions
def example_basic_loading():
    """Example: Basic dataset loading and exploration."""
    dataset = CentralFloridaPlantsDataset("./huggingface_dataset")
    
    # Get dataset statistics
    stats = dataset.get_dataset_stats()
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Get all species
    species_list = dataset.get_species_list()
    print(f"\nFound {len(species_list)} species")
    print("First 5 species:", species_list[:5])


def example_filtered_queries():
    """Example: Filtering observations by various criteria."""
    dataset = CentralFloridaPlantsDataset("./huggingface_dataset")
    
    # Get recent observations from a specific area
    recent_orlando = dataset.get_observations(
        year_range=(2020, 2025),
        bbox=(28.3, -81.5, 28.7, -81.2),  # Orlando area
        has_vision=True
    )
    print(f"Recent Orlando observations with vision: {len(recent_orlando)}")
    
    # Get observations of a specific species
    species_obs = dataset.get_observations(species=["Quercus virginiana"])
    print(f"Quercus virginiana observations: {len(species_obs)}")


def example_vision_embeddings():
    """Example: Working with vision embeddings."""
    dataset = CentralFloridaPlantsDataset("./huggingface_dataset")
    
    # Get an observation with vision embedding
    vision_obs = dataset.get_observations(has_vision=True).iloc[0]
    gbif_id = vision_obs['gbif_id']
    
    # Load the embedding
    embedding = dataset.load_vision_embedding(gbif_id)
    if embedding is not None:
        print(f"Loaded embedding shape: {embedding.shape}")
        
        # Reshape to 4D
        embedding_4d = dataset.reshape_vision_embedding(embedding)
        print(f"Reshaped to: {embedding_4d.shape}")
        
        # Convert to PyTorch tensor
        embedding_tensor = torch.from_numpy(embedding_4d).float()
        print(f"PyTorch tensor: {embedding_tensor.shape}")


def example_language_embeddings():
    """Example: Working with language embeddings."""
    dataset = CentralFloridaPlantsDataset("./huggingface_dataset")
    
    # Get language embedding for a species
    species_name = "Sabal palmetto"
    lang_embedding = dataset.get_language_embedding(species_name)
    
    if lang_embedding is not None:
        print(f"Language embedding for {species_name}:")
        print(f"  Shape: {lang_embedding.shape}")
        print(f"  Dimensions: {len(lang_embedding)}")
        print(f"  Mean: {lang_embedding.mean():.4f}")
        print(f"  Std: {lang_embedding.std():.4f}")


if __name__ == "__main__":
    print("Central Florida Native Plants Dataset Examples")
    print("=" * 50)
    
    # Run examples (comment out if dataset not downloaded)
    # example_basic_loading()
    # example_filtered_queries()
    # example_vision_embeddings()
    # example_language_embeddings()
    
    print("\nTo run examples, uncomment the function calls above.")
    print("Make sure to download the dataset first using:")
    print("  ./download_dataset.sh")