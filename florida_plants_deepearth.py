"""
DeepEarth integration with Central Florida Native Plants dataset
"""

import torch
import torch.nn as nn
from datasets import load_dataset
from PIL import Image
import numpy as np
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FloridaPlantsProcessor:
    """Process Central Florida Native Plants dataset for DeepEarth"""
    
    def __init__(self, cache_dir: Optional[str] = "./data"):
        logger.info("Loading Central Florida Native Plants dataset from HuggingFace...")
        
        # Load the dataset
        self.dataset = load_dataset(
            "deepearth/central-florida-native-plants",
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        logger.info(f"Dataset loaded with {len(self.dataset['train'])} samples")
        
        # Extract unique species for embeddings
        self.species_list = self._extract_species()
        self.species_to_idx = {s: i for i, s in enumerate(self.species_list)}
        logger.info(f"Found {len(self.species_list)} unique species")
        
    def _extract_species(self) -> List[str]:
        """Extract unique species names"""
        species = set()
        for sample in self.dataset['train']:
        # FIX: The dataset uses 'scientificName', not 'species_name'
           name = sample.get('scientificName')
           if name:
              species.add(name)

        return sorted(list(species))
    
    def prepare_batch(self, indices: List[int], device: str = "cuda") -> Dict[str, torch.Tensor]:
        """Prepare a batch for DeepEarth processing"""
        batch_data = {
            'images': [],
            'descriptions': [],
            'locations': [],
            'species_ids': [],
            'metadata': []
        }
        
        for idx in indices:
            sample = self.dataset['train'][idx]
            
            # Process image
            if 'image' in sample and sample['image'] is not None:
                img = sample['image']
                if isinstance(img, Image.Image):
                    # Resize to standard size
                    img = img.resize((224, 224), Image.Resampling.LANCZOS)
                    # Convert to tensor
                    img_array = np.array(img).astype(np.float32) / 255.0
                    if len(img_array.shape) == 2:  # Grayscale
                        img_array = np.stack([img_array] * 3, axis=-1)
                    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                    batch_data['images'].append(img_tensor)
            
            # Process text description
            description_parts = []
            if 'common_name' in sample:
                description_parts.append(f"Common name: {sample['common_name']}")
            if 'scientific_name' in sample:
                description_parts.append(f"Scientific name: {sample['scientific_name']}")
            if 'description' in sample:
                description_parts.append(sample['description'])
            if 'habitat' in sample:
                description_parts.append(f"Habitat: {sample['habitat']}")
            
            description = " ".join(description_parts)
            batch_data['descriptions'].append(description)
            
            # Process location (lat, lon, elevation)
            lat = sample.get('latitude', 28.5)  # Default to central Florida
            lon = sample.get('longitude', -81.4)
            elev = sample.get('elevation', 30.0)  # Default elevation in meters
            
            # Convert to ECEF coordinates for DeepEarth
            from geospatial.geo2xyz import geo2xyz
            x, y, z = geo2xyz(lat, lon, elev)
            
            # Add time component (normalized day of year)
            doy = sample.get('observation_date', {}).get('day_of_year', 180) / 365.0
            
            location = torch.tensor([x, y, z, doy], dtype=torch.float32)
            batch_data['locations'].append(location)
            
            # Species ID for embedding
            species = sample.get('species_name', sample.get('scientific_name', 'unknown'))
            species_id = self.species_to_idx.get(species, -1)
            batch_data['species_ids'].append(species_id)
            
            # Additional metadata
            metadata = {
                'flowering_months': sample.get('flowering_months', []),
                'fruiting_months': sample.get('fruiting_months', []),
                'native_status': sample.get('native_status', 'unknown'),
                'growth_form': sample.get('growth_form', 'unknown')
            }
            batch_data['metadata'].append(metadata)
        
        # Convert lists to tensors
        if batch_data['images']:
            batch_data['images'] = torch.stack(batch_data['images']).to(device)
        else:
            batch_data['images'] = torch.zeros(len(indices), 3, 224, 224).to(device)
            
        batch_data['locations'] = torch.stack(batch_data['locations']).to(device)
        batch_data['species_ids'] = torch.tensor(batch_data['species_ids']).to(device)
        
        return batch_data


def test_florida_plants_integration():
    """Test the Florida plants dataset with DeepEarth"""
    
    # Initialize processor
    processor = FloridaPlantsProcessor()
    
    # Get a sample batch
    batch_indices = list(range(8))  # First 8 samples
    batch = processor.prepare_batch(batch_indices)
    
    print("\nBatch prepared:")
    print(f"  Images shape: {batch['images'].shape}")
    print(f"  Locations shape: {batch['locations'].shape}")
    print(f"  Species IDs: {batch['species_ids'].tolist()}")
    print(f"  Descriptions: {len(batch['descriptions'])} texts")
    
    # Now let's create a simple DeepEarth-style processing
    from core.inductive_simulator import create_inductive_simulator
    
    print("\nCreating simulator for Florida plants...")
    simulator = create_inductive_simulator(preset="fast").cuda()
    
    # Process vision features (simplified - in real DeepEarth this would use V-JEPA2)
    vision_encoder = nn.Sequential(
        nn.Conv2d(3, 64, 7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((7, 7)),
        nn.Flatten(),
        nn.Linear(64 * 49, 2048)
    ).cuda()
    
    # Extract features
    with torch.no_grad():
        vision_features = vision_encoder(batch['images'])  # (B, 2048)
        
        # Add spatial encoding from locations
        location_encoder = nn.Linear(4, 2048).cuda()
        location_features = location_encoder(batch['locations'])  # (B, 2048)
        
        # Combine features
        combined_features = torch.stack([vision_features, location_features], dim=1)  # (B, 2, 2048)
        
        # Process through simulator
        output = simulator(combined_features)
        
    print(f"\nSimulator output shape: {output['simulated_features'].shape}")
    print("\n✓ Florida plants dataset successfully integrated with DeepEarth!")
    
    return processor, batch


def train_on_florida_plants():
    """Training loop for Florida plants"""
    from torch.utils.data import DataLoader, Dataset
    
    class FloridaPlantsDataset(Dataset):
        def __init__(self, processor, split='train'):
            self.processor = processor
            self.indices = list(range(len(processor.dataset[split])))
            
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            return self.processor.prepare_batch([self.indices[idx]])
    
    # Create dataset and dataloader
    processor = FloridaPlantsProcessor()
    dataset = FloridaPlantsDataset(processor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    
    # Create simulator
    simulator = create_inductive_simulator(preset="standard").cuda()
    optimizer = torch.optim.AdamW(simulator.parameters(), lr=1e-4)
    
    print("Starting training on Florida plants dataset...")
    
    for epoch in range(5):
        for batch_idx, batch in enumerate(dataloader):
            # Training step would go here
            # This is where you'd integrate with the full DeepEarth pipeline
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}")
                
            if batch_idx > 50:  # Just a demo
                break
                
    print("Training demo complete!")


if __name__ == "__main__":
    # Test the integration
    processor, sample_batch = test_florida_plants_integration()
    
    # Uncomment to run training demo
    # train_on_florida_plants()
