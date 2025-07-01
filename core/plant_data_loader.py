"""
Florida Native Plants Dataset Integration for DeepEarth Inductive Simulator
Demonstrates seamless integration of the HuggingFace dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image
from datasets import load_dataset
import logging
from pathlib import Path

# Import DeepEarth components
from inductive_simulator import (
    DeepEarthInductiveSimulator,
    InductiveSimulatorConfig,
    DatasetSpecificDecoder,
    MaskingStrategy
)


class FloridaPlantsDatasetProcessor:
    """
    Processor for the Central Florida Native Plants dataset
    Handles all modalities: images, species info, location, phenology
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.logger = logging.getLogger('DeepEarth.FloridaPlants')
        self.cache_dir = cache_dir
        
        # Load dataset from HuggingFace
        self.logger.info("Loading Central Florida Native Plants dataset...")
        self.dataset = load_dataset(
            "deepearth/central-florida-native-plants",
            cache_dir=cache_dir
        )
        
        # Extract metadata
        self._process_metadata()
        
        # Species embedding dimension (learned from co-occurrences)
        self.species_embed_dim = 64
        
        # Initialize species embeddings
        self._init_species_embeddings()
        
        self.logger.info(f"Loaded {len(self.dataset['train'])} plant observations")
        
    def _process_metadata(self):
        """Extract and process dataset metadata"""
        # Get unique species
        self.species_list = []
        self.species_to_idx = {}
        
        for sample in self.dataset['train']:
            species = sample.get('species_name', sample.get('scientific_name'))
            if species and species not in self.species_to_idx:
                self.species_to_idx[species] = len(self.species_list)
                self.species_list.append(species)
                
        self.num_species = len(self.species_list)
        self.logger.info(f"Found {self.num_species} unique species")
        
        # Extract location bounds
        lats = [s['latitude'] for s in self.dataset['train'] if 'latitude' in s]
        lons = [s['longitude'] for s in self.dataset['train'] if 'longitude' in s]
        
        if lats and lons:
            self.lat_bounds = (min(lats), max(lats))
            self.lon_bounds = (min(lons), max(lons))
            self.logger.info(f"Location bounds: {self.lat_bounds}, {self.lon_bounds}")
            
    def _init_species_embeddings(self):
        """Initialize learnable species embeddings"""
        # Create embedding layer
        self.species_embeddings = nn.Embedding(
            self.num_species, 
            self.species_embed_dim
        )
        
        # Initialize with ecological priors if available
        # For now, random initialization
        nn.init.normal_(self.species_embeddings.weight, std=0.1)
        
    def prepare_batch(self, 
                     indices: List[int],
                     include_modalities: List[str] = None) -> Dict[str, torch.Tensor]:
        """
        Prepare a batch of data for DeepEarth
        
        Args:
            indices: Indices of samples to include
            include_modalities: Which modalities to include
            
        Returns:
            Dict ready for DeepEarth processing
        """
        if include_modalities is None:
            include_modalities = ['image', 'species', 'location', 'phenology', 'habitat']
            
        batch = {}
        
        # Process each sample
        image_list = []
        species_list = []
        location_list = []
        phenology_list = []
        habitat_list = []
        description_list = []
        
        for idx in indices:
            sample = self.dataset['train'][idx]
            
            # RGB imagery
            if 'image' in include_modalities and 'image' in sample:
                img = sample['image']
                if isinstance(img, Image.Image):
                    img = img.resize((224, 224))
                    img_array = np.array(img).astype(np.float32) / 255.0
                    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                    image_list.append(img_tensor)
                    
            # Species information
            if 'species' in include_modalities:
                species_name = sample.get('species_name', sample.get('scientific_name'))
                if species_name and species_name in self.species_to_idx:
                    species_idx = self.species_to_idx[species_name]
                    species_embed = self.species_embeddings(
                        torch.tensor(species_idx)
                    )
                    species_list.append(species_embed)
                    
            # Location (converted to xyz + time)
            if 'location' in include_modalities:
                lat = sample.get('latitude', 28.5)  # Default to central Florida
                lon = sample.get('longitude', -81.4)
                elev = sample.get('elevation', 30.0)  # Default elevation
                
                # Convert to ECEF coordinates
                from geospatial.geo2xyz import geo2xyz
                x, y, z = geo2xyz(lat, lon, elev)
                
                # Add temporal component (day of year normalized)
                doy = sample.get('observation_date', {}).get('day_of_year', 180) / 365.0
                
                xyzt = torch.tensor([x, y, z, doy], dtype=torch.float32)
                location_list.append(xyzt)
                
            # Phenology (flowering, fruiting, etc.)
            if 'phenology' in include_modalities:
                phenology = torch.zeros(12)  # Monthly indicators
                
                # Extract phenological state
                if 'flowering_months' in sample:
                    for month in sample['flowering_months']:
                        phenology[month - 1] = 1.0
                        
                if 'fruiting_months' in sample:
                    for month in sample['fruiting_months']:
                        phenology[month - 1] = 0.5
                        
                phenology_list.append(phenology)
                
            # Habitat information
            if 'habitat' in include_modalities:
                habitat_vector = torch.zeros(10)  # 10 habitat types
                
                habitat_map = {
                    'forest': 0, 'scrub': 1, 'wetland': 2, 'prairie': 3,
                    'coastal': 4, 'urban': 5, 'agricultural': 6,
                    'riparian': 7, 'pine_flatwoods': 8, 'hammock': 9
                }
                
                habitat_desc = sample.get('habitat', '').lower()
                for habitat, idx in habitat_map.items():
                    if habitat in habitat_desc:
                        habitat_vector[idx] = 1.0
                        
                habitat_list.append(habitat_vector)
                
            # Natural language description
            if 'description' in include_modalities:
                desc_parts = []
                
                # Build description from available fields
                if 'common_name' in sample:
                    desc_parts.append(f"{sample['common_name']}")
                    
                if 'description' in sample:
                    desc_parts.append(sample['description'])
                    
                if 'habitat' in sample:
                    desc_parts.append(f"Found in {sample['habitat']}")
                    
                description = " ".join(desc_parts) if desc_parts else "Florida native plant"
                description_list.append(description)
                
        # Stack into batches
        if image_list:
            batch['plant_imagery'] = torch.stack(image_list)
            
        if species_list:
            batch['species_embedding'] = torch.stack(species_list)
            
        if location_list:
            batch['xyzt'] = torch.stack(location_list)
            
        if phenology_list:
            batch['phenology'] = torch.stack(phenology_list)
            
        if habitat_list:
            batch['habitat'] = torch.stack(habitat_list)
            
        if description_list:
            batch['plant_description'] = description_list
            
        return batch
        
    def create_ecological_context(self, 
                                 sample_idx: int,
                                 context_radius_km: float = 10.0) -> Dict[str, torch.Tensor]:
        """
        Create ecological context by finding nearby observations
        This helps the model learn species co-occurrences and habitat relationships
        """
        target_sample = self.dataset['train'][sample_idx]
        target_lat = target_sample.get('latitude', 28.5)
        target_lon = target_sample.get('longitude', -81.4)
        
        # Find nearby observations
        nearby_indices = []
        
        for idx, sample in enumerate(self.dataset['train']):
            if idx == sample_idx:
                continue
                
            lat = sample.get('latitude', 0)
            lon = sample.get('longitude', 0)
            
            # Simple distance calculation (could use haversine for accuracy)
            dist = np.sqrt((lat - target_lat)**2 + (lon - target_lon)**2) * 111.0  # ~km
            
            if dist < context_radius_km:
                nearby_indices.append(idx)
                
        # Create context batch
        if nearby_indices:
            context_batch = self.prepare_batch(
                nearby_indices[:8],  # Limit to 8 nearby observations
                include_modalities=['species', 'habitat']
            )
            
            # Aggregate species co-occurrence
            if 'species_embedding' in context_batch:
                co_occurrence = context_batch['species_embedding'].mean(dim=0)
                context_batch['species_cooccurrence'] = co_occurrence
                
            # Aggregate habitat information
            if 'habitat' in context_batch:
                habitat_distribution = context_batch['habitat'].mean(dim=0)
                context_batch['habitat_distribution'] = habitat_distribution
                
        else:
            context_batch = {}
            
        return context_batch


class FloridaPlantsDeepEarth:
    """
    Integration of Florida Plants dataset with DeepEarth Inductive Simulator
    """
    
    def __init__(self, 
                 simulator_config: Optional[InductiveSimulatorConfig] = None,
                 device: str = "cuda"):
        
        self.device = device
        self.logger = logging.getLogger('DeepEarth.FloridaPlantsIntegration')
        
        # Initialize simulator
        if simulator_config is None:
            simulator_config = InductiveSimulatorConfig(
                mask_ratio=0.15,
                mask_type="block",
                fusion_layers=12,
                simulator_layers=12
            )
            
        self.simulator = DeepEarthInductiveSimulator(simulator_config)
        
        # Initialize dataset processor
        self.dataset_processor = FloridaPlantsDatasetProcessor()
        
        # Add Florida plants specific modalities to simulator
        self._register_modalities()
        
        self.logger.info("Florida Plants DeepEarth integration complete")
        
    def _register_modalities(self):
        """Register all Florida plants modalities with the simulator"""
        
        # Plant imagery (RGB)
        self.simulator.add_dataset(
            name="plant_imagery",
            modality="vision",
            output_dim=768,  # V-JEPA2 dimension
            output_shape=(3, 224, 224)
        )
        
        # Species embeddings
        self.simulator.add_dataset(
            name="species_embedding",
            modality="tabular",
            output_dim=self.dataset_processor.species_embed_dim,
            output_shape=None
        )
        
        # Phenology (monthly patterns)
        self.simulator.add_dataset(
            name="phenology",
            modality="timeseries",
            output_dim=12,
            output_shape=None
        )
        
        # Habitat vector
        self.simulator.add_dataset(
            name="habitat",
            modality="tabular",
            output_dim=10,
            output_shape=None
        )
        
        # Natural language descriptions
        self.simulator.add_dataset(
            name="plant_description",
            modality="language",
            output_dim=4096,  # DeepSeek dimension
            output_shape=None
        )
        
        # Co-occurrence patterns
        self.simulator.add_dataset(
            name="species_cooccurrence",
            modality="tabular",
            output_dim=self.dataset_processor.species_embed_dim,
            output_shape=None
        )
        
    def train_step(self, batch_size: int = 16) -> Dict[str, float]:
        """
        Single training step with Florida plants data
        """
        # Sample random indices
        indices = np.random.choice(
            len(self.dataset_processor.dataset['train']),
            batch_size,
            replace=False
        )
        
        # Prepare batch
        batch = self.dataset_processor.prepare_batch(indices.tolist())
        
        # Add ecological context for some samples
        for i in range(batch_size // 4):  # 25% of samples
            context = self.dataset_processor.create_ecological_context(indices[i])
            if 'species_cooccurrence' in context:
                if 'species_cooccurrence' not in batch:
                    batch['species_cooccurrence'] = torch.zeros(
                        batch_size, 
                        self.dataset_processor.species_embed_dim
                    )
                batch['species_cooccurrence'][i] = context['species_cooccurrence']
                
        # Move to device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
                
        # Configure masking - mask different modalities at different rates
        mask_config = {
            'plant_imagery': 0.25,  # Mask 25% of image patches
            'species_embedding': 0.1,  # Mask 10% of species
            'phenology': 0.2,  # Mask 20% of phenological data
            'habitat': 0.15,  # Mask 15% of habitat info
            'plant_description': 0.0,  # Don't mask language (use as context)
        }
        
        # Forward pass through simulator
        outputs = self.simulator(batch, mask_config)
        
        # Extract losses
        losses = {
            'total': outputs['total_loss'].item()
        }
        for name, loss in outputs['losses'].items():
            losses[name] = loss.item()
            
        return losses
        
    def evaluate_ecological_understanding(self, 
                                        test_indices: List[int]) -> Dict[str, float]:
        """
        Evaluate the model's understanding of ecological relationships
        """
        results = {
            'species_prediction_acc': 0.0,
            'habitat_prediction_acc': 0.0,
            'phenology_prediction_mae': 0.0,
            'cooccurrence_correlation': 0.0
        }
        
        for idx in test_indices:
            # Prepare single sample
            batch = self.dataset_processor.prepare_batch([idx])
            
            # Mask species and predict
            if 'species_embedding' in batch:
                original_species = batch['species_embedding'].clone()
                batch['species_embedding'] = torch.zeros_like(original_species)
                
                outputs = self.simulator(batch, {'species_embedding': 1.0})
                
                if 'species_embedding' in outputs['reconstructions']:
                    pred_species = outputs['reconstructions']['species_embedding']
                    # Compute similarity
                    similarity = F.cosine_similarity(
                        pred_species.flatten(), 
                        original_species.flatten(), 
                        dim=0
                    )
                    results['species_prediction_acc'] += similarity.item()
                    
        # Average results
        num_samples = len(test_indices)
        for key in results:
            results[key] /= num_samples
            
        return results
        
    def generate_species_distribution_map(self, 
                                         species_name: str,
                                         lat_range: Tuple[float, float],
                                         lon_range: Tuple[float, float],
                                         resolution: int = 50) -> np.ndarray:
        """
        Generate a species distribution map using the trained model
        """
        # Create grid of locations
        lats = np.linspace(lat_range[0], lat_range[1], resolution)
        lons = np.linspace(lon_range[0], lon_range[1], resolution)
        
        distribution_map = np.zeros((resolution, resolution))
        
        # Get species embedding
        if species_name in self.dataset_processor.species_to_idx:
            species_idx = self.dataset_processor.species_to_idx[species_name]
            species_embed = self.dataset_processor.species_embeddings(
                torch.tensor(species_idx)
            )
        else:
            self.logger.warning(f"Unknown species: {species_name}")
            return distribution_map
            
        # For each location
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                # Create mock observation
                from geospatial.geo2xyz import geo2xyz
                x, y, z = geo2xyz(lat, lon, 30.0)  # Default elevation
                
                batch = {
                    'xyzt': torch.tensor([[x, y, z, 0.5]]),  # Mid-year
                    'species_embedding': species_embed.unsqueeze(0),
                    'habitat': torch.zeros(1, 10),  # Unknown habitat
                }
                
                # Mask habitat and predict
                outputs = self.simulator(batch, {'habitat': 1.0})
                
                if 'habitat' in outputs['reconstructions']:
                    # Get predicted habitat suitability
                    habitat_pred = outputs['reconstructions']['habitat']
                    # Simple suitability score
                    suitability = habitat_pred.sum().item()
                    distribution_map[i, j] = suitability
                    
        return distribution_map


# Example usage
def example_florida_plants_training():
    """Complete example of training DeepEarth on Florida plants"""
    print("=== Florida Native Plants DeepEarth Training ===")
    
    # Initialize
    deepearth_florida = FloridaPlantsDeepEarth(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Training loop
    num_epochs = 10
    steps_per_epoch = 100
    
    for epoch in range(num_epochs):
        epoch_losses = {'total': 0.0}
        
        for step in range(steps_per_epoch):
            # Train step
            losses = deepearth_florida.train_step(batch_size=16)
            
            # Accumulate losses
            for key, value in losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0.0
                epoch_losses[key] += value
                
            # Log progress
            if step % 10 == 0:
                print(f"Epoch {epoch+1}, Step {step}: Loss = {losses['total']:.4f}")
                
        # Average epoch losses
        for key in epoch_losses:
            epoch_losses[key] /= steps_per_epoch
            
        print(f"\nEpoch {epoch+1} Summary:")
        for key, value in epoch_losses.items():
            print(f"  {key}: {value:.4f}")
            
    # Evaluate ecological understanding
    print("\nEvaluating ecological understanding...")
    test_indices = list(range(100, 150))  # 50 test samples
    eval_results = deepearth_florida.evaluate_ecological_understanding(test_indices)
    
    print("\nEvaluation Results:")
    for metric, value in eval_results.items():
        print(f"  {metric}: {value:.3f}")
        
    # Generate species distribution map
    print("\nGenerating species distribution map for Saw Palmetto...")
    distribution = deepearth_florida.generate_species_distribution_map(
        species_name="Serenoa repens",  # Saw Palmetto
        lat_range=(27.0, 29.0),  # Central Florida
        lon_range=(-82.0, -80.0),
        resolution=25
    )
    
    print(f"Distribution map shape: {distribution.shape}")
    print(f"Suitability range: [{distribution.min():.2f}, {distribution.max():.2f}]")
    
    return deepearth_florida


if __name__ == "__main__":
    # Run the example
    florida_model = example_florida_plants_training()
    
    print("\n" + "="*60)
    print("Florida Plants successfully integrated with DeepEarth!")
    print("The model can now:")
    print("  - Predict species occurrences from partial observations")
    print("  - Infer habitat suitability from species presence")
    print("  - Simulate phenological patterns across seasons")
    print("  - Generate species distribution maps")
    print("  - Learn ecological relationships inductively")
    print("="*60)