# deepearth/models/flowering/preprocess_flowering_data.py
"""
Angiosperm Dataset Bridge for DeepEarth
═══════════════════════════════════════

Specialized data loader for the angiosperm flowering dataset with pre-computed
embeddings from multiple encoders (AlphaEarth, BioCLIP 2, PhenoVision).

This bridge handles the specific structure and semantics of the angiosperm
dataset without any silent failures or fallbacks.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
from tqdm import tqdm


class FloweringDatasetPreprocessor:
    """
    Bridge between angiosperm flowering dataset and DeepEarth architecture.
    
    Handles the specific file structure and encoding combinations:
    - (D1, M1, E1): AlphaEarth embeddings of spatiotemporal coordinates
    - (D1, M1, E2): Earth4D embeddings of spatiotemporal coordinates
    - (D2, M2, E3): BioCLIP embeddings of iNaturalist images
    - (D2, M3, E3): BioCLIP embeddings of species names
    - (D2, M2, E4): PhenoVision flowering classifications
    """
    
    # Define valid combinations as class constants for clarity
    DATASET_IDS = {
        'spacetime_coordinates': 0,  # D1
        'inaturalist': 1             # D2
    }
    
    MODALITY_IDS = {
        'spacetime': 0,  # M1
        'images': 1,     # M2
        'text': 2        # M3
    }
    
    ENCODER_IDS = {
        'alphaearth': 0,   # E1
        'earth4d': 1,      # E2
        'bioclip2': 2,     # E3
        'phenovision': 3   # E4
    }
    
    # Valid (dataset, modality, encoder) combinations
    VALID_COMBINATIONS = [
        (0, 0, 0),  # (D1, M1, E1) - AlphaEarth spatiotemporal
        (0, 0, 1),  # (D1, M1, E2) - Earth4D spatiotemporal
        (1, 1, 2),  # (D2, M2, E3) - BioCLIP images
        (1, 2, 2),  # (D2, M3, E3) - BioCLIP text
        (1, 1, 3),  # (D2, M2, E4) - PhenoVision flowering
    ]
    
    def __init__(self, config, data_dir: str):
        """
        Initialize angiosperm dataset bridge.
        
        Args:
            config: DeepEarth configuration
            data_dir: Directory containing angiosperm dataset files
        """
        self.config = config
        self.data_dir = Path(data_dir)
        
        print(f"\n{'='*70}")
        print(f"Angiosperm Dataset Bridge Initialization")
        print(f"{'='*70}")
        print(f"Data directory: {self.data_dir}")
        
        # Validate directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Initialize coordinate transformer for Earth4D
        import sys
        sys.path.append('/opt/ecodash/deepearth')
        from core.preprocessor import CoordinateTransformer
        self.coord_transformer = CoordinateTransformer(config.time_range)
        
        # Device for tensor operations
        self.device = torch.device(config.device)
    
    def load_dataset(self) -> Dict:
        """
        Load and process the complete angiosperm dataset.

        Returns:
            Dictionary with processed tensors ready for DeepEarth
        """
        # Check for cached version first
        cache_dir = Path(self.config.cache_dir) / 'flowering_dataset'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / 'processed_data.pt'
        cache_metadata = cache_dir / 'metadata.json'

        # Generate cache key based on config
        import hashlib
        cache_key = hashlib.md5(str({
            'coordinate_system': self.config.coordinate_system,
            'time_range': self.config.time_range,
            'regenerate_cache': self.config.regenerate_cache
        }).encode()).hexdigest()

        # Check if cache exists and is valid
        if cache_file.exists() and cache_metadata.exists() and not self.config.regenerate_cache:
            try:
                with open(cache_metadata, 'r') as f:
                    cached_meta = json.load(f)

                if cached_meta.get('cache_key') == cache_key:
                    print(f"\n✓ Loading preprocessed data from cache: {cache_file}")
                    result = torch.load(cache_file, weights_only=False)  # Can't use weights_only with dict
                    print(f"  Loaded {result['n_samples']:,} samples from cache")
                    return result
                else:
                    print(f"\n  Cache key mismatch, regenerating...")
            except Exception as e:
                print(f"\n  Cache loading failed: {e}, regenerating...")

        print(f"\nLoading angiosperm dataset files...")
        
        # ═══════════════════════════════════════════════════════════
        # Step 1: Validate and load all required files
        # ═══════════════════════════════════════════════════════════
        
        required_files = {
            'observations': self.data_dir / 'angiosperm_observations.csv',
            'species': self.data_dir / 'angiosperms.csv',
            'vision_embeddings': self.data_dir / 'angiosperm_bioclip2_vision.pt',
            'alphaearth_embeddings': self.data_dir / 'angiosperm_alphaearth_embeddings.pt',
            'species_embeddings': self.data_dir / 'angiosperm_species_bioclip.pt'
        }
        
        # Check all files exist
        for name, path in required_files.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"Required file missing: {path}\n"
                    f"Expected files in {self.data_dir}:\n"
                    f"  - angiosperm_observations.csv\n"
                    f"  - angiosperms.csv\n"
                    f"  - angiosperm_bioclip2_vision.pt\n"
                    f"  - angiosperm_alphaearth_embeddings.pt\n"
                    f"  - angiosperm_species_bioclip.pt"
                )
            print(f"  ✓ Found {name}: {path.name}")
        
        # ═══════════════════════════════════════════════════════════
        # Step 2: Load CSV data
        # ═══════════════════════════════════════════════════════════
        
        print(f"\nLoading observation data...")
        observations_df = pd.read_csv(required_files['observations'])
        print(f"  Loaded {len(observations_df):,} observations")
        
        print(f"\nLoading species data...")
        species_df = pd.read_csv(required_files['species'])
        print(f"  Loaded {len(species_df):,} species")
        
        # Create species name to embedding index mapping
        species_to_embedding = dict(zip(
            species_df['species_name'].values,
            species_df['species_bioclip_embedding_index'].values
        ))
        
        # ═══════════════════════════════════════════════════════════
        # Step 3: Load pre-computed embeddings
        # ═══════════════════════════════════════════════════════════
        
        print(f"\nLoading embedding tensors...")
        
        # BioCLIP vision embeddings
        vision_data = torch.load(required_files['vision_embeddings'], weights_only=True)
        if isinstance(vision_data, dict) and 'embeddings' in vision_data:
            vision_embeddings = vision_data['embeddings']
        else:
            vision_embeddings = vision_data
        print(f"  BioCLIP vision: {vision_embeddings.shape} ({vision_embeddings.dtype})")
        
        # AlphaEarth embeddings
        alphaearth_data = torch.load(required_files['alphaearth_embeddings'], weights_only=True)
        if isinstance(alphaearth_data, dict) and 'embeddings' in alphaearth_data:
            alphaearth_embeddings = alphaearth_data['embeddings']
        else:
            alphaearth_embeddings = alphaearth_data
        
        # Convert int8 to float32 if needed
        if alphaearth_embeddings.dtype == torch.int8:
            print(f"  Converting AlphaEarth from int8 to float32...")
            alphaearth_embeddings = alphaearth_embeddings.float() / 127.0
        print(f"  AlphaEarth: {alphaearth_embeddings.shape} ({alphaearth_embeddings.dtype})")
        
        # BioCLIP species text embeddings
        species_data = torch.load(required_files['species_embeddings'], weights_only=True)
        if isinstance(species_data, dict) and 'embeddings' in species_data:
            species_embeddings = species_data['embeddings']
        else:
            species_embeddings = species_data
        print(f"  BioCLIP species: {species_embeddings.shape} ({species_embeddings.dtype})")
        
        # ═══════════════════════════════════════════════════════════
        # Step 4: Process spatiotemporal coordinates
        # ═══════════════════════════════════════════════════════════
        
        print(f"\nProcessing spatiotemporal coordinates...")
        
        n_samples = len(observations_df)
        xyzt_list = []
        time_components_list = []
        
        for idx in tqdm(range(n_samples), desc="Processing coordinates"):
            row = observations_df.iloc[idx]
            
            # Extract coordinates
            lat = float(row['latitude'])
            lon = float(row['longitude'])
            elev = float(row['elevation']) if pd.notna(row['elevation']) else 0.0
            
            # Parse datetime - format is DD/MM/YYYY HH:MM:SS
            dt_str = row['datetime']
            try:
                dt = pd.to_datetime(dt_str, format='%d/%m/%Y %H:%M:%S')
            except ValueError as e:
                raise ValueError(
                    f"Failed to parse datetime at row {idx}: '{dt_str}'\n"
                    f"Expected format: DD/MM/YYYY HH:MM:SS\n"
                    f"Error: {e}"
                )
            
            # Normalize time - convert pandas Timestamp to datetime if needed
            if hasattr(dt, 'to_pydatetime'):
                dt = dt.to_pydatetime()
            t_norm = self.coord_transformer.normalize_time(dt)
            t_day, t_year, t_hist = self.coord_transformer.time_to_components(dt)
            
            # Apply coordinate transformation
            if self.config.coordinate_system == 'ecef':
                x, y, z = self.coord_transformer.latlon_to_ecef(lat, lon, elev)
                x, y, z = self.coord_transformer.normalize_ecef(
                    torch.tensor(x), torch.tensor(y), torch.tensor(z)
                )
                x, y, z = x.item(), y.item(), z.item()
            else:
                x, y, z = lat, lon, elev
            
            xyzt_list.append([x, y, z, t_norm])
            time_components_list.append([t_day, t_year, t_hist])
        
        xyzt_tensor = torch.tensor(xyzt_list, dtype=torch.float32)
        time_components_tensor = torch.tensor(time_components_list, dtype=torch.float32)
        
        print(f"  Coordinates shape: {xyzt_tensor.shape}")
        print(f"  Time components shape: {time_components_tensor.shape}")
        
        # ═══════════════════════════════════════════════════════════
        # Step 5: Organize embeddings by encoder
        # ═══════════════════════════════════════════════════════════
        
        print(f"\nOrganizing embeddings by encoder...")
        
        # We'll create separate entries for each encoder
        # This preserves the original embedding dimensions
        
        encoded_data = {
            0: [],  # AlphaEarth
            2: [],  # BioCLIP (both vision and text)
            3: []   # PhenoVision (flowering probability)
        }
        
        dataset_modality_encoder_list = []
        encoded_file_indices = []
        encoded_row_indices = []
        
        for idx in tqdm(range(n_samples), desc="Organizing embeddings"):
            row = observations_df.iloc[idx]
            
            # ───────────────────────────────────────────────────────
            # AlphaEarth embedding (D1, M1, E1)
            # ───────────────────────────────────────────────────────
            
            alphaearth_idx = int(row['alphaearth_embedding_index'])
            if alphaearth_idx < 0 or alphaearth_idx >= len(alphaearth_embeddings):
                raise IndexError(
                    f"Invalid AlphaEarth embedding index at row {idx}: {alphaearth_idx}\n"
                    f"Valid range: [0, {len(alphaearth_embeddings)-1}]"
                )
            
            encoded_data[0].append(alphaearth_embeddings[alphaearth_idx])
            dataset_modality_encoder_list.append([0, 0, 0])  # D1, M1, E1
            encoded_file_indices.append(0)
            encoded_row_indices.append(len(encoded_data[0]) - 1)
            
            # ───────────────────────────────────────────────────────
            # BioCLIP vision embedding (D2, M2, E3) - if available
            # ───────────────────────────────────────────────────────
            
            vision_idx = int(row['vision_embedding_index']) if pd.notna(row['vision_embedding_index']) else -1
            
            if vision_idx >= 0:
                if vision_idx >= len(vision_embeddings):
                    raise IndexError(
                        f"Invalid vision embedding index at row {idx}: {vision_idx}\n"
                        f"Valid range: [0, {len(vision_embeddings)-1}]"
                    )
                
                encoded_data[2].append(vision_embeddings[vision_idx])
                dataset_modality_encoder_list.append([1, 1, 2])  # D2, M2, E3
                encoded_file_indices.append(2)
                encoded_row_indices.append(len(encoded_data[2]) - 1)
            
            # ───────────────────────────────────────────────────────
            # BioCLIP species text embedding (D2, M3, E3)
            # ───────────────────────────────────────────────────────
            
            species_name = row['species_name']
            if species_name not in species_to_embedding:
                raise KeyError(
                    f"Species '{species_name}' at row {idx} not found in species table"
                )
            
            species_embed_idx = species_to_embedding[species_name]
            if species_embed_idx >= len(species_embeddings):
                raise IndexError(
                    f"Invalid species embedding index for '{species_name}': {species_embed_idx}\n"
                    f"Valid range: [0, {len(species_embeddings)-1}]"
                )
            
            encoded_data[2].append(species_embeddings[species_embed_idx])
            dataset_modality_encoder_list.append([1, 2, 2])  # D2, M3, E3
            encoded_file_indices.append(2)
            encoded_row_indices.append(len(encoded_data[2]) - 1)
            
            # ───────────────────────────────────────────────────────
            # PhenoVision flowering probability (D2, M2, E4)
            # ───────────────────────────────────────────────────────
            
            flowering_prob = float(row['flowering_prob'])
            if not (0.0 <= flowering_prob <= 1.0):
                raise ValueError(
                    f"Invalid flowering probability at row {idx}: {flowering_prob}\n"
                    f"Expected range: [0.0, 1.0]"
                )
            
            # Store as 1D tensor
            encoded_data[3].append(torch.tensor([flowering_prob], dtype=torch.float32))
            dataset_modality_encoder_list.append([1, 1, 3])  # D2, M2, E4
            encoded_file_indices.append(3)
            encoded_row_indices.append(len(encoded_data[3]) - 1)
        
        # ═══════════════════════════════════════════════════════════
        # Step 6: Stack embeddings by encoder
        # ═══════════════════════════════════════════════════════════
        
        print(f"\nStacking embeddings by encoder...")
        
        encoded_tensors = {}
        for encoder_idx, embeddings_list in encoded_data.items():
            if embeddings_list:
                encoded_tensors[encoder_idx] = torch.stack(embeddings_list)
                encoder_name = list(self.ENCODER_IDS.keys())[encoder_idx]
                print(f"  Encoder {encoder_idx} ({encoder_name}): {encoded_tensors[encoder_idx].shape}")
        
        # ═══════════════════════════════════════════════════════════
        # Step 7: Prepare Earth4D placeholders (computed during training)
        # ═══════════════════════════════════════════════════════════

        print(f"\nPreparing Earth4D placeholders (will be computed during training)...")

        # Earth4D will be computed dynamically during training
        # We just need to set up the metadata for it
        # Using encoder_id = 1 for Earth4D

        # Add Earth4D entries to metadata
        for idx in range(n_samples):
            dataset_modality_encoder_list.append([0, 0, 1])  # D1, M1, E2
            encoded_file_indices.append(1)
            encoded_row_indices.append(idx)

        # Create placeholder for Earth4D - just store the xyzt coordinates
        # The actual Earth4D encoding will happen in the forward pass
        encoded_tensors[1] = xyzt_tensor  # Store coordinates, not embeddings
        print(f"  Encoder 1 (earth4d): Will compute dynamically from {encoded_tensors[1].shape}")
        
        # ═══════════════════════════════════════════════════════════
        # Step 8: Extract target variable
        # ═══════════════════════════════════════════════════════════
        
        print(f"\nExtracting target variables...")
        
        flowering_probs = torch.tensor(
            observations_df['flowering_prob'].values,
            dtype=torch.float32
        )
        
        print(f"  Flowering probabilities: {flowering_probs.shape}")
        print(f"  Mean: {flowering_probs.mean():.3f}")
        print(f"  Std: {flowering_probs.std():.3f}")
        print(f"  Min: {flowering_probs.min():.3f}")
        print(f"  Max: {flowering_probs.max():.3f}")
        
        # ═══════════════════════════════════════════════════════════
        # Step 9: Assemble final dataset
        # ═══════════════════════════════════════════════════════════
        
        print(f"\nAssembling final dataset...")
        
        result = {
            # Core data
            'xyzt': xyzt_tensor,
            'time_components': time_components_tensor,
            'dataset_modality_encoder': torch.tensor(dataset_modality_encoder_list, dtype=torch.int16),
            'encoded_data': encoded_tensors,
            'encoded_file_indices': torch.tensor(encoded_file_indices, dtype=torch.int16),
            'encoded_row_indices': torch.tensor(encoded_row_indices, dtype=torch.int64),
            
            # Mappings
            'dataset_map': self.DATASET_IDS,
            'modality_map': self.MODALITY_IDS,
            'encoder_map': self.ENCODER_IDS,
            
            # Metadata
            'n_samples': n_samples,
            'n_observations': len(dataset_modality_encoder_list),  # Multiple per sample
            'spatial_dims': 3,
            
            # Target and reference data
            'target': flowering_probs,
            'species_names': observations_df['species_name'].values,
            'species_ids': observations_df['species_id'].values,
            'observation_ids': observations_df['occurrence_id'].values,
            'geospatial_uncertainty': observations_df['geospatial_uncertainty'].values
        }
        
        print(f"\n{'='*70}")
        print(f"Dataset Loading Complete")
        print(f"{'='*70}")
        print(f"  Observations: {n_samples:,}")
        print(f"  Total entries: {len(dataset_modality_encoder_list):,}")
        print(f"  Unique species: {len(np.unique(result['species_names'])):,}")
        print(f"  Encoders loaded: {len(encoded_tensors)}")
        print(f"  Valid combinations: {len(set(map(tuple, dataset_modality_encoder_list)))}")
        
        # Validate all combinations are valid
        invalid_combos = []
        for combo in set(map(tuple, dataset_modality_encoder_list)):
            if combo not in self.VALID_COMBINATIONS:
                invalid_combos.append(combo)
        
        if invalid_combos:
            raise ValueError(
                f"Invalid (dataset, modality, encoder) combinations found:\n"
                f"{invalid_combos}\n"
                f"Valid combinations: {self.VALID_COMBINATIONS}"
            )
        
        print(f"  ✓ All combinations valid")
        print(f"{'='*70}")

        # Save to cache for next time
        print(f"\nSaving preprocessed data to cache...")
        try:
            # Save the main data
            torch.save(result, cache_file)

            # Save metadata
            meta_to_save = {
                'cache_key': cache_key,
                'n_samples': result['n_samples'],
                'n_observations': result['n_observations'],
                'created_at': str(datetime.now()),
                'config': {
                    'coordinate_system': self.config.coordinate_system,
                    'time_range': self.config.time_range
                }
            }
            with open(cache_metadata, 'w') as f:
                json.dump(meta_to_save, f, indent=2)

            cache_size_mb = cache_file.stat().st_size / (1024 * 1024)
            print(f"  ✓ Cached {cache_size_mb:.1f} MB to {cache_file}")
        except Exception as e:
            print(f"  ⚠ Failed to save cache: {e}")

        print(f"{'='*70}\n")

        return result
