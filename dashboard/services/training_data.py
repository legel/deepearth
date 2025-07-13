"""
Training data services for DeepEarth ML pipelines.

Provides efficient batch loading of multimodal biodiversity data for PyTorch training.
Supports both direct Python import and Flask API integration for maximum flexibility.

    🔬 Research Data ──► 🧠 ML Tensors ──► 🚀 Training Pipeline
    
    OBSERVATION_ID → Species, Images, Coordinates, Time, Embeddings
"""

import logging
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


def get_training_batch(
    cache, 
    observation_ids: List[str],
    include_vision: bool = True,
    include_language: bool = True,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    🎯 Core Training Data Pipeline
    
    Efficiently loads multimodal training data for a batch of observations.
    Returns PyTorch-ready tensors optimized for ML training workflows.
    
    Args:
        cache: UnifiedDataCache instance for data access
        observation_ids: List of OBSERVATION_IDs in format "{gbif_id}_{image_idx}"  
        include_vision: Whether to load vision embeddings (8×24×24×1408)
        include_language: Whether to load language embeddings (S×7168)
        device: PyTorch device for tensor placement ('cpu', 'cuda')
        
    Returns:
        Dict containing:
            'species': List[str] - Latin species names
            'image_urls': List[str] - Direct image URLs  
            'locations': torch.Tensor - GPS coordinates [N, 3] (lat, lng, elevation)
            'timestamps': torch.Tensor - Time data [N, 6] (year, month, day, hour, min, sec)
            'language_embeddings': torch.Tensor - Species embeddings [N, S, 7168] 
            'vision_embeddings': torch.Tensor - Vision embeddings [N, 8, 24, 24, 1408]
            'metadata': Dict - Additional metadata for debugging
            
    Raises:
        ValueError: If observation_ids format is invalid or data is missing
        RuntimeError: If batch loading fails
    """
    logger.info(f"Loading training batch: {len(observation_ids)} observations")
    
    # Parse OBSERVATION_IDs to extract GBIF IDs and image indices
    gbif_ids, image_indices = _parse_observation_ids(observation_ids)
    
    # Load observations data for the batch
    observations = _load_observations_batch(cache, gbif_ids)
    
    # Initialize result containers
    batch_data = {
        'species': [],
        'image_urls': [],
        'locations': [],
        'timestamps': [],
        'metadata': {
            'observation_ids': observation_ids,
            'gbif_ids': gbif_ids,
            'batch_size': len(observation_ids),
            'loaded_at': datetime.now().isoformat()
        }
    }
    
    # Process each observation in batch
    for i, (obs_id, gbif_id, img_idx) in enumerate(zip(observation_ids, gbif_ids, image_indices)):
        
        obs_row = observations[observations['gbif_id'] == gbif_id].iloc[0]
        
        # 1. Species (Latin name)
        species_name = obs_row['taxon_name']
        batch_data['species'].append(species_name)
        
        # 2. Image URL (specific image for this OBSERVATION_ID)
        image_urls = obs_row['image_urls']
        if isinstance(image_urls, np.ndarray) and len(image_urls) >= img_idx:
            image_url = image_urls[img_idx - 1]  # img_idx is 1-based
        else:
            image_url = image_urls[0] if len(image_urls) > 0 else ""
        batch_data['image_urls'].append(image_url)
        
        # 3. Location (Latitude, Longitude, Elevation)
        lat = float(obs_row['latitude'])
        lng = float(obs_row['longitude']) 
        # Note: Elevation not currently in dataset, defaulting to 0.0
        elevation = 0.0  # TODO: Add elevation data when available
        batch_data['locations'].append([lat, lng, elevation])
        
        # 4. Time (Year, Month, Day, Hour, Minute, Second)
        timestamp = [
            int(obs_row.get('year', 0)),
            int(obs_row.get('month', 0)), 
            int(obs_row.get('day', 0)),
            int(obs_row.get('hour', 0)),
            int(obs_row.get('minute', 0)),
            int(obs_row.get('second', 0))
        ]
        batch_data['timestamps'].append(timestamp)
    
    # Convert lists to tensors
    batch_data['locations'] = torch.tensor(batch_data['locations'], dtype=torch.float32, device=device)
    batch_data['timestamps'] = torch.tensor(batch_data['timestamps'], dtype=torch.int32, device=device)
    
    # 5. Species embeddings (Language)
    if include_language:
        batch_data['language_embeddings'] = _load_language_embeddings_batch(
            cache, observations, gbif_ids, device
        )
    
    # 6. Vision embeddings
    if include_vision:
        batch_data['vision_embeddings'] = _load_vision_embeddings_batch(
            cache, observations, gbif_ids, device
        )
    
    logger.info(f"Successfully loaded batch with {len(batch_data['species'])} observations")
    return batch_data


def create_observation_id(gbif_id: int, image_index: int = 1) -> str:
    """
    🏷️ OBSERVATION_ID Creation
    
    Creates standardized OBSERVATION_ID from GBIF ID and image index.
    Format: "{gbif_id}_{image_idx}" for human-readable identification.
    
    Args:
        gbif_id: GBIF occurrence identifier
        image_index: Image number for this observation (1-based)
        
    Returns:
        str: Formatted OBSERVATION_ID
    """
    return f"{gbif_id}_{image_index}"


def parse_observation_id(observation_id: str) -> Tuple[int, int]:
    """
    🔍 OBSERVATION_ID Parsing
    
    Extracts GBIF ID and image index from OBSERVATION_ID string.
    
    Args:
        observation_id: Formatted OBSERVATION_ID string
        
    Returns:
        Tuple[int, int]: (gbif_id, image_index)
        
    Raises:
        ValueError: If observation_id format is invalid
    """
    try:
        parts = observation_id.split('_')
        if len(parts) != 2:
            raise ValueError(f"Invalid OBSERVATION_ID format: {observation_id}")
        
        gbif_id = int(parts[0])
        image_index = int(parts[1])
        
        if image_index < 1:
            raise ValueError(f"Image index must be >= 1, got: {image_index}")
            
        return gbif_id, image_index
        
    except (ValueError, IndexError) as e:
        raise ValueError(f"Failed to parse OBSERVATION_ID '{observation_id}': {e}")


def get_available_observation_ids(
    cache,
    has_vision: bool = True,
    has_language: bool = True,
    limit: Optional[int] = None
) -> List[str]:
    """
    📋 Available Observations Discovery
    
    Returns list of all valid OBSERVATION_IDs in the dataset that meet criteria.
    Useful for train/test split generation and batch sampling.
    
    Args:
        cache: UnifiedDataCache instance
        has_vision: Only include observations with vision embeddings
        has_language: Only include observations with language embeddings  
        limit: Maximum number of observation_ids to return
        
    Returns:
        List[str]: Available OBSERVATION_IDs
    """
    observations = cache.load_observations()
    
    # Apply filters
    if has_vision:
        observations = observations[observations['has_vision'] == True]
    
    if has_language:
        # Language embeddings should be available for all observations
        observations = observations[observations['language_embedding'].notna()]
    
    observation_ids = []
    
    for _, row in observations.iterrows():
        gbif_id = row['gbif_id']
        
        # For now, assume 1 image per observation (can extend later)
        # TODO: Parse actual number of images when multi-image support is added
        num_images = 1
        
        for img_idx in range(1, num_images + 1):
            obs_id = create_observation_id(gbif_id, img_idx)
            observation_ids.append(obs_id)
            
            if limit and len(observation_ids) >= limit:
                break
        
        if limit and len(observation_ids) >= limit:
            break
    
    logger.info(f"Found {len(observation_ids)} available observation IDs")
    return observation_ids


def _parse_observation_ids(observation_ids: List[str]) -> Tuple[List[int], List[int]]:
    """Parse batch of OBSERVATION_IDs into GBIF IDs and image indices."""
    gbif_ids = []
    image_indices = []
    
    for obs_id in observation_ids:
        gbif_id, img_idx = parse_observation_id(obs_id)
        gbif_ids.append(gbif_id)
        image_indices.append(img_idx)
    
    return gbif_ids, image_indices


def _load_observations_batch(cache, gbif_ids: List[int]) -> pd.DataFrame:
    """Load observations DataFrame for batch of GBIF IDs."""
    observations = cache.load_observations()
    
    # Filter to requested GBIF IDs
    batch_obs = observations[observations['gbif_id'].isin(gbif_ids)]
    
    if len(batch_obs) != len(set(gbif_ids)):
        missing_ids = set(gbif_ids) - set(batch_obs['gbif_id'].values)
        logger.warning(f"Missing observations for GBIF IDs: {missing_ids}")
    
    return batch_obs


def _load_language_embeddings_batch(
    cache, 
    observations: pd.DataFrame, 
    gbif_ids: List[int],
    device: str
) -> torch.Tensor:
    """Load language embeddings for batch of observations."""
    embeddings = []
    
    for gbif_id in gbif_ids:
        obs_row = observations[observations['gbif_id'] == gbif_id].iloc[0]
        lang_emb = obs_row['language_embedding']
        
        if isinstance(lang_emb, np.ndarray):
            embeddings.append(lang_emb)
        else:
            # Fallback: load via taxon_id
            taxon_id = obs_row['taxon_id']
            lang_emb = cache.get_language_embedding(taxon_id)
            embeddings.append(lang_emb)
    
    # Stack into batch tensor [N, 7168]
    # Note: For now assuming single token per species. Multi-token support can be added later.
    embeddings_tensor = torch.tensor(np.stack(embeddings), dtype=torch.float32, device=device)
    
    return embeddings_tensor


def _load_vision_embeddings_batch(
    cache, 
    observations: pd.DataFrame,
    gbif_ids: List[int],
    device: str
) -> torch.Tensor:
    """Load vision embeddings for batch of GBIF IDs."""
    embeddings = []
    
    for gbif_id in gbif_ids:
        try:
            # Get taxon_id for this observation
            obs_row = observations[observations['gbif_id'] == gbif_id].iloc[0]
            taxon_id = obs_row['taxon_id']
            
            # Load raw embedding with both required parameters
            vision_emb = cache.get_vision_embedding(gbif_id, taxon_id)
            
            if vision_emb is not None:
                # Reshape to 4D: [8, 24, 24, 1408]
                vision_emb_4d = cache.loader.reshape_vision_embedding(vision_emb)
                embeddings.append(vision_emb_4d)
            else:
                # Create zero tensor for missing embeddings
                zero_emb = np.zeros((8, 24, 24, 1408), dtype=np.float32)
                embeddings.append(zero_emb)
                logger.warning(f"Missing vision embedding for GBIF ID {gbif_id}, using zeros")
        
        except Exception as e:
            logger.error(f"Failed to load vision embedding for GBIF ID {gbif_id}: {e}")
            # Create zero tensor as fallback
            zero_emb = np.zeros((8, 24, 24, 1408), dtype=np.float32)
            embeddings.append(zero_emb)
    
    # Stack into batch tensor [N, 8, 24, 24, 1408]
    embeddings_tensor = torch.tensor(np.stack(embeddings), dtype=torch.float32, device=device)
    
    return embeddings_tensor