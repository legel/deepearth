"""
Data cache module for DeepEarth Dashboard.

Central data cache for managing and caching of:
- Biodiversity observations from HuggingFace dataset
- Vision embeddings via memory-mapped files
- Language embeddings for species
- Precomputed UMAP projections
- Grid-based spatial statistics
"""

import torch
import numpy as np
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
from datetime import datetime
from collections import defaultdict, OrderedDict
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging
import traceback
import hdbscan
import hashlib
import pickle

from huggingface_data_loader import HuggingFaceDataLoader
from mmap_embedding_loader import MMapEmbeddingLoader
from umap_optimized import OptimizedUMAP
from utils.colors import generate_hsv_colors
from vision.attention_utils import generate_attention_overlay

logger = logging.getLogger(__name__)


class UnifiedDataCache:
    """
    Central data cache for the dashboard.
    
    Manages loading and caching of:
    - Biodiversity observations from HuggingFace dataset
    - Vision embeddings via memory-mapped files
    - Language embeddings for species
    - Precomputed UMAP projections
    - Grid-based spatial statistics
    """
    
    def __init__(self, config_path):
        """Initialize data loaders and caches"""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize HuggingFace data loader for observations and language embeddings
        self.loader = HuggingFaceDataLoader(str(config_path))
        
        # Initialize memory-mapped loader for fast vision embedding access
        try:
            # Use absolute paths to avoid working directory issues
            dashboard_dir = Path(__file__).parent
            embeddings_file = dashboard_dir / "embeddings.mmap"
            index_db = dashboard_dir / "embeddings_index.db"
            
            self.mmap_loader = MMapEmbeddingLoader(
                embeddings_file=str(embeddings_file),
                index_db=str(index_db)
            )
            logger.info("✅ Memory-mapped vision embedding loader initialized")
        except Exception as e:
            logger.warning(f"⚠️ Memory-mapped loader failed to initialize: {e}")
            logger.warning("Falling back to HuggingFace parquet loader for vision embeddings")
            self.mmap_loader = None
        
        # Cache for computed results with LRU behavior
        self.umap_cache = OrderedDict()
        self.grid_cache = OrderedDict()
        self.max_cache_size = 1000
        self.precomputed_language_umap = None
        self.language_clusters = None
        self.precomputed_vision_umap = None  # Cache for unfiltered vision UMAP
        
        # Progress tracking for long operations
        self.current_progress = None
    
    def _manage_cache_size(self, cache_dict, max_size=None):
        """
        Maintain cache size using LRU eviction policy.
        
        Args:
            cache_dict: OrderedDict cache to manage
            max_size: Maximum cache size (defaults to self.max_cache_size)
        """
        max_size = max_size or self.max_cache_size
        while len(cache_dict) > max_size:
            # Remove oldest item (LRU)
            cache_dict.popitem(last=False)
        
    def load_observations(self):
        """Load biodiversity observations from HuggingFace dataset"""
        return self.loader.observations
    
    def load_vision_metadata(self):
        """Load vision embeddings metadata"""
        return self.loader.vision_index
    
    def get_language_embedding(self, taxon_id):
        """
        Load language embedding for a taxon.
        
        Args:
            taxon_id: Species identifier
            
        Returns:
            numpy.ndarray: DeepSeek-V3 embedding (7,168 dimensions)
        """
        return self.loader.get_language_embedding(str(taxon_id))
    
    def get_vision_embedding(self, gbif_id, taxon_id, image_num=1):
        """
        Load vision embedding for an observation.
        
        Attempts to use memory-mapped loader first for performance,
        falls back to HuggingFace loader if unavailable.
        
        Args:
            gbif_id: Global Biodiversity Information Facility ID
            taxon_id: Species identifier (used for fallback)
            image_num: Image number (for multi-image observations)
            
        Returns:
            torch.Tensor: V-JEPA-2 embedding (6,488,064 dimensions)
        """
        # Try memory-mapped loader first for best performance
        if self.mmap_loader:
            try:
                embedding = self.mmap_loader.get_vision_embedding(gbif_id)
                if embedding is not None:
                    return embedding
            except Exception as e:
                logger.warning(f"Memory-mapped loader failed for GBIF {gbif_id}: {e}")
        
        # Fall back to HuggingFace loader
        embedding = self.loader.get_vision_embedding(gbif_id, image_num)
        return embedding
    
    def compute_and_cache_language_umap_clusters(self, progress_callback=None):
        """
        Precompute UMAP projection and HDBSCAN clusters for all language embeddings.
        
        This computation is expensive (~30s) so results are cached to disk.
        Provides 3D UMAP coordinates and cluster assignments for all species.
        
        Args:
            progress_callback: Optional callback for progress updates
        """
        base_dir = Path(__file__).parent
        cache_path = base_dir / "cache" / "language_umap_clusters.pkl"
        cache_path.parent.mkdir(exist_ok=True)
        
        # Try to load from cache
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.precomputed_language_umap = cached_data['umap']
                    self.language_clusters = cached_data['clusters']
                    logger.info("Loaded precomputed UMAP and clusters from cache")
                    return
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        logger.info("Computing UMAP and HDBSCAN clusters for all language embeddings...")
        
        # Get all unique taxon IDs
        obs = self.load_observations()
        unique_taxa = obs['taxon_id'].unique()
        total_taxa = len(unique_taxa)
        
        # Load all embeddings with progress tracking
        embeddings = []
        valid_taxa = []
        for i, taxon_id in enumerate(unique_taxa):
            if progress_callback and i % 100 == 0:
                progress_callback(i, total_taxa, f"Loading embeddings: {i}/{total_taxa}")
            
            emb = self.get_language_embedding(taxon_id)
            if emb is not None:
                embeddings.append(emb)
                valid_taxa.append(taxon_id)
        
        if len(embeddings) < 10:
            logger.warning(f"Not enough embeddings for clustering: {len(embeddings)}")
            return
        
        embeddings = np.array(embeddings)
        
        if progress_callback:
            progress_callback(total_taxa, total_taxa, "Standardizing embeddings...")
        
        # Standardize embeddings to prevent any features from dominating
        # This is crucial when some embeddings (like Quercus) have very different scales
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        if progress_callback:
            progress_callback(total_taxa, total_taxa, "Computing UMAP projection...")
        
        # Compute UMAP on standardized embeddings
        n_samples = len(embeddings)
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=min(30, n_samples - 1),  # Moderate neighbors for balanced view
            min_dist=0.1,  # Standard distance for good clustering
            metric='euclidean',  # Euclidean on standardized data
            random_state=42
        )
        coords_3d = reducer.fit_transform(embeddings_scaled)
        
        # Compute HDBSCAN clusters for ecological communities
        # Adjust parameters for standardized UMAP space
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=3,  # Smaller minimum for more granular clusters
            min_samples=1,  # Allow single-linkage for better connectivity
            metric='euclidean',
            cluster_selection_method='eom',
            cluster_selection_epsilon=0.3,  # Lower epsilon for more clusters
            prediction_data=True
        )
        cluster_labels = clusterer.fit_predict(coords_3d)
        
        # Generate perceptually uniform colors for clusters
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        cluster_colors = generate_hsv_colors(n_clusters)
        
        # Create color mapping (gray for noise points)
        color_map = {}
        cluster_idx = 0
        for label in sorted(set(cluster_labels)):
            if label == -1:
                color_map[label] = "#666666"  # Gray for unclustered points
            else:
                color_map[label] = cluster_colors[cluster_idx]
                cluster_idx += 1
        
        # Store results with metadata
        result = []
        for i, taxon_id in enumerate(valid_taxa):
            taxon_data = obs[obs['taxon_id'] == taxon_id].iloc[0]
            result.append({
                'taxon_id': taxon_id,
                'name': taxon_data['taxon_name'],
                'x': float(coords_3d[i, 0]),
                'y': float(coords_3d[i, 1]),
                'z': float(coords_3d[i, 2]),
                'cluster': int(cluster_labels[i]),
                'color': color_map[cluster_labels[i]],
                'count': len(obs[obs['taxon_id'] == taxon_id])
            })
        
        self.precomputed_language_umap = result
        
        # Convert numpy types for JSON serialization
        cluster_sizes = {}
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            cluster_sizes[int(label)] = int(count)
        
        self.language_clusters = {
            'labels': cluster_labels.tolist(),
            'colors': {int(k): v for k, v in color_map.items()},
            'n_clusters': int(n_clusters),
            'cluster_sizes': cluster_sizes
        }
        
        # Save to cache
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'umap': result,
                    'clusters': self.language_clusters
                }, f)
            logger.info("Saved precomputed UMAP and clusters to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def compute_language_umap(self, taxon_ids=None):
        """
        Compute UMAP projection for a subset of language embeddings.
        
        Args:
            taxon_ids: List of taxon IDs to include (None for all)
            
        Returns:
            List of dicts with UMAP coordinates and metadata
        """
        cache_key = "all" if taxon_ids is None else "_".join(sorted(map(str, taxon_ids)))
        
        # Check cache and move to end for LRU
        if cache_key in self.umap_cache:
            # Move to end (most recently used)
            result = self.umap_cache.pop(cache_key)
            self.umap_cache[cache_key] = result
            return result
        
        logger.info(f"Computing language UMAP for {cache_key}...")
        
        # Get unique taxon IDs
        obs = self.load_observations()
        if taxon_ids is None:
            unique_taxa = obs['taxon_id'].unique()
        else:
            unique_taxa = [str(tid) for tid in taxon_ids]
        
        # Load embeddings
        embeddings = []
        valid_taxa = []
        for taxon_id in unique_taxa:
            emb = self.get_language_embedding(taxon_id)
            if emb is not None:
                embeddings.append(emb)
                valid_taxa.append(taxon_id)
        
        if len(embeddings) < 3:
            logger.warning(f"Not enough embeddings for UMAP: {len(embeddings)}")
            return None
        
        embeddings = np.array(embeddings)
        
        # Standardize embeddings first
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        # Compute UMAP on standardized embeddings
        n_samples = len(embeddings)
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=min(30, n_samples - 1),
            min_dist=0.1,
            metric='euclidean',  # Euclidean on standardized data
            random_state=42
        )
        coords_3d = reducer.fit_transform(embeddings_scaled)
        
        # Create result with metadata
        result = []
        for i, taxon_id in enumerate(valid_taxa):
            taxon_data = obs[obs['taxon_id'] == taxon_id].iloc[0]
            result.append({
                'taxon_id': taxon_id,
                'name': taxon_data['taxon_name'],
                'x': float(coords_3d[i, 0]),
                'y': float(coords_3d[i, 1]),
                'z': float(coords_3d[i, 2]),
                'count': len(obs[obs['taxon_id'] == taxon_id])
            })
        
        self.umap_cache[cache_key] = result
        self._manage_cache_size(self.umap_cache)
        
        return self.umap_cache[cache_key]
    
    def compute_vision_umap_for_region(self, bounds):
        """
        Compute UMAP projection for vision embeddings in a geographic region.
        
        Args:
            bounds: Dict with 'north', 'south', 'east', 'west' keys
            
        Returns:
            List of dicts with UMAP coordinates and metadata
        """
        obs = self.load_observations()
        vision_meta = self.load_vision_metadata()
        
        # Filter observations by geographic bounds
        mask = (
            (obs['latitude'] >= bounds['south']) &
            (obs['latitude'] <= bounds['north']) &
            (obs['longitude'] >= bounds['west']) &
            (obs['longitude'] <= bounds['east'])
        )
        region_obs = obs[mask]
        
        if len(region_obs) == 0:
            return []
        
        # Get vision embeddings for these observations
        embeddings = []
        metadata = []
        
        for _, row in region_obs.iterrows():
            gbif_id = row['gbif_id']
            taxon_id = row['taxon_id']
            
            # Check if we have vision data
            if row.get('has_vision', False):
                emb = self.get_vision_embedding(gbif_id, taxon_id, 1)
                if emb is not None:
                    # Reshape and extract spatial features
                    emb_4d = self.loader.reshape_vision_embedding(emb)
                    mean_features = self.loader.get_image_level_embedding(emb_4d)
                    
                    embeddings.append(mean_features)
                    metadata.append({
                        'gbif_id': int(gbif_id),
                        'taxon_id': taxon_id,
                        'taxon_name': row['taxon_name'],
                        'lat': row['latitude'],
                        'lon': row['longitude'],
                        'eventDate': row.get('eventDate', None),
                        'year': row.get('year', None),
                        'month': row.get('month', None),
                        'day': row.get('day', None),
                        'hour': row.get('hour', None)
                    })
        
        if len(embeddings) < 3:
            return []
        
        embeddings = np.array(embeddings)
        
        # Compute UMAP
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=min(15, len(embeddings) - 1),
            min_dist=0.1,
            random_state=42
        )
        coords_3d = reducer.fit_transform(embeddings)
        
        # Generate perceptually uniform colors for species
        unique_taxa = list(set(m['taxon_id'] for m in metadata))
        colors = generate_hsv_colors(len(unique_taxa))
        taxon_colors = {tid: colors[i] for i, tid in enumerate(unique_taxa)}
        
        # Create result
        result = []
        for i, meta in enumerate(metadata):
            # Create proper image ID format for frontend
            image_id = f"gbif_{meta['gbif_id']}_taxon_{meta['taxon_id']}_img_1"
            result.append({
                **meta,
                'gbif_id': image_id,  # Use formatted ID for frontend compatibility
                'x': float(coords_3d[i, 0]),
                'y': float(coords_3d[i, 1]),
                'z': float(coords_3d[i, 2]),
                'color': taxon_colors[meta['taxon_id']]
            })
        
        return result
    
    def get_grid_statistics(self, bounds, grid_size, temporal_filters=None):
        """
        Compute statistics for a geographic grid cell.
        
        Args:
            bounds: Geographic bounds dict
            grid_size: Size of grid cell in degrees
            temporal_filters: Optional temporal filtering dict
            
        Returns:
            Dict with species counts, temporal distribution, etc.
        """
        cache_key = f"{bounds}_{grid_size}_{temporal_filters}"
        
        # Check cache and move to end for LRU
        if cache_key in self.grid_cache:
            # Move to end (most recently used)
            result = self.grid_cache.pop(cache_key)
            self.grid_cache[cache_key] = result
            return result
        
        obs = self.load_observations()
        
        # Filter observations by bounds
        mask = (
            (obs['latitude'] >= bounds['south']) &
            (obs['latitude'] <= bounds['north']) &
            (obs['longitude'] >= bounds['west']) &
            (obs['longitude'] <= bounds['east'])
        )
        
        # Apply temporal filters if provided
        if temporal_filters:
            mask &= (obs['year'] >= temporal_filters['year_min']) & (obs['year'] <= temporal_filters['year_max'])
            # Handle missing month/hour values
            if 'month' in obs.columns:
                month_mask = obs['month'].isna() | ((obs['month'] >= temporal_filters['month_min']) & (obs['month'] <= temporal_filters['month_max']))
                mask &= month_mask
            if 'hour' in obs.columns:
                hour_mask = obs['hour'].isna() | ((obs['hour'] >= temporal_filters['hour_min']) & (obs['hour'] <= temporal_filters['hour_max']))
                mask &= hour_mask
        
        cell_obs = obs[mask]
        
        if len(cell_obs) == 0:
            return None
        
        # Compute statistics efficiently
        species_counts = cell_obs['taxon_id'].value_counts()
        yearly_counts = cell_obs.groupby('year').size()
        
        # Get unique taxon names efficiently
        taxon_names = cell_obs.drop_duplicates(subset=['taxon_id'])[['taxon_id', 'taxon_name']].set_index('taxon_id')['taxon_name'].to_dict()
        
        stats = {
            'total_observations': len(cell_obs),
            'total_species': len(species_counts),
            'species': [
                {
                    'taxon_id': str(tid),
                    'name': taxon_names.get(tid, 'Unknown'),
                    'count': int(count)
                }
                for tid, count in species_counts.items()
            ],
            'yearly_counts': [
                {'year': int(year), 'count': int(count)}
                for year, count in yearly_counts.items()
            ],
            'bounds': bounds
        }
        
        self.grid_cache[cache_key] = stats
        self._manage_cache_size(self.grid_cache)
        
        return self.grid_cache[cache_key]
    
    def compute_spatial_attention(self, features, temporal_mode='mean', compression='mean', visualization='l2norm'):
        """
        Compute spatial attention map from V-JEPA-2 vision features.
        
        V-JEPA-2 produces 8 temporal frames with 24x24 spatial patches,
        each with 1,408-dimensional features.
        
        Args:
            features: PyTorch tensor [6,488,064] flattened
            temporal_mode: 'mean' for static or 'temporal' for animation
            compression: Method to combine temporal frames ('mean', 'max', 'rms')
            visualization: Method to convert features to attention ('l2norm', 'pca1', 'variance', 'entropy')
            
        Returns:
            torch.Tensor: Attention map(s)
        """
        # Reshape flattened features to temporal-spatial structure
        features = features.view(8, 576, 1408)  # [T, S, D] = [8 frames, 24x24 patches, 1408 dims]
        
        if temporal_mode == 'mean':
            # Apply compression method across the 8 temporal frames
            if compression == 'mean':
                # Average across temporal dimension
                spatial_features = features.mean(dim=0)  # [576, 1408]
            elif compression == 'max':
                # Max pooling across temporal frames
                spatial_features, _ = features.max(dim=0)  # [576, 1408]
            elif compression == 'rms':
                # Root Mean Square (preserves magnitude information)
                spatial_features = torch.sqrt((features ** 2).mean(dim=0))  # [576, 1408]
            
            # Apply visualization method to convert features to scalar attention values
            attention = self._apply_visualization_method(spatial_features, visualization)
            
            # Normalize to [0, 1] for display
            attention_min = attention.min()
            attention_max = attention.max()
            attention = (attention - attention_min) / (attention_max - attention_min + 1e-8)
        else:
            # Apply visualization method to each temporal frame independently
            attention_frames = []
            for t in range(8):
                frame_features = features[t]  # [576, 1408]
                frame_attention = self._apply_visualization_method(frame_features, visualization)
                # Normalize each frame independently
                frame_min = frame_attention.min()
                frame_max = frame_attention.max()
                frame_attention = (frame_attention - frame_min) / (frame_max - frame_min + 1e-8)
                attention_frames.append(frame_attention)
            
            attention = torch.stack(attention_frames)  # [8, 576]
        
        return attention
    
    def _apply_visualization_method(self, spatial_features, visualization):
        """
        Apply the selected visualization method to spatial features.
        
        Args:
            spatial_features: torch.Tensor [576, 1408] spatial patches
            visualization: Method name
            
        Returns:
            torch.Tensor [576] attention values
        """
        if visualization == 'l2norm':
            # L2 norm: magnitude of feature vectors (default)
            attention = torch.norm(spatial_features, dim=-1)  # [576]
            
        elif visualization.startswith('pca'):
            # PCA components show principal directions of variation
            import time
            start_time = time.time()
            logger.info(f"🔧 Starting PCA computation for {visualization}")
            
            from sklearn.decomposition import PCA
            
            # Extract component number (e.g., 'pca1' -> 1)
            n_component = 1 if visualization == 'pca' else int(visualization[3:])
            
            # Convert to numpy for sklearn
            features_numpy = spatial_features.detach().cpu().numpy()
            logger.info(f"📊 PCA input shape: {features_numpy.shape}, dtype: {features_numpy.dtype}")
            
            # Need to fit PCA with at least as many components as requested
            pca = PCA(n_components=max(n_component, 1), svd_solver='randomized')  # Use randomized for speed
            pca_features = pca.fit_transform(features_numpy)
            logger.info(f"✅ PCA computed in {time.time() - start_time:.2f}s")
            
            # Get the requested component (0-indexed)
            component_idx = n_component - 1
            if component_idx < pca_features.shape[1]:
                attention = torch.from_numpy(pca_features[:, component_idx])
            else:
                # If requested component doesn't exist, use the last available
                attention = torch.from_numpy(pca_features[:, -1])
            
        elif visualization == 'variance':
            # Feature variance: shows diversity of activations
            attention = spatial_features.var(dim=-1)
            
        elif visualization == 'entropy':
            # Feature entropy: information content
            # Normalize features to probabilities
            features_positive = torch.abs(spatial_features) + 1e-10
            features_normalized = features_positive / features_positive.sum(dim=-1, keepdim=True)
            # Compute entropy
            entropy = -(features_normalized * torch.log(features_normalized)).sum(dim=-1)
            attention = entropy
            
        return attention