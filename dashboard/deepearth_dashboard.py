#!/usr/bin/env python3
"""
DeepEarth Dashboard - ML-Ready Visualization System

Interactive visualization and data indexing for the DeepEarth Self-Supervised 
Spatiotemporal Multimodality Simulator. This dashboard demonstrates multimodal
machine learning capabilities using biodiversity data.

Key Features:
- Memory-mapped tensor storage for direct ML pipeline integration
- Real-time embedding space visualization (vision and language)
- Spatiotemporal data indexing for training set construction
- Foundation for integrated ML control systems (planned)

Technical Stack:
- Vision: V-JEPA-2 embeddings (6.4M dimensions)
- Language: DeepSeek-V3 embeddings (7.2K dimensions)
- Performance: Sub-100ms retrieval via mmap indexing

Author: DeepEarth Project
License: MIT
Version: 1.0.0
"""

from flask import Flask, render_template, jsonify, send_file, request, send_from_directory, redirect
import torch
import numpy as np
from pathlib import Path
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import io
import base64
from datetime import datetime
from collections import defaultdict
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import colorsys
import logging
import traceback
import hdbscan
import hashlib
import pickle
from huggingface_data_loader import HuggingFaceDataLoader
from mmap_embedding_loader import MMapEmbeddingLoader

# Initialize Flask application
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration paths
BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "dataset_config.json"

# Load configuration
with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)

# Set data paths based on configuration
DATA_DIR = BASE_DIR / CONFIG['data_paths']['base_dir']


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
    
    def __init__(self):
        """Initialize data loaders and caches"""
        # Initialize HuggingFace data loader for observations and language embeddings
        self.loader = HuggingFaceDataLoader(str(CONFIG_PATH))
        
        # Initialize memory-mapped loader for fast vision embedding access
        try:
            self.mmap_loader = MMapEmbeddingLoader()
            logger.info("✅ Memory-mapped vision embedding loader initialized")
        except Exception as e:
            logger.warning(f"⚠️ Memory-mapped loader failed to initialize: {e}")
            logger.warning("Falling back to HuggingFace parquet loader for vision embeddings")
            self.mmap_loader = None
        
        # Cache for computed results
        self.umap_cache = {}
        self.grid_cache = {}
        self.precomputed_language_umap = None
        self.language_clusters = None
        self.precomputed_vision_umap = None  # Cache for unfiltered vision UMAP
        
        # Progress tracking for long operations
        self.current_progress = None
        
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
        cache_path = BASE_DIR / "cache" / "language_umap_clusters.pkl"
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
            progress_callback(total_taxa, total_taxa, "Computing UMAP projection...")
        
        # Compute UMAP with cosine distance for semantic similarity
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=min(15, len(embeddings) - 1),
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        coords_3d = reducer.fit_transform(embeddings)
        
        # Compute HDBSCAN clusters for ecological communities
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=3,
            min_samples=2,
            metric='euclidean',
            cluster_selection_method='eom',
            cluster_selection_epsilon=0.3,
            prediction_data=True
        )
        cluster_labels = clusterer.fit_predict(coords_3d)
        
        # Generate perceptually uniform colors for clusters
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        cluster_colors = generate_husl_colors(n_clusters)
        
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
        
        if cache_key not in self.umap_cache:
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
            
            # Compute UMAP
            reducer = umap.UMAP(
                n_components=3,
                n_neighbors=min(15, len(embeddings) - 1),
                min_dist=0.1,
                metric='cosine',
                random_state=42
            )
            coords_3d = reducer.fit_transform(embeddings)
            
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
                        'lon': row['longitude']
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
        colors = generate_husl_colors(len(unique_taxa))
        taxon_colors = {tid: colors[i] for i, tid in enumerate(unique_taxa)}
        
        # Create result
        result = []
        for i, meta in enumerate(metadata):
            result.append({
                **meta,
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
        
        if cache_key not in self.grid_cache:
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
            
            # Limit cache size
            if len(self.grid_cache) > 1000:
                # Remove oldest entries
                self.grid_cache.clear()
            
            self.grid_cache[cache_key] = stats
            
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
            from sklearn.decomposition import PCA
            
            # Extract component number (e.g., 'pca1' -> 1)
            n_component = 1 if visualization == 'pca' else int(visualization[3:])
            
            # Convert to numpy for sklearn
            features_numpy = spatial_features.detach().cpu().numpy()
            pca = PCA(n_components=n_component)
            pca_features = pca.fit_transform(features_numpy)
            # Get the requested component (0-indexed)
            attention = torch.from_numpy(pca_features[:, n_component-1])
            
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


def generate_husl_colors(n):
    """
    Generate n perceptually uniform colors using HSV color space.
    
    Args:
        n: Number of colors needed
        
    Returns:
        List of RGB color strings
    """
    colors = []
    for i in range(n):
        hue = i * 360.0 / n
        # Use high saturation and medium lightness for vivid, distinct colors
        rgb = colorsys.hsv_to_rgb(hue/360.0, 0.9, 0.8)
        colors.append(f"rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})")
    return colors


# Initialize global cache instance
cache = UnifiedDataCache()

# Precompute language UMAP and clusters on startup (in background)
import threading
logger.info("Initializing precomputed data...")

def precompute_with_progress():
    """Background task to precompute expensive operations"""
    try:
        if CONFIG['caching']['precompute_language_umap']:
            cache.compute_and_cache_language_umap_clusters()
    except Exception as e:
        logger.error(f"Error during precomputation: {e}")
        logger.error(traceback.format_exc())

precompute_thread = threading.Thread(target=precompute_with_progress)
precompute_thread.daemon = True
precompute_thread.start()


# Flask Routes
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html', config=CONFIG)


@app.route('/api/config')
def get_config():
    """Get dataset configuration"""
    return jsonify(CONFIG)


@app.route('/api/progress')
def get_progress():
    """Get current progress of long-running operations"""
    if cache.current_progress:
        return jsonify(cache.current_progress)
    return jsonify({'status': 'idle', 'message': 'No operations in progress'})


@app.route('/api/species_umap_colors')
def get_species_umap_colors():
    """
    Get RGB colors for all species based on their UMAP positions.
    
    This endpoint provides consistent colors across the map and 3D visualization
    by mapping UMAP coordinates to RGB values.
    """
    try:
        # Use precomputed language UMAP if available
        if cache.precomputed_language_umap:
            umap_data = cache.precomputed_language_umap
        else:
            # Compute if not available
            cache.compute_and_cache_language_umap_clusters()
            umap_data = cache.precomputed_language_umap
        
        if not umap_data:
            return jsonify({'error': 'UMAP data not available'}), 500
        
        # Extract coordinates
        coords = np.array([[pt['x'], pt['y'], pt['z']] for pt in umap_data])
        
        # Normalize coordinates to [0, 1] range
        coords_min = coords.min(axis=0)
        coords_max = coords.max(axis=0)
        coords_normalized = (coords - coords_min) / (coords_max - coords_min + 1e-8)
        
        # Convert to RGB (0-255)
        rgb_values = (coords_normalized * 255).astype(int)
        
        # Create taxon_id to RGB mapping
        taxon_colors = {}
        for i, pt in enumerate(umap_data):
            taxon_colors[pt['taxon_id']] = {
                'r': int(rgb_values[i, 0]),
                'g': int(rgb_values[i, 1]),
                'b': int(rgb_values[i, 2]),
                'hex': '#{:02x}{:02x}{:02x}'.format(
                    int(rgb_values[i, 0]),
                    int(rgb_values[i, 1]),
                    int(rgb_values[i, 2])
                )
            }
        
        return jsonify({
            'taxon_colors': taxon_colors,
            'total_species': len(taxon_colors)
        })
    except Exception as e:
        logger.error(f"Error in get_species_umap_colors: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/observations')
def get_observations():
    """Get all observations for map display"""
    try:
        obs = cache.load_observations()
        vision_meta = cache.load_vision_metadata()
        
        # Get unique gbif_ids that have vision embeddings
        vision_gbif_ids = set()
        if vision_meta is not None:
            vision_gbif_ids = set(vision_meta['gbif_id'].unique())
        
        # Prepare data for frontend - use list comprehension for efficiency
        data = []
        for _, row in obs.iterrows():
            has_vision = row.get('has_vision', False) or int(row['gbif_id']) in vision_gbif_ids
            
            data.append({
                'gbif_id': int(row['gbif_id']),
                'taxon_id': row['taxon_id'],
                'taxon_name': row['taxon_name'],
                'lat': row['latitude'],
                'lon': row['longitude'],
                'year': int(row['year']),
                'month': int(row['month']) if pd.notna(row['month']) else None,
                'day': int(row['day']) if pd.notna(row['day']) else None,
                'hour': int(row['hour']) if pd.notna(row['hour']) else None,
                'minute': int(row['minute']) if pd.notna(row['minute']) else None,
                'second': int(row['second']) if pd.notna(row['second']) else None,
                'has_vision': has_vision,
                'split': row.get('split', 'unknown')
            })
        
        return jsonify({
            'observations': data,
            'total': len(data),
            'bounds': {
                'north': float(obs['latitude'].max()),
                'south': float(obs['latitude'].min()),
                'east': float(obs['longitude'].max()),
                'west': float(obs['longitude'].min())
            }
        })
    except Exception as e:
        logger.error(f"Error in get_observations: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/observation/<int:gbif_id>')
def get_observation_details(gbif_id):
    """Get detailed information for a specific observation"""
    try:
        # Get observation using HuggingFaceDataLoader
        obs_data = cache.loader.get_observation(gbif_id)
        if obs_data is None:
            return jsonify({'error': 'Observation not found'}), 404
        
        taxon_id = obs_data['taxon_id']
        
        # Get images from image URLs
        images = []
        image_urls = obs_data.get('image_urls', [])
        if isinstance(image_urls, list):
            for i, url in enumerate(image_urls):
                images.append({
                    'filename': f"image_{i+1}.jpg",
                    'image_id': f"gbif_{gbif_id}_taxon_{taxon_id}_img_{i+1}",
                    'url': url,
                    'local_url': f"/api/image_proxy/{gbif_id}/{i+1}"
                })
        
        # Get vision embedding info
        has_vision = bool(obs_data.get('has_vision', False))
        
        # Get language embedding (exists for all taxa in v0.2.0)
        has_language = bool(obs_data.get('language_embedding') is not None)
        
        result = {
            'gbif_id': int(gbif_id),
            'taxon_id': taxon_id,
            'taxon_name': obs_data['taxon_name'],
            'location': {
                'latitude': float(obs_data['latitude']),
                'longitude': float(obs_data['longitude'])
            },
            'temporal': {
                'year': int(obs_data['year']),
                'month': int(obs_data['month']) if pd.notna(obs_data['month']) else None,
                'day': int(obs_data['day']) if pd.notna(obs_data['day']) else None,
                'hour': int(obs_data['hour']) if pd.notna(obs_data['hour']) else None,
                'minute': int(obs_data['minute']) if pd.notna(obs_data['minute']) else None,
                'second': int(obs_data['second']) if pd.notna(obs_data['second']) else None
            },
            'images': images,
            'has_vision_embedding': has_vision,
            'has_language_embedding': has_language,
            'split': obs_data.get('split', 'unknown')
        }
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in get_observation_details: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/image_proxy/<int:gbif_id>/<int:image_num>')
def get_image_proxy(gbif_id, image_num):
    """Proxy for serving images from remote URLs"""
    try:
        # Get observation
        obs_data = cache.loader.get_observation(gbif_id)
        if obs_data is None:
            return "Observation not found", 404
        
        # Get image URL from dataset
        image_urls = obs_data.get('image_urls', [])
        if isinstance(image_urls, list) and len(image_urls) >= image_num:
            url = image_urls[image_num - 1]
            return redirect(url)
        
        return "Image not found", 404
    except Exception as e:
        logger.error(f"Error serving image: {str(e)}")
        return "Error serving image", 500


@app.route('/api/language_embeddings/umap')
def get_language_umap():
    """Get UMAP projection of language embeddings with optional filtering"""
    try:
        # Check for precomputed full dataset
        use_precomputed = request.args.get('precomputed', 'true').lower() == 'true'
        force_recompute = request.args.get('force_recompute', 'false').lower() == 'true'
        taxon_ids = request.args.getlist('taxon_ids')
        
        # Check for geographic bounds
        north = request.args.get('north', type=float)
        south = request.args.get('south', type=float)
        east = request.args.get('east', type=float)
        west = request.args.get('west', type=float)
        
        if force_recompute:
            cache.precomputed_language_umap = None
            cache.language_clusters = None
            cache.compute_and_cache_language_umap_clusters()
        
        # If geographic bounds are provided, filter taxon IDs
        if north is not None and south is not None and east is not None and west is not None:
            obs = cache.load_observations()
            mask = (
                (obs['latitude'] >= south) & (obs['latitude'] <= north) &
                (obs['longitude'] >= west) & (obs['longitude'] <= east)
            )
            
            # Add temporal filters if provided
            year_min = request.args.get('year_min', type=int)
            year_max = request.args.get('year_max', type=int)
            month_min = request.args.get('month_min', type=int)
            month_max = request.args.get('month_max', type=int)
            hour_min = request.args.get('hour_min', type=int)
            hour_max = request.args.get('hour_max', type=int)
            
            if year_min is not None and year_max is not None:
                mask = mask & (obs['year'] >= year_min) & (obs['year'] <= year_max)
            
            if month_min is not None and month_max is not None:
                mask = mask & (obs['month'] >= month_min) & (obs['month'] <= month_max)
            
            if hour_min is not None and hour_max is not None:
                mask = mask & (obs['hour'] >= hour_min) & (obs['hour'] <= hour_max)
            
            filtered_obs = obs[mask]
            taxon_ids = filtered_obs['taxon_id'].unique().tolist()
            taxon_ids = [str(tid) for tid in taxon_ids]
            use_precomputed = False  # Don't use precomputed when filtering
        
        if use_precomputed and not taxon_ids and cache.precomputed_language_umap:
            # Return precomputed data with clusters
            return jsonify({
                'embeddings': cache.precomputed_language_umap,
                'clusters': cache.language_clusters,
                'total': len(cache.precomputed_language_umap),
                'precomputed': True
            })
        
        # Fall back to dynamic computation
        if taxon_ids:
            taxon_ids = [str(tid) for tid in taxon_ids]
        else:
            taxon_ids = None
        
        result = cache.compute_language_umap(taxon_ids)
        if result is None:
            return jsonify({'error': 'Not enough data for UMAP'}), 400
        
        return jsonify({
            'embeddings': result,
            'total': len(result),
            'precomputed': False
        })
    except Exception as e:
        logger.error(f"Error in get_language_umap: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/vision_features/<int:gbif_id>')
def get_vision_features(gbif_id):
    """Get vision features and attention maps for an observation"""
    try:
        obs = cache.load_observations()
        obs_data = obs[obs['gbif_id'] == gbif_id]
        
        if len(obs_data) == 0:
            return jsonify({'error': 'Observation not found'}), 404
        
        taxon_id = obs_data.iloc[0]['taxon_id']
        
        # Load vision embedding
        features = cache.get_vision_embedding(gbif_id, taxon_id, 1)
        if features is None:
            return jsonify({'error': 'Vision features not found'}), 404
        
        # Compute attention maps
        temporal_mode = request.args.get('temporal', 'mean')
        visualization = request.args.get('visualization', 'l2norm')
        
        # Compute attention using the cache method
        attention = cache.compute_spatial_attention(features, temporal_mode, 'mean', visualization)
        
        if temporal_mode == 'mean':
            # Generate visualization
            attention_img = generate_attention_overlay(attention)
            
            return jsonify({
                'mode': 'spatial',
                'attention_map': attention_img,
                'stats': {
                    'max': float(attention.max()),
                    'mean': float(attention.mean()),
                    'std': float(attention.std())
                }
            })
        else:
            # Return temporal sequence
            attention_frames = []
            for t in range(8):
                attention_img = generate_attention_overlay(attention[t])
                attention_frames.append(attention_img)
            
            return jsonify({
                'mode': 'temporal',
                'attention_frames': attention_frames,
                'num_frames': 8
            })
    except Exception as e:
        logger.error(f"Error in get_vision_features: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/grid_statistics')
def get_grid_statistics():
    """Get statistics for a geographic grid cell"""
    try:
        # Get bounds from query params
        north = float(request.args.get('north'))
        south = float(request.args.get('south'))
        east = float(request.args.get('east'))
        west = float(request.args.get('west'))
        grid_size = float(request.args.get('grid_size', 0.01))
        
        bounds = {
            'north': north,
            'south': south,
            'east': east,
            'west': west
        }
        
        # Get temporal filters
        temporal_filters = None
        year_min = request.args.get('year_min', type=int)
        if year_min is not None:
            temporal_filters = {
                'year_min': year_min,
                'year_max': request.args.get('year_max', type=int, default=2025),
                'month_min': request.args.get('month_min', type=int, default=1),
                'month_max': request.args.get('month_max', type=int, default=12),
                'hour_min': request.args.get('hour_min', type=int, default=0),
                'hour_max': request.args.get('hour_max', type=int, default=23)
            }
        
        stats = cache.get_grid_statistics(bounds, grid_size, temporal_filters)
        if stats is None:
            return jsonify({'error': 'No data in selected region'}), 404
        
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error in get_grid_statistics: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/vision_embeddings/available')
def get_available_vision_embeddings():
    """Get list of observations with vision embeddings that match filters"""
    try:
        # Get filter parameters
        north = float(request.args.get('north', 90))
        south = float(request.args.get('south', -90))
        east = float(request.args.get('east', 180))
        west = float(request.args.get('west', -180))
        max_images = int(request.args.get('max_images', 250))
        
        # Temporal filters
        year_min = request.args.get('year_min', type=int, default=2010)
        year_max = request.args.get('year_max', type=int, default=2025)
        month_min = request.args.get('month_min', type=int, default=1)
        month_max = request.args.get('month_max', type=int, default=12)
        hour_min = request.args.get('hour_min', type=int, default=0)
        hour_max = request.args.get('hour_max', type=int, default=23)
        
        # Load data
        obs = cache.load_observations()
        vision_meta = cache.load_vision_metadata()
        
        # Check if vision metadata exists
        if vision_meta is None or len(vision_meta) == 0:
            return jsonify({
                'count': 0,
                'observations': [],
                'error': 'No vision embeddings available'
            })
        
        # Filter observations
        mask = (
            (obs['latitude'] >= south) &
            (obs['latitude'] <= north) &
            (obs['longitude'] >= west) &
            (obs['longitude'] <= east) &
            (obs['year'] >= year_min) &
            (obs['year'] <= year_max)
        )
        
        # Handle potentially missing month/hour columns
        if 'month' in obs.columns:
            mask &= (obs['month'].isna() | ((obs['month'] >= month_min) & (obs['month'] <= month_max)))
        if 'hour' in obs.columns:
            mask &= (obs['hour'].isna() | ((obs['hour'] >= hour_min) & (obs['hour'] <= hour_max)))
        
        filtered_obs = obs[mask]
        
        # Find observations with vision embeddings
        obs_with_vision = filtered_obs[filtered_obs['has_vision'] == True]
        
        # Limit to max_images
        if len(obs_with_vision) > max_images:
            obs_with_vision = obs_with_vision.sample(n=max_images, random_state=42)
        
        # Return minimal metadata for preloading
        result = {
            'count': len(obs_with_vision),
            'observations': obs_with_vision[['gbif_id', 'taxon_id']].to_dict('records')
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error getting available vision embeddings: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/vision_embeddings/umap')
def get_vision_umap():
    """Get UMAP projection of vision embeddings with filtering"""
    try:
        # Get bounds
        north = float(request.args.get('north', 90))
        south = float(request.args.get('south', -90))
        east = float(request.args.get('east', 180))
        west = float(request.args.get('west', -180))
        
        bounds = {
            'north': north,
            'south': south,
            'east': east,
            'west': west
        }
        
        result = cache.compute_vision_umap_for_region(bounds)
        
        return jsonify({
            'embeddings': result,
            'total': len(result),
            'bounds': bounds,
            'species_colors': {}
        })
        
    except Exception as e:
        logger.error(f"Error in get_vision_umap: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/ecosystem_analysis')
def get_ecosystem_analysis():
    """Perform ecosystem community analysis for a region"""
    try:
        # Get bounds
        north = float(request.args.get('north'))
        south = float(request.args.get('south'))
        east = float(request.args.get('east'))
        west = float(request.args.get('west'))
        analysis_type = request.args.get('type', 'language')  # 'language' or 'vision'
        
        bounds = {
            'north': north,
            'south': south,
            'east': east,
            'west': west
        }
        
        if analysis_type == 'language':
            # Get taxa in region
            obs = cache.load_observations()
            mask = (
                (obs['latitude'] >= south) &
                (obs['latitude'] <= north) &
                (obs['longitude'] >= west) &
                (obs['longitude'] <= east)
            )
            region_taxa = obs[mask]['taxon_id'].unique()
            
            if len(region_taxa) < 3:
                return jsonify({'error': 'Not enough species in region'}), 400
            
            # Compute UMAP for these taxa
            result = cache.compute_language_umap(region_taxa)
            
            return jsonify({
                'type': 'language',
                'embeddings': result,
                'total_species': len(result)
            })
        else:
            # Vision embeddings analysis
            result = cache.compute_vision_umap_for_region(bounds)
            
            return jsonify({
                'type': 'vision',
                'embeddings': result,
                'total_observations': len(result)
            })
    except Exception as e:
        logger.error(f"Error in get_ecosystem_analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500


def generate_attention_overlay(attention_map, colormap='plasma', alpha=0.7):
    """
    Generate base64 encoded attention overlay image.
    
    Args:
        attention_map: PyTorch tensor or numpy array [576] or [24, 24]
        colormap: Matplotlib colormap name
        alpha: Transparency for overlay
        
    Returns:
        Base64 encoded PNG image string
    """
    from scipy.ndimage import zoom
    
    logger.info(f"🎨 Generating attention overlay: colormap={colormap}, alpha={alpha}")
    
    # Convert PyTorch tensor to numpy if needed
    if isinstance(attention_map, torch.Tensor):
        logger.info(f"🔄 Converting PyTorch tensor to numpy: {attention_map.shape}, dtype={attention_map.dtype}")
        attention_map = attention_map.detach().cpu().numpy()
    
    logger.info(f"📊 Attention map stats: shape={attention_map.shape}, min={attention_map.min():.3f}, max={attention_map.max():.3f}, mean={attention_map.mean():.3f}")
    
    # Reshape from flat [576] to spatial [24, 24] if needed
    if attention_map.ndim == 1 and len(attention_map) == 576:
        logger.info("🔄 Reshaping flat attention map to 24x24 spatial layout")
        attention_map = attention_map.reshape(24, 24)
    
    # Check for degenerate attention maps
    if attention_map.max() == attention_map.min():
        logger.warning(f"⚠️ Attention map is constant (value: {attention_map.max():.3f})")
    
    # V-JEPA uses 224x224 input images, but outputs 24x24 spatial features
    # We need to resize to match the display size while maintaining alignment
    zoom_factor = 16  # 24x24 -> 384x384 to match image display size
    logger.info(f"🔍 Upsampling attention map by factor {zoom_factor}")
    
    # Convert to float32 if needed (scipy doesn't support float16)
    if attention_map.dtype == np.float16:
        attention_map = attention_map.astype(np.float32)
    
    # Upsample with cubic interpolation for smooth visualization
    attention_hr = zoom(attention_map, zoom_factor, order=3)
    logger.info(f"✅ Upsampled to: {attention_hr.shape}, range: {attention_hr.min():.3f} to {attention_hr.max():.3f}")
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    attention_colored = cmap(attention_hr)
    
    # Set alpha channel
    attention_colored[:, :, 3] = attention_hr * alpha
    
    # Convert to image
    img = Image.fromarray((attention_colored * 255).astype(np.uint8))
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"


@app.route('/api/features/<image_id>/attention')
def get_attention_map(image_id):
    """Generate attention map for an image (compatibility endpoint)"""
    try:
        start_time = datetime.now()
        logger.info(f"🔄 Loading attention map for image: {image_id}")
        
        # Extract GBIF ID from image_id (format: gbif_XXXXXXX_taxon_XXXXXXX_img_N)
        import re
        match = re.match(r'gbif_(\d+)_taxon_\d+_img_(\d+)', image_id)
        
        if not match:
            logger.error(f"❌ Invalid image_id format: {image_id}")
            return jsonify({'error': 'Invalid image_id format'}), 400
            
        gbif_id = int(match.group(1))
        logger.info(f"📍 Extracted GBIF ID: {gbif_id}")
        
        # Load vision embedding with verification
        logger.info(f"🧠 Loading vision embedding for GBIF {gbif_id}")
        
        # Verify the observation exists and get its details
        obs_data = cache.loader.get_observation(gbif_id)
        if obs_data is None:
            logger.error(f"❌ Observation {gbif_id} not found in dataset")
            return jsonify({'error': 'Observation not found'}), 404
        
        logger.info(f"📋 Observation details: species='{obs_data['taxon_name']}', taxon_id={obs_data['taxon_id']}, has_vision={obs_data.get('has_vision', False)}")
        
        features = cache.get_vision_embedding(gbif_id, obs_data['taxon_id'], 1)
        if features is None:
            logger.warning(f"❌ No vision features found for GBIF {gbif_id} (species: {obs_data['taxon_name']})")
            return jsonify({'error': 'Vision features not found'}), 404
        
        logger.info(f"✅ Loaded vision features for {obs_data['taxon_name']}, shape: {features.shape}, dtype: {features.dtype}")
        
        # Verify the feature values are reasonable
        feature_stats = {
            'min': float(features.min().item()),
            'max': float(features.max().item()), 
            'mean': float(features.mean().item()),
            'std': float(features.std().item())
        }
        logger.info(f"📊 Feature stats: {feature_stats}")
        
        # Check for potential issues
        if feature_stats['max'] == feature_stats['min']:
            logger.warning(f"⚠️ Features appear to be constant (all same value: {feature_stats['max']})")
        if abs(feature_stats['mean']) > 100:
            logger.warning(f"⚠️ Features have unusually large magnitude (mean: {feature_stats['mean']})")
        
        # Get parameters
        temporal_mode = request.args.get('temporal', 'mean')
        visualization = request.args.get('visualization', 'l2norm')
        colormap = request.args.get('colormap', 'plasma')
        alpha = float(request.args.get('alpha', 0.7))
        
        logger.info(f"🎛️ Parameters: temporal={temporal_mode}, viz={visualization}, colormap={colormap}, alpha={alpha}")
        
        # Compute attention using the cache method
        logger.info("🔥 Computing spatial attention...")
        attention = cache.compute_spatial_attention(features, temporal_mode, 'mean', visualization)
        logger.info(f"✅ Attention computed, shape: {attention.shape if hasattr(attention, 'shape') else type(attention)}")
        
        if temporal_mode == 'mean':
            # Generate visualization
            logger.info(f"🎨 Generating attention overlay with {colormap} colormap")
            attention_img = generate_attention_overlay(attention, colormap, alpha)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"⚡ Attention map completed in {processing_time:.2f}s")
            
            return jsonify({
                'mode': 'spatial',
                'attention_map': attention_img,
                'stats': {
                    'max': float(attention.max().item() if isinstance(attention, torch.Tensor) else attention.max()),
                    'mean': float(attention.mean().item() if isinstance(attention, torch.Tensor) else attention.mean()),
                    'std': float(attention.std().item() if isinstance(attention, torch.Tensor) else attention.std())
                }
            })
        else:
            # Return temporal sequence
            attention_frames = []
            for t in range(8):
                attention_img = generate_attention_overlay(attention[t], colormap, alpha)
                attention_frames.append(attention_img)
            
            return jsonify({
                'mode': 'temporal',
                'attention_frames': attention_frames,
                'num_frames': 8
            })
            
    except Exception as e:
        logger.error(f"Error in get_attention_map: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/features/<image_id>/umap-rgb')
def get_umap_rgb(image_id):
    """
    Compute UMAP RGB visualization for spatial features.
    
    This endpoint creates a false-color image where each spatial patch's
    color represents its position in UMAP space, revealing semantic structure
    in the vision features.
    """
    try:
        start_time = datetime.now()
        logger.info(f"🌈 Computing UMAP RGB for image: {image_id}")
        
        # Extract GBIF ID from image_id
        import re
        match = re.match(r'gbif_(\d+)_taxon_\d+_img_(\d+)', image_id)
        
        if not match:
            logger.error(f"❌ Invalid image_id format for UMAP: {image_id}")
            return jsonify({'error': 'Invalid image_id format'}), 400
            
        gbif_id = int(match.group(1))
        logger.info(f"📍 Computing UMAP for GBIF ID: {gbif_id}")
        
        # Get observation data first
        obs_data = cache.loader.get_observation(gbif_id)
        if obs_data is None:
            logger.error(f"❌ Observation {gbif_id} not found")
            return jsonify({'error': 'Observation not found'}), 404
        
        # Load vision embedding
        features = cache.get_vision_embedding(gbif_id, obs_data['taxon_id'], 1)
        if features is None:
            logger.warning(f"❌ No vision features found for GBIF {gbif_id}")
            return jsonify({'error': 'Vision features not found'}), 404
        
        logger.info(f"✅ Loaded vision features, shape: {features.shape}")
        
        # Use fast PyTorch view operations
        features = features.view(8, 576, 1408).mean(dim=0)  # [576, 1408]
        logger.info(f"🔄 Reshaped to spatial features: {features.shape}")
        
        # Convert to numpy for sklearn operations
        features_flat = features.detach().cpu().numpy()  # [576, 1408]
        logger.info(f"📊 Feature range: min={features_flat.min():.3f}, max={features_flat.max():.3f}, mean={features_flat.mean():.3f}")
        
        # Apply UMAP to reduce to 3D
        from sklearn.decomposition import PCA
        
        logger.info("📉 Applying PCA for dimensionality reduction...")
        # First reduce dimensionality with PCA for speed
        pca = PCA(n_components=50)
        features_pca = pca.fit_transform(features_flat)
        logger.info(f"🔄 PCA reduced to: {features_pca.shape}")
        
        logger.info("🗺️ Applying UMAP...")
        # Then apply UMAP
        reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
        coords_3d = reducer.fit_transform(features_pca)
        logger.info(f"✅ UMAP coords shape: {coords_3d.shape}")
        logger.info(f"📊 UMAP range: min={coords_3d.min(axis=0)}, max={coords_3d.max(axis=0)}")
        
        # Normalize to [0,1] for RGB with better handling
        coords_min = coords_3d.min(axis=0)
        coords_max = coords_3d.max(axis=0)
        coords_range = coords_max - coords_min
        
        # Check for degenerate cases
        for i, (min_val, max_val, range_val) in enumerate(zip(coords_min, coords_max, coords_range)):
            logger.info(f"🎨 Dimension {i}: min={min_val:.3f}, max={max_val:.3f}, range={range_val:.3f}")
            if range_val < 1e-6:
                logger.warning(f"⚠️ Dimension {i} has very small range, may cause normalization issues")
        
        # Avoid division by zero
        coords_range = np.maximum(coords_range, 1e-6)
        coords_normalized = (coords_3d - coords_min) / coords_range
        
        # Ensure values are in [0,1]
        coords_normalized = np.clip(coords_normalized, 0, 1)
        logger.info(f"🎨 Normalized RGB range: min={coords_normalized.min(axis=0)}, max={coords_normalized.max(axis=0)}")
        
        # Check for all-black or all-white issues
        if coords_normalized.max() < 0.1:
            logger.warning("⚠️ UMAP RGB values are very dark, may appear black")
        elif coords_normalized.min() > 0.9:
            logger.warning("⚠️ UMAP RGB values are very bright, may appear white")
        
        # Reshape back to spatial layout
        rgb_spatial = coords_normalized.reshape(24, 24, 3)
        logger.info(f"🖼️ RGB spatial shape: {rgb_spatial.shape}")
        
        # Sample some RGB values for verification
        sample_pixels = rgb_spatial[:3, :3, :].reshape(-1, 3)
        logger.info(f"🔍 Sample RGB pixels: {sample_pixels[:3].tolist()}")
        
        # Upsample for better visualization
        from PIL import Image
        rgb_uint8 = (rgb_spatial * 255).astype(np.uint8)
        logger.info(f"🎨 RGB uint8 range: min={rgb_uint8.min()}, max={rgb_uint8.max()}")
        
        img = Image.fromarray(rgb_uint8)
        img = img.resize((384, 384), Image.NEAREST)  # Match overlay size
        logger.info(f"🖼️ Resized image to: {img.size}")
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"🌈 UMAP RGB completed in {processing_time:.2f}s")
        
        # Return both the image and raw RGB values for client-side processing
        rgb_values_list = coords_normalized.flatten().tolist()  # Flatten to 1D array for JS
        logger.info(f"📊 Returning {len(rgb_values_list)} RGB values (flattened {rgb_spatial.shape} -> 1D)")
        
        return jsonify({
            'umap_rgb': f"data:image/png;base64,{img_str}",
            'rgb_values': rgb_values_list,  # Raw RGB values for client-side alpha blending
            'coords_3d': coords_3d.tolist(),
            'shape': [24, 24, 3]
        })
        
    except Exception as e:
        logger.error(f"Error in get_umap_rgb: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/features/<image_id>/statistics')
def get_feature_statistics(image_id):
    """Get detailed statistics for image features"""
    try:
        # Extract GBIF ID from image_id
        import re
        match = re.match(r'gbif_(\d+)_taxon_\d+_img_(\d+)', image_id)
        
        if not match:
            return jsonify({'error': 'Invalid image_id format'}), 400
            
        gbif_id = int(match.group(1))
        
        # Get observation data first
        obs_data = cache.loader.get_observation(gbif_id)
        if obs_data is None:
            return jsonify({'error': 'Observation not found'}), 404
        
        # Load vision embedding
        features = cache.get_vision_embedding(gbif_id, obs_data['taxon_id'], 1)
        if features is None:
            return jsonify({'error': 'Vision features not found'}), 404
        
        # Reshape to temporal and spatial using PyTorch operations
        features = features.view(8, 576, 1408)
        
        # Compute spatial diversity (variance across spatial locations)
        spatial_diversity = float(features.var(dim=(0, 2)).mean().item())
        
        # Compute temporal stability (1 - variance across time)
        temporal_variance = float(features.var(dim=0).mean().item())
        temporal_stability = max(0, 1 - temporal_variance)
        
        # Overall feature magnitude
        feature_magnitude = float(features.norm(dim=-1).mean().item())
        
        # Information density (entropy approximation)
        features_abs = torch.abs(features) + 1e-10
        features_normalized = features_abs / features_abs.sum(dim=-1, keepdim=True)
        entropy = -(features_normalized * torch.log(features_normalized)).sum(dim=-1).mean()
        information_density = float(entropy.item())
        
        return jsonify({
            'spatial_diversity': spatial_diversity,
            'temporal_stability': temporal_stability,
            'feature_magnitude': feature_magnitude,
            'information_density': information_density,
            'total_features': int(features.numel()),
            'shape': list(features.shape)
        })
        
    except Exception as e:
        logger.error(f"Error in get_feature_statistics: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health')
def health_check():
    """Health check endpoint for monitoring"""
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'dataset': {
                'name': CONFIG['dataset_name'],
                'version': CONFIG['dataset_version']
            },
            'data_loaded': {
                'observations': len(cache.loader.observations) if cache.loader.observations is not None else 0,
                'vision_metadata': len(cache.loader.vision_index) if cache.loader.vision_index is not None else 0,
                'species': len(cache.loader.taxon_to_gbifs) if hasattr(cache.loader, 'taxon_to_gbifs') else 0
            },
            'cache_stats': cache.loader.get_cache_stats(),
            'mmap_loader': {
                'enabled': cache.mmap_loader is not None,
                'cache_stats': cache.mmap_loader.get_cache_stats() if cache.mmap_loader else None
            },
            'precomputed_data': {
                'language_umap': cache.precomputed_language_umap is not None,
                'language_clusters': cache.language_clusters is not None,
                'vision_umap': cache.precomputed_vision_umap is not None
            }
        }
        return jsonify(health_status)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/deepearth-static/<path:path>')
def serve_static(path):
    """Serve static files from a unique path to avoid conflicts with main site"""
    return send_from_directory('static', path)


if __name__ == '__main__':
    print("\n" + "="*80)
    print(f"🌍 DeepEarth Multimodal Geospatial Dashboard")
    print("="*80)
    print(f"Dataset: {CONFIG['dataset_name']}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Starting server on http://localhost:5000")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)