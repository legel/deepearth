# deepearth/core/sampling.py
"""
DeepEarth Data Sampling Engine
══════════════════════════════

Intelligent context window construction using learned similarity indices.
The sampling engine builds efficient data structures for finding related
observations across space, time, and semantic dimensions.

Sampling Philosophy:
    Earth observations are not independent - they exhibit strong correlations
    across multiple dimensions. By sampling related observations together in
    context windows, we help the model learn these natural patterns.

Index Types:
    ┌─────────────┬────────────────────────────────────────┐
    │   Temporal  │ Time of day, season, historical era   │
    ├─────────────┼────────────────────────────────────────┤
    │   Spatial   │ Geographic proximity (3D distance)     │
    ├─────────────┼────────────────────────────────────────┤
    │   Semantic  │ Within-modality similarity (UMAP)      │
    ├─────────────┼────────────────────────────────────────┤
    │  Universal  │ Cross-modal similarity (joint UMAP)    │
    └─────────────┴────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pickle
from tqdm import tqdm


class DeepEarthDataSamplingEngine:
    """
    Efficient sampling engine for context window construction.
    
    The engine maintains multiple sorted indices for different similarity
    dimensions, enabling fast nearest-neighbor queries during training.
    All operations are performed on GPU for maximum efficiency.
    """
    
    def __init__(self, config, data_dict: Dict, device: Optional[torch.device] = None):
        """
        Initialize sampling engine with data indices.
        
        Args:
            config: DeepEarthConfig instance
            data_dict: Preprocessed data dictionary
            device: Target device for GPU acceleration
        """
        self.config = config
        self.data = data_dict
        self.device = device or torch.device(config.device)
        self.n_samples = data_dict['n_samples']
        
        print(f"\n{'='*70}")
        print(f"Initializing DeepEarth Data Sampling Engine")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Total samples: {self.n_samples:,}")
        print(f"Sampling strategy: {config.sampling_strategy.sampling_type}")
        print(f"Clusters per context: {config.sampling_strategy.clusters_per_context}")
        print(f"Samples per cluster: {config.sampling_strategy.samples_per_cluster}")
        
        # Move data to GPU for fast access
        self._move_data_to_device()
        
        # Build sampling indices
        self.sampling_indices = {}
        self.build_sampling_indices()
        
        print(f"\nSampling engine ready with {len(self.sampling_indices)} indices")
        print(f"{'='*70}\n")
    
    def _move_data_to_device(self):
        """Move frequently accessed data to GPU."""
        print(f"\n[SamplingEngine] Moving data to {self.device}...")
        
        # Move coordinate and time data
        self.xyzt_gpu = self.data['xyzt'].to(self.device)
        self.time_components_gpu = self.data['time_components'].to(self.device)
        
        # Move metadata
        self.dataset_modality_encoder_gpu = self.data['dataset_modality_encoder'].to(self.device)
        
        print(f"[SamplingEngine] Data moved to GPU")
    
    def build_sampling_indices(self):
        """
        Build all sampling indices for efficient nearest neighbor queries.
        
        This method constructs sorted indices for each dimension of similarity,
        enabling O(log n) nearest neighbor searches during training.
        """
        print(f"\n[SamplingEngine] Building sampling indices...")
        
        # ═══════════════════════════════════════════════════════════
        # Temporal Indices
        # ═══════════════════════════════════════════════════════════
        
        print(f"\n  Building temporal indices...")
        
        # Time of day index (diurnal cycles)
        time_of_day = self.time_components_gpu[:, 0]
        self.sampling_indices['time_of_day'] = self._build_sorted_index(
            time_of_day, "Time of day"
        )
        
        # Time of year index (seasonal cycles)
        time_of_year = self.time_components_gpu[:, 1]
        self.sampling_indices['time_of_year'] = self._build_sorted_index(
            time_of_year, "Time of year"
        )
        
        # Time of history index (long-term trends)
        time_of_history = self.time_components_gpu[:, 2]
        self.sampling_indices['time_of_history'] = self._build_sorted_index(
            time_of_history, "Time of history"
        )
        
        # ═══════════════════════════════════════════════════════════
        # Spatial Index
        # ═══════════════════════════════════════════════════════════
        
        print(f"\n  Building spatial index...")
        
        # Compute 3D Euclidean distance from origin
        # For normalized coordinates, this provides a simple spatial ordering
        spatial_dist = torch.norm(self.xyzt_gpu[:, :3], dim=1)
        self.sampling_indices['spatial'] = self._build_sorted_index(
            spatial_dist, "Spatial distance"
        )
        
        # ═══════════════════════════════════════════════════════════
        # Modality-Specific Indices (UMAP)
        # ═══════════════════════════════════════════════════════════
        
        print(f"\n  Building modality indices...")
        self._build_modality_indices()
        
        # ═══════════════════════════════════════════════════════════
        # Universal Index (Joint UMAP)
        # ═══════════════════════════════════════════════════════════
        
        print(f"\n  Building universal index...")
        self._build_universal_index()
        
    def _build_sorted_index(self, values: torch.Tensor, name: str) -> Dict:
        """
        Build sorted index for efficient nearest neighbor queries.
        
        The sorted index enables binary search for finding similar samples
        in O(log n) time instead of O(n) linear search.
        
        Index structure:
            values: Sorted similarity values
            indices: Original sample indices in sorted order
            reverse: Mapping from original to sorted position
        
        Args:
            values: 1D tensor of values to sort
            name: Name for progress reporting
            
        Returns:
            Dictionary with sorted index components
        """
        print(f"    → Sorting {name} ({len(values):,} values)...", end='')
        
        # Sort values and get indices
        sorted_values, sort_indices = torch.sort(values)
        
        # Create reverse mapping (original index → sorted position)
        reverse_indices = torch.empty_like(sort_indices)
        reverse_indices[sort_indices] = torch.arange(len(sort_indices), device=self.device)
        
        print(f" Done! Range: [{sorted_values[0]:.3f}, {sorted_values[-1]:.3f}]")
        
        return {
            'values': sorted_values,      # Sorted values for binary search
            'indices': sort_indices,       # Original indices in sorted order
            'reverse': reverse_indices,    # Original → sorted position
            'name': name                  # For debugging
        }
    
    def _build_modality_indices(self):
        """
        Build UMAP-based similarity indices for each modality.
        
        UMAP (Uniform Manifold Approximation and Projection) learns a
        low-dimensional representation that preserves local structure
        in the high-dimensional modality embedding space.
        """
        cache_path = Path(self.config.cache_dir) / 'umap_indices'
        cache_path.mkdir(exist_ok=True)
        
        # Check for torchdr availability
        try:
            from torchdr import UMAP as torchdrUMAP
            print(f"    → Using TorchDR UMAP (GPU-accelerated)")
        except ImportError:
            print(f"    ✗ TorchDR not available! Install with: pip install torchdr")
            print(f"      Also recommended: conda install -c pytorch -c nvidia faiss-gpu")
            raise ImportError("TorchDR required for UMAP indices")
        
        # Process each encoder's data
        for encoder_idx, encoder_data in self.data['encoded_data'].items():
            encoder_name = self.data['encoder_map'].get(encoder_idx, f"encoder_{encoder_idx}")
            cache_file = cache_path / f'umap_encoder_{encoder_idx}.pkl'
            
            print(f"    → Processing {encoder_name} (encoder {encoder_idx})...")
            
            if cache_file.exists() and not self.config.regenerate_cache:
                # Load from cache
                print(f"      Loading cached UMAP...")
                with open(cache_file, 'rb') as f:
                    umap_data = pickle.load(f)
            else:
                # Train new UMAP
                print(f"      Training UMAP on {len(encoder_data):,} samples...")
                
                # Move data to GPU
                encoder_data_gpu = encoder_data.to(self.device)
                n_samples = len(encoder_data_gpu)
                
                # Sample subset for training if needed
                if n_samples > self.config.umap_max_samples:
                    print(f"      Subsampling {self.config.umap_max_samples:,} for training...")
                    sample_indices = torch.randperm(n_samples)[:self.config.umap_max_samples]
                    train_data = encoder_data_gpu[sample_indices]
                else:
                    train_data = encoder_data_gpu
                
                # Configure UMAP
                umap = torchdrUMAP(
                    n_neighbors=self.config.umap_n_neighbors,
                    min_dist=self.config.umap_min_dist,
                    n_components=self.config.umap_dim,
                    device=self.device,
                    verbose=False
                )
                
                # Fit and transform
                print(f"      Fitting UMAP model...")
                embeddings = umap.fit_transform(train_data)
                
                # Apply to all data if trained on subset
                if n_samples > self.config.umap_max_samples:
                    print(f"      Transforming all {n_samples:,} samples...")
                    embeddings = umap.transform(encoder_data_gpu)
                
                # Save to cache
                umap_data = {
                    'embeddings': embeddings.cpu(),  # Store on CPU to save GPU memory
                    'model': umap
                }
                
                with open(cache_file, 'wb') as f:
                    pickle.dump(umap_data, f)
                print(f"      Cached UMAP model")
            
            # Build sorted index from UMAP embeddings
            embeddings_1d = umap_data['embeddings'].squeeze().to(self.device)
            
            # Store both UMAP model and sorted index
            self.sampling_indices[f'modality_{encoder_idx}'] = self._build_sorted_index(
                embeddings_1d, f"Modality {encoder_name}"
            )
            self.sampling_indices[f'modality_{encoder_idx}']['umap_model'] = umap_data['model']
    
    def _build_universal_index(self):
        """
        Build universal index combining all dimensions.
        
        The universal index captures cross-modal, cross-temporal, and
        cross-spatial similarities in a single unified representation.
        This is achieved by concatenating all individual dimensions and
        applying UMAP to the combined feature space.
        """
        print(f"    → Building universal index...")
        
        cache_path = Path(self.config.cache_dir) / 'umap_indices'
        cache_file = cache_path / 'umap_universal.pkl'
        
        if cache_file.exists() and not self.config.regenerate_cache:
            print(f"      Loading cached universal UMAP...")
            with open(cache_file, 'rb') as f:
                umap_data = pickle.load(f)
            universal_embeddings = umap_data['embeddings'].to(self.device)
        else:
            # ───────────────────────────────────────────────────────
            # Concatenate all feature dimensions
            # ───────────────────────────────────────────────────────
            
            print(f"      Concatenating features...")
            
            features = []
            
            # Add temporal features
            features.append(self.time_components_gpu)  # [N, 3]
            
            # Add spatial features  
            features.append(self.xyzt_gpu[:, :3])  # [N, 3]
            
            # Add modality embeddings (already reduced to 1D)
            for encoder_idx in self.data['encoded_data'].keys():
                key = f'modality_{encoder_idx}'
                if key in self.sampling_indices:
                    # Get UMAP embeddings for this modality
                    modality_values = self.sampling_indices[key]['values']
                    # Normalize to [0, 1] for consistent scaling
                    modality_norm = (modality_values - modality_values.min()) / (
                        modality_values.max() - modality_values.min() + 1e-8
                    )
                    features.append(modality_norm.unsqueeze(1))  # [N, 1]
            
            # Concatenate all features
            combined_features = torch.cat(features, dim=1)  # [N, total_dims]
            
            print(f"      Combined feature dimensions: {combined_features.shape}")
            
            # ───────────────────────────────────────────────────────
            # Apply UMAP to combined features
            # ───────────────────────────────────────────────────────
            
            try:
                from torchdr import UMAP as torchdrUMAP
                
                # Configure universal UMAP
                print(f"      Training universal UMAP...")
                
                # Sample if needed
                n_samples = len(combined_features)
                if n_samples > self.config.umap_max_samples:
                    print(f"      Subsampling {self.config.umap_max_samples:,} for training...")
                    sample_indices = torch.randperm(n_samples)[:self.config.umap_max_samples]
                    train_features = combined_features[sample_indices]
                else:
                    train_features = combined_features
                
                # Train UMAP
                umap = torchdrUMAP(
                    n_neighbors=self.config.umap_n_neighbors * 2,  # Larger neighborhood for global structure
                    min_dist=self.config.umap_min_dist,
                    n_components=self.config.umap_dim,
                    device=self.device,
                    verbose=False
                )
                
                universal_embeddings = umap.fit_transform(train_features)
                
                # Apply to all data if trained on subset
                if n_samples > self.config.umap_max_samples:
                    print(f"      Transforming all samples...")
                    universal_embeddings = umap.transform(combined_features)
                
                # Cache results
                umap_data = {
                    'embeddings': universal_embeddings.cpu(),
                    'model': umap,
                    'feature_dims': combined_features.shape[1]
                }
                
                with open(cache_file, 'wb') as f:
                    pickle.dump(umap_data, f)
                print(f"      Cached universal UMAP")
                
                universal_embeddings = universal_embeddings.to(self.device)
                
            except ImportError:
                print(f"      ✗ UMAP unavailable, using PCA fallback")
                # Simple PCA as fallback
                centered = combined_features - combined_features.mean(0)
                U, S, V = torch.svd(centered)
                universal_embeddings = U[:, 0]  # First principal component
        
        # Build sorted index
        self.sampling_indices['universal'] = self._build_sorted_index(
            universal_embeddings.squeeze(), "Universal"
        )
        
    def sample_context(self, strategy: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """
        Sample a context window using the configured strategy.
        
        Context sampling follows a hierarchical approach:
        1. Select which similarity dimension to use (based on weights)
        2. Sample cluster centers in that dimension
        3. Gather neighboring samples around each cluster
        
        Args:
            strategy: Optional override for sampling weights
            
        Returns:
            Tensor of sample indices for the context window
        """
        if strategy is None:
            strategy = {
                'time_of_day': self.config.sampling_strategy.time_of_day_weight,
                'time_of_year': self.config.sampling_strategy.time_of_year_weight,
                'time_of_history': self.config.sampling_strategy.time_of_history_weight,
                'spatial': self.config.sampling_strategy.spatial_weight,
                'modality': self.config.sampling_strategy.modality_weight,
                'universal': self.config.sampling_strategy.universal_weight
            }
        
        context_samples = []
        
        # Sample clusters
        for cluster_idx in range(self.config.sampling_strategy.clusters_per_context):
            
            # ───────────────────────────────────────────────────────
            # Choose sampling dimension based on strategy weights
            # ───────────────────────────────────────────────────────
            
            # Filter to available indices
            available_dims = [dim for dim in strategy.keys() if dim in self.sampling_indices]
            
            # Handle modality dimension (may have multiple indices)
            if 'modality' in strategy and 'modality' not in self.sampling_indices:
                # Add all modality indices
                modality_indices = [k for k in self.sampling_indices.keys() 
                                  if k.startswith('modality_')]
                if modality_indices:
                    # Split modality weight equally
                    modality_weight = strategy['modality'] / len(modality_indices)
                    for mod_idx in modality_indices:
                        available_dims.append(mod_idx)
                        strategy[mod_idx] = modality_weight
                    del strategy['modality']
            
            if not available_dims:
                continue
            
            # Normalize weights
            weights = torch.tensor([strategy.get(dim, 0.0) for dim in available_dims])
            weights = weights / weights.sum()
            
            # Sample dimension
            dim_idx = torch.multinomial(weights, 1).item()
            selected_dim = available_dims[dim_idx]
            
            # ───────────────────────────────────────────────────────
            # Sample cluster center
            # ───────────────────────────────────────────────────────
            
            center_idx = torch.randint(0, self.n_samples, (1,), device=self.device).item()
            
            # ───────────────────────────────────────────────────────
            # Gather neighbors
            # ───────────────────────────────────────────────────────
            
            if self.config.sampling_strategy.sampling_type == 'contiguous':
                neighbors = self._get_contiguous_neighbors(selected_dim, center_idx)
            else:  # probabilistic
                neighbors = self._get_probabilistic_neighbors(selected_dim, center_idx)
            
            context_samples.extend(neighbors)
        
        # ═══════════════════════════════════════════════════════════
        # Limit to context window size
        # ═══════════════════════════════════════════════════════════
        
        if len(context_samples) > self.config.context_window:
            # Randomly subsample to fit context window
            context_samples = torch.tensor(context_samples, device=self.device)
            perm = torch.randperm(len(context_samples))[:self.config.context_window]
            context_samples = context_samples[perm]
        else:
            context_samples = torch.tensor(context_samples, device=self.device)
        
        return context_samples
    
    def _get_contiguous_neighbors(self, index_name: str, center_idx: int) -> List[int]:
        """
        Get contiguous neighbors in sorted similarity space.
        
        Retrieves samples that are adjacent in the sorted index,
        representing the most similar samples to the center.
        
        Args:
            index_name: Name of similarity index to use
            center_idx: Original sample index of cluster center
            
        Returns:
            List of neighboring sample indices
        """
        index_data = self.sampling_indices[index_name]
        
        # Find position of center in sorted array
        sorted_pos = index_data['reverse'][center_idx].item()
        
        # Get neighbors within radius
        radius = self.config.sampling_strategy.samples_per_cluster
        start = max(0, sorted_pos - radius)
        end = min(len(index_data['indices']), sorted_pos + radius + 1)
        
        # Return original indices of neighbors
        neighbors = index_data['indices'][start:end].cpu().tolist()
        
        return neighbors
    
    def _get_probabilistic_neighbors(self, index_name: str, center_idx: int) -> List[int]:
        """
        Sample neighbors probabilistically within similarity bin.
        
        Instead of taking all contiguous neighbors, this method defines
        a similarity range and randomly samples within it, providing
        more diversity in the context window.
        
        Args:
            index_name: Name of similarity index to use
            center_idx: Original sample index of cluster center
            
        Returns:
            List of sampled neighbor indices
        """
        index_data = self.sampling_indices[index_name]
        
        # Find center value
        sorted_pos = index_data['reverse'][center_idx]
        center_value = index_data['values'][sorted_pos]
        
        # Define similarity bin (1% of range)
        value_range = index_data['values'][-1] - index_data['values'][0]
        bin_width = value_range * 0.01
        
        # Find bin boundaries using binary search
        min_val = center_value - bin_width
        max_val = center_value + bin_width
        
        start = torch.searchsorted(index_data['values'], min_val).item()
        end = torch.searchsorted(index_data['values'], max_val).item()
        
        # Sample from bin
        n_samples = min(
            self.config.sampling_strategy.samples_per_cluster * 2 + 1,
            end - start
        )
        
        if n_samples > 0:
            # Sample positions within bin
            bin_positions = torch.randint(start, end, (n_samples,), device=self.device)
            neighbors = index_data['indices'][bin_positions].cpu().tolist()
        else:
            neighbors = [center_idx]
        
        return neighbors
    
    def get_sample_similarities(self, idx: int, index_name: str, k: int = 10) -> Dict:
        """
        Get k most similar samples to a given sample.
        
        Useful for debugging and understanding what the model considers
        similar in different dimensions.
        
        Args:
            idx: Sample index
            index_name: Which similarity index to use
            k: Number of neighbors to return
            
        Returns:
            Dictionary with neighbor indices and distances
        """
        if index_name not in self.sampling_indices:
            raise ValueError(f"Unknown index: {index_name}")
        
        index_data = self.sampling_indices[index_name]
        
        # Find position in sorted array
        sorted_pos = index_data['reverse'][idx].item()
        center_value = index_data['values'][sorted_pos]
        
        # Get k neighbors on each side
        start = max(0, sorted_pos - k)
        end = min(len(index_data['indices']), sorted_pos + k + 1)
        
        neighbor_indices = index_data['indices'][start:end]
        neighbor_values = index_data['values'][start:end]
        neighbor_distances = torch.abs(neighbor_values - center_value)
        
        return {
            'indices': neighbor_indices.cpu().numpy(),
            'distances': neighbor_distances.cpu().numpy(),
            'center_value': center_value.cpu().item()
        }
