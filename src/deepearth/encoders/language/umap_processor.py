#!/usr/bin/env python3
"""
Compute 3D UMAP projection and HDBSCAN clustering for embeddings
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan
import json
from typing import Dict, List, Optional, Union, Tuple
import warnings
warnings.filterwarnings('ignore')

def generate_color_palette(n_colors: int) -> Dict[int, str]:
    """Generate perceptually uniform colors for clusters"""
    import colorsys
    
    colors = {}
    for i in range(-1, n_colors):
        if i == -1:
            # Noise points in gray
            colors[i] = "#808080"
        else:
            # Use HSL color space for perceptually uniform colors
            hue = (i * 360 / n_colors) % 360
            saturation = 0.7
            lightness = 0.5
            
            # Convert HSL to RGB
            r, g, b = colorsys.hls_to_rgb(hue/360, lightness, saturation)
            colors[i] = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    
    return colors

def load_embeddings_from_directory(embeddings_dir: Path, 
                                 file_pattern: str = "*.pt",
                                 embedding_field: str = 'mean_embedding',
                                 label_field: str = 'species_name') -> Tuple[np.ndarray, List[str], List[Dict]]:
    """
    Load embeddings from a directory of .pt files
    
    Returns:
        embeddings: numpy array of shape (n_samples, embedding_dim)
        labels: list of labels for each embedding
        metadata: list of metadata dicts for each embedding
    """
    embeddings = []
    labels = []
    metadata = []
    
    for pt_file in sorted(embeddings_dir.glob(file_pattern)):
        try:
            data = torch.load(pt_file, map_location='cpu', weights_only=False)
            
            # Get embedding
            if embedding_field in data:
                embedding = data[embedding_field]
                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.numpy()
                embeddings.append(embedding)
                
                # Get label
                label = data.get(label_field, pt_file.stem)
                labels.append(label)
                
                # Collect metadata
                meta = {
                    'filename': pt_file.name,
                    'label': label
                }
                for key in ['taxon_id', 'num_tokens', 'prompt']:
                    if key in data:
                        meta[key] = data[key]
                metadata.append(meta)
        except Exception as e:
            print(f"Error loading {pt_file}: {e}")
            continue
    
    return np.array(embeddings), labels, metadata

def compute_3d_umap_and_clusters(embeddings: Union[np.ndarray, str, Path],
                               labels: Optional[List[str]] = None,
                               output_path: Optional[Union[str, Path]] = None,
                               n_neighbors: int = 15,
                               min_dist: float = 0.1,
                               metric: str = 'cosine',
                               min_cluster_size: int = 5,
                               normalize: bool = True) -> Dict:
    """
    Compute 3D UMAP projection and HDBSCAN clustering
    
    Args:
        embeddings: Either numpy array of embeddings or path to directory containing .pt files
        labels: Optional list of labels for each embedding
        output_path: Optional path to save results
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        metric: Distance metric for UMAP
        min_cluster_size: HDBSCAN min_cluster_size
        normalize: Whether to normalize embeddings before UMAP
        
    Returns:
        Dictionary containing:
            - points: List of point data with 3D coordinates and metadata
            - stats: Statistics about the projection
            - colors: Color mapping for clusters
    """
    
    # Load embeddings if path provided
    if isinstance(embeddings, (str, Path)):
        embeddings_dir = Path(embeddings)
        print(f"Loading embeddings from {embeddings_dir}...")
        X, labels, metadata = load_embeddings_from_directory(embeddings_dir)
    else:
        X = embeddings
        metadata = [{'label': label} for label in (labels or [])]
    
    if labels is None:
        labels = [f"Point_{i}" for i in range(len(X))]
    
    print(f"Processing {len(X)} embeddings, shape: {X.shape}")
    
    # Normalize embeddings
    if normalize:
        print("Normalizing embeddings...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    # 3D UMAP
    print("Computing 3D UMAP projection...")
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=1.0,
        metric=metric,
        random_state=42,
        n_epochs=1000,
        init='spectral',
        verbose=True
    )
    
    X_3d = reducer.fit_transform(X_scaled)
    
    # HDBSCAN clustering
    print("Performing HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=3,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    
    cluster_labels = clusterer.fit_predict(X_3d)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"Found {n_clusters} clusters")
    
    # Generate colors
    colors = generate_color_palette(max(cluster_labels) + 1)
    
    # Prepare data for export
    data_points = []
    for i, (label, coords, cluster) in enumerate(zip(labels, X_3d, cluster_labels)):
        point_data = {
            'id': i,
            'name': label,
            'x': float(coords[0]),
            'y': float(coords[1]),
            'z': float(coords[2]),
            'cluster': int(cluster),
            'color': colors[cluster]
        }
        
        # Add metadata if available
        if i < len(metadata):
            for key, value in metadata[i].items():
                if key not in point_data:
                    point_data[key] = value
        
        data_points.append(point_data)
    
    # Calculate statistics
    cluster_sizes = {}
    for cluster_id in range(n_clusters):
        cluster_sizes[cluster_id] = int(sum(cluster_labels == cluster_id))
    
    stats = {
        'total_points': len(labels),
        'n_clusters': n_clusters,
        'noise_points': int(sum(cluster_labels == -1)),
        'dimensions': int(X.shape[1]),
        'cluster_sizes': cluster_sizes,
        'x_range': [float(X_3d[:, 0].min()), float(X_3d[:, 0].max())],
        'y_range': [float(X_3d[:, 1].min()), float(X_3d[:, 1].max())],
        'z_range': [float(X_3d[:, 2].min()), float(X_3d[:, 2].max())]
    }
    
    # Prepare output
    output_data = {
        'points': data_points,
        'stats': stats,
        'colors': colors
    }
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved 3D UMAP data to {output_path}")
        
        # Also save as CSV for verification
        df = pd.DataFrame(data_points)
        csv_path = output_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV to {csv_path}")
    
    # Print statistics
    print("\nStatistics:")
    print(f"  Total points: {stats['total_points']}")
    print(f"  Embedding dimensions: {stats['dimensions']}")
    print(f"  Clusters: {stats['n_clusters']}")
    print(f"  Noise points: {stats['noise_points']}")
    print(f"  Coordinate ranges:")
    print(f"    X: {stats['x_range']}")
    print(f"    Y: {stats['y_range']}")
    print(f"    Z: {stats['z_range']}")
    
    return output_data

def create_visualization_config(umap_data: Dict,
                              title: str = "3D Embeddings",
                              subtitle: str = "",
                              images_enabled: bool = False,
                              output_path: Optional[Union[str, Path]] = None) -> Dict:
    """
    Create a configuration file for the 3D visualization tool
    """
    config = {
        "visualization": {
            "title": title,
            "subtitle": subtitle,
            "defaultSphereSize": 0.05,
            "defaultTextSize": 0.3,
            "scaleFactor": 3.0
        },
        "data": {
            "sourceFile": "umap_3d_data.json",
            "fields": {
                "id": "id",
                "label": "name",
                "cluster": "cluster",
                "metadata": []
            }
        },
        "images": {
            "enabled": images_enabled
        },
        "server": {
            "host": "localhost",
            "port": 8080,
            "openBrowser": True
        }
    }
    
    # Add metadata fields if present
    if umap_data['points']:
        sample_point = umap_data['points'][0]
        metadata_fields = [k for k in sample_point.keys() 
                          if k not in ['id', 'name', 'x', 'y', 'z', 'cluster', 'color']]
        config['data']['fields']['metadata'] = metadata_fields
    
    if output_path:
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved visualization config to {output_path}")
    
    return config

def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute 3D UMAP for embeddings')
    parser.add_argument('embeddings_dir', help='Directory containing .pt embedding files')
    parser.add_argument('--output', '-o', default='umap_3d_data.json', help='Output file path')
    parser.add_argument('--n-neighbors', type=int, default=15, help='UMAP n_neighbors parameter')
    parser.add_argument('--min-dist', type=float, default=0.1, help='UMAP min_dist parameter')
    parser.add_argument('--min-cluster-size', type=int, default=5, help='HDBSCAN min_cluster_size')
    parser.add_argument('--no-normalize', action='store_true', help='Skip normalization')
    
    args = parser.parse_args()
    
    # Compute UMAP
    umap_data = compute_3d_umap_and_clusters(
        args.embeddings_dir,
        output_path=args.output,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        min_cluster_size=args.min_cluster_size,
        normalize=not args.no_normalize
    )
    
    # Create visualization config
    config_path = Path(args.output).with_name('visualization_config.json')
    create_visualization_config(
        umap_data,
        title="Language Embeddings Visualization",
        output_path=config_path
    )

if __name__ == "__main__":
    main()