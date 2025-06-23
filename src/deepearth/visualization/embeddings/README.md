# 3D Embedding Visualization

A web application for interactive visualization of high-dimensional embeddings projected into 3D space using dimensionality reduction techniques like UMAP.

## Features

- **Interactive 3D Visualization**: Explore embedding spaces with smooth camera controls and animations
- **Clustering Support**: Visualize and navigate between different clusters in your data
- **Search & Filter**: Quickly find specific points in the embedding space
- **Configurable**: Adapt to any embedding dataset through configuration files
- **Responsive Design**: Works on desktop and mobile devices

## Quick Start

### Basic Usage

```bash
# Visualize embeddings with default settings
python server.py --data my_embeddings.json --title "My Embeddings"

# Use a custom configuration file
python server.py --config my_config.json

# Run without opening browser
python server.py --data embeddings.json --no-browser --port 8090
```

### Data Format

Your embedding data should be in JSON format with the following structure:

```json
{
  "points": [
    {
      "id": 1,
      "name": "Point 1",
      "x": 1.234,
      "y": 2.345,
      "z": 3.456,
      "cluster": 0
    },
    ...
  ],
  "colors": {
    "0": "#ff6b6b",
    "1": "#4ecdc4",
    "-1": "#888888"
  },
  "stats": {
    "total_points": 100,
    "n_clusters": 5,
    "cluster_sizes": {
      "0": 20,
      "1": 30
    }
  }
}
```

## Configuration

Create a configuration file to customize the visualization:

```json
{
  "visualization": {
    "title": "My Embedding Space",
    "defaultSphereSize": 0.05,
    "defaultTextSize": 0.3,
    "scaleFactor": 3.0
  },
  "data": {
    "sourceFile": "embeddings.json",
    "fields": {
      "id": "id",
      "label": "name",
      "cluster": "cluster",
      "metadata": ["category", "score"]
    }
  },
  "images": {
    "enabled": true,
    "sourcePath": "static/images",
    "urlPattern": "/static/images/{directory}/",
    "directoryPattern": "item_{id}",
    "filePattern": "img_{index}.jpg",
    "maxImages": 5
  },
  "server": {
    "host": "localhost",
    "port": 8080,
    "openBrowser": true
  }
}
```

### Image Configuration

The image system is flexible and can adapt to different data structures:

- **Simple ID-based**: Images in folders named by item ID
  ```json
  "images": {
    "enabled": true,
    "urlPattern": "/static/images/{id}/",
    "filePattern": "image_{index}.jpg"
  }
  ```

- **Custom directory patterns**: Use any field from your data
  ```json
  "images": {
    "enabled": true,
    "urlPattern": "/static/products/{category}/{sku}/",
    "directoryPattern": "{category}/{sku}",
    "filePattern": "photo_{index}.png"
  }
  ```

- **With summary file**: Map complex relationships
  ```json
  "images": {
    "enabled": true,
    "summaryFile": "image_mapping.json",
    "summaryKeyField": "name",
    "summaryImageField": "gallery_id",
    "urlPattern": "/static/galleries/{directory}/",
    "directoryPattern": "gallery_{gallery_id}"
  }
  ```

See example configurations in the `configs/` directory for different use cases.

## API Endpoints

- `GET /` - Main visualization interface
- `GET /config` - Current configuration
- `GET /data` - Embedding data
- `GET /health` - Health check endpoint

## Requirements

- Python 3.7+
- Modern web browser with WebGL support

## Creating Embeddings

This tool visualizes pre-computed 3D embeddings. To create embeddings from high-dimensional data:

1. **Extract embeddings** from your model (e.g., neural network activations, word embeddings)
2. **Reduce dimensions** using UMAP, t-SNE, or PCA to get 3D coordinates
3. **Cluster** your data (optional) using algorithms like HDBSCAN or K-means
4. **Format** as JSON following the schema above

Example using Python:

```python
import umap
import json
from sklearn.cluster import HDBSCAN

# Your high-dimensional embeddings
embeddings = ...  # shape: (n_samples, n_features)
labels = ...      # shape: (n_samples,)

# Reduce to 3D
reducer = umap.UMAP(n_components=3)
coords_3d = reducer.fit_transform(embeddings)

# Cluster
clusterer = HDBSCAN(min_cluster_size=5)
clusters = clusterer.fit_predict(coords_3d)

# Format for visualization
data = {
    "points": [
        {
            "id": i,
            "name": labels[i],
            "x": float(coords_3d[i, 0]),
            "y": float(coords_3d[i, 1]),
            "z": float(coords_3d[i, 2]),
            "cluster": int(clusters[i])
        }
        for i in range(len(embeddings))
    ],
    "colors": {
        str(i): f"#{hash(i) % 0xFFFFFF:06x}"
        for i in set(clusters)
    }
}

with open('embeddings.json', 'w') as f:
    json.dump(data, f)
```

## Development

```bash
# Run in debug mode
python server.py --debug --data test_data.json

# Custom host for network access
python server.py --host 0.0.0.0 --port 8080
```

## License

MIT License - See LICENSE file for details