---
license: mit
task_categories:
- image-classification
- feature-extraction
- zero-shot-classification
language:
- en
tags:
- biodiversity
- biology
- computer-vision
- multimodal
- self-supervised-learning
- florida
- plants
pretty_name: DeepEarth Central Florida Native Plants
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
    - split: train
      path: observations.parquet
    - split: test
      path: observations.parquet
---

# DeepEarth Central Florida Native Plants Dataset v0.2.0

## 🌿 Dataset Summary

A comprehensive multimodal dataset featuring **33,665 observations** of **232 native plant species** from Central Florida. This dataset combines citizen science observations with state-of-the-art vision and language embeddings for advancing multimodal self-supervised ecological intelligence research.

### Key Features
- 🌍 **Spatiotemporal Coverage**: Complete GPS coordinates and timestamps for all observations
- 🖼️ **Multimodal**: 31,136 observations with images, 7,113 with vision embeddings
- 🧬 **Language Embeddings**: DeepSeek-V3 embeddings for all 232 species
- 👁️ **Vision Embeddings**: V-JEPA-2 self-supervised features (6.5M dimensions)
- 📊 **Rigorous Splits**: Spatiotemporal train/test splits for robust evaluation

## 📦 Dataset Structure

```
observations.parquet         # Main dataset (500MB)
vision_index.parquet        # Vision embeddings index
vision_embeddings/          # Vision features (50GB total)
├── embeddings_000000.parquet
├── embeddings_000001.parquet
└── ... (159 files)
```

## 🚀 Quick Start

```python
from datasets import load_dataset
import pandas as pd

# Load main dataset
dataset = load_dataset("deepearth/central-florida-plants")

# Access data
train_data = dataset['train']
print(f"Training samples: {len(train_data)}")
print(f"Features: {train_data.features}")

# Load vision embeddings (download required due to size)
vision_index = pd.read_parquet("vision_index.parquet")
vision_data = pd.read_parquet("vision_embeddings/embeddings_000000.parquet")
```

## 📊 Data Fields

Each observation contains:

| Field | Type | Description |
|-------|------|-------------|
| `gbif_id` | int64 | Unique GBIF occurrence ID |
| `taxon_id` | string | GBIF taxon ID |
| `taxon_name` | string | Scientific species name |
| `latitude` | float | GPS latitude |
| `longitude` | float | GPS longitude |
| `year` | int | Observation year |
| `month` | int | Observation month |
| `day` | int | Observation day |
| `hour` | int | Observation hour (nullable) |
| `minute` | int | Observation minute (nullable) |
| `second` | int | Observation second (nullable) |
| `image_urls` | List[string] | URLs to observation images |
| `num_images` | int | Relative image number in GBIF occurrence |
| `has_vision` | bool | Vision embeddings available |
| `vision_file_indices` | List[int] | Indices to vision files |
| `language_embedding` | List[float] | 7,168-dim DeepSeek-V3 embedding |
| `split` | string | train/spatial_test/temporal_test |

## 🔄 Data Splits

The dataset uses rigorous spatiotemporal splits:

{
  "train": 30935,
  "temporal_test": 2730
}

- **Temporal Test**: All 2025 observations (future generalization)
- **Spatial Test**: 5 non-overlapping geographic regions
- **Train**: Remaining observations

## 🤖 Embeddings

### Language Embeddings (DeepSeek-V3)
- **Dimensions**: 7,168
- **Source**: Scientific species descriptions
- **Coverage**: All 232 species

### Vision Embeddings (V-JEPA-2)
- **Dimensions**: 6,488,064 values per embedding
- **Structure**: 8 temporal frames × 24×24 spatial patches × 1408 features
- **Model**: Vision Transformer Giant with self-supervised pretraining
- **Coverage**: 7,113 images
- **Storage**: Flattened arrays in parquet files (use provided utilities to reshape)

## 💡 Usage Examples

### Working with V-JEPA 2 Embeddings
```python
import numpy as np
import ast

# Load vision embedding
vision_df = pd.read_parquet("vision_embeddings/embeddings_000000.parquet")
row = vision_df.iloc[0]

# Reshape from flattened to 4D structure
embedding = row['embedding']
original_shape = ast.literal_eval(row['original_shape'])  # [4608, 1408]

# First to 2D: (4608 patches, 1408 features)
embedding_2d = embedding.reshape(original_shape)

# Then to 4D: (8 temporal, 24 height, 24 width, 1408 features)
embedding_4d = embedding_2d.reshape(8, 24, 24, 1408)

# Get specific temporal frame (0-7)
frame_0 = embedding_4d[0]  # Shape: (24, 24, 1408)

# Get mean embedding for image-level tasks
image_embedding = embedding_4d.mean(axis=(0, 1, 2))  # Shape: (1408,)
```

### Species Distribution Modeling
```python
# Filter observations for a specific species
species_data = dataset.filter(lambda x: x['taxon_name'] == 'Quercus virginiana')

# Use spatiotemporal data for distribution modeling
coords = [(d['latitude'], d['longitude']) for d in species_data]
```

### Multimodal Learning
```python
# Combine vision and language embeddings
for sample in dataset:
    if sample['has_vision']:
        lang_emb = sample['language_embedding']
        vision_idx = sample['vision_file_indices'][0]
        # Load corresponding vision embedding
        vision_emb = load_vision_embedding(vision_idx)
```

### Zero-shot Species Classification
```python
# Use language embeddings for zero-shot classification
species_embeddings = {
    species['taxon_name']: species['language_embedding']
    for species in dataset.unique('taxon_name')
}
```

## 📄 License

This dataset is released under the **MIT License**.

## 📚 Citation

If you use this dataset, please cite:

```bibtex
@dataset{deepearth_cf_plants_2024,
  title={DeepEarth Central Florida Native Plants: A Multimodal Biodiversity Dataset},
  author={DeepEarth Team},
  year={2024},
  version={0.2.0},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/deepearth/central-florida-plants}
}
```

## 🌟 Acknowledgments

We thank all citizen scientists who contributed observations through iNaturalist and GBIF. This dataset was created as part of the DeepEarth initiative for multimodal self-supervised ecological intelligence research.

## 🔗 Related Resources

- [DeepEarth Project](https://github.com/deepearth)
- [V-JEPA Model](https://ai.meta.com/vjepa/)
- [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)
- [GBIF Portal](https://www.gbif.org)


## 📈 Dataset Statistics

- **Total Size**: ~51 GB
- **Main Dataset**: 500 MB
- **Vision Embeddings**: 50 GB
- **Image URLs**: 31,136 total images referenced
- **Temporal Range**: 2019-2025
- **Geographic Scope**: Central Florida, USA

---
*Dataset prepared by the DeepEarth team for advancing multimodal self-supervised ecological intelligence research.*
