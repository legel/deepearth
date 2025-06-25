# DeepEarth Central Florida Native Plants Dataset v0.2.0

<p align="center">
  <img src="https://github.com/deepearth/assets/deepearth-logo.png" alt="DeepEarth" width="200"/>
</p>

**🔗 Hugging Face Dataset**: https://huggingface.co/datasets/deepearth/central-florida-native-plants

## 🌿 Dataset Summary

A comprehensive multimodal biodiversity dataset featuring **33,665 observations** of **232 native plant species** from Central Florida. This dataset combines citizen science observations with state-of-the-art vision and language embeddings for advancing biodiversity monitoring and species distribution modeling.

### Key Features
- 🌍 **Spatiotemporal Coverage**: Complete GPS coordinates and timestamps for all observations
- 🖼️ **Multimodal**: 31,136 observations with images, 7,113 with vision embeddings
- 🧬 **Language Embeddings**: DeepSeek-V3 embeddings for all 232 species
- 👁️ **Vision Embeddings**: V-JEPA-2 self-supervised features (6.5M dimensions)
- 📊 **Rigorous Splits**: Spatiotemporal train/test splits for robust evaluation

## 📦 Dataset Structure

```
observations.parquet         # Main dataset (922MB)
vision_index.parquet        # Vision embeddings index
vision_embeddings/          # Vision features (~50GB total)
├── embeddings_000000.parquet
├── embeddings_000001.parquet
└── ... (159 files)
dataset_info.json           # Dataset metadata
vision_metadata.json        # Vision embedding metadata
README.md                   # This documentation
```

## 🚀 Quick Start

### Loading from Hugging Face

```python
from datasets import load_dataset
import pandas as pd
import numpy as np

# Load main dataset
dataset = load_dataset("deepearth/central-florida-native-plants")

# Access data
train_data = dataset['train']
print(f"Training samples: {len(train_data)}")
print(f"Features: {train_data.features}")

# Load vision embeddings (download required due to size)
vision_index = pd.read_parquet("vision_index.parquet")
vision_data = pd.read_parquet("vision_embeddings/embeddings_000000.parquet")
```

### Using the Dataset Class

```python
# Use our provided dataset class for easy access
from dataset_usage_examples import CentralFloridaPlantsDataset

# Initialize dataset
dataset = CentralFloridaPlantsDataset("path/to/downloaded/dataset")

# Get observations with filters
recent_plants = dataset.get_observations(
    year_range=(2020, 2025),
    bbox=(28.0, -82.0, 29.0, -81.0)  # Central Florida region
)

# Load and reshape vision embeddings
gbif_id = recent_plants.iloc[0]['gbif_id']
vision_embedding = dataset.load_vision_embedding(gbif_id)
embedding_4d = dataset.reshape_vision_embedding(vision_embedding)
```

## 📊 Data Fields

Each observation contains:

| Field | Type | Description |
|-------|------|-------------|
| `gbif_id` | int64 | Unique GBIF occurrence ID |
| `taxon_id` | string | Species identifier |
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
| `num_images` | int | Number of images |
| `has_vision` | bool | Vision embeddings available |
| `language_embedding` | List[float] | 7,168-dim DeepSeek-V3 embedding |
| `split` | string | train/temporal_test |

## 🔄 Data Splits

The dataset uses rigorous spatiotemporal splits:

- **Train**: 30,935 observations (91.9%) - 2010-2024 data
- **Temporal Test**: 2,730 observations (8.1%) - All 2025 observations for future generalization

## 🤖 Embeddings

### Language Embeddings (DeepSeek-V3)
- **Dimensions**: 7,168
- **Source**: Scientific species descriptions
- **Coverage**: All 232 species
- **Quality**: Full precision (not quantized)

### Vision Embeddings (V-JEPA-2)
- **Dimensions**: 6,488,064 values per embedding
- **Structure**: 8 temporal frames × 24×24 spatial patches × 1,408 features
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

# Reshape from flattened to 4D structure (validated implementation)
embedding = row['embedding']
original_shape = ast.literal_eval(row['original_shape'])  # [4608, 1408]

# Step 1: Reshape to 2D
embedding_2d = embedding.reshape(original_shape)  # (4608, 1408)

# Step 2: Reshape to 3D (dashboard validated structure)
embedding_3d = embedding_2d.reshape(8, 576, 1408)  # (temporal, spatial, features)

# Step 3: Reshape spatial patches to 2D grid
embedding_4d = embedding_3d.reshape(8, 24, 24, 1408)  # (temporal, height, width, features)

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
        gbif_id = sample['gbif_id']
        # Load corresponding vision embedding using vision_index
        vision_emb = load_vision_embedding(gbif_id)
```

### Zero-shot Species Classification

```python
# Use language embeddings for zero-shot classification
species_embeddings = {
    species['taxon_name']: species['language_embedding']
    for species in dataset.unique('taxon_name')
}
```

## 📍 Geographic Coverage

- **Region**: Central Florida, USA
- **Bounds**: 28.033°N to 28.978°N, 80.902°W to 81.934°W
- **Area**: ~115 × 105 km
- **Habitats**: Wetlands, forests, scrublands, coastal areas

## 🔗 Original Data Sources

Each observation retains complete traceability:

- **Original Image URLs**: Preserved in `image_urls` field
- **Original Metadata**: Encoded in vision embedding filenames
- **GBIF Records**: All observations linked to GBIF occurrence IDs
- **Filename Format**: `gbif_{gbif_id}_taxon_{taxon_id}_img_{image_num}_{split}_features.pt`

## 🛠️ Installation & Tools

### Download from Hugging Face

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Download the dataset
huggingface-cli download deepearth/central-florida-native-plants --repo-type dataset
```

### Validation & Testing

```bash
# Run comprehensive integrity tests
python3 test_dataset_integrity.py --dataset-dir path/to/dataset

# Explore the dataset with examples
python3 dataset_usage_examples.py --dataset-dir path/to/dataset
```

### Dependencies

```bash
pip install pandas numpy torch datasets huggingface_hub tqdm scikit-learn scipy
```

## 📈 Dataset Statistics

- **Total Size**: ~51 GB
- **Main Dataset**: 922 MB
- **Vision Embeddings**: ~50 GB (159 files)
- **Image URLs**: 31,136 total images referenced
- **Temporal Range**: 2010-2025
- **Geographic Scope**: Central Florida, USA

## 🧪 Validation

This dataset has been thoroughly validated with comprehensive integrity tests covering:

- ✅ V-JEPA 2 embedding structure validation (based on production dashboard implementation)
- ✅ Language embedding consistency 
- ✅ Spatiotemporal data integrity
- ✅ Cross-references between all components
- ✅ File integrity and accessibility
- ✅ Statistical accuracy

## 🔬 Research Applications

- **Multimodal species classification**
- **Spatiotemporal distribution modeling**
- **Cross-modal retrieval** (text ↔ image)
- **Ecological niche modeling**
- **Biodiversity monitoring with AI**
- **Self-supervised learning** for biodiversity
- **Vision-language model training**

## 📄 License

This dataset is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

## 📚 Citation

If you use this dataset, please cite:

```bibtex
@dataset{deepearth_cf_plants_2025,
  title={DeepEarth Central Florida Native Plants: A Multimodal Biodiversity Dataset},
  author={DeepEarth Team},
  year={2025},
  version={0.2.0},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/deepearth/central-florida-native-plants}
}
```

## 🌟 Acknowledgments

We thank all citizen scientists who contributed observations through iNaturalist and GBIF. This dataset was created as part of the DeepEarth initiative for multimodal biodiversity monitoring.

## 🔗 Related Resources

- [DeepEarth Project](https://github.com/deepearth)
- [V-JEPA Model](https://github.com/facebookresearch/jepa)
- [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)
- [GBIF Portal](https://www.gbif.org)

## ⚠️ Ethical Considerations

- All observations are from public citizen science platforms
- No endangered species location data is included at fine resolution
- Please respect local regulations when using location data

## 🤝 Contributing

We welcome contributions! Please see our [contribution guidelines](CONTRIBUTING.md) for details on:

- Reporting issues
- Suggesting improvements
- Contributing additional data
- Extending the analysis tools

## 📧 Contact

For questions, collaborations, or support:

- **Issues**: [GitHub Issues](https://github.com/deepearth/central_florida_native_plants/issues)
- **Discussions**: [GitHub Discussions](https://github.com/deepearth/central_florida_native_plants/discussions)
- **Email**: contact@deepearth.ai

---

*Dataset prepared by the DeepEarth team for advancing multimodal biodiversity research.*

**Version**: 0.2.0  
**Last Updated**: June 2025  
**Dataset URL**: https://huggingface.co/datasets/deepearth/central-florida-native-plants