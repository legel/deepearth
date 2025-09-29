# Angiosperm Flowering Intelligence Dataset

## Overview
A comprehensive dataset of angiosperm (flowering plant) observations with flowering predictions and multimodal embeddings for machine learning applications.

## Dataset Statistics
- **Total Observations**: 561,381
- **Unique Species**: 7,831
- **Taxonomic Orders**: 56
- **Taxonomic Families**: 242
- **Geographic Coverage**: Global

## Files

### Core Data Files
1. **`angiosperms.csv`** (7,831 rows)
   - Species-level information with taxonomic hierarchy
   - Observation counts per species
   - Links to species text embeddings

2. **`angiosperm_observations.csv`** (561,381 rows)
   - Individual observation records
   - Spatiotemporal coordinates (lat, lon, elevation, datetime)
   - Flowering probability predictions
   - Links to vision and environmental embeddings

### Embedding Files
1. **`angiosperm_bioclip2_vision.pt`**
   - BioCLIP 2 vision embeddings for observation images
   - Shape: [539332, 768]
   - Type: float32

2. **`angiosperm_alphaearth_embeddings.pt`**
   - AlphaEarth environmental embeddings
   - Shape: [561381, 768]
   - Type: int8 (quantized for efficiency)

3. **`angiosperm_species_bioclip.pt`**
   - BioCLIP 2 text embeddings for species taxonomic descriptions
   - Shape: [7831, 768]
   - Type: float32

## Coverage

### Embedding Coverage
- Vision embeddings: 96.1% of observations
- AlphaEarth embeddings: 100.0% of observations
- Elevation data: 100.0% of observations

### Flowering Predictions
- Mean flowering probability: 0.634
- High confidence (P > 0.8): 308,387 observations
- Confirmed flowering (P = 1.0): 68,023 observations

### Top Species by Observations
1. Acmispon glaber: 1,000 observations
2. Ficaria verna: 1,000 observations
3. Achillea millefolium: 1,000 observations
4. Hesperis matronalis: 1,000 observations
5. Taraxacum officinale: 1,000 observations
6. Symphyotrichum novae-angliae: 1,000 observations
7. Asclepias incarnata: 1,000 observations
8. Symplocarpus foetidus: 1,000 observations
9. Alliaria petiolata: 1,000 observations
10. Reynoutria japonica: 1,000 observations

## Data Schema

### angiosperms.csv
- `species_name`: Scientific name of the species
- `species_id`: GBIF species identifier
- `kingdom`, `phylum`, `class`, `order`, `family`, `genus`, `species`: Taxonomic hierarchy
- `observation_count`: Number of observations for this species
- `species_bioclip_embedding_index`: Index in species embeddings file

### angiosperm_observations.csv
- `species_name`, `species_id`: Species identifiers
- `latitude`, `longitude`: Geographic coordinates (WGS84)
- `elevation`: Elevation in meters (from AlphaEarth)
- `datetime`: Observation timestamp
- `geospatial_uncertainty`: Location accuracy in meters
- `occurrence_id`: GBIF occurrence identifier
- `flowering_prob`: Flowering probability (0-1) from PhenoVision
- `image_count`: Number of images for this observation
- `vision_embedding_index`: Index in BioCLIP vision embeddings
- `alphaearth_embedding_index`: Index in AlphaEarth embeddings

## Usage Example

```python
import pandas as pd
import torch

# Load species information
species = pd.read_csv('angiosperms.csv')

# Load observations
observations = pd.read_csv('angiosperm_observations.csv')

# Load embeddings
vision_embeddings = torch.load('angiosperm_bioclip2_vision.pt')
alphaearth_embeddings = torch.load('angiosperm_alphaearth_embeddings.pt')
species_embeddings = torch.load('angiosperm_species_bioclip.pt')

# Get embedding for a specific observation
obs = observations.iloc[0]
if obs['vision_embedding_index'] >= 0:
    vision_emb = vision_embeddings['embeddings'][obs['vision_embedding_index']]
```

## Citation
If you use this dataset, please cite:
[Citation information to be added]

## License
[License information to be added]

## Data Sources
- Observations from GBIF (Global Biodiversity Information Facility)
- Flower visitation data from Global Biotic Interactions (GloBI)
- Images from iNaturalist
- Elevation data from Copernicus DEM via AlphaEarth
- Taxonomic backbone from GBIF

## Processing Pipeline
1. GBIF angiosperm occurrences filtered by taxonomy
2. Flower visitation events integrated
3. Quality filters applied (coordinate uncertainty ≤50m, ≥5 observations per species)
4. BioCLIP 2 embeddings extracted for images and species text
5. PhenoVision flowering predictions computed
6. AlphaEarth environmental embeddings integrated

---
Generated: 2025-09-29
