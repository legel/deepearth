# Central Florida Native Plants Dataset

This dataset contains language embeddings for 232 native plant species from Central Florida.

## Contents

- **embeddings/**: PyTorch tensor files (.pt) containing 5120-dimensional language embeddings
- **tokens/**: CSV files containing token mappings for each species
- **metadata.json**: Dataset metadata including model information

## Model Information

- **Model**: DeepSeek-V3-0324-UD-Q4_K_XL (671B parameters, 4.5-bit quantized)
- **Embedding Dimension**: 5120
- **Prompt Template**: "Ecophysiology of {species_name}:"

## Download Instructions

### Prerequisites

1. Install Google Cloud SDK: https://cloud.google.com/sdk/install
2. Authenticate with Google Cloud:
   ```bash
   gcloud auth login
   ```

### Download Dataset

Run the download script:
```bash
./download_dataset.sh
```

This will download:
- 232 embedding files (`.pt` format)
- 232 token mapping files (`.csv` format)
- Dataset metadata

## File Format

### Embeddings
Each `.pt` file contains a PyTorch tensor of shape `[5120]` representing the language embedding for one species. Files are named by GBIF taxon ID.

### Token Mappings
Each `.csv` file contains the tokenization information with columns:
- `position`: Token position in sequence
- `token_id`: Token ID in model vocabulary
- `token`: Token string

## Usage Example

```python
import torch
import pandas as pd

# Load embedding
species_id = "2650927"  # Example GBIF ID
embedding = torch.load(f"embeddings/{species_id}.pt")

# Load token mapping
tokens = pd.read_csv(f"tokens/{species_id}.csv")
```

## Citation

If you use this dataset, please cite the DeepEarth project and DeepSeek model.