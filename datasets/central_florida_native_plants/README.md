---
license: mit
task_categories:
- feature-extraction
language:
- en
tags:
- biology
- ecology
- plants
- embeddings
- florida
- biodiversity
pretty_name: Central Florida Native Plants Language Embeddings
size_categories:
- n<1K
---

# Central Florida Native Plants Language Embeddings

This dataset contains language embeddings for 232 native plant species from Central Florida, extracted using the DeepSeek-V3 language model.

## Dataset Description

- **Curated by:** DeepEarth Project
- **Language(s):** English
- **License:** MIT

### Dataset Summary

This dataset provides pre-computed language embeddings for Central Florida plant species. Each species has been encoded using the prompt "Ecophysiology of {species_name}:" to capture semantic information about the plant's ecological characteristics.

## Dataset Structure

### Data Instances

Each species is represented by:
- A PyTorch file (`.pt`) containing a dictionary with embeddings and metadata
- A CSV file containing the token mappings

### Embedding File Structure

Each `.pt` file contains a dictionary with:
- `mean_embedding`: Tensor of shape `[7168]` - mean-pooled embedding across all tokens (including prompt)
- `token_embeddings`: Tensor of shape `[num_tokens, 7168]` - individual token embeddings
- `species_name`: String - the species name
- `taxon_id`: String - GBIF taxon ID
- `num_tokens`: Integer - number of tokens (typically 18-20)
- `embedding_stats`: Dictionary with embedding statistics
- `timestamp`: String - when the embedding was created

### Dataset Viewer Structure

The Parquet files in the dataset viewer contain:
- `taxon_id`: GBIF taxonomic identifier
- `species_name`: Scientific name of the plant species
- `timestamp`: When the embedding was created
- `token_position`: Position of token in sequence
- `token_id`: Token ID in model vocabulary
- `token_str`: String representation of token
- `is_species_token`: Whether this token is part of the species name
- `token_embedding`: 7168-dimensional embedding vector for this specific token
- `species_mean_embedding`: 7168-dimensional mean embedding of species name tokens only
- `all_tokens_mean_embedding`: 7168-dimensional mean embedding across all tokens (including prompt)
- `num_tokens`: Total number of tokens for this species
- `num_species_tokens`: Number of tokens that are part of the species name

### Token Mapping Structure

Token mapping CSV files contain:
- `position`: Token position in sequence
- `token_id`: Token ID in model vocabulary  
- `token`: Token string representation

### Data Splits

This dataset contains a single split with embeddings for all 232 species.

## Important Note on Embeddings

This dataset provides two types of mean embeddings:

1. **`species_mean_embedding`** (in dataset viewer): The mean embedding calculated from ONLY the tokens that represent the species name itself. This provides a more focused representation of the species.

2. **`all_tokens_mean_embedding`** or `mean_embedding` (in .pt files): The mean embedding calculated from ALL tokens in the prompt, including "Ecophysiology of", the species name, and the ":" token. This is the original embedding as extracted from the model.

For most use cases, `species_mean_embedding` is recommended as it captures the semantic representation of the species name without the influence of the prompt template.

## Dataset Creation

### Model Information

- **Model**: DeepSeek-V3-0324-UD-Q4_K_XL
- **Parameters**: 671B (4.5-bit quantized GGUF format)
- **Embedding Dimension**: 7168
- **Context**: 2048 tokens
- **Prompt Template**: "Ecophysiology of {species_name}:"

### Source Data

Species names are based on GBIF (Global Biodiversity Information Facility) taxonomy for plants native to Central Florida.

## Usage

### Loading Embeddings

```python
import torch
import pandas as pd
from huggingface_hub import hf_hub_download

# Download a specific embedding
repo_id = "deepearth/central_florida_native_plants"
species_id = "2650927"  # Example GBIF ID

# Download embedding file
embedding_path = hf_hub_download(
    repo_id=repo_id,
    filename=f"embeddings/{species_id}.pt",
    repo_type="dataset"
)

# Load embedding dictionary
data = torch.load(embedding_path)

# Access embeddings
mean_embedding = data['mean_embedding']  # Shape: [7168] - mean of all tokens
token_embeddings = data['token_embeddings']  # Shape: [num_tokens, 7168]
species_name = data['species_name']

print(f"Species: {species_name}")
print(f"Mean embedding shape: {mean_embedding.shape}")
print(f"Token embeddings shape: {token_embeddings.shape}")

# For species-only mean embedding, use the dataset viewer or compute from species tokens
# The dataset viewer provides 'species_mean_embedding' which is the mean of only
# the tokens that are part of the species name (excluding prompt tokens)

# Download and load token mapping
token_path = hf_hub_download(
    repo_id=repo_id,
    filename=f"tokens/{species_id}.csv",
    repo_type="dataset"
)
tokens = pd.read_csv(token_path)
```

### Batch Download

```python
from huggingface_hub import snapshot_download

# Download entire dataset
local_dir = snapshot_download(
    repo_id="deepearth/central_florida_native_plants",
    repo_type="dataset",
    local_dir="./florida_plants"
)
```

## Additional Information

### Dataset Curators

This dataset was created by the [DeepEarth Project](https://github.com/legel/deepearth) to enable machine learning research on biodiversity and ecology.

### Licensing Information

This dataset is licensed under the MIT License.

### Citation Information

```bibtex
@dataset{deepearth_florida_plants_2025,
  title={Central Florida Native Plants Language Embeddings},
  author={DeepEarth Project},
  year={2025},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/deepearth/central_florida_native_plants}}
}
```

### Contributions

Thanks to [@legel](https://github.com/legel) for creating this dataset.