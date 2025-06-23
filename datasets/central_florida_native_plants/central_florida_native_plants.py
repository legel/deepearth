"""Central Florida Native Plants Embeddings Dataset."""

import csv
import json
import os
from pathlib import Path

import datasets
import torch


_DESCRIPTION = """\
This dataset contains language embeddings for Central Florida native plant species.
Each example includes the species name, taxon ID, mean embedding vector, and associated metadata.
The embeddings were generated using DeepSeek-V3-0324-UD-Q4_K_XL model with the prompt template
"Ecophysiology of {species_name}:".
"""

_CITATION = """\
@dataset{central_florida_native_plants_2025,
  title={Central Florida Native Plants Language Embeddings},
  author={Unknown},
  year={2025},
  publisher={Hugging Face}
}
"""

_HOMEPAGE = ""

_LICENSE = ""

_VERSION = "1.0.0"


class CentralFloridaNativePlantsConfig(datasets.BuilderConfig):
    """BuilderConfig for Central Florida Native Plants."""

    def __init__(self, **kwargs):
        """BuilderConfig for Central Florida Native Plants.
        
        Args:
            **kwargs: keyword arguments forwarded to super.
        """
        super(CentralFloridaNativePlantsConfig, self).__init__(**kwargs)


class CentralFloridaNativePlants(datasets.GeneratorBasedBuilder):
    """Central Florida Native Plants embeddings dataset."""

    VERSION = datasets.Version(_VERSION)

    BUILDER_CONFIGS = [
        CentralFloridaNativePlantsConfig(
            name="default",
            version=VERSION,
            description="Central Florida Native Plants embeddings dataset",
        ),
    ]

    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        features = datasets.Features(
            {
                "species_name": datasets.Value("string"),
                "taxon_id": datasets.Value("string"),
                "mean_embedding": datasets.Sequence(datasets.Value("float32"), length=7168),
                "num_tokens": datasets.Value("int32"),
                "timestamp": datasets.Value("string"),
                "embedding_stats": {
                    "mean": datasets.Value("float32"),
                    "std": datasets.Value("float32"),
                    "min": datasets.Value("float32"),
                    "max": datasets.Value("float32"),
                },
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # The dataset files are already available locally
        data_dir = Path(dl_manager.manual_dir if dl_manager.manual_dir else ".")
        
        # Check if we're in the right directory
        embeddings_dir = data_dir / "embeddings"
        tokens_dir = data_dir / "tokens"
        metadata_path = data_dir / "metadata.json"
        
        # If not found, try current directory
        if not embeddings_dir.exists():
            data_dir = Path(".")
            embeddings_dir = data_dir / "embeddings"
            tokens_dir = data_dir / "tokens"
            metadata_path = data_dir / "metadata.json"
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "embeddings_dir": str(embeddings_dir),
                    "tokens_dir": str(tokens_dir),
                    "metadata_path": str(metadata_path),
                },
            ),
        ]

    def _generate_examples(self, embeddings_dir, tokens_dir, metadata_path):
        """Yields examples."""
        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Get all embedding files
        embedding_files = sorted(Path(embeddings_dir).glob("*.pt"))
        
        for idx, embedding_file in enumerate(embedding_files):
            taxon_id = embedding_file.stem
            
            # Load embedding data
            try:
                data = torch.load(str(embedding_file), weights_only=True)
                
                # Extract fields
                species_name = data["species_name"]
                mean_embedding = data["mean_embedding"].numpy().tolist()
                num_tokens = data["num_tokens"]
                timestamp = data["timestamp"]
                embedding_stats = data["embedding_stats"]
                
                # Yield the example
                yield idx, {
                    "species_name": species_name,
                    "taxon_id": taxon_id,
                    "mean_embedding": mean_embedding,
                    "num_tokens": num_tokens,
                    "timestamp": timestamp,
                    "embedding_stats": {
                        "mean": float(embedding_stats["mean"]),
                        "std": float(embedding_stats["std"]),
                        "min": float(embedding_stats["min"]),
                        "max": float(embedding_stats["max"]),
                    },
                }
                
            except Exception as e:
                print(f"Error loading {embedding_file}: {e}")
                continue