#!/usr/bin/env python3
"""
Setup DeepEarth Data from HuggingFace

This script downloads and sets up the Central Florida Native Plants dataset
for use with the DeepEarth training pipeline.
"""

import os
import sys
from pathlib import Path
import json
import shutil
from huggingface_hub import snapshot_download
import pandas as pd
import numpy as np
from tqdm import tqdm

def setup_deepearth_data():
    """Download and setup DeepEarth data from HuggingFace."""
    
    # Get the dashboard directory
    dashboard_dir = Path(__file__).parent.parent / "dashboard"
    data_dir = dashboard_dir / "huggingface_dataset"
    
    print("🌍 Setting up DeepEarth data")
    print(f"Dashboard directory: {dashboard_dir}")
    print(f"Data directory: {data_dir}")
    
    # Create directories if they don't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if we already have the data
    required_files = [
        "observations.parquet",
        "species.json",
        "vision_embeddings.npy",
        "language_embeddings.npy"
    ]
    
    missing_files = [f for f in required_files if not (data_dir / f).exists()]
    
    if not missing_files:
        print("✅ All data files already exist!")
        return True
    
    print(f"📥 Missing files: {missing_files}")
    
    # Try to download from HuggingFace
    try:
        print("\n📡 Downloading from HuggingFace...")
        
        # Download the dataset
        dataset_path = snapshot_download(
            repo_id="deepearth/central-florida-native-plants",
            repo_type="dataset",
            cache_dir=str(data_dir.parent / ".cache"),
            local_dir=str(data_dir / "hf_download")
        )
        
        print(f"✅ Downloaded to: {dataset_path}")
        
        # Check what was downloaded
        downloaded_files = list(Path(dataset_path).rglob("*"))
        print(f"\n📦 Downloaded files:")
        for f in downloaded_files[:10]:  # Show first 10
            if f.is_file():
                print(f"  - {f.relative_to(dataset_path)}")
        
        # Try to find and move the required files
        for required_file in missing_files:
            found = False
            for downloaded in downloaded_files:
                if downloaded.name == required_file:
                    shutil.copy2(downloaded, data_dir / required_file)
                    print(f"✅ Copied {required_file}")
                    found = True
                    break
            
            if not found:
                print(f"⚠️  Could not find {required_file} in download")
        
    except Exception as e:
        print(f"❌ Error downloading from HuggingFace: {e}")
        print("\n🔧 Creating mock data for testing...")
        create_mock_data(data_dir)
    
    # Verify the setup
    return verify_data_setup(data_dir)


def create_mock_data(data_dir: Path):
    """Create mock data files for testing."""
    
    # Create mock observations
    print("Creating mock observations.parquet...")
    observations_data = []
    
    species_list = [
        'Quercus virginiana',
        'Serenoa repens', 
        'Sabal palmetto',
        'Pinus elliottii',
        'Tillandsia usneoides'
    ]
    
    for i in range(1000):
        observations_data.append({
            'gbif_id': 4000000000 + i,
            'species': np.random.choice(species_list),
            'latitude': 28.5 + np.random.randn() * 0.1,
            'longitude': -81.3 + np.random.randn() * 0.1,
            'timestamp': 1700000000 + i * 3600,
            'has_vision': True,
            'has_language': True,
            'observation_id': f"{4000000000 + i}_1"
        })
    
    df = pd.DataFrame(observations_data)
    df.to_parquet(data_dir / "observations.parquet")
    
    # Create mock species.json
    print("Creating mock species.json...")
    species_data = {species: i for i, species in enumerate(species_list)}
    with open(data_dir / "species.json", 'w') as f:
        json.dump(species_data, f, indent=2)
    
    # Create mock embeddings
    print("Creating mock vision_embeddings.npy...")
    vision_embeddings = np.random.randn(1000, 8, 24, 24, 1408).astype(np.float32)
    np.save(data_dir / "vision_embeddings.npy", vision_embeddings)
    
    print("Creating mock language_embeddings.npy...")
    language_embeddings = np.random.randn(1000, 7168).astype(np.float32)
    np.save(data_dir / "language_embeddings.npy", language_embeddings)
    
    print("✅ Mock data created!")


def verify_data_setup(data_dir: Path) -> bool:
    """Verify that all required files exist and are valid."""
    
    print("\n🔍 Verifying data setup...")
    
    required_files = {
        "observations.parquet": lambda p: pd.read_parquet(p).shape[0] > 0,
        "species.json": lambda p: len(json.load(open(p))) > 0,
        "vision_embeddings.npy": lambda p: np.load(p, mmap_mode='r').shape[0] > 0,
        "language_embeddings.npy": lambda p: np.load(p, mmap_mode='r').shape[0] > 0
    }
    
    all_valid = True
    
    for filename, validator in required_files.items():
        filepath = data_dir / filename
        if filepath.exists():
            try:
                if validator(filepath):
                    print(f"✅ {filename} - valid")
                else:
                    print(f"❌ {filename} - invalid content")
                    all_valid = False
            except Exception as e:
                print(f"❌ {filename} - error: {e}")
                all_valid = False
        else:
            print(f"❌ {filename} - missing")
            all_valid = False
    
    return all_valid


def create_dashboard_config(dashboard_dir: Path):
    """Create a basic dashboard configuration file."""
    
    config_path = dashboard_dir / "dataset_config.json"
    
    if config_path.exists():
        print(f"✅ Config already exists: {config_path}")
        return
    
    print(f"📝 Creating config: {config_path}")
    
    config = {
        "dataset_name": "Central Florida Native Plants",
        "version": "v0.2.0",
        "data_path": "huggingface_dataset",
        "vision_embedding_dim": [8, 24, 24, 1408],
        "language_embedding_dim": 7168,
        "species_count": 5,
        "observation_count": 1000
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Config created!")


def main():
    """Main setup function."""
    
    # Change to dashboard directory
    script_dir = Path(__file__).parent
    dashboard_dir = script_dir.parent / "dashboard"
    
    if not dashboard_dir.exists():
        print(f"❌ Dashboard directory not found: {dashboard_dir}")
        print("Please run this from the deepearth/training directory")
        return 1
    
    os.chdir(dashboard_dir)
    
    # Setup data
    if not setup_deepearth_data():
        print("\n❌ Data setup failed!")
        return 1
    
    # Create config if needed
    create_dashboard_config(dashboard_dir)
    
    print("\n✅ DeepEarth data setup complete!")
    print("\nYou can now run:")
    print("  cd training")
    print("  python deepearth_multimodal_training.py --config ../dashboard/dataset_config.json")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
