#!/usr/bin/env python3
"""
DeepEarth Dashboard Setup and Validation Script

This script helps set up and validate the DeepEarth dashboard installation,
checking for required files, dependencies, and configuration.

Author: DeepEarth Project
License: MIT
"""

import sys
import os
import json
from pathlib import Path
import subprocess
import argparse


class DashboardSetup:
    """Setup and validation for DeepEarth dashboard"""
    
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir).resolve()
        self.config_file = self.base_dir / "dataset_config.json"
        self.issues = []
        self.warnings = []
        self.success = []
        
    def check_python_version(self):
        """Check Python version"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            self.issues.append(f"❌ Python 3.8+ required (found {version.major}.{version.minor})")
        else:
            self.success.append(f"✅ Python {version.major}.{version.minor} is supported")
    
    def check_required_files(self):
        """Check for required files"""
        required_files = [
            ("deepearth_dashboard.py", "Main dashboard application"),
            ("dataset_config.json", "Dataset configuration"),
            ("requirements.txt", "Python dependencies"),
            ("templates/dashboard.html", "HTML template"),
            ("static/js/dashboard.js", "JavaScript frontend"),
            ("static/css/dashboard.css", "CSS styles"),
        ]
        
        for file_path, description in required_files:
            full_path = self.base_dir / file_path
            if full_path.exists():
                self.success.append(f"✅ Found: {file_path} ({description})")
            else:
                self.issues.append(f"❌ Missing: {file_path} ({description})")
    
    def check_dataset_config(self):
        """Validate dataset configuration"""
        if not self.config_file.exists():
            self.issues.append("❌ dataset_config.json not found")
            return
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            # Check dataset directory
            base_dir = config.get('data_paths', {}).get('base_dir', '')
            if base_dir.startswith(".."):
                # Relative path
                dataset_path = (self.base_dir / base_dir).resolve()
            else:
                dataset_path = Path(base_dir)
            
            if dataset_path.exists():
                self.success.append(f"✅ Dataset directory found: {dataset_path}")
                
                # Check specific files
                obs_file = dataset_path / config['data_paths'].get('observations', '')
                if obs_file.exists():
                    self.success.append(f"✅ Observations file found: {obs_file.name}")
                else:
                    self.issues.append(f"❌ Observations file missing: {obs_file}")
                
                vision_dir = dataset_path / config['data_paths'].get('vision_embeddings_dir', '')
                if vision_dir.exists():
                    parquet_files = list(vision_dir.glob("*.parquet"))
                    self.success.append(f"✅ Vision embeddings directory found with {len(parquet_files)} files")
                else:
                    self.issues.append(f"❌ Vision embeddings directory missing: {vision_dir}")
            else:
                self.issues.append(f"❌ Dataset directory not found: {dataset_path}")
                self.warnings.append(f"💡 Update 'base_dir' in dataset_config.json to point to your dataset")
            
        except Exception as e:
            self.issues.append(f"❌ Error reading dataset_config.json: {e}")
    
    def check_mmap_files(self):
        """Check for memory-mapped embedding files"""
        mmap_file = self.base_dir / "embeddings.mmap"
        index_db = self.base_dir / "embeddings_index.db"
        
        if mmap_file.exists():
            size_gb = mmap_file.stat().st_size / (1024**3)
            self.success.append(f"✅ Memory-mapped file found: embeddings.mmap ({size_gb:.1f} GB)")
        else:
            self.warnings.append("⚠️ embeddings.mmap not found - run prepare_embeddings.py to create it")
        
        if index_db.exists():
            self.success.append("✅ SQLite index found: embeddings_index.db")
        else:
            self.warnings.append("⚠️ embeddings_index.db not found - run prepare_embeddings.py to create it")
    
    def check_cache_directory(self):
        """Check and create cache directory"""
        cache_dir = self.base_dir / "cache"
        if not cache_dir.exists():
            try:
                cache_dir.mkdir(parents=True)
                self.success.append("✅ Created cache directory")
            except Exception as e:
                self.issues.append(f"❌ Failed to create cache directory: {e}")
        else:
            self.success.append("✅ Cache directory exists")
    
    def check_dependencies(self):
        """Check Python dependencies"""
        try:
            import flask
            self.success.append("✅ Flask is installed")
        except ImportError:
            self.issues.append("❌ Flask not installed - run: pip install -r requirements.txt")
        
        try:
            import torch
            self.success.append("✅ PyTorch is installed")
        except ImportError:
            self.issues.append("❌ PyTorch not installed - run: pip install torch")
        
        try:
            import pandas
            self.success.append("✅ Pandas is installed")
        except ImportError:
            self.issues.append("❌ Pandas not installed - run: pip install -r requirements.txt")
        
        try:
            import umap
            self.success.append("✅ UMAP is installed")
        except ImportError:
            self.warnings.append("⚠️ UMAP not installed - some features may not work")
        
        try:
            import hdbscan
            self.success.append("✅ HDBSCAN is installed")
        except ImportError:
            self.warnings.append("⚠️ HDBSCAN not installed - clustering features will not work")
    
    def create_setup_script(self):
        """Create a setup script for common tasks"""
        setup_content = """#!/bin/bash
# DeepEarth Dashboard Quick Setup

echo "🌍 DeepEarth Dashboard Setup"
echo "=========================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create cache directory
mkdir -p cache

# Check for dataset
if [ ! -d "../huggingface_dataset_v0.2.0" ]; then
    echo ""
    echo "⚠️  Dataset not found!"
    echo "Please download the dataset using one of these methods:"
    echo ""
    echo "1. Using prepare_embeddings.py:"
    echo "   python prepare_embeddings.py --download deepearth/central-florida-native-plants"
    echo ""
    echo "2. Manually from HuggingFace:"
    echo "   https://huggingface.co/datasets/deepearth/central-florida-native-plants"
    echo ""
fi

# Check for memory-mapped files
if [ ! -f "embeddings.mmap" ]; then
    echo ""
    echo "⚠️  Memory-mapped embeddings not found!"
    echo "Run: python prepare_embeddings.py /path/to/dataset"
    echo ""
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the dashboard:"
echo "  python deepearth_dashboard.py"
echo ""
"""
        
        setup_script = self.base_dir / "setup.sh"
        with open(setup_script, 'w') as f:
            f.write(setup_content)
        setup_script.chmod(0o755)
        self.success.append(f"✅ Created setup.sh script")
    
    def update_config_for_portability(self):
        """Update configuration to be more portable"""
        if not self.config_file.exists():
            return
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            # Check if we need to update paths
            base_dir = config.get('data_paths', {}).get('base_dir', '')
            if base_dir == "../huggingface_dataset_v0.2.0":
                # This is the default - suggest using environment variable
                updated_config = config.copy()
                
                # Add note about configuration
                updated_config['_configuration_note'] = (
                    "Update 'base_dir' to point to your dataset location. "
                    "You can use environment variables like ${DEEPEARTH_DATA_DIR}"
                )
                
                # Save updated config
                with open(self.config_file, 'w') as f:
                    json.dump(updated_config, f, indent=2)
                
                self.warnings.append(
                    "💡 Added configuration note to dataset_config.json. "
                    "Update 'base_dir' to point to your dataset."
                )
        except Exception as e:
            self.warnings.append(f"⚠️ Could not update config: {e}")
    
    def print_summary(self):
        """Print setup summary"""
        print("\n" + "="*60)
        print("🌍 DeepEarth Dashboard Setup Summary")
        print("="*60)
        
        if self.success:
            print("\n✅ Successful checks:")
            for item in self.success:
                print(f"  {item}")
        
        if self.warnings:
            print("\n⚠️  Warnings:")
            for item in self.warnings:
                print(f"  {item}")
        
        if self.issues:
            print("\n❌ Issues found:")
            for item in self.issues:
                print(f"  {item}")
            print("\n❗ Please fix these issues before running the dashboard")
        else:
            print("\n✅ All critical checks passed!")
            print("\nNext steps:")
            print("1. If you haven't already, prepare the embeddings:")
            print("   python prepare_embeddings.py /path/to/dataset")
            print("")
            print("2. Start the dashboard:")
            print("   python deepearth_dashboard.py")
        
        print("\n" + "="*60)
    
    def run_all_checks(self):
        """Run all validation checks"""
        print("🔍 Checking DeepEarth Dashboard setup...")
        
        self.check_python_version()
        self.check_required_files()
        self.check_dataset_config()
        self.check_mmap_files()
        self.check_cache_directory()
        self.check_dependencies()
        self.create_setup_script()
        self.update_config_for_portability()
        
        self.print_summary()
        
        return len(self.issues) == 0


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(
        description="Setup and validate DeepEarth Dashboard installation"
    )
    parser.add_argument('--dir', type=str, default='.',
                       help='Dashboard directory (default: current directory)')
    
    args = parser.parse_args()
    
    setup = DashboardSetup(args.dir)
    success = setup.run_all_checks()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())