#!/bin/bash

# DeepEarth Multimodal Training Setup Script
# This script sets up the environment for masked language/vision reconstruction

echo "🌍 DeepEarth Multimodal Training Setup"
echo "======================================"

# Check if we're in the right directory
if [ ! -d "dashboard" ]; then
    echo "❌ Error: Please run this script from the DeepEarth root directory"
    echo "   Expected to find 'dashboard' directory"
    exit 1
fi

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p training/{models/{checkpoints,final},configs,results/{multimodal_masking_results,visualizations,training_curves},scripts}
mkdir -p data/{embeddings/{vision,language},raw}
mkdir -p docs/{training_guides,api_reference}
mkdir -p notebooks tests

# Create reconstruction directory for compatibility
mkdir -p reconstruction/mlp_unet/{checkpoints,logs,visualizations}

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "🐍 Found Python version: $PYTHON_VERSION"

# Check if virtual environment exists
VENV_NAME="venv_mlp_unet"
if [ -d "$VENV_NAME" ]; then
    echo "✅ Using existing virtual environment: $VENV_NAME"
else
    echo "🔧 Creating virtual environment: $VENV_NAME"
    python3 -m venv "$VENV_NAME"
fi

# Activate virtual environment
echo "🚀 Activating virtual environment..."
source "$VENV_NAME/bin/activate"

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (with CUDA support if available)
echo "🔥 Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected, installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "ℹ️  No NVIDIA GPU detected, installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio
fi

# Install core dependencies for multimodal training
echo "📚 Installing multimodal training dependencies..."
pip install numpy>=1.24.0
pip install pandas>=2.0.0
pip install matplotlib>=3.7.0
pip install scikit-learn>=1.3.0
pip install tqdm>=4.65.0
pip install umap-learn>=0.5.3

# Dashboard dependencies (needed for data loading)
echo "🌐 Installing dashboard dependencies..."
pip install flask>=2.3.0
pip install pyarrow>=12.0.0  # For parquet files
pip install pillow>=9.5.0
pip install requests>=2.31.0

# HuggingFace for datasets
echo "🤗 Installing HuggingFace libraries..."
pip install huggingface-hub>=0.16.0
pip install datasets>=2.14.0
pip install transformers>=4.30.0

# V-JEPA2 dependencies
echo "🎥 Installing V-JEPA2 dependencies..."
pip install einops>=0.6.0
pip install timm>=0.9.0

# Logging and visualization
echo "📊 Installing logging tools..."
pip install tensorboard>=2.13.0
pip install wandb>=0.15.0  # Optional: for cloud logging
pip install seaborn>=0.12.0

# Development tools
echo "🛠️  Installing development tools..."
pip install jupyter ipython black flake8 pytest

# Install scipy for additional scientific computing
pip install scipy>=1.10.0

# Create requirements.txt for the project
echo "📝 Creating requirements.txt..."
cat > requirements.txt << EOF
# Core dependencies
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
tqdm>=4.65.0
umap-learn>=0.5.3
scipy>=1.10.0

# Dashboard dependencies
flask>=2.3.0
pyarrow>=12.0.0
pillow>=9.5.0
requests>=2.31.0

# HuggingFace
huggingface-hub>=0.16.0
datasets>=2.14.0
transformers>=4.30.0

# V-JEPA2 dependencies
einops>=0.6.0
timm>=0.9.0

# Logging and visualization
tensorboard>=2.13.0
wandb>=0.15.0
seaborn>=0.12.0

# Development
jupyter>=1.0.0
ipython>=8.12.0
black>=23.0.0
flake8>=6.0.0
pytest>=7.4.0
EOF

# Create __init__.py files
touch training/__init__.py
touch reconstruction/__init__.py
touch reconstruction/mlp_unet/__init__.py

# Create .gitignore if it doesn't exist
if [ ! -f .gitignore ]; then
    echo "📝 Creating .gitignore..."
    cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv*/
.env

# Data (too large for git)
data/embeddings/
data/raw/
*.npy
*.pt
*.pth
*.parquet

# Model checkpoints
training/models/checkpoints/
training/models/final/
reconstruction/mlp_unet/checkpoints/
*.ckpt

# Results
training/results/
reconstruction/mlp_unet/visualizations/
*.png
*.jpg

# Logs
*.log
logs/
tensorboard/
wandb/

# Notebooks
notebooks/.ipynb_checkpoints/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Dashboard cache
dashboard/.cache/
dashboard/huggingface_dataset/hf_download/
EOF
fi

# Create a default multimodal config
echo "⚙️  Creating default configuration..."
mkdir -p training/configs
cat > training/configs/multimodal_config.yaml << EOF
# DeepEarth Multimodal Training Configuration
data:
  dashboard_config: "../dashboard/dataset_config.json"
  max_samples: 10000
  train_split: 0.8
  batch_size: 32
  
model:
  vision_dim: 1408      # V-JEPA2 output dimension
  language_dim: 7168    # DeepSeek output dimension
  hidden_dim: 256       # MLP hidden dimension
  universal_dim: 2048   # Universal embedding dimension
  dropout: 0.1
  
training:
  epochs: 50
  learning_rate: 0.001
  device: "auto"
  mask_language: true
  mask_prob: 0.5
  visualize_every: 10
  
species:
  # Central Florida species to focus on
  target_species:
    - "Quercus virginiana"      # Southern live oak
    - "Serenoa repens"          # Saw palmetto
    - "Sabal palmetto"          # Cabbage palm
    - "Pinus elliottii"         # Slash pine
    - "Tillandsia usneoides"    # Spanish moss
    - "Zamia integrifolia"      # Coontie
    - "Magnolia grandiflora"    # Southern magnolia
    - "Ilex vomitoria"          # Yaupon holly
    - "Juniperus virginiana"    # Eastern red cedar
    - "Vaccinium darrowii"      # Darrow's blueberry
    
logging:
  use_tensorboard: true
  use_wandb: false
  log_dir: "logs/"
  checkpoint_dir: "models/checkpoints/"
EOF

# Check if dashboard data exists
echo ""
echo "🔍 Checking dashboard data..."
if [ -d "dashboard/huggingface_dataset" ]; then
    echo "✅ Found dashboard data directory"
else
    echo "⚠️  Dashboard data directory not found"
    echo "   You may need to download the data from HuggingFace"
fi

# Clone V-JEPA2 if requested
echo ""
read -p "📥 Would you like to clone V-JEPA2 repository? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ ! -d "vjepa2" ]; then
        echo "Cloning V-JEPA2..."
        git clone https://github.com/facebookresearch/vjepa2.git
    else
        echo "V-JEPA2 already exists"
    fi
fi

# Display summary
echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "📍 Project structure created:"
echo "   training/"
echo "   ├── configs/"
echo "   │   └── multimodal_config.yaml"
echo "   ├── models/"
echo "   │   ├── checkpoints/"
echo "   │   └── final/"
echo "   ├── results/"
echo "   │   ├── multimodal_masking_results/"
echo "   │   ├── visualizations/"
echo "   │   └── training_curves/"
echo "   └── scripts/"
echo ""
echo "   data/"
echo "   ├── embeddings/"
echo "   │   ├── vision/"
echo "   │   └── language/"
echo "   └── raw/"
echo ""
echo "🎯 Next steps:"
echo "   1. Ensure dashboard data is available:"
echo "      python training/setup_deepearth_data.py"
echo ""
echo "   2. Start multimodal training:"
echo "      cd training"
echo "      python deepearth_multimodal_training.py --config ../dashboard/dataset_config.json"
echo ""
echo "   3. Or use the YAML config:"
echo "      python deepearth_multimodal_training.py --config configs/multimodal_config.yaml"
echo ""
echo "💡 To monitor training:"
echo "   tensorboard --logdir=training/logs"
echo ""
echo "🔍 To test V-JEPA2 integration:"
echo "   python training/vjepa2_integration.py"
echo ""
echo "📊 Virtual environment info:"
echo "   Activate: source $VENV_NAME/bin/activate"
echo "   Deactivate: deactivate"
