#!/bin/bash
# Download script for Central Florida Native Plants dataset from Hugging Face

DATASET_NAME="central_florida_native_plants"
REPO_ID="deepearth/central_florida_native_plants"
LOCAL_DIR="$(dirname "$0")"

echo "======================================"
echo "Central Florida Native Plants Dataset"
echo "======================================"
echo ""

# Check if Python and required packages are installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed."
    echo "Please install Python 3: https://www.python.org/downloads/"
    exit 1
fi

# Create Python script for downloading
cat > /tmp/download_hf_dataset.py << 'EOF'
import os
import sys
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
local_dir = sys.argv[2]

print(f"Downloading from Hugging Face: {repo_id}")
print(f"Destination: {local_dir}")

try:
    # Download the dataset
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        resume_download=True
    )
    print("\nDownload complete!")
    
    # Count files
    vision_embeddings_dir = os.path.join(local_dir, "vision_embeddings")
    if os.path.exists(vision_embeddings_dir):
        embeddings_count = len([f for f in os.listdir(vision_embeddings_dir) if f.endswith('.parquet')])
    else:
        embeddings_count = 0
    
    print(f"Vision embeddings: {embeddings_count} parquet files")
    
    # Check for other key files
    if os.path.exists(os.path.join(local_dir, "observations.parquet")):
        print("✓ observations.parquet found")
    if os.path.exists(os.path.join(local_dir, "vision_index.parquet")):
        print("✓ vision_index.parquet found")
    if os.path.exists(os.path.join(local_dir, "dataset_info.json")):
        print("✓ dataset_info.json found")
    
    # Calculate total size
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(local_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
    
    print(f"\nTotal size: {total_size / (1024**2):.1f} MB")
    
except Exception as e:
    print(f"Error downloading dataset: {e}")
    print("\nPlease ensure huggingface-hub is installed:")
    print("  pip install huggingface-hub")
    sys.exit(1)
EOF

# Check if huggingface-hub is installed
python3 -c "import huggingface_hub" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing huggingface-hub..."
    pip install huggingface-hub
fi

# Run the download
python3 /tmp/download_hf_dataset.py "$REPO_ID" "$LOCAL_DIR"

# Clean up
rm -f /tmp/download_hf_dataset.py