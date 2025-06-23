#!/bin/bash
# Download script for Central Florida Native Plants dataset
# This script downloads language embeddings and token mappings from Google Cloud Storage

DATASET_NAME="central_florida_native_plants"
GCS_BUCKET="gs://deepearth"
GCS_PATH="${GCS_BUCKET}/encodings/language/taxa"
LOCAL_DIR="$(dirname "$0")"

echo "======================================"
echo "Central Florida Native Plants Dataset"
echo "======================================"
echo ""

# Check if gsutil is installed
if ! command -v gsutil &> /dev/null; then
    echo "Error: gsutil is not installed."
    echo "Please install Google Cloud SDK: https://cloud.google.com/sdk/install"
    exit 1
fi

# Check authentication
echo "Checking Google Cloud authentication..."
if ! gsutil ls "$GCS_BUCKET" &> /dev/null; then
    echo ""
    echo "Authentication required. Please run:"
    echo "  gcloud auth login"
    echo ""
    echo "Then try this script again."
    exit 1
fi

echo "Downloading ${DATASET_NAME} dataset..."
echo "Source: ${GCS_PATH}"
echo "Destination: ${LOCAL_DIR}"

# Create local directories
mkdir -p "${LOCAL_DIR}/embeddings"
mkdir -p "${LOCAL_DIR}/tokens"

# Download embeddings
echo ""
echo "Downloading embeddings (.pt files)..."
gsutil -m cp "${GCS_PATH}/embeddings/*.pt" "${LOCAL_DIR}/embeddings/"

# Download token mappings
echo ""
echo "Downloading token mappings (.csv files)..."
gsutil -m cp "${GCS_PATH}/tokens/*.csv" "${LOCAL_DIR}/tokens/"

# Download metadata
echo ""
echo "Downloading metadata..."
gsutil cp "${GCS_PATH}/metadata.json" "${LOCAL_DIR}/"

# Count downloaded files
echo ""
echo "Download complete!"
echo "Embeddings: $(ls -1 ${LOCAL_DIR}/embeddings/*.pt 2>/dev/null | wc -l) files"
echo "Token mappings: $(ls -1 ${LOCAL_DIR}/tokens/*.csv 2>/dev/null | wc -l) files"
echo ""
echo "Total size: $(du -sh ${LOCAL_DIR} | cut -f1)"