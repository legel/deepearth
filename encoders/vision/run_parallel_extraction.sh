#!/bin/bash
# Parallel V-JEPA 2 feature extraction script for multiple GPUs

echo "V-JEPA 2 Parallel Feature Extraction"
echo "===================================="

# Configuration
IMAGE_DIR="${1:-/path/to/images}"
OUTPUT_DIR="${2:-/path/to/output}"
NUM_GPUS="${3:-2}"
CHUNK_SIZE="${4:-1000}"

# Check if directories exist
if [ ! -d "$IMAGE_DIR" ]; then
    echo "Error: Image directory not found: $IMAGE_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get total number of images
TOTAL_IMAGES=$(find "$IMAGE_DIR" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" | wc -l)
echo "Total images found: $TOTAL_IMAGES"

# Calculate images per GPU
IMAGES_PER_GPU=$((TOTAL_IMAGES / NUM_GPUS))
echo "Images per GPU: ~$IMAGES_PER_GPU"

# Check GPU availability
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# Function to run extraction on a specific GPU
run_gpu_extraction() {
    local gpu_id=$1
    local start_idx=$2
    local end_idx=$3
    
    echo "Starting GPU $gpu_id: processing images $start_idx to $end_idx"
    
    python vjepa2_extractor.py \
        --image_dir "$IMAGE_DIR" \
        --output_dir "$OUTPUT_DIR/gpu_$gpu_id" \
        --device "cuda:$gpu_id" \
        --chunk_size "$CHUNK_SIZE" \
        > "$OUTPUT_DIR/gpu_${gpu_id}.log" 2>&1 &
    
    echo "GPU $gpu_id process started (PID: $!)"
}

# Launch parallel processes
echo "Launching $NUM_GPUS parallel extraction processes..."
for ((i=0; i<$NUM_GPUS; i++)); do
    start_idx=$((i * IMAGES_PER_GPU))
    end_idx=$(((i + 1) * IMAGES_PER_GPU))
    
    # Last GPU handles remaining images
    if [ $i -eq $((NUM_GPUS - 1)) ]; then
        end_idx=$TOTAL_IMAGES
    fi
    
    run_gpu_extraction $i $start_idx $end_idx
done

echo ""
echo "All GPU processes launched. Monitor progress with:"
echo "  tail -f $OUTPUT_DIR/gpu_*.log"
echo ""
echo "To check GPU usage:"
echo "  watch -n 1 nvidia-smi"

# Wait for all processes to complete
wait

echo ""
echo "All extraction processes completed!"

# Merge results from different GPUs
echo "Merging results..."
python -c "
import torch
from pathlib import Path
import shutil

output_dir = Path('$OUTPUT_DIR')
final_features = {}

# Collect features from all GPU subdirectories
for gpu_dir in sorted(output_dir.glob('gpu_*')):
    if gpu_dir.is_dir():
        for chunk_file in gpu_dir.glob('features_chunk_*.pt'):
            print(f'Loading {chunk_file}')
            chunk_data = torch.load(chunk_file)
            final_features.update(chunk_data)

# Save merged features
if final_features:
    # Save in chunks
    chunk_size = $CHUNK_SIZE
    chunk_id = 0
    current_chunk = {}
    
    for img_id, features in final_features.items():
        current_chunk[img_id] = features
        
        if len(current_chunk) >= chunk_size:
            output_file = output_dir / f'merged_features_chunk_{chunk_id:04d}.pt'
            torch.save(current_chunk, output_file)
            print(f'Saved {output_file}')
            current_chunk = {}
            chunk_id += 1
    
    # Save final chunk
    if current_chunk:
        output_file = output_dir / f'merged_features_chunk_{chunk_id:04d}.pt'
        torch.save(current_chunk, output_file)
        print(f'Saved {output_file}')
    
    print(f'Total images processed: {len(final_features)}')
"

echo "Feature extraction complete!"
echo "Results saved in: $OUTPUT_DIR"