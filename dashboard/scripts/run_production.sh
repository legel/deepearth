#!/bin/bash
# Production startup script for DeepEarth Multimodal Explorer

# Set working directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
echo "Checking dependencies..."
pip install -q -r requirements.txt

# Check if gunicorn is installed
if ! command -v gunicorn &> /dev/null; then
    echo "Installing gunicorn..."
    pip install gunicorn
fi

# Check critical files
echo "Verifying data files..."
if [ ! -f "embeddings.mmap" ]; then
    echo "ERROR: embeddings.mmap not found!"
    echo "This file is required for vision embedding access."
    exit 1
fi

if [ ! -f "embeddings_index.db" ]; then
    echo "ERROR: embeddings_index.db not found!"
    echo "This SQLite index is required for fast lookups."
    exit 1
fi

# Set production environment variables
export FLASK_ENV=production
export PYTHONUNBUFFERED=1

# Start gunicorn with optimal settings
echo "Starting DeepEarth Multimodal Explorer..."
echo "Dashboard will be available at http://localhost:5000"

# Run with 4 workers, 2 threads each, 120s timeout for large embeddings
exec gunicorn \
    --workers 4 \
    --threads 2 \
    --worker-class sync \
    --bind 0.0.0.0:5000 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    app_v2:app