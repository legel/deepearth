#!/bin/bash
# Deploy DeepEarth Dashboard to ecodash.ai server

echo "Deploying DeepEarth Dashboard to ecodash.ai..."

# Server details
SERVER="photon@34.71.213.117"
REMOTE_DIR="/var/www/ecodash/private/deepearth"
LOCAL_DIR="/home/photon/4tb/deepseek/deepearth/dashboard"

# Files to copy (excluding large binary files)
echo "Copying dashboard files..."
rsync -avz --progress \
    --exclude "*.mmap" \
    --exclude "*.db" \
    --exclude "__pycache__" \
    --exclude "*.pyc" \
    --exclude "venv/" \
    --exclude ".git/" \
    --exclude "huggingface_dataset/" \
    --exclude "cache/" \
    -e "ssh -i ~/.ssh/id_ed25519" \
    "$LOCAL_DIR/" "$SERVER:$REMOTE_DIR/"

echo "Deployment complete!"
echo "Note: You will need to:"
echo "1. Set up the Python environment on the server"
echo "2. Configure the dataset paths"
echo "3. Add the route to web_server.py"