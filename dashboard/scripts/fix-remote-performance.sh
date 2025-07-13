#!/bin/bash
# Script to fix DeepEarth performance issues on remote server

echo "🔧 Fixing DeepEarth performance issues..."

# 1. Update the systemd service with optimizations
echo "📝 Updating systemd service configuration..."
sudo tee /etc/systemd/system/deepearth.service << 'EOF'
[Unit]
Description=DeepEarth Dashboard Flask Application (Optimized)
After=network.target

[Service]
Type=simple
User=photon
Group=www-data
WorkingDirectory=/var/www/ecodash/private/deepearth/dashboard
Environment=PATH=/var/www/ecodash/private/deepearth/dashboard/venv/bin
Environment=FLASK_ENV=production
# Optimizations:
# - Increased timeout from 120s to 300s
# - Removed --preload to reduce memory usage
# - Added worker-tmp-dir for better performance
# - Added max-requests to prevent memory leaks
ExecStart=/var/www/ecodash/private/deepearth/dashboard/venv/bin/gunicorn \
    -w 2 \
    -b 127.0.0.1:5003 \
    --timeout 300 \
    --worker-tmp-dir /dev/shm \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    deepearth_dashboard:app \
    --access-logfile /var/www/ecodash/private/logs/deepearth-access.log \
    --error-logfile /var/www/ecodash/private/logs/deepearth-error.log
Restart=always
RestartSec=10
TimeoutStartSec=300
LimitNOFILE=65536
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
EOF

# 2. Reload systemd and restart service
echo "🔄 Reloading systemd and restarting service..."
sudo systemctl daemon-reload
sudo systemctl restart deepearth

# 3. Check service status
echo "✅ Checking service status..."
sudo systemctl status deepearth --no-pager

# 4. Monitor memory usage
echo "📊 Current memory usage:"
free -h

echo "🎯 Performance optimizations applied!"
echo "Key changes:"
echo "  - Timeout increased: 120s → 300s"
echo "  - Removed --preload flag to reduce memory usage"
echo "  - Added worker temp directory in RAM"
echo "  - Added max requests to prevent memory leaks"