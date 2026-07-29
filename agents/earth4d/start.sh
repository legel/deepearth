#!/bin/bash
# Clean (re)start of exactly two Earth4D loop drivers on one GPU. Idempotent. In-repo/durable.
# Restores the data symlink (lost on container reboot) so the probe cache resolves, then launches spatial+temporal.
# Usage: bash agents/earth4d/start.sh [gpu-index]   (default GPU 1)
GPU=${1:-1}
HERE="$(cd "$(dirname "$0")" && pwd)"
pkill -9 -f "earth4d.*loop.sh" 2>/dev/null; pkill -9 -f "spacetime.probe" 2>/dev/null; pkill -9 -f "spacetime.calib_probe" 2>/dev/null
sleep 4
[ -e /workspace/data ] || ln -s /workspace/deepearth/data /workspace/data   # reboot drops this symlink
cd /workspace; export PYTHONPATH=/workspace
nohup setsid bash "$HERE/loop.sh" "$GPU" spatial  </dev/null >/tmp/loop0.log 2>&1 &
nohup setsid bash "$HERE/loop.sh" "$GPU" temporal </dev/null >/tmp/loop1.log 2>&1 &
sleep 2
echo "started driver procs: $(pgrep -f 'earth4d.*loop.sh' | grep -v pgrep | wc -l) on GPU$GPU"
