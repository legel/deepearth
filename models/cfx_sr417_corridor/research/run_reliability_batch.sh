#!/bin/bash
set -e
# Resolve the project root from this script's own location, so the batch runs from anywhere.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=python3
SIM=research/train_grid_transformer_surrogate.py

echo "=== Phase 1: small vol-loss-weight/epoch sweep (single seed=0) ==="
for w in 1 2 10 20; do
  echo "--- weight=$w epochs=20 ---"
  $PY $SIM --epochs 20 --batch-size 8 --vol-loss-weight $w --seed 0 --tag "sweep_w${w}_e20"
done
for e in 10 15 30; do
  echo "--- weight=5 epochs=$e ---"
  $PY $SIM --epochs $e --batch-size 8 --vol-loss-weight 5.0 --seed 0 --tag "sweep_w5_e${e}"
done

echo "=== Phase 2: multi-seed for the 3 original variants (seeds 1,2 — seed 0 already exists) ==="
for s in 1 2; do
  echo "--- baseline seed=$s ---"
  $PY $SIM --epochs 80 --batch-size 8 --seed $s --tag "baseline_s${s}"
  echo "--- reweighted seed=$s ---"
  $PY $SIM --epochs 80 --batch-size 8 --loss-weight-alpha 8.0 --seed $s --tag "reweighted_s${s}"
  echo "--- rollout3 seed=$s ---"
  $PY $SIM --epochs 60 --batch-size 8 --rollout-steps 3 --seed $s --tag "rollout3_s${s}"
done

echo "=== Phase 3: also get a clean seed=0 for baseline/reweighted/rollout3 (originals predate --seed flag) ==="
$PY $SIM --epochs 80 --batch-size 8 --seed 0 --tag "baseline_s0"
$PY $SIM --epochs 80 --batch-size 8 --loss-weight-alpha 8.0 --seed 0 --tag "reweighted_s0"
$PY $SIM --epochs 60 --batch-size 8 --rollout-steps 3 --seed 0 --tag "rollout3_s0"

echo "=== ALL DONE ==="
