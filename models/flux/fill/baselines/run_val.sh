#!/bin/bash
# Validation-site development run of the FluxFill-Bench baselines (validation sites only; never test sites).
# GPU part: one process at a time on GPU 0, each capped at FLUXFILL_GPU_GB. Steps run in order; a failed step is
# logged and the next one runs. Usage: run_val.sh gpu | cpu | export
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/home/photon/4tb/envs/fluxfill/bin/python
OUT=/home/photon/4tb/fluxfill/bench/val/parts
WIN=/home/photon/4tb/fluxfill/bench/val/windows_val_list.json
PP=/home/photon/4tb/fluxfill/baselines/pypots
mkdir -p "$OUT" "$PP"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HOME=/home/photon/4tb/hf
step() { echo "== $(date '+%F %T') $*"; "$@" || echo "== FAILED: $*"; }
bench() { step $PY -u "$HERE/../bench/run.py" --windows "$WIN" --families M2 --methods "$1" --out "$OUT/$1.parquet"; }
case "${1:-gpu}" in
  gpu)
    export CUDA_VISIBLE_DEVICES=0 FLUXFILL_GPU_GB=${FLUXFILL_GPU_GB:-1.6} FLUXFILL_DE_DEVICE=cuda
    [ -f "$PP/saits_history.json" ] || step $PY -u "$HERE/pypots_imputers.py" train saits --out "$PP" --steps 6000
    step $PY -u "$HERE/pypots_imputers.py" spread --ckpt "$PP/saits.pt"
    bench vekuri_de
    bench saits
    [ -f "$PP/csdi_history.json" ] || step $PY -u "$HERE/pypots_imputers.py" train csdi --out "$PP" --steps 8000
    bench chronos2
    bench csdi
    ;;
  export)
    # Dashboard Windows tab: every method on the two existing dev windows (training sites) and one validation window.
    export CUDA_VISIBLE_DEVICES=0 FLUXFILL_GPU_GB=${FLUXFILL_GPU_GB:-1.6} FLUXFILL_DE_DEVICE=cuda
    D=/home/photon/4tb/fluxfill/bench/dashboard/windows
    T=/home/photon/4tb/fluxfill/towers/hh
    M="climatology mds irvin_rf vekuri_de liu_xgb saits csdi chronos2"
    for w in "CA-TP4 2022-04-01" "US-Bi1 2022-04-01" "US-Ha1 2021-08-31"; do
      set -- $w
      ll=$($PY -c "import pandas as p; r=p.read_csv('/home/photon/4tb/fluxfill/towers/sites.csv').set_index('site_id').loc['$1']; print(r.lat, r.lon)")
      note="Source: FluxFill tower dataset (FLUXNET ONEFlux v1.3, AmeriFlux BASE, FLUXNET-CH4, ICOS; measured values only, quality rules applied; frozen split). Methods: climatology; MDS (REddyProc 1.3.2); RF ensemble (Irvin et al. 2021); deep ensemble (Vekuri et al. 2025); XGBoost (Liu et al. 2025); SAITS with a residual-based Gaussian spread; CSDI; Chronos-2 zero-shot. Details: models/flux/fill/baselines/README.md."
      sp=$(awk -F, -v s="$1" '$1==s{print $2}' /home/photon/4tb/fluxfill/towers/split.csv)
      if [ "$sp" = "val" ]; then note="$note $1 is a validation site; these are validation-site development numbers."
      else note="$note $1 is a $sp site of the trained methods (SAITS, CSDI), so their curves here are in-sample."; fi
      step $PY -u "$HERE/../bench/export_dashboard.py" "$T/$1.parquet" "$1" "$2" "$D" --methods $M --lat ${ll% *} --lon ${ll#* } --note "$note"
    done
    ;;
  publish)
    # Combine the parts into baselines_val.parquet, write the Results tables, rebuild and deploy the dashboard.
    step $PY -c "import glob, pandas as pd; d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('$OUT/*.parquet'))], ignore_index=True); d['split'] = 'validation sites (development)'; d.to_parquet('/home/photon/4tb/fluxfill/bench/baselines_val.parquet', index=False); print(d.groupby('method').size())"
    step $PY "$HERE/../bench/results_dashboard.py" /home/photon/4tb/fluxfill/bench/baselines_val.parquet \
      --out /home/photon/4tb/fluxfill/bench/dashboard/results.json --keep-other \
      --title "Validation-site development run" \
      --note "Validation-site development numbers (provisional split; not test-site benchmark results). M2 masks, all tower channels hidden in one block per window."
    step "$HERE/../dashboard/deploy.sh"
    ;;
  cpu)
    export CUDA_VISIBLE_DEVICES=""
    bench liu_xgb
    bench climatology
    bench mds
    ;;
esac
echo "== $(date '+%F %T') done"
