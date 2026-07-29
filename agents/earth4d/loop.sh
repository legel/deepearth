#!/bin/bash
# Earth4D agentic loop driver (in-repo, durable). Robust: fast reliable modes only (no GNN/field-decode that
# wedge the GPU), 240s per-probe timeout so any hang self-heals, and --ensue so EVERY run is published to the
# LOOP-earth4d-<capability> taxonomy (win or dead-end, with reason). Both DATA and ARCHITECTURE levers.
# Usage: bash loop.sh <gpu-index> <spatial|temporal>
GPU=$1; TRACK=$2
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=/workspace
cd /workspace
pick(){ local a=("$@"); echo "${a[$((RANDOM % ${#a[@]}))]}"; }
while true; do
  pm=""
  if [ "$TRACK" = spatial ]; then
    ff=$(pick 0 256 512 1024); th=$(pick 0 8 16); hh=$(pick 0 256 512)
    m=$(pick fc fc famenv sdm)
    if [ "$m" = fc ]; then
      metric=family_from_spacetime; flags="--forecast --head_hidden $hh --fourier $ff --time_harmonics $th --n_shards 12"; tag="fc_hh${hh}_ff${ff}_th${th}"
    elif [ "$m" = famenv ]; then
      ec=$(pick all worldclim alphaearth); metric=family_from_env; flags="--env --env_channels $ec --n_shards 12"; tag="famenv_${ec}"
    else
      ch=$(pick all alphaearth worldclim); metric=species_from_env; flags="--sdm_presence --sdm_hard --sdm_channels $ch --n_shards 16"; tag="sdm_${ch}"
    fi
  else
    m=$(pick fc fc cal cooccur pheno)
    if [ "$m" = cal ]; then
      metric=calibration; pm="--probe-module deepearth.autoresearch.programs.spacetime.calib_probe"; flags="--feature earth4d --n_shards 8 --steps 400 --ensemble 3"; tag="cal_earth4d"
    elif [ "$m" = cooccur ]; then
      mech=$(pick env space both); metric=community_from_env; flags="--cooccur --cooccur_mech $mech --n_shards 12"; tag="coocc_${mech}"
    elif [ "$m" = pheno ]; then
      pe=$(pick "" "--pheno_env"); metric=flowering_peak_month; flags="--phenology --forecast --recurrence $pe --pheno_feats e4d --n_shards 12"; tag="pheno_rec${pe:+_env}"
    else
      pp=$(pick "" "--recurrence --rec_hidden 512"); hh=$(pick 256 512); ff=$(pick 512 1024); th=$(pick 0 8)
      metric=family_from_spacetime; flags="--forecast $pp --head_hidden $hh --fourier $ff --time_harmonics $th --n_shards 12"; tag="fcp_hh${hh}_ff${ff}_th${th}"
    fi
  fi
  echo "=== [$(date +%H:%M)] GPU$GPU $tag ($metric) ==="
  timeout 240 /usr/bin/python3.12 -m deepearth.agents.earth4d.trace \
    --metric "$metric" --probe "$flags" $pm --tag "$tag" --device cuda:0 --ensue 2>&1 \
    | grep -E "RECORD =|primary\(|Ensue logged" | head -3
done
