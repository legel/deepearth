#!/usr/bin/env bash
set -euo pipefail

mkdir -p /root/.cache/huggingface

if [[ -n "${USGS_M2M_TOKEN_VALUE:-}" ]]; then
  umask 077
  printf '%s' "$USGS_M2M_TOKEN_VALUE" > /root/.usgs_m2m_token
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  umask 077
  printf '%s' "$HF_TOKEN" > /root/.cache/huggingface/token
elif [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  umask 077
  printf '%s' "$HUGGING_FACE_HUB_TOKEN" > /root/.cache/huggingface/token
fi

echo "usgs_token_present=$(test -f /root/.usgs_m2m_token && echo true || echo false)"
echo "hf_token_present=$(test -f /root/.cache/huggingface/token && echo true || echo false)"
