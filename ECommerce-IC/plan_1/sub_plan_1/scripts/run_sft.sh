#!/usr/bin/env bash
set -euo pipefail

TRAIN_JSONL=${1:-/path/to/captions_train.jsonl}
IMAGES_TSV=${2:-/path/to/images_train.tsv}
OUTPUT_DIR=${3:-./outputs/sft}
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=0
export HF_HUB_DISABLE_TELEMETRY=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCH_HOME=/mnt/d/HuggingFaceModels/
export HF_ENDPOINT=https://hf-mirror.com
export https_proxy=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
export all_proxy=socks5://127.0.0.1:7890
export WANDB_DISABLED=true
export CURL_CA_BUNDLE=

python -u $(dirname $0)/../train/sft_train.py \
  --train_jsonl "$TRAIN_JSONL" \
  --images_tsv "$IMAGES_TSV" \
  --output_dir "$OUTPUT_DIR" \
  --config $(dirname $0)/../configs/sft.json
