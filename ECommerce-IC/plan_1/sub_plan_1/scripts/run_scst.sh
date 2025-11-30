#!/usr/bin/env bash
set -euo pipefail

TRAIN_JSONL=${1:-/path/to/captions_train.jsonl}
IMAGES_TSV=${2:-/path/to/images_train.tsv}
SFT_MODEL=${3:-./outputs/sft}
OUTPUT_DIR=${4:-./outputs/scst}
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

python -u $(dirname $0)/../train/scst_train_cider.py \
  --train_jsonl "$TRAIN_JSONL" \
  --images_tsv "$IMAGES_TSV" \
  --sft_model "$SFT_MODEL" \
  --output_dir "$OUTPUT_DIR" \
  --config $(dirname $0)/../configs/scst.json
