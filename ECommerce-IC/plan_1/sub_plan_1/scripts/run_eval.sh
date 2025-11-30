#!/usr/bin/env bash
set -euo pipefail

REFS_JSONL=${1:-/path/to/captions_ref.jsonl}
PREDS_JSONL=${2:-./outputs/preds.jsonl}
# 额外参数（与 eval/evaluate_cider.py 对齐，可按需传入第3-5个位置参数）
IMAGES_TSV=${3:-/path/to/images_test.tsv}
PICS_FOR_ONE_EVAL_ROUND=${4:-0}
NUM_EVAL_ROUNDS=${5:-1}
export TOKENIZERS_PARALLELISM=false
export TORCH_HOME=/mnt/d/HuggingFaceModels/
export HF_ENDPOINT=https://hf-mirror.com
export https_proxy=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
export all_proxy=socks5://127.0.0.1:7890
export WANDB_DISABLED=true
export CURL_CA_BUNDLE=

python -u $(dirname $0)/../eval/evaluate_cider.py \
  --refs_jsonl "$REFS_JSONL" \
  --preds_jsonl "$PREDS_JSONL" \
  --images_tsv "$IMAGES_TSV" \
  --pics_for_one_eval_round "$PICS_FOR_ONE_EVAL_ROUND" \
  --num_eval_rounds "$NUM_EVAL_ROUNDS"
