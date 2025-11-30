#!/usr/bin/env bash
set -euo pipefail

TEST_JSONL=${1:-/path/to/captions_test.jsonl}
IMAGES_TSV=${2:-/path/to/images_test.tsv}
MODEL_DIR=${3:-./outputs/scst}
OUTPUT_JSONL=${4:-./outputs/preds.jsonl}
# 额外参数（与 infer/generate.py 对齐，可按需传入第5-8个位置参数）
BEAM_SIZE=${5:-4}
MAX_GEN_LEN=${6:-64}
PICS_FOR_ONE_PRED_ROUND=${7:-10000}
NUM_PRED_ROUNDS=${8:-1}
# 更多可选参数（可通过环境变量或位置参数覆盖）
DEVICE=${9:-cuda}
IMAGE_SIZE=${10:-448}
TORCH_DTYPE=${11:-bf16}
DEVICE_MAP=${12:-auto}



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

python -u $(dirname $0)/../infer/generate.py \
  --test_jsonl "$TEST_JSONL" \
  --images_tsv "$IMAGES_TSV" \
  --model_dir "$MODEL_DIR" \
  --output_jsonl "$OUTPUT_JSONL" \
  --beam_size "$BEAM_SIZE" \
  --max_gen_len "$MAX_GEN_LEN" \
  --pics_for_one_pred_round "$PICS_FOR_ONE_PRED_ROUND" \
  --num_pred_rounds "$NUM_PRED_ROUNDS" \
  --device "$DEVICE"  \
  --image_size "$IMAGE_SIZE" \
  --torch_dtype "$TORCH_DTYPE" \
  --device_map "$DEVICE_MAP" 
