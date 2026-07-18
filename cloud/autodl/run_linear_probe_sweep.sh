#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/PatchTST-pretrain}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/shd-af-clean-data}"
OUT_ROOT="${OUT_ROOT:-/root/autodl-tmp/outputs/linear_probe}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-1e-3}"
NUM_WORKERS="${NUM_WORKERS:-2}"

cd "$PROJECT_DIR"
source .venv/bin/activate

run_probe() {
  local name="$1"
  local config="$2"
  local pretrained="$3"
  local out_dir="$OUT_ROOT/$name"
  local log_file="$out_dir/train.log"

  mkdir -p "$out_dir"
  echo "[$(date '+%F %T')] start $name" | tee -a "$OUT_ROOT/sweep.log"

  python linear_probe.py \
    --config "$config" \
    --pretrained "$pretrained" \
    --data-root "$DATA_ROOT" \
    --train-csv "$PROJECT_DIR/splits/train.csv" \
    --val-csv "$PROJECT_DIR/splits/val.csv" \
    --output-dir "$out_dir" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --num-workers "$NUM_WORKERS" \
    --label-policy af_normal_only \
    --device cuda \
    2>&1 | tee "$log_file"

  echo "[$(date '+%F %T')] done $name" | tee -a "$OUT_ROOT/sweep.log"
}

mkdir -p "$OUT_ROOT"

run_probe \
  "patch25_random_ratio0p2" \
  "$PROJECT_DIR/configs/_tmp_mask_ratio_sweep_wfdb_fp32_patch25_bs128_random_20260718_213618_ratio0p2.yaml" \
  "/root/autodl-tmp/outputs/mask_ratio_sweep/wfdb_fp32_patch25_bs128_random_20260718_213618/ratio0p2/pretrained_encoder.pt"

run_probe \
  "patch25_random_ratio0p6" \
  "$PROJECT_DIR/configs/_tmp_mask_ratio_sweep_wfdb_fp32_patch25_bs128_random_20260718_213618_ratio0p6.yaml" \
  "/root/autodl-tmp/outputs/mask_ratio_sweep/wfdb_fp32_patch25_bs128_random_20260718_213618/ratio0p6/pretrained_encoder.pt"

run_probe \
  "patch50_random_ratio0p4" \
  "/root/autodl-tmp/outputs/patch_sweep/wfdb_fp32_bs128_20260718_210821/patch50/resolved_config.yaml" \
  "/root/autodl-tmp/outputs/patch_sweep/wfdb_fp32_bs128_20260718_210821/patch50/pretrained_encoder.pt"
