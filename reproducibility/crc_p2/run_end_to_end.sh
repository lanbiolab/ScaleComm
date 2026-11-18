#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 || $# > 4 )); then
  echo "Usage: $0 RAW_GEO_DIRECTORY NEW_OUTPUT_DIRECTORY [cuda|cpu] [--paper]" >&2
  exit 2
fi
raw_dir=$(cd -- "$1" && pwd)
output=$2
device=cuda
epochs=1000
train_seed=17
checks=(--check-cohort)
for arg in "${@:3}"; do
  case "$arg" in
    cpu|cuda) device=$arg ;;
    --paper) epochs=200; train_seed=20260624; checks=(--check-reference) ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done
mkdir -p -- "$(dirname -- "$output")"
mkdir -- "$output"
output=$(cd -- "$output" && pwd)
here=$(cd -- "$(dirname -- "$0")" && pwd)
dataset=visium_hd_crc_p2crc_rich_targets_subset
prepared="$output/prepared/$dataset"
results="$output/results"

python "$here/prepare.py" \
  --raw-dir "$raw_dir" --output-dir "$prepared" --dataset-id "$dataset" \
  --target-profile rich --n-spots 6000 --max-neighbors 28 \
  --max-candidates-per-lr 120000 --seed 33

python "$here/../tonsil_a1/run_scalecomm.py" "$prepared" \
  --output-dir "$results" --model-name spatialself_wavelet_scale \
  --k-nn 12 --diffusion-steps 1 2 4 8 --h-dim 64 --lr-dim 24 \
  --epochs "$epochs" --learning-rate 0.001 --bce-loss-weight 1.0 \
  --positive-quantile 0.9 --negative-quantile 0.2 \
  --pseudo-label-strategy mechanism_niche_expression_distance \
  --weak-positive-loss-weight 1.0 --train-sampling-strategy balanced_per_lr \
  --hard-negative-loss-weight 0.05 --context-contrastive-loss-weight 0.0 \
  --scale-target-loss-weight 1.1 --scale-diversity-loss-weight 0.45 \
  --scale-responsibility-loss-weight 0.35 --scale-responsibility-margin 0.35 \
  --scale-responsibility-edges 2048 --lr-scale-target-strategy ranklocal \
  --mechanism-guard-loss-weight 0.03 --lr-cap-positive 50 --lr-cap-negative 200 \
  --max-train-samples 12000 --seed "$train_seed" --device "$device"

python "$here/compare_receivers.py" \
  "$prepared" "$results/${dataset}__spatialself_wavelet_scale_predictions.csv" \
  "$here/../artifacts/real_data_threshold_free/source_data/w12_receiver_scores.csv" \
  "$results" "${checks[@]}"
