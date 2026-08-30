#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 /path/to/tonsil_A1_rna_adt.h5ad [cuda|cpu] [--ui]" >&2
  exit 2
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
INPUT_H5AD=$1
DEVICE=cuda
UI_ARGS=()
for arg in "${@:2}"; do
  case "$arg" in
    cuda|cpu) DEVICE="$arg" ;;
    --ui) UI_ARGS+=(--ui) ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done
WORKFLOW="$ROOT/reproducibility/tonsil_a1"
PREPARED="$WORKFLOW/work/prepared/spamosaic_tonsil_A1_rich_targets_subset"
RESULTS="$WORKFLOW/work/results/a1_rich_scalecomm_e200"

python "$WORKFLOW/prepare_tonsil_a1.py" \
  --input-h5ad "$INPUT_H5AD" \
  --output-dir "reproducibility/tonsil_a1/work/prepared/spamosaic_tonsil_A1_rich_targets_subset" \
  --dataset-id spamosaic_tonsil_A1_rich_targets_subset \
  --n-cells 2500 \
  --seed 17 \
  --max-neighbors 30 \
  --max-candidates-per-lr 90000 \
  --target-profile rich

python "$WORKFLOW/run_scalecomm.py" "$PREPARED" \
  --output-dir "$RESULTS" \
  --model-name spatialself_wavelet_scale \
  --k-nn 12 \
  --diffusion-steps 1 2 4 8 \
  --h-dim 64 \
  --lr-dim 24 \
  --epochs 200 \
  --learning-rate 0.001 \
  --positive-quantile 0.90 \
  --negative-quantile 0.20 \
  --pseudo-label-strategy mechanism_niche_expression_distance \
  --train-sampling-strategy balanced_per_lr \
  --hard-negative-loss-weight 0.05 \
  --scale-target-loss-weight 1.10 \
  --scale-diversity-loss-weight 0.45 \
  --scale-responsibility-loss-weight 0.35 \
  --scale-responsibility-margin 0.35 \
  --scale-responsibility-edges 2048 \
  --lr-scale-target-strategy ranklocal \
  --mechanism-guard-loss-weight 0.03 \
  --max-train-samples 12000 \
  --seed 7 \
  --device "$DEVICE" \
  "${UI_ARGS[@]}"

PREDICTION=$(find "$RESULTS" -maxdepth 1 -type f -name '*__spatialself_wavelet_scale_predictions.csv' -print -quit)
if [[ -z "$PREDICTION" ]]; then
  echo "ScaleComm completed without a prediction CSV in $RESULTS" >&2
  exit 1
fi

python "$WORKFLOW/validate_tonsil_a1.py" \
  --prepared-dir "$PREPARED" \
  --prediction "$PREDICTION" \
  --output-dir "$RESULTS"
