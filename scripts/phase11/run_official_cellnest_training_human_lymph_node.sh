#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CELLNEST_ROOT="/home/xd/y/BenchmarkBaselines/CellNEST"
RUN_ROOT="${PROJECT_ROOT}/outputs/phase11/official_method_reproduction/cellnest_human_lymph_node"
mkdir -p "${RUN_ROOT}/logs"

export PATH="/home/Anaconda3/envs/py39/bin:${PATH}"
export PYTHONNOUSERSITE=0

cd "${CELLNEST_ROOT}"

for run_id in 1 2 3 4 5; do
  gpu=$(( (run_id - 1) % 4 ))
  seed="${run_id}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    bash cellnest run \
      --data_name='V1_Human_Lymph_Node_spatial' \
      --num_epoch 80000 \
      --manual_seed='yes' \
      --seed="${seed}" \
      --model_name='CellNEST_V1_Human_Lymph_Node_spatial' \
      --run_id="${run_id}" \
      > "${RUN_ROOT}/logs/train_run${run_id}.stdout.log" \
      2> "${RUN_ROOT}/logs/train_run${run_id}.stderr.log"
  ) &
done

wait

bash cellnest postprocess \
  --data_name='V1_Human_Lymph_Node_spatial' \
  --model_name='CellNEST_V1_Human_Lymph_Node_spatial' \
  --total_runs=5 \
  > "${RUN_ROOT}/logs/postprocess.stdout.log" \
  2> "${RUN_ROOT}/logs/postprocess.stderr.log"

bash cellnest visualize \
  --data_name='V1_Human_Lymph_Node_spatial' \
  --model_name='CellNEST_V1_Human_Lymph_Node_spatial' \
  --top_edge_count=3000 \
  --annotation_file_path='data/V1_Human_Lymph_Node_spatial_annotation.csv' \
  > "${RUN_ROOT}/logs/visualize_top3000.stdout.log" \
  2> "${RUN_ROOT}/logs/visualize_top3000.stderr.log"
