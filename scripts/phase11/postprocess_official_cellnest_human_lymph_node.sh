#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CELLNEST_ROOT="/home/xd/y/BenchmarkBaselines/CellNEST"
RUN_ROOT="${PROJECT_ROOT}/outputs/phase11/official_method_reproduction/cellnest_human_lymph_node"

export PATH="/home/xd/.conda/envs/Benchmark/bin:${PATH}"
export PYTHONNOUSERSITE=1

cd "${CELLNEST_ROOT}"

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

/home/xd/.conda/envs/Benchmark/bin/python "${PROJECT_ROOT}/scripts/phase11/extract_official_cellnest_ccr7.py" \
  --cellnest-output "${CELLNEST_ROOT}/output/V1_Human_Lymph_Node_spatial/CellNEST_V1_Human_Lymph_Node_spatial_top20percent.csv" \
  --output-dir "${RUN_ROOT}" \
  > "${RUN_ROOT}/logs/extract_ccr7.stdout.log" \
  2> "${RUN_ROOT}/logs/extract_ccr7.stderr.log"
