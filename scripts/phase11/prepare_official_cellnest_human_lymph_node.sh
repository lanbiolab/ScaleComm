#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CELLNEST_ROOT="/home/xd/y/BenchmarkBaselines/CellNEST"
RUN_ROOT="${PROJECT_ROOT}/outputs/phase11/official_method_reproduction/cellnest_human_lymph_node"
DATA_DIR="${CELLNEST_ROOT}/data/V1_Human_Lymph_Node_spatial"
RAW_DIR="${PROJECT_ROOT}/data/optional/human_lymph_node_visium_full/raw"

mkdir -p "${RUN_ROOT}/logs" "${DATA_DIR}"
ln -sf "${RAW_DIR}/V1_Human_Lymph_Node_filtered_feature_bc_matrix.h5" "${DATA_DIR}/filtered_feature_bc_matrix.h5"
ln -sfn "${RAW_DIR}/spatial" "${DATA_DIR}/spatial"

{
  echo "cd ${CELLNEST_ROOT}"
  echo "bash cellnest preprocess --data_name='V1_Human_Lymph_Node_spatial' --data_from='data/V1_Human_Lymph_Node_spatial/' --filter_min_cell=1 --data_type=visium"
  echo "bash cellnest run --data_name='V1_Human_Lymph_Node_spatial' --num_epoch 80000 --manual_seed='yes' --seed=1 --model_name='CellNEST_V1_Human_Lymph_Node_spatial' --run_id=1"
  echo "bash cellnest run --data_name='V1_Human_Lymph_Node_spatial' --num_epoch 80000 --manual_seed='yes' --seed=2 --model_name='CellNEST_V1_Human_Lymph_Node_spatial' --run_id=2"
  echo "bash cellnest run --data_name='V1_Human_Lymph_Node_spatial' --num_epoch 80000 --manual_seed='yes' --seed=3 --model_name='CellNEST_V1_Human_Lymph_Node_spatial' --run_id=3"
  echo "bash cellnest run --data_name='V1_Human_Lymph_Node_spatial' --num_epoch 80000 --manual_seed='yes' --seed=4 --model_name='CellNEST_V1_Human_Lymph_Node_spatial' --run_id=4"
  echo "bash cellnest run --data_name='V1_Human_Lymph_Node_spatial' --num_epoch 80000 --manual_seed='yes' --seed=5 --model_name='CellNEST_V1_Human_Lymph_Node_spatial' --run_id=5"
  echo "bash cellnest postprocess --data_name='V1_Human_Lymph_Node_spatial' --model_name='CellNEST_V1_Human_Lymph_Node_spatial' --total_runs=5"
  echo "bash cellnest visualize --data_name='V1_Human_Lymph_Node_spatial' --model_name='CellNEST_V1_Human_Lymph_Node_spatial' --top_edge_count=3000 --annotation_file_path='data/V1_Human_Lymph_Node_spatial_annotation.csv'"
} > "${RUN_ROOT}/official_cellnest_commands.sh"

cd "${CELLNEST_ROOT}"
PYTHONNOUSERSITE=1 /home/xd/.conda/envs/Benchmark/bin/python data_preprocess_CellNEST.py \
  --data_name V1_Human_Lymph_Node_spatial \
  --data_from data/V1_Human_Lymph_Node_spatial/ \
  --filter_min_cell 1 \
  --data_type visium \
  > "${RUN_ROOT}/logs/preprocess.stdout.log" \
  2> "${RUN_ROOT}/logs/preprocess.stderr.log"

find input_graph/V1_Human_Lymph_Node_spatial metadata/V1_Human_Lymph_Node_spatial -maxdepth 1 -type f -printf '%p\t%k KB\n' \
  > "${RUN_ROOT}/preprocess_artifacts.tsv"
