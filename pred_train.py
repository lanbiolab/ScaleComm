# -*- coding: utf-8 -*-
"""
predict_visium.py
---------------------------------
在真实 Visium 图上做全图 LR 推理：

输入：
  visium_lr_benchmark/
    - graph_visium.pt      （构图脚本生成）
    - lr_table.csv         （可用的 LR 对列表，含 ligand / receptor / lr_id）
    - lr_model_visium.pt   （train_visium_unsupervised.py 训练得到的最优模型）

输出：
  visium_lr_benchmark/pred_all_visium.csv
    每一行是一个 (src, dst, LR) 的预测：
      src, dst        : 节点索引
      lr_id           : LR 对编号（与 lr_table 对应）
      ligand, receptor: 基因 symbol
      dist            : 空间距离（来自 graph.edge_attr[:,0]）
      logit           : 模型输出的 raw logit
      prob            : sigmoid(logit)
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch_geometric.data import Data  # 只是为了类型提示 / IDE
from src.spitialself.models import LRModelVisium as SharedLRModelVisium

# ================== 路径与配置 ==================

ROOT = "visium_lr_benchmark"

GRAPH_PATH = os.path.join(ROOT, "graph_visium.pt")
LR_TABLE_PATH = os.path.join(ROOT, "lr_table.csv")
MODEL_PATH = os.path.join(ROOT, "lr_model_visium.pt")

OUT_CSV = os.path.join(ROOT, "pred_all_visium.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 需要和训练脚本保持一致
H_DIM = 64
LR_DIM = 16
GAT_HEADS = 4
GAT_DROPOUT = 0.2

BATCH_EDGES = 10000   # 以边为单位分批跑，防止一次吞太多


# ================== 推理主逻辑 ==================

@torch.no_grad()
def main():
    print(f"🧠 Using device: {DEVICE}")

    # ---------- 1. 加载图 ----------
    print(f"📥 Loading graph from: {GRAPH_PATH}")
    data: Data = torch.load(GRAPH_PATH, map_location=DEVICE, weights_only=False)
    print(f"  nodes = {data.num_nodes}, edges = {data.num_edges}")
    print(f"  x shape = {tuple(data.x.shape)}, edge_attr shape = {tuple(data.edge_attr.shape)}")

    # edge_index: [2, E] → [E, 2]
    edge_index_np = data.edge_index.cpu().numpy().T
    src_all = edge_index_np[:, 0].astype(np.int64)
    dst_all = edge_index_np[:, 1].astype(np.int64)

    # 距离在 edge_attr[:,0]
    dist_all = data.edge_attr[:, 0].cpu().numpy().astype(np.float32)
    n_edges = len(src_all)
    print(f"  usable edges = {n_edges}")

    # ---------- 2. 加载 LR 表 ----------
    lr_table = pd.read_csv(LR_TABLE_PATH)
    required_cols = {"lr_id", "ligand", "receptor"}
    if not required_cols.issubset(lr_table.columns):
        raise ValueError(f"lr_table.csv 必须包含列: {required_cols}，当前列：{list(lr_table.columns)}")

    lr_table = lr_table.sort_values("lr_id").reset_index(drop=True)
    n_lr = lr_table["lr_id"].max() + 1
    print(f"📑 Loaded LR table: n_lr = {n_lr}")
    print(lr_table.head())

    # ---------- 3. 构建并加载模型 ----------
    in_dim = data.x.size(1)
    model = SharedLRModelVisium(
        in_dim=in_dim,
        h_dim=H_DIM,
        lr_dim=LR_DIM,
        n_lr=n_lr,
        heads=GAT_HEADS,
        dropout=GAT_DROPOUT,
    ).to(DEVICE)

    print(f"📦 Loading model weights from: {MODEL_PATH}")
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    print("  ✅ Model loaded.")

    # ---------- 4. 全图节点编码 ----------
    print("🧬 Encoding node embeddings (GATv2 on full graph)...")
    h = model.encode_nodes(data).to(DEVICE)  # [N, H_DIM]
    print("  ✅ Node embeddings ready.")

    # 把边信息放到 GPU，减少来回拷贝
    src_tensor_all = torch.from_numpy(src_all).long().to(DEVICE)
    dst_tensor_all = torch.from_numpy(dst_all).long().to(DEVICE)
    dist_tensor_all = torch.from_numpy(dist_all).float().unsqueeze(-1).to(DEVICE)

    # ---------- 5. 遍历 LR，分批推理 ----------
    if os.path.exists(OUT_CSV):
        os.remove(OUT_CSV)

    print(f"📝 Will write predictions to: {OUT_CSV}")
    first_chunk = True
    total_rows = 0

    for _, lr_row in lr_table.iterrows():
        lr_id = int(lr_row["lr_id"])
        ligand = str(lr_row["ligand"])
        receptor = str(lr_row["receptor"])
        lr_name = f"{ligand}|{receptor}"

        print(f"\n🔍 Predicting for LR #{lr_id}: {lr_name}")

        # 当前 LR 的 lr_ids 向量
        lr_ids_full = torch.full((n_edges,), lr_id, dtype=torch.long, device=DEVICE)

        n_batches = (n_edges + BATCH_EDGES - 1) // BATCH_EDGES
        chunk_rows = []

        for b in range(n_batches):
            start = b * BATCH_EDGES
            end = min((b + 1) * BATCH_EDGES, n_edges)
            if start >= end:
                continue

            src_b = src_tensor_all[start:end]
            dst_b = dst_tensor_all[start:end]
            dist_b = dist_tensor_all[start:end]
            lr_ids_b = lr_ids_full[start:end]

            logits_b = model.score_edges(h, src_b, dst_b, lr_ids_b, dist_b)
            probs_b = torch.sigmoid(logits_b)

            # 回 CPU，转 numpy
            src_np = src_b.cpu().numpy().astype(np.int32)
            dst_np = dst_b.cpu().numpy().astype(np.int32)
            dist_np = dist_b[:, 0].cpu().numpy().astype(np.float32)
            logit_np = logits_b.cpu().numpy().astype(np.float32)
            prob_np = probs_b.cpu().numpy().astype(np.float32)

            df_chunk = pd.DataFrame({
                "src": src_np,
                "dst": dst_np,
                "lr_id": lr_id,
                "ligand": ligand,
                "receptor": receptor,
                "dist": dist_np,
                "logit": logit_np,
                "prob": prob_np,
            })
            chunk_rows.append(df_chunk)

        if not chunk_rows:
            continue

        df_lr = pd.concat(chunk_rows, ignore_index=True)
        total_rows += len(df_lr)

        df_lr.to_csv(OUT_CSV, index=False, mode="a", header=first_chunk)
        first_chunk = False

        print(f"  ✓ Done LR #{lr_id}: wrote {len(df_lr)} rows, total so far = {total_rows}")

    print("\n🎉 All LR predictions finished.")
    print(f"📄 Final prediction file: {OUT_CSV}")
    print(f"   Total rows ≈ {total_rows}  (≈ n_edges × n_lr)")
    print("   后续可以用来：画 LR 距离曲线、region enrichment、和 CellChat 结果对齐等。")


if __name__ == "__main__":
    with torch.no_grad():
        main()
