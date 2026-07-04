# -*- coding: utf-8 -*-
"""
train_visium_unsupervised.py (GATv2, end-to-end)
----------------------------------
在 Visium 图上，使用自监督样本 S_vis 训练：
  GATv2 NodeEncoder + LREncoder + EdgeScorer

目标：
  对每个 (src,dst,LR)：
    - high 组（表达高 + 距离合适） 的预测分数 > low 组的预测分数
  使用 pairwise margin ranking loss + BCE。

输入：
  - visium_lr_benchmark/graph_visium.pt
  - visium_lr_benchmark/lr_table.csv
  - visium_lr_benchmark/S_vis.npy
      每行: [src, dst, lr_id, dist, score_init, rank_label]

输出：
  - visium_lr_benchmark/lr_model_visium.pt  （训练好的模型参数）
"""

import os
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from sklearn.metrics import average_precision_score, roc_auc_score
from src.spitialself.models import LRModelVisium as SharedLRModelVisium

# ================== 配置 ==================
ROOT = "visium_lr_benchmark"
GRAPH_PATH = os.path.join(ROOT, "graph_visium.pt")
LR_TABLE_PATH = os.path.join(ROOT, "lr_table.csv")
S_PATH = os.path.join(ROOT, "S_vis.npy")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 4096
N_EPOCHS = 20
LR_INIT = 1e-3

H_DIM = 64      # 节点表示维度
LR_DIM = 16     # LR embedding 维度

MARGIN = 0.1    # ranking margin
VAL_FRAC = 0.2  # S_vis 中留一部分做验证

GAT_HEADS = 4
GAT_DROPOUT = 0.2

RANDOM_SEED = 2025
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ================== 数据加载 ==================

def load_data():
    data = torch.load(GRAPH_PATH, map_location=DEVICE, weights_only=False)
    lr_table = pd.read_csv(LR_TABLE_PATH)
    S_all = np.load(S_PATH).astype(np.float32)  # [src, dst, lr_id, dist, score_init, rank_label]

    return data, lr_table, S_all

def split_train_val(S_all, val_frac=VAL_FRAC):
    n = len(S_all)
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_val = int(n * val_frac)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return S_all[train_idx], S_all[val_idx]

def iter_minibatches(S_array, batch_size):
    n = len(S_array)
    idx = np.arange(n)
    np.random.shuffle(idx)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield S_array[idx[start:end]]

# ================== 损失 & 评估 ==================

def ranking_loss_from_batch(logits, labels, margin=MARGIN):
    """
    純 Torch 版本，不使用 numpy。
    labels: [B] float tensor (0/1)
    logits: [B] float tensor
    """
    # 转成 boolean mask
    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_idx = pos_mask.nonzero(as_tuple=True)[0]
    neg_idx = neg_mask.nonzero(as_tuple=True)[0]

    # 不足以形成 pair
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return torch.tensor(0.0, device=logits.device, dtype=torch.float32)

    # pair 数量 = 两组最小数
    m = min(len(pos_idx), len(neg_idx))

    # 从 torch tensor 中抽样，仍然使用 torch.randperm
    pos_sel = pos_idx[torch.randperm(len(pos_idx), device=logits.device)[:m]]
    neg_sel = neg_idx[torch.randperm(len(neg_idx), device=logits.device)[:m]]

    pos_scores = logits[pos_sel]
    neg_scores = logits[neg_sel]

    # margin ranking：希望 pos_score - neg_score > margin
    diff = pos_scores - neg_scores
    loss = F.relu(margin - diff).mean()

    return loss


@torch.no_grad()
def eval_on_split(model, data, S_split, name="val"):
    """
    虽然 rank_label 是伪标签，但 AUC / AUPR 仍然可以用来监控训练。
    """
    model.eval()
    h = model.encode_nodes(data)

    src = torch.tensor(S_split[:, 0], dtype=torch.long, device=DEVICE)
    dst = torch.tensor(S_split[:, 1], dtype=torch.long, device=DEVICE)
    lr_ids = torch.tensor(S_split[:, 2], dtype=torch.long, device=DEVICE)
    dists = torch.tensor(S_split[:, 3:4], dtype=torch.float32, device=DEVICE)
    labels = S_split[:, 5].astype(int)

    logits_list = []
    B = 4096
    for start in range(0, len(S_split), B):
        end = min(start + B, len(S_split))
        logit = model.score_edges(
            h,
            src[start:end],
            dst[start:end],
            lr_ids[start:end],
            dists[start:end]
        )
        logits_list.append(logit.detach().cpu().numpy())
    logits = np.concatenate(logits_list, axis=0)
    probs = 1.0 / (1.0 + np.exp(-logits))

    try:
        aupr = average_precision_score(labels, probs)
    except ValueError:
        aupr = np.nan
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = np.nan

    print(f"[{name}] n={len(S_split)}, AUPR={aupr:.4f}, AUC={auc:.4f}")
    return dict(aupr=aupr, auc=auc)

# ================== 训练主流程 ==================

def train_one_epoch(model, data, S_train, optimizer):
    """
    ⚠️ 这里是关键修复点：
    不再用 no_grad 冻结 NodeEncoder，
    而是每个 batch 都重新 encode 整图，让梯度能传回 GATv2。
    Graph 只有 4k 节点，这样做是完全可承受的。
    """
    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in iter_minibatches(S_train, BATCH_SIZE):
        # 1) 对整图做一次 GATv2 编码（允许梯度）
        h = model.encode_nodes(data)

        # 2) 从 S_train 里取出一小批边
        src = torch.tensor(batch[:, 0], dtype=torch.long, device=DEVICE)
        dst = torch.tensor(batch[:, 1], dtype=torch.long, device=DEVICE)
        lr_ids = torch.tensor(batch[:, 2], dtype=torch.long, device=DEVICE)
        dists = torch.tensor(batch[:, 3:4], dtype=torch.float32, device=DEVICE)
        labels = torch.tensor(batch[:, 5], dtype=torch.float32, device=DEVICE)

        # 3) 边级打分
        logits = model.score_edges(h, src, dst, lr_ids, dists)

        # 4) ranking + BCE 组合损失
        loss_rank = ranking_loss_from_batch(logits, labels)
        loss_bce = F.binary_cross_entropy_with_logits(logits, labels)
        loss = loss_rank + 0.5 * loss_bce

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch)
        total_samples += len(batch)

    avg_loss = total_loss / total_samples
    return avg_loss

def main():
    print(f"Using device: {DEVICE}")
    data, lr_table, S_all = load_data()
    data = data.to(DEVICE)

    n_lr = lr_table["lr_id"].max() + 1
    in_dim = data.x.size(1)

    print(f"Graph: nodes={data.num_nodes}, edges={data.num_edges}")
    print(f"LR pairs used: n_lr={n_lr}")
    print(f"S_all shape: {S_all.shape}")

    S_train, S_val = split_train_val(S_all, val_frac=VAL_FRAC)
    print(f"S_train={len(S_train)}, S_val={len(S_val)}")

    model = SharedLRModelVisium(
        in_dim=in_dim,
        h_dim=H_DIM,
        lr_dim=LR_DIM,
        n_lr=n_lr,
        heads=GAT_HEADS,
        dropout=GAT_DROPOUT,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT, weight_decay=1e-4)

    best_val_aupr = -1.0
    best_state = None

    for epoch in range(1, N_EPOCHS + 1):
        avg_loss = train_one_epoch(model, data, S_train, optimizer)
        print(f"Epoch {epoch:03d} | train_loss={avg_loss:.4f}")

        val_metrics = eval_on_split(model, data, S_val, name="val")
        if val_metrics["aupr"] > best_val_aupr:
            best_val_aupr = val_metrics["aupr"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            print(f"  🔥 New best val AUPR: {best_val_aupr:.4f}")

    if best_state is not None:
        save_path = os.path.join(ROOT, "lr_model_visium.pt")
        torch.save(best_state, save_path)
        print(f"✅ Saved best model to {save_path}")

        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        _ = eval_on_split(model, data, S_all, name="all_S")

    print("\n🎉 Training on Visium (self-supervised, GATv2, end-to-end) finished.")

if __name__ == "__main__":
    main()
