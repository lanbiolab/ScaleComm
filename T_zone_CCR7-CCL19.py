# -*- coding: utf-8 -*-
"""
analyze_CCR7_CCL19_visium.py
----------------------------------
基于你“老版本”的区域划分逻辑（cell2location + 手写规则），
再叠加我们现在这个 GNN 模型对 CCR7–CCL19 的预测结果，看看是否在 T cell zone 富集。

输入：
  - ../data/Human_Lymph_Node.h5ad
  - visium_lr_benchmark/pred_all_visium.csv   （推理脚本输出）

输出：
  - visium_lr_benchmark/spot_region_from_c2l.csv
  - visium_lr_benchmark/CCR7_CCL19_node_scores_with_region.csv
  - 终端打印各 region 的 CCR7–CCL19 平均 prob 和描述统计
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc


# ================== 1. 区域划分（完全复刻你旧脚本的逻辑） ==================

def build_region_from_c2l(h5ad_path):
    """
    用你之前的方式：
      - adata.obsm["means_cell_abundance_w_sf"]
      - 每行做归一化
      - assign_region(row)

    返回：
      coords: DataFrame(index=spot_barcode, columns=[x, y, region])
      regions: Series(index=spot_barcode, name="region")
    """
    print(f"📥 读取 h5ad: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)

    # 坐标
    coords = pd.DataFrame(
        adata.obsm["spatial"],
        columns=["x", "y"],
        index=adata.obs_names
    )

    # cell2location abundance
    cell_ab = pd.DataFrame(
        adata.obsm["means_cell_abundance_w_sf"],
        index=adata.obs_names
    )

    # 行归一化（你之前就是这么做的）
    cell_ab_norm = cell_ab.div(cell_ab.sum(axis=1), axis=0).fillna(0.0)

    # 和你原来一模一样的区域划分规则
    def assign_region(row):
        names = " ".join(row.index[row > 0.05])
        if row.filter(like="B_GC").sum() > 0.2 or "B_GC" in names:
            return "Germinal center"
        elif row.filter(like="B_").sum() > 0.35:
            return "B follicle"
        elif row.filter(like="T_CD4").sum() > 0.35 or row.filter(like="T_TIM3").sum() > 0.2:
            return "T cell zone"
        else:
            return "Mixed"

    coords["region"] = cell_ab_norm.apply(assign_region, axis=1)
    regions = coords["region"].copy()
    regions.name = "region"

    print("\n✅ 区域标签分布（沿用旧逻辑）：")
    print(regions.value_counts())

    # 保存一份 region 注释，方便别的脚本用
    out_region = "visium_lr_benchmark/spot_region_from_c2l.csv"
    os.makedirs(os.path.dirname(out_region), exist_ok=True)
    coords_out = coords[["x", "y", "region"]].copy()
    coords_out.insert(0, "spot", coords_out.index)
    coords_out.to_csv(out_region, index=False)
    print(f"✅ 已保存 region 注释到: {out_region}")

    return adata, coords, regions


# ================== 2. 从预测中抽取 CCR7–CCL19 ==================

def load_predictions_for_lr(pred_path, ligand_name="CCL19", receptor_name="CCR7"):
    """
    从 pred_all_visium.csv 中抽取 CCR7–CCL19 这对 LR 的边
      - (ligand, receptor) = (CCL19, CCR7)
      - 或 (CCR7, CCL19)
    """
    print(f"\n📑 读取预测结果: {pred_path}")
    pred = pd.read_csv(pred_path)

    required_cols = {"src", "dst", "ligand", "receptor", "prob"}
    if not required_cols.issubset(pred.columns):
        raise ValueError(
            f"pred_all_visium.csv 必须至少包含列: {required_cols}；当前列为: {list(pred.columns)}"
        )

    mask = (
        ((pred["ligand"] == ligand_name) & (pred["receptor"] == receptor_name)) |
        ((pred["ligand"] == receptor_name) & (pred["receptor"] == ligand_name))
    )
    pred_lr = pred[mask].copy()

    print(f"🔍 选中 LR = {receptor_name}-{ligand_name} (两种顺序都算)：")
    print(f"  边数量 = {len(pred_lr)}")

    if len(pred_lr) == 0:
        print("  ⚠️ 没有找到这对 LR 的预测，请检查 lr_table / pred_all_visium.csv 里的 gene 名。")

    return pred_lr


# ================== 3. 边级 → 节点级 ==================

def aggregate_edge_to_node(pred_lr, n_nodes):
    """
    把边级 LR 预测汇总成节点级：
      - 每条边 (src, dst, prob)
      - 把 prob 记到 src 和 dst
      - 对每个节点取 mean(prob)

    返回：
      node_mean: Series(index=node_idx, name="lr_prob_mean")
    """
    if len(pred_lr) == 0:
        return pd.Series(
            np.zeros(n_nodes, dtype=np.float32),
            index=np.arange(n_nodes),
            name="lr_prob_mean"
        )

    df_src = pred_lr[["src", "prob"]].rename(columns={"src": "node"})
    df_dst = pred_lr[["dst", "prob"]].rename(columns={"dst": "node"})
    df_nodes = pd.concat([df_src, df_dst], axis=0, ignore_index=True)

    node_mean = df_nodes.groupby("node")["prob"].mean()

    all_nodes = pd.Index(np.arange(n_nodes), name="node")
    node_mean = node_mean.reindex(all_nodes, fill_value=0.0)
    node_mean.name = "lr_prob_mean"

    return node_mean


# ================== 4. 主流程 ==================

def main():
    H5AD_PATH = "Human_Lymph_Node.h5ad"
    PRED_PATH = "visium_lr_benchmark/pred_all_visium.csv"

    # 1) 区域划分（老逻辑）
    adata, coords, regions = build_region_from_c2l(H5AD_PATH)
    n_nodes = adata.n_obs
    print(f"\nGraph 节点数（假定与模型索引对齐）: {n_nodes}")

    # 2) 读预测，抽取 CCR7–CCL19
    pred_lr = load_predictions_for_lr(PRED_PATH, ligand_name="CCL19", receptor_name="CCR7")

    # 3) 边 → 节点
    node_score = aggregate_edge_to_node(pred_lr, n_nodes)

    # 4) 拼一个节点表：node_idx + spot + region + lr_prob_mean
    df_nodes = pd.DataFrame({
        "node": np.arange(n_nodes, dtype=int),
        "spot": adata.obs_names.values,
        "region": regions.reindex(adata.obs_names).values,
        "lr_prob_mean": node_score.values
    })

    out_detail = "visium_lr_benchmark/CCR7_CCL19_node_scores_with_region.csv"
    df_nodes.to_csv(out_detail, index=False)
    print(f"\n✅ 已保存节点级 CCR7–CCL19 强度 + region 到: {out_detail}")

    # 5) 按区域看富集情况
    print("\n📊 各区域 CCR7–CCL19 节点平均 prob：")
    region_mean = df_nodes.groupby("region")["lr_prob_mean"].mean().sort_values(ascending=False)
    print(region_mean)

    print("\n📊 各区域 CCR7–CCL19 节点 prob 分布（描述统计）：")
    region_desc = df_nodes.groupby("region")["lr_prob_mean"].describe()
    print(region_desc)


if __name__ == "__main__":
    main()
