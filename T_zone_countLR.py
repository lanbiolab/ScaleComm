# -*- coding: utf-8 -*-
"""
plot_Tzone_LR_top10_Square.py
----------------------------------
只画一个柱状图：
    - T cell zone 内部 top10 LR 对数量
    - 画布调整为正方形 (Square-ish)
    - 字体整体调大
    - 自动扩大 y 轴上限（避免顶到上沿）
    - CCR7–CCL19 标签红色加粗
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

# ========= 路径配置 =========
H5_PATH = "Human_Lymph_Node.h5ad"
REGION_CSV = "visium_lr_benchmark/spot_region_from_c2l.csv"
PRED_CSV = "visium_lr_benchmark/pred_all_visium.csv"
FIG_DIR = "fig"
os.makedirs(FIG_DIR, exist_ok=True)

PROB_THR = 0.8   # T cell zone 内部“高置信度”阈值


def main():
    # -----------------------------------------
    # 1) 读取 Visium，拿到 node->spot 映射
    # -----------------------------------------
    print(f"📥 读取 Visium h5ad: {H5_PATH}")
    # 注意：如果数据很大，这里可能耗时
    adata = sc.read_h5ad(H5_PATH)
    n_nodes = adata.n_obs

    node_df = pd.DataFrame({
        "node": np.arange(n_nodes),
        "spot": adata.obs_names.to_list()
    })

    # -----------------------------------------
    # 2) 读取 region 注释（c2l）
    # -----------------------------------------
    print(f"📥 读取 region 注释: {REGION_CSV}")
    region_df = pd.read_csv(REGION_CSV)

    need_cols = {"spot", "x", "y", "region"}
    if not need_cols.issubset(region_df.columns):
        raise ValueError(f"{REGION_CSV} 缺少列：{need_cols}")

    node_df = node_df.merge(region_df[["spot", "x", "y", "region"]],
                            on="spot", how="left")
    node_df["region"] = node_df["region"].fillna("Unknown")

    node_region = node_df.set_index("node")["region"]

    print("\n📊 region 分布：")
    print(node_df["region"].value_counts())

    # -----------------------------------------
    # 3) 读取所有 LR 预测
    # -----------------------------------------
    print(f"\n📥 读取 LR 预测: {PRED_CSV}")
    pred_all = pd.read_csv(PRED_CSV)

    need_pred = {"src", "dst", "ligand", "receptor", "prob"}
    if not need_pred.issubset(pred_all.columns):
        raise ValueError(f"{PRED_CSV} 缺少列：{need_pred}")

    # -----------------------------------------
    # 4) 给边标注 src/dst 的 region
    # -----------------------------------------
    pred_all["src_region"] = pred_all["src"].map(node_region)
    pred_all["dst_region"] = pred_all["dst"].map(node_region)

    mask_t = (
        (pred_all["src_region"] == "T cell zone") &
        (pred_all["dst_region"] == "T cell zone")
    )
    pred_t = pred_all[mask_t].copy()

    print(f"\n🔍 T cell zone 内部边数量: {len(pred_t)}")
    if len(pred_t) == 0:
        print("⚠️ 没有边，无法绘制柱状图。")
        return

    # -----------------------------------------
    # 5) 选择高置信度 or 全部
    # -----------------------------------------
    pred_t_strong = pred_t[pred_t["prob"] >= PROB_THR]
    if len(pred_t_strong) > 0:
        base_df = pred_t_strong
        print(f"  使用 prob ≥ {PROB_THR} 的边: {len(pred_t_strong)}")
    else:
        base_df = pred_t
        print("  无高置信度边 → 使用所有边")

    # -----------------------------------------
    # 6) 统计 top10 LR pair
    # -----------------------------------------
    base_df["lr_pair"] = base_df["ligand"].astype(str) + "-" + base_df["receptor"].astype(str)
    top10 = base_df["lr_pair"].value_counts().head(10)

    print("\n📊 Top10 LR pairs inside T cell zone:")
    print(top10)

    # -----------------------------------------
    # 7) Plot: Square Shape & Larger Font
    # -----------------------------------------
    x_labels = top10.index.tolist()
    y_vals = top10.values
    x_pos = np.arange(len(x_labels))

    # y 轴最大值自动扩展：25% buffer
    max_y = y_vals.max()
    ymax = max_y * 1.25

    # 【修改】画布改为正方形比例 (6.5 x 6.5)
    plt.figure(figsize=(6.5, 6.5))
    ax = plt.gca()

    # 不画顶部和右侧框线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 蓝色柱子
    bars = ax.bar(x_pos, y_vals, color="#4c72b0", width=0.7)

    # 固定 ylimit
    ax.set_ylim(0, ymax)

    # 【修改】增大刻度标签字号
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)

    # x 轴标签设置
    ax.set_xticks(x_pos)
    # 【修改】rotation 加大一点，防止变窄后挤压，字号加大到 12
    ax.set_xticklabels(x_labels, rotation=50, ha="right", fontsize=12)

    # 特殊标记 CCR7-CCL19
    ccr_set = {"CCR7-CCL19", "CCL19-CCR7"}
    for tick_lbl, name in zip(ax.get_xticklabels(), x_labels):
        if name in ccr_set:
            tick_lbl.set_color("#d62728")
            tick_lbl.set_fontweight("bold")

    # 柱顶数字
    for x, v in zip(x_pos, y_vals):
        # 【修改】字号加大到 11
        ax.text(x, v + (ymax * 0.015), f"{int(v)}", ha="center", va="bottom", fontsize=11)

    # 轴标签
    # 【修改】字号加大到 14
    ax.set_ylabel("Number of communication edges", fontsize=14)
    ax.set_xlabel("Ligand–receptor pair", fontsize=14)

    plt.tight_layout()

    out_f = os.path.join(FIG_DIR, "Visium_Tzone_LR_Top10_Square.png")
    plt.savefig(out_f, dpi=450, bbox_inches="tight", pad_inches=0.05)
    plt.close()

    print(f"\n✅ 图已保存: {out_f}")


if __name__ == "__main__":
    main()