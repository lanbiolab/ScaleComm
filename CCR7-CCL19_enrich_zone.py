# -*- coding: utf-8 -*-
"""
plot_CCR7_CCL19_enrichment_heatmap_LargeFont.py
-----------------------------------------
在 Visium 切片上画 CCR7–CCL19 Top 20% 边的“富集热图”：
  - 字体已调大 (Title=18, Label=14, Ticks=12)
  - 使用 Top20% CCR7–CCL19 边
  - 对每个 spot 统计：被这些高置信边命中的次数（src/dst 都算）
  - 把这个 edge_count 映射到切片坐标上，用红蓝色热图展示

输入：
  - ../data/Human_Lymph_Node.h5ad
  - visium_lr_benchmark/CCR7_CCL19_top20pct_edges_with_region.csv

输出：
  - fig/CCR7_CCL19_enrichment_heatmap_LargeFont.png
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

# ========== 路径配置 ==========
H5_PATH = "Human_Lymph_Node.h5ad"
TOP20_EDGES_PATH = "visium_lr_benchmark/CCR7_CCL19_top20pct_edges_with_region.csv"

FIG_DIR = "fig"
os.makedirs(FIG_DIR, exist_ok=True)


# ========== 区域划分函数 ==========
def assign_region(row: pd.Series) -> str:
    """
    row: 单个 spot 的细胞丰度（已按行归一化）
    规则：
      - B_GC 总和 > 0.2 或 出现 B_GC 细胞 → Germinal center
      - 所有 B_ 前缀之和 > 0.35 → B follicle
      - T_CD4+ 总和 > 0.35 或 T_TIM3+ 总和 > 0.2 → T cell zone
      - 否则 Mixed
    """
    names = " ".join(row.index[row > 0.05])

    if row.filter(like="B_GC").sum() > 0.2 or "B_GC" in names:
        return "Germinal center"
    elif row.filter(like="B_").sum() > 0.35:
        return "B follicle"
    elif row.filter(like="T_CD4").sum() > 0.35 or row.filter(like="T_TIM3").sum() > 0.2:
        return "T cell zone"
    else:
        return "Mixed"


def main():
    # ================= 1. 读取 Visium & 坐标 =================
    print(f"📥 读取 Visium h5ad: {H5_PATH}")
    # 注意：根据你的环境，如果文件不在当前目录，请修改路径
    adata = sc.read_h5ad(H5_PATH)
    n_spots = adata.n_obs
    print(f"  spots = {n_spots}")

    # 坐标：index 对应节点 0..n-1
    coords = pd.DataFrame(
        adata.obsm["spatial"],
        columns=["x", "y"]
    )
    coords["node"] = np.arange(n_spots)
    coords = coords.set_index("node")

    # 背景坐标（注意 y 取负号，让方向和之前所有图保持一致）
    bg_x = coords["x"].values
    bg_y = -coords["y"].values

    # ================= 2. 基于 means_cell_abundance_w_sf 划分区域（只是为了信息） =================
    cell_ab = pd.DataFrame(
        adata.obsm["means_cell_abundance_w_sf"],
        index=adata.obs_names
    )
    cell_ab_norm = cell_ab.div(cell_ab.sum(axis=1), axis=0).fillna(0.0)
    region_series = cell_ab_norm.apply(assign_region, axis=1)

    region_df = pd.DataFrame(
        {
            "node": np.arange(n_spots),
            "region": region_series.values
        }
    ).set_index("node")

    print("📊 Spot 区域分布（来自 cell2location）：")
    print(region_df["region"].value_counts())
    print()

    # ================= 3. 读取 Top20% CCR7–CCL19 边 =================
    print(f"📥 读取 Top20% CCR7–CCL19 边: {TOP20_EDGES_PATH}")
    edges = pd.read_csv(TOP20_EDGES_PATH)

    if not {"src", "dst", "prob"}.issubset(edges.columns):
        raise ValueError(f"{TOP20_EDGES_PATH} 必须包含 src, dst, prob 三列。")

    src = edges["src"].to_numpy(dtype=int)
    dst = edges["dst"].to_numpy(dtype=int)

    # 防止越界
    valid_mask = (
        (src >= 0) & (src < n_spots) &
        (dst >= 0) & (dst < n_spots)
    )
    if not np.all(valid_mask):
        print(f"⚠️ 有 {np.sum(~valid_mask)} 条边 src/dst 越界，将被丢弃。")
        src = src[valid_mask]
        dst = dst[valid_mask]

    # ================= 4. 统计每个 spot 被命中的次数 =================
    edge_count = np.zeros(n_spots, dtype=float)

    # 每条边给 src 和 dst 都 +1
    for a, b in zip(src, dst):
        edge_count[a] += 1.0
        edge_count[b] += 1.0

    # 把 edge_count 挂回到 coords 和 region_df 里
    coords["edge_count"] = edge_count
    region_df["edge_count"] = edge_count

    # 看一下各区域平均命中次数
    region_mean = region_df.groupby("region")["edge_count"].mean()
    print("📊 各区域平均 Top20% CCR7–CCL19 命中次数（per spot）：")
    print(region_mean.sort_values(ascending=False))
    print()

    # ================= 5. 画富集热图（大字号版） =================
    # 取 99 分位作为 vmax，避免极少数特别高的点把颜色拉爆
    heat_vals = coords["edge_count"].values
    vmax = np.percentile(heat_vals[heat_vals > 0], 99) if np.any(heat_vals > 0) else 1.0

    # 保持正方形比例 6.5 x 6.5
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_aspect("equal", "datalim")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # 散点图
    sc_ = ax.scatter(
        bg_x,
        bg_y,
        s=18,
        c=heat_vals,
        cmap="coolwarm",      # 蓝低红高
        vmin=0,
        vmax=vmax,
        linewidths=0
    )

    ax.axis("off")

    # 设置视野范围
    margin = 150
    ax.set_xlim(bg_x.min() - margin, bg_x.max() + margin)
    ax.set_ylim(bg_y.min() - margin, bg_y.max() + margin)

    # === Colorbar 调整 ===
    cbar = plt.colorbar(sc_, ax=ax, fraction=0.046, pad=0.04)
    
    # 【修改】增大 Colorbar 侧边的标签字号
    cbar.set_label("Top 20% CCR7–CCL19 edge count per spot", fontsize=14)
    
    # 【修改】增大 Colorbar 刻度数字的字号
    cbar.ax.tick_params(labelsize=12)

    # === 标题调整 ===
    # 【修改】增大标题字号并增加下边距
    ax.set_title("CCR7–CCL19 enrichment heatmap", fontsize=18, pad=12)

    plt.tight_layout()

    out_path = os.path.join(FIG_DIR, "CCR7_CCL19_enrichment_heatmap_LargeFont.png")
    plt.savefig(out_path, dpi=450, bbox_inches="tight", pad_inches=0.05)
    plt.close()

    print(f"📸 已保存大字号富集热图: {out_path}")


if __name__ == "__main__":
    main()