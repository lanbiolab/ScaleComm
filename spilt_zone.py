# -*- coding: utf-8 -*-
"""
plot_region_from_c2l.py
----------------------------------
从 Human_Lymph_Node.h5ad 中：
  - 使用 means_cell_abundance_w_sf 做行归一化
  - 按 c2l 规则划分区域：
        Germinal center / B follicle / T cell zone / Mixed
  - 在 Visium 切片上标注并画出区域分布

图像风格基本与 SpotType_VisiumLymphNode_v2 一致。
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

H5_PATH = "Human_Lymph_Node.h5ad"
FIG_DIR = "fig"
os.makedirs(FIG_DIR, exist_ok=True)


def assign_region_c2l(row: pd.Series) -> str:
    """
    区域分类规则（与热图 / CCR7-CCL19 分析一致）：
      - GC: B_GC 相关 abundance 总和 > 0.2 或 任意 B_GC* 列 > 0.05
      - B follicle: 所有 B_* abundance 总和 > 0.35
      - T cell zone: T_CD4* 总和 > 0.35 或 T_TIM3* > 0.2
      - 其余: Mixed
    其中 row 是归一化后的 abundance（和细胞类型对应的一行）
    """
    # > 0.05 的细胞类型名，方便字符串判断
    names = " ".join(row.index[row > 0.05])

    # GC: B_GC* 较强
    if row.filter(like="B_GC").sum() > 0.2 or "B_GC" in names:
        return "Germinal center"
    # B follicle: B_* 整体占比高
    elif row.filter(like="B_").sum() > 0.35:
        return "B follicle"
    # T cell zone: T_CD4* 或 T_TIM3* 占比较高
    elif row.filter(like="T_CD4").sum() > 0.35 or row.filter(like="T_TIM3").sum() > 0.2:
        return "T cell zone"
    else:
        return "Mixed"


def main():
    # === 1. 加载数据 ===
    print(f"📥 加载 h5ad: {H5_PATH}")
    adata = sc.read_h5ad(H5_PATH)

    # 坐标
    if "spatial" not in adata.obsm_keys():
        raise KeyError("adata.obsm 中未找到 'spatial'，请检查 h5ad。")
    coords = pd.DataFrame(
        adata.obsm["spatial"],
        columns=["x", "y"],
        index=adata.obs_names
    )

    # cell2location abundance
    if "means_cell_abundance_w_sf" not in adata.obsm_keys():
        raise KeyError(
            "adata.obsm 中未找到 'means_cell_abundance_w_sf'，"
            "请确认是否运行过 cell2location/cell-typing。"
        )

    cell_ab = pd.DataFrame(
        adata.obsm["means_cell_abundance_w_sf"],
        index=adata.obs_names
    )

    # 行归一化
    cell_ab_norm = cell_ab.div(cell_ab.sum(axis=1), axis=0).fillna(0.0)

    print("✅ abundance 矩阵形状:", cell_ab_norm.shape)

    # === 2. 区域分类 ===
    print("🧭 开始按照 c2l 规则划分区域...")
    coords["region"] = cell_ab_norm.apply(assign_region_c2l, axis=1)

    print("\n📊 区域分布统计：")
    print(coords["region"].value_counts())

    # 可选：保存一个 CSV 给其他脚本复用
    region_out = os.path.join("visium_lr_benchmark", "spot_region_from_c2l_recomputed.csv")
    os.makedirs(os.path.dirname(region_out), exist_ok=True)
    out_df = coords[["x", "y", "region"]].copy()
    out_df.insert(0, "spot", coords.index)
    out_df.to_csv(region_out, index=False)
    print(f"\n💾 已保存区域注释 CSV: {region_out}")

    # === 3. 绘图参数 ===
    styles = {
        "T cell zone": dict(marker="s", color="#e41a1c", s=38, alpha=0.88,
                            label="T cell zone", zorder=3),
        "B follicle": dict(marker="^", color="#9d9d9d", s=33, alpha=0.8,
                           label="B follicle", zorder=2),
        "Germinal center": dict(marker="+", color="#984ea3", s=80, alpha=1.0,
                                label="Germinal center", zorder=4),
        "Mixed": dict(marker="o", color="#252525", s=42, alpha=0.5,
                      label="Mixed", zorder=1),
    }

    # === 4. 绘制 ===
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', adjustable='datalim')

    # 背景点（灰白格子感）
    plt.hexbin(
        coords["x"],
        -coords["y"],
        gridsize=70,
        cmap="Greys",
        mincnt=1,
        alpha=0.10,
        linewidths=0.0
    )

    # 按顺序绘制各区域
    for region in ["B follicle", "T cell zone", "Germinal center", "Mixed"]:
        subset = coords[coords["region"] == region]
        if len(subset) > 0:
            ax.scatter(subset["x"], -subset["y"], **styles[region])

    # 样式优化
    plt.title("Spot type distribution (Visium lymph node)", fontsize=14, pad=10)
    ax.axis("off")

    # 图例 — 放到右上角外侧
    ax.legend(
        frameon=False,
        loc='upper right',
        bbox_to_anchor=(1.18, 1.02),
        title="Spot type",
        title_fontsize=11,
        fontsize=10,
        labelspacing=0.4,
        handletextpad=0.5,
        borderpad=0.2
    )

    # 图像比例自然
    plt.xlim(coords["x"].min() - 150, coords["x"].max() + 150)
    plt.ylim(-coords["y"].max() - 150, -coords["y"].min() + 150)
    plt.tight_layout()

    # === 5. 保存 ===
    save_path = os.path.join(FIG_DIR, "SpotType_VisiumLymphNode_c2lRegion.png")
    plt.savefig(save_path, dpi=400, bbox_inches="tight", pad_inches=0.05)
    plt.close()

    print(f"\n✅ 已保存高质量区域图: {save_path}")


if __name__ == "__main__":
    main()
