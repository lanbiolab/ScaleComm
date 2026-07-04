#!/usr/bin/env python
"""Append evidence panels to the selected manuscript figures without redesigning them.

The key design rule for this corrective version is append-only:
the selected Fig2-Fig5 PNGs are preserved as the main figure body, and new
panels are added below using compatible bar/heatmap/spatial-map forms.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PANDAS_USE_NUMEXPR", "0")
os.environ.setdefault("PANDAS_USE_BOTTLENECK", "0")
sys.modules["numexpr"] = None
sys.modules["bottleneck"] = None

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[2]
SELECTED = ROOT / "outputs/manuscript_experiment_figures_selected"
OUT = ROOT / "outputs/manuscript_experiment_figures_preserve_style_v3"
FIG_DIR = OUT / "figures"
QA_DIR = OUT / "qa"

CRC_ART = ROOT / "outputs/article_style_figure_additions/crc_visium_hd_article_style/source_data"
HP_SRC = ROOT / "outputs/article_style_figure_additions/high_priority_remaining/source_data"
DSS_ART = ROOT / "outputs/article_style_figure_additions/dss_merfish_article_style/source_data"
COSMX_ART = ROOT / "outputs/article_style_figure_additions/cosmx_nsclc_article_style/source_data"
TONSIL_ART = ROOT / "outputs/article_style_figure_additions/tonsil_article_style/source_data"

COL = {
    "ink": "#222222",
    "text": "#333333",
    "muted": "#6F6A61",
    "grid": "#E8E2D6",
    "neutral": "#D8D2C8",
    "red": "#C94F48",
    "red_dark": "#8F3431",
    "blue": "#2F78A8",
    "teal": "#237A78",
    "amber": "#D9992B",
    "gold": "#E6C06A",
    "purple": "#7E6C9A",
    "pale": "#F7F4EC",
}

HEAT = LinearSegmentedColormap.from_list("preserve_heat", ["#E6C06A", "#F7F4EC", "#2F78A8"])
RED_HEAT = LinearSegmentedColormap.from_list("preserve_red", ["#F7F4EC", "#E7C6C2", "#B85450"])
GOLD_HEAT = LinearSegmentedColormap.from_list("preserve_gold", ["#F7F4EC", "#F3D78A", "#B85450"])


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 7.5,
            "axes.labelsize": 6.6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 5.7,
            "axes.linewidth": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=10, fontweight="bold", color=COL["ink"])


def clean_axis(ax: plt.Axes, grid: str | None = None) -> None:
    ax.tick_params(width=0.45, length=2.0, pad=1.5, colors=COL["text"])
    ax.spines["left"].set_linewidth(0.55)
    ax.spines["bottom"].set_linewidth(0.55)
    ax.set_axisbelow(True)
    if grid == "x":
        ax.grid(axis="x", color=COL["grid"], lw=0.35, zorder=0)
    elif grid == "y":
        ax.grid(axis="y", color=COL["grid"], lw=0.35, zorder=0)
    else:
        ax.grid(False)


def nice(value: str) -> str:
    return str(value).replace("_proxy", "").replace("_", " ")


def load_base(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def make_append_figure(base_png: Path, out_stem: Path, add_height: float, draw_additions, ncols: int, *, top_width: float = 7.2) -> None:
    im = load_base(base_png)
    base_h = top_width * im.height / im.width
    fig = plt.figure(figsize=(top_width, base_h + add_height))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[base_h, add_height],
        left=0.035,
        right=0.985,
        bottom=0.035,
        top=0.985,
        hspace=0.06,
    )
    ax_base = fig.add_subplot(gs[0, 0])
    ax_base.imshow(im)
    ax_base.set_axis_off()
    sub = gs[1, 0].subgridspec(1, ncols, wspace=0.36)
    axes = [fig.add_subplot(sub[0, i]) for i in range(ncols)]
    draw_additions(axes)
    save(fig, out_stem)


def save(fig: plt.Figure, stem: Path, dpi: int = 500) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def draw_heatmap(ax: plt.Axes, matrix: pd.DataFrame, *, cmap=GOLD_HEAT, center: float | None = None, cbar: bool = True) -> None:
    mat = matrix.astype(float)
    if center is None:
        norm = Normalize(float(np.nanmin(mat.values)), float(np.nanmax(mat.values)) if float(np.nanmax(mat.values)) > 0 else 1)
    else:
        norm = TwoSlopeNorm(vmin=float(np.nanmin(mat.values)), vcenter=center, vmax=float(np.nanmax(mat.values)))
    ax.pcolormesh(
        np.arange(mat.shape[1] + 1),
        np.arange(mat.shape[0] + 1),
        mat.values,
        cmap=cmap,
        norm=norm,
        edgecolors="#5B554C",
        linewidth=0.18,
    )
    ax.set_xlim(0, mat.shape[1])
    ax.set_ylim(mat.shape[0], 0)
    ax.set_xticks(np.arange(mat.shape[1]) + 0.5)
    ax.set_xticklabels(mat.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(mat.shape[0]) + 0.5)
    ax.set_yticklabels(mat.index)
    ax.tick_params(length=0, pad=1.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if cbar:
        cax = ax.inset_axes([1.02, 0.12, 0.035, 0.76])
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
        cb.ax.tick_params(labelsize=5, length=1.5, width=0.35, pad=1)
        cb.outline.set_linewidth(0.35)


def draw_barh(ax: plt.Axes, labels: list[str], values: list[float], *, highlight: set[str] | None, color: str, xlabel: str) -> None:
    df = pd.DataFrame({"label": labels, "value": values}).dropna().sort_values("value", ascending=True)
    y = np.arange(len(df))
    colors = [color if highlight and lab in highlight else COL["neutral"] for lab in df["label"]]
    ax.barh(y, df["value"], color=colors, edgecolor="#77736B", linewidth=0.35, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.set_xlabel(xlabel)
    clean_axis(ax, "x")


def fig2() -> None:
    def additions(axes: list[plt.Axes]) -> None:
        genes = read_csv(CRC_ART / "target_gene_heatmap.csv")
        layers = read_csv(CRC_ART / "sender_receiver_layers.csv")
        pairs = read_csv(HP_SRC / "realdata_crc_tgfb1_tgfbr2_celltype_pairs.csv")

        ax = axes[0]
        pivot = genes.pivot_table(index="gene", columns="group", values="z", aggfunc="mean").fillna(0)
        keep_cols = [c for c in ["other", "ScaleComm top", "response hotspot"] if c in pivot.columns]
        pivot = pivot[keep_cols]
        pivot.columns = [c.replace("ScaleComm top", "ScaleComm").replace("response hotspot", "response") for c in pivot.columns]
        draw_heatmap(ax, pivot, cmap=HEAT, center=0)
        ax.set_title("Receiver target genes", pad=3)
        add_panel_label(ax, "j")

        ax = axes[1]
        ax.scatter(layers["x_plot"], layers["y_plot"], s=2.0, color="#D8D2C8", alpha=0.22, linewidths=0)
        for role, col in [("sender", COL["teal"]), ("receiver", COL["red"])]:
            sub = layers[layers["role"].eq(role)]
            ax.scatter(sub["x_plot"], sub["y_plot"], s=8 + 20 * (sub["score"] / max(sub["score"].max(), 1e-9)), color=col, alpha=0.62, linewidths=0, label=role)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.legend(loc="lower left", bbox_to_anchor=(0, -0.03), ncol=2, handletextpad=0.25, columnspacing=0.7)
        ax.set_title("Sending / receiving layers", pad=3)
        add_panel_label(ax, "k")

        ax = axes[2]
        mat = pairs.set_index("index").apply(pd.to_numeric, errors="coerce").fillna(0)
        rows = mat.sum(axis=1).sort_values(ascending=False).head(6).index
        cols = mat.sum(axis=0).sort_values(ascending=False).head(6).index
        mat = mat.loc[rows, cols]
        mat.index = [nice(v) for v in mat.index]
        mat.columns = [nice(v) for v in mat.columns]
        draw_heatmap(ax, mat, cmap=GOLD_HEAT)
        ax.set_title("Cell-type pair support", pad=3)
        ax.set_xlabel("receiver")
        ax.set_ylabel("")
        add_panel_label(ax, "l")

    make_append_figure(SELECTED / "fig2_crc_visium_hd_tgfb1_tgfbr2.png", FIG_DIR / "fig2_crc_visium_hd_tgfb1_tgfbr2_preserve_v3", 1.95, additions, 3)


def fig3() -> None:
    def additions(axes: list[plt.Axes]) -> None:
        volcano = read_csv(DSS_ART / "volcano.csv")
        flow = read_csv(DSS_ART / "flow.csv")

        ax = axes[0]
        v = volcano.copy()
        x = pd.to_numeric(v["effect_d9_d0"], errors="coerce").fillna(0)
        y = pd.to_numeric(v["evidence_plot"], errors="coerce").fillna(0)
        cats = v["biological_category"].astype(str)
        colors = np.where(cats.str.contains("Chemokine", case=False, na=False), COL["blue"], np.where(cats.str.contains("IL|TNF", case=False, na=False), COL["red"], COL["gold"]))
        ax.scatter(x, y, s=15, color=colors, alpha=0.70, edgecolors="white", linewidths=0.25)
        for lr in ["Cxcl10-Cxcr3", "Tgfb3-Acvrl1", "Il1b-Il1r2"]:
            row = v[v["lr_pair"].eq(lr)]
            if not row.empty:
                ax.scatter(row["effect_d9_d0"], row["evidence_plot"], s=42, facecolors="white", edgecolors=COL["red"], linewidths=0.9, zorder=4)
                ax.text(float(row["effect_d9_d0"].iloc[0]) + 0.015, float(row["evidence_plot"].iloc[0]), lr, fontsize=5.6, color=COL["red"], va="center")
        ax.axvline(0, color=COL["muted"], lw=0.55, ls=(0, (2, 2)))
        ax.set_xlabel("DSS9 - D0 communication activity")
        ax.set_ylabel("evidence")
        ax.set_title("Differential signaling landscape", pad=3)
        clean_axis(ax, "y")
        add_panel_label(ax, "k")

        ax = axes[1]
        pivot = flow.pivot_table(index="flow", columns="stage_label", values="stage_fraction", aggfunc="sum").fillna(0)
        top = pivot.sum(axis=1).sort_values(ascending=False).head(9).index
        pivot = pivot.loc[top, [c for c in ["D0", "D3", "D9", "D21"] if c in pivot.columns]]
        pivot.index = [str(v).replace(" -> ", "\n-> ") for v in pivot.index]
        draw_heatmap(ax, pivot, cmap=GOLD_HEAT)
        ax.set_title("Compartment communication rewiring", pad=3)
        add_panel_label(ax, "l")

    make_append_figure(SELECTED / "fig3_mouse_colon_dss_merfish_timecourse.png", FIG_DIR / "fig3_mouse_colon_dss_merfish_timecourse_preserve_v3", 1.65, additions, 2)


def fig4() -> None:
    def additions(axes: list[plt.Axes]) -> None:
        support = read_csv(COSMX_ART / "support.csv")
        burden = read_csv(HP_SRC / "realdata_cosmx_receiver_celltype_burden.csv")

        ax = axes[0]
        dat = support.nlargest(7, "top_receiver_target_lift").sort_values("top_receiver_target_lift")
        draw_barh(ax, dat["LR"].tolist(), dat["top_receiver_target_lift"].tolist(), highlight={"CXCL12-CXCR4", "CXCL10-CXCR3"}, color=COL["red"], xlabel="target-response lift")
        ax.set_title("Receiver target support", pad=3)
        add_panel_label(ax, "h")

        ax = axes[1]
        mat = burden.set_index("LR").apply(pd.to_numeric, errors="coerce").fillna(0)
        keep = [lr for lr in ["CXCL12-CXCR4", "CXCL10-CXCR3", "TGFB1-TGFBR2", "SPP1-CD44", "VEGFA-KDR"] if lr in mat.index]
        mat = mat.loc[keep]
        mat.index = [s.replace("-", "\n") for s in mat.index]
        draw_heatmap(ax, mat, cmap=GOLD_HEAT, cbar=True)
        ax.set_title("Receiver cell-type burden", pad=3)
        add_panel_label(ax, "i")

    make_append_figure(SELECTED / "fig4_cosmx_nsclc_scale_context.png", FIG_DIR / "fig4_cosmx_nsclc_scale_context_preserve_v3", 1.75, additions, 2)


def fig5() -> None:
    def additions(axes: list[plt.Axes]) -> None:
        sender_receiver = read_csv(TONSIL_ART / "sender_receiver.csv")
        replicate = read_csv(TONSIL_ART / "replicate.csv")
        scale = read_csv(TONSIL_ART / "scale.csv")

        ax = axes[0]
        sr = sender_receiver[sender_receiver["lr_pair"].eq("CXCL13-CXCR5")]
        mat = sr.pivot_table(index="sender_cell_type", columns="receiver_cell_type", values="n_edges", aggfunc="sum", fill_value=0)
        mat = mat.loc[mat.sum(axis=1).sort_values(ascending=False).head(6).index, mat.sum(axis=0).sort_values(ascending=False).head(6).index]
        mat.index = [nice(v) for v in mat.index]
        mat.columns = [nice(v) for v in mat.columns]
        draw_heatmap(ax, mat, cmap=GOLD_HEAT)
        ax.set_title("Sender-receiver cell pairs", pad=3)
        ax.set_xlabel("receiver")
        ax.set_ylabel("sender")
        add_panel_label(ax, "i")

        ax = axes[1]
        rep = replicate.melt(id_vars=["section"], value_vars=["target_lift_top_vs_background", "spearman_incoming_target", "gc_fold_enrichment"], var_name="metric", value_name="value")
        order = ["target_lift_top_vs_background", "spearman_incoming_target", "gc_fold_enrichment"]
        labels = ["target\nlift", "rho", "GC\nfold"]
        colors = [COL["red"], COL["blue"], COL["teal"]]
        for i, metric in enumerate(order):
            vals = rep.loc[rep["metric"].eq(metric), "value"].astype(float).to_numpy()
            x = np.full(len(vals), i) + np.linspace(-0.04, 0.04, len(vals))
            ax.scatter(x, vals, s=26, color=colors[i], edgecolors="white", linewidths=0.25, zorder=3)
            ax.plot([i - 0.18, i + 0.18], [vals.mean(), vals.mean()], color=COL["ink"], lw=0.8)
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels)
        ax.set_ylabel("support value")
        ax.set_title("A1/D1 support", pad=3)
        clean_axis(ax, "y")
        add_panel_label(ax, "j")

        ax = axes[2]
        scale_cols = [c for c in [f"scale_{i}" for i in range(6)] if c in scale.columns]
        keep = ["CXCL13-CXCR5", "CCL21-CCR7", "CD40LG-CD40", "IL21-IL21R", "TNFSF13B-TNFRSF13C"]
        smat = scale[scale["lr_pair"].isin(keep)].drop_duplicates("lr_pair").set_index("lr_pair")[scale_cols]
        smat.index = [s.replace("-", "\n") for s in smat.index]
        smat.columns = [f"s{i}" for i in range(len(smat.columns))]
        draw_heatmap(ax, smat, cmap=GOLD_HEAT)
        ax.set_title("LR scale profiles", pad=3)
        add_panel_label(ax, "k")

    make_append_figure(SELECTED / "fig5_spamosaic_tonsil_cxcl13_cxcr5.png", FIG_DIR / "fig5_spamosaic_tonsil_cxcl13_cxcr5_preserve_v3", 2.05, additions, 3)


def contact_sheet() -> None:
    paths = [
        FIG_DIR / "fig2_crc_visium_hd_tgfb1_tgfbr2_preserve_v3.png",
        FIG_DIR / "fig3_mouse_colon_dss_merfish_timecourse_preserve_v3.png",
        FIG_DIR / "fig4_cosmx_nsclc_scale_context_preserve_v3.png",
        FIG_DIR / "fig5_spamosaic_tonsil_cxcl13_cxcr5_preserve_v3.png",
    ]
    imgs = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        im.thumbnail((920, 1100), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (950, 1160), "white")
        canvas.paste(im, ((950 - im.width) // 2, 12))
        imgs.append(canvas)
    out = Image.new("RGB", (1900, 2320), "white")
    for i, im in enumerate(imgs):
        out.paste(im, ((i % 2) * 950, (i // 2) * 1160))
    QA_DIR.mkdir(parents=True, exist_ok=True)
    out.save(QA_DIR / "fig2_fig5_preserve_v3_contact_sheet.png", dpi=(220, 220))


def write_readme() -> None:
    spec = {
        "principle": "append-only preservation of outputs/manuscript_experiment_figures_selected figure forms",
        "base_figures": str(SELECTED),
        "new_panels": {
            "fig2": ["j receiver target genes", "k sender/receiver layers", "l cell-type pair support"],
            "fig3": ["k differential signaling landscape", "l compartment communication rewiring"],
            "fig4": ["h receiver target support", "i receiver cell-type burden"],
            "fig5": ["i sender-receiver cell pairs", "j A1/D1 support", "k LR scale profiles"],
        },
        "note": "The selected base figures are embedded unchanged as the main body. New panels are redrawn from existing source tables.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "README.md").write_text(json.dumps(spec, indent=2), encoding="utf-8")


def main() -> None:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    fig2()
    fig3()
    fig4()
    fig5()
    contact_sheet()
    write_readme()
    print(f"Wrote preserve-style v3 figures to {FIG_DIR}")
    print(f"Wrote QA contact sheet to {QA_DIR / 'fig2_fig5_preserve_v3_contact_sheet.png'}")


if __name__ == "__main__":
    main()
