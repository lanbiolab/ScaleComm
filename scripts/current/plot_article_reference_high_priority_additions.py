#!/usr/bin/env python
"""High-priority article-reference figure additions.

This script implements the remaining high-value panel families from
`article_figures/`: benchmark sensitivity/error-mode panels and real-data
interaction network / per-location communication burden panels. It does not
replace the current main DSS/CRC/CosMx/tonsil figures.
"""

from __future__ import annotations

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
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/article_style_figure_additions/high_priority_remaining"
SOURCE = OUT / "source_data"

BENCH = ROOT / "outputs/reference_based_benchmark/fig2_mechanism_benchmark"
EPOCH = ROOT / "outputs/reference_based_benchmark/scdesign3_mechanism_200lr/epoch_sensitivity/epoch_sensitivity_50_1000_3000_summary.csv"
SCALE_MECH = ROOT / "outputs/current_fig5_lr_scale_mechanism/source_data"
DSS_BASE = ROOT / "outputs/current_mouse_colon_ibd_merfish"
TONSIL = ROOT / "outputs/current_spamosaic_tonsil_scalecomm_story/source_data"
CRC_ARTICLE = ROOT / "outputs/article_style_figure_additions/crc_visium_hd_article_style/source_data"
COSMX = ROOT / "outputs/current_fig5_cosmx_nsclc_scale_context/source_tables"
AUDIT = ROOT / "outputs/article_style_figure_additions/article_figures_75panel_audit.csv"

INK = "#181512"
TEXT = "#2D2A27"
MUTED = "#6F675D"
GRID = "#E7E1D8"
BG = "#FFFFFF"
TAUPE = "#BDB8AE"
PALE = "#F7F4EC"
RED = "#B85450"
DEEP_RED = "#7A2828"
BLUE = "#2F78A8"
TEAL = "#237A78"
AMBER = "#D7B56D"
PURPLE = "#7E6C9A"
GREEN = "#6FA27A"

METHOD_ORDER = [
    "ScaleComm",
    "COMMOT",
    "SpaOTsc",
    "CellChat",
    "CellNEST",
    "CellPhoneDB",
    "NICHES",
    "CytoSignal",
    "DeepTalk",
    "SpaTalk",
]

METHOD_COLORS = {
    "ScaleComm": RED,
    "COMMOT": BLUE,
    "SpaOTsc": GREEN,
    "CellChat": AMBER,
    "CellNEST": PURPLE,
    "CellPhoneDB": "#B88A52",
    "NICHES": "#82976B",
    "CytoSignal": "#6E8BA8",
    "DeepTalk": "#767676",
    "SpaTalk": "#B9A7C9",
}

RANGE_ORDER = ["contact", "short", "medium", "long_niche"]
RANGE_LABELS = {
    "contact": "contact",
    "short": "short",
    "medium": "medium",
    "long_niche": "long\nniche",
}
STAGE_LABELS = ["D0", "D3", "D9", "D21"]
COMPARTMENTS = ["Epithelial", "Stromal", "Immune", "Vascular", "Neural"]
COMPARTMENT_SHORT = {"Epithelial": "Epi", "Stromal": "Str", "Immune": "Imm", "Vascular": "Vas", "Neural": "Neu"}

HEAT_HIGH = LinearSegmentedColormap.from_list("heat_high", [PALE, "#E6C06A", BLUE])
HEAT_ERR = LinearSegmentedColormap.from_list("heat_err", [PALE, "#E6C06A", RED, DEEP_RED])
BURDEN = LinearSegmentedColormap.from_list("burden", [PALE, "#F0C47B", RED, DEEP_RED])
NETWORK = LinearSegmentedColormap.from_list("network", [PALE, "#D9B66F", "#A8514E"])


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.2,
            "axes.titlesize": 7.1,
            "axes.labelsize": 6.2,
            "xtick.labelsize": 5.4,
            "ytick.labelsize": 5.4,
            "legend.fontsize": 5.1,
            "axes.linewidth": 0.55,
            "axes.facecolor": BG,
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def display_method(value: str) -> str:
    lookup = {
        "SpatialSelf": "ScaleComm",
        "spatialself_range_context": "ScaleComm",
        "commot": "COMMOT",
        "cellchat": "CellChat",
        "cellnest": "CellNEST",
        "deeptalk": "DeepTalk",
        "spaotsc": "SpaOTsc",
        "spatalk": "SpaTalk",
        "cellphonedb": "CellPhoneDB",
        "niches": "NICHES",
        "cytosignal": "CytoSignal",
    }
    return lookup.get(str(value), str(value))


def method_sort_key(method: str) -> int:
    return METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)


def clean_axis(ax: plt.Axes, grid: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_linewidth(0.55)
    ax.spines["bottom"].set_linewidth(0.55)
    ax.tick_params(width=0.45, length=2.0, pad=1.4, colors=TEXT)
    if grid == "x":
        ax.grid(axis="x", color=GRID, lw=0.34, zorder=0)
    elif grid == "y":
        ax.grid(axis="y", color=GRID, lw=0.34, zorder=0)
    ax.set_axisbelow(True)


def panel_label(fig: plt.Figure, ax: plt.Axes, label: str, dx: float = -0.030, dy: float = 0.012) -> None:
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, label, ha="left", va="bottom", fontsize=9.0, fontweight="bold", color=INK)


def robust_norm(values: pd.Series, qlo: float = 0.02, qhi: float = 0.98) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").astype(float)
    lo, hi = arr.quantile(qlo), arr.quantile(qhi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = arr.min(), arr.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(arr)), index=arr.index)
    return ((arr - lo) / (hi - lo)).clip(0, 1)


def pretty_label(value: str) -> str:
    text = str(value).replace("_proxy", "").replace("_", " ")
    text = text.replace("spp1 macrophage", "SPP1 macrophage")
    text = text.replace("t cell", "T cell")
    text = text.replace("b cell", "B cell")
    text = text.replace("fdc", "FDC")
    text = text.replace("tfh", "Tfh")
    text = text.replace("caf", "CAF")
    return text


def add_gradient_legend(ax: plt.Axes, cmap: LinearSegmentedColormap, label: str, x: float = 0.72, y: float = 1.015) -> None:
    cax = ax.inset_axes([x, y, 0.26, 0.032])
    grad = np.linspace(0, 1, 128).reshape(1, -1)
    cax.imshow(grad, cmap=cmap, aspect="auto")
    cax.set_xticks([0, 127])
    cax.set_xticklabels(["low", "high"])
    cax.set_yticks([])
    cax.tick_params(length=0, labelsize=4.7, pad=0.6)
    cax.text(-0.04, 0.5, label, transform=cax.transAxes, ha="right", va="center", fontsize=4.8, color=MUTED)
    for spine in cax.spines.values():
        spine.set_visible(False)


def draw_heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    cmap: LinearSegmentedColormap,
    vmin: float | None = None,
    vmax: float | None = None,
    annotate: bool = False,
    fmt: str = ".2f",
    edge: str = "#FFFFFF",
    lw: float = 0.34,
) -> None:
    data = matrix.to_numpy(dtype=float)
    finite = data[np.isfinite(data)]
    if vmin is None:
        vmin = float(np.nanmin(finite)) if finite.size else 0.0
    if vmax is None:
        vmax = float(np.nanmax(finite)) if finite.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-9
    norm = Normalize(vmin=vmin, vmax=vmax)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            color = "#F1EDE6" if not np.isfinite(val) else cmap(norm(val))
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor=edge, linewidth=lw))
            if annotate and np.isfinite(val):
                ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=4.9, color=INK)
    ax.set_xlim(-0.5, data.shape[1] - 0.5)
    ax.set_ylim(data.shape[0] - 0.5, -0.5)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(matrix.columns.tolist())
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(matrix.index.tolist())
    ax.tick_params(length=0, pad=1.2)
    for spine in ax.spines.values():
        spine.set_visible(False)


def spatial_style(ax: plt.Axes, frame: pd.DataFrame, x_col: str = "x_plot", y_col: str = "y_plot") -> None:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    px = (float(frame[x_col].max()) - float(frame[x_col].min())) * 0.025
    py = (float(frame[y_col].max()) - float(frame[y_col].min())) * 0.025
    ax.set_xlim(frame[x_col].min() - px, frame[x_col].max() + px)
    ax.set_ylim(frame[y_col].min() - py, frame[y_col].max() + py)
    for spine in ax.spines.values():
        spine.set_visible(False)


def load_benchmark_tables() -> dict[str, pd.DataFrame]:
    clean = pd.read_csv(BENCH / "fig2_clean_overall_metrics_source.csv")
    noisy = pd.read_csv(BENCH / "fig2_noisy_overall_metrics_source.csv")
    mech = pd.read_csv(BENCH / "fig2_panel_d_source.csv")
    epoch = pd.read_csv(EPOCH)
    ablation = pd.read_csv(SCALE_MECH / "panel_h_scale_context_ablation.csv")
    wrong = pd.read_csv(SCALE_MECH / "panel_f_wrong_range_suppression.csv")
    for df in [clean, noisy, mech]:
        df["method_label"] = df["method"].map(display_method)
    wrong["method_label"] = wrong["method_label"].map(display_method)
    return {"clean": clean, "noisy": noisy, "mech": mech, "epoch": epoch, "ablation": ablation, "wrong": wrong}


def plot_overall_metric_strip(ax: plt.Axes, clean: pd.DataFrame) -> pd.DataFrame:
    clean = clean.copy()
    clean["fdr_control"] = 1.0 - clean["empirical_fdr_at_threshold"].astype(float)
    metrics = [
        ("auroc", "AUROC"),
        ("aupr", "AUPR"),
        ("top_k_precision", "top-k\nprecision"),
        ("fdr_control", "1 - FDR"),
    ]
    rows = []
    for _, row in clean.iterrows():
        method = row["method_label"]
        if method not in METHOD_ORDER:
            continue
        for metric, label in metrics:
            value = float(row[metric])
            rows.append({"method": method, "metric": metric, "metric_label": label, "value": value})
    long = pd.DataFrame(rows)
    x_lookup = {m: i for i, (_, m) in enumerate(metrics)}
    for method in METHOD_ORDER:
        sub = long[long["method"].eq(method)]
        if sub.empty:
            continue
        xs = [x_lookup[m] for m in sub["metric_label"]]
        color = METHOD_COLORS.get(method, TAUPE)
        alpha = 0.95 if method in ["ScaleComm", "COMMOT", "CellNEST"] else 0.45
        z = 5 if method == "ScaleComm" else 3
        ax.plot(xs, sub["value"], color=color, lw=0.70 if method == "ScaleComm" else 0.45, alpha=alpha * 0.75, zorder=z)
        ax.scatter(xs, sub["value"], s=18 if method == "ScaleComm" else 12, color=color, edgecolor="white", linewidth=0.28, alpha=alpha, zorder=z + 1)
        if method in ["ScaleComm", "COMMOT", "CellNEST"]:
            last = sub[sub["metric_label"].eq("1 - FDR")]
            if not last.empty:
                ax.text(3.08, float(last["value"].iloc[0]), method, fontsize=5.4, va="center", ha="left", color=color)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.25, 3.85)
    ax.set_ylabel("higher is better")
    ax.set_title("Overall semi-synthetic benchmark", pad=4)
    clean_axis(ax, "y")
    return long


def plot_mechanism_heatmap(ax: plt.Axes, mech: pd.DataFrame) -> pd.DataFrame:
    dat = mech[mech["method_label"].isin(METHOD_ORDER)].copy()
    dat["method_label"] = pd.Categorical(dat["method_label"], categories=METHOD_ORDER, ordered=True)
    mat = (
        dat.pivot_table(index="method_label", columns="range_class", values="aupr", observed=False)
        .reindex(index=METHOD_ORDER, columns=RANGE_ORDER)
        .dropna(how="all")
    )
    mat.columns = [RANGE_LABELS[c] for c in mat.columns]
    draw_heatmap(ax, mat, HEAT_HIGH, vmin=0, vmax=max(0.82, float(np.nanmax(mat.to_numpy()))), annotate=False, edge="#FFFFFF", lw=0.45)
    ax.set_title("AUPR by LR range class", pad=4)
    add_gradient_legend(ax, HEAT_HIGH, "AUPR", x=0.73, y=1.015)
    for tick in ax.get_yticklabels():
        tick.set_color(METHOD_COLORS.get(tick.get_text(), TEXT))
    return mat.reset_index().rename(columns={"method_label": "method"})


def plot_noise_robustness(ax: plt.Axes, noisy: pd.DataFrame) -> pd.DataFrame:
    dat = noisy[noisy["method_label"].isin(METHOD_ORDER)].copy()
    dat["noise_x"] = dat["noise_level"].map({"low_noise": 0, "high_noise": 1})
    for method in METHOD_ORDER:
        sub = dat[dat["method_label"].eq(method)].sort_values("noise_x")
        if sub.shape[0] < 2:
            continue
        color = METHOD_COLORS.get(method, TAUPE)
        alpha = 1.0 if method in ["ScaleComm", "COMMOT", "CellNEST", "CellChat"] else 0.42
        lw = 1.25 if method == "ScaleComm" else 0.62
        ax.plot(sub["noise_x"], sub["aupr"], color=color, lw=lw, alpha=alpha, zorder=4 if method == "ScaleComm" else 2)
        ax.scatter(sub["noise_x"], sub["aupr"], s=23 if method == "ScaleComm" else 13, color=color, edgecolor="white", linewidth=0.28, alpha=alpha, zorder=5)
        if method in ["ScaleComm", "COMMOT", "CellNEST", "CellChat"]:
            y = float(sub[sub["noise_x"].eq(1)]["aupr"].iloc[0])
            ax.text(1.04, y, method, fontsize=5.2, color=color, va="center", ha="left")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["low\nnoise", "high\nnoise"])
    ax.set_ylim(0, 0.66)
    ax.set_xlim(-0.14, 1.58)
    ax.set_ylabel("AUPR")
    ax.set_title("Noise robustness", pad=4)
    clean_axis(ax, "y")
    return dat[["noise_level", "method_label", "aupr", "empirical_fdr_at_threshold"]].rename(columns={"method_label": "method"})


def plot_error_heatmap(ax: plt.Axes, clean: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "empirical_fdr_at_threshold": "FDR",
        "wrong_range_call_rate": "wrong\nrange",
        "wrong_niche_call_rate": "wrong\nniche",
        "too_close_long_range_call_rate": "too-close\nlong",
    }
    dat = clean[clean["method_label"].isin(METHOD_ORDER)].copy()
    mat = dat.set_index("method_label")[list(cols)].rename(columns=cols).reindex(METHOD_ORDER).dropna(how="all")
    draw_heatmap(ax, mat, HEAT_ERR, vmin=0, vmax=1, annotate=False, edge="#FFFFFF", lw=0.45)
    ax.set_title("Error-mode rates", pad=4)
    add_gradient_legend(ax, HEAT_ERR, "rate", x=0.73, y=1.015)
    for tick in ax.get_yticklabels():
        tick.set_color(METHOD_COLORS.get(tick.get_text(), TEXT))
    return mat.reset_index().rename(columns={"method_label": "method"})


def plot_epoch_sensitivity(ax: plt.Axes, epoch: pd.DataFrame) -> pd.DataFrame:
    colors = {"lymph": BLUE, "melanoma": RED}
    labels = {"lymph": "LN reference", "melanoma": "MEL reference"}
    for tissue in ["lymph", "melanoma"]:
        sub = epoch[epoch["tissue"].eq(tissue)].sort_values("epoch")
        ax.plot(sub["epoch"], sub["aupr"], marker="o", ms=3.2, lw=1.05, color=colors[tissue], label=labels[tissue])
        for _, row in sub.iterrows():
            if int(row["epoch"]) == 1000:
                ax.scatter([row["epoch"]], [row["aupr"]], s=42, facecolor="white", edgecolor=colors[tissue], linewidth=0.9, zorder=4)
    ax.set_xscale("log")
    ax.set_xticks([50, 1000, 3000])
    ax.set_xticklabels(["50", "1k", "3k"])
    ax.set_ylim(0.40, 0.72)
    ax.set_ylabel("AUPR")
    ax.set_xlabel("training epochs")
    ax.set_title("Epoch sensitivity", pad=4)
    ax.legend(loc="lower right", frameon=False, handlelength=1.3)
    clean_axis(ax, "y")
    return epoch.copy()


def plot_ablation(ax: plt.Axes, ablation: pd.DataFrame) -> pd.DataFrame:
    dat = ablation.copy()
    dat["plot_label"] = dat["variant_label"].replace({"Final gate": "scale/context gate"})
    dat = dat.sort_values("selection_score", ascending=True)
    y = np.arange(len(dat))
    colors = [RED if v == "scale/context gate" else "#CFC8BD" for v in dat["plot_label"]]
    ax.barh(y, dat["selection_score"], color=colors, height=0.58, edgecolor=INK, linewidth=0.26, alpha=0.94)
    ax.set_yticks(y)
    ax.set_yticklabels(dat["plot_label"].tolist())
    ax.set_xlim(0.50, 0.84)
    ax.set_xlabel("integrated selection score")
    ax.set_title("Scale/context ablation", pad=4)
    clean_axis(ax, "x")
    return dat


def plot_benchmark_additions() -> None:
    tables = load_benchmark_tables()
    fig = plt.figure(figsize=(7.2, 7.35))
    gs = fig.add_gridspec(3, 2, left=0.065, right=0.985, top=0.965, bottom=0.070, hspace=0.58, wspace=0.42)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]

    source_tables = {
        "overall_metric_strip": plot_overall_metric_strip(axes[0], tables["clean"]),
        "mechanism_aupr_heatmap": plot_mechanism_heatmap(axes[1], tables["mech"]),
        "noise_robustness": plot_noise_robustness(axes[2], tables["noisy"]),
        "error_mode_heatmap": plot_error_heatmap(axes[3], tables["clean"]),
        "epoch_sensitivity": plot_epoch_sensitivity(axes[4], tables["epoch"]),
        "scale_context_ablation": plot_ablation(axes[5], tables["ablation"]),
    }
    for label, ax in zip("abcdef", axes, strict=False):
        panel_label(fig, ax, label)

    for name, table in source_tables.items():
        table.to_csv(SOURCE / f"benchmark_{name}.csv", index=False)

    base = OUT / "article_reference_benchmark_high_priority"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def split_flow(flow: str) -> tuple[str, str]:
    left, right = flow.split(" -> ")
    return left.strip(), right.strip()


def plot_dss_flow_matrix(ax: plt.Axes, flow: pd.DataFrame) -> pd.DataFrame:
    outer = ax
    outer.set_axis_off()
    outer.set_title("Compartment communication rewiring", pad=4)
    stage_order = ["D0", "D3", "D9", "D21"]
    records = flow.copy()
    records[["sender", "receiver"]] = records["flow"].apply(lambda x: pd.Series(split_flow(x)))
    records = records[records["sender"].isin(COMPARTMENTS) & records["receiver"].isin(COMPARTMENTS)]
    max_val = records["stage_fraction"].quantile(0.98)
    for idx, stage in enumerate(stage_order):
        ax_i = outer.inset_axes([0.01 + idx * 0.247, 0.12, 0.225, 0.76])
        sub = records[records["stage_label"].eq(stage)]
        mat = (
            sub.pivot_table(index="sender", columns="receiver", values="stage_fraction", aggfunc="sum")
            .reindex(index=COMPARTMENTS, columns=COMPARTMENTS)
            .fillna(0)
        )
        draw_heatmap(ax_i, mat, NETWORK, vmin=0, vmax=max_val, annotate=False, edge="#FFFFFF", lw=0.38)
        if idx > 0:
            ax_i.set_yticklabels([])
        else:
            ax_i.set_yticklabels([COMPARTMENT_SHORT.get(x, x) for x in COMPARTMENTS])
        if idx < 3:
            ax_i.set_xticklabels([])
        else:
            ax_i.set_xticklabels([COMPARTMENT_SHORT.get(x, x) for x in COMPARTMENTS], rotation=35, ha="right")
        ax_i.set_title(stage, fontsize=6.3, pad=1.5)
    add_gradient_legend(outer, NETWORK, "flow", x=0.68, y=0.005)
    return records


def dss_receiver_burden(stage: str = "DSS9", replicate_tag: str = "rep1") -> pd.DataFrame:
    rep_dir = DSS_BASE / "multislice_replicates"
    pred_root = DSS_BASE / "multislice_spatialself_scale_v21/predictions"
    candidates = sorted(rep_dir.glob(f"mouse_colon_ibd_merfish_{stage}_{replicate_tag}_*/candidate_interactions.csv"))[0]
    sample_dir = candidates.parent
    sample_name = sample_dir.name
    pred = pred_root / stage / "spatialself_wavelet_scale" / f"{sample_name}__spatialself_wavelet_scale_predictions.csv"
    cells = pd.read_csv(sample_dir / "cells.csv")
    cand = pd.read_csv(candidates)
    scores = pd.read_csv(pred)
    merged = cand[["candidate_id", "receiver_id", "ligand", "receptor", "sender_cell_type", "receiver_cell_type"]].merge(scores, on="candidate_id", how="inner")
    threshold = merged["score"].quantile(0.80)
    top = merged[merged["score"].ge(threshold)]
    counts = top.groupby("receiver_id", as_index=False).agg(burden=("score", "size"), max_score=("score", "max"))
    out = cells.merge(counts.rename(columns={"receiver_id": "cell_id"}), on="cell_id", how="left")
    out["burden"] = out["burden"].fillna(0).astype(int)
    out["max_score"] = out["max_score"].fillna(0.0)
    out["x_plot"] = out["x"].astype(float)
    out["y_plot"] = -out["y"].astype(float)
    out["stage_label"] = "D9"
    return out


def plot_spatial_burden(ax: plt.Axes, frame: pd.DataFrame, title: str, value_col: str = "burden") -> None:
    ax.scatter(frame["x_plot"], frame["y_plot"], s=3.0, color="#D8D3CC", alpha=0.34, linewidths=0, rasterized=True)
    active = frame[frame[value_col].gt(0)].copy().sort_values(value_col)
    vals = active[value_col].astype(float)
    sizes = 7 + 22 * robust_norm(vals).to_numpy()
    vmax = max(float(vals.quantile(0.98)), 1.0)
    ax.scatter(
        active["x_plot"],
        active["y_plot"],
        s=sizes,
        c=vals,
        cmap=BURDEN,
        norm=Normalize(0, vmax),
        alpha=0.92,
        linewidths=0,
        rasterized=True,
    )
    spatial_style(ax, frame)
    ax.set_title(title, pad=4)
    cax = ax.inset_axes([0.66, -0.050, 0.28, 0.035])
    grad = np.linspace(0, 1, 128).reshape(1, -1)
    cax.imshow(grad, cmap=BURDEN, aspect="auto")
    cax.set_xticks([0, 127])
    cax.set_xticklabels(["low", "high"])
    cax.set_yticks([])
    cax.tick_params(length=0, labelsize=4.8, pad=0.6)
    for spine in cax.spines.values():
        spine.set_visible(False)


def plot_tonsil_network(ax: plt.Axes, table: pd.DataFrame) -> pd.DataFrame:
    dat = table[table["lr_pair"].eq("CXCL13-CXCR5")].copy()
    top_types = (
        pd.concat([dat["sender_cell_type"], dat["receiver_cell_type"]])
        .value_counts()
        .head(6)
        .index.tolist()
    )
    mat = (
        dat[dat["sender_cell_type"].isin(top_types) & dat["receiver_cell_type"].isin(top_types)]
        .pivot_table(index="sender_cell_type", columns="receiver_cell_type", values="n_edges", aggfunc="sum")
        .reindex(index=top_types, columns=top_types)
        .fillna(0)
    )
    mat.index = [pretty_label(x) for x in mat.index]
    mat.columns = [pretty_label(x) for x in mat.columns]
    draw_heatmap(ax, mat, NETWORK, vmin=0, vmax=max(float(mat.to_numpy().max()), 1.0), annotate=False, edge="#FFFFFF", lw=0.42)
    ax.set_title("Tonsil GC sender-receiver network", pad=4)
    ax.set_xlabel("receiver")
    ax.set_ylabel("sender")
    ax.tick_params(axis="x", labelrotation=35)
    add_gradient_legend(ax, NETWORK, "edges", x=0.72, y=1.015)
    return mat.reset_index().rename(columns={"sender_cell_type": "sender_cell_type"})


def plot_tonsil_burden(ax: plt.Axes, frame: pd.DataFrame) -> pd.DataFrame:
    plot = frame.copy()
    plot["burden"] = plot["incoming_norm"].astype(float)
    ax.scatter(plot["x_plot"], plot["y_plot"], s=3.0, color="#D8D3CC", alpha=0.35, linewidths=0, rasterized=True)
    active = plot[plot["burden"].gt(plot["burden"].quantile(0.70))].sort_values("burden")
    ax.scatter(
        active["x_plot"],
        active["y_plot"],
        s=7.5,
        c=active["burden"],
        cmap=BURDEN,
        norm=Normalize(0, max(float(active["burden"].quantile(0.99)), 1e-9)),
        linewidths=0,
        alpha=0.92,
        rasterized=True,
    )
    spatial_style(ax, plot)
    ax.set_title("CXCL13-CXCR5 burden", pad=4)
    return plot[["cell_id", "x_plot", "y_plot", "cell_type", "region", "incoming_score", "incoming_norm", "receiver_rank"]]


def plot_crc_pair_matrix(ax: plt.Axes, edges: pd.DataFrame) -> pd.DataFrame:
    sender_order = edges["sender_cell_type"].value_counts().head(7).index.tolist()
    receiver_order = edges["receiver_cell_type"].value_counts().head(7).index.tolist()
    mat = (
        edges[edges["sender_cell_type"].isin(sender_order) & edges["receiver_cell_type"].isin(receiver_order)]
        .pivot_table(index="sender_cell_type", columns="receiver_cell_type", values="score", aggfunc="count")
        .reindex(index=sender_order, columns=receiver_order)
        .fillna(0)
    )
    mat.index = [pretty_label(x) for x in mat.index]
    mat.columns = [pretty_label(x) for x in mat.columns]
    draw_heatmap(ax, mat, NETWORK, vmin=0, vmax=max(float(mat.to_numpy().max()), 1.0), annotate=False, edge="#FFFFFF", lw=0.42)
    ax.set_title("CRC TGFB1-TGFBR2 cell-type pairs", pad=4)
    ax.set_xlabel("receiver")
    ax.set_ylabel("sender")
    ax.tick_params(axis="x", labelrotation=35)
    add_gradient_legend(ax, NETWORK, "edges", x=0.72, y=1.015)
    return mat.reset_index().rename(columns={"sender_cell_type": "sender_cell_type"})


def plot_cosmx_receiver_matrix(ax: plt.Axes, hotspots: pd.DataFrame) -> pd.DataFrame:
    selected_lrs = ["CXCL12-CXCR4", "CXCL10-CXCR3", "TGFB1-TGFBR2", "VEGFA-KDR", "SPP1-CD44", "CSF1-CSF1R"]
    dat = hotspots[hotspots["LR"].isin(selected_lrs)].copy()
    dat["activity_rank"] = dat.groupby("LR")["receiver_activity"].rank(pct=True)
    top = dat[dat["activity_rank"].ge(0.95)]
    ct_order = top["cell_type"].value_counts().head(7).index.tolist()
    mat = (
        top[top["cell_type"].isin(ct_order)]
        .pivot_table(index="LR", columns="cell_type", values="cell_id", aggfunc="count")
        .reindex(index=selected_lrs, columns=ct_order)
        .fillna(0)
    )
    row_sum = mat.sum(axis=1).replace(0, np.nan)
    mat_frac = mat.div(row_sum, axis=0).fillna(0)
    mat_frac.columns = [pretty_label(x) for x in mat_frac.columns]
    draw_heatmap(ax, mat_frac, NETWORK, vmin=0, vmax=max(float(mat_frac.to_numpy().max()), 1e-9), annotate=False, edge="#FFFFFF", lw=0.42)
    ax.set_title("CosMx receiver cell-type burden", pad=4)
    ax.set_xlabel("receiver cell type")
    ax.set_ylabel("LR axis")
    ax.tick_params(axis="x", labelrotation=35)
    add_gradient_legend(ax, NETWORK, "fraction", x=0.72, y=1.015)
    return mat_frac.reset_index()


def plot_real_network_burden_additions() -> None:
    flow = pd.read_csv(DSS_BASE / "progression_story/dominant_compartment_flows_by_stage.csv")
    dss_burden = dss_receiver_burden()
    tonsil_network = pd.read_csv(TONSIL / "source_sender_receiver_summary.csv")
    tonsil_hotspot = pd.read_csv(TONSIL / "panel_c_scalecomm_receiver_hotspot.csv")
    crc_edges = pd.read_csv(CRC_ARTICLE / "top_edges.csv")
    cosmx_hotspots = pd.read_csv(COSMX / "cosmx_hotspot_receiver_cells.csv")

    fig = plt.figure(figsize=(7.2, 7.45))
    gs = fig.add_gridspec(3, 2, left=0.065, right=0.985, top=0.965, bottom=0.075, hspace=0.60, wspace=0.42)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]

    source_tables = {
        "dss_compartment_flow": plot_dss_flow_matrix(axes[0], flow),
        "dss_d9_receiver_burden": dss_burden,
        "tonsil_cxcl13_cxcr5_network": plot_tonsil_network(axes[2], tonsil_network),
        "tonsil_cxcl13_cxcr5_receiver_burden": plot_tonsil_burden(axes[3], tonsil_hotspot),
        "crc_tgfb1_tgfbr2_celltype_pairs": plot_crc_pair_matrix(axes[4], crc_edges),
        "cosmx_receiver_celltype_burden": plot_cosmx_receiver_matrix(axes[5], cosmx_hotspots),
    }
    plot_spatial_burden(axes[1], dss_burden, "DSS9 per-location interaction burden")

    for label, ax in zip("abcdef", axes, strict=False):
        panel_label(fig, ax, label)
    for name, table in source_tables.items():
        table.to_csv(SOURCE / f"realdata_{name}.csv", index=False)

    base = OUT / "article_reference_realdata_network_burden"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def update_audit_table() -> None:
    if not AUDIT.exists():
        return
    audit = pd.read_csv(AUDIT)
    module = []
    status = []
    for fam in audit["reference_family"]:
        if fam == "benchmark AUC/AUPRC/parameter sensitivity":
            module.append("article_reference_benchmark_high_priority")
            status.append("implemented_now")
        elif fam == "error mode benchmark":
            module.append("article_reference_benchmark_high_priority")
            status.append("implemented_now")
        elif fam == "cell-type/region interaction network":
            module.append("article_reference_realdata_network_burden")
            status.append("implemented_now")
        elif fam == "per-location interaction burden":
            module.append("article_reference_realdata_network_burden")
            status.append("implemented_now")
        elif fam in {"spatial LR score expression strip", "sender/receiver layered map", "downstream GO/target functional validation", "differential/time-course signaling"}:
            module.append("previous_article_style_additions")
            status.append("implemented_before_or_story_specific")
        elif fam in {"orthogonal PLA benchmark/validation", "temporal dynamics / velocity"}:
            module.append("")
            status.append("not_implemented_no_orthogonal_data")
        else:
            module.append("")
            status.append("not_prioritized_or_presentation_only")
    audit["implemented_status_2026_07_02"] = status
    audit["implemented_module_2026_07_02"] = module
    audit.to_csv(OUT / "article_figures_75panel_audit_implemented.csv", index=False)


def write_readme() -> None:
    text = """# High-priority article-reference additions

These files are a systematic follow-up to the 75-panel `article_figures/` audit.
They add the high-value reference families that were not fully covered by the
earlier DSS/CRC/CosMx/tonsil article-style panels.

## Files

- `article_reference_benchmark_high_priority.pdf`
  - Overall semi-synthetic benchmark summary.
  - LR range-class AUPR grid.
  - Low/high-noise robustness.
  - Error-mode rates.
  - Epoch sensitivity.
  - Scale/context ablation.

- `article_reference_realdata_network_burden.pdf`
  - DSS compartment communication rewiring.
  - DSS9 per-location interaction burden.
  - Tonsil CXCL13-CXCR5 sender-receiver network.
  - Tonsil CXCL13-CXCR5 receiver burden.
  - CRC TGFB1-TGFBR2 cell-type pair network.
  - CosMx receiver cell-type burden.

## Caveats

- PLA/protein-image validation panels from the reference article are not
  implemented because the project does not currently contain matched PLA/protein
  ground-truth data.
- RNA-velocity/LR-velocity panels are not implemented because the current DSS
  time-course analysis does not include velocity estimates.
- These additions are intended as Extended Data/Supplement candidates or as
  modules to strengthen manuscript Fig2-Fig5, not as replacements for the
  current main figures.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    setup_style()
    plot_benchmark_additions()
    plot_real_network_burden_additions()
    update_audit_table()
    write_readme()


if __name__ == "__main__":
    main()
