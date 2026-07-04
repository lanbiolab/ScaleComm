#!/usr/bin/env python
"""Generate first-pass Phase 3 manuscript scaffold figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]

METHOD_ORDER = ["spatialself_wavelet_edge_prior", "commot", "cellchat", "spatalk", "cellnest"]
METHOD_LABELS = {
    "spatialself_wavelet_edge_prior": "SpatialSelf",
    "commot": "COMMOT",
    "cellchat": "CellChat",
    "spatalk": "SpaTalk",
    "cellnest": "CellNEST",
}
PALETTE = {
    "spatialself_wavelet_edge_prior": "#3B6EA8",
    "commot": "#6B8E7A",
    "cellchat": "#B07AA1",
    "spatalk": "#C49A55",
    "cellnest": "#7C7C7C",
    "complete": "#5A8F6B",
    "missing": "#B86B5E",
    "neutral": "#E9ECEF",
    "ink": "#202124",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/phase3/end_to_end")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _set_style()

    data = _load_phase3_tables(output_dir)
    figure_specs = [
        ("Fig1", plot_fig1),
        ("Fig2", plot_fig2),
        ("Fig3", plot_fig3),
        ("Fig4", plot_fig4),
        ("Fig5", plot_fig5),
        ("Fig6", plot_fig6),
        ("Fig7", plot_fig7),
        ("Fig8", plot_fig8),
    ]
    qa_rows = []
    for figure_id, func in figure_specs:
        fig = func(data)
        out_base = fig_dir / f"{figure_id.lower()}_phase3_scaffold"
        _save_pub(fig, out_base)
        plt.close(fig)
        qa_rows.append(
            {
                "figure_id": figure_id,
                "pdf": _rel(out_base.with_suffix(".pdf")),
                "svg": _rel(out_base.with_suffix(".svg")),
                "tiff": _rel(out_base.with_suffix(".tiff")),
                "status": "rendered",
                "qa_note": "first-pass scaffold; source-linked; not final polished manuscript art",
            }
        )
    qa_path = output_dir / "phase3_figure_render_qa.csv"
    _write_csv(qa_path, qa_rows, ["figure_id", "pdf", "svg", "tiff", "status", "qa_note"])
    summary = {"n_rendered_figures": len(qa_rows), "qa_csv": _rel(qa_path)}
    (output_dir / "phase3_figure_render_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def plot_fig1(data: dict[str, list[dict[str, str]]]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    ax.axis("off")
    steps = ["data", "predictions", "metrics", "FDR", "bio validation", "source tables"]
    y = np.linspace(0.85, 0.15, len(steps))
    for idx, (step, yy) in enumerate(zip(steps, y)):
        ax.text(0.08, yy, step, ha="left", va="center", fontsize=8, weight="bold", color=PALETTE["ink"])
        ax.scatter([0.02], [yy], s=80, color="#DDE8F2", edgecolor=PALETTE["spatialself_wavelet_edge_prior"], linewidth=1)
        if idx < len(steps) - 1:
            ax.annotate("", xy=(0.02, y[idx + 1] + 0.035), xytext=(0.02, yy - 0.035), arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": "#6A6A6A"})
    ax.set_title("Phase 3 pipeline", loc="left", fontsize=9, weight="bold")
    manifest = data["manifest"]
    labels = ["required ready", "optional placeholders"]
    values = [
        sum(row["dataset_status"] == "ready" and _truthy(row["required"]) for row in manifest),
        sum(not _truthy(row["required"]) for row in manifest),
    ]
    axes[1].bar(labels, values, color=["#5A8F6B", "#B8C1CC"], width=0.55)
    axes[1].set_ylabel("datasets")
    axes[1].set_title("Unified data entry", loc="left", fontsize=9, weight="bold")
    _annotate_bars(axes[1])
    _caption(fig, "Claim: Phase 3 has a data-to-source-table scaffold. Limitation: this is a wiring figure, not a performance claim.")
    return fig


def plot_fig2(data: dict[str, list[dict[str, str]]]) -> plt.Figure:
    rows = [row for row in data["metrics"] if row["dataset_id"] == "controlled_simulator_v2" and row["method"] in METHOD_ORDER]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=False)
    for ax, metric, title in zip(axes, ["auroc", "aupr", "high_expression_false_positive_rate"], ["AUROC", "AUPR", "High-expression FP"]):
        _method_bar(ax, rows, metric)
        ax.set_title(title, loc="left", fontsize=9, weight="bold")
    _caption(fig, "Claim: controlled simulator v2 can report truth-level metrics. Limitation: Phase 3 freezes replicate_00 only.")
    return fig


def plot_fig3(data: dict[str, list[dict[str, str]]]) -> plt.Figure:
    rows = [row for row in data["metrics"] if row["dataset_id"].startswith("real_background_") and row["method"] == "spatialself_wavelet_edge_prior"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))
    datasets = [row["dataset_id"].replace("real_background_", "") for row in rows]
    for ax, metric, title in zip(axes, ["auroc", "aupr", "weak_signal_recall"], ["AUROC", "AUPR", "Weak recall"]):
        values = [_float(row.get(metric)) for row in rows]
        ax.bar(datasets, values, color="#3B6EA8", width=0.6)
        ax.set_ylim(0, max(1.0, max(values or [0]) * 1.15))
        ax.set_title(title, loc="left", fontsize=9, weight="bold")
        ax.tick_params(axis="x", labelrotation=25)
        _annotate_bars(ax)
    _caption(fig, "Claim: real-background semi-synthetic datasets are wired into unified truth/control metrics. Limitation: truth is injected.")
    return fig


def plot_fig4(data: dict[str, list[dict[str, str]]]) -> plt.Figure:
    rows = [row for row in data["metrics"] if row["dataset_id"].startswith("real_background_") and row["method"] in METHOD_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    _grouped_metric(axes[0], rows, "aupr", "AUPR across real-background panel")
    _grouped_metric(axes[1], rows, "high_expression_false_positive_rate", "High-expression FP rate")
    _caption(fig, "Claim: retained baselines can be compared in one schema. Limitation: CellChat/SpaTalk projection and CellNEST adapter-light caveats remain.")
    return fig


def plot_fig5(data: dict[str, list[dict[str, str]]]) -> plt.Figure:
    rows = [row for row in data["metrics"] if row["dataset_id"].startswith("real_background_") and row["method"] in METHOD_ORDER]
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for method in METHOD_ORDER:
        method_rows = [row for row in rows if row["method"] == method]
        x = np.mean([_float(row["high_expression_false_positive_rate"]) for row in method_rows])
        y = np.mean([_float(row["weak_signal_recall"]) for row in method_rows])
        ax.scatter(x, y, s=55, color=PALETTE[method], label=METHOD_LABELS[method], edgecolor="white", linewidth=0.7)
        ax.text(x + 0.01, y, METHOD_LABELS[method], fontsize=7, va="center")
    ax.set_xlabel("mean high-expression FP rate")
    ax.set_ylabel("mean weak-signal recall")
    ax.set_title("Weak signal versus FP tradeoff", loc="left", fontsize=9, weight="bold")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    _caption(fig, "Claim: the weak-signal/FP tradeoff is measurable. Limitation: not a universal highest-recall claim.")
    return fig


def plot_fig6(data: dict[str, list[dict[str, str]]]) -> plt.Figure:
    rows = [row for row in data["fdr"] if row["status"] == "complete"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    nulls = ["spatial_permutation", "receiver_permutation", "lr_identity_permutation"]
    counts = [sum(row["null_model"] == null for row in rows) for null in nulls]
    axes[0].bar([_short_null(n) for n in nulls], counts, color="#5A8F6B")
    axes[0].set_ylabel("complete rows")
    axes[0].set_title("100-perm FDR coverage", loc="left", fontsize=9, weight="bold")
    _annotate_bars(axes[0])
    lr_min = []
    labels = []
    for row in rows:
        if row["null_model"] == "lr_identity_permutation":
            labels.append(row["dataset_id"].replace("real_background_", ""))
            lr_min.append(_float(row["min_lr_pair_bh_fdr"]))
    axes[1].bar(labels, lr_min, color="#3B6EA8")
    axes[1].axhline(0.05, color="#8B2E2E", lw=1, ls="--")
    axes[1].set_ylabel("minimum LR-pair BH FDR")
    axes[1].set_title("LR identity null", loc="left", fontsize=9, weight="bold")
    axes[1].tick_params(axis="x", labelrotation=25)
    _caption(fig, "Claim: permutation/FDR outputs are attached for real-background SpatialSelf. Limitation: final 200-500 permutation panel pending.")
    return fig


def plot_fig7(data: dict[str, list[dict[str, str]]]) -> plt.Figure:
    rows = [row for row in data["bio"] if row["method"] == "spatialself_wavelet_edge_prior"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    labels = [row["dataset_id"].replace("real_background_", "").replace("_subset", "") for row in rows]
    pos = [_float(row["best_positive_rank"]) for row in rows]
    neg = [_float(row["best_negative_rank"]) for row in rows]
    x = np.arange(len(rows))
    axes[0].bar(x - 0.18, pos, width=0.36, color="#3B6EA8", label="best positive")
    axes[0].bar(x + 0.18, neg, width=0.36, color="#B8C1CC", label="best negative")
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].set_ylabel("rank (lower is better)")
    axes[0].set_title("Control ranking", loc="left", fontsize=9, weight="bold")
    axes[0].legend()
    ds_rows = [row for row in rows if row["mean_positive_downstream_spearman"]]
    axes[1].bar(
        [row["dataset_id"].replace("real_background_", "") for row in ds_rows],
        [_float(row["mean_positive_downstream_spearman"]) for row in ds_rows],
        color="#6B8E7A",
    )
    axes[1].set_ylabel("mean positive downstream Spearman")
    axes[1].set_title("Curated downstream hook", loc="left", fontsize=9, weight="bold")
    axes[1].tick_params(axis="x", labelrotation=25)
    _caption(fig, "Claim: biological validation is indexed across real contexts. Limitation: framework-level validation with limited curated targets.")
    return fig


def plot_fig8(data: dict[str, list[dict[str, str]]]) -> plt.Figure:
    rows = [row for row in data["predictions"] if row["dataset_id"] == "high_resolution_tonsil_subset" and row["method"] in METHOD_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    complete = [row for row in rows if row["status"] == "completed"]
    axes[0].bar([METHOD_LABELS[row["method"]] for row in complete], [_float(row["runtime_seconds"]) for row in complete], color=[PALETTE[row["method"]] for row in complete])
    axes[0].set_ylabel("seconds")
    axes[0].set_title("High-resolution subset runtime", loc="left", fontsize=9, weight="bold")
    axes[0].tick_params(axis="x", labelrotation=25)
    pred_rows = [_float(row["n_prediction_rows"]) for row in complete]
    axes[1].bar([METHOD_LABELS[row["method"]] for row in complete], pred_rows, color="#B8C1CC")
    axes[1].set_ylabel("prediction rows")
    axes[1].set_title("Candidate-scale output", loc="left", fontsize=9, weight="bold")
    axes[1].tick_params(axis="x", labelrotation=25)
    _caption(fig, "Claim: runtime/scalability source data are wired. Limitation: high-resolution CellNEST and full-platform scale are pending.")
    return fig


def _load_phase3_tables(output_dir: Path) -> dict[str, list[dict[str, str]]]:
    return {
        "manifest": _read_csv(output_dir / "phase3_dataset_manifest.csv"),
        "predictions": _read_csv(output_dir / "phase3_prediction_index.csv"),
        "metrics": _read_csv(output_dir / "phase3_metric_index.csv"),
        "fdr": _read_csv(output_dir / "phase3_fdr_index.csv"),
        "bio": _read_csv(output_dir / "phase3_biological_validation_index.csv"),
    }


def _method_bar(ax: plt.Axes, rows: list[dict[str, str]], metric: str) -> None:
    ordered = [next((row for row in rows if row["method"] == method), None) for method in METHOD_ORDER]
    ordered = [row for row in ordered if row is not None]
    ax.bar([METHOD_LABELS[row["method"]] for row in ordered], [_float(row[metric]) for row in ordered], color=[PALETTE[row["method"]] for row in ordered])
    ax.tick_params(axis="x", labelrotation=30)
    if metric != "high_expression_false_positive_rate":
        ax.set_ylim(0, 1)
    else:
        ax.set_ylim(0, max(1.0, max(_float(row[metric]) for row in ordered) * 1.15))


def _grouped_metric(ax: plt.Axes, rows: list[dict[str, str]], metric: str, title: str) -> None:
    datasets = sorted({row["dataset_id"].replace("real_background_", "") for row in rows})
    x = np.arange(len(datasets))
    width = 0.14
    for idx, method in enumerate(METHOD_ORDER):
        vals = []
        for dataset in datasets:
            match = next((row for row in rows if row["dataset_id"].replace("real_background_", "") == dataset and row["method"] == method), None)
            vals.append(_float(match.get(metric, "")) if match else np.nan)
        ax.bar(x + (idx - 2) * width, vals, width=width, color=PALETTE[method], label=METHOD_LABELS[method])
    ax.set_xticks(x, datasets, rotation=25, ha="right")
    ax.set_title(title, loc="left", fontsize=9, weight="bold")
    ax.set_ylim(0, 1)
    if metric == "aupr":
        ax.set_ylabel("AUPR")
    else:
        ax.set_ylabel("rate")
    ax.legend(fontsize=6, ncol=2)


def _caption(fig: plt.Figure, text: str) -> None:
    fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=6.2, color="#4A4A4A")
    fig.tight_layout(rect=[0, 0.08, 1, 1])


def _annotate_bars(ax: plt.Axes) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        ax.text(patch.get_x() + patch.get_width() / 2, height, f"{height:g}", ha="center", va="bottom", fontsize=6)


def _short_null(null_model: str) -> str:
    return null_model.replace("_permutation", "").replace("_", "\n")


def _float(value: Any) -> float:
    try:
        if value is None or str(value) == "":
            return np.nan
        return float(value)
    except ValueError:
        return np.nan


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "1.0", "yes", "y"}


def _set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save_pub(fig: plt.Figure, out_base: Path) -> None:
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
