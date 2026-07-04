#!/usr/bin/env python
"""Render Phase 5 real-validation manuscript figures."""

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

BLUE = "#3B6EA8"
GREEN = "#5A8F6B"
GOLD = "#C49A55"
RED = "#B86B5E"
GRAY = "#7C7C7C"
LIGHT = "#E9ECEF"
INK = "#202124"

DATASET_LABELS = {
    "lymph_node": "Lymph node",
    "tonsil": "Tonsil",
    "melanoma": "Melanoma",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/phase5/manuscript_submission_lock")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _set_style()

    data = _load_tables(output_dir)
    specs = [
        (
            "Fig7",
            "Lymph node validation",
            plot_fig7,
            "Phase 5 lymph-node control ranking and downstream validation",
            "Lymph-node positives rank above high-expression negatives and show higher receiver-response evidence.",
            "Real-background semi-synthetic controls and key lymph-node downstream validation do not constitute complete native ground truth.",
        ),
        (
            "Fig8",
            "Tonsil and melanoma validation",
            plot_fig8,
            "Phase 5 cross-tissue positive/negative control validation",
            "Tonsil and melanoma panels preserve positive/negative control separation with low high-expression FP rates.",
            "Tonsil and melanoma currently use receiver-response proxy rather than curated downstream target validation.",
        ),
    ]

    qa_rows = []
    source_rows = []
    for fig_id, title, func, source_label, claim, limitation in specs:
        fig = func(data)
        out_base = fig_dir / f"{fig_id.lower()}_phase5_real_validation"
        _save_pub(fig, out_base)
        plt.close(fig)
        qa_rows.append(
            {
                "figure_id": fig_id,
                "pdf": _rel(out_base.with_suffix(".pdf")),
                "svg": _rel(out_base.with_suffix(".svg")),
                "tiff": _rel(out_base.with_suffix(".tiff")),
                "status": "rendered",
                "qa_note": "phase5 real-validation manuscript figure; source-linked; final journal layout polish may still be needed",
            }
        )
        source_rows.append(
            {
                "figure_id": fig_id,
                "title": title,
                "source_label": source_label,
                "source_tables": ";".join(
                    [
                        _rel(output_dir / "goal2_real_validation/phase5_real_validation_tissue_summary.csv"),
                        _rel(output_dir / "goal2_real_validation/phase5_real_validation_method_summary.csv"),
                        _rel(output_dir / "goal2_real_validation/phase5_real_validation_lr_panel.csv"),
                        _rel(output_dir / "goal2_real_validation/phase5_real_validation_claims.csv"),
                    ]
                ),
                "supported_claim": claim,
                "limitation": limitation,
                "pdf": _rel(out_base.with_suffix(".pdf")),
                "svg": _rel(out_base.with_suffix(".svg")),
                "tiff": _rel(out_base.with_suffix(".tiff")),
            }
        )

    _write_csv(output_dir / "phase5_real_validation_figure_qa.csv", qa_rows, qa_rows[0].keys())
    _write_csv(output_dir / "phase5_real_validation_figure_source_index.csv", source_rows, source_rows[0].keys())
    summary = {
        "n_rendered_figures": len(qa_rows),
        "qa_csv": _rel(output_dir / "phase5_real_validation_figure_qa.csv"),
        "source_index_csv": _rel(output_dir / "phase5_real_validation_figure_source_index.csv"),
        "claim_level": "phase5_real_validation_figures_for_manuscript_draft",
    }
    (output_dir / "phase5_real_validation_figure_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def plot_fig7(data: dict[str, list[dict[str, str]]]) -> plt.Figure:
    tissue = _row_by(data["tissue"], "dataset", "lymph_node")
    lr_rows = [row for row in data["lr_panel"] if row.get("dataset") == "lymph_node"]
    method_rows = [row for row in data["methods"] if row.get("dataset") == "lymph_node"]

    fig = plt.figure(figsize=(7.2, 4.2))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], width_ratios=[1.0, 1.0, 1.25])
    ax_rank = fig.add_subplot(grid[0, 0])
    ax_fp = fig.add_subplot(grid[0, 1])
    ax_delta = fig.add_subplot(grid[0, 2])
    ax_lr = fig.add_subplot(grid[1, :])

    ax_rank.bar(
        ["positive", "negative"],
        [_float(tissue.get("spatialself_mean_positive_rank")), _float(tissue.get("spatialself_mean_negative_rank"))],
        color=[GREEN, GRAY],
        width=0.55,
    )
    ax_rank.invert_yaxis()
    ax_rank.set_ylabel("mean LR rank")
    ax_rank.set_title("A  Control ranks", loc="left", fontsize=9, weight="bold")
    _annotate_bars(ax_rank, fmt="{:.1f}", offset=0.04)

    ax_fp.bar(["SpatialSelf", "best baseline"], [_float(tissue.get("spatialself_high_expression_fp_rate_q95")), _float(tissue.get("baseline_min_high_expression_fp_rate_q95"))], color=[BLUE, GRAY], width=0.55)
    ax_fp.set_ylabel("high-expression FP rate")
    ax_fp.set_title("B  FP suppression", loc="left", fontsize=9, weight="bold")
    _annotate_bars(ax_fp, fmt="{:.3g}")

    labels = [_method_short(row["method"]) for row in method_rows]
    values = [_float(row.get("receiver_response_proxy_delta")) for row in method_rows]
    colors = [BLUE if row.get("method") == "spatialself_wavelet_edge_prior" else GRAY for row in method_rows]
    ax_delta.bar(labels, values, color=colors, width=0.62)
    ax_delta.axhline(0, color=INK, lw=0.7)
    ax_delta.set_ylabel("positive - negative proxy")
    ax_delta.set_title("C  Receiver-response proxy", loc="left", fontsize=9, weight="bold")
    ax_delta.tick_params(axis="x", rotation=25)

    _draw_lr_rank_panel(ax_lr, lr_rows, "D  Lymph-node LR control panel")
    _caption(
        fig,
        "Lymph-node validation: positives rank above high-expression negatives; downstream/proxy evidence is supportive, not native complete ground truth.",
    )
    return fig


def plot_fig8(data: dict[str, list[dict[str, str]]]) -> plt.Figure:
    tissue_rows = [row for row in data["tissue"] if row.get("dataset") in {"tonsil", "melanoma"}]
    method_rows = [row for row in data["methods"] if row.get("dataset") in {"tonsil", "melanoma"} and row.get("method") == "spatialself_wavelet_edge_prior"]
    lr_rows = [row for row in data["lr_panel"] if row.get("dataset") in {"tonsil", "melanoma"}]

    fig = plt.figure(figsize=(7.2, 4.4))
    grid = fig.add_gridspec(2, 3, height_ratios=[0.95, 1.15], width_ratios=[1.0, 1.0, 1.2])
    ax_rank = fig.add_subplot(grid[0, 0])
    ax_top = fig.add_subplot(grid[0, 1])
    ax_fp = fig.add_subplot(grid[0, 2])
    ax_lr = fig.add_subplot(grid[1, :])

    x = np.arange(len(tissue_rows))
    width = 0.34
    ax_rank.bar(x - width / 2, [_float(row.get("spatialself_mean_positive_rank")) for row in tissue_rows], width=width, color=GREEN, label="positive")
    ax_rank.bar(x + width / 2, [_float(row.get("spatialself_mean_negative_rank")) for row in tissue_rows], width=width, color=GRAY, label="negative")
    ax_rank.set_xticks(x)
    ax_rank.set_xticklabels([DATASET_LABELS.get(row["dataset"], row["dataset"]) for row in tissue_rows])
    ax_rank.invert_yaxis()
    ax_rank.set_ylabel("mean LR rank")
    ax_rank.set_title("A  Control rank separation", loc="left", fontsize=9, weight="bold")
    ax_rank.legend(loc="upper right", fontsize=6.5)

    ax_top.bar(x - width / 2, [_float(row.get("spatialself_positive_top3_count")) for row in tissue_rows], width=width, color=GREEN, label="positive")
    ax_top.bar(x + width / 2, [_float(row.get("spatialself_negative_top3_count")) for row in tissue_rows], width=width, color=GRAY, label="negative")
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([DATASET_LABELS.get(row["dataset"], row["dataset"]) for row in tissue_rows])
    ax_top.set_ylabel("top-three count")
    ax_top.set_title("B  Top-ranked controls", loc="left", fontsize=9, weight="bold")
    ax_top.set_ylim(0, 3.4)

    labels = [DATASET_LABELS.get(row["dataset"], row["dataset"]) for row in tissue_rows]
    fp_values = [_float(row.get("spatialself_high_expression_fp_rate_q95")) for row in tissue_rows]
    proxy_values = [_float(row.get("spatialself_proxy_spearman_delta")) for row in tissue_rows]
    ax_fp.bar(labels, fp_values, color=BLUE, width=0.55)
    ax_fp.set_ylabel("FP rate at q95")
    ax_fp.set_title("C  Low high-expression FP", loc="left", fontsize=9, weight="bold")
    _annotate_bars(ax_fp, fmt="{:.3g}")
    ax2 = ax_fp.twinx()
    ax2.plot(labels, proxy_values, marker="o", color=GOLD, lw=1.4)
    ax2.set_ylabel("proxy delta", color=GOLD)
    ax2.tick_params(axis="y", colors=GOLD)
    ax2.spines["top"].set_visible(False)

    _draw_lr_rank_panel(ax_lr, lr_rows, "D  Tonsil and melanoma LR controls")
    _caption(
        fig,
        "Tonsil/melanoma validation: positive controls separate from high-expression negatives; proxy evidence is not curated downstream validation.",
    )
    return fig


def _draw_lr_rank_panel(ax: plt.Axes, rows: list[dict[str, str]], title: str) -> None:
    rows = sorted(rows, key=lambda row: (row.get("dataset", ""), _float(row.get("global_rank"))))
    labels = [f"{DATASET_LABELS.get(row['dataset'], row['dataset'])}\n{row['ligand']}-{row['receptor']}" for row in rows]
    ranks = [_float(row.get("global_rank")) for row in rows]
    colors = [GREEN if _truthy(row.get("positive_control")) else RED for row in rows]
    ax.bar(np.arange(len(rows)), ranks, color=colors, width=0.72)
    ax.invert_yaxis()
    ax.set_ylabel("global LR rank")
    ax.set_title(title, loc="left", fontsize=9, weight="bold")
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6.5)
    ax.axhline(3.5, color="#AAB2BD", lw=0.8, ls="--")
    ax.text(len(rows) - 0.2, 3.55, "top-3 boundary", ha="right", va="bottom", fontsize=6, color="#60656F")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=GREEN, label="positive"),
        plt.Rectangle((0, 0), 1, 1, color=RED, label="high-expression negative"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=6.5)


def _load_tables(output_dir: Path) -> dict[str, list[dict[str, str]]]:
    base = output_dir / "goal2_real_validation"
    return {
        "tissue": _read_csv(base / "phase5_real_validation_tissue_summary.csv"),
        "methods": _read_csv(base / "phase5_real_validation_method_summary.csv"),
        "lr_panel": _read_csv(base / "phase5_real_validation_lr_panel.csv"),
        "claims": _read_csv(base / "phase5_real_validation_claims.csv"),
    }


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
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
        }
    )


def _save_pub(fig: plt.Figure, out_base: Path) -> None:
    fig.subplots_adjust(left=0.08, right=0.94, top=0.90, bottom=0.26, wspace=0.45, hspace=0.62)
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def _caption(fig: plt.Figure, text: str) -> None:
    fig.text(0.01, 0.015, text, ha="left", va="bottom", fontsize=6.5, color="#4A4A4A")


def _annotate_bars(ax: plt.Axes, fmt: str = "{:.2g}", offset: float = 0.02) -> None:
    ymin, ymax = ax.get_ylim()
    span = abs(ymax - ymin) if ymax != ymin else 1
    for patch in ax.patches:
        height = patch.get_height()
        if not np.isfinite(height):
            continue
        y = height + span * offset if ymax > ymin else height - span * offset
        ax.text(patch.get_x() + patch.get_width() / 2, y, fmt.format(height), ha="center", va="bottom", fontsize=6.5)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: Any) -> None:
    columns = list(columns)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _row_by(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    return {}


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "1.0", "yes", "y", "supported"}


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _method_short(method: str) -> str:
    return {
        "spatialself_wavelet_edge_prior": "SpatialSelf",
        "spatialself_wavelet": "SS base",
        "commot": "COMMOT",
        "cellchat": "CellChat",
        "spatalk": "SpaTalk",
        "cellnest": "CellNEST",
    }.get(method, method)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
