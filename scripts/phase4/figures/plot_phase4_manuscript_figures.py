#!/usr/bin/env python
"""Render Phase 4 manuscript evidence-lock figures."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/phase4/manuscript_evidence_lock")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _set_style()

    data = _load_tables(output_dir)
    specs = [
        ("Fig1", "Phase 4 evidence lock", plot_fig1, "Method and evidence overview", "Phase 4 locks benchmark, baseline, FDR, downstream, and ablation evidence.", "Figures are evidence summaries, not final graphical design."),
        ("Fig2", "Formal benchmarks", plot_fig2, "Controlled and real-background benchmark gates", "Benchmarks now meet five-replicate gates and mechanism coverage.", "Benchmark truth is injected; not unlabeled real tissue ground truth."),
        ("Fig3", "Baseline comparison", plot_fig3, "Replicated baseline comparison", "Replicated baseline panel supports metric-specific tradeoff claims.", "Does not support global dominance across every metric."),
        ("Fig4", "Weak signal and false positives", plot_fig4, "Weak-signal/FP tradeoff guardrail", "Weak-signal claims are explicitly tradeoff-gated.", "Raw weak recall is not claimed to be globally highest."),
        ("Fig5", "Ablation", plot_fig5, "Component contribution", "Full SpatialSelf outperforms registered component ablations on controlled five-replicate AUPR.", "This is a 10-epoch formal coverage panel."),
        ("Fig6", "FDR", plot_fig6, "500-permutation FDR key panel", "A 500-permutation key lymph-node panel supports group-level FDR.", "Edge-level discovery power remains limited."),
        ("Fig7", "Biological validation", plot_fig7, "Controls and downstream validation", "Positive controls rank above negatives and key downstream positives exceed negative controls on mean target correlation.", "Cross-tissue downstream validation remains partial."),
        ("Fig8", "Runtime and placement", plot_fig8, "Scalability evidence placement", "Runtime evidence supports scalability wiring.", "Fig8 should remain supplement unless larger 50k/100k runs are generated."),
    ]

    qa_rows = []
    source_rows = []
    for fig_id, title, func, source_label, claim, limitation in specs:
        fig = func(data)
        out_base = fig_dir / f"{fig_id.lower()}_phase4_manuscript"
        _save_pub(fig, out_base)
        plt.close(fig)
        qa_rows.append(
            {
                "figure_id": fig_id,
                "pdf": _rel(out_base.with_suffix(".pdf")),
                "svg": _rel(out_base.with_suffix(".svg")),
                "tiff": _rel(out_base.with_suffix(".tiff")),
                "status": "rendered",
                "qa_note": "phase4 evidence-lock figure; source-linked; final visual polish may still be needed",
            }
        )
        source_rows.append(
            {
                "figure_id": fig_id,
                "title": title,
                "source_label": source_label,
                "supported_claim": claim,
                "limitation": limitation,
                "pdf": _rel(out_base.with_suffix(".pdf")),
                "svg": _rel(out_base.with_suffix(".svg")),
                "tiff": _rel(out_base.with_suffix(".tiff")),
            }
        )

    _write_csv(output_dir / "phase4_figure_render_qa.csv", qa_rows, qa_rows[0].keys())
    _write_csv(output_dir / "phase4_figure_source_index.csv", source_rows, source_rows[0].keys())
    summary = {
        "n_rendered_figures": len(qa_rows),
        "qa_csv": _rel(output_dir / "phase4_figure_render_qa.csv"),
        "source_index_csv": _rel(output_dir / "phase4_figure_source_index.csv"),
        "claim_level": "phase4_evidence_lock_figures_not_final_visual_design",
    }
    (output_dir / "phase4_figure_render_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def plot_fig1(data: dict[str, list[dict[str, str]]]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.1, 1.0]})
    ax = axes[0]
    ax.axis("off")
    steps = ["benchmark", "baselines", "weak signal", "FDR", "downstream", "ablation", "figures"]
    y = np.linspace(0.88, 0.14, len(steps))
    rows = data["evidence"]
    status_by_goal = {row["goal_id"]: row["status"] for row in rows}
    goal_ids = [
        "goal1_formalize_benchmarks",
        "goal2_strengthen_baselines",
        "goal4_weak_signal_evidence",
        "goal3_upgrade_fdr",
        "goal5_real_validation",
        "goal6_ablation_lock",
        "goal7_manuscript_figures",
    ]
    for idx, (step, yy, goal_id) in enumerate(zip(steps, y, goal_ids)):
        complete = any(row["goal_id"] == goal_id and row["status"] == "complete" for row in rows)
        color = GREEN if complete else GOLD
        ax.scatter([0.05], [yy], s=110, color=color, edgecolor=INK, linewidth=0.4)
        ax.text(0.12, yy, step, va="center", ha="left", fontsize=8, weight="bold")
        if idx < len(steps) - 1:
            ax.plot([0.05, 0.05], [yy - 0.05, y[idx + 1] + 0.05], color="#AAB2BD", lw=1)
    ax.set_title("Evidence chain", loc="left", fontsize=9, weight="bold")

    counts = _counts(row["status"] for row in rows)
    labels = list(counts)
    values = [counts[label] for label in labels]
    colors = [GREEN if label == "complete" else GOLD if "review" in label else RED if "recompute" in label else GRAY for label in labels]
    axes[1].bar(labels, values, color=colors, width=0.62)
    axes[1].set_ylabel("evidence rows")
    axes[1].set_title("Phase 4 status", loc="left", fontsize=9, weight="bold")
    axes[1].tick_params(axis="x", rotation=25)
    _annotate_bars(axes[1])
    _caption(fig, "Core claim: evidence is locked by explicit gates. Wording guardrail: no global dominance claim.")
    return fig


def plot_fig2(data: dict[str, list[dict[str, str]]]) -> plt.Figure:
    evidence = data["evidence"]
    rows = [row for row in evidence if row["goal_id"] == "goal1_formalize_benchmarks"]
    labels = [row["evidence_area"].replace("_", "\n") for row in rows]
    values = [1 if row["status"] == "complete" else 0 for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.bar(labels, values, color=[GREEN if v else RED for v in values])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("gate passed")
    ax.set_title("Benchmark formalization gates", loc="left", fontsize=9, weight="bold")
    for idx, row in enumerate(rows):
        ax.text(idx, 1.04 if values[idx] else 0.08, row["tier"], ha="center", va="bottom", fontsize=7)
    _caption(fig, "Controlled and real-background benchmarks meet five-replicate and mechanism coverage gates.")
    return fig


def plot_fig3(data: dict[str, list[dict[str, str]]]) -> plt.Figure:
    rows = data["baseline"]
    methods = [row["method"] for row in rows]
    aupr = [_float(row["aupr_mean"]) for row in rows]
    weak = [_float(row["weak_signal_recall_at_threshold_mean"]) for row in rows]
    fp = [_float(row["false_positive_high_expression_rate_at_threshold_mean"]) for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))
    for ax, vals, title, color in zip(axes, [aupr, weak, fp], ["AUPR", "Weak recall", "High-expression FP"], [BLUE, GREEN, RED]):
        ax.bar([_method_label(m) for m in methods], vals, color=color, width=0.62)
        ax.set_title(title, loc="left", fontsize=9, weight="bold")
        ax.tick_params(axis="x", rotation=30)
    _caption(fig, "Replicated baseline panel: winners are metric-specific; no global dominance wording.")
    return fig


def plot_fig4(data: dict[str, list[dict[str, str]]]) -> plt.Figure:
    rows = data["weak_claims"]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    labels = [row["claim_id"].replace("G2_", "").replace("_", "\n") for row in rows]
    status_to_value = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}
    values = [status_to_value.get(row["status"], 0.25) for row in rows]
    colors = [GREEN if row["status"] == "supported" else GOLD if row["status"] == "partial" else RED for row in rows]
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("claim support")
    ax.set_title("Weak-signal wording guardrails", loc="left", fontsize=9, weight="bold")
    _caption(fig, "Supported claim is tradeoff-based; raw highest weak recall and tissue dominance remain disallowed.")
    return fig


def plot_fig5(data: dict[str, list[dict[str, str]]]) -> plt.Figure:
    rows = data["ablation"]
    keep = [
        "wavelet_mechanism_niche_edge_prior",
        "wavelet_no_lr_embedding",
        "wavelet_no_spatial_graph",
        "wavelet_no_distance",
        "wavelet_no_expression_features",
        "gat_hard_expression_distance",
        "mlp_hard_expression_distance",
        "wavelet_soft_expression_distance",
    ]
    rows = [row for row in rows if row["ablation"] in keep]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    labels = [_short_ablation(row["ablation"]) for row in rows]
    values = [_float(row["aupr_mean"]) for row in rows]
    errors = [_float(row["aupr_std"]) for row in rows]
    colors = [BLUE if row["ablation"] == "wavelet_mechanism_niche_edge_prior" else GRAY for row in rows]
    ax.bar(labels, values, yerr=errors, color=colors, width=0.66, capsize=2)
    ax.set_ylabel("AUPR mean +/- SD")
    ax.set_title("Registered ablation panel", loc="left", fontsize=9, weight="bold")
    ax.tick_params(axis="x", rotation=35)
    _caption(fig, "Full SpatialSelf has higher AUPR than LR identity, graph, distance, expression, GAT/MLP, and pseudo-label ablations.")
    return fig


def plot_fig6(data: dict[str, list[dict[str, str]]]) -> plt.Figure:
    summary = data["fdr500"]
    group = data["fdr500_group"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    nulls = [row["null_model"] for row in summary]
    axes[0].bar([_short_null(n) for n in nulls], [_float(row["min_bh_fdr"]) for row in summary], color=BLUE)
    axes[0].axhline(0.05, color=RED, ls="--", lw=1)
    axes[0].set_ylabel("min edge BH FDR")
    axes[0].set_title("Edge-level calibration", loc="left", fontsize=9, weight="bold")
    ct_rows = [row for row in group if row["group_level"] == "cell_type_pair"]
    lr_rows = [row for row in group if row["group_level"] == "lr_pair"]
    labels = ["LR pair", "cell-type pair"]
    vals = [sum(int(float(row["n_significant_groups"])) for row in lr_rows), sum(int(float(row["n_significant_groups"])) for row in ct_rows)]
    axes[1].bar(labels, vals, color=[GOLD, GREEN], width=0.55)
    axes[1].set_ylabel("significant groups")
    axes[1].set_title("500-perm group discoveries", loc="left", fontsize=9, weight="bold")
    _annotate_bars(axes[1])
    _caption(fig, "500-permutation key panel supports group-level FDR; edge-level discoveries are absent at alpha 0.05.")
    return fig


def plot_fig7(data: dict[str, list[dict[str, str]]]) -> plt.Figure:
    downstream = _first(data["downstream_summary"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    axes[0].bar(["positive", "negative"], [_float(downstream["mean_positive_spearman"]), _float(downstream["mean_negative_spearman"])], color=[GREEN, GRAY], width=0.55)
    axes[0].set_ylabel("mean downstream Spearman")
    axes[0].set_title("Key lymph-node downstream", loc="left", fontsize=9, weight="bold")
    axes[1].bar(["positive LR", "negative LR"], [int(float(downstream["n_positive_tested"])), int(float(downstream["n_negative_tested"]))], color=[GREEN, GRAY], width=0.55)
    axes[1].set_ylabel("tested controls")
    axes[1].set_title("Target coverage", loc="left", fontsize=9, weight="bold")
    _annotate_bars(axes[1])
    _caption(fig, "Positive controls show higher mean receiver target correlation, but negative controls remain correlated.")
    return fig


def plot_fig8(data: dict[str, list[dict[str, str]]]) -> plt.Figure:
    runtime = _first(data["runtime"])
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    labels = ["profile cells", "high-res cells", "high-res edges / 1000"]
    vals = [
        _float(runtime.get("profile_n_cells_or_spots")),
        _float(runtime.get("highres_n_cells_max")),
        _float(runtime.get("highres_n_candidate_edges_max")) / 1000.0,
    ]
    ax.bar(labels, vals, color=[BLUE, GREEN, GOLD], width=0.58)
    ax.set_ylabel("count")
    ax.set_title("Scalability evidence scope", loc="left", fontsize=9, weight="bold")
    ax.tick_params(axis="x", rotation=20)
    _caption(fig, "Runtime evidence supports wiring and subset scale; full 50k/100k high-resolution claim remains supplementary.")
    return fig


def _load_tables(output_dir: Path) -> dict[str, list[dict[str, str]]]:
    return {
        "evidence": _read_csv(output_dir / "phase4_evidence_index.csv"),
        "baseline": _read_csv(PROJECT_ROOT / "outputs/phase2_revised/goal2_replicated_tuned_baseline_panel/goal2_replicated_tuned_metric_summary.csv"),
        "weak_claims": _read_csv(PROJECT_ROOT / "outputs/phase2_revised/goal2_weak_signal_tradeoff_readiness/goal2_weak_signal_tradeoff_claims.csv"),
        "ablation": _read_csv(output_dir / "goal6_formal_ablation_controlled_5rep/spatialself_ablation_summary.csv"),
        "fdr500": _read_csv(output_dir / "goal3_fdr_500perm_lymph_node/goal5_formal_fdr_summary.csv"),
        "fdr500_group": _read_csv(output_dir / "goal3_fdr_500perm_lymph_node/goal5_formal_fdr_group_summary.csv"),
        "downstream_summary": _read_csv(output_dir / "goal5_downstream_key_lymph_node/phase4_downstream_key_lymph_node_summary.csv"),
        "runtime": _read_csv(PROJECT_ROOT / "outputs/phase2_revised/goal7_runtime_scalability/goal7_runtime_scalability_claims.csv"),
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
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.25, wspace=0.38)
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def _caption(fig: plt.Figure, text: str) -> None:
    fig.text(0.01, 0.02, text, ha="left", va="bottom", fontsize=6.5, color="#4A4A4A")


def _annotate_bars(ax: plt.Axes) -> None:
    ymax = ax.get_ylim()[1]
    for patch in ax.patches:
        height = patch.get_height()
        if not np.isfinite(height):
            continue
        ax.text(patch.get_x() + patch.get_width() / 2, height + ymax * 0.02, f"{height:.2g}", ha="center", va="bottom", fontsize=6.5)


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


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _first(rows: list[dict[str, str]]) -> dict[str, str]:
    return rows[0] if rows else {}


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _method_label(method: str) -> str:
    labels = {
        "spatialself_wavelet_edge_prior": "SpatialSelf",
        "spatialself_wavelet": "SpatialSelf\nbase",
        "commot": "COMMOT",
        "cellchat": "CellChat",
        "spatalk": "SpaTalk",
        "cellnest": "CellNEST",
    }
    return labels.get(method, method)


def _short_ablation(name: str) -> str:
    labels = {
        "wavelet_mechanism_niche_edge_prior": "full",
        "wavelet_no_lr_embedding": "no LR id",
        "wavelet_no_spatial_graph": "no graph",
        "wavelet_no_distance": "no distance",
        "wavelet_no_expression_features": "no expr",
        "gat_hard_expression_distance": "GAT",
        "mlp_hard_expression_distance": "MLP",
        "wavelet_soft_expression_distance": "soft labels",
    }
    return labels.get(name, name)


def _short_null(null_model: str) -> str:
    return {
        "spatial_permutation": "spatial",
        "receiver_permutation": "receiver",
        "lr_identity_permutation": "LR id",
    }.get(null_model, null_model)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
