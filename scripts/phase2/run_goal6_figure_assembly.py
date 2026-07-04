#!/usr/bin/env python
"""Assemble Phase 2 paper-ready source tables and an overview figure."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PALETTE = {
    "blue": "#1D3557",
    "mid_blue": "#457B9D",
    "green": "#2A9D8F",
    "gold": "#E9C46A",
    "red": "#E76F51",
    "gray": "#6D6875",
    "light_gray": "#E8E8E8",
    "dark": "#222222",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/phase2/goal6_paper_figures")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    source_dir = output_dir / "source_tables"
    figure_dir = output_dir / "figures"
    panel_dir = output_dir / "manuscript_panels"
    source_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)

    source_tables = build_source_tables(source_dir)
    figure_paths = plot_overview(source_tables, figure_dir)
    panel_paths = plot_manuscript_panels(source_tables, panel_dir)
    all_figure_paths = figure_paths + panel_paths
    manifest = {
        "goal_id": "6",
        "analysis_label": "phase2_overview_figure_assembly",
        "evidence_status": "initial_figure_sources",
        "source_tables": {key: str(path.relative_to(PROJECT_ROOT)) for key, path in source_tables.items()},
        "figures": [str(path.relative_to(PROJECT_ROOT)) for path in figure_paths],
        "manuscript_panels": [str(path.relative_to(PROJECT_ROOT)) for path in panel_paths],
        "figure_brief": {
            "claim_scope": (
                "Phase 2 overview of current evidence. This is a source-table and figure assembly "
                "artifact, not a final manuscript claim that all Phase 2 goals are complete."
            ),
            "panels": [
                "retained baseline comparison",
                "component ablation",
                "controlled weak-signal recall",
                "mechanism-specific weak-signal recall",
                "full-slide weak-recall/FDR operating point",
                "LR-pair-balanced operating-rule diagnostic",
                "direct group-level FDR",
                "model-selection score",
                "tonsil high-resolution controls",
                "tonsil baseline-matched subset controls",
                "tonsil CCC module/control separation",
                "downstream and LR-embedding interpretability diagnostics",
                "mechanism/category interpretability diagnostics",
                "lymph node real-background injected-control validation",
                "lymph node baseline-matched injected-control validation",
            ],
        },
        "limitations": [
            "Goal 4 remains subset-level high-resolution evidence.",
            "Goal 5 downstream evidence is mixed and LR embedding clustering is weak.",
            "Baseline comparison sources include full controlled, adapter-light, projection-only, and single-replicate "
            "full-slide rows; interpretation must preserve these evidence levels.",
            "Component ablation source is Phase 1 controlled component-level evidence, not a final Phase 2 default ablation.",
            "Lymph node validation source is full-slide sparse real-background injected-control evidence, "
            "not unlabeled real-data ground truth.",
            "Baseline-matched lymph-node validation is a single injected-control replicate; CellChat, SpaTalk, "
            "and CellNEST rows are projected to candidate edges and should be interpreted with that scope.",
            "High-resolution scaling is subset-level tonsil evidence up to 1,800 cells, not a full dataset run.",
            "Tonsil CCC modules support module/control separation on one 720-cell subset, not strong spatial confinement "
            "or unlabeled real-data ground truth.",
            "Goal 6 is an initial reproducible figure assembly with separate manuscript-style panels, "
            "not the final full paper figure suite.",
        ],
    }
    manifest_path = output_dir / "goal6_figure_assembly_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary_path = output_dir / "goal6_figure_assembly_summary.csv"
    pd.DataFrame(
        [
            {
                "goal_id": "6",
                "analysis_label": "phase2_overview_figure_assembly",
                "evidence_status": "initial_figure_sources",
                "n_source_tables": len(source_tables),
                "n_figure_files": len(all_figure_paths),
                "n_manuscript_panel_files": len(panel_paths),
                "overview_png": str((figure_dir / "phase2_overview.png").relative_to(PROJECT_ROOT)),
                "overview_pdf": str((figure_dir / "phase2_overview.pdf").relative_to(PROJECT_ROOT)),
                "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
            }
        ]
    ).to_csv(summary_path, index=False)
    print(f"Goal 6 summary:  {summary_path}")
    print(f"Goal 6 manifest: {manifest_path}")
    for path in all_figure_paths:
        print(f"Figure: {path}")


def build_source_tables(source_dir: Path) -> dict[str, Path]:
    controlled = _read_required(
        "outputs/phase2/goal1_weak_signal_recovery/controlled_five_replicate_initial/weak_signal_recovery_summary.csv",
        [
            "ablation",
            "weak_signal_recall_at_threshold_mean",
            "weak_signal_recall_at_threshold_std",
            "aupr_mean",
            "empirical_fdr_at_threshold_mean",
            "false_positive_high_expression_rate_at_threshold_mean",
        ],
    )
    controlled_source = controlled[
        [
            "ablation",
            "weak_signal_recall_at_threshold_mean",
            "weak_signal_recall_at_threshold_std",
            "aupr_mean",
            "empirical_fdr_at_threshold_mean",
            "false_positive_high_expression_rate_at_threshold_mean",
        ]
    ].copy()
    controlled_source["display_label"] = controlled_source["ablation"].map(_display_model_label)
    controlled_source["metric_context"] = "controlled_benchmark"
    mechanism_weak = _goal1_mechanism_source()

    tradeoff = _read_required(
        "outputs/phase2/goal1_weak_signal_recovery/full_slide_sparse_threshold_tradeoff/threshold_tradeoff_summary.csv",
        [
            "label",
            "status",
            "best_quantile",
            "best_weak_signal_recall",
            "best_fdr",
            "best_fdr_delta_vs_95",
            "best_false_positive_high_expression_rate",
        ],
    )
    tradeoff_source = tradeoff.copy()
    tradeoff_source["display_label"] = tradeoff_source["label"].map(_display_tradeoff_label)
    lr_balanced_tradeoff = _goal1_lr_balanced_operating_source()
    lr_balanced_stability = _goal1_lr_balanced_stability_source()

    fdr = _read_required(
        "outputs/phase2/goal2_fdr_calibration/full_slide_edge_prior_direct_group_grid_500perm_focused/direct_group_grid_summary.csv",
        [
            "null_model",
            "group_level",
            "top_fraction",
            "positive_control_min_group_bh_fdr",
            "positive_control_significant_groups",
            "negative_control_significant_groups",
            "phase2_goal2_pass",
            "status",
        ],
    )
    fdr_source = fdr[
        [
            "null_model",
            "group_level",
            "top_fraction",
            "positive_control_min_group_bh_fdr",
            "positive_control_significant_groups",
            "negative_control_significant_groups",
            "phase2_goal2_pass",
            "status",
        ]
    ].copy()
    fdr_source["display_label"] = (
        fdr_source["null_model"].map(_display_null_label)
        + "\n"
        + fdr_source["group_level"].map(_display_group_level)
        + ", top "
        + (fdr_source["top_fraction"].astype(float) * 100).map(lambda value: f"{value:g}%")
    )
    fdr_replicate_source = _goal2_replicate_stability_source()

    selection = _read_required(
        "outputs/phase2/goal3_model_selection/model_selection_evidence.csv",
        [
            "candidate_id",
            "label",
            "selection_score",
            "recommended_default",
            "controlled_weak_recall",
            "operating_weak_recall",
            "operating_point_source",
            "operating_lr_quantile",
            "operating_global_floor_quantile",
            "lr_balanced_weak_recall",
            "lr_balanced_fdr",
            "lr_balanced_fdr_delta_vs_reference",
            "lr_balanced_fp_high_expr_rate",
            "group_fdr_positive_bh_fdr",
            "group_fdr_seed_stability_passing_fraction",
            "group_fdr_seed_stability_max_negative_sig_groups",
            "group_fdr_replicate_stability_passing_fraction",
            "group_fdr_replicate_stability_max_negative_sig_groups",
        ],
    )
    selection_source = selection[
        [
            "candidate_id",
            "label",
            "selection_score",
            "recommended_default",
            "controlled_weak_recall",
            "operating_weak_recall",
            "operating_point_source",
            "operating_lr_quantile",
            "operating_global_floor_quantile",
            "lr_balanced_weak_recall",
            "lr_balanced_fdr",
            "lr_balanced_fdr_delta_vs_reference",
            "lr_balanced_fp_high_expr_rate",
            "group_fdr_positive_bh_fdr",
            "group_fdr_seed_stability_passing_fraction",
            "group_fdr_seed_stability_max_negative_sig_groups",
            "group_fdr_replicate_stability_passing_fraction",
            "group_fdr_replicate_stability_max_negative_sig_groups",
        ]
    ].copy()
    selection_source["display_label"] = selection_source["candidate_id"].map(_display_candidate_label)

    highres_profile = _read_required(
        "outputs/phase2/goal4_high_resolution/tonsil_fresh_wavelet_calibrated_subset/tonsil_fresh_dataset_profile.csv",
        ["n_cells", "n_lr_pairs", "n_candidate_edges", "wall_clock_seconds", "peak_rss_mb"],
    )
    highres_controls = _read_required(
        "outputs/phase2/goal4_high_resolution/tonsil_fresh_wavelet_calibrated_subset/tonsil_fresh_control_summary.csv",
        ["control_name", "control_class", "rank_by_score_mean", "rank_percentile_by_score_mean"],
    )
    highres_source = highres_controls[["control_name", "control_class", "rank_by_score_mean", "rank_percentile_by_score_mean"]].copy()
    highres_source["profile_n_cells"] = int(highres_profile.iloc[0]["n_cells"])
    highres_source["profile_n_candidate_edges"] = int(highres_profile.iloc[0]["n_candidate_edges"])
    highres_source["profile_wall_clock_seconds"] = float(highres_profile.iloc[0]["wall_clock_seconds"])
    highres_source["profile_peak_rss_mb"] = float(highres_profile.iloc[0]["peak_rss_mb"])
    highres_scaling = _high_resolution_scaling_source()
    tonsil_baseline = _tonsil_baseline_matched_source()
    tonsil_modules = _tonsil_ccc_modules_source()
    tonsil_modules_n1800 = _tonsil_ccc_modules_n1800_source()

    interp = _read_required(
        "outputs/phase2/goal5_interpretability/tonsil_fresh_subset/tonsil_interpretability_control_summary.csv",
        ["control_class", "mean_residual_spearman", "n_residual_bh_fdr_0_10", "n_tested"],
    )
    embedding = _read_required(
        "outputs/phase2/goal5_interpretability/lr_embedding_tonsil_fresh_subset/goal5_lr_embedding_summary.csv",
        [
            "silhouette_mechanism",
            "silhouette_control_class",
            "n_significant_cluster_enrichments_fdr_0_10",
            "embedding_norm_spatial_range_spearman",
        ],
    )
    interp_source = interp[["control_class", "mean_residual_spearman", "n_residual_bh_fdr_0_10", "n_tested"]].copy()
    for column in [
        "silhouette_mechanism",
        "silhouette_control_class",
        "n_significant_cluster_enrichments_fdr_0_10",
        "embedding_norm_spatial_range_spearman",
    ]:
        interp_source[f"embedding_{column}"] = embedding.iloc[0][column]
    interp_n1800 = _goal5_n1800_interpretability_source()
    mechanism_category = _mechanism_category_source()
    lymph_node_interpretability = _goal5_lymph_node_interpretability_source()
    interpretability_synthesis = _goal5_interpretability_synthesis_source()

    baseline_controlled = _baseline_controlled_source()
    baseline_full_slide = _baseline_full_slide_source()
    component_ablation = _component_ablation_source()
    lymph_node = _lymph_node_global_validation_source()
    lymph_baseline = _lymph_node_baseline_matched_source()
    claim_trace = _claim_traceability_source(
        controlled_source=controlled_source,
        tradeoff_source=tradeoff_source,
        lr_balanced_tradeoff=lr_balanced_tradeoff,
        lr_balanced_stability=lr_balanced_stability,
        fdr_source=fdr_source,
        fdr_replicate_source=fdr_replicate_source,
        selection_source=selection_source,
        highres_source=highres_source,
        highres_scaling=highres_scaling,
        tonsil_baseline=tonsil_baseline,
        tonsil_modules=tonsil_modules,
        tonsil_modules_n1800=tonsil_modules_n1800,
        interp_source=interp_source,
        interp_n1800=interp_n1800,
        mechanism_category=mechanism_category,
        lymph_node_interpretability=lymph_node_interpretability,
        interpretability_synthesis=interpretability_synthesis,
        baseline_controlled=baseline_controlled,
        baseline_full_slide=baseline_full_slide,
        component_ablation=component_ablation,
        lymph_node=lymph_node,
        lymph_baseline=lymph_baseline,
    )

    source_tables = {
        "goal6_controlled_baseline_comparison": source_dir / "goal6_controlled_baseline_comparison.csv",
        "goal6_full_slide_baseline_comparison": source_dir / "goal6_full_slide_baseline_comparison.csv",
        "goal6_component_ablation": source_dir / "goal6_component_ablation.csv",
        "goal6_lymph_node_global_validation": source_dir / "goal6_lymph_node_global_validation.csv",
        "goal6_lymph_node_baseline_matched": source_dir / "goal6_lymph_node_baseline_matched.csv",
        "goal6_claim_traceability": source_dir / "goal6_claim_traceability.csv",
        "goal1_controlled_weak_signal": source_dir / "goal1_controlled_weak_signal.csv",
        "goal1_mechanism_weak_signal": source_dir / "goal1_mechanism_weak_signal.csv",
        "goal1_full_slide_operating_point": source_dir / "goal1_full_slide_operating_point.csv",
        "goal1_lr_balanced_operating_point": source_dir / "goal1_lr_balanced_operating_point.csv",
        "goal1_lr_balanced_controlled_stability": source_dir / "goal1_lr_balanced_controlled_stability.csv",
        "goal2_direct_group_fdr": source_dir / "goal2_direct_group_fdr.csv",
        "goal2_direct_group_replicate_stability": source_dir / "goal2_direct_group_replicate_stability.csv",
        "goal3_model_selection": source_dir / "goal3_model_selection.csv",
        "goal4_high_resolution_controls": source_dir / "goal4_high_resolution_controls.csv",
        "goal4_high_resolution_scaling": source_dir / "goal4_high_resolution_scaling.csv",
        "goal4_tonsil_baseline_matched": source_dir / "goal4_tonsil_baseline_matched.csv",
        "goal4_tonsil_ccc_modules": source_dir / "goal4_tonsil_ccc_modules.csv",
        "goal4_tonsil_ccc_modules_n1800": source_dir / "goal4_tonsil_ccc_modules_n1800.csv",
        "goal5_interpretability": source_dir / "goal5_interpretability.csv",
        "goal5_interpretability_n1800": source_dir / "goal5_interpretability_n1800.csv",
        "goal5_mechanism_category": source_dir / "goal5_mechanism_category.csv",
        "goal5_lymph_node_interpretability": source_dir / "goal5_lymph_node_interpretability.csv",
        "goal5_interpretability_synthesis": source_dir / "goal5_interpretability_synthesis.csv",
    }
    controlled_source.to_csv(source_tables["goal1_controlled_weak_signal"], index=False)
    mechanism_weak.to_csv(source_tables["goal1_mechanism_weak_signal"], index=False)
    tradeoff_source.to_csv(source_tables["goal1_full_slide_operating_point"], index=False)
    lr_balanced_tradeoff.to_csv(source_tables["goal1_lr_balanced_operating_point"], index=False)
    lr_balanced_stability.to_csv(source_tables["goal1_lr_balanced_controlled_stability"], index=False)
    fdr_source.to_csv(source_tables["goal2_direct_group_fdr"], index=False)
    fdr_replicate_source.to_csv(source_tables["goal2_direct_group_replicate_stability"], index=False)
    selection_source.to_csv(source_tables["goal3_model_selection"], index=False)
    highres_source.to_csv(source_tables["goal4_high_resolution_controls"], index=False)
    highres_scaling.to_csv(source_tables["goal4_high_resolution_scaling"], index=False)
    tonsil_baseline.to_csv(source_tables["goal4_tonsil_baseline_matched"], index=False)
    tonsil_modules.to_csv(source_tables["goal4_tonsil_ccc_modules"], index=False)
    tonsil_modules_n1800.to_csv(source_tables["goal4_tonsil_ccc_modules_n1800"], index=False)
    interp_source.to_csv(source_tables["goal5_interpretability"], index=False)
    interp_n1800.to_csv(source_tables["goal5_interpretability_n1800"], index=False)
    mechanism_category.to_csv(source_tables["goal5_mechanism_category"], index=False)
    lymph_node_interpretability.to_csv(source_tables["goal5_lymph_node_interpretability"], index=False)
    interpretability_synthesis.to_csv(source_tables["goal5_interpretability_synthesis"], index=False)
    baseline_controlled.to_csv(source_tables["goal6_controlled_baseline_comparison"], index=False)
    baseline_full_slide.to_csv(source_tables["goal6_full_slide_baseline_comparison"], index=False)
    component_ablation.to_csv(source_tables["goal6_component_ablation"], index=False)
    lymph_node.to_csv(source_tables["goal6_lymph_node_global_validation"], index=False)
    lymph_baseline.to_csv(source_tables["goal6_lymph_node_baseline_matched"], index=False)
    claim_trace.to_csv(source_tables["goal6_claim_traceability"], index=False)
    return source_tables


def plot_overview(source_tables: dict[str, Path], figure_dir: Path) -> list[Path]:
    _apply_style()
    controlled = pd.read_csv(source_tables["goal1_controlled_weak_signal"])
    tradeoff = pd.read_csv(source_tables["goal1_full_slide_operating_point"])
    fdr = pd.read_csv(source_tables["goal2_direct_group_fdr"])
    selection = pd.read_csv(source_tables["goal3_model_selection"])
    highres = pd.read_csv(source_tables["goal4_high_resolution_controls"])
    interp = pd.read_csv(source_tables["goal5_interpretability"])
    baseline = pd.read_csv(source_tables["goal6_full_slide_baseline_comparison"])
    ablation = pd.read_csv(source_tables["goal6_component_ablation"])
    lymph_node = pd.read_csv(source_tables["goal6_lymph_node_global_validation"])
    lymph_baseline = pd.read_csv(source_tables["goal6_lymph_node_baseline_matched"])

    fig, axes = plt.subplot_mosaic(
        [["A", "B", "C"], ["D", "E", "F"], ["G", "H", "I"], ["J", "J", "J"]],
        figsize=(12.2, 12.2),
        layout="constrained",
    )
    fig.suptitle("Phase 2 SpatialSelf Evidence Assembly", fontsize=14, fontweight="semibold")

    _panel_baseline_comparison(axes["A"], baseline)
    _panel_component_ablation(axes["B"], ablation)
    _panel_controlled_weak_recall(axes["C"], controlled)
    _panel_tradeoff(axes["D"], tradeoff)
    _panel_group_fdr(axes["E"], fdr)
    _panel_model_selection(axes["F"], selection)
    _panel_highres_controls(axes["G"], highres)
    _panel_interpretability(axes["H"], interp)
    _panel_lymph_node_validation(axes["I"], lymph_node)
    _panel_lymph_node_baseline_matched(axes["J"], lymph_baseline)

    for label, ax in axes.items():
        ax.text(
            -0.13,
            1.06,
            label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left",
        )

    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        figure_dir / "phase2_overview.png",
        figure_dir / "phase2_overview.pdf",
    ]
    for path in outputs:
        fig.savefig(path, bbox_inches="tight", dpi=320)
    plt.close(fig)
    return outputs


def _baseline_controlled_source() -> pd.DataFrame:
    summary = _read_required(
        "outputs/phase1/exp03_baseline_comparison/full_controlled_retained_unified/baseline_metrics_summary.csv",
        [
            "source_run",
            "method",
            "auroc_mean",
            "aupr_mean",
            "weak_signal_recall_at_threshold_mean",
            "empirical_fdr_at_threshold_mean",
            "false_positive_high_expression_rate_at_threshold_mean",
            "auroc_count",
            "baseline_evidence_level",
            "interpretation_note",
        ],
    )
    status = _read_required(
        "outputs/phase1/exp03_baseline_comparison/full_controlled_retained_unified/baseline_status.csv",
        ["method", "status", "detail", "baseline_evidence_level"],
    )
    status = status.drop_duplicates(["method", "baseline_evidence_level"])
    table = summary.merge(
        status,
        on=["method", "baseline_evidence_level"],
        how="left",
        suffixes=("", "_status"),
    )
    table["dataset_context"] = "controlled_five_replicate"
    table["display_label"] = table["method"].map(_display_method_label)
    table["claim_eligible"] = (
        (table["status"] == "completed")
        & (table["auroc_count"].fillna(0).astype(float) >= 5)
        & (table["baseline_evidence_level"] == "full_controlled_core")
    )
    table["claim_scope"] = [
        _baseline_claim_scope(row)
        for row in table.to_dict(orient="records")
    ]
    return table[
        [
            "dataset_context",
            "source_run",
            "method",
            "display_label",
            "baseline_evidence_level",
            "status",
            "claim_eligible",
            "claim_scope",
            "auroc_mean",
            "aupr_mean",
            "weak_signal_recall_at_threshold_mean",
            "empirical_fdr_at_threshold_mean",
            "false_positive_high_expression_rate_at_threshold_mean",
            "auroc_count",
            "interpretation_note",
            "detail",
        ]
    ]


def _goal1_lr_balanced_operating_source() -> pd.DataFrame:
    table = _read_required(
        "outputs/phase2/goal1_weak_signal_recovery/full_slide_sparse_lr_balanced_threshold/lr_balanced_threshold_summary.csv",
        [
            "strategy",
            "status",
            "best_lr_quantile",
            "best_global_floor_quantile",
            "best_weak_signal_recall",
            "best_fdr",
            "best_fdr_delta_vs_reference",
            "best_false_positive_high_expression_rate",
            "n_passing_settings",
        ],
    ).copy()
    table["display_label"] = table["strategy"].map(_display_lr_balanced_strategy)
    table["analysis_scope"] = (
        "Post-hoc LR-pair-balanced operating-rule diagnostic from existing full-slide edge-prior predictions; "
        "not a newly trained model and not a formal FDR procedure."
    )
    return table


def _goal1_lr_balanced_stability_source() -> pd.DataFrame:
    table = _read_required(
        (
            "outputs/phase2/goal1_weak_signal_recovery/controlled_lr_balanced_threshold_stability/"
            "lr_balanced_replicate_aggregate.csv"
        ),
        [
            "strategy",
            "n_replicates",
            "n_passing_replicates",
            "passing_replicate_fraction",
            "best_weak_signal_recall_mean",
            "best_fdr_delta_vs_reference_mean",
            "best_false_positive_high_expression_rate_mean",
            "aggregate_status",
        ],
    ).copy()
    table["display_label"] = table["strategy"].map(_display_lr_balanced_strategy)
    table["analysis_scope"] = (
        "Controlled benchmark cross-replicate stability diagnostic for post-hoc LR-balanced operating rules. "
        "A failed pass fraction means the full-slide LR-balanced diagnostic should not be promoted to a "
        "cross-replicate default operating rule."
    )
    return table


def _baseline_full_slide_source() -> pd.DataFrame:
    summary = _read_required(
        "outputs/phase1/exp03_baseline_comparison/full_slide_sparse_semisynthetic_retained/baseline_metrics_summary.csv",
        [
            "method",
            "auroc_mean",
            "aupr_mean",
            "weak_signal_recall_at_threshold_mean",
            "empirical_fdr_at_threshold_mean",
            "false_positive_high_expression_rate_at_threshold_mean",
            "auroc_count",
        ],
    )
    status = _read_required(
        "outputs/phase1/exp03_baseline_comparison/full_slide_sparse_semisynthetic_retained/baseline_status.csv",
        ["method", "status", "detail"],
    )
    table = summary.merge(status, on="method", how="left")
    table["dataset_context"] = "full_slide_sparse_semisynthetic_single_replicate"
    table["source_run"] = "full_slide_sparse_semisynthetic_retained"
    table["baseline_evidence_level"] = "single_full_slide_replicate"
    table["display_label"] = table["method"].map(_display_method_label)
    table["claim_eligible"] = (
        (table["status"].isin(["completed", "completed_or_model_required"]))
        & (table["auroc_count"].fillna(0).astype(float) >= 1)
    )
    table.loc[table["method"] == "deeptalk", "claim_eligible"] = False
    table["claim_scope"] = [
        _baseline_claim_scope(row)
        for row in table.to_dict(orient="records")
    ]
    return table[
        [
            "dataset_context",
            "source_run",
            "method",
            "display_label",
            "baseline_evidence_level",
            "status",
            "claim_eligible",
            "claim_scope",
            "auroc_mean",
            "aupr_mean",
            "weak_signal_recall_at_threshold_mean",
            "empirical_fdr_at_threshold_mean",
            "false_positive_high_expression_rate_at_threshold_mean",
            "auroc_count",
            "detail",
        ]
    ]


def _component_ablation_source() -> pd.DataFrame:
    summary = _read_required(
        "outputs/phase1/exp04_model_ablation/full_architecture_identity_component_ablation_epoch10/spatialself_ablation_summary.csv",
        [
            "ablation",
            "aupr_mean",
            "aupr_std",
            "aupr_count",
            "weak_signal_recall_at_threshold_mean",
            "weak_signal_recall_at_threshold_std",
            "empirical_fdr_at_threshold_mean",
            "false_positive_high_expression_rate_at_threshold_mean",
        ],
    )
    status = _read_required(
        "outputs/phase1/exp04_model_ablation/full_architecture_identity_component_ablation_epoch10/spatialself_ablation_status.csv",
        ["ablation", "status", "detail"],
    )
    table = summary.merge(status, on="ablation", how="left")
    table["dataset_context"] = "phase1_controlled_component_ablation"
    table["display_label"] = table["ablation"].map(_display_ablation_label)
    table["component_group"] = table["ablation"].map(_ablation_component_group)
    reference = table[table["ablation"] == "wavelet_hard_expression_distance"]
    reference_aupr = float(reference.iloc[0]["aupr_mean"]) if not reference.empty else math.nan
    table["delta_aupr_vs_wavelet_reference"] = table["aupr_mean"].astype(float) - reference_aupr
    table["claim_scope"] = "Phase 1 controlled component ablation; use as component-level evidence only"
    return table[
        [
            "dataset_context",
            "ablation",
            "display_label",
            "component_group",
            "status",
            "aupr_mean",
            "aupr_std",
            "aupr_count",
            "delta_aupr_vs_wavelet_reference",
            "weak_signal_recall_at_threshold_mean",
            "weak_signal_recall_at_threshold_std",
            "empirical_fdr_at_threshold_mean",
            "false_positive_high_expression_rate_at_threshold_mean",
            "claim_scope",
            "detail",
        ]
    ]


def _lymph_node_global_validation_source() -> pd.DataFrame:
    control = _read_required(
        (
            "outputs/phase2/goal6_lymph_node_validation/full_slide_sparse_wavelet_edge_prior/"
            "control_summary/real_validation_control_lr_summary.csv"
        ),
        [
            "method",
            "lr_id",
            "control_type",
            "true_lr",
            "weak_signal",
            "false_positive_high_expression",
            "global_rank",
            "mean_score",
            "top_region",
            "best_region_enrichment_score",
            "best_region_bh_fdr",
            "downstream_spearman_r",
            "downstream_bh_fdr",
            "downstream_status",
        ],
    )
    method_summary = _read_required(
        (
            "outputs/phase2/goal6_lymph_node_validation/full_slide_sparse_wavelet_edge_prior/"
            "control_summary/real_validation_control_method_summary.csv"
        ),
        [
            "method",
            "n_positive_controls",
            "n_negative_controls",
            "mean_positive_rank",
            "mean_negative_rank",
            "best_negative_rank",
            "negative_top3_count",
            "mean_positive_downstream_spearman",
            "mean_positive_region_enrichment",
            "mean_negative_region_enrichment",
        ],
    )
    downstream = _read_required(
        (
            "outputs/phase2/goal6_lymph_node_validation/full_slide_sparse_wavelet_edge_prior/"
            "downstream_target/"
            "human_lymph_node_visium_replicate_00__spatialself_wavelet_edge_prior_predictions__"
            "downstream_target_correlation.csv"
        ),
        ["lr_id", "n_receiver_cells", "n_targets_present", "spearman_r", "bh_fdr", "status"],
    )
    region = _read_required(
        (
            "outputs/phase2/goal6_lymph_node_validation/full_slide_sparse_wavelet_edge_prior/"
            "region_enrichment/"
            "human_lymph_node_visium_replicate_00__spatialself_wavelet_edge_prior_predictions__"
            "region_enrichment.csv"
        ),
        ["lr_id", "region", "region_enrichment_score", "bh_fdr", "n_permutations"],
    )
    table = control[control["method"] == "spatialself_wavelet_edge_prior"].copy()
    downstream_fields = downstream[
        ["lr_id", "n_receiver_cells", "n_targets_present", "spearman_r", "bh_fdr", "status"]
    ].rename(
        columns={
            "spearman_r": "source_downstream_spearman_r",
            "bh_fdr": "source_downstream_bh_fdr",
            "status": "source_downstream_status",
        }
    )
    best_region = (
        region.sort_values(["lr_id", "region_enrichment_score"], ascending=[True, False])
        .drop_duplicates("lr_id")
        [["lr_id", "region", "region_enrichment_score", "bh_fdr", "n_permutations"]]
        .rename(
            columns={
                "region": "source_best_region",
                "region_enrichment_score": "source_best_region_enrichment_score",
                "bh_fdr": "source_best_region_bh_fdr",
            }
        )
    )
    table = table.merge(downstream_fields, on="lr_id", how="left").merge(best_region, on="lr_id", how="left")
    summary = method_summary[method_summary["method"] == "spatialself_wavelet_edge_prior"].iloc[0]
    table["dataset_context"] = "human_lymph_node_visium_full_slide_sparse_real_background_injected_controls"
    table["analysis_scope"] = (
        "Phase 2 full-slide sparse real-background injected-control validation; "
        "not unlabeled real-data ground truth"
    )
    table["display_label"] = table["lr_id"].map(_display_lr_control_label)
    table["display_control_class"] = table["control_type"].map(_display_control_type)
    for column in [
        "n_positive_controls",
        "n_negative_controls",
        "mean_positive_rank",
        "mean_negative_rank",
        "best_negative_rank",
        "negative_top3_count",
        "mean_positive_downstream_spearman",
        "mean_positive_region_enrichment",
        "mean_negative_region_enrichment",
    ]:
        table[f"method_{column}"] = summary[column]
    return table[
        [
            "dataset_context",
            "analysis_scope",
            "method",
            "lr_id",
            "display_label",
            "control_type",
            "display_control_class",
            "true_lr",
            "weak_signal",
            "false_positive_high_expression",
            "global_rank",
            "mean_score",
            "top_region",
            "best_region_enrichment_score",
            "best_region_bh_fdr",
            "downstream_spearman_r",
            "downstream_bh_fdr",
            "downstream_status",
            "source_downstream_spearman_r",
            "source_downstream_bh_fdr",
            "source_downstream_status",
            "source_best_region",
            "source_best_region_enrichment_score",
            "source_best_region_bh_fdr",
            "n_permutations",
            "method_n_positive_controls",
            "method_n_negative_controls",
            "method_mean_positive_rank",
            "method_mean_negative_rank",
            "method_best_negative_rank",
            "method_negative_top3_count",
            "method_mean_positive_downstream_spearman",
            "method_mean_positive_region_enrichment",
            "method_mean_negative_region_enrichment",
        ]
    ]


def _high_resolution_scaling_source() -> pd.DataFrame:
    runs = _read_required(
        "outputs/phase2/goal4_high_resolution/tonsil_scaling_wavelet_calibrated/tonsil_scaling_run_summary.csv",
        [
            "n_cells",
            "n_candidate_edges",
            "wall_clock_seconds",
            "train_and_inference_seconds",
            "peak_rss_mb",
            "n_positive_controls_top_decile",
            "n_negative_controls_top_decile",
            "best_positive_control",
            "best_positive_rank_percentile",
            "best_negative_rank_percentile",
        ],
    )
    summary = _read_required(
        "outputs/phase2/goal4_high_resolution/tonsil_scaling_wavelet_calibrated/tonsil_scaling_summary.csv",
        [
            "n_runs",
            "n_cells_max",
            "n_candidate_edges_max",
            "largest_run_wall_clock_seconds",
            "largest_run_peak_rss_mb",
            "runtime_seconds_per_candidate_edge",
            "interpretation",
        ],
    )
    table = runs.copy()
    table["dataset_context"] = "human_tonsil_slide_tags_subset_scaling"
    table["analysis_scope"] = "Phase 2 high-resolution subset scaling; not full-data high-resolution evidence"
    for column in [
        "n_runs",
        "n_cells_max",
        "n_candidate_edges_max",
        "largest_run_wall_clock_seconds",
        "largest_run_peak_rss_mb",
        "runtime_seconds_per_candidate_edge",
        "interpretation",
    ]:
        table[f"summary_{column}"] = summary.iloc[0][column]
    return table


def _goal1_mechanism_source() -> pd.DataFrame:
    summary = _read_required(
        "outputs/phase2/goal1_weak_signal_recovery/goal1_mechanism_weak_signal_combined_summary.csv",
        [
            "analysis_label",
            "ablation",
            "mechanism",
            "threshold_quantile",
            "n_weak_positive_mean",
            "weak_signal_recall_at_threshold_mean",
            "empirical_fdr_at_threshold_mean",
            "false_positive_high_expression_rate_at_threshold_mean",
            "aupr_mean",
        ],
    )
    keep_ablations = {
        "wavelet_mechanism_niche_edge_prior_ranking",
        "phase2_focal_weak_mining_edge_prior_wavelet",
    }
    keep_analyses = {
        "controlled_five_replicate_initial",
        "full_slide_sparse_initial",
        "full_slide_sparse_operating_q94",
    }
    table = summary[
        summary["ablation"].isin(keep_ablations)
        & summary["analysis_label"].isin(keep_analyses)
        & (summary["mechanism"] != "false_positive_high_expression")
    ].copy()
    table["display_ablation"] = table["ablation"].map(_display_model_label)
    table["display_mechanism"] = table["mechanism"].map(_display_mechanism_label)
    table["display_analysis"] = table["analysis_label"].map(
        {
            "controlled_five_replicate_initial": "controlled q0.95",
            "full_slide_sparse_initial": "full-slide q0.95",
            "full_slide_sparse_operating_q94": "full-slide q0.94",
        }
    )
    table["analysis_scope"] = (
        "Post-hoc Goal 1 mechanism-specific weak-signal diagnostic from existing predictions; "
        "not a new trained model."
    )
    return table


def _goal2_replicate_stability_source() -> pd.DataFrame:
    per_replicate = _read_required(
        "outputs/phase2/goal2_fdr_calibration/controlled_edge_prior_direct_group_replicate_stability/direct_group_replicate_stability_per_replicate.csv",
        [
            "replicate",
            "status",
            "phase2_goal2_pass",
            "positive_control_significant_groups",
            "positive_control_min_group_bh_fdr",
            "negative_control_significant_groups",
            "negative_control_min_group_bh_fdr",
            "n_permutations",
            "null_model",
            "group_level",
            "top_fraction",
        ],
    )
    summary = _read_required(
        "outputs/phase2/goal2_fdr_calibration/controlled_edge_prior_direct_group_replicate_stability/direct_group_replicate_stability_summary.csv",
        [
            "status",
            "n_replicates",
            "n_completed_replicates",
            "passing_replicate_fraction",
            "min_positive_control_significant_groups",
            "max_positive_control_significant_groups",
            "max_negative_control_significant_groups",
            "claim_level",
        ],
    )
    table = per_replicate.copy()
    table["dataset_context"] = "controlled_benchmark_five_replicates"
    table["analysis_scope"] = (
        "Selected direct group-level FDR replicate-stability check on controlled synthetic ground-truth data; "
        "does not replace full-slide realistic-background or real-data FDR validation"
    )
    for column in [
        "status",
        "n_replicates",
        "n_completed_replicates",
        "passing_replicate_fraction",
        "min_positive_control_significant_groups",
        "max_positive_control_significant_groups",
        "max_negative_control_significant_groups",
        "claim_level",
    ]:
        table[f"summary_{column}"] = summary.iloc[0][column]
    return table


def _tonsil_baseline_matched_source() -> pd.DataFrame:
    method_summary = _read_required(
        "outputs/phase2/goal4_high_resolution/tonsil_baseline_matched_fresh_subset/tonsil_baseline_matched_method_summary.csv",
        [
            "method",
            "status",
            "method_scope",
            "n_prediction_rows",
            "n_unique_scores",
            "n_nonzero_scores",
            "n_positive_controls_top_decile",
            "n_negative_controls_top_decile",
            "best_positive_control",
            "best_positive_control_rank",
            "best_positive_rank_percentile",
            "best_negative_control",
            "best_negative_control_rank",
            "best_negative_rank_percentile",
            "negative_top3_count",
            "wall_clock_seconds",
            "rank_validation_pass",
        ],
    )
    status = _read_required(
        "outputs/phase2/goal4_high_resolution/tonsil_baseline_matched_fresh_subset/tonsil_baseline_matched_status.csv",
        ["method", "detail", "claim_eligible"],
    )
    overview = _read_required(
        "outputs/phase2/goal4_high_resolution/tonsil_baseline_matched_fresh_subset/tonsil_baseline_matched_summary.csv",
        [
            "n_cells",
            "n_candidate_edges",
            "n_methods_completed",
            "n_methods_rank_validation_pass",
            "best_rank_validation_method",
            "spatialself_rank_validation_pass",
        ],
    )
    table = method_summary.merge(status[["method", "detail", "claim_eligible"]], on="method", how="left")
    table["dataset_context"] = "human_tonsil_slide_tags_fresh_subset_controls"
    table["analysis_scope"] = (
        "Phase 2 baseline-matched fresh tonsil subset control ranking; "
        "not ground-truth AUROC/AUPR and not final baseline superiority evidence"
    )
    table["display_label"] = table["method"].map(_display_method_label)
    for column in [
        "n_cells",
        "n_candidate_edges",
        "n_methods_completed",
        "n_methods_rank_validation_pass",
        "best_rank_validation_method",
        "spatialself_rank_validation_pass",
    ]:
        table[f"overview_{column}"] = overview.iloc[0][column]
    return table


def _tonsil_ccc_modules_source() -> pd.DataFrame:
    return _tonsil_ccc_modules_source_from(
        root="outputs/phase2/goal4_high_resolution/tonsil_ccc_modules_fresh_subset",
        dataset_context="human_tonsil_slide_tags_fresh_subset_ccc_modules",
        analysis_scope=(
            "Phase 2 CCC module diagnostic on one 720-cell tonsil subset; supports module/control separation, "
            "not strong spatial confinement or unlabeled real-data ground truth"
        ),
    )


def _tonsil_ccc_modules_n1800_source() -> pd.DataFrame:
    return _tonsil_ccc_modules_source_from(
        root="outputs/phase2/goal4_high_resolution/tonsil_ccc_modules_n1800_subset",
        dataset_context="human_tonsil_slide_tags_n1800_subset_ccc_modules",
        analysis_scope=(
            "Larger-subset CCC module stress test; records weakening of positive/housekeeping module separation "
            "and should not be used as a positive module-interpretability claim"
        ),
    )


def _tonsil_ccc_modules_source_from(*, root: str, dataset_context: str, analysis_scope: str) -> pd.DataFrame:
    summary = _read_required(
        f"{root}/tonsil_ccc_module_summary.csv",
        [
            "ccc_module",
            "n_lr_pairs",
            "dominant_mechanism",
            "dominant_mechanism_fraction",
            "n_positive_controls",
            "n_housekeeping_negative_controls",
            "dominant_cell_type_pair",
            "receiver_spatial_concentration",
            "top_lr_pairs",
            "control_classes",
        ],
    )
    enrichment = _read_required(
        f"{root}/tonsil_ccc_module_control_enrichment.csv",
        ["ccc_module", "control_class", "n_in_module", "bh_fdr"],
    )
    overview = _read_required(
        f"{root}/tonsil_ccc_module_overview.csv",
        [
            "evidence_status",
            "n_lr_pairs",
            "n_modules",
            "best_positive_module",
            "best_positive_module_spatial_concentration",
            "mean_module_cell_type_pair_concentration",
            "operating_quantile",
            "interpretation",
        ],
    )
    table = summary.copy()
    enrichment_wide = (
        enrichment.pivot_table(
            index="ccc_module",
            columns="control_class",
            values="bh_fdr",
            aggfunc="min",
        )
        .reset_index()
        .rename(
            columns={
                "tonsil_positive_prior": "positive_control_bh_fdr",
                "housekeeping_negative_control": "housekeeping_negative_bh_fdr",
                "background_lr": "background_bh_fdr",
            }
        )
    )
    table = table.merge(enrichment_wide, on="ccc_module", how="left")
    table["display_module"] = "M" + table["ccc_module"].astype(str)
    table["display_mechanism"] = table["dominant_mechanism"].map(_display_mechanism_label)
    table["dataset_context"] = dataset_context
    table["analysis_scope"] = analysis_scope
    for column in [
        "evidence_status",
        "n_lr_pairs",
        "n_modules",
        "best_positive_module",
        "best_positive_module_spatial_concentration",
        "mean_module_cell_type_pair_concentration",
        "operating_quantile",
        "interpretation",
    ]:
        table[f"overview_{column}"] = overview.iloc[0][column]
    return table


def _goal5_n1800_interpretability_source() -> pd.DataFrame:
    control = _read_required(
        "outputs/phase2/goal5_interpretability/tonsil_n1800_subset/tonsil_interpretability_control_summary.csv",
        ["control_class", "mean_residual_spearman", "n_residual_bh_fdr_0_10", "n_tested"],
    )
    summary = _read_required(
        "outputs/phase2/goal5_interpretability/tonsil_n1800_subset/goal5_interpretability_summary.csv",
        [
            "n_tested_lr_pairs",
            "positive_control_mean_residual_spearman",
            "housekeeping_negative_mean_residual_spearman",
            "background_lr_mean_residual_spearman",
            "significant_positive_controls_residual_fdr_0_10",
            "significant_negative_controls_residual_fdr_0_10",
            "n_permutations",
        ],
    )
    table = control[["control_class", "mean_residual_spearman", "n_residual_bh_fdr_0_10", "n_tested"]].copy()
    table["dataset_context"] = "human_tonsil_slide_tags_n1800_subset"
    table["analysis_scope"] = (
        "Larger-subset downstream interpretability stress test; records instability of positive-control "
        "residual enrichment and should not be used as a positive biological-interpretability claim"
    )
    for column in [
        "n_tested_lr_pairs",
        "positive_control_mean_residual_spearman",
        "housekeeping_negative_mean_residual_spearman",
        "background_lr_mean_residual_spearman",
        "significant_positive_controls_residual_fdr_0_10",
        "significant_negative_controls_residual_fdr_0_10",
        "n_permutations",
    ]:
        table[f"summary_{column}"] = summary.iloc[0][column]
    return table


def _mechanism_category_source() -> pd.DataFrame:
    mechanism = _read_required(
        (
            "outputs/phase2/goal5_interpretability/mechanism_category_tonsil_fresh_subset/"
            "mechanism_interpretability_summary.csv"
        ),
        [
            "mechanism",
            "n_lr_pairs",
            "n_positive_controls",
            "n_housekeeping_negative_controls",
            "mean_residual_spearman",
            "n_residual_bh_fdr_0_10",
            "mean_receiver_spatial_concentration",
            "top_lr_by_residual",
        ],
    )
    overview = _read_required(
        (
            "outputs/phase2/goal5_interpretability/mechanism_category_tonsil_fresh_subset/"
            "goal5_mechanism_category_summary.csv"
        ),
        [
            "positive_minus_housekeeping_residual_spearman",
            "n_positive_controls_residual_bh_fdr_0_10",
            "n_housekeeping_negatives_residual_bh_fdr_0_10",
            "best_mechanism_by_residual",
            "best_category_by_residual",
            "n_significant_group_enrichments_fdr_0_10",
            "embedding_cluster_category_purity",
            "spatial_range_residual_spearman",
            "spatial_concentration_residual_spearman",
        ],
    )
    category = _read_required(
        (
            "outputs/phase2/goal5_interpretability/mechanism_category_tonsil_fresh_subset/"
            "category_interpretability_summary.csv"
        ),
        ["lr_category", "mean_residual_spearman", "n_lr_pairs", "n_residual_bh_fdr_0_10"],
    )
    table = mechanism.copy()
    table["display_label"] = table["mechanism"].map(_display_mechanism_label)
    table["analysis_scope"] = (
        "Phase 2 mechanism/category interpretability on one fresh tonsil subset; "
        "association evidence only, not proof of LR mechanism recovery"
    )
    for column in overview.columns:
        table[f"overview_{column}"] = overview.iloc[0][column]
    top_category = category.sort_values("mean_residual_spearman", ascending=False).iloc[0]
    table["top_category_by_residual"] = top_category["lr_category"]
    table["top_category_mean_residual_spearman"] = top_category["mean_residual_spearman"]
    return table


def _goal5_interpretability_synthesis_source() -> pd.DataFrame:
    summary = _read_required(
        "outputs/phase2/goal5_interpretability/synthesis/interpretability_synthesis_summary.csv",
        [
            "claim_level",
            "support_score",
            "contradiction_score",
            "fresh_positive_minus_housekeeping_residual",
            "fresh_positive_minus_background_residual",
            "n1800_positive_minus_housekeeping_residual",
            "n1800_positive_minus_background_residual",
            "module720_positive_bh_fdr",
            "module1800_positive_bh_fdr",
            "embedding_silhouette_mechanism",
            "embedding_significant_cluster_enrichments",
            "interpretation",
        ],
    )
    claims = _read_required(
        "outputs/phase2/goal5_interpretability/synthesis/interpretability_synthesis_claims.csv",
        ["claim_id", "support_status", "primary_metric", "primary_value", "contradiction", "evidence_scope"],
    )
    table = claims.copy()
    for column in summary.columns:
        table[f"summary_{column}"] = summary.iloc[0][column]
    table["analysis_scope"] = (
        "Goal 5 synthesis across 720-cell tonsil, 1,800-cell stress test, CCC modules, LR embeddings, "
        "and mechanism/category diagnostics; intended to constrain manuscript claims."
    )
    return table


def _goal5_lymph_node_interpretability_source() -> pd.DataFrame:
    summary = _read_required(
        "outputs/phase2/goal5_interpretability/lymph_node_real_background/lymph_node_goal5_interpretability_summary.csv",
        [
            "claim_level",
            "n_positive_downstream_supported",
            "n_weak_positive_downstream_supported",
            "n_negative_top3",
            "mean_positive_downstream_spearman",
            "mean_positive_region_enrichment",
            "mean_negative_region_enrichment",
            "interpretation",
        ],
    )
    claims = _read_required(
        "outputs/phase2/goal5_interpretability/lymph_node_real_background/lymph_node_goal5_interpretability_claims.csv",
        ["claim_id", "support_status", "primary_metric", "primary_value", "secondary_metric", "secondary_value"],
    )
    table = claims.copy()
    for column in summary.columns:
        table[f"summary_{column}"] = summary.iloc[0][column]
    table["analysis_scope"] = (
        "Goal 5 lymph-node real-background injected-control interpretability; downstream/rank support only, "
        "not unlabeled real-data ground truth."
    )
    return table


def _lymph_node_baseline_matched_source() -> pd.DataFrame:
    method_summary = _read_required(
        (
            "outputs/phase2/goal6_lymph_node_validation/baseline_matched_full_slide_sparse/"
            "baseline_matched_method_summary.csv"
        ),
        [
            "method",
            "status",
            "detail",
            "n_lr_pairs_ranked",
            "mean_positive_rank",
            "mean_negative_rank",
            "best_negative_rank",
            "negative_top3_count",
            "positive_controls_rank_ahead_of_negatives",
            "mean_positive_downstream_spearman",
            "mean_positive_region_enrichment",
            "mean_negative_region_enrichment",
            "n_downstream_tested",
            "n_downstream_bh_fdr_0_05",
            "best_region_bh_fdr",
            "rank_validation_pass",
        ],
    )
    status = _read_required(
        (
            "outputs/phase2/goal6_lymph_node_validation/baseline_matched_full_slide_sparse/"
            "baseline_matched_status.csv"
        ),
        ["method", "n_prediction_rows", "n_unique_scores", "n_nonzero_scores", "claim_eligible"],
    )
    overview = _read_required(
        (
            "outputs/phase2/goal6_lymph_node_validation/baseline_matched_full_slide_sparse/"
            "baseline_matched_validation_summary.csv"
        ),
        [
            "n_cells",
            "n_candidate_edges",
            "n_claim_eligible_methods",
            "n_methods_rank_validation_pass",
            "best_rank_validation_method",
        ],
    )
    table = method_summary.merge(
        status[["method", "n_prediction_rows", "n_unique_scores", "n_nonzero_scores", "claim_eligible"]],
        on="method",
        how="left",
    )
    table["dataset_context"] = "human_lymph_node_visium_full_slide_sparse_real_background_injected_controls"
    table["analysis_scope"] = (
        "Phase 2 baseline-matched single-replicate injected-control validation; "
        "projected baseline rows keep their original evidence scope"
    )
    table["display_label"] = table["method"].map(_display_method_label)
    table["method_evidence_scope"] = table.apply(_baseline_matched_method_scope, axis=1)
    for column in [
        "n_cells",
        "n_candidate_edges",
        "n_claim_eligible_methods",
        "n_methods_rank_validation_pass",
        "best_rank_validation_method",
    ]:
        table[f"overview_{column}"] = overview.iloc[0][column]
    return table[
        [
            "dataset_context",
            "analysis_scope",
            "method",
            "display_label",
            "status",
            "detail",
            "method_evidence_scope",
            "claim_eligible",
            "n_prediction_rows",
            "n_unique_scores",
            "n_nonzero_scores",
            "n_lr_pairs_ranked",
            "mean_positive_rank",
            "mean_negative_rank",
            "best_negative_rank",
            "negative_top3_count",
            "positive_controls_rank_ahead_of_negatives",
            "mean_positive_downstream_spearman",
            "mean_positive_region_enrichment",
            "mean_negative_region_enrichment",
            "n_downstream_tested",
            "n_downstream_bh_fdr_0_05",
            "best_region_bh_fdr",
            "rank_validation_pass",
            "overview_n_cells",
            "overview_n_candidate_edges",
            "overview_n_claim_eligible_methods",
            "overview_n_methods_rank_validation_pass",
            "overview_best_rank_validation_method",
        ]
    ]


def _claim_traceability_source(
    *,
    controlled_source: pd.DataFrame,
    tradeoff_source: pd.DataFrame,
    lr_balanced_tradeoff: pd.DataFrame,
    lr_balanced_stability: pd.DataFrame,
    fdr_source: pd.DataFrame,
    fdr_replicate_source: pd.DataFrame,
    selection_source: pd.DataFrame,
    highres_source: pd.DataFrame,
    highres_scaling: pd.DataFrame,
    tonsil_baseline: pd.DataFrame,
    tonsil_modules: pd.DataFrame,
    tonsil_modules_n1800: pd.DataFrame,
    interp_source: pd.DataFrame,
    interp_n1800: pd.DataFrame,
    mechanism_category: pd.DataFrame,
    lymph_node_interpretability: pd.DataFrame,
    interpretability_synthesis: pd.DataFrame,
    baseline_controlled: pd.DataFrame,
    baseline_full_slide: pd.DataFrame,
    component_ablation: pd.DataFrame,
    lymph_node: pd.DataFrame,
    lymph_baseline: pd.DataFrame,
) -> pd.DataFrame:
    edge = controlled_source[controlled_source["ablation"] == "wavelet_mechanism_niche_edge_prior_ranking"].iloc[0]
    trade = tradeoff_source[tradeoff_source["label"] == "edge_prior_ranking"].iloc[0]
    lr_balanced = lr_balanced_tradeoff[
        lr_balanced_tradeoff["strategy"] == "lrwise_quantile_with_global_floor"
    ].iloc[0]
    lr_balanced_controlled = lr_balanced_stability[
        lr_balanced_stability["strategy"] == "lrwise_quantile_with_global_floor"
    ].iloc[0]
    fdr_pass = fdr_source[fdr_source["phase2_goal2_pass"].astype(str).str.lower() == "true"]
    fdr_replicate_summary = fdr_replicate_source.iloc[0]
    default = selection_source[selection_source["recommended_default"].astype(str).str.lower() == "true"].iloc[0]
    pos_top_decile = int(
        (
            (highres_source["control_class"] == "tonsil_positive_prior")
            & (highres_source["rank_percentile_by_score_mean"].astype(float) <= 0.10)
        ).sum()
    )
    neg_top_decile = int(
        (
            (highres_source["control_class"] != "tonsil_positive_prior")
            & (highres_source["rank_percentile_by_score_mean"].astype(float) <= 0.10)
        ).sum()
    )
    largest_highres = highres_scaling.sort_values("n_cells").iloc[-1]
    tonsil_spatialself = tonsil_baseline[tonsil_baseline["method"] == "spatialself_wavelet_calibrated"].iloc[0]
    tonsil_pass = tonsil_baseline[tonsil_baseline["rank_validation_pass"].astype(str).str.lower() == "true"]
    best_module = tonsil_modules[tonsil_modules["ccc_module"] == tonsil_modules.iloc[0]["overview_best_positive_module"]]
    if best_module.empty:
        best_module = tonsil_modules.sort_values("n_positive_controls", ascending=False).head(1)
    best_module = best_module.iloc[0]
    housekeeping_modules = tonsil_modules[tonsil_modules["n_housekeeping_negative_controls"].astype(float) > 0]
    mixed_n1800_modules = tonsil_modules_n1800[
        (tonsil_modules_n1800["n_positive_controls"].astype(float) > 0)
        & (tonsil_modules_n1800["n_housekeeping_negative_controls"].astype(float) > 0)
    ]
    best_n1800_positive_fdr = pd.to_numeric(tonsil_modules_n1800["positive_control_bh_fdr"], errors="coerce").min()
    pos_interp = interp_source[interp_source["control_class"] == "tonsil_positive_prior"].iloc[0]
    neg_interp = interp_source[interp_source["control_class"] == "housekeeping_negative_control"].iloc[0]
    pos_interp_n1800 = interp_n1800[interp_n1800["control_class"] == "tonsil_positive_prior"].iloc[0]
    neg_interp_n1800 = interp_n1800[interp_n1800["control_class"] == "housekeeping_negative_control"].iloc[0]
    mechanism_overview = mechanism_category.iloc[0]
    lymph_node_interpretability_overview = lymph_node_interpretability.iloc[0]
    synthesis_overview = interpretability_synthesis.iloc[0]
    eligible_controlled = baseline_controlled[baseline_controlled["claim_eligible"]]
    eligible_full = baseline_full_slide[baseline_full_slide["claim_eligible"]]
    ablation_best = component_ablation.sort_values("aupr_mean", ascending=False).iloc[0]
    lymph_positive = lymph_node[lymph_node["true_lr"].astype(str).str.lower() == "true"]
    lymph_negative = lymph_node[lymph_node["false_positive_high_expression"].astype(str).str.lower() == "true"]
    lymph_summary = lymph_node.iloc[0]
    lymph_baseline_pass = lymph_baseline[lymph_baseline["rank_validation_pass"].astype(str).str.lower() == "true"]
    lymph_spatialself = lymph_baseline[lymph_baseline["method"] == "spatialself_wavelet_edge_prior"].iloc[0]
    return pd.DataFrame(
        [
            {
                "claim_id": "goal1_controlled_weak_signal",
                "claim_text": "Edge-prior Phase 2 candidate improves controlled weak-signal recall over the 0.155 reference.",
                "support_status": "supported_initial",
                "primary_metric": "controlled weak recall",
                "primary_value": float(edge["weak_signal_recall_at_threshold_mean"]),
                "evidence_artifact": "outputs/phase2/goal1_weak_signal_recovery/controlled_five_replicate_initial/weak_signal_recovery_summary.csv",
                "limitation": "Controlled benchmark only; focal weak mining does not transfer to full-slide realistic background.",
            },
            {
                "claim_id": "goal1_full_slide_operating_point",
                "claim_text": "Calibrated q=0.94 operating point improves full-slide weak-signal recall without large FDR/FP-rate increase.",
                "support_status": "supported_initial",
                "primary_metric": "full-slide weak recall",
                "primary_value": float(trade["best_weak_signal_recall"]),
                "evidence_artifact": "outputs/phase2/goal1_weak_signal_recovery/full_slide_sparse_threshold_tradeoff/threshold_tradeoff_summary.csv",
                "limitation": "Single full-slide semi-synthetic replicate; FDR remains high in absolute terms.",
            },
            {
                "claim_id": "goal1_lr_balanced_operating_point",
                "claim_text": (
                    "LR-pair-balanced operating rule with a global score floor further improves full-slide "
                    "weak-signal recall while retaining the same FDR/FP-rate constraints."
                ),
                "support_status": "supported_initial",
                "primary_metric": "full-slide weak recall",
                "primary_value": float(lr_balanced["best_weak_signal_recall"]),
                "evidence_artifact": (
                    "outputs/phase2/goal1_weak_signal_recovery/full_slide_sparse_lr_balanced_threshold/"
                    "lr_balanced_threshold_summary.csv"
                ),
                "limitation": (
                    "Post-hoc operating-rule diagnostic from one full-slide semi-synthetic replicate; "
                    "not a newly trained model and not a formal FDR procedure. Controlled replicate "
                    f"stability failed under the same constraints: pass fraction "
                    f"{float(lr_balanced_controlled['passing_replicate_fraction']):.3f}, mean FDR delta "
                    f"{float(lr_balanced_controlled['best_fdr_delta_vs_reference_mean']):.3f}."
                ),
            },
            {
                "claim_id": "goal2_direct_group_fdr",
                "claim_text": "Direct group-level permutation finds positive-control LR-pair/module discoveries with high-expression negatives suppressed.",
                "support_status": "supported_initial",
                "primary_metric": "passing direct group FDR rows",
                "primary_value": int(len(fdr_pass)),
                "evidence_artifact": "outputs/phase2/goal2_fdr_calibration/full_slide_edge_prior_direct_group_grid_500perm_focused/direct_group_grid_summary.csv",
                "limitation": "Receiver permutation remains stricter and low power; selected null/statistic needs replicate validation.",
            },
            {
                "claim_id": "goal2_controlled_replicate_fdr",
                "claim_text": (
                    "The selected LR-identity LR-pair direct FDR setting is stable across the five controlled "
                    "ground-truth benchmark replicates."
                ),
                "support_status": "supported_initial",
                "primary_metric": "passing controlled replicate fraction",
                "primary_value": float(fdr_replicate_summary["summary_passing_replicate_fraction"]),
                "evidence_artifact": "outputs/phase2/goal6_paper_figures/source_tables/goal2_direct_group_replicate_stability.csv",
                "limitation": (
                    "Controlled synthetic replicate support only; full-slide realistic-background validation "
                    "is still single-replicate and real-data validation uses injected controls."
                ),
            },
            {
                "claim_id": "goal3_default_model",
                "claim_text": "Phase 2 default candidate is wavelet_calibrated_fdr.",
                "support_status": "supported_initial",
                "primary_metric": "selection score",
                "primary_value": float(default["selection_score"]),
                "evidence_artifact": "outputs/phase2/goal3_model_selection/model_selection_evidence.csv",
                "limitation": "Default for current Phase 2 experiments, not final manuscript default.",
            },
            {
                "claim_id": "goal4_high_resolution_subset",
                "claim_text": "Fresh tonsil subset run is scalable and suppresses housekeeping high-expression negative controls.",
                "support_status": "supported_initial",
                "primary_metric": "positive/negative controls in top decile",
                "primary_value": float(pos_top_decile - neg_top_decile),
                "evidence_artifact": "outputs/phase2/goal4_high_resolution/tonsil_fresh_wavelet_calibrated_subset/tonsil_fresh_control_summary.csv",
                "limitation": "Subset-level high-resolution evidence; no retained-baseline comparison on the same tonsil subset yet.",
            },
            {
                "claim_id": "goal4_high_resolution_scaling",
                "claim_text": "Fresh tonsil subset scaling reaches 1,800 cells and 1.64M candidate edges with high-expression negatives suppressed.",
                "support_status": "supported_initial",
                "primary_metric": "largest scaling candidate edges",
                "primary_value": int(largest_highres["n_candidate_edges"]),
                "evidence_artifact": "outputs/phase2/goal6_paper_figures/source_tables/goal4_high_resolution_scaling.csv",
                "limitation": "Subset-level scaling only; not full Slide-tags tonsil, Xenium, or Visium HD evidence.",
            },
            {
                "claim_id": "goal4_tonsil_baseline_matched_controls",
                "claim_text": (
                    "On the fresh tonsil subset, SpatialSelf passes control-rank validation while retained "
                    "baseline rows do not pass the same positive/negative control criterion."
                ),
                "support_status": "supported_initial",
                "primary_metric": "rank-validation passing methods",
                "primary_value": int(len(tonsil_pass)),
                "evidence_artifact": "outputs/phase2/goal6_paper_figures/source_tables/goal4_tonsil_baseline_matched.csv",
                "limitation": (
                    "Single 720-cell tonsil subset with biological controls, not ground-truth AUROC/AUPR. "
                    f"SpatialSelf positive/negative top-decile controls="
                    f"{int(tonsil_spatialself['n_positive_controls_top_decile'])}/"
                    f"{int(tonsil_spatialself['n_negative_controls_top_decile'])}."
                ),
            },
            {
                "claim_id": "goal4_tonsil_ccc_modules",
                "claim_text": (
                    "CCC module diagnostics separate a long-range chemokine positive-control module from "
                    "a housekeeping high-expression negative-control module on the fresh tonsil subset."
                ),
                "support_status": "supported_initial",
                "primary_metric": "best positive module controls",
                "primary_value": float(best_module["n_positive_controls"]),
                "evidence_artifact": "outputs/phase2/goal6_paper_figures/source_tables/goal4_tonsil_ccc_modules.csv",
                "limitation": (
                    "Single 720-cell subset with biological controls, not ground-truth AUROC/AUPR. "
                    "Spatial concentration values are weak, so this supports module/control separation rather "
                    "than strong spatial confinement. "
                    f"housekeeping_modules={len(housekeeping_modules)}."
                ),
            },
            {
                "claim_id": "goal4_tonsil_n1800_ccc_module_stress_test",
                "claim_text": (
                    "The 1,800-cell tonsil CCC module stress test weakens the clean positive/housekeeping "
                    "module separation seen in the 720-cell subset."
                ),
                "support_status": "mixed_initial",
                "primary_metric": "n1800 mixed positive-housekeeping modules",
                "primary_value": int(len(mixed_n1800_modules)),
                "evidence_artifact": "outputs/phase2/goal6_paper_figures/source_tables/goal4_tonsil_ccc_modules_n1800.csv",
                "limitation": (
                    "This is a larger-subset stress test on the same tonsil data. "
                    f"Best positive-control module BH-FDR={best_n1800_positive_fdr:.3g}; "
                    "use it to constrain module interpretability claims."
                ),
            },
            {
                "claim_id": "goal5_downstream_interpretability",
                "claim_text": "Positive controls exceed housekeeping negatives after receiver-cell-type residualization.",
                "support_status": "mixed_initial",
                "primary_metric": "positive minus housekeeping residual Spearman",
                "primary_value": float(pos_interp["mean_residual_spearman"] - neg_interp["mean_residual_spearman"]),
                "evidence_artifact": "outputs/phase2/goal5_interpretability/tonsil_fresh_subset/tonsil_interpretability_control_summary.csv",
                "limitation": "Broad background immune LR still show stronger downstream correlations; LR embedding clustering is weak.",
            },
            {
                "claim_id": "goal5_n1800_downstream_stress_test",
                "claim_text": (
                    "The larger 1,800-cell tonsil downstream stress test does not preserve the 720-cell "
                    "positive-control residual enrichment pattern."
                ),
                "support_status": "mixed_initial",
                "primary_metric": "n1800 positive minus housekeeping residual Spearman",
                "primary_value": float(
                    pos_interp_n1800["mean_residual_spearman"] - neg_interp_n1800["mean_residual_spearman"]
                ),
                "evidence_artifact": "outputs/phase2/goal6_paper_figures/source_tables/goal5_interpretability_n1800.csv",
                "limitation": (
                    "This is a larger-subset stress test on the same tonsil data, not an independent dataset. "
                    "It argues against overclaiming biological interpretability from the 720-cell subset."
                ),
            },
            {
                "claim_id": "goal5_mechanism_category_interpretability",
                "claim_text": (
                    "Mechanism/category summaries show positive controls exceed housekeeping negatives after "
                    "residualization, but broad background/short-medium paracrine groups remain dominant."
                ),
                "support_status": "mixed_initial",
                "primary_metric": "positive minus housekeeping residual Spearman",
                "primary_value": float(mechanism_overview["overview_positive_minus_housekeeping_residual_spearman"]),
                "evidence_artifact": "outputs/phase2/goal6_paper_figures/source_tables/goal5_mechanism_category.csv",
                "limitation": (
                    "Single tonsil subset; mechanism labels are partly heuristic and significant group enrichments "
                    "are driven mainly by broad background/short-medium groups, not clean positive-control categories."
                ),
            },
            {
                "claim_id": "goal5_interpretability_synthesis",
                "claim_text": (
                    "Goal 5 interpretability evidence is mixed and not yet manuscript-ready as a broad "
                    "biological validation claim."
                ),
                "support_status": "constraint_ready",
                "primary_metric": "synthesis claim level",
                "primary_value": str(synthesis_overview["summary_claim_level"]),
                "evidence_artifact": "outputs/phase2/goal6_paper_figures/source_tables/goal5_interpretability_synthesis.csv",
                "limitation": (
                    f"Support score={synthesis_overview['summary_support_score']}, "
                    f"contradiction score={synthesis_overview['summary_contradiction_score']}; "
                    "use this source to avoid overclaiming 720-cell tonsil interpretability."
                ),
            },
            {
                "claim_id": "goal5_lymph_node_real_background_interpretability",
                "claim_text": (
                    "Lymph-node real-background injected controls support downstream response for positives, "
                    "including the weak positive control, while high-expression negatives are rank-suppressed."
                ),
                "support_status": "supported_initial",
                "primary_metric": "positive downstream support / weak support / negative top3",
                "primary_value": (
                    f"{lymph_node_interpretability_overview['summary_n_positive_downstream_supported']}/"
                    f"{lymph_node_interpretability_overview['summary_n_weak_positive_downstream_supported']}/"
                    f"{lymph_node_interpretability_overview['summary_n_negative_top3']}"
                ),
                "evidence_artifact": "outputs/phase2/goal6_paper_figures/source_tables/goal5_lymph_node_interpretability.csv",
                "limitation": (
                    "Real-background injected-control validation from one lymph-node replicate; "
                    "not unlabeled real-data ground truth."
                ),
            },
            {
                "claim_id": "baseline_controlled_traceability",
                "claim_text": "Controlled benchmark has retained baseline source rows with explicit evidence levels.",
                "support_status": "traceability_ready",
                "primary_metric": "eligible controlled baseline rows",
                "primary_value": int(len(eligible_controlled)),
                "evidence_artifact": "outputs/phase2/goal6_paper_figures/source_tables/goal6_controlled_baseline_comparison.csv",
                "limitation": "Adapter-light and projection/status-only rows must not be interpreted as fully tuned baselines.",
            },
            {
                "claim_id": "baseline_full_slide_traceability",
                "claim_text": "Full-slide benchmark has retained baseline source rows with explicit single-replicate scope.",
                "support_status": "traceability_ready",
                "primary_metric": "eligible full-slide baseline rows",
                "primary_value": int(len(eligible_full)),
                "evidence_artifact": "outputs/phase2/goal6_paper_figures/source_tables/goal6_full_slide_baseline_comparison.csv",
                "limitation": "Single full-slide semi-synthetic replicate; not enough for final broad baseline superiority claim.",
            },
            {
                "claim_id": "component_ablation_traceability",
                "claim_text": "Component ablation source identifies which architecture/input variants retain or lose performance.",
                "support_status": "traceability_ready",
                "primary_metric": "best Phase 1 component-ablation AUPR",
                "primary_value": float(ablation_best["aupr_mean"]),
                "evidence_artifact": "outputs/phase2/goal6_paper_figures/source_tables/goal6_component_ablation.csv",
                "limitation": "Phase 1 controlled component-level evidence; not a final Phase 2 default ablation under calibrated-FDR settings.",
            },
            {
                "claim_id": "lymph_node_real_background_controls",
                "claim_text": "Human lymph-node real-background injected controls show positive LR controls ranked before high-expression negatives.",
                "support_status": "supported_initial",
                "primary_metric": "mean positive rank minus mean negative rank",
                "primary_value": float(
                    lymph_summary["method_mean_positive_rank"] - lymph_summary["method_mean_negative_rank"]
                ),
                "evidence_artifact": "outputs/phase2/goal6_paper_figures/source_tables/goal6_lymph_node_global_validation.csv",
                "limitation": (
                    "Full-slide sparse real-background injected-control evidence; downstream targets are available "
                    "for positive controls but absent for the housekeeping-like high-expression negatives."
                ),
            },
            {
                "claim_id": "lymph_node_baseline_matched_controls",
                "claim_text": (
                    "Baseline-matched lymph-node injected controls show SpatialSelf edge-prior passes "
                    "rank validation while some retained baselines rank high-expression negatives in the top three."
                ),
                "support_status": "supported_initial",
                "primary_metric": "rank-validation passing methods",
                "primary_value": int(len(lymph_baseline_pass)),
                "evidence_artifact": "outputs/phase2/goal6_paper_figures/source_tables/goal6_lymph_node_baseline_matched.csv",
                "limitation": (
                    "Single injected-control replicate; CellChat, SpaTalk, and CellNEST are projected "
                    "candidate-edge rows, so use as validation support rather than broad superiority proof. "
                    f"SpatialSelf edge-prior positive/negative mean ranks="
                    f"{lymph_spatialself['mean_positive_rank']:.2f}/{lymph_spatialself['mean_negative_rank']:.2f}."
                ),
            },
        ]
    )


def plot_manuscript_panels(source_tables: dict[str, Path], panel_dir: Path) -> list[Path]:
    _apply_style()
    panel_dir.mkdir(parents=True, exist_ok=True)
    panel_specs = [
        (
            "fig_goal6_full_slide_baseline_comparison",
            "Retained baseline comparison with evidence levels",
            source_tables["goal6_full_slide_baseline_comparison"],
            _panel_baseline_comparison,
            (6.1, 4.2),
        ),
        (
            "fig_goal6_component_ablation",
            "Component ablation source",
            source_tables["goal6_component_ablation"],
            _panel_component_ablation,
            (6.1, 4.4),
        ),
        (
            "fig_goal1_controlled_weak_signal",
            "Controlled weak-signal recovery",
            source_tables["goal1_controlled_weak_signal"],
            _panel_controlled_weak_recall,
            (5.4, 3.2),
        ),
        (
            "fig_goal1_mechanism_weak_signal",
            "Mechanism-specific weak-signal recovery",
            source_tables["goal1_mechanism_weak_signal"],
            _panel_goal1_mechanism_weak_signal,
            (7.2, 4.2),
        ),
        (
            "fig_goal1_full_slide_operating_point",
            "Full-slide weak-signal/FDR operating point",
            source_tables["goal1_full_slide_operating_point"],
            _panel_tradeoff,
            (5.4, 2.7),
        ),
        (
            "fig_goal2_direct_group_fdr",
            "Direct group-level FDR calibration",
            source_tables["goal2_direct_group_fdr"],
            _panel_group_fdr,
            (5.7, 4.0),
        ),
        (
            "fig_goal3_model_selection",
            "Phase 2 default model selection",
            source_tables["goal3_model_selection"],
            _panel_model_selection,
            (5.5, 3.8),
        ),
        (
            "fig_goal4_high_resolution_controls",
            "High-resolution tonsil subset controls",
            source_tables["goal4_high_resolution_controls"],
            _panel_highres_controls,
            (5.5, 4.2),
        ),
        (
            "fig_goal4_high_resolution_scaling",
            "High-resolution tonsil subset scaling",
            source_tables["goal4_high_resolution_scaling"],
            _panel_highres_scaling,
            (6.2, 4.0),
        ),
        (
            "fig_goal4_tonsil_baseline_matched",
            "Tonsil subset baseline-matched controls",
            source_tables["goal4_tonsil_baseline_matched"],
            _panel_tonsil_baseline_matched,
            (6.4, 4.1),
        ),
        (
            "fig_goal4_tonsil_ccc_modules",
            "Tonsil subset CCC module/control separation",
            source_tables["goal4_tonsil_ccc_modules"],
            _panel_tonsil_ccc_modules,
            (6.7, 4.3),
        ),
        (
            "fig_goal5_interpretability_diagnostics",
            "Biological interpretability diagnostics",
            source_tables["goal5_interpretability"],
            _panel_interpretability,
            (5.1, 3.4),
        ),
        (
            "fig_goal5_mechanism_category",
            "Mechanism/category interpretability diagnostics",
            source_tables["goal5_mechanism_category"],
            _panel_mechanism_category,
            (6.4, 4.2),
        ),
        (
            "fig_goal6_lymph_node_global_validation",
            "Lymph node real-background injected controls",
            source_tables["goal6_lymph_node_global_validation"],
            _panel_lymph_node_validation,
            (5.8, 4.3),
        ),
        (
            "fig_goal6_lymph_node_baseline_matched",
            "Baseline-matched lymph node injected-control validation",
            source_tables["goal6_lymph_node_baseline_matched"],
            _panel_lymph_node_baseline_matched,
            (8.0, 4.5),
        ),
    ]
    outputs: list[Path] = []
    for stem, title, source_path, plotter, figsize in panel_specs:
        data = pd.read_csv(source_path)
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        plotter(ax, data)
        fig.suptitle(title, fontsize=11, fontweight="semibold")
        png = panel_dir / f"{stem}.png"
        pdf = panel_dir / f"{stem}.pdf"
        fig.savefig(png, bbox_inches="tight", dpi=320)
        fig.savefig(pdf, bbox_inches="tight", dpi=320)
        plt.close(fig)
        outputs.extend([png, pdf])
    return outputs


def _panel_controlled_weak_recall(ax, controlled: pd.DataFrame) -> None:
    ordered = controlled.sort_values("weak_signal_recall_at_threshold_mean", ascending=True)
    colors = [
        PALETTE["green"] if value else PALETTE["mid_blue"]
        for value in ordered["ablation"].isin(
            [
                "wavelet_mechanism_niche_edge_prior_ranking",
                "phase2_focal_weak_mining_edge_prior_wavelet",
            ]
        )
    ]
    ax.barh(
        ordered["display_label"],
        ordered["weak_signal_recall_at_threshold_mean"],
        xerr=ordered["weak_signal_recall_at_threshold_std"],
        color=colors,
        edgecolor="none",
        alpha=0.92,
    )
    ax.axvline(0.155, color=PALETTE["red"], lw=1.2, ls="--", label="reference 0.155")
    ax.set_title("Controlled weak-signal recovery")
    ax.set_xlabel("Weak-signal recall")
    ax.set_xlim(0, max(0.19, ordered["weak_signal_recall_at_threshold_mean"].max() + 0.05))
    ax.legend(loc="lower right", fontsize=7)


def _panel_goal1_mechanism_weak_signal(ax, mechanism: pd.DataFrame) -> None:
    plot_rows = mechanism.copy()
    plot_rows = plot_rows[
        plot_rows["ablation"].isin(
            [
                "wavelet_mechanism_niche_edge_prior_ranking",
                "phase2_focal_weak_mining_edge_prior_wavelet",
            ]
        )
    ].copy()
    plot_rows["panel_label"] = plot_rows["display_analysis"].astype(str) + "\n" + plot_rows["display_mechanism"].astype(str)
    preferred = [
        "controlled q0.95\ncontact",
        "controlled q0.95\nshort/medium\nparacrine",
        "controlled q0.95\nlong-range\nchemokine",
        "full-slide q0.95\nlong-range\nchemokine",
        "full-slide q0.94\nlong-range\nchemokine",
    ]
    plot_rows["_priority"] = plot_rows["panel_label"].map({name: idx for idx, name in enumerate(preferred)}).fillna(99)
    plot_rows = plot_rows.sort_values(["_priority", "display_ablation"])
    labels = list(dict.fromkeys(plot_rows["panel_label"]))
    x = np.arange(len(labels))
    width = 0.34
    ablations = [
        "wavelet_mechanism_niche_edge_prior_ranking",
        "phase2_focal_weak_mining_edge_prior_wavelet",
    ]
    offsets = [-width / 2, width / 2]
    colors = [PALETTE["green"], PALETTE["gray"]]
    for ablation, offset, color in zip(ablations, offsets, colors):
        subset = plot_rows[plot_rows["ablation"] == ablation].set_index("panel_label")
        values = [subset.loc[label, "weak_signal_recall_at_threshold_mean"] if label in subset.index else np.nan for label in labels]
        ax.bar(x + offset, values, width=width, color=color, edgecolor="none", label=_display_model_label(ablation))
    ax.axhline(0.155, color=PALETTE["red"], lw=1.0, ls="--", label="controlled ref")
    ax.axhline(0.040, color=PALETTE["dark"], lw=0.9, ls=":", label="full-slide ref")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Weak-signal recall")
    ax.set_title("Mechanism-specific weak-signal recovery")
    ax.set_ylim(0, max(0.62, plot_rows["weak_signal_recall_at_threshold_mean"].max(skipna=True) + 0.06))
    ax.legend(loc="upper right", fontsize=6.8, ncol=2)
    ax.text(
        0.02,
        0.95,
        "post-hoc diagnostic\nfull-slide has long-range weak positives",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "#FFFFFF", "edgecolor": PALETTE["light_gray"]},
    )


def _panel_baseline_comparison(ax, baseline: pd.DataFrame) -> None:
    plot_rows = baseline.copy()
    plot_rows = plot_rows[plot_rows["aupr_mean"].notna()]
    if plot_rows.empty:
        ax.text(0.5, 0.5, "No baseline metrics available", ha="center", va="center")
        ax.set_axis_off()
        return
    plot_rows["_priority"] = plot_rows["method"].map(
        {
            "spatialself_wavelet_edge_prior": 0,
            "spatialself_wavelet": 1,
            "spatialself_gat": 2,
            "commot": 3,
            "cellchat": 4,
            "spatalk": 5,
            "cellnest": 6,
            "deeptalk": 7,
        }
    ).fillna(99)
    plot_rows = plot_rows.sort_values("_priority")
    x = np.arange(len(plot_rows))
    colors = [
        PALETTE["green"]
        if method == "spatialself_wavelet_edge_prior"
        else (PALETTE["mid_blue"] if str(method).startswith("spatialself") else PALETTE["gray"])
        for method in plot_rows["method"]
    ]
    ax.bar(x, plot_rows["aupr_mean"], color=colors, edgecolor="none", width=0.72)
    for idx, row in enumerate(plot_rows.itertuples(index=False)):
        marker = "*" if bool(getattr(row, "claim_eligible")) else "!"
        ax.text(idx, float(getattr(row, "aupr_mean")) + 0.012, marker, ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_rows["display_label"], rotation=35, ha="right", fontsize=7.2)
    ax.set_ylabel("AUPR")
    ax.set_title("Retained baseline source")
    ax.set_ylim(0, max(0.55, plot_rows["aupr_mean"].max() + 0.08))
    ax.text(
        0.02,
        0.96,
        "* claim-eligible row\n! limited/status row",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "#FFFFFF", "edgecolor": PALETTE["light_gray"]},
    )


def _panel_component_ablation(ax, ablation: pd.DataFrame) -> None:
    plot_rows = ablation.copy()
    preferred = [
        "wavelet_hard_expression_distance",
        "wavelet_no_lr_embedding",
        "wavelet_one_hot_lr_identity",
        "wavelet_no_distance",
        "wavelet_no_spatial_graph",
        "wavelet_no_expression_features",
        "gat_hard_expression_distance",
        "gcn_hard_expression_distance",
        "mlp_hard_expression_distance",
    ]
    plot_rows["_priority"] = plot_rows["ablation"].map({name: idx for idx, name in enumerate(preferred)}).fillna(99)
    plot_rows = plot_rows.sort_values("_priority")
    y = np.arange(len(plot_rows))
    colors = [
        PALETTE["green"]
        if name == "wavelet_hard_expression_distance"
        else (PALETTE["mid_blue"] if group == "lr_identity" else PALETTE["gray"])
        for name, group in zip(plot_rows["ablation"], plot_rows["component_group"])
    ]
    ax.barh(
        y,
        plot_rows["aupr_mean"],
        xerr=plot_rows["aupr_std"],
        color=colors,
        edgecolor="none",
        alpha=0.92,
    )
    ref = plot_rows[plot_rows["ablation"] == "wavelet_hard_expression_distance"]
    if not ref.empty:
        ax.axvline(float(ref.iloc[0]["aupr_mean"]), color=PALETTE["dark"], lw=1.0, ls=":", label="wavelet ref")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_rows["display_label"], fontsize=7.1)
    ax.invert_yaxis()
    ax.set_title("Component ablation source")
    ax.set_xlabel("AUPR")
    ax.set_xlim(0, max(0.14, plot_rows["aupr_mean"].max() + plot_rows["aupr_std"].fillna(0).max() + 0.025))
    ax.text(
        0.98,
        0.05,
        "Phase 1 controlled\ncomponent-level evidence",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "#FFFFFF", "edgecolor": PALETTE["light_gray"]},
    )


def _panel_tradeoff(ax, tradeoff: pd.DataFrame) -> None:
    ordered = tradeoff.sort_values("best_weak_signal_recall")
    y = np.arange(len(ordered))
    ax.scatter(
        ordered["best_fdr"],
        y,
        s=260 * np.clip(ordered["best_false_positive_high_expression_rate"] + 0.004, 0.004, None),
        color=[PALETTE["green"] if status == "passing_operating_point" else PALETTE["red"] for status in ordered["status"]],
        alpha=0.9,
    )
    for _, row in ordered.iterrows():
        ax.text(
            row["best_fdr"] + 0.005,
            list(ordered.index).index(row.name),
            f"weak={row['best_weak_signal_recall']:.3f}, q={row['best_quantile']:.2f}",
            fontsize=7.4,
            va="center",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["display_label"])
    ax.set_title("Full-slide operating point")
    ax.set_xlabel("Empirical FDR")
    ax.set_xlim(0.42, 0.70)


def _panel_group_fdr(ax, fdr: pd.DataFrame) -> None:
    plot_rows = fdr[
        (fdr["group_level"] == "lr_pair")
        & (fdr["top_fraction"].astype(float).isin([0.05, 0.10]))
    ].copy()
    plot_rows = plot_rows.sort_values(["null_model", "top_fraction"])
    y = np.arange(len(plot_rows))
    values = plot_rows["positive_control_min_group_bh_fdr"].astype(float).replace([np.inf, -np.inf], np.nan)
    values = values.fillna(1.0)
    ax.barh(
        y,
        values,
        color=[
            PALETTE["green"] if str(pass_flag).lower() == "true" else PALETTE["gray"]
            for pass_flag in plot_rows["phase2_goal2_pass"]
        ],
        edgecolor="none",
    )
    ax.axvline(0.05, color=PALETTE["red"], lw=1.2, ls="--", label="BH-FDR 0.05")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_rows["display_label"], fontsize=7)
    ax.set_xscale("log")
    ax.set_xlim(0.001, 1.2)
    ax.set_title("Direct group-level FDR")
    ax.set_xlabel("Positive-control min BH-FDR")
    ax.legend(loc="lower right", fontsize=7)


def _panel_model_selection(ax, selection: pd.DataFrame) -> None:
    plot_rows = selection.sort_values("selection_score", ascending=True).tail(7)
    colors = [
        PALETTE["green"] if str(value).lower() == "true" else PALETTE["mid_blue"]
        for value in plot_rows["recommended_default"]
    ]
    ax.barh(plot_rows["display_label"], plot_rows["selection_score"], color=colors, edgecolor="none")
    ax.set_title("Phase 2 model selection")
    ax.set_xlabel("Selection score")
    ax.set_xlim(0, max(8.5, plot_rows["selection_score"].max() + 0.8))


def _panel_highres_controls(ax, highres: pd.DataFrame) -> None:
    plot_rows = highres.sort_values("rank_percentile_by_score_mean")
    colors = [
        PALETTE["green"] if control == "tonsil_positive_prior" else PALETTE["red"]
        for control in plot_rows["control_class"]
    ]
    labels = [_short_pair(name) for name in plot_rows["control_name"]]
    y = np.arange(len(plot_rows))
    ax.barh(y, plot_rows["rank_percentile_by_score_mean"], color=colors, edgecolor="none")
    ax.axvline(0.10, color=PALETTE["dark"], lw=1.0, ls=":", label="top decile")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_title("Tonsil subset controls")
    ax.set_xlabel("Rank percentile by mean score")
    ax.set_xlim(0, 1.02)
    profile = highres.iloc[0]
    ax.text(
        0.98,
        0.06,
        f"{int(profile['profile_n_cells'])} cells\n{int(profile['profile_n_candidate_edges']):,} edges\n{profile['profile_wall_clock_seconds']:.1f}s",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.4,
    )


def _panel_highres_scaling(ax, scaling: pd.DataFrame) -> None:
    plot_rows = scaling.sort_values("n_cells").copy()
    ax2 = ax.twinx()
    ax.plot(
        plot_rows["n_cells"],
        plot_rows["n_candidate_edges"],
        marker="o",
        color=PALETTE["mid_blue"],
        lw=1.8,
        label="candidate edges",
    )
    ax2.plot(
        plot_rows["n_cells"],
        plot_rows["peak_rss_mb"],
        marker="s",
        color=PALETTE["gold"],
        lw=1.6,
        label="peak RSS",
    )
    ax.set_title("Tonsil subset scaling")
    ax.set_xlabel("Cells")
    ax.set_ylabel("Candidate edges")
    ax2.set_ylabel("Peak RSS (MB)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(6, 6))
    ax.grid(axis="y", color=PALETTE["light_gray"], lw=0.8)
    for _, row in plot_rows.iterrows():
        ax.text(
            row["n_cells"],
            row["n_candidate_edges"],
            f"{row['wall_clock_seconds']:.0f}s\npos top10={int(row['n_positive_controls_top_decile'])}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    handles = [
        plt.Line2D([0], [0], color=PALETTE["mid_blue"], marker="o", lw=1.8, label="candidate edges"),
        plt.Line2D([0], [0], color=PALETTE["gold"], marker="s", lw=1.6, label="peak RSS"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=7)
    largest = plot_rows.iloc[-1]
    ax.text(
        0.98,
        0.05,
        f"largest: {int(largest['n_cells'])} cells\n{int(largest['n_candidate_edges']):,} edges\nneg top10={int(largest['n_negative_controls_top_decile'])}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "#FFFFFF", "edgecolor": PALETTE["light_gray"]},
    )


def _panel_tonsil_baseline_matched(ax, baseline: pd.DataFrame) -> None:
    plot_rows = baseline.copy()
    order = ["spatialself_wavelet_calibrated", "commot", "cellchat", "spatalk", "cellnest"]
    plot_rows["_priority"] = plot_rows["method"].map({name: idx for idx, name in enumerate(order)}).fillna(99)
    plot_rows = plot_rows.sort_values("_priority")
    x = np.arange(len(plot_rows))
    colors = [
        PALETTE["green"] if str(value).lower() == "true" else PALETTE["gray"]
        for value in plot_rows["rank_validation_pass"]
    ]
    ax.bar(
        x - 0.18,
        plot_rows["n_positive_controls_top_decile"].astype(float),
        width=0.34,
        color=colors,
        label="positive top decile",
    )
    ax.bar(
        x + 0.18,
        plot_rows["n_negative_controls_top_decile"].astype(float),
        width=0.34,
        color=PALETTE["red"],
        alpha=0.82,
        label="negative top decile",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(plot_rows["display_label"], rotation=28, ha="right", fontsize=7)
    ax.set_ylabel("Control LR count")
    ax.set_ylim(0, max(6.5, plot_rows["n_positive_controls_top_decile"].max() + 1))
    ax.set_title("Tonsil baseline-matched controls")
    ax.legend(loc="upper right", fontsize=7)
    for idx, row in enumerate(plot_rows.itertuples(index=False)):
        best_pos = _short_pair(getattr(row, "best_positive_control"))
        best_neg = _short_pair(getattr(row, "best_negative_control"))
        marker = "pass" if str(getattr(row, "rank_validation_pass")).lower() == "true" else "limited"
        if int(getattr(row, "n_unique_scores")) <= 1:
            marker = "flat scores"
        ax.text(
            idx,
            0.18,
            f"{marker}\n+{best_pos}\n-{best_neg}",
            ha="center",
            va="bottom",
            fontsize=6.7,
            rotation=0,
        )
    profile = plot_rows.iloc[0]
    ax.text(
        0.02,
        0.96,
        f"{int(profile['overview_n_cells'])} cells\n{int(profile['overview_n_candidate_edges']):,} edges\n"
        "control-rank validation",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "#FFFFFF", "edgecolor": PALETTE["light_gray"]},
    )


def _panel_tonsil_ccc_modules(ax, modules: pd.DataFrame) -> None:
    plot_rows = modules.sort_values("ccc_module").copy()
    x = np.arange(len(plot_rows))
    width = 0.34
    ax.bar(
        x - width / 2,
        plot_rows["n_positive_controls"].astype(float),
        width=width,
        color=PALETTE["green"],
        edgecolor="none",
        label="positive controls",
    )
    ax.bar(
        x + width / 2,
        plot_rows["n_housekeeping_negative_controls"].astype(float),
        width=width,
        color=PALETTE["red"],
        edgecolor="none",
        alpha=0.86,
        label="housekeeping negatives",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(plot_rows["display_module"], fontsize=7.6)
    ax.set_ylabel("LR count in module")
    ax.set_title("Tonsil CCC module/control separation")
    ax.set_ylim(0, max(4.4, plot_rows[["n_positive_controls", "n_housekeeping_negative_controls"]].max().max() + 0.9))
    ax.legend(loc="upper right", fontsize=7)
    for idx, row in enumerate(plot_rows.itertuples(index=False)):
        pos_fdr = getattr(row, "positive_control_bh_fdr", np.nan)
        neg_fdr = getattr(row, "housekeeping_negative_bh_fdr", np.nan)
        fdr_parts = []
        if pd.notna(pos_fdr) and float(pos_fdr) <= 0.15:
            fdr_parts.append(f"pos FDR={float(pos_fdr):.3g}")
        if pd.notna(neg_fdr) and float(neg_fdr) <= 0.15:
            fdr_parts.append(f"neg FDR={float(neg_fdr):.3g}")
        if not fdr_parts:
            fdr_parts.append("FDR n.s.")
        ax.text(
            idx,
            0.05,
            f"{getattr(row, 'display_mechanism')}\n{'; '.join(fdr_parts)}",
            ha="center",
            va="bottom",
            fontsize=6.7,
        )
        ax.text(
            idx,
            max(float(getattr(row, "n_positive_controls")), float(getattr(row, "n_housekeeping_negative_controls"))) + 0.18,
            _short_pair(getattr(row, "top_lr_pairs")),
            ha="center",
            va="bottom",
            fontsize=6.5,
            rotation=20,
        )
    overview = plot_rows.iloc[0]
    ax.text(
        0.02,
        0.96,
        f"{int(overview['overview_n_modules'])} modules, {int(overview['overview_n_lr_pairs'])} LR pairs\n"
        "single 720-cell subset\nspatial concentration weak",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "#FFFFFF", "edgecolor": PALETTE["light_gray"]},
    )


def _panel_interpretability(ax, interp: pd.DataFrame) -> None:
    order = ["housekeeping_negative_control", "tonsil_positive_prior", "background_lr"]
    plot_rows = interp.set_index("control_class").loc[[x for x in order if x in set(interp["control_class"])]].reset_index()
    labels = [x.replace("_", "\n") for x in plot_rows["control_class"]]
    colors = [
        PALETTE["red"] if "negative" in value else (PALETTE["green"] if "positive" in value else PALETTE["gray"])
        for value in plot_rows["control_class"]
    ]
    x = np.arange(len(plot_rows))
    ax.bar(x, plot_rows["mean_residual_spearman"], color=colors, edgecolor="none", width=0.68)
    ax.axhline(0, color=PALETTE["dark"], lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_title("Interpretability diagnostics")
    ax.set_ylabel("Mean residual Spearman")
    ax.set_ylim(-0.04, max(0.18, plot_rows["mean_residual_spearman"].max() + 0.04))
    emb = interp.iloc[0]
    ax.text(
        0.02,
        0.96,
        "LR embedding weak:\n"
        f"sil(mech)={emb['embedding_silhouette_mechanism']:.3f}\n"
        f"sig enrich={int(emb['embedding_n_significant_cluster_enrichments_fdr_0_10'])}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#FFFFFF", "edgecolor": PALETTE["light_gray"]},
    )


def _panel_mechanism_category(ax, mechanism: pd.DataFrame) -> None:
    plot_rows = mechanism.sort_values("mean_residual_spearman", ascending=True).copy()
    y = np.arange(len(plot_rows))
    colors = [
        PALETTE["red"]
        if str(name) == "false_positive_high_expression"
        else (PALETTE["green"] if pos > 0 else PALETTE["mid_blue"])
        for name, pos in zip(plot_rows["mechanism"], plot_rows["n_positive_controls"])
    ]
    ax.barh(y, plot_rows["mean_residual_spearman"], color=colors, edgecolor="none", alpha=0.92)
    ax.axvline(0, color=PALETTE["dark"], lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_rows["display_label"], fontsize=7.2)
    ax.set_xlabel("Mean residual Spearman")
    ax.set_title("Mechanism-level interpretability")
    xmax = max(0.18, float(plot_rows["mean_residual_spearman"].max()) + 0.035)
    xmin = min(-0.04, float(plot_rows["mean_residual_spearman"].min()) - 0.025)
    ax.set_xlim(xmin, xmax)
    for idx, row in enumerate(plot_rows.itertuples(index=False)):
        sig = int(getattr(row, "n_residual_bh_fdr_0_10"))
        lr_count = int(getattr(row, "n_lr_pairs"))
        ax.text(
            getattr(row, "mean_residual_spearman") + 0.004,
            idx,
            f"{sig}/{lr_count} sig",
            va="center",
            ha="left",
            fontsize=6.9,
        )
    overview = plot_rows.iloc[0]
    ax.text(
        0.98,
        0.05,
        "mixed evidence\n"
        f"pos-neg={overview['overview_positive_minus_housekeeping_residual_spearman']:.3f}\n"
        f"cluster purity={overview['overview_embedding_cluster_category_purity']:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "#FFFFFF", "edgecolor": PALETTE["light_gray"]},
    )


def _panel_lymph_node_validation(ax, lymph: pd.DataFrame) -> None:
    plot_rows = lymph.sort_values("global_rank").copy()
    y = np.arange(len(plot_rows))
    colors = [
        PALETTE["green"] if str(true_lr).lower() == "true" else PALETTE["red"]
        for true_lr in plot_rows["true_lr"]
    ]
    ax.barh(y, plot_rows["global_rank"].astype(float), color=colors, edgecolor="none", alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_rows["display_label"], fontsize=7)
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel("Global rank (1 = highest)")
    ax.set_title("Lymph node injected controls")
    ax.set_xlim(max(5.6, plot_rows["global_rank"].max() + 0.6), 0)
    for idx, row in enumerate(plot_rows.itertuples(index=False)):
        enrichment = getattr(row, "best_region_enrichment_score")
        spearman = getattr(row, "downstream_spearman_r")
        status = getattr(row, "downstream_status")
        spearman_text = f"rho={spearman:.2f}" if pd.notna(spearman) else str(status).replace("_", " ")
        ax.text(
            0.02,
            idx,
            f"enr={enrichment:.3f}; {spearman_text}",
            va="center",
            ha="left",
            fontsize=7,
            color=PALETTE["dark"],
        )
    summary = plot_rows.iloc[0]
    ax.text(
        0.98,
        0.06,
        "real-background\ninjected controls\n"
        f"neg top3={int(summary['method_negative_top3_count'])}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "#FFFFFF", "edgecolor": PALETTE["light_gray"]},
    )


def _panel_lymph_node_baseline_matched(ax, baseline: pd.DataFrame) -> None:
    plot_rows = baseline.copy()
    order = [
        "spatialself_wavelet_edge_prior",
        "spatialself_wavelet",
        "spatialself_gat",
        "commot",
        "cellchat",
        "spatalk",
        "cellnest",
    ]
    plot_rows["_priority"] = plot_rows["method"].map({name: idx for idx, name in enumerate(order)}).fillna(99)
    plot_rows = plot_rows.sort_values("_priority")
    x = np.arange(len(plot_rows))
    colors = [
        PALETTE["green"]
        if str(pass_flag).lower() == "true"
        else PALETTE["red"]
        for pass_flag in plot_rows["rank_validation_pass"]
    ]
    ax.bar(x - 0.18, plot_rows["mean_positive_rank"].astype(float), width=0.34, color=colors, alpha=0.9, label="positive mean rank")
    ax.bar(x + 0.18, plot_rows["mean_negative_rank"].astype(float), width=0.34, color=PALETTE["gray"], alpha=0.78, label="negative mean rank")
    ax.invert_yaxis()
    ax.set_xticks(x)
    ax.set_xticklabels(plot_rows["display_label"], rotation=28, ha="right", fontsize=7)
    ax.set_ylabel("Global LR rank (lower is better)")
    ax.set_title("Lymph node baseline-matched controls")
    ax.set_ylim(5.35, 0.6)
    ax.legend(loc="upper right", fontsize=7)
    for idx, row in enumerate(plot_rows.itertuples(index=False)):
        neg_top3 = int(getattr(row, "negative_top3_count"))
        spearman = getattr(row, "mean_positive_downstream_spearman")
        marker = "pass" if str(getattr(row, "rank_validation_pass")).lower() == "true" else f"neg top3={neg_top3}"
        ax.text(
            idx,
            5.12,
            f"{marker}\nrho={spearman:.2f}",
            ha="center",
            va="top",
            fontsize=6.7,
        )
    profile = plot_rows.iloc[0]
    ax.text(
        0.01,
        0.04,
        f"{int(profile['overview_n_cells'])} cells, {int(profile['overview_n_candidate_edges']):,} edges\n"
        "single injected-control replicate",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "#FFFFFF", "edgecolor": PALETTE["light_gray"]},
    )


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.9,
            "axes.titleweight": "semibold",
            "axes.titlesize": 10,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _read_required(path: str, required_columns: list[str]) -> pd.DataFrame:
    full_path = PROJECT_ROOT / path
    if not full_path.exists():
        raise FileNotFoundError(f"Required Goal 6 input is missing: {path}")
    df = pd.read_csv(full_path)
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"Required columns missing from {path}: {missing}")
    return df


def _display_model_label(value: str) -> str:
    mapping = {
        "wavelet_mechanism_niche_lr_prior_lrwise_ranking": "wavelet default",
        "wavelet_mechanism_niche_edge_prior_ranking": "edge prior",
        "phase2_weak_mining_edge_prior_wavelet": "weak mining",
        "phase2_focal_weak_mining_edge_prior_wavelet": "focal weak mining",
    }
    return mapping.get(str(value), str(value))


def _display_tradeoff_label(value: str) -> str:
    mapping = {
        "edge_prior_ranking": "edge prior",
        "focal_weak_mining_edge_prior": "focal weak mining",
    }
    return mapping.get(str(value), str(value))


def _display_lr_balanced_strategy(value: str) -> str:
    mapping = {
        "global_reference": "global q0.95",
        "global_quantile": "global q",
        "lrwise_quantile": "LR-wise q",
        "lrwise_quantile_with_global_floor": "LR-wise q +\nglobal floor",
        "global_reference_plus_lrwise_floor": "global q0.95 +\nLR-wise floor",
        "lrwise_top_k": "LR-wise top-k",
        "global_reference_plus_lr_top_k_floor": "global q0.95 +\nLR top-k floor",
    }
    return mapping.get(str(value), str(value).replace("_", " "))


def _display_null_label(value: str) -> str:
    mapping = {
        "score_permutation": "score null",
        "lr_identity_permutation": "LR identity null",
        "receiver_permutation": "receiver null",
    }
    return mapping.get(str(value), str(value).replace("_", " "))


def _display_group_level(value: str) -> str:
    return str(value).replace("_", "-")


def _display_mechanism_label(value: str) -> str:
    mapping = {
        "false_positive_high_expression": "high-expression\nnegative",
        "juxtacrine_contact": "contact",
        "long_range_chemokine_niche": "long-range\nchemokine",
        "short_medium_paracrine": "short/medium\nparacrine",
    }
    return mapping.get(str(value), str(value).replace("_", "\n"))


def _display_candidate_label(value: str) -> str:
    mapping = {
        "wavelet_calibrated_fdr": "calibrated FDR",
        "wavelet_edge_prior": "edge prior",
        "wavelet_default": "wavelet default",
        "gat_reference": "GAT reference",
        "wavelet_focal_weak_mining": "focal mining",
        "wavelet_weak_mining": "weak mining",
        "no_lr_identity": "no LR ID",
        "no_distance": "no distance",
        "no_expression": "no expression",
        "hybrid_wavelet_gat": "hybrid wavelet-GAT",
    }
    return mapping.get(str(value), str(value))


def _display_ablation_label(value: str) -> str:
    mapping = {
        "gat_hard_expression_distance": "GAT",
        "wavelet_hard_expression_distance": "wavelet ref",
        "gcn_hard_expression_distance": "GCN",
        "mlp_hard_expression_distance": "MLP",
        "wavelet_no_lr_embedding": "no LR emb",
        "wavelet_fixed_random_lr_embedding": "fixed random LR",
        "wavelet_one_hot_lr_identity": "one-hot LR",
        "wavelet_no_distance": "no distance",
        "wavelet_no_spatial_graph": "no graph",
        "wavelet_no_expression_features": "no expression",
    }
    return mapping.get(str(value), str(value))


def _ablation_component_group(value: str) -> str:
    text = str(value)
    if "spatial_graph" in text or text.startswith(("gat_", "gcn_", "mlp_")):
        return "architecture_graph"
    if "lr_embedding" in text or "lr_identity" in text:
        return "lr_identity"
    if "distance" in text:
        return "distance"
    if "expression" in text:
        return "expression"
    return "reference"


def _display_method_label(value: str) -> str:
    mapping = {
        "spatialself_wavelet_calibrated": "SpatialSelf\ncalibrated",
        "spatialself_wavelet_edge_prior": "SpatialSelf\nedge prior",
        "spatialself_wavelet": "SpatialSelf\nwavelet",
        "spatialself_gat": "SpatialSelf\nGAT",
        "commot": "COMMOT",
        "cellchat": "CellChat",
        "spatalk": "SpaTalk",
        "cellnest": "CellNEST",
        "deeptalk": "DeepTalk",
    }
    return mapping.get(str(value), str(value))


def _display_lr_control_label(value: object) -> str:
    text = str(value)
    mapping = {
        "SPP1_CD44_boundary_semisynthetic": "SPP1-CD44",
        "CCL19_CCR7_Tzone_semisynthetic": "CCL19-CCR7",
        "CXCL13_CXCR5_GC_semisynthetic_weak": "CXCL13-CXCR5\nweak",
        "ACTB_EGFR_false_positive_real_background": "ACTB-EGFR\nnegative",
        "GAPDH_ITGB1_false_positive_real_background": "GAPDH-ITGB1\nnegative",
    }
    return mapping.get(text, text.replace("_", "\n"))


def _display_control_type(value: object) -> str:
    mapping = {
        "positive": "positive",
        "positive_weak": "weak positive",
        "negative_high_expression": "high-expression negative",
    }
    return mapping.get(str(value), str(value).replace("_", " "))


def _baseline_matched_method_scope(row: pd.Series) -> str:
    method = str(row.get("method", ""))
    if method.startswith("spatialself"):
        return "candidate-level SpatialSelf row"
    if method == "commot":
        return "candidate-level COMMOT adapter row"
    if method in {"cellchat", "spatalk"}:
        return "cell-type/LR score projected to candidate edges"
    if method == "cellnest":
        return "CellNEST output projected to candidate edges"
    return "retained baseline row"


def _baseline_claim_scope(row: dict[str, object]) -> str:
    method = str(row.get("method", ""))
    status = str(row.get("status", ""))
    level = str(row.get("baseline_evidence_level", ""))
    if method == "deeptalk":
        return "status/projection only; no comparable metric row"
    if status != "completed":
        return f"not claim eligible: {status}"
    if level == "full_controlled_core":
        return "full controlled five-replicate candidate-edge baseline"
    if level == "full_controlled_adapter_light":
        return "adapter-light reproducibility row; show with limitation"
    if level == "single_full_slide_replicate":
        return "single full-slide semi-synthetic replicate"
    return "limited baseline evidence"


def _short_pair(value: object) -> str:
    text = str(value)
    if len(text) <= 18:
        return text
    return text[:16] + "..."


if __name__ == "__main__":
    main()
