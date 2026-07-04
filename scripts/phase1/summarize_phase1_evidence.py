#!/usr/bin/env python
"""Build a compact evidence matrix for the eight Phase 1 goals."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs/phase1/phase1_evidence_matrix",
        help="Directory for phase1_evidence_matrix.csv/json/md.",
    )
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_evidence_matrix()

    csv_path = output_dir / "phase1_evidence_matrix.csv"
    json_path = output_dir / "phase1_evidence_matrix.json"
    md_path = output_dir / "phase1_evidence_matrix.md"
    fieldnames = [
        "goal_id",
        "goal",
        "evidence_status",
        "primary_artifacts",
        "key_findings",
        "caveats",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=True), encoding="utf-8")
    md_path.write_text(_to_markdown(rows), encoding="utf-8")

    print(f"Evidence matrix CSV:  {csv_path}")
    print(f"Evidence matrix JSON: {json_path}")
    print(f"Evidence matrix MD:   {md_path}")
    print(f"Rows: {len(rows)}")


def build_evidence_matrix() -> list[dict[str, str]]:
    return [
        _controlled_benchmark_row(),
        _semisynthetic_row(),
        _baseline_row(),
        _model_variants_row(),
        _ablation_row(),
        _fdr_row(),
        _real_validation_row(),
        _runtime_row(),
    ]


def _controlled_benchmark_row() -> dict[str, str]:
    root = PROJECT_ROOT / "outputs/phase1/exp01_controlled_benchmark/datasets"
    mechanisms = _read_csv(root / "controlled_mechanism_summary.csv")
    validation = _read_csv(root / "validation_summary.csv")
    mechanism_names = sorted({row.get("mechanism", "") for row in mechanisms})
    contact_max = _max(
        _to_float(row.get("positive_max_distance", "nan"))
        for row in mechanisms
        if row.get("mechanism") == "juxtacrine_contact"
    )
    contact_median = _mean(
        _to_float(row.get("positive_median_distance", "nan"))
        for row in mechanisms
        if row.get("mechanism") == "juxtacrine_contact"
    )
    long_median = _mean(
        _to_float(row.get("positive_median_distance", "nan"))
        for row in mechanisms
        if row.get("mechanism") == "long_range_chemokine_niche"
    )
    fp_positive = sum(
        int(_to_float(row.get("positives", "0")) or 0)
        for row in mechanisms
        if row.get("mechanism") == "false_positive_high_expression"
    )
    n_reps = len({row.get("replicate", "") for row in validation}) if validation else 0
    return _row(
        "1",
        "Controlled ground-truth CCC benchmark",
        "complete",
        [
            "outputs/phase1/exp01_controlled_benchmark/datasets/validation_summary.csv",
            "outputs/phase1/exp01_controlled_benchmark/datasets/controlled_mechanism_summary.csv",
            "outputs/phase1/exp01_controlled_benchmark/datasets/controlled_lr_summary.csv",
        ],
        (
            f"{n_reps} replicates; mechanisms={mechanism_names}; "
            f"contact median/max distance={contact_median:.4f}/{contact_max:.4f}; "
            f"long-range median distance={long_median:.4f}; high-expression FP positives={fp_positive}."
        ),
        "Synthetic benchmark is now mechanism-driven, but still needs external real-data validation for biological claims.",
    )


def _semisynthetic_row() -> dict[str, str]:
    root = PROJECT_ROOT / "outputs/phase1/exp02_semisynthetic_real_background/full_slide_sparse_datasets"
    realism_root = PROJECT_ROOT / "outputs/phase1/exp02_semisynthetic_real_background/full_slide_sparse_realism"
    validation = _first(_read_csv(root / "validation_summary.csv"))
    mechanisms = _read_csv(root / "semisynthetic_mechanism_summary.csv")
    realism = _read_csv(realism_root / "semisynthetic_realism_summary.csv")
    long_median = _first_float(mechanisms, "mechanism", "long_range_chemokine_niche", "positive_median_distance")
    short_median = _first_float(mechanisms, "mechanism", "short_medium_paracrine", "positive_median_distance")
    fp_labels = _first_float(mechanisms, "mechanism", "false_positive_high_expression", "false_positive_labels")
    bg_mean_corr = _first_float(realism, "metric", "gene_mean_pearson_background", "value")
    bg_sparsity_corr = _first_float(realism, "metric", "gene_sparsity_pearson_background", "value")
    moran_corr = _first_float(realism, "metric", "moran_pearson_background", "value")
    lr_delta = _first_float(realism, "metric", "median_mean_delta_lr_genes", "value")
    return _row(
        "2",
        "Real ST background semi-synthetic benchmark",
        "complete",
        [
            "outputs/phase1/exp02_semisynthetic_real_background/full_slide_sparse_datasets/validation_summary.csv",
            "outputs/phase1/exp02_semisynthetic_real_background/full_slide_sparse_datasets/semisynthetic_injection_summary.csv",
            "outputs/phase1/exp02_semisynthetic_real_background/full_slide_sparse_datasets/semisynthetic_mechanism_summary.csv",
            "outputs/phase1/exp02_semisynthetic_real_background/full_slide_sparse_realism/semisynthetic_realism_summary.csv",
            "outputs/phase1/exp02_semisynthetic_real_background/datasets/validation_summary.csv",
        ],
        (
            f"{_fmt(validation, 'cells')} spots, {_fmt(validation, 'expression_genes')} genes, "
            f"{_fmt(validation, 'candidate_interactions')} candidate interactions; "
            f"{_fmt(validation, 'positive_interactions')} positives, "
            f"{_fmt(validation, 'weak_positive_interactions')} weak positives; "
            f"long/short median distance={long_median:.3f}/{short_median:.3f}; FP labels={fp_labels:.0f}. "
            f"Realism: background mean/sparsity/Moran correlations={bg_mean_corr:.4f}/{bg_sparsity_corr:.4f}/{moran_corr:.4f}; "
            f"median LR mean shift={lr_delta:.3f}."
        ),
        "Full-slide sparse evidence is lymph-node based; tonsil/melanoma backgrounds and real high-resolution platforms remain next-stage extensions.",
    )


def _baseline_row() -> dict[str, str]:
    root = PROJECT_ROOT / "outputs/phase1/exp03_baseline_comparison/full_controlled_retained_unified"
    sparse_root = PROJECT_ROOT / "outputs/phase1/exp03_baseline_comparison/full_slide_sparse_semisynthetic_retained"
    summary = _read_csv(root / "baseline_metrics_summary.csv")
    status = _read_csv(root / "baseline_status.csv")
    sparse_summary = _read_csv(sparse_root / "baseline_metrics_summary.csv")
    sparse_status = _read_csv(sparse_root / "baseline_status.csv")
    methods = sorted({row.get("method", "") for row in status})
    metric_methods = sorted({row.get("method", "") for row in summary})
    sparse_methods = sorted({row.get("method", "") for row in sparse_status})
    niches_status = _first_value(status, "method", "niches", "status")
    edge = _first_row(summary, "method", "spatialself_wavelet_edge_prior")
    default = _first_row(summary, "method", "spatialself_wavelet")
    commot = _first_row(summary, "method", "commot")
    sparse_commot = _first_row(sparse_summary, "method", "commot")
    sparse_default = _first_row(sparse_summary, "method", "spatialself_wavelet")
    sparse_edge = _first_row(sparse_summary, "method", "spatialself_wavelet_edge_prior")
    sparse_cellnest = _first_row(sparse_summary, "method", "cellnest")
    return _row(
        "3",
        "Systematic baseline comparison framework",
        "complete",
        [
            "outputs/phase1/exp03_baseline_comparison/full_controlled_retained_unified/baseline_metrics_summary.csv",
            "outputs/phase1/exp03_baseline_comparison/full_controlled_retained_unified/baseline_status.csv",
            "outputs/phase1/exp03_baseline_comparison/full_controlled_retained_unified/manifest.json",
            "outputs/phase1/exp03_baseline_comparison/full_slide_sparse_semisynthetic_retained/baseline_metrics_summary.csv",
        ],
        (
            f"Status methods={methods}; metric methods={metric_methods}; NICHES={niches_status}; "
            f"default/edge-prior AUPR={_to_float(default.get('aupr_mean')):.3f}/{_to_float(edge.get('aupr_mean')):.3f}; "
            f"COMMOT weak recall={_to_float(commot.get('weak_signal_recall_at_threshold_mean')):.3f}. "
            f"Full-slide sparse methods={sparse_methods}; COMMOT/SpatialSelf AUPR="
            f"{_to_float(sparse_commot.get('aupr_mean')):.3f}/{_to_float(sparse_default.get('aupr_mean')):.3f}; "
            f"edge-prior weak recall={_to_float(sparse_edge.get('weak_signal_recall_at_threshold_mean')):.3f}; "
            f"CellNEST FDR={_to_float(sparse_cellnest.get('empirical_fdr_at_threshold_mean')):.3f}."
        ),
        "CellChat/SpaTalk/CellNEST are adapter-light reproducibility checks; DeepTalk is projection/model-required unless model files are supplied. The expression-distance baseline is intentionally removed by user decision.",
    )


def _model_variants_row() -> dict[str, str]:
    path = PROJECT_ROOT / "outputs/phase1/exp04_model_ablation/full_architecture_identity_component_ablation_epoch10/spatialself_ablation_summary.csv"
    rows = _read_csv(path)
    variants = sorted(row.get("ablation", "") for row in rows)
    wavelet = _first_row(rows, "ablation", "wavelet_hard_expression_distance")
    gat = _first_row(rows, "ablation", "gat_hard_expression_distance")
    return _row(
        "4",
        "Replaceable model structures",
        "complete",
        [
            "src/spitialself/models.py",
            "src/spitialself/model_factory.py",
            "outputs/phase1/exp04_model_ablation/full_architecture_identity_component_ablation_epoch10/spatialself_ablation_summary.csv",
        ],
        (
            f"Registered wavelet/GAT/GCN/MLP/SAGE variants; full architecture panel has {len(variants)} ablations. "
            f"Wavelet AUPR={_to_float(wavelet.get('aupr_mean')):.3f}; GAT AUPR={_to_float(gat.get('aupr_mean')):.3f}."
        ),
        "Current architecture comparison shows tradeoffs rather than a universal winner; model choice should remain benchmark-driven.",
    )


def _ablation_row() -> dict[str, str]:
    architecture = _read_csv(
        PROJECT_ROOT / "outputs/phase1/exp04_model_ablation/full_architecture_identity_component_ablation_epoch10/spatialself_ablation_summary.csv"
    )
    edge = _read_csv(
        PROJECT_ROOT / "outputs/phase1/exp04_model_ablation/full_edge_prior_focused_ablation_epoch10/spatialself_ablation_summary.csv"
    )
    stress = _read_csv(
        PROJECT_ROOT / "outputs/phase1/exp05_weak_signal_false_positive/full_controlled_spatialself_stress/weak_signal_false_positive_metrics.csv"
    )
    edge_candidate = _first_row(edge, "ablation", "wavelet_mechanism_niche_edge_prior_ranking")
    default = _first_row(edge, "ablation", "wavelet_mechanism_niche_lr_prior_lrwise_ranking")
    stress_methods = sorted({row.get("method", "") for row in stress})
    return _row(
        "5",
        "Ablation and weak-signal/false-positive stress framework",
        "complete",
        [
            "outputs/phase1/exp04_model_ablation/full_architecture_identity_component_ablation_epoch10/spatialself_ablation_summary.csv",
            "outputs/phase1/exp04_model_ablation/full_edge_prior_focused_ablation_epoch10/spatialself_ablation_summary.csv",
            "outputs/phase1/exp05_weak_signal_false_positive/full_controlled_spatialself_stress/weak_signal_false_positive_metrics.csv",
        ],
        (
            f"{len(architecture)} identity/architecture/component ablations; stress methods={stress_methods}; "
            f"default/edge-prior AUPR={_to_float(default.get('aupr_mean')):.3f}/{_to_float(edge_candidate.get('aupr_mean')):.3f}; "
            f"default/edge-prior weak recall={_to_float(default.get('weak_signal_recall_at_threshold_mean')):.3f}/{_to_float(edge_candidate.get('weak_signal_recall_at_threshold_mean')):.3f}."
        ),
        "Some ablations improve AUPR while losing weak-signal recall, so results should be framed as component tradeoffs.",
    )


def _fdr_row() -> dict[str, str]:
    root = PROJECT_ROOT / "outputs/phase1/exp06_permutation_fdr/full_controlled_spatialself_panel"
    sparse_root = PROJECT_ROOT / "outputs/phase1/exp06_permutation_fdr/full_slide_sparse_semisynthetic_panel"
    summary = _read_csv(root / "permutation_fdr_panel_summary.csv")
    method_summary = _read_csv(root / "permutation_fdr_panel_method_summary.csv")
    sparse_summary = _read_csv(sparse_root / "permutation_fdr_panel_summary.csv")
    sparse_method_summary = _read_csv(sparse_root / "permutation_fdr_panel_method_summary.csv")
    methods = sorted({row.get("method_panel", "") for row in summary})
    nulls = sorted({row.get("null_model", "") for row in summary})
    reps = sorted({row.get("replicate", "") for row in summary})
    discoveries = sum(int(_to_float(row.get("n_significant_edges", "0")) or 0) for row in summary)
    sparse_methods = sorted({row.get("method_panel", "") for row in sparse_summary})
    sparse_nulls = sorted({row.get("null_model", "") for row in sparse_summary})
    sparse_discoveries = sum(
        int(_to_float(row.get("n_significant_edges", "0")) or 0) for row in sparse_summary
    )
    return _row(
        "6",
        "Permutation p-value and BH-FDR framework",
        "complete",
        [
            "outputs/phase1/exp06_permutation_fdr/full_controlled_spatialself_panel/permutation_fdr_panel_summary.csv",
            "outputs/phase1/exp06_permutation_fdr/full_controlled_spatialself_panel/permutation_fdr_panel_method_summary.csv",
            "outputs/phase1/exp06_permutation_fdr/full_controlled_spatialself_panel/permutation_fdr_panel_manifest.json",
            "outputs/phase1/exp06_permutation_fdr/full_slide_sparse_semisynthetic_panel/permutation_fdr_panel_summary.csv",
        ],
        (
            f"Methods={methods}; nulls={nulls}; replicates={len(reps)}; panel rows={len(summary)}; "
            f"method-null rows={len(method_summary)}; significant edges at BH-FDR 0.05={discoveries}. "
            f"Full-slide sparse methods={sparse_methods}; nulls={sparse_nulls}; panel rows={len(sparse_summary)}; "
            f"method-null rows={len(sparse_method_summary)}; significant edges={sparse_discoveries}."
        ),
        "Current FDR procedure is conservative with no discoveries in controlled or full-slide sparse panels; next power improvement should focus on calibration or group/model-rerun nulls.",
    )


def _real_validation_row() -> dict[str, str]:
    root = PROJECT_ROOT / "outputs/phase1/exp07_lymph_node_global_validation"
    default_downstream = _read_csv(root / "full_slide_sparse_downstream_target/spatialself_wavelet/human_lymph_node_visium_replicate_00__spatialself_wavelet_predictions__downstream_target_correlation.csv")
    edge_downstream = _read_csv(root / "full_slide_sparse_downstream_target/spatialself_wavelet_edge_prior/human_lymph_node_visium_replicate_00__spatialself_wavelet_edge_prior_predictions__downstream_target_correlation.csv")
    control = _read_csv(root / "full_slide_sparse_control_summary/real_validation_control_method_summary.csv")
    default_tested = _tested_spearman_mean(default_downstream)
    edge_tested = _tested_spearman_mean(edge_downstream)
    default_control = _first_row(control, "method", "spatialself_wavelet")
    edge_control = _first_row(control, "method", "spatialself_wavelet_edge_prior")
    return _row(
        "7",
        "Human lymph node all-LR global validation",
        "complete",
        [
            "outputs/phase1/exp07_lymph_node_global_validation/full_slide_sparse_region_enrichment",
            "outputs/phase1/exp07_lymph_node_global_validation/full_slide_sparse_downstream_target",
            "outputs/phase1/exp07_lymph_node_global_validation/full_slide_sparse_control_summary/real_validation_control_method_summary.csv",
        ],
        (
            f"Full-slide sparse all 5 injected LR pairs ranked and tested across 4035 spots; "
            f"downstream mean Spearman default/edge-prior={default_tested:.3f}/{edge_tested:.3f}; "
            f"positive-vs-negative mean ranks default={_to_float(default_control.get('mean_positive_rank')):.1f}/{_to_float(default_control.get('mean_negative_rank')):.1f}, "
            f"edge-prior={_to_float(edge_control.get('mean_positive_rank')):.1f}/{_to_float(edge_control.get('mean_negative_rank')):.1f}."
        ),
        "This is full-slide semi-synthetic lymph-node validation; broader real datasets and pathway enrichment are next-stage.",
    )


def _runtime_row() -> dict[str, str]:
    path = PROJECT_ROOT / "outputs/phase1/exp08_runtime_memory/controlled_medium_scaling/runtime_memory_scaling.csv"
    sparse_path = PROJECT_ROOT / "outputs/phase1/exp08_runtime_memory/sparse_spatialself_scaling/runtime_memory_scaling.csv"
    sparse_50k_path = PROJECT_ROOT / "outputs/phase1/exp08_runtime_memory/sparse_50k_spatialself_gat_sparse_knn_profile/runtime_memory_profile.csv"
    sparse_50k_wavelet_path = PROJECT_ROOT / "outputs/phase1/exp08_runtime_memory/sparse_50k_spatialself_wavelet_profile/runtime_memory_profile.csv"
    sparse_50k_highcap_path = PROJECT_ROOT / "outputs/phase1/exp08_runtime_memory/sparse_50k_highcap_spatialself/runtime_memory_scaling.csv"
    sparse_100k_path = PROJECT_ROOT / "outputs/phase1/exp08_runtime_memory/sparse_100k_spatialself_smoke/runtime_memory_scaling.csv"
    full_slide_sparse_path = PROJECT_ROOT / "outputs/phase1/exp08_runtime_memory/full_slide_sparse_semisynthetic_retained_profile/runtime_memory_profile.csv"
    rows = _read_csv(path)
    sparse_rows = _read_csv(sparse_path)
    sparse_50k_rows = _read_csv(sparse_50k_path) + _read_csv(sparse_50k_wavelet_path)
    sparse_50k_highcap_rows = _read_csv(sparse_50k_highcap_path)
    sparse_100k_rows = _read_csv(sparse_100k_path)
    full_slide_sparse_rows = _read_csv(full_slide_sparse_path)
    methods = sorted({row.get("method", "") for row in rows})
    max_edges = _max(_to_float(row.get("n_candidate_edges", "nan")) for row in rows)
    scale1000 = [row for row in rows if int(_to_float(row.get("scale_n_cells", "0")) or 0) == 1000]
    sparse_methods = sorted({row.get("method", "") for row in sparse_rows})
    sparse_scales = sorted({
        int(_to_float(row.get("scale_n_cells", "0")) or 0)
        for row in sparse_rows
    })
    sparse_10k = [
        row
        for row in sparse_rows
        if int(_to_float(row.get("scale_n_cells", "0")) or 0) == 10000
    ]
    sparse_10k_edges = _max(_to_float(row.get("n_candidate_edges", "nan")) for row in sparse_10k)
    sparse_10k_time = {
        row.get("method", ""): _to_float(row.get("wall_clock_seconds", "nan"))
        for row in sparse_10k
    }
    sparse_10k_memory = {
        row.get("method", ""): _to_float(row.get("peak_cpu_memory_mb", "nan"))
        for row in sparse_10k
    }
    sparse_50k_edges = _max(_to_float(row.get("n_candidate_edges", "nan")) for row in sparse_50k_rows)
    sparse_50k_time = {
        row.get("method", ""): _to_float(row.get("wall_clock_seconds", "nan"))
        for row in sparse_50k_rows
    }
    sparse_50k_memory = {
        row.get("method", ""): _to_float(row.get("peak_cpu_memory_mb", "nan"))
        for row in sparse_50k_rows
    }
    sparse_50k_highcap_edges = _max(_to_float(row.get("n_candidate_edges", "nan")) for row in sparse_50k_highcap_rows)
    sparse_50k_highcap_time = {
        row.get("method", ""): _to_float(row.get("wall_clock_seconds", "nan"))
        for row in sparse_50k_highcap_rows
    }
    sparse_50k_highcap_memory = {
        row.get("method", ""): _to_float(row.get("peak_cpu_memory_mb", "nan"))
        for row in sparse_50k_highcap_rows
    }
    sparse_100k_edges = _max(_to_float(row.get("n_candidate_edges", "nan")) for row in sparse_100k_rows)
    sparse_100k_time = {
        row.get("method", ""): _to_float(row.get("wall_clock_seconds", "nan"))
        for row in sparse_100k_rows
    }
    sparse_100k_memory = {
        row.get("method", ""): _to_float(row.get("peak_cpu_memory_mb", "nan"))
        for row in sparse_100k_rows
    }
    full_slide_completed = [
        row for row in full_slide_sparse_rows if row.get("status") == "completed"
    ]
    full_slide_status = {
        row.get("method", ""): row.get("status", "")
        for row in full_slide_sparse_rows
    }
    full_slide_cells = _max(
        _to_float(row.get("n_cells_or_spots", "nan")) for row in full_slide_sparse_rows
    )
    full_slide_edges = _max(
        _to_float(row.get("n_candidate_edges", "nan")) for row in full_slide_sparse_rows
    )
    full_slide_time = {
        row.get("method", ""): _to_float(row.get("wall_clock_seconds", "nan"))
        for row in full_slide_completed
    }
    full_slide_memory = {
        row.get("method", ""): _to_float(row.get("peak_cpu_memory_mb", "nan"))
        for row in full_slide_completed
    }
    timing = {
        row.get("method", ""): _to_float(row.get("wall_clock_seconds", "nan"))
        for row in scale1000
    }
    memory = {
        row.get("method", ""): _to_float(row.get("peak_cpu_memory_mb", "nan"))
        for row in scale1000
    }
    return _row(
        "8",
        "Runtime and memory benchmark",
        "complete",
        [
            "outputs/phase1/exp08_runtime_memory/controlled_medium_scaling/runtime_memory_scaling.csv",
            "outputs/phase1/exp08_runtime_memory/controlled_medium_scaling/runtime_scaling_manifest.json",
            "outputs/phase1/exp08_runtime_memory/sparse_spatialself_scaling/runtime_memory_scaling.csv",
            "outputs/phase1/exp08_runtime_memory/sparse_50k_spatialself_gat_sparse_knn_profile/runtime_memory_profile.csv",
            "outputs/phase1/exp08_runtime_memory/sparse_50k_spatialself_wavelet_profile/runtime_memory_profile.csv",
            "outputs/phase1/exp08_runtime_memory/sparse_50k_highcap_spatialself/runtime_memory_scaling.csv",
            "outputs/phase1/exp08_runtime_memory/sparse_100k_spatialself_smoke/runtime_memory_scaling.csv",
            "outputs/phase1/exp08_runtime_memory/full_slide_sparse_semisynthetic_retained_profile/runtime_memory_profile.csv",
        ],
        (
            f"Methods={methods}; scales=500/1000 cells; max candidate edges={max_edges:.0f}; "
            f"1000-cell wall-clock seconds={_format_map(timing)}; 1000-cell CPU MB={_format_map(memory)}. "
            f"Sparse SpatialSelf methods={sparse_methods}; sparse scales={sparse_scales}; "
            f"10k capped sparse edges={sparse_10k_edges:.0f}; 10k seconds={_format_map(sparse_10k_time)}; "
            f"10k CPU MB={_format_map(sparse_10k_memory)}. "
            f"After KD-tree KNN, 50k capped sparse edges={sparse_50k_edges:.0f}; "
            f"50k seconds={_format_map(sparse_50k_time)}; "
            f"50k CPU MB={_format_map(sparse_50k_memory)}. "
            f"50k high-cap sparse edges={sparse_50k_highcap_edges:.0f}; "
            f"50k high-cap seconds={_format_map(sparse_50k_highcap_time)}; "
            f"50k high-cap CPU MB={_format_map(sparse_50k_highcap_memory)}. "
            f"100k capped sparse edges={sparse_100k_edges:.0f}; "
            f"100k seconds={_format_map(sparse_100k_time)}; "
            f"100k CPU MB={_format_map(sparse_100k_memory)}. "
            f"Full-slide sparse retained profile: {full_slide_cells:.0f} spots, "
            f"{full_slide_edges:.0f} candidate edges, statuses={full_slide_status}; "
            f"seconds={_format_map(full_slide_time)}; CPU MB={_format_map(full_slide_memory)}."
        ),
        "Dense all-pairs evidence covers baselines to 1000 cells. Sparse evidence reaches 100000 cells and 326963 candidate edges for SpatialSelf variants; the full-slide retained-method profile covers the realistic-background 4035-spot benchmark. Real high-resolution subsets and still higher candidate caps remain next-stage scalability work.",
    )


def _row(
    goal_id: str,
    goal: str,
    evidence_status: str,
    primary_artifacts: list[str],
    key_findings: str,
    caveats: str,
) -> dict[str, str]:
    return {
        "goal_id": goal_id,
        "goal": goal,
        "evidence_status": evidence_status,
        "primary_artifacts": "; ".join(primary_artifacts),
        "key_findings": key_findings,
        "caveats": caveats,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _first(rows: list[dict[str, str]]) -> dict[str, str]:
    return rows[0] if rows else {}


def _first_row(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    return {}


def _first_value(rows: list[dict[str, str]], key: str, value: str, column: str) -> str:
    return _first_row(rows, key, value).get(column, "")


def _first_float(rows: list[dict[str, str]], key: str, value: str, column: str) -> float:
    return _to_float(_first_value(rows, key, value, column))


def _fmt(row: dict[str, str], column: str) -> str:
    value = _to_float(row.get(column, "nan"))
    if not math.isfinite(value):
        return "NA"
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}"
    return f"{value:.3f}"


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _finite(values) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def _mean(values) -> float:
    finite = _finite(values)
    return sum(finite) / len(finite) if finite else float("nan")


def _max(values) -> float:
    finite = _finite(values)
    return max(finite) if finite else float("nan")


def _tested_spearman_mean(rows: list[dict[str, str]]) -> float:
    values = [
        _to_float(row.get("spearman_r", "nan"))
        for row in rows
        if row.get("status") == "tested"
    ]
    return _mean(values)


def _format_map(values: dict[str, float]) -> str:
    parts = []
    for key in sorted(values):
        value = values[key]
        parts.append(f"{key}:{value:.2f}" if math.isfinite(value) else f"{key}:NA")
    return "{" + ", ".join(parts) + "}"


def _to_markdown(rows: list[dict[str, str]]) -> str:
    lines = [
        "# Phase 1 Evidence Matrix",
        "",
        "| Goal | Status | Primary artifacts | Key findings | Caveats |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(row["goal"]),
                    _md(row["evidence_status"]),
                    _md(row["primary_artifacts"]),
                    _md(row["key_findings"]),
                    _md(row["caveats"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _md(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
