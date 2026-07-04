#!/usr/bin/env python
"""Strict completion gate for the eight Phase 1 goals.

This is intentionally separate from ``audit_phase1.py``. The audit records many
artifact-level facts, including smoke outputs. This gate asks whether the
current repository has enough authoritative evidence to treat the user-defined
Phase 1 objective as complete.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REMOVED_BASELINE_TOKENS = (
    "LR-" + "Coexp",
    "lr_" + "coexp",
    "LR_" + "Coexp",
    "lr" + "coexp",
)
RESIDUE_PATTERN = re.compile("|".join(re.escape(token) for token in REMOVED_BASELINE_TOKENS))
SOURCE_AUDIT_ROOTS = ("README.md", "docs", "configs", "scripts", "src")
THIS_FILE = Path(__file__).resolve()


@dataclass
class GateRow:
    goal_id: str
    goal: str
    requirement: str
    status: str
    evidence: str
    detail: str = ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs/phase1/phase1_completion_gate",
        help="Directory for phase1_completion_gate.csv/json/md.",
    )
    args = parser.parse_args()

    rows: list[GateRow] = []
    rows.extend(_goal1_controlled_benchmark())
    rows.extend(_goal2_semisynthetic_benchmark())
    rows.extend(_goal3_baseline_framework())
    rows.extend(_goal4_model_variants())
    rows.extend(_goal5_ablation_framework())
    rows.extend(_goal6_fdr_framework())
    rows.extend(_goal7_global_validation())
    rows.extend(_goal8_runtime_memory())
    rows.extend(_cross_goal_repository_hygiene())

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "phase1_completion_gate.csv"
    json_path = output_dir / "phase1_completion_gate.json"
    md_path = output_dir / "phase1_completion_gate.md"

    fieldnames = ["goal_id", "goal", "requirement", "status", "evidence", "detail"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    json_path.write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(rows), encoding="utf-8")

    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row.status] = status_counts.get(row.status, 0) + 1
    print(f"Completion gate CSV:  {csv_path}")
    print(f"Completion gate JSON: {json_path}")
    print(f"Completion gate MD:   {md_path}")
    print("Status counts:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    failed = [row for row in rows if row.status != "pass"]
    if failed:
        print("\nFailed rows:")
        for row in failed:
            print(f"  {row.goal_id} | {row.status} | {row.requirement} | {row.detail}")
        raise SystemExit(1)
    print("\nAll Phase 1 completion checks passed.")


def _goal1_controlled_benchmark() -> list[GateRow]:
    goal = "Controlled ground-truth CCC benchmark"
    validation_path = PROJECT_ROOT / "outputs/phase1/exp01_controlled_benchmark/datasets/validation_summary.csv"
    mechanism_path = PROJECT_ROOT / "outputs/phase1/exp01_controlled_benchmark/datasets/controlled_mechanism_summary.csv"
    lr_path = PROJECT_ROOT / "outputs/phase1/exp01_controlled_benchmark/datasets/controlled_lr_summary.csv"
    validation = _read_csv(validation_path)
    mechanisms = _read_csv(mechanism_path)
    lr_rows = _read_csv(lr_path)
    mechanism_names = {row.get("mechanism", "") for row in mechanisms}
    required_mechanisms = {
        "juxtacrine_contact",
        "short_medium_paracrine",
        "long_range_chemokine_niche",
        "false_positive_high_expression",
    }
    false_positive_labels = sum(
        _to_int(row.get("false_positive_labels"))
        for row in mechanisms
        if row.get("mechanism") == "false_positive_high_expression"
    )
    rule_values = [
        _to_float(row.get(column))
        for row in mechanisms
        if row.get("mechanism") != "false_positive_high_expression"
        for column in ("type_rule_positive_fraction", "range_rule_positive_fraction", "niche_rule_positive_fraction")
    ]
    contact_max = _max(
        _to_float(row.get("positive_max_distance"))
        for row in mechanisms
        if row.get("mechanism") == "juxtacrine_contact"
    )
    long_median = _mean(
        _to_float(row.get("positive_median_distance"))
        for row in mechanisms
        if row.get("mechanism") == "long_range_chemokine_niche"
    )
    short_median = _mean(
        _to_float(row.get("positive_median_distance"))
        for row in mechanisms
        if row.get("mechanism") == "short_medium_paracrine"
    )
    return [
        _row(
            "1",
            goal,
            "full controlled benchmark has five replicates with candidate truth tables",
            len(validation) >= 5
            and all(_to_float(row.get("candidate_interactions")) > 0 for row in validation)
            and all(_to_float(row.get("positive_interactions")) > 0 for row in validation),
            validation_path,
            f"n_replicates={len(validation)}",
        ),
        _row(
            "1",
            goal,
            "mechanism truth covers contact, short-range, long-range niche, and high-expression controls",
            required_mechanisms.issubset(mechanism_names),
            mechanism_path,
            f"mechanisms={sorted(mechanism_names)}",
        ),
        _row(
            "1",
            goal,
            "positive labels satisfy non-expression mechanism rules and controls have zero positives",
            rule_values
            and min(value for value in rule_values if math.isfinite(value)) >= 0.999
            and false_positive_labels == 0,
            mechanism_path,
            f"min_rule={min(rule_values) if rule_values else float('nan'):.3f}; false_positive_labels={false_positive_labels}",
        ),
        _row(
            "1",
            goal,
            "distance regimes are explicitly separated",
            math.isfinite(contact_max)
            and math.isfinite(short_median)
            and math.isfinite(long_median)
            and contact_max <= 0.055 + 1e-9
            and long_median > short_median,
            mechanism_path,
            f"contact_max={contact_max:.4f}; short_median={short_median:.4f}; long_median={long_median:.4f}",
        ),
        _row(
            "1",
            goal,
            "LR-level summary records weak signals and false-positive high-expression pairs",
            bool(lr_rows)
            and any(_as_bool(row.get("weak_signal")) and _to_float(row.get("positives")) > 0 for row in lr_rows)
            and any(row.get("mechanism") == "false_positive_high_expression" for row in lr_rows),
            lr_path,
            f"n_lr_rows={len(lr_rows)}",
        ),
    ]


def _goal2_semisynthetic_benchmark() -> list[GateRow]:
    goal = "Real ST background semi-synthetic benchmark"
    root = PROJECT_ROOT / "outputs/phase1/exp02_semisynthetic_real_background/full_slide_sparse_datasets"
    validation_path = root / "validation_summary.csv"
    injection_path = root / "semisynthetic_injection_summary.csv"
    mechanism_path = root / "semisynthetic_mechanism_summary.csv"
    realism_path = PROJECT_ROOT / "outputs/phase1/exp02_semisynthetic_real_background/full_slide_sparse_realism/semisynthetic_realism_summary.csv"
    validation = _first(_read_csv(validation_path))
    injections = _read_csv(injection_path)
    mechanisms = _read_csv(mechanism_path)
    realism = _read_csv(realism_path)
    mechanism_names = {row.get("mechanism", "") for row in mechanisms}
    required_mechanisms = {
        "short_medium_paracrine",
        "long_range_chemokine_niche",
        "false_positive_high_expression",
    }
    fp_labels = _first_float(mechanisms, "mechanism", "false_positive_high_expression", "false_positive_labels")
    bg_mean = _first_float(realism, "metric", "gene_mean_pearson_background", "value")
    bg_sparsity = _first_float(realism, "metric", "gene_sparsity_pearson_background", "value")
    moran = _first_float(realism, "metric", "moran_pearson_background", "value")
    return [
        _row(
            "2",
            goal,
            "full-slide real-background benchmark preserves real coordinate/expression scale",
            _to_float(validation.get("cells")) >= 4000
            and _to_float(validation.get("expression_genes")) >= 2000
            and _to_float(validation.get("candidate_interactions")) >= 300000
            and _to_float(validation.get("regions")) >= 3
            and _to_float(validation.get("cell_types")) >= 4,
            validation_path,
            (
                f"cells={validation.get('cells')}; genes={validation.get('expression_genes')}; "
                f"candidates={validation.get('candidate_interactions')}; regions={validation.get('regions')}; "
                f"cell_types={validation.get('cell_types')}"
            ),
        ),
        _row(
            "2",
            goal,
            "known LR signals and high-expression false positives are injected under mechanism rules",
            required_mechanisms.issubset(mechanism_names)
            and _to_float(validation.get("positive_interactions")) > 0
            and _to_float(validation.get("weak_positive_interactions")) > 0
            and fp_labels == 0,
            mechanism_path,
            f"mechanisms={sorted(mechanism_names)}; fp_labels={fp_labels:.0f}",
        ),
        _row(
            "2",
            goal,
            "injection summary contains true, weak, and false-positive LR controls",
            bool(injections)
            and any(_as_bool(row.get("weak_signal")) and _to_float(row.get("positives")) > 0 for row in injections)
            and any(row.get("mechanism") == "false_positive_high_expression" for row in injections),
            injection_path,
            f"n_injected_lr={len(injections)}",
        ),
        _row(
            "2",
            goal,
            "realism report quantifies preserved background statistics",
            bg_mean >= 0.99 and bg_sparsity >= 0.99 and moran >= 0.99,
            realism_path,
            f"background_mean={bg_mean:.4f}; sparsity={bg_sparsity:.4f}; moran={moran:.4f}",
        ),
    ]


def _goal3_baseline_framework() -> list[GateRow]:
    goal = "Systematic baseline comparison framework"
    status_path = PROJECT_ROOT / "outputs/phase1/exp03_baseline_comparison/full_slide_sparse_semisynthetic_retained/baseline_status.csv"
    metrics_path = PROJECT_ROOT / "outputs/phase1/exp03_baseline_comparison/full_slide_sparse_semisynthetic_retained/baseline_metrics_summary.csv"
    status = _read_csv(status_path)
    metrics = _read_csv(metrics_path)
    statuses = {row.get("method", ""): row.get("status", "") for row in status}
    metric_methods = {row.get("method", "") for row in metrics}
    required_status = {
        "spatialself_wavelet",
        "spatialself_wavelet_edge_prior",
        "spatialself_gat",
        "commot",
        "cellchat",
        "spatalk",
        "cellnest",
        "deeptalk",
        "nichenet",
        "niches",
    }
    required_metrics = {
        "spatialself_wavelet",
        "spatialself_wavelet_edge_prior",
        "spatialself_gat",
        "commot",
        "cellchat",
        "spatalk",
        "cellnest",
        "deeptalk",
    }
    metric_columns = {
        "auroc_mean",
        "aupr_mean",
        "top_k_precision_mean",
        "weak_signal_recall_at_threshold_mean",
        "empirical_fdr_at_threshold_mean",
        "calibration_error_10bin_mean",
    }
    return [
        _row(
            "3",
            goal,
            "retained baseline status table represents all selected methods after expression-distance baseline removal",
            required_status.issubset(statuses)
            and statuses.get("nichenet") == "downstream_validation_only"
            and statuses.get("niches") == "optional_blocked",
            status_path,
            f"statuses={statuses}",
        ),
        _row(
            "3",
            goal,
            "candidate-scoring baselines produce shared AUROC/AUPR/FDR metrics",
            required_metrics.issubset(metric_methods)
            and bool(metrics)
            and metric_columns.issubset(metrics[0]),
            metrics_path,
            f"metric_methods={sorted(metric_methods)}",
        ),
        _row(
            "3",
            goal,
            "removed expression-distance baseline is absent from active source/config/docs and outputs",
            not _removed_baseline_hits()[0] and not _removed_baseline_hits()[1],
            "README.md,docs,configs,scripts,src,outputs/phase1",
            f"source_hits={len(_removed_baseline_hits()[0])}; output_hits={len(_removed_baseline_hits()[1])}",
        ),
    ]


def _goal4_model_variants() -> list[GateRow]:
    goal = "Replaceable model structures"
    factory = PROJECT_ROOT / "src/spitialself/model_factory.py"
    models = PROJECT_ROOT / "src/spitialself/models.py"
    ablation_path = PROJECT_ROOT / "outputs/phase1/exp04_model_ablation/full_architecture_identity_component_ablation_epoch10/spatialself_ablation_summary.csv"
    factory_text = factory.read_text(encoding="utf-8")
    models_text = models.read_text(encoding="utf-8")
    ablations = {row.get("ablation", "") for row in _read_csv(ablation_path)}
    required_models = {
        "spatialself_wavelet",
        "spatialself_gat",
        "spatialself_gcn",
        "spatialself_mlp",
        "spatialself_sage",
    }
    required_ablation_models = {
        "wavelet_hard_expression_distance",
        "gat_hard_expression_distance",
        "gcn_hard_expression_distance",
        "mlp_hard_expression_distance",
    }
    return [
        _row(
            "4",
            goal,
            "model factory registers graph-wavelet, GAT, and additional replaceable variants",
            all(name in factory_text for name in required_models)
            and "GraphWaveletNodeEncoder" in models_text
            and "GATv2NodeEncoder" in models_text,
            f"{factory.relative_to(PROJECT_ROOT)},{models.relative_to(PROJECT_ROOT)}",
            f"required_models={sorted(required_models)}",
        ),
        _row(
            "4",
            goal,
            "full architecture ablation compares wavelet, GAT, GCN, and MLP variants",
            required_ablation_models.issubset(ablations),
            ablation_path,
            f"ablations={sorted(ablations)}",
        ),
    ]


def _goal5_ablation_framework() -> list[GateRow]:
    goal = "Key ablation framework"
    architecture_path = PROJECT_ROOT / "outputs/phase1/exp04_model_ablation/full_architecture_identity_component_ablation_epoch10/spatialself_ablation_summary.csv"
    edge_path = PROJECT_ROOT / "outputs/phase1/exp04_model_ablation/full_edge_prior_focused_ablation_epoch10/spatialself_ablation_summary.csv"
    stress_path = PROJECT_ROOT / "outputs/phase1/exp05_weak_signal_false_positive/full_controlled_spatialself_stress/weak_signal_false_positive_metrics.csv"
    architecture = _read_csv(architecture_path)
    edge = _read_csv(edge_path)
    stress = _read_csv(stress_path)
    ablations = {row.get("ablation", "") for row in architecture}
    edge_ablations = {row.get("ablation", "") for row in edge}
    stress_methods = {row.get("method", "") for row in stress}
    required_ablation_axes = {
        "wavelet_no_lr_embedding",
        "wavelet_fixed_random_lr_embedding",
        "wavelet_one_hot_lr_identity",
        "wavelet_no_distance",
        "wavelet_no_spatial_graph",
        "wavelet_no_expression_features",
        "gat_hard_expression_distance",
        "wavelet_hard_expression_distance",
    }
    summary_metrics = {
        "auroc_mean",
        "aupr_mean",
        "weak_signal_recall_at_threshold_mean",
        "empirical_fdr_at_threshold_mean",
        "false_positive_high_expression_rate_at_threshold_mean",
    }
    stress_metrics = {
        "auroc",
        "aupr",
        "weak_signal_recall_at_threshold",
        "empirical_fdr_at_threshold",
        "false_positive_high_expression_rate_at_threshold",
    }
    return [
        _row(
            "5",
            goal,
            "full ablation summary covers LR identity, distance, spatial graph, expression, GAT, and wavelet axes",
            required_ablation_axes.issubset(ablations)
            and bool(architecture)
            and summary_metrics.issubset(architecture[0]),
            architecture_path,
            f"missing={sorted(required_ablation_axes - ablations)}",
        ),
        _row(
            "5",
            goal,
            "pseudo-label/edge-prior strategy ablations are recorded separately from architecture",
            {
                "wavelet_mechanism_niche_lr_prior_lrwise_ranking",
                "wavelet_mechanism_niche_edge_prior_ranking",
            }.issubset(edge_ablations),
            edge_path,
            f"edge_ablations={sorted(edge_ablations)}",
        ),
        _row(
            "5",
            goal,
            "weak-signal and high-expression false-positive stress metrics cover current model candidates",
            {"spatialself_wavelet", "spatialself_wavelet_edge_prior", "spatialself_gat"}.issubset(stress_methods)
            and bool(stress)
            and stress_metrics.issubset(stress[0]),
            stress_path,
            f"stress_methods={sorted(stress_methods)}",
        ),
    ]


def _goal6_fdr_framework() -> list[GateRow]:
    goal = "FDR/permutation significance framework"
    controlled_path = PROJECT_ROOT / "outputs/phase1/exp06_permutation_fdr/full_controlled_spatialself_panel/permutation_fdr_panel_summary.csv"
    sparse_path = PROJECT_ROOT / "outputs/phase1/exp06_permutation_fdr/full_slide_sparse_semisynthetic_panel/permutation_fdr_panel_summary.csv"
    controlled = _read_csv(controlled_path)
    sparse = _read_csv(sparse_path)
    nulls = {row.get("null_model", "") for row in controlled}
    sparse_nulls = {row.get("null_model", "") for row in sparse}
    required_nulls = {"spatial_permutation", "receiver_permutation", "lr_identity_permutation"}
    required_columns = {
        "min_p_value",
        "min_bh_fdr",
        "fdr_path",
        "lr_pair_fdr_path",
        "cell_type_pair_fdr_path",
        "n_permutations",
    }
    return [
        _row(
            "6",
            goal,
            "controlled FDR panel covers all required null models across five replicates",
            required_nulls.issubset(nulls)
            and len({row.get("replicate", "") for row in controlled}) >= 5
            and bool(controlled)
            and required_columns.issubset(controlled[0]),
            controlled_path,
            f"nulls={sorted(nulls)}; replicates={sorted({row.get('replicate', '') for row in controlled})}",
        ),
        _row(
            "6",
            goal,
            "semi-synthetic full-slide FDR panel covers the same null models",
            required_nulls.issubset(sparse_nulls)
            and bool(sparse)
            and required_columns.issubset(sparse[0]),
            sparse_path,
            f"nulls={sorted(sparse_nulls)}; methods={sorted({row.get('method_panel', '') for row in sparse})}",
        ),
        _row(
            "6",
            goal,
            "FDR outputs include empirical p-values and BH-adjusted FDR in edge/group tables",
            _linked_fdr_tables_valid(controlled[:3] + sparse[:3]),
            f"{controlled_path.relative_to(PROJECT_ROOT)},{sparse_path.relative_to(PROJECT_ROOT)}",
            "sampled linked edge/LR-pair/cell-type-pair FDR tables",
        ),
    ]


def _goal7_global_validation() -> list[GateRow]:
    goal = "Human lymph node all-LR global validation"
    region_path = PROJECT_ROOT / "outputs/phase1/exp07_lymph_node_global_validation/full_slide_sparse_region_enrichment/spatialself_wavelet/human_lymph_node_visium_replicate_00__spatialself_wavelet_predictions__region_enrichment.csv"
    ranking_path = PROJECT_ROOT / "outputs/phase1/exp07_lymph_node_global_validation/full_slide_sparse_region_enrichment/spatialself_wavelet/human_lymph_node_visium_replicate_00__spatialself_wavelet_predictions__lr_global_ranking.csv"
    downstream_path = PROJECT_ROOT / "outputs/phase1/exp07_lymph_node_global_validation/full_slide_sparse_downstream_target/spatialself_wavelet/human_lymph_node_visium_replicate_00__spatialself_wavelet_predictions__downstream_target_correlation.csv"
    control_path = PROJECT_ROOT / "outputs/phase1/exp07_lymph_node_global_validation/full_slide_sparse_control_summary/real_validation_control_method_summary.csv"
    region = _read_csv(region_path)
    ranking = _read_csv(ranking_path)
    downstream = _read_csv(downstream_path)
    control = _read_csv(control_path)
    tested = [row for row in downstream if row.get("status") == "tested"]
    return [
        _row(
            "7",
            goal,
            "full-slide lymph-node validation ranks all LR pairs globally and by region",
            len(ranking) >= 5
            and len({row.get("lr_id", "") for row in region}) >= 5
            and len({row.get("region", "") for row in region}) >= 3,
            f"{region_path.relative_to(PROJECT_ROOT)},{ranking_path.relative_to(PROJECT_ROOT)}",
            f"n_lr_ranked={len(ranking)}; n_region_rows={len(region)}",
        ),
        _row(
            "7",
            goal,
            "downstream target correlation tests positive LR controls with FDR columns",
            len(tested) >= 3
            and bool(downstream)
            and {"spearman_r", "empirical_p_value", "bh_fdr", "status"}.issubset(downstream[0]),
            downstream_path,
            f"n_tested={len(tested)}",
        ),
        _row(
            "7",
            goal,
            "positive controls rank ahead of high-expression negative controls",
            bool(control)
            and all(_to_float(row.get("mean_positive_rank")) < _to_float(row.get("mean_negative_rank")) for row in control),
            control_path,
            f"methods={[row.get('method') for row in control]}",
        ),
    ]


def _goal8_runtime_memory() -> list[GateRow]:
    goal = "Runtime/memory benchmark"
    medium_path = PROJECT_ROOT / "outputs/phase1/exp08_runtime_memory/controlled_medium_scaling/runtime_memory_scaling.csv"
    sparse_100k_path = PROJECT_ROOT / "outputs/phase1/exp08_runtime_memory/sparse_100k_spatialself_smoke/runtime_memory_scaling.csv"
    retained_path = PROJECT_ROOT / "outputs/phase1/exp08_runtime_memory/full_slide_sparse_semisynthetic_retained_profile/runtime_memory_profile.csv"
    medium = _read_csv(medium_path)
    sparse_100k = _read_csv(sparse_100k_path)
    retained = _read_csv(retained_path)
    medium_methods = {row.get("method", "") for row in medium}
    medium_scales = {_to_float(row.get("scale_n_cells")) for row in medium}
    retained_completed = {row.get("method", "") for row in retained if row.get("status") == "completed"}
    retained_required = {
        "spatialself_wavelet",
        "spatialself_wavelet_edge_prior",
        "spatialself_gat",
        "commot",
        "cellchat",
        "spatalk",
        "cellnest",
        "deeptalk",
    }
    return [
        _row(
            "8",
            goal,
            "controlled runtime scaling records wall-clock and memory for SpatialSelf variants and COMMOT",
            {500.0, 1000.0}.issubset(medium_scales)
            and {"spatialself_wavelet", "spatialself_wavelet_edge_prior", "spatialself_gat", "commot"}.issubset(medium_methods)
            and _runtime_scaling_rows_valid(medium),
            medium_path,
            f"methods={sorted(medium_methods)}; scales={sorted(medium_scales)}",
        ),
        _row(
            "8",
            goal,
            "sparse SpatialSelf scaling reaches 100k cells/spots with candidate/resource accounting",
            any(_to_float(row.get("scale_n_cells")) >= 100000 for row in sparse_100k)
            and _runtime_scaling_rows_valid(sparse_100k),
            sparse_100k_path,
            f"max_scale={_max(_to_float(row.get('scale_n_cells')) for row in sparse_100k):.0f}",
        ),
        _row(
            "8",
            goal,
            "full-slide retained-method runtime profile covers realistic-background benchmark and baselines",
            retained_required.issubset(retained_completed)
            and _runtime_profile_rows_valid([row for row in retained if row.get("status") == "completed"])
            and max(_to_float(row.get("n_candidate_edges")) for row in retained) >= 300000,
            retained_path,
            f"completed={sorted(retained_completed)}",
        ),
    ]


def _cross_goal_repository_hygiene() -> list[GateRow]:
    goal = "Cross-goal repository hygiene"
    evidence_matrix = PROJECT_ROOT / "outputs/phase1/phase1_evidence_matrix/phase1_evidence_matrix.csv"
    audit = PROJECT_ROOT / "outputs/phase1/phase1_audit/phase1_audit_summary.csv"
    evidence_rows = _read_csv(evidence_matrix)
    audit_rows = _read_csv(audit)
    return [
        _row(
            "phase1",
            goal,
            "evidence matrix records all eight goals as complete",
            len(evidence_rows) == 8
            and all(row.get("evidence_status") == "complete" for row in evidence_rows),
            evidence_matrix,
            f"n_rows={len(evidence_rows)}",
        ),
        _row(
            "phase1",
            goal,
            "artifact audit has no partial or missing rows",
            bool(audit_rows)
            and all(row.get("status") in {"complete", "smoke_ready"} for row in audit_rows)
            and not any(row.get("status") in {"partial", "missing"} for row in audit_rows),
            audit,
            f"status_counts={_status_counts(audit_rows)}",
        ),
    ]


def _runtime_scaling_rows_valid(rows: list[dict[str, str]]) -> bool:
    required = {
        "scale_n_cells",
        "n_cells_or_spots",
        "n_candidate_edges",
        "n_lr_pairs",
        "method",
        "status",
        "wall_clock_seconds",
        "peak_cpu_memory_mb",
    }
    if not rows or not required.issubset(rows[0]):
        return False
    completed = [row for row in rows if row.get("status") == "completed"]
    return bool(completed) and all(
        _to_float(row.get("n_candidate_edges")) > 0
        and _to_float(row.get("n_lr_pairs")) > 0
        and _to_float(row.get("wall_clock_seconds")) > 0
        and _to_float(row.get("peak_cpu_memory_mb")) > 0
        for row in completed
    )


def _runtime_profile_rows_valid(rows: list[dict[str, str]]) -> bool:
    required = {
        "n_cells_or_spots",
        "n_candidate_edges",
        "n_lr_pairs",
        "method",
        "status",
        "wall_clock_seconds",
        "peak_cpu_memory_mb",
        "total_runtime_seconds",
        "returncode",
        "stdout_log",
        "stderr_log",
    }
    if not rows or not required.issubset(rows[0]):
        return False
    return all(
        _to_float(row.get("n_candidate_edges")) > 0
        and _to_float(row.get("n_lr_pairs")) > 0
        and _to_float(row.get("wall_clock_seconds")) > 0
        and _to_float(row.get("peak_cpu_memory_mb")) > 0
        and int(_to_float(row.get("returncode"))) == 0
        and _project_path(row.get("stdout_log", "")).exists()
        and _project_path(row.get("stderr_log", "")).exists()
        for row in rows
    )


def _linked_fdr_tables_valid(rows: list[dict[str, str]]) -> bool:
    for row in rows:
        for column in ("fdr_path", "lr_pair_fdr_path", "cell_type_pair_fdr_path"):
            path = _project_path(row.get(column, ""))
            table = _read_csv(path)
            if not table:
                return False
            columns = set(table[0])
            if column == "fdr_path":
                if "empirical_p_value" not in columns or "bh_fdr" not in columns:
                    return False
                p_column = "empirical_p_value"
                fdr_column = "bh_fdr"
            else:
                if "group_p_value" not in columns or "group_bh_fdr" not in columns:
                    return False
                p_column = "group_p_value"
                fdr_column = "group_bh_fdr"
            for record in table[:50]:
                p_value = _to_float(record.get(p_column))
                bh_fdr = _to_float(record.get(fdr_column))
                if not (0.0 <= p_value <= 1.0 and 0.0 <= bh_fdr <= 1.0):
                    return False
    return True


def _removed_baseline_hits() -> tuple[list[str], list[str]]:
    source_hits: list[str] = []
    for root in SOURCE_AUDIT_ROOTS:
        path = PROJECT_ROOT / root
        if path.is_file():
            _scan_file_for_residue(path, source_hits)
        elif path.is_dir():
            for file_path in path.rglob("*"):
                if file_path.is_file():
                    _scan_file_for_residue(file_path, source_hits)
    output_hits: list[str] = []
    output_root = PROJECT_ROOT / "outputs/phase1"
    if output_root.exists():
        for path in output_root.rglob("*"):
            if any(token.lower() in path.name.lower() for token in REMOVED_BASELINE_TOKENS[1:]):
                output_hits.append(str(path.relative_to(PROJECT_ROOT)))
    return source_hits, output_hits


def _scan_file_for_residue(path: Path, hits: list[str]) -> None:
    if path.resolve() == THIS_FILE:
        return
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return
    if RESIDUE_PATTERN.search(text):
        hits.append(str(path.relative_to(PROJECT_ROOT)))


def _row(goal_id: str, goal: str, requirement: str, passed: bool, evidence: Path | str, detail: str = "") -> GateRow:
    evidence_text = str(evidence.relative_to(PROJECT_ROOT)) if isinstance(evidence, Path) else evidence
    return GateRow(goal_id, goal, requirement, "pass" if passed else "fail", evidence_text, detail)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _first(rows: list[dict[str, str]]) -> dict[str, str]:
    return rows[0] if rows else {}


def _first_float(rows: list[dict[str, str]], key: str, value: str, column: str) -> float:
    for row in rows:
        if row.get(key) == value:
            return _to_float(row.get(column))
    return float("nan")


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _to_int(value: object) -> int:
    number = _to_float(value)
    return int(number) if math.isfinite(number) else 0


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _finite(values) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def _mean(values) -> float:
    finite = _finite(values)
    return sum(finite) / len(finite) if finite else float("nan")


def _max(values) -> float:
    finite = _finite(values)
    return max(finite) if finite else float("nan")


def _status_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = row.get("status", "")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _to_markdown(rows: list[GateRow]) -> str:
    lines = [
        "# Phase 1 Completion Gate",
        "",
        "| Goal | Requirement | Status | Evidence | Detail |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(f"{row.goal_id}. {row.goal}"),
                    _md(row.requirement),
                    _md(row.status),
                    _md(row.evidence),
                    _md(row.detail),
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
