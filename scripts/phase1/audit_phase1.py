#!/usr/bin/env python
"""Audit Phase 1 implementation and output artifacts.

The audit is intentionally evidence-based: it checks current files and output
tables, then records which Phase 1 items are full, smoke-ready, partial, or
missing. It does not run experiments.
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
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spitialself.experiments import list_phase1_experiments


REMOVED_BASELINE_TOKENS = (
    "LR-" + "Coexp",
    "lr_" + "coexp",
    "LR_" + "Coexp",
    "lr" + "coexp",
)
SOURCE_RESIDUE_PATTERN = re.compile("|".join(re.escape(token) for token in REMOVED_BASELINE_TOKENS))
SOURCE_AUDIT_ROOTS = ("README.md", "docs", "configs", "scripts", "src")


@dataclass
class AuditRow:
    exp_id: str
    requirement: str
    status: str
    evidence: str
    detail: str = ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase 1 SpatialSelf artifacts.")
    parser.add_argument(
        "--output-dir",
        default="outputs/phase1/phase1_audit",
        help="Directory for phase1_audit_summary.csv/json",
    )
    args = parser.parse_args()

    rows: list[AuditRow] = []
    rows.extend(_audit_registry())
    rows.extend(_audit_removed_baseline())
    rows.extend(_audit_evidence_matrix())
    rows.extend(_audit_model_variants())
    rows.extend(_audit_exp01())
    rows.extend(_audit_exp02())
    rows.extend(_audit_exp03())
    rows.extend(_audit_exp04())
    rows.extend(_audit_exp05())
    rows.extend(_audit_exp06())
    rows.extend(_audit_exp07())
    rows.extend(_audit_exp08())

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "phase1_audit_summary.csv"
    json_path = output_dir / "phase1_audit_summary.json"

    fieldnames = ["exp_id", "requirement", "status", "evidence", "detail"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    with json_path.open("w") as handle:
        json.dump([asdict(row) for row in rows], handle, indent=2)

    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row.status] = status_counts.get(row.status, 0) + 1

    print(f"Audit CSV:  {csv_path}")
    print(f"Audit JSON: {json_path}")
    print("Status counts:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    print("\nNon-complete rows:")
    for row in rows:
        if row.status != "complete":
            print(f"  {row.exp_id} | {row.status} | {row.requirement} | {row.detail}")


def _audit_registry() -> list[AuditRow]:
    rows: list[AuditRow] = []
    experiments = list_phase1_experiments()
    evidence_matrix = PROJECT_ROOT / "outputs/phase1/phase1_evidence_matrix/phase1_evidence_matrix.csv"
    evidence_rows = _read_csv(evidence_matrix) if evidence_matrix.exists() else []
    evidence_complete_count = sum(1 for row in evidence_rows if row.get("evidence_status") == "complete")
    rows.append(
        AuditRow(
            "phase1",
            "registry has eight experiments",
            "complete" if len(experiments) == 8 else "partial",
            "src/spitialself/experiments/phase1.py",
            f"n_experiments={len(experiments)}",
        )
    )
    statuses = {exp.exp_id: exp.status for exp in experiments}
    rows.append(
        AuditRow(
            "phase1",
            "registry experiment statuses match current evidence matrix",
            "complete"
            if len(experiments) == 8
            and evidence_complete_count == 8
            and all(exp.status == "evidence_complete" for exp in experiments)
            else "partial",
            "src/spitialself/experiments/phase1.py,outputs/phase1/phase1_evidence_matrix/phase1_evidence_matrix.csv",
            f"statuses={statuses}; evidence_complete_count={evidence_complete_count}",
        )
    )
    for exp in experiments:
        rows.append(
            AuditRow(
                exp.exp_id,
                "config file exists",
                _exists_status(PROJECT_ROOT / exp.config_path),
                exp.config_path,
            )
        )
    return rows


def _audit_removed_baseline() -> list[AuditRow]:
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
    for path in output_root.rglob("*"):
        if any(token.lower() in path.name.lower() for token in REMOVED_BASELINE_TOKENS[1:]):
            output_hits.append(str(path.relative_to(PROJECT_ROOT)))
    return [
        AuditRow(
            "phase1",
            "removed expression-distance baseline absent from active source/config/docs",
            "complete" if not source_hits else "partial",
            ",".join(source_hits[:20]) if source_hits else "README.md,docs,configs,scripts,src",
            f"n_hits={len(source_hits)}",
        ),
        AuditRow(
            "phase1",
            "no removed-baseline output artifacts",
            "complete" if not output_hits else "partial",
            ",".join(output_hits[:20]) if output_hits else "outputs/phase1",
            f"n_hits={len(output_hits)}",
        ),
    ]


def _audit_evidence_matrix() -> list[AuditRow]:
    root = PROJECT_ROOT / "outputs/phase1/phase1_evidence_matrix"
    csv_path = root / "phase1_evidence_matrix.csv"
    json_path = root / "phase1_evidence_matrix.json"
    md_path = root / "phase1_evidence_matrix.md"
    rows = [
        AuditRow("phase1", "phase1 evidence matrix CSV exists", _exists_status(csv_path), str(csv_path.relative_to(PROJECT_ROOT))),
        AuditRow("phase1", "phase1 evidence matrix JSON exists", _exists_status(json_path), str(json_path.relative_to(PROJECT_ROOT))),
        AuditRow("phase1", "phase1 evidence matrix Markdown exists", _exists_status(md_path), str(md_path.relative_to(PROJECT_ROOT))),
    ]
    if csv_path.exists():
        records = _read_csv(csv_path)
        required_columns = {
            "goal_id",
            "goal",
            "evidence_status",
            "primary_artifacts",
            "key_findings",
            "caveats",
        }
        goal_ids = {row.get("goal_id", "") for row in records}
        missing_columns = sorted(required_columns - set(records[0] if records else {}))
        rows.append(
            AuditRow(
                "phase1",
                "phase1 evidence matrix maps all eight goals to artifacts and caveats",
                "complete"
                if (
                    len(records) == 8
                    and goal_ids == {str(i) for i in range(1, 9)}
                    and not missing_columns
                    and all(row.get("evidence_status") == "complete" for row in records)
                    and all(row.get("primary_artifacts") and row.get("key_findings") and row.get("caveats") for row in records)
                )
                else "partial",
                str(csv_path.relative_to(PROJECT_ROOT)),
                f"n_rows={len(records)}; goal_ids={sorted(goal_ids)}; missing_columns={missing_columns}",
            )
        )
    return rows


def _audit_model_variants() -> list[AuditRow]:
    rows: list[AuditRow] = []
    factory = PROJECT_ROOT / "src/spitialself/model_factory.py"
    models = PROJECT_ROOT / "src/spitialself/models.py"
    spatialself_runner = PROJECT_ROOT / "scripts/benchmark/run_spatialself.py"
    baseline_runner = PROJECT_ROOT / "scripts/benchmark/run_baseline_comparison.py"
    runtime_runner = PROJECT_ROOT / "scripts/benchmark/profile_runtime_memory.py"
    full_edge_summary = PROJECT_ROOT / "outputs/phase1/exp03_baseline_comparison/full_controlled_edge_prior_core/baseline_metrics_summary.csv"
    smoke_ablation_summary = PROJECT_ROOT / "outputs/phase1/exp04_model_ablation/smoke_controlled/phase_runner/spatialself_ablation_summary.csv"
    full_architecture_ablation_summary = PROJECT_ROOT / "outputs/phase1/exp04_model_ablation/full_architecture_identity_component_ablation_epoch10/spatialself_ablation_summary.csv"
    smoke_runtime_scaling = PROJECT_ROOT / "outputs/phase1/exp08_runtime_memory/smoke_scaling_controlled/runtime_memory_scaling.csv"
    medium_runtime_scaling = PROJECT_ROOT / "outputs/phase1/exp08_runtime_memory/controlled_medium_scaling/runtime_memory_scaling.csv"

    factory_text = factory.read_text(encoding="utf-8") if factory.exists() else ""
    models_text = models.read_text(encoding="utf-8") if models.exists() else ""
    runner_text = spatialself_runner.read_text(encoding="utf-8") if spatialself_runner.exists() else ""
    baseline_text = baseline_runner.read_text(encoding="utf-8") if baseline_runner.exists() else ""
    runtime_text = runtime_runner.read_text(encoding="utf-8") if runtime_runner.exists() else ""
    required_model_names = {
        "spatialself_wavelet",
        "spatialself_gat",
        "spatialself_gcn",
        "spatialself_mlp",
        "spatialself_sage",
    }
    rows.append(
        AuditRow(
            "phase1",
            "model factory registers replaceable SpatialSelf architectures",
            "complete" if all(name in factory_text for name in required_model_names) else "partial",
            str(factory.relative_to(PROJECT_ROOT)),
            f"missing={[name for name in sorted(required_model_names) if name not in factory_text]}",
        )
    )
    required_classes = {
        "GATv2NodeEncoder",
        "GraphWaveletNodeEncoder",
        "SpatialSelfWaveletModel",
        "SpatialSelfGCNModel",
        "SpatialSelfMLPModel",
        "LRSpecificSAGEModel",
    }
    rows.append(
        AuditRow(
            "phase1",
            "model module contains GAT, graph-wavelet, GCN, MLP, and SAGE implementations",
            "complete" if all(name in models_text for name in required_classes) else "partial",
            str(models.relative_to(PROJECT_ROOT)),
            f"missing={[name for name in sorted(required_classes) if name not in models_text]}",
        )
    )
    edge_prior_ok = (
        "edge_feature_dim" in models_text
        and "use_edge_prior_features" in runner_text
        and "--use-edge-prior-features" in runner_text
        and "spatialself_wavelet_edge_prior" in baseline_text
        and "spatialself_wavelet_edge_prior" in runtime_text
    )
    rows.append(
        AuditRow(
            "phase1",
            "edge-prior candidate is wired through model scorer, runner, baseline, and runtime profiler",
            "complete" if edge_prior_ok else "partial",
            "src/spitialself/models.py,scripts/benchmark/run_spatialself.py,scripts/benchmark/run_baseline_comparison.py,scripts/benchmark/profile_runtime_memory.py",
        )
    )
    if full_edge_summary.exists():
        records = _read_csv(full_edge_summary)
        methods = {row.get("method", "") for row in records}
        required_methods = {"spatialself_wavelet", "spatialself_wavelet_edge_prior", "spatialself_gat"}
        rows.append(
            AuditRow(
                "phase1",
                "full baseline outputs compare wavelet, edge-prior, and GAT variants",
                "complete" if required_methods.issubset(methods) else "partial",
                str(full_edge_summary.relative_to(PROJECT_ROOT)),
                f"missing={sorted(required_methods - methods)}",
            )
        )
    architecture_ablation_summary = (
        full_architecture_ablation_summary if full_architecture_ablation_summary.exists() else smoke_ablation_summary
    )
    if architecture_ablation_summary.exists():
        records = _read_csv(architecture_ablation_summary)
        ablations = {row.get("ablation", "") for row in records}
        required_ablations = {
            "gat_hard_expression_distance",
            "wavelet_hard_expression_distance",
            "gcn_hard_expression_distance",
            "mlp_hard_expression_distance",
            "wavelet_no_spatial_graph",
        }
        rows.append(
            AuditRow(
                "phase1",
                "ablation outputs compare graph architecture and graph-free variants",
                "complete" if full_architecture_ablation_summary.exists() and required_ablations.issubset(ablations) else "smoke_ready" if required_ablations.issubset(ablations) else "partial",
                str(architecture_ablation_summary.relative_to(PROJECT_ROOT)),
                f"missing={sorted(required_ablations - ablations)}",
            )
        )
    runtime_scaling = medium_runtime_scaling if medium_runtime_scaling.exists() else smoke_runtime_scaling
    if runtime_scaling.exists():
        records = _read_csv(runtime_scaling)
        methods = {row.get("method", "") for row in records}
        scales = {_to_float(row.get("scale_n_cells", "nan")) for row in records}
        required_methods = {"spatialself_wavelet", "spatialself_wavelet_edge_prior", "spatialself_gat"}
        medium_ok = (
            runtime_scaling == medium_runtime_scaling
            and required_methods.issubset(methods)
            and "commot" in methods
            and {500.0, 1000.0}.issubset(scales)
        )
        rows.append(
            AuditRow(
                "phase1",
                "runtime outputs profile wavelet, edge-prior, GAT, and COMMOT variants",
                "complete"
                if medium_ok
                else "smoke_ready"
                if required_methods.issubset(methods)
                else "partial",
                str(runtime_scaling.relative_to(PROJECT_ROOT)),
                f"methods={sorted(methods)}; scales={sorted(value for value in scales if math.isfinite(value))}",
            )
        )
    return rows


def _audit_exp01() -> list[AuditRow]:
    root = PROJECT_ROOT / "outputs/phase1/exp01_controlled_benchmark/datasets"
    replicate_dirs = _replicate_dirs(root)
    mechanism_summary = root / "controlled_mechanism_summary.csv"
    lr_summary = root / "controlled_lr_summary.csv"
    validation = root / "validation_summary.csv"
    rows = [
        AuditRow("exp01_controlled_benchmark", "full controlled replicates exist", "complete" if len(replicate_dirs) >= 5 else "partial", str(root.relative_to(PROJECT_ROOT)), f"n_replicates={len(replicate_dirs)}"),
        AuditRow("exp01_controlled_benchmark", "validation summary exists", _exists_status(validation), str(validation.relative_to(PROJECT_ROOT))),
        AuditRow("exp01_controlled_benchmark", "mechanism summary exists", _exists_status(mechanism_summary), str(mechanism_summary.relative_to(PROJECT_ROOT))),
        AuditRow("exp01_controlled_benchmark", "LR summary exists", _exists_status(lr_summary), str(lr_summary.relative_to(PROJECT_ROOT))),
    ]
    if mechanism_summary.exists():
        mechanism_rows = _read_csv(mechanism_summary)
        mechanisms = {row.get("mechanism", "") for row in mechanism_rows}
        required = {
            "juxtacrine_contact",
            "short_medium_paracrine",
            "long_range_chemokine_niche",
            "false_positive_high_expression",
        }
        fp_positive = sum(
            int(float(row.get("positives", 0) or 0))
            for row in mechanism_rows
            if row.get("mechanism") == "false_positive_high_expression"
        )
        rows.append(
            AuditRow(
                "exp01_controlled_benchmark",
                "mechanism truth categories present",
                "complete" if required.issubset(mechanisms) else "partial",
                str(mechanism_summary.relative_to(PROJECT_ROOT)),
                f"mechanisms={sorted(mechanisms)}",
            )
        )
        rows.append(
            AuditRow(
                "exp01_controlled_benchmark",
                "high-expression controls have zero positives",
                "complete" if fp_positive == 0 else "partial",
                str(mechanism_summary.relative_to(PROJECT_ROOT)),
                f"false_positive_high_expression_positives={fp_positive}",
            )
        )
        diagnostic_columns = {
            "positive_median_distance",
            "positive_max_distance",
            "type_rule_positive_fraction",
            "range_rule_positive_fraction",
            "niche_rule_positive_fraction",
            "heuristic_coexpr_distance_auroc",
            "heuristic_coexpr_distance_aupr",
        }
        rows.append(
            AuditRow(
                "exp01_controlled_benchmark",
                "mechanism summary records rule and coexpression-distance diagnostics",
                "complete" if mechanism_rows and diagnostic_columns.issubset(mechanism_rows[0]) else "partial",
                str(mechanism_summary.relative_to(PROJECT_ROOT)),
                f"missing={sorted(diagnostic_columns - set(mechanism_rows[0] if mechanism_rows else {}))}",
            )
        )
        grouped_distances: dict[str, list[float]] = {}
        contact_max_values: list[float] = []
        rule_values: list[float] = []
        for row in mechanism_rows:
            mechanism = row.get("mechanism", "")
            try:
                median_distance = float(row.get("positive_median_distance", "nan"))
                positive_max = float(row.get("positive_max_distance", "nan"))
            except ValueError:
                median_distance = positive_max = float("nan")
            if math.isfinite(median_distance):
                grouped_distances.setdefault(mechanism, []).append(median_distance)
            if mechanism == "juxtacrine_contact" and math.isfinite(positive_max):
                contact_max_values.append(positive_max)
            for column in ("type_rule_positive_fraction", "range_rule_positive_fraction", "niche_rule_positive_fraction"):
                try:
                    value = float(row.get(column, "nan"))
                except ValueError:
                    value = float("nan")
                if mechanism != "false_positive_high_expression" and math.isfinite(value):
                    rule_values.append(value)
        contact_median = _mean(grouped_distances.get("juxtacrine_contact", []))
        chemokine_median = _mean(grouped_distances.get("long_range_chemokine_niche", []))
        rows.append(
            AuditRow(
                "exp01_controlled_benchmark",
                "mechanism distance regimes are separated",
                "complete"
                if (
                    math.isfinite(contact_median)
                    and math.isfinite(chemokine_median)
                    and chemokine_median > contact_median
                    and contact_max_values
                    and max(contact_max_values) <= 0.055 + 1e-9
                )
                else "partial",
                str(mechanism_summary.relative_to(PROJECT_ROOT)),
                (
                    f"contact_median={contact_median:.4f}; chemokine_median={chemokine_median:.4f}; "
                    f"max_contact_positive_distance={max(contact_max_values) if contact_max_values else float('nan'):.4f}"
                ),
            )
        )
        rows.append(
            AuditRow(
                "exp01_controlled_benchmark",
                "positive labels satisfy mechanism rules",
                "complete" if rule_values and min(rule_values) >= 0.999 else "partial",
                str(mechanism_summary.relative_to(PROJECT_ROOT)),
                f"min_positive_rule_fraction={min(rule_values) if rule_values else float('nan'):.3f}",
            )
        )
    return rows


def _audit_exp02() -> list[AuditRow]:
    full_root = PROJECT_ROOT / "outputs/phase1/exp02_semisynthetic_real_background/datasets"
    full_slide_sparse_root = PROJECT_ROOT / "outputs/phase1/exp02_semisynthetic_real_background/full_slide_sparse_datasets"
    smoke_root = PROJECT_ROOT / "outputs/phase1/exp02_semisynthetic_real_background/smoke_datasets"
    validation = smoke_root / "validation_summary.csv"
    injection_summary = smoke_root / "semisynthetic_injection_summary.csv"
    mechanism_summary = smoke_root / "semisynthetic_mechanism_summary.csv"
    full_validation = full_root / "validation_summary.csv"
    full_injection_summary = full_root / "semisynthetic_injection_summary.csv"
    full_mechanism_summary = full_root / "semisynthetic_mechanism_summary.csv"
    full_slide_validation = full_slide_sparse_root / "validation_summary.csv"
    full_slide_injection_summary = full_slide_sparse_root / "semisynthetic_injection_summary.csv"
    full_slide_mechanism_summary = full_slide_sparse_root / "semisynthetic_mechanism_summary.csv"
    full_slide_realism_root = PROJECT_ROOT / "outputs/phase1/exp02_semisynthetic_real_background/full_slide_sparse_realism"
    full_slide_realism_summary = full_slide_realism_root / "semisynthetic_realism_summary.csv"
    full_slide_realism_gene_stats = full_slide_realism_root / "semisynthetic_realism_gene_stats.csv"
    full_slide_realism_moran = full_slide_realism_root / "semisynthetic_realism_moran.csv"
    full_slide_realism_composition = full_slide_realism_root / "semisynthetic_realism_composition_geometry.csv"
    full_slide_realism_manifest = full_slide_realism_root / "semisynthetic_realism_manifest.json"
    rows = [
        AuditRow("exp02_semisynthetic_real_background", "smoke semi-synthetic validation exists", _exists_status(validation, existing_status="smoke_ready"), str(validation.relative_to(PROJECT_ROOT))),
        AuditRow("exp02_semisynthetic_real_background", "full semi-synthetic dataset exists", "complete" if _replicate_dirs(full_root) else "partial", str(full_root.relative_to(PROJECT_ROOT)), f"n_full_replicates={len(_replicate_dirs(full_root))}"),
        AuditRow("exp02_semisynthetic_real_background", "full-slide sparse semi-synthetic dataset exists", "complete" if _replicate_dirs(full_slide_sparse_root) else "partial", str(full_slide_sparse_root.relative_to(PROJECT_ROOT)), f"n_full_slide_sparse_replicates={len(_replicate_dirs(full_slide_sparse_root))}"),
        AuditRow("exp02_semisynthetic_real_background", "smoke semi-synthetic injection summary exists", _exists_status(injection_summary, existing_status="smoke_ready"), str(injection_summary.relative_to(PROJECT_ROOT))),
        AuditRow("exp02_semisynthetic_real_background", "smoke semi-synthetic mechanism summary exists", _exists_status(mechanism_summary, existing_status="smoke_ready"), str(mechanism_summary.relative_to(PROJECT_ROOT))),
        AuditRow("exp02_semisynthetic_real_background", "full semi-synthetic validation exists", _exists_status(full_validation), str(full_validation.relative_to(PROJECT_ROOT))),
        AuditRow("exp02_semisynthetic_real_background", "full semi-synthetic injection summary exists", _exists_status(full_injection_summary), str(full_injection_summary.relative_to(PROJECT_ROOT))),
        AuditRow("exp02_semisynthetic_real_background", "full semi-synthetic mechanism summary exists", _exists_status(full_mechanism_summary), str(full_mechanism_summary.relative_to(PROJECT_ROOT))),
        AuditRow("exp02_semisynthetic_real_background", "full-slide sparse semi-synthetic validation exists", _exists_status(full_slide_validation), str(full_slide_validation.relative_to(PROJECT_ROOT))),
        AuditRow("exp02_semisynthetic_real_background", "full-slide sparse semi-synthetic injection summary exists", _exists_status(full_slide_injection_summary), str(full_slide_injection_summary.relative_to(PROJECT_ROOT))),
        AuditRow("exp02_semisynthetic_real_background", "full-slide sparse semi-synthetic mechanism summary exists", _exists_status(full_slide_mechanism_summary), str(full_slide_mechanism_summary.relative_to(PROJECT_ROOT))),
        AuditRow("exp02_semisynthetic_real_background", "full-slide sparse realism summary exists", _exists_status(full_slide_realism_summary), str(full_slide_realism_summary.relative_to(PROJECT_ROOT))),
        AuditRow("exp02_semisynthetic_real_background", "full-slide sparse realism gene statistics exist", _exists_status(full_slide_realism_gene_stats), str(full_slide_realism_gene_stats.relative_to(PROJECT_ROOT))),
        AuditRow("exp02_semisynthetic_real_background", "full-slide sparse realism Moran statistics exist", _exists_status(full_slide_realism_moran), str(full_slide_realism_moran.relative_to(PROJECT_ROOT))),
        AuditRow("exp02_semisynthetic_real_background", "full-slide sparse realism composition/geometry statistics exist", _exists_status(full_slide_realism_composition), str(full_slide_realism_composition.relative_to(PROJECT_ROOT))),
        AuditRow("exp02_semisynthetic_real_background", "full-slide sparse realism manifest exists", _exists_status(full_slide_realism_manifest), str(full_slide_realism_manifest.relative_to(PROJECT_ROOT))),
    ]
    if validation.exists():
        summary = _read_csv(validation)
        first = summary[0] if summary else {}
        preserved_columns = {
            "median_library_size",
            "library_size_iqr",
            "coordinate_span_x",
            "coordinate_span_y",
            "median_nn_distance",
            "median_local_density",
            "dominant_region_fraction",
            "dominant_cell_type_fraction",
        }
        rows.append(
            AuditRow(
                "exp02_semisynthetic_real_background",
                "real-background smoke preserves broad expression background",
                "smoke_ready" if int(float(first.get("expression_genes", 0) or 0)) >= 1000 else "partial",
                str(validation.relative_to(PROJECT_ROOT)),
                f"expression_genes={first.get('expression_genes')}, mean_observed_sparsity={first.get('mean_observed_sparsity')}",
            )
        )
        rows.append(
            AuditRow(
                "exp02_semisynthetic_real_background",
                "real-background smoke records coordinates, library size, local density, and composition",
                "smoke_ready" if preserved_columns.issubset(first) else "partial",
                str(validation.relative_to(PROJECT_ROOT)),
                f"missing={sorted(preserved_columns - set(first))}",
            )
        )
        try:
            coord_span_x = float(first.get("coordinate_span_x", "nan"))
            coord_span_y = float(first.get("coordinate_span_y", "nan"))
            library_iqr = float(first.get("library_size_iqr", "nan"))
            median_nn = float(first.get("median_nn_distance", "nan"))
            median_density = float(first.get("median_local_density", "nan"))
            dominant_region = float(first.get("dominant_region_fraction", "nan"))
            dominant_cell_type = float(first.get("dominant_cell_type_fraction", "nan"))
        except ValueError:
            coord_span_x = coord_span_y = library_iqr = median_nn = median_density = float("nan")
            dominant_region = dominant_cell_type = float("nan")
        rows.append(
            AuditRow(
                "exp02_semisynthetic_real_background",
                "real-background smoke statistics are non-degenerate",
                "smoke_ready"
                if (
                    math.isfinite(coord_span_x)
                    and math.isfinite(coord_span_y)
                    and math.isfinite(library_iqr)
                    and math.isfinite(median_nn)
                    and math.isfinite(median_density)
                    and coord_span_x > 0
                    and coord_span_y > 0
                    and library_iqr > 0
                    and median_nn > 0
                    and median_density > 0
                    and dominant_region < 1.0
                    and dominant_cell_type < 1.0
                )
                else "partial",
                str(validation.relative_to(PROJECT_ROOT)),
                (
                    f"span=({coord_span_x:.3f},{coord_span_y:.3f}); "
                    f"library_iqr={library_iqr:.3f}; median_nn={median_nn:.3f}; "
                    f"median_density={median_density:.3f}; dominant_region={dominant_region:.3f}; "
                    f"dominant_cell_type={dominant_cell_type:.3f}"
                ),
            )
        )
        heuristic_auroc = _to_float(first.get("heuristic_coexpr_distance_auroc", "nan"))
        heuristic_aupr = _to_float(first.get("heuristic_coexpr_distance_aupr", "nan"))
        rows.append(
            AuditRow(
                "exp02_semisynthetic_real_background",
                "real-background smoke records coexpression-distance heuristic diagnostics",
                "smoke_ready"
                if (
                    math.isfinite(heuristic_auroc)
                    and 0.0 <= heuristic_auroc <= 1.0
                    and math.isfinite(heuristic_aupr)
                    and 0.0 <= heuristic_aupr <= 1.0
                )
                else "partial",
                str(validation.relative_to(PROJECT_ROOT)),
                f"heuristic_auroc={heuristic_auroc:.3f}; heuristic_aupr={heuristic_aupr:.3f}",
            )
        )
    if injection_summary.exists():
        injection_rows = _read_csv(injection_summary)
        rows.extend(_audit_semisynthetic_injection_summary(injection_summary, injection_rows, "smoke_ready"))
    if mechanism_summary.exists():
        mechanism_rows = _read_csv(mechanism_summary)
        rows.extend(_audit_semisynthetic_mechanism_summary(mechanism_summary, mechanism_rows, "smoke_ready"))
    if full_injection_summary.exists():
        full_rows = _read_csv(full_injection_summary)
        rows.extend(_audit_semisynthetic_injection_summary(full_injection_summary, full_rows, "complete"))
    if full_validation.exists():
        full_validation_rows = _read_csv(full_validation)
        rows.extend(
            _audit_semisynthetic_validation_summary(
                full_validation,
                full_validation_rows,
                "complete",
                "full real-background",
            )
        )
    if full_mechanism_summary.exists():
        full_mechanism_rows = _read_csv(full_mechanism_summary)
        rows.extend(_audit_semisynthetic_mechanism_summary(full_mechanism_summary, full_mechanism_rows, "complete"))
    if full_slide_validation.exists():
        full_slide_validation_rows = _read_csv(full_slide_validation)
        rows.extend(
            _audit_semisynthetic_validation_summary(
                full_slide_validation,
                full_slide_validation_rows,
                "complete",
                "full-slide sparse real-background",
            )
        )
        first = full_slide_validation_rows[0] if full_slide_validation_rows else {}
        rows.append(
            AuditRow(
                "exp02_semisynthetic_real_background",
                "full-slide sparse semi-synthetic uses complete lymph-node Visium slide with broad candidate set",
                "complete"
                if _to_float(first.get("cells", "nan")) >= 4000
                and _to_float(first.get("candidate_interactions", "nan")) >= 300000
                and _to_float(first.get("expression_genes", "nan")) >= 2000
                else "partial",
                str(full_slide_validation.relative_to(PROJECT_ROOT)),
                (
                    f"cells={first.get('cells')}; genes={first.get('expression_genes')}; "
                    f"candidates={first.get('candidate_interactions')}; positives={first.get('positive_interactions')}; "
                    f"weak={first.get('weak_positive_interactions')}"
                ),
            )
        )
    if full_slide_injection_summary.exists():
        full_slide_rows = _read_csv(full_slide_injection_summary)
        rows.extend(_audit_semisynthetic_injection_summary(full_slide_injection_summary, full_slide_rows, "complete"))
    if full_slide_mechanism_summary.exists():
        full_slide_mechanism_rows = _read_csv(full_slide_mechanism_summary)
        rows.extend(_audit_semisynthetic_mechanism_summary(full_slide_mechanism_summary, full_slide_mechanism_rows, "complete"))
    if full_slide_realism_summary.exists():
        rows.extend(_audit_semisynthetic_realism(full_slide_realism_summary))
    return rows


def _audit_exp03() -> list[AuditRow]:
    root = PROJECT_ROOT / "outputs/phase1/exp03_baseline_comparison/smoke_controlled/combined_full_adapter_check"
    status_path = root / "baseline_status.csv"
    metrics_path = root / "baseline_metrics.csv"
    phase_runner_root = PROJECT_ROOT / "outputs/phase1/exp03_baseline_comparison/smoke_controlled/phase_runner"
    phase_runner_status = phase_runner_root / "baseline_status.csv"
    phase_runner_metrics = phase_runner_root / "baseline_metrics.csv"
    full_core_root = PROJECT_ROOT / "outputs/phase1/exp03_baseline_comparison/full_controlled_retained_core_fair"
    full_core_summary = full_core_root / "baseline_metrics_summary.csv"
    full_core_status = full_core_root / "baseline_status.csv"
    edge_prior_core_root = PROJECT_ROOT / "outputs/phase1/exp03_baseline_comparison/full_controlled_edge_prior_core"
    edge_prior_core_summary = edge_prior_core_root / "baseline_metrics_summary.csv"
    edge_prior_core_status = edge_prior_core_root / "baseline_status.csv"
    full_adapter_root = PROJECT_ROOT / "outputs/phase1/exp03_baseline_comparison/full_controlled_retained_adapter_light"
    full_adapter_summary = full_adapter_root / "baseline_metrics_summary.csv"
    full_adapter_status = full_adapter_root / "baseline_status.csv"
    unified_root = PROJECT_ROOT / "outputs/phase1/exp03_baseline_comparison/full_controlled_retained_unified"
    unified_summary = unified_root / "baseline_metrics_summary.csv"
    unified_status = unified_root / "baseline_status.csv"
    unified_manifest = unified_root / "manifest.json"
    full_slide_sparse_root = PROJECT_ROOT / "outputs/phase1/exp03_baseline_comparison/full_slide_sparse_semisynthetic_retained"
    full_slide_sparse_summary = full_slide_sparse_root / "baseline_metrics_summary.csv"
    full_slide_sparse_status = full_slide_sparse_root / "baseline_status.csv"
    rows = [
        AuditRow("exp03_baseline_comparison", "combined adapter smoke status exists", _exists_status(status_path, existing_status="smoke_ready"), str(status_path.relative_to(PROJECT_ROOT))),
        AuditRow("exp03_baseline_comparison", "combined adapter smoke metrics exists", _exists_status(metrics_path, existing_status="smoke_ready"), str(metrics_path.relative_to(PROJECT_ROOT))),
        AuditRow("exp03_baseline_comparison", "phase-runner baseline smoke status exists", _exists_status(phase_runner_status, existing_status="smoke_ready"), str(phase_runner_status.relative_to(PROJECT_ROOT))),
        AuditRow("exp03_baseline_comparison", "phase-runner baseline smoke metrics exists", _exists_status(phase_runner_metrics, existing_status="smoke_ready"), str(phase_runner_metrics.relative_to(PROJECT_ROOT))),
        AuditRow("exp03_baseline_comparison", "full controlled core baseline summary exists", _exists_status(full_core_summary), str(full_core_summary.relative_to(PROJECT_ROOT))),
        AuditRow("exp03_baseline_comparison", "full controlled core baseline status exists", _exists_status(full_core_status), str(full_core_status.relative_to(PROJECT_ROOT))),
        AuditRow("exp03_baseline_comparison", "full controlled edge-prior core baseline summary exists", _exists_status(edge_prior_core_summary), str(edge_prior_core_summary.relative_to(PROJECT_ROOT))),
        AuditRow("exp03_baseline_comparison", "full controlled edge-prior core baseline status exists", _exists_status(edge_prior_core_status), str(edge_prior_core_status.relative_to(PROJECT_ROOT))),
        AuditRow("exp03_baseline_comparison", "full controlled adapter-light baseline summary exists", _exists_status(full_adapter_summary), str(full_adapter_summary.relative_to(PROJECT_ROOT))),
        AuditRow("exp03_baseline_comparison", "full controlled adapter-light baseline status exists", _exists_status(full_adapter_status), str(full_adapter_status.relative_to(PROJECT_ROOT))),
        AuditRow("exp03_baseline_comparison", "unified full controlled retained baseline summary exists", _exists_status(unified_summary), str(unified_summary.relative_to(PROJECT_ROOT))),
        AuditRow("exp03_baseline_comparison", "unified full controlled retained baseline status exists", _exists_status(unified_status), str(unified_status.relative_to(PROJECT_ROOT))),
        AuditRow("exp03_baseline_comparison", "unified full controlled retained baseline manifest exists", _exists_status(unified_manifest), str(unified_manifest.relative_to(PROJECT_ROOT))),
        AuditRow("exp03_baseline_comparison", "full-slide sparse semi-synthetic retained baseline summary exists", _exists_status(full_slide_sparse_summary), str(full_slide_sparse_summary.relative_to(PROJECT_ROOT))),
        AuditRow("exp03_baseline_comparison", "full-slide sparse semi-synthetic retained baseline status exists", _exists_status(full_slide_sparse_status), str(full_slide_sparse_status.relative_to(PROJECT_ROOT))),
    ]
    if status_path.exists():
        status_rows = _read_csv(status_path)
        methods = {row.get("method"): row.get("status") for row in status_rows}
        required = {
            "spatialself_wavelet",
            "commot",
            "cellchat",
            "spatalk",
            "cellnest",
            "deeptalk",
            "nichenet",
            "niches",
        }
        missing = sorted(required - set(methods))
        rows.append(
            AuditRow(
                "exp03_baseline_comparison",
                "required retained baseline adapters represented",
                "smoke_ready" if not missing else "partial",
                str(status_path.relative_to(PROJECT_ROOT)),
                f"missing={missing}; statuses={methods}",
            )
        )
        rows.append(
            AuditRow(
                "exp03_baseline_comparison",
                "NICHES optional blocked reason recorded",
                "smoke_ready" if methods.get("niches") == "optional_blocked" else "partial",
                str(status_path.relative_to(PROJECT_ROOT)),
                f"niches_status={methods.get('niches')}",
            )
        )
        rows.append(
            AuditRow(
                "exp03_baseline_comparison",
                "retained baseline status rows include reproducibility details",
                "smoke_ready" if _baseline_status_details_valid(status_rows) else "partial",
                str(status_path.relative_to(PROJECT_ROOT)),
                f"n_status_rows={len(status_rows)}",
            )
        )
    if phase_runner_status.exists():
        phase_rows = _read_csv(phase_runner_status)
        methods = {row.get("method"): row.get("status") for row in phase_rows}
        rows.append(
            AuditRow(
                "exp03_baseline_comparison",
                "edge-prior SpatialSelf candidate represented in phase-runner baseline smoke",
                "smoke_ready" if methods.get("spatialself_wavelet_edge_prior") == "completed" else "partial",
                str(phase_runner_status.relative_to(PROJECT_ROOT)),
                f"edge_prior_status={methods.get('spatialself_wavelet_edge_prior')}; methods={sorted(methods)}",
            )
        )
        rows.append(
            AuditRow(
                "exp03_baseline_comparison",
                "phase-runner baseline status rows include model-required/downstream/blocked details",
                "smoke_ready" if _baseline_status_details_valid(phase_rows) else "partial",
                str(phase_runner_status.relative_to(PROJECT_ROOT)),
                f"n_status_rows={len(phase_rows)}",
            )
        )
    if full_core_summary.exists():
        summary_rows = _read_csv(full_core_summary)
        full_methods = {row.get("method", "") for row in summary_rows}
        required_full = {"spatialself_wavelet", "spatialself_gat", "commot", "deeptalk"}
        rows.append(
            AuditRow(
                "exp03_baseline_comparison",
                "full controlled core methods represented",
                "complete" if required_full.issubset(full_methods) else "partial",
                str(full_core_summary.relative_to(PROJECT_ROOT)),
                f"missing={sorted(required_full - full_methods)}",
            )
        )
        rows.append(
            AuditRow(
                "exp03_baseline_comparison",
                "full controlled core baseline metrics have valid benchmark columns",
                "complete" if _baseline_summary_metrics_valid(summary_rows, min_count=5) else "partial",
                str(full_core_summary.relative_to(PROJECT_ROOT)),
                f"methods={sorted(full_methods)}",
            )
        )
    if edge_prior_core_summary.exists():
        summary_rows = _read_csv(edge_prior_core_summary)
        methods = {row.get("method", ""): row for row in summary_rows}
        required_full = {"spatialself_wavelet", "spatialself_wavelet_edge_prior", "spatialself_gat", "commot", "deeptalk"}
        rows.append(
            AuditRow(
                "exp03_baseline_comparison",
                "full controlled edge-prior core methods represented",
                "complete" if required_full.issubset(methods) else "partial",
                str(edge_prior_core_summary.relative_to(PROJECT_ROOT)),
                f"missing={sorted(required_full - set(methods))}",
            )
        )
        rows.append(
            AuditRow(
                "exp03_baseline_comparison",
                "full controlled edge-prior core metrics have valid benchmark columns",
                "complete" if _baseline_summary_metrics_valid(summary_rows, min_count=5) else "partial",
                str(edge_prior_core_summary.relative_to(PROJECT_ROOT)),
                f"methods={sorted(methods)}",
            )
        )
        default = methods.get("spatialself_wavelet", {})
        candidate = methods.get("spatialself_wavelet_edge_prior", {})
        default_aupr = float(default.get("aupr_mean", 0) or 0)
        candidate_aupr = float(candidate.get("aupr_mean", 0) or 0)
        default_weak = float(default.get("weak_signal_recall_at_threshold_mean", 0) or 0)
        candidate_weak = float(candidate.get("weak_signal_recall_at_threshold_mean", 0) or 0)
        rows.append(
            AuditRow(
                "exp03_baseline_comparison",
                "edge-prior candidate improves controlled core AUPR and weak recall over current default",
                "complete" if candidate_aupr >= default_aupr and candidate_weak >= default_weak else "partial",
                str(edge_prior_core_summary.relative_to(PROJECT_ROOT)),
                (
                    f"default_aupr={default_aupr:.3f}; candidate_aupr={candidate_aupr:.3f}; "
                    f"default_weak={default_weak:.3f}; candidate_weak={candidate_weak:.3f}"
                ),
            )
        )
    if full_adapter_summary.exists():
        summary_rows = _read_csv(full_adapter_summary)
        adapter_methods = {row.get("method", "") for row in summary_rows}
        required_adapter = {"cellchat", "spatalk", "cellnest"}
        rows.append(
            AuditRow(
                "exp03_baseline_comparison",
                "full controlled adapter-light methods represented",
                "complete" if required_adapter.issubset(adapter_methods) else "partial",
                str(full_adapter_summary.relative_to(PROJECT_ROOT)),
                f"missing={sorted(required_adapter - adapter_methods)}",
            )
        )
        rows.append(
            AuditRow(
                "exp03_baseline_comparison",
                "full controlled adapter-light metrics have valid benchmark columns",
                "complete" if _baseline_summary_metrics_valid(summary_rows, min_count=5) else "partial",
                str(full_adapter_summary.relative_to(PROJECT_ROOT)),
                f"methods={sorted(adapter_methods)}",
            )
        )
    for label, path in (
        ("full controlled core", full_core_status),
        ("full controlled edge-prior core", edge_prior_core_status),
        ("full controlled adapter-light", full_adapter_status),
    ):
        if not path.exists():
            continue
        status_rows = _read_csv(path)
        rows.append(
            AuditRow(
                "exp03_baseline_comparison",
                f"{label} baseline status rows include reproducibility details",
                "complete" if _baseline_status_details_valid(status_rows) else "partial",
                str(path.relative_to(PROJECT_ROOT)),
                f"n_status_rows={len(status_rows)}",
            )
        )
    if unified_summary.exists() and unified_status.exists():
        rows.extend(_audit_unified_full_baseline(unified_summary, unified_status, unified_manifest))
    if full_slide_sparse_summary.exists() and full_slide_sparse_status.exists():
        rows.extend(_audit_full_slide_sparse_baseline(full_slide_sparse_summary, full_slide_sparse_status))
    return rows


def _audit_exp04() -> list[AuditRow]:
    full_summary = PROJECT_ROOT / "outputs/phase1/exp04_model_ablation/full_fair_focused_ablation/spatialself_ablation_summary.csv"
    smoke_summary = PROJECT_ROOT / "outputs/phase1/exp04_model_ablation/smoke_controlled/phase_runner/spatialself_ablation_summary.csv"
    calibration_summary = PROJECT_ROOT / "outputs/phase1/exp04_model_ablation/full_fair_score_calibration_nonoracle/calibration_metrics_summary.csv"
    edge_prior_summary = PROJECT_ROOT / "outputs/phase1/exp04_model_ablation/full_edge_prior_focused_ablation_epoch10/spatialself_ablation_summary.csv"
    architecture_root = PROJECT_ROOT / "outputs/phase1/exp04_model_ablation/full_architecture_identity_component_ablation_epoch10"
    architecture_summary = architecture_root / "spatialself_ablation_summary.csv"
    architecture_status = architecture_root / "spatialself_ablation_status.csv"
    rows = [
        AuditRow("exp04_model_ablation", "full fair focused ablation summary exists", _exists_status(full_summary), str(full_summary.relative_to(PROJECT_ROOT))),
        AuditRow("exp04_model_ablation", "phase-runner ablation smoke summary exists", _exists_status(smoke_summary, existing_status="smoke_ready"), str(smoke_summary.relative_to(PROJECT_ROOT))),
        AuditRow("exp04_model_ablation", "full fair score-calibration diagnostic exists", _exists_status(calibration_summary), str(calibration_summary.relative_to(PROJECT_ROOT))),
        AuditRow("exp04_model_ablation", "full edge-prior focused ablation diagnostic exists", _exists_status(edge_prior_summary), str(edge_prior_summary.relative_to(PROJECT_ROOT))),
        AuditRow("exp04_model_ablation", "full architecture/identity/component ablation summary exists", _exists_status(architecture_summary), str(architecture_summary.relative_to(PROJECT_ROOT))),
        AuditRow("exp04_model_ablation", "full architecture/identity/component ablation status exists", _exists_status(architecture_status), str(architecture_status.relative_to(PROJECT_ROOT))),
    ]
    ablations = set()
    if full_summary.exists():
        ablations.update(row.get("ablation", "") for row in _read_csv(full_summary))
    if smoke_summary.exists():
        ablations.update(row.get("ablation", "") for row in _read_csv(smoke_summary))
    if architecture_summary.exists():
        ablations.update(row.get("ablation", "") for row in _read_csv(architecture_summary))
    required_present = {
        "gat_hard_expression_distance",
        "wavelet_hard_expression_distance",
        "wavelet_no_lr_embedding",
        "wavelet_fixed_random_lr_embedding",
        "wavelet_one_hot_lr_identity",
        "wavelet_no_distance",
        "wavelet_no_spatial_graph",
        "gcn_hard_expression_distance",
        "mlp_hard_expression_distance",
    }
    missing_core = sorted(required_present - ablations)
    core_evidence = architecture_summary if architecture_summary.exists() else smoke_summary
    rows.append(
        AuditRow(
            "exp04_model_ablation",
            "core identity/architecture/component ablations represented",
            "complete" if architecture_summary.exists() and not missing_core else "smoke_ready" if not missing_core else "partial",
            str(core_evidence.relative_to(PROJECT_ROOT)),
            f"missing={missing_core}",
        )
    )
    required_metric_columns = {
        "auroc_mean",
        "aupr_mean",
        "weak_signal_recall_at_threshold_mean",
        "empirical_fdr_at_threshold_mean",
        "false_positive_high_expression_rate_at_threshold_mean",
        "calibration_error_10bin_mean",
        "n_pseudo_positive_mean",
        "n_pseudo_negative_mean",
        "n_ranking_pairs_mean",
    }
    summary_tables = [
        ("full fair focused", full_summary, "complete"),
        ("phase-runner smoke", smoke_summary, "smoke_ready"),
        ("edge-prior focused", edge_prior_summary, "complete"),
        ("full architecture/identity/component", architecture_summary, "complete"),
    ]
    for label, path, status in summary_tables:
        if not path.exists():
            continue
        records = _read_csv(path)
        missing_columns = sorted(required_metric_columns - set(records[0] if records else {}))
        metric_values_ok = _ablation_metric_values_valid(records, required_metric_columns)
        rows.append(
            AuditRow(
                "exp04_model_ablation",
                f"{label} ablation summary has weak/FDR/false-positive metric columns",
                status if records and not missing_columns and metric_values_ok else "partial",
                str(path.relative_to(PROJECT_ROOT)),
                f"missing={missing_columns}; n_rows={len(records)}",
            )
        )
    if architecture_summary.exists():
        architecture_rows = _read_csv(architecture_summary)
        by_ablation = {row.get("ablation", ""): row for row in architecture_rows}
        required_architecture = {
            "gat_hard_expression_distance",
            "wavelet_hard_expression_distance",
            "gcn_hard_expression_distance",
            "mlp_hard_expression_distance",
            "wavelet_no_lr_embedding",
            "wavelet_fixed_random_lr_embedding",
            "wavelet_one_hot_lr_identity",
            "wavelet_no_distance",
            "wavelet_no_spatial_graph",
            "wavelet_no_expression_features",
        }
        missing_architecture = sorted(required_architecture - set(by_ablation))
        replicate_counts_ok = all(
            int(float(row.get("aupr_count", 0) or 0)) >= 5
            and int(float(row.get("weak_signal_recall_at_threshold_count", 0) or 0)) >= 5
            for row in architecture_rows
        )
        metric_files = list(architecture_root.glob("*/*_metrics.json"))
        rows.append(
            AuditRow(
                "exp04_model_ablation",
                "full architecture/identity/component ablation covers all planned variants across five replicates",
                "complete" if not missing_architecture and replicate_counts_ok and len(metric_files) >= 50 else "partial",
                str(architecture_summary.relative_to(PROJECT_ROOT)),
                f"missing={missing_architecture}; n_metric_files={len(metric_files)}",
            )
        )
        architecture_set = {
            "gat_hard_expression_distance",
            "wavelet_hard_expression_distance",
            "gcn_hard_expression_distance",
            "mlp_hard_expression_distance",
            "wavelet_no_spatial_graph",
        }
        identity_set = {
            "wavelet_hard_expression_distance",
            "wavelet_no_lr_embedding",
            "wavelet_fixed_random_lr_embedding",
            "wavelet_one_hot_lr_identity",
        }
        component_set = {
            "wavelet_hard_expression_distance",
            "wavelet_no_distance",
            "wavelet_no_spatial_graph",
            "wavelet_no_expression_features",
        }
        rows.append(
            AuditRow(
                "exp04_model_ablation",
                "full ablation supports graph-wavelet versus GAT/GCN/MLP/no-graph comparison",
                "complete" if architecture_set.issubset(by_ablation) and _ablation_metric_values_valid([by_ablation[name] for name in architecture_set], required_metric_columns) else "partial",
                str(architecture_summary.relative_to(PROJECT_ROOT)),
                f"variants={sorted(architecture_set & set(by_ablation))}",
            )
        )
        rows.append(
            AuditRow(
                "exp04_model_ablation",
                "full ablation supports LR identity embedding comparison",
                "complete" if identity_set.issubset(by_ablation) and _ablation_metric_values_valid([by_ablation[name] for name in identity_set], required_metric_columns) else "partial",
                str(architecture_summary.relative_to(PROJECT_ROOT)),
                f"variants={sorted(identity_set & set(by_ablation))}",
            )
        )
        rows.append(
            AuditRow(
                "exp04_model_ablation",
                "full ablation supports distance, spatial graph, and expression component comparison",
                "complete" if component_set.issubset(by_ablation) and _ablation_metric_values_valid([by_ablation[name] for name in component_set], required_metric_columns) else "partial",
                str(architecture_summary.relative_to(PROJECT_ROOT)),
                f"variants={sorted(component_set & set(by_ablation))}",
            )
        )
    if architecture_status.exists():
        status_rows = _read_csv(architecture_status)
        completed = [row for row in status_rows if row.get("status") == "completed"]
        rows.append(
            AuditRow(
                "exp04_model_ablation",
                "full architecture/identity/component ablation statuses completed",
                "complete" if len(completed) >= 10 and len(completed) == len(status_rows) else "partial",
                str(architecture_status.relative_to(PROJECT_ROOT)),
                f"n_completed={len(completed)}; n_status_rows={len(status_rows)}",
            )
        )
    if full_summary.exists():
        fair_rows = _read_csv(full_summary)
        fair_ablations = {row.get("ablation", "") for row in fair_rows}
        required_fair = {
            "wavelet_mechanism_niche_lr_capped",
            "wavelet_mechanism_niche_weak_augmented",
            "wavelet_mechanism_niche_lr_prior_lrwise_ranking",
        }
        rows.append(
            AuditRow(
                "exp04_model_ablation",
                "full fair ablation covers pseudo-label, weak augmentation, and LR-wise ranking",
                "complete" if required_fair.issubset(fair_ablations) else "partial",
                str(full_summary.relative_to(PROJECT_ROOT)),
                f"missing={sorted(required_fair - fair_ablations)}",
            )
        )
        weak_aug = _row_by_key(fair_rows, "ablation", "wavelet_mechanism_niche_weak_augmented")
        lrwise = _row_by_key(fair_rows, "ablation", "wavelet_mechanism_niche_lr_prior_lrwise_ranking")
        lr_capped = _row_by_key(fair_rows, "ablation", "wavelet_mechanism_niche_lr_capped")
        weak_gain = _to_float(weak_aug.get("weak_signal_recall_at_threshold_mean", "nan")) - _to_float(
            lr_capped.get("weak_signal_recall_at_threshold_mean", "nan")
        )
        fp_reduction = _to_float(lr_capped.get("false_positive_high_expression_rate_at_threshold_mean", "nan")) - _to_float(
            lrwise.get("false_positive_high_expression_rate_at_threshold_mean", "nan")
        )
        rows.append(
            AuditRow(
                "exp04_model_ablation",
                "full fair ablation quantifies weak-signal gain and false-positive suppression tradeoff",
                "complete" if math.isfinite(weak_gain) and math.isfinite(fp_reduction) else "partial",
                str(full_summary.relative_to(PROJECT_ROOT)),
                f"weak_aug_minus_lr_capped={weak_gain:.3f}; fp_reduction_lrwise_vs_lr_capped={fp_reduction:.3f}",
            )
        )
    if calibration_summary.exists():
        calibration_rows = _read_csv(calibration_summary)
        transforms = {row.get("score_transform", "") for row in calibration_rows}
        required_transforms = {"raw", "lr_rank", "lr_mechanism_distance_rank", "lr_mechanism_prior_rank"}
        raw_rows = [row for row in calibration_rows if row.get("score_transform") == "raw"]
        distance_rows = [
            row for row in calibration_rows if row.get("score_transform") == "lr_mechanism_distance_rank"
        ]
        raw_aupr = float(raw_rows[0].get("aupr_mean", 0) or 0) if raw_rows else 0.0
        distance_aupr = float(distance_rows[0].get("aupr_mean", 0) or 0) if distance_rows else 0.0
        rows.append(
            AuditRow(
                "exp04_model_ablation",
                "score-calibration transforms represented",
                "complete" if required_transforms.issubset(transforms) else "partial",
                str(calibration_summary.relative_to(PROJECT_ROOT)),
                f"transforms={sorted(transforms)}",
            )
        )
        rows.append(
            AuditRow(
                "exp04_model_ablation",
                "non-oracle score calibration recorded as diagnostic rather than default",
                "complete" if raw_aupr >= distance_aupr else "partial",
                str(calibration_summary.relative_to(PROJECT_ROOT)),
                f"raw_aupr_mean={raw_aupr:.3f}; lr_mechanism_distance_aupr_mean={distance_aupr:.3f}",
            )
        )
    if edge_prior_summary.exists():
        edge_rows = _read_csv(edge_prior_summary)
        by_ablation = {row.get("ablation", ""): row for row in edge_rows}
        required_edge = {
            "wavelet_mechanism_niche_lr_prior_lrwise_ranking",
            "wavelet_mechanism_niche_edge_prior",
            "wavelet_mechanism_niche_edge_prior_soft_weak",
            "wavelet_mechanism_niche_edge_prior_ranking",
        }
        edge_ablations = set(by_ablation)
        rows.append(
            AuditRow(
                "exp04_model_ablation",
                "edge-prior focused ablation covers plain, soft-weak, and ranking variants",
                "complete" if required_edge.issubset(edge_ablations) else "partial",
                str(edge_prior_summary.relative_to(PROJECT_ROOT)),
                f"missing={sorted(required_edge - edge_ablations)}",
            )
        )
        baseline = by_ablation.get("wavelet_mechanism_niche_lr_prior_lrwise_ranking", {})
        candidate = by_ablation.get("wavelet_mechanism_niche_edge_prior_ranking", {})
        candidate_aupr = float(candidate.get("aupr_mean", 0) or 0)
        baseline_aupr = float(baseline.get("aupr_mean", 0) or 0)
        candidate_weak = float(candidate.get("weak_signal_recall_at_threshold_mean", 0) or 0)
        baseline_weak = float(baseline.get("weak_signal_recall_at_threshold_mean", 0) or 0)
        rows.append(
            AuditRow(
                "exp04_model_ablation",
                "edge-prior ranking candidate evaluated against current fair default",
                "complete" if candidate_aupr >= baseline_aupr and candidate_weak >= baseline_weak else "partial",
                str(edge_prior_summary.relative_to(PROJECT_ROOT)),
                (
                    f"default_aupr={baseline_aupr:.3f}; candidate_aupr={candidate_aupr:.3f}; "
                    f"default_weak={baseline_weak:.3f}; candidate_weak={candidate_weak:.3f}"
                ),
            )
        )
    return rows


def _audit_exp05() -> list[AuditRow]:
    root = PROJECT_ROOT / "outputs/phase1/exp05_weak_signal_false_positive/smoke_controlled/phase_runner"
    metrics = root / "weak_signal_false_positive_metrics.csv"
    status = root / "weak_signal_false_positive_status.csv"
    stress_dir = root / "stress_tables"
    full_root = PROJECT_ROOT / "outputs/phase1/exp05_weak_signal_false_positive/full_controlled_spatialself_stress"
    full_metrics = full_root / "weak_signal_false_positive_metrics.csv"
    full_status = full_root / "weak_signal_false_positive_status.csv"
    full_stress_dir = full_root / "stress_tables"
    required_methods = {"spatialself_wavelet", "spatialself_wavelet_edge_prior", "spatialself_gat"}
    rows = [
        AuditRow("exp05_weak_signal_false_positive", "stress smoke metrics exists", _exists_status(metrics, existing_status="smoke_ready"), str(metrics.relative_to(PROJECT_ROOT))),
        AuditRow("exp05_weak_signal_false_positive", "stress smoke status exists", _exists_status(status, existing_status="smoke_ready"), str(status.relative_to(PROJECT_ROOT))),
        AuditRow("exp05_weak_signal_false_positive", "full controlled SpatialSelf stress metrics exists", _exists_status(full_metrics), str(full_metrics.relative_to(PROJECT_ROOT))),
        AuditRow("exp05_weak_signal_false_positive", "full controlled SpatialSelf stress status exists", _exists_status(full_status), str(full_status.relative_to(PROJECT_ROOT))),
    ]
    if metrics.exists():
        metric_rows = _read_csv(metrics)
        methods = sorted({row.get("method", "") for row in metric_rows})
        rows.append(
            AuditRow(
                "exp05_weak_signal_false_positive",
                "wavelet, edge-prior candidate, and GAT stress methods represented",
                "smoke_ready" if required_methods.issubset(methods) else "partial",
                str(metrics.relative_to(PROJECT_ROOT)),
                f"missing={sorted(required_methods - set(methods))}; methods={methods}",
            )
        )
        rows.append(
            AuditRow(
                "exp05_weak_signal_false_positive",
                "stress metrics include AUROC/AUPR, weak recall, FDR, high-expression FP rate, and calibration",
                "smoke_ready" if _weak_stress_metric_table_valid(metric_rows) else "partial",
                str(metrics.relative_to(PROJECT_ROOT)),
                f"n_rows={len(metric_rows)}",
            )
        )
        if {"spatialself_wavelet", "spatialself_wavelet_edge_prior"}.issubset(methods):
            default = _row_by_key(metric_rows, "method", "spatialself_wavelet")
            candidate = _row_by_key(metric_rows, "method", "spatialself_wavelet_edge_prior")
            default_fp = _to_float(default.get("false_positive_high_expression_rate_at_threshold", "nan"))
            candidate_fp = _to_float(candidate.get("false_positive_high_expression_rate_at_threshold", "nan"))
            default_weak = _to_float(default.get("weak_signal_recall_at_threshold", "nan"))
            candidate_weak = _to_float(candidate.get("weak_signal_recall_at_threshold", "nan"))
            rows.append(
                AuditRow(
                    "exp05_weak_signal_false_positive",
                    "edge-prior smoke suppresses high-expression false positives with weak-recall tradeoff recorded",
                    "smoke_ready"
                    if (
                        math.isfinite(default_fp)
                        and math.isfinite(candidate_fp)
                        and math.isfinite(default_weak)
                        and math.isfinite(candidate_weak)
                        and candidate_fp <= default_fp
                    )
                    else "partial",
                    str(metrics.relative_to(PROJECT_ROOT)),
                    (
                        f"default_fp_rate={default_fp:.4f}; edge_prior_fp_rate={candidate_fp:.4f}; "
                        f"default_weak_recall={default_weak:.4f}; edge_prior_weak_recall={candidate_weak:.4f}"
                    ),
                )
            )
    if status.exists():
        status_rows = _read_csv(status)
        rows.append(
            AuditRow(
                "exp05_weak_signal_false_positive",
                "stress status rows completed with prediction paths",
                "smoke_ready" if _weak_stress_status_valid(status_rows, required_methods) else "partial",
                str(status.relative_to(PROJECT_ROOT)),
                f"n_rows={len(status_rows)}",
            )
        )
    if stress_dir.exists():
        rows.extend(_audit_weak_stress_tables(stress_dir, required_methods))
    else:
        rows.append(
            AuditRow(
                "exp05_weak_signal_false_positive",
                "stress table directory exists",
                "missing",
                str(stress_dir.relative_to(PROJECT_ROOT)),
            )
        )
    if full_metrics.exists():
        full_metric_rows = _read_csv(full_metrics)
        rows.extend(_audit_full_weak_stress_metrics(full_metrics, full_metric_rows, required_methods))
    if full_status.exists():
        full_status_rows = _read_csv(full_status)
        rows.append(
            AuditRow(
                "exp05_weak_signal_false_positive",
                "full stress status rows complete with prediction paths",
                "complete" if _full_weak_stress_status_valid(full_status_rows, required_methods, min_replicates=5) else "partial",
                str(full_status.relative_to(PROJECT_ROOT)),
                f"n_rows={len(full_status_rows)}",
            )
        )
    if full_stress_dir.exists():
        rows.extend(
            _audit_weak_stress_tables(
                full_stress_dir,
                required_methods,
                existing_status="complete",
                min_replicates=5,
            )
        )
    return rows


def _audit_exp06() -> list[AuditRow]:
    root = PROJECT_ROOT / "outputs/phase1/exp06_permutation_fdr/smoke_spatialself_wavelet_fair_phase_runner"
    summary = root / "permutation_fdr_summary.csv"
    edge_root = PROJECT_ROOT / "outputs/phase1/exp06_permutation_fdr/smoke_spatialself_wavelet_edge_prior_phase_runner"
    edge_summary = edge_root / "permutation_fdr_summary.csv"
    rows = [
        AuditRow("exp06_permutation_fdr", "fair FDR smoke summary exists", _exists_status(summary, existing_status="smoke_ready"), str(summary.relative_to(PROJECT_ROOT))),
        AuditRow("exp06_permutation_fdr", "edge-prior candidate FDR smoke summary exists", _exists_status(edge_summary, existing_status="smoke_ready"), str(edge_summary.relative_to(PROJECT_ROOT))),
    ]
    if summary.exists():
        nulls = {row.get("null_model", "") for row in _read_csv(summary)}
        required = {"spatial_permutation", "receiver_permutation", "lr_identity_permutation"}
        rows.append(
            AuditRow(
                "exp06_permutation_fdr",
                "three required null models represented",
                "smoke_ready" if required.issubset(nulls) else "partial",
                str(summary.relative_to(PROJECT_ROOT)),
                f"null_models={sorted(nulls)}",
            )
        )
        rows.extend(_audit_fdr_summary_schema("fair", summary, existing_status="smoke_ready"))
    full_summary = PROJECT_ROOT / "outputs/phase1/exp06_permutation_fdr/full_controlled_replicate00_spatialself_lr_prior/permutation_fdr_summary.csv"
    edge_full_summary = PROJECT_ROOT / "outputs/phase1/exp06_permutation_fdr/full_controlled_replicate00_spatialself_edge_prior/permutation_fdr_summary.csv"
    panel_root = PROJECT_ROOT / "outputs/phase1/exp06_permutation_fdr/full_controlled_spatialself_panel"
    panel_summary = panel_root / "permutation_fdr_panel_summary.csv"
    panel_status = panel_root / "permutation_fdr_panel_status.csv"
    panel_method_summary = panel_root / "permutation_fdr_panel_method_summary.csv"
    panel_manifest = panel_root / "permutation_fdr_panel_manifest.json"
    sparse_panel_root = PROJECT_ROOT / "outputs/phase1/exp06_permutation_fdr/full_slide_sparse_semisynthetic_panel"
    sparse_panel_summary = sparse_panel_root / "permutation_fdr_panel_summary.csv"
    sparse_panel_status = sparse_panel_root / "permutation_fdr_panel_status.csv"
    sparse_panel_method_summary = sparse_panel_root / "permutation_fdr_panel_method_summary.csv"
    sparse_panel_manifest = sparse_panel_root / "permutation_fdr_panel_manifest.json"
    rows.append(
        AuditRow("exp06_permutation_fdr", "full replicate00 FDR pilot exists", _exists_status(full_summary), str(full_summary.relative_to(PROJECT_ROOT)))
    )
    rows.append(
        AuditRow("exp06_permutation_fdr", "edge-prior full replicate00 FDR pilot exists", _exists_status(edge_full_summary), str(edge_full_summary.relative_to(PROJECT_ROOT)))
    )
    rows.extend(
        [
            AuditRow("exp06_permutation_fdr", "full controlled SpatialSelf FDR panel summary exists", _exists_status(panel_summary), str(panel_summary.relative_to(PROJECT_ROOT))),
            AuditRow("exp06_permutation_fdr", "full controlled SpatialSelf FDR panel status exists", _exists_status(panel_status), str(panel_status.relative_to(PROJECT_ROOT))),
            AuditRow("exp06_permutation_fdr", "full controlled SpatialSelf FDR panel method summary exists", _exists_status(panel_method_summary), str(panel_method_summary.relative_to(PROJECT_ROOT))),
            AuditRow("exp06_permutation_fdr", "full controlled SpatialSelf FDR panel manifest exists", _exists_status(panel_manifest), str(panel_manifest.relative_to(PROJECT_ROOT))),
            AuditRow("exp06_permutation_fdr", "full-slide sparse semi-synthetic FDR panel summary exists", _exists_status(sparse_panel_summary), str(sparse_panel_summary.relative_to(PROJECT_ROOT))),
            AuditRow("exp06_permutation_fdr", "full-slide sparse semi-synthetic FDR panel status exists", _exists_status(sparse_panel_status), str(sparse_panel_status.relative_to(PROJECT_ROOT))),
            AuditRow("exp06_permutation_fdr", "full-slide sparse semi-synthetic FDR panel method summary exists", _exists_status(sparse_panel_method_summary), str(sparse_panel_method_summary.relative_to(PROJECT_ROOT))),
            AuditRow("exp06_permutation_fdr", "full-slide sparse semi-synthetic FDR panel manifest exists", _exists_status(sparse_panel_manifest), str(sparse_panel_manifest.relative_to(PROJECT_ROOT))),
        ]
    )
    if edge_summary.exists():
        nulls = {row.get("null_model", "") for row in _read_csv(edge_summary)}
        required = {"spatial_permutation", "receiver_permutation", "lr_identity_permutation"}
        rows.append(
            AuditRow(
                "exp06_permutation_fdr",
                "edge-prior FDR smoke has three required null models",
                "smoke_ready" if required.issubset(nulls) else "partial",
                str(edge_summary.relative_to(PROJECT_ROOT)),
                f"null_models={sorted(nulls)}",
            )
        )
        rows.extend(_audit_fdr_summary_schema("edge-prior", edge_summary, existing_status="smoke_ready"))
    if panel_summary.exists() and panel_status.exists() and panel_method_summary.exists():
        rows.extend(
            _audit_fdr_panel(
                panel_summary,
                panel_status,
                panel_method_summary,
                panel_manifest,
                label="full controlled SpatialSelf",
                required_methods={"spatialself_wavelet", "spatialself_wavelet_edge_prior"},
                min_replicates=5,
                min_permutations=100,
            )
        )
    if sparse_panel_summary.exists() and sparse_panel_status.exists() and sparse_panel_method_summary.exists():
        rows.extend(
            _audit_fdr_panel(
                sparse_panel_summary,
                sparse_panel_status,
                sparse_panel_method_summary,
                sparse_panel_manifest,
                label="full-slide sparse semi-synthetic",
                required_methods={"spatialself_wavelet", "spatialself_wavelet_edge_prior", "commot"},
                min_replicates=1,
                min_permutations=50,
            )
        )
    return rows


def _audit_fdr_panel(
    panel_summary: Path,
    panel_status: Path,
    panel_method_summary: Path,
    panel_manifest: Path,
    *,
    label: str,
    required_methods: set[str],
    min_replicates: int,
    min_permutations: int,
) -> list[AuditRow]:
    rows: list[AuditRow] = []
    summary_rows = _read_csv(panel_summary)
    status_rows = _read_csv(panel_status)
    method_rows = _read_csv(panel_method_summary)
    required_nulls = {"spatial_permutation", "receiver_permutation", "lr_identity_permutation"}
    methods = {row.get("method_panel", "") for row in summary_rows}
    nulls = {row.get("null_model", "") for row in summary_rows}
    replicates = {row.get("replicate", "") for row in summary_rows}
    completed = [row for row in status_rows if row.get("status") == "completed"]
    expected_summary_rows = len(required_methods) * len(required_nulls) * min_replicates
    coverage_ok = (
        required_methods.issubset(methods)
        and required_nulls.issubset(nulls)
        and len(replicates) >= min_replicates
        and len(summary_rows) >= expected_summary_rows
        and len(completed) >= len(required_methods) * min_replicates
        and len(completed) == len(status_rows)
    )
    rows.append(
        AuditRow(
            "exp06_permutation_fdr",
            f"{label} FDR panel covers required methods, replicates, and three null models",
            "complete" if coverage_ok else "partial",
            str(panel_summary.relative_to(PROJECT_ROOT)),
            (
                f"methods={sorted(methods)}; nulls={sorted(nulls)}; "
                f"n_replicates={len(replicates)}; n_summary_rows={len(summary_rows)}; "
                f"n_completed={len(completed)}"
            ),
        )
    )
    required_summary_columns = {
        "method_panel",
        "replicate",
        "null_model",
        "n_candidates",
        "n_significant_edges",
        "min_p_value",
        "min_bh_fdr",
        "truth_diagnostics_available",
        "realized_recall",
        "realized_weak_signal_recall",
        "realized_false_positive_high_expression_call_rate",
        "p_value_ks_uniform_stat",
        "fdr_path",
        "lr_pair_fdr_path",
        "cell_type_pair_fdr_path",
        "n_permutations",
    }
    missing_summary_columns = sorted(required_summary_columns - set(summary_rows[0] if summary_rows else {}))
    summary_values_ok = (
        bool(summary_rows)
        and not missing_summary_columns
        and _probability_columns_in_unit_interval(
            summary_rows,
            (
                "min_p_value",
                "min_bh_fdr",
                "realized_recall",
                "realized_weak_signal_recall",
                "realized_false_positive_high_expression_call_rate",
                "p_value_ks_uniform_stat",
            ),
        )
        and all(int(float(row.get("n_permutations", 0) or 0)) >= min_permutations for row in summary_rows)
    )
    rows.append(
        AuditRow(
            "exp06_permutation_fdr",
            f"{label} FDR panel summary schema and probability fields are valid",
            "complete" if summary_values_ok else "partial",
            str(panel_summary.relative_to(PROJECT_ROOT)),
            f"missing={missing_summary_columns}; n_rows={len(summary_rows)}",
        )
    )
    path_ok = True
    path_labels: list[str] = []
    for row in summary_rows:
        path_labels.append(f"{row.get('method_panel')}:{row.get('replicate')}:{row.get('null_model')}")
        for column in ("fdr_path", "lr_pair_fdr_path", "cell_type_pair_fdr_path"):
            target = _project_path(row.get(column, ""))
            if not target.exists():
                path_ok = False
                continue
            target_rows = _read_csv(target)
            if not target_rows:
                path_ok = False
    rows.append(
        AuditRow(
            "exp06_permutation_fdr",
            f"{label} FDR panel writes edge, LR-pair, and cell-type-pair result files for every run",
            "complete" if path_ok and len(path_labels) >= expected_summary_rows else "partial",
            str(panel_summary.relative_to(PROJECT_ROOT)),
            f"n_checked_runs={len(path_labels)}",
        )
    )
    method_pairs = {(row.get("method_panel", ""), row.get("null_model", "")) for row in method_rows}
    method_replicates_ok = all(
        int(float(row.get("n_replicates", 0) or 0)) >= min_replicates for row in method_rows
    )
    rows.append(
        AuditRow(
            "exp06_permutation_fdr",
            f"{label} FDR panel method-null summary aggregates all replicates",
            "complete"
            if required_methods.issubset({method for method, _ in method_pairs})
            and required_nulls.issubset({null for _, null in method_pairs})
            and method_replicates_ok
            else "partial",
            str(panel_method_summary.relative_to(PROJECT_ROOT)),
            f"n_method_null_rows={len(method_rows)}",
        )
    )
    if panel_manifest.exists():
        try:
            manifest = json.loads(panel_manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
        manifest_ok = (
            int(manifest.get("n_replicates", 0) or 0) >= min_replicates
            and int(manifest.get("n_completed_runs", 0) or 0) >= len(required_methods) * min_replicates
            and int(manifest.get("n_failed_runs", -1)) == 0
            and set(manifest.get("null_models", [])) >= required_nulls
        )
        rows.append(
            AuditRow(
                "exp06_permutation_fdr",
                f"{label} FDR panel manifest records completed run design",
                "complete" if manifest_ok else "partial",
                str(panel_manifest.relative_to(PROJECT_ROOT)),
                (
                    f"n_replicates={manifest.get('n_replicates')}; "
                    f"n_completed_runs={manifest.get('n_completed_runs')}; "
                    f"n_failed_runs={manifest.get('n_failed_runs')}"
                ),
            )
        )
    return rows


def _audit_fdr_summary_schema(label: str, summary_path: Path, *, existing_status: str) -> list[AuditRow]:
    rows: list[AuditRow] = []
    summary_rows = _read_csv(summary_path)
    required_summary_columns = {
        "null_model",
        "n_candidates",
        "n_significant_edges",
        "min_p_value",
        "min_bh_fdr",
        "truth_diagnostics_available",
        "realized_fdr",
        "realized_weak_signal_recall",
        "realized_false_positive_high_expression_call_rate",
        "p_value_ks_uniform_stat",
        "fdr_path",
        "lr_pair_fdr_path",
        "cell_type_pair_fdr_path",
    }
    missing_summary = sorted(required_summary_columns - set(summary_rows[0] if summary_rows else {}))
    rows.append(
        AuditRow(
            "exp06_permutation_fdr",
            f"{label} FDR summary has p-value, BH-FDR, truth, and output path columns",
            existing_status if summary_rows and not missing_summary else "partial",
            str(summary_path.relative_to(PROJECT_ROOT)),
            f"missing={missing_summary}",
        )
    )

    required_edge_columns = {
        "candidate_id",
        "method",
        "null_model",
        "score",
        "empirical_p_value",
        "bh_fdr",
        "null_mean",
        "null_std",
        "n_permutations",
    }
    required_group_columns = {
        "null_model",
        "method",
        "n_candidate_edges",
        "n_significant_edges",
        "min_edge_p_value",
        "group_p_value",
        "group_bh_fdr",
        "group_significant",
    }
    edge_ok = True
    group_ok = True
    p_value_ok = True
    path_details = []
    for row in summary_rows:
        fdr_path = _project_path(row.get("fdr_path", ""))
        lr_path = _project_path(row.get("lr_pair_fdr_path", ""))
        cell_type_path = _project_path(row.get("cell_type_pair_fdr_path", ""))
        path_details.append(row.get("null_model", "unknown"))
        if not fdr_path.exists():
            edge_ok = False
        else:
            edge_rows = _read_csv(fdr_path)
            edge_ok = edge_ok and bool(edge_rows) and required_edge_columns.issubset(edge_rows[0])
            p_value_ok = p_value_ok and _probability_columns_in_unit_interval(
                edge_rows,
                ("empirical_p_value", "bh_fdr"),
            )
        for group_path in (lr_path, cell_type_path):
            if not group_path.exists():
                group_ok = False
                continue
            group_rows = _read_csv(group_path)
            group_ok = group_ok and bool(group_rows) and required_group_columns.issubset(group_rows[0])
            p_value_ok = p_value_ok and _probability_columns_in_unit_interval(
                group_rows,
                ("min_edge_p_value", "group_p_value", "group_bh_fdr"),
            )
    rows.append(
        AuditRow(
            "exp06_permutation_fdr",
            f"{label} FDR edge-level schema and p-value ranges are valid",
            existing_status if edge_ok and p_value_ok else "partial",
            str(summary_path.relative_to(PROJECT_ROOT)),
            f"null_models={path_details}",
        )
    )
    rows.append(
        AuditRow(
            "exp06_permutation_fdr",
            f"{label} FDR LR-pair and cell-type-pair group schemas are valid",
            existing_status if group_ok and p_value_ok else "partial",
            str(summary_path.relative_to(PROJECT_ROOT)),
            f"null_models={path_details}",
        )
    )

    truth_available = all(_as_bool(row.get("truth_diagnostics_available", "")) for row in summary_rows)
    ks_values = [_to_float(row.get("p_value_ks_uniform_stat", "nan")) for row in summary_rows]
    ks_ok = bool(ks_values) and all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in ks_values)
    min_p_values = [_to_float(row.get("min_p_value", "nan")) for row in summary_rows]
    min_bh_values = [_to_float(row.get("min_bh_fdr", "nan")) for row in summary_rows]
    calibration_ok = (
        truth_available
        and ks_ok
        and all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in min_p_values)
        and all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in min_bh_values)
    )
    rows.append(
        AuditRow(
            "exp06_permutation_fdr",
            f"{label} FDR truth diagnostics and p-value calibration fields are finite",
            existing_status if calibration_ok else "partial",
            str(summary_path.relative_to(PROJECT_ROOT)),
            f"ks_values={[round(value, 4) for value in ks_values if math.isfinite(value)]}",
        )
    )
    return rows


def _audit_exp07() -> list[AuditRow]:
    root = PROJECT_ROOT / "outputs/phase1/exp07_lymph_node_global_validation/smoke_phase_runner"
    full_root = PROJECT_ROOT / "outputs/phase1/exp07_lymph_node_global_validation"
    region = root / "region_enrichment/human_lymph_node_visium_replicate_00__spatialself_wavelet_predictions__region_enrichment.csv"
    ranking = root / "region_enrichment/human_lymph_node_visium_replicate_00__spatialself_wavelet_predictions__lr_global_ranking.csv"
    downstream = root / "downstream_target/human_lymph_node_visium_replicate_00__spatialself_wavelet_predictions__downstream_target_correlation.csv"
    edge_region = root / "region_enrichment/human_lymph_node_visium_replicate_00__spatialself_wavelet_edge_prior_predictions__region_enrichment.csv"
    edge_ranking = root / "region_enrichment/human_lymph_node_visium_replicate_00__spatialself_wavelet_edge_prior_predictions__lr_global_ranking.csv"
    edge_downstream = root / "downstream_target/human_lymph_node_visium_replicate_00__spatialself_wavelet_edge_prior_predictions__downstream_target_correlation.csv"
    control_lr = root / "control_summary/real_validation_control_lr_summary.csv"
    control_method = root / "control_summary/real_validation_control_method_summary.csv"
    full_region = full_root / "region_enrichment/human_lymph_node_visium_replicate_00__spatialself_wavelet_predictions__region_enrichment.csv"
    full_ranking = full_root / "region_enrichment/human_lymph_node_visium_replicate_00__spatialself_wavelet_predictions__lr_global_ranking.csv"
    full_downstream = full_root / "downstream_target/human_lymph_node_visium_replicate_00__spatialself_wavelet_predictions__downstream_target_correlation.csv"
    full_edge_region = full_root / "region_enrichment/human_lymph_node_visium_replicate_00__spatialself_wavelet_edge_prior_predictions__region_enrichment.csv"
    full_edge_ranking = full_root / "region_enrichment/human_lymph_node_visium_replicate_00__spatialself_wavelet_edge_prior_predictions__lr_global_ranking.csv"
    full_edge_downstream = full_root / "downstream_target/human_lymph_node_visium_replicate_00__spatialself_wavelet_edge_prior_predictions__downstream_target_correlation.csv"
    full_control_lr = full_root / "control_summary/real_validation_control_lr_summary.csv"
    full_control_method = full_root / "control_summary/real_validation_control_method_summary.csv"
    full_slide_sparse_region = full_root / "full_slide_sparse_region_enrichment/spatialself_wavelet/human_lymph_node_visium_replicate_00__spatialself_wavelet_predictions__region_enrichment.csv"
    full_slide_sparse_ranking = full_root / "full_slide_sparse_region_enrichment/spatialself_wavelet/human_lymph_node_visium_replicate_00__spatialself_wavelet_predictions__lr_global_ranking.csv"
    full_slide_sparse_downstream = full_root / "full_slide_sparse_downstream_target/spatialself_wavelet/human_lymph_node_visium_replicate_00__spatialself_wavelet_predictions__downstream_target_correlation.csv"
    full_slide_sparse_edge_region = full_root / "full_slide_sparse_region_enrichment/spatialself_wavelet_edge_prior/human_lymph_node_visium_replicate_00__spatialself_wavelet_edge_prior_predictions__region_enrichment.csv"
    full_slide_sparse_edge_ranking = full_root / "full_slide_sparse_region_enrichment/spatialself_wavelet_edge_prior/human_lymph_node_visium_replicate_00__spatialself_wavelet_edge_prior_predictions__lr_global_ranking.csv"
    full_slide_sparse_edge_downstream = full_root / "full_slide_sparse_downstream_target/spatialself_wavelet_edge_prior/human_lymph_node_visium_replicate_00__spatialself_wavelet_edge_prior_predictions__downstream_target_correlation.csv"
    full_slide_sparse_control_lr = full_root / "full_slide_sparse_control_summary/real_validation_control_lr_summary.csv"
    full_slide_sparse_control_method = full_root / "full_slide_sparse_control_summary/real_validation_control_method_summary.csv"
    rows = [
        AuditRow("exp07_lymph_node_global_validation", "region enrichment smoke output exists", _exists_status(region, existing_status="smoke_ready"), str(region.relative_to(PROJECT_ROOT))),
        AuditRow("exp07_lymph_node_global_validation", "global LR ranking smoke output exists", _exists_status(ranking, existing_status="smoke_ready"), str(ranking.relative_to(PROJECT_ROOT))),
        AuditRow("exp07_lymph_node_global_validation", "downstream target smoke output exists", _exists_status(downstream, existing_status="smoke_ready"), str(downstream.relative_to(PROJECT_ROOT))),
        AuditRow("exp07_lymph_node_global_validation", "edge-prior region enrichment smoke output exists", _exists_status(edge_region, existing_status="smoke_ready"), str(edge_region.relative_to(PROJECT_ROOT))),
        AuditRow("exp07_lymph_node_global_validation", "edge-prior global LR ranking smoke output exists", _exists_status(edge_ranking, existing_status="smoke_ready"), str(edge_ranking.relative_to(PROJECT_ROOT))),
        AuditRow("exp07_lymph_node_global_validation", "edge-prior downstream target smoke output exists", _exists_status(edge_downstream, existing_status="smoke_ready"), str(edge_downstream.relative_to(PROJECT_ROOT))),
        AuditRow("exp07_lymph_node_global_validation", "positive/negative control LR summary exists", _exists_status(control_lr, existing_status="smoke_ready"), str(control_lr.relative_to(PROJECT_ROOT))),
        AuditRow("exp07_lymph_node_global_validation", "positive/negative control method summary exists", _exists_status(control_method, existing_status="smoke_ready"), str(control_method.relative_to(PROJECT_ROOT))),
        AuditRow("exp07_lymph_node_global_validation", "full region enrichment output exists", _exists_status(full_region), str(full_region.relative_to(PROJECT_ROOT))),
        AuditRow("exp07_lymph_node_global_validation", "full global LR ranking output exists", _exists_status(full_ranking), str(full_ranking.relative_to(PROJECT_ROOT))),
        AuditRow("exp07_lymph_node_global_validation", "full downstream target output exists", _exists_status(full_downstream), str(full_downstream.relative_to(PROJECT_ROOT))),
        AuditRow("exp07_lymph_node_global_validation", "full edge-prior region enrichment output exists", _exists_status(full_edge_region), str(full_edge_region.relative_to(PROJECT_ROOT))),
        AuditRow("exp07_lymph_node_global_validation", "full edge-prior global LR ranking output exists", _exists_status(full_edge_ranking), str(full_edge_ranking.relative_to(PROJECT_ROOT))),
        AuditRow("exp07_lymph_node_global_validation", "full edge-prior downstream target output exists", _exists_status(full_edge_downstream), str(full_edge_downstream.relative_to(PROJECT_ROOT))),
        AuditRow("exp07_lymph_node_global_validation", "full positive/negative control LR summary exists", _exists_status(full_control_lr), str(full_control_lr.relative_to(PROJECT_ROOT))),
        AuditRow("exp07_lymph_node_global_validation", "full positive/negative control method summary exists", _exists_status(full_control_method), str(full_control_method.relative_to(PROJECT_ROOT))),
        AuditRow("exp07_lymph_node_global_validation", "full-slide sparse region enrichment output exists", _exists_status(full_slide_sparse_region), str(full_slide_sparse_region.relative_to(PROJECT_ROOT))),
        AuditRow("exp07_lymph_node_global_validation", "full-slide sparse global LR ranking output exists", _exists_status(full_slide_sparse_ranking), str(full_slide_sparse_ranking.relative_to(PROJECT_ROOT))),
        AuditRow("exp07_lymph_node_global_validation", "full-slide sparse downstream target output exists", _exists_status(full_slide_sparse_downstream), str(full_slide_sparse_downstream.relative_to(PROJECT_ROOT))),
        AuditRow("exp07_lymph_node_global_validation", "full-slide sparse edge-prior region enrichment output exists", _exists_status(full_slide_sparse_edge_region), str(full_slide_sparse_edge_region.relative_to(PROJECT_ROOT))),
        AuditRow("exp07_lymph_node_global_validation", "full-slide sparse edge-prior global LR ranking output exists", _exists_status(full_slide_sparse_edge_ranking), str(full_slide_sparse_edge_ranking.relative_to(PROJECT_ROOT))),
        AuditRow("exp07_lymph_node_global_validation", "full-slide sparse edge-prior downstream target output exists", _exists_status(full_slide_sparse_edge_downstream), str(full_slide_sparse_edge_downstream.relative_to(PROJECT_ROOT))),
        AuditRow("exp07_lymph_node_global_validation", "full-slide sparse positive/negative control LR summary exists", _exists_status(full_slide_sparse_control_lr), str(full_slide_sparse_control_lr.relative_to(PROJECT_ROOT))),
        AuditRow("exp07_lymph_node_global_validation", "full-slide sparse positive/negative control method summary exists", _exists_status(full_slide_sparse_control_method), str(full_slide_sparse_control_method.relative_to(PROJECT_ROOT))),
    ]
    if downstream.exists():
        tested = [row for row in _read_csv(downstream) if row.get("status") == "tested"]
        rows.append(
            AuditRow(
                "exp07_lymph_node_global_validation",
                "downstream target correlations tested",
                "smoke_ready" if tested else "partial",
                str(downstream.relative_to(PROJECT_ROOT)),
                f"n_tested={len(tested)}",
            )
        )
    rows.extend(
        _audit_real_validation_method_outputs(
            "default",
            ranking,
            region,
            downstream,
            expected_lr_pairs=5,
            existing_status="smoke_ready",
        )
    )
    rows.extend(
        _audit_real_validation_method_outputs(
            "edge-prior",
            edge_ranking,
            edge_region,
            edge_downstream,
            expected_lr_pairs=5,
            existing_status="smoke_ready",
        )
    )
    if downstream.exists() and edge_downstream.exists():
        default_rows = _read_csv(downstream)
        edge_rows = _read_csv(edge_downstream)
        default_tested = [
            float(row.get("spearman_r", "nan"))
            for row in default_rows
            if row.get("status") == "tested" and row.get("spearman_r")
        ]
        edge_tested = [
            float(row.get("spearman_r", "nan"))
            for row in edge_rows
            if row.get("status") == "tested" and row.get("spearman_r")
        ]
        default_mean = sum(default_tested) / len(default_tested) if default_tested else 0.0
        edge_mean = sum(edge_tested) / len(edge_tested) if edge_tested else 0.0
        rows.append(
            AuditRow(
                "exp07_lymph_node_global_validation",
                "edge-prior downstream target correlation improves over default smoke",
                "smoke_ready" if edge_mean >= default_mean else "partial",
                str(edge_downstream.relative_to(PROJECT_ROOT)),
                f"default_mean_spearman={default_mean:.3f}; edge_prior_mean_spearman={edge_mean:.3f}",
            )
        )
    if control_method.exists():
        records = _read_csv(control_method)
        methods = {row.get("method", "") for row in records}
        required_methods = {"spatialself_wavelet", "spatialself_wavelet_edge_prior"}
        region_values = []
        for row in records:
            for column in ("mean_positive_region_enrichment", "mean_negative_region_enrichment"):
                try:
                    region_values.append(float(row.get(column, "nan")))
                except ValueError:
                    region_values.append(float("nan"))
        all_region_values_finite = all(math.isfinite(value) for value in region_values)
        rows.append(
            AuditRow(
                "exp07_lymph_node_global_validation",
                "control summary includes default and edge-prior methods",
                "smoke_ready" if required_methods.issubset(methods) else "partial",
                str(control_method.relative_to(PROJECT_ROOT)),
                f"methods={sorted(methods)}",
            )
        )
        rows.append(
            AuditRow(
                "exp07_lymph_node_global_validation",
                "control region enrichment summaries are finite",
                "smoke_ready" if records and all_region_values_finite else "partial",
                str(control_method.relative_to(PROJECT_ROOT)),
                f"n_region_summary_values={len(region_values)}",
            )
        )
        rank_ok = True
        rank_details = []
        for row in records:
            method = row.get("method", "")
            mean_positive_rank = _to_float(row.get("mean_positive_rank", "nan"))
            mean_negative_rank = _to_float(row.get("mean_negative_rank", "nan"))
            best_negative_rank = _to_float(row.get("best_negative_rank", "nan"))
            negative_top3_count = _to_float(row.get("negative_top3_count", "nan"))
            method_ok = (
                math.isfinite(mean_positive_rank)
                and math.isfinite(mean_negative_rank)
                and math.isfinite(best_negative_rank)
                and math.isfinite(negative_top3_count)
                and mean_positive_rank < mean_negative_rank
                and negative_top3_count == 0
            )
            rank_ok = rank_ok and method_ok
            rank_details.append(
                f"{method}:pos={mean_positive_rank:.1f},neg={mean_negative_rank:.1f},neg_top3={negative_top3_count:.0f}"
            )
        rows.append(
            AuditRow(
                "exp07_lymph_node_global_validation",
                "positive controls rank ahead of high-expression negative controls",
                "smoke_ready" if records and rank_ok else "partial",
                str(control_method.relative_to(PROJECT_ROOT)),
                "; ".join(rank_details),
            )
        )
    if control_lr.exists():
        records = _read_csv(control_lr)
        control_types = {row.get("control_type", "") for row in records}
        required_controls = {"positive", "positive_weak", "negative_high_expression"}
        rows.append(
            AuditRow(
                "exp07_lymph_node_global_validation",
                "control LR summary includes positive, weak, and high-expression negative controls",
                "smoke_ready" if required_controls.issubset(control_types) else "partial",
                str(control_lr.relative_to(PROJECT_ROOT)),
                f"control_types={sorted(control_types)}",
            )
        )
    rows.extend(
        _audit_real_validation_method_outputs(
            "full default",
            full_ranking,
            full_region,
            full_downstream,
            expected_lr_pairs=5,
            existing_status="complete",
        )
    )
    rows.extend(
        _audit_real_validation_method_outputs(
            "full edge-prior",
            full_edge_ranking,
            full_edge_region,
            full_edge_downstream,
            expected_lr_pairs=5,
            existing_status="complete",
        )
    )
    if full_control_method.exists():
        rows.extend(_audit_full_real_validation_controls(full_control_method, full_control_lr))
    rows.extend(
        _audit_real_validation_method_outputs(
            "full-slide sparse default",
            full_slide_sparse_ranking,
            full_slide_sparse_region,
            full_slide_sparse_downstream,
            expected_lr_pairs=5,
            existing_status="complete",
        )
    )
    rows.extend(
        _audit_real_validation_method_outputs(
            "full-slide sparse edge-prior",
            full_slide_sparse_edge_ranking,
            full_slide_sparse_edge_region,
            full_slide_sparse_edge_downstream,
            expected_lr_pairs=5,
            existing_status="complete",
        )
    )
    if full_slide_sparse_control_method.exists():
        rows.extend(
            _audit_full_real_validation_controls(
                full_slide_sparse_control_method,
                full_slide_sparse_control_lr,
                label_prefix="full-slide sparse",
            )
        )
    return rows


def _audit_real_validation_method_outputs(
    label: str,
    ranking_path: Path,
    region_path: Path,
    downstream_path: Path,
    *,
    expected_lr_pairs: int,
    existing_status: str,
) -> list[AuditRow]:
    rows: list[AuditRow] = []
    ranking_rows = _read_csv(ranking_path) if ranking_path.exists() else []
    region_rows = _read_csv(region_path) if region_path.exists() else []
    downstream_rows = _read_csv(downstream_path) if downstream_path.exists() else []
    ranking_columns = {
        "lr_id",
        "ligand",
        "receptor",
        "mechanism",
        "method",
        "n_edges",
        "mean_score",
        "max_score",
        "top_region",
        "top_region_mean_score",
        "global_rank",
    }
    region_columns = {
        "lr_id",
        "ligand",
        "receptor",
        "mechanism",
        "region",
        "method",
        "n_edges_in_region",
        "region_mean_score",
        "lr_global_mean_score",
        "region_enrichment_score",
        "empirical_p_value",
        "bh_fdr",
        "n_permutations",
    }
    downstream_columns = {
        "lr_id",
        "ligand",
        "receptor",
        "mechanism",
        "method",
        "n_edges",
        "n_receiver_cells",
        "n_targets_present",
        "target_source",
        "spearman_r",
        "empirical_p_value",
        "status",
        "bh_fdr",
    }
    ranking_ok = (
        bool(ranking_rows)
        and ranking_columns.issubset(ranking_rows[0])
        and len({row.get("lr_id", "") for row in ranking_rows}) >= expected_lr_pairs
        and _real_validation_ranking_values_valid(ranking_rows)
    )
    region_ok = (
        bool(region_rows)
        and region_columns.issubset(region_rows[0])
        and len({row.get("lr_id", "") for row in region_rows}) >= expected_lr_pairs
        and len({row.get("region", "") for row in region_rows}) >= 2
        and _probability_columns_in_unit_interval(region_rows, ("empirical_p_value", "bh_fdr"))
    )
    downstream_ok = (
        bool(downstream_rows)
        and downstream_columns.issubset(downstream_rows[0])
        and len({row.get("lr_id", "") for row in downstream_rows}) >= expected_lr_pairs
        and _downstream_values_valid(downstream_rows)
    )
    rows.append(
        AuditRow(
            "exp07_lymph_node_global_validation",
            f"{label} all-LR global ranking schema and values are valid",
            existing_status if ranking_ok else "partial",
            str(ranking_path.relative_to(PROJECT_ROOT)),
            f"n_lr_pairs={len({row.get('lr_id', '') for row in ranking_rows})}",
        )
    )
    rows.append(
        AuditRow(
            "exp07_lymph_node_global_validation",
            f"{label} all-LR region enrichment schema and p/FDR values are valid",
            existing_status if region_ok else "partial",
            str(region_path.relative_to(PROJECT_ROOT)),
            f"n_lr_pairs={len({row.get('lr_id', '') for row in region_rows})}; n_regions={len({row.get('region', '') for row in region_rows})}",
        )
    )
    tested_count = sum(1 for row in downstream_rows if row.get("status") == "tested")
    rows.append(
        AuditRow(
            "exp07_lymph_node_global_validation",
            f"{label} downstream target correlation schema and tested LR pairs are valid",
            existing_status if downstream_ok and tested_count >= 3 else "partial",
            str(downstream_path.relative_to(PROJECT_ROOT)),
            f"n_lr_pairs={len({row.get('lr_id', '') for row in downstream_rows})}; n_tested={tested_count}",
        )
    )
    return rows


def _audit_full_real_validation_controls(
    method_summary_path: Path,
    lr_summary_path: Path,
    *,
    label_prefix: str = "full",
) -> list[AuditRow]:
    rows: list[AuditRow] = []
    method_rows = _read_csv(method_summary_path) if method_summary_path.exists() else []
    lr_rows = _read_csv(lr_summary_path) if lr_summary_path.exists() else []
    methods = {row.get("method", "") for row in method_rows}
    required_methods = {"spatialself_wavelet", "spatialself_wavelet_edge_prior"}
    rank_ok = True
    downstream_ok = True
    enrichment_values_finite = True
    detail = []
    downstream_by_method = {}
    for row in method_rows:
        method = row.get("method", "")
        mean_positive_rank = _to_float(row.get("mean_positive_rank", "nan"))
        mean_negative_rank = _to_float(row.get("mean_negative_rank", "nan"))
        negative_top3_count = _to_float(row.get("negative_top3_count", "nan"))
        downstream = _to_float(row.get("mean_positive_downstream_spearman", "nan"))
        positive_enrichment = _to_float(row.get("mean_positive_region_enrichment", "nan"))
        negative_enrichment = _to_float(row.get("mean_negative_region_enrichment", "nan"))
        rank_ok = rank_ok and (
            math.isfinite(mean_positive_rank)
            and math.isfinite(mean_negative_rank)
            and mean_positive_rank < mean_negative_rank
            and math.isfinite(negative_top3_count)
            and negative_top3_count == 0
        )
        downstream_ok = downstream_ok and math.isfinite(downstream) and downstream > 0
        enrichment_values_finite = enrichment_values_finite and (
            math.isfinite(positive_enrichment) and math.isfinite(negative_enrichment)
        )
        downstream_by_method[method] = downstream
        detail.append(
            f"{method}:pos_rank={mean_positive_rank:.1f},neg_rank={mean_negative_rank:.1f},"
            f"neg_top3={negative_top3_count:.0f},downstream={downstream:.3f}"
        )
    rows.append(
        AuditRow(
            "exp07_lymph_node_global_validation",
            f"{label_prefix} control summary includes default and edge-prior methods",
            "complete" if required_methods.issubset(methods) else "partial",
            str(method_summary_path.relative_to(PROJECT_ROOT)),
            f"methods={sorted(methods)}",
        )
    )
    rows.append(
        AuditRow(
            "exp07_lymph_node_global_validation",
            f"{label_prefix} positive controls rank ahead of high-expression negative controls",
            "complete" if method_rows and rank_ok else "partial",
            str(method_summary_path.relative_to(PROJECT_ROOT)),
            "; ".join(detail),
        )
    )
    edge_gain_ok = (
        math.isfinite(downstream_by_method.get("spatialself_wavelet", float("nan")))
        and math.isfinite(downstream_by_method.get("spatialself_wavelet_edge_prior", float("nan")))
        and downstream_by_method["spatialself_wavelet_edge_prior"] >= downstream_by_method["spatialself_wavelet"]
    )
    rows.append(
        AuditRow(
            "exp07_lymph_node_global_validation",
            f"{label_prefix} edge-prior improves mean positive downstream target correlation over default",
            "complete" if edge_gain_ok and downstream_ok else "partial",
            str(method_summary_path.relative_to(PROJECT_ROOT)),
            (
                f"default={downstream_by_method.get('spatialself_wavelet', float('nan')):.3f}; "
                f"edge_prior={downstream_by_method.get('spatialself_wavelet_edge_prior', float('nan')):.3f}"
            ),
        )
    )
    rows.append(
        AuditRow(
            "exp07_lymph_node_global_validation",
            f"{label_prefix} control region enrichment summaries are finite",
            "complete" if method_rows and enrichment_values_finite else "partial",
            str(method_summary_path.relative_to(PROJECT_ROOT)),
            f"n_methods={len(method_rows)}",
        )
    )
    if lr_rows:
        control_types = {row.get("control_type", "") for row in lr_rows}
        positive_tested = [
            row
            for row in lr_rows
            if row.get("control_type", "").startswith("positive") and row.get("downstream_status") == "tested"
        ]
        negative_controls = [row for row in lr_rows if row.get("control_type") == "negative_high_expression"]
        fdr_values = [
            _to_float(row.get("best_region_bh_fdr", "nan"))
            for row in lr_rows
            if row.get("best_region_bh_fdr", "")
        ]
        rows.append(
            AuditRow(
                "exp07_lymph_node_global_validation",
                f"{label_prefix} LR control summary includes positive, weak, and high-expression negative controls",
                "complete"
                if {"positive", "positive_weak", "negative_high_expression"}.issubset(control_types)
                and len(positive_tested) >= 6
                and len(negative_controls) >= 4
                and fdr_values
                and all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in fdr_values)
                else "partial",
                str(lr_summary_path.relative_to(PROJECT_ROOT)),
                (
                    f"control_types={sorted(control_types)}; n_positive_tested={len(positive_tested)}; "
                    f"n_negative_controls={len(negative_controls)}"
                ),
            )
        )
    return rows


def _audit_exp08() -> list[AuditRow]:
    scaling = PROJECT_ROOT / "outputs/phase1/exp08_runtime_memory/smoke_scaling_controlled/runtime_memory_scaling.csv"
    medium_scaling = PROJECT_ROOT / "outputs/phase1/exp08_runtime_memory/controlled_medium_scaling/runtime_memory_scaling.csv"
    sparse_scaling = PROJECT_ROOT / "outputs/phase1/exp08_runtime_memory/sparse_spatialself_scaling/runtime_memory_scaling.csv"
    sparse_scaling_manifest = PROJECT_ROOT / "outputs/phase1/exp08_runtime_memory/sparse_spatialself_scaling/runtime_scaling_manifest.json"
    sparse_50k_gat = PROJECT_ROOT / "outputs/phase1/exp08_runtime_memory/sparse_50k_spatialself_gat_sparse_knn_profile/runtime_memory_profile.csv"
    sparse_50k_wavelet = PROJECT_ROOT / "outputs/phase1/exp08_runtime_memory/sparse_50k_spatialself_wavelet_profile/runtime_memory_profile.csv"
    sparse_100k_scaling = PROJECT_ROOT / "outputs/phase1/exp08_runtime_memory/sparse_100k_spatialself_smoke/runtime_memory_scaling.csv"
    sparse_100k_manifest = PROJECT_ROOT / "outputs/phase1/exp08_runtime_memory/sparse_100k_spatialself_smoke/runtime_scaling_manifest.json"
    sparse_50k_highcap = PROJECT_ROOT / "outputs/phase1/exp08_runtime_memory/sparse_50k_highcap_spatialself/runtime_memory_scaling.csv"
    sparse_50k_highcap_manifest = PROJECT_ROOT / "outputs/phase1/exp08_runtime_memory/sparse_50k_highcap_spatialself/runtime_scaling_manifest.json"
    full_slide_sparse_profile = PROJECT_ROOT / "outputs/phase1/exp08_runtime_memory/full_slide_sparse_semisynthetic_retained_profile/runtime_memory_profile.csv"
    sparse_candidates = PROJECT_ROOT / "outputs/phase1/exp08_runtime_memory/sparse_candidate_smoke/dataset/replicate_00/candidate_interactions.csv"
    sparse_runtime = PROJECT_ROOT / "outputs/phase1/exp08_runtime_memory/sparse_candidate_smoke/runtime/runtime_memory_scaling.csv"
    sparse_manifest = PROJECT_ROOT / "outputs/phase1/exp08_runtime_memory/sparse_candidate_smoke/runtime/runtime_scaling_manifest.json"
    sparse_semisynthetic = PROJECT_ROOT / "outputs/phase1/exp08_runtime_memory/sparse_candidate_smoke/semisynthetic_dataset/human_lymph_node_visium_replicate_00/candidate_interactions.csv"
    rows = [
        AuditRow("exp08_runtime_memory", "runtime scaling smoke output exists", _exists_status(scaling, existing_status="smoke_ready"), str(scaling.relative_to(PROJECT_ROOT))),
        AuditRow("exp08_runtime_memory", "controlled medium runtime scaling output exists", _exists_status(medium_scaling), str(medium_scaling.relative_to(PROJECT_ROOT))),
        AuditRow("exp08_runtime_memory", "sparse SpatialSelf 1k/5k/10k runtime scaling output exists", _exists_status(sparse_scaling), str(sparse_scaling.relative_to(PROJECT_ROOT))),
        AuditRow("exp08_runtime_memory", "sparse 50k SpatialSelf-GAT KD-tree profile exists", _exists_status(sparse_50k_gat), str(sparse_50k_gat.relative_to(PROJECT_ROOT))),
        AuditRow("exp08_runtime_memory", "sparse 50k SpatialSelf wavelet profile exists", _exists_status(sparse_50k_wavelet), str(sparse_50k_wavelet.relative_to(PROJECT_ROOT))),
        AuditRow("exp08_runtime_memory", "sparse 50k high-cap SpatialSelf runtime scaling output exists", _exists_status(sparse_50k_highcap), str(sparse_50k_highcap.relative_to(PROJECT_ROOT))),
        AuditRow("exp08_runtime_memory", "sparse 100k SpatialSelf runtime scaling output exists", _exists_status(sparse_100k_scaling), str(sparse_100k_scaling.relative_to(PROJECT_ROOT))),
        AuditRow("exp08_runtime_memory", "full-slide sparse semi-synthetic retained-method runtime profile exists", _exists_status(full_slide_sparse_profile), str(full_slide_sparse_profile.relative_to(PROJECT_ROOT))),
        AuditRow("exp08_runtime_memory", "sparse candidate generation smoke candidate table exists", _exists_status(sparse_candidates, existing_status="smoke_ready"), str(sparse_candidates.relative_to(PROJECT_ROOT))),
        AuditRow("exp08_runtime_memory", "sparse candidate runtime scaling smoke output exists", _exists_status(sparse_runtime, existing_status="smoke_ready"), str(sparse_runtime.relative_to(PROJECT_ROOT))),
        AuditRow("exp08_runtime_memory", "sparse semi-synthetic candidate smoke table exists", _exists_status(sparse_semisynthetic, existing_status="smoke_ready"), str(sparse_semisynthetic.relative_to(PROJECT_ROOT))),
    ]
    if scaling.exists():
        records = _read_csv(scaling)
        scales = sorted({row.get("scale_n_cells") for row in records})
        methods = sorted({row.get("method") for row in records})
        completed = [row for row in records if row.get("status") == "completed"]
        rows.append(
            AuditRow(
                "exp08_runtime_memory",
                "runtime smoke has multiple scales and completed methods",
                "smoke_ready" if len(scales) >= 2 and len(completed) >= 2 else "partial",
                str(scaling.relative_to(PROJECT_ROOT)),
                f"scales={scales}; methods={methods}; n_completed={len(completed)}",
            )
        )
        rows.append(
            AuditRow(
                "exp08_runtime_memory",
                "runtime smoke includes edge-prior SpatialSelf candidate",
                "smoke_ready"
                if any(
                    row.get("method") == "spatialself_wavelet_edge_prior"
                    and row.get("status") == "completed"
                    for row in records
                )
                else "partial",
                str(scaling.relative_to(PROJECT_ROOT)),
                f"methods={methods}",
            )
        )
        rows.append(
            AuditRow(
                "exp08_runtime_memory",
                "runtime smoke schema includes wall-clock, memory, candidate edges, LR pairs, logs, and commands",
                "smoke_ready" if _runtime_schema_valid(records) else "partial",
                str(scaling.relative_to(PROJECT_ROOT)),
                f"n_rows={len(records)}",
            )
        )
        rows.append(
            AuditRow(
                "exp08_runtime_memory",
                "runtime smoke resource values are non-degenerate for completed methods",
                "smoke_ready" if _runtime_resource_values_valid(completed) else "partial",
                str(scaling.relative_to(PROJECT_ROOT)),
                f"n_completed={len(completed)}",
            )
        )
        rows.append(
            AuditRow(
                "exp08_runtime_memory",
                "runtime smoke candidate edges increase with scale for each method",
                "smoke_ready" if _runtime_edges_increase_by_scale(completed) else "partial",
                str(scaling.relative_to(PROJECT_ROOT)),
                f"scales={scales}; methods={methods}",
            )
        )
    if medium_scaling.exists():
        rows.extend(_audit_medium_runtime_scaling(medium_scaling))
    if sparse_scaling.exists():
        rows.extend(_audit_sparse_spatialself_scaling(sparse_scaling, sparse_scaling_manifest))
    if sparse_50k_gat.exists():
        rows.extend(_audit_sparse_50k_spatialself_profiles(sparse_50k_gat, sparse_50k_wavelet))
    if sparse_50k_highcap.exists():
        rows.extend(_audit_sparse_50k_highcap_scaling(sparse_50k_highcap, sparse_50k_highcap_manifest))
    if sparse_100k_scaling.exists():
        rows.extend(_audit_sparse_100k_spatialself_scaling(sparse_100k_scaling, sparse_100k_manifest))
    if full_slide_sparse_profile.exists():
        rows.extend(_audit_full_slide_sparse_runtime_profile(full_slide_sparse_profile))
    if sparse_candidates.exists():
        rows.extend(_audit_sparse_candidate_smoke(sparse_candidates))
    if sparse_runtime.exists():
        rows.extend(_audit_sparse_runtime_smoke(sparse_runtime, sparse_manifest))
    if sparse_semisynthetic.exists():
        rows.extend(_audit_sparse_semisynthetic_candidate_smoke(sparse_semisynthetic))
    return rows


def _audit_sparse_50k_spatialself_profiles(gat_path: Path, wavelet_path: Path) -> list[AuditRow]:
    records = _read_csv(gat_path)
    if wavelet_path.exists():
        records.extend(_read_csv(wavelet_path))
    completed = [row for row in records if row.get("status") == "completed"]
    completed_methods = {row.get("method", "") for row in completed}
    required_methods = {
        "spatialself_wavelet",
        "spatialself_wavelet_edge_prior",
        "spatialself_gat",
    }
    evidence = f"{gat_path.relative_to(PROJECT_ROOT)},{wavelet_path.relative_to(PROJECT_ROOT)}"
    rows = [
        AuditRow(
            "exp08_runtime_memory",
            "sparse 50k SpatialSelf profile schemas include wall-clock, memory, candidate edges, LR pairs, logs, and commands",
            "complete" if _runtime_profile_schema_valid(records) else "partial",
            evidence,
            f"n_rows={len(records)}",
        ),
        AuditRow(
            "exp08_runtime_memory",
            "sparse 50k SpatialSelf variants completed with non-degenerate resources",
            "complete"
            if required_methods.issubset(completed_methods)
            and _runtime_profile_resource_values_valid(completed)
            else "partial",
            evidence,
            f"completed_methods={sorted(completed_methods)}",
        ),
    ]
    if completed:
        n_cells = min(_to_float(row.get("n_cells_or_spots", "nan")) for row in completed)
        n_edges = min(_to_float(row.get("n_candidate_edges", "nan")) for row in completed)
        max_wall_clock = max(_to_float(row.get("wall_clock_seconds", "nan")) for row in completed)
        max_memory = max(_to_float(row.get("peak_cpu_memory_mb", "nan")) for row in completed)
        per_method = {
            row.get("method", ""): (
                _to_float(row.get("wall_clock_seconds", "nan")),
                _to_float(row.get("peak_cpu_memory_mb", "nan")),
            )
            for row in completed
        }
        rows.append(
            AuditRow(
                "exp08_runtime_memory",
                "sparse 50k SpatialSelf profiles demonstrate KD-tree KNN memory fix",
                "complete"
                if math.isfinite(n_cells)
                and n_cells >= 50000
                and math.isfinite(n_edges)
                and n_edges > 0
                and required_methods.issubset(completed_methods)
                and math.isfinite(max_wall_clock)
                and max_wall_clock < 60.0
                and math.isfinite(max_memory)
                and max_memory < 8000.0
                else "partial",
                evidence,
                f"n_cells={n_cells:.0f}; n_edges={n_edges:.0f}; per_method={per_method}",
            )
        )
    return rows


def _audit_full_slide_sparse_runtime_profile(path: Path) -> list[AuditRow]:
    records = _read_csv(path)
    completed = [row for row in records if row.get("status") == "completed"]
    completed_methods = {row.get("method", "") for row in completed}
    statuses = {row.get("method", ""): row.get("status", "") for row in records}
    required_methods = {
        "spatialself_wavelet",
        "spatialself_wavelet_edge_prior",
        "spatialself_gat",
        "commot",
        "cellchat",
        "spatalk",
        "cellnest",
        "deeptalk",
    }
    rows = [
        AuditRow(
            "exp08_runtime_memory",
            "full-slide sparse runtime profile covers retained runnable methods",
            "complete" if required_methods.issubset(completed_methods) else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            f"completed_methods={sorted(completed_methods)}",
        ),
        AuditRow(
            "exp08_runtime_memory",
            "full-slide sparse runtime profile schema includes wall-clock, memory, candidate edges, LR pairs, logs, and commands",
            "complete" if _runtime_profile_schema_valid(records) else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            f"n_rows={len(records)}",
        ),
        AuditRow(
            "exp08_runtime_memory",
            "full-slide sparse runtime completed methods have non-degenerate resource values",
            "complete" if _runtime_profile_resource_values_valid(completed) else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            f"n_completed={len(completed)}",
        ),
        AuditRow(
            "exp08_runtime_memory",
            "full-slide sparse runtime records NicheNet and NICHES non-runnable status separately",
            "complete"
            if statuses.get("nichenet") == "pending_adapter"
            and statuses.get("niches") == "optional_blocked"
            else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            f"statuses={statuses}",
        ),
    ]
    if records:
        max_cells = max(_to_float(row.get("n_cells_or_spots", "nan")) for row in records)
        max_edges = max(_to_float(row.get("n_candidate_edges", "nan")) for row in records)
        n_lr_pairs = max(_to_float(row.get("n_lr_pairs", "nan")) for row in records)
        per_method = {
            row.get("method", ""): (
                _to_float(row.get("wall_clock_seconds", "nan")),
                _to_float(row.get("peak_cpu_memory_mb", "nan")),
            )
            for row in completed
        }
        rows.append(
            AuditRow(
                "exp08_runtime_memory",
                "full-slide sparse runtime profiles realistic-background benchmark scale",
                "complete"
                if math.isfinite(max_cells)
                and max_cells >= 4000
                and math.isfinite(max_edges)
                and max_edges >= 300000
                and math.isfinite(n_lr_pairs)
                and n_lr_pairs >= 5
                else "partial",
                str(path.relative_to(PROJECT_ROOT)),
                (
                    f"n_cells_or_spots={max_cells:.0f}; n_candidate_edges={max_edges:.0f}; "
                    f"n_lr_pairs={n_lr_pairs:.0f}; per_method_seconds_memory={per_method}"
                ),
            )
        )
    return rows


def _audit_sparse_100k_spatialself_scaling(path: Path, manifest_path: Path) -> list[AuditRow]:
    records = _read_csv(path)
    completed = [row for row in records if row.get("status") == "completed"]
    completed_methods = {row.get("method", "") for row in completed}
    required_methods = {
        "spatialself_wavelet",
        "spatialself_wavelet_edge_prior",
        "spatialself_gat",
    }
    manifest_ok = False
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest_ok = (
            len(manifest) >= 1
            and all(row.get("candidate_generation") == "sparse_sampled" for row in manifest)
            and any(float(row.get("scale_n_cells", 0)) >= 100000 for row in manifest)
        )
    rows = [
        AuditRow(
            "exp08_runtime_memory",
            "sparse 100k SpatialSelf scaling covers all replaceable SpatialSelf variants",
            "complete" if required_methods.issubset(completed_methods) else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            f"completed_methods={sorted(completed_methods)}",
        ),
        AuditRow(
            "exp08_runtime_memory",
            "sparse 100k SpatialSelf scaling schema includes wall-clock, memory, candidate edges, LR pairs, logs, and commands",
            "complete" if _runtime_schema_valid(records) else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            f"n_rows={len(records)}",
        ),
        AuditRow(
            "exp08_runtime_memory",
            "sparse 100k SpatialSelf scaling all profiled methods completed with non-degenerate resources",
            "complete"
            if required_methods.issubset(completed_methods)
            and len(completed) == len(records)
            and _runtime_medium_resource_values_valid(completed)
            else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            f"n_completed={len(completed)}; n_rows={len(records)}",
        ),
        AuditRow(
            "exp08_runtime_memory",
            "sparse 100k SpatialSelf scaling manifest records sparse candidate generation",
            "complete" if manifest_ok else "partial",
            str(manifest_path.relative_to(PROJECT_ROOT)),
        ),
    ]
    if completed:
        min_cells = min(_to_float(row.get("n_cells_or_spots", "nan")) for row in completed)
        min_edges = min(_to_float(row.get("n_candidate_edges", "nan")) for row in completed)
        max_wall_clock = max(_to_float(row.get("wall_clock_seconds", "nan")) for row in completed)
        max_memory = max(_to_float(row.get("peak_cpu_memory_mb", "nan")) for row in completed)
        rows.append(
            AuditRow(
                "exp08_runtime_memory",
                "sparse 100k SpatialSelf scaling reaches 100k cells with capped candidate edges",
                "complete"
                if math.isfinite(min_cells)
                and min_cells >= 100000
                and math.isfinite(min_edges)
                and min_edges > 0
                and math.isfinite(max_wall_clock)
                and max_wall_clock < 120.0
                and math.isfinite(max_memory)
                and max_memory < 12000.0
                else "partial",
                str(path.relative_to(PROJECT_ROOT)),
                f"min_cells={min_cells:.0f}; min_edges={min_edges:.0f}; max_wall_clock={max_wall_clock:.2f}; max_cpu_mb={max_memory:.1f}",
            )
        )
    return rows


def _audit_sparse_50k_highcap_scaling(path: Path, manifest_path: Path) -> list[AuditRow]:
    records = _read_csv(path)
    completed = [row for row in records if row.get("status") == "completed"]
    completed_methods = {row.get("method", "") for row in completed}
    required_methods = {
        "spatialself_wavelet",
        "spatialself_wavelet_edge_prior",
        "spatialself_gat",
    }
    manifest_ok = False
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest_ok = (
            len(manifest) >= 1
            and all(row.get("candidate_generation") == "sparse_sampled" for row in manifest)
            and all(int(row.get("max_candidates_per_lr", 0)) >= 50000 for row in manifest)
            and all(int(row.get("max_candidates_per_sender_lr", 0)) >= 48 for row in manifest)
        )
    rows = [
        AuditRow(
            "exp08_runtime_memory",
            "sparse 50k high-cap scaling covers all replaceable SpatialSelf variants",
            "complete" if required_methods.issubset(completed_methods) else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            f"completed_methods={sorted(completed_methods)}",
        ),
        AuditRow(
            "exp08_runtime_memory",
            "sparse 50k high-cap scaling schema includes wall-clock, memory, candidate edges, LR pairs, logs, and commands",
            "complete" if _runtime_schema_valid(records) else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            f"n_rows={len(records)}",
        ),
        AuditRow(
            "exp08_runtime_memory",
            "sparse 50k high-cap scaling all profiled methods completed with non-degenerate resources",
            "complete"
            if required_methods.issubset(completed_methods)
            and len(completed) == len(records)
            and _runtime_medium_resource_values_valid(completed)
            else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            f"n_completed={len(completed)}; n_rows={len(records)}",
        ),
        AuditRow(
            "exp08_runtime_memory",
            "sparse 50k high-cap manifest records larger candidate caps",
            "complete" if manifest_ok else "partial",
            str(manifest_path.relative_to(PROJECT_ROOT)),
        ),
    ]
    if completed:
        min_cells = min(_to_float(row.get("n_cells_or_spots", "nan")) for row in completed)
        min_edges = min(_to_float(row.get("n_candidate_edges", "nan")) for row in completed)
        max_wall_clock = max(_to_float(row.get("wall_clock_seconds", "nan")) for row in completed)
        max_memory = max(_to_float(row.get("peak_cpu_memory_mb", "nan")) for row in completed)
        rows.append(
            AuditRow(
                "exp08_runtime_memory",
                "sparse 50k high-cap scaling reaches more than 300k candidate edges",
                "complete"
                if math.isfinite(min_cells)
                and min_cells >= 50000
                and math.isfinite(min_edges)
                and min_edges >= 300000
                and math.isfinite(max_wall_clock)
                and max_wall_clock < 120.0
                and math.isfinite(max_memory)
                and max_memory < 12000.0
                else "partial",
                str(path.relative_to(PROJECT_ROOT)),
                f"min_cells={min_cells:.0f}; min_edges={min_edges:.0f}; max_wall_clock={max_wall_clock:.2f}; max_cpu_mb={max_memory:.1f}",
            )
        )
    return rows


def _audit_sparse_spatialself_scaling(path: Path, manifest_path: Path) -> list[AuditRow]:
    records = _read_csv(path)
    completed = [row for row in records if row.get("status") == "completed"]
    scales = sorted({_to_float(row.get("scale_n_cells", "nan")) for row in records})
    methods = {row.get("method", "") for row in records}
    required_scales = {1000.0, 5000.0, 10000.0}
    required_methods = {"spatialself_wavelet", "spatialself_wavelet_edge_prior", "spatialself_gat"}
    manifest_ok = False
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest_ok = (
            len(manifest) >= 3
            and all(row.get("candidate_generation") == "sparse_sampled" for row in manifest)
            and {float(row.get("scale_n_cells", 0)) for row in manifest}.issuperset(required_scales)
        )
    rows = [
        AuditRow(
            "exp08_runtime_memory",
            "sparse SpatialSelf scaling covers 1k/5k/10k and all replaceable SpatialSelf variants",
            "complete"
            if required_scales.issubset(set(scales)) and required_methods.issubset(methods)
            else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            f"scales={[int(scale) for scale in scales if math.isfinite(scale)]}; methods={sorted(methods)}",
        ),
        AuditRow(
            "exp08_runtime_memory",
            "sparse SpatialSelf scaling schema includes wall-clock, memory, candidate edges, LR pairs, logs, and commands",
            "complete" if _runtime_schema_valid(records) else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            f"n_rows={len(records)}",
        ),
        AuditRow(
            "exp08_runtime_memory",
            "sparse SpatialSelf scaling all profiled methods completed with non-degenerate resource values",
            "complete" if len(completed) == len(records) and _runtime_medium_resource_values_valid(completed) else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            f"n_completed={len(completed)}; n_rows={len(records)}",
        ),
        AuditRow(
            "exp08_runtime_memory",
            "sparse SpatialSelf scaling manifest records sparse candidate generation",
            "complete" if manifest_ok else "partial",
            str(manifest_path.relative_to(PROJECT_ROOT)),
        ),
    ]
    if completed:
        max_scale = max(_to_float(row.get("scale_n_cells", "nan")) for row in completed)
        max_rows = [row for row in completed if _to_float(row.get("scale_n_cells", "nan")) == max_scale]
        max_edges = max(_to_float(row.get("n_candidate_edges", "nan")) for row in max_rows)
        max_memory = max(_to_float(row.get("peak_cpu_memory_mb", "nan")) for row in max_rows)
        max_time = max(_to_float(row.get("wall_clock_seconds", "nan")) for row in max_rows)
        rows.append(
            AuditRow(
                "exp08_runtime_memory",
                "sparse SpatialSelf scaling reaches 10k cells with bounded candidate edges",
                "complete"
                if math.isfinite(max_scale)
                and max_scale >= 10000
                and math.isfinite(max_edges)
                and max_edges > 0
                and math.isfinite(max_memory)
                and max_memory > 0
                and math.isfinite(max_time)
                and max_time > 0
                else "partial",
                str(path.relative_to(PROJECT_ROOT)),
                f"max_scale={max_scale:.0f}; max_edges={max_edges:.0f}; max_wall_clock={max_time:.2f}; max_cpu_mb={max_memory:.1f}",
            )
        )
    return rows


def _audit_sparse_candidate_smoke(
    path: Path,
    *,
    exp_id: str = "exp08_runtime_memory",
) -> list[AuditRow]:
    records = _read_csv(path)
    mechanisms = {row.get("mechanism", "") for row in records}
    required_mechanisms = {
        "juxtacrine_contact",
        "short_medium_paracrine",
        "long_range_chemokine_niche",
        "false_positive_high_expression",
    }
    positives = [row for row in records if int(_to_float(row.get("ground_truth", "0"))) == 1]
    weak_positives = [
        row
        for row in positives
        if _as_bool(row.get("weak_signal", "false"))
    ]
    false_positive_labels = [
        row
        for row in positives
        if _as_bool(row.get("false_positive_high_expression", "false"))
    ]
    per_lr_counts: dict[str, int] = {}
    for row in records:
        lr_id = row.get("lr_id", "")
        per_lr_counts[lr_id] = per_lr_counts.get(lr_id, 0) + 1
    contact_positive_distances = [
        _to_float(row.get("distance", "nan"))
        for row in positives
        if row.get("mechanism") == "juxtacrine_contact"
    ]
    long_positive_distances = [
        _to_float(row.get("distance", "nan"))
        for row in positives
        if row.get("mechanism") == "long_range_chemokine_niche"
    ]
    contact_max = max(contact_positive_distances) if contact_positive_distances else float("nan")
    long_median = _median(long_positive_distances)
    rows = [
        AuditRow(
            exp_id,
            "sparse candidate smoke preserves all controlled mechanisms",
            "smoke_ready" if required_mechanisms.issubset(mechanisms) else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            f"mechanisms={sorted(mechanisms)}; n_rows={len(records)}",
        ),
        AuditRow(
            exp_id,
            "sparse candidate smoke has positives, weak positives, and zero high-expression false-positive labels",
            "smoke_ready"
            if records and positives and weak_positives and not false_positive_labels
            else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            (
                f"n_rows={len(records)}; positives={len(positives)}; "
                f"weak_positives={len(weak_positives)}; false_positive_labels={len(false_positive_labels)}; "
                f"per_lr_max={max(per_lr_counts.values()) if per_lr_counts else 0}"
            ),
        ),
        AuditRow(
            exp_id,
            "sparse candidate smoke keeps contact and long-range distance regimes distinct",
            "smoke_ready"
            if math.isfinite(contact_max)
            and math.isfinite(long_median)
            and contact_max <= 0.06
            and long_median > contact_max
            else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            f"contact_positive_max={contact_max:.5f}; long_range_positive_median={long_median:.5f}",
        ),
    ]
    return rows


def _audit_sparse_runtime_smoke(path: Path, manifest_path: Path) -> list[AuditRow]:
    records = _read_csv(path)
    completed = [
        row
        for row in records
        if row.get("method") == "spatialself_wavelet" and row.get("status") == "completed"
    ]
    manifest_ok = False
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest_ok = bool(manifest) and all(
            row.get("candidate_generation") == "sparse_sampled" for row in manifest
        )
    resource_ok = _runtime_resource_values_valid(completed)
    rows = [
        AuditRow(
            "exp08_runtime_memory",
            "sparse runtime smoke manifest records sparse candidate generation",
            "smoke_ready" if manifest_ok else "partial",
            str(manifest_path.relative_to(PROJECT_ROOT)),
        ),
        AuditRow(
            "exp08_runtime_memory",
            "sparse runtime smoke profiles SpatialSelf wavelet with non-degenerate resources",
            "smoke_ready" if len(completed) == 1 and resource_ok else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            f"n_rows={len(records)}; n_completed_wavelet={len(completed)}",
        ),
    ]
    return rows


def _audit_sparse_semisynthetic_candidate_smoke(path: Path) -> list[AuditRow]:
    records = _read_csv(path)
    mechanisms = {row.get("mechanism", "") for row in records}
    required_mechanisms = {
        "short_medium_paracrine",
        "long_range_chemokine_niche",
        "false_positive_high_expression",
    }
    positives = [row for row in records if int(_to_float(row.get("ground_truth", "0"))) == 1]
    weak_positives = [
        row
        for row in positives
        if _as_bool(row.get("weak_signal", "false"))
    ]
    false_positive_labels = [
        row
        for row in positives
        if _as_bool(row.get("false_positive_high_expression", "false"))
    ]
    distance_values = [_to_float(row.get("distance", "nan")) for row in records]
    max_distance = max(distance_values) if distance_values else float("nan")
    rows = [
        AuditRow(
            "exp02_semisynthetic_real_background",
            "sparse semi-synthetic smoke preserves injected paracrine, long-range, and false-positive mechanisms",
            "smoke_ready" if required_mechanisms.issubset(mechanisms) else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            f"mechanisms={sorted(mechanisms)}; n_rows={len(records)}",
        ),
        AuditRow(
            "exp02_semisynthetic_real_background",
            "sparse semi-synthetic smoke has positives, weak positives, and zero high-expression false-positive labels",
            "smoke_ready"
            if records and positives and weak_positives and not false_positive_labels
            else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            (
                f"n_rows={len(records)}; positives={len(positives)}; "
                f"weak_positives={len(weak_positives)}; false_positive_labels={len(false_positive_labels)}; "
                f"max_distance={max_distance:.5f}"
            ),
        ),
    ]
    return rows


def _scan_file_for_residue(path: Path, hits: list[str]) -> None:
    if path.suffix in {".pyc", ".png", ".jpg", ".jpeg", ".gif", ".h5ad"}:
        return
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return
    if SOURCE_RESIDUE_PATTERN.search(text):
        hits.append(str(path.relative_to(PROJECT_ROOT)))


def _replicate_dirs(dataset_root: Path) -> list[Path]:
    if not dataset_root.exists():
        return []
    dirs = sorted(path for path in dataset_root.glob("replicate_*") if path.is_dir())
    dirs += sorted(path for path in dataset_root.glob("*_replicate_*") if path.is_dir())
    return dirs


def _exists_status(path: Path, *, existing_status: str = "complete") -> str:
    return existing_status if path.exists() else "missing"


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _median(values: list[float]) -> float:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return float("nan")
    mid = len(clean) // 2
    if len(clean) % 2:
        return float(clean[mid])
    return float((clean[mid - 1] + clean[mid]) / 2.0)


def _project_path(value: str) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _probability_columns_in_unit_interval(rows: list[dict[str, str]], columns: tuple[str, ...]) -> bool:
    if not rows:
        return False
    for row in rows:
        for column in columns:
            value = _to_float(row.get(column, "nan"))
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                return False
    return True


def _ablation_metric_values_valid(rows: list[dict[str, str]], columns: set[str]) -> bool:
    if not rows:
        return False
    probability_columns = {
        "auroc_mean",
        "aupr_mean",
        "weak_signal_recall_at_threshold_mean",
        "empirical_fdr_at_threshold_mean",
        "false_positive_high_expression_rate_at_threshold_mean",
        "calibration_error_10bin_mean",
    }
    count_columns = {
        "n_pseudo_positive_mean",
        "n_pseudo_negative_mean",
        "n_ranking_pairs_mean",
    }
    for row in rows:
        for column in columns:
            value = _to_float(row.get(column, "nan"))
            if not math.isfinite(value):
                return False
            if column in probability_columns and not (0.0 <= value <= 1.0):
                return False
            if column in count_columns and value < 0.0:
                return False
    return True


def _row_by_key(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    return {}


def _baseline_status_details_valid(rows: list[dict[str, str]]) -> bool:
    if not rows:
        return False
    allowed_status = {
        "completed",
        "completed_or_model_required",
        "downstream_validation_only",
        "pending_adapter",
        "blocked",
        "optional_blocked",
    }
    for row in rows:
        method = str(row.get("method", ""))
        status = str(row.get("status", ""))
        detail = str(row.get("detail", "")).strip()
        if not method or status not in allowed_status or not detail:
            return False
        detail_lower = detail.lower()
        if method == "deeptalk" and "model" not in detail_lower:
            return False
        if method == "nichenet" and "downstream" not in detail_lower:
            return False
        if method == "niches" and status != "optional_blocked":
            return False
        if method == "niches" and not any(token in detail_lower for token in ("blocked", "failed", "install")):
            return False
    return True


def _baseline_summary_metrics_valid(rows: list[dict[str, str]], *, min_count: int) -> bool:
    if not rows:
        return False
    required_columns = {
        "method",
        "auroc_mean",
        "auroc_count",
        "aupr_mean",
        "aupr_count",
        "top_k_precision_mean",
        "weak_signal_recall_at_threshold_mean",
        "empirical_fdr_at_threshold_mean",
        "false_positive_high_expression_rate_at_threshold_mean",
        "calibration_error_10bin_mean",
    }
    probability_columns = {
        "auroc_mean",
        "aupr_mean",
        "top_k_precision_mean",
        "weak_signal_recall_at_threshold_mean",
        "empirical_fdr_at_threshold_mean",
        "false_positive_high_expression_rate_at_threshold_mean",
        "calibration_error_10bin_mean",
    }
    for row in rows:
        if not required_columns.issubset(row):
            return False
        method = row.get("method", "")
        auroc_count = _to_float(row.get("auroc_count", "nan"))
        if method == "deeptalk" and auroc_count == 0:
            continue
        if not math.isfinite(auroc_count) or auroc_count < min_count:
            return False
        for column in probability_columns:
            value = _to_float(row.get(column, "nan"))
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                return False
    return True


def _audit_unified_full_baseline(
    summary_path: Path,
    status_path: Path,
    manifest_path: Path,
) -> list[AuditRow]:
    rows: list[AuditRow] = []
    summary_rows = _read_csv(summary_path)
    status_rows = _read_csv(status_path)
    methods = {row.get("method", "") for row in summary_rows}
    status_methods = {row.get("method", "") for row in status_rows}
    required_metric_methods = {
        "spatialself_wavelet",
        "spatialself_wavelet_edge_prior",
        "spatialself_gat",
        "commot",
        "cellchat",
        "spatalk",
        "cellnest",
    }
    required_status_methods = required_metric_methods | {"deeptalk", "nichenet", "niches"}
    required_columns = {"source_run", "baseline_evidence_level", "interpretation_note", "method"}
    rows.append(
        AuditRow(
            "exp03_baseline_comparison",
            "unified full baseline summary includes core, deep, and adapter-light methods",
            "complete" if required_metric_methods.issubset(methods) else "partial",
            str(summary_path.relative_to(PROJECT_ROOT)),
            f"missing={sorted(required_metric_methods - methods)}; methods={sorted(methods)}",
        )
    )
    rows.append(
        AuditRow(
            "exp03_baseline_comparison",
            "unified full baseline summary records source run, evidence level, and interpretation notes",
            "complete" if summary_rows and required_columns.issubset(summary_rows[0]) else "partial",
            str(summary_path.relative_to(PROJECT_ROOT)),
            f"missing={sorted(required_columns - set(summary_rows[0] if summary_rows else {}))}",
        )
    )
    metric_rows = [row for row in summary_rows if row.get("method") != "deeptalk"]
    rows.append(
        AuditRow(
            "exp03_baseline_comparison",
            "unified full baseline metrics have valid benchmark columns for runnable metric methods",
            "complete" if _baseline_summary_metrics_valid(metric_rows, min_count=5) else "partial",
            str(summary_path.relative_to(PROJECT_ROOT)),
            f"n_metric_rows={len(metric_rows)}",
        )
    )
    rows.append(
        AuditRow(
            "exp03_baseline_comparison",
            "unified full baseline status includes completed, model-required, downstream-only, and optional-blocked methods",
            "complete"
            if required_status_methods.issubset(status_methods) and _baseline_status_details_valid(status_rows)
            else "partial",
            str(status_path.relative_to(PROJECT_ROOT)),
            f"missing={sorted(required_status_methods - status_methods)}; n_status_rows={len(status_rows)}",
        )
    )
    by_method = {row.get("method", ""): row for row in summary_rows}
    default = by_method.get("spatialself_wavelet", {})
    edge = by_method.get("spatialself_wavelet_edge_prior", {})
    commot = by_method.get("commot", {})
    cellchat = by_method.get("cellchat", {})
    edge_aupr = _to_float(edge.get("aupr_mean", "nan"))
    default_aupr = _to_float(default.get("aupr_mean", "nan"))
    edge_weak = _to_float(edge.get("weak_signal_recall_at_threshold_mean", "nan"))
    default_weak = _to_float(default.get("weak_signal_recall_at_threshold_mean", "nan"))
    commot_weak = _to_float(commot.get("weak_signal_recall_at_threshold_mean", "nan"))
    cellchat_evidence = cellchat.get("baseline_evidence_level", "")
    tradeoff_ok = (
        math.isfinite(edge_aupr)
        and math.isfinite(default_aupr)
        and math.isfinite(edge_weak)
        and math.isfinite(default_weak)
        and math.isfinite(commot_weak)
        and edge_aupr >= default_aupr
        and edge_weak >= default_weak
        and commot_weak >= edge_weak
        and cellchat_evidence == "full_controlled_adapter_light"
    )
    rows.append(
        AuditRow(
            "exp03_baseline_comparison",
            "unified full baseline captures edge-prior gain, COMMOT weak-recall advantage, and adapter-light caveat",
            "complete" if tradeoff_ok else "partial",
            str(summary_path.relative_to(PROJECT_ROOT)),
            (
                f"default_aupr={default_aupr:.3f}; edge_aupr={edge_aupr:.3f}; "
                f"default_weak={default_weak:.3f}; edge_weak={edge_weak:.3f}; "
                f"commot_weak={commot_weak:.3f}; cellchat_level={cellchat_evidence}"
            ),
        )
    )
    manifest_ok = False
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
        manifest_ok = (
            int(manifest.get("n_methods_with_metrics", 0)) >= 8
            and int(manifest.get("n_status_rows", 0)) >= 10
            and "adapter-light" in str(manifest.get("note", ""))
        )
    rows.append(
        AuditRow(
            "exp03_baseline_comparison",
            "unified full baseline manifest documents source runs and adapter-light caveat",
            "complete" if manifest_ok else "partial",
            str(manifest_path.relative_to(PROJECT_ROOT)),
        )
    )
    return rows


def _audit_full_slide_sparse_baseline(summary_path: Path, status_path: Path) -> list[AuditRow]:
    rows: list[AuditRow] = []
    summary_rows = _read_csv(summary_path)
    status_rows = _read_csv(status_path)
    methods = {row.get("method", "") for row in summary_rows}
    status_methods = {row.get("method", "") for row in status_rows}
    required_metric_methods = {
        "spatialself_wavelet",
        "spatialself_wavelet_edge_prior",
        "spatialself_gat",
        "commot",
        "cellchat",
        "spatalk",
        "cellnest",
    }
    required_status_methods = required_metric_methods | {"deeptalk", "nichenet", "niches"}
    rows.append(
        AuditRow(
            "exp03_baseline_comparison",
            "full-slide sparse baseline includes SpatialSelf, traditional, adapter-light, and deep baseline rows",
            "complete" if required_metric_methods.issubset(methods) else "partial",
            str(summary_path.relative_to(PROJECT_ROOT)),
            f"missing={sorted(required_metric_methods - methods)}; methods={sorted(methods)}",
        )
    )
    runnable_rows = [row for row in summary_rows if row.get("method") != "deeptalk"]
    rows.append(
        AuditRow(
            "exp03_baseline_comparison",
            "full-slide sparse baseline metrics have valid benchmark columns for runnable methods",
            "complete" if _baseline_summary_metrics_valid(runnable_rows, min_count=1) else "partial",
            str(summary_path.relative_to(PROJECT_ROOT)),
            f"n_metric_rows={len(runnable_rows)}",
        )
    )
    rows.append(
        AuditRow(
            "exp03_baseline_comparison",
            "full-slide sparse baseline status records model-required, downstream-only, and optional-blocked details",
            "complete"
            if required_status_methods.issubset(status_methods) and _baseline_status_details_valid(status_rows)
            else "partial",
            str(status_path.relative_to(PROJECT_ROOT)),
            f"missing={sorted(required_status_methods - status_methods)}; n_status_rows={len(status_rows)}",
        )
    )
    by_method = {row.get("method", ""): row for row in summary_rows}
    commot = by_method.get("commot", {})
    default = by_method.get("spatialself_wavelet", {})
    edge = by_method.get("spatialself_wavelet_edge_prior", {})
    gat = by_method.get("spatialself_gat", {})
    cellnest = by_method.get("cellnest", {})
    commot_aupr = _to_float(commot.get("aupr_mean", "nan"))
    default_aupr = _to_float(default.get("aupr_mean", "nan"))
    edge_weak = _to_float(edge.get("weak_signal_recall_at_threshold_mean", "nan"))
    default_weak = _to_float(default.get("weak_signal_recall_at_threshold_mean", "nan"))
    gat_topk = _to_float(gat.get("top_k_precision_mean", "nan"))
    cellnest_fdr = _to_float(cellnest.get("empirical_fdr_at_threshold_mean", "nan"))
    tradeoff_values_ok = all(
        math.isfinite(value)
        for value in (commot_aupr, default_aupr, edge_weak, default_weak, gat_topk, cellnest_fdr)
    )
    rows.append(
        AuditRow(
            "exp03_baseline_comparison",
            "full-slide sparse baseline records realistic-background method tradeoffs",
            "complete" if tradeoff_values_ok else "partial",
            str(summary_path.relative_to(PROJECT_ROOT)),
            (
                f"commot_aupr={commot_aupr:.3f}; spatialself_aupr={default_aupr:.3f}; "
                f"default_weak={default_weak:.3f}; edge_weak={edge_weak:.3f}; "
                f"gat_topk={gat_topk:.2f}; cellnest_fdr={cellnest_fdr:.3f}"
            ),
        )
    )
    return rows


def _audit_semisynthetic_injection_summary(
    path: Path,
    records: list[dict[str, str]],
    existing_status: str,
) -> list[AuditRow]:
    required_columns = {
        "lr_id",
        "ligand",
        "receptor",
        "mechanism",
        "weak_signal",
        "false_positive_high_expression",
        "candidates",
        "positives",
        "positive_rate",
        "type_rule_positive_fraction",
        "range_rule_positive_fraction",
        "niche_rule_positive_fraction",
        "heuristic_coexpr_distance_auroc",
        "heuristic_coexpr_distance_aupr",
    }
    rows: list[AuditRow] = []
    missing = sorted(required_columns - set(records[0] if records else {}))
    rows.append(
        AuditRow(
            "exp02_semisynthetic_real_background",
            f"{existing_status} injection summary has LR-level truth diagnostic columns",
            existing_status if records and not missing else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            f"missing={missing}; n_rows={len(records)}",
        )
    )
    if not records:
        return rows

    true_records = [row for row in records if not _as_bool(row.get("false_positive_high_expression", ""))]
    fp_records = [row for row in records if _as_bool(row.get("false_positive_high_expression", ""))]
    weak_records = [row for row in records if _as_bool(row.get("weak_signal", ""))]
    true_positive_counts = [_to_float(row.get("positives", "nan")) for row in true_records]
    fp_positive_counts = [_to_float(row.get("positives", "nan")) for row in fp_records]
    weak_positive_counts = [_to_float(row.get("positives", "nan")) for row in weak_records]
    truth_ok = (
        bool(true_records)
        and bool(fp_records)
        and bool(weak_records)
        and all(math.isfinite(value) and value > 0 for value in true_positive_counts)
        and all(math.isfinite(value) and value == 0 for value in fp_positive_counts)
        and all(math.isfinite(value) and value > 0 for value in weak_positive_counts)
    )
    rows.append(
        AuditRow(
            "exp02_semisynthetic_real_background",
            f"{existing_status} injected true LR, weak LR, and false-positive controls have expected labels",
            existing_status if truth_ok else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            (
                f"true_lr={len(true_records)}; weak_lr={len(weak_records)}; "
                f"false_positive_lr={len(fp_records)}; fp_positive_counts={fp_positive_counts}"
            ),
        )
    )

    rule_values = []
    for row in true_records:
        if _to_float(row.get("positives", "nan")) <= 0:
            continue
        for column in ("type_rule_positive_fraction", "range_rule_positive_fraction", "niche_rule_positive_fraction"):
            rule_values.append(_to_float(row.get(column, "nan")))
    rows.append(
        AuditRow(
            "exp02_semisynthetic_real_background",
            f"{existing_status} positive injected labels satisfy cell-type, range, and niche rules",
            existing_status if rule_values and min(rule_values) >= 0.999 else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            f"min_positive_rule_fraction={min(rule_values) if rule_values else float('nan'):.3f}",
        )
    )

    heuristic_values = []
    for row in true_records:
        for column in ("heuristic_coexpr_distance_auroc", "heuristic_coexpr_distance_aupr"):
            value = _to_float(row.get(column, "nan"))
            if math.isfinite(value):
                heuristic_values.append(value)
    rows.append(
        AuditRow(
            "exp02_semisynthetic_real_background",
            f"{existing_status} injection summary records finite heuristic diagnostics for true LR pairs",
            existing_status if heuristic_values and all(0.0 <= value <= 1.0 for value in heuristic_values) else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            f"n_finite_heuristic_values={len(heuristic_values)}",
        )
    )
    return rows


def _audit_semisynthetic_validation_summary(
    path: Path,
    records: list[dict[str, str]],
    existing_status: str,
    label: str,
) -> list[AuditRow]:
    rows: list[AuditRow] = []
    first = records[0] if records else {}
    preserved_columns = {
        "cells",
        "expression_genes",
        "mean_observed_sparsity",
        "median_library_size",
        "library_size_iqr",
        "coordinate_span_x",
        "coordinate_span_y",
        "median_nn_distance",
        "median_local_density",
        "dominant_region_fraction",
        "dominant_cell_type_fraction",
        "regions",
        "cell_types",
        "lr_pairs",
        "candidate_interactions",
        "positive_interactions",
        "weak_positive_interactions",
        "false_positive_labels",
        "heuristic_coexpr_distance_auroc",
        "heuristic_coexpr_distance_aupr",
    }
    missing = sorted(preserved_columns - set(first))
    rows.append(
        AuditRow(
            "exp02_semisynthetic_real_background",
            f"{label} validation records real background, truth, and heuristic diagnostic columns",
            existing_status if records and not missing else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            f"missing={missing}; n_rows={len(records)}",
        )
    )
    if not records:
        return rows

    cells = _to_float(first.get("cells", "nan"))
    expression_genes = _to_float(first.get("expression_genes", "nan"))
    sparsity = _to_float(first.get("mean_observed_sparsity", "nan"))
    library_iqr = _to_float(first.get("library_size_iqr", "nan"))
    coord_span_x = _to_float(first.get("coordinate_span_x", "nan"))
    coord_span_y = _to_float(first.get("coordinate_span_y", "nan"))
    median_nn = _to_float(first.get("median_nn_distance", "nan"))
    median_density = _to_float(first.get("median_local_density", "nan"))
    dominant_region = _to_float(first.get("dominant_region_fraction", "nan"))
    dominant_cell_type = _to_float(first.get("dominant_cell_type_fraction", "nan"))
    regions = _to_float(first.get("regions", "nan"))
    cell_types = _to_float(first.get("cell_types", "nan"))
    candidates = _to_float(first.get("candidate_interactions", "nan"))
    positives = _to_float(first.get("positive_interactions", "nan"))
    weak_positives = _to_float(first.get("weak_positive_interactions", "nan"))
    false_positive_labels = _to_float(first.get("false_positive_labels", "nan"))
    background_ok = (
        math.isfinite(cells)
        and cells >= 1000
        and math.isfinite(expression_genes)
        and expression_genes >= 1000
        and math.isfinite(sparsity)
        and 0.0 < sparsity < 1.0
        and math.isfinite(library_iqr)
        and library_iqr > 0
        and math.isfinite(coord_span_x)
        and coord_span_x > 0
        and math.isfinite(coord_span_y)
        and coord_span_y > 0
        and math.isfinite(median_nn)
        and median_nn > 0
        and math.isfinite(median_density)
        and median_density > 0
        and math.isfinite(dominant_region)
        and dominant_region < 1.0
        and math.isfinite(dominant_cell_type)
        and dominant_cell_type < 1.0
        and math.isfinite(regions)
        and regions >= 2
        and math.isfinite(cell_types)
        and cell_types >= 2
    )
    rows.append(
        AuditRow(
            "exp02_semisynthetic_real_background",
            f"{label} validation preserves non-degenerate real ST background statistics",
            existing_status if background_ok else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            (
                f"cells={cells:.0f}; genes={expression_genes:.0f}; sparsity={sparsity:.3f}; "
                f"library_iqr={library_iqr:.1f}; median_nn={median_nn:.4f}; "
                f"dominant_region={dominant_region:.3f}; dominant_cell_type={dominant_cell_type:.3f}"
            ),
        )
    )
    truth_ok = (
        math.isfinite(candidates)
        and candidates > 0
        and math.isfinite(positives)
        and positives > 0
        and math.isfinite(weak_positives)
        and weak_positives > 0
        and math.isfinite(false_positive_labels)
        and false_positive_labels == 0
    )
    rows.append(
        AuditRow(
            "exp02_semisynthetic_real_background",
            f"{label} validation confirms injected positives, weak positives, and zero false-positive labels",
            existing_status if truth_ok else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            (
                f"candidates={candidates:.0f}; positives={positives:.0f}; "
                f"weak_positives={weak_positives:.0f}; false_positive_labels={false_positive_labels:.0f}"
            ),
        )
    )
    heuristic_auroc = _to_float(first.get("heuristic_coexpr_distance_auroc", "nan"))
    heuristic_aupr = _to_float(first.get("heuristic_coexpr_distance_aupr", "nan"))
    rows.append(
        AuditRow(
            "exp02_semisynthetic_real_background",
            f"{label} validation records finite coexpression-distance heuristic diagnostics",
            existing_status
            if (
                math.isfinite(heuristic_auroc)
                and 0.0 <= heuristic_auroc <= 1.0
                and math.isfinite(heuristic_aupr)
                and 0.0 <= heuristic_aupr <= 1.0
            )
            else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            f"heuristic_auroc={heuristic_auroc:.3f}; heuristic_aupr={heuristic_aupr:.3f}",
        )
    )
    return rows


def _audit_semisynthetic_realism(summary_path: Path) -> list[AuditRow]:
    rows: list[AuditRow] = []
    records = _read_csv(summary_path)
    by_metric = {row.get("metric", ""): _to_float(row.get("value", "nan")) for row in records}
    required = {
        "cells",
        "genes_compared",
        "lr_genes_compared",
        "selected_library_size_pearson",
        "cell_sparsity_pearson",
        "gene_mean_pearson_background",
        "gene_variance_pearson_background",
        "gene_sparsity_pearson_background",
        "moran_pearson_background",
        "median_mean_delta_lr_genes",
        "coordinate_span_x",
        "coordinate_span_y",
        "n_regions",
        "n_cell_types",
    }
    missing = sorted(required - set(by_metric))
    rows.append(
        AuditRow(
            "exp02_semisynthetic_real_background",
            "full-slide sparse realism summary contains key ST statistics",
            "complete" if not missing else "partial",
            str(summary_path.relative_to(PROJECT_ROOT)),
            f"missing={missing}; n_metrics={len(records)}",
        )
    )
    background_ok = (
        by_metric.get("selected_library_size_pearson", 0.0) >= 0.99
        and by_metric.get("cell_sparsity_pearson", 0.0) >= 0.99
        and by_metric.get("gene_mean_pearson_background", 0.0) >= 0.99
        and by_metric.get("gene_variance_pearson_background", 0.0) >= 0.99
        and by_metric.get("gene_sparsity_pearson_background", 0.0) >= 0.99
        and by_metric.get("moran_pearson_background", 0.0) >= 0.95
        and abs(by_metric.get("cell_sparsity_mean_delta", 1.0)) <= 0.01
    )
    rows.append(
        AuditRow(
            "exp02_semisynthetic_real_background",
            "full-slide sparse realism preserves real expression, sparsity, and spatial autocorrelation background",
            "complete" if background_ok else "partial",
            str(summary_path.relative_to(PROJECT_ROOT)),
            (
                f"library_corr={by_metric.get('selected_library_size_pearson', float('nan')):.4f}; "
                f"cell_sparsity_corr={by_metric.get('cell_sparsity_pearson', float('nan')):.4f}; "
                f"bg_mean_corr={by_metric.get('gene_mean_pearson_background', float('nan')):.4f}; "
                f"bg_moran_corr={by_metric.get('moran_pearson_background', float('nan')):.4f}"
            ),
        )
    )
    injection_ok = (
        by_metric.get("lr_genes_compared", 0.0) >= 5
        and by_metric.get("median_mean_delta_lr_genes", 0.0) > 0.0
        and by_metric.get("max_mean_delta_lr_genes", 0.0) > by_metric.get("median_mean_delta_lr_genes", 0.0)
    )
    rows.append(
        AuditRow(
            "exp02_semisynthetic_real_background",
            "full-slide sparse realism localizes expression shifts to injected LR genes",
            "complete" if injection_ok else "partial",
            str(summary_path.relative_to(PROJECT_ROOT)),
            (
                f"lr_genes={by_metric.get('lr_genes_compared', float('nan')):.0f}; "
                f"median_lr_mean_delta={by_metric.get('median_mean_delta_lr_genes', float('nan')):.3f}; "
                f"max_lr_mean_delta={by_metric.get('max_mean_delta_lr_genes', float('nan')):.3f}"
            ),
        )
    )
    geometry_ok = (
        by_metric.get("cells", 0.0) >= 4000
        and by_metric.get("genes_compared", 0.0) >= 2000
        and by_metric.get("coordinate_span_x", 0.0) > 0.0
        and by_metric.get("coordinate_span_y", 0.0) > 0.0
        and by_metric.get("n_regions", 0.0) >= 3
        and by_metric.get("n_cell_types", 0.0) >= 3
    )
    rows.append(
        AuditRow(
            "exp02_semisynthetic_real_background",
            "full-slide sparse realism preserves geometry, regions, and cell-type composition support",
            "complete" if geometry_ok else "partial",
            str(summary_path.relative_to(PROJECT_ROOT)),
            (
                f"cells={by_metric.get('cells', float('nan')):.0f}; "
                f"genes={by_metric.get('genes_compared', float('nan')):.0f}; "
                f"regions={by_metric.get('n_regions', float('nan')):.0f}; "
                f"cell_types={by_metric.get('n_cell_types', float('nan')):.0f}"
            ),
        )
    )
    return rows


def _audit_semisynthetic_mechanism_summary(
    path: Path,
    records: list[dict[str, str]],
    existing_status: str,
) -> list[AuditRow]:
    required_columns = {
        "mechanism",
        "candidates",
        "positives",
        "weak_positives",
        "false_positive_candidates",
        "false_positive_labels",
        "positive_rate",
        "positive_median_distance",
        "type_rule_positive_fraction",
        "range_rule_positive_fraction",
        "niche_rule_positive_fraction",
    }
    expected_mechanisms = {
        "long_range_chemokine_niche",
        "short_medium_paracrine",
        "false_positive_high_expression",
    }
    rows: list[AuditRow] = []
    missing = sorted(required_columns - set(records[0] if records else {}))
    mechanisms = {row.get("mechanism", "") for row in records}
    rows.append(
        AuditRow(
            "exp02_semisynthetic_real_background",
            "semi-synthetic mechanism summary covers long-range, short-range, and false-positive mechanisms",
            existing_status if records and not missing and expected_mechanisms.issubset(mechanisms) else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            f"missing={missing}; mechanisms={sorted(mechanisms)}",
        )
    )

    by_mechanism = {row.get("mechanism", ""): row for row in records}
    long_distance = _to_float(
        by_mechanism.get("long_range_chemokine_niche", {}).get("positive_median_distance", "nan")
    )
    short_distance = _to_float(
        by_mechanism.get("short_medium_paracrine", {}).get("positive_median_distance", "nan")
    )
    fp_labels = _to_float(
        by_mechanism.get("false_positive_high_expression", {}).get("false_positive_labels", "nan")
    )
    distance_ok = (
        math.isfinite(long_distance)
        and math.isfinite(short_distance)
        and long_distance > short_distance
        and math.isfinite(fp_labels)
        and fp_labels == 0
    )
    rows.append(
        AuditRow(
            "exp02_semisynthetic_real_background",
            "semi-synthetic mechanism distances and false-positive labels match injected design",
            existing_status if distance_ok else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            f"long_range_median={long_distance:.3f}; short_range_median={short_distance:.3f}; fp_labels={fp_labels:.0f}",
        )
    )
    return rows


def _real_validation_ranking_values_valid(rows: list[dict[str, str]]) -> bool:
    if not rows:
        return False
    ranks = []
    for row in rows:
        for column in ("n_edges", "mean_score", "max_score", "top_region_mean_score", "global_rank"):
            value = _to_float(row.get(column, "nan"))
            if not math.isfinite(value):
                return False
            if column in {"mean_score", "max_score", "top_region_mean_score"} and not (0.0 <= value <= 1.0):
                return False
            if column in {"n_edges", "global_rank"} and value <= 0:
                return False
        ranks.append(int(_to_float(row.get("global_rank", "nan"))))
    return sorted(ranks) == list(range(1, len(ranks) + 1))


def _downstream_values_valid(rows: list[dict[str, str]]) -> bool:
    if not rows:
        return False
    for row in rows:
        n_edges = _to_float(row.get("n_edges", "nan"))
        n_receiver_cells = _to_float(row.get("n_receiver_cells", "nan"))
        n_targets = _to_float(row.get("n_targets_present", "nan"))
        if not (math.isfinite(n_edges) and n_edges > 0 and math.isfinite(n_receiver_cells) and n_receiver_cells > 0):
            return False
        if not (math.isfinite(n_targets) and n_targets >= 0):
            return False
        if row.get("status") == "tested":
            spearman = _to_float(row.get("spearman_r", "nan"))
            empirical_p = _to_float(row.get("empirical_p_value", "nan"))
            bh_fdr = _to_float(row.get("bh_fdr", "nan"))
            if not (math.isfinite(spearman) and -1.0 <= spearman <= 1.0):
                return False
            if not (math.isfinite(empirical_p) and 0.0 <= empirical_p <= 1.0):
                return False
            if not (math.isfinite(bh_fdr) and 0.0 <= bh_fdr <= 1.0):
                return False
    return True


def _weak_stress_metric_table_valid(rows: list[dict[str, str]]) -> bool:
    if not rows:
        return False
    required_columns = {
        "method",
        "auroc",
        "aupr",
        "top_k_precision",
        "recall_at_threshold",
        "weak_signal_recall_at_threshold",
        "empirical_fdr_at_threshold",
        "false_positive_high_expression_rate_at_threshold",
        "calibration_error_10bin",
        "candidate_path",
        "prediction_path",
    }
    probability_columns = {
        "auroc",
        "aupr",
        "top_k_precision",
        "recall_at_threshold",
        "weak_signal_recall_at_threshold",
        "empirical_fdr_at_threshold",
        "false_positive_high_expression_rate_at_threshold",
        "calibration_error_10bin",
    }
    for row in rows:
        if not required_columns.issubset(row):
            return False
        for column in probability_columns:
            value = _to_float(row.get(column, "nan"))
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                return False
        for column in ("candidate_path", "prediction_path"):
            if not _project_path(row.get(column, "")).exists():
                return False
    return True


def _weak_stress_status_valid(rows: list[dict[str, str]], required_methods: set[str]) -> bool:
    if not rows:
        return False
    methods = {row.get("method", "") for row in rows}
    if not required_methods.issubset(methods):
        return False
    for row in rows:
        if row.get("status") != "completed":
            return False
        if not _project_path(row.get("prediction_path", "")).exists():
            return False
    return True


def _full_weak_stress_status_valid(
    rows: list[dict[str, str]],
    required_methods: set[str],
    *,
    min_replicates: int,
) -> bool:
    if not _weak_stress_status_valid(rows, required_methods):
        return False
    for method in required_methods:
        method_rows = [row for row in rows if row.get("method") == method]
        replicates = {row.get("replicate", "") for row in method_rows}
        if len(replicates) < int(min_replicates):
            return False
    return True


def _audit_full_weak_stress_metrics(
    path: Path,
    records: list[dict[str, str]],
    required_methods: set[str],
) -> list[AuditRow]:
    rows: list[AuditRow] = []
    methods = {row.get("method", "") for row in records}
    replicates = {row.get("replicate", "") for row in records}
    rows.append(
        AuditRow(
            "exp05_weak_signal_false_positive",
            "full stress metrics cover 5 controlled replicates and SpatialSelf variants",
            "complete" if len(replicates) >= 5 and required_methods.issubset(methods) else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            f"n_replicates={len(replicates)}; methods={sorted(methods)}; n_rows={len(records)}",
        )
    )
    rows.append(
        AuditRow(
            "exp05_weak_signal_false_positive",
            "full stress metrics include valid weak/FDR/false-positive/calibration columns",
            "complete" if _weak_stress_metric_table_valid(records) else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            f"n_rows={len(records)}",
        )
    )
    means = _mean_metric_by_method(
        records,
        (
            "aupr",
            "weak_signal_recall_at_threshold",
            "empirical_fdr_at_threshold",
            "false_positive_high_expression_rate_at_threshold",
            "auroc",
        ),
    )
    default = means.get("spatialself_wavelet", {})
    candidate = means.get("spatialself_wavelet_edge_prior", {})
    gat = means.get("spatialself_gat", {})
    default_aupr = default.get("aupr", float("nan"))
    candidate_aupr = candidate.get("aupr", float("nan"))
    default_weak = default.get("weak_signal_recall_at_threshold", float("nan"))
    candidate_weak = candidate.get("weak_signal_recall_at_threshold", float("nan"))
    default_fp = default.get("false_positive_high_expression_rate_at_threshold", float("nan"))
    candidate_fp = candidate.get("false_positive_high_expression_rate_at_threshold", float("nan"))
    candidate_tradeoff_ok = (
        math.isfinite(default_aupr)
        and math.isfinite(candidate_aupr)
        and math.isfinite(default_weak)
        and math.isfinite(candidate_weak)
        and math.isfinite(default_fp)
        and math.isfinite(candidate_fp)
        and candidate_aupr >= default_aupr
        and candidate_weak >= default_weak
        and candidate_fp <= default_fp
    )
    rows.append(
        AuditRow(
            "exp05_weak_signal_false_positive",
            "full edge-prior stress improves AUPR/weak recall and suppresses high-expression false positives over default",
            "complete" if candidate_tradeoff_ok else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            (
                f"default_aupr={default_aupr:.3f}; edge_aupr={candidate_aupr:.3f}; "
                f"default_weak={default_weak:.3f}; edge_weak={candidate_weak:.3f}; "
                f"default_fp={default_fp:.3f}; edge_fp={candidate_fp:.3f}"
            ),
        )
    )
    gat_aupr = gat.get("aupr", float("nan"))
    gat_fdr = gat.get("empirical_fdr_at_threshold", float("nan"))
    rows.append(
        AuditRow(
            "exp05_weak_signal_false_positive",
            "full GAT stress records weak-recall/FDR tradeoff separately from wavelet variants",
            "complete" if math.isfinite(gat_aupr) and math.isfinite(gat_fdr) and gat_fdr >= 0.0 else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            f"gat_aupr={gat_aupr:.3f}; gat_fdr={gat_fdr:.3f}",
        )
    )
    return rows


def _mean_metric_by_method(
    records: list[dict[str, str]],
    columns: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, list[float]]] = {}
    for row in records:
        method = row.get("method", "")
        if not method:
            continue
        values.setdefault(method, {column: [] for column in columns})
        for column in columns:
            value = _to_float(row.get(column, "nan"))
            if math.isfinite(value):
                values[method][column].append(value)
    return {
        method: {
            column: _mean(column_values)
            for column, column_values in method_values.items()
        }
        for method, method_values in values.items()
    }


def _audit_weak_stress_tables(
    stress_dir: Path,
    required_methods: set[str],
    *,
    existing_status: str = "smoke_ready",
    min_replicates: int = 1,
) -> list[AuditRow]:
    rows: list[AuditRow] = []
    expected_suffixes = {
        "threshold": "__threshold_stress.csv",
        "topk": "__topk_stress.csv",
        "lr": "__lr_stress.csv",
        "mechanism": "__mechanism_stress.csv",
        "summary": "__stress_summary.json",
    }
    expected_mechanisms = {
        "juxtacrine_contact",
        "short_medium_paracrine",
        "long_range_chemokine_niche",
        "false_positive_high_expression",
    }
    all_files_present = True
    table_schemas_valid = True
    lr_coverage_ok = True
    mechanism_coverage_ok = True
    summary_json_ok = True
    details: list[str] = []

    for method in sorted(required_methods):
        method_files = {
            label: sorted(stress_dir.glob(f"*__{method}{suffix}"))
            for label, suffix in expected_suffixes.items()
        }
        missing_labels = [
            label for label, matches in method_files.items() if len(matches) < int(min_replicates)
        ]
        if missing_labels:
            all_files_present = False
            details.append(f"{method}:missing={missing_labels}")
            continue
        details.append(f"{method}:n={len(method_files['summary'])}")

        for idx in range(int(min_replicates)):
            threshold_rows = _read_csv(method_files["threshold"][idx])
            topk_rows = _read_csv(method_files["topk"][idx])
            lr_rows = _read_csv(method_files["lr"][idx])
            mechanism_rows = _read_csv(method_files["mechanism"][idx])
            table_schemas_valid = table_schemas_valid and _weak_threshold_stress_valid(threshold_rows)
            table_schemas_valid = table_schemas_valid and _weak_topk_stress_valid(topk_rows)
            table_schemas_valid = table_schemas_valid and _weak_lr_stress_valid(lr_rows)
            table_schemas_valid = table_schemas_valid and _weak_mechanism_stress_valid(mechanism_rows)

            has_weak_lr = any(_as_bool(row.get("weak_signal", "")) for row in lr_rows)
            has_false_positive_lr = any(_as_bool(row.get("false_positive_high_expression", "")) for row in lr_rows)
            lr_coverage_ok = lr_coverage_ok and has_weak_lr and has_false_positive_lr

            mechanisms = {row.get("mechanism", "") for row in mechanism_rows}
            mechanism_coverage_ok = mechanism_coverage_ok and expected_mechanisms.issubset(mechanisms)

            try:
                with method_files["summary"][idx].open() as handle:
                    summary = json.load(handle)
            except (OSError, json.JSONDecodeError):
                summary_json_ok = False
            else:
                summary_json_ok = summary_json_ok and _weak_stress_summary_valid(summary)

    rows.append(
        AuditRow(
            "exp05_weak_signal_false_positive",
            "per-method threshold/top-k/LR/mechanism stress tables and summaries exist",
            existing_status if all_files_present else "partial",
            str(stress_dir.relative_to(PROJECT_ROOT)),
            "; ".join(details),
        )
    )
    rows.append(
        AuditRow(
            "exp05_weak_signal_false_positive",
            "stress tables have valid schemas and bounded metric values",
            existing_status if all_files_present and table_schemas_valid and summary_json_ok else "partial",
            str(stress_dir.relative_to(PROJECT_ROOT)),
            f"summary_json_ok={summary_json_ok}",
        )
    )
    rows.append(
        AuditRow(
            "exp05_weak_signal_false_positive",
            "LR stress covers weak-signal and high-expression false-positive LR pairs",
            existing_status if all_files_present and lr_coverage_ok else "partial",
            str(stress_dir.relative_to(PROJECT_ROOT)),
        )
    )
    rows.append(
        AuditRow(
            "exp05_weak_signal_false_positive",
            "mechanism stress covers contact, paracrine, long-range niche, and false-positive categories",
            existing_status if all_files_present and mechanism_coverage_ok else "partial",
            str(stress_dir.relative_to(PROJECT_ROOT)),
        )
    )
    return rows


def _weak_threshold_stress_valid(rows: list[dict[str, str]]) -> bool:
    required_columns = {
        "threshold_quantile",
        "threshold_score",
        "n_predicted_positive",
        "true_positive",
        "false_positive",
        "recall",
        "weak_signal_recall",
        "empirical_fdr",
        "false_positive_high_expression_call_rate",
        "false_positive_high_expression_fraction_among_calls",
    }
    if not rows or not required_columns.issubset(rows[0]):
        return False
    probability_columns = (
        "threshold_quantile",
        "threshold_score",
        "recall",
        "weak_signal_recall",
        "empirical_fdr",
        "false_positive_high_expression_call_rate",
        "false_positive_high_expression_fraction_among_calls",
    )
    count_columns = ("n_predicted_positive", "true_positive", "false_positive")
    for row in rows:
        if not _probability_columns_in_unit_interval([row], probability_columns):
            return False
        for column in count_columns:
            value = _to_float(row.get(column, "nan"))
            if not math.isfinite(value) or value < 0:
                return False
    return True


def _weak_topk_stress_valid(rows: list[dict[str, str]]) -> bool:
    required_columns = {
        "top_k",
        "precision",
        "weak_true_count",
        "false_positive_high_expression_count",
        "false_positive_high_expression_fraction",
        "mechanism_counts",
    }
    if not rows or not required_columns.issubset(rows[0]):
        return False
    for row in rows:
        top_k = _to_float(row.get("top_k", "nan"))
        weak_count = _to_float(row.get("weak_true_count", "nan"))
        fp_count = _to_float(row.get("false_positive_high_expression_count", "nan"))
        if not (math.isfinite(top_k) and top_k > 0 and math.isfinite(weak_count) and weak_count >= 0):
            return False
        if not math.isfinite(fp_count) or fp_count < 0:
            return False
        if not _probability_columns_in_unit_interval(
            [row],
            ("precision", "false_positive_high_expression_fraction"),
        ):
            return False
        try:
            mechanism_counts = json.loads(row.get("mechanism_counts", "{}"))
        except json.JSONDecodeError:
            return False
        if not isinstance(mechanism_counts, dict):
            return False
    return True


def _weak_lr_stress_valid(rows: list[dict[str, str]]) -> bool:
    required_columns = {
        "lr_id",
        "mechanism",
        "weak_signal",
        "false_positive_high_expression",
        "n_candidates",
        "n_positive",
        "mean_score",
        "q95_score",
        "auroc",
    }
    if not rows or not required_columns.issubset(rows[0]):
        return False
    for row in rows:
        n_candidates = _to_float(row.get("n_candidates", "nan"))
        n_positive = _to_float(row.get("n_positive", "nan"))
        mean_score = _to_float(row.get("mean_score", "nan"))
        q95_score = _to_float(row.get("q95_score", "nan"))
        if not (math.isfinite(n_candidates) and n_candidates > 0 and math.isfinite(n_positive) and n_positive >= 0):
            return False
        if not (math.isfinite(mean_score) and 0.0 <= mean_score <= 1.0):
            return False
        if not (math.isfinite(q95_score) and 0.0 <= q95_score <= 1.0):
            return False
        auroc = _to_float(row.get("auroc", "nan"))
        if n_positive > 0 and (not math.isfinite(auroc) or auroc < 0.0 or auroc > 1.0):
            return False
    return True


def _weak_mechanism_stress_valid(rows: list[dict[str, str]]) -> bool:
    required_columns = {
        "mechanism",
        "n_candidates",
        "n_positive",
        "weak_positive",
        "false_positive_high_expression_candidates",
        "mean_score",
        "auroc",
    }
    if not rows or not required_columns.issubset(rows[0]):
        return False
    for row in rows:
        n_candidates = _to_float(row.get("n_candidates", "nan"))
        n_positive = _to_float(row.get("n_positive", "nan"))
        weak_positive = _to_float(row.get("weak_positive", "nan"))
        fp_candidates = _to_float(row.get("false_positive_high_expression_candidates", "nan"))
        mean_score = _to_float(row.get("mean_score", "nan"))
        if not (
            math.isfinite(n_candidates)
            and n_candidates > 0
            and math.isfinite(n_positive)
            and n_positive >= 0
            and math.isfinite(weak_positive)
            and weak_positive >= 0
            and math.isfinite(fp_candidates)
            and fp_candidates >= 0
        ):
            return False
        if not (math.isfinite(mean_score) and 0.0 <= mean_score <= 1.0):
            return False
        auroc = _to_float(row.get("auroc", "nan"))
        if n_positive > 0 and (not math.isfinite(auroc) or auroc < 0.0 or auroc > 1.0):
            return False
    return True


def _weak_stress_summary_valid(summary: object) -> bool:
    if not isinstance(summary, dict):
        return False
    required_probability_keys = {
        "auroc",
        "aupr",
        "top_k_precision",
        "recall_at_threshold",
        "weak_signal_recall_at_threshold",
        "empirical_fdr_at_threshold",
        "false_positive_high_expression_rate_at_threshold",
        "calibration_error_10bin",
    }
    for key in required_probability_keys:
        value = _to_float(summary.get(key, "nan"))
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            return False
    for key in ("candidate_path", "prediction_path"):
        if not _project_path(str(summary.get(key, ""))).exists():
            return False
    return True


def _runtime_schema_valid(rows: list[dict[str, str]]) -> bool:
    if not rows:
        return False
    required_columns = {
        "scale_n_cells",
        "dataset_root",
        "n_replicates",
        "n_cells_or_spots",
        "n_candidate_edges",
        "n_lr_pairs",
        "method",
        "status",
        "detail",
        "wall_clock_seconds",
        "peak_cpu_memory_mb",
        "peak_gpu_memory_mb",
        "gpu_monitor_status",
        "runtime_per_epoch_seconds",
        "total_runtime_seconds",
        "returncode",
        "command",
        "stdout_log",
        "stderr_log",
    }
    return required_columns.issubset(rows[0])


def _runtime_profile_schema_valid(rows: list[dict[str, str]]) -> bool:
    if not rows:
        return False
    required_columns = {
        "dataset_root",
        "n_replicates",
        "n_cells_or_spots",
        "n_candidate_edges",
        "n_lr_pairs",
        "method",
        "status",
        "detail",
        "wall_clock_seconds",
        "peak_cpu_memory_mb",
        "peak_gpu_memory_mb",
        "gpu_monitor_status",
        "runtime_per_epoch_seconds",
        "total_runtime_seconds",
        "returncode",
        "command",
        "stdout_log",
        "stderr_log",
    }
    return required_columns.issubset(rows[0])


def _runtime_profile_resource_values_valid(rows: list[dict[str, str]]) -> bool:
    if not rows:
        return False
    for row in rows:
        numeric_positive = (
            "n_cells_or_spots",
            "n_candidate_edges",
            "n_lr_pairs",
            "wall_clock_seconds",
            "peak_cpu_memory_mb",
            "total_runtime_seconds",
        )
        for column in numeric_positive:
            value = _to_float(row.get(column, "nan"))
            if not math.isfinite(value) or value <= 0:
                return False
        if row.get("method", "").startswith("spatialself_"):
            runtime_per_epoch = _to_float(row.get("runtime_per_epoch_seconds", "nan"))
            if not math.isfinite(runtime_per_epoch) or runtime_per_epoch <= 0:
                return False
        returncode = _to_float(row.get("returncode", "nan"))
        if not math.isfinite(returncode) or int(returncode) != 0:
            return False
        if not str(row.get("command", "")).strip():
            return False
        for log_column in ("stdout_log", "stderr_log"):
            if not _project_path(row.get(log_column, "")).exists():
                return False
        gpu_memory = _to_float(row.get("peak_gpu_memory_mb", "nan"))
        gpu_status = str(row.get("gpu_monitor_status", ""))
        if math.isfinite(gpu_memory) and gpu_memory < 0:
            return False
        if not math.isfinite(gpu_memory) and not gpu_status:
            return False
    return True


def _runtime_resource_values_valid(rows: list[dict[str, str]]) -> bool:
    if not rows:
        return False
    for row in rows:
        numeric_positive = (
            "scale_n_cells",
            "n_cells_or_spots",
            "n_candidate_edges",
            "n_lr_pairs",
            "wall_clock_seconds",
            "peak_cpu_memory_mb",
            "runtime_per_epoch_seconds",
            "total_runtime_seconds",
        )
        for column in numeric_positive:
            value = _to_float(row.get(column, "nan"))
            if not math.isfinite(value) or value <= 0:
                return False
        returncode = _to_float(row.get("returncode", "nan"))
        if not math.isfinite(returncode) or int(returncode) != 0:
            return False
        if not str(row.get("command", "")).strip():
            return False
        for log_column in ("stdout_log", "stderr_log"):
            path = _project_path(row.get(log_column, ""))
            if not path.exists():
                return False
        gpu_memory = _to_float(row.get("peak_gpu_memory_mb", "nan"))
        gpu_status = str(row.get("gpu_monitor_status", ""))
        if math.isfinite(gpu_memory) and gpu_memory < 0:
            return False
        if not math.isfinite(gpu_memory) and not gpu_status:
            return False
    return True


def _runtime_medium_resource_values_valid(rows: list[dict[str, str]]) -> bool:
    if not rows:
        return False
    for row in rows:
        numeric_positive = (
            "scale_n_cells",
            "n_cells_or_spots",
            "n_candidate_edges",
            "n_lr_pairs",
            "wall_clock_seconds",
            "peak_cpu_memory_mb",
            "total_runtime_seconds",
        )
        for column in numeric_positive:
            value = _to_float(row.get(column, "nan"))
            if not math.isfinite(value) or value <= 0:
                return False
        returncode = _to_float(row.get("returncode", "nan"))
        if not math.isfinite(returncode) or int(returncode) != 0:
            return False
        if row.get("method", "").startswith("spatialself_"):
            runtime_per_epoch = _to_float(row.get("runtime_per_epoch_seconds", "nan"))
            if not math.isfinite(runtime_per_epoch) or runtime_per_epoch <= 0:
                return False
        if not str(row.get("command", "")).strip():
            return False
        for log_column in ("stdout_log", "stderr_log"):
            if not _project_path(row.get(log_column, "")).exists():
                return False
        gpu_memory = _to_float(row.get("peak_gpu_memory_mb", "nan"))
        gpu_status = str(row.get("gpu_monitor_status", ""))
        if math.isfinite(gpu_memory) and gpu_memory < 0:
            return False
        if not math.isfinite(gpu_memory) and not gpu_status:
            return False
    return True


def _runtime_edges_increase_by_scale(rows: list[dict[str, str]]) -> bool:
    by_method: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        scale = _to_float(row.get("scale_n_cells", "nan"))
        edges = _to_float(row.get("n_candidate_edges", "nan"))
        method = row.get("method", "")
        if not (method and math.isfinite(scale) and math.isfinite(edges)):
            return False
        by_method.setdefault(method, []).append((scale, edges))
    if not by_method:
        return False
    for values in by_method.values():
        if len(values) < 2:
            return False
        values = sorted(values)
        for (_, prev_edges), (_, current_edges) in zip(values, values[1:]):
            if current_edges <= prev_edges:
                return False
    return True


def _audit_medium_runtime_scaling(path: Path) -> list[AuditRow]:
    rows: list[AuditRow] = []
    records = _read_csv(path)
    completed = [row for row in records if row.get("status") == "completed"]
    scales = sorted({_to_float(row.get("scale_n_cells", "nan")) for row in records})
    methods = {row.get("method", "") for row in records}
    required_methods = {"spatialself_wavelet", "spatialself_wavelet_edge_prior", "spatialself_gat", "commot"}
    required_scales = {500.0, 1000.0}
    rows.append(
        AuditRow(
            "exp08_runtime_memory",
            "controlled medium runtime covers 500/1000 cells and SpatialSelf variants plus COMMOT",
            "complete"
            if required_scales.issubset(set(scales)) and required_methods.issubset(methods)
            else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            f"scales={[int(scale) for scale in scales if math.isfinite(scale)]}; methods={sorted(methods)}",
        )
    )
    rows.append(
        AuditRow(
            "exp08_runtime_memory",
            "controlled medium runtime schema includes wall-clock, memory, candidate edges, LR pairs, logs, and commands",
            "complete" if _runtime_schema_valid(records) else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            f"n_rows={len(records)}",
        )
    )
    rows.append(
        AuditRow(
            "exp08_runtime_memory",
            "controlled medium runtime all profiled methods completed with non-degenerate resource values",
            "complete" if len(completed) == len(records) and _runtime_medium_resource_values_valid(completed) else "partial",
            str(path.relative_to(PROJECT_ROOT)),
            f"n_completed={len(completed)}; n_rows={len(records)}",
        )
    )
    rows.append(
        AuditRow(
            "exp08_runtime_memory",
            "controlled medium runtime candidate edges increase with scale for every method",
            "complete" if _runtime_edges_increase_by_scale(completed) else "partial",
            str(path.relative_to(PROJECT_ROOT)),
        )
    )
    if completed:
        max_scale = max(_to_float(row.get("scale_n_cells", "nan")) for row in completed)
        max_scale_rows = [row for row in completed if _to_float(row.get("scale_n_cells", "nan")) == max_scale]
        max_edges = max(_to_float(row.get("n_candidate_edges", "nan")) for row in max_scale_rows)
        spatialself_times = [
            _to_float(row.get("wall_clock_seconds", "nan"))
            for row in max_scale_rows
            if row.get("method", "").startswith("spatialself_")
        ]
        commot_rows = [row for row in max_scale_rows if row.get("method") == "commot"]
        commot_time = _to_float(commot_rows[0].get("wall_clock_seconds", "nan")) if commot_rows else float("nan")
        commot_memory = _to_float(commot_rows[0].get("peak_cpu_memory_mb", "nan")) if commot_rows else float("nan")
        medium_scale_ok = (
            math.isfinite(max_scale)
            and max_scale >= 1000
            and math.isfinite(max_edges)
            and max_edges >= 900000
            and spatialself_times
            and all(math.isfinite(value) and value > 0 for value in spatialself_times)
            and math.isfinite(commot_time)
            and commot_time > 0
            and math.isfinite(commot_memory)
            and commot_memory > 0
        )
        rows.append(
            AuditRow(
                "exp08_runtime_memory",
                "controlled medium runtime reaches near-million candidate edges with comparable COMMOT timing",
                "complete" if medium_scale_ok else "partial",
                str(path.relative_to(PROJECT_ROOT)),
                (
                    f"max_scale={max_scale:.0f}; max_edges={max_edges:.0f}; "
                    f"spatialself_time_range=({min(spatialself_times):.2f},{max(spatialself_times):.2f}); "
                    f"commot_time={commot_time:.2f}; commot_memory_mb={commot_memory:.1f}"
                ),
            )
        )
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
