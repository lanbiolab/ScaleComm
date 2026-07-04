#!/usr/bin/env python
"""Audit the revised Phase 2 evidence plan."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class AuditRow:
    goal_id: str
    requirement: str
    status: str
    evidence: str
    detail: str = ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/phase2_revised/audit")
    args = parser.parse_args()

    rows: list[AuditRow] = []
    rows.extend(_audit_plan())
    rows.extend(_audit_goal1_baseline_matched_validation())
    rows.extend(_audit_goal1_formal_baseline_matched_validation())
    rows.extend(_audit_goal1_injected_downstream_proxy())
    rows.extend(_audit_goal1_curated_downstream_panel())
    rows.extend(_audit_goal1_biological_validation_readiness())
    rows.extend(_audit_goal2_weak_signal_tradeoff())
    rows.extend(_audit_goal2_formal_baseline_panel())
    rows.extend(_audit_goal2_formal_tuned_small_panel())
    rows.extend(_audit_goal2_tuned_guardrail_sensitivity())
    rows.extend(_audit_goal2_preregistered_endpoint_policy())
    rows.extend(_audit_goal2_preregistered_paired_diagnostics())
    rows.extend(_audit_goal2_replicate_power_plan())
    rows.extend(_audit_goal2_replicated_tuned_job_matrix())
    rows.extend(_audit_goal4_replicated_tuned_datasets())
    rows.extend(_audit_goal2_weak_signal_stress_formal())
    rows.extend(_audit_goal2_weak_signal_tradeoff_readiness())
    rows.extend(_audit_goal3_simulator_v2())
    rows.extend(_audit_goal3_formal_simulator_v2())
    rows.extend(_audit_goal4_real_background_v2())
    rows.extend(_audit_goal4_formal_real_background_v2())
    rows.extend(_audit_goal34_benchmark_quality_report())
    rows.extend(_audit_goal34_truth_contract_validator())
    rows.extend(_audit_goal5_formal_fdr())
    rows.extend(_audit_goal5_fdr_200perm())
    rows.extend(_audit_goal5_real_background_fdr_panel())
    rows.extend(_audit_goal5_real_background_fdr_100perm())
    rows.extend(_audit_goal6_manuscript_assembly())
    rows.extend(_audit_goal6_results_synthesis())
    rows.extend(_audit_phase2_evidence_lock())
    rows.extend(_audit_goal6_revised_figures())
    rows.extend(_audit_goal6_runtime_scalability())
    rows.extend(_audit_coexpression_baseline_removed())

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "phase2_revised_audit_summary.csv"
    json_path = out_dir / "phase2_revised_audit_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["goal_id", "requirement", "status", "evidence", "detail"])
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    json_path.write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")

    counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    print(f"Revised Phase 2 audit CSV:  {csv_path}")
    print(f"Revised Phase 2 audit JSON: {json_path}")
    print("Status counts:")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")
    print("\nNon-complete rows:")
    for row in rows:
        if row.status != "complete":
            print(f"  {row.goal_id} | {row.status} | {row.requirement} | {row.detail}")


def _audit_plan() -> list[AuditRow]:
    docs_path = PROJECT_ROOT / "docs/PHASE2_REVISED_GOALS.md"
    common_path = PROJECT_ROOT / "configs/experiments/phase2_revised_common.yaml"
    goal1_path = PROJECT_ROOT / "configs/experiments/phase2_revised_goal1_baseline_matched_validation.yaml"
    goal2_path = PROJECT_ROOT / "configs/experiments/phase2_revised_goal2_weak_signal_tradeoff.yaml"
    goal3_path = PROJECT_ROOT / "configs/experiments/phase2_revised_goal3_simulator_v2.yaml"
    goal4_path = PROJECT_ROOT / "configs/experiments/phase2_revised_goal4_real_background_v2.yaml"
    goal5_path = PROJECT_ROOT / "configs/experiments/phase2_revised_goal5_formal_fdr.yaml"
    goal6_path = PROJECT_ROOT / "configs/experiments/phase2_revised_goal6_manuscript_assembly.yaml"
    full_baseline_path = PROJECT_ROOT / "configs/experiments/phase2_revised_goal2_full_baseline_panel.yaml"
    return [
        AuditRow("phase2_revised", "revised six-goal plan exists", _exists(docs_path), _rel(docs_path)),
        AuditRow("phase2_revised", "common revised config exists", _exists(common_path), _rel(common_path)),
        AuditRow("phase2_revised_goal1_baseline_matched_validation", "baseline-matched validation config exists", _exists(goal1_path), _rel(goal1_path)),
        AuditRow("phase2_revised_goal2_weak_signal_tradeoff", "weak-signal tradeoff config exists", _exists(goal2_path), _rel(goal2_path)),
        AuditRow("phase2_revised_goal3_simulator_v2", "simulator v2 config exists", _exists(goal3_path), _rel(goal3_path)),
        AuditRow("phase2_revised_goal4_real_background_v2", "real-background v2 config exists", _exists(goal4_path), _rel(goal4_path)),
        AuditRow("phase2_revised_goal5_formal_fdr", "formal FDR config exists", _exists(goal5_path), _rel(goal5_path)),
        AuditRow("phase2_revised_goal6_manuscript_assembly", "manuscript assembly config exists", _exists(goal6_path), _rel(goal6_path)),
        AuditRow("phase2_revised_goal2_full_baseline_panel", "adapter-aware full baseline config exists", _exists(full_baseline_path), _rel(full_baseline_path)),
    ]


def _audit_goal1_baseline_matched_validation() -> list[AuditRow]:
    import pandas as pd

    root = PROJECT_ROOT / "outputs/phase2_revised/goal1_baseline_matched_validation"
    method_path = root / "goal1_baseline_matched_method_summary.csv"
    lr_path = root / "goal1_baseline_matched_lr_control_summary.csv"
    status_path = root / "goal1_baseline_matched_status.csv"
    claims_path = root / "goal1_baseline_matched_claims.csv"
    manifest_path = root / "goal1_baseline_matched_manifest.json"
    rows = [
        AuditRow("phase2_revised_goal1_baseline_matched_validation", "method summary exists", _exists(method_path), _rel(method_path)),
        AuditRow("phase2_revised_goal1_baseline_matched_validation", "LR control summary exists", _exists(lr_path), _rel(lr_path)),
        AuditRow("phase2_revised_goal1_baseline_matched_validation", "status table exists", _exists(status_path), _rel(status_path)),
        AuditRow("phase2_revised_goal1_baseline_matched_validation", "claim table exists", _exists(claims_path), _rel(claims_path)),
        AuditRow("phase2_revised_goal1_baseline_matched_validation", "manifest exists", _exists(manifest_path), _rel(manifest_path)),
    ]
    if not (method_path.exists() and lr_path.exists() and status_path.exists() and claims_path.exists() and manifest_path.exists()):
        return rows
    method = pd.read_csv(method_path)
    lr = pd.read_csv(lr_path)
    status = pd.read_csv(status_path)
    claims = pd.read_csv(claims_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_datasets = {
        "lymph_node_real_background_smoke",
        "tonsil_real_background_smoke",
        "melanoma_real_background_smoke",
    }
    required_methods = {"spatialself_wavelet_edge_prior", "spatialself_wavelet", "commot"}
    datasets = set(method["dataset"].astype(str))
    complete_status = status[status["status"].astype(str) == "completed"]
    rows.extend(
        [
            AuditRow(
                "phase2_revised_goal1_baseline_matched_validation",
                "validation covers lymph, tonsil, and melanoma injected-control backgrounds",
                "complete" if required_datasets.issubset(datasets) else "partial",
                _rel(method_path),
                f"missing={sorted(required_datasets - datasets)}",
            ),
            AuditRow(
                "phase2_revised_goal1_baseline_matched_validation",
                "validation includes SpatialSelf and COMMOT methods for every dataset",
                "complete"
                if all(
                    required_methods.issubset(set(method.loc[method["dataset"].astype(str) == dataset, "method"].astype(str)))
                    for dataset in required_datasets
                )
                else "partial",
                _rel(method_path),
                f"rows={len(method)}",
            ),
            AuditRow(
                "phase2_revised_goal1_baseline_matched_validation",
                "all baseline-matched smoke method runs completed",
                "complete" if len(complete_status) >= 9 and not (status["status"].astype(str) == "failed").any() else "partial",
                _rel(status_path),
                f"completed={len(complete_status)} failed={int((status['status'].astype(str) == 'failed').sum())}",
            ),
            AuditRow(
                "phase2_revised_goal1_baseline_matched_validation",
                "LR table contains positive and high-expression negative controls",
                "complete"
                if {"positive_control", "negative_control"}.issubset(lr.columns)
                and lr["positive_control"].astype(bool).any()
                and lr["negative_control"].astype(bool).any()
                else "partial",
                _rel(lr_path),
                f"rows={len(lr)}",
            ),
            AuditRow(
                "phase2_revised_goal1_baseline_matched_validation",
                "claim table remains injected-control proxy level",
                "complete"
                if set(claims["claim_level"].astype(str)) == {"injected_control_receiver_response_proxy_not_manuscript_ready"}
                and manifest.get("claim_level") == "injected_control_receiver_response_proxy_not_manuscript_ready"
                else "partial",
                _rel(claims_path),
                f"claim_levels={sorted(set(claims['claim_level'].astype(str))) if 'claim_level' in claims else []}; manifest={manifest.get('claim_level')}",
            ),
        ]
    )
    rows.append(_audit_goal1_spatialself_rank_fp(method_path, method))
    rows.extend(_audit_goal1_full_baseline_matched_validation())
    rows.extend(_audit_goal1_full_baseline_matched_validation_all())
    return rows


def _audit_goal1_full_baseline_matched_validation() -> list[AuditRow]:
    import pandas as pd

    root = PROJECT_ROOT / "outputs/phase2_revised/goal1_full_baseline_matched_validation"
    method_path = root / "goal1_baseline_matched_method_summary.csv"
    lr_path = root / "goal1_baseline_matched_lr_control_summary.csv"
    status_path = root / "goal1_baseline_matched_status.csv"
    claims_path = root / "goal1_baseline_matched_claims.csv"
    manifest_path = root / "goal1_baseline_matched_manifest.json"
    rows = [
        AuditRow("phase2_revised_goal1_full_baseline_matched_validation", "full-baseline matched method summary exists", _exists(method_path), _rel(method_path)),
        AuditRow("phase2_revised_goal1_full_baseline_matched_validation", "full-baseline matched LR control summary exists", _exists(lr_path), _rel(lr_path)),
        AuditRow("phase2_revised_goal1_full_baseline_matched_validation", "full-baseline matched status table exists", _exists(status_path), _rel(status_path)),
        AuditRow("phase2_revised_goal1_full_baseline_matched_validation", "full-baseline matched claims table exists", _exists(claims_path), _rel(claims_path)),
        AuditRow("phase2_revised_goal1_full_baseline_matched_validation", "full-baseline matched manifest exists", _exists(manifest_path), _rel(manifest_path)),
    ]
    if not all(path.exists() for path in [method_path, lr_path, status_path, claims_path, manifest_path]):
        return rows
    method = pd.read_csv(method_path)
    status = pd.read_csv(status_path)
    claims = pd.read_csv(claims_path)
    methods = set(method["method"].astype(str)) if "method" in method else set()
    required_metric_methods = {
        "spatialself_wavelet_edge_prior",
        "spatialself_wavelet",
        "commot",
        "cellchat",
        "spatalk",
        "cellnest",
    }
    spatialself = method[method["method"].astype(str).str.startswith("spatialself")]
    non_spatial = method[~method["method"].astype(str).str.startswith("spatialself")]
    rows.extend(
        [
            AuditRow(
                "phase2_revised_goal1_full_baseline_matched_validation",
                "tonsil full-baseline matched validation includes six metric methods",
                "complete" if required_metric_methods.issubset(methods) else "partial",
                _rel(method_path),
                f"methods={sorted(methods)}",
            ),
            AuditRow(
                "phase2_revised_goal1_full_baseline_matched_validation",
                "all six metric-method control validations completed",
                "complete"
                if len(status) >= 6 and (status["status"].astype(str) == "completed").sum() >= 6
                else "partial",
                _rel(status_path),
                f"completed={int((status['status'].astype(str) == 'completed').sum()) if 'status' in status else 0}",
            ),
            AuditRow(
                "phase2_revised_goal1_full_baseline_matched_validation",
                "SpatialSelf has lower high-expression FP than non-SpatialSelf baselines in tonsil full panel",
                "complete"
                if not spatialself.empty
                and not non_spatial.empty
                and float(spatialself["high_expression_false_positive_rate_at_q95"].min())
                < float(non_spatial["high_expression_false_positive_rate_at_q95"].min())
                else "partial",
                _rel(method_path),
                (
                    f"spatialself_min_fp={float(spatialself['high_expression_false_positive_rate_at_q95'].min()) if not spatialself.empty else 'NA'}; "
                    f"non_spatial_min_fp={float(non_spatial['high_expression_false_positive_rate_at_q95'].min()) if not non_spatial.empty else 'NA'}"
                ),
            ),
            AuditRow(
                "phase2_revised_goal1_full_baseline_matched_validation",
                "SpatialSelf keeps high-expression negative controls out of top3 in tonsil full panel",
                "complete"
                if not spatialself.empty and int(spatialself["negative_top3_count"].max()) == 0
                else "partial",
                _rel(method_path),
                f"spatialself_negative_top3_max={int(spatialself['negative_top3_count'].max()) if not spatialself.empty else 'NA'}",
            ),
            AuditRow(
                "phase2_revised_goal1_full_baseline_matched_validation",
                "claim table preserves receiver-response proxy scope",
                "complete"
                if len(claims) >= 1 and set(claims["claim_level"].astype(str)) == {"injected_control_receiver_response_proxy_not_manuscript_ready"}
                else "partial",
                _rel(claims_path),
                f"claim_levels={sorted(set(claims['claim_level'].astype(str))) if 'claim_level' in claims else []}",
            ),
        ]
    )
    return rows


def _audit_goal1_full_baseline_matched_validation_all() -> list[AuditRow]:
    import pandas as pd

    root = PROJECT_ROOT / "outputs/phase2_revised/goal1_full_baseline_matched_validation_all"
    script_path = PROJECT_ROOT / "scripts/phase2/merge_goal1_full_baseline_matched_validation.py"
    method_path = root / "goal1_baseline_matched_method_summary.csv"
    claims_path = root / "goal1_baseline_matched_claims.csv"
    manifest_path = root / "goal1_baseline_matched_manifest.json"
    rows = [
        AuditRow("phase2_revised_goal1_full_baseline_matched_validation_all", "three-dataset matched validation merge script exists", _exists(script_path), _rel(script_path)),
        AuditRow("phase2_revised_goal1_full_baseline_matched_validation_all", "three-dataset matched method summary exists", _exists(method_path), _rel(method_path)),
        AuditRow("phase2_revised_goal1_full_baseline_matched_validation_all", "three-dataset matched claims table exists", _exists(claims_path), _rel(claims_path)),
        AuditRow("phase2_revised_goal1_full_baseline_matched_validation_all", "three-dataset matched manifest exists", _exists(manifest_path), _rel(manifest_path)),
    ]
    if not all(path.exists() for path in [method_path, claims_path, manifest_path]):
        return rows
    method = pd.read_csv(method_path)
    claims = pd.read_csv(claims_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_datasets = {
        "lymph_node_real_background_smoke",
        "tonsil_real_background_smoke",
        "melanoma_real_background_smoke",
    }
    required_methods = {
        "spatialself_wavelet_edge_prior",
        "spatialself_wavelet",
        "commot",
        "cellchat",
        "spatalk",
        "cellnest",
    }
    datasets = set(method["dataset"].astype(str)) if "dataset" in method else set()
    methods = set(method["method"].astype(str)) if "method" in method else set()
    spatialself = method[method["method"].astype(str).str.startswith("spatialself")]
    rows.extend(
        [
            AuditRow(
                "phase2_revised_goal1_full_baseline_matched_validation_all",
                "merged matched validation covers lymph, tonsil, and melanoma",
                "complete" if required_datasets.issubset(datasets) else "partial",
                _rel(method_path),
                f"datasets={sorted(datasets)}",
            ),
            AuditRow(
                "phase2_revised_goal1_full_baseline_matched_validation_all",
                "merged matched validation includes six metric methods",
                "complete" if required_methods.issubset(methods) else "partial",
                _rel(method_path),
                f"methods={sorted(methods)}",
            ),
            AuditRow(
                "phase2_revised_goal1_full_baseline_matched_validation_all",
                "SpatialSelf keeps high-expression negative controls out of top3 in all three datasets",
                "complete"
                if not spatialself.empty
                and spatialself.groupby("dataset")["negative_top3_count"].max().le(0).all()
                and required_datasets.issubset(set(spatialself["dataset"].astype(str)))
                else "partial",
                _rel(method_path),
                spatialself.groupby("dataset")["negative_top3_count"].max().to_json() if not spatialself.empty else "",
            ),
            AuditRow(
                "phase2_revised_goal1_full_baseline_matched_validation_all",
                "SpatialSelf has lowest high-expression FP in each merged dataset",
                "complete"
                if len(claims) >= 3
                and required_datasets.issubset(set(claims["dataset"].astype(str)))
                and claims["spatialself_lowest_fp"].astype(bool).all()
                else "partial",
                _rel(claims_path),
                claims[["dataset", "lowest_high_expression_fp_method", "lowest_high_expression_fp_rate"]].to_json(orient="records") if len(claims) else "",
            ),
            AuditRow(
                "phase2_revised_goal1_full_baseline_matched_validation_all",
                "merged matched validation preserves proxy-level downstream caveat",
                "complete"
                if manifest.get("evidence_status") == "three_real_background_injected_control_proxy_not_final_curated_downstream"
                else "partial",
                _rel(manifest_path),
                f"evidence_status={manifest.get('evidence_status')}",
            ),
        ]
    )
    return rows


def _audit_goal1_spatialself_rank_fp(method_path: Path, method) -> AuditRow:
    comparisons = []
    for dataset, group in method.groupby("dataset", sort=False):
        commot = group[group["method"].astype(str) == "commot"]
        spatialself = group[group["method"].astype(str).str.startswith("spatialself")]
        if commot.empty or spatialself.empty:
            continue
        spatial_rank_pass = bool(spatialself["rank_validation_pass"].astype(bool).any())
        spatial_fp = float(spatialself["high_expression_false_positive_rate_at_q95"].min())
        commot_fp = float(commot["high_expression_false_positive_rate_at_q95"].min())
        comparisons.append(spatial_rank_pass and spatial_fp <= commot_fp)
    return AuditRow(
        "phase2_revised_goal1_baseline_matched_validation",
        "SpatialSelf passes rank validation and has lower/equal high-expression FP than COMMOT in smoke datasets",
        "complete" if comparisons and all(comparisons) else "partial",
        _rel(method_path),
        f"passes={sum(comparisons)}/{len(comparisons)}",
    )


def _audit_goal1_formal_baseline_matched_validation() -> list[AuditRow]:
    import pandas as pd

    root = PROJECT_ROOT / "outputs/phase2_revised/goal1_formal_baseline_matched_validation"
    config_path = PROJECT_ROOT / "configs/experiments/phase2_revised_goal1_formal_baseline_matched_validation.yaml"
    method_path = root / "goal1_baseline_matched_method_summary.csv"
    lr_path = root / "goal1_baseline_matched_lr_control_summary.csv"
    status_path = root / "goal1_baseline_matched_status.csv"
    claims_path = root / "goal1_baseline_matched_claims.csv"
    manifest_path = root / "goal1_baseline_matched_manifest.json"
    rows = [
        AuditRow("phase2_revised_goal1_formal_baseline_matched_validation", "formal Goal1 config exists", _exists(config_path), _rel(config_path)),
        AuditRow("phase2_revised_goal1_formal_baseline_matched_validation", "formal method summary exists", _exists(method_path), _rel(method_path)),
        AuditRow("phase2_revised_goal1_formal_baseline_matched_validation", "formal LR control summary exists", _exists(lr_path), _rel(lr_path)),
        AuditRow("phase2_revised_goal1_formal_baseline_matched_validation", "formal status table exists", _exists(status_path), _rel(status_path)),
        AuditRow("phase2_revised_goal1_formal_baseline_matched_validation", "formal claim table exists", _exists(claims_path), _rel(claims_path)),
        AuditRow("phase2_revised_goal1_formal_baseline_matched_validation", "formal manifest exists", _exists(manifest_path), _rel(manifest_path)),
    ]
    if not all(path.exists() for path in [method_path, lr_path, status_path, claims_path, manifest_path]):
        return rows

    method = pd.read_csv(method_path)
    lr = pd.read_csv(lr_path)
    status = pd.read_csv(status_path)
    claims = pd.read_csv(claims_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_datasets = {
        "real_background_v2_formal_lymph_node",
        "real_background_v2_formal_tonsil",
        "real_background_v2_formal_melanoma",
    }
    required_methods = {
        "spatialself_wavelet_edge_prior",
        "spatialself_wavelet",
        "commot",
        "cellchat",
        "spatalk",
        "cellnest",
    }
    datasets = set(method["dataset"].astype(str)) if "dataset" in method else set()
    methods = set(method["method"].astype(str)) if "method" in method else set()
    completed = status[status["status"].astype(str) == "completed"]
    spatialself = method[method["method"].astype(str).str.startswith("spatialself")]
    comparator_subset = method[method["method"].astype(str).isin(["commot", "cellchat", "cellnest"])]
    spatalk = method[method["method"].astype(str) == "spatalk"]

    spatialself_fp_passes = []
    spatalk_caveat_passes = []
    for dataset, group in method.groupby("dataset", sort=False):
        spatial_group = group[group["method"].astype(str).str.startswith("spatialself")]
        comparator_group = group[group["method"].astype(str).isin(["commot", "cellchat", "cellnest"])]
        spatalk_group = group[group["method"].astype(str) == "spatalk"]
        if not spatial_group.empty and not comparator_group.empty:
            spatialself_fp_passes.append(
                float(spatial_group["high_expression_false_positive_rate_at_q95"].min())
                < float(comparator_group["high_expression_false_positive_rate_at_q95"].min())
            )
        if not spatalk_group.empty:
            spatalk_caveat_passes.append(float(spatalk_group["high_expression_false_positive_rate_at_q95"].min()) == 0.0)

    rows.extend(
        [
            AuditRow(
                "phase2_revised_goal1_formal_baseline_matched_validation",
                "formal validation covers lymph node, tonsil, and melanoma real-background datasets",
                "complete" if required_datasets.issubset(datasets) else "partial",
                _rel(method_path),
                f"datasets={sorted(datasets)}",
            ),
            AuditRow(
                "phase2_revised_goal1_formal_baseline_matched_validation",
                "formal validation includes SpatialSelf, COMMOT, CellChat, SpaTalk, and CellNEST metric rows",
                "complete"
                if required_methods.issubset(methods)
                and all(
                    required_methods.issubset(set(method.loc[method["dataset"].astype(str) == dataset, "method"].astype(str)))
                    for dataset in required_datasets
                )
                else "partial",
                _rel(method_path),
                f"methods={sorted(methods)} rows={len(method)}",
            ),
            AuditRow(
                "phase2_revised_goal1_formal_baseline_matched_validation",
                "all formal baseline-matched method validations completed",
                "complete"
                if len(completed) >= 18 and not (status["status"].astype(str) == "failed").any()
                else "partial",
                _rel(status_path),
                f"completed={len(completed)} failed={int((status['status'].astype(str) == 'failed').sum())}",
            ),
            AuditRow(
                "phase2_revised_goal1_formal_baseline_matched_validation",
                "formal LR table contains positive and high-expression negative controls",
                "complete"
                if {"positive_control", "negative_control"}.issubset(lr.columns)
                and lr["positive_control"].astype(bool).any()
                and lr["negative_control"].astype(bool).any()
                else "partial",
                _rel(lr_path),
                f"rows={len(lr)} positives={int(lr['positive_control'].astype(bool).sum()) if 'positive_control' in lr else 'NA'} negatives={int(lr['negative_control'].astype(bool).sum()) if 'negative_control' in lr else 'NA'}",
            ),
            AuditRow(
                "phase2_revised_goal1_formal_baseline_matched_validation",
                "SpatialSelf keeps high-expression negative controls out of top3 in all formal datasets",
                "complete"
                if not spatialself.empty
                and spatialself.groupby("dataset")["negative_top3_count"].max().le(0).all()
                and required_datasets.issubset(set(spatialself["dataset"].astype(str)))
                else "partial",
                _rel(method_path),
                spatialself.groupby("dataset")["negative_top3_count"].max().to_json() if not spatialself.empty else "",
            ),
            AuditRow(
                "phase2_revised_goal1_formal_baseline_matched_validation",
                "SpatialSelf has lower high-expression FP than COMMOT, CellChat, and CellNEST in each formal dataset",
                "complete" if spatialself_fp_passes and all(spatialself_fp_passes) else "partial",
                _rel(method_path),
                f"passes={sum(spatialself_fp_passes)}/{len(spatialself_fp_passes)}; comparator_rows={len(comparator_subset)}",
            ),
            AuditRow(
                "phase2_revised_goal1_formal_baseline_matched_validation",
                "SpaTalk zero high-expression FP is retained as adapter-light caveat evidence rather than SpatialSelf-only overclaim",
                "complete" if spatalk_caveat_passes and all(spatalk_caveat_passes) else "partial",
                _rel(method_path),
                f"spatalk_fp={spatalk[['dataset', 'high_expression_false_positive_rate_at_q95']].to_json(orient='records') if len(spatalk) else []}",
            ),
            AuditRow(
                "phase2_revised_goal1_formal_baseline_matched_validation",
                "formal claim table and manifest preserve receiver-response proxy caveat",
                "complete"
                if len(claims) >= 3
                and set(claims["claim_level"].astype(str)) == {"injected_control_receiver_response_proxy_not_manuscript_ready"}
                and manifest.get("claim_level") == "formal_injected_control_receiver_response_proxy_not_manuscript_ready"
                else "partial",
                _rel(claims_path),
                f"claim_levels={sorted(set(claims['claim_level'].astype(str))) if 'claim_level' in claims else []}; manifest={manifest.get('claim_level')}",
            ),
        ]
    )
    return rows


def _audit_goal1_injected_downstream_proxy() -> list[AuditRow]:
    import pandas as pd

    root = PROJECT_ROOT / "outputs/phase2_revised/goal1_injected_downstream_proxy"
    script_path = PROJECT_ROOT / "scripts/phase2/run_goal1_injected_downstream_proxy.py"
    lr_path = root / "goal1_injected_downstream_proxy_lr_summary.csv"
    method_path = root / "goal1_injected_downstream_proxy_method_summary.csv"
    claims_path = root / "goal1_injected_downstream_proxy_claims.csv"
    status_path = root / "goal1_injected_downstream_proxy_status.csv"
    manifest_path = root / "goal1_injected_downstream_proxy_manifest.json"
    rows = [
        AuditRow("phase2_revised_goal1_injected_downstream_proxy", "injected downstream proxy runner exists", _exists(script_path), _rel(script_path)),
        AuditRow("phase2_revised_goal1_injected_downstream_proxy", "injected downstream proxy LR summary exists", _exists(lr_path), _rel(lr_path)),
        AuditRow("phase2_revised_goal1_injected_downstream_proxy", "injected downstream proxy method summary exists", _exists(method_path), _rel(method_path)),
        AuditRow("phase2_revised_goal1_injected_downstream_proxy", "injected downstream proxy claims exists", _exists(claims_path), _rel(claims_path)),
        AuditRow("phase2_revised_goal1_injected_downstream_proxy", "injected downstream proxy status exists", _exists(status_path), _rel(status_path)),
        AuditRow("phase2_revised_goal1_injected_downstream_proxy", "injected downstream proxy manifest exists", _exists(manifest_path), _rel(manifest_path)),
    ]
    if not all(path.exists() for path in [lr_path, method_path, claims_path, status_path, manifest_path]):
        return rows

    lr = pd.read_csv(lr_path)
    method = pd.read_csv(method_path)
    claims = pd.read_csv(claims_path)
    status = pd.read_csv(status_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_datasets = {
        "real_background_v2_formal_lymph_node",
        "real_background_v2_formal_tonsil",
        "real_background_v2_formal_melanoma",
    }
    required_methods = {
        "spatialself_wavelet_edge_prior",
        "spatialself_wavelet",
        "commot",
        "cellchat",
        "spatalk",
        "cellnest",
    }
    datasets = set(method["dataset"].astype(str)) if "dataset" in method else set()
    methods = set(method["method"].astype(str)) if "method" in method else set()
    rows.extend(
        [
            AuditRow(
                "phase2_revised_goal1_injected_downstream_proxy",
                "injected downstream proxy covers formal lymph node, tonsil, and melanoma",
                "complete" if required_datasets.issubset(datasets) else "partial",
                _rel(method_path),
                f"datasets={sorted(datasets)}",
            ),
            AuditRow(
                "phase2_revised_goal1_injected_downstream_proxy",
                "injected downstream proxy includes six metric methods",
                "complete"
                if required_methods.issubset(methods)
                and all(
                    required_methods.issubset(set(method.loc[method["dataset"].astype(str) == dataset, "method"].astype(str)))
                    for dataset in required_datasets
                )
                else "partial",
                _rel(method_path),
                f"methods={sorted(methods)} rows={len(method)}",
            ),
            AuditRow(
                "phase2_revised_goal1_injected_downstream_proxy",
                "injected downstream proxy LR rows contain positive and high-expression negative controls",
                "complete"
                if {"true_lr", "false_positive_high_expression", "response_proxy_spearman"}.issubset(lr.columns)
                and lr["true_lr"].astype(bool).any()
                and lr["false_positive_high_expression"].astype(bool).any()
                else "partial",
                _rel(lr_path),
                f"rows={len(lr)}",
            ),
            AuditRow(
                "phase2_revised_goal1_injected_downstream_proxy",
                "injected downstream proxy completed all method joins",
                "complete"
                if len(status) >= 18 and (status["status"].astype(str) == "completed").all()
                else "partial",
                _rel(status_path),
                f"rows={len(status)} statuses={sorted(set(status['status'].astype(str))) if 'status' in status else []}",
            ),
            AuditRow(
                "phase2_revised_goal1_injected_downstream_proxy",
                "injected downstream proxy records mixed evidence instead of SpatialSelf downstream overclaim",
                "complete"
                if len(claims) >= 3
                and required_datasets.issubset(set(claims["dataset"].astype(str)))
                and not claims["spatialself_best_response_delta"].astype(bool).any()
                and not claims["spatialself_beats_non_spatial_on_response_delta"].astype(bool).any()
                else "partial",
                _rel(claims_path),
                claims[["dataset", "best_response_delta_method", "spatialself_max_response_delta", "non_spatial_max_response_delta"]].to_json(orient="records") if len(claims) else "",
            ),
            AuditRow(
                "phase2_revised_goal1_injected_downstream_proxy",
                "injected downstream proxy manifest preserves non-curated downstream caveat",
                "complete"
                if manifest.get("claim_level") == "formal_injected_receiver_response_proxy_not_curated_downstream"
                else "partial",
                _rel(manifest_path),
                f"claim_level={manifest.get('claim_level')}",
            ),
        ]
    )
    return rows


def _audit_goal1_curated_downstream_panel() -> list[AuditRow]:
    import pandas as pd

    root = PROJECT_ROOT / "outputs/phase2_revised/goal1_curated_downstream_panel"
    script_path = PROJECT_ROOT / "scripts/phase2/run_goal1_curated_downstream_panel.py"
    results_path = root / "goal1_curated_downstream_results.csv"
    method_path = root / "goal1_curated_downstream_method_summary.csv"
    coverage_path = root / "goal1_curated_downstream_target_coverage.csv"
    claims_path = root / "goal1_curated_downstream_claims.csv"
    status_path = root / "goal1_curated_downstream_status.csv"
    manifest_path = root / "goal1_curated_downstream_manifest.json"
    rows = [
        AuditRow("phase2_revised_goal1_curated_downstream_panel", "curated downstream panel runner exists", _exists(script_path), _rel(script_path)),
        AuditRow("phase2_revised_goal1_curated_downstream_panel", "curated downstream results table exists", _exists(results_path), _rel(results_path)),
        AuditRow("phase2_revised_goal1_curated_downstream_panel", "curated downstream method summary exists", _exists(method_path), _rel(method_path)),
        AuditRow("phase2_revised_goal1_curated_downstream_panel", "curated downstream target coverage exists", _exists(coverage_path), _rel(coverage_path)),
        AuditRow("phase2_revised_goal1_curated_downstream_panel", "curated downstream claims exists", _exists(claims_path), _rel(claims_path)),
        AuditRow("phase2_revised_goal1_curated_downstream_panel", "curated downstream status exists", _exists(status_path), _rel(status_path)),
        AuditRow("phase2_revised_goal1_curated_downstream_panel", "curated downstream manifest exists", _exists(manifest_path), _rel(manifest_path)),
    ]
    if not all(path.exists() for path in [results_path, method_path, coverage_path, claims_path, status_path, manifest_path]):
        return rows

    results = pd.read_csv(results_path)
    method = pd.read_csv(method_path)
    coverage = pd.read_csv(coverage_path)
    claims = pd.read_csv(claims_path)
    status = pd.read_csv(status_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_datasets = {
        "real_background_v2_formal_lymph_node",
        "real_background_v2_formal_tonsil",
        "real_background_v2_formal_melanoma",
    }
    required_methods = {
        "spatialself_wavelet",
        "spatialself_wavelet_edge_prior",
        "commot",
        "cellchat",
        "spatalk",
        "cellnest",
    }
    datasets = set(method["dataset"].astype(str)) if "dataset" in method else set()
    methods = set(method["method"].astype(str)) if "method" in method else set()
    rows.extend(
        [
            AuditRow(
                "phase2_revised_goal1_curated_downstream_panel",
                "curated downstream panel covers formal lymph node, tonsil, and melanoma",
                "complete" if required_datasets.issubset(datasets) else "partial",
                _rel(method_path),
                f"datasets={sorted(datasets)}",
            ),
            AuditRow(
                "phase2_revised_goal1_curated_downstream_panel",
                "curated downstream panel includes six matched methods",
                "complete"
                if required_methods.issubset(methods)
                and all(
                    required_methods.issubset(set(method.loc[method["dataset"].astype(str) == dataset, "method"].astype(str)))
                    for dataset in required_datasets
                )
                else "partial",
                _rel(method_path),
                f"methods={sorted(methods)} rows={len(method)}",
            ),
            AuditRow(
                "phase2_revised_goal1_curated_downstream_panel",
                "curated downstream target coverage is explicitly quantified and limited for positives",
                "complete"
                if len(coverage) >= 3
                and required_datasets.issubset(set(coverage["dataset"].astype(str)))
                and (coverage["positive_target_coverage_fraction"].astype(float) < 1.0).any()
                and (coverage["negative_target_coverage_fraction"].astype(float) >= 1.0).all()
                else "partial",
                _rel(coverage_path),
                coverage[["dataset", "positive_target_coverage_fraction", "negative_target_coverage_fraction", "coverage_limitation"]].to_json(orient="records") if len(coverage) else "",
            ),
            AuditRow(
                "phase2_revised_goal1_curated_downstream_panel",
                "curated downstream panel completed all method joins",
                "complete"
                if len(status) >= 18 and (status["status"].astype(str) == "completed").all()
                else "partial",
                _rel(status_path),
                f"rows={len(status)} statuses={sorted(set(status['status'].astype(str))) if 'status' in status else []}",
            ),
            AuditRow(
                "phase2_revised_goal1_curated_downstream_panel",
                "curated downstream claims preserve mixed evidence and do not support SpatialSelf downstream overclaim",
                "complete"
                if len(claims) >= 3
                and required_datasets.issubset(set(claims["dataset"].astype(str)))
                and not claims["spatialself_best_delta"].astype(bool).any()
                and claims["coverage_limited"].astype(bool).all()
                else "partial",
                _rel(claims_path),
                claims[["dataset", "best_delta_method", "spatialself_max_delta", "non_spatial_max_delta", "coverage_limited"]].to_json(orient="records") if len(claims) else "",
            ),
            AuditRow(
                "phase2_revised_goal1_curated_downstream_panel",
                "curated downstream manifest preserves coverage-limited caveat",
                "complete"
                if manifest.get("claim_level") == "curated_downstream_coverage_limited_not_manuscript_ready"
                else "partial",
                _rel(manifest_path),
                f"claim_level={manifest.get('claim_level')}",
            ),
            AuditRow(
                "phase2_revised_goal1_curated_downstream_panel",
                "curated downstream result rows include tested and too-few-target statuses",
                "complete"
                if {"tested", "too_few_targets"}.issubset(set(results["status"].astype(str)))
                else "partial",
                _rel(results_path),
                f"statuses={sorted(set(results['status'].astype(str))) if 'status' in results else []}",
            ),
        ]
    )
    return rows


def _audit_goal1_biological_validation_readiness() -> list[AuditRow]:
    import pandas as pd

    root = PROJECT_ROOT / "outputs/phase2_revised/goal1_biological_validation_readiness"
    script_path = PROJECT_ROOT / "scripts/phase2/run_goal1_biological_validation_readiness.py"
    dataset_path = root / "goal1_biological_validation_dataset_gate.csv"
    method_path = root / "goal1_biological_validation_method_gate.csv"
    lr_path = root / "goal1_biological_validation_lr_gate.csv"
    claims_path = root / "goal1_biological_validation_claims.csv"
    manifest_path = root / "goal1_biological_validation_readiness_manifest.json"
    rows = [
        AuditRow("phase2_revised_goal1_biological_validation_readiness", "biological validation readiness runner exists", _exists(script_path), _rel(script_path)),
        AuditRow("phase2_revised_goal1_biological_validation_readiness", "dataset readiness gate exists", _exists(dataset_path), _rel(dataset_path)),
        AuditRow("phase2_revised_goal1_biological_validation_readiness", "method readiness gate exists", _exists(method_path), _rel(method_path)),
        AuditRow("phase2_revised_goal1_biological_validation_readiness", "LR readiness gate exists", _exists(lr_path), _rel(lr_path)),
        AuditRow("phase2_revised_goal1_biological_validation_readiness", "biological validation claims exists", _exists(claims_path), _rel(claims_path)),
        AuditRow("phase2_revised_goal1_biological_validation_readiness", "biological validation readiness manifest exists", _exists(manifest_path), _rel(manifest_path)),
    ]
    if not all(path.exists() for path in [dataset_path, method_path, lr_path, claims_path, manifest_path]):
        return rows

    dataset = pd.read_csv(dataset_path)
    method = pd.read_csv(method_path)
    lr = pd.read_csv(lr_path)
    claims = pd.read_csv(claims_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_datasets = {
        "real_background_v2_formal_lymph_node",
        "real_background_v2_formal_tonsil",
        "real_background_v2_formal_melanoma",
    }
    rows.extend(
        [
            AuditRow(
                "phase2_revised_goal1_biological_validation_readiness",
                "readiness gate covers formal lymph node, tonsil, and melanoma",
                "complete" if required_datasets.issubset(set(dataset["dataset"].astype(str))) else "partial",
                _rel(dataset_path),
                f"datasets={sorted(set(dataset['dataset'].astype(str))) if 'dataset' in dataset else []}",
            ),
            AuditRow(
                "phase2_revised_goal1_biological_validation_readiness",
                "readiness gate marks all datasets main-text ready for control ranking, FP, and FDR with downstream limitation",
                "complete"
                if len(dataset) >= 3
                and set(dataset["biological_validation_status"].astype(str))
                == {"main_text_ready_control_ranking_fp_fdr_downstream_limited"}
                and dataset["spatialself_rank_validation_pass"].astype(bool).all()
                and dataset["spatialself_absolute_low_fp_pass"].astype(bool).all()
                and dataset["spatialself_fp_better_than_commot_cellchat_cellnest"].astype(bool).all()
                and not dataset["spatialself_lowest_among_all_baselines"].astype(bool).any()
                else "partial",
                _rel(dataset_path),
                dataset[
                    [
                        "dataset",
                        "biological_validation_status",
                        "spatialself_absolute_low_fp_pass",
                        "spatialself_fp_better_than_commot_cellchat_cellnest",
                        "spatialself_lowest_among_all_baselines",
                    ]
                ].to_json(orient="records") if len(dataset) else "",
            ),
            AuditRow(
                "phase2_revised_goal1_biological_validation_readiness",
                "readiness gate preserves curated downstream limitation",
                "complete"
                if len(dataset) >= 3
                and not dataset["curated_downstream_superiority_supported"].astype(bool).any()
                and not dataset["curated_downstream_coverage_pass"].astype(bool).any()
                and set(claims.loc[claims["claim_id"].astype(str) == "G1_C3_curated_downstream", "status"].astype(str))
                == {"unsupported"}
                else "partial",
                _rel(claims_path),
                claims[["claim_id", "status", "allowed_wording", "disallowed_wording"]].to_json(orient="records") if len(claims) else "",
            ),
            AuditRow(
                "phase2_revised_goal1_biological_validation_readiness",
                "readiness gate includes method-level and LR-level supporting rows",
                "complete"
                if len(method) >= 18
                and len(lr) >= 15
                and {"supported_for_control_ranking", "supported_for_fp_suppression"}.issubset(set(method.columns))
                else "partial",
                _rel(method_path),
                f"method_rows={len(method)} lr_rows={len(lr)}",
            ),
            AuditRow(
                "phase2_revised_goal1_biological_validation_readiness",
                "readiness manifest preserves non-final downstream superiority caveat",
                "complete"
                if manifest.get("claim_level") == "goal1_readiness_gate_not_final_downstream_superiority"
                else "partial",
                _rel(manifest_path),
                f"claim_level={manifest.get('claim_level')}",
            ),
        ]
    )
    return rows


def _audit_goal2_weak_signal_tradeoff() -> list[AuditRow]:
    import pandas as pd

    root = PROJECT_ROOT / "outputs/phase2_revised/goal2_weak_signal_tradeoff"
    metrics_path = root / "goal2_weak_signal_tradeoff_metrics.csv"
    status_path = root / "goal2_weak_signal_tradeoff_status.csv"
    claims_path = root / "goal2_weak_signal_tradeoff_claims.csv"
    manifest_path = root / "goal2_weak_signal_tradeoff_manifest.json"
    rows = [
        AuditRow("phase2_revised_goal2_weak_signal_tradeoff", "metrics table exists", _exists(metrics_path), _rel(metrics_path)),
        AuditRow("phase2_revised_goal2_weak_signal_tradeoff", "status table exists", _exists(status_path), _rel(status_path)),
        AuditRow("phase2_revised_goal2_weak_signal_tradeoff", "claim table exists", _exists(claims_path), _rel(claims_path)),
        AuditRow("phase2_revised_goal2_weak_signal_tradeoff", "manifest exists", _exists(manifest_path), _rel(manifest_path)),
    ]
    if not (metrics_path.exists() and status_path.exists() and claims_path.exists()):
        return rows

    metrics = pd.read_csv(metrics_path)
    status = pd.read_csv(status_path)
    claims = pd.read_csv(claims_path)
    required_datasets = {
        "lymph_node_real_background_smoke",
        "tonsil_real_background_smoke",
        "melanoma_real_background_smoke",
    }
    required_methods = {"spatialself_wavelet_edge_prior", "spatialself_wavelet", "commot"}
    datasets = set(metrics["dataset"].astype(str))
    methods_by_dataset = {
        dataset: set(metrics.loc[metrics["dataset"].astype(str) == dataset, "method"].astype(str))
        for dataset in datasets
    }
    complete_status = status[status["status"].astype(str) == "completed"]
    rows.extend(
        [
            AuditRow(
                "phase2_revised_goal2_weak_signal_tradeoff",
                "tradeoff metrics cover lymph, tonsil, and melanoma smoke datasets",
                "complete" if required_datasets.issubset(datasets) else "partial",
                _rel(metrics_path),
                f"missing={sorted(required_datasets - datasets)}",
            ),
            AuditRow(
                "phase2_revised_goal2_weak_signal_tradeoff",
                "each smoke dataset has SpatialSelf and COMMOT comparators",
                "complete"
                if all(required_methods.issubset(methods_by_dataset.get(dataset, set())) for dataset in required_datasets)
                else "partial",
                _rel(metrics_path),
                json.dumps({k: sorted(v) for k, v in methods_by_dataset.items()}, ensure_ascii=True),
            ),
            AuditRow(
                "phase2_revised_goal2_weak_signal_tradeoff",
                "all smoke method runs completed",
                "complete" if len(complete_status) >= 9 and not (status["status"].astype(str) == "failed").any() else "partial",
                _rel(status_path),
                f"completed={len(complete_status)} failed={int((status['status'].astype(str) == 'failed').sum())}",
            ),
            AuditRow(
                "phase2_revised_goal2_weak_signal_tradeoff",
                "claim table remains smoke-level and does not overclaim manuscript readiness",
                "complete"
                if len(claims) >= 3
                and set(claims["claim_level"].astype(str)) == {"smoke_tradeoff_evidence_not_manuscript_ready"}
                else "partial",
                _rel(claims_path),
                f"claim_levels={sorted(set(claims['claim_level'].astype(str))) if 'claim_level' in claims else []}",
            ),
        ]
    )
    rows.append(_audit_goal2_spatialself_fp_advantage(metrics_path, metrics))
    rows.extend(_audit_goal2_full_baseline_panel())
    rows.extend(_audit_goal2_full_baseline_panel_all())
    return rows


def _audit_goal2_full_baseline_panel() -> list[AuditRow]:
    import pandas as pd

    root = PROJECT_ROOT / "outputs/phase2_revised/goal2_full_baseline_panel"
    script_path = PROJECT_ROOT / "scripts/phase2/run_goal2_full_baseline_panel.py"
    metrics_path = root / "goal2_full_baseline_metrics.csv"
    summary_path = root / "goal2_full_baseline_summary.csv"
    status_path = root / "goal2_full_baseline_status.csv"
    claims_path = root / "goal2_full_baseline_claims.csv"
    manifest_path = root / "goal2_full_baseline_manifest.json"
    rows = [
        AuditRow("phase2_revised_goal2_full_baseline_panel", "full baseline panel runner exists", _exists(script_path), _rel(script_path)),
        AuditRow("phase2_revised_goal2_full_baseline_panel", "full baseline metrics table exists", _exists(metrics_path), _rel(metrics_path)),
        AuditRow("phase2_revised_goal2_full_baseline_panel", "full baseline summary table exists", _exists(summary_path), _rel(summary_path)),
        AuditRow("phase2_revised_goal2_full_baseline_panel", "full baseline status table exists", _exists(status_path), _rel(status_path)),
        AuditRow("phase2_revised_goal2_full_baseline_panel", "full baseline claims table exists", _exists(claims_path), _rel(claims_path)),
        AuditRow("phase2_revised_goal2_full_baseline_panel", "full baseline manifest exists", _exists(manifest_path), _rel(manifest_path)),
    ]
    if not all(path.exists() for path in [metrics_path, summary_path, status_path, claims_path, manifest_path]):
        return rows

    metrics = pd.read_csv(metrics_path)
    status = pd.read_csv(status_path)
    claims = pd.read_csv(claims_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status_methods = set(status["method"].astype(str)) if "method" in status else set()
    metric_methods = set(metrics["method"].astype(str)) if "method" in metrics else set()
    rows.extend(
        [
            AuditRow(
                "phase2_revised_goal2_full_baseline_panel",
                "adapter-aware status includes SpatialSelf, COMMOT, CellChat, SpaTalk, CellNEST, DeepTalk, and NICHES",
                "complete"
                if {
                    "spatialself_wavelet_edge_prior",
                    "spatialself_wavelet",
                    "commot",
                    "cellchat",
                    "spatalk",
                    "cellnest",
                    "deeptalk",
                    "niches",
                }.issubset(status_methods)
                else "partial",
                _rel(status_path),
                f"methods={sorted(status_methods)}",
            ),
            AuditRow(
                "phase2_revised_goal2_full_baseline_panel",
                "full baseline metrics include SpatialSelf, COMMOT, CellChat, SpaTalk, and CellNEST",
                "complete"
                if {
                    "spatialself_wavelet_edge_prior",
                    "spatialself_wavelet",
                    "commot",
                    "cellchat",
                    "spatalk",
                    "cellnest",
                }.issubset(metric_methods)
                else "partial",
                _rel(metrics_path),
                f"metric_methods={sorted(metric_methods)}",
            ),
            AuditRow(
                "phase2_revised_goal2_full_baseline_panel",
                "CellChat, SpaTalk, and CellNEST produce adapter-light metric rows",
                "complete"
                if {"cellchat", "spatalk", "cellnest"}.issubset(metric_methods)
                and not metrics.loc[metrics["method"].astype(str).isin(["cellchat", "spatalk", "cellnest"]), "aupr"].isna().any()
                else "partial",
                _rel(metrics_path),
                f"adapter_metric_methods={sorted(metric_methods & {'cellchat', 'spatalk', 'cellnest'})}",
            ),
            AuditRow(
                "phase2_revised_goal2_full_baseline_panel",
                "DeepTalk model-required status is recorded separately from metric-producing baselines",
                "complete"
                if "deeptalk" in status_methods
                and status.loc[status["method"].astype(str) == "deeptalk", "status"].astype(str).str.contains("model_required").any()
                else "partial",
                _rel(status_path),
                status.loc[status["method"].astype(str) == "deeptalk", ["status", "detail"]].to_json(orient="records") if "method" in status else "",
            ),
            AuditRow(
                "phase2_revised_goal2_full_baseline_panel",
                "NICHES dependency blocker is recorded explicitly",
                "complete"
                if "niches" in status_methods
                and (status.loc[status["method"].astype(str) == "niches", "status"].astype(str) == "optional_blocked").any()
                else "partial",
                _rel(status_path),
                status.loc[status["method"].astype(str) == "niches", ["status", "detail"]].to_json(orient="records") if "method" in status else "",
            ),
            AuditRow(
                "phase2_revised_goal2_full_baseline_panel",
                "full baseline claim table preserves adapter-aware smoke scope",
                "complete"
                if len(claims) >= 1 and set(claims["claim_level"].astype(str)) == {"adapter_aware_smoke_not_final_tuned_baseline"}
                else "partial",
                _rel(claims_path),
                f"claim_levels={sorted(set(claims['claim_level'].astype(str))) if 'claim_level' in claims else []}",
            ),
            AuditRow(
                "phase2_revised_goal2_full_baseline_panel",
                "full baseline smoke supports SpatialSelf tradeoff rather than highest weak recall claim",
                "complete"
                if len(claims) >= 1
                and str(claims.iloc[0].get("best_tradeoff_method", "")).startswith("spatialself")
                and str(claims.iloc[0].get("highest_weak_recall_method", "")) in {"spatalk", "cellnest"}
                else "partial",
                _rel(claims_path),
                claims.to_json(orient="records"),
            ),
            AuditRow(
                "phase2_revised_goal2_full_baseline_panel",
                "manifest records non-final tuned-baseline status",
                "complete"
                if manifest.get("status_scope") == "adapter-aware smoke/status panel; not final tuned baseline comparison"
                else "partial",
                _rel(manifest_path),
                f"status_scope={manifest.get('status_scope')}",
            ),
        ]
    )
    return rows


def _audit_goal2_full_baseline_panel_all() -> list[AuditRow]:
    import pandas as pd

    root = PROJECT_ROOT / "outputs/phase2_revised/goal2_full_baseline_panel_all"
    script_path = PROJECT_ROOT / "scripts/phase2/merge_goal2_full_baseline_panels.py"
    metrics_path = root / "goal2_full_baseline_metrics.csv"
    status_path = root / "goal2_full_baseline_status.csv"
    claims_path = root / "goal2_full_baseline_claims.csv"
    manifest_path = root / "goal2_full_baseline_manifest.json"
    rows = [
        AuditRow("phase2_revised_goal2_full_baseline_panel_all", "three-dataset full baseline merge script exists", _exists(script_path), _rel(script_path)),
        AuditRow("phase2_revised_goal2_full_baseline_panel_all", "three-dataset full baseline metrics table exists", _exists(metrics_path), _rel(metrics_path)),
        AuditRow("phase2_revised_goal2_full_baseline_panel_all", "three-dataset full baseline status table exists", _exists(status_path), _rel(status_path)),
        AuditRow("phase2_revised_goal2_full_baseline_panel_all", "three-dataset full baseline claims table exists", _exists(claims_path), _rel(claims_path)),
        AuditRow("phase2_revised_goal2_full_baseline_panel_all", "three-dataset full baseline manifest exists", _exists(manifest_path), _rel(manifest_path)),
    ]
    if not all(path.exists() for path in [metrics_path, status_path, claims_path, manifest_path]):
        return rows
    metrics = pd.read_csv(metrics_path)
    status = pd.read_csv(status_path)
    claims = pd.read_csv(claims_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_datasets = {
        "lymph_node_real_background_smoke",
        "tonsil_real_background_smoke",
        "melanoma_real_background_smoke",
    }
    required_metric_methods = {
        "spatialself_wavelet_edge_prior",
        "spatialself_wavelet",
        "commot",
        "cellchat",
        "spatalk",
        "cellnest",
    }
    datasets = set(metrics["dataset"].astype(str)) if "dataset" in metrics else set()
    metric_methods = set(metrics["method"].astype(str)) if "method" in metrics else set()
    status_methods = set(status["method"].astype(str)) if "method" in status else set()
    rows.extend(
        [
            AuditRow(
                "phase2_revised_goal2_full_baseline_panel_all",
                "merged full baseline covers lymph, tonsil, and melanoma",
                "complete" if required_datasets.issubset(datasets) else "partial",
                _rel(metrics_path),
                f"datasets={sorted(datasets)}",
            ),
            AuditRow(
                "phase2_revised_goal2_full_baseline_panel_all",
                "merged full baseline includes six metric methods",
                "complete" if required_metric_methods.issubset(metric_methods) else "partial",
                _rel(metrics_path),
                f"metric_methods={sorted(metric_methods)}",
            ),
            AuditRow(
                "phase2_revised_goal2_full_baseline_panel_all",
                "merged full baseline records DeepTalk and NICHES statuses",
                "complete" if {"deeptalk", "niches"}.issubset(status_methods) else "partial",
                _rel(status_path),
                f"status_methods={sorted(status_methods)}",
            ),
            AuditRow(
                "phase2_revised_goal2_full_baseline_panel_all",
                "SpatialSelf is best FP/FDR-constrained tradeoff and lowest high-expression FP in each merged dataset",
                "complete"
                if len(claims) >= 3
                and required_datasets.issubset(set(claims["dataset"].astype(str)))
                and claims["spatialself_best_tradeoff"].astype(bool).all()
                and claims["spatialself_lowest_fp"].astype(bool).all()
                and set(claims["best_tradeoff_status"].astype(str)) == {"fp_fdr_guardrail_met"}
                else "partial",
                _rel(claims_path),
                claims[
                    [
                        "dataset",
                        "best_tradeoff_method",
                        "best_tradeoff_status",
                        "best_unconstrained_tradeoff_method",
                        "lowest_high_expression_fp_method",
                        "highest_weak_recall_method",
                    ]
                ].to_json(orient="records") if len(claims) else "",
            ),
            AuditRow(
                "phase2_revised_goal2_full_baseline_panel_all",
                "merged full baseline preserves adapter-aware smoke scope",
                "complete"
                if manifest.get("evidence_status") == "three_real_background_adapter_aware_smoke_not_final_tuned_baseline"
                else "partial",
                _rel(manifest_path),
                f"evidence_status={manifest.get('evidence_status')}",
            ),
        ]
    )
    return rows


def _audit_goal2_formal_baseline_panel() -> list[AuditRow]:
    import pandas as pd

    root = PROJECT_ROOT / "outputs/phase2_revised/goal2_formal_baseline_panel_all"
    config_path = PROJECT_ROOT / "configs/experiments/phase2_revised_goal2_formal_baseline_panel.yaml"
    merge_script_path = PROJECT_ROOT / "scripts/phase2/merge_goal2_formal_baseline_panels.py"
    metrics_path = root / "goal2_formal_baseline_metrics.csv"
    summary_path = root / "goal2_formal_baseline_summary.csv"
    status_path = root / "goal2_formal_baseline_status.csv"
    claims_path = root / "goal2_formal_baseline_claims.csv"
    manifest_path = root / "goal2_formal_baseline_manifest.json"
    rows = [
        AuditRow("phase2_revised_goal2_formal_baseline_panel", "formal baseline panel config exists", _exists(config_path), _rel(config_path)),
        AuditRow("phase2_revised_goal2_formal_baseline_panel", "formal baseline panel merge script exists", _exists(merge_script_path), _rel(merge_script_path)),
        AuditRow("phase2_revised_goal2_formal_baseline_panel", "formal baseline metrics table exists", _exists(metrics_path), _rel(metrics_path)),
        AuditRow("phase2_revised_goal2_formal_baseline_panel", "formal baseline summary table exists", _exists(summary_path), _rel(summary_path)),
        AuditRow("phase2_revised_goal2_formal_baseline_panel", "formal baseline status table exists", _exists(status_path), _rel(status_path)),
        AuditRow("phase2_revised_goal2_formal_baseline_panel", "formal baseline claims table exists", _exists(claims_path), _rel(claims_path)),
        AuditRow("phase2_revised_goal2_formal_baseline_panel", "formal baseline manifest exists", _exists(manifest_path), _rel(manifest_path)),
    ]
    if not all(path.exists() for path in [metrics_path, summary_path, status_path, claims_path, manifest_path]):
        return rows
    metrics = pd.read_csv(metrics_path)
    status = pd.read_csv(status_path)
    claims = pd.read_csv(claims_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_datasets = {"controlled_v2_formal", "real_background_v2_formal"}
    required_metric_methods = {
        "spatialself_wavelet_edge_prior",
        "spatialself_wavelet",
        "commot",
        "cellchat",
        "spatalk",
        "cellnest",
    }
    datasets = set(metrics["dataset"].astype(str)) if "dataset" in metrics else set()
    metric_methods = set(metrics["method"].astype(str)) if "method" in metrics else set()
    status_methods = set(status["method"].astype(str)) if "method" in status else set()
    spatial = metrics[metrics["method"].astype(str) == "spatialself_wavelet_edge_prior"]
    commot = metrics[metrics["method"].astype(str) == "commot"]
    cellchat = metrics[metrics["method"].astype(str) == "cellchat"]
    spatial_fp = float(spatial["false_positive_high_expression_rate_at_threshold"].mean()) if len(spatial) else float("nan")
    commot_fp = float(commot["false_positive_high_expression_rate_at_threshold"].mean()) if len(commot) else float("nan")
    cellchat_fp = float(cellchat["false_positive_high_expression_rate_at_threshold"].mean()) if len(cellchat) else float("nan")
    rows.extend(
        [
            AuditRow(
                "phase2_revised_goal2_formal_baseline_panel",
                "formal baseline panel covers formal controlled and real-background datasets",
                "complete" if required_datasets.issubset(datasets) else "partial",
                _rel(metrics_path),
                f"datasets={sorted(datasets)}",
            ),
            AuditRow(
                "phase2_revised_goal2_formal_baseline_panel",
                "formal baseline panel includes SpatialSelf, COMMOT, CellChat, SpaTalk, and CellNEST metric rows",
                "complete"
                if required_metric_methods.issubset(metric_methods)
                and all(
                    required_metric_methods.issubset(set(metrics.loc[metrics["dataset"].astype(str) == dataset, "method"].astype(str)))
                    for dataset in required_datasets
                )
                else "partial",
                _rel(metrics_path),
                f"metric_methods={sorted(metric_methods)}",
            ),
            AuditRow(
                "phase2_revised_goal2_formal_baseline_panel",
                "formal baseline panel records DeepTalk and NICHES statuses",
                "complete" if {"deeptalk", "niches"}.issubset(status_methods) else "partial",
                _rel(status_path),
                f"status_methods={sorted(status_methods)}",
            ),
            AuditRow(
                "phase2_revised_goal2_formal_baseline_panel",
                "SpatialSelf has lower mean high-expression FP than COMMOT and CellChat in formal panel",
                "complete" if spatial_fp <= commot_fp and spatial_fp <= cellchat_fp else "partial",
                _rel(metrics_path),
                f"spatial_fp={spatial_fp}; commot_fp={commot_fp}; cellchat_fp={cellchat_fp}",
            ),
            AuditRow(
                "phase2_revised_goal2_formal_baseline_panel",
                "SpatialSelf is best FP/FDR-constrained tradeoff in both formal benchmark groups",
                "complete"
                if required_datasets.issubset(set(claims["dataset"].astype(str)))
                and claims["spatialself_best_tradeoff"].astype(bool).all()
                and set(claims["best_tradeoff_status"].astype(str)) == {"fp_fdr_guardrail_met"}
                else "partial",
                _rel(claims_path),
                claims.to_json(orient="records") if len(claims) else "",
            ),
            AuditRow(
                "phase2_revised_goal2_formal_baseline_panel",
                "formal baseline claim table is initial adapter-aware and not final tuned comparison",
                "complete"
                if len(claims) >= 2
                and set(claims["claim_level"].astype(str)) == {"formal_adapter_aware_not_final_tuned_baseline"}
                and manifest.get("evidence_status") == "formal_adapter_aware_panel_not_final_tuned_baseline"
                else "partial",
                _rel(claims_path),
                f"claim_levels={sorted(set(claims['claim_level'].astype(str))) if 'claim_level' in claims else []}; evidence_status={manifest.get('evidence_status')}",
            ),
        ]
    )
    return rows


def _audit_goal2_formal_tuned_small_panel() -> list[AuditRow]:
    import pandas as pd

    root = PROJECT_ROOT / "outputs/phase2_revised/goal2_formal_tuned_small_panel"
    config_path = PROJECT_ROOT / "configs/experiments/phase2_revised_goal2_formal_tuned_small_panel.yaml"
    script_path = PROJECT_ROOT / "scripts/phase2/run_goal2_formal_tuned_small_panel.py"
    metrics_path = root / "goal2_formal_tuned_small_metrics.csv"
    summary_path = root / "goal2_formal_tuned_small_summary.csv"
    status_path = root / "goal2_formal_tuned_small_status.csv"
    claims_path = root / "goal2_formal_tuned_small_claims.csv"
    params_path = root / "goal2_formal_tuned_small_parameters.csv"
    manifest_path = root / "goal2_formal_tuned_small_manifest.json"
    rows = [
        AuditRow("phase2_revised_goal2_formal_tuned_small_panel", "formal tuned-small config exists", _exists(config_path), _rel(config_path)),
        AuditRow("phase2_revised_goal2_formal_tuned_small_panel", "formal tuned-small runner exists", _exists(script_path), _rel(script_path)),
        AuditRow("phase2_revised_goal2_formal_tuned_small_panel", "formal tuned-small metrics table exists", _exists(metrics_path), _rel(metrics_path)),
        AuditRow("phase2_revised_goal2_formal_tuned_small_panel", "formal tuned-small summary table exists", _exists(summary_path), _rel(summary_path)),
        AuditRow("phase2_revised_goal2_formal_tuned_small_panel", "formal tuned-small status table exists", _exists(status_path), _rel(status_path)),
        AuditRow("phase2_revised_goal2_formal_tuned_small_panel", "formal tuned-small claims table exists", _exists(claims_path), _rel(claims_path)),
        AuditRow("phase2_revised_goal2_formal_tuned_small_panel", "formal tuned-small parameter table exists", _exists(params_path), _rel(params_path)),
        AuditRow("phase2_revised_goal2_formal_tuned_small_panel", "formal tuned-small manifest exists", _exists(manifest_path), _rel(manifest_path)),
    ]
    if not all(path.exists() for path in [metrics_path, summary_path, status_path, claims_path, params_path, manifest_path]):
        return rows

    metrics = pd.read_csv(metrics_path)
    status = pd.read_csv(status_path)
    claims = pd.read_csv(claims_path)
    params = pd.read_csv(params_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_replicates = {
        "human_lymph_node_visium_replicate_00",
        "human_tonsil_slide_tags_replicate_00",
        "human_melanoma_multiome_replicate_00",
    }
    required_metric_methods = {
        "spatialself_wavelet_edge_prior",
        "spatialself_wavelet",
        "commot",
        "cellchat",
        "spatalk",
        "cellnest",
    }
    metric_methods = set(metrics["method"].astype(str)) if "method" in metrics else set()
    status_methods = set(status["method"].astype(str)) if "method" in status else set()
    replicate_methods_complete = all(
        required_metric_methods.issubset(set(metrics.loc[metrics["replicate"].astype(str) == replicate, "method"].astype(str)))
        for replicate in required_replicates
    )
    param_lookup = {
        (str(row.method), str(row.parameter)): str(row.value)
        for row in params.itertuples(index=False)
        if hasattr(row, "method") and hasattr(row, "parameter") and hasattr(row, "value")
    }
    tuned_params_ok = (
        param_lookup.get(("commot", "cot_nitermax")) == "500"
        and param_lookup.get(("cellchat", "nboot")) == "20"
        and param_lookup.get(("spatalk", "per_num")) == "100"
        and param_lookup.get(("cellnest", "num_epoch")) == "2"
        and param_lookup.get(("cellnest", "hidden")) == "32"
    )
    aggregate = claims[claims["scope"].astype(str) == "aggregate_real_background_v2_formal"]
    tissue_claims = claims[claims["scope"].astype(str) != "aggregate_real_background_v2_formal"]
    rows.extend(
        [
            AuditRow(
                "phase2_revised_goal2_formal_tuned_small_panel",
                "formal tuned-small panel covers lymph, tonsil, and melanoma formal real-background replicates",
                "complete" if required_replicates.issubset(set(metrics["replicate"].astype(str))) else "partial",
                _rel(metrics_path),
                f"replicates={sorted(set(metrics['replicate'].astype(str))) if 'replicate' in metrics else []}",
            ),
            AuditRow(
                "phase2_revised_goal2_formal_tuned_small_panel",
                "formal tuned-small panel includes six metric methods for every replicate",
                "complete" if required_metric_methods.issubset(metric_methods) and replicate_methods_complete else "partial",
                _rel(metrics_path),
                f"metric_methods={sorted(metric_methods)}",
            ),
            AuditRow(
                "phase2_revised_goal2_formal_tuned_small_panel",
                "formal tuned-small panel records DeepTalk and NICHES statuses",
                "complete"
                if {"deeptalk", "niches"}.issubset(status_methods)
                and status.loc[status["method"].astype(str) == "deeptalk", "status"].astype(str).str.contains("model_required").any()
                and (status.loc[status["method"].astype(str) == "niches", "status"].astype(str) == "optional_blocked").any()
                else "partial",
                _rel(status_path),
                f"status_methods={sorted(status_methods)}",
            ),
            AuditRow(
                "phase2_revised_goal2_formal_tuned_small_panel",
                "formal tuned-small parameter table records non-adapter-light settings",
                "complete" if tuned_params_ok else "partial",
                _rel(params_path),
                json.dumps({f"{method}.{parameter}": value for (method, parameter), value in param_lookup.items()}, ensure_ascii=True, sort_keys=True),
            ),
            AuditRow(
                "phase2_revised_goal2_formal_tuned_small_panel",
                "formal tuned-small aggregate supports SpatialSelf tradeoff but tissue-level claims remain mixed",
                "complete"
                if len(aggregate) == 1
                and bool(aggregate.iloc[0]["spatialself_best_tradeoff"])
                and len(tissue_claims) >= 3
                and not tissue_claims["spatialself_best_tradeoff"].astype(bool).all()
                else "partial",
                _rel(claims_path),
                claims[["scope", "best_tradeoff_method", "highest_weak_recall_method", "spatialself_best_tradeoff"]].to_json(orient="records") if len(claims) else "",
            ),
            AuditRow(
                "phase2_revised_goal2_formal_tuned_small_panel",
                "formal tuned-small manifest preserves non-final replicated baseline scope",
                "complete"
                if manifest.get("evidence_status") == "formal_tuned_small_panel_not_final_replicated_baseline"
                and manifest.get("status_scope") == "tuned small real-background panel; not final replicated manuscript baseline"
                else "partial",
                _rel(manifest_path),
                f"evidence_status={manifest.get('evidence_status')}; status_scope={manifest.get('status_scope')}",
            ),
        ]
    )
    return rows


def _audit_goal2_tuned_guardrail_sensitivity() -> list[AuditRow]:
    import pandas as pd

    root = PROJECT_ROOT / "outputs/phase2_revised/goal2_tuned_guardrail_sensitivity"
    script_path = PROJECT_ROOT / "scripts/phase2/run_goal2_tuned_guardrail_sensitivity.py"
    sensitivity_path = root / "goal2_tuned_guardrail_sensitivity.csv"
    summary_path = root / "goal2_tuned_guardrail_sensitivity_summary.csv"
    claims_path = root / "goal2_tuned_guardrail_sensitivity_claims.csv"
    status_path = root / "goal2_tuned_guardrail_sensitivity_status.csv"
    manifest_path = root / "goal2_tuned_guardrail_sensitivity_manifest.json"
    rows = [
        AuditRow("phase2_revised_goal2_tuned_guardrail_sensitivity", "guardrail sensitivity runner exists", _exists(script_path), _rel(script_path)),
        AuditRow("phase2_revised_goal2_tuned_guardrail_sensitivity", "guardrail sensitivity table exists", _exists(sensitivity_path), _rel(sensitivity_path)),
        AuditRow("phase2_revised_goal2_tuned_guardrail_sensitivity", "guardrail sensitivity summary exists", _exists(summary_path), _rel(summary_path)),
        AuditRow("phase2_revised_goal2_tuned_guardrail_sensitivity", "guardrail sensitivity claims exists", _exists(claims_path), _rel(claims_path)),
        AuditRow("phase2_revised_goal2_tuned_guardrail_sensitivity", "guardrail sensitivity status exists", _exists(status_path), _rel(status_path)),
        AuditRow("phase2_revised_goal2_tuned_guardrail_sensitivity", "guardrail sensitivity manifest exists", _exists(manifest_path), _rel(manifest_path)),
    ]
    if not all(path.exists() for path in [sensitivity_path, summary_path, claims_path, status_path, manifest_path]):
        return rows

    sensitivity = pd.read_csv(sensitivity_path)
    summary = pd.read_csv(summary_path)
    claims = pd.read_csv(claims_path)
    status = pd.read_csv(status_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_scopes = {
        "aggregate_real_background_v2_formal",
        "human_lymph_node_visium_replicate_00",
        "human_tonsil_slide_tags_replicate_00",
        "human_melanoma_multiome_replicate_00",
    }
    rows.extend(
        [
            AuditRow(
                "phase2_revised_goal2_tuned_guardrail_sensitivity",
                "guardrail sensitivity covers aggregate and three tissue replicates",
                "complete" if required_scopes.issubset(set(summary["scope"].astype(str))) else "partial",
                _rel(summary_path),
                f"scopes={sorted(set(summary['scope'].astype(str))) if 'scope' in summary else []}",
            ),
            AuditRow(
                "phase2_revised_goal2_tuned_guardrail_sensitivity",
                "guardrail sensitivity scans FP and FDR threshold grid",
                "complete"
                if len(sensitivity) >= 120
                and sensitivity["max_high_expression_fp_rate"].nunique() >= 5
                and sensitivity["max_empirical_fdr"].nunique() >= 6
                else "partial",
                _rel(sensitivity_path),
                f"rows={len(sensitivity)} fp_thresholds={sensitivity['max_high_expression_fp_rate'].nunique() if 'max_high_expression_fp_rate' in sensitivity else 0} fdr_thresholds={sensitivity['max_empirical_fdr'].nunique() if 'max_empirical_fdr' in sensitivity else 0}",
            ),
            AuditRow(
                "phase2_revised_goal2_tuned_guardrail_sensitivity",
                "guardrail sensitivity records mixed evidence instead of SpatialSelf dominance",
                "complete"
                if len(claims) >= 2
                and not claims["supported"].astype(bool).any()
                and (summary["spatialself_win_fraction"].astype(float) < 0.5).any()
                else "partial",
                _rel(claims_path),
                claims.to_json(orient="records") if len(claims) else "",
            ),
            AuditRow(
                "phase2_revised_goal2_tuned_guardrail_sensitivity",
                "guardrail sensitivity status and manifest preserve non-final baseline scope",
                "complete"
                if len(status) == 1
                and status.iloc[0].get("status") == "completed"
                and manifest.get("evidence_status") == "guardrail_sensitivity_for_tuned_small_panel_not_final_baseline"
                else "partial",
                _rel(manifest_path),
                f"status={status.to_json(orient='records') if len(status) else []}; evidence_status={manifest.get('evidence_status')}",
            ),
        ]
    )
    return rows


def _audit_goal2_preregistered_endpoint_policy() -> list[AuditRow]:
    import pandas as pd

    root = PROJECT_ROOT / "outputs/phase2_revised/goal2_preregistered_endpoint_policy"
    config_path = PROJECT_ROOT / "configs/experiments/phase2_revised_goal2_preregistered_endpoint_policy.yaml"
    script_path = PROJECT_ROOT / "scripts/phase2/run_goal2_preregistered_endpoint_policy.py"
    scores_path = root / "goal2_preregistered_endpoint_scores.csv"
    summary_path = root / "goal2_preregistered_endpoint_summary.csv"
    claims_path = root / "goal2_preregistered_endpoint_claims.csv"
    policy_path = root / "goal2_preregistered_endpoint_policy_table.csv"
    fp_path = root / "goal2_preregistered_fp_suppression_by_tissue.csv"
    status_path = root / "goal2_preregistered_endpoint_status.csv"
    manifest_path = root / "goal2_preregistered_endpoint_manifest.json"
    rows = [
        AuditRow("phase2_revised_goal2_preregistered_endpoint_policy", "preregistered endpoint config exists", _exists(config_path), _rel(config_path)),
        AuditRow("phase2_revised_goal2_preregistered_endpoint_policy", "preregistered endpoint runner exists", _exists(script_path), _rel(script_path)),
        AuditRow("phase2_revised_goal2_preregistered_endpoint_policy", "preregistered endpoint scores exist", _exists(scores_path), _rel(scores_path)),
        AuditRow("phase2_revised_goal2_preregistered_endpoint_policy", "preregistered endpoint summary exists", _exists(summary_path), _rel(summary_path)),
        AuditRow("phase2_revised_goal2_preregistered_endpoint_policy", "preregistered endpoint claims exist", _exists(claims_path), _rel(claims_path)),
        AuditRow("phase2_revised_goal2_preregistered_endpoint_policy", "preregistered endpoint policy table exists", _exists(policy_path), _rel(policy_path)),
        AuditRow("phase2_revised_goal2_preregistered_endpoint_policy", "preregistered FP suppression by tissue table exists", _exists(fp_path), _rel(fp_path)),
        AuditRow("phase2_revised_goal2_preregistered_endpoint_policy", "preregistered endpoint status exists", _exists(status_path), _rel(status_path)),
        AuditRow("phase2_revised_goal2_preregistered_endpoint_policy", "preregistered endpoint manifest exists", _exists(manifest_path), _rel(manifest_path)),
    ]
    if not all(path.exists() for path in [scores_path, summary_path, claims_path, policy_path, fp_path, status_path, manifest_path]):
        return rows

    scores = pd.read_csv(scores_path)
    summary = pd.read_csv(summary_path)
    claims = pd.read_csv(claims_path)
    policy = pd.read_csv(policy_path)
    fp = pd.read_csv(fp_path)
    status = pd.read_csv(status_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_metrics = {
        "aupr",
        "top_k_precision",
        "weak_signal_recall_at_threshold",
        "empirical_fdr_at_threshold",
        "false_positive_high_expression_rate_at_threshold",
    }
    aggregate = summary[summary["scope"].astype(str) == "aggregate_real_background_v2_formal"]
    tissues = summary[summary["scope"].astype(str) != "aggregate_real_background_v2_formal"]
    rows.extend(
        [
            AuditRow(
                "phase2_revised_goal2_preregistered_endpoint_policy",
                "policy table freezes weighted primary endpoint components",
                "complete"
                if required_metrics.issubset(set(policy["metric"].astype(str)))
                and abs(float(policy["weight"].astype(float).sum()) - 1.0) < 1e-9
                else "partial",
                _rel(policy_path),
                policy.to_json(orient="records") if len(policy) else "",
            ),
            AuditRow(
                "phase2_revised_goal2_preregistered_endpoint_policy",
                "endpoint scores cover aggregate and three tissue scopes",
                "complete"
                if len(scores) >= 36
                and {
                    "aggregate_real_background_v2_formal",
                    "human_lymph_node_visium_replicate_00",
                    "human_melanoma_multiome_replicate_00",
                    "human_tonsil_slide_tags_replicate_00",
                }.issubset(set(scores["scope"].astype(str)))
                else "partial",
                _rel(scores_path),
                f"rows={len(scores)} scopes={sorted(set(scores['scope'].astype(str))) if 'scope' in scores else []}",
            ),
            AuditRow(
                "phase2_revised_goal2_preregistered_endpoint_policy",
                "frozen endpoint favors SpatialSelf at aggregate level and two of three tissues",
                "complete"
                if len(aggregate) == 1
                and bool(aggregate.iloc[0]["spatialself_primary_winner"])
                and int(tissues["spatialself_primary_winner"].astype(bool).sum()) >= 2
                else "partial",
                _rel(summary_path),
                summary.to_json(orient="records") if len(summary) else "",
            ),
            AuditRow(
                "phase2_revised_goal2_preregistered_endpoint_policy",
                "endpoint claims preserve policy-application non-final scope and no-overclaim rule",
                "complete"
                if len(claims) >= 3
                and (claims["claim_level"].astype(str) == "preregistered_policy_application_not_final_replicated_baseline").all()
                and claims["summary"].astype(str).str.contains("do not claim final tuned superiority").any()
                else "partial",
                _rel(claims_path),
                claims.to_json(orient="records") if len(claims) else "",
            ),
            AuditRow(
                "phase2_revised_goal2_preregistered_endpoint_policy",
                "FP suppression tissue table records all three tissues",
                "complete" if len(fp) >= 3 else "partial",
                _rel(fp_path),
                fp.to_json(orient="records") if len(fp) else "",
            ),
            AuditRow(
                "phase2_revised_goal2_preregistered_endpoint_policy",
                "endpoint status and manifest preserve diagnostic-not-final scope",
                "complete"
                if len(status) == 1
                and status.iloc[0].get("status") == "completed"
                and manifest.get("evidence_status") == "preregistered_policy_application_not_final_replicated_baseline"
                else "partial",
                _rel(manifest_path),
                f"status={status.to_json(orient='records') if len(status) else []}; evidence_status={manifest.get('evidence_status')}",
            ),
        ]
    )
    return rows


def _audit_goal2_preregistered_paired_diagnostics() -> list[AuditRow]:
    import pandas as pd

    root = PROJECT_ROOT / "outputs/phase2_revised/goal2_preregistered_paired_diagnostics"
    script_path = PROJECT_ROOT / "scripts/phase2/run_goal2_preregistered_paired_diagnostics.py"
    deltas_path = root / "goal2_preregistered_paired_deltas.csv"
    summary_path = root / "goal2_preregistered_paired_summary.csv"
    claims_path = root / "goal2_preregistered_paired_claims.csv"
    status_path = root / "goal2_preregistered_paired_status.csv"
    manifest_path = root / "goal2_preregistered_paired_manifest.json"
    rows = [
        AuditRow("phase2_revised_goal2_preregistered_paired_diagnostics", "preregistered paired diagnostic runner exists", _exists(script_path), _rel(script_path)),
        AuditRow("phase2_revised_goal2_preregistered_paired_diagnostics", "preregistered paired deltas exist", _exists(deltas_path), _rel(deltas_path)),
        AuditRow("phase2_revised_goal2_preregistered_paired_diagnostics", "preregistered paired summary exists", _exists(summary_path), _rel(summary_path)),
        AuditRow("phase2_revised_goal2_preregistered_paired_diagnostics", "preregistered paired claims exist", _exists(claims_path), _rel(claims_path)),
        AuditRow("phase2_revised_goal2_preregistered_paired_diagnostics", "preregistered paired status exists", _exists(status_path), _rel(status_path)),
        AuditRow("phase2_revised_goal2_preregistered_paired_diagnostics", "preregistered paired manifest exists", _exists(manifest_path), _rel(manifest_path)),
    ]
    if not all(path.exists() for path in [deltas_path, summary_path, claims_path, status_path, manifest_path]):
        return rows

    deltas = pd.read_csv(deltas_path)
    summary = pd.read_csv(summary_path)
    claims = pd.read_csv(claims_path)
    status = pd.read_csv(status_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_baselines = {"commot", "cellchat", "spatalk", "cellnest"}
    rows.extend(
        [
            AuditRow(
                "phase2_revised_goal2_preregistered_paired_diagnostics",
                "paired diagnostics compare SpatialSelf representative against four baselines across three tissues",
                "complete"
                if required_baselines.issubset(set(summary["baseline_method"].astype(str)))
                and len(deltas) >= 12
                and deltas["replicate"].nunique() >= 3
                else "partial",
                _rel(deltas_path),
                f"rows={len(deltas)} baselines={sorted(set(summary['baseline_method'].astype(str))) if 'baseline_method' in summary else []}",
            ),
            AuditRow(
                "phase2_revised_goal2_preregistered_paired_diagnostics",
                "paired diagnostics show positive mean delta but underpowered CI caveat",
                "complete"
                if (summary["mean_delta"].astype(float) > 0).all()
                and summary["underpowered"].astype(bool).all()
                and (summary["bootstrap_ci_low"].astype(float) <= 0).any()
                else "partial",
                _rel(summary_path),
                summary.to_json(orient="records") if len(summary) else "",
            ),
            AuditRow(
                "phase2_revised_goal2_preregistered_paired_diagnostics",
                "paired diagnostic claims preserve unsupported CI significance",
                "complete"
                if len(claims) >= 3
                and claims.loc[claims["claim_id"].astype(str) == "paired_endpoint_mean_delta", "supported"].astype(bool).any()
                and claims.loc[claims["claim_id"].astype(str) == "paired_endpoint_sign_consistency", "supported"].astype(bool).any()
                and not claims.loc[claims["claim_id"].astype(str) == "paired_endpoint_bootstrap_ci", "supported"].astype(bool).any()
                else "partial",
                _rel(claims_path),
                claims.to_json(orient="records") if len(claims) else "",
            ),
            AuditRow(
                "phase2_revised_goal2_preregistered_paired_diagnostics",
                "paired diagnostic status and manifest preserve underpowered-not-final scope",
                "complete"
                if len(status) == 1
                and status.iloc[0].get("status") == "completed"
                and manifest.get("evidence_status") == "paired_diagnostic_underpowered_not_final_replicated_baseline"
                else "partial",
                _rel(manifest_path),
                f"status={status.to_json(orient='records') if len(status) else []}; evidence_status={manifest.get('evidence_status')}",
            ),
        ]
    )
    return rows


def _audit_goal2_replicate_power_plan() -> list[AuditRow]:
    import pandas as pd

    root = PROJECT_ROOT / "outputs/phase2_revised/goal2_replicate_power_plan"
    script_path = PROJECT_ROOT / "scripts/phase2/run_goal2_replicate_power_plan.py"
    by_baseline_path = root / "goal2_replicate_power_plan_by_baseline.csv"
    overall_path = root / "goal2_replicate_power_plan_overall.csv"
    design_path = root / "goal2_replicate_power_plan_design.csv"
    claims_path = root / "goal2_replicate_power_plan_claims.csv"
    status_path = root / "goal2_replicate_power_plan_status.csv"
    manifest_path = root / "goal2_replicate_power_plan_manifest.json"
    rows = [
        AuditRow("phase2_revised_goal2_replicate_power_plan", "replicate power planning runner exists", _exists(script_path), _rel(script_path)),
        AuditRow("phase2_revised_goal2_replicate_power_plan", "replicate power by-baseline table exists", _exists(by_baseline_path), _rel(by_baseline_path)),
        AuditRow("phase2_revised_goal2_replicate_power_plan", "replicate power overall table exists", _exists(overall_path), _rel(overall_path)),
        AuditRow("phase2_revised_goal2_replicate_power_plan", "replicate power design table exists", _exists(design_path), _rel(design_path)),
        AuditRow("phase2_revised_goal2_replicate_power_plan", "replicate power claims exist", _exists(claims_path), _rel(claims_path)),
        AuditRow("phase2_revised_goal2_replicate_power_plan", "replicate power status exists", _exists(status_path), _rel(status_path)),
        AuditRow("phase2_revised_goal2_replicate_power_plan", "replicate power manifest exists", _exists(manifest_path), _rel(manifest_path)),
    ]
    if not all(path.exists() for path in [by_baseline_path, overall_path, design_path, claims_path, status_path, manifest_path]):
        return rows

    by_baseline = pd.read_csv(by_baseline_path)
    overall = pd.read_csv(overall_path)
    design = pd.read_csv(design_path)
    claims = pd.read_csv(claims_path)
    status = pd.read_csv(status_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows.extend(
        [
            AuditRow(
                "phase2_revised_goal2_replicate_power_plan",
                "replicate power plan covers four tuned baseline comparators",
                "complete"
                if {"commot", "cellchat", "spatalk", "cellnest"}.issubset(set(by_baseline["baseline_method"].astype(str)))
                else "partial",
                _rel(by_baseline_path),
                by_baseline.to_json(orient="records") if len(by_baseline) else "",
            ),
            AuditRow(
                "phase2_revised_goal2_replicate_power_plan",
                "replicate power plan identifies CellNEST bottleneck and at least 18 matched replicates",
                "complete"
                if len(overall) == 1
                and str(overall.iloc[0]["bottleneck_baseline"]) == "cellnest"
                and int(overall.iloc[0]["target_min_replicates"]) >= 18
                and int(overall.iloc[0]["additional_replicates_needed"]) >= 15
                else "partial",
                _rel(overall_path),
                overall.to_json(orient="records") if len(overall) else "",
            ),
            AuditRow(
                "phase2_revised_goal2_replicate_power_plan",
                "replicate power design freezes endpoint, baseline params, representative rule, seeds, and status handling",
                "complete"
                if _table_text_contains(design, ["endpoint policy", "baseline parameters", "SpatialSelf representative", "random seeds", "failure/status"])
                else "partial",
                _rel(design_path),
                design.to_json(orient="records") if len(design) else "",
            ),
            AuditRow(
                "phase2_revised_goal2_replicate_power_plan",
                "replicate power claims preserve planning-not-proof scope",
                "complete"
                if len(claims) >= 3
                and (claims["claim_level"].astype(str) == "replicate_planning_not_power_proof").all()
                and claims["summary"].astype(str).str.contains("not proof").any()
                else "partial",
                _rel(claims_path),
                claims.to_json(orient="records") if len(claims) else "",
            ),
            AuditRow(
                "phase2_revised_goal2_replicate_power_plan",
                "replicate power status and manifest preserve planning-not-proof scope",
                "complete"
                if len(status) == 1
                and status.iloc[0].get("status") == "completed"
                and manifest.get("evidence_status") == "replicate_planning_not_power_proof"
                else "partial",
                _rel(manifest_path),
                f"status={status.to_json(orient='records') if len(status) else []}; evidence_status={manifest.get('evidence_status')}",
            ),
        ]
    )
    return rows


def _audit_goal2_replicated_tuned_job_matrix() -> list[AuditRow]:
    import pandas as pd

    root = PROJECT_ROOT / "outputs/phase2_revised/goal2_replicated_tuned_baseline_jobs"
    config_path = PROJECT_ROOT / "configs/experiments/phase2_revised_goal2_replicated_tuned_baseline_jobs.yaml"
    script_path = PROJECT_ROOT / "scripts/phase2/run_goal2_replicated_tuned_job_matrix.py"
    execution_path = root / "goal2_replicated_tuned_execution_plan.csv"
    replicate_path = root / "goal2_replicated_tuned_replicate_manifest.csv"
    jobs_path = root / "goal2_replicated_tuned_method_jobs.csv"
    status_path = root / "goal2_replicated_tuned_job_status_template.csv"
    claims_path = root / "goal2_replicated_tuned_job_claims.csv"
    manifest_path = root / "goal2_replicated_tuned_job_manifest.json"
    rows = [
        AuditRow("phase2_revised_goal2_replicated_tuned_job_matrix", "replicated tuned job config exists", _exists(config_path), _rel(config_path)),
        AuditRow("phase2_revised_goal2_replicated_tuned_job_matrix", "replicated tuned job runner exists", _exists(script_path), _rel(script_path)),
        AuditRow("phase2_revised_goal2_replicated_tuned_job_matrix", "replicated tuned execution plan exists", _exists(execution_path), _rel(execution_path)),
        AuditRow("phase2_revised_goal2_replicated_tuned_job_matrix", "replicated tuned replicate manifest exists", _exists(replicate_path), _rel(replicate_path)),
        AuditRow("phase2_revised_goal2_replicated_tuned_job_matrix", "replicated tuned method jobs exists", _exists(jobs_path), _rel(jobs_path)),
        AuditRow("phase2_revised_goal2_replicated_tuned_job_matrix", "replicated tuned status template exists", _exists(status_path), _rel(status_path)),
        AuditRow("phase2_revised_goal2_replicated_tuned_job_matrix", "replicated tuned job claims exists", _exists(claims_path), _rel(claims_path)),
        AuditRow("phase2_revised_goal2_replicated_tuned_job_matrix", "replicated tuned job manifest exists", _exists(manifest_path), _rel(manifest_path)),
    ]
    if not all(path.exists() for path in [execution_path, replicate_path, jobs_path, status_path, claims_path, manifest_path]):
        return rows

    execution = pd.read_csv(execution_path)
    replicates = pd.read_csv(replicate_path)
    jobs = pd.read_csv(jobs_path)
    status = pd.read_csv(status_path)
    claims = pd.read_csv(claims_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows.extend(
        [
            AuditRow(
                "phase2_revised_goal2_replicated_tuned_job_matrix",
                "replicated tuned plan defines 18 matched real-background replicates across three datasets",
                "complete"
                if len(replicates) == 18
                and {"lymph_node", "tonsil", "melanoma"}.issubset(set(replicates["dataset_key"].astype(str)))
                and (replicates["planned_status"].astype(str) == "generated").all()
                else "partial",
                _rel(replicate_path),
                f"rows={len(replicates)} datasets={sorted(set(replicates['dataset_key'].astype(str))) if 'dataset_key' in replicates else []}",
            ),
            AuditRow(
                "phase2_revised_goal2_replicated_tuned_job_matrix",
                "replicated tuned method job table defines 144 method/status jobs",
                "complete"
                if len(jobs) == 144
                and {"spatialself_wavelet_edge_prior", "spatialself_wavelet", "commot", "cellchat", "spatalk", "cellnest", "deeptalk", "niches"}.issubset(set(jobs["method"].astype(str)))
                and len(status) == 144
                else "partial",
                _rel(jobs_path),
                f"jobs={len(jobs)} status_rows={len(status)} methods={sorted(set(jobs['method'].astype(str))) if 'method' in jobs else []}",
            ),
            AuditRow(
                "phase2_revised_goal2_replicated_tuned_job_matrix",
                "replicated tuned execution plan has generated datasets and pending baselines",
                "complete"
                if len(execution) >= 4
                and (execution.loc[execution["step"].astype(str) == "generate_replicated_real_background_datasets", "status"].astype(str) == "completed").any()
                and (execution.loc[execution["step"].astype(str) == "run_metric_baselines_per_replicate", "status"].astype(str) == "planned_not_run").any()
                else "partial",
                _rel(execution_path),
                execution.to_json(orient="records") if len(execution) else "",
            ),
            AuditRow(
                "phase2_revised_goal2_replicated_tuned_job_matrix",
                "replicated tuned claims explicitly avoid result claims",
                "complete"
                if len(claims) >= 3
                and claims["claim_level"].astype(str)
                .isin(
                    {
                        "execution_plan_not_run_results",
                        "execution_plan_dataset_generated_no_baseline_results",
                    }
                )
                .all()
                and claims["summary"].astype(str).str.contains("no replicated tuned baseline result is claimed").any()
                else "partial",
                _rel(claims_path),
                claims.to_json(orient="records") if len(claims) else "",
            ),
            AuditRow(
                "phase2_revised_goal2_replicated_tuned_job_matrix",
                "replicated tuned manifest records datasets-generated-baselines-not-run scope",
                "complete"
                if manifest.get("evidence_status") == "datasets_generated_baselines_not_run"
                and int(manifest.get("target_matched_replicates", 0)) == 18
                and int(manifest.get("generated_matched_replicates", 0)) == 18
                and int(manifest.get("method_jobs", 0)) == 144
                else "partial",
                _rel(manifest_path),
                f"evidence_status={manifest.get('evidence_status')}; target={manifest.get('target_matched_replicates')}; generated={manifest.get('generated_matched_replicates')}; jobs={manifest.get('method_jobs')}",
            ),
        ]
    )
    return rows


def _audit_goal4_replicated_tuned_datasets() -> list[AuditRow]:
    import pandas as pd

    root = PROJECT_ROOT / "outputs/phase2_revised/goal4_real_background_v2/replicated_tuned_datasets"
    status_path = root / "formal_dataset_status.csv"
    validation_path = root / "formal_validation_summary.csv"
    summary_path = root / "formal_real_background_summary.csv"
    claims_path = root / "formal_real_background_claims.csv"
    manifest_path = root / "manifest.csv"
    json_manifest_path = root / "formal_real_background_manifest.json"
    rows = [
        AuditRow("phase2_revised_goal4_replicated_tuned_datasets", "replicated tuned dataset status exists", _exists(status_path), _rel(status_path)),
        AuditRow("phase2_revised_goal4_replicated_tuned_datasets", "replicated tuned validation summary exists", _exists(validation_path), _rel(validation_path)),
        AuditRow("phase2_revised_goal4_replicated_tuned_datasets", "replicated tuned real-background summary exists", _exists(summary_path), _rel(summary_path)),
        AuditRow("phase2_revised_goal4_replicated_tuned_datasets", "replicated tuned claims exist", _exists(claims_path), _rel(claims_path)),
        AuditRow("phase2_revised_goal4_replicated_tuned_datasets", "replicated tuned manifest exists", _exists(manifest_path), _rel(manifest_path)),
        AuditRow("phase2_revised_goal4_replicated_tuned_datasets", "replicated tuned JSON manifest exists", _exists(json_manifest_path), _rel(json_manifest_path)),
    ]
    if not all(path.exists() for path in [status_path, validation_path, summary_path, claims_path, manifest_path, json_manifest_path]):
        return rows

    status = pd.read_csv(status_path)
    validation = pd.read_csv(validation_path)
    summary = pd.read_csv(summary_path)
    claims = pd.read_csv(claims_path)
    manifest = pd.read_csv(manifest_path)
    json_manifest = json.loads(json_manifest_path.read_text(encoding="utf-8"))
    rows.extend(
        [
            AuditRow(
                "phase2_revised_goal4_replicated_tuned_datasets",
                "replicated tuned datasets generated six replicates per background",
                "complete"
                if len(status) == 18
                and (status["status"].astype(str) == "generated").all()
                and set(status.groupby("dataset_key").size().to_dict().values()) == {6}
                else "partial",
                _rel(status_path),
                status.groupby(["dataset_key", "status"]).size().to_json() if len(status) else "",
            ),
            AuditRow(
                "phase2_revised_goal4_replicated_tuned_datasets",
                "replicated tuned dataset files have intact validation and truth summaries",
                "complete"
                if len(validation) == 18
                and len(manifest) == 18
                and (validation["cells"].astype(int) == 900).all()
                and (validation["positive_interactions"].astype(float) > 0).all()
                and (validation["weak_positive_interactions"].astype(float) > 0).all()
                and (validation["false_positive_labels"].astype(int) == 0).all()
                else "partial",
                _rel(validation_path),
                f"validation_rows={len(validation)} manifest_rows={len(manifest)}",
            ),
            AuditRow(
                "phase2_revised_goal4_replicated_tuned_datasets",
                "replicated tuned summary covers lymph, tonsil, and melanoma",
                "complete"
                if {"lymph_node", "tonsil", "melanoma"}.issubset(set(summary["dataset_key"].astype(str)))
                and (summary["n_replicates"].astype(int) == 6).all()
                else "partial",
                _rel(summary_path),
                summary.to_json(orient="records") if len(summary) else "",
            ),
            AuditRow(
                "phase2_revised_goal4_replicated_tuned_datasets",
                "replicated tuned dataset claims are supported but baseline-free",
                "complete"
                if len(claims) >= 4
                and (claims["status"].astype(str) == "supported").all()
                and json_manifest.get("evidence_status") == "formal_real_background_structure_generated_not_final_manuscript_scale"
                else "partial",
                _rel(claims_path),
                f"claims={claims.to_json(orient='records') if len(claims) else []}; evidence_status={json_manifest.get('evidence_status')}",
            ),
        ]
    )
    return rows


def _audit_goal2_spatialself_fp_advantage(metrics_path: Path, metrics) -> AuditRow:
    comparisons = []
    for dataset, group in metrics.groupby("dataset", sort=False):
        commot = group[group["method"].astype(str) == "commot"]
        spatialself = group[group["method"].astype(str).str.startswith("spatialself")]
        if commot.empty or spatialself.empty:
            continue
        commot_fp = float(commot["false_positive_high_expression_rate_at_threshold"].min())
        spatialself_fp = float(spatialself["false_positive_high_expression_rate_at_threshold"].min())
        comparisons.append(spatialself_fp <= commot_fp)
    return AuditRow(
        "phase2_revised_goal2_weak_signal_tradeoff",
        "SpatialSelf variants have lower or equal high-expression FP rate than COMMOT in smoke datasets",
        "complete" if comparisons and all(comparisons) else "partial",
        _rel(metrics_path),
        f"passes={sum(comparisons)}/{len(comparisons)}",
    )


def _audit_goal2_weak_signal_stress_formal() -> list[AuditRow]:
    import pandas as pd

    root = PROJECT_ROOT / "outputs/phase2_revised/goal2_weak_signal_stress_formal"
    script_path = PROJECT_ROOT / "scripts/phase2/run_goal2_weak_signal_stress_summary.py"
    metrics_path = root / "goal2_weak_signal_stress_metrics.csv"
    summary_path = root / "goal2_weak_signal_stress_summary.csv"
    claims_path = root / "goal2_weak_signal_stress_claims.csv"
    status_path = root / "goal2_weak_signal_stress_status.csv"
    manifest_path = root / "goal2_weak_signal_stress_manifest.json"
    rows = [
        AuditRow("phase2_revised_goal2_weak_signal_stress_formal", "weak-signal stress runner exists", _exists(script_path), _rel(script_path)),
        AuditRow("phase2_revised_goal2_weak_signal_stress_formal", "weak-signal stress metrics table exists", _exists(metrics_path), _rel(metrics_path)),
        AuditRow("phase2_revised_goal2_weak_signal_stress_formal", "weak-signal stress summary table exists", _exists(summary_path), _rel(summary_path)),
        AuditRow("phase2_revised_goal2_weak_signal_stress_formal", "weak-signal stress claims table exists", _exists(claims_path), _rel(claims_path)),
        AuditRow("phase2_revised_goal2_weak_signal_stress_formal", "weak-signal stress status table exists", _exists(status_path), _rel(status_path)),
        AuditRow("phase2_revised_goal2_weak_signal_stress_formal", "weak-signal stress manifest exists", _exists(manifest_path), _rel(manifest_path)),
    ]
    if not all(path.exists() for path in [metrics_path, summary_path, claims_path, status_path, manifest_path]):
        return rows

    metrics = pd.read_csv(metrics_path)
    summary = pd.read_csv(summary_path)
    claims = pd.read_csv(claims_path)
    status = pd.read_csv(status_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_datasets = {"controlled_v2_formal", "real_background_v2_formal"}
    required_mechanisms = {
        "all_mechanisms",
        "juxtacrine_contact",
        "short_medium_paracrine",
        "long_range_chemokine_niche",
    }
    required_methods = {
        "spatialself_wavelet_edge_prior",
        "spatialself_wavelet",
        "commot",
        "cellchat",
        "spatalk",
        "cellnest",
    }
    datasets = set(metrics["dataset"].astype(str)) if "dataset" in metrics else set()
    mechanisms = set(metrics["mechanism"].astype(str)) if "mechanism" in metrics else set()
    methods = set(metrics["method"].astype(str)) if "method" in metrics else set()
    claim_pairs = set(zip(claims["dataset"].astype(str), claims["mechanism"].astype(str))) if len(claims) else set()
    required_pairs = {(dataset, mechanism) for dataset in required_datasets for mechanism in required_mechanisms}
    raw_weak_not_spatial = (
        not claims.empty
        and (~claims["highest_weak_recall_method"].astype(str).str.startswith("spatialself")).any()
    )
    rows.extend(
        [
            AuditRow(
                "phase2_revised_goal2_weak_signal_stress_formal",
                "weak-stress metrics cover controlled and real-background formal datasets",
                "complete" if required_datasets.issubset(datasets) else "partial",
                _rel(metrics_path),
                f"datasets={sorted(datasets)}",
            ),
            AuditRow(
                "phase2_revised_goal2_weak_signal_stress_formal",
                "weak-stress metrics cover all mechanisms plus contact, paracrine, and long-range strata",
                "complete" if required_mechanisms.issubset(mechanisms) else "partial",
                _rel(metrics_path),
                f"mechanisms={sorted(mechanisms)}",
            ),
            AuditRow(
                "phase2_revised_goal2_weak_signal_stress_formal",
                "weak-stress metrics include six metric methods for every formal dataset-mechanism pair",
                "complete"
                if required_methods.issubset(methods)
                and all(
                    required_methods.issubset(
                        set(
                            metrics.loc[
                                (metrics["dataset"].astype(str) == dataset)
                                & (metrics["mechanism"].astype(str) == mechanism),
                                "method",
                            ].astype(str)
                        )
                    )
                    for dataset, mechanism in required_pairs
                )
                else "partial",
                _rel(metrics_path),
                f"methods={sorted(methods)} rows={len(metrics)}",
            ),
            AuditRow(
                "phase2_revised_goal2_weak_signal_stress_formal",
                "weak-stress claims cover every formal dataset-mechanism pair",
                "complete" if required_pairs.issubset(claim_pairs) else "partial",
                _rel(claims_path),
                f"missing={sorted(required_pairs - claim_pairs)}",
            ),
            AuditRow(
                "phase2_revised_goal2_weak_signal_stress_formal",
                "SpatialSelf wins FP/FDR-guardrailed tradeoff across weak-stress dataset-mechanism pairs",
                "complete"
                if len(claims) >= len(required_pairs)
                and required_pairs.issubset(claim_pairs)
                and claims["spatialself_best_guardrailed_tradeoff"].astype(bool).all()
                and set(claims["best_guardrailed_tradeoff_status"].astype(str)) == {"fp_fdr_guardrail_met"}
                else "partial",
                _rel(claims_path),
                claims[
                    [
                        "dataset",
                        "mechanism",
                        "best_guardrailed_tradeoff_method",
                        "highest_weak_recall_method",
                        "lowest_high_expression_fp_method",
                    ]
                ].to_json(orient="records") if len(claims) else "",
            ),
            AuditRow(
                "phase2_revised_goal2_weak_signal_stress_formal",
                "weak-stress claims explicitly preserve raw weak-recall caveat",
                "complete" if raw_weak_not_spatial else "partial",
                _rel(claims_path),
                claims[["dataset", "mechanism", "highest_weak_recall_method"]].to_json(orient="records") if len(claims) else "",
            ),
            AuditRow(
                "phase2_revised_goal2_weak_signal_stress_formal",
                "all weak-stress method-prediction joins completed",
                "complete"
                if not status.empty and (status["status"].astype(str) == "completed").all() and len(status) >= 36
                else "partial",
                _rel(status_path),
                f"rows={len(status)} statuses={sorted(set(status['status'].astype(str))) if 'status' in status else []}",
            ),
            AuditRow(
                "phase2_revised_goal2_weak_signal_stress_formal",
                "weak-stress manifest preserves non-final tuned-baseline caveat",
                "complete"
                if manifest.get("claim_level") == "formal_weak_signal_stress_not_final_tuned_baseline"
                else "partial",
                _rel(manifest_path),
                f"claim_level={manifest.get('claim_level')}",
            ),
        ]
    )
    return rows


def _audit_goal34_benchmark_quality_report() -> list[AuditRow]:
    import pandas as pd

    root = PROJECT_ROOT / "outputs/phase2_revised/goal34_benchmark_quality_report"
    script_path = PROJECT_ROOT / "scripts/phase2/run_goal34_benchmark_quality_report.py"
    quality_path = root / "goal34_benchmark_quality_summary.csv"
    mechanism_path = root / "goal34_mechanism_coverage_matrix.csv"
    properties_path = root / "goal34_real_background_property_summary.csv"
    truth_path = root / "goal34_truth_contract_summary.csv"
    claims_path = root / "goal34_benchmark_quality_claims.csv"
    manifest_path = root / "goal34_benchmark_quality_manifest.json"
    rows = [
        AuditRow("phase2_revised_goal34_benchmark_quality_report", "benchmark quality report runner exists", _exists(script_path), _rel(script_path)),
        AuditRow("phase2_revised_goal34_benchmark_quality_report", "benchmark quality summary exists", _exists(quality_path), _rel(quality_path)),
        AuditRow("phase2_revised_goal34_benchmark_quality_report", "mechanism coverage matrix exists", _exists(mechanism_path), _rel(mechanism_path)),
        AuditRow("phase2_revised_goal34_benchmark_quality_report", "real-background property summary exists", _exists(properties_path), _rel(properties_path)),
        AuditRow("phase2_revised_goal34_benchmark_quality_report", "truth contract summary exists", _exists(truth_path), _rel(truth_path)),
        AuditRow("phase2_revised_goal34_benchmark_quality_report", "benchmark quality claims exists", _exists(claims_path), _rel(claims_path)),
        AuditRow("phase2_revised_goal34_benchmark_quality_report", "benchmark quality manifest exists", _exists(manifest_path), _rel(manifest_path)),
    ]
    if not all(path.exists() for path in [quality_path, mechanism_path, properties_path, truth_path, claims_path, manifest_path]):
        return rows

    quality = pd.read_csv(quality_path)
    mechanism = pd.read_csv(mechanism_path)
    properties = pd.read_csv(properties_path)
    truth = pd.read_csv(truth_path)
    claims = pd.read_csv(claims_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_quality_keys = {"controlled", "lymph_node", "tonsil", "melanoma"}
    quality_keys = set(quality["dataset_key"].astype(str)) if "dataset_key" in quality else set()
    required_mechanisms = {
        "juxtacrine_contact",
        "short_medium_paracrine",
        "long_range_chemokine_niche",
        "false_positive_high_expression",
    }
    controlled_mechanisms = set(
        mechanism.loc[mechanism["dataset_key"].astype(str) == "controlled", "mechanism"].astype(str)
    ) if "dataset_key" in mechanism else set()
    rows.extend(
        [
            AuditRow(
                "phase2_revised_goal34_benchmark_quality_report",
                "quality report covers controlled, lymph node, tonsil, and melanoma benchmarks",
                "complete" if required_quality_keys.issubset(quality_keys) else "partial",
                _rel(quality_path),
                f"dataset_keys={sorted(quality_keys)}",
            ),
            AuditRow(
                "phase2_revised_goal34_benchmark_quality_report",
                "quality report marks all benchmark quality rows as passing",
                "complete"
                if len(quality) >= 4
                and quality["benchmark_quality_pass"].astype(bool).all()
                and quality["truth_contract_ok"].astype(bool).all()
                and quality["heuristic_not_trivial"].astype(bool).all()
                else "partial",
                _rel(quality_path),
                quality[["dataset_key", "benchmark_quality_pass", "heuristic_mean_auroc", "heuristic_mean_aupr"]].to_json(orient="records") if len(quality) else "",
            ),
            AuditRow(
                "phase2_revised_goal34_benchmark_quality_report",
                "controlled simulator mechanism matrix covers required mechanisms",
                "complete"
                if required_mechanisms.issubset(controlled_mechanisms)
                and mechanism.loc[mechanism["dataset_key"].astype(str) == "controlled", "present"].astype(bool).all()
                else "partial",
                _rel(mechanism_path),
                f"controlled_mechanisms={sorted(controlled_mechanisms)}",
            ),
            AuditRow(
                "phase2_revised_goal34_benchmark_quality_report",
                "real-background property summary records measured ST properties for three datasets",
                "complete"
                if {"lymph_node", "tonsil", "melanoma"}.issubset(set(properties["dataset_key"].astype(str)))
                and properties["property_pass"].astype(bool).all()
                else "partial",
                _rel(properties_path),
                properties[["dataset_key", "mean_observed_sparsity", "median_library_size", "regions", "cell_types", "property_pass"]].to_json(orient="records") if len(properties) else "",
            ),
            AuditRow(
                "phase2_revised_goal34_benchmark_quality_report",
                "truth contract summary records weak positives and zero false-positive truth edges",
                "complete"
                if len(truth) >= 4
                and truth["truth_contract_pass"].astype(bool).all()
                and (truth["weak_truth_edges"].astype(int) > 0).all()
                and (truth["false_positive_truth_edges"].astype(int) == 0).all()
                else "partial",
                _rel(truth_path),
                truth[["dataset_key", "weak_truth_edges", "false_positive_truth_edges", "truth_contract_pass"]].to_json(orient="records") if len(truth) else "",
            ),
            AuditRow(
                "phase2_revised_goal34_benchmark_quality_report",
                "benchmark quality claims are supported and preserve non-final realism caveat",
                "complete"
                if len(claims) >= 4
                and set(claims["status"].astype(str)) == {"supported"}
                and set(claims["claim_level"].astype(str)) == {"benchmark_quality_realism_summary_not_final_external_realism_test"}
                and manifest.get("claim_level") == "benchmark_quality_realism_summary_not_final_external_realism_test"
                else "partial",
                _rel(claims_path),
                f"statuses={sorted(set(claims['status'].astype(str))) if 'status' in claims else []}; manifest={manifest.get('claim_level')}",
            ),
        ]
    )
    return rows


def _audit_goal34_truth_contract_validator() -> list[AuditRow]:
    import pandas as pd

    root = PROJECT_ROOT / "outputs/phase2_revised/goal34_truth_contract"
    script_path = PROJECT_ROOT / "scripts/phase2/validate_goal34_truth_contract.py"
    replicate_path = root / "goal34_truth_contract_replicate_summary.csv"
    dataset_path = root / "goal34_truth_contract_dataset_summary.csv"
    claims_path = root / "goal34_truth_contract_claims.csv"
    manifest_path = root / "goal34_truth_contract_manifest.json"
    rows = [
        AuditRow("phase2_revised_goal34_truth_contract", "truth contract validator exists", _exists(script_path), _rel(script_path)),
        AuditRow("phase2_revised_goal34_truth_contract", "replicate contract summary exists", _exists(replicate_path), _rel(replicate_path)),
        AuditRow("phase2_revised_goal34_truth_contract", "dataset contract summary exists", _exists(dataset_path), _rel(dataset_path)),
        AuditRow("phase2_revised_goal34_truth_contract", "truth contract claims exists", _exists(claims_path), _rel(claims_path)),
        AuditRow("phase2_revised_goal34_truth_contract", "truth contract manifest exists", _exists(manifest_path), _rel(manifest_path)),
    ]
    if not all(path.exists() for path in [replicate_path, dataset_path, claims_path, manifest_path]):
        return rows
    replicate = pd.read_csv(replicate_path)
    dataset = pd.read_csv(dataset_path)
    claims = pd.read_csv(claims_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_dataset_keys = {"controlled", "lymph_node", "tonsil", "melanoma"}
    dataset_keys = set(dataset["dataset_key"].astype(str)) if "dataset_key" in dataset else set()
    rows.extend(
        [
            AuditRow(
                "phase2_revised_goal34_truth_contract",
                "truth contract validates controlled plus 18 real-background replicates",
                "complete"
                if len(replicate) >= 21
                and (replicate["status"].astype(str) == "complete").all()
                and int(manifest.get("complete_replicates", 0)) >= 21
                else "partial",
                _rel(replicate_path),
                f"rows={len(replicate)} statuses={sorted(set(replicate['status'].astype(str))) if 'status' in replicate else []}; manifest_complete={manifest.get('complete_replicates')}",
            ),
            AuditRow(
                "phase2_revised_goal34_truth_contract",
                "truth contract dataset summary covers controlled, lymph node, tonsil, and melanoma",
                "complete" if required_dataset_keys.issubset(dataset_keys) else "partial",
                _rel(dataset_path),
                f"dataset_keys={sorted(dataset_keys)}",
            ),
            AuditRow(
                "phase2_revised_goal34_truth_contract",
                "truth contract confirms weak truths and zero false-positive truth edges",
                "complete"
                if len(dataset) >= 4
                and (dataset["weak_truth_edges"].astype(int) > 0).all()
                and (dataset["false_positive_true_edges"].astype(int) == 0).all()
                and dataset["truth_rows_match_all_replicates"].astype(bool).all()
                else "partial",
                _rel(dataset_path),
                dataset[["dataset_key", "weak_truth_edges", "false_positive_true_edges", "truth_rows_match_all_replicates"]].to_json(orient="records") if len(dataset) else "",
            ),
            AuditRow(
                "phase2_revised_goal34_truth_contract",
                "truth contract claims are supported",
                "complete"
                if len(claims) >= 4 and set(claims["status"].astype(str)) == {"supported"}
                else "partial",
                _rel(claims_path),
                f"statuses={sorted(set(claims['status'].astype(str))) if 'status' in claims else []}",
            ),
        ]
    )
    return rows


def _audit_goal5_formal_fdr() -> list[AuditRow]:
    import pandas as pd

    root = PROJECT_ROOT / "outputs/phase2_revised/goal5_formal_fdr"
    summary_path = root / "goal5_formal_fdr_summary.csv"
    group_summary_path = root / "goal5_formal_fdr_group_summary.csv"
    manifest_path = root / "goal5_formal_fdr_manifest.json"
    rows = [
        AuditRow("phase2_revised_goal5_formal_fdr", "summary table exists", _exists(summary_path), _rel(summary_path)),
        AuditRow("phase2_revised_goal5_formal_fdr", "group summary table exists", _exists(group_summary_path), _rel(group_summary_path)),
        AuditRow("phase2_revised_goal5_formal_fdr", "manifest exists", _exists(manifest_path), _rel(manifest_path)),
    ]
    if not (summary_path.exists() and group_summary_path.exists() and manifest_path.exists()):
        return rows
    summary = pd.read_csv(summary_path)
    group_summary = pd.read_csv(group_summary_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_nulls = {"spatial_permutation", "receiver_permutation", "lr_identity_permutation"}
    nulls = set(summary["null_model"].astype(str))
    group_nulls = set(group_summary["null_model"].astype(str))
    fdr_paths = manifest.get("fdr_paths", {})
    rows.extend(
        [
            AuditRow(
                "phase2_revised_goal5_formal_fdr",
                "FDR summary covers fixed spatial, receiver, and LR-identity nulls",
                "complete" if required_nulls == nulls else "partial",
                _rel(summary_path),
                f"nulls={sorted(nulls)}",
            ),
            AuditRow(
                "phase2_revised_goal5_formal_fdr",
                "group summary covers LR-pair and cell-type-pair outputs for all nulls",
                "complete"
                if required_nulls.issubset(group_nulls)
                and {"lr_pair", "cell_type_pair"}.issubset(set(group_summary["group_level"].astype(str)))
                else "partial",
                _rel(group_summary_path),
                f"nulls={sorted(group_nulls)}; levels={sorted(set(group_summary['group_level'].astype(str)))}",
            ),
            AuditRow(
                "phase2_revised_goal5_formal_fdr",
                "manifest records smoke-level claim without manuscript overclaim",
                "complete" if manifest.get("claim_level") == "formal_fdr_smoke_not_manuscript_ready" else "partial",
                _rel(manifest_path),
                f"claim_level={manifest.get('claim_level')}",
            ),
            AuditRow(
                "phase2_revised_goal5_formal_fdr",
                "manifest records reusable FDR framework module",
                "complete"
                if manifest.get("framework_module") == "src.spitialself.fdr.run_formal_fdr_panel"
                else "partial",
                _rel(manifest_path),
                f"framework_module={manifest.get('framework_module')}",
            ),
        ]
    )
    expected_suffixes = (
        "candidate_edge",
        "lr_pair",
        "cell_type_pair",
        "significant_spatial_edges",
        "significant_lr_pairs",
        "significant_cell_type_pairs",
    )
    missing_paths = []
    for null_model in required_nulls:
        for suffix in expected_suffixes:
            key = f"{null_model}:{suffix}"
            path = Path(fdr_paths.get(key, ""))
            if not path.exists():
                missing_paths.append(key)
    rows.append(
        AuditRow(
            "phase2_revised_goal5_formal_fdr",
            "candidate, group, and significant-output CSVs exist for every fixed null",
            "complete" if not missing_paths else "missing",
            _rel(root),
            f"missing={missing_paths[:20]}",
        )
    )
    rows.append(
        AuditRow(
            "phase2_revised_goal5_formal_fdr",
            "truth diagnostics are available for benchmark FDR smoke",
            "complete" if summary["truth_diagnostics_available"].astype(bool).all() else "partial",
            _rel(summary_path),
            f"rows={len(summary)}",
        )
    )
    return rows


def _audit_goal5_fdr_200perm() -> list[AuditRow]:
    import pandas as pd

    root = PROJECT_ROOT / "outputs/phase2_revised/goal5_fdr_200perm"
    config_path = PROJECT_ROOT / "configs/experiments/phase2_revised_goal5_fdr_200perm.yaml"
    summary_path = root / "goal5_formal_fdr_summary.csv"
    group_summary_path = root / "goal5_formal_fdr_group_summary.csv"
    manifest_path = root / "goal5_formal_fdr_manifest.json"
    rows = [
        AuditRow("phase2_revised_goal5_fdr_200perm", "200-permutation FDR config exists", _exists(config_path), _rel(config_path)),
        AuditRow("phase2_revised_goal5_fdr_200perm", "200-permutation FDR summary exists", _exists(summary_path), _rel(summary_path)),
        AuditRow("phase2_revised_goal5_fdr_200perm", "200-permutation FDR group summary exists", _exists(group_summary_path), _rel(group_summary_path)),
        AuditRow("phase2_revised_goal5_fdr_200perm", "200-permutation FDR manifest exists", _exists(manifest_path), _rel(manifest_path)),
    ]
    if not all(path.exists() for path in [summary_path, group_summary_path, manifest_path]):
        return rows

    summary = pd.read_csv(summary_path)
    group_summary = pd.read_csv(group_summary_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_nulls = {"spatial_permutation", "receiver_permutation", "lr_identity_permutation"}
    nulls = set(summary["null_model"].astype(str))
    group_nulls = set(group_summary["null_model"].astype(str))
    rows.extend(
        [
            AuditRow(
                "phase2_revised_goal5_fdr_200perm",
                "200-permutation panel covers fixed spatial, receiver, and LR-identity nulls",
                "complete" if required_nulls == nulls and required_nulls.issubset(group_nulls) else "partial",
                _rel(summary_path),
                f"summary_nulls={sorted(nulls)}; group_nulls={sorted(group_nulls)}",
            ),
            AuditRow(
                "phase2_revised_goal5_fdr_200perm",
                "all 200-permutation summary rows record n_permutations=200",
                "complete"
                if len(summary) == 3 and set(summary["n_permutations"].astype(int)) == {200}
                else "partial",
                _rel(summary_path),
                f"n_permutations={sorted(set(summary['n_permutations'].astype(int))) if 'n_permutations' in summary else []}",
            ),
            AuditRow(
                "phase2_revised_goal5_fdr_200perm",
                "200-permutation manifest records reusable FDR framework module",
                "complete"
                if manifest.get("framework_module") == "src.spitialself.fdr.run_formal_fdr_panel"
                else "partial",
                _rel(manifest_path),
                f"framework_module={manifest.get('framework_module')}",
            ),
            AuditRow(
                "phase2_revised_goal5_fdr_200perm",
                "200-permutation p-value resolution improves over 20-permutation smoke",
                "complete"
                if float(summary["min_p_value"].min()) <= 1.0 / 201.0 + 1e-12
                else "partial",
                _rel(summary_path),
                f"min_p={float(summary['min_p_value'].min())}",
            ),
            AuditRow(
                "phase2_revised_goal5_fdr_200perm",
                "200-permutation group-level FDR reports LR-pair significant discoveries where supported",
                "complete"
                if int(group_summary["n_significant_groups"].sum()) >= 1
                and ((group_summary["group_level"].astype(str) == "lr_pair") & (group_summary["n_significant_groups"].astype(int) > 0)).any()
                else "partial",
                _rel(group_summary_path),
                f"total_significant_groups={int(group_summary['n_significant_groups'].sum())}",
            ),
            AuditRow(
                "phase2_revised_goal5_fdr_200perm",
                "200-permutation manifest preserves stronger-smoke caveat",
                "complete"
                if manifest.get("claim_level") == "stronger_fdr_200perm_smoke_not_manuscript_ready"
                else "partial",
                _rel(manifest_path),
                f"claim_level={manifest.get('claim_level')}",
            ),
        ]
    )
    expected_suffixes = (
        "candidate_edge",
        "lr_pair",
        "cell_type_pair",
        "significant_spatial_edges",
        "significant_lr_pairs",
        "significant_cell_type_pairs",
    )
    fdr_paths = manifest.get("fdr_paths", {})
    missing_paths = []
    for null_model in required_nulls:
        for suffix in expected_suffixes:
            key = f"{null_model}:{suffix}"
            path = Path(fdr_paths.get(key, ""))
            if not path.exists():
                missing_paths.append(key)
    rows.append(
        AuditRow(
            "phase2_revised_goal5_fdr_200perm",
            "200-permutation candidate, group, and significant-output CSVs exist for every fixed null",
            "complete" if not missing_paths else "missing",
            _rel(root),
            f"missing={missing_paths[:20]}",
        )
    )
    return rows


def _audit_goal5_real_background_fdr_panel() -> list[AuditRow]:
    import pandas as pd

    root = PROJECT_ROOT / "outputs/phase2_revised/goal5_real_background_fdr_panel"
    config_path = PROJECT_ROOT / "configs/experiments/phase2_revised_goal5_real_background_fdr_panel.yaml"
    script_path = PROJECT_ROOT / "scripts/phase2/run_goal5_real_background_fdr_panel.py"
    summary_path = root / "goal5_real_background_fdr_panel_summary.csv"
    group_summary_path = root / "goal5_real_background_fdr_panel_group_summary.csv"
    claims_path = root / "goal5_real_background_fdr_panel_claims.csv"
    status_path = root / "goal5_real_background_fdr_panel_status.csv"
    manifest_path = root / "goal5_real_background_fdr_panel_manifest.json"
    rows = [
        AuditRow("phase2_revised_goal5_real_background_fdr_panel", "real-background FDR panel config exists", _exists(config_path), _rel(config_path)),
        AuditRow("phase2_revised_goal5_real_background_fdr_panel", "real-background FDR panel runner exists", _exists(script_path), _rel(script_path)),
        AuditRow("phase2_revised_goal5_real_background_fdr_panel", "real-background FDR panel summary exists", _exists(summary_path), _rel(summary_path)),
        AuditRow("phase2_revised_goal5_real_background_fdr_panel", "real-background FDR panel group summary exists", _exists(group_summary_path), _rel(group_summary_path)),
        AuditRow("phase2_revised_goal5_real_background_fdr_panel", "real-background FDR panel claims exists", _exists(claims_path), _rel(claims_path)),
        AuditRow("phase2_revised_goal5_real_background_fdr_panel", "real-background FDR panel status exists", _exists(status_path), _rel(status_path)),
        AuditRow("phase2_revised_goal5_real_background_fdr_panel", "real-background FDR panel manifest exists", _exists(manifest_path), _rel(manifest_path)),
    ]
    if not all(path.exists() for path in [summary_path, group_summary_path, claims_path, status_path, manifest_path]):
        return rows

    summary = pd.read_csv(summary_path)
    group_summary = pd.read_csv(group_summary_path)
    claims = pd.read_csv(claims_path)
    status = pd.read_csv(status_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_datasets = {
        "real_background_v2_formal_lymph_node",
        "real_background_v2_formal_tonsil",
        "real_background_v2_formal_melanoma",
    }
    required_nulls = {"spatial_permutation", "receiver_permutation", "lr_identity_permutation"}
    datasets = set(summary["dataset"].astype(str)) if "dataset" in summary else set()
    nulls = set(summary["null_model"].astype(str)) if "null_model" in summary else set()
    rows.extend(
        [
            AuditRow(
                "phase2_revised_goal5_real_background_fdr_panel",
                "real-background FDR panel covers lymph node, tonsil, and melanoma formal datasets",
                "complete" if required_datasets.issubset(datasets) else "partial",
                _rel(summary_path),
                f"datasets={sorted(datasets)}",
            ),
            AuditRow(
                "phase2_revised_goal5_real_background_fdr_panel",
                "real-background FDR panel covers fixed spatial, receiver, and LR-identity nulls for each dataset",
                "complete"
                if required_nulls == nulls
                and all(
                    required_nulls
                    == set(summary.loc[summary["dataset"].astype(str) == dataset, "null_model"].astype(str))
                    for dataset in required_datasets
                )
                else "partial",
                _rel(summary_path),
                f"nulls={sorted(nulls)} rows={len(summary)}",
            ),
            AuditRow(
                "phase2_revised_goal5_real_background_fdr_panel",
                "real-background FDR panel uses 50 permutations and improves beyond schema smoke",
                "complete"
                if len(summary) >= 9 and set(summary["n_permutations"].astype(int)) == {50}
                else "partial",
                _rel(summary_path),
                f"n_permutations={sorted(set(summary['n_permutations'].astype(int))) if 'n_permutations' in summary else []}",
            ),
            AuditRow(
                "phase2_revised_goal5_real_background_fdr_panel",
                "real-background FDR panel writes LR-pair and cell-type-pair group summaries for every null",
                "complete"
                if len(group_summary) >= 18
                and {"lr_pair", "cell_type_pair"}.issubset(set(group_summary["group_level"].astype(str)))
                and required_nulls.issubset(set(group_summary["null_model"].astype(str)))
                else "partial",
                _rel(group_summary_path),
                f"rows={len(group_summary)} levels={sorted(set(group_summary['group_level'].astype(str))) if 'group_level' in group_summary else []}",
            ),
            AuditRow(
                "phase2_revised_goal5_real_background_fdr_panel",
                "real-background FDR panel reports group-level LR discoveries while preserving absent candidate-edge discovery caveat",
                "complete"
                if len(claims) >= 3
                and int(claims["lr_pair_direct_discoveries"].sum()) >= 3
                and int(claims["candidate_edge_discoveries"].sum()) == 0
                else "partial",
                _rel(claims_path),
                claims[["dataset", "candidate_edge_discoveries", "lr_pair_direct_discoveries", "cell_type_pair_direct_discoveries"]].to_json(orient="records") if len(claims) else "",
            ),
            AuditRow(
                "phase2_revised_goal5_real_background_fdr_panel",
                "real-background FDR panel completed all dataset runs",
                "complete"
                if len(status) >= 3 and (status["status"].astype(str) == "completed").all()
                else "partial",
                _rel(status_path),
                f"rows={len(status)} statuses={sorted(set(status['status'].astype(str))) if 'status' in status else []}",
            ),
            AuditRow(
                "phase2_revised_goal5_real_background_fdr_panel",
                "real-background FDR panel manifest records reusable framework and non-final caveat",
                "complete"
                if manifest.get("framework_module") == "src.spitialself.fdr.run_formal_fdr_panel"
                and manifest.get("claim_level") == "formal_real_background_50perm_fdr_panel_not_manuscript_ready"
                else "partial",
                _rel(manifest_path),
                f"framework_module={manifest.get('framework_module')}; claim_level={manifest.get('claim_level')}",
            ),
        ]
    )
    return rows


def _audit_goal5_real_background_fdr_100perm() -> list[AuditRow]:
    import pandas as pd

    root = PROJECT_ROOT / "outputs/phase2_revised/goal5_real_background_fdr_100perm"
    config_path = PROJECT_ROOT / "configs/experiments/phase2_revised_goal5_real_background_fdr_100perm.yaml"
    script_path = PROJECT_ROOT / "scripts/phase2/run_goal5_real_background_fdr_panel.py"
    summary_path = root / "goal5_real_background_fdr_panel_summary.csv"
    group_summary_path = root / "goal5_real_background_fdr_panel_group_summary.csv"
    claims_path = root / "goal5_real_background_fdr_panel_claims.csv"
    status_path = root / "goal5_real_background_fdr_panel_status.csv"
    manifest_path = root / "goal5_real_background_fdr_panel_manifest.json"
    rows = [
        AuditRow("phase2_revised_goal5_real_background_fdr_100perm", "100-permutation real-background FDR config exists", _exists(config_path), _rel(config_path)),
        AuditRow("phase2_revised_goal5_real_background_fdr_100perm", "100-permutation real-background FDR runner exists", _exists(script_path), _rel(script_path)),
        AuditRow("phase2_revised_goal5_real_background_fdr_100perm", "100-permutation real-background FDR summary exists", _exists(summary_path), _rel(summary_path)),
        AuditRow("phase2_revised_goal5_real_background_fdr_100perm", "100-permutation real-background FDR group summary exists", _exists(group_summary_path), _rel(group_summary_path)),
        AuditRow("phase2_revised_goal5_real_background_fdr_100perm", "100-permutation real-background FDR claims exists", _exists(claims_path), _rel(claims_path)),
        AuditRow("phase2_revised_goal5_real_background_fdr_100perm", "100-permutation real-background FDR status exists", _exists(status_path), _rel(status_path)),
        AuditRow("phase2_revised_goal5_real_background_fdr_100perm", "100-permutation real-background FDR manifest exists", _exists(manifest_path), _rel(manifest_path)),
    ]
    if not all(path.exists() for path in [summary_path, group_summary_path, claims_path, status_path, manifest_path]):
        return rows

    summary = pd.read_csv(summary_path)
    group_summary = pd.read_csv(group_summary_path)
    claims = pd.read_csv(claims_path)
    status = pd.read_csv(status_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_datasets = {
        "real_background_v2_formal_lymph_node",
        "real_background_v2_formal_tonsil",
        "real_background_v2_formal_melanoma",
    }
    required_nulls = {"spatial_permutation", "receiver_permutation", "lr_identity_permutation"}
    datasets = set(summary["dataset"].astype(str)) if "dataset" in summary else set()
    nulls = set(summary["null_model"].astype(str)) if "null_model" in summary else set()
    rows.extend(
        [
            AuditRow(
                "phase2_revised_goal5_real_background_fdr_100perm",
                "100-permutation FDR panel covers lymph node, tonsil, and melanoma formal datasets",
                "complete" if required_datasets.issubset(datasets) else "partial",
                _rel(summary_path),
                f"datasets={sorted(datasets)}",
            ),
            AuditRow(
                "phase2_revised_goal5_real_background_fdr_100perm",
                "100-permutation FDR panel covers fixed spatial, receiver, and LR-identity nulls for each dataset",
                "complete"
                if required_nulls == nulls
                and all(
                    required_nulls
                    == set(summary.loc[summary["dataset"].astype(str) == dataset, "null_model"].astype(str))
                    for dataset in required_datasets
                )
                else "partial",
                _rel(summary_path),
                f"nulls={sorted(nulls)} rows={len(summary)}",
            ),
            AuditRow(
                "phase2_revised_goal5_real_background_fdr_100perm",
                "100-permutation FDR panel improves p-value resolution beyond 50-permutation panel",
                "complete"
                if len(summary) >= 9
                and set(summary["n_permutations"].astype(int)) == {100}
                and float(summary["min_p_value"].min()) <= 1.0 / 101.0 + 1e-9
                else "partial",
                _rel(summary_path),
                f"n_permutations={sorted(set(summary['n_permutations'].astype(int))) if 'n_permutations' in summary else []}; min_p={float(summary['min_p_value'].min()) if 'min_p_value' in summary and len(summary) else 'NA'}",
            ),
            AuditRow(
                "phase2_revised_goal5_real_background_fdr_100perm",
                "100-permutation FDR panel writes LR-pair and cell-type-pair group summaries for every null",
                "complete"
                if len(group_summary) >= 18
                and {"lr_pair", "cell_type_pair"}.issubset(set(group_summary["group_level"].astype(str)))
                and required_nulls.issubset(set(group_summary["null_model"].astype(str)))
                else "partial",
                _rel(group_summary_path),
                f"rows={len(group_summary)} levels={sorted(set(group_summary['group_level'].astype(str))) if 'group_level' in group_summary else []}",
            ),
            AuditRow(
                "phase2_revised_goal5_real_background_fdr_100perm",
                "100-permutation FDR panel reports group-level LR discoveries while preserving absent candidate-edge discovery caveat",
                "complete"
                if len(claims) >= 3
                and int(claims["lr_pair_direct_discoveries"].sum()) >= 3
                and int(claims["candidate_edge_discoveries"].sum()) == 0
                else "partial",
                _rel(claims_path),
                claims[["dataset", "candidate_edge_discoveries", "lr_pair_direct_discoveries", "cell_type_pair_direct_discoveries"]].to_json(orient="records") if len(claims) else "",
            ),
            AuditRow(
                "phase2_revised_goal5_real_background_fdr_100perm",
                "100-permutation FDR panel completed all dataset runs",
                "complete"
                if len(status) >= 3 and (status["status"].astype(str) == "completed").all()
                else "partial",
                _rel(status_path),
                f"rows={len(status)} statuses={sorted(set(status['status'].astype(str))) if 'status' in status else []}",
            ),
            AuditRow(
                "phase2_revised_goal5_real_background_fdr_100perm",
                "100-permutation FDR panel manifest records reusable framework and non-final 500-permutation caveat",
                "complete"
                if manifest.get("framework_module") == "src.spitialself.fdr.run_formal_fdr_panel"
                and manifest.get("claim_level") == "representative_real_background_100perm_fdr_panel_not_final_500perm"
                else "partial",
                _rel(manifest_path),
                f"framework_module={manifest.get('framework_module')}; claim_level={manifest.get('claim_level')}",
            ),
        ]
    )
    return rows


def _audit_goal6_manuscript_assembly() -> list[AuditRow]:
    import pandas as pd

    root = PROJECT_ROOT / "outputs/phase2_revised/goal6_manuscript_assembly"
    source_index_path = root / "source_table_index.csv"
    claim_matrix_path = root / "claim_evidence_matrix.csv"
    limitations_path = root / "manuscript_limitations.csv"
    figure_plan_path = root / "figure_panel_plan.csv"
    manifest_path = root / "goal6_manuscript_assembly_manifest.json"
    rows = [
        AuditRow("phase2_revised_goal6_manuscript_assembly", "source table index exists", _exists(source_index_path), _rel(source_index_path)),
        AuditRow("phase2_revised_goal6_manuscript_assembly", "claim evidence matrix exists", _exists(claim_matrix_path), _rel(claim_matrix_path)),
        AuditRow("phase2_revised_goal6_manuscript_assembly", "limitations table exists", _exists(limitations_path), _rel(limitations_path)),
        AuditRow("phase2_revised_goal6_manuscript_assembly", "figure panel plan exists", _exists(figure_plan_path), _rel(figure_plan_path)),
        AuditRow("phase2_revised_goal6_manuscript_assembly", "manifest exists", _exists(manifest_path), _rel(manifest_path)),
    ]
    if not all(path.exists() for path in [source_index_path, claim_matrix_path, limitations_path, figure_plan_path, manifest_path]):
        return rows
    source_index = pd.read_csv(source_index_path)
    claim_matrix = pd.read_csv(claim_matrix_path)
    limitations = pd.read_csv(limitations_path)
    figure_plan = pd.read_csv(figure_plan_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows.extend(
        [
            AuditRow(
                "phase2_revised_goal6_manuscript_assembly",
                "all assembled source tables are copied, non-empty, and plot-ready",
                "complete"
                if len(source_index) >= 14
                and source_index["ready_for_plotting"].astype(bool).all()
                and (source_index["status"].astype(str) == "copied").all()
                else "partial",
                _rel(source_index_path),
                f"tables={len(source_index)} ready={int(source_index['ready_for_plotting'].astype(bool).sum())}",
            ),
            AuditRow(
                "phase2_revised_goal6_manuscript_assembly",
                "claim matrix covers simulator, real background, tradeoff, validation, FDR, and audit claims",
                "complete"
                if {
                    "C1_simulator_v2_mechanism_layered_truth",
                    "C2_real_background_v2_preserves_st_properties",
                    "C3_spatialself_fp_suppression_tradeoff",
                    "C4_baseline_matched_control_ranking",
                    "C5_formal_fdr_schema",
                    "C6_auditability_reproducibility",
                    "C6_runtime_scalability_support",
                }.issubset(set(claim_matrix["claim_id"].astype(str)))
                and set(claim_matrix["evidence_status"].astype(str)) == {"source_ready"}
                else "partial",
                _rel(claim_matrix_path),
                f"claims={len(claim_matrix)} statuses={sorted(set(claim_matrix['evidence_status'].astype(str)))}",
            ),
            AuditRow(
                "phase2_revised_goal6_manuscript_assembly",
                "source table assembly includes merged full-baseline all-panel tables",
                "complete"
                if {
                    "goal1_method_summary",
                    "goal1_lr_control_summary",
                    "goal1_claims",
                    "goal2_metrics",
                    "goal2_summary",
                    "goal2_claims",
                    "goal2_status",
                }.issubset(set(source_index["table_id"].astype(str)))
                and int(source_index.loc[source_index["table_id"].astype(str) == "goal2_metrics", "n_rows"].iloc[0]) >= 21
                and int(source_index.loc[source_index["table_id"].astype(str) == "goal1_method_summary", "n_rows"].iloc[0]) >= 18
                else "partial",
                _rel(source_index_path),
                f"tables={sorted(set(source_index['table_id'].astype(str)))}",
            ),
            AuditRow(
                "phase2_revised_goal6_manuscript_assembly",
                "source table assembly includes formal Goal1 baseline-matched validation tables",
                "complete"
                if {
                    "goal1_formal_method_summary",
                    "goal1_formal_lr_control_summary",
                    "goal1_formal_claims",
                    "goal1_formal_status",
                }.issubset(set(source_index["table_id"].astype(str)))
                and int(source_index.loc[source_index["table_id"].astype(str) == "goal1_formal_method_summary", "n_rows"].iloc[0]) >= 18
                and int(source_index.loc[source_index["table_id"].astype(str) == "goal1_formal_lr_control_summary", "n_rows"].iloc[0]) >= 18
                else "partial",
                _rel(source_index_path),
                f"tables={sorted(set(source_index['table_id'].astype(str)))}",
            ),
            AuditRow(
                "phase2_revised_goal6_manuscript_assembly",
                "source table assembly includes formal Goal1 injected downstream proxy tables",
                "complete"
                if {
                    "goal1_downstream_proxy_lr_summary",
                    "goal1_downstream_proxy_method_summary",
                    "goal1_downstream_proxy_claims",
                    "goal1_downstream_proxy_status",
                }.issubset(set(source_index["table_id"].astype(str)))
                and int(source_index.loc[source_index["table_id"].astype(str) == "goal1_downstream_proxy_method_summary", "n_rows"].iloc[0]) >= 18
                and int(source_index.loc[source_index["table_id"].astype(str) == "goal1_downstream_proxy_claims", "n_rows"].iloc[0]) >= 3
                else "partial",
                _rel(source_index_path),
                f"tables={sorted(set(source_index['table_id'].astype(str)))}",
            ),
            AuditRow(
                "phase2_revised_goal6_manuscript_assembly",
                "source table assembly includes formal Goal1 curated downstream panel tables",
                "complete"
                if {
                    "goal1_curated_downstream_results",
                    "goal1_curated_downstream_method_summary",
                    "goal1_curated_downstream_coverage",
                    "goal1_curated_downstream_claims",
                    "goal1_curated_downstream_status",
                }.issubset(set(source_index["table_id"].astype(str)))
                and int(source_index.loc[source_index["table_id"].astype(str) == "goal1_curated_downstream_method_summary", "n_rows"].iloc[0]) >= 18
                and int(source_index.loc[source_index["table_id"].astype(str) == "goal1_curated_downstream_claims", "n_rows"].iloc[0]) >= 3
                else "partial",
                _rel(source_index_path),
                f"tables={sorted(set(source_index['table_id'].astype(str)))}",
            ),
            AuditRow(
                "phase2_revised_goal6_manuscript_assembly",
                "source table assembly includes Goal1 biological validation readiness gate tables",
                "complete"
                if {
                    "goal1_biological_validation_dataset_gate",
                    "goal1_biological_validation_method_gate",
                    "goal1_biological_validation_lr_gate",
                    "goal1_biological_validation_claims",
                }.issubset(set(source_index["table_id"].astype(str)))
                and int(source_index.loc[source_index["table_id"].astype(str) == "goal1_biological_validation_dataset_gate", "n_rows"].iloc[0]) >= 3
                and int(source_index.loc[source_index["table_id"].astype(str) == "goal1_biological_validation_method_gate", "n_rows"].iloc[0]) >= 18
                and int(source_index.loc[source_index["table_id"].astype(str) == "goal1_biological_validation_claims", "n_rows"].iloc[0]) >= 4
                else "partial",
                _rel(source_index_path),
                f"tables={sorted(set(source_index['table_id'].astype(str)))}",
            ),
            AuditRow(
                "phase2_revised_goal6_manuscript_assembly",
                "source table assembly includes formal Goal 2 baseline panel tables",
                "complete"
                if {
                    "goal2_formal_metrics",
                    "goal2_formal_summary",
                    "goal2_formal_claims",
                    "goal2_formal_status",
                }.issubset(set(source_index["table_id"].astype(str)))
                and int(source_index.loc[source_index["table_id"].astype(str) == "goal2_formal_metrics", "n_rows"].iloc[0]) >= 18
                else "partial",
                _rel(source_index_path),
                f"tables={sorted(set(source_index['table_id'].astype(str)))}",
            ),
            AuditRow(
                "phase2_revised_goal6_manuscript_assembly",
                "source table assembly includes tuned-small guardrail sensitivity tables",
                "complete"
                if {
                    "goal2_tuned_guardrail_sensitivity",
                    "goal2_tuned_guardrail_sensitivity_summary",
                    "goal2_tuned_guardrail_sensitivity_claims",
                    "goal2_tuned_guardrail_sensitivity_status",
                }.issubset(set(source_index["table_id"].astype(str)))
                and int(source_index.loc[source_index["table_id"].astype(str) == "goal2_tuned_guardrail_sensitivity", "n_rows"].iloc[0]) >= 120
                and int(source_index.loc[source_index["table_id"].astype(str) == "goal2_tuned_guardrail_sensitivity_claims", "n_rows"].iloc[0]) >= 2
                else "partial",
                _rel(source_index_path),
                f"tables={sorted(set(source_index['table_id'].astype(str)))}",
            ),
            AuditRow(
                "phase2_revised_goal6_manuscript_assembly",
                "source table assembly includes preregistered endpoint policy tables",
                "complete"
                if {
                    "goal2_preregistered_endpoint_scores",
                    "goal2_preregistered_endpoint_summary",
                    "goal2_preregistered_endpoint_claims",
                    "goal2_preregistered_endpoint_policy_table",
                    "goal2_preregistered_endpoint_status",
                    "goal2_preregistered_fp_suppression_by_tissue",
                }.issubset(set(source_index["table_id"].astype(str)))
                and int(source_index.loc[source_index["table_id"].astype(str) == "goal2_preregistered_endpoint_scores", "n_rows"].iloc[0]) >= 36
                and int(source_index.loc[source_index["table_id"].astype(str) == "goal2_preregistered_endpoint_policy_table", "n_rows"].iloc[0]) >= 5
                else "partial",
                _rel(source_index_path),
                f"tables={sorted(set(source_index['table_id'].astype(str)))}",
            ),
            AuditRow(
                "phase2_revised_goal6_manuscript_assembly",
                "source table assembly includes preregistered paired diagnostic tables",
                "complete"
                if {
                    "goal2_preregistered_paired_deltas",
                    "goal2_preregistered_paired_summary",
                    "goal2_preregistered_paired_claims",
                    "goal2_preregistered_paired_status",
                }.issubset(set(source_index["table_id"].astype(str)))
                and int(source_index.loc[source_index["table_id"].astype(str) == "goal2_preregistered_paired_deltas", "n_rows"].iloc[0]) >= 12
                and int(source_index.loc[source_index["table_id"].astype(str) == "goal2_preregistered_paired_summary", "n_rows"].iloc[0]) >= 4
                else "partial",
                _rel(source_index_path),
                f"tables={sorted(set(source_index['table_id'].astype(str)))}",
            ),
            AuditRow(
                "phase2_revised_goal6_manuscript_assembly",
                "source table assembly includes replicate power planning tables",
                "complete"
                if {
                    "goal2_replicate_power_plan_by_baseline",
                    "goal2_replicate_power_plan_overall",
                    "goal2_replicate_power_plan_design",
                    "goal2_replicate_power_plan_claims",
                    "goal2_replicate_power_plan_status",
                }.issubset(set(source_index["table_id"].astype(str)))
                and int(source_index.loc[source_index["table_id"].astype(str) == "goal2_replicate_power_plan_by_baseline", "n_rows"].iloc[0]) >= 4
                and int(source_index.loc[source_index["table_id"].astype(str) == "goal2_replicate_power_plan_design", "n_rows"].iloc[0]) >= 5
                else "partial",
                _rel(source_index_path),
                f"tables={sorted(set(source_index['table_id'].astype(str)))}",
            ),
            AuditRow(
                "phase2_revised_goal6_manuscript_assembly",
                "source table assembly includes replicated tuned execution-plan tables",
                "complete"
                if {
                    "goal2_replicated_tuned_execution_plan",
                    "goal2_replicated_tuned_replicate_manifest",
                    "goal2_replicated_tuned_method_jobs",
                    "goal2_replicated_tuned_job_status_template",
                    "goal2_replicated_tuned_job_claims",
                }.issubset(set(source_index["table_id"].astype(str)))
                and int(source_index.loc[source_index["table_id"].astype(str) == "goal2_replicated_tuned_replicate_manifest", "n_rows"].iloc[0]) == 18
                and int(source_index.loc[source_index["table_id"].astype(str) == "goal2_replicated_tuned_method_jobs", "n_rows"].iloc[0]) == 144
                else "partial",
                _rel(source_index_path),
                f"tables={sorted(set(source_index['table_id'].astype(str)))}",
            ),
            AuditRow(
                "phase2_revised_goal6_manuscript_assembly",
                "source table assembly includes formal Goal2 weak-signal stress tables",
                "complete"
                if {
                    "goal2_weak_stress_metrics",
                    "goal2_weak_stress_summary",
                    "goal2_weak_stress_claims",
                    "goal2_weak_stress_status",
                }.issubset(set(source_index["table_id"].astype(str)))
                and int(source_index.loc[source_index["table_id"].astype(str) == "goal2_weak_stress_metrics", "n_rows"].iloc[0]) >= 48
                and int(source_index.loc[source_index["table_id"].astype(str) == "goal2_weak_stress_claims", "n_rows"].iloc[0]) >= 8
                else "partial",
                _rel(source_index_path),
                f"tables={sorted(set(source_index['table_id'].astype(str)))}",
            ),
            AuditRow(
                "phase2_revised_goal6_manuscript_assembly",
                "source table assembly includes Goal2 weak-signal tradeoff readiness gate tables",
                "complete"
                if {
                    "goal2_weak_tradeoff_method_gate",
                    "goal2_weak_tradeoff_tissue_gate",
                    "goal2_weak_tradeoff_stress_gate",
                    "goal2_weak_tradeoff_paired_gate",
                    "goal2_weak_tradeoff_claims",
                }.issubset(set(source_index["table_id"].astype(str)))
                and int(source_index.loc[source_index["table_id"].astype(str) == "goal2_weak_tradeoff_method_gate", "n_rows"].iloc[0]) >= 6
                and int(source_index.loc[source_index["table_id"].astype(str) == "goal2_weak_tradeoff_claims", "n_rows"].iloc[0]) >= 5
                else "partial",
                _rel(source_index_path),
                f"tables={sorted(set(source_index['table_id'].astype(str)))}",
            ),
            AuditRow(
                "phase2_revised_goal6_manuscript_assembly",
                "source table assembly includes formal Goal 3 and Goal 4 benchmark summaries",
                "complete"
                if {
                    "goal3_formal_benchmark_summary",
                    "goal3_formal_claims",
                    "goal4_formal_real_background_summary",
                    "goal4_formal_claims",
                }.issubset(set(source_index["table_id"].astype(str)))
                and int(source_index.loc[source_index["table_id"].astype(str) == "goal3_formal_benchmark_summary", "n_rows"].iloc[0]) >= 1
                and int(source_index.loc[source_index["table_id"].astype(str) == "goal4_formal_real_background_summary", "n_rows"].iloc[0]) >= 3
                else "partial",
                _rel(source_index_path),
                f"tables={sorted(set(source_index['table_id'].astype(str)))}",
            ),
            AuditRow(
                "phase2_revised_goal6_manuscript_assembly",
                "source table assembly includes replicated tuned real-background dataset summaries",
                "complete"
                if {
                    "goal4_replicated_tuned_dataset_status",
                    "goal4_replicated_tuned_validation_summary",
                    "goal4_replicated_tuned_real_background_summary",
                    "goal4_replicated_tuned_claims",
                    "goal4_replicated_tuned_manifest",
                }.issubset(set(source_index["table_id"].astype(str)))
                and int(source_index.loc[source_index["table_id"].astype(str) == "goal4_replicated_tuned_dataset_status", "n_rows"].iloc[0]) == 18
                and int(source_index.loc[source_index["table_id"].astype(str) == "goal4_replicated_tuned_validation_summary", "n_rows"].iloc[0]) == 18
                else "partial",
                _rel(source_index_path),
                f"tables={sorted(set(source_index['table_id'].astype(str)))}",
            ),
            AuditRow(
                "phase2_revised_goal6_manuscript_assembly",
                "source table assembly includes Goal3/4 benchmark quality report tables",
                "complete"
                if {
                    "goal34_benchmark_quality_summary",
                    "goal34_mechanism_coverage_matrix",
                    "goal34_real_background_property_summary",
                    "goal34_truth_contract_summary",
                    "goal34_benchmark_quality_claims",
                }.issubset(set(source_index["table_id"].astype(str)))
                and int(source_index.loc[source_index["table_id"].astype(str) == "goal34_benchmark_quality_summary", "n_rows"].iloc[0]) >= 4
                and int(source_index.loc[source_index["table_id"].astype(str) == "goal34_benchmark_quality_claims", "n_rows"].iloc[0]) >= 4
                else "partial",
                _rel(source_index_path),
                f"tables={sorted(set(source_index['table_id'].astype(str)))}",
            ),
            AuditRow(
                "phase2_revised_goal6_manuscript_assembly",
                "source table assembly includes runtime and high-resolution scaling support tables",
                "complete"
                if {
                    "goal6_runtime_method_profile",
                    "goal6_runtime_highres_scaling_runs",
                    "goal6_runtime_claims",
                }.issubset(set(source_index["table_id"].astype(str)))
                and int(source_index.loc[source_index["table_id"].astype(str) == "goal6_runtime_method_profile", "n_rows"].iloc[0]) >= 8
                and int(source_index.loc[source_index["table_id"].astype(str) == "goal6_runtime_highres_scaling_runs", "n_rows"].iloc[0]) >= 3
                else "partial",
                _rel(source_index_path),
                f"tables={sorted(set(source_index['table_id'].astype(str)))}",
            ),
            AuditRow(
                "phase2_revised_goal6_manuscript_assembly",
                "source table assembly includes real-background multi-dataset FDR panel tables",
                "complete"
                if {
	                    "goal5_real_background_fdr_summary",
	                    "goal5_real_background_fdr_group_summary",
	                    "goal5_real_background_fdr_claims",
	                    "goal5_real_background_fdr_status",
	                    "goal5_real_background_fdr_100perm_summary",
	                    "goal5_real_background_fdr_100perm_group_summary",
	                    "goal5_real_background_fdr_100perm_claims",
	                    "goal5_real_background_fdr_100perm_status",
	                    "goal5_fixed_null_cli_smoke_summary",
	                    "goal5_fixed_null_cli_smoke_group_summary",
	                }.issubset(set(source_index["table_id"].astype(str)))
	                and int(source_index.loc[source_index["table_id"].astype(str) == "goal5_real_background_fdr_summary", "n_rows"].iloc[0]) >= 9
	                and int(source_index.loc[source_index["table_id"].astype(str) == "goal5_real_background_fdr_claims", "n_rows"].iloc[0]) >= 3
	                and int(source_index.loc[source_index["table_id"].astype(str) == "goal5_real_background_fdr_100perm_summary", "n_rows"].iloc[0]) >= 9
	                and int(source_index.loc[source_index["table_id"].astype(str) == "goal5_real_background_fdr_100perm_claims", "n_rows"].iloc[0]) >= 3
	                and int(source_index.loc[source_index["table_id"].astype(str) == "goal5_fixed_null_cli_smoke_summary", "n_rows"].iloc[0]) == 3
	                else "partial",
                _rel(source_index_path),
                f"tables={sorted(set(source_index['table_id'].astype(str)))}",
            ),
            AuditRow(
                "phase2_revised_goal6_manuscript_assembly",
                "claim matrix records adapter-aware FP/FDR-constrained baseline wording",
                "complete"
	                if _claim_text_contains(
	                    claim_matrix,
	                    "C3_spatialself_fp_suppression_tradeoff",
	                    [
	                        "FP/FDR-constrained",
	                        "CellChat",
	                        "SpaTalk",
	                        "CellNEST",
	                        "DeepTalk",
	                        "NICHES",
	                        "Goal2 readiness gate",
	                        "mechanism-stratified stress gate",
	                        "aggregate frozen endpoint",
	                        "highest raw weak recall",
	                        "tissue-level dominance",
	                        "paired significance",
	                        "method-mean lowest FP",
	                        "highest raw weak recall favors CellNEST",
	                    ],
	                )
                else "partial",
                _rel(claim_matrix_path),
                _claim_text(claim_matrix, "C3_spatialself_fp_suppression_tradeoff"),
            ),
            AuditRow(
                "phase2_revised_goal6_manuscript_assembly",
                "limitations table records remaining manuscript risks",
                "complete" if len(limitations) >= 5 else "partial",
                _rel(limitations_path),
                f"limitations={len(limitations)}",
            ),
            AuditRow(
                "phase2_revised_goal6_manuscript_assembly",
                "figure panel plan has source-ready panels and preserves FDR smoke caveat",
                "complete"
                if len(figure_plan) >= 6
                and (
                    "source_ready_smoke_only" in set(figure_plan["status"].astype(str))
                    or "source_ready_200perm_smoke" in set(figure_plan["status"].astype(str))
                    or "source_ready_fixed_null_cli_and_200perm_smoke" in set(figure_plan["status"].astype(str))
                    or "source_ready_fixed_null_cli_200perm_smoke_and_100perm_representative_panel"
                    in set(figure_plan["status"].astype(str))
                )
                else "partial",
                _rel(figure_plan_path),
                f"panels={len(figure_plan)} statuses={sorted(set(figure_plan['status'].astype(str)))}",
            ),
            AuditRow(
                "phase2_revised_goal6_manuscript_assembly",
                "manifest records source-table assembly rather than final figure bundle",
                "complete"
                if manifest.get("claim_level") == "source_table_assembly_not_final_figure_bundle"
                else "partial",
                _rel(manifest_path),
                f"claim_level={manifest.get('claim_level')}",
            ),
        ]
    )
    return rows


def _audit_goal6_results_synthesis() -> list[AuditRow]:
    import pandas as pd

    root = PROJECT_ROOT / "outputs/phase2_revised/goal6_manuscript_assembly"
    script_path = PROJECT_ROOT / "scripts/phase2/run_goal6_results_synthesis.py"
    outline_path = root / "manuscript_results_outline.csv"
    gap_path = root / "manuscript_gap_priority.csv"
    readiness_path = root / "manuscript_readiness_summary.csv"
    guardrail_path = root / "manuscript_wording_guardrails.csv"
    manifest_path = root / "manuscript_results_synthesis_manifest.json"
    rows = [
        AuditRow("phase2_revised_goal6_results_synthesis", "results synthesis runner exists", _exists(script_path), _rel(script_path)),
        AuditRow("phase2_revised_goal6_results_synthesis", "manuscript results outline exists", _exists(outline_path), _rel(outline_path)),
        AuditRow("phase2_revised_goal6_results_synthesis", "manuscript gap priority table exists", _exists(gap_path), _rel(gap_path)),
        AuditRow("phase2_revised_goal6_results_synthesis", "manuscript readiness summary exists", _exists(readiness_path), _rel(readiness_path)),
        AuditRow("phase2_revised_goal6_results_synthesis", "manuscript wording guardrails exists", _exists(guardrail_path), _rel(guardrail_path)),
        AuditRow("phase2_revised_goal6_results_synthesis", "manuscript results synthesis manifest exists", _exists(manifest_path), _rel(manifest_path)),
    ]
    if not all(path.exists() for path in [outline_path, gap_path, readiness_path, guardrail_path, manifest_path]):
        return rows

    outline = pd.read_csv(outline_path)
    gaps = pd.read_csv(gap_path)
    readiness = pd.read_csv(readiness_path)
    guardrails = pd.read_csv(guardrail_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_results = {"R1", "R2", "R3", "R4", "R5"}
    rows.extend(
        [
            AuditRow(
                "phase2_revised_goal6_results_synthesis",
                "results outline covers benchmark, tradeoff, validation, FDR, and evidence-chain sections",
                "complete"
                if required_results.issubset(set(outline["result_id"].astype(str)))
                and outline["ready_for_main_text"].astype(bool).all()
                else "partial",
                _rel(outline_path),
                f"results={sorted(set(outline['result_id'].astype(str))) if 'result_id' in outline else []}",
            ),
            AuditRow(
                "phase2_revised_goal6_results_synthesis",
                "gap priority table ranks downstream, baseline, FDR, scale, runtime, and figure gaps",
                "complete"
                if {"L3_downstream_proxy", "L2_baseline_panel_incomplete", "L4_fdr_power", "L1_smoke_scale", "L6_runtime_scope", "L5_no_final_figures"}.issubset(set(gaps["limitation_id"].astype(str)))
                and int(gaps["priority_rank"].min()) == 1
                else "partial",
                _rel(gap_path),
                gaps[["limitation_id", "priority_rank", "severity"]].to_json(orient="records") if len(gaps) else "",
            ),
            AuditRow(
                "phase2_revised_goal6_results_synthesis",
                "readiness summary marks scaffold ready but manuscript final not ready",
                "complete"
                if len(readiness) == 1
                and bool(readiness.iloc[0]["main_text_scaffold_ready"])
                and not bool(readiness.iloc[0]["manuscript_final_ready"])
                and int(readiness.iloc[0]["source_ready_claims"]) == int(readiness.iloc[0]["total_claims"])
                else "partial",
                _rel(readiness_path),
                readiness.to_json(orient="records") if len(readiness) else "",
            ),
            AuditRow(
                "phase2_revised_goal6_results_synthesis",
                "wording guardrails include weak-recall, downstream, FDR, and high-resolution overclaim blocks",
                "complete"
                if _guardrails_contain(
                    guardrails,
	                    [
	                        "highest weak recall",
	                        "dominates guardrail sensitivity",
	                        "paired significance",
	                        "power proof",
	                        "global superiority",
	                        "endpoint rules",
	                        "tissue-specific winners",
	                        "downstream-superiority",
	                        "500-permutation",
                        "100k",
                    ],
                )
                else "partial",
                _rel(guardrail_path),
                f"rows={len(guardrails)}",
            ),
            AuditRow(
                "phase2_revised_goal6_results_synthesis",
                "results synthesis manifest preserves scaffold-only caveat",
                "complete"
                if manifest.get("claim_level") == "manuscript_scaffold_not_final_text"
                else "partial",
                _rel(manifest_path),
                f"claim_level={manifest.get('claim_level')}",
            ),
        ]
    )
    return rows


def _audit_phase2_evidence_lock() -> list[AuditRow]:
    import pandas as pd

    root = PROJECT_ROOT / "outputs/phase2_revised/goal6_manuscript_assembly"
    script_path = PROJECT_ROOT / "scripts/phase2/run_phase2_evidence_lock.py"
    goal_lock_path = root / "phase2_goal_evidence_lock.csv"
    guardrail_path = root / "phase2_claim_guardrail_lock.csv"
    actions_path = root / "phase2_remaining_actions.csv"
    completion_path = root / "phase2_goal_completion_matrix.csv"
    manifest_path = root / "phase2_evidence_lock_manifest.json"
    rows = [
        AuditRow("phase2_revised_evidence_lock", "evidence lock runner exists", _exists(script_path), _rel(script_path)),
        AuditRow("phase2_revised_evidence_lock", "goal evidence lock exists", _exists(goal_lock_path), _rel(goal_lock_path)),
        AuditRow("phase2_revised_evidence_lock", "claim guardrail lock exists", _exists(guardrail_path), _rel(guardrail_path)),
        AuditRow("phase2_revised_evidence_lock", "remaining actions table exists", _exists(actions_path), _rel(actions_path)),
        AuditRow("phase2_revised_evidence_lock", "goal completion matrix exists", _exists(completion_path), _rel(completion_path)),
        AuditRow("phase2_revised_evidence_lock", "evidence lock manifest exists", _exists(manifest_path), _rel(manifest_path)),
    ]
    if not all(path.exists() for path in [goal_lock_path, guardrail_path, actions_path, completion_path, manifest_path]):
        return rows

    goal_lock = pd.read_csv(goal_lock_path)
    guardrails = pd.read_csv(guardrail_path)
    actions = pd.read_csv(actions_path)
    completion = pd.read_csv(completion_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_goals = {"Goal 1", "Goal 2", "Goal 3", "Goal 4", "Goal 5", "Goal 6"}
    expected_priority = {
        "Goal 3": 1,
        "Goal 4": 2,
        "Goal 1": 3,
        "Goal 5": 4,
        "Goal 2": 5,
        "Goal 6": 6,
    }
    priority_lookup = {
        str(row.goal_id): int(row.priority_order)
        for row in goal_lock.itertuples(index=False)
        if hasattr(row, "goal_id") and hasattr(row, "priority_order")
    }
    rows.extend(
        [
            AuditRow(
                "phase2_revised_evidence_lock",
                "evidence lock represents exactly the six canonical revised Phase 2 goals",
                "complete" if set(goal_lock["goal_id"].astype(str)) == required_goals and len(goal_lock) == 6 else "partial",
                _rel(goal_lock_path),
                f"goals={sorted(set(goal_lock['goal_id'].astype(str))) if 'goal_id' in goal_lock else []}",
            ),
            AuditRow(
                "phase2_revised_evidence_lock",
                "evidence lock preserves requested execution priority order",
                "complete" if priority_lookup == expected_priority else "partial",
                _rel(goal_lock_path),
                json.dumps(priority_lookup, ensure_ascii=True, sort_keys=True),
            ),
            AuditRow(
                "phase2_revised_evidence_lock",
                "all locked goals have source-table gates passing",
                "complete" if set(goal_lock["source_table_gate"].astype(str)) == {"pass"} else "partial",
                _rel(goal_lock_path),
                goal_lock[["goal_id", "source_table_gate", "missing_assembled_source_tables"]].to_json(orient="records") if len(goal_lock) else "",
            ),
            AuditRow(
                "phase2_revised_evidence_lock",
                "Goal 1 lock keeps downstream superiority unsupported",
                "complete"
                if _table_text_contains(
                    goal_lock[goal_lock["goal_id"].astype(str) == "Goal 1"],
                    ["downstream-superiority", "coverage-limited", "SpaTalk"],
                )
                else "partial",
                _rel(goal_lock_path),
                goal_lock[goal_lock["goal_id"].astype(str) == "Goal 1"].to_json(orient="records"),
            ),
            AuditRow(
                "phase2_revised_evidence_lock",
                "Goal 2 lock frames weak signal as tradeoff and blocks highest weak-recall claim",
                "complete"
                if _table_text_contains(
                    goal_lock[goal_lock["goal_id"].astype(str) == "Goal 2"],
                    ["tradeoff", "highest weak recall", "tissue winners remain mixed", "paired significance"],
                )
                else "partial",
                _rel(goal_lock_path),
                goal_lock[goal_lock["goal_id"].astype(str) == "Goal 2"].to_json(orient="records"),
            ),
            AuditRow(
                "phase2_revised_evidence_lock",
                "Goal 3 and Goal 4 locks preserve truth-contract and real-background distinctions",
                "complete"
                if _table_text_contains(
                    goal_lock[goal_lock["goal_id"].astype(str).isin(["Goal 3", "Goal 4"])],
                    ["truth-table contract", "Real ST backgrounds", "True unlabeled real-data ground truth", "Fully realistic simulator"],
                )
                else "partial",
                _rel(goal_lock_path),
                goal_lock[goal_lock["goal_id"].astype(str).isin(["Goal 3", "Goal 4"])].to_json(orient="records"),
            ),
            AuditRow(
                "phase2_revised_evidence_lock",
                "Goal 5 lock records FDR schema and blocks final 500-permutation overclaim",
                "complete"
                if _table_text_contains(
                    goal_lock[goal_lock["goal_id"].astype(str) == "Goal 5"],
                    ["fixed three-null", "100-permutation", "500-permutation", "candidate-edge discovery power"],
                )
                else "partial",
                _rel(goal_lock_path),
                goal_lock[goal_lock["goal_id"].astype(str) == "Goal 5"].to_json(orient="records"),
            ),
            AuditRow(
                "phase2_revised_evidence_lock",
                "guardrail lock includes claim-matrix and goal-level wording blocks",
                "complete"
                if {"claim_evidence_matrix", "phase2_goal_lock"}.issubset(set(guardrails["source"].astype(str)))
                and _table_text_contains(
                    guardrails,
                    ["highest weak recall", "downstream target superiority", "500-permutation", "100k"],
                )
                else "partial",
                _rel(guardrail_path),
                f"sources={sorted(set(guardrails['source'].astype(str))) if 'source' in guardrails else []}",
            ),
            AuditRow(
                "phase2_revised_evidence_lock",
                "remaining actions include benchmark, real-background, validation, FDR, weak-signal, and figure convergence work",
                "complete"
                if required_goals.issubset(set(actions["scope"].astype(str)))
                and bool(actions["blocks_final_manuscript"].astype(bool).any())
                else "partial",
                _rel(actions_path),
                actions[["scope", "priority_order", "severity", "blocks_final_manuscript"]].to_json(orient="records") if len(actions) else "",
            ),
            AuditRow(
                "phase2_revised_evidence_lock",
                "completion matrix states goals are locked but not experimentally complete/final manuscript ready",
                "complete"
                if set(completion["goal_id"].astype(str)) == required_goals
                and completion["phase2_goal_locked"].astype(bool).all()
                and not completion["manuscript_final_ready"].astype(bool).any()
                and set(completion["completion_interpretation"].astype(str)) == {"goal_definition_and_current_evidence_locked_not_experimentally_complete"}
                else "partial",
                _rel(completion_path),
                completion.to_json(orient="records") if len(completion) else "",
            ),
            AuditRow(
                "phase2_revised_evidence_lock",
                "manifest preserves evidence-lock-not-completion scope",
                "complete"
                if manifest.get("claim_level") == "phase2_goal_lock_not_phase2_completion"
                and int(manifest.get("canonical_goal_count", 0)) == 6
                else "partial",
                _rel(manifest_path),
                f"claim_level={manifest.get('claim_level')}; canonical_goal_count={manifest.get('canonical_goal_count')}",
            ),
        ]
    )
    return rows


def _audit_goal6_revised_figures() -> list[AuditRow]:
    import pandas as pd

    root = PROJECT_ROOT / "outputs/phase2_revised/goal6_manuscript_assembly/figures"
    manifest_path = root / "phase2_revised_figure_manifest.json"
    contract_path = root / "figure_contract.json"
    qa_summary_path = root / "figure_qa_summary.csv"
    qa_detail_path = root / "figure_qa_detail.csv"
    script_path = PROJECT_ROOT / "scripts/phase2/run_goal6_revised_figures.py"
    rows = [
        AuditRow("phase2_revised_goal6_figures", "revised figure script exists", _exists(script_path), _rel(script_path)),
        AuditRow("phase2_revised_goal6_figures", "figure manifest exists", _exists(manifest_path), _rel(manifest_path)),
        AuditRow("phase2_revised_goal6_figures", "figure contract exists", _exists(contract_path), _rel(contract_path)),
        AuditRow("phase2_revised_goal6_figures", "figure QA summary exists", _exists(qa_summary_path), _rel(qa_summary_path)),
        AuditRow("phase2_revised_goal6_figures", "figure QA detail exists", _exists(qa_detail_path), _rel(qa_detail_path)),
    ]
    if not all(path.exists() for path in [manifest_path, contract_path, qa_summary_path, qa_detail_path]):
        return rows

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    qa_summary = pd.read_csv(qa_summary_path)
    qa_detail = pd.read_csv(qa_detail_path)
    figure_rows = qa_detail[qa_detail["artifact_type"].astype(str) == "figure"]
    source_rows = qa_detail[qa_detail["artifact_type"].astype(str) == "source_table"]
    formats = set(figure_rows["format"].astype(str))
    rows.extend(
        [
            AuditRow(
                "phase2_revised_goal6_figures",
                "figure bundle exports SVG, PDF, TIFF, and PNG",
                "complete" if {"svg", "pdf", "tiff", "png"}.issubset(formats) else "partial",
                _rel(qa_detail_path),
                f"formats={sorted(formats)}",
            ),
            AuditRow(
                "phase2_revised_goal6_figures",
                "figure QA has no failed rows",
                "complete"
                if not qa_detail.empty
                and "qa_pass" in qa_detail
                and qa_detail["qa_pass"].astype(bool).all()
                and int(qa_summary.iloc[0].get("n_failed_qa_rows", 1)) == 0
                else "partial",
                _rel(qa_summary_path),
                f"rows={len(qa_detail)} failed={int((~qa_detail['qa_pass'].astype(bool)).sum()) if 'qa_pass' in qa_detail else 'NA'}",
            ),
            AuditRow(
                "phase2_revised_goal6_figures",
                "figure contract preserves smoke/source-ready claim scope",
                "complete"
                if contract.get("archetype") == "quantitative grid"
                and contract.get("backend") == "python_matplotlib"
                and "smoke scale is not manuscript-final evidence" in contract.get("review_risks", [])
                else "partial",
                _rel(contract_path),
                f"backend={contract.get('backend')} archetype={contract.get('archetype')}",
            ),
            AuditRow(
                "phase2_revised_goal6_figures",
                "figure manifest records non-final manuscript status",
                "complete"
                if manifest.get("evidence_status") == "source_ready_smoke_figures_not_final_manuscript"
                else "partial",
                _rel(manifest_path),
                f"evidence_status={manifest.get('evidence_status')}",
            ),
            AuditRow(
                "phase2_revised_goal6_figures",
                "figure source tables are present and non-empty",
                "complete"
                if len(source_rows) >= 13 and source_rows["exists"].astype(bool).all() and source_rows["non_empty"].astype(bool).all()
                else "partial",
                _rel(qa_detail_path),
                f"sources={len(source_rows)}",
            ),
        ]
    )
    return rows


def _audit_goal6_runtime_scalability() -> list[AuditRow]:
    import pandas as pd

    root = PROJECT_ROOT / "outputs/phase2_revised/goal7_runtime_scalability"
    script_path = PROJECT_ROOT / "scripts/phase2/run_goal7_runtime_scalability_summary.py"
    method_path = root / "goal7_method_runtime_profile.csv"
    highres_run_path = root / "goal7_highres_scaling_runs.csv"
    highres_summary_path = root / "goal7_highres_scaling_summary.csv"
    claims_path = root / "goal7_runtime_scalability_claims.csv"
    manifest_path = root / "goal7_runtime_scalability_manifest.json"
    rows = [
        AuditRow("phase2_revised_goal6_runtime_scalability_support", "runtime scalability summary script exists", _exists(script_path), _rel(script_path)),
        AuditRow("phase2_revised_goal6_runtime_scalability_support", "method runtime profile exists", _exists(method_path), _rel(method_path)),
        AuditRow("phase2_revised_goal6_runtime_scalability_support", "high-resolution scaling run table exists", _exists(highres_run_path), _rel(highres_run_path)),
        AuditRow("phase2_revised_goal6_runtime_scalability_support", "high-resolution scaling summary exists", _exists(highres_summary_path), _rel(highres_summary_path)),
        AuditRow("phase2_revised_goal6_runtime_scalability_support", "runtime scalability claims table exists", _exists(claims_path), _rel(claims_path)),
        AuditRow("phase2_revised_goal6_runtime_scalability_support", "runtime scalability manifest exists", _exists(manifest_path), _rel(manifest_path)),
    ]
    if not all(path.exists() for path in [method_path, highres_run_path, highres_summary_path, claims_path, manifest_path]):
        return rows
    method = pd.read_csv(method_path)
    highres_runs = pd.read_csv(highres_run_path)
    claims = pd.read_csv(claims_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completed_methods = set(method.loc[method["status"].astype(str) == "completed", "method"].astype(str))
    claim = claims.iloc[0] if len(claims) else {}
    rows.extend(
        [
            AuditRow(
                "phase2_revised_goal6_runtime_scalability_support",
                "runtime profile includes SpatialSelf, COMMOT, CellChat, SpaTalk, CellNEST, and DeepTalk completed rows",
                "complete"
                if {
                    "spatialself_wavelet",
                    "spatialself_wavelet_edge_prior",
                    "spatialself_gat",
                    "commot",
                    "cellchat",
                    "spatalk",
                    "cellnest",
                    "deeptalk",
                }.issubset(completed_methods)
                else "partial",
                _rel(method_path),
                f"completed={sorted(completed_methods)}",
            ),
            AuditRow(
                "phase2_revised_goal6_runtime_scalability_support",
                "runtime profile records wall-clock and CPU memory for SpatialSelf edge-prior",
                "complete"
                if float(claim.get("spatialself_edge_prior_wall_clock_seconds", 0)) > 0
                and float(claim.get("spatialself_edge_prior_peak_cpu_memory_mb", 0)) > 0
                else "partial",
                _rel(claims_path),
                f"seconds={claim.get('spatialself_edge_prior_wall_clock_seconds', 'NA')}; rss={claim.get('spatialself_edge_prior_peak_cpu_memory_mb', 'NA')}",
            ),
            AuditRow(
                "phase2_revised_goal6_runtime_scalability_support",
                "runtime profile records SpatialSelf edge-prior speedup over COMMOT",
                "complete"
                if float(claim.get("spatialself_edge_prior_speedup_vs_commot", 0)) > 1.0
                else "partial",
                _rel(claims_path),
                f"speedup={claim.get('spatialself_edge_prior_speedup_vs_commot', 'NA')}",
            ),
            AuditRow(
                "phase2_revised_goal6_runtime_scalability_support",
                "high-resolution tonsil subset scaling reaches at least 1,800 cells and 1M candidate edges",
                "complete"
                if int(claim.get("highres_n_cells_max", 0)) >= 1800
                and int(claim.get("highres_n_candidate_edges_max", 0)) >= 1_000_000
                and len(highres_runs) >= 3
                else "partial",
                _rel(highres_run_path),
                f"n_cells_max={claim.get('highres_n_cells_max', 'NA')}; edges_max={claim.get('highres_n_candidate_edges_max', 'NA')}; runs={len(highres_runs)}",
            ),
            AuditRow(
                "phase2_revised_goal6_runtime_scalability_support",
                "runtime scalability manifest preserves non-final 100k benchmark caveat",
                "complete"
                if manifest.get("evidence_status") == "supporting_runtime_scalability_not_final_100k_benchmark"
                and manifest.get("canonical_goal") == "goal6_manuscript_evidence_figure_convergence"
                else "partial",
                _rel(manifest_path),
                f"evidence_status={manifest.get('evidence_status')}; canonical_goal={manifest.get('canonical_goal')}",
            ),
        ]
    )
    return rows


def _audit_goal2_weak_signal_tradeoff_readiness() -> list[AuditRow]:
    import pandas as pd

    root = PROJECT_ROOT / "outputs/phase2_revised/goal2_weak_signal_tradeoff_readiness"
    script_path = PROJECT_ROOT / "scripts/phase2/run_goal2_weak_signal_tradeoff_readiness.py"
    method_path = root / "goal2_weak_signal_tradeoff_method_gate.csv"
    tissue_path = root / "goal2_weak_signal_tradeoff_tissue_gate.csv"
    stress_path = root / "goal2_weak_signal_tradeoff_stress_gate.csv"
    paired_path = root / "goal2_weak_signal_tradeoff_paired_gate.csv"
    claims_path = root / "goal2_weak_signal_tradeoff_claims.csv"
    manifest_path = root / "goal2_weak_signal_tradeoff_readiness_manifest.json"
    rows = [
        AuditRow("phase2_revised_goal2_weak_signal_tradeoff_readiness", "weak-signal tradeoff readiness runner exists", _exists(script_path), _rel(script_path)),
        AuditRow("phase2_revised_goal2_weak_signal_tradeoff_readiness", "method tradeoff gate exists", _exists(method_path), _rel(method_path)),
        AuditRow("phase2_revised_goal2_weak_signal_tradeoff_readiness", "tissue tradeoff gate exists", _exists(tissue_path), _rel(tissue_path)),
        AuditRow("phase2_revised_goal2_weak_signal_tradeoff_readiness", "stress tradeoff gate exists", _exists(stress_path), _rel(stress_path)),
        AuditRow("phase2_revised_goal2_weak_signal_tradeoff_readiness", "paired tradeoff gate exists", _exists(paired_path), _rel(paired_path)),
        AuditRow("phase2_revised_goal2_weak_signal_tradeoff_readiness", "tradeoff readiness claims exists", _exists(claims_path), _rel(claims_path)),
        AuditRow("phase2_revised_goal2_weak_signal_tradeoff_readiness", "tradeoff readiness manifest exists", _exists(manifest_path), _rel(manifest_path)),
    ]
    if not all(path.exists() for path in [method_path, tissue_path, stress_path, paired_path, claims_path, manifest_path]):
        return rows

    method = pd.read_csv(method_path)
    tissue = pd.read_csv(tissue_path)
    stress = pd.read_csv(stress_path)
    paired = pd.read_csv(paired_path)
    claims = pd.read_csv(claims_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    claim_status = dict(zip(claims["claim_id"].astype(str), claims["status"].astype(str)))
    rows.extend(
        [
            AuditRow(
                "phase2_revised_goal2_weak_signal_tradeoff_readiness",
                "readiness gate captures replicated-panel method mean caveats",
                "complete"
                if len(method) >= 6
                and not bool(method["spatialself_has_highest_raw_weak_recall"].iloc[0])
                and not bool(method["spatialself_has_lowest_fp_overall"].iloc[0])
                and not bool(method["spatialself_has_best_method_mean_tradeoff"].iloc[0])
                and bool(method["spatialself_has_best_aupr"].iloc[0])
                and bool(method["spatialself_has_best_topk_precision"].iloc[0])
                else "partial",
                _rel(method_path),
                method[
                    [
                        "method",
                        "weak_signal_recall_at_threshold_mean",
                        "false_positive_high_expression_rate_at_threshold_mean",
                        "weak_fp_tradeoff_score",
                    ]
                ].to_json(orient="records") if len(method) else "",
            ),
            AuditRow(
                "phase2_revised_goal2_weak_signal_tradeoff_readiness",
                "readiness gate captures mixed tissue-level winners",
                "complete"
                if len(tissue) >= 3
                and not tissue["spatialself_endpoint_winner"].astype(bool).all()
                and not tissue["spatialself_lowest_fp"].astype(bool).any()
                else "partial",
                _rel(tissue_path),
                tissue[["dataset_key", "endpoint_winner", "spatialself_endpoint_winner", "lowest_fp_method"]].to_json(orient="records") if len(tissue) else "",
            ),
            AuditRow(
                "phase2_revised_goal2_weak_signal_tradeoff_readiness",
                "readiness gate supports mechanism-stratified SpatialSelf guardrailed tradeoff",
                "complete"
                if len(stress) >= 8
                and stress["spatialself_tradeoff_supported"].astype(bool).all()
                and stress["raw_weak_recall_caveat"].astype(bool).any()
                and stress["lowest_fp_caveat"].astype(bool).any()
                else "partial",
                _rel(stress_path),
                stress[["dataset", "mechanism", "best_guardrailed_tradeoff_method", "highest_weak_recall_method", "lowest_high_expression_fp_method"]].to_json(orient="records") if len(stress) else "",
            ),
            AuditRow(
                "phase2_revised_goal2_weak_signal_tradeoff_readiness",
                "readiness gate preserves paired diagnostic CI limitation",
                "complete"
                if len(paired) >= 4
                and paired["positive_mean_delta"].astype(bool).all()
                and paired["two_thirds_positive"].astype(bool).all()
                and not paired["ci_above_zero"].astype(bool).all()
                else "partial",
                _rel(paired_path),
                paired[["baseline_method", "mean_delta", "positive_delta_fraction", "bootstrap_ci_low", "bootstrap_ci_high"]].to_json(orient="records") if len(paired) else "",
            ),
            AuditRow(
                "phase2_revised_goal2_weak_signal_tradeoff_readiness",
                "readiness claims separate supported tradeoff from unsupported raw recall and dominance claims",
                "complete"
                if claim_status.get("G2_C1_weak_fp_tradeoff") == "supported"
                and claim_status.get("G2_C2_raw_weak_recall") == "unsupported"
                and claim_status.get("G2_C3_false_positive_suppression") == "partial"
                and claim_status.get("G2_C4_tissue_dominance") == "unsupported"
                and claim_status.get("G2_C5_paired_diagnostics") == "partial"
                else "partial",
                _rel(claims_path),
                claims[["claim_id", "status", "allowed_wording", "disallowed_wording"]].to_json(orient="records") if len(claims) else "",
            ),
            AuditRow(
                "phase2_revised_goal2_weak_signal_tradeoff_readiness",
                "readiness manifest preserves no-global-dominance caveat",
                "complete"
                if manifest.get("claim_level") == "goal2_tradeoff_gate_not_global_dominance"
                else "partial",
                _rel(manifest_path),
                f"claim_level={manifest.get('claim_level')}",
            ),
        ]
    )
    return rows


def _audit_goal3_simulator_v2() -> list[AuditRow]:
    root = PROJECT_ROOT / "outputs/phase2_revised/goal3_simulator_v2/datasets"
    manifest = root / "manifest.csv"
    rows = [
        AuditRow(
            "phase2_revised_goal3_simulator_v2",
            "simulator v2 dataset manifest exists",
            _exists(manifest),
            _rel(manifest),
        )
    ]
    replicate_dirs = sorted(path for path in root.glob("replicate_*") if path.is_dir()) if root.exists() else []
    rows.append(
        AuditRow(
            "phase2_revised_goal3_simulator_v2",
            "simulator v2 has at least one replicate",
            "complete" if replicate_dirs else "missing",
            _rel(root),
            f"replicates={len(replicate_dirs)}",
        )
    )
    if not replicate_dirs:
        return rows

    required_files = {
        "cells.csv",
        "expression.csv",
        "lr_pairs.csv",
        "candidate_interactions.csv",
        "ground_truth_table.csv",
        "truth_lr_pairs.csv",
        "truth_cell_type_pairs.csv",
        "truth_spatial_niches.csv",
        "metadata.json",
    }
    for rep_dir in replicate_dirs:
        missing = sorted(name for name in required_files if not (rep_dir / name).exists())
        rows.append(
            AuditRow(
                "phase2_revised_goal3_simulator_v2",
                f"{rep_dir.name} truth-table files exist",
                "complete" if not missing else "missing",
                _rel(rep_dir),
                f"missing={missing}",
            )
        )
        if missing:
            continue
        rows.extend(_audit_truth_tables(rep_dir, "phase2_revised_goal3_simulator_v2"))
    return rows


def _audit_goal3_formal_simulator_v2() -> list[AuditRow]:
    import pandas as pd

    root = PROJECT_ROOT / "outputs/phase2_revised/goal3_simulator_v2/formal_datasets"
    manifest_path = root / "formal_simulator_manifest.json"
    validation_path = root / "formal_validation_summary.csv"
    benchmark_path = root / "formal_benchmark_summary.csv"
    claims_path = root / "formal_simulator_claims.csv"
    rows = [
        AuditRow("phase2_revised_goal3_formal_simulator_v2", "formal simulator manifest exists", _exists(manifest_path), _rel(manifest_path)),
        AuditRow("phase2_revised_goal3_formal_simulator_v2", "formal validation summary exists", _exists(validation_path), _rel(validation_path)),
        AuditRow("phase2_revised_goal3_formal_simulator_v2", "formal benchmark summary exists", _exists(benchmark_path), _rel(benchmark_path)),
        AuditRow("phase2_revised_goal3_formal_simulator_v2", "formal simulator claims table exists", _exists(claims_path), _rel(claims_path)),
    ]
    if not all(path.exists() for path in [manifest_path, validation_path, benchmark_path, claims_path]):
        return rows
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = pd.read_csv(validation_path)
    benchmark = pd.read_csv(benchmark_path)
    claims = pd.read_csv(claims_path)
    row = benchmark.iloc[0] if len(benchmark) else {}
    rows.extend(
        [
            AuditRow(
                "phase2_revised_goal3_formal_simulator_v2",
                "formal simulator has at least three generated replicates",
                "complete" if int(row.get("n_replicates", 0)) >= 3 and len(validation) >= 3 else "partial",
                _rel(benchmark_path),
                f"n_replicates={row.get('n_replicates', 'NA')}; validation_rows={len(validation)}",
            ),
            AuditRow(
                "phase2_revised_goal3_formal_simulator_v2",
                "formal simulator covers required mechanisms and weak positives",
                "complete"
                if bool(row.get("required_mechanisms_present", False))
                and float(row.get("median_weak_positive_interactions", 0)) > 0
                else "partial",
                _rel(benchmark_path),
                f"mechanisms={row.get('mechanisms_observed', '')}; weak={row.get('median_weak_positive_interactions', 'NA')}",
            ),
            AuditRow(
                "phase2_revised_goal3_formal_simulator_v2",
                "formal simulator high-expression controls remain negative",
                "complete" if int(row.get("false_positive_labels_total", -1)) == 0 else "partial",
                _rel(benchmark_path),
                f"false_positive_labels_total={row.get('false_positive_labels_total', 'NA')}",
            ),
            AuditRow(
                "phase2_revised_goal3_formal_simulator_v2",
                "formal simulator claims are supported and scoped",
                "complete"
                if set(claims["status"].astype(str)) == {"supported"}
                and manifest.get("evidence_status") == "formal_benchmark_structure_generated_not_final_manuscript_scale"
                else "partial",
                _rel(claims_path),
                f"statuses={sorted(set(claims['status'].astype(str)))}; evidence_status={manifest.get('evidence_status')}",
            ),
        ]
    )
    return rows


def _audit_goal4_real_background_v2() -> list[AuditRow]:
    roots = {
        "lymph_node_smoke": PROJECT_ROOT / "outputs/phase2_revised/goal4_real_background_v2/smoke_datasets",
        "tonsil_smoke": PROJECT_ROOT / "outputs/phase2_revised/goal4_real_background_v2/smoke_tonsil_datasets",
        "melanoma_smoke": PROJECT_ROOT / "outputs/phase2_revised/goal4_real_background_v2/smoke_melanoma_datasets",
    }
    rows: list[AuditRow] = []
    required_files = {
        "cells.csv",
        "expression.csv",
        "lr_pairs.csv",
        "candidate_interactions.csv",
        "ground_truth_table.csv",
        "truth_lr_pairs.csv",
        "truth_cell_type_pairs.csv",
        "truth_spatial_niches.csv",
        "metadata.json",
    }
    for label, root in roots.items():
        replicate_dirs = sorted(path for path in root.glob("*_replicate_*") if path.is_dir()) if root.exists() else []
        rows.append(
            AuditRow(
                "phase2_revised_goal4_real_background_v2",
                f"{label} has a generated replicate",
                "complete" if replicate_dirs else "missing",
                _rel(root),
                f"replicates={len(replicate_dirs)}",
            )
        )
        for rep_dir in replicate_dirs:
            missing = sorted(name for name in required_files if not (rep_dir / name).exists())
            rows.append(
                AuditRow(
                    "phase2_revised_goal4_real_background_v2",
                    f"{rep_dir.name} truth-table files exist",
                    "complete" if not missing else "missing",
                    _rel(rep_dir),
                    f"missing={missing}",
                )
            )
            if missing:
                continue
            rows.extend(
                _audit_truth_tables(
                    rep_dir,
                    "phase2_revised_goal4_real_background_v2",
                    require_all_mechanisms=False,
                )
            )
            rows.extend(_audit_real_background_properties(rep_dir))
    rows.append(_audit_goal4_combined_mechanism_coverage(roots))
    return rows


def _audit_goal4_formal_real_background_v2() -> list[AuditRow]:
    import pandas as pd

    root = PROJECT_ROOT / "outputs/phase2_revised/goal4_real_background_v2/formal_datasets"
    manifest_path = root / "formal_real_background_manifest.json"
    status_path = root / "formal_dataset_status.csv"
    validation_path = root / "formal_validation_summary.csv"
    summary_path = root / "formal_real_background_summary.csv"
    claims_path = root / "formal_real_background_claims.csv"
    rows = [
        AuditRow("phase2_revised_goal4_formal_real_background_v2", "formal real-background manifest exists", _exists(manifest_path), _rel(manifest_path)),
        AuditRow("phase2_revised_goal4_formal_real_background_v2", "formal real-background dataset status exists", _exists(status_path), _rel(status_path)),
        AuditRow("phase2_revised_goal4_formal_real_background_v2", "formal real-background validation summary exists", _exists(validation_path), _rel(validation_path)),
        AuditRow("phase2_revised_goal4_formal_real_background_v2", "formal real-background summary exists", _exists(summary_path), _rel(summary_path)),
        AuditRow("phase2_revised_goal4_formal_real_background_v2", "formal real-background claims table exists", _exists(claims_path), _rel(claims_path)),
    ]
    if not all(path.exists() for path in [manifest_path, status_path, validation_path, summary_path, claims_path]):
        return rows
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = pd.read_csv(status_path)
    validation = pd.read_csv(validation_path)
    summary = pd.read_csv(summary_path)
    claims = pd.read_csv(claims_path)
    required_datasets = {"lymph_node", "tonsil", "melanoma"}
    completed = set(status.loc[status["status"].astype(str) == "generated", "dataset_key"].astype(str))
    rows.extend(
        [
            AuditRow(
                "phase2_revised_goal4_formal_real_background_v2",
                "formal real-background benchmark covers lymph node, tonsil, and melanoma",
                "complete" if required_datasets.issubset(completed) and required_datasets.issubset(set(summary["dataset_key"].astype(str))) else "partial",
                _rel(summary_path),
                f"completed={sorted(completed)}",
            ),
            AuditRow(
                "phase2_revised_goal4_formal_real_background_v2",
                "formal real-background benchmark preserves measured ST properties",
                "complete"
                if bool(summary["preserved_property_fields_present"].astype(bool).all())
                and (validation["median_library_size"] > 0).all()
                and validation["mean_observed_sparsity"].between(0, 1).all()
                and (validation["regions"] >= 2).all()
                and (validation["cell_types"] >= 2).all()
                else "partial",
                _rel(validation_path),
                f"rows={len(validation)}",
            ),
            AuditRow(
                "phase2_revised_goal4_formal_real_background_v2",
                "formal real-background weak positives and high-expression negatives are present",
                "complete"
                if (summary["median_weak_positive_interactions"] > 0).all()
                and int(summary["false_positive_labels_total"].sum()) == 0
                and (summary["high_expression_negative_lr_pairs"] > 0).all()
                else "partial",
                _rel(summary_path),
                f"false_positive_labels_total={int(summary['false_positive_labels_total'].sum())}",
            ),
            AuditRow(
                "phase2_revised_goal4_formal_real_background_v2",
                "formal real-background claims are supported and scoped",
                "complete"
                if set(claims["status"].astype(str)) == {"supported"}
                and manifest.get("evidence_status") == "formal_real_background_structure_generated_not_final_manuscript_scale"
                else "partial",
                _rel(claims_path),
                f"statuses={sorted(set(claims['status'].astype(str)))}; evidence_status={manifest.get('evidence_status')}",
            ),
        ]
    )
    return rows


def _audit_truth_tables(
    rep_dir: Path,
    goal_id: str,
    *,
    require_all_mechanisms: bool = True,
) -> list[AuditRow]:
    import pandas as pd

    lr_pairs = pd.read_csv(rep_dir / "lr_pairs.csv")
    candidates = pd.read_csv(rep_dir / "candidate_interactions.csv")
    truth = pd.read_csv(rep_dir / "ground_truth_table.csv")
    truth_lr = pd.read_csv(rep_dir / "truth_lr_pairs.csv")
    truth_ct = pd.read_csv(rep_dir / "truth_cell_type_pairs.csv")
    truth_niche = pd.read_csv(rep_dir / "truth_spatial_niches.csv")
    mechanisms = set(lr_pairs["mechanism"].astype(str))
    required_mechanisms = {
        "juxtacrine_contact",
        "short_medium_paracrine",
        "long_range_chemokine_niche",
        "false_positive_high_expression",
    }
    weak_positive_edges = int(truth.loc[truth["weak_signal"].astype(bool), "true_edge"].sum())
    fp_positive_edges = int(
        truth.loc[truth["false_positive_high_expression"].astype(bool), "true_edge"].sum()
    )
    return [
        AuditRow(
            goal_id,
            f"{rep_dir.name} covers required mechanisms",
            "complete" if required_mechanisms.issubset(mechanisms) or not require_all_mechanisms else "partial",
            _rel(rep_dir / "lr_pairs.csv"),
            f"missing={sorted(required_mechanisms - mechanisms)}",
        ),
        AuditRow(
            goal_id,
            f"{rep_dir.name} candidate truth table matches candidate rows",
            "complete" if len(truth) == len(candidates) and int(truth["true_edge"].sum()) == int(candidates["ground_truth"].sum()) else "partial",
            _rel(rep_dir / "ground_truth_table.csv"),
            f"truth_rows={len(truth)} candidates={len(candidates)}",
        ),
        AuditRow(
            goal_id,
            f"{rep_dir.name} includes weak positive edges",
            "complete" if weak_positive_edges > 0 else "partial",
            _rel(rep_dir / "ground_truth_table.csv"),
            f"weak_positive_edges={weak_positive_edges}",
        ),
        AuditRow(
            goal_id,
            f"{rep_dir.name} high-expression controls remain negative",
            "complete" if fp_positive_edges == 0 else "partial",
            _rel(rep_dir / "ground_truth_table.csv"),
            f"false_positive_control_true_edges={fp_positive_edges}",
        ),
        AuditRow(
            goal_id,
            f"{rep_dir.name} LR/cell-type/niche truth summaries have positives",
            "complete"
            if int(truth_lr["true_lr_pair"].sum()) > 0
            and int(truth_ct["true_cell_type_pair"].sum()) > 0
            and int(truth_niche["true_spatial_niche"].sum()) > 0
            else "partial",
            _rel(rep_dir),
            (
                f"lr={int(truth_lr['true_lr_pair'].sum())}; "
                f"cell_type={int(truth_ct['true_cell_type_pair'].sum())}; "
                f"niche={int(truth_niche['true_spatial_niche'].sum())}"
            ),
        ),
    ]


def _audit_goal4_combined_mechanism_coverage(roots: dict[str, Path]) -> AuditRow:
    import pandas as pd

    mechanisms: set[str] = set()
    for root in roots.values():
        if not root.exists():
            continue
        for rep_dir in sorted(path for path in root.glob("*_replicate_*") if path.is_dir()):
            lr_path = rep_dir / "lr_pairs.csv"
            if lr_path.exists():
                mechanisms.update(pd.read_csv(lr_path)["mechanism"].astype(str).tolist())
    required = {
        "juxtacrine_contact",
        "short_medium_paracrine",
        "long_range_chemokine_niche",
        "false_positive_high_expression",
    }
    return AuditRow(
        "phase2_revised_goal4_real_background_v2",
        "combined lymph/tonsil/melanoma real-background smoke covers required mechanisms",
        "complete" if required.issubset(mechanisms) else "partial",
        "outputs/phase2_revised/goal4_real_background_v2/smoke_*_datasets",
        f"missing={sorted(required - mechanisms)}",
    )


def _audit_real_background_properties(rep_dir: Path) -> list[AuditRow]:
    import json
    import pandas as pd

    cells = pd.read_csv(rep_dir / "cells.csv")
    metadata = json.loads((rep_dir / "metadata.json").read_text(encoding="utf-8"))
    preserved = set(metadata.get("preserved_real_properties", []))
    required = {
        "spatial_coordinates",
        "library_size",
        "observed_expression_sparsity",
        "cell_type_composition",
        "local_density",
        "tissue_domain_structure",
    }
    library_size = pd.to_numeric(cells["library_size"], errors="coerce")
    sparsity = pd.to_numeric(cells["observed_sparsity"], errors="coerce")
    return [
        AuditRow(
            "phase2_revised_goal4_real_background_v2",
            f"{rep_dir.name} records required preserved real properties",
            "complete" if required.issubset(preserved) else "partial",
            _rel(rep_dir / "metadata.json"),
            f"missing={sorted(required - preserved)}",
        ),
        AuditRow(
            "phase2_revised_goal4_real_background_v2",
            f"{rep_dir.name} real coordinates and tissue labels are non-degenerate",
            "complete"
            if cells["x"].nunique() > 1
            and cells["y"].nunique() > 1
            and cells["cell_type"].nunique() >= 2
            and cells["region"].nunique() >= 2
            else "partial",
            _rel(rep_dir / "cells.csv"),
            f"cell_types={cells['cell_type'].nunique()}; regions={cells['region'].nunique()}",
        ),
        AuditRow(
            "phase2_revised_goal4_real_background_v2",
            f"{rep_dir.name} library size and sparsity are preserved as measured fields",
            "complete"
            if float(library_size.median()) > 0
            and float(sparsity.mean()) > 0
            and float(sparsity.mean()) < 1
            else "partial",
            _rel(rep_dir / "cells.csv"),
            f"median_library={float(library_size.median())}; mean_sparsity={float(sparsity.mean())}",
        ),
    ]


def _audit_coexpression_baseline_removed() -> list[AuditRow]:
    search_roots = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "docs",
        PROJECT_ROOT / "configs",
        PROJECT_ROOT / "scripts",
        PROJECT_ROOT / "src",
    ]
    needles = ("LR-" + "Coexp", "lr_" + "coexp", "lr" + "coexp", "LR_" + "Coexp")
    hits: list[str] = []
    for root in search_roots:
        if not root.exists():
            continue
        paths = [root] if root.is_file() else list(root.rglob("*"))
        for path in paths:
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for needle in needles:
                if needle in text:
                    hits.append(f"{_rel(path)}:{needle}")
    return [
        AuditRow(
            "phase2_revised",
            "removed coexpression baseline name is absent from docs/configs/scripts/src",
            "complete" if not hits else "partial",
            "README.md docs configs scripts src",
            f"hits={hits[:20]}",
        )
    ]


def _exists(path: Path) -> str:
    return "complete" if path.exists() else "missing"


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _claim_text(claim_matrix, claim_id: str) -> str:
    rows = claim_matrix[claim_matrix["claim_id"].astype(str) == claim_id]
    if rows.empty:
        return ""
    row = rows.iloc[0]
    fields = ["supported_statement", "claim_scope", "allowed_wording", "disallowed_wording"]
    return " ".join(str(row.get(field, "")) for field in fields)


def _claim_text_contains(claim_matrix, claim_id: str, needles: list[str]) -> bool:
    text = _claim_text(claim_matrix, claim_id)
    return all(needle in text for needle in needles)


def _guardrails_contain(guardrails, needles: list[str]) -> bool:
    text = " ".join(
        guardrails.astype(str).fillna("").agg(" ".join, axis=1).tolist()
    )
    return all(needle in text for needle in needles)


def _table_text_contains(table, needles: list[str]) -> bool:
    text = " ".join(table.astype(str).fillna("").agg(" ".join, axis=1).tolist())
    return all(needle in text for needle in needles)


if __name__ == "__main__":
    main()
