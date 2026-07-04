#!/usr/bin/env python
"""Create a conservative closure contract for Phase 12 Goal 1."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
OUT_DIR = PHASE12_DIR / "goal1_closure_contract"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_contract_rows()
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "phase12_goal1_closure_contract.csv", index=False)

    status = pd.DataFrame(build_status_rows(table))
    status.to_csv(OUT_DIR / "phase12_goal1_closure_contract_status.csv", index=False)
    write_markdown(table, status)
    manifest = {
        "task": "phase12_goal1_closure_contract",
        "outputs": [
            rel(OUT_DIR / "phase12_goal1_closure_contract.csv"),
            rel(OUT_DIR / "phase12_goal1_closure_contract_status.csv"),
            rel(OUT_DIR / "phase12_goal1_closure_contract.md"),
        ],
        "claim_boundary": (
            "This is a requirement contract for deciding whether Goal 1 can be closed. "
            "It does not create new discovery evidence."
        ),
    }
    (OUT_DIR / "phase12_goal1_closure_contract_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(status.to_string(index=False))
    print()
    print(table.to_string(index=False))


def build_contract_rows() -> list[dict[str, object]]:
    completion = read_csv(PHASE12_DIR / "completion_audit/phase12_requirement_completion_audit.csv")
    fdr_story = read_csv(PHASE12_DIR / "goal1_fdr_bottleneck/phase12_goal1_fdr_bottleneck_story.csv")
    fdr_gap = read_csv(PHASE12_DIR / "fdr_margin_gap/phase12_fdr_margin_gap_story.csv")
    integrated = read_csv(
        PHASE12_DIR
        / "integrated_guarded_spatial_training_permutation_audit/"
        / "phase12_integrated_guarded_spatial_training_permutation_story.csv"
    )
    highres = read_csv(
        PHASE12_DIR
        / "integrated_guarded_spatial_training_highres_stability/"
        / "phase12_integrated_guarded_spatial_highres_stability_story.csv"
    )
    leakage = read_csv(
        PHASE12_DIR
        / "integrated_guarded_spatial_coordinate_leakage/"
        / "phase12_integrated_coordinate_leakage_summary.csv"
    )
    magnitude = read_csv(
        PHASE12_DIR
        / "integrated_guarded_spatial_coordinate_magnitude_gate/"
        / "phase12_integrated_coordinate_magnitude_gate_best.csv"
    )
    background_targets = read_csv(
        PHASE12_DIR
        / "integrated_guarded_spatial_coordinate_background_targets/"
        / "phase12_integrated_coordinate_background_target_summary.csv"
    )
    receptor_shared = read_csv(
        PHASE12_DIR / "receptor_shared_atlas_audit/phase12_receptor_shared_atlas_dataset_summary.csv"
    )
    method_comparison = read_csv(
        PHASE12_DIR / "discovery_eligible_method_comparison/phase12_discovery_eligible_method_story.csv"
    )

    rows: list[dict[str, object]] = []
    rows.append(
        contract_row(
            "large_lr_atlas_runs",
            "pass",
            completion_proof(completion, "At least two large-LR real-background atlas runs"),
            "Keep strict full-transcriptome atlas claims separate from supplementary expanded-panel claims.",
            "Goal 1 can keep the current three strict expanded atlases as the base evidence.",
            "completion_audit/phase12_requirement_completion_audit.csv",
        )
    )
    rows.append(
        contract_row(
            "whole_lr_ranking",
            "pass",
            completion_proof(completion, "Whole-LR ranking table"),
            "Ranking is not discovery FDR.",
            "Keep calibrated and raw rankings linked as source data.",
            "completion_audit/phase12_requirement_completion_audit.csv",
        )
    )
    rows.append(
        contract_row(
            "strict_broad_all_lr_fdr",
            "fail_current_reporting_baseline",
            fdr_summary(fdr_story, fdr_gap),
            "Do not claim global whole-atlas discovery FDR for the current reporting baseline.",
            "A replacement model must pass strict broad all-LR q<=0.1 discovery without increasing hard-negative contamination.",
            "goal1_fdr_bottleneck/phase12_goal1_fdr_bottleneck_story.csv; fdr_margin_gap/phase12_fdr_margin_gap_story.csv",
        )
    )
    rows.append(
        contract_row(
            "integrated_candidate_exact_score",
            "candidate_support",
            integrated_summary(integrated, highres),
            "Exact-score support is not end-to-end coordinate/null retraining FDR and does not yet replace the baseline.",
            "Use integrated guarded spatial training as the leading candidate, then rerun end-to-end null audits.",
            "integrated_guarded_spatial_training_permutation_audit/phase12_integrated_guarded_spatial_training_permutation_story.csv; integrated_guarded_spatial_training_highres_stability/phase12_integrated_guarded_spatial_highres_stability_story.csv",
        )
    )
    rows.append(
        contract_row(
            "coordinate_rerun_leakage",
            "fail_broad_background",
            coordinate_leakage_summary(leakage),
            "Coordinate smoke suppresses high-expression negatives but broad/background LR still dominate minimum p-values.",
            "Scale coordinate reruns only after adding broad/background suppression or a prespecified magnitude/background guard.",
            "integrated_guarded_spatial_coordinate_leakage/phase12_integrated_coordinate_leakage_summary.csv",
        )
    )
    rows.append(
        contract_row(
            "magnitude_background_guard",
            "candidate_guard_not_final_rule",
            magnitude_background_summary(magnitude, background_targets),
            "The gate was selected diagnostically and cannot be treated as the final unbiased FDR rule.",
            "Predefine this as a model/calibration component and rerun exact-score plus coordinate-null FDR.",
            "integrated_guarded_spatial_coordinate_magnitude_gate/phase12_integrated_coordinate_magnitude_gate_best.csv; integrated_guarded_spatial_coordinate_background_targets/phase12_integrated_coordinate_background_target_summary.csv",
        )
    )
    rows.append(
        contract_row(
            "receptor_shared_and_high_expression_controls",
            "mostly_pass_with_scope",
            receptor_summary(receptor_shared),
            "This is control-contamination evidence, not native whole-atlas FDR.",
            "Keep receptor-shared and high-expression controls as mandatory gates for every baseline replacement.",
            "receptor_shared_atlas_audit/phase12_receptor_shared_atlas_dataset_summary.csv",
        )
    )
    rows.append(
        contract_row(
            "same_universe_method_comparison",
            "partial_adapter_light",
            method_summary(method_comparison),
            "Adapter-light/projected same-universe comparison does not replace fully tuned official end-to-end baselines.",
            "Before final manuscript claims, either run official tuned baselines or clearly label this comparison as same-universe adapter-light.",
            "discovery_eligible_method_comparison/phase12_discovery_eligible_method_story.csv",
        )
    )
    rows.append(
        contract_row(
            "baseline_replacement_decision",
            "not_ready",
            "Current reporting baseline has calibrated ranking/control support; integrated candidate has stronger selected exact-score evidence but not end-to-end coordinate FDR closure.",
            "Do not replace the baseline solely based on diagnostic magnitude gates or exact-score candidate hits.",
            "Baseline replacement requires strict broad FDR, clean high-expression/receptor-shared/broad-background controls, and end-to-end coordinate/null rerun support.",
            "phase12_current_status_table.csv; final_limitations_next_model/phase12_final_limitations_next_model.csv",
        )
    )
    return rows


def build_status_rows(table: pd.DataFrame) -> list[dict[str, object]]:
    n_pass = int(table["contract_status"].astype(str).str.startswith("pass").sum())
    n_candidate = int(table["contract_status"].astype(str).str.contains("candidate|partial|mostly", regex=True).sum())
    n_fail = int(table["contract_status"].astype(str).str.contains("fail|not_ready", regex=True).sum())
    return [
        {
            "topic": "goal1_closure_contract",
            "overall_status": "not_ready_for_goal1_closure",
            "n_contract_items": int(len(table)),
            "n_pass": n_pass,
            "n_candidate_or_partial": n_candidate,
            "n_fail_or_not_ready": n_fail,
            "main_blocker": "strict broad whole-atlas FDR and end-to-end coordinate broad/background leakage",
            "next_required_action": (
                "promote magnitude/background guardrails into a prespecified integrated model, "
                "then rerun exact-score and coordinate/null FDR before baseline replacement"
            ),
        }
    ]


def contract_row(
    requirement: str,
    status: str,
    evidence: str,
    claim_boundary: str,
    next_action: str,
    evidence_file: str,
) -> dict[str, object]:
    return {
        "requirement": requirement,
        "contract_status": status,
        "current_evidence": evidence,
        "claim_boundary": claim_boundary,
        "next_required_action": next_action,
        "evidence_file": rel_path_string(evidence_file),
    }


def completion_proof(table: pd.DataFrame, requirement: str) -> str:
    if table.empty:
        return "completion audit missing"
    matched = table[table["requirement"].astype(str).eq(requirement)]
    if matched.empty:
        return f"completion audit row missing for {requirement}"
    row = matched.iloc[0]
    return f"{row['completion_status']}: {row['proof']}"


def fdr_summary(fdr_story: pd.DataFrame, fdr_gap: pd.DataFrame) -> str:
    parts = []
    if not fdr_story.empty:
        row = fdr_story.iloc[0]
        parts.append(
            f"overall={row['overall_state']}; broad-supported datasets={int(row['n_broad_supported'])}/"
            f"{int(row['n_datasets'])}; spatial-distance bottleneck={int(row['n_spatial_distance_bottleneck'])}"
        )
    if not fdr_gap.empty:
        row = fdr_gap.iloc[0]
        parts.append(
            f"best variant={row['best_variant_by_min_q_gap']}; min q-gap={float(row['best_variant_min_q_gap_to_0_1']):.3g}x; "
            f"dominant null={row['best_variant_dominant_null_counts']}"
        )
    return "; ".join(parts) if parts else "FDR bottleneck files missing."


def integrated_summary(integrated: pd.DataFrame, highres: pd.DataFrame) -> str:
    parts = []
    if not integrated.empty:
        row = integrated.iloc[0]
        parts.append(
            f"overall={row['overall_call']}; total true q<=0.1={int(row['total_true_q_0_1'])}; "
            f"best true q={float(row['best_true_q_min']):.3g}"
        )
    if not highres.empty:
        row = highres.iloc[0]
        parts.append(
            f"highres={row['overall_call']}; best={row['best_highres_true']} q={float(row['best_highres_true_q']):.3g}"
        )
    return "; ".join(parts) if parts else "Integrated candidate files missing."


def coordinate_leakage_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Coordinate leakage file missing."
    row = table.iloc[0]
    return (
        f"call={row['leakage_call']}; min-p LR={int(row['n_min_p_lr'])}; "
        f"true={int(row['n_min_p_true_lr'])}; high-expression negative={int(row['n_min_p_high_expression_negative'])}; "
        f"broad/background={int(row['n_min_p_broad_background'])}; "
        f"best true={row['best_true_lr']} BH={float(row['best_true_bh_fdr']):.3g}"
    )


def magnitude_background_summary(magnitude: pd.DataFrame, background: pd.DataFrame) -> str:
    parts = []
    if not magnitude.empty:
        row = magnitude.iloc[0]
        parts.append(
            f"magnitude gate={float(row['score_threshold']):.3g}; retained LR={int(row['n_lr'])}; "
            f"true={int(row['n_true_lr'])}; high-expression negative={int(row['n_high_expression_negative'])}; "
            f"broad/background={int(row['n_broad_background'])}"
        )
    if not background.empty:
        row = background.iloc[0]
        parts.append(
            f"background min-p={int(row['n_min_p_broad_background'])}; "
            f"top10={int(row['n_broad_background_top10'])}; "
            f"score gap={float(row['retained_true_to_max_broad_score_gap']):.3g}x; "
            f"background above gate={int(row['n_broad_background_above_gate'])}"
        )
    return "; ".join(parts) if parts else "Magnitude/background files missing."


def receptor_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Receptor-shared audit file missing."
    n_rows = len(table)
    if "top10_receptor_shared_decoy" in table.columns:
        top10 = pd.to_numeric(table["top10_receptor_shared_decoy"], errors="coerce").fillna(0)
        clean_top10 = int((top10 == 0).sum())
    else:
        clean_top10 = 0
    return f"receptor-shared dataset rows={n_rows}; clean top10 rows={clean_top10}"


def method_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "Discovery-eligible method comparison story missing."
    row = table.iloc[0]
    return "; ".join(f"{col}={row[col]}" for col in table.columns[: min(len(table.columns), 5)])


def write_markdown(table: pd.DataFrame, status: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Goal 1 Closure Contract",
        "",
        "This file defines the current evidence boundary for closing Goal 1.",
        "It is intentionally conservative: diagnostic gates do not count as final FDR closure.",
        "",
    ]
    if len(status):
        row = status.iloc[0]
        lines.extend(
            [
                f"- Overall status: `{row['overall_status']}`",
                f"- Contract items: {int(row['n_contract_items'])}",
                f"- Pass: {int(row['n_pass'])}",
                f"- Candidate/partial: {int(row['n_candidate_or_partial'])}",
                f"- Fail/not ready: {int(row['n_fail_or_not_ready'])}",
                f"- Main blocker: {row['main_blocker']}",
                "",
            ]
        )
    for row in table.itertuples(index=False):
        lines.extend(
            [
                f"## {row.requirement}",
                "",
                f"- Status: `{row.contract_status}`",
                f"- Evidence: {row.current_evidence}",
                f"- Claim boundary: {row.claim_boundary}",
                f"- Next action: {row.next_required_action}",
                f"- Evidence file: `{row.evidence_file}`",
                "",
            ]
        )
    (OUT_DIR / "phase12_goal1_closure_contract.md").write_text("\n".join(lines), encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def rel_path_string(paths: str) -> str:
    parts = []
    for item in paths.split(";"):
        item = item.strip()
        if not item:
            continue
        if item.startswith("outputs/"):
            parts.append(item)
        else:
            parts.append(f"outputs/phase12/{item}")
    return "; ".join(parts)


if __name__ == "__main__":
    main()
