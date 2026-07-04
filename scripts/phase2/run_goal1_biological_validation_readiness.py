#!/usr/bin/env python
"""Synthesize Goal 1 biological-validation readiness and claim gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SPATIALSELF_METHODS = ("spatialself_wavelet_edge_prior", "spatialself_wavelet")
FP_COMPARATOR_METHODS = ("commot", "cellchat", "cellnest")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/phase2_revised/goal1_biological_validation_readiness")
    parser.add_argument(
        "--formal-method-summary",
        default="outputs/phase2_revised/goal1_formal_baseline_matched_validation/goal1_baseline_matched_method_summary.csv",
    )
    parser.add_argument(
        "--formal-lr-summary",
        default="outputs/phase2_revised/goal1_formal_baseline_matched_validation/goal1_baseline_matched_lr_control_summary.csv",
    )
    parser.add_argument(
        "--curated-method-summary",
        default="outputs/phase2_revised/goal1_curated_downstream_panel/goal1_curated_downstream_method_summary.csv",
    )
    parser.add_argument(
        "--curated-coverage",
        default="outputs/phase2_revised/goal1_curated_downstream_panel/goal1_curated_downstream_target_coverage.csv",
    )
    parser.add_argument(
        "--curated-claims",
        default="outputs/phase2_revised/goal1_curated_downstream_panel/goal1_curated_downstream_claims.csv",
    )
    parser.add_argument(
        "--fdr-claims",
        default="outputs/phase2_revised/goal5_real_background_fdr_100perm/goal5_real_background_fdr_panel_claims.csv",
    )
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    formal_method = pd.read_csv(PROJECT_ROOT / args.formal_method_summary)
    formal_lr = pd.read_csv(PROJECT_ROOT / args.formal_lr_summary)
    curated_method = pd.read_csv(PROJECT_ROOT / args.curated_method_summary)
    curated_coverage = pd.read_csv(PROJECT_ROOT / args.curated_coverage)
    curated_claims = pd.read_csv(PROJECT_ROOT / args.curated_claims)
    fdr_claims = pd.read_csv(PROJECT_ROOT / args.fdr_claims)

    dataset_gate = _dataset_gate(formal_method, curated_coverage, curated_claims, fdr_claims)
    method_gate = _method_gate(formal_method, curated_method)
    lr_gate = _lr_gate(formal_lr)
    claims = _claims(dataset_gate, method_gate)

    dataset_path = output_dir / "goal1_biological_validation_dataset_gate.csv"
    method_path = output_dir / "goal1_biological_validation_method_gate.csv"
    lr_path = output_dir / "goal1_biological_validation_lr_gate.csv"
    claims_path = output_dir / "goal1_biological_validation_claims.csv"
    manifest_path = output_dir / "goal1_biological_validation_readiness_manifest.json"
    dataset_gate.to_csv(dataset_path, index=False)
    method_gate.to_csv(method_path, index=False)
    lr_gate.to_csv(lr_path, index=False)
    claims.to_csv(claims_path, index=False)
    manifest = {
        "analysis_label": "phase2_goal1_biological_validation_readiness",
        "canonical_goal": "goal1_baseline_matched_biological_validation",
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "claim_level": "goal1_readiness_gate_not_final_downstream_superiority",
        "inputs": {
            "formal_method_summary": args.formal_method_summary,
            "formal_lr_summary": args.formal_lr_summary,
            "curated_method_summary": args.curated_method_summary,
            "curated_coverage": args.curated_coverage,
            "curated_claims": args.curated_claims,
            "fdr_claims": args.fdr_claims,
        },
        "outputs": {
            "dataset_gate": str(dataset_path.relative_to(PROJECT_ROOT)),
            "method_gate": str(method_path.relative_to(PROJECT_ROOT)),
            "lr_gate": str(lr_path.relative_to(PROJECT_ROOT)),
            "claims": str(claims_path.relative_to(PROJECT_ROOT)),
        },
        "interpretation_guardrail": (
            "Use this gate to separate supported control-ranking/false-positive "
            "suppression claims from unsupported curated downstream superiority claims."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(dataset_gate.to_string(index=False))
    print(claims.to_string(index=False))
    print(f"Wrote Goal1 biological-validation readiness outputs to {output_dir}")


def _dataset_gate(
    formal_method: pd.DataFrame,
    curated_coverage: pd.DataFrame,
    curated_claims: pd.DataFrame,
    fdr_claims: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for dataset, group in formal_method.groupby("dataset", sort=False):
        spatial = group[group["method"].astype(str).isin(SPATIALSELF_METHODS)]
        non_spatial = group[~group["method"].astype(str).isin(SPATIALSELF_METHODS)]
        fp_comparators = group[group["method"].astype(str).isin(FP_COMPARATOR_METHODS)]
        best_spatial_fp = float(spatial["high_expression_false_positive_rate_at_q95"].min()) if len(spatial) else float("nan")
        best_non_spatial_fp = float(non_spatial["high_expression_false_positive_rate_at_q95"].min()) if len(non_spatial) else float("nan")
        best_fp_comparator = (
            float(fp_comparators["high_expression_false_positive_rate_at_q95"].min())
            if len(fp_comparators)
            else float("nan")
        )
        best_spatial_rank = float(spatial["mean_positive_rank"].min()) if len(spatial) else float("nan")
        best_non_spatial_rank = float(non_spatial["mean_positive_rank"].min()) if len(non_spatial) else float("nan")
        spatial_rank_pass = bool(spatial["rank_validation_pass"].astype(bool).any()) if len(spatial) else False
        non_spatial_rank_pass = bool(non_spatial["rank_validation_pass"].astype(bool).any()) if len(non_spatial) else False
        cov = curated_coverage[curated_coverage["dataset"].astype(str) == str(dataset)]
        curated = curated_claims[curated_claims["dataset"].astype(str) == str(dataset)]
        fdr_dataset = _fdr_dataset_name(dataset)
        fdr = fdr_claims[fdr_claims["dataset"].astype(str) == fdr_dataset]
        positive_coverage = float(cov["positive_target_coverage_fraction"].iloc[0]) if len(cov) else float("nan")
        spatial_downstream_winner = bool(curated["spatialself_best_delta"].iloc[0]) if len(curated) else False
        rows.append(
            {
                "dataset": dataset,
                "best_spatialself_fp_rate": best_spatial_fp,
                "best_non_spatial_fp_rate": best_non_spatial_fp,
                "best_commot_cellchat_cellnest_fp_rate": best_fp_comparator,
                "spatialself_absolute_low_fp_pass": bool(best_spatial_fp <= 0.01),
                "spatialself_fp_better_than_commot_cellchat_cellnest": bool(best_spatial_fp <= best_fp_comparator),
                "spatialself_lowest_among_all_baselines": bool(best_spatial_fp <= best_non_spatial_fp),
                "best_spatialself_positive_rank": best_spatial_rank,
                "best_non_spatial_positive_rank": best_non_spatial_rank,
                "spatialself_rank_validation_pass": spatial_rank_pass,
                "non_spatial_rank_validation_pass": non_spatial_rank_pass,
                "positive_target_coverage_fraction": positive_coverage,
                "curated_downstream_coverage_pass": bool(positive_coverage >= 0.8),
                "spatialself_curated_downstream_winner": spatial_downstream_winner,
                "curated_downstream_superiority_supported": bool(positive_coverage >= 0.8 and spatial_downstream_winner),
                "fdr_schema_complete": bool(fdr["fdr_schema_complete"].iloc[0]) if len(fdr) else False,
                "fdr_n_permutations": int(fdr["n_permutations"].iloc[0]) if len(fdr) else 0,
                "fdr_lr_pair_discoveries": int(fdr["lr_pair_direct_discoveries"].iloc[0]) if len(fdr) else 0,
                "fdr_candidate_edge_discoveries": int(fdr["candidate_edge_discoveries"].iloc[0]) if len(fdr) else 0,
                "biological_validation_status": _dataset_status(
                    fp_pass=bool(best_spatial_fp <= 0.01 and best_spatial_fp <= best_fp_comparator),
                    rank_pass=spatial_rank_pass,
                    fdr_pass=bool(len(fdr) and fdr["fdr_schema_complete"].iloc[0]),
                    downstream_pass=bool(positive_coverage >= 0.8 and spatial_downstream_winner),
                ),
            }
        )
    return pd.DataFrame(rows)


def _method_gate(formal_method: pd.DataFrame, curated_method: pd.DataFrame) -> pd.DataFrame:
    merged = formal_method.merge(
        curated_method[
            [
                "dataset",
                "method",
                "n_tested_lr_pairs",
                "n_positive_tested",
                "n_negative_tested",
                "mean_positive_spearman",
                "mean_negative_spearman",
                "positive_negative_spearman_delta",
                "significant_positive_count",
                "significant_negative_count",
            ]
        ],
        on=["dataset", "method"],
        how="left",
    )
    rows = []
    for row in merged.itertuples(index=False):
        rows.append(
            {
                "dataset": row.dataset,
                "method": row.method,
                "rank_validation_pass": bool(row.rank_validation_pass),
                "negative_top3_count": int(row.negative_top3_count),
                "empirical_fdr_at_q95": float(row.empirical_fdr_at_q95),
                "high_expression_false_positive_rate_at_q95": float(row.high_expression_false_positive_rate_at_q95),
                "n_curated_tested_lr_pairs": int(row.n_tested_lr_pairs) if pd.notna(row.n_tested_lr_pairs) else 0,
                "curated_positive_negative_spearman_delta": float(row.positive_negative_spearman_delta)
                if pd.notna(row.positive_negative_spearman_delta)
                else float("nan"),
                "curated_significant_positive_count": int(row.significant_positive_count)
                if pd.notna(row.significant_positive_count)
                else 0,
                "curated_significant_negative_count": int(row.significant_negative_count)
                if pd.notna(row.significant_negative_count)
                else 0,
                "supported_for_control_ranking": bool(row.rank_validation_pass and int(row.negative_top3_count) == 0),
                "supported_for_fp_suppression": bool(float(row.high_expression_false_positive_rate_at_q95) <= 0.01),
                "supported_for_curated_downstream": False,
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        for dataset, idx in table.groupby("dataset").groups.items():
            dataset_rows = table.loc[list(idx)]
            best_delta = dataset_rows["curated_positive_negative_spearman_delta"].max(skipna=True)
            table.loc[list(idx), "supported_for_curated_downstream"] = (
                table.loc[list(idx), "method"].isin(SPATIALSELF_METHODS)
                & table.loc[list(idx), "curated_positive_negative_spearman_delta"].eq(best_delta)
            )
    return table


def _lr_gate(formal_lr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, lr_id), group in formal_lr.groupby(["dataset", "lr_id"], sort=False):
        spatial = group[group["method"].astype(str).isin(SPATIALSELF_METHODS)]
        non_spatial = group[~group["method"].astype(str).isin(SPATIALSELF_METHODS)]
        first = group.iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "lr_id": lr_id,
                "mechanism": first["mechanism"],
                "positive_control": bool(first["positive_control"]),
                "negative_control": bool(first["negative_control"]),
                "weak_signal": bool(first["weak_signal"]),
                "best_spatialself_rank": float(spatial["global_rank"].min()) if len(spatial) else float("nan"),
                "best_non_spatial_rank": float(non_spatial["global_rank"].min()) if len(non_spatial) else float("nan"),
                "spatialself_keeps_negative_below_top3": bool(
                    (not bool(first["negative_control"])) or (len(spatial) and int(spatial["global_rank"].min()) > 3)
                ),
                "spatialself_positive_top3": bool(
                    bool(first["positive_control"]) and len(spatial) and int(spatial["global_rank"].min()) <= 3
                ),
            }
        )
    return pd.DataFrame(rows)


def _claims(dataset_gate: pd.DataFrame, method_gate: pd.DataFrame) -> pd.DataFrame:
    spatial_methods = method_gate[method_gate["method"].astype(str).isin(SPATIALSELF_METHODS)]
    rows = [
        {
            "claim_id": "G1_C1_control_ranking",
            "status": "supported"
            if dataset_gate["spatialself_rank_validation_pass"].astype(bool).all()
            else "partial",
            "statement": "SpatialSelf ranks injected positive controls ahead of high-expression negative controls across the three formal real-background datasets.",
            "allowed_wording": "baseline-matched injected-control ranking supports SpatialSelf control-prioritization",
            "disallowed_wording": "unlabeled real tissue ground truth is proven",
        },
        {
            "claim_id": "G1_C2_high_expression_fp_suppression",
            "status": "supported"
            if dataset_gate["spatialself_absolute_low_fp_pass"].astype(bool).all()
            and dataset_gate["spatialself_fp_better_than_commot_cellchat_cellnest"].astype(bool).all()
            and bool(spatial_methods["supported_for_fp_suppression"].astype(bool).any())
            else "partial",
            "statement": "SpatialSelf keeps high-expression false-positive call rates low and below COMMOT, CellChat, and CellNEST in the matched formal real-background validation; SpaTalk can be equally or more conservative in these panels.",
            "allowed_wording": "high-expression false-positive suppression is supported relative to COMMOT, CellChat, and CellNEST, with SpaTalk noted as conservative",
            "disallowed_wording": "SpatialSelf has the lowest false-positive rate against every baseline or eliminates all false positives in real unlabeled tissue",
        },
        {
            "claim_id": "G1_C3_curated_downstream",
            "status": "unsupported"
            if not dataset_gate["curated_downstream_superiority_supported"].astype(bool).any()
            else "partial",
            "statement": "Curated pathway/Reactome/TRRUST downstream evidence is coverage-limited and does not support SpatialSelf downstream-superiority yet.",
            "allowed_wording": "curated downstream validation remains coverage-limited and mixed",
            "disallowed_wording": "SpatialSelf has demonstrated global downstream target superiority",
        },
        {
            "claim_id": "G1_C4_fdr_integration",
            "status": "supported"
            if dataset_gate["fdr_schema_complete"].astype(bool).all()
            and (dataset_gate["fdr_n_permutations"].astype(int) >= 100).all()
            else "partial",
            "statement": "Goal 1 biological validation can now cite the representative 100-permutation FDR panel as group-level calibration support.",
            "allowed_wording": "representative 100-permutation group-level FDR support is available",
            "disallowed_wording": "candidate-edge FDR discovery power is established",
        },
    ]
    return pd.DataFrame(rows)


def _dataset_status(*, fp_pass: bool, rank_pass: bool, fdr_pass: bool, downstream_pass: bool) -> str:
    if fp_pass and rank_pass and fdr_pass and downstream_pass:
        return "main_text_ready_full_biological_validation"
    if fp_pass and rank_pass and fdr_pass:
        return "main_text_ready_control_ranking_fp_fdr_downstream_limited"
    if fp_pass and rank_pass:
        return "partial_control_ranking_fp_only"
    return "insufficient"


def _fdr_dataset_name(dataset: str) -> str:
    if dataset.endswith("lymph_node"):
        return "real_background_v2_formal_lymph_node"
    if dataset.endswith("tonsil"):
        return "real_background_v2_formal_tonsil"
    if dataset.endswith("melanoma"):
        return "real_background_v2_formal_melanoma"
    return dataset


if __name__ == "__main__":
    main()
