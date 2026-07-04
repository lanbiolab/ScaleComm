#!/usr/bin/env python
"""Run Phase 7 Goal 1 downstream target validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_downstream_target_correlation import run_downstream_target_correlation
from src.spitialself.fdr import bh_adjust


LYMPH_NODE_TARGETS = [
    {
        "ligand": "CCL19",
        "receptor": "CCR7",
        "target_genes": "CCR7,CXCR4,IL7R,LTB,JUNB,NFKBIA,STAT1,TCF7,SELL,CD3D,CD3E",
        "target_source": "phase7_curated_lymphoid_chemokine_response",
        "evidence_note": "CCR7-axis T-zone receiver response signature.",
    },
    {
        "ligand": "CCL21",
        "receptor": "CCR7",
        "target_genes": "CCR7,CXCR4,IL7R,LTB,JUNB,NFKBIA,STAT1,TCF7,SELL,CD3D,CD3E",
        "target_source": "phase7_curated_lymphoid_chemokine_response",
        "evidence_note": "CCR7-axis T-zone receiver response signature.",
    },
    {
        "ligand": "CXCL13",
        "receptor": "CXCR5",
        "target_genes": "CXCR5,MS4A1,PRDM1,JUNB,NFKBIA,LTB,CD79A,CD79B,FCER2",
        "target_source": "phase7_curated_gc_bcell_response",
        "evidence_note": "CXCR5-axis follicle/germinal-center receiver response signature.",
    },
    {
        "ligand": "CD48",
        "receptor": "CD244",
        "target_genes": "CD244,CD48,CD247,LCK,FYN,ZAP70,RUNX3,CD3D,CD3E,NCR3,KLRD1,BTLA,CD96,STAT1,JUNB,NFKBIA",
        "target_source": "phase7_curated_tcell_contact_activation_response",
        "evidence_note": "CD244-associated T/NK contact and activation receiver response signature.",
    },
    {
        "ligand": "SPP1",
        "receptor": "CD44",
        "target_genes": "CD44,SPP1,JUNB,NFKBIA,STAT1,CXCR4,LTB,VIM,ITGB1",
        "target_source": "phase7_curated_inflammatory_paracrine_response",
        "evidence_note": "SPP1-CD44 inflammatory paracrine receiver response signature.",
    },
    {
        "ligand": "ACTB",
        "receptor": "EGFR",
        "target_genes": "EGFR,JUNB,NFKBIA,STAT1",
        "target_source": "phase7_negative_high_expression_control",
        "evidence_note": "High-expression negative-control response panel.",
    },
    {
        "ligand": "GAPDH",
        "receptor": "ITGB1",
        "target_genes": "ITGB1,JUNB,NFKBIA,VIM,ACTG1",
        "target_source": "phase7_negative_high_expression_control",
        "evidence_note": "High-expression negative-control response panel.",
    },
]


PREDICTION_PATHS = {
    "spatialself": "outputs/phase6/expanded_lr_universe/baselines/lymph_node_phase6_expanded/spatialself_wavelet_edge_prior/human_lymph_node_visium_phase6_expanded_replicate_00__spatialself_wavelet_predictions.csv",
    "commot": "outputs/phase6/expanded_lr_universe/baselines/lymph_node_phase6_expanded/commot/human_lymph_node_visium_phase6_expanded_replicate_00__commot_predictions.csv",
    "cellchat": "outputs/phase6/expanded_lr_universe/baselines/lymph_node_phase6_expanded/cellchat/human_lymph_node_visium_phase6_expanded_replicate_00__cellchat_predictions.csv",
    "spatalk": "outputs/phase6/expanded_lr_universe/baselines/lymph_node_phase6_expanded/spatalk/human_lymph_node_visium_phase6_expanded_replicate_00__spatalk_predictions.csv",
    "cellnest": "outputs/phase6/expanded_lr_universe/baselines/lymph_node_phase6_expanded/cellnest/human_lymph_node_visium_phase6_expanded_replicate_00__cellnest_predictions.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/phase7_functional_cross_platform_validation.yaml",
    )
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=2701)
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    goal_dir = PROJECT_ROOT / config["goals"]["goal1_downstream_target_validation"]["output_dir"]
    raw_dir = goal_dir / "method_raw"
    goal_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = PROJECT_ROOT / config["source_artifacts"]["phase6_expanded_dataset_dir"]
    candidate_path = dataset_dir / "candidate_interactions.csv"
    expression_path = dataset_dir / "expression.csv"
    target_path = goal_dir / "phase7_downstream_target_signature_table.csv"
    _write_target_table(target_path)

    result_frames = []
    status_frames = []
    for method, rel_path in PREDICTION_PATHS.items():
        prediction_path = PROJECT_ROOT / rel_path
        method_dir = raw_dir / method
        result = run_downstream_target_correlation(
            candidate_path,
            prediction_path,
            expression_path,
            method_dir,
            pathway_csv=target_path,
            reactome_tf_ppr_csv=None,
            trrust_dir=None,
            score_column="score",
            min_targets=3,
            max_targets=200,
            n_permutations=args.n_permutations,
            random_seed=args.random_seed,
        )
        result["method"] = method
        result["prediction_path"] = str(prediction_path.relative_to(PROJECT_ROOT))
        result_frames.append(result)

        status_files = sorted(method_dir.glob("*__downstream_target_status.csv"))
        if status_files:
            status = pd.read_csv(status_files[-1])
            status["method"] = method
            status_frames.append(status)

    correlation = pd.concat(result_frames, ignore_index=True)
    correlation["phase7_priority_lr"] = correlation["lr_id"].isin(
        {
            "CCL19_CCR7_Tzone_semisynthetic",
            "CCL21_CCR7_Tzone_phase6_weak",
            "CXCL13_CXCR5_GC_semisynthetic_weak",
            "CD48_CD244_Tcell_phase6_contact_weak",
            "SPP1_CD44_boundary_semisynthetic",
        }
    )
    correlation_path = goal_dir / "phase7_lr_target_response_correlation.csv"
    correlation.to_csv(correlation_path, index=False)

    if status_frames:
        target_status = pd.concat(status_frames, ignore_index=True)
    else:
        target_status = pd.DataFrame()
    target_status.to_csv(goal_dir / "phase7_downstream_target_status_by_method.csv", index=False)

    comparison = _build_method_comparison(correlation)
    comparison_path = goal_dir / "phase7_method_target_response_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    fdr = _build_fdr_summary(correlation)
    fdr_path = goal_dir / "phase7_downstream_fdr.csv"
    fdr.to_csv(fdr_path, index=False)

    cross_tissue = _build_cross_tissue_summary(goal_dir, correlation)
    cross_tissue_path = goal_dir / "phase7_goal1_cross_tissue_downstream_summary.csv"
    cross_tissue.to_csv(cross_tissue_path, index=False)

    claims = _build_goal1_claims(cross_tissue)
    claims_path = goal_dir / "phase7_goal1_downstream_claims.csv"
    claims.to_csv(claims_path, index=False)

    summary = {
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "candidate_path": str(candidate_path.relative_to(PROJECT_ROOT)),
        "expression_path": str(expression_path.relative_to(PROJECT_ROOT)),
        "target_signature_table": str(target_path.relative_to(PROJECT_ROOT)),
        "correlation": str(correlation_path.relative_to(PROJECT_ROOT)),
        "method_comparison": str(comparison_path.relative_to(PROJECT_ROOT)),
        "fdr": str(fdr_path.relative_to(PROJECT_ROOT)),
        "cross_tissue_summary": str(cross_tissue_path.relative_to(PROJECT_ROOT)),
        "claims": str(claims_path.relative_to(PROJECT_ROOT)),
        "n_permutations": int(args.n_permutations),
        "methods": sorted(PREDICTION_PATHS),
        "n_tested_rows": int((correlation["status"] == "tested").sum()),
        "n_priority_tested_rows": int(((correlation["status"] == "tested") & correlation["phase7_priority_lr"]).sum()),
        "scope": "lymph_node_phase6_expanded_only",
        "has_lymph_node_target_validation": bool(((correlation["status"] == "tested") & correlation["phase7_priority_lr"]).any()),
        "has_non_lymph_node_target_validation": bool(cross_tissue["dataset_group"].isin(["tonsil", "melanoma"]).any()),
        "completion_note": "Goal 1 has lymph-node Phase 7 target signatures plus imported tonsil/melanoma curated downstream panels; non-lymph-node target coverage remains limited for several priority LR pairs.",
    }
    (goal_dir / "phase7_goal1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _write_target_table(path: Path) -> None:
    rows = []
    for row in LYMPH_NODE_TARGETS:
        rows.append(
            {
                "ligand": row["ligand"],
                "receptor": row["receptor"],
                "target_genes": row["target_genes"],
                "targets": row["target_genes"],
                "target_source": row["target_source"],
                "evidence_note": row["evidence_note"],
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _build_method_comparison(correlation: pd.DataFrame) -> pd.DataFrame:
    tested = correlation.loc[correlation["status"] == "tested"].copy()
    if tested.empty:
        return pd.DataFrame()
    rank_cols = ["lr_id", "spearman_r", "empirical_p_value", "bh_fdr"]
    rows = []
    for lr_id, group in tested.groupby("lr_id", sort=False):
        ranked = group.sort_values(["spearman_r", "bh_fdr"], ascending=[False, True]).reset_index(drop=True)
        spatial = ranked.loc[ranked["method"] == "spatialself"]
        best = ranked.iloc[0]
        row = {
            "lr_id": lr_id,
            "ligand": group["ligand"].iloc[0],
            "receptor": group["receptor"].iloc[0],
            "mechanism": group["mechanism"].iloc[0],
            "n_methods_tested": int(group["method"].nunique()),
            "best_method_by_target_r": best["method"],
            "best_target_spearman_r": best["spearman_r"],
            "spatialself_tested": not spatial.empty,
            "spatialself_target_spearman_r": float(spatial["spearman_r"].iloc[0]) if not spatial.empty else float("nan"),
            "spatialself_bh_fdr": float(spatial["bh_fdr"].iloc[0]) if not spatial.empty else float("nan"),
            "spatialself_rank_by_target_r": int(ranked.index[ranked["method"] == "spatialself"][0] + 1) if not spatial.empty else "",
            "phase7_priority_lr": bool(group["phase7_priority_lr"].iloc[0]),
        }
        for method in sorted(group["method"].unique()):
            method_row = group.loc[group["method"] == method].iloc[0]
            row[f"{method}_spearman_r"] = method_row["spearman_r"]
            row[f"{method}_bh_fdr"] = method_row["bh_fdr"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["phase7_priority_lr", "spatialself_target_spearman_r"], ascending=[False, False])


def _build_fdr_summary(correlation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in correlation.groupby("method", sort=True):
        tested = group["status"] == "tested"
        adjusted = bh_adjust(group.loc[tested, "empirical_p_value"].to_numpy(dtype=float)) if tested.any() else []
        method_group = group.copy()
        method_group["phase7_method_level_bh_fdr"] = float("nan")
        if tested.any():
            method_group.loc[tested, "phase7_method_level_bh_fdr"] = adjusted
        rows.append(method_group)
    fdr = pd.concat(rows, ignore_index=True)
    return fdr[
        [
            "method",
            "lr_id",
            "ligand",
            "receptor",
            "mechanism",
            "n_targets_present",
            "target_source",
            "spearman_r",
            "empirical_p_value",
            "bh_fdr",
            "phase7_method_level_bh_fdr",
            "status",
            "phase7_priority_lr",
        ]
    ].sort_values(["phase7_priority_lr", "method", "phase7_method_level_bh_fdr"], ascending=[False, True, True])


def _build_cross_tissue_summary(goal_dir: Path, lymph_node_correlation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.extend(_summarize_dataset_target_results("lymph_node", "phase7_expanded", lymph_node_correlation))
    imports = [
        (
            "tonsil",
            "phase2_revised_curated_downstream_panel",
            PROJECT_ROOT
            / "outputs/phase2_revised/goal1_curated_downstream_panel/real_background_v2_formal_tonsil",
        ),
        (
            "melanoma",
            "phase2_revised_curated_downstream_panel",
            PROJECT_ROOT
            / "outputs/phase2_revised/goal1_curated_downstream_panel/real_background_v2_formal_melanoma",
        ),
    ]
    imported_dir = goal_dir / "imported_phase2_curated_downstream"
    imported_dir.mkdir(parents=True, exist_ok=True)
    for dataset_group, source_label, source_dir in imports:
        imported = _load_phase2_curated_panel(source_dir)
        if imported.empty:
            rows.append(
                {
                    "dataset_group": dataset_group,
                    "source_label": source_label,
                    "method": "",
                    "n_lr_pairs": 0,
                    "n_tested_lr_pairs": 0,
                    "n_priority_positive_tested": 0,
                    "mean_priority_positive_spearman": float("nan"),
                    "mean_negative_spearman": float("nan"),
                    "positive_negative_delta": float("nan"),
                    "significant_priority_positive_count": 0,
                    "significant_negative_count": 0,
                    "target_coverage_note": "missing_imported_panel",
                    "source_path": str(source_dir.relative_to(PROJECT_ROOT)),
                }
            )
            continue
        imported.to_csv(imported_dir / f"{dataset_group}_curated_downstream_results.csv", index=False)
        rows.extend(_summarize_dataset_target_results(dataset_group, source_label, imported, source_dir))
    return pd.DataFrame(rows)


def _load_phase2_curated_panel(source_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(source_dir.glob("*/*__downstream_target_correlation.csv")):
        method = path.parent.name
        table = pd.read_csv(path)
        table["method"] = method
        table["source_path"] = str(path.relative_to(PROJECT_ROOT))
        frames.append(table)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _summarize_dataset_target_results(
    dataset_group: str,
    source_label: str,
    results: pd.DataFrame,
    source_dir: Path | None = None,
) -> list[dict]:
    rows = []
    if results.empty:
        return rows
    priority_by_group = {
        "lymph_node": {
            "CCL19_CCR7_Tzone_semisynthetic",
            "CCL21_CCR7_Tzone_phase6_weak",
            "CXCL13_CXCR5_GC_semisynthetic_weak",
            "CD48_CD244_Tcell_phase6_contact_weak",
            "SPP1_CD44_boundary_semisynthetic",
        },
        "tonsil": {
            "IL21_IL21R_tonsil_Tfh_B_weak",
            "CD40LG_CD40_tonsil_Tfh_B",
            "CXCL13_CXCR5_tonsil_GC",
        },
        "melanoma": {
            "CXCL9_CXCR3_melanoma_macrophage_T_weak",
            "SPP1_CD44_melanoma_macrophage_T_boundary",
            "PDCD1_CD274_melanoma_checkpoint",
        },
    }
    priority = priority_by_group.get(dataset_group, set())
    results = results.copy()
    results["is_priority_positive"] = results["lr_id"].isin(priority)
    results["is_negative"] = results["mechanism"].astype(str).eq("false_positive_high_expression")
    for method, group in results.groupby("method", sort=True):
        tested = group[group["status"].astype(str).eq("tested")]
        positive = tested[tested["is_priority_positive"]]
        negative = tested[tested["is_negative"]]
        priority_all = group[group["is_priority_positive"]].drop_duplicates("lr_id")
        rows.append(
            {
                "dataset_group": dataset_group,
                "source_label": source_label,
                "method": method,
                "n_lr_pairs": int(group["lr_id"].nunique()),
                "n_tested_lr_pairs": int(tested["lr_id"].nunique()),
                "n_priority_positive": int(len(priority_all)),
                "n_priority_positive_tested": int(positive["lr_id"].nunique()),
                "priority_positive_coverage_fraction": _safe_divide(
                    int(positive["lr_id"].nunique()), int(len(priority_all))
                ),
                "mean_priority_positive_spearman": _mean(positive["spearman_r"]),
                "mean_negative_spearman": _mean(negative["spearman_r"]),
                "positive_negative_delta": _mean(positive["spearman_r"]) - _mean(negative["spearman_r"]),
                "significant_priority_positive_count": int(
                    ((pd.to_numeric(positive["bh_fdr"], errors="coerce") <= 0.1)
                     & (pd.to_numeric(positive["spearman_r"], errors="coerce") > 0)).sum()
                )
                if len(positive)
                else 0,
                "significant_negative_count": int(
                    ((pd.to_numeric(negative["bh_fdr"], errors="coerce") <= 0.1)
                     & (pd.to_numeric(negative["spearman_r"], errors="coerce") > 0)).sum()
                )
                if len(negative)
                else 0,
                "target_coverage_note": "coverage_limited"
                if int(positive["lr_id"].nunique()) < int(len(priority_all))
                else "priority_targets_covered",
                "source_path": str(source_dir.relative_to(PROJECT_ROOT)) if source_dir else "phase7_goal1_lymph_node_outputs",
            }
        )
    return rows


def _build_goal1_claims(cross_tissue: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if cross_tissue.empty:
        return pd.DataFrame()
    for dataset_group, group in cross_tissue.groupby("dataset_group", sort=False):
        ranked = group.sort_values("positive_negative_delta", ascending=False)
        best = ranked.iloc[0]
        spatial = group[group["method"].astype(str).str.startswith("spatialself")]
        rows.append(
            {
                "dataset_group": dataset_group,
                "claim_status": "complete_with_coverage_limit"
                if group["n_priority_positive_tested"].max() > 0
                else "no_priority_target_validation",
                "best_delta_method": best["method"],
                "best_delta": best["positive_negative_delta"],
                "spatialself_best_delta": bool(str(best["method"]).startswith("spatialself")),
                "spatialself_max_delta": float(spatial["positive_negative_delta"].max()) if len(spatial) else float("nan"),
                "n_methods": int(group["method"].nunique()),
                "max_priority_positive_coverage_fraction": float(group["priority_positive_coverage_fraction"].max()),
                "limitation": "Some priority LR pairs lack usable target signatures; high-expression negatives can also correlate with broad response signatures.",
            }
        )
    return pd.DataFrame(rows)


def _mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")


def _safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


if __name__ == "__main__":
    main()
