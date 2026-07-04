#!/usr/bin/env python
"""Run formal curated downstream target validation panel.

This uses available pathway/Reactome/TRRUST target resources. Coverage is
reported explicitly because several chemokine-like injected controls do not
have observed target signatures in the current formal expression panels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_downstream_target_correlation import run_downstream_target_correlation


DATASETS = {
    "real_background_v2_formal_lymph_node": {
        "replicate_dir": "outputs/phase2_revised/goal4_real_background_v2/formal_datasets/human_lymph_node_visium_replicate_00",
        "prediction_dataset": "real_background_v2_formal",
    },
    "real_background_v2_formal_tonsil": {
        "replicate_dir": "outputs/phase2_revised/goal4_real_background_v2/formal_datasets/human_tonsil_slide_tags_replicate_00",
        "prediction_dataset": "real_background_v2_formal",
    },
    "real_background_v2_formal_melanoma": {
        "replicate_dir": "outputs/phase2_revised/goal4_real_background_v2/formal_datasets/human_melanoma_multiome_replicate_00",
        "prediction_dataset": "real_background_v2_formal",
    },
}

PREDICTION_ROOTS = {
    "spatialself_wavelet": "outputs/phase2_revised/goal2_formal_baseline_panel_extended",
    "spatialself_wavelet_edge_prior": "outputs/phase2_revised/goal2_formal_baseline_panel",
    "commot": "outputs/phase2_revised/goal2_formal_baseline_panel",
    "cellchat": "outputs/phase2_revised/goal2_formal_baseline_panel",
    "spatalk": "outputs/phase2_revised/goal2_formal_baseline_panel_extended",
    "cellnest": "outputs/phase2_revised/goal2_formal_baseline_panel_extended",
}

METHODS = (
    "spatialself_wavelet",
    "spatialself_wavelet_edge_prior",
    "commot",
    "cellchat",
    "spatalk",
    "cellnest",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/phase2_revised/goal1_curated_downstream_panel")
    parser.add_argument("--n-permutations", type=int, default=50)
    parser.add_argument("--min-targets", type=int, default=3)
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    result_parts: list[pd.DataFrame] = []
    status_rows: list[dict] = []
    for dataset, dataset_cfg in DATASETS.items():
        replicate_dir = PROJECT_ROOT / dataset_cfg["replicate_dir"]
        candidate_path = replicate_dir / "candidate_interactions.csv"
        expression_path = replicate_dir / "expression.csv"
        for method in METHODS:
            prediction_path = _prediction_path(
                PROJECT_ROOT / PREDICTION_ROOTS[method],
                dataset_cfg["prediction_dataset"],
                method,
                replicate_dir.name,
            )
            method_out = output_dir / dataset / method
            try:
                result = run_downstream_target_correlation(
                    candidate_path,
                    prediction_path,
                    expression_path,
                    method_out,
                    pathway_csv=PROJECT_ROOT / "downstreamgene_eva/valid_single_pathway.csv",
                    reactome_tf_ppr_csv=PROJECT_ROOT / "downstreamgene_eva/scseqcomm/data/TF_PPR_REACTOME_human.csv",
                    trrust_dir=PROJECT_ROOT / "downstreamgene_eva/scseqcomm/data/TF_TG_TRRUSTv2",
                    min_targets=int(args.min_targets),
                    max_targets=200,
                    n_permutations=int(args.n_permutations),
                    random_seed=2026,
                )
                result = result.copy()
                result["dataset"] = dataset
                result["prediction_path"] = str(prediction_path.relative_to(PROJECT_ROOT))
                result_parts.append(result)
                status_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "status": "completed",
                        "detail": "",
                        "n_rows": int(len(result)),
                        "n_tested": int((result["status"].astype(str) == "tested").sum()),
                        "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                status_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "status": "failed",
                        "detail": f"{type(exc).__name__}: {exc}",
                        "n_rows": 0,
                        "n_tested": 0,
                        "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)) if prediction_path.exists() else str(prediction_path),
                    }
                )

    results = pd.concat(result_parts, ignore_index=True) if result_parts else pd.DataFrame()
    status = pd.DataFrame(status_rows)
    method_summary = _method_summary(results)
    coverage = _coverage_summary(results)
    claims = _claims(method_summary, coverage, status)

    results.to_csv(output_dir / "goal1_curated_downstream_results.csv", index=False)
    method_summary.to_csv(output_dir / "goal1_curated_downstream_method_summary.csv", index=False)
    coverage.to_csv(output_dir / "goal1_curated_downstream_target_coverage.csv", index=False)
    claims.to_csv(output_dir / "goal1_curated_downstream_claims.csv", index=False)
    status.to_csv(output_dir / "goal1_curated_downstream_status.csv", index=False)
    manifest = {
        "analysis_label": "phase2_revised_goal1_curated_downstream_panel",
        "canonical_goal": "goal1_baseline_matched_biological_validation",
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "datasets": sorted(DATASETS),
        "methods": list(METHODS),
        "n_permutations": int(args.n_permutations),
        "min_targets": int(args.min_targets),
        "claim_level": "curated_downstream_coverage_limited_not_manuscript_ready",
        "target_sources": [
            "downstreamgene_eva/valid_single_pathway.csv",
            "downstreamgene_eva/scseqcomm/data/TF_PPR_REACTOME_human.csv",
            "downstreamgene_eva/scseqcomm/data/TF_TG_TRRUSTv2",
        ],
        "outputs": {
            "results": str((output_dir / "goal1_curated_downstream_results.csv").relative_to(PROJECT_ROOT)),
            "method_summary": str((output_dir / "goal1_curated_downstream_method_summary.csv").relative_to(PROJECT_ROOT)),
            "coverage": str((output_dir / "goal1_curated_downstream_target_coverage.csv").relative_to(PROJECT_ROOT)),
            "claims": str((output_dir / "goal1_curated_downstream_claims.csv").relative_to(PROJECT_ROOT)),
            "status": str((output_dir / "goal1_curated_downstream_status.csv").relative_to(PROJECT_ROOT)),
        },
        "limitations": [
            "Target coverage is incomplete for several chemokine-like positive controls.",
            "Housekeeping false-positive controls can have broad Reactome/TRRUST target coverage.",
            "This panel is useful as a coverage-limited status result, not as final downstream superiority evidence.",
        ],
    }
    (output_dir / "goal1_curated_downstream_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not claims.empty:
        print(claims.to_string(index=False))
    print(f"Wrote curated downstream panel outputs to {output_dir}")


def _prediction_path(prediction_root: Path, dataset: str, method: str, replicate_name: str) -> Path:
    if method == "spatialself_wavelet_edge_prior":
        filename = f"{replicate_name}__spatialself_wavelet_predictions.csv"
    else:
        filename = f"{replicate_name}__{method}_predictions.csv"
    return prediction_root / dataset / method / filename


def _method_summary(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    rows = []
    for (dataset, method), group in results.groupby(["dataset", "method"], sort=False):
        tested = group[group["status"].astype(str) == "tested"]
        positive = tested[~tested["mechanism"].astype(str).eq("false_positive_high_expression")]
        negative = tested[tested["mechanism"].astype(str).eq("false_positive_high_expression")]
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "n_lr_pairs": int(group["lr_id"].nunique()),
                "n_tested_lr_pairs": int(tested["lr_id"].nunique()),
                "n_positive_tested": int(positive["lr_id"].nunique()),
                "n_negative_tested": int(negative["lr_id"].nunique()),
                "mean_positive_spearman": _mean(positive["spearman_r"]),
                "mean_negative_spearman": _mean(negative["spearman_r"]),
                "positive_negative_spearman_delta": _mean(positive["spearman_r"]) - _mean(negative["spearman_r"]),
                "significant_positive_count": int(((positive["bh_fdr"].astype(float) <= 0.1) & (positive["spearman_r"].astype(float) > 0)).sum()) if len(positive) else 0,
                "significant_negative_count": int(((negative["bh_fdr"].astype(float) <= 0.1) & (negative["spearman_r"].astype(float) > 0)).sum()) if len(negative) else 0,
                "target_validation_level": "curated_pathway_reactome_trrust_coverage_limited",
            }
        )
    return pd.DataFrame(rows)


def _coverage_summary(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    rows = []
    for dataset, group in results.groupby("dataset", sort=False):
        lr_once = group.drop_duplicates(["lr_id"])
        positive = lr_once[~lr_once["mechanism"].astype(str).eq("false_positive_high_expression")]
        negative = lr_once[lr_once["mechanism"].astype(str).eq("false_positive_high_expression")]
        rows.append(
            {
                "dataset": dataset,
                "n_lr_pairs": int(lr_once["lr_id"].nunique()),
                "n_positive_lr_pairs": int(positive["lr_id"].nunique()),
                "n_negative_lr_pairs": int(negative["lr_id"].nunique()),
                "n_positive_with_targets": int((positive["n_targets_present"].astype(int) >= 3).sum()),
                "n_negative_with_targets": int((negative["n_targets_present"].astype(int) >= 3).sum()),
                "positive_target_coverage_fraction": _safe_divide(int((positive["n_targets_present"].astype(int) >= 3).sum()), int(positive["lr_id"].nunique())),
                "negative_target_coverage_fraction": _safe_divide(int((negative["n_targets_present"].astype(int) >= 3).sum()), int(negative["lr_id"].nunique())),
                "coverage_limitation": "chemokine_positive_targets_missing" if int((positive["n_targets_present"].astype(int) >= 3).sum()) < int(positive["lr_id"].nunique()) else "positive_targets_covered",
            }
        )
    return pd.DataFrame(rows)


def _claims(method_summary: pd.DataFrame, coverage: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    if method_summary.empty:
        return pd.DataFrame()
    rows = []
    for dataset, group in method_summary.groupby("dataset", sort=False):
        ranked = group.sort_values("positive_negative_spearman_delta", ascending=False)
        best = ranked.iloc[0]
        spatial = group[group["method"].astype(str).str.startswith("spatialself")]
        non_spatial = group[~group["method"].astype(str).str.startswith("spatialself")]
        cov = coverage[coverage["dataset"].astype(str) == str(dataset)]
        rows.append(
            {
                "dataset": dataset,
                "claim_level": "curated_downstream_coverage_limited_not_manuscript_ready",
                "best_delta_method": best["method"],
                "best_delta": best["positive_negative_spearman_delta"],
                "spatialself_best_delta": str(best["method"]).startswith("spatialself"),
                "spatialself_max_delta": float(spatial["positive_negative_spearman_delta"].max()) if len(spatial) else float("nan"),
                "non_spatial_max_delta": float(non_spatial["positive_negative_spearman_delta"].max()) if len(non_spatial) else float("nan"),
                "positive_target_coverage_fraction": float(cov["positive_target_coverage_fraction"].iloc[0]) if len(cov) else float("nan"),
                "negative_target_coverage_fraction": float(cov["negative_target_coverage_fraction"].iloc[0]) if len(cov) else float("nan"),
                "coverage_limited": bool(len(cov) and float(cov["positive_target_coverage_fraction"].iloc[0]) < 1.0),
                "completed_methods": int((status[(status["dataset"] == dataset)]["status"].astype(str) == "completed").sum()) if not status.empty else 0,
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
