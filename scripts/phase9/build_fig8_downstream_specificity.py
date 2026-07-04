#!/usr/bin/env python
"""Build Fig. 8 multi-method downstream specificity source data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_downstream_target_correlation import run_downstream_target_correlation
from scripts.phase7.run_goal5_post_xenium_colon_downstream_target_validation import (
    TARGET_SIGNATURES,
)


GOAL5_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
DATASET_DIR = GOAL5_DIR / "standardized_inputs/post_xenium_human_colon_cancer_subset"

PREDICTIONS = {
    "spatialself": GOAL5_DIR
    / "post_xenium_human_colon_cancer_spatialself_calibrated/spatialself_wavelet_edge_prior/"
    / "post_xenium_human_colon_cancer_subset__spatialself_wavelet_predictions.csv",
    "commot": GOAL5_DIR
    / "post_xenium_human_colon_cancer_baseline_panel/commot/"
    / "post_xenium_human_colon_cancer_subset__commot_predictions.csv",
    "cellchat": GOAL5_DIR
    / "post_xenium_human_colon_cancer_extended_baselines/cellchat/"
    / "post_xenium_human_colon_cancer_subset__cellchat_predictions.csv",
    "cellnest": GOAL5_DIR
    / "post_xenium_human_colon_cancer_extended_baselines/cellnest/"
    / "post_xenium_human_colon_cancer_subset__cellnest_predictions.csv",
    "spatalk": GOAL5_DIR
    / "post_xenium_human_colon_cancer_extended_baselines/spatalk/"
    / "post_xenium_human_colon_cancer_subset__spatalk_predictions.csv",
}

METHOD_ORDER = ["spatialself", "commot", "cellchat", "cellnest", "spatalk"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-permutations", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=7719)
    parser.add_argument(
        "--output-dir",
        default="outputs/phase9/fig8_post_xenium_spp1_cd44/downstream_specificity",
    )
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    raw_dir = output_dir / "method_raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    target_path = output_dir / "fig8_post_xenium_colon_curated_target_signatures.csv"
    write_target_table(target_path)

    candidate_path = DATASET_DIR / "candidate_interactions.csv"
    expression_path = DATASET_DIR / "expression.csv"

    frames = []
    status_rows = []
    for method in METHOD_ORDER:
        prediction_path = PREDICTIONS[method]
        score_status = inspect_prediction_scores(prediction_path)
        status_rows.append({"method": method, **score_status})

        if score_status["is_constant"]:
            result = constant_method_result(candidate_path, method)
        else:
            result = run_downstream_target_correlation(
                candidate_path,
                prediction_path,
                expression_path,
                raw_dir / method,
                pathway_csv=target_path,
                reactome_tf_ppr_csv=None,
                trrust_dir=None,
                score_column="score",
                min_targets=3,
                max_targets=50,
                n_permutations=args.n_permutations,
                random_seed=args.random_seed,
            )
        result["method"] = method
        result["prediction_path"] = str(prediction_path.relative_to(PROJECT_ROOT))
        frames.append(result)

    correlation = pd.concat(frames, ignore_index=True)
    specificity = build_specificity(correlation)
    status = pd.DataFrame(status_rows)

    correlation_path = output_dir / "fig8_multimethod_downstream_correlation.csv"
    specificity_path = output_dir / "fig8_downstream_specificity.csv"
    status_path = output_dir / "fig8_downstream_method_status.csv"
    correlation.to_csv(correlation_path, index=False)
    specificity.to_csv(specificity_path, index=False)
    status.to_csv(status_path, index=False)

    manifest = {
        "dataset_dir": str(DATASET_DIR.relative_to(PROJECT_ROOT)),
        "candidate_path": str(candidate_path.relative_to(PROJECT_ROOT)),
        "expression_path": str(expression_path.relative_to(PROJECT_ROOT)),
        "target_signatures": str(target_path.relative_to(PROJECT_ROOT)),
        "correlation": str(correlation_path.relative_to(PROJECT_ROOT)),
        "specificity": str(specificity_path.relative_to(PROJECT_ROOT)),
        "method_status": str(status_path.relative_to(PROJECT_ROOT)),
        "methods": METHOD_ORDER,
        "n_permutations": int(args.n_permutations),
        "claim_boundary": (
            "Downstream target response is a selected-gene receiver-state association in a real-background "
            "post-Xenium colon subset. It is not perturbation evidence or native CCC ground truth."
        ),
    }
    (output_dir / "fig8_downstream_specificity_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(specificity.to_string(index=False))
    print(json.dumps(manifest, indent=2))


def write_target_table(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "ligand": row["ligand"],
                "receptor": row["receptor"],
                "target_genes": row["target_genes"],
                "targets": row["target_genes"],
                "target_source": row["target_source"],
                "evidence_note": row["evidence_note"],
            }
            for row in TARGET_SIGNATURES
        ]
    ).to_csv(path, index=False)


def inspect_prediction_scores(path: Path) -> dict[str, object]:
    scores = pd.to_numeric(pd.read_csv(path, usecols=["score"])["score"], errors="coerce").fillna(0)
    return {
        "n_edges": int(len(scores)),
        "n_unique_scores": int(scores.nunique()),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_std": float(scores.std()),
        "is_constant": bool(scores.nunique() <= 1 or scores.std() <= 1e-12),
    }


def constant_method_result(candidate_path: Path, method: str) -> pd.DataFrame:
    candidates = pd.read_csv(candidate_path)
    rows = []
    for row in candidates[["lr_id", "ligand", "receptor", "mechanism"]].drop_duplicates().itertuples(index=False):
        rows.append(
            {
                "lr_id": row.lr_id,
                "ligand": row.ligand,
                "receptor": row.receptor,
                "mechanism": row.mechanism,
                "method": method,
                "n_edges": int((candidates["lr_id"] == row.lr_id).sum()),
                "n_receiver_cells": int(candidates.loc[candidates["lr_id"] == row.lr_id, "receiver_id"].nunique()),
                "n_targets_present": np.nan,
                "target_source": "not_evaluated_constant_scores",
                "spearman_r": np.nan,
                "spearman_p_value": np.nan,
                "empirical_p_value": np.nan,
                "bh_fdr": np.nan,
                "status": "constant_prediction_scores",
            }
        )
    return pd.DataFrame(rows)


def build_specificity(correlation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    correlation = correlation.copy()
    correlation["pair"] = correlation["ligand"].astype(str) + "-" + correlation["receptor"].astype(str)
    for method in METHOD_ORDER:
        group = correlation[correlation["method"].eq(method)].copy()
        tested = group[group["status"].eq("tested")].copy()
        spp1 = tested[tested["pair"].eq("SPP1-CD44")]
        negatives = tested[tested["mechanism"].astype(str).eq("negative_control_high_expression")]
        if spp1.empty or negatives.empty:
            rows.append(
                {
                    "method": method,
                    "status": "not_evaluable",
                    "spp1_rho": np.nan,
                    "spp1_fdr": np.nan,
                    "best_negative_pair": "",
                    "best_negative_rho": np.nan,
                    "best_negative_fdr": np.nan,
                    "delta_rho": np.nan,
                    "n_tested": int(tested["lr_id"].nunique()),
                }
            )
            continue
        spp1_row = spp1.iloc[0]
        neg_row = negatives.sort_values(["spearman_r", "bh_fdr"], ascending=[False, True]).iloc[0]
        rows.append(
            {
                "method": method,
                "status": "tested",
                "spp1_rho": float(spp1_row["spearman_r"]),
                "spp1_fdr": float(spp1_row["bh_fdr"]),
                "best_negative_pair": str(neg_row["pair"]),
                "best_negative_rho": float(neg_row["spearman_r"]),
                "best_negative_fdr": float(neg_row["bh_fdr"]),
                "delta_rho": float(spp1_row["spearman_r"] - neg_row["spearman_r"]),
                "n_tested": int(tested["lr_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
