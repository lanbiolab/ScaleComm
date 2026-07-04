#!/usr/bin/env python
"""Run downstream target validation for post-Xenium colon cancer Goal 5."""

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


GOAL5_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
DATASET_DIR = GOAL5_DIR / "standardized_inputs/post_xenium_human_colon_cancer_subset"

TARGET_SIGNATURES = [
    {
        "ligand": "TGFB1",
        "receptor": "TGFBR2",
        "target_genes": "TGFBR2,COL1A1,COL1A2,DCN,LUM,ACTA2",
        "target_source": "phase7_post_xenium_crc_within_panel_tgfb_ecm_proxy",
        "evidence_note": "TGF-beta/ECM receiver-state genes available in the Visium whole-transcriptome probe set.",
    },
    {
        "ligand": "SPP1",
        "receptor": "CD44",
        "target_genes": "CD44,SPP1,LYZ,CD68,CD14,C1QA,C1QB,COL1A1",
        "target_source": "phase7_post_xenium_crc_within_panel_spp1_myeloid_stromal_proxy",
        "evidence_note": "SPP1-CD44 inflammatory myeloid/stromal receiver-state proxy genes.",
    },
    {
        "ligand": "CXCL12",
        "receptor": "CXCR4",
        "target_genes": "CXCR4,CD3D,CD3E,CD4,CD8A,PTPRC,LYZ",
        "target_source": "phase7_post_xenium_crc_within_panel_cxcl12_immune_proxy",
        "evidence_note": "CXCR4 and immune receiver-state genes available in the selected panel.",
    },
    {
        "ligand": "VEGFA",
        "receptor": "KDR",
        "target_genes": "KDR,PECAM1,VWF,ACTA2,COL1A1,COL1A2",
        "target_source": "phase7_post_xenium_crc_within_panel_vegfa_vascular_proxy",
        "evidence_note": "Vascular/stromal receiver-state proxy genes for VEGFA-KDR.",
    },
    {
        "ligand": "EGF",
        "receptor": "EGFR",
        "target_genes": "EGFR,EPCAM,KRT8,KRT19,MUC1",
        "target_source": "phase7_post_xenium_crc_within_panel_egf_epithelial_proxy",
        "evidence_note": "Epithelial receiver-state proxy genes for EGF-EGFR.",
    },
    {
        "ligand": "ACTB",
        "receptor": "KRT8",
        "target_genes": "KRT8,EPCAM,KRT19,MUC1,EGFR",
        "target_source": "phase7_post_xenium_crc_negative_high_expression_control_proxy",
        "evidence_note": "Negative high-expression housekeeping/epithelial pair; retained to audit false-positive target correlation.",
    },
    {
        "ligand": "COL1A1",
        "receptor": "EPCAM",
        "target_genes": "EPCAM,KRT8,KRT19,MUC1,EGFR",
        "target_source": "phase7_post_xenium_crc_negative_high_expression_control_proxy",
        "evidence_note": "Negative high-expression ECM/epithelial marker pair; retained to audit false-positive target correlation.",
    },
]

PREDICTIONS = {
    "spatialself_default": GOAL5_DIR
    / "post_xenium_human_colon_cancer_baseline_panel/spatialself_wavelet_edge_prior/"
    / "post_xenium_human_colon_cancer_subset__spatialself_wavelet_predictions.csv",
    "commot": GOAL5_DIR
    / "post_xenium_human_colon_cancer_baseline_panel/commot/"
    / "post_xenium_human_colon_cancer_subset__commot_predictions.csv",
    "spatialself_calibrated": GOAL5_DIR
    / "post_xenium_human_colon_cancer_spatialself_calibrated/spatialself_wavelet_edge_prior/"
    / "post_xenium_human_colon_cancer_subset__spatialself_wavelet_predictions.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=7719)
    parser.add_argument(
        "--output-dir",
        default="outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_downstream_target_validation",
    )
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    raw_dir = output_dir / "method_raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    target_path = output_dir / "post_xenium_colon_curated_target_signatures.csv"
    _write_target_table(target_path)

    candidate_path = DATASET_DIR / "candidate_interactions.csv"
    expression_path = DATASET_DIR / "expression.csv"

    frames = []
    for method, prediction_path in PREDICTIONS.items():
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
    correlation.to_csv(output_dir / "post_xenium_colon_downstream_target_correlation.csv", index=False)
    method_summary = _method_summary(correlation)
    method_summary.to_csv(output_dir / "post_xenium_colon_downstream_method_summary.csv", index=False)
    claims = _claim_table(method_summary)
    claims.to_csv(output_dir / "post_xenium_colon_downstream_claims.csv", index=False)

    manifest = {
        "dataset_dir": str(DATASET_DIR.relative_to(PROJECT_ROOT)),
        "candidate_path": str(candidate_path.relative_to(PROJECT_ROOT)),
        "expression_path": str(expression_path.relative_to(PROJECT_ROOT)),
        "target_signatures": str(target_path.relative_to(PROJECT_ROOT)),
        "correlation": str((output_dir / "post_xenium_colon_downstream_target_correlation.csv").relative_to(PROJECT_ROOT)),
        "method_summary": str((output_dir / "post_xenium_colon_downstream_method_summary.csv").relative_to(PROJECT_ROOT)),
        "claims": str((output_dir / "post_xenium_colon_downstream_claims.csv").relative_to(PROJECT_ROOT)),
        "methods": sorted(PREDICTIONS),
        "n_permutations": int(args.n_permutations),
        "claim_boundary": (
            "Within-panel/selected-gene target response proxy on a real 10x Visium CytAssist colon cancer subset. "
            "This is not perturbation evidence and does not provide native CCC ground truth."
        ),
    }
    (output_dir / "post_xenium_colon_downstream_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(method_summary.to_string(index=False))
    print(json.dumps(manifest, indent=2))


def _write_target_table(path: Path) -> None:
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


def _method_summary(correlation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in correlation.groupby("method", sort=True):
        tested = group[group["status"].astype(str).eq("tested")].copy()
        positive = tested[~tested["lr_id"].astype(str).str.contains("negative", case=False, na=False)]
        negative = tested[tested["lr_id"].astype(str).str.contains("negative", case=False, na=False)]
        rows.append(
            {
                "method": method,
                "n_lr_pairs": int(group["lr_id"].nunique()),
                "n_tested": int(tested["lr_id"].nunique()),
                "best_positive_lr": _best_lr(positive),
                "best_positive_spearman": _best_value(positive, "spearman_r"),
                "best_positive_bh_fdr": _best_value(positive, "bh_fdr"),
                "best_negative_lr": _best_lr(negative),
                "best_negative_spearman": _best_value(negative, "spearman_r"),
                "best_negative_bh_fdr": _best_value(negative, "bh_fdr"),
                "n_positive_fdr_0_1": int(((positive["bh_fdr"] <= 0.1) & (positive["spearman_r"] > 0)).sum()) if len(positive) else 0,
                "n_negative_fdr_0_1": int(((negative["bh_fdr"] <= 0.1) & (negative["spearman_r"] > 0)).sum()) if len(negative) else 0,
                "mean_positive_spearman": _mean(positive["spearman_r"]),
                "mean_negative_spearman": _mean(negative["spearman_r"]),
                "positive_negative_delta": _mean(positive["spearman_r"]) - _mean(negative["spearman_r"]),
            }
        )
    return pd.DataFrame(rows)


def _claim_table(method_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in method_summary.itertuples(index=False):
        negative_sig = int(getattr(row, "n_negative_fdr_0_1")) > 0
        positive_sig = int(getattr(row, "n_positive_fdr_0_1")) > 0
        delta = float(getattr(row, "positive_negative_delta"))
        if positive_sig and not negative_sig and delta > 0:
            status = "functional_support_without_negative_fdr_signal"
        elif positive_sig and negative_sig:
            status = "mixed_functional_support_negative_controls_also_correlate"
        elif positive_sig:
            status = "positive_signal_but_no_positive_negative_separation"
        else:
            status = "no_positive_functional_support"
        rows.append(
            {
                "method": row.method,
                "claim_status": status,
                "allowed_claim": (
                    "Selected-gene target response association is available, but claim strength is limited by "
                    "negative-control behavior, proxy target signatures, and absence of perturbation evidence."
                ),
                "best_positive_lr": row.best_positive_lr,
                "best_negative_lr": row.best_negative_lr,
                "positive_negative_delta": row.positive_negative_delta,
            }
        )
    return pd.DataFrame(rows)


def _best_lr(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    row = frame.sort_values(["spearman_r", "bh_fdr"], ascending=[False, True]).iloc[0]
    return str(row["lr_id"])


def _best_value(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        return float("nan")
    row = frame.sort_values(["spearman_r", "bh_fdr"], ascending=[False, True]).iloc[0]
    return float(row[column])


def _mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")


if __name__ == "__main__":
    main()
