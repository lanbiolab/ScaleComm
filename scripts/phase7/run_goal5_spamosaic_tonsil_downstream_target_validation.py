#!/usr/bin/env python
"""Run downstream target validation for the SPAMOSAIC tonsil Goal 5 subset."""

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
DEFAULT_SECTION_ID = "spamosaic_tonsil_A1"
DATASET_DIR = GOAL5_DIR / f"standardized_inputs/{DEFAULT_SECTION_ID}_subset"

TARGET_SIGNATURES = [
    {
        "ligand": "CXCL13",
        "receptor": "CXCR5",
        "target_genes": "CXCR5,MS4A1,CD19,CR2,CD40,TNFRSF13C",
        "target_source": "phase7_spamosaic_tonsil_gc_bcell_proxy",
        "evidence_note": "Germinal-center B/FDC receiver-state proxy genes available in the RNA panel.",
    },
    {
        "ligand": "CCL21",
        "receptor": "CCR7",
        "target_genes": "CCR7,CD3D,CD3E,CD4,CD8A",
        "target_source": "phase7_spamosaic_tonsil_tzone_proxy",
        "evidence_note": "CCR7/T-cell receiver-state proxy genes available in the RNA panel.",
    },
    {
        "ligand": "CD40LG",
        "receptor": "CD40",
        "target_genes": "CD40,MS4A1,CD19,TNFRSF13C,CR2",
        "target_source": "phase7_spamosaic_tonsil_tfh_bcell_proxy",
        "evidence_note": "B-cell activation receiver-state proxy for CD40 signaling.",
    },
    {
        "ligand": "IL21",
        "receptor": "IL21R",
        "target_genes": "IL21R,MS4A1,CD19,CD40,TNFRSF13C,CR2",
        "target_source": "phase7_spamosaic_tonsil_il21_bcell_proxy",
        "evidence_note": "IL21R/B-cell receiver-state proxy genes in tonsil.",
    },
    {
        "ligand": "TNFSF13B",
        "receptor": "TNFRSF13C",
        "target_genes": "TNFRSF13C,MS4A1,CD19,CD40,CR2",
        "target_source": "phase7_spamosaic_tonsil_baff_bcell_proxy",
        "evidence_note": "BAFF receptor and B-cell receiver-state proxy genes.",
    },
    {
        "ligand": "ACTB",
        "receptor": "EGFR",
        "target_genes": "EGFR,EPCAM,VIM,ACTA2",
        "target_source": "phase7_spamosaic_tonsil_negative_control_proxy",
        "evidence_note": "Negative high-expression housekeeping/epithelial pair for target-correlation leakage audit.",
    },
    {
        "ligand": "MS4A1",
        "receptor": "CD3E",
        "target_genes": "CD3E,CD3D,CD4,CD8A",
        "target_source": "phase7_spamosaic_tonsil_negative_control_proxy",
        "evidence_note": "Negative high-expression B/T marker pair for target-correlation leakage audit.",
    },
]

PREDICTIONS = {
    "spatialself_wavelet_edge_prior": "{section_id}_spatialself/spatialself_wavelet_edge_prior/replicate_00__spatialself_wavelet_predictions.csv",
    "commot": "{section_id}_commot/replicate_00__commot_predictions.csv",
    "cellchat": "{section_id}_cellchat/replicate_00__cellchat_predictions.csv",
    "spatalk": "{section_id}_spatalk/replicate_00__spatalk_predictions.csv",
    "cellnest": "{section_id}_cellnest/replicate_00__cellnest_predictions.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section-id", default=DEFAULT_SECTION_ID)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=7721)
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/"
            "spamosaic_tonsil_A1_downstream_target_validation"
        ),
    )
    args = parser.parse_args()

    section_id = str(args.section_id)
    dataset_dir = PROJECT_ROOT / args.dataset_dir if args.dataset_dir else GOAL5_DIR / f"standardized_inputs/{section_id}_subset"
    output_dir = PROJECT_ROOT / args.output_dir
    raw_dir = output_dir / "method_raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    candidate_path = dataset_dir / "candidate_interactions.csv"
    expression_path = dataset_dir / "expression.csv"
    dataset_target_path = dataset_dir / "target_signatures.csv"
    if dataset_target_path.exists():
        target_path = dataset_target_path
    else:
        target_path = output_dir / f"{section_id}_curated_target_signatures.csv"
        _write_target_table(target_path)

    frames = []
    for method, prediction_template in PREDICTIONS.items():
        prediction_path = GOAL5_DIR / prediction_template.format(section_id=section_id)
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
    correlation.to_csv(output_dir / f"{section_id}_downstream_target_correlation.csv", index=False)
    method_summary = _method_summary(correlation)
    method_summary.to_csv(output_dir / f"{section_id}_downstream_method_summary.csv", index=False)
    claims = _claim_table(method_summary)
    claims.to_csv(output_dir / f"{section_id}_downstream_claims.csv", index=False)

    manifest = {
        "section_id": section_id,
        "dataset_dir": str(dataset_dir.relative_to(PROJECT_ROOT)),
        "candidate_path": str(candidate_path.relative_to(PROJECT_ROOT)),
        "expression_path": str(expression_path.relative_to(PROJECT_ROOT)),
        "target_signatures": str(target_path.relative_to(PROJECT_ROOT)),
        "correlation": str((output_dir / f"{section_id}_downstream_target_correlation.csv").relative_to(PROJECT_ROOT)),
        "method_summary": str((output_dir / f"{section_id}_downstream_method_summary.csv").relative_to(PROJECT_ROOT)),
        "claims": str((output_dir / f"{section_id}_downstream_claims.csv").relative_to(PROJECT_ROOT)),
        "methods": sorted(PREDICTIONS),
        "n_permutations": int(args.n_permutations),
        "claim_boundary": (
            "Curated RNA-panel target response proxy only. This is not perturbation evidence; "
            "cell-type/niche labels are ADT marker-rule proxies."
        ),
    }
    (output_dir / f"{section_id}_downstream_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
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
                "n_positive_fdr_0_1": int((pd.to_numeric(positive.get("bh_fdr", pd.Series(dtype=float)), errors="coerce") <= 0.1).sum()),
                "n_negative_fdr_0_1": int((pd.to_numeric(negative.get("bh_fdr", pd.Series(dtype=float)), errors="coerce") <= 0.1).sum()),
            }
        )
    return pd.DataFrame(rows)


def _claim_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in summary.itertuples(index=False):
        positive = float(getattr(row, "best_positive_spearman", 0) or 0) > 0 and float(
            getattr(row, "best_positive_bh_fdr", 1) or 1
        ) <= 0.1
        negative = int(getattr(row, "n_negative_fdr_0_1", 0) or 0) > 0
        if positive and not negative:
            status = "functional_support_without_negative_fdr_signal"
        elif positive and negative:
            status = "mixed_functional_support_negative_controls_also_correlate"
        else:
            status = "no_positive_functional_support"
        rows.append(
            {
                "method": row.method,
                "claim_status": status,
                "allowed_claim": (
                    "Selected LR activity is associated with receiver target-state genes without negative-control leakage."
                    if status == "functional_support_without_negative_fdr_signal"
                    else "Do not claim successful functional validation under the current control panel."
                ),
            }
        )
    return pd.DataFrame(rows)


def _best_lr(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    row = frame.sort_values(["bh_fdr", "spearman_r"], ascending=[True, False]).iloc[0]
    return str(row.get("canonical_pair", f"{row.get('ligand', '')}-{row.get('receptor', '')}"))


def _best_value(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    row = frame.sort_values(["bh_fdr", "spearman_r"], ascending=[True, False]).iloc[0]
    return float(row[column])


if __name__ == "__main__":
    main()
