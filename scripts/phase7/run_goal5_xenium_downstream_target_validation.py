#!/usr/bin/env python
"""Run downstream target validation for the Phase 7 Xenium subset.

The Xenium preview panel is targeted and small, so this uses curated
within-panel receiver signatures. These signatures are functional proxies for
cross-platform validation, not perturbation evidence or complete NicheNet
target sets.
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

GOAL5_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
DATASET_DIR = GOAL5_DIR / "standardized_inputs/xenium_human_lymph_node_subset"


TARGET_SIGNATURES = [
    {
        "ligand": "CCL19",
        "receptor": "CCR7",
        "target_genes": "CCR7,CD3D,CD3E,CD4,CD8A,CD28,CD27",
        "target_source": "phase7_xenium_within_panel_tzone_proxy",
        "evidence_note": "CCR7/T-cell receiver state genes available in the Xenium targeted panel.",
    },
    {
        "ligand": "CD274",
        "receptor": "PDCD1",
        "target_genes": "PDCD1,CD3D,CD3E,CD4,CD8A,CD28,CD27",
        "target_source": "phase7_xenium_within_panel_tcell_checkpoint_proxy",
        "evidence_note": "PDCD1/T-cell receiver state genes available in the Xenium targeted panel.",
    },
    {
        "ligand": "CD86",
        "receptor": "CD28",
        "target_genes": "CD28,CD3D,CD3E,CD4,CD8A,CD27,PDCD1",
        "target_source": "phase7_xenium_within_panel_tcell_costimulation_proxy",
        "evidence_note": "CD28/T-cell receiver state genes available in the Xenium targeted panel.",
    },
    {
        "ligand": "CD70",
        "receptor": "CD27",
        "target_genes": "CD27,CD3D,CD3E,CD4,CD8A,CD28,PDCD1",
        "target_source": "phase7_xenium_within_panel_tcell_costimulation_proxy",
        "evidence_note": "CD27/T-cell receiver state genes available in the Xenium targeted panel.",
    },
    {
        "ligand": "MS4A1",
        "receptor": "CD3D",
        "target_genes": "CD3D,CD3E,CD4,CD8A,CD28,CD27",
        "target_source": "phase7_xenium_negative_high_expression_control_proxy",
        "evidence_note": "Negative high-expression B/T marker pair; retained to audit false-positive target correlation.",
    },
    {
        "ligand": "CD19",
        "receptor": "CD3E",
        "target_genes": "CD3E,CD3D,CD4,CD8A,CD28,CD27",
        "target_source": "phase7_xenium_negative_high_expression_control_proxy",
        "evidence_note": "Negative high-expression B/T marker pair; retained to audit false-positive target correlation.",
    },
]


PREDICTIONS = {
    "spatialself_wavelet_edge_prior": GOAL5_DIR
    / "xenium_human_lymph_node_baseline_panel/spatialself_wavelet_edge_prior/"
    / "xenium_human_lymph_node_subset__spatialself_wavelet_predictions.csv",
    "commot": GOAL5_DIR
    / "xenium_human_lymph_node_commot/xenium_human_lymph_node_replicate_00__commot_predictions.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=7705)
    parser.add_argument(
        "--output-dir",
        default="outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_downstream_target_validation",
    )
    parser.add_argument("--dataset-dir", default=None)
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    dataset_dir = PROJECT_ROOT / args.dataset_dir if args.dataset_dir else DATASET_DIR
    raw_dir = output_dir / "method_raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    dataset_target_path = dataset_dir / "target_signatures.csv"
    if dataset_target_path.exists():
        target_path = dataset_target_path
    else:
        target_path = output_dir / "xenium_curated_within_panel_target_signatures.csv"
        _write_target_table(target_path)

    candidate_path = dataset_dir / "candidate_interactions.csv"
    expression_path = dataset_dir / "expression.csv"

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
    correlation.to_csv(output_dir / "xenium_downstream_target_correlation.csv", index=False)
    method_summary = _method_summary(correlation)
    method_summary.to_csv(output_dir / "xenium_downstream_method_summary.csv", index=False)
    claims = _claim_table(method_summary)
    claims.to_csv(output_dir / "xenium_downstream_claims.csv", index=False)

    manifest = {
        "dataset_dir": str(dataset_dir.relative_to(PROJECT_ROOT)),
        "candidate_path": str(candidate_path.relative_to(PROJECT_ROOT)),
        "expression_path": str(expression_path.relative_to(PROJECT_ROOT)),
        "target_signatures": str(target_path.relative_to(PROJECT_ROOT)),
        "correlation": str((output_dir / "xenium_downstream_target_correlation.csv").relative_to(PROJECT_ROOT)),
        "method_summary": str((output_dir / "xenium_downstream_method_summary.csv").relative_to(PROJECT_ROOT)),
        "claims": str((output_dir / "xenium_downstream_claims.csv").relative_to(PROJECT_ROOT)),
        "methods": sorted(PREDICTIONS),
        "n_permutations": int(args.n_permutations),
        "claim_boundary": (
            "Within-panel Xenium target response proxy only. This is not perturbation evidence, "
            "and the small Xenium gene panel limits target coverage."
        ),
    }
    (output_dir / "xenium_downstream_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(method_summary.to_string(index=False))
    print(json.dumps(manifest, indent=2))


def _write_target_table(path: Path) -> None:
    rows = []
    for row in TARGET_SIGNATURES:
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
        if positive_sig and not negative_sig:
            status = "functional_support_without_negative_fdr_signal"
        elif positive_sig and negative_sig:
            status = "mixed_functional_support_negative_controls_also_correlate"
        else:
            status = "no_positive_functional_support"
        rows.append(
            {
                "method": row.method,
                "claim_status": status,
                "allowed_claim": (
                    "Within-panel target response association is available, but claim strength is limited by "
                    "negative-control behavior and targeted-panel gene coverage."
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
