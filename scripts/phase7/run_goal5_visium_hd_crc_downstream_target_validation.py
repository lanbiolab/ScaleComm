#!/usr/bin/env python
"""Run downstream target validation for Visium HD CRC P2CRC Goal 5."""

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
DATASET_DIR = GOAL5_DIR / "standardized_inputs/visium_hd_crc_p2crc_subset"
PREDICTIONS = {
    "spatialself_wavelet_edge_prior": GOAL5_DIR
    / "visium_hd_crc_p2crc_baseline_panel/spatialself_wavelet_edge_prior/"
    / "visium_hd_crc_p2crc_subset__spatialself_wavelet_predictions.csv",
    "spatialself_calibrated": GOAL5_DIR
    / "visium_hd_crc_p2crc_spatialself_calibrated/spatialself_wavelet_edge_prior/"
    / "visium_hd_crc_p2crc_subset__spatialself_wavelet_predictions.csv",
    "commot": GOAL5_DIR
    / "visium_hd_crc_p2crc_baseline_panel/commot/"
    / "visium_hd_crc_p2crc_subset__commot_predictions.csv",
    "cellchat": GOAL5_DIR
    / "visium_hd_crc_p2crc_baseline_panel/cellchat/"
    / "visium_hd_crc_p2crc_subset__cellchat_predictions.csv",
    "cellnest": GOAL5_DIR
    / "visium_hd_crc_p2crc_baseline_panel/cellnest/"
    / "visium_hd_crc_p2crc_subset__cellnest_predictions.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=202617)
    parser.add_argument(
        "--output-dir",
        default="outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_downstream_target_validation",
    )
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = PROJECT_ROOT / dataset_dir
    raw_dir = output_dir / "method_raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    target_path = dataset_dir / "target_signatures.csv"
    candidate_path = dataset_dir / "candidate_interactions.csv"
    expression_path = dataset_dir / "expression.csv"

    frames = []
    status_rows = []
    for method, prediction_path in PREDICTIONS.items():
        if not prediction_path.exists():
            status_rows.append(
                {
                    "method": method,
                    "status": "prediction_missing",
                    "prediction_path": _rel(prediction_path),
                    "detail": "prediction file not found",
                }
            )
            continue
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
        result["prediction_path"] = _rel(prediction_path)
        frames.append(result)
        status_rows.append(
            {
                "method": method,
                "status": "completed",
                "prediction_path": _rel(prediction_path),
                "detail": "downstream target validation completed",
            }
        )

    correlation = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    correlation.to_csv(output_dir / "visium_hd_crc_p2crc_downstream_target_correlation.csv", index=False)
    method_summary = _method_summary(correlation)
    method_summary.to_csv(output_dir / "visium_hd_crc_p2crc_downstream_method_summary.csv", index=False)
    claims = _claim_table(method_summary)
    claims.to_csv(output_dir / "visium_hd_crc_p2crc_downstream_claims.csv", index=False)
    pd.DataFrame(status_rows).to_csv(output_dir / "visium_hd_crc_p2crc_downstream_status.csv", index=False)

    manifest = {
        "dataset_dir": _rel(dataset_dir),
        "candidate_path": _rel(candidate_path),
        "expression_path": _rel(expression_path),
        "target_signatures": _rel(target_path),
        "correlation": _rel(output_dir / "visium_hd_crc_p2crc_downstream_target_correlation.csv"),
        "method_summary": _rel(output_dir / "visium_hd_crc_p2crc_downstream_method_summary.csv"),
        "claims": _rel(output_dir / "visium_hd_crc_p2crc_downstream_claims.csv"),
        "methods": sorted(PREDICTIONS),
        "n_permutations": int(args.n_permutations),
        "claim_boundary": (
            "Curated within-panel target-response association on a real 10x Visium HD CRC subset. "
            "This is functional-support evidence, not perturbation ground truth."
        ),
    }
    (output_dir / "visium_hd_crc_p2crc_downstream_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(method_summary.to_string(index=False))
    print(json.dumps(manifest, indent=2))


def _method_summary(correlation: pd.DataFrame) -> pd.DataFrame:
    if correlation.empty:
        return pd.DataFrame()
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
                "best_positive_lr": row.best_positive_lr,
                "best_negative_lr": row.best_negative_lr,
                "positive_negative_delta": row.positive_negative_delta,
                "allowed_claim": (
                    "Selected-gene target response association is available, but claim strength is limited by "
                    "proxy target signatures and absence of perturbation evidence."
                ),
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


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
