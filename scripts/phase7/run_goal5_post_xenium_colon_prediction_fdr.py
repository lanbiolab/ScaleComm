#!/usr/bin/env python
"""Run prediction-level permutation/FDR for post-Xenium colon Goal 5."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_permutation_fdr import run_permutation_fdr


GOAL_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
DATASET_DIR = GOAL_DIR / "standardized_inputs/post_xenium_human_colon_cancer_subset"
METHODS = {
    "spatialself_default": "post_xenium_human_colon_cancer_baseline_panel/spatialself_wavelet_edge_prior/post_xenium_human_colon_cancer_subset__spatialself_wavelet_predictions.csv",
    "commot": "post_xenium_human_colon_cancer_baseline_panel/commot/post_xenium_human_colon_cancer_subset__commot_predictions.csv",
    "spatialself_calibrated": "post_xenium_human_colon_cancer_spatialself_calibrated/spatialself_wavelet_edge_prior/post_xenium_human_colon_cancer_subset__spatialself_wavelet_predictions.csv",
}
NULL_MODELS = ("spatial_permutation", "receiver_permutation", "lr_identity_permutation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-dir", default=str(GOAL_DIR))
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--n-permutations", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=202607)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    goal_dir = Path(args.goal_dir)
    dataset_dir = Path(args.dataset_dir)
    output_dir = goal_dir / f"post_xenium_colon_prediction_fdr_{int(args.n_permutations)}perm"
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_path = dataset_dir / "candidate_interactions.csv"
    lr_pairs_path = dataset_dir / "lr_pairs.csv"
    lr_pairs = pd.read_csv(lr_pairs_path)

    summary_rows = []
    lr_rows = []
    status_rows = []
    for method, relative_prediction in METHODS.items():
        prediction_path = goal_dir / relative_prediction
        method_dir = output_dir / method
        if not prediction_path.exists():
            status_rows.append(_status(method, "prediction_missing", prediction_path, "prediction file not found"))
            continue
        try:
            summary = run_permutation_fdr(
                candidate_path,
                prediction_path,
                method_dir,
                null_models=NULL_MODELS,
                n_permutations=int(args.n_permutations),
                random_seed=int(args.random_seed),
                score_column="score",
                alpha=float(args.alpha),
            )
        except Exception as exc:  # pragma: no cover - audit wrapper.
            status_rows.append(_status(method, "failed", prediction_path, str(exc)))
            continue
        summary.insert(0, "method", method)
        summary_rows.append(summary)
        lr_rows.append(_collect_lr_fdr(method_dir, prediction_path.stem, method, lr_pairs))
        status_rows.append(_status(method, "completed", prediction_path, "prediction FDR panel completed"))

    summary_table = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    lr_table = pd.concat(lr_rows, ignore_index=True) if lr_rows else pd.DataFrame()
    status_table = pd.DataFrame(status_rows)
    control_table = _summarize_controls(lr_table, alpha=float(args.alpha))

    summary_table.to_csv(output_dir / "post_xenium_colon_prediction_fdr_summary.csv", index=False)
    lr_table.to_csv(output_dir / "post_xenium_colon_prediction_lr_pair_fdr.csv", index=False)
    control_table.to_csv(output_dir / "post_xenium_colon_prediction_control_fdr_summary.csv", index=False)
    status_table.to_csv(output_dir / "post_xenium_colon_prediction_fdr_status.csv", index=False)

    manifest = {
        "dataset_dir": _rel(dataset_dir),
        "candidate_path": _rel(candidate_path),
        "lr_pairs_path": _rel(lr_pairs_path),
        "output_dir": _rel(output_dir),
        "null_models": list(NULL_MODELS),
        "n_permutations": int(args.n_permutations),
        "alpha": float(args.alpha),
        "claim_boundary": (
            "Real-data prediction-level permutation/FDR panel without native CCC ground truth. It summarizes "
            "positive/negative control behavior under score-permutation nulls, not realized benchmark FDR."
        ),
        "outputs": {
            "method_summary": _rel(output_dir / "post_xenium_colon_prediction_fdr_summary.csv"),
            "lr_pair_fdr": _rel(output_dir / "post_xenium_colon_prediction_lr_pair_fdr.csv"),
            "control_summary": _rel(output_dir / "post_xenium_colon_prediction_control_fdr_summary.csv"),
            "status": _rel(output_dir / "post_xenium_colon_prediction_fdr_status.csv"),
        },
    }
    (output_dir / "post_xenium_colon_prediction_fdr_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(control_table.to_string(index=False))


def _collect_lr_fdr(method_dir: Path, prediction_stem: str, method: str, lr_pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for null_model in NULL_MODELS:
        path = method_dir / f"{prediction_stem}__{null_model}__lr_pair_fdr.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["method"] = method
        if list(frame.columns).index("method") != 0:
            method_col = frame.pop("method")
            frame.insert(0, "method", method_col)
        frame["lr_pair_fdr_path"] = _rel(path)
        rows.append(frame.merge(lr_pairs, on=["lr_id", "ligand", "receptor", "mechanism"], how="left"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _summarize_controls(lr_table: pd.DataFrame, *, alpha: float) -> pd.DataFrame:
    if lr_table.empty:
        return pd.DataFrame()
    rows = []
    for (method, null_model), group in lr_table.groupby(["method", "null_model"], sort=False):
        positives = group[group["true_lr"].fillna(False).astype(bool)]
        negatives = group[group["false_positive_high_expression"].fillna(False).astype(bool)]
        rows.append(
            {
                "method": method,
                "null_model": null_model,
                "best_positive_control": _best_control(positives),
                "best_positive_group_bh_fdr": _best_value(positives, "group_bh_fdr"),
                "best_negative_control": _best_control(negatives),
                "best_negative_group_bh_fdr": _best_value(negatives, "group_bh_fdr"),
                "n_positive_significant": int((positives["group_bh_fdr"].astype(float) <= alpha).sum()),
                "n_negative_significant": int((negatives["group_bh_fdr"].astype(float) <= alpha).sum()),
                "control_fdr_status": _control_status(positives, negatives, alpha),
            }
        )
    return pd.DataFrame(rows)


def _best_control(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    row = frame.sort_values(["group_bh_fdr", "group_p_value", "max_score"], ascending=[True, True, False]).iloc[0]
    return str(row.get("control_name", f"{row.get('ligand', '')}-{row.get('receptor', '')}"))


def _best_value(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    return float(pd.to_numeric(frame[column], errors="coerce").min())


def _control_status(positives: pd.DataFrame, negatives: pd.DataFrame, alpha: float) -> str:
    if positives.empty or negatives.empty:
        return "missing_positive_or_negative_controls"
    n_pos = int((positives["group_bh_fdr"].astype(float) <= alpha).sum())
    n_neg = int((negatives["group_bh_fdr"].astype(float) <= alpha).sum())
    if n_pos > 0 and n_neg == 0:
        return "positive_significant_negative_suppressed"
    if n_pos > 0 and n_neg > 0:
        return "positive_and_negative_controls_significant"
    if n_pos == 0 and n_neg > 0:
        return "negative_controls_significant_without_positive"
    return "no_control_significant"


def _status(method: str, status: str, prediction_path: Path, detail: str) -> dict[str, Any]:
    return {
        "method": method,
        "status": status,
        "prediction_path": _rel(prediction_path),
        "detail": detail,
    }


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
