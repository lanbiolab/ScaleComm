#!/usr/bin/env python
"""Check replicate stability for the selected direct group-level FDR setting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2.run_goal2_direct_group_grid import run_direct_group_grid


DEFAULT_REPLICATES = tuple(range(5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="outputs/phase1/exp01_controlled_benchmark/datasets",
    )
    parser.add_argument(
        "--prediction-root",
        default=(
            "outputs/phase2/goal1_weak_signal_recovery/controlled_five_replicate_initial/"
            "wavelet_mechanism_niche_edge_prior_ranking"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase2/goal2_fdr_calibration/controlled_edge_prior_direct_group_replicate_stability",
    )
    parser.add_argument("--replicate", action="append", type=int, dest="replicates")
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=3021)
    parser.add_argument("--null-model", default="lr_identity_permutation")
    parser.add_argument("--group-level", default="lr_pair")
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    dataset_root = PROJECT_ROOT / args.dataset_root
    prediction_root = PROJECT_ROOT / args.prediction_root
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    replicates = tuple(args.replicates) if args.replicates else DEFAULT_REPLICATES

    rows: list[dict[str, object]] = []
    for replicate in replicates:
        replicate_label = f"replicate_{int(replicate):02d}"
        candidate_path = dataset_root / replicate_label / "candidate_interactions.csv"
        prediction_path = prediction_root / f"{replicate_label}__spatialself_wavelet_predictions.csv"
        replicate_dir = output_dir / replicate_label
        if not candidate_path.exists() or not prediction_path.exists():
            rows.append(
                {
                    "replicate": replicate_label,
                    "status": "missing_input",
                    "candidate_path": str(candidate_path.relative_to(PROJECT_ROOT)),
                    "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)),
                }
            )
            continue
        summary = run_direct_group_grid(
            candidate_path,
            prediction_path,
            replicate_dir,
            null_models=(args.null_model,),
            group_levels=(args.group_level,),
            top_fractions=(float(args.top_fraction),),
            n_permutations=int(args.n_permutations),
            random_seed=int(args.random_seed) + int(replicate),
            score_column="score",
            alpha=float(args.alpha),
        )
        if summary.empty:
            rows.append(
                {
                    "replicate": replicate_label,
                    "status": "empty_summary",
                    "candidate_path": str(candidate_path.relative_to(PROJECT_ROOT)),
                    "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)),
                }
            )
            continue
        row = summary.iloc[0].to_dict()
        row["replicate"] = replicate_label
        row["candidate_path"] = str(candidate_path.relative_to(PROJECT_ROOT))
        row["prediction_path"] = str(prediction_path.relative_to(PROJECT_ROOT))
        row["replicate_output_dir"] = str(replicate_dir.relative_to(PROJECT_ROOT))
        rows.append(row)

    per_replicate = pd.DataFrame(rows)
    per_replicate_path = output_dir / "direct_group_replicate_stability_per_replicate.csv"
    summary_path = output_dir / "direct_group_replicate_stability_summary.csv"
    manifest_path = output_dir / "direct_group_replicate_stability_manifest.json"
    per_replicate.to_csv(per_replicate_path, index=False)
    aggregate = _aggregate(per_replicate, args)
    aggregate.to_csv(summary_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_root": args.dataset_root,
                "prediction_root": args.prediction_root,
                "output_dir": args.output_dir,
                "replicates": [int(value) for value in replicates],
                "n_permutations": int(args.n_permutations),
                "random_seed": int(args.random_seed),
                "null_model": args.null_model,
                "group_level": args.group_level,
                "top_fraction": float(args.top_fraction),
                "alpha": float(args.alpha),
                "analysis_scope": (
                    "Controlled-benchmark replicate-stability check for the selected direct group-level FDR "
                    "setting. This supports cross-replicate calibration on synthetic ground-truth data, but "
                    "does not replace full-slide realistic-background or real-data FDR validation."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(aggregate.to_string(index=False))
    print(f"Wrote replicate-stability outputs to {output_dir}")


def _aggregate(per_replicate: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    completed = (
        per_replicate[per_replicate["status"].astype(str) == "completed"].copy()
        if "status" in per_replicate.columns
        else per_replicate.iloc[0:0].copy()
    )
    if completed.empty:
        return pd.DataFrame(
            [
                {
                    "status": "no_completed_replicate",
                    "null_model": args.null_model,
                    "group_level": args.group_level,
                    "top_fraction": float(args.top_fraction),
                    "n_permutations": int(args.n_permutations),
                    "n_replicates": int(len(per_replicate)),
                    "n_completed_replicates": 0,
                }
            ]
        )

    pass_flags = completed["phase2_goal2_pass"].astype(bool)
    pos_sig = pd.to_numeric(completed["positive_control_significant_groups"], errors="coerce")
    neg_sig = pd.to_numeric(completed["negative_control_significant_groups"], errors="coerce")
    pos_fdr = pd.to_numeric(completed["positive_control_min_group_bh_fdr"], errors="coerce")
    neg_fdr = pd.to_numeric(completed["negative_control_min_group_bh_fdr"], errors="coerce")
    total_sig = pd.to_numeric(completed["n_significant_groups"], errors="coerce")
    return pd.DataFrame(
        [
            {
                "status": "replicate_stability_completed",
                "null_model": args.null_model,
                "group_level": args.group_level,
                "top_fraction": float(args.top_fraction),
                "n_permutations": int(args.n_permutations),
                "n_replicates": int(len(per_replicate)),
                "n_completed_replicates": int(len(completed)),
                "n_passing_replicates": int(pass_flags.sum()),
                "passing_replicate_fraction": float(pass_flags.mean()),
                "min_positive_control_significant_groups": int(pos_sig.min()),
                "max_positive_control_significant_groups": int(pos_sig.max()),
                "mean_positive_control_significant_groups": float(pos_sig.mean()),
                "max_negative_control_significant_groups": int(neg_sig.max()),
                "mean_negative_control_significant_groups": float(neg_sig.mean()),
                "mean_total_significant_groups": float(total_sig.mean()),
                "mean_positive_control_min_group_bh_fdr": float(pos_fdr.mean()),
                "max_positive_control_min_group_bh_fdr": float(pos_fdr.max()),
                "min_negative_control_min_group_bh_fdr": float(neg_fdr.min()),
                "claim_level": (
                    "controlled cross-replicate support"
                    if bool(pass_flags.all()) and int(neg_sig.max()) == 0
                    else "mixed controlled replicate support"
                ),
            }
        ]
    )


if __name__ == "__main__":
    main()
