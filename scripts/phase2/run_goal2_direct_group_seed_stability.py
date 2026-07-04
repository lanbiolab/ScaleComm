#!/usr/bin/env python
"""Check random-seed stability for the selected direct group-level FDR setting."""

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


DEFAULT_SEEDS = (2021, 2022, 2023, 2024, 2025)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        default=(
            "outputs/phase1/exp02_semisynthetic_real_background/full_slide_sparse_datasets/"
            "human_lymph_node_visium_replicate_00/candidate_interactions.csv"
        ),
    )
    parser.add_argument(
        "--predictions",
        default=(
            "outputs/phase2/goal1_weak_signal_recovery/full_slide_sparse_initial/"
            "wavelet_mechanism_niche_edge_prior_ranking/"
            "human_lymph_node_visium_replicate_00__spatialself_wavelet_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase2/goal2_fdr_calibration/full_slide_edge_prior_direct_group_seed_stability",
    )
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--null-model", default="lr_identity_permutation")
    parser.add_argument("--group-level", default="lr_pair")
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    candidate_path = PROJECT_ROOT / args.candidates
    prediction_path = PROJECT_ROOT / args.predictions
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(args.seeds) if args.seeds else DEFAULT_SEEDS

    rows = []
    for seed in seeds:
        seed_dir = output_dir / f"seed_{int(seed)}"
        summary = run_direct_group_grid(
            candidate_path,
            prediction_path,
            seed_dir,
            null_models=(args.null_model,),
            group_levels=(args.group_level,),
            top_fractions=(float(args.top_fraction),),
            n_permutations=int(args.n_permutations),
            random_seed=int(seed),
            score_column="score",
            alpha=float(args.alpha),
        )
        if summary.empty:
            rows.append(
                {
                    "random_seed": int(seed),
                    "status": "empty_summary",
                }
            )
            continue
        row = summary.iloc[0].to_dict()
        row["random_seed"] = int(seed)
        row["seed_output_dir"] = str(seed_dir.relative_to(PROJECT_ROOT))
        rows.append(row)

    per_seed = pd.DataFrame(rows)
    per_seed_path = output_dir / "direct_group_seed_stability_per_seed.csv"
    aggregate_path = output_dir / "direct_group_seed_stability_summary.csv"
    manifest_path = output_dir / "direct_group_seed_stability_manifest.json"
    per_seed.to_csv(per_seed_path, index=False)
    aggregate = _aggregate(per_seed, args)
    aggregate.to_csv(aggregate_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "candidate_path": args.candidates,
                "prediction_path": args.predictions,
                "output_dir": args.output_dir,
                "seeds": list(seeds),
                "n_permutations": int(args.n_permutations),
                "null_model": args.null_model,
                "group_level": args.group_level,
                "top_fraction": float(args.top_fraction),
                "alpha": float(args.alpha),
                "analysis_scope": (
                    "Seed-stability check for the selected direct group-level FDR setting; "
                    "single replicate and moderate permutation count, not final manuscript calibration."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(aggregate.to_string(index=False))
    print(f"Wrote seed-stability outputs to {output_dir}")


def _aggregate(per_seed: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if "status" not in per_seed.columns:
        completed = per_seed.iloc[0:0].copy()
    else:
        completed = per_seed[per_seed["status"].astype(str) == "completed"].copy()
    if completed.empty:
        return pd.DataFrame(
            [
                {
                    "status": "no_completed_seed",
                    "null_model": args.null_model,
                    "group_level": args.group_level,
                    "top_fraction": float(args.top_fraction),
                    "n_permutations": int(args.n_permutations),
                    "n_seeds": int(len(per_seed)),
                    "n_completed_seeds": 0,
                }
            ]
        )
    pass_flags = completed["phase2_goal2_pass"].astype(bool)
    neg_sig = pd.to_numeric(completed["negative_control_significant_groups"], errors="coerce")
    pos_sig = pd.to_numeric(completed["positive_control_significant_groups"], errors="coerce")
    pos_fdr = pd.to_numeric(completed["positive_control_min_group_bh_fdr"], errors="coerce")
    neg_fdr = pd.to_numeric(completed["negative_control_min_group_bh_fdr"], errors="coerce")
    return pd.DataFrame(
        [
            {
                "status": "seed_stability_completed",
                "null_model": args.null_model,
                "group_level": args.group_level,
                "top_fraction": float(args.top_fraction),
                "n_permutations": int(args.n_permutations),
                "n_seeds": int(len(per_seed)),
                "n_completed_seeds": int(len(completed)),
                "n_passing_seeds": int(pass_flags.sum()),
                "passing_seed_fraction": float(pass_flags.mean()),
                "min_positive_control_significant_groups": int(pos_sig.min()),
                "max_positive_control_significant_groups": int(pos_sig.max()),
                "mean_positive_control_significant_groups": float(pos_sig.mean()),
                "max_negative_control_significant_groups": int(neg_sig.max()),
                "mean_positive_control_min_group_bh_fdr": float(pos_fdr.mean()),
                "max_positive_control_min_group_bh_fdr": float(pos_fdr.max()),
                "min_negative_control_min_group_bh_fdr": float(neg_fdr.min()),
                "claim_level": (
                    "seed-stable single-replicate support"
                    if bool(pass_flags.all()) and int(neg_sig.max()) == 0
                    else "mixed seed-stability support"
                ),
            }
        ]
    )


if __name__ == "__main__":
    main()
