#!/usr/bin/env python
"""Run a deeper formal FDR panel for the expanded tonsil Goal 4 candidate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spitialself.fdr import FormalFDRPanelConfig, run_formal_fdr_panel


DEFAULT_CANDIDATES = (
    "outputs/phase2/goal4_high_resolution/tonsil_fresh_wavelet_calibrated_subset/"
    "constructed_replicate/candidate_interactions.csv"
)
DEFAULT_PREDICTIONS = (
    "outputs/phase2/goal4_high_resolution/tonsil_baseline_matched_fresh_subset/"
    "predictions/spatialself_wavelet_calibrated/"
    "constructed_replicate__spatialself_wavelet_calibrated_predictions.csv"
)
DEFAULT_OUTPUT = (
    "outputs/phase7/functional_cross_platform_validation/"
    "goal4_expanded_tonsil_subset_screen/fdr_formal_100perm"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=20260704)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    candidate_path = PROJECT_ROOT / args.candidates
    prediction_path = PROJECT_ROOT / args.predictions
    output_dir = PROJECT_ROOT / args.output_dir

    candidates = pd.read_csv(candidate_path)
    predictions = pd.read_csv(prediction_path)
    config = FormalFDRPanelConfig(
        null_models=("spatial_permutation", "receiver_permutation", "lr_identity_permutation"),
        n_permutations=int(args.n_permutations),
        random_seed=int(args.random_seed),
        score_column="score",
        alpha=float(args.alpha),
        direct_group_top_fraction=0.1,
        panel_metadata={
            "dataset": "tonsil_highres_45lr_subset",
            "replicate": "constructed_replicate",
        },
        claim_level=f"phase7_goal4_formal_{int(args.n_permutations)}perm_screen",
    )
    summary, group_summary = run_formal_fdr_panel(
        candidates,
        predictions,
        output_dir=output_dir,
        candidate_path=candidate_path,
        prediction_path=prediction_path,
        config=config,
    )
    print(summary.to_string(index=False))
    print(group_summary.to_string(index=False))


if __name__ == "__main__":
    main()
