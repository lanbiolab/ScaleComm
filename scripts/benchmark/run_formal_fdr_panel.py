#!/usr/bin/env python
"""Run the formal SpatialSelf CCC FDR panel with the fixed three null models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spitialself.fdr import FormalFDRPanelConfig, run_formal_fdr_panel

FORMAL_NULL_MODELS = (
    "spatial_permutation",
    "receiver_permutation",
    "lr_identity_permutation",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="candidate_interactions.csv path.")
    parser.add_argument("--predictions", required=True, help="Prediction CSV with candidate_id and score.")
    parser.add_argument("--output-dir", required=True, help="Directory for standard formal FDR outputs.")
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=2025)
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--replicate", default="")
    parser.add_argument("--method", default="")
    parser.add_argument("--direct-group-top-fraction", type=float, default=0.1)
    parser.add_argument("--claim-level", default="formal_fdr_panel_fixed_three_nulls")
    args = parser.parse_args()

    candidate_path = PROJECT_ROOT / args.candidates
    prediction_path = PROJECT_ROOT / args.predictions
    output_dir = PROJECT_ROOT / args.output_dir
    candidates = pd.read_csv(candidate_path)
    predictions = pd.read_csv(prediction_path)
    metadata = {
        "dataset": args.dataset,
        "replicate": args.replicate or candidate_path.parent.name,
        "method": args.method,
    }
    config = FormalFDRPanelConfig(
        null_models=FORMAL_NULL_MODELS,
        n_permutations=int(args.n_permutations),
        random_seed=int(args.random_seed),
        score_column=str(args.score_column),
        alpha=float(args.alpha),
        direct_group_top_fraction=float(args.direct_group_top_fraction),
        panel_metadata=metadata,
        claim_level=str(args.claim_level),
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
    print(f"Wrote formal fixed-null FDR panel to {output_dir}")


if __name__ == "__main__":
    main()

