#!/usr/bin/env python
"""Run revised Phase 2 formal FDR smoke with fixed CCC null models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spitialself.fdr import FormalFDRPanelConfig, run_formal_fdr_panel


REQUIRED_NULL_MODELS = ("spatial_permutation", "receiver_permutation", "lr_identity_permutation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/phase2_revised_goal5_formal_fdr.yaml")
    parser.add_argument("--candidate-path", default="")
    parser.add_argument("--prediction-path", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--n-permutations", type=int)
    parser.add_argument("--alpha", type=float)
    args = parser.parse_args()

    config = _load_yaml(PROJECT_ROOT / args.config)
    panel = config["default_panel"]
    fdr_cfg = config["fdr"]
    candidate_path = PROJECT_ROOT / (args.candidate_path or panel["candidate_path"])
    prediction_path = PROJECT_ROOT / (args.prediction_path or panel["prediction_path"])
    output_dir = PROJECT_ROOT / (args.output_dir or config.get("output_dir", "outputs/phase2_revised/goal5_formal_fdr"))
    n_permutations = int(args.n_permutations or fdr_cfg.get("n_permutations", 20))
    alpha = float(args.alpha or fdr_cfg.get("alpha", 0.05))
    null_models = tuple(config.get("null_models") or REQUIRED_NULL_MODELS)
    if set(null_models) != set(REQUIRED_NULL_MODELS):
        raise ValueError(f"Goal 5 requires exactly these null models: {REQUIRED_NULL_MODELS}")

    summary, group_summary = run_formal_fdr(
        candidate_path,
        prediction_path,
        output_dir,
        null_models=null_models,
        n_permutations=n_permutations,
        random_seed=int(fdr_cfg.get("random_seed", 2026)),
        score_column=str(fdr_cfg.get("score_column", "score")),
        alpha=alpha,
        direct_group_top_fraction=float(fdr_cfg.get("direct_group_top_fraction", 0.1)),
        panel_metadata=panel,
        claim_level=str(config.get("claim_level", "formal_fdr_smoke_not_manuscript_ready")),
    )
    print(summary.to_string(index=False))
    print(f"Wrote revised Goal 5 FDR outputs to {output_dir}")


def run_formal_fdr(
    candidate_path: Path,
    prediction_path: Path,
    output_dir: Path,
    *,
    null_models: tuple[str, ...],
    n_permutations: int,
    random_seed: int,
    score_column: str,
    alpha: float,
    direct_group_top_fraction: float,
    panel_metadata: dict,
    claim_level: str = "formal_fdr_smoke_not_manuscript_ready",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_csv(candidate_path)
    predictions = pd.read_csv(prediction_path)
    config = FormalFDRPanelConfig(
        null_models=tuple(null_models),
        n_permutations=n_permutations,
        random_seed=random_seed,
        score_column=score_column,
        alpha=float(alpha),
        direct_group_top_fraction=float(direct_group_top_fraction),
        panel_metadata=dict(panel_metadata),
        claim_level=claim_level,
    )
    return run_formal_fdr_panel(
        candidates,
        predictions,
        output_dir=output_dir,
        candidate_path=candidate_path,
        prediction_path=prediction_path,
        config=config,
    )


def _load_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


if __name__ == "__main__":
    main()
