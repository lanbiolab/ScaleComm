#!/usr/bin/env python
"""Paired diagnostics for the preregistered tuned-small endpoint policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SCORES = "outputs/phase2_revised/goal2_preregistered_endpoint_policy/goal2_preregistered_endpoint_scores.csv"
DEFAULT_OUTPUT_DIR = "outputs/phase2_revised/goal2_preregistered_paired_diagnostics"
BASELINES = ("commot", "cellchat", "spatalk", "cellnest")
N_BOOTSTRAP = 10000
RANDOM_SEED = 1729


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", default=DEFAULT_SCORES)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--evidence-status", default="", help="Override evidence status written to manifest/status/claims.")
    args = parser.parse_args()

    scores_path = PROJECT_ROOT / args.scores
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_status = str(args.evidence_status or "paired_diagnostic_underpowered_not_final_replicated_baseline")

    scores = pd.read_csv(scores_path)
    aggregate = scores[scores["scope"].astype(str) == "aggregate_real_background_v2_formal"].copy()
    paired = _paired_table(aggregate)
    summary = _summary(paired, n_bootstrap=args.bootstrap)
    claims = _claims(summary)
    status = pd.DataFrame(
        [
            {
                "analysis": "goal2_preregistered_paired_diagnostics",
                "status": "completed",
                "scores_source": str(scores_path.relative_to(PROJECT_ROOT)),
                "n_bootstrap": int(args.bootstrap),
                "claim_level": evidence_status,
            }
        ]
    )
    paired.to_csv(output_dir / "goal2_preregistered_paired_deltas.csv", index=False)
    summary.to_csv(output_dir / "goal2_preregistered_paired_summary.csv", index=False)
    claims.to_csv(output_dir / "goal2_preregistered_paired_claims.csv", index=False)
    status.to_csv(output_dir / "goal2_preregistered_paired_status.csv", index=False)
    manifest = {
        "analysis_label": "phase2_revised_goal2_preregistered_paired_diagnostics",
        "canonical_goal": "goal2_weak_signal_false_positive_tradeoff",
        "scores_source": str(scores_path.relative_to(PROJECT_ROOT)),
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "evidence_status": evidence_status,
        "bootstrap": int(args.bootstrap),
        "spatialself_representative": "best SpatialSelf variant per replicate under aggregate-normalized frozen endpoint",
        "baseline_methods": BASELINES,
        "outputs": {
            "paired_deltas": str((output_dir / "goal2_preregistered_paired_deltas.csv").relative_to(PROJECT_ROOT)),
            "summary": str((output_dir / "goal2_preregistered_paired_summary.csv").relative_to(PROJECT_ROOT)),
            "claims": str((output_dir / "goal2_preregistered_paired_claims.csv").relative_to(PROJECT_ROOT)),
            "status": str((output_dir / "goal2_preregistered_paired_status.csv").relative_to(PROJECT_ROOT)),
        },
        "limitations": [
            "SpatialSelf representative is selected per replicate from existing variants; final manuscript should state this representative rule.",
            "Bootstrap CIs are diagnostic over the available matched replicates and do not replace a full statistical testing plan.",
            "This table supports endpoint interpretation, not global dominance across all metrics.",
        ],
    }
    (output_dir / "goal2_preregistered_paired_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(summary.to_string(index=False))
    print(claims.to_string(index=False))
    print(f"Wrote preregistered paired diagnostics to {output_dir}")


def _paired_table(aggregate_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for replicate, group in aggregate_scores.groupby("replicate", sort=False):
        spatial = group[group["method"].astype(str).str.startswith("spatialself")]
        if spatial.empty:
            continue
        spatial_best = spatial.sort_values("primary_endpoint_score", ascending=False).iloc[0]
        for baseline in BASELINES:
            base = group[group["method"].astype(str) == baseline]
            if base.empty:
                continue
            base_row = base.iloc[0]
            rows.append(
                {
                    "replicate": str(replicate),
                    "spatialself_method": spatial_best["method"],
                    "baseline_method": baseline,
                    "spatialself_score": float(spatial_best["primary_endpoint_score"]),
                    "baseline_score": float(base_row["primary_endpoint_score"]),
                    "delta_spatialself_minus_baseline": float(spatial_best["primary_endpoint_score"] - base_row["primary_endpoint_score"]),
                    "spatialself_aupr": float(spatial_best["aupr"]),
                    "baseline_aupr": float(base_row["aupr"]),
                    "spatialself_fp": float(spatial_best["false_positive_high_expression_rate_at_threshold"]),
                    "baseline_fp": float(base_row["false_positive_high_expression_rate_at_threshold"]),
                    "spatialself_fdr": float(spatial_best["empirical_fdr_at_threshold"]),
                    "baseline_fdr": float(base_row["empirical_fdr_at_threshold"]),
                    "spatialself_weak_recall": float(spatial_best["weak_signal_recall_at_threshold"]),
                    "baseline_weak_recall": float(base_row["weak_signal_recall_at_threshold"]),
                }
            )
    return pd.DataFrame(rows)


def _summary(paired: pd.DataFrame, *, n_bootstrap: int) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for baseline, group in paired.groupby("baseline_method", sort=False):
        deltas = group["delta_spatialself_minus_baseline"].astype(float).to_numpy()
        if len(deltas) == 0:
            continue
        boot = []
        for _ in range(n_bootstrap):
            sample = rng.choice(deltas, size=len(deltas), replace=True)
            boot.append(float(sample.mean()))
        rows.append(
            {
                "baseline_method": baseline,
                "n_tissues": int(len(deltas)),
                "mean_delta": float(deltas.mean()),
                "median_delta": float(np.median(deltas)),
                "n_positive_delta": int((deltas > 0).sum()),
                "positive_delta_fraction": float((deltas > 0).mean()),
                "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
                "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
                "underpowered": bool(len(deltas) < 18),
                "claim_level": "paired_endpoint_diagnostic",
            }
        )
    return pd.DataFrame(rows)


def _claims(summary: pd.DataFrame) -> pd.DataFrame:
    all_positive = bool((summary["mean_delta"].astype(float) > 0).all()) if len(summary) else False
    all_sign = bool((summary["positive_delta_fraction"].astype(float) >= 2 / 3).all()) if len(summary) else False
    ci_clear = bool((summary["bootstrap_ci_low"].astype(float) > 0).all()) if len(summary) else False
    return pd.DataFrame(
        [
            {
                "claim_id": "paired_endpoint_mean_delta",
                "claim_level": "paired_endpoint_diagnostic",
                "supported": all_positive,
                "summary": "SpatialSelf representative has positive mean endpoint delta against every tested baseline."
                if all_positive
                else "SpatialSelf representative does not have positive mean endpoint delta against every tested baseline.",
            },
            {
                "claim_id": "paired_endpoint_sign_consistency",
                "claim_level": "paired_endpoint_diagnostic",
                "supported": all_sign,
                "summary": "SpatialSelf representative beats every tested baseline in at least two-thirds of matched replicates."
                if all_sign
                else "SpatialSelf representative does not beat every tested baseline in at least two-thirds of matched replicates.",
            },
            {
                "claim_id": "paired_endpoint_bootstrap_ci",
                "claim_level": "paired_endpoint_diagnostic",
                "supported": ci_clear,
                "summary": "All diagnostic bootstrap CIs are above zero."
                if ci_clear
                else "Diagnostic bootstrap CIs are not all above zero; evidence remains underpowered.",
            },
        ]
    )


if __name__ == "__main__":
    main()
