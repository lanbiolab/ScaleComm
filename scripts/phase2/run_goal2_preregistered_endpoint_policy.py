#!/usr/bin/env python
"""Apply the preregistered endpoint policy to the tuned-small panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG = "configs/experiments/phase2_revised_goal2_preregistered_endpoint_policy.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--metrics", default="", help="Override source metrics CSV.")
    parser.add_argument("--evidence-status", default="", help="Override evidence status written to manifest/status/claims.")
    args = parser.parse_args()

    config = _load_yaml(PROJECT_ROOT / args.config)
    output_dir = PROJECT_ROOT / (args.output_dir or config.get("output_dir", "outputs/phase2_revised/goal2_preregistered_endpoint_policy"))
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = PROJECT_ROOT / (args.metrics or config["source_panel"]["metrics"])
    metrics = pd.read_csv(metrics_path)
    evidence_status = str(args.evidence_status or "preregistered_policy_application_not_final_replicated_baseline")

    endpoint_scores = _endpoint_scores(metrics, config)
    summary = _summary(endpoint_scores, metrics)
    claims = _claims(summary, config)
    policy_table = _policy_table(config)
    status = pd.DataFrame(
        [
            {
                "analysis": "goal2_preregistered_endpoint_policy",
                "status": "completed",
                "config": args.config,
                "metrics_source": str(metrics_path.relative_to(PROJECT_ROOT)),
                "claim_level": evidence_status,
            }
        ]
    )

    endpoint_scores.to_csv(output_dir / "goal2_preregistered_endpoint_scores.csv", index=False)
    summary.to_csv(output_dir / "goal2_preregistered_endpoint_summary.csv", index=False)
    claims.to_csv(output_dir / "goal2_preregistered_endpoint_claims.csv", index=False)
    policy_table.to_csv(output_dir / "goal2_preregistered_endpoint_policy_table.csv", index=False)
    status.to_csv(output_dir / "goal2_preregistered_endpoint_status.csv", index=False)
    manifest = {
        "analysis_label": "phase2_revised_goal2_preregistered_endpoint_policy",
        "canonical_goal": "goal2_weak_signal_false_positive_tradeoff",
        "config": args.config,
        "metrics_source": str(metrics_path.relative_to(PROJECT_ROOT)),
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "evidence_status": evidence_status,
        "outputs": {
            "scores": str((output_dir / "goal2_preregistered_endpoint_scores.csv").relative_to(PROJECT_ROOT)),
            "summary": str((output_dir / "goal2_preregistered_endpoint_summary.csv").relative_to(PROJECT_ROOT)),
            "claims": str((output_dir / "goal2_preregistered_endpoint_claims.csv").relative_to(PROJECT_ROOT)),
            "policy": str((output_dir / "goal2_preregistered_endpoint_policy_table.csv").relative_to(PROJECT_ROOT)),
            "status": str((output_dir / "goal2_preregistered_endpoint_status.csv").relative_to(PROJECT_ROOT)),
        },
        "limitations": [
            "Policy weights are frozen before this application; interpretation depends on the supplied source panel.",
            "Primary endpoint weights are explicit and auditable; sensitivity remains reported separately.",
            "This output does not make the manuscript final.",
        ],
    }
    (output_dir / "goal2_preregistered_endpoint_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(summary.to_string(index=False))
    print(claims.to_string(index=False))
    print(f"Wrote preregistered endpoint policy outputs to {output_dir}")


def _endpoint_scores(metrics: pd.DataFrame, config: dict) -> pd.DataFrame:
    components = config["primary_endpoint"]["components"]
    rows = []
    for scope, group in _scope_groups(metrics):
        numeric = group.dropna(subset=list(components.keys()), how="any").copy()
        if numeric.empty:
            continue
        normalized = numeric[["method"]].copy()
        for metric, spec in components.items():
            values = numeric[metric].astype(float)
            if values.max() == values.min():
                scaled = pd.Series(1.0, index=values.index)
            else:
                scaled = (values - values.min()) / (values.max() - values.min())
            if spec["direction"] == "lower_better":
                scaled = 1.0 - scaled
            normalized[f"{metric}_normalized"] = scaled
        normalized["primary_endpoint_score"] = 0.0
        for metric, spec in components.items():
            normalized["primary_endpoint_score"] += float(spec["weight"]) * normalized[f"{metric}_normalized"]
        normalized["scope"] = scope
        normalized["replicate"] = numeric["replicate"].astype(str).values if "replicate" in numeric else scope
        if "dataset_key" in numeric:
            normalized["dataset_key"] = numeric["dataset_key"].astype(str).values
        for metric in components:
            normalized[metric] = numeric[metric].astype(float).values
        rows.append(normalized)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _scope_groups(metrics: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    groups = [("aggregate_real_background_v2_formal", metrics)]
    if "dataset_key" in metrics:
        groups.extend((f"tissue:{dataset_key}", group.copy()) for dataset_key, group in metrics.groupby("dataset_key", sort=False))
    groups.extend((f"replicate:{replicate}", group.copy()) for replicate, group in metrics.groupby("replicate", sort=False))
    return groups


def _summary(scores: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, group in scores.groupby("scope", sort=False):
        best = group.sort_values("primary_endpoint_score", ascending=False).iloc[0]
        rows.append(
            {
                "scope": scope,
                "primary_endpoint_winner": best["method"],
                "primary_endpoint_score": best["primary_endpoint_score"],
                "spatialself_primary_winner": str(best["method"]).startswith("spatialself"),
                "n_scored_methods": int(group["method"].nunique()),
                "claim_level": "preregistered_endpoint_policy_application",
            }
        )
    fp = metrics.dropna(subset=["false_positive_high_expression_rate_at_threshold"]).copy()
    if not fp.empty:
        fp_summary = []
        group_key = "dataset_key" if "dataset_key" in fp else "replicate"
        for replicate, group in fp.groupby(group_key, sort=False):
            min_fp = float(group["false_positive_high_expression_rate_at_threshold"].min())
            near = group[group["false_positive_high_expression_rate_at_threshold"].astype(float) <= min_fp + 0.01]
            fp_summary.append(
                {
                group_key: str(replicate),
                    "min_fp": min_fp,
                    "methods_within_0p01": ";".join(sorted(near["method"].astype(str).unique())),
                }
            )
        fp_path = PROJECT_ROOT / "outputs/phase2_revised/goal2_preregistered_endpoint_policy/goal2_preregistered_fp_suppression_by_tissue.csv"
        fp_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(fp_summary).to_csv(fp_path, index=False)
    return pd.DataFrame(rows)


def _claims(summary: pd.DataFrame, config: dict) -> pd.DataFrame:
    aggregate = summary[summary["scope"] == "aggregate_real_background_v2_formal"]
    tissues = summary[summary["scope"].astype(str).str.startswith("tissue:")]
    aggregate_winner = str(aggregate["primary_endpoint_winner"].iloc[0]) if len(aggregate) else ""
    tissue_spatial_wins = int(tissues["spatialself_primary_winner"].astype(bool).sum()) if len(tissues) else 0
    tissue_total = int(len(tissues))
    rows = [
        {
            "claim_id": "preregistered_primary_endpoint_aggregate",
            "claim_level": "preregistered_endpoint_policy_application",
            "supported": bool(aggregate_winner.startswith("spatialself")),
            "summary": f"Aggregate primary endpoint winner under frozen policy: {aggregate_winner}",
        },
        {
            "claim_id": "preregistered_primary_endpoint_tissue_robustness",
            "claim_level": "preregistered_endpoint_policy_application",
            "supported": bool(tissue_spatial_wins >= 2),
            "summary": f"SpatialSelf wins {tissue_spatial_wins}/{tissue_total} tissue-level primary endpoints under frozen policy.",
        },
        {
            "claim_id": "preregistered_policy_no_overclaim",
            "claim_level": "preregistered_endpoint_policy_application",
            "supported": True,
            "summary": "; ".join(config["claim_rules"]["no_overclaim"]),
        },
    ]
    return pd.DataFrame(rows)


def _policy_table(config: dict) -> pd.DataFrame:
    rows = []
    for metric, spec in config["primary_endpoint"]["components"].items():
        rows.append(
            {
                "endpoint": config["primary_endpoint"]["name"],
                "metric": metric,
                "weight": spec["weight"],
                "direction": spec["direction"],
                "normalization": config["normalization"]["method"],
            }
        )
    return pd.DataFrame(rows)


def _load_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


if __name__ == "__main__":
    main()
