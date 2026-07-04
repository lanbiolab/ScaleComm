#!/usr/bin/env python
"""Estimate replicate needs for the next tuned baseline panel."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DELTAS = "outputs/phase2_revised/goal2_preregistered_paired_diagnostics/goal2_preregistered_paired_deltas.csv"
DEFAULT_OUTPUT_DIR = "outputs/phase2_revised/goal2_replicate_power_plan"
Z_ALPHA_ONE_SIDED_0P05 = 1.6448536269514722
Z_POWER_0P80 = 0.8416212335729143
MIN_RECOMMENDED_N = 8
MAX_RECOMMENDED_N = 30


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deltas", default=DEFAULT_DELTAS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    deltas_path = PROJECT_ROOT / args.deltas
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    deltas = pd.read_csv(deltas_path)

    plan = _power_plan(deltas)
    overall = _overall_plan(plan)
    design = _design_table(overall)
    claims = _claims(plan, overall)
    status = pd.DataFrame(
        [
            {
                "analysis": "goal2_replicate_power_plan",
                "status": "completed",
                "deltas_source": str(deltas_path.relative_to(PROJECT_ROOT)),
                "claim_level": "replicate_planning_not_power_proof",
            }
        ]
    )

    plan.to_csv(output_dir / "goal2_replicate_power_plan_by_baseline.csv", index=False)
    overall.to_csv(output_dir / "goal2_replicate_power_plan_overall.csv", index=False)
    design.to_csv(output_dir / "goal2_replicate_power_plan_design.csv", index=False)
    claims.to_csv(output_dir / "goal2_replicate_power_plan_claims.csv", index=False)
    status.to_csv(output_dir / "goal2_replicate_power_plan_status.csv", index=False)
    manifest = {
        "analysis_label": "phase2_revised_goal2_replicate_power_plan",
        "canonical_goal": "goal2_weak_signal_false_positive_tradeoff",
        "deltas_source": str(deltas_path.relative_to(PROJECT_ROOT)),
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "evidence_status": "replicate_planning_not_power_proof",
        "assumptions": {
            "test": "one-sided paired mean-delta planning approximation",
            "alpha": 0.05,
            "power": 0.80,
            "minimum_recommended_n": MIN_RECOMMENDED_N,
            "maximum_recommended_n_cap": MAX_RECOMMENDED_N,
        },
        "outputs": {
            "by_baseline": str((output_dir / "goal2_replicate_power_plan_by_baseline.csv").relative_to(PROJECT_ROOT)),
            "overall": str((output_dir / "goal2_replicate_power_plan_overall.csv").relative_to(PROJECT_ROOT)),
            "design": str((output_dir / "goal2_replicate_power_plan_design.csv").relative_to(PROJECT_ROOT)),
            "claims": str((output_dir / "goal2_replicate_power_plan_claims.csv").relative_to(PROJECT_ROOT)),
            "status": str((output_dir / "goal2_replicate_power_plan_status.csv").relative_to(PROJECT_ROOT)),
        },
        "limitations": [
            "Planning uses only three existing formal real-background tissues and is not a power proof.",
            "Variance estimates are unstable; recommended n is capped and should be treated as a minimum planning target.",
            "Expanded replicate runs should freeze endpoint rules, SpatialSelf representative selection, and baseline parameters before execution.",
        ],
    }
    (output_dir / "goal2_replicate_power_plan_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(plan.to_string(index=False))
    print(overall.to_string(index=False))
    print(design.to_string(index=False))
    print(claims.to_string(index=False))
    print(f"Wrote replicate power plan to {output_dir}")


def _power_plan(deltas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for baseline, group in deltas.groupby("baseline_method", sort=False):
        values = group["delta_spatialself_minus_baseline"].astype(float)
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        effect_size = mean / sd if sd and sd > 0 else float("inf")
        raw_n = ((Z_ALPHA_ONE_SIDED_0P05 + Z_POWER_0P80) / effect_size) ** 2 if math.isfinite(effect_size) and effect_size > 0 else float("inf")
        estimated_n = int(math.ceil(raw_n)) if math.isfinite(raw_n) else MAX_RECOMMENDED_N
        recommended_n = max(MIN_RECOMMENDED_N, min(MAX_RECOMMENDED_N, estimated_n))
        rows.append(
            {
                "baseline_method": baseline,
                "current_n": int(len(values)),
                "mean_delta": mean,
                "sd_delta": sd,
                "paired_effect_size": effect_size,
                "estimated_n_for_80pct_power_one_sided_alpha_0p05": estimated_n,
                "recommended_min_replicates": recommended_n,
                "n_positive_delta": int((values > 0).sum()),
                "positive_delta_fraction": float((values > 0).mean()),
                "planning_limitation": "estimated from n=3 diagnostic panel; treat as minimum planning target",
            }
        )
    return pd.DataFrame(rows)


def _overall_plan(plan: pd.DataFrame) -> pd.DataFrame:
    bottleneck = plan.sort_values(["recommended_min_replicates", "estimated_n_for_80pct_power_one_sided_alpha_0p05"], ascending=False).iloc[0]
    target_n = int(plan["recommended_min_replicates"].max())
    current_n = int(plan["current_n"].max())
    additional = max(0, target_n - current_n)
    return pd.DataFrame(
        [
            {
                "target_min_replicates": target_n,
                "current_replicates": current_n,
                "additional_replicates_needed": additional,
                "bottleneck_baseline": bottleneck["baseline_method"],
                "bottleneck_recommended_n": int(bottleneck["recommended_min_replicates"]),
                "claim_level": "replicate_planning_not_power_proof",
            }
        ]
    )


def _design_table(overall: pd.DataFrame) -> pd.DataFrame:
    target = int(overall.iloc[0]["target_min_replicates"])
    additional = int(overall.iloc[0]["additional_replicates_needed"])
    return pd.DataFrame(
        [
            {
                "design_item": "replicate_unit",
                "recommendation": "independent tissue tile, seed, or donor/background replicate with the frozen endpoint policy",
                "rationale": "paired baseline comparison requires matched method outputs per replicate",
            },
            {
                "design_item": "minimum_total_replicates",
                "recommendation": str(target),
                "rationale": "worst-case baseline planning target after applying min/cap guardrails",
            },
            {
                "design_item": "additional_replicates_to_generate",
                "recommendation": str(additional),
                "rationale": "current formal real-background tuned-small panel has three tissue replicates",
            },
            {
                "design_item": "must_freeze_before_running",
                "recommendation": "endpoint policy, baseline parameters, SpatialSelf representative rule, random seeds, and failure/status handling",
                "rationale": "prevents post-hoc endpoint or threshold selection",
            },
            {
                "design_item": "reporting",
                "recommendation": "paired deltas, bootstrap CI, sign consistency, runtime, memory, failure status",
                "rationale": "answers baseline fairness and scalability concerns together",
            },
        ]
    )


def _claims(plan: pd.DataFrame, overall: pd.DataFrame) -> pd.DataFrame:
    target = int(overall.iloc[0]["target_min_replicates"])
    additional = int(overall.iloc[0]["additional_replicates_needed"])
    bottleneck = str(overall.iloc[0]["bottleneck_baseline"])
    return pd.DataFrame(
        [
            {
                "claim_id": "replicate_power_planning_target",
                "claim_level": "replicate_planning_not_power_proof",
                "supported": True,
                "summary": f"Plan next tuned baseline panel for at least {target} matched replicates, requiring {additional} additional replicates beyond the current diagnostic panel.",
            },
            {
                "claim_id": "replicate_power_bottleneck",
                "claim_level": "replicate_planning_not_power_proof",
                "supported": True,
                "summary": f"Worst-case planning bottleneck baseline is {bottleneck}.",
            },
            {
                "claim_id": "replicate_power_no_overclaim",
                "claim_level": "replicate_planning_not_power_proof",
                "supported": True,
                "summary": "This is a planning analysis, not proof of powered superiority or final manuscript readiness.",
            },
        ]
    )


if __name__ == "__main__":
    main()
