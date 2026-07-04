#!/usr/bin/env python
"""Merge revised Goal 1 full-baseline matched validation outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_INPUTS = (
    "outputs/phase2_revised/goal1_full_baseline_matched_validation",
    "outputs/phase2_revised/goal1_full_baseline_matched_validation_lymph_melanoma",
    "outputs/phase2_revised/goal1_full_baseline_matched_validation_melanoma",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", dest="input_dirs")
    parser.add_argument("--output-dir", default="outputs/phase2_revised/goal1_full_baseline_matched_validation_all")
    args = parser.parse_args()

    input_dirs = [PROJECT_ROOT / path for path in (args.input_dirs or DEFAULT_INPUTS)]
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    method = _read_many(input_dirs, "goal1_baseline_matched_method_summary.csv")
    lr = _read_many(input_dirs, "goal1_baseline_matched_lr_control_summary.csv")
    status = _read_many(input_dirs, "goal1_baseline_matched_status.csv")
    if method.empty:
        raise ValueError(f"No method summary rows found in {input_dirs}")

    method.to_csv(output_dir / "goal1_baseline_matched_method_summary.csv", index=False)
    lr.to_csv(output_dir / "goal1_baseline_matched_lr_control_summary.csv", index=False)
    status.to_csv(output_dir / "goal1_baseline_matched_status.csv", index=False)
    _claims(method, status).to_csv(output_dir / "goal1_baseline_matched_claims.csv", index=False)
    manifest = {
        "analysis_label": "phase2_revised_goal1_full_baseline_matched_validation_all",
        "evidence_status": "three_real_background_injected_control_proxy_not_final_curated_downstream",
        "input_dirs": [str(path.relative_to(PROJECT_ROOT)) for path in input_dirs],
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "datasets": sorted(method["dataset"].astype(str).unique()),
        "methods": sorted(method["method"].astype(str).unique()),
        "outputs": {
            "method_summary": str((output_dir / "goal1_baseline_matched_method_summary.csv").relative_to(PROJECT_ROOT)),
            "lr_control_summary": str((output_dir / "goal1_baseline_matched_lr_control_summary.csv").relative_to(PROJECT_ROOT)),
            "claims": str((output_dir / "goal1_baseline_matched_claims.csv").relative_to(PROJECT_ROOT)),
            "status": str((output_dir / "goal1_baseline_matched_status.csv").relative_to(PROJECT_ROOT)),
        },
        "limitations": [
            "Receiver-response Spearman is a receptor-expression proxy, not curated LR-target validation.",
            "CellChat, SpaTalk, and CellNEST predictions are adapter-light smoke outputs.",
            "SpaTalk can pass rank validation while still calling all high-expression false positives.",
        ],
    }
    (output_dir / "goal1_baseline_matched_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    cols = [
        "dataset",
        "method",
        "mean_positive_rank",
        "mean_negative_rank",
        "negative_top3_count",
        "empirical_fdr_at_q95",
        "high_expression_false_positive_rate_at_q95",
        "rank_validation_pass",
    ]
    print(method[cols].to_string(index=False))
    print(f"Wrote merged full-baseline matched validation to {output_dir}")


def _read_many(input_dirs: list[Path], filename: str) -> pd.DataFrame:
    parts = []
    for directory in input_dirs:
        path = directory / filename
        if path.exists():
            part = pd.read_csv(path)
            part["source_validation_dir"] = str(directory.relative_to(PROJECT_ROOT))
            parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _claims(method: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in method.groupby("dataset", sort=False):
        spatialself = group[group["method"].astype(str).str.startswith("spatialself")]
        non_spatial = group[~group["method"].astype(str).str.startswith("spatialself")]
        best_rank = group.sort_values(["rank_validation_pass", "mean_positive_rank"], ascending=[False, True]).iloc[0]
        best_fp = group.sort_values("high_expression_false_positive_rate_at_q95", ascending=True).iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "claim_level": "three_background_injected_control_proxy_not_final_curated_downstream",
                "best_rank_method": best_rank["method"],
                "best_rank_validation_pass": bool(best_rank["rank_validation_pass"]),
                "lowest_high_expression_fp_method": best_fp["method"],
                "lowest_high_expression_fp_rate": best_fp["high_expression_false_positive_rate_at_q95"],
                "spatialself_negative_top3_max": int(spatialself["negative_top3_count"].max()) if not spatialself.empty else -1,
                "non_spatial_negative_top3_min": int(non_spatial["negative_top3_count"].min()) if not non_spatial.empty else -1,
                "spatialself_min_high_expression_fp": float(spatialself["high_expression_false_positive_rate_at_q95"].min()) if not spatialself.empty else float("nan"),
                "non_spatial_min_high_expression_fp": float(non_spatial["high_expression_false_positive_rate_at_q95"].min()) if not non_spatial.empty else float("nan"),
                "spatialself_lowest_fp": str(best_fp["method"]).startswith("spatialself"),
                "completed_methods": int(((status["dataset"] == dataset) & (status["status"].astype(str) == "completed")).sum()) if not status.empty else 0,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
