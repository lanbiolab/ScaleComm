#!/usr/bin/env python
"""Run a multi-dataset formal real-background FDR panel."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--config", default="configs/experiments/phase2_revised_goal5_real_background_fdr_panel.yaml")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--n-permutations", type=int)
    parser.add_argument("--panel", action="append", dest="panels", help="Dataset key to run. Repeatable.")
    args = parser.parse_args()

    config = _load_yaml(PROJECT_ROOT / args.config)
    output_dir = PROJECT_ROOT / (args.output_dir or config.get("output_dir", "outputs/phase2_revised/goal5_real_background_fdr_panel"))
    output_dir.mkdir(parents=True, exist_ok=True)
    fdr_cfg = config["fdr"]
    null_models = tuple(config.get("null_models") or REQUIRED_NULL_MODELS)
    if set(null_models) != set(REQUIRED_NULL_MODELS):
        raise ValueError(f"Goal 5 requires exactly these null models: {REQUIRED_NULL_MODELS}")

    selected = set(args.panels or [])
    panels = [panel for panel in config["panels"] if not selected or panel["dataset"] in selected]
    n_permutations = int(args.n_permutations or fdr_cfg.get("n_permutations", 50))
    summary_parts: list[pd.DataFrame] = []
    group_parts: list[pd.DataFrame] = []
    status_rows: list[dict] = []
    panel_manifests: list[dict] = []

    for panel in panels:
        panel_output = output_dir / str(panel["dataset"])
        candidate_path = PROJECT_ROOT / panel["candidate_path"]
        prediction_path = PROJECT_ROOT / panel["prediction_path"]
        try:
            candidates = pd.read_csv(candidate_path)
            predictions = pd.read_csv(prediction_path)
            fdr_config = FormalFDRPanelConfig(
                null_models=null_models,
                n_permutations=n_permutations,
                random_seed=int(fdr_cfg.get("random_seed", 2026)),
                score_column=str(fdr_cfg.get("score_column", "score")),
                alpha=float(fdr_cfg.get("alpha", 0.05)),
                direct_group_top_fraction=float(fdr_cfg.get("direct_group_top_fraction", 0.1)),
                panel_metadata=dict(panel),
                claim_level=str(config.get("claim_level", "formal_real_background_fdr_panel_not_manuscript_ready")),
            )
            summary, group_summary = run_formal_fdr_panel(
                candidates,
                predictions,
                output_dir=panel_output,
                candidate_path=candidate_path,
                prediction_path=prediction_path,
                config=fdr_config,
            )
            summary_parts.append(summary)
            group_parts.append(group_summary)
            panel_manifest_path = panel_output / "goal5_formal_fdr_manifest.json"
            panel_manifests.append(json.loads(panel_manifest_path.read_text(encoding="utf-8")))
            status_rows.append(
                {
                    "dataset": panel["dataset"],
                    "replicate": panel["replicate"],
                    "method": panel["method"],
                    "status": "completed",
                    "detail": "",
                    "candidate_path": str(candidate_path.relative_to(PROJECT_ROOT)),
                    "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)),
                    "output_dir": str(panel_output.relative_to(PROJECT_ROOT)),
                }
            )
        except Exception as exc:  # noqa: BLE001
            status_rows.append(
                {
                    "dataset": panel.get("dataset", ""),
                    "replicate": panel.get("replicate", ""),
                    "method": panel.get("method", ""),
                    "status": "failed",
                    "detail": f"{type(exc).__name__}: {exc}",
                    "candidate_path": str(candidate_path.relative_to(PROJECT_ROOT)) if "candidate_path" in locals() else "",
                    "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)) if "prediction_path" in locals() else "",
                    "output_dir": str(panel_output.relative_to(PROJECT_ROOT)) if "panel_output" in locals() else "",
                }
            )

    summary_table = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    group_summary_table = pd.concat(group_parts, ignore_index=True) if group_parts else pd.DataFrame()
    status_table = pd.DataFrame(status_rows)
    claims = _claims(summary_table, group_summary_table, status_table, config, n_permutations)

    summary_table.to_csv(output_dir / "goal5_real_background_fdr_panel_summary.csv", index=False)
    group_summary_table.to_csv(output_dir / "goal5_real_background_fdr_panel_group_summary.csv", index=False)
    status_table.to_csv(output_dir / "goal5_real_background_fdr_panel_status.csv", index=False)
    claims.to_csv(output_dir / "goal5_real_background_fdr_panel_claims.csv", index=False)

    manifest = {
        "analysis_label": "phase2_revised_goal5_real_background_fdr_panel",
        "canonical_goal": "goal5_fdr_calibration_formal_module",
        "config": args.config,
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "datasets": [panel["dataset"] for panel in panels],
        "null_models": list(null_models),
        "n_permutations": n_permutations,
        "alpha": float(fdr_cfg.get("alpha", 0.05)),
        "direct_group_top_fraction": float(fdr_cfg.get("direct_group_top_fraction", 0.1)),
        "claim_level": str(config.get("claim_level", "formal_real_background_fdr_panel_not_manuscript_ready")),
        "framework_module": "src.spitialself.fdr.run_formal_fdr_panel",
        "panel_manifests": panel_manifests,
        "outputs": {
            "summary": str((output_dir / "goal5_real_background_fdr_panel_summary.csv").relative_to(PROJECT_ROOT)),
            "group_summary": str((output_dir / "goal5_real_background_fdr_panel_group_summary.csv").relative_to(PROJECT_ROOT)),
            "status": str((output_dir / "goal5_real_background_fdr_panel_status.csv").relative_to(PROJECT_ROOT)),
            "claims": str((output_dir / "goal5_real_background_fdr_panel_claims.csv").relative_to(PROJECT_ROOT)),
        },
        "limitations": [
            "This panel uses 50 permutations and is not a final 200-500 permutation manuscript calibration panel.",
            "The panel currently calibrates SpatialSelf wavelet predictions on formal real-background benchmarks.",
            "Candidate-edge discoveries can remain absent even when group-level LR or cell-type discoveries are present.",
        ],
    }
    (output_dir / "goal5_real_background_fdr_panel_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not claims.empty:
        print(claims.to_string(index=False))
    print(f"Wrote formal real-background FDR panel to {output_dir}")


def _claims(
    summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    status: pd.DataFrame,
    config: dict,
    n_permutations: int,
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for dataset, group in summary.groupby("dataset", sort=False):
        group_rows = group_summary[group_summary["dataset"].astype(str) == str(dataset)] if not group_summary.empty else pd.DataFrame()
        status_rows = status[status["dataset"].astype(str) == str(dataset)] if not status.empty else pd.DataFrame()
        rows.append(
            {
                "dataset": dataset,
                "claim_level": str(config.get("claim_level", "formal_real_background_fdr_panel_not_manuscript_ready")),
                "n_null_models": int(group["null_model"].nunique()),
                "n_permutations": int(n_permutations),
                "min_p_value": float(group["min_p_value"].min()),
                "min_bh_fdr": float(group["min_bh_fdr"].min()),
                "candidate_edge_discoveries": int(group["n_significant_edges"].sum()),
                "lr_pair_direct_discoveries": int(group["n_significant_lr_pairs_direct"].sum()),
                "cell_type_pair_direct_discoveries": int(group["n_significant_cell_type_pairs_direct"].sum()),
                "mean_realized_fdr": float(group["realized_fdr"].dropna().mean()) if "realized_fdr" in group else float("nan"),
                "mean_realized_weak_signal_recall": float(group["realized_weak_signal_recall"].dropna().mean()) if "realized_weak_signal_recall" in group else float("nan"),
                "mean_high_expression_fp_call_rate": float(group["realized_false_positive_high_expression_call_rate"].dropna().mean()) if "realized_false_positive_high_expression_call_rate" in group else float("nan"),
                "group_summary_rows": int(len(group_rows)),
                "group_discoveries": int(group_rows["n_significant_groups"].sum()) if "n_significant_groups" in group_rows else 0,
                "status": str(status_rows["status"].iloc[0]) if len(status_rows) else "missing_status",
                "fdr_schema_complete": int(group["null_model"].nunique()) == 3
                and set(group["null_model"].astype(str)) == set(REQUIRED_NULL_MODELS)
                and len(group_rows) >= 6,
            }
        )
    return pd.DataFrame(rows)


def _load_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


if __name__ == "__main__":
    main()
