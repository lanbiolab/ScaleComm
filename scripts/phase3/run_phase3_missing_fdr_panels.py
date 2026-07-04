#!/usr/bin/env python
"""Run missing Phase 3 smoke FDR panels for manifest datasets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spitialself.fdr import FormalFDRPanelConfig, run_formal_fdr_panel

NULL_MODELS = ("spatial_permutation", "receiver_permutation", "lr_identity_permutation")
TARGET_DATASETS = ("controlled_simulator_v2", "high_resolution_tonsil_subset")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="outputs/phase3/end_to_end/phase3_dataset_manifest.csv")
    parser.add_argument("--prediction-index", default="outputs/phase3/end_to_end/phase3_prediction_index.csv")
    parser.add_argument("--output-dir", default="outputs/phase3/end_to_end/fdr_smoke_missing")
    parser.add_argument("--n-permutations", type=int, default=20)
    args = parser.parse_args()

    manifest = {row["dataset_id"]: row for row in _read_csv(PROJECT_ROOT / args.manifest)}
    predictions = _read_csv(PROJECT_ROOT / args.prediction_index)
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_parts: list[pd.DataFrame] = []
    group_parts: list[pd.DataFrame] = []
    status_rows: list[dict[str, object]] = []
    panel_manifests: list[dict] = []

    for dataset_id in TARGET_DATASETS:
        dataset = manifest[dataset_id]
        pred = _find_spatialself_prediction(dataset_id, predictions)
        panel_output = output_dir / dataset_id
        candidate_path = PROJECT_ROOT / dataset["path"] / "candidate_interactions.csv"
        prediction_path = PROJECT_ROOT / pred["prediction_path"]
        try:
            candidates = pd.read_csv(candidate_path)
            prediction_table = pd.read_csv(prediction_path)
            cfg = FormalFDRPanelConfig(
                null_models=NULL_MODELS,
                n_permutations=int(args.n_permutations),
                random_seed=2026,
                score_column="score",
                alpha=0.05,
                direct_group_top_fraction=0.1,
                panel_metadata={
                    "dataset": dataset_id,
                    "replicate": pred.get("dataset_key", ""),
                    "method": "spatialself_wavelet_edge_prior",
                },
                claim_level="phase3_smoke_fdr_panel_not_final_100_or_500perm",
            )
            summary, group_summary = run_formal_fdr_panel(
                candidates,
                prediction_table,
                output_dir=panel_output,
                candidate_path=candidate_path,
                prediction_path=prediction_path,
                config=cfg,
            )
            summary_parts.append(summary)
            group_parts.append(group_summary)
            panel_manifests.append(json.loads((panel_output / "goal5_formal_fdr_manifest.json").read_text(encoding="utf-8")))
            status_rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_key": pred.get("dataset_key", ""),
                    "method": "spatialself_wavelet_edge_prior",
                    "status": "completed",
                    "detail": "",
                    "candidate_path": str(candidate_path.relative_to(PROJECT_ROOT)),
                    "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)),
                    "output_dir": str(panel_output.relative_to(PROJECT_ROOT)),
                    "n_permutations": int(args.n_permutations),
                }
            )
        except Exception as exc:  # noqa: BLE001
            status_rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_key": pred.get("dataset_key", "") if pred else "",
                    "method": "spatialself_wavelet_edge_prior",
                    "status": "failed",
                    "detail": f"{type(exc).__name__}: {exc}",
                    "candidate_path": str(candidate_path.relative_to(PROJECT_ROOT)) if "candidate_path" in locals() else "",
                    "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)) if "prediction_path" in locals() else "",
                    "output_dir": str(panel_output.relative_to(PROJECT_ROOT)) if "panel_output" in locals() else "",
                    "n_permutations": int(args.n_permutations),
                }
            )

    summary_table = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    group_table = pd.concat(group_parts, ignore_index=True) if group_parts else pd.DataFrame()
    status_table = pd.DataFrame(status_rows)
    summary_table.to_csv(output_dir / "phase3_missing_fdr_summary.csv", index=False)
    group_table.to_csv(output_dir / "phase3_missing_fdr_group_summary.csv", index=False)
    status_table.to_csv(output_dir / "phase3_missing_fdr_status.csv", index=False)
    manifest_doc = {
        "analysis_label": "phase3_missing_smoke_fdr_panels",
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "datasets": list(TARGET_DATASETS),
        "null_models": list(NULL_MODELS),
        "n_permutations": int(args.n_permutations),
        "claim_level": "phase3_smoke_fdr_panel_not_final_100_or_500perm",
        "panel_manifests": panel_manifests,
        "outputs": {
            "summary": str((output_dir / "phase3_missing_fdr_summary.csv").relative_to(PROJECT_ROOT)),
            "group_summary": str((output_dir / "phase3_missing_fdr_group_summary.csv").relative_to(PROJECT_ROOT)),
            "status": str((output_dir / "phase3_missing_fdr_status.csv").relative_to(PROJECT_ROOT)),
        },
    }
    (output_dir / "phase3_missing_fdr_manifest.json").write_text(json.dumps(manifest_doc, indent=2), encoding="utf-8")
    print(status_table.to_string(index=False))
    print(f"Wrote Phase 3 missing FDR panels to {output_dir}")


def _find_spatialself_prediction(dataset_id: str, rows: list[dict[str, str]]) -> dict[str, str]:
    for row in rows:
        if (
            row.get("dataset_id") == dataset_id
            and row.get("method") == "spatialself_wavelet_edge_prior"
            and row.get("status") == "completed"
            and row.get("prediction_path")
        ):
            return row
    raise ValueError(f"No completed SpatialSelf prediction indexed for {dataset_id}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
