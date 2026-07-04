#!/usr/bin/env python
"""Run formal injected receiver-response proxy validation.

This is deliberately labeled as an injected-target proxy. It does not replace
curated NicheNet/Reactome downstream validation, but it is stricter than plain
receptor-expression correlation because the receiver response includes local
truth exposure from the injected CCC benchmark.
"""

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


DATASETS = {
    "real_background_v2_formal_lymph_node": {
        "replicate_dir": "outputs/phase2_revised/goal4_real_background_v2/formal_datasets/human_lymph_node_visium_replicate_00",
        "prediction_dataset": "real_background_v2_formal",
    },
    "real_background_v2_formal_tonsil": {
        "replicate_dir": "outputs/phase2_revised/goal4_real_background_v2/formal_datasets/human_tonsil_slide_tags_replicate_00",
        "prediction_dataset": "real_background_v2_formal",
    },
    "real_background_v2_formal_melanoma": {
        "replicate_dir": "outputs/phase2_revised/goal4_real_background_v2/formal_datasets/human_melanoma_multiome_replicate_00",
        "prediction_dataset": "real_background_v2_formal",
    },
}

PREDICTION_ROOTS = {
    "spatialself_wavelet_edge_prior": "outputs/phase2_revised/goal2_formal_baseline_panel",
    "commot": "outputs/phase2_revised/goal2_formal_baseline_panel",
    "cellchat": "outputs/phase2_revised/goal2_formal_baseline_panel",
    "spatialself_wavelet": "outputs/phase2_revised/goal2_formal_baseline_panel_extended",
    "spatalk": "outputs/phase2_revised/goal2_formal_baseline_panel_extended",
    "cellnest": "outputs/phase2_revised/goal2_formal_baseline_panel_extended",
}

METHODS = (
    "spatialself_wavelet_edge_prior",
    "spatialself_wavelet",
    "commot",
    "cellchat",
    "spatalk",
    "cellnest",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/phase2_revised/goal1_injected_downstream_proxy")
    parser.add_argument("--exposure-weight", type=float, default=1.0)
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    lr_rows: list[dict] = []
    status_rows: list[dict] = []
    for dataset, dataset_cfg in DATASETS.items():
        replicate_dir = PROJECT_ROOT / dataset_cfg["replicate_dir"]
        try:
            candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
            expression = pd.read_csv(replicate_dir / "expression.csv")
            lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
        except Exception as exc:  # noqa: BLE001
            status_rows.append({"dataset": dataset, "method": "dataset", "status": "failed", "detail": str(exc)})
            continue

        for method in METHODS:
            prediction_path = _prediction_path(
                PROJECT_ROOT / PREDICTION_ROOTS[method],
                dataset_cfg["prediction_dataset"],
                method,
                replicate_dir.name,
            )
            if not prediction_path.exists():
                status_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "status": "missing_prediction",
                        "detail": str(prediction_path.relative_to(PROJECT_ROOT)),
                    }
                )
                continue
            try:
                predictions = pd.read_csv(prediction_path)
                annotated = _merge(candidates, predictions)
                lr_rows.extend(
                    _lr_proxy_rows(
                        annotated,
                        expression,
                        lr_pairs,
                        dataset=dataset,
                        method=method,
                        prediction_path=prediction_path,
                        exposure_weight=float(args.exposure_weight),
                    )
                )
                status_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "status": "completed",
                        "detail": "",
                        "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                status_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "status": "failed",
                        "detail": f"{type(exc).__name__}: {exc}",
                        "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)),
                    }
                )

    lr_table = pd.DataFrame(lr_rows)
    method_summary = _method_summary(lr_table)
    claims = _claims(method_summary, lr_table)
    status = pd.DataFrame(status_rows)

    lr_table.to_csv(output_dir / "goal1_injected_downstream_proxy_lr_summary.csv", index=False)
    method_summary.to_csv(output_dir / "goal1_injected_downstream_proxy_method_summary.csv", index=False)
    claims.to_csv(output_dir / "goal1_injected_downstream_proxy_claims.csv", index=False)
    status.to_csv(output_dir / "goal1_injected_downstream_proxy_status.csv", index=False)
    manifest = {
        "analysis_label": "phase2_revised_goal1_injected_downstream_proxy",
        "canonical_goal": "goal1_baseline_matched_biological_validation",
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "datasets": sorted(DATASETS),
        "methods": list(METHODS),
        "exposure_weight": float(args.exposure_weight),
        "claim_level": "formal_injected_receiver_response_proxy_not_curated_downstream",
        "outputs": {
            "lr_summary": str((output_dir / "goal1_injected_downstream_proxy_lr_summary.csv").relative_to(PROJECT_ROOT)),
            "method_summary": str((output_dir / "goal1_injected_downstream_proxy_method_summary.csv").relative_to(PROJECT_ROOT)),
            "claims": str((output_dir / "goal1_injected_downstream_proxy_claims.csv").relative_to(PROJECT_ROOT)),
            "status": str((output_dir / "goal1_injected_downstream_proxy_status.csv").relative_to(PROJECT_ROOT)),
        },
        "limitations": [
            "Receiver response combines receptor expression and injected truth exposure; it is not curated LR-target biology.",
            "This proxy is used to compare methods under matched injected controls, not to validate unlabeled real-tissue ground truth.",
            "Curated NicheNet/Reactome/OmniPath target signatures remain required for final manuscript downstream claims.",
        ],
    }
    (output_dir / "goal1_injected_downstream_proxy_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not claims.empty:
        print(claims.to_string(index=False))
    print(f"Wrote formal injected downstream proxy outputs to {output_dir}")


def _prediction_path(prediction_root: Path, dataset: str, method: str, replicate_name: str) -> Path:
    if method == "spatialself_wavelet_edge_prior":
        filename = f"{replicate_name}__spatialself_wavelet_predictions.csv"
    else:
        filename = f"{replicate_name}__{method}_predictions.csv"
    return prediction_root / dataset / method / filename


def _merge(candidates: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"candidate_id", "score"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction missing columns: {sorted(missing)}")
    merged = candidates.merge(predictions[["candidate_id", "score"]], on="candidate_id", how="inner", validate="one_to_one")
    if len(merged) != len(candidates):
        raise ValueError(f"Prediction/candidate row mismatch: {len(merged)} != {len(candidates)}")
    merged["score"] = pd.to_numeric(merged["score"], errors="coerce").fillna(0.0)
    return merged


def _lr_proxy_rows(
    annotated: pd.DataFrame,
    expression: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    *,
    dataset: str,
    method: str,
    prediction_path: Path,
    exposure_weight: float,
) -> list[dict]:
    expression = expression.set_index("cell_id")
    lr_meta = lr_pairs.set_index("lr_id")
    rows = []
    for lr_id, group in annotated.groupby("lr_id", sort=False):
        lr = lr_meta.loc[lr_id]
        receptor = str(lr["receptor"])
        if receptor not in expression.columns:
            rows.append(_base_row(dataset, method, prediction_path, lr_id, lr, "receptor_not_observed"))
            continue
        receiver_score = group.groupby("receiver_id", sort=False)["score"].mean()
        truth_exposure = group.assign(_truth_score=group["ground_truth"].astype(float) * (1.0 / (1.0 + group["distance"].astype(float)))).groupby("receiver_id", sort=False)["_truth_score"].sum()
        cells = expression.index.intersection(receiver_score.index)
        if len(cells) < 4:
            rows.append(_base_row(dataset, method, prediction_path, lr_id, lr, "too_few_receiver_cells"))
            continue
        communication = receiver_score.loc[cells].to_numpy(dtype=float)
        receptor_expr = np.log1p(expression.loc[cells, receptor].to_numpy(dtype=float))
        exposure = truth_exposure.reindex(cells).fillna(0.0).to_numpy(dtype=float)
        exposure = _zscore(exposure)
        response = _zscore(receptor_expr) + exposure_weight * exposure
        if np.nanstd(communication) <= 1e-12 or np.nanstd(response) <= 1e-12:
            rows.append(_base_row(dataset, method, prediction_path, lr_id, lr, "constant_vector"))
            continue
        rows.append(
            {
                **_base_row(dataset, method, prediction_path, lr_id, lr, "tested"),
                "n_receiver_cells": int(len(cells)),
                "n_true_edges": int(group["ground_truth"].astype(int).sum()),
                "response_proxy_spearman": _spearman(communication, response),
                "receptor_only_spearman": _spearman(communication, receptor_expr),
                "truth_exposure_only_spearman": _spearman(communication, exposure),
                "mean_truth_exposure": float(np.mean(truth_exposure.to_numpy(dtype=float))) if len(truth_exposure) else 0.0,
            }
        )
    return rows


def _base_row(dataset: str, method: str, prediction_path: Path, lr_id: str, lr: pd.Series, status: str) -> dict:
    return {
        "dataset": dataset,
        "method": method,
        "lr_id": lr_id,
        "ligand": lr["ligand"],
        "receptor": lr["receptor"],
        "mechanism": lr["mechanism"],
        "true_lr": bool(lr.get("true_lr", True)) and not bool(lr.get("false_positive_high_expression", False)),
        "weak_signal": bool(lr.get("weak_signal", False)),
        "false_positive_high_expression": bool(lr.get("false_positive_high_expression", False)),
        "status": status,
        "n_receiver_cells": 0,
        "n_true_edges": 0,
        "response_proxy_spearman": float("nan"),
        "receptor_only_spearman": float("nan"),
        "truth_exposure_only_spearman": float("nan"),
        "mean_truth_exposure": float("nan"),
        "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)),
    }


def _method_summary(lr_table: pd.DataFrame) -> pd.DataFrame:
    if lr_table.empty:
        return pd.DataFrame()
    rows = []
    for (dataset, method), group in lr_table.groupby(["dataset", "method"], sort=False):
        tested = group[group["status"].astype(str) == "tested"]
        positives = tested[tested["true_lr"].astype(bool)]
        negatives = tested[tested["false_positive_high_expression"].astype(bool)]
        weak = positives[positives["weak_signal"].astype(bool)]
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "n_lr_pairs": int(len(group)),
                "n_tested_lr_pairs": int(len(tested)),
                "n_positive_controls": int(len(positives)),
                "n_negative_controls": int(len(negatives)),
                "mean_positive_response_proxy_spearman": _mean(positives["response_proxy_spearman"]),
                "mean_negative_response_proxy_spearman": _mean(negatives["response_proxy_spearman"]),
                "mean_positive_receptor_only_spearman": _mean(positives["receptor_only_spearman"]),
                "mean_negative_receptor_only_spearman": _mean(negatives["receptor_only_spearman"]),
                "mean_positive_truth_exposure_spearman": _mean(positives["truth_exposure_only_spearman"]),
                "mean_negative_truth_exposure_spearman": _mean(negatives["truth_exposure_only_spearman"]),
                "weak_positive_response_proxy_spearman": _mean(weak["response_proxy_spearman"]),
                "positive_negative_response_delta": _mean(positives["response_proxy_spearman"]) - _mean(negatives["response_proxy_spearman"]),
                "positive_negative_truth_exposure_delta": _mean(positives["truth_exposure_only_spearman"]) - _mean(negatives["truth_exposure_only_spearman"]),
                "proxy_level": "injected_receptor_plus_truth_exposure_not_curated_downstream",
            }
        )
    return pd.DataFrame(rows)


def _claims(method_summary: pd.DataFrame, lr_table: pd.DataFrame) -> pd.DataFrame:
    if method_summary.empty:
        return pd.DataFrame()
    rows = []
    for dataset, group in method_summary.groupby("dataset", sort=False):
        ranked = group.sort_values("positive_negative_response_delta", ascending=False)
        best_delta = ranked.iloc[0]
        spatial = group[group["method"].astype(str).str.startswith("spatialself")]
        non_spatial = group[~group["method"].astype(str).str.startswith("spatialself")]
        rows.append(
            {
                "dataset": dataset,
                "claim_level": "formal_injected_receiver_response_proxy_not_curated_downstream",
                "best_response_delta_method": best_delta["method"],
                "best_response_delta": best_delta["positive_negative_response_delta"],
                "spatialself_best_response_delta": str(best_delta["method"]).startswith("spatialself"),
                "spatialself_max_response_delta": float(spatial["positive_negative_response_delta"].max()) if len(spatial) else float("nan"),
                "non_spatial_max_response_delta": float(non_spatial["positive_negative_response_delta"].max()) if len(non_spatial) else float("nan"),
                "spatialself_beats_non_spatial_on_response_delta": bool(
                    len(spatial)
                    and len(non_spatial)
                    and float(spatial["positive_negative_response_delta"].max()) >= float(non_spatial["positive_negative_response_delta"].max())
                ),
                "tested_methods": int(group["method"].nunique()),
                "tested_lr_rows": int(len(lr_table[(lr_table["dataset"] == dataset) & (lr_table["status"] == "tested")])),
            }
        )
    return pd.DataFrame(rows)


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    std = np.nanstd(values)
    if std <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - np.nanmean(values)) / std


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    if np.std(xr) <= 1e-12 or np.std(yr) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def _mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")


if __name__ == "__main__":
    main()
