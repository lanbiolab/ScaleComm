#!/usr/bin/env python
"""Build a reusable broad/background guard target panel for the next model.

The formal-parameter coordinate rerun shows that high-expression decoys are
suppressed, while many low-score broad/background LR share the minimum coarse
coordinate p-value. This script turns that failure mode into a concrete
training/calibration target panel.

This is not a discovery result and not a final FDR rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
LEAKAGE_DIR = PHASE12_DIR / "integrated_guarded_spatial_coordinate_leakage"
MAGNITUDE_DIR = PHASE12_DIR / "integrated_guarded_spatial_coordinate_magnitude_gate"
BACKGROUND_DIR = PHASE12_DIR / "integrated_guarded_spatial_coordinate_background_targets"
OUT_DIR = PHASE12_DIR / "integrated_background_guard_training_targets"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    leakage = pd.read_csv(LEAKAGE_DIR / "phase12_integrated_coordinate_leakage_lr.csv")
    background_top = read_csv(BACKGROUND_DIR / "phase12_integrated_coordinate_background_target_lr.csv")
    gate = pd.read_csv(MAGNITUDE_DIR / "phase12_integrated_coordinate_magnitude_gate_best.csv")

    gate_threshold = float(gate.iloc[0]["score_threshold"]) if len(gate) else 0.03
    leakage = normalize(leakage, gate_threshold)
    target_panel = build_target_panel(leakage, background_top, gate_threshold)
    spec = build_training_spec(target_panel, leakage, gate_threshold)
    category = summarize_categories(target_panel)

    target_panel.to_csv(OUT_DIR / "phase12_integrated_background_guard_target_panel.csv", index=False)
    category.to_csv(OUT_DIR / "phase12_integrated_background_guard_category_summary.csv", index=False)
    spec.to_csv(OUT_DIR / "phase12_integrated_background_guard_training_spec.csv", index=False)
    write_status(spec, category)
    write_manifest()

    print(spec.to_string(index=False))
    print()
    print(category.head(20).to_string(index=False))


def normalize(table: pd.DataFrame, gate_threshold: float) -> pd.DataFrame:
    out = table.copy()
    for col in [
        "true_lr",
        "false_positive_high_expression",
        "broad_background_lr",
        "priority_true_lr",
        "high_expression_negative",
    ]:
        if col in out:
            out[col] = out[col].fillna(False).astype(bool)
    for col in [
        "rank",
        "p_value",
        "bh_fdr",
        "calibrated_score",
        "null_q95_calibrated_score",
        "coordinate_rerun_p_value",
        "coordinate_rerun_bh_fdr",
    ]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["is_min_p"] = np.isclose(out["p_value"], float(out["p_value"].min()))
    out["above_magnitude_gate"] = out["calibrated_score"].ge(float(gate_threshold))
    return out


def build_target_panel(leakage: pd.DataFrame, background_top: pd.DataFrame, gate_threshold: float) -> pd.DataFrame:
    broad = leakage[leakage["broad_background_lr"]].copy()
    top_lr = set(background_top["lr_pair"].astype(str)) if "lr_pair" in background_top else set()
    broad["target_priority"] = np.select(
        [
            broad["is_min_p"] & broad["lr_pair"].astype(str).isin(top_lr),
            broad["is_min_p"],
            broad["rank"].le(50),
        ],
        [
            "P1_min_p_top_background",
            "P2_min_p_background",
            "P3_top50_background",
        ],
        default="P4_background_pool",
    )
    broad["next_model_role"] = np.select(
        [
            broad["target_priority"].eq("P1_min_p_top_background"),
            broad["target_priority"].eq("P2_min_p_background"),
            broad["target_priority"].eq("P3_top50_background"),
        ],
        [
            "primary_background_hard_negative",
            "background_hard_negative",
            "background_calibration_negative",
        ],
        default="background_pool_negative",
    )
    broad["recommended_loss_weight"] = np.select(
        [
            broad["target_priority"].eq("P1_min_p_top_background"),
            broad["target_priority"].eq("P2_min_p_background"),
            broad["target_priority"].eq("P3_top50_background"),
        ],
        [1.0, 0.7, 0.4],
        default=0.2,
    )
    broad["recommended_margin_to_gate"] = (gate_threshold - broad["calibrated_score"]).clip(lower=0.0)
    broad["failure_mode"] = "low_score_broad_background_min_p_tie"
    broad["guard_objective"] = (
        "Keep coordinate-null support for true anchors while forcing low-score broad/background LR below "
        "a prespecified magnitude/background guard."
    )
    cols = [
        "dataset_key",
        "lr_id",
        "lr_pair",
        "ligand",
        "receptor",
        "rank",
        "biological_category",
        "mechanism",
        "selected_reason",
        "p_value",
        "bh_fdr",
        "calibrated_score",
        "null_q95_calibrated_score",
        "is_min_p",
        "above_magnitude_gate",
        "target_priority",
        "next_model_role",
        "recommended_loss_weight",
        "recommended_margin_to_gate",
        "failure_mode",
        "guard_objective",
    ]
    cols = [col for col in cols if col in broad.columns]
    return broad[cols].sort_values(
        ["target_priority", "rank", "lr_pair"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def build_training_spec(target_panel: pd.DataFrame, leakage: pd.DataFrame, gate_threshold: float) -> pd.DataFrame:
    true = leakage[leakage["true_lr"]].copy()
    high = leakage[leakage["high_expression_negative"]].copy()
    min_p = leakage[leakage["is_min_p"]].copy()
    return pd.DataFrame(
        [
            {
                "dataset_key": str(leakage["dataset_key"].iloc[0]) if len(leakage) else "",
                "n_lr_family": int(len(leakage)),
                "n_completed_coordinate_reruns": int(leakage["n_completed_permutations"].max()),
                "minimum_p_value": float(leakage["p_value"].min()),
                "n_min_p_lr": int(len(min_p)),
                "n_min_p_true_lr": int(min_p["true_lr"].sum()),
                "n_min_p_high_expression_negative": int(min_p["high_expression_negative"].sum()),
                "n_min_p_broad_background": int(min_p["broad_background_lr"].sum()),
                "n_background_targets": int(len(target_panel)),
                "n_primary_background_targets": int(target_panel["target_priority"].eq("P1_min_p_top_background").sum()),
                "n_high_expression_negative": int(len(high)),
                "best_true_lr": best_lr(true),
                "best_true_coordinate_bh_fdr": best_value(true, "coordinate_rerun_bh_fdr"),
                "best_high_expression_negative_lr": best_lr(high),
                "best_high_expression_negative_p": best_value(high, "p_value"),
                "magnitude_gate_threshold": float(gate_threshold),
                "n_background_above_gate": int(target_panel["above_magnitude_gate"].sum()),
                "recommended_next_model_change": (
                    "Promote broad/background target panel into training as background-hard negatives, "
                    "pair it with a prespecified magnitude floor, and rerun exact-score plus end-to-end "
                    "coordinate/null FDR."
                ),
                "success_criterion_before_baseline_replacement": (
                    "No high-expression or broad/background target should pass q<=0.1 under the prespecified "
                    "guarded coordinate family, while at least one prespecified true anchor remains q<=0.1."
                ),
                "claim_boundary": (
                    "This is a next-model training target specification derived from formal20 diagnostics, "
                    "not a discovery result or FDR closure."
                ),
            }
        ]
    )


def summarize_categories(target_panel: pd.DataFrame) -> pd.DataFrame:
    if target_panel.empty:
        return pd.DataFrame()
    return (
        target_panel.groupby(["dataset_key", "biological_category", "target_priority"], dropna=False)
        .agg(
            n_targets=("lr_pair", "count"),
            min_rank=("rank", "min"),
            max_score=("calibrated_score", "max"),
            median_score=("calibrated_score", "median"),
            mean_loss_weight=("recommended_loss_weight", "mean"),
        )
        .reset_index()
        .sort_values(["target_priority", "n_targets", "max_score"], ascending=[True, False, False])
    )


def write_status(spec: pd.DataFrame, category: pd.DataFrame) -> None:
    row = spec.iloc[0]
    lines = [
        "# Integrated Background Guard Training Targets",
        "",
        "## Scope",
        "",
        "This file converts the formal-parameter coordinate-rerun broad/background leakage into a next-model target panel.",
        "",
        "## Result",
        "",
        f"- Dataset: `{row.dataset_key}`",
        f"- Coordinate reruns: {int(row.n_completed_coordinate_reruns)}",
        f"- Minimum-p LR: {int(row.n_min_p_lr)} total; true={int(row.n_min_p_true_lr)}, high-expression negative={int(row.n_min_p_high_expression_negative)}, broad/background={int(row.n_min_p_broad_background)}",
        f"- Background target LR: {int(row.n_background_targets)}",
        f"- Primary background targets: {int(row.n_primary_background_targets)}",
        f"- Magnitude gate threshold: {float(row.magnitude_gate_threshold):.4g}",
        f"- Background targets above gate: {int(row.n_background_above_gate)}",
        "",
        "## Next Model Requirement",
        "",
        str(row.recommended_next_model_change),
        "",
        "## Claim Boundary",
        "",
        str(row.claim_boundary),
    ]
    if not category.empty:
        lines.extend(["", "## Top Target Categories", ""])
        for item in category.head(8).itertuples(index=False):
            lines.append(
                f"- {item.biological_category} / {item.target_priority}: n={int(item.n_targets)}, "
                f"min rank={int(item.min_rank)}, max score={float(item.max_score):.4g}"
            )
    (OUT_DIR / "phase12_integrated_background_guard_training_targets_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_manifest() -> None:
    manifest = {
        "task": "phase12_integrated_background_guard_training_targets",
        "inputs": [
            rel(LEAKAGE_DIR / "phase12_integrated_coordinate_leakage_lr.csv"),
            rel(MAGNITUDE_DIR / "phase12_integrated_coordinate_magnitude_gate_best.csv"),
            rel(BACKGROUND_DIR / "phase12_integrated_coordinate_background_target_lr.csv"),
        ],
        "outputs": [
            rel(OUT_DIR / "phase12_integrated_background_guard_target_panel.csv"),
            rel(OUT_DIR / "phase12_integrated_background_guard_category_summary.csv"),
            rel(OUT_DIR / "phase12_integrated_background_guard_training_spec.csv"),
            rel(OUT_DIR / "phase12_integrated_background_guard_training_targets_status.md"),
        ],
        "claim_boundary": "Next-model target specification, not discovery evidence.",
    }
    (OUT_DIR / "phase12_integrated_background_guard_training_targets_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def best_lr(table: pd.DataFrame) -> str:
    if table.empty:
        return ""
    return str(table.sort_values(["p_value", "rank"]).iloc[0]["lr_pair"])


def best_value(table: pd.DataFrame, column: str) -> float:
    if table.empty or column not in table:
        return float("nan")
    values = pd.to_numeric(table[column], errors="coerce").dropna()
    return float(values.min()) if len(values) else float("nan")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
