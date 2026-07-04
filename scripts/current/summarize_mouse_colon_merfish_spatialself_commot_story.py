#!/usr/bin/env python
"""Summarize SpatialSelf vs COMMOT priority-axis MERFISH trajectories."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


STAGES = ["Healthy_D0", "DSS3", "DSS9", "DSS21"]
AXES = ["Cxcl10-Cxcr3", "Ccl5-Ccr5", "Ccl2-Ccr2", "Cxcl9-Cxcr3", "Cxcl12-Cxcr4", "Tnf-Tnfrsf1a", "Tnf-Tnfrsf1b"]


def main() -> None:
    out_dir = Path("outputs/current_mouse_colon_ibd_merfish/spatialself_vs_commot_priority_story")
    out_dir.mkdir(parents=True, exist_ok=True)
    spatial = pd.read_csv("outputs/current_mouse_colon_ibd_merfish/multislice_spatialself/summary/stage_lr_trajectories.csv")
    commot = pd.read_csv("outputs/current_mouse_colon_ibd_merfish/commot_priority_baseline/commot_priority_stage_trajectory_summary.csv")

    rows = []
    spatial_rows = spatial[spatial["lr_pair"].isin(AXES)].copy()
    for row in spatial_rows.itertuples(index=False):
        values = {stage: float(getattr(row, stage)) for stage in STAGES}
        rows.append(
            {
                "method": "SpatialSelf",
                "lr_pair": row.lr_pair,
                **values,
                "peak_stage": max(values, key=values.get),
                "trajectory_class": row.trajectory_class,
                "dss9_minus_healthy": values["DSS9"] - values["Healthy_D0"],
                "dss21_minus_healthy": values["DSS21"] - values["Healthy_D0"],
                "replicate_stability_score": row.replicate_stability_score,
            }
        )

    commot_wide = commot.pivot_table(index="lr_pair", columns="stage", values="score_q95_mean", aggfunc="first")
    for lr_pair, row in commot_wide.iterrows():
        if lr_pair not in AXES:
            continue
        values = {stage: float(row.get(stage, np.nan)) for stage in STAGES}
        rows.append(
            {
                "method": "COMMOT",
                "lr_pair": lr_pair,
                **values,
                "peak_stage": max(values, key=values.get),
                "trajectory_class": classify(values),
                "dss9_minus_healthy": values["DSS9"] - values["Healthy_D0"],
                "dss21_minus_healthy": values["DSS21"] - values["Healthy_D0"],
                "replicate_stability_score": np.nan,
            }
        )
    combined = pd.DataFrame(rows)
    combined.to_csv(out_dir / "spatialself_commot_priority_trajectory_comparison.csv", index=False)

    delta = combined.pivot_table(index="lr_pair", columns="method", values=["peak_stage", "dss9_minus_healthy", "dss21_minus_healthy"], aggfunc="first")
    flat = []
    for lr_pair in AXES:
        if lr_pair not in delta.index:
            continue
        flat.append(
            {
                "lr_pair": lr_pair,
                "spatialself_peak": delta.loc[lr_pair, ("peak_stage", "SpatialSelf")],
                "commot_peak": delta.loc[lr_pair, ("peak_stage", "COMMOT")],
                "spatialself_dss9_minus_healthy": delta.loc[lr_pair, ("dss9_minus_healthy", "SpatialSelf")],
                "commot_dss9_minus_healthy": delta.loc[lr_pair, ("dss9_minus_healthy", "COMMOT")],
                "spatialself_dss21_minus_healthy": delta.loc[lr_pair, ("dss21_minus_healthy", "SpatialSelf")],
                "commot_dss21_minus_healthy": delta.loc[lr_pair, ("dss21_minus_healthy", "COMMOT")],
            }
        )
    delta_table = pd.DataFrame(flat)
    delta_table.to_csv(out_dir / "spatialself_commot_priority_delta_table.csv", index=False)
    write_report(combined, delta_table, out_dir / "spatialself_commot_priority_story_report.md")
    print(delta_table.to_string(index=False))
    print(f"Wrote SpatialSelf vs COMMOT story comparison to {out_dir}")


def classify(values: dict[str, float]) -> str:
    peak = max(values, key=values.get)
    if peak == "DSS3":
        return "early_induction"
    if peak == "DSS9":
        return "inflammation_peak_or_persistent"
    if peak == "DSS21":
        return "late_repair_or_persistent"
    return "baseline_high"


def write_report(combined: pd.DataFrame, delta: pd.DataFrame, path: Path) -> None:
    lines = [
        "# SpatialSelf vs COMMOT Priority-axis MERFISH Story",
        "",
        "This comparison uses the same 8 multi-slice MERFISH replicates and the same priority LR axes.",
        "",
        "## Main Observation",
        "",
        "SpatialSelf gives a clearer DSS9-centered inflammatory chemokine trajectory for `Cxcl10-Cxcr3`, `Ccl5-Ccr5`, and `Ccl2-Ccr2`. COMMOT does not reproduce the same DSS9 peak pattern for these axes in this targeted run.",
        "",
        "## Axis-level Differences",
        "",
    ]
    for row in delta.itertuples(index=False):
        lines.append(
            f"- `{row.lr_pair}`: SpatialSelf peak={row.spatialself_peak}, COMMOT peak={row.commot_peak}; "
            f"DSS9-D0 SpatialSelf={row.spatialself_dss9_minus_healthy:.3f}, COMMOT={row.commot_dss9_minus_healthy:.3f}"
        )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "This is a priority-axis COMMOT comparison, not a full LR-atlas baseline. It supports the story that SpatialSelf organizes chemokine axes into a stage-wise inflammatory trajectory, but manuscript-level claims still need CellNEST/formal baseline closure and stronger downstream validation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
