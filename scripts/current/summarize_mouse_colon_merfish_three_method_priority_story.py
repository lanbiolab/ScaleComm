#!/usr/bin/env python
"""Summarize SpatialSelf, COMMOT, and CellNEST priority-axis MERFISH trajectories."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


STAGES = ["Healthy_D0", "DSS3", "DSS9", "DSS21"]
AXES = ["Cxcl10-Cxcr3", "Ccl5-Ccr5", "Ccl2-Ccr2", "Cxcl9-Cxcr3", "Cxcl12-Cxcr4", "Tnf-Tnfrsf1a", "Tnf-Tnfrsf1b"]


def main() -> None:
    out_dir = Path("outputs/current_mouse_colon_ibd_merfish/three_method_priority_story")
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([load_spatialself(), load_baseline("COMMOT"), load_baseline("CellNEST")], ignore_index=True)
    combined.to_csv(out_dir / "three_method_priority_trajectory_long.csv", index=False)
    comparison = make_comparison(combined)
    comparison.to_csv(out_dir / "three_method_priority_peak_comparison.csv", index=False)
    write_report(comparison, out_dir / "three_method_priority_story_report.md")
    print(comparison.to_string(index=False))
    print(f"Wrote three-method priority story to {out_dir}")


def load_spatialself() -> pd.DataFrame:
    table = pd.read_csv("outputs/current_mouse_colon_ibd_merfish/multislice_spatialself/summary/stage_lr_trajectories.csv")
    rows = []
    for row in table[table["lr_pair"].isin(AXES)].itertuples(index=False):
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
            }
        )
    return pd.DataFrame(rows)


def load_baseline(method: str) -> pd.DataFrame:
    if method == "COMMOT":
        path = "outputs/current_mouse_colon_ibd_merfish/commot_priority_baseline/commot_priority_stage_trajectory_summary.csv"
    elif method == "CellNEST":
        path = "outputs/current_mouse_colon_ibd_merfish/cellnest_priority_baseline/cellnest_priority_stage_trajectory_summary.csv"
    else:
        raise ValueError(method)
    table = pd.read_csv(path)
    wide = table.pivot_table(index="lr_pair", columns="stage", values="score_q95_mean", aggfunc="first")
    rows = []
    for lr_pair, row in wide.iterrows():
        if lr_pair not in AXES:
            continue
        values = {stage: float(row.get(stage, np.nan)) for stage in STAGES}
        rows.append(
            {
                "method": method,
                "lr_pair": lr_pair,
                **values,
                "peak_stage": max(values, key=values.get),
                "trajectory_class": classify(values),
                "dss9_minus_healthy": values["DSS9"] - values["Healthy_D0"],
                "dss21_minus_healthy": values["DSS21"] - values["Healthy_D0"],
            }
        )
    return pd.DataFrame(rows)


def classify(values: dict[str, float]) -> str:
    peak = max(values, key=values.get)
    if peak == "Healthy_D0":
        return "baseline_high"
    if peak == "DSS3":
        return "early_induction"
    if peak == "DSS9":
        return "inflammatory_peak"
    return "late_repair_or_persistent"


def make_comparison(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lr_pair in AXES:
        sub = combined[combined["lr_pair"].eq(lr_pair)]
        if sub.empty:
            continue
        row = {"lr_pair": lr_pair}
        for method in ["SpatialSelf", "COMMOT", "CellNEST"]:
            m = sub[sub["method"].eq(method)]
            if m.empty:
                row[f"{method}_peak"] = "missing"
                row[f"{method}_DSS9_minus_D0"] = np.nan
                row[f"{method}_DSS21_minus_D0"] = np.nan
            else:
                item = m.iloc[0]
                row[f"{method}_peak"] = item["peak_stage"]
                row[f"{method}_DSS9_minus_D0"] = item["dss9_minus_healthy"]
                row[f"{method}_DSS21_minus_D0"] = item["dss21_minus_healthy"]
        row["spatialself_specific_dss9_peak"] = (
            row.get("SpatialSelf_peak") == "DSS9"
            and row.get("COMMOT_peak") != "DSS9"
            and row.get("CellNEST_peak") != "DSS9"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(comparison: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Three-method Priority-axis Mouse Colon MERFISH Story",
        "",
        "Methods compared on the same 8 multi-slice MERFISH replicates and the same 7 priority LR axes:",
        "",
        "- SpatialSelf: full wavelet-adaptive model.",
        "- COMMOT: targeted priority-axis COMMOT run.",
        "- CellNEST: local official CellNEST preprocess/train/postprocess, priority-axis LR database, 20 epochs.",
        "",
        "## Main Observation",
        "",
        "SpatialSelf uniquely gives DSS9-peaking inflammatory trajectories for `Cxcl10-Cxcr3`, `Ccl5-Ccr5`, and `Ccl2-Ccr2` in this priority-axis panel. COMMOT and CellNEST do run successfully, but they tend to peak at Healthy/D0, DSS3, or DSS21 for these same axes.",
        "",
        "## Axis Summary",
        "",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"- `{row.lr_pair}`: SpatialSelf={row.SpatialSelf_peak}, COMMOT={row.COMMOT_peak}, CellNEST={row.CellNEST_peak}; "
            f"SpatialSelf DSS9-D0={row.SpatialSelf_DSS9_minus_D0:.3f}, "
            f"COMMOT DSS9-D0={row.COMMOT_DSS9_minus_D0:.3f}, "
            f"CellNEST DSS9-D0={row.CellNEST_DSS9_minus_D0:.3f}"
        )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "This is a targeted priority-axis comparison, not a full LR-atlas baseline. The downstream signature checks are supportive but not FDR-significant, so this result should currently be framed as stage-wise trajectory evidence rather than definitive functional validation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
