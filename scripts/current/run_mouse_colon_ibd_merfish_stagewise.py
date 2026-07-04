#!/usr/bin/env python
"""Run SpatialSelf on stage-wise mouse colon IBD MERFISH replicates and summarize trajectories."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_spatialself import run_spatialself_on_replicate


STAGE_ORDER = {"Healthy_D0": 0, "DSS3": 3, "DSS9": 9, "DSS21": 21}
STAGE_COLUMNS = ["Healthy_D0", "DSS3", "DSS9", "DSS21"]

PRIORITY_AXES = {
    "Ccl2-Ccr2",
    "Ccl5-Ccr5",
    "Cxcl9-Cxcr3",
    "Cxcl10-Cxcr3",
    "Cxcl12-Cxcr4",
    "Spp1-Cd44",
    "Tnf-Tnfrsf1a",
    "Tnf-Tnfrsf1b",
    "Il1b-Il1r1",
    "Il6-Il6st",
    "Tgfb1-Tgfbr2",
    "Pdgfa-Pdgfra",
    "Pdgfb-Pdgfrb",
    "Vegfa-Kdr",
    "Kitl-Kit",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replicate-root",
        default="outputs/current_mouse_colon_ibd_merfish/stagewise_replicates",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/current_mouse_colon_ibd_merfish/stagewise_spatialself",
    )
    parser.add_argument("--model-name", default="spatialself_wavelet_scale")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--k-nn", type=int, default=18)
    parser.add_argument("--max-train-samples", type=int, default=18000)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    replicate_root = Path(args.replicate_root)
    output_dir = Path(args.output_dir)
    prediction_root = output_dir / "predictions"
    summary_dir = output_dir / "summary"
    prediction_root.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    replicate_dirs = find_replicates(replicate_root)
    if not replicate_dirs:
        raise FileNotFoundError(f"No stage-wise replicates found under {replicate_root}")

    status_rows = []
    for replicate_dir in replicate_dirs:
        stage = stage_from_replicate(replicate_dir)
        pred_dir = prediction_root / stage / args.model_name
        pred_file = pred_dir / f"{replicate_dir.name}__{args.model_name}_predictions.csv"
        started = time.perf_counter()
        if args.skip_existing and pred_file.exists():
            status_rows.append(
                {
                    "stage": stage,
                    "status": "skipped_existing",
                    "replicate_dir": str(replicate_dir),
                    "prediction_path": str(pred_file),
                    "runtime_seconds": 0.0,
                    "error": "",
                }
            )
            continue
        try:
            metrics = run_spatialself_on_replicate(
                replicate_dir,
                pred_dir,
                model_name=args.model_name,
                k_nn=int(args.k_nn),
                epochs=int(args.epochs),
                pseudo_label_strategy="mechanism_niche_expression_distance",
                train_sampling_strategy="balanced_per_lr",
                hard_negative_loss_weight=0.05,
                mechanism_guard_loss_weight=0.03,
                scale_contrastive_loss_weight=0.04 if args.model_name == "spatialself_wavelet_adaptive" else 0.0,
                context_contrastive_loss_weight=0.04,
                context_contrastive_negative_mode="hard_receiver",
                lr_cap_positive=28,
                lr_cap_negative=70,
                max_train_samples=int(args.max_train_samples),
                seed=20260620 + STAGE_ORDER.get(stage, 0),
            )
            status_rows.append(
                {
                    "stage": stage,
                    "status": "completed",
                    "replicate_dir": str(replicate_dir),
                    "prediction_path": metrics["prediction_path"],
                    "runtime_seconds": float(time.perf_counter() - started),
                    "error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            status_rows.append(
                {
                    "stage": stage,
                    "status": "failed",
                    "replicate_dir": str(replicate_dir),
                    "prediction_path": "",
                    "runtime_seconds": float(time.perf_counter() - started),
                    "error": repr(exc),
                }
            )

    status = pd.DataFrame(status_rows)
    status.to_csv(output_dir / "stagewise_spatialself_status.csv", index=False)
    completed = status[status["prediction_path"].astype(str) != ""]
    if not completed.empty:
        lr_summary = summarize_stage_lr(completed)
        lr_summary.to_csv(summary_dir / "stage_lr_summary_long.csv", index=False)
        trajectories = summarize_trajectories(lr_summary)
        trajectories.to_csv(summary_dir / "stage_lr_trajectories.csv", index=False)
        story = select_story_candidates(trajectories)
        story.to_csv(summary_dir / "stage_lr_story_candidates.csv", index=False)
        write_story_markdown(story, output_dir / "stagewise_story_summary.md")
        print(story.head(40).to_string(index=False))
    print(status.to_string(index=False))


def find_replicates(root: Path) -> list[Path]:
    required = {"cells.csv", "expression.csv", "lr_pairs.csv", "candidate_interactions.csv"}
    reps = []
    for path in sorted(root.iterdir()) if root.exists() else []:
        if path.is_dir() and required.issubset({file.name for file in path.iterdir() if file.is_file()}):
            reps.append(path)
    return sorted(reps, key=lambda path: STAGE_ORDER.get(stage_from_replicate(path), 999))


def stage_from_replicate(path: Path) -> str:
    name = path.name
    for stage in STAGE_ORDER:
        if stage in name:
            return stage
    metadata_path = path / "metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8")).get("stage", name)
    return name


def summarize_stage_lr(status: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in status.itertuples(index=False):
        stage = row.stage
        replicate_dir = Path(row.replicate_dir)
        candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
        lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
        pred = pd.read_csv(row.prediction_path)
        merged = candidates.merge(pred[["candidate_id", "score"]], on="candidate_id", how="left")
        merged["score"] = pd.to_numeric(merged["score"], errors="coerce").fillna(0.0)
        merged["lr_pair"] = merged["ligand"].astype(str) + "-" + merged["receptor"].astype(str)
        merged["expr_product"] = np.log1p(merged["ligand_expr"].astype(float)) * np.log1p(
            merged["receptor_expr"].astype(float)
        )
        agg = (
            merged.groupby(["lr_id", "lr_pair", "ligand", "receptor", "mechanism"], dropna=False)
            .agg(
                n_edges=("candidate_id", "size"),
                score_sum=("score", "sum"),
                score_mean=("score", "mean"),
                score_q90=("score", lambda x: float(np.quantile(x, 0.90))),
                score_q95=("score", lambda x: float(np.quantile(x, 0.95))),
                score_max=("score", "max"),
                expr_product_mean=("expr_product", "mean"),
                expressed_edge_fraction=("expr_product", lambda x: float((x > 0).mean())),
                median_distance=("distance", "median"),
            )
            .reset_index()
        )
        cell_pair = (
            merged.groupby(["lr_id", "sender_cell_type", "receiver_cell_type"], dropna=False)["score"]
            .sum()
            .reset_index()
            .sort_values(["lr_id", "score"], ascending=[True, False])
            .drop_duplicates("lr_id")
            .rename(columns={"score": "top_cell_type_pair_score"})
        )
        cell_pair["top_cell_type_pair"] = cell_pair["sender_cell_type"].astype(str) + " -> " + cell_pair[
            "receiver_cell_type"
        ].astype(str)
        agg = agg.merge(cell_pair[["lr_id", "top_cell_type_pair", "top_cell_type_pair_score"]], on="lr_id", how="left")
        lr_meta_cols = [
            col
            for col in [
                "lr_id",
                "biological_category",
                "spatial_range",
                "selected_reason",
                "false_positive_high_expression",
            ]
            if col in lr_pairs.columns
        ]
        agg = agg.merge(lr_pairs[lr_meta_cols].drop_duplicates("lr_id"), on="lr_id", how="left")
        agg["stage"] = stage
        agg["stage_day"] = STAGE_ORDER.get(stage, np.nan)
        agg["replicate"] = replicate_dir.name
        metadata_path = replicate_dir / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            agg["source_slice_file"] = metadata.get("source_slice_file", "")
            agg["mouse_id"] = metadata.get("mouse_id", "")
            agg["stage_replicate"] = metadata.get("stage_replicate", np.nan)
        else:
            agg["source_slice_file"] = ""
            agg["mouse_id"] = ""
            agg["stage_replicate"] = np.nan
        agg["rank_q95"] = agg["score_q95"].rank(method="min", ascending=False)
        rows.append(agg)
    return pd.concat(rows, ignore_index=True)


def summarize_trajectories(lr_summary: pd.DataFrame) -> pd.DataFrame:
    metric = "score_q95"
    index_cols = ["lr_pair", "ligand", "receptor", "mechanism"]
    wide = lr_summary.pivot_table(index=index_cols, columns="stage", values=metric, aggfunc="mean")
    for stage in STAGE_COLUMNS:
        if stage not in wide:
            wide[stage] = np.nan
    wide = wide[STAGE_COLUMNS].reset_index()
    replicate_stats = lr_summary.groupby(index_cols + ["stage"], dropna=False)[metric].agg(["mean", "std", "count"]).reset_index()
    for stat in ["mean", "std", "count"]:
        stat_wide = replicate_stats.pivot_table(index=index_cols, columns="stage", values=stat, aggfunc="first")
        stat_wide = stat_wide.rename(columns={stage: f"{stage}_{stat}" for stage in stat_wide.columns}).reset_index()
        wide = wide.merge(stat_wide, on=index_cols, how="left")
    meta = (
        lr_summary.sort_values(["lr_pair", "score_q95"], ascending=[True, False])
        .groupby("lr_pair", as_index=False)
        .first()[
            [
                "lr_pair",
                "biological_category",
                "spatial_range",
                "expr_product_mean",
                "expressed_edge_fraction",
                "top_cell_type_pair",
            ]
        ]
    )
    out = wide.merge(meta, on="lr_pair", how="left")
    eps = 1e-6
    out["dss3_delta"] = out["DSS3"] - out["Healthy_D0"]
    out["dss9_delta"] = out["DSS9"] - out["Healthy_D0"]
    out["dss21_delta"] = out["DSS21"] - out["Healthy_D0"]
    out["dss9_vs_dss3"] = out["DSS9"] - out["DSS3"]
    out["recovery_drop"] = out["DSS9"] - out["DSS21"]
    stage_values = out[STAGE_COLUMNS]
    out["peak_stage"] = stage_values.idxmax(axis=1)
    out["peak_score"] = stage_values.max(axis=1)
    out["healthy_score"] = out["Healthy_D0"]
    out["peak_delta_vs_healthy"] = out["peak_score"] - out["healthy_score"]
    out["peak_ratio_vs_healthy"] = (out["peak_score"] + eps) / (out["healthy_score"] + eps)
    count_cols = [f"{stage}_count" for stage in STAGE_COLUMNS if f"{stage}_count" in out.columns]
    std_cols = [f"{stage}_std" for stage in STAGE_COLUMNS if f"{stage}_std" in out.columns]
    out["min_stage_replicates"] = out[count_cols].min(axis=1) if count_cols else 1
    out["mean_stage_std"] = out[std_cols].mean(axis=1) if std_cols else 0.0
    out["replicate_stability_score"] = 1.0 / (1.0 + out["mean_stage_std"].fillna(0.0))
    out["trajectory_class"] = out.apply(classify_trajectory, axis=1)
    out["priority_axis"] = out["lr_pair"].isin(PRIORITY_AXES)
    out["story_score"] = (
        out["peak_delta_vs_healthy"].fillna(0)
        + 0.5 * out["recovery_drop"].clip(lower=0).fillna(0)
        + 0.08 * out["priority_axis"].astype(float)
        + 0.05 * out["replicate_stability_score"].fillna(0)
    )
    return out.sort_values(["story_score", "peak_score"], ascending=False)


def classify_trajectory(row: pd.Series) -> str:
    if row["peak_stage"] == "DSS3" and row["dss3_delta"] > 0:
        return "early_induction"
    if row["peak_stage"] == "DSS9" and row["dss9_delta"] > 0:
        if row["DSS21"] >= row["DSS9"] - 0.05 * max(abs(row["DSS9"]), 1e-6):
            return "persistent_inflammation"
        return "inflammation_peak_recovery"
    if row["peak_stage"] == "DSS21" and row["dss21_delta"] > 0:
        return "late_repair_or_persistent"
    if row["peak_delta_vs_healthy"] > 0:
        return "modest_stage_shift"
    return "flat_or_decreased"


def select_story_candidates(trajectories: pd.DataFrame) -> pd.DataFrame:
    keep = trajectories[
        (trajectories["peak_delta_vs_healthy"] > 0)
        & (trajectories["expressed_edge_fraction"].fillna(0) >= 0.05)
    ].copy()
    priority = keep[keep["priority_axis"]].sort_values(["story_score", "peak_score"], ascending=False)
    top_by_class = (
        keep.sort_values(["trajectory_class", "story_score"], ascending=[True, False])
        .groupby("trajectory_class", as_index=False)
        .head(12)
    )
    story = pd.concat([priority, top_by_class], ignore_index=True).drop_duplicates("lr_pair")
    return story.sort_values(["priority_axis", "story_score"], ascending=[False, False])


def write_story_markdown(story: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Mouse Colon IBD MERFISH Stage-wise CCC Story",
        "",
        "Working claim:",
        "",
        "> SpatialSelf is used to trace communication programs across healthy colon, early DSS injury, peak inflammation, and recovery/persistent remodeling.",
        "",
        "This is a representative-slice pilot. Full-slice/full-animal replication and baseline comparison are still required before manuscript-level claims.",
        "",
        "## Candidate axes",
        "",
    ]
    for row in story.head(30).itertuples(index=False):
        lines.append(
            f"- `{row.lr_pair}`: {row.trajectory_class}; peak={row.peak_stage}; "
            f"healthy={row.Healthy_D0:.3f}, DSS3={row.DSS3:.3f}, DSS9={row.DSS9:.3f}, DSS21={row.DSS21:.3f}; "
            f"top cell-type pair={row.top_cell_type_pair}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
