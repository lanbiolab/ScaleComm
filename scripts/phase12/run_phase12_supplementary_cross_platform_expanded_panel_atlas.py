#!/usr/bin/env python
"""Run supplementary Phase 12 expanded-panel atlas scoring.

This extends the strict full-transcriptome Phase 12 atlas with real-background
constructed replicates that have 2011-gene expression panels for tonsil,
melanoma and lymph node. The claim boundary is intentionally narrower than the
full-transcriptome atlas: this is a cross-platform expanded-panel supplement,
not broad whole-transcriptome FDR closure.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_spatialself import run_spatialself_on_replicate
import scripts.phase12.run_phase12_atlas_model_pilot as pilot


READINESS_DIR = PROJECT_ROOT / "outputs/phase12/atlas_readiness"
SOURCE_ROOT = PROJECT_ROOT / "outputs/phase2_revised/goal4_real_background_v2/formal_datasets"
OUT_DIR = PROJECT_ROOT / "outputs/phase12/supplementary_cross_platform_expanded_panel_atlas"
REPLICATE_ROOT = OUT_DIR / "replicates"
PRED_ROOT = OUT_DIR / "predictions"

DATASETS = [
    {
        "dataset_key": "human_tonsil_slide_tags_expanded_panel",
        "source_replicate": "human_tonsil_slide_tags_replicate_00",
        "label": "Human tonsil Slide-tags real-background expanded-panel atlas",
        "platform_scope": "Slide-tags-derived real-background constructed replicate; 2011-gene panel",
        "max_spots": 360,
        "max_lr": 9999,
    },
    {
        "dataset_key": "human_melanoma_multiome_expanded_panel",
        "source_replicate": "human_melanoma_multiome_replicate_00",
        "label": "Human melanoma multiome real-background expanded-panel atlas",
        "platform_scope": "Melanoma spatial multiome real-background constructed replicate; 2011-gene panel",
        "max_spots": 360,
        "max_lr": 9999,
    },
    {
        "dataset_key": "human_lymph_node_visium_expanded_panel",
        "source_replicate": "human_lymph_node_visium_replicate_00",
        "label": "Human lymph node Visium real-background expanded-panel atlas",
        "platform_scope": "Visium real-background constructed replicate; 2011-gene panel",
        "max_spots": 360,
        "max_lr": 9999,
    },
]

TOP_K_VALUES = (5, 10, 20, 50)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPLICATE_ROOT.mkdir(parents=True, exist_ok=True)
    PRED_ROOT.mkdir(parents=True, exist_ok=True)

    lr_universe = pd.read_csv(READINESS_DIR / "phase12_lr_universe.csv")
    hard_templates = pd.read_csv(READINESS_DIR / "phase12_hard_negative_templates.csv")

    all_rankings: list[pd.DataFrame] = []
    all_hard: list[pd.DataFrame] = []
    all_status: list[dict[str, Any]] = []
    all_lr_manifest: list[pd.DataFrame] = []

    for spec in DATASETS:
        started = time.perf_counter()
        try:
            replicate_dir, build_summary, lr_pairs = build_replicate(spec, lr_universe, hard_templates)
            pred_dir = PRED_ROOT / spec["dataset_key"] / "spatialself_wavelet_edge_prior"
            if pred_dir.exists():
                shutil.rmtree(pred_dir)
            metrics = run_spatialself_on_replicate(
                replicate_dir,
                pred_dir,
                model_name="spatialself_wavelet",
                k_nn=18,
                epochs=10,
                positive_quantile=0.9,
                negative_quantile=0.2,
                pseudo_label_strategy="lr_prior_lr_capped_mechanism_niche_expression_distance",
                train_sampling_strategy="balanced_per_lr",
                weak_signal_reweighting=2.0,
                weak_positive_fraction=0.15,
                weak_positive_label=0.75,
                weak_positive_mode="non_contact",
                ranking_loss_weight=0.18,
                feasibility_loss_weight=0.12,
                hard_negative_loss_weight=0.12,
                context_contrastive_loss_weight=0.12,
                context_contrastive_margin=0.5,
                context_contrastive_negative_mode="random_receiver",
                lr_identity_contrastive_loss_weight=0.06,
                lr_identity_contrastive_margin=0.35,
                control_contrastive_loss_weight=0.12,
                control_contrastive_margin=0.5,
                null_candidate_loss_weight=0.06,
                lr_cap_positive=18,
                lr_cap_negative=45,
                use_edge_prior_features=True,
                max_train_samples=14000,
                seed=1212,
            )
            pred_path = Path(metrics["prediction_path"])
            ranking = pilot.summarize_lr_ranking(replicate_dir, pred_path, lr_pairs, spec["dataset_key"])
            hard = pilot.summarize_hard_negative(replicate_dir, ranking, spec["dataset_key"])
            all_rankings.append(ranking)
            all_hard.append(hard)
            all_lr_manifest.append(lr_pairs.assign(dataset_key=spec["dataset_key"]))
            all_status.append(
                {
                    "dataset_key": spec["dataset_key"],
                    "source_replicate": spec["source_replicate"],
                    "status": "completed",
                    "runtime_seconds": float(time.perf_counter() - started),
                    "replicate_dir": rel(replicate_dir),
                    "prediction_path": rel(pred_path),
                    **build_summary,
                    "n_candidates_scored": int(metrics.get("n_candidates", 0)),
                    "score_min": float(metrics.get("score_min", np.nan)),
                    "score_max": float(metrics.get("score_max", np.nan)),
                    "error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            all_status.append(
                {
                    "dataset_key": spec["dataset_key"],
                    "source_replicate": spec["source_replicate"],
                    "status": "failed",
                    "runtime_seconds": float(time.perf_counter() - started),
                    "replicate_dir": "",
                    "prediction_path": "",
                    "error": repr(exc),
                }
            )

    ranking_table = pd.concat(all_rankings, ignore_index=True) if all_rankings else pd.DataFrame()
    hard_table = pd.concat(all_hard, ignore_index=True) if all_hard else pd.DataFrame()
    status_table = pd.DataFrame(all_status)
    lr_manifest = pd.concat(all_lr_manifest, ignore_index=True) if all_lr_manifest else pd.DataFrame()
    topk_table = summarize_topk(ranking_table)
    dataset_table = summarize_dataset(ranking_table, topk_table, hard_table, status_table)

    ranking_table.to_csv(OUT_DIR / "phase12_supplementary_expanded_panel_lr_ranking.csv", index=False)
    hard_table.to_csv(OUT_DIR / "phase12_supplementary_expanded_panel_hard_negative_leakage.csv", index=False)
    status_table.to_csv(OUT_DIR / "phase12_supplementary_expanded_panel_status.csv", index=False)
    lr_manifest.to_csv(OUT_DIR / "phase12_supplementary_expanded_panel_lr_manifest.csv", index=False)
    topk_table.to_csv(OUT_DIR / "phase12_supplementary_expanded_panel_topk_summary.csv", index=False)
    dataset_table.to_csv(OUT_DIR / "phase12_supplementary_expanded_panel_dataset_summary.csv", index=False)
    write_status(dataset_table, topk_table, hard_table, status_table)
    write_manifest(status_table)

    print(status_table.to_string(index=False))
    if not dataset_table.empty:
        print()
        print(dataset_table.to_string(index=False))
    if not ranking_table.empty:
        print()
        print(ranking_table.groupby("dataset_key").head(15).to_string(index=False))


def build_replicate(
    spec: dict[str, Any],
    lr_universe: pd.DataFrame,
    hard_templates: pd.DataFrame,
) -> tuple[Path, dict[str, Any], pd.DataFrame]:
    source_dir = SOURCE_ROOT / spec["source_replicate"]
    cells_raw = pd.read_csv(source_dir / "cells.csv")
    expression_raw = pd.read_csv(source_dir / "expression.csv")
    expression_raw["cell_id"] = expression_raw["cell_id"].astype(str)
    genes = [column for column in expression_raw.columns if column != "cell_id"]
    gene_set = set(genes)

    cells = cells_raw[cells_raw["cell_id"].astype(str).isin(set(expression_raw["cell_id"]))].copy()
    cells = pilot.select_spots(cells, int(spec["max_spots"]))
    cells = pilot.normalize_cells(cells)
    expression_raw = expression_raw[expression_raw["cell_id"].astype(str).isin(set(cells["cell_id"].astype(str)))]

    lr_pairs = pilot.select_lr_pairs(lr_universe, hard_templates, spec["dataset_key"], gene_set, int(spec["max_lr"]))
    required_genes = sorted(set(lr_pairs["ligand"]) | set(lr_pairs["receptor"]))
    missing = sorted(set(required_genes) - gene_set)
    if missing:
        raise ValueError(f"Missing required panel genes for {spec['dataset_key']}: {missing[:10]}")

    expression = expression_raw[["cell_id", *required_genes]].copy()
    expression = cells[["cell_id"]].merge(expression, on="cell_id", how="left").fillna(0.0)
    candidates = pilot.build_candidates(cells, expression, lr_pairs, spec["dataset_key"])

    replicate_dir = REPLICATE_ROOT / spec["dataset_key"]
    if replicate_dir.exists():
        shutil.rmtree(replicate_dir)
    replicate_dir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(replicate_dir / "cells.csv", index=False)
    expression.to_csv(replicate_dir / "expression.csv", index=False)
    lr_pairs.to_csv(replicate_dir / "lr_pairs.csv", index=False)
    candidates.to_csv(replicate_dir / "candidate_interactions.csv", index=False)
    (replicate_dir / "metadata.json").write_text(
        json.dumps(
            {
                "dataset_key": spec["dataset_key"],
                "source_replicate": spec["source_replicate"],
                "label": spec["label"],
                "platform_scope": spec["platform_scope"],
                "purpose": "Phase 12 supplementary cross-platform expanded-panel atlas model run",
                "candidate_edges": "multi-scale nearest, medium-range and longer-range sender-receiver candidates",
                "claim_boundary": "Expanded-panel cross-platform supplement; not strict full-transcriptome atlas FDR closure.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary = {
        "n_cells": int(len(cells)),
        "n_panel_genes": int(len(genes)),
        "n_expression_genes": int(len(required_genes)),
        "n_lr_pairs": int(len(lr_pairs)),
        "n_candidates": int(len(candidates)),
        "n_priority_anchor_lr": int(lr_pairs["true_lr"].sum()),
        "n_hard_negative_lr": int(lr_pairs["false_positive_high_expression"].sum()),
    }
    return replicate_dir, summary, lr_pairs


def summarize_topk(ranking: pd.DataFrame) -> pd.DataFrame:
    if ranking.empty:
        return pd.DataFrame()
    rows = []
    annotated = ranking.copy()
    annotated["true_lr"] = annotated["true_lr"].astype(bool)
    annotated["false_positive_high_expression"] = annotated["false_positive_high_expression"].astype(bool)
    annotated["zero_expr_lr"] = pd.to_numeric(annotated["n_nonzero_expr"], errors="coerce").fillna(0).le(0)
    annotated["negative_or_risk"] = annotated["false_positive_high_expression"] | annotated["zero_expr_lr"]
    for dataset_key, sub in annotated.groupby("dataset_key"):
        sub = sub.sort_values("rank")
        for top_k in TOP_K_VALUES:
            top = sub.head(top_k)
            if top.empty:
                continue
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "top_k": int(top_k),
                    "n_lr": int(len(top)),
                    "n_true_lr": int(top["true_lr"].sum()),
                    "true_lr_fraction": safe_div(top["true_lr"].sum(), len(top)),
                    "n_high_expression_negative": int(top["false_positive_high_expression"].sum()),
                    "high_expression_negative_fraction": safe_div(top["false_positive_high_expression"].sum(), len(top)),
                    "n_zero_expr_lr": int(top["zero_expr_lr"].sum()),
                    "zero_expr_lr_fraction": safe_div(top["zero_expr_lr"].sum(), len(top)),
                    "n_any_negative_or_risk": int(top["negative_or_risk"].sum()),
                    "negative_or_risk_fraction": safe_div(top["negative_or_risk"].sum(), len(top)),
                    "top_lr_pairs": ";".join(top["lr_pair"].astype(str).head(12)),
                }
            )
    return pd.DataFrame(rows)


def summarize_dataset(
    ranking: pd.DataFrame,
    topk: pd.DataFrame,
    hard: pd.DataFrame,
    status: pd.DataFrame,
) -> pd.DataFrame:
    if ranking.empty:
        return pd.DataFrame()
    rows = []
    for dataset_key, sub in ranking.groupby("dataset_key"):
        t10 = topk[(topk["dataset_key"].eq(dataset_key)) & (topk["top_k"].eq(10))].head(1)
        t20 = topk[(topk["dataset_key"].eq(dataset_key)) & (topk["top_k"].eq(20))].head(1)
        t10_row = t10.iloc[0].to_dict() if len(t10) else {}
        t20_row = t20.iloc[0].to_dict() if len(t20) else {}
        hard_sub = hard[hard["dataset_key"].eq(dataset_key)] if not hard.empty else pd.DataFrame()
        status_sub = status[status["dataset_key"].eq(dataset_key)].head(1)
        rows.append(
            {
                "dataset_key": dataset_key,
                "n_lr": int(len(sub)),
                "n_true_lr": int(sub["true_lr"].astype(bool).sum()),
                "n_high_expression_negative": int(sub["false_positive_high_expression"].astype(bool).sum()),
                "top10_true_lr": int(t10_row.get("n_true_lr", 0)),
                "top10_negative_or_risk_fraction": float(t10_row.get("negative_or_risk_fraction", np.nan)),
                "top20_true_lr": int(t20_row.get("n_true_lr", 0)),
                "top20_negative_or_risk_fraction": float(t20_row.get("negative_or_risk_fraction", np.nan)),
                "hard_negative_best_comparison_beats": int(hard_sub["negative_beats_positive"].sum()) if len(hard_sub) else 0,
                "hard_negative_best_comparisons": int(len(hard_sub)),
                "runtime_seconds": float(status_sub["runtime_seconds"].iloc[0]) if len(status_sub) else np.nan,
                "n_candidates": int(status_sub["n_candidates"].iloc[0]) if len(status_sub) and "n_candidates" in status_sub else 0,
                "supplementary_call": classify_dataset(t10_row, hard_sub),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset_key")


def classify_dataset(t10: dict[str, object], hard_sub: pd.DataFrame) -> str:
    top10_true = int(t10.get("n_true_lr", 0))
    top10_risk = float(t10.get("negative_or_risk_fraction", np.nan))
    hard_beat_fraction = safe_div(hard_sub["negative_beats_positive"].sum(), len(hard_sub)) if len(hard_sub) else 0.0
    if top10_true >= 2 and top10_risk <= 0.2 and hard_beat_fraction <= 0.2:
        return "expanded_panel_cross_platform_support"
    if top10_risk >= 0.5 or hard_beat_fraction > 0.5:
        return "expanded_panel_calibration_risk"
    return "expanded_panel_mixed_support"


def write_status(dataset: pd.DataFrame, topk: pd.DataFrame, hard: pd.DataFrame, status: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Supplementary Cross-Platform Expanded-Panel Atlas",
        "",
        "## Scope",
        "",
        "This run extends Phase 12 beyond the strict full-transcriptome atlas by using real-background constructed replicates with 2011-gene expression panels.",
        "It is useful for cross-platform/tissue support, but it is not a replacement for broad pooled whole-atlas FDR closure.",
        "",
        "## Dataset Status",
        "",
    ]
    for row in status.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}`: {row.status}; source={row.source_replicate}; cells={getattr(row, 'n_cells', 'NA')}; "
            f"panel genes={getattr(row, 'n_panel_genes', 'NA')}; LR={getattr(row, 'n_lr_pairs', 'NA')}; "
            f"candidates={getattr(row, 'n_candidates', 'NA')}; runtime={float(row.runtime_seconds):.1f}s"
        )
        if str(getattr(row, "error", "")):
            lines.append(f"  - error: {row.error}")
    if not dataset.empty:
        lines += ["", "## Dataset Summary", ""]
        for row in dataset.itertuples(index=False):
            lines.append(
                f"- `{row.dataset_key}` `{row.supplementary_call}`: LR={int(row.n_lr)}, "
                f"top10 true={int(row.top10_true_lr)}, top10 risk={fmt(row.top10_negative_or_risk_fraction)}, "
                f"hard-negative best-beat={int(row.hard_negative_best_comparison_beats)}/{int(row.hard_negative_best_comparisons)}"
            )
    if not topk.empty:
        lines += ["", "## Top-K", ""]
        for row in topk[topk["top_k"].isin([10, 20])].itertuples(index=False):
            lines.append(
                f"- `{row.dataset_key}` top{int(row.top_k)}: true={int(row.n_true_lr)}, "
                f"high-expression negative={int(row.n_high_expression_negative)}, zero-expression={int(row.n_zero_expr_lr)}, "
                f"risk={fmt(row.negative_or_risk_fraction)}"
            )
    if not hard.empty:
        lines += ["", "## Hard-Negative Leakage", ""]
        grouped = hard.groupby("dataset_key")["negative_beats_positive"].agg(["sum", "count"])
        for dataset_key, row in grouped.iterrows():
            lines.append(f"- `{dataset_key}`: {int(row['sum'])}/{int(row['count'])} hard-negative comparisons beat their positive anchor.")
    lines += [
        "",
        "## Claim Boundary",
        "",
        "- Safe claim: this adds supplementary cross-platform expanded-panel evidence for ranking positive/context LR above high-expression decoys.",
        "- Unsafe claim: this closes broad whole-atlas discovery FDR or proves full-transcriptome generalization for tonsil/melanoma.",
    ]
    (OUT_DIR / "phase12_supplementary_expanded_panel_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(status: pd.DataFrame) -> None:
    manifest = {
        "task": "phase12_supplementary_cross_platform_expanded_panel_atlas",
        "model": "spatialself_wavelet_edge_prior",
        "datasets": status["dataset_key"].tolist() if "dataset_key" in status else [],
        "outputs": [
            "phase12_supplementary_expanded_panel_lr_ranking.csv",
            "phase12_supplementary_expanded_panel_hard_negative_leakage.csv",
            "phase12_supplementary_expanded_panel_status.csv",
            "phase12_supplementary_expanded_panel_lr_manifest.csv",
            "phase12_supplementary_expanded_panel_topk_summary.csv",
            "phase12_supplementary_expanded_panel_dataset_summary.csv",
            "phase12_supplementary_expanded_panel_status.md",
        ],
        "claim_boundary": "Supplementary 2011-gene expanded-panel atlas only; strict full-transcriptome whole-atlas FDR remains separate and partial.",
    }
    (OUT_DIR / "phase12_supplementary_expanded_panel_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if float(den) else float("nan")


def fmt(value: object) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.3f}"
    except Exception:  # noqa: BLE001
        return str(value)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
