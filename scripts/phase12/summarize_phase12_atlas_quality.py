#!/usr/bin/env python
"""Summarize Phase 12 atlas pilot quality, contamination and communication modules."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
PILOT_DIR = PHASE12_DIR / "atlas_model_pilot"
OUT_DIR = PILOT_DIR

RECEPTOR_SHARED_RISK = {"CCR7", "CXCR4", "CXCR3", "CD44", "CD274", "CD40", "ITGB1", "EGFR"}
TOP_KS = [5, 10, 20]


def main() -> None:
    ranking = pd.read_csv(PILOT_DIR / "phase12_atlas_model_pilot_lr_ranking.csv")
    quality = summarize_quality(ranking)
    modules = build_modules(ranking)
    quality.to_csv(OUT_DIR / "phase12_atlas_model_pilot_quality.csv", index=False)
    modules.to_csv(OUT_DIR / "phase12_atlas_model_pilot_modules.csv", index=False)
    write_status(quality, modules)
    write_manifest()
    print(quality.to_string(index=False))
    print()
    print(modules.to_string(index=False))


def summarize_quality(ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in ranking.groupby("dataset_key"):
        sub = sub.sort_values("rank").copy()
        sub["receptor_shared_risk"] = sub["receptor"].astype(str).isin(RECEPTOR_SHARED_RISK)
        for top_k in TOP_KS:
            top = sub.head(top_k)
            categories = top["biological_category"].value_counts()
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "top_k": int(top_k),
                    "n_lr": int(len(top)),
                    "n_priority_anchor": int(top["true_lr"].astype(bool).sum()),
                    "priority_anchor_fraction": safe_divide(int(top["true_lr"].astype(bool).sum()), len(top)),
                    "n_high_expression_negative": int(top["false_positive_high_expression"].astype(bool).sum()),
                    "high_expression_negative_fraction": safe_divide(
                        int(top["false_positive_high_expression"].astype(bool).sum()), len(top)
                    ),
                    "n_receptor_shared_risk": int(top["receptor_shared_risk"].sum()),
                    "receptor_shared_risk_fraction": safe_divide(int(top["receptor_shared_risk"].sum()), len(top)),
                    "top_biological_category": str(categories.index[0]) if len(categories) else "",
                    "top_biological_category_count": int(categories.iloc[0]) if len(categories) else 0,
                    "top_lr_pairs": ";".join(top["lr_pair"].astype(str).tolist()),
                    "top_negative_pairs": ";".join(
                        top[top["false_positive_high_expression"].astype(bool)]["lr_pair"].astype(str).tolist()
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_modules(ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in ranking.groupby("dataset_key"):
        matrix, lr_ids = incoming_matrix(dataset_key)
        if matrix.size == 0 or len(lr_ids) < 5:
            continue
        lr_meta = sub.set_index("lr_id")
        valid = np.nanstd(matrix, axis=1) > 1e-10
        matrix = matrix[valid]
        lr_ids = [lr_id for lr_id, keep in zip(lr_ids, valid) if keep]
        if len(lr_ids) < 5:
            continue
        matrix = matrix - matrix.mean(axis=1, keepdims=True)
        norms = np.sqrt(np.sum(matrix**2, axis=1, keepdims=True))
        matrix = matrix / np.maximum(norms, 1e-8)
        n_modules = min(6, max(2, len(lr_ids) // 12))
        clusters = fcluster(linkage(pdist(matrix, metric="cosine"), method="average"), t=n_modules, criterion="maxclust")
        assign = pd.DataFrame({"lr_id": lr_ids, "module": clusters}).merge(
            sub,
            on="lr_id",
            how="left",
        )
        for module, mod in assign.groupby("module"):
            mod = mod.sort_values("rank")
            categories = mod["biological_category"].value_counts()
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "module": f"M{int(module)}",
                    "n_lr": int(len(mod)),
                    "mean_rank": float(mod["rank"].mean()),
                    "best_rank": int(mod["rank"].min()),
                    "n_priority_anchor": int(mod["true_lr"].astype(bool).sum()),
                    "n_high_expression_negative": int(mod["false_positive_high_expression"].astype(bool).sum()),
                    "top_biological_category": str(categories.index[0]) if len(categories) else "",
                    "top_lr_pairs": ";".join(mod["lr_pair"].astype(str).head(8).tolist()),
                    "module_warning": module_warning(mod),
                }
            )
    return pd.DataFrame(rows).sort_values(["dataset_key", "best_rank", "module"]) if rows else pd.DataFrame()


def incoming_matrix(dataset_key: str) -> tuple[np.ndarray, list[str]]:
    replicate = PILOT_DIR / "replicates" / dataset_key
    pred_path = (
        PILOT_DIR
        / "predictions"
        / dataset_key
        / "spatialself_wavelet_edge_prior"
        / f"{dataset_key}__spatialself_wavelet_predictions.csv"
    )
    if not replicate.exists() or not pred_path.exists():
        return np.empty((0, 0)), []
    cells = pd.read_csv(replicate / "cells.csv")
    candidates = pd.read_csv(replicate / "candidate_interactions.csv", usecols=["candidate_id", "receiver_id", "lr_id"])
    pred = pd.read_csv(pred_path)
    table = candidates.merge(pred[["candidate_id", "score"]], on="candidate_id", how="inner")
    receivers = cells["cell_id"].astype(str).tolist()
    receiver_index = {cell_id: i for i, cell_id in enumerate(receivers)}
    lr_ids = sorted(table["lr_id"].astype(str).unique())
    matrix = np.zeros((len(lr_ids), len(receivers)), dtype=np.float32)
    for i, lr_id in enumerate(lr_ids):
        sub = table[table["lr_id"].astype(str).eq(lr_id)]
        agg = sub.groupby("receiver_id")["score"].sum()
        for receiver_id, value in agg.items():
            j = receiver_index.get(str(receiver_id))
            if j is not None:
                matrix[i, j] = float(value)
    return matrix, lr_ids


def module_warning(mod: pd.DataFrame) -> str:
    if int(mod["false_positive_high_expression"].astype(bool).sum()) >= max(2, len(mod) // 3):
        return "negative-control-heavy module"
    if int(mod["true_lr"].astype(bool).sum()) >= max(2, len(mod) // 4):
        return "priority-anchor-enriched module"
    return ""


def write_status(quality: pd.DataFrame, modules: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Atlas Quality Status",
        "",
        "## Top-k Contamination",
        "",
    ]
    for row in quality.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` top{int(row.top_k)}: priority={int(row.n_priority_anchor)}, high-expression negatives={int(row.n_high_expression_negative)}, receptor-risk={int(row.n_receptor_shared_risk)}, top category={row.top_biological_category}"
        )
    if not modules.empty:
        lines += ["", "## Communication Modules", ""]
        for row in modules.itertuples(index=False):
            warning = f"; {row.module_warning}" if str(row.module_warning) else ""
            lines.append(
                f"- `{row.dataset_key}` {row.module}: n={int(row.n_lr)}, best rank={int(row.best_rank)}, anchors={int(row.n_priority_anchor)}, negatives={int(row.n_high_expression_negative)}, category={row.top_biological_category}{warning}"
            )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "This summarizes the Phase 12 pilot model ranking and receiver-profile modules. It is not final whole-atlas validation.",
    ]
    (OUT_DIR / "phase12_atlas_model_pilot_quality_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "task": "phase12_atlas_model_pilot_quality",
        "outputs": [
            "phase12_atlas_model_pilot_quality.csv",
            "phase12_atlas_model_pilot_modules.csv",
            "phase12_atlas_model_pilot_quality_status.md",
        ],
        "claim_boundary": "Pilot top-k contamination and module summary, not final whole-atlas validation.",
    }
    (OUT_DIR / "phase12_atlas_model_pilot_quality_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def safe_divide(num: int | float, den: int | float) -> float:
    return float(num / den) if den else np.nan


if __name__ == "__main__":
    main()
