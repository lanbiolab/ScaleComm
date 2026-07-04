#!/usr/bin/env python
"""Run a Phase 12 whole-LR atlas pilot through the SpatialSelf wavelet model.

This is intentionally smaller than the final atlas run. It builds benchmark-format
replicates from full-transcriptome backgrounds, keeps multi-scale candidate edges,
adds hard-negative LR pairs, and calls the existing SpatialSelf implementation.
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
from scipy import sparse
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_spatialself import run_spatialself_on_replicate


READINESS_DIR = PROJECT_ROOT / "outputs/phase12/atlas_readiness"
OUT_DIR = PROJECT_ROOT / "outputs/phase12/atlas_model_pilot"
REPLICATE_ROOT = OUT_DIR / "replicates"
PRED_ROOT = OUT_DIR / "predictions"

DATASETS = [
    {
        "dataset_key": "visium_hd_crc_full",
        "label": "Visium HD CRC P2CRC full-transcriptome pilot",
        "cells": "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/visium_hd_crc_p2crc_subset/cells.csv",
        "genes": "outputs/phase10/full_transcriptome_commot_style/visium_hd_crc_p2crc_subset/full_transcriptome/genes.csv",
        "barcodes": "outputs/phase10/full_transcriptome_commot_style/visium_hd_crc_p2crc_subset/full_transcriptome/barcodes.csv",
        "counts": "outputs/phase10/full_transcriptome_commot_style/visium_hd_crc_p2crc_subset/full_transcriptome/expression_full_transcriptome_counts.npz",
        "max_spots": 360,
        "max_lr": 55,
    },
    {
        "dataset_key": "post_xenium_colon_full",
        "label": "Post-Xenium/CytAssist colon cancer full-transcriptome pilot",
        "cells": "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/post_xenium_human_colon_cancer_subset/cells.csv",
        "genes": "outputs/phase10/full_transcriptome_commot_style/post_xenium_human_colon_cancer_subset/full_transcriptome/genes.csv",
        "barcodes": "outputs/phase10/full_transcriptome_commot_style/post_xenium_human_colon_cancer_subset/full_transcriptome/barcodes.csv",
        "counts": "outputs/phase10/full_transcriptome_commot_style/post_xenium_human_colon_cancer_subset/full_transcriptome/expression_full_transcriptome_counts.npz",
        "max_spots": 360,
        "max_lr": 55,
    },
    {
        "dataset_key": "human_lymph_node_visium_raw_full",
        "label": "Human lymph node Visium raw full-transcriptome pilot",
        "cells": "outputs/phase11/commot_style_spatial_story/human_lymph_node_visium_full_subset/dataset/cells.csv",
        "genes": "outputs/phase12/full_transcriptome_inputs/human_lymph_node_visium_raw_full/full_transcriptome/genes.csv",
        "barcodes": "outputs/phase12/full_transcriptome_inputs/human_lymph_node_visium_raw_full/full_transcriptome/barcodes.csv",
        "counts": "outputs/phase12/full_transcriptome_inputs/human_lymph_node_visium_raw_full/full_transcriptome/expression_full_transcriptome_counts.npz",
        "max_spots": 360,
        "max_lr": 55,
    },
]

PRIORITY_RANGE = {
    "chemokine_recruitment": 700.0,
    "immune_contact_checkpoint": 220.0,
    "notch_contact": 160.0,
    "angiogenesis": 520.0,
    "ecm_myeloid_remodeling": 420.0,
    "tgfb_stromal": 420.0,
    "high_expression_negative_risk": 420.0,
    "other": 420.0,
}


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
                epochs=12,
                positive_quantile=0.88,
                negative_quantile=0.25,
                pseudo_label_strategy="lr_prior_lr_capped_mechanism_niche_expression_distance",
                train_sampling_strategy="balanced_per_lr",
                weak_signal_reweighting=2.0,
                weak_positive_fraction=0.2,
                weak_positive_label=0.75,
                weak_positive_mode="non_contact",
                ranking_loss_weight=0.2,
                feasibility_loss_weight=0.15,
                hard_negative_loss_weight=0.12,
                context_contrastive_loss_weight=0.15,
                context_contrastive_margin=0.5,
                context_contrastive_negative_mode="random_receiver",
                lr_identity_contrastive_loss_weight=0.08,
                lr_identity_contrastive_margin=0.35,
                control_contrastive_loss_weight=0.12,
                control_contrastive_margin=0.5,
                null_candidate_loss_weight=0.08,
                lr_cap_positive=45,
                lr_cap_negative=120,
                use_edge_prior_features=True,
                max_train_samples=9000,
                seed=1212,
            )
            pred_path = Path(metrics["prediction_path"])
            ranking = summarize_lr_ranking(replicate_dir, pred_path, lr_pairs, spec["dataset_key"])
            hard = summarize_hard_negative(replicate_dir, ranking, spec["dataset_key"])
            all_rankings.append(ranking)
            all_hard.append(hard)
            all_lr_manifest.append(lr_pairs.assign(dataset_key=spec["dataset_key"]))
            all_status.append(
                {
                    "dataset_key": spec["dataset_key"],
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

    ranking_table.to_csv(OUT_DIR / "phase12_atlas_model_pilot_lr_ranking.csv", index=False)
    hard_table.to_csv(OUT_DIR / "phase12_atlas_model_pilot_hard_negative_leakage.csv", index=False)
    status_table.to_csv(OUT_DIR / "phase12_atlas_model_pilot_status.csv", index=False)
    lr_manifest.to_csv(OUT_DIR / "phase12_atlas_model_pilot_lr_manifest.csv", index=False)
    write_status(ranking_table, hard_table, status_table)
    write_manifest(status_table)

    print(status_table.to_string(index=False))
    if not ranking_table.empty:
        print()
        print(ranking_table.groupby("dataset_key").head(12).to_string(index=False))
    if not hard_table.empty:
        print()
        print(hard_table.groupby("dataset_key").head(12).to_string(index=False))


def build_replicate(
    spec: dict[str, Any],
    lr_universe: pd.DataFrame,
    hard_templates: pd.DataFrame,
) -> tuple[Path, dict[str, Any], pd.DataFrame]:
    cells_raw = pd.read_csv(PROJECT_ROOT / spec["cells"])
    genes = pd.read_csv(PROJECT_ROOT / spec["genes"])["gene"].astype(str).str.upper().tolist()
    barcodes = pd.read_csv(PROJECT_ROOT / spec["barcodes"])["cell_id"].astype(str).tolist()
    counts = sparse.load_npz(PROJECT_ROOT / spec["counts"]).tocsr()
    if counts.shape[0] != len(barcodes):
        raise ValueError(f"Count matrix/barcode mismatch for {spec['dataset_key']}")

    cells = cells_raw[cells_raw["cell_id"].astype(str).isin(barcodes)].copy()
    cells = select_spots(cells, int(spec["max_spots"]))
    cells = normalize_cells(cells)
    row_lookup = {cell_id: idx for idx, cell_id in enumerate(barcodes)}
    row_indices = [row_lookup[cell_id] for cell_id in cells["cell_id"].astype(str)]

    gene_set = set(genes)
    lr_pairs = select_lr_pairs(lr_universe, hard_templates, spec["dataset_key"], gene_set, int(spec["max_lr"]))
    required_genes = sorted(set(lr_pairs["ligand"]) | set(lr_pairs["receptor"]))
    expression = load_expression(cells, counts, row_indices, genes, required_genes)
    candidates = build_candidates(cells, expression, lr_pairs, spec["dataset_key"])

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
                "label": spec["label"],
                "purpose": "Phase 12 SpatialSelf atlas model pilot",
                "candidate_edges": "multi-scale nearest, medium-range and longer-range sender-receiver candidates",
                "claim_boundary": "Pilot-scale model run; not final whole-atlas FDR/downstream validation.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary = {
        "n_cells": int(len(cells)),
        "n_expression_genes": int(len(required_genes)),
        "n_lr_pairs": int(len(lr_pairs)),
        "n_candidates": int(len(candidates)),
        "n_priority_anchor_lr": int(lr_pairs["true_lr"].sum()),
        "n_hard_negative_lr": int(lr_pairs["false_positive_high_expression"].sum()),
    }
    return replicate_dir, summary, lr_pairs


def normalize_cells(cells: pd.DataFrame) -> pd.DataFrame:
    cells = cells.copy()
    for column in ("cell_type", "region", "niche"):
        if column not in cells.columns:
            cells[column] = "unknown"
        cells[column] = cells[column].fillna("unknown").astype(str)
    if "library_size" not in cells.columns:
        if "total_counts" in cells.columns:
            cells["library_size"] = cells["total_counts"]
        else:
            cells["library_size"] = np.nan
    keep = ["cell_id", "x", "y", "cell_type", "region", "niche", "library_size"]
    for optional in ("array_row", "array_col"):
        if optional in cells.columns:
            keep.append(optional)
    return cells[keep].reset_index(drop=True)


def select_spots(cells: pd.DataFrame, max_spots: int) -> pd.DataFrame:
    cells = cells.copy()
    if len(cells) <= max_spots:
        return cells.reset_index(drop=True)
    group_col = "niche" if "niche" in cells.columns else ("cell_type" if "cell_type" in cells.columns else None)
    if group_col is None:
        return cells.sample(max_spots, random_state=1212).reset_index(drop=True)
    parts = []
    for _, group in cells.groupby(group_col, sort=False):
        take = max(1, int(round(max_spots * len(group) / len(cells))))
        parts.append(group.sample(min(take, len(group)), random_state=1212))
    sampled = pd.concat(parts, ignore_index=True)
    if len(sampled) > max_spots:
        sampled = sampled.sample(max_spots, random_state=1213)
    elif len(sampled) < max_spots:
        missing = max_spots - len(sampled)
        remaining = cells[~cells["cell_id"].astype(str).isin(sampled["cell_id"].astype(str))]
        if len(remaining):
            sampled = pd.concat([sampled, remaining.sample(min(missing, len(remaining)), random_state=1214)])
    return sampled.reset_index(drop=True)


def select_lr_pairs(
    lr_universe: pd.DataFrame,
    hard_templates: pd.DataFrame,
    dataset_key: str,
    genes: set[str],
    max_lr: int,
) -> pd.DataFrame:
    covered = lr_universe[lr_universe["ligand"].isin(genes) & lr_universe["receptor"].isin(genes)].copy()
    covered["selected_reason"] = "background"
    selected = []
    priority = covered[covered["phase12_priority_anchor"].astype(bool)].copy()
    priority["selected_reason"] = "phase12_priority_anchor"
    selected.append(priority)
    prior_neg = covered[covered["any_prior_high_expression_negative"].astype(bool)].copy()
    prior_neg["selected_reason"] = "prior_high_expression_negative"
    selected.append(prior_neg)
    templ = hard_templates[hard_templates["dataset_key"].eq(dataset_key)]
    if len(templ):
        template_pairs = templ[["negative_ligand", "negative_receptor"]].drop_duplicates()
        template_pairs["lr_pair"] = template_pairs["negative_ligand"] + "-" + template_pairs["negative_receptor"]
        template_neg = covered[covered["lr_pair"].isin(set(template_pairs["lr_pair"]))].copy()
        template_neg["selected_reason"] = "hard_negative_template"
        selected.append(template_neg)
    category_parts = []
    remaining_budget = max(max_lr - sum(len(x) for x in selected), 0)
    per_category = max(4, remaining_budget // max(int(covered["biological_category"].nunique()), 1))
    for _, group in covered.groupby("biological_category", sort=False):
        part = group.sort_values(["n_sources", "lr_pair"], ascending=[False, True]).head(per_category).copy()
        part["selected_reason"] = "category_balanced"
        category_parts.append(part)
    if category_parts:
        selected.append(pd.concat(category_parts, ignore_index=True))
    fill = covered.sort_values(["n_sources", "lr_pair"], ascending=[False, True]).head(max_lr).copy()
    fill["selected_reason"] = "source_count_fill"
    selected.append(fill)

    table = pd.concat(selected, ignore_index=True)
    table = table.drop_duplicates("lr_pair").head(max_lr).copy()
    rows = []
    for i, row in enumerate(table.itertuples(index=False)):
        category = str(row.biological_category)
        mechanism = mechanism_from_category(category)
        rows.append(
            {
                "lr_id": f"phase12_lr_{i:03d}_{row.ligand}_{row.receptor}",
                "ligand": row.ligand,
                "receptor": row.receptor,
                "mechanism": mechanism,
                "spatial_range": float(PRIORITY_RANGE.get(category, 420.0)),
                "active_niche": "any",
                "control_name": row.lr_pair,
                "control_class": "positive_anchor" if bool(row.phase12_priority_anchor) else str(row.selected_reason),
                "biological_category": category,
                "false_positive_high_expression": bool(row.any_prior_high_expression_negative)
                or category == "high_expression_negative_risk"
                or str(row.selected_reason) == "hard_negative_template",
                "true_lr": bool(row.phase12_priority_anchor),
                "source_count": int(row.n_sources),
                "selected_reason": str(row.selected_reason),
            }
        )
    out = pd.DataFrame(rows)
    if out["false_positive_high_expression"].sum() == 0:
        raise ValueError(f"No hard-negative LR pairs selected for {dataset_key}")
    return out


def mechanism_from_category(category: str) -> str:
    if category in {"immune_contact_checkpoint", "notch_contact"}:
        return "juxtacrine_contact"
    if category == "chemokine_recruitment":
        return "long_range_chemokine_niche"
    return "short_medium_paracrine"


def load_expression(
    cells: pd.DataFrame,
    counts: sparse.csr_matrix,
    row_indices: list[int],
    genes: list[str],
    required_genes: list[str],
) -> pd.DataFrame:
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    cols = [gene_to_idx[gene] for gene in required_genes]
    block = counts[row_indices, :][:, cols].astype(np.float32).toarray()
    expr = pd.DataFrame(block, columns=required_genes)
    expr.insert(0, "cell_id", cells["cell_id"].astype(str).to_numpy())
    return expr


def build_candidates(
    cells: pd.DataFrame,
    expression: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    dataset_key: str,
) -> pd.DataFrame:
    coords = cells[["x", "y"]].to_numpy(dtype=float)
    tree = cKDTree(coords)
    edge_pairs = multiscale_edges(coords, tree, max_edges=600)
    expr_index = expression.set_index("cell_id")
    cell_lookup = cells.set_index("cell_id")
    rows = []
    counter = 0
    for lr in lr_pairs.itertuples(index=False):
        ligand_values = expr_index[lr.ligand].to_numpy(dtype=float)
        receptor_values = expr_index[lr.receptor].to_numpy(dtype=float)
        ligand_q65 = float(np.quantile(ligand_values, 0.65)) if len(ligand_values) else 0.0
        receptor_q65 = float(np.quantile(receptor_values, 0.65)) if len(receptor_values) else 0.0
        for sender_idx, receiver_idx, distance in edge_pairs:
            sender = cells.iloc[int(sender_idx)]
            receiver = cells.iloc[int(receiver_idx)]
            ligand_expr = float(ligand_values[int(sender_idx)])
            receptor_expr = float(receptor_values[int(receiver_idx)])
            type_ok = type_rule(str(lr.control_name), str(sender["cell_type"]), str(receiver["cell_type"]))
            range_ok = float(distance) <= float(lr.spatial_range)
            niche_ok = niche_rule(str(lr.active_niche), str(sender["niche"]), str(receiver["niche"]))
            coexp = np.log1p(ligand_expr) * np.log1p(receptor_expr) / (1.0 + float(distance))
            rows.append(
                {
                    "candidate_id": f"{dataset_key}_atlas_cand_{counter:09d}",
                    "sender_id": sender["cell_id"],
                    "receiver_id": receiver["cell_id"],
                    "lr_id": lr.lr_id,
                    "ligand": lr.ligand,
                    "receptor": lr.receptor,
                    "mechanism": lr.mechanism,
                    "distance": float(distance),
                    "sender_cell_type": sender["cell_type"],
                    "receiver_cell_type": receiver["cell_type"],
                    "sender_niche": sender["niche"],
                    "receiver_niche": receiver["niche"],
                    "weak_signal": bool(ligand_expr <= ligand_q65 or receptor_expr <= receptor_q65),
                    "false_positive_high_expression": bool(lr.false_positive_high_expression),
                    "true_lr": bool(lr.true_lr),
                    "type_rule_satisfied": bool(type_ok),
                    "range_rule_satisfied": bool(range_ok),
                    "niche_rule_satisfied": bool(niche_ok),
                    "ligand_expr": ligand_expr,
                    "receptor_expr": receptor_expr,
                    "coexpression_distance_score": float(coexp),
                }
            )
            counter += 1
    candidates = pd.DataFrame(rows)
    candidates = cap_candidates_per_lr(candidates, max_per_lr=600)
    candidates["candidate_id"] = [f"{dataset_key}_atlas_cand_{i:09d}" for i in range(len(candidates))]
    return candidates


def multiscale_edges(coords: np.ndarray, tree: cKDTree, *, max_edges: int) -> list[tuple[int, int, float]]:
    n = coords.shape[0]
    k_values = [6, 18, 36]
    pairs: dict[tuple[int, int], float] = {}
    for k in k_values:
        kk = min(k + 1, n)
        distances, neighbors = tree.query(coords, k=kk)
        if np.asarray(neighbors).ndim == 1:
            distances = np.asarray(distances)[:, None]
            neighbors = np.asarray(neighbors)[:, None]
        for i in range(n):
            for dist, j in zip(distances[i][1:], neighbors[i][1:]):
                if i == int(j):
                    continue
                pairs[(i, int(j))] = float(dist)
    rng = np.random.default_rng(1212)
    if n > 2:
        random_receivers = rng.integers(0, n, size=(n, 4))
        for i in range(n):
            for j in random_receivers[i]:
                if i == int(j):
                    continue
                dist = float(np.linalg.norm(coords[i] - coords[int(j)]))
                pairs[(i, int(j))] = dist
    table = pd.DataFrame([(i, j, d) for (i, j), d in pairs.items()], columns=["sender_idx", "receiver_idx", "distance"])
    if len(table) <= max_edges:
        return [(int(r.sender_idx), int(r.receiver_idx), float(r.distance)) for r in table.itertuples(index=False)]
    table["distance_bin"] = pd.qcut(table["distance"].rank(method="first"), q=min(5, len(table)), labels=False, duplicates="drop")
    parts = []
    per_bin = max(int(max_edges) // max(int(table["distance_bin"].nunique()), 1), 1)
    for _, group in table.groupby("distance_bin", sort=False):
        parts.append(group.sample(min(per_bin, len(group)), random_state=1212))
    sampled = pd.concat(parts, ignore_index=True)
    if len(sampled) > max_edges:
        sampled = sampled.sample(max_edges, random_state=1213)
    return [(int(r.sender_idx), int(r.receiver_idx), float(r.distance)) for r in sampled.itertuples(index=False)]


def cap_candidates_per_lr(candidates: pd.DataFrame, max_per_lr: int) -> pd.DataFrame:
    rng = np.random.default_rng(1212)
    parts = []
    for _, group in candidates.groupby("lr_id", sort=False):
        if len(group) <= max_per_lr:
            parts.append(group)
            continue
        score = (
            group["type_rule_satisfied"].astype(int) * 2.0
            + group["range_rule_satisfied"].astype(int)
            + group["niche_rule_satisfied"].astype(int)
            + (group["ligand_expr"].gt(0) & group["receptor_expr"].gt(0)).astype(int)
        )
        keep_top = group.assign(_keep_score=score).sort_values("_keep_score", ascending=False).head(max_per_lr // 2)
        remaining = group.drop(index=keep_top.index)
        take = min(max_per_lr - len(keep_top), len(remaining))
        keep_random = remaining.sample(take, random_state=int(rng.integers(1, 1_000_000))) if take else remaining.head(0)
        parts.append(pd.concat([keep_top.drop(columns="_keep_score"), keep_random], ignore_index=True))
    return pd.concat(parts, ignore_index=True)


def type_rule(pair_name: str, sender_type: str, receiver_type: str) -> bool:
    pair = pair_name.upper()
    sender = sender_type.lower()
    receiver = receiver_type.lower()
    if "SPP1-CD44" in pair:
        return any(x in sender for x in ("myeloid", "macrophage", "fibroblast", "caf")) and any(
            x in receiver for x in ("tumor", "epithelial", "fibroblast", "caf", "immune", "t")
        )
    if "CCL19-CCR7" in pair or "CCL21-CCR7" in pair:
        return any(x in receiver for x in ("t", "immune", "lymph"))
    if "CXCL13-CXCR5" in pair:
        return any(x in receiver for x in ("b", "follicle", "lymph"))
    if "CXCL9-CXCR3" in pair or "CXCL10-CXCR3" in pair:
        return any(x in receiver for x in ("t", "immune"))
    if "CD48-CD244" in pair or "CD40LG-CD40" in pair or "PDCD1-CD274" in pair or "CD274-PDCD1" in pair:
        return any(x in sender + receiver for x in ("t", "immune", "myeloid", "tumor"))
    if "VEGFA-KDR" in pair:
        return any(x in receiver for x in ("endo", "vascular"))
    if "TGFB1-TGFBR2" in pair:
        return any(x in sender + receiver for x in ("strom", "fibro", "caf", "tumor"))
    return True


def niche_rule(active_niche: str, sender_niche: str, receiver_niche: str) -> bool:
    if active_niche in {"", "any", "nan"}:
        return True
    return active_niche == sender_niche or active_niche == receiver_niche


def summarize_lr_ranking(replicate_dir: Path, pred_path: Path, lr_pairs: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
    pred = pd.read_csv(pred_path)
    table = candidates.merge(pred[["candidate_id", "score"]], on="candidate_id", how="inner")
    summary = (
        table.groupby("lr_id", as_index=False)
        .agg(
            mean_score=("score", "mean"),
            q95_score=("score", lambda x: float(np.quantile(x, 0.95))),
            max_score=("score", "max"),
            n_candidates=("score", "size"),
            n_nonzero_expr=("coexpression_distance_score", lambda x: int((x > 0).sum())),
            rule_satisfied_fraction=("type_rule_satisfied", "mean"),
            range_satisfied_fraction=("range_rule_satisfied", "mean"),
        )
        .merge(lr_pairs, on="lr_id", how="left")
    )
    summary["lr_pair"] = summary["ligand"] + "-" + summary["receptor"]
    summary["dataset_key"] = dataset_key
    summary["method"] = "spatialself_wavelet_edge_prior"
    summary["rank"] = summary["mean_score"].rank(ascending=False, method="min").astype(int)
    summary["rank_q95"] = summary["q95_score"].rank(ascending=False, method="min").astype(int)
    columns = [
        "dataset_key",
        "method",
        "rank",
        "rank_q95",
        "lr_pair",
        "lr_id",
        "ligand",
        "receptor",
        "biological_category",
        "mechanism",
        "selected_reason",
        "true_lr",
        "false_positive_high_expression",
        "mean_score",
        "q95_score",
        "max_score",
        "n_candidates",
        "n_nonzero_expr",
        "rule_satisfied_fraction",
        "range_satisfied_fraction",
    ]
    return summary[columns].sort_values(["dataset_key", "rank", "lr_pair"]).reset_index(drop=True)


def summarize_hard_negative(replicate_dir: Path, ranking: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    rank_lookup = ranking.set_index("lr_pair").to_dict("index")
    positives = ranking[ranking["true_lr"].astype(bool)]
    negatives = ranking[ranking["false_positive_high_expression"].astype(bool)]
    rows = []
    for pos in positives.itertuples(index=False):
        receptor_matched = negatives[negatives["receptor"].eq(pos.receptor)]
        ligand_matched = negatives[negatives["ligand"].eq(pos.ligand)]
        high_expr = negatives
        for neg_type, pool in [
            ("same_receptor_or_receptor_shared", receptor_matched),
            ("same_ligand", ligand_matched),
            ("any_high_expression_negative", high_expr),
        ]:
            if pool.empty:
                continue
            neg = pool.sort_values(["rank", "mean_score"], ascending=[True, False]).iloc[0]
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "positive_lr_pair": pos.lr_pair,
                    "negative_pair": neg["lr_pair"],
                    "negative_type": neg_type,
                    "positive_rank": int(pos.rank),
                    "negative_rank": int(neg["rank"]),
                    "positive_score": float(pos.mean_score),
                    "negative_score": float(neg["mean_score"]),
                    "negative_minus_positive_score": float(neg["mean_score"] - pos.mean_score),
                    "negative_beats_positive": bool(int(neg["rank"]) < int(pos.rank)),
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values(["negative_beats_positive", "negative_minus_positive_score"], ascending=[False, False])


def write_status(ranking: pd.DataFrame, hard: pd.DataFrame, status: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Atlas Model Pilot Status",
        "",
        "## Scope",
        "",
        "This run connects Phase 12 whole-LR atlas inputs to the existing SpatialSelf wavelet model. It is a pilot-scale formal model run, not the final 957-LR whole-atlas FDR/downstream analysis.",
        "",
        "## Dataset Status",
        "",
    ]
    for row in status.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}`: {row.status}; cells={getattr(row, 'n_cells', 'NA')}; LR={getattr(row, 'n_lr_pairs', 'NA')}; candidates={getattr(row, 'n_candidates', 'NA')}; runtime={float(row.runtime_seconds):.1f}s"
        )
        if str(getattr(row, "error", "")):
            lines.append(f"  - error: {row.error}")
    if not ranking.empty:
        lines += ["", "## Top SpatialSelf LR", ""]
        for dataset_key, sub in ranking.groupby("dataset_key"):
            top = sub.sort_values("rank").head(8)
            top_text = "; ".join(
                f"{r.lr_pair}#{int(r.rank)}{' NEG' if bool(r.false_positive_high_expression) else ''}"
                for r in top.itertuples(index=False)
            )
            lines.append(f"- `{dataset_key}`: {top_text}")
    if not hard.empty:
        lines += ["", "## Hard-Negative Leakage", ""]
        grouped = hard.groupby("dataset_key")["negative_beats_positive"].agg(["sum", "count"])
        for dataset_key, row in grouped.iterrows():
            lines.append(f"- `{dataset_key}`: {int(row['sum'])}/{int(row['count'])} hard-negative comparisons beat their positive anchor.")
    lines += [
        "",
        "## Claim Boundary",
        "",
        "This supports atlas-pipeline readiness for the real SpatialSelf model and exposes hard-negative behavior at pilot scale. It does not yet prove whole-atlas generalization, empirical FDR control, or whole-transcriptome downstream validity.",
    ]
    (OUT_DIR / "phase12_atlas_model_pilot_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(status: pd.DataFrame) -> None:
    manifest = {
        "task": "phase12_atlas_model_pilot",
        "model": "spatialself_wavelet_edge_prior",
        "datasets": status["dataset_key"].tolist() if "dataset_key" in status else [],
        "outputs": [
            "phase12_atlas_model_pilot_lr_ranking.csv",
            "phase12_atlas_model_pilot_hard_negative_leakage.csv",
            "phase12_atlas_model_pilot_lr_manifest.csv",
            "phase12_atlas_model_pilot_status.csv",
            "phase12_atlas_model_pilot_status.md",
        ],
        "claim_boundary": "Pilot-scale model run only; final Phase 12 still needs full-LR FDR, whole-transcriptome downstream validation, second strong biological story, directionality metrics and manuscript figures.",
    }
    (OUT_DIR / "phase12_atlas_model_pilot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
