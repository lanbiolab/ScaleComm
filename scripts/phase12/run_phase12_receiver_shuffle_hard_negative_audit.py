#!/usr/bin/env python
"""Receiver-shuffle hard-negative audit for Phase 12 expanded atlas edges.

This is a fixed-score diagnostic null. It keeps sender cells, LR identities and
model edge scores fixed, then shuffles receiver cells and recomputes receiver
expression/context guardrails. It asks whether high-scoring LR edges still look
plausible after the receiver side of the communication edge is broken.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase12.run_phase12_atlas_model_pilot import niche_rule, type_rule


ATLAS_DIR = PROJECT_ROOT / "outputs/phase12/atlas_model_expanded"
OUT_DIR = PROJECT_ROOT / "outputs/phase12/receiver_shuffle_hard_negative"
N_PERMUTATIONS = 200
TOP_FRACTION = 0.10
SEED = 121207

DATASETS = {
    "post_xenium_colon_full": {
        "replicate_dir": ATLAS_DIR / "replicates/post_xenium_colon_full",
        "prediction": ATLAS_DIR
        / "predictions/post_xenium_colon_full/spatialself_wavelet_edge_prior/"
        "post_xenium_colon_full__spatialself_wavelet_predictions.csv",
    },
    "human_lymph_node_visium_raw_full": {
        "replicate_dir": ATLAS_DIR / "replicates/human_lymph_node_visium_raw_full",
        "prediction": ATLAS_DIR
        / "predictions/human_lymph_node_visium_raw_full/spatialself_wavelet_edge_prior/"
        "human_lymph_node_visium_raw_full__spatialself_wavelet_predictions.csv",
    },
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_lr_rows = []
    all_dataset_rows = []
    for dataset_key, spec in DATASETS.items():
        lr_rows, dataset_rows = audit_dataset(dataset_key, Path(spec["replicate_dir"]), Path(spec["prediction"]))
        all_lr_rows.append(lr_rows)
        all_dataset_rows.append(dataset_rows)

    lr_table = pd.concat(all_lr_rows, ignore_index=True) if all_lr_rows else pd.DataFrame()
    dataset_summary = pd.concat(all_dataset_rows, ignore_index=True) if all_dataset_rows else pd.DataFrame()
    if not lr_table.empty:
        lr_table["receiver_shuffle_bh_fdr"] = bh_adjust_by_group(
            lr_table, group_col="dataset_key", p_col="receiver_shuffle_empirical_p"
        )
        dataset_summary = summarize_datasets(lr_table)

    lr_table.to_csv(OUT_DIR / "phase12_receiver_shuffle_lr_summary.csv", index=False)
    dataset_summary.to_csv(OUT_DIR / "phase12_receiver_shuffle_dataset_summary.csv", index=False)
    write_status(dataset_summary, lr_table)
    write_manifest(lr_table, dataset_summary)
    print(dataset_summary.to_string(index=False))
    print()
    print(lr_table.sort_values(["dataset_key", "observed_rank"]).to_string(index=False))


def audit_dataset(dataset_key: str, replicate_dir: Path, prediction_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = pd.read_csv(replicate_dir / "cells.csv")
    expression = pd.read_csv(replicate_dir / "expression.csv").set_index("cell_id")
    lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
    candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
    prediction = pd.read_csv(prediction_path)
    ranking = pd.read_csv(ATLAS_DIR / "phase12_atlas_model_expanded_calibrated_lr_ranking.csv")
    ranking = ranking[ranking["dataset_key"].eq(dataset_key)].copy()

    table = candidates.merge(prediction[["candidate_id", "score"]], on="candidate_id", how="inner", validate="one_to_one")
    table["score"] = pd.to_numeric(table["score"], errors="coerce").fillna(0.0)
    table["receiver_guardrail_score"] = table["score"] * receiver_guardrail_weight(table)

    cell_lookup = cells.set_index("cell_id")
    coords = cell_lookup[["x", "y"]].to_numpy(dtype=float)
    cell_ids = cell_lookup.index.astype(str).to_numpy()
    id_to_pos = {cell_id: i for i, cell_id in enumerate(cell_ids)}
    expr_index = expression.reindex(cell_ids)
    true_lr = lr_pairs[lr_pairs.get("true_lr", pd.Series(False, index=lr_pairs.index)).fillna(False).astype(bool)]
    rank_lookup = ranking.set_index("lr_id").to_dict("index") if not ranking.empty else {}

    rng = np.random.default_rng(SEED + stable_int(dataset_key))
    rows = []
    for lr in true_lr.itertuples(index=False):
        lr_sub = table[table["lr_id"].astype(str).eq(str(lr.lr_id))].copy()
        if lr_sub.empty:
            continue
        observed_positive = lr_sub[
            lr_sub["type_rule_satisfied"].fillna(False).astype(bool)
            & lr_sub["range_rule_satisfied"].fillna(False).astype(bool)
            & lr_sub["niche_rule_satisfied"].fillna(False).astype(bool)
            & (pd.to_numeric(lr_sub["ligand_expr"], errors="coerce").fillna(0) > 0)
            & (pd.to_numeric(lr_sub["receptor_expr"], errors="coerce").fillna(0) > 0)
        ].copy()
        if observed_positive.empty:
            observed_positive = lr_sub[lr_sub["true_lr"].fillna(False).astype(bool)].copy()

        observed_stat = edge_stat(observed_positive, "receiver_guardrail_score")
        observed_top_context = top_context_fraction(observed_positive, "receiver_guardrail_score")
        null_stats = []
        null_contexts = []
        null_expr_fracs = []
        null_rule_fracs = []
        for _ in range(N_PERMUTATIONS):
            shuffled = shuffled_receiver_table(lr_sub, lr, cell_lookup, expr_index, coords, cell_ids, id_to_pos, rng)
            shuffled["receiver_shuffle_guardrail_score"] = shuffled["score"] * receiver_guardrail_weight(shuffled)
            null_stats.append(edge_stat(shuffled, "receiver_shuffle_guardrail_score")["top_tail_mean"])
            null_contexts.append(top_context_fraction(shuffled, "receiver_shuffle_guardrail_score"))
            null_expr_fracs.append(float(((shuffled["ligand_expr"] > 0) & (shuffled["receptor_expr"] > 0)).mean()))
            null_rule_fracs.append(float((shuffled["type_rule_satisfied"] & shuffled["range_rule_satisfied"] & shuffled["niche_rule_satisfied"]).mean()))

        null = np.asarray(null_stats, dtype=float)
        row_rank = rank_lookup.get(str(lr.lr_id), {})
        rows.append(
            {
                "dataset_key": dataset_key,
                "lr_id": str(lr.lr_id),
                "lr_pair": f"{lr.ligand}-{lr.receptor}",
                "biological_category": getattr(lr, "biological_category", ""),
                "mechanism": getattr(lr, "mechanism", ""),
                "observed_rank": to_int(row_rank.get("rank"), -1),
                "observed_calibrated_score": to_float(row_rank.get("calibrated_score")),
                "observed_n_edges": int(observed_stat["n_edges"]),
                "observed_top_tail_mean": float(observed_stat["top_tail_mean"]),
                "observed_mean_score": float(observed_stat["mean_score"]),
                "observed_top_context_fraction": observed_top_context,
                "receiver_shuffle_null_mean": float(np.nanmean(null)),
                "receiver_shuffle_null_q95": float(np.nanquantile(null, 0.95)),
                "receiver_shuffle_null_max": float(np.nanmax(null)),
                "receiver_shuffle_empirical_p": empirical_p(null, observed_stat["top_tail_mean"]),
                "receiver_shuffle_delta_top_tail": float(observed_stat["top_tail_mean"] - np.nanmean(null)),
                "receiver_shuffle_ratio_top_tail": safe_ratio(observed_stat["top_tail_mean"], np.nanmean(null)),
                "receiver_shuffle_mean_top_context_fraction": float(np.nanmean(null_contexts)),
                "receiver_shuffle_mean_nonzero_expr_fraction": float(np.nanmean(null_expr_fracs)),
                "receiver_shuffle_mean_rule_fraction": float(np.nanmean(null_rule_fracs)),
                "receiver_shuffle_leakage": bool(np.nanquantile(null, 0.95) >= observed_stat["top_tail_mean"]),
                "n_permutations": N_PERMUTATIONS,
            }
        )
    lr_table = pd.DataFrame(rows)
    return lr_table, pd.DataFrame()


def shuffled_receiver_table(
    lr_sub: pd.DataFrame,
    lr: Any,
    cell_lookup: pd.DataFrame,
    expr_index: pd.DataFrame,
    coords: np.ndarray,
    cell_ids: np.ndarray,
    id_to_pos: dict[str, int],
    rng: np.random.Generator,
) -> pd.DataFrame:
    shuffled = lr_sub.copy()
    n = len(shuffled)
    receiver_pos = rng.integers(0, len(cell_ids), size=n)
    shuffled_receiver_ids = cell_ids[receiver_pos]
    sender_pos = np.asarray([id_to_pos[str(x)] for x in shuffled["sender_id"].astype(str)], dtype=int)

    shuffled["receiver_id"] = shuffled_receiver_ids
    shuffled["receiver_cell_type"] = cell_lookup.loc[shuffled_receiver_ids, "cell_type"].astype(str).to_numpy()
    shuffled["receiver_niche"] = cell_lookup.loc[shuffled_receiver_ids, "niche"].astype(str).to_numpy()
    distance = np.linalg.norm(coords[sender_pos] - coords[receiver_pos], axis=1)
    shuffled["distance"] = distance
    receptor_expr = expr_index.loc[shuffled_receiver_ids, str(lr.receptor)].to_numpy(dtype=float)
    shuffled["receptor_expr"] = receptor_expr
    shuffled["type_rule_satisfied"] = [
        type_rule(str(lr.control_name), str(sender_type), str(receiver_type))
        for sender_type, receiver_type in zip(shuffled["sender_cell_type"], shuffled["receiver_cell_type"])
    ]
    shuffled["range_rule_satisfied"] = distance <= float(lr.spatial_range)
    shuffled["niche_rule_satisfied"] = [
        niche_rule(str(lr.active_niche), str(sender_niche), str(receiver_niche))
        for sender_niche, receiver_niche in zip(shuffled["sender_niche"], shuffled["receiver_niche"])
    ]
    shuffled["coexpression_distance_score"] = (
        np.log1p(pd.to_numeric(shuffled["ligand_expr"], errors="coerce").fillna(0).to_numpy(dtype=float))
        * np.log1p(receptor_expr)
        / (1.0 + distance)
    )
    return shuffled


def receiver_guardrail_weight(table: pd.DataFrame) -> np.ndarray:
    ligand = pd.to_numeric(table.get("ligand_expr", pd.Series(0, index=table.index)), errors="coerce").fillna(0).to_numpy()
    receptor = pd.to_numeric(table.get("receptor_expr", pd.Series(0, index=table.index)), errors="coerce").fillna(0).to_numpy()
    both_present = (ligand > 0) & (receptor > 0)
    one_missing = (ligand > 0) ^ (receptor > 0)
    weight = np.where(both_present, 1.0, np.where(one_missing, 0.12, 0.03))
    for column, penalty in [
        ("type_rule_satisfied", 0.35),
        ("range_rule_satisfied", 0.20),
        ("niche_rule_satisfied", 0.30),
    ]:
        if column in table:
            weight *= np.where(table[column].fillna(False).astype(bool).to_numpy(), 1.0, penalty)
    return np.clip(weight, 0.0, 1.0)


def edge_stat(table: pd.DataFrame, score_column: str) -> dict[str, float]:
    if table.empty:
        return {"n_edges": 0, "mean_score": np.nan, "top_tail_mean": np.nan}
    score = pd.to_numeric(table[score_column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    n_top = max(int(np.ceil(len(score) * TOP_FRACTION)), 1)
    return {
        "n_edges": int(len(table)),
        "mean_score": float(score.mean()),
        "top_tail_mean": float(np.sort(score)[-n_top:].mean()),
    }


def top_context_fraction(table: pd.DataFrame, score_column: str) -> float:
    if table.empty:
        return np.nan
    score = pd.to_numeric(table[score_column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    n_top = max(int(np.ceil(len(score) * TOP_FRACTION)), 1)
    top = table.iloc[np.argsort(score)[-n_top:]].copy()
    ok = (
        top["type_rule_satisfied"].fillna(False).astype(bool)
        & top["range_rule_satisfied"].fillna(False).astype(bool)
        & top["niche_rule_satisfied"].fillna(False).astype(bool)
        & (pd.to_numeric(top["ligand_expr"], errors="coerce").fillna(0) > 0)
        & (pd.to_numeric(top["receptor_expr"], errors="coerce").fillna(0) > 0)
    )
    return float(ok.mean())


def summarize_datasets(lr_table: pd.DataFrame) -> pd.DataFrame:
    if lr_table.empty:
        return pd.DataFrame()
    rows = []
    for dataset_key, sub in lr_table.groupby("dataset_key", sort=False):
        fdr = sub["receiver_shuffle_bh_fdr"] if "receiver_shuffle_bh_fdr" in sub else pd.Series(np.nan, index=sub.index)
        rows.append(
            {
                "dataset_key": dataset_key,
                "n_true_lr_tested": int(len(sub)),
                "n_receiver_shuffle_fdr_0_1": int(pd.to_numeric(fdr, errors="coerce").fillna(1.0).le(0.1).sum()),
                "n_receiver_shuffle_leakage": int(sub["receiver_shuffle_leakage"].fillna(False).sum()),
                "receiver_shuffle_leakage_fraction": safe_ratio(
                    int(sub["receiver_shuffle_leakage"].fillna(False).sum()), len(sub)
                ),
                "median_receiver_shuffle_delta_top_tail": float(sub["receiver_shuffle_delta_top_tail"].median()),
                "median_receiver_shuffle_ratio_top_tail": float(sub["receiver_shuffle_ratio_top_tail"].median()),
                "best_supported_lr": best_supported_lr(sub),
                "hardest_leakage_lr": hardest_leakage_lr(sub),
                "interpretation": interpret_dataset(sub, fdr),
            }
        )
    return pd.DataFrame(rows)


def interpret_dataset(sub: pd.DataFrame, fdr: pd.Series) -> str:
    sig = int(pd.to_numeric(fdr, errors="coerce").fillna(1.0).le(0.1).sum())
    leak_frac = float(sub["receiver_shuffle_leakage"].fillna(False).mean()) if len(sub) else 1.0
    if sig >= max(1, len(sub) // 3) and leak_frac <= 0.25:
        return "receiver-context dependence is strong for multiple anchors"
    if sig > 0:
        return "some anchors show receiver-context dependence; leakage remains"
    return "receiver shuffle does not yet provide significant support"


def best_supported_lr(sub: pd.DataFrame) -> str:
    if sub.empty:
        return ""
    table = sub.copy()
    if "receiver_shuffle_bh_fdr" in table:
        table["_fdr"] = pd.to_numeric(table["receiver_shuffle_bh_fdr"], errors="coerce").fillna(1.0)
    else:
        table["_fdr"] = pd.to_numeric(table["receiver_shuffle_empirical_p"], errors="coerce").fillna(1.0)
    row = table.sort_values(["_fdr", "receiver_shuffle_delta_top_tail"], ascending=[True, False]).iloc[0]
    return f"{row.lr_pair} (q={row._fdr:.3g}, delta={row.receiver_shuffle_delta_top_tail:.3g})"


def hardest_leakage_lr(sub: pd.DataFrame) -> str:
    leaking = sub[sub["receiver_shuffle_leakage"].fillna(False)].copy()
    if leaking.empty:
        return ""
    row = leaking.sort_values("receiver_shuffle_null_q95", ascending=False).iloc[0]
    return f"{row.lr_pair} (nullQ95={row.receiver_shuffle_null_q95:.3g})"


def empirical_p(null_stats: np.ndarray, observed: float) -> float:
    null_stats = np.asarray(null_stats, dtype=float)
    null_stats = null_stats[np.isfinite(null_stats)]
    if len(null_stats) == 0 or not np.isfinite(observed):
        return np.nan
    return float((1 + np.sum(null_stats >= observed)) / (1 + len(null_stats)))


def bh_adjust_by_group(table: pd.DataFrame, *, group_col: str, p_col: str) -> np.ndarray:
    out = np.full(len(table), np.nan, dtype=float)
    for _, idx in table.groupby(group_col).groups.items():
        idx = np.asarray(list(idx), dtype=int)
        out[idx] = bh_adjust(table.iloc[idx][p_col].to_numpy(dtype=float))
    return out


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    out = np.full(len(pvalues), np.nan, dtype=float)
    finite = np.isfinite(pvalues)
    if not finite.any():
        return out
    idx = np.flatnonzero(finite)
    p = pvalues[idx]
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(ranked) / (np.arange(len(ranked)) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out[idx[order]] = np.minimum(adjusted, 1.0)
    return out


def safe_ratio(num: float, den: float) -> float:
    return float(num / den) if np.isfinite(den) and den != 0 else np.nan


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def to_int(value: Any, default: int) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def stable_int(text: str) -> int:
    return int(sum((i + 1) * ord(ch) for i, ch in enumerate(text)) % 100_000)


def write_status(dataset_summary: pd.DataFrame, lr_table: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Receiver-Shuffle Hard-Negative Audit",
        "",
        "## Scope",
        "",
        "This fixed-score diagnostic keeps sender cells, LR identities and SpatialSelf edge scores fixed, but randomly reassigns receiver cells. Receiver expression, distance, cell-type rule and niche rule are recomputed after the shuffle. It tests whether a candidate LR depends on the correct receiver context rather than only high expression/background spatial hotspots.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset_summary.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}`: tested={int(row.n_true_lr_tested)}, receiver-shuffle q<=0.1={int(row.n_receiver_shuffle_fdr_0_1)}, "
            f"leakage={int(row.n_receiver_shuffle_leakage)}/{int(row.n_true_lr_tested)}, median delta={float(row.median_receiver_shuffle_delta_top_tail):.3g}; {row.interpretation}"
        )
        lines.append(f"  - best supported: {row.best_supported_lr}")
        if str(row.hardest_leakage_lr):
            lines.append(f"  - hardest leakage: {row.hardest_leakage_lr}")
    if not lr_table.empty:
        lines += ["", "## Top LR Details", ""]
        cols = ["dataset_key", "observed_rank", "lr_pair", "receiver_shuffle_empirical_p", "receiver_shuffle_bh_fdr", "receiver_shuffle_delta_top_tail", "receiver_shuffle_leakage"]
        detail = lr_table[cols].sort_values(["dataset_key", "observed_rank"]).head(30)
        for row in detail.itertuples(index=False):
            lines.append(
                f"- `{row.dataset_key}` {row.lr_pair} rank={int(row.observed_rank)}, p={float(row.receiver_shuffle_empirical_p):.3g}, "
                f"q={float(row.receiver_shuffle_bh_fdr):.3g}, delta={float(row.receiver_shuffle_delta_top_tail):.3g}, leakage={bool(row.receiver_shuffle_leakage)}"
            )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "This is a fixed-score receiver-context stress test. It is stronger than a simple expression control for the highlighted edges, but it is not a model-rerun null and does not establish global whole-atlas FDR.",
    ]
    (OUT_DIR / "phase12_receiver_shuffle_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(lr_table: pd.DataFrame, dataset_summary: pd.DataFrame) -> None:
    manifest = {
        "task": "phase12_receiver_shuffle_hard_negative",
        "datasets": sorted(dataset_summary["dataset_key"].dropna().astype(str).unique().tolist()) if not dataset_summary.empty else [],
        "n_permutations": N_PERMUTATIONS,
        "top_fraction": TOP_FRACTION,
        "n_lr_rows": int(len(lr_table)),
        "outputs": {
            "lr_summary": rel(OUT_DIR / "phase12_receiver_shuffle_lr_summary.csv"),
            "dataset_summary": rel(OUT_DIR / "phase12_receiver_shuffle_dataset_summary.csv"),
            "status": rel(OUT_DIR / "phase12_receiver_shuffle_status.md"),
        },
        "claim_boundary": "Fixed-score receiver shuffle stress test; not model-rerun global FDR.",
    }
    (OUT_DIR / "phase12_receiver_shuffle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
