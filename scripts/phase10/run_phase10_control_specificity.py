#!/usr/bin/env python
"""Additional control-specificity checks for Phase 10 SPP1-CD44 validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import fisher_exact, hypergeom, rankdata, t as student_t
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOAL5_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"

CURATED_TARGETS = {
    "SPP1-CD44": [
        "LYZ",
        "CD68",
        "CD14",
        "C1QA",
        "C1QB",
        "C1QC",
        "LST1",
        "FCGR3A",
        "MSR1",
        "MRC1",
        "COL1A1",
        "COL1A2",
        "FN1",
        "LGALS3",
        "CTSS",
        "APOE",
        "TREM2",
    ],
}


DATASETS = {
    "visium_hd_crc": {
        "label": "Visium HD CRC P2CRC 8 um bin subset",
        "slug": "visium_hd_crc_p2crc_subset",
        "standardized": "visium_hd_crc_p2crc_subset",
        "focus_pair": "SPP1-CD44",
        "focus_lr_id": "visium_hd_crc_lr_000_SPP1_CD44",
        "focus_method": "spatialself",
    },
    "post_xenium": {
        "label": "post-Xenium human colon cancer 10x Visium CytAssist subset",
        "slug": "post_xenium_human_colon_cancer_subset",
        "standardized": "post_xenium_human_colon_cancer_subset",
        "focus_pair": "SPP1-CD44",
        "focus_lr_id": "post_xenium_crc_lr_001_SPP1_CD44",
        "focus_method": "spatialself",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="visium_hd_crc")
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--min-cells", type=int, default=25)
    parser.add_argument("--random-seed", type=int, default=10403)
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    phase10_dir = PROJECT_ROOT / f"outputs/phase10/full_transcriptome_commot_style/{cfg['slug']}"
    result_dir = phase10_dir / "results"
    full_tx_dir = phase10_dir / "full_transcriptome"
    standardized_dir = GOAL5_DIR / f"standardized_inputs/{cfg['standardized']}"

    cells = pd.read_csv(standardized_dir / "cells.csv")
    candidates = pd.read_csv(standardized_dir / "candidate_interactions.csv")
    incoming = pd.read_csv(result_dir / "receiver_level_incoming_scores.csv")
    counts = sparse.load_npz(full_tx_dir / "expression_full_transcriptome_counts.npz").tocsr()
    genes = pd.read_csv(full_tx_dir / "genes.csv")["gene"].astype(str).to_numpy()
    barcodes = pd.read_csv(full_tx_dir / "barcodes.csv")["cell_id"].astype(str).tolist()
    if cells["cell_id"].astype(str).tolist() != barcodes:
        raise ValueError("Cell order mismatch between cells.csv and full-transcriptome matrix.")

    expressed = np.asarray((counts > 0).sum(axis=0)).ravel() >= args.min_cells
    gene_idx = np.flatnonzero(expressed)
    genes = genes[gene_idx]
    log_counts = counts[:, gene_idx].astype(np.float64)
    log_counts.data = np.log1p(log_counts.data)
    log_dense = np.asarray(log_counts.toarray(), dtype=np.float32)
    ranked_expr, valid_mask = rank_expression_matrix(log_dense)
    genes = genes[valid_mask]
    log_dense = log_dense[:, valid_mask]

    focus_incoming = (
        incoming[incoming["method"].eq(cfg["focus_method"])][["cell_id", cfg["focus_pair"]]]
        .rename(columns={cfg["focus_pair"]: "incoming"})
        .merge(cells[["cell_id", "cell_type", "niche", "x", "y"]], on="cell_id", how="left")
    )
    if len(focus_incoming) != len(cells):
        raise ValueError("Incoming score table does not cover all cells for the focus method.")

    rng = np.random.default_rng(args.random_seed)
    controls = build_controls(focus_incoming, incoming, candidates, cfg["focus_lr_id"], cfg["focus_pair"], cfg["focus_method"], rng)
    rows = []
    scans = []
    for control_name, control in controls.items():
        signal = control["signal"]
        scan = scan_signal(signal, ranked_expr, genes)
        scan.insert(0, "control_type", control_name)
        scan.insert(1, "control_description", control["description"])
        scans.append(scan.sort_values(["spearman_r", "bh_fdr"], ascending=[False, True]).head(300))
        rows.append(
            {
                "dataset": cfg["label"],
                "method": cfg["focus_method"],
                "lr_pair": cfg["focus_pair"],
                "control_type": control_name,
                "control_description": control["description"],
                "n_nonzero_receivers": int(np.sum(signal > 0)),
                "signal_mean": float(np.mean(signal)),
                "signal_sd": float(np.std(signal)),
                **target_enrichment(scan, genes, CURATED_TARGETS[cfg["focus_pair"]], args.top_n),
                **communication_impact(signal, log_dense, genes, cells, CURATED_TARGETS[cfg["focus_pair"]], args.random_seed),
            }
        )

    summary = pd.DataFrame(rows)
    scans_top = pd.concat(scans, ignore_index=True)
    baseline = summary[summary["control_type"].eq("observed")].iloc[0]
    summary["target_rank_specificity_vs_observed"] = baseline["mean_target_rank"] - summary["mean_target_rank"]
    summary["impact_delta_r2_vs_observed"] = summary["delta_cv_r2"] - baseline["delta_cv_r2"]
    summary["observed_minus_control_target_rank"] = summary["mean_target_rank"] - baseline["mean_target_rank"]
    summary["observed_minus_control_delta_cv_r2"] = baseline["delta_cv_r2"] - summary["delta_cv_r2"]
    summary["control_status"] = summary.apply(classify_control, axis=1)

    summary.to_csv(result_dir / "spp1_cd44_control_specificity_summary.csv", index=False)
    scans_top.to_csv(result_dir / "spp1_cd44_control_specificity_top_genes.csv", index=False)
    manifest = {
        "dataset_key": args.dataset,
        "dataset_label": cfg["label"],
        "focus_method": cfg["focus_method"],
        "focus_lr_pair": cfg["focus_pair"],
        "focus_lr_id": cfg["focus_lr_id"],
        "controls": {k: v["description"] for k, v in controls.items()},
        "top_n": args.top_n,
        "min_cells": args.min_cells,
        "outputs": [
            "spp1_cd44_control_specificity_summary.csv",
            "spp1_cd44_control_specificity_top_genes.csv",
        ],
        "claim_boundary": (
            "Control specificity checks compare observed SPP1-CD44 incoming scores with shuffled and "
            "context-masked variants. They support specificity but are not perturbation evidence."
        ),
    }
    (result_dir / "spp1_cd44_control_specificity_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


def build_controls(
    focus_incoming: pd.DataFrame,
    all_incoming: pd.DataFrame,
    candidates: pd.DataFrame,
    lr_id: str,
    focus_pair: str,
    focus_method: str,
    rng: np.random.Generator,
) -> dict[str, dict[str, object]]:
    signal = focus_incoming["incoming"].to_numpy(dtype=float)
    lr_candidates = candidates[candidates["lr_id"].eq(lr_id)].copy()
    receiver_ids = focus_incoming["cell_id"].astype(str).tolist()
    receiver_index = {cell_id: i for i, cell_id in enumerate(receiver_ids)}
    method_incoming = all_incoming[all_incoming["method"].eq(focus_method)].set_index("cell_id").reindex(receiver_ids)

    def mask_signal(mask: pd.Series) -> np.ndarray:
        selected = lr_candidates[mask].copy()
        out = np.zeros(len(receiver_ids), dtype=float)
        if selected.empty:
            return out
        # Use mean observed incoming score of receivers represented in this masked edge set.
        represented = selected["receiver_id"].astype(str).map(receiver_index).dropna().astype(int).unique()
        out[represented] = signal[represented]
        return out

    wrong_niche = ~lr_candidates["niche_rule_satisfied"].astype(bool)
    wrong_type = ~lr_candidates["type_rule_satisfied"].astype(bool)
    wrong_receiver_cells = ~focus_incoming["cell_type"].astype(str).isin(["caf_proxy", "myeloid_proxy", "spp1_macrophage_proxy"])
    wrong_receiver_signal = signal.copy()
    wrong_receiver_signal[~wrong_receiver_cells.to_numpy()] = 0.0
    shuffled_lr_signal, shuffled_lr_source = shuffled_lr_identity_signal(method_incoming, focus_pair, rng)

    return {
        "observed": {
            "signal": signal,
            "description": "Observed SpatialSelf SPP1-CD44 incoming score.",
        },
        "shuffled_lr_identity": {
            "signal": shuffled_lr_signal,
            "description": f"Incoming score from a randomly selected non-SPP1-CD44 LR column: {shuffled_lr_source}.",
        },
        "shuffled_receiver": {
            "signal": rng.permutation(signal),
            "description": "Receiver-level incoming scores randomly permuted across spots/bins.",
        },
        "wrong_niche_masked": {
            "signal": mask_signal(wrong_niche),
            "description": "Observed receiver scores retained only for SPP1-CD44 candidate edges outside the expected niche.",
        },
        "wrong_type_masked": {
            "signal": mask_signal(wrong_type),
            "description": "Observed receiver scores retained only for SPP1-CD44 candidate edges failing sender/receiver type rule.",
        },
        "wrong_receiver_masked": {
            "signal": wrong_receiver_signal,
            "description": "Observed receiver scores retained only in receiver cell types not expected for SPP1-CD44 response.",
        },
    }


def shuffled_lr_identity_signal(method_incoming: pd.DataFrame, focus_pair: str, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    candidates = [
        col
        for col in method_incoming.columns
        if col not in {"method"} and col != focus_pair and pd.api.types.is_numeric_dtype(method_incoming[col])
    ]
    if not candidates:
        return np.zeros(len(method_incoming), dtype=float), "none_available"
    source = str(rng.choice(candidates))
    return method_incoming[source].fillna(0.0).to_numpy(dtype=float), source


def scan_signal(signal: np.ndarray, ranked_expression: np.ndarray, genes: np.ndarray) -> pd.DataFrame:
    if np.nanstd(signal) <= 1e-12:
        return pd.DataFrame(
            {
                "gene": genes,
                "spearman_r": np.nan,
                "spearman_p": np.nan,
                "bh_fdr": np.nan,
                "status": "constant_signal",
            }
        )
    signal_rank = rankdata(signal).astype(np.float64)
    signal_centered = signal_rank - signal_rank.mean()
    signal_norm = np.sqrt(np.sum(signal_centered**2))
    expr_centered = ranked_expression - ranked_expression.mean(axis=0, keepdims=True)
    expr_norm = np.sqrt(np.sum(expr_centered**2, axis=0))
    rho = (signal_centered @ expr_centered) / (signal_norm * expr_norm)
    rho = np.clip(rho, -0.999999, 0.999999)
    n = ranked_expression.shape[0]
    t_stat = rho * np.sqrt((n - 2) / np.maximum(1e-12, 1 - rho**2))
    pvals = 2 * student_t.sf(np.abs(t_stat), df=n - 2)
    out = pd.DataFrame({"gene": genes, "spearman_r": rho, "spearman_p": pvals, "status": "tested"})
    out["bh_fdr"] = bh_adjust(out["spearman_p"].to_numpy(dtype=float))
    return out


def target_enrichment(scan: pd.DataFrame, universe_genes: np.ndarray, target_genes: list[str], top_n: int) -> dict[str, object]:
    targets = set(target_genes) & set(universe_genes)
    tested = scan[scan["status"].eq("tested")].sort_values(["spearman_r", "bh_fdr"], ascending=[False, True]).copy()
    top = set(tested.head(top_n)["gene"].astype(str))
    universe = set(tested["gene"].astype(str))
    overlap = len(top & targets)
    n_targets = len(targets & universe)
    n_universe = len(universe)
    if n_targets == 0:
        p_value = np.nan
        fold = np.nan
        odds = np.nan
    else:
        p_value = float(hypergeom.sf(overlap - 1, n_universe, n_targets, top_n))
        fold = float((overlap / top_n) / (n_targets / n_universe))
        table = [[overlap, top_n - overlap], [n_targets - overlap, n_universe - n_targets - top_n + overlap]]
        odds = float(fisher_exact(table, alternative="greater")[0])
    ranks = tested.reset_index(drop=True)
    ranks["rank"] = np.arange(1, len(ranks) + 1)
    target_ranks = ranks[ranks["gene"].isin(targets)]["rank"]
    return {
        "target_overlap_top_n": overlap,
        "n_targets": n_targets,
        "target_enrichment_p": p_value,
        "target_enrichment_fold": fold,
        "target_enrichment_odds_ratio": odds,
        "mean_target_rank": float(target_ranks.mean()) if len(target_ranks) else np.nan,
        "top_target_genes": ",".join([g for g in tested.head(top_n)["gene"].astype(str) if g in targets][:20]),
    }


def communication_impact(
    signal: np.ndarray,
    log_counts: np.ndarray,
    genes: np.ndarray,
    cells: pd.DataFrame,
    target_genes: list[str],
    seed: int,
) -> dict[str, object]:
    present = [g for g in target_genes if g in set(genes)]
    if not present or np.nanstd(signal) <= 1e-12:
        return {"delta_cv_r2": np.nan, "cv_r2_without_comm": np.nan, "cv_r2_with_comm": np.nan, "n_targets_for_impact": len(present)}
    idx = [int(np.where(genes == g)[0][0]) for g in present]
    y = np.asarray(log_counts[:, idx].mean(axis=1)).ravel()
    x_base = pd.DataFrame(
        {
            "library_log_sum": np.asarray(log_counts.sum(axis=1)).ravel(),
            "x": pd.to_numeric(cells["x"], errors="coerce").to_numpy(),
            "y": pd.to_numeric(cells["y"], errors="coerce").to_numpy(),
        }
    )
    x_full = x_base.copy()
    x_full["incoming"] = signal
    return {**cross_validated_delta_r2(x_base, x_full, y, seed), "n_targets_for_impact": len(present)}


def cross_validated_delta_r2(x_base: pd.DataFrame, x_full: pd.DataFrame, y: np.ndarray, seed: int) -> dict[str, float]:
    if np.nanstd(y) <= 1e-12:
        return {"cv_r2_without_comm": np.nan, "cv_r2_with_comm": np.nan, "delta_cv_r2": np.nan}
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    base_scores = []
    full_scores = []
    for train, test in kf.split(x_base):
        base = LinearRegression().fit(x_base.iloc[train], y[train])
        full = LinearRegression().fit(x_full.iloc[train], y[train])
        base_scores.append(r2_score(y[test], base.predict(x_base.iloc[test])))
        full_scores.append(r2_score(y[test], full.predict(x_full.iloc[test])))
    base_mean = float(np.mean(base_scores))
    full_mean = float(np.mean(full_scores))
    return {"cv_r2_without_comm": base_mean, "cv_r2_with_comm": full_mean, "delta_cv_r2": full_mean - base_mean}


def classify_control(row: pd.Series) -> str:
    if row["control_type"] == "observed":
        return "observed_reference"
    rank_drop = row.get("observed_minus_control_target_rank", np.nan)
    impact_drop = row.get("observed_minus_control_delta_cv_r2", np.nan)
    if pd.notna(rank_drop) and rank_drop > 0 and pd.notna(impact_drop) and impact_drop > 0:
        return "observed_more_specific_than_control"
    if pd.notna(rank_drop) and rank_drop > 0:
        return "observed_has_better_target_rank"
    if pd.notna(impact_drop) and impact_drop > 0:
        return "observed_has_better_impact"
    return "control_remains_competitive"


def rank_expression_matrix(log_dense: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    std = log_dense.std(axis=0)
    valid = std > 1e-12
    ranked = np.apply_along_axis(rankdata, 0, log_dense[:, valid]).astype(np.float32)
    return ranked, valid


def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty(n, dtype=float)
    out[order] = adjusted
    return out


if __name__ == "__main__":
    main()
