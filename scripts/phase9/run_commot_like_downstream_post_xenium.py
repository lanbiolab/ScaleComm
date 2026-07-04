#!/usr/bin/env python
"""COMMOT-like downstream validation for the post-Xenium colon SPP1-CD44 case.

This treats the receiver-level incoming LR signal as a spatial covariate and asks
whether receiver genes or selected receiver-state signatures vary with it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOAL5_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
DATASET_DIR = GOAL5_DIR / "standardized_inputs/post_xenium_human_colon_cancer_subset"

PREDICTIONS = {
    "spatialself": GOAL5_DIR
    / "post_xenium_human_colon_cancer_spatialself_calibrated/spatialself_wavelet_edge_prior/"
    / "post_xenium_human_colon_cancer_subset__spatialself_wavelet_predictions.csv",
    "commot": GOAL5_DIR
    / "post_xenium_human_colon_cancer_baseline_panel/commot/"
    / "post_xenium_human_colon_cancer_subset__commot_predictions.csv",
}

LR_ID = "post_xenium_crc_lr_001_SPP1_CD44"
NEGATIVE_LR_IDS = [
    "post_xenium_crc_lr_005_ACTB_KRT8_negative",
    "post_xenium_crc_lr_006_COL1A1_EPCAM_negative",
]

SIGNATURES = {
    "spp1_cd44_clean_myeloid_stromal": ["LYZ", "CD68", "CD14", "C1QA", "C1QB", "COL1A1"],
    "spp1_cd44_full_proxy": ["CD44", "SPP1", "LYZ", "CD68", "CD14", "C1QA", "C1QB", "COL1A1"],
    "negative_epithelial_proxy": ["EPCAM", "KRT8", "KRT19", "MUC1", "EGFR"],
}

CONTROL_FEATURES = [
    "library_size",
    "x",
    "y",
    "CD44",
    "SPP1",
    "COL1A1",
    "EPCAM",
    "KRT8",
    "ACTB",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs/phase9/commot_like_downstream_post_xenium",
    )
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=9187)
    args = parser.parse_args()

    out = PROJECT_ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(DATASET_DIR / "cells.csv")
    expression = pd.read_csv(DATASET_DIR / "expression.csv")
    candidates = pd.read_csv(DATASET_DIR / "candidate_interactions.csv")

    receiver_tables = []
    gene_tables = []
    signature_tables = []
    impact_tables = []
    for method, prediction_path in PREDICTIONS.items():
        predictions = pd.read_csv(prediction_path)
        receiver = build_receiver_table(cells, expression, candidates, predictions, method)
        receiver_tables.append(receiver)

        gene_tables.append(
            gene_covariate_scan(
                receiver,
                method,
                args.n_permutations,
                args.random_seed,
            )
        )
        signature_tables.append(
            signature_covariate_scan(
                receiver,
                method,
                args.n_permutations,
                args.random_seed,
            )
        )
        impact_tables.append(
            communication_impact(
                receiver,
                method,
                args.random_seed,
            )
        )

    receiver_all = pd.concat(receiver_tables, ignore_index=True)
    gene_all = pd.concat(gene_tables, ignore_index=True)
    signature_all = pd.concat(signature_tables, ignore_index=True)
    impact_all = pd.concat(impact_tables, ignore_index=True)

    gene_all = add_bh_by_method(gene_all, "empirical_p")
    signature_all = add_bh_by_method(signature_all, "empirical_p")

    receiver_all.to_csv(out / "receiver_level_incoming_covariates.csv", index=False)
    gene_all.to_csv(out / "commot_like_gene_covariate_scan.csv", index=False)
    signature_all.to_csv(out / "commot_like_signature_covariate_scan.csv", index=False)
    impact_all.to_csv(out / "commot_like_communication_impact.csv", index=False)
    write_summary(out, signature_all, impact_all)

    manifest = {
        "dataset": str(DATASET_DIR.relative_to(PROJECT_ROOT)),
        "target_lr_id": LR_ID,
        "negative_lr_ids": NEGATIVE_LR_IDS,
        "predictions": {k: str(v.relative_to(PROJECT_ROOT)) for k, v in PREDICTIONS.items()},
        "n_permutations": args.n_permutations,
        "analysis_note": (
            "COMMOT-like validation: receiver-level incoming signal is used as a spatial covariate. "
            "The current post-Xenium panel has selected genes only, so this is selected-gene receiver-state "
            "validation rather than whole-transcriptome communication DEG analysis."
        ),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print((out / "commot_like_signature_covariate_scan.csv").relative_to(PROJECT_ROOT))
    print((out / "commot_like_communication_impact.csv").relative_to(PROJECT_ROOT))
    print(signature_all.to_string(index=False))
    print(impact_all.to_string(index=False))


def build_receiver_table(
    cells: pd.DataFrame,
    expression: pd.DataFrame,
    candidates: pd.DataFrame,
    predictions: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    table = candidates.merge(predictions[["candidate_id", "score"]], on="candidate_id", how="inner")
    incoming = []
    for lr_id in [LR_ID, *NEGATIVE_LR_IDS]:
        lr_table = table[table["lr_id"].eq(lr_id)]
        agg = lr_table.groupby("receiver_id", as_index=False)["score"].mean()
        pair = lr_table[["ligand", "receptor"]].drop_duplicates().iloc[0]
        pair_name = f"{pair['ligand']}-{pair['receptor']}"
        agg = agg.rename(columns={"receiver_id": "cell_id", "score": f"incoming_{pair_name}"})
        incoming.append(agg)

    receiver = cells[["cell_id", "x", "y", "cell_type", "niche"]].merge(expression, on="cell_id", how="left")
    for frame in incoming:
        receiver = receiver.merge(frame, on="cell_id", how="left")
    incoming_cols = [col for col in receiver.columns if col.startswith("incoming_")]
    receiver[incoming_cols] = receiver[incoming_cols].fillna(0.0)
    receiver["method"] = method
    gene_cols = [col for col in expression.columns if col != "cell_id"]
    receiver["library_size"] = receiver[gene_cols].sum(axis=1)
    receiver["spp1_cd44_incoming"] = receiver["incoming_SPP1-CD44"]
    return receiver


def gene_covariate_scan(
    receiver: pd.DataFrame,
    method: str,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    signal = receiver["spp1_cd44_incoming"].to_numpy(dtype=float)
    genes = [
        col
        for col in receiver.columns
        if col.isupper() and col not in {"SPP1", "CD44"}
    ]
    rows = []
    for gene in genes:
        expr = np.log1p(receiver[gene].to_numpy(dtype=float))
        rows.append(
            {
                "method": method,
                "gene": gene,
                **spearman_with_permutation(signal, expr, rng, n_permutations),
            }
        )
    return pd.DataFrame(rows).sort_values(["method", "spearman_r"], ascending=[True, False])


def signature_covariate_scan(
    receiver: pd.DataFrame,
    method: str,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    signal = receiver["spp1_cd44_incoming"].to_numpy(dtype=float)
    rows = []
    for name, genes in SIGNATURES.items():
        present = [gene for gene in genes if gene in receiver.columns]
        score = np.log1p(receiver[present].to_numpy(dtype=float)).mean(axis=1)
        rows.append(
            {
                "method": method,
                "signature": name,
                "genes_present": ",".join(present),
                "n_genes": len(present),
                **spearman_with_permutation(signal, score, rng, n_permutations),
            }
        )
    return pd.DataFrame(rows)


def communication_impact(receiver: pd.DataFrame, method: str, seed: int) -> pd.DataFrame:
    rows = []
    for name, genes in SIGNATURES.items():
        present = [gene for gene in genes if gene in receiver.columns]
        y = np.log1p(receiver[present].to_numpy(dtype=float)).mean(axis=1)
        controls = [feature for feature in CONTROL_FEATURES if feature in receiver.columns]
        x_base = receiver[controls].copy()
        x_base["x"] = pd.to_numeric(receiver["x"], errors="coerce")
        x_base["y"] = pd.to_numeric(receiver["y"], errors="coerce")
        x_base = x_base.loc[:, ~x_base.columns.duplicated()]
        x_full = x_base.copy()
        x_full["spp1_cd44_incoming"] = receiver["spp1_cd44_incoming"].to_numpy(dtype=float)

        rows.append(
            {
                "method": method,
                "signature": name,
                "genes_present": ",".join(present),
                "n_genes": len(present),
                **cross_validated_delta_r2(x_base, x_full, y, seed),
                **random_forest_importance(x_full, y, seed),
            }
        )
    return pd.DataFrame(rows)


def spearman_with_permutation(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_permutations: int,
) -> dict[str, float | str]:
    if np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return {
            "spearman_r": np.nan,
            "spearman_p": np.nan,
            "empirical_p": np.nan,
            "status": "constant_vector",
        }
    corr, p_value = spearmanr(x, y)
    exceed = 0
    for _ in range(n_permutations):
        null_y = rng.permutation(y)
        null_corr, _ = spearmanr(x, null_y)
        if abs(float(null_corr)) >= abs(float(corr)):
            exceed += 1
    return {
        "spearman_r": float(corr),
        "spearman_p": float(p_value),
        "empirical_p": float((exceed + 1) / (n_permutations + 1)),
        "status": "tested",
    }


def cross_validated_delta_r2(
    x_base: pd.DataFrame,
    x_full: pd.DataFrame,
    y: np.ndarray,
    seed: int,
) -> dict[str, float]:
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    base_scores = []
    full_scores = []
    for train_idx, test_idx in kf.split(x_base):
        base_model = LinearRegression()
        full_model = LinearRegression()
        base_model.fit(x_base.iloc[train_idx], y[train_idx])
        full_model.fit(x_full.iloc[train_idx], y[train_idx])
        base_scores.append(r2_score(y[test_idx], base_model.predict(x_base.iloc[test_idx])))
        full_scores.append(r2_score(y[test_idx], full_model.predict(x_full.iloc[test_idx])))
    base_mean = float(np.mean(base_scores))
    full_mean = float(np.mean(full_scores))
    return {
        "cv_r2_without_comm": base_mean,
        "cv_r2_with_comm": full_mean,
        "delta_cv_r2": full_mean - base_mean,
    }


def random_forest_importance(x_full: pd.DataFrame, y: np.ndarray, seed: int) -> dict[str, float]:
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_full, y)
    importances = pd.Series(model.feature_importances_, index=x_full.columns).sort_values(ascending=False)
    comm_importance = float(importances.get("spp1_cd44_incoming", 0.0))
    rank = int(importances.index.get_loc("spp1_cd44_incoming") + 1) if "spp1_cd44_incoming" in importances.index else -1
    return {
        "rf_comm_importance": comm_importance,
        "rf_comm_importance_rank": rank,
        "rf_top_features": ";".join([f"{k}:{v:.4f}" for k, v in importances.head(5).items()]),
    }


def add_bh_by_method(df: pd.DataFrame, p_col: str) -> pd.DataFrame:
    out = df.copy()
    out["bh_fdr"] = np.nan
    for method, idx in out.groupby("method").groups.items():
        pvals = out.loc[idx, p_col].to_numpy(dtype=float)
        mask = np.isfinite(pvals)
        adjusted = np.full(len(pvals), np.nan)
        adjusted[mask] = bh_adjust(pvals[mask])
        out.loc[idx, "bh_fdr"] = adjusted
    return out


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


def write_summary(out: Path, signatures: pd.DataFrame, impact: pd.DataFrame) -> None:
    focus = signatures[signatures["signature"].eq("spp1_cd44_clean_myeloid_stromal")]
    negative = signatures[signatures["signature"].eq("negative_epithelial_proxy")]
    focus_impact = impact[impact["signature"].eq("spp1_cd44_clean_myeloid_stromal")]
    negative_impact = impact[impact["signature"].eq("negative_epithelial_proxy")]
    rows = []
    for method in sorted(signatures["method"].unique()):
        s = focus[focus["method"].eq(method)].iloc[0]
        n = negative[negative["method"].eq(method)].iloc[0]
        i = focus_impact[focus_impact["method"].eq(method)].iloc[0]
        ni = negative_impact[negative_impact["method"].eq(method)].iloc[0]
        rows.append(
            {
                "method": method,
                "clean_signature_spearman": s["spearman_r"],
                "clean_signature_fdr": s["bh_fdr"],
                "negative_signature_spearman": n["spearman_r"],
                "commot_like_downstream_specificity": float(s["spearman_r"] - n["spearman_r"]),
                "delta_cv_r2": i["delta_cv_r2"],
                "negative_delta_cv_r2": ni["delta_cv_r2"],
                "impact_specificity_delta_r2": float(i["delta_cv_r2"] - ni["delta_cv_r2"]),
                "rf_comm_importance": i["rf_comm_importance"],
                "rf_comm_importance_rank": i["rf_comm_importance_rank"],
                "interpretation": summarize_method(s, i),
            }
        )
    pd.DataFrame(rows).to_csv(out / "commot_like_story_summary.csv", index=False)


def summarize_method(signature_row: pd.Series, impact_row: pd.Series) -> str:
    if pd.isna(signature_row["spearman_r"]):
        return "not evaluable"
    corr_ok = float(signature_row["spearman_r"]) > 0 and float(signature_row["bh_fdr"]) <= 0.1
    impact_ok = float(impact_row["delta_cv_r2"]) > 0
    if corr_ok and impact_ok:
        return "received signal is positively associated with receiver-state signature and adds predictive value"
    if corr_ok:
        return "received signal is positively associated with receiver-state signature but independent predictive value is weak"
    if impact_ok:
        return "received signal adds predictive value but marginal correlation is weak"
    return "no strong selected-gene downstream support"


if __name__ == "__main__":
    main()
