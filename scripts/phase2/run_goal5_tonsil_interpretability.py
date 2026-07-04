#!/usr/bin/env python
"""Interpretability validation for the fresh Phase 2 tonsil subset.

The analysis is deliberately real-data oriented: it correlates predicted LR
communication scores with receiver-side target/module signatures, summarizes
mechanism/pathway categories, and builds simple CCC modules from LR activity
profiles. It does not treat the tonsil data as ground truth.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import fisher_exact, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spitialself.fdr import bh_adjust  # noqa: E402


TARGET_SIGNATURES = {
    "germinal_center_b_cell": ["BCL6", "AICDA", "S1PR2", "MEF2B", "POU2AF1", "CD79A", "MS4A1"],
    "t_cell_migration": ["CCR7", "TCF7", "IL7R", "LTB", "SELL", "LEF1"],
    "nfkb_activation": ["NFKBIA", "NFKB1", "REL", "BCL2A1", "CD83", "CD86"],
    "tfh_activation": ["ICOSLG", "IL21", "CXCR5", "MAF", "PDCD1", "TOX"],
    "plasma_b_cell_response": ["MZB1", "XBP1", "PRDM1", "JCHAIN", "IGKC", "TNFRSF17", "CD38"],
    "chemokine_response": ["CXCR4", "CCR7", "BCL2", "MYC", "JUN", "FOS"],
    "early_response": ["JUN", "FOS", "FOSB", "IRF4", "STAT3", "BATF", "CD69"],
    "antigen_presentation": ["HLA-DRA", "HLA-DRB1", "HLA-DPB1", "HLA-DPA1", "CD74", "CD83"],
    "adhesion_activation": ["ITGAL", "ICAM1", "CD44", "CD40", "CD86", "CD58"],
    "housekeeping_negative": ["ACTB", "GAPDH", "ITGB1", "EGFR"],
}

CONTROL_TARGETS = {
    "CXCL13-CXCR5": ["germinal_center_b_cell", "chemokine_response"],
    "CXCL12-CXCR4": ["chemokine_response", "germinal_center_b_cell"],
    "CD40-CD40LG": ["nfkb_activation", "germinal_center_b_cell"],
    "ICOSLG-ICOS": ["tfh_activation", "early_response"],
    "TNFSF13B-TNFRSF13C": ["plasma_b_cell_response", "nfkb_activation"],
    "CCL21-CCR7": ["t_cell_migration", "chemokine_response"],
    "ACTB-GAPDH": ["housekeeping_negative"],
    "GAPDH-ITGB1": ["housekeeping_negative"],
    "ACTB-EGFR": ["housekeeping_negative"],
}
CONTROL_TARGETS_CANONICAL = {
    _key: _value for _key, _value in CONTROL_TARGETS.items()
}
CONTROL_EXPECTED_RECEIVERS = {
    "CXCL13-CXCR5": {"B_germinal_center", "B_naive", "B_memory", "T_follicular_helper"},
    "CXCL12-CXCR4": {"B_germinal_center", "B_naive", "B_memory", "T_CD4", "T_follicular_helper"},
    "CD40-CD40LG": {"B_germinal_center", "B_naive", "B_memory", "FDC"},
    "ICOSLG-ICOS": {"T_follicular_helper", "T_CD4"},
    "TNFSF13B-TNFRSF13C": {"B_germinal_center", "B_naive", "B_memory", "plasma"},
    "CCL21-CCR7": {"T_CD4", "T_CD8", "T_follicular_helper", "mDC"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--goal4-dir",
        default="outputs/phase2/goal4_high_resolution/tonsil_fresh_wavelet_calibrated_subset",
    )
    parser.add_argument("--raw-expression", default="high_spatial/data/HumanTonsil_expression.csv.gz")
    parser.add_argument("--output-dir", default="outputs/phase2/goal5_interpretability/tonsil_fresh_subset")
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--min-receiver-cells", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=2026)
    args = parser.parse_args()

    goal4_dir = PROJECT_ROOT / args.goal4_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(goal4_dir / "constructed_replicate/cells.csv")
    candidates = pd.read_csv(goal4_dir / "constructed_replicate/candidate_interactions.csv")
    lr_pairs = pd.read_csv(goal4_dir / "constructed_replicate/lr_pairs.csv")
    predictions = pd.read_csv(goal4_dir / "tonsil_fresh_spatialself_wavelet_predictions.csv")
    table = candidates.merge(
        predictions[["candidate_id", "method", args.score_column]],
        on="candidate_id",
        how="inner",
        validate="one_to_one",
    )
    target_genes = sorted({gene for genes in TARGET_SIGNATURES.values() for gene in genes})
    target_expression = _read_target_expression(
        PROJECT_ROOT / args.raw_expression,
        cell_ids=cells["cell_id"].tolist(),
        genes=target_genes,
    )
    signature_scores = _signature_scores(target_expression)
    lr_targets = _assign_lr_targets(lr_pairs)
    downstream = _downstream_correlations(
        table,
        signature_scores,
        lr_targets,
        score_column=args.score_column,
        n_permutations=args.n_permutations,
        min_receiver_cells=args.min_receiver_cells,
        random_seed=args.random_seed,
    )
    pathway_summary = _pathway_summary(downstream)
    module_profiles, module_summary = _communication_modules(table, lr_pairs, score_column=args.score_column)
    control_summary = _control_summary(downstream)
    goal_summary = _goal_summary(downstream, pathway_summary, module_summary, control_summary, args)

    downstream.to_csv(output_dir / "tonsil_downstream_target_correlation.csv", index=False)
    pathway_summary.to_csv(output_dir / "tonsil_pathway_category_summary.csv", index=False)
    module_profiles.to_csv(output_dir / "tonsil_lr_activity_profiles.csv", index=False)
    module_summary.to_csv(output_dir / "tonsil_ccc_module_summary.csv", index=False)
    control_summary.to_csv(output_dir / "tonsil_interpretability_control_summary.csv", index=False)
    pd.DataFrame([goal_summary]).to_csv(output_dir / "goal5_interpretability_summary.csv", index=False)
    (output_dir / "goal5_interpretability_manifest.json").write_text(
        json.dumps(
            {
                "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
                "goal4_dir": args.goal4_dir,
                "raw_expression": args.raw_expression,
                "output_dir": args.output_dir,
                "n_permutations": args.n_permutations,
                "target_signatures": TARGET_SIGNATURES,
                "control_targets": CONTROL_TARGETS,
                "summary": goal_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Goal 5 downstream: {output_dir / 'tonsil_downstream_target_correlation.csv'}")
    print(f"Goal 5 summary:    {output_dir / 'goal5_interpretability_summary.csv'}")


def _read_target_expression(path: Path, *, cell_ids: list[str], genes: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    gene_column = header[0]
    missing_cells = sorted(set(cell_ids) - set(header))
    if missing_cells:
        raise ValueError(f"Raw expression is missing selected cells: {missing_cells[:5]}")
    raw = pd.read_csv(path, usecols=[gene_column, *cell_ids])
    raw = raw[raw[gene_column].astype(str).isin(genes)].copy()
    raw = raw.groupby(gene_column, as_index=True).sum(numeric_only=True)
    for gene in genes:
        if gene not in raw.index:
            raw.loc[gene] = 0.0
    expression = raw.loc[genes].T.reset_index().rename(columns={"index": "cell_id"})
    return expression


def _signature_scores(expression: pd.DataFrame) -> pd.DataFrame:
    scores = pd.DataFrame({"cell_id": expression["cell_id"]})
    matrix = expression.set_index("cell_id")
    for signature, genes in TARGET_SIGNATURES.items():
        present = [gene for gene in genes if gene in matrix.columns]
        if present:
            scores[signature] = np.log1p(matrix[present].to_numpy(dtype=float)).mean(axis=1)
        else:
            scores[signature] = 0.0
    return scores


def _assign_lr_targets(lr_pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in lr_pairs.itertuples(index=False):
        canonical = _canonical_pair(row.ligand, row.receptor)
        signatures = _control_target_signatures(canonical)
        if signatures is None:
            signatures = _mechanism_signatures(str(row.mechanism), str(row.ligand), str(row.receptor))
        genes = _unique([gene for signature in signatures for gene in TARGET_SIGNATURES.get(signature, [])])
        rows.append(
            {
                "lr_id": row.lr_id,
                "ligand": row.ligand,
                "receptor": row.receptor,
                "canonical_pair": canonical,
                "mechanism": row.mechanism,
                "control_class": row.control_class,
                "target_signatures": ";".join(signatures),
                "target_genes": ";".join(genes),
                "n_target_genes": len(genes),
            }
        )
    return pd.DataFrame(rows)


def _downstream_correlations(
    table: pd.DataFrame,
    signature_scores: pd.DataFrame,
    lr_targets: pd.DataFrame,
    *,
    score_column: str,
    n_permutations: int,
    min_receiver_cells: int,
    random_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    signature_scores = signature_scores.set_index("cell_id")
    target_by_lr = lr_targets.set_index("lr_id")
    rows = []
    for lr_id, group in table.groupby("lr_id", sort=False):
        target = target_by_lr.loc[lr_id]
        signatures = [item for item in str(target["target_signatures"]).split(";") if item]
        receiver_score = group.groupby("receiver_id", sort=False)[score_column].mean()
        receiver_cell_type = group.groupby("receiver_id", sort=False)["receiver_cell_type"].first()
        common = signature_scores.index.intersection(receiver_score.index)
        row = {
            "lr_id": lr_id,
            "ligand": target["ligand"],
            "receptor": target["receptor"],
            "canonical_pair": target["canonical_pair"],
            "mechanism": target["mechanism"],
            "control_class": target["control_class"],
            "target_signatures": ";".join(signatures),
            "n_target_genes": int(target["n_target_genes"]),
            "n_receiver_cells": int(len(common)),
            "communication_mean": float(receiver_score.mean()),
            "spearman_r": float("nan"),
            "spearman_p_value": float("nan"),
            "empirical_p_value": float("nan"),
            "residual_spearman_r": float("nan"),
            "residual_empirical_p_value": float("nan"),
            "within_expected_receiver_spearman_r": float("nan"),
            "within_expected_receiver_n_cells": 0,
            "status": "not_tested",
        }
        if len(common) < min_receiver_cells or not signatures:
            row["status"] = "too_few_receiver_cells_or_targets"
            rows.append(row)
            continue
        communication = receiver_score.loc[common].to_numpy(dtype=float)
        target_signature = signature_scores.loc[common, signatures].mean(axis=1).to_numpy(dtype=float)
        if np.nanstd(communication) <= 1e-12 or np.nanstd(target_signature) <= 1e-12:
            row["status"] = "constant_vector"
            rows.append(row)
            continue
        corr, p_value = spearmanr(communication, target_signature)
        exceed = 0
        for _ in range(int(n_permutations)):
            null = rng.permutation(target_signature)
            null_corr, _ = spearmanr(communication, null)
            if abs(float(null_corr)) >= abs(float(corr)):
                exceed += 1
        receiver_types = receiver_cell_type.loc[common].astype(str).to_numpy()
        comm_resid = _residualize_by_group(communication, receiver_types)
        target_resid = _residualize_by_group(target_signature, receiver_types)
        residual_corr = float("nan")
        residual_empirical = float("nan")
        if np.nanstd(comm_resid) > 1e-12 and np.nanstd(target_resid) > 1e-12:
            residual_corr, _ = spearmanr(comm_resid, target_resid)
            residual_exceed = 0
            for _ in range(int(n_permutations)):
                null = rng.permutation(target_resid)
                null_corr, _ = spearmanr(comm_resid, null)
                if abs(float(null_corr)) >= abs(float(residual_corr)):
                    residual_exceed += 1
            residual_empirical = float((residual_exceed + 1.0) / (int(n_permutations) + 1.0))
        expected_receivers = _expected_receivers(str(target["canonical_pair"]))
        expected_corr = float("nan")
        expected_n = 0
        if expected_receivers:
            expected_mask = np.isin(receiver_types, sorted(expected_receivers))
            expected_n = int(expected_mask.sum())
            if expected_n >= min_receiver_cells:
                expected_comm = communication[expected_mask]
                expected_target = target_signature[expected_mask]
                if np.nanstd(expected_comm) > 1e-12 and np.nanstd(expected_target) > 1e-12:
                    expected_corr, _ = spearmanr(expected_comm, expected_target)
        row.update(
            {
                "spearman_r": float(corr),
                "spearman_p_value": float(p_value),
                "empirical_p_value": float((exceed + 1.0) / (int(n_permutations) + 1.0)),
                "residual_spearman_r": float(residual_corr),
                "residual_empirical_p_value": float(residual_empirical),
                "within_expected_receiver_spearman_r": float(expected_corr),
                "within_expected_receiver_n_cells": expected_n,
                "status": "tested",
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    tested = result["status"] == "tested"
    result["bh_fdr"] = np.nan
    if tested.any():
        result.loc[tested, "bh_fdr"] = bh_adjust(result.loc[tested, "empirical_p_value"].to_numpy(dtype=float))
    result["residual_bh_fdr"] = np.nan
    if tested.any():
        result.loc[tested, "residual_bh_fdr"] = bh_adjust(
            result.loc[tested, "residual_empirical_p_value"].fillna(1.0).to_numpy(dtype=float)
        )
    result["rank_by_spearman"] = result["spearman_r"].rank(method="min", ascending=False, na_option="bottom").astype(int)
    result["rank_by_residual_spearman"] = result["residual_spearman_r"].rank(method="min", ascending=False, na_option="bottom").astype(int)
    return result.sort_values(["status", "bh_fdr", "rank_by_spearman"], ascending=[True, True, True])


def _pathway_summary(downstream: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tested = downstream[downstream["status"] == "tested"]
    for signature in sorted({sig for value in downstream["target_signatures"].dropna() for sig in str(value).split(";") if sig}):
        subset = tested[tested["target_signatures"].str.contains(signature, regex=False, na=False)]
        rest = tested[~tested["target_signatures"].str.contains(signature, regex=False, na=False)]
        sig_pos = int(((subset["control_class"] == "tonsil_positive_prior") & (subset["spearman_r"] > 0)).sum())
        sig_non = int(len(subset) - sig_pos)
        rest_pos = int(((rest["control_class"] == "tonsil_positive_prior") & (rest["spearman_r"] > 0)).sum())
        rest_non = int(len(rest) - rest_pos)
        _, p_value = fisher_exact([[sig_pos, sig_non], [rest_pos, rest_non]], alternative="greater") if len(tested) else (math.nan, math.nan)
        rows.append(
            {
                "target_signature": signature,
                "n_lr_pairs": int(len(subset)),
                "mean_spearman": float(subset["spearman_r"].mean()) if len(subset) else math.nan,
                "median_spearman": float(subset["spearman_r"].median()) if len(subset) else math.nan,
                "n_positive_control_pairs": int((subset["control_class"] == "tonsil_positive_prior").sum()),
                "n_housekeeping_negative_pairs": int((subset["control_class"] == "housekeeping_negative_control").sum()),
                "n_background_pairs": int((subset["control_class"] == "background_lr").sum()),
                "positive_control_positive_corr_fisher_p": float(p_value),
            }
        )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["signature_bh_fdr"] = bh_adjust(summary["positive_control_positive_corr_fisher_p"].fillna(1.0).to_numpy(dtype=float))
    return summary


def _communication_modules(table: pd.DataFrame, lr_pairs: pd.DataFrame, *, score_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    profiles = (
        table.pivot_table(index="lr_id", columns="receiver_cell_type", values=score_column, aggfunc="mean", fill_value=0.0)
        .reset_index()
    )
    lr_meta = lr_pairs[["lr_id", "ligand", "receptor", "mechanism", "control_class"]]
    profiles = lr_meta.merge(profiles, on="lr_id", how="left")
    value_columns = [col for col in profiles.columns if col not in {"lr_id", "ligand", "receptor", "mechanism", "control_class"}]
    values = profiles[value_columns].to_numpy(dtype=float)
    if len(profiles) >= 3 and values.shape[1] >= 2:
        corr = np.corrcoef(values)
        corr = np.nan_to_num(corr, nan=0.0)
        distance = np.clip(1.0 - corr, 0.0, 2.0)
        np.fill_diagonal(distance, 0.0)
        clusters = fcluster(linkage(squareform(distance, checks=False), method="average"), t=0.75, criterion="distance")
    else:
        clusters = np.ones(len(profiles), dtype=int)
    profiles["ccc_module"] = clusters.astype(int)
    rows = []
    for module, subset in profiles.groupby("ccc_module", sort=True):
        mechanism_counts = subset["mechanism"].value_counts()
        control_counts = subset["control_class"].value_counts()
        receiver_means = subset[value_columns].mean().sort_values(ascending=False)
        rows.append(
            {
                "ccc_module": int(module),
                "n_lr_pairs": int(len(subset)),
                "dominant_mechanism": mechanism_counts.index[0] if len(mechanism_counts) else "",
                "dominant_mechanism_fraction": float(mechanism_counts.iloc[0] / len(subset)) if len(mechanism_counts) else 0.0,
                "n_positive_controls": int((subset["control_class"] == "tonsil_positive_prior").sum()),
                "n_housekeeping_negative_controls": int((subset["control_class"] == "housekeeping_negative_control").sum()),
                "n_background_lr": int((subset["control_class"] == "background_lr").sum()),
                "top_receiver_cell_types": ";".join(receiver_means.head(3).index.astype(str)),
                "top_receiver_score_mean": float(receiver_means.iloc[0]) if len(receiver_means) else math.nan,
            }
        )
    return profiles, pd.DataFrame(rows)


def _control_summary(downstream: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for control_class, subset in downstream.groupby("control_class", sort=False):
        tested = subset[subset["status"] == "tested"]
        rows.append(
            {
                "control_class": control_class,
                "n_lr_pairs": int(len(subset)),
                "n_tested": int(len(tested)),
                "mean_spearman": float(tested["spearman_r"].mean()) if len(tested) else math.nan,
                "median_spearman": float(tested["spearman_r"].median()) if len(tested) else math.nan,
                "mean_residual_spearman": float(tested["residual_spearman_r"].mean()) if len(tested) else math.nan,
                "median_residual_spearman": float(tested["residual_spearman_r"].median()) if len(tested) else math.nan,
                "mean_expected_receiver_spearman": float(tested["within_expected_receiver_spearman_r"].mean()) if len(tested) else math.nan,
                "n_positive_spearman": int((tested["spearman_r"] > 0).sum()),
                "n_positive_residual_spearman": int((tested["residual_spearman_r"] > 0).sum()),
                "n_bh_fdr_0_10": int((tested["bh_fdr"] <= 0.10).sum()),
                "n_residual_bh_fdr_0_10": int((tested["residual_bh_fdr"] <= 0.10).sum()),
                "best_rank_by_spearman": int(tested["rank_by_spearman"].min()) if len(tested) else 0,
                "best_rank_by_residual_spearman": int(tested["rank_by_residual_spearman"].min()) if len(tested) else 0,
            }
        )
    return pd.DataFrame(rows)


def _goal_summary(
    downstream: pd.DataFrame,
    pathway_summary: pd.DataFrame,
    module_summary: pd.DataFrame,
    control_summary: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, object]:
    positives = downstream[(downstream["control_class"] == "tonsil_positive_prior") & (downstream["status"] == "tested")]
    negatives = downstream[
        (downstream["control_class"] == "housekeeping_negative_control") & (downstream["status"] == "tested")
    ]
    background = downstream[(downstream["control_class"] == "background_lr") & (downstream["status"] == "tested")]
    positive_mean = float(positives["spearman_r"].mean()) if len(positives) else math.nan
    negative_mean = float(negatives["spearman_r"].mean()) if len(negatives) else math.nan
    significant_positive = int(((positives["spearman_r"] > 0) & (positives["bh_fdr"] <= 0.10)).sum())
    significant_negative = int(((negatives["spearman_r"] > 0) & (negatives["bh_fdr"] <= 0.10)).sum())
    background_mean = float(background["spearman_r"].mean()) if len(background) else math.nan
    positive_residual_mean = float(positives["residual_spearman_r"].mean()) if len(positives) else math.nan
    negative_residual_mean = float(negatives["residual_spearman_r"].mean()) if len(negatives) else math.nan
    background_residual_mean = float(background["residual_spearman_r"].mean()) if len(background) else math.nan
    positive_expected_mean = float(positives["within_expected_receiver_spearman_r"].mean()) if len(positives) else math.nan
    significant_positive_residual = int(
        ((positives["residual_spearman_r"] > 0) & (positives["residual_bh_fdr"] <= 0.10)).sum()
    )
    significant_negative_residual = int(
        ((negatives["residual_spearman_r"] > 0) & (negatives["residual_bh_fdr"] <= 0.10)).sum()
    )
    top_lr = downstream.sort_values("rank_by_spearman").head(5)
    top_residual_lr = downstream.sort_values("rank_by_residual_spearman").head(5)
    return {
        "goal_id": "5",
        "dataset": "human_tonsil_slide_tags",
        "analysis_label": "tonsil_fresh_subset_downstream_interpretability",
        "evidence_status": "initial_interpretability_evidence",
        "n_lr_pairs": int(len(downstream)),
        "n_tested_lr_pairs": int((downstream["status"] == "tested").sum()),
        "positive_control_mean_spearman": positive_mean,
        "housekeeping_negative_mean_spearman": negative_mean,
        "background_lr_mean_spearman": background_mean,
        "positive_control_mean_residual_spearman": positive_residual_mean,
        "housekeeping_negative_mean_residual_spearman": negative_residual_mean,
        "background_lr_mean_residual_spearman": background_residual_mean,
        "positive_control_mean_expected_receiver_spearman": positive_expected_mean,
        "positive_minus_negative_mean_spearman": positive_mean - negative_mean
        if math.isfinite(positive_mean) and math.isfinite(negative_mean)
        else math.nan,
        "positive_minus_negative_mean_residual_spearman": positive_residual_mean - negative_residual_mean
        if math.isfinite(positive_residual_mean) and math.isfinite(negative_residual_mean)
        else math.nan,
        "significant_positive_controls_fdr_0_10": significant_positive,
        "significant_negative_controls_fdr_0_10": significant_negative,
        "significant_positive_controls_residual_fdr_0_10": significant_positive_residual,
        "significant_negative_controls_residual_fdr_0_10": significant_negative_residual,
        "n_pathway_signatures": int(len(pathway_summary)),
        "n_ccc_modules": int(len(module_summary)),
        "top_lr_by_downstream_spearman": ";".join(
            f"{row.canonical_pair}:{row.spearman_r:.3f}" for row in top_lr.itertuples(index=False)
        ),
        "top_lr_by_residual_downstream_spearman": ";".join(
            f"{row.canonical_pair}:{row.residual_spearman_r:.3f}" for row in top_residual_lr.itertuples(index=False)
        ),
        "n_permutations": int(args.n_permutations),
        "interpretation": (
            "Initial receiver-side target/module validation on a fresh tonsil subset; "
            "supports cautious biological interpretability only if positive controls exceed negative controls."
        ),
    }


def _mechanism_signatures(mechanism: str, ligand: str, receptor: str) -> list[str]:
    if mechanism == "long_range_chemokine_niche" or ligand.startswith(("CCL", "CXCL")) or receptor.startswith(("CCR", "CXCR")):
        return ["chemokine_response", "t_cell_migration"]
    if mechanism == "juxtacrine_contact" or any(token in ligand + receptor for token in ("ICAM", "ITG", "CD2", "CD58", "CD28", "CD86")):
        return ["adhesion_activation", "early_response"]
    if any(token in ligand + receptor for token in ("HLA", "CD74")):
        return ["antigen_presentation", "early_response"]
    if any(token in ligand + receptor for token in ("TNF", "CD40", "BTLA")):
        return ["nfkb_activation", "early_response"]
    return ["early_response", "antigen_presentation"]


def _control_target_signatures(canonical_pair: str) -> list[str] | None:
    direct = CONTROL_TARGETS.get(canonical_pair)
    if direct is not None:
        return direct
    for pair, signatures in CONTROL_TARGETS.items():
        left, right = pair.split("-", 1)
        if _canonical_pair(left, right) == canonical_pair:
            return signatures
    return None


def _expected_receivers(canonical_pair: str) -> set[str]:
    direct = CONTROL_EXPECTED_RECEIVERS.get(canonical_pair)
    if direct is not None:
        return set(direct)
    for pair, receivers in CONTROL_EXPECTED_RECEIVERS.items():
        left, right = pair.split("-", 1)
        if _canonical_pair(left, right) == canonical_pair:
            return set(receivers)
    return set()


def _residualize_by_group(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups, dtype=str)
    residual = values.copy()
    for group in np.unique(groups):
        mask = groups == group
        residual[mask] = values[mask] - float(np.nanmean(values[mask]))
    return np.nan_to_num(residual, nan=0.0)


def _canonical_pair(a: object, b: object) -> str:
    left, right = sorted([str(a), str(b)])
    return f"{left}-{right}"


def _unique(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


if __name__ == "__main__":
    main()
