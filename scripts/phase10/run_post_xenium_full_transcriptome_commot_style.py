#!/usr/bin/env python
"""Full-transcriptome COMMOT-style downstream validation for post-Xenium colon."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import fisher_exact, hypergeom, rankdata, t as student_t
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOAL5_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"

DATASET_DIR: Path
FULL_TX_DIR: Path
NICHENET_TARGETS_PATH: Path
PREDICTIONS: dict[str, Path]
FOCUS_LR: dict[str, str]
POSITIVE_LR: set[str]
NEGATIVE_LR: set[str]
DATASET_LABEL: str
DATASET_KEY: str
DEFAULT_OUTPUT_DIR: Path
CLAIM_BOUNDARY: str

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
    "TGFB1-TGFBR2": [
        "COL1A1",
        "COL1A2",
        "COL3A1",
        "COL6A1",
        "DCN",
        "LUM",
        "ACTA2",
        "TAGLN",
        "FN1",
        "SERPINE1",
        "THBS1",
        "CTGF",
        "MMP2",
        "TIMP1",
        "SPARC",
    ],
    "VEGFA-KDR": [
        "KDR",
        "FLT1",
        "PECAM1",
        "VWF",
        "ENG",
        "ESAM",
        "RAMP2",
        "EMCN",
        "CDH5",
        "ANGPT2",
        "SOX18",
        "PLVAP",
        "MCAM",
    ],
    "CXCL12-CXCR4": [
        "CXCR4",
        "CXCL12",
        "JUN",
        "FOS",
        "FOSB",
        "EGR1",
        "EGR2",
        "NR4A1",
        "DUSP1",
        "DUSP2",
        "SOCS3",
        "NFKBIA",
        "ICAM1",
        "MMP9",
        "PTGS2",
    ],
    "CD274-PDCD1": [
        "PDCD1",
        "CD274",
        "CD8A",
        "CD8B",
        "LAG3",
        "HAVCR2",
        "TIGIT",
        "CTLA4",
        "TOX",
        "IFNG",
        "GZMB",
        "PRF1",
        "CXCL9",
        "CXCL10",
        "STAT1",
        "IRF1",
    ],
    "ACTB-KRT8": ["EPCAM", "KRT8", "KRT18", "KRT19", "MUC1", "MUC13", "CLDN4", "TFF3"],
    "COL1A1-EPCAM": ["EPCAM", "KRT8", "KRT18", "KRT19", "MUC1", "MUC13", "CLDN4", "TFF3"],
}


def configure_dataset(dataset: str) -> dict[str, object]:
    """Set dataset-specific paths and LR panels while keeping one analysis implementation."""
    global DATASET_DIR
    global FULL_TX_DIR
    global NICHENET_TARGETS_PATH
    global PREDICTIONS
    global FOCUS_LR
    global POSITIVE_LR
    global NEGATIVE_LR
    global DATASET_LABEL
    global DATASET_KEY
    global DEFAULT_OUTPUT_DIR
    global CLAIM_BOUNDARY

    DATASET_KEY = dataset
    if dataset == "post_xenium":
        DATASET_LABEL = "post-Xenium human colon cancer 10x Visium CytAssist subset"
        dataset_slug = "post_xenium_human_colon_cancer_subset"
        DATASET_DIR = GOAL5_DIR / "standardized_inputs/post_xenium_human_colon_cancer_subset"
        FULL_TX_DIR = (
            PROJECT_ROOT
            / "outputs/phase10/full_transcriptome_commot_style/post_xenium_human_colon_cancer_subset/full_transcriptome"
        )
        NICHENET_TARGETS_PATH = (
            PROJECT_ROOT
            / "outputs/phase10/full_transcriptome_commot_style/post_xenium_human_colon_cancer_subset/"
            / "nichenet_targets/lr_pairs__nichenet_targets.csv"
        )
        PREDICTIONS = {
            "spatialself": GOAL5_DIR
            / "post_xenium_human_colon_cancer_spatialself_calibrated/spatialself_wavelet_edge_prior/"
            / "post_xenium_human_colon_cancer_subset__spatialself_wavelet_predictions.csv",
            "commot": GOAL5_DIR
            / "post_xenium_human_colon_cancer_baseline_panel/commot/"
            / "post_xenium_human_colon_cancer_subset__commot_predictions.csv",
            "cellchat": GOAL5_DIR
            / "post_xenium_human_colon_cancer_extended_baselines/cellchat/"
            / "post_xenium_human_colon_cancer_subset__cellchat_predictions.csv",
            "cellnest": GOAL5_DIR
            / "post_xenium_human_colon_cancer_extended_baselines/cellnest/"
            / "post_xenium_human_colon_cancer_subset__cellnest_predictions.csv",
            "spatalk": GOAL5_DIR
            / "post_xenium_human_colon_cancer_extended_baselines/spatalk/"
            / "post_xenium_human_colon_cancer_subset__spatalk_predictions.csv",
        }
        FOCUS_LR = {
            "SPP1-CD44": "post_xenium_crc_lr_001_SPP1_CD44",
            "TGFB1-TGFBR2": "post_xenium_crc_lr_000_TGFB1_TGFBR2",
            "VEGFA-KDR": "post_xenium_crc_lr_003_VEGFA_KDR",
            "ACTB-KRT8": "post_xenium_crc_lr_005_ACTB_KRT8_negative",
            "COL1A1-EPCAM": "post_xenium_crc_lr_006_COL1A1_EPCAM_negative",
        }
        POSITIVE_LR = {"SPP1-CD44", "TGFB1-TGFBR2", "VEGFA-KDR"}
        NEGATIVE_LR = {"ACTB-KRT8", "COL1A1-EPCAM"}
    elif dataset == "visium_hd_crc":
        DATASET_LABEL = "Visium HD CRC P2CRC 8 um bin subset"
        dataset_slug = "visium_hd_crc_p2crc_subset"
        DATASET_DIR = GOAL5_DIR / "standardized_inputs/visium_hd_crc_p2crc_subset"
        FULL_TX_DIR = PROJECT_ROOT / "outputs/phase10/full_transcriptome_commot_style/visium_hd_crc_p2crc_subset/full_transcriptome"
        NICHENET_TARGETS_PATH = (
            PROJECT_ROOT
            / "outputs/phase10/full_transcriptome_commot_style/visium_hd_crc_p2crc_subset/"
            / "nichenet_targets/lr_pairs__nichenet_targets.csv"
        )
        PREDICTIONS = {
            "spatialself": GOAL5_DIR
            / "visium_hd_crc_p2crc_spatialself_calibrated/spatialself_wavelet_edge_prior/"
            / "visium_hd_crc_p2crc_subset__spatialself_wavelet_predictions.csv",
            "commot": GOAL5_DIR
            / "visium_hd_crc_p2crc_baseline_panel/commot/"
            / "visium_hd_crc_p2crc_subset__commot_predictions.csv",
            "cellchat": GOAL5_DIR
            / "visium_hd_crc_p2crc_baseline_panel/cellchat/"
            / "visium_hd_crc_p2crc_subset__cellchat_predictions.csv",
            "cellnest": GOAL5_DIR
            / "visium_hd_crc_p2crc_baseline_panel/cellnest/"
            / "visium_hd_crc_p2crc_subset__cellnest_predictions.csv",
        }
        FOCUS_LR = {
            "SPP1-CD44": "visium_hd_crc_lr_000_SPP1_CD44",
            "CXCL12-CXCR4": "visium_hd_crc_lr_001_CXCL12_CXCR4",
            "VEGFA-KDR": "visium_hd_crc_lr_002_VEGFA_KDR",
            "TGFB1-TGFBR2": "visium_hd_crc_lr_003_TGFB1_TGFBR2",
            "CD274-PDCD1": "visium_hd_crc_lr_004_CD274_PDCD1",
            "ACTB-KRT8": "visium_hd_crc_lr_005_ACTB_KRT8_negative",
            "COL1A1-EPCAM": "visium_hd_crc_lr_006_COL1A1_EPCAM_negative",
        }
        POSITIVE_LR = {"SPP1-CD44", "CXCL12-CXCR4", "VEGFA-KDR", "TGFB1-TGFBR2", "CD274-PDCD1"}
        NEGATIVE_LR = {"ACTB-KRT8", "COL1A1-EPCAM"}
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    DEFAULT_OUTPUT_DIR = PROJECT_ROOT / f"outputs/phase10/full_transcriptome_commot_style/{dataset_slug}/results"
    CLAIM_BOUNDARY = (
        f"Full-transcriptome receiver-side association on a real {DATASET_LABEL}. "
        "Associations are observational and not perturbation evidence."
    )
    return {
        "dataset_key": DATASET_KEY,
        "dataset_label": DATASET_LABEL,
        "dataset_slug": dataset_slug,
        "dataset_dir": DATASET_DIR,
        "full_transcriptome": FULL_TX_DIR,
        "predictions": PREDICTIONS,
        "focus_lr": FOCUS_LR,
        "positive_lr": sorted(POSITIVE_LR),
        "negative_lr": sorted(NEGATIVE_LR),
        "default_output_dir": DEFAULT_OUTPUT_DIR,
    }


def json_ready(value):
    if isinstance(value, Path):
        return str(value.relative_to(PROJECT_ROOT)) if value.is_absolute() and PROJECT_ROOT in value.parents else str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["post_xenium", "visium_hd_crc"], default="post_xenium")
    parser.add_argument(
        "--output-dir",
        default="",
    )
    parser.add_argument("--min-cells", type=int, default=25)
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=10401)
    args = parser.parse_args()

    dataset_config = configure_dataset(args.dataset)
    out = PROJECT_ROOT / args.output_dir if args.output_dir else DEFAULT_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    counts, genes, barcodes = load_full_transcriptome()
    cells = pd.read_csv(DATASET_DIR / "cells.csv")
    candidates = pd.read_csv(DATASET_DIR / "candidate_interactions.csv")
    target_sources = load_target_sources()
    validate_order(cells, barcodes)

    expressed_mask = np.asarray((counts > 0).sum(axis=0)).ravel() >= args.min_cells
    expressed_gene_idx = np.flatnonzero(expressed_mask)
    expressed_genes = genes.loc[expressed_gene_idx, "gene"].astype(str).to_numpy()
    log_counts = counts[:, expressed_gene_idx].astype(np.float64)
    log_counts.data = np.log1p(log_counts.data)
    log_dense = np.asarray(log_counts.toarray(), dtype=np.float32)
    ranked_expression, valid_gene_mask = rank_expression_matrix(log_dense)
    expressed_genes = expressed_genes[valid_gene_mask]
    log_dense = log_dense[:, valid_gene_mask]
    rng = np.random.default_rng(args.random_seed)

    scan_frames = []
    enrichment_rows = []
    specificity_rows = []
    impact_rows = []
    incoming_frames = []
    available_predictions = {method: path for method, path in PREDICTIONS.items() if path.exists()}
    missing_predictions = {method: str(path) for method, path in PREDICTIONS.items() if not path.exists()}
    if not available_predictions:
        raise FileNotFoundError("No prediction files were found for this dataset.")
    for method, prediction_path in PREDICTIONS.items():
        if not prediction_path.exists():
            print(f"Skipping {method}: prediction file not found at {prediction_path}")
            continue
        predictions = pd.read_csv(prediction_path)
        incoming = build_incoming(candidates, predictions, cells["cell_id"].astype(str).tolist())
        incoming["method"] = method
        incoming_frames.append(incoming)
        method_curated_enrichment_rows = []
        for pair, signal in incoming_signal_iter(incoming):
            status = "positive" if pair in POSITIVE_LR else "negative_control"
            scan = scan_genes(method, pair, status, signal, ranked_expression, expressed_genes, rng, args.n_permutations)
            scan_frames.append(scan)
            for target_source, target_map in target_sources.items():
                enrichment_row = enrich_targets(
                    method,
                    pair,
                    status,
                    scan,
                    expressed_genes,
                    args.top_n,
                    target_source,
                    target_map.get(pair, []),
                )
                enrichment_rows.append(enrichment_row)
                if target_source == "curated":
                    method_curated_enrichment_rows.append(enrichment_row)
            impact_rows.append(communication_impact(method, pair, status, signal, log_dense, expressed_genes, cells, args.random_seed))
        specificity_rows.extend(build_specificity_rows(method, method_curated_enrichment_rows, impact_rows[-len(FOCUS_LR) :]))

    scan_all = pd.concat(scan_frames, ignore_index=True)
    enrichment = pd.DataFrame(enrichment_rows)
    specificity = pd.DataFrame(specificity_rows)
    impact = pd.DataFrame(impact_rows)
    incoming_all = pd.concat(incoming_frames, ignore_index=True)
    curated_enrichment = enrichment[enrichment["target_source"].eq("curated")].copy()
    story = build_story_summary(curated_enrichment, specificity, impact)

    scan_all.to_csv(out / "gene_association_scan_all.csv", index=False)
    scan_all.groupby(["method", "lr_pair"], sort=False).head(300).to_csv(
        out / "gene_association_top300_by_method_lr.csv", index=False
    )
    enrichment.to_csv(out / "target_enrichment_summary.csv", index=False)
    write_target_database_status(out, target_sources)
    specificity.to_csv(out / "downstream_specificity_summary.csv", index=False)
    impact.to_csv(out / "communication_impact_summary.csv", index=False)
    incoming_all.to_csv(out / "receiver_level_incoming_scores.csv", index=False)
    story.to_csv(out / "phase10_story_summary.csv", index=False)

    manifest = {
        "dataset": str(DATASET_DIR.relative_to(PROJECT_ROOT)),
        "dataset_key": DATASET_KEY,
        "dataset_label": DATASET_LABEL,
        "full_transcriptome": str(FULL_TX_DIR.relative_to(PROJECT_ROOT)),
        "methods": sorted(available_predictions),
        "missing_methods": missing_predictions,
        "focus_lr": FOCUS_LR,
        "positive_lr": sorted(POSITIVE_LR),
        "negative_lr": sorted(NEGATIVE_LR),
        "min_cells": args.min_cells,
        "n_expressed_genes_tested": int(len(expressed_genes)),
        "top_n_for_enrichment": args.top_n,
        "n_permutations": args.n_permutations,
        "target_sources": sorted(target_sources),
        "nichenet_targets": str(NICHENET_TARGETS_PATH.relative_to(PROJECT_ROOT)) if NICHENET_TARGETS_PATH.exists() else "",
        "claim_boundary": CLAIM_BOUNDARY,
        "dataset_config": json_ready(dataset_config),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_story_notes(out, story)

    print(story.to_string(index=False))
    print(json.dumps(manifest, indent=2))


def load_full_transcriptome() -> tuple[sparse.csr_matrix, pd.DataFrame, pd.DataFrame]:
    counts = sparse.load_npz(FULL_TX_DIR / "expression_full_transcriptome_counts.npz").tocsr()
    genes = pd.read_csv(FULL_TX_DIR / "genes.csv")
    barcodes = pd.read_csv(FULL_TX_DIR / "barcodes.csv")
    return counts, genes, barcodes


def load_target_sources() -> dict[str, dict[str, list[str]]]:
    sources = {"curated": {pair: list(genes) for pair, genes in CURATED_TARGETS.items()}}
    if NICHENET_TARGETS_PATH.exists():
        lr_lookup = {v: k for k, v in FOCUS_LR.items()}
        nichenet = {}
        table = pd.read_csv(NICHENET_TARGETS_PATH)
        for _, row in table.iterrows():
            pair = lr_lookup.get(str(row.get("lr_id", "")))
            if not pair:
                continue
            genes = parse_gene_list(row.get("target_genes", ""))
            nichenet[pair] = genes
        sources["nichenet"] = nichenet
    return sources


def parse_gene_list(value) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [gene.strip() for gene in text.replace(",", ";").split(";") if gene.strip()]


def validate_order(cells: pd.DataFrame, barcodes: pd.DataFrame) -> None:
    left = cells["cell_id"].astype(str).tolist()
    right = barcodes["cell_id"].astype(str).tolist()
    if left != right:
        raise ValueError("Cell order mismatch between cells.csv and full-transcriptome matrix.")


def build_incoming(candidates: pd.DataFrame, predictions: pd.DataFrame, cell_ids: list[str]) -> pd.DataFrame:
    table = candidates.merge(predictions[["candidate_id", "score"]], on="candidate_id", how="inner")
    out = pd.DataFrame({"cell_id": cell_ids})
    for pair, lr_id in FOCUS_LR.items():
        lr_table = table[table["lr_id"].eq(lr_id)]
        agg = lr_table.groupby("receiver_id", as_index=False)["score"].mean()
        agg = agg.rename(columns={"receiver_id": "cell_id", "score": pair})
        out = out.merge(agg, on="cell_id", how="left")
    pair_cols = list(FOCUS_LR)
    out[pair_cols] = out[pair_cols].fillna(0.0)
    return out


def incoming_signal_iter(incoming: pd.DataFrame):
    for pair in FOCUS_LR:
        yield pair, incoming[pair].to_numpy(dtype=float)


def scan_genes(
    method: str,
    pair: str,
    lr_status: str,
    signal: np.ndarray,
    ranked_expression: np.ndarray,
    genes: np.ndarray,
    rng: np.random.Generator,
    n_permutations: int,
) -> pd.DataFrame:
    if np.nanstd(signal) <= 1e-12:
        return pd.DataFrame(
            {
                "method": method,
                "lr_pair": pair,
                "lr_status": lr_status,
                "gene": genes,
                "spearman_r": np.nan,
                "spearman_p": np.nan,
                "bh_fdr": np.nan,
                "permutation_p_pos": np.nan,
                "permutation_bh_fdr_pos": np.nan,
                "n_permutations": n_permutations,
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
    df = pd.DataFrame(
        {
            "gene": genes,
            "spearman_r": rho.astype(float),
            "spearman_p": pvals.astype(float),
            "status": "tested",
        }
    )
    df.insert(0, "lr_status", lr_status)
    df.insert(0, "lr_pair", pair)
    df.insert(0, "method", method)
    mask = df["status"].eq("tested") & np.isfinite(df["spearman_p"])
    df["bh_fdr"] = np.nan
    df.loc[mask, "bh_fdr"] = bh_adjust(df.loc[mask, "spearman_p"].to_numpy(dtype=float))
    df["permutation_p_pos"] = np.nan
    df["permutation_bh_fdr_pos"] = np.nan
    df["n_permutations"] = n_permutations
    if n_permutations > 0:
        perm_p = permutation_p_positive(
            observed_rho=rho,
            signal_rank=signal_rank,
            ranked_expression=ranked_expression,
            rng=rng,
            n_permutations=n_permutations,
        )
        df.loc[mask, "permutation_p_pos"] = perm_p
        df.loc[mask, "permutation_bh_fdr_pos"] = bh_adjust(perm_p)
    return df.sort_values(["status", "spearman_r"], ascending=[True, False])


def permutation_p_positive(
    observed_rho: np.ndarray,
    signal_rank: np.ndarray,
    ranked_expression: np.ndarray,
    rng: np.random.Generator,
    n_permutations: int,
) -> np.ndarray:
    expr_centered = ranked_expression - ranked_expression.mean(axis=0, keepdims=True)
    expr_norm = np.sqrt(np.sum(expr_centered**2, axis=0))
    perms = np.vstack([rng.permutation(signal_rank) for _ in range(n_permutations)]).astype(np.float32)
    perms -= perms.mean(axis=1, keepdims=True)
    perm_norm = np.sqrt(np.sum(perms**2, axis=1, keepdims=True))
    perm_rho = (perms @ expr_centered) / (perm_norm * expr_norm[None, :])
    exceed = np.sum(perm_rho >= observed_rho[None, :], axis=0)
    return (exceed + 1) / (n_permutations + 1)


def enrich_targets(
    method: str,
    pair: str,
    lr_status: str,
    scan: pd.DataFrame,
    universe_genes: np.ndarray,
    top_n: int,
    target_source: str,
    target_genes: list[str],
) -> dict[str, object]:
    targets = set(target_genes) & set(universe_genes)
    tested = scan[scan["status"].eq("tested")].copy()
    tested = tested.sort_values(["spearman_r", "bh_fdr"], ascending=[False, True])
    top = set(tested.head(top_n)["gene"].astype(str))
    universe = set(tested["gene"].astype(str))
    overlap = len(top & targets)
    n_targets = len(targets & universe)
    n_top = len(top)
    n_universe = len(universe)
    if n_targets == 0 or n_top == 0:
        p_value = np.nan
        odds_ratio = np.nan
        fold = np.nan
    else:
        p_value = float(hypergeom.sf(overlap - 1, n_universe, n_targets, n_top))
        table = [[overlap, n_top - overlap], [n_targets - overlap, n_universe - n_targets - n_top + overlap]]
        odds_ratio = float(fisher_exact(table, alternative="greater")[0])
        fold = float((overlap / n_top) / (n_targets / n_universe))
    target_ranks = tested.reset_index(drop=True)
    target_ranks["rank"] = np.arange(1, len(target_ranks) + 1)
    target_rank_values = target_ranks[target_ranks["gene"].isin(targets)]["rank"]
    return {
        "method": method,
        "lr_pair": pair,
        "lr_status": lr_status,
        "target_source": target_source,
        "top_n": top_n,
        "n_universe": n_universe,
        "n_targets": n_targets,
        "target_overlap_top_n": overlap,
        "target_enrichment_p": p_value,
        "target_enrichment_fold": fold,
        "target_enrichment_odds_ratio": odds_ratio,
        "mean_target_rank": float(target_rank_values.mean()) if len(target_rank_values) else np.nan,
        "top_target_genes": ",".join([g for g in tested.head(top_n)["gene"].astype(str) if g in targets][:20]),
    }


def write_target_database_status(out: Path, target_sources: dict[str, dict[str, list[str]]]) -> None:
    rows = []
    for source, target_map in target_sources.items():
        for pair in FOCUS_LR:
            rows.append(
                {
                    "target_source": source,
                    "lr_pair": pair,
                    "n_targets_raw": len(target_map.get(pair, [])),
                    "status": "targets_found" if target_map.get(pair) else "no_targets_available",
                }
            )
    requested_sources = ["curated", "nichenet", "omnipath", "reactome", "kegg", "go", "cellchatdb"]
    for source in requested_sources:
        if source not in target_sources:
            rows.append(
                {
                    "target_source": source,
                    "lr_pair": "all",
                    "n_targets_raw": 0,
                    "status": "not_integrated_in_phase10_current_run",
                }
            )
    pd.DataFrame(rows).to_csv(out / "target_database_status.csv", index=False)


def build_specificity_rows(method: str, enrichment_rows: list[dict[str, object]], impact_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_pair = {row["lr_pair"]: row for row in enrichment_rows}
    impact_by_pair = {row["lr_pair"]: row for row in impact_rows}
    rows = []
    for pair in POSITIVE_LR:
        if pair not in by_pair:
            continue
        pos = by_pair[pair]
        pos_fold = safe_float(pos.get("target_enrichment_fold"))
        pos_mean_rank = safe_float(pos.get("mean_target_rank"))
        pos_delta_r2 = safe_float(impact_by_pair.get(pair, {}).get("delta_cv_r2"))

        neg_candidates = []
        for neg in NEGATIVE_LR:
            if neg not in by_pair:
                continue
            neg_row = by_pair[neg]
            neg_candidates.append(
                {
                    "lr_pair": neg,
                    "target_enrichment_fold": safe_float(neg_row.get("target_enrichment_fold")),
                    "mean_target_rank": safe_float(neg_row.get("mean_target_rank")),
                    "delta_cv_r2": safe_float(impact_by_pair.get(neg, {}).get("delta_cv_r2")),
                }
            )
        best_neg_fold = max([x["target_enrichment_fold"] for x in neg_candidates if np.isfinite(x["target_enrichment_fold"])], default=np.nan)
        best_neg_rank = min([x["mean_target_rank"] for x in neg_candidates if np.isfinite(x["mean_target_rank"])], default=np.nan)
        best_neg_delta_r2 = max([x["delta_cv_r2"] for x in neg_candidates if np.isfinite(x["delta_cv_r2"])], default=np.nan)
        best_neg_fold_pair = next((x["lr_pair"] for x in neg_candidates if x["target_enrichment_fold"] == best_neg_fold), "")
        best_neg_rank_pair = next((x["lr_pair"] for x in neg_candidates if x["mean_target_rank"] == best_neg_rank), "")
        best_neg_impact_pair = next((x["lr_pair"] for x in neg_candidates if x["delta_cv_r2"] == best_neg_delta_r2), "")
        rows.append(
            {
                "method": method,
                "positive_lr": pair,
                "positive_target_enrichment_fold": pos_fold,
                "best_negative_enrichment_lr": best_neg_fold_pair,
                "best_negative_target_enrichment_fold": best_neg_fold,
                "target_enrichment_specificity": pos_fold - best_neg_fold if np.isfinite(pos_fold) and np.isfinite(best_neg_fold) else np.nan,
                "positive_mean_target_rank": pos_mean_rank,
                "best_negative_rank_lr": best_neg_rank_pair,
                "best_negative_mean_target_rank": best_neg_rank,
                "target_rank_specificity": best_neg_rank - pos_mean_rank if np.isfinite(pos_mean_rank) and np.isfinite(best_neg_rank) else np.nan,
                "positive_delta_cv_r2": pos_delta_r2,
                "best_negative_impact_lr": best_neg_impact_pair,
                "best_negative_delta_cv_r2": best_neg_delta_r2,
                "impact_specificity_delta_r2": pos_delta_r2 - best_neg_delta_r2 if np.isfinite(pos_delta_r2) and np.isfinite(best_neg_delta_r2) else np.nan,
            }
        )
    return rows


def safe_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out


def communication_impact(
    method: str,
    pair: str,
    lr_status: str,
    signal: np.ndarray,
    log_counts: np.ndarray,
    genes: np.ndarray,
    cells: pd.DataFrame,
    seed: int,
) -> dict[str, object]:
    targets = [g for g in CURATED_TARGETS.get(pair, []) if g in set(genes)]
    if not targets or np.nanstd(signal) <= 1e-12:
        return {
            "method": method,
            "lr_pair": pair,
            "lr_status": lr_status,
            "n_targets": len(targets),
            "delta_cv_r2": np.nan,
            "rf_comm_importance": np.nan,
            "rf_comm_importance_rank": np.nan,
            "status": "not_evaluable",
        }
    idx = [int(np.where(genes == g)[0][0]) for g in targets]
    y = np.asarray(log_counts[:, idx].mean(axis=1)).ravel()
    lib = np.asarray(log_counts.sum(axis=1)).ravel()
    x_base = pd.DataFrame(
        {
            "library_log_sum": lib,
            "x": pd.to_numeric(cells["x"], errors="coerce").to_numpy(),
            "y": pd.to_numeric(cells["y"], errors="coerce").to_numpy(),
        }
    )
    x_full = x_base.copy()
    x_full["incoming"] = signal
    delta = cross_validated_delta_r2(x_base, x_full, y, seed)
    rf = random_forest_importance(x_full, y, seed)
    return {
        "method": method,
        "lr_pair": pair,
        "lr_status": lr_status,
        "n_targets": len(targets),
        "target_genes_present": ",".join(targets),
        **delta,
        **rf,
        "status": "tested",
    }


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
    return {
        "cv_r2_without_comm": base_mean,
        "cv_r2_with_comm": full_mean,
        "delta_cv_r2": full_mean - base_mean,
    }


def random_forest_importance(x_full: pd.DataFrame, y: np.ndarray, seed: int) -> dict[str, object]:
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=15,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_full, y)
    imp = pd.Series(model.feature_importances_, index=x_full.columns).sort_values(ascending=False)
    return {
        "rf_comm_importance": float(imp.get("incoming", 0.0)),
        "rf_comm_importance_rank": int(imp.index.get_loc("incoming") + 1) if "incoming" in imp.index else -1,
        "rf_top_features": ";".join([f"{k}:{v:.4f}" for k, v in imp.items()]),
    }


def build_story_summary(enrichment: pd.DataFrame, specificity: pd.DataFrame, impact: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in sorted(enrichment["method"].unique()):
        e = enrichment[(enrichment["method"].eq(method)) & (enrichment["lr_pair"].eq("SPP1-CD44"))]
        s = specificity[(specificity["method"].eq(method)) & (specificity["positive_lr"].eq("SPP1-CD44"))]
        i = impact[(impact["method"].eq(method)) & (impact["lr_pair"].eq("SPP1-CD44"))]
        if e.empty:
            continue
        er = e.iloc[0]
        sr = s.iloc[0] if not s.empty else {}
        ir = i.iloc[0] if not i.empty else {}
        rows.append(
            {
                "method": method,
                "spp1_cd44_target_overlap_top200": er["target_overlap_top_n"],
                "spp1_cd44_target_enrichment_fold": er["target_enrichment_fold"],
                "spp1_cd44_target_enrichment_p": er["target_enrichment_p"],
                "spp1_cd44_mean_target_rank": er["mean_target_rank"],
                "spp1_cd44_target_enrichment_specificity": sr.get("target_enrichment_specificity", np.nan),
                "spp1_cd44_target_rank_specificity": sr.get("target_rank_specificity", np.nan),
                "spp1_cd44_impact_specificity_delta_r2": sr.get("impact_specificity_delta_r2", np.nan),
                "spp1_cd44_delta_cv_r2": ir.get("delta_cv_r2", np.nan),
                "spp1_cd44_rf_comm_importance": ir.get("rf_comm_importance", np.nan),
                "support_class": classify_story(er, sr, ir),
                "interpretation": interpret_story(er, sr, ir),
            }
        )
    return pd.DataFrame(rows)


def story_flags(enrichment_row: pd.Series, specificity_row, impact_row) -> dict[str, bool]:
    enrich_ok = pd.notna(enrichment_row["target_enrichment_p"]) and float(enrichment_row["target_enrichment_p"]) <= 0.05
    overlap_ok = int(enrichment_row["target_overlap_top_n"]) >= 2 if pd.notna(enrichment_row["target_overlap_top_n"]) else False
    spec_value = specificity_row.get("target_enrichment_specificity", np.nan) if hasattr(specificity_row, "get") else np.nan
    rank_spec_value = specificity_row.get("target_rank_specificity", np.nan) if hasattr(specificity_row, "get") else np.nan
    impact_spec_value = specificity_row.get("impact_specificity_delta_r2", np.nan) if hasattr(specificity_row, "get") else np.nan
    impact_value = impact_row.get("delta_cv_r2", np.nan) if hasattr(impact_row, "get") else np.nan
    return {
        "target_enrichment": bool(enrich_ok and overlap_ok),
        "enrichment_specificity": bool(pd.notna(spec_value) and float(spec_value) > 0),
        "rank_specificity": bool(pd.notna(rank_spec_value) and float(rank_spec_value) > 0),
        "communication_impact": bool(pd.notna(impact_value) and float(impact_value) > 0),
        "impact_specificity": bool(pd.notna(impact_spec_value) and float(impact_spec_value) > 0),
        "negative_enrichment_dominated": bool(pd.notna(spec_value) and float(spec_value) < 0),
        "negative_rank_dominated": bool(pd.notna(rank_spec_value) and float(rank_spec_value) < 0),
        "negative_impact_dominated": bool(pd.notna(impact_spec_value) and float(impact_spec_value) < 0),
    }


def classify_story(enrichment_row: pd.Series, specificity_row, impact_row) -> str:
    flags = story_flags(enrichment_row, specificity_row, impact_row)
    if (
        flags["target_enrichment"]
        and flags["enrichment_specificity"]
        and flags["rank_specificity"]
        and flags["communication_impact"]
        and flags["impact_specificity"]
    ):
        return "strong_specific_downstream_support"
    if flags["target_enrichment"] and (flags["enrichment_specificity"] or flags["rank_specificity"]):
        return "partial_specific_downstream_support"
    if flags["target_enrichment"] and (flags["negative_enrichment_dominated"] or flags["negative_rank_dominated"]):
        return "target_enrichment_with_negative_control_risk"
    if flags["target_enrichment"]:
        return "target_enrichment_only"
    if flags["enrichment_specificity"] or flags["rank_specificity"] or flags["impact_specificity"]:
        return "specificity_signal_without_target_enrichment"
    return "limited_full_transcriptome_downstream_support"


def interpret_story(enrichment_row: pd.Series, specificity_row, impact_row) -> str:
    support_class = classify_story(enrichment_row, specificity_row, impact_row)
    if support_class == "strong_specific_downstream_support":
        return "target enrichment is significant and stronger than negative controls, with communication-impact support"
    if support_class == "partial_specific_downstream_support":
        return "target enrichment with partial negative-control specificity; impact support is not complete"
    if support_class == "target_enrichment_with_negative_control_risk":
        return "target enrichment is significant, but high-expression negative controls remain competitive"
    if support_class == "target_enrichment_only":
        return "target enrichment support only"
    if support_class == "specificity_signal_without_target_enrichment":
        return "some specificity signal, but curated target enrichment is not significant"
    return "limited full-transcriptome downstream support"


def write_story_notes(out: Path, story: pd.DataFrame) -> None:
    text = [
        "# Phase 10 full-transcriptome COMMOT-style downstream validation",
        "",
        f"Dataset: {DATASET_LABEL}.",
        "",
        "This upgrades the previous selected-gene receiver-state check to all expressed genes in the raw 10x feature matrix.",
        "",
        "Main table:",
        story.to_string(index=False),
        "",
        "Interpretation rule: strong support requires target enrichment plus enrichment/rank/impact specificity over high-expression negative controls. Partial support is kept separate from full support to avoid overclaiming.",
        "",
        f"Claim boundary: {CLAIM_BOUNDARY}",
    ]
    (out / "phase10_story_notes.md").write_text("\n".join(text), encoding="utf-8")


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


def rank_expression_matrix(log_dense: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    std = log_dense.std(axis=0)
    valid = std > 1e-12
    ranked = np.apply_along_axis(rankdata, 0, log_dense[:, valid]).astype(np.float32)
    return ranked, valid


if __name__ == "__main__":
    main()
