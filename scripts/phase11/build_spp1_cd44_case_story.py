#!/usr/bin/env python
"""Build SPP1-CD44 biological case source tables for Visium HD CRC."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE10_DIR = PROJECT_ROOT / "outputs/phase10/full_transcriptome_commot_style/visium_hd_crc_p2crc_subset"
PHASE11_DIR = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story/visium_hd_crc_p2crc_subset"
DATASET_DIR = (
    PROJECT_ROOT
    / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/visium_hd_crc_p2crc_subset"
)

TARGET_GENES = ["APOE", "CD68", "COL1A1", "FN1", "C1QA", "C1QB", "C1QC", "MSR1", "LYZ", "TREM2", "MRC1", "LST1", "CD14"]
MAP_GENES = ["SPP1", "CD44", "APOE", "CD68", "COL1A1", "FN1", "C1QA", "C1QB", "C1QC"]
CONTROL_RANDOM_SEED = 10403
EXPECTED_SPP1_CD44_RECEIVERS = {"caf_proxy", "myeloid_proxy", "spp1_macrophage_proxy"}


def main() -> None:
    out = PHASE11_DIR / "results"
    out.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(DATASET_DIR / "cells.csv")
    incoming = pd.read_csv(PHASE10_DIR / "results/receiver_level_incoming_scores.csv")
    candidates = pd.read_csv(DATASET_DIR / "candidate_interactions.csv")
    lr_pairs = pd.read_csv(DATASET_DIR / "lr_pairs.csv")
    counts, genes, barcodes = load_expression()
    expr = extract_expression(counts, genes, MAP_GENES + TARGET_GENES, barcodes)
    spatialself_incoming = incoming[incoming["method"].eq("spatialself")].set_index("cell_id")
    commot_incoming = incoming[incoming["method"].eq("commot")].set_index("cell_id")

    case = cells.set_index("cell_id").join(expr, how="left")
    case["spatialself_spp1_cd44_incoming"] = spatialself_incoming["SPP1-CD44"].reindex(case.index).fillna(0.0)
    case["commot_spp1_cd44_incoming"] = commot_incoming["SPP1-CD44"].reindex(case.index).fillna(0.0)
    control_spatial = build_spatial_control_maps(case, incoming, candidates, lr_pairs, controls_seed=CONTROL_RANDOM_SEED)
    case = case.join(control_spatial, how="left")
    present_targets = [g for g in TARGET_GENES if g in case.columns]
    case["target_signature"] = zscore_frame(case[present_targets]).mean(axis=1)
    case["sender_signature"] = zscore_frame(case[[g for g in ["SPP1", "CD68", "C1QA", "C1QB", "C1QC"] if g in case.columns]]).mean(axis=1)
    case["receiver_signature"] = zscore_frame(case[[g for g in ["CD44", "COL1A1", "FN1"] if g in case.columns]]).mean(axis=1)
    case = case.reset_index()

    edge_focus = build_edge_focus(candidates, lr_pairs, cells)
    niche_summary = build_niche_summary(case)
    target_summary = build_target_summary(case, present_targets)
    edge_summary = build_edge_summary(edge_focus)
    controls = pd.read_csv(PHASE10_DIR / "results/spp1_cd44_control_specificity_summary.csv")

    case.to_csv(out / "spp1_cd44_case_spatial_features.csv", index=False)
    niche_summary.to_csv(out / "spp1_cd44_case_niche_summary.csv", index=False)
    target_summary.to_csv(out / "spp1_cd44_case_target_gene_summary.csv", index=False)
    edge_summary.to_csv(out / "spp1_cd44_case_edge_summary.csv", index=False)
    controls.to_csv(out / "spp1_cd44_case_control_summary.csv", index=False)
    manifest = {
        "dataset": "visium_hd_crc_p2crc_subset",
        "focus_lr": "SPP1-CD44",
        "map_genes": MAP_GENES,
        "target_genes_used": present_targets,
        "outputs": [
            "spp1_cd44_case_spatial_features.csv",
            "spp1_cd44_case_niche_summary.csv",
            "spp1_cd44_case_target_gene_summary.csv",
            "spp1_cd44_case_edge_summary.csv",
            "spp1_cd44_case_control_summary.csv",
        ],
        "spatial_control_maps": [
            "spatialself_high_expression_negative_incoming",
            "spatialself_shuffled_receiver_spp1_cd44",
            "spatialself_shuffled_lr_identity_incoming",
            "spatialself_wrong_niche_spp1_cd44",
            "spatialself_wrong_receiver_spp1_cd44",
            "spatialself_wrong_type_spp1_cd44",
        ],
        "claim_boundary": "This is an observational spatial consistency story: SPP1-CD44 incoming, sender/receiver expression, target signature and controls. It is not perturbational proof.",
    }
    (out / "spp1_cd44_case_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(niche_summary.sort_values("mean_spatialself_spp1_cd44_incoming", ascending=False).to_string(index=False))
    print()
    print(target_summary.to_string(index=False))


def load_expression() -> tuple[sparse.csr_matrix, list[str], list[str]]:
    counts = sparse.load_npz(PHASE10_DIR / "full_transcriptome/expression_full_transcriptome_counts.npz").tocsr()
    genes = pd.read_csv(PHASE10_DIR / "full_transcriptome/genes.csv")["gene"].astype(str).tolist()
    barcodes = pd.read_csv(PHASE10_DIR / "full_transcriptome/barcodes.csv")["cell_id"].astype(str).tolist()
    return counts, genes, barcodes


def extract_expression(counts: sparse.csr_matrix, genes: list[str], wanted: list[str], barcodes: list[str]) -> pd.DataFrame:
    gene_index = {gene: i for i, gene in enumerate(genes)}
    data = {}
    for gene in dict.fromkeys(wanted):
        if gene in gene_index:
            data[gene] = np.log1p(counts[:, gene_index[gene]].toarray().ravel())
    return pd.DataFrame(data, index=barcodes)


def build_spatial_control_maps(
    case: pd.DataFrame,
    incoming: pd.DataFrame,
    candidates: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    controls_seed: int,
) -> pd.DataFrame:
    """Build receiver-level spatial control maps matching the Phase 10 control definitions."""
    index = case.index.astype(str)
    method_incoming = incoming[incoming["method"].eq("spatialself")].set_index("cell_id").reindex(index)
    signal = method_incoming["SPP1-CD44"].fillna(0.0).to_numpy(dtype=float)

    negative_pairs = lr_pairs.loc[
        lr_pairs["false_positive_high_expression"].astype(str).str.lower().eq("true"),
        "control_name",
    ].astype(str).tolist()
    present_negative_pairs = [pair for pair in negative_pairs if pair in method_incoming.columns]
    if present_negative_pairs:
        high_expression_negative = method_incoming[present_negative_pairs].fillna(0.0).max(axis=1).to_numpy(dtype=float)
    else:
        high_expression_negative = np.zeros(len(index), dtype=float)

    rng = np.random.default_rng(controls_seed)
    shuffle_source = choose_shuffled_lr_source(method_incoming, focus_pair="SPP1-CD44", preferred="CD274-PDCD1", rng=rng)

    focus_lr_id = lr_pairs.loc[lr_pairs["control_name"].eq("SPP1-CD44"), "lr_id"].iloc[0]
    lr_candidates = candidates[candidates["lr_id"].eq(focus_lr_id)].copy()
    receiver_index = {cell_id: i for i, cell_id in enumerate(index)}

    def masked_signal(mask: pd.Series) -> np.ndarray:
        out = np.zeros(len(index), dtype=float)
        selected = lr_candidates[mask].copy()
        if selected.empty:
            return out
        represented = selected["receiver_id"].astype(str).map(receiver_index).dropna().astype(int).unique()
        out[represented] = signal[represented]
        return out

    wrong_receiver = ~case["cell_type"].astype(str).isin(EXPECTED_SPP1_CD44_RECEIVERS)
    wrong_receiver_signal = signal.copy()
    wrong_receiver_signal[~wrong_receiver.to_numpy()] = 0.0

    return pd.DataFrame(
        {
            "spatialself_high_expression_negative_incoming": high_expression_negative,
            "spatialself_shuffled_receiver_spp1_cd44": rng.permutation(signal),
            "spatialself_shuffled_lr_identity_incoming": method_incoming[shuffle_source].fillna(0.0).to_numpy(dtype=float)
            if shuffle_source in method_incoming.columns
            else np.zeros(len(index), dtype=float),
            "spatialself_wrong_niche_spp1_cd44": masked_signal(~lr_candidates["niche_rule_satisfied"].astype(bool)),
            "spatialself_wrong_receiver_spp1_cd44": wrong_receiver_signal,
            "spatialself_wrong_type_spp1_cd44": masked_signal(~lr_candidates["type_rule_satisfied"].astype(bool)),
        },
        index=index,
    )


def choose_shuffled_lr_source(method_incoming: pd.DataFrame, focus_pair: str, preferred: str, rng: np.random.Generator) -> str:
    numeric_pairs = [
        col
        for col in method_incoming.columns
        if col != focus_pair and pd.api.types.is_numeric_dtype(method_incoming[col])
    ]
    if preferred in numeric_pairs:
        return preferred
    if not numeric_pairs:
        return ""
    return str(rng.choice(numeric_pairs))


def zscore_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(index=frame.index)
    values = frame.astype(float)
    return (values - values.mean(axis=0)) / values.std(axis=0).replace(0, np.nan)


def build_edge_focus(candidates: pd.DataFrame, lr_pairs: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    spp1_id = lr_pairs.loc[lr_pairs["control_name"].eq("SPP1-CD44"), "lr_id"].iloc[0]
    edges = candidates[candidates["lr_id"].eq(spp1_id)].copy()
    pred_path = (
        PROJECT_ROOT
        / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/"
        / "visium_hd_crc_p2crc_spatialself_calibrated/spatialself_wavelet_edge_prior/"
        / "visium_hd_crc_p2crc_subset__spatialself_wavelet_predictions.csv"
    )
    pred = pd.read_csv(pred_path)
    edges = edges.merge(pred[["candidate_id", "score"]], on="candidate_id", how="left")
    cell_meta = cells[["cell_id", "x", "y", "cell_type"]]
    edges = edges.merge(cell_meta.add_prefix("sender_"), left_on="sender_id", right_on="sender_cell_id", how="left")
    edges = edges.merge(cell_meta.add_prefix("receiver_"), left_on="receiver_id", right_on="receiver_cell_id", how="left")
    return edges


def build_niche_summary(case: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for niche, sub in case.groupby("niche", sort=False):
        rows.append(
            {
                "niche": niche,
                "n_cells": int(len(sub)),
                "mean_spatialself_spp1_cd44_incoming": float(sub["spatialself_spp1_cd44_incoming"].mean()),
                "q90_spatialself_spp1_cd44_incoming": float(sub["spatialself_spp1_cd44_incoming"].quantile(0.9)),
                "mean_commot_spp1_cd44_incoming": float(sub["commot_spp1_cd44_incoming"].mean()),
                "mean_SPP1": float(sub["SPP1"].mean()) if "SPP1" in sub else np.nan,
                "mean_CD44": float(sub["CD44"].mean()) if "CD44" in sub else np.nan,
                "mean_target_signature": float(sub["target_signature"].mean()),
                "mean_sender_signature": float(sub["sender_signature"].mean()),
                "mean_receiver_signature": float(sub["receiver_signature"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_target_summary(case: pd.DataFrame, target_genes: list[str]) -> pd.DataFrame:
    rows = []
    signal = case["spatialself_spp1_cd44_incoming"].to_numpy(dtype=float)
    target_sig = case["target_signature"].to_numpy(dtype=float)
    rows.append(
        {
            "feature": "target_signature",
            "spearman_with_spatialself_incoming": float(pd.Series(signal).corr(pd.Series(target_sig), method="spearman")),
            "mean_expression": float(np.nanmean(target_sig)),
            "q90_expression": float(np.nanquantile(target_sig, 0.9)),
        }
    )
    for gene in target_genes:
        values = case[gene].to_numpy(dtype=float)
        rows.append(
            {
                "feature": gene,
                "spearman_with_spatialself_incoming": float(pd.Series(signal).corr(pd.Series(values), method="spearman")),
                "mean_expression": float(np.nanmean(values)),
                "q90_expression": float(np.nanquantile(values, 0.9)),
            }
        )
    return pd.DataFrame(rows).sort_values("spearman_with_spatialself_incoming", ascending=False)


def build_edge_summary(edges: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sender_niche, receiver_niche), sub in edges.groupby(["sender_niche", "receiver_niche"], sort=False):
        rows.append(
            {
                "sender_niche": sender_niche,
                "receiver_niche": receiver_niche,
                "n_edges": int(len(sub)),
                "mean_score": float(pd.to_numeric(sub["score"], errors="coerce").mean()),
                "total_score": float(pd.to_numeric(sub["score"], errors="coerce").sum()),
                "type_rule_fraction": float(sub["type_rule_satisfied"].mean()),
                "niche_rule_fraction": float(sub["niche_rule_satisfied"].mean()),
                "median_distance": float(sub["distance"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values("total_score", ascending=False)


if __name__ == "__main__":
    main()
