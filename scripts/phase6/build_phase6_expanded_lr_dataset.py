#!/usr/bin/env python
"""Build a Phase 6 lymph-node dataset with an expanded LR universe."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spitialself.benchmarks.semisynthetic import (  # noqa: E402
    SemiSyntheticConfig,
    SemiSyntheticLRSpec,
    _candidate_range,
    _inject_lr_expression,
    _make_semisynthetic_candidates,
    _make_semisynthetic_truth_tables,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        default="outputs/phase4/manuscript_evidence_lock/goal1_real_background_v2_5rep/human_lymph_node_visium_replicate_00",
    )
    parser.add_argument("--output-dir", default="outputs/phase6/expanded_lr_universe/datasets")
    parser.add_argument("--seed", type=int, default=2606)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    source_dir = PROJECT_ROOT / args.source_dir
    output_root = PROJECT_ROOT / args.output_dir
    rep_dir = output_root / "human_lymph_node_visium_phase6_expanded_replicate_00"
    rep_dir.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(source_dir / "cells.csv")
    expression = pd.read_csv(source_dir / "expression.csv")
    specs = _phase6_specs()
    lr_pairs = pd.DataFrame(asdict(spec) for spec in specs)
    expression = _ensure_gene_columns(expression, lr_pairs, rng)

    config = SemiSyntheticConfig(
        dataset_name="human_lymph_node_visium_phase6_expanded",
        injection_set="phase6_expanded_lymph_node",
        output_dir=str(output_root),
        random_seed=int(args.seed),
        replicate_id=0,
        max_candidate_range_quantile=0.08,
        negative_edge_fraction=0.20,
        candidate_generation="sparse_sampled",
        max_candidates_per_lr=9500,
        max_candidates_per_sender_lr=24,
        expression_boost_strong=6.0,
        expression_boost_weak=1.7,
    )
    expression = _inject_lr_expression(expression, cells, lr_pairs, config, rng)
    candidate_radius = _candidate_range(cells[["x", "y"]].to_numpy(dtype=float), config.max_candidate_range_quantile)
    candidates = _make_semisynthetic_candidates(cells, expression, lr_pairs, candidate_radius, config, rng)
    truth = _make_semisynthetic_truth_tables(candidates, lr_pairs, candidate_radius)

    cells.to_csv(rep_dir / "cells.csv", index=False)
    expression.to_csv(rep_dir / "expression.csv", index=False)
    lr_pairs.to_csv(rep_dir / "lr_pairs.csv", index=False)
    candidates.to_csv(rep_dir / "candidate_interactions.csv", index=False)
    for name, table in truth.items():
        table.to_csv(rep_dir / f"{name}.csv", index=False)
    if (source_dir / "metadata.json").exists():
        shutil.copy2(source_dir / "metadata.json", rep_dir / "source_metadata.json")
    metadata = {
        "benchmark": "phase6_expanded_lr_universe",
        "source_dir": str(source_dir.relative_to(PROJECT_ROOT)),
        "dataset_name": "human_lymph_node_visium_phase6_expanded",
        "replicate": 0,
        "purpose": "Enable strict SpatialSelf top10 versus retained-baseline rank >10 LR-pair screening.",
        "counts": {
            "cells": int(len(cells)),
            "lr_pairs": int(len(lr_pairs)),
            "candidate_interactions": int(len(candidates)),
            "true_lr_pairs": int(lr_pairs["true_lr"].astype(bool).sum()),
            "weak_lr_pairs": int(lr_pairs["weak_signal"].astype(bool).sum()),
            "false_positive_high_expression_lr_pairs": int(
                lr_pairs["false_positive_high_expression"].astype(bool).sum()
            ),
            "truth_edges": int(truth["ground_truth_table"]["true_edge"].sum()),
        },
        "selection_rule": (
            "LR pairs include known lymph-node positives, weak/context-specific positives, "
            "neutral background LR-like pairs, and high-expression false-positive decoys. "
            "This expanded universe is for Phase 6 rank-discovery screening only."
        ),
    }
    (rep_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame({"replicate": [0], "path": [str(rep_dir)]}).to_csv(output_root / "manifest.csv", index=False)
    print(json.dumps({"replicate_dir": str(rep_dir), **metadata["counts"]}, indent=2))


def _phase6_specs() -> tuple[SemiSyntheticLRSpec, ...]:
    specs: list[SemiSyntheticLRSpec] = [
        SemiSyntheticLRSpec(
            "CCL19_CCR7_Tzone_semisynthetic",
            "CCL19",
            "CCR7",
            "long_range_chemokine_niche",
            ("T", "B", "Stromal", "FDC", "Endo"),
            ("T", "DC"),
            ("T cell zone", "T_zone"),
            3.0,
        ),
        SemiSyntheticLRSpec(
            "CXCL13_CXCR5_GC_semisynthetic_weak",
            "CXCL13",
            "CXCR5",
            "long_range_chemokine_niche",
            ("FDC", "B"),
            ("B", "T"),
            ("Germinal center", "Germinal_center", "B follicle"),
            2.4,
            weak_signal=True,
        ),
        SemiSyntheticLRSpec(
            "CCL21_CCR7_Tzone_phase6_weak",
            "CCL21",
            "CCR7",
            "long_range_chemokine_niche",
            ("Stromal", "FDC", "Endo"),
            ("T", "DC"),
            ("T cell zone", "T_zone"),
            2.7,
            weak_signal=True,
        ),
        SemiSyntheticLRSpec(
            "CXCL12_CXCR4_follicle_phase6_context",
            "CXCL12",
            "CXCR4",
            "long_range_chemokine_niche",
            ("Stromal", "FDC", "B"),
            ("B", "T"),
            ("B follicle", "Germinal center", "Germinal_center"),
            2.2,
            weak_signal=True,
        ),
        SemiSyntheticLRSpec(
            "MIF_CD74_boundary_phase6_context",
            "MIF",
            "CD74",
            "short_medium_paracrine",
            ("Macrophage", "B", "T"),
            ("B", "T", "DC"),
            ("Mixed", "Boundary", "T cell zone", "B follicle"),
            1.4,
        ),
        SemiSyntheticLRSpec(
            "SPP1_CD44_boundary_semisynthetic",
            "SPP1",
            "CD44",
            "short_medium_paracrine",
            ("T", "B", "Macrophage", "Myeloid"),
            ("T", "B", "Stromal", "Other"),
            ("Mixed", "Boundary"),
            1.4,
        ),
        SemiSyntheticLRSpec(
            "CD48_CD244_Tcell_phase6_contact_weak",
            "CD48",
            "CD244",
            "juxtacrine_contact",
            ("B", "T", "DC"),
            ("T",),
            ("T cell zone", "Mixed"),
            0.7,
            weak_signal=True,
        ),
    ]
    neutral_pairs = [
        ("LTB_TNFRSF1A_background", "LTB", "TNFRSF1A"),
        ("CD47_SIRPA_background", "CD47", "SIRPA"),
        ("APP_CD74_background", "APP", "CD74"),
        ("HLA_DRA_CD4_background", "HLA-DRA", "CD4"),
        ("ICAM3_ITGAL_background", "ICAM3", "ITGAL"),
        ("SELL_SELPLG_background", "SELL", "SELPLG"),
        ("IL16_CD4_background", "IL16", "CD4"),
        ("CX3CL1_CX3CR1_background", "CX3CL1", "CX3CR1"),
        ("FLT3LG_FLT3_background", "FLT3LG", "FLT3"),
        ("GAS6_MERTK_background", "GAS6", "MERTK"),
    ]
    for lr_id, ligand, receptor in neutral_pairs:
        specs.append(
            SemiSyntheticLRSpec(
                lr_id,
                ligand,
                receptor,
                "background_lr_like_decoy",
                ("B", "T", "FDC", "Macrophage", "Stromal", "DC", "Other"),
                ("B", "T", "FDC", "Macrophage", "Stromal", "DC", "Other"),
                ("any",),
                1.8,
                true_lr=False,
            )
        )
    fp_pairs = [
        ("ACTB_EGFR_false_positive_phase6", "ACTB", "EGFR"),
        ("GAPDH_ITGB1_false_positive_phase6", "GAPDH", "ITGB1"),
        ("RPLP0_CD44_false_positive_phase6", "RPLP0", "CD44"),
        ("UBC_IL6ST_false_positive_phase6", "UBC", "IL6ST"),
        ("EEF1A1_CXCR4_false_positive_phase6", "EEF1A1", "CXCR4"),
        ("B2M_TNFRSF1A_false_positive_phase6", "B2M", "TNFRSF1A"),
        ("MALAT1_EGFR_false_positive_phase6", "MALAT1", "EGFR"),
        ("FTL_ITGB1_false_positive_phase6", "FTL", "ITGB1"),
    ]
    for lr_id, ligand, receptor in fp_pairs:
        specs.append(
            SemiSyntheticLRSpec(
                lr_id,
                ligand,
                receptor,
                "false_positive_high_expression",
                ("B", "T", "FDC", "Macrophage", "Endo", "DC", "Stromal", "Other"),
                ("B", "T", "FDC", "Macrophage", "Endo", "DC", "Stromal", "Other"),
                ("any",),
                2.8,
                false_positive_high_expression=True,
                true_lr=False,
            )
        )
    return tuple(specs)


def _ensure_gene_columns(expression: pd.DataFrame, lr_pairs: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    expr = expression.copy()
    existing = [column for column in expr.columns if column != "cell_id"]
    low_mean = np.asarray(expr[existing].mean(axis=0), dtype=float)
    fallback_mean = max(float(np.nanmedian(low_mean)), 1.0)
    for gene in sorted(set(lr_pairs["ligand"]) | set(lr_pairs["receptor"])):
        if gene in expr.columns:
            continue
        expr[gene] = rng.poisson(lam=max(fallback_mean * 0.15, 0.05), size=len(expr)).astype(float)
    return expr


if __name__ == "__main__":
    main()
