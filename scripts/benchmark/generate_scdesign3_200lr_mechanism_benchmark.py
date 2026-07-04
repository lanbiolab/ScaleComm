#!/usr/bin/env python
"""Build 200-LR scDesign3 mechanism benchmark replicates from retained backgrounds."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spitialself.benchmarks.semisynthetic import (
    SemiSyntheticConfig,
    _candidate_range,
    _inject_lr_expression,
    _injection_specs,
    _make_semisynthetic_candidates,
    _make_semisynthetic_truth_tables,
    _real_background_summary,
)


DATASETS = {
    "lymph_scdesign3_200lr": {
        "source": "outputs/reference_based_benchmark/native_rerun/recovered_inputs/lymph/reference_lymph_node_scdesign3_replicate_00",
        "injection_set": "range_lymph_node_200",
        "dataset_name": "reference_lymph_node_scdesign3_200lr",
        "tissue": "lymph_node",
        "seed_offset": 11,
    },
    "melanoma_scdesign3_200lr": {
        "source": "outputs/reference_based_benchmark/native_rerun/recovered_inputs/melanoma/reference_melanoma_scdesign3_replicate_00",
        "injection_set": "range_melanoma_200",
        "dataset_name": "reference_melanoma_scdesign3_200lr",
        "tissue": "melanoma",
        "seed_offset": 29,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs/reference_based_benchmark/scdesign3_mechanism_200lr/datasets",
    )
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--max-candidates-per-lr", type=int, default=700)
    parser.add_argument("--max-candidates-per-sender-lr", type=int, default=32)
    parser.add_argument("--negative-edge-fraction", type=float, default=0.35)
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for dataset_id, cfg in DATASETS.items():
        source = PROJECT_ROOT / cfg["source"]
        replicate_dir = _build_dataset(
            dataset_id=dataset_id,
            source=source,
            output_dir=output_dir,
            injection_set=str(cfg["injection_set"]),
            dataset_name=str(cfg["dataset_name"]),
            tissue=str(cfg["tissue"]),
            seed=int(args.seed + cfg["seed_offset"]),
            max_candidates_per_lr=int(args.max_candidates_per_lr),
            max_candidates_per_sender_lr=int(args.max_candidates_per_sender_lr),
            negative_edge_fraction=float(args.negative_edge_fraction),
        )
        manifest_rows.append(
            {
                "dataset_id": dataset_id,
                "tissue": cfg["tissue"],
                "source_dir": str(source.relative_to(PROJECT_ROOT)),
                "replicate_dir": str(replicate_dir.relative_to(PROJECT_ROOT)),
                "injection_set": cfg["injection_set"],
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_dir / "scdesign3_200lr_manifest.csv", index=False)
    (output_dir / "scdesign3_200lr_manifest.json").write_text(
        json.dumps(manifest_rows, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(manifest.to_string(index=False))
    print(f"Wrote 200-LR scDesign3 mechanism benchmark to {output_dir}")


def _build_dataset(
    *,
    dataset_id: str,
    source: Path,
    output_dir: Path,
    injection_set: str,
    dataset_name: str,
    tissue: str,
    seed: int,
    max_candidates_per_lr: int,
    max_candidates_per_sender_lr: int,
    negative_edge_fraction: float,
) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"Missing source replicate: {source}")
    rng = np.random.default_rng(seed)
    cells = pd.read_csv(source / "cells.csv")
    expression_background = pd.read_csv(source / "expression.csv")
    source_meta = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    source_config = dict(source_meta.get("config", {}))
    config = SemiSyntheticConfig(
        **{
            **source_config,
            "dataset_name": dataset_name,
            "injection_set": injection_set,
            "output_dir": str(output_dir),
            "random_seed": seed,
            "replicate_id": 0,
            "candidate_generation": "range_stress",
            "max_candidates_per_lr": max_candidates_per_lr,
            "max_candidates_per_sender_lr": max_candidates_per_sender_lr,
            "negative_edge_fraction": negative_edge_fraction,
            "background_simulator": "scdesign3_reused",
        }
    )
    lr_pairs = pd.DataFrame(asdict(spec) for spec in _injection_specs(injection_set))
    expression = _prepare_expression(expression_background, lr_pairs, rng)
    expression = _inject_lr_expression(expression, cells, lr_pairs, config, rng)
    coords = cells[["x", "y"]].to_numpy(dtype=float)
    candidate_range = _candidate_range(coords, config.max_candidate_range_quantile)
    candidates = _make_semisynthetic_candidates(cells, expression, lr_pairs, candidate_range, config, rng)
    truth_tables = _make_semisynthetic_truth_tables(candidates, lr_pairs, candidate_range)

    rep_dir = output_dir / f"{dataset_name}_replicate_00"
    rep_dir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(rep_dir / "cells.csv", index=False)
    expression.to_csv(rep_dir / "expression.csv", index=False)
    lr_pairs.to_csv(rep_dir / "lr_pairs.csv", index=False)
    candidates.to_csv(rep_dir / "candidate_interactions.csv", index=False)
    truth_tables["ground_truth_table"].to_csv(rep_dir / "ground_truth_table.csv", index=False)
    truth_tables["truth_lr_pairs"].to_csv(rep_dir / "truth_lr_pairs.csv", index=False)
    truth_tables["truth_cell_type_pairs"].to_csv(rep_dir / "truth_cell_type_pairs.csv", index=False)
    truth_tables["truth_spatial_niches"].to_csv(rep_dir / "truth_spatial_niches.csv", index=False)

    metadata = {
        "benchmark": "scdesign3_reused_background_200lr_mechanism_benchmark",
        "benchmark_version": "v4_200lr_mechanism_panel",
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "tissue": tissue,
        "source_replicate": str(source.relative_to(PROJECT_ROOT)),
        "injection_set": injection_set,
        "config": asdict(config),
        "candidate_range": float(candidate_range),
        "real_background_summary": _real_background_summary(cells),
        "lr_panel_design": {
            "total_lr_pairs": 200,
            "true_lr_pairs": 120,
            "hard_negative_lr_pairs": 80,
            "true_per_mechanism": {
                "juxtacrine_contact": 30,
                "short_range_paracrine": 30,
                "medium_range_paracrine": 30,
                "long_range_chemokine_niche": 30,
            },
            "hard_negative_classes": {
                "false_positive_high_expression": 40,
                "wrong_niche_long_range": 20,
                "wrong_cell_type_medium_range": 20,
            },
        },
        "counts": {
            "cells": int(len(cells)),
            "genes": int(expression.shape[1] - 1),
            "lr_pairs": int(len(lr_pairs)),
            "candidate_interactions": int(len(candidates)),
            "truth_edges": int(candidates["ground_truth"].astype(int).sum()),
            "truth_lr_positive_pairs": int(truth_tables["truth_lr_pairs"]["true_lr_pair"].sum()),
            "weak_positive_interactions": int(candidates.loc[candidates["weak_signal"], "ground_truth"].sum()),
            "wrong_range_false_positive_candidates": int(candidates["wrong_range_false_positive"].astype(bool).sum()),
            "wrong_niche_candidates": int((~candidates["niche_rule_satisfied"].astype(bool)).sum()),
        },
    }
    (rep_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")
    _write_design_summary(rep_dir, lr_pairs, candidates, truth_tables["truth_lr_pairs"])
    return rep_dir


def _prepare_expression(expression: pd.DataFrame, lr_pairs: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    expr = expression.copy()
    existing_genes = [column for column in expr.columns if column != "cell_id"]
    if not existing_genes:
        raise ValueError("Expression table has no gene columns")
    base = expr[existing_genes].to_numpy(dtype=float)
    gene_mean = np.maximum(np.nanmean(base, axis=0), 0.05)
    required = sorted(set(lr_pairs["ligand"].astype(str)) | set(lr_pairs["receptor"].astype(str)))
    new_columns = {}
    for gene in required:
        if gene in expr.columns:
            continue
        sampled_mean = float(rng.choice(gene_mean))
        background = rng.poisson(lam=max(sampled_mean * 0.08, 0.02), size=len(expr))
        new_columns[gene] = background.astype(float)
    if new_columns:
        expr = pd.concat([expr, pd.DataFrame(new_columns, index=expr.index)], axis=1)
    ordered = ["cell_id"] + required + [gene for gene in existing_genes if gene not in required]
    return expr[ordered].copy()


def _write_design_summary(
    rep_dir: Path,
    lr_pairs: pd.DataFrame,
    candidates: pd.DataFrame,
    truth_lr_pairs: pd.DataFrame,
) -> None:
    lr_summary = lr_pairs.copy()
    lr_summary["range_class"] = lr_summary["mechanism"].map(_range_class)
    lr_summary.to_csv(rep_dir / "lr_panel_design.csv", index=False)
    candidate_summary = (
        candidates.groupby("mechanism", as_index=False)
        .agg(
            candidate_edges=("candidate_id", "count"),
            true_edges=("ground_truth", "sum"),
            wrong_range_candidates=("wrong_range_false_positive", "sum"),
            wrong_niche_candidates=("niche_rule_satisfied", lambda x: int((~x.astype(bool)).sum())),
        )
        .sort_values("mechanism")
    )
    candidate_summary.to_csv(rep_dir / "candidate_mechanism_summary.csv", index=False)
    truth_lr_pairs.to_csv(rep_dir / "truth_lr_pair_summary.csv", index=False)


def _range_class(mechanism: str) -> str:
    if mechanism == "juxtacrine_contact":
        return "contact"
    if mechanism == "short_range_paracrine":
        return "short"
    if mechanism in {"short_medium_paracrine", "medium_range_paracrine"}:
        return "medium"
    if mechanism == "long_range_chemokine_niche":
        return "long_niche"
    return "hard_negative"


if __name__ == "__main__":
    main()
