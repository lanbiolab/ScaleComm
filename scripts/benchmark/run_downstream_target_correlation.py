#!/usr/bin/env python
"""Validate LR scores against receiver-side downstream target signatures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spitialself.fdr import bh_adjust


def run_downstream_target_correlation(
    candidate_path: Path,
    prediction_path: Path,
    expression_path: Path,
    output_dir: Path,
    *,
    pathway_csv: Path | None,
    reactome_tf_ppr_csv: Path | None,
    trrust_dir: Path | None,
    score_column: str = "score",
    min_targets: int = 3,
    max_targets: int = 200,
    n_permutations: int = 100,
    random_seed: int = 2025,
) -> pd.DataFrame:
    candidates = pd.read_csv(candidate_path)
    predictions = pd.read_csv(prediction_path)
    expression = pd.read_csv(expression_path)

    table = _merge_candidate_predictions(candidates, predictions, score_column)
    target_map, target_status = _build_target_map(
        candidates,
        expression_columns=[col for col in expression.columns if col != "cell_id"],
        pathway_csv=pathway_csv,
        reactome_tf_ppr_csv=reactome_tf_ppr_csv,
        trrust_dir=trrust_dir,
        max_targets=max_targets,
    )
    results = _correlate_lr_targets(
        table,
        expression,
        target_map,
        target_status,
        score_column=score_column,
        min_targets=min_targets,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{prediction_path.stem}__downstream_target_correlation.csv"
    status_path = output_dir / f"{prediction_path.stem}__downstream_target_status.csv"
    summary_path = output_dir / f"{prediction_path.stem}__downstream_target_summary.json"
    results.to_csv(result_path, index=False)
    pd.DataFrame(target_status).to_csv(status_path, index=False)
    summary = {
        "candidate_path": str(candidate_path),
        "prediction_path": str(prediction_path),
        "expression_path": str(expression_path),
        "result_path": str(result_path),
        "status_path": str(status_path),
        "n_lr_pairs_tested": int((results["status"] == "tested").sum()) if len(results) else 0,
        "n_lr_pairs_total": int(candidates["lr_id"].nunique()),
        "n_permutations": int(n_permutations),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    return results


def _merge_candidate_predictions(
    candidates: pd.DataFrame,
    predictions: pd.DataFrame,
    score_column: str,
) -> pd.DataFrame:
    required_candidates = {"candidate_id", "receiver_id", "lr_id", "ligand", "receptor", "mechanism"}
    missing_candidates = required_candidates - set(candidates.columns)
    if missing_candidates:
        raise ValueError(f"Candidate table missing columns: {sorted(missing_candidates)}")
    required_predictions = {"candidate_id", score_column}
    missing_predictions = required_predictions - set(predictions.columns)
    if missing_predictions:
        raise ValueError(f"Prediction table missing columns: {sorted(missing_predictions)}")

    prediction_columns = ["candidate_id", score_column]
    if "method" in predictions.columns:
        prediction_columns.append("method")
    table = candidates.merge(
        predictions[prediction_columns],
        on="candidate_id",
        how="inner",
        validate="one_to_one",
    )
    table[score_column] = pd.to_numeric(table[score_column], errors="coerce").fillna(0.0)
    return table


def _build_target_map(
    candidates: pd.DataFrame,
    *,
    expression_columns: list[str],
    pathway_csv: Path | None,
    reactome_tf_ppr_csv: Path | None,
    trrust_dir: Path | None,
    max_targets: int,
) -> tuple[dict[str, list[str]], list[dict]]:
    expression_genes = set(expression_columns)
    lr_table = candidates[["lr_id", "ligand", "receptor", "mechanism"]].drop_duplicates()
    pathway_targets = _load_pathway_targets(pathway_csv) if pathway_csv else {}
    reactome_targets = _build_reactome_targets(reactome_tf_ppr_csv, trrust_dir) if reactome_tf_ppr_csv and trrust_dir else {}

    target_map: dict[str, list[str]] = {}
    status = []
    for row in lr_table.itertuples(index=False):
        key = (str(row.ligand), str(row.receptor))
        targets = pathway_targets.get(key, [])
        source = "pathway_csv"
        if not targets:
            targets = reactome_targets.get(str(row.receptor), [])
            source = "reactome_trrust"

        present = [gene for gene in _unique_preserve_order(targets) if gene in expression_genes]
        if max_targets > 0:
            present = present[:max_targets]
        target_map[str(row.lr_id)] = present
        status.append(
            {
                "lr_id": row.lr_id,
                "ligand": row.ligand,
                "receptor": row.receptor,
                "mechanism": row.mechanism,
                "target_source": source if targets else "none",
                "n_targets_raw": int(len(set(targets))),
                "n_targets_present": int(len(present)),
                "status": "usable" if present else "no_observed_targets",
            }
        )
    return target_map, status


def _load_pathway_targets(pathway_csv: Path) -> dict[tuple[str, str], list[str]]:
    if not pathway_csv.exists():
        return {}
    table = pd.read_csv(pathway_csv)
    if not {"ligand", "receptor"}.issubset(table.columns):
        return {}
    target_column = None
    for column in ["TGs", "target_genes", "targets", "TargetGenes"]:
        if column in table.columns:
            target_column = column
            break
    if target_column is None:
        return {}
    out: dict[tuple[str, str], list[str]] = {}
    for row in table.itertuples(index=False):
        ligand = str(getattr(row, "ligand"))
        receptor = str(getattr(row, "receptor"))
        targets = _split_gene_list(getattr(row, target_column))
        out.setdefault((ligand, receptor), []).extend(targets)
        out.setdefault((ligand.upper(), receptor.upper()), []).extend(targets)
    return out


def _build_reactome_targets(reactome_tf_ppr_csv: Path, trrust_dir: Path) -> dict[str, list[str]]:
    reactome = pd.read_csv(reactome_tf_ppr_csv)
    if not {"receptor", "tf"}.issubset(reactome.columns):
        return {}
    tf_target_cache: dict[str, list[str]] = {}
    receptor_targets: dict[str, list[str]] = {}
    for row in reactome.itertuples(index=False):
        receptors = _split_gene_list(getattr(row, "receptor"))
        tf = str(getattr(row, "tf"))
        targets = tf_target_cache.get(tf)
        if targets is None:
            targets = _load_trrust_targets(trrust_dir / f"{tf}.csv")
            tf_target_cache[tf] = targets
        for receptor in receptors:
            receptor_targets.setdefault(receptor, []).extend(targets)
    return {key: _unique_preserve_order(value) for key, value in receptor_targets.items()}


def _load_trrust_targets(path: Path) -> list[str]:
    if not path.exists():
        return []
    table = pd.read_csv(path)
    if "x" not in table.columns:
        return []
    return [str(value) for value in table["x"].dropna().tolist()]


def _correlate_lr_targets(
    table: pd.DataFrame,
    expression: pd.DataFrame,
    target_map: dict[str, list[str]],
    target_status: list[dict],
    *,
    score_column: str,
    min_targets: int,
    n_permutations: int,
    random_seed: int,
) -> pd.DataFrame:
    expression = expression.set_index("cell_id")
    status_by_lr = {entry["lr_id"]: entry for entry in target_status}
    rng = np.random.default_rng(random_seed)
    rows = []
    for lr_id, group in table.groupby("lr_id", sort=False):
        lr_id = str(lr_id)
        targets = target_map.get(lr_id, [])
        base = status_by_lr.get(lr_id, {})
        row = {
            "lr_id": lr_id,
            "ligand": group["ligand"].iloc[0],
            "receptor": group["receptor"].iloc[0],
            "mechanism": group["mechanism"].iloc[0],
            "method": str(group["method"].iloc[0]) if "method" in group.columns else "unknown",
            "n_edges": int(len(group)),
            "n_receiver_cells": int(group["receiver_id"].nunique()),
            "n_targets_present": int(len(targets)),
            "target_source": base.get("target_source", "none"),
            "spearman_r": float("nan"),
            "spearman_p_value": float("nan"),
            "empirical_p_value": float("nan"),
            "status": "not_tested",
        }
        if len(targets) < min_targets:
            row["status"] = "too_few_targets"
            rows.append(row)
            continue

        comm = group.groupby("receiver_id", sort=False)[score_column].mean()
        common_cells = expression.index.intersection(comm.index)
        if len(common_cells) < 4:
            row["status"] = "too_few_receiver_cells"
            rows.append(row)
            continue

        communication = comm.loc[common_cells].to_numpy(dtype=float)
        target_signature = np.log1p(expression.loc[common_cells, targets].to_numpy(dtype=float)).mean(axis=1)
        if np.nanstd(communication) <= 1e-12 or np.nanstd(target_signature) <= 1e-12:
            row["status"] = "constant_vector"
            rows.append(row)
            continue

        corr, p_value = spearmanr(communication, target_signature)
        exceed = 0
        for _ in range(int(n_permutations)):
            null_signature = rng.permutation(target_signature)
            null_corr, _ = spearmanr(communication, null_signature)
            if abs(float(null_corr)) >= abs(float(corr)):
                exceed += 1
        row.update(
            {
                "spearman_r": float(corr),
                "spearman_p_value": float(p_value),
                "empirical_p_value": float((exceed + 1.0) / (int(n_permutations) + 1.0)),
                "status": "tested",
            }
        )
        rows.append(row)

    result = pd.DataFrame(rows)
    tested = result["status"] == "tested"
    result["bh_fdr"] = np.nan
    if tested.any():
        result.loc[tested, "bh_fdr"] = bh_adjust(
            result.loc[tested, "empirical_p_value"].to_numpy(dtype=float)
        )
    return result.sort_values(["status", "bh_fdr", "spearman_r"], ascending=[True, True, False])


def _split_gene_list(value) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value)
    for sep in [";", "|"]:
        text = text.replace(sep, ",")
    return [item.strip() for item in text.split(",") if item.strip()]


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="candidate_interactions.csv path.")
    parser.add_argument("--predictions", required=True, help="Prediction CSV path.")
    parser.add_argument("--expression", required=True, help="expression.csv path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pathway-csv", default="downstreamgene_eva/valid_single_pathway.csv")
    parser.add_argument(
        "--reactome-tf-ppr-csv",
        default="downstreamgene_eva/scseqcomm/data/TF_PPR_REACTOME_human.csv",
    )
    parser.add_argument("--trrust-dir", default="downstreamgene_eva/scseqcomm/data/TF_TG_TRRUSTv2")
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--min-targets", type=int, default=3)
    parser.add_argument("--max-targets", type=int, default=200)
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=2025)
    args = parser.parse_args()

    results = run_downstream_target_correlation(
        Path(args.candidates),
        Path(args.predictions),
        Path(args.expression),
        Path(args.output_dir),
        pathway_csv=Path(args.pathway_csv) if args.pathway_csv else None,
        reactome_tf_ppr_csv=Path(args.reactome_tf_ppr_csv) if args.reactome_tf_ppr_csv else None,
        trrust_dir=Path(args.trrust_dir) if args.trrust_dir else None,
        score_column=args.score_column,
        min_targets=args.min_targets,
        max_targets=args.max_targets,
        n_permutations=args.n_permutations,
        random_seed=args.random_seed,
    )
    print(
        results[
            [
                "lr_id",
                "method",
                "n_targets_present",
                "spearman_r",
                "empirical_p_value",
                "bh_fdr",
                "status",
            ]
        ].to_string(index=False)
    )
    print(f"Wrote downstream target correlation outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
