#!/usr/bin/env python
"""Export paper-linked CXCL13-CXCR5 receiver validation from a fresh A1 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr


PAIR = "CXCL13-CXCR5"
TOP_N = 300


def target_score(expression: pd.DataFrame, signatures: pd.DataFrame) -> pd.DataFrame:
    signature = signatures[
        signatures["ligand"].eq("CXCL13") & signatures["receptor"].eq("CXCR5")
    ]
    if signature.empty:
        raise ValueError("CXCL13-CXCR5 target signature is missing.")
    genes = [gene.strip() for gene in signature.iloc[0]["targets"].split(",")]
    genes = [gene for gene in genes if gene in expression.columns and gene != "CXCR5"]
    matrix = np.log1p(expression.set_index("cell_id")[genes].astype(float))
    z = (matrix - matrix.mean()) / matrix.std(ddof=0).replace(0, np.nan)
    result = z.mean(axis=1).rename("target_score").reset_index()
    result["target_genes_used"] = ",".join(genes)
    return result


def build_receiver_table(prepared: Path, prediction_path: Path) -> pd.DataFrame:
    cells = pd.read_csv(prepared / "cells.csv")
    expression = pd.read_csv(prepared / "expression.csv")
    candidates = pd.read_csv(prepared / "candidate_interactions.csv")
    signatures = pd.read_csv(prepared / "target_signatures.csv")
    prediction = pd.read_csv(prediction_path, usecols=["candidate_id", "score"])

    edges = candidates.merge(prediction, on="candidate_id", how="inner", validate="one_to_one")
    edges = edges[(edges["ligand"] + "-" + edges["receptor"]).eq(PAIR)]
    incoming = edges.groupby("receiver_id", as_index=False)["score"].mean()
    incoming = incoming.rename(columns={"receiver_id": "cell_id", "score": "incoming_score"})
    table = cells.merge(incoming, on="cell_id", how="left").merge(
        target_score(expression, signatures), on="cell_id", how="left"
    )
    table["incoming_score"] = table["incoming_score"].fillna(0.0)
    if len(table) < 2:
        raise ValueError("At least two receiver cells are required for validation.")
    top_n = min(TOP_N, len(table) - 1)
    table["receiver_rank"] = table["incoming_score"].rank(ascending=False, method="first").astype(int)
    table["top_receiver"] = table["receiver_rank"].le(top_n)
    table["top_n"] = top_n
    table["lr_pair"] = PAIR
    return table


def summarize(table: pd.DataFrame) -> dict[str, float | int]:
    top = table["top_receiver"].to_numpy(dtype=bool)
    target = table["target_score"].to_numpy(dtype=float)
    gc = table["niche"].eq("germinal_center_proxy").to_numpy(dtype=bool)
    spearman = spearmanr(table["incoming_score"], target)
    fisher = fisher_exact(
        [[int(gc[top].sum()), int((~gc[top]).sum())], [int(gc[~top].sum()), int((~gc[~top]).sum())]],
        alternative="greater",
    )
    return {
        "n_receivers": int(len(table)),
        "n_top_receivers": int(top.sum()),
        "target_response_lift": float(target[top].mean() - target[~top].mean()),
        "mannwhitney_greater_p": float(mannwhitneyu(target[top], target[~top], alternative="greater").pvalue),
        "spearman_incoming_target": float(spearman.statistic),
        "spearman_p_value": float(spearman.pvalue),
        "gc_fold_enrichment": float(gc[top].mean() / gc.mean()),
        "gc_fisher_greater_p": float(fisher.pvalue),
    }


def plot(table: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    colors = np.where(table["top_receiver"], "#1f5f99", "#d7dce1")
    ax.scatter(table["x"], table["y"], s=5, c=colors, linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlabel("spatial x")
    ax.set_ylabel("spatial y")
    ax.set_title("CXCL13-CXCR5 top receiver map")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    table = build_receiver_table(args.prepared_dir, args.prediction)
    metrics = summarize(table)
    table.to_csv(args.output_dir / "tonsil_a1_cxcl13_cxcr5_receiver_validation.csv", index=False)
    (args.output_dir / "tonsil_a1_cxcl13_cxcr5_validation_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    plot(table, args.output_dir / "tonsil_a1_cxcl13_cxcr5_receiver_map.png")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
