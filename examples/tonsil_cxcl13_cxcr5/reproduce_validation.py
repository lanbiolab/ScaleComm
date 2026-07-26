"""Reproduce the tonsil CXCL13-CXCR5 receiver validation summary."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr


HERE = Path(__file__).resolve().parent
DEFAULT_TABLE = HERE / "data" / "receiver_validation_a1.csv"
EXPECTED = {
    "n_receivers": 2500.0,
    "n_top_receivers": 300.0,
    "target_response_lift": 0.8067007343679484,
    "mannwhitney_greater_p": 6.690272013172949e-81,
    "spearman_incoming_target": 0.6215619065547007,
    "spearman_p_value": 3.4031902907382826e-267,
    "top_receiver_gc_fraction": 0.8066666666666666,
    "remaining_receiver_gc_fraction": 0.2818181818181818,
    "tissue_gc_fraction": 0.3448,
    "gc_fold_enrichment": 2.339520494972931,
    "top_receiver_gc_count": 242.0,
    "remaining_receiver_gc_count": 620.0,
    "gc_fisher_greater_p": 4.0352806810850986e-69,
}


def compute_metrics(table_path: Path = DEFAULT_TABLE) -> dict[str, float]:
    table = pd.read_csv(table_path)
    required = {
        "section",
        "region",
        "incoming_score",
        "top_receiver",
        "target_score",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    top = table["top_receiver"].astype(str).str.lower().eq("true").to_numpy()
    incoming = pd.to_numeric(table["incoming_score"], errors="raise").to_numpy()
    target = pd.to_numeric(table["target_score"], errors="raise").to_numpy()
    gc = table["region"].astype(str).eq("germinal_center_proxy").to_numpy()

    top_gc_fraction = float(gc[top].mean())
    remaining_gc_fraction = float(gc[~top].mean())
    tissue_gc_fraction = float(gc.mean())
    spearman = spearmanr(incoming, target)
    fisher = fisher_exact(
        [
            [int(gc[top].sum()), int((~gc[top]).sum())],
            [int(gc[~top].sum()), int((~gc[~top]).sum())],
        ],
        alternative="greater",
    )
    return {
        "n_receivers": float(len(table)),
        "n_top_receivers": float(top.sum()),
        "target_response_lift": float(target[top].mean() - target[~top].mean()),
        "mannwhitney_greater_p": float(
            mannwhitneyu(target[top], target[~top], alternative="greater").pvalue
        ),
        "spearman_incoming_target": float(spearman.statistic),
        "spearman_p_value": float(spearman.pvalue),
        "top_receiver_gc_fraction": top_gc_fraction,
        "remaining_receiver_gc_fraction": remaining_gc_fraction,
        "tissue_gc_fraction": tissue_gc_fraction,
        "gc_fold_enrichment": top_gc_fraction / tissue_gc_fraction,
        "top_receiver_gc_count": float(gc[top].sum()),
        "remaining_receiver_gc_count": float(gc[~top].sum()),
        "gc_fisher_greater_p": float(fisher.pvalue),
    }


def validate_metrics(metrics: dict[str, float]) -> None:
    for name, expected in EXPECTED.items():
        atol = 0.0 if "_p" in name else 1e-12
        if not np.isclose(metrics[name], expected, rtol=1e-8, atol=atol):
            raise AssertionError(
                f"{name} differs from the versioned expectation: "
                f"observed={metrics[name]!r}, expected={expected!r}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tonsil_cxcl13_cxcr5_reproduction.csv"),
    )
    args = parser.parse_args()

    metrics = compute_metrics(args.table)
    validate_metrics(metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)

    print("ScaleComm tonsil reproduction completed")
    print(f"receivers: {int(metrics['n_receivers'])}")
    print(f"top_receivers: {int(metrics['n_top_receivers'])}")
    print(f"target_response_lift: {metrics['target_response_lift']:.6f}")
    print(f"mannwhitney_greater_p: {metrics['mannwhitney_greater_p']:.6e}")
    print(f"spearman_incoming_target: {metrics['spearman_incoming_target']:.6f}")
    print(f"spearman_p_value: {metrics['spearman_p_value']:.6e}")
    print(f"gc_fold_enrichment: {metrics['gc_fold_enrichment']:.6f}")
    print(f"gc_fisher_greater_p: {metrics['gc_fisher_greater_p']:.6e}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
