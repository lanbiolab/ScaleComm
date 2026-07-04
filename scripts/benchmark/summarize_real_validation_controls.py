#!/usr/bin/env python
"""Summarize positive/negative LR controls from real-validation outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def summarize_real_validation_controls(
    lr_pairs_path: Path,
    ranking_paths: list[Path],
    enrichment_paths: list[Path],
    downstream_paths: list[Path],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lr_pairs = pd.read_csv(lr_pairs_path)
    control_columns = [
        "lr_id",
        "true_lr",
        "weak_signal",
        "false_positive_high_expression",
        "active_regions",
    ]
    controls = lr_pairs[[col for col in control_columns if col in lr_pairs.columns]].copy()
    if "true_lr" not in controls.columns:
        controls["true_lr"] = True
    if "false_positive_high_expression" not in controls.columns:
        controls["false_positive_high_expression"] = False
    if "weak_signal" not in controls.columns:
        controls["weak_signal"] = False

    rankings = [_read_csv(path) for path in ranking_paths if path.exists()]
    enrichments = [_read_csv(path) for path in enrichment_paths if path.exists()]
    downstream = [_read_csv(path) for path in downstream_paths if path.exists()]
    ranking = pd.concat(rankings, ignore_index=True) if rankings else pd.DataFrame()
    enrichment = pd.concat(enrichments, ignore_index=True) if enrichments else pd.DataFrame()
    downstream_table = pd.concat(downstream, ignore_index=True) if downstream else pd.DataFrame()

    rows = []
    if not ranking.empty:
        merged = ranking.merge(controls, on="lr_id", how="left")
        for row in merged.itertuples(index=False):
            rows.append(
                {
                    "method": getattr(row, "method"),
                    "lr_id": getattr(row, "lr_id"),
                    "control_type": _control_type(row),
                    "true_lr": bool(getattr(row, "true_lr", False)),
                    "weak_signal": bool(getattr(row, "weak_signal", False)),
                    "false_positive_high_expression": bool(
                        getattr(row, "false_positive_high_expression", False)
                    ),
                    "global_rank": int(getattr(row, "global_rank")),
                    "mean_score": float(getattr(row, "mean_score")),
                    "top_region": getattr(row, "top_region"),
                    "top_region_mean_score": float(getattr(row, "top_region_mean_score")),
                    "best_region_enrichment_score": _best_region_value(
                        enrichment,
                        getattr(row, "method"),
                        getattr(row, "lr_id"),
                        "region_enrichment_score",
                    ),
                    "best_region_bh_fdr": _best_region_value(
                        enrichment,
                        getattr(row, "method"),
                        getattr(row, "lr_id"),
                        "bh_fdr",
                    ),
                    "downstream_spearman_r": _downstream_value(
                        downstream_table,
                        getattr(row, "method"),
                        getattr(row, "lr_id"),
                        "spearman_r",
                    ),
                    "downstream_bh_fdr": _downstream_value(
                        downstream_table,
                        getattr(row, "method"),
                        getattr(row, "lr_id"),
                        "bh_fdr",
                    ),
                    "downstream_status": _downstream_value(
                        downstream_table,
                        getattr(row, "method"),
                        getattr(row, "lr_id"),
                        "status",
                    ),
                }
            )

    control_table = pd.DataFrame(rows)
    summary = _method_summary(control_table)
    output_dir.mkdir(parents=True, exist_ok=True)
    control_table.to_csv(output_dir / "real_validation_control_lr_summary.csv", index=False)
    summary.to_csv(output_dir / "real_validation_control_method_summary.csv", index=False)
    return control_table, summary


def _read_csv(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    if "method" not in table.columns:
        table["method"] = path.stem
    return table


def _control_type(row: object) -> str:
    if bool(getattr(row, "false_positive_high_expression", False)):
        return "negative_high_expression"
    if bool(getattr(row, "weak_signal", False)):
        return "positive_weak"
    if bool(getattr(row, "true_lr", False)):
        return "positive"
    return "negative"


def _best_region_value(
    enrichment: pd.DataFrame,
    method: str,
    lr_id: str,
    column: str,
) -> float:
    if enrichment.empty or column not in enrichment.columns:
        return float("nan")
    table = enrichment[(enrichment["method"] == method) & (enrichment["lr_id"] == lr_id)]
    if table.empty:
        return float("nan")
    best_idx = table["region_enrichment_score"].astype(float).idxmax()
    return float(table.loc[best_idx, column])


def _downstream_value(
    downstream: pd.DataFrame,
    method: str,
    lr_id: str,
    column: str,
) -> object:
    if downstream.empty or column not in downstream.columns:
        return float("nan")
    table = downstream[(downstream["method"] == method) & (downstream["lr_id"] == lr_id)]
    if table.empty:
        return float("nan")
    value = table.iloc[0][column]
    if pd.isna(value):
        return float("nan")
    return value


def _method_summary(control_table: pd.DataFrame) -> pd.DataFrame:
    if control_table.empty:
        return pd.DataFrame()
    rows = []
    for method, group in control_table.groupby("method", sort=False):
        positives = group[group["true_lr"].astype(bool)]
        negatives = group[group["false_positive_high_expression"].astype(bool)]
        tested = positives[pd.to_numeric(positives["downstream_spearman_r"], errors="coerce").notna()]
        rows.append(
            {
                "method": method,
                "n_positive_controls": int(len(positives)),
                "n_negative_controls": int(len(negatives)),
                "mean_positive_rank": float(positives["global_rank"].mean()) if len(positives) else float("nan"),
                "mean_negative_rank": float(negatives["global_rank"].mean()) if len(negatives) else float("nan"),
                "best_negative_rank": float(negatives["global_rank"].min()) if len(negatives) else float("nan"),
                "negative_top3_count": int((negatives["global_rank"] <= 3).sum()) if len(negatives) else 0,
                "mean_positive_downstream_spearman": float(
                    pd.to_numeric(tested["downstream_spearman_r"], errors="coerce").mean()
                )
                if len(tested)
                else float("nan"),
                "mean_positive_region_enrichment": float(
                    pd.to_numeric(positives["best_region_enrichment_score"], errors="coerce").mean()
                )
                if len(positives)
                else float("nan"),
                "mean_negative_region_enrichment": float(
                    pd.to_numeric(negatives["best_region_enrichment_score"], errors="coerce").mean()
                )
                if len(negatives)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr-pairs", required=True)
    parser.add_argument("--ranking", action="append", default=[])
    parser.add_argument("--enrichment", action="append", default=[])
    parser.add_argument("--downstream", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    control_table, summary = summarize_real_validation_controls(
        Path(args.lr_pairs),
        [Path(path) for path in args.ranking],
        [Path(path) for path in args.enrichment],
        [Path(path) for path in args.downstream],
        Path(args.output_dir),
    )
    print(control_table.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
