#!/usr/bin/env python
"""High-vs-low receiver hotspot signature enrichment for MERFISH CCC axes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.current.run_mouse_colon_merfish_downstream_signature_validation import (
    CURATED_SIGNATURES,
    LR_SIGNATURE_MAP,
    STAGE_ORDER,
    find_replicates,
    infer_stage,
    read_metadata,
)
from src.spitialself.fdr import bh_adjust


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicate-root", default="outputs/current_mouse_colon_ibd_merfish/multislice_replicates")
    parser.add_argument("--prediction-root", default="outputs/current_mouse_colon_ibd_merfish/multislice_spatialself/predictions")
    parser.add_argument("--output-dir", default="outputs/current_mouse_colon_ibd_merfish/receiver_hotspot_signature_enrichment")
    parser.add_argument("--model-name", default="spatialself_wavelet_scale")
    parser.add_argument("--top-quantile", type=float, default=0.80)
    parser.add_argument("--bottom-quantile", type=float, default=0.20)
    parser.add_argument("--min-group-cells", type=int, default=5)
    parser.add_argument("--min-signature-genes", type=int, default=3)
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260620)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, replicate_dir in enumerate(find_replicates(Path(args.replicate_root))):
        metadata = read_metadata(replicate_dir)
        stage = metadata.get("stage", infer_stage(replicate_dir.name))
        pred_path = Path(args.prediction_root) / stage / args.model_name / f"{replicate_dir.name}__{args.model_name}_predictions.csv"
        if not pred_path.exists():
            continue
        rows.extend(
            validate_replicate(
                replicate_dir,
                pred_path,
                metadata=metadata,
                stage=stage,
                top_quantile=float(args.top_quantile),
                bottom_quantile=float(args.bottom_quantile),
                min_group_cells=int(args.min_group_cells),
                min_signature_genes=int(args.min_signature_genes),
                n_permutations=int(args.n_permutations),
                seed=int(args.seed) + idx,
            )
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        tested = result["status"].eq("tested")
        result["bh_fdr"] = np.nan
        if tested.any():
            result.loc[tested, "bh_fdr"] = bh_adjust(result.loc[tested, "empirical_p_value"].to_numpy(float))
        result = result.sort_values(["lr_pair", "stage_day", "replicate"])
    result.to_csv(output_dir / "per_replicate_receiver_hotspot_signature_enrichment.csv", index=False)
    summary = summarize(result)
    summary.to_csv(output_dir / "receiver_hotspot_signature_enrichment_summary.csv", index=False)
    write_report(summary, output_dir / "receiver_hotspot_signature_enrichment_report.md")
    print(summary.to_string(index=False))
    print(f"Wrote receiver hotspot enrichment to {output_dir}")


def validate_replicate(
    replicate_dir: Path,
    prediction_path: Path,
    *,
    metadata: dict,
    stage: str,
    top_quantile: float,
    bottom_quantile: float,
    min_group_cells: int,
    min_signature_genes: int,
    n_permutations: int,
    seed: int,
) -> list[dict]:
    candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
    predictions = pd.read_csv(prediction_path)
    expression = pd.read_csv(replicate_dir / "expression.csv").set_index("cell_id")
    table = candidates.merge(predictions[["candidate_id", "score"]], on="candidate_id", how="inner")
    table["lr_pair"] = table["ligand"].astype(str) + "-" + table["receptor"].astype(str)
    available = set(expression.columns)
    rng = np.random.default_rng(seed)
    rows = []
    for lr_pair, signature_name in LR_SIGNATURE_MAP.items():
        genes = [gene for gene in CURATED_SIGNATURES[signature_name] if gene in available]
        base = {
            "replicate": replicate_dir.name,
            "stage": stage,
            "stage_day": STAGE_ORDER.get(stage, np.nan),
            "source_slice_file": metadata.get("source_slice_file", ""),
            "mouse_id": metadata.get("mouse_id", ""),
            "lr_pair": lr_pair,
            "signature_name": signature_name,
            "n_signature_genes_present": len(genes),
            "signature_genes_present": ";".join(genes),
        }
        group = table[table["lr_pair"].eq(lr_pair)].copy()
        if group.empty:
            rows.append({**base, "status": "lr_not_in_candidates"})
            continue
        if len(genes) < min_signature_genes:
            rows.append({**base, "status": "too_few_signature_genes"})
            continue
        receiver_score = group.groupby("receiver_id")["score"].mean()
        common = expression.index.intersection(receiver_score.index)
        if len(common) < min_group_cells * 2:
            rows.append({**base, "status": "too_few_receiver_cells", "n_receiver_cells": int(len(common))})
            continue
        comm = receiver_score.loc[common]
        sig = pd.Series(np.log1p(expression.loc[common, genes].to_numpy(float)).mean(axis=1), index=common)
        top_thr = comm.quantile(top_quantile)
        bot_thr = comm.quantile(bottom_quantile)
        top_cells = comm.index[comm >= top_thr]
        bot_cells = comm.index[comm <= bot_thr]
        if len(top_cells) < min_group_cells or len(bot_cells) < min_group_cells:
            rows.append(
                {
                    **base,
                    "status": "too_few_group_cells",
                    "n_receiver_cells": int(len(common)),
                    "n_top_cells": int(len(top_cells)),
                    "n_bottom_cells": int(len(bot_cells)),
                }
            )
            continue
        top_mean = float(sig.loc[top_cells].mean())
        bottom_mean = float(sig.loc[bot_cells].mean())
        effect = top_mean - bottom_mean
        pooled = float(np.sqrt(0.5 * (sig.loc[top_cells].var(ddof=1) + sig.loc[bot_cells].var(ddof=1))))
        cohen_d = float(effect / pooled) if pooled > 1e-12 else np.nan
        labels = np.array([1] * len(top_cells) + [0] * len(bot_cells))
        values = pd.concat([sig.loc[top_cells], sig.loc[bot_cells]]).to_numpy(float)
        exceed = 0
        for _ in range(n_permutations):
            perm = rng.permutation(values)
            null_effect = float(perm[labels == 1].mean() - perm[labels == 0].mean())
            if abs(null_effect) >= abs(effect):
                exceed += 1
        top_cell_pair = (
            group[group["receiver_id"].isin(top_cells)]
            .groupby(["sender_cell_type", "receiver_cell_type"])["score"]
            .sum()
            .reset_index()
            .sort_values("score", ascending=False)
            .head(1)
        )
        rows.append(
            {
                **base,
                "status": "tested",
                "n_receiver_cells": int(len(common)),
                "n_top_cells": int(len(top_cells)),
                "n_bottom_cells": int(len(bot_cells)),
                "top_score_mean": float(comm.loc[top_cells].mean()),
                "bottom_score_mean": float(comm.loc[bot_cells].mean()),
                "top_signature_mean": top_mean,
                "bottom_signature_mean": bottom_mean,
                "signature_effect": effect,
                "cohen_d": cohen_d,
                "empirical_p_value": float((exceed + 1) / (n_permutations + 1)),
                "top_hotspot_cell_type_pair": ""
                if top_cell_pair.empty
                else f"{top_cell_pair.iloc[0]['sender_cell_type']} -> {top_cell_pair.iloc[0]['receiver_cell_type']}",
            }
        )
    return rows


def summarize(result: pd.DataFrame) -> pd.DataFrame:
    if result.empty:
        return pd.DataFrame()
    tested = result[result["status"].eq("tested")].copy()
    if tested.empty:
        return pd.DataFrame()
    return (
        tested.groupby(["lr_pair", "signature_name"], as_index=False)
        .agg(
            n_replicates=("replicate", "nunique"),
            mean_effect=("signature_effect", "mean"),
            median_effect=("signature_effect", "median"),
            min_effect=("signature_effect", "min"),
            max_effect=("signature_effect", "max"),
            mean_cohen_d=("cohen_d", "mean"),
            n_positive_effect=("signature_effect", lambda x: int((x > 0).sum())),
            mean_empirical_p=("empirical_p_value", "mean"),
            min_bh_fdr=("bh_fdr", "min"),
            n_fdr_lt_0_1=("bh_fdr", lambda x: int((x < 0.10).sum())),
            n_fdr_lt_0_2=("bh_fdr", lambda x: int((x < 0.20).sum())),
            mean_signature_genes_present=("n_signature_genes_present", "mean"),
        )
        .sort_values(["n_fdr_lt_0_2", "mean_effect"], ascending=False)
    )


def write_report(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Mouse Colon MERFISH Receiver Hotspot Signature Enrichment",
        "",
        "This test compares receiver cells in high-SpatialSelf communication hotspots against low-communication receiver cells for each LR axis.",
        "",
        "## Summary",
        "",
    ]
    if summary.empty:
        lines.append("No hotspot enrichment tests were available.")
    else:
        for row in summary.head(12).itertuples(index=False):
            lines.append(
                f"- `{row.lr_pair}` / {row.signature_name}: mean effect={row.mean_effect:.3f}, "
                f"mean d={row.mean_cohen_d:.3f}, positive={row.n_positive_effect}/{row.n_replicates}, "
                f"FDR<0.2={row.n_fdr_lt_0_2}/{row.n_replicates}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
