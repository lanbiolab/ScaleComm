#!/usr/bin/env python
"""Validate mouse colon MERFISH CCC scores with receiver-side response signatures."""

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


STAGE_ORDER = {"Healthy_D0": 0, "DSS3": 3, "DSS9": 9, "DSS21": 21}

CURATED_SIGNATURES = {
    "Cxcr3_ifng_response": [
        "Stat1",
        "Stat2",
        "Irf1",
        "Irf7",
        "Irf9",
        "Isg15",
        "Ifit1",
        "Ifit2",
        "Ifit3",
        "Ifi27",
        "Ifi44",
        "Ifi44l",
        "Mx1",
        "Mx2",
        "Oas1a",
        "Oas1g",
        "Oasl1",
        "Oasl2",
        "Cxcl9",
        "Cxcl10",
        "Cxcl11",
        "Tap1",
        "B2m",
        "H2-Ab1",
        "H2-Eb1",
        "Cd74",
    ],
    "Ccr2_monocyte_inflammation": [
        "Ccr2",
        "Lyz2",
        "S100a8",
        "S100a9",
        "Il1b",
        "Tnf",
        "Nfkb1",
        "Nfkbia",
        "Ccl2",
        "Ccl3",
        "Ccl4",
        "Ccl7",
        "Ccl8",
        "Cxcl2",
        "Cxcl3",
        "Mmp9",
        "Itgam",
        "Fcgr3",
    ],
    "Ccr5_tcell_myeloid_inflammation": [
        "Ccr5",
        "Ccl3",
        "Ccl4",
        "Ccl5",
        "Cxcl9",
        "Cxcl10",
        "Ifng",
        "Gzmb",
        "Prf1",
        "Nkg7",
        "Cd3d",
        "Cd3e",
        "Cd8a",
        "Cd4",
        "Tnf",
        "Il1b",
    ],
    "Cxcr4_repair_remodeling": [
        "Cxcr4",
        "Cxcl12",
        "Col1a1",
        "Col1a2",
        "Col3a1",
        "Fn1",
        "Acta2",
        "Tagln",
        "Pdgfra",
        "Pdgfrb",
        "Vegfa",
        "Kdr",
        "Mmp2",
        "Timp1",
        "Tgfb1",
        "Tgfbr2",
    ],
    "Tnf_nfkb_response": [
        "Tnf",
        "Tnfrsf1a",
        "Tnfrsf1b",
        "Nfkb1",
        "Nfkbia",
        "Rel",
        "Rela",
        "Il1b",
        "Il6",
        "Cxcl1",
        "Cxcl2",
        "Cxcl3",
        "Ccl2",
        "Ccl5",
        "Icam1",
        "Vcam1",
    ],
}

LR_SIGNATURE_MAP = {
    "Cxcl9-Cxcr3": "Cxcr3_ifng_response",
    "Cxcl10-Cxcr3": "Cxcr3_ifng_response",
    "Ccl2-Ccr2": "Ccr2_monocyte_inflammation",
    "Ccl5-Ccr5": "Ccr5_tcell_myeloid_inflammation",
    "Cxcl12-Cxcr4": "Cxcr4_repair_remodeling",
    "Tnf-Tnfrsf1a": "Tnf_nfkb_response",
    "Tnf-Tnfrsf1b": "Tnf_nfkb_response",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replicate-root",
        default="outputs/current_mouse_colon_ibd_merfish/multislice_replicates",
    )
    parser.add_argument(
        "--prediction-root",
        default="outputs/current_mouse_colon_ibd_merfish/multislice_spatialself/predictions",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/current_mouse_colon_ibd_merfish/downstream_signature_validation",
    )
    parser.add_argument("--model-name", default="spatialself_wavelet_scale")
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--min-signature-genes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260620)
    args = parser.parse_args()

    replicate_root = Path(args.replicate_root)
    prediction_root = Path(args.prediction_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    status_rows = []
    for replicate_dir in find_replicates(replicate_root):
        metadata = read_metadata(replicate_dir)
        stage = metadata.get("stage", infer_stage(replicate_dir.name))
        pred_path = prediction_root / stage / args.model_name / f"{replicate_dir.name}__{args.model_name}_predictions.csv"
        if not pred_path.exists():
            status_rows.append({"replicate": replicate_dir.name, "stage": stage, "status": "missing_prediction", "prediction_path": str(pred_path)})
            continue
        replicate_rows, replicate_status = validate_replicate(
            replicate_dir,
            pred_path,
            stage=stage,
            metadata=metadata,
            min_signature_genes=int(args.min_signature_genes),
            n_permutations=int(args.n_permutations),
            seed=int(args.seed) + len(rows),
        )
        rows.extend(replicate_rows)
        status_rows.extend(replicate_status)

    result = pd.DataFrame(rows)
    status = pd.DataFrame(status_rows)
    if not result.empty:
        tested = result["status"].eq("tested")
        result["bh_fdr"] = np.nan
        if tested.any():
            result.loc[tested, "bh_fdr"] = bh_adjust(result.loc[tested, "empirical_p_value"].to_numpy(float))
        result = result.sort_values(["lr_pair", "stage_day", "replicate"])
    result.to_csv(output_dir / "per_replicate_downstream_signature_validation.csv", index=False)
    status.to_csv(output_dir / "downstream_signature_status.csv", index=False)
    summary = summarize(result)
    summary.to_csv(output_dir / "downstream_signature_summary.csv", index=False)
    write_report(summary, status, output_dir / "downstream_signature_report.md")
    print(summary.to_string(index=False))
    print(f"Wrote downstream signature validation to {output_dir}")


def find_replicates(root: Path) -> list[Path]:
    required = {"cells.csv", "expression.csv", "lr_pairs.csv", "candidate_interactions.csv"}
    reps = []
    for path in sorted(root.iterdir()) if root.exists() else []:
        if path.is_dir() and required.issubset({file.name for file in path.iterdir() if file.is_file()}):
            reps.append(path)
    return sorted(reps, key=lambda p: (STAGE_ORDER.get(infer_stage(p.name), 999), p.name))


def read_metadata(replicate_dir: Path) -> dict:
    path = replicate_dir / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def infer_stage(name: str) -> str:
    for stage in STAGE_ORDER:
        if stage in name:
            return stage
    return "unknown"


def validate_replicate(
    replicate_dir: Path,
    prediction_path: Path,
    *,
    stage: str,
    metadata: dict,
    min_signature_genes: int,
    n_permutations: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
    predictions = pd.read_csv(prediction_path)
    expression = pd.read_csv(replicate_dir / "expression.csv")
    cells = pd.read_csv(replicate_dir / "cells.csv")
    table = candidates.merge(predictions[["candidate_id", "score"]], on="candidate_id", how="inner")
    table["lr_pair"] = table["ligand"].astype(str) + "-" + table["receptor"].astype(str)
    expression = expression.set_index("cell_id")
    available_genes = set(expression.columns)
    rng = np.random.default_rng(seed)
    rows = []
    status_rows = []
    for lr_pair, signature_name in LR_SIGNATURE_MAP.items():
        signature_genes = [gene for gene in CURATED_SIGNATURES[signature_name] if gene in available_genes]
        status_base = {
            "replicate": replicate_dir.name,
            "stage": stage,
            "stage_day": STAGE_ORDER.get(stage, np.nan),
            "source_slice_file": metadata.get("source_slice_file", ""),
            "mouse_id": metadata.get("mouse_id", ""),
            "lr_pair": lr_pair,
            "signature_name": signature_name,
            "n_signature_genes_present": len(signature_genes),
            "signature_genes_present": ";".join(signature_genes),
        }
        group = table[table["lr_pair"].eq(lr_pair)].copy()
        if group.empty:
            status_rows.append({**status_base, "status": "lr_not_in_candidates"})
            continue
        if len(signature_genes) < min_signature_genes:
            status_rows.append({**status_base, "status": "too_few_signature_genes"})
            rows.append(make_empty_result(status_base, group, "too_few_signature_genes"))
            continue
        receiver_score = group.groupby("receiver_id")["score"].mean()
        common = expression.index.intersection(receiver_score.index)
        if len(common) < 5:
            status_rows.append({**status_base, "status": "too_few_receiver_cells"})
            rows.append(make_empty_result(status_base, group, "too_few_receiver_cells"))
            continue
        comm = receiver_score.loc[common].to_numpy(float)
        sig = np.log1p(expression.loc[common, signature_genes].to_numpy(float)).mean(axis=1)
        if np.std(comm) <= 1e-12 or np.std(sig) <= 1e-12:
            status_rows.append({**status_base, "status": "constant_vector"})
            rows.append(make_empty_result(status_base, group, "constant_vector"))
            continue
        corr, p_value = spearmanr(comm, sig)
        exceed = 0
        for _ in range(n_permutations):
            null_corr, _ = spearmanr(comm, rng.permutation(sig))
            if abs(float(null_corr)) >= abs(float(corr)):
                exceed += 1
        status_rows.append({**status_base, "status": "tested"})
        top_cell_pair = (
            group.groupby(["sender_cell_type", "receiver_cell_type"])["score"]
            .sum()
            .reset_index()
            .sort_values("score", ascending=False)
            .head(1)
        )
        rows.append(
            {
                **status_base,
                "status": "tested",
                "n_edges": int(len(group)),
                "n_receiver_cells": int(group["receiver_id"].nunique()),
                "mean_score": float(group["score"].mean()),
                "score_q95": float(group["score"].quantile(0.95)),
                "spearman_r": float(corr),
                "spearman_p_value": float(p_value),
                "empirical_p_value": float((exceed + 1) / (n_permutations + 1)),
                "top_cell_type_pair": ""
                if top_cell_pair.empty
                else f"{top_cell_pair.iloc[0]['sender_cell_type']} -> {top_cell_pair.iloc[0]['receiver_cell_type']}",
            }
        )
    return rows, status_rows


def make_empty_result(status_base: dict, group: pd.DataFrame, status: str) -> dict:
    return {
        **status_base,
        "status": status,
        "n_edges": int(len(group)),
        "n_receiver_cells": int(group["receiver_id"].nunique()) if "receiver_id" in group else 0,
        "mean_score": np.nan,
        "score_q95": np.nan,
        "spearman_r": np.nan,
        "spearman_p_value": np.nan,
        "empirical_p_value": np.nan,
        "top_cell_type_pair": "",
    }


def summarize(result: pd.DataFrame) -> pd.DataFrame:
    if result.empty:
        return pd.DataFrame()
    tested = result[result["status"].eq("tested")].copy()
    if tested.empty:
        return pd.DataFrame()
    summary = (
        tested.groupby(["lr_pair", "signature_name"], as_index=False)
        .agg(
            n_replicates=("replicate", "nunique"),
            mean_spearman_r=("spearman_r", "mean"),
            median_spearman_r=("spearman_r", "median"),
            min_spearman_r=("spearman_r", "min"),
            max_spearman_r=("spearman_r", "max"),
            mean_empirical_p=("empirical_p_value", "mean"),
            min_bh_fdr=("bh_fdr", "min"),
            n_nominal_positive=("spearman_r", lambda x: int((x > 0).sum())),
            n_fdr_lt_0_1=("bh_fdr", lambda x: int((x < 0.10).sum())),
            n_fdr_lt_0_2=("bh_fdr", lambda x: int((x < 0.20).sum())),
            mean_signature_genes_present=("n_signature_genes_present", "mean"),
        )
        .sort_values(["n_fdr_lt_0_2", "mean_spearman_r"], ascending=False)
    )
    return summary


def write_report(summary: pd.DataFrame, status: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Mouse Colon MERFISH Downstream Signature Validation",
        "",
        "This panel-aware validation correlates SpatialSelf receiver-side communication scores with observed MERFISH response signatures.",
        "",
        "Because the dataset is a targeted MERFISH panel, only genes present in the panel are used.",
        "",
        "## Tested Axes",
        "",
    ]
    if status.empty:
        lines.append("No status rows were generated.")
    else:
        tested = status[status["status"].eq("tested")]
        lines.append(f"- Tested replicate-axis pairs: {len(tested)}")
        lines.append(f"- Total status rows: {len(status)}")
    lines += ["", "## Summary", ""]
    if summary.empty:
        lines.append("No downstream signature correlations were testable.")
    else:
        for row in summary.head(12).itertuples(index=False):
            lines.append(
                f"- `{row.lr_pair}` / {row.signature_name}: mean rho={row.mean_spearman_r:.3f}, "
                f"positive replicates={row.n_nominal_positive}/{row.n_replicates}, "
                f"FDR<0.2={row.n_fdr_lt_0_2}/{row.n_replicates}, "
                f"observed signature genes~{row.mean_signature_genes_present:.1f}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
