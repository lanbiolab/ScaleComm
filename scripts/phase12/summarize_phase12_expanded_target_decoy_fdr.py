#!/usr/bin/env python
"""Target-decoy/control-FDR audit for Phase 12 expanded atlas rankings.

The permutation audit asks whether LR scores exceed randomized null scores.
This complementary audit asks a different whole-atlas question: as we move down
the full LR ranking, how many pre-defined hard controls/decoys are admitted?

It is a global ranking audit over all covered LR pairs, but it is not an
unbiased discovery FDR for real tissue because most background LR pairs have no
native ground truth. Keep this distinction explicit in manuscript claims.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
EXPANDED_DIR = PHASE12_DIR / "atlas_model_expanded"
OUT_DIR = EXPANDED_DIR / "target_decoy_fdr"
TOP_K_VALUES = (5, 10, 20, 50, 100, 200)
FDR_THRESHOLDS = (0.05, 0.10, 0.20)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ranking = pd.read_csv(EXPANDED_DIR / "phase12_atlas_model_expanded_calibrated_lr_ranking.csv")
    annotated = annotate_decoys(ranking)
    lr_table = add_control_fdr(annotated)
    topk = summarize_topk(lr_table)
    dataset = summarize_dataset(lr_table, topk)

    lr_table.to_csv(OUT_DIR / "phase12_expanded_target_decoy_lr_fdr.csv", index=False)
    topk.to_csv(OUT_DIR / "phase12_expanded_target_decoy_topk_summary.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_expanded_target_decoy_dataset_summary.csv", index=False)
    write_status(dataset, topk)
    write_manifest(lr_table, topk, dataset)
    print(dataset.to_string(index=False))
    print()
    print(topk.to_string(index=False))
    print()
    print(lr_table.groupby("dataset_key").head(20).to_string(index=False))


def annotate_decoys(ranking: pd.DataFrame) -> pd.DataFrame:
    out = ranking.copy()
    out["true_lr"] = out["true_lr"].fillna(False).astype(bool)
    out["false_positive_high_expression"] = out["false_positive_high_expression"].fillna(False).astype(bool)
    out["rank"] = pd.to_numeric(out["rank"], errors="coerce")
    out["calibrated_score"] = pd.to_numeric(out["calibrated_score"], errors="coerce")
    out["n_nonzero_expr"] = pd.to_numeric(out["n_nonzero_expr"], errors="coerce").fillna(0)
    out["paired_expr_fraction"] = pd.to_numeric(out.get("paired_expr_fraction", 0), errors="coerce").fillna(0)
    out["receptor_shared_decoy"] = False
    out["ligand_shared_decoy"] = False
    for dataset_key, sub in out.groupby("dataset_key", sort=False):
        true_receptors = set(sub[sub["true_lr"]]["receptor"].astype(str))
        true_ligands = set(sub[sub["true_lr"]]["ligand"].astype(str))
        mask = out["dataset_key"].eq(dataset_key)
        out.loc[mask, "receptor_shared_decoy"] = (
            out.loc[mask, "receptor"].astype(str).isin(true_receptors)
            & ~out.loc[mask, "true_lr"].astype(bool)
        )
        out.loc[mask, "ligand_shared_decoy"] = (
            out.loc[mask, "ligand"].astype(str).isin(true_ligands)
            & ~out.loc[mask, "true_lr"].astype(bool)
        )
    out["zero_expression_decoy"] = out["n_nonzero_expr"].le(0)
    out["primary_control_decoy"] = (
        out["false_positive_high_expression"]
        | out["receptor_shared_decoy"]
        | out["ligand_shared_decoy"]
    )
    out["strict_control_decoy"] = out["primary_control_decoy"] | out["zero_expression_decoy"]
    out["decoy_class"] = out.apply(decoy_class, axis=1)
    return out


def add_control_fdr(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in table.groupby("dataset_key", sort=False):
        sub = sub.sort_values(["rank", "lr_pair"]).reset_index(drop=True).copy()
        sub["rank_position"] = np.arange(1, len(sub) + 1)
        for mode, column in [
            ("primary_control", "primary_control_decoy"),
            ("strict_control", "strict_control_decoy"),
        ]:
            decoy = sub[column].fillna(False).astype(bool).to_numpy()
            target = ~decoy
            cum_decoy = np.cumsum(decoy).astype(float)
            cum_target = np.cumsum(target).astype(float)
            # Prefix FDR is the directly interpretable top-k contamination
            # estimate. The +1 pseudo-count avoids reporting zero FDR before
            # any hard controls are observed.
            prefix_fdr = (cum_decoy + 1.0) / np.maximum(cum_target, 1.0)
            # Monotone target-decoy q is included for completeness, but should
            # be interpreted together with prefix FDR because decoys may be
            # enriched near the bottom after calibration.
            q_value = np.minimum.accumulate(prefix_fdr[::-1])[::-1]
            sub[f"{mode}_cum_decoys"] = cum_decoy.astype(int)
            sub[f"{mode}_cum_targets"] = cum_target.astype(int)
            sub[f"{mode}_prefix_fdr"] = prefix_fdr
            sub[f"{mode}_target_decoy_q"] = q_value
        rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_topk(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in table.groupby("dataset_key", sort=False):
        sub = sub.sort_values("rank_position")
        for top_k in TOP_K_VALUES:
            top = sub.head(top_k)
            if top.empty:
                continue
            last = top.iloc[-1]
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "top_k": top_k,
                    "n_lr": int(len(top)),
                    "n_true_lr": int(top["true_lr"].sum()),
                    "n_primary_control_decoy": int(top["primary_control_decoy"].sum()),
                    "n_strict_control_decoy": int(top["strict_control_decoy"].sum()),
                    "n_high_expression_decoy": int(top["false_positive_high_expression"].sum()),
                    "n_receptor_shared_decoy": int(top["receptor_shared_decoy"].sum()),
                    "n_ligand_shared_decoy": int(top["ligand_shared_decoy"].sum()),
                    "n_zero_expression_decoy": int(top["zero_expression_decoy"].sum()),
                    "primary_control_prefix_fdr": float(last["primary_control_prefix_fdr"]),
                    "strict_control_prefix_fdr": float(last["strict_control_prefix_fdr"]),
                    "primary_control_target_decoy_q": float(last["primary_control_target_decoy_q"]),
                    "strict_control_target_decoy_q": float(last["strict_control_target_decoy_q"]),
                    "top_lr_pairs": ";".join(top["lr_pair"].astype(str).head(12)),
                }
            )
    return pd.DataFrame(rows)


def summarize_dataset(table: pd.DataFrame, topk: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in table.groupby("dataset_key", sort=False):
        row = {
            "dataset_key": dataset_key,
            "n_lr": int(len(sub)),
            "n_true_lr": int(sub["true_lr"].sum()),
            "n_primary_control_decoy": int(sub["primary_control_decoy"].sum()),
            "n_strict_control_decoy": int(sub["strict_control_decoy"].sum()),
            "n_high_expression_decoy": int(sub["false_positive_high_expression"].sum()),
            "n_receptor_shared_decoy": int(sub["receptor_shared_decoy"].sum()),
            "n_ligand_shared_decoy": int(sub["ligand_shared_decoy"].sum()),
            "n_zero_expression_decoy": int(sub["zero_expression_decoy"].sum()),
        }
        dataset_topk = topk[topk["dataset_key"].eq(dataset_key)].copy()
        for threshold in FDR_THRESHOLDS:
            key = str(threshold).replace(".", "_")
            row[f"reported_topk_primary_fdr_le_{key}"] = topk_pass_list(
                dataset_topk, "primary_control_prefix_fdr", threshold
            )
            row[f"reported_topk_strict_fdr_le_{key}"] = topk_pass_list(
                dataset_topk, "strict_control_prefix_fdr", threshold
            )
        top10 = topk[(topk["dataset_key"].eq(dataset_key)) & (topk["top_k"].eq(10))].head(1)
        top100 = topk[(topk["dataset_key"].eq(dataset_key)) & (topk["top_k"].eq(100))].head(1)
        row["top10_primary_control_prefix_fdr"] = value(top10, "primary_control_prefix_fdr")
        row["top10_strict_control_prefix_fdr"] = value(top10, "strict_control_prefix_fdr")
        row["top100_primary_control_prefix_fdr"] = value(top100, "primary_control_prefix_fdr")
        row["top100_strict_control_prefix_fdr"] = value(top100, "strict_control_prefix_fdr")
        row["control_fdr_call"] = classify_dataset(row)
        rows.append(row)
    return pd.DataFrame(rows)


def topk_pass_list(table: pd.DataFrame, column: str, threshold: float) -> str:
    if table.empty:
        return ""
    passing = table[pd.to_numeric(table[column], errors="coerce").le(threshold)]
    if passing.empty:
        return ""
    return ";".join(str(int(x)) for x in passing["top_k"].tolist())


def classify_dataset(row: dict[str, object]) -> str:
    if float(row["top10_primary_control_prefix_fdr"]) <= 0.1 and float(row["top10_strict_control_prefix_fdr"]) <= 0.1:
        return "top10_control_fdr_clean"
    if float(row["top10_primary_control_prefix_fdr"]) <= 0.1:
        return "top10_primary_control_clean_strict_caveat"
    if float(row["top100_primary_control_prefix_fdr"]) <= 0.1:
        return "control_fdr_improves_by_top100_top10_caveat"
    return "control_fdr_not_clean"


def decoy_class(row: pd.Series) -> str:
    if bool(row.get("false_positive_high_expression", False)):
        return "high_expression_control"
    if bool(row.get("receptor_shared_decoy", False)):
        return "receptor_shared_control"
    if bool(row.get("ligand_shared_decoy", False)):
        return "ligand_shared_control"
    if bool(row.get("zero_expression_decoy", False)):
        return "zero_expression_control"
    if bool(row.get("true_lr", False)):
        return "prespecified_positive_anchor"
    return "unlabeled_background_lr"


def value(table: pd.DataFrame, column: str) -> float:
    if table.empty or column not in table:
        return np.nan
    return float(table[column].iloc[0])


def write_status(dataset: pd.DataFrame, topk: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Expanded Atlas Target-Decoy / Control-FDR Audit",
        "",
        "## Scope",
        "",
        "This audit scans the full calibrated LR ranking and estimates cumulative hard-control contamination using pre-defined decoys: high-expression false LR, receptor-shared controls and ligand-shared controls. A stricter diagnostic mode additionally treats zero-expression LR as controls.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.control_fdr_call}`: LR={int(row.n_lr)}, "
            f"primary decoys={int(row.n_primary_control_decoy)}, strict decoys={int(row.n_strict_control_decoy)}, "
            f"top10 primary FDR={fmt(row.top10_primary_control_prefix_fdr)}, top10 strict FDR={fmt(row.top10_strict_control_prefix_fdr)}, "
            f"top100 primary FDR={fmt(row.top100_primary_control_prefix_fdr)}, top100 strict FDR={fmt(row.top100_strict_control_prefix_fdr)}"
        )
    lines += ["", "## Top-K Control FDR", ""]
    for row in topk[topk["top_k"].isin([10, 20, 50, 100])].itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` top{int(row.top_k)}: true={int(row.n_true_lr)}, "
            f"primary decoy={int(row.n_primary_control_decoy)}, strict decoy={int(row.n_strict_control_decoy)}, "
            f"primary prefix FDR={fmt(row.primary_control_prefix_fdr)}, strict prefix FDR={fmt(row.strict_control_prefix_fdr)}"
        )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "This is a whole-ranking control-FDR audit, not native ground-truth FDR for all real LR pairs. It should be reported alongside the stricter permutation audit, which currently has no all-LR BH q<=0.1 discoveries.",
    ]
    (OUT_DIR / "phase12_expanded_target_decoy_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(lr_table: pd.DataFrame, topk: pd.DataFrame, dataset: pd.DataFrame) -> None:
    manifest = {
        "task": "phase12_expanded_target_decoy_fdr",
        "n_lr_rows": int(len(lr_table)),
        "n_topk_rows": int(len(topk)),
        "n_dataset_rows": int(len(dataset)),
        "decoy_definitions": {
            "primary_control_decoy": "high-expression false LR OR receptor-shared control OR ligand-shared control",
            "strict_control_decoy": "primary_control_decoy OR zero-expression LR",
        },
        "outputs": {
            "lr_fdr": rel(OUT_DIR / "phase12_expanded_target_decoy_lr_fdr.csv"),
            "topk_summary": rel(OUT_DIR / "phase12_expanded_target_decoy_topk_summary.csv"),
            "dataset_summary": rel(OUT_DIR / "phase12_expanded_target_decoy_dataset_summary.csv"),
            "status": rel(OUT_DIR / "phase12_expanded_target_decoy_status.md"),
        },
        "claim_boundary": "Whole-ranking hard-control contamination audit; not unbiased real-tissue discovery FDR.",
    }
    (OUT_DIR / "phase12_expanded_target_decoy_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def fmt(value_: object) -> str:
    try:
        if pd.isna(value_):
            return "NA"
        val = float(value_)
    except (TypeError, ValueError):
        return str(value_)
    if abs(val) >= 10:
        return f"{val:.2f}"
    return f"{val:.3g}"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
