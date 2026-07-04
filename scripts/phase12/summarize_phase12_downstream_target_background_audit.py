#!/usr/bin/env python
"""Target-background audit for Phase 12 whole-transcriptome downstream scans.

The expanded downstream scan already computes receiver-side whole-transcriptome
gene ranks for selected top LR, priority anchors, and module representatives.
This summary makes the claim boundary explicit: it asks whether curated target
genes rank above all expressed genes for the scanned LR/module universe, and
separates true/priority anchors from high-expression controls and background
LR. It is not perturbational validation and not pan-atlas functional proof for
every LR in the expanded ranking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
DOWNSTREAM_DIR = PHASE12_DIR / "expanded_atlas_downstream_scan"
OUT_DIR = PHASE12_DIR / "downstream_target_background_audit"
ALPHA = 0.10


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lr = read_csv(DOWNSTREAM_DIR / "phase12_expanded_atlas_downstream_lr_summary.csv")
    gene = read_csv(DOWNSTREAM_DIR / "phase12_expanded_atlas_downstream_gene_table.csv")
    module = read_csv(DOWNSTREAM_DIR / "phase12_expanded_atlas_downstream_module_summary.csv")
    dataset = read_csv(DOWNSTREAM_DIR / "phase12_expanded_atlas_downstream_dataset_summary.csv")
    if lr.empty:
        raise FileNotFoundError("Missing expanded downstream LR summary")

    lr_audit = build_lr_audit(lr, gene)
    dataset_audit = build_dataset_audit(lr_audit, dataset)
    module_audit = build_module_audit(module, lr_audit)
    gene_program_audit = build_gene_program_audit(gene, lr_audit)

    lr_audit.to_csv(OUT_DIR / "phase12_downstream_target_background_lr_audit.csv", index=False)
    dataset_audit.to_csv(OUT_DIR / "phase12_downstream_target_background_dataset_audit.csv", index=False)
    module_audit.to_csv(OUT_DIR / "phase12_downstream_target_background_module_audit.csv", index=False)
    gene_program_audit.to_csv(OUT_DIR / "phase12_downstream_target_background_gene_program_audit.csv", index=False)
    write_status(dataset_audit, lr_audit, module_audit, gene_program_audit)
    write_manifest(lr_audit, dataset_audit, module_audit, gene_program_audit)
    print(dataset_audit.to_string(index=False))
    print()
    print(lr_audit.sort_values(["dataset_key", "target_background_call", "rank"], ascending=[True, True, True]).head(40).to_string(index=False))


def build_lr_audit(lr: pd.DataFrame, gene: pd.DataFrame) -> pd.DataFrame:
    out = lr.copy()
    out["lr_pair"] = out["lr_pair"].astype(str)
    out["selection_sources"] = out.get("selection_sources", pd.Series("", index=out.index)).astype(str)
    out["has_curated_target_set"] = pd.to_numeric(out.get("n_targets_present", pd.Series(dtype=float)), errors="coerce").fillna(0).gt(0)
    out["target_enriched_q_0_1"] = pd.to_numeric(
        out.get("target_top300_enrichment_bh_fdr", pd.Series(dtype=float)),
        errors="coerce",
    ).le(ALPHA)
    out["target_enriched_q_0_2"] = pd.to_numeric(
        out.get("target_top300_enrichment_bh_fdr", pd.Series(dtype=float)),
        errors="coerce",
    ).le(0.20)
    out["target_above_random_mean"] = pd.to_numeric(
        out.get("target_mean_percentile", pd.Series(dtype=float)),
        errors="coerce",
    ).gt(0.50)
    out["target_enrichment_gt_1"] = pd.to_numeric(
        out.get("target_top300_enrichment", pd.Series(dtype=float)),
        errors="coerce",
    ).gt(1.0)
    out["top_rank_selected"] = out["selection_sources"].str.contains("top50", regex=False)
    out["module_representative"] = out["selection_sources"].str.contains("module_", regex=False)
    out["priority_anchor"] = out["selection_sources"].str.contains("priority_rank_le_", regex=False)
    out["true_lr"] = out.get("true_lr", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    out["false_positive_high_expression"] = out.get(
        "false_positive_high_expression",
        pd.Series(False, index=out.index),
    ).fillna(False).astype(bool)
    out["background_or_nonanchor"] = ~out["true_lr"] & ~out["false_positive_high_expression"]
    out["target_support_score"] = target_support_score(out)
    out["target_background_call"] = [
        target_call(row)
        for row in out.itertuples(index=False)
    ]

    if not gene.empty:
        gene = gene.copy()
        gene["lr_pair"] = gene["lr_pair"].astype(str)
        gene["is_target"] = gene.get("is_target", pd.Series(False, index=gene.index)).fillna(False).astype(bool)
        gene["whole_transcriptome_rank"] = pd.to_numeric(gene["whole_transcriptome_rank"], errors="coerce")
        top_rows = gene[gene["whole_transcriptome_rank"].le(300)].copy()
        target_rows = gene[gene["is_target"]].copy()
        top_gene_summary = (
            top_rows.groupby(["dataset_key", "lr_pair"], as_index=False)
            .agg(
                n_top300_gene_rows=("gene", "size"),
                top300_mean_spearman=("spearman_r", "mean"),
                strongest_top300_gene=("gene", first_joined),
            )
        )
        target_gene_summary = (
            target_rows.sort_values(["dataset_key", "lr_pair", "whole_transcriptome_rank"])
            .groupby(["dataset_key", "lr_pair"], as_index=False)
            .agg(
                n_target_gene_rows=("gene", "size"),
                best_target_gene=("gene", first_joined),
                best_target_gene_rank=("whole_transcriptome_rank", "min"),
            )
        )
        out = out.merge(top_gene_summary, on=["dataset_key", "lr_pair"], how="left")
        out = out.merge(target_gene_summary, on=["dataset_key", "lr_pair"], how="left")
    return out.sort_values(["dataset_key", "rank", "lr_pair"])


def target_support_score(out: pd.DataFrame) -> pd.Series:
    q = pd.to_numeric(out.get("target_top300_enrichment_bh_fdr", pd.Series(dtype=float)), errors="coerce")
    enrich = pd.to_numeric(out.get("target_top300_enrichment", pd.Series(dtype=float)), errors="coerce")
    pct = pd.to_numeric(out.get("target_mean_percentile", pd.Series(dtype=float)), errors="coerce")
    score = pd.Series(0.0, index=out.index)
    score += q.le(ALPHA).astype(float) * 3.0
    score += q.le(0.20).astype(float) * 1.0
    score += enrich.gt(1.0).astype(float)
    score += pct.gt(0.50).astype(float)
    score += np.clip(enrich.fillna(0.0), 0.0, 5.0) / 5.0
    return score


def target_call(row: Any) -> str:
    if not bool(row.has_curated_target_set):
        return "no_curated_target_set"
    if bool(row.false_positive_high_expression) and bool(row.target_enriched_q_0_1):
        return "high_expression_negative_target_leakage"
    if bool(row.target_enriched_q_0_1) and bool(row.target_enrichment_gt_1):
        if bool(row.true_lr):
            return "true_lr_target_rank_supported"
        return "nonanchor_target_rank_supported"
    if bool(row.target_enriched_q_0_2) and bool(row.target_enrichment_gt_1):
        return "suggestive_target_rank_support"
    return "target_set_not_above_background"


def build_dataset_audit(lr: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_key, sub in lr.groupby("dataset_key", sort=False):
        has_targets = sub["has_curated_target_set"].astype(bool)
        supported = sub["target_background_call"].astype(str).isin(
            ["true_lr_target_rank_supported", "nonanchor_target_rank_supported"]
        )
        true_lr = sub["true_lr"].astype(bool)
        high_neg = sub["false_positive_high_expression"].astype(bool)
        priority = sub["priority_anchor"].astype(bool)
        module = sub["module_representative"].astype(bool)
        top = sub["top_rank_selected"].astype(bool)
        leakage = sub["target_background_call"].astype(str).eq("high_expression_negative_target_leakage")
        row0 = pick(dataset, dataset_key=dataset_key)
        rows.append(
            {
                "dataset_key": dataset_key,
                "n_cells": int(get(row0, "n_cells", np.nan)) if finite(get(row0, "n_cells", np.nan)) else np.nan,
                "n_genes_tested": int(get(row0, "n_genes_tested", np.nan)) if finite(get(row0, "n_genes_tested", np.nan)) else np.nan,
                "n_lr_scanned": int(len(sub)),
                "n_lr_with_curated_targets": int(has_targets.sum()),
                "n_top_rank_lr_scanned": int(top.sum()),
                "n_module_representative_lr_scanned": int(module.sum()),
                "n_priority_anchor_lr_scanned": int(priority.sum()),
                "n_target_supported_q_0_1": int(supported.sum()),
                "n_true_lr_target_supported_q_0_1": int((supported & true_lr).sum()),
                "n_priority_anchor_target_supported_q_0_1": int((supported & priority).sum()),
                "n_module_rep_target_supported_q_0_1": int((supported & module).sum()),
                "n_high_expression_negative_target_leakage": int(leakage.sum()),
                "n_high_expression_negative_with_targets": int((high_neg & has_targets).sum()),
                "target_support_fraction_among_targeted_lr": float(supported.sum() / has_targets.sum()) if has_targets.sum() else np.nan,
                "true_lr_support_fraction_among_targeted_true_lr": fraction(supported & true_lr, has_targets & true_lr),
                "median_target_mean_percentile": median(sub.loc[has_targets, "target_mean_percentile"]),
                "median_target_top300_enrichment": median(sub.loc[has_targets, "target_top300_enrichment"]),
                "best_supported_lr": best_supported_lr(sub),
                "dataset_target_background_call": dataset_call(sub),
            }
        )
    return pd.DataFrame(rows)


def build_module_audit(module: pd.DataFrame, lr: pd.DataFrame) -> pd.DataFrame:
    if module.empty:
        return pd.DataFrame()
    rows = []
    by_dataset_pair = {
        dataset_key: sub.set_index("lr_pair")
        for dataset_key, sub in lr.groupby("dataset_key", sort=False)
    }
    for row in module.itertuples(index=False):
        dataset_key = str(row.dataset_key)
        pairs = [p for p in str(row.top_pairs_tested).split(";") if p]
        lr_sub = by_dataset_pair.get(dataset_key, pd.DataFrame()).reindex(pairs) if pairs else pd.DataFrame()
        if lr_sub.empty:
            supported = pd.Series(dtype=bool)
            has_targets = pd.Series(dtype=bool)
        else:
            supported = lr_sub["target_background_call"].astype(str).isin(
                ["true_lr_target_rank_supported", "nonanchor_target_rank_supported"]
            )
            has_targets = lr_sub["has_curated_target_set"].astype(bool)
        rows.append(
            {
                "dataset_key": dataset_key,
                "module": row.module,
                "n_lr_in_module": int(row.n_lr_in_module),
                "n_top_pairs_tested": int(row.n_top_pairs_tested),
                "n_top_pairs_with_targets": int(has_targets.sum()) if len(has_targets) else 0,
                "n_target_supported_pairs": int(supported.sum()) if len(supported) else 0,
                "best_target_enrichment_fdr": get(row, "best_target_enrichment_fdr", np.nan),
                "mean_target_top300_enrichment": get(row, "mean_target_top300_enrichment", np.nan),
                "module_target_background_call": module_call(has_targets, supported),
                "top_pairs_tested": row.top_pairs_tested,
            }
        )
    return pd.DataFrame(rows)


def build_gene_program_audit(gene: pd.DataFrame, lr: pd.DataFrame) -> pd.DataFrame:
    if gene.empty:
        return pd.DataFrame()
    supported_pairs = set(
        zip(
            lr.loc[
                lr["target_background_call"].astype(str).isin(
                    ["true_lr_target_rank_supported", "nonanchor_target_rank_supported"]
                ),
                "dataset_key",
            ].astype(str),
            lr.loc[
                lr["target_background_call"].astype(str).isin(
                    ["true_lr_target_rank_supported", "nonanchor_target_rank_supported"]
                ),
                "lr_pair",
            ].astype(str),
        )
    )
    gene = gene.copy()
    gene["key"] = list(zip(gene["dataset_key"].astype(str), gene["lr_pair"].astype(str)))
    supported_gene = gene[gene["key"].isin(supported_pairs)].copy()
    if supported_gene.empty:
        return pd.DataFrame()
    supported_gene["whole_transcriptome_rank"] = pd.to_numeric(
        supported_gene["whole_transcriptome_rank"],
        errors="coerce",
    )
    rows = []
    for (dataset_key, lr_pair), sub in supported_gene.groupby(["dataset_key", "lr_pair"], sort=False):
        top = sub.sort_values("whole_transcriptome_rank").head(20)
        target = sub[sub.get("is_target", pd.Series(False, index=sub.index)).fillna(False).astype(bool)]
        rows.append(
            {
                "dataset_key": dataset_key,
                "lr_pair": lr_pair,
                "n_gene_rows": int(len(sub)),
                "n_target_gene_rows": int(len(target)),
                "top20_genes": ";".join(top["gene"].astype(str).head(20).tolist()),
                "top_target_genes": ";".join(
                    target.sort_values("whole_transcriptome_rank")["gene"].astype(str).head(12).tolist()
                ),
            }
        )
    return pd.DataFrame(rows)


def dataset_call(sub: pd.DataFrame) -> str:
    supported_true = int(
        (
            sub["target_background_call"].astype(str).eq("true_lr_target_rank_supported")
            & sub["true_lr"].astype(bool)
        ).sum()
    )
    leakage = int(sub["target_background_call"].astype(str).eq("high_expression_negative_target_leakage").sum())
    module_supported = int(
        (
            sub["target_background_call"].astype(str).isin(
                ["true_lr_target_rank_supported", "nonanchor_target_rank_supported"]
            )
            & sub["module_representative"].astype(bool)
        ).sum()
    )
    if supported_true >= 3 and leakage == 0:
        return "strong_selected_lr_target_background_support"
    if supported_true >= 1 and leakage == 0:
        return "selected_lr_target_background_support"
    if supported_true >= 1:
        return "target_support_with_negative_leakage_caveat"
    if module_supported >= 1:
        return "module_level_target_background_support"
    return "limited_target_background_support"


def module_call(has_targets: pd.Series, supported: pd.Series) -> str:
    if not len(has_targets) or int(has_targets.sum()) == 0:
        return "no_curated_targets_in_tested_module_pairs"
    if int(supported.sum()) > 0:
        return "module_has_target_supported_representative"
    return "module_targets_not_above_background"


def best_supported_lr(sub: pd.DataFrame) -> str:
    supported = sub[
        sub["target_background_call"].astype(str).isin(
            ["true_lr_target_rank_supported", "nonanchor_target_rank_supported"]
        )
    ].copy()
    if supported.empty:
        return ""
    supported = supported.sort_values(
        ["target_support_score", "target_top300_enrichment_bh_fdr", "rank"],
        ascending=[False, True, True],
    )
    return ";".join(supported["lr_pair"].astype(str).head(6).tolist())


def first_joined(values: pd.Series) -> str:
    vals = [str(v) for v in values.dropna().astype(str).tolist() if str(v)]
    return vals[0] if vals else ""


def fraction(numer: pd.Series, denom: pd.Series) -> float:
    denom_n = int(denom.sum())
    return float(int(numer.sum()) / denom_n) if denom_n else np.nan


def median(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    return float(vals.median()) if len(vals) else np.nan


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def pick(table: pd.DataFrame, **criteria: Any) -> pd.Series:
    if table.empty:
        return pd.Series(dtype=object)
    mask = pd.Series(True, index=table.index)
    for key, value in criteria.items():
        if key not in table.columns:
            return pd.Series(dtype=object)
        mask &= table[key].astype(str).eq(str(value))
    if not mask.any():
        return pd.Series(dtype=object)
    return table.loc[mask].iloc[0]


def get(obj: Any, key: str, default: Any = "") -> Any:
    if isinstance(obj, pd.Series):
        return obj[key] if key in obj.index else default
    return getattr(obj, key, default)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_status(
    dataset: pd.DataFrame,
    lr: pd.DataFrame,
    module: pd.DataFrame,
    gene_program: pd.DataFrame,
) -> None:
    lines = [
        "# Phase 12 Downstream Target-Background Audit",
        "",
        "## Scope",
        "",
        "This audit summarizes whether curated receiver target genes for scanned top LR, priority anchors and module representatives rank above all expressed genes in the receiver-side whole-transcriptome association. It is observational and residualized; it is not perturbational validation and not proof that every expanded-atlas LR has target support.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.dataset_target_background_call}`: scanned={int(row.n_lr_scanned)}, "
            f"with targets={int(row.n_lr_with_curated_targets)}, supported={int(row.n_target_supported_q_0_1)}, "
            f"true-supported={int(row.n_true_lr_target_supported_q_0_1)}, "
            f"module-supported={int(row.n_module_rep_target_supported_q_0_1)}, "
            f"high-expression leakage={int(row.n_high_expression_negative_target_leakage)}, "
            f"best={row.best_supported_lr or 'none'}."
        )
    lines += ["", "## Top Supported LR", ""]
    supported = lr[
        lr["target_background_call"].astype(str).isin(
            ["true_lr_target_rank_supported", "nonanchor_target_rank_supported"]
        )
    ].sort_values(["dataset_key", "target_support_score", "rank"], ascending=[True, False, True])
    for row in supported.head(20).itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.lr_pair}`: call={row.target_background_call}, rank={int(row.rank)}, "
            f"targets={int(row.n_targets_present)}, top300={int(row.n_targets_top300)}, "
            f"enrichment={fmt(get(row, 'target_top300_enrichment', np.nan))}, "
            f"FDR={fmt(get(row, 'target_top300_enrichment_bh_fdr', np.nan))}, "
            f"best target={get(row, 'best_target_gene', '')}."
        )
    lines += ["", "## Module Summary", ""]
    for row in module.itertuples(index=False):
        if str(row.module_target_background_call) == "module_has_target_supported_representative":
            lines.append(
                f"- `{row.dataset_key}` `{row.module}`: supported pairs={int(row.n_target_supported_pairs)}/{int(row.n_top_pairs_with_targets)} with target sets; best FDR={fmt(row.best_target_enrichment_fdr)}."
            )
    lines += ["", "## Claim Boundary", ""]
    lines.append(
        "Use this as selected-LR/module functional support. It supports communication programs where target sets rank above the full gene background, but it must not be written as pan-atlas downstream support for every predicted LR."
    )
    lines.append(f"Gene-program rows for supported LR: {len(gene_program)}.")
    (OUT_DIR / "phase12_downstream_target_background_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_manifest(
    lr: pd.DataFrame,
    dataset: pd.DataFrame,
    module: pd.DataFrame,
    gene_program: pd.DataFrame,
) -> None:
    manifest = {
        "task": "phase12_downstream_target_background_audit",
        "alpha": ALPHA,
        "inputs": [
            rel(DOWNSTREAM_DIR / "phase12_expanded_atlas_downstream_lr_summary.csv"),
            rel(DOWNSTREAM_DIR / "phase12_expanded_atlas_downstream_gene_table.csv"),
            rel(DOWNSTREAM_DIR / "phase12_expanded_atlas_downstream_module_summary.csv"),
        ],
        "outputs": [
            rel(OUT_DIR / "phase12_downstream_target_background_lr_audit.csv"),
            rel(OUT_DIR / "phase12_downstream_target_background_dataset_audit.csv"),
            rel(OUT_DIR / "phase12_downstream_target_background_module_audit.csv"),
            rel(OUT_DIR / "phase12_downstream_target_background_gene_program_audit.csv"),
            rel(OUT_DIR / "phase12_downstream_target_background_status.md"),
        ],
        "n_lr_rows": int(len(lr)),
        "n_dataset_rows": int(len(dataset)),
        "n_module_rows": int(len(module)),
        "n_gene_program_rows": int(len(gene_program)),
        "claim_boundary": "Selected top/priority/module LR target-background audit; not perturbational or pan-atlas functional validation.",
    }
    (OUT_DIR / "phase12_downstream_target_background_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def fmt(value: Any) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.3g}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
