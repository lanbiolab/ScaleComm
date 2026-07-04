#!/usr/bin/env python
"""Pathway-specific receiver-program audit for Phase 12 Goal 2.

This post-processes the gene-program enrichment output to separate clean
pathway-specific support from broad housekeeping-dominated programs and
negative-control leakage.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
IN_DIR = PHASE12_DIR / "gene_program_enrichment"
OUT_DIR = PHASE12_DIR / "pathway_specific_programs"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    specificity_path = IN_DIR / "phase12_gene_program_specificity_summary.csv"
    enrichment_path = IN_DIR / "phase12_gene_program_enrichment_summary.csv"
    specificity = pd.read_csv(specificity_path)
    enrichment = pd.read_csv(enrichment_path)

    program_audit = build_program_audit(specificity)
    dataset_summary = summarize_datasets(program_audit)
    candidate_table = build_candidate_table(program_audit)
    term_table = build_specific_term_table(enrichment, program_audit)

    program_audit.to_csv(OUT_DIR / "phase12_pathway_specific_program_audit.csv", index=False)
    dataset_summary.to_csv(OUT_DIR / "phase12_pathway_specific_dataset_summary.csv", index=False)
    candidate_table.to_csv(OUT_DIR / "phase12_pathway_specific_candidate_programs.csv", index=False)
    term_table.to_csv(OUT_DIR / "phase12_pathway_specific_terms.csv", index=False)
    write_status(dataset_summary, candidate_table)
    write_manifest(program_audit, dataset_summary, candidate_table, term_table)
    print(dataset_summary.to_string(index=False))
    print()
    print(candidate_table.to_string(index=False))


def build_program_audit(specificity: pd.DataFrame) -> pd.DataFrame:
    table = specificity.copy()
    table["is_true_nonnegative_program"] = table["true_lr"].fillna(False).astype(bool) & ~table[
        "false_positive_high_expression"
    ].fillna(False).astype(bool)
    table["has_specific_terms"] = pd.to_numeric(table["n_specific_significant_terms_fdr_0_05"], errors="coerce").fillna(0).gt(0)
    table["has_broad_terms"] = pd.to_numeric(table["n_broad_significant_terms_fdr_0_05"], errors="coerce").fillna(0).gt(0)
    table["specific_beats_best_negative"] = pd.to_numeric(
        table["specific_terms_minus_best_negative"], errors="coerce"
    ).fillna(-np.inf).gt(0)
    table["clean_pathway_specific_support"] = (
        table["is_true_nonnegative_program"]
        & table["has_specific_terms"]
        & table["specific_beats_best_negative"]
        & ~table["has_broad_terms"]
    )
    table["mixed_specific_support_broad_dominant"] = (
        table["is_true_nonnegative_program"]
        & table["has_specific_terms"]
        & table["specific_beats_best_negative"]
        & table["has_broad_terms"]
    )
    table["negative_specific_leakage_flag"] = table["program_specificity_class"].astype(str).eq("negative_specific_leakage")
    table["pathway_specific_call"] = np.select(
        [
            table["clean_pathway_specific_support"],
            table["mixed_specific_support_broad_dominant"],
            table["negative_specific_leakage_flag"],
            table["has_specific_terms"],
        ],
        [
            "clean_specific_support",
            "specific_support_broad_dominant",
            "negative_specific_leakage",
            "specific_terms_present_but_not_clean",
        ],
        default="no_specific_support",
    )
    return table


def summarize_datasets(program_audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in program_audit.groupby("dataset_key", sort=False):
        true_sub = sub[sub["is_true_nonnegative_program"]]
        rows.append(
            {
                "dataset_key": dataset_key,
                "n_programs": int(len(sub)),
                "n_true_nonnegative_programs": int(len(true_sub)),
                "n_clean_specific_support": int(sub["clean_pathway_specific_support"].sum()),
                "n_mixed_specific_support_broad_dominant": int(sub["mixed_specific_support_broad_dominant"].sum()),
                "n_negative_specific_leakage": int(sub["negative_specific_leakage_flag"].sum()),
                "best_clean_program": best_program(sub[sub["clean_pathway_specific_support"]]),
                "best_mixed_program": best_program(sub[sub["mixed_specific_support_broad_dominant"]]),
                "dataset_program_call": dataset_call(sub),
            }
        )
    return pd.DataFrame(rows)


def build_candidate_table(program_audit: pd.DataFrame) -> pd.DataFrame:
    keep = program_audit[
        program_audit["pathway_specific_call"].isin(
            ["clean_specific_support", "specific_support_broad_dominant", "negative_specific_leakage"]
        )
    ].copy()
    if keep.empty:
        return keep
    keep["sort_fdr"] = pd.to_numeric(keep["best_specific_term_fdr"], errors="coerce").fillna(np.inf)
    keep["sort_specific"] = pd.to_numeric(keep["n_specific_significant_terms_fdr_0_05"], errors="coerce").fillna(0)
    cols = [
        "dataset_key",
        "program_label",
        "program_type",
        "module",
        "lr_pair",
        "pathway_specific_call",
        "true_lr",
        "false_positive_high_expression",
        "n_specific_significant_terms_fdr_0_05",
        "n_broad_significant_terms_fdr_0_05",
        "specific_fraction_of_significant",
        "best_specific_term",
        "best_specific_term_category",
        "best_specific_term_fdr",
        "specific_terms_minus_best_negative",
        "program_specificity_class",
    ]
    return keep.sort_values(
        ["dataset_key", "pathway_specific_call", "sort_fdr", "sort_specific"],
        ascending=[True, True, True, False],
    )[cols]


def build_specific_term_table(enrichment: pd.DataFrame, program_audit: pd.DataFrame) -> pd.DataFrame:
    supported_ids = set(
        program_audit[
            program_audit["pathway_specific_call"].isin(["clean_specific_support", "specific_support_broad_dominant"])
        ]["program_id"].astype(str)
    )
    table = enrichment[
        enrichment["program_id"].astype(str).isin(supported_ids)
        & ~enrichment["best_term_is_broad_program"].fillna(False).astype(bool)
        & (pd.to_numeric(enrichment["best_specific_term_fdr"], errors="coerce").fillna(np.inf) <= 0.05)
    ].copy()
    cols = [
        "dataset_key",
        "program_label",
        "program_type",
        "library",
        "best_specific_term",
        "best_specific_term_category",
        "best_specific_term_fdr",
        "n_specific_significant_terms_fdr_0_05",
        "significant_terms_minus_best_negative",
    ]
    existing = [c for c in cols if c in table.columns]
    return table.sort_values(["dataset_key", "best_specific_term_fdr"])[existing]


def dataset_call(sub: pd.DataFrame) -> str:
    clean = int(sub["clean_pathway_specific_support"].sum())
    mixed = int(sub["mixed_specific_support_broad_dominant"].sum())
    leakage = int(sub["negative_specific_leakage_flag"].sum())
    if clean >= 2 and leakage == 0:
        return "clean_pathway_specific_support"
    if clean + mixed > leakage:
        return "usable_specific_support_with_caveats"
    if leakage > 0:
        return "specific_signal_confounded_by_negative_leakage"
    return "no_clean_pathway_specific_support"


def best_program(table: pd.DataFrame) -> str:
    if table.empty:
        return ""
    tmp = table.copy()
    tmp["sort_fdr"] = pd.to_numeric(tmp["best_specific_term_fdr"], errors="coerce").fillna(np.inf)
    row = tmp.sort_values(["sort_fdr", "n_specific_significant_terms_fdr_0_05"], ascending=[True, False]).iloc[0]
    return f"{row.program_label}: {row.best_specific_term} (FDR={row.best_specific_term_fdr:.3g})"


def write_status(dataset_summary: pd.DataFrame, candidate_table: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Pathway-Specific Receiver Program Audit",
        "",
        "## Scope",
        "",
        "This audit separates clean pathway-specific receiver programs from broad housekeeping-dominated programs and negative-control leakage using the existing whole-transcriptome gene-program enrichment results.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset_summary.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}`: clean={int(row.n_clean_specific_support)}, mixed-broad={int(row.n_mixed_specific_support_broad_dominant)}, "
            f"negative leakage={int(row.n_negative_specific_leakage)}; call={row.dataset_program_call}; best clean=`{row.best_clean_program}`; best mixed=`{row.best_mixed_program}`"
        )
    lines.extend(["", "## Candidate Programs", ""])
    for row in candidate_table.head(30).itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.program_label}` {row.pathway_specific_call}: "
            f"{row.best_specific_term} ({row.best_specific_term_category}) FDR={row.best_specific_term_fdr:.3g}; "
            f"specific={int(row.n_specific_significant_terms_fdr_0_05)}, broad={int(row.n_broad_significant_terms_fdr_0_05)}"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This is observational receiver-side gene association, not perturbation. Broad translation/ribosome terms and negative-specific leakage remain explicit caveats, especially for lymph-node chemokine programs.",
        ]
    )
    (OUT_DIR / "phase12_pathway_specific_program_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(
    program_audit: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    candidate_table: pd.DataFrame,
    term_table: pd.DataFrame,
) -> None:
    manifest = {
        "task": "phase12_pathway_specific_program_audit",
        "inputs": {
            "specificity_summary": rel(IN_DIR / "phase12_gene_program_specificity_summary.csv"),
            "enrichment_summary": rel(IN_DIR / "phase12_gene_program_enrichment_summary.csv"),
        },
        "outputs": {
            "program_audit": rel(OUT_DIR / "phase12_pathway_specific_program_audit.csv"),
            "dataset_summary": rel(OUT_DIR / "phase12_pathway_specific_dataset_summary.csv"),
            "candidate_programs": rel(OUT_DIR / "phase12_pathway_specific_candidate_programs.csv"),
            "specific_terms": rel(OUT_DIR / "phase12_pathway_specific_terms.csv"),
            "status": rel(OUT_DIR / "phase12_pathway_specific_program_status.md"),
        },
        "n_programs": int(len(program_audit)),
        "n_candidate_programs": int(len(candidate_table)),
        "n_specific_term_rows": int(len(term_table)),
        "claim_boundary": "Post-processing audit of observational receiver-gene enrichment; not perturbational validation.",
    }
    (OUT_DIR / "phase12_pathway_specific_program_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
