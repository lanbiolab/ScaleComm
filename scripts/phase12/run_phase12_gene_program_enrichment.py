#!/usr/bin/env python
"""Phase 12 communication-dependent gene-program enrichment.

This extends the atlas downstream scan from LR target-rank support to
gene-program summaries for top LR pairs and LR modules. It uses local cached
GO/Reactome/KEGG/CellChatDB/OmniPath gene sets when available.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, hypergeom


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
OUT_DIR = PHASE12_DIR / "gene_program_enrichment"

GENE_SET_SOURCES = {
    "visium_hd_crc_full": PROJECT_ROOT / "outputs/phase10/full_transcriptome_commot_style/visium_hd_crc_p2crc_subset",
    "post_xenium_colon_full": PROJECT_ROOT / "outputs/phase10/full_transcriptome_commot_style/post_xenium_human_colon_cancer_subset",
    "human_lymph_node_visium_raw_full": PROJECT_ROOT / "outputs/phase12/full_transcriptome_inputs/human_lymph_node_visium_raw_full",
}

FULL_TX_DIRS = {
    "visium_hd_crc_full": PROJECT_ROOT / "outputs/phase10/full_transcriptome_commot_style/visium_hd_crc_p2crc_subset/full_transcriptome",
    "post_xenium_colon_full": PROJECT_ROOT / "outputs/phase10/full_transcriptome_commot_style/post_xenium_human_colon_cancer_subset/full_transcriptome",
    "human_lymph_node_visium_raw_full": PROJECT_ROOT / "outputs/phase12/full_transcriptome_inputs/human_lymph_node_visium_raw_full/full_transcriptome",
}

COMMOT_IMPACT_SOURCES = {
    "visium_hd_crc_full": PROJECT_ROOT
    / "outputs/phase10/full_transcriptome_commot_style/visium_hd_crc_p2crc_subset/results/communication_impact_summary.csv",
    "post_xenium_colon_full": PROJECT_ROOT
    / "outputs/phase10/full_transcriptome_commot_style/post_xenium_human_colon_cancer_subset/results/communication_impact_summary.csv",
}

LIBRARIES = {
    "reactome": "Reactome_2022.txt",
    "kegg": "KEGG_2021_Human.txt",
    "go_bp": "GO_Biological_Process_2023.txt",
}

BROAD_PROGRAM_KEYWORDS = (
    "ribosome",
    "translation",
    "peptide chain",
    "elongation",
    "initiation",
    "rrna",
    "trna",
    "mrna splicing",
    "spliceosome",
    "nonsense-mediated decay",
    "eif2",
    "eif4",
    "amino acid deficiency",
    "protein folding",
)

SPECIFIC_TERM_KEYWORDS = {
    "immune_chemokine": (
        "chemokine",
        "cytokine",
        "interleukin",
        "interferon",
        "tnf",
        "t cell",
        "b cell",
        "lymphocyte",
        "leukocyte",
        "macrophage",
        "monocyte",
        "antigen",
        "mhc",
        "complement",
        "inflammatory",
        "inflammation",
    ),
    "ecm_stroma": (
        "collagen",
        "extracellular matrix",
        "ecm",
        "integrin",
        "cell adhesion",
        "focal adhesion",
        "matrix organization",
        "laminin",
        "fibronectin",
    ),
    "migration_boundary": (
        "migration",
        "motility",
        "chemotaxis",
        "invasion",
        "wound healing",
        "angiogenesis",
        "vascular",
        "endothelial",
    ),
    "signaling_pathway": (
        "tgf",
        "notch",
        "wnt",
        "vegf",
        "jak-stat",
        "mapk",
        "pi3k",
        "nf-kappa",
        "rho",
        "rhoa",
        "rhoj",
        "rhov",
        "rhou",
        "gtpase cycle",
        "ras",
        "apoptosis",
    ),
}

SPECIFIC_CATEGORIES = set(SPECIFIC_TERM_KEYWORDS)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gene_table = pd.read_csv(PHASE12_DIR / "expanded_atlas_downstream_scan/phase12_expanded_atlas_downstream_gene_table.csv")
    lr_summary = pd.read_csv(PHASE12_DIR / "expanded_atlas_downstream_scan/phase12_expanded_atlas_downstream_lr_summary.csv")
    ranking = pd.read_csv(PHASE12_DIR / "atlas_model_expanded/phase12_atlas_model_expanded_calibrated_lr_ranking.csv")
    modules = pd.read_csv(PHASE12_DIR / "atlas_model_expanded/phase12_expanded_atlas_modules.csv")
    gene_table, lr_summary = normalize_expanded_inputs(gene_table, lr_summary)

    all_enrichment = []
    all_programs = []
    all_status = []
    for dataset_key, sub_gene in gene_table.groupby("dataset_key"):
        gene_sets, status = load_gene_sets(dataset_key)
        all_status.append(status.assign(dataset_key=dataset_key))
        programs = build_programs(dataset_key, sub_gene, lr_summary, ranking, modules)
        all_programs.append(programs)
        if gene_sets:
            universe = load_gene_universe(dataset_key)
            enrichment = run_enrichment(dataset_key, programs, gene_sets, universe, top_n=200)
        else:
            enrichment = pd.DataFrame()
        all_enrichment.append(enrichment)

    program_table = pd.concat(all_programs, ignore_index=True) if all_programs else pd.DataFrame()
    enrichment_table = pd.concat(all_enrichment, ignore_index=True) if all_enrichment else pd.DataFrame()
    enrichment_table = annotate_enrichment_terms(enrichment_table)
    status_table = pd.concat(all_status, ignore_index=True) if all_status else pd.DataFrame()
    summary = summarize_enrichment(enrichment_table, program_table)
    specificity = summarize_program_specificity(program_table, enrichment_table)
    module_summary = summarize_modules(program_table, enrichment_table, specificity)
    commot = summarize_commot_impact(lr_summary)

    program_table.to_csv(OUT_DIR / "phase12_gene_program_top_genes.csv", index=False)
    enrichment_table.to_csv(OUT_DIR / "phase12_gene_program_enrichment_all.csv", index=False)
    summary.to_csv(OUT_DIR / "phase12_gene_program_enrichment_summary.csv", index=False)
    specificity.to_csv(OUT_DIR / "phase12_gene_program_specificity_summary.csv", index=False)
    module_summary.to_csv(OUT_DIR / "phase12_module_gene_program_summary.csv", index=False)
    commot.to_csv(OUT_DIR / "phase12_commot_impact_comparison_summary.csv", index=False)
    status_table.to_csv(OUT_DIR / "phase12_gene_program_resource_status.csv", index=False)
    write_status(summary, specificity, module_summary, commot, status_table)
    write_manifest(status_table)

    print(summary.head(40).to_string(index=False) if not summary.empty else "No enrichment summary")
    print()
    print(specificity.to_string(index=False) if not specificity.empty else "No specificity summary")
    print()
    print(module_summary.to_string(index=False) if not module_summary.empty else "No module summary")


def normalize_expanded_inputs(gene_table: pd.DataFrame, lr_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gene_table = gene_table.copy()
    lr_summary = lr_summary.copy()
    if "is_nichenet_target" not in gene_table and "is_target" in gene_table:
        gene_table["is_nichenet_target"] = gene_table["is_target"]
    if "true_lr" not in gene_table and "lr_pair" in gene_table and "true_lr" in lr_summary:
        gene_table = gene_table.merge(
            lr_summary[["dataset_key", "lr_pair", "true_lr", "false_positive_high_expression"]].drop_duplicates(),
            on=["dataset_key", "lr_pair"],
            how="left",
        )
    if "false_positive_high_expression" not in gene_table:
        gene_table["false_positive_high_expression"] = False
    if "lr_rank" not in lr_summary and "rank" in lr_summary:
        lr_summary["lr_rank"] = pd.to_numeric(lr_summary["rank"], errors="coerce")
    if "mean_lr_score" not in lr_summary and "calibrated_score" in lr_summary:
        lr_summary["mean_lr_score"] = pd.to_numeric(lr_summary["calibrated_score"], errors="coerce")
    if "n_nichenet_targets_present" not in lr_summary and "n_targets_present" in lr_summary:
        lr_summary["n_nichenet_targets_present"] = pd.to_numeric(lr_summary["n_targets_present"], errors="coerce")
    return gene_table, lr_summary


def build_programs(
    dataset_key: str,
    gene_table: pd.DataFrame,
    lr_summary: pd.DataFrame,
    ranking: pd.DataFrame,
    modules: pd.DataFrame,
    *,
    top_k: int = 500,
) -> pd.DataFrame:
    rows = []
    lr_sub = lr_summary[lr_summary["dataset_key"].eq(dataset_key)].copy()
    rank_sub = ranking[ranking["dataset_key"].eq(dataset_key)].copy()
    candidate_pairs = set(lr_sub.sort_values("lr_rank")["lr_pair"].head(10).astype(str))
    candidate_pairs |= set(rank_sub.sort_values("rank")["lr_pair"].head(10).astype(str))
    candidate_pairs |= set(lr_sub[lr_sub["true_lr"].astype(bool)]["lr_pair"].astype(str))
    for lr_pair in sorted(candidate_pairs):
        genes = top_genes_for_lr(gene_table, lr_pair, top_k)
        if not genes:
            continue
        rank_row = rank_sub[rank_sub["lr_pair"].eq(lr_pair)].head(1)
        lr_row = lr_sub[lr_sub["lr_pair"].eq(lr_pair)].head(1)
        rows.extend(
            {
                "dataset_key": dataset_key,
                "program_id": f"{dataset_key}::LR::{lr_pair}",
                "program_label": f"LR::{lr_pair}",
                "program_type": "lr_pair",
                "lr_pair": lr_pair,
                "module": "",
                "gene": gene,
                "gene_rank": i + 1,
                "spearman_r": corr,
                "true_lr": bool(rank_row["true_lr"].iloc[0]) if len(rank_row) else bool(lr_row["true_lr"].iloc[0]) if len(lr_row) else False,
                "false_positive_high_expression": bool(rank_row["false_positive_high_expression"].iloc[0]) if len(rank_row) else bool(lr_row["false_positive_high_expression"].iloc[0]) if len(lr_row) else False,
                "lr_rank": int(rank_row["rank"].iloc[0]) if len(rank_row) else int(lr_row["lr_rank"].iloc[0]) if len(lr_row) else -1,
            }
            for i, (gene, corr) in enumerate(genes)
        )
    mod_sub = modules[modules["dataset_key"].eq(dataset_key)].copy()
    for mod in mod_sub.itertuples(index=False):
        lr_pairs = [x.strip() for x in str(mod.top_lr_pairs).split(";") if x.strip()]
        genes = combine_module_genes(gene_table, lr_pairs, top_k=top_k)
        if not genes:
            continue
        rows.extend(
            {
                "dataset_key": dataset_key,
                "program_id": f"{dataset_key}::MODULE::{mod.module}",
                "program_label": f"MODULE::{mod.module}",
                "program_type": "module",
                "lr_pair": ";".join(lr_pairs),
                "module": mod.module,
                "gene": gene,
                "gene_rank": i + 1,
                "spearman_r": score,
                "true_lr": bool(int(mod.n_priority_anchor) > 0),
                "false_positive_high_expression": bool(int(mod.n_high_expression_negative) > 0),
                "lr_rank": int(mod.best_rank),
            }
            for i, (gene, score) in enumerate(genes)
        )
    return pd.DataFrame(rows)


def top_genes_for_lr(gene_table: pd.DataFrame, lr_pair: str, top_k: int) -> list[tuple[str, float]]:
    sub = gene_table[gene_table["lr_pair"].astype(str).eq(lr_pair)].copy()
    if sub.empty:
        return []
    sub = sub.sort_values(["whole_transcriptome_rank", "spearman_r"], ascending=[True, False]).head(top_k)
    return [(str(r.gene).upper(), float(r.spearman_r)) for r in sub.itertuples(index=False)]


def combine_module_genes(gene_table: pd.DataFrame, lr_pairs: list[str], top_k: int) -> list[tuple[str, float]]:
    sub = gene_table[gene_table["lr_pair"].astype(str).isin(lr_pairs)].copy()
    if sub.empty:
        return []
    sub["weighted_score"] = pd.to_numeric(sub["spearman_r"], errors="coerce").fillna(0.0)
    agg = (
        sub.groupby("gene", as_index=False)
        .agg(score=("weighted_score", "mean"), best_rank=("whole_transcriptome_rank", "min"), n_lr=("lr_pair", "nunique"))
        .sort_values(["n_lr", "score", "best_rank"], ascending=[False, False, True])
        .head(top_k)
    )
    return [(str(r.gene).upper(), float(r.score)) for r in agg.itertuples(index=False)]


def load_gene_sets(dataset_key: str) -> tuple[dict[str, dict[str, set[str]]], pd.DataFrame]:
    base = GENE_SET_SOURCES.get(dataset_key)
    rows = []
    out: dict[str, dict[str, set[str]]] = {}
    if base is None:
        return {}, pd.DataFrame(
            [
                {
                    "library": "all",
                    "status": "missing_pathway_gene_set_directory",
                    "n_terms": 0,
                    "source_path": "",
                }
            ]
        )
    gene_set_dir = base / "pathway_gene_sets"
    cache_gene_set_dir = (
        PROJECT_ROOT
        / "outputs/phase10/full_transcriptome_commot_style/post_xenium_human_colon_cancer_subset/pathway_gene_sets"
    )
    for library, filename in LIBRARIES.items():
        path = gene_set_dir / filename
        if not path.exists():
            cached = cache_gene_set_dir / filename
            if cached.exists():
                path = cached
        terms = parse_enrichr_text(path) if path.exists() else {}
        if terms:
            out[library] = terms
        rows.append({"library": library, "status": "loaded" if terms else "missing_or_empty", "n_terms": len(terms), "source_path": rel(path)})
    cellchat_path = base / "cellchatdb/cellchatdb_human_pathway_genes.csv"
    if not cellchat_path.exists():
        cached_cellchat = (
            PROJECT_ROOT
            / "outputs/phase10/full_transcriptome_commot_style/post_xenium_human_colon_cancer_subset/cellchatdb/cellchatdb_human_pathway_genes.csv"
        )
        if cached_cellchat.exists():
            cellchat_path = cached_cellchat
    cellchat = parse_cellchatdb(cellchat_path)
    if cellchat:
        out["cellchatdb"] = cellchat
    rows.append({"library": "cellchatdb", "status": "loaded" if cellchat else "missing_or_empty", "n_terms": len(cellchat), "source_path": rel(cellchat_path)})
    omnipath_path = gene_set_dir / "omnipath_focus_lr_onehop_gene_sets.csv"
    if not omnipath_path.exists():
        cached_omnipath = cache_gene_set_dir / "omnipath_focus_lr_onehop_gene_sets.csv"
        if cached_omnipath.exists():
            omnipath_path = cached_omnipath
    omnipath = parse_omnipath(omnipath_path)
    if omnipath:
        out["omnipath"] = omnipath
    rows.append({"library": "omnipath", "status": "loaded" if omnipath else "missing_or_empty", "n_terms": len(omnipath), "source_path": rel(omnipath_path)})
    return out, pd.DataFrame(rows)


def run_enrichment(
    dataset_key: str,
    programs: pd.DataFrame,
    gene_sets: dict[str, dict[str, set[str]]],
    universe: set[str],
    *,
    top_n: int,
    min_term_size: int = 5,
    max_term_size: int = 500,
) -> pd.DataFrame:
    rows = []
    universe = {g.upper() for g in universe}
    if not universe:
        universe = {g.upper() for g in programs["gene"].astype(str)}
    n_universe = len(universe)
    if n_universe == 0:
        return pd.DataFrame()
    for program_id, sub in programs.groupby("program_id", sort=False):
        top = sub.sort_values("gene_rank").head(top_n)
        top_genes = {g.upper() for g in top["gene"].astype(str)} & universe
        meta = sub.iloc[0]
        for library, terms in gene_sets.items():
            lib_rows = []
            pvals = []
            for term, raw_genes in terms.items():
                term_genes = raw_genes & universe
                n_term = len(term_genes)
                if n_term < min_term_size or n_term > max_term_size:
                    continue
                overlap_genes = top_genes & term_genes
                overlap = len(overlap_genes)
                if overlap == 0:
                    continue
                p_value = float(hypergeom.sf(overlap - 1, n_universe, n_term, min(top_n, len(top_genes))))
                fold = float((overlap / max(min(top_n, len(top_genes)), 1)) / (n_term / max(n_universe, 1)))
                odds_ratio = float(
                    fisher_exact(
                        [
                            [overlap, max(min(top_n, len(top_genes)) - overlap, 0)],
                            [n_term - overlap, max(n_universe - n_term - min(top_n, len(top_genes)) + overlap, 0)],
                        ],
                        alternative="greater",
                    )[0]
                )
                pvals.append(p_value)
                lib_rows.append(
                    {
                        "dataset_key": dataset_key,
                        "program_id": program_id,
                        "program_type": meta.program_type,
                        "module": meta.module,
                        "lr_pair": meta.lr_pair,
                        "true_lr": bool(meta.true_lr),
                        "false_positive_high_expression": bool(meta.false_positive_high_expression),
                        "library": library,
                        "term": term,
                        "top_n": min(top_n, len(top_genes)),
                        "n_universe": n_universe,
                        "n_term_genes": n_term,
                        "overlap": overlap,
                        "p_value": p_value,
                        "fold_enrichment": fold,
                        "odds_ratio": odds_ratio,
                        "overlap_genes": ",".join(sorted(overlap_genes)),
                    }
                )
            if lib_rows:
                fdr = bh_adjust(np.asarray(pvals, dtype=float))
                for row, q in zip(lib_rows, fdr):
                    row["bh_fdr"] = float(q)
                rows.extend(lib_rows)
    return pd.DataFrame(rows)


def load_gene_universe(dataset_key: str) -> set[str]:
    full_tx_dir = FULL_TX_DIRS.get(dataset_key)
    if full_tx_dir is None:
        return set()
    genes_path = full_tx_dir / "genes.csv"
    if not genes_path.exists():
        return set()
    genes = pd.read_csv(genes_path)["gene"].astype(str).str.upper()
    return set(genes)


def annotate_enrichment_terms(enrichment: pd.DataFrame) -> pd.DataFrame:
    if enrichment.empty:
        return enrichment
    out = enrichment.copy()
    categories = [classify_term_category(term, library) for term, library in zip(out["term"], out["library"])]
    out["term_category"] = categories
    out["is_broad_program_term"] = out["term_category"].eq("broad_housekeeping")
    out["is_communication_specific_term"] = out["term_category"].isin(SPECIFIC_CATEGORIES)
    return out


def classify_term_category(term: object, library: object = "") -> str:
    text = str(term).lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", text)
    if any(keyword_matches(normalized, keyword) for keyword in BROAD_PROGRAM_KEYWORDS):
        return "broad_housekeeping"
    for category, keywords in SPECIFIC_TERM_KEYWORDS.items():
        if any(keyword_matches(normalized, keyword) for keyword in keywords):
            return category
    if "omnipath:" in text or "cellchatdb:" in text:
        return "lr_pathway_resource"
    return "other"


def keyword_matches(text: str, keyword: str) -> bool:
    normalized_keyword = re.sub(r"[^a-z0-9]+", " ", keyword.lower()).strip()
    if not normalized_keyword:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(normalized_keyword) + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def summarize_enrichment(enrichment: pd.DataFrame, programs: pd.DataFrame) -> pd.DataFrame:
    if enrichment.empty:
        return pd.DataFrame()
    rows = []
    for keys, sub in enrichment.groupby(["dataset_key", "program_id", "program_type", "library"], sort=False):
        dataset_key, program_id, program_type, library = keys
        sig = sub[sub["bh_fdr"].le(0.05)]
        best = sub.sort_values(["bh_fdr", "p_value"]).iloc[0]
        specific_sig = sig[sig["is_communication_specific_term"].astype(bool)]
        broad_sig = sig[sig["is_broad_program_term"].astype(bool)]
        best_specific = specific_sig.sort_values(["bh_fdr", "p_value"]).head(1)
        program_mask = programs["dataset_key"].eq(dataset_key) & programs["program_id"].eq(program_id)
        meta = programs[program_mask].iloc[0]
        rows.append(
            {
                "dataset_key": dataset_key,
                "program_id": program_id,
                "program_label": meta.program_label,
                "program_type": program_type,
                "module": meta.module,
                "lr_pair": meta.lr_pair,
                "library": library,
                "true_lr": bool(meta.true_lr),
                "false_positive_high_expression": bool(meta.false_positive_high_expression),
                "n_program_genes": int(programs[program_mask]["gene"].nunique()),
                "n_enriched_terms": int(len(sub)),
                "n_significant_terms_fdr_0_05": int(len(sig)),
                "n_broad_significant_terms_fdr_0_05": int(len(broad_sig)),
                "n_specific_significant_terms_fdr_0_05": int(len(specific_sig)),
                "best_term": best.term,
                "best_term_category": best.term_category,
                "best_term_is_broad_program": bool(best.is_broad_program_term),
                "best_term_fdr": float(best.bh_fdr),
                "best_term_fold": float(best.fold_enrichment),
                "best_term_overlap": int(best.overlap),
                "best_specific_term": best_specific["term"].iloc[0] if len(best_specific) else "",
                "best_specific_term_category": best_specific["term_category"].iloc[0] if len(best_specific) else "",
                "best_specific_term_fdr": float(best_specific["bh_fdr"].iloc[0]) if len(best_specific) else np.nan,
            }
        )
    summary = pd.DataFrame(rows)
    return add_program_specificity(summary)


def add_program_specificity(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    for (dataset_key, library), sub in summary.groupby(["dataset_key", "library"]):
        neg = sub[sub["false_positive_high_expression"].astype(bool)]
        best_neg_sig = int(neg["n_significant_terms_fdr_0_05"].max()) if len(neg) else 0
        best_neg_fdr = float(neg["best_term_fdr"].min()) if len(neg) else np.nan
        mask = summary["dataset_key"].eq(dataset_key) & summary["library"].eq(library)
        summary.loc[mask, "best_negative_significant_terms"] = best_neg_sig
        summary.loc[mask, "best_negative_best_term_fdr"] = best_neg_fdr
        summary.loc[mask, "significant_terms_minus_best_negative"] = (
            summary.loc[mask, "n_significant_terms_fdr_0_05"].astype(float) - best_neg_sig
        )
    return summary.sort_values(["dataset_key", "program_type", "program_id", "library"])


def summarize_program_specificity(programs: pd.DataFrame, enrichment: pd.DataFrame) -> pd.DataFrame:
    if programs.empty:
        return pd.DataFrame()
    rows = []
    for (dataset_key, program_id), sub in programs.groupby(["dataset_key", "program_id"]):
        enr = enrichment[
            enrichment["dataset_key"].eq(dataset_key) & enrichment["program_id"].eq(program_id)
        ] if not enrichment.empty else pd.DataFrame()
        sig = enr[enr["bh_fdr"].le(0.05)] if not enr.empty else pd.DataFrame()
        broad_sig = sig[sig["is_broad_program_term"].astype(bool)] if not sig.empty else pd.DataFrame()
        specific_sig = sig[sig["is_communication_specific_term"].astype(bool)] if not sig.empty else pd.DataFrame()
        best = sig.sort_values(["bh_fdr", "p_value"]).head(1) if not sig.empty else pd.DataFrame()
        best_specific = specific_sig.sort_values(["bh_fdr", "p_value"]).head(1) if not specific_sig.empty else pd.DataFrame()
        meta = sub.iloc[0]
        classification = classify_program_specificity(meta, sig, broad_sig, specific_sig)
        rows.append(
            {
                "dataset_key": dataset_key,
                "program_id": program_id,
                "program_label": meta.program_label,
                "program_type": meta.program_type,
                "module": meta.module,
                "lr_pair": meta.lr_pair,
                "true_lr": bool(meta.true_lr),
                "false_positive_high_expression": bool(meta.false_positive_high_expression),
                "n_program_genes": int(sub["gene"].nunique()),
                "n_significant_terms_fdr_0_05": int(len(sig)),
                "n_broad_significant_terms_fdr_0_05": int(len(broad_sig)),
                "n_specific_significant_terms_fdr_0_05": int(len(specific_sig)),
                "specific_fraction_of_significant": float(len(specific_sig) / len(sig)) if len(sig) else 0.0,
                "best_term": best["term"].iloc[0] if len(best) else "",
                "best_term_category": best["term_category"].iloc[0] if len(best) else "",
                "best_term_fdr": float(best["bh_fdr"].iloc[0]) if len(best) else np.nan,
                "best_specific_term": best_specific["term"].iloc[0] if len(best_specific) else "",
                "best_specific_term_category": best_specific["term_category"].iloc[0] if len(best_specific) else "",
                "best_specific_term_fdr": float(best_specific["bh_fdr"].iloc[0]) if len(best_specific) else np.nan,
                "program_specificity_class": classification,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for dataset_key, sub in out.groupby("dataset_key"):
        neg = sub[sub["false_positive_high_expression"].astype(bool)]
        best_neg_specific = int(neg["n_specific_significant_terms_fdr_0_05"].max()) if len(neg) else 0
        best_neg_broad = int(neg["n_broad_significant_terms_fdr_0_05"].max()) if len(neg) else 0
        mask = out["dataset_key"].eq(dataset_key)
        out.loc[mask, "specific_terms_minus_best_negative"] = (
            out.loc[mask, "n_specific_significant_terms_fdr_0_05"].astype(float) - best_neg_specific
        )
        out.loc[mask, "broad_terms_minus_best_negative"] = (
            out.loc[mask, "n_broad_significant_terms_fdr_0_05"].astype(float) - best_neg_broad
        )
    return out.sort_values(["dataset_key", "program_type", "program_id"])


def classify_program_specificity(
    meta: pd.Series,
    sig: pd.DataFrame,
    broad_sig: pd.DataFrame,
    specific_sig: pd.DataFrame,
) -> str:
    if sig.empty:
        return "no_enrichment"
    if specific_sig.empty and not broad_sig.empty:
        return "broad_program_warning"
    if bool(meta.false_positive_high_expression) and not specific_sig.empty:
        return "negative_specific_leakage"
    if bool(meta.true_lr) and not specific_sig.empty and len(specific_sig) >= len(broad_sig):
        return "specific_true_lr_support"
    if bool(meta.true_lr) and not specific_sig.empty:
        return "mixed_true_lr_support_broad_dominant"
    if not specific_sig.empty:
        return "specific_non_anchor_or_background"
    return "mixed_or_unclear"


def summarize_modules(programs: pd.DataFrame, enrichment: pd.DataFrame, specificity: pd.DataFrame) -> pd.DataFrame:
    if programs.empty:
        return pd.DataFrame()
    rows = []
    modules = programs[programs["program_type"].eq("module")]
    if modules.empty:
        return pd.DataFrame()
    for (dataset_key, program_id), sub in modules.groupby(["dataset_key", "program_id"]):
        enr = enrichment[
            enrichment["dataset_key"].eq(dataset_key) & enrichment["program_id"].eq(program_id)
        ] if not enrichment.empty else pd.DataFrame()
        sig = enr[enr["bh_fdr"].le(0.05)] if not enr.empty else pd.DataFrame()
        best = enr.sort_values(["bh_fdr", "p_value"]).head(1)
        meta = sub.iloc[0]
        rows.append(
            {
                "dataset_key": meta.dataset_key,
                "program_id": program_id,
                "program_label": meta.program_label,
                "module": meta.module,
                "lr_pair": meta.lr_pair,
                "n_program_genes": int(sub["gene"].nunique()),
                "n_lr_pairs": len([x for x in str(meta.lr_pair).split(";") if x]),
                "true_lr_module": bool(meta.true_lr),
                "negative_containing_module": bool(meta.false_positive_high_expression),
                "n_significant_terms_fdr_0_05": int(len(sig)),
                "best_library": best["library"].iloc[0] if len(best) else "",
                "best_term": best["term"].iloc[0] if len(best) else "",
                "best_term_category": best["term_category"].iloc[0] if len(best) else "",
                "best_term_fdr": float(best["bh_fdr"].iloc[0]) if len(best) else np.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values(["dataset_key", "module"])
    if not specificity.empty and not out.empty:
        keep = [
            "dataset_key",
            "program_id",
            "n_broad_significant_terms_fdr_0_05",
            "n_specific_significant_terms_fdr_0_05",
            "specific_fraction_of_significant",
            "best_specific_term",
            "best_specific_term_category",
            "best_specific_term_fdr",
            "specific_terms_minus_best_negative",
            "program_specificity_class",
        ]
        out = out.merge(specificity[keep], on=["dataset_key", "program_id"], how="left")
    return out


def summarize_commot_impact(lr_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, path in COMMOT_IMPACT_SOURCES.items():
        if not path.exists():
            rows.append({"dataset_key": dataset_key, "status": "missing_commot_impact", "source_path": rel(path)})
            continue
        impact = pd.read_csv(path)
        sub = impact[impact["method"].isin(["spatialself", "commot"])].copy()
        if sub.empty:
            rows.append({"dataset_key": dataset_key, "status": "no_spatialself_commot_rows", "source_path": rel(path)})
            continue
        for method, table in sub.groupby("method"):
            pos = table[table["lr_status"].eq("positive")]
            neg = table[table["lr_status"].astype(str).str.contains("negative", na=False)]
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "method": method,
                    "status": "completed",
                    "n_positive_lr": int(len(pos)),
                    "n_negative_lr": int(len(neg)),
                    "mean_positive_delta_cv_r2": float(pos["delta_cv_r2"].mean()) if len(pos) else np.nan,
                    "max_negative_delta_cv_r2": float(neg["delta_cv_r2"].max()) if len(neg) else np.nan,
                    "mean_positive_rf_importance": float(pos["rf_comm_importance"].mean()) if len(pos) else np.nan,
                    "max_negative_rf_importance": float(neg["rf_comm_importance"].max()) if len(neg) else np.nan,
                    "best_positive_lr_by_delta": str(pos.sort_values("delta_cv_r2", ascending=False)["lr_pair"].iloc[0]) if len(pos) else "",
                    "best_positive_delta_cv_r2": float(pos["delta_cv_r2"].max()) if len(pos) else np.nan,
                    "source_path": rel(path),
                }
            )
    return pd.DataFrame(rows)


def parse_enrichr_text(path: Path) -> dict[str, set[str]]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        fields = [x.strip() for x in line.split("\t") if x.strip()]
        if len(fields) < 3:
            continue
        out[fields[0]] = {g.upper() for g in fields[2:] if g}
    return out


def parse_cellchatdb(path: Path) -> dict[str, set[str]]:
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    out = {}
    for row in table.itertuples(index=False):
        genes = parse_genes(getattr(row, "genes", ""))
        if genes:
            out[f"CellChatDB:{getattr(row, 'pathway', '')}"] = genes
    return out


def parse_omnipath(path: Path) -> dict[str, set[str]]:
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    out = {}
    for row in table.itertuples(index=False):
        genes = parse_genes(getattr(row, "genes", ""))
        if genes:
            out[f"OmniPath:{getattr(row, 'lr_pair', '')}:{getattr(row, 'direction', '')}"] = genes
    return out


def parse_genes(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {g.strip().upper() for g in str(value).replace(",", ";").split(";") if g.strip()}


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    out = np.full(len(pvalues), np.nan, dtype=float)
    finite = np.isfinite(pvalues)
    if not finite.any():
        return out
    idx = np.flatnonzero(finite)
    p = pvalues[idx]
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(ranked) / (np.arange(len(ranked)) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out[idx[order]] = np.minimum(adjusted, 1.0)
    return out


def write_status(
    summary: pd.DataFrame,
    specificity: pd.DataFrame,
    module_summary: pd.DataFrame,
    commot: pd.DataFrame,
    status: pd.DataFrame,
) -> None:
    lines = [
        "# Phase 12 Gene-Program Enrichment",
        "",
        "## Scope",
        "",
        "This summarizes communication-dependent gene programs from the Phase 12 atlas downstream scan. It uses top receiver-associated genes for LR pairs and LR modules, then tests cached GO/Reactome/KEGG/CellChatDB/OmniPath gene sets when available.",
        "",
        "## Resource Status",
        "",
    ]
    for row in status.itertuples(index=False):
        lines.append(f"- `{row.dataset_key}` {row.library}: {row.status}; terms={row.n_terms}; source=`{row.source_path}`")
    if not summary.empty:
        lines += ["", "## LR Program Highlights", ""]
        top = summary.sort_values(["dataset_key", "program_type", "true_lr", "best_term_fdr"], ascending=[True, True, False, True]).groupby(["dataset_key", "program_type"]).head(5)
        for row in top.itertuples(index=False):
            lines.append(
                f"- `{row.dataset_key}` {row.program_id} {row.library}: best `{row.best_term}` ({row.best_term_category}) FDR={fmt(row.best_term_fdr)}, significant terms={int(row.n_significant_terms_fdr_0_05)}, specific terms={int(row.n_specific_significant_terms_fdr_0_05)}, delta-vs-best-negative={fmt(row.significant_terms_minus_best_negative)}"
            )
    if not specificity.empty:
        lines += ["", "## Program Specificity", ""]
        for dataset_key, sub in specificity.groupby("dataset_key"):
            class_counts = sub["program_specificity_class"].value_counts().to_dict()
            pieces = ", ".join(f"{k}={v}" for k, v in class_counts.items())
            support_mask = sub["program_specificity_class"].isin(
                ["specific_true_lr_support", "mixed_true_lr_support_broad_dominant"]
            )
            leakage_count = int(sub["program_specificity_class"].eq("negative_specific_leakage").sum())
            top_specific = sub[support_mask].sort_values(
                ["n_specific_significant_terms_fdr_0_05", "best_specific_term_fdr"],
                ascending=[False, True],
            ).head(3)
            top_text = "; ".join(
                f"{r.program_label}: {int(r.n_specific_significant_terms_fdr_0_05)} specific, best {r.best_specific_term} FDR={fmt(r.best_specific_term_fdr)}"
                for r in top_specific.itertuples(index=False)
            )
            lines.append(
                f"- `{dataset_key}` classes: {pieces}. True/mixed support examples: {top_text or 'none'}; negative leakage programs={leakage_count}"
            )
    if not module_summary.empty:
        lines += ["", "## Module Programs", ""]
        for row in module_summary.itertuples(index=False):
            lines.append(
                f"- `{row.dataset_key}` {row.module}: LR={row.lr_pair}; significant terms={int(row.n_significant_terms_fdr_0_05)}; specific terms={fmt(getattr(row, 'n_specific_significant_terms_fdr_0_05', np.nan))}; best={row.best_term} ({row.best_term_category}) FDR={fmt(row.best_term_fdr)}"
            )
    if not commot.empty:
        lines += ["", "## COMMOT Impact Comparison", ""]
        for row in commot.itertuples(index=False):
            if getattr(row, "status", "") != "completed":
                lines.append(f"- `{row.dataset_key}`: {row.status}")
                continue
            lines.append(
                f"- `{row.dataset_key}` {row.method}: mean positive delta R2={fmt(row.mean_positive_delta_cv_r2)}, max negative delta R2={fmt(row.max_negative_delta_cv_r2)}, best positive={row.best_positive_lr_by_delta} ({fmt(row.best_positive_delta_cv_r2)})"
            )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "- Enrichment is computed from observational receiver-gene associations, not perturbation data.",
        "- GO/Reactome/KEGG/CellChatDB/OmniPath resources are loaded for all processed Phase 12 atlas datasets; lymph node currently shows broad translation/ribosome-dominated programs rather than clean chemokine-specific downstream biology.",
    ]
    (OUT_DIR / "phase12_gene_program_enrichment_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(status: pd.DataFrame) -> None:
    manifest = {
        "task": "phase12_gene_program_enrichment",
        "inputs": [
            "outputs/phase12/expanded_atlas_downstream_scan/phase12_expanded_atlas_downstream_gene_table.csv",
            "outputs/phase12/expanded_atlas_downstream_scan/phase12_expanded_atlas_downstream_lr_summary.csv",
            "outputs/phase12/atlas_model_expanded/phase12_expanded_atlas_modules.csv",
        ],
        "outputs": [
            "phase12_gene_program_top_genes.csv",
            "phase12_gene_program_enrichment_all.csv",
            "phase12_gene_program_enrichment_summary.csv",
            "phase12_gene_program_specificity_summary.csv",
            "phase12_module_gene_program_summary.csv",
            "phase12_commot_impact_comparison_summary.csv",
            "phase12_gene_program_resource_status.csv",
            "phase12_gene_program_enrichment_status.md",
        ],
        "resource_status_rows": int(len(status)),
        "claim_boundary": "Observational gene-program enrichment from Phase 12 atlas downstream scan; not causal validation.",
    }
    (OUT_DIR / "phase12_gene_program_enrichment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def fmt(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "NA"
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(val) >= 100:
        return f"{val:.0f}"
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
