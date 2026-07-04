#!/usr/bin/env python
"""Run pathway-level enrichment for Phase 10 full-transcriptome gene scans."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, hypergeom


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOAL5_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
PHASE10_DIR: Path
RESULT_DIR: Path
GENE_SET_DIR: Path
CELLCHATDB_PATH: Path
STANDARD_INPUT_DIR: Path
FOCUS_LR: dict[str, str]
POSITIVE_LR: set[str]
NEGATIVE_LR: set[str]
DATASET_KEY: str
DATASET_LABEL: str
ENRICHR_URL = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName={library}"
OMNIPATH_URL = (
    "https://omnipathdb.org/interactions?genesymbols=1&fields=sources,references&format=json"
    "&datasets=omnipath&organisms=9606&partners={genes}"
)

LIBRARIES = {
    "reactome": "Reactome_2022",
    "kegg": "KEGG_2021_Human",
    "go_bp": "GO_Biological_Process_2023",
}


def configure_dataset(dataset: str) -> None:
    global PHASE10_DIR
    global RESULT_DIR
    global GENE_SET_DIR
    global CELLCHATDB_PATH
    global STANDARD_INPUT_DIR
    global FOCUS_LR
    global POSITIVE_LR
    global NEGATIVE_LR
    global DATASET_KEY
    global DATASET_LABEL

    DATASET_KEY = dataset
    if dataset == "post_xenium":
        DATASET_LABEL = "post-Xenium human colon cancer 10x Visium CytAssist subset"
        slug = "post_xenium_human_colon_cancer_subset"
        STANDARD_INPUT_DIR = GOAL5_DIR / "standardized_inputs/post_xenium_human_colon_cancer_subset"
        FOCUS_LR = {
            "SPP1-CD44": "post_xenium_crc_lr_001_SPP1_CD44",
            "TGFB1-TGFBR2": "post_xenium_crc_lr_000_TGFB1_TGFBR2",
            "VEGFA-KDR": "post_xenium_crc_lr_003_VEGFA_KDR",
            "ACTB-KRT8": "post_xenium_crc_lr_005_ACTB_KRT8_negative",
            "COL1A1-EPCAM": "post_xenium_crc_lr_006_COL1A1_EPCAM_negative",
        }
        POSITIVE_LR = {"SPP1-CD44", "TGFB1-TGFBR2", "VEGFA-KDR"}
        NEGATIVE_LR = {"ACTB-KRT8", "COL1A1-EPCAM"}
    elif dataset == "visium_hd_crc":
        DATASET_LABEL = "Visium HD CRC P2CRC 8 um bin subset"
        slug = "visium_hd_crc_p2crc_subset"
        STANDARD_INPUT_DIR = GOAL5_DIR / "standardized_inputs/visium_hd_crc_p2crc_subset"
        FOCUS_LR = {
            "SPP1-CD44": "visium_hd_crc_lr_000_SPP1_CD44",
            "CXCL12-CXCR4": "visium_hd_crc_lr_001_CXCL12_CXCR4",
            "VEGFA-KDR": "visium_hd_crc_lr_002_VEGFA_KDR",
            "TGFB1-TGFBR2": "visium_hd_crc_lr_003_TGFB1_TGFBR2",
            "CD274-PDCD1": "visium_hd_crc_lr_004_CD274_PDCD1",
            "ACTB-KRT8": "visium_hd_crc_lr_005_ACTB_KRT8_negative",
            "COL1A1-EPCAM": "visium_hd_crc_lr_006_COL1A1_EPCAM_negative",
        }
        POSITIVE_LR = {"SPP1-CD44", "CXCL12-CXCR4", "VEGFA-KDR", "TGFB1-TGFBR2", "CD274-PDCD1"}
        NEGATIVE_LR = {"ACTB-KRT8", "COL1A1-EPCAM"}
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    PHASE10_DIR = PROJECT_ROOT / f"outputs/phase10/full_transcriptome_commot_style/{slug}"
    RESULT_DIR = PHASE10_DIR / "results"
    GENE_SET_DIR = PHASE10_DIR / "pathway_gene_sets"
    CELLCHATDB_PATH = PHASE10_DIR / "cellchatdb/cellchatdb_human_pathway_genes.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["post_xenium", "visium_hd_crc"], default="post_xenium")
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--min-term-size", type=int, default=5)
    parser.add_argument("--max-term-size", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    configure_dataset(args.dataset)
    out = RESULT_DIR
    out.mkdir(parents=True, exist_ok=True)
    GENE_SET_DIR.mkdir(parents=True, exist_ok=True)

    scan = pd.read_csv(RESULT_DIR / "gene_association_scan_all.csv")
    tested = scan[scan["status"].eq("tested")].copy()
    universe = set(tested["gene"].astype(str).unique())

    gene_sets = load_gene_sets(args.timeout)
    status = build_status(gene_sets, universe)
    enrichment = run_enrichment(tested, gene_sets, universe, args.top_n, args.min_term_size, args.max_term_size)
    top_terms = (
        enrichment.sort_values(["method", "lr_pair", "library", "bh_fdr", "p_value"], ascending=[True, True, True, True, True])
        .groupby(["method", "lr_pair", "library"], as_index=False)
        .head(10)
    )
    summary = summarize(enrichment)
    specificity = summarize_specificity(summary)

    enrichment.to_csv(out / "pathway_enrichment_all.csv", index=False)
    top_terms.to_csv(out / "pathway_enrichment_top_terms.csv", index=False)
    summary.to_csv(out / "pathway_enrichment_summary.csv", index=False)
    specificity.to_csv(out / "pathway_specificity_summary.csv", index=False)
    status.to_csv(out / "pathway_database_status.csv", index=False)

    manifest = {
        "input_scan": str((RESULT_DIR / "gene_association_scan_all.csv").relative_to(PROJECT_ROOT)),
        "dataset_key": DATASET_KEY,
        "dataset_label": DATASET_LABEL,
        "libraries": {
            **LIBRARIES,
            "cellchatdb": str(CELLCHATDB_PATH.relative_to(PROJECT_ROOT)) if CELLCHATDB_PATH.exists() else "",
            "omnipath": "OmniPath focus-LR one-hop interaction gene sets from omnipathdb.org",
        },
        "top_n": args.top_n,
        "min_term_size": args.min_term_size,
        "max_term_size": args.max_term_size,
        "n_universe_genes": len(universe),
        "focus_lr": FOCUS_LR,
        "positive_lr": sorted(POSITIVE_LR),
        "negative_lr": sorted(NEGATIVE_LR),
        "outputs": [
            "pathway_enrichment_all.csv",
            "pathway_enrichment_top_terms.csv",
            "pathway_enrichment_summary.csv",
            "pathway_specificity_summary.csv",
            "pathway_database_status.csv",
        ],
        "claim_boundary": "Pathway enrichment is based on ranked receiver-gene associations and public Enrichr gene-set libraries; it is not perturbation evidence.",
    }
    (out / "pathway_enrichment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(manifest, indent=2))


def load_gene_sets(timeout: int) -> dict[str, dict[str, set[str]]]:
    out: dict[str, dict[str, set[str]]] = {}
    for key, library in LIBRARIES.items():
        path = GENE_SET_DIR / f"{library}.txt"
        if not path.exists():
            cached = find_cached_library(library)
            if cached:
                path.write_bytes(cached.read_bytes())
            else:
                download_library(library, path, timeout)
        out[key] = parse_enrichr_text(path)
    cellchatdb = load_cellchatdb_gene_sets()
    if cellchatdb:
        out["cellchatdb"] = cellchatdb
    omnipath = load_or_download_omnipath_gene_sets(timeout)
    if omnipath:
        out["omnipath"] = omnipath
    return out


def find_cached_library(library: str) -> Path | None:
    for path in (PROJECT_ROOT / "outputs/phase10/full_transcriptome_commot_style").glob(f"*/pathway_gene_sets/{library}.txt"):
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def load_cellchatdb_gene_sets() -> dict[str, set[str]]:
    if not CELLCHATDB_PATH.exists():
        return {}
    table = pd.read_csv(CELLCHATDB_PATH)
    out = {}
    for _, row in table.iterrows():
        genes = parse_semicolon_genes(row.get("genes", ""))
        if genes:
            out[f"CellChatDB:{row['pathway']}"] = genes
    return out


def load_or_download_omnipath_gene_sets(timeout: int) -> dict[str, set[str]]:
    path = GENE_SET_DIR / "omnipath_focus_lr_onehop_gene_sets.csv"
    if path.exists():
        table = pd.read_csv(path)
    else:
        table = build_omnipath_focus_lr_gene_sets(timeout)
        table.to_csv(path, index=False)
    out = {}
    for _, row in table.iterrows():
        genes = parse_semicolon_genes(row.get("genes", ""))
        if genes:
            out[f"OmniPath:{row['lr_pair']}:{row['direction']}"] = genes
    return out


def build_omnipath_focus_lr_gene_sets(timeout: int) -> pd.DataFrame:
    lr_pairs = pd.read_csv(STANDARD_INPUT_DIR / "lr_pairs.csv")
    lr_by_id = lr_pairs.set_index("lr_id")
    rows = []
    for pair, lr_id in FOCUS_LR.items():
        if lr_id not in lr_by_id.index:
            continue
        ligand = str(lr_by_id.loc[lr_id, "ligand"]).upper()
        receptor = str(lr_by_id.loc[lr_id, "receptor"]).upper()
        for direction, seed in [("ligand_onehop", ligand), ("receptor_onehop", receptor), ("ligand_receptor_onehop", f"{ligand},{receptor}")]:
            genes = fetch_omnipath_partners(seed, timeout)
            rows.append(
                {
                    "lr_pair": pair,
                    "lr_id": lr_id,
                    "direction": direction,
                    "seed_genes": seed,
                    "n_genes": len(genes),
                    "genes": ";".join(sorted(genes)),
                }
            )
    return pd.DataFrame(rows)


def fetch_omnipath_partners(seed_genes: str, timeout: int) -> set[str]:
    url = OMNIPATH_URL.format(genes=quote(seed_genes))
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return set()
    genes = set()
    for row in payload:
        for key in ["source_genesymbol", "target_genesymbol"]:
            value = str(row.get(key, "")).upper()
            if value and value != "NAN" and not value.startswith("COMPLEX:"):
                for part in value.replace("_", ";").split(";"):
                    if part:
                        genes.add(part)
    return genes


def parse_semicolon_genes(value) -> set[str]:
    if pd.isna(value):
        return set()
    return {gene.strip().upper() for gene in str(value).replace(",", ";").split(";") if gene.strip()}


def download_library(library: str, path: Path, timeout: int) -> None:
    url = ENRICHR_URL.format(library=library)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        data = response.read()
    path.write_bytes(data)


def parse_enrichr_text(path: Path) -> dict[str, set[str]]:
    gene_sets: dict[str, set[str]] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        fields = [field.strip() for field in line.split("\t") if field.strip()]
        if len(fields) < 3:
            continue
        term = fields[0]
        genes = {gene.upper() for gene in fields[2:] if gene}
        if genes:
            gene_sets[term] = genes
    return gene_sets


def build_status(gene_sets: dict[str, dict[str, set[str]]], universe: set[str]) -> pd.DataFrame:
    rows = []
    upper_universe = {g.upper() for g in universe}
    for library, terms in gene_sets.items():
        covered_genes = set().union(*terms.values()) if terms else set()
        rows.append(
            {
                "library": library,
                "n_terms": len(terms),
                "n_genes_raw": len(covered_genes),
                "n_genes_in_universe": len(covered_genes & upper_universe),
                "status": "loaded" if terms else "empty_or_failed",
            }
        )
    for expected in ["omnipath", "cellchatdb"]:
        if expected not in gene_sets:
            rows.append(
                {
                    "library": expected,
                    "n_terms": 0,
                    "n_genes_raw": 0,
                    "n_genes_in_universe": 0,
                    "status": "not_integrated_or_empty",
                }
            )
    return pd.DataFrame(rows)


def run_enrichment(
    tested: pd.DataFrame,
    gene_sets: dict[str, dict[str, set[str]]],
    universe: set[str],
    top_n: int,
    min_term_size: int,
    max_term_size: int,
) -> pd.DataFrame:
    upper_universe = {g.upper() for g in universe}
    n_universe = len(upper_universe)
    rows = []
    grouped = tested.sort_values(["method", "lr_pair", "spearman_r", "bh_fdr"], ascending=[True, True, False, True]).groupby(
        ["method", "lr_pair", "lr_status"], sort=False
    )
    for (method, lr_pair, lr_status), table in grouped:
        top_genes = {g.upper() for g in table.head(top_n)["gene"].astype(str)}
        for library, terms in gene_sets.items():
            pvals = []
            lib_rows = []
            for term, term_genes_raw in terms.items():
                term_genes = term_genes_raw & upper_universe
                n_term = len(term_genes)
                if n_term < min_term_size or n_term > max_term_size:
                    continue
                overlap_genes = top_genes & term_genes
                overlap = len(overlap_genes)
                if overlap == 0:
                    continue
                p_value = float(hypergeom.sf(overlap - 1, n_universe, n_term, top_n))
                fold = float((overlap / top_n) / (n_term / n_universe))
                table2 = [[overlap, top_n - overlap], [n_term - overlap, n_universe - n_term - top_n + overlap]]
                odds_ratio = float(fisher_exact(table2, alternative="greater")[0])
                pvals.append(p_value)
                lib_rows.append(
                    {
                        "method": method,
                        "lr_pair": lr_pair,
                        "lr_status": lr_status,
                        "library": library,
                        "term": term,
                        "top_n": top_n,
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
                adjusted = bh_adjust(np.asarray(pvals, dtype=float))
                for row, fdr in zip(lib_rows, adjusted):
                    row["bh_fdr"] = float(fdr)
                rows.extend(lib_rows)
    if not rows:
        return pd.DataFrame(
            columns=[
                "method",
                "lr_pair",
                "lr_status",
                "library",
                "term",
                "top_n",
                "n_universe",
                "n_term_genes",
                "overlap",
                "p_value",
                "fold_enrichment",
                "odds_ratio",
                "overlap_genes",
                "bh_fdr",
            ]
        )
    return pd.DataFrame(rows).sort_values(["method", "lr_pair", "library", "bh_fdr", "p_value"])


def summarize(enrichment: pd.DataFrame) -> pd.DataFrame:
    if enrichment.empty:
        return pd.DataFrame()
    rows = []
    for (method, lr_pair, lr_status, library), table in enrichment.groupby(["method", "lr_pair", "lr_status", "library"]):
        sig = table[table["bh_fdr"].le(0.05)]
        best = table.sort_values(["bh_fdr", "p_value"]).iloc[0]
        rows.append(
            {
                "method": method,
                "lr_pair": lr_pair,
                "lr_status": lr_status,
                "library": library,
                "n_enriched_terms": len(table),
                "n_significant_terms_fdr_0_05": int(len(sig)),
                "best_term": best["term"],
                "best_term_fdr": best["bh_fdr"],
                "best_term_p": best["p_value"],
                "best_term_fold": best["fold_enrichment"],
                "best_term_overlap": best["overlap"],
            }
        )
    return pd.DataFrame(rows).sort_values(["method", "lr_pair", "library"])


def summarize_specificity(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for (method, library), table in summary.groupby(["method", "library"]):
        neg_table = table[table["lr_pair"].isin(NEGATIVE_LR)]
        if neg_table.empty:
            continue
        best_neg_sig = int(neg_table["n_significant_terms_fdr_0_05"].max())
        best_neg_sig_pair = str(
            neg_table.sort_values("n_significant_terms_fdr_0_05", ascending=False).iloc[0]["lr_pair"]
        )
        best_neg_fdr = float(neg_table["best_term_fdr"].min())
        best_neg_fdr_pair = str(neg_table.sort_values("best_term_fdr", ascending=True).iloc[0]["lr_pair"])
        for pair in sorted(POSITIVE_LR):
            pos = table[table["lr_pair"].eq(pair)]
            if pos.empty:
                continue
            pos_row = pos.iloc[0]
            rows.append(
                {
                    "method": method,
                    "library": library,
                    "positive_lr": pair,
                    "positive_significant_terms": int(pos_row["n_significant_terms_fdr_0_05"]),
                    "best_negative_significant_terms": best_neg_sig,
                    "best_negative_significant_lr": best_neg_sig_pair,
                    "significant_term_specificity": int(pos_row["n_significant_terms_fdr_0_05"]) - best_neg_sig,
                    "positive_best_term": pos_row["best_term"],
                    "positive_best_term_fdr": float(pos_row["best_term_fdr"]),
                    "best_negative_best_term_fdr": best_neg_fdr,
                    "best_negative_fdr_lr": best_neg_fdr_pair,
                    "best_term_log10_fdr_specificity": -np.log10(float(pos_row["best_term_fdr"]) + 1e-300)
                    - (-np.log10(best_neg_fdr + 1e-300)),
                    "specificity_status": pathway_specificity_status(
                        int(pos_row["n_significant_terms_fdr_0_05"]) - best_neg_sig,
                        float(pos_row["best_term_fdr"]),
                        best_neg_fdr,
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["method", "positive_lr", "library"])


def pathway_specificity_status(term_delta: int, pos_fdr: float, neg_fdr: float) -> str:
    if term_delta > 0 and pos_fdr < neg_fdr:
        return "positive_more_specific_than_negative_controls"
    if term_delta > 0:
        return "positive_has_more_significant_terms_but_best_negative_is_strong"
    if term_delta == 0:
        return "similar_to_negative_controls"
    return "negative_controls_have_more_significant_terms"


def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty(n, dtype=float)
    out[order] = adjusted
    return out


if __name__ == "__main__":
    main()
